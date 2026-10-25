"""Store of measured coefficients, one JSON file keyed by what each number was measured on.

A measurement only replaces a prior for the same identity: node, device, backend build, model fingerprint, context
size, power state, path class and kind (what was measured, for example `link.rtt_ms`). Fields that do not apply are
None and still take part in matching, so a link measurement never stands in for a model-specific one.

Writes go to a temp file in the same directory followed by os.replace, so a reader sees the old file or the new one,
never a partial write. Concurrent writers in one process are serialized; across processes the last writer wins. A file
that does not parse is moved aside with a `.corrupt` suffix and the store starts empty, with a warning in the log.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import config
from .types import Coefficient

log = logging.getLogger("hypercompute.capacity.measurements")

STORE_FILE = "measurements.json"
FORMAT_VERSION = 1
KEEP_PER_IDENTITY = 16  # newest records kept per identity; older ones are dropped on put
_REPLACE_ATTEMPTS = 5  # os.replace fails on Windows while another process holds the target open
_REPLACE_BACKOFF_S = 0.05
_WRITE_LOCK = threading.RLock()  # re-entered by put -> _load -> _quarantine

# The unit each kind capacity/bridge.py reads must carry: costs.merge raises when a measurement's unit differs from the
# prior it replaces, so a record in another unit would stop every plan that reads it. MeasurementRecord refuses such a
# record, so put() cannot write one and a file holding one does not parse (it is moved aside like any corrupt file);
# kinds not listed here are stored as given.
KIND_UNITS: dict[str, str] = {
    "compute.decode_ms_per_gib": "ms/GiB",
    "compute.prefill_ms_per_gib_token": "ms/GiB/token",
    "link.rtt_ms": "ms",
    "link.goodput_bytes_per_s": "bytes/s",
    "link.round_trips_per_token": "round trips",
    "storage.read_bytes_per_s": "bytes/s",
    "storage.stream_efficiency": "fraction",
}


def default_directory() -> Path:
    """Runtime home (HYPERCOMPUTE_HOME, default ~/.hypercompute) / capacity."""
    return config.paths().home / "capacity"


@dataclass(frozen=True, kw_only=True)
class MeasurementIdentity:
    node_id: str
    device: str
    backend_build: str
    model_fingerprint: str | None = None
    ctx: int | None = None
    power_state: str | None = None
    path_class: str | None = None
    kind: str

    def __post_init__(self) -> None:
        for name in ("node_id", "device", "backend_build", "kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"identity field {name} must be a non-empty string")
        for name in ("model_fingerprint", "power_state", "path_class"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"identity field {name} must be None or a non-empty string")
        if self.ctx is not None and (isinstance(self.ctx, bool) or not isinstance(self.ctx, int) or self.ctx <= 0):
            raise ValueError("identity field ctx must be None or a positive int")

    def as_tuple(self) -> tuple[str, str, str, str | None, int | None, str | None, str | None, str]:
        return (self.node_id, self.device, self.backend_build, self.model_fingerprint, self.ctx, self.power_state,
                self.path_class, self.kind)


@dataclass(frozen=True, kw_only=True)
class MeasurementRecord:
    identity: MeasurementIdentity
    value: float
    unit: str
    measured_at: datetime  # timezone-aware
    samples: int
    note: str = ""
    evidence: str | None = None  # evidence file the number came from

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int | float) or not _finite(self.value):
            raise ValueError("measurement value must be a finite number")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("measurement unit must be a non-empty string")
        expected = KIND_UNITS.get(self.identity.kind) if isinstance(self.identity, MeasurementIdentity) else None
        if expected is not None and self.unit != expected:
            raise ValueError(f"{self.identity.kind} is measured in {expected!r}, not {self.unit!r}")
        if not isinstance(self.measured_at, datetime) or self.measured_at.tzinfo is None:
            raise ValueError("measured_at must be a timezone-aware datetime")
        try:
            self.measured_at.astimezone(UTC)  # age arithmetic in latest() must not overflow
        except OverflowError as exc:
            raise ValueError(f"measured_at {self.measured_at!r} is out of range") from exc
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 0:
            raise ValueError("samples must be a non-negative int")


def _finite(value: float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def to_coefficient(record: MeasurementRecord) -> Coefficient:
    return Coefficient(value=float(record.value), unit=record.unit, provenance="measured",
                       measured_at=record.measured_at, samples=record.samples, note=record.note)


def from_coefficient(identity: MeasurementIdentity, coefficient: Coefficient, *,
                     evidence: str | None = None) -> MeasurementRecord:
    """A record for a measured coefficient; priors and estimates are refused, they do not belong in this store."""
    if coefficient.provenance != "measured" or coefficient.measured_at is None:
        raise ValueError(f"only measured coefficients with a timestamp can be stored, got {coefficient.provenance}")
    return MeasurementRecord(identity=identity, value=coefficient.value, unit=coefficient.unit,
                             measured_at=coefficient.measured_at, samples=coefficient.samples, note=coefficient.note,
                             evidence=evidence)


# --------------------------------------------------------------------------------------------------- json


def _record_to_json(record: MeasurementRecord) -> dict[str, Any]:
    i = record.identity
    return {"identity": {"node_id": i.node_id, "device": i.device, "backend_build": i.backend_build,
                         "model_fingerprint": i.model_fingerprint, "ctx": i.ctx, "power_state": i.power_state,
                         "path_class": i.path_class, "kind": i.kind},
            "value": record.value, "unit": record.unit, "measured_at": record.measured_at.isoformat(),
            "samples": record.samples, "note": record.note, "evidence": record.evidence}


def _record_from_json(raw: Any) -> MeasurementRecord:
    if not isinstance(raw, dict) or not isinstance(raw.get("identity"), dict):
        raise ValueError("record is not an object with an identity")
    measured_at = raw["measured_at"]
    if not isinstance(measured_at, str):
        raise ValueError("measured_at is not a string")
    return MeasurementRecord(identity=MeasurementIdentity(**raw["identity"]), value=raw["value"], unit=raw["unit"],
                             measured_at=datetime.fromisoformat(measured_at), samples=raw["samples"],
                             note=raw.get("note", ""), evidence=raw.get("evidence"))


def _parse(text: str) -> tuple[MeasurementRecord, ...]:
    doc = json.loads(text)
    if not isinstance(doc, dict) or doc.get("version") != FORMAT_VERSION or not isinstance(doc.get("records"), list):
        raise ValueError(f"not a version {FORMAT_VERSION} measurement store")
    return tuple(_record_from_json(r) for r in doc["records"])


def _trim(records: tuple[MeasurementRecord, ...], identity: MeasurementIdentity) -> tuple[MeasurementRecord, ...]:
    same = sorted((r for r in records if r.identity == identity), key=lambda r: r.measured_at)
    others = tuple(r for r in records if r.identity != identity)
    return others + tuple(same[-KEEP_PER_IDENTITY:])


def _replace(src: str, dst: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)


# --------------------------------------------------------------------------------------------------- store


@dataclass(frozen=True)
class MeasurementStore:
    directory: Path = field(default_factory=default_directory)

    @property
    def path(self) -> Path:
        return self.directory / STORE_FILE

    def records(self) -> tuple[MeasurementRecord, ...]:
        return self._load()

    def put(self, record: MeasurementRecord) -> None:
        """Add `record` and rewrite the file atomically (KIND_UNITS was checked when the record was built)."""
        if not isinstance(record, MeasurementRecord):
            raise TypeError(f"expected a MeasurementRecord, got {type(record).__name__}")
        with _WRITE_LOCK:
            self._write(_trim((*self._load(), record), record.identity))

    def latest(self, identity: MeasurementIdentity, max_age_s: float, *,
               now: datetime | None = None) -> MeasurementRecord | None:
        """The freshest record for exactly this identity measured at most `max_age_s` before `now`, else None.

        Records stamped after `now` are not returned: their age cannot be shown.
        """
        if not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("max_age_s must be a finite non-negative number")
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        fresh = [r for r in self._load()
                 if r.identity == identity and 0 <= (now - r.measured_at).total_seconds() <= max_age_s]
        return max(fresh, key=lambda r: r.measured_at, default=None)

    def _load(self) -> tuple[MeasurementRecord, ...]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return ()
        try:
            return _parse(raw.decode("utf-8"))
        # JSONDecodeError and UnicodeDecodeError are ValueErrors; deeply nested JSON raises RecursionError
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            self._quarantine(raw, exc)
            return ()

    def _quarantine(self, raw: bytes, exc: BaseException) -> None:
        """Move the corrupt file aside, unless a writer replaced it after `raw` was read (then it is not ours)."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.path.with_name(f"{STORE_FILE}.{stamp}.corrupt")
        with _WRITE_LOCK:
            try:
                if self.path.read_bytes() != raw:
                    log.warning("measurement store %s was corrupt (%s) but has since been rewritten; leaving it",
                                self.path, exc)
                    return
                os.replace(self.path, target)
            except FileNotFoundError:
                return  # another writer replaced or moved it first
            except OSError as move_exc:
                raise OSError(f"measurement store {self.path} is corrupt ({exc}) and could not be moved aside: "
                              f"{move_exc}") from move_exc
        log.warning("measurement store %s is corrupt (%s); moved to %s and starting empty", self.path, exc, target)

    def _write(self, records: tuple[MeasurementRecord, ...]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"version": FORMAT_VERSION, "records": [_record_to_json(r) for r in records]}, indent=1)
        fd, tmp = tempfile.mkstemp(prefix=f"{STORE_FILE}.", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            _replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
