"""Deployment-side wiring of the hcstream extension: capability, overlay launch, stats and verification.

The planner asks `stream_capability` whether this host can run a GGUF in the bounded stream tier and with which exact
unit buffers; the llama adapter asks `prepare_stream` for the overlay binary, the generated config and the environment;
`verify_stats` turns the extension's own stats file into a pass/fail participation check after the model loads.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from .. import config
from ..gguf import GgufError, read_tensor_table
from ..schemas import StreamSpec
from ..services.planner import StreamCapability
from ..units import fmt_bytes
from . import overlay
from .stream_config import StreamConfigError, StreamOptions, block_count, layer_layouts, render_config, stream_plan

GIB = 1 << 30
SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$")
CALIBRATION_FILE = "storage-calibration.json"
CALIBRATION_MAX_AGE_S = 7 * 24 * 3600
PRIOR_READ_BYTES_PER_S = 3.0 * GIB  # labelled prior until `hypercompute capacity calibrate` measures the volume


class StreamRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.code, self.fix = code, fix


@dataclass(frozen=True)
class StreamLaunch:
    bin_dir: Path
    config_path: Path
    stats_path: Path
    env: dict[str, str]
    proxy_sha256: str


def model_shards(first: Path) -> list[Path]:
    """All parts of a split GGUF given its first part (llama.cpp's -00001-of-0000N naming), else just the file."""
    m = SPLIT_RE.match(first.name)
    if not m:
        return [first]
    n = int(m["n"])
    return [first.with_name(f"{m['stem']}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


@lru_cache(maxsize=16)
def _unit_buffers(key: tuple[tuple[str, int, int], ...]) -> tuple[int, ...] | None:
    paths = [Path(p) for p, _, _ in key]
    table = read_tensor_table(paths)
    if any(t.nbytes is None for t in table if t.layer is not None or t.name.startswith("output")):
        return None
    layouts = layer_layouts(table)
    n = block_count(table)
    if any(i not in layouts for i in range(n + 1)):
        return None
    return tuple(layouts[i].buffer_bytes for i in range(n + 1))


def unit_buffers(paths: list[Path]) -> tuple[int, ...] | None:
    """Aligned per-unit buffer sizes (blocks, then the head); None when a tensor type's size is unknown."""
    try:
        key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
        return _unit_buffers(key)
    except (OSError, GgufError):
        return None


def _calibration_path() -> Path:
    return config.paths().coordinator / CALIBRATION_FILE


def record_storage_rate(volume: str, read_bytes_per_s: float, *, readers: int, chunk_bytes: int,
                        sample_bytes: int, source: str) -> None:
    """Persist a measured unbuffered read rate for a volume (drive letter or mount)."""
    path = _calibration_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    data[volume.upper()] = {"read_bytes_per_s": read_bytes_per_s, "readers": readers, "chunk_bytes": chunk_bytes,
                            "sample_bytes": sample_bytes, "source": source, "measured_at": time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def storage_rate(model_path: Path) -> tuple[float, str]:
    """(bytes per second, provenance) for unbuffered reads on the volume holding the model."""
    volume = (model_path.resolve().drive or model_path.resolve().anchor).upper()
    try:
        rec = json.loads(_calibration_path().read_text(encoding="utf-8")).get(volume) or {}
    except (OSError, json.JSONDecodeError):
        rec = {}
    fresh = rec and time.time() - float(rec.get("measured_at", 0)) < CALIBRATION_MAX_AGE_S
    if fresh and float(rec.get("read_bytes_per_s", 0)) > 0:
        return float(rec["read_bytes_per_s"]), "measured"
    return PRIOR_READ_BYTES_PER_S, "prior"


def stream_capability(paths: list[Path], *, slots: int = 2) -> StreamCapability | None:
    """The bounded stream tier for these GGUF files on this host, or None when it cannot be used here."""
    if not paths or not overlay.supported() or overlay.find_proxy_dll() is None:
        return None
    if not all(p.is_file() for p in paths):
        return None
    buffers = unit_buffers(paths)
    if buffers is None:
        return None
    rate, provenance = storage_rate(paths[0])
    return StreamCapability(unit_buffer_bytes=buffers, slots=slots, read_bytes_per_s=rate,
                            read_provenance=provenance)  # type: ignore[arg-type]


def config_paths(work_dir: Path, deployment_id: str) -> tuple[Path, Path]:
    return work_dir / f"hcstream-{deployment_id}.cfg", work_dir / f"hcstream-{deployment_id}.json"


def prepare_stream(engine_dir: Path, model_path: Path, spec: StreamSpec, work_dir: Path, deployment_id: str,
                   options: StreamOptions | None = None) -> StreamLaunch:
    """Overlay engine, config file and environment for one stream-tier deployment.

    The layout is recomputed from the GGUF and must match what the plan admitted; a different slot or resident size
    means the files changed after planning, and the deployment refuses instead of committing unadmitted memory.
    """
    dll = overlay.find_proxy_dll()
    if dll is None:
        raise StreamRuntimeError("hcstream_missing", "the hcstream proxy DLL is not built",
                                 fix="uv run --no-project --with ziglang python scripts/build_hcstream.py")
    try:
        ov = overlay.ensure_overlay(engine_dir, dll)
    except overlay.OverlayError as exc:
        raise StreamRuntimeError("hcstream_overlay", str(exc)) from exc
    shards = model_shards(model_path)
    try:
        table = read_tensor_table(shards)
        layouts = layer_layouts(table)
        planned = stream_plan(layouts, spec.resident_units, spec.streamed_units, max(spec.slots, 1))
    except (OSError, GgufError, StreamConfigError) as exc:
        raise StreamRuntimeError("hcstream_config", f"cannot lay out the GGUF for streaming: {exc}") from exc
    if planned.resident_bytes != spec.resident_bytes or (spec.streamed_units and planned.slot_bytes != spec.slot_bytes):
        raise StreamRuntimeError("hcstream_changed", "the GGUF layout differs from the admitted plan (resident "
                                 f"{fmt_bytes(planned.resident_bytes)} vs {fmt_bytes(spec.resident_bytes)}, slot "
                                 f"{fmt_bytes(planned.slot_bytes)} vs {fmt_bytes(spec.slot_bytes)})",
                                 fix="plan the deployment again")
    cfg_path, stats_path = config_paths(work_dir, deployment_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats_path.unlink(missing_ok=True)
    try:
        text = render_config(shards, table, planned, stats_path, options or StreamOptions())
    except StreamConfigError as exc:
        raise StreamRuntimeError("hcstream_config", str(exc)) from exc
    cfg_path.write_text(text, encoding="utf-8")
    return StreamLaunch(bin_dir=ov.bin_dir, config_path=cfg_path, stats_path=stats_path,
                        env={"HC_STREAM_CONFIG": str(cfg_path)}, proxy_sha256=ov.proxy_sha256)


def read_stats(path: Path) -> dict | None:
    for _ in range(3):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)  # the extension replaces the file atomically; a reader can still race the rename
    return None


def verify_stats(stats: dict | None, spec: StreamSpec) -> tuple[bool, str]:
    """Did the extension attach with the admitted layout and read without errors?"""
    if stats is None:
        return False, "hcstream wrote no stats: the extension did not attach to the context"
    problems = []
    state = stats.get("state")
    if state == "passthrough":
        problems.append("state passthrough: the extension loaded but manages no layer, so nothing is bounded")
    elif state not in ("attached", "running"):
        problems.append(f"state {state}")
    for key in ("verify_failures", "io_errors", "mmap_fallbacks"):
        if int(stats.get(key, 0)):
            problems.append(f"{key} {stats[key]}")
    if spec.streamed_units and int(stats.get("slot_bytes", 0)) != spec.slot_bytes:
        problems.append(f"slot bytes {stats.get('slot_bytes')} != planned {spec.slot_bytes}")
    committed = stats.get("slot_bytes_committed")  # absent from builds before the field existed
    if spec.streamed_units and committed is not None and int(committed) != spec.slots * spec.slot_bytes:
        problems.append(f"slot buffers committed {committed} != planned {spec.slots} x {spec.slot_bytes}")
    resident = int(stats.get("resident_bytes", 0))
    if resident and resident != spec.resident_bytes:
        problems.append(f"resident bytes {resident} != planned {spec.resident_bytes}")
    detail = (f"hcstream: {len(spec.resident_units)} unit(s) resident ({fmt_bytes(resident or 0)} loaded of "
              f"{fmt_bytes(spec.resident_bytes)} planned), {len(spec.streamed_units)} streamed through "
              f"{stats.get('slots')} slot(s) of {fmt_bytes(int(stats.get('slot_bytes', 0)))}, "
              f"{fmt_bytes(int(stats.get('bytes_read', 0)))} read unbuffered so far, prefetch "
              f"{'off' if stats.get('no_prefetch') else 'on'}")
    return not problems, detail + ("; problems: " + ", ".join(problems) if problems else "")


def stats_summary(stats: dict | None, now: datetime | None = None) -> dict:
    """Stats plus derived per-pass figures for the API and CLI (all values straight from the extension)."""
    if stats is None:
        return {"available": False}
    passes = int(stats.get("passes", 0))
    read = int(stats.get("bytes_read", 0))
    now = now or datetime.now(UTC)
    age = max(0.0, now.timestamp() - int(stats.get("updated_unix_ms", 0)) / 1000)
    return {**stats, "available": True, "age_s": round(age, 1),
            "bytes_per_pass": read // passes if passes else None,
            # read_ns sums every reader thread's time, so this is the rate of one reader, not the drive's total
            "per_reader_gib_per_s": round(read / (int(stats.get("read_ns", 0)) / 1e9) / GIB, 2)
            if int(stats.get("read_ns", 0)) else None,
            "stall_s": round(int(stats.get("stall_ns", 0)) / 1e9, 3)}
