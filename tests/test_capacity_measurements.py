"""Measured-coefficient store: atomic writes, freshness, corrupt files and exact identity matching."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypercompute.capacity import measurements
from hypercompute.capacity.measurements import (
    KEEP_PER_IDENTITY,
    MeasurementIdentity,
    MeasurementRecord,
    MeasurementStore,
    from_coefficient,
    to_coefficient,
)
from hypercompute.capacity.types import Coefficient

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
LINK = MeasurementIdentity(node_id="mac", device="rpc0", backend_build="ggml-rpc-8.0.0", path_class="relay",
                           kind="link.rtt_ms")


def rec(value: float = 20.0, *, age_s: float = 0.0, identity: MeasurementIdentity = LINK) -> MeasurementRecord:
    return MeasurementRecord(identity=identity, value=value, unit="ms", measured_at=NOW - timedelta(seconds=age_s),
                             samples=20, note="test", evidence="results/capacity/x/link.json")


def test_default_directory_is_under_the_runtime_home(isolated_home):
    assert MeasurementStore().path == isolated_home / "capacity" / "measurements.json"


def test_put_then_latest_round_trips(tmp_path):
    store = MeasurementStore(tmp_path)
    store.put(rec(21.5))
    got = store.latest(LINK, max_age_s=60, now=NOW)
    assert got == rec(21.5)
    doc = json.loads(store.path.read_text(encoding="utf-8"))
    assert doc["version"] == 1 and doc["records"][0]["identity"]["kind"] == "link.rtt_ms"
    assert list(tmp_path.iterdir()) == [store.path]  # no temp file left behind


def test_put_writes_a_temp_file_in_the_same_directory_then_replaces(tmp_path, monkeypatch):
    store = MeasurementStore(tmp_path)
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((str(src), str(dst)))
        assert os.path.dirname(src) == str(tmp_path) and json.loads(Path(src).read_text(encoding="utf-8"))
        real_replace(src, dst)

    monkeypatch.setattr(measurements.os, "replace", spy)
    store.put(rec())
    assert len(calls) == 1 and calls[0][1] == str(store.path) and calls[0][0].endswith(".tmp")


def test_failed_replace_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    store = MeasurementStore(tmp_path)
    store.put(rec(1.0))
    before = store.path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(measurements.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.put(rec(2.0, age_s=-1))
    assert store.path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [store.path]


def test_latest_never_returns_a_record_older_than_max_age(tmp_path):
    store = MeasurementStore(tmp_path)
    store.put(rec(10.0, age_s=600))
    assert store.latest(LINK, max_age_s=599, now=NOW) is None
    assert store.latest(LINK, max_age_s=600, now=NOW).value == 10.0
    store.put(rec(30.0, age_s=60))
    store.put(rec(20.0, age_s=300))
    assert store.latest(LINK, max_age_s=3600, now=NOW).value == 30.0  # freshest, not last written
    assert store.latest(LINK, max_age_s=59, now=NOW) is None


def test_latest_skips_records_from_the_future(tmp_path):
    store = MeasurementStore(tmp_path)
    store.put(rec(5.0, age_s=-30))
    assert store.latest(LINK, max_age_s=3600, now=NOW) is None


@pytest.mark.parametrize("max_age", [-1.0, math.inf, math.nan])
def test_latest_rejects_bad_max_age(tmp_path, max_age):
    with pytest.raises(ValueError):
        MeasurementStore(tmp_path).latest(LINK, max_age_s=max_age, now=NOW)


def test_latest_requires_an_aware_now(tmp_path):
    with pytest.raises(ValueError):
        MeasurementStore(tmp_path).latest(LINK, max_age_s=1, now=datetime(2026, 1, 1))


@pytest.mark.parametrize("change", [{"ctx": 4096}, {"path_class": "lan"}, {"path_class": None},
                                    {"model_fingerprint": "abc"}, {"power_state": "battery"},
                                    {"backend_build": "ggml-rpc-8.1.0"}, {"device": "rpc1"}, {"node_id": "pc"},
                                    {"kind": "link.goodput_bytes_per_s"}])
def test_identity_must_match_exactly(tmp_path, change):
    store = MeasurementStore(tmp_path)
    store.put(rec())
    assert store.latest(replace(LINK, **change), max_age_s=60, now=NOW) is None
    assert store.latest(LINK, max_age_s=60, now=NOW) is not None


def test_records_for_other_identities_are_kept(tmp_path):
    store = MeasurementStore(tmp_path)
    other = replace(LINK, kind="link.goodput_bytes_per_s")
    store.put(rec(1.0))
    store.put(MeasurementRecord(identity=other, value=1e6, unit="bytes/s", measured_at=NOW, samples=1))
    assert {r.identity.kind for r in store.records()} == {"link.rtt_ms", "link.goodput_bytes_per_s"}


def test_history_per_identity_is_bounded(tmp_path):
    store = MeasurementStore(tmp_path)
    for i in range(KEEP_PER_IDENTITY + 3):
        store.put(rec(float(i), age_s=1000 - i))
    kept = store.records()
    assert len(kept) == KEEP_PER_IDENTITY
    assert min(r.value for r in kept) == 3.0


def _one_record_file(**overrides: object) -> str:
    raw = {"identity": {"node_id": "mac", "device": "rpc0", "backend_build": "ggml-rpc-8.0.0",
                        "model_fingerprint": None, "ctx": None, "power_state": None, "path_class": "relay",
                        "kind": "link.rtt_ms"},
           "value": 1.0, "unit": "ms", "measured_at": NOW.isoformat(), "samples": 1, "note": "", "evidence": None}
    text = json.dumps({"version": 1, "records": [{**raw, **{k: "@" + k for k in overrides}}]})
    for key, literal in overrides.items():  # splice raw JSON literals json.dumps would not produce
        text = text.replace(f'"@{key}"', str(literal))
    return text


@pytest.mark.parametrize("content", ["{not json", '{"version": 1, "records": [{"identity": {}}]}', "[]",
                                     '{"version": 2, "records": []}', b"\xff\xfe\x00garbage",
                                     "[" * 100_000,  # json raises RecursionError, not ValueError
                                     _one_record_file(value="1" + "0" * 400),  # int too large for a float
                                     _one_record_file(measured_at='"0001-01-01T00:00:00+14:00"'),  # age overflows
                                     _one_record_file(unit='"s"')],  # costs.merge would raise on it in every plan
                         ids=["not-json", "bad-identity", "list", "version-2", "not-utf8", "deep-nesting",
                              "huge-int", "date-overflow", "known-kind-wrong-unit"])
def test_corrupt_file_is_moved_aside_and_the_store_starts_empty(tmp_path, caplog, content):
    store = MeasurementStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    if isinstance(content, bytes):
        store.path.write_bytes(content)
    else:
        store.path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hypercompute.capacity.measurements"):
        assert store.latest(LINK, max_age_s=60, now=NOW) is None
    assert "corrupt" in caplog.text
    (moved,) = tmp_path.glob("measurements.json.*.corrupt")
    assert moved.read_bytes() == (content if isinstance(content, bytes) else content.encode())
    assert not store.path.exists()
    store.put(rec(7.0))
    assert store.latest(LINK, max_age_s=60, now=NOW).value == 7.0


def test_valid_record_file_helper_parses(tmp_path):
    store = MeasurementStore(tmp_path)
    store.path.write_text(_one_record_file(), encoding="utf-8")
    assert store.latest(LINK, max_age_s=60, now=NOW).value == 1.0


def test_quarantine_leaves_a_file_rewritten_after_the_corrupt_read(tmp_path):
    # a reader saw corrupt bytes, then a writer replaced the file: the reader must not move the good file aside
    store = MeasurementStore(tmp_path)
    store.put(rec(4.0))
    store._quarantine(b"garbage seen earlier", ValueError("stale"))
    assert list(tmp_path.glob("*.corrupt")) == []
    assert store.latest(LINK, max_age_s=60, now=NOW).value == 4.0


def test_corrupt_file_found_by_put_is_kept(tmp_path):
    store = MeasurementStore(tmp_path)
    store.path.write_text("garbage", encoding="utf-8")
    store.put(rec())
    assert len(list(tmp_path.glob("*.corrupt"))) == 1
    assert len(store.records()) == 1


def test_record_validation():
    with pytest.raises(ValueError):
        replace(rec(), value=math.nan)
    with pytest.raises(ValueError):
        replace(rec(), value=10**400)  # math.isfinite would raise OverflowError
    with pytest.raises(ValueError):
        replace(rec(), measured_at=datetime.fromisoformat("0001-01-01T00:00:00+14:00"))
    with pytest.raises(ValueError):
        replace(rec(), measured_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        replace(rec(), samples=-1)
    with pytest.raises(ValueError):
        replace(rec(), unit="")
    with pytest.raises(ValueError):
        replace(LINK, node_id="")
    with pytest.raises(ValueError):
        replace(LINK, ctx=0)
    with pytest.raises(TypeError):
        MeasurementStore().put({"value": 1})


def test_to_coefficient_is_measured():
    c = to_coefficient(rec(12.5))
    assert c == Coefficient(value=12.5, unit="ms", provenance="measured", measured_at=NOW, samples=20, note="test")


def test_from_coefficient_refuses_priors():
    measured = Coefficient(value=3.0, unit="ms", provenance="measured", measured_at=NOW, samples=4, note="n")
    r = from_coefficient(LINK, measured, evidence="e.json")
    assert (r.value, r.samples, r.evidence, r.measured_at) == (3.0, 4, "e.json", NOW)
    with pytest.raises(ValueError):
        from_coefficient(LINK, replace(measured, provenance="prior"))
    with pytest.raises(ValueError):
        from_coefficient(LINK, replace(measured, measured_at=None))


def test_kind_units_are_the_units_of_the_priors_the_bridge_replaces():
    from hypercompute.capacity import bridge, costs

    device = costs.device_prior("cpu", "local", "CPU", "ram")
    link = costs.link_prior("relay")
    storage = costs.storage_prior("local")
    expected = {bridge.KIND_DECODE: device.decode_ms_per_gib.unit,
                bridge.KIND_PREFILL: device.prefill_ms_per_gib_token.unit,
                bridge.KIND_RTT: link.rtt_ms.unit, bridge.KIND_GOODPUT: link.goodput_bytes_per_s.unit,
                bridge.KIND_ROUND_TRIPS: link.round_trips_per_token.unit,
                bridge.KIND_STORAGE_READ: storage.read_bytes_per_s.unit,
                bridge.KIND_STREAM_EFFICIENCY: storage.stream_efficiency.unit}
    assert measurements.KIND_UNITS == expected


def test_put_refuses_a_known_kind_in_another_unit(tmp_path):
    store = MeasurementStore(tmp_path)
    with pytest.raises(ValueError, match="link.rtt_ms is measured in 'ms'"):
        store.put(replace(rec(), unit="s"))  # costs.merge would raise on it in every plan that reads it
    with pytest.raises(ValueError, match="compute.decode_ms_per_gib is measured in 'ms/GiB'"):
        replace(rec(), identity=replace(LINK, kind="compute.decode_ms_per_gib"))  # the record itself is refused
    assert store.records() == ()
    custom = replace(LINK, kind="experiment.anything")
    store.put(MeasurementRecord(identity=custom, value=1.0, unit="widgets", measured_at=NOW, samples=1))
    assert [r.unit for r in store.records()] == ["widgets"]
