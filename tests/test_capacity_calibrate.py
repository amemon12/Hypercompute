"""scripts/capacity/calibrate.py: coefficient math on synthetic bench documents (built with benchlib's own result types
and bench.py's presets), refusal rules, the identity the bridge looks up, store round trips and the bounded storage
probe. No engine, GPU or disk-speed measurement is involved."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypercompute.capacity import benchlib as bl
from hypercompute.capacity import bridge, storage_probe
from hypercompute.capacity.measurements import MeasurementIdentity, MeasurementRecord, MeasurementStore
from hypercompute.capacity.types import ModelManifest, UnitSpec
from hypercompute.schemas import ModelArtifact, ModelFile
from test_capacity_manifest import base_meta, model_tensors, write_gguf

ROOT = Path(__file__).resolve().parents[1]
MIB, GIB = 1 << 20, 1 << 30
NOW = datetime(2026, 10, 22, 6, 0, tzinfo=UTC)
RAN_AT = NOW - timedelta(minutes=10)
N_LAYER = 4
BLOCK, HEAD, EMBD = 100 * MIB, 50 * MIB, 30 * MIB
UNITS_GIB = (N_LAYER * BLOCK + HEAD) / GIB  # 450 MiB: token_embd is not computed per token
MODEL = "q-test"
FP = "f" * 64


def _load(name: str, file: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "capacity" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


cal = _load("capacity_calibrate_script", "calibrate.py")
bench = _load("capacity_bench_script", "bench.py")


def make_manifest(fingerprint: str = FP) -> ModelManifest:
    units = [UnitSpec(i, "block", BLOCK, BLOCK) for i in range(N_LAYER)] + [UnitSpec(N_LAYER, "head", HEAD, HEAD)]
    return ModelManifest(
        model_id=MODEL, revision="r1", shards=(), architecture="qwen2", file_type="Q4_K_M", n_layer=N_LAYER,
        n_embd=1024, n_head=16, n_head_kv=4, key_length=64, value_length=64, context_length=4096, rope=(),
        tokenizer_model="gpt2", vocab_size=1000, chat_template_sha256=None, units=tuple(units),
        token_embd_bytes=EMBD, other_bytes=0, tied_embeddings=False, streamable=True, fingerprint=fingerprint)


MANIFEST = make_manifest()


def case(preset: str, tmp_path: Path, label: str | None = None) -> bl.Case:
    return bench.case_for_preset(bench.parse_preset(preset), [str(tmp_path / "m.gguf")], N_LAYER,
                                 label=label or preset, ctx=2048, threads=12, windows=True)


def rep(index: int, decode_ms: float, *, warmup: bool = False, ok: bool = True, prefill_ms: float = 0.5,
        prompt_n: int = 512, power: str = "ac", hc_per_pass: float | None = None) -> bl.RepResult:
    """A rep whose uncached phase decodes 64 tokens at `decode_ms` per step and prefills at `prefill_ms` per token."""
    t = bl.Timings(prompt_n=prompt_n, prompt_ms=prefill_ms * prompt_n, predicted_n=64, predicted_ms=decode_ms * 63,
                   cache_n=0)
    io: dict = {"seconds": 1.0}
    if hc_per_pass is not None:
        io["hcstream"] = {"bytes_read": int(hc_per_pass * 65), "passes": 65}
    phases = (bl.PhaseResult(phase="uncached", ok=ok, error=None if ok else "boom", timings=t, io=io),
              bl.PhaseResult(phase="prefix_prime", ok=True), bl.PhaseResult(phase="prefix_reuse", ok=True))
    return bl.RepResult(index=index, warmup=warmup, ok=ok, phases=phases, gpu=None, power={"source": power}, rpc=None,
                        started_at=RAN_AT.isoformat())


def reps(decode_ms: list[float], **kw) -> list[bl.RepResult]:
    return [rep(0, decode_ms[0] * 3, warmup=True, **kw)] + [rep(i + 1, d, **kw) for i, d in enumerate(decode_ms)]


def write_doc(tmp_path: Path, c: bl.Case, rs: list[bl.RepResult], *, status: str = "ok", power: str = "ac",
              log: tuple[str, ...] = (), launch: dict | None = None, schema: str | None = None) -> Path:
    load = bl.LoadResult(ok=True, seconds=1.0, error=None, log_lines=log)
    doc = bl.results_document(case=c, workload=bl.BenchWorkload(output_tokens=64), status=status,
                              started_at=RAN_AT.isoformat(), host={"power": {"source": power}},
                              model={"model_id": MODEL, "shards": [], "n_layer": N_LAYER}, preflight=None,
                              engine={"release": "b11770", "build": 11770}, launch=launch, load=load, reps=rs)
    doc["finished_at"] = RAN_AT.isoformat()
    if schema is not None:
        doc["schema"] = schema
    path = tmp_path / "runs" / c.label / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, default=str), encoding="utf-8")
    return path


def run(paths: list[Path], existing: tuple = (), manifest: ModelManifest = MANIFEST, **kw) -> tuple:
    return cal.calibrate_runs(paths, existing, lambda doc: manifest, now=NOW, **kw)


def by_kind(outcome) -> dict[str, MeasurementRecord]:
    return {r.identity.kind: r for r in outcome.records}


GPU_LOG = ("load_tensors: offloading output layer to GPU", "load_tensors: offloaded 5/5 layers to GPU")


# --------------------------------------------------------------------------------------------------- compute math


def test_cpu_decode_and_prefill_coefficients(tmp_path):
    path = write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0, 10.0, 8.0], prefill_ms=0.6))
    (out,) = run([path])
    assert out.refused is None and not out.skipped
    rec = by_kind(out)
    decode, prefill = rec[bridge.KIND_DECODE], rec[bridge.KIND_PREFILL]
    assert decode.value == pytest.approx(9.0 / UNITS_GIB)  # median ms per step over the unit weight GiB
    assert decode.unit == "ms/GiB" and prefill.unit == "ms/GiB/token"
    assert prefill.value == pytest.approx(0.6 / UNITS_GIB)
    assert decode.samples == 3 and decode.measured_at == RAN_AT and decode.evidence == str(path.resolve())
    i = decode.identity
    assert (i.node_id, i.device, i.backend_build, i.ctx, i.power_state, i.path_class, i.model_fingerprint) == \
        ("local", "CPU", "b11770", 2048, "ac", None, FP)
    assert "token_embd lookup excluded" in decode.note and "512 prompt tokens" in prefill.note


def test_warmup_reps_do_not_count(tmp_path):
    path = write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0, 9.0, 9.0]))  # warmup ran at 27 ms
    (out,) = run([path])
    assert by_kind(out)[bridge.KIND_DECODE].value == pytest.approx(9.0 / UNITS_GIB)


def test_gpu_coefficient_needs_every_unit_offloaded(tmp_path):
    c = case("stock-gpu", tmp_path)
    (ok,) = run([write_doc(tmp_path, c, reps([1.0, 1.1, 0.9]), log=GPU_LOG)])
    rec = by_kind(ok)[bridge.KIND_DECODE]
    assert rec.identity.device == "CUDA0" and rec.value == pytest.approx(1.0 / UNITS_GIB)
    partial = replace(c, label="partial")
    (short,) = run([write_doc(tmp_path, partial, reps([1.0] * 3), log=("load_tensors: offloaded 3/5 layers to GPU",))])
    assert not short.records and "offloaded 3/5" in short.skipped[0]
    (silent,) = run([write_doc(tmp_path, replace(c, label="silent"), reps([1.0] * 3))])
    assert not silent.records and "no 'offloaded" in silent.skipped[0]


@pytest.mark.parametrize("flags", [("-ot", r"blk\.[0-3]\.ffn_.*=CPU"), ("--override-tensor=blk\\..*=CPU",),
                                   ("--cpu-moe",), ("-nkvo",), ("--n-cpu-moe", "2")])
def test_flags_that_move_tensors_off_the_gpu_give_no_compute_coefficient(tmp_path, flags):
    c = case("stock-gpu", tmp_path)
    moved = replace(c, engine=replace(c.engine, flags=(*c.engine.flags, *flags)))
    (out,) = run([write_doc(tmp_path, moved, reps([1.0] * 3), log=GPU_LOG)])  # the log still says 5/5 offloaded
    assert not out.records and "move tensors or ops" in out.skipped[0]


def test_tied_embeddings_note_and_repack_state(tmp_path):
    tied = replace(MANIFEST, tied_embeddings=True, token_embd_bytes=0)
    (out,) = run([write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3))], manifest=tied)
    note = by_kind(out)[bridge.KIND_DECODE].note
    assert "tied head, counted once" in note and "lookup excluded" not in note and "CPU repack on" in note


@pytest.mark.parametrize("preset", ["gpu-ram-2", "stream-r1-s2-k2"])
def test_split_and_stream_cases_give_no_compute_coefficient(tmp_path, preset):
    path = write_doc(tmp_path, case(preset, tmp_path), reps([5.0] * 3, hc_per_pass=200 * MIB), log=GPU_LOG)
    (out,) = run([path])
    assert not any(r.identity.kind.startswith("compute.") for r in out.records)
    if preset == "gpu-ram-2":
        assert "single-device" in out.skipped[0] and "split" in out.skipped[0]


def test_model_agnostic_copies_are_optional(tmp_path):
    path = write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3))
    (plain,) = run([path])
    (both,) = run([path], agnostic=True)
    assert {r.identity.model_fingerprint for r in plain.records} == {FP}
    assert {r.identity.model_fingerprint for r in both.records} == {FP, None} and len(both.records) == 4


# --------------------------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("status", ["partial", "failed", "refused", "interrupted", "dry_run"])
def test_runs_that_are_not_ok_are_refused(tmp_path, status):
    (out,) = run([write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3), status=status)])
    assert out.refused and status in out.refused and not out.records


def test_fewer_than_three_measured_reps_are_refused(tmp_path):
    rs = reps([9.0, 9.0, 9.0])
    rs[2] = rep(2, 9.0, ok=False)
    (out,) = run([write_doc(tmp_path, case("stock-cpu", tmp_path), rs)])  # status says ok, two reps succeeded
    assert out.refused and "2 measured reps" in out.refused


def test_power_rules(tmp_path):
    c = case("stock-cpu", tmp_path)
    (battery,) = run([write_doc(tmp_path, c, reps([9.0] * 3, power="battery"), power="battery")])
    assert battery.refused and "battery" in battery.refused
    (allowed,) = run([write_doc(tmp_path, c, reps([9.0] * 3, power="battery"), power="battery")], allow_battery=True)
    assert by_kind(allowed)[bridge.KIND_DECODE].identity.power_state == "battery"
    mixed = reps([9.0] * 3)
    mixed[3] = rep(3, 9.0, power="battery")
    (out,) = run([write_doc(tmp_path, c, mixed)])
    assert out.refused and "power source changed" in out.refused


@pytest.mark.parametrize("power", ["unknown", "solar", None])
def test_unknown_power_is_refused_like_battery(tmp_path, power):
    c = case("stock-cpu", tmp_path)
    path = write_doc(tmp_path, c, reps([9.0] * 3, power=power), power=power)  # it may have run on battery
    (out,) = run([path])
    assert out.refused and "no known power source" in out.refused and not out.records
    (allowed,) = run([path], allow_battery=True)
    assert by_kind(allowed)[bridge.KIND_DECODE].identity.power_state is None


def test_wrong_schema_and_unreadable_files_are_refused(tmp_path):
    (schema,) = run([write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3), schema="other/1")])
    assert schema.refused and "schema" in schema.refused
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    broken, missing = run([bad, tmp_path / "missing.json"])
    assert all(o.refused and "cannot read" in o.refused and not o.records for o in (broken, missing))


# --------------------------------------------------------------------------------------------------- bridge lookup


def stored(tmp_path: Path, outcomes) -> MeasurementStore:
    store = MeasurementStore(tmp_path / "store")
    for o in outcomes:
        for r in o.records:
            store.put(r)
    return store


def test_bridge_finds_the_stored_compute_coefficients(tmp_path):
    cpu = write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3))
    gpu = write_doc(tmp_path, case("stock-gpu", tmp_path), reps([1.0] * 3), log=GPU_LOG)
    records = stored(tmp_path, run([cpu, gpu])).records()
    cpu_p = bridge.device_profile("cpu", "local", bridge.HOST_CPU, "ram", None, records, MANIFEST, NOW)
    gpu_p = bridge.device_profile("cuda", "local", "CUDA0", "vram:CUDA0", None, records, MANIFEST, NOW)
    assert cpu_p.decode_ms_per_gib.provenance == "measured" and "on this model" in cpu_p.decode_ms_per_gib.note
    assert cpu_p.decode_ms_per_gib.value == pytest.approx(9.0 / UNITS_GIB)
    assert cpu_p.prefill_ms_per_gib_token.provenance == "measured"
    assert gpu_p.decode_ms_per_gib.value == pytest.approx(1.0 / UNITS_GIB)
    other = make_manifest("e" * 64)  # another model: a model-specific record does not price it
    assert bridge.device_profile("cpu", "local", "CPU", "ram", None, records, other, NOW) \
        .decode_ms_per_gib.provenance == "prior"
    stale = bridge.device_profile("cpu", "local", "CPU", "ram", None, records, MANIFEST, NOW + timedelta(days=8))
    assert stale.decode_ms_per_gib.provenance == "prior"


def test_bridge_uses_the_model_agnostic_copy_for_other_models(tmp_path):
    cpu = write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3))
    records = stored(tmp_path, run([cpu], agnostic=True)).records()
    other = bridge.device_profile("cpu", "local", "CPU", "ram", None, records, make_manifest("e" * 64), NOW)
    assert other.decode_ms_per_gib.provenance == "measured" and "model-agnostic" in other.decode_ms_per_gib.note


def test_store_round_trip_keeps_every_field(tmp_path):
    (out,) = run([write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3))])
    back = stored(tmp_path, [out]).records()
    assert sorted(back, key=lambda r: r.identity.kind) == sorted(out.records, key=lambda r: r.identity.kind)


# --------------------------------------------------------------------------------------------------- stream


def terms(**kw) -> cal.StreamTerms:
    base = dict(decode_ms=100.0, bytes_per_pass=200 * MIB, planned_bytes=200 * MIB, cpu_compute_ms=6.0,
                gpu_compute_ms=0.0, slots=2, read_bytes_per_s=4 * GIB)
    return cal.StreamTerms(**{**base, **kw})


def test_efficiency_with_overlapping_slots_reads_the_io_bound_stage():
    eff, how = cal.derive_efficiency(terms(decode_ms=100.5, gpu_compute_ms=0.5))
    assert eff == pytest.approx(200 * MIB / (4 * GIB * 0.1)) and "K=2" in how


def test_efficiency_with_one_slot_subtracts_compute():
    eff, how = cal.derive_efficiency(terms(slots=1, decode_ms=106.0))
    assert eff == pytest.approx(200 * MIB / (4 * GIB * 0.1)) and "K=1" in how


@pytest.mark.parametrize(("kw", "reason"), [
    ({"decode_ms": 7.0}, "not separable"),  # K=2 and the stage barely exceeds compute: the I/O may be hidden
    ({"slots": 1, "decode_ms": 7.0}, "not separable"),  # 1 ms of I/O in a 7 ms stage
    ({"bytes_per_pass": 260 * MIB}, "planned"),  # the run streamed another layout
    ({"decode_ms": 20.0}, "outside (0, 1]"),  # faster than the probed read rate
])
def test_efficiency_is_not_derived_when_it_cannot_be_separated(kw, reason):
    eff, why = cal.derive_efficiency(terms(**kw))
    assert eff is None and reason in why


def storage_read(device: str, value: float = 4 * GIB) -> MeasurementRecord:
    identity = MeasurementIdentity(node_id="local", device=device, backend_build="storage_probe-4r-8MiB",
                                   kind=bridge.KIND_STORAGE_READ)
    return MeasurementRecord(identity=identity, value=value, unit="bytes/s", measured_at=NOW - timedelta(hours=1),
                             samples=3, evidence="storage.json")


def stream_inputs(tmp_path: Path) -> list[Path]:
    """stock-cpu at 9 ms, stock-gpu at 1 ms (both over 450 MiB), and stream-r1-s2-k2: CPU units 0-2 (6 ms of compute),
    blocks 1-2 streamed (200 MiB per pass), GPU block 3 + head (1/3 ms), 100 ms of CPU stage."""
    cpu = write_doc(tmp_path, case("stock-cpu", tmp_path), reps([9.0] * 3))
    gpu = write_doc(tmp_path, case("stock-gpu", tmp_path), reps([1.0] * 3), log=GPU_LOG)
    stream = write_doc(tmp_path, case("stream-r1-s2-k2", tmp_path), reps([100.0 + 1 / 3] * 3, hc_per_pass=200 * MIB),
                       log=GPU_LOG, launch={"proxy_sha256": "ab" * 32})
    return [stream, cpu, gpu]  # stream first: compute is still derived before it


def test_stream_efficiency_reaches_the_bridge(tmp_path):
    volume = storage_probe.volume_of(tmp_path)
    outcomes = run(stream_inputs(tmp_path), existing=(storage_read(volume),))
    rec = by_kind(outcomes[0])[bridge.KIND_STREAM_EFFICIENCY]
    assert rec.value == pytest.approx(200 * MIB / (4 * GIB * 0.1), rel=1e-6) and rec.unit == "fraction"
    i = rec.identity
    assert i.device == volume and i.backend_build == "b11770+hcstream-abababababab"
    records = (*stored(tmp_path, outcomes).records(), storage_read(volume))
    profile = bridge.storage_profile(records, "local", None, NOW)
    assert profile.stream_efficiency.provenance == "measured"
    assert profile.stream_efficiency.value == pytest.approx(rec.value)


def test_stream_efficiency_needs_a_measured_read_rate(tmp_path):
    outcomes = run(stream_inputs(tmp_path))
    assert not outcomes[0].records and "storage read rate" in outcomes[0].skipped[0]


def test_stream_efficiency_needs_the_read_rate_of_the_models_volume(tmp_path):
    volume = storage_probe.volume_of(tmp_path)
    other = "Q:" if volume != "Q:" else "R:"  # a faster or slower drive says nothing about this run's efficiency
    outcomes = run(stream_inputs(tmp_path), existing=(storage_read(other),))
    assert not outcomes[0].records and f"storage read rate of {volume}" in outcomes[0].skipped[0]


def test_stream_efficiency_needs_measured_compute(tmp_path):
    (out,) = run(stream_inputs(tmp_path)[:1], existing=(storage_read(storage_probe.volume_of(tmp_path)),))
    assert not out.records and "CPU" in out.skipped[0] and "CUDA0" in out.skipped[0]


# --------------------------------------------------------------------------------------------------- catalog


def test_catalog_manifest_uses_the_fingerprint_the_bridge_computes(tmp_path):
    shard = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    st = shard.stat()
    doc = {"model": {"model_id": MODEL, "n_layer": 2,
                     "shards": [{"path": str(shard), "size": st.st_size, "mtime_ns": st.st_mtime_ns}]}}
    art = ModelArtifact(id=MODEL, display_name="t", family="qwen2", params_b=0.1, quantization="Q8_0", repo="r/t",
                        revision="abc", license="mit",
                        files=[ModelFile(name="m.gguf", url="https://example.invalid/m.gguf", sha256="1" * 64,
                                         size_bytes=st.st_size)])
    m = cal.catalog_manifest(doc, [art])
    assert m.fingerprint == bridge.read_manifest(art, [shard]).fingerprint
    with pytest.raises(cal.CalibrationError, match="not a GGUF in the model catalog"):
        cal.catalog_manifest(doc, [])
    doc["model"]["shards"][0]["mtime_ns"] += 1
    with pytest.raises(cal.CalibrationError, match="changed since the run"):
        cal.catalog_manifest(doc, [art])


# --------------------------------------------------------------------------------------------------- storage


def test_calibration_ranges_are_bounded_and_aligned():
    ranges = storage_probe.calibration_ranges(10 * GIB, 4 * GIB, 3, 8 * MIB)
    assert sum(n for _, n in ranges) <= 4 * GIB and all(n % (8 * MIB) == 0 for _, n in ranges)
    assert [s for s, _ in ranges] == [0, ranges[0][1], 2 * ranges[0][1]]
    small = storage_probe.calibration_ranges(100 * MIB + 5, 4 * GIB, 2, 8 * MIB)
    assert sum(n for _, n in small) <= 100 * MIB
    for bad in ((10 * GIB, 4 * GIB + 1, 3, 8 * MIB), (10 * GIB, 0, 3, 8 * MIB), (10 * GIB, GIB, 0, 8 * MIB),
                (10 * GIB, GIB, 3, 1000), (8 * MIB, GIB, 2, 8 * MIB)):
        with pytest.raises(ValueError):
            storage_probe.calibration_ranges(*bad)


def fake_measure(rates: list[float], calls: list):
    def measure(path, *, total_bytes, readers, chunk_bytes, start_offset):
        calls.append((start_offset, total_bytes, readers, chunk_bytes))
        return storage_probe.StorageMeasurement(str(path), readers, chunk_bytes, total_bytes,
                                                total_bytes / rates[len(calls) - 1])
    return measure


def test_storage_calibration_median_and_record_reach_the_bridge(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * (64 * MIB))
    calls: list = []
    result = storage_probe.calibrate(f, total_bytes=48 * MIB, runs=3, chunk_bytes=8 * MIB,
                                     measure_fn=fake_measure([3 * GIB, 5 * GIB, 4 * GIB], calls))
    assert result.error is None and result.median_bytes_per_s == pytest.approx(4 * GIB)
    assert [c[0] for c in calls] == [0, 16 * MIB, 32 * MIB] and result.bytes_read == 48 * MIB
    rec = cal.storage_record(result, node_id="local", power_state="ac", measured_at=NOW - timedelta(minutes=1),
                             evidence="storage.json", readers=4, chunk_bytes=8 * MIB)
    store = MeasurementStore(tmp_path / "store")
    store.put(rec)
    profile = bridge.storage_profile(store.records(), "local", None, NOW)
    read = profile.read_bytes_per_s
    assert read.provenance == "measured" and read.value == pytest.approx(4 * GIB)
    assert rec.identity.device == storage_probe.volume_of(f) and rec.samples == 3


def test_a_failed_storage_run_is_not_a_measurement(tmp_path):
    bad = storage_probe.StorageMeasurement("x", 4, 8 * MIB, 0, 0.0, error="ReadFile failed at 0: 87")
    good = storage_probe.StorageMeasurement("x", 4, 8 * MIB, 8 * MIB, 0.01)
    assert "ReadFile failed" in storage_probe.StorageCalibration("x", "C:", (good, bad)).error
    assert storage_probe.StorageCalibration("x", "C:", ()).error == "a run read nothing"


def test_storage_command_stores_and_keeps_evidence(tmp_path, monkeypatch):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * (32 * MIB))
    calls: list = []
    monkeypatch.setattr(cal, "host_power", lambda: {"state": "ac", "via": "test"})
    monkeypatch.setattr(storage_probe, "measure", fake_measure([5 * GIB] * 3, calls))
    store_dir, results = tmp_path / "store", tmp_path / "res"
    code = cal.main(["storage", "--model", str(f), "--bytes", str(24 * MIB), "--results-dir", str(results),
                     "--store-dir", str(store_dir)])
    assert code == 0 and len(calls) == 3
    (evidence,) = results.glob("*-storage-*/storage.json")
    doc = json.loads(evidence.read_text(encoding="utf-8"))
    assert doc["median_bytes_per_s"] == pytest.approx(5 * GIB) and doc["power"]["state"] == "ac"
    (rec,) = MeasurementStore(store_dir).records()
    assert rec.evidence == str(evidence.resolve()) and rec.identity.kind == bridge.KIND_STORAGE_READ


def test_storage_command_refuses_on_battery(tmp_path, monkeypatch):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * (32 * MIB))
    monkeypatch.setattr(cal, "host_power", lambda: {"state": "battery", "via": "test"})
    monkeypatch.setattr(storage_probe, "measure", lambda *a, **k: pytest.fail("the disk must not be read"))
    code = cal.main(["storage", "--model", str(f), "--results-dir", str(tmp_path / "res"),
                     "--store-dir", str(tmp_path / "store")])
    assert code == 2 and not (tmp_path / "store").exists()
    (evidence,) = (tmp_path / "res").glob("*-storage-*/storage.json")
    assert json.loads(evidence.read_text(encoding="utf-8"))["refused"] == "on battery power"


def test_storage_command_refuses_when_the_power_source_is_unknown(tmp_path, monkeypatch):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * (32 * MIB))
    monkeypatch.setattr(cal, "host_power", lambda: {"state": None, "via": "Win32_Battery", "error": "timeout"})
    monkeypatch.setattr(storage_probe, "measure", lambda *a, **k: pytest.fail("the disk must not be read"))
    code = cal.main(["storage", "--model", str(f), "--results-dir", str(tmp_path / "res"),
                     "--store-dir", str(tmp_path / "store")])
    assert code == 2 and not (tmp_path / "store").exists()
    (evidence,) = (tmp_path / "res").glob("*-storage-*/storage.json")
    assert json.loads(evidence.read_text(encoding="utf-8"))["refused"] == "power source unknown"


def test_storage_bytes_are_capped_at_4_gib():
    with pytest.raises(SystemExit):
        cal.parse_args(["storage", "--model", "x", "--bytes", str(4 * GIB + 1)])


# --------------------------------------------------------------------------------------------------- power and ids


@pytest.mark.parametrize(("codes", "state"), [([2], "ac"), ([3], "ac"), ([6], "ac"), ([], "ac"), ([1], "battery"),
                                              ([2, 1], "battery"), ([10], None), ([11], None)])
def test_power_from_battery_status(codes, state):
    assert cal.power_from_battery_status(codes) == state


@pytest.mark.skipif(sys.platform != "win32", reason="Win32_Battery path")
def test_host_power_parses_the_cim_query():
    def fake(cmd, **kw):
        assert "Win32_Battery" in cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout="1\r\n", stderr="")
    assert cal.host_power(run=fake) == {"state": "battery", "via": "Win32_Battery.BatteryStatus",
                                        "battery_status": [1]}


@pytest.mark.parametrize("node_id", ["mac", "LOCAL", "A" * 32, "a" * 31, "", "a" * 32 + "\n", "local\n"])
def test_node_ids_must_be_nodeview_ids(node_id):
    with pytest.raises(SystemExit):
        cal.parse_args(["bench", "x.json", "--node-id", node_id])
    assert cal.parse_args(["bench", "x.json", "--node-id", "a" * 32]).node_id == "a" * 32
