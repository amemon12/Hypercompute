"""scripts/capacity/rank_layouts.py and ranklib.py: layouts, scoring through the service code, the adapter argv to bench
case translation, the AC gate and the rank statistics. Synthetic nodes and manifests; no engine, no hardware."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from factories import contribution, make_inventory
from hypercompute.capacity import benchlib as bl
from hypercompute.capacity import bridge, costs
from hypercompute.capacity.measurements import MeasurementStore
from hypercompute.capacity.optimizer import evaluate_layout
from hypercompute.capacity.pools import HostMemory
from hypercompute.capacity.stream_config import StreamOptions
from hypercompute.capacity.types import ModelManifest, UnitSpec, Workload
from hypercompute.schemas import ModelArtifact, ModelFile, ModelView, NodeState, NodeView
from hypercompute.services import planner

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 22, 7, 0, tzinfo=UTC)
MIB, GIB = 1 << 20, 1 << 30
MODEL = "q-rank-test"
N_LAYER = 8
WORKLOAD = Workload(ctx_size=2048, prompt_tokens=512, output_tokens=64, warm=True)


def _script():
    name = "capacity_rank_layouts_script"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "capacity" / "rank_layouts.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


rank = _script()
rl = rank.rl


def _manifest() -> ModelManifest:
    units = [UnitSpec(i, "block", 150 * MIB, 150 * MIB) for i in range(N_LAYER)]
    units.append(UnitSpec(N_LAYER, "head", 500 * MIB, 500 * MIB))
    return ModelManifest(model_id=MODEL, revision="r1", shards=(), architecture="qwen2", file_type="Q4_K_M",
                         n_layer=N_LAYER, n_embd=1024, n_head=8, n_head_kv=2, key_length=128, value_length=128,
                         context_length=32768, rope=(), tokenizer_model="gpt2", vocab_size=152064,
                         chat_template_sha256=None, units=tuple(units), token_embd_bytes=0, other_bytes=0,
                         tied_embeddings=False, streamable=True, fingerprint="f" * 64)


class FakeModels:
    def __init__(self, first: Path) -> None:
        self.first = first

    def view(self, model_id: str) -> ModelView:
        art = ModelArtifact(id=MODEL, display_name=MODEL, family="qwen2", params_b=1, quantization="Q4_K_M",
                            repo="r", revision="r1", license="apache-2.0",
                            files=[ModelFile(name=f"{MODEL}.gguf", url="https://example.invalid/m", sha256="0" * 64,
                                             size_bytes=1)])
        return ModelView(artifact=art, status="downloaded", local_path=None, downloaded_bytes=0, total_bytes=0,
                         verified_sha256=True, gguf=None)

    def primary_path(self, model_id: str) -> Path:
        return self.first


def _node() -> NodeView:
    inv = make_inventory(ram_total=32 * GIB, ram_free=16 * GIB, vram_total=12 * GIB, vram_free=11 * GIB)
    return NodeView(id="local", display_name="win (this machine)", is_local=True, session_id=None,
                    state=NodeState.eligible, state_reason=None, joined_at=NOW, last_heartbeat_at=NOW,
                    heartbeat_age_s=0.0, inventory=inv, contribution=contribution(["CUDA0"], max_gib=64, reserve_gib=4),
                    worker=None, worker_state=None, verification=None)


def _host(tmp_path: Path, *, ram_gib: float = 16.0):
    m = _manifest()
    cap = planner.StreamCapability(unit_buffer_bytes=tuple(u.buffer_bytes for u in m.units), slots=2)
    mem = HostMemory(total=32 * GIB, available=int(ram_gib * GIB), commit_charge=None, commit_limit=None,
                     system_cache=None, observed_at=NOW, source="test")
    probes = bridge.Probes(memory=lambda: mem, disk_free=lambda _p: 500 * GIB, manifest=lambda _a, _s: m,
                           stream=lambda _s: cap, clock=lambda: NOW)
    first = tmp_path / "models" / MODEL / f"{MODEL}.gguf"
    return rank.Host(local=_node(), models=FakeModels(first), store=MeasurementStore(tmp_path / "store"),
                     probes=probes), first


def _cpu_prior() -> float:
    return costs.device_prior("cpu", "local", "CPU", "ram").decode_ms_per_gib.value


def _cuda_prior() -> float:
    return costs.device_prior("cuda", "local", "CUDA0", "vram:CUDA0").decode_ms_per_gib.value


# --------------------------------------------------------------------------------------------------- layouts


def test_layouts_parse_and_print_back():
    a = rl.parse_layout("a=CUDA0:0-9", N_LAYER)
    assert a.spans == (("CUDA0", 0, 9),) and not a.cpu_only and a.streamed == () and a.text() == "CUDA0:0-9"
    b = rl.parse_layout("b=CPU:0-3,CUDA0:3-9", N_LAYER)
    assert b.spans == (("CPU", 0, 3), ("CUDA0", 3, 9)) and not b.cpu_only
    d = rl.parse_layout("d=CPU:0-9;stream=2-8;slots=3", N_LAYER)
    assert d.cpu_only and d.streamed == tuple(range(2, 8)) and d.slots == 3
    assert d.text() == "CPU:0-9;stream=2-8;slots=3" and rl.parse_layout(f"d={d.text()}", N_LAYER).key() == d.key()
    assert rl.parse_layout("e=CPU:0-9;stream=2-8", N_LAYER).slots == 2  # two slots unless named
    f = rl.parse_layout("f=CPU:0-6,CUDA0:6-9;stream=1-2,3-5;slots=2", N_LAYER)  # split_cpu may stream a gapped set
    assert f.streamed == (1, 3, 4) and f.text() == "CPU:0-6,CUDA0:6-9;stream=1-2,3-5;slots=2"


@pytest.mark.parametrize("text", ["x=CPU:0-3,CUDA0:4-9", "x=CUDA0:0-8", "x=CUDA0:0-3,CPU:3-9", "x=CUDA0:0-9;stream=0-2",
                                  "x=CPU:0-3,CUDA0:3-9;stream=2-5", "x=CPU:0-9;slots=2", "=CPU:0-9", "x y=CPU:0-9",
                                  "x=CPU:0-9;stream=0-2;slots=0", "x=CPU:3-3,CUDA0:3-9", "x=CPU:0-9;stream=0-2;k=1",
                                  "x=CPU0-9", "x=CPU:0-4,CPU:4-9", "x=CPU:0-9;stream=1-4,3-5"])
def test_bad_layouts_are_refused(text):
    with pytest.raises(rl.RankError):
        rl.parse_layout(text, N_LAYER)


# --------------------------------------------------------------------------------------------------- scoring and cases


def test_an_all_gpu_layout_runs_what_the_adapter_writes(tmp_path):
    host, first = _host(tmp_path)
    found = rank.search(host, host.local, MODEL, WORKLOAD)
    s = rank.score(rl.parse_layout("a=CUDA0:0-9", N_LAYER), found)
    assert s.plan.tier == "gpu" and s.plan.device_list == ["CUDA0"] and s.plan.stream is None
    assert s.candidate.decode_ms == pytest.approx((8 * 150 + 500) / 1024 * _cuda_prior())
    build = rl.bench_case(s.plan, [str(first)], "rank-a", 12)
    e = build.case.engine
    assert (e.devices, e.tensor_split, e.ngl, e.load_mode, e.threads, e.ctx) == (("CUDA0",), (9,), "all", "mmap", 12,
                                                                               2048)
    assert e.flags == ("--cache-ram", "0") and build.case.stream is None
    diff = rl.argv_diff(build.adapter_argv, build.bench_argv)
    assert diff == {"adapter_only": ["--alias rank-a", "--metrics"],
                    "bench_only": ["--cache-ram 0", "--load-mode mmap", "-t 12"]}
    dropped = [n.split(" ", 1)[0] for n in build.notes if " dropped: " in n]
    added = [n.split(" added: ", 1)[0] for n in build.notes if " added: " in n]
    assert dropped == ["--alias", "--host", "--port", "--metrics"]
    assert added == ["--cache-ram 0", "--load-mode mmap", "-t 12"]
    assert bl.case_from_dict(json.loads(json.dumps(asdict(build.case)))) == build.case  # what bench.py reads back


def test_a_cpu_prefix_layout_is_the_windows_ram_tier(tmp_path, monkeypatch):
    monkeypatch.setattr("hypercompute.adapters.llama_rpc.engine.host_os", lambda: "windows")
    host, first = _host(tmp_path)
    s = rank.score(rl.parse_layout("b=CPU:0-3,CUDA0:3-9", N_LAYER), rank.search(host, host.local, MODEL, WORKLOAD))
    assert s.plan.tier == "gpu_ram" and s.plan.cpu_layers == 3 and s.plan.n_gpu_units == 6
    want = 3 * 150 / 1024 * _cpu_prior() + (5 * 150 + 500) / 1024 * _cuda_prior()
    assert s.candidate.decode_ms == pytest.approx(want)
    build = rl.bench_case(s.plan, [str(first)], "rank-b", 12)
    e = build.case.engine
    assert (e.devices, e.tensor_split, e.ngl, e.load_mode) == (("CUDA0",), (6,), 6, "none")
    assert e.flags == ("--no-host", "--no-repack", "--cache-ram", "0")
    assert rl.argv_diff(build.adapter_argv, build.bench_argv) == {"adapter_only": ["--alias rank-b", "--metrics"],
                                                                  "bench_only": ["-t 12"]}
    terms = rank.stage_terms(s)
    assert [t["stage"] for t in terms] == ["local:CPU", "local:CUDA0"]
    assert terms[0]["terms"][0]["coefficients"][0]["provenance"] == "prior"
    demand = rank.forced_cpu_demand(s.spec, s.inputs)
    assert demand["peak_bytes"] == s.split_cpu.stages[0].demand.peak and demand["fits"]


def test_a_streamed_cpu_only_layout_is_scored_with_its_own_residency(tmp_path):
    host, first = _host(tmp_path)
    found = rank.search(host, rank.without_accelerators(host.local), MODEL, WORKLOAD)
    assert found.inputs.host.gpu is None
    s = rank.score(rl.parse_layout("d=CPU:0-9;stream=2-8;slots=2", N_LAYER), found)
    assert s.split_cpu.stages[0].streamed_units == ()  # 16 GiB: split_cpu would keep everything resident
    spec = s.plan.stream
    assert s.plan.tier == "stream" and spec.streamed_units == list(range(2, 8)) and spec.resident_units == [0, 1, 8]
    assert spec.slots == 2 and spec.slot_bytes == 150 * MIB and spec.resident_bytes == 800 * MIB
    compute = (8 * 150 + 500) / 1024 * _cpu_prior()
    io = 900 * MIB / (costs.STORAGE_READ_BYTES_PER_S * costs.STREAM_EFFICIENCY) * 1000
    assert s.candidate.decode_ms == pytest.approx(max(io, compute)) and s.candidate.stream_bytes_per_token == 900 * MIB
    build = rl.bench_case(s.plan, [str(first)], "rank-d", 12)
    assert build.case.stream == bl.StreamLayout(resident_units=(0, 1, 8), streamed_units=tuple(range(2, 8)), slots=2,
                                                options=StreamOptions())
    e = build.case.engine
    assert (e.devices, e.ngl, e.load_mode) == ((), 0, "mmap")
    assert e.flags == ("--no-repack", "--no-op-offload", "--cache-ram", "0")
    assert rl.argv_diff(build.adapter_argv, build.bench_argv)["bench_only"] == ["-t 12"]
    demand = rank.forced_cpu_demand(s.spec, s.inputs)
    assert demand["weights"] == 800 * MIB and demand["staging"] == 2 * 150 * MIB and demand["fits"]
    io_term = rank.stage_terms(s)[0]["terms"][1]
    assert io_term["term"] == "exposed I/O" and io_term["ms"] == pytest.approx(max(io, compute) - compute)


def test_the_layout_demand_equals_the_optimizers_when_split_cpu_streams_the_same_units(tmp_path):
    host, _ = _host(tmp_path, ram_gib=6.0)  # 6 GiB - 4 GiB reserve - 512 MiB safety: the CPU-only model must stream
    found = rank.search(host, rank.without_accelerators(host.local), MODEL, WORKLOAD)
    base = evaluate_layout(found.inputs.manifest, WORKLOAD, found.inputs.host, (), [("local:CPU", 0, 9)], slots=2)
    assert base.stages[0].streamed_units
    spec = rl.spec_from_candidate("x", base, "split_cpu")
    s = rank.score(spec, found)
    assert s.candidate.decode_ms == pytest.approx(base.decode_ms)
    assert rank.forced_cpu_demand(spec, found.inputs)["peak_bytes"] == base.stages[0].demand.peak


def test_the_optimizer_best_is_added_only_when_no_listed_layout_is_it(tmp_path):
    host, _ = _host(tmp_path)
    args = rank.parse_args(["--model", MODEL, "--layout", "a=CUDA0:0-9", "--nodes-json", "unused.json"])
    scored, paths, context = rank.prepare(args, host)
    assert [s.spec.name for s in scored] == ["a"] and context["optimizer"]["best_is_listed_layout"] == "a"
    assert paths == (str(host.models.first),)
    args = rank.parse_args(["--model", MODEL, "--layout", "b=CPU:0-3,CUDA0:3-9", "--layout", "d=CPU:0-9;stream=2-8",
                            "--nodes-json", "unused.json"])
    scored, _, context = rank.prepare(args, host)
    assert [s.spec.name for s in scored] == ["b", "d", "opt-best"] and scored[2].spec.source == "optimizer best"
    assert scored[1].inputs.host.gpu is None and scored[0].inputs.host.gpu is not None


def test_power_off_stops_the_campaign_before_bench_runs(tmp_path, monkeypatch):
    host, first = _host(tmp_path)
    s = rank.score(rl.parse_layout("b=CPU:0-3,CUDA0:3-9", N_LAYER), rank.search(host, host.local, MODEL, WORKLOAD))
    monkeypatch.setattr(rank, "power_online", lambda: (False, "PowerOnline False"))
    monkeypatch.setattr(rank, "run_bench", lambda *_a: pytest.fail("bench.py must not start off AC"))
    args = rank.parse_args(["--model", MODEL, "--layout", "b=CPU:0-3,CUDA0:3-9"])
    doc = {"layouts": [], "power_checks": []}
    c = rank.Campaign(args=args, out_dir=tmp_path, model_paths=(str(first),), doc=doc)
    assert rank.run_layout(s, c) == "stopped before layout b: PowerOnline False"
    assert doc["layouts"][0]["skipped"] == "not run: PowerOnline False" and doc["power_checks"][0]["online"] is False
    assert json.loads((tmp_path / "cases" / "b.json").read_text(encoding="utf-8"))[0]["label"] == f"rank-{MODEL}-b"


def test_a_layout_whose_gpu_stage_overflows_is_skipped(tmp_path, monkeypatch):
    host, first = _host(tmp_path)
    big = rank.Host(local=host.local.model_copy(update={"inventory": make_inventory(vram_total=2 * GIB,
                                                                                   vram_free=1 * GIB)}),
                    models=host.models, store=host.store, probes=host.probes)
    s = rank.score(rl.parse_layout("a=CUDA0:0-9", N_LAYER), rank.search(big, big.local, MODEL, WORKLOAD))
    assert rank.gpu_overflows(s)
    monkeypatch.setattr(rank, "power_online", lambda: pytest.fail("no power check for a layout that is not run"))
    args = rank.parse_args(["--model", MODEL, "--layout", "a=CUDA0:0-9"])
    c = rank.Campaign(args=args, out_dir=tmp_path, model_paths=(str(first),), doc={"layouts": [], "power_checks": []})
    assert rank.run_layout(s, c) is None and c.doc["layouts"][0]["skipped"].startswith("not run: the optimizer says")


# --------------------------------------------------------------------------------------------------- outputs


@pytest.mark.parametrize(("stdout", "code", "want"), [("True\r\n", 0, True), ("False\n", 0, False),
                                                      ("True\nFalse\n", 0, False), ("True\nTrue\n", 0, True),
                                                      ("", 0, None), ("True", 1, None), ("Yes\n", 0, None)])
def test_power_online_is_true_only_when_every_battery_says_so(stdout, code, want):
    assert rl.parse_power(stdout, code)[0] is want


def test_the_bench_result_line_gives_status_and_path():
    lines = ["note: something", "[rank-m-a] ok: results/capacity/20261022T070000Z-bench-rank-m-a/bench.json\n"]
    status, path = rl.parse_bench_output(lines, ROOT)
    assert status == "ok" and path == ROOT / "results/capacity/20261022T070000Z-bench-rank-m-a/bench.json"
    assert rl.parse_bench_output(["[x] refused: results/capacity/y/bench.json"], ROOT)[0] == "refused"
    assert rl.parse_bench_output(["nothing here"], ROOT) == (None, None)


def test_ranks_and_rank_correlations():
    assert rl.ranks([3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0] and rl.ranks([1.0, 2.0, 2.0, 4.0]) == [1.0, 2.5, 2.5, 4.0]
    x = [1.0, 2.0, 3.0, 4.0]
    assert rl.kendall_tau_b(x, x) == 1.0 and rl.kendall_tau_b(x, x[::-1]) == -1.0
    assert rl.kendall_tau_b(x, [1.0, 3.0, 2.0, 4.0]) == pytest.approx(4 / 6)
    assert rl.kendall_tau_b([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]) == pytest.approx(2 / math.sqrt(6))  # one tie in y
    assert rl.spearman_rho(x, x) == pytest.approx(1.0) and rl.spearman_rho(x, x[::-1]) == pytest.approx(-1.0)
    assert rl.spearman_rho(x, [1.0, 3.0, 2.0, 4.0]) == pytest.approx(0.8)
    assert rl.spearman_rho([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]) == pytest.approx(1.5 / math.sqrt(3))
    assert rl.kendall_tau_b([1.0, 1.0], [1.0, 2.0]) is None and rl.spearman_rho([1.0], [2.0]) is None


def _row(name: str, pred: float, meas: list[float] | None, power: tuple[str, ...] = ("ac",)) -> dict:
    spread = bl.spread(meas or [])
    stages = [{"stage": "local:CPU", "stage_ms": pred / 2, "terms": [{"term": "compute", "ms": pred / 2,
                                                                      "weight_gib": 0.5}]},
              {"stage": "local:CUDA0", "stage_ms": pred / 2, "terms": [{"term": "compute", "ms": pred / 2,
                                                                        "weight_gib": 2.0}]}]
    return {"name": name, "spans_text": "CPU:0-3,CUDA0:3-9", "prediction": {"decode_ms": pred, "stages": stages},
            "measured": {"status": "ok", "decode_ms_per_token": spread, "power": sorted(power)}}


def test_ranking_uses_layouts_with_three_measured_reps_and_reads_the_terms():
    rows = [_row("a", 10.0, [12.0, 11.0, 13.0]), _row("b", 20.0, [18.0, 19.0, 20.0]),
            _row("c", 30.0, [25.0, 26.0, 24.0]), _row("d", 5.0, [7.0, 8.0])]
    r = rl.ranking(rows)
    assert r["n"] == 3 and r["order_matches"] and r["kendall_tau_b"] == 1.0 and r["spearman_rho"] == pytest.approx(1)
    assert r["excluded"] == [{"name": "d", "why": "ok"}] and rl.decode_median(rows[3]) is None
    assert rl.abs_pct_error(rows[0]) == pytest.approx(abs(10 - 12) / 12 * 100)
    check = rl.term_check(rows[0])
    assert check["residual_ms"] == pytest.approx(2.0)
    assert check["implied_cpu_decode_ms_per_gib"] == pytest.approx((12.0 - 5.0) / 0.5)
    assert check["implied_gpu_decode_ms_per_gib"] == pytest.approx((12.0 - 5.0) / 2.0)
    lines = rl.table(rows, r)
    assert len(lines) == 2 + 4 + 1 and lines[-1].startswith("n 3; order matches True; Kendall tau-b 1.000")


def test_layouts_whose_reps_were_not_all_on_ac_are_not_ranked():
    # AC is checked before each layout only; a cable pulled mid-run leaves battery reps in bench.json, and those must
    # not enter the ranking, the error column or the term check as if they were comparable
    rows = [_row("a", 10.0, [12.0, 11.0, 13.0]), _row("b", 20.0, [18.0, 19.0, 20.0]),
            _row("c", 30.0, [25.0, 26.0, 24.0]), _row("e", 5.0, [40.0, 41.0, 42.0], power=("ac", "battery")),
            _row("f", 6.0, [9.0, 9.5, 9.2], power=("None",))]
    r = rl.ranking(rows)
    assert r["n"] == 3 and set(r["measured_rank"]) == {"a", "b", "c"} and r["kendall_tau_b"] == 1.0
    why = {x["name"]: x["why"] for x in r["excluded"]}
    assert "['ac', 'battery']" in why["e"] and "not AC only" in why["f"]
    assert rl.decode_median(rows[3]) is None and rl.abs_pct_error(rows[3]) is None and rl.term_check(rows[3]) is None
    lines = rl.table(rows, r)
    assert lines[5].endswith("ok (power ac,battery)") and lines[2].endswith("ok")


def test_no_implied_compute_coefficient_for_a_stage_predicted_io_bound_with_two_slots():
    io_bound = {"stage": "local:CPU", "stage_ms": 663.0, "terms": [
        {"term": "compute", "ms": 69.0, "weight_gib": 4.07},
        {"term": "exposed I/O", "ms": 594.0, "slots": 2}]}
    row = {"name": "d", "prediction": {"decode_ms": 663.0, "stages": [io_bound]},
           "measured": {"decode_ms_per_token": bl.spread([400.0, 401.0, 399.0]), "power": ["ac"]}}
    check = rl.term_check(row)
    # (400 - 594) / 4.07 would be a negative ms/GiB: the I/O prior's error, not a compute coefficient
    assert check["implied_cpu_decode_ms_per_gib"] is None and "I/O-bound" in check["implied_cpu_not_derived"]
    assert check["residual_ms"] == pytest.approx(-263.0)
    one_slot = {**io_bound, "terms": [io_bound["terms"][0], {**io_bound["terms"][1], "slots": 1}]}
    serial = rl.term_check({**row, "prediction": {"decode_ms": 663.0, "stages": [one_slot]}})
    assert serial["implied_cpu_decode_ms_per_gib"] == pytest.approx((400.0 - 594.0) / 4.07)  # K=1: io + compute


def test_a_campaign_that_crashes_records_why_instead_of_running(tmp_path, monkeypatch):
    host, first = _host(tmp_path)
    s = rank.score(rl.parse_layout("b=CPU:0-3,CUDA0:3-9", N_LAYER), rank.search(host, host.local, MODEL, WORKLOAD))
    monkeypatch.setattr(rank, "RESULTS", tmp_path / "results")

    def boom(*_a):
        raise RuntimeError("no installed llama.cpp engine")

    monkeypatch.setattr(rank, "run_layout", boom)
    args = rank.parse_args(["--model", MODEL, "--layout", "b=CPU:0-3,CUDA0:3-9"])
    with pytest.raises(RuntimeError):
        rank.campaign(args, [s, s], (str(first),), {}, [])
    (doc_path,) = (tmp_path / "results").glob("*-ranking-*/ranking.json")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    assert doc["status"] == "failed at layout b: RuntimeError: no installed llama.cpp engine"
    assert doc["not_run"] == ["b"]


def test_measured_fields_come_from_the_bench_document(tmp_path):
    hc = {"stall_ns": 64_000_000, "passes": 64, "bytes_read": 64 * 900 * MIB}
    reps = [{"warmup": i == 0, "power": {"source": "ac"}, "gpu": None,
             "phases": [{"phase": "uncached", "ok": True, "io": {"hcstream": hc}}]} for i in range(4)]
    doc = {"status": "ok", "summary": {"n_reps": 3, "n_reps_ok": 3, "failures": [],
                                       "metrics": {"decode_ms_per_token": bl.spread([700.0, 710.0, 705.0])}},
           "reps": reps, "load": {"seconds": 3.5, "log_lines": ["load_tensors: offloaded 0/29 layers to GPU", "x"]},
           "stop": {"memory_whole_run": {"private_peak": 2 * GIB}}}
    m = rl.measured_json(doc, tmp_path / "bench.json")
    assert m["decode_ms_per_token"]["median"] == 705.0 and m["load_s"] == 3.5 and m["power"] == ["ac"]
    assert m["hcstream_stall_ms_per_pass"] == {"n": 3, "median": 1.0, "low": 1.0, "high": 1.0, "spread": "min/max"}
    assert m["load_lines"] == ["load_tensors: offloaded 0/29 layers to GPU"] and m["prefill_ms"] == {"n": 0}


def test_stream_configs_compare_without_the_stats_line(tmp_path):
    a, b, c = tmp_path / "a.cfg", tmp_path / "b.cfg", tmp_path / "c.cfg"
    a.write_text("hcstream 1\nstats C:/x/one.json\nslots 2\nend\n", encoding="utf-8")
    b.write_text("hcstream 1\nstats C:/y/two.json\nslots 2\nend\n", encoding="utf-8")
    c.write_text("hcstream 1\nstats C:/y/two.json\nslots 3\nend\n", encoding="utf-8")
    assert rl.same_config(a, b)[0] and not rl.same_config(a, c)[0]
    assert not rl.same_config(a, tmp_path / "missing.cfg")[0]


def test_unknown_or_repeated_adapter_flags_are_refused():
    with pytest.raises(rl.RankError):
        rl.argv_pairs(["-m", "x.gguf", "--mlock"])
    with pytest.raises(rl.RankError):
        rl.argv_pairs(["-c", "2048", "-c", "4096"])
    with pytest.raises(rl.RankError):
        rl.argv_pairs(["-c"])
    assert rl.argv_pairs(["--metrics", "-c", "2048"]) == [("--metrics", None), ("-c", "2048")]
