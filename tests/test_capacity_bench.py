"""Capacity benchmark harness: specs, presets, refusals, phase separation against a fake llama-server, summaries.

No hardware: a httpx.MockTransport plays llama-server (/completion streaming and not, /tokenize) with llama-server
shaped `timings`, and the only real process is a sleeping Python child used to prove the tree kill.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import psutil
import pytest
from hypercompute.capacity import benchlib as bl
from hypercompute.capacity.stream_config import StreamOptions
from hypercompute.capacity.types import GIB

ROOT = Path(__file__).resolve().parents[1]


def _load_bench_script():
    name = "capacity_bench_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "capacity" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------------- fake server


class FakeLlama:
    """llama-server shaped responses; one slot that remembers the last prompt, like -np 1."""

    def __init__(self, *, chunks: int = 3, fail_on: set[int] | None = None, drop_timings: bool = False,
                 force_cache_n: int | None = None, step_ms: dict[int, float] | None = None) -> None:
        self.requests: list[dict] = []
        self.slot: list[int] = []
        self.vocab: dict[str, int] = {}
        self.chunks, self.fail_on, self.drop_timings, self.force_cache_n = chunks, fail_on or set(), drop_timings, \
            force_cache_n
        self.step_ms = step_ms or {}  # request index -> ms per decode step (default 10)

    def tokens(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            return prompt
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in prompt.split()]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": self.tokens(body["content"])})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        self.requests.append(body)
        if len(self.requests) - 1 in self.fail_on:
            return httpx.Response(500, text="slot crashed: out of memory")
        return self._complete(body)

    def _complete(self, body: dict) -> httpx.Response:
        toks = self.tokens(body["prompt"])
        reuse = body["cache_prompt"] and self.slot[: len(toks)] == toks
        cache_n = self.force_cache_n if self.force_cache_n is not None else (len(toks) - 1 if reuse else 0)
        self.slot = toks
        n = body["n_predict"]
        step = self.step_ms.get(len(self.requests) - 1, 10.0)
        # llama-server b11770: predicted_ms starts when the first token is sampled from the prefill logits, so it
        # spans n - 1 decode steps, and predicted_per_second = (n - 1) / predicted_ms
        timings = {"cache_n": cache_n, "prompt_n": len(toks) - cache_n, "prompt_ms": 2.0 * (len(toks) - cache_n),
                   "predicted_n": n, "predicted_ms": step * max(n - 1, 0), "predicted_per_second": 1000.0 / step}
        final = {"content": "" if body["stream"] else "x " * n, "stop": True}
        if not self.drop_timings:
            final["timings"] = timings
        if not body["stream"]:
            return httpx.Response(200, json=final)
        lines = [f"data: {json.dumps({'content': 'x', 'tokens': [7], 'stop': False})}\n\n" for _ in range(self.chunks)]
        lines.append(f"data: {json.dumps(final)}\n\n")
        return httpx.Response(200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"})


class FakeObserver:
    def __init__(self) -> None:
        self.reads = 0

    def counters(self, fresh_after_unix_ms: int | None = None) -> bl.Counters:
        self.reads += 1000
        stats = {"bytes_read": self.reads * 2, "passes": self.reads // 1000}
        return bl.Counters(t=time.perf_counter(), proc_read_bytes=self.reads, sys_disk_read_bytes=self.reads * 3,
                           page_faults=self.reads // 100, working_set=100, private=200, stream_stats=stats)

    def peaks(self, t0: float, t1: float) -> tuple[int, int]:
        return 150, 250

    def gpu_memory(self) -> dict | None:
        return None

    def power(self) -> dict:
        return {"source": "ac", "via": "fake"}


def _client(server: FakeLlama) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server.handler), base_url="http://fake")


def _case(**kw) -> bl.Case:
    return bl.Case(label=kw.pop("label", "t"), model_paths=kw.pop("model_paths", ("m.gguf",)), **kw)


WORKLOAD = bl.BenchWorkload(output_tokens=16, reps=3, warmup=1)


def _run(server: FakeLlama, workload: bl.BenchWorkload = WORKLOAD, case: bl.Case | None = None) -> list[bl.RepResult]:
    case = case or _case()
    with _client(server) as client:
        return [bl.run_rep(client, case, workload, i, warmup=i < workload.warmup, observer=FakeObserver())
                for i in range(workload.warmup + workload.reps)]


# --------------------------------------------------------------------------------------------------- summaries


def test_spread_reports_min_max_below_ten_values():
    s = bl.spread([5.0, 1.0, 3.0, None])
    assert s == {"n": 3, "median": 3.0, "low": 1.0, "high": 5.0, "spread": "min/max"}


def test_spread_reports_p10_p90_from_ten_values():
    s = bl.spread(range(1, 11))
    assert s["spread"] == "p10/p90" and s["n"] == 10 and s["median"] == 5.5
    assert s["low"] == pytest.approx(1.9) and s["high"] == pytest.approx(9.1)


def test_spread_and_percentile_edges():
    assert bl.spread([]) == {"n": 0}
    assert bl.percentile([4.0], 90) == 4.0
    assert bl.percentile([0.0, 10.0], 25) == 2.5
    with pytest.raises(bl.BenchError):
        bl.percentile([], 50)


# --------------------------------------------------------------------------------------------------- specs


@pytest.mark.parametrize("kwargs", [{"reps": 2}, {"warmup": 0}, {"temperature": 0.7}, {"top_k": 40},
                                    {"output_tokens": 0}, {"output_tokens": 1}, {"reuse_output_tokens": 0},
                                    {"prompt_tokens": 0}, {"prompt": "   "}])
def test_workload_enforces_reps_warmup_and_fixed_sampling(kwargs):
    with pytest.raises(bl.BenchError):
        bl.BenchWorkload(**kwargs)


def test_workload_defaults_are_meaningful():
    w = bl.BenchWorkload()
    assert (w.output_tokens, w.reps, w.warmup, w.temperature, w.top_k) == (128, 3, 1, 0.0, 1)


@pytest.mark.parametrize("kwargs", [{"label": "bad label"}, {"label": ""}, {"model_paths": ()},
                                    {"rpc_path_class": "carrier-pigeon"},
                                    # a bare string would be iterated as characters ("C" as the model path)
                                    {"model_paths": "C:/models/m.gguf"}, {"rpc_endpoints": "10.0.0.2:50052"}])
def test_case_validation(kwargs):
    with pytest.raises(bl.BenchError):
        _case(**kwargs)


@pytest.mark.parametrize("kwargs", [{"ctx": 8}, {"ctx": 4096.5}, {"ngl": -1}, {"ngl": "most"}, {"threads": 0},
                                    {"threads": "8"}, {"devices": ("CUDA0",), "tensor_split": (1, 2)},
                                    {"devices": "CUDA0"}, {"flags": "--no-mmap"},
                                    {"devices": ("CUDA0",), "tensor_split": (1.5,)}])
def test_engine_args_validation(kwargs):
    with pytest.raises(bl.BenchError):
        bl.EngineArgs(**kwargs)


def test_stream_layout_rejects_overlap_and_zero_slots():
    with pytest.raises(bl.BenchError):
        bl.StreamLayout(resident_units=(0, 1), streamed_units=(1, 2))
    with pytest.raises(bl.BenchError):
        bl.StreamLayout(resident_units=(), streamed_units=(1,), slots=0)


def test_case_json_round_trip_and_rejection():
    case = _case(label="stream-case", model_paths=("a.gguf", "b.gguf"),
                 engine=bl.EngineArgs(devices=("CUDA0",), tensor_split=(5,), ngl=5, ctx=1024, flags=("--no-repack",)),
                 stream=bl.StreamLayout(resident_units=(0,), streamed_units=(1, 2), slots=3,
                                        options=StreamOptions(readers=2, stats_interval_ms=100)),
                 rpc_endpoints=("10.0.0.2:50052",), rpc_path_class="lan")
    assert bl.case_from_dict(json.loads(json.dumps(asdict(case)))) == case
    for bad in ([1], {"label": "x"}, {"label": "x", "model_paths": ["m"], "engine": {"gpus": 1}},
                {"label": "x", "model_paths": ["m"], "stream": "all"}, {"label": "x", "model_paths": "m.gguf"},
                {"label": "x", "model_paths": ["m"], "engine": {"flags": "--no-mmap"}}):
        with pytest.raises(bl.BenchError):
            bl.case_from_dict(bad)


# --------------------------------------------------------------------------------------------------- presets


def test_parse_preset_names():
    bench = _load_bench_script()
    assert bench.parse_preset("stock-cpu").kind == "stock-cpu"
    assert bench.parse_preset("stock-gpu").kind == "stock-gpu"
    assert bench.parse_preset("gpu-ram-6") == bench.Preset(name="gpu-ram-6", kind="gpu-ram", cpu_layers=6)
    assert bench.parse_preset("stream-r8-s20-k2") == bench.Preset(name="stream-r8-s20-k2", kind="stream", resident=8,
                                                                  streamed=20, slots=2)
    for bad in ("gpu", "gpu-ram-0", "gpu-ram-x", "stream-r0-s0-k2", "stream-r1-s4-k0", "stream-r8-s20"):
        with pytest.raises(bl.BenchError):
            bench.parse_preset(bad)


def _preset_case(name: str, n_layer: int = 28, windows: bool = True) -> bl.Case:
    bench = _load_bench_script()
    return bench.case_for_preset(bench.parse_preset(name), ["m.gguf"], n_layer, label="p", ctx=1024, threads=8,
                                 windows=windows)


def test_preset_engine_args_follow_the_tiers():
    cpu = _preset_case("stock-cpu").engine
    assert cpu.devices == () and cpu.ngl == 0 and cpu.load_mode == "mmap" and cpu.ctx == 1024 and cpu.threads == 8
    gpu = _preset_case("stock-gpu").engine
    assert gpu.devices == ("CUDA0",) and gpu.ngl == "all" and gpu.tensor_split == (29,)
    ram = _preset_case("gpu-ram-6").engine
    assert ram.ngl == 23 and ram.tensor_split == (23,) and ram.load_mode == "none"
    assert ram.flags == ("--no-host", "--no-repack", "--cache-ram", "0")
    assert _preset_case("gpu-ram-6", windows=False).engine.load_mode == "mmap"
    assert all(_preset_case(p).engine.flags[-2:] == ("--cache-ram", "0") for p in ("stock-cpu", "stock-gpu"))
    with pytest.raises(bl.BenchError):
        _preset_case("gpu-ram-28")


def test_stream_presets_cpu_only_and_gpu_suffix():
    full = _preset_case("stream-r8-s20-k2")
    assert full.stream.resident_units == (*range(8), 28) and full.stream.streamed_units == tuple(range(8, 28))
    assert full.stream.slots == 2 and full.engine.devices == ()
    assert "--no-op-offload" in full.engine.flags and "--cache-ram" in full.engine.flags
    part = _preset_case("stream-r2-s4-k3")
    assert part.stream.resident_units == (0, 1) and part.stream.streamed_units == (2, 3, 4, 5)
    assert part.engine.devices == ("CUDA0",) and part.engine.ngl == 23
    with pytest.raises(bl.BenchError):
        _preset_case("stream-r20-s20-k2")


def test_build_argv_for_cpu_gpu_and_rpc_cases():
    cpu = bl.build_argv(_preset_case("stock-cpu"), Path("llama-server.exe"), 4242)
    assert cpu[cpu.index("--device") + 1] == "none" and cpu[cpu.index("-ngl") + 1] == "0"
    assert cpu[cpu.index("--port") + 1] == "4242" and cpu[cpu.index("--host") + 1] == "127.0.0.1"
    assert cpu[cpu.index("-c") + 1] == "1024" and cpu[cpu.index("-t") + 1] == "8" and "--rpc" not in cpu
    ram = bl.build_argv(_preset_case("gpu-ram-6"), Path("s"), 1)
    assert ram[ram.index("--load-mode") + 1] == "none" and ram[ram.index("--tensor-split") + 1] == "23"
    rpc_case = _case(engine=bl.EngineArgs(devices=("RPC0", "CUDA0"), tensor_split=(10, 19), ngl="all"),
                     rpc_endpoints=("10.0.0.2:50052",), rpc_path_class="lan")
    rpc = bl.build_argv(rpc_case, Path("s"), 1)
    assert rpc[rpc.index("--rpc") + 1] == "10.0.0.2:50052" and rpc[rpc.index("--device") + 1] == "RPC0,CUDA0"


# --------------------------------------------------------------------------------------------------- refusals


FACTS = bl.ModelFacts(n_layer=4, unit_bytes=(100, 100, 100, 100, 50), token_embd_bytes=30, tied_embeddings=False,
                      kv_bytes_per_layer_token=2, file_bytes=480)


def test_predict_host_bytes_per_tier():
    cpu = bl.predict_host_bytes(_case(engine=bl.EngineArgs(ctx=100)), FACTS)
    assert cpu.total_bytes == 450 + 30 + 4 * 100 * 2 + bl.COMPUTE_PRIOR_BYTES and cpu.provenance == "estimate"
    gpu = bl.predict_host_bytes(_case(engine=bl.EngineArgs(devices=("CUDA0",), ngl="all", ctx=100)), FACTS)
    assert gpu.total_bytes == 30 + bl.COMPUTE_PRIOR_BYTES
    ram = bl.predict_host_bytes(_case(engine=bl.EngineArgs(devices=("CUDA0",), ngl=3, ctx=100)), FACTS)
    assert ram.total_bytes == 200 + 30 + 2 * 100 * 2 + bl.COMPUTE_PRIOR_BYTES
    stream = bl.predict_host_bytes(_case(engine=bl.EngineArgs(ctx=100), stream=bl.StreamLayout(
        resident_units=(0, 4), streamed_units=(1, 2, 3), slots=2)), FACTS)
    assert dict(stream.parts)["hcstream resident"] == 150 and dict(stream.parts)["hcstream slots"] == 200
    assert dict(stream.parts)["cpu unit weights"] == 0
    tied = bl.predict_host_bytes(_case(engine=bl.EngineArgs(ctx=100)),
                                 bl.ModelFacts(**{**asdict(FACTS), "tied_embeddings": True}))
    assert "token_embd" not in dict(tied.parts)
    with pytest.raises(bl.BenchError):
        bl.predict_host_bytes(_case(engine=bl.EngineArgs(devices=("CUDA0",), ngl="all"), stream=bl.StreamLayout(
            resident_units=(0,), streamed_units=())), FACTS)


def _preflight(**kw) -> bl.Preflight:
    base = {"port": 5000, "port_listener": None, "available_ram_bytes": 16 * GIB,
            "predicted": bl.Residency(total_bytes=2 * GIB, parts=()), "free_disk_bytes": 40 * GIB, "disk_path": "C:"}
    return bl.Preflight(**{**base, **kw})


def test_refusal_rules():
    assert bl.refusals(_preflight()) == ()
    assert "another llama-server" in bl.refusals(_preflight(port_listener="llama-server.exe (pid 7)"))[0]
    assert "python.exe (pid 9)" in bl.refusals(_preflight(port_listener="python.exe (pid 9)"))[0]
    ram = bl.refusals(_preflight(available_ram_bytes=int(5.5 * GIB)))
    assert len(ram) == 1 and "below the 4 GiB floor" in ram[0]
    assert bl.refusals(_preflight(available_ram_bytes=6 * GIB)) == ()  # exactly 4 GiB left is allowed
    disk = bl.refusals(_preflight(free_disk_bytes=4 * GIB))
    assert len(disk) == 1 and "free disk" in disk[0]
    assert len(bl.refusals(_preflight(port_listener="x (pid 1)", available_ram_bytes=GIB, free_disk_bytes=0))) == 3


# --------------------------------------------------------------------------------------------------- phases


def test_phases_are_separated_and_sent_as_specified():
    server = FakeLlama()
    reps = _run(server)
    assert len(server.requests) == 3 * len(reps)
    uncached = [r for r in server.requests if r["stream"]]
    assert all(r["cache_prompt"] is False and r["n_predict"] == 16 for r in uncached)
    assert all(r["temperature"] == 0 and r["top_k"] == 1 and r["seed"] == 7 and r["ignore_eos"] for r in uncached)
    assert len({r["prompt"].split("]")[0] for r in uncached}) == len(reps)  # a unique prefix per rep
    assert len({len(r["prompt"].split()) for r in uncached}) == 1  # same length for every rep
    heads = [rep.phase("uncached").prompt_head for rep in reps]
    assert len(set(heads)) == len(reps)  # the recorded head itself shows the prompts differ
    for i in range(len(reps)):
        prime, reuse = server.requests[3 * i + 1], server.requests[3 * i + 2]
        assert prime["prompt"] == reuse["prompt"] and not prime["cache_prompt"] and reuse["cache_prompt"]
    rep = reps[1]
    assert rep.ok and not rep.warmup and reps[0].warmup
    u, r = rep.phase("uncached"), rep.phase("prefix_reuse")
    assert u.timings.cache_n == 0 and u.ttft_ms is not None and u.cache_prompt is False
    assert r.timings.prompt_n == 1 and r.timings.cache_n > 0
    m = bl.rep_metrics(rep)
    assert m["prefill_ms"] == 2.0 * m["prefill_tokens"] and m["decode_ms"] == 150.0 and m["decode_steps"] == 15
    assert m["decode_tok_s"] == pytest.approx(100.0) and m["decode_ms_per_token"] == pytest.approx(10.0)
    assert m["reuse_cache_n"] == r.timings.cache_n and m["working_set_peak"] == 150 and m["private_peak"] == 250


def test_decode_uses_engine_timings_not_chunk_counts():
    reps = _run(FakeLlama(chunks=2))
    assert all(bl.rep_metrics(r)["decode_tokens"] == 16 for r in reps)


def test_io_window_counts_bytes_per_token_and_hcstream_deltas():
    rep = _run(FakeLlama())[1]
    io = rep.phase("uncached").io
    assert io["proc_read_bytes"] == 1000 and io["sys_disk_read_bytes"] == 3000 and io["page_faults"] == 10
    assert io["hcstream"]["bytes_read"] == 2000 and io["hcstream"]["passes"] == 1
    m = bl.rep_metrics(rep)
    assert m["proc_read_bytes_per_token"] == 1000 / 16 and m["hcstream_bytes_per_token"] == 2000 / 16
    assert m["hcstream_bytes_per_pass"] == 2000  # bytes_read delta over the passes delta


def test_failed_rep_is_recorded_with_its_error_and_never_dropped():
    reps = _run(FakeLlama(fail_on={6}))  # request 6 is the uncached request of rep 2
    assert len(reps) == 4
    bad = reps[2]
    assert not bad.ok and "HTTP 500" in bad.phase("uncached").error and "out of memory" in bad.phase("uncached").error
    assert bad.phase("prefix_reuse").ok  # later phases of the rep still ran
    summary = bl.summarize(reps)
    assert summary["n_reps"] == 3 and summary["n_reps_ok"] == 2 and summary["n_warmup"] == 1
    assert summary["failures"] == [{"rep": 2, "warmup": False, "phase": "uncached",
                                    "error": bad.phase("uncached").error}]
    assert summary["metrics"]["decode_tok_s"]["n"] == 2 and summary["metrics"]["reuse_cache_n"]["n"] == 3
    assert bl.run_status(bl.LoadResult(ok=True, seconds=1.0, error=None), reps, 3) == "partial"


def test_failed_prime_marks_reuse_not_run():
    reps = _run(FakeLlama(fail_on={1}))  # the warmup's priming request
    reuse = reps[0].phase("prefix_reuse")
    assert not reuse.ok and "priming request failed" in reuse.error
    assert {f["phase"] for f in bl.summarize(reps)["failures"]} == {"prefix_prime", "prefix_reuse"}


def test_uncached_phase_that_reused_cache_is_a_failure():
    reps = _run(FakeLlama(force_cache_n=5))
    assert all("not an uncached prefill" in r.phase("uncached").error for r in reps)


def test_reuse_that_reused_nothing_is_a_failure():
    reps = _run(FakeLlama(force_cache_n=0))
    assert all("reused nothing" in r.phase("prefix_reuse").error for r in reps)
    assert all(r.phase("uncached").ok for r in reps)


def test_missing_timings_is_a_failure():
    reps = _run(FakeLlama(drop_timings=True))
    assert all("no engine timings" in r.phase("uncached").error for r in reps)
    assert bl.summarize(reps)["n_reps_ok"] == 0


def test_short_decode_is_a_failure():
    res = bl.PhaseResult(phase="uncached", ok=True, timings=bl.Timings(prompt_n=5, prompt_ms=1.0, predicted_n=10,
                                                                         predicted_ms=1.0, cache_n=0))
    assert "decoded 10 of 16 tokens" in bl.check_uncached(res, 16).error


def test_decode_rate_counts_decode_steps_like_llama_server():
    # timings of the smoke run's rep 3: predicted_ms spans 15 decode steps, not 16
    t = bl.Timings(prompt_n=89, prompt_ms=146.348, predicted_n=16, predicted_ms=130.425, cache_n=0)
    assert t.decode_steps == 15 and t.decode_tok_s == pytest.approx(15 / 0.130425)
    assert t.decode_ms_per_token == pytest.approx(130.425 / 15)
    one = bl.Timings(prompt_n=5, prompt_ms=1.0, predicted_n=1, predicted_ms=0.001, cache_n=0)
    assert one.decode_steps == 0 and one.decode_tok_s is None and one.decode_ms_per_token is None


def test_warmup_reps_are_kept_but_excluded_from_summaries():
    reps = _run(FakeLlama(step_ms={0: 100.0}))  # request 0 is the warmup's uncached request: 10 tok/s
    assert reps[0].warmup and bl.rep_metrics(reps[0])["decode_tok_s"] == pytest.approx(10.0)
    s = bl.summarize(reps)
    assert s["n_warmup"] == 1 and len(reps) == 4
    assert s["metrics"]["decode_tok_s"]["n"] == 3
    assert s["metrics"]["decode_tok_s"]["low"] == s["metrics"]["decode_tok_s"]["high"] == pytest.approx(100.0)


def test_unexpected_request_exceptions_are_recorded_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("stream torn down")  # not an httpx.HTTPError, like httpx.StreamError

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fake") as client:
        rep = bl.run_rep(client, _case(), WORKLOAD, 1, warmup=False, observer=FakeObserver())
    assert not rep.ok and all("stream torn down" in p.error for p in rep.phases)


def test_memory_peaks_come_from_successful_phases_only():
    good = bl.PhaseResult(phase="uncached", ok=True, timings=bl.Timings(5, 1.0, 16, 150.0, 0),
                          io={"working_set_peak": 10, "private_peak": 20})
    bad = bl.PhaseResult(phase="prefix_prime", ok=False, error="HTTP 500", io={"working_set_peak": 999,
                                                                                "private_peak": 999})
    rep = bl.RepResult(index=1, warmup=False, ok=False, phases=(good, bad), gpu=None, power={}, rpc=None,
                       started_at="")
    m = bl.rep_metrics(rep)
    assert m["working_set_peak"] == 10 and m["private_peak"] == 20


def test_prompt_token_target_sends_exact_token_lists():
    server = FakeLlama()
    reps = _run(server, bl.BenchWorkload(prompt="alpha beta gamma", prompt_tokens=11, output_tokens=4))
    assert all(r.ok for r in reps)
    sent = [r["prompt"] for r in server.requests]
    assert all(isinstance(p, list) and len(p) == 11 for p in sent)
    uncached = [r["prompt"] for r in server.requests if r["stream"]]
    assert len({tuple(p[:8]) for p in uncached}) == len(uncached)


def test_prompt_token_target_is_reached_when_the_tag_outweighs_the_prompt():
    # The rep tag is prefixed once, not once per copy. Sizing the copies by the tokens of tag + prompt fell short in
    # the first real run (a ~45-token tag, a ~55-token paragraph, a 512-token target): every rep failed.
    server = FakeLlama()
    reps = _run(server, bl.BenchWorkload(prompt="alpha beta gamma", prompt_tokens=40, output_tokens=4))
    assert all(r.ok for r in reps), bl.failures(reps)
    assert all(isinstance(r["prompt"], list) and len(r["prompt"]) == 40 for r in server.requests)


def test_rpc_cases_record_endpoints_and_path_class():
    case = _case(rpc_endpoints=("192.168.1.9:50052",), rpc_path_class="lan",
                 engine=bl.EngineArgs(devices=("RPC0",), ngl="all"))
    rep = _run(FakeLlama(), case=case)[0]
    assert rep.rpc == {"endpoints": ["192.168.1.9:50052"], "path_class": "lan"}


def test_window_io_without_stream_stats_and_with_errors():
    a = bl.Counters(t=0.0, proc_read_bytes=10, sys_disk_read_bytes=0, page_faults=0, working_set=5, private=900)
    b = bl.Counters(t=1.0, proc_read_bytes=30, sys_disk_read_bytes=0, page_faults=0, working_set=0, private=0,
                    error="counters unavailable: NoSuchProcess")
    io = bl.window_io(a, b, (7, 8))
    # b's counters failed: its zero placeholders give no deltas, never negative or cumulative "reads"
    assert io["proc_read_bytes"] is None and io["page_faults"] is None and io["sys_disk_read_bytes"] is None
    assert io["working_set_peak"] == 7 and io["private_peak"] == 900
    assert "hcstream" not in io and io["errors"] == ["counters unavailable: NoSuchProcess"]
    assert bl.window_io(a, a, (0, 0))["proc_read_bytes"] == 0


# --------------------------------------------------------------------------------------------------- load


def test_wait_healthy_until_ok_or_exit_or_timeout():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200 if calls["n"] >= 3 else 503)

    clock = iter(float(i) for i in range(1000))
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fake") as c:
        ok, seconds, err = bl.wait_healthy(c, lambda: True, 60, 0.0, clock=lambda: next(clock), sleep=lambda s: None)
        assert ok and err is None and calls["n"] == 3
        ok, _, err = bl.wait_healthy(c, lambda: False, 60, 0.0, sleep=lambda s: None)
        assert not ok and "exited" in err
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)), base_url="http://f") as c:
        clock2 = iter(float(i) for i in range(1000))
        ok, _, err = bl.wait_healthy(c, lambda: True, 5, 0.0, clock=lambda: next(clock2), sleep=lambda s: None)
        assert not ok and "not healthy after 5s" in err


def test_load_log_lines_and_build_parsing():
    log = ("0.00.28 I cmn  common_param: common_params_print_info: build 11770 (39da130fc) with Clang 20.1.8\n"
           "0.00.30 I srv  init_listene: The UI is disabled\n"
           "0.00.56 I load_tensors:   CPU_Mapped model buffer size =    89.26 MiB\n"
           "hcstream: attached: 24 streamed layers in 2 slots of 10887168 bytes\n"
           "0.00.84 I srv  llama_server: model loaded\n")
    lines = bl.load_log_lines(log)
    assert len(lines) == 4 and not any("UI is disabled" in line for line in lines)
    assert bl.parse_build(log) == {"build": 11770, "commit": "39da130fc", "build_source": "llama-server log"}
    assert bl.parse_build("nothing here") is None


def test_load_log_lines_keep_the_last_lines_past_the_limit():
    text = "\n".join(f"load_tensors: buffer {i}" for i in range(300)) + "\nllama_server: model loaded\n"
    lines = bl.load_log_lines(text, limit=20)
    assert len(lines) == 20 and lines[0] == "load_tensors: buffer 0" and lines[-1] == "llama_server: model loaded"
    assert "282 load lines omitted" in lines[10]
    assert bl.log_tail("a\n\nb\nc\n", 2) == ("b", "c")


def test_parse_nvidia_smi():
    bench = _load_bench_script()
    gpus = "0, NVIDIA GeForce RTX 5070 Ti Laptop GPU, 1234, 12227\nbad line\n"
    apps = "4242, 900\n77, [N/A]\n"
    out = bench.parse_nvidia_smi(gpus, apps, 4242)
    assert out["gpus"] == [{"index": 0, "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU", "used_mib": 1234,
                            "total_mib": 12227}]
    assert out["process_used_mib"] == 900 and bench.parse_nvidia_smi(gpus, apps, 77)["process_used_mib"] is None


# --------------------------------------------------------------------------------------------------- document


def test_results_document_schema_fields():
    reps = _run(FakeLlama())
    load = bl.LoadResult(ok=True, seconds=1.25, error=None, log_lines=("x",), build={"build": 11770})
    doc = bl.results_document(case=_case(), workload=WORKLOAD, status=bl.run_status(load, reps, 3),
                              started_at="2026-10-21T00:00:00+00:00", host={"system": "Windows"},
                              model={"fingerprint": None}, preflight=_preflight(), engine={"build": 11770},
                              launch={"argv": ["s"]}, load=load, reps=reps, stop={"exit_code": 0})
    for key in ("schema", "status", "started_at", "finished_at", "case", "workload", "host", "llama_cpp", "model",
                "preflight", "refused", "launch", "load", "reps", "summary", "stop", "error", "provenance"):
        assert key in doc
    assert doc["schema"] == "hypercompute.capacity.bench/1" and doc["status"] == "ok"
    assert len(doc["reps"]) == 4 and doc["reps"][0]["phases"][0]["timings"]["predicted_n"] == 16
    assert doc["summary"]["metrics"]["decode_tok_s"]["spread"] == "min/max"
    assert json.loads(json.dumps(doc))["case"]["label"] == "t"


def test_run_status():
    ok = bl.LoadResult(ok=True, seconds=1.0, error=None)
    reps = _run(FakeLlama())
    assert bl.run_status(None, reps, 3) == "failed"
    assert bl.run_status(bl.LoadResult(ok=False, seconds=1.0, error="x"), reps, 3) == "failed"
    assert bl.run_status(ok, reps, 3) == "ok"
    assert bl.run_status(ok, reps[:2], 3) == "partial"
    assert bl.run_status(ok, _run(FakeLlama(drop_timings=True)), 3) == "failed"


# --------------------------------------------------------------------------------------------------- script


def test_script_refuses_without_launching(tmp_path, monkeypatch):
    bench = _load_bench_script()
    monkeypatch.setattr(bench, "RESULTS", tmp_path)
    monkeypatch.setattr(bench, "model_identity", lambda paths: {"model_id": "m"})
    monkeypatch.setattr(bench, "model_facts", lambda paths: FACTS)
    monkeypatch.setattr(bench, "preflight", lambda case, facts, port: _preflight(available_ram_bytes=GIB))
    monkeypatch.setattr(bench, "execute", lambda *a, **k: pytest.fail("a refused case must not launch"))
    opts = bench.Options(port=0, min_available_gib=2.0, dry_run=False)
    path, status = bench.run_case(_case(label="refuse-me"), WORKLOAD, opts)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert status == "refused" and doc["status"] == "refused" and "4 GiB floor" in doc["refused"][0]
    assert path.parent.name.endswith("-bench-refuse-me") and doc["schema"] == bl.SCHEMA
    again, _ = bench.run_case(_case(label="refuse-me"), WORKLOAD, opts)  # same label, very likely the same second
    assert again != path and path.is_file() and again.is_file()


def test_runs_of_one_label_in_one_second_never_share_a_directory(tmp_path, monkeypatch):
    bench = _load_bench_script()
    monkeypatch.setattr(bench, "RESULTS", tmp_path)
    started = datetime(2026, 10, 21, tzinfo=UTC)
    first, second = bench.new_out_dir("x", started), bench.new_out_dir("x", started)
    assert first.name == "20261021T000000Z-bench-x" and second.name == "20261021T000000Z-bench-x-2"
    assert first.is_dir() and second.is_dir()


def test_script_reads_case_files(tmp_path):
    bench = _load_bench_script()
    case = _case(label="from-file", engine=bl.EngineArgs(ctx=2048))
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([asdict(case)]), encoding="utf-8")
    args = bench.parse_args(["--case-file", str(path), "--output-tokens", "16"])
    assert bench.cases_from_args(args) == [case]
    assert bench.workload_from_args(args).output_tokens == 16
    path.write_text(json.dumps([asdict(case), asdict(case)]), encoding="utf-8")
    with pytest.raises(bl.BenchError, match="unique"):
        bench.cases_from_args(args)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(bl.BenchError):
        bench.cases_from_args(args)


@pytest.mark.parametrize("kwargs", [{"port": 70000}, {"port": -1}, {"min_available_gib": float("inf")},
                                    {"min_available_gib": float("nan")}, {"min_available_gib": -1.0}])
def test_script_options_validation(kwargs):
    bench = _load_bench_script()
    with pytest.raises(bl.BenchError):
        bench.Options(**{"port": 0, "min_available_gib": 2.0, "dry_run": False, **kwargs})


def test_model_shards_requires_one_complete_model(tmp_path):
    bench = _load_bench_script()
    split = tmp_path / "split"
    split.mkdir()
    for i in (1, 3):
        (split / f"m-{i:05d}-of-00003.gguf").write_bytes(b"x")
    with pytest.raises(bl.BenchError, match="not one complete model"):
        bench.model_shards(str(split))  # shard 2 missing: shard indices would be misread
    (split / "m-00002-of-00003.gguf").write_bytes(b"x")
    assert [p.name for p in bench.model_shards(str(split))] == [f"m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    (split / "other.gguf").write_bytes(b"x")
    with pytest.raises(bl.BenchError, match="not one complete model"):
        bench.model_shards(str(split))
    single = tmp_path / "single"
    single.mkdir()
    (single / "a.gguf").write_bytes(b"x")
    (single / "mmproj-a.gguf").write_bytes(b"x")
    assert bench.model_shards(str(single)) == (single / "a.gguf",)


def test_prepare_launch_strips_llama_arg_env_and_keeps_the_key_out_of_meta(tmp_path, monkeypatch):
    bench = _load_bench_script()
    fake = SimpleNamespace(bin_dir=tmp_path, release="b11770", variant="cpu", accelerator="cpu", source="test")
    monkeypatch.setattr(bench.engine, "find_installed", lambda: fake)
    monkeypatch.setenv("LLAMA_ARG_UBATCH", "64")
    launch, info = bench.prepare_launch(_case(), tmp_path, 4242, "secret-key-value")
    assert "LLAMA_ARG_UBATCH" not in launch.env and "LLAMA_ARG_UBATCH" in launch.meta["env_removed"]
    assert launch.env["LLAMA_API_KEY"] == "secret-key-value"
    assert "secret-key-value" not in json.dumps(launch.meta) + json.dumps(info)


def _sleeping_launch(bench, tmp_path: Path):
    return bench.Launch(argv=(sys.executable, "-c", "import time; time.sleep(60)"), cwd=tmp_path, env=dict(os.environ),
                        stats_path=None, meta={"log": str(tmp_path / "child.log")})


@pytest.mark.parametrize("stop_with", [KeyboardInterrupt, RuntimeError])
def test_execute_keeps_finished_work_and_kills_the_child(tmp_path, monkeypatch, stop_with):
    bench = _load_bench_script()
    monkeypatch.setattr(bench, "prepare_launch", lambda *a: (_sleeping_launch(bench, tmp_path), {"release": "t"}))
    reps = _run(FakeLlama())
    procs = []

    def measure(client, proc, case, workload, observer, t_start, log_path):
        procs.append(proc)
        yield bl.LoadResult(ok=True, seconds=1.0, error=None, memory={"loaded_at_s": 0.0})
        yield from reps[:3]  # the warmup and two measured reps finish, then the run stops
        raise stop_with("stopped mid-run")

    monkeypatch.setattr(bench, "measure", measure)
    fields = bench.execute(_case(), WORKLOAD, _preflight(), tmp_path,
                           bench.Options(port=0, min_available_gib=0.0, dry_run=False))
    assert fields["load"].ok and fields["reps"] == reps[:3]
    if stop_with is KeyboardInterrupt:
        assert fields["status"] == "interrupted" and fields["error"] == "interrupted by the operator"
    else:
        assert fields["status"] == "partial" and "stopped mid-run" in fields["error"]
    assert fields["stop"]["child_alive_after_stop"] is False and procs[0].poll() is not None
    assert (tmp_path / "samples.json").is_file()


def test_failed_load_records_exit_code_and_log_tail(tmp_path):
    bench = _load_bench_script()
    log = tmp_path / "srv.log"
    with log.open("wb") as out:
        proc = subprocess.Popen([sys.executable, "-c", "import sys; print('error: invalid argument: --bogus'); "
                                 "sys.exit(3)"], stdout=out, stderr=subprocess.STDOUT)
        proc.wait(timeout=30)
    observer = bench.ChildObserver(proc.pid, bench.probe.ProcessSampler(pid=proc.pid), time.perf_counter(), None,
                                   cuda=False)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)), base_url="http://f") as client:
        load = bench.load_result(client, proc, WORKLOAD, observer, time.perf_counter(), log)
    assert not load.ok and "exit code 3" in load.error
    assert any("invalid argument: --bogus" in line for line in load.log_tail)


def test_child_observer_drops_stale_or_missing_hcstream_stats(tmp_path, monkeypatch):
    bench = _load_bench_script()
    monkeypatch.setattr(bench.ChildObserver, "STATS_WAIT_S", 0.05)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"updated_unix_ms": 1000, "bytes_read": 5}), encoding="utf-8")
    observer = bench.ChildObserver(os.getpid(), bench.probe.ProcessSampler(pid=os.getpid()), time.perf_counter(),
                                   stats, cuda=False)
    stale = observer.counters(fresh_after_unix_ms=2000)
    assert stale.stream_stats is None and "not refreshed" in stale.stats_error and stale.error is None
    fresh = observer.counters(fresh_after_unix_ms=1000)
    assert fresh.stream_stats["bytes_read"] == 5 and fresh.stats_error is None
    stats.unlink()
    assert "unreadable" in observer.counters(fresh_after_unix_ms=0).stats_error
    io = bl.window_io(fresh, stale, (0, 0))  # a stale end gives no hcstream delta at all, never an undercount
    assert "hcstream" not in io and any("not refreshed" in e for e in io["errors"])


def test_kill_tree_leaves_a_reaped_childs_pid_alone(monkeypatch):
    bench = _load_bench_script()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    monkeypatch.setattr(bench.psutil, "Process", lambda pid: pytest.fail("looked up a pid the child no longer owns"))
    result = bench.kill_tree(proc)
    assert result["children"] == 0 and result["error"] is None and result["exit_code"] == 0


def test_kill_tree_stops_the_child_and_its_children():
    bench = _load_bench_script()
    code = ("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 20
        while not psutil.Process(proc.pid).children(recursive=True) and time.monotonic() < deadline:
            time.sleep(0.1)
        kids = psutil.Process(proc.pid).children(recursive=True)
        assert kids, "the grandchild never started"
        result = bench.kill_tree(proc)
        assert result["child_alive_after_stop"] is False and result["children_alive"] == []
        assert not any(k.is_running() and k.status() != psutil.STATUS_ZOMBIE for k in kids)
    finally:
        if proc.poll() is None:
            proc.kill()
