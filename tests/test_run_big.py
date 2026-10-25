"""scripts/capacity/run_big.py and hostmon.py against a mocked gateway (httpx.MockTransport) and fake host probes.

No engine, GPU or network: the fake gateway serves the owner bootstrap, plans, explain, a deployment that loads and
becomes ready, a streamed chat whose engine timings follow the prompt length, nodes, network and runs.
A Mac host is simulated with the darwin HostProfile and probes that report what psutil reports on macOS.
"""

from __future__ import annotations

import _thread
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "owner-token-not-for-records"
DEP_ID = "d" * 32
GIB = 1 << 30


def _script():
    name = "capacity_run_big_script"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "capacity" / "run_big.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


rb = _script()
hm = sys.modules["hostmon"]


def _plan(feasible: bool = True) -> dict[str, Any]:
    return {"id": "p1", "model_id": "big", "feasible": feasible, "tier": "gpu_ram", "device_list": ["CUDA0", "RPC0"],
            "reasons": [] if feasible else ["does not fit"],
            "prediction": {"decode_ms": 200.0, "load_ms": 60000.0, "stream_bytes_per_token": 0}}


class FakeGateway:
    """Routes the runner uses; state advances on each poll. Thread safe (the guard stops from the sampler thread)."""

    def __init__(self, *, capacity: bool = True, active: bool = False, downloaded: bool = True,
                 feasible: bool = True, final: str = "ready", stay_loading: bool = False,
                 chat_error: bool = False, interrupt_chat: bool = False, broken_runs: bool = False,
                 create_raises: BaseException | None = None, interrupt_at_chat: int | None = None,
                 parallel: int | None = None, tokens_per_word: float = 1.0) -> None:
        self.capacity, self.active, self.downloaded, self.feasible = capacity, active, downloaded, feasible
        self.broken_runs = broken_runs
        self.final, self.stay_loading, self.chat_error, self.interrupt_chat = final, stay_loading, chat_error, \
            interrupt_chat
        self.create_raises = create_raises  # raised after the deployment exists: the reply is lost
        self.interrupt_at_chat = interrupt_at_chat  # Ctrl+C when this chat (1-based) is requested
        self.parallel, self.tokens_per_word = parallel, tokens_per_word  # -np N -kvu in engine_args; prompt_n per word
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str, dict[str, str], Any]] = []
        self.dep: dict[str, Any] | None = None
        self.polls = 0
        self.chats = 0
        self.cpu = 10.0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport(), base_url="http://127.0.0.1:8001")

    def calls(self, method: str, path: str) -> list[Any]:
        return [r for r in self.requests if r[0] == method and r[1] == path]

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        with self.lock:
            self.requests.append((request.method, request.url.path, dict(request.headers), body))
        writes = request.method in ("POST", "DELETE")
        if writes and request.headers.get("x-hypercompute-owner") != TOKEN:
            return httpx.Response(403, json={"detail": "missing or wrong owner token"})
        route = (request.method, request.url.path)
        if route == ("POST", "/v1/chat/completions"):
            return self.chat(body)
        if request.url.path.startswith("/api/deployments/"):
            return self.deployment(request.method)
        handler = self.routes().get(route)
        return handler(request) if handler else httpx.Response(404, json={"detail": "Not Found"})

    def routes(self) -> dict[tuple[str, str], Any]:
        return {
            ("GET", "/api/owner/bootstrap"): lambda r: httpx.Response(200, json={"owner_token": TOKEN}),
            ("GET", "/api/health"): lambda r: httpx.Response(200, json={"status": "ok", "version": "0.1.0"}),
            ("GET", "/api/capacity/pools"): lambda r: (httpx.Response(200, json={"pools": []}) if self.capacity
                                                       else httpx.Response(404, json={"detail": "Not Found"})),
            ("GET", "/api/deployments"): lambda r: httpx.Response(200, json=[{"id": "e" * 32, "state": "ready"}]
                                                                  if self.active else [d for d in [self.dep] if d]),
            ("GET", "/api/nodes"): lambda r: httpx.Response(200, json=self.nodes()),
            ("GET", "/api/network"): lambda r: httpx.Response(200, json={"transport": "iroh", "peers": [
                {"node_id": "m" * 32, "connected": True, "path": {"mode": "relay"}}]}),
            ("GET", "/api/models"): lambda r: httpx.Response(200, json=[self.model()]),
            ("POST", "/api/plans"): lambda r: httpx.Response(200, json=_plan(self.feasible)),
            ("GET", "/api/capacity/explain"): lambda r: httpx.Response(200, json={"lines": ["best: CUDA0 + RPC0"]}),
            ("POST", "/api/deployments"): lambda r: self.create(r),
            ("GET", "/api/runs"): lambda r: (httpx.Response(500, json={"detail": "boom"}) if self.broken_runs else
                                             httpx.Response(200, json=[{"id": "run1", "deployment_id": DEP_ID},
                                                                       {"id": "other", "deployment_id": "x"}])),
        }

    def model(self) -> dict[str, Any]:
        return {"artifact": {"id": "big"}, "status": "downloaded" if self.downloaded else "not_downloaded",
                "verified_sha256": self.downloaded, "local_path": "C:/m/big.gguf", "downloaded_bytes": 0,
                "total_bytes": 40 * GIB, "gguf": {"n_layer": 48}}

    def nodes(self) -> list[dict[str, Any]]:
        return [{"id": "local", "display_name": "G14", "is_local": True, "state": "eligible", "inventory": {}},
                {"id": "m" * 32, "display_name": "Mac", "is_local": False, "state": "eligible",
                 "worker_cpu_seconds": self.cpu, "inventory": {"memory_pools": [], "engine": {"devices": []}}}]

    def create(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.dep = {"id": DEP_ID, "state": "planned", "plan": _plan(),
                        "idempotency_key": request.headers.get("idempotency-key")}
        if self.create_raises is not None:
            raise self.create_raises
        return httpx.Response(202, json=self.dep)

    def deployment(self, method: str) -> httpx.Response:
        with self.lock:
            assert self.dep is not None
            if method == "DELETE":
                self.dep = {**self.dep, "state": "stopped" if self.dep["state"] not in ("failed",) else "failed"}
                return httpx.Response(200, json=self.dep)
            self.polls += 1
            if self.dep["state"] in ("stopped", "failed"):
                return httpx.Response(200, json=self.dep)
            if self.stay_loading or self.polls < 3:
                self.dep = {**self.dep, "state": "loading", "pid": 4242, "progress": f"loading {self.polls}"}
            else:
                self.dep = {**self.dep, "state": self.final, "pid": 4242, "progress": None,
                            "error": None if self.final == "ready" else "engine exited",
                            "placement": {"offloaded": "49/49", "devices": []}, "participation_verified": True,
                            "participation_detail": "RPC0 holds 3.1 GiB",
                            "engine_args": ["-m", "x.gguf", "-ngl", "99", "--n-cpu-moe", "30", "-ot",
                                            "blk\\.4[0-7]\\.ffn_.*_exps=RPC0", "--no-repack", "-lm", "mmap",
                                            "--no-host", "--no-op-offload", "--cache-ram", "0", "--fit", "off",
                                            "--rpc", "127.0.0.1:52110", "-c", "4096", *self.slot_args()]}
            return httpx.Response(200, json=self.dep)

    def slot_args(self) -> list[str]:
        return ["-np", "1"] if self.parallel is None else ["-np", str(self.parallel), "-kvu"]

    def chat(self, body: dict[str, Any]) -> httpx.Response:
        if self.interrupt_chat:
            raise KeyboardInterrupt
        with self.lock:
            self.chats += 1
            self.cpu += 4.0
            if self.chats == self.interrupt_at_chat:
                raise KeyboardInterrupt
        if self.chat_error:
            return _sse([{"error": {"code": "interrupted", "message": "peer lost"}}])
        words = len(body["messages"][0]["content"].split())
        n = body["max_tokens"]
        timings = {"prompt_n": 12 + round(words * self.tokens_per_word), "prompt_ms": 400.0, "predicted_n": n,
                   "predicted_ms": 126.0 * (n - 1) / 63, "cache_n": 0, "predicted_per_second": 1.0}
        return _sse([{"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]},
                     {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
                     {"choices": [{"delta": {}, "finish_reason": "length"}], "timings": timings}])


def _sse(payloads: list[dict[str, Any]]) -> httpx.Response:
    text = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=text.encode())


class FakeProbes:
    def __init__(self, available: Any = 8 * GIB, drive_free: int = 40 * GIB) -> None:
        self.available, self.free, self.reads, self.lock = available, drive_free, 0, threading.Lock()

    def host_memory(self) -> dict[str, int | None]:
        avail = self.available() if callable(self.available) else self.available
        return {"available": avail, "commit_charge": 20 * GIB, "commit_limit": 40 * GIB, "system_cache": 3 * GIB}

    def disk_read_bytes(self) -> int:
        with self.lock:
            self.reads += 1 << 20
            return self.reads

    def drive_free(self) -> int:
        return self.free

    def process(self, pid: int) -> dict[str, int]:
        return {"rss": GIB, "private": 2 * GIB, "read_bytes": 5, "page_faults": 7}

    def gpu_used(self) -> int:
        return 10 * GIB


def _env() -> Any:
    return rb.Env(clock=time.perf_counter, sleep=lambda s: time.sleep(0.002), say=lambda m: None)


def _opts(tmp_path: Path, **kw: Any) -> Any:
    base = {"model": "big", "home": tmp_path / "home", "results_dir": tmp_path / "results", "poll_s": 0.0,
            "heartbeat_wait_s": 0.0, "sample_interval_s": 0.002}
    return rb.Options(**{**base, **kw})


def _windows(tmp_path: Path) -> Any:
    return hm.host_profile("win32", tmp_path / "home", {"SystemDrive": "C:"})


def _run(tmp_path: Path, gw: FakeGateway, probes: FakeProbes | None = None, profile: Any = None, env: Any = None,
         **kw: Any) -> tuple[dict[str, Any], str]:
    with gw.client() as client:
        path, status = rb.run(_opts(tmp_path, **kw), client, probes or FakeProbes(), env or _env(),
                              profile or _windows(tmp_path))
    text = path.read_text(encoding="utf-8")
    assert TOKEN not in text  # the owner token is never recorded
    return json.loads(text), status


def test_full_run_records_predictions_measurements_and_stops(tmp_path: Path) -> None:
    logs = tmp_path / "home" / "coordinator" / "logs"
    logs.mkdir(parents=True)
    (logs / f"llama-server-{DEP_ID}.log").write_text(
        "load_tensors:   CPU_Mapped model buffer size = 30000.50 MiB\n"
        "load_tensors:        CUDA0 model buffer size =  9000.25 MiB\n"
        "load_tensors: RPC0[127.0.0.1:52110] model buffer size = 3100.00 MiB\n", encoding="utf-8")
    gw = FakeGateway()
    doc, status = _run(tmp_path, gw)
    assert status == "ok" and doc["status"] == "ok" and doc["schema"] == rb.SCHEMA
    assert doc["predicted"]["plan"]["id"] == "p1" and doc["predicted"]["explain"]["lines"]
    chats = doc["measured"]["chats"]
    assert chats["summary"]["ok"] == 3 and chats["summary"]["failed"] == 0
    rep = chats["reps"][-1]
    assert rep["decode_tok_s"] == pytest.approx(63 / (126.0 / 1000))  # (predicted_n - 1) / predicted_ms
    assert abs(rep["prompt_n"] - 128) <= 6 and rep["reasoning_chars"] == 8 and rep["ttft_ms"] is not None
    assert rep["host"]["samples"] >= 0 and "disk_read_bytes_per_output_token" in rep
    placement = doc["measured"]["placement"]
    assert placement["participation_verified"] is True
    assert placement["engine"]["moe"] == {"cpu_moe": False, "n_cpu_moe": "30", "override_tensor": [
        "blk\\.4[0-7]\\.ffn_.*_exps=RPC0"], "no_repack": True, "no_host": True, "no_op_offload": True,
        "load_mode": "mmap", "fit": "off"}
    assert placement["engine"]["flags"]["--cache-ram"] == "0" and placement["engine"]["flags"]["--ctx-size"] == "4096"
    assert doc["measured"]["engine_log"]["file_backed_host_mib"] == 30000.5
    assert doc["measured"]["peer_worker_cpu"]["nodes"]["m" * 32]["delta_s"] > 0
    assert [r["id"] for r in doc["measured"]["gateway_runs"]["runs"]] == ["run1"]
    assert doc["comparison"]["predicted_decode_ms"] == 200.0
    assert doc["comparison"]["measured_over_predicted"] == pytest.approx(2.0 / 200.0, rel=1e-3)
    assert doc["measured"]["host"]["samples"] and doc["measured"]["host"]["guard_tripped"] is None
    assert len(gw.calls("DELETE", f"/api/deployments/{DEP_ID}")) == 1 and doc["stop"]["state"] == "stopped"


def test_chat_requests_fix_output_length_and_defeat_prefix_reuse(tmp_path: Path) -> None:
    gw = FakeGateway()
    _run(tmp_path, gw, reps=2)
    bodies = [b for _, _, _, b in gw.calls("POST", "/v1/chat/completions")]
    assert bodies and all(b["ignore_eos"] is True and b["cache_prompt"] is False and b["max_tokens"] == 64
                          and b["stream"] is True for b in bodies)
    firsts = [b["messages"][0]["content"].split(".")[0] for b in bodies]
    assert len(set(firsts)) == len(firsts)  # a unique prefix per chat
    writes = [h for m, _, h, _ in gw.requests if m in ("POST", "DELETE")]
    assert all(h.get("x-hypercompute-owner") == TOKEN for h in writes)


def test_a_failed_read_after_the_chats_keeps_the_measurements(tmp_path: Path) -> None:
    doc, status = _run(tmp_path, FakeGateway(broken_runs=True))
    assert status == "ok" and doc["measured"]["chats"]["summary"]["ok"] == 3
    assert doc["measured"]["gateway_runs"]["runs"] == []


def test_keep_leaves_the_deployment_running(tmp_path: Path) -> None:
    gw = FakeGateway()
    doc, status = _run(tmp_path, gw, keep=True)
    assert status == "ok" and doc["kept"] is True and doc["stop"] is None
    assert not gw.calls("DELETE", f"/api/deployments/{DEP_ID}")


@pytest.mark.parametrize(("gateway", "needle"), [
    (FakeGateway(capacity=False), "predates the capacity code"),
    (FakeGateway(active=True), "never stops a deployment it did not create"),
    (FakeGateway(downloaded=False), "not_downloaded"),
])
def test_refusals_create_nothing(tmp_path: Path, gateway: FakeGateway, needle: str) -> None:
    doc, status = _run(tmp_path, gateway)
    assert status == "refused" and needle in doc["error"]["message"] and doc["error"]["fix"]
    assert not gateway.calls("POST", "/api/deployments") and not gateway.calls("POST", "/api/plans")


def test_infeasible_plan_and_plan_only_deploy_nothing(tmp_path: Path) -> None:
    gw = FakeGateway(feasible=False)
    doc, status = _run(tmp_path, gw)
    assert status == "infeasible" and doc["predicted"]["plan"]["reasons"] == ["does not fit"]
    gw2 = FakeGateway()
    doc2, status2 = _run(tmp_path, gw2, plan_only=True)
    assert status2 == "plan_only" and doc2["predicted"]["explain"]
    assert not gw.calls("POST", "/api/deployments") and not gw2.calls("POST", "/api/deployments")


def test_load_failure_records_placement_and_skips_chats(tmp_path: Path) -> None:
    gw = FakeGateway(final="failed")
    doc, status = _run(tmp_path, gw)
    assert status == "load_failed" and doc["measured"]["placement"]["error"] == "engine exited"
    assert not gw.calls("POST", "/v1/chat/completions")


def test_load_timeout_stops_the_deployment(tmp_path: Path) -> None:
    gw = FakeGateway(stay_loading=True)
    doc, status = _run(tmp_path, gw, load_timeout_s=0.05)
    assert status == "load_timeout" and "still loading" in doc["measured"]["load"]["error"]
    assert gw.calls("DELETE", f"/api/deployments/{DEP_ID}")


def test_guard_trip_during_load_stops_the_deployment(tmp_path: Path) -> None:
    gw = FakeGateway(stay_loading=True)
    probes = FakeProbes(available=lambda: GIB // 4 if gw.polls > 0 else 8 * GIB)  # low once loading is polled
    doc, status = _run(tmp_path, gw, probes)
    assert status == "aborted_by_guard" and "available RAM" in doc["measured"]["host"]["guard_tripped"]
    assert doc["measured"]["host"]["guard_stop"]["state"] == "stopped"
    assert not gw.calls("POST", "/v1/chat/completions")


def test_guard_trip_before_create_starts_nothing(tmp_path: Path) -> None:
    gw = FakeGateway()
    doc, status = _run(tmp_path, gw, FakeProbes(drive_free=GIB))
    assert status == "aborted_by_guard" and "pagefile" in doc["measured"]["host"]["guard_tripped"]
    assert not gw.calls("POST", "/api/deployments")


def test_stream_error_events_fail_the_reps(tmp_path: Path) -> None:
    gw = FakeGateway(chat_error=True)
    doc, status = _run(tmp_path, gw)
    assert status == "chat_failed" and doc["measured"]["chats"]["summary"]["failed"] == 3
    assert doc["measured"]["chats"]["reps"][-1]["error"] == "peer lost"


def test_ctrl_c_still_stops_the_deployment_and_writes_evidence(tmp_path: Path) -> None:
    gw = FakeGateway(interrupt_chat=True)
    doc, status = _run(tmp_path, gw)
    assert status == "interrupted" and doc["error"]["type"] == "KeyboardInterrupt"
    assert gw.calls("DELETE", f"/api/deployments/{DEP_ID}")


def test_comparison_falls_back_to_moe_estimates() -> None:
    chats = {"summary": {"decode_ms_per_token": {"median": 300.0},
                         "disk_read_bytes_per_output_token": {"median": 2.0e8}}}
    moe_plan = {"moe": {"decode_ms_est": 150.0, "ssd_bytes_per_token_est": 1.0e8, "miss_rate_est": 0.4}}
    out = rb.comparison(moe_plan, chats, 12.5)
    assert out["predicted_decode_ms"] == 150.0 and out["predicted_decode_source"] == "plan.moe (estimate)"
    assert out["measured_over_predicted"] == 2.0 and out["predicted_moe_ssd_bytes_per_token_est"] == 1.0e8
    scored = rb.comparison({"prediction": {"decode_ms": 100.0}, **moe_plan}, chats, None)
    assert scored["predicted_decode_ms"] == 100.0 and scored["predicted_decode_source"] == "plan.prediction"
    assert rb.comparison({}, {}, None)["predicted_decode_source"] is None


def test_engine_flags_canonical_names_and_switches() -> None:
    out = rb.engine_flags(["-cmoe", "-ncmoe", "12", "--override-tensor", "a=CPU", "-ot", "b=CPU", "-nr", "-dev",
                           "CUDA0,RPC0", "-ts", "40,8", "--load-mode", "none", "--no-host", "-c"])
    assert out["flags"]["--device"] == "CUDA0,RPC0" and out["flags"]["--tensor-split"] == "40,8"
    assert out["flags"]["--no-host"] is True and "--ctx-size" not in out["flags"]  # a trailing value flag is ignored
    assert out["moe"] == {"cpu_moe": True, "n_cpu_moe": "12", "override_tensor": ["a=CPU", "b=CPU"],
                          "no_repack": True, "no_host": True, "no_op_offload": False, "load_mode": "none",
                          "fit": None}


def test_log_evidence_separates_file_backed_and_private_host_buffers(tmp_path: Path) -> None:
    log = tmp_path / "e.log"
    log.write_text("x\nload_tensors: CPU_Mapped model buffer size = 100.5 MiB\n"
                   "load_tensors: CPU_Mapped model buffer size = 50.0 MiB\n"
                   "load_tensors: CPU_REPACK model buffer size = 10.0 MiB\n"
                   "load_tensors: CUDA_Host model buffer size = 2.5 MiB\nunrelated\n", encoding="utf-8")
    ev = rb.log_evidence(log)
    assert ev["file_backed_host_mib"] == 150.5 and ev["private_host_mib"] == 12.5 and len(ev["lines"]) == 4
    assert rb.log_evidence(tmp_path / "missing.log")["available"] is False


def test_next_words_solves_overhead_and_rate() -> None:
    assert rb.next_words([(100, 112)], 128) == 114  # proportional from one point
    assert rb.next_words([(96, 108), (114, 126)], 128) == 116  # 12 overhead + 1 token per word
    assert rb.next_words([(50, 0)], 128) == 50


def test_read_stream_ttft_on_reasoning_and_missing_timings() -> None:
    ticks = iter([0.5, 0.75, 1.0])
    raw = b"data: {\"choices\": [{\"delta\": {\"reasoning_content\": \"hm\"}}]}\n\ndata: [DONE]\n\n"
    out = rb.read_stream([raw[:20], raw[20:]], 0.0, lambda: next(ticks))
    assert out["ttft_s"] == 0.5 and out["reasoning"] == 2 and out["done"] and out["timings"] is None


def test_window_brackets_counters_and_takes_extremes() -> None:
    samples = [hm.Sample(t=0.0, disk_read_bytes=100, available=9), hm.Sample(t=1.0, disk_read_bytes=150, available=4),
               hm.Sample(t=2.0, disk_read_bytes=400, available=6)]
    w = hm.window(samples, 0.5, 1.5)
    assert w["disk_read_bytes"] == 300 and w["available_min"] == 4 and w["samples"] == 1
    assert hm.counter_delta(samples, "engine_read_bytes", 0, 2) is None


def test_monitor_guard_trips_once_and_attaches_the_engine() -> None:
    trips: list[str] = []
    mon = hm.Monitor(FakeProbes(available=GIB // 2), interval_s=1.0, gpu_every=2, min_available=GIB,
                     min_drive_free=GIB, on_trip=trips.append)
    mon.attach(99)
    first, second = mon.sample_once(), mon.sample_once()
    assert len(trips) == 1 and mon.tripped == trips[0] and mon.marks[0][0].startswith("guard:")
    assert first.gpu_used == 10 * GIB and second.gpu_used is None and first.engine_private == 2 * GIB


def test_parse_args_validates_the_workload() -> None:
    opts = rb.parse_args(["--model", "big", "--node", "a", "--node", "b", "--strategy", "fastest_fit"])
    assert opts.nodes == ("a", "b") and opts.port == 8001 and opts.prompt_tokens == 128 and opts.output_tokens == 64
    assert opts.max_commit_growth_gib == 0.25
    assert rb.parse_args(["--model", "m", "--max-commit-growth-gib", "1"]).max_commit_growth_gib == 1
    with pytest.raises(SystemExit):
        rb.parse_args(["--model", "big", "--ctx", "256", "--prompt-tokens", "250"])


# ----------------------------------------------------------------------------- review regressions


def test_counter_delta_falls_back_to_the_nearest_sample_inside_the_window() -> None:
    samples = [hm.Sample(t=0.0, disk_read_bytes=100), hm.Sample(t=1.0, disk_read_bytes=150),
               hm.Sample(t=2.0, disk_read_bytes=400)]
    # the window ends after the last sample (a chat that just finished): the end is the last sample, not the first
    assert hm.counter_delta(samples, "disk_read_bytes", 0.5, 3.0) == 300
    # the window starts before the first sample: the start is the first sample, not the last
    assert hm.counter_delta(samples, "disk_read_bytes", -1.0, 1.5) == 300
    assert hm.counter_delta(samples, "disk_read_bytes", 5.0, 6.0) == 0  # only one sample brackets both ends


def test_missing_model_fix_names_the_gateway_download_not_the_cli(tmp_path: Path) -> None:
    doc, status = _run(tmp_path, FakeGateway(downloaded=False))
    # `hypercompute models pull` downloads into the runner shell's HYPERCOMPUTE_HOME, not the gateway's
    assert status == "refused" and "models pull" not in doc["error"]["fix"]
    assert "/api/models/big/download" in doc["error"]["fix"]
    assert "models pull" not in (rb.__doc__ or "")


@pytest.mark.parametrize("lost", [httpx.ReadTimeout("reply lost"), KeyboardInterrupt()])
def test_a_create_whose_reply_is_lost_is_found_by_its_key_and_stopped(tmp_path: Path, lost: BaseException) -> None:
    gw = FakeGateway(create_raises=lost)
    doc, status = _run(tmp_path, gw)
    assert status in ("error", "interrupted") and doc["deployment_id"] == DEP_ID
    assert gw.calls("DELETE", f"/api/deployments/{DEP_ID}") and doc["stop"]["state"] == "stopped"
    key = gw.calls("POST", "/api/deployments")[0][2]["idempotency-key"]
    assert gw.dep is not None and gw.dep["idempotency_key"] == key


def test_ctrl_c_mid_chats_keeps_the_finished_reps_load_and_placement(tmp_path: Path) -> None:
    gw = FakeGateway(interrupt_at_chat=4)  # two warmups size the prompt, one rep, then Ctrl+C on the next
    doc, status = _run(tmp_path, gw)
    assert status == "interrupted" and gw.calls("DELETE", f"/api/deployments/{DEP_ID}")
    chats = doc["measured"]["chats"]
    assert chats["interrupted"] is True and chats["summary"]["ok"] >= 1
    assert doc["measured"]["load"]["seconds"] is not None and doc["measured"]["placement"]["state"] == "ready"


def _result(**kw: Any) -> Any:
    timings = hm_timings(prompt_n=128, prompt_ms=900.0, predicted_n=4, predicted_ms=3000.0)
    return rb.ChatResult(**{"label": "rep0", "ok": True, "timings": timings, **kw})


def hm_timings(**kw: Any) -> Any:
    from hypercompute.capacity.benchlib import Timings
    return Timings(cache_n=0, **kw)


def test_decode_disk_reads_exclude_the_prefill_reads() -> None:
    samples = [hm.Sample(t=float(t), disk_read_bytes=v)
               for t, v in ((0, 0), (1, 5000), (2, 5000), (3, 5100), (4, 5200), (5, 5300))]
    rep = _result(ttft_ms=1500.0, t_start=0.0, t_end=5.0)
    chats = rb.chat_section([rep], [], samples)
    row = chats["reps"][0]
    assert row["disk_read_bytes_per_output_token"] == 5300 / 4  # the whole chat, prefill included
    assert row["decode_disk_read_bytes_per_token"] == 300 / 3  # first token to the end, per decode step
    out = rb.comparison({"moe": {"ssd_bytes_per_token_est": 120}}, chats, None)
    assert out["measured_decode_disk_read_bytes_per_token_median"] == 100.0


def test_gateway_repr_hides_the_owner_token() -> None:
    with httpx.Client() as client:
        assert TOKEN not in repr(rb.Gateway(client, TOKEN))


def test_guard_trips_when_the_commit_limit_grows() -> None:
    """A growing commit limit means the pagefile is growing on the system drive (private memory past RAM): trip at
    once, before the drive-free floor, whatever the available RAM (file-backed experts can take that low safely)."""
    limits = iter([40 * GIB, 40 * GIB, 40 * GIB + (1 << 30)])
    probes = FakeProbes()
    probes.host_memory = lambda: {"available": 8 * GIB, "commit_charge": 39 * GIB,  # type: ignore[method-assign]
                                  "commit_limit": next(limits), "system_cache": GIB}
    trips: list[str] = []
    mon = hm.Monitor(probes, interval_s=1.0, gpu_every=1, min_available=GIB, min_drive_free=GIB,
                     on_trip=trips.append, max_commit_growth=256 << 20)
    mon.sample_once()
    mon.sample_once()
    assert not trips
    mon.sample_once()
    assert len(trips) == 1 and "commit limit grew" in trips[0] and "pagefile" in trips[0]


# ----------------------------------------------------------------------------- macOS host (simulated)

MacMem = namedtuple("MacMem", "rss vms pfaults pageins")  # psutil's pmem on macOS


class FakeMacProbes(FakeProbes):
    """What MacProbes reports: no commit, file cache, private or read bytes, or GPU memory; swap used instead."""

    def __init__(self, swap: Any = GIB, **kw: Any) -> None:
        super().__init__(**kw)
        self.swap = swap

    def host_memory(self) -> dict[str, int | None]:
        avail = self.available() if callable(self.available) else self.available
        swap = self.swap() if callable(self.swap) else self.swap
        return {"available": avail, "commit_charge": None, "commit_limit": None, "system_cache": None,
                "swap_used": swap}

    def process(self, pid: int) -> dict[str, int]:
        return {"rss": GIB, "page_faults": 7}

    def gpu_used(self) -> None:
        return None


class MacProcess:
    """psutil.Process as it is on macOS: memory_info has pfaults and pageins, and there is no io_counters."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def memory_info(self) -> Any:
        return MacMem(rss=3 * GIB, vms=420 * GIB, pfaults=1234, pageins=56)


def _mac(tmp_path: Path) -> Any:
    return hm.host_profile("darwin", tmp_path / "home")


PAGE = 16384  # Apple silicon kernel page size, as vm_stat's header prints it


def vm_stat_output(free: int, active: int, inactive: int, file_backed: int, anonymous: int, wired: int) -> str:
    """vm_stat's real layout (system_cmds vm_stat.c: `printf("%-30s %16llu.\\n", ...)`), from byte counts."""
    rows = [("Pages free:", free), ("Pages active:", active), ("Pages inactive:", inactive),
            ("Pages speculative:", 0), ("Pages throttled:", 0), ("Pages wired down:", wired),
            ("Pages purgeable:", 0), ('"Translation faults":', 123456789 * PAGE), ("Pages copy-on-write:", 0),
            ("File-backed pages:", file_backed), ("Anonymous pages:", anonymous),
            ("Pages occupied by compressor:", 0), ("Swapins:", 0), ("Swapouts:", 0)]
    body = "".join(f"{label:<30} {n // PAGE:>16}.\n" for label, n in rows)
    return f"Mach Virtual Memory Statistics: (page size of {PAGE} bytes)\n{body}"


# 16 GiB Mac mid-run: the mapped experts are 8 GiB of active file-backed pages; free + inactive is only 0.5 GiB
MAPPED = vm_stat_output(free=GIB // 4, active=9 * GIB, inactive=GIB // 4, file_backed=8 * GIB, anonymous=GIB + GIB // 2,
                        wired=6 * GIB)
# the same Mac with private memory past RAM: little file cache left, anonymous pages everywhere
EXHAUSTED = vm_stat_output(free=GIB // 4, active=9 * GIB, inactive=GIB // 4, file_backed=GIB // 4,
                           anonymous=9 * GIB, wired=6 * GIB)


def _fake_mac_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    def virtual_memory() -> None:
        raise AssertionError("psutil's macOS available (free + inactive) must not feed the floor")

    monkeypatch.setattr(hm.psutil, "virtual_memory", virtual_memory)
    monkeypatch.setattr(hm.psutil, "swap_memory", lambda: SimpleNamespace(used=GIB, total=2 * GIB))
    monkeypatch.setattr(hm.psutil, "Process", MacProcess)


def test_mac_probes_read_vm_stat_and_never_call_io_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_mac_psutil(monkeypatch)
    probes = hm.MacProbes("/Users/azim/.hypercompute/models", vm_stat=lambda: MAPPED)
    assert probes.host_memory() == {"available": GIB // 4 + 8 * GIB, "commit_charge": None, "commit_limit": None,
                                    "system_cache": 8 * GIB, "swap_used": GIB}
    # vms (420 GiB of address space, the mapped GGUF included) is not reported as private memory
    assert probes.process(42) == {"rss": 3 * GIB, "page_faults": 1234}
    assert probes.gpu_used() is None
    # the Windows probes call Process.io_counters, which psutil does not define on macOS
    with pytest.raises(AttributeError):
        hm.SystemProbes("/").process(42)


def test_mac_probes_turn_psutil_failures_into_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise OSError("sysctl failed")

    def gone(pid: int) -> None:
        raise hm.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(hm.psutil, "swap_memory", boom)
    monkeypatch.setattr(hm.psutil, "Process", gone)
    probes = hm.MacProbes("/", vm_stat=lambda: MAPPED)
    assert probes.host_memory()["swap_used"] is None and probes.host_memory()["available"] == GIB // 4 + 8 * GIB
    assert probes.process(7) is None


def test_parse_vm_stat_reads_the_real_layout() -> None:
    pages = hm.parse_vm_stat(MAPPED)
    assert pages["page size"] == PAGE and pages["Pages free"] == GIB // 4 // PAGE
    assert pages["File-backed pages"] == 8 * GIB // PAGE and pages["Translation faults"] == 123456789
    assert "Mach Virtual Memory Statistics" not in pages
    with pytest.raises(ValueError):
        hm.parse_vm_stat("Pages free: 12.\n")  # no header, so no page size


def test_mapped_experts_do_not_trip_the_mac_floor_but_private_memory_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil's macOS available (free + inactive) reads 0.5 GiB on MAPPED, under a 1 GiB floor, although 8 GiB of
    it is file cache macOS drops at once; free + file-backed pages reads 8.25 GiB. EXHAUSTED is the real failure."""
    _fake_mac_psutil(monkeypatch)
    snapshots = iter([MAPPED, EXHAUSTED])
    trips: list[str] = []
    mon = hm.Monitor(hm.MacProbes("/", vm_stat=lambda: next(snapshots)), interval_s=1.0, gpu_every=1,
                     min_available=GIB, min_drive_free=0, on_trip=trips.append)
    assert mon.sample_once().available == 8 * GIB + GIB // 4 and not trips
    assert mon.sample_once().system_cache == GIB // 4
    assert trips == ["available RAM 0.50 GiB below 1.00 GiB"]


def _raises(exc: Exception) -> Any:
    def reader() -> str:
        raise exc

    return reader


@pytest.mark.parametrize("reader", [
    _raises(FileNotFoundError(2, "No such file or directory")), _raises(hm.subprocess.CalledProcessError(1, "x")),
    _raises(hm.subprocess.TimeoutExpired("vm_stat", 5)), lambda: "garbage",
    lambda: MAPPED.replace("File-backed pages:", "Something else:")],
    ids=["missing", "failed", "timeout", "no-header", "no-file-backed-line"])
def test_a_failed_vm_stat_leaves_the_mac_floor_inactive_never_psutil(monkeypatch: pytest.MonkeyPatch,
                                                                     reader: Any) -> None:
    _fake_mac_psutil(monkeypatch)  # psutil's available would raise here: no silent fallback to it
    mon = hm.Monitor(hm.MacProbes("/", vm_stat=reader), interval_s=1.0, gpu_every=1, min_available=GIB,
                     min_drive_free=0, on_trip=lambda r: None)
    s = mon.sample_once()
    assert s.available is None and s.system_cache is None and s.swap_used == GIB and mon.error is None
    assert not mon.tripped and mon.guard_status()["min_available"] == {
        "limit_bytes": GIB, "active": False, "inactive_because": "this host reported no available reading"}


def test_vm_stat_text_runs_the_binary_without_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def run(argv: list[str], **kw: Any) -> Any:
        seen.update(argv=argv, **kw)
        return SimpleNamespace(stdout=MAPPED)

    monkeypatch.setattr(hm.subprocess, "run", run)
    assert hm.vm_stat_text() == MAPPED
    assert seen["argv"] == ["/usr/bin/vm_stat"] and seen["check"] and seen["timeout"] <= 5
    assert hm.MacProbes("/").vm_stat is hm.vm_stat_text


def test_the_monitor_keeps_sampling_a_mac_engine_after_attach(monkeypatch: pytest.MonkeyPatch,
                                                              tmp_path: Path) -> None:
    """With SystemProbes every sample after attach raised (no io_counters on macOS): no samples, no guard."""
    _fake_mac_psutil(monkeypatch)
    kw = {"interval_s": 1.0, "gpu_every": 1, "min_available": 0, "min_drive_free": GIB, "on_trip": lambda r: None}
    windows = hm.Monitor(hm.SystemProbes(str(tmp_path)), **kw)
    windows.attach(42)
    windows._safe_sample()
    assert not windows.samples and "AttributeError" in (windows.error or "")
    mac = hm.Monitor(hm.MacProbes(str(tmp_path), vm_stat=lambda: MAPPED), **kw)
    mac.attach(42)
    s = mac.sample_once()
    assert mac.error is None and s.engine_rss == 3 * GIB and s.engine_page_faults == 1234
    assert s.engine_private is None and s.engine_read_bytes is None and s.gpu_used is None
    assert s.swap_used == GIB and s.commit_limit is None and s.drive_free is not None


def test_swap_growth_trips_the_guard_and_commit_is_inactive_on_a_mac() -> None:
    swaps = iter([GIB, 2 * GIB, 3 * GIB + GIB // 2])
    probes = FakeMacProbes(swap=lambda: next(swaps))
    trips: list[str] = []
    mon = hm.Monitor(probes, interval_s=1.0, gpu_every=1, min_available=0, min_drive_free=GIB,
                     on_trip=trips.append, max_swap_growth=2 * GIB)
    mon.sample_once()
    mon.sample_once()
    assert not trips  # 1 GiB of growth is under the 2 GiB cap
    mon.sample_once()
    assert len(trips) == 1 and "swap used grew 2.50 GiB to 3.50 GiB" in trips[0]
    status = mon.guard_status()
    assert status["max_swap_growth"] == {"limit_bytes": 2 * GIB, "active": True, "inactive_because": None}
    assert status["max_commit_growth"]["active"] is False
    assert "no commit_limit reading" in status["max_commit_growth"]["inactive_because"]
    assert status["min_available"] == {"limit_bytes": 0, "active": False, "inactive_because": "turned off"}


def test_the_drive_floor_names_the_models_volume_on_a_mac(tmp_path: Path) -> None:
    profile = _mac(tmp_path)
    trips: list[str] = []
    mon = hm.Monitor(FakeMacProbes(drive_free=GIB), interval_s=1.0, gpu_every=1, min_available=0,
                     min_drive_free=5 * GIB, on_trip=trips.append, drive_label=profile.drive_label,
                     drive_note=profile.drive_note)
    mon.sample_once()
    assert trips == ["models volume free 1.00 GiB below 5.00 GiB (the models and the swap files fill it)"]


def test_a_mac_run_with_the_real_mac_probes_keeps_the_ram_floor_on_and_untripped(monkeypatch: pytest.MonkeyPatch,
                                                                                 tmp_path: Path) -> None:
    """MacProbes end to end (vm_stat and psutil faked): mapped experts as active file pages do not trip the default
    1 GiB floor, and run.json carries available and the file cache."""
    _fake_mac_psutil(monkeypatch)
    doc, status = _run(tmp_path, FakeGateway(), hm.MacProbes(str(tmp_path), vm_stat=lambda: MAPPED),
                       profile=_mac(tmp_path))
    host = doc["measured"]["host"]
    assert status == "ok" and host["guard_tripped"] is None and host["error"] is None
    assert host["guards"]["min_available"] == {"limit_bytes": GIB, "active": True, "inactive_because": None}
    assert host["whole_run"]["available_min"] == 8 * GIB + GIB // 4
    assert host["whole_run"]["system_cache_peak"] == 8 * GIB and host["whole_run"]["engine_rss_peak"] == 3 * GIB


def test_a_mac_left_with_3_gib_after_the_download_never_starts_loading(tmp_path: Path) -> None:
    """The first runbook's 62 GiB prerequisite: 59.03 GiB of model left about 3 GiB, under the 5 GiB floor."""
    gw = FakeGateway()
    doc, status = _run(tmp_path, gw, FakeMacProbes(drive_free=3 * GIB), profile=_mac(tmp_path),
                       min_available_gib=0.0)
    assert status == "aborted_by_guard" and "models volume free 3.00 GiB" in doc["measured"]["host"]["guard_tripped"]
    assert not gw.calls("POST", "/api/deployments")


def test_host_profile_per_platform(tmp_path: Path) -> None:
    win = hm.host_profile("win32", tmp_path, {"SystemDrive": "D:"})
    assert (win.platform, win.drive, win.drive_label, win.source) == ("windows", "D:" + os.sep, "system drive",
                                                                      hm.WINDOWS_SOURCE)
    # the models folder does not exist yet: the nearest existing parent is on the same volume
    assert hm.host_profile("darwin", tmp_path / "home").drive == str(tmp_path)
    (tmp_path / "home" / "models").mkdir(parents=True)
    mac = _mac(tmp_path)
    assert (mac.platform, mac.drive, mac.drive_label) == ("macos", str(tmp_path / "home" / "models"),
                                                          "models volume")
    assert "macOS" in mac.source and "sudo" in mac.source
    assert hm.host_profile("linux", tmp_path).drive == "/"
    assert type(hm.probes_for(mac)) is hm.MacProbes and type(hm.probes_for(win)) is hm.SystemProbes


def test_a_mac_host_run_records_platform_guards_and_metal_buffers(tmp_path: Path) -> None:
    logs = tmp_path / "home" / "coordinator" / "logs"
    logs.mkdir(parents=True)
    (logs / f"llama-server-{DEP_ID}.log").write_text(
        "load_tensors:   CPU_Mapped model buffer size = 30000.50 MiB\n"
        "load_tensors:  MTL0_Mapped model buffer size =  6000.25 MiB\n"
        "load_tensors:  MTL0_Mapped model buffer size =  1000.00 MiB\n"
        "load_tensors:         MTL0 model buffer size =   512.00 MiB\n"
        "llama_kv_cache:       MTL0 KV buffer size =   160.00 MiB\n", encoding="utf-8")
    doc, status = _run(tmp_path, FakeGateway(), FakeMacProbes(available=GIB // 2), profile=_mac(tmp_path),
                       min_available_gib=0.0)
    host = doc["measured"]["host"]
    assert status == "ok" and host["platform"] == "macos" and host["source"] == hm.MACOS_SOURCE
    guards = host["guards"]
    assert guards["max_swap_growth"]["active"] and guards["max_swap_growth"]["limit_bytes"] == 2 * GIB
    assert guards["min_drive_free"]["active"] and guards["min_drive_free"]["path"] == str(tmp_path / "home")
    assert not guards["max_commit_growth"]["active"] and not guards["min_available"]["active"]
    whole = host["whole_run"]
    assert whole["swap_used_peak"] == GIB and whole["commit_limit_peak"] is None and whole["gpu_used_peak"] is None
    assert whole["engine_private_peak"] is None and whole["engine_rss_peak"] == GIB
    ev = doc["measured"]["engine_log"]
    assert ev["file_backed_host_mib"] == 30000.5 and ev["metal_mapped_mib"] == 7000.25
    assert ev["metal_private_mib"] == 512.0 and ev["private_host_mib"] == 0
    assert doc["requested"]["max_swap_growth_gib"] == 2.0


def test_swap_growth_on_a_mac_host_stops_the_deployment(tmp_path: Path) -> None:
    gw = FakeGateway(stay_loading=True)
    probes = FakeMacProbes(swap=lambda: 4 * GIB if gw.polls > 0 else GIB)  # swap grows once loading is polled
    doc, status = _run(tmp_path, gw, probes, profile=_mac(tmp_path), min_available_gib=0.0)
    assert status == "aborted_by_guard" and "swap used grew" in doc["measured"]["host"]["guard_tripped"]
    assert doc["measured"]["host"]["guard_stop"]["state"] == "stopped"
    assert not gw.calls("POST", "/v1/chat/completions")


def test_a_windows_run_keeps_its_source_and_guards(tmp_path: Path) -> None:
    doc, status = _run(tmp_path, FakeGateway())
    host = doc["measured"]["host"]
    assert status == "ok" and host["platform"] == "windows"
    assert host["source"] == ("hostmon: GetPerformanceInfo (available, commit, file cache), psutil disk_io_counters "
                              "(system-wide physical reads, every process), shutil.disk_usage (system drive), psutil "
                              "engine process counters (ReadFile bytes exclude mmap page-fault reads), nvidia-smi (all "
                              "GPUs)")
    guards = host["guards"]
    assert guards["max_commit_growth"] == {"limit_bytes": GIB // 4, "active": True, "inactive_because": None}
    assert guards["min_available"]["active"] and guards["min_drive_free"]["path"] == "C:" + os.sep
    assert not guards["max_swap_growth"]["active"]  # Windows probes report no swap; the commit guard covers it
    assert host["whole_run"]["swap_used_peak"] is None


@pytest.mark.parametrize(("platform", "probe_type", "drive"), [
    ("darwin", "MacProbes", "models"), ("win32", "SystemProbes", "C:" + os.sep)])
def test_main_picks_the_probes_for_the_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str,
                                                probe_type: str, drive: str) -> None:
    (tmp_path / "home" / "models").mkdir(parents=True)
    seen: dict[str, Any] = {}

    def fake_run(opts: Any, client: Any, probes: Any, env: Any = None, profile: Any = None) -> tuple[Path, str]:
        seen.update(opts=opts, probes=probes, profile=profile)
        return tmp_path / "run.json", "ok"

    monkeypatch.setattr(rb, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(rb, "run", fake_run)
    monkeypatch.setenv("SystemDrive", "C:")
    assert rb.main(["--model", "big", "--home", str(tmp_path / "home"), "--port", "8000"]) == 0
    assert type(seen["probes"]).__name__ == probe_type and seen["probes"].drive == seen["profile"].drive
    assert seen["profile"].drive.endswith(drive) and seen["opts"].port == 8000


def test_parse_args_swap_growth() -> None:
    assert rb.parse_args(["--model", "m"]).max_swap_growth_gib == 2.0
    assert rb.parse_args(["--model", "m", "--max-swap-growth-gib", "0.5"]).max_swap_growth_gib == 0.5


# ----------------------------------------------------------------------------- the Mac host runbook

RUNBOOK = ROOT / "docs" / "research" / "mac-host-runbook.md"


def _runbook_commands() -> list[list[str]]:
    """Every `uv run ...` line of the runbook's sh blocks, continuation lines joined, pipes and comments cut."""
    blocks = re.findall(r"```sh\n(.*?)```", RUNBOOK.read_text(encoding="utf-8"), re.S)
    lines = "\n".join(blocks).replace("\\\n", " ").splitlines()
    argvs = (shlex.split(line.split("|")[0], comments=True) for line in lines)
    return [argv[2:] for argv in argvs if argv[:2] == ["uv", "run"]]


def test_every_runbook_command_parses_with_the_real_argparse() -> None:
    from hypercompute.cli import build_parser

    cmds = _runbook_commands()
    cli = [c[1:] for c in cmds if c[0] == "hypercompute"]
    big = [c[2:] for c in cmds if c[:2] == ["python", "scripts/capacity/run_big.py"]]
    assert len(cli) >= 12 and len(big) == 3 and len(cli) + len(big) == len(cmds)
    for argv in cli:
        build_parser().parse_args(argv)  # argparse exits on an unknown command or flag
    runs = [rb.parse_args(argv) for argv in big]
    assert all(o.model == "gpt-oss-120b-mxfp4" and o.port == 8000 for o in runs)
    full = [o for o in runs if not o.plan_only]
    assert {o.strategy for o in full} == {"all_selected", "local_only"}
    assert all(o.min_available_gib == 0 and o.min_drive_free_gib == 5 and o.max_swap_growth_gib == 2 for o in full)


def test_the_runbook_disk_prerequisite_leaves_the_drive_floor_after_the_download() -> None:
    """62 GiB free before a 59.03 GiB download left 3 GiB, under the 5 GiB models-volume floor: run_big stopped before
    loading anything (aborted_by_guard)."""
    from hypercompute.services.models import load_manifest

    art = next(a for a in load_manifest(ROOT / "configs" / "models.example.yaml") if a.id == "gpt-oss-120b-mxfp4")
    need = re.search(r'"Avail" must be at least (\d+)Gi before the model download', RUNBOOK.read_text("utf-8"))
    assert need is not None
    left = int(need.group(1)) * GIB - sum(f.size_bytes for f in art.files)
    assert left >= rb.Options(model="m").min_drive_free_gib * GIB + 4 * GIB  # the floor plus room for swap and logs


# ----------------------------------------------------------------------------- --seed-peers

MAC_A, MAC_B = "a" * 32, "b" * 32
_SEED_ROUTE = re.compile(r"/api/peers/([0-9a-f]{32})/seed(/cancel)?")


def _seed_plan(pid: str = "p1", b_layers: list[int] | None = None) -> dict[str, Any]:
    return {**_plan(), "id": pid, "assignments": [
        {"node_id": MAC_A, "device": "RPC0", "layers": [0, 1, 2]},
        {"node_id": MAC_B, "device": "RPC1", "layers": b_layers or [3, 4]},
        {"node_id": "local", "device": "CUDA0", "layers": [5, 6, 7]}]}


class SeedGateway(FakeGateway):
    """FakeGateway plus the managed-peer routes. After `polls` GETs of /api/peers each peer's seed reaches
    outcomes[node]: "done", "failed" (the peer's job failed), "host_failed" (the host's job failed) or "stuck"
    (never ends). Nodes in `refuse` answer the seed POST with a 409, as a peer without --max-disk does."""

    def __init__(self, outcomes: dict[str, str], *, refuse: frozenset[str] = frozenset(), polls: int = 2,
                 moved_plan: bool = False, broken_peers: bool = False, interrupt_peers: bool = False,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.outcomes, self.refuse, self.after, self.moved_plan = outcomes, refuse, polls, moved_plan
        self.broken_peers, self.interrupt_peers = broken_peers, interrupt_peers
        self.peer_polls, self.plans = 0, 0
        self.jobs: dict[str, str] = {}

    def routes(self) -> dict[tuple[str, str], Any]:
        return {**super().routes(), ("POST", "/api/plans"): lambda r: self.plan(),
                ("GET", "/api/peers"): lambda r: self.peers()}

    def plan(self) -> httpx.Response:
        with self.lock:
            self.plans += 1
            n = self.plans
        return httpx.Response(200, json=_seed_plan(f"p{n}", [9] if self.moved_plan and n > 1 else None))

    def handle(self, request: httpx.Request) -> httpx.Response:
        m = _SEED_ROUTE.fullmatch(request.url.path)
        if request.method != "POST" or m is None:
            return super().handle(request)
        with self.lock:
            self.requests.append((request.method, request.url.path, dict(request.headers),
                                  json.loads(request.content) if request.content else None))
        if request.headers.get("x-hypercompute-owner") != TOKEN:
            return httpx.Response(403, json={"detail": "missing or wrong owner token"})
        if m[2]:
            return httpx.Response(202, json={"command": {"kind": "seed_cancel"}})
        if m[1] in self.refuse:
            return httpx.Response(409, json={"detail": f"node {m[1][:8]} did not allow cache downloads",
                                             "code": "seeding_not_allowed", "fix": "rejoin with --managed --max-disk"})
        with self.lock:
            self.jobs[m[1]] = f"{len(self.jobs) + 1:032x}"
            return httpx.Response(202, json=self.job_view(m[1], "hashing"))

    def job_view(self, node: str, state: str) -> dict[str, Any]:
        return {"job_id": self.jobs[node], "node_id": node, "model_id": "big", "layers": "0-2", "state": state,
                "detail": "bad GGUF" if state == "failed" else None, "tensors": 15, "bytes_total": 30 * GIB}

    def peers(self) -> httpx.Response:
        if self.interrupt_peers:
            raise KeyboardInterrupt
        if self.broken_peers:
            return httpx.Response(500, json={"detail": "boom"})
        with self.lock:
            self.peer_polls += 1
            finished = self.peer_polls >= self.after
            return httpx.Response(200, json=[self.peer_row(n, finished) for n in (MAC_A, MAC_B)])

    def peer_row(self, node: str, finished: bool) -> dict[str, Any]:
        outcome, job = self.outcomes.get(node, "stuck"), self.jobs.get(node)
        host = "failed" if finished and outcome == "host_failed" else ("delivered" if finished else "queued")
        state = outcome if finished and outcome in ("done", "failed") else "running"
        seed = None if job is None or outcome == "host_failed" else {
            "job_id": job, "model_id": "big", "state": state, "parts_received": 1, "parts": 1, "tensors_total": 15,
            "tensors_done": 15 if state == "done" else 4, "tensors_present": 2,
            "tensors_failed": 3 if state == "failed" else 0, "bytes_total": 30 * GIB,
            "bytes_done": 30 * GIB if state == "done" else 8 * GIB,
            "failures": [{"fnv64": "0" * 16, "error": "HTTP 404"}] if state == "failed" else [],
            "error": "3 tensors failed" if state == "failed" else None}
        return {"node_id": node, "display_name": f"mac-{node[0]}", "managed": True, "report": {"seed": seed},
                "seed_request": self.job_view(node, host) if job else None}


def _seed_run(tmp_path: Path, gw: FakeGateway, **kw: Any) -> tuple[dict[str, Any], str]:
    return _run(tmp_path, gw, **{"seed_peers": True, "seed_poll_s": 0.0, **kw})


def _paths(gw: FakeGateway, method: str) -> list[str]:
    return [p for m, p, _, _ in gw.requests if m == method]


def test_seed_peers_records_each_peer_and_a_failed_seed_does_not_stop_the_load(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "failed"})
    doc, status = _seed_run(tmp_path, gw)
    assert status == "ok" and doc["measured"]["chats"]["summary"]["ok"] == 3
    seed = doc["measured"]["seed"]
    assert seed["plan_id"] == "p1" and seed["done"] == 1 and seed["failed"] == 1 and "replan" not in seed
    a, b = seed["peers"][MAC_A], seed["peers"][MAC_B]
    assert a["state"] == "done" and a["seconds"] >= 0 and a["bytes_done"] == a["bytes_total"] == 30 * GIB
    assert a["tensors_total"] == a["tensors_done"] == 15 and a["tensors_present"] == 2 and a["failures"] == []
    assert a["host_job"]["state"] == "delivered" and a["job_id"] == f"{1:032x}"
    assert b["state"] == "failed" and "peer job failed: 3 tensors failed" in b["error"]
    assert b["tensors_failed"] == 3 and b["failures"] == [{"fnv64": "0" * 16, "error": "HTTP 404"}]
    posts = _paths(gw, "POST")
    assert posts.index(f"/api/peers/{MAC_A}/seed") < posts.index("/api/deployments")  # seeded before the load
    bodies = [body for m, p, _, body in gw.requests if m == "POST" and p.endswith("/seed")]
    assert bodies == [{"model_id": "big", "plan_id": "p1"}] * 2
    assert gw.calls("POST", "/api/deployments")[0][3] == {"plan_id": "p1"} and gw.plans == 1
    assert "seed:start" in str(doc["measured"]["host"]["marks"]) and not any(p.endswith("/cancel") for p in posts)


def test_a_refused_seed_and_a_failed_host_job_are_recorded_and_the_load_goes_ahead(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "host_failed"}, refuse=frozenset({MAC_B}))
    doc, status = _seed_run(tmp_path, gw)
    assert status == "ok" and gw.calls("POST", "/api/deployments")
    a, b = doc["measured"]["seed"]["peers"][MAC_A], doc["measured"]["seed"]["peers"][MAC_B]
    assert a["state"] == "failed" and a["error"] == "host job failed: bad GGUF" and a["bytes_done"] is None
    assert b["state"] == "failed" and "seeding_not_allowed" in b["error"] and b["fix"] and b["seconds"] == 0.0
    assert doc["measured"]["seed"]["done"] == 0 and doc["measured"]["seed"]["failed"] == 2


def test_seeds_still_running_at_the_timeout_are_cancelled_and_the_load_goes_ahead(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "stuck"}, polls=1)
    doc, status = _seed_run(tmp_path, gw, seed_timeout_s=0.05)
    peers = doc["measured"]["seed"]["peers"]
    assert status == "ok" and peers[MAC_A]["state"] == "done"
    assert peers[MAC_B]["state"] == "timeout" and "cancel queued" in peers[MAC_B]["error"]
    assert peers[MAC_B]["seconds"] >= 0.05 and peers[MAC_B]["bytes_done"] == 8 * GIB  # the last progress seen
    assert _paths(gw, "POST").count(f"/api/peers/{MAC_B}/seed/cancel") == 1
    assert f"/api/peers/{MAC_A}/seed/cancel" not in _paths(gw, "POST")


def test_without_seed_peers_nothing_is_seeded(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "done"})
    doc, status = _run(tmp_path, gw)
    assert status == "ok" and "seed" not in doc["measured"] and doc["requested"]["seed_peers"] is False
    assert not any(p.startswith("/api/peers") for _, p, _, _ in gw.requests)


def test_a_plan_without_peer_layers_seeds_nothing(tmp_path: Path) -> None:
    gw = FakeGateway()
    doc, status = _seed_run(tmp_path, gw)
    assert status == "ok" and doc["measured"]["seed"]["peers"] == {} and doc["measured"]["seed"]["note"]
    assert not any(p.startswith("/api/peers") for _, p, _, _ in gw.requests)


@pytest.mark.parametrize("moved", [False, True])
def test_a_plan_aged_by_seeding_is_planned_again_before_the_deployment(tmp_path: Path, moved: bool) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "done"}, moved_plan=moved)
    doc, status = _seed_run(tmp_path, gw, replan_after_s=0.0)
    assert status == "ok" and gw.plans == 2
    assert doc["predicted"]["seed_plan"]["id"] == "p1" and doc["predicted"]["plan"]["id"] == "p2"
    assert gw.calls("POST", "/api/deployments")[0][3] == {"plan_id": "p2"}
    replan = doc["measured"]["seed"]["replan"]
    assert replan["seeded_plan_id"] == "p1" and replan["plan_id"] == "p2"
    assert replan["peers_with_moved_layers"] == ([MAC_B] if moved else [])


def test_seeding_stops_polling_a_gateway_that_keeps_failing(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "done"}, broken_peers=True)
    doc, status = _seed_run(tmp_path, gw)
    peers = doc["measured"]["seed"]["peers"]
    assert status == "ok" and all(p["state"] == "failed" and "stopped polling" in p["error"] for p in peers.values())
    assert len(gw.calls("GET", "/api/peers")) == 5


def test_ctrl_c_while_seeding_keeps_the_started_seeds_and_deploys_nothing(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "done"}, interrupt_peers=True)
    doc, status = _seed_run(tmp_path, gw)
    assert status == "interrupted" and doc["measured"]["seed"]["interrupted"] is True
    assert set(doc["measured"]["seed"]["peers"]) == {MAC_A, MAC_B} and not gw.calls("POST", "/api/deployments")


def test_seed_outcome_reads_the_host_job_and_the_peer_report_of_the_same_job() -> None:
    job = "1" * 32
    row = {"seed_request": {"job_id": job, "state": "delivered"}, "report": {"seed": {"job_id": job, "state": "done"}}}
    assert rb.seed_outcome(job, row) == ("done", None)
    assert rb.seed_outcome(job, None)[0] == "failed" and "no longer listed" in rb.seed_outcome(job, None)[1]
    assert "no longer tracks" in rb.seed_outcome("2" * 32, row)[1] and rb.seed_outcome(None, row)[0] == "failed"
    stale_report = {**row, "report": {"seed": {"job_id": "2" * 32, "state": "done"}}}  # a report of an older job
    assert rb.seed_outcome(job, stale_report) == ("pending", None)
    assert rb.seed_outcome(job, {**row, "report": None}) == ("pending", None)
    cancelled = {**row, "report": {"seed": {"job_id": job, "state": "cancelled", "error": "cancelled by the host"}}}
    assert rb.seed_outcome(job, cancelled) == ("failed", "peer job cancelled: cancelled by the host")
    refused = {"seed_request": {"job_id": job, "state": "refused", "detail": "needs 90 GiB"}}
    assert rb.seed_outcome(job, refused) == ("failed", "host job refused: needs 90 GiB")


def test_plan_peer_layers_merges_a_peers_devices_and_skips_the_host() -> None:
    plan = {"assignments": [{"node_id": MAC_A, "layers": [4, 5]}, {"node_id": "local", "layers": [6]},
                            {"node_id": MAC_A, "layers": [2, 3]}, {"node_id": MAC_B, "layers": []}, "junk"]}
    assert rb.plan_peer_layers(plan) == {MAC_A: [2, 3, 4, 5]} and rb.plan_peer_layers({}) == {}


def test_parse_args_seed_flags() -> None:
    plain = rb.parse_args(["--model", "m"])
    assert plain.seed_peers is False and plain.seed_timeout_s == 3600.0
    opts = rb.parse_args(["--model", "m", "--seed-peers", "--seed-timeout", "900"])
    assert opts.seed_peers is True and opts.seed_timeout_s == 900.0
    with pytest.raises(SystemExit):
        rb.parse_args(["--model", "m", "--seed-peers", "--seed-timeout", "0"])


# ----------------------------------------------------------------------------- --layout (explicit layouts)

LAYOUT = [{"node_id": MAC_A, "device": "MTL0", "layers": 3},
          {"node_id": MAC_A, "device": "CPU", "layers": 6, "residency": "file_backed"},
          {"node_id": "local", "device": "CUDA0", "layers": 27}]


class LayoutGateway(FakeGateway):
    """FakeGateway whose plan route answers an explicit layout as the gateway does: the plan carries it back, or a 422
    with the gateway's code and fix when `reject` is set."""

    def __init__(self, *, reject: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.reject = reject

    def routes(self) -> dict[tuple[str, str], Any]:
        return {**super().routes(), ("POST", "/api/plans"): self.plan}

    def plan(self, request: httpx.Request) -> httpx.Response:
        if self.reject:
            return httpx.Response(422, json={"detail": "mac-a MTL0 needs ~8.4 GiB of RAM", "code": "layout_memory",
                                             "fix": "give that stage fewer layers"})
        body = json.loads(request.content)
        return httpx.Response(200, json={**_plan(), "tier": "moe_offload", "layout": body.get("layout"),
                                         "device_list": ["RPC0", "RPC1", "CUDA0"], "tensor_split": [3, 6, 28],
                                         "assignments": [
                                             {"node_id": MAC_A, "device": "RPC0", "layers": [0, 1, 2],
                                              "file_backed": False, "weight_bytes": 5 * GIB},
                                             {"node_id": MAC_A, "device": "RPC1", "layers": list(range(3, 9)),
                                              "file_backed": True, "weight_bytes": 10 * GIB},
                                             {"node_id": "local", "device": "CUDA0", "layers": list(range(9, 36))}]})


def test_parse_layout_reads_inline_json_or_a_file_and_refuses_what_cannot_be_a_layout(tmp_path: Path) -> None:
    inline = rb.parse_layout(json.dumps(LAYOUT))
    assert list(inline) == LAYOUT
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"stages": LAYOUT}), encoding="utf-8")
    assert rb.parse_layout(str(path)) == inline
    for bad, needle in (("[]", "non-empty"), ("[{", "not valid JSON"), (str(tmp_path / "missing.json"), "readable"),
                        ('[{"node_id": "local", "device": "CUDA0"}]', "stage 0"),
                        ('[{"node_id": "local", "device": "CUDA0", "layers": 0}]', "at least 1"),
                        ('[{"node_id": "local", "device": "CUDA0", "layers": true}]', "whole number"),
                        ('[{"node_id": "local", "device": "CUDA0", "layers": 4, "gpu": 1}]', "nothing else"),
                        ('[{"node_id": "local", "device": "CUDA0", "layers": 4, "residency": "ram"}]', "residency")):
        with pytest.raises(ValueError, match=re.escape(needle)):
            rb.parse_layout(bad)


def test_parse_args_layout_flag() -> None:
    assert rb.parse_args(["--model", "m"]).layout is None
    opts = rb.parse_args(["--model", "m", "--layout", json.dumps(LAYOUT)])
    assert opts.layout == tuple(LAYOUT) and rb.plan_body(opts)["layout"] == LAYOUT
    assert "layout" not in rb.plan_body(rb.parse_args(["--model", "m"]))
    with pytest.raises(SystemExit):
        rb.parse_args(["--model", "m", "--layout", "[]"])


def test_a_layout_run_sends_the_layout_and_records_it_requested_and_predicted(tmp_path: Path) -> None:
    gw = LayoutGateway()
    doc, status = _run(tmp_path, gw, layout=tuple(LAYOUT))
    assert status == "ok"
    (plan_call,) = gw.calls("POST", "/api/plans")
    assert plan_call[3]["layout"] == LAYOUT and plan_call[3]["model_id"] == "big"
    assert doc["requested"]["layout"] == LAYOUT
    layout = doc["predicted"]["layout"]
    assert layout["stages"] == LAYOUT and layout["tier"] == "moe_offload" and layout["tensor_split"] == [3, 6, 28]
    assert [a["file_backed"] for a in layout["assignments"]] == [False, True, None]
    assert doc["predicted"]["plan"]["layout"] == LAYOUT and len(gw.calls("POST", "/api/deployments")) == 1


def test_a_layout_the_gateway_rejects_ends_the_run_refused_with_its_reason(tmp_path: Path) -> None:
    gw = LayoutGateway(reject=True)
    doc, status = _run(tmp_path, gw, layout=tuple(LAYOUT))
    assert status == "refused" and "rejected --layout" in doc["error"]["message"]
    assert "layout_memory" in doc["error"]["message"] and doc["error"]["fix"] == "give that stage fewer layers"
    assert not gw.calls("POST", "/api/deployments") and "layout" not in doc["predicted"]


# ----------------------------------------------------------------------------- --concurrency N (batched chats)

COMPARISON_KEYS = {"predicted_decode_ms", "predicted_decode_source", "measured_decode_ms_per_token_median",
                   "measured_over_predicted", "predicted_load_ms", "measured_load_s",
                   "predicted_stream_bytes_per_token", "predicted_moe_ssd_bytes_per_token_est",
                   "predicted_moe_miss_rate_est", "measured_disk_read_bytes_per_output_token_median",
                   "measured_decode_disk_read_bytes_per_token_median", "note"}


def _is_batch(body: dict[str, Any]) -> bool:
    return re.match(r"Run (warmup-)?b\d", body["messages"][0]["content"]) is not None


def _sse_line(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class BatchGateway(FakeGateway):
    """FakeGateway whose batch requests (prompt tag b<k>r<j> or warmup-b<j>) meet at a Barrier(N) in the handler, so
    a batch that is not sent concurrently breaks it, then stream: the first token at once, then one text delta per
    remaining output token, spread over `decode_s`."""

    def __init__(self, n: int, *, decode_s: float = 0.15, batch_prompt_n: int = 128, **kw: Any) -> None:
        super().__init__(parallel=n, **kw)
        self.n, self.decode_s, self.meet = n, decode_s, threading.Barrier(n, timeout=5.0)
        self.batch_prompt_n = batch_prompt_n
        self.batch_bodies: list[dict[str, Any]] = []

    def chat(self, body: dict[str, Any]) -> httpx.Response:
        if not _is_batch(body):
            return super().chat(body)
        with self.lock:
            self.batch_bodies.append(body)
        self.meet.wait()
        return self.stream_batch(body)

    def stream_batch(self, body: dict[str, Any]) -> httpx.Response:
        n, delay = body["max_tokens"], self.decode_s
        timings = {"prompt_n": self.batch_prompt_n, "prompt_ms": 400.0, "predicted_n": n,
                   "predicted_ms": delay * 1000, "cache_n": 0, "predicted_per_second": 1.0}

        def chunks() -> Any:
            yield _sse_line({"choices": [{"delta": {"content": "a"}, "finish_reason": None}]})
            for _ in range(n - 1):
                time.sleep(delay / (n - 1))
                yield _sse_line({"choices": [{"delta": {"content": "b"}, "finish_reason": None}]})
            yield _sse_line({"choices": [{"delta": {}, "finish_reason": "length"}], "timings": timings})
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())


class Clients:
    """Env.new_client for the batch workers: one fresh client on the gateway's transport per call, counted."""

    def __init__(self, gw: FakeGateway) -> None:
        self.gw, self.made, self.lock = gw, 0, threading.Lock()

    def __call__(self) -> httpx.Client:
        with self.lock:
            self.made += 1
        return self.gw.client()


def _batch_env(clients: Clients) -> Any:
    return rb.Env(clock=time.perf_counter, sleep=lambda s: time.sleep(0.002), say=lambda m: None, new_client=clients)


ENGINE_LOG_KEYS = {"available", "path", "source", "model_buffers_mib", "file_backed_host_mib", "private_host_mib",
                   "metal_mapped_mib", "metal_private_mib", "lines", "lines_dropped"}  # before --concurrency
MEASURED_KEYS = {"gateway_before", "model", "load", "placement", "engine_log", "chats", "peer_worker_cpu", "hcstream",
                 "gateway_runs", "gateway_after", "host"}
CHAT_KEYS = {"source", "reps", "summary"}
CHAT_SUMMARY_KEYS = {"ok", "failed", "ttft_ms", "prompt_n", "prefill_tok_s", "decode_tok_s", "decode_ms_per_token",
                     "disk_read_bytes_per_output_token", "decode_disk_read_bytes_per_token"}


def test_default_run_unchanged(tmp_path: Path) -> None:
    logs = tmp_path / "home" / "coordinator" / "logs"
    logs.mkdir(parents=True)
    (logs / f"llama-server-{DEP_ID}.log").write_text(
        "load_tensors:   CPU_Mapped model buffer size = 30000.50 MiB\n"
        "srv    load_model: initializing slots, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'\n",
        encoding="utf-8")
    gw = FakeGateway()
    doc, status = _run(tmp_path, gw)
    assert status == "ok"
    (plan_call,) = gw.calls("POST", "/api/plans")
    assert "parallel" not in plan_call[3]
    chats = doc["measured"]["chats"]
    assert [r["label"] for r in chats["reps"]] == ["warmup0", "warmup1", "rep0", "rep1", "rep2"]
    assert len(gw.calls("POST", "/v1/chat/completions")) == 5
    assert not {"batches", "batch_summary", "warm_batch", "concurrency"} & set(chats)
    assert all("decode_disk_read_bytes_per_token" in r for r in chats["reps"])
    assert "slots" not in doc["measured"] and set(doc["comparison"]) == COMPARISON_KEYS
    assert set(doc["measured"]) == MEASURED_KEYS and set(chats) == CHAT_KEYS
    assert set(chats["summary"]) == CHAT_SUMMARY_KEYS
    # the only differences an N=1 row gains: requested.concurrency, the engine's own -np 1 parsed into
    # placement.engine.flags["--parallel"], and engine_log.slots from the engine log
    assert doc["requested"]["concurrency"] == 1 and "barrier_timeout_s" not in doc["requested"]
    flags = doc["measured"]["placement"]["engine"]["flags"]
    assert flags["--parallel"] == "1" and "--kv-unified" not in flags
    assert set(doc["measured"]["engine_log"]) == ENGINE_LOG_KEYS | {"slots"}
    assert doc["measured"]["engine_log"]["slots"]["n_slots"] == 1
    assert all("delta_s" not in r for r in chats["reps"])  # the delta times feed the batch math only


def test_concurrency_rejects_overfull_pool(tmp_path: Path) -> None:
    gw = BatchGateway(8)
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=8, ctx=1024)
    assert status == "refused" and "KV pool" in doc["error"]["message"] and doc["error"]["fix"]
    assert not gw.calls("POST", "/api/plans")  # refused before anything is planned or loaded
    # the target fits (8 x (128 + 4 + 64) = 1568), but the first sizing warmup overshot it: pool_short (not retried
    # by matrix) before any batch, with the chats recorded and the deployment stopped
    gw = BatchGateway(8, tokens_per_word=1.4)
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=8, ctx=1600)
    measured = max(r["prompt_n"] for r in doc["measured"]["chats"]["reps"])
    assert measured > 128 and status == "pool_short" and doc["status"] == "pool_short" and not gw.batch_bodies
    assert f"({measured} prompt + 4 slack + 64 output tokens)" in doc["error"]["message"]
    assert doc["error"]["type"] == "PoolShort" and doc["error"]["fix"]
    assert gw.calls("DELETE", f"/api/deployments/{DEP_ID}") and doc["measured"]["chats"]["reps"]
    assert "interrupted" not in doc["measured"]["chats"]
    opts = _opts(tmp_path, concurrency=8, ctx=1536)
    assert rb.pool_problem(opts, 128) is None and "1544" in (rb.pool_problem(opts, 129) or "")
    assert rb.parse_args(["--model", "m", "--concurrency", "8", "--ctx", "1536"]).concurrency == 8
    for argv in (["--concurrency", "8", "--ctx", "1024"], ["--concurrency", "0"], ["--concurrency", "17"]):
        with pytest.raises(SystemExit):
            rb.parse_args(["--model", "m", *argv])
    assert rb.parse_args(["--model", "m"]).concurrency == 1


def test_sizing_that_fits_only_without_slack_is_refused_before_any_batch(tmp_path: Path) -> None:
    """The sizing warmups land just under the bound (4 x (158 + 64) = 888 = --ctx), but the batch prompts use an
    extrapolated word count and their own tags, so they can be a few tokens longer: refused before any batch."""
    gw = BatchGateway(4, tokens_per_word=1.4)
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4, ctx=888)
    sizing = [r["prompt_n"] for r in doc["measured"]["chats"]["reps"]]
    assert max(sizing) == 158 and rb.pool_problem(_opts(tmp_path, concurrency=4, ctx=888), max(sizing)) is None
    assert status == "pool_short" and not gw.batch_bodies and gw.calls("DELETE", f"/api/deployments/{DEP_ID}")
    # the sizing prompts carry tags shaped like the warm batch's warmup-b<j>-<hex6>
    prompts = [body["messages"][0]["content"] for _, _, _, body in gw.calls("POST", "/v1/chat/completions")]
    assert prompts and all(re.match(r"Run warmup-s\d-[0-9a-f]{6}\. ", text) for text in prompts)


def test_a_warm_batch_that_overfills_the_pool_ends_before_the_measured_batches(tmp_path: Path) -> None:
    gw = BatchGateway(4, batch_prompt_n=140)  # sizing measures 126, the warm batch's own prompts 140
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4, ctx=784)
    assert status == "pool_short" and "(140 prompt + 4 slack" in doc["error"]["message"]
    assert len(gw.batch_bodies) == 4 and doc["measured"]["chats"]["batches"] == []
    assert doc["measured"]["chats"]["warm_batch"]["requests"] == [f"warmup-b{j}" for j in range(4)]


def test_concurrency_without_a_client_factory_is_refused_before_anything_loads(tmp_path: Path) -> None:
    gw = BatchGateway(4)
    doc, status = _run(tmp_path, gw, env=_env(), concurrency=4)  # Env() has no new_client
    assert status == "refused" and "Env.new_client" in doc["error"]["message"]
    assert not gw.calls("POST", "/api/plans") and not gw.calls("POST", "/api/deployments")


def test_main_exits_2_on_pool_short(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(rb, "run", lambda *a, **kw: (tmp_path / "run.json", "pool_short"))
    monkeypatch.setenv("SystemDrive", "C:")
    assert rb.main(["--model", "big", "--home", str(tmp_path / "home")]) == 2


def test_batch_overlaps(tmp_path: Path) -> None:
    gw = BatchGateway(4)
    clients = Clients(gw)
    doc, status = _run(tmp_path, gw, env=_batch_env(clients), concurrency=4, reps=3, output_tokens=8)
    assert status == "ok"
    (plan_call,) = gw.calls("POST", "/api/plans")
    assert plan_call[3]["parallel"] == 4
    chats = doc["measured"]["chats"]
    assert chats["concurrency"] == 4 and chats["summary"]["ok"] == 12 and chats["summary"]["failed"] == 0
    assert [r["label"] for r in chats["reps"] if r["label"].startswith("b")] == [
        f"b{k}r{j}" for k in range(3) for j in range(4)]
    assert clients.made == 16 and len(gw.batch_bodies) == 16  # one unmeasured warm batch, three measured; a client each
    assert chats["warm_batch"]["requests"] == [f"warmup-b{j}" for j in range(4)]
    for k, batch in enumerate(chats["batches"]):
        assert batch["label"] == f"b{k}" and batch["status"] == "ok" and batch["window_coverage"] >= 0.8
        assert batch["basis"] == "token_times" and len(batch["window_deltas"]) == 4
        assert all(c["deltas"] == 8 for c in batch["window_deltas"])  # one per output token, as streamed
        assert batch["aggregate_decode_tok_s"] == pytest.approx(batch["window_tokens"] / batch["window"]["s"])
        assert batch["window_tokens"] <= 4 * 7 and batch["missed_first_iteration"] == 0
    assert chats["batch_summary"]["ok"] == 3 and chats["batch_summary"]["not_concurrent"] == 0
    # overlapping windows would count every byte N times: no per-request disk figures
    batch_rows = [r for r in chats["reps"] if r["label"].startswith("b")]
    assert all(r["decode_disk_read_bytes_per_token"] is None and r["disk_read_bytes_per_output_token"] is None
               for r in batch_rows)
    assert doc["requested"]["concurrency"] == 4 and doc["measured"]["slots"]["short"] is False
    assert doc["comparison"]["concurrency"] == 4 and "single stream" in doc["comparison"]["predicted_decode_basis"]


def test_batch_of_one_matches_host_fields() -> None:
    """A batch of one is an N=1 rep: the window is its decode span, its steps are predicted_n - 1, its SSD bytes per
    step are the rep's decode_disk_read_bytes_per_token."""
    samples = [hm.Sample(t=float(t), disk_read_bytes=v)
               for t, v in ((0, 0), (1, 5000), (2, 5000), (3, 5100), (4, 5200), (5, 5300))]
    rep = _result(ttft_ms=1500.0, t_start=0.0, t_end=5.0)
    batch = rb.batch_record("b0", [rep], samples)
    host = rb._host_fields(rep, samples)
    assert batch["status"] == "ok" and batch["requests"] == ["rep0"] and batch["window_coverage"] == 1.0
    assert batch["ssd_bytes_per_step"] == host["decode_disk_read_bytes_per_token"] == 100.0
    # decode starts at end - predicted_ms (2.0 s), not at the first visible delta (1.5 s): the engine's own rate
    assert batch["window"]["t0"] == 2.0 and batch["aggregate_decode_tok_s"] == pytest.approx(3 / 3.0)
    assert batch["step_ms"] == pytest.approx(1000.0)
    assert batch["first_token_skew_s"] == 0.0 and batch["missed_first_iteration"] == 0
    failed = rb.batch_record("b1", [rep, rb.ChatResult("rep1", False, error="peer lost")], samples)
    assert failed["status"] == "incomplete" and failed["failed"] == ["rep1"] and "aggregate_decode_tok_s" not in failed


def test_span_anchors_decode_to_the_engine_and_keeps_the_delta_times() -> None:
    """A late first visible delta (gpt-oss harmony) does not move the decode start: it is end - predicted_ms."""
    rep = _result(ttft_ms=2600.0, t_start=10.0, t_end=15.0, delta_s=(2.6, 3.4, 4.2, 5.0))
    span = rb._span(rep)
    assert span.first == pytest.approx(15.0 - 3.0) and span.end == 15.0 and span.output_tokens == 4
    assert span.deltas == pytest.approx((12.6, 13.4, 14.2, 15.0))
    early = rb._span(_result(ttft_ms=100.0, t_start=10.0, t_end=12.0))  # predicted_ms past the send: clamped
    assert early.first == 10.0 and early.deltas is None
    assert rb._span(_result(ttft_ms=None, t_start=0.0, t_end=1.0)) is None  # never streamed text


def _batch_results(k: int, *, ttft: float | None = 100.0) -> list[Any]:
    return [_result(label=f"b{k}r{j}", ttft_ms=ttft if j == 0 else 100.0, t_start=0.0, t_end=3.5) for j in range(4)]


def test_a_batch_whose_request_never_streamed_text_fails_the_run() -> None:
    """The request finished with timings (summary ok) but streamed no text delta, so its batch is incomplete."""
    samples = [hm.Sample(t=0.0, disk_read_bytes=0), hm.Sample(t=4.0, disk_read_bytes=100)]
    one = rb.chat_section(_batch_results(0, ttft=None), [], samples,
                          rb.BatchLog(measured=[_batch_results(0, ttft=None)]), concurrency=4)
    assert one["summary"]["failed"] == 0 and one["batch_summary"]["incomplete"] == 1
    assert rb.chat_status(one) == "chat_failed"
    groups = [_batch_results(0, ttft=None), _batch_results(1)]
    two = rb.chat_section([r for g in groups for r in g], [], samples, rb.BatchLog(measured=groups), concurrency=4)
    assert two["batch_summary"]["incomplete"] == 1 and two["batch_summary"]["ok"] == 1
    assert rb.chat_status(two) == "partial"
    clean = rb.chat_section(_batch_results(1), [], samples, rb.BatchLog(measured=[_batch_results(1)]), concurrency=4)
    assert rb.chat_status(clean) == "ok"


def test_read_stream_records_each_text_delta_arrival() -> None:
    ticks = iter([0.5, 0.75, 1.0])
    raw = (b'data: {"choices": [{"delta": {"reasoning_content": "hm"}}]}\n\n'
           b'data: {"choices": [{"delta": {}}]}\n\n'
           b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n')
    out = rb.read_stream([raw], 0.0, lambda: next(ticks))
    assert out["ttft_s"] == 0.5 and out["deltas"] == (0.5, 0.75)  # an empty delta is not a token


def test_slots_short_skips_chats(tmp_path: Path) -> None:
    gw = FakeGateway()  # an engine launched with -np 1: an old gateway, or one that ignored the request
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4)
    assert status == "slots_short" and not gw.calls("POST", "/v1/chat/completions")
    slots = doc["measured"]["slots"]
    assert slots["short"] is True and slots["requested"] == 4 and slots["engine_parallel"] == "1"
    assert gw.calls("DELETE", f"/api/deployments/{DEP_ID}")


class StuckBatchGateway(BatchGateway):
    """The first batch request presses Ctrl+C (in the runner's main thread); every batch request then hangs until the
    deployment is deleted, as an engine stream does."""

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.deleted = threading.Event()

    def stream_batch(self, body: dict[str, Any]) -> httpx.Response:
        with self.lock:
            first = body is self.batch_bodies[0]
        if first:
            _thread.interrupt_main()
        self.deleted.wait(10.0)
        return super().stream_batch(body)

    def deployment(self, method: str) -> httpx.Response:
        if method == "DELETE":
            self.deleted.set()
        return super().deployment(method)


def test_interrupt_mid_batch_stops_deployment(tmp_path: Path) -> None:
    gw = StuckBatchGateway(4)
    t0 = time.perf_counter()
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4)
    # the runner did not wait for the hung chats (they end only once the deployment is deleted)
    assert time.perf_counter() - t0 < 8.0
    assert status == "interrupted" and gw.calls("DELETE", f"/api/deployments/{DEP_ID}")
    assert doc["measured"]["chats"]["interrupted"] is True and doc["measured"]["load"]["seconds"] is not None


def test_interrupt_mid_batch_with_keep_holds_no_thread_open(tmp_path: Path) -> None:
    """--keep: nothing ends the hung chats, but the workers are daemon threads and the barrier is aborted, so run()
    returns at once and no worker would keep the process from exiting."""
    gw = StuckBatchGateway(4)
    t0 = time.perf_counter()
    try:
        doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4, keep=True)
        alive = [t for t in threading.enumerate() if t.name.startswith("batch") and t.is_alive() and not t.daemon]
    finally:
        gw.deleted.set()  # release the hung fake streams
    assert time.perf_counter() - t0 < 8.0 and alive == []
    assert status == "interrupted" and not gw.calls("DELETE", f"/api/deployments/{DEP_ID}") and doc["kept"] is True


def test_interrupted_process_exits_while_batch_chats_hang(tmp_path: Path) -> None:
    """A subprocess whose batch workers hang forever still exits right after the Ctrl+C reaches run_together."""
    code = ("import _thread, sys, threading, time\n"
            f"sys.path.insert(0, {str(ROOT / 'scripts' / 'capacity')!r})\n"
            "import batchlib\n"
            "def hang(start):\n    start()\n    threading.Event().wait()\n"
            "threading.Timer(0.2, _thread.interrupt_main).start()\n"
            "try:\n    batchlib.run_together([hang, hang], lambda i, e: None, 30.0, poll_s=0.02)\n"
            "except KeyboardInterrupt:\n    print('interrupted', flush=True)\n")
    t0 = time.perf_counter()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert out.stdout.strip() == "interrupted" and time.perf_counter() - t0 < 10.0


class StaleGateway(BatchGateway):
    """A gateway whose PlanRequest has no `parallel` (or that refuses the layout): FastAPI's 422 detail list."""

    def __init__(self, loc: str, **kw: Any) -> None:
        super().__init__(4, **kw)
        self.loc = loc

    def routes(self) -> dict[tuple[str, str], Any]:
        detail = [{"type": "extra_forbidden", "loc": ["body", self.loc], "msg": "Extra inputs are not permitted"}]
        return {**super().routes(), ("POST", "/api/plans"): lambda r: httpx.Response(422, json={"detail": detail})}


@pytest.mark.parametrize("layout", [None, tuple(LAYOUT)])
def test_a_stale_gateway_rejecting_parallel_is_named_with_or_without_layout(tmp_path: Path, layout: Any) -> None:
    gw = StaleGateway("parallel")
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4, layout=layout)
    assert status == "refused" and "rejected parallel 4" in doc["error"]["message"]
    assert "body.parallel: Extra inputs are not permitted" in doc["error"]["message"]
    assert "restart the gateway" in doc["error"]["fix"] and not gw.calls("POST", "/api/deployments")


def test_a_layout_422_with_concurrency_still_names_the_layout(tmp_path: Path) -> None:
    gw = StaleGateway("layout")
    doc, status = _run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=4, layout=tuple(LAYOUT))
    assert status == "refused" and "rejected --layout" in doc["error"]["message"]
    assert "body.layout" in doc["error"]["message"] and "parallel" not in doc["error"]["message"]


def test_engine_flags_parse_parallel_kvu() -> None:
    short = rb.engine_flags(["-np", "4", "-kvu", "-c", "4096"])["flags"]
    long = rb.engine_flags(["--parallel", "4", "--kv-unified", "--ctx-size", "4096"])["flags"]
    assert short == long == {"--parallel": "4", "--kv-unified": True, "--ctx-size": "4096"}


def test_log_evidence_parses_slots_and_kv_cells(tmp_path: Path) -> None:
    log = tmp_path / "e.log"
    log.write_text("llama_context: n_seq_max             = 8\nllama_context: n_ctx                 = 4096\n"
                   "llama_context: n_ctx_seq             = 4096\n"
                   "llama_kv_cache:      CUDA0 KV buffer size =   244.00 MiB\n"
                   "llama_kv_cache: size =  244.00 MiB (  4096 cells,  61 layers,  8/1 seqs), K (q8_0): 122.00 MiB, "
                   "V (q8_0): 122.00 MiB\n"
                   "srv    load_model: initializing, n_slots = 8, n_ctx_slot = 4096, kv_unified = 'true'\n",
                   encoding="utf-8")
    slots = rb.log_evidence(log)["slots"]
    assert slots == {"n_slots": 8, "n_ctx_slot": 4096, "kv_unified": True, "n_seq_max": 8, "n_ctx": 4096,
                     "n_ctx_seq": 4096, "kv_caches": [{"mib": 244.0, "cells": 4096, "layers": 61, "n_seq_max": 8,
                                                       "n_stream": 1}]}
    old = tmp_path / "old.log"
    old.write_text("load_tensors: CPU_Mapped model buffer size = 1.0 MiB\n", encoding="utf-8")
    assert rb.log_evidence(old)["slots"]["n_slots"] is None and rb.log_evidence(old)["slots"]["kv_caches"] == []


def test_fresh_plan_replans_with_parallel(tmp_path: Path) -> None:
    gw = SeedGateway({MAC_A: "done", MAC_B: "done"}, parallel=2)
    _seed_run(tmp_path, gw, env=_batch_env(Clients(gw)), concurrency=2, replan_after_s=0.0)
    bodies = [body for _, _, _, body in gw.calls("POST", "/api/plans")]
    assert gw.plans == 2 and [b["parallel"] for b in bodies] == [2, 2]
