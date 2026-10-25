"""Review fixes for the deployment lifecycle: host-RAM leases, engines left running by a crashed gateway, engine load
and idle budgets, an engine still busy after a disconnect, non-streamed chats, benchmarks of a deployment that stops,
manifest paths and tuner trials. Fake engines only; nothing touches a GPU, WSL or port 8000."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from factories import make_inventory
from hypercompute import config
from hypercompute.adapters.base import CompletionResult, DeploymentHandle, EngineAdapterError, StopResult
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.agent.contribution import default_contribution
from hypercompute.agent.supervisor import Supervisor
from hypercompute.schemas import (
    BenchmarkRequest,
    CreateDeploymentRequest,
    DeploymentState,
    ModelFile,
    PlanRequest,
    PlanStrategy,
    utcnow,
)
from hypercompute.services.benchmarks import BenchmarkService
from hypercompute.services.deployments import (
    DeploymentError,
    DeploymentManager,
    cpu_budget,
    local_budgets,
    warmup_timeout,
)
from hypercompute.services.leases import LeaseManager
from hypercompute.services.models import ModelError, ModelService
from hypercompute.services.nodes import NodeRegistry
from hypercompute.services.planner import ModelShape, make_plan
from hypercompute.services.sessions import SessionService
from hypercompute.services.streaming import INTERRUPTED, InferenceService
from hypercompute.units import GiB
from test_app_http import OWNER, make_coord, open_client
from test_deployments import FakeAdapter, FakeModels, _plan, _setup, _wait_terminal_or_ready
from test_inference_slots import DONE, Models, deployment, drain, event, service
from test_ram_tier import qwen14b_q6k

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
OWNER_HEADERS = {"X-Hypercompute-Owner": OWNER}


async def until(cond, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "condition not reached"
        await asyncio.sleep(0.005)


# --------------------------------------------------------------------------- 3: host-RAM tier lease
@pytest.mark.anyio
async def test_gpu_ram_plan_reserves_its_host_ram_layers(store, events, tmp_path):
    inv = make_inventory()
    sessions = SessionService(store, events, b"k" * 32, 8000, lambda: ["10.0.0.2"])
    reg = NodeRegistry(store, events, sessions, b"k" * 32, inv.engine, inv, default_contribution(inv))
    cpu = cpu_budget(reg.local)
    req = PlanRequest(model_id="m", strategy=PlanStrategy.local_only)
    plan = make_plan(req, ModelShape.from_gguf(qwen14b_q6k(), 4096), local_budgets(reg.local), [],
                     cpu.budget_bytes + 4 * GiB, cpu)
    host = next(a for a in plan.assignments if a.device == "CPU")
    assert plan.feasible and plan.tier == "gpu_ram" and host.layers
    # the coordinator's default allowance counts only the GPU, so CPU + GPU is over it: the planner's own sum
    assert sum(a.total_bytes_est for a in plan.assignments) > reg.local.contribution.max_bytes
    plan = plan.model_copy(update={"id": "ram-plan", "created_at": utcnow()})
    store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?)",
                  (plan.id, plan.created_at.isoformat(), plan.model_dump_json()))
    leases = LeaseManager(store)
    mgr = DeploymentManager(store, events, reg, FakeModels(), leases, FakeAdapter(), Supervisor(tmp_path / "m.json"))
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id=plan.id), None)).id)
    assert dep.state == DeploymentState.ready, dep.error
    held = {r["pool_id"]: r["bytes"] for r in leases.active()}
    assert set(held) == {"ram", "vram:CUDA0"} and held["ram"] == host.total_bytes_est


# --------------------------------------------------------------------------- 5 + 11: engines a crash left running
class NoProcessAdapter(FakeAdapter):
    """An engine with no local process (like EXO): pid 0, no Supervisor record, and a cleanup_orphan hook."""

    def __init__(self, gone: bool = True, clean_stop: bool = True, fail: bool = False) -> None:
        super().__init__()
        self.gone, self.clean_stop, self.fail = gone, clean_stop, fail
        self.orphans: list[str] = []

    async def prepare(self, spec):
        handle = await super().prepare(spec)
        return replace(handle, pid=0, process_key=f"exo:{spec.deployment_id}")

    async def stop(self, handle):
        self.stopped += 1
        return StopResult(clean=self.clean_stop, detail="deleted" if self.clean_stop else "EXO DELETE failed")

    async def cleanup_orphan(self, dep):
        self.orphans.append(dep.id)
        if self.fail:
            raise httpx.ConnectError("EXO unreachable")
        return (True, "EXO instance i-1 deleted") if self.gone else (False, "EXO did not answer the DELETE")


async def _crash_with_ready_deployment(store, events, tmp_path, adapter):
    mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.ready and len(leases.active()) == 2
    for task in list(mgr._tasks.values()):  # the gateway dies: no stop, no cleanup
        task.cancel()
    return dep, reg, leases


def _restarted(store, events, tmp_path, reg, leases, adapter) -> DeploymentManager:
    return DeploymentManager(store, events, reg, FakeModels(), leases, adapter, Supervisor(tmp_path / "restart.json"))


@pytest.mark.anyio
async def test_restart_holds_leases_until_the_engine_removes_its_orphan(store, events, tmp_path):
    adapter = NoProcessAdapter()
    dep, reg, leases = await _crash_with_ready_deployment(store, events, tmp_path, adapter)
    fresh = _restarted(store, events, tmp_path, reg, leases, adapter)
    notes = fresh.reconcile_on_start()  # sync: marks it failed, keeps the leases for the async check
    after = fresh.get(dep.id)
    assert after.state == DeploymentState.failed and len(leases.active()) == 2 and adapter.orphans == []
    assert "no orphaned process found" not in after.cleanup and "released" not in " ".join(notes)
    assert any(dep.id[:8] in n and "leases" in n for n in notes)
    notes = await fresh.cleanup_orphans()
    after = fresh.get(dep.id)
    assert adapter.orphans == [dep.id] and leases.active() == []
    assert "EXO instance i-1 deleted" in after.cleanup and "released 2 lease(s)" in after.cleanup
    assert any(dep.id[:8] in n and "deleted" in n for n in notes)
    assert await fresh.cleanup_orphans() == []  # once only


@pytest.mark.anyio
@pytest.mark.parametrize("adapter_kw", [{"gone": False}, {"fail": True}])
async def test_restart_keeps_leases_when_the_orphan_is_not_confirmed_gone(store, events, tmp_path, adapter_kw):
    adapter = NoProcessAdapter(**adapter_kw)
    dep, reg, leases = await _crash_with_ready_deployment(store, events, tmp_path, adapter)
    fresh = _restarted(store, events, tmp_path, reg, leases, adapter)
    fresh.reconcile_on_start()
    notes = await fresh.cleanup_orphans()
    after = fresh.get(dep.id)
    assert adapter.orphans == [dep.id] and len(leases.active()) == 2
    assert "held" in after.cleanup and "released" not in after.cleanup and dep.id[:8] in notes[0]


@pytest.mark.anyio
async def test_unclean_stop_of_an_engine_without_a_process_is_cleaned_at_the_next_start(store, events, tmp_path):
    adapter = NoProcessAdapter(clean_stop=False)
    mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    stopped = await mgr.stop(dep.id)
    assert stopped.state == DeploymentState.failed and len(leases.active()) == 2
    assert "pid 0" not in stopped.cleanup and "may still be running" in stopped.cleanup
    adapter.clean_stop = True
    fresh = _restarted(store, events, tmp_path, reg, leases, adapter)
    fresh.reconcile_on_start()
    assert len(leases.active()) == 2
    await fresh.cleanup_orphans()
    assert adapter.orphans == [dep.id] and leases.active() == []
    assert "EXO DELETE failed" in fresh.get(dep.id).cleanup  # the stop's own story is kept


@pytest.mark.anyio
@pytest.mark.parametrize("unclean_stop", [False, True])
async def test_restarts_with_the_engine_unreachable_replace_the_orphan_note(store, events, tmp_path, unclean_stop):
    adapter = NoProcessAdapter(fail=True, clean_stop=not unclean_stop)
    if unclean_stop:
        mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
        dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
        await mgr.stop(dep.id)
    else:
        dep, reg, leases = await _crash_with_ready_deployment(store, events, tmp_path, adapter)

    async def restart() -> str:
        fresh = _restarted(store, events, tmp_path, reg, leases, adapter)
        fresh.reconcile_on_start()
        await fresh.cleanup_orphans()
        return fresh.get(dep.id).cleanup

    texts = [await restart() for _ in range(3)]  # EXO stays unreachable for three gateway starts
    assert texts[0] == texts[1] == texts[2] and texts[0].count("EXO unreachable") == 1, texts
    assert ("EXO DELETE failed" in texts[0]) == unclean_stop and len(leases.active()) == 2
    adapter.fail = False
    final = await restart()
    assert leases.active() == [] and "EXO unreachable" not in final and final.count("i-1 deleted") == 1
    assert ("EXO DELETE failed" in final) == unclean_stop  # the stop's own story survives every restart


def test_gateway_start_runs_the_orphan_cleanup_and_reports_it(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    calls: list[bool] = []

    async def cleanup_orphans():
        calls.append(True)
        return ["deployment abcd1234: EXO instance i-1 deleted; released 2 lease(s)"]

    monkeypatch.setattr(coord.deployments, "cleanup_orphans", cleanup_orphans, raising=False)
    with open_client(coord):
        end = time.monotonic() + 5
        while not any("i-1 deleted" in e.message for e in coord.events.since(0)):
            assert time.monotonic() < end, "orphan cleanup did not run at startup"
            time.sleep(0.01)
    assert calls == [True]
    assert [e.type for e in coord.events.since(0) if "i-1 deleted" in e.message] == ["gateway.startup"]


# --------------------------------------------------------------------------- 6 + 22: engine load budget
class LoadBudgetAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.waited: list[float] = []

    def load_timeout_s(self, plan) -> float:
        return 4 * 3600.0

    async def wait_ready(self, handle, timeout_s=config.LOAD_TIMEOUT_S, on_progress=None):
        self.waited.append(timeout_s)


@pytest.mark.anyio
async def test_wait_ready_gets_the_engines_own_load_budget(store, events, tmp_path):
    adapter = LoadBudgetAdapter()
    mgr, _, _ = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.ready and adapter.waited == [4 * 3600.0]


# --------------------------------------------------------------------------- 18: capacity-mode warmup budget
def test_capacity_warmup_uses_the_deployments_own_idle_budget():
    plan = _plan(remote=False).model_copy(update={"capacity_mode": True})
    adapter = SimpleNamespace(idle_timeout_s=900.0, idle_timeout_for=lambda p: 11571.0)
    assert warmup_timeout(plan, adapter) == 2 * 11571.0
    assert warmup_timeout(plan, SimpleNamespace(idle_timeout_s=900.0)) == 1800.0
    assert warmup_timeout(_plan(remote=False), adapter) == config.WARMUP_TIMEOUT_S


class SilentAdapter:
    idle_timeout_s = 900.0

    def idle_timeout_for(self, plan) -> float:
        return 11571.0

    async def stream(self, handle, req):
        raise httpx.ReadTimeout("no data")
        yield  # pragma: no cover

    async def cancel(self, request_id):
        return None


@pytest.mark.anyio
async def test_stream_timeout_message_names_the_deployments_idle_budget(store, events):
    inf, _ = service(store, events, SilentAdapter(), deployment())
    out = await drain(inf.chat_stream(dict(BODY)))
    assert b"no upstream data for 11571s" in out and inf.runs()[0].status == "timeout"


# --------------------------------------------------------------------------- 12: engine still busy after a disconnect
class BusyThenFreeAdapter:
    """An AirLLM worker still finishing a cancelled request: it answers busy (HTTP 429) for the first `busy` calls."""

    def __init__(self, busy: int) -> None:
        self.busy, self.calls = busy, 0

    async def stream(self, handle, req):
        self.calls += 1
        if self.calls <= self.busy:
            raise EngineAdapterError("engine_busy", "AirLLM worker HTTP 429: serves one request at a time")
        yield event("hi", "stop")
        yield DONE

    async def cancel(self, request_id):
        return None


@pytest.mark.anyio
async def test_an_engine_still_busy_is_waited_for_like_a_slot(store, events):
    adapter = BusyThenFreeAdapter(busy=2)
    inf, _ = service(store, events, adapter, deployment(), busy_wait_s=5.0)
    inf.busy_retry_s = 0.01
    out = await drain(inf.chat_stream(dict(BODY)))
    assert b"error" not in out and b"[DONE]" in out and adapter.calls == 3
    assert [r.status for r in inf.runs()] == ["ok"]
    adapter.calls = 0
    status, _ = await inf.chat_complete(dict(BODY))
    assert status == 200 and adapter.calls == 3


@pytest.mark.anyio
async def test_a_stop_during_the_busy_wait_sends_the_engine_nothing_more(store, events):
    adapter = BusyThenFreeAdapter(busy=1)  # free on the second call: a retry after the stop would be served
    inf, _ = service(store, events, adapter, deployment(), busy_wait_s=5.0)
    inf.busy_retry_s = 0.5
    chat = asyncio.create_task(drain(inf.chat_stream(dict(BODY))))
    await until(lambda: adapter.calls == 1)  # answered busy: the chat now waits before it asks again
    await inf.interrupt("d1")
    out = await asyncio.wait_for(chat, 2)
    assert adapter.calls == 1 and b'"code": "interrupted"' in out
    assert [(r.status, r.error) for r in inf.runs()] == [("cancelled", INTERRUPTED)]
    adapter.calls = 0
    complete = asyncio.create_task(inf.chat_complete(dict(BODY)))
    await until(lambda: adapter.calls == 1)
    await inf.interrupt("d1")
    with pytest.raises(DeploymentError) as err:
        await asyncio.wait_for(complete, 2)
    assert adapter.calls == 1 and err.value.status == 502 and INTERRUPTED in str(err.value)
    assert [(r.status, r.stream) for r in inf.runs(1)] == [("cancelled", False)]


@pytest.mark.anyio
async def test_an_engine_busy_past_the_wait_is_refused_as_busy(store, events):
    adapter = BusyThenFreeAdapter(busy=10**6)
    inf, _ = service(store, events, adapter, deployment(), busy_wait_s=0.15)
    inf.busy_retry_s = 0.02
    started = time.monotonic()
    out = await drain(inf.chat_stream(dict(BODY)))
    assert b'"code": "busy"' in out and time.monotonic() - started >= 0.15 and adapter.calls > 2
    with pytest.raises(DeploymentError) as err:
        await inf.chat_complete(dict(BODY))
    assert (err.value.status, err.value.code) == (429, "busy")


# --------------------------------------------------------------------------- 13: non-streamed chats
TOKENS, STEP = 5, 0.15


class SlowEngine(BaseHTTPRequestHandler):
    """llama-server-like: streams a token every STEP; stream=false sends nothing until the whole answer is done."""

    seen: list[dict] = []

    def log_message(self, *args):
        return None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        try:
            if body.get("stream"):
                self._stream()
            else:
                self._whole()
        except OSError:  # the gateway closed the connection
            return

    def _send(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
        self._send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None},
                                         "finish_reason": None}]})
        for i in range(TOKENS):
            time.sleep(STEP)
            self._send({**base, "choices": [{"index": 0, "delta": {"content": "x"},
                                             "finish_reason": "stop" if i == TOKENS - 1 else None}]})
        self._send({**base, "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": TOKENS},
                    "timings": {"prompt_n": 9, "prompt_ms": 5.0, "predicted_n": TOKENS, "predicted_ms": 750.0}})
        self.wfile.write(b"data: [DONE]\n\n")

    def _whole(self) -> None:
        time.sleep(TOKENS * STEP)
        out = json.dumps({"choices": [{"index": 0, "message": {"role": "assistant", "content": "x" * TOKENS},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 9, "completion_tokens": TOKENS}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def slow_engine():
    SlowEngine.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SlowEngine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.mark.anyio
async def test_a_long_non_streamed_answer_is_not_cut_off_by_the_idle_timeout(store, events, slow_engine, monkeypatch):
    monkeypatch.setattr(config, "UPSTREAM_IDLE_TIMEOUT_S", 0.5)  # the answer takes 0.75 s; events come every 0.15 s
    adapter = LlamaRpcAdapter(None, None, None, Path("."))  # type: ignore[arg-type]
    handle = DeploymentHandle("d1", "deployment:d1", slow_engine, Path("x.log"), 1, [])
    deps = SimpleNamespace(ready=lambda: (deployment(), handle))
    inf = InferenceService(store, events, deps, Models(), adapter)
    try:
        status, body = await inf.chat_complete({**BODY, "max_tokens": TOKENS})
    finally:
        await adapter.aclose()
    assert status == 200, body
    assert body["object"] == "chat.completion" and body["id"] == "chatcmpl-1"
    assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "x" * TOKENS},
                                "finish_reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 9, "completion_tokens": TOKENS} and body["timings"]["predicted_n"] == 5
    run = inf.runs(1)[0]
    assert (run.kind, run.status, run.stream, run.output_text, run.finish_reason) == ("chat", "ok", False, "xxxxx",
                                                                                      "stop")
    assert run.completion_tokens == TOKENS and run.ttft_ms is None and run.total_ms >= TOKENS * STEP * 1000 * 0.9
    assert run.timings.predicted_n == TOKENS and SlowEngine.seen[0]["stream"] is True


def test_assembled_completion_keeps_reasoning_tool_calls_and_logprobs():
    from hypercompute.services.streaming import assemble_completion

    def chunk(delta, finish=None, logprobs=None, **extra):
        part = {"index": 0, "delta": delta, "finish_reason": finish, **({"logprobs": logprobs} if logprobs else {})}
        return {"id": "c1", "object": "chat.completion.chunk", "created": 7, "model": "m", "choices": [part], **extra}

    call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_", "arguments": '{"ci'}}
    body = assemble_completion([
        chunk({"role": "assistant"}),
        chunk({"reasoning_content": "think "}, logprobs={"content": [{"token": "a"}]}),
        chunk({"reasoning_content": "more"}, logprobs={"content": [{"token": "b"}]}),
        chunk({"tool_calls": [call]}),
        chunk({"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": 'ty": "Oslo"}'}}]}),
        chunk({}, "tool_calls"),
        {"id": "c1", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 6}},
    ])
    assert body == {
        "id": "c1", "created": 7, "model": "m", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "logprobs": {"content": [{"token": "a"}, {"token": "b"}]},
                     "message": {"role": "assistant", "content": None, "reasoning_content": "think more",
                                 "tool_calls": [{"id": "call_1", "type": "function",
                                                 "function": {"name": "get_weather",
                                                              "arguments": '{"city": "Oslo"}'}}]}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 6}}


class RejectingAdapter:
    """The engine refuses the request (llama-server: context too long)."""

    MESSAGE = '{"error":{"code":400,"message":"the request exceeds the available context size"}}'

    async def stream(self, handle, req):
        raise EngineAdapterError("upstream_error", f"llama-server HTTP 400: {self.MESSAGE}")
        yield  # pragma: no cover

    async def complete(self, handle, req):
        return CompletionResult(400, json.loads(self.MESSAGE))

    async def cancel(self, request_id):
        return None


@pytest.mark.anyio
async def test_an_engine_rejection_keeps_its_status_for_a_non_streamed_chat(store, events):
    inf, _ = service(store, events, RejectingAdapter(), deployment())
    status, body = await inf.chat_complete(dict(BODY))
    assert status == 400 and "exceeds the available context size" in json.dumps(body)
    run = inf.runs(1)[0]
    assert (run.status, run.stream) == ("error", False) and "context size" in run.error


# --------------------------------------------------------------------------- 27: benchmark of a deployment that stops
class StoppableStreamAdapter(FakeAdapter):
    """Streams a token every 20 ms until cancelled; after stop() its port refuses connections."""

    def __init__(self) -> None:
        super().__init__()
        self.open: set[str] = set()
        self.cancelled: set[str] = set()
        self.down = False

    async def stream(self, handle, req):
        if self.down:
            raise httpx.ConnectError("All connection attempts failed")
        self.open.add(req.request_id)
        try:
            for i in range(25):
                if req.request_id in self.cancelled:
                    raise httpx.ReadError("upstream connection closed")
                yield event(f"t{i} ")
                await asyncio.sleep(0.02)
            yield event("end", "stop")
            yield DONE
        finally:
            self.open.discard(req.request_id)

    async def cancel(self, request_id):
        self.cancelled.add(request_id)

    async def stop(self, handle):
        self.down = True
        return await super().stop(handle)


@pytest.mark.anyio
async def test_a_benchmark_fails_when_its_deployment_stops(store, events, tmp_path):
    adapter = StoppableStreamAdapter()
    mgr, _, _ = await _setup(store, events, adapter, tmp_path)
    inf = InferenceService(store, events, mgr, FakeModels(), adapter)
    mgr.warmup = None
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    bench = BenchmarkService(store, events, mgr, inf)
    started = bench.start(BenchmarkRequest(prompt_set="short", warmup=0, repetitions=3, max_tokens=8))
    await until(lambda: adapter.open)
    await mgr.stop(dep.id)
    await until(lambda: bench.get(started.id).status != "running")
    result = bench.get(started.id)
    assert result.status == "failed" and dep.id[:8] in result.error and "no longer ready" in result.error
    assert len(result.run_ids) == 1  # the interrupted repetition; nothing was sent to the stopped engine


# --------------------------------------------------------------------------- 29: manifest paths
def _file(path: str) -> ModelFile:
    return ModelFile(name="x.json", url="https://x", sha256="0" * 64, size_bytes=1, path=path)


@pytest.mark.parametrize("path", ["/Users/Public/evil.json", "\\Windows\\Temp\\x.cmd", "D:evil.json", "C:/x/evil.json",
                                  "C:\\x\\evil.json", "\\\\server\\share\\x", "../x", "a/../../x", "a\\..\\..\\x"])
def test_manifest_paths_must_stay_inside_the_model_dir(tmp_path, path):
    svc = ModelService(None, tmp_path / "models")
    with pytest.raises(ModelError):
        svc._local("m", _file(path))


def test_plain_manifest_paths_and_ids_resolve_inside_the_models_dir(tmp_path):
    svc = ModelService(None, tmp_path / "models")
    got = svc._local("m", _file("sub/dir/model.safetensors"))
    assert got == tmp_path / "models" / "m" / "sub" / "dir" / "model.safetensors"
    with pytest.raises(ModelError):
        svc._local("../evil", _file("x.json"))


# --------------------------------------------------------------------------- 31: tuner trials
@pytest.mark.anyio
async def test_a_tuner_trial_is_never_the_ready_run(store, events, tmp_path):
    adapter = FakeAdapter()
    mgr, _, _ = await _setup(store, events, adapter, tmp_path)
    inf = InferenceService(store, events, mgr, FakeModels(), adapter)
    created = await mgr.create(CreateDeploymentRequest(plan_id="plan1", purpose="tune"), None)
    dep = await _wait_terminal_or_ready(mgr, created.id)
    assert dep.state == DeploymentState.ready and created.purpose == dep.purpose == "tune"
    assert [d.purpose for d in mgr.list()] == ["tune"]
    for refused in (mgr.ready, lambda: BenchmarkService(store, events, mgr, inf).start(BenchmarkRequest())):
        with pytest.raises(DeploymentError) as err:
            refused()
        assert (err.value.status, err.value.code) == (409, "tuning") and "tune" in str(err.value)
    with pytest.raises(DeploymentError) as err:
        await inf.chat_complete(dict(BODY))
    assert err.value.code == "tuning"
    assert mgr.handle(dep.id) is not None  # the tuner reaches its trial by id


def test_v1_models_chat_and_benchmarks_skip_a_tuner_trial(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    trial = deployment("trial").model_copy(update={"purpose": "tune"})
    coord.deployments._deps[trial.id] = trial
    coord.deployments._handles[trial.id] = DeploymentHandle("trial", "k", "http://127.0.0.1:1", Path("x"), 1, [])
    with open_client(coord) as client:
        try:
            assert client.get("/v1/models").json()["data"] == []
            r = client.post("/v1/chat/completions", json=BODY, headers=OWNER_HEADERS)
            assert r.status_code == 409 and r.json()["code"] == "tuning"
            r = client.post("/api/benchmarks", json={}, headers=OWNER_HEADERS)
            assert r.status_code == 409 and r.json()["code"] == "tuning"
        finally:
            coord.deployments._deps.pop(trial.id)  # the fake trial has no engine for shutdown to stop
