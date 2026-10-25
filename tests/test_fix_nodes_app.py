"""Regression tests: node rechecks and loss handling, node-id takeover, benchmark final status, run identity."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest
from hypercompute.adapters.base import CompletionResult, GenerationEvent
from hypercompute.adapters.llama_rpc import ProbeResult
from hypercompute.schemas import (
    BenchmarkRequest,
    Deployment,
    DeploymentState,
    DeviceAssignment,
    EngineDevice,
    GgufSummary,
    HeartbeatRequest,
    ModelArtifact,
    ModelFile,
    NodeState,
    Plan,
    PlanStrategy,
    WorkerState,
    utcnow,
)
from hypercompute.services.benchmarks import BenchmarkService
from hypercompute.services.sessions import JoinRejected
from hypercompute.services.streaming import InferenceService
from test_sessions_and_join import join_request, registry, sessions

NODE = "a" * 32


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def running_hb() -> HeartbeatRequest:
    return HeartbeatRequest(worker_state=WorkerState.running, sent_at=utcnow())


class ToggleProbe:
    def __init__(self, ok: bool) -> None:
        self.ok, self.calls = ok, 0

    async def __call__(self, endpoints):
        self.calls += 1
        if not self.ok:
            return ProbeResult(False, [], {endpoints[0]: []}, "a worker endpoint refused the RPC connection")
        dev = EngineDevice(name="RPC0", description=endpoints[0], total_bytes=16 << 30, free_bytes=10 << 30)
        return ProbeResult(True, [dev], {endpoints[0]: [dev]}, "worker answered the llama.cpp RPC handshake")


# ---------------------------------------------------------------------- finding 3: stuck in 'verifying'
@pytest.mark.anyio
async def test_verifying_node_is_rechecked_once_its_deployment_ends(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    busy: list[str] = []
    reg.busy = lambda node_id: list(busy)
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    await settle()
    assert reg.get(j.node_id).state == NodeState.eligible
    busy.append("dep1")
    await reg.check_staleness(utcnow() + timedelta(seconds=30))  # Wi-Fi drop while serving
    await reg.heartbeat(j.node_id, running_hb())  # heartbeat resumes before the deployment cleanup ended
    await settle()
    assert reg.get(j.node_id).state == NodeState.verifying  # verify skipped: the node was still busy
    busy.clear()
    for _ in range(3):
        await reg.heartbeat(j.node_id, running_hb())
    assert reg.schedule_rechecks() == [j.node_id]
    await settle()
    assert reg.get(j.node_id).state == NodeState.eligible


@pytest.mark.anyio
async def test_recheck_skips_busy_and_in_flight_nodes(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    reg.busy = lambda node_id: ["dep1"]
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    await settle()
    assert reg.get(j.node_id).state == NodeState.verifying
    assert reg.schedule_rechecks() == []  # never probe a worker that serves a deployment


# ---------------------------------------------------------------------- finding 5: unreachable re-probe
@pytest.mark.anyio
async def test_unreachable_worker_is_reprobed_on_a_backoff_not_every_heartbeat(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    probe = ToggleProbe(ok=False)
    reg.probe = probe
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", "10.0.0.2:8000")
    await settle()
    assert reg.get(j.node_id).state == NodeState.ineligible and probe.calls == 1
    for _ in range(4):
        resp = await reg.heartbeat(j.node_id, running_hb())
        assert resp.state == NodeState.ineligible and resp.state_reason == "worker unreachable from host"
    await settle()
    assert probe.calls == 1
    assert reg.schedule_rechecks() == []  # too soon after the last probe
    probe.ok = True  # the friend fixed their firewall
    assert reg.schedule_rechecks(utcnow() + timedelta(seconds=31)) == [j.node_id]
    await settle()
    assert probe.calls == 2 and reg.get(j.node_id).state == NodeState.eligible


@pytest.mark.anyio
async def test_worker_restart_still_triggers_a_recheck(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    probe = ToggleProbe(ok=True)
    reg.probe = probe
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    await settle()
    exited = HeartbeatRequest(worker_state=WorkerState.exited, worker_exit_code=1, sent_at=utcnow())
    assert (await reg.heartbeat(j.node_id, exited)).state == NodeState.ineligible
    resp = await reg.heartbeat(j.node_id, running_hb())
    assert resp.state == NodeState.verifying and resp.state_reason == "worker running again; re-checking"
    await settle()
    assert probe.calls == 2 and reg.get(j.node_id).state == NodeState.eligible


@pytest.mark.anyio
async def test_heartbeat_updates_the_extra_engines_the_host_plans_with(store, events):
    """EXO can crash or be paused after join; the host must stop planning onto it on the next heartbeat."""
    from hypercompute.schemas import ExtraEngineInfo

    svc = sessions(store, events)
    reg = registry(store, events, svc)
    reg.probe = ToggleProbe(ok=True)
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    await settle()
    running = ExtraEngineInfo(engine="exo", installed=True, status="running", endpoint="10.0.0.5:52415")
    stopped = running.model_copy(update={"status": "installed", "endpoint": None})
    await reg.heartbeat(j.node_id, running_hb().model_copy(update={"extra_engines": [running]}))
    assert reg.get(j.node_id).inventory.extra_engines == [running]
    await reg.heartbeat(j.node_id, running_hb())  # an older agent sends no field: keep what the host knows
    assert reg.get(j.node_id).inventory.extra_engines == [running]
    await reg.heartbeat(j.node_id, running_hb().model_copy(update={"extra_engines": [stopped]}))
    assert reg.get(j.node_id).inventory.extra_engines == [stopped]


# ---------------------------------------------------------------------- non-blocking node loss
@pytest.mark.anyio
async def test_heartbeat_does_not_wait_for_loss_cleanup_and_never_runs_it_twice(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    release = asyncio.Event()
    calls: list[str] = []

    async def on_lost(node_id, reason):
        calls.append(reason)
        await release.wait()  # a llama-server that hangs on the dead socket

    reg.on_lost = on_lost
    reg.busy = lambda node_id: ["dep1"]
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    exited = HeartbeatRequest(worker_state=WorkerState.exited, worker_exit_code=1, sent_at=utcnow())
    resp = await asyncio.wait_for(reg.heartbeat(j.node_id, exited), timeout=1)
    assert resp.state == NodeState.ineligible
    await asyncio.wait_for(reg.heartbeat(j.node_id, exited), timeout=1)  # the agent's retry
    await settle()
    assert len(calls) == 1
    leave = asyncio.create_task(reg.leave(j.node_id, by_owner=False))
    await settle()
    assert not leave.done()  # leave() still waits for the cleanup in flight
    release.set()
    await asyncio.wait_for(leave, timeout=1)
    assert len(calls) == 1 and reg.get(j.node_id).state == NodeState.left


@pytest.mark.anyio
async def test_leave_awaits_loss_handling(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    calls: list[str] = []

    async def on_lost(node_id, reason):
        await asyncio.sleep(0.01)
        calls.append(reason)

    reg.on_lost = on_lost
    reg.busy = lambda node_id: ["dep1"]
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    await reg.leave(j.node_id, by_owner=True)
    assert calls == ["removed by host"]


# ---------------------------------------------------------------------- finding 9: node id takeover
@pytest.mark.anyio
async def test_join_cannot_take_over_another_machines_node_id(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    sid = svc.create("d").id
    a = await reg.join(join_request(svc.issue_invite(sid).invite.code, node_id=NODE), "10.0.0.5", None)
    with pytest.raises(JoinRejected) as e:
        await reg.join(join_request(svc.issue_invite(sid).invite.code, node_id=NODE, worker_ip="10.0.0.66"),
                       "10.0.0.66", None)
    assert e.value.status == 409 and e.value.code == "node_id_in_use"
    reg.authenticate(NODE, f"Bearer {a.credential}")  # A keeps its credential and its worker
    assert reg.get(NODE).worker.host == "10.0.0.5"
    again = await reg.join(join_request(svc.issue_invite(sid).invite.code, node_id=NODE), "10.0.0.5", None)
    reg.authenticate(NODE, f"Bearer {again.credential}")  # the same machine may rejoin after an agent restart
    await reg.leave(NODE, by_owner=False)
    other = await reg.join(join_request(svc.issue_invite(sid).invite.code, node_id=NODE, worker_ip="10.0.0.66"),
                           "10.0.0.66", None)
    assert other.node_id == NODE  # a node id that left is free again


# ---------------------------------------------------------------------- finding 4: benchmark final status
class BenchDeps:
    def __init__(self) -> None:
        self.dep = SimpleNamespace(id="dep1", placement=None, plan=SimpleNamespace(model_id="m",
                                                                                 device_list=["CUDA0"],
                                                                                 engine="llama_rpc", cpu_layers=0))

    def ready(self):
        return self.dep, object()


class BenchInference:
    def __init__(self, collect) -> None:
        self._collect = collect

    async def acquire(self, dep, exclusive: bool = False, timeout: float = 30.0):
        assert exclusive, "benchmarks hold every inference slot"
        return self

    def release(self) -> None:
        return None

    async def collect(self, *args):
        return await self._collect()


async def wait_until_not_running(svc: BenchmarkService, bench_id: str):
    for _ in range(200):
        result = svc.get(bench_id)
        if result.status != "running":
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("benchmark still running")


async def hang():
    await asyncio.Event().wait()


@pytest.mark.anyio
async def test_benchmark_records_failure_for_unexpected_errors(store, events):
    async def closed_client():
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    svc = BenchmarkService(store, events, BenchDeps(), BenchInference(closed_client))
    started = svc.start(BenchmarkRequest())
    result = await wait_until_not_running(svc, started.id)
    assert result.status == "failed" and "client has been closed" in result.error


@pytest.mark.anyio
async def test_benchmark_shutdown_records_cancellation(store, events):
    svc = BenchmarkService(store, events, BenchDeps(), BenchInference(hang))
    started = svc.start(BenchmarkRequest())
    await settle()
    await svc.shutdown()
    result = svc.get(started.id)
    assert result.status == "failed" and "cancelled" in result.error


@pytest.mark.anyio
async def test_running_benchmarks_are_failed_after_restart(store, events):
    before = BenchmarkService(store, events, BenchDeps(), BenchInference(hang))
    started = before.start(BenchmarkRequest())
    await settle()
    after = BenchmarkService(store, events, BenchDeps(), BenchInference(hang))
    notes = after.reconcile_on_start()
    assert len(notes) == 1
    result = after.get(started.id)
    assert result.status == "failed" and result.error == "gateway restarted"
    await before.shutdown()


# ---------------------------------------------------------------------- finding 29: run identity
class IdentityModels:
    def get(self, model_id):
        return ModelArtifact(id=model_id, display_name="m", family="qwen3", params_b=0.6, quantization="Q4_K_M",
                             repo="r", revision="x", license="apache-2.0", max_output_tokens=512,
                             files=[ModelFile(name="m.gguf", url="https://x", sha256="0" * 64, size_bytes=1)])

    def summary(self, model_id):
        return GgufSummary(architecture="qwen3", name="m", file_type=None, n_layer=28, n_embd=1024, n_head=16,
                           n_head_kv=8, context_length=40960, vocab_size=151936, tensor_count=1,
                           total_tensor_bytes=1, layer_bytes=[1], token_embd_bytes=1, output_bytes=1,
                           output_tied=True, other_bytes=0, chat_template_present=True)


class SlowAdapter:
    def __init__(self) -> None:
        self.called_at = None

    async def stream(self, handle, req):
        self.called_at = utcnow()
        await asyncio.sleep(0.05)
        payload = {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop", "index": 0}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        yield GenerationEvent(raw=b"data: x\n\n", data="x", payload=payload)
        yield GenerationEvent(raw=b"data: [DONE]\n\n", data="[DONE]", payload=None, done=True)

    async def complete(self, handle, req):
        self.called_at = utcnow()
        await asyncio.sleep(0.05)
        return CompletionResult(200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    async def cancel(self, request_id):
        return None


def assignment(node_id: str, device: str, endpoint: str | None, layers: list[int]) -> DeviceAssignment:
    return DeviceAssignment(node_id=node_id, device=device, endpoint=endpoint, pool_id="p", layers=layers,
                            holds_output=False, weight_bytes=1, kv_bytes=1, compute_bytes_est=1,
                            overhead_bytes_est=1, total_bytes_est=4, budget_bytes=8, fits=True)


def two_node_deployment(model_id: str) -> Deployment:
    plan = Plan(id="p1", model_id=model_id, strategy=PlanStrategy.all_selected, ctx_size=4096, feasible=True,
                fit_label="estimated_fit", heuristic=True,
                assignments=[assignment("local", "CUDA0", None, [0, 1, 2]),
                             assignment(NODE, "RPC0", "10.0.0.5:47052", [3, 4])],
                device_list=["CUDA0", "RPC0"], tensor_split=[3, 2], rpc_endpoints=["10.0.0.5:47052"],
                cpu_resident_bytes=0, reasons=[], warnings=[], candidates=[], created_at=utcnow())
    return Deployment(id="dep1", plan=plan, state=DeploymentState.ready, history=[], created_at=utcnow())


async def identity_setup(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    await reg.join(join_request(svc.create("d").invite.code, node_id=NODE), "10.0.0.5", None)
    dep = two_node_deployment("qwen3-0.6b")
    adapter = SlowAdapter()
    deps = SimpleNamespace(ready=lambda: (dep, None))
    return InferenceService(store, events, deps, IdentityModels(), adapter, nodes=reg), dep, adapter


@pytest.mark.anyio
async def test_run_record_names_hosts_model_shape_and_start_time(store, events):
    inf, dep, adapter = await identity_setup(store, events)
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "hi"}]}
    run = await inf.collect(dep, None, body, "chat")
    assert run.status == "ok"
    assert run.started_at <= adapter.called_at  # request start, not the time the stream ended
    hosts = {h["node_id"]: h for h in inf.runs(1)[0].engine["hosts"]}
    remote, local = hosts[NODE], hosts["local"]
    assert remote["display_name"] == "friend-mac" and remote["os"]["system"] == "macos"
    assert remote["engine"]["release"] == "b11770" and remote["engine"]["rpc_proto"] == "8.0.0"
    assert remote["devices"] == [{"device": "RPC0", "endpoint": "10.0.0.5:47052", "layers": 2}]
    assert remote["contributed_devices"] == ["MTL0"] and remote["hostname"] == "macos-fixture"
    assert local["display_name"].endswith("(this machine)") and local["devices"][0]["device"] == "CUDA0"
    assert run.model["architecture"] == "qwen3" and run.model["chat_template"] is True
    assert run.model["context_length"] == 40960 and run.model["max_output_tokens"] == 512


@pytest.mark.anyio
async def test_non_streamed_run_uses_request_start_time(store, events):
    inf, _, adapter = await identity_setup(store, events)
    status, _ = await inf.chat_complete({"messages": [{"role": "user", "content": "hi"}]})
    run = inf.runs(1)[0]
    assert status == 200 and run.started_at <= adapter.called_at
    assert {h["node_id"] for h in run.engine["hosts"]} == {"local", NODE}
