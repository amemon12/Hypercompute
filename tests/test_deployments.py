"""Deployment lifecycle with a fake engine adapter (protocol tests, no llama.cpp involved)."""

import asyncio

import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute.adapters.base import DeploymentHandle, EngineAdapterError, GenerationEvent, ValidationResult
from hypercompute.agent.supervisor import Supervisor
from hypercompute.schemas import (
    CreateDeploymentRequest,
    DeploymentState,
    DeviceAssignment,
    Plan,
    PlanStrategy,
    utcnow,
)
from hypercompute.services.deployments import DeploymentError, DeploymentManager
from hypercompute.services.leases import LeaseManager
from hypercompute.services.nodes import NodeRegistry
from hypercompute.services.sessions import SessionService
from hypercompute.services.streaming import InferenceService
from hypercompute.units import GiB

SPLIT_LOG = """load_tensors: offloaded 25/25 layers to GPU
load_tensors:        CUDA0 model buffer size =   129.39 MiB
load_tensors: RPC0[10.0.0.5:47052] model buffer size =   {rpc_mib} MiB
llama_kv_cache:      CUDA0 KV buffer size =    26.00 MiB
llama_kv_cache: RPC0[10.0.0.5:47052] KV buffer size =    22.00 MiB
llama_kv_cache: size =   48.00 MiB (  4096 cells,  24 layers,  1/1 seqs), K (f16):   24.00 MiB, V (f16):   24.00 MiB
"""


class FakeAdapter:
    def __init__(self, rpc_mib="244.33", ready=True, alive=True, warm_ok=True):
        self.rpc_mib, self.ready, self.alive, self.warm_ok = rpc_mib, ready, alive, warm_ok
        self.started, self.stopped = 0, 0

    async def validate(self, spec):
        return ValidationResult(ok=True, problems=[], rpc_device_map={"10.0.0.5:47052": ["RPC0"]})

    async def prepare(self, spec):
        self.started += 1
        return DeploymentHandle(deployment_id=spec.deployment_id, process_key="k", base_url="http://127.0.0.1:1",
                                log_path=None, pid=1, args=["--rpc", "10.0.0.5:47052"])  # type: ignore[arg-type]

    async def wait_ready(self, handle, timeout_s=0, on_progress=None):
        if not self.ready:
            raise EngineAdapterError("load_timeout", "llama-server was not ready")
        if on_progress:
            on_progress("model loaded")

    def read_log(self, handle):
        return SPLIT_LOG.format(rpc_mib=self.rpc_mib)

    async def observe(self, handle, spec, validation):  # same check as the llama.cpp adapter, on the fake log
        from hypercompute.adapters.base import PlacementReport
        from hypercompute.adapters.llama_output import check_participation, parse_placement

        placement = parse_placement(self.read_log(handle))
        ok, detail = check_participation(spec.plan, placement, validation.rpc_device_map)
        return PlacementReport(observed=placement, verified=ok, detail=detail)

    def process_alive(self, handle):
        return self.alive

    async def stop(self, handle):
        self.stopped += 1
        from hypercompute.adapters.base import StopResult

        return StopResult(clean=True, detail="terminated")

    async def stream(self, handle, req):
        if not self.warm_ok:
            raise EngineAdapterError("upstream_error", "boom")
        payload = {"choices": [{"delta": {"content": "ready"}, "finish_reason": "stop", "index": 0}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                   "timings": {"predicted_n": 1, "predicted_ms": 3.0, "cache_n": 0}}
        import json

        yield GenerationEvent(raw=b"data: x\n\n", data=json.dumps(payload), payload=payload)
        yield GenerationEvent(raw=b"data: [DONE]\n\n", data="[DONE]", payload=None, done=True)

    async def cancel(self, request_id):
        return None


def _plan(remote=True):
    a = [DeviceAssignment(node_id="local", device="CUDA0", endpoint=None, pool_id="vram:CUDA0", layers=list(range(13)),
                          holds_output=False, weight_bytes=GiB, kv_bytes=0, compute_bytes_est=0,
                          overhead_bytes_est=0, total_bytes_est=GiB, budget_bytes=8 * GiB, fits=True)]
    if remote:
        a.append(DeviceAssignment(node_id="a" * 32, device="RPC0", endpoint="10.0.0.5:47052", pool_id="uma",
                                  layers=list(range(13, 24)), holds_output=True, weight_bytes=GiB, kv_bytes=0,
                                  compute_bytes_est=0, overhead_bytes_est=0, total_bytes_est=GiB,
                                  budget_bytes=8 * GiB, fits=True))
    return Plan(id="plan1", model_id="qwen2.5-0.5b-instruct-q4_k_m", strategy=PlanStrategy.all_selected,
                ctx_size=4096, feasible=True, fit_label="estimated_fit", heuristic=True, assignments=a,
                device_list=[x.device for x in a], tensor_split=[13, 12] if remote else [25],
                rpc_endpoints=["10.0.0.5:47052"] if remote else [], cpu_resident_bytes=0, reasons=[], warnings=[],
                candidates=[], created_at=utcnow())


class FakeModels:
    def primary_path(self, model_id):
        from pathlib import Path

        return Path("model.gguf")

    def model_dir(self, model_id):
        from pathlib import Path

        return Path(".")

    def get(self, model_id):
        from hypercompute.schemas import ModelArtifact, ModelFile

        return ModelArtifact(id=model_id, display_name="m", family="qwen2", params_b=0.6, quantization="Q4_K_M",
                             repo="r", revision="x", license="apache-2.0",
                             files=[ModelFile(name="m.gguf", url="https://x", sha256="0" * 64, size_bytes=1)])

    def summary(self, model_id):
        return None

    def projector_path(self, model_id):  # no vision projector
        return None


async def _setup(store, events, adapter, tmp_path, node_state_eligible=True):
    svc = SessionService(store, events, b"k" * 32, 8000, lambda: ["10.0.0.2"])
    reg = NodeRegistry(store, events, svc, b"k" * 32, engine_info(), make_inventory(), contribution(["CUDA0"]))
    if node_state_eligible:
        from hypercompute.schemas import NodeState, NodeView, WorkerEndpoint, WorkerState

        view = NodeView(id="a" * 32, display_name="mac", is_local=False, session_id="s", state=NodeState.eligible,
                        state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(), heartbeat_age_s=0,
                        inventory=make_inventory("macos"), contribution=contribution(["MTL0"]),
                        worker=WorkerEndpoint(host="10.0.0.5", port=47052), worker_state=WorkerState.running,
                        verification=None)
        reg._save(view)
    leases = LeaseManager(store)
    mgr = DeploymentManager(store, events, reg, FakeModels(), leases, adapter, Supervisor(tmp_path / "m.json"))
    InferenceService(store, events, mgr, FakeModels(), adapter)
    store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?)",
                  ("plan1", utcnow().isoformat(), _plan().model_dump_json()))
    return mgr, reg, leases


async def _wait_terminal_or_ready(mgr, dep_id):
    for _ in range(1000):  # generous: the host may be busy with other work during the suite
        d = mgr.get(dep_id)
        if d.state in (DeploymentState.ready, DeploymentState.failed, DeploymentState.stopped):
            return d
        await asyncio.sleep(0.01)
    raise AssertionError("deployment did not settle")


@pytest.mark.anyio
async def test_ready_requires_load_participation_and_warmup(store, events, tmp_path):
    adapter = FakeAdapter()
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await mgr.create(CreateDeploymentRequest(plan_id="plan1"), "idem-1")
    dep = await _wait_terminal_or_ready(mgr, dep.id)
    assert dep.state == DeploymentState.ready
    states = [h.state for h in dep.history]
    assert states.index(DeploymentState.loading) < states.index(DeploymentState.warming) < states.index(
        DeploymentState.ready)
    assert dep.participation_verified and len(leases.active()) == 2
    # idempotency: same key returns the same deployment; another key is refused while one is active
    again = await mgr.create(CreateDeploymentRequest(plan_id="plan1"), "idem-1")
    assert again.id == dep.id
    with pytest.raises(DeploymentError) as e:
        await mgr.create(CreateDeploymentRequest(plan_id="plan1"), "idem-2")
    assert e.value.code == "deployment_active"
    stopped = await mgr.stop(dep.id)
    assert stopped.state == DeploymentState.stopped and leases.active() == [] and adapter.stopped == 1


@pytest.mark.anyio
async def test_peer_with_zero_layers_fails_deployment(store, events, tmp_path):
    adapter = FakeAdapter(rpc_mib="0.00")
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.failed and "received no layers" in dep.error
    assert leases.active() == [] and adapter.stopped == 1 and "released" in dep.cleanup


@pytest.mark.anyio
async def test_failed_warmup_is_not_ready(store, events, tmp_path):
    adapter = FakeAdapter(warm_ok=False)
    mgr, _, _ = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.failed and "warmup" in dep.error
    runs = store.query("SELECT status, kind FROM runs")
    assert [(r["kind"], r["status"]) for r in runs] == [("warmup", "error")]  # failed runs are kept


@pytest.mark.anyio
async def test_listening_but_not_ready_times_out_and_cleans_up(store, events, tmp_path):
    adapter = FakeAdapter(ready=False)
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.failed and adapter.stopped == 1 and leases.active() == []


@pytest.mark.anyio
async def test_node_loss_invalidates_ready_deployment(store, events, tmp_path):
    adapter = FakeAdapter()
    mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.ready and mgr.busy("a" * 32) == [dep.id]
    await mgr.on_node_lost("a" * 32, "no heartbeat for 16s")
    dep = mgr.get(dep.id)
    assert dep.state == DeploymentState.failed and "lost" in dep.error and leases.active() == []
    with pytest.raises(DeploymentError):
        mgr.ready()


@pytest.mark.anyio
async def test_ineligible_peer_blocks_deployment_before_launch(store, events, tmp_path):
    adapter = FakeAdapter()
    mgr, _, leases = await _setup(store, events, adapter, tmp_path, node_state_eligible=False)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.failed and adapter.started == 0 and leases.active() == []


@pytest.mark.anyio
async def test_restart_reconciliation_marks_active_failed(store, events, tmp_path):
    adapter = FakeAdapter()
    mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    # a new manager over the same store simulates a gateway restart
    fresh = DeploymentManager(store, events, reg, FakeModels(), leases, adapter, Supervisor(tmp_path / "m2.json"))
    notes = fresh.reconcile_on_start()
    assert fresh.get(dep.id).state == DeploymentState.failed
    assert leases.active() == [] and any("released" in n for n in notes)


class SlowStreamAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.upstream_closed = False

    async def stream(self, handle, req):
        try:
            for i in range(100):
                payload = {"choices": [{"delta": {"content": f"t{i} "}, "finish_reason": None, "index": 0}]}
                yield GenerationEvent(raw=b"data: x\n\n", data="x", payload=payload)
                await asyncio.sleep(0.01)
        finally:
            self.upstream_closed = True


@pytest.mark.anyio
async def test_client_disconnect_closes_upstream_and_records_cancel(store, events, tmp_path):
    adapter = SlowStreamAdapter()
    mgr, _, _ = await _setup(store, events, adapter, tmp_path)
    inf = InferenceService(store, events, mgr, FakeModels(), adapter)
    mgr.warmup = None  # skip warmup so the slow adapter is only used for the chat below
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.ready
    gen = inf.chat_stream({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    first = await gen.__anext__()
    assert first.startswith(b"data:")
    await gen.aclose()  # what a disconnected client looks like once the response is finalized
    assert adapter.upstream_closed
    run = inf.runs(1)[0]
    assert run.status == "cancelled" and run.output_text.startswith("t0")
    # the inference slot was released
    hold = await asyncio.wait_for(inf.acquire(dep, exclusive=True, timeout=1), timeout=2)
    hold.release()
