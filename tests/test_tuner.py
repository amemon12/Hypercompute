"""Measured split tuner and profile reuse, with fake deployments and inference (no engine, no GPU)."""

import asyncio
import json
import time
import uuid
from types import SimpleNamespace

import pytest
from factories import contribution, make_inventory
from hypercompute import config
from hypercompute.adapters.base import DeploymentHandle, GenerationEvent, PlacementReport, StopResult, ValidationResult
from hypercompute.agent.supervisor import Supervisor
from hypercompute.schemas import (
    TERMINAL_STATES,
    Deployment,
    DeploymentState,
    DeviceAssignment,
    EngineDevice,
    EngineTimings,
    ModelArtifact,
    ModelFile,
    NodeState,
    NodeView,
    Plan,
    PlanStrategy,
    RunRecord,
    TuneCandidate,
    TuneRequest,
    TuneResult,
    VerificationResult,
    WorkerEndpoint,
    WorkerState,
    utcnow,
)
from hypercompute.services import planner
from hypercompute.services import tuner as tuner_mod
from hypercompute.services.deployments import (
    DeploymentError,
    DeploymentManager,
    local_budgets,
    remote_budgets,
    warmup_timeout,
)
from hypercompute.services.leases import LeaseManager
from hypercompute.services.models import ModelError
from hypercompute.services.nodes import NodeRegistry
from hypercompute.services.profiles import ProfileStore, signature
from hypercompute.services.selector import profile_devices
from hypercompute.services.sessions import SessionService
from hypercompute.services.streaming import InferenceService
from hypercompute.services.tuner import TunerService, decode_rate, describe, prefill_rate, score
from hypercompute.units import GiB

MODEL = "qwen2.5-7b-instruct-q4_k_m"
PEER = "a" * 32
EP = "10.0.0.5:50052"
HANDLE = DeploymentHandle(deployment_id="d", process_key="k", base_url="http://127.0.0.1:1", log_path=None,  # type: ignore[arg-type]
                          pid=1, args=[])


# --------------------------------------------------------------------------- builders


def asg(node, device, layers, *, head=False, total=GiB, budget=8 * GiB, endpoint=None, pool="vram:CUDA0"):
    return DeviceAssignment(node_id=node, device=device, endpoint=endpoint, pool_id=pool, layers=list(layers),
                            holds_output=head, weight_bytes=total, kv_bytes=0, compute_bytes_est=0,
                            overhead_bytes_est=0, total_bytes_est=total, budget_bytes=budget, fits=True)


def mkplan(assignments, strategy=PlanStrategy.auto, cpu_layers=0, feasible=True, engine="llama_rpc", capacity=False):
    accel = [a for a in assignments if a.device != "CPU"]
    return Plan(id=uuid.uuid4().hex, model_id=MODEL, strategy=strategy, ctx_size=4096, feasible=feasible,
                fit_label="estimated_fit" if feasible else "does_not_fit", heuristic=True, assignments=assignments,
                device_list=[a.device for a in accel], tensor_split=[len(a.layers) + a.holds_output for a in accel],
                rpc_endpoints=sorted({a.endpoint for a in assignments if a.endpoint}), cpu_resident_bytes=0,
                reasons=[] if feasible else ["CUDA0 over budget"], warnings=[], candidates=[], created_at=utcnow(),
                engine=engine, tier="gpu_ram" if cpu_layers else "gpu", cpu_layers=cpu_layers,
                n_gpu_units=(29 - cpu_layers) if cpu_layers else None, capacity_mode=capacity)


LOCAL_FIT = mkplan([asg("local", "CUDA0", range(28), head=True, total=6 * GiB, budget=10 * GiB)])
PEER_SPLIT = mkplan([asg(PEER, "RPC0", range(6), endpoint=EP, pool="uma", total=3 * GiB),
                     asg("local", "CUDA0", range(6, 28), head=True, total=int(9.5 * GiB), budget=10 * GiB)])
RAM_SPLIT = mkplan([asg("local", "CPU", range(4), pool="ram", total=2 * GiB, budget=12 * GiB),
                    asg("local", "CUDA0", range(4, 28), head=True, total=9 * GiB, budget=10 * GiB)],
                   strategy=PlanStrategy.local_only, cpu_layers=4)
HEAD_ON_PEER = mkplan([asg("local", "CUDA0", range(20), total=7 * GiB, budget=10 * GiB),
                       asg(PEER, "RPC0", range(20, 28), head=True, endpoint=EP, pool="uma", total=3 * GiB)],
                      strategy=PlanStrategy.all_selected)


def node_views():
    local = NodeView(id="local", display_name="g14 (this machine)", is_local=True, session_id=None,
                     state=NodeState.eligible, state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(),
                     heartbeat_age_s=0, inventory=make_inventory(), contribution=contribution(["CUDA0"]),
                     worker=None, worker_state=None, verification=None)
    ver = VerificationResult(checked_at=utcnow(), reachable=True, engine_match=True, detail="ok",
                             rpc_devices=[EngineDevice(name="RPC0", description=EP, total_bytes=12 * GiB,
                                                       free_bytes=10 * GiB)])
    peer = NodeView(id=PEER, display_name="mac", is_local=False, session_id="s", state=NodeState.eligible,
                    state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(), heartbeat_age_s=0,
                    inventory=make_inventory("macos"), contribution=contribution(["MTL0"]),
                    worker=WorkerEndpoint(host="10.0.0.5", port=50052), worker_state=WorkerState.running,
                    verification=ver)
    return local, peer


class FakeNodes:
    def __init__(self):
        self.local, self.peer = node_views()

    def views(self):
        return [self.local, self.peer]

    def eligible_remote(self, node_ids):
        return [self.peer] if node_ids is None or PEER in node_ids else []


def run_record(dep, label, ttft, decode, prefill, timings=True):
    t = EngineTimings(prompt_n=1000, prompt_ms=1000 * 1000 / prefill, prompt_per_second=prefill,
                      predicted_n=64, predicted_ms=63 * 1000 / decode, predicted_per_second=decode)
    return RunRecord(id=uuid.uuid4().hex, attempt_id="x", deployment_id=dep.id, kind="benchmark", status="ok",
                     model={}, engine={}, placement=[], ctx_size=4096, stream=True, prompt_tokens=1000,
                     completion_tokens=64, ttft_ms=ttft, total_ms=ttft + 63 * 1000 / decode,
                     timings=t if timings else None, finish_reason="length", sampling={}, output_text="x",
                     started_at=utcnow(), prompt_label=label)


class FakeDeployments:
    def __init__(self, store, plans, fail=lambda plan: None, adapter=None):
        self.store, self.plans, self.fail = store, plans, fail
        self.adapter = adapter  # what adapter_for returns; None: a llama-server-like one
        self.deps: dict[str, Deployment] = {}
        self.log: list[str] = []
        self.requests = []
        self.current: str | None = None
        self.external: Deployment | None = None  # a deployment someone else started

    def plan(self, req):
        self.requests.append(req)
        key = req.overflow if req.overflow != "auto" else req.strategy  # the tuner forces each overflow tier
        return self.plans[key].model_copy(update={"id": uuid.uuid4().hex, "created_at": utcnow()})

    def active(self):
        return self.external or (self.deps.get(self.current) if self.current else None)

    async def create(self, req, key):
        if self.active() is not None:
            raise DeploymentError(409, "deployment_active", "a deployment is active; stop it first")
        row = self.store.query_one("SELECT doc FROM plans WHERE id=?", (req.plan_id,))
        plan = Plan.model_validate(json.loads(row["doc"]))
        assert (utcnow() - plan.created_at).total_seconds() < 600  # what DeploymentManager.stored_plan enforces
        error = self.fail(plan)
        state = DeploymentState.failed if error else DeploymentState.loading
        dep = Deployment(id=uuid.uuid4().hex, plan=plan, state=state, history=[], error=error, created_at=utcnow(),
                         purpose=req.purpose)
        self.deps[dep.id] = dep
        self.current = None if error else dep.id
        self.log.append(f"create {describe(plan)}")
        return dep

    def get(self, dep_id):
        dep = self.deps[dep_id]
        if dep.state == DeploymentState.loading:  # becomes ready on the next poll
            dep = self.deps[dep_id] = dep.model_copy(update={"state": DeploymentState.ready})
        return dep

    def handle(self, dep_id):
        return None if self.deps[dep_id].state in TERMINAL_STATES else HANDLE

    def adapter_for(self, engine_id):
        return self.adapter or TimingAdapter()

    def ready(self):  # what chat and benchmarks use: a tuner trial is never offered to them
        dep = self.active()
        if dep is None or dep.state != DeploymentState.ready or dep.purpose == "tune":
            raise DeploymentError(503, "no_ready_deployment", "no model is ready")
        return dep, HANDLE

    async def stop(self, dep_id, reason=""):
        self.log.append("stop")
        dep = self.deps[dep_id]
        if dep.state not in TERMINAL_STATES:
            dep = self.deps[dep_id] = dep.model_copy(update={"state": DeploymentState.stopped})
        if self.current == dep_id:
            self.current = None
        return dep


class FakeInference:
    def __init__(self, rates, block=None, on_collect=None):
        self.rates, self.block, self.on_collect = rates, block, on_collect
        self.held = False
        self.calls = []
        self.entered = asyncio.Event()

    async def acquire(self, dep, exclusive=False, timeout=30.0):
        assert exclusive and not self.held, "the tuner must hold every inference slot alone"
        self.held = True
        return self

    def release(self):
        self.held = False

    async def collect(self, dep, handle, body, kind, benchmark_id=None, prompt_label=None):
        assert self.held
        self.calls.append((prompt_label, body, kind, benchmark_id))
        self.entered.set()
        if self.on_collect:
            self.on_collect(dep)
        if self.block is not None:
            await self.block.wait()
        return run_record(dep, prompt_label, *self.rates(dep.plan))


def overflow_plans():
    return {PlanStrategy.auto: PEER_SPLIT, "host_ram": RAM_SPLIT, "peers": PEER_SPLIT}


def rates(plan):
    """A: fast first token, slow prefill. B: slow first token, fast prefill (interactive vs throughput winner)."""
    if plan.cpu_layers:
        return 2000.0, 38.0, 2000.0  # B
    if plan.engine_flags.get("ubatch") == 1024:
        return 150.0, 35.0, 300.0
    return 100.0, 40.0, 200.0  # A


def service(store, events, deps, inf, nodes=None):
    svc = TunerService(store, events, deps, inf, ProfileStore(store), None, nodes or FakeNodes())
    svc.poll_s = 0.001
    return svc


async def settle(svc, tune_id, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s  # wall clock: short sleeps can return at once on Windows
    while time.monotonic() < deadline:
        result = svc.get(tune_id)
        if result.status != "running":
            return result
        await asyncio.sleep(0.005)
    raise AssertionError("tune did not finish")


# --------------------------------------------------------------------------- candidate generation


def test_model_that_fits_locally_gets_only_flag_variants(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    svc = service(store, events, deps, FakeInference(rates))
    picked = svc.candidate_plans(TuneRequest(model_id=MODEL, max_candidates=4))
    assert [r.strategy for r in deps.requests] == [PlanStrategy.auto]  # no split variants: they only add latency
    assert deps.requests[0].use_profiles is False
    flags = [p.engine_flags for _, p in picked]
    # micro-batch 512 is llama-server's default, so it collapses into the planner pick; safest memory first
    assert flags == [{"ubatch": 256}, {}, {"flash_attn": "off"}, {"ubatch": 1024}]
    assert all(p.device_list == ["CUDA0"] and p.tensor_split == [29] for _, p in picked)


def test_no_flags_keeps_only_the_planner_pick(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    picked = service(store, events, deps, FakeInference(rates)).candidate_plans(
        TuneRequest(model_id=MODEL, tune_flags=False))
    assert len(picked) == 1 and picked[0][0].startswith("planner pick (gpu)")


def test_overflow_adds_ram_tier_and_peers_deduped_and_ordered(store, events):
    deps = FakeDeployments(store, overflow_plans())
    picked = service(store, events, deps, FakeInference(rates)).candidate_plans(
        TuneRequest(model_id=MODEL, node_ids=[PEER], max_candidates=4))
    by_overflow = {r.overflow: r for r in deps.requests}
    assert by_overflow["host_ram"].allow_ram_tier is True and by_overflow["host_ram"].strategy == PlanStrategy.auto
    assert by_overflow["peers"].allow_ram_tier is False and by_overflow["peers"].strategy == PlanStrategy.auto
    assert all(r.allow_capacity_mode is False and r.engine == "llama_rpc" and r.node_ids == [PEER]
               for r in deps.requests[1:])
    labels = [label.split(":")[0] for label, _ in picked]
    # the forced peer overflow has the planner pick's layout, so it is dropped; RAM split is the most conservative
    assert labels == ["host RAM overflow (gpu_ram)", "planner pick (gpu)", "flash attention off", "micro-batch 1024"]
    assert picked[0][1].cpu_layers == 4


def test_head_on_peer_variant_is_never_added(store, events):
    plans = {**overflow_plans(), "peers": HEAD_ON_PEER}
    picked = service(store, events, FakeDeployments(store, plans), FakeInference(rates)).candidate_plans(
        TuneRequest(model_id=MODEL, tune_flags=False, max_candidates=8))
    assert all(not (a.holds_output and a.endpoint) for _, p in picked for a in p.assignments)


@pytest.mark.parametrize("engine,planned", [(None, "llama_rpc"), ("llama_rpc", "llama_rpc"), ("vllm", "vllm")])
def test_tune_plans_llama_cpp_unless_the_request_names_an_engine(store, events, engine, planned):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    service(store, events, deps, FakeInference(rates)).candidate_plans(TuneRequest(model_id=MODEL, engine=engine))
    assert deps.requests[0].engine == planned  # never the selector's pick (vLLM for throughput, EXO on Macs)


def test_candidates_keep_the_owners_plan_limits(store, events):
    deps = FakeDeployments(store, {**overflow_plans(), PlanStrategy.fastest_fit: PEER_SPLIT})
    picked = service(store, events, deps, FakeInference(rates)).candidate_plans(TuneRequest(
        model_id=MODEL, node_ids=[PEER], strategy=PlanStrategy.fastest_fit, allow_ram_tier=False,
        allow_capacity_mode=True))
    base, *alts = deps.requests
    assert base.strategy == PlanStrategy.fastest_fit and base.allow_ram_tier is False and base.allow_capacity_mode
    assert [r.overflow for r in alts] == ["peers"]  # no host RAM candidate: the owner turned the RAM tier off
    assert all(r.strategy == PlanStrategy.fastest_fit and r.allow_ram_tier is False for r in alts)
    assert picked and all(p.cpu_layers == 0 for _, p in picked)


def test_a_local_only_tune_is_keyed_like_a_local_only_plan(store, events):
    svc = service(store, events, FakeDeployments(store, {}), FakeInference(rates))
    local, peer = node_views()
    for strategy, remote in ((PlanStrategy.local_only, []), (PlanStrategy.auto, remote_budgets(peer))):
        sig = svc.profile_signature(TuneRequest(model_id=MODEL, node_ids=[PEER], strategy=strategy))
        assert sig == signature(MODEL, 4096, profile_devices(local_budgets(local), remote))


@pytest.mark.anyio
async def test_start_refuses_infeasible_or_capacity_mode_plan(store, events):
    for plan, code in ((mkplan(LOCAL_FIT.assignments, feasible=False), "plan_not_feasible"),
                       (LOCAL_FIT.model_copy(update={"capacity_mode": True, "engine": "airllm"}), "capacity_mode")):
        svc = service(store, events, FakeDeployments(store, {PlanStrategy.auto: plan}), FakeInference(rates))
        with pytest.raises(DeploymentError) as err:
            await svc.start(TuneRequest(model_id=MODEL))
        assert err.value.status == 422 and err.value.code == code


def _manifest(fmt):
    def get(model_id):
        if fmt is None:
            raise ModelError(f"unknown model {model_id!r}")
        return ModelArtifact(id=model_id, display_name="m", family="qwen2", params_b=7.6, quantization="Q4_K_M",
                             repo="r", revision="x", license="apache-2.0", format=fmt, size_bytes=GiB,
                             files=[] if fmt == "mlx" else [ModelFile(name="m", url="https://x", sha256="0" * 64,
                                                                      size_bytes=1)])
    return SimpleNamespace(get=get)


@pytest.mark.anyio
@pytest.mark.parametrize("fmt,engine", [("hf", "vllm"), ("mlx", "exo"), ("gguf", None), (None, None)])
async def test_a_model_llama_cpp_cannot_serve_gets_a_fix_naming_gguf_and_its_engine(store, events, fmt, engine):
    refused = mkplan([], feasible=False)  # selector._llama: llama.cpp serves GGUF files and the model is not one
    svc = TunerService(store, events, FakeDeployments(store, {PlanStrategy.auto: refused}), FakeInference(rates),
                       ProfileStore(store), _manifest(fmt), FakeNodes())
    with pytest.raises(DeploymentError) as err:
        await svc.start(TuneRequest(model_id=MODEL))
    if engine is None:  # a GGUF (or unknown) model: downloading it or a peer can make it fit
        assert err.value.fix == "check the plan in the dashboard: download the model or add a peer"
    else:
        assert err.value.fix == (f"llama.cpp needs a GGUF file: tune a GGUF variant of {MODEL}, or tune it with "
                                 f"engine {engine} (--engine {engine})")


# --------------------------------------------------------------------------- running


@pytest.mark.anyio
async def test_start_refuses_while_a_deployment_is_active(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    deps.external = Deployment(id="e" * 32, plan=LOCAL_FIT, state=DeploymentState.ready, history=[],
                               created_at=utcnow())
    with pytest.raises(DeploymentError) as err:
        await service(store, events, deps, FakeInference(rates)).start(TuneRequest(model_id=MODEL))
    assert err.value.status == 409 and err.value.code == "deployment_active"


@pytest.mark.anyio
async def test_start_refuses_while_another_tune_runs(store, events):
    block = asyncio.Event()
    inf = FakeInference(rates, block=block)
    svc = service(store, events, FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT}), inf)
    first = await svc.start(TuneRequest(model_id=MODEL))
    await inf.entered.wait()
    with pytest.raises(DeploymentError) as err:
        await svc.start(TuneRequest(model_id=MODEL))
    assert err.value.status == 409 and err.value.code == "tune_running"
    await svc.cancel(first.id)


@pytest.mark.anyio
async def test_decode_tokens_must_fit_the_context(store, events):
    svc = service(store, events, FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT}), FakeInference(rates))
    with pytest.raises(DeploymentError) as err:
        await svc.start(TuneRequest(model_id=MODEL, ctx_size=256, decode_tokens=512))
    assert err.value.status == 422


@pytest.mark.anyio
@pytest.mark.parametrize("objective,winner", [("interactive", 1), ("throughput", 0)])
async def test_sequential_run_stops_every_candidate_and_saves_profile(store, events, objective, winner):
    deps = FakeDeployments(store, overflow_plans(),
                           fail=lambda p: "CUDA0 out of memory" if p.engine_flags.get("flash_attn") == "off" else None)
    inf = FakeInference(rates)
    svc = service(store, events, deps, inf)
    started = await svc.start(TuneRequest(model_id=MODEL, objective=objective, decode_tokens=64))
    assert started.status == "running" and all(c.status == "pending" for c in started.candidates)
    result = await settle(svc, started.id)

    assert result.status == "done"
    assert deps.log[1::2] == ["stop"] * 4 and all(e.startswith("create") for e in deps.log[0::2])  # stop after each
    assert deps.active() is None and not inf.held
    assert [c.status for c in result.candidates] == ["done", "done", "failed", "done"]
    assert result.candidates[2].error == "CUDA0 out of memory" and result.candidates[2].score is None
    assert all(c.load_s is not None and len(c.run_ids) == 3 for c in result.candidates if c.status == "done")
    assert len({c.plan.id for c in result.candidates}) == 4  # each candidate has its own stored plan row
    assert result.best_index == winner

    best = result.candidates[winner]
    profile = ProfileStore(store).list(MODEL)[0]
    assert result.profile_id == profile.id and profile.objective == objective and profile.score == best.score
    assert result.note is None
    # keyed like the selector looks it up: every device the request can use, named as the planner names them
    assert profile.signature == signature(MODEL, 4096, [("local", "CUDA0"), (PEER, f"{EP}#0")])
    local, peer = node_views()
    sig = signature(MODEL, 4096, profile_devices(local_budgets(local), remote_budgets(peer)))
    assert ProfileStore(store).best(MODEL, sig, objective).id == profile.id
    assert profile.plan_fields["tensor_split"] == best.plan.tensor_split and profile.run_ids == best.run_ids

    labels = [c[0] for c in inf.calls[:3]]
    bodies = {c[0]: c[1] for c in inf.calls[:3]}
    assert labels == ["tune-ttft", "tune-decode", "tune-prefill"]
    assert all(b["temperature"] == 0 and b["seed"] == 42 and b["cache_prompt"] is False for b in bodies.values())
    assert bodies["tune-decode"]["max_tokens"] == 64 and bodies["tune-decode"]["ignore_eos"] is True
    assert all(c[2] == "benchmark" and c[3] == started.id for c in inf.calls)


@pytest.mark.anyio
async def test_cancel_stops_the_running_candidate(store, events):
    block = asyncio.Event()
    inf = FakeInference(rates, block=block)
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    svc = service(store, events, deps, inf)
    started = await svc.start(TuneRequest(model_id=MODEL))
    await inf.entered.wait()
    result = await svc.cancel(started.id)
    assert result.status == "cancelled" and result.error == "cancelled by owner"
    assert deps.log == [deps.log[0], "stop"] and deps.active() is None and not inf.held
    assert [c.status for c in result.candidates] == ["skipped"] * len(result.candidates)
    assert (await svc.cancel(started.id)).status == "cancelled"  # a second cancel is a no-op


@pytest.mark.anyio
async def test_shutdown_cancels_and_records_reason(store, events):
    inf = FakeInference(rates, block=asyncio.Event())
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    svc = service(store, events, deps, inf)
    started = await svc.start(TuneRequest(model_id=MODEL))
    await inf.entered.wait()
    await svc.shutdown()
    result = svc.get(started.id)
    assert result.status == "cancelled" and result.error == "gateway shutting down" and deps.active() is None


@pytest.mark.anyio
async def test_deployment_started_elsewhere_aborts_the_tune(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    other = Deployment(id="e" * 32, plan=LOCAL_FIT, state=DeploymentState.loading, history=[], created_at=utcnow())

    def intrude(dep):
        deps.external = other  # the owner clicks 'Run' in the dashboard while the first candidate is measured

    svc = service(store, events, deps, FakeInference(rates, on_collect=intrude))
    result = await settle(svc, (await svc.start(TuneRequest(model_id=MODEL))).id)
    assert result.status == "failed" and "another deployment started" in result.error
    assert result.candidates[0].status == "done" and {c.status for c in result.candidates[1:]} == {"skipped"}


@pytest.mark.anyio
async def test_every_candidate_failing_fails_the_tune(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT}, fail=lambda p: "out of memory")
    svc = service(store, events, deps, FakeInference(rates))
    result = await settle(svc, (await svc.start(TuneRequest(model_id=MODEL, max_candidates=2))).id)
    assert result.status == "failed" and "out of memory" in result.error and result.profile_id is None
    assert deps.log[1::2] == ["stop", "stop"]


@pytest.mark.anyio
@pytest.mark.parametrize("engine,model_id", [("vllm", MODEL), ("exo", MODEL), (None, "qwen2.5-7b-instruct-awq")])
async def test_a_winner_no_plan_replays_is_not_saved_as_a_profile(store, events, engine, model_id):
    plan = LOCAL_FIT.model_copy(update={"engine": engine or "llama_rpc", "model_id": model_id})
    svc = service(store, events, FakeDeployments(store, {PlanStrategy.auto: plan}), FakeInference(rates))
    result = await settle(svc, (await svc.start(TuneRequest(model_id=MODEL, engine=engine, tune_flags=False))).id)
    assert result.status == "done" and result.best_index == 0 and result.profile_id is None
    assert ProfileStore(store).list(MODEL) == []  # plans never replay it
    assert result.note == (f"the winner runs {plan.engine} on {model_id}, and plans "
                           f"replay only llama.cpp splits of {MODEL}")  # the UI and CLI show it, not just a server log


@pytest.mark.anyio
async def test_trials_are_tune_deployments_measured_by_id(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})  # its ready() never offers a tune trial
    svc = service(store, events, deps, FakeInference(rates))
    result = await settle(svc, (await svc.start(TuneRequest(model_id=MODEL, max_candidates=2))).id)
    assert result.status == "done" and [c.status for c in result.candidates] == ["done", "done"]
    assert [d.purpose for d in deps.deps.values()] == ["tune", "tune"]


def test_reconcile_marks_running_tunes_failed(store, events):
    deps = FakeDeployments(store, {PlanStrategy.auto: LOCAL_FIT})
    svc = service(store, events, deps, FakeInference(rates))
    running = TuneResult(id="t" * 32, request=TuneRequest(model_id=MODEL), status="running", created_at=utcnow(),
                         candidates=[TuneCandidate(label="a", plan=LOCAL_FIT, status="running"),
                                     TuneCandidate(label="b", plan=LOCAL_FIT, status="pending")])
    svc._save(running)
    fresh = service(store, events, deps, FakeInference(rates))
    notes = fresh.reconcile_on_start()
    result = fresh.get(running.id)
    assert notes == [f"tune {running.id[:8]} marked failed after restart"]
    assert result.status == "failed" and result.error == "gateway restarted"
    assert [c.status for c in result.candidates] == ["skipped", "skipped"]
    assert fresh.reconcile_on_start() == [] and [t.id for t in fresh.list()] == [running.id]
    with pytest.raises(DeploymentError):
        fresh.get("missing")


# --------------------------------------------------------------------------- scoring and rates


def test_score_for_both_objectives():
    assert score("interactive", 100.0, 40.0, 200.0) == pytest.approx(0.1 + 256 / 40)
    assert score("throughput", 100.0, 40.0, 200.0) == pytest.approx(1024 / 200 + 256 / 40)
    assert score("interactive", None, 40.0, 200.0) is None
    assert score("throughput", 100.0, 40.0, None) is None
    assert score("interactive", 100.0, None, 200.0) is None


def test_rates_fall_back_to_client_clocks():
    dep = Deployment(id="d" * 32, plan=LOCAL_FIT, state=DeploymentState.ready, history=[], created_at=utcnow())
    timed = run_record(dep, "x", 100.0, 40.0, 500.0)
    assert decode_rate(timed) == 40.0 and prefill_rate(timed) == 500.0
    bare = run_record(dep, "x", 200.0, 40.0, 500.0, timings=False)  # 64 tokens, 63 after the first in 1.575 s
    assert decode_rate(bare) == pytest.approx(40.0) and prefill_rate(bare) == pytest.approx(1000 / 0.2)
    partial = bare.model_copy(update={"timings": EngineTimings(predicted_n=11, predicted_ms=500.0, prompt_n=100,
                                                               prompt_ms=50.0)})
    assert decode_rate(partial) == pytest.approx(20.0) and prefill_rate(partial) == pytest.approx(2000.0)


def test_prefill_prompt_shrinks_with_context():
    assert len(tuner_mod.prefill_prompt(256)) < len(tuner_mod.prefill_prompt(4096))


# --------------------------------------------------------------------------- flag variants


def test_micro_batch_variant_rescales_compute_and_drops_what_no_longer_fits():
    extra = 60 * 1024 * 1024  # what -ub 512 adds over the fixed compute base for this model
    compute = planner.COMPUTE_BASE + extra
    roomy = mkplan([asg("local", "CUDA0", range(28), head=True, total=6 * GiB, budget=10 * GiB)])
    roomy = roomy.model_copy(update={"assignments": [a.model_copy(update={"compute_bytes_est": compute})
                                                     for a in roomy.assignments]})
    big = tuner_mod.with_ubatch(roomy, 1024)
    assert big.engine_flags == {"ubatch": 1024}
    assert big.assignments[0].compute_bytes_est == compute + extra
    assert big.assignments[0].total_bytes_est == 6 * GiB + extra
    tight = roomy.model_copy(update={"assignments": [a.model_copy(update={"budget_bytes": 6 * GiB + extra // 2})
                                                     for a in roomy.assignments]})
    assert tuner_mod.with_ubatch(tight, 1024) is None
    assert tuner_mod.with_ubatch(tight, 256).assignments[0].fits


def test_flag_variants_skip_what_llama_server_would_reject():
    small_batch = LOCAL_FIT.model_copy(update={"engine_flags": {"batch": 512}})
    labels = [label.split(":")[0] for label, _ in tuner_mod.flag_variants(small_batch)]
    assert "micro-batch 1024" not in labels and "micro-batch 256" in labels


# --------------------------------------------------------------------------- real DeploymentManager + InferenceService


class FakeModels:
    def primary_path(self, model_id):
        from pathlib import Path

        return Path("model.gguf")

    def model_dir(self, model_id):
        from pathlib import Path

        return Path(".")

    def get(self, model_id):
        return ModelArtifact(id=model_id, display_name="m", family="qwen2", params_b=7.6, quantization="Q4_K_M",
                             repo="r", revision="x", license="apache-2.0",
                             files=[ModelFile(name="m.gguf", url="https://x", sha256="0" * 64, size_bytes=1)])

    def summary(self, model_id):
        return None

    def projector_path(self, model_id):  # no vision projector
        return None


class TimingAdapter:
    """Fake llama-server: ready at once, and every stream ends with llama.cpp-style usage and timings."""

    engine_id = "llama_rpc"
    idle_timeout_s = 5.0

    def __init__(self):
        self.started, self.stopped, self.bodies = 0, 0, []

    async def validate(self, spec):
        return ValidationResult(ok=True, problems=[])

    async def prepare(self, spec):
        self.started += 1
        return DeploymentHandle(deployment_id=spec.deployment_id, process_key="k", base_url="http://127.0.0.1:1",
                                log_path=None, pid=1, args=[])  # type: ignore[arg-type]

    async def wait_ready(self, handle, timeout_s=0, on_progress=None):
        return None

    async def observe(self, handle, spec, validation):
        return PlacementReport(observed=None, verified=True, detail="local only")

    def read_log(self, handle):
        return ""

    def process_alive(self, handle):
        return True

    async def stop(self, handle):
        self.stopped += 1
        return StopResult(clean=True, detail="terminated")

    async def cancel(self, request_id):
        return None

    async def stream(self, handle, req):
        self.bodies.append(req.body)
        n = req.body.get("max_tokens", 8)
        payload = {"choices": [{"delta": {"content": "x"}, "finish_reason": "length", "index": 0}],
                   "usage": {"prompt_tokens": 900, "completion_tokens": n},
                   "timings": {"prompt_n": 900, "prompt_ms": 450.0, "prompt_per_second": 2000.0, "predicted_n": n,
                               "predicted_ms": 20.0 * n, "predicted_per_second": 50.0, "cache_n": 0}}
        yield GenerationEvent(raw=b"data: x\n\n", data=json.dumps(payload), payload=payload)
        yield GenerationEvent(raw=b"data: [DONE]\n\n", data="[DONE]", payload=None, done=True)


@pytest.mark.anyio
async def test_tune_through_the_real_deployment_manager(store, events, tmp_path):
    sessions = SessionService(store, events, b"k" * 32, 8000, lambda: ["10.0.0.2"])
    reg = NodeRegistry(store, events, sessions, b"k" * 32, make_inventory().engine, make_inventory(),
                       contribution(["CUDA0"]))
    adapter = TimingAdapter()
    mgr = DeploymentManager(store, events, reg, FakeModels(), LeaseManager(store), adapter,
                            Supervisor(tmp_path / "m.json"))
    inf = InferenceService(store, events, mgr, FakeModels(), adapter)
    plan = mkplan([asg("local", "CUDA0", range(28), head=True, total=2 * GiB, budget=6 * GiB)])
    mgr.plan = lambda req: plan.model_copy(update={"id": uuid.uuid4().hex, "created_at": utcnow()})  # type: ignore[method-assign]
    svc = TunerService(store, events, mgr, inf, ProfileStore(store), FakeModels(), reg)
    svc.poll_s = 0.005

    result = await settle(svc, (await svc.start(TuneRequest(model_id=MODEL, max_candidates=3))).id)
    assert result.status == "done", result.error
    assert adapter.started == adapter.stopped == 3 and mgr.active() is None
    best = result.candidates[result.best_index]
    assert best.decode_tokens_per_s == 50.0 and best.prefill_tokens_per_s == 2000.0
    assert best.score == pytest.approx(best.ttft_ms / 1000 + 256 / 50.0)
    assert ProfileStore(store).list(MODEL)[0].signature == signature(MODEL, 4096, [("local", "CUDA0")])
    runs = [r for r in inf.runs(50) if r.benchmark_id == result.id]
    assert len(runs) == 9 and {r.prompt_label for r in runs} == {"tune-ttft", "tune-decode", "tune-prefill"}
    flags_sent = [b for b in adapter.bodies if b.get("ignore_eos")]
    assert len(flags_sent) == 3 and all(b["max_tokens"] == 64 for b in flags_sent)


# --------------------------------------------------------------------------- trial load budget


class BudgetAdapter(TimingAdapter):
    """An engine with its own load budget (EXO: every Mac downloads the whole repo first, up to 4 h)."""

    engine_id = "exo"

    def __init__(self, budget_s):
        super().__init__()
        self.budget_s = budget_s

    def load_timeout_s(self, plan):
        return self.budget_s


class DefaultBudgetAdapter(TimingAdapter):
    """No load_timeout_s hook: the manager leaves wait_ready at its own default."""

    async def wait_ready(self, handle, timeout_s=4 * 3600.0, on_progress=None):
        return None


class SlowLoad(FakeDeployments):
    """Each trial stays loading for load_s (a download) before it becomes ready."""

    def __init__(self, store, plans, adapter, load_s):
        super().__init__(store, plans, adapter=adapter)
        self.load_s, self.ready_at = load_s, {}

    async def create(self, req, key):
        dep = await super().create(req, key)
        self.ready_at[dep.id] = time.monotonic() + self.load_s
        return dep

    def get(self, dep_id):
        return self.deps[dep_id] if time.monotonic() < self.ready_at[dep_id] else super().get(dep_id)


def test_ready_timeout_is_the_engines_own_load_and_warmup_budget():
    for adapter, load in ((BudgetAdapter(4 * 3600.0), 4 * 3600.0), (DefaultBudgetAdapter(), 4 * 3600.0),
                          (TimingAdapter(), config.LOAD_TIMEOUT_S)):  # the last is llama.cpp's budget
        expected = load + warmup_timeout(LOCAL_FIT, adapter) + tuner_mod.READY_SLACK_S
        assert tuner_mod.ready_timeout(LOCAL_FIT, adapter) == expected
    # the manager's warmup budget by its public name, so a rename of the private alias cannot break the tuner
    assert tuner_mod.warmup_timeout is warmup_timeout and not hasattr(tuner_mod, "_warmup_timeout")


@pytest.mark.anyio
@pytest.mark.parametrize("budget_s,load_s,status", [(60.0, 0.4, "done"), (0.05, 3600.0, "failed")])
async def test_a_trial_waits_for_its_engines_load_budget_not_llama_cpps(store, events, monkeypatch, budget_s, load_s,
                                                                       status):
    # scaled: llama.cpp's load + warmup budget is 0.1 s here; the EXO trial loads for load_s within budget_s
    monkeypatch.setattr(config, "LOAD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(config, "WARMUP_TIMEOUT_S", 0.05)
    monkeypatch.setattr(tuner_mod, "READY_SLACK_S", 0.0)
    plan = LOCAL_FIT.model_copy(update={"engine": "exo"})
    deps = SlowLoad(store, {PlanStrategy.auto: plan}, BudgetAdapter(budget_s), load_s)
    svc = service(store, events, deps, FakeInference(rates))
    result = await settle(svc, (await svc.start(TuneRequest(model_id=MODEL, engine="exo"))).id)
    cand = result.candidates[0]
    assert cand.status == status and deps.log[1:] == ["stop"], cand.error
    assert status == "done" or cand.error == "not ready after 0s (still loading)"
