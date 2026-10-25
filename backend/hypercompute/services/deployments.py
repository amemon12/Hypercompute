"""Deployment lifecycle: plan -> reserve -> prepare -> load -> verify placement -> warm -> ready -> stop.

One active deployment at a time. Failures keep their error and cleanup status. Losing a node that a
deployment depends on fails that deployment and stops its llama-server; there is no live migration.

replace() swaps a ready deployment for another plan at a request boundary (a reload, not a migration): drain, stop the
old engine, release its leases once it is gone, then reserve, load, verify and warm the new plan under the next
revision. A failed new load rolls back to the previous plan only when its resources still admit it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .. import config
from ..adapters import registry
from ..adapters.base import ENGINE_PROCESS_LABEL, DeploymentHandle, DeploymentSpec, EngineAdapterError
from ..adapters.llama_output import _rpc_number, check_participation, detect_failure  # noqa: F401 (re-exported)
from ..adapters.llama_rpc import PROCESS_KEY_PREFIX, LlamaRpcAdapter, parallel_slots
from ..agent.supervisor import Supervisor
from ..capacity import runtime as stream_runtime
from ..schemas import (
    TERMINAL_STATES,
    CreateDeploymentRequest,
    Deployment,
    DeploymentState,
    DeviceAssignment,
    DeviceBudget,
    ModelView,
    NodeState,
    NodeView,
    Plan,
    PlanLimits,
    PlanRequest,
    StateTransition,
    utcnow,
)
from ..storage import Store, dumps
from ..units import GiB, MiB, fmt_bytes
from .events import EventBus
from .layout import LayoutError, layout_plan, peer_facts
from .leases import LeaseConflict, LeaseManager, requests_from_plan
from .models import ModelService
from .nodes import LOCAL_ID, NodeRegistry
from .planner import ModelShape, StreamCapability
from .selector import EnginePlanner, Inputs, choose_plan

log = logging.getLogger("hypercompute.deployments")


def idle_budget(plan: Plan, adapter: Any, default: float) -> float:
    """Longest silence the engine allows this deployment: adapter.idle_timeout_for(plan), else its idle_timeout_s."""
    idle_for = getattr(adapter, "idle_timeout_for", None)
    return float(idle_for(plan)) if idle_for is not None else float(getattr(adapter, "idle_timeout_s", default))


def warmup_timeout(plan: Plan, adapter: Any) -> float:
    """Capacity mode can take minutes for one token, so its single-token warmup gets the deployment's idle budget."""
    if plan.capacity_mode:
        return max(config.WARMUP_TIMEOUT_S, 2 * idle_budget(plan, adapter, config.WARMUP_TIMEOUT_S))
    return config.WARMUP_TIMEOUT_S




def _load_budget(plan: Plan, adapter: Any) -> dict[str, float]:
    """wait_ready's timeout_s from adapter.load_timeout_s(plan); an adapter without the hook keeps its own default."""
    load_timeout = getattr(adapter, "load_timeout_s", None)
    return {"timeout_s": float(load_timeout(plan))} if load_timeout is not None else {}


def lease_budgets(plan: Plan, nodes: NodeRegistry) -> dict[str, int]:
    """Each node's lease allowance: its contribution limit. The coordinator's host CPU tier is planned against free RAM
    (cpu_budget), not against the contributed devices' max_bytes, so its budget is added to the local allowance."""
    budgets = {a.node_id: (n.contribution.max_bytes if (n := nodes.get(a.node_id)) else a.budget_bytes)
               for a in plan.assignments}
    host_cpu = sum(a.budget_bytes for a in plan.assignments
                   if a.node_id == LOCAL_ID and a.device == "CPU" and a.endpoint is None)
    return {node_id: b + (host_cpu if node_id == LOCAL_ID else 0) for node_id, b in budgets.items()}


def _port(base_url: str) -> int | None:
    tail = base_url.rstrip("/").rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None
VRAM_HEADROOM = 512 * MiB
ACCEL_PREFIXES = ("CUDA", "MTL", "Vulkan", "ROCm")
# Prior for the vision encoder's compute buffer on top of the projector file (mtmd warms up on its largest image);
# llama-server logs the real figure as "compute buffer size" after "clip_model_loader".
PROJECTOR_COMPUTE_EST = 1 * GiB
ORPHAN_NOTE = "at the last gateway start: "  # starts the orphan cleanup's note; each restart replaces it, not appends
DRAIN_TIMEOUT_S = 30.0  # how long replace() waits for the request in flight on the old deployment


class DeploymentError(Exception):
    def __init__(self, status: int, code: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.fix = status, code, fix


class InferenceHold(Protocol):
    """Exclusive inference on a deployment (streaming.SlotHold): no new request starts until release()."""

    def release(self) -> None: ...


class _NoHold:
    def release(self) -> None:
        return None


def _pool_kind(node: NodeView, pool_id: str) -> str:
    pool = next((p for p in node.inventory.memory_pools if p.id == pool_id), None)
    return pool.kind if pool else "system_ram"


def _pool_free(node: NodeView, pool_id: str) -> int | None:
    pool = next((p for p in node.inventory.memory_pools if p.id == pool_id), None)
    return pool.free_bytes if pool else None


def _unified_budget(node: NodeView, device: str, pool_id: str, device_free: int | None) -> tuple[int, str]:
    """A local GPU on a unified pool (a Mac's MTL0) has two limits. Its own allocation cap is what llama.cpp reports as
    the device's free memory (Metal: recommendedMaxWorkingSetSize minus this process's allocations), which keeps
    VRAM_HEADROOM like a discrete card. The RAM it shares with the CPU is the pool's free bytes now minus the owner's
    reserve, applied to the pool once (never to the Metal cap as well). Without a pool observation the Metal figure
    stands in for the pool. The smallest of these and the contribution limit wins."""
    reserve = node.contribution.reserve_bytes
    pool_free = _pool_free(node, pool_id)
    shared = (device_free or 0) if pool_free is None else pool_free
    own = f"{device} working-set limit minus {VRAM_HEADROOM // MiB} MiB headroom"
    caps = [(node.contribution.max_bytes, "contribution limit"), ((device_free or 0) - VRAM_HEADROOM, own),
            (shared - reserve, f"free unified memory now minus {reserve // MiB} MiB reserve")]
    cap, basis = min(caps, key=lambda c: c[0])
    return max(0, cap), basis


def local_budgets(node: NodeView) -> list[DeviceBudget]:
    out = []
    for d in node.inventory.engine.devices:
        if not d.name.startswith(ACCEL_PREFIXES) or d.name not in node.contribution.devices:
            continue
        inv_dev = next((x for x in node.inventory.devices if x.id == d.name), None)
        pool_id = inv_dev.pool_id if inv_dev else f"vram:{d.name}"
        kind = _pool_kind(node, pool_id)
        fresh = inv_dev.free_bytes if inv_dev and inv_dev.free_bytes is not None else d.free_bytes
        if kind == "unified":
            budget, basis = _unified_budget(node, d.name, pool_id, fresh)
        else:
            headroom = node.contribution.reserve_bytes if kind != "discrete_vram" else VRAM_HEADROOM
            free_cap = (fresh or 0) - headroom
            budget = max(0, min(node.contribution.max_bytes, free_cap))
            basis = ("contribution limit" if node.contribution.max_bytes <= free_cap else
                     f"free memory now minus {headroom // MiB} MiB headroom")
        out.append(DeviceBudget(node_id=LOCAL_ID, device=d.name, description=d.description, endpoint=None,
                                pool_id=pool_id, pool_kind=kind, budget_bytes=budget, budget_basis=basis))  # type: ignore[arg-type]
    return out


def cpu_budget(node: NodeView) -> DeviceBudget | None:
    """Host RAM the coordinator's own CPU may use for layers that do not fit on GPUs (soft budget)."""
    pool = next((p for p in node.inventory.memory_pools if p.id in ("ram", "uma")), None)
    if pool is None or pool.free_bytes is None:
        return None
    reserve = node.contribution.reserve_bytes
    budget = max(0, pool.free_bytes - reserve - 512 * MiB)
    return DeviceBudget(node_id=LOCAL_ID, device="CPU", description="host CPU (system RAM)", endpoint=None,
                        pool_id=pool.id, pool_kind=pool.kind, budget_bytes=budget,  # type: ignore[arg-type]
                        budget_basis=f"free RAM now minus {reserve // MiB} MiB reserve and 512 MiB safety")


def remote_budgets(node: NodeView, *, whole_contribution: bool = False) -> list[DeviceBudget]:
    """Each verified RPC device's budget: its share of the contribution by reported free memory, within its free
    memory minus headroom. whole_contribution: every device may take the whole contribution within its own free
    memory (an explicit layout then checks the node's total itself, services/layout.py)."""
    ver = node.verification
    if ver is None or node.worker is None:
        return []
    endpoint = f"{node.worker.host}:{node.worker.port}"
    rpc = ver.rpc_devices
    total_free = sum(d.free_bytes or 0 for d in rpc) or 1
    out = []
    for i, d in enumerate(rpc):
        contributed = node.contribution.devices[i] if i < len(node.contribution.devices) else None
        inv_dev = next((x for x in node.inventory.devices if x.id == contributed), None)
        pool_id = inv_dev.pool_id if inv_dev else "ram"
        kind = _pool_kind(node, pool_id)
        free = d.free_bytes or 0
        fresh_pool = _pool_free(node, pool_id)
        if kind != "discrete_vram" and fresh_pool is not None:
            free = min(free, fresh_pool)
        elif kind == "discrete_vram" and inv_dev and inv_dev.free_bytes is not None:
            free = min(free, inv_dev.free_bytes)  # heartbeats refresh the card's free VRAM; the probe is from join
        headroom = node.contribution.reserve_bytes if kind != "discrete_vram" else VRAM_HEADROOM
        share = (node.contribution.max_bytes if whole_contribution
                 else node.contribution.max_bytes * (d.free_bytes or 0) // total_free)
        budget = max(0, min(share, free - headroom))
        basis = "contribution limit" if share <= free - headroom else "free memory minus headroom"
        # an RPC device's own description is only its endpoint; the planner needs the backend for its overhead
        description = f"{inv_dev.backend} {inv_dev.name} via {endpoint}" if inv_dev else d.description
        out.append(DeviceBudget(node_id=node.id, device=f"{endpoint}#{i}", description=description,
                                endpoint=endpoint, pool_id=pool_id, pool_kind=kind, budget_bytes=budget,  # type: ignore[arg-type]
                                budget_basis=basis))
    return out


def _host_ram_free(node: NodeView) -> int | None:
    return next((p.free_bytes for p in node.inventory.memory_pools if p.id in ("ram", "uma")), None)


def projector_need(view: ModelView) -> int:
    """Bytes the model's vision projector takes on this machine once llama-server loads it: 0 until it is verified."""
    mmproj = view.artifact.mmproj
    return mmproj.size_bytes + PROJECTOR_COMPUTE_EST if mmproj and view.projector_status == "downloaded" else 0


def reserve_projector(local: list[DeviceBudget], cpu: DeviceBudget | None,
                      need: int) -> tuple[list[DeviceBudget], DeviceBudget | None]:
    """Budgets with the vision projector's bytes taken off the device it runs on: this machine's last GPU (where
    llama_rpc.projector_args puts it), else the host CPU. llama-server runs with --fit off, so nothing else makes
    room for it."""
    if need <= 0:
        return local, cpu

    def less(b: DeviceBudget) -> DeviceBudget:
        basis = f"{b.budget_basis} minus ~{fmt_bytes(need)} for the vision projector"
        return b.model_copy(update={"budget_bytes": max(0, b.budget_bytes - need), "budget_basis": basis})
    if local:
        return [*local[:-1], less(local[-1])], cpu
    return local, less(cpu) if cpu is not None else None


def projector_warnings(plan: Plan, local: list[DeviceBudget], cpu: DeviceBudget | None, need: int) -> list[str]:
    """What a llama.cpp plan means for the vision projector of `need` bytes, against the unreserved budgets. Plans
    from the capacity optimizer or a measured profile never saw the reservation, so this checks every plan."""
    if need <= 0 or plan.engine != "llama_rpc":
        return []
    if plan.tier == "stream":
        return ["the stream tier runs without the vision projector, so this deployment answers text only"]
    gpus = [a for a in plan.assignments if a.endpoint is None and a.device != "CPU"]
    device = gpus[-1].device if gpus else "CPU"
    budget = next((b.budget_bytes for b in [*local, *([cpu] if cpu else [])] if b.device == device), None)
    used = next((a.total_bytes_est for a in plan.assignments if a.endpoint is None and a.device == device), 0)
    if budget is None or budget - used >= need:
        return []
    return [f"the vision projector needs ~{fmt_bytes(need)} on {device} but this plan leaves "
            f"~{fmt_bytes(max(0, budget - used))} there; llama-server may fail to load it"]


def _with_parallel(req: PlanRequest, plan: Plan) -> Plan:
    """req.parallel as engine_options["parallel"], set after selection so profile replay (engine_flags) and the
    explicit-layout path keep it; absent at 1, so a default plan does not change. Only llama.cpp takes it (-np)."""
    if req.parallel == 1:
        return plan
    if plan.engine != "llama_rpc":
        raise DeploymentError(422, "bad_request", f"parallel {req.parallel} needs the llama_rpc engine; this plan runs "
                              f"{plan.engine}", fix="plan with engine llama_rpc, or without parallel")
    plan = plan.model_copy(update={"engine_options": {**plan.engine_options, "parallel": req.parallel}})
    try:
        parallel_slots(plan)
    except EngineAdapterError as exc:
        raise DeploymentError(422, exc.code, str(exc), exc.fix) from exc
    return plan


def rpc_device_counts(nodes: list[NodeView]) -> dict[str, int]:
    """Devices each verified worker exposes, by endpoint: the counts assign_rpc_names numbers a plan with."""
    return {f"{n.worker.host}:{n.worker.port}": max(1, len(n.verification.rpc_devices))
            for n in nodes if n.worker and n.verification}


def assign_rpc_names(plan: Plan, device_counts: dict[str, int]) -> Plan:
    """Name remote devices the way llama.cpp registers them: RPC0.. in --rpc endpoint order."""
    names: dict[str, list[str]] = {}
    k = 0
    for ep in plan.rpc_endpoints:
        names[ep] = [f"RPC{k + i}" for i in range(device_counts.get(ep, 1))]
        k += device_counts.get(ep, 1)
    assignments = []
    for a in plan.assignments:
        if a.endpoint:
            idx = int(a.device.rsplit("#", 1)[1]) if "#" in a.device else 0
            a = a.model_copy(update={"device": names[a.endpoint][idx]})
        assignments.append(a)
    return plan.model_copy(update={"assignments": assignments, "device_list": _device_list(assignments)})


def _device_list(assignments: list[DeviceAssignment]) -> list[str]:
    """--device order. The CPU tier is implied by -ngl and llama.cpp rejects `--device CPU`."""
    return [a.device for a in assignments if a.device != "CPU"]


def remap_rpc_names(plan: Plan, by_endpoint: dict[str, list[str]],
                    device_counts: dict[str, int] | None = None) -> Plan:
    """Replace provisional RPC names with the names a combined probe actually reported.

    Each device keeps its index within its own RPC server: its RPC number minus its endpoint's first number under
    the counts the plan was named with (device_counts, else the probe's own counts). A device whose index the probe
    no longer lists gets its provisional 'endpoint#index' name back, so validation reports it as missing.
    """
    counts = device_counts if device_counts is not None else {ep: len(n) for ep, n in by_endpoint.items()}
    first: dict[str, int] = {}
    k = 0
    for ep in plan.rpc_endpoints:
        first[ep], k = k, k + counts.get(ep, 1)
    assignments = []
    for a in plan.assignments:
        number = _rpc_number(a.device)
        actual = by_endpoint.get(a.endpoint or "") or []
        if a.endpoint and number is not None and actual:
            idx = number - first.get(a.endpoint, 0)
            a = a.model_copy(update={"device": actual[idx] if 0 <= idx < len(actual) else f"{a.endpoint}#{idx}"})
        assignments.append(a)
    return plan.model_copy(update={"assignments": assignments, "device_list": _device_list(assignments)})


class DeploymentManager:
    def __init__(self, store: Store, events: EventBus, nodes: NodeRegistry, models: ModelService,
                 leases: LeaseManager, adapter: LlamaRpcAdapter, supervisor: Supervisor,
                 extra_adapters: Mapping[str, Any] | None = None) -> None:
        self._store, self._events, self._nodes, self._models = store, events, nodes, models
        self._leases, self._adapter, self._sup = leases, adapter, supervisor
        self._adapters: dict[str, Any] = {"llama_rpc": adapter, **(extra_adapters or {})}
        self.adapter_ctx: Any = None  # adapters.registry.AdapterContext, set by the coordinator
        self.profiles: Any = None  # services.profiles.ProfileStore, set by the coordinator
        self.capacity: Any = None  # capacity.service.CapacityService, set by the coordinator; None: planner only
        self._deps: dict[str, Deployment] = {}
        self._handles: dict[str, DeploymentHandle] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cleanups: dict[str, asyncio.Task[Deployment]] = {}
        self._create_lock = asyncio.Lock()
        self._cancel_reason: dict[str, tuple[DeploymentState, str]] = {}
        self._orphans: dict[str, str | None] = {}  # held at restart for cleanup_orphan -> the cleanup text kept
        self.warmup: Callable[[Deployment, DeploymentHandle], Awaitable[None]] | None = None
        self.interrupt: Callable[[str], Awaitable[None]] | None = None
        # replace(): (deployment, timeout_s) -> exclusive inference hold, set by the coordinator; None: no wait
        self.drain: Callable[[Deployment, float], Awaitable[InferenceHold]] | None = None
        self.drain_timeout_s = DRAIN_TIMEOUT_S
        self._replacing: str | None = None  # the deployment replace() is swapping out, while it runs
        nodes.busy = self.busy
        nodes.on_lost = self.on_node_lost

    # ------------------------------------------------------------------ persistence / queries
    def _save(self, dep: Deployment) -> Deployment:
        self._deps[dep.id] = dep
        self._store.execute(
            "INSERT INTO deployments (id, state, idempotency_key, created_at, doc) VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET state=excluded.state, doc=excluded.doc",
            (dep.id, dep.state.value, dep.idempotency_key, dep.created_at.isoformat(),
             dumps(dep.model_dump(mode="json"))))
        return dep

    def _transition(self, dep_id: str, state: DeploymentState, detail: str | None = None, **extra: object
                    ) -> Deployment:
        dep = self._deps[dep_id]
        history = [*dep.history, StateTransition(state=state, at=utcnow(), detail=detail)]
        dep = self._save(dep.model_copy(update={"state": state, "history": history, **extra}))
        self._events.publish("deployment.state", f"Deployment {dep_id[:8]}: {state.value}"
                             + (f" ({detail})" if detail else ""), deployment_id=dep_id, state=state.value,
                             revision=dep.revision)
        return dep

    def _progress(self, dep_id: str, text: str) -> None:
        dep = self._deps.get(dep_id)
        if dep and dep.progress != text:
            self._save(dep.model_copy(update={"progress": text}))
            self._events.publish("deployment.progress", text, deployment_id=dep_id, revision=dep.revision)

    def list(self) -> list[Deployment]:
        rows = self._store.query("SELECT doc FROM deployments ORDER BY created_at DESC LIMIT 50")
        return [self._deps.get(d.id, d) for d in (Deployment.model_validate(json.loads(r["doc"])) for r in rows)]

    def get(self, dep_id: str) -> Deployment:
        dep = self._deps.get(dep_id)
        if dep is None:
            row = self._store.query_one("SELECT doc FROM deployments WHERE id=?", (dep_id,))
            if row is None:
                raise DeploymentError(404, "not_found", "no such deployment")
            dep = Deployment.model_validate(json.loads(row["doc"]))
        return dep

    def active(self) -> Deployment | None:
        return next((d for d in self._deps.values() if d.state not in TERMINAL_STATES), None)

    def busy(self, node_id: str) -> list[str]:
        return [d.id for d in self._deps.values() if d.state not in TERMINAL_STATES
                and any(a.node_id == node_id for a in d.plan.assignments)]

    def ready(self) -> tuple[Deployment, DeploymentHandle]:
        """The owner's ready run. A tuner trial is never one: the tuner reaches it by id (get, handle)."""
        dep = self.active()
        if dep is None or dep.state != DeploymentState.ready or dep.id not in self._handles:
            raise DeploymentError(503, "no_ready_deployment", "no model is ready",
                                  fix="choose a model and click 'Run together' in the dashboard")
        if dep.purpose == "tune":
            raise DeploymentError(409, "tuning", "a tune is measuring this model right now; chat and benchmarks "
                                  "can use it once the tune finishes",
                                  fix="wait for the tune to finish, or cancel it in the Tune card")
        return dep, self._handles[dep.id]

    def handle(self, dep_id: str) -> DeploymentHandle | None:
        return self._handles.get(dep_id)

    # ------------------------------------------------------------------ engines
    def engines(self) -> list[str]:
        return list(self._adapters)

    def adapter_for(self, engine_id: str) -> Any:
        try:
            return self._adapters[engine_id]
        except KeyError:
            raise DeploymentError(422, "engine_unavailable", f"the {engine_id} engine is not installed on this host",
                                  fix=f"uv run hypercompute engine install --engine {engine_id}") from None

    def _adapter_of(self, dep_id: str) -> Any:
        return self.adapter_for(self._deps[dep_id].plan.engine)

    # ------------------------------------------------------------------ planning
    def plan(self, req: PlanRequest) -> Plan:
        """The selector's plan (engine, tier, split) for this request, named for llama.cpp and stored for create()."""
        try:
            view = self._models.view(req.model_id)
        except Exception as exc:
            raise DeploymentError(404, "unknown_model", str(exc)) from exc
        local_node = self._nodes.local
        eligible = local_node.state == NodeState.eligible
        remote_nodes = [] if req.strategy.value == "local_only" else self._nodes.eligible_remote(req.node_ids)
        need = projector_need(view)
        local, cpu = (local_budgets(local_node), cpu_budget(local_node)) if eligible else ([], None)
        if req.layout is not None:  # an explicit layout: checked, never chosen (no selector, optimizer or profile)
            plan = self._layout_plan(req, view, remote_nodes, need)
        else:
            reserved_local, reserved_cpu = reserve_projector(local, cpu, need)
            plan = choose_plan(Inputs(
                req=req, artifact=view.artifact, stream=self._stream_capability(view),
                shape=ModelShape.from_gguf(view.gguf, req.ctx_size) if view.gguf else None,  # None for HF / MLX
                local=reserved_local,
                remote=[b for n in remote_nodes for b in remote_budgets(n)],
                cpu=reserved_cpu,
                host_ram_free=_host_ram_free(local_node),
                artifacts=self._models.artifacts(), remote_nodes=remote_nodes, available=frozenset(self.engines()),
                status=lambda model_id: self._models.view(model_id).status,
                plan_engine=self._engine_planner(local_node, remote_nodes),
                profile=self.profiles.best if self.profiles is not None else None,
                capacity=self.capacity.plan if self.capacity is not None else None))
        if plan.engine == "llama_rpc":
            plan = self._llama_names(plan, rpc_device_counts(remote_nodes))
        plan = _with_parallel(req, plan)
        if notes := projector_warnings(plan, local, cpu, need):
            plan = plan.model_copy(update={"warnings": [*plan.warnings, *notes]})
        if plan.engine == "llama_rpc" and view.artifact.mmproj is not None:
            # the plan decides, so a projector fetched after planning never loads into a split sized without it
            loads = bool(need) and plan.tier != "stream"
            held = [f"~{fmt_bytes(need)} of {local[-1].device if local else 'CPU'} held back for the vision projector "
                    f"(its file plus a {fmt_bytes(PROJECTOR_COMPUTE_EST)} compute prior)"] if loads else []
            plan = plan.model_copy(update={"engine_options": {**plan.engine_options, "mmproj": loads},
                                           "reasons": [*plan.reasons, *held]})
        # every plan keeps its request's limits, so a replan of it never leaves them (CapacityService.evaluate)
        plan = self._download_gate(req, plan).model_copy(update={"limits": PlanLimits.of(req)})
        self._store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?)",
                            (plan.id, plan.created_at.isoformat(), dumps(plan.model_dump(mode="json"))))
        return plan

    def _layout_plan(self, req: PlanRequest, view: Any, remote: list[NodeView], projector: int = 0) -> Plan:
        """req.layout checked against the budgets free now (services/layout.py), less the vision projector's bytes;
        422 with the reason when invalid. A peer's file-backed stage needs its managed report: hcpeer on, and its
        cache volume for the disk budget."""
        local = self._nodes.local
        eligible = local.state == NodeState.eligible
        host_gpus, host_cpu = reserve_projector(local_budgets(local) if eligible else [],
                                                cpu_budget(local) if eligible else None, projector)
        managed = getattr(self._nodes, "managed", None)  # services.peer_seed.ManagedPeers: the last managed reports
        peers = {n.id: peer_facts(n, remote_budgets(n, whole_contribution=True),
                                  managed.report(n.id) if managed is not None else None,
                                  managed.seeded(n.id) if managed is not None else None) for n in remote}
        try:
            return layout_plan(req, ModelShape.from_gguf(view.gguf, req.ctx_size) if view.gguf else None,
                               host_gpus, host_cpu, peers, _host_ram_free(local))
        except LayoutError as exc:
            raise DeploymentError(422, exc.code, str(exc), exc.fix) from exc

    def _stream_capability(self, view: Any) -> StreamCapability | None:
        """The bounded stream tier for a downloaded GGUF on this host (Windows with the hcstream proxy built)."""
        if view.artifact.format != "gguf" or view.status != "downloaded" or "llama_rpc" not in self._adapters:
            return None
        try:
            return stream_runtime.stream_capability(self._models.file_paths(view.artifact.id))
        except Exception:  # noqa: BLE001 - a broken GGUF table must not break planning; the tier is just absent
            log.warning("stream tier unavailable for %s", view.artifact.id, exc_info=True)
            return None

    def _engine_planner(self, local: NodeView, remote: list[NodeView]) -> EnginePlanner:
        """registry.plan_for bound to this request's nodes; None (module missing) until the coordinator wires a ctx."""
        def plan_engine(engine_id: str, art: Any, req: PlanRequest) -> Plan | None:
            if self.adapter_ctx is None:
                return None
            return registry.plan_for(engine_id, req, art, local, remote, self.adapter_ctx)
        return plan_engine

    @staticmethod
    def _llama_names(plan: Plan, device_counts: dict[str, int]) -> Plan:
        """Peer devices get the RPC names llama.cpp will give them; CPU stays out of --device."""
        return assign_rpc_names(plan, device_counts)

    def _download_gate(self, req: PlanRequest, plan: Plan) -> Plan:
        """A plan can only run once its weights are here (EXO fetches its own on each node: engine_managed)."""
        status = self._models.view(plan.model_id).status
        if status in ("downloaded", "engine_managed"):
            return plan
        what = "model" if plan.model_id == req.model_id else f"model {plan.model_id}"
        return plan.model_copy(update={"feasible": False, "reasons": [
            f"{what} is {status.replace('_', ' ')}; download it first", *plan.reasons]})

    def find_plan(self, plan_id: str) -> Plan | None:
        """A stored plan of any age, read only (None when there is no such plan). Only stored_plan refuses stale
        plans, because only a deployment needs the cluster to still look as it did at planning time."""
        row = self._store.query_one("SELECT doc FROM plans WHERE id=?", (plan_id,))
        return Plan.model_validate(json.loads(row["doc"])) if row is not None else None

    def stored_plan(self, plan_id: str) -> Plan:
        plan = self.find_plan(plan_id)
        if plan is None:
            raise DeploymentError(404, "plan_not_found", "plan not found; check the plan again")
        if (utcnow() - plan.created_at).total_seconds() > 600:
            raise DeploymentError(409, "plan_stale", "plan is older than 10 minutes; check the plan again")
        return plan

    # ------------------------------------------------------------------ create / run
    async def create(self, req: CreateDeploymentRequest, idempotency_key: str | None) -> Deployment:
        if self._replacing is not None:
            raise DeploymentError(409, "deployment_replacing",
                                  f"deployment {self._replacing[:8]} is being replaced; try again once it is ready")
        async with self._create_lock:
            if idempotency_key:
                row = self._store.query_one("SELECT id FROM deployments WHERE idempotency_key=?",
                                            (idempotency_key,))
                if row:
                    return self.get(row["id"])
            current = self.active()
            if current is not None:
                raise DeploymentError(409, "deployment_active",
                                      f"deployment {current.id[:8]} is {current.state.value}; stop it first")
            if req.plan_id:
                plan = self.stored_plan(req.plan_id)
            elif req.plan:  # off the loop: the optimizer search takes up to ~0.65 s with 4 peer devices
                plan = await asyncio.to_thread(self.plan, req.plan)
            else:
                raise DeploymentError(422, "plan_required", "provide plan_id or plan")
            if not plan.feasible:
                raise DeploymentError(422, "plan_not_feasible", "; ".join(plan.reasons))
            dep = Deployment(id=uuid.uuid4().hex, plan=plan, state=DeploymentState.planned,
                             history=[StateTransition(state=DeploymentState.planned, at=utcnow())],
                             created_at=utcnow(), idempotency_key=idempotency_key, purpose=req.purpose,
                             revision=plan.revision)
            self._launch(dep)
            return dep

    def _launch(self, dep: Deployment) -> asyncio.Task[bool]:
        """Record a planned deployment and start its run; the task returns True once the deployment is ready."""
        self._save(dep)
        self._events.publish("deployment.state", f"Deployment {dep.id[:8]}: planned", deployment_id=dep.id,
                             state="planned", revision=dep.revision)
        task = asyncio.get_running_loop().create_task(self._run(dep.id))
        self._tasks[dep.id] = task
        return task

    async def _run(self, dep_id: str) -> bool:
        """Reserve, prepare, load, verify and warm one deployment. True once ready; a failure is cleaned up here."""
        try:
            spec, validation = await self._admit_and_prepare(dep_id)
            handle, detail = await self._load_and_verify(dep_id, spec, validation)
            await self._warm_and_activate(dep_id, handle, detail)
            return True
        except asyncio.CancelledError:
            final, reason = self._cancel_reason.pop(dep_id, (DeploymentState.stopped, "stopped during startup"))
            await self._cleanup(dep_id, final, reason)
            raise
        except (EngineAdapterError, LeaseConflict, DeploymentError, TimeoutError, OSError) as exc:
            log.warning("deployment %s failed: %s", dep_id, exc)
            await self._cleanup(dep_id, DeploymentState.failed, str(exc) or exc.__class__.__name__)
        except Exception as exc:  # keep a record of unexpected failures instead of losing them
            log.exception("deployment %s crashed", dep_id)
            await self._cleanup(dep_id, DeploymentState.failed, f"internal error: {exc}")
        return False

    async def _admit_and_prepare(self, dep_id: str) -> tuple[DeploymentSpec, Any]:
        """Reserve the plan's leases under its revision, re-check its peers and validate it with the engine."""
        dep = self._deps[dep_id]
        self._transition(dep_id, DeploymentState.reserving)
        self._leases.reserve(dep_id, requests_from_plan(dep.plan, lease_budgets(dep.plan, self._nodes)),
                             revision=dep.revision, chain=self._chain_root(dep))

        self._transition(dep_id, DeploymentState.preparing, "re-checking peers and model file")
        self._require_nodes(dep.plan)
        adapter = self.adapter_for(dep.plan.engine)
        spec = self._spec(dep_id, dep.plan)
        validation = await adapter.validate(spec)
        if validation.rpc_device_map:
            plan = remap_rpc_names(dep.plan, validation.rpc_device_map, self._rpc_counts(dep.plan))
            spec = replace(spec, plan=plan)
            validation = await adapter.validate(spec) if plan != dep.plan else validation
            self._save(self._deps[dep_id].model_copy(update={"plan": plan}))
        if not validation.ok:
            raise EngineAdapterError("validation_failed", "; ".join(validation.problems))
        return spec, validation

    async def _load_and_verify(self, dep_id: str, spec: DeploymentSpec, validation: Any
                               ) -> tuple[DeploymentHandle, str]:
        """Start the engine, wait until it serves, and check that every planned peer really holds work."""
        adapter = self.adapter_for(spec.plan.engine)
        label = ENGINE_PROCESS_LABEL.get(spec.plan.engine, spec.plan.engine)
        self._transition(dep_id, DeploymentState.loading, f"starting {label}")
        handle = await self._start_engine(dep_id, spec)
        self._save(self._deps[dep_id].model_copy(update={"engine_args": handle.args, "pid": handle.pid,
                                                         "upstream_port": _port(handle.base_url)}))
        await adapter.wait_ready(handle, on_progress=lambda t: self._progress(dep_id, t),
                                 **_load_budget(spec.plan, adapter))

        report = await adapter.observe(handle, spec, validation)
        self._save(self._deps[dep_id].model_copy(update={"placement": report.observed,
                                                         "participation_verified": report.verified,
                                                         "participation_detail": report.detail}))
        if (spec.plan.rpc_endpoints or spec.plan.engine == "exo") and not report.verified:
            raise EngineAdapterError("no_peer_work", f"a planned peer received no layers: {report.detail}")
        if spec.plan.tier in ("stream", "moe_offload") and not report.verified:
            raise EngineAdapterError("stream_not_attached" if spec.plan.tier == "stream" else "moe_not_file_backed",
                                     f"the planned host residency is not in effect: {report.detail}")
        return handle, report.detail

    async def _warm_and_activate(self, dep_id: str, handle: DeploymentHandle, detail: str) -> None:
        """One short request through the engine (the probe), then ready and watched."""
        self._transition(dep_id, DeploymentState.warming, "first request")
        if self.warmup:
            dep = self._deps[dep_id]
            await asyncio.wait_for(self.warmup(dep, handle),
                                   timeout=warmup_timeout(dep.plan, self.adapter_for(dep.plan.engine)))
        self._transition(dep_id, DeploymentState.ready, detail, progress="ready")
        self._tasks[dep_id] = asyncio.get_running_loop().create_task(self._watch(dep_id))

    def _chain_root(self, dep: Deployment) -> str:
        """The first deployment of the replacement chain `dep` belongs to (its leases share the chain's revisions)."""
        seen = {dep.id}
        while dep.replaces is not None and dep.replaces not in seen:
            seen.add(dep.replaces)
            try:
                dep = self.get(dep.replaces)
            except DeploymentError:  # the record is gone: the oldest one still known starts the chain
                return dep.replaces
        return dep.id

    def _require_nodes(self, plan: Plan) -> None:
        for node_id in {a.node_id for a in plan.assignments}:
            node = self._nodes.get(node_id)
            if node is None or node.state != NodeState.eligible:
                if node is None:
                    raise DeploymentError(409, "node_not_eligible", f"peer {node_id[:8]} is no longer in the session")
                why = f" ({node.state_reason})" if node.state_reason else ""
                raise DeploymentError(409, "node_not_eligible",
                                      f"{node.display_name} is {node.state.value}{why}; check the plan again",
                                      fix="wait for the friend to show 'Ready to contribute', or untick them")

    def _spec(self, dep_id: str, plan: Plan) -> DeploymentSpec:
        art = self._models.get(plan.model_id)
        return DeploymentSpec(deployment_id=dep_id, plan=plan, model_path=self._models.primary_path(plan.model_id),
                              alias=plan.model_id, artifact=art, model_dir=self._models.model_dir(plan.model_id),
                              mmproj_path=self._projector_for(plan))

    def _projector_for(self, plan: Plan) -> Path | None:
        """The verified projector when the plan loads one. plan() sets engine_options["mmproj"]; a plan made elsewhere
        (the tuner, a replan) without the key loads a verified projector."""
        return self._models.projector_path(plan.model_id) if plan.engine_options.get("mmproj", True) else None

    def _rpc_counts(self, plan: Plan) -> dict[str, int]:
        nodes = [n for node_id in {a.node_id for a in plan.assignments if a.endpoint}
                 if (n := self._nodes.get(node_id)) is not None]
        return rpc_device_counts(nodes)

    async def _start_engine(self, dep_id: str, spec: DeploymentSpec) -> DeploymentHandle:
        """Start llama-server and register its handle, even when a cancel lands while it is starting.

        prepare() starts the process from a worker thread that a cancel cannot interrupt, so a cancel waits for the
        start to finish and registers the handle before re-raising; the cancellation cleanup then stops the process.
        """
        task = asyncio.ensure_future(self._adapter_of(dep_id).prepare(spec))
        cancelled = False
        while not task.done():
            try:
                await asyncio.wait({task})  # like a shield: cancelling this wait does not cancel the start
            except asyncio.CancelledError:
                cancelled = True
        if not task.cancelled() and task.exception() is None:
            self._handles[dep_id] = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return task.result()

    async def _watch(self, dep_id: str) -> None:
        while True:
            await asyncio.sleep(2.0)
            dep = self._deps.get(dep_id)
            handle = self._handles.get(dep_id)
            if dep is None or dep.state in TERMINAL_STATES or handle is None:
                return
            adapter = self._adapter_of(dep_id)
            if not adapter.process_alive(handle):
                label = ENGINE_PROCESS_LABEL.get(dep.plan.engine, dep.plan.engine)
                text = adapter.read_log(handle)
                explain = getattr(adapter, "failure_reason", None) or detect_failure
                reason = explain(text) or f"{label} exited unexpectedly"
                await self._cleanup(dep_id, DeploymentState.failed, reason)
                return

    async def _cleanup(self, dep_id: str, final: DeploymentState, reason: str) -> Deployment:
        """Stop llama-server, release leases and record the terminal state, once per deployment.

        Every caller awaits one shared task through a shield, so cancelling a caller cannot cut the cleanup in half.
        The first caller's state and reason win; an already terminal deployment is returned as it is.
        """
        task = self._cleanups.get(dep_id)
        if task is None:
            if self._deps[dep_id].state in TERMINAL_STATES:
                return self._deps[dep_id]
            task = asyncio.get_running_loop().create_task(self._stop_and_release(dep_id, final, reason))
            self._cleanups[dep_id] = task
        return await asyncio.shield(task)

    async def _stop_and_release(self, dep_id: str, final: DeploymentState, reason: str) -> Deployment:
        try:
            self._cancel_reason.pop(dep_id, None)
            handle = self._handles.pop(dep_id, None)
            notes = []
            if self.interrupt:
                await self.interrupt(dep_id)
            if handle is not None:
                result = await self._adapter_of(dep_id).stop(handle)
                notes.append(result.detail)
                label = ENGINE_PROCESS_LABEL.get(self._deps[dep_id].plan.engine, "engine")
                if not result.clean:  # the process may still hold its memory: keep the leases that account for it
                    who = f"{label} pid {handle.pid}" if handle.pid else label  # pid 0: no local process (EXO)
                    notes.append(f"{who} may still be running, so its leases are held "
                                 "until the next gateway start reconciles them")
                    error = reason if final == DeploymentState.failed else f"{label} did not stop ({reason})"
                    return self._transition(dep_id, DeploymentState.failed, reason, error=error,
                                            cleanup="; ".join(notes))
            notes.append(self._release_leases(dep_id))
            if final == DeploymentState.failed:
                return self._transition(dep_id, final, reason, error=reason, cleanup="; ".join(notes))
            return self._transition(dep_id, final, reason, cleanup="; ".join(notes))
        finally:
            self._cleanups.pop(dep_id, None)

    def _release_leases(self, dep_id: str) -> str:
        """Release the stopped deployment's leases for its own revision; the note says what happened."""
        try:
            return f"released {self._leases.release(dep_id, revision=self._deps[dep_id].revision)} lease(s)"
        except LeaseConflict as exc:  # a newer revision of the chain reserved after this one: never free its memory
            log.warning("lease release for deployment %s refused: %s", dep_id, exc)
            return f"lease release refused ({exc}); the next gateway start reconciles them"

    async def _cancel_task(self, dep_id: str, final: DeploymentState, reason: str) -> None:
        """Cancel the deployment's startup or watch task and wait until it has run its own cleanup."""
        task = self._tasks.get(dep_id)
        if task is None or task.done():
            return
        self._cancel_reason.setdefault(dep_id, (final, reason))  # the first canceller's outcome wins
        task.cancel()
        await asyncio.wait({task})  # a cancel of this caller still propagates; the task's outcome is read below
        self._cancel_reason.pop(dep_id, None)
        if not task.cancelled() and task.exception() is not None:
            log.debug("task for %s ended: %r", dep_id, task.exception())

    # ------------------------------------------------------------------ stop / loss / restart
    async def stop(self, dep_id: str, reason: str = "stopped by owner") -> Deployment:
        dep = self.get(dep_id)
        if dep.state in TERMINAL_STATES:
            return dep
        await self._cancel_task(dep_id, DeploymentState.stopped, reason)
        if self._deps[dep_id].state not in TERMINAL_STATES and dep_id not in self._cleanups:
            self._transition(dep_id, DeploymentState.draining, reason)
        return await self._cleanup(dep_id, DeploymentState.stopped, reason)

    async def on_node_lost(self, node_id: str, reason: str) -> None:
        msg = f"peer {node_id[:8]} lost: {reason}"
        for dep_id in self.busy(node_id):
            await self._cancel_task(dep_id, DeploymentState.failed, msg)
            await self._cleanup(dep_id, DeploymentState.failed, msg)

    # ------------------------------------------------------------------ replace (reload at a request boundary)
    async def replace(self, dep_id: str, plan: Plan, reason: str, fallback: Plan | None = None) -> Deployment:
        """Swap the ready deployment for `plan` as revision old + 1: check that the plan would be admitted once the old
        leases are gone (else 409 not_admitted and nothing changes), drain, stop the old engine (its leases are released
        only once its process is gone), then admit, load, verify and warm the new plan through the same steps as
        create(). Generation in flight is not migrated; requests that arrive meanwhile find no ready model.

        If the new deployment fails, the previous plan is redeployed under a further revision when its nodes and pools
        still admit it, else `fallback`, else the last attempt is returned failed with the whole reason. Old and new
        never run an engine or hold leases at the same time. Returns the deployment that ends up serving, or not.
        """
        async with self._create_lock:
            old = self._replaceable(dep_id, plan, fallback)
            target = plan.model_copy(update={"revision": old.revision + 1})
            # admit before drain: a plan whose peers or pools cannot take it even once the old leases are gone is
            # refused while the old deployment still serves (the reservation after the stop re-checks it for real)
            problem = self._admission_problem(target, releasing=dep_id)
            if problem is not None:
                text = f"plan {target.id[:8]} is not admitted ({problem}); the deployment keeps serving"
                self._announce("deployment.replace_failed", old, target.revision, text, outcome="refused")
                raise DeploymentError(409, "not_admitted", text)
            self._replacing = dep_id
            try:
                self._announce("deployment.replacing", old, target.revision,
                               f"replacing with plan {target.id[:8]}: {reason}", plan_id=target.id, reason=reason)
                await self._retire(old, target.revision, reason)
                new = await self._attempt(target, old, f"replaces {dep_id[:8]}: {reason}")
                if new.state == DeploymentState.ready:
                    return self._replaced(old, new, "activated")
                if new.state == DeploymentState.stopped:  # stopped by the owner while it loaded: nothing to recover
                    return new
                return await self._recover(old, new, fallback)
            finally:
                self._replacing = None

    def _replaceable(self, dep_id: str, plan: Plan, fallback: Plan | None) -> Deployment:
        old = self.get(dep_id)
        if old.state != DeploymentState.ready or dep_id not in self._handles:
            raise DeploymentError(409, "not_ready", f"deployment {dep_id[:8]} is {old.state.value}; only a ready "
                                  "deployment can be replaced")
        if old.purpose == "tune":
            raise DeploymentError(409, "tuning", "a tuner trial is not replaced; it ends with its tune")
        for candidate in (plan, fallback):
            if candidate is not None and not candidate.feasible:
                raise DeploymentError(422, "plan_not_feasible",
                                      f"plan {candidate.id[:8]}: " + "; ".join(candidate.reasons))
            if candidate is not None and candidate.model_id != old.plan.model_id:
                raise DeploymentError(422, "model_mismatch", f"plan {candidate.id[:8]} serves {candidate.model_id}, "
                                      f"not {old.plan.model_id}; a replacement keeps the served model")
            if candidate is not None:
                self.adapter_for(candidate.engine)  # 422 engine_unavailable before anything is stopped
        return old

    def _announce(self, type_: str, dep: Deployment, revision: int, text: str, **data: object) -> None:
        self._events.publish(type_, f"Deployment {dep.id[:8]} (revision {revision}): {text}", deployment_id=dep.id,
                             revision=revision, **data)

    async def _retire(self, old: Deployment, revision: int, reason: str) -> None:
        """Drain, then stop the old engine. Without a clean stop nothing new starts: its memory may still be in use."""
        hold = await self._drain(old, revision)
        try:
            state = self._deps[old.id].state
            if state != DeploymentState.ready:  # stopped by the owner or a lost peer while it drained
                raise DeploymentError(409, "not_ready", f"deployment {old.id[:8]} became {state.value} while it "
                                      "drained; nothing was replaced")
            self._announce("deployment.drained", old, revision, "no request in flight; stopping it")
            stopped = await self.stop(old.id, f"replaced: {reason}")
        finally:
            hold.release()
        if stopped.state != DeploymentState.stopped:  # a lost peer or a dead engine ended it first, or the stop failed
            held = ("its leases stay held" if self._holds_leases(old.id) else "its leases were released")
            text = (f"deployment {old.id[:8]} ended {stopped.state.value} instead of stopped "
                    f"({stopped.error or stopped.cleanup}); {held} and no new engine was started")
            self._announce("deployment.replace_failed", old, revision, text, outcome="failed")
            raise DeploymentError(500, "replace_failed", text)

    async def _drain(self, old: Deployment, revision: int) -> InferenceHold:
        """Exclusive inference on the old deployment: waits for the request in flight and keeps new ones out."""
        if self.drain is None:
            return _NoHold()
        try:
            return await self.drain(old, self.drain_timeout_s)
        except (DeploymentError, TimeoutError) as exc:
            text = (f"a request was still running after {self.drain_timeout_s:.0f} s ({exc}); the deployment keeps "
                    "serving and nothing was replaced")
            self._announce("deployment.replace_failed", old, revision, text, outcome="refused")
            raise DeploymentError(409, "drain_timeout", text) from exc

    async def _attempt(self, plan: Plan, previous: Deployment, detail: str) -> Deployment:
        """Run `plan` as a new deployment that replaces `previous`, and wait until it is ready or terminal."""
        dep = Deployment(id=uuid.uuid4().hex, plan=plan, state=DeploymentState.planned,
                         history=[StateTransition(state=DeploymentState.planned, at=utcnow(), detail=detail)],
                         created_at=utcnow(), purpose=previous.purpose, revision=plan.revision, replaces=previous.id)
        task = self._launch(dep)
        await asyncio.wait({task})  # a cancel of the caller leaves the run going; the run cleans up its own failure
        return self._deps[dep.id]

    async def _recover(self, old: Deployment, failed: Deployment, fallback: Plan | None) -> Deployment:
        """After a failed new load: the previous plan if its nodes and pools still admit it, else the fallback."""
        notes = [f"revision {failed.revision} failed: {failed.error}"]
        last = failed
        for outcome, plan in (("rolled_back", old.plan), ("fell_back", fallback)):
            if plan is None:
                continue
            if self._holds_leases(last.id):
                notes.append(f"deployment {last.id[:8]} may still be running, so nothing else was started")
                break
            target = plan.model_copy(update={"revision": last.revision + 1})
            what = "previous plan" if outcome == "rolled_back" else f"fallback plan {target.id[:8]}"
            problem = self._admission_problem(target)
            self._announce("deployment.recovery", last, target.revision, f"{what}: " + (problem or "admitted"),
                           outcome=outcome, admitted=problem is None)
            if problem is not None:
                notes.append(f"{what} not admitted: {problem}")
                continue
            attempt = await self._attempt(target, last, f"{outcome.replace('_', ' ')} after {notes[0]}")
            if attempt.state == DeploymentState.ready:
                return self._replaced(old, attempt, outcome)
            if attempt.state == DeploymentState.stopped:
                return attempt
            notes.append(f"{what} failed: {attempt.error}")
            last = attempt
        return self._replace_failed(old, last, "; ".join(notes))

    def _admission_problem(self, plan: Plan, releasing: str | None = None) -> str | None:
        """Why `plan` cannot start right now (a peer gone or not eligible, a pool without room), else None. The
        leases of `releasing` count as released."""
        try:
            self._require_nodes(plan)
        except DeploymentError as exc:
            return str(exc)
        return self._leases.admits(requests_from_plan(plan, lease_budgets(plan, self._nodes)), releasing)

    def _holds_leases(self, dep_id: str) -> bool:
        return any(r["deployment_id"] == dep_id for r in self._leases.active())

    def _replaced(self, old: Deployment, new: Deployment, outcome: str) -> Deployment:
        self._announce("deployment.replaced", new, new.revision,
                       f"{outcome.replace('_', ' ')}; replaced {old.id[:8]} (revision {old.revision})",
                       replaced=old.id, outcome=outcome)
        return new

    def _replace_failed(self, old: Deployment, last: Deployment, why: str) -> Deployment:
        text = f"replacement of {old.id[:8]} failed: {why}"
        dep = self._save(self._deps[last.id].model_copy(update={"error": text}))
        self._announce("deployment.replace_failed", dep, dep.revision, text, replaced=old.id, outcome="failed")
        return dep

    def reconcile_on_start(self) -> list[str]:
        """Fail what was active and release stale leases. An engine with a cleanup_orphan hook may have left something
        running with no local process (an EXO instance on the Macs): its leases are held for cleanup_orphans()."""
        notes = self._sup.reconcile_orphans()
        survivors = self._surviving_deployments()  # their llama-server still holds memory: keep those leases
        leased = {r["deployment_id"] for r in self._leases.active()}
        for row in self._store.query("SELECT doc FROM deployments"):
            dep = Deployment.model_validate(json.loads(row["doc"]))
            active = dep.state not in TERMINAL_STATES
            orphan = (active or dep.id in leased) and dep.id not in survivors and self._has_orphan_hook(dep)
            label = ENGINE_PROCESS_LABEL.get(dep.plan.engine, dep.plan.engine)
            if active:
                cleanup = (f"the {label} it may have left running is removed after startup; leases held until then"
                           if orphan else "; ".join(notes) or "no orphaned process found")
                history = [*dep.history, StateTransition(state=DeploymentState.failed, at=utcnow(),
                                                         detail="gateway restarted")]
                self._save(dep.model_copy(update={"state": DeploymentState.failed, "history": history,
                                                  "error": "gateway restarted while deployment was active",
                                                  "cleanup": cleanup}))
                notes.append(f"deployment {dep.id[:8]} marked failed after restart")
            if orphan:
                self._orphans[dep.id] = None if active else dep.cleanup
                notes.append(f"deployment {dep.id[:8]}: leases held until the {label} it left running is removed")
        released = self._leases.reconcile(survivors | set(self._orphans))
        if released:
            notes.append(f"released {released} stale lease(s)")
        if survivors:
            notes.append(f"kept the leases of {len(survivors)} deployment(s) whose llama-server could not be stopped")
        return notes

    def _has_orphan_hook(self, dep: Deployment) -> bool:
        return getattr(self._adapters.get(dep.plan.engine), "cleanup_orphan", None) is not None

    async def cleanup_orphans(self) -> list[str]:
        """After a restart (from the gateway's startup): each engine removes what it left running for the deployments
        reconcile_on_start held. Leases are released only for those the engine confirms gone."""
        pending, self._orphans = self._orphans, {}
        return [await self._cleanup_orphan(dep_id, kept) for dep_id, kept in pending.items()]

    async def _cleanup_orphan(self, dep_id: str, kept: str | None) -> str:
        dep = self.get(dep_id)
        try:
            gone, detail = await self._adapters[dep.plan.engine].cleanup_orphan(dep)
        except Exception as exc:  # an unreachable engine is no proof that nothing is left: keep the leases
            log.warning("orphan cleanup for deployment %s failed", dep_id, exc_info=True)
            gone, detail = False, f"cleanup failed: {str(exc) or exc.__class__.__name__}"
        text = (f"{detail}; released {self._leases.release(dep_id)} lease(s)" if gone else
                f"{detail}; leases held until a later gateway start confirms it is gone")
        story = (kept or "").split(ORPHAN_NOTE, 1)[0].rstrip("; ")  # the stop's own notes, minus an earlier attempt
        self._save(dep.model_copy(update={"cleanup": "; ".join(t for t in (story, ORPHAN_NOTE + text) if t)}))
        return f"deployment {dep_id[:8]}: {text}"

    def _surviving_deployments(self) -> set[str]:
        keys = (str(r.get("key", "")) for r in self._sup.recorded())
        return {k.removeprefix(PROCESS_KEY_PREFIX) for k in keys if k.startswith(PROCESS_KEY_PREFIX)}

    async def shutdown(self) -> None:
        dep = self.active()
        if dep is not None:
            await self.stop(dep.id, "gateway shutting down")
