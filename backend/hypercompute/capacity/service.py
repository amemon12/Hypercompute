"""The capacity layer as one gateway service: pools, optimizer plans and explanations, and the replan evaluator.

This file fixes the interface the HTTP routes, the CLI and the replan controller code against. The implementation
lives in capacity/bridge.py (fresh inventory and measurements to optimizer inputs, optimizer candidates to Plans).
Every method returns JSON-ready data or schema models; none of them starts or stops an engine.

Plans are scored for a workload of the request's context, DEFAULT_PROMPT_TOKENS prompt tokens and
DEFAULT_OUTPUT_TOKENS output tokens (PlanRequest carries neither), cold. Every number is a prediction; a plan says which
priors it used and is `heuristic` unless all of its coefficients were measured on this hardware.

The replan evaluator searches under the limits stored on the running plan (Plan.limits: tiers, selected peers,
strategy) and returns a services.replan.Proposal whose current_prediction is the running layout scored on the same
inputs, with the CPU residency the deployment runs (running_split).

Mixture-of-experts models are left to the planner: the optimizer's layer search does not model expert offload, so
plan() raises CapacityUnavailable("moe") (the selector keeps the planner's moe_offload plan), explain() shows the
planner's plan and its estimates, and the evaluator proposes nothing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from .. import config
from ..schemas import Deployment, NodeState, NodeView, Plan, PlanLimits, PlanPrediction, PlanRequest, PlanStrategy
from ..services import planner
from ..services.deployments import assign_rpc_names, cpu_budget, local_budgets, remote_budgets, rpc_device_counts
from ..services.models import ModelError
from ..services.replan import Proposal
from . import bridge, costs, runtime
from .manifest import ManifestError
from .measurements import MeasurementStore
from .optimizer import OptimizeResult, evaluate_layout, optimize
from .types import ModelManifest, PathClass, PlacementCandidate, StagePlan, Workload

if TYPE_CHECKING:
    from ..services.leases import LeaseManager
    from ..services.models import ModelService
    from ..services.nodes import NodeRegistry

log = logging.getLogger("hypercompute.capacity.service")

Objective = Literal["speed", "fit"]
PathLookup = Callable[[str], PathClass]  # node id -> "lan" / "direct" / "relay" / "unknown" (from the transport)

MAX_CANDIDATES = 256  # the whole ranked pool: at most two layouts per device order, 65 orders with 4 peer devices
SHOWN_CANDIDATES = 5
DEFAULT_PROMPT_TOKENS = bridge.DEFAULT_PROMPT_TOKENS
DEFAULT_OUTPUT_TOKENS = bridge.DEFAULT_OUTPUT_TOKENS
MOE_NOTE = ("mixture-of-experts model: the optimizer's layer search does not model expert offload (routed experts on "
            "the host CPU, file-backed), so the planner's plan is used; its numbers are estimates from priors")
STRATEGY_REASON = ("strategy: capacity optimizer (exact search over a host CPU prefix, peer spans and the local GPU "
                   f"suffix; J = load + prefill of {DEFAULT_PROMPT_TOKENS} tokens + {DEFAULT_OUTPUT_TOKENS} x decode)")
UNIFIED_REFUSAL = ("; this machine's CPU and GPU share one unified pool and the optimizer cannot place the planner's "
                   "layout within it, so that plan is not marked feasible either")


class CapacityUnavailable(RuntimeError):
    """The optimizer cannot plan this request (no GGUF manifest, not downloaded, engine not llama.cpp, ...)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Search:
    inputs: bridge.BridgeInputs
    result: OptimizeResult


@dataclass(frozen=True)
class _Scored:
    candidate: PlacementCandidate | None
    why: str = ""
    spans: tuple[bridge.Span, ...] = ()


def _unknown_path(_node_id: str) -> PathClass:
    return "unknown"


def _no_deployment_ready(_deployment_id: str) -> bool:
    return False


def _warn(plan: Plan, *notes: str) -> Plan:
    return plan.model_copy(update={"warnings": [*plan.warnings, *notes]})


def _unified_host(inputs: bridge.BridgeInputs) -> bool:
    """This machine's CPU and local GPU allocate from one pool (a GB10's or a Mac's unified memory)."""
    gpu = inputs.host.gpu
    return gpu is not None and gpu.pool_id == inputs.host.cpu.pool_id


def _over_unified_pool(plan: Plan, scored: _Scored) -> str | None:
    """Why the planner's plan cannot be kept on a unified host the optimizer found no feasible layout on: its layout
    scored over budget, or it pages CPU layers from disk (gpu_disk), which the optimizer cannot score; else None."""
    if scored.candidate is not None:
        return None if scored.candidate.feasible else "; ".join(scored.candidate.reasons[:2]) or "over budget"
    return scored.why if plan.tier == "gpu_disk" else None


def _refused(plan: Plan, why: str) -> Plan:
    return plan.model_copy(update={"feasible": False, "fit_label": "does_not_fit", "reasons": [*plan.reasons, why]})


def _from_bridge(exc: bridge.BridgeError) -> CapacityUnavailable:
    """A BridgeError as CapacityUnavailable: its own code (app.CAPACITY_STATUS classifies it), else "bridge"."""
    return CapacityUnavailable(exc.code or "bridge", str(exc))


def _bridge_request(req: PlanRequest, workload: Workload, releasing: str | None = None) -> bridge.Request:
    """What the request allows the search: no host CPU units with the RAM tier off, no streaming with either off."""
    return bridge.Request(workload=workload, allow_cpu=req.allow_ram_tier,
                          allow_stream=req.allow_ram_tier and req.allow_capacity_mode, releasing=releasing)


def replan_request(plan: Plan) -> PlanRequest:
    """The request a replan of `plan` searches under: the limits the plan was made with (Plan.limits). A plan stored
    before limits existed keeps its strategy and allows every tier and every eligible peer, as replans did before."""
    lim = plan.limits
    extra: dict[str, Any] = {"strategy": plan.strategy} if lim is None else {
        "strategy": lim.strategy, "node_ids": lim.node_ids, "allow_ram_tier": lim.allow_ram_tier,
        "allow_capacity_mode": lim.allow_capacity_mode}
    try:
        return PlanRequest(model_id=plan.model_id, ctx_size=plan.ctx_size, **extra)
    except ValidationError as exc:
        raise CapacityUnavailable("bad_request", f"plan {plan.id[:8]} has no valid plan request: {exc}") from exc


def limit_violation(plan: Plan, limits: PlanLimits | None, host_id: str) -> str | None:
    """Why `plan` leaves the request limits a replan keeps (a RAM or stream tier turned off, a peer not selected or
    any peer for local_only), else None. No limits (a plan stored before they existed): nothing to keep."""
    if limits is None:
        return None
    on_host_cpu = any(a.node_id == host_id and a.endpoint is None and a.device == bridge.HOST_CPU
                      for a in plan.assignments)
    if not limits.allow_ram_tier and (on_host_cpu or plan.cpu_layers):
        return "it keeps layers on this machine's CPU, and the request turned the host RAM tier off"
    streams = plan.capacity_mode or bool(plan.stream is not None and plan.stream.streamed_units)
    if streams and not (limits.allow_ram_tier and limits.allow_capacity_mode):
        return "it streams weights from the SSD, and the request turned capacity mode off"
    peers = sorted({a.node_id for a in plan.assignments if a.node_id != host_id})
    if peers and limits.strategy == PlanStrategy.local_only:
        return "it uses peers, and the request is local only"
    outside = [p[:8] for p in peers if limits.node_ids is not None and p not in limits.node_ids]
    if outside:
        return "it uses peers the request did not select: " + ", ".join(outside)
    return None


# --------------------------------------------------------------------------- the running CPU residency


def _bottleneck(stages: Sequence[StagePlan], decode: float) -> str:
    """optimizer._bottleneck's text for stages scored outside a search."""
    terms = [(t, f"{s.node_id}:{s.device} {name}") for s in stages
             for t, name in ((s.compute_ms, "compute"), (s.exposed_io_ms, "exposed storage I/O"), (s.hop_ms, "hop"))]
    value, label = max(terms, key=lambda t: t[0], default=(0.0, "none"))
    share = 100.0 * value / decode if decode > 0 else 0.0
    return f"{label} {value:.1f} ms of {decode:.1f} ms per token ({share:.0f} percent)"


def _running_cpu_stage(first: StagePlan, running: tuple[int, ...], slots: int, inputs: bridge.BridgeInputs
                       ) -> tuple[StagePlan, int]:
    """The host CPU stage with `running` streamed through `slots` slots, and its streamed bytes per token. The I/O
    terms are optimizer._build's: stream_io_ms over the streamed buffers, overlapped with compute when slots >= 2."""
    size = {u.index: u.buffer_bytes for u in inputs.manifest.units}
    stream = sum(size[u] for u in running)
    io = costs.stream_io_ms(stream, inputs.storage)
    stage_ms = costs.cpu_stage_ms(first.compute_ms, io, slots) if stream else first.compute_ms
    resident = tuple(u for u in first.units if u not in running)
    return replace(first, resident_units=resident, streamed_units=running, slots=slots if running else 0,
                   exposed_io_ms=stage_ms - first.compute_ms), stream


def running_split(c: PlacementCandidate, plan: Plan, inputs: bridge.BridgeInputs) -> PlacementCandidate:
    """`c`, the running plan's layout as evaluate_layout scored it, with the host CPU stage's storage I/O taken from
    the residency the deployment runs (the units its StreamSpec streams through its slots; none without a StreamSpec)
    instead of costs.split_cpu for today's budget, which evaluate_layout applies and cannot be given another split.

    Approximations left: the stage's memory demand and the candidate's feasibility stay split_cpu's for today's
    budget, the read rate is today's storage profile (not the one the plan was made with), and prefill and load are
    unchanged. Only decode (and J through it) and the streamed bytes per token follow the running split.
    """
    first = c.stages[0] if c.stages else None
    if first is None or (first.node_id, first.device) != (inputs.host.node_id, inputs.host.cpu.device):
        return c
    running = tuple(sorted(plan.stream.streamed_units)) if plan.stream is not None else ()
    slots = plan.stream.slots if plan.stream is not None and running else 0
    # one slot serializes reads and compute, two or more overlap them: the same units through K = 1 are another score
    if set(running) == set(first.streamed_units) and (not running or (slots >= 2) == (first.slots >= 2)):
        return c
    if not set(running) <= set(first.units):
        log.info("plan %s streams units outside its CPU span; scored with split_cpu", plan.id[:8])
        return c
    try:
        stage, stream = _running_cpu_stage(first, running, slots, inputs)
    except ValueError as exc:  # a storage profile without a positive read rate
        log.info("running split of plan %s not scored: %s", plan.id[:8], exc)
        return c
    delta = stage.exposed_io_ms - first.exposed_io_ms
    stages = (stage, *c.stages[1:])
    scored = set(first.streamed_units)
    before = sum(u.buffer_bytes for u in inputs.manifest.units if u.index in scored)
    note = (f"current layout scored with the residency it runs: {len(running)} streamed CPU unit(s) through {slots} "
            f"slot(s) from its StreamSpec, where split_cpu for today's budget streams {len(first.streamed_units)} "
            f"through {first.slots}; its memory demand is still split_cpu's")
    return replace(c, stages=stages, decode_ms=c.decode_ms + delta,
                   objective_ms=c.objective_ms + inputs.workload.output_tokens * delta,
                   stream_bytes_per_token=c.stream_bytes_per_token - before + stream,
                   bottleneck=_bottleneck(stages, c.decode_ms + delta),
                   assumptions=tuple(dict.fromkeys((*c.assumptions, *_stream_priors(running, inputs), note))))


def _stream_priors(running: Sequence[int], inputs: bridge.BridgeInputs) -> tuple[str, ...]:
    """The storage priors a streamed running split relies on, in optimizer._assumptions' words (D4: every prior the
    score used is named, also when split_cpu for today's budget streamed nothing and so named none)."""
    s = inputs.storage
    return tuple(f"{label} {c.value:.4g} {c.unit} ({c.provenance}: {c.note})"
                 for label, c in (("storage read", s.read_bytes_per_s), ("stream efficiency", s.stream_efficiency))
                 if running and c.provenance != "measured")


class CapacityService:
    """Construct with the coordinator's NodeRegistry, ModelService, LeaseManager and a MeasurementStore.

    deployment_ready(deployment_id) says whether a lease's deployment has loaded (its bytes are then inside the free
    observation); without it every lease counts as not loaded yet, the conservative reading. `path_of` prices each peer
    hop by its path class; the gateway sets it once the transport is up.
    """

    def __init__(self, nodes: NodeRegistry, models: ModelService, leases: LeaseManager,
                 measurements: MeasurementStore, *, deployment_ready: Callable[[str], bool] | None = None,
                 probes: bridge.Probes | None = None, models_dir: Path | None = None) -> None:
        self._nodes, self._models, self._leases, self._store = nodes, models, leases, measurements
        self._ready = deployment_ready or _no_deployment_ready
        self._probes = probes or bridge.Probes()
        self._models_dir = models_dir
        self.path_of: PathLookup = _unknown_path

    # ------------------------------------------------------------------ interface
    def pools(self) -> dict[str, Any]:
        """Fresh pool report: {"observed_at", "host_memory": {...}, "pools": [...], "excluded": [...]}.

        Each pool carries node_id, pool_id, kind, total/free/reserve/cap/admissible bytes, devices, source and age.
        """
        models_dir = self._models_dir or config.paths().models
        snap = self._snapshot(self._nodes.eligible_remote(None), models_dir, None)
        return bridge.pools_json(snap)

    def explain(self, model_id: str, *, ctx_size: int = 4096, prompt_tokens: int = 512, output_tokens: int = 256,
                warm: bool = False, node_ids: list[str] | None = None, slots: int = 2) -> dict[str, Any]:
        """Optimizer view of a model on the current nodes, without planning a deployment.

        Returns {"model_id", "workload", "feasible", "best": PlanPrediction dump or None, "plan": Plan dump or None,
        "candidates": [...], "explain": [lines], "rejected": [...], "assumptions": [...], "measured": [...]}.
        Raises CapacityUnavailable with a code when the model cannot be planned by the optimizer.
        """
        req = self._explain_request(model_id, ctx_size, node_ids)
        if prompt_tokens < 0 or output_tokens < 0 or slots < 1:
            raise CapacityUnavailable("bad_request", "prompt_tokens and output_tokens must be >= 0 and slots >= 1")
        workload = Workload(ctx_size=ctx_size, prompt_tokens=prompt_tokens, output_tokens=output_tokens, warm=warm)
        try:
            search = self._search(req, bridge.Request(workload=workload, slots=slots))
        except CapacityUnavailable as exc:
            if exc.code != "moe":
                raise
            return self._moe_explain(req, workload)
        result, inputs = search.result, search.inputs
        plan = self._own_plan(req, result.best, search) if result.best is not None else None
        head = inputs.manifest.n_layer
        return {"model_id": model_id, "workload": bridge.workload_json(workload), "feasible": result.best is not None,
                "best": plan.prediction.model_dump(mode="json") if plan and plan.prediction else None,
                "plan": plan.model_dump(mode="json") if plan else None,
                "candidates": [bridge.candidate_json(c, head) for c in result.candidates[:SHOWN_CANDIDATES]],
                "explain": result.explain(), "rejected": list(result.rejected),
                "assumptions": list(dict.fromkeys([*inputs.notes, *result.assumptions])),
                "measured": list(inputs.measured), "warnings": list(inputs.warnings)}

    def plan(self, req: PlanRequest, legacy: Plan | None = None) -> Plan:
        """An optimizer Plan for a llama.cpp GGUF request (with plan.prediction), or CapacityUnavailable.

        `legacy` is the existing planner's plan for the same request; when given it is scored too and the faster
        feasible one is returned, with the comparison in plan.warnings.
        """
        self._check_request(req)
        workload = Workload(ctx_size=req.ctx_size, prompt_tokens=DEFAULT_PROMPT_TOKENS,
                            output_tokens=DEFAULT_OUTPUT_TOKENS, warm=False)
        search = self._search(req, _bridge_request(req, workload))
        chosen, note = self._choose(req, search)
        own = self._own_plan(req, chosen, search, reasons=[note] if note else [])
        kept = own if legacy is None else self._compare(own, chosen, legacy, search)
        return kept.model_copy(update={"limits": PlanLimits.of(req)})  # also on the planner's plan when it is kept

    async def evaluate(self, current: Deployment, objective: Objective) -> Proposal | None:
        """The replan evaluator (services.replan.Evaluator): a replan Proposal for the deployment, or None. The
        proposal carries the running layout's fresh score as current_prediction."""
        if objective not in ("speed", "fit"):
            raise ValueError(f"objective must be speed or fit, not {objective!r}")
        return await asyncio.to_thread(self._evaluate, current, objective)

    def _moe_explain(self, req: PlanRequest, workload: Workload) -> dict[str, Any]:
        """explain() for an MoE model: the planner's plan for this host and the eligible peers (moe_offload when the
        accelerators cannot hold it whole), in explain()'s shape. Every number is the planner's estimate."""
        art, shards = self._artifact(req.model_id)
        shape = bridge.model_shape(self._manifest(art, shards), req.ctx_size)
        local = self._nodes.local
        remote = [] if req.strategy == PlanStrategy.local_only else self._nodes.eligible_remote(req.node_ids)
        ram_free = next((p.free_bytes for p in local.inventory.memory_pools if p.id in ("ram", "uma")), None)
        plan = planner.make_plan(req, shape, local_budgets(local), [b for n in remote for b in remote_budgets(n)],
                                 ram_free, cpu_budget(local))
        plan = assign_rpc_names(plan, rpc_device_counts(remote))
        priors = [f"decode {planner.DECODE_MS_PER_GB} ms per GB read (prior)",
                  f"peer round trip {planner.HOP_MS:.0f} ms per device per token (prior)"]
        if plan.moe is not None:
            priors.append(f"page-fault read rate {plan.moe.read_bytes_per_s / (1 << 30):.1f} GiB/s (prior)")
        candidates = [{"layout": f"{c.label}: {', '.join(c.devices)}", "feasible": c.feasible, "reasons": [c.reason]}
                      for c in plan.candidates]
        return {"model_id": req.model_id, "workload": bridge.workload_json(workload), "feasible": plan.feasible,
                "best": None, "plan": plan.model_dump(mode="json"), "candidates": candidates,
                "explain": [MOE_NOTE, *plan.reasons, *plan.warnings], "rejected": [] if plan.feasible else plan.reasons,
                "assumptions": [MOE_NOTE, *priors], "measured": [], "warnings": []}

    # ------------------------------------------------------------------ inputs
    def _snapshot(self, remote: Sequence[NodeView], storage_path: Path, releasing: str | None) -> bridge.Snapshot:
        try:
            return bridge.take_snapshot(self._nodes.local, remote, self._leases.active(), self._ready, self._probes,
                                        storage_path, releasing)
        except bridge.BridgeError as exc:
            raise _from_bridge(exc) from exc
        except ValueError as exc:  # pools.pools_report / admissible_bytes refuse inconsistent figures
            raise CapacityUnavailable("pools", f"pool report failed: {exc}") from exc
        except OSError as exc:  # the memory or free-disk probe could not read this machine
            raise CapacityUnavailable("io", f"cannot read this machine's memory or disk state: {exc}") from exc

    def _artifact(self, model_id: str) -> tuple[Any, list[Path]]:
        try:
            view = self._models.view(model_id)
        except ModelError as exc:
            raise CapacityUnavailable("unknown_model", str(exc)) from exc
        art = view.artifact
        if art.format != "gguf":
            raise CapacityUnavailable("not_gguf", f"{model_id} is a {art.format} artifact; the optimizer places "
                                      "llama.cpp GGUF units")
        if view.status != "downloaded":
            raise CapacityUnavailable("not_downloaded", f"{model_id} is {view.status.replace('_', ' ')}; the "
                                      "optimizer reads its exact unit sizes from the downloaded GGUF")
        try:
            first = Path(self._models.primary_path(model_id))
            return art, runtime.model_shards(first)
        except ModelError as exc:  # an unsafe file path in the catalog entry
            raise CapacityUnavailable("model_files", str(exc)) from exc
        except OSError as exc:  # the model directory cannot be read
            raise CapacityUnavailable("model_files", f"cannot list the files of {model_id}: {exc}") from exc

    def _manifest(self, art: Any, shards: list[Path]) -> ModelManifest:
        try:
            return self._probes.manifest(art, shards)
        except ManifestError as exc:
            raise CapacityUnavailable("manifest", f"cannot build the manifest of {art.id}: {exc}") from exc
        except OSError as exc:
            raise CapacityUnavailable("io", f"cannot read {art.id}: {exc}") from exc

    def _search(self, req: PlanRequest, want: bridge.Request) -> _Search:
        """Fresh inputs for the request's nodes and the optimizer's ranked layouts."""
        art, shards = self._artifact(req.model_id)
        local = self._nodes.local
        if local.state != NodeState.eligible:
            raise CapacityUnavailable("host_ineligible", f"this machine is {local.state.value}"
                                      + (f" ({local.state_reason})" if local.state_reason else ""))
        remote = [] if req.strategy == PlanStrategy.local_only else self._nodes.eligible_remote(req.node_ids)
        manifest = self._manifest(art, shards)
        if manifest.is_moe:
            raise CapacityUnavailable("moe", f"{req.model_id} is a mixture-of-experts model ({manifest.expert_count} "
                                      f"experts, {manifest.expert_used_count} per token); the optimizer's layer search "
                                      "does not model expert offload, so the planner's moe_offload plan is used")
        snap = self._snapshot(remote, shards[0], want.releasing)
        try:
            inputs = bridge.build_inputs(manifest, want, local, remote, snap, self._store, self.path_of,
                                         self._probes, shards)
            # 'run together' filters the whole ranked pool; other requests need only the layouts they show
            ranked = MAX_CANDIDATES if req.strategy == PlanStrategy.all_selected else SHOWN_CANDIDATES
            result = optimize(manifest, want.workload, inputs.host, inputs.peers, slots=inputs.slots,
                              max_candidates=ranked)
        except bridge.BridgeError as exc:
            raise _from_bridge(exc) from exc
        except ValueError as exc:  # a malformed measurement or profile: costs.merge and optimize refuse it
            raise CapacityUnavailable("optimizer_input", str(exc)) from exc
        except OSError as exc:  # the stream probe reads the GGUF tables and the storage calibration
            raise CapacityUnavailable("io", f"cannot read {req.model_id} or its storage calibration: {exc}") from exc
        return _Search(inputs=inputs, result=result)

    # ------------------------------------------------------------------ plans
    @staticmethod
    def _explain_request(model_id: str, ctx_size: int, node_ids: list[str] | None) -> PlanRequest:
        try:
            return PlanRequest(model_id=model_id, ctx_size=ctx_size, node_ids=node_ids,
                               strategy=PlanStrategy.fastest_fit)
        except ValidationError as exc:
            raise CapacityUnavailable("bad_request", str(exc)) from exc

    @staticmethod
    def _check_request(req: PlanRequest) -> None:
        if req.engine not in (None, "llama_rpc"):
            raise CapacityUnavailable("engine", f"the request forces {req.engine}; the optimizer plans llama.cpp")
        if req.objective != "interactive":
            raise CapacityUnavailable("objective", "the optimizer scores single-stream latency; a throughput plan "
                                      "keeps the planner's proportional split")
        if req.overflow != "auto":
            raise CapacityUnavailable("overflow_pinned", f"the request pins the GPU overflow to {req.overflow} (the "
                                      "tuner's alternative); the planner builds exactly that layout")

    @staticmethod
    def _choose(req: PlanRequest, search: _Search) -> tuple[PlacementCandidate | None, str | None]:
        """The candidate to plan: the best by J; for 'run together' the best one giving every peer device a span.
        Without a feasible layout, the closest overflowing one (its plan is infeasible and says why)."""
        result, inputs = search.result, search.inputs
        if req.strategy != PlanStrategy.all_selected:
            return result.best or (result.candidates[0] if result.candidates else None), None
        if inputs.left_out or inputs.local_left_out:
            raise CapacityUnavailable("participation", "'run together' places every selected device, and the "
                                      "optimizer cannot place "
                                      + ", ".join([*(f"local {n}" for n in inputs.local_left_out), *inputs.left_out]))
        wanted = {ref.key for ref in inputs.refs}
        if not wanted:
            raise CapacityUnavailable("no_peers", "'run together' needs at least one eligible peer device")
        fits = [c for c in result.candidates
                if c.feasible and wanted <= {f"{s.node_id}:{s.device}" for s in c.stages}]
        if not fits:
            raise CapacityUnavailable("participation", f"no feasible layout gives each of the {len(wanted)} selected "
                                      "peer device(s) a span")
        return fits[0], "strategy: run together (the best layout in which every selected peer device holds a span)"

    def _own_plan(self, req: PlanRequest, chosen: PlacementCandidate | None, search: _Search,
                  reasons: list[str] | None = None) -> Plan:
        result, inputs = search.result, search.inputs
        head = inputs.manifest.n_layer
        summaries = [bridge.candidate_summary(c, head, f"optimizer {i}")
                     for i, c in enumerate(result.candidates[:SHOWN_CANDIDATES], 1)]
        lines = result.explain()
        if chosen is not None and chosen is not result.best:
            lines = bridge.candidate_lines(chosen, head, "chosen") + lines
        try:
            return bridge.to_plan(req, chosen, inputs, reasons=[STRATEGY_REASON, *(reasons or []),
                                                                *(() if result.best else result.rejected)],
                                  warnings=list(inputs.warnings), candidates=summaries, explain=lines,
                                  context=bridge.context_notes(result))
        except bridge.BridgeError as exc:
            raise _from_bridge(exc) from exc

    @staticmethod
    def _score(plan: Plan, inputs: bridge.BridgeInputs, *, warm: bool) -> _Scored:
        """The plan's layout scored with the optimizer's formulas on these inputs (CPU split by costs.split_cpu)."""
        try:
            spans = bridge.plan_layout(plan, inputs)
            cand = evaluate_layout(inputs.manifest, replace(inputs.workload, warm=warm), inputs.host, inputs.peers,
                                   spans, slots=inputs.slots)
        except ValueError as exc:
            return _Scored(candidate=None, why=str(exc))
        return _Scored(candidate=cand, spans=tuple(spans))

    def _compare(self, own: Plan, chosen: PlacementCandidate | None, legacy: Plan, search: _Search) -> Plan:
        """The feasible plan with the lower predicted decode; ties and unscorable planner plans keep the planner's."""
        if not legacy.feasible:
            if own.feasible:
                why = "; ".join(legacy.reasons[:2]) or legacy.fit_label
                return _warn(own, f"capacity optimizer: the planner found no feasible plan ({why}); the optimizer's "
                                  f"{own.tier} plan is used")
            return _warn(legacy, "capacity optimizer: no feasible layout either: " + "; ".join(own.reasons[1:3]))
        scored = self._score(legacy, search.inputs, warm=False)
        if not own.feasible or chosen is None:
            text = "capacity optimizer: no feasible layout (" + "; ".join(own.reasons[1:3]) + ")"
            if _unified_host(search.inputs) and (why := _over_unified_pool(legacy, scored)) is not None:
                return _refused(legacy, f"{text}{UNIFIED_REFUSAL} ({why})")
            return self._legacy_kept(legacy, scored, search, text + "; the planner's plan is kept")
        if scored.candidate is None:
            if legacy.tier == "gpu_disk":
                return _warn(own, f"capacity optimizer: the planner's plan pages layers from disk (gpu_disk), which "
                                  f"the optimizer does not model; its {own.tier} plan is used instead")
            return _warn(legacy, f"capacity optimizer: the planner's plan could not be scored ({scored.why}); it is "
                                 f"kept. The optimizer's layout predicts {chosen.decode_ms:.1f} ms/token decode")
        ours, theirs = chosen.decode_ms, scored.candidate.decode_ms
        if ours < theirs:
            text = (f"capacity optimizer: its plan won, predicted decode {ours:.1f} ms/token against {theirs:.1f} "
                    f"for the planner's {legacy.tier} plan ({(theirs - ours) / theirs:.0%} lower)")
            summary = bridge.candidate_summary(scored.candidate, search.inputs.manifest.n_layer,
                                               f"planner {legacy.tier} plan (scored)")
            return own.model_copy(update={"warnings": [*own.warnings, text],
                                          "candidates": [*own.candidates, summary]})
        text = (f"capacity optimizer: the planner's {legacy.tier} plan was kept, predicted decode {theirs:.1f} "
                f"ms/token against {ours:.1f} for the optimizer's {own.tier} plan"
                + (f" ({(ours - theirs) / ours:.0%} lower)" if ours > theirs else " (equal predictions keep it)"))
        return self._legacy_kept(legacy, scored, search, text)

    @staticmethod
    def _legacy_kept(legacy: Plan, scored: _Scored, search: _Search, text: str) -> Plan:
        """The planner's plan with its optimizer score attached (when it has one) and the optimizer's best listed."""
        inputs, result = search.inputs, search.result
        head = inputs.manifest.n_layer
        update: dict[str, Any] = {"warnings": [*legacy.warnings, text]}
        if scored.candidate is not None:
            lines = bridge.candidate_lines(scored.candidate, head, "planner layout scored with the optimizer")
            update["prediction"] = bridge.prediction(scored.candidate, inputs, lines, bridge.context_notes(result))
        if result.best is not None:
            update["candidates"] = [*legacy.candidates, bridge.candidate_summary(result.best, head, "optimizer best")]
        return legacy.model_copy(update=update)

    # ------------------------------------------------------------------ replan evaluator
    def _evaluate(self, current: Deployment, objective: Objective) -> Proposal | None:
        """The deployment's request limits stay (replan_request): its strategy (local_only stays on this machine,
        'run together' keeps every eligible peer device; for "fit", recovery, the best executable layout when that is
        impossible now), the peers it selected, and the RAM and stream tiers it allowed. Both objectives take the best
        feasible layout; the running one (same spans and the same CPU units streamed) is never proposed."""
        plan = current.plan
        if plan.engine != "llama_rpc":
            raise CapacityUnavailable("engine", f"deployment {current.id[:8]} runs {plan.engine}; the optimizer "
                                      "plans llama.cpp only")
        if plan.engine_options.get("parallel", 1) != 1:  # the search plans one slot: a proposal would relaunch -np 1
            log.info("no replan proposal for deployment %s: it runs %s server slots", current.id[:8],
                     plan.engine_options.get("parallel"))
            return None
        req = replan_request(plan)
        workload = Workload(ctx_size=plan.ctx_size, prompt_tokens=DEFAULT_PROMPT_TOKENS,
                            output_tokens=DEFAULT_OUTPUT_TOKENS, warm=False)
        try:
            search = self._search(req, _bridge_request(req, workload, releasing=current.id))
        except CapacityUnavailable as exc:
            if exc.code != "moe":
                raise
            log.info("no replan proposal for deployment %s: %s", current.id[:8], exc)
            return None
        current_score = self._score(plan, search.inputs, warm=True)
        try:
            best, note = self._choose(req, search)
        except CapacityUnavailable as exc:  # 'run together' with a peer device that cannot be placed now
            if objective != "fit":
                log.info("no speed proposal for deployment %s: %s", current.id[:8], exc)
                return None
            best, note = search.result.best, f"'run together' is not possible now ({exc}); the best layout that fits"
        if best is None or not best.feasible or bridge.same_layout(plan, current_score.spans, best):
            return None
        return self._proposal(current, req, best, note, search, current_score)

    def _proposal(self, current: Deployment, req: PlanRequest, best: PlacementCandidate, note: str | None,
                  search: _Search, current_score: _Scored) -> Proposal | None:
        """The candidate as a named Plan with its reload and replay costs; None if it leaves the request limits (the
        search inputs already exclude what the limits turn off; this is the check that the plan really keeps them)."""
        inputs = search.inputs
        proposed = assign_rpc_names(self._own_plan(req, best, search, reasons=[note] if note else []),
                                    {ref.endpoint: ref.endpoint_devices for ref in inputs.refs})
        why = limit_violation(proposed, current.plan.limits, inputs.host.node_id)
        if why is not None:
            log.warning("no proposal for deployment %s: plan %s leaves the request limits (%s)", current.id[:8],
                        proposed.id[:8], why)
            return None
        return Proposal(plan=proposed, reload_ms=best.load_ms, replay_ms=self._replay_ms(best, inputs, req.ctx_size),
                        confidence=bridge.confidence(best, inputs),
                        current_prediction=self._current_prediction(current.plan, current_score, search))

    @staticmethod
    def _replay_ms(best: PlacementCandidate, inputs: bridge.BridgeInputs, ctx_size: int) -> float:
        """Prefill of a full context window on the candidate: the most a reload can make the next request redo."""
        replay = Workload(ctx_size=ctx_size, prompt_tokens=ctx_size, output_tokens=0, warm=True)
        return evaluate_layout(inputs.manifest, replay, inputs.host, inputs.peers, bridge.layout(best),
                               slots=inputs.slots).prefill_ms

    @staticmethod
    def _current_prediction(plan: Plan, scored: _Scored, search: _Search) -> PlanPrediction | None:
        """The running layout's warm score on the search's inputs, with the CPU residency it runs (running_split)."""
        if scored.candidate is None:
            log.info("current deployment layout not scored: %s", scored.why)
            return None
        cand = running_split(scored.candidate, plan, search.inputs)
        lines = bridge.candidate_lines(cand, search.inputs.manifest.n_layer, "current layout (warm)")
        return bridge.prediction(cand, search.inputs, lines, bridge.context_notes(search.result))
