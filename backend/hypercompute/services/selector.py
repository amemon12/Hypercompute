"""Engine selector: one plan per request, down the tier ladder of docs/ENGINE_TIERS.md.

llama.cpp first for GGUF artifacts (gpu -> gpu_ram -> gpu_disk inside planner.make_plan). vLLM takes over for
objective=throughput when a variant of the same logical model fits one local NVIDIA GPU. EXO is a candidate when
every selected peer is a Mac running EXO and an MLX variant exists; it serves only when forced or when llama.cpp
can do no better than disk paging. AirLLM (the disk tier) serves an HF variant only when no llama.cpp tier fits.
With a capacity hook (Inputs.capacity, wired to CapacityService.plan) the llama.cpp plan is the one of the planner's
and the capacity optimizer's with the lower predicted decode; when the optimizer cannot plan the request the planner's
plan stands with the reason in its warnings. The best measured profile for the same model, context and device set
that this request's limits allow then replaces the heuristic split.

Pure: every input is injected (DeploymentManager.plan wires the live ones), and it never starts a process.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from ..adapters.base import EngineAdapterError, artifact_engines
from ..adapters.llama_rpc import engine_flag_args
from ..schemas import (
    CandidateSummary,
    DeviceBudget,
    MeasuredProfile,
    ModelArtifact,
    NodeView,
    Plan,
    PlanRequest,
    PlanStrategy,
    utcnow,
)
from . import planner
from .planner import ModelShape
from .profiles import signature

log = logging.getLogger("hypercompute.selector")
INSTALL = "uv run hypercompute engine install --engine {engine}"
READY = ("downloaded", "engine_managed")
_RPC_NAME = re.compile(r"RPC(\d+)")

EnginePlanner = Callable[[str, ModelArtifact, PlanRequest], "Plan | None"]  # registry.plan_for for these nodes
CapacityPlanner = Callable[[PlanRequest, Plan], Plan]  # capacity.service.CapacityService.plan
# ProfileStore.best(model_id, sig, objective, accept): the lowest-score profile that accept takes
ProfileLookup = Callable[[str, str, str, Callable[[MeasuredProfile], bool]], "MeasuredProfile | None"]


def _all_downloaded(_model_id: str) -> str:
    return "downloaded"


@dataclass(frozen=True)
class Inputs:
    """Everything one plan depends on."""

    req: PlanRequest
    artifact: ModelArtifact
    shape: ModelShape | None  # GGUF shape of the requested artifact; None when not downloaded or not a GGUF
    local: list[DeviceBudget]
    remote: list[DeviceBudget]
    cpu: DeviceBudget | None
    host_ram_free: int | None
    artifacts: list[ModelArtifact] = field(default_factory=list)  # the manifest: variants share a logical_id
    remote_nodes: list[NodeView] = field(default_factory=list)
    available: frozenset[str] = frozenset({"llama_rpc"})  # engines with an adapter on this host
    status: Callable[[str], str] = _all_downloaded  # ModelView.status by model id
    plan_engine: EnginePlanner | None = None
    profile: ProfileLookup | None = None
    stream: planner.StreamCapability | None = None  # the bounded stream tier on this host for this GGUF
    # CapacityService.plan(req, legacy): the optimizer's plan or the faster of the two; None keeps the planner's plan
    capacity: CapacityPlanner | None = None


def choose_plan(inp: Inputs) -> Plan:
    plan = _forced(inp, inp.req.engine) if inp.req.engine else _ladder(inp)
    return _with_profile(inp, plan)


# --------------------------------------------------------------------------- engine choice


def _forced(inp: Inputs, engine: str) -> Plan:
    """req.engine set: that engine only, on the requested artifact or else its variant that the engine serves."""
    if engine == "llama_rpc":
        return _llama(inp)
    art = inp.artifact
    if engine in artifact_engines(art):
        return _engine_plan(inp, engine, art)
    variant = _variant(inp, lambda a: engine in artifact_engines(a))
    if variant is None:
        return _infeasible(inp, art.id, engine, f"{engine} cannot serve {art.id} (a {art.format} artifact) and the "
                           f"manifest lists no variant of it that {engine} can serve")
    return _warn(_engine_plan(inp, engine, variant), f"{engine} serves {variant.id}, the {variant.format} variant of "
                 f"{art.id} (a different model id)")


def _ladder(inp: Inputs) -> Plan:
    if inp.artifact.format != "gguf":
        return _own_engines(inp)
    llama = _llama(inp)
    tried: list[Plan] = []
    if inp.req.objective == "throughput" and (vllm := _vllm(inp)) is not None:
        if vllm.feasible:
            return _pick(vllm, [llama], f"vLLM serves {vllm.model_id} (a different model id than {inp.artifact.id}) "
                         "for objective=throughput: about the same single-user speed as llama.cpp, several times "
                         "the throughput with concurrent requests")
        tried.append(vllm)
    if (exo := _exo(inp)) is not None:
        if exo.feasible and (llama.capacity_mode or llama.fit_label == "does_not_fit"):
            return _pick(exo, [llama, *tried], f"EXO serves {exo.model_id} (a different model id) on the Macs, "
                         "because llama.cpp can fit this model only by paging layers from disk, or not at all")
        tried.append(exo)
    if llama.fit_label == "does_not_fit" and inp.req.allow_capacity_mode:
        air = _airllm(inp, llama)
        if air.feasible:
            return _pick(air, [llama, *tried], f"capacity mode: AirLLM serves {air.model_id}, the HF variant of "
                         f"{inp.artifact.id} (a different model id), because no llama.cpp tier fits. Every layer is "
                         "read from disk for every token: expect seconds to minutes per token.")
        if air is not llama:
            tried.append(air)
            llama = llama.model_copy(update={"reasons": [*llama.reasons, f"AirLLM could serve {air.model_id} in "
                                                          f"capacity mode, but: {'; '.join(air.reasons)}"]})
    return _pick(llama, tried, None)


def _llama(inp: Inputs) -> Plan:
    art = inp.artifact
    if art.format != "gguf":
        gguf = _variant(inp, lambda a: a.format == "gguf")
        return _infeasible(inp, art.id, "llama_rpc", f"llama.cpp serves GGUF files and {art.id} is a {art.format} "
                           "artifact" + (f"; plan its GGUF variant {gguf.id} instead" if gguf else ""))
    legacy = planner.make_plan(inp.req, inp.shape, inp.local, inp.remote, inp.host_ram_free, inp.cpu,
                               stream=inp.stream)
    return _with_capacity(inp, legacy)


def _with_capacity(inp: Inputs, legacy: Plan) -> Plan:
    """The capacity optimizer's choice between its own plan and the planner's; the planner's plan, with the reason,
    when the optimizer cannot plan this request."""
    if inp.capacity is None:
        return legacy
    # imported here: capacity.service imports services.deployments, which imports this module
    from ..capacity.service import CapacityUnavailable

    try:
        return inp.capacity(inp.req, legacy)
    except CapacityUnavailable as exc:
        return _warn(legacy, f"capacity optimizer not used ({exc.code}): {exc}")
    except ValueError as exc:
        return _warn(legacy, f"capacity optimizer not used (invalid input): {exc}")
    except Exception as exc:  # an optional optimizer must never break planning; logged with its traceback
        log.exception("the capacity optimizer failed for %s", inp.req.model_id)
        text = f"capacity optimizer failed ({type(exc).__name__}: {exc}); the planner's plan is used"
        return _warn(legacy, text[:400])


def _own_engines(inp: Inputs) -> Plan:
    """An HF or MLX artifact was requested: its own engines, fastest first; AirLLM only in capacity mode."""
    art = inp.artifact
    order = [e for e in ("vllm", "exo", "airllm") if e in artifact_engines(art)]
    if not order:
        return _llama(inp)
    plans: list[Plan] = []
    for engine in order:
        if engine == "airllm" and not inp.req.allow_capacity_mode:
            plans.append(_infeasible(inp, art.id, engine, "AirLLM is capacity mode, which this request turns off "
                                     "(allow_capacity_mode=false)"))
            continue
        plan = _engine_plan(inp, engine, art)
        if plan.feasible:
            note = ("capacity mode: AirLLM reads every layer from disk for every token; expect seconds to minutes per "
                    "token" if engine == "airllm" else None)
            return _pick(plan, plans, note)
        plans.append(plan)
    best = next((p for p in plans if p.engine in inp.available), plans[0])
    return _pick(best, [p for p in plans if p is not best], None)


def _vllm(inp: Inputs) -> Plan | None:
    art = _variant(inp, lambda a: "vllm" in artifact_engines(a))
    if art is None:
        return None
    plan = _engine_plan(inp, "vllm", art)
    if plan.feasible and inp.status(art.id) not in READY:
        return plan.model_copy(update={"feasible": False, "reasons": [
            f"{art.id} is not downloaded; download it to serve this model with vLLM", *plan.reasons]})
    return plan


def _runs_exo(node: NodeView) -> bool:
    return node.inventory.os.system == "macos" and any(
        e.engine == "exo" and e.status == "running" for e in node.inventory.extra_engines)


def _exo(inp: Inputs) -> Plan | None:
    if not inp.remote_nodes or not all(_runs_exo(n) for n in inp.remote_nodes):
        return None
    art = _variant(inp, lambda a: a.format == "mlx" and "exo" in artifact_engines(a))
    return _engine_plan(inp, "exo", art) if art else None


def _airllm(inp: Inputs, llama: Plan) -> Plan:
    """The AirLLM plan for an HF variant of the same logical model, or `llama` itself when there is none."""
    art = _variant(inp, lambda a: a.format == "hf" and "airllm" in artifact_engines(a))
    return _engine_plan(inp, "airllm", art) if art else llama


def _engine_plan(inp: Inputs, engine: str, art: ModelArtifact) -> Plan:
    """An optional engine's own plan (adapters.registry.plan_for), or an infeasible plan that says why not."""
    if engine not in artifact_engines(art):
        return _infeasible(inp, art.id, engine, f"{engine} cannot serve {art.id} (a {art.format} artifact)")
    if engine not in inp.available:
        return _infeasible(inp, art.id, engine, f"the {engine} engine is not installed on this host",
                           fix=INSTALL.format(engine=engine))
    if inp.plan_engine is None:
        return _infeasible(inp, art.id, engine, f"no {engine} planner is wired into this gateway")
    req = inp.req.model_copy(update={"model_id": art.id, "engine": engine})
    try:
        plan = inp.plan_engine(engine, art, req)
    except Exception as exc:  # an optional engine's planner must never break planning for the others
        log.exception("the %s planner failed for %s", engine, art.id)
        return _infeasible(inp, art.id, engine, f"the {engine} planner failed: {exc}"[:400])
    if plan is None:
        return _infeasible(inp, art.id, engine, f"the {engine} adapter module is missing from this build",
                           fix=INSTALL.format(engine=engine))
    if plan.engine != engine or plan.model_id != art.id:
        return _infeasible(inp, art.id, engine, f"the {engine} planner returned a plan for {plan.engine} / "
                           f"{plan.model_id}; that is a bug in the {engine} adapter")
    return plan


def _variant(inp: Inputs, want: Callable[[ModelArtifact], bool]) -> ModelArtifact | None:
    """Another artifact of the same logical model that `want` accepts, downloaded ones first."""
    lid = inp.artifact.logical_id
    if not lid:
        return None
    found = [a for a in inp.artifacts if a.logical_id == lid and a.id != inp.artifact.id and want(a)]
    found.sort(key=lambda a: inp.status(a.id) not in READY)
    return found[0] if found else None


def _infeasible(inp: Inputs, model_id: str, engine: str, reason: str, fix: str | None = None) -> Plan:
    return Plan(id=uuid.uuid4().hex, model_id=model_id, strategy=inp.req.strategy, ctx_size=inp.req.ctx_size,
                feasible=False, fit_label="fit_unknown", heuristic=True, assignments=[], device_list=[],
                tensor_split=[], rpc_endpoints=[], cpu_resident_bytes=0,
                reasons=[reason, *([f"fix: {fix}"] if fix else [])], warnings=[], candidates=[],
                created_at=utcnow(), engine=engine)  # type: ignore[arg-type]


def _summary(p: Plan) -> CandidateSummary:
    why = [r for r in p.reasons if not r.startswith(("strategy:", "contributed /"))]
    reason = f"estimated fit ({p.tier} tier)" if p.feasible else ("; ".join(why[:2]) or p.fit_label)
    return CandidateSummary(label=f"{p.engine}: {p.model_id}", feasible=p.feasible, reason=reason[:400],
                            devices=sorted({f"{a.node_id}:{a.device}" for a in p.assignments}))


def _pick(chosen: Plan, others: list[Plan], note: str | None) -> Plan:
    """The chosen plan, with the other engines' plans as candidates and the reason for the choice."""
    return chosen.model_copy(update={"candidates": [*chosen.candidates, *(_summary(p) for p in others)],
                                     "warnings": [*chosen.warnings, *([note] if note else [])]})


def _warn(plan: Plan, note: str) -> Plan:
    return plan.model_copy(update={"warnings": [*plan.warnings, note]})


# --------------------------------------------------------------------------- measured profiles


def profile_devices(local: list[DeviceBudget], remote: list[DeviceBudget]) -> list[tuple[str, str]]:
    """What a measured profile is keyed by (with services.profiles.signature): every device available to the request,
    named as the planner names them (CUDA0 / MTL0 here, <host:port>#<index> on a peer). The tuner keys with this."""
    return [(b.node_id, b.device) for b in [*local, *remote]]


def _with_profile(inp: Inputs, plan: Plan) -> Plan:
    if (not inp.req.use_profiles or inp.profile is None or plan.engine != "llama_rpc" or inp.shape is None
            or plan.model_id != inp.artifact.id):
        return plan
    sig = signature(plan.model_id, inp.req.ctx_size, profile_devices(inp.local, inp.remote))
    replayed: list[Plan] = []
    refused: list[str] = []

    def accept(prof: MeasuredProfile) -> bool:  # the best profile this request can replay, not just the best one
        out, why = _replay(inp, plan, prof)
        if out is None:
            refused.append(f"measured profile {prof.id[:8]} ignored: {why}")
            return False
        replayed.append(out)
        return True

    try:
        inp.profile(plan.model_id, sig, inp.req.objective, accept)
    except (sqlite3.Error, ValueError) as exc:  # a broken profile row must not block planning
        log.warning("reading measured profiles failed", exc_info=True)
        return _warn(plan, f"measured profiles could not be read ({exc}); using the heuristic plan")
    if replayed:
        return replayed[0]
    return _warn(plan, f"{refused[0]}; using the heuristic plan") if refused else plan


@dataclass(frozen=True)
class _Split:
    names: list[str]  # llama.cpp device names, --device order
    units: list[int]  # --tensor-split
    cpu_layers: int
    tier: str
    flags: dict
    endpoints: list[str]  # --rpc order, which numbers the RPC devices


def _split(fields: dict) -> _Split | str:
    """The stored split of a profile (data from the database, checked before use), or why it is unusable."""
    try:
        if fields.get("engine", "llama_rpc") != "llama_rpc":
            return f"it was measured with {fields['engine']}"
        cpu_layers = int(fields.get("cpu_layers") or 0)
        split = _Split(names=[str(n) for n in fields["device_list"] if n != "CPU"],
                       units=[int(u) for u in fields["tensor_split"]], cpu_layers=cpu_layers,
                       tier=str(fields.get("tier") or ("gpu_ram" if cpu_layers else "gpu")),
                       flags=dict(fields.get("engine_flags") or {}),
                       endpoints=[str(e) for e in fields.get("rpc_endpoints") or []])
    except (KeyError, TypeError, ValueError) as exc:
        return f"its stored split is unreadable ({exc!r})"
    if split.tier == "moe_offload":
        return "it offloads MoE experts by an -ot pattern, which a stored split does not rebuild"
    if split.tier not in ("gpu", "gpu_ram", "gpu_disk") or len(split.names) != len(split.units):
        return "its stored split is inconsistent"
    return split


def _budgets_for(s: _Split, local: list[DeviceBudget], remote: list[DeviceBudget]) -> list[DeviceBudget] | None:
    """Today's budgets for the profile's devices: local names as they are, RPC<n> through the numbering
    deployments.assign_rpc_names gives (in --rpc order, each endpoint taking one number per device it exposes)."""
    by_name = {b.device: b for b in [*local, *remote]}
    counts = Counter(b.endpoint for b in remote)
    first: dict[str, int] = {}
    k = 0
    for ep in s.endpoints:
        first[ep], k = k, k + max(1, counts.get(ep, 0))
    out: list[DeviceBudget] = []
    for name in s.names:
        if m := _RPC_NAME.fullmatch(name):
            n = int(m.group(1))
            ep = next((e for e in s.endpoints if first[e] <= n < first[e] + max(1, counts.get(e, 0))), None)
            name = f"{ep}#{n - first[ep]}" if ep else ""
        b = by_name.get(name)
        if b is None or b in out:
            return None
        out.append(b)
    return out


def _replay_blocker(inp: Inputs, s: _Split) -> str | None:
    if s.cpu_layers and not inp.req.allow_ram_tier:
        return "it keeps layers in host RAM, which this request turns off"
    if s.tier == "gpu_disk" and not inp.req.allow_capacity_mode:
        return "it pages layers from disk (capacity mode), which this request turns off"
    if s.tier == "gpu_disk" and inp.cpu is not None and inp.cpu.pool_kind == "unified":
        return ("it pages layers from disk, and this machine's CPU and GPU share one unified pool, where the planner "
                "keeps CPU layers resident (gpu_ram) or does not run them")
    try:
        engine_flag_args(s.flags)
    except EngineAdapterError as exc:
        return f"its engine flags are invalid ({exc})"
    return None


def _replay(inp: Inputs, plan: Plan, prof: MeasuredProfile) -> tuple[Plan | None, str]:
    """The heuristic plan with the profile's split and flags, re-checked against the memory free now."""
    s = _split(prof.plan_fields)
    if isinstance(s, str):
        return None, s
    if (blocker := _replay_blocker(inp, s)) is not None:
        return None, blocker
    devices = _budgets_for(s, inp.local, inp.remote)
    if devices is None:
        return None, "its devices are not all present now"
    if inp.req.strategy == PlanStrategy.all_selected and {b.node_id for b in inp.remote} - {b.node_id for b in devices}:
        return None, "it leaves out a selected peer, and 'run together' uses every one"
    ub = s.flags.get("ubatch")
    shape = inp.shape.with_ubatch(ub) if isinstance(ub, int) else inp.shape  # type: ignore[union-attr]
    layout, ok = planner.layout_from_units(shape, devices, s.units, inp.cpu, s.cpu_layers,  # type: ignore[arg-type]
                                           paged=s.tier == "gpu_disk")
    if not ok:
        return None, "it no longer fits in the memory free now"
    kept = [r for r in plan.reasons if not r.startswith("contributed /") and "over budget" not in r]
    heuristic = _summary(plan).model_copy(update={"label": f"heuristic {plan.tier} plan"})
    out = planner.finish_plan(
        inp.req, shape, layout, True, paged=s.tier == "gpu_disk",  # type: ignore[arg-type]
        reasons=[f"measured profile {prof.id[:8]} (score {prof.score:.2f}) replaces the heuristic split", *kept],
        warnings=[], candidates=[*plan.candidates, heuristic], host_ram_free=inp.host_ram_free,
        total_budget=sum(b.budget_bytes for b in [*inp.local, *inp.remote]))
    return out.model_copy(update={"engine_flags": s.flags, "profile_id": prof.id, "heuristic": False}), ""
