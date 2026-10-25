"""The three MCP tools Droid gets, served in-process by droid_sdk.mcp (a loopback HTTP server with a per-run bearer
token). All three are read-only or plan-only against the gateway:

- pool_snapshot: GET /api/nodes + GET /api/network
- current_layout: GET /api/deployments + GET /api/benchmarks
- check_layout: POST /api/plans with an explicit layout, which only checks it and stores a plan row (never loads)

Each handler asks the guard first and writes what it saw to the run log, so the page shows the planner's verdict
next to Droid's call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from droid_sdk import DroidTool
from droid_sdk.mcp import tool
from pydantic import BaseModel, Field

from . import layouts
from .gateway import PlanOnlyGateway
from .guard import ToolGuard
from .models import Prediction, Stage, Verdict
from .pool import LOCAL_ID, PoolSnapshot, clean, snapshot
from .runlog import RunStore

GiB = 1024**3


def gib(n: int | None) -> float | None:
    return None if n is None else round(n / GiB, 2)


# ---------------------------------------------------------------- tool result shapes (what Droid reads)
class DeviceOut(BaseModel):
    name: str
    description: str
    total_gib: float | None
    free_gib: float | None


class NodeOut(BaseModel):
    node_id: str
    name: str
    role: str
    devices: list[DeviceOut]
    contribution_max_gib: float | None
    path: str | None
    rtt_ms: float | None


class PoolOut(BaseModel):
    transport: str
    nodes: list[NodeOut]
    note: str


class RowOut(BaseModel):
    node_id: str
    node: str
    device: str
    layers: int
    layer_range: str
    est_gib: float
    budget_gib: float
    fits: bool
    file_backed: bool


class MeasuredOut(BaseModel):
    decode_tok_s_median: float | None
    prefill_tok_s_median: float | None
    label: str | None
    benchmark_id: str | None


class CurrentOut(BaseModel):
    model_id: str
    deployment_id: str | None
    tier: str | None
    stages: list[Stage] | None
    rows: list[RowOut]
    hops: int | None
    measured: MeasuredOut | None
    note: str


class VerdictOut(BaseModel):
    accepted: bool
    code: str | None
    reason: str
    fix: str | None
    plan_id: str | None
    tier: str | None
    hops: int | None
    rows: list[RowOut]
    predicted_decode_ms: float | None
    predicted_source: str | None
    predicted_ssd_mib_per_token: float | None
    warnings: list[str]
    checks_left: int


@dataclass
class RunContext:
    run_id: str
    store: RunStore
    gateway: PlanOnlyGateway
    guard: ToolGuard
    snap: PoolSnapshot
    model_id: str
    ctx_size: int
    current_plan: dict[str, Any] | None
    deployment_id: str | None
    measured: dict[str, Any] | None
    verdicts: list[Verdict] = field(default_factory=list)
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)  # plan_id -> the gateway's stored plan


def _row_out(r: Any) -> RowOut:
    rng = "-" if r.first_layer is None else (f"{r.first_layer}" if r.first_layer == r.last_layer
                                             else f"{r.first_layer}-{r.last_layer}")
    return RowOut(node_id=r.node_id, node=r.node_name, device=r.device, layers=r.layers, layer_range=rng,
                  est_gib=gib(r.total_bytes_est) or 0.0, budget_gib=gib(r.budget_bytes) or 0.0, fits=r.fits,
                  file_backed=r.file_backed)


def pool_out(snap: PoolSnapshot) -> PoolOut:
    nodes = []
    for n in snap.nodes:
        if not n.is_local and not n.eligible:
            continue
        role = ('this machine, node_id "local": the last stage, holds the output head' if n.is_local
                else "peer: may hold stages before the local one")
        nodes.append(NodeOut(node_id=n.id, name=n.name, role=role, devices=[
            DeviceOut(name=d.name, description=d.description, total_gib=gib(d.total_bytes), free_gib=gib(d.free_bytes))
            for d in n.devices], contribution_max_gib=gib(n.max_bytes), path=n.path, rtt_ms=n.rtt_ms))
    return PoolOut(transport=snap.transport, nodes=nodes,
                   note="devices are named as each worker names them; check_layout is the only judge of fit")


def measured_out(bench: dict[str, Any] | None) -> MeasuredOut | None:
    if not bench:
        return None
    summaries = bench.get("summaries") or []
    short = next((s for s in summaries if s.get("prompt_label") == "short"), summaries[0] if summaries else {})
    return MeasuredOut(decode_tok_s_median=(short.get("decode_tokens_per_s") or {}).get("median"),
                       prefill_tok_s_median=(short.get("prompt_tokens_per_s") or {}).get("median"),
                       label=bench.get("placement_label") or bench.get("label"), benchmark_id=bench.get("id"))


def verdict_from(outcome: Any, stages: list[Stage], label: str, snap: PoolSnapshot,
                 checked_by: str = "droid") -> Verdict:
    """The planner's answer as a Verdict: a 4xx refusal keeps its code, reason and fix; a stored plan that is not
    feasible (weights missing, say) is a refusal too, with the plan's own reason."""
    if outcome.plan is None:
        return Verdict(accepted=False, label=label, stages=stages, code=outcome.code, reason=outcome.detail,
                       fix=outcome.fix, checked_by=checked_by)  # type: ignore[arg-type]
    plan = outcome.plan
    rows = layouts.rows(plan, snap)
    feasible = bool(plan.get("feasible"))
    reasons = [str(r) for r in plan.get("reasons") or []]
    return Verdict(
        accepted=feasible, label=label, stages=stages, code=None if feasible else "infeasible",
        reason=reasons[0] if reasons else ("accepted" if feasible else "the planner marked the plan infeasible"),
        plan_id=plan.get("id"), tier=plan.get("tier"), fit_label=plan.get("fit_label"), hops=layouts.hops(rows),
        rows=rows, prediction=layouts.prediction(plan), warnings=[str(w) for w in plan.get("warnings") or []][:4],
        checked_by=checked_by)  # type: ignore[arg-type]


def verdict_out(v: Verdict, checks_left: int) -> VerdictOut:
    p: Prediction = v.prediction
    return VerdictOut(accepted=v.accepted, code=v.code, reason=v.reason, fix=v.fix, plan_id=v.plan_id, tier=v.tier,
                      hops=v.hops, rows=[_row_out(r) for r in v.rows], predicted_decode_ms=p.decode_ms,
                      predicted_source=p.decode_source,
                      predicted_ssd_mib_per_token=None if p.ssd_bytes_per_token is None
                      else round(p.ssd_bytes_per_token / 2**20, 1),
                      warnings=v.warnings, checks_left=checks_left)


def plan_body(model_id: str, stages: list[Stage], ctx_size: int) -> dict[str, Any]:
    return {"model_id": model_id, "ctx_size": ctx_size,
            "layout": [s.model_dump(exclude_none=True) for s in stages]}


def _refusal(reason: str, checks_left: int) -> VerdictOut:
    return VerdictOut(accepted=False, code="guard_refused", reason=reason, fix=None, plan_id=None, tier=None,
                      hops=None, rows=[], predicted_decode_ms=None, predicted_source=None,
                      predicted_ssd_mib_per_token=None, warnings=[], checks_left=checks_left)


def build_tools(ctx: RunContext) -> list[DroidTool]:
    def checks_left() -> int:
        return max(0, ctx.guard.max_checks - ctx.guard.checks)

    async def pool_snapshot() -> PoolOut:
        refused = ctx.guard.admit("pool_snapshot")
        if refused:
            return PoolOut(transport=ctx.snap.transport, nodes=[], note=f"refused: {refused}")
        ctx.snap = snapshot(await ctx.gateway.nodes(), await ctx.gateway.network())
        out = pool_out(ctx.snap)
        ctx.store.step(ctx.run_id, "tool", "pool_snapshot", {"nodes": [n.name for n in out.nodes]})
        return out

    async def current_layout() -> CurrentOut:
        refused = ctx.guard.admit("current_layout")
        if refused:
            return CurrentOut(model_id=ctx.model_id, deployment_id=None, tier=None, stages=None, rows=[], hops=None,
                              measured=None, note=f"refused: {refused}")
        plan = ctx.current_plan
        rows = layouts.rows(plan, ctx.snap) if plan else []
        out = CurrentOut(
            model_id=ctx.model_id, deployment_id=ctx.deployment_id, tier=plan.get("tier") if plan else None,
            stages=layouts.stages(rows) if rows else None, rows=[_row_out(r) for r in rows],
            hops=layouts.hops(rows) if rows else None, measured=measured_out(ctx.measured),
            note=("the served placement; stages is null when it uses a tier a layout cannot name (host CPU layers)"
                  if plan else "no ready deployment: candidates are plan-only, nothing can be applied"))
        ctx.store.step(ctx.run_id, "tool", "current_layout", {"tier": out.tier, "hops": out.hops,
                                                              "measured": out.measured.model_dump() if out.measured
                                                              else None})
        return out

    async def check_layout(stages: Annotated[list[Stage], Field(min_length=1, max_length=64)],
                           label: Annotated[str, Field(max_length=80)] = "") -> VerdictOut:
        refused = ctx.guard.admit("check_layout")
        if refused:
            return _refusal(refused, checks_left())
        # droid_sdk.mcp reads hints without Annotated extras, so the bounds are checked here as well
        if not 1 <= len(stages) <= 64 or len(label) > 80:
            return _refusal("need 1 to 64 stages and a label of at most 80 characters", checks_left())
        # droid_sdk.mcp validates the arguments, then passes model_dump() values: stages arrive as dicts
        stages = [s if isinstance(s, Stage) else Stage.model_validate(s) for s in stages]
        # always the served context size: the evidence is checked at what apply will check it at
        outcome = await ctx.gateway.check_plan(plan_body(ctx.model_id, stages, ctx.ctx_size))
        verdict = verdict_from(outcome, stages, clean(label, 80) if label else f"layout {ctx.guard.checks}",
                               ctx.snap)
        ctx.verdicts.append(verdict)
        if outcome.plan is not None and verdict.plan_id:
            ctx.plans[verdict.plan_id] = outcome.plan
        ctx.store.step(ctx.run_id, "verdict", f"planner {'accepted' if verdict.accepted else 'refused'} "
                       f"{verdict.label}", {"verdict": verdict.model_dump(mode="json")})
        return verdict_out(verdict, checks_left())

    return [
        tool("pool_snapshot", "The eligible machines in the pool: node ids, worker device names and sizes, Iroh "
             "path (direct or relay) and RTT. Read-only.", pool_snapshot),
        tool("current_layout", "The served deployment's layout folded into stages, its tier, hops per token and its "
             "last measured benchmark. Read-only.", current_layout),
        tool("check_layout", "Ask the gateway planner to check one explicit layout for the served model. Plan-only: "
             "it never loads anything. Returns accepted or the planner's refusal code, reason and fix, the per-device "
             f"memory estimate, hops and any planner decode estimate. Capped per run; stages run peers first and "
             f'end on node_id "{LOCAL_ID}".', check_layout),
    ]
