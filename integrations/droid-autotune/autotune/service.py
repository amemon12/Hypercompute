"""The autotune loop: watch the pool, run Droid when it settles after a join or leave, verify Droid's answer
independently, and apply a layout only when the owner approves it.

Droid's structured answer is not trusted on its own. Every candidate it returns is matched to the verdict the
planner gave through check_layout in this run; a candidate it never checked is checked by the service itself (marked
"service" on the page), and only accepted candidates get a diff and an approve button.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import httpx

from . import layouts
from .config import Settings
from .gateway import GatewayClient, GatewayError, PlanOnlyGateway
from .guard import ToolGuard
from .models import AutotuneProposal, Stage, Verdict
from .pool import PoolChange, PoolSnapshot, snapshot
from .prompt import task_prompt
from .runlog import RunRecord, RunStore
from .runner import Runner
from .tools import RunContext, plan_body, verdict_from
from .trigger import PoolWatcher, Tick, busy_reason

log = logging.getLogger("autotune")

APPROVABLE = frozenset({"awaiting_approval", "no_change"})


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def ready_deployment(deployments: list[dict[str, Any]]) -> dict[str, Any] | None:
    ready = [d for d in deployments if d.get("state") == "ready" and d.get("purpose", "run") == "run"]
    return max(ready, key=lambda d: str(d.get("created_at", "")), default=None)


def latest_bench(benchmarks: list[dict[str, Any]], dep_id: str | None) -> dict[str, Any] | None:
    done = [b for b in benchmarks if b.get("status") == "done" and b.get("deployment_id") == dep_id]
    return max(done, key=lambda b: str(b.get("created_at", "")), default=None)


class AutotuneService:
    def __init__(self, settings: Settings, client: GatewayClient, runner_factory: Callable[[], Runner],
                 store: RunStore, *, runner_name: str = "droid",
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self.runner_name = runner_name
        self.client = client
        self.runner_factory = runner_factory
        self.store = store
        self.clock = clock
        self.watcher = PoolWatcher(settings.settle_s)
        self.snap: PoolSnapshot | None = None
        self.gateway_status: dict[str, Any] = {"ok": None, "detail": "not polled yet"}
        self._run_task: asyncio.Task[None] | None = None
        self._apply_task: asyncio.Task[None] | None = None
        self._runner: Runner | None = None
        self._last_tick: str | None = None
        self._auto_starts: deque[float] = deque()  # clock() of each automatic run start, for the rate cap

    # ---------------------------------------------------------------- watching
    @property
    def run_active(self) -> bool:
        return any(t is not None and not t.done() for t in (self._run_task, self._apply_task))

    async def _busy(self) -> str | None:
        return busy_reason(await self.client.deployments(), await self.client.tunes(), self.run_active,
                           await self.client.benchmarks())

    def _cooldown(self) -> str | None:
        """Automatic runs cost Factory credits: at most one per min_interval_s and max_runs_per_hour per hour, so a
        peer flapping slower than the settle window cannot start back-to-back sessions. The owner's Run now is
        not limited."""
        now = self.clock()
        while self._auto_starts and now - self._auto_starts[0] > 3600:
            self._auto_starts.popleft()
        if self._auto_starts and now - self._auto_starts[-1] < self.settings.min_interval_s:
            return (f"cooling down: the last automatic run started {now - self._auto_starts[-1]:.0f} s ago "
                    f"(minimum {self.settings.min_interval_s:.0f} s)")
        if len(self._auto_starts) >= self.settings.max_runs_per_hour:
            return f"cooling down: {self.settings.max_runs_per_hour} automatic runs in the last hour"
        return None

    async def poll_once(self) -> Tick:
        snap = snapshot(await self.client.nodes(), await self.client.network())
        self.snap = snap
        tick = self.watcher.observe(snap, self.clock(), await self._busy() or self._cooldown())
        self.gateway_status = {"ok": True, "detail": f"{len(snap.peers)} eligible peer(s)"}
        line = f"{tick.action}: {tick.change.describe() if tick.change else ''} {tick.detail}".strip()
        if tick.action not in ("idle", "settling") and line != self._last_tick:
            self.store.watch(tick.action, line)
        self._last_tick = line
        if tick.action == "fire":
            change = self.watcher.pending
            self.start_run({"kind": "pool_change", "reason": change.describe() if change else "pool change",
                            "joined": list(change.joined) if change else [],
                            "left": list(change.left) if change else []}, change)
            self.watcher.consume()  # only once the run exists, so a failed start keeps the change pending
            self._auto_starts.append(self.clock())
        return tick

    async def watch_forever(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception as exc:  # noqa: BLE001 - gateway down, httpx transport errors: keep watching
                detail = f"{type(exc).__name__}: {str(exc)[:280]}"
                if detail != self.gateway_status.get("detail"):  # once per new error, not every poll
                    log.warning("poll failed: %s", detail)
                self.gateway_status = {"ok": False, "detail": detail}
            await asyncio.sleep(self.settings.poll_s)

    # ---------------------------------------------------------------- runs
    def start_run(self, trigger: dict[str, Any], change: PoolChange | None) -> RunRecord:
        if self.run_active:
            raise ServiceError(409, "an autotune run or an apply is already in progress")
        for old in self.store.list():
            if old.state in APPROVABLE:
                self.store.step(old.id, "owner", "superseded by a newer pool change", {})
                self.store.update(old.id, state="superseded")
        run = self.store.create(runner=self.runner_name, trigger=trigger)
        self.store.step(run.id, "trigger", trigger.get("reason", "triggered"), trigger)
        self._run_task = asyncio.create_task(self._run(run.id, change, trigger.get("reason", "")))
        return run

    async def trigger(self, reason: str = "owner asked for a run") -> RunRecord:
        try:
            busy = await self._busy()
        except (GatewayError, httpx.HTTPError) as exc:
            raise ServiceError(503, f"the gateway is not answering: {exc}"[:300]) from exc
        if busy:
            raise ServiceError(409, busy)
        run = self.start_run({"kind": "owner", "reason": reason}, None)
        self.watcher.consume()
        return run

    async def _context(self, run_id: str) -> RunContext | None:
        snap = snapshot(await self.client.nodes(), await self.client.network())
        self.snap = snap
        dep = ready_deployment(await self.client.deployments())
        plan = dep.get("plan") if dep else None
        model_id = (plan or {}).get("model_id") or self.settings.fallback_model
        if not model_id:
            self.store.step(run_id, "context", "nothing to tune: no ready deployment and no AUTOTUNE_MODEL", {})
            self.store.update(run_id, state="failed", error="no ready deployment and no AUTOTUNE_MODEL")
            return None
        bench = latest_bench(await self.client.benchmarks(), dep.get("id") if dep else None)
        rows = layouts.rows(plan, snap) if plan else []
        self.store.update(run_id, model_id=model_id, deployment_id=dep.get("id") if dep else None,
                          current_rows=rows, current_plan=plan, measured=bench,
                          node_names={n.id: n.name for n in snap.nodes})
        self.store.step(run_id, "context", f"served: {model_id} on {len({r.node_id for r in rows})} machine(s)"
                        if plan else f"no ready deployment; plan-only for {model_id}",
                        {"peers": [p.name for p in snap.peers], "tier": (plan or {}).get("tier"),
                         "hops": layouts.hops(rows) if rows else None})

        def report(kind: str, title: str, detail: dict[str, Any]) -> None:
            self.store.step(run_id, kind, title, detail)

        def on_plan(text: str, title: str | None) -> None:
            self.store.update(run_id, plan_text=text, plan_title=title)
            self.store.step(run_id, "plan", title or "Droid's plan", {"text": text})

        guard = ToolGuard(max_checks=self.settings.max_checks, max_calls=self.settings.max_calls, report=report,
                          on_plan=on_plan)
        ctx_size = int((plan or {}).get("ctx_size") or self.settings.ctx_size)
        return RunContext(run_id=run_id, store=self.store, gateway=PlanOnlyGateway(self.client), guard=guard,
                          snap=snap, model_id=model_id, ctx_size=ctx_size, current_plan=plan,
                          deployment_id=dep.get("id") if dep else None, measured=bench)

    async def _run(self, run_id: str, change: PoolChange | None, reason: str) -> None:
        try:
            ctx = await self._context(run_id)
            if ctx is None:
                return
            runner = self.runner_factory()
            self._runner = runner
            outcome = await runner.run(ctx, task_prompt(change, reason, ctx.model_id, self.settings.max_checks))
            self.store.update(run_id, usage=outcome.usage)
            if outcome.proposal is None:
                self.store.step(run_id, "proposal", "no proposal",
                                {"error": outcome.error, "text": outcome.text[:2000]})
                self.store.update(run_id, state="failed", error=outcome.error or "no proposal")
                return
            await self._finish(run_id, ctx, outcome.proposal)
        except asyncio.CancelledError:
            self.store.update(run_id, state="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - a failed run is recorded, the watcher keeps going
            log.exception("run %s failed", run_id)
            self.store.step(run_id, "proposal", "run failed", {"error": f"{type(exc).__name__}: {exc}"[:600]})
            self.store.update(run_id, state="failed", error=f"{type(exc).__name__}: {exc}"[:600])
        finally:
            self._runner = None

    async def _verify(self, ctx: RunContext, stages: list[Stage], label: str, plan_id: str | None) -> Verdict:
        """The planner's verdict for a candidate: the one from Droid's own check_layout call when the stages match,
        else a fresh service-side check (not counted against Droid's cap)."""
        mine = [v for v in ctx.verdicts if v.stages == stages]
        match = next((v for v in mine if plan_id and v.plan_id == plan_id), None) or (mine[-1] if mine else None)
        if match is not None:
            return match
        outcome = await ctx.gateway.check_plan(plan_body(ctx.model_id, stages, ctx.ctx_size))
        verdict = verdict_from(outcome, stages, label, ctx.snap, checked_by="service")
        if outcome.plan is not None and verdict.plan_id:
            ctx.plans[verdict.plan_id] = outcome.plan
        self.store.step(ctx.run_id, "verdict", f"service re-checked {label}: "
                        f"{'accepted' if verdict.accepted else 'refused'}",
                        {"verdict": verdict.model_dump(mode="json")})
        return verdict

    async def _finish(self, run_id: str, ctx: RunContext, proposal: AutotuneProposal) -> None:
        verdicts = list(ctx.verdicts)
        index_of: dict[int, int] = {}
        diffs = {}
        for i, cand in enumerate(proposal.candidates):
            verdict = await self._verify(ctx, list(cand.stages), cand.label, cand.plan_id)
            if verdict not in verdicts:
                verdicts.append(verdict)
            index_of[i] = verdicts.index(verdict)
            plan = ctx.plans.get(verdict.plan_id or "")
            if verdict.accepted and plan is not None:
                diffs[i] = layouts.layout_diff(ctx.current_plan, plan, ctx.snap)
        changed = [i for i, d in diffs.items() if not d.unchanged]
        state = "awaiting_approval" if proposal.recommendation == "apply" and changed else "no_change"
        self.store.update(run_id, proposal=proposal, verdicts=verdicts, diffs=diffs, candidate_verdicts=index_of,
                          state=state)
        self.store.step(run_id, "proposal", f"Droid recommends {proposal.recommendation}: {proposal.summary[:160]}",
                        {"candidates": len(proposal.candidates), "accepted": len(diffs),
                         "recommended_index": proposal.recommended_index})
        if state == "awaiting_approval":
            self.store.step(run_id, "owner", "waiting for the owner to approve or reject", {})

    # ---------------------------------------------------------------- owner actions
    def _approvable(self, run_id: str, index: int) -> tuple[RunRecord, Verdict]:
        run = self.store.get(run_id)
        if run is None:
            raise ServiceError(404, "no such run")
        if run.state not in APPROVABLE:
            raise ServiceError(409, f"run is {run.state}")
        if run.proposal is None or not 0 <= index < len(run.proposal.candidates) or index not in run.diffs:
            raise ServiceError(422, "that candidate was not accepted by the planner")
        return run, run.verdicts[run.candidate_verdicts[index]]

    async def approve(self, run_id: str, index: int) -> RunRecord:
        run, verdict = self._approvable(run_id, index)
        if self.run_active:
            raise ServiceError(409, "an autotune run or an apply is already in progress")
        label = run.proposal.candidates[index].label if run.proposal else verdict.label
        self.store.step(run_id, "owner", f"owner approved {label}", {"index": index})
        if run.deployment_id is None:
            return self.store.update(run_id, state="approved",
                                     applied={"index": index, "note": "plan-only: no deployment was serving"})
        updated = self.store.update(run_id, state="applying")
        self._apply_task = asyncio.create_task(self._apply(run_id, index, verdict, label))
        return updated

    async def _after_timeout(self, dep_id: str) -> dict[str, Any]:
        """The replace call outlived this client: ask the gateway what serves now instead of guessing."""
        deps = await self.client.deployments()
        newer = [d for d in deps if d.get("replaces") == dep_id]
        if newer:
            return max(newer, key=lambda d: str(d.get("created_at", "")))
        return next((d for d in deps if d.get("id") == dep_id), {"id": dep_id, "state": "unknown"})

    async def _apply(self, run_id: str, index: int, approved: Verdict, label: str) -> None:
        run = self.store.get(run_id)
        assert run is not None and run.model_id is not None and run.deployment_id is not None
        stages = approved.stages
        try:
            ctx_size = int((run.current_plan or {}).get("ctx_size") or self.settings.ctx_size)
            outcome = await self.client.check_plan(plan_body(run.model_id, stages, ctx_size))
            fresh = verdict_from(outcome, stages, label, self.snap or PoolSnapshot(()), checked_by="service")
            self.store.step(run_id, "apply", f"re-checked {label} before apply: "
                            f"{'accepted' if fresh.accepted else 'refused'}", {"plan_id": fresh.plan_id,
                                                                               "reason": fresh.reason})
            if not fresh.accepted or not fresh.plan_id:
                self.store.update(run_id, state="apply_failed", error=f"the planner refused it now: {fresh.reason}")
                return
            drift = layouts.placement_drift(approved, fresh)
            if drift:
                self.store.step(run_id, "apply", "not applied: the plan changed since the approval", {"drift": drift})
                self.store.update(run_id, state="apply_failed", error="the planner now places this layout "
                                  f"differently ({'; '.join(drift)}); run again and approve the new plan")
                return
            self.store.step(run_id, "apply", f"replacing deployment {run.deployment_id[:8]} (the gateway rolls back "
                            "on failure)", {"plan_id": fresh.plan_id})
            try:
                dep = await self.client.replace(run.deployment_id, fresh.plan_id)
            except httpx.TimeoutException:
                self.store.step(run_id, "apply", "the replace call timed out; reading what the gateway serves", {})
                dep = await self._after_timeout(run.deployment_id)
            served = (dep.get("plan") or {}).get("id")
            ok = dep.get("state") == "ready" and served == fresh.plan_id
            result = {"index": index, "deployment_id": dep.get("id"), "state": dep.get("state"),
                      "served_plan_id": served, "error": dep.get("error")}
            self.store.step(run_id, "apply", "applied" if ok else f"not applied: {dep.get('state')}", result)
            self.store.update(run_id, state="applied" if ok else "apply_failed", applied=result,
                              error=None if ok else str(dep.get("error") or "the gateway kept the old layout"))
        except GatewayError as exc:
            self.store.step(run_id, "apply", "apply failed", {"code": exc.code, "detail": exc.detail, "fix": exc.fix})
            self.store.update(run_id, state="apply_failed", error=f"{exc.code}: {exc.detail}")
        except Exception as exc:  # noqa: BLE001 - the record must say what happened
            log.exception("apply %s failed", run_id)
            self.store.update(run_id, state="apply_failed", error=f"{type(exc).__name__}: {exc}"[:600])

    def reject(self, run_id: str) -> RunRecord:
        run = self.store.get(run_id)
        if run is None:
            raise ServiceError(404, "no such run")
        if run.state not in APPROVABLE:
            raise ServiceError(409, f"run is {run.state}")
        self.store.step(run_id, "owner", "owner rejected the proposal; nothing changed", {})
        return self.store.update(run_id, state="rejected")

    async def cancel(self, run_id: str) -> RunRecord:
        run = self.store.get(run_id)
        if run is None:
            raise ServiceError(404, "no such run")
        if run.state != "running" or self._run_task is None or self._run_task.done():
            raise ServiceError(409, f"run is {run.state}")
        self.store.step(run_id, "owner", "owner cancelled the run", {})
        if self._runner is not None:
            with contextlib.suppress(Exception):
                await self._runner.interrupt()
        self._run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._run_task
        current = self.store.get(run_id)
        assert current is not None
        # the run may have finished while interrupt() was awaited: a finished state is kept
        return self.store.update(run_id, state="cancelled") if current.state == "running" else current

    async def shutdown(self) -> None:
        for task in (self._run_task, self._apply_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
