"""Replan controller: decides when the ready deployment should be reloaded onto another plan, and does it through
DeploymentManager.replace at a request boundary.

A join, a state change or a departure is a reason to evaluate, not an instruction to use every device. A switch needs
inputs stable for dwell_s, a relative decode gain of at least `hysteresis`, savings over horizon_tokens above the
reload + replay cost, and cooldown_s since the last switch (its own, or a replacement the owner started: the controller
follows the deployment.replaced / replace_failed events). The numbers are the optimizer's predictions (the evaluator's
fresh score of the running layout when it gives one, else Plan.prediction), not measurements. Memory pressure is
recovery, not speed: it takes an executable "fit" plan even when that plan decodes slower. Every decision, taken or
not, is kept with its numbers.

A switch is a reload: generation in flight on the old deployment is not migrated, and requests that arrive during the
swap find no ready model. Losing a node the active deployment uses is handled by DeploymentManager.on_node_lost (it
stops the deployment); this controller then only proposes a plan on the remaining nodes.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable, Set
from contextlib import aclosing
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Literal, Protocol

from ..capacity.types import Workload
from ..schemas import Deployment, DeploymentState, Event, Plan, PlanPrediction, ReplanDecision, utcnow
from ..storage import Store
from .deployments import DeploymentError, DeploymentManager
from .events import EventBus

log = logging.getLogger("hypercompute.replan")

Objective = Literal["speed", "fit"]
TOPICS = frozenset({"node.verified", "node.state", "node.left"})
# DeploymentManager.replace events that mean the served model was reloaded, whoever asked for it (this controller or
# the owner's replace route): each restarts the cooldown. A replacement refused before the drain, or one whose drain
# timed out, stopped nothing (outcome "refused" or none) and does not count.
SWITCH_OUTCOMES = frozenset({"activated", "rolled_back", "fell_back", "failed"})
RING_SIZE = 50
BUSY_RETRY_S = 1.0  # shortest wait before an input that arrived during a running replan is evaluated
HORIZON_TOKENS = int(next(f.default for f in fields(Workload) if f.name == "horizon_tokens"))  # type: ignore[call-overload]


@dataclass(frozen=True)
class ReplanPolicy:
    dwell_s: float = 30.0  # inputs unchanged this long before a switch
    hysteresis: float = 0.15  # minimum relative decode gain
    cooldown_s: float = 120.0  # since the last switch
    horizon_tokens: int = HORIZON_TOKENS  # output tokens a switch has to pay itself back over
    min_confidence: float = 0.0  # proposals with a lower (or unknown, when > 0) evaluator confidence are deferred

    def __post_init__(self) -> None:
        if min(self.dwell_s, self.hysteresis, self.cooldown_s) < 0 or self.horizon_tokens <= 0:
            raise ValueError("replan policy needs non-negative dwell, hysteresis and cooldown and a positive horizon")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence is a fraction between 0 and 1")


@dataclass(frozen=True)
class Proposal:
    plan: Plan  # the candidate; plan.prediction carries the optimizer's numbers
    reload_ms: float  # stop + load + verify + warm the candidate
    replay_ms: float  # re-prefill of the context the reload drops
    confidence: float | None = None  # the evaluator's trust in the prediction (0..1), when it knows
    fallback: Plan | None = None  # executable alternative if the candidate fails and the previous plan cannot return
    # the evaluator's fresh score of the running layout on the same inputs as the candidate; decide() compares against
    # it instead of current.plan.prediction (which may be missing or scored on older inputs) whenever it is set
    current_prediction: PlanPrediction | None = None

    def __post_init__(self) -> None:
        # NaN compares false with everything, so an unchecked NaN cost would pass every gate in decide()
        if not all(math.isfinite(v) and v >= 0 for v in (self.reload_ms, self.replay_ms)):
            raise ValueError("reload_ms and replay_ms are durations: finite and not negative")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:  # also refuses NaN
            raise ValueError("confidence is a fraction between 0 and 1")


class Evaluator(Protocol):
    """Proposes a plan for the deployment: objective "speed" (faster decode) or "fit" (executable under pressure)."""

    def __call__(self, current: Deployment, objective: Objective) -> Awaitable[Proposal | None]: ...


async def no_proposal(current: Deployment, objective: Objective) -> Proposal | None:
    """The default evaluator until the optimizer bridge is wired: never proposes a plan."""
    return None


# --------------------------------------------------------------------------- pure decision


@dataclass(frozen=True)
class Numbers:
    current_ms: float
    proposed_ms: float
    gain: float  # (current - proposed) / current
    savings_ms: float  # horizon_tokens * (current - proposed)
    reload_ms: float
    replay_ms: float
    horizon_tokens: int

    @property
    def cost_ms(self) -> float:
        return self.reload_ms + self.replay_ms

    @property
    def break_even_tokens(self) -> float | None:
        delta = self.current_ms - self.proposed_ms
        return self.cost_ms / delta if delta > 0 else None

    def text(self) -> str:
        return (f"decode {self.current_ms:.1f} -> {self.proposed_ms:.1f} ms/token (gain {self.gain:+.1%}), saves "
                f"{self.savings_ms:.0f} ms over {self.horizon_tokens} tokens vs reload {self.reload_ms:.0f} + replay "
                f"{self.replay_ms:.0f} ms")

    def fields(self) -> dict[str, float | None]:
        return {"current_decode_ms": self.current_ms, "proposed_decode_ms": self.proposed_ms,
                "relative_gain": self.gain, "reload_ms": self.reload_ms, "replay_ms": self.replay_ms,
                "break_even_tokens": self.break_even_tokens}


def usable(prediction: PlanPrediction | None) -> bool:
    """A per-token decode time decide() can compare: present, finite and positive (NaN or inf would pass the gates)."""
    return prediction is not None and math.isfinite(prediction.decode_ms) and prediction.decode_ms > 0


def current_score(current: Deployment, proposal: Proposal) -> PlanPrediction | None:
    """The running layout's score decide() compares against: the evaluator's fresh one when the proposal carries it,
    else the prediction stored with the running plan."""
    return proposal.current_prediction if proposal.current_prediction is not None else current.plan.prediction


def numbers(current: Deployment, proposal: Proposal, horizon_tokens: int) -> Numbers | None:
    """The comparison behind a decision; None when either plan has no usable prediction."""
    cur, new = current_score(current, proposal), proposal.plan.prediction
    if cur is None or new is None or not usable(cur) or not usable(new):
        return None
    delta = cur.decode_ms - new.decode_ms
    return Numbers(current_ms=cur.decode_ms, proposed_ms=new.decode_ms, gain=delta / cur.decode_ms,
                   savings_ms=horizon_tokens * delta, reload_ms=proposal.reload_ms, replay_ms=proposal.replay_ms,
                   horizon_tokens=horizon_tokens)


def _gate(n: Numbers, policy: ReplanPolicy, now: datetime, last_switch_at: datetime | None,
          stable_since: datetime | None, confidence: float | None) -> tuple[str, str]:
    if stable_since is None:
        return "defer", f"inputs have no stable window yet; {n.text()}"
    held = (now - stable_since).total_seconds()
    if held < policy.dwell_s:
        return "defer", f"inputs stable for {held:.1f} s, need {policy.dwell_s:.1f} s; {n.text()}"
    if last_switch_at is not None and (since := (now - last_switch_at).total_seconds()) < policy.cooldown_s:
        return "defer", f"last switch {since:.1f} s ago, cooldown {policy.cooldown_s:.1f} s; {n.text()}"
    if policy.min_confidence > 0 and (confidence is None or confidence < policy.min_confidence):
        shown = "unknown" if confidence is None else f"{confidence:.2f}"
        return "defer", f"prediction confidence {shown} below {policy.min_confidence:.2f}; {n.text()}"
    if n.gain < policy.hysteresis:
        return "keep", f"gain {n.gain:.1%} below hysteresis {policy.hysteresis:.1%}; {n.text()}"
    if n.savings_ms <= n.cost_ms:
        return "keep", f"savings {n.savings_ms:.0f} ms do not cover reload + replay {n.cost_ms:.0f} ms; {n.text()}"
    return "replace", f"gain {n.gain:.1%} >= {policy.hysteresis:.1%} and savings exceed the cost; {n.text()}"


def decide(current: Deployment, proposal: Proposal | None, policy: ReplanPolicy, now: datetime,
           last_switch_at: datetime | None, stable_since: datetime | None, trigger: str = "manual") -> ReplanDecision:
    """Keep, defer or replace, from the plans' predictions. Pure: the same inputs give the same decision."""
    base = {"at": now, "trigger": trigger, "deployment_id": current.id, "revision": current.revision}
    if current.plan.layout is not None:  # PlanRequest.layout (services/layout.py): checked, never chosen
        return ReplanDecision(**base, action="keep", reason="it runs an explicit layout the owner named "
                              "(PlanRequest.layout); a speed replan never replaces it")
    if proposal is None:
        return ReplanDecision(**base, action="keep", reason="the evaluator proposed no other plan")
    plan_fields = {"proposed_plan_id": proposal.plan.id, "reload_ms": proposal.reload_ms,
                   "replay_ms": proposal.replay_ms}
    if not proposal.plan.feasible:
        return ReplanDecision(**base, **plan_fields, action="keep", reason="proposed plan is not executable: "
                              + ("; ".join(proposal.plan.reasons) or "no reason given"))
    n = numbers(current, proposal, policy.horizon_tokens)
    if n is None:
        which = [name for name, p in (("current", current_score(current, proposal)),
                                      ("proposed", proposal.plan.prediction)) if not usable(p)]
        return ReplanDecision(**base, **plan_fields, action="defer",
                              reason=f"no usable prediction for the {' and '.join(which)} plan; cannot compare")
    action, reason = _gate(n, policy, now, last_switch_at, stable_since, proposal.confidence)
    return ReplanDecision(**base, **n.fields(), proposed_plan_id=proposal.plan.id, action=action,  # type: ignore[arg-type]
                          reason=reason)


def decide_pressure(current: Deployment, proposal: Proposal | None, now: datetime, node_id: str, detail: str,
                    horizon_tokens: int = HORIZON_TOKENS) -> ReplanDecision:
    """Memory pressure: reload onto an executable plan that fits, even when it decodes slower. No dwell, hysteresis
    or cooldown; the speed numbers are recorded when both plans have predictions."""
    base = {"at": now, "trigger": "pressure", "deployment_id": current.id, "revision": current.revision}
    what = f"memory pressure on {node_id[:8]} ({detail})"
    if proposal is None or not proposal.plan.feasible:
        return ReplanDecision(**base, action="failed", reason=f"{what}; no executable plan that fits was proposed")
    n = numbers(current, proposal, horizon_tokens)
    speed = n.text() if n is not None else "no speed prediction to compare"
    extra = n.fields() if n is not None else {"reload_ms": proposal.reload_ms, "replay_ms": proposal.replay_ms}
    return ReplanDecision(**base, **extra, proposed_plan_id=proposal.plan.id, action="fallback",  # type: ignore[arg-type]
                          reason=f"{what}; reloading onto plan {proposal.plan.id[:8]} to fit; {speed}")


def is_switch(event: Event) -> bool:
    """Whether a deployment event reports a finished replacement that reloaded the served model."""
    if event.type == "deployment.replaced":
        return True
    return event.type == "deployment.replace_failed" and event.data.get("outcome") in SWITCH_OUTCOMES


def outcome_of(result: Deployment, proposal: Proposal, previous: Deployment) -> str:
    """What a replacement ended as, from the deployment replace() returned."""
    if result.state != DeploymentState.ready:
        return "stopped" if result.state == DeploymentState.stopped else "failed"
    if result.replaces == previous.id:  # only the first attempt replaces it directly; recoveries replace that attempt
        return "activated"
    return "rolled_back" if result.plan.id == previous.plan.id else "fell_back"


# --------------------------------------------------------------------------- controller


class ReplanController:
    """Follows node events, waits for stable inputs, asks the evaluator and replaces the ready deployment when
    decide() says so. One evaluation or replacement at a time; every decision is recorded (JSONL "replan" and the
    last RING_SIZE in memory)."""

    def __init__(self, deployments: DeploymentManager, events: EventBus, store: Store | None, evaluator: Evaluator,
                 policy: ReplanPolicy | None = None, *, enabled: bool = True,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._deps, self._events, self._store, self._evaluator = deployments, events, store, evaluator
        self.policy = policy or ReplanPolicy()
        self.enabled = enabled
        self._clock = clock
        self._ring: deque[ReplanDecision] = deque(maxlen=RING_SIZE)
        self._stable_since: datetime = clock()
        self._last_switch_at: datetime | None = None
        self._running = False
        self._timer: asyncio.Task[None] | None = None  # the debounce timer while it sleeps
        self._tasks: set[asyncio.Task[None]] = set()  # every timer task until it ends, sleeping or evaluating
        self._closed = False
        self._pending: str | None = None  # the latest input change's topic, evaluated once the inputs hold still
        self._lost: set[str] = set()  # nodes that left since the last evaluation

    def decisions(self) -> list[ReplanDecision]:
        return list(self._ring)

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ inputs
    async def run(self) -> None:
        """Follow node events until cancelled (the gateway starts this when the controller is enabled)."""
        if not self.enabled:
            return
        start = max((e.seq for e in self._events.since(0)), default=0)
        async with aclosing(self._events.stream(start)) as stream:
            async for event in stream:
                if event is not None:
                    self.observe(event)

    def observe(self, event: Event) -> None:
        """An input changed: restart the stability window and evaluate once it has held for dwell_s. A finished
        replacement (is_switch) restarts the cooldown instead, so an owner's replace counts like one of ours."""
        if not self.enabled:
            return
        if is_switch(event):
            self._last_switch_at = self._clock()  # when this controller saw it, on the clock decide() reads
            return
        if event.type not in TOPICS:
            return
        self._stable_since = self._clock()
        self._pending = event.type
        node_id = event.data.get("node_id")
        if event.type == "node.left" and isinstance(node_id, str):
            self._lost.add(node_id)
        self._schedule(self.policy.dwell_s)

    def _schedule(self, delay_s: float) -> None:
        if self._closed:
            return
        if self._timer is not None and not self._timer.done():
            self._timer.cancel()  # only ever cancelled while it sleeps; it clears itself before evaluating
        task = asyncio.get_running_loop().create_task(self._after(delay_s))
        self._tasks.add(task)  # the loop keeps only a weak reference; _timer drops it once evaluation starts
        task.add_done_callback(self._tasks.discard)
        self._timer = task

    async def _after(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        self._timer = None
        trigger, self._pending = self._pending or "manual", None
        lost, self._lost = frozenset(self._lost), set()
        try:
            decision = await self.evaluate(trigger, lost)
        except Exception:  # a timer task has no caller to report to: log it with context and keep following events
            log.exception("replan evaluation after %s failed", trigger)
            return
        retry = self._retry_after_s() if decision is not None and decision.action == "defer" else 0.0
        if retry > 0 and self._timer is None:
            self._pending = self._pending or trigger
            self._schedule(retry)

    def _retry_after_s(self) -> float:
        """Seconds until the stability window, the cooldown or a running replan no longer holds a switch back (0: none
        of them does, so a retry would decide the same)."""
        now = self._clock()
        dwell = self.policy.dwell_s - (now - self._stable_since).total_seconds()
        cool = (self.policy.cooldown_s - (now - self._last_switch_at).total_seconds()
                if self._last_switch_at is not None else 0.0)
        busy = max(self.policy.dwell_s, BUSY_RETRY_S) if self._running else 0.0  # dwell 0 must not drop the input
        return max(dwell, cool, busy, 0.0)

    async def aclose(self) -> None:
        """Stop following inputs: cancel the timer and any evaluation it started (a replacement it was running stops
        where it is; DeploymentManager.shutdown stops what is left), and wait for them to end."""
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)
        self._timer = None

    # ------------------------------------------------------------------ triggers
    async def trigger(self) -> ReplanDecision:
        """Manual trigger: evaluate now under the same policy."""
        decision = await self.evaluate("manual")
        if decision is None:
            return self._note("manual", None, "defer", "no ready deployment to replan")
        return decision

    async def evaluate(self, trigger: str, lost: Set[str] = frozenset()) -> ReplanDecision | None:
        """One speed evaluation of the ready deployment (None: nothing to evaluate)."""
        if not self.enabled:
            return self._disabled(trigger)
        if self._running:
            return self._note(trigger, self._ready(), "defer", "a replan is already running")
        self._running = True
        try:
            current = self._ready()
            if current is None:  # `lost` decides, not the trigger: a later node.state can follow node.left in a window
                return await self._propose_after_loss(lost) if lost else None
            proposal, failure = await self._propose(current, "speed", trigger)
            if failure is not None:
                return failure
            decision = decide(current, proposal, self.policy, self._clock(), self._last_switch_at,
                              self._stable_since, trigger)
            if decision.action != "replace" or proposal is None:
                return self._record(decision)
            return self._record(await self._switch(current, proposal, decision))
        finally:
            self._running = False

    async def on_pressure(self, node_id: str, detail: str) -> ReplanDecision | None:
        """Sustained memory pressure on a node of the ready deployment: reload onto an executable plan that fits, even
        a slower one. Skips dwell, hysteresis and cooldown; never runs beside another evaluation or replacement."""
        if not self.enabled:
            return self._disabled("pressure")
        current = self._ready()
        if self._running:
            return self._note("pressure", current, "defer", f"memory pressure on {node_id[:8]} ({detail}) while a "
                              "replan is already running")
        if current is None:
            return None
        if all(a.node_id != node_id for a in current.plan.assignments):
            return self._note("pressure", current, "keep", f"memory pressure on {node_id[:8]} ({detail}); the "
                              "ready deployment does not use that node")
        self._running = True
        try:
            proposal, failure = await self._propose(current, "fit", "pressure")
            if failure is not None:
                return failure
            decision = decide_pressure(current, proposal, self._clock(), node_id, detail, self.policy.horizon_tokens)
            if decision.action != "fallback" or proposal is None:
                return self._record(decision)
            return self._record(await self._switch(current, proposal, decision))
        finally:
            self._running = False

    # ------------------------------------------------------------------ steps
    def _ready(self) -> Deployment | None:
        dep = self._deps.active()
        return dep if dep is not None and dep.state == DeploymentState.ready and dep.purpose == "run" else None

    async def _propose(self, current: Deployment, objective: Objective, trigger: str
                       ) -> tuple[Proposal | None, ReplanDecision | None]:
        try:
            return await self._evaluator(current, objective), None
        except Exception as exc:  # the evaluator is a plug-in: its failure is a recorded decision, not a crash
            log.exception("replan evaluator failed for deployment %s", current.id)
            return None, self._note(trigger, current, "failed",
                                    f"evaluator ({objective}) failed: {str(exc) or exc.__class__.__name__}")

    async def _switch(self, current: Deployment, proposal: Proposal, decision: ReplanDecision) -> ReplanDecision:
        """Run the replacement and return the decision with its outcome."""
        try:
            result = await self._deps.replace(current.id, proposal.plan, decision.reason, fallback=proposal.fallback)
        except DeploymentError as exc:
            if exc.code != "replace_failed":  # refused before anything changed (drain timeout, not ready, ...)
                return decision.model_copy(update={"outcome": f"refused: {exc}"})
            self._last_switch_at = self._clock()
            return decision.model_copy(update={"outcome": f"failed: {exc}"})
        except Exception as exc:  # recorded with the decision and logged; the deployment records keep their own state
            log.exception("replacement of deployment %s crashed", current.id)
            self._last_switch_at = self._clock()
            return decision.model_copy(update={"outcome": f"failed: internal error: {exc}"})
        self._last_switch_at = self._clock()
        outcome = outcome_of(result, proposal, current)
        if outcome == "failed" and result.error:
            outcome = f"failed: {result.error}"
        return decision.model_copy(update={"outcome": outcome})

    async def _propose_after_loss(self, lost: Set[str]) -> ReplanDecision | None:
        """The loss of a node stopped the deployment (on_node_lost): propose a plan on the remaining nodes. Nothing is
        started; the owner decides."""
        latest = next(iter(self._deps.list()), None)
        if latest is None or latest.state != DeploymentState.failed or latest.purpose != "run":
            return None
        if all(a.node_id not in lost for a in latest.plan.assignments):
            return None
        proposal, failure = await self._propose(latest, "fit", "node.left")
        if failure is not None:
            return failure
        if proposal is None or not proposal.plan.feasible:
            return self._note("node.left", latest, "keep", "a node it used left and stopped it; no plan on the "
                              "remaining nodes was proposed")
        return self._record(ReplanDecision(
            at=self._clock(), trigger="node.left", deployment_id=latest.id, revision=latest.revision, action="defer",
            proposed_plan_id=proposal.plan.id, reload_ms=proposal.reload_ms, replay_ms=proposal.replay_ms,
            proposed_decode_ms=proposal.plan.prediction.decode_ms if proposal.plan.prediction else None,
            reason=f"a node it used left and stopped it; plan {proposal.plan.id[:8]} on the remaining nodes is "
                   "proposed, start it to use it"))

    # ------------------------------------------------------------------ records
    def _note(self, trigger: str, current: Deployment | None, action: str, reason: str) -> ReplanDecision:
        return self._record(ReplanDecision(at=self._clock(), trigger=trigger,
                                           deployment_id=current.id if current else None,
                                           revision=current.revision if current else 0,
                                           action=action, reason=reason))  # type: ignore[arg-type]

    def _disabled(self, trigger: str) -> ReplanDecision:
        """Not recorded: a disabled controller writes nothing."""
        return ReplanDecision(at=self._clock(), trigger=trigger, deployment_id=None, revision=0, action="defer",
                              reason="the replan controller is disabled (set HYPERCOMPUTE_REPLAN=1)")

    def _record(self, decision: ReplanDecision) -> ReplanDecision:
        self._ring.append(decision)
        if self._store is not None:
            self._store.append_jsonl("replan", decision.model_dump(mode="json"))
        self._events.publish("replan.decision", f"Replan ({decision.trigger}): {decision.action}: {decision.reason}",
                             deployment_id=decision.deployment_id, revision=decision.revision,
                             action=decision.action, outcome=decision.outcome)
        return decision
