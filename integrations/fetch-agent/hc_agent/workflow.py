"""The multi-step tool execution behind one request: look at the pool, pick a model that fits (plan-only checks),
deploy it, run the prompt, and read back which machine served which layers and how fast it went.

Two phases so a human can approve in between: propose() only reads and plans (nothing loads), execute() deploys
and chats. Failures come back as WorkflowError with a code, a plain message and what to do next; recoverable ones
(a peer that dropped, a plan that aged out, a load that failed) are retried once on a fresh plan before giving up.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import Settings
from .gateway import Gateway, GatewayError
from .intent import Intent
from .planner import (
    TERMINAL,
    ModelSummary,
    NoMatchingModel,
    PlanView,
    PoolSnapshot,
    choose,
    missing_peers,
    plan_ok,
    rank_candidates,
    read_plan,
    ready_deployment,
    snapshot,
    with_observed,
)
from .store import read_json, write_json

log = logging.getLogger(__name__)
Progress = Callable[[str], Awaitable[None]]
MAX_PLAN_CALLS = 12
PLAN_BUDGET_S = 150.0  # stop trying smaller models after this long
PEER_ERROR_HINTS = ("peer", "rpc", "node", "worker", "unreachable", "connection", "stale", "left", "lost")


class WorkflowError(RuntimeError):
    def __init__(self, code: str, message: str, fix: str | None = None, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.fix, self.detail = code, message, fix, detail


@dataclass(frozen=True)
class Proposal:
    pool: PoolSnapshot
    model: ModelSummary
    view: PlanView
    strategy: str
    reuse: bool  # chat on a deployment that is already ready; nothing loads
    deployment_id: str | None = None
    notes: tuple[str, ...] = ()
    planned_at: float = field(default_factory=time.monotonic)
    alternatives: tuple[str, ...] = ()  # bigger models skipped, with the reason
    pick: tuple[str, str] | None = None  # (preference, scope) of an unhinted request, for warm reuse later


@dataclass(frozen=True)
class RunOutcome:
    answer: str
    reasoning: str | None
    finish_reason: str | None
    model: ModelSummary
    view: PlanView
    deployment_id: str
    reused: bool
    decode_tok_s: float | None
    prefill_tok_s: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    total_s: float
    load_s: float | None
    participation_verified: bool
    participation_detail: str | None
    notes: tuple[str, ...] = ()
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def _gateway_problem(exc: GatewayError) -> WorkflowError:
    if exc.unreachable:
        return WorkflowError("gateway_down", "The pool's host is not answering.",
                             exc.fix or "start it with `hypercompute host` on the host machine", detail=str(exc))
    if exc.status == 403:
        return WorkflowError("gateway_auth", "The host refused the agent's owner request.",
                             "run the agent on the host machine itself (owner calls are loopback-only)",
                             detail=str(exc))
    return WorkflowError(exc.code or "gateway_error", f"The host refused: {exc.detail}", exc.fix, detail=str(exc))


def decode_rate(timings: Mapping[str, Any]) -> float | None:
    """Decode tok/s as the repo's benchmarks define it: (predicted_n - 1) / predicted_ms."""
    n, ms = timings.get("predicted_n"), timings.get("predicted_ms")
    if isinstance(n, (int, float)) and isinstance(ms, (int, float)) and n > 1 and ms > 0:
        return (n - 1) / ms * 1000.0
    rate = timings.get("predicted_per_second")
    return float(rate) if isinstance(rate, (int, float)) and rate > 0 else None


def prefill_rate(timings: Mapping[str, Any]) -> float | None:
    rate = timings.get("prompt_per_second")
    return float(rate) if isinstance(rate, (int, float)) and rate > 0 else None


def _peer_failure(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(h in lowered for h in PEER_ERROR_HINTS)


class OwnedDeployments:
    """Deployments this agent started (the only ones it ever stops), persisted so a restart can still clean up."""

    def __init__(self, path: Path) -> None:
        self.path = path
        now = time.monotonic()
        self._last_used: dict[str, float] = {str(d): now for d in read_json(path, {}).get("owned", [])}

    def _save(self) -> None:
        try:
            write_json(self.path, {"owned": sorted(self._last_used)})
        except OSError:  # in-memory tracking still works; only a restart would forget
            log.exception("could not save %s", self.path)

    def add(self, dep_id: str) -> None:
        self._last_used = {**self._last_used, dep_id: time.monotonic()}
        self._save()

    def touch(self, dep_id: str) -> None:
        if dep_id in self._last_used:
            self._last_used = {**self._last_used, dep_id: time.monotonic()}

    def discard(self, dep_id: str) -> None:
        if dep_id in self._last_used:
            self._last_used = {k: v for k, v in self._last_used.items() if k != dep_id}
            self._save()

    def __contains__(self, dep_id: object) -> bool:
        return dep_id in self._last_used

    def idle(self, older_than_s: float) -> list[str]:
        now = time.monotonic()
        return [d for d, t in self._last_used.items() if now - t >= older_than_s]

    def adopt_replacements(self, deployments: Iterable[Mapping[str, Any]]) -> None:
        """The gateway's replan controller swaps a deployment for a new one that names it in `replaces`; the new
        one is still this agent's."""
        for dep in deployments:
            if dep.get("replaces") in self and dep.get("id") not in self and dep.get("state") not in TERMINAL:
                self.add(str(dep["id"]))


class Workflow:
    def __init__(self, gateway: Gateway, settings: Settings, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, poll_s: float = 2.0) -> None:
        self.gw, self.settings = gateway, settings
        self._sleep, self._clock, self._poll_s = sleep, clock, poll_s
        self.owned = OwnedDeployments(Path(settings.data_dir) / "owned_deployments.json")
        self.run_lock = asyncio.Lock()
        # deployment id -> (preference, scope, online peer ids) of the unhinted request that picked it
        self._picks: dict[str, tuple[str, str, frozenset[str]]] = {}

    # ------------------------------------------------------------------ reads
    async def pool(self) -> PoolSnapshot:
        try:
            nodes, network, models, deps = await asyncio.gather(
                self.gw.nodes(), self.gw.network(), self.gw.models(), self.gw.deployments())
        except GatewayError as exc:
            raise _gateway_problem(exc) from exc
        self.owned.adopt_replacements(deps)
        return snapshot(nodes, network, models, deps)

    def _strategy(self, intent: Intent) -> str:
        if intent.scope == "local":
            return "local_only"
        if intent.preference == "fastest" and self.settings.strategy == "all_selected":
            return "auto"
        return self.settings.strategy

    async def _plan(self, model_id: str, strategy: str, pool: PoolSnapshot) -> Mapping[str, Any]:
        node_ids = None if strategy == "local_only" else [n.id for n in pool.online_peers]
        try:
            return await self.gw.plan(model_id, strategy=strategy, node_ids=node_ids)
        except GatewayError as exc:
            if exc.unreachable or exc.status == 403:
                raise _gateway_problem(exc) from exc
            return {"id": None, "model_id": model_id, "feasible": False, "fit_label": "does_not_fit",
                    "reasons": [exc.detail], "warnings": []}

    # ------------------------------------------------------------------ propose
    async def propose(self, intent: Intent, *, may_unload: bool = True) -> Proposal:
        """may_unload=False (a request that still has to be paid) never changes the pool: my own idle model stays
        loaded while planning, and _deploy unloads it only once the run is approved and paid."""
        pool = await self.pool()
        try:
            candidates = rank_candidates(pool.models, intent)
        except NoMatchingModel as exc:
            have = ", ".join(exc.available) or "none"
            raise WorkflowError("no_match", f"No downloaded model matches “{exc.hint}”. The pool has: {have}.",
                                "name one of those, or say “the biggest model”") from exc
        if not candidates:
            raise WorkflowError("no_models", "The host has no downloaded, verified models yet.",
                                "download one from the dashboard, then ask again")
        strategy = self._strategy(intent)
        notes = []
        held: str | None = None  # my idle model left loaded because this plan is not paid yet

        active = pool.active()
        ready = ready_deployment(pool)
        if active is not None and (active.get("state") != "ready" or active.get("purpose", "run") != "run"):
            ours = active.get("id") in self.owned
            what = "tuning" if active.get("purpose") == "tune" else str(active.get("state"))
            raise WorkflowError("busy", f"The pool is busy: {active.get('plan', {}).get('model_id', 'a model')} is "
                                        f"{what}{' (started by me)' if ours else ''}.", "ask again in a few minutes")
        if ready is not None:
            ready_id = str(ready["plan"]["model_id"])
            ready_model = next((m for m in pool.models if m.id == ready_id), None)
            ours = ready.get("id") in self.owned
            best = candidates[0]
            if ready_model is not None and (ready_model.id == best.id or (not ours and self._may_reuse(intent, ready_model,
                                                                                                    candidates))):
                view = with_observed(read_plan(ready["plan"], pool), ready)
                skipped = () if ready_model.id == best.id else (
                    f"{best.name} would need the pool free; {ready_model.name} is loaded and serving now",)
                return Proposal(pool=pool, model=ready_model, view=view, strategy=str(ready["plan"].get("strategy")),
                                reuse=True, deployment_id=str(ready["id"]), notes=self._pool_notes(pool, strategy),
                                alternatives=skipped)
            if not ours:
                raise WorkflowError("busy", f"The pool is serving {ready_id}, which someone else loaded, and you asked "
                                            f"for {best.name}. I never unload someone else's model.",
                                    f"say “use {ready_model.name if ready_model else ready_id}” to run on what "
                                    f"is loaded, or ask again later")
            if ready_model is not None and self._still_best(str(ready["id"]), intent, pool):
                return Proposal(pool=pool, model=ready_model, view=with_observed(read_plan(ready["plan"], pool), ready),
                                strategy=str(ready["plan"].get("strategy")), reuse=True,
                                deployment_id=str(ready["id"]), notes=self._pool_notes(pool, strategy))
            if may_unload:  # my own idle model holds memory the planner would count as used, so it goes first
                if self.run_lock.locked():
                    raise WorkflowError("busy", "The pool is busy with another request.", "ask again in a minute")
                async with self.run_lock:  # uncontended (checked above): the unload can never cut off a run
                    unloaded = await self._stop_quietly(str(ready["id"]))
                if not unloaded:
                    raise WorkflowError("busy", f"{ready_id} is still loaded and I could not unload it yet.",
                                        "ask again in a minute")
                notes.append(f"I unloaded {ready_id} (which I loaded earlier) to make room.")
                pool = await self.pool()
            else:
                held = ready_id
        notes = [*notes, *self._pool_notes(pool, strategy)]

        tried: list[tuple[ModelSummary, Mapping[str, Any] | None]] = []
        started = self._clock()
        for model in candidates[:MAX_PLAN_CALLS]:
            if tried and self._clock() - started > PLAN_BUDGET_S:
                break
            plan = await self._plan(model.id, strategy, pool)
            tried.append((model, plan))
            if plan_ok(plan) and (intent.preference != "fastest" or
                                  (not plan.get("capacity_mode") and plan.get("tier") in ("gpu", "gpu_ram"))):
                break
        picked = choose(intent.preference, tried)
        if picked is None and held is not None:
            raise WorkflowError("busy", f"Nothing fits while {held}, which I loaded earlier, is still loaded. I unload "
                                        f"it only for a run that is approved and paid.",
                                f"say “use {held}” to run on it, or ask again once it unloads after "
                                f"{self.settings.keep_warm_s / 60:.0f} idle minutes")
        if picked is None:
            why = "; ".join(f"{m.name}: {(p or {}).get('reasons', ['no plan'])[-1] if p else 'no plan'}"
                            for m, p in tried)
            gib = pool.pooled_bytes / 1024**3
            raise WorkflowError("nothing_fits", f"Nothing fits on the machines online now ({gib:.0f} GiB contributed). "
                                                f"Tried {why}.",
                                "bring another friend's laptop online (`hypercompute join`) or pick a smaller model")
        model, plan = picked
        skipped = tuple(f"{m.name} does not fit: {((p or {}).get('reasons') or ['no plan'])[-1]}"
                        for m, p in tried if m.id != model.id and m.params_b > model.params_b)
        hinted = bool(intent.model_hint or intent.size_hint_b) or held is not None
        return Proposal(pool=pool, model=model, view=read_plan(plan, pool), strategy=strategy, reuse=False,
                        notes=tuple(notes), alternatives=skipped,
                        pick=None if hinted else (intent.preference, intent.scope))

    @staticmethod
    def _pool_notes(pool: PoolSnapshot, strategy: str) -> tuple[str, ...]:
        notes = []
        if strategy != "local_only" and not pool.online_peers:
            notes.append("No friends' machines are online right now, so this runs on the host alone.")
        if pool.offline_peers:
            notes.append("Offline: " + ", ".join(p.name for p in pool.offline_peers) + ".")
        return tuple(notes)

    def _still_best(self, dep_id: str, intent: Intent, pool: PoolSnapshot) -> bool:
        """My loaded model was the best fit when I picked it; with the same kind of request and no friend joined
        since, re-planning would pick it again, so it is reused instead of unloaded and loaded back."""
        pick = self._picks.get(dep_id)
        if pick is None or intent.model_hint or intent.size_hint_b:
            return False
        preference, scope, peers = pick
        return (preference, scope) == (intent.preference, intent.scope) and \
            {n.id for n in pool.online_peers} <= peers

    def _may_reuse(self, intent: Intent, ready_model: ModelSummary, candidates: list[ModelSummary]) -> bool:
        """A ready deployment someone else started is used as is when the user did not insist on another model."""
        if not self.settings.reuse_foreign_ready:
            return False
        if intent.model_hint or intent.size_hint_b:
            return ready_model.id in {m.id for m in candidates}
        return True

    # ------------------------------------------------------------------ execute
    async def _wait_ready(self, dep: Mapping[str, Any], model: ModelSummary, progress: Progress) -> Mapping[str, Any]:
        start = self._clock()
        last_state, last_said = None, start
        dep_id = str(dep["id"])
        while True:
            try:
                dep = await self.gw.deployment(dep_id)
            except GatewayError as exc:
                raise _gateway_problem(exc) from exc
            state = dep.get("state")
            if state == "ready" or state in TERMINAL:
                return dep
            now = self._clock()
            if now - start > self.settings.load_timeout_s:
                await self._stop_quietly(dep_id)
                raise WorkflowError("load_timeout", f"{model.name} was still {state} after "
                                                    f"{self.settings.load_timeout_s // 60} minutes, so I stopped it.",
                                    "try a smaller model, or ask the owner to pre-load this one from the dashboard")
            if state != last_state or now - last_said >= self.settings.progress_every_s:
                detail = f" ({dep['progress']})" if dep.get("progress") else ""
                await progress(f"Loading {model.name}: {state}{detail}, {now - start:.0f} s so far.")
                last_state, last_said = state, now
            await self._sleep(self._poll_s)

    async def _stop_quietly(self, dep_id: str) -> bool:
        """Stop one of this agent's deployments. Ownership is dropped only once the host confirms (or no longer knows
        it), so a failed stop is retried by the idle reaper instead of being forgotten."""
        try:
            await self.gw.stop(dep_id)
        except GatewayError as exc:
            if exc.status != 404:
                log.error("could not stop deployment %s: %s", dep_id[:8], exc)
                return False
        self.owned.discard(dep_id)
        return True

    async def _fresh_plan(self, proposal: Proposal, reason: str, progress: Progress) -> Proposal:
        pool = await self.pool()
        await progress(reason)
        plan = await self._plan(proposal.model.id, proposal.strategy, pool)
        if not plan_ok(plan):
            log.warning("replan of %s failed: %s", proposal.model.id, plan.get("reasons"))
            raise WorkflowError("nothing_fits", f"{proposal.model.name} no longer fits on the machines online: "
                                                f"{(plan.get('reasons') or ['no plan'])[-1]}",
                                "bring the missing laptop back online, or ask for a smaller model")
        return replace(proposal, pool=pool, view=read_plan(plan, pool), planned_at=self._clock())

    async def _create(self, plan_id: str) -> Mapping[str, Any]:
        """POST the deployment; a timed-out request is repeated with the same Idempotency-Key, so it cannot create
        a second deployment."""
        key = uuid.uuid4().hex
        for attempt in (1, 2):
            try:
                return await self.gw.deploy(plan_id, key)
            except GatewayError as exc:
                if exc.code == "timeout" and attempt == 1:
                    continue
                raise
        raise AssertionError("unreachable")

    async def _deploy(self, proposal: Proposal, progress: Progress) -> tuple[Proposal, Mapping[str, Any], float]:
        notes = list(proposal.notes)
        if self._clock() - proposal.planned_at > self.settings.plan_ttl_s:
            proposal = await self._fresh_plan(proposal, "The plan aged out while waiting; planning again.", progress)
        current = await self.pool()
        gone = missing_peers(proposal.view, current)
        if gone:
            proposal = await self._fresh_plan(proposal, f"{', '.join(gone)} went offline since planning; replanning "
                                                        f"without {'it' if len(gone) == 1 else 'them'}.", progress)
            notes.append(f"Replanned without {', '.join(gone)}.")
        for dep in current.deployments:
            if dep.get("state") == "ready" and dep.get("id") in self.owned:
                await progress(f"Unloading {dep['plan']['model_id']} to make room.")
                if not await self._stop_quietly(str(dep["id"])):
                    raise WorkflowError("busy", f"I could not unload {dep['plan']['model_id']} to make room.",
                                        "ask again in a minute")
        for attempt in (1, 2):
            started = self._clock()
            try:
                dep = await self._create(proposal.view.plan_id)
            except GatewayError as exc:
                if exc.code == "deployment_active":
                    raise WorkflowError("busy", "Someone else started a model on the pool just now.",
                                        "ask again in a few minutes") from exc
                if exc.unreachable or exc.status == 403 or attempt == 2:
                    raise _gateway_problem(exc) from exc
                log.warning("deploy of plan %s refused: %s", proposal.view.plan_id[:8], exc)
                proposal = await self._fresh_plan(proposal, "The host refused that plan; planning again.", progress)
                continue
            self.owned.add(str(dep["id"]))
            machines = ", ".join(proposal.view.machines)
            await progress(f"Deploying {proposal.model.name} across {machines}. Big models take minutes to load.")
            dep = await self._wait_ready(dep, proposal.model, progress)
            if dep.get("state") == "ready":
                return replace(proposal, notes=tuple(notes)), dep, self._clock() - started
            self.owned.discard(str(dep["id"]))
            if dep.get("state") == "stopped":
                raise WorkflowError("stopped_by_owner", f"The load of {proposal.model.name} was stopped on the host.",
                                    "ask again when the owner is ready")
            error = dep.get("error") or "the engine did not come up"
            log.warning("load of %s failed (attempt %d): %s", proposal.model.id, attempt, error)
            if attempt == 2:
                raise WorkflowError("load_failed", f"Loading {proposal.model.name} failed twice. Last error: {error}",
                                    "check the host log, or ask for a smaller model")
            reason = "a friend's machine dropped during the load" if _peer_failure(error) else "the load failed"
            proposal = await self._fresh_plan(proposal, f"Recovering: {reason}. Planning again and retrying once.",
                                              progress)
            notes.append(f"Recovered from a failed load ({reason}).")
        raise AssertionError("unreachable")

    def _messages(self, prompt: str) -> list[dict]:
        out = [{"role": "system", "content": self.settings.system_prompt}] if self.settings.system_prompt else []
        return [*out, {"role": "user", "content": prompt}]

    async def _chat(self, prompt: str, max_tokens: int) -> Mapping[str, Any]:
        last: GatewayError | None = None
        for attempt in (1, 2):
            try:
                return await self.gw.chat(self._messages(prompt), max_tokens=max_tokens,
                                          timeout=float(self.settings.chat_timeout_s), extra=self.settings.chat_extra)
            except GatewayError as exc:
                last = exc
                transient = exc.status in (502, 504)  # an engine hiccup; a client timeout is never retried
                if not transient or attempt == 2:
                    break
                await self._sleep(2.0)
        assert last is not None
        if last.code == "timeout":
            raise WorkflowError("chat_timeout", f"The model did not finish within {self.settings.chat_timeout_s} s.",
                                "ask for fewer tokens, or a smaller model") from last
        if last.status == 503:
            raise WorkflowError("not_ready", "The model stopped serving before it answered.",
                                "ask again: I will reload it") from last
        raise _gateway_problem(last) from last

    async def execute(self, proposal: Proposal, prompt: str, max_tokens: int, progress: Progress) -> RunOutcome:
        load_s = None
        dep: Mapping[str, Any] | None = None
        if proposal.reuse and proposal.deployment_id:
            try:
                dep = await self.gw.deployment(proposal.deployment_id)
            except GatewayError as exc:
                if exc.status != 404:
                    raise _gateway_problem(exc) from exc
            if dep is not None and dep.get("state") == "ready":
                await progress(f"{proposal.model.name} is already loaded across "
                               f"{', '.join(proposal.view.machines)}. Running your prompt.")
            else:  # it was unloaded since the plan was shown: load it again on a fresh plan
                await progress(f"{proposal.model.name} was unloaded since I checked; loading it again.")
                proposal = replace(proposal, reuse=False, deployment_id=None, planned_at=float("-inf"))
                dep = None
        if dep is None:
            proposal, dep, load_s = await self._deploy(proposal, progress)
            if proposal.pick is not None:
                self._picks[str(dep["id"])] = (*proposal.pick, frozenset(n.id for n in proposal.pool.online_peers))
            await progress(f"{proposal.model.name} is up after {load_s:.0f} s. Running your prompt.")
        dep_id = str(dep["id"])
        self.owned.touch(dep_id)
        chat_started = self._clock()
        body = await self._chat(prompt, max_tokens)
        self.owned.touch(dep_id)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        timings = body.get("timings") or {}
        usage = body.get("usage") or {}
        try:
            pool = await self.pool()
            view = with_observed(read_plan(dep.get("plan") or {}, pool), dep) if dep.get("plan") else proposal.view
        except Exception:  # the answer is in hand; nothing in this bookkeeping read may lose it
            log.exception("could not read the placement after the chat")
            view = proposal.view
        return RunOutcome(
            answer=str(message.get("content") or "").strip(), reasoning=message.get("reasoning_content") or None,
            finish_reason=choice.get("finish_reason"), model=proposal.model, view=view, deployment_id=dep_id,
            reused=proposal.reuse, decode_tok_s=decode_rate(timings), prefill_tok_s=prefill_rate(timings),
            prompt_tokens=timings.get("prompt_n") or usage.get("prompt_tokens"),
            output_tokens=timings.get("predicted_n") or usage.get("completion_tokens"),
            total_s=self._clock() - chat_started, load_s=load_s,
            participation_verified=bool(dep.get("participation_verified")),
            participation_detail=dep.get("participation_detail"), notes=proposal.notes)

    # ------------------------------------------------------------------ housekeeping
    async def stop_owned(self) -> tuple[list[str], list[str]]:
        """Stop every deployment this agent started. Returns (stopped, failed) model ids. The caller holds run_lock."""
        stopped, failed = [], []
        try:
            deps = await self.gw.deployments()
        except GatewayError as exc:
            raise _gateway_problem(exc) from exc
        self.owned.adopt_replacements(deps)
        for dep in deps:
            dep_id, model_id = str(dep.get("id")), str((dep.get("plan") or {}).get("model_id"))
            if dep_id in self.owned and dep.get("state") not in TERMINAL:
                (stopped if await self._stop_quietly(dep_id) else failed).append(model_id)
            elif dep_id in self.owned:
                self.owned.discard(dep_id)
        return stopped, failed

    async def reap_idle(self) -> list[str]:
        """Unload models this agent loaded once nobody has used them for keep_warm_s (frees friends' machines).
        Holds run_lock while it works, so it can never unload a model in the middle of a run."""
        if self.settings.keep_warm_s <= 0 or self.run_lock.locked():
            return []
        stopped = []
        async with self.run_lock:  # uncontended here (checked above), so this does not wait
            for dep_id in self.owned.idle(self.settings.keep_warm_s):
                try:
                    dep = await self.gw.deployment(dep_id)
                except GatewayError as exc:
                    if exc.status == 404:
                        self.owned.discard(dep_id)
                    continue
                if dep.get("state") in TERMINAL:
                    self.owned.discard(dep_id)
                    continue
                if dep_id in self.owned.idle(self.settings.keep_warm_s) and await self._stop_quietly(dep_id):
                    stopped.append(str((dep.get("plan") or {}).get("model_id")))
        return stopped
