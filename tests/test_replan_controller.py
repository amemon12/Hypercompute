"""Replan policy (pure decide) and the controller: triggers, pressure path, one-at-a-time, recorded decisions."""

import asyncio
import json
from datetime import timedelta

import pytest
from hypercompute.schemas import (
    CreateDeploymentRequest,
    Deployment,
    DeploymentState,
    Event,
    LayoutStage,
    PlanPrediction,
    utcnow,
)
from hypercompute.services import replan
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.replan import (
    HORIZON_TOKENS,
    Proposal,
    ReplanController,
    ReplanPolicy,
    decide,
    decide_pressure,
    outcome_of,
)
from hypercompute.services.streaming import InferenceService
from test_deployments import FakeAdapter, FakeModels, _plan, _setup, _wait_terminal_or_ready

PEER = "a" * 32


def _pred(decode_ms):
    return PlanPrediction(decode_ms=decode_ms, prefill_ms=0.0, load_ms=0.0, objective_ms=0.0, bottleneck="fixture")


def _with_pred(plan, decode_ms, plan_id=None):
    update = {"prediction": _pred(decode_ms) if decode_ms is not None else None}
    return plan.model_copy(update={**update, "id": plan_id} if plan_id else update)


def _dep(decode_ms=100.0):
    return Deployment(id="d" * 32, plan=_with_pred(_plan(), decode_ms), state=DeploymentState.ready, history=[],
                      created_at=utcnow(), revision=3)


def _proposal(decode_ms=50.0, reload_ms=1000.0, replay_ms=500.0, plan_id="plan2", remote=True, **kw):
    return Proposal(plan=_with_pred(_plan(remote), decode_ms, plan_id), reload_ms=reload_ms, replay_ms=replay_ms, **kw)


def _decide(now, proposal, current=None, policy=None, last_switch_at=None, stable_for_s=60.0):
    return decide(current or _dep(), proposal, policy or ReplanPolicy(), now, last_switch_at,
                  now - timedelta(seconds=stable_for_s), "node.verified")


# --------------------------------------------------------------------------- decide()


def test_policy_defaults_and_validation():
    policy = ReplanPolicy()
    assert (policy.dwell_s, policy.hysteresis, policy.cooldown_s) == (30.0, 0.15, 120.0)
    assert policy.horizon_tokens == HORIZON_TOKENS == 2048
    with pytest.raises(ValueError):
        ReplanPolicy(hysteresis=-0.1)
    with pytest.raises(ValueError):
        ReplanPolicy(min_confidence=2.0)
    with pytest.raises(ValueError):
        Proposal(plan=_plan(), reload_ms=-1.0, replay_ms=0.0)


def test_replace_when_every_gate_passes(now):
    d = _decide(now, _proposal())
    assert d.action == "replace" and d.trigger == "node.verified" and d.revision == 3
    assert (d.current_decode_ms, d.proposed_decode_ms, d.relative_gain) == (100.0, 50.0, 0.5)
    assert (d.reload_ms, d.replay_ms, d.break_even_tokens) == (1000.0, 500.0, 30.0)
    assert d.proposed_plan_id == "plan2" and "100.0 -> 50.0 ms/token" in d.reason and "102400 ms" in d.reason


def test_unstable_inputs_defer(now):
    d = _decide(now, _proposal(), stable_for_s=10.0)
    assert d.action == "defer" and "stable for 10.0 s, need 30.0 s" in d.reason
    assert d.relative_gain == 0.5  # the numbers are kept even when the decision waits


def test_gain_below_hysteresis_keeps(now):
    d = _decide(now, _proposal(decode_ms=90.0))
    assert d.action == "keep" and "gain 10.0% below hysteresis 15.0%" in d.reason
    assert d.relative_gain == pytest.approx(0.1)


def test_savings_below_reload_plus_replay_keep(now):
    # 20% faster saves 2048 * 20 = 40960 ms; reload 40000 + replay 1000 = 41000 ms costs more
    d = _decide(now, _proposal(decode_ms=80.0, reload_ms=40_000.0, replay_ms=1000.0))
    assert d.action == "keep" and "savings 40960 ms do not cover reload + replay 41000 ms" in d.reason
    assert d.break_even_tokens == pytest.approx(2050.0)


def test_cooldown_defers(now):
    d = _decide(now, _proposal(), last_switch_at=now - timedelta(seconds=60))
    assert d.action == "defer" and "last switch 60.0 s ago, cooldown 120.0 s" in d.reason
    assert _decide(now, _proposal(), last_switch_at=now - timedelta(seconds=121)).action == "replace"


@pytest.mark.parametrize(("current_ms", "proposed_ms", "which"), [(None, 50.0, "current"), (100.0, None, "proposed"),
                                                                  (None, None, "current and proposed")])
def test_missing_prediction_defers(now, current_ms, proposed_ms, which):
    d = _decide(now, _proposal(decode_ms=proposed_ms), current=_dep(current_ms))
    assert d.action == "defer" and f"no usable prediction for the {which} plan" in d.reason
    assert d.proposed_plan_id == "plan2" and d.reload_ms == 1000.0


@pytest.mark.parametrize(("current_ms", "proposed_ms", "which"), [
    (100.0, float("nan"), "proposed"), (100.0, float("inf"), "proposed"), (100.0, -5.0, "proposed"),
    (100.0, 0.0, "proposed"), (float("nan"), 50.0, "current"), (float("inf"), 50.0, "current")])
def test_non_finite_or_non_positive_predictions_defer(now, current_ms, proposed_ms, which):
    # NaN compares false with every threshold, so without the check a NaN prediction passed every gate to "replace"
    d = _decide(now, _proposal(decode_ms=proposed_ms), current=_dep(current_ms))
    assert d.action == "defer" and f"no usable prediction for the {which} plan" in d.reason
    p = decide_pressure(_dep(current_ms), _proposal(decode_ms=proposed_ms), now, PEER, "low")
    assert p.action == "fallback" and "no speed prediction to compare" in p.reason and p.relative_gain is None


@pytest.mark.parametrize("bad", [{"reload_ms": float("nan")}, {"replay_ms": float("inf")}, {"confidence": float("nan")},
                                 {"confidence": 1.5}, {"confidence": -0.1}])
def test_proposal_refuses_non_finite_costs_and_confidence(bad):
    args = {"reload_ms": 1.0, "replay_ms": 1.0, **bad}
    with pytest.raises(ValueError):
        Proposal(plan=_plan(), **args)


def test_outcome_is_activated_only_for_the_first_attempt(now):
    previous = _dep()
    proposal = _proposal(plan_id=previous.plan.id)  # the evaluator re-proposed the plan id already serving
    attempt = previous.model_copy(update={"id": "e" * 32, "replaces": previous.id, "revision": 4})
    rolled = previous.model_copy(update={"id": "f" * 32, "replaces": attempt.id, "revision": 5})
    assert outcome_of(attempt, proposal, previous) == "activated"
    assert outcome_of(rolled, proposal, previous) == "rolled_back"
    fell = rolled.model_copy(update={"plan": _plan(remote=False).model_copy(update={"id": "plan3"})})
    assert outcome_of(fell, proposal, previous) == "fell_back"


def test_confidence_no_proposal_and_infeasible(now):
    strict = ReplanPolicy(min_confidence=0.5)
    assert "confidence unknown below 0.50" in _decide(now, _proposal(), policy=strict).reason
    assert _decide(now, _proposal(confidence=0.3), policy=strict).action == "defer"
    assert _decide(now, _proposal(confidence=0.8), policy=strict).action == "replace"
    assert _decide(now, None).action == "keep"
    bad = Proposal(plan=_plan().model_copy(update={"feasible": False, "reasons": ["over budget"]}), reload_ms=1.0,
                   replay_ms=1.0)
    d = _decide(now, bad)
    assert d.action == "keep" and "not executable: over budget" in d.reason


def test_an_explicit_layout_is_kept_against_a_faster_plan(now):
    """A plan made from PlanRequest.layout holds the placement the owner named (services/layout.py): a faster proposal
    must not swap it for the optimizer's layout. Memory pressure is recovery and still may (decide_pressure)."""
    fixed = _dep().model_copy(update={"plan": _dep().plan.model_copy(update={
        "layout": [LayoutStage(node_id="local", device="CUDA0", layers=4)]})})
    d = _decide(now, _proposal(), current=fixed)
    assert d.action == "keep" and "explicit layout" in d.reason and d.proposed_plan_id is None
    assert _decide(now, _proposal()).action == "replace"  # the same proposal against a planner-chosen layout
    assert decide_pressure(fixed, _proposal(), now, PEER, "low").action == "fallback"


def test_pressure_decision_accepts_a_slower_plan(now):
    d = decide_pressure(_dep(), _proposal(decode_ms=300.0), now, PEER, "free RAM 200 MiB")
    assert d.action == "fallback" and d.trigger == "pressure" and d.relative_gain == pytest.approx(-2.0)
    assert "memory pressure on aaaaaaaa (free RAM 200 MiB)" in d.reason and "100.0 -> 300.0" in d.reason
    assert decide_pressure(_dep(), None, now, PEER, "x").action == "failed"


# --------------------------------------------------------------------------- controller


class Clock:
    """Advances a minute on every reading, so dwell and cooldown windows are always long past."""

    def __init__(self, start):
        self.t = start

    def __call__(self):
        self.t += timedelta(seconds=60)
        return self.t


class Evaluator:
    def __init__(self, proposals=None, gate=None, error=None):
        self.proposals = proposals or {}  # objective -> Proposal
        self.calls: list[tuple[str, str]] = []
        self.gate, self.error = gate, error
        self.entered = asyncio.Event()

    async def __call__(self, current, objective):
        self.calls.append((current.id, objective))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.proposals.get(objective)


async def _controller(store, events, tmp_path, now, evaluator, policy=None, enabled=True):
    adapter = FakeAdapter()
    mgr, reg, _ = await _setup(store, events, adapter, tmp_path)
    inf = InferenceService(store, events, mgr, FakeModels(), adapter)
    mgr.drain = lambda dep, timeout_s: inf.acquire(dep, exclusive=True, timeout=timeout_s)
    current = _with_pred(_plan(), 100.0, "planp")
    store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?)",
                  (current.id, utcnow().isoformat(), current.model_dump_json()))
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="planp"), None)).id)
    assert dep.state == DeploymentState.ready
    ctl = ReplanController(mgr, events, store, evaluator, policy, enabled=enabled, clock=Clock(now))
    return ctl, mgr, dep


@pytest.mark.anyio
async def test_controller_replaces_and_records_the_decision_with_numbers(store, events, tmp_path, now):
    evaluator = Evaluator({"speed": _proposal()})
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator)
    decision = await ctl.trigger()
    assert decision.action == "replace" and decision.outcome == "activated" and decision.trigger == "manual"
    serving = mgr.ready()[0]
    assert serving.plan.id == "plan2" and serving.revision == 1 and serving.replaces == dep.id
    assert evaluator.calls == [(dep.id, "speed")] and ctl.decisions() == [decision]
    lines = (tmp_path / "records" / "replan.jsonl").read_text(encoding="utf-8").splitlines()
    recorded = json.loads(lines[-1])
    assert recorded["relative_gain"] == 0.5 and recorded["break_even_tokens"] == 30.0
    assert recorded["outcome"] == "activated" and recorded["proposed_plan_id"] == "plan2"
    # the test clock reads one minute later: inside the 120 s cooldown, so the next evaluation waits
    again = await ctl.evaluate("node.verified")
    assert again.action == "defer" and "cooldown 120.0 s" in again.reason and mgr.ready()[0].id == serving.id


@pytest.mark.anyio
async def test_keep_is_recorded_and_nothing_is_replaced(store, events, tmp_path, now):
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, Evaluator({"speed": _proposal(decode_ms=95.0)}))
    decision = await ctl.evaluate("node.state")
    assert decision.action == "keep" and decision.outcome is None and "below hysteresis" in decision.reason
    assert mgr.ready()[0].id == dep.id
    assert [e.data["action"] for e in events.since(0) if e.type == "replan.decision"] == ["keep"]


@pytest.mark.anyio
async def test_pressure_reloads_onto_a_fitting_plan_even_when_slower(store, events, tmp_path, now):
    fit = _proposal(decode_ms=300.0, plan_id="planfit", remote=False)
    evaluator = Evaluator({"fit": fit})
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator,
                                      policy=ReplanPolicy(dwell_s=1e6, cooldown_s=1e6))
    decision = await ctl.on_pressure(PEER, "free RAM 200 MiB")
    assert decision.action == "fallback" and decision.outcome == "activated"
    assert decision.relative_gain == pytest.approx(-2.0) and evaluator.calls == [(dep.id, "fit")]
    assert mgr.ready()[0].plan.id == "planfit"
    # pressure on a node the new deployment does not use is not acted on
    unused = await ctl.on_pressure(PEER, "still low")
    assert unused.action == "keep" and "does not use that node" in unused.reason


@pytest.mark.anyio
async def test_pressure_without_a_fitting_plan_is_a_recorded_failure(store, events, tmp_path, now):
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, Evaluator())
    decision = await ctl.on_pressure(PEER, "free RAM 100 MiB")
    assert decision.action == "failed" and "no executable plan" in decision.reason
    assert mgr.ready()[0].id == dep.id and ctl.decisions() == [decision]


@pytest.mark.anyio
async def test_one_evaluation_or_replacement_at_a_time(store, events, tmp_path, now):
    evaluator = Evaluator({"speed": _proposal(), "fit": _proposal()}, gate=asyncio.Event())
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator)
    first = asyncio.create_task(ctl.evaluate("node.verified"))
    await asyncio.wait_for(evaluator.entered.wait(), 5)
    assert ctl.running
    manual = await ctl.trigger()
    pressure = await ctl.on_pressure(PEER, "low")
    assert manual.action == pressure.action == "defer" and "already running" in manual.reason
    assert "already running" in pressure.reason and len(evaluator.calls) == 1
    evaluator.gate.set()
    done = await asyncio.wait_for(first, 5)
    assert done.action == "replace" and done.outcome == "activated" and not ctl.running
    assert [d.action for d in ctl.decisions()] == ["defer", "defer", "replace"]


@pytest.mark.anyio
async def test_debounced_node_events_lead_to_one_evaluation(store, events, tmp_path, now):
    evaluator = Evaluator()
    ctl, _, _ = await _controller(store, events, tmp_path, now, evaluator, policy=ReplanPolicy(dwell_s=0.05))
    for topic in ("node.state", "deployment.state", "node.state", "node.verified"):
        ctl.observe(Event(seq=1, at=utcnow(), type=topic, message="x", data={"node_id": PEER}))
    for _ in range(200):
        if ctl.decisions():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    assert len(evaluator.calls) == 1
    assert [(d.trigger, d.action) for d in ctl.decisions()] == [("node.verified", "keep")]
    await ctl.aclose()


@pytest.mark.anyio
async def test_node_left_proposes_a_plan_on_the_remaining_nodes_without_starting_it(store, events, tmp_path, now):
    local = _proposal(decode_ms=120.0, plan_id="planlocal", remote=False)
    evaluator = Evaluator({"fit": local})
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator)
    await mgr.on_node_lost(PEER, "no heartbeat for 16s")  # the manager stops it; the controller does not
    decision = await ctl.evaluate("node.left", {PEER})
    assert decision.action == "defer" and decision.proposed_plan_id == "planlocal"
    assert decision.deployment_id == dep.id and "remaining nodes" in decision.reason
    assert evaluator.calls == [(dep.id, "fit")] and mgr.active() is None


async def _until(predicate, tries=300):
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def _node_event(topic, node_id):
    return Event(seq=1, at=utcnow(), type=topic, message="x", data={"node_id": node_id})


@pytest.mark.anyio
async def test_node_left_followed_by_another_node_event_still_proposes(store, events, tmp_path, now):
    evaluator = Evaluator({"fit": _proposal(decode_ms=120.0, plan_id="planlocal", remote=False)})
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator, policy=ReplanPolicy(dwell_s=0.05))
    await mgr.on_node_lost(PEER, "contributor stopped sharing")
    ctl.observe(_node_event("node.left", PEER))
    ctl.observe(_node_event("node.state", "b" * 32))  # a later change in the same window becomes the trigger
    await _until(lambda: ctl.decisions())
    [decision] = ctl.decisions()
    assert decision.trigger == "node.left" and decision.proposed_plan_id == "planlocal"
    assert evaluator.calls == [(dep.id, "fit")]
    await ctl.aclose()


@pytest.mark.anyio
async def test_aclose_cancels_an_evaluation_in_flight_and_stops_following(store, events, tmp_path, now):
    evaluator = Evaluator(gate=asyncio.Event())
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator, policy=ReplanPolicy(dwell_s=0.01))
    ctl.observe(_node_event("node.verified", PEER))
    await asyncio.wait_for(evaluator.entered.wait(), 5)
    assert ctl.running
    await asyncio.wait_for(ctl.aclose(), 5)  # the evaluating task was still referenced, so it is reached here
    assert not ctl.running and ctl.decisions() == []
    ctl.observe(_node_event("node.verified", PEER))  # closed: nothing is scheduled
    await asyncio.sleep(0.05)
    assert len(evaluator.calls) == 1 and mgr.ready()[0].id == dep.id


@pytest.mark.anyio
async def test_input_during_a_running_replan_is_evaluated_after_it_with_zero_dwell(store, events, tmp_path, now,
                                                                                   monkeypatch):
    monkeypatch.setattr(replan, "BUSY_RETRY_S", 0.02)
    evaluator = Evaluator(gate=asyncio.Event())
    ctl, _, _ = await _controller(store, events, tmp_path, now, evaluator, policy=ReplanPolicy(dwell_s=0.0))
    first = asyncio.create_task(ctl.evaluate("manual"))
    await asyncio.wait_for(evaluator.entered.wait(), 5)
    ctl.observe(_node_event("node.verified", PEER))  # arrives while the first evaluation runs
    await _until(lambda: any("already running" in d.reason for d in ctl.decisions()))
    evaluator.gate.set()
    await asyncio.wait_for(first, 5)
    await _until(lambda: len(evaluator.calls) == 2)  # retried, not dropped
    await _until(lambda: any(d.trigger == "node.verified" and d.action == "keep" for d in ctl.decisions()))
    await ctl.aclose()


@pytest.mark.anyio
async def test_refused_replacement_and_evaluator_failure_are_recorded(store, events, tmp_path, now):
    evaluator = Evaluator({"speed": _proposal()})
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, evaluator)

    async def busy(dep, timeout_s):
        raise DeploymentError(429, "busy", "all 1 inference slot(s) of the deployment are busy")

    mgr.drain = busy
    refused = await ctl.trigger()
    assert refused.action == "replace" and refused.outcome.startswith("refused: ")
    assert mgr.ready()[0].id == dep.id
    evaluator.error = RuntimeError("optimizer exploded")
    failed = await ctl.trigger()
    assert failed.action == "failed" and "optimizer exploded" in failed.reason


@pytest.mark.anyio
async def test_disabled_controller_does_nothing_and_the_ring_keeps_fifty(store, events, tmp_path, now):
    evaluator = Evaluator()
    ctl, _, _ = await _controller(store, events, tmp_path, now, evaluator, enabled=False)
    ctl.observe(Event(seq=1, at=utcnow(), type="node.verified", message="x", data={}))
    await ctl.run()  # returns at once when disabled
    assert (await ctl.trigger()).action == "defer" and evaluator.calls == [] and ctl.decisions() == []
    assert not (tmp_path / "records" / "replan.jsonl").exists()
    ctl.enabled = True
    for _ in range(60):
        await ctl.trigger()
    assert len(ctl.decisions()) == 50
