"""The closed replan loop: the evaluator's fresh score of the running layout (Proposal.current_prediction), owner
replacements restarting the cooldown, request limits kept on plans and by the evaluator, capacity I/O and bridge errors
as CapacityUnavailable codes, planning off the event loop in create(), and the running CPU residency in the score."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta

import pytest
from hypercompute.capacity import bridge, runtime
from hypercompute.capacity import service as service_mod
from hypercompute.capacity.service import CapacityUnavailable, limit_violation
from hypercompute.schemas import (
    CreateDeploymentRequest,
    Deployment,
    DeploymentState,
    Event,
    Plan,
    PlanLimits,
    PlanRequest,
    PlanStrategy,
    utcnow,
)
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.replan import Proposal, ReplanPolicy, decide, decide_pressure, is_switch
from test_capacity_bridge import BOX, MAC, MODEL, local_node, mac_node, make_manifest, make_service, req
from test_capacity_evaluator import DEP, deployment, loaded, streaming_setup, with_vram
from test_deployments import FakeAdapter, _plan, _setup, _wait_terminal_or_ready
from test_replace_lifecycle import TrackingAdapter, _manager, _variant
from test_replan_controller import PEER, Evaluator, _controller, _decide, _dep, _pred, _proposal, _until, _with_pred

POLICY = ReplanPolicy(dwell_s=0.0, hysteresis=0.15, cooldown_s=0.0)


def evaluate(svc, current: Deployment, objective: str = "speed") -> Proposal | None:
    return asyncio.run(svc.evaluate(current, objective))


# --------------------------------------------------------------------------- 1. current_prediction


def test_decide_compares_against_the_fresh_score_of_the_running_layout(now):
    stored = _dep(100.0)
    d = _decide(now, _proposal(decode_ms=50.0, current_prediction=_pred(60.0)), current=stored)
    assert d.action == "replace" and d.current_decode_ms == 60.0 and d.relative_gain == pytest.approx(1 / 6)
    # 8% faster than the fresh score (45% against the stored one): below hysteresis
    assert _decide(now, _proposal(decode_ms=55.0, current_prediction=_pred(60.0)), current=stored).action == "keep"
    # a deployment without a stored prediction is compared once the evaluator scores it
    assert _decide(now, _proposal(current_prediction=_pred(100.0)), current=_dep(None)).action == "replace"
    # an unusable fresh score is not swapped for the stored one: decide waits
    d = _decide(now, _proposal(current_prediction=_pred(float("nan"))), current=stored)
    assert d.action == "defer" and "no usable prediction for the current plan" in d.reason
    p = decide_pressure(stored, _proposal(decode_ms=300.0, current_prediction=_pred(150.0)), now, PEER, "low")
    assert p.action == "fallback" and p.current_decode_ms == 150.0 and p.relative_gain == pytest.approx(-1.0)


def test_the_capacity_evaluator_returns_a_plain_proposal(tmp_path):
    assert not hasattr(service_mod, "CapacityProposal")
    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    proposal = evaluate(svc, deployment(plan))
    assert type(proposal) is Proposal and proposal.current_prediction is not None
    # same streamed set as the running plan: the score is evaluate_layout's, with no residency correction
    assert not any("residency it runs" in a for a in proposal.current_prediction.assumptions)


# --------------------------------------------------------------------------- 2. owner replacements and the cooldown


def _event(type_: str, **data: object) -> Event:
    return Event(seq=1, at=utcnow(), type=type_, message="x", data=dict(data))


@pytest.mark.parametrize(("type_", "outcome", "counts"), [
    ("deployment.replaced", "activated", True), ("deployment.replaced", "rolled_back", True),
    ("deployment.replaced", "fell_back", True), ("deployment.replace_failed", "rolled_back", True),
    ("deployment.replace_failed", "fell_back", True), ("deployment.replace_failed", "failed", True),
    ("deployment.replace_failed", "refused", False), ("deployment.replace_failed", None, False),
    ("deployment.state", "activated", False), ("node.verified", None, False)])
def test_which_deployment_events_are_switches(type_, outcome, counts):
    assert is_switch(_event(type_, **({} if outcome is None else {"outcome": outcome}))) is counts


@pytest.mark.anyio
async def test_an_owner_replace_through_the_manager_restarts_the_cooldown(store, events, tmp_path, now):
    events.bind_loop()
    ctl, mgr, dep = await _controller(store, events, tmp_path, now, Evaluator({"speed": _proposal()}))
    follow = asyncio.create_task(ctl.run())
    await asyncio.sleep(0)  # run() takes its starting sequence number
    # what POST /api/deployments/{id}/replace calls: the controller is not involved
    owner = await mgr.replace(dep.id, _with_pred(_plan(), 80.0, "planowner"), "requested by the owner")
    assert owner.state == DeploymentState.ready
    await _until(lambda: ctl._last_switch_at is not None)
    decision = await ctl.evaluate("node.verified")
    assert decision.action == "defer" and "cooldown 120.0 s" in decision.reason
    assert mgr.ready()[0].id == owner.id
    follow.cancel()
    await asyncio.gather(follow, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize(("outcome", "action"), [("refused", "replace"), ("rolled_back", "defer"),
                                                 ("fell_back", "defer"), ("failed", "defer")])
async def test_replace_failed_counts_only_when_something_was_reloaded(store, events, tmp_path, now, outcome, action):
    ctl, _, _ = await _controller(store, events, tmp_path, now, Evaluator({"speed": _proposal()}))
    ctl.observe(_event("deployment.replace_failed", outcome=outcome))
    decision = await ctl.evaluate("node.verified")
    assert decision.action == action
    assert ("cooldown" in decision.reason) is (action == "defer")


@pytest.mark.anyio
async def test_a_disabled_controller_ignores_switch_events(store, events, tmp_path, now):
    ctl, _, _ = await _controller(store, events, tmp_path, now, Evaluator(), enabled=False)
    ctl.observe(_event("deployment.replaced", outcome="activated"))
    assert ctl._last_switch_at is None


def _replace_failed(events) -> Event:
    [event] = [e for e in events.since(0) if e.type == "deployment.replace_failed"]
    return event


@pytest.mark.anyio
async def test_a_drain_timeout_is_a_refused_replacement(store, events, tmp_path):
    mgr, _, inf, old = await _manager(store, events, tmp_path, TrackingAdapter())
    mgr.drain_timeout_s = 0.05
    chat = await inf.acquire(old)  # a request that outlives the drain: nothing is stopped
    with pytest.raises(DeploymentError) as exc:
        await mgr.replace(old.id, _variant("plan2"), "manual")
    chat.release()
    event = _replace_failed(events)
    assert exc.value.code == "drain_timeout" and event.data["outcome"] == "refused" and not is_switch(event)


@pytest.mark.anyio
async def test_an_old_engine_that_did_not_stop_is_a_failed_switch(store, events, tmp_path):
    mgr, _, _, old = await _manager(store, events, tmp_path, TrackingAdapter(clean_stop=False))
    with pytest.raises(DeploymentError) as exc:
        await mgr.replace(old.id, _variant("plan2"), "manual")
    event = _replace_failed(events)  # the old deployment ended: the served model is gone, as after a reload
    assert exc.value.code == "replace_failed" and event.data["outcome"] == "failed" and is_switch(event)


# --------------------------------------------------------------------------- 3. request limits


def test_plan_limits_come_from_the_request_and_old_plans_have_none():
    request = PlanRequest(model_id=MODEL, node_ids=[MAC], strategy=PlanStrategy.local_only, allow_ram_tier=False,
                          allow_capacity_mode=False)
    limits = PlanLimits.of(request)
    assert limits == PlanLimits(allow_ram_tier=False, allow_capacity_mode=False, node_ids=[MAC],
                                strategy=PlanStrategy.local_only)
    old = _plan().model_dump(mode="json")
    old.pop("limits")
    assert Plan.model_validate(old).limits is None  # a plan stored before the field existed
    with_limits = _plan().model_copy(update={"limits": limits})
    assert Plan.model_validate_json(with_limits.model_dump_json()).limits == limits


def test_optimizer_plans_carry_the_request_limits(tmp_path):
    from test_capacity_bridge import _legacy

    request = req(node_ids=[MAC], allow_capacity_mode=False)
    plan = make_service(tmp_path, make_manifest(), local_node(), [mac_node()]).plan(request)
    assert plan.limits == PlanLimits.of(request)
    # the planner's plan, when the comparison keeps it, carries them too
    m, auto = make_manifest(), req(PlanStrategy.auto, allow_ram_tier=False)
    legacy = _legacy(local_node(), [], auto, m)
    assert legacy.limits is None
    kept = make_service(tmp_path, m, local_node()).plan(auto, legacy)
    assert kept.id == legacy.id and kept.limits == PlanLimits.of(auto)


def test_deployment_manager_plans_carry_and_store_the_request_limits(store, events, tmp_path):
    from test_selector import Models, art, summary
    from test_selector import _manager as selector_manager

    gguf = art("q14-gguf")
    mgr = selector_manager(store, events, tmp_path, Models([gguf], {gguf.id: "downloaded"},
                                                           {gguf.id: summary(48, 225, 5120, 40, 8, 609)}))
    request = PlanRequest(model_id=gguf.id, strategy=PlanStrategy.local_only, allow_capacity_mode=False)
    plan = mgr.plan(request)  # the planner path (no capacity service)
    assert plan.limits == PlanLimits.of(request)
    assert mgr.stored_plan(plan.id).limits == plan.limits


def test_the_evaluator_keeps_the_host_ram_tier_off(tmp_path):
    m = make_manifest()
    plan = make_service(tmp_path, m, local_node()).plan(req(allow_ram_tier=False))
    assert plan.tier == "gpu" and plan.limits is not None and not plan.limits.allow_ram_tier
    _, ram, leases = loaded(plan, local_node(), 16.0)
    svc = make_service(tmp_path, m, with_vram(local_node(), 0.25), ram_gib=ram, leases=leases, ready={DEP})
    # the GPU shrank: only a CPU prefix would fit, which the request turned off
    assert evaluate(svc, deployment(plan), "fit") is None
    old = evaluate(svc, deployment(plan.model_copy(update={"limits": None})), "fit")
    assert old is not None and old.plan.tier == "gpu_ram"  # a plan without limits behaves as before


def test_the_evaluator_keeps_capacity_mode_off(tmp_path):
    m = make_manifest()
    local = local_node(vram_free_gib=4.0)
    plan = make_service(tmp_path, m, local, ram_gib=16.0, stream=True).plan(req(allow_capacity_mode=False))
    assert plan.tier == "gpu_ram" and plan.limits is not None and not plan.limits.allow_capacity_mode
    now_local, _, leases = loaded(plan, local, 16.0)
    # RAM shrank: the CPU prefix fits only by streaming part of it from the SSD
    svc = make_service(tmp_path, m, now_local, ram_gib=3.0, stream=True, leases=leases, ready={DEP})
    assert evaluate(svc, deployment(plan), "fit") is None
    old = evaluate(svc, deployment(plan.model_copy(update={"limits": None})), "fit")
    assert old is not None and old.plan.tier == "stream" and old.plan.capacity_mode


def test_replan_refuses_multi_slot(tmp_path):
    """The replan search builds plans without PlanRequest.parallel, so a proposal would relaunch a multi-slot
    deployment with -np 1: the evaluator proposes nothing (None, like the moe case), and never raises."""
    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    assert evaluate(svc, deployment(plan)) is not None  # the same deployment with one slot gets a proposal
    multi = plan.model_copy(update={"engine_options": {**plan.engine_options, "parallel": 4}})
    assert evaluate(svc, deployment(multi)) is None
    assert evaluate(svc, deployment(multi), "fit") is None


def test_the_evaluator_keeps_the_selected_peers(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    picky = plan.model_copy(update={"limits": PlanLimits.of(req(node_ids=[BOX]))})  # the Mac was not selected
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    assert evaluate(svc, deployment(picky)) is None
    chosen = plan.model_copy(update={"limits": PlanLimits.of(req(node_ids=[MAC]))})
    proposal = evaluate(svc, deployment(chosen))
    assert proposal is not None and {a.node_id for a in proposal.plan.assignments} == {"local", MAC}
    assert proposal.plan.limits == chosen.limits  # the proposal keeps the limits for the next replan


def test_limit_violation_names_each_limit(tmp_path):
    m = make_manifest()
    local = local_node(vram_free_gib=4.0)
    gpu_ram = make_service(tmp_path, m, local).plan(req())
    stream = make_service(tmp_path, m, local, ram_gib=6.0, stream=True).plan(req())
    peers = make_service(tmp_path, m, local_node(vram_free_gib=4.0), [mac_node()], path="lan").plan(req(
        PlanStrategy.all_selected))
    assert gpu_ram.cpu_layers and stream.stream and stream.stream.streamed_units
    assert any(a.node_id == MAC for a in peers.assignments)
    every = PlanLimits()
    assert limit_violation(gpu_ram, None, "local") is None and limit_violation(peers, every, "local") is None
    assert "host RAM tier off" in limit_violation(gpu_ram, PlanLimits(allow_ram_tier=False), "local")
    assert "capacity mode off" in limit_violation(stream, PlanLimits(allow_capacity_mode=False), "local")
    assert limit_violation(gpu_ram, PlanLimits(allow_capacity_mode=False), "local") is None  # mapped, no streaming
    assert "local only" in limit_violation(peers, PlanLimits(strategy=PlanStrategy.local_only), "local")
    assert MAC[:8] in limit_violation(peers, PlanLimits(node_ids=[BOX]), "local")
    assert limit_violation(peers, PlanLimits(node_ids=[MAC]), "local") is None


# --------------------------------------------------------------------------- 4. errors as codes


def _raise(exc: BaseException):
    def fail(*_args: object, **_kwargs: object):
        raise exc
    return fail


def test_os_errors_reading_model_files_and_probes_are_capacity_codes(tmp_path, monkeypatch):
    m = make_manifest()
    svc = make_service(tmp_path, m, local_node())
    monkeypatch.setattr(runtime, "model_shards", _raise(PermissionError("access denied")))
    with pytest.raises(CapacityUnavailable) as exc:
        svc.explain(MODEL)
    assert exc.value.code == "model_files" and "access denied" in str(exc.value)
    monkeypatch.undo()
    for probe in ("memory", "stream", "manifest"):
        svc = make_service(tmp_path, m, local_node(), stream=True)
        svc._probes = bridge.Probes(**{**svc._probes.__dict__, probe: _raise(OSError(5, "device error"))})
        with pytest.raises(CapacityUnavailable) as exc:
            svc.explain(MODEL)
        assert exc.value.code == "io" and "device error" in str(exc.value), probe


@pytest.mark.parametrize(("code", "expected"), [("candidate", "candidate"), ("", "bridge")])
def test_bridge_errors_while_building_the_plan_are_capacity_codes(tmp_path, monkeypatch, code, expected):
    svc = make_service(tmp_path, make_manifest(), local_node())
    monkeypatch.setattr(bridge, "to_plan", _raise(bridge.BridgeError(code, "stage without a demand")))
    with pytest.raises(CapacityUnavailable) as exc:
        svc.explain(MODEL)
    assert exc.value.code == expected and "stage without a demand" in str(exc.value)


def test_an_evaluator_io_error_is_a_capacity_code_not_a_crash(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, ram_gib=ram, stream=True, leases=leases, ready={DEP})
    svc._probes = bridge.Probes(**{**svc._probes.__dict__, "memory": _raise(OSError(5, "device error"))})
    with pytest.raises(CapacityUnavailable) as exc:
        evaluate(svc, deployment(plan))
    assert exc.value.code == "io"


# --------------------------------------------------------------------------- 5. create() plans off the loop


@pytest.mark.anyio
async def test_create_plans_off_the_event_loop_and_keeps_the_create_lock(store, events, tmp_path):
    mgr, _, _ = await _setup(store, events, FakeAdapter(), tmp_path)
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def slow_plan(_req: PlanRequest) -> Plan:
        threads.append(threading.get_ident())
        time.sleep(0.2)  # the optimizer search with several peer devices
        return _plan()

    mgr.plan = slow_plan  # type: ignore[method-assign]
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(tick())
    body = CreateDeploymentRequest(plan=PlanRequest(model_id="m"))
    first, second = await asyncio.gather(mgr.create(body, None), mgr.create(body, None), return_exceptions=True)
    ticker.cancel()
    assert threads and threads[0] != loop_thread and ticks >= 5  # the loop kept running while the plan was made
    # the lock still serializes creates: the second waited, then found the first one active, and planned nothing
    assert isinstance(first, Deployment) and isinstance(second, DeploymentError)
    assert second.code == "deployment_active" and len(threads) == 1
    assert (await _wait_terminal_or_ready(mgr, first.id)).state == DeploymentState.ready


# --------------------------------------------------------------------------- 6. the running CPU residency


def test_the_current_score_uses_the_units_the_deployment_streams(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    assert plan.stream is not None and plan.prediction is not None
    # 2 GiB came free: split_cpu for today's budget streams nothing, but the deployment still streams its units
    svc = make_service(tmp_path, make_manifest(), local, ram_gib=ram + 2.0, stream=True, leases=leases, ready={DEP})
    proposal = evaluate(svc, deployment(plan))
    assert proposal is not None and proposal.current_prediction is not None
    current = proposal.current_prediction
    assert current.stages[0].streamed_units == plan.stream.streamed_units
    assert current.decode_ms == pytest.approx(plan.prediction.decode_ms)
    assert current.stream_bytes_per_token == plan.stream.stream_bytes_per_pass
    assert any("residency it runs" in a for a in current.assumptions)
    assert proposal.plan.prediction.decode_ms < current.decode_ms  # type: ignore[union-attr]
    # scored with split_cpu the running layout would equal the proposal (gain 0, keep); with its own split it moves
    decision = decide(deployment(plan), proposal, POLICY, plan.created_at, None, plan.created_at - timedelta(minutes=1))
    assert decision.action == "replace" and decision.current_decode_ms == pytest.approx(current.decode_ms)


def test_a_mapped_cpu_prefix_is_scored_without_streaming(tmp_path):
    m = make_manifest()
    local = local_node(vram_free_gib=4.0)
    plan = make_service(tmp_path, m, local, ram_gib=16.0, stream=True).plan(req())
    assert plan.tier == "gpu_ram" and plan.stream is None
    now_local, _, leases = loaded(plan, local, 16.0)
    # RAM shrank: split_cpu for today's budget would stream part of the prefix the deployment maps
    svc = make_service(tmp_path, m, now_local, ram_gib=3.0, stream=True, leases=leases, ready={DEP})
    proposal = evaluate(svc, deployment(plan), "fit")
    assert proposal is not None and proposal.plan.tier == "stream"
    current = proposal.current_prediction
    assert current is not None and current.stages[0].streamed_units == []
    assert current.stages[0].exposed_io_ms == 0.0 and current.stream_bytes_per_token == 0
    assert current.decode_ms == pytest.approx(plan.prediction.decode_ms)  # type: ignore[union-attr]


def test_the_current_score_names_the_storage_priors_of_the_split_it_runs(tmp_path, monkeypatch):
    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, ram_gib=ram + 2.0, stream=True, leases=leases, ready={DEP})
    current = evaluate(svc, deployment(plan)).current_prediction  # type: ignore[union-attr]
    # split_cpu for today's budget streams nothing and names no storage prior; the running split streams through both
    assert any(a.startswith("stream efficiency 0.7 fraction (prior") for a in current.assumptions)
    assert any(a.startswith("storage read") and "(prior" in a for a in current.assumptions)
    monkeypatch.setattr(service_mod, "_stream_priors", lambda *_a: ())
    bare = evaluate(svc, deployment(plan)).current_prediction  # type: ignore[union-attr]
    assert not any(a.startswith("stream efficiency") for a in bare.assumptions)  # what the score said before


def test_the_current_score_serializes_io_for_a_single_slot_deployment(tmp_path):
    from hypercompute.capacity import costs

    plan, local, ram, leases = streaming_setup(tmp_path)
    assert plan.stream is not None and plan.stream.slots == 2
    one = plan.model_copy(update={"stream": plan.stream.model_copy(update={"slots": 1})})
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    two_slots = evaluate(svc, deployment(plan)).current_prediction  # type: ignore[union-attr]
    one_slot = evaluate(svc, deployment(one)).current_prediction  # type: ignore[union-attr]
    cpu2, cpu1 = two_slots.stages[0], one_slot.stages[0]
    assert cpu1.streamed_units == cpu2.streamed_units == plan.stream.streamed_units  # the same set as split_cpu's
    io = costs.stream_io_ms(plan.stream.stream_bytes_per_pass, costs.storage_prior("local"))
    assert cpu2.exposed_io_ms == pytest.approx(max(io, cpu2.compute_ms) - cpu2.compute_ms)  # K = 2 overlaps
    assert cpu1.exposed_io_ms == pytest.approx(io)  # K = 1 reads, then computes
    assert one_slot.decode_ms - two_slots.decode_ms == pytest.approx(min(io, cpu2.compute_ms))
    assert any("through 1 slot(s)" in a for a in one_slot.assumptions)
