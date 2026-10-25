"""CapacityService.evaluate, the replan evaluator: fresh inputs on the nodes eligible now, the running deployment's own
memory counted as released, proposals with reload / replay costs, and the replan controller's decision on them."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from hypercompute.capacity.measurements import MeasurementIdentity, MeasurementRecord, MeasurementStore
from hypercompute.capacity.service import CapacityService
from hypercompute.schemas import Deployment, DeploymentState, NodeView, Plan, PlanStrategy
from hypercompute.services.leases import requests_from_plan
from hypercompute.services.replan import Proposal, ReplanPolicy, decide
from hypercompute.units import GiB
from test_capacity_bridge import (
    EP,
    MAC,
    MODEL,
    NOW,
    RELAY_GOODPUT,
    FakeLeases,
    local_node,
    mac_node,
    make_manifest,
    make_service,
    req,
    store_link,
)

DEP = "d" * 32
POLICY = ReplanPolicy(dwell_s=0.0, hysteresis=0.15, cooldown_s=0.0)


def deployment(plan: Plan) -> Deployment:
    return Deployment(id=DEP, plan=plan.model_copy(update={"revision": 1}), state=DeploymentState.ready, history=[],
                      created_at=NOW, revision=1)


def loaded(plan: Plan, local: NodeView, ram_gib: float) -> tuple[NodeView, float, FakeLeases]:
    """The host while `plan` runs: its leases are loaded, so the fresh free-memory figures no longer include them."""
    rows = [{"id": f"l{i}", "deployment_id": DEP, "node_id": r.node_id, "pool_id": r.pool_id, "bytes": r.bytes,
             "revision": 1} for i, r in enumerate(requests_from_plan(plan, {}))]
    used = {(r["node_id"], r["pool_id"]): r["bytes"] for r in rows}
    pools = [p.model_copy(update={"free_bytes": p.free_bytes - used.get(("local", p.id), 0)})
             for p in local.inventory.memory_pools]
    devices = [d.model_copy(update={"free_bytes": d.free_bytes - used.get(("local", d.pool_id), 0)})
               if d.free_bytes is not None else d for d in local.inventory.devices]
    inv = local.inventory.model_copy(update={"memory_pools": pools, "devices": devices})
    return local.model_copy(update={"inventory": inv}), ram_gib - used.get(("local", "ram"), 0) / GiB, FakeLeases(rows)


def streaming_setup(tmp_path: Path) -> tuple[Plan, NodeView, float, FakeLeases]:
    """A 7B-scale model (28 x 150 MiB blocks + a 500 MiB head) on a 4 GiB GPU with 1.5 GiB of admissible RAM: the
    host streams part of its CPU prefix from the SSD."""
    m = make_manifest()
    local = local_node(vram_free_gib=4.0)
    plan = make_service(tmp_path, m, local, ram_gib=6.0, stream=True).plan(req())
    assert plan.feasible and plan.tier == "stream" and plan.stream and plan.stream.streamed_units
    now_local, ram, leases = loaded(plan, local, 6.0)
    return plan, now_local, ram, leases


def evaluate(svc: CapacityService, current: Deployment, objective: str = "speed") -> Proposal | None:
    return asyncio.run(svc.evaluate(current, objective))  # type: ignore[arg-type]


def test_no_proposal_when_the_best_layout_is_the_running_one(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, ram_gib=ram, stream=True, leases=leases, ready={DEP})
    assert evaluate(svc, deployment(plan)) is None


def test_a_fast_lan_peer_joining_is_proposed_with_its_reload_and_replay_cost(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    current = deployment(plan)
    proposal = evaluate(svc, current)
    assert isinstance(proposal, Proposal) and proposal.plan.feasible
    new = proposal.plan
    assert new.rpc_endpoints == [EP] and "RPC0" in new.device_list and new.revision == 0
    assert new.prediction is not None and new.prediction.decode_ms < plan.prediction.decode_ms  # type: ignore[union-attr]
    assert proposal.reload_ms == pytest.approx(new.prediction.load_ms) and proposal.reload_ms > 0
    assert proposal.replay_ms > new.prediction.prefill_ms  # the full 4096-token window against a 512-token prompt
    assert proposal.confidence == 0.0  # every coefficient is a prior on this fixture
    assert proposal.current_prediction is not None
    assert proposal.current_prediction.decode_ms == pytest.approx(plan.prediction.decode_ms)  # type: ignore[union-attr]
    decision = decide(current, proposal, POLICY, NOW, None, NOW - timedelta(seconds=60))
    assert decision.action == "replace" and decision.proposed_plan_id == new.id


def test_a_relay_peer_at_the_measured_goodput_is_skipped_for_a_cold_7b_upload(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    store_link(tmp_path / "measurements", MAC, "relay")
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="relay")
    assert evaluate(svc, deployment(plan)) is None  # the best layout is still the running one
    # the same cluster before anything runs: the optimizer says why the Mac is left out
    idle = make_service(tmp_path, make_manifest(), local_node(vram_free_gib=4.0), [mac_node()], ram_gib=6.0,
                        stream=True, path="relay")
    out = idle.explain(MODEL)
    skipped = [s for s in out["best"]["skipped"] if s.startswith(f"{MAC}:MTL0: not used")]
    assert skipped and "s load" in skipped[0]
    assert any(f"{MAC}:MTL0 goodput 1.59e+06 bytes/s (measured" in line for line in out["measured"])
    # one 150 MiB block over the measured relay goodput takes about 99 s to upload
    assert 150 * (1 << 20) / RELAY_GOODPUT == pytest.approx(98.9, abs=0.1)


def with_vram(node: NodeView, free_gib: float) -> NodeView:
    """The node with CUDA0's fresh free VRAM set (pool and device agree)."""
    free = int(free_gib * GiB)
    pools = [p.model_copy(update={"free_bytes": free}) if p.id == "vram:CUDA0" else p
             for p in node.inventory.memory_pools]
    devices = [d.model_copy(update={"free_bytes": free}) if d.id == "CUDA0" else d for d in node.inventory.devices]
    return node.model_copy(update={"inventory": node.inventory.model_copy(update={"memory_pools": pools,
                                                                                    "devices": devices})})


def test_fit_proposes_an_executable_plan_when_the_gpu_shrinks(tmp_path):
    m = make_manifest()
    local = local_node()
    plan = make_service(tmp_path, m, local).plan(req())
    assert plan.tier == "gpu"
    _, ram, leases = loaded(plan, local, 16.0)
    # another program took VRAM: 0.25 GiB is free while this deployment holds its lease, so once it stops the GPU
    # admits 0.25 GiB less than the all-GPU layout needs (lease + 0.25 GiB free - 0.5 GiB headroom)
    svc = make_service(tmp_path, m, with_vram(local, 0.25), ram_gib=ram, leases=leases, ready={DEP})
    current = deployment(plan)
    assert evaluate(svc, current, "speed") is not None  # the running layout no longer fits either
    proposal = evaluate(svc, current, "fit")
    assert proposal is not None and proposal.plan.feasible and proposal.plan.tier == "gpu_ram"
    assert proposal.plan.cpu_layers > 0 and all(a.fits for a in proposal.plan.assignments)
    assert proposal.current_prediction is not None  # the running layout is still scored, though it overflows now


def test_fit_proposes_nothing_when_no_layout_fits(tmp_path):
    m = make_manifest()
    local = local_node()
    plan = make_service(tmp_path, m, local).plan(req())
    _, _, leases = loaded(plan, local, 16.0)
    # no free VRAM beyond this deployment's own lease and no RAM above the reserve: nothing fits after the stop
    svc = make_service(tmp_path, m, with_vram(local, 0.0), ram_gib=4.5, leases=leases, ready={DEP})
    assert evaluate(svc, deployment(plan), "fit") is None


def test_fresh_measurements_raise_confidence_and_stale_ones_do_not(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)

    def proposal_after(age: timedelta) -> Proposal:
        root = tmp_path / f"case-{age.days}"
        store = MeasurementStore(root / "measurements")
        for device, value in (("CPU", 12.0), ("CUDA0", 2.5)):
            identity = MeasurementIdentity(node_id="local", device=device, backend_build="b11770",
                                           kind="compute.decode_ms_per_gib")
            store.put(MeasurementRecord(identity=identity, value=value, unit="ms/GiB", measured_at=NOW - age,
                                        samples=10))
        svc = make_service(root, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                           ready={DEP}, path="lan")
        got = evaluate(svc, deployment(plan))
        assert got is not None
        return got

    fresh, stale = proposal_after(timedelta(hours=1)), proposal_after(timedelta(days=8))
    assert stale.confidence == 0.0 and 0.0 < fresh.confidence < 1.0
    priors = next(w for w in fresh.plan.warnings if w.startswith("predicted with priors"))
    assert "local:CUDA0 decode" not in priors and f"{MAC}:MTL0 decode" in priors


def test_a_deployment_without_a_prediction_gets_its_score_on_the_proposal(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    current = deployment(plan.model_copy(update={"prediction": None}))  # planned before the optimizer was wired
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    proposal = evaluate(svc, current)
    assert proposal is not None and proposal.current_prediction is not None
    assert proposal.current_prediction.decode_ms == pytest.approx(plan.prediction.decode_ms)  # type: ignore[union-attr]
    stable = NOW - timedelta(seconds=60)
    # decide() compares against the proposal's fresh score of the running layout: no stored prediction is needed
    decision = decide(current, proposal, POLICY, NOW, None, stable)
    assert decision.action == "replace"
    assert decision.current_decode_ms == pytest.approx(proposal.current_prediction.decode_ms)
    # without the fresh score it cannot compare and waits
    bare = Proposal(plan=proposal.plan, reload_ms=proposal.reload_ms, replay_ms=proposal.replay_ms)
    assert decide(current, bare, POLICY, NOW, None, stable).action == "defer"


def test_local_only_deployments_are_not_moved_onto_peers(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    assert plan.limits is not None
    local_only = PlanStrategy.local_only
    local_plan = plan.model_copy(update={"strategy": local_only,
                                         "limits": plan.limits.model_copy(update={"strategy": local_only})})
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], ram_gib=ram, stream=True, leases=leases,
                       ready={DEP}, path="lan")
    assert evaluate(svc, deployment(local_plan)) is None
    # a plan stored before limits existed keeps its own strategy field, as before
    assert evaluate(svc, deployment(local_plan.model_copy(update={"limits": None}))) is None


def test_a_non_llama_deployment_is_refused_with_a_reason(tmp_path):
    from hypercompute.capacity.service import CapacityUnavailable

    plan, local, ram, leases = streaming_setup(tmp_path)
    svc = make_service(tmp_path, make_manifest(), local, ram_gib=ram, stream=True, leases=leases, ready={DEP})
    with pytest.raises(CapacityUnavailable) as exc:
        evaluate(svc, deployment(plan.model_copy(update={"engine": "exo"})))
    assert exc.value.code == "engine"
    with pytest.raises(ValueError):
        evaluate(svc, deployment(plan), "fastest")


def test_a_running_rpc_named_peer_layout_is_recognized_as_the_best_one(tmp_path):
    from hypercompute.services.deployments import DeploymentManager

    m = make_manifest()
    local, mac = local_node(vram_free_gib=4.0), mac_node(free_gib=6.0, max_gib=1.0)
    plan = make_service(tmp_path, m, local, [mac], ram_gib=6.5).plan(req())
    named = DeploymentManager._llama_names(plan, {EP: 1})  # what the deployment runs: RPC0 instead of host:port#0
    assert [a.device for a in named.assignments] == ["CPU", "RPC0", "CUDA0"]
    now_local, ram, leases = loaded(named, local, 6.5)
    svc = make_service(tmp_path, m, now_local, [mac], ram_gib=ram, leases=leases, ready={DEP})
    assert evaluate(svc, deployment(named)) is None


def test_explain_refuses_a_context_outside_the_plan_limits(tmp_path):
    from hypercompute.capacity.service import CapacityUnavailable

    svc = make_service(tmp_path, make_manifest(), local_node())
    with pytest.raises(CapacityUnavailable) as exc:
        svc.explain(MODEL, ctx_size=64)
    assert exc.value.code == "bad_request"
    with pytest.raises(CapacityUnavailable) as exc:
        svc.explain("missing-model")
    assert exc.value.code == "unknown_model"


# --------------------------------------------------------------------------------------------------- review regressions


def test_fit_never_proposes_the_running_layout(tmp_path):
    m = make_manifest()
    local = local_node()
    plan = make_service(tmp_path, m, local).plan(req())
    now_local, ram, leases = loaded(plan, local, 16.0)
    svc = make_service(tmp_path, m, now_local, ram_gib=ram, leases=leases, ready={DEP})
    # under memory pressure a reload onto the same layout frees nothing; the controller then records "failed"
    assert evaluate(svc, deployment(plan), "fit") is None
    assert evaluate(svc, deployment(plan), "speed") is None


def test_the_same_devices_streaming_fewer_units_is_a_new_plan(tmp_path):
    plan, local, ram, leases = streaming_setup(tmp_path)
    # 2 GiB of RAM came free since the plan: the same CPU prefix can keep more of its units resident
    svc = make_service(tmp_path, make_manifest(), local, ram_gib=ram + 2.0, stream=True, leases=leases, ready={DEP})
    proposal = evaluate(svc, deployment(plan))
    assert proposal is not None and proposal.plan.cpu_layers == plan.cpu_layers
    assert [a.device for a in proposal.plan.assignments] == [a.device for a in plan.assignments]
    new_stream = proposal.plan.stream.streamed_units if proposal.plan.stream else []
    assert len(new_stream) < len(plan.stream.streamed_units)  # type: ignore[union-attr]
    assert proposal.plan.prediction.decode_ms < plan.prediction.decode_ms  # type: ignore[union-attr]


def test_run_together_deployments_keep_every_peer(tmp_path):
    m = make_manifest()
    local, mac = local_node(), mac_node()
    plan = make_service(tmp_path, m, local, [mac], path="lan").plan(req(PlanStrategy.all_selected))
    assert plan.feasible and plan.rpc_endpoints == [EP]  # the model fits CUDA0 alone; the owner asked for both
    from hypercompute.services.deployments import DeploymentManager

    named = DeploymentManager._llama_names(plan, {EP: 1})
    now_local, ram, leases = loaded(named, local, 16.0)
    svc = make_service(tmp_path, m, now_local, [mac], ram_gib=ram, leases=leases, ready={DEP}, path="lan")
    proposal = evaluate(svc, deployment(named))
    # local-only would decode faster, but a replan keeps the strategy: the running layout is already the best one
    assert proposal is None
    # the Mac left: no speed proposal can run together, recovery ("fit") takes the best layout on what is left
    alone = make_service(tmp_path, m, now_local, [], ram_gib=ram, leases=leases, ready={DEP}, path="lan")
    assert evaluate(alone, deployment(named), "speed") is None
    fit = evaluate(alone, deployment(named), "fit")
    assert fit is not None and fit.plan.feasible and fit.plan.device_list == ["CUDA0"]
    assert any(r.startswith("'run together' is not possible now") for r in fit.plan.reasons)
