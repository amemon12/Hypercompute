"""Planner fit math, llama.cpp split reproduction, atomic leases and UMA accounting."""

import pytest
from hypercompute.schemas import DeviceAssignment, DeviceBudget, GgufSummary, Plan, PlanRequest, PlanStrategy, utcnow
from hypercompute.services.deployments import assign_rpc_names, check_participation, remap_rpc_names
from hypercompute.services.leases import LeaseConflict, LeaseManager, LeaseRequest, requests_from_plan
from hypercompute.services.planner import ModelShape, make_plan
from hypercompute.units import GiB, MiB


def qwen05_summary() -> GgufSummary:
    # Shapes from the real qwen2.5-0.5b-instruct-q4_k_m.gguf (24 layers, n_embd 896, 14 heads, 2 KV heads)
    return GgufSummary(architecture="qwen2", name="qwen", file_type="Q4_K_M", n_layer=24, n_embd=896, n_head=14,
                       n_head_kv=2, context_length=32768, vocab_size=151936, tensor_count=290,
                       total_tensor_bytes=491 * MiB, layer_bytes=[int(10.5 * MiB)] * 24, token_embd_bytes=89 * MiB,
                       output_bytes=89 * MiB, output_tied=True, other_bytes=0, chat_template_present=True)


def budget(node, device, gib, endpoint=None, pool="vram:CUDA0", kind="discrete_vram"):
    return DeviceBudget(node_id=node, device=device, description=device, endpoint=endpoint, pool_id=pool,
                        pool_kind=kind, budget_bytes=int(gib * GiB), budget_basis="test")


def test_kv_estimate_matches_observed_llama_cpp():
    shape = ModelShape.from_gguf(qwen05_summary(), 4096)
    assert shape.kv_per_layer == 2 * MiB  # llama_kv_cache: 48 MiB for 24 layers at ctx 4096 on b11770


def test_split_counts_reproduce_llama_cpp_assignment():
    """llama.cpp gives unit il to the first device whose cumulative share exceeds il/(n_layer+1).
    With --tensor-split 1,1 on 24 layers it put 13 layers on CUDA0 and 11 + output on RPC0 (real log)."""
    n_units = 25
    split = [1, 1]
    cum = [sum(split[: i + 1]) / sum(split) for i in range(len(split))]
    owner = [next(j for j, c in enumerate(cum) if il / n_units < c) for il in range(n_units)]
    assert owner.count(0) == 13 and owner.count(1) == 12  # 11 layers + the output unit
    # integer unit counts are reproduced exactly
    counts = [20, 5]
    cum = [sum(counts[: i + 1]) / n_units for i in range(2)]
    owner = [next(j for j, c in enumerate(cum) if il / n_units < c) for il in range(n_units)]
    assert owner.count(0) == 20 and owner.count(1) == 5


def test_fastest_fit_stays_local_when_it_fits():
    req = PlanRequest(model_id="m", strategy=PlanStrategy.fastest_fit)
    plan = make_plan(req, ModelShape.from_gguf(qwen05_summary(), 4096), [budget("local", "CUDA0", 10)],
                     [budget("n1", "10.0.0.5:47052#0", 8, "10.0.0.5:47052", "uma", "unified")], 16 * GiB)
    assert plan.feasible and plan.device_list == ["CUDA0"] and plan.rpc_endpoints == []


def test_run_together_gives_every_peer_layers_and_warns_when_unnecessary():
    req = PlanRequest(model_id="m", strategy=PlanStrategy.all_selected)
    plan = make_plan(req, ModelShape.from_gguf(qwen05_summary(), 4096), [budget("local", "CUDA0", 10)],
                     [budget("n1", "10.0.0.5:47052#0", 8, "10.0.0.5:47052", "uma", "unified")], 16 * GiB)
    assert plan.feasible and len(plan.assignments) == 2
    assert all(a.layers for a in plan.assignments)
    assert sum(plan.tensor_split) == 25 and plan.assignments[-1].holds_output
    assert any("fits on this machine alone" in w for w in plan.warnings)


def test_does_not_fit_is_reported_not_substituted():
    req = PlanRequest(model_id="m", strategy=PlanStrategy.local_only)
    plan = make_plan(req, ModelShape.from_gguf(qwen05_summary(), 4096), [budget("local", "CUDA0", 0.3)], [],
                     16 * GiB)
    assert not plan.feasible and plan.fit_label == "does_not_fit"
    assert any("over budget" in r for r in plan.reasons)


def test_unknown_model_shape_is_fit_unknown():
    plan = make_plan(PlanRequest(model_id="m"), None, [budget("local", "CUDA0", 10)], [], None)
    assert not plan.feasible and plan.fit_label == "fit_unknown"


def test_run_together_without_peers_is_not_feasible():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected),
                     ModelShape.from_gguf(qwen05_summary(), 4096), [budget("local", "CUDA0", 10)], [], 16 * GiB)
    assert not plan.feasible and any("at least one joined" in r for r in plan.reasons)


def _plan_with(assignments, endpoints):
    return Plan(id="p", model_id="m", strategy=PlanStrategy.all_selected, ctx_size=4096, feasible=True,
                fit_label="estimated_fit", heuristic=True, assignments=assignments,
                device_list=[a.device for a in assignments], tensor_split=[len(a.layers) for a in assignments],
                rpc_endpoints=endpoints, cpu_resident_bytes=0, reasons=[], warnings=[], candidates=[],
                created_at=utcnow())


def _assign(node, device, layers, endpoint=None, pool="vram:CUDA0", total=GiB, budget_b=2 * GiB):
    return DeviceAssignment(node_id=node, device=device, endpoint=endpoint, pool_id=pool, layers=layers,
                            holds_output=False, weight_bytes=total, kv_bytes=0, compute_bytes_est=0,
                            overhead_bytes_est=0, total_bytes_est=total, budget_bytes=budget_b, fits=True)


def test_rpc_names_follow_endpoint_order_and_remap():
    a = [_assign("local", "CUDA0", [0, 1]), _assign("n2", "10.0.0.9:47052#0", [2], "10.0.0.9:47052"),
         _assign("n1", "10.0.0.5:47052#0", [3], "10.0.0.5:47052")]
    plan = assign_rpc_names(_plan_with(a, ["10.0.0.5:47052", "10.0.0.9:47052"]),
                            {"10.0.0.5:47052": 1, "10.0.0.9:47052": 1})
    assert plan.device_list == ["CUDA0", "RPC1", "RPC0"]
    remapped = remap_rpc_names(plan, {"10.0.0.5:47052": ["RPC0"], "10.0.0.9:47052": ["RPC1"]})
    assert remapped.device_list == ["CUDA0", "RPC1", "RPC0"]


def test_remap_keeps_each_device_on_its_index_within_a_multi_gpu_peer():
    ep = "10.0.0.5:47052"
    # peers are ordered by budget, so the 24 GiB '#1' comes before the 8 GiB '#0' of the same server
    a = [_assign("local", "CUDA0", [0]), _assign("n1", f"{ep}#1", list(range(1, 18)), ep),
         _assign("n1", f"{ep}#0", list(range(18, 22)), ep)]
    plan = assign_rpc_names(_plan_with(a, [ep]), {ep: 2})
    assert plan.device_list == ["CUDA0", "RPC1", "RPC0"]
    assert remap_rpc_names(plan, {ep: ["RPC0", "RPC1"]}).device_list == ["CUDA0", "RPC1", "RPC0"]
    fastest = assign_rpc_names(_plan_with(a[:2], [ep]), {ep: 2})  # only the larger GPU was needed
    assert remap_rpc_names(fastest, {ep: ["RPC0", "RPC1"]}).device_list == ["CUDA0", "RPC1"]


def test_remap_follows_device_counts_and_flags_a_device_that_went_away():
    a, b = "10.0.0.5:47052", "10.0.0.9:47052"
    plan = assign_rpc_names(_plan_with([_assign("local", "CUDA0", [0]), _assign("n1", f"{a}#1", [1], a),
                                        _assign("n2", f"{b}#0", [2], b)], [a, b]), {a: 2, b: 1})
    assert plan.device_list == ["CUDA0", "RPC1", "RPC2"]
    # peer a now exposes one device, so b's device is RPC1 and a's second device no longer exists
    remapped = remap_rpc_names(plan, {a: ["RPC0"], b: ["RPC1"]}, {a: 2, b: 1})
    assert remapped.device_list == ["CUDA0", f"{a}#1", "RPC1"]


def _placement(*devices):
    from hypercompute.schemas import ObservedDevice, ObservedPlacement

    return ObservedPlacement(offloaded="25/25", evidence_lines=[], parsed_at=utcnow(), devices=[
        ObservedDevice(name=n, model_bytes=int(mib * MiB), kv_bytes=MiB, compute_bytes=0, layers_est=1)
        for n, mib in devices])


def test_participation_checks_every_device_of_a_multi_gpu_peer():
    ep = "10.0.0.5:47052"
    plan = _plan_with([_assign("local", "CUDA0", [0]), _assign("n1", "RPC1", list(range(1, 18)), ep),
                       _assign("n1", "RPC0", list(range(18, 22)), ep)], [ep])
    # llama.cpp names buffers RPC<index within the server>[endpoint]; RPC1 got nothing
    one_missing = _placement(("CUDA0", 100), (f"RPC0[{ep}]", 200))
    for by_endpoint in ({ep: ["RPC0", "RPC1"]}, None):
        ok, detail = check_participation(plan, one_missing, by_endpoint)
        assert not ok and "RPC1: no buffers observed" in detail and detail.count(f"RPC0[{ep}]") == 1
    ok, detail = check_participation(plan, _placement(("CUDA0", 100), (f"RPC0[{ep}]", 50), (f"RPC1[{ep}]", 200)),
                                     {ep: ["RPC0", "RPC1"]})
    assert ok and f"RPC1[{ep}]: model 200 MiB" in detail and f"RPC0[{ep}]: model 50 MiB" in detail


def test_participation_uses_the_server_index_when_only_a_later_gpu_is_planned():
    a, b = "10.0.0.5:47052", "10.0.0.9:47052"
    plan = _plan_with([_assign("local", "CUDA0", [0]), _assign("n1", "RPC1", [1], a), _assign("n2", "RPC2", [2], b)],
                      [a, b])
    placement = _placement(("CUDA0", 100), (f"RPC1[{a}]", 20), (f"RPC0[{b}]", 30))
    ok, detail = check_participation(plan, placement, {a: ["RPC0", "RPC1"], b: ["RPC2"]})
    assert ok, detail


def _peer(system, contributed, probe_free, **inventory):
    from factories import contribution, make_inventory
    from hypercompute.schemas import (
        EngineDevice,
        NodeState,
        NodeView,
        VerificationResult,
        WorkerEndpoint,
        WorkerState,
    )

    ver = VerificationResult(checked_at=utcnow(), reachable=True, engine_match=True, detail="ok", rpc_devices=[
        EngineDevice(name="RPC0", description="10.0.0.5:47052", total_bytes=24 * GiB, free_bytes=probe_free)])
    return NodeView(id="b" * 32, display_name="peer", is_local=False, session_id="s", state=NodeState.eligible,
                    state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(), heartbeat_age_s=0,
                    inventory=make_inventory(system, **inventory),
                    contribution=contribution([contributed], max_gib=20, reserve_gib=1),
                    worker=WorkerEndpoint(host="10.0.0.5", port=47052), worker_state=WorkerState.running,
                    verification=ver)


def test_remote_vram_budget_uses_the_fresh_heartbeat_value():
    from hypercompute.services.deployments import VRAM_HEADROOM, remote_budgets

    # joined with 22 GiB free; the latest heartbeat (nvidia-smi) says a game now leaves 6 GiB
    [b] = remote_budgets(_peer("windows", "CUDA0", 22 * GiB, vram_total=24 * GiB, vram_free=6 * GiB))
    assert b.pool_kind == "discrete_vram" and b.budget_bytes == 6 * GiB - VRAM_HEADROOM
    # no heartbeat figure yet: the probe value is all there is
    [b] = remote_budgets(_peer("windows", "CUDA0", 22 * GiB, vram_total=24 * GiB, vram_free=None))
    assert b.budget_bytes == 20 * GiB


def test_remote_device_description_names_its_backend_for_the_overhead_estimate():
    from hypercompute.services.deployments import remote_budgets
    from hypercompute.services.planner import OVERHEAD_BY_KIND, RPC_STAGING, _overhead

    [mac] = remote_budgets(_peer("macos", "MTL0", 20 * GiB))
    assert mac.description == "metal Apple M4 (10-core GPU) via 10.0.0.5:47052"
    assert _overhead(mac) == OVERHEAD_BY_KIND["metal"] + RPC_STAGING
    [rtx] = remote_budgets(_peer("windows", "CUDA0", 11 * GiB))
    assert "cuda NVIDIA GeForce" in rtx.description and _overhead(rtx) == OVERHEAD_BY_KIND["cuda"] + RPC_STAGING


def test_registration_is_not_participation(fixture_text):
    from hypercompute.adapters.llama_output import parse_placement

    placement = parse_placement(fixture_text("llama_server_rpc_split_b11770.log"))
    good = _plan_with([_assign("local", "CUDA0", list(range(13))),
                       _assign("n1", "RPC0", list(range(13, 24)), "127.0.0.1:47052")], ["127.0.0.1:47052"])
    ok, detail = check_participation(good, placement)
    assert ok and "RPC0[127.0.0.1:47052]" in detail
    # a joined, eligible peer that the log shows no buffers for does not count as participating
    ghost = _plan_with([_assign("local", "CUDA0", list(range(13))),
                        _assign("n2", "RPC1", [13], "10.0.0.77:47052")], ["10.0.0.77:47052"])
    ok, _ = check_participation(ghost, placement)
    assert not ok


def test_leases_are_atomic_and_count_uma_once(store):
    leases = LeaseManager(store)
    # Mac: CPU and GPU draw from one unified pool; two requests on the same pool share one budget
    reqs = [LeaseRequest("mac", "uma", 6 * GiB, pool_budget=10 * GiB, node_budget=10 * GiB),
            LeaseRequest("local", "vram:CUDA0", 4 * GiB, pool_budget=11 * GiB, node_budget=11 * GiB)]
    leases.reserve("d1", reqs)
    with pytest.raises(LeaseConflict):
        leases.reserve("d2", [LeaseRequest("local", "vram:CUDA0", 1 * GiB, 11 * GiB, 11 * GiB),
                              LeaseRequest("mac", "uma", 5 * GiB, 10 * GiB, 10 * GiB)])
    # rollback: the CUDA0 part of the failed reservation was not kept
    active = leases.active()
    assert sorted(r["deployment_id"] for r in active) == ["d1", "d1"]
    assert leases.release("d1") == 2 and leases.active() == []


def test_lease_requests_group_by_pool():
    a = [_assign("mac", "RPC0", [0], "10.0.0.5:47052", pool="uma", total=3 * GiB, budget_b=10 * GiB),
         _assign("mac", "RPC1", [1], "10.0.0.5:47052", pool="uma", total=2 * GiB, budget_b=10 * GiB)]
    reqs = requests_from_plan(_plan_with(a, ["10.0.0.5:47052"]), {"mac": 10 * GiB})
    assert len(reqs) == 1 and reqs[0].bytes == 5 * GiB


def test_lease_reconcile_releases_orphans(store):
    leases = LeaseManager(store)
    leases.reserve("ghost", [LeaseRequest("n", "ram", GiB, 2 * GiB, 2 * GiB)])
    assert leases.reconcile(set()) == 1 and leases.active() == []
