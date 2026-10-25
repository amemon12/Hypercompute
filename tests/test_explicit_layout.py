"""Explicit layouts (PlanRequest.layout, services/layout.py): good and bad layouts, the plan and llama-server arguments
for a peer whose worker runs -d MTL0,CPU with a file-backed CPU stage (hcpeer), the naming and placement check of its
two RPC devices, the gateway path (422s, stored plan, leases) and the agent's hcpeer report. No hardware is touched.

Model shapes are the gpt-oss-120b approximation of test_moe_placement, not exact figures.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from factories import contribution, make_inventory
from hypercompute.adapters.base import DeploymentSpec
from hypercompute.adapters.llama_output import check_participation, parse_placement
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.agent.hcpeer import HcpeerBanner
from hypercompute.agent.join import heartbeat_exclude
from hypercompute.agent.supervisor import Supervisor
from hypercompute.schemas import (
    HCPEER_REPORT_FEATURE,
    MANAGED_PEERS_FEATURE,
    CommandAck,
    EngineDevice,
    HeartbeatRequest,
    LayoutStage,
    ManagedReport,
    ModelView,
    NodeState,
    NodeView,
    Plan,
    PlanRequest,
    SeedProgress,
    VerificationResult,
    WorkerEndpoint,
    WorkerState,
)
from hypercompute.services import planner
from hypercompute.services.deployments import DeploymentError, DeploymentManager, lease_budgets, remote_budgets
from hypercompute.services.layout import LayoutError, PeerFacts, disk_budget, layout_plan, peer_facts
from hypercompute.services.leases import LeaseManager, requests_from_plan
from hypercompute.services.peer_seed import ManagedPeers, SeedBuild, SeededShard
from hypercompute.units import GiB
from pydantic import ValidationError
from test_capacity_bridge import MODEL, NOW, artifact, local_node
from test_managed_peers import REPO, REV, consent, make_agent, tensors
from test_moe_placement import oss120, oss120_summary
from test_split_order import cpu, gpu

MAC_A, MAC_B = "a" * 32, "b" * 32
EP_A, EP_B = "127.0.0.1:50052", "127.0.0.1:50053"


def mac(node_id: str = MAC_A, endpoint: str = EP_A, devices: tuple[str, ...] = ("MTL0", "CPU"), *,
        max_gib: float = 7.5, free_gib: float = 14.0) -> NodeView:
    """An M4 Air whose worker runs -d MTL0,CPU: two RPC devices, both described by the endpoint (ggml-rpc.cpp)."""
    host, port = endpoint.rsplit(":", 1)
    rpc = [EngineDevice(name=f"RPC{i}", description=endpoint, total_bytes=int(12.7 * GiB), free_bytes=int(12.7 * GiB))
           for i in range(len(devices))]
    return NodeView(id=node_id, display_name=f"mac-{node_id[0]}", is_local=False, session_id="s",
                    state=NodeState.eligible, state_reason=None, joined_at=NOW, last_heartbeat_at=NOW,
                    heartbeat_age_s=1.0, inventory=make_inventory("macos", ram_total=16 * GiB,
                                                                  ram_free=int(free_gib * GiB)),
                    contribution=contribution(list(devices), max_gib=max_gib, reserve_gib=0),
                    worker=WorkerEndpoint(host=host, port=int(port)), worker_state=None,
                    verification=VerificationResult(checked_at=NOW, reachable=True, engine_match=True,
                                                    rpc_devices=rpc, detail="fixture"))


def report(**over: Any) -> ManagedReport:
    base: dict[str, Any] = {"devices_allowed": ["MTL0", "CPU"], "max_disk_bytes": 120 * GiB,
                            "min_free_bytes": 20 * GiB, "max_threads": 10, "cache_enabled": True,
                            "worker_devices": ["MTL0", "CPU"], "worker_generation": 1, "cache_bytes": 10 * GiB,
                            "disk_free_bytes": 150 * GiB, "hcpeer_mmap": True}
    return ManagedReport(**{**base, **over})


def peers(*nodes: tuple[NodeView, ManagedReport | None]) -> dict[str, PeerFacts]:
    return {n.id: peer_facts(n, remote_budgets(n, whole_contribution=True), r) for n, r in nodes}


def stage(node_id: str, device: str, layers: int, residency: str | None = None) -> LayoutStage:
    return LayoutStage(node_id=node_id, device=device, layers=layers, residency=residency)  # type: ignore[arg-type]


A_M3, A_C6, LOCAL27 = stage(MAC_A, "MTL0", 3), stage(MAC_A, "CPU", 6), stage("local", "CUDA0", 27)
GOOD = [A_M3, A_C6, LOCAL27]


def plan_for(stages: list[LayoutStage], facts: dict[str, PeerFacts] | None = None, *, shape: Any = "oss",
             **req: Any) -> Plan:
    request = PlanRequest(model_id="oss", layout=stages, **req)
    return layout_plan(request, oss120() if shape == "oss" else shape, [gpu(10.27)], cpu(6),
                       peers((mac(), report())) if facts is None else facts, 10 * GiB)


def args_for(plan: Plan) -> list[str]:
    adapter = object.__new__(LlamaRpcAdapter)  # build_args reads only the spec
    return adapter.build_args(DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m"), 9000)


# --------------------------------------------------------------------------------------------------- good layouts


def test_a_mac_with_metal_and_a_file_backed_cpu_stage_becomes_one_moe_plan():
    shape = oss120()
    plan = plan_for(GOOD)
    assert plan.feasible and plan.tier == "moe_offload" and plan.layout == GOOD and plan.moe is not None
    host_cpu, metal, mac_cpu, cuda = plan.assignments
    assert (host_cpu.device, metal.device, mac_cpu.device, cuda.device) == ("CPU", f"{EP_A}#0", f"{EP_A}#1", "CUDA0")
    assert metal.layers == [0, 1, 2] and mac_cpu.layers == list(range(3, 9)) and cuda.layers == list(range(9, 36))
    assert not metal.file_backed and mac_cpu.file_backed and cuda.holds_output
    assert mac_cpu.weight_bytes == sum(shape.layer_bytes[i] for i in range(3, 9))  # all on the Mac's disk
    # file-backed: only the dense parts, KV and buffers stay in the Mac's RAM (a peer CPU device: CPU overhead)
    assert mac_cpu.total_bytes_est == (sum(shape.dense(i) for i in range(3, 9)) + 6 * shape.kv_per_layer
                                       + shape.compute_est + planner.OVERHEAD_BY_KIND["cpu"] + planner.RPC_STAGING)
    assert metal.total_bytes_est == (metal.weight_bytes + 3 * shape.kv_per_layer + shape.compute_est
                                     + planner.OVERHEAD_BY_KIND["metal"] + planner.RPC_STAGING)
    assert plan.device_list == [f"{EP_A}#0", f"{EP_A}#1", "CUDA0"] and plan.tensor_split == [3, 6, 28]
    assert plan.rpc_endpoints == [EP_A] and plan.capacity_mode and plan.cpu_layers == 0
    assert set(plan.moe.cpu_expert_layers) <= set(range(9, 36))  # -ot names only the host GPU's layers
    assert plan.reasons[0].startswith("explicit layout as requested")
    assert any("file-backed by hcpeer" in w and "estimates" in w for w in plan.warnings)
    assert [c.label for c in plan.candidates] == ["explicit layout"]


def test_the_rpc_devices_are_named_in_endpoint_order_then_worker_d_order_and_reach_llama_server():
    stages = [stage(MAC_B, "MTL0", 3), stage(MAC_B, "CPU", 3), stage(MAC_A, "MTL0", 3), stage(MAC_A, "CPU", 3),
              stage("local", "CUDA0", 24)]
    facts = peers((mac(), report()), (mac(MAC_B, EP_B), report()))
    plan = DeploymentManager._llama_names(plan_for(stages, facts), {EP_A: 2, EP_B: 2})
    # --rpc lists EP_A first, so its MTL0 and CPU are RPC0 and RPC1; the chain still visits mac B first
    assert plan.rpc_endpoints == [EP_A, EP_B] and plan.device_list == ["RPC2", "RPC3", "RPC0", "RPC1", "CUDA0"]
    args = args_for(plan)
    assert args[args.index("--rpc") + 1] == f"{EP_A},{EP_B}" and args.index("--rpc") < args.index("-ot")
    assert args[args.index("--device") + 1] == "RPC2,RPC3,RPC0,RPC1,CUDA0"
    assert args[args.index("--tensor-split") + 1] == "3,3,3,3,25" and args[args.index("-ngl") + 1] == "all"
    assert plan.moe is not None and args[args.index("-ot") + 1] == plan.moe.override
    assert plan.moe.override.endswith("=CPU") and "RPC" not in plan.moe.override


def test_placement_verification_accepts_the_rpc_cpu_buffers_of_a_two_device_worker():
    plan = DeploymentManager._llama_names(plan_for(GOOD), {EP_A: 2})
    log = "\n".join([
        f"load_tensors: RPC0[{EP_A}] model buffer size =  5022.00 MiB",
        f"load_tensors: RPC1[{EP_A}] model buffer size = 10044.50 MiB",
        "load_tensors:   CPU_Mapped model buffer size = 37000.00 MiB",
        "load_tensors:        CUDA0 model buffer size =  9000.00 MiB",
        f"llama_kv_cache: RPC0[{EP_A}] KV buffer size =    24.00 MiB",
        f"llama_kv_cache: RPC1[{EP_A}] KV buffer size =    48.00 MiB",
        "llama_kv_cache:      CUDA0 KV buffer size =   216.00 MiB",
        "llama_kv_cache: size =  288.00 MiB (  4096 cells,  36 layers,  1/1 seqs)"])
    ok, detail = check_participation(plan, parse_placement(log), {EP_A: ["RPC0", "RPC1"]})
    assert ok, detail
    assert f"RPC1[{EP_A}]: model 10044 MiB, KV 48 MiB, ~6 layers (planned 6, file-backed)" in detail
    assert f"RPC0[{EP_A}]: model 5022 MiB, KV 24 MiB, ~3 layers (planned 3)" in detail
    without_cpu = "\n".join(line for line in log.splitlines() if "RPC1" not in line)
    ok, detail = check_participation(plan, parse_placement(without_cpu), {EP_A: ["RPC0", "RPC1"]})
    assert not ok and "RPC1: no buffers observed" in detail


def test_a_layout_the_gpu_holds_alone_is_a_plain_gpu_plan():
    plan = plan_for([stage("local", "CUDA0", 4)], {}, shape=oss120(n_layer=4))
    assert plan.feasible and plan.tier == "gpu" and plan.moe is None and not plan.capacity_mode
    assert plan.device_list == ["CUDA0"] and plan.tensor_split == [5] and plan.rpc_endpoints == []


def test_without_hcpeer_a_peer_cpu_stage_is_resident_and_file_backed_is_refused():
    off = peers((mac(), report(hcpeer_mmap=False)))
    plan = plan_for([A_M3, stage(MAC_A, "CPU", 1), stage("local", "CUDA0", 32)], off)
    mac_cpu = next(a for a in plan.assignments if a.device == f"{EP_A}#1")
    assert not mac_cpu.file_backed and mac_cpu.total_bytes_est > mac_cpu.weight_bytes  # weights in its RAM
    with pytest.raises(LayoutError) as exc:
        plan_for([A_M3, stage(MAC_A, "CPU", 6, "resident"), LOCAL27], off)
    assert exc.value.code == "layout_memory" and "(weights, KV and buffers)" in str(exc.value)
    for facts, why in ((off, "hcpeer mmap : on"), (peers((mac(), None)), "not a managed peer"),
                       (peers((mac(), report(hcpeer_mmap=None))), "does not report hcpeer")):
        with pytest.raises(LayoutError) as exc:
            plan_for([A_M3, stage(MAC_A, "CPU", 6, "file_backed"), LOCAL27], facts)
        assert exc.value.code == "hcpeer_not_reported" and why in str(exc.value)
        assert exc.value.fix is not None and "HC_PEER_MMAP=1" in exc.value.fix


def test_disk_budget_is_what_hcpeer_still_lets_the_worker_save():
    """hcpeer v4 checks every save against HC_PEER_CACHE_MAX_GIB (--max-disk in whole GiB) minus every file in the
    cache directory, and against the free-space floor; the agent's disk guard frees room below the cap only a pass at
    a time, which a load can outrun, so files already cached are not room."""
    room, basis = disk_budget(report())  # 120 GiB cap, 10 GiB cached, 150 GiB free, 20 GiB floor
    assert room == (120 - 10) * GiB and "--max-disk" in basis and "120 GiB" in basis and "10.0 GiB cached" in basis
    assert disk_budget(report(max_disk_bytes=12 * GiB, cache_bytes=8 * GiB))[0] == 4 * GiB
    assert disk_budget(report(max_disk_bytes=int(12.9 * GiB), cache_bytes=8 * GiB))[0] == 4 * GiB  # whole GiB
    room, basis = disk_budget(report(max_disk_bytes=0))  # no cap: the volume above its floor, cached files no room
    assert room == (150 - 20) * GiB and "floor" in basis
    assert disk_budget(report(cache_bytes=0, disk_free_bytes=30 * GiB))[0] == 10 * GiB  # the floor binds first
    assert disk_budget(report(cache_bytes=0, disk_free_bytes=10 * GiB))[0] == 0  # never below zero
    assert disk_budget(report(cache_bytes=130 * GiB))[0] == 0
    assert disk_budget(report(cache_enabled=False))[0] is None
    assert disk_budget(report(disk_free_bytes=None))[0] is None
    assert disk_budget(report(cache_bytes=None))[0] is None


# --------------------------------------------------------------------------------------------------- bad layouts


@pytest.mark.parametrize("stages, code, needle", [
    ([A_M3, A_C6, stage("local", "CUDA0", 26)], "layout_layer_count", "hold 35 layer(s)"),
    ([LOCAL27, A_M3, A_C6], "layout_order", "last stage must be this machine's GPU"),
    ([A_M3, stage("local", "CUDA0", 6), LOCAL27], "layout_order", "only the last stage"),
    ([stage("c" * 32, "MTL0", 3), A_C6, LOCAL27], "layout_node", "not an eligible peer"),
    ([stage(MAC_A, "MTL1", 3), A_C6, LOCAL27], "layout_device", "no verified RPC device MTL1"),
    ([A_C6, A_M3, LOCAL27], "layout_order", "-d order (MTL0,CPU)"),
    ([A_M3, stage(MAC_A, "MTL0", 6), LOCAL27], "layout_order", "each device once"),
    ([stage(MAC_A, "MTL0", 3, "file_backed"), A_C6, LOCAL27], "layout_residency", "only CPU-buffer weights"),
    ([A_M3, A_C6, stage("local", "CUDA0", 27, "file_backed")], "layout_residency", "only a peer's CPU device"),
    ([A_M3, A_C6, stage("local", "MTL0", 27)], "layout_device", "no contributed accelerator MTL0"),
    ([stage(MAC_A, "MTL0", 5), stage(MAC_A, "CPU", 4), LOCAL27], "layout_memory", "mac-a MTL0 needs"),
    ([stage(MAC_A, "MTL0", 4), stage(MAC_A, "CPU", 5), LOCAL27], "layout_memory", "of RAM together"),
    ([A_M3, stage(MAC_A, "CPU", 6, "resident"), LOCAL27], "layout_residency", "CPU stage cannot be resident"),
])
def test_invalid_layouts_are_refused_with_the_reason(stages, code, needle):
    with pytest.raises(LayoutError) as exc:
        plan_for(stages)
    assert exc.value.code == code and needle in str(exc.value), str(exc.value)


def test_a_peer_may_be_visited_again_later_in_its_worker_d_order():
    # Every peer's Metal stage first, then the file-backed CPU stages: no stage hands off from a peer's MTL0 to its own
    # CPU, a same-server copy that fails and falls back to a get and a set (two round trips instead of one).
    stages = [A_M3, stage(MAC_B, "MTL0", 3), stage(MAC_A, "CPU", 3), stage(MAC_B, "CPU", 3),
              stage("local", "CUDA0", 24)]
    facts = peers((mac(), report()), (mac(MAC_B, EP_B), report()))
    plan = DeploymentManager._llama_names(plan_for(stages, facts), {EP_A: 2, EP_B: 2})
    assert plan.rpc_endpoints == [EP_A, EP_B] and plan.device_list == ["RPC0", "RPC2", "RPC1", "RPC3", "CUDA0"]
    args = args_for(plan)
    assert args[args.index("--device") + 1] == "RPC0,RPC2,RPC1,RPC3,CUDA0"
    assert args[args.index("--tensor-split") + 1] == "3,3,3,3,25"
    layers = {a.device: (a.layers[0], a.layers[-1]) for a in plan.assignments if a.layers and a.endpoint}
    assert layers == {"RPC0": (0, 2), "RPC2": (3, 5), "RPC1": (6, 8), "RPC3": (9, 11)}


@pytest.mark.parametrize("revisit", [stage(MAC_A, "MTL0", 3), stage(MAC_A, "CPU", 3)])
def test_a_revisit_still_follows_the_worker_d_order_and_uses_each_device_once(revisit):
    facts = peers((mac(), report()), (mac(MAC_B, EP_B), report()))
    first = A_M3 if revisit.device == "MTL0" else A_C6
    peers_part = [first, stage(MAC_B, "MTL0", 3), revisit]
    gpu = stage("local", "CUDA0", 36 - sum(s.layers for s in peers_part))  # the "oss" shape has 36 layers
    with pytest.raises(LayoutError, match="-d order"):
        plan_for([*peers_part, gpu], facts)


def test_file_backed_bytes_must_fit_the_peer_disk_budget_and_capacity_mode():
    small = peers((mac(), report(cache_bytes=0, disk_free_bytes=25 * GiB)))  # 5 GiB above the 20 GiB floor
    with pytest.raises(LayoutError) as exc:
        plan_for(GOOD, small)
    assert exc.value.code == "layout_disk" and "~9.8 GiB" in str(exc.value) and "floor" in str(exc.value)
    with pytest.raises(LayoutError) as exc:
        plan_for(GOOD, allow_capacity_mode=False)
    assert exc.value.code == "layout_capacity_mode"
    assert exc.value.fix is not None and "HC_PEER_MMAP=1" in exc.value.fix  # resident needs hcpeer off


def test_a_cache_full_of_another_model_leaves_no_room_for_the_file_backed_stage():
    """Regression: the budget counted cached files as room, so a 12 GiB cap with 8 GiB of another model cached
    admitted 9.8 GiB file-backed; hcpeer saves 4 GiB at most and copies the rest of the CPU stage into the Mac's RAM."""
    with pytest.raises(LayoutError) as exc:
        plan_for(GOOD, peers((mac(), report(max_disk_bytes=12 * GiB, cache_bytes=8 * GiB))))
    assert exc.value.code == "layout_disk" and "4.0 GiB more" in str(exc.value)
    assert exc.value.fix is not None and "/seed" in exc.value.fix


def test_the_resident_metal_stage_saves_into_the_same_cache_first():
    """rpc-server -c saves every hash-cached weight it is sent, Metal's included, and b11770 loads RPC0 (MTL0) before
    RPC1 (CPU): an empty 12 GiB cache takes the 4.9 GiB Metal stage and only part of the 9.8 GiB CPU stage."""
    with pytest.raises(LayoutError) as exc:
        plan_for(GOOD, peers((mac(), report(max_disk_bytes=12 * GiB, cache_bytes=0))))
    assert exc.value.code == "layout_disk" and "~14.7 GiB" in str(exc.value)
    assert plan_for(GOOD, peers((mac(), report(max_disk_bytes=15 * GiB, cache_bytes=0)))).feasible


def test_a_finished_seed_of_the_peer_layers_is_already_in_its_cache():
    """Seeded tensors are cache hits at load (mapped or copied, never saved again), so a finished seed of this model's
    layers on that peer is credited even when it filled the cap; another model's, or fewer layers, are not."""
    full = report(max_disk_bytes=15 * GiB, cache_bytes=15 * GiB)

    def facts(seeded: SeededShard | None) -> dict[str, PeerFacts]:
        return {MAC_A: peer_facts(mac(), remote_budgets(mac(), whole_contribution=True), full, seeded)}

    assert plan_for(GOOD, facts(SeededShard("oss", frozenset(range(9))))).feasible
    for other in (SeededShard("other", frozenset(range(9))), SeededShard("oss", frozenset(range(8))), None):
        with pytest.raises(LayoutError) as exc:
            plan_for(GOOD, facts(other))
        assert exc.value.code == "layout_disk"


def test_a_resident_cpu_stage_is_refused_on_a_peer_whose_worker_runs_hcpeer():
    """hcpeer has no per-stage opt-out: with HC_PEER_MMAP=1 and -c it maps every cached CPU weight over 10 MiB, so a
    stage planned and recorded as resident would run file-backed. A peer that reports hcpeer on but no cache volume
    cannot plan its CPU stage resident either."""
    one = [A_M3, stage(MAC_A, "CPU", 1, "resident"), stage("local", "CUDA0", 32)]
    with pytest.raises(LayoutError) as exc:
        plan_for(one)
    assert exc.value.code == "layout_residency" and "hcpeer" in str(exc.value)
    assert exc.value.fix is not None and "without HC_PEER_MMAP=1" in exc.value.fix
    unset = [A_M3, stage(MAC_A, "CPU", 1), stage("local", "CUDA0", 32)]
    with pytest.raises(LayoutError) as exc:
        plan_for(unset, peers((mac(), report(disk_free_bytes=None))))
    assert exc.value.code == "hcpeer_not_reported" and "no free space" in str(exc.value)


@pytest.mark.anyio
async def test_the_host_credits_a_seed_only_once_the_peer_reports_it_done():
    mp = ManagedPeers()
    mp.on_heartbeat(MAC_A, report())

    def progress(job_id: str, **over: Any) -> SeedProgress:
        base: dict[str, Any] = {"job_id": job_id, "model_id": "oss", "state": "done", "parts_received": 1,
                                "parts": 1, "tensors_total": 2, "tensors_done": 2, "tensors_present": 0,
                                "tensors_failed": 0, "bytes_total": 2 * GiB, "bytes_done": 2 * GiB, "updated_at": NOW}
        return SeedProgress(**{**base, **over})

    async def deliver(layers: list[int], build: SeedBuild) -> str:
        job = mp.start_seed(MAC_A, "oss", layers, lambda stop: build)
        await asyncio.sleep(0.05)
        cmds, _ = mp.on_heartbeat(MAC_A, report())
        assert mp.seeded(MAC_A) is None  # queued for the peer, not done there
        ack = CommandAck(id=cmds[0].id, kind="seed", status="accepted")
        mp.on_heartbeat(MAC_A, report(acks=[ack], seed=progress(job.job_id, state="running")))
        assert job.state == "delivered"
        return job.job_id

    job_id = await deliver(list(range(9)), SeedBuild(REPO, REV, tensors(2), 2 * GiB, []))
    for bad in (progress(job_id, state="running"), progress(job_id, state="failed", tensors_failed=1),
                progress("f" * 32)):
        mp.on_heartbeat(MAC_A, report(seed=bad))
        assert mp.seeded(MAC_A) is None
    mp.on_heartbeat(MAC_A, report(seed=progress(job_id)))
    assert mp.seeded(MAC_A) == SeededShard("oss", frozenset(range(9)))
    skipped = await deliver([0], SeedBuild(REPO, REV, tensors(1), GiB, ["blk.0.x"]))
    mp.on_heartbeat(MAC_A, report(seed=progress(skipped)))
    assert mp.seeded(MAC_A) is None  # a tensor the host could not size was never seeded


def test_the_host_side_must_fit_and_the_request_must_be_a_gguf_llama_split():
    dense = replace(oss120(), expert_bytes=(), expert_count=0, expert_used_count=0)
    with pytest.raises(LayoutError, match="cannot hold its 36 layer"):
        plan_for([stage("local", "CUDA0", 36)], {}, shape=dense)
    with pytest.raises(LayoutError, match="turned off"):
        plan_for([stage("local", "CUDA0", 36)], {}, allow_ram_tier=False)
    with pytest.raises(LayoutError) as exc:
        plan_for(GOOD, engine="exo")
    assert exc.value.code == "layout_engine"
    with pytest.raises(LayoutError) as exc:
        plan_for(GOOD, shape=None)
    assert exc.value.code == "layout_needs_gguf"


def test_a_peer_cpu_device_is_priced_and_sized_as_a_cpu():
    caps = remote_budgets(mac(), whole_contribution=True)
    assert [planner._kind(b) for b in caps] == ["metal", "cpu"]
    assert [b.budget_bytes for b in caps] == [int(7.5 * GiB)] * 2  # each may take the whole contribution
    shared = remote_budgets(mac())  # the planner's own split keeps sharing it by reported free memory
    assert sum(b.budget_bytes for b in shared) <= int(7.5 * GiB) and shared[0].budget_bytes < int(7.5 * GiB)


# --------------------------------------------------------------------------------------------------- gateway path


class Nodes:
    def __init__(self, remote: list[NodeView], reports: dict[str, ManagedReport],
                 seeds: dict[str, SeededShard] | None = None) -> None:
        self.local, self.remote = local_node(), remote
        # services.peer_seed.ManagedPeers.report and .seeded
        self.managed = SimpleNamespace(report=reports.get, seeded=(seeds or {}).get)

    def eligible_remote(self, node_ids: list[str] | None) -> list[NodeView]:
        return [n for n in self.remote if node_ids is None or n.id in node_ids]

    def get(self, node_id: str) -> NodeView | None:
        return self.local if node_id == "local" else next((n for n in self.remote if n.id == node_id), None)


class Models:
    def view(self, model_id: str) -> ModelView:
        return ModelView(artifact=artifact(), status="downloaded", local_path=None, downloaded_bytes=0, total_bytes=0,
                         verified_sha256=True, gguf=oss120_summary())

    def artifacts(self) -> list:
        return [artifact()]


def manager(store, events, tmp_path: Path, nodes: Nodes) -> DeploymentManager:
    return DeploymentManager(store, events, nodes, Models(), LeaseManager(store), object(),  # type: ignore[arg-type]
                             Supervisor(tmp_path / "m.json"))


def test_the_gateway_plans_names_stores_and_admits_an_explicit_layout(store, events, tmp_path):
    nodes = Nodes([mac()], {MAC_A: report()})
    mgr = manager(store, events, tmp_path, nodes)
    plan = mgr.plan(PlanRequest(model_id=MODEL, layout=GOOD))
    assert plan.feasible and plan.layout == GOOD and plan.device_list == ["RPC0", "RPC1", "CUDA0"]
    assert [a.file_backed for a in plan.assignments if a.endpoint] == [False, True] and plan.limits is not None
    assert mgr.stored_plan(plan.id) == plan  # what POST /api/deployments {plan_id} deploys
    requests = requests_from_plan(plan, lease_budgets(plan, nodes))  # type: ignore[arg-type]
    assert LeaseManager(store).admits(requests) is None
    mac_lease = next(r for r in requests if r.node_id == MAC_A)
    assert mac_lease.bytes <= mac_lease.pool_budget == int(7.5 * GiB)  # MTL0 and CPU share its uma pool


def test_the_gateway_answers_an_invalid_layout_with_a_422(store, events, tmp_path):
    mgr = manager(store, events, tmp_path, Nodes([mac()], {MAC_A: report()}))
    with pytest.raises(DeploymentError) as exc:
        mgr.plan(PlanRequest(model_id=MODEL, layout=[A_M3, A_C6, stage("local", "CUDA0", 26)]))
    assert exc.value.status == 422 and exc.value.code == "layout_layer_count"
    unmanaged = manager(store, events, tmp_path, Nodes([mac()], {}))  # no report: its CPU stage is resident
    with pytest.raises(DeploymentError) as exc:
        unmanaged.plan(PlanRequest(model_id=MODEL, layout=GOOD))
    assert exc.value.status == 422 and exc.value.code == "layout_memory"


def test_the_gateway_credits_the_peer_finished_seed_of_its_layers(store, events, tmp_path):
    full = {MAC_A: report(max_disk_bytes=15 * GiB, cache_bytes=15 * GiB)}
    with pytest.raises(DeploymentError) as exc:
        manager(store, events, tmp_path, Nodes([mac()], full)).plan(PlanRequest(model_id=MODEL, layout=GOOD))
    assert exc.value.status == 422 and exc.value.code == "layout_disk"
    seeded = Nodes([mac()], full, {MAC_A: SeededShard(MODEL, frozenset(range(9)))})
    assert manager(store, events, tmp_path, seeded).plan(PlanRequest(model_id=MODEL, layout=GOOD)).feasible


def test_without_a_layout_the_gateway_plans_as_before(store, events, tmp_path):
    plan = manager(store, events, tmp_path, Nodes([mac()], {MAC_A: report()})).plan(PlanRequest(model_id=MODEL))
    assert plan.layout is None and not any(a.file_backed for a in plan.assignments)
    assert [c.label for c in plan.candidates] != ["explicit layout"]


# --------------------------------------------------------------------------------------------------- hcpeer report


def test_the_agent_reports_hcpeer_only_to_a_host_that_lists_the_feature():
    agent = make_agent(consent())
    agent.hcpeer_banner = HcpeerBanner(on=True, version=4)
    agent.host_features = [MANAGED_PEERS_FEATURE]
    old_host = agent.managed_heartbeat()
    assert old_host["managed"].hcpeer_mmap is None
    hb = HeartbeatRequest(worker_state=WorkerState.running, sent_at=NOW, **old_host)
    assert "hcpeer_mmap" not in hb.model_dump(mode="json", exclude=heartbeat_exclude(old_host))["managed"]
    agent.host_features = [MANAGED_PEERS_FEATURE, HCPEER_REPORT_FEATURE]
    new_host = agent.managed_heartbeat()
    hb = HeartbeatRequest(worker_state=WorkerState.running, sent_at=NOW, **new_host)
    assert hb.model_dump(mode="json", exclude=heartbeat_exclude(new_host))["managed"]["hcpeer_mmap"] is True
    agent.hcpeer_banner = None  # a stock library prints no hcpeer line
    assert agent.managed_heartbeat()["managed"].hcpeer_mmap is False
    assert heartbeat_exclude({}) == {"managed"}


def test_the_host_reads_the_reported_hcpeer_into_the_peer_facts():
    facts = peer_facts(mac(), remote_budgets(mac(), whole_contribution=True), report())
    assert facts.devices == ("MTL0", "CPU") and facts.node_budget == int(7.5 * GiB)
    assert facts.disk_budget == 110 * GiB and facts.caps[1].device == f"{EP_A}#1" and facts.hcpeer_on
    assert peer_facts(mac(), [], report()).node_budget == 0  # an unverified worker offers nothing


# --------------------------------------------------------------------------------------------------- server slots


def test_parallel_stored_in_engine_options(store, events, tmp_path):
    """PlanRequest.parallel lands in engine_options after selection, on the selector's plan and on an explicit
    layout's; the split is the one-slot plan's, and a default request stores no key."""
    mgr = manager(store, events, tmp_path, Nodes([mac()], {MAC_A: report()}))
    for layout in (None, GOOD):
        one = mgr.plan(PlanRequest(model_id=MODEL, layout=layout))
        four = mgr.plan(PlanRequest(model_id=MODEL, layout=layout, parallel=4))
        assert "parallel" not in one.engine_options and four.engine_options["parallel"] == 4
        assert four.assignments == one.assignments and four.device_list == one.device_list
        assert mgr.stored_plan(four.id).engine_options["parallel"] == 4  # what the deployment launches with -np
    for bad in (0, 17):
        with pytest.raises(ValidationError):
            PlanRequest(model_id=MODEL, parallel=bad)
    with pytest.raises(DeploymentError) as exc:
        mgr.plan(PlanRequest(model_id=MODEL, engine="vllm", parallel=4))
    assert exc.value.status == 422 and exc.value.code == "bad_request"
