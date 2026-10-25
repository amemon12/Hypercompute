"""A Mac as the HOST, simulated on any OS: a 16 GiB M4 Air runs `hypercompute host` and serves gpt-oss-120b MXFP4 with
two 16 GiB M4 Air peers over Iroh (tonight's topology). engine.host_os() is patched to "macos"; the host inventory comes
from agent.inventory.apple_devices with its probes faked, so CPU and MTL0 sit in one unified `uma` pool and MTL0's total
and free are what ggml-metal reports at b11770 (the Metal working-set limit, less this process's allocations).

Observed: Azim's 16 GiB M4 Air reported MTL0 at 12712935424 B total (12124 MiB, the Metal working set) through
`llama-server --list-devices` and through its rpc-server, at b11770 (results/capacity/20261022T070333Z-pair-g14-m4air-
relay/nodes-0700Z.json). 10922 MiB (2/3 of 16 GiB, what older macOS releases report) stays as the low case. Assumed, not
observed: the free RAM figures. The model shape is the real gpt-oss-120b one, read off the Windows run's plan
(results/capacity/20261022T085102Z-*/run.json): 36 layers of 1646.24 MiB, 1617.89 MiB of them routed experts, a
586.82 MiB head and token embeddings, 128 experts with 4 used. Nothing here touches hardware.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute import engine, gguf
from hypercompute.adapters.base import DeploymentSpec, ValidationResult
from hypercompute.adapters.llama_output import (
    check_moe_placement,
    check_participation,
    parse_list_devices,
    parse_overrides,
    parse_placement,
)
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.agent import inventory
from hypercompute.agent.contribution import default_contribution
from hypercompute.schemas import (
    EngineDevice,
    GgufSummary,
    HostInventory,
    NodeState,
    NodeView,
    Plan,
    PlanRequest,
    PlanStrategy,
    VerificationResult,
    WorkerEndpoint,
    utcnow,
)
from hypercompute.services.deployments import (
    VRAM_HEADROOM,
    DeploymentManager,
    cpu_budget,
    lease_budgets,
    local_budgets,
    remote_budgets,
    rpc_device_counts,
)
from hypercompute.services.leases import requests_from_plan
from hypercompute.services.moe_tier import gpu_cap, moe_offload_layout
from hypercompute.services.nodes import LOCAL_ID
from hypercompute.services.planner import (
    OVERHEAD_BY_KIND,
    RPC_STAGING,
    ModelShape,
    _assignment,
    _cpu_side,
    make_plan,
)
from hypercompute.units import GiB, MiB, fmt_bytes

WS16_MIB = 12124  # MTL0 working set Azim's 16 GiB M4 Air reported at b11770 (12712935424 B, nodes-0700Z.json)
WS_LOW_MIB = 10922  # 2/3 of 16 GiB, the low case
EP1, EP2 = "127.0.0.1:50101", "127.0.0.1:50102"  # the host's loopback Iroh bridges to the two peers
LAYER, EXPERTS, HEAD = 1646.24462890625 * MiB, 1617.890625 * MiB, 586.82373046875 * MiB
MODEL = "gpt-oss-120b-mxfp4"


# --------------------------------------------------------------------------------------------------- the machines


def mac_list_devices(ws_mib: int = WS16_MIB) -> str:
    """`llama-server --list-devices` on Apple silicon at b11770: MTL0 reports the working-set limit as total and free
    (nothing allocated yet); an Accelerate BLAS device, if the build has one, is not an accelerator the planner uses."""
    return (f"Available devices:\n  MTL0: Apple M4 ({ws_mib} MiB, {ws_mib} MiB free)\n"
            "  BLAS: Accelerate (0 MiB, 0 MiB free)\n")


def mac_engine(ws_mib: int = WS16_MIB):
    return engine_info(devices=parse_list_devices(mac_list_devices(ws_mib)), accelerator="metal")


def host_inventory(monkeypatch, *, ram_gib: float = 16, free_gib: float = 9.5, ws_mib: int = WS16_MIB
                   ) -> HostInventory:
    """agent.inventory.apple_devices with psutil, sysctl and system_profiler faked: the real Mac code path."""
    vm = SimpleNamespace(total=int(ram_gib * GiB), available=int(free_gib * GiB))
    monkeypatch.setattr(inventory.psutil, "virtual_memory", lambda: vm)
    probes = {"sysctl": str(int(ram_gib * GiB)),
              "system_profiler": json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M4", "sppci_cores": "10"}]})}
    monkeypatch.setattr(inventory, "_run", lambda args, timeout=20.0: probes.get(args[0]))
    eng = mac_engine(ws_mib)
    pools, devices = inventory.apple_devices(eng, "Apple M4")
    return make_inventory("macos", engine=eng).model_copy(update={"memory_pools": pools, "devices": devices})


def host_node(inv: HostInventory, **limits: float) -> NodeView:
    """The Mac host's own node with `hypercompute host` defaults (or explicit limits in GiB)."""
    contrib = default_contribution(inv) if not limits else contribution(["MTL0"], **limits)
    now = utcnow()
    return NodeView(id=LOCAL_ID, display_name="azim (this machine)", is_local=True, session_id=None,
                    state=NodeState.eligible, state_reason=None, joined_at=now, last_heartbeat_at=now,
                    heartbeat_age_s=0.0, inventory=inv, contribution=contrib, worker=None, worker_state=None,
                    verification=None)


def peer_node(ep: str, *, free_gib: float = 11.0, ws_mib: int = WS16_MIB) -> NodeView:
    """A 16 GiB M4 Air joined with --max-memory 10GiB --reserve 4GiB; its rpc-server reports MTL0's working set."""
    host, port = ep.rsplit(":", 1)
    inv = make_inventory("macos", engine=mac_engine(ws_mib), ram_total=16 * GiB, ram_free=int(free_gib * GiB))
    rpc = [EngineDevice(name="RPC0", description=ep, total_bytes=ws_mib * MiB, free_bytes=ws_mib * MiB)]
    now = utcnow()
    return NodeView(id=f"{port}" * 4 + "ab" * 6, display_name=f"air {port}", is_local=False, session_id="s",
                    state=NodeState.eligible, state_reason=None, joined_at=now, last_heartbeat_at=now,
                    heartbeat_age_s=1.0, inventory=inv, contribution=contribution(["MTL0"], max_gib=10, reserve_gib=4),
                    worker=WorkerEndpoint(host=host, port=int(port)), worker_state=None,
                    verification=VerificationResult(checked_at=now, reachable=True, engine_match=True,
                                                    rpc_devices=rpc, detail="fixture"))


def gpt_oss() -> ModelShape:
    s = GgufSummary(architecture="gpt-oss", name="gpt-oss-120b", file_type="MXFP4_MOE", n_layer=36, n_embd=2880,
                    n_head=64, n_head_kv=8, context_length=131072, vocab_size=201088, tensor_count=1,
                    total_tensor_bytes=0, layer_bytes=[int(LAYER)] * 36, token_embd_bytes=int(HEAD),
                    output_bytes=int(HEAD), output_tied=False, other_bytes=0, chat_template_present=True,
                    key_length=64, value_length=64, expert_count=128, expert_used_count=4,
                    layer_expert_bytes=[int(EXPERTS)] * 36)
    shape = ModelShape.from_gguf(s, 4096)
    assert shape is not None and shape.is_moe
    return shape


def plan_for(host: NodeView, peers: list[NodeView], strategy: PlanStrategy = PlanStrategy.all_selected) -> Plan:
    """What DeploymentManager.plan does for llama.cpp: budgets from the nodes, the planner, then llama.cpp's names."""
    free = next(p.free_bytes for p in host.inventory.memory_pools if p.id == "uma")
    plan = make_plan(PlanRequest(model_id=MODEL, strategy=strategy), gpt_oss(), local_budgets(host),
                     [b for n in peers for b in remote_budgets(n)], free, cpu_budget(host))
    return DeploymentManager._llama_names(plan, rpc_device_counts(peers))


# --------------------------------------------------------------------------------------------------- independent sums


def expected(free_gib: float, peer_layers: int, *, ws_mib: int = WS16_MIB, max_gib: float = 10) -> dict[str, int]:
    """The host split by hand, from the inputs only (no planner code): one pool, the reserve once, the Metal cap."""
    shape, reserve = gpt_oss(), 4 * GiB
    mtl_budget = min(int(max_gib * GiB), ws_mib * MiB - VRAM_HEADROOM, int(free_gib * GiB) - reserve)
    cpu_adm = int(free_gib * GiB) - reserve - 512 * MiB - shape.token_embd_bytes  # cpu_budget, then the embeddings
    cpu_private = shape.compute_est + OVERHEAD_BY_KIND["cpu"]
    host = 36 - peer_layers
    fixed = (host * shape.dense(0) + shape.output_bytes + host * shape.kv_per_layer + shape.compute_est
             + OVERHEAD_BY_KIND["metal"])
    cap = min(mtl_budget, cpu_adm - cpu_private)
    on_gpu = (cap - fixed) // shape.experts(0)
    return {"mtl_budget": mtl_budget, "cpu_adm": cpu_adm, "cpu_private": cpu_private, "host": host,
            "gpu_experts": on_gpu, "metal_total": fixed + on_gpu * shape.experts(0)}


def peer_whole_layers(budget: int) -> int:
    shape = gpt_oss()
    fixed = shape.compute_est + OVERHEAD_BY_KIND["metal"] + RPC_STAGING
    return int((budget - fixed) // (shape.layer_bytes[0] + shape.kv_per_layer))


# --------------------------------------------------------------------------------------------------- inventory, budgets


def test_the_mac_inventory_puts_cpu_and_mtl0_in_one_uma_pool_with_the_metal_working_set(monkeypatch):
    inv = host_inventory(monkeypatch)
    [pool] = inv.memory_pools
    assert (pool.id, pool.kind, pool.total_bytes, pool.free_bytes) == ("uma", "unified", 16 * GiB, int(9.5 * GiB))
    cpu, mtl = inv.devices
    assert (cpu.id, cpu.pool_id, mtl.id, mtl.pool_id, mtl.backend) == ("CPU", "uma", "MTL0", "uma", "metal")
    assert mtl.total_bytes == mtl.free_bytes == WS16_MIB * MiB  # the working-set limit, not RAM
    assert [d.name for d in inv.engine.devices] == ["MTL0", "BLAS"]
    limits = default_contribution(inv)  # `hypercompute host` on a 16 GiB Mac
    assert (limits.devices, limits.max_bytes, limits.reserve_bytes) == (["MTL0"], 10 * GiB, 4 * GiB)


@pytest.mark.parametrize("ram, free, ws, max_gib, want, basis", [
    (16, 9.5, WS16_MIB, None, int(5.5 * GiB), "free unified memory now minus 4096 MiB reserve"),  # RAM binds
    (16, 6.0, WS16_MIB, None, 2 * GiB, "free unified memory now minus 4096 MiB reserve"),  # old: 6.67 GiB
    (16, 3.0, WS16_MIB, None, 0, "free unified memory now minus 4096 MiB reserve"),  # under the reserve
    (32, 28.0, 21845, 30, 21845 * MiB - VRAM_HEADROOM, "MTL0 working-set limit minus 512 MiB headroom"),
    (32, 28.0, 21845, 12, 12 * GiB, "contribution limit"),
])
def test_the_metal_budget_is_the_smallest_of_ram_after_the_reserve_the_working_set_and_the_cap(
        monkeypatch, ram, free, ws, max_gib, want, basis):
    inv = host_inventory(monkeypatch, ram_gib=ram, free_gib=free, ws_mib=ws)
    node = host_node(inv) if max_gib is None else host_node(inv, max_gib=max_gib, reserve_gib=4)
    [mtl] = local_budgets(node)  # BLAS is not an accelerator the planner places on
    assert (mtl.device, mtl.pool_id, mtl.pool_kind) == ("MTL0", "uma", "unified")
    assert mtl.budget_bytes == want and mtl.budget_basis == basis
    cpu = cpu_budget(node)  # the CPU's figure is unchanged: free minus the reserve minus 512 MiB, same pool
    assert cpu is not None and cpu.pool_id == "uma"
    assert cpu.budget_bytes == max(0, int(free * GiB) - 4 * GiB - 512 * MiB)


def test_a_discrete_gpu_keeps_its_own_budget_in_the_moe_tier():
    from test_split_order import cpu, gpu

    cpu_side = _cpu_side(gpt_oss(), cpu(1), 0, True)  # far less host RAM than the GPU budget: it must not matter
    assert cpu_side.budget_bytes < gpu(10.27).budget_bytes == gpu_cap(gpu(10.27), cpu_side)


# --------------------------------------------------------------------------------------------------- the 120B plan


@pytest.mark.parametrize("ws_mib, per_peer", [(WS16_MIB, 4), (WS_LOW_MIB, 3)])
@pytest.mark.parametrize("free_gib", [9.0, 9.5, 10.0])
def test_gpt_oss_on_a_16gib_mac_host_with_two_16gib_mac_peers(monkeypatch, free_gib, ws_mib, per_peer):
    host = host_node(host_inventory(monkeypatch, free_gib=free_gib, ws_mib=ws_mib))
    peers = [peer_node(EP1, ws_mib=ws_mib), peer_node(EP2, ws_mib=ws_mib)]  # 11 GiB free each
    plan = plan_for(host, peers)
    assert plan.feasible and plan.tier == "moe_offload" and plan.moe is not None, plan.reasons
    cpu_a, p1, p2, mtl = plan.assignments
    assert [a.device for a in (cpu_a, p1, p2, mtl)] == ["CPU", "RPC0", "RPC1", "MTL0"]
    assert plan.device_list == ["RPC0", "RPC1", "MTL0"] and plan.rpc_endpoints == [EP1, EP2]
    shape = gpt_oss()
    # each peer: whole layers (dense part and experts), as many as its admissible bytes hold: the smaller of its free
    # RAM and its Metal working set, minus its 4 GiB reserve, within its 10 GiB cap
    for peer, budget in zip((p1, p2), remote_budgets(peers[0]) + remote_budgets(peers[1]), strict=True):
        assert peer.weight_bytes == sum(shape.layer_bytes[i] for i in peer.layers) and peer.fits
        assert peer.total_bytes_est <= budget.budget_bytes == min(10 * GiB, min(ws_mib * MiB, 11 * GiB) - 4 * GiB)
        assert peer.total_bytes_est <= ws_mib * MiB - VRAM_HEADROOM  # under the peer's own Metal working set
        assert len(peer.layers) == peer_whole_layers(budget.budget_bytes) == per_peer
    assert p1.layers == list(range(per_peer)) and p2.layers == list(range(per_peer, 2 * per_peer))
    host_layers = 36 - 2 * per_peer
    assert plan.tensor_split == [per_peer, per_peer, host_layers + 1]
    want = expected(free_gib, peer_layers=2 * per_peer, ws_mib=ws_mib)
    assert mtl.layers == list(range(2 * per_peer, 36)) and mtl.holds_output and mtl.budget_bytes == want["mtl_budget"]
    # the Metal part: under the working-set cap and inside the pool, with the experts that fit
    assert mtl.total_bytes_est == want["metal_total"] <= ws_mib * MiB - VRAM_HEADROOM
    assert plan.moe.gpu_expert_bytes == want["gpu_experts"] * shape.experts(0)
    assert len(plan.moe.cpu_expert_layers) == want["host"] - want["gpu_experts"]
    assert set(plan.moe.cpu_expert_layers) < set(mtl.layers)  # the rest stay file-backed on the host CPU
    assert plan.moe.cpu_expert_bytes == cpu_a.weight_bytes == len(plan.moe.cpu_expert_layers) * shape.experts(0)
    assert plan.moe.override == gguf.expert_override_pattern(plan.moe.cpu_expert_layers) + "=CPU"
    assert cpu_a.budget_bytes == want["cpu_adm"] and cpu_a.total_bytes_est == want["cpu_private"]


def test_one_pool_is_counted_once_and_the_ssd_estimate_follows_from_it(monkeypatch):
    free = 9.5
    plan = plan_for(host_node(host_inventory(monkeypatch, free_gib=free)), [peer_node(EP1), peer_node(EP2)])
    cpu_a, mtl, moe = plan.assignments[0], plan.assignments[-1], plan.moe
    assert moe is not None
    # Metal part + CPU buffers + file cache + token embeddings + 512 MiB safety = free RAM minus the reserve, exactly
    assert mtl.total_bytes_est + cpu_a.total_bytes_est + moe.page_cache_bytes_est == cpu_a.budget_bytes
    assert cpu_a.budget_bytes + gpt_oss().token_embd_bytes + 512 * MiB == int(free * GiB) - 4 * GiB
    assert 0 < moe.page_cache_bytes_est < GiB  # the GPU took what the pool allowed; little is left for the cache
    miss = 1 - moe.page_cache_bytes_est / moe.cpu_expert_bytes
    assert moe.miss_rate_est == pytest.approx(miss)
    assert moe.ssd_bytes_per_token_est == int(4 / 128 * moe.cpu_expert_bytes * miss) > GiB  # an estimate, labelled
    assert plan.capacity_mode and moe.provenance == "estimate"
    text = " ".join(plan.warnings)
    assert "share the uma memory pool, counted once" in text and "not a measurement" in text


def test_the_old_metal_budget_would_have_overcommitted_the_pool(monkeypatch):
    """Before: MTL0 = min(cap, working set - reserve) whatever RAM was free, and the GPU experts were sized against it
    alone, so the pool check refused the tier (or, with RAM low, the GPU budget exceeded free RAM)."""
    node = host_node(host_inventory(monkeypatch, free_gib=6.0))
    old = min(10 * GiB, WS16_MIB * MiB - 4 * GiB)
    [mtl] = local_budgets(node)
    assert mtl.budget_bytes == 2 * GiB < old and mtl.budget_bytes <= 6 * GiB - 4 * GiB


def test_the_mac_host_alone_also_plans_the_moe_tier_inside_its_pool(monkeypatch):
    host = host_node(host_inventory(monkeypatch, free_gib=9.5))
    plan = plan_for(host, [], PlanStrategy.local_only)
    assert plan.feasible and plan.tier == "moe_offload" and plan.device_list == ["MTL0"] and plan.tensor_split == [37]
    cpu_a, mtl = plan.assignments
    want = expected(9.5, peer_layers=0)
    assert mtl.total_bytes_est == want["metal_total"] and mtl.total_bytes_est + cpu_a.total_bytes_est <= (
        cpu_a.budget_bytes)


def test_a_mac_with_plenty_of_ram_is_held_to_the_metal_working_set(monkeypatch):
    host = host_node(host_inventory(monkeypatch, ram_gib=32, free_gib=28, ws_mib=21845), max_gib=30, reserve_gib=4)
    plan = plan_for(host, [], PlanStrategy.local_only)
    cpu_a, mtl = plan.assignments
    assert plan.feasible and plan.tier == "moe_offload" and mtl.budget_bytes == 21845 * MiB - VRAM_HEADROOM
    assert mtl.total_bytes_est <= mtl.budget_bytes and plan.moe is not None
    assert mtl.budget_bytes < cpu_a.budget_bytes - cpu_a.total_bytes_est  # the Metal cap binds here, not the pool
    assert plan.moe.page_cache_bytes_est == cpu_a.budget_bytes - cpu_a.total_bytes_est - mtl.total_bytes_est > 0


def test_the_leases_admit_the_mac_host_plan(monkeypatch):
    host, peers = host_node(host_inventory(monkeypatch)), [peer_node(EP1), peer_node(EP2)]
    plan = plan_for(host, peers)
    registry = SimpleNamespace(get=lambda node_id: next((n for n in [host, *peers] if n.id == node_id), None))
    reqs = requests_from_plan(plan, lease_budgets(plan, registry))  # type: ignore[arg-type]
    uma = next(r for r in reqs if (r.node_id, r.pool_id) == (LOCAL_ID, "uma"))
    assert uma.bytes == plan.assignments[0].total_bytes_est + plan.assignments[-1].total_bytes_est  # one request
    assert all(r.bytes <= r.pool_budget and r.bytes <= r.node_budget for r in reqs)


def _moe_reasons(plan: Plan) -> list[str]:
    return [c.reason for c in plan.candidates if "MoE experts on the host CPU" in c.label]


@pytest.mark.parametrize("free_gib, pool_left", [(7.0, "1.7 GiB"), (4.8, "0 B")])
def test_a_host_short_of_free_memory_names_the_shared_pool_never_a_negative_size(monkeypatch, free_gib, pool_left):
    """Regression: the refusal named MTL0's capacity as the shared pool's remainder, and below 5 GiB free it printed a
    negative byte count ("of -214958080 B"). It now names the uma pool, never goes below zero, and says what to do."""
    host = host_node(host_inventory(monkeypatch, free_gib=free_gib))
    plan = plan_for(host, [], PlanStrategy.local_only)
    [why] = _moe_reasons(plan)
    assert not plan.feasible and f"of the {pool_left} its uma memory pool leaves" in why, why
    assert "of -" not in why and why.endswith("free memory on this machine, then plan again")


def test_a_discrete_gpu_keeps_its_dense_overflow_message_byte_for_byte():
    from test_split_order import cpu, gpu

    shape, budget = gpt_oss(), gpu(1.0)
    dense = _assignment(shape, budget, list(range(36)), True, kept_off=sum(shape.experts(i) for i in range(36)))
    _, _, ok, why = moe_offload_layout(shape, [budget], cpu(24), keep_all=False)
    assert not ok and why == (f"CUDA0 cannot hold the dense part, head and KV of 36 layer(s): "
                              f"~{fmt_bytes(dense.total_bytes_est)} of {fmt_bytes(budget.budget_bytes)}")


def test_tonights_operating_floor_with_the_observed_working_set(monkeypatch):
    """What the three Airs need free (psutil available) for the 120B plan, at the observed 12124 MiB working set and
    the 4 GiB reserve: the host about 7.3 GiB with two peers at 11 GiB; a peer under the reserve plus one whole
    layer (about 6.1 GiB) cannot take part, so all_selected is refused and names it."""
    peers = [peer_node(EP1), peer_node(EP2)]
    assert plan_for(host_node(host_inventory(monkeypatch, free_gib=7.5)), peers).feasible
    short = plan_for(host_node(host_inventory(monkeypatch, free_gib=7.0)), peers)
    assert not short.feasible and "uma memory pool leaves" in " ".join(_moe_reasons(short))
    low_peer = [peer_node(EP1, free_gib=6.0), peer_node(EP2)]
    refused = plan_for(host_node(host_inventory(monkeypatch, free_gib=9.5)), low_peer)
    assert not refused.feasible and any("cannot hold one whole layer" in r for r in _moe_reasons(refused))


def test_a_dense_model_on_a_mac_host_keeps_metal_and_cpu_layers_in_one_pool(monkeypatch):
    """gpu_ram on a Mac exists to get past the Metal working set, not past RAM: the CPU layers and the Metal part are
    checked together against the one pool."""
    from test_split_order import summary

    host = host_node(host_inventory(monkeypatch, ram_gib=32, free_gib=28, ws_mib=21845), max_gib=30, reserve_gib=4)
    shape = ModelShape.from_gguf(summary(40, 500, 5120, 40, 8, 500), 4096)
    plan = make_plan(PlanRequest(model_id="d", strategy=PlanStrategy.local_only), shape, local_budgets(host), [],
                     28 * GiB, cpu_budget(host))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.device_list == ["MTL0"], plan.reasons
    cpu_a, mtl = plan.assignments
    assert mtl.total_bytes_est <= 21845 * MiB - VRAM_HEADROOM
    assert cpu_a.total_bytes_est + mtl.total_bytes_est <= cpu_a.budget_bytes


# --------------------------------------------------------------------------------------------------- llama-server args


def args_on(monkeypatch, os_name: str, plan: Plan) -> list[str]:
    monkeypatch.setattr(engine, "host_os", lambda: os_name)
    adapter = object.__new__(LlamaRpcAdapter)  # build_args reads only the spec
    return adapter.build_args(DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m"), 9000)


def flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def test_build_args_on_a_mac_host_pass_the_moe_flags_with_metal_and_rpc_names(monkeypatch):
    plan = plan_for(host_node(host_inventory(monkeypatch)), [peer_node(EP1), peer_node(EP2)])
    args = args_on(monkeypatch, "macos", plan)
    assert flag(args, "--load-mode") == "mmap" and args.count("--load-mode") == 1  # never the Windows "none"
    assert {"--no-repack", "--no-host", "--no-op-offload"} <= set(args) and "--n-cpu-moe" not in args
    assert flag(args, "--cache-ram") == "0" and flag(args, "-ot") == plan.moe.override  # type: ignore[union-attr]
    assert flag(args, "-ngl") == "all" and flag(args, "--fit") == "off"
    assert flag(args, "--device") == "RPC0,RPC1,MTL0" and flag(args, "--rpc") == f"{EP1},{EP2}"
    assert flag(args, "--tensor-split") == "4,4,29" and flag(args, "--split-mode") == "layer"
    assert args == args_on(monkeypatch, "windows", plan)  # the MoE tier passes the same flags on both systems


def test_the_windows_ram_tier_flags_stay_off_a_mac_host(monkeypatch):
    from test_split_order import summary

    host = host_node(host_inventory(monkeypatch, ram_gib=32, free_gib=28, ws_mib=21845), max_gib=30, reserve_gib=4)
    plan = make_plan(PlanRequest(model_id="d", strategy=PlanStrategy.local_only),
                     ModelShape.from_gguf(summary(40, 500, 5120, 40, 8, 500), 4096), local_budgets(host), [],
                     28 * GiB, cpu_budget(host))
    assert plan.tier == "gpu_ram"
    mac, win = args_on(monkeypatch, "macos", plan), args_on(monkeypatch, "windows", plan)
    assert "--load-mode" not in mac and "--no-host" not in mac and "--no-repack" not in mac
    assert flag(mac, "--cache-ram") == "0" and flag(mac, "-ngl") == str(plan.n_gpu_units)
    assert flag(win, "--load-mode") == "none" and {"--no-host", "--no-repack"} <= set(win)


# --------------------------------------------------------------------------------------------------- load verification


def _line(t: str, prefix: str, name: str, kind: str, mib: float) -> str:
    """One b11770 buffer line: load_tensors pads the name to 12 (%12s), llama_kv_cache and sched_reserve to 10."""
    width = 12 if prefix == "load_tensors" else 10
    return f"0.{t} I {prefix}: {name:>{width}} {kind} buffer size = {mib:8.2f} MiB"


def mac_host_log(plan: Plan, *, stray_expert_layer: int | None = None, mapped_gpu: bool = True) -> str:
    """The -lv 4 load log of a Mac host for `plan`, in b11770's formats. Metal names (ggml-metal.cpp): model buffers
    wrapped from the mapped GGUF are MTL0_Mapped (one per contiguous range: llama-model.cpp maps each range of the
    context's own tensors), KV and compute buffers MTL0. The CPU experts are CPU_Mapped in the two ranges per layer the
    Windows 120B log shows (1078.59 + 539.30 MiB). gpt-oss logs two KV buffers per device (full and sliding window)."""
    shape, moe = gpt_oss(), plan.moe
    assert moe is not None
    mtl = plan.assignments[-1]
    on_cpu = [i for i in moe.cpu_expert_layers if i != stray_expert_layer]
    gpu_name = "MTL0_Mapped" if mapped_gpu else "MTL0_Private"
    lines = ["0.01.147.276 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
             "0.01.150.000 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using "
             "--load-mode none for better performance",
             "0.04.186.042 I load_tensors: offloading output layer to GPU",
             "0.04.186.051 I load_tensors: offloading 35 repeating layers to GPU",
             "0.04.186.052 I load_tensors: offloaded 37/37 layers to GPU",
             _line("04.186.060", "load_tensors", "CPU_Mapped", "model", shape.token_embd_bytes / MiB)]
    for i in on_cpu:
        lines += [_line("04.186.061", "load_tensors", "CPU_Mapped", "model", 1078.59),
                  _line("04.186.062", "load_tensors", "CPU_Mapped", "model", 539.30)]
    for i in mtl.layers:
        size = shape.dense(i) + (0 if i in on_cpu else shape.experts(i))
        lines.append(_line("04.186.063", "load_tensors", gpu_name, "model", size / MiB))
    lines.append(_line("04.186.064", "load_tensors", gpu_name, "model", shape.output_bytes / MiB))
    for a in plan.assignments[1:-1]:
        lines.append(_line("04.186.065", "load_tensors", f"RPC0[{a.endpoint}]", "model", a.weight_bytes / MiB))
    lines += [_line("09.633.530", "llama_kv_cache", "MTL0", "KV", 120.0),
              _line("09.633.531", "llama_kv_cache", "MTL0", "KV", 60.0),
              *(_line("09.633.532", "llama_kv_cache", f"RPC0[{a.endpoint}]", "KV", 12.0)
                for a in plan.assignments[1:-1]),
              _line("09.633.539", "sched_reserve", "MTL0", "compute", 398.38),
              *(_line("09.633.540", "sched_reserve", f"RPC0[{a.endpoint}]", "compute", 101.13)
                for a in plan.assignments[1:-1]),
              _line("09.633.541", "sched_reserve", "CPU", "compute", 18.83)]
    return "\n".join(lines)


def test_a_mac_host_load_log_with_mtl0_mapped_is_verified(monkeypatch):
    plan = plan_for(host_node(host_inventory(monkeypatch)), [peer_node(EP1), peer_node(EP2)])
    text = mac_host_log(plan)
    placement = parse_placement(text)
    names = {d.name for d in placement.devices}
    assert {"CPU_Mapped", "MTL0_Mapped", "MTL0", f"RPC0[{EP1}]", f"RPC0[{EP2}]", "CPU"} == names
    ok, detail = check_moe_placement(plan, placement, parse_overrides(text))
    assert ok and detail.startswith("moe_offload verified: MTL0 model"), detail
    ok, detail = check_participation(plan, placement, {EP1: ["RPC0"], EP2: ["RPC1"]})
    mapped = sum(d.model_bytes for d in placement.devices if d.name == "MTL0_Mapped")
    assert ok and f"MTL0: model {mapped / MiB:.0f} MiB" in detail and f"RPC0[{EP2}]" in detail, detail


def test_a_mac_host_log_with_an_expert_layer_left_on_metal_is_not_verified(monkeypatch):
    plan = plan_for(host_node(host_inventory(monkeypatch)), [peer_node(EP1), peer_node(EP2)])
    stray = plan.moe.cpu_expert_layers[0]  # type: ignore[union-attr]
    text = mac_host_log(plan, stray_expert_layer=stray)
    ok, detail = check_moe_placement(plan, parse_placement(text), parse_overrides(text))
    assert not ok and "MTL0 holds" in detail and "CPU_Mapped spans" in detail


def test_a_mac_host_log_without_metal_weights_is_not_verified(monkeypatch):
    plan = plan_for(host_node(host_inventory(monkeypatch)), [], PlanStrategy.local_only)
    text = "\n".join(line for line in mac_host_log(plan).splitlines() if "MTL0_Mapped" not in line)
    ok, detail = check_moe_placement(plan, parse_placement(text), parse_overrides(text))
    assert not ok and "MTL0 holds 0 MiB of weights" in detail


def test_the_private_metal_buffer_name_also_counts_as_the_gpu(monkeypatch):
    plan = plan_for(host_node(host_inventory(monkeypatch)), [], PlanStrategy.local_only)
    text = mac_host_log(plan, mapped_gpu=False)  # MTL0_Private: what a non-mapped Metal load would log
    ok, detail = check_moe_placement(plan, parse_placement(text), parse_overrides(text))
    assert ok, detail


def test_observe_verifies_the_mac_host_deployment_with_both_peers(monkeypatch):
    plan = plan_for(host_node(host_inventory(monkeypatch)), [peer_node(EP1), peer_node(EP2)])
    text = mac_host_log(plan)
    adapter = object.__new__(LlamaRpcAdapter)
    monkeypatch.setattr(adapter, "read_log", lambda _h: text, raising=False)
    spec = DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m")
    report = asyncio.run(adapter.observe(None, spec, ValidationResult(  # type: ignore[arg-type]
        ok=True, problems=[], rpc_device_map={EP1: ["RPC0"], EP2: ["RPC1"]})))
    assert report.verified and "moe_offload verified" in report.detail, report.detail
