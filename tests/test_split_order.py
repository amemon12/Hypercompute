"""Device order (head on the local GPU), fill order, the overflow cost prior, the gpu_disk tier and engine flags."""

import random
from pathlib import Path

import pytest
from hypercompute.adapters.base import DeploymentSpec, EngineAdapterError
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.schemas import DeviceAssignment, DeviceBudget, GgufSummary, PlanRequest, PlanStrategy
from hypercompute.services import planner
from hypercompute.services.deployments import DeploymentManager
from hypercompute.services.planner import (
    DECODE_MS_PER_GB,
    HOP_MS,
    ModelShape,
    decode_ms,
    make_plan,
    order_devices,
    ram_tier_layout,
)
from hypercompute.units import GiB, MiB

EP5, EP6 = "10.0.0.5:47052", "10.0.0.6:47052"


def summary(n_layer: int, layer_mib: float, n_embd: int, n_head: int, n_head_kv: int, head_mib: int) -> GgufSummary:
    return GgufSummary(architecture="qwen2", name="m", file_type="Q6_K", n_layer=n_layer, n_embd=n_embd,
                       n_head=n_head, n_head_kv=n_head_kv, context_length=32768, vocab_size=152064, tensor_count=1,
                       total_tensor_bytes=int(n_layer * layer_mib * MiB), layer_bytes=[int(layer_mib * MiB)] * n_layer,
                       token_embd_bytes=head_mib * MiB, output_bytes=head_mib * MiB, output_tied=False, other_bytes=0,
                       chat_template_present=True)


def shape14() -> ModelShape:  # qwen2.5-14b q6_k: 48 x 225 MiB layers, 609 MiB head, 16 MiB KV per layer at 4096
    return ModelShape.from_gguf(summary(48, 225, 5120, 40, 8, 609), 4096)


def shape05() -> ModelShape:  # qwen2.5-0.5b q4_k_m
    return ModelShape.from_gguf(summary(24, 10.5, 896, 14, 2, 89), 4096)


def gpu(gib: float) -> DeviceBudget:
    return DeviceBudget(node_id="local", device="CUDA0", description="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                        endpoint=None, pool_id="vram:CUDA0", pool_kind="discrete_vram", budget_bytes=int(gib * GiB),
                        budget_basis="test")


def mac(gib: float, ep: str = EP5) -> DeviceBudget:
    return DeviceBudget(node_id=f"mac-{ep}", device=f"{ep}#0", description=f"metal Apple M4 (10-core GPU) via {ep}",
                        endpoint=ep, pool_id="uma", pool_kind="unified", budget_bytes=int(gib * GiB),
                        budget_basis="test")


def rtx_peer(gib: float, ep: str = EP6) -> DeviceBudget:
    return DeviceBudget(node_id=f"pc-{ep}", device=f"{ep}#0", description=f"cuda NVIDIA GeForce RTX 4090 via {ep}",
                        endpoint=ep, pool_id="vram:CUDA0", pool_kind="discrete_vram", budget_bytes=int(gib * GiB),
                        budget_basis="test")


def cpu(gib: float) -> DeviceBudget:
    return DeviceBudget(node_id="local", device="CPU", description="host CPU (system RAM)", endpoint=None,
                        pool_id="ram", pool_kind="system_ram", budget_bytes=int(gib * GiB), budget_basis="test")


def req(strategy: PlanStrategy = PlanStrategy.auto, **kw) -> PlanRequest:
    return PlanRequest(model_id="m", strategy=strategy, **kw)


def args_for(plan) -> list[str]:
    adapter = object.__new__(LlamaRpcAdapter)  # build_args reads only the spec; no engine, process or client needed
    return adapter.build_args(DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m"), 9000)


# ----------------------------------------------------------------------------- device order


def test_order_devices_puts_peers_first_by_ascending_budget_and_the_local_gpu_last():
    small, big = mac(4, EP5), rtx_peer(12, EP6)
    assert [d.device for d in order_devices([gpu(10)], [big, small])] == [small.device, big.device, "CUDA0"]


def test_the_local_gpu_holds_the_head_so_logits_never_cross_the_network():
    plan = make_plan(req(PlanStrategy.fastest_fit), shape14(), [gpu(6)], [mac(12)], 32 * GiB)
    assert plan.feasible and plan.tier == "gpu"
    peer, local = plan.assignments
    assert peer.endpoint == EP5 and peer.layers[0] == 0 and not peer.holds_output
    assert local.device == "CUDA0" and local.holds_output and local.layers[-1] == 47
    assert len(local.layers) == 20  # filled to its capacity first; the peer gets only the overflow
    assert plan.tensor_split == [len(peer.layers), len(local.layers) + 1]
    named = DeploymentManager._llama_names(plan, {EP5: 1})
    args = args_for(named)
    assert named.device_list == ["RPC0", "CUDA0"] and args[args.index("--device") + 1] == "RPC0,CUDA0"
    assert args.index("--rpc") < args.index("--device")


def test_a_gpu_too_small_for_the_head_leaves_it_on_a_peer_and_says_so():
    plan = make_plan(req(PlanStrategy.fastest_fit), shape14(), [gpu(1.0)], [rtx_peer(16)], 32 * GiB)
    assert plan.feasible and plan.assignments[0].device == "CUDA0" and not plan.assignments[0].holds_output
    assert plan.assignments[-1].endpoint == EP6 and plan.assignments[-1].holds_output
    assert any("cannot hold the output head" in w for w in plan.warnings)


# ----------------------------------------------------------------------------- fill order


@pytest.mark.parametrize("strategy", [PlanStrategy.fastest_fit, PlanStrategy.auto])
def test_interactive_fill_keeps_peers_empty_when_the_local_gpu_fits(strategy):
    plan = make_plan(req(strategy), shape05(), [gpu(10)], [mac(12), rtx_peer(20)], 16 * GiB, cpu(8))
    assert plan.feasible and plan.tier == "gpu" and plan.device_list == ["CUDA0"] and plan.rpc_endpoints == []


def test_run_together_fills_the_local_gpu_first_and_gives_each_peer_the_minimum():
    plan = make_plan(req(PlanStrategy.all_selected), shape05(), [gpu(10)], [mac(12, EP5), mac(8, EP6)], 16 * GiB)
    assert plan.feasible and [len(a.layers) for a in plan.assignments] == [1, 1, 22]
    assert [a.endpoint for a in plan.assignments] == [EP6, EP5, None] and plan.assignments[-1].holds_output


def test_throughput_splits_by_capacity_while_interactive_fills_the_local_gpu_first():
    shape, peer = shape14(), rtx_peer(12)
    inter = make_plan(req(PlanStrategy.fastest_fit), shape, [gpu(6)], [peer], 32 * GiB)
    thr = make_plan(req(PlanStrategy.fastest_fit, objective="throughput"), shape, [gpu(6)], [peer], 32 * GiB)
    assert inter.feasible and thr.feasible
    assert inter.assignments[-1].device == thr.assignments[-1].device == "CUDA0"  # the head stays local either way
    assert len(inter.assignments[-1].layers) == 20
    assert [len(a.layers) for a in thr.assignments] == planner._counts_proportional(shape, [peer, gpu(6)])
    assert len(thr.assignments[-1].layers) < 20 and any("objective: throughput" in r for r in thr.reasons)


def test_run_together_for_throughput_shares_by_capacity_even_when_local_fits():
    plan = make_plan(req(PlanStrategy.all_selected, objective="throughput"), shape05(), [gpu(10)], [mac(12)],
                     16 * GiB)
    assert plan.feasible and len(plan.assignments[0].layers) > 1 and sum(plan.tensor_split) == 25


def test_rebalance_checks_room_for_the_layer_that_moves_not_layer_0():
    """Layers 0 and 37 are 600 MiB, the rest 150. The avg-based fill gives CUDA0 layer 37 and overflows it; the
    layer that then moves to the Mac is a 150 MiB one, which fits there, while layer 0 + KV would not."""
    lb = (600 * MiB, *[150 * MiB] * 36, 600 * MiB)
    shape = ModelShape(n_layer=38, layer_bytes=lb, output_bytes=600 * MiB, token_embd_bytes=600 * MiB,
                       kv_per_layer=16 * MiB, compute_est=176 * MiB)
    plan = make_plan(req(PlanStrategy.fastest_fit, allow_ram_tier=False), shape, [gpu(3.28)], [mac(5.55)], 32 * GiB)
    assert plan.feasible and plan.tier == "gpu" and plan.tensor_split == [28, 11]


def test_rebalance_finds_a_two_device_split_whenever_one_fits():
    """Brute force over every boundary on random non-uniform shapes: the planner may never miss a split that fits."""
    rng = random.Random(1)
    for _ in range(1000):
        n, base = rng.randint(20, 60), rng.randint(80, 300)
        lb = tuple(int(base * rng.choice([1, 1, 1, 1.2, 1.5, 2, 4]) * MiB) for _ in range(n))
        shape = ModelShape(n_layer=n, layer_bytes=lb, output_bytes=rng.randint(100, 900) * MiB, token_embd_bytes=0,
                           kv_per_layer=rng.randint(2, 40) * MiB, compute_est=176 * MiB)
        devs = order_devices([gpu(rng.uniform(2, 10))], [mac(rng.uniform(2, 16))])
        some = any(all(a.fits for a in planner._layout(shape, devs, [c, n - c])) for c in range(n + 1))
        assert all(a.fits for a in planner._evaluate(shape, devs, keep_all=False)) == some, (n, lb[:3])


# ----------------------------------------------------------------------------- overflow: peers vs host RAM


def test_decode_prior_reproduces_the_research_break_even():
    """X GB of overflow costs X*c_cpu on the host CPU and X*c_mac + H on a Mac: equal near 2.4 GB (§3.4)."""
    def one(device: str, endpoint: str | None, gb: float) -> list[DeviceAssignment]:
        return [DeviceAssignment(node_id="n", device=device, endpoint=endpoint, pool_id="p", layers=[0],
                                 holds_output=False, weight_bytes=int(gb * 1e9), kv_bytes=0, compute_bytes_est=0,
                                 overhead_bytes_est=0, total_bytes_est=0, budget_bytes=1, fits=True)]
    host, peer = [cpu(1).model_copy(update={"node_id": "n"})], [mac(1).model_copy(update={"node_id": "n",
                                                                                          "device": "M"})]
    x_star = HOP_MS / (DECODE_MS_PER_GB["cpu"] - DECODE_MS_PER_GB["metal"])
    assert x_star == pytest.approx(2.42, abs=0.01)
    assert decode_ms(one("CPU", None, 1.0), host) < decode_ms(one("M", EP5, 1.0), peer)
    assert decode_ms(one("CPU", None, 4.0), host) > decode_ms(one("M", EP5, 4.0), peer)


def test_a_small_overflow_goes_to_host_ram():
    plan = make_plan(req(), shape14(), [gpu(11.5)], [mac(12)], 32 * GiB, cpu(16))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.rpc_endpoints == [] and plan.cpu_layers > 0
    by = {c.label: c for c in plan.candidates}
    assert by["overflow to 1 peer device(s)"].feasible and by["overflow to host RAM"].feasible
    assert any(w.startswith("overflow goes to host RAM") for w in plan.warnings)


def test_a_large_overflow_goes_to_the_peer():
    plan = make_plan(req(), shape14(), [gpu(6)], [mac(12)], 32 * GiB, cpu(16))
    assert plan.feasible and plan.tier == "gpu" and plan.cpu_layers == 0 and plan.rpc_endpoints == [EP5]
    assert plan.assignments[-1].device == "CUDA0" and plan.assignments[-1].holds_output
    assert any(w.startswith("overflow goes to the peers") for w in plan.warnings)
    assert any(c.label == "overflow to host RAM" and c.feasible for c in plan.candidates)


def test_the_overflow_tier_can_be_pinned_to_build_the_tuners_alternative():
    to_peer = make_plan(req(), shape14(), [gpu(11.5)], [mac(12)], 32 * GiB, cpu(16), overflow="peers")
    assert to_peer.feasible and to_peer.tier == "gpu" and to_peer.rpc_endpoints == [EP5]
    to_ram = make_plan(req(), shape14(), [gpu(6)], [mac(12)], 32 * GiB, cpu(16), overflow="host_ram")
    assert to_ram.feasible and to_ram.tier == "gpu_ram" and to_ram.rpc_endpoints == []


def test_peers_and_host_ram_together_when_neither_alone_fits():
    plan = make_plan(req(), shape14(), [gpu(6)], [mac(4)], 32 * GiB, cpu(5))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.rpc_endpoints == [EP5]
    assert [a.device for a in plan.assignments][0] == "CPU" and plan.assignments[-1].device == "CUDA0"
    named = DeploymentManager._llama_names(plan, {EP5: 1})  # the CPU tier is -ngl, never a --device entry
    assert named.device_list == ["RPC0", "CUDA0"] and sum(named.tensor_split) == named.n_gpu_units
    args = args_for(named)
    assert args[args.index("-ngl") + 1] == str(named.n_gpu_units) and "CPU" not in args[args.index("--device") + 1]


# ----------------------------------------------------------------------------- gpu_disk


def test_a_cpu_part_larger_than_free_ram_pages_from_disk_in_capacity_mode():
    [_cpu_a, *_], ok = ram_tier_layout(shape14(), [gpu(6)], cpu(64), keep_all=False)  # CPU layers the GPU needs
    plan = make_plan(req(PlanStrategy.local_only), shape14(), [gpu(6)], [], 4 * GiB, cpu(3))
    assert ok and plan.feasible and plan.tier == "gpu_disk" and plan.capacity_mode
    host = plan.assignments[0]
    assert host.device == "CPU" and plan.cpu_layers == len(_cpu_a.layers) == 28
    assert host.weight_bytes > host.budget_bytes  # the weights stay in the memory-mapped GGUF on disk
    assert host.total_bytes_est == host.kv_bytes + host.compute_bytes_est + host.overhead_bytes_est
    assert host.total_bytes_est <= host.budget_bytes  # what must stay resident (and is leased) fits in RAM
    assert sum(plan.tensor_split) == plan.n_gpu_units == 49 - plan.cpu_layers
    assert any("capacity mode" in w and "disk" in w for w in plan.warnings)
    assert any("paged from disk" in c.label and c.feasible for c in plan.candidates)


def test_disk_paging_needs_capacity_mode_and_comes_after_the_ram_tier():
    off = make_plan(req(PlanStrategy.local_only, allow_capacity_mode=False), shape14(), [gpu(6)], [], 4 * GiB, cpu(3))
    assert not off.feasible and not off.capacity_mode and any("over budget" in r for r in off.reasons)
    ram = make_plan(req(PlanStrategy.local_only), shape14(), [gpu(6)], [], 32 * GiB, cpu(16))
    assert ram.feasible and ram.tier == "gpu_ram" and not ram.capacity_mode


# ----------------------------------------------------------------------------- engine flags


def _local_plan():
    return make_plan(req(PlanStrategy.local_only), shape05(), [gpu(10)], [], 16 * GiB)


def test_default_plans_carry_no_engine_flags():
    plan = _local_plan()
    assert plan.engine_flags == {} and not {"-fa", "-b", "-ub", "-t", "-tb"} & set(args_for(plan))


def test_engine_flags_become_llama_server_arguments():
    flags = {"flash_attn": "on", "ubatch": 256, "batch": 1024, "threads": 8, "threads_batch": 12}
    args = args_for(_local_plan().model_copy(update={"engine_flags": flags}))
    for flag, value in [("-fa", "on"), ("-b", "1024"), ("-ub", "256"), ("-t", "8"), ("-tb", "12")]:
        assert args[args.index(flag) + 1] == value
    assert args[args.index("-fa") + 1] == "on" and args.index("--host") > args.index("-tb")


@pytest.mark.parametrize("flags", [
    {"no_mmap": True}, {"flash_attn": "yes"}, {"flash_attn": 1}, {"threads": 0}, {"threads": -1},
    {"threads": True}, {"ubatch": "512"}, {"batch": 0}, {"ubatch": 1024, "batch": 512}, {"ubatch": 4096},
    {"threads_batch": 10_000},
])
def test_bad_engine_flags_are_rejected_with_a_fix(flags):
    with pytest.raises(EngineAdapterError) as exc:
        args_for(_local_plan().model_copy(update={"engine_flags": flags}))
    assert exc.value.code == "bad_engine_flags" and exc.value.fix


# ----------------------------------------------------------------------------- server slots (-np)


def _np(args: list[str]) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == "-np"]


def test_default_plan_launches_one_slot():
    plan = _local_plan()
    args = args_for(plan)
    assert _np(args) == ["1"] and "-kvu" not in args and "parallel" not in plan.engine_options
    assert object.__new__(LlamaRpcAdapter).max_concurrency(plan) == 1


def test_parallel_launches_n_unified_slots():
    plan = _local_plan().model_copy(update={"engine_options": {"parallel": 4}})
    args = args_for(plan)
    assert _np(args) == ["4"] and args[args.index("-np") + 2] == "-kvu" and args.count("-kvu") == 1
    assert object.__new__(LlamaRpcAdapter).max_concurrency(plan) == 4
    # everything but the slot flags is the default plan's argv, in the same order
    default = args_for(_local_plan())
    assert [a for a in args if a != "-kvu"] == [("4" if i == default.index("-np") + 1 else a)
                                                 for i, a in enumerate(default)]


def test_a_moe_offload_capacity_mode_plan_takes_parallel_slots():
    """Every 671B and Kimi moe_offload plan has capacity_mode (an SSD estimate); only the stream tier is refused."""
    from test_moe_placement import moe_plan

    for peer in (False, True):
        base = moe_plan(peer).model_copy(update={"capacity_mode": True})
        plan = base.model_copy(update={"engine_options": {"parallel": 4}})
        args, default = args_for(plan), args_for(base)
        assert _np(args) == ["4"] and "-kvu" in args
        for flag in ("-ot", "--cache-ram", "--load-mode"):
            assert args[args.index(flag) + 1] == default[default.index(flag) + 1]
        assert {"--no-repack", "--no-host", "--no-op-offload"} <= set(args)


@pytest.mark.parametrize("value", [0, 17, -1, True, "4", 2.0, None])
def test_bad_parallel_refused(value):
    plan = _local_plan().model_copy(update={"engine_options": {"parallel": value}})
    with pytest.raises(EngineAdapterError) as exc:
        args_for(plan)
    assert exc.value.code == "bad_plan" and exc.value.fix
    assert object.__new__(LlamaRpcAdapter).max_concurrency(plan) == 1  # never raises: the gateway's cap falls back


def test_parallel_on_the_stream_tier_is_refused():
    from hypercompute.adapters.llama_rpc import parallel_slots

    plan = _local_plan().model_copy(update={"tier": "stream", "engine_options": {"parallel": 2}})
    with pytest.raises(EngineAdapterError) as exc:
        parallel_slots(plan)
    assert exc.value.code == "bad_plan" and "stream" in str(exc.value)
    assert parallel_slots(plan.model_copy(update={"engine_options": {}})) == 1  # one slot is fine on any tier
