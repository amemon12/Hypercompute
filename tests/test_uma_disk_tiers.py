"""A unified-memory host (GB10, Apple) does not run CPU layers from disk. Its GPU, its CPU and the file cache share one
pool, so CPU layers paged from the GGUF (gpu_disk) are charged like resident ones, CPU units are not streamed, MoE
models keep moe_offload, and the planner's plan is not kept feasible where the capacity optimizer found no feasible
layout.

The two cases are the GX10 rows of the paper (docs/research/PAPER-RUNS-2026-10-22.md): Mistral Large 2411 Q8_0 alone
(R6.x, a gpu_disk plan that thrashed the box) and Llama 4 Maverick run together with three peers whose caps were too
small for moe_offload (R8, a gpu_disk plan that stopped SSH until a reboot). Budgets are the GX10's as its plans show
them: CUDA0 79.06 GiB, the CPU tier ~93 GiB, both on the one `uma` pool.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from factories import contribution, engine_info, make_inventory
from hypercompute.capacity import bridge
from hypercompute.capacity.measurements import MeasurementStore
from hypercompute.capacity.pools import HostMemory
from hypercompute.capacity.service import CapacityService
from hypercompute.schemas import (
    Device,
    DeviceBudget,
    EngineDevice,
    GgufSummary,
    MeasuredProfile,
    MemoryPool,
    NodeView,
    PlanRequest,
    PlanStrategy,
)
from hypercompute.services import planner
from hypercompute.services.deployments import cpu_budget, local_budgets
from hypercompute.services.planner import (
    UNIFIED_NO_STREAM,
    ModelShape,
    StreamCapability,
    layout_from_units,
    make_plan,
)
from hypercompute.services.selector import Inputs, choose_plan
from hypercompute.units import GiB, MiB
from test_capacity_bridge import FakeLeases, FakeModels, FakeNodes, local_node, make_manifest, make_service
from test_selector import art
from test_split_order import EP5, EP6, cpu, mac, rtx_peer, shape14, summary
from test_stream_tier import cap14

EP7 = "10.0.0.7:47052"
NOW = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
GB10_TOTAL = int(121.6 * GiB)


def gb10_gpu(gib: float = 79.06) -> DeviceBudget:
    return DeviceBudget(node_id="local", device="CUDA0", description="NVIDIA GB10", endpoint=None, pool_id="uma",
                        pool_kind="unified", budget_bytes=int(gib * GiB), budget_basis="test")


def gb10_cpu(gib: float = 93.5, kind: str = "unified") -> DeviceBudget:
    return DeviceBudget(node_id="local", device="CPU", description="host CPU (system RAM)", endpoint=None,
                        pool_id="uma", pool_kind=kind, budget_bytes=int(gib * GiB), budget_basis="test")  # type: ignore[arg-type]


def mistral_large() -> ModelShape:  # Q8_0: 88 x 1.37 GiB layers, 410 MiB head, 120.5 GiB of layers (R6.x)
    shape = ModelShape.from_gguf(summary(88, 1402.6, 12288, 96, 8, 410), 4096)
    assert shape is not None
    return shape


def maverick() -> ModelShape:
    """Llama 4 Maverick UD-Q2_K_XL as its GX10 plans show it (results/capacity/20261025T05*): 48 layers, every odd one
    MoE (128 experts, 1 used), layers 1 and 3 kept larger (~6.2 GiB) than the later MoE layers (~5.5 GiB), dense layers
    ~165 MiB; 138 GiB of layers."""
    layer, expert = [], []
    for i in range(48):
        big = i < 5
        layer.append(165 * MiB if i % 2 == 0 else (6370 if big else 5670) * MiB)
        expert.append(0 if i % 2 == 0 else (6170 if big else 5470) * MiB)
    s = GgufSummary(architecture="llama4", name="maverick", file_type="Q2_K", n_layer=48, n_embd=5120, n_head=40,
                    n_head_kv=8, context_length=1048576, vocab_size=202048, tensor_count=1,
                    total_tensor_bytes=sum(layer), layer_bytes=layer, token_embd_bytes=820 * MiB,
                    output_bytes=820 * MiB, output_tied=False, other_bytes=0, chat_template_present=True,
                    key_length=128, value_length=128, expert_count=128, expert_used_count=1, layer_expert_bytes=expert)
    shape = ModelShape.from_gguf(s, 4096)
    assert shape is not None and shape.is_moe
    return shape


def req(strategy: PlanStrategy, **kw: object) -> PlanRequest:
    return PlanRequest(model_id="m", strategy=strategy, **kw)  # type: ignore[arg-type]


def small_peers() -> list[DeviceBudget]:  # too small for layer 1 or 3 whole, room for one later MoE layer each
    return [rtx_peer(6.5, EP6), mac(6.6, EP5), mac(6.6, EP7)]


def roomy_peers() -> list[DeviceBudget]:  # the caps of the run that loaded (results/capacity/20261025T053913Z-*)
    return [rtx_peer(9.0, EP6), mac(10.67, EP5), mac(10.67, EP7)]


def paged_candidate(plan):
    return next(c for c in plan.candidates if "paged from disk" in c.label)


# --------------------------------------------------------------------------------------------------- the planner


def test_mistral_large_alone_on_a_gb10_is_refused_instead_of_paged_from_disk():
    plan = make_plan(req(PlanStrategy.local_only), mistral_large(), [gb10_gpu()], [], 117 * GiB, gb10_cpu(93.54))
    assert not plan.feasible and plan.tier != "gpu_disk" and not plan.capacity_mode
    paged = paged_candidate(plan)
    assert not paged.feasible and "unified" in paged.reason and "file cache" in paged.reason


def test_the_same_split_still_pages_from_disk_on_a_host_with_its_own_gpu_memory():
    gpu = gb10_gpu().model_copy(update={"pool_id": "vram:CUDA0", "pool_kind": "discrete_vram"})
    cpu = gb10_cpu(32).model_copy(update={"pool_id": "ram", "pool_kind": "system_ram"})  # less RAM than the CPU part
    plan = make_plan(req(PlanStrategy.local_only), mistral_large(), [gpu], [], 117 * GiB, cpu)
    assert plan.feasible and plan.tier == "gpu_disk" and plan.cpu_layers == 32 and plan.capacity_mode


def test_maverick_run_together_with_small_peers_on_a_gb10_is_not_paged_from_disk():
    plan = make_plan(req(PlanStrategy.all_selected), maverick(), [gb10_gpu()], small_peers(), 117 * GiB, gb10_cpu())
    assert not plan.feasible and plan.tier != "gpu_disk" and not plan.capacity_mode
    moe = next(c for c in plan.candidates if "MoE experts" in c.label)
    assert not moe.feasible and "cannot hold one whole layer" in moe.reason  # why moe_offload could not run
    assert not paged_candidate(plan).feasible


def test_maverick_on_a_gb10_keeps_moe_offload_alone_and_with_roomy_peers():
    alone = make_plan(req(PlanStrategy.local_only), maverick(), [gb10_gpu()], [], 117 * GiB, gb10_cpu())
    assert alone.feasible and alone.tier == "moe_offload" and alone.device_list == ["CUDA0"]
    together = make_plan(req(PlanStrategy.all_selected), maverick(), [gb10_gpu()], roomy_peers(), 117 * GiB,
                         gb10_cpu())
    assert together.feasible and together.tier == "moe_offload"
    assert [len(a.layers) for a in together.assignments] == [0, 3, 2, 2, 41]  # as the 05:39 run loaded it


def test_a_measured_gpu_disk_split_is_not_replayed_on_a_unified_host():
    shape = mistral_large()
    _, ok = layout_from_units(shape, [gb10_gpu()], [57], gb10_cpu(93.54), 32, paged=True)
    assert not ok
    gpu = gb10_gpu().model_copy(update={"pool_id": "vram:CUDA0", "pool_kind": "discrete_vram"})
    ram = gb10_cpu(32, "system_ram").model_copy(update={"pool_id": "ram"})
    _, discrete_ok = layout_from_units(shape, [gpu], [57], ram, 32, paged=True)
    assert discrete_ok


def test_a_unified_host_does_not_stream_cpu_units_from_the_ssd():
    shape = mistral_large()
    cap = StreamCapability(unit_buffer_bytes=(*shape.layer_bytes, shape.output_bytes))
    plan = make_plan(req(PlanStrategy.local_only), shape, [gb10_gpu()], [], 117 * GiB, gb10_cpu(93.54), stream=cap)
    assert not plan.feasible and plan.tier not in ("stream", "gpu_disk") and plan.stream is None
    assert not next(c for c in plan.candidates if "streamed" in c.label).feasible


def test_a_unified_host_without_a_gpu_does_not_stream_either():
    shape = shape14()  # 10.9 GiB on a 4 GiB CPU tier: fits only by streaming, which a discrete host does
    ram = cpu(4)
    assert make_plan(req(PlanStrategy.local_only), shape, [], [], 6 * GiB, ram, stream=cap14()).tier == "stream"
    plan = make_plan(req(PlanStrategy.local_only), shape, [], [], 6 * GiB,
                     ram.model_copy(update={"pool_id": "uma", "pool_kind": "unified"}), stream=cap14())
    assert not plan.feasible and plan.stream is None and not plan.capacity_mode
    streamed = next(c for c in plan.candidates if "streamed" in c.label)
    assert not streamed.feasible and streamed.reason == UNIFIED_NO_STREAM


def test_a_mac_host_does_not_page_from_disk_either():
    mtl = DeviceBudget(node_id="local", device="MTL0", description="Apple M4 (10-core GPU)", endpoint=None,
                       pool_id="uma", pool_kind="unified", budget_bytes=5 * GiB, budget_basis="test")
    plan = make_plan(req(PlanStrategy.local_only), shape14(), [mtl], [], 8 * GiB, gb10_cpu(6))
    assert not plan.feasible and plan.tier != "gpu_disk" and not paged_candidate(plan).feasible
    with_ram = make_plan(req(PlanStrategy.local_only), shape14(), [mtl], [], 20 * GiB, gb10_cpu(16))
    assert with_ram.feasible and with_ram.tier == "gpu_ram"  # a pool that holds it all still runs it


def test_a_measured_gpu_disk_profile_is_not_replayed_on_a_unified_host():
    shape = ModelShape.from_gguf(summary(44, 1400, 12288, 96, 8, 410), 4096)  # 60 GiB: fits the pool resident
    request = req(PlanStrategy.local_only)
    profile = MeasuredProfile(id="p" * 32, model_id="m", signature="s", objective="interactive",
                              plan_fields={"device_list": ["CUDA0"], "tensor_split": [31], "cpu_layers": 14,
                                           "tier": "gpu_disk", "engine_flags": {}, "rpc_endpoints": []},
                              ttft_ms=None, decode_tokens_per_s=1.0, prefill_tokens_per_s=None, score=1.0,
                              tune_id=None, run_ids=[], created_at=NOW)

    def lookup(_model_id, _sig, _objective, accept):
        return profile if accept(profile) else None

    plan = choose_plan(Inputs(req=request, artifact=art("m"), shape=shape, local=[gb10_gpu(45)], remote=[],
                              cpu=gb10_cpu(), host_ram_free=117 * GiB, profile=lookup))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.profile_id is None  # the heuristic plan, resident
    assert any("ignored" in w and "unified pool" in w for w in plan.warnings)


# --------------------------------------------------------------------------------------------------- the service


def gb10_node(free_gib: float = 117.0) -> NodeView:
    free = int(free_gib * GiB)
    pools = [MemoryPool(id="uma", kind="unified", total_bytes=GB10_TOTAL, free_bytes=free, measured_at=NOW,
                        source="fixture")]
    devices = [Device(id="CPU", kind="cpu", name="NVIDIA Grace", pool_id="uma", backend="cpu", source="fixture"),
               Device(id="CUDA0", kind="gpu", name="NVIDIA GB10", pool_id="uma", backend="cuda",
                      total_bytes=GB10_TOTAL, free_bytes=free, source="fixture")]
    engine = engine_info(devices=[EngineDevice(name="CUDA0", description="NVIDIA GB10", total_bytes=GB10_TOTAL,
                                               free_bytes=free)])
    inv = make_inventory("linux", engine=engine).model_copy(update={"memory_pools": pools, "devices": devices})
    return local_node().model_copy(update={"inventory": inv, "contribution": contribution(
        ["CUDA0"], max_gib=79.06, reserve_gib=24.3)})


def gb10_service(tmp_path: Path, m, node: NodeView, *, stream: bool = False) -> CapacityService:
    cap = StreamCapability(unit_buffer_bytes=tuple(u.buffer_bytes for u in m.units)) if stream else None
    host = HostMemory(total=GB10_TOTAL, available=117 * GiB, commit_charge=None, commit_limit=None, system_cache=None,
                      observed_at=NOW, source="test")
    probes = bridge.Probes(memory=lambda: host, disk_free=lambda _p: 500 * GiB, manifest=lambda _a, _s: m,
                           stream=lambda _s: cap, clock=lambda: NOW)
    return CapacityService(FakeNodes(node), FakeModels(), FakeLeases(), MeasurementStore(tmp_path / "measurements"),  # type: ignore[arg-type]
                           probes=probes, models_dir=tmp_path / "models")


def mistral_manifest():
    return make_manifest(n_layer=88, block_mib=1402.6, head_mib=410, n_embd=12288, embd_mib=410)


def test_the_optimizers_no_feasible_layout_is_not_overruled_on_a_unified_host(tmp_path):
    m, node = mistral_manifest(), gb10_node()
    request = PlanRequest(model_id=m.model_id, strategy=PlanStrategy.local_only)
    shape = bridge.model_shape(m, request.ctx_size)
    # the planner's plan from before the fix: its CPU tier did not know it shares the GPU's pool
    stale_cpu = cpu_budget(node).model_copy(update={"pool_kind": "system_ram"})  # type: ignore[union-attr]
    legacy = planner.make_plan(request, shape, local_budgets(node), [], 117 * GiB, stale_cpu)
    assert legacy.feasible and legacy.tier == "gpu_disk"
    plan = gb10_service(tmp_path, m, node).plan(request, legacy)
    assert not plan.feasible and plan.fit_label == "does_not_fit"
    assert any("no feasible layout" in r and "unified" in r for r in plan.reasons)


def test_a_discrete_host_keeps_the_planners_gpu_disk_plan_where_the_optimizer_has_none(tmp_path):
    m, node = make_manifest(n_layer=80, block_mib=400), local_node()  # 31 GiB on 11 GiB of VRAM and 16 GiB of RAM
    request = PlanRequest(model_id=m.model_id, strategy=PlanStrategy.local_only)
    legacy = planner.make_plan(request, bridge.model_shape(m, request.ctx_size), local_budgets(node), [], 16 * GiB,
                               cpu_budget(node))
    assert legacy.feasible and legacy.tier == "gpu_disk"
    plan = make_service(tmp_path, m, node).plan(request, legacy)
    assert plan.feasible and plan.tier == "gpu_disk" and plan.id == legacy.id
    assert any("no feasible layout" in w and "the planner's plan is kept" in w for w in plan.warnings)


def test_the_fixed_planner_and_the_optimizer_agree_the_gb10_cannot_run_it(tmp_path):
    m, node = mistral_manifest(), gb10_node()
    request = PlanRequest(model_id=m.model_id, strategy=PlanStrategy.local_only)
    legacy = planner.make_plan(request, bridge.model_shape(m, request.ctx_size), local_budgets(node), [], 117 * GiB,
                               cpu_budget(node))
    assert not legacy.feasible and legacy.tier != "gpu_disk"
    assert not gb10_service(tmp_path, m, node).plan(request, legacy).feasible


def test_the_optimizer_does_not_stream_on_a_unified_host(tmp_path):
    m = mistral_manifest()
    explained = gb10_service(tmp_path, m, gb10_node(), stream=True).explain(m.model_id)
    assert not explained["feasible"]
    assert any("unified" in a and "stream" in a for a in explained["assumptions"])
