"""Host-RAM tier: the first layers run on the CPU when the GPUs cannot hold the whole model."""

from hypercompute.adapters.llama_output import parse_placement
from hypercompute.schemas import DeviceBudget, GgufSummary, PlanRequest, PlanStrategy
from hypercompute.services.planner import ModelShape, make_plan, ram_tier_layout
from hypercompute.units import GiB, MiB


def qwen14b_q6k() -> GgufSummary:
    # Shapes of the real qwen2.5-14b-instruct-q6_k GGUF: 48 layers, n_embd 5120, 40 heads, 8 KV heads.
    return GgufSummary(architecture="qwen2", name="qwen14b", file_type="Q6_K", n_layer=48, n_embd=5120, n_head=40,
                       n_head_kv=8, context_length=32768, vocab_size=152064, tensor_count=579,
                       total_tensor_bytes=int(11.29 * GiB), layer_bytes=[int(225 * MiB)] * 48,
                       token_embd_bytes=int(609 * MiB), output_bytes=int(609 * MiB), output_tied=False,
                       other_bytes=0, chat_template_present=True)


def gpu(gib: float) -> DeviceBudget:
    return DeviceBudget(node_id="local", device="CUDA0", description="RTX 5070 Ti", endpoint=None,
                        pool_id="vram:CUDA0", pool_kind="discrete_vram", budget_bytes=int(gib * GiB),
                        budget_basis="test")


def cpu(gib: float, pool: str = "ram", kind: str = "system_ram") -> DeviceBudget:
    return DeviceBudget(node_id="local", device="CPU", description="host CPU", endpoint=None, pool_id=pool,
                        pool_kind=kind, budget_bytes=int(gib * GiB), budget_basis="test")


def shape() -> ModelShape:
    return ModelShape.from_gguf(qwen14b_q6k(), 4096)


def test_local_gpu_alone_does_not_fit_but_ram_tier_does():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only), shape(), [gpu(10.6)], [],
                     8 * GiB, cpu(8))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.cpu_layers > 0
    assert plan.assignments[0].device == "CPU" and plan.assignments[0].layers[0] == 0  # input side stays on CPU
    assert plan.device_list == ["CUDA0"]  # the CPU is implied by -ngl, never listed in --device
    assert sum(plan.tensor_split) == plan.n_gpu_units == 48 + 1 - plan.cpu_layers
    assert any("host RAM" in c.label for c in plan.candidates)
    assert any("from system RAM" in w for w in plan.warnings)


def test_ram_tier_uses_the_fewest_cpu_layers():
    layout, ok = ram_tier_layout(shape(), [gpu(10.6)], cpu(8), keep_all=False)
    fewer, ok_fewer = ram_tier_layout(shape(), [gpu(11.5)], cpu(8), keep_all=False)
    assert ok and ok_fewer and len(fewer[0].layers) < len(layout[0].layers)


def test_ram_tier_can_be_disabled():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only, allow_ram_tier=False), shape(),
                     [gpu(10.6)], [], 8 * GiB, cpu(8))
    assert not plan.feasible and plan.tier == "gpu" and plan.cpu_layers == 0


def test_ram_tier_respects_the_host_ram_budget():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only), shape(), [gpu(10.6)], [],
                     1 * GiB, cpu(0.7))
    assert not plan.feasible  # token embeddings alone eat the CPU budget


def test_unified_memory_host_counts_its_pool_once():
    mac_gpu = DeviceBudget(node_id="local", device="MTL0", description="Apple M4", endpoint=None, pool_id="uma",
                           pool_kind="unified", budget_bytes=int(9 * GiB), budget_basis="test")
    layout, ok = ram_tier_layout(shape(), [mac_gpu], cpu(10, pool="uma", kind="unified"), keep_all=False)
    if ok:
        used = sum(a.total_bytes_est for a in layout)
        assert used <= 10 * GiB  # CPU and GPU draw from the same 10 GiB, never 19 GiB


def test_gpu_tier_is_preferred_when_it_fits():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.auto), shape(), [gpu(14)], [], 8 * GiB, cpu(8))
    assert plan.feasible and plan.tier == "gpu" and plan.cpu_layers == 0 and plan.n_gpu_units is None


def test_real_14b_cpu_tier_log_is_parsed(fixture_text):
    """llama.cpp b11770 with -ngl 40 on the 14B Q6_K: 9 layers on the CPU, 39 + output on CUDA0 (real run)."""
    p = parse_placement(fixture_text("llama_server_14b_ngl40_cpu_tier_b11770.log"))
    by = {d.name: d for d in p.devices}
    assert p.offloaded == "40/49"
    assert by["CPU"].layers_est == 9 and by["CUDA0"].layers_est == 39
    assert by["CPU_Mapped"].model_bytes > 2 * GiB  # token embeddings + the 9 CPU layers, memory-mapped
