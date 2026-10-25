"""MoE-aware placement end to end: expert bytes from GGUF headers, the moe_offload planner tier (dense parts on the GPU,
experts of the fewest host layers on the CPU, file-backed), peers first with whole layers, the -ot pattern, the
llama-server arguments, the load-log check and the capacity optimizer stepping aside. No hardware is touched.

Model shapes named after real models (gpt-oss-120b) are approximations of their per-layer sizes, not exact figures.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import replace
from pathlib import Path

import pytest
from hypercompute import gguf
from hypercompute.adapters.base import DeploymentSpec, EngineAdapterError, ValidationResult
from hypercompute.adapters.llama_output import check_moe_placement, parse_overrides, parse_placement
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter, moe_offload_args
from hypercompute.capacity import bridge
from hypercompute.capacity.manifest import ManifestError, build_manifest
from hypercompute.capacity.service import CapacityUnavailable
from hypercompute.capacity.types import ModelManifest
from hypercompute.schemas import Deployment, DeploymentState, GgufSummary, Plan, PlanRequest, PlanStrategy
from hypercompute.services import planner
from hypercompute.services.deployments import DeploymentManager, cpu_budget, local_budgets, remote_budgets
from hypercompute.services.moe_tier import MMAP_READ_BYTES_PER_S, cpu_expert_layers, moe_offload_layout
from hypercompute.services.planner import DECODE_MS_PER_GB, OVERHEAD_BY_KIND, ModelShape, make_plan
from hypercompute.services.selector import Inputs, choose_plan
from hypercompute.units import GiB, MiB
from test_capacity_bridge import MODEL, NOW, artifact, local_node, mac_node, make_manifest, make_service
from test_capacity_manifest import Q8_0, Tensor, base_meta, write_gguf
from test_split_order import EP5, cpu, gpu, mac

MXFP4 = 39
N_EXPERT, N_USED = 8, 2


# --------------------------------------------------------------------------------------------------- synthetic GGUFs


def oss_block(i: int, embd: int = 64, ff: int = 32) -> list[Tensor]:
    """One gpt-oss style block: attention with sinks, a router with a bias, and MXFP4 experts with F32 biases."""
    return [
        Tensor(f"blk.{i}.attn_norm.weight", (embd,)), Tensor(f"blk.{i}.attn_q.weight", (embd, embd), Q8_0),
        Tensor(f"blk.{i}.attn_k.weight", (embd, 16), Q8_0), Tensor(f"blk.{i}.attn_v.weight", (embd, 16), Q8_0),
        Tensor(f"blk.{i}.attn_output.weight", (embd, embd), Q8_0), Tensor(f"blk.{i}.attn_sinks.weight", (4,)),
        Tensor(f"blk.{i}.post_attention_norm.weight", (embd,)),
        Tensor(f"blk.{i}.ffn_gate_inp.weight", (embd, N_EXPERT)), Tensor(f"blk.{i}.ffn_gate_inp.bias", (N_EXPERT,)),
        *(Tensor(f"blk.{i}.ffn_{p}_exps.weight", (embd if p != "down" else ff, ff if p != "down" else embd, N_EXPERT),
                 MXFP4) for p in ("gate", "up", "down")),
        *(Tensor(f"blk.{i}.ffn_{p}_exps.bias", (ff if p != "down" else embd, N_EXPERT))
          for p in ("gate", "up", "down")),
    ]


def glm_block(i: int, embd: int = 64, ff: int = 32) -> list[Tensor]:
    """A GLM-4.5 style block: layer 0 is a dense FFN; later layers have a shared expert, exp_probs_b and experts."""
    attn = [Tensor(f"blk.{i}.attn_norm.weight", (embd,)), Tensor(f"blk.{i}.attn_q.weight", (embd, embd), Q8_0)]
    if i == 0:
        return [*attn, *(Tensor(f"blk.0.ffn_{p}.weight", (embd, ff) if p != "down" else (ff, embd), Q8_0)
                         for p in ("gate", "up", "down"))]
    return [*attn, Tensor(f"blk.{i}.ffn_gate_inp.weight", (embd, N_EXPERT)), Tensor(f"blk.{i}.exp_probs_b.bias",
                                                                                   (N_EXPERT,)),
            *(Tensor(f"blk.{i}.ffn_{p}_shexp.weight", (embd, ff), Q8_0) for p in ("gate", "up", "down")),
            *(Tensor(f"blk.{i}.ffn_{p}_exps.weight", (embd, ff, N_EXPERT), Q8_0) for p in ("gate", "up", "down"))]


def moe_meta(arch: str, n_layer: int, **changes: object) -> dict[str, object]:
    meta = {k.replace("qwen2.", f"{arch}."): v for k, v in base_meta().items()}
    return {**meta, "general.architecture": arch, f"{arch}.block_count": n_layer, f"{arch}.expert_count": N_EXPERT,
            f"{arch}.expert_used_count": N_USED, **changes}


def head() -> list[Tensor]:
    return [Tensor("token_embd.weight", (64, 100), Q8_0), Tensor("output_norm.weight", (64,)),
            Tensor("output.weight", (64, 100), Q8_0)]


def expert_nbytes(tensors: list[Tensor], layer: int) -> int:
    return sum(t.data_bytes for t in tensors if gguf.is_expert_tensor(t.name) and t.name.startswith(f"blk.{layer}."))


def test_manifest_splits_gpt_oss_blocks_into_dense_and_expert_bytes(tmp_path):
    tensors = [*head(), *oss_block(0), *oss_block(1), *oss_block(2)]
    m = build_manifest("oss", [write_gguf(tmp_path / "oss.gguf", tensors, moe_meta("gpt-oss", 3))])
    assert m.is_moe and (m.expert_count, m.expert_used_count) == (N_EXPERT, N_USED)
    assert m.active_expert_fraction == pytest.approx(N_USED / N_EXPERT)
    for i in range(3):
        u = m.block(i)
        assert u.expert_bytes == expert_nbytes(tensors, i) > 0  # exact ggml sizes, weights and biases
        assert u.dense_bytes == u.weight_bytes - u.expert_bytes > 0  # attention, sinks, router stay dense
    assert m.head.expert_bytes == 0


def test_shared_experts_router_bias_and_dense_first_layers_count_as_dense(tmp_path):
    tensors = [*head(), *glm_block(0), *glm_block(1)]
    m = build_manifest("glm", [write_gguf(tmp_path / "glm.gguf", tensors,
                                          moe_meta("glm4moe", 2, **{"glm4moe.expert_shared_count": 1}))])
    assert m.block(0).expert_bytes == 0 and m.block(1).expert_bytes == expert_nbytes(tensors, 1)
    assert m.expert_shared_count == 1 and m.block(1).dense_bytes > 0  # ffn_*_shexp and exp_probs_b are dense
    assert not any(gguf.is_expert_tensor(n) for n in ("blk.1.ffn_up_shexp.weight", "blk.1.ffn_gate_inp.weight",
                                                      "blk.1.exp_probs_b.bias", "blk.0.ffn_up.weight"))
    assert all(gguf.is_expert_tensor(n) for n in ("blk.9.ffn_gate_up_exps.weight", "blk.9.ffn_down_chexps.weight",
                                                  "blk.10.ffn_down_exps.bias", "blk.3.ffn_up_exps.scale"))


def test_dense_models_keep_a_dense_manifest_and_shape(tmp_path):
    tensors = [Tensor("token_embd.weight", (64, 100)), Tensor("output.weight", (64, 100)),
               *(Tensor(f"blk.{i}.ffn_up.weight", (64, 32)) for i in range(2))]
    path = write_gguf(tmp_path / "dense.gguf", tensors, base_meta())
    m = build_manifest("dense", [path])
    assert not m.is_moe and m.expert_count == 0 and all(u.expert_bytes == 0 for u in m.units)
    assert m.active_expert_fraction == 1.0
    s = gguf.summarize([path])
    assert s.layer_expert_bytes == [] and s.expert_count is None
    shape = ModelShape.from_gguf(s, 4096)
    assert shape is not None and not shape.is_moe and shape.expert_bytes == () and shape.dense(0) == s.layer_bytes[0]
    assert bridge.model_shape(m, 4096).expert_bytes == ()


@pytest.mark.parametrize("meta, why", [({"gpt-oss.expert_count": None, "gpt-oss.expert_used_count": None},
                                        "no gpt-oss.expert_count"),
                                       ({"gpt-oss.expert_used_count": N_EXPERT + 1}, "expert_used_count")])
def test_inconsistent_expert_metadata_is_refused(tmp_path, meta, why):
    path = write_gguf(tmp_path / "bad.gguf", [*head(), *oss_block(0)], moe_meta("gpt-oss", 1, **meta))
    with pytest.raises(ManifestError, match=why):
        build_manifest("bad", [path])


def test_summary_and_both_shapes_carry_the_expert_split(tmp_path):
    tensors = [*head(), *oss_block(0), *oss_block(1)]
    path = write_gguf(tmp_path / "oss.gguf", tensors, moe_meta("gpt-oss", 2))
    s = gguf.summarize([path])
    assert (s.expert_count, s.expert_used_count) == (N_EXPERT, N_USED)
    assert len(s.layer_expert_bytes) == 2 and all(0 < e < b for e, b in zip(s.layer_expert_bytes, s.layer_bytes))
    shape = ModelShape.from_gguf(s, 4096)
    assert shape is not None and shape.is_moe and shape.active_fraction == pytest.approx(0.25)
    m_shape = bridge.model_shape(build_manifest("oss", [path]), 4096)
    assert m_shape.is_moe and m_shape.expert_bytes == tuple(expert_nbytes(tensors, i) for i in range(2))


# --------------------------------------------------------------------------------------------------- -ot pattern


def test_number_alternation_matches_exactly_the_given_layers():
    rng = random.Random(7)
    for _ in range(200):
        layers = set(rng.sample(range(130), rng.randint(1, 60)))
        alt = gguf.number_alternation(layers)
        assert "," not in alt  # llama.cpp splits -ot values at commas
        rx = re.compile(f"^({alt})$")
        assert {n for n in range(1000) if rx.match(str(n))} == layers
    assert gguf.number_alternation(range(3, 28)) == "[3-9]|1[0-9]|2[0-7]"
    with pytest.raises(ValueError):
        gguf.number_alternation([])


def test_override_pattern_names_only_the_expert_tensors_of_its_layers():
    names = [f"blk.{i}.{t}" for i in range(48) for t in (
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_down_exps.bias",
        "ffn_gate_up_exps.weight", "ffn_up_chexps.weight", "ffn_up_shexp.weight", "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight", "exp_probs_b.bias", "attn_q.weight", "ffn_norm.weight")]
    layers = [5, 6, 7, 8, 30, 31, 47]
    rx = re.compile(gguf.expert_override_pattern(layers))  # llama.cpp uses std::regex_search
    hits = {n for n in [*names, "output.weight", "token_embd.weight"] if rx.search(n)}
    assert hits == {n for n in names if gguf.is_expert_tensor(n) and int(n.split(".")[1]) in layers}


# --------------------------------------------------------------------------------------------------- planner


def oss120_summary(n_layer: int = 36, dense_mib: float = 60, expert_mib: float = 1614) -> GgufSummary:
    """Approximately gpt-oss-120b MXFP4: 36 blocks of ~60 MiB attention + ~1.6 GiB of experts, 128 experts, 4 used,
    n_embd 2880, 64 heads of 64, 8 KV heads, ~620 MiB head and token embeddings."""
    return GgufSummary(architecture="gpt-oss", name="oss", file_type="MXFP4_MOE", n_layer=n_layer, n_embd=2880,
                       n_head=64, n_head_kv=8, context_length=131072, vocab_size=201088, tensor_count=1,
                       total_tensor_bytes=0, layer_bytes=[int((dense_mib + expert_mib) * MiB)] * n_layer,
                       token_embd_bytes=620 * MiB, output_bytes=620 * MiB, output_tied=False, other_bytes=0,
                       chat_template_present=True, key_length=64, value_length=64, expert_count=128,
                       expert_used_count=4, layer_expert_bytes=[int(expert_mib * MiB)] * n_layer)


def oss120(**kw: float) -> ModelShape:
    shape = ModelShape.from_gguf(oss120_summary(**kw), 4096)  # type: ignore[arg-type]
    assert shape is not None and shape.is_moe
    return shape


def fewest_by_arithmetic(shape: ModelShape, host: list[int], budget: int) -> int:
    """Independent of the planner: CUDA0 = dense + head + KV + compute + overhead + the experts left on it."""
    fixed = (sum(shape.dense(i) for i in host) + shape.output_bytes + shape.kv_per_layer * len(host)
             + shape.compute_est + OVERHEAD_BY_KIND["cuda"])
    experts = sorted((shape.experts(i) for i in host), reverse=True)
    return next(k for k in range(len(host) + 1) if fixed + sum(experts[k:]) <= budget)


def test_moe_offload_moves_the_fewest_expert_layers_that_make_cuda0_fit():
    shape = oss120()
    layout, spec, ok, why = moe_offload_layout(shape, [gpu(10.27)], cpu(6), keep_all=False)
    assert ok and spec is not None, why
    k = fewest_by_arithmetic(shape, list(range(36)), int(10.27 * GiB))
    assert len(spec.cpu_expert_layers) == k == 32
    cpu_a, gpu_a = layout
    assert gpu_a.fits and gpu_a.layers == list(range(36)) and gpu_a.holds_output
    assert gpu_a.weight_bytes == sum(shape.dense(i) for i in range(36)) + shape.output_bytes + 4 * shape.experts(0)
    assert gpu_a.total_bytes_est + shape.experts(0) > gpu(10.27).budget_bytes  # one CPU layer fewer overflows
    assert cpu_a.device == "CPU" and cpu_a.layers == [] and cpu_a.weight_bytes == spec.cpu_expert_bytes
    assert spec.cpu_expert_bytes == 32 * shape.experts(0) and spec.gpu_expert_bytes == 4 * shape.experts(0)
    assert cpu_a.total_bytes_est == shape.compute_est + OVERHEAD_BY_KIND["cpu"]  # file-backed: only buffers are private


def test_random_unequal_models_get_the_fewest_cpu_expert_layers_that_fit():
    rng = random.Random(11)
    for _ in range(300):
        n = rng.randint(2, 12)
        experts = tuple(rng.choice([0, rng.randint(50, 2000)]) * MiB for _ in range(n))
        shape = replace(oss120(n_layer=n), layer_bytes=tuple(e + rng.randint(10, 300) * MiB for e in experts),
                        expert_bytes=experts)
        if not shape.is_moe:
            continue
        budget = gpu(rng.uniform(1.0, 12.0))
        layout, spec, ok, why = moe_offload_layout(shape, [budget], cpu(64), keep_all=False)
        try:
            k = fewest_by_arithmetic(shape, list(range(n)), budget.budget_bytes)
        except StopIteration:  # the dense part alone overflows the GPU
            assert not ok and "cannot hold the dense part" in why
            continue
        assert ok and layout[-1].fits and layout[-1].total_bytes_est <= budget.budget_bytes, why
        assert len(spec.cpu_expert_layers if spec else []) == k


def test_unequal_layers_put_the_largest_expert_sets_on_the_cpu_first():
    shape = replace(oss120(n_layer=6), expert_bytes=(0, 900 * MiB, 1600 * MiB, 1200 * MiB, 1600 * MiB, 300 * MiB))
    host = list(range(6))
    assert cpu_expert_layers(shape, host, 2000 * MiB) == [2, 3, 4]  # 1600 + 1600 + 1200 leave 1200 <= 2000
    assert cpu_expert_layers(shape, host, 10 * GiB) == []
    assert cpu_expert_layers(shape, host, -1) is None


def test_ssd_bytes_per_token_follow_the_active_fraction_and_the_cache_miss_rate():
    shape = oss120()
    _, spec, _, _ = moe_offload_layout(shape, [gpu(10.27)], cpu(6), keep_all=False)
    assert spec is not None and spec.provenance == "estimate"
    private = shape.compute_est + OVERHEAD_BY_KIND["cpu"]
    cache = int(6 * GiB) - shape.token_embd_bytes - private
    assert spec.page_cache_bytes_est == cache
    assert spec.miss_rate_est == pytest.approx(1 - cache / spec.cpu_expert_bytes)
    assert spec.ssd_bytes_per_token_est == int(4 / 128 * spec.cpu_expert_bytes * spec.miss_rate_est)
    io_ms = 1000 * spec.ssd_bytes_per_token_est / MMAP_READ_BYTES_PER_S
    assert spec.decode_ms_est > io_ms + DECODE_MS_PER_GB["cpu"] * spec.cpu_expert_bytes * 4 / 128 / 1e9
    _, roomy, _, _ = moe_offload_layout(shape, [gpu(10.27)], cpu(64), keep_all=False)
    assert roomy is not None and roomy.miss_rate_est == 0 and roomy.ssd_bytes_per_token_est == 0


def test_plan_for_a_large_moe_on_this_host_is_moe_offload_with_honest_numbers():
    plan = make_plan(PlanRequest(model_id="oss", strategy=PlanStrategy.local_only), oss120(), [gpu(10.27)], [],
                     10 * GiB, cpu(6))
    assert plan.feasible and plan.tier == "moe_offload" and plan.moe is not None
    assert plan.device_list == ["CUDA0"] and plan.tensor_split == [37] and plan.cpu_layers == 0
    assert plan.n_gpu_units is None and plan.capacity_mode  # expected SSD reads per token make it capacity mode
    assert plan.moe.override == gguf.expert_override_pattern(plan.moe.cpu_expert_layers) + "=CPU"
    text = " ".join(plan.warnings)
    assert "estimate" in text and "not a measurement" in text and "tokens/s" in text and "file cache" in text
    assert any("MoE experts on the host CPU" in c.label and c.feasible for c in plan.candidates)


def test_without_capacity_mode_moe_offload_must_fit_the_file_cache():
    req = PlanRequest(model_id="oss", strategy=PlanStrategy.local_only, allow_capacity_mode=False)
    assert make_plan(req, oss120(), [gpu(10.27)], [], 10 * GiB, cpu(6)).tier != "moe_offload"
    roomy = make_plan(req, oss120(), [gpu(10.27)], [], 80 * GiB, cpu(64))
    assert roomy.feasible and roomy.tier == "moe_offload" and not roomy.capacity_mode


def test_dense_models_never_take_the_moe_tier():
    dense = replace(oss120(), expert_bytes=(), expert_count=0, expert_used_count=0)
    plan = make_plan(PlanRequest(model_id="d", strategy=PlanStrategy.local_only), dense, [gpu(10.27)], [], 10 * GiB,
                     cpu(6))
    assert plan.tier != "moe_offload" and plan.moe is None
    assert moe_offload_layout(dense, [gpu(10.27)], cpu(6), keep_all=False)[2] is False


def test_peers_first_take_whole_layers_and_the_pattern_names_only_host_layers():
    shape, peer = oss120(), mac(10)
    plan = make_plan(PlanRequest(model_id="oss", strategy=PlanStrategy.all_selected), shape, [gpu(10.27)], [peer],
                     10 * GiB, cpu(6))
    assert plan.feasible and plan.tier == "moe_offload" and plan.moe is not None
    cpu_a, mac_a, gpu_a = plan.assignments
    assert (cpu_a.device, mac_a.endpoint, gpu_a.device) == ("CPU", EP5, "CUDA0")
    assert mac_a.layers == list(range(len(mac_a.layers))) and mac_a.weight_bytes == sum(
        shape.layer_bytes[i] for i in mac_a.layers)  # whole layers: dense part and experts
    assert mac_a.fits and planner._assignment(shape, peer, [*mac_a.layers, len(mac_a.layers)], False).fits is False
    assert gpu_a.layers == list(range(len(mac_a.layers), 36)) and plan.tensor_split == [len(mac_a.layers), 37 - len(
        mac_a.layers)]
    host = set(gpu_a.layers)
    assert set(plan.moe.cpu_expert_layers) <= host
    rx = re.compile(plan.moe.override.removesuffix("=CPU"))
    for i in range(36):
        assert bool(rx.search(f"blk.{i}.ffn_down_exps.weight")) == (i in plan.moe.cpu_expert_layers)
    assert any("peers hold whole layers" in w for w in plan.warnings)


def test_peer_budget_respects_the_contributor_reserve_and_cap():
    shape = oss120()
    for node in (mac_node(free_gib=14, max_gib=12, reserve_gib=2), mac_node(free_gib=9, max_gib=12, reserve_gib=4)):
        (peer,) = remote_budgets(node)  # min(cap share, free - reserve): what admission allows
        layout, _, ok, _ = moe_offload_layout(shape, [peer, gpu(10.27)], cpu(6), keep_all=True)
        assert ok and layout[1].endpoint is not None and layout[1].total_bytes_est <= peer.budget_bytes


def test_auto_compares_with_and_without_peers_and_keeps_both_candidates():
    plan = make_plan(PlanRequest(model_id="oss", strategy=PlanStrategy.auto), oss120(), [gpu(10.27)], [mac(10)],
                     10 * GiB, cpu(6))
    assert plan.feasible and plan.tier == "moe_offload"
    labels = [c.label for c in plan.candidates if "MoE experts" in c.label]
    assert len(labels) == 2 and any("peer device" in label for label in labels)
    assert any(w.startswith("MoE placement") and "estimated decode" in w for w in plan.warnings)


def test_run_together_needs_a_layer_for_every_peer():
    layout, spec, ok, why = moe_offload_layout(oss120(), [mac(0.5), gpu(10.27)], cpu(6), keep_all=True)
    assert not ok and spec is None and "cannot hold one whole layer" in why


def test_a_gpu_that_cannot_hold_the_dense_part_is_refused_with_its_numbers():
    _, _, ok, why = moe_offload_layout(oss120(dense_mib=400), [gpu(10.27)], cpu(6), keep_all=False)
    assert not ok and "cannot hold the dense part" in why and "CUDA0" in why


# --------------------------------------------------------------------------------------------------- adapter


def moe_plan(peer: bool = False):
    plan = make_plan(PlanRequest(model_id="oss", strategy=PlanStrategy.all_selected if peer else
                                 PlanStrategy.local_only), oss120(), [gpu(10.27)], [mac(10)] if peer else [],
                     10 * GiB, cpu(6))
    assert plan.tier == "moe_offload"
    return DeploymentManager._llama_names(plan, {EP5: 1}) if peer else plan


def args_for(plan) -> list[str]:
    adapter = object.__new__(LlamaRpcAdapter)  # build_args reads only the spec
    return adapter.build_args(DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m"), 9000)


@pytest.mark.parametrize("peer", [False, True])
def test_build_args_keep_cpu_experts_file_backed_and_every_layer_offloaded(peer):
    plan = moe_plan(peer)
    args = args_for(plan)
    value = args[args.index("-ot") + 1]
    assert value == plan.moe.override and value.endswith("=CPU") and "," not in value
    assert args[args.index("--load-mode") + 1] == "mmap" and args[args.index("-ngl") + 1] == "all"
    assert {"--no-repack", "--no-host", "--no-op-offload"} <= set(args) and "--n-cpu-moe" not in args
    assert args[args.index("--cache-ram") + 1] == "0" and args[args.index("--fit") + 1] == "off"
    assert args[args.index("--device") + 1] == ",".join(plan.device_list)
    assert ("--rpc" in args) == peer


def test_build_args_refuse_a_pattern_that_names_a_peer_layer():
    plan = moe_plan(peer=True)
    bad = plan.moe.model_copy(update={"cpu_expert_layers": [0, *plan.moe.cpu_expert_layers]})
    bad = bad.model_copy(update={"override": gguf.expert_override_pattern(bad.cpu_expert_layers) + "=CPU"})
    with pytest.raises(EngineAdapterError, match="host-side"):
        moe_offload_args(plan.model_copy(update={"moe": bad}))
    with pytest.raises(EngineAdapterError):
        moe_offload_args(plan.model_copy(update={"moe": plan.moe.model_copy(update={"override": "blk=CPU"})}))


def load_log(plan, *, cpu_buft: str = "CPU", mapped_mib: float | None = None, gpu_mib: float | None = None,
             skip: int | None = None, extra_buffer: str = "", offloaded: str = "37/37") -> str:
    """A -lv 4 load log in llama.cpp b11770's formats for `plan` (override lines from the llama.dll format string)."""
    moe = plan.moe
    gpu_a = next(a for a in plan.assignments if a.device == "CUDA0")
    lines = [f"0.01.000.000 D tensor blk.{i}.ffn_{p}_exps.weight (531 MiB mxfp4) buffer type overridden to {cpu_buft}"
             for i in moe.cpu_expert_layers if i != skip for p in ("gate", "up", "down")]
    lines += [f"0.01.000.001 D tensor blk.{i}.ffn_{p}_exps.bias (1 MiB f32) buffer type overridden to {cpu_buft}"
              for i in moe.cpu_expert_layers if i != skip for p in ("gate", "up", "down")]
    mapped = mapped_mib if mapped_mib is not None else (moe.cpu_expert_bytes + 620 * MiB) / MiB
    lines += ["0.04.000.000 I load_tensors: offloading output layer to GPU",
              f"0.04.000.001 I load_tensors: offloaded {offloaded} layers to GPU",
              f"0.04.000.002 I load_tensors:   CPU_Mapped model buffer size = {mapped:.2f} MiB",
              f"0.04.000.003 I load_tensors:        CUDA0 model buffer size = "
              f"{gpu_mib if gpu_mib is not None else gpu_a.weight_bytes / MiB - 3:.2f} MiB",
              *([extra_buffer] if extra_buffer else []),
              "0.09.000.000 I llama_kv_cache:      CUDA0 KV buffer size =   288.00 MiB",
              "0.09.000.001 I sched_reserve:      CUDA0 compute buffer size =   159.38 MiB",
              "0.09.000.002 I sched_reserve:  CUDA_Host compute buffer size =    44.02 MiB"]
    return "\n".join(lines)


def check(plan, text: str) -> tuple[bool, str]:
    return check_moe_placement(plan, parse_placement(text), parse_overrides(text))


def test_load_log_with_experts_in_cpu_mapped_is_verified():
    plan = moe_plan()
    ok, detail = check(plan, load_log(plan))
    assert ok and detail.startswith("moe_offload verified") and "CPU_Mapped" in detail
    assert len(parse_overrides(load_log(plan))) == 6 * len(plan.moe.cpu_expert_layers)


@pytest.mark.parametrize("change, problem", [
    ({"skip": 20}, "missing [20]"),
    ({"cpu_buft": "CUDA0"}, "unexpected overrides"),
    ({"mapped_mib": 1000.0}, "CPU_Mapped spans"),
    ({"extra_buffer": "0.04.000.004 I load_tensors:   CPU_REPACK model buffer size = 40000.00 MiB"}, "private host"),
    ({"gpu_mib": 20000.0}, "CUDA0 holds"),
    ({"gpu_mib": 100.0}, "CUDA0 holds 100 MiB"),
    ({"offloaded": "30/37"}, "offloaded 30/37"),
])
def test_load_log_that_differs_from_the_plan_is_not_verified(change, problem):
    plan = moe_plan()
    ok, detail = check(plan, load_log(plan, **change))
    assert not ok and problem in detail


def test_observe_uses_the_moe_check_and_peer_participation(monkeypatch):
    plan = moe_plan(peer=True)
    mac_a = next(a for a in plan.assignments if a.endpoint)
    text = load_log(plan) + f"\n0.04.000.009 I load_tensors: RPC0[{EP5}] model buffer size = 8000.00 MiB"
    adapter = object.__new__(LlamaRpcAdapter)
    monkeypatch.setattr(adapter, "read_log", lambda _h: text, raising=False)
    spec = DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m")
    report = asyncio.run(adapter.observe(None, spec, ValidationResult(ok=True, problems=[],  # type: ignore[arg-type]
                                                                        rpc_device_map={EP5: [mac_a.device]})))
    assert report.verified and "moe_offload verified" in report.detail and "RPC0" in report.detail


# --------------------------------------------------------------------------------------------------- optimizer


def moe_manifest(n_layer: int = 36, dense_mib: float = 60, expert_mib: float = 1614) -> ModelManifest:
    base = make_manifest(n_layer=n_layer, block_mib=dense_mib + expert_mib, head_mib=620, n_embd=2880, embd_mib=620)
    blocks = tuple(replace(u, expert_bytes=int(expert_mib * MiB)) for u in base.units[:n_layer])
    return replace(base, units=(*blocks, base.head), expert_count=128, expert_used_count=4, n_head=64, n_head_kv=8,
                   key_length=64, value_length=64)


def test_capacity_optimizer_steps_aside_for_moe_and_the_selector_keeps_the_planner_plan(tmp_path):
    m = moe_manifest()
    svc = make_service(tmp_path, m, local_node(), ram_gib=10.0)
    with pytest.raises(CapacityUnavailable) as exc:
        svc.plan(PlanRequest(model_id=MODEL, strategy=PlanStrategy.local_only))
    assert exc.value.code == "moe"
    node = local_node()  # the node the service sees
    inp = Inputs(req=PlanRequest(model_id=MODEL, strategy=PlanStrategy.local_only), artifact=artifact(),
                 shape=bridge.model_shape(m, 4096), local=local_budgets(node), remote=[], cpu=cpu_budget(node),
                 host_ram_free=10 * GiB, capacity=svc.plan)
    plan = choose_plan(inp)
    assert plan.tier == "moe_offload" and plan.feasible
    assert any(w.startswith("capacity optimizer not used (moe)") for w in plan.warnings)


def test_explain_shows_the_planner_moe_plan_and_its_estimates(tmp_path):
    svc = make_service(tmp_path, moe_manifest(), local_node(), [mac_node()], ram_gib=10.0)
    body = svc.explain(MODEL)
    assert body["plan"]["tier"] == "moe_offload" and body["feasible"] and body["best"] is None
    assert body["explain"][0].startswith("mixture-of-experts model") and body["candidates"]
    assert any("page-fault read rate" in a for a in body["assumptions"])
    assert body["plan"]["moe"]["provenance"] == "estimate"


def test_the_replan_evaluator_proposes_nothing_for_an_moe_deployment(tmp_path):
    m = moe_manifest()
    svc = make_service(tmp_path, m, local_node(), ram_gib=10.0)
    plan = svc.explain(MODEL)["plan"]
    dep = Deployment(id="d" * 32, plan=Plan.model_validate(plan), state=DeploymentState.ready, history=[],
                     created_at=NOW, revision=1)
    assert asyncio.run(svc.evaluate(dep, "speed")) is None


# --------------------------------------------------------------------------------------------------- review regressions


def info_log(plan, *, gpu_extra: float = 0.0) -> str:
    """The load log llama-server really writes at -lv 4: its -lv help says 4 is trace and 5 is debug, and the -lv 4
    logs of this host (tests/fixtures/*_b11770.log) carry no debug lines, so the debug-level -ot override lines are
    absent. Only the WARN that a CPU override runs under mmap and the INFO buffer lines remain."""
    gpu_a = next(a for a in plan.assignments if a.device == "CUDA0")
    mapped_mib = (plan.moe.cpu_expert_bytes + 620 * MiB) / MiB
    return "\n".join([
        "0.00.191.914 I cmn  common_param: common_params_print_info: verbosity = 4 (adjust with the `-lv N` CLI arg)",
        "0.01.147.276 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
        "0.01.150.000 W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using "
        "--load-mode none for better performance",
        "0.04.186.052 I load_tensors: offloaded 37/37 layers to GPU",
        f"0.04.186.060 I load_tensors:   CPU_Mapped model buffer size = {mapped_mib:.2f} MiB",
        f"0.04.186.063 I load_tensors:        CUDA0 model buffer size = {gpu_a.weight_bytes / MiB + gpu_extra:.2f} MiB",
        "0.09.633.539 I sched_reserve:      CUDA0 compute buffer size =   159.38 MiB"])


def test_a_real_lv4_load_log_without_debug_override_lines_is_verified_by_buffer_sizes():
    plan = moe_plan()
    text = info_log(plan)
    assert parse_overrides(text) == {} and args_for(plan)[args_for(plan).index("-lv") + 1] == "4"
    ok, detail = check(plan, text)
    assert ok, detail
    assert "debug" in detail and "buffer sizes" in detail


def test_without_override_lines_experts_left_on_the_gpu_still_fail_the_check():
    plan = moe_plan()
    one_layer = plan.moe.cpu_expert_bytes / len(plan.moe.cpu_expert_layers) / MiB
    ok, detail = check(plan, info_log(plan, gpu_extra=one_layer))  # the -ot pattern missed one planned layer
    assert not ok and "CUDA0 holds" in detail


def test_a_peer_moe_deployment_with_a_real_lv4_log_is_not_failed_as_no_peer_work(monkeypatch):
    plan = moe_plan(peer=True)
    mac_a = next(a for a in plan.assignments if a.endpoint)
    text = info_log(plan) + f"\n0.04.186.064 I load_tensors: RPC0[{EP5}] model buffer size = 8000.00 MiB"
    adapter = object.__new__(LlamaRpcAdapter)
    monkeypatch.setattr(adapter, "read_log", lambda _h: text, raising=False)
    spec = DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m")
    report = asyncio.run(adapter.observe(None, spec, ValidationResult(ok=True, problems=[],  # type: ignore[arg-type]
                                                                        rpc_device_map={EP5: [mac_a.device]})))
    assert report.verified, report.detail  # deployments raises no_peer_work for an unverified plan with peers


def test_load_budget_covers_the_weights_uploaded_to_peers():
    from hypercompute import config
    from hypercompute.adapters import llama_rpc

    adapter = object.__new__(LlamaRpcAdapter)
    local, peer = moe_plan(), moe_plan(peer=True)
    sent = sum(a.weight_bytes for a in peer.assignments if a.endpoint)
    assert adapter.load_timeout_s(local) == config.LOAD_TIMEOUT_S  # nothing crosses the network
    assert adapter.load_timeout_s(peer) == pytest.approx(min(
        llama_rpc.MAX_LOAD_S, config.LOAD_TIMEOUT_S + sent / llama_rpc.PEER_UPLOAD_FLOOR_BPS))
    assert sent / 1.59e6 < adapter.load_timeout_s(peer) - config.LOAD_TIMEOUT_S  # the measured relay goodput fits
