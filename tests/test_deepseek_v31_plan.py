"""DeepSeek-V3.1 UD-TQ1_0 (deepseek2) planned from its real GGUF header: what the reader and the manifest take from it,
which tensors count as routed experts, the MLA KV estimate, and moe_offload plans on a G14-like host alone and with
three Mac peers. Plan only: nothing is loaded and no hardware is touched.

tests/fixtures/deepseek_v31_ud_tq1_0_header.json is the header of DeepSeek-V3.1-UD-TQ1_0.gguf
(unsloth/DeepSeek-V3.1-GGUF at feb73eff, 170499768256 bytes), taken from the file's first 16 MiB: every metadata key
except the tokenizer arrays and the chat template, each tensor kind's dims, and each tensor's ggml type in file order.
The file has no padding between tensors (every size is a multiple of the 32-byte alignment), so the offsets follow from
the sizes; the fixture rebuilt the real table (names, dims, types, offsets) exactly when it was made. The tests write
that header alone and let Path.stat report the length the complete file would have, so gguf.summarize and the manifest
read the real table.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path

import pytest
from hypercompute import gguf
from hypercompute.adapters.llama_rpc import moe_offload_args
from hypercompute.capacity.manifest import build_manifest
from hypercompute.schemas import DeviceAssignment, GgufSummary, Plan, PlanRequest, PlanStrategy
from hypercompute.services.models import load_manifest
from hypercompute.services.planner import KV_BYTES_PER_ELEM, ModelShape, make_plan
from hypercompute.units import GiB
from test_capacity_manifest import Tensor
from test_split_order import EP5, EP6, cpu, gpu, mac

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v31_ud_tq1_0_header.json"
DOC = json.loads(FIXTURE.read_text(encoding="utf-8"))
SOURCE = DOC["source"]
DATA_BYTES = SOURCE["size_bytes"] - SOURCE["data_start"]  # every tensor byte of the real file: 170494459904
TYPE_IDS = {name: tid for tid, (name, _, _) in gguf.GGML_TYPES.items()}
N_LAYER, DENSE_LEAD = 61, 3  # deepseek2.block_count, deepseek2.leading_dense_block_count
MOE_LAYERS = range(DENSE_LEAD, N_LAYER)
ROUTED = ("ffn_down_exps.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight")
SHARED = ("ffn_down_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight")
MLA = ("attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_q_a.weight",
       "attn_q_a_norm.weight", "attn_q_b.weight", "attn_output.weight")
EP7 = "10.0.0.7:47052"
CTX = 4096  # the catalog's default_ctx


def tensors() -> list[Tensor]:
    """The real tensor table in file order: the head and token embeddings, then each block's tensors."""
    head = [Tensor(name, tuple(DOC["dims"][name]), TYPE_IDS[t]) for name, t in DOC["head"]]
    return head + [Tensor(f"blk.{i}.{suffix}", tuple(DOC["dims"][suffix]), TYPE_IDS[t])
                   for i, block in enumerate(DOC["blocks"]) for suffix, t in block.items()]


TENSORS = tensors()
NBYTES = {t.name: t.data_bytes for t in TENSORS}


def block_bytes(i: int, suffixes: tuple[str, ...] | None = None) -> int:
    """Bytes of block i's tensors (only those with the given suffixes, when given), from the fixture alone."""
    return sum(n for name, n in NBYTES.items() if name.startswith(f"blk.{i}.")
               and (suffixes is None or name.removeprefix(f"blk.{i}.") in suffixes))


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _value(vtype: int, value: object) -> bytes:
    if vtype == 4:
        return struct.pack("<I", value)
    if vtype == 6:
        return struct.pack("<f", value)
    if vtype == 7:
        return struct.pack("<?", value)
    if vtype == 8:
        return _s(str(value))
    if vtype == 9:
        etype, items = value  # type: ignore[misc]
        return struct.pack("<IQ", etype, len(items)) + b"".join(_value(etype, v) for v in items)
    raise TypeError(f"no writer for GGUF value type {vtype}")


def write_header(path: Path) -> int:
    """The GGUF header of the fixture (no tensor data) at `path`; returns the length the complete file would have."""
    kvs = b"".join(_s(k) + struct.pack("<I", t) + _value(t, v) for k, t, v in DOC["metadata"])
    body = b"GGUF" + struct.pack("<IQQ", 3, len(TENSORS), len(DOC["metadata"])) + kvs
    offset = 0
    for t in TENSORS:
        body += _s(t.name) + struct.pack("<I", len(t.dims)) + b"".join(struct.pack("<Q", d) for d in t.dims)
        body += struct.pack("<IQ", t.gtype, offset)
        assert t.data_bytes % 32 == 0, t.name  # no padding in the real file
        offset += t.data_bytes
    data_start = (len(body) + 31) // 32 * 32
    path.write_bytes(body + b"\0" * (data_start - len(body)))
    return data_start + offset


@pytest.fixture
def header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The header-only GGUF, which Path.stat reports at its complete length (nothing reads past the header)."""
    path = tmp_path / SOURCE["file"]
    size = write_header(path)
    target = os.path.normcase(os.path.realpath(path))
    real_stat = Path.stat

    def stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        st = real_stat(self, follow_symlinks=follow_symlinks)
        if os.path.normcase(os.path.realpath(self)) != target:
            return st
        fields = list(st)  # the ten sequence fields; st_size is the seventh
        fields[6] = size
        return os.stat_result(fields, {"st_mtime_ns": st.st_mtime_ns})

    monkeypatch.setattr(Path, "stat", stat)
    return path


@pytest.fixture
def summary(header: Path) -> GgufSummary:
    return gguf.summarize([header])


# --------------------------------------------------------------------------------------------------- header facts


def test_the_fixture_rebuilds_every_tensor_byte_of_the_real_file():
    assert len(TENSORS) == 1086 and sum(NBYTES.values()) == DATA_BYTES == 170_494_459_904
    names = {gguf.GGML_TYPES[t.gtype][0] for t in TENSORS}
    assert names == {"F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_XXS", "IQ1_S"}
    assert "TQ1_0" not in names  # the UD-TQ1_0 name is unsloth's label, not a tensor type


def test_the_catalog_entry_pins_this_file():
    catalog = Path(__file__).resolve().parents[1] / "configs" / "models.example.yaml"
    art = next(a for a in load_manifest(catalog) if a.id == "deepseek-v3.1-ud-tq1_0")
    (f,) = art.files
    assert (art.repo, art.revision, art.license) == (SOURCE["repo"], SOURCE["revision"], "mit")
    assert (art.family, art.quantization, art.logical_id, art.role) == ("deepseek2", "UD-TQ1_0", "deepseek-v3.1",
                                                                      "capacity_candidate")
    assert (art.format, art.default_ctx, art.max_output_tokens) == ("gguf", CTX, 1024)
    assert (f.name, f.size_bytes, f.sha256) == (SOURCE["file"], SOURCE["size_bytes"], SOURCE["sha256"])
    assert f.url == f"https://huggingface.co/{art.repo}/resolve/{art.revision}/{f.name}" and f.path is None
    assert art.params_b == round(sum(math.prod(t.dims) for t in TENSORS) / 1e9, 2) == 671.03  # the HF GGUF total


def test_summary_reads_the_deepseek2_header(summary: GgufSummary):
    s = summary
    assert (s.architecture, s.n_layer, s.n_embd, s.n_head, s.n_head_kv) == ("deepseek2", 61, 7168, 128, 1)
    assert (s.key_length, s.value_length, s.context_length) == (576, 512, 163840)  # MLA: 512 latent + 64 rope dims
    assert (s.expert_count, s.expert_used_count, s.expert_shared_count) == (256, 8, 1)
    assert s.file_type == "IQ1_S" and s.tensor_count == 1086  # general.file_type 24, whatever the file name says
    assert s.total_tensor_bytes == DATA_BYTES
    assert sum(s.layer_bytes) + s.token_embd_bytes + s.output_bytes + s.other_bytes == DATA_BYTES
    assert s.layer_bytes == [block_bytes(i) for i in range(N_LAYER)]
    assert s.layer_bytes[-1] == block_bytes(60) > 0  # the last tensor in the file is sized from the file length


def test_head_and_token_embeddings_are_separate(summary: GgufSummary):
    assert not summary.output_tied and summary.other_bytes == 0
    assert summary.output_bytes == NBYTES["output.weight"] + NBYTES["output_norm.weight"]
    assert summary.token_embd_bytes == NBYTES["token_embd.weight"]


def test_only_routed_experts_count_as_expert_bytes(summary: GgufSummary):
    e = summary.layer_expert_bytes
    assert len(e) == N_LAYER and e[:DENSE_LEAD] == [0, 0, 0]  # blocks 0-2 have a dense FFN and no experts
    assert all(e[i] == block_bytes(i, ROUTED) > 0 for i in MOE_LAYERS)
    # attention (MLA), norms, the router, exp_probs_b and the shared expert are the dense part of each MoE block
    dense = MLA + SHARED + ("attn_norm.weight", "ffn_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias")
    assert all(summary.layer_bytes[i] - e[i] == block_bytes(i, dense) for i in MOE_LAYERS)
    assert all(not gguf.is_expert_tensor(f"blk.{i}.{s}") for i in range(N_LAYER) for s in DOC["blocks"][i]
               if s not in ROUTED)
    assert sum(e) / DATA_BYTES > 0.9  # the routed experts are most of the file


def test_the_override_pattern_names_exactly_the_routed_experts_of_its_layers():
    for layers in (list(MOE_LAYERS), [3, 4, 5, 17, 30, 59, 60], [10]):
        rx = re.compile(gguf.expert_override_pattern(layers))  # llama.cpp searches each name (std::regex_search)
        hits = {t.name for t in TENSORS if rx.search(t.name)}
        assert hits == {f"blk.{i}.{s}" for i in layers for s in ROUTED}


def test_manifest_matches_the_summary(header: Path, summary: GgufSummary):
    m = build_manifest("deepseek-v3.1-ud-tq1_0", [header])
    assert m.is_moe and (m.expert_count, m.expert_used_count, m.expert_shared_count) == (256, 8, 1)
    assert [m.block(i).expert_bytes for i in range(N_LAYER)] == summary.layer_expert_bytes
    assert [m.block(i).weight_bytes for i in range(N_LAYER)] == summary.layer_bytes
    assert m.head.weight_bytes == summary.output_bytes and m.head.expert_bytes == 0
    assert m.token_embd_bytes == summary.token_embd_bytes and m.weight_bytes == DATA_BYTES
    assert (m.n_head_kv, m.key_length, m.value_length) == (1, 576, 512)


# --------------------------------------------------------------------------------------------------- MLA KV cache


def test_the_kv_estimate_is_the_latent_cache_with_a_v_row_on_top(summary: GgufSummary, header: Path):
    """llama.cpp b11770 (src/llama-kv-cache.cpp: has_v = !is_mla; src/models/deepseek2.cpp reads key_length_mla) caches
    one K row of key_length (576: the 512 latent and 64 rope dims) per token and layer for an MLA model and no V. The
    planner's standard K+V formula adds a 512-wide V row: 1.89x the real cache, conservative and small next to the
    weights, and far below the 128-head cache of the decompressed attention."""
    shape = ModelShape.from_gguf(summary, CTX)
    assert shape is not None
    latent = CTX * 1 * 576 * KV_BYTES_PER_ELEM
    assert shape.kv_per_layer == CTX * 1 * (576 + 512) * KV_BYTES_PER_ELEM == build_manifest(
        "deepseek-v3.1-ud-tq1_0", [header]).kv_bytes_per_layer(CTX)
    assert latent < shape.kv_per_layer < 2 * latent
    assert shape.kv_per_layer * 30 < CTX * 128 * (192 + 128) * KV_BYTES_PER_ELEM  # 128 heads of K 192 and V 128
    assert shape.kv_per_layer * N_LAYER < 0.6 * GiB  # the whole cache at 4096 tokens


# --------------------------------------------------------------------------------------------------- plans


G14_GPU_GIB, G14_RAM_GIB = 11.1, 31.0
MACS = (mac(10.67, EP5), mac(10.67, EP6), mac(11.84, EP7))  # Metal working-set limits: two at 10.67, mac-azim 11.84


def plan_for(summary: GgufSummary, strategy: PlanStrategy, peers: tuple = ()) -> Plan:
    shape = ModelShape.from_gguf(summary, CTX)
    assert shape is not None and shape.is_moe
    return make_plan(PlanRequest(model_id="deepseek-v3.1-ud-tq1_0", strategy=strategy, ctx_size=CTX), shape,
                     [gpu(G14_GPU_GIB)], list(peers), int(G14_RAM_GIB * GiB), cpu(G14_RAM_GIB))


def placed_bytes(plan: Plan, s: GgufSummary) -> list[int]:
    """Weight bytes the plan places for each layer, from each assignment's layers and the CPU expert layers."""
    cpu_layers = set(plan.moe.cpu_expert_layers) if plan.moe else set()
    out = [0] * N_LAYER
    for a in plan.assignments:
        if a.device == "CPU":
            for i in cpu_layers:
                out[i] += s.layer_expert_bytes[i]
            continue
        for i in a.layers:
            out[i] += s.layer_bytes[i] - (s.layer_expert_bytes[i] if a.endpoint is None and i in cpu_layers else 0)
    return out


def check_moe_plan(plan: Plan, s: GgufSummary) -> tuple[DeviceAssignment, DeviceAssignment]:
    """The checks every moe_offload plan of this model passes; returns (CPU, local GPU) assignments."""
    assert plan.feasible and plan.tier == "moe_offload" and plan.moe is not None, plan.reasons
    cpu_a, gpu_a = plan.assignments[0], plan.assignments[-1]
    assert (cpu_a.device, cpu_a.layers, gpu_a.device, gpu_a.holds_output) == ("CPU", [], "CUDA0", True)
    assert all(a.fits and a.total_bytes_est <= a.budget_bytes for a in plan.assignments)
    layers = [i for a in plan.assignments for i in a.layers]
    assert layers == list(range(N_LAYER))  # every layer once, in --device order
    # no layer gets more bytes than it has, and every byte of the file is placed once (head, embeddings included)
    assert placed_bytes(plan, s) == s.layer_bytes
    assert sum(a.weight_bytes for a in plan.assignments) + s.token_embd_bytes == DATA_BYTES
    # only routed experts of host MoE layers go to the CPU; the dense first blocks never do
    assert set(plan.moe.cpu_expert_layers) <= set(gpu_a.layers) & set(MOE_LAYERS)
    assert cpu_a.weight_bytes == plan.moe.cpu_expert_bytes == sum(s.layer_expert_bytes[i]
                                                                  for i in plan.moe.cpu_expert_layers)
    assert moe_offload_args(plan)[-1] == plan.moe.override
    rx = re.compile(plan.moe.override.removesuffix("=CPU"))
    assert {t.name for t in TENSORS if rx.search(t.name)} == {
        f"blk.{i}.{r}" for i in plan.moe.cpu_expert_layers for r in ROUTED}
    assert plan.moe.active_fraction == pytest.approx(8 / 256)
    return cpu_a, gpu_a


def test_g14_alone_keeps_every_routed_expert_on_the_cpu(summary: GgufSummary):
    plan = plan_for(summary, PlanStrategy.local_only)
    cpu_a, gpu_a = check_moe_plan(plan, summary)
    # the dense parts of all 61 blocks, the head and the KV take ~9.7 GiB of the 11.1: no MoE block's ~2.6 GB of
    # experts fits beside them
    assert plan.moe.cpu_expert_layers == list(MOE_LAYERS) and plan.moe.gpu_expert_bytes == 0
    assert gpu_a.weight_bytes == sum(summary.layer_bytes) - sum(summary.layer_expert_bytes) + summary.output_bytes
    assert plan.moe.cpu_expert_bytes > 0.9 * DATA_BYTES and plan.capacity_mode  # far more than 31 GiB of file cache
    assert plan.moe.ssd_bytes_per_token_est > 0 and plan.device_list == ["CUDA0"] and plan.tensor_split == [62]


def test_three_macs_take_whole_layers_first_and_the_g14_keeps_the_rest(summary: GgufSummary):
    plan = plan_for(summary, PlanStrategy.all_selected, MACS)
    cpu_a, gpu_a = check_moe_plan(plan, summary)
    peers = plan.assignments[1:-1]
    assert [a.endpoint for a in peers] == [EP5, EP6, EP7]  # ascending budgets: mac-azim (11.84 GiB) last
    assert all(a.layers and a.weight_bytes == sum(summary.layer_bytes[i] for i in a.layers) for a in peers)
    assert peers[0].layers[:DENSE_LEAD] == [0, 1, 2]  # the dense first blocks are the first peer's
    for a, budget in zip(peers, MACS, strict=True):  # each peer is full: one more whole layer would overflow it
        nxt = a.layers[-1] + 1
        assert a.total_bytes_est + summary.layer_bytes[nxt] + a.kv_bytes // len(a.layers) > budget.budget_bytes
    host = gpu_a.layers
    assert host == list(range(peers[-1].layers[-1] + 1, N_LAYER))
    assert plan.tensor_split == [len(a.layers) for a in peers] + [len(host) + 1]
    # with 13 dense parts off the GPU it has room for the experts of the smallest host block(s); the rest go to the CPU
    kept = sorted(set(host) - set(plan.moe.cpu_expert_layers))
    expert = summary.layer_expert_bytes
    assert kept and plan.moe.gpu_expert_bytes == sum(expert[i] for i in kept)
    assert max(expert[i] for i in kept) <= min(expert[i] for i in plan.moe.cpu_expert_layers)
    alone = plan_for(summary, PlanStrategy.local_only)
    assert plan.moe.cpu_expert_bytes < alone.moe.cpu_expert_bytes  # the peers' layers leave the host's file cache
