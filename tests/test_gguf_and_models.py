"""GGUF header parsing on a synthetic file, plus a hardware-marked check on the real downloaded model."""

import struct
from pathlib import Path

import pytest
from hypercompute import gguf


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_u32(key, v):
    return _str(key) + struct.pack("<I", 4) + struct.pack("<I", v)


def _kv_str(key, v):
    return _str(key) + struct.pack("<I", 8) + _str(v)


def _kv_str_array(key, items):
    out = _str(key) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(items))
    return out + b"".join(_str(i) for i in items)


def write_gguf(path: Path, tensors: list[tuple[str, int]], tied: bool = True, extra: list[bytes] = ()) -> None:
    kvs = [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 2),
           _kv_u32("llama.embedding_length", 64), _kv_u32("llama.attention.head_count", 4),
           _kv_u32("llama.attention.head_count_kv", 2), _kv_u32("llama.context_length", 2048),
           _kv_u32("general.file_type", 15), _kv_str_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(100)]),
           _kv_str("tokenizer.chat_template", "{{ messages }}"), *extra]
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    header += b"".join(kvs)
    offset = 0
    infos = b""
    for name, size in tensors:
        infos += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", size) + struct.pack("<I", 0) + struct.pack(
            "<Q", offset)
        offset += (size + 31) // 32 * 32
    body = header + infos
    pad = (32 - len(body) % 32) % 32
    path.write_bytes(body + b"\0" * pad + b"\1" * offset)


def test_synthetic_gguf_layer_sizes(tmp_path):
    p = tmp_path / "tiny.gguf"
    write_gguf(p, [("token_embd.weight", 640), ("blk.0.attn_q.weight", 256), ("blk.0.ffn_up.weight", 512),
                   ("blk.1.attn_q.weight", 256), ("output_norm.weight", 64)])
    s = gguf.summarize([p])
    assert s.architecture == "llama" and s.n_layer == 2 and s.n_head_kv == 2 and s.vocab_size == 100
    assert s.layer_bytes == [768, 256]
    # the head is the tied embeddings plus output_norm (llama.cpp's output layer), so nothing is left unplaced
    assert s.token_embd_bytes == 640 and s.output_tied and s.output_bytes == 640 + 64 and s.other_bytes == 0
    assert s.file_type == "Q4_K_M" and s.chat_template_present
    assert s.key_length is None and s.value_length is None  # absent keys: planner falls back to n_embd/n_head


def test_gguf_attention_key_and_value_length_are_read(tmp_path):
    p = tmp_path / "qwen3ish.gguf"
    write_gguf(p, [("blk.0.attn_q.weight", 256), ("blk.1.attn_q.weight", 256)],
               extra=[_kv_u32("llama.attention.key_length", 128), _kv_u32("llama.attention.value_length", 96)])
    s = gguf.summarize([p])
    assert s.key_length == 128 and s.value_length == 96 and s.n_embd // s.n_head == 16


def test_not_a_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(gguf.GgufError):
        gguf.read_file(p)


@pytest.mark.hardware
def test_real_qwen_05b_summary():
    from hypercompute.services.models import ModelService

    svc = ModelService()
    s = svc.summary("qwen2.5-0.5b-instruct-q4_k_m")
    if s is None:
        pytest.skip("model not downloaded on this host")
    assert s.n_layer == 24 and s.n_head_kv == 2 and s.n_embd == 896 and not s.output_tied
    # llama.cpp b11770 on the G14 reported "CUDA0 model buffer size = 374 MiB" (all layers + output)
    # and "CPU_Mapped model buffer size = 89.26 MiB" (token embeddings) for this exact file.
    resident_mib = (sum(s.layer_bytes) + s.output_bytes) / 2**20
    assert abs(resident_mib - 374) < 2 and abs(s.token_embd_bytes / 2**20 - 89.26) < 1
