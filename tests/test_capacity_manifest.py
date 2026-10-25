"""Model manifest on synthetic GGUF files with real tensor dims and ggml types, plus a hardware check on the 0.5B."""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from hypercompute import gguf
from hypercompute.capacity import manifest
from hypercompute.capacity.manifest import ManifestError, build_manifest
from hypercompute.capacity.stream_config import ALIGN, layer_layouts
from hypercompute.capacity.types import ModelManifest

ALIGNMENT = 32  # GGUF default general.alignment
F32, F16, Q8_0, I32 = 0, 1, 8, 26
TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    gtype: int = F32
    size: int | None = None  # data bytes for a ggml type gguf.py does not know

    @property
    def data_bytes(self) -> int:
        n = gguf.tensor_nbytes(self.gtype, self.dims)
        if n is None:
            assert self.size is not None, f"{self.name} needs an explicit size"
            return self.size
        return n


def _align(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value: object) -> bytes:
    if isinstance(value, str):
        return _s(key) + struct.pack("<I", 8) + _s(value)
    if isinstance(value, float):
        return _s(key) + struct.pack("<If", 6, value)
    if isinstance(value, int):
        return _s(key) + struct.pack("<II", 4, value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return _s(key) + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_s(v) for v in value)
    if isinstance(value, list):
        return _s(key) + struct.pack("<IIQ", 9, 4, len(value)) + b"".join(struct.pack("<I", v) for v in value)
    raise TypeError(f"unsupported metadata value for {key}")


def write_gguf(path: Path, tensors: list[Tensor], meta: dict[str, object]) -> Path:
    """GGUF v3 with the given metadata (None values are left out) and tensors laid out in list order."""
    kvs = [_kv(k, v) for k, v in meta.items() if v is not None]
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    offset = 0
    for t in tensors:
        body += _s(t.name) + struct.pack("<I", len(t.dims)) + b"".join(struct.pack("<Q", d) for d in t.dims)
        body += struct.pack("<IQ", t.gtype, offset)
        offset = _align(offset + t.data_bytes)
    body += b"\0" * (_align(len(body)) - len(body))
    path.write_bytes(body + b"\x01" * offset)
    return path


def base_meta(changes: dict[str, object] | None = None) -> dict[str, object]:
    meta: dict[str, object] = {
        "general.architecture": "qwen2", "general.file_type": 15, "qwen2.block_count": 2,
        "qwen2.embedding_length": 64, "qwen2.attention.head_count": 4, "qwen2.attention.head_count_kv": 2,
        "qwen2.context_length": 4096, "qwen2.rope.freq_base": 1000000.0, "qwen2.rope.dimension_count": 16,
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": [f"t{i}" for i in range(100)],
        "tokenizer.chat_template": TEMPLATE,
    }
    return {**meta, **(changes or {})}


def split_meta(no: int, count: int, tensors: int | None = None) -> dict[str, object]:
    """gguf-split keys; every split carries all three (None leaves split.tensors.count out)."""
    return {"split.no": no, "split.count": count, "split.tensors.count": tensors}


TOKEN_EMBD = Tensor("token_embd.weight", (64, 100), F16)  # 12800 bytes
BLOCK0 = [Tensor("blk.0.attn_norm.weight", (64,)), Tensor("blk.0.attn_q.weight", (64, 64), F16),
          Tensor("blk.0.ffn_up.weight", (256, 64), Q8_0)]  # 256 + 8192 + 17408
BLOCK1 = [Tensor("blk.1.attn_norm.weight", (64,)), Tensor("blk.1.attn_q.weight", (64, 64), F16)]  # 256 + 8192
OUTPUT_NORM = Tensor("output_norm.weight", (64,))  # 256
OUTPUT = Tensor("output.weight", (64, 100), Q8_0)  # 6800
ROPE_FREQS = Tensor("rope_freqs.weight", (8,))  # 32, in no unit


def model_tensors(tied: bool = False) -> list[Tensor]:
    head = [OUTPUT_NORM] if tied else [OUTPUT_NORM, OUTPUT]
    return [TOKEN_EMBD, *BLOCK0, *BLOCK1, *head, ROPE_FREQS]


def nbytes(tensors: list[Tensor]) -> int:
    return sum(t.data_bytes for t in tensors)


def assert_buffers_match_layouts(m: ModelManifest, shards: list[Path]) -> None:
    layouts = layer_layouts(gguf.read_tensor_table(shards))
    assert [u.buffer_bytes for u in m.units] == [layouts[i].buffer_bytes for i in range(m.n_layer + 1)]
    assert all(u.buffer_bytes % ALIGN == 0 and u.buffer_bytes >= u.weight_bytes for u in m.units)


# --------------------------------------------------------------------------------------------------- units


def test_unequal_blocks_have_exact_bytes_and_type_layouts(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    m = build_manifest("m", [p])
    assert m.n_layer == 2 and [u.index for u in m.units] == [0, 1, 2]
    assert [u.kind for u in m.units] == ["block", "block", "head"]
    assert m.block(0).weight_bytes == nbytes(BLOCK0) == 25856
    assert m.block(1).weight_bytes == nbytes(BLOCK1) == 8448
    assert m.block(0).types == (("Q8_0", 17408), ("F16", 8192), ("F32", 256))  # bytes descending
    assert m.streamable
    assert_buffers_match_layouts(m, [p])


def test_untied_head_holds_output_tensors_and_token_embd_stays_outside(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(tied=False), base_meta())
    m = build_manifest("m", [p])
    assert not m.tied_embeddings
    assert m.head.kind == "head" and m.head.index == 2
    assert m.head.weight_bytes == nbytes([OUTPUT_NORM, OUTPUT]) == 7056
    assert m.token_embd_bytes == TOKEN_EMBD.data_bytes == 12800
    assert m.other_bytes == ROPE_FREQS.data_bytes == 32
    assert m.weight_bytes == nbytes(model_tensors(tied=False))
    assert_buffers_match_layouts(m, [p])


def test_tied_head_holds_token_embd(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(tied=True), base_meta())
    m = build_manifest("m", [p])
    assert m.tied_embeddings
    assert m.head.weight_bytes == nbytes([TOKEN_EMBD, OUTPUT_NORM]) == 13056
    assert m.head.types == (("F16", 12800), ("F32", 256))
    assert m.token_embd_bytes == 0  # it belongs to the head
    assert m.weight_bytes == nbytes(model_tensors(tied=True))
    assert_buffers_match_layouts(m, [p])


def test_block_straddling_two_shards(tmp_path):
    first = write_gguf(tmp_path / "m-00001-of-00002.gguf", [TOKEN_EMBD, *BLOCK0, BLOCK1[0]],
                       base_meta(split_meta(0, 2, 8)))
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", [BLOCK1[1], OUTPUT_NORM, OUTPUT], split_meta(1, 2, 8))
    m = build_manifest("m", [first, second])
    assert [Path(s.path).name for s in m.shards] == [first.name, second.name]
    assert [s.size for s in m.shards] == [first.stat().st_size, second.stat().st_size]
    assert all(s.sha256 is None for s in m.shards)
    assert m.block(1).weight_bytes == nbytes(BLOCK1)
    # one tensor of block 1 per shard, so its buffer is exactly one 4 KiB-aligned run in each shard
    runs = [-(-(t.offset + t.span) // ALIGN) * ALIGN - t.offset // ALIGN * ALIGN
            for t in gguf.read_tensor_table([first, second]) if t.layer == 1]
    assert len(runs) == 2 and m.block(1).buffer_bytes == sum(runs)
    assert m.weight_bytes == nbytes([TOKEN_EMBD, *BLOCK0, *BLOCK1, OUTPUT_NORM, OUTPUT])
    assert_buffers_match_layouts(m, [first, second])


def test_unknown_ggml_type_in_a_block_is_not_streamable(tmp_path):
    odd = Tensor("blk.1.ffn_up.weight", (64, 4), gtype=99, size=100)
    p = write_gguf(tmp_path / "m.gguf", [TOKEN_EMBD, *BLOCK0, *BLOCK1, odd, OUTPUT_NORM, OUTPUT], base_meta())
    m = build_manifest("m", [p])
    assert not m.streamable
    assert ("type99", 128) in m.block(1).types  # the file span (100 bytes padded to 128), not an invented size
    assert m.block(1).weight_bytes == nbytes(BLOCK1) + 128
    assert_buffers_match_layouts(m, [p])


def test_unknown_ggml_type_outside_units_keeps_streaming(tmp_path):
    odd = Tensor("rope_freqs.weight", (8,), gtype=99, size=40)
    p = write_gguf(tmp_path / "m.gguf", [TOKEN_EMBD, *BLOCK0, *BLOCK1, OUTPUT_NORM, OUTPUT, odd], base_meta())
    m = build_manifest("m", [p])
    assert m.streamable and m.other_bytes == 64  # last tensor: span runs to the end of the padded file


def test_unknown_ggml_type_in_the_head_is_not_streamable(tmp_path):
    odd = Tensor("output.bias", (100,), gtype=99, size=40)
    p = write_gguf(tmp_path / "m.gguf", [TOKEN_EMBD, *BLOCK0, *BLOCK1, OUTPUT_NORM, OUTPUT, odd], base_meta())
    m = build_manifest("m", [p])
    assert not m.streamable and ("type99", 64) in m.head.types
    assert_buffers_match_layouts(m, [p])


def test_tied_model_without_token_embd_has_no_head(tmp_path):
    # output_norm alone used to pass as a head; llama.cpp cannot load a model with neither projection
    p = write_gguf(tmp_path / "m.gguf", [*BLOCK0, *BLOCK1, OUTPUT_NORM], base_meta())
    with pytest.raises(ManifestError, match="neither output.weight nor token_embd.weight"):
        build_manifest("m", [p])


# --------------------------------------------------------------------------------------------------- splits


def two_splits(tmp_path: Path, tensors_count: int = 8) -> tuple[Path, Path]:
    """A gguf-split pair whose second split holds only output.weight."""
    first = write_gguf(tmp_path / "m-00001-of-00002.gguf", [TOKEN_EMBD, *BLOCK0, *BLOCK1, OUTPUT_NORM],
                       base_meta(split_meta(0, 2, tensors_count)))
    second = write_gguf(tmp_path / "m-00002-of-00002.gguf", [OUTPUT], split_meta(1, 2, tensors_count))
    return first, second


def test_partial_split_set_is_refused(tmp_path):
    first, second = two_splits(tmp_path)
    assert not build_manifest("m", [first, second]).tied_embeddings  # output.weight lives in split 2
    # without the split check the first file alone read as a tied model whose head is token_embd
    with pytest.raises(ManifestError, match="1 shards given but the first declares split.count 2"):
        build_manifest("m", [first])


def test_splits_out_of_order_are_refused(tmp_path):
    first, second = two_splits(tmp_path)
    with pytest.raises(ManifestError, match="shard 0 declares split.no 1"):
        build_manifest("m", [second, first])


def test_several_files_without_split_metadata_are_refused(tmp_path):
    first = write_gguf(tmp_path / "a.gguf", [TOKEN_EMBD, *BLOCK0, *BLOCK1, OUTPUT_NORM], base_meta())
    second = write_gguf(tmp_path / "b.gguf", [OUTPUT], {})
    with pytest.raises(ManifestError, match="2 shards given but the first declares no split.count"):
        build_manifest("m", [first, second])


def test_split_tensor_count_mismatch_is_refused(tmp_path):
    first, second = two_splits(tmp_path, tensors_count=9)
    with pytest.raises(ManifestError, match="split.tensors.count is 9 but the shards hold 8 tensors"):
        build_manifest("m", [first, second])


# --------------------------------------------------------------------------------------------------- metadata


def test_metadata_fields(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    m = build_manifest("qwen-tiny", [p], revision="abc123")
    assert (m.model_id, m.revision, m.architecture, m.file_type) == ("qwen-tiny", "abc123", "qwen2", "Q4_K_M")
    assert (m.n_embd, m.n_head, m.n_head_kv, m.context_length) == (64, 4, 2, 4096)
    assert (m.tokenizer_model, m.vocab_size) == ("gpt2", 100)
    assert m.rope == (("qwen2.rope.dimension_count", "16"), ("qwen2.rope.freq_base", "1000000.0"))
    assert m.chat_template_sha256 == hashlib.sha256(TEMPLATE.encode()).hexdigest()
    assert TEMPLATE not in repr(m)  # hashed, never stored


def test_kv_bytes_per_layer_defaults_to_head_dim(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    m = build_manifest("m", [p])
    assert (m.key_length, m.value_length) == (16, 16)  # n_embd / n_head
    assert m.kv_bytes_per_layer(1024) == 1024 * 2 * (16 + 16) * 2


def test_kv_bytes_per_layer_uses_explicit_key_and_value_length(tmp_path):
    changes = {"qwen2.attention.key_length": 128, "qwen2.attention.value_length": 96,
               "qwen2.attention.head_count_kv": None}
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta(changes))
    m = build_manifest("m", [p])
    assert (m.key_length, m.value_length, m.n_head_kv) == (128, 96, 4)  # head_count_kv absent: n_head
    assert m.kv_bytes_per_layer(512, bytes_per_elem=1) == 512 * 4 * (128 + 96)


def test_per_layer_head_count_kv_array_takes_the_maximum(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta({"qwen2.attention.head_count_kv": [1, 4]}))
    assert build_manifest("m", [p]).n_head_kv == 4  # conservative for KV sizing


def test_optional_metadata_absent(tmp_path):
    changes = {"general.file_type": None, "qwen2.context_length": None, "tokenizer.ggml.model": None,
               "tokenizer.ggml.tokens": None, "tokenizer.chat_template": None}
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta(changes))
    m = build_manifest("m", [p])
    assert m.file_type is None and m.context_length is None and m.tokenizer_model is None
    assert m.vocab_size is None and m.chat_template_sha256 is None


def test_per_layer_head_count_takes_the_head_size_from_the_first_layer(tmp_path):
    changes = {"qwen2.attention.head_count": [2, 4], "qwen2.attention.head_count_kv": None}
    m = build_manifest("m", [write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta(changes))])
    # llama.cpp: n_embd_head_k = n_embd / n_head(il=0) = 64 / 2, and n_head_kv defaults to n_head per layer (max 4).
    # Dividing by the maximum (16) would halve the KV estimate.
    assert (m.n_head, m.n_head_kv, m.key_length, m.value_length) == (4, 4, 32, 32)


@pytest.mark.parametrize(("changes", "message"), [
    ({"qwen2.context_length": float("nan")}, "context_length is nan, not a finite number"),
    ({"qwen2.embedding_length": float("inf")}, "embedding_length is inf, not a finite number"),
    ({"qwen2.attention.head_count_kv": [1] * 70}, "head_count_kv is an array longer than the header reader keeps"),
    ({"qwen2.embedding_length": 2}, r"no qwen2.attention.key_length and its default \(0\) is not positive"),
    ({"qwen2.attention.head_count": [0, 4]}, "has no qwen2.attention.key_length$"),
    ({"general.alignment": "abc"}, "cannot read GGUF m.gguf"),
    ({"general.alignment": [1, 2]}, "cannot read GGUF m.gguf"),
])
def test_malformed_metadata_is_a_manifest_error(tmp_path, changes, message):
    # these used to escape as a bare ValueError/OverflowError/TypeError, or to give key_length 0 (KV of 0 bytes),
    # or to replace an unread head_count_kv array with the n_head default
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta(changes))
    with pytest.raises(ManifestError, match=message):
        build_manifest("m", [p])


# --------------------------------------------------------------------------------------------------- fingerprint


def test_fingerprint_is_stable_and_independent_of_location(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = write_gguf(tmp_path / "a" / "m.gguf", model_tensors(), base_meta())
    b = write_gguf(tmp_path / "b" / "m.gguf", model_tensors(), base_meta())
    fa, fb = build_manifest("m", [a]).fingerprint, build_manifest("other-id", [b]).fingerprint
    assert fa == fb and len(fa) == 64
    manifest._cached_manifest.cache_clear()
    assert build_manifest("m", [a]).fingerprint == fa  # rebuilt from the file, not served from the cache


def _swap(tensors: list[Tensor], a: str, b: str) -> list[Tensor]:
    i, j = (next(k for k, t in enumerate(tensors) if t.name == n) for n in (a, b))
    out = list(tensors)
    out[i], out[j] = out[j], out[i]
    return out


def _changed(field: str, value: object, name: str) -> list[Tensor]:
    return [replace(t, **{field: value}) if t.name == name else t for t in model_tensors()]


@pytest.mark.parametrize(("label", "tensors", "same_offsets"), [
    ("type", _changed("gtype", I32, "blk.1.attn_norm.weight"), True),  # F32 -> I32, both 256 bytes
    ("dims", _changed("dims", (128, 32), "blk.1.attn_q.weight"), True),  # F16 64x64 -> 128x32, both 8192 bytes
    ("name", _changed("name", "blk.1.attn_k.weight", "blk.1.attn_q.weight"), True),  # same name length
    ("offset", _swap(model_tensors(), "blk.1.attn_norm.weight", "blk.1.attn_q.weight"), False),
])
def test_fingerprint_changes_with_each_tensor_field_alone(tmp_path, label, tensors, same_offsets):
    # Same file name and size, so only the field under test can move the fingerprint (the shard name and size
    # are part of it; an earlier version of this test changed both and proved nothing about tensors).
    (tmp_path / "base").mkdir()
    (tmp_path / "variant").mkdir()
    base = write_gguf(tmp_path / "base" / "m.gguf", model_tensors(), base_meta())
    variant = write_gguf(tmp_path / "variant" / "m.gguf", tensors, base_meta())
    assert variant.stat().st_size == base.stat().st_size
    offsets = [[(t.name, t.offset) for t in gguf.read_tensor_table([p])] for p in (base, variant)]
    assert ([o for _, o in offsets[0]] == [o for _, o in offsets[1]]) == same_offsets
    assert build_manifest("m", [variant]).fingerprint != build_manifest("m", [base]).fingerprint, label


def test_shard_hashes_are_recorded_and_enter_the_fingerprint(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    plain = build_manifest("m", [p])
    by_name = build_manifest("m", [p], shard_hashes={"m.gguf": digest.upper()})
    by_path = build_manifest("m", [p], shard_hashes={str(p): digest})
    assert by_name.shards[0].sha256 == digest
    assert by_name is by_path  # same normalized key, one cached manifest
    assert by_name.fingerprint != plain.fingerprint


def test_shard_hash_keys_accept_any_spelling_of_the_shard_path(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    p = write_gguf(tmp_path / "sub" / "m.gguf", model_tensors(), base_meta())
    digest = "ab" * 32
    forward = p.as_posix()  # forward slashes: on Windows str(Path(forward)) differs from the key as given
    assert build_manifest("m", [forward], shard_hashes={forward: digest}).shards[0].sha256 == digest
    monkeypatch.chdir(tmp_path)
    assert build_manifest("m", [p], shard_hashes={"sub/m.gguf": digest}).shards[0].sha256 == digest
    assert build_manifest("m", ["sub/m.gguf"], shard_hashes={str(p): digest}).shards[0].sha256 == digest


def test_two_hash_entries_for_one_shard_are_refused(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    with pytest.raises(ManifestError, match="more than one entry"):
        build_manifest("m", [p], shard_hashes={"m.gguf": "0" * 64, str(p): "0" * 64})


# --------------------------------------------------------------------------------------------------- cache


def test_cache_reuses_a_manifest_until_the_file_changes(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    first = build_manifest("m", [p])
    assert build_manifest("m", [p]) is first
    assert build_manifest("m", [p], revision="r2").revision == "r2"
    assert build_manifest("other", [p]).model_id == "other"
    before = p.stat()
    write_gguf(p, [replace(t, gtype=I32) if t.name == "blk.0.attn_norm.weight" else t for t in model_tensors()],
               base_meta())
    assert p.stat().st_size == before.st_size  # same size: only the mtime in the key can expose the change
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
    changed = build_manifest("m", [p])
    assert changed is not first and changed.fingerprint != first.fingerprint


def test_a_shard_changing_during_the_build_is_refused(tmp_path, monkeypatch):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    real = gguf.read_tensor_table

    def read_then_touch(paths):
        table = real(paths)
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        return table

    monkeypatch.setattr(manifest.gguf, "read_tensor_table", read_then_touch)
    with pytest.raises(ManifestError, match="changed on disk"):
        build_manifest("m", [p])


# --------------------------------------------------------------------------------------------------- errors


@pytest.mark.parametrize(("changes", "message"), [
    ({"qwen2.block_count": None}, "qwen2.block_count"),
    ({"general.architecture": None}, "general.architecture"),
    ({"qwen2.attention.head_count": None}, "head_count"),
    ({"qwen2.block_count": 0}, "positive"),
])
def test_missing_required_metadata(tmp_path, changes, message):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta(changes))
    with pytest.raises(ManifestError, match=message):
        build_manifest("m", [p])


def test_missing_blocks(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta({"qwen2.block_count": 3}))
    with pytest.raises(ManifestError, match=r"blocks \[2\] have no tensors"):
        build_manifest("m", [p])


def test_blocks_beyond_block_count(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta({"qwen2.block_count": 1}))
    with pytest.raises(ManifestError, match=r"blocks \[1\] lie beyond block_count 1"):
        build_manifest("m", [p])


def test_duplicate_tensor_in_one_shard(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [*model_tensors(), BLOCK0[1]], base_meta())
    with pytest.raises(ManifestError, match="blk.0.attn_q.weight"):
        build_manifest("m", [p])


def test_duplicate_tensor_across_shards(tmp_path):
    first = write_gguf(tmp_path / "a.gguf", model_tensors(), base_meta(split_meta(0, 2)))
    second = write_gguf(tmp_path / "b.gguf", [BLOCK1[0]], split_meta(1, 2))
    with pytest.raises(ManifestError, match="blk.1.attn_norm.weight"):
        build_manifest("m", [first, second])


def test_no_output_head(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [*BLOCK0, *BLOCK1], base_meta())
    with pytest.raises(ManifestError, match="no output head"):
        build_manifest("m", [p])


def test_not_a_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(ManifestError, match="not a GGUF"):
        build_manifest("m", [p])


def test_missing_and_empty_inputs(tmp_path):
    with pytest.raises(ManifestError, match="cannot read shard"):
        build_manifest("m", [tmp_path / "absent.gguf"])
    with pytest.raises(ManifestError, match="no GGUF shards"):
        build_manifest("m", [])
    with pytest.raises(ManifestError, match="model_id"):
        build_manifest("", [tmp_path / "absent.gguf"])
    with pytest.raises(ManifestError, match="not a regular file"):
        build_manifest("m", [tmp_path])
    with pytest.raises(ManifestError, match="not a single path"):
        build_manifest("m", str(tmp_path / "m.gguf"))  # a str is a Sequence of one-character "paths"


@pytest.mark.parametrize(("hashes", "message"), [
    ({"m.gguf": "not-hex"}, "sha256 hex digest"),
    ({"other.gguf": "0" * 64}, "not a shard of this model"),
    ({"": "0" * 64}, "non-empty strings"),
])
def test_bad_shard_hashes(tmp_path, hashes, message):
    p = write_gguf(tmp_path / "m.gguf", model_tensors(), base_meta())
    with pytest.raises(ManifestError, match=message):
        build_manifest("m", [p], shard_hashes=hashes)


def test_ambiguous_shard_hash_key(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write_gguf(tmp_path / "a" / "m.gguf", [TOKEN_EMBD, *BLOCK0], base_meta(split_meta(0, 2)))
    second = write_gguf(tmp_path / "b" / "m.gguf", [*BLOCK1, OUTPUT_NORM, OUTPUT], split_meta(1, 2))
    with pytest.raises(ManifestError, match="more than one shard"):
        build_manifest("m", [first, second], shard_hashes={"m.gguf": "0" * 64})


# --------------------------------------------------------------------------------------------------- hardware


@pytest.mark.hardware
def test_real_qwen_05b_manifest():
    model_dir = Path.home() / ".hypercompute" / "models" / "qwen2.5-0.5b-instruct-q4_k_m"
    shards = sorted(model_dir.glob("*.gguf"))
    if not shards:
        pytest.skip("qwen2.5-0.5b-instruct-q4_k_m is not downloaded on this host")
    m = build_manifest("qwen2.5-0.5b-instruct-q4_k_m", shards)
    table = gguf.read_tensor_table(shards)
    assert m.n_layer == 24 and len(m.units) == 25 and m.n_head_kv == 2 and m.n_embd == 896
    assert not m.tied_embeddings and m.streamable
    assert sum(u.weight_bytes for u in m.units) + m.token_embd_bytes + m.other_bytes == sum(
        t.nbytes for t in table)
    assert_buffers_match_layouts(m, shards)
    # hcstream itself reported slot_bytes 10887168 with all 24 blocks streamed on this file
    # (results/capacity/20261022T024357Z-stream-q05b-all24-k2/hcstream-stats-57453.json).
    assert max(u.buffer_bytes for u in m.units[:24]) == 10887168
