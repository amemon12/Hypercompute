"""GGUF reader limits: file type names, duplicate and empty tensor tables, general.alignment, per-layer arrays."""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path

import pytest
from hypercompute import gguf
from hypercompute.capacity.manifest import ManifestError, build_manifest

U8, I32, U32, F32V, BOOL, STR, ARR, U64 = 0, 5, 4, 6, 7, 8, 9, 10
_FMT = {U8: "<B", I32: "<i", U32: "<I", F32V: "<f", BOOL: "<?", U64: "<Q"}
LLAMA_H = Path(__file__).resolve().parents[1] / "native" / "hcstream" / "third_party" / "llama.cpp" / "llama.h"


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _value(vtype: int, value: object) -> bytes:
    return _s(value) if vtype == STR else struct.pack(_FMT[vtype], value)  # type: ignore[arg-type]


def kv(key: str, vtype: int, value: object, etype: int | None = None) -> bytes:
    """One metadata entry; for an array, `etype` is the element type and `value` the list of elements."""
    if vtype == ARR:
        assert etype is not None and isinstance(value, list)
        return _s(key) + struct.pack("<IIQ", ARR, etype, len(value)) + b"".join(_value(etype, v) for v in value)
    return _s(key) + struct.pack("<I", vtype) + _value(vtype, value)


def _up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def write_gguf(path: Path, kvs: list[bytes], tensors: list[tuple[str, tuple[int, ...]]] = (),
               layout_alignment: int = 32) -> int:
    """GGUF v3 with F32 tensors laid out in list order at `layout_alignment`; returns the header length (the data
    section starts at the next multiple of `layout_alignment`)."""
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    offset = 0
    for name, dims in tensors:
        body += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        body += struct.pack("<IQ", 0, offset)
        offset = _up(offset + 4 * math.prod(dims), layout_alignment)
    data_start = _up(len(body), layout_alignment)
    path.write_bytes(body + b"\0" * (data_start - len(body)) + b"\x01" * offset)
    return len(body)


def llama_kvs(blocks: int, *extra: bytes) -> list[bytes]:
    return [kv("general.architecture", STR, "llama"), kv("llama.block_count", U32, blocks),
            kv("llama.embedding_length", U32, 32), *extra]


# --------------------------------------------------------------------------------------------------- file types


def test_file_type_names_include_the_iq_ternary_and_mxfp4_types():
    assert [gguf.file_type_name(ft) for ft in (19, 21, 27, 30, 31, 36, 37, 38)] == [
        "IQ2_XXS", "Q2_K_S", "IQ3_M", "IQ4_XS", "IQ1_M", "TQ1_0", "TQ2_0", "MXFP4_MOE"]
    assert gguf.file_type_name(15) == "Q4_K_M" and gguf.file_type_name(32) == "BF16"
    assert gguf.file_type_name(33) == "33" and gguf.file_type_name(1024) == "1024"  # retired, guessed: no name
    with pytest.raises(TypeError):
        gguf.FILE_TYPES[99] = "X"  # type: ignore[index]


def test_file_types_match_the_pinned_llama_h():
    if not LLAMA_H.is_file():
        pytest.skip("pinned llama.h is not present in native/hcstream/third_party")
    enum: dict[int, str] = {}
    for line in LLAMA_H.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*LLAMA_FTYPE_(?:ALL|MOSTLY)_(\w+)\s*=\s*(\d+)\s*,", line)  # commented-out values never match
        if m:
            enum[int(m.group(2))] = m.group(1)
    assert len(enum) >= 36
    assert dict(gguf.FILE_TYPES) == enum


def test_summary_names_an_iq_file_type(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, kv("general.file_type", U32, 30)))
    assert gguf.summarize([p]).file_type == "IQ4_XS"


# --------------------------------------------------------------------------------------------------- tensor table


def test_duplicate_tensor_inside_one_shard(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1), [("blk.0.a.weight", (8,)), ("blk.0.a.weight", (8,))])
    with pytest.raises(gguf.GgufError, match=r"tensor blk\.0\.a\.weight appears more than once \(twice in shard 0\)"):
        gguf.read_tensor_table([p])


def test_duplicate_tensor_across_shards(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    write_gguf(a, llama_kvs(1), [("blk.0.a.weight", (8,))])
    write_gguf(b, llama_kvs(1), [("output.weight", (8,)), ("blk.0.a.weight", (8,))])
    with pytest.raises(gguf.GgufError, match=r"blk\.0\.a\.weight appears more than once \(in shards 0 and 1\)"):
        gguf.read_tensor_table([a, b])


def test_files_without_tensors_and_no_files(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1))
    with pytest.raises(gguf.GgufError, match="^no tensors in the given GGUF files$"):
        gguf.read_tensor_table([p, p])
    with pytest.raises(gguf.GgufError, match="^no GGUF files given$"):
        gguf.read_tensor_table([])


# --------------------------------------------------------------------------------------------------- alignment


def test_alignment_other_than_32_places_the_data_section(tmp_path):
    p = tmp_path / "m.gguf"
    tensors = [("blk.0.a.weight", (8,)), ("blk.0.b.weight", (24,))]  # 32 and 96 bytes
    for pad in range(64):  # a header length where rounding up to 32 and to 64 differ, so the default would misread
        kvs = llama_kvs(1, kv("general.alignment", U32, 64), kv("filler", STR, "x" * pad))
        header_len = write_gguf(p, kvs, tensors, layout_alignment=64)
        if _up(header_len, 32) != _up(header_len, 64):
            break
    else:
        raise AssertionError("no filler length separates the two roundings")
    data_start = _up(header_len, 64)
    table = gguf.read_tensor_table([p])
    assert [(t.offset, t.nbytes, t.span) for t in table] == [(data_start, 32, 64), (data_start + 64, 96, 128)]
    assert gguf.read_file(p).tensors == {"blk.0.a.weight": 64, "blk.0.b.weight": 128}


@pytest.mark.parametrize("entry", [
    kv("general.alignment", U32, 0),
    kv("general.alignment", U32, 48),
    kv("general.alignment", U32, 3),
    kv("general.alignment", I32, -32),
    kv("general.alignment", F32V, 32.0),
    kv("general.alignment", BOOL, True),
    kv("general.alignment", STR, "32"),
    kv("general.alignment", ARR, [32], etype=U32),
], ids=["zero", "48", "3", "negative", "float", "bool", "string", "array"])
def test_bad_alignment_is_refused_by_both_readers(tmp_path, entry):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, entry), [("blk.0.a.weight", (8,))])
    with pytest.raises(gguf.GgufError, match="general.alignment is .*, expected a positive power-of-two integer"):
        gguf.read_file(p)
    with pytest.raises(gguf.GgufError, match="expected a positive power-of-two integer"):
        gguf.read_tensor_table([p])


def test_bad_alignment_is_a_manifest_error(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, kv("general.alignment", U32, 0)), [("blk.0.a.weight", (8,)), ("output.weight", (8,))])
    with pytest.raises(ManifestError, match="cannot read GGUF m.gguf: general.alignment is 0"):
        build_manifest("m", [p])


# --------------------------------------------------------------------------------------------------- arrays


@pytest.mark.parametrize("suffix", ["head_count", "head_count_kv", "key_length", "value_length"])
def test_per_layer_attention_arrays_are_kept_up_to_block_count(tmp_path, suffix):
    values = [2] * 79 + [8]
    p = tmp_path / "m.gguf"
    # the array comes before block_count: the limit does not depend on key order
    write_gguf(p, [kv(f"llama.attention.{suffix}", ARR, values, etype=I32), *llama_kvs(80)])
    f = gguf.read_file(p)
    assert f.metadata[f"llama.attention.{suffix}"] == values and f.array_lengths[f"llama.attention.{suffix}"] == 80
    assert gguf.metadata_int(f.metadata, f"llama.attention.{suffix}") == 8


def test_per_layer_array_longer_than_block_count_is_skipped(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(2, kv("llama.attention.head_count_kv", ARR, [1] * 70, etype=U32)))
    f = gguf.read_file(p)
    key = "llama.attention.head_count_kv"
    assert f.metadata[key] is None and f.array_lengths[key] == 70
    with pytest.raises(gguf.GgufError, match="head_count_kv is an array longer than the header reader keeps"):
        gguf.metadata_int(f.metadata, key)


def test_per_layer_array_beyond_llama_max_layers_is_skipped(tmp_path):
    n = gguf.MAX_LAYER_ARRAY_KEEP + 1
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(n, kv("llama.attention.head_count", ARR, [4] * n, etype=U32)))
    f = gguf.read_file(p)
    assert f.metadata["llama.attention.head_count"] is None and f.array_lengths["llama.attention.head_count"] == n


def test_other_long_arrays_are_still_skipped(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(80, kv("tokenizer.ggml.token_type", ARR, [1] * 100, etype=I32),
                            kv("llama.feed_forward_length", ARR, [128] * 80, etype=U32),
                            kv("llama.attention.head_count_kv", ARR, [2.0] * 80, etype=F32V),
                            kv("tokenizer.ggml.tokens", ARR, [f"t{i}" for i in range(100)], etype=STR)))
    f = gguf.read_file(p)
    for key, n in [("tokenizer.ggml.token_type", 100), ("llama.feed_forward_length", 80),
                   ("llama.attention.head_count_kv", 80), ("tokenizer.ggml.tokens", 100)]:
        assert f.metadata[key] is None and f.array_lengths[key] == n, key


def test_summary_reads_a_long_per_layer_head_count_kv(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(80, kv("llama.attention.head_count", U32, 8),
                            kv("llama.attention.head_count_kv", ARR, [1] * 79 + [4], etype=U32)))
    s = gguf.summarize([p])
    assert (s.n_layer, s.n_head, s.n_head_kv) == (80, 8, 4)  # was the n_head fallback (8) with a 64-entry limit


def test_manifest_reads_a_long_per_layer_head_count_kv(tmp_path):
    blocks = 70
    tensors = [("token_embd.weight", (32, 10)), *[(f"blk.{i}.attn_norm.weight", (32,)) for i in range(blocks)],
               ("output_norm.weight", (32,)), ("output.weight", (32, 10))]
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(blocks, kv("llama.attention.head_count", ARR, [8] * blocks, etype=U32),
                            kv("llama.attention.head_count_kv", ARR, [2] * (blocks - 1) + [4], etype=U32)), tensors)
    m = build_manifest("m", [p])
    assert (m.n_layer, m.n_head, m.n_head_kv, m.key_length) == (blocks, 8, 4, 4)
    assert len(m.units) == blocks + 1


# --------------------------------------------------------------------------------------------------- metadata_int


def test_metadata_int_values():
    meta: dict[str, object] = {"a": 7, "b": [1, 9, 3], "c": True, "d": "12", "e": 2.0, "f": [True, 2], "g": []}
    assert [gguf.metadata_int(meta, k) for k in "abcdefg"] == [7, 9, None, None, 2, 2, None]
    assert gguf.metadata_int(meta, "absent") is None


@pytest.mark.parametrize(("value", "message"), [
    (None, "is an array longer than the header reader keeps"),
    (float("nan"), "x is nan, not a finite number"),
    (float("-inf"), "x is -inf, not a finite number"),
])
def test_metadata_int_errors(value, message):
    with pytest.raises(gguf.GgufError, match=message):
        gguf.metadata_int({"x": value}, "x")


def test_summary_treats_unknown_values_as_absent():
    meta: dict[str, object] = {"x": float("inf"), "y": None}
    assert gguf._int(meta, "x") is None and gguf._int(meta, "y") is None


# --------------------------------------------------------------------------------------------------- malformed arrays


def _array_header(key: str, etype: int, n: int) -> bytes:
    """An array entry's key, type and length with no elements after it."""
    return _s(key) + struct.pack("<IIQ", ARR, etype, n)


@pytest.mark.parametrize("n", [2**62, 1000], ids=["seek-overflow", "past-eof"])
def test_skipped_array_past_the_end_of_the_file_is_a_gguf_error(tmp_path, n):
    # a skipped scalar array is passed over with a seek; a length past the end of the file must not reach the seek
    # (2**62 values raised a bare ValueError, which ModelService and the stream runtime do not catch), and must not
    # read as a valid file when it is the last key
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, _array_header("tokenizer.ggml.token_type", U32, n)))
    for read in (gguf.read_file, lambda q: gguf.read_tensor_table([q]), lambda q: gguf.summarize([q])):
        with pytest.raises(gguf.GgufError, match="tokenizer.ggml.token_type .*past the end of the file"):
            read(p)


def test_nested_array_is_refused(tmp_path):
    # llama.cpp refuses an array of arrays (gguf_init_from_file_impl, invalid GGUF type)
    inner = struct.pack("<IQ", U32, 1) + struct.pack("<I", 5)
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, _array_header("x", ARR, 1) + inner))
    with pytest.raises(gguf.GgufError, match="x is an array of arrays"):
        gguf.read_file(p)


def test_deeply_nested_array_is_a_gguf_error_not_a_recursion_error(tmp_path):
    depth = 5000
    p = tmp_path / "m.gguf"
    entry = _s("x") + struct.pack("<I", ARR) + struct.pack("<IQ", ARR, 1) * depth + struct.pack("<IQ", U32, 0)
    write_gguf(p, llama_kvs(1, entry))
    with pytest.raises(gguf.GgufError, match="array of arrays"):
        gguf.summarize([p])


def test_array_lengths_within_the_file_still_read(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, kv("tokenizer.ggml.token_type", ARR, [1] * 100, etype=U32)),
               [("blk.0.a.weight", (8,))])
    f = gguf.read_file(p)
    assert f.metadata["tokenizer.ggml.token_type"] is None and f.array_lengths["tokenizer.ggml.token_type"] == 100
    assert f.tensors == {"blk.0.a.weight": 32}


# --------------------------------------------------------------------------------------------------- data section


def test_alignment_that_puts_the_data_section_past_the_end_is_refused(tmp_path):
    # a power of two, so the alignment check passes, but no file this small has data that far in; the summary
    # reader used to report the last tensor as 0 bytes
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, kv("general.alignment", U64, 2**40)), [("blk.0.a.weight", (8,))])
    with pytest.raises(gguf.GgufError, match="data section starts at 1099511627776, past the end of the file"):
        gguf.read_file(p)
    with pytest.raises(gguf.GgufError, match="past the end of the file"):
        gguf.read_tensor_table([p])


def test_large_alignment_without_tensors_still_reads(tmp_path):
    # a vocab-only GGUF has no data section, so its start may lie past the end (llama.cpp only seeks to it when
    # there are tensors)
    p = tmp_path / "m.gguf"
    write_gguf(p, llama_kvs(1, kv("general.alignment", U32, 2**20)))
    assert gguf.read_file(p).tensors == {}
