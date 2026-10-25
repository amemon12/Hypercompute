"""Minimal GGUF (v2/v3) header reader: metadata plus per-tensor sizes for placement planning.

Tensor sizes come from data-offset differences, so no per-quantization block table is needed
(each size includes < `alignment` bytes of padding). Multi-part (split) GGUFs are summed.

llama.cpp references are to the pinned b11770 release (commit 39da130f): include/llama.h, ggml/include/gguf.h,
ggml/src/gguf.cpp, src/llama-hparams.h and src/llama-model-loader.cpp.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

from .schemas import GgufSummary

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_INTEGERS = frozenset({0, 1, 2, 3, 4, 5, 10, 11})  # GGUF integer value types, uint8 through int64
_STRING, _ARRAY = 8, 9
_LAYER_RE = re.compile(r"^blk\.(\d+)\.")
# Routed-expert tensors of a block, the set llama.cpp's --n-cpu-moe moves (llama-common.dll of the b11770 release
# carries "blk\.\d+\.ffn_(up|down|gate_up|gate)_(ch|)exps"): their .weight, .bias and .scale tensors all match
# (gpt-oss has expert biases). Shared experts (ffn_*_shexp), the router (ffn_gate_inp*) and exp_probs_b are dense:
# every token reads them.
EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:up|down|gate_up|gate)_(?:ch)?exps\.")
_EXPERT_OVERRIDE = r"^blk\.({layers})\.ffn_(up|down|gate_up|gate)_(ch|)exps\."
# general.file_type (enum llama_ftype in include/llama.h at b11770) -> name. 4-6 and 33-35 are retired values that
# llama.h keeps only as comments; LLAMA_FTYPE_GUESSED (1024) is never written to a file.
FILE_TYPES: Mapping[int, str] = MappingProxyType({
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
})
ALIGNMENT_KEY = "general.alignment"
DEFAULT_ALIGNMENT = 32  # GGUF_DEFAULT_ALIGNMENT in gguf.h, used when the key is absent
MAX_ARRAY_KEEP = 64  # arrays longer than this are skipped (only their length is kept)
# Integer arrays of the per-layer attention keys are kept up to the model's block_count entries and never more than
# LLAMA_MAX_LAYERS (src/llama-hparams.h). llama.cpp reads head_count and head_count_kv per layer and refuses an array
# whose length is not block_count (get_key_or_arr in src/llama-model-loader.cpp), so a longer array is skipped like
# any other long array instead of being reduced to a value.
MAX_LAYER_ARRAY_KEEP = 512
_LAYER_ARRAY_RE = re.compile(r"^(.+)\.attention\.(?:head_count|head_count_kv|key_length|value_length)$")
# ggml type id -> (name, elements per block, bytes per block), from ggml.h/ggml-common.h at the pinned b11770 commit
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}


class GgufError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    """One tensor's exact location: what a streaming reader needs to fetch its bytes without the OS cache."""

    name: str
    shard: int  # index into the ordered GGUF paths
    ggml_type: int
    dims: tuple[int, ...]
    offset: int  # absolute file offset of the data
    nbytes: int | None  # exact ggml_nbytes; None for a type this table does not know
    span: int  # bytes up to the next tensor or the end of the file (>= nbytes, includes alignment padding)

    @property
    def layer(self) -> int | None:
        m = _LAYER_RE.match(self.name)
        return int(m.group(1)) if m else None

    @property
    def type_name(self) -> str:
        known = GGML_TYPES.get(self.ggml_type)
        return known[0] if known else f"type{self.ggml_type}"


def tensor_nbytes(ggml_type: int, dims: tuple[int, ...]) -> int | None:
    known = GGML_TYPES.get(ggml_type)
    if known is None or not dims:
        return None
    _, block, size = known
    if dims[0] % block:
        return None
    n = dims[0] // block * size
    for d in dims[1:]:
        n *= d
    return n


@dataclass
class GgufFile:
    path: Path
    version: int
    metadata: dict[str, object]
    array_lengths: dict[str, int]
    tensors: dict[str, int] = field(default_factory=dict)  # name -> bytes


def _read(fh: BinaryIO, fmt: str) -> object:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise GgufError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_str(fh: BinaryIO) -> str:
    n = int(_read(fh, "<Q"))  # type: ignore[arg-type]
    if n > 1 << 24:
        raise GgufError("implausible string length in GGUF header")
    return fh.read(n).decode("utf-8", errors="replace")


def _array_keep(key: str, etype: int) -> int:
    """How many entries of an array value are kept; integer per-layer attention arrays get the per-layer cap."""
    return MAX_LAYER_ARRAY_KEEP if etype in _INTEGERS and _LAYER_ARRAY_RE.match(key) else MAX_ARRAY_KEEP


def _skip(fh: BinaryIO, nbytes: int, key: str) -> None:
    """Move past `nbytes` of an array value the reader does not keep.

    GgufError when they run past the end of the file, as in llama.cpp (gguf_reader checks the remaining file size).
    A seek that far would otherwise raise a bare ValueError for a huge length, or succeed silently for the last key.
    """
    here = fh.tell()
    end = fh.seek(0, 2)
    if nbytes > end - here:
        raise GgufError(f"GGUF metadata {key} is an array of {nbytes} bytes, past the end of the file")
    fh.seek(here + nbytes)


def _read_value(fh: BinaryIO, vtype: int, key: str, lengths: dict[str, int]) -> object:
    if vtype in _SCALARS:
        return _read(fh, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_str(fh)
    if vtype == _ARRAY:
        etype = int(_read(fh, "<I"))  # type: ignore[arg-type]
        n = int(_read(fh, "<Q"))  # type: ignore[arg-type]
        if etype == _ARRAY:  # llama.cpp refuses this type for array elements; reading it would recurse without bound
            raise GgufError(f"GGUF metadata {key} is an array of arrays, which llama.cpp does not read")
        lengths[key] = n
        keep = _array_keep(key, etype)
        if etype in _SCALARS and n > keep:
            _skip(fh, struct.calcsize(_SCALARS[etype]) * n, key)
            return None
        items = []
        for _ in range(n):
            v = _read_value(fh, etype, key, {})
            if n <= keep:
                items.append(v)
        return items if n <= keep else None
    raise GgufError(f"unknown GGUF value type {vtype} for {key}")


def _read_preamble(fh: BinaryIO, path: Path) -> tuple[int, int, int]:
    """(version, tensor count, metadata key count) after checking the magic."""
    if fh.read(4) != b"GGUF":
        raise GgufError(f"{path.name} is not a GGUF file")
    version = int(_read(fh, "<I"))  # type: ignore[arg-type]
    if version not in (2, 3):
        raise GgufError(f"unsupported GGUF version {version}")
    n_tensors = int(_read(fh, "<Q"))  # type: ignore[arg-type]
    n_kv = int(_read(fh, "<Q"))  # type: ignore[arg-type]
    return version, n_tensors, n_kv


def _limit_layer_arrays(meta: Mapping[str, object]) -> dict[str, object]:
    """A copy of `meta` where a per-layer attention array longer than its block_count reads as skipped (None)."""
    out = dict(meta)
    for key, value in meta.items():
        match = _LAYER_ARRAY_RE.match(key)
        if match is None or not isinstance(value, list):
            continue
        blocks = meta.get(f"{match.group(1)}.block_count")
        if isinstance(blocks, int) and not isinstance(blocks, bool) and len(value) > blocks:
            out[key] = None
    return out


def _read_metadata(fh: BinaryIO, n_kv: int) -> tuple[dict[str, object], dict[str, int]]:
    """(values, array lengths) of every metadata key. A skipped array's value is None; its length is kept."""
    meta: dict[str, object] = {}
    lengths: dict[str, int] = {}
    for _ in range(n_kv):
        key = _read_str(fh)
        vtype = int(_read(fh, "<I"))  # type: ignore[arg-type]
        meta[key] = _read_value(fh, vtype, key, lengths)
    return _limit_layer_arrays(meta), lengths


def _alignment(meta: Mapping[str, object]) -> int:
    """general.alignment, or the GGUF default when absent.

    llama.cpp (gguf_init_from_file_impl in ggml/src/gguf.cpp) refuses a file whose alignment is 0 or not a power of
    two. A value that is not an integer is refused too: no data section offset can be derived from it.
    """
    if ALIGNMENT_KEY not in meta:
        return DEFAULT_ALIGNMENT
    value = meta[ALIGNMENT_KEY]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value & (value - 1):
        raise GgufError(f"{ALIGNMENT_KEY} is {value!r}, expected a positive power-of-two integer")
    return value


def _data_start(header_end: int, alignment: int, file_size: int, n_tensors: int, path: Path) -> int:
    """Offset of the data section: the end of the tensor infos rounded up to `alignment`.

    GgufError when the file declares tensors but the data section would start past its end (a power-of-two
    alignment can still be far too large, and the last tensor's size would then read as 0). Without tensors there
    is no data section, so its start is not checked, as in llama.cpp.
    """
    start = (header_end + alignment - 1) // alignment * alignment
    if n_tensors and start > file_size:
        raise GgufError(f"{path.name}: the data section starts at {start}, past the end of the file ({file_size} "
                        f"bytes); {ALIGNMENT_KEY} is {alignment}")
    return start


def read_file(path: Path) -> GgufFile:
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        version, n_tensors, n_kv = _read_preamble(fh, path)
        meta, lengths = _read_metadata(fh, n_kv)
        alignment = _alignment(meta)
        infos: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = _read_str(fh)
            n_dims = int(_read(fh, "<I"))  # type: ignore[arg-type]
            fh.seek(8 * n_dims, 1)
            _read(fh, "<I")  # ggml type (sizes come from offsets)
            infos.append((name, int(_read(fh, "<Q"))))  # type: ignore[arg-type]
        data_start = _data_start(fh.tell(), alignment, file_size, n_tensors, path)
    ordered = sorted(infos, key=lambda t: t[1])
    tensors: dict[str, int] = {}
    for i, (name, offset) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else file_size - data_start
        tensors[name] = max(0, end - offset)
    return GgufFile(path=path, version=version, metadata=meta, array_lengths=lengths, tensors=tensors)


def _read_tensor_infos(path: Path, shard: int) -> list[TensorInfo]:
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        _, n_tensors, n_kv = _read_preamble(fh, path)
        meta, _ = _read_metadata(fh, n_kv)
        alignment = _alignment(meta)
        raw: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name = _read_str(fh)
            n_dims = int(_read(fh, "<I"))  # type: ignore[arg-type]
            if n_dims > 8:
                raise GgufError(f"{name} claims {n_dims} dimensions")
            dims = tuple(int(_read(fh, "<Q")) for _ in range(n_dims))  # type: ignore[arg-type]
            raw.append((name, dims, int(_read(fh, "<I")), int(_read(fh, "<Q"))))  # type: ignore[arg-type]
        data_start = _data_start(fh.tell(), alignment, file_size, n_tensors, path)
    ordered = sorted(raw, key=lambda t: t[3])
    out: list[TensorInfo] = []
    for i, (name, dims, gtype, rel) in enumerate(ordered):
        end = ordered[i + 1][3] if i + 1 < len(ordered) else file_size - data_start
        nbytes = tensor_nbytes(gtype, dims)
        span = max(0, end - rel)
        if nbytes is not None and nbytes > span:
            raise GgufError(f"{name} needs {nbytes} bytes but only {span} precede the next tensor")
        out.append(TensorInfo(name=name, shard=shard, ggml_type=gtype, dims=dims, offset=data_start + rel,
                              nbytes=nbytes, span=span))
    return out


def read_tensor_table(paths: Iterable[Path]) -> list[TensorInfo]:
    """Every tensor of a (possibly split) GGUF with its shard, absolute offset, exact size and type."""
    table: list[TensorInfo] = []
    first_shard: dict[str, int] = {}
    n_files = 0
    for shard, path in enumerate(paths):
        n_files += 1
        for info in _read_tensor_infos(path, shard):
            if info.name in first_shard:
                earlier = first_shard[info.name]
                where = f"twice in shard {shard}" if earlier == shard else f"in shards {earlier} and {shard}"
                raise GgufError(f"tensor {info.name} appears more than once ({where})")
            first_shard[info.name] = shard
            table.append(info)
    if not n_files:
        raise GgufError("no GGUF files given")
    if not table:
        raise GgufError("no tensors in the given GGUF files")
    return table


def file_type_name(ft: int) -> str:
    """Name of a general.file_type value, or the number as text for one FILE_TYPES does not list."""
    return FILE_TYPES.get(ft, str(ft))


def metadata_int(meta: Mapping[str, object], key: str) -> int | None:
    """Integer metadata value; a per-layer array gives its maximum (the conservative value for sizing).

    None when the key is absent or not numeric. Raises GgufError when the key holds an array the header reader
    skipped, so a default never stands in for a value the file does carry, and when the value is a non-finite float.
    """
    if key in meta and meta[key] is None:
        raise GgufError(f"GGUF metadata {key} is an array longer than the header reader keeps ({MAX_ARRAY_KEEP} "
                        f"values; integer per-layer attention arrays up to block_count, at most "
                        f"{MAX_LAYER_ARRAY_KEEP}), so its value is unknown")
    v = meta.get(key)
    if isinstance(v, list):
        nums = [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
        return max(nums) if nums else None
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    if isinstance(v, float) and not math.isfinite(v):
        raise GgufError(f"GGUF metadata {key} is {v}, not a finite number")
    return int(v)


def is_expert_tensor(name: str) -> bool:
    return EXPERT_TENSOR_RE.match(name) is not None


@dataclass(frozen=True)
class MoeCounts:
    """<arch>.expert_count, expert_used_count and expert_shared_count; zeros for a dense model."""

    expert_count: int = 0
    expert_used_count: int = 0
    expert_shared_count: int = 0


def moe_counts(meta: Mapping[str, object], arch: str) -> MoeCounts:
    """The MoE counts of a GGUF header. GgufError for a negative count, or a used count that is zero or above the
    expert count while experts are declared (llama.cpp refuses such a model at load)."""
    counts = MoeCounts(*(metadata_int(meta, f"{arch}.{key}") or 0
                         for key in ("expert_count", "expert_used_count", "expert_shared_count")))
    if min(counts.expert_count, counts.expert_used_count, counts.expert_shared_count) < 0:
        raise GgufError(f"negative MoE expert count in {arch} metadata: {counts}")
    if counts.expert_count and not 0 < counts.expert_used_count <= counts.expert_count:
        raise GgufError(f"{arch}.expert_used_count is {counts.expert_used_count}, expected 1..{counts.expert_count}")
    return counts


def _digit_ranges(lo: int, hi: int) -> list[str]:
    """Regex alternatives matching exactly the decimal numbers lo..hi, which have the same number of digits."""
    out: list[str] = []
    top = 10 ** (len(str(hi)) - 1)  # the place of the leading digit: a larger step would allow a leading zero
    while lo <= hi:
        step = 1
        while step * 10 <= top and lo % (step * 10) == 0 and lo + step * 10 - 1 <= hi:
            step *= 10
        first = lo // step
        n = min((hi - lo + 1) // step, 10 - first % 10)
        digit = str(first % 10) if n == 1 else f"[{first % 10}-{first % 10 + n - 1}]"
        out.append(str(first)[:-1] + digit + "[0-9]" * (len(str(step)) - 1))
        lo += n * step
    return out


def number_alternation(values: Iterable[int]) -> str:
    """'|'-joined regex alternatives that match exactly the given non-negative integers, with runs compacted into
    digit classes (3..27 -> [3-9]|1[0-9]|2[0-7]). Never contains a comma: llama.cpp splits -ot values at commas."""
    nums = sorted(set(values))
    if not nums or nums[0] < 0:
        raise ValueError(f"need non-negative integers, got {nums[:4]}")
    runs: list[tuple[int, int]] = []
    for v in nums:
        if runs and v == runs[-1][1] + 1 and len(str(v)) == len(str(runs[-1][0])):
            runs[-1] = (runs[-1][0], v)
        else:
            runs.append((v, v))
    return "|".join(part for lo, hi in runs for part in _digit_ranges(lo, hi))


def expert_override_pattern(layers: Iterable[int]) -> str:
    """The llama.cpp tensor pattern (std::regex, searched in each tensor name) matching exactly the routed-expert
    tensors (EXPERT_TENSOR_RE) of `layers` and nothing else."""
    return _EXPERT_OVERRIDE.format(layers=number_alternation(layers))


def _int(meta: Mapping[str, object], key: str) -> int | None:
    """metadata_int for summaries: a skipped array or a non-finite number reads as absent."""
    try:
        return metadata_int(meta, key)
    except GgufError:
        return None


def summarize(paths: Iterable[Path]) -> GgufSummary:
    files = [read_file(p) for p in paths]
    if not files:
        raise GgufError("no GGUF files given")
    meta = files[0].metadata
    tensors: dict[str, int] = {}
    for f in files:
        tensors.update(f.tensors)
    arch = meta.get("general.architecture")
    a = str(arch) if arch else ""
    n_layer = _int(meta, f"{a}.block_count")
    layer_bytes = [0] * (n_layer or 0)
    expert_bytes = [0] * (n_layer or 0)
    token_embd = output = other = 0
    has_output = False
    for name, size in tensors.items():
        m = _LAYER_RE.match(name)
        if m and n_layer and int(m.group(1)) < n_layer:
            layer_bytes[int(m.group(1))] += size
            expert_bytes[int(m.group(1))] += size if is_expert_tensor(name) else 0
        elif name.startswith("token_embd."):
            token_embd += size
        elif name.startswith("output"):  # output.* and output_norm.*: llama.cpp's output layer, as in the manifest
            output += size
            has_output = has_output or name == "output.weight"
        else:
            other += size
    ft = _int(meta, "general.file_type")
    return GgufSummary(
        architecture=a or None,
        name=str(meta["general.name"]) if meta.get("general.name") else None,
        file_type=file_type_name(ft) if ft is not None else None,
        n_layer=n_layer,
        n_embd=_int(meta, f"{a}.embedding_length"),
        n_head=_int(meta, f"{a}.attention.head_count"),
        n_head_kv=_int(meta, f"{a}.attention.head_count_kv") or _int(meta, f"{a}.attention.head_count"),
        context_length=_int(meta, f"{a}.context_length"),
        vocab_size=files[0].array_lengths.get("tokenizer.ggml.tokens"),
        tensor_count=len(tensors),
        total_tensor_bytes=sum(tensors.values()),
        layer_bytes=layer_bytes,
        token_embd_bytes=token_embd,
        output_bytes=output if has_output else output + token_embd,
        output_tied=not has_output,
        other_bytes=other,
        chat_template_present="tokenizer.chat_template" in meta,
        key_length=_int(meta, f"{a}.attention.key_length"),
        value_length=_int(meta, f"{a}.attention.value_length"),
        expert_count=_int(meta, f"{a}.expert_count"),
        expert_used_count=_int(meta, f"{a}.expert_used_count"),
        expert_shared_count=_int(meta, f"{a}.expert_shared_count"),
        layer_expert_bytes=expert_bytes if any(expert_bytes) else [],
    )


def key_value_dims(meta_summary: GgufSummary) -> tuple[int, int] | None:
    """(n_head_kv, head_dim) for KV sizing, when known."""
    s = meta_summary
    if not (s.n_embd and s.n_head and s.n_head_kv):
        return None
    return s.n_head_kv, s.n_embd // s.n_head
