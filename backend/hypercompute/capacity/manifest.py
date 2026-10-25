"""Exact per-unit manifest of one GGUF artifact: what the planner places, sizes and identifies.

Units are the blocks `0..n_layer-1` followed by the output head at index `n_layer` (output tensors, or token_embd
when the embeddings are tied), the same assignment `stream_config.unit_of` uses. `weight_bytes` is the exact sum of
ggml tensor sizes; `buffer_bytes` is taken from `stream_config.layer_layouts`, so admission and the native extension
see the same 4 KiB-aligned numbers.

A mixture-of-experts GGUF (<arch>.expert_count > 0) also records each block's routed-expert bytes
(UnitSpec.expert_bytes: the tensors gguf.EXPERT_TENSOR_RE names, the set llama.cpp's --n-cpu-moe moves) and the expert
counts; attention, norms, the router and shared experts are the block's dense bytes. A dense GGUF keeps zeros.

Every number here is read from the file headers, none is estimated. A shard sha256 is recorded only when the caller
passes one it has verified; this module never hashes shard data. The chat template is hashed and its text is not
kept. Manifests are cached by (model id, shard paths, sizes, mtimes, revision, shard hashes).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

from .. import gguf
from ..gguf import TensorInfo
from .stream_config import LayerLayout, layer_layouts, unit_of
from .types import ModelManifest, ShardInfo, UnitSpec

FINGERPRINT_VERSION = 1
CACHE_SIZE = 32
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

ShardStat = tuple[str, int, int]  # resolved path, size, mtime_ns


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class _Meta:
    """The metadata fields of a ModelManifest, read from the first shard's GGUF header."""

    architecture: str
    file_type: str | None
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    key_length: int
    value_length: int
    context_length: int | None
    rope: tuple[tuple[str, str], ...]
    tokenizer_model: str | None
    vocab_size: int | None
    chat_template_sha256: str | None
    expert_count: int
    expert_used_count: int
    expert_shared_count: int


def build_manifest(model_id: str, shards: Sequence[Path], *, revision: str | None = None,
                   shard_hashes: Mapping[str, str] | None = None) -> ModelManifest:
    """Manifest of a (possibly split) GGUF. `shards` are every split, in split order.

    `shard_hashes` maps a shard (by bare file name, or by any path that resolves to it) to a sha256 the caller has
    already verified. Shards without an entry get `sha256=None`. Raises ManifestError for anything that is not a
    complete, consistent GGUF.
    """
    if not model_id:
        raise ManifestError("model_id is empty")
    if isinstance(shards, str | bytes | os.PathLike):
        raise ManifestError("shards must be a sequence of paths, not a single path")
    paths = tuple(Path(s) for s in shards)
    if not paths:
        raise ManifestError("no GGUF shards given")
    stats = _stat_shards(paths)
    hashes = _shard_hash_list(paths, stats, shard_hashes or {})
    return _cached_manifest(model_id, stats, revision, hashes)


@lru_cache(maxsize=CACHE_SIZE)
def _cached_manifest(model_id: str, stats: tuple[ShardStat, ...], revision: str | None,
                     hashes: tuple[str | None, ...]) -> ModelManifest:
    paths = [Path(s[0]) for s in stats]
    try:
        table = gguf.read_tensor_table(paths)
        headers = [gguf.read_file(p) for p in paths]
    except (OSError, ValueError, TypeError, OverflowError) as err:  # GgufError is a ValueError; the others come
        # from malformed header values (for example a non-numeric general.alignment)
        raise ManifestError(f"cannot read GGUF {', '.join(p.name for p in paths)}: {err}") from err
    _check_splits([h.metadata for h in headers], len(table))
    shards = tuple(ShardInfo(path=path, size=size, sha256=sha)
                   for (path, size, _), sha in zip(stats, hashes, strict=True))
    built = _assemble(model_id, revision, shards, table, _read_meta(headers[0].metadata, headers[0].array_lengths))
    if _stat_shards(tuple(paths)) != stats:
        raise ManifestError("a shard changed on disk while its manifest was being built")
    return built


def _assemble(model_id: str, revision: str | None, shards: tuple[ShardInfo, ...], table: Sequence[TensorInfo],
              meta: _Meta) -> ModelManifest:
    _check_blocks(table, meta.n_layer)
    tied = not any(t.name == "output.weight" for t in table)
    if tied and not any(t.name == "token_embd.weight" for t in table):
        # llama.cpp cannot load this either; without the check the head would silently be output_norm alone
        raise ManifestError("the GGUF has no output head: neither output.weight nor token_embd.weight is present")
    layouts = layer_layouts(table)  # the head key n_layer exists: it holds output.weight or token_embd.weight
    units = tuple(_unit_spec(layouts[i], meta.n_layer) for i in range(meta.n_layer + 1))
    if meta.expert_count == 0 and any(u.expert_bytes for u in units):
        # without the counts the active fraction is unknown, and llama.cpp cannot route this model either
        raise ManifestError(f"the GGUF has routed-expert tensors but no {meta.architecture}.expert_count")
    streamable = all(t.nbytes is not None for i in range(meta.n_layer + 1) for t in layouts[i].tensors)
    token_embd, other = _outside_bytes(table, meta.n_layer, tied)
    return ModelManifest(
        model_id=model_id, revision=revision, shards=shards, units=units, token_embd_bytes=token_embd,
        other_bytes=other, tied_embeddings=tied, streamable=streamable, fingerprint=_fingerprint(shards, table),
        **{f.name: getattr(meta, f.name) for f in fields(meta)},
    )


# --------------------------------------------------------------------------------------------------- shards


def _stat_shards(paths: tuple[Path, ...]) -> tuple[ShardStat, ...]:
    out: list[ShardStat] = []
    for p in paths:
        try:
            resolved = p.resolve()
            st = resolved.stat()
        except OSError as err:
            raise ManifestError(f"cannot read shard {p}: {err}") from err
        if not stat.S_ISREG(st.st_mode):
            raise ManifestError(f"shard {p} is not a regular file")
        out.append((str(resolved), st.st_size, st.st_mtime_ns))
    return tuple(out)


def _shard_hash_list(paths: tuple[Path, ...], stats: tuple[ShardStat, ...],
                     shard_hashes: Mapping[str, str]) -> tuple[str | None, ...]:
    """One verified sha256 (lowercase hex) or None per shard, in shard order.

    A key is a bare file name, or a path in any spelling (forward or back slashes, relative or absolute) that
    resolves to the shard. Every key must name exactly one shard and no shard may have two entries.
    """
    resolved = [Path(s[0]) for s in stats]
    found: dict[int, str] = {}
    for key, value in shard_hashes.items():
        hits = _shards_for_key(key, paths, resolved)
        if not hits:
            raise ManifestError(f"shard_hashes key {key!r} is not a shard of this model")
        if len(hits) > 1:
            raise ManifestError(f"shard hash key {key!r} matches more than one shard; key them by full path")
        index = hits[0]
        if index in found:
            raise ManifestError(f"shard {paths[index].name} has more than one entry in shard_hashes")
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip().lower()):
            raise ManifestError(f"shard hash for {paths[index].name} is not a 64-character sha256 hex digest")
        found[index] = value.strip().lower()
    return tuple(found.get(i) for i in range(len(paths)))


def _shards_for_key(key: object, paths: tuple[Path, ...], resolved: Sequence[Path]) -> list[int]:
    if not isinstance(key, str) or not key:
        raise ManifestError(f"shard_hashes keys must be non-empty strings, got {key!r}")
    key_path = Path(key)
    if len(key_path.parts) == 1 and not key_path.anchor:
        return [i for i, p in enumerate(paths) if p.name == key]
    try:
        target = key_path.resolve()
    except OSError as err:
        raise ManifestError(f"cannot resolve shard_hashes key {key!r}: {err}") from err
    return [i for i, r in enumerate(resolved) if r == target]


def _check_splits(metas: Sequence[Mapping[str, object]], n_tensors: int) -> None:
    """The shards are one whole gguf-split set in split order, the way llama.cpp loads them.

    llama.cpp reads split.count from the first file and loads exactly that many splits, each checked by split.no.
    A partial set would otherwise give a silently wrong manifest (for example a missing last split holding
    output.weight makes the model look tied).
    """
    count = _meta_int(metas[0], "split.count")
    expected = count if count is not None and count > 1 else 1
    if len(metas) != expected:
        declared = "no split.count" if count is None else f"split.count {count}"
        raise ManifestError(f"{len(metas)} shards given but the first declares {declared}; pass every split in order")
    for i, meta in enumerate(metas):
        number = _meta_int(meta, "split.no")
        if number != i and not (i == 0 and number is None):
            raise ManifestError(f"shard {i} declares split.no {number}; pass the splits in split order")
    total = _meta_int(metas[0], "split.tensors.count")
    if total is not None and total != n_tensors:
        raise ManifestError(f"split.tensors.count is {total} but the shards hold {n_tensors} tensors")


# --------------------------------------------------------------------------------------------------- metadata


def _meta_int(meta: Mapping[str, object], key: str) -> int | None:
    """gguf.metadata_int (per-layer arrays give their maximum; a skipped array or a non-finite number is an error,
    so a default never stands in for a value the file does carry) with its errors raised as ManifestError."""
    try:
        return gguf.metadata_int(meta, key)
    except gguf.GgufError as err:
        raise ManifestError(str(err)) from err


def _first_layer_int(meta: Mapping[str, object], key: str) -> int | None:
    """Layer 0's value of a scalar-or-per-layer key, which is what llama.cpp's `n_head()` returns."""
    v = meta.get(key)
    if isinstance(v, list):
        return v[0] if v and isinstance(v[0], int) and not isinstance(v[0], bool) else None
    return _meta_int(meta, key)


def _positive_int(meta: Mapping[str, object], key: str, default: int | None = None) -> int:
    """A positive integer value; `default` when the key is absent, ManifestError when absent without a usable
    default (None or not positive)."""
    value = _meta_int(meta, key)
    if value is not None:
        if value <= 0:
            raise ManifestError(f"GGUF metadata {key} is {value}, expected a positive integer")
        return value
    if default is None:
        raise ManifestError(f"GGUF metadata has no {key}")
    if default <= 0:
        raise ManifestError(f"GGUF metadata has no {key} and its default ({default}) is not positive")
    return default


def _meta_str(meta: Mapping[str, object], key: str) -> str | None:
    v = meta.get(key)
    return v if isinstance(v, str) and v else None


def _rope_items(meta: Mapping[str, object], lengths: Mapping[str, int]) -> tuple[tuple[str, str], ...]:
    """Every metadata key containing "rope", value stringified. Long arrays the reader skipped show their length."""
    items: list[tuple[str, str]] = []
    for key, value in meta.items():
        if "rope" not in key:
            continue
        text = f"array[{lengths[key]}]" if value is None and key in lengths else str(value)
        items.append((key, text))
    return tuple(sorted(items))


def _read_meta(meta: Mapping[str, object], lengths: Mapping[str, int]) -> _Meta:
    arch = _meta_str(meta, "general.architecture")
    if arch is None:
        raise ManifestError("GGUF metadata has no general.architecture")
    n_embd = _positive_int(meta, f"{arch}.embedding_length")
    n_head = _positive_int(meta, f"{arch}.attention.head_count")
    # llama.cpp's default when key/value length are absent: n_embd / n_head(il=0), the first layer's head count
    # (not the maximum, which would understate the head size of a per-layer head_count array). None when layer 0
    # has no attention heads, so an absent key/value length is then an error.
    first_heads = _first_layer_int(meta, f"{arch}.attention.head_count")
    head_dim = n_embd // first_heads if first_heads is not None and first_heads > 0 else None
    ft = _meta_int(meta, "general.file_type")
    template = meta.get("tokenizer.chat_template")
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None
    try:
        moe = gguf.moe_counts(meta, arch)
    except gguf.GgufError as err:
        raise ManifestError(str(err)) from err
    return _Meta(
        architecture=arch,
        file_type=gguf.file_type_name(ft) if ft is not None else None,
        n_layer=_positive_int(meta, f"{arch}.block_count"),
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=_positive_int(meta, f"{arch}.attention.head_count_kv", n_head),
        key_length=_positive_int(meta, f"{arch}.attention.key_length", head_dim),
        value_length=_positive_int(meta, f"{arch}.attention.value_length", head_dim),
        context_length=_meta_int(meta, f"{arch}.context_length"),
        rope=_rope_items(meta, lengths),
        tokenizer_model=_meta_str(meta, "tokenizer.ggml.model"),
        vocab_size=lengths.get("tokenizer.ggml.tokens"),
        chat_template_sha256=template_sha,
        expert_count=moe.expert_count,
        expert_used_count=moe.expert_used_count,
        expert_shared_count=moe.expert_shared_count,
    )


# --------------------------------------------------------------------------------------------------- tensors


def _check_blocks(table: Sequence[TensorInfo], n_layer: int) -> None:
    """Every block 0..n_layer-1 has tensors and none lies beyond, so unit indices match stream_config's."""
    present = {t.layer for t in table if t.layer is not None}
    beyond = sorted(i for i in present if i >= n_layer)
    if beyond:
        raise ManifestError(f"tensors for blocks {beyond[:8]} lie beyond block_count {n_layer}")
    missing = [i for i in range(n_layer) if i not in present]
    if missing:
        raise ManifestError(f"blocks {missing[:8]} have no tensors (block_count {n_layer}, {len(missing)} missing)")


def _tensor_bytes(t: TensorInfo) -> int:
    """Exact ggml size; the file span (an upper bound) only for a type whose size is unknown."""
    return t.nbytes if t.nbytes is not None else t.span


def _unit_spec(layout: LayerLayout, n_layer: int) -> UnitSpec:
    by_type: dict[str, int] = {}
    for t in layout.tensors:
        by_type[t.type_name] = by_type.get(t.type_name, 0) + _tensor_bytes(t)
    types = tuple(sorted(by_type.items(), key=lambda item: (-item[1], item[0])))
    experts = sum(_tensor_bytes(t) for t in layout.tensors if gguf.is_expert_tensor(t.name))
    return UnitSpec(index=layout.layer, kind="head" if layout.layer == n_layer else "block",
                    weight_bytes=sum(size for _, size in types), buffer_bytes=layout.buffer_bytes, types=types,
                    expert_bytes=experts)


def _outside_bytes(table: Sequence[TensorInfo], n_layer: int, tied: bool) -> tuple[int, int]:
    """(token_embd bytes, other bytes) for tensors in no unit. Tied token_embd is in the head, so it counts 0 here."""
    token_embd = other = 0
    for t in table:
        if unit_of(t, n_layer, tied) is not None:
            continue
        if t.name.startswith("token_embd"):
            token_embd += _tensor_bytes(t)
        else:
            other += _tensor_bytes(t)
    return token_embd, other


def _fingerprint(shards: tuple[ShardInfo, ...], table: Sequence[TensorInfo]) -> str:
    """sha256 over shard identities (file name, size, verified sha256) and every tensor's name, type, dims, place.

    File names, not directories, so the same artifact at two locations has one fingerprint.
    """
    doc = {
        "version": FINGERPRINT_VERSION,
        "shards": [{"name": Path(s.path).name, "size": s.size, "sha256": s.sha256} for s in shards],
        "tensors": [{"name": t.name, "type": t.ggml_type, "dims": list(t.dims), "shard": t.shard, "offset": t.offset}
                    for t in sorted(table, key=lambda t: t.name)],
    }
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()
