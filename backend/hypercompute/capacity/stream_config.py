"""Config for the hcstream native extension: which CPU layers stay resident and which stream from the GGUF.

The byte layout here mirrors `hc_build_runs` in native/hcstream/hcstream.c exactly (tensors sorted by layer, shard,
offset; spans merged into 4 KiB-aligned file runs). Admission uses these numbers, and the extension reports the same
`slot_bytes` back in its stats file, so a mismatch is visible instead of silently overcommitting memory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..gguf import TensorInfo

ALIGN = 4096
CONFIG_VERSION = 1

# Limits of hc_parse_config in native/hcstream/hcstream.c. A config outside them makes the child exit with code 86
# before it serves anything, so render_config refuses it here instead. Each comment quotes the C condition.
MAX_SLOTS = 8  # `slots`: `a >= 1 && a <= 8`, otherwise "unknown or invalid entry"
MAX_READERS = 16  # `readers`: `a >= 1 && a <= HC_MAX_READERS` (#define HC_MAX_READERS 16)
MIN_STATS_INTERVAL_MS = 50  # `stats_interval_ms`: `a >= 50`
MAX_STATS_INTERVAL_MS = 0xFFFFFFFF  # then `(DWORD)a`: a larger value wraps, to a busy loop below 50 ms for 2**32
MAX_SHARDS = 64  # `shard`: `idx >= HC_MAX_SHARDS` (#define HC_MAX_SHARDS 64) is "shards must be listed in order"
MAX_UNITS = 1024  # `layer` and `tensor`: `a >= HC_MAX_LAYERS` (#define HC_MAX_LAYERS 1024)
MAX_NAME_BYTES = 63  # `tensor`: `strlen(name) >= GGML_MAX_NAME` (64, counted in bytes, the NUL needs the last one)
MAX_LINE_BYTES = 4095  # `fgets(line, sizeof line, f)` with `char line[4096]`: a longer line splits into a bogus one


class StreamConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LayerLayout:
    layer: int
    tensors: tuple[TensorInfo, ...]
    buffer_bytes: int  # aligned bytes one slot (or the resident copy) needs for this layer
    read_bytes: int  # bytes read from the SSD per load (same as buffer_bytes: whole aligned runs)


@dataclass(frozen=True)
class StreamPlan:
    """The resident/streamed split of the CPU layers and the memory it commits."""

    resident_layers: tuple[int, ...]
    stream_layers: tuple[int, ...]
    slots: int
    slot_bytes: int  # largest streamed layer buffer
    resident_bytes: int
    stream_bytes_per_pass: int  # SSD bytes read for every forward pass (one token in decode)

    @property
    def committed_bytes(self) -> int:
        return self.slots * self.slot_bytes + self.resident_bytes


def _down(v: int) -> int:
    return v // ALIGN * ALIGN


def _up(v: int) -> int:
    return (v + ALIGN - 1) // ALIGN * ALIGN


def block_count(table: Iterable[TensorInfo]) -> int:
    return max((t.layer for t in table if t.layer is not None), default=-1) + 1


def unit_of(t: TensorInfo, n_layer: int, tied: bool) -> int | None:
    """Placement unit of a tensor: its block, `n_layer` for the output head (llama.cpp's extra GPU unit), or None.

    With tied embeddings the head matmul reads all of token_embd every token, so token_embd belongs to the head;
    otherwise token_embd is only row-gathered per token and stays file-backed.
    """
    if t.layer is not None:
        return t.layer
    if t.name.startswith("output") or (tied and t.name.startswith("token_embd")):
        return n_layer
    return None


def layer_layouts(table: Iterable[TensorInfo]) -> dict[int, LayerLayout]:
    """Byte layout per placement unit; key `n_layer` is the output head."""
    table = list(table)
    n_layer = block_count(table)
    tied = not any(t.name == "output.weight" for t in table)
    by_layer: dict[int, list[TensorInfo]] = {}
    for t in table:
        unit = unit_of(t, n_layer, tied)
        if unit is not None:
            by_layer.setdefault(unit, []).append(t)
    out: dict[int, LayerLayout] = {}
    for layer, tensors in by_layer.items():
        ordered = sorted(tensors, key=lambda t: (t.shard, t.offset))
        pos = 0
        run: tuple[int, int, int] | None = None  # shard, aligned start, aligned end
        for t in ordered:
            a, b = _down(t.offset), _up(t.offset + t.span)
            if run and run[0] == t.shard and a <= run[2]:
                if b > run[2]:
                    pos += b - run[2]
                    run = (run[0], run[1], b)
            else:
                pos += b - a
                run = (t.shard, a, b)
        out[layer] = LayerLayout(layer=layer, tensors=tuple(ordered), buffer_bytes=pos, read_bytes=pos)
    return out


def stream_plan(layouts: dict[int, LayerLayout], resident: Sequence[int], stream: Sequence[int],
                slots: int) -> StreamPlan:
    if set(resident) & set(stream):
        raise StreamConfigError("a layer cannot be both resident and streamed")
    missing = [layer for layer in (*resident, *stream) if layer not in layouts]
    if missing:
        raise StreamConfigError(f"layers {missing} have no tensors in the GGUF")
    if stream and slots < 1:
        raise StreamConfigError("streaming needs at least one slot")
    slot_bytes = max((layouts[layer].buffer_bytes for layer in stream), default=0)
    return StreamPlan(resident_layers=tuple(sorted(resident)), stream_layers=tuple(sorted(stream)),
                      slots=slots if stream else 0, slot_bytes=slot_bytes,
                      resident_bytes=sum(layouts[layer].buffer_bytes for layer in resident),
                      stream_bytes_per_pass=sum(layouts[layer].read_bytes for layer in stream))


@dataclass(frozen=True)
class StreamOptions:
    readers: int = 4
    chunk_bytes: int = 8 << 20
    no_prefetch: bool = True
    on_io_error: str = "abort"  # or "mmap": fall back to the file mapping (correct but unbounded)
    stats_interval_ms: int = 500
    fault_read_error_after_bytes: int = 0  # test-only fault injection
    fault_read_delay_us: int = 0  # test-only slow-disk simulation


def _check_limits(shards: Sequence[Path], plan: StreamPlan, options: StreamOptions) -> None:
    """Refuse plans and options the native parser would kill the child for (see the MAX_* constants)."""
    if options.on_io_error not in ("abort", "mmap"):
        raise StreamConfigError("on_io_error must be abort or mmap")
    if options.chunk_bytes < ALIGN or options.chunk_bytes % ALIGN:
        raise StreamConfigError("chunk_bytes must be a positive multiple of 4096")
    if plan.slots > MAX_SLOTS or (plan.stream_layers and plan.slots < 1):
        raise StreamConfigError(f"slots must be 1..{MAX_SLOTS}, got {plan.slots}")
    if not 1 <= options.readers <= MAX_READERS:
        raise StreamConfigError(f"readers must be 1..{MAX_READERS}, got {options.readers}")
    if options.stats_interval_ms < MIN_STATS_INTERVAL_MS:
        raise StreamConfigError(f"stats_interval_ms must be at least {MIN_STATS_INTERVAL_MS}, "
                                f"got {options.stats_interval_ms}")
    if options.stats_interval_ms > MAX_STATS_INTERVAL_MS:
        raise StreamConfigError(f"stats_interval_ms must be at most {MAX_STATS_INTERVAL_MS}, "
                                f"got {options.stats_interval_ms}")
    if len(shards) > MAX_SHARDS:
        raise StreamConfigError(f"at most {MAX_SHARDS} shards can be streamed, got {len(shards)}")
    units = sorted({*plan.resident_layers, *plan.stream_layers})
    if units and not 0 <= units[0] <= units[-1] < MAX_UNITS:
        raise StreamConfigError(f"unit indexes must be 0..{MAX_UNITS - 1}, got {units[0]}..{units[-1]}")


def _tensor_line(t: TensorInfo, unit: int, n_shards: int) -> str:
    if t.nbytes is None:
        raise StreamConfigError(f"{t.name} has ggml type {t.ggml_type}, whose size this build does not know")
    if any(c.isspace() for c in t.name) or len(t.name.encode("utf-8")) > MAX_NAME_BYTES:
        raise StreamConfigError(f"tensor name {t.name!r} cannot be streamed: it must be at most {MAX_NAME_BYTES} "
                                "UTF-8 bytes without whitespace")
    if not 0 <= t.shard < n_shards:
        raise StreamConfigError(f"{t.name} is in shard {t.shard}, but only {n_shards} shard paths were given")
    if t.nbytes <= 0 or t.span < t.nbytes:
        raise StreamConfigError(f"{t.name} has {t.nbytes} bytes in a {t.span}-byte span")
    return f"tensor {unit} {t.shard} {t.offset} {t.nbytes} {t.span} {t.name}"


def _tensor_lines(table: Sequence[TensorInfo], managed: set[int], n_shards: int) -> list[str]:
    """One line per tensor of a managed unit; every managed unit must list at least one."""
    n_layer = block_count(table)
    tied = not any(t.name == "output.weight" for t in table)
    lines: list[str] = []
    listed: set[int] = set()
    names: set[str] = set()
    for t in table:
        unit = unit_of(t, n_layer, tied)
        if unit is None or unit not in managed:
            continue
        if t.name in names:  # `duplicate tensor %s`
            raise StreamConfigError(f"tensor {t.name} is listed twice")
        names.add(t.name)
        lines.append(_tensor_line(t, unit, n_shards))
        listed.add(unit)
    if managed - listed:  # `layer %d is managed but lists no tensors`
        raise StreamConfigError(f"units {sorted(managed - listed)} are managed but have no tensors in the table")
    return lines


def _check_lines(lines: Sequence[str]) -> None:
    """Every line must reach the parser whole: fgets splits a long one, and strlen stops at a NUL byte."""
    long = [line.split(" ", 1)[0] for line in lines if len(line.encode("utf-8")) > MAX_LINE_BYTES]
    if long:
        raise StreamConfigError(f"config lines {long} exceed {MAX_LINE_BYTES} UTF-8 bytes (a path is too long)")
    nul = [line.split(" ", 1)[0] for line in lines if "\x00" in line]
    if nul:
        raise StreamConfigError(f"config lines {nul} contain a NUL character")


def render_config(shards: Sequence[Path], table: Sequence[TensorInfo], plan: StreamPlan, stats_path: Path,
                  options: StreamOptions | None = None) -> str:
    """The text config the extension parses; every managed layer lists all of its tensors.

    Shard and stats paths are written absolute: the child resolves a relative one against its own working directory,
    which is the overlay engine directory, not the caller's.
    """
    options = options or StreamOptions()
    _check_limits(shards, plan, options)
    managed = set(plan.resident_layers) | set(plan.stream_layers)
    lines = [f"hcstream {CONFIG_VERSION}", "mode stream", f"slots {max(plan.slots, 1)}",
             f"max_slot_bytes {plan.slot_bytes}", f"max_resident_bytes {plan.resident_bytes}",
             f"readers {options.readers}", f"chunk_bytes {options.chunk_bytes}",
             f"no_prefetch {int(options.no_prefetch)}", f"on_io_error {options.on_io_error}",
             f"stats {os.path.abspath(stats_path)}", f"stats_interval_ms {options.stats_interval_ms}"]
    if options.fault_read_error_after_bytes:
        lines.append(f"fault_read_error_after_bytes {options.fault_read_error_after_bytes}")
    if options.fault_read_delay_us:
        lines.append(f"fault_read_delay_us {options.fault_read_delay_us}")
    for i, shard in enumerate(shards):
        lines.append(f"shard {i} {Path(shard).resolve()}")
    lines += [f"layer {layer} resident" for layer in plan.resident_layers]
    lines += [f"layer {layer} stream" for layer in plan.stream_layers]
    lines += _tensor_lines(table, managed, len(shards))
    lines.append("end")
    _check_lines(lines)
    return "\n".join(lines) + "\n"
