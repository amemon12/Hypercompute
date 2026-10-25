"""hcstream config generation and the overlay engine directory, on synthetic tensor tables and fake engine files.

Nothing here needs the native extension, an engine or a model. The layout tests compare `layer_layouts` with a
line-by-line Python transliteration of `hc_build_runs` in native/hcstream/hcstream.c, because admission trusts the
Python numbers and the extension allocates by the C ones.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import replace
from pathlib import Path

import pytest
from hypercompute import engine
from hypercompute.capacity import overlay
from hypercompute.capacity.stream_config import (
    ALIGN,
    MAX_LINE_BYTES,
    MAX_NAME_BYTES,
    MAX_READERS,
    MAX_SHARDS,
    MAX_SLOTS,
    MAX_STATS_INTERVAL_MS,
    MAX_UNITS,
    MIN_STATS_INTERVAL_MS,
    StreamConfigError,
    StreamOptions,
    StreamPlan,
    block_count,
    layer_layouts,
    render_config,
    stream_plan,
    unit_of,
)
from hypercompute.gguf import TensorInfo

windows_only = pytest.mark.skipif(os.name != "nt", reason="the overlay hard-links the Windows engine release")
NATIVE = Path(__file__).resolve().parents[1] / "native" / "hcstream"


def tensor(name: str, offset: int, nbytes: int, *, span: int | None = None, shard: int = 0) -> TensorInfo:
    return TensorInfo(name=name, shard=shard, ggml_type=0, dims=(max(1, nbytes // 4),), offset=offset,
                      nbytes=nbytes, span=nbytes if span is None else span)


def contiguous(names: list[str], sizes: list[int], *, start: int = 4096, shard: int = 0) -> list[TensorInfo]:
    """Tensors laid out back to back with GGUF's 32-byte alignment padding, like a real file."""
    out: list[TensorInfo] = []
    pos = start
    for name, size in zip(names, sizes, strict=True):
        span = (size + 31) // 32 * 32
        out.append(tensor(name, pos, size, span=span, shard=shard))
        pos += span
    return out


def small_model(*, tied: bool, n_layer: int = 4) -> list[TensorInfo]:
    names = ["token_embd.weight"]
    sizes = [50_000]
    for i in range(n_layer):
        names += [f"blk.{i}.attn_norm.weight", f"blk.{i}.attn_q.weight", f"blk.{i}.ffn_down.weight"]
        sizes += [256, 6_000 + 1_000 * i, 9_000 + 3_000 * i]
    names.append("output_norm.weight")
    sizes.append(256)
    if not tied:
        names.append("output.weight")
        sizes.append(40_000)
    return contiguous(names, sizes)


def c_build_runs(tensors: list[TensorInfo]) -> tuple[int, list[list[int]], dict[str, int]]:
    """Transliteration of hc_build_runs for one layer: (buf_bytes, runs [shard, file_off, len, buf_off], slot_off)."""

    def down(v: int) -> int:
        return v & ~(ALIGN - 1)

    def up(v: int) -> int:
        return (v + ALIGN - 1) & ~(ALIGN - 1)

    runs: list[list[int]] = []
    slot_off: dict[str, int] = {}
    pos = 0
    for t in sorted(tensors, key=lambda x: (x.shard, x.offset)):
        a, b = down(t.offset), up(t.offset + t.span)
        r = runs[-1] if runs else None
        if r and r[0] == t.shard and a <= r[1] + r[2]:
            if b > r[1] + r[2]:
                pos += b - (r[1] + r[2])
                r[2] = b - r[1]
        else:
            r = [t.shard, a, b - a, pos]
            runs.append(r)
            pos += r[2]
        slot_off[t.name] = r[3] + (t.offset - r[1])
    return pos, runs, slot_off


def parse_config(text: str) -> tuple[list[str], dict[str, str], list[tuple[int, str]], list[tuple]]:
    lines = text.splitlines()
    keys: dict[str, str] = {}
    layers: list[tuple[int, str]] = []
    tensors: list[tuple] = []
    for line in lines:
        key, _, rest = line.partition(" ")
        if key == "layer":
            idx, kind = rest.split()
            layers.append((int(idx), kind))
        elif key == "tensor":
            unit, shard, offset, nbytes, span, name = rest.split()
            tensors.append((int(unit), int(shard), int(offset), int(nbytes), int(span), name))
        elif key != "shard":
            keys[key] = rest
    return lines, keys, layers, tensors


# --------------------------------------------------------------------------------------------- layer_layouts


def test_adjacent_tensors_merge_into_one_aligned_run():
    layouts = layer_layouts([tensor("blk.0.a", 8192, 1000, span=1024), tensor("blk.0.b", 9216, 5000)])
    assert layouts[0].buffer_bytes == 8192  # one run [8192, 16384)
    assert layouts[0].read_bytes == layouts[0].buffer_bytes


def test_unaligned_tensor_rounds_out_to_whole_pages():
    layouts = layer_layouts([tensor("blk.0.a", 4100, 10)])
    assert layouts[0].buffer_bytes == ALIGN  # [4096, 8192)


def test_a_page_gap_starts_a_new_run():
    layouts = layer_layouts([tensor("blk.0.a", 0, 100), tensor("blk.0.b", 40960, 100)])
    assert layouts[0].buffer_bytes == 2 * ALIGN


def test_tensor_order_in_the_table_does_not_matter():
    a, b, c = tensor("blk.0.a", 4096, 3000), tensor("blk.0.b", 7096, 3000), tensor("blk.0.c", 81920, 100)
    forward, backward = layer_layouts([a, b, c]), layer_layouts([c, b, a])
    assert forward[0].buffer_bytes == backward[0].buffer_bytes == 3 * ALIGN
    assert [t.name for t in backward[0].tensors] == ["blk.0.a", "blk.0.b", "blk.0.c"]


def test_a_unit_split_across_two_shards_never_merges_across_files():
    # Shard 1's run starts at a lower offset than shard 0's run ends; merging them would undercount by a page.
    table = [tensor("blk.0.b", 0, 4096, shard=1), tensor("blk.0.a", 4096, 4096, shard=0)]
    layout = layer_layouts(table)[0]
    assert layout.buffer_bytes == 2 * ALIGN
    assert [(t.shard, t.name) for t in layout.tensors] == [(0, "blk.0.a"), (1, "blk.0.b")]


def test_untied_head_holds_output_tensors_and_token_embd_stays_unmanaged():
    table = small_model(tied=False)
    layouts = layer_layouts(table)
    n_layer = block_count(table)
    assert n_layer == 4 and sorted(layouts) == [0, 1, 2, 3, 4]
    assert {t.name for t in layouts[n_layer].tensors} == {"output_norm.weight", "output.weight"}
    assert all(t.name != "token_embd.weight" for lay in layouts.values() for t in lay.tensors)


def test_tied_head_owns_token_embd():
    table = small_model(tied=True)
    layouts = layer_layouts(table)
    assert {t.name for t in layouts[4].tensors} == {"output_norm.weight", "token_embd.weight"}
    assert unit_of(table[0], 4, tied=True) == 4
    assert unit_of(table[0], 4, tied=False) is None


def test_unit_of_ignores_tensors_outside_blocks_and_head():
    assert unit_of(tensor("rope_freqs.weight", 0, 64), 4, tied=True) is None
    assert unit_of(tensor("blk.3.attn_q.weight", 0, 64), 4, tied=False) == 3


def test_block_count():
    assert block_count([]) == 0
    assert block_count([tensor("blk.0.a", 0, 4), tensor("blk.3.a", 4096, 4), tensor("output.weight", 8192, 4)]) == 4


def _random_table(rng: random.Random) -> list[TensorInfo]:
    table: list[TensorInfo] = []
    n_layer = rng.randint(1, 5)
    for shard in range(rng.randint(1, 3)):
        count = rng.randint(1, 12)
        names = [f"blk.{rng.randrange(n_layer)}.t{shard}_{i}.weight" for i in range(count)]
        sizes = [rng.choice([32, 96, 4000, 4096, 5000, 20_000, 70_000]) for _ in names]
        table += contiguous(names, sizes, start=rng.choice([32, 4096, 6144]), shard=shard)
    if rng.random() < 0.5:
        table.append(tensor("output.weight", 900_000, 12_345, shard=0))
    return table


@pytest.mark.parametrize("seed", range(40))
def test_layouts_match_the_native_run_builder(seed):
    table = _random_table(random.Random(seed))
    layouts = layer_layouts(table)
    for unit, layout in layouts.items():
        buf_bytes, runs, slot_off = c_build_runs(list(layout.tensors))
        assert layout.buffer_bytes == buf_bytes, f"unit {unit}"
        assert buf_bytes % ALIGN == 0
        assert buf_bytes >= sum(t.nbytes for t in layout.tensors)
        for t in layout.tensors:  # every tensor's bytes sit inside the buffer the extension allocates
            assert 0 <= slot_off[t.name] and slot_off[t.name] + t.nbytes <= buf_bytes


# --------------------------------------------------------------------------------------------- stream_plan


def test_stream_plan_numbers():
    layouts = layer_layouts(small_model(tied=True))
    plan = stream_plan(layouts, resident=[4, 0], stream=[3, 1, 2], slots=2)
    assert plan.resident_layers == (0, 4) and plan.stream_layers == (1, 2, 3)
    assert plan.slot_bytes == max(layouts[i].buffer_bytes for i in (1, 2, 3))
    assert plan.resident_bytes == layouts[0].buffer_bytes + layouts[4].buffer_bytes
    assert plan.stream_bytes_per_pass == sum(layouts[i].read_bytes for i in (1, 2, 3))
    assert plan.committed_bytes == 2 * plan.slot_bytes + plan.resident_bytes


def test_stream_plan_without_streamed_layers_commits_no_slots():
    plan = stream_plan(layer_layouts(small_model(tied=True)), resident=[], stream=[], slots=0)
    assert (plan.slots, plan.slot_bytes, plan.committed_bytes) == (0, 0, 0)


@pytest.mark.parametrize(("resident", "stream", "slots", "message"), [
    ([1], [1, 2], 2, "both resident and streamed"),
    ([], [9], 2, "no tensors"),
    ([7], [], 2, "no tensors"),
    ([], [0], 0, "at least one slot"),
])
def test_stream_plan_rejects_invalid_splits(resident, stream, slots, message):
    with pytest.raises(StreamConfigError, match=message):
        stream_plan(layer_layouts(small_model(tied=True)), resident, stream, slots)


# --------------------------------------------------------------------------------------------- render_config


@pytest.fixture
def rendered(tmp_path):
    table = small_model(tied=False)
    plan = stream_plan(layer_layouts(table), resident=[0, 4], stream=[2, 3], slots=3)
    stats = tmp_path / "stats.json"
    shards = [tmp_path / "m.gguf"]
    return table, plan, stats, shards, render_config(shards, table, plan, stats)


def test_render_config_header(rendered):
    _, plan, stats, shards, text = rendered
    lines, keys, _, _ = parse_config(text)
    assert lines[:2] == ["hcstream 1", "mode stream"]
    assert keys["slots"] == "3"
    assert keys["max_slot_bytes"] == str(plan.slot_bytes)
    assert keys["max_resident_bytes"] == str(plan.resident_bytes)
    assert (keys["readers"], keys["chunk_bytes"], keys["no_prefetch"]) == ("4", str(8 << 20), "1")
    assert keys["on_io_error"] == "abort" and keys["stats_interval_ms"] == "500"
    assert keys["stats"] == str(stats)
    assert f"shard 0 {shards[0].resolve()}" in lines
    assert "fault_read_error_after_bytes" not in keys and "fault_read_delay_us" not in keys


def test_render_config_lists_every_managed_tensor_with_its_unit(rendered):
    table, plan, _, _, text = rendered
    lines, _, layers, tensors = parse_config(text)
    assert layers == [(0, "resident"), (4, "resident"), (2, "stream"), (3, "stream")]
    managed = {0, 2, 3, 4}
    expected = {(unit_of(t, 4, tied=False), t.shard, t.offset, t.nbytes, t.span, t.name)
                for t in table if unit_of(t, 4, tied=False) in managed}
    assert set(tensors) == expected and len(tensors) == len(expected)
    assert {name for unit, *_, name in tensors if unit == 4} == {"output_norm.weight", "output.weight"}
    assert all(0 < nbytes <= span for _, _, _, nbytes, span, _ in tensors)  # the native parser's own checks
    assert lines[-1] == "end" and text.endswith("end\n")


@pytest.mark.parametrize("tied", [True, False])
def test_rendered_tensors_rebuild_the_bytes_the_plan_admitted(tmp_path, tied):
    # The extension builds its buffers from the config's tensor lines, while admission used layer_layouts. Both pick
    # the head's tensors on their own (tied or not), so the rebuilt sizes must equal the admitted plan exactly.
    table = small_model(tied=tied)
    plan = stream_plan(layer_layouts(table), resident=[0, 4], stream=[1, 2, 3], slots=2)
    _, keys, layers, tensors = parse_config(render_config([tmp_path / "m.gguf"], table, plan, tmp_path / "s"))
    by_unit: dict[int, list[TensorInfo]] = {}
    for unit, shard, offset, nbytes, span, name in tensors:
        by_unit.setdefault(unit, []).append(tensor(name, offset, nbytes, span=span, shard=shard))
    sizes = {unit: c_build_runs(ts)[0] for unit, ts in by_unit.items()}
    assert sorted(sizes) == sorted(unit for unit, _ in layers) == [0, 1, 2, 3, 4]
    assert sum(sizes[u] for u in (0, 4)) == plan.resident_bytes == int(keys["max_resident_bytes"])
    assert max(sizes[u] for u in (1, 2, 3)) == plan.slot_bytes == int(keys["max_slot_bytes"])
    head = {t.name for t in by_unit[4]}
    assert ("token_embd.weight" in head) is tied and ("output.weight" in head) is not tied


def test_render_config_without_managed_layers_still_parses(tmp_path):
    table = small_model(tied=True)
    plan = stream_plan(layer_layouts(table), [], [], slots=2)
    lines, keys, layers, tensors = parse_config(render_config([tmp_path / "m.gguf"], table, plan, tmp_path / "s"))
    assert keys["slots"] == "1" and keys["max_slot_bytes"] == "0"
    assert layers == [] and tensors == [] and lines[-1] == "end"


def test_render_config_fault_and_fallback_options(rendered):
    table, plan, stats, shards, _ = rendered
    opts = StreamOptions(on_io_error="mmap", no_prefetch=False, fault_read_error_after_bytes=50 << 20,
                         fault_read_delay_us=2000, chunk_bytes=3 * ALIGN, readers=2, stats_interval_ms=100)
    _, keys, _, _ = parse_config(render_config(shards, table, plan, stats, opts))
    assert keys["on_io_error"] == "mmap" and keys["no_prefetch"] == "0"
    assert keys["fault_read_error_after_bytes"] == str(50 << 20)
    assert keys["fault_read_delay_us"] == "2000"
    assert (keys["chunk_bytes"], keys["readers"], keys["stats_interval_ms"]) == (str(3 * ALIGN), "2", "100")


@pytest.mark.parametrize(("options", "message"), [
    (StreamOptions(on_io_error="retry"), "abort or mmap"),
    (StreamOptions(chunk_bytes=1000), "multiple of 4096"),
    (StreamOptions(chunk_bytes=ALIGN + 1), "multiple of 4096"),
    (StreamOptions(chunk_bytes=0), "multiple of 4096"),
])
def test_render_config_rejects_bad_options(rendered, options, message):
    table, plan, stats, shards, _ = rendered
    with pytest.raises(StreamConfigError, match=message):
        render_config(shards, table, plan, stats, options)


def test_render_config_rejects_unknown_ggml_size_in_a_managed_unit(rendered):
    table, plan, stats, shards, _ = rendered
    bad = [replace(t, nbytes=None) if t.name == "blk.2.attn_q.weight" else t for t in table]
    with pytest.raises(StreamConfigError, match="does not know"):
        render_config(shards, bad, plan, stats)


def test_render_config_ignores_unknown_ggml_size_outside_managed_units(rendered):
    table, plan, stats, shards, _ = rendered
    other = [replace(t, nbytes=None) if t.name.startswith(("blk.1.", "token_embd")) else t for t in table]
    assert render_config(shards, other, plan, stats) == render_config(shards, table, plan, stats)


@pytest.mark.parametrize("name", ["blk.2.attn q.weight", "blk.2.attn\tq.weight", "blk.2." + "x" * 60])
def test_render_config_rejects_names_the_native_parser_cannot_read(rendered, name):
    table, plan, stats, shards, _ = rendered
    bad = [replace(t, name=name) if t.name == "blk.2.attn_q.weight" else t for t in table]
    with pytest.raises(StreamConfigError, match="cannot be streamed"):
        render_config(shards, bad, plan, stats)


# --------------------------------------------------------------------------------------------- native parser limits


def test_parser_limits_match_the_native_source():
    """The Python limits are copies of hc_parse_config's conditions; a change on either side must fail here."""
    src = (NATIVE / "hcstream.c").read_text(encoding="utf-8")
    ggml_h = (NATIVE / "third_party" / "llama.cpp" / "ggml.h").read_text(encoding="utf-8")
    assert f'!strcmp(key, "slots") && sscanf(rest, "%llu", &a) == 1 && a >= 1 && a <= {MAX_SLOTS})' in src
    assert f"#define HC_MAX_READERS {MAX_READERS}\n" in src
    assert "a >= 1 && a <= HC_MAX_READERS)" in src
    assert f'"stats_interval_ms") && sscanf(rest, "%llu", &a) == 1 && a >= {MIN_STATS_INTERVAL_MS})' in src
    assert "st->stats_interval_ms = (DWORD)a;" in src and MAX_STATS_INTERVAL_MS == 0xFFFFFFFF
    assert 'hc_die(NULL, "duplicate tensor %s"' in src
    assert f"#define HC_MAX_SHARDS {MAX_SHARDS}\n" in src and "idx >= HC_MAX_SHARDS" in src
    assert f"#define HC_MAX_LAYERS {MAX_UNITS}\n" in src and src.count("a >= HC_MAX_LAYERS") == 2  # layer, tensor
    assert "strlen(name) >= GGML_MAX_NAME" in src
    max_name = re.search(r"#\s*define GGML_MAX_NAME\s+(\d+)", ggml_h)
    assert max_name is not None and int(max_name.group(1)) == MAX_NAME_BYTES + 1
    assert f"char line[{MAX_LINE_BYTES + 1}];" in src and "fgets(line, sizeof line, f)" in src


@pytest.mark.parametrize(("change", "message"), [
    ({"readers": 0}, r"readers must be 1\.\.16, got 0"),
    ({"readers": MAX_READERS + 1}, r"readers must be 1\.\.16, got 17"),
    ({"stats_interval_ms": MIN_STATS_INTERVAL_MS - 1}, "stats_interval_ms must be at least 50, got 49"),
    ({"stats_interval_ms": 0}, "stats_interval_ms must be at least 50"),
])
def test_render_config_rejects_options_the_native_parser_kills_for(rendered, change, message):
    table, plan, stats, shards, _ = rendered
    with pytest.raises(StreamConfigError, match=message):
        render_config(shards, table, plan, stats, StreamOptions(**change))


@pytest.mark.parametrize("change", [{"readers": 1}, {"readers": MAX_READERS},
                                    {"stats_interval_ms": MIN_STATS_INTERVAL_MS}])
def test_render_config_accepts_option_limits(rendered, change):
    table, plan, stats, shards, _ = rendered
    assert render_config(shards, table, plan, stats, StreamOptions(**change)).endswith("end\n")


@pytest.mark.parametrize("slots", [0, MAX_SLOTS + 1, 64])
def test_render_config_rejects_slot_counts_outside_1_to_8(rendered, slots):
    table, plan, stats, shards, _ = rendered
    with pytest.raises(StreamConfigError, match=rf"slots must be 1\.\.8, got {slots}"):
        render_config(shards, table, replace(plan, slots=slots), stats)


def test_render_config_slot_limits(rendered, tmp_path):
    table, plan, stats, shards, _ = rendered
    _, keys, _, _ = parse_config(render_config(shards, table, replace(plan, slots=MAX_SLOTS), stats))
    assert keys["slots"] == "8"
    idle = stream_plan(layer_layouts(table), [0], [], slots=0)  # nothing streamed: 'slots 1' is written
    assert parse_config(render_config(shards, table, idle, stats))[1]["slots"] == "1"
    with pytest.raises(StreamConfigError, match="got 9"):  # even with nothing streamed the parser would refuse it
        render_config(shards, table, replace(idle, slots=MAX_SLOTS + 1), stats)


def test_render_config_rejects_more_than_64_shards(rendered, tmp_path):
    table, plan, stats, _, _ = rendered
    many = [tmp_path / f"m-{i:05d}.gguf" for i in range(MAX_SHARDS + 1)]
    with pytest.raises(StreamConfigError, match="at most 64 shards can be streamed, got 65"):
        render_config(many, table, plan, stats)
    lines = render_config(many[:MAX_SHARDS], table, plan, stats).splitlines()
    assert sum(line.startswith("shard ") for line in lines) == MAX_SHARDS


@pytest.mark.parametrize(("resident", "stream"), [((MAX_UNITS,), ()), ((), (MAX_UNITS, MAX_UNITS + 5)), ((-1,), ())])
def test_render_config_rejects_unit_indexes_from_1024(tmp_path, resident, stream):
    table = small_model(tied=True)
    plan = StreamPlan(resident_layers=resident, stream_layers=stream, slots=2 if stream else 0, slot_bytes=0,
                      resident_bytes=0, stream_bytes_per_pass=0)
    with pytest.raises(StreamConfigError, match=r"unit indexes must be 0\.\.1023"):
        render_config([tmp_path / "m.gguf"], table, plan, tmp_path / "s")


def test_render_config_accepts_unit_1023(tmp_path):
    table = [tensor("blk.1023.ffn_down.weight", 4096, 5000)]
    plan = stream_plan(layer_layouts(table), [MAX_UNITS - 1], [], slots=0)
    _, _, layers, tensors = parse_config(render_config([tmp_path / "m.gguf"], table, plan, tmp_path / "s"))
    assert layers == [(1023, "resident")] and tensors[0][0] == 1023


@pytest.mark.parametrize(("name", "ok"), [
    ("blk.2." + "x" * 57, True),  # 63 bytes
    ("blk.2." + "x" * 58, False),  # 64 bytes
    ("blk.2." + "\u00e9" * 28 + "x", True),  # 35 characters, 63 bytes
    ("blk.2." + "\u00e9" * 29, False),  # 35 characters, 64 bytes: a character count would let it through
    ("blk.2." + "\u6f22" * 20, False),  # 26 characters, 66 bytes
])
def test_render_config_counts_tensor_name_limit_in_utf8_bytes(rendered, name, ok):
    table, plan, stats, shards, _ = rendered
    renamed = [replace(t, name=name) if t.name == "blk.2.attn_q.weight" else t for t in table]
    if ok:
        assert f" {name}" in render_config(shards, renamed, plan, stats)
    else:
        with pytest.raises(StreamConfigError, match="at most 63 UTF-8 bytes"):
            render_config(shards, renamed, plan, stats)


@pytest.mark.parametrize(("change", "message"), [
    ({"shard": 1}, "in shard 1, but only 1 shard paths"),  # `(int)b >= st->n_shards`
    ({"nbytes": 0}, "has 0 bytes"),  # `d == 0`
    ({"span": 100}, "in a 100-byte span"),  # `e < d`
])
def test_render_config_rejects_tensor_lines_the_parser_refuses(rendered, change, message):
    table, plan, stats, shards, _ = rendered
    bad = [replace(t, **change) if t.name == "blk.2.attn_q.weight" else t for t in table]
    with pytest.raises(StreamConfigError, match=message):
        render_config(shards, bad, plan, stats)


def test_render_config_rejects_a_managed_unit_without_tensors(tmp_path):
    # stream_plan refuses this, but a hand-built plan would reach `layer %d is managed but lists no tensors`
    table = small_model(tied=True)
    plan = StreamPlan(resident_layers=(0, 7), stream_layers=(), slots=0, slot_bytes=0, resident_bytes=0,
                      stream_bytes_per_pass=0)
    with pytest.raises(StreamConfigError, match=r"units \[7\] are managed but have no tensors"):
        render_config([tmp_path / "m.gguf"], table, plan, tmp_path / "s")


def test_render_config_rejects_lines_longer_than_the_parser_buffer(rendered, tmp_path):
    table, plan, _, shards, _ = rendered
    root = Path(tmp_path.anchor)  # 'C:\' or '/': absolute, so the rendered path is exactly this one
    fits = root / ("s" * (MAX_LINE_BYTES - len(f"stats {root}")))
    assert len(f"stats {fits}".encode()) == MAX_LINE_BYTES
    assert f"stats {fits}\n" in render_config(shards, table, plan, fits)
    with pytest.raises(StreamConfigError, match=r"config lines \['stats'\] exceed 4095 UTF-8 bytes"):
        render_config(shards, table, plan, Path(str(fits) + "s"))
    wide = root / ("\u00e9" * 2045)  # 2045 characters, 4090 bytes after 'stats ' and the root
    with pytest.raises(StreamConfigError, match="exceed 4095 UTF-8 bytes"):
        render_config(shards, table, plan, wide)


def test_render_config_writes_a_relative_stats_path_as_absolute(rendered, tmp_path, monkeypatch):
    # The child runs in the overlay engine directory: a relative stats path would put the file there, where the
    # caller never looks (the validator's relative --out failed every overlay case this way).
    table, plan, _, shards, _ = rendered
    monkeypatch.chdir(tmp_path)
    _, keys, _, _ = parse_config(render_config(shards, table, plan, Path("runs") / "s.json"))
    assert Path(keys["stats"]).is_absolute() and keys["stats"] == str(tmp_path / "runs" / "s.json")


def test_render_config_rejects_a_managed_tensor_listed_twice(rendered):
    # read_tensor_table refuses duplicates, but render_config takes any table; the parser dies on `duplicate tensor`
    table, plan, stats, shards, _ = rendered
    twin = next(t for t in table if t.name == "blk.2.attn_q.weight")
    with pytest.raises(StreamConfigError, match="tensor blk.2.attn_q.weight is listed twice"):
        render_config(shards, [*table, replace(twin, offset=twin.offset + 4096)], plan, stats)
    unmanaged = next(t for t in table if t.name == "blk.1.attn_q.weight")  # unit 1 is neither resident nor streamed
    assert render_config(shards, [*table, unmanaged], plan, stats) == render_config(shards, table, plan, stats)


def test_render_config_rejects_nul_characters(rendered, tmp_path):
    # fgets keeps the NUL, then strlen and sscanf stop at it: the tensor line loses fields, the path its tail
    table, plan, stats, shards, _ = rendered
    bad = [replace(t, name="blk.2.attn\x00q.weight") if t.name == "blk.2.attn_q.weight" else t for t in table]
    with pytest.raises(StreamConfigError, match=r"config lines \['tensor'\] contain a NUL"):
        render_config(shards, bad, plan, stats)
    with pytest.raises(StreamConfigError, match=r"config lines \['stats'\] contain a NUL"):
        render_config(shards, table, plan, tmp_path / "s\x00.json")


def test_render_config_rejects_stats_intervals_that_wrap_the_native_dword(rendered):
    table, plan, stats, shards, _ = rendered
    top = StreamOptions(stats_interval_ms=MAX_STATS_INTERVAL_MS)
    assert parse_config(render_config(shards, table, plan, stats, top))[1]["stats_interval_ms"] == "4294967295"
    with pytest.raises(StreamConfigError, match="stats_interval_ms must be at most 4294967295, got 4294967296"):
        render_config(shards, table, plan, stats, StreamOptions(stats_interval_ms=1 << 32))


# --------------------------------------------------------------------------------------------- overlay


@pytest.fixture
def fake_engine(tmp_path):
    eng = tmp_path / "engines" / "llama-b1-fake"
    eng.mkdir(parents=True)
    for name, data in (("ggml.dll", b"real ggml"), ("llama.dll", b"llama"), ("llama-server.exe", b"server"),
                       (engine.MARKER, b"{}")):
        (eng / name).write_bytes(data)
    (eng / "subdir").mkdir()
    proxy = tmp_path / "build" / "ggml.dll"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy v1")
    return eng, proxy, tmp_path / "overlays"


@windows_only
def test_ensure_overlay_links_the_release_and_copies_the_proxy(fake_engine):
    eng, proxy, root = fake_engine
    ov = overlay.ensure_overlay(eng, proxy, root=root)
    digest = overlay._sha256(proxy)
    assert ov.bin_dir == root / f"{eng.name}-hcstream-{digest[:12]}" and ov.proxy_sha256 == digest
    assert ov.engine_dir == eng
    assert os.path.samefile(ov.bin_dir / "llama.dll", eng / "llama.dll")
    assert os.path.samefile(ov.bin_dir / "llama-server.exe", eng / "llama-server.exe")
    assert os.path.samefile(ov.bin_dir / overlay.ORIG_NAME, eng / "ggml.dll")
    assert (ov.bin_dir / overlay.PROXY_NAME).read_bytes() == b"proxy v1"
    assert not os.path.samefile(ov.bin_dir / overlay.PROXY_NAME, proxy)  # a rebuild must not edit a live overlay
    assert not (ov.bin_dir / engine.MARKER).exists() and not (ov.bin_dir / "subdir").exists()
    meta = json.loads((ov.bin_dir / overlay.MARKER).read_text(encoding="utf-8"))
    assert meta == {"engine_dir": str(eng), "proxy_sha256": digest, "orig": overlay.ORIG_NAME}
    assert [p.name for p in root.iterdir()] == [ov.bin_dir.name]  # no temporary directory left behind


@windows_only
def test_ensure_overlay_is_idempotent(fake_engine):
    eng, proxy, root = fake_engine
    first = overlay.ensure_overlay(eng, proxy, root=root)
    marker = first.bin_dir / overlay.MARKER
    before = (marker.stat().st_mtime_ns, (first.bin_dir / overlay.PROXY_NAME).stat().st_ino)
    second = overlay.ensure_overlay(eng, proxy, root=root)
    assert second == first
    assert (marker.stat().st_mtime_ns, (first.bin_dir / overlay.PROXY_NAME).stat().st_ino) == before


@windows_only
def test_a_changed_proxy_gets_a_new_directory(fake_engine):
    eng, proxy, root = fake_engine
    old = overlay.ensure_overlay(eng, proxy, root=root)
    proxy.write_bytes(b"proxy v2")
    new = overlay.ensure_overlay(eng, proxy, root=root)
    assert new.bin_dir != old.bin_dir and new.proxy_sha256 != old.proxy_sha256
    assert (new.bin_dir / overlay.PROXY_NAME).read_bytes() == b"proxy v2"
    assert (old.bin_dir / overlay.PROXY_NAME).read_bytes() == b"proxy v1"  # the old overlay is left intact


@windows_only
@pytest.mark.parametrize("side", ["overlay", "engine"])
def test_validation_fails_when_a_linked_file_is_replaced(fake_engine, side):
    eng, proxy, root = fake_engine
    ov = overlay.ensure_overlay(eng, proxy, root=root)
    victim = (ov.bin_dir if side == "overlay" else eng) / "llama.dll"
    victim.unlink()
    victim.write_bytes(b"llama")  # same bytes, different file: the overlay no longer runs the pinned release
    assert not overlay._valid(ov.bin_dir, eng, ov.proxy_sha256)
    with pytest.raises(overlay.OverlayError, match="failed validation"):
        overlay.ensure_overlay(eng, proxy, root=root)


@windows_only
def test_validation_fails_when_the_overlay_proxy_is_edited(fake_engine):
    eng, proxy, root = fake_engine
    ov = overlay.ensure_overlay(eng, proxy, root=root)
    (ov.bin_dir / overlay.PROXY_NAME).write_bytes(b"tampered")
    assert not overlay._valid(ov.bin_dir, eng, ov.proxy_sha256)


@windows_only
def test_ensure_overlay_needs_the_real_ggml(fake_engine):
    eng, proxy, root = fake_engine
    (eng / "ggml.dll").unlink()
    with pytest.raises(overlay.OverlayError, match="missing"):
        overlay.ensure_overlay(eng, proxy, root=root)


def test_ensure_overlay_refuses_other_hosts(fake_engine, monkeypatch):
    eng, proxy, root = fake_engine
    monkeypatch.setattr(engine, "host_os", lambda: "linux")
    with pytest.raises(overlay.OverlayError, match="Windows-only"):
        overlay.ensure_overlay(eng, proxy, root=root)
    assert not root.exists()


def test_find_proxy_dll_honours_the_override(tmp_path, monkeypatch):
    dll = tmp_path / "ggml.dll"
    dll.write_bytes(b"x")
    monkeypatch.setenv(overlay.DLL_ENV, str(dll))
    assert overlay.find_proxy_dll() == dll
    monkeypatch.setenv(overlay.DLL_ENV, str(tmp_path / "missing.dll"))
    assert overlay.find_proxy_dll() is None
