"""hcstream native extension: the validation matrix rules, and a reduced run on real llama.cpp children.

The portable tests check scripts/capacity/validate_hcstream.py itself: the case matrix, the SSD read estimate and
the pass/fail rules, on synthetic tensor tables and synthetic child outcomes, plus the child handling on short-lived
Python children (exit code, stop at the deadline, stats polling). The hardware tests (deselected by
default, run with `uv run pytest tests/test_hcstream_native.py -m hardware -q`) start real CPU-only children through
the same helpers: stock against two-slot streaming, a malformed config, an injected read error with
`on_io_error abort`, a config without managed layers (prefetch off only), and the final shutdown and fatal snapshots
while another reader holds the stats file open. They skip unless the proxy DLL
native/hcstream/build/ggml.dll, the installed engine and ~/.hypercompute/models/qwen2.5-0.5b-instruct-q4_k_m exist.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from hypercompute.capacity.stream_config import StreamOptions, StreamPlan, layer_layouts, render_config, stream_plan
from hypercompute.gguf import TensorInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "capacity"))

import validate_hcstream as vh  # noqa: E402

MIB = 1 << 20
GIB = 1 << 30
N_LAYER = 24
# Short timeouts and 12 tokens keep a hardware run near 3.5 GB of streamed reads (14 passes x 247 MB on the 0.5B).
HW = vh.Settings(tokens=12, load_timeout_s=60.0, request_timeout_s=120.0, exit_timeout_s=60.0)


def synthetic_table(n_layer: int = N_LAYER) -> tuple[TensorInfo, ...]:
    """A 0.5B-shaped table: n_layer blocks of about 10 MiB, an untied 100 MiB output head and a token embedding."""
    sizes = [("token_embd.weight", 130 * MIB)]
    for i in range(n_layer):
        sizes += [(f"blk.{i}.attn_norm.weight", 3584), (f"blk.{i}.ffn_down.weight", 10 * MIB + 1000)]
    sizes += [("output_norm.weight", 3584), ("output.weight", 100 * MIB)]
    out: list[TensorInfo] = []
    pos = 4096
    for name, size in sizes:
        span = (size + 31) // 32 * 32
        out.append(TensorInfo(name=name, shard=0, ggml_type=0, dims=(size // 4,), offset=pos, nbytes=size, span=span))
        pos += span
    return tuple(out)


def synthetic_target(tmp_path: Path) -> vh.Target:
    return vh.Target(model=tmp_path / "m", shards=(tmp_path / "m" / "m.gguf",), table=synthetic_table(),
                     n_layer=N_LAYER, stock_bin=tmp_path / "engine", overlay_bin=tmp_path / "overlay",
                     proxy_sha256="0" * 64)


def outcome(**fields: object) -> vh.Outcome:
    base: dict[str, object] = {"served": True, "exit_code": None, "completions": (), "stats": None, "log_text": "",
                               "seconds": 1.0, "memory": {}, "error": None, "resident_peak": 0,
                               "slot_peak": 0}
    return vh.Outcome(**{**base, **fields})  # type: ignore[arg-type]


def failed(result: vh.CaseResult) -> str:
    return "; ".join(f"{c.name}: {c.detail}" for c in result.checks if not c.ok) + f" log={result.log}"


def by_name(checks: tuple[vh.Check, ...] | list[vh.Check]) -> dict[str, bool]:
    return {c.name: c.ok for c in checks}


# --------------------------------------------------------------------------------------------------- matrix


def test_matrix_covers_the_native_tier_gate():
    cases = {c.name: c for c in vh.build_matrix(N_LAYER)}
    assert sorted(cases) == ["a-stock", "b-stream-k1", "c-stream-k2", "d-stream-k3", "e-mixed-k2", "f-noprefetch",
                             "g-fault-abort", "h-fault-mmap", "i-slow-disk", "j-malformed", "k-teardown"]
    assert cases["a-stock"].engine == "stock" and cases["a-stock"].completions == 2
    assert [cases[n].slots for n in ("b-stream-k1", "c-stream-k2", "d-stream-k3")] == [1, 2, 3]
    assert all(cases[n].stream == "0-23" and not cases[n].resident for n in ("b-stream-k1", "c-stream-k2"))
    assert (cases["e-mixed-k2"].resident, cases["e-mixed-k2"].stream) == ("head,0-11", "12-23")
    assert not cases["f-noprefetch"].stream and not cases["f-noprefetch"].resident
    g, h = cases["g-fault-abort"].options, cases["h-fault-mmap"].options
    assert g.fault_read_error_after_bytes == h.fault_read_error_after_bytes == 50 * MIB
    assert (g.on_io_error, h.on_io_error) == ("abort", "mmap")
    assert cases["i-slow-disk"].options.fault_read_delay_us == 2000
    assert cases["j-malformed"].malformed and cases["j-malformed"].expect == "fatal"
    assert cases["k-teardown"].expect == "teardown" and cases["k-teardown"].completions == 0
    assert cases["k-teardown"].options.stats_interval_ms == 100  # frequent snapshots catch the resident peak


def test_select_cases_keeps_the_stock_baseline_first():
    matrix = vh.build_matrix(N_LAYER)
    assert [c.name for c in vh.select_cases(matrix, "j-malformed, c-stream-k2")] == [
        "a-stock", "c-stream-k2", "j-malformed"]
    assert vh.select_cases(matrix, None) == matrix
    with pytest.raises(vh.SetupError, match="unknown cases"):
        vh.select_cases(matrix, "z-nope")


def test_plans_follow_the_case_layer_specs(tmp_path):
    target = synthetic_target(tmp_path)
    cases = {c.name: c for c in vh.build_matrix(N_LAYER)}
    assert vh.plan_for(target, cases["a-stock"]) is None
    mixed = vh.plan_for(target, cases["e-mixed-k2"])
    assert mixed is not None
    assert mixed.resident_layers == (*range(12), N_LAYER) and mixed.stream_layers == tuple(range(12, 24))
    noprefetch = vh.plan_for(target, cases["f-noprefetch"])
    assert noprefetch is not None and noprefetch.committed_bytes == 0


def test_ssd_estimate_for_the_whole_matrix_stays_under_the_budget(tmp_path):
    target = synthetic_target(tmp_path)
    s = vh.Settings()
    total = 0
    for case in vh.build_matrix(N_LAYER):
        plan = vh.plan_for(target, case)
        est = vh.estimate_ssd_bytes(plan, case, s)
        if case.options.fault_read_error_after_bytes:  # every read after the threshold fails without disk I/O
            opts = case.options
            assert plan is not None and est == opts.fault_read_error_after_bytes + opts.readers * opts.chunk_bytes
        if case.malformed or case.engine == "stock":
            assert est == 0
        total += est
    assert 20 * GIB < total < 40 * GIB


def test_ssd_estimate_counts_resident_copies_and_every_pass():
    plan = StreamPlan(resident_layers=(0,), stream_layers=(1, 2), slots=2, slot_bytes=8192, resident_bytes=4096,
                      stream_bytes_per_pass=16384)
    case = vh.Case("x", "x", "tokens", stream="1-2", resident="0", completions=3)
    s = vh.Settings(tokens=5)
    assert vh.estimate_ssd_bytes(plan, case, s) == 4096 + (5 + vh.EXTRA_PASSES) * 3 * 16384
    teardown = replace(case, expect="teardown", completions=0)
    assert vh.estimate_ssd_bytes(plan, teardown, s) == 4096 + (vh.TEARDOWN_TOKENS + vh.EXTRA_PASSES) * 16384


def test_ssd_estimate_with_a_read_fault_allows_one_chunk_in_flight_per_reader():
    # Each reader checks the threshold before its chunk, against bytes_read counted only after finished chunks, so
    # up to `readers` whole chunks can start just below it. Measured on the 0.5B: 59 MiB read for a 50 MiB threshold.
    plan = StreamPlan(resident_layers=(), stream_layers=(0, 1), slots=2, slot_bytes=10 * MIB, resident_bytes=0,
                      stream_bytes_per_pass=20 * MIB)
    opts = StreamOptions(fault_read_error_after_bytes=50 * MIB, readers=4, chunk_bytes=8 * MIB)
    case = vh.Case("g", "g", "fatal", stream="0-1", options=opts)
    assert vh.estimate_ssd_bytes(plan, case, vh.Settings()) == 50 * MIB + 4 * 8 * MIB
    late = replace(case, options=replace(opts, fault_read_error_after_bytes=1 << 40))
    assert vh.estimate_ssd_bytes(plan, late, vh.Settings()) == (24 + vh.EXTRA_PASSES) * 20 * MIB


def test_truncated_config_keeps_whole_lines_and_drops_end(tmp_path):
    table = synthetic_table()
    plan = stream_plan(layer_layouts(table), [], list(range(N_LAYER)), 2)
    text = render_config([tmp_path / "m.gguf"], table, plan, tmp_path / "s.json")
    cut = vh.truncate_config(text)
    full_lines, cut_lines = text.splitlines(), cut.splitlines()
    assert cut_lines[0] == "hcstream 1" and "end" not in cut_lines
    assert cut_lines == full_lines[: len(cut_lines)] and len(cut_lines) < len(full_lines) - 1
    assert any(line.startswith("tensor ") for line in cut_lines)  # cut inside the tensor list
    with pytest.raises(ValueError, match="ending in 'end'"):
        vh.truncate_config("hcstream 1\nmode stream\n")


# --------------------------------------------------------------------------------------------------- pass/fail rules


def test_token_rules_need_a_live_child_and_identical_tokens():
    case = vh.Case("c", "c", "tokens", stream="0-23")
    good = outcome(completions=({"tokens": list(range(24))},))
    assert all(by_name(vh.token_checks(good, case, tuple(range(24)), vh.Settings())).values())
    other = by_name(vh.token_checks(good, case, tuple(range(1, 25)), vh.Settings()))
    assert not other["tokens_match_stock"]
    dead = by_name(vh.token_checks(outcome(completions=good.completions, exit_code=86,
                                           log_text="hcstream: FATAL: x"), case, tuple(range(24)), vh.Settings()))
    assert not dead["alive_after_completions"] and not dead["no_fatal"]
    short = by_name(vh.token_checks(outcome(completions=({"tokens": [1, 2]},)), case, (1, 2), vh.Settings()))
    assert not short["token_count"]


def test_an_overlay_case_without_a_stock_baseline_fails():
    case = vh.Case("c", "c", "tokens", stream="0-23")
    checks = vh.evaluate(case, None, outcome(completions=({"tokens": list(range(24))},)), None, vh.Settings())
    assert by_name(checks)["stock_baseline"] is False


@pytest.mark.parametrize(("exit_code", "log", "served", "ok"), [
    (86, "hcstream: FATAL: config x is truncated (no 'end' line)", False, True),
    (0, "", False, False),
    (86, "", False, False),  # exit code alone is not enough: the reason must be logged
    (86, "hcstream: FATAL: config x is truncated (no 'end' line)", True, False),  # must refuse before serving
])
def test_malformed_config_rules(exit_code, log, served, ok):
    case = vh.Case("j", "j", "fatal", stream="0-23", malformed=True)
    checks = vh.fatal_checks(outcome(exit_code=exit_code, log_text=log, served=served), case)
    assert all(c.ok for c in checks) is ok


FAULT_LINE = "hcstream: FATAL: unbuffered read failed for layer 6 (win32 error {code} at offset {offset})"


def test_fault_abort_rules_read_the_fatal_stats():
    case = vh.Case("g", "g", "fatal", stream="0-23", options=StreamOptions(fault_read_error_after_bytes=50 * MIB))
    log = FAULT_LINE.format(code=30, offset=123731968)
    good = outcome(exit_code=86, log_text=log, stats={"state": "fatal", "fault_injected": 2})
    assert all(c.ok for c in vh.fatal_checks(good, case))
    stale = by_name(vh.fatal_checks(replace(good, stats={"state": "running", "fault_injected": 0}), case))
    assert not stale["stats_state_fatal"] and not stale["fault_injected"]


@pytest.mark.parametrize(("code", "offset", "failing"), [
    (0, 0, {"fatal_error_code", "fatal_offset"}),  # the pre-fix line: the injected fault set no error at all
    (30, 0, {"fatal_offset"}),
    (5, 123731968, {"fatal_error_code"}),  # a real access error is not the injected fault
    (30, 123731968 + 512, {"fatal_offset"}),  # chunks start on 4 KiB boundaries
    (30, -4096, {"fatal_offset"}),
])
def test_fault_abort_line_must_name_read_fault_and_offset(code, offset, failing):
    checks = by_name(vh.read_fault_checks(FAULT_LINE.format(code=code, offset=offset)))
    assert {name for name, ok in checks.items() if not ok} == failing
    assert vh.ERROR_READ_FAULT == 30


def test_fault_abort_without_a_fatal_read_line_fails_every_read_check():
    checks = by_name(vh.read_fault_checks("hcstream: FATAL: cannot commit slot 0 (1 bytes)"))
    assert checks == {"reason_read_failed": False, "fatal_error_code": False, "fatal_offset": False}


TEARDOWN_STATS = {"state": "shutdown", "resident_bytes": 0, "resident_bytes_planned": 4096, "bytes_read": 8192,
                  "slots": 2, "slot_bytes": 8192, "slot_bytes_committed": 0}
TEARDOWN_LOG = "hcstream: attached\nhcstream: detached: buffers released"


def test_teardown_rules_need_released_buffers():
    good = outcome(exit_code=0, stats=TEARDOWN_STATS, log_text=TEARDOWN_LOG, resident_peak=4096, slot_peak=16384)
    assert all(c.ok for c in vh.teardown_checks(good))
    leaked = by_name(vh.teardown_checks(replace(good, stats={**TEARDOWN_STATS, "resident_bytes": 4096})))
    assert not leaked["resident_released"]
    running = by_name(vh.teardown_checks(replace(good, stats={**TEARDOWN_STATS, "state": "running"})))
    assert not running["state_shutdown"]


@pytest.mark.parametrize(("stats", "slot_peak", "failing"), [
    ({**TEARDOWN_STATS, "slot_bytes_committed": 16384}, 16384, "slots_released"),  # the slots were never freed
    ({k: v for k, v in TEARDOWN_STATS.items() if k != "slot_bytes_committed"}, 16384, "slots_released"),  # old DLL
    (TEARDOWN_STATS, 0, "slots_were_committed"),  # no live snapshot ever showed the slots
    (TEARDOWN_STATS, 8192, "slots_were_committed"),  # one slot of two
    ({**TEARDOWN_STATS, "slots": 0, "slot_bytes": 0}, 0, "slots_were_committed"),  # nothing streamed: no proof
])
def test_teardown_proves_slot_release_from_the_stats(stats, slot_peak, failing):
    checks = by_name(vh.teardown_checks(outcome(exit_code=0, stats=stats, log_text=TEARDOWN_LOG, resident_peak=4096,
                                                slot_peak=slot_peak)))
    assert [name for name, ok in checks.items() if not ok] == [failing]


def test_teardown_needs_a_running_snapshot_with_the_resident_copy():
    # bytes_read also counts streamed layers, so it exceeds the resident plan even when no resident layer was loaded
    # (k-teardown read 1309 MiB against 258 MiB planned resident). Only a live snapshot shows the resident copy.
    stats = {"state": "shutdown", "resident_bytes": 0, "resident_bytes_planned": 270639104, "bytes_read": 1372269408}
    log = "hcstream: attached\nhcstream: detached: buffers released"
    for peak in (0, 270639104 - 4096):
        checks = by_name(vh.teardown_checks(outcome(exit_code=0, stats=stats, log_text=log, resident_peak=peak)))
        assert not checks["resident_was_loaded"]


def test_layout_rules_compare_the_extension_stats_with_the_python_plan():
    plan = StreamPlan(resident_layers=(), stream_layers=(0, 1), slots=2, slot_bytes=8192, resident_bytes=0,
                      stream_bytes_per_pass=12288)
    case = vh.Case("c", "c", "tokens", stream="0-1")
    stats = {"slot_bytes": 8192, "stream_bytes_per_pass": 12288, "resident_bytes_planned": 0, "streamed_layers": 2,
             "verify_failures": 0, "slots": 2, "resident_bytes": 0, "io_errors": 0, "passes": 3, "binds": 6,
             "slot_bytes_committed": 16384}
    assert all(c.ok for c in vh.layout_checks(plan, case, stats, 1))
    assert not by_name(vh.layout_checks(plan, case, {**stats, "slot_bytes": 4096}, 1))["stats_slot_bytes"]
    # a live token case must show both slots committed; a missing counter (an old proxy build) fails too
    one_slot = by_name(vh.layout_checks(plan, case, {**stats, "slot_bytes_committed": 8192}, 1))
    assert not one_slot["stats_slot_bytes_committed"]
    old = {k: v for k, v in stats.items() if k != "slot_bytes_committed"}
    assert not by_name(vh.layout_checks(plan, case, old, 1))["stats_slot_bytes_committed"]
    assert vh.layout_checks(plan, case, None, 1) == [vh.Check("stats_present", False, "no stats file")]


def _streamed_stats(passes: int, binds: int) -> dict:
    return {"slot_bytes": 8192, "stream_bytes_per_pass": 3 * 8192, "resident_bytes_planned": 0, "streamed_layers": 3,
            "verify_failures": 0, "slots": 2, "resident_bytes": 0, "io_errors": 0, "passes": passes, "binds": binds,
            "slot_bytes_committed": 2 * 8192}


def test_streamed_passes_must_cover_every_generated_token():
    # Streaming that stopped after the loader's warmup still gives stock tokens (the mapping holds the same bytes),
    # so one pass is not evidence that the completion itself ran from the slots.
    plan = StreamPlan(resident_layers=(), stream_layers=(0, 1, 2), slots=2, slot_bytes=8192, resident_bytes=0,
                      stream_bytes_per_pass=3 * 8192)
    case = vh.Case("c", "c", "tokens", stream="0-2", completions=2)
    s = vh.Settings(tokens=24)
    need = vh.min_passes(plan, case, s)
    assert need == 48
    assert by_name(vh.layout_checks(plan, case, _streamed_stats(need, 3 * need), need))["stats_passes"]
    assert not by_name(vh.layout_checks(plan, case, _streamed_stats(1, 3), need))["stats_passes"]
    teardown = replace(case, expect="teardown", completions=0)
    assert vh.min_passes(plan, teardown, s) == vh.TEARDOWN_TOKENS
    fault = replace(case, options=StreamOptions(fault_read_error_after_bytes=MIB, on_io_error="mmap"))
    assert vh.min_passes(plan, fault, s) == 0  # the injected error stops streaming early by design
    fits = StreamPlan(resident_layers=(), stream_layers=(0,), slots=2, slot_bytes=8192, resident_bytes=0,
                      stream_bytes_per_pass=8192)
    assert vh.min_passes(fits, case, s) == 0  # a layer that never leaves its slot is bound once
    one = by_name(vh.layout_checks(fits, case, {**_streamed_stats(1, 1), "streamed_layers": 1,
                                                "stream_bytes_per_pass": 8192}, 0))
    assert one["stats_passes"] and "stats_binds" not in one  # still at least one streamed pass
    single = vh.min_passes(plan, replace(case, completions=1), vh.Settings(tokens=1))
    assert single == 1 and "stats_binds" in by_name(vh.layout_checks(plan, case, _streamed_stats(3, 9), single))


def test_every_streamed_layer_is_bound_from_a_slot_on_every_pass():
    plan = StreamPlan(resident_layers=(), stream_layers=(0, 1, 2), slots=2, slot_bytes=8192, resident_bytes=0,
                      stream_bytes_per_pass=3 * 8192)
    case = vh.Case("c", "c", "tokens", stream="0-2")
    assert by_name(vh.layout_checks(plan, case, _streamed_stats(26, 78), 24))["stats_binds"]
    skipped = by_name(vh.layout_checks(plan, case, _streamed_stats(26, 77), 24))  # one layer ran from the mapping
    assert not skipped["stats_binds"]
    fault = replace(case, options=StreamOptions(fault_read_error_after_bytes=MIB, on_io_error="mmap"))
    assert "stats_binds" not in by_name(vh.layout_checks(plan, fault, _streamed_stats(1, 2), 0))


PASSTHROUGH = {"state": "passthrough", "prefetch_noop_calls": 1, "prefetch_noop_bytes": 491400032,
               "slot_bytes_committed": 0, "resident_bytes": 0, "bytes_read": 0, "streamed_layers": 0}
NOPREFETCH_LOG = "hcstream: enabled (0 tensors, 0 streamed layers, 1 slots, prefetch off)"
EMPTY_PLAN = StreamPlan(resident_layers=(), stream_layers=(), slots=0, slot_bytes=0, resident_bytes=0,
                        stream_bytes_per_pass=0)


def _noprefetch_failures(stats: dict | None, options: StreamOptions = StreamOptions()) -> set[str]:
    case = vh.Case("f", "f", "tokens", options=options)
    checks = vh.option_checks(EMPTY_PLAN, case, outcome(stats=stats, log_text=NOPREFETCH_LOG))
    return {c.name for c in checks if not c.ok}


def test_noprefetch_case_checks_the_passthrough_snapshot():
    assert _noprefetch_failures(PASSTHROUGH) == set()


@pytest.mark.parametrize(("stats", "failing"), [
    (None, {"stats_passthrough", "prefetch_noop_counted", "passthrough_idle"}),  # no snapshot was written
    ({**PASSTHROUGH, "prefetch_noop_calls": 0, "prefetch_noop_bytes": 0}, {"prefetch_noop_counted"}),
    ({**PASSTHROUGH, "prefetch_noop_bytes": 0}, {"prefetch_noop_counted"}),
    ({**PASSTHROUGH, "state": "attached"}, {"stats_passthrough"}),
    ({**PASSTHROUGH, "bytes_read": 4096}, {"passthrough_idle"}),
    ({**PASSTHROUGH, "slot_bytes_committed": 8192}, {"passthrough_idle"}),
])
def test_noprefetch_case_failures(stats, failing):
    assert _noprefetch_failures(stats) == failing


def test_with_prefetch_on_the_noop_must_never_run():
    on = StreamOptions(no_prefetch=False)
    assert _noprefetch_failures(PASSTHROUGH, on) == {"prefetch_noop_counted"}
    assert _noprefetch_failures({**PASSTHROUGH, "prefetch_noop_calls": 0, "prefetch_noop_bytes": 0}, on) == set()


# --------------------------------------------------------------------------------------------------- children


class FakeProc:
    """A child that stays alive until wait() is called, which returns its exit code at once."""

    def __init__(self, code: int = 0) -> None:
        self.code = code

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return self.code


def test_settle_polls_stats_only_until_the_resident_copy_shows(monkeypatch):
    snapshots = iter([None, {"resident_bytes": 0}, {"resident_bytes": 2048}, {"resident_bytes": 4096},
                      {"resident_bytes": 4096}])
    calls: list[int] = []

    def fake_read(path: Path | None) -> dict | None:
        calls.append(1)
        return next(snapshots)

    monkeypatch.setattr(vh, "read_stats", fake_read)
    code, peak, slot_peak = vh._settle(FakeProc(0), 5.0, Path("s.json"), 4096)  # type: ignore[arg-type]
    # Polling stops at the first full snapshot: an open handle would make the extension's final rename fail.
    assert (code, peak, slot_peak, len(calls)) == (0, 4096, 0, 4)


def test_settle_waits_for_both_the_resident_copy_and_the_slots(monkeypatch):
    snapshots = iter([{"resident_bytes": 4096, "slot_bytes_committed": 0},
                      {"resident_bytes": 4096, "slot_bytes_committed": 8192},
                      {"resident_bytes": 4096, "slot_bytes_committed": 16384},
                      {"resident_bytes": 4096, "slot_bytes_committed": 16384}])
    calls: list[int] = []

    def fake_read(path: Path | None) -> dict | None:
        calls.append(1)
        return next(snapshots)

    monkeypatch.setattr(vh, "read_stats", fake_read)
    result = vh._settle(FakeProc(0), 5.0, Path("s.json"), 4096, 16384)  # type: ignore[arg-type]
    assert (*result, len(calls)) == (0, 4096, 16384, 3)


def test_settle_without_a_target_never_reads_stats(monkeypatch):
    monkeypatch.setattr(vh, "read_stats", lambda path: pytest.fail("stats read without a resident target"))
    assert vh._settle(FakeProc(86), 5.0, Path("s.json"), 0) == (86, 0, 0)  # type: ignore[arg-type]
    assert vh._settle(FakeProc(0), 0.0, None, 0) == (None, 0, 0)  # type: ignore[arg-type]


def _python_child(tmp_path: Path, code: str, wait_s: float) -> tuple[vh.Outcome, int]:
    log = tmp_path / "child.log"
    argv = [sys.executable, "-c", f"import os, sys, time; print(os.getpid(), flush=True); {code}"]
    out = vh.run_child(argv, vh.child_env(None), tmp_path, log, None, vh.Settings(), port=None, completions=0,
                       wait_s=wait_s)
    return out, int(out.log_text.split()[0])


def test_run_child_reports_the_child_exit_code(tmp_path):
    out, _ = _python_child(tmp_path, "sys.exit(3)", 30.0)
    assert out.exit_code == 3 and out.served is False and out.error is None
    assert "sampler_error" in out.memory  # a dead memory guard must show in the evidence


def test_run_child_never_leaves_a_child_running(tmp_path):
    psutil = pytest.importorskip("psutil")
    out, pid = _python_child(tmp_path, "time.sleep(120)", 2.0)
    assert out.exit_code is None  # still running at the deadline, so it was stopped, not exited
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_output_dir_is_absolute_because_children_run_in_the_engine_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = vh.datetime(2026, 10, 22, 4, 48, 12, tzinfo=vh.UTC)
    rel = vh.output_dir("runs/pfad-ü-漢", started)
    assert rel.is_absolute() and rel == (tmp_path / "runs" / "pfad-ü-漢").resolve()
    default = vh.output_dir(None, started)
    assert default.is_absolute() and default.name == "20261022T044812Z-validate"
    assert default.parent == (vh.ROOT / "results" / "capacity").resolve()


def test_report_helpers():
    text = "load\nhcstream: enabled (1 tensors)\nother\nhcstream: FATAL: boom\n"
    assert vh.hcstream_lines(text) == ("hcstream: enabled (1 tensors)", "hcstream: FATAL: boom")
    broken = vh.failed_result(vh.Case("x", "x", "tokens"), "RuntimeError: no engine")
    assert not broken.passed and broken.checks[0].detail == "RuntimeError: no engine"
    assert vh.format_table([broken]).splitlines()[-1] == "overall: FAIL"
    assert vh.format_table([]).splitlines()[-1] == "overall: FAIL"


# --------------------------------------------------------------------------------------------------- hardware


@pytest.fixture(scope="module")
def target() -> vh.Target:
    model = vh.default_model()
    missing = vh.missing_requirements(model)
    if missing:
        pytest.skip("; ".join(missing))
    return vh.prepare(model)


@pytest.fixture(scope="module")
def cases(target: vh.Target) -> dict[str, vh.Case]:
    return {c.name: c for c in vh.build_matrix(target.n_layer)}


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("hcstream-native")


@pytest.fixture(scope="module")
def stock_tokens(target: vh.Target, cases: dict[str, vh.Case], out_dir: Path) -> tuple[int, ...]:
    result = vh.run_case(target, replace(cases["a-stock"], completions=1), HW, out_dir, None)
    assert result.passed, failed(result)
    assert len(result.tokens) == HW.tokens
    return result.tokens


@pytest.mark.hardware
def test_two_slot_streaming_matches_stock_tokens(target, cases, out_dir, stock_tokens):
    result = vh.run_case(target, cases["c-stream-k2"], HW, out_dir, stock_tokens)
    assert result.passed, failed(result)
    assert result.token_match is True and result.tokens == stock_tokens
    assert result.stats is not None and result.plan is not None
    assert result.stats["slot_bytes"] == result.plan["slot_bytes"]
    assert result.stats["bytes_read"] >= result.plan["stream_bytes_per_pass"]  # at least one real streamed pass


@pytest.mark.hardware
def test_malformed_config_exits_86_before_serving(target, cases, out_dir):
    result = vh.run_case(target, cases["j-malformed"], HW, out_dir, None)
    assert result.passed, failed(result)
    assert result.exit_code == vh.FATAL_EXIT
    assert any("FATAL" in line and "truncated" in line for line in result.hcstream_log)


@pytest.mark.hardware
def test_read_error_with_abort_exits_86(target, cases, out_dir):
    result = vh.run_case(target, cases["g-fault-abort"], HW, out_dir, None)
    assert result.passed, failed(result)
    assert result.exit_code == vh.FATAL_EXIT
    assert result.stats is not None and result.stats["state"] == "fatal" and result.stats["fault_injected"] > 0
    assert any("win32 error 30 at offset " in line and "offset 0)" not in line for line in result.hcstream_log)


class StatsHolder:
    """A slow monitor of the stats file: holds it open `hold_s` at a time with `gap_s` gaps until stopped.

    Python's open() does not share delete access, so while it holds the file the extension's MoveFileExW fails with
    ERROR_ACCESS_DENIED. One try at the final rename then succeeds about gap/(hold+gap) of the time.
    """

    def __init__(self, path: Path, hold_s: float = 0.6, gap_s: float = 0.05) -> None:
        self.path, self.hold_s, self.gap_s = path, hold_s, gap_s
        self.holds = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self.path.open("rb"):
                    self.holds += 1
                    self._stop.wait(self.hold_s)
            except OSError:
                pass  # not written yet
            self._stop.wait(self.gap_s)

    def __enter__(self) -> StatsHolder:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


@pytest.mark.hardware
@pytest.mark.parametrize(("name", "state"), [("k-teardown", "shutdown"), ("g-fault-abort", "fatal")])
def test_final_snapshot_survives_a_reader_holding_the_stats_file(target, cases, out_dir, name, state):
    # The last snapshot is the evidence: shutdown proves the buffers were released, fatal names the failure. A
    # monitor polling the file (runtime.read_stats, the validator) must not make the extension drop it.
    case = cases[name]
    work = out_dir / f"held-{name}"
    work.mkdir()
    plan = vh.plan_for(target, case)
    assert plan is not None
    cfg, stats = vh.write_config(target, case, plan, work)
    port = None if case.expect == "teardown" else vh.free_loopback_port()
    if port is None:
        argv, wait_s = vh.completion_argv(target.overlay_bin, target.shards[0], HW), HW.request_timeout_s
    else:
        argv, wait_s = vh.server_argv(target.overlay_bin, target.shards[0], port, HW), HW.exit_timeout_s
    with StatsHolder(stats) as holder:
        out = vh.run_child(argv, vh.child_env(cfg), target.overlay_bin, work / f"{name}.log", None, HW, port=port,
                           completions=0 if port is None else 1, wait_s=wait_s)
    final = vh.read_stats(stats)
    assert holder.holds > 0
    assert final is not None and final["state"] == state, f"{final} log={work / f'{name}.log'}"
    assert "was not published" not in out.log_text
    assert not list(work.glob(f"{stats.name}.*.tmp"))  # failed renames clean up their temporary file
    if state == "shutdown":
        assert out.exit_code == 0 and final["slot_bytes_committed"] == 0 and final["resident_bytes"] == 0
    else:
        assert out.exit_code == vh.FATAL_EXIT and final["fault_injected"] > 0


@pytest.mark.hardware
def test_config_without_layers_writes_a_passthrough_snapshot(target, cases, out_dir, stock_tokens):
    result = vh.run_case(target, cases["f-noprefetch"], HW, out_dir, stock_tokens)
    assert result.passed, failed(result)
    assert result.stats is not None and result.stats["state"] == "passthrough"
    assert result.stats["prefetch_noop_calls"] >= 1 and result.stats["prefetch_noop_bytes"] > 0
