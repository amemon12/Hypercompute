"""scripts/capacity/batchlib.py: the all-active window of a batch of concurrent chats, the tokens inside it, and the
runner that starts N chats together and still returns on Ctrl+C. Synthetic spans, no gateway."""

from __future__ import annotations

import _thread
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "capacity"))

import batchlib as bl  # noqa: E402


def _skewed(s: float, t: float = 10.0, n: int = 101) -> list[bl.Span]:
    """Seven requests decode from 0, one starts s later; each decodes n - 1 tokens over t seconds at a steady rate."""
    return [*(bl.Span(f"r{j}", 0.0, 0.0, t, n) for j in range(7)), bl.Span("r7", 0.0, s, s + t, n)]


def _timed(s: float, t: float = 8.0, n: int = 65) -> list[bl.Span]:
    """_skewed with a text delta at every engine step (binary-exact times): seven decode from 0, one from s."""
    def span(label: str, first: float) -> bl.Span:
        return bl.Span(label, 0.0, first, first + t, n, tuple(first + k * t / (n - 1) for k in range(n)))
    return [*(span(f"r{j}", 0.0) for j in range(7)), span("r7", s)]


def test_skewed_batch_window_does_not_inflate() -> None:
    """A late first token (one 671B prefill against a short decode) must not count the tokens the early requests made
    before it: only tokens inside W = [last first token, first end] are divided by W."""
    s, t = 1.1, 10.0
    spans = _skewed(s, t)
    out = bl.batch_metrics(spans)
    assert out["window"] == {"t0": s, "t1": t, "s": pytest.approx(t - s)}
    assert out["window_tokens"] == pytest.approx(8 * 100 * (t - s) / t)  # 712, not the 800 decoded in all
    assert out["aggregate_decode_tok_s"] == pytest.approx(80.0)  # 8 requests at 10 tok/s each, all active
    # the design's window (last first token to last end) holds fewer tokens than sum(n - 1): 800 / 10 s reads
    # 8 / (8 - 7 s / t) times what was decoded in it
    naive = 800 / t
    inside = bl.tokens_inside(spans, s, s + t) / t
    assert naive / inside == pytest.approx(8 / (8 - 7 * s / t))
    assert out["first_token_skew_s"] == pytest.approx(s) and out["window_coverage"] == pytest.approx((t - s) / t)
    # one step is W over the tokens per sequence inside it (0.1 s): the late request missed the first iteration, and
    # without delta times the even spread cannot tell a stalled step from steady decode, so the batch is not pooled
    assert out["step_ms"] == pytest.approx(100.0) and out["missed_first_iteration"] == 1
    assert out["basis"] == "even_spread" and out["status"] == "skewed" and out["window_deltas"] is None


def test_delta_times_count_the_tokens_that_arrive_inside_the_window() -> None:
    timed = bl.batch_metrics(_timed(1.0))
    assert timed["basis"] == "token_times" and timed["status"] == "ok" and timed["missed_first_iteration"] == 1
    assert timed["window_tokens"] == 8 * 56 and timed["aggregate_decode_tok_s"] == pytest.approx(64.0)
    assert timed["step_ms"] == pytest.approx(125.0)
    late = timed["window_deltas"][-1]
    assert late == {"label": "r7", "deltas": 65, "before": 1, "inside": 56, "after": 8, "tokens": 56.0}
    # fewer deltas than predicted_n (harmony tokens not streamed): each stands for (predicted_n - 1) / deltas tokens
    span = bl.Span("a", 0.0, 0.0, 4.0, 9, (0.0, 1.0, 2.0, 3.0))
    assert bl.delta_counts(span, 0.5, 4.0) == {"label": "a", "deltas": 4, "before": 1, "inside": 3, "after": 0,
                                               "tokens": 6.0}
    mixed = bl.batch_metrics([*_timed(1.0)[:7], bl.Span("r7", 0.0, 1.0, 9.0, 65)])  # one request without times
    assert mixed["basis"] == "even_spread"


def _stalled(tau: float = 0.125, p1: float = 2.0, n: int = 129, timed: bool = True) -> list[bl.Span]:
    """N=8, one late: the early seven decode three steps of tau, then the late request's prefill runs as one stalled
    mixed step of p1 >> tau (a token for each early request and the late one's first token at its end), then every
    request decodes at a steady tau. The late request missed the first iteration."""
    join = 3 * tau + p1
    early = (0.0, tau, 2 * tau, 3 * tau, join, *(join + k * tau for k in range(1, n - 4)))
    late = tuple(join + k * tau for k in range(n))
    spans = [*(bl.Span(f"r{j}", 0.0, 0.0, early[-1], n, early) for j in range(7)),
             bl.Span("r7", 0.2, join, late[-1], n, late)]
    return spans if timed else [bl.Span(x.label, x.start, x.first, x.end, x.output_tokens) for x in spans]


def test_a_stalled_mixed_step_before_the_window_does_not_slow_the_rate() -> None:
    tau = 0.125
    out = bl.batch_metrics(_stalled(tau))
    assert out["basis"] == "token_times" and out["status"] == "ok" and out["missed_first_iteration"] == 1
    assert out["aggregate_decode_tok_s"] == pytest.approx(8 / tau) and out["step_ms"] == pytest.approx(tau * 1000)
    # the even spread counts part of the stalled step inside W: it reads slow, so it stays out of the ok pool
    spread = bl.batch_metrics(_stalled(tau, timed=False))
    assert spread["basis"] == "even_spread" and spread["status"] == "skewed"
    assert spread["aggregate_decode_tok_s"] < 0.95 * 8 / tau  # 58 tok/s, not 64
    pooled = bl.summarize([out, spread])
    assert (pooled["ok"], pooled["skewed"]) == (1, 1)
    assert pooled["aggregate_decode_tok_s"]["median"] == pytest.approx(8 / tau)
    assert pooled["missed_first_iteration"]["median"] == 1 and pooled["window_coverage"]["n"] == 1


def test_batch_of_one_anchors_decode_to_the_engine_not_the_first_visible_delta() -> None:
    """gpt-oss harmony: the first visible delta comes several engine steps after the engine's first token. The decode
    span is [end - predicted_ms, end], so a batch of one reads the engine's own decode rate, within the end lag."""
    lag, predicted_ms, n = 0.05, 4000.0, 41  # engine first token at 0.95 s, engine end 4.95 s, client end 5.0 s
    row = {"label": "rep0", "ok": True, "t_start": 0.0, "t_end": 4.95 + lag, "ttft_ms": 1400.0, "predicted_n": n,
           "predicted_ms": predicted_ms, "decode_disk_read_bytes_per_token": None}
    out = bl.batch_of_one(row)
    assert out["window"]["t0"] == pytest.approx(1.0) and out["status"] == "ok"
    engine = (n - 1) / (predicted_ms / 1000)
    assert out["aggregate_decode_tok_s"] == pytest.approx(engine)
    assert (n - 1) / (predicted_ms / 1000 + lag) <= out["aggregate_decode_tok_s"] <= engine
    visible = bl.batch_of_one({**row, "predicted_ms": None})  # no engine timings: sent + client ttft
    assert visible["window"]["t0"] == pytest.approx(1.4) and visible["aggregate_decode_tok_s"] > 1.1 * engine
    clamped = bl.batch_of_one({**row, "predicted_ms": 9000.0})  # never before the send
    assert clamped["window"]["t0"] == 0.0


def test_a_window_under_80_percent_of_the_decode_span_is_not_concurrent() -> None:
    out = bl.batch_metrics(_skewed(3.0))
    assert out["window_coverage"] == pytest.approx(0.7) and out["status"] == "not_concurrent"
    serial = bl.batch_metrics([bl.Span("a", 0.0, 0.1, 1.0, 10), bl.Span("b", 1.0, 1.1, 2.0, 10)])
    assert serial["status"] == "not_concurrent" and serial["window"]["s"] == 0.0
    assert serial["aggregate_decode_tok_s"] is None and serial["window_tokens"] is None


def test_ssd_bytes_are_counted_once_per_window() -> None:
    seen: list[tuple[float, float]] = []

    def reads(t0: float, t1: float) -> int:
        seen.append((t0, t1))
        return 712_000

    out = bl.batch_metrics(_skewed(1.1), reads)
    assert seen == [(1.1, 10.0)] and out["window_disk_read_bytes"] == 712_000
    assert out["ssd_bytes_per_aggregate_token"] == pytest.approx(1000.0)
    assert out["ssd_bytes_per_step"] == pytest.approx(8000.0)  # 89 steps inside W
    assert bl.batch_metrics(_skewed(1.1))["ssd_bytes_per_step"] is None  # no disk counter


def test_summary_pools_only_concurrent_batches() -> None:
    good, late, skewed = bl.batch_metrics(_timed(0.5)), bl.batch_metrics(_skewed(3.0)), bl.batch_metrics(_skewed(0.5))
    out = bl.summarize([good, good, late, skewed, {"status": "incomplete"}])
    assert (out["batches"], out["ok"], out["not_concurrent"], out["skewed"], out["incomplete"]) == (5, 2, 1, 1, 1)
    assert out["aggregate_decode_tok_s"]["n"] == 2 and out["aggregate_decode_tok_s"]["median"] == pytest.approx(64.0)
    assert bl.summarize([])["aggregate_decode_tok_s"] == {"n": 0}


def test_run_together_releases_every_task_at_once_and_keeps_order() -> None:
    started: list[float] = []
    lock = threading.Lock()

    def task(j: int):
        def run(start):
            time.sleep(0.02 * j)  # setup takes longer for later tasks
            start()
            with lock:
                started.append(time.perf_counter())
            return j
        return run

    assert bl.run_together([task(j) for j in range(4)], lambda j, exc: -1, barrier_timeout_s=5.0) == [0, 1, 2, 3]
    assert max(started) - min(started) < 0.05


def test_a_task_failing_before_the_barrier_fails_the_others_at_once() -> None:
    def bad(start):
        raise ValueError("no client")

    def good(start):
        start()
        return "ok"

    t0 = time.perf_counter()
    out = bl.run_together([good, bad, good], lambda j, exc: type(exc).__name__, barrier_timeout_s=30.0)
    assert out == ["BrokenBarrierError", "ValueError", "BrokenBarrierError"]
    assert time.perf_counter() - t0 < 5.0  # not the 30 s barrier timeout


def test_an_interrupt_aborts_the_barrier_and_leaves_no_thread_holding_the_process() -> None:
    """Ctrl+C mid-batch: the KeyboardInterrupt reaches the caller at once, a task still at start() gets
    BrokenBarrierError instead of sending, and every worker is a daemon, so none keeps the process alive."""
    sent: list[int] = []
    outcomes: list[str] = []
    lock = threading.Lock()

    def task(setup_s: float, j: int):
        def run(start):
            time.sleep(setup_s)
            start()
            with lock:
                sent.append(j)  # the send body
            return "sent"
        return run

    def failed(j: int, exc: Exception) -> str:
        with lock:
            outcomes.append(type(exc).__name__)
        return type(exc).__name__

    timer = threading.Timer(0.1, _thread.interrupt_main)
    timer.start()
    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        bl.run_together([task(0.0, 0), task(0.0, 1), task(0.5, 2)], failed, barrier_timeout_s=30.0, poll_s=0.02)
    assert time.perf_counter() - t0 < 0.4  # not the slow task's setup, not the barrier timeout
    assert [t for t in threading.enumerate() if t.name.startswith("batch") and t.is_alive() and not t.daemon] == []
    deadline = time.perf_counter() + 5.0
    while len(outcomes) < 3 and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert outcomes == ["BrokenBarrierError"] * 3 and sent == []


def test_concurrency_of_reads_missing_as_one() -> None:
    assert bl.concurrency_of({}) == 1 and bl.concurrency_of({"requested": {"chat_settings": {}}}) == 1
    assert bl.concurrency_of({"requested": {"concurrency": 8}}) == 8
