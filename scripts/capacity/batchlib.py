"""Batches of concurrent chats (run_big.py --concurrency N): the arithmetic of one batch and the runner that starts N
chats together. Standard library only, so the paper scripts (summarize.py, stats.py) read run.json with the same code.

A batch's numbers come from the monitor clock of each request (sent, end, and the arrival of each streamed text
delta) and its engine timings:
  - A request's decode span is [end - predicted_ms, end]: the engine's own first-token-to-last-token duration, ending
    at the client end. The first visible delta is not the start, because gpt-oss harmony tokens are not streamed, so
    it can come several engine steps after the engine's first token.
  - W, the all-active window, runs from the last decode start to the first end, the only time every request decodes.
  - Tokens inside W (basis token_times, when every request has delta arrival times): each request's deltas that
    arrive in (t0, t1], one token each, or (predicted_n - 1) / deltas each when its delta count is not predicted_n.
    One decode step is W over the median request's tokens inside W. Counting arrivals keeps a stalled mixed step
    before W (a late request's prefill, 10-24 s on the 671B, inside every earlier request's span) out of W's rate.
  - Fallback (basis even_spread: an old row, or a request without delta times): each request's predicted_n - 1
    decode tokens spread evenly over its decode span, times the fraction of that span inside W, and one step is W
    over the tokens per sequence. A stalled step spread that way reads a few percent slow, so a batch of N > 1 with a
    request that missed the first iteration is skewed, and only ok batches are pooled.
  - aggregate decode tok/s = tokens in W / W.
  - A batch whose W covers under 80% of the median request's decode span did not run concurrently (not_concurrent).
  - A request whose decode starts more than half a step after the earliest missed the server's first iteration:
    llama-server takes new tasks only between update_slots() calls, so its prompt ran in a later, mixed batch.
  - SSD reads are counted once over W, per aggregate token and per step (overlapping per-request windows would count
    every byte N times).
A batch of one is an N=1 rep: W is its decode span, its steps are predicted_n - 1.
"""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

T = TypeVar("T")
MIN_COVERAGE = 0.8  # W over the median decode span below this: not_concurrent
MISSED_STEP_FRACTION = 0.5  # decode start later than this many steps after the earliest: missed the first iteration
POLL_S = 0.25  # the runner wakes this often, so a Ctrl+C in the main thread is seen while chats are in flight
SUMMARY_METRICS = ("aggregate_decode_tok_s", "aggregate_output_tok_s", "step_ms", "first_token_skew_s",
                   "ssd_bytes_per_step", "ssd_bytes_per_aggregate_token", "missed_first_iteration", "window_coverage")
STATUSES = ("not_concurrent", "skewed", "incomplete")  # counted by summarize() beside ok
NO_WORK = ("window_tokens", "aggregate_decode_tok_s", "step_ms", "missed_first_iteration", "window_disk_read_bytes",
           "ssd_bytes_per_aggregate_token", "ssd_bytes_per_step", "window_deltas")


@dataclass(frozen=True)
class Span:
    """One request of a batch on one clock (seconds): sent, decode start, end, its engine predicted_n, and the arrival
    time of each streamed text delta when the client recorded them."""

    label: str
    start: float
    first: float
    end: float
    output_tokens: int
    deltas: tuple[float, ...] | None = None

    @property
    def decode_tokens(self) -> int:
        return max(self.output_tokens - 1, 0)

    @property
    def decode_s(self) -> float:
        return self.end - self.first


def tokens_inside(spans: Iterable[Span], t0: float, t1: float) -> float:
    """Decode tokens inside [t0, t1], each request's tokens spread evenly over its decode span."""
    total = 0.0
    for s in spans:
        if s.decode_s > 0:
            total += s.decode_tokens * max(0.0, min(s.end, t1) - max(s.first, t0)) / s.decode_s
    return total


def delta_counts(s: Span, t0: float, t1: float) -> dict[str, Any]:
    """A request's delta arrivals up to t0, in (t0, t1] and after t1, and the decode tokens they stand for in W."""
    deltas = s.deltas or ()
    before, after = sum(1 for d in deltas if d <= t0), sum(1 for d in deltas if d > t1)
    inside = len(deltas) - before - after
    per_delta = 1.0 if len(deltas) == s.output_tokens else (s.decode_tokens / len(deltas) if deltas else 0.0)
    return {"label": s.label, "deltas": len(deltas), "before": before, "inside": inside, "after": after,
            "tokens": inside * per_delta}


def _ratio(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den else None


def _window_work(spans: Sequence[Span], w0: float, w1: float) -> tuple[str, float, float, list[dict[str, Any]] | None]:
    """(basis, decode tokens inside W, decode steps inside W, per-request delta counts or None)."""
    if all(s.deltas for s in spans):
        counts = [delta_counts(s, w0, w1) for s in spans]
        tokens = sum(c["tokens"] for c in counts)
        return "token_times", tokens, statistics.median(c["tokens"] for c in counts), counts
    tokens = tokens_inside(spans, w0, w1)
    return "even_spread", tokens, tokens / len(spans), None


def _status(coverage: float, basis: str, missed: int, n: int) -> str:
    if coverage < MIN_COVERAGE:
        return "not_concurrent"
    return "skewed" if basis == "even_spread" and n > 1 and missed > 0 else "ok"


def batch_metrics(spans: Sequence[Span], reads: Callable[[float, float], int | None] | None = None) -> dict[str, Any]:
    """The numbers of one batch whose every request finished (see the module docstring); reads(t0, t1) gives the
    system-wide disk read bytes between two times on the spans' clock, or None."""
    w0, w1 = max(s.first for s in spans), min(s.end for s in spans)
    wall = max(s.end for s in spans) - min(s.start for s in spans)
    earliest, median_span = min(s.first for s in spans), statistics.median(s.decode_s for s in spans)
    width = max(w1 - w0, 0.0)
    coverage = width / median_span if median_span > 0 else 0.0
    base = {"requests": [s.label for s in spans], "wall_s": wall,
            "aggregate_output_tok_s": _ratio(sum(s.output_tokens for s in spans), wall),
            "first_token_skew_s": w0 - earliest, "median_decode_span_s": median_span,
            "window": {"t0": w0, "t1": w1, "s": width}, "window_coverage": coverage}
    basis, tokens, steps, counts = _window_work(spans, w0, w1) if width > 0 else ("even_spread", 0.0, 0.0, None)
    if tokens <= 0 or steps <= 0:
        return {**base, "status": _status(coverage, basis, 0, len(spans)), "basis": basis,
                **{k: None for k in NO_WORK}}
    step_s = width / steps
    missed = sum(1 for s in spans if s.first - earliest > MISSED_STEP_FRACTION * step_s)
    disk = reads(w0, w1) if reads is not None else None
    return {**base, "status": _status(coverage, basis, missed, len(spans)), "basis": basis, "window_tokens": tokens,
            "aggregate_decode_tok_s": tokens / width, "step_ms": step_s * 1000, "missed_first_iteration": missed,
            "window_disk_read_bytes": disk, "ssd_bytes_per_aggregate_token": _ratio(disk, tokens),
            "ssd_bytes_per_step": _ratio(disk, steps), "window_deltas": counts}


def _stats(values: Iterable[float | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals)}


def summarize(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts by status, and median, min and max of each batch metric over the batches with status ok."""
    ok = [b for b in batches if b.get("status") == "ok"]
    return {"batches": len(batches), "ok": len(ok),
            **{status: sum(1 for b in batches if b.get("status") == status) for status in STATUSES},
            **{m: _stats(b.get(m) for b in ok) for m in SUMMARY_METRICS}}


# ----------------------------------------------------------------------------- run.json readers (paper scripts)


def concurrency_of(run: Mapping[str, Any]) -> int:
    """requested.concurrency of a run.json; rows written before --concurrency existed ran one chat at a time."""
    value = (run.get("requested") or {}).get("concurrency")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 1


def decode_start(start: float, end: float, predicted_ms: float | None, ttft_ms: float | None) -> float | None:
    """When the engine's decode began on the client clock: end - predicted_ms, never before the send; sent + client
    ttft only when the row has no predicted_ms."""
    if predicted_ms is not None:
        return max(start, end - predicted_ms / 1000)
    return start + ttft_ms / 1000 if ttft_ms is not None else None


def batch_of_one(row: Mapping[str, Any]) -> dict[str, Any]:
    """An N=1 rep row of run.json as a batch of one, from its client timestamps and engine predicted_ms; its SSD bytes
    are the rep's decode-only reads (decode_disk_read_bytes_per_token)."""
    label = str(row.get("label"))
    start, end, n = row.get("t_start"), row.get("t_end"), row.get("predicted_n")
    first = None if None in (start, end) else decode_start(float(start), float(end), row.get("predicted_ms"),
                                                            row.get("ttft_ms"))
    if not row.get("ok") or None in (first, n) or row.get("ttft_ms") is None:
        return {"requests": [label], "status": "incomplete", "failed": [label]}
    span = Span(label, float(start), float(first), float(end), int(n))
    per_step = row.get("decode_disk_read_bytes_per_token")
    reads = None if per_step is None else (lambda t0, t1: per_step * span.decode_tokens)
    return batch_metrics([span], reads)


def batches_of(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The measured batches of a run.json: its batch records when it ran N > 1, else one batch of one per measured
    rep (labels rep*), so N=1 and N>1 rows share one arithmetic."""
    chats = (run.get("measured") or {}).get("chats") or {}
    if concurrency_of(run) > 1:
        return [dict(b) for b in chats.get("batches") or []]
    return [batch_of_one(r) for r in chats.get("reps") or [] if str(r.get("label", "")).startswith("rep")]


# ----------------------------------------------------------------------------- the runner


def run_together(tasks: Sequence[Callable[[Callable[[], None]], T]], failed: Callable[[int, Exception], T],
                 barrier_timeout_s: float, poll_s: float = POLL_S) -> list[T]:
    """Runs each task on its own thread and returns their results in task order. A task gets start(): it does its
    setup, then calls start(), which returns to every task at once (a threading.Barrier with a timeout). A task that
    raises is that task's failed(index, exc); it also breaks the barrier, so the others fail at once (with
    BrokenBarrierError) instead of waiting out the timeout. The workers are daemon threads joined every poll_s, so a
    Ctrl+C reaches the caller while chats are still in flight, and neither waits for them nor holds the process open;
    the barrier is then aborted, so a worker that has not passed start() never sends its chat."""
    barrier = threading.Barrier(len(tasks), timeout=barrier_timeout_s)
    results: list[Any] = [None] * len(tasks)

    def start() -> None:
        barrier.wait()

    def guarded(index: int, task: Callable[[Callable[[], None]], T]) -> T:
        try:
            return task(start)
        except Exception as exc:  # noqa: BLE001 - recorded as that request's failure
            barrier.abort()
            return failed(index, exc)

    def worker(index: int, task: Callable[[Callable[[], None]], T]) -> None:
        results[index] = guarded(index, task)

    threads = [threading.Thread(target=worker, args=(i, task), name=f"batch-{i}", daemon=True)
               for i, task in enumerate(tasks)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(timeout=poll_s)
    except BaseException:
        barrier.abort()
        raise
    return cast(list[T], results)
