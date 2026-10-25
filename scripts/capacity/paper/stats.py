"""Pool measured reps per condition and bootstrap the ratio of means between two conditions.

Usage: python -I stats.py [--batches] GROUP_A_PREFIX GROUP_B_PREFIX INDEX.jsonl [INDEX.jsonl ...]
                          [--only-labels L1,L2,...]
A label belongs to a group when it starts with the prefix (e.g. 235b-3mac vs 235b-local). Only runs with status ok.
Prints per-group n, mean, median for decode and prefill tok/s and SSD MiB/token, and a 95% percentile bootstrap
interval (10,000 resamples, fixed seed) for mean(B)/mean(A) of each metric, resampling reps within each group.
Reps come from N=1 runs only (requested.concurrency, missing = 1): the requests of an N>1 batch share every decode
step, so they are not independent reps.
--batches: resample batches instead (batchlib.py): each N>1 run's batches with status ok, and each N=1 rep as a batch
of one; metrics are aggregate decode tok/s over the all-active window, the step time and SSD MiB per step, and each
group's missed-first-iteration counts and window coverage are reported beside them (no ratio). A group must hold one
concurrency, so name labels so no prefix catches another (671b-n8-gxA, not 671b-gxA-n8).
batchlib.py must be at ../batchlib.py (the repo layout) or beside this script (a standalone copy).
Run dirs are read from $HC_PAPER_REPO/results/capacity (default: the G14 live-host checkout).
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve()
# scripts/capacity: batchlib (standard library only); a standalone copy keeps batchlib.py beside it
sys.path[:0] = [str(here.parents[1]), str(here.parent)]

from batchlib import batches_of, concurrency_of  # noqa: E402

REPO = Path(os.environ.get("HC_PAPER_REPO", r"C:\Users\operator\.hypercompute\worktrees\live-host"))
RESULTS = REPO / "results" / "capacity"
METRICS = (("decode_tok_s", 1.0), ("prefill_tok_s", 1.0), ("decode_disk_read_bytes_per_token", 1 / 2**20))
BATCH_METRICS = (("aggregate_decode_tok_s", 1.0), ("step_ms", 1.0), ("ssd_bytes_per_step", 1 / 2**20))
BATCH_REPORTS = ("missed_first_iteration", "window_coverage")  # per group, no bootstrap: a count can have mean 0


def ok_runs(prefix: str, recs: list[dict], only: set[str] | None) -> list[dict]:
    """The run.json of every record in the group whose run ended ok."""
    runs = []
    for rec in recs:
        if not rec["label"].startswith(prefix) or (only and rec["label"] not in only) or not rec.get("result_dir"):
            continue
        path = RESULTS / rec["result_dir"] / "run.json"
        if not path.is_file():
            print(f"warning: {rec['label']}: no {path} (set HC_PAPER_REPO?)", file=sys.stderr)
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") == "ok":
            runs.append(run)
    return runs


def reps_for(prefix: str, recs: list[dict], only: set[str] | None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {m: [] for m, _ in METRICS}
    for run in ok_runs(prefix, recs, only):
        if concurrency_of(run) != 1:
            continue
        for rep in run["measured"]["chats"]["reps"]:
            if rep.get("ok") and rep["label"].startswith("rep"):
                for metric, scale in METRICS:
                    if rep.get(metric) is not None:
                        out[metric].append(rep[metric] * scale)
    return out


def batches_for(prefix: str, recs: list[dict], only: set[str] | None) -> dict[str, list[float]]:
    """One value per batch with status ok; ValueError when the group mixes concurrencies."""
    out: dict[str, list[float]] = {m: [] for m in (*(m for m, _ in BATCH_METRICS), *BATCH_REPORTS)}
    runs = ok_runs(prefix, recs, only)
    levels = sorted({concurrency_of(run) for run in runs})
    if len(levels) > 1:
        raise ValueError(f"group {prefix!r} mixes concurrency {levels}: name labels so no prefix catches another "
                         "(671b-n8-gxA, not 671b-gxA-n8), or pass --only-labels")
    for run in runs:
        for batch in batches_of(run):
            if batch.get("status") == "ok":
                for metric, scale in (*BATCH_METRICS, *((m, 1.0) for m in BATCH_REPORTS)):
                    if batch.get(metric) is not None:
                        out[metric].append(batch[metric] * scale)
    return out


def report(metric: str, pa: str, a: list[float], pb: str, b: list[float]) -> str:
    """A metric's n, median and max per group, without a ratio."""
    def one(prefix: str, vals: list[float]) -> str:
        if not vals:
            return f"{prefix} n=0"
        return f"{prefix} n={len(vals)} median {statistics.median(vals):.3f} max {max(vals):.3f}"
    return f"{metric}: {one(pa, a)} | {one(pb, b)}"


def boot_ratio(a: list[float], b: list[float], n: int = 10_000) -> tuple[float, float]:
    rng = random.Random(7)  # noqa: S311 - bootstrap resampling, not security
    ratios = sorted(statistics.fmean(rng.choices(b, k=len(b))) / statistics.fmean(rng.choices(a, k=len(a)))
                    for _ in range(n))
    return ratios[int(0.025 * n)], ratios[int(0.975 * n) - 1]


def main() -> int:
    args = sys.argv[1:]
    only, batches = None, "--batches" in args
    if batches:
        args.remove("--batches")
    if "--only-labels" in args:
        i = args.index("--only-labels")
        only = set(args[i + 1].split(","))
        del args[i:i + 2]
    pa, pb, files = args[0], args[1], args[2:]
    recs = [json.loads(line) for f in files for line in Path(f).read_text(encoding="utf-8").splitlines() if line]
    collect, metrics = (batches_for, BATCH_METRICS) if batches else (reps_for, METRICS)
    try:
        ga, gb = collect(pa, recs, only), collect(pb, recs, only)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for metric, _ in metrics:
        a, b = ga[metric], gb[metric]
        if len(a) < 2 or len(b) < 2:
            print(f"{metric}: not enough {'batches' if batches else 'reps'} ({len(a)}, {len(b)})")
            continue
        lo, hi = boot_ratio(a, b)
        print(f"{metric}: {pa} n={len(a)} mean {statistics.fmean(a):.3f} median {statistics.median(a):.3f} | "
              f"{pb} n={len(b)} mean {statistics.fmean(b):.3f} median {statistics.median(b):.3f} | "
              f"ratio {statistics.fmean(b) / statistics.fmean(a):.3f} [95% CI {lo:.3f}, {hi:.3f}]")
    for metric in BATCH_REPORTS if batches else ():
        print(report(metric, pa, ga[metric], pb, gb[metric]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
