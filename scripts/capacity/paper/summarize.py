"""Summarize matrix INDEX.jsonl files into markdown tables (measured reps only, warmups excluded).

The first table has the per-request reps of N=1 runs (requested.concurrency, missing = 1); an N>1 run's requests share
every decode step, so they stay out of it. The second has every run's batches (batchlib.py): an N>1 run's measured
batches, and for an N=1 run each rep as a batch of one, from the same client-clock arithmetic.

Usage: python -I summarize.py OUT.md INDEX.jsonl [INDEX.jsonl ...]
batchlib.py must be at ../batchlib.py (the repo layout) or beside this script (a standalone copy).
Run dirs are read from $HC_PAPER_REPO/results/capacity (default: the G14 live-host checkout).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

here = Path(__file__).resolve()
# scripts/capacity: batchlib (standard library only); a standalone copy keeps batchlib.py beside it
sys.path[:0] = [str(here.parents[1]), str(here.parent)]

from batchlib import batch_of_one, batches_of, concurrency_of  # noqa: E402, F401 (batch_of_one re-exported)
from batchlib import summarize as summarize_batches  # noqa: E402

REPO = Path(os.environ.get("HC_PAPER_REPO", r"C:\Users\operator\.hypercompute\worktrees\live-host"))
RESULTS = REPO / "results" / "capacity"


def rpc_bytes(net: object) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(net, dict):
        for p in net.get("peers", []):
            rpc = p.get("rpc") or {}
            out[p["node_id"]] = int(rpc.get("tcp_to_quic_bytes", 0)) + int(rpc.get("quic_to_tcp_bytes", 0))
    return out


def rtts(net: object) -> list[int]:
    if not isinstance(net, dict):
        return []
    return [p["path"]["rtt_ms"] for p in net.get("peers", []) if p.get("connected") and isinstance(p.get("path"), dict)
            and p["path"].get("rtt_ms") is not None]


def row(rec: dict) -> dict:
    out = {"label": rec["label"], "model": rec["model"], "peers": len(rec["nodes"]), "exit": rec["exit"],
           "wall_s": rec["wall_s"], "dir": rec.get("result_dir")}
    run_path = RESULTS / str(rec.get("result_dir")) / "run.json"
    if not rec.get("result_dir") or not run_path.is_file():
        out["status"] = "no result"
        return out
    run = json.loads(run_path.read_text(encoding="utf-8"))
    out["status"] = run.get("status")
    m = run.get("measured", {})
    out["load_s"] = (m.get("load") or {}).get("seconds")
    out["concurrency"] = concurrency_of(run)
    out["batch"] = summarize_batches(batches_of(run))
    reps = [r for r in (m.get("chats") or {}).get("reps", []) if r.get("ok") and r["label"].startswith("rep")
            and out["concurrency"] == 1]
    dec = [r["decode_tok_s"] for r in reps if r.get("decode_tok_s")]
    pre = [r["prefill_tok_s"] for r in reps if r.get("prefill_tok_s")]
    disk = [r["decode_disk_read_bytes_per_token"] for r in reps if r.get("decode_disk_read_bytes_per_token")]
    out["reps"] = len(dec)
    out["decode_tok_s"] = [round(x, 3) for x in dec]
    out["decode_med"] = round(statistics.median(dec), 3) if dec else None
    out["prefill_med"] = round(statistics.median(pre), 3) if pre else None
    out["disk_mib_per_tok"] = round(statistics.median(disk) / 2**20, 1) if disk else None
    before, after = rpc_bytes(rec.get("network_before")), rpc_bytes(rec.get("network_after"))
    moved = sum(max(0, after.get(k, 0) - before.get(k, 0)) for k in after)
    out["rpc_mib_total"] = round(moved / 2**20, 1)
    out["rtt_ms"] = rtts(rec.get("network_before"))
    out["error"] = (run.get("error") or {}).get("message") if isinstance(run.get("error"), dict) else run.get("error")
    return out


def _median(stats: dict, scale: float = 1.0, digits: int = 3) -> float | None:
    return round(stats["median"] * scale, digits) if stats.get("n") else None


def batch_table(rows: list[dict]) -> str:
    """Batches (ok ones pooled): aggregate decode tok/s over the all-active window, the step time, first-token skew,
    SSD bytes counted once per window, and the requests that missed the first iteration and the window coverage."""
    head = ("| label | model | N | status | batches | aggregate decode tok/s (median) | step ms (median) | "
            "first-token skew s (median) | SSD MiB/step (median) | SSD MiB/aggregate token (median) | "
            "missed first iteration (median) | window coverage (median) |")
    lines = ["Batches (N=1: each rep as a batch of one)", "", head, "|" + "---|" * 12]
    for r in rows:
        b = r.get("batch")
        if not b:
            continue
        count = (f"{b['batches']} ({b['ok']} ok, {b['not_concurrent']} not concurrent, {b.get('skewed', 0)} "
                 "skewed)")
        lines.append(f"| {r['label']} | {r['model']} | {r['concurrency']} | {r.get('status')} | {count} | "
                     f"{_median(b['aggregate_decode_tok_s'])} | {_median(b['step_ms'], digits=1)} | "
                     f"{_median(b['first_token_skew_s'])} | {_median(b['ssd_bytes_per_step'], 1 / 2**20, 1)} | "
                     f"{_median(b['ssd_bytes_per_aggregate_token'], 1 / 2**20, 1)} | "
                     f"{_median(b.get('missed_first_iteration') or {}, digits=1)} | "
                     f"{_median(b.get('window_coverage') or {})} |")
    return "\n".join(lines)


def main() -> int:
    rows = []
    for idx in sys.argv[2:]:
        for line in Path(idx).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(row(json.loads(line)))
    head = ("| label | model | peers | status | load s | decode tok/s (reps) | decode median | prefill median | "
            "SSD MiB/token | RTT ms | wall s |")
    lines = [head, "|" + "---|" * 11]
    for r in rows:
        lines.append(f"| {r['label']} | {r['model']} | {r['peers']} | {r.get('status')} | {r.get('load_s')} | "
                     f"{r.get('decode_tok_s')} | {r.get('decode_med')} | {r.get('prefill_med')} | "
                     f"{r.get('disk_mib_per_tok')} | {r.get('rtt_ms')} | {r['wall_s']} |")
    errs = [f"- {r['label']}: {r['error']}" for r in rows if r.get("error")]
    text = ("\n".join(lines) + "\n\n" + batch_table(rows) + ("\n\nErrors\n" + "\n".join(errs) if errs else "")
            + "\n")
    Path(sys.argv[1]).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
