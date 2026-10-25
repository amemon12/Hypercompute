"""The arithmetic and rendering behind `hcbench.py report`: run-index rows, pooled-vs-alone pairs, bench/RESULTS.md.

Per-row values follow scripts/capacity/paper/summarize.py `row()`. Pairs follow scripts/capacity/paper/stats.py: group
by label prefix, status ok, reps from N = 1 runs or batches from batchlib, ratio of means B/A with a percentile
bootstrap (random.Random(7), 10,000 resamples). First token follows the scratchpad ttft_stats.py (reps) and
conc_ttft.py (batch means) that docs/pitch/NUMBERS.md names. Neither summarize.py nor stats.py is imported: both read
HC_PAPER_REPO when imported. Tests prove the numbers match. Standard library only, plus scripts/capacity/batchlib.py.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "capacity"))  # batchlib.py, standard library only

from batchlib import batches_of, concurrency_of  # noqa: E402
from batchlib import summarize as summarize_batches  # noqa: E402

DEFAULT_GLOB = "results/capacity/paper-2026-10-2*/index_*.jsonl"
ARCHIVED = ("_refused", "_aborted", "_failed")
SECRETISH = re.compile(r"[A-Za-z0-9_-]{40,}")  # matrix.py:33
MIB = 2**20
RESAMPLES = 10_000
ERROR_CHARS = 200
RECORD_KEYS = ("label", "model", "nodes", "exit", "wall_s")  # what summarize.row reads without .get
# key: (run.json field, scale, label). In batches mode first_token and answer are each ok batch's mean over its chats.
REP_METRICS = {"decode": ("decode_tok_s", 1.0, "Decode tok/s"),
               "prefill": ("prefill_tok_s", 1.0, "Prefill tok/s"),
               "ssd": ("decode_disk_read_bytes_per_token", 1 / MIB, "SSD MiB per token"),
               "first_token": ("ttft_ms", 1 / 1000, "First token, s"),
               "answer": ("total_ms", 1 / 1000, "Full answer, s")}
BATCH_METRICS = {"aggregate_decode": ("aggregate_decode_tok_s", 1.0, "Aggregate decode tok/s"),
                 "step": ("step_ms", 1.0, "Step, ms"),
                 "ssd_step": ("ssd_bytes_per_step", 1 / MIB, "SSD MiB per step"),
                 "first_token": ("ttft_ms", 1 / 1000, "First token per chat, s"),
                 "answer": ("total_ms", 1 / 1000, "Full answer per chat, s")}
DECODE_KEY = {"reps": "decode", "batches": "aggregate_decode"}
SECONDS_KEYS = ("first_token", "answer")  # printed with 2 decimals, as NUMBERS.md prints them
MIB_KEYS = ("ssd", "ssd_step")  # printed with mib_digits: 1 decimal, 3 under 1 MiB


class ReportError(ValueError):
    """A problem with the inputs that stops the report (bad index line, bad pairs.json, unreadable run.json)."""


@dataclass(frozen=True)
class IndexFile:
    path: Path
    name: str  # "<paper dir>/<index file>", as NUMBERS.md and pairs.json name it
    records: tuple[dict[str, Any], ...]

    @property
    def archived(self) -> bool:
        return any(tag in self.path.name for tag in ARCHIVED)


@dataclass
class RunStore:
    """run.json files under one results root, each read at most once."""

    root: Path
    cache: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    def get(self, rec: Mapping[str, Any]) -> dict[str, Any] | None:
        result_dir = rec.get("result_dir")
        if not result_dir:
            return None
        key = str(result_dir)
        if key not in self.cache:
            path = self.root / key / "run.json"
            try:
                self.cache[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            except (OSError, ValueError) as exc:
                raise ReportError(f"cannot read {path}: {exc}") from exc
        return self.cache[key]



def results_root(arg: str | None, repo: Path = REPO) -> Path:
    """--results-root, else $HC_PAPER_REPO/results/capacity, else <repo>/results/capacity (summarize.py:26 to 27)."""
    if arg:
        return Path(arg)
    env = os.environ.get("HC_PAPER_REPO")
    return (Path(env) if env else repo) / "results" / "capacity"


# ----------------------------------------------------------------------------- reading the indexes


def parse_index(path: Path, warn: list[str]) -> tuple[dict[str, Any], ...]:
    """Every record of one index. The last non-blank line may be cut off by a live append: skipped with a warning."""
    lines = path.read_text(encoding="utf-8").splitlines()
    numbered = [(i + 1, line) for i, line in enumerate(lines) if line.strip()]
    records = []
    for pos, (lineno, line) in enumerate(numbered):
        try:
            rec = json.loads(line)
        except ValueError as exc:
            if pos == len(numbered) - 1:
                warn.append(f"warning: {path}:{lineno}: last line is not valid JSON (a live append?); skipped")
                continue
            raise ReportError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
        missing = [k for k in RECORD_KEYS if not isinstance(rec, dict) or k not in rec]
        if missing:
            raise ReportError(f"{path}:{lineno}: record lacks {', '.join(missing)}")
        records.append(rec)
    return tuple(records)


def load_indexes(pattern: str, repo: Path, warn: list[str]) -> list[IndexFile]:
    """Index files matching the glob (relative to the repo root), in sorted path order."""
    paths = sorted(Path(p) for p in glob.glob(str(repo / pattern)))
    return [IndexFile(p, f"{p.parent.name}/{p.name}", parse_index(p, warn)) for p in paths]


# ----------------------------------------------------------------------------- per-row values (summarize.row)


def rtts(net: object) -> list[int]:
    if not isinstance(net, dict):
        return []
    return [p["path"]["rtt_ms"] for p in net.get("peers", []) if p.get("connected") and isinstance(p.get("path"), dict)
            and p["path"].get("rtt_ms") is not None]


def rpc_bytes(net: object) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(net, dict):
        for p in net.get("peers", []):
            rpc = p.get("rpc") or {}
            out[p["node_id"]] = int(rpc.get("tcp_to_quic_bytes", 0)) + int(rpc.get("quic_to_tcp_bytes", 0))
    return out


def _median(values: Sequence[float], digits: int, scale: float = 1.0) -> float | None:
    return round(statistics.median(values) * scale, digits) if values else None


def row_values(rec: Mapping[str, Any], store: RunStore) -> dict[str, Any]:
    """summarize.row(rec), plus seed_s (measured.seed.seconds), ttft_med_s (median first token of the same reps) and
    disk_mib_fine (the SSD median at 3 decimals, for values that summarize.row's 1 decimal shows as 0.0)."""
    out: dict[str, Any] = {"label": rec["label"], "model": rec["model"], "peers": len(rec["nodes"]),
                           "exit": rec["exit"], "wall_s": rec["wall_s"], "dir": rec.get("result_dir")}
    run = store.get(rec)
    if run is None:
        return {**out, "status": "no result"}
    m = run.get("measured", {})
    conc = concurrency_of(run)
    reps = [r for r in (m.get("chats") or {}).get("reps", []) if r.get("ok") and r["label"].startswith("rep")
            and conc == 1]
    dec = [r["decode_tok_s"] for r in reps if r.get("decode_tok_s")]
    pre = [r["prefill_tok_s"] for r in reps if r.get("prefill_tok_s")]
    disk = [r["decode_disk_read_bytes_per_token"] for r in reps if r.get("decode_disk_read_bytes_per_token")]
    ttft = [r["ttft_ms"] for r in reps if r.get("ttft_ms")]
    before, after = rpc_bytes(rec.get("network_before")), rpc_bytes(rec.get("network_after"))
    moved = sum(max(0, after.get(k, 0) - before.get(k, 0)) for k in after)
    err = run.get("error")
    seed = m.get("seed")
    return {**out, "status": run.get("status"), "load_s": (m.get("load") or {}).get("seconds"), "concurrency": conc,
            "batch": summarize_batches(batches_of(run)), "reps": len(dec),
            "decode_tok_s": [round(x, 3) for x in dec], "decode_med": _median(dec, 3),
            "prefill_med": _median(pre, 3), "disk_mib_per_tok": _median(disk, 1, 1 / MIB),
            "rpc_mib_total": round(moved / MIB, 1), "rtt_ms": rtts(rec.get("network_before")),
            "error": err.get("message") if isinstance(err, dict) else err,
            "seed_s": seed.get("seconds") if isinstance(seed, dict) else None, "ttft_med_s": _median(ttft, 2, 1e-3),
            "disk_mib_fine": _median(disk, 3, 1 / MIB)}


# ----------------------------------------------------------------------------- pairs (stats.py)


@dataclass(frozen=True)
class Member:
    index: str
    position: int
    rec: dict[str, Any]
    run: dict[str, Any]


def group_members(prefix: str, files: Sequence[IndexFile], only: set[str] | None, store: RunStore,
                  warn: list[str]) -> list[Member]:
    """stats.ok_runs: label prefix, optional label list, a run.json, status ok (stats.py:39 to 52)."""
    out = []
    for f in files:
        for pos, rec in enumerate(f.records):
            if not rec["label"].startswith(prefix) or (only and rec["label"] not in only) or not rec.get("result_dir"):
                continue
            run = store.get(rec)
            if run is None:
                warn.append(f"warning: {rec['label']}: no run.json under {store.root}")
                continue
            if run.get("status") == "ok":
                out.append(Member(f.name, pos, rec, run))
    return out


def rep_values(members: Iterable[Member]) -> tuple[dict[str, list[float]], list[Member]]:
    """stats.reps_for: ok reps labelled rep* of N = 1 runs (stats.py:55 to 65), and first token as ttft_stats.py."""
    out: dict[str, list[float]] = {k: [] for k in REP_METRICS}
    used = []
    for mem in members:
        if concurrency_of(mem.run) != 1:
            continue
        used.append(mem)
        for rep in mem.run["measured"]["chats"]["reps"]:
            if rep.get("ok") and rep["label"].startswith("rep"):
                for key, (metric, scale, _) in REP_METRICS.items():
                    if rep.get(metric) is not None:
                        out[key].append(rep[metric] * scale)
    return out, used


def chat_mean(run: Mapping[str, Any], batch: Mapping[str, Any], metric: str) -> float | None:
    """A batch's mean of one chat field over its ok chats (conc_ttft.py), from the run's chat rows by label."""
    rows = {r.get("label"): r for r in ((run.get("measured") or {}).get("chats") or {}).get("reps", [])}
    vals = [rows[lab][metric] for lab in batch.get("requests", []) if lab in rows and rows[lab].get("ok")
            and rows[lab].get(metric) is not None]
    return statistics.fmean(vals) if vals else None


def batch_values(prefix: str, members: Sequence[Member]) -> tuple[dict[str, list[float]], list[Member]]:
    """stats.batches_for: every ok batch of a group that holds one concurrency (stats.py:68 to 82)."""
    levels = sorted({concurrency_of(mem.run) for mem in members})
    if len(levels) > 1:
        raise ReportError(f"group {prefix!r} mixes concurrency {levels}")
    out: dict[str, list[float]] = {k: [] for k in BATCH_METRICS}
    for mem in members:
        for batch in batches_of(mem.run):
            if batch.get("status") != "ok":
                continue
            values = {**batch, **{m: chat_mean(mem.run, batch, m) for m in ("ttft_ms", "total_ms")}}
            for key, (metric, scale, _) in BATCH_METRICS.items():
                if values.get(metric) is not None:
                    out[key].append(values[metric] * scale)
    return out, list(members)


def boot_ratio(a: Sequence[float], b: Sequence[float], n: int = RESAMPLES) -> tuple[float, float] | None:
    """stats.boot_ratio (stats.py:94 to 98); None when a resample of A has mean 0 (stats.py raises there)."""
    rng = random.Random(7)  # noqa: S311 - bootstrap resampling, not security
    ratios = []
    for _ in range(n):
        num = statistics.fmean(rng.choices(b, k=len(b)))
        den = statistics.fmean(rng.choices(a, k=len(a)))
        if den == 0:
            return None
        ratios.append(num / den)
    ratios.sort()
    return ratios[int(0.025 * n)], ratios[int(0.975 * n) - 1]


def compare(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """n, means, ratio of means B/A and its 95% interval, or a note saying why there is no ratio."""
    out: dict[str, Any] = {"n_a": len(a), "n_b": len(b), "mean_a": statistics.fmean(a) if a else None,
                           "mean_b": statistics.fmean(b) if b else None}
    if len(a) < 2 or len(b) < 2:
        return {**out, "note": f"n/a (n {len(a)}, {len(b)}: under 2)"}
    if out["mean_a"] == 0 or out["mean_b"] == 0:
        return {**out, "note": "n/a (mean 0)"}
    ci = boot_ratio(a, b)
    if ci is None:
        return {**out, "note": "n/a (a resample of A has mean 0)"}
    return {**out, "ratio": out["mean_b"] / out["mean_a"], "lo": ci[0], "hi": ci[1]}


def pair_result(pair: Mapping[str, Any], files: Sequence[IndexFile], store: RunStore,
                warn: list[str]) -> dict[str, Any]:
    """Every metric of one pair, and the records it used."""
    chosen = [f for f in files if f.name in pair["indexes"]]
    missing = sorted(set(pair["indexes"]) - {f.name for f in chosen})
    if missing:
        return {"error": f"index not in the glob: {', '.join(missing)}", "metrics": {}, "used": [], "dropped": []}
    only = set(pair["only_labels"]) if pair.get("only_labels") else None
    dropped = dropped_members((pair["a_prefix"], pair["b_prefix"]), chosen, only, store, warn, pair["mode"] == "reps")
    try:
        sides = []
        for prefix in (pair["a_prefix"], pair["b_prefix"]):
            members = group_members(prefix, chosen, only, store, warn)
            sides.append(rep_values(members) if pair["mode"] == "reps" else batch_values(prefix, members))
    except ReportError as exc:
        return {"error": str(exc), "metrics": {}, "used": [], "dropped": dropped}
    (va, used_a), (vb, used_b) = sides
    metrics = {key: compare(va[key], vb[key]) for key in pair["metrics"]}
    return {"error": None, "metrics": metrics, "used": [*used_a, *used_b], "dropped": dropped}


def dropped_members(prefixes: Sequence[str], files: Sequence[IndexFile], only: set[str] | None, store: RunStore,
                    warn: list[str], reps: bool) -> list[Member]:
    """The ok runs a label filter leaves out although their label matches a prefix (N = 1 runs only in reps mode)."""
    if not only:
        return []
    out = []
    for prefix in prefixes:
        for mem in group_members(prefix, files, None, store, warn):
            if mem.rec["label"] not in only and (not reps or concurrency_of(mem.run) == 1) and mem not in out:
                out.append(mem)
    return out


# ----------------------------------------------------------------------------- formatting


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def mib_digits(value: float | None, want: str | None = None) -> int:
    """SSD MiB: the decimals NUMBERS.md quotes, else 3 under 1 MiB (so 0.021 does not print as 0.0), else 1."""
    if want is not None:
        return len(want.split(".")[1]) if "." in want else 0
    return 3 if value is not None and abs(value) < 1 else 1


def fmt_ratio(res: Mapping[str, Any]) -> str:
    if "note" in res:
        return res["note"]
    return f"{res['ratio']:.3f} [{res['lo']:.3f}, {res['hi']:.3f}]"


def clean(text: object, limit: int = ERROR_CHARS) -> str:
    """One table-safe line: secrets filtered first (matrix.py:33), then cut to limit characters."""
    line = SECRETISH.sub("<r>", " ".join(str(text).split()))[:limit]
    return line.replace("|", "\\|")


def check_value(expected: str, actual: float | None) -> bool:
    """A NUMBERS.md value matches when ours, printed at its number of decimals, is the same text."""
    if actual is None:
        return False
    digits = len(expected.split(".")[1]) if "." in expected else 0
    return f"{actual:.{digits}f}" == expected


def check_metric(expect: Mapping[str, str], res: Mapping[str, Any]) -> list[str]:
    """The NUMBERS.md fields of one metric that do not match, as 'field NUMBERS x, here y'."""
    keys = {"a": "mean_a", "b": "mean_b", "ratio": "ratio", "lo": "lo", "hi": "hi"}
    out = []
    for fld, want in expect.items():
        got = res.get(keys[fld])
        if not check_value(want, got):
            digits = len(want.split(".")[1]) if "." in want else 0
            out.append(f"{fld} NUMBERS {want}, here {fmt(got, digits)}")
    return out


# ----------------------------------------------------------------------------- rendering bench/RESULTS.md

METHOD = """\
- Ratio is B / A, the `B/A` of `stats.py`. In a pooled-vs-alone pair A is the host alone and B the pool. In an
  N-vs-N = 1 pair A is one chat at a time. For tok/s, above 1 means B is faster. For times, below 1 means B is sooner.
- 95% CI is a percentile bootstrap of the ratio of means, 10,000 resamples, seed 7 (`stats.py:94` to `98`). It
  resamples reps at N = 1 and batches at N > 1. First token at N > 1 is each batch's mean over its chats, resampled
  by batch, as `conc_ttft.py` in NUMBERS does.
- n is measured reps (or batches). Warmups are excluded.
- Common to every row: temperature 0, seed 7, `ignore_eos`, no prompt cache, fixed-length prompts. Decode is the
  engine's own (predicted_n - 1) / predicted_ms. SSD is system-wide physical reads per decoded token, in MiB.
- Small samples from one row per side (2 vs 2, 3 vs 3) show rep-to-rep spread inside one load, not run-to-run
  spread. Say "one pair of runs" when that is the case.
- A group whose mean is 0 prints `n/a (mean 0)` instead of a ratio. `stats.py` divides by zero there."""

FAMILY_HEAD = ("| Pair | Model | Host | Peers | Network | Prompt/answer | N | A vs B | n A | n B | {d} A | {d} B | "
               "{d} B/A [95% CI] | First token s A | First token s B | First token B/A [95% CI] | Index files |")
METRIC_HEAD = ("| Pair | Model | Condition | Prompt/answer | N | n A | n B | Metric | A mean | B mean | "
               "Ratio B/A [95% CI] | NUMBERS | Index files |")
SINGLE_HEAD = ("| Group | Model | Host | Prompt/answer | N | n | Decode tok/s mean | Prefill tok/s mean | "
               "First token s mean | Full answer s mean | SSD MiB/token mean | NUMBERS | Index files |")
RUN_HEAD = ("| Index | Label | Model | Peers | N | Status | Load s | Seed s | Decode tok/s median | "
            "Prefill tok/s median | First token s median | SSD MiB/token median | "
            "Aggregate decode tok/s median (N > 1) | RTT ms before | Wall s |")


def _table(head: str, rows: Iterable[str]) -> list[str]:
    return [head, "|" + "---|" * (head.count(" | ") + 1), *rows]


def _n_text(pair: Mapping[str, Any]) -> str:
    return f"1 vs {pair['n']}" if pair["kind"] == "n_vs_n1" else str(pair["n"])


def _indexes(item: Mapping[str, Any]) -> str:
    return ", ".join(f"`{i}`" for i in item["indexes"])


def _metric_label(pair: Mapping[str, Any], key: str) -> str:
    table = REP_METRICS if pair["mode"] == "reps" else BATCH_METRICS
    return table[key][2]


def family_row(pair: Mapping[str, Any], cond: Mapping[str, str], res: Mapping[str, Any]) -> str:
    dec = res["metrics"].get(DECODE_KEY[pair["mode"]], {})
    ttft = res["metrics"].get("first_token", {})
    sides = f"`{pair['a_prefix']}` vs `{pair['b_prefix']}`"
    cells = [pair["id"], pair["model"], cond["host"], pair.get("peers") or cond["peers"], cond["network"],
             pair["prompt_answer"], _n_text(pair), sides, str(dec.get("n_a", 0)), str(dec.get("n_b", 0)),
             fmt(dec.get("mean_a")), fmt(dec.get("mean_b")), fmt_ratio(dec) if dec else "n/a",
             fmt(ttft.get("mean_a"), 2), fmt(ttft.get("mean_b"), 2), fmt_ratio(ttft) if ttft else "n/a",
             _indexes(pair)]
    if res["error"]:
        cells[10:16] = ["n/a"] * 5 + [clean(res["error"])]
    return "| " + " | ".join(cells) + " |"


def numbers_cell(ref: str, expect: Mapping[str, str] | None, res: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The NUMBERS column text and the list of mismatches for one metric."""
    if not expect:
        return f"{ref}: not quoted", []
    bad = check_metric(expect, res)
    return (f"{ref}: differs ({'; '.join(bad)})" if bad else f"{ref}: match"), bad


def metric_rows(pair: Mapping[str, Any], res: Mapping[str, Any], mismatches: list[str]) -> list[str]:
    head = f"| {pair['id']} | {pair['model']} | {pair['condition']} | {pair['prompt_answer']} | {_n_text(pair)} |"
    if res["error"]:
        mismatches.append(f"{pair['id']}: {res['error']}")
        return [f"{head} n/a | n/a | all | n/a | n/a | {clean(res['error'])} | {pair['numbers_ref']} | "
                f"{_indexes(pair)} |"]
    rows = []
    for key in pair["metrics"]:
        m = res["metrics"][key]
        expect = (pair.get("expect") or {}).get(key)
        cell, bad = numbers_cell(pair["numbers_ref"], expect, m)
        mismatches.extend(f"{pair['id']} {_metric_label(pair, key)}: {b}" for b in bad)
        means = []
        for side in ("a", "b"):
            value = m[f"mean_{side}"]
            digits = (2 if key in SECONDS_KEYS else
                      mib_digits(value, (expect or {}).get(side)) if key in MIB_KEYS else 3)
            means.append(fmt(value, digits))
        rows.append(f"{head} {m['n_a']} | {m['n_b']} | {_metric_label(pair, key)} | {means[0]} | {means[1]} | "
                    f"{fmt_ratio(m)} | {cell} | {_indexes(pair)} |")
    return rows


def single_result(single: Mapping[str, Any], files: Sequence[IndexFile], store: RunStore,
                  warn: list[str]) -> tuple[dict[str, dict[str, Any]], list[Member]]:
    """One group's means over its N = 1 reps (a host alone with no pair to compare against)."""
    chosen = [f for f in files if f.name in single["indexes"]]
    only = set(single["only_labels"]) if single.get("only_labels") else None
    vals, used = rep_values(group_members(single["prefix"], chosen, only, store, warn))
    return {k: {"n_a": len(v), "mean_a": statistics.fmean(v) if v else None} for k, v in vals.items()}, used


def single_row(single: Mapping[str, Any], cond: Mapping[str, str], res: Mapping[str, dict[str, Any]],
               mismatches: list[str]) -> str:
    bad = []
    for key, expect in (single.get("expect") or {}).items():
        bad.extend(f"{REP_METRICS[key][2]} {b}" for b in check_metric(expect, res[key]))
    mismatches.extend(f"{single['id']} {b}" for b in bad)
    ref = single["numbers_ref"]
    cell = (f"{ref}: differs ({'; '.join(bad)})" if bad else f"{ref}: match") + (
        f". {single['note']}" if single.get("note") else "")
    ssd = res["ssd"]["mean_a"]
    ssd_digits = mib_digits(ssd, ((single.get("expect") or {}).get("ssd") or {}).get("a"))
    return (f"| {single['id']} | {single['model']} | {cond['host']} | {single['prompt_answer']} | {single['n']} | "
            f"{res['decode']['n_a']} | {fmt(res['decode']['mean_a'])} | {fmt(res['prefill']['mean_a'])} | "
            f"{fmt(res['first_token']['mean_a'], 2)} | {fmt(res['answer']['mean_a'], 2)} | "
            f"{fmt(ssd, ssd_digits)} | {cell} | {_indexes(single)} |")


def run_row(index: str, rec: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    batch = row.get("batch") or {}
    agg = batch.get("aggregate_decode_tok_s") or {}
    agg_med = fmt(agg["median"]) if row.get("concurrency", 1) > 1 and agg.get("n") else "n/a"
    rtt = ", ".join(str(x) for x in row.get("rtt_ms") or []) or "n/a"
    n = str(row["concurrency"]) if "concurrency" in row else "n/a"
    fine = row.get("disk_mib_fine")
    disk = fmt(fine, 3) if fine is not None and abs(fine) < 1 else fmt(row.get("disk_mib_per_tok"), 1)
    return (f"| {index} | {row['label']} | {row['model']} | {row['peers']} | {n} | {row['status']} | "
            f"{fmt(row.get('load_s'), 1)} | {fmt(row.get('seed_s'), 1)} | {fmt(row.get('decode_med'))} | "
            f"{fmt(row.get('prefill_med'))} | {fmt(row.get('ttft_med_s'), 2)} | "
            f"{disk} | {agg_med} | {rtt} | {fmt(rec['wall_s'], 1)} |")


def exclusion(f: IndexFile, row: Mapping[str, Any], used: bool, filtered_by: Sequence[str] = ()) -> str | None:
    """Why a record is in no ratio, or None when a pair or a single group used it."""
    label, status = row["label"], row["status"]
    err = f", {clean(row['error'])}" if row.get("error") else ""
    if f.archived:
        return f"{status}, archived index file{err}"
    if label.endswith("-refused"):
        return f"{status}, first attempt of a row matrix retried{err}"
    if label.endswith("-w"):
        return f"{status}, warm-up run"
    if status == "no result":
        return f"no result, no run.json (exit {row['exit']})"
    if status != "ok":
        return f"{status}{err or ', no error message'}"
    if used:
        return None
    if filtered_by:
        return f"ok, outside the only_labels of pair {', '.join(filtered_by)} in bench/pairs.json"
    return "ok, not in any pair or group of bench/pairs.json"


def filter_note(pair: Mapping[str, Any], res: Mapping[str, Any]) -> str | None:
    """How to get a filtered pair's numbers from stats.py, and which runs the filter leaves out."""
    if not pair.get("only_labels"):
        return None
    head = f"- {pair['id']}: label filter, as `stats.py --only-labels {','.join(pair['only_labels'])}`."
    dropped = ", ".join(f"`{m.rec['label']}`" for m in res.get("dropped", []))
    if not dropped:
        return f"{head} It leaves out no ok run, so the two prefixes alone give the same numbers."
    return f"{head} It leaves out {dropped}, so `stats.py` on the two prefixes alone gives other numbers."


# ----------------------------------------------------------------------------- pairs.json and the whole report

PAIR_KEYS = ("id", "family", "kind", "condition", "model", "prompt_answer", "n", "a_prefix", "b_prefix", "indexes",
             "only_labels", "mode", "metrics", "numbers_ref")
SINGLE_KEYS = ("id", "family", "condition", "model", "prompt_answer", "n", "prefix", "indexes", "only_labels",
               "numbers_ref")
KINDS = ("pooled_vs_alone", "n_vs_n1", "pooled_vs_pooled")
ROW_FIELDS = ("status", "exit", "load_s", "seed_s", "wall_s")
NUMBER_TEXT = re.compile(r"^-?\d+(\.\d+)?$")


def _check_expect(where: str, expect: object, allowed: Iterable[str], fields: Sequence[str]) -> None:
    if not isinstance(expect, dict):
        raise ReportError(f"{where}: expect must be an object")
    for key, vals in expect.items():
        if key not in allowed or not isinstance(vals, dict):
            raise ReportError(f"{where}: expect has an unknown metric {key!r}")
        for fld, text in vals.items():
            if fld not in fields or not isinstance(text, str) or not NUMBER_TEXT.match(text):
                raise ReportError(f"{where}: expect {key}.{fld} must be one of {list(fields)} with a number string")


def _check_pair(pair: Mapping[str, Any], doc: Mapping[str, Any], families: set[str]) -> None:
    where = f"pair {pair.get('id')!r}"
    missing = [k for k in PAIR_KEYS if k not in pair]
    if missing:
        raise ReportError(f"{where}: missing {', '.join(missing)}")
    table = {"reps": REP_METRICS, "batches": BATCH_METRICS}.get(pair["mode"])
    if table is None or pair["kind"] not in KINDS or pair["family"] not in families:
        raise ReportError(f"{where}: bad mode, kind or family")
    if pair["condition"] not in doc["conditions"] or not set(pair["metrics"]) <= set(table):
        raise ReportError(f"{where}: unknown condition or metric")
    _check_expect(where, pair.get("expect", {}), pair["metrics"], ("a", "b", "ratio", "lo", "hi"))


def validate_pairs(doc: object) -> dict[str, Any]:
    """bench/pairs.json: conditions, families, pairs and single groups, each with the keys the report reads."""
    if not isinstance(doc, dict) or not all(k in doc for k in ("conditions", "families", "pairs", "singles")):
        raise ReportError("pairs.json needs conditions, families, pairs and singles")
    families = {f.get("id") for f in doc["families"] if isinstance(f, dict) and "title" in f and "intro" in f}
    for cond in doc["conditions"].values():
        if not all(isinstance(cond.get(k), str) for k in ("host", "peers", "network", "detail", "source")):
            raise ReportError("each condition needs host, peers, network, detail and source strings")
    ids = [p.get("id") for p in doc["pairs"]] + [s.get("id") for s in doc["singles"]]
    if len(ids) != len(set(ids)):
        raise ReportError("pair and group ids must be unique")
    for pair in doc["pairs"]:
        _check_pair(pair, doc, families)
    for single in doc["singles"]:
        where = f"group {single.get('id')!r}"
        missing = [k for k in SINGLE_KEYS if k not in single]
        if missing or single["family"] not in families or single["condition"] not in doc["conditions"]:
            raise ReportError(f"{where}: missing {', '.join(missing) or 'a known family or condition'}")
        _check_expect(where, single.get("expect", {}), REP_METRICS, ("a",))
    for chk in doc.get("row_checks", []):
        expect = chk.get("expect") if isinstance(chk, dict) else None
        if not isinstance(expect, dict) or not all(isinstance(chk.get(k), str) for k in ("index", "label",
                                                                                          "numbers_ref")):
            raise ReportError("each row check needs index, label, numbers_ref and an expect object")
        if not set(expect) <= set(ROW_FIELDS) or not all(isinstance(v, str) for v in expect.values()):
            raise ReportError(f"row check {chk['label']!r}: expect fields are {list(ROW_FIELDS)}, as strings")
    return doc


def load_pairs(path: Path) -> dict[str, Any]:
    try:
        return validate_pairs(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ReportError(f"cannot use {path}: {exc}") from exc


@dataclass(frozen=True)
class Report:
    text: str
    mismatches: tuple[str, ...]
    checked: int


def _family_sections(doc: Mapping[str, Any], results: Mapping[str, Any], singles: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    conds = doc["conditions"]
    for num, fam in enumerate(doc["families"], start=1):
        pairs = [p for p in doc["pairs"] if p["family"] == fam["id"]]
        out += ["", f"### 1.{num} {fam['title']}", "", fam["intro"], ""]
        if pairs:
            label = "Aggregate decode tok/s" if all(p["mode"] == "batches" for p in pairs) else "Decode tok/s"
            out += _table(FAMILY_HEAD.format(d=label),
                          (family_row(p, conds[p["condition"]], results[p["id"]]) for p in pairs))
            notes = [n for n in (filter_note(p, results[p["id"]]) for p in pairs) if n]
            out += ["", "Pairs that use only some of the runs under their prefixes:", "", *notes] if notes else []
        group = [s for s in doc["singles"] if s["family"] == fam["id"]]
        if group:
            out += ["", "One host alone, no ratio:", ""] if pairs else []
            out += _table(SINGLE_HEAD, (singles[s["id"]] for s in group))
    return out


def _run_sections(files: Sequence[IndexFile], rows: Mapping[tuple[str, int], dict[str, Any]],
                  used: set[tuple[str, int]],
                  filtered: Mapping[tuple[str, int], list[str]]) -> tuple[list[str], list[str]]:
    every, excluded = [], []
    for f in files:
        if not f.records:
            excluded.append(f"- {f.name}: empty index file")
        for pos, rec in enumerate(f.records):
            row = rows[(f.name, pos)]
            every.append(run_row(f.name, rec, row))
            why = exclusion(f, row, (f.name, pos) in used, filtered.get((f.name, pos), ()))
            if why:
                excluded.append(f"- {f.name} {row['label']}: {why}")
    return _table(RUN_HEAD, every), excluded


def row_check_lines(checks: Sequence[Mapping[str, Any]], rows: Mapping[tuple[str, int], dict[str, Any]],
                    files: Sequence[IndexFile], mismatches: list[str]) -> list[str]:
    """One table line per checked field of a run row (load, seeding, wall, status, exit) against NUMBERS.md."""
    lines = []
    for chk in checks:
        found = [rows[(f.name, pos)] for f in files if f.name == chk["index"]
                 for pos, rec in enumerate(f.records) if rec["label"] == chk["label"]]
        row = found[0] if found else None
        for fld, want in chk["expect"].items():
            got = None if row is None else row.get(fld)
            if fld in ("status", "exit"):
                ok, shown = got is not None and str(got) == want, "n/a" if got is None else str(got)
            else:
                digits = len(want.split(".")[1]) if "." in want else 0
                ok, shown = check_value(want, got), fmt(got, digits)
            if not ok:
                mismatches.append(f"{chk['index']} {chk['label']} {fld}: NUMBERS {want}, here {shown}")
            lines.append(f"| {chk['index']} | {chk['label']} | {fld} | {want} | {shown} | "
                         f"{chk['numbers_ref']}: {'match' if ok else 'differs'} |")
    return lines


def _header(files: Sequence[IndexFile], rows: Mapping[Any, dict[str, Any]], pattern: str,
            doc: Mapping[str, Any]) -> list[str]:
    records = sum(len(f.records) for f in files)
    with_run = sum(1 for r in rows.values() if r["status"] != "no result")
    lines = ["# Hypercompute benchmark results", "",
             f"Generated by `uv run --no-sync python bench/hcbench.py report` from {len(files)} index files matching "
             f"`{pattern}`:", f"{records} records, {with_run} with a run.json. Do not edit by hand. "
             "docs/pitch/NUMBERS.md is the sheet to quote from.", "", "## Method", "", METHOD, "",
             "## Condition keys", ""]
    lines += _table("| Key | Host | Peers | Network | Source |",
                    (f"| {k} | {c['host']} | {c['peers']} | {c['detail']} | {c['source']} |"
                     for k, c in doc["conditions"].items()))
    return lines


def build_report(doc: Mapping[str, Any], files: Sequence[IndexFile], store: RunStore, pattern: str,
                 warn: list[str]) -> Report:
    rows = {(f.name, pos): row_values(rec, store) for f in files for pos, rec in enumerate(f.records)}
    results = {p["id"]: pair_result(p, files, store, warn) for p in doc["pairs"]}
    mismatches: list[str] = []
    used = {(m.index, m.position) for r in results.values() for m in r["used"]}
    filtered: dict[tuple[str, int], list[str]] = {}
    for pid, r in results.items():
        for m in r["dropped"]:
            filtered.setdefault((m.index, m.position), []).append(pid)
    singles = {}
    for s in doc["singles"]:
        res, members = single_result(s, files, store, warn)
        used |= {(m.index, m.position) for m in members}
        singles[s["id"]] = single_row(s, doc["conditions"][s["condition"]], res, mismatches)
    metric_lines = [line for p in doc["pairs"] for line in metric_rows(p, results[p["id"]], mismatches)]
    checks = doc.get("row_checks", [])
    row_lines = row_check_lines(checks, rows, files, mismatches)
    checked = sum(len(v) for item in (*doc["pairs"], *doc["singles"]) for v in (item.get("expect") or {}).values())
    checked += len(row_lines)
    every, excluded = _run_sections(files, rows, used, filtered)
    lines = _header(files, rows, pattern, doc)
    lines += ["", "## 1. Pairs", "", "One row per pair with decode and first token. Every metric of every pair, with "
              "the NUMBERS.md check, is in the last subsection."]
    lines += _family_sections(doc, results, singles)
    fams = len(doc["families"])
    lines += ["", f"### 1.{fams + 1} Every metric, checked against docs/pitch/NUMBERS.md", ""]
    lines += _table(METRIC_HEAD, metric_lines)
    family_refs = "1.1" if fams == 1 else f"1.1 to 1.{fams}"
    lines += ["", "## 2. Every run", "", *every, "", "## 3. Not in any ratio", "", *(excluded or ["- none"]), "",
              "## 4. Cross-check against docs/pitch/NUMBERS.md", "",
              f"Pair and group values are checked in 1.{fams + 1} and {family_refs}. Run-row values quoted in "
              "NUMBERS.md:", ""]
    lines += _table("| Index | Label | Field | NUMBERS | Here | Check |", row_lines)
    lines += ["", f"{checked} quoted values checked at the precision NUMBERS.md prints them. {len(mismatches)} differ"
              + (":" if mismatches else "."), *(f"- {m}" for m in mismatches)]
    return Report("\n".join(lines) + "\n", tuple(mismatches), checked)


def report_main(pattern: str, results_arg: str | None, pairs_path: Path, out: Path, check: bool,
                repo: Path = REPO) -> int:
    """hcbench.py report. Exit 0 when written or up to date, 1 when --check finds a difference, 2 on bad inputs."""
    warn: list[str] = []
    try:
        doc = load_pairs(pairs_path)
        files = load_indexes(pattern, repo, warn)
        if not files:
            raise ReportError(f"no index files match {pattern!r} under {repo}")
        rep = build_report(doc, files, RunStore(results_root(results_arg, repo)), pattern, warn)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for line in dict.fromkeys(warn):
        print(line, file=sys.stderr)
    for m in rep.mismatches:
        print(f"NUMBERS mismatch: {m}", file=sys.stderr)
    if check:
        current = out.read_text(encoding="utf-8") if out.is_file() else None
        if current != rep.text:
            print(f"{out} is out of date: run hcbench.py report", file=sys.stderr)
            return 1
        print(f"{out} is up to date ({rep.checked} NUMBERS values checked, {len(rep.mismatches)} differ)")
        return 0
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(rep.text)
    print(f"wrote {out}: {sum(len(f.records) for f in files)} records from {len(files)} index files, "
          f"{rep.checked} NUMBERS values checked, {len(rep.mismatches)} differ")
    return 0
