"""Parity with the real records: summarize.row per record, the PACKAGING-SPEC 4.5 values, and stats.py itself.

These tests only read files under results/capacity/. They are skipped when the campaign folder is missing.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import hcbench_report as rep
import pytest

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "capacity"
STATS = REPO / "scripts" / "capacity" / "paper" / "stats.py"
pytestmark = pytest.mark.skipif(not (RESULTS / "paper-2026-10-24-laptop").is_dir(),
                                reason="campaign results not in this checkout")

# PACKAGING-SPEC.md 4.5, reproduced there with stats.py on 2026-10-24: (ratio, lo, hi) to 3 decimals
SPEC_45 = {
    ("d70", "decode"): ("5.444", "5.423", "5.464"), ("d70", "prefill"): ("1.967", "1.896", "2.043"),
    ("d70", "ssd"): ("0.118", "0.118", "0.118"),
    ("g120", "decode"): ("2.283", "2.000", "2.538"), ("g120", "prefill"): ("7.759", "6.945", "8.855"),
    ("vl72", "decode"): ("2.592", "2.571", "2.613"),
    ("g120n4", "aggregate_decode"): ("1.464", "1.295", "1.674"),
    ("g120n8", "aggregate_decode"): ("1.349", "1.246", "1.479"), ("g120n8", "step"): ("0.742", "0.677", "0.811"),
    ("g120n8", "ssd_step"): ("0.171", "0.140", "0.201"),
    ("n80", "decode"): ("0.468", "0.362", "0.611"),
    ("671b", "decode"): ("0.830", "0.769", "0.903"), ("671b", "prefill"): ("2.418", "2.194", "2.721"),
    ("kimi", "decode"): ("0.856", "0.818", "0.886"), ("kimi", "prefill"): ("1.507", "1.412", "1.587"),
    ("kimi", "ssd"): ("0.379", "0.365", "0.397"),
    ("g120gx-n8", "aggregate_decode"): ("2.731", "2.583", "2.901"),
    ("r2-3mac", "decode"): ("1.274", "1.105", "1.464"), ("r2-3mac", "prefill"): ("2.994", "2.639", "3.349"),
    ("r2-3mac", "ssd"): ("0.151", "0.117", "0.188"),
    ("r2-1mac", "decode"): ("1.253", "1.106", "1.407"),
    ("r1-235b", "decode"): ("1.079", "0.976", "1.184"),
}
# The R7 laptop indexes may still grow (a second d70 pair is queued, NUMBERS 1a): pin them to the rows NUMBERS used.
FROZEN = {
    "d70": ["d70-g14A-1", "d70-g14P2-1", "d70-g14A-2"], "g120": ["g120-g14A-1", "g120-g14P2-1"],
    "g120-batches": ["g120-g14A-1", "g120-g14P2-1"], "vl72": ["vl72-g14A-1", "vl72-g14P2-1"],
    "g120n4": ["g120n4-g14A-1", "g120n4-g14P2-1"], "g120n8": ["g120n8-g14A-1", "g120n8-g14P2-1"],
    "n80": ["n80-g14A-1", "n80-g14P2-1", "n80-g14A-2"],
}
STATS_NAMES = {"decode_tok_s": "decode", "prefill_tok_s": "prefill", "decode_disk_read_bytes_per_token": "ssd",
               "aggregate_decode_tok_s": "aggregate_decode", "step_ms": "step", "ssd_bytes_per_step": "ssd_step"}
STATS_LINE = re.compile(r"^(\w+): (\S+) n=(\d+) mean ([\d.]+) median [\d.]+ \| (\S+) n=(\d+) mean ([\d.]+) "
                        r"median [\d.]+ \| ratio ([\d.]+) \[95% CI ([\d.]+), ([\d.]+)\]$")


@pytest.fixture(scope="module")
def files() -> list[rep.IndexFile]:
    """Every default-glob index, read once: the live campaign may append between reads."""
    return rep.load_indexes(rep.DEFAULT_GLOB, REPO, [])


def pinned(pair: dict) -> dict:
    return {**pair, "only_labels": pair["only_labels"] or FROZEN.get(pair["id"])}


@pytest.fixture(scope="module")
def pairs() -> dict:
    return {p["id"]: pinned(p) for p in rep.load_pairs(REPO / "bench" / "pairs.json")["pairs"]}


def label_args(pair: dict) -> list[str]:
    return ["--only-labels", ",".join(pair["only_labels"])] if pair["only_labels"] else []


def load_summarize(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201 - a module
    monkeypatch.setenv("HC_PAPER_REPO", str(REPO))
    path = REPO / "scripts" / "capacity" / "paper" / "summarize.py"
    spec = importlib.util.spec_from_file_location("hcbench_parity_summarize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rows_equal_summarize_row(files, monkeypatch):
    summarize = load_summarize(monkeypatch)
    store = rep.RunStore(RESULTS)
    count = 0
    for f in files:
        for rec in f.records:
            ours = rep.row_values(rec, store)
            extra = {k: ours.pop(k) for k in ("seed_s", "ttft_med_s", "disk_mib_fine") if k in ours}
            assert ours == summarize.row(rec), rec["label"]
            assert set(extra) <= {"seed_s", "ttft_med_s", "disk_mib_fine"}
            if extra.get("disk_mib_fine") is not None:  # the same median, at 3 decimals instead of 1
                assert round(extra["disk_mib_fine"], 1) == pytest.approx(ours["disk_mib_per_tok"], abs=0.05001)
            count += 1
    assert count >= 96


def test_spec_45_values(files, pairs):
    store = rep.RunStore(RESULTS)
    results = {pid: rep.pair_result(pairs[pid], files, store, []) for pid in {k[0] for k in SPEC_45}}
    for (pid, metric), want in SPEC_45.items():
        res = results[pid]["metrics"][metric]
        assert (rep.fmt(res["ratio"]), rep.fmt(res["lo"]), rep.fmt(res["hi"])) == want, (pid, metric)


def stats_lines(*args: str) -> tuple[subprocess.CompletedProcess, dict[str, tuple[str, ...]]]:
    env = {**os.environ, "HC_PAPER_REPO": str(REPO)}
    proc = subprocess.run([sys.executable, "-I", str(STATS), *args], capture_output=True, text=True, env=env,
                          timeout=300, check=False)
    out = {}
    for line in proc.stdout.splitlines():
        m = STATS_LINE.match(line)
        if m:
            out[STATS_NAMES[m.group(1)]] = (m.group(3), m.group(4), m.group(6), m.group(7), m.group(8), m.group(9),
                                            m.group(10))
    return proc, out


def ours(pair: dict, files: list[rep.IndexFile]) -> dict[str, tuple[str, ...]]:
    res = rep.pair_result(pair, files, rep.RunStore(RESULTS), [])
    return {k: (str(m["n_a"]), rep.fmt(m["mean_a"]), str(m["n_b"]), rep.fmt(m["mean_b"]), rep.fmt(m["ratio"]),
                rep.fmt(m["lo"]), rep.fmt(m["hi"])) for k, m in res["metrics"].items() if "ratio" in m}


@pytest.mark.parametrize("pid", ["d70", "kimi"])
def test_stats_py_reps_line_for_line(files, pairs, pid):
    pair = pairs[pid]
    proc, theirs = stats_lines(pair["a_prefix"], pair["b_prefix"], *(str(RESULTS / i) for i in pair["indexes"]),
                               *label_args(pair))
    assert proc.returncode == 0, proc.stderr
    assert set(theirs) == {"decode", "prefill", "ssd"}
    mine = ours(pair, files)
    for metric, line in theirs.items():
        assert mine[metric] == line, metric


def test_stats_py_batches_line_for_line(files, pairs):
    pair = pairs["g120n8"]
    proc, theirs = stats_lines("--batches", pair["a_prefix"], pair["b_prefix"],
                               *(str(RESULTS / i) for i in pair["indexes"]), *label_args(pair))
    assert proc.returncode == 0, proc.stderr
    assert set(theirs) == {"aggregate_decode", "step", "ssd_step"}
    mine = ours(pair, files)
    for metric, line in theirs.items():
        assert mine[metric] == line, metric


def test_zero_mean_group_where_stats_py_divides_by_zero(files, pairs):
    """PACKAGING-SPEC 4.4: stats.py --batches on the GX10 120B N = 8 vs N = 1 hits a zero SSD mean."""
    pair = pairs["g120gx-n8"]
    proc, theirs = stats_lines("--batches", pair["a_prefix"], pair["b_prefix"],
                               *(str(RESULTS / i) for i in pair["indexes"]))
    if proc.returncode != 0:
        assert "ZeroDivisionError" in proc.stderr
    res = rep.pair_result(pair, files, rep.RunStore(RESULTS), [])
    assert res["metrics"]["ssd_step"]["note"] == "n/a (mean 0)"
    assert ours(pair, files)["aggregate_decode"] == theirs["aggregate_decode"]


def test_results_md_matches_numbers(files):
    """Every NUMBERS.md value that bench/pairs.json quotes matches the records (the RESULTS.md section 4 claim)."""
    doc = rep.load_pairs(REPO / "bench" / "pairs.json")
    doc = {**doc, "pairs": [pinned(p) for p in doc["pairs"]]}
    report = rep.build_report(doc, files, rep.RunStore(RESULTS), rep.DEFAULT_GLOB, [])
    assert report.mismatches == ()
    assert report.checked > 400
