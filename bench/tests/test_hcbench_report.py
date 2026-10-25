"""hcbench report on hand-made fixtures: row values, pair math, the zero-mean guard, exclusions and --check."""

from __future__ import annotations

import json
import random
import shutil
import statistics
from pathlib import Path

import hcbench_report as rep
import pytest


def run_report(fixtures: Path, tmp_path: Path, glob_dir: Path | None = None, pairs: Path | None = None,
               check: bool = False, out: Path | None = None) -> tuple[int, Path]:
    out = out or tmp_path / "RESULTS.md"
    pattern = str((glob_dir or fixtures / "capacity" / "paper-fx") / "index_*.jsonl")
    code = rep.report_main(pattern, str(fixtures / "capacity"), pairs or fixtures / "pairs_fx.json", out, check)
    return code, out


def section(text: str, title: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(title)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start + 1:end]


def metric_line(text: str, pair: str, metric: str) -> list[str]:
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 13 and cells[0] == pair and cells[7] == metric:
            return cells
    raise AssertionError(f"no metric row {pair} {metric}")


def test_report_tables(fixtures, tmp_path, capsys):
    code, out = run_report(fixtures, tmp_path)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "from 3 index files" in text and "6 records, 5 with a run.json" in text
    every = section(text, "## 2. Every run")
    assert ("| paper-fx/index_fx.jsonl | fx-hA-1 | fx-model-q4 | 0 | 1 | ok | 12.3 | n/a | 3.000 | 15.000 | 2.00 | "
            "n/a | n/a | 5, 7 | 100.0 |") in every
    assert ("| paper-fx/index_fx.jsonl | fx-hP2-1 | fx-model-q4 | 2 | 1 | ok | 45.0 | 30.1 | 6.000 | 30.000 | 0.50 | "
            "1.0 | n/a | 6 | 200.0 |") in every
    assert ("| paper-fx/index_fx.jsonl | fxn4-hA-1 | fx-model-q4 | 0 | 4 | ok | 20.0 | n/a | n/a | n/a | n/a | n/a | "
            "10.000 | n/a | 300.0 |") in every
    assert ("| paper-fx/index_fx.jsonl | fx-hA-2 | fx-model-q4 | 0 | n/a | no result | n/a | n/a | n/a | n/a | n/a | "
            "n/a | n/a | n/a | 400.0 |") in every
    err = capsys.readouterr().err
    assert "fx-hA-2: no run.json" in err


def test_pair_math(fixtures, tmp_path):
    _, out = run_report(fixtures, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert metric_line(text, "fx", "Decode tok/s")[5:12] == ["2", "2", "Decode tok/s", "3.000", "6.000",
                                                              "2.000 [1.500, 3.000]", "fx1: match"]
    assert metric_line(text, "fx", "Prefill tok/s")[10] == "2.000 [1.500, 3.000]"
    assert metric_line(text, "fx", "First token, s")[8:11] == ["2.00", "0.50", "0.250 [0.167, 0.500]"]
    assert metric_line(text, "fx", "Full answer, s")[8:11] == ["14.00", "6.00", "0.429 [0.353, 0.545]"]
    # batches: the not_concurrent batch b2 is out; first token is each batch's mean over its chats
    assert metric_line(text, "fxn4", "Aggregate decode tok/s")[8:11] == ["3.000", "10.000", "3.333 [2.000, 6.000]"]
    assert metric_line(text, "fxn4", "Step, ms")[8:11] == ["375.000", "450.000", "1.200 [0.800, 2.000]"]
    assert metric_line(text, "fxn4", "First token per chat, s")[8:11] == ["2.00", "5.00", "2.500 [1.333, 6.000]"]
    assert metric_line(text, "fxn4", "Full answer per chat, s")[8:11] == ["14.00", "22.00", "1.571 [1.176, 2.182]"]


def test_zero_mean_prints_na(fixtures, tmp_path):
    """stats.py divides by a zero mean here (its --batches crash on the GX10 120B SSD); the report does not."""
    _, out = run_report(fixtures, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert metric_line(text, "fx", "SSD MiB per token")[8:11] == ["0.000", "1.0", "n/a (mean 0)"]
    assert metric_line(text, "fxn4", "SSD MiB per step")[8:11] == ["0.000", "0.000", "n/a (mean 0)"]
    assert rep.compare([0.0, 0.0], [1.0, 2.0])["note"] == "n/a (mean 0)"
    assert rep.compare([1.0], [1.0, 2.0])["note"].startswith("n/a (n 1, 2")


def test_resample_with_zero_mean_is_na():
    """A has a non-zero mean, but some resamples of A are all zeros: no ratio, no crash."""
    res = rep.compare([0.0, 1.0], [1.0, 1.0])
    assert res["note"] == "n/a (a resample of A has mean 0)"


def test_exclusions_and_secret_filter(fixtures, tmp_path):
    _, out = run_report(fixtures, tmp_path)
    text = out.read_text(encoding="utf-8")
    excluded = [ln for ln in section(text, "## 3. Not in any ratio") if ln]
    assert excluded[0] == "- paper-fx/index_empty.jsonl: empty index file"
    labels = [ln.split(":")[0] for ln in excluded[1:]]
    assert labels == ["- paper-fx/index_fx.jsonl fx-hP2-2", "- paper-fx/index_fx.jsonl fx-hA-2",
                      "- paper-fx/index_fx_refused.jsonl fx-hP2-3-refused"]
    refused = excluded[1]
    assert "<r>" in refused and "abcdefghij" * 4 not in refused
    assert "\\|" in refused  # a pipe in the error cannot break the list into a table
    message = refused.split("refused, ", 1)[1]
    assert len(message) <= rep.ERROR_CHARS + 2
    assert "no run.json (exit 137)" in excluded[2]
    assert "archived index file" in excluded[3]


def test_family_single_and_row_checks(fixtures, tmp_path):
    _, out = run_report(fixtures, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert ("| fx | Fixture model | fixture host | mac-a, mac-b | fixture network | 64/32 | 1 | `fx-hA` vs `fx-hP2` | "
            "2 | 2 | 3.000 | 6.000 | 2.000 [1.500, 3.000] | 2.00 | 0.50 | 0.250 [0.167, 0.500] | "
            "`paper-fx/index_fx.jsonl` |") in text
    assert "| fx-alone | Fixture model | fixture host | 64/32 | 1 | 2 | 3.000 | 15.000 | 2.00 | 14.00 | 0.000 |" in text
    assert "| paper-fx/index_fx.jsonl | fx-hA-1 | load_s | 12.3 | 12.3 | fx4: match |" in text
    # 7 + 3 pair values, 1 group value, 2 row fields
    assert text.rstrip().endswith("13 quoted values checked at the precision NUMBERS.md prints them. 0 differ.")


def test_cross_check_names_the_metric_subsection(fixtures, tmp_path):
    """Section 4 points at the subsections that exist: the fixture has one family, so 1.1 and the metric table 1.2."""
    _, out = run_report(fixtures, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "### 1.2 Every metric, checked against docs/pitch/NUMBERS.md" in text
    assert "Pair and group values are checked in 1.2 and 1.1. Run-row values quoted in NUMBERS.md:" in text


def test_mismatch_is_reported(fixtures, tmp_path, capsys):
    doc = json.loads((fixtures / "pairs_fx.json").read_text(encoding="utf-8"))
    doc["pairs"][0]["expect"]["decode"]["b"] = "6.001"
    doc["row_checks"][0]["expect"]["load_s"] = "12.4"
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(doc), encoding="utf-8")
    code, out = run_report(fixtures, tmp_path, pairs=pairs)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "fx1: differs (b NUMBERS 6.001, here 6.000)" in metric_line(text, "fx", "Decode tok/s")[11]
    assert "- fx Decode tok/s: b NUMBERS 6.001, here 6.000" in text
    assert "- paper-fx/index_fx.jsonl fx-hA-1 load_s: NUMBERS 12.4, here 12.3" in text
    assert "NUMBERS mismatch: fx Decode tok/s" in capsys.readouterr().err


def test_truncated_last_line_is_skipped(fixtures, tmp_path, capsys):
    paper = tmp_path / "paper-fx"
    shutil.copytree(fixtures / "capacity" / "paper-fx", paper)
    with (paper / "index_fx.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"label": "fx-hA-3", "mod')
    code, out = run_report(fixtures, tmp_path, glob_dir=paper)
    assert code == 0
    assert "6 records" in out.read_text(encoding="utf-8")
    assert "last line is not valid JSON" in capsys.readouterr().err


def test_bad_line_in_the_middle_is_an_error(fixtures, tmp_path, capsys):
    paper = tmp_path / "paper-fx"
    shutil.copytree(fixtures / "capacity" / "paper-fx", paper)
    lines = (paper / "index_fx.jsonl").read_text(encoding="utf-8").splitlines()
    lines.insert(1, "{not json")
    (paper / "index_fx.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    code, out = run_report(fixtures, tmp_path, glob_dir=paper)
    assert code == 2 and not out.exists()
    assert "index_fx.jsonl:2: not valid JSON" in capsys.readouterr().err


def test_check_detects_a_change(fixtures, tmp_path):
    code, out = run_report(fixtures, tmp_path)
    assert code == 0
    assert run_report(fixtures, tmp_path, check=True, out=out)[0] == 0
    out.write_text(out.read_text(encoding="utf-8") + "edited\n", encoding="utf-8", newline="\n")
    before = out.read_bytes()
    assert run_report(fixtures, tmp_path, check=True, out=out)[0] == 1
    assert out.read_bytes() == before  # --check writes nothing
    missing = tmp_path / "missing.md"
    assert run_report(fixtures, tmp_path, check=True, out=missing)[0] == 1 and not missing.exists()


def test_output_is_deterministic_with_lf(fixtures, tmp_path):
    _, first = run_report(fixtures, tmp_path, out=tmp_path / "a.md")
    _, second = run_report(fixtures, tmp_path, out=tmp_path / "b.md")
    assert first.read_bytes() == second.read_bytes()
    assert b"\r" not in first.read_bytes()


def test_bad_pairs_file_is_refused(fixtures, tmp_path, capsys):
    doc = json.loads((fixtures / "pairs_fx.json").read_text(encoding="utf-8"))
    doc["pairs"][0]["metrics"] = ["decode", "step"]  # step is a batches metric
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(doc), encoding="utf-8")
    assert run_report(fixtures, tmp_path, pairs=pairs)[0] == 2
    assert "unknown condition or metric" in capsys.readouterr().err


def test_mixed_concurrency_group_is_an_error_in_the_pair(fixtures, tmp_path):
    doc = json.loads((fixtures / "pairs_fx.json").read_text(encoding="utf-8"))
    doc["pairs"][1]["b_prefix"] = "fx"  # catches fx-hA-1 (N = 1) and fxn4-hA-1 (N = 4)
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(doc), encoding="utf-8")
    _, out = run_report(fixtures, tmp_path, pairs=pairs)
    text = out.read_text(encoding="utf-8")
    assert "group 'fx' mixes concurrency [1, 4]" in text


def test_boot_ratio_matches_stats_py_arithmetic():
    """The bootstrap draws B then A from one Random(7), 10,000 times, and takes sorted ratios at 250 and 9749."""
    gen = random.Random(1)  # noqa: S311 - test data
    a = [gen.uniform(1, 3) for _ in range(5)]
    b = [gen.uniform(2, 6) for _ in range(4)]
    rng = random.Random(7)  # noqa: S311 - the bootstrap under test
    ratios = sorted(statistics.fmean(rng.choices(b, k=len(b))) / statistics.fmean(rng.choices(a, k=len(a)))
                    for _ in range(10_000))
    assert rep.boot_ratio(a, b) == (ratios[250], ratios[9749])


@pytest.mark.parametrize(("expected", "actual", "ok"), [("0.830", 0.8304, True), ("0.830", 0.8306, False),
                                                        ("23454", 23454.244, True), ("6.4", 6.431, True),
                                                        ("6.5", 6.431, False), ("10.02", None, False)])
def test_check_value(expected, actual, ok):
    assert rep.check_value(expected, actual) is ok


@pytest.mark.parametrize(("value", "want", "digits"), [(0.021, None, 3), (0.0, None, 3), (2484.9, None, 1),
                                                       (None, None, 1), (0.0214, "0.021", 3), (23454.2, "23454", 0),
                                                       (9.974, "9.974", 3)])
def test_mib_digits(value, want, digits):
    assert rep.mib_digits(value, want) == digits


def test_small_ssd_values_keep_three_decimals(fixtures):
    """0.021 MiB per token must not print as 0.0 next to a NUMBERS match."""
    doc = json.loads((fixtures / "pairs_fx.json").read_text(encoding="utf-8"))
    row = {"label": "x", "model": "m", "peers": 0, "status": "ok", "wall_s": 1.0, "concurrency": 1,
           "disk_mib_per_tok": 0.0, "disk_mib_fine": 0.002}
    assert "| 0.002 |" in rep.run_row("i", {"wall_s": 1.0}, row)
    assert "| 1.2 |" in rep.run_row("i", {"wall_s": 1.0}, {**row, "disk_mib_per_tok": 1.2, "disk_mib_fine": 1.234})
    single = {**doc["singles"][0], "expect": {"ssd": {"a": "0.021"}}}
    res = {k: {"n_a": 2, "mean_a": 1.0} for k in rep.REP_METRICS} | {"ssd": {"n_a": 2, "mean_a": 0.0214}}
    line = rep.single_row(single, doc["conditions"]["FX"], res, [])
    assert "| 0.021 | fx3: match |" in line


def test_label_filter_is_shown_and_explains_exclusions(fixtures, tmp_path):
    doc = json.loads((fixtures / "pairs_fx.json").read_text(encoding="utf-8"))
    doc["pairs"][0]["only_labels"] = ["fx-hA-1", "fx-hP2-9"]
    doc["pairs"][0].pop("expect")
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(doc), encoding="utf-8")
    _, out = run_report(fixtures, tmp_path, pairs=pairs)
    text = out.read_text(encoding="utf-8")
    assert ("- fx: label filter, as `stats.py --only-labels fx-hA-1,fx-hP2-9`. It leaves out `fx-hP2-1`, so "
            "`stats.py` on the two prefixes alone gives other numbers.") in text
    excluded = section(text, "## 3. Not in any ratio")
    assert ("- paper-fx/index_fx.jsonl fx-hP2-1: ok, outside the only_labels of pair fx in bench/pairs.json"
            in excluded)


def test_label_filter_that_drops_nothing_says_so(fixtures, tmp_path):
    doc = json.loads((fixtures / "pairs_fx.json").read_text(encoding="utf-8"))
    doc["pairs"][0]["only_labels"] = ["fx-hA-1", "fx-hP2-1"]
    pairs = tmp_path / "pairs.json"
    pairs.write_text(json.dumps(doc), encoding="utf-8")
    _, out = run_report(fixtures, tmp_path, pairs=pairs)
    text = out.read_text(encoding="utf-8")
    assert "It leaves out no ok run, so the two prefixes alone give the same numbers." in text
    assert "outside the only_labels" not in text
