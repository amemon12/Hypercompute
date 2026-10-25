"""scripts/capacity/paper/summarize.py and stats.py with --concurrency rows: an N > 1 run's batches never pool into
the N = 1 groups (missing requested.concurrency is 1), N = 1 reps become batches of one from their client timestamps,
and stats --batches resamples batches, not the requests inside them. Fake run.json files only."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "scripts" / "capacity" / "paper"


def _load(name: str) -> Any:
    key = f"paper_{name}_script"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, PAPER / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


summarize, stats = _load("summarize"), _load("stats")


def rep(label: str, decode: float, *, t0: float = 0.0, disk: float | None = 2.0 * 2**20) -> dict[str, Any]:
    """An N=1 rep row: 64 tokens, first token after 2 s, 63 steps over 6.3 s of client time."""
    return {"label": label, "ok": True, "decode_tok_s": decode, "prefill_tok_s": 50.0, "predicted_n": 64,
            "ttft_ms": 2000.0, "t_start": t0, "t_end": t0 + 8.3, "decode_disk_read_bytes_per_token": disk}


def batch(label: str, agg: float, status: str = "ok") -> dict[str, Any]:
    return {"label": label, "status": status, "aggregate_decode_tok_s": agg, "aggregate_output_tok_s": agg * 0.8,
            "step_ms": 8000.0 / agg, "first_token_skew_s": 0.4, "ssd_bytes_per_step": 3.0 * 2**20,
            "ssd_bytes_per_aggregate_token": 2**20}


def write_run(results: Path, name: str, *, concurrency: int | None, reps: list[dict], batches: list[dict] | None = None,
              status: str = "ok") -> str:
    requested = {"model": "big"} if concurrency is None else {"model": "big", "concurrency": concurrency}
    chats: dict[str, Any] = {"reps": reps}
    if batches is not None:
        chats = {**chats, "batches": batches}
    doc = {"status": status, "requested": requested, "measured": {"load": {"seconds": 300.0}, "chats": chats}}
    (results / name).mkdir(parents=True)
    (results / name / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    return name


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    results = tmp_path / "results"
    monkeypatch.setattr(summarize, "RESULTS", results)
    monkeypatch.setattr(stats, "RESULTS", results)
    old = write_run(results, "old", concurrency=None, reps=[rep("warmup0", 9.0), rep("rep0", 1.0), rep("rep1", 1.2)])
    one = write_run(results, "one", concurrency=1, reps=[rep("rep0", 1.4), rep("rep1", 1.6), rep("rep2", 1.5)])
    n8_reps = [{**rep(f"b{k}r{j}", 0.9), "decode_disk_read_bytes_per_token": None} for k in range(3) for j in range(8)]
    n8_reps = [*n8_reps, rep("rep0", 0.9)]  # a rep-labelled row of an N=8 run still stays out of N=1 groups
    n8 = write_run(results, "n8", concurrency=8, reps=n8_reps,
                   batches=[batch("b0", 6.0), batch("b1", 6.4), batch("b2", 2.0, "not_concurrent")])
    lines = [{"label": label, "model": "671b", "nodes": [], "exit": 0, "wall_s": 900, "result_dir": d}
             for label, d in (("671b-gxA", old), ("671b-gxA", one), ("671b-gxA-n8", n8))]
    path = tmp_path / "INDEX.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def _recs(index: Path) -> list[dict]:
    return [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x]


def test_summarize_keeps_n8_batches_out_of_the_rep_columns(index: Path) -> None:
    rows = [summarize.row(r) for r in _recs(index)]
    old, one, n8 = rows
    assert old["concurrency"] == 1 and old["decode_tok_s"] == [1.0, 1.2]  # no concurrency key: an N=1 row
    assert one["reps"] == 3 and n8["concurrency"] == 8 and n8["reps"] == 0 and n8["decode_med"] is None
    assert n8["batch"]["batches"] == 3 and n8["batch"]["ok"] == 2 and n8["batch"]["not_concurrent"] == 1
    assert n8["batch"]["aggregate_decode_tok_s"]["median"] == pytest.approx(6.2)


def test_summarize_batch_of_one_matches_the_rep_fields(index: Path) -> None:
    r = rep("rep0", 1.4)
    one = summarize.batch_of_one(r)
    assert one["status"] == "ok" and one["requests"] == ["rep0"]
    assert one["ssd_bytes_per_step"] == pytest.approx(r["decode_disk_read_bytes_per_token"])
    assert one["aggregate_decode_tok_s"] == pytest.approx(63 / 6.3)  # client clock: first token to the end
    assert one["step_ms"] == pytest.approx(100.0) and one["first_token_skew_s"] == 0.0
    assert summarize.batch_of_one({**r, "ttft_ms": None})["status"] == "incomplete"
    rows = [summarize.row(x) for x in _recs(index)]
    assert rows[1]["batch"]["batches"] == 3
    assert rows[1]["batch"]["aggregate_decode_tok_s"]["median"] == pytest.approx(10.0)


def test_summarize_writes_a_batch_table(index: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out.md"
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(out), str(index)])
    assert summarize.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "Batches" in text and "| 671b-gxA-n8 | 671b | 8 | ok | 3 (2 ok, 1 not concurrent, 0 skewed) |" in text


def test_stats_reps_ignore_n8_rows_whose_label_shares_the_prefix(index: Path) -> None:
    reps = stats.reps_for("671b-gxA", _recs(index), None)
    assert reps["decode_tok_s"] == [1.0, 1.2, 1.4, 1.6, 1.5]  # the N=1 reps only, warmups out


def test_stats_batches_resamples_batches_not_requests(index: Path) -> None:
    recs = _recs(index)
    n8 = stats.batches_for("671b-gxA-n8", recs, None)
    assert n8["aggregate_decode_tok_s"] == [6.0, 6.4]  # two concurrent batches of 8, not 24 requests
    one = stats.batches_for("671b-gxA", recs, {"671b-gxA"})
    with pytest.raises(ValueError, match="concurrency"):  # the prefix catches the n8 row too
        stats.batches_for("671b-gxA", recs, None)
    assert len(one["aggregate_decode_tok_s"]) == 5  # every N=1 rep is a batch of one
    # the bootstrap draws len(batches) values per resample from the batches themselves
    lo, hi = stats.boot_ratio(one["aggregate_decode_tok_s"], n8["aggregate_decode_tok_s"])
    assert 6.0 / 10.0 <= lo <= hi <= 6.4 / 10.0 + 1e-9


def test_stats_main_batches_mode(index: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    # labels named so no prefix catches another (671b-n8-gxA, not 671b-gxA-n8)
    recs = [{**r, "label": "671b-n8-gxA"} if r["result_dir"] == "n8" else r for r in _recs(index)]
    renamed = tmp_path / "RENAMED.jsonl"
    renamed.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["stats.py", "--batches", "671b-gxA", "671b-n8-gxA", str(renamed)])
    assert stats.main() == 0
    out = capsys.readouterr().out
    assert "aggregate_decode_tok_s: 671b-gxA n=5" in out and "671b-n8-gxA n=2" in out and "95% CI" in out


def test_stats_batches_mode_reports_missed_first_iteration_and_coverage(index: Path, capsys: pytest.CaptureFixture[str],
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stats.py", "--batches", "671b-gxA", "671b-gxA-n8", str(index),
                                      "--only-labels", "671b-gxA"])
    assert stats.main() == 0
    out = capsys.readouterr().out
    assert "missed_first_iteration: 671b-gxA n=5 median 0.000" in out
    assert "window_coverage: 671b-gxA n=5 median 1.000" in out and "| 671b-gxA-n8 n=0" in out


def test_standalone_copies_run_with_batchlib_beside_them(tmp_path: Path) -> None:
    """The ops scripts copy summarize.py, stats.py and batchlib.py into one directory and run them with python -I;
    the rep row is picked by its label, since the batch table now comes last."""
    copy = tmp_path / "standalone"
    copy.mkdir()
    for name in ("paper/summarize.py", "paper/stats.py", "batchlib.py"):
        shutil.copy(ROOT / "scripts" / "capacity" / name, copy)
    results = tmp_path / "repo" / "results" / "capacity"
    one = write_run(results, "one", concurrency=None, reps=[rep("rep0", 1.4), rep("rep1", 1.6)])
    idx = tmp_path / "INDEX.jsonl"
    idx.write_text(json.dumps({"label": "671b-local-1", "model": "671b", "nodes": [], "exit": 0, "wall_s": 900,
                               "result_dir": one}) + "\n", encoding="utf-8")
    env = {**os.environ, "HC_PAPER_REPO": str(tmp_path / "repo")}
    out = subprocess.run([sys.executable, "-I", str(copy / "summarize.py"), str(tmp_path / "out.md"), str(idx)],
                         capture_output=True, text=True, env=env, timeout=60, check=True).stdout
    rows = [line for line in out.splitlines() if line.startswith("| 671b-local-1 ")]
    assert len(rows) == 2 and "[1.4, 1.6]" in rows[0]  # rep table first, then the batch table
    stat = subprocess.run([sys.executable, "-I", str(copy / "stats.py"), "671b", "671b-local", str(idx)],
                          capture_output=True, text=True, env=env, timeout=60, check=True).stdout
    assert "decode_tok_s: 671b n=2" in stat
