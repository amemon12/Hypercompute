"""hcbench plan: labels, row args, flag checks, run_big's own parse_args, layout warnings, --force."""

from __future__ import annotations

import json
from pathlib import Path

import hcbench
import pytest

MAC_A = "6dad6cf75b2e4cf197b7bc7f4d0a67e4"
MAC_B = "d4e65f83c4be4a41aad722e40983b08b"


def plan(tmp_path: Path, *extra: str, name: str = "plan.json") -> tuple[int, Path]:
    out = tmp_path / name
    argv = ["plan", "--tag", "d70", "--model", "llama-3.3-70b-instruct-q4_k_m", "--host", "g14", "--peer", MAC_A,
            "--peer", MAC_B, *extra, "--out", str(out)]
    return hcbench.main(argv), out


def rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ap_labels_and_nodes(tmp_path):
    code, out = plan(tmp_path)
    assert code == 0
    got = rows(out)
    assert [r["label"] for r in got] == ["d70-g14A-1", "d70-g14P2-1"]
    assert got[0]["nodes"] == [] and got[1]["nodes"] == [MAC_A, MAC_B]
    assert [r["timeout_s"] for r in got] == [7200, 9000]
    assert got[0]["args"] == ["--prompt-tokens", "128", "--output-tokens", "64", "--warmup", "1", "--reps", "3",
                              "--load-timeout", "3600"]
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.startswith(b'[\n {\n  "label"')  # indent=1, LF


def test_apap_counts_within_each_group(tmp_path):
    _, out = plan(tmp_path, "--order", "APAP")
    assert [r["label"] for r in rows(out)] == ["d70-g14A-1", "d70-g14P2-1", "d70-g14A-2", "d70-g14P2-2"]


def test_start_continues_the_numbering(tmp_path):
    _, out = plan(tmp_path, "--order", "PA", "--start", "3")
    assert [r["label"] for r in rows(out)] == ["d70-g14P2-3", "d70-g14A-3"]


def test_concurrency_tag_and_seed_peers_on_pooled_rows_only(tmp_path):
    out = tmp_path / "n4.json"
    argv = ["plan", "--tag", "g120n4", "--model", "gpt-oss-120b-mxfp4", "--host", "g14", "--peer", MAC_A, "--peer",
            MAC_B, "--order", "AP", "--concurrency", "4", "--max-commit-growth-gib", "24", "--seed-peers",
            "--out", str(out)]
    assert hcbench.main(argv) == 0
    alone, pooled = rows(out)
    assert alone["label"] == "g120n4-g14A-1" and pooled["label"] == "g120n4-g14P2-1"
    assert "--seed-peers" not in alone["args"] and "--concurrency" in alone["args"]
    assert pooled["args"][-3:] == ["--seed-peers", "--seed-timeout", "3600"]


def test_pooled_row_matches_the_campaign_plan(tmp_path, repo):
    """Same flags as the R7.5 N = 4 row: the args list equals paper-2026-10-24-laptop/plan_g120n4_row.json."""
    campaign = repo / "results" / "capacity" / "paper-2026-10-24-laptop" / "plan_g120n4_row.json"
    if not campaign.is_file():
        pytest.skip("campaign plan not present")
    out = tmp_path / "n4.json"
    argv = ["plan", "--tag", "g120n4", "--model", "gpt-oss-120b-mxfp4", "--host", "g14", "--peer", MAC_A, "--peer",
            MAC_B, "--order", "P", "--concurrency", "4", "--max-commit-growth-gib", "24", "--seed-peers",
            "--out", str(out)]
    assert hcbench.main(argv) == 0
    assert rows(out) == json.loads(campaign.read_text(encoding="utf-8"))


@pytest.mark.parametrize("flags", [["--tag", "d-70"], ["--host", "G14"], ["--order", "APX"], ["--order", "APAPAPAPA"]])
def test_bad_names_are_rejected(tmp_path, capsys, flags):
    code, out = plan(tmp_path, *flags)
    assert code == 2 and not out.exists()
    assert "error:" in capsys.readouterr().err


def test_bad_node_id_is_rejected(tmp_path, capsys):
    out = tmp_path / "p.json"
    argv = ["plan", "--tag", "d70", "--model", "m", "--host", "g14", "--peer", "mac-a", "--out", str(out)]
    assert hcbench.main(argv) == 2
    assert "32 lowercase hex node ids" in capsys.readouterr().err


def test_pooled_rows_need_a_peer(tmp_path, capsys):
    out = tmp_path / "p.json"
    assert hcbench.main(["plan", "--tag", "d70", "--model", "m", "--host", "g14", "--out", str(out)]) == 2
    assert "no --peer" in capsys.readouterr().err
    assert hcbench.main(["plan", "--tag", "d70", "--model", "m", "--host", "g14", "--order", "A",
                         "--out", str(out)]) == 0
    assert rows(out)[0]["nodes"] == []


def test_run_big_rejects_concurrency_17(tmp_path, capsys):
    code, out = plan(tmp_path, "--concurrency", "17")
    assert code == 2 and not out.exists()
    assert "run_big.py: error: --concurrency is 1 to 16" in capsys.readouterr().err


def test_run_big_rejects_a_kv_pool_overflow(tmp_path, capsys):
    """8 x (512 + 64) = 4608 > --ctx 4096: one unified KV pool cannot hold the batch."""
    code, _ = plan(tmp_path, "--concurrency", "8", "--prompt-tokens", "512", "--output-tokens", "64")
    assert code == 2
    assert "must fit in --ctx (one unified KV pool)" in capsys.readouterr().err


def test_layout_warnings_and_path(tmp_path, capsys):
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps([{"node_id": MAC_A, "device": "RPC0", "layers": 4},
                                  {"node_id": "local", "device": "CPU", "layers": 6},
                                  {"node_id": "local", "device": "CUDA0", "layers": 20}]), encoding="utf-8")
    code, out = plan(tmp_path, "--layout", str(layout))
    assert code == 0
    err = capsys.readouterr().err
    assert "names RPC0 for peer 6dad6cf7" in err and "puts layers on the local CPU" in err
    alone, pooled = rows(out)
    assert "--layout" not in alone["args"]
    assert pooled["args"][-2:] == ["--layout", layout.resolve().as_posix()]


def test_unreadable_layout_is_rejected(tmp_path, capsys):
    code, _ = plan(tmp_path, "--layout", str(tmp_path / "nope.json"))
    assert code == 2
    assert "--layout" in capsys.readouterr().err


def test_no_capacity_mode_scopes(tmp_path):
    _, out = plan(tmp_path, "--no-capacity-mode", "pooled", name="a.json")
    alone, pooled = rows(out)
    assert "--no-capacity-mode" not in alone["args"] and "--no-capacity-mode" in pooled["args"]
    _, out = plan(tmp_path, "--no-capacity-mode", "all", name="b.json")
    assert all("--no-capacity-mode" in r["args"] for r in rows(out))


def test_overwrite_needs_force(tmp_path, capsys):
    assert plan(tmp_path)[0] == 0
    assert plan(tmp_path, "--order", "A")[0] == 2
    assert "pass --force" in capsys.readouterr().err
    code, out = plan(tmp_path, "--order", "A", "--force")
    assert code == 0 and len(rows(out)) == 1
