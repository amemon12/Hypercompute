"""hcbench run: --dry-run, --yes, index rules, --append, the lock, presets and the environment a stub driver sees.

No test starts the real matrix.py: a fake repo under tmp_path holds a stub scripts/capacity/paper/matrix.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import hcbench
import pytest

MAC_A = "6dad6cf75b2e4cf197b7bc7f4d0a67e4"
STUB = '''import json, os, sys
from pathlib import Path
plan, index = sys.argv[1], sys.argv[2]
keys = ("HC_PAPER_REPO", "HC_PAPER_HOME", "HC_PAPER_UV", "HC_PAPER_PROBE_MACS", "HYPERCOMPUTE_LLAMA_BIN_DIR")
seen = {"argv": sys.argv[1:], "isolated": sys.flags.isolated, "env": {k: os.environ.get(k) for k in keys},
        "lock": Path(index + ".lock").exists(), "cwd": os.getcwd()}
Path(index).with_name("stub_seen.json").write_text(json.dumps(seen), encoding="utf-8")
with open(index, "a", encoding="utf-8") as f:
    f.write(json.dumps({"label": "stub-1"}) + "\\n")
sys.exit(3)
'''


@pytest.fixture
def plan_file(tmp_path: Path) -> Path:
    rows = [{"label": "d70-g14A-1", "model": "llama-3.3-70b-instruct-q4_k_m", "nodes": [],
             "args": ["--prompt-tokens", "64", "--output-tokens", "32"], "timeout_s": 7200},
            {"label": "d70-g14P1-1", "model": "llama-3.3-70b-instruct-q4_k_m", "nodes": [MAC_A],
             "args": ["--prompt-tokens", "64", "--output-tokens", "32", "--seed-peers"], "timeout_s": 9000}]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.fixture
def stub_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    matrix = repo / "scripts" / "capacity" / "paper" / "matrix.py"
    matrix.parent.mkdir(parents=True)
    matrix.write_text(STUB, encoding="utf-8")
    return repo


def test_dry_run_prints_each_rows_run_big_argv(plan_file, tmp_path, capsys):
    index = tmp_path / "paper-x" / "index_d70.jsonl"
    code = hcbench.main(["run", str(plan_file), "--index", str(index), "--dry-run", "--home", "H", "--uv", "UV"])
    assert code == 0
    out = capsys.readouterr().out
    assert "127.0.0.1:8001" in out and "retried once after 60 s" in out
    assert "d70-g14A-1: UV run --no-sync python scripts/capacity/run_big.py --model llama-3.3-70b-instruct-q4_k_m " \
           "--min-available-gib 0 --home H --strategy local_only --prompt-tokens 64 --output-tokens 32" in out
    assert f"d70-g14P1-1: UV run --no-sync python scripts/capacity/run_big.py --model llama-3.3-70b-instruct-q4_k_m " \
           f"--min-available-gib 0 --home H --node {MAC_A} --prompt-tokens 64 --output-tokens 32 --seed-peers" in out
    assert "matrix.py" in out.split("driver: ", 1)[1].splitlines()[0]
    assert not index.parent.exists()  # a dry run writes nothing


def test_real_run_needs_yes(plan_file, tmp_path, stub_repo, capsys):
    index = tmp_path / "index_d70.jsonl"
    assert hcbench.main(["run", str(plan_file), "--index", str(index), "--repo", str(stub_repo)]) == 2
    assert "pass --yes" in capsys.readouterr().err
    assert not index.exists() and not (tmp_path / "stub_seen.json").exists()


def test_stub_driver_gets_plan_index_and_environment(plan_file, tmp_path, stub_repo):
    index = tmp_path / "paper-x" / "index_d70.jsonl"
    code = hcbench.main(["run", str(plan_file), "--index", str(index), "--repo", str(stub_repo), "--home", "HOME1",
                         "--uv", "UV1", "--no-probe-macs", "--llama-bin-dir", "BIN1", "--yes"])
    assert code == 3  # matrix's exit code
    seen = json.loads((index.parent / "stub_seen.json").read_text(encoding="utf-8"))
    assert seen["argv"] == [str(plan_file.resolve()), str(index.resolve())]
    assert seen["isolated"] == 1 and seen["lock"] is True
    assert Path(seen["cwd"]).resolve() == stub_repo.resolve()
    assert seen["env"] == {"HC_PAPER_REPO": str(stub_repo), "HC_PAPER_HOME": "HOME1", "HC_PAPER_UV": "UV1",
                           "HC_PAPER_PROBE_MACS": "0", "HYPERCOMPUTE_LLAMA_BIN_DIR": "BIN1"}
    assert not Path(str(index) + ".lock").exists()
    assert index.read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parametrize("name", ["index-d70.jsonl", "index_D70.jsonl", "results.jsonl", "index_d70.json",
                                  "index_d70_refused.jsonl", "index_d70_aborted_guard.jsonl", "index_x_failed.jsonl"])
def test_index_name_rules(plan_file, tmp_path, capsys, name):
    assert hcbench.main(["run", str(plan_file), "--index", str(tmp_path / name), "--dry-run"]) == 2
    assert "error:" in capsys.readouterr().err


def test_append_rule(plan_file, tmp_path, stub_repo, capsys):
    index = tmp_path / "index_d70.jsonl"
    index.write_text('{"label": "old"}\n', encoding="utf-8")
    assert hcbench.main(["run", str(plan_file), "--index", str(index), "--repo", str(stub_repo), "--yes"]) == 2
    assert "pass --append" in capsys.readouterr().err
    assert hcbench.main(["run", str(plan_file), "--index", str(index), "--repo", str(stub_repo), "--yes",
                         "--append"]) == 3
    assert index.read_text(encoding="utf-8").count("\n") == 2
    empty = tmp_path / "index_empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert hcbench.main(["run", str(plan_file), "--index", str(empty), "--repo", str(stub_repo), "--yes"]) == 3


def test_held_lock_refuses_and_is_left_alone(plan_file, tmp_path, stub_repo, capsys):
    index = tmp_path / "index_d70.jsonl"
    lock = Path(str(index) + ".lock")
    lock.write_text("1234\n", encoding="utf-8")
    assert hcbench.main(["run", str(plan_file), "--index", str(index), "--repo", str(stub_repo), "--yes"]) == 2
    assert "another run holds this index" in capsys.readouterr().err
    assert lock.read_text(encoding="utf-8") == "1234\n" and not index.exists()


def test_bad_plan_is_refused(tmp_path, capsys):
    bad = tmp_path / "plan.json"
    bad.write_text(json.dumps([{"label": "x", "model": "m", "nodes": []}]), encoding="utf-8")
    assert hcbench.main(["run", str(bad), "--index", str(tmp_path / "index_x.jsonl"), "--dry-run"]) == 2
    assert "lacks or mistypes args, timeout_s" in capsys.readouterr().err


def test_gx10_preset_is_what_matrix_sh_exports(repo):
    """Parse the export lines of scripts/capacity/paper/gx10/matrix.sh with HOME=/home/asus and PATH=/usr/bin."""
    text = (repo / "scripts" / "capacity" / "paper" / "gx10" / "matrix.sh").read_text(encoding="utf-8")
    want: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("export "):
            for key, value in re.findall(r"(\w+)=(\S+)", line[len("export "):]):
                want[key] = value.replace("$HOME", "/home/asus").replace("$PATH", "/usr/bin")
    assert hcbench.preset_env("gx10", "/home/asus", "/usr/bin") == want


def test_g14_preset_is_matrix_defaults(repo):
    text = (repo / "scripts" / "capacity" / "paper" / "matrix.py").read_text(encoding="utf-8")
    env = hcbench.preset_env("g14", r"C:\Users\operator", "")
    for key in ("HC_PAPER_REPO", "HC_PAPER_HOME", "HC_PAPER_UV"):
        default = re.search(rf'os\.environ\.get\("{key}", r"([^"]+)"\)', text)
        assert default is not None and env[key] == default.group(1)


def test_flags_override_the_preset():
    opts = hcbench.build_parser().parse_args(["run", "p.json", "--index", "index_x.jsonl", "--preset", "gx10",
                                              "--home", "/data/hc"])
    env = hcbench.run_env(opts, {"PATH": "/usr/bin"})
    assert env["HC_PAPER_HOME"] == "/data/hc" and env["HC_PAPER_PROBE_MACS"] == "0"


def test_ctrl_c_waits_for_the_driver():
    class Proc:
        calls = 0

        def wait(self) -> int:
            Proc.calls += 1
            if Proc.calls == 1:
                raise KeyboardInterrupt
            return 130

    assert hcbench.wait_for(Proc()) == 130 and Proc.calls == 2
