"""An invite typed after the command is refused and never echoed (PACKAGING-SPEC 1 and 2.6).

Covers oneclick.py's own arguments and the two bootstraps. The bootstrap tests point HC_REPO at a missing folder, so
even a broken check stops at "not inside a Hypercompute checkout" and never reaches uv or hypercompute.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import oneclick
import pytest
from oneclick_testkit import CODE, INVITE_LINE, TICKET
from test_join_wrappers import JOIN, find_bash

RAW_CODE = CODE.replace("-", "")
REFUSED = "Run this with nothing after it"
NOT_CHECKOUT = "This folder is not inside a Hypercompute checkout"


def no_secret(text: str) -> bool:
    return TICKET not in text and CODE not in text and RAW_CODE not in text


@pytest.mark.parametrize("extra", [
    INVITE_LINE.split(),
    [TICKET, CODE],
    [CODE, TICKET],
    ["--dry-run", *INVITE_LINE.split()],
    [f"--ticket={TICKET}", f"--code={CODE}"],
], ids=["full-line", "bare", "bare-reversed", "dry-run", "equals"])
def test_invite_after_the_command_is_refused_unechoed(capsys, tmp_path, extra: list[str]) -> None:
    rc = oneclick.main(["join", "--repo", str(tmp_path), *extra])
    captured = capsys.readouterr()
    assert rc == 2
    assert REFUSED in captured.err
    assert no_secret(captured.out + captured.err)


@pytest.mark.parametrize("argv", [
    ["join", "--port", TICKET],
    ["join", "--port", CODE],
    [TICKET],
], ids=["port-ticket", "port-code", "action-ticket"])
def test_argparse_errors_are_redacted(capsys, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        oneclick.main(argv)
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "oneclick.py: error:" in captured.err
    assert no_secret(captured.out + captured.err)


def run_bash_bootstrap(tmp_path, *args: str) -> subprocess.CompletedProcess[str]:
    bash = find_bash()
    if bash is None:
        pytest.skip("bash is not on PATH")
    script = 'HC_HERE="$1"; shift; . "$HC_HERE/lib/hc-common.sh"; hc_run "$@"'
    env = {**os.environ, "HC_REPO": str(tmp_path / "missing"), "HC_NO_PAUSE": "1"}
    return subprocess.run([bash, "-c", script, "bash", JOIN.as_posix(), "join", *args], capture_output=True,
                          text=True, timeout=60, check=False, env=env)


@pytest.mark.parametrize("extra", [INVITE_LINE.split(), [TICKET, CODE], [CODE], ["--dry-run", "--code", CODE]],
                         ids=["full-line", "bare", "code-only", "dry-run"])
def test_bash_bootstrap_refuses_before_uv(tmp_path, extra: list[str]) -> None:
    p = run_bash_bootstrap(tmp_path, *extra)
    assert p.returncode == 2, p.stdout + p.stderr
    assert REFUSED in p.stdout
    assert NOT_CHECKOUT not in p.stdout
    assert no_secret(p.stdout + p.stderr)


def test_bash_bootstrap_lets_normal_arguments_through(tmp_path) -> None:
    long_path = (tmp_path / ("x" * 36) / "inventory.json").as_posix()
    p = run_bash_bootstrap(tmp_path, "--dry-run", "--inventory-file", long_path, "--port", "8001")
    assert p.returncode == 1
    assert NOT_CHECKOUT in p.stdout and REFUSED not in p.stdout


def run_ps_bootstrap(tmp_path, *args: str) -> subprocess.CompletedProcess[str]:
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if os.name != "nt" or ps is None:
        pytest.skip("Windows PowerShell is only on Windows")
    quoted = ", ".join("'" + a.replace("'", "''") + "'" for a in args)
    script = (f". '{JOIN / 'lib' / 'hc-common.ps1'}'; "
              f"$r = Initialize-HcBootstrap 'join' '{JOIN}' @({quoted}); "
              "Write-Host ('result code=' + $r.Code + ' ok=' + $r.Ok)")
    env = {**os.environ, "HC_REPO": str(tmp_path / "missing")}
    return subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
                          timeout=60, check=False, env=env)


@pytest.mark.parametrize("extra", [INVITE_LINE.split(), [TICKET, CODE], [f"--ticket={TICKET}"]],
                         ids=["full-line", "bare", "equals"])
def test_ps_bootstrap_refuses_before_uv(tmp_path, extra: list[str]) -> None:
    p = run_ps_bootstrap(tmp_path, *extra)
    assert "result code=2 ok=False" in p.stdout, p.stdout + p.stderr
    assert REFUSED in p.stdout
    assert NOT_CHECKOUT not in p.stdout
    assert no_secret(p.stdout + p.stderr)


def test_ps_bootstrap_lets_normal_arguments_through(tmp_path) -> None:
    long_path = str(tmp_path / ("x" * 36) / "inventory.json")
    p = run_ps_bootstrap(tmp_path, "--dry-run", "--inventory-file", long_path)
    assert "result code=1 ok=False" in p.stdout, p.stdout + p.stderr
    assert NOT_CHECKOUT in p.stdout and REFUSED not in p.stdout
