"""The OS wrappers: line endings, shell syntax, PowerShell parse, and the exact .cmd template (PACKAGING-SPEC 2.2)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

JOIN = Path(__file__).resolve().parents[1]
ACTIONS = ("join", "leave", "status", "host", "invite")
POSIX = sorted([*JOIN.glob("*.sh"), *JOIN.glob("*.command"), JOIN / "lib" / "hc-common.sh"])
PS1 = sorted([*JOIN.glob("*.ps1"), JOIN / "lib" / "hc-common.ps1"])
CMD_TEMPLATE = ("@echo off\r\nsetlocal\r\n"
                '"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -ExecutionPolicy Bypass '
                '-File "%~dp0hypercompute-{action}.ps1" %*\r\n'
                "set rc=%ERRORLEVEL%\r\n"
                'if not "%HC_NO_PAUSE%"=="1" pause\r\n'
                "exit /b %rc%\r\n")


def find_bash() -> str | None:
    """Git Bash on Windows; never the WSL launcher in System32, which would run inside Linux."""
    if os.name == "nt":
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(candidate).exists():
                return candidate
        found = shutil.which("bash")
        return None if not found or "system32" in found.lower() else found
    return shutil.which("bash")


def test_every_wrapper_exists() -> None:
    for action in ACTIONS:
        for ext in (".command", ".sh", ".ps1", ".cmd"):
            assert (JOIN / f"hypercompute-{action}{ext}").is_file(), f"missing hypercompute-{action}{ext}"


@pytest.mark.parametrize("path", POSIX, ids=lambda p: p.name)
def test_posix_files_have_no_carriage_return(path: Path) -> None:
    assert b"\r" not in path.read_bytes()


@pytest.mark.parametrize("path", POSIX, ids=lambda p: p.name)
def test_bash_syntax(path: Path) -> None:
    bash = find_bash()
    if bash is None:
        pytest.skip("bash is not on PATH")
    p = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True, timeout=30, check=False)
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("action", ACTIONS)
def test_posix_wrapper_is_three_lines_plus_shebang(action: str) -> None:
    for ext in (".command", ".sh"):
        lines = (JOIN / f"hypercompute-{action}{ext}").read_text(encoding="ascii").splitlines()
        assert lines == ["#!/bin/bash", 'HC_HERE="$(cd "$(dirname "$0")" && pwd)" || exit 1',
                         '. "$HC_HERE/lib/hc-common.sh"', f'hc_main {action} "$@"']


@pytest.mark.parametrize("action", ACTIONS)
def test_cmd_matches_the_template(action: str) -> None:
    assert (JOIN / f"hypercompute-{action}.cmd").read_bytes().decode("ascii") == CMD_TEMPLATE.format(action=action)


@pytest.mark.parametrize("path", PS1, ids=lambda p: p.name)
def test_ps1_is_ascii_with_crlf(path: Path) -> None:
    data = path.read_bytes()
    data.decode("ascii")  # Windows PowerShell 5.1 reads a file without a BOM as ANSI
    assert data.count(b"\n") == data.count(b"\r\n")


@pytest.mark.parametrize("path", PS1, ids=lambda p: p.name)
def test_ps1_parses(path: Path) -> None:
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if os.name != "nt" or ps is None:
        pytest.skip("Windows PowerShell is only on Windows")
    script = ("$t = $null; $e = $null; "
              f"[void][System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$t, [ref]$e); "
              "if ($e.Count -gt 0) { $e | ForEach-Object { $_.ToString() }; exit 1 }")
    p = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
                       timeout=60, check=False)
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.parametrize("action", ACTIONS)
def test_ps1_wrapper_hands_over_to_oneclick(action: str) -> None:
    text = (JOIN / f"hypercompute-{action}.ps1").read_text(encoding="ascii")
    assert f"& $hc.Uv run --no-sync python join\\lib\\oneclick.py {action} --repo $hc.Repo @args" in text
    assert "2>&1" not in text and "&&" not in text


def test_gitattributes() -> None:
    assert (JOIN / ".gitattributes").read_text(encoding="ascii").splitlines() == [
        "*.sh text eol=lf", "*.command text eol=lf", "*.ps1 text eol=crlf", "*.cmd text eol=crlf"]


def test_bootstraps_never_fetch_an_installer() -> None:
    for path in (JOIN / "lib" / "hc-common.sh", JOIN / "lib" / "hc-common.ps1"):
        text = path.read_text(encoding="ascii").lower()
        for banned in ("curl ", "wget ", "invoke-webrequest", "irm ", "iex", "| sh", "|sh"):
            assert banned not in text, f"{path.name} contains {banned!r}"
        assert "https://docs.astral.sh/uv/getting-started/installation/" in text
        assert "sync --frozen" in text
