"""The background supervisor and the composed join command (PACKAGING-SPEC 2.4 step 12, 2.6)."""

from __future__ import annotations

import os
import sys

import pytest
from oneclick import run_join
from oneclick_core import Stop
from oneclick_supervise import substitute, supervise, supervise_main
from oneclick_testkit import CODE, FIXTURES, INVITE_LINE, PY, TICKET, make_ctx


def test_supervise_redacts_writes_exit_line_and_removes_pidfile(tmp_path) -> None:
    log, pid = tmp_path / "join.log", tmp_path / "join.pid"
    env = {**os.environ, "HC_ONECLICK_TICKET": TICKET, "HC_ONECLICK_CODE": CODE}
    script = f"print({TICKET!r}); print('code {CODE} raw {CODE.replace('-', '')}'); print('multi-level')"
    rc = supervise(log, pid, [sys.executable, "-c", script], env=env, keep_awake=False)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "<ticket>" in text and TICKET not in text
    assert CODE not in text and CODE.replace("-", "") not in text and "code <code> raw <code>" in text
    assert "multi-level" in text
    assert text.rstrip().endswith("hypercompute exited with code 0")
    assert not pid.exists()
    assert "HC_ONECLICK_TICKET" not in env and "HC_ONECLICK_CODE" not in env


def test_supervise_reports_the_child_exit_code(tmp_path) -> None:
    log, pid = tmp_path / "x.log", tmp_path / "x.pid"
    child = [sys.executable, "-c", "import sys; print('error: boom'); sys.exit(3)"]
    rc = supervise(log, pid, child, env=dict(os.environ), keep_awake=False)
    assert rc == 3
    assert log.read_text(encoding="utf-8").splitlines()[-2:] == ["error: boom", "hypercompute exited with code 3"]


def test_supervise_fills_the_placeholders_for_the_child_only(tmp_path) -> None:
    log, pid = tmp_path / "s.log", tmp_path / "s.pid"
    cmd = [sys.executable, "-c", "import sys; print(sys.argv[2] == 'T' + 'X' * 40, sys.argv[4] == 'C')",
           "--ticket", "<ticket>", "--code", "<code>"]
    rc = supervise(log, pid, cmd, env={**os.environ, "HC_ONECLICK_TICKET": "T" + "X" * 40, "HC_ONECLICK_CODE": "C"},
                   keep_awake=False)
    assert rc == 0
    assert "True True" in log.read_text(encoding="utf-8")


def test_supervise_without_the_secret_fails_closed(tmp_path) -> None:
    log, pid = tmp_path / "f.log", tmp_path / "f.pid"
    rc = supervise(log, pid, [sys.executable, "--ticket", "<ticket>"], env=dict(os.environ), keep_awake=False)
    assert rc == 127
    assert "did not reach the background process" in log.read_text(encoding="utf-8")


def test_substitute_only_after_the_flags() -> None:
    cmd = ["join", "--ticket", "<ticket>", "--code", "<code>", "--name", "<code>"]
    assert substitute(cmd, "T", "C") == ["join", "--ticket", "T", "--code", "C", "--name", "<code>"]


@pytest.mark.parametrize(("probe_rc", "wrapped"), [(0, True), (1, False)])
def test_linux_keep_awake_only_when_systemd_inhibit_works(monkeypatch, probe_rc: int, wrapped: bool) -> None:
    import oneclick_supervise as sup

    class Done:
        returncode = probe_rc

    monkeypatch.setattr(sup.sys, "platform", "linux")
    monkeypatch.setattr(sup.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sup.subprocess, "run", lambda *_a, **_k: Done())
    argv = sup._child_argv(["hc", "join"], None, None, keep_awake=True)
    assert (argv[0] == "/usr/bin/systemd-inhibit") is wrapped
    assert argv[-2:] == ["hc", "join"]


def test_mac_keep_awake_lets_the_display_sleep(monkeypatch) -> None:
    import oneclick_supervise as sup

    seen: list[list[str]] = []

    class FakePopen:
        def __init__(self, argv, **_kwargs) -> None:
            seen.append(list(argv))

    monkeypatch.setattr(sup.sys, "platform", "darwin")
    monkeypatch.setattr(sup.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sup.subprocess, "Popen", FakePopen)
    sup._start_keep_awake(4321)
    assert seen == [["/usr/bin/caffeinate", "-ims", "-w", "4321"]]
    letters = "".join(flag.lstrip("-") for flag in seen[0][1:-2])
    assert "d" not in letters and "u" not in letters


def test_linux_keep_awake_takes_no_idle_inhibitor() -> None:
    import oneclick_supervise as sup

    assert sup.INHIBIT_ARGS[0] == "--what=sleep"
    assert "idle" not in " ".join(sup.INHIBIT_ARGS)


@pytest.mark.skipif(os.name != "nt", reason="SetThreadExecutionState is Windows only")
def test_windows_keep_awake_sets_no_display_flag(monkeypatch) -> None:
    import ctypes

    import oneclick_supervise as sup

    states: list[int] = []
    monkeypatch.setattr(ctypes.windll.kernel32, "SetThreadExecutionState", states.append)
    sup._start_keep_awake(1)
    sup._stop_keep_awake(None)
    es_display_required = 0x00000002
    assert states == [sup.ES_CONTINUOUS | sup.ES_SYSTEM_REQUIRED, sup.ES_CONTINUOUS]
    assert not any(state & es_display_required for state in states)


def test_dry_run_states_keep_awake_and_the_process_list(tmp_path) -> None:
    _, out = dry_join(tmp_path)
    assert "  Keep awake:               yes, while sharing (the screen can still turn off and lock)" in out
    assert any("other accounts on this computer can see them in the process list" in line for line in out)


def test_supervise_main_needs_a_command() -> None:
    assert supervise_main(["--log", "a", "--pidfile", "b"]) == 2


def dry_join(tmp_path, fixture: str = "inventory_g14.json", **env: str) -> tuple[int, list[str]]:
    base = {"HC_INVITE": INVITE_LINE, **env}
    ctx, out = make_ctx(tmp_path, env=base, dry_run=True, inventory_file=FIXTURES / fixture)
    rc = run_join(ctx)
    assert all(TICKET not in line and CODE not in line for line in out)
    return rc, out


def command_line(out: list[str]) -> str:
    return next(line.strip() for line in out if " -m hypercompute.cli join " in line)


def test_dry_run_g14_default(tmp_path) -> None:
    rc, out = dry_join(tmp_path)
    assert rc == 0
    assert command_line(out) == (f"{PY} -m hypercompute.cli join --ticket <ticket> --code <code> --name g14 "
                                 "--devices CUDA0 --max-memory 9GiB --reserve 6GiB --cache --yes")


def test_dry_run_cache_off(tmp_path) -> None:
    _, out = dry_join(tmp_path, HC_CACHE="0")
    assert command_line(out).endswith("--reserve 6GiB --yes")


def test_dry_run_managed_on(tmp_path) -> None:
    _, out = dry_join(tmp_path, HC_MANAGED="1")
    assert command_line(out).endswith("--cache --yes --managed --devices-allowed CUDA0 --max-disk 24GiB")


def test_dry_run_managed_consent_names_restarts_threads_and_the_wait(tmp_path) -> None:
    """--managed with --yes: the CLI's own consent text only reaches the log, so the consent screen says it first."""
    _, out = dry_join(tmp_path, HC_MANAGED="1")
    cores = max(1, os.cpu_count() or 1)
    restart = (f"the host may also restart this computer's worker on CUDA0 and change how many CPU threads it uses, "
               f"from 1 up to all {cores} cores")
    unattended = ("if the host goes away, this computer keeps waiting for it for up to 2 hours, and a crashed worker "
                  "is started again (at most 3 times in 30 minutes)")
    assert f"  With host downloads on:   {restart}" in out
    assert f"  Unattended:               {unattended}" in out
    asked = next(i for i, line in enumerate(out) if "would ask: Share this computer now?" in line)
    assert out.index(f"  With host downloads on:   {restart}") < asked
    assert any(line.startswith("With host downloads on, the host may also restart") for line in out)


def test_dry_run_managed_off_keeps_restarts_off_the_consent_screen(tmp_path) -> None:
    """The question still explains the terms before the default no, but the consent screen lists no terms."""
    _, out = dry_join(tmp_path)
    assert any(line.startswith("With host downloads on, the host may also restart") for line in out)
    assert "  Host downloads:           off" in out
    assert not any(line.startswith(("  With host downloads on:", "  Unattended:")) for line in out)


def test_dry_run_managed_turns_the_cache_on(tmp_path) -> None:
    _, out = dry_join(tmp_path, HC_MANAGED="1", HC_CACHE="0")
    assert " --cache --yes --managed " in command_line(out)
    assert "Host downloads go into the cache, so the cache is on." in out


def test_dry_run_relay_url(tmp_path) -> None:
    _, out = dry_join(tmp_path, HC_RELAY_URL="https://relay.example.test")
    assert command_line(out).endswith("--cache --yes --relay-url https://relay.example.test")


def test_dry_run_mac_and_name(tmp_path) -> None:
    _, out = dry_join(tmp_path, "inventory_mac_m4_16gib.json", HC_NAME="Ana's Mac")
    assert '--name "Ana\'s Mac" --devices MTL0 --max-memory 10GiB --reserve 4GiB --cache --yes' in command_line(out)


def test_dry_run_bad_relay_stops(tmp_path) -> None:
    ctx, _ = make_ctx(tmp_path, env={"HC_INVITE": INVITE_LINE, "HC_RELAY_URL": "http://plain.example"},
                        dry_run=True, inventory_file=FIXTURES / "inventory_g14.json")
    with pytest.raises(Stop, match="https://"):
        run_join(ctx)


def test_dry_run_without_inventory_uses_placeholders(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, env={"HC_INVITE": INVITE_LINE}, dry_run=True)
    assert run_join(ctx) == 0
    expected = "--name <hostname> --devices <device> --max-memory <N>GiB --reserve <R>GiB --cache --yes"
    assert expected in command_line(out)
