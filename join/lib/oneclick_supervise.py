"""The detached supervisor of PACKAGING-SPEC 2.6: run hypercompute, redact its output into a log, keep awake.

Started by oneclick.py as `[python, oneclick.py, "_supervise", "--log", LOG, "--pidfile", PID, "--", CMD...]`.
The ticket and code arrive in HC_ONECLICK_TICKET and HC_ONECLICK_CODE. They are removed from this process's
environment before the child starts, and reach the child only as the arguments after --ticket and --code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import MutableMapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from oneclick_invite import CODE_TOKEN, TICKET_TOKEN, make_redactor

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
# Idle, disk and system sleep only. No -d (display) and no -u (user active), so the screen can lock.
CAFFEINATE_FLAGS = ("-ims",)


def substitute(cmd: Sequence[str], ticket: str | None, code: str | None) -> list[str]:
    """Fill the placeholders that follow --ticket and --code, and nothing else."""
    out: list[str] = []
    for i, arg in enumerate(cmd):
        prev = cmd[i - 1] if i else ""
        if prev == "--ticket" and arg == TICKET_TOKEN:
            if not ticket:
                raise ValueError("the invite ticket did not reach the background process")
            arg = ticket
        elif prev == "--code" and arg == CODE_TOKEN:
            if not code:
                raise ValueError("the invite code did not reach the background process")
            arg = code
        out.append(arg)
    return out


def _start_keep_awake(child_pid: int) -> subprocess.Popen | None:
    """macOS: caffeinate while the child lives. Windows: this thread's execution state, for this process only.

    Neither keeps the display on, so the screen can still turn off and lock while the computer shares.
    """
    if sys.platform == "darwin":
        caffeinate = shutil.which("caffeinate")
        if caffeinate:
            return subprocess.Popen([caffeinate, *CAFFEINATE_FLAGS, "-w", str(child_pid)], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    return None


def _stop_keep_awake(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _write_pidfile(pidfile: Path, child_pid: int) -> None:
    tmp = pidfile.with_name(pidfile.name + ".tmp")
    tmp.write_text(json.dumps({"supervisor_pid": os.getpid(), "child_pid": child_pid,
                               "started_at": datetime.now(UTC).isoformat()}), encoding="utf-8")
    os.replace(tmp, pidfile)


# Sleep only. An idle inhibitor would also stop some desktops from blanking and locking the screen.
INHIBIT_ARGS = ("--what=sleep", "--who=hypercompute", "--why=sharing", "--mode=block")


def _inhibitor() -> list[str]:
    """systemd-inhibit, only when it works here: without polkit rights (an SSH session) it exits before the child."""
    inhibit, true = shutil.which("systemd-inhibit"), shutil.which("true")
    if not inhibit or not true:
        return []
    try:
        probe = subprocess.run([inhibit, *INHIBIT_ARGS, true], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [inhibit, *INHIBIT_ARGS] if probe.returncode == 0 else []


def _child_argv(cmd: Sequence[str], ticket: str | None, code: str | None, keep_awake: bool) -> list[str]:
    argv = substitute(cmd, ticket, code)
    if keep_awake and sys.platform.startswith("linux"):
        argv = [*_inhibitor(), *argv]
    return argv


def supervise(log: Path, pidfile: Path, cmd: Sequence[str], env: MutableMapping[str, str] | None = None,
              keep_awake: bool = True) -> int:
    """Run the child, write its merged output to the log line by line (redacted, flushed), return its exit code."""
    env = os.environ if env is None else env
    ticket, code = env.pop("HC_ONECLICK_TICKET", None), env.pop("HC_ONECLICK_CODE", None)
    redact = make_redactor(ticket, [code, code.replace("-", "")] if code else [])
    rc = 1
    with open(log, "a", encoding="utf-8", newline="\n") as fh:
        try:
            child = subprocess.Popen(_child_argv(cmd, ticket, code, keep_awake), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                     errors="replace",
                                     env={**env, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, ValueError) as exc:
            fh.write(redact(f"could not start hypercompute: {exc}") + "\n")
            fh.write("hypercompute exited with code 127\n")
            return 127
        _write_pidfile(pidfile, child.pid)
        awake = _start_keep_awake(child.pid) if keep_awake else None
        try:
            assert child.stdout is not None
            for line in child.stdout:
                fh.write(redact(line.rstrip("\r\n")) + "\n")
                fh.flush()
            rc = child.wait()
            fh.write(f"hypercompute exited with code {rc}\n")
        finally:
            if keep_awake:
                _stop_keep_awake(awake)
            pidfile.unlink(missing_ok=True)
    return rc


def supervise_main(argv: list[str]) -> int:
    if "--" not in argv:
        print("usage: oneclick.py _supervise --log LOG --pidfile PID -- COMMAND...")
        return 2
    i = argv.index("--")
    p = argparse.ArgumentParser(prog="oneclick.py _supervise")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--pidfile", type=Path, required=True)
    args = p.parse_args(argv[:i])
    return supervise(args.log, args.pidfile, argv[i + 1:]) if argv[i + 1:] else 2
