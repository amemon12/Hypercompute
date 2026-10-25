"""Shared parts of the one-click helper: the command runner, the context and the prompts (PACKAGING-SPEC 2.3)."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oneclick_invite import iroh_ticket_check, redact_generic, show_cmd

GIB = 1024**3
GIB_RE = re.compile(r"\s*(\d{1,5})\s*(?:gib|gb|g)?\s*", re.IGNORECASE)


class Stop(Exception):
    """Ends the action with a message and an exit code."""

    def __init__(self, message: str = "", code: int = 1) -> None:
        super().__init__(message)
        self.message, self.code = message, code


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def lines(self) -> list[str]:
        return [*self.stdout.splitlines(), *self.stderr.splitlines()]


class Runner:
    """Runs commands for real. Tests inject a stub with the same four methods."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    @staticmethod
    def _env() -> dict[str, str]:
        return {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def capture(self, argv: Sequence[str], timeout: float = 120.0) -> Result:
        try:
            p = subprocess.run(list(argv), cwd=self.cwd, env=self._env(), stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return Result(124, "", f"timed out after {timeout:.0f} s: {show_cmd(argv[:5])}")
        except OSError as exc:
            return Result(127, "", f"could not start {Path(argv[0]).name}: {exc}")
        return Result(p.returncode, p.stdout or "", p.stderr or "")

    def attached(self, argv: Sequence[str]) -> int:
        """Foreground with the terminal attached, so the CLI's own prompts work."""
        try:
            return subprocess.call(list(argv), cwd=self.cwd)
        except OSError as exc:
            print(f"could not start {Path(argv[0]).name}: {exc}")
            return 127

    def feed(self, argv: Sequence[str], text: str, timeout: float = 10.0) -> int:
        """Write text to a tool's stdin (clipboard tools), never as an argument."""
        try:
            p = subprocess.run(list(argv), input=text, text=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return 1
        return p.returncode

    def background(self, argv: Sequence[str], env: Mapping[str, str]) -> subprocess.Popen:
        kwargs: dict[str, Any] = {"cwd": self.cwd, "env": dict(env), "stdin": subprocess.DEVNULL,
                                  "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "close_fds": True}
        if os.name != "nt":
            return subprocess.Popen(list(argv), start_new_session=True, **kwargs)
        # A hidden console of its own: closing this window cannot stop it, and the child of the venv's python.exe
        # launcher inherits that console instead of opening a window.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            return subprocess.Popen(list(argv), creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:  # the job this process runs in forbids breakaway
            return subprocess.Popen(list(argv), creationflags=flags, **kwargs)


def nvidia_gpu0() -> bool:
    """`nvidia-smi -L` exits 0 within 15 s and lists `GPU 0:`."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        p = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
                           check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0 and any(line.startswith("GPU 0:") for line in p.stdout.splitlines())


def gateway_health(port: int) -> bool:
    import httpx

    try:
        return httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=3.0, trust_env=False).is_success
    except httpx.HTTPError:
        return False


def read_secret(prompt: str) -> str:
    """Read the invite without showing it. Without a terminal, read one line from stdin."""
    if sys.stdin is not None and sys.stdin.isatty():
        value = getpass.getpass(prompt + ": ")
        _drain_pending_input()
        return value
    sys.stdout.write(prompt + "\n")
    sys.stdout.flush()
    line = sys.stdin.readline() if sys.stdin is not None else ""
    if not line:
        raise EOFError
    return line


def _drain_pending_input() -> None:
    """Drop a pasted second line (the host's Expires: line) so it does not answer the next prompt."""
    time.sleep(0.1)
    with contextlib.suppress(OSError, ImportError, ValueError):  # best effort: no console to drain is fine
        if os.name == "nt":
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)


@dataclass(frozen=True)
class Ctx:
    repo: Path
    port: int
    dry_run: bool
    home: Path
    runner: Any
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    python: str = sys.executable
    inventory_file: Path | None = None
    out: Callable[[str], None] = print
    read_line: Callable[[str], str] = input
    read_secret: Callable[[str], str] = read_secret
    nvidia_ok: Callable[[], bool] = nvidia_gpu0
    health: Callable[[int], bool] = gateway_health
    ticket_check: Callable[[str], object] | None = iroh_ticket_check
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    open_url: Callable[[str], object] = webbrowser.open

    def say(self, msg: str = "") -> None:
        self.out(redact_generic(msg))

    def note(self, msg: str) -> None:
        self.say(f"[dry-run] {msg}")

    def cli(self, *args: str) -> list[str]:
        return [self.python, "-m", "hypercompute.cli", *args]

    @property
    def state_dir(self) -> Path:
        return self.home / "oneclick"

    def show_lines(self, result: Result, indent: str = "") -> None:
        for line in result.lines:
            self.say(indent + line)


def ask(ctx: Ctx, prompt: str, default: str, env_key: str | None = None) -> str:
    """Every prompt shows its default in brackets and Enter accepts it. An HC_* variable replaces the prompt."""
    if env_key and env_key in ctx.env:
        value = ctx.env[env_key].strip()
        ctx.say(f"{prompt} {value} (from {env_key})")
        return value
    try:
        answer = ctx.read_line(f"{prompt} ")
    except EOFError:
        ctx.say("")
        answer = ""
    return answer.strip() or default


def ask_yes(ctx: Ctx, prompt: str, default: bool, env_key: str | None = None) -> bool:
    for _ in range(3):
        answer = ask(ctx, prompt, "y" if default else "n", env_key).lower()
        if answer in ("y", "yes", "1"):
            return True
        if answer in ("n", "no", "0"):
            return False
        if env_key and env_key in ctx.env:
            raise Stop(f"{env_key} must be 0 or 1.")
        ctx.say("Please answer y or n.")
    return False


def parse_gib(text: str) -> int | None:
    m = GIB_RE.fullmatch(text)
    return int(m.group(1)) if m else None


def ask_gib(ctx: Ctx, prompt: str, default: int, lo: int, hi: int, env_key: str) -> int:
    for _ in range(3):
        value = parse_gib(ask(ctx, prompt, str(default), env_key))
        if value is not None and lo <= value <= hi:
            return value
        if env_key in ctx.env:
            raise Stop(f"{env_key} must be a whole number of GiB from {lo} to {hi}.")
        ctx.say(f"Type a whole number of GiB from {lo} to {hi}, or press Enter for {default}.")
    raise Stop("No valid answer. Nothing was changed.")


def clean_name(text: str) -> str:
    """Drop control and format characters (unicode category C)."""
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("C")).strip()


def ask_name(ctx: Ctx, hostname: str) -> str:
    default = clean_name(hostname)[:64] or "friend"
    for _ in range(3):
        name = clean_name(ask(ctx, f"Name shown to the host [{default}]", default, "HC_NAME"))
        if 1 <= len(name) <= 64:
            return name
        if "HC_NAME" in ctx.env:
            raise Stop("HC_NAME must be 1 to 64 characters.")
        ctx.say("Use 1 to 64 characters.")
    raise Stop("No valid name. Nothing was changed.")
