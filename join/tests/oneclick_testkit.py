"""Stubs for the one-click tests: a scripted command runner and a context factory. Synthetic secrets only."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oneclick_core import Ctx, Result

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PY = "/venv/bin/python"
# Synthetic values in the real shapes. Never paste a real ticket or code into a test.
TICKET = "endpointsynthetic" + "q7" * 24 + "zz9k"
CODE = "ABCDE-FGHJK"
INVITE_LINE = f"uv run hypercompute join --ticket {TICKET} --code {CODE}"


def inventory(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class NoRun:
    """A runner for dry runs: any call fails the test."""

    def __getattr__(self, name: str) -> Callable[..., Any]:
        raise AssertionError(f"a dry run called runner.{name}")


@dataclass
class FakeProc:
    code: int | None = None

    def poll(self) -> int | None:
        return self.code


@dataclass
class StubRunner:
    """Answers capture() by the CLI subcommand (the words after `-m hypercompute.cli`)."""

    answers: dict[str, Result] = field(default_factory=dict)
    attached_rc: int = 0
    feed_rc: int = 0
    on_background: Callable[[Sequence[str], Mapping[str, str]], FakeProc] | None = None
    calls: list[tuple[str, list[str]]] = field(default_factory=list)
    background_env: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def key(argv: Sequence[str]) -> str:
        argv = list(argv)
        if "hypercompute.cli" in argv:
            return " ".join(argv[argv.index("hypercompute.cli") + 1:])
        return " ".join(argv)

    def capture(self, argv: Sequence[str], timeout: float = 120.0) -> Result:
        k = self.key(argv)
        self.calls.append(("capture", list(argv)))
        for prefix in sorted(self.answers, key=len, reverse=True):
            if k.startswith(prefix):
                return self.answers[prefix]
        return Result(0, "")

    def attached(self, argv: Sequence[str]) -> int:
        self.calls.append(("attached", list(argv)))
        return self.attached_rc

    def feed(self, argv: Sequence[str], text: str, timeout: float = 10.0) -> int:
        self.calls.append(("feed", list(argv)))
        return self.feed_rc

    def background(self, argv: Sequence[str], env: Mapping[str, str]) -> FakeProc:
        self.calls.append(("background", list(argv)))
        self.background_env = dict(env)
        return self.on_background(argv, env) if self.on_background else FakeProc()


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_ctx(tmp_path: Path, *, runner: Any = None, env: Mapping[str, str] | None = None, dry_run: bool = False,
             answers: Sequence[str] = (), nvidia: bool = False, inventory_file: Path | None = None,
             health: Callable[[int], bool] | None = None, ticket_check: Any = None,
             secrets: Sequence[str] = ()) -> tuple[Ctx, list[str]]:
    """A context whose prompts read `answers` in order (then EOF) and whose output goes to the returned list."""
    out: list[str] = []
    replies, pasted, clock = list(answers), list(secrets), Clock()

    def read_line(prompt: str) -> str:
        out.append(prompt)
        if not replies:
            raise EOFError
        return replies.pop(0)

    def read_secret(prompt: str) -> str:
        out.append(prompt)
        if not pasted:
            raise EOFError
        return pasted.pop(0)

    ctx = Ctx(repo=tmp_path, port=8001, dry_run=dry_run, home=tmp_path / "home", runner=runner or NoRun(),
              env=dict(env or {}), python=PY, inventory_file=inventory_file, out=out.append, read_line=read_line,
              read_secret=read_secret, nvidia_ok=lambda: nvidia, health=health or (lambda _port: False),
              ticket_check=ticket_check, sleep=clock.sleep, clock=clock, open_url=lambda url: out.append(f"OPEN {url}"))
    return ctx, out
