"""Puts bench/ and scripts/capacity/ on sys.path, keeps HC_PAPER_* out of the environment, and blocks every socket."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
for path in (BENCH, REPO / "scripts" / "capacity", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIXTURES = HERE / "fixtures"
sys.dont_write_bytecode = True  # imports from scripts/ and backend/ leave no .pyc in the campaign checkout


class SocketBlocked(AssertionError):
    pass


def _blocked(*_: object, **__: object) -> None:
    raise SocketBlocked("bench tests must not open a socket")


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HC_PAPER_REPO", "HC_PAPER_HOME", "HC_PAPER_UV", "HC_PAPER_PROBE_MACS", "HYPERCOMPUTE_LLAMA_BIN_DIR"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def repo() -> Path:
    return REPO
