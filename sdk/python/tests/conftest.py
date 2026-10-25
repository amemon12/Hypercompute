"""Puts sdk/python/src and this folder on sys.path, keeps the environment clean, and blocks every socket."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for path in (SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SocketBlocked(AssertionError):
    pass


def _blocked(*_: object, **__: object) -> None:
    raise SocketBlocked("sdk tests must not open a socket; use httpx.MockTransport")


@pytest.fixture(autouse=True)
def _no_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYPERCOMPUTE_OWNER_TOKEN", raising=False)
    monkeypatch.delenv("HYPERCOMPUTE_URL", raising=False)
