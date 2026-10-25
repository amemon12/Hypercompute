"""join/tests: put join/lib on sys.path and make every socket connection fail."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

JOIN = Path(__file__).resolve().parents[1]
for path in (JOIN / "lib", JOIN / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("join tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
