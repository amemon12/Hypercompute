"""Shared fixtures for portable protocol tests. None of these touch real hardware or the network."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from factories import make_inventory  # noqa: E402
from hypercompute.services.events import EventBus  # noqa: E402
from hypercompute.storage import Store  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_home(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("hardware"):
        return None  # hardware tests read the real ~/.hypercompute (engine, downloaded models)
    monkeypatch.setenv("HYPERCOMPUTE_HOME", str(tmp_path / "hc-home"))
    return tmp_path / "hc-home"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite", tmp_path / "records")
    yield s
    s.close()


@pytest.fixture
def events(store):
    return EventBus(store)


@pytest.fixture
def fixture_text():
    def read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return read


@pytest.fixture
def now():
    return datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def local_inventory():
    return make_inventory()
