from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from autotune.config import Settings
from autotune.gateway import GatewayClient
from autotune.runlog import RunStore
from mock_gateway.app import PEERS, create_app

MAC_A = PEERS["mac-a"].node_id
MAC_B = PEERS["mac-b"].node_id


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(gateway="http://127.0.0.1:8010", settle_s=20, poll_s=0.2, max_checks=8, max_calls=12,
                    runs_dir=tmp_path / "runs", runner="scripted")


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


def make_client(app) -> GatewayClient:  # noqa: ANN001
    return GatewayClient("http://127.0.0.1:8010", transport=httpx.ASGITransport(app=app))


@pytest.fixture
async def mock_g14() -> AsyncIterator[tuple[object, GatewayClient]]:
    app = create_app("g14", verify_s=0, replace_delay_s=0)
    client = make_client(app)
    yield app, client
    await client.aclose()
