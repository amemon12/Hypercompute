from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from hc_agent.config import Settings
from hc_agent.gateway import Gateway
from hc_agent.workflow import Workflow
from mock_gateway.server import MockState, create_app

DATA = Path(__file__).parent / "data"
MEMBER = "agent1q" + "m" * 58
GUEST = "agent1q" + "g" * 58


def make_settings(tmp_path: Path, **env: str) -> Settings:
    base = {"HC_AGENT_SEED": "x" * 40, "HC_ALLOWED_SENDERS": MEMBER, "HC_GATEWAY_URL": "http://127.0.0.1:8765",
            "HC_DATA_DIR": str(tmp_path / "agent_data"), "HC_PROGRESS_EVERY_S": "5"}
    return Settings.from_env({**base, **env})


@pytest.fixture
def recorded() -> dict:
    return json.loads((DATA / "kimi_pooled.json").read_text(encoding="utf-8"))


@pytest.fixture
def mock_state() -> MockState:
    return MockState(load_s=0.05, chat_delay_s=0.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
async def gateway(mock_state: MockState) -> AsyncIterator[Gateway]:
    gw = Gateway("http://127.0.0.1:8765", transport=httpx.ASGITransport(app=create_app(mock_state)))
    yield gw
    await gw.aclose()


@pytest.fixture
def workflow(gateway: Gateway, settings: Settings) -> Workflow:
    return Workflow(gateway, replace(settings, load_timeout_s=30), poll_s=0.01)


class Recorder:
    """An Outbox that keeps everything the agent would have sent."""

    def __init__(self) -> None:
        self.replies: list = []
        self.payments: list = []
        self.completed: list[str] = []
        self.cancelled: list[tuple[str | None, str]] = []

    undeliverable_prefix: str | None = None  # replies starting with this "fail" to deliver

    async def reply(self, to: str, reply) -> bool:
        self.replies.append(reply)
        return not (self.undeliverable_prefix and reply.text.startswith(self.undeliverable_prefix))

    async def request_payment(self, to: str, ask) -> bool:
        self.payments.append(ask)
        return self.deliver_payments

    deliver_payments = True

    async def complete_payment(self, to: str, tx_id: str) -> None:
        self.completed.append(tx_id)

    async def cancel_payment(self, to: str, tx_id: str | None, reason: str) -> None:
        self.cancelled.append((tx_id, reason))

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.replies]

    @property
    def last(self):
        return self.replies[-1]


@pytest.fixture
def out() -> Recorder:
    return Recorder()
