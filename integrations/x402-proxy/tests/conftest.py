"""Shared test setup: the mock gateway, a stub facilitator, and a real x402 buyer with a throwaway in-memory key.

The buyer key is generated per test run and never funded; the stub facilitator never touches a chain. Nothing in
these tests moves any asset, test or real.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from eth_account import Account
from x402 import x402Client
from x402.http.clients.httpx import x402AsyncTransport
from x402.mechanisms.evm.exact.client import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner
from x402.schemas import SettleResponse, SupportedKind, SupportedResponse, VerifyResponse

from mock_gateway import create_mock_app
from x402proxy.app import create_app
from x402proxy.config import BASE_SEPOLIA, load

PAY_TO = "0x" + "ab" * 20


@dataclass
class StubFacilitator:
    """Checks that the signed authorization pays the right address the quoted amount; settles without a chain."""

    fail_settle: bool = False
    verified: list = field(default_factory=list)
    settled: list = field(default_factory=list)
    settle_calls: int = 0

    def get_supported(self) -> SupportedResponse:
        return SupportedResponse(kinds=[SupportedKind(x402_version=2, scheme="exact", network=BASE_SEPOLIA)])

    async def verify(self, payload, requirements) -> VerifyResponse:
        auth = payload.payload["authorization"]
        ok = auth["value"] == requirements.amount and auth["to"].lower() == requirements.pay_to.lower()
        self.verified.append((auth["value"], requirements.amount))
        return VerifyResponse(is_valid=ok, invalid_reason=None if ok else "amount_or_payee_mismatch",
                              payer=auth["from"])

    async def settle(self, payload, requirements) -> SettleResponse:
        self.settle_calls += 1
        payer = payload.payload["authorization"]["from"]
        if self.fail_settle:
            return SettleResponse(success=False, error_reason="insufficient_funds", transaction="",
                                  network=requirements.network, payer=payer)
        tx = "0x" + secrets.token_hex(32)
        self.settled.append((tx, requirements.amount))
        return SettleResponse(success=True, transaction=tx, network=requirements.network, payer=payer,
                              amount=requirements.amount)


@dataclass
class Rig:
    proxy: object
    mock: object
    facilitator: StubFacilitator
    config: object

    def plain(self) -> httpx.AsyncClient:
        """A client that does not pay."""
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.proxy), base_url="http://proxy")

    def paying(self, account=None) -> httpx.AsyncClient:
        """A real x402 buyer: signs EIP-3009 authorizations with a fresh (or the given) unfunded key."""
        client = x402Client()
        client.register(BASE_SEPOLIA, ExactEvmScheme(EthAccountSigner(account or Account.create())))
        transport = x402AsyncTransport(client, transport=httpx.ASGITransport(app=self.proxy))
        return httpx.AsyncClient(transport=transport, base_url="http://proxy")

    async def mode(self, mode: str) -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.mock), base_url="http://mock") as c:
            (await c.post("/mock/mode", json={"mode": mode})).raise_for_status()


def make_rig(tmp_path: Path, **env_overrides: str) -> Rig:
    env = {"X402_PAY_TO": PAY_TO, "X402_LEDGER_PATH": str(tmp_path / "ledger.sqlite3"),
           "HYPERCOMPUTE_GATEWAY": "http://mock", "X402_PAYEES": "mac-a=0x" + "cd" * 20, **env_overrides}
    config = load(env)
    mock = create_mock_app(token_delay_s=0.0)
    facilitator = StubFacilitator()
    proxy = create_app(config, env, facilitator=facilitator, gateway_transport=httpx.ASGITransport(app=mock))
    return Rig(proxy=proxy, mock=mock, facilitator=facilitator, config=config)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)
