"""The MCP tool an agent calls: real MCP client in process, real x402 client, proxy rig with the stub facilitator."""

from __future__ import annotations

import json

import httpx
from eth_account import Account
from mcp import Client
from x402.mechanisms.evm.signers import EthAccountSigner

from conftest import make_rig
from pool_mcp import Budget, build_server
from x402proxy.wallet import Wallet


async def _noop() -> None:
    return None


def server_for(rig, per_call: int = 50_000, total: int = 250_000):
    async def open_signer() -> Wallet:
        account = Account.create()  # fresh, unfunded, in memory only
        return Wallet(EthAccountSigner(account), account.address, "test key", _noop)

    budget = Budget(per_call_atomic=per_call, total_atomic=total)
    return build_server("http://proxy", open_signer, budget, transport=httpx.ASGITransport(app=rig.proxy)), budget


async def call(server, tool: str, args: dict) -> dict:
    async with Client(server) as client:
        result = await client.call_tool(tool, args)
    return json.loads(result.content[0].text)


async def test_agent_pays_and_gets_an_answer_with_receipt(tmp_path):
    rig = make_rig(tmp_path)
    server, budget = server_for(rig)
    out = await call(server, "ask_pool", {"prompt": "Explain MoE routing", "max_tokens": 48})
    assert out["answer"].startswith("Mixture-of-experts") and out["testnet"] is True
    assert out["paid_usdc"] == "0.001960" and out["tx"].startswith("0x")
    assert out["explorer"].startswith("https://sepolia.basescan.org/tx/0x")
    assert [s["machine"] for s in out["split"]][0] == "gx10"
    assert budget.spent_atomic == 1960 and len(rig.facilitator.settled) == 1


async def test_quote_is_free(tmp_path):
    rig = make_rig(tmp_path)
    server, budget = server_for(rig)
    out = await call(server, "pool_quote", {"max_tokens": 256})
    assert out["price_usdc"] == "0.006120" and budget.spent_atomic == 0 and rig.facilitator.verified == []


async def test_caps_and_budget_refuse_before_signing(tmp_path):
    rig = make_rig(tmp_path)
    server, budget = server_for(rig, per_call=50_000, total=3_000)
    out = await call(server, "ask_pool", {"prompt": "hi", "max_tokens": 48, "max_usdc": "0.001"})
    assert out["charged"] is False and "cap" in out["error"]  # 0.00196 > the agent's own 0.001 cap
    assert (await call(server, "ask_pool", {"prompt": "hi", "max_tokens": 48}))["budget_left_usdc"] == "0.001040"
    out = await call(server, "ask_pool", {"prompt": "hi", "max_tokens": 48})
    assert out["charged"] is False and out["budget_left_usdc"] == "0.001040"  # second paid call would overspend
    assert len(rig.facilitator.settled) == 1 and len(rig.facilitator.verified) == 1


async def test_pool_down_costs_nothing(tmp_path):
    rig = make_rig(tmp_path)
    await rig.mode("not_ready")
    server, budget = server_for(rig)
    out = await call(server, "ask_pool", {"prompt": "hi"})
    assert out["charged"] is False and out["status"] == 503 and budget.spent_atomic == 0
