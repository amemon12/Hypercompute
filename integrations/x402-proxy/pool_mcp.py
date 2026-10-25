"""An MCP server that gives any agent (Claude Code, Cursor, ...) a paid tool: ask a Hypercompute pool, paying per call
in Base Sepolia test USDC over x402, from a CDP server wallet. Testnet only.

    claude mcp add hypercompute-pool -- uv run --directory <this folder> python pool_mcp.py

Tools
  pool_quote(max_tokens)                  free: price, model, and which machines would earn what
  ask_pool(prompt, max_tokens, max_usdc)  pays, then answers; returns the answer, the price, the tx and the split

Guard rails the agent cannot change: a per-call cap (X402_MCP_MAX_PER_CALL_USDC, default 0.05) and a budget for the
server's lifetime (X402_MCP_BUDGET_USDC, default 0.25). A call settles only after the full reply arrives, so a failed
or cut answer costs nothing.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from x402.http.clients.httpx import PaymentError, x402AsyncTransport
from x402.http.utils import decode_payment_required_header, decode_payment_response_header

from x402proxy.config import BASE_SEPOLIA, EXPLORERS, read_env_file, safe_console
from x402proxy.pricing import PriceError, atomic_to_usd, usd_to_atomic
from x402proxy.wallet import Wallet, WalletError, default_signer, open_wallet, paying_client

HERE = Path(__file__).resolve().parent
TIMEOUT = httpx.Timeout(900, connect=10)  # a pooled model can take minutes; the proxy bounds it too


@dataclass
class Budget:
    per_call_atomic: int
    total_atomic: int
    spent_atomic: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def left_atomic(self) -> int:
        return self.total_atomic - self.spent_atomic


def build_server(url: str, open_signer: Callable[[], Awaitable[Wallet]], budget: Budget,
                 transport: httpx.AsyncBaseTransport | None = None) -> MCPServer:
    """`open_signer` is called once, on the first paid call; `transport` lets tests route to an in-process proxy."""
    server = MCPServer(name="hypercompute-pool", instructions=(
        "ask_pool sends a prompt to a pool of friends' computers running a large open model and pays for it in "
        "Base Sepolia TEST USDC (no real money) via x402. Call pool_quote first if price matters. Each call is "
        "capped and the server has a total budget."))
    state: dict[str, Wallet] = {}

    def http(via: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
        chosen = via or transport
        return httpx.AsyncClient(base_url=url, timeout=TIMEOUT, trust_env=False, **({"transport": chosen} if chosen else {}))

    @server.tool(description="Price of one pool answer for max_tokens, the served model, and the per-machine split. "
                             "Free; moves no money.")
    async def pool_quote(max_tokens: int = 128) -> dict:
        async with http() as c:
            r = await c.get("/x402/quote", params={"max_tokens": max_tokens})
        return r.json() if r.status_code == 200 else {"error": r.json(), "status": r.status_code}

    @server.tool(description="Ask the Hypercompute pool a question and pay for the answer in Base Sepolia test USDC "
                             "via x402. max_usdc caps this call (the server also enforces its own cap and budget).")
    async def ask_pool(prompt: str, max_tokens: int = 128, max_usdc: str = "0.02") -> dict:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False}
        async with http() as c:  # unpaid first: learn the price before signing anything
            r = await c.post("/v1/chat/completions", json=body)
        if r.status_code != 402:
            return {"error": "the pool cannot take a paid request now", "status": r.status_code, "detail": r.json(),
                    "charged": False}
        price = int(decode_payment_required_header(r.headers["payment-required"]).accepts[0].amount)
        try:
            cap = min(usd_to_atomic(max_usdc), budget.per_call_atomic)
        except PriceError as exc:
            return {"error": str(exc), "charged": False}
        async with budget.lock:  # one paid call at a time, so the budget check and the spend cannot race
            if price > cap or price > budget.left_atomic:
                return {"error": "price over the cap or the remaining budget", "price_usdc": atomic_to_usd(price),
                        "cap_usdc": atomic_to_usd(cap), "budget_left_usdc": atomic_to_usd(budget.left_atomic),
                        "charged": False}
            if "wallet" not in state:
                state["wallet"] = await open_signer()
            client = paying_client(state["wallet"].signer, atomic_to_usd(cap))
            try:
                async with http(via=x402AsyncTransport(client, transport=transport)) as c:
                    paid = await c.post("/v1/chat/completions", json=body)
            except PaymentError as exc:
                return {"error": f"did not pay: {exc}", "charged": False}
            if paid.status_code != 200:
                return {"error": "payment or answer failed", "status": paid.status_code, "charged": False,
                        "detail": paid.text[:500]}
            budget.spent_atomic += price
            receipt = decode_payment_response_header(paid.headers["payment-response"])
            message = paid.json()["choices"][0]["message"]
            return {"answer": message.get("content") or "", "reasoning": message.get("reasoning_content"),
                    "model": paid.json().get("model"), "paid_usdc": atomic_to_usd(price), "network": BASE_SEPOLIA,
                    "testnet": True, "tx": receipt.transaction,
                    "explorer": EXPLORERS[BASE_SEPOLIA] + receipt.transaction if receipt.transaction else None,
                    "payer": receipt.payer, "split": r.json().get("split"),
                    "budget_left_usdc": atomic_to_usd(budget.left_atomic)}

    return server


def main() -> int:
    safe_console()
    env = {**read_env_file(HERE / ".env"), **read_env_file(HERE / ".env.buyer"), **os.environ}
    try:
        budget = Budget(per_call_atomic=usd_to_atomic(env.get("X402_MCP_MAX_PER_CALL_USDC", "0.05")),
                        total_atomic=usd_to_atomic(env.get("X402_MCP_BUDGET_USDC", "0.25")))
    except PriceError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    kind = env.get("X402_MCP_SIGNER") or default_signer(env)

    async def open_signer() -> Wallet:
        try:
            return await open_wallet(kind, env)
        except WalletError as exc:
            raise RuntimeError(f"no wallet for paid calls: {exc}") from exc

    url = env.get("X402_PROXY_URL", "http://127.0.0.1:8402").rstrip("/")
    build_server(url, open_signer, budget).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
