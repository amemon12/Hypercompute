"""The paywall proxy's HTTP surface.

POST /v1/chat/completions  paid (x402). OpenAI-compatible, so an agent only changes its base URL.
GET  /v1/models            free: the model the pool serves now
GET  /x402/quote           free: price for a max_tokens, and who would earn what from it
GET  /ledger, /api/ledger  free, read-only: what each friend's machine has earned
GET  /health               free

Order of checks on a paid chat, so a buyer is never asked to pay for something that cannot run: body valid, a slot
free, a model ready, the model matches; only then the 402 or the verified payment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from x402.http.facilitator_client_base import FacilitatorResponseError
from x402.http.types import HTTPResponseInstructions

from . import __version__
from .config import ProxyConfig
from .gateway import GatewayClient, GatewayError, PoolSnapshot
from .ledger import Ledger
from .ledger_page import LEDGER_HTML
from .paywall import BadRequest, build_http_server, facilitator_client, parse_chat, request_context
from .pricing import PriceError, atomic_to_usd, quote_atomic, reserved_tokens
from .relay import PaidRelay, payer_of
from .split import contributions, split

log = logging.getLogger("x402proxy")
BODY_LIMIT = 1024 * 1024


def _error(status: int, code: str, detail: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "charged": False, **extra})


def _reject_constant(name: str) -> None:
    raise BadRequest(f"{name} is not a number this API accepts")


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _once(fn: Callable[[], None]) -> Callable[[], None]:
    done = False

    def call() -> None:
        nonlocal done
        if not done:
            done = True
            fn()

    return call


async def read_body(request: Request, limit: int) -> bytes | None:
    """The body, or None once it passes `limit`; a declared length over the limit is refused before reading."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > limit:
            return None
    return bytes(raw)


@dataclass
class RateLimiter:
    """At most `per_minute` requests per client address in any rolling minute."""

    per_minute: int
    _seen: dict[str, deque] = field(default_factory=dict)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        stamps = self._seen.setdefault(client, deque())
        while stamps and now - stamps[0] > 60:
            stamps.popleft()
        if len(stamps) >= self.per_minute:
            return False
        stamps.append(now)
        if len(self._seen) > 10_000:  # bound memory under a spray of addresses
            self._seen = {k: v for k, v in self._seen.items() if v and now - v[-1] <= 60}
        return True


def quote_body(config: ProxyConfig, max_tokens: int, snap: PoolSnapshot) -> dict:
    amount = quote_atomic(max_tokens, config.pricing)
    shares = split(amount, contributions(snap.plan, snap.names), config.host_fee_bps, config.payees, config.pay_to)
    return {
        "price_usdc": atomic_to_usd(amount), "amount_atomic": str(amount), "network": config.network,
        "testnet": True, "asset": "USDC (Base Sepolia test token, no real value)", "pay_to": config.pay_to,
        "max_tokens_reserved": max_tokens, "model": snap.model_id, "deployment_id": snap.deployment_id,
        "pricing": {"base_usdc": atomic_to_usd(config.pricing.base_atomic),
                    "per_1k_tokens_usdc": atomic_to_usd(config.pricing.per_1k_atomic),
                    "host_fee_pct": config.host_fee_bps / 100},
        "split": [{"machine": s.display_name, "host": s.is_host, "layers": s.layers, "units": s.units,
                   "earns_usdc": atomic_to_usd(s.amount_atomic)} for s in shares],
    }


def _instructions(resp: HTTPResponseInstructions | None) -> Response:
    if resp is None:
        return _error(402, "payment_required", "payment required")
    if resp.is_html:
        return HTMLResponse(content=resp.body, status_code=resp.status, headers=resp.headers)
    return JSONResponse(content=resp.body or {}, status_code=resp.status, headers=resp.headers)


def _ledger_json(ledger: Ledger, config: ProxyConfig) -> dict:
    usd = atomic_to_usd
    summary = ledger.summary()
    return {
        "network": config.network, "testnet": True, "explorer_tx_url": config.explorer_tx_url,
        "summary": {**summary, "paid_usdc": usd(summary["paid_atomic"]),
                    "refunds_owed_usdc": usd(summary["refunds_owed_atomic"])},
        "totals": [{**t, "earned_usdc": usd(t["earned_atomic"]), "paid_out_usdc": usd(t["paid_out_atomic"])}
                   for t in ledger.totals()],
        "payouts": [{**p, "amount_usdc": usd(p["amount_atomic"])} for p in ledger.payouts(10)],
        "recent": [{**p, "amount_usdc": usd(p["amount_atomic"]), "refund_owed_usdc": usd(p["refund_owed_atomic"]),
                    "shares": [{**s, "amount_usdc": usd(s["amount_atomic"])} for s in p["shares"]]}
                   for p in ledger.recent(25)],
    }


def create_app(config: ProxyConfig, env: Mapping[str, str], *, facilitator: Any = None,
               gateway_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    ledger = Ledger(config.ledger_path)
    gateway = GatewayClient(config.gateway_url, transport=gateway_transport)
    http_server = build_http_server(config, facilitator or facilitator_client(config, env),
                                    lambda adapter: quote_body(config, adapter.chat.max_tokens, adapter.snapshot))
    relay = PaidRelay(config, http_server, gateway, ledger)
    slots = asyncio.Semaphore(config.max_inflight)
    limiter = RateLimiter(config.rate_per_min)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await gateway.aclose()
        ledger.close()

    app = FastAPI(title="Hypercompute x402 proxy", version=__version__, lifespan=lifespan, docs_url=None,
                  redoc_url=None)
    app.state.ledger = ledger
    app.state.slots = slots

    @app.post("/v1/chat/completions")
    async def paid_chat(request: Request) -> Response:
        if not limiter.allow(_client(request)):
            return _error(429, "rate_limited", "too many requests; slow down")
        raw = await read_body(request, BODY_LIMIT)
        if raw is None:
            return _error(413, "body_too_large", "request body is over 1 MiB")
        try:
            body = json.loads(raw, parse_constant=_reject_constant)
            chat = parse_chat(body, config)
            requested_model = body.get("model")
        except BadRequest as exc:
            return _error(400, "bad_request", str(exc))
        except (ValueError, RecursionError):  # bad JSON, bad UTF-8, absurd nesting or digit counts
            return _error(400, "bad_json", "request body must be JSON")
        # Unpaid and unverified requests never hold the slot; they only see whether it is free.
        if slots.locked():
            return _error(503, "busy", "the pool is serving another paid chat; retry in a few seconds")
        try:
            snap = await gateway.snapshot()
            if requested_model not in (None, "", snap.model_id):
                return _error(404, "model_not_found", f"the pool serves {snap.model_id!r}, not {requested_model!r}")
            context = request_context(request, chat, snap)
            result = await http_server.process_http_request(context)
        except GatewayError as exc:
            return _error(exc.status, exc.code, exc.detail)
        except FacilitatorResponseError as exc:
            log.warning("facilitator error: %s", exc)
            return _error(502, "facilitator_error", "the x402 facilitator did not answer as expected")
        if result.type != "payment-verified":
            return _instructions(result.response)
        if relay.strikes.blocked(payer_of(result)):  # verified, then spent elsewhere, too often: no free pool time
            return await relay.unsettled(result, 429, {"error": "payer_blocked",
                                                       "detail": "too many verified payments failed to settle"})
        if slots.locked():  # another verified payment got the slot first: release this one uncharged
            return await relay.unsettled(result, 503, {"error": "busy", "detail": "retry in a few seconds"})
        await slots.acquire()  # no await between the check and here, so this cannot block
        return await relay.serve(result, context, chat, snap, _once(slots.release), request.is_disconnected)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if not limiter.allow(_client(request)):
            return _error(429, "rate_limited", "too many requests; slow down")
        try:
            snap = await gateway.snapshot()
        except GatewayError as exc:
            return _error(exc.status, exc.code, exc.detail)
        return JSONResponse({"object": "list", "data": [{"id": snap.model_id, "object": "model",
                                                         "owned_by": "hypercompute"}]})

    @app.get("/x402/quote")
    async def quote(request: Request, max_tokens: int | None = Query(default=None)) -> Response:
        if not limiter.allow(_client(request)):
            return _error(429, "rate_limited", "too many requests; slow down")
        try:
            tokens = reserved_tokens(max_tokens, config.pricing)
            snap = await gateway.snapshot()
        except PriceError as exc:
            return _error(400, "bad_request", str(exc))
        except GatewayError as exc:
            return _error(exc.status, exc.code, exc.detail)
        return JSONResponse(quote_body(config, tokens, snap))

    page_limiter = RateLimiter(config.rate_per_min * 2)  # the ledger page polls every 3 s by itself

    @app.get("/api/ledger")
    async def ledger_json(request: Request) -> Response:
        if not page_limiter.allow(_client(request)):
            return _error(429, "rate_limited", "too many requests; slow down")
        return JSONResponse(_ledger_json(ledger, config))

    @app.get("/ledger", response_class=HTMLResponse)
    async def ledger_page() -> str:
        return LEDGER_HTML

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__, "network": config.network, "testnet": True,
                "facilitator": "cdp" if config.facilitator == "cdp" else config.facilitator,
                "pay_to": config.pay_to, "max_inflight": config.max_inflight}

    return app
