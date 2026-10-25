"""A verified payment's chat: when to settle, what the buyer gets, what the ledger records.

Streamed chat: settle at the first token. The proxy opens the gateway stream and holds the events until the first one
with generated text. A pool that is not ready, busy, silent, or loses a peer before that point answers an error and
the signed authorization is never submitted, so nothing moves on chain. Once text exists, the proxy settles, sends the
PAYMENT-RESPONSE header with the stream's headers, and streams the rest as it comes. A cut after that point (a peer
lost mid-reply) is recorded as interrupted with the tokens delivered and a refund owed for the rest.

Non-streamed chat: settle after the full reply, so any failure at all means no charge. The reply is held until then,
which is what a non-streamed reply does anyway.

Both: a buyer that hangs up before settlement is not charged, and everything that could fail after the money moves
(the split, the reply body) is computed before settling. The ledger write after settlement cannot fail the buyer: if
it errors, the settlement goes to a fallback file and the log, and the buyer still gets the reply.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send
from x402.http.types import HTTPProcessResult, HTTPRequestContext, HTTPTransportContext
from x402.http.x402_http_server import x402HTTPResourceServer
from x402.schemas import VerifiedPaymentCancelOptions

from . import sse
from .config import ProxyConfig
from .gateway import GatewayClient, GatewayError, PoolSnapshot
from .ledger import Ledger, Payment
from .paywall import ChatRequest
from .pricing import quote_atomic
from .split import Share, SplitError, contributions, split

log = logging.getLogger("x402proxy.relay")
ERROR_BODY_MAX = 64 * 1024
COMPLETION_MAX_BYTES = 16 * 1024 * 1024
DISCONNECT_POLL_S = 0.5
NOT_CHARGED = "nothing was charged: the signed payment was not submitted"
GATEWAY_ERROR_KEYS = ("code", "detail", "fix", "message")
Failure = tuple[int, dict]


@dataclass
class Tally:
    """What reached the buyer. Tokens come from llama.cpp's usage when the stream has it, else one per text chunk."""

    chunks_with_text: int = 0
    usage_tokens: int | None = None
    finished: bool = False
    error: dict | None = None

    def observe(self, ev: sse.Event) -> None:
        if ev.text:
            self.chunks_with_text += 1
        if ev.usage and isinstance(ev.usage.get("completion_tokens"), int):
            self.usage_tokens = ev.usage["completion_tokens"]
        if ev.finish_reason or ev.done:
            self.finished = True
        if ev.error:
            self.error = ev.error

    @property
    def tokens(self) -> int:
        return self.usage_tokens if self.usage_tokens is not None else self.chunks_with_text


@dataclass
class Assembly:
    """A non-streamed reply built from the stream's chunks, in OpenAI chat.completion form."""

    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    first: dict | None = None
    finish_reason: str | None = None
    usage: dict | None = None

    def add(self, ev: sse.Event) -> None:
        if not ev.payload:
            return
        self.first = self.first or ev.payload
        for choice in ev.payload.get("choices") or []:
            delta = choice.get("delta") or {}
            self.content.append(delta.get("content") or "")
            self.reasoning.append(delta.get("reasoning_content") or "")
        self.finish_reason = ev.finish_reason or self.finish_reason
        self.usage = ev.usage or self.usage

    def completion(self, model_id: str) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": "".join(self.content)}
        if any(self.reasoning):
            message["reasoning_content"] = "".join(self.reasoning)
        first = self.first or {}
        return {"id": first.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                "created": first.get("created"), "model": model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
                "usage": self.usage}


@dataclass
class StreamState:
    """A paid stream's bookkeeping, finalized exactly once by whichever side ends first."""

    tally: Tally = field(default_factory=Tally)
    outcome: str = "buyer_left"  # set to delivered or interrupted when the stream runs to its end
    closed: bool = False


class GuardedStream(StreamingResponse):
    """A StreamingResponse that always runs `on_close` when the response ends. Starlette does not close the body
    iterator when the client goes away while the iterator sits at a yield; without this, the slot and the upstream
    would wait for garbage collection."""

    def __init__(self, content: AsyncIterator[bytes], on_close: Callable[[], Awaitable[None]], **kw: Any) -> None:
        super().__init__(content, **kw)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._on_close()


def _error_status(error: dict) -> int:
    return 503 if error.get("code") in ("busy", "replacing", "no_model_ready") else 502


def _safe_gateway_detail(detail: Any) -> dict:
    """Only the gateway's error code and message reach a buyer, never the whole body."""
    if not isinstance(detail, dict):
        return {"detail": "gateway error"}
    return {k: str(detail[k])[:300] for k in GATEWAY_ERROR_KEYS if k in detail}


async def _watch_disconnect(is_disconnected: Callable[[], Awaitable[bool]], stop: asyncio.Event) -> bool:
    """True once the buyer is gone, False once told to stop. Stopped by the event, never by task.cancel():
    Starlette's is_disconnected() runs inside an already-cancelled anyio scope that would swallow a cancel."""
    while not stop.is_set():
        if await is_disconnected():
            return True
        try:
            await asyncio.wait_for(stop.wait(), DISCONNECT_POLL_S)
        except TimeoutError:
            pass
    return False


async def race_disconnect(work: Coroutine[Any, Any, Any],
                          is_disconnected: Callable[[], Awaitable[bool]]) -> tuple[bool, Any]:
    """(buyer_gone, result). The work is cancelled if the buyer hangs up first; its exceptions propagate."""
    stop = asyncio.Event()
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_watch_disconnect(is_disconnected, stop))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)
    if task.cancelled():
        return True, None
    return False, task.result()


@dataclass
class PayerStrikes:
    """Payers whose verified payment then failed to settle (spent or cancelled after verify). Each such attempt got
    pool time for free, so a payer with `limit` of them in the last `window_s` is refused before the slot."""

    limit: int = 3
    window_s: float = 3600.0
    _seen: dict[str, list[float]] = field(default_factory=dict)

    def strike(self, payer: str | None) -> None:
        if payer:
            now = time.monotonic()
            self._seen[payer.lower()] = [s for s in self._seen.get(payer.lower(), []) if now - s < self.window_s] + [now]

    def blocked(self, payer: str | None) -> bool:
        now = time.monotonic()
        return bool(payer) and len([s for s in self._seen.get(payer.lower(), []) if now - s < self.window_s]) \
            >= self.limit


def payer_of(verified: HTTPProcessResult) -> str | None:
    """The signer of an exact EVM payment (its EIP-3009 authorization's 'from')."""
    payload = getattr(verified.payment_payload, "payload", None)
    auth = payload.get("authorization") if isinstance(payload, dict) else None
    payer = auth.get("from") if isinstance(auth, dict) else None
    return payer if isinstance(payer, str) else None


class PaidRelay:
    def __init__(self, config: ProxyConfig, http_server: x402HTTPResourceServer, gateway: GatewayClient,
                 ledger: Ledger, fallback_path: Path | None = None) -> None:
        self.config, self.http_server, self.gateway, self.ledger = config, http_server, gateway, ledger
        self.fallback_path = fallback_path or config.ledger_path.with_name("unrecorded-settlements.jsonl")
        self.strikes = PayerStrikes()

    async def serve(self, verified: HTTPProcessResult, context: HTTPRequestContext, chat: ChatRequest,
                    snapshot: PoolSnapshot, release: Callable[[], None],
                    is_disconnected: Callable[[], Awaitable[bool]]) -> Response:
        """Owns `release` (the proxy's concurrency slot) from here on: it is called once the reply is finished."""
        handed_off = False
        try:
            try:
                upstream = await self.gateway.open_chat_stream(chat.body)
            except GatewayError as exc:
                return await self.unsettled(verified, exc.status, {"error": exc.code, "detail": exc.detail})
            except ValueError:  # the body could not be encoded for the gateway
                return await self.unsettled(verified, 400, {"error": "bad_request"})
            try:
                if upstream.status_code != 200:
                    return await self._upstream_refused(verified, upstream)
                if chat.stream:
                    response = await self._stream(verified, context, chat, snapshot, upstream, release,
                                                  is_disconnected)
                    handed_off = isinstance(response, GuardedStream)
                    return response
                return await self._complete(verified, context, chat, snapshot, upstream, is_disconnected)
            finally:
                if not handed_off:
                    with anyio.CancelScope(shield=True):
                        await upstream.aclose()
        finally:
            if not handed_off:
                release()

    # ------------------------------------------------------------------ no charge
    async def unsettled(self, verified: HTTPProcessResult, status: int, body: dict) -> JSONResponse:
        """An error before settlement: tell the SDK the handler failed, answer without charging."""
        cancel = None
        if verified.cancellation_dispatcher is not None:
            cancel = await verified.cancellation_dispatcher.cancel(
                VerifiedPaymentCancelOptions(reason="handler_failed", response_status=status))
        headers = self.http_server.create_failure_path_settlement_headers(
            cancel, verified.before_handler_settlement, verified.payment_payload) or {}
        return JSONResponse(status_code=status, content={**body, "charged": False, "note": NOT_CHARGED},
                            headers=headers)

    async def _upstream_refused(self, verified: HTTPProcessResult, upstream: httpx.Response) -> JSONResponse:
        raw = b""
        async for chunk in upstream.aiter_raw():
            raw += chunk
            if len(raw) > ERROR_BODY_MAX:
                break
        try:
            detail = json.loads(raw[:ERROR_BODY_MAX])
        except (ValueError, RecursionError):
            detail = None
        status = upstream.status_code if 400 <= upstream.status_code < 600 else 502
        return await self.unsettled(verified, status, {"error": "gateway_refused",
                                                       "gateway": _safe_gateway_detail(detail)})

    # ------------------------------------------------------------------ settlement
    async def _settle(self, verified: HTTPProcessResult, context: HTTPRequestContext, content_type: str) -> Any:
        transport = HTTPTransportContext(request=context, response_headers={"content-type": content_type})
        return await self.http_server.process_settlement(
            verified.payment_payload, verified.payment_requirements, context=context,
            declared_extensions=verified.declared_extensions, transport_context=transport,
            before_handler_settlement=verified.before_handler_settlement)

    async def _credit_snapshot(self, priced: PoolSnapshot) -> PoolSnapshot:
        """The pool that served the chat: normally the one it was priced on, but a redeploy in between means the new
        plan did the work. If the gateway cannot say, the priced plan is used."""
        try:
            return await self.gateway.snapshot()
        except GatewayError:
            return priced

    def _shares(self, verified: HTTPProcessResult, snapshot: PoolSnapshot) -> list[Share]:
        return split(int(verified.payment_requirements.amount), contributions(snapshot.plan, snapshot.names),
                     self.config.host_fee_bps, self.config.payees, self.config.pay_to)

    def _record(self, verified: HTTPProcessResult, settled: Any, chat: ChatRequest, snapshot: PoolSnapshot,
                shares: list[Share], settle_point: str) -> str:
        """Never raises: the money already moved, so a ledger failure must not take the reply away from the buyer."""
        payment = Payment(
            id=f"pay_{uuid.uuid4().hex[:12]}", settle_point=settle_point, tx=settled.transaction or "",
            network=settled.network or self.config.network, payer=settled.payer,
            amount_atomic=int(verified.payment_requirements.amount), max_tokens=chat.max_tokens,
            model_id=snapshot.model_id, deployment_id=snapshot.deployment_id, plan_id=str(snapshot.plan.get("id", "")))
        try:
            self.ledger.record_settlement(payment, shares)
            log.info("settled %s: %d atomic USDC, tx %s", payment.id, payment.amount_atomic, payment.tx)
        except Exception:
            log.critical("SETTLED BUT NOT RECORDED %s tx %s amount %d; written to %s", payment.id, payment.tx,
                         payment.amount_atomic, self.fallback_path, exc_info=True)
            self._write_fallback(payment, shares)
        return payment.id

    def _write_fallback(self, payment: Payment, shares: list[Share]) -> None:
        self._append_fallback({"kind": "settlement", "payment": asdict(payment), "shares": [asdict(s) for s in shares]})

    def _append_fallback(self, record: dict) -> None:
        line = json.dumps({"at": datetime.now(UTC).isoformat(), **record})
        try:
            with self.fallback_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            log.critical("fallback write failed too; record: %s", line)

    def _settle_failed(self, verified: HTTPProcessResult, settled: Any) -> Response:
        self.strikes.strike(payer_of(verified))
        log.warning("settlement failed after a valid verify (%s), payer %s", settled.error_reason,
                    payer_of(verified))
        resp = settled.response
        if resp is None:
            return JSONResponse(status_code=402, content={"error": "settlement_failed", "charged": False})
        return JSONResponse(status_code=resp.status, content=resp.body or {"error": "settlement_failed"},
                            headers=resp.headers)

    def _paid_headers(self, settled: Any, payment_id: str) -> dict[str, str]:
        return {**settled.headers, "Cache-Control": "no-store, private", "X-Hypercompute-Payment": payment_id}

    def _refund_owed(self, verified: HTTPProcessResult, tokens: int) -> int:
        amount = int(verified.payment_requirements.amount)
        delivered = quote_atomic(tokens, self.config.pricing) if tokens > 0 else 0
        return max(0, amount - delivered)

    async def _prepare(self, verified: HTTPProcessResult, snapshot: PoolSnapshot,
                       is_disconnected: Callable[[], Awaitable[bool]]) -> tuple[list[Share], PoolSnapshot] | Failure:
        """The last checks before money moves: the split works, and the buyer is still there."""
        credit = await self._credit_snapshot(snapshot)
        try:
            shares = self._shares(verified, credit)
        except (SplitError, KeyError, TypeError, ValueError) as exc:
            log.error("cannot split payment over plan %s: %s", credit.plan.get("id"), exc)
            return 503, {"error": "the pool's plan cannot be split, so the payment was not taken"}
        if await is_disconnected():
            return 499, {"error": "buyer disconnected before settlement"}
        return shares, credit

    # ------------------------------------------------------------------ streamed
    @staticmethod
    async def _read_head(events: AsyncIterator[sse.Event]) -> tuple[list[sse.Event], Failure | None]:
        """Events up to and including the first one with generated text, or why there is none."""
        head: list[sse.Event] = []
        try:
            async for ev in events:
                head.append(ev)
                if ev.error:
                    return head, (_error_status(ev.error), {"error": ev.error})
                if ev.text:
                    return head, None
                if ev.done or ev.finish_reason:
                    return head, (502, {"error": "the model returned no output"})
        except (httpx.HTTPError, sse.SSEError) as exc:
            return head, (502, {"error": f"gateway stream failed: {type(exc).__name__}"})
        return head, (502, {"error": "stream ended before any output"})

    async def _stream(self, verified: HTTPProcessResult, context: HTTPRequestContext, chat: ChatRequest,
                      snapshot: PoolSnapshot, upstream: httpx.Response, release: Callable[[], None],
                      is_disconnected: Callable[[], Awaitable[bool]]) -> Response:
        events = sse.events(upstream.aiter_raw())
        gone, result = await race_disconnect(self._read_head(events), is_disconnected)
        head, failure = result if not gone else ([], (499, {"error": "buyer disconnected before any output"}))
        prepared = await self._prepare(verified, snapshot, is_disconnected) if failure is None else failure
        if isinstance(prepared[0], int):
            with anyio.CancelScope(shield=True):
                await events.aclose()
            return await self.unsettled(verified, *prepared)
        shares, credit = prepared
        settled = await self._settle(verified, context, "text/event-stream")
        if not settled.success:
            await events.aclose()
            return self._settle_failed(verified, settled)
        payment_id = self._record(verified, settled, chat, credit, shares, "first_token")
        if await is_disconnected():  # gone during the settlement round trip: charged, nothing delivered
            self._book_outcome(verified, payment_id, "interrupted", 0, "buyer left during settlement")
            with anyio.CancelScope(shield=True):
                await events.aclose()
            return JSONResponse(status_code=499, content={"error": "buyer disconnected", "refund_owed": True})
        state = StreamState()

        async def finalize() -> None:
            if state.closed:
                return
            state.closed = True
            self._finish(verified, payment_id, state)
            release()  # before any await: a cancelled stream would raise again at the next await
            with anyio.CancelScope(shield=True):
                await events.aclose()
                await upstream.aclose()

        body = self._stream_body(payment_id, head, events, state, finalize)
        return GuardedStream(body, finalize, media_type="text/event-stream",
                             headers={**self._paid_headers(settled, payment_id), "X-Accel-Buffering": "no"})

    async def _stream_body(self, payment_id: str, head: list[sse.Event], events: AsyncIterator[sse.Event],
                           state: StreamState, finalize: Callable[[], Awaitable[None]]) -> AsyncIterator[bytes]:
        tally = state.tally
        try:
            for ev in head:
                tally.observe(ev)
                yield ev.raw
            async for ev in events:
                tally.observe(ev)
                yield ev.raw
                if ev.done or ev.error:
                    break
            state.outcome = "interrupted" if tally.error or not tally.finished else "delivered"
        except (httpx.HTTPError, sse.SSEError) as exc:
            state.outcome = "interrupted"
            tally.error = {"code": "interrupted", "message": f"gateway stream failed: {type(exc).__name__}"}
            yield sse.sse({"error": {**tally.error, "payment_id": payment_id}})
        finally:
            await finalize()

    def _finish(self, verified: HTTPProcessResult, payment_id: str, state: StreamState) -> None:
        tally = state.tally
        note = str((tally.error or {}).get("message") or "stream ended without a finish reason")
        # delivered, or the buyer hung up mid-stream: what it was sent is what it paid for
        outcome = "interrupted" if state.outcome == "interrupted" else "delivered"
        self._book_outcome(verified, payment_id, outcome, tally.tokens, note)

    def _book_outcome(self, verified: HTTPProcessResult, payment_id: str, outcome: str, tokens: int,
                      note: str) -> None:
        """Never raises. If the ledger cannot take it (for instance the payment row itself went to the fallback
        file), the outcome goes to the fallback file too, so `python -m x402proxy reconcile` can replay both."""
        refund = self._refund_owed(verified, tokens) if outcome == "interrupted" else 0
        try:
            if outcome == "interrupted":
                self.ledger.mark_interrupted(payment_id, tokens, refund, note[:300], self.config.host_fee_bps)
            else:
                self.ledger.mark_delivered(payment_id, tokens)
        except Exception:
            log.exception("ledger update failed for %s (%s); writing it to %s", payment_id, outcome,
                          self.fallback_path)
            self._append_fallback({"kind": "outcome", "payment_id": payment_id, "outcome": outcome,
                                   "completion_tokens": tokens, "refund_owed_atomic": refund, "note": note[:300],
                                   "host_fee_bps": self.config.host_fee_bps})

    # ------------------------------------------------------------------ non-streamed
    @staticmethod
    async def _read_all(events: AsyncIterator[sse.Event]) -> tuple[Assembly, Tally, Failure | None]:
        reply, tally, size = Assembly(), Tally(), 0
        try:
            async for ev in events:
                size += len(ev.raw)
                if size > COMPLETION_MAX_BYTES:
                    return reply, tally, (502, {"error": "reply over 16 MiB"})
                tally.observe(ev)
                if ev.error:
                    return reply, tally, (_error_status(ev.error), {"error": ev.error})
                reply.add(ev)
                if ev.done:
                    break
        except (httpx.HTTPError, sse.SSEError) as exc:
            return reply, tally, (502, {"error": f"gateway stream failed: {type(exc).__name__}"})
        if not tally.finished:
            return reply, tally, (502, {"error": "reply ended without a finish reason"})
        if tally.chunks_with_text == 0:
            return reply, tally, (502, {"error": "the model returned no output"})
        return reply, tally, None

    async def _complete(self, verified: HTTPProcessResult, context: HTTPRequestContext, chat: ChatRequest,
                        snapshot: PoolSnapshot, upstream: httpx.Response,
                        is_disconnected: Callable[[], Awaitable[bool]]) -> Response:
        events = sse.events(upstream.aiter_raw())
        try:
            gone, result = await race_disconnect(self._read_all(events), is_disconnected)
        finally:
            with anyio.CancelScope(shield=True):
                await events.aclose()
        reply, tally, failure = result if not gone else (None, None, (499, {"error": "buyer disconnected"}))
        prepared = await self._prepare(verified, snapshot, is_disconnected) if failure is None else failure
        if isinstance(prepared[0], int):
            return await self.unsettled(verified, *prepared)
        shares, credit = prepared
        try:  # rendered before settling (JSON encoding can fail, e.g. on a NaN): nothing can fail once money moves
            response = JSONResponse(content=reply.completion(credit.model_id))
        except (ValueError, TypeError):
            return await self.unsettled(verified, 502, {"error": "the reply could not be encoded"})
        settled = await self._settle(verified, context, "application/json")
        if not settled.success:
            return self._settle_failed(verified, settled)
        payment_id = self._record(verified, settled, chat, credit, shares, "complete")
        if await is_disconnected():  # gone during the settlement round trip: charged, nothing delivered
            self._book_outcome(verified, payment_id, "interrupted", 0, "buyer left during settlement")
        else:
            self._book_outcome(verified, payment_id, "delivered", tally.tokens, "")
        response.headers.update(self._paid_headers(settled, payment_id))
        return response
