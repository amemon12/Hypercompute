"""The x402 side: payment requirements for one chat, and the facilitator that verifies and settles them.

Built on the x402 SDK's framework-free x402HTTPResourceServer (the same object its FastAPI middleware wraps), so the
proxy decides when to settle: at the first token for a streamed chat, after the full reply for a non-streamed one.
The middleware cannot do that, it buffers the whole reply before settling.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from x402 import x402ResourceServer
from x402.http import FacilitatorConfig, HTTPFacilitatorClient
from x402.http.facilitator_client_base import FacilitatorResponseError
from x402.http.types import HTTPRequestContext, HTTPResponseBody, PaymentOption, RouteConfig
from x402.http.x402_http_server import x402HTTPResourceServer
from x402.mechanisms.evm.exact.server import ExactEvmScheme as ExactEvmServerScheme
from x402.schemas import SettleResponse, VerifyResponse

from .config import ProxyConfig
from .gateway import PoolSnapshot
from .pricing import PriceError, atomic_to_usd, quote_atomic, reserved_tokens

log = logging.getLogger("x402proxy.paywall")
ROUTE = "POST /v1/chat/completions"
# How long a signed authorization stays valid. A non-streamed chat settles after the whole reply, and a pooled 1T
# model at about 1.5 tok/s needs over 11 minutes for 1024 tokens plus prefill, so this leaves room for that.
PAYMENT_TIMEOUT_S = 1800
# The keys a paid chat may carry to the engine. Anything else (resume_from, tools, n, logit_bias...) is refused, so a
# payment buys exactly what was priced.
FLOAT_KEYS = ("temperature", "top_p", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty")
INT_KEYS = ("top_k", "seed")
ALLOWED_KEYS = frozenset({"model", "messages", "max_tokens", "stop", "stream", "reasoning_format",
                          "chat_template_kwargs", *FLOAT_KEYS, *INT_KEYS})
ROLES = frozenset({"system", "user", "assistant"})
MAX_MESSAGES = 256
MAX_STOP = 4  # stop strings, each at most MAX_STOP_CHARS
MAX_STOP_CHARS = 64
# chat_template_kwargs reaches the model's chat template (e.g. gpt-oss reasoning_effort); flat scalars only
MAX_TEMPLATE_KWARGS = 8


class BadRequest(ValueError):
    pass


@dataclass(frozen=True)
class ChatRequest:
    body: dict  # what goes to the gateway: allowed keys only, messages rebuilt, max_tokens set
    max_tokens: int
    stream: bool


def _messages(raw: Any, max_chars: int) -> list[dict]:
    """Rebuilt as {role, content} only, so nothing else inside a message reaches the engine."""
    if not isinstance(raw, list) or not raw or len(raw) > MAX_MESSAGES:
        raise BadRequest(f"'messages' must be a list of 1 to {MAX_MESSAGES} messages")
    out = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            raise BadRequest("each message needs a role (system, user or assistant) and a string 'content'")
        try:
            m["content"].encode("utf-8")  # a lone surrogate cannot be sent on to the gateway
        except UnicodeEncodeError:
            raise BadRequest("message content is not valid UTF-8 text") from None
        out.append({"role": m["role"], "content": m["content"]})
    if sum(len(m["content"]) for m in out) > max_chars:
        raise BadRequest(f"the prompt is over {max_chars} characters")
    return out


def _sampling(body: dict) -> dict:
    """Sampling settings with their types checked; only the keys the buyer sent."""
    out: dict[str, Any] = {}
    for key in FLOAT_KEYS:
        if key in body:
            v = body[key]
            if isinstance(v, bool) or not isinstance(v, int | float) or not -1e6 < v < 1e6:
                raise BadRequest(f"'{key}' must be a finite number")
            out[key] = v
    for key in INT_KEYS:
        if key in body:
            if isinstance(body[key], bool) or not isinstance(body[key], int) or not -2**63 < body[key] < 2**63:
                raise BadRequest(f"'{key}' must be an integer")
            out[key] = body[key]
    if "stop" in body:
        stop = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        if not isinstance(stop, list) or len(stop) > MAX_STOP or not all(
                isinstance(s, str) and 0 < len(s) <= MAX_STOP_CHARS for s in stop):
            raise BadRequest(f"'stop' must be a string or up to {MAX_STOP} strings of {MAX_STOP_CHARS} characters")
        out["stop"] = stop
    return out


def _template(body: dict) -> dict:
    out: dict[str, Any] = {}
    kwargs = body.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict) or len(kwargs) > MAX_TEMPLATE_KWARGS or not all(
            isinstance(k, str) and len(k) <= 64 and isinstance(v, str | int | float | bool)
            and (not isinstance(v, str) or len(v) <= 256) and (not isinstance(v, float) or math.isfinite(v))
            and (not isinstance(v, int) or -2**63 < v < 2**63) for k, v in kwargs.items()):
        raise BadRequest(f"'chat_template_kwargs' must be an object of up to {MAX_TEMPLATE_KWARGS} plain values")
    if kwargs:
        out["chat_template_kwargs"] = dict(kwargs)
    if "reasoning_format" in body:
        if not isinstance(body["reasoning_format"], str) or len(body["reasoning_format"]) > 32:
            raise BadRequest("'reasoning_format' must be a short string")
        out["reasoning_format"] = body["reasoning_format"]
    return out


def parse_chat(body: Any, config: ProxyConfig) -> ChatRequest:
    """Validate a buyer's chat body before anything is priced or charged."""
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    extra = sorted(set(body) - ALLOWED_KEYS)
    if extra:
        raise BadRequest(f"unsupported keys for a paid chat: {', '.join(extra)}")
    if not isinstance(body.get("model", ""), str):
        raise BadRequest("'model' must be a string")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise BadRequest("'stream' must be true or false")
    try:
        max_tokens = reserved_tokens(body.get("max_tokens"), config.pricing)
    except PriceError as exc:
        raise BadRequest(str(exc)) from exc
    forwarded = {"messages": _messages(body.get("messages"), config.max_prompt_chars), **_sampling(body),
                 **_template(body), "max_tokens": max_tokens}
    return ChatRequest(body=forwarded, max_tokens=max_tokens, stream=stream)


class ProxyAdapter:
    """x402's HTTPAdapter over a FastAPI request whose JSON body was already read and validated. It also carries
    the pool snapshot the request was priced against, so the 402 quote and the ledger use the same plan."""

    def __init__(self, request: Request, chat: ChatRequest, snapshot: PoolSnapshot) -> None:
        self._request, self.chat, self.snapshot = request, chat, snapshot

    def get_header(self, name: str) -> str | None:
        return self._request.headers.get(name)

    def get_method(self) -> str:
        return self._request.method

    def get_path(self) -> str:
        return self._request.url.path

    def get_url(self) -> str:
        return str(self._request.url)

    def get_accept_header(self) -> str:
        return self._request.headers.get("accept", "")

    def get_user_agent(self) -> str:
        return self._request.headers.get("user-agent", "")

    def get_query_params(self) -> dict[str, str | list[str]]:
        return dict(self._request.query_params)

    def get_query_param(self, name: str) -> str | None:
        return self._request.query_params.get(name)

    def get_body(self) -> dict:
        return self.chat.body


def request_context(request: Request, chat: ChatRequest, snapshot: PoolSnapshot) -> HTTPRequestContext:
    adapter = ProxyAdapter(request, chat, snapshot)
    return HTTPRequestContext(adapter=adapter, path=request.url.path, method=request.method,
                              payment_header=adapter.get_header("payment-signature") or adapter.get_header("x-payment"))


def facilitator_client(config: ProxyConfig, env: Mapping[str, str]) -> HTTPFacilitatorClient:
    if config.facilitator == "cdp":
        from cdp.x402 import create_facilitator_config  # imported here: only the CDP path needs the CDP SDK

        return HTTPFacilitatorClient(create_facilitator_config(env["CDP_API_KEY_ID"], env["CDP_API_KEY_SECRET"]))
    return HTTPFacilitatorClient(FacilitatorConfig(url=config.facilitator))


_REASON = re.compile(r"^[a-z0-9_]{1,64}$")


def _reason_from_error(exc: Exception) -> tuple[str | None, str | None]:
    """The CDP facilitator answers an invalid payment with HTTP 400 and a JSON body, which the SDK raises as
    'Facilitator verify failed (400): {...}'. Pull out only the machine reason code and the payer address."""
    text = str(exc)
    start = text.find("{")
    if start < 0:
        return None, None
    try:
        body = json.loads(text[start:])
    except ValueError:
        return None, None
    reason = body.get("invalidReason") or body.get("errorReason")
    payer = body.get("payer")
    return (reason if isinstance(reason, str) and _REASON.match(reason) else None,
            payer if isinstance(payer, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", payer) else None)


class QuietFacilitator:
    """Keeps the facilitator's raw error text (it can include its HTTP response body) out of what buyers see: the SDK
    puts a failed verify's or settle's exception message into the 402 it returns. A buyer gets the reason code only
    (e.g. invalid_payload, insufficient_funds); the full text goes to the log."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def get_supported(self) -> Any:
        return self._inner.get_supported()

    async def verify(self, payload: Any, requirements: Any) -> Any:
        try:
            return await self._inner.verify(payload, requirements)
        except FacilitatorResponseError as exc:  # its message can hold the facilitator's body: keep it in the log
            log.warning("facilitator verify answered badly: %s", exc)
            raise FacilitatorResponseError("facilitator_bad_response") from None
        except Exception as exc:
            log.warning("facilitator verify failed: %s", exc)
            reason, payer = _reason_from_error(exc)
            if reason is None:
                raise RuntimeError("facilitator_verify_failed") from None
            return VerifyResponse(is_valid=False, invalid_reason=reason, payer=payer)

    async def settle(self, payload: Any, requirements: Any) -> Any:
        try:
            return await self._inner.settle(payload, requirements)
        except FacilitatorResponseError as exc:
            log.warning("facilitator settle answered badly: %s", exc)
            raise FacilitatorResponseError("facilitator_bad_response") from None
        except Exception as exc:
            log.warning("facilitator settle failed: %s", exc)
            reason, payer = _reason_from_error(exc)
            if reason is None:
                raise RuntimeError("facilitator_settle_failed") from None
            return SettleResponse(success=False, error_reason=reason, transaction="", network=requirements.network,
                                  payer=payer)


def build_http_server(config: ProxyConfig, facilitator: Any,
                      unpaid_body: Callable[[ProxyAdapter], dict]) -> x402HTTPResourceServer:
    """`unpaid_body` turns the request into the JSON a 402 carries next to the PAYMENT-REQUIRED header (the quote)."""
    server = x402ResourceServer(QuietFacilitator(facilitator))
    server.register(config.network, ExactEvmServerScheme())

    def price(ctx: HTTPRequestContext) -> str:
        return "$" + atomic_to_usd(quote_atomic(ctx.adapter.chat.max_tokens, config.pricing))

    route = RouteConfig(
        accepts=PaymentOption(scheme="exact", pay_to=config.pay_to, price=price, network=config.network,
                              max_timeout_seconds=PAYMENT_TIMEOUT_S),
        description="Chat completion from a Hypercompute pool: one model split across friends' machines. "
                    "Testnet USDC only.",
        mime_type="text/event-stream or application/json",
        service_name="Hypercompute pool",
        tags=["llm", "inference", "openai-compatible"],
        unpaid_response_body=lambda ctx: HTTPResponseBody(content_type="application/json",
                                                          body=unpaid_body(ctx.adapter)),
    )
    http_server = x402HTTPResourceServer(server, {ROUTE: route})
    http_server.initialize()  # asks the facilitator what it supports; fails fast on a wrong network
    return http_server
