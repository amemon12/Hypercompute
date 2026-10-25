"""The local Hypercompute gateway, over loopback HTTP only (the owner API refuses anything else).

Rates come from llama.cpp's own timings (the per-token `timings` when the engine sends them, else the last chunk's),
never from counting stream events.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import BridgeConfig

try:  # the gateway's own fragmentation-tolerant SSE parser, when the bridge runs in the host's venv
    from hypercompute.sse import SSEParser
except ImportError:  # pragma: no cover - exercised only outside the repo venv
    from ._sse_fallback import SSEParser  # type: ignore[no-redef]

log = logging.getLogger("room_bridge.gateway")

OWNER_HEADER = "X-Hypercompute-Owner"
BUSY_CODES = {"busy", "engine_busy"}
LOST_CODES = {"interrupted", "timeout", "upstream_error"}


class GatewayBusy(RuntimeError):
    """Refused before any output: no slot free (429), engine busy, or no ready deployment (503)."""


class GatewayLost(RuntimeError):
    """The gateway went away or the engine stream broke after it started; the prompt should be retried."""


class GatewayRefused(RuntimeError):
    """The gateway rejected the request itself (4xx other than busy); retrying the same prompt will not help."""


@dataclass(frozen=True)
class Delta:
    content: str
    reasoning: str
    live_tps: float | None  # predicted_per_second from per-token engine timings, when present


@dataclass(frozen=True)
class Done:
    finish_reason: str | None
    tokens: int | None
    prompt_tokens: int | None
    decode_tps: float | None
    prefill_tps: float | None


@dataclass(frozen=True)
class GatewayState:
    reachable: bool
    nodes: list[dict[str, Any]]
    deployments: list[dict[str, Any]]
    network: dict[str, Any]


class Gateway:
    def __init__(self, cfg: BridgeConfig, client: httpx.AsyncClient) -> None:
        self._cfg = cfg
        self._client = client
        self._token: str | None = None

    async def _owner_token(self, refresh: bool = False) -> str:
        if self._token is None or refresh:
            r = await self._client.get(f"{self._cfg.gateway_url}/api/owner/bootstrap", timeout=10.0)
            r.raise_for_status()
            try:
                body = r.json()
            except ValueError as exc:
                raise GatewayBusy("the gateway's owner bootstrap did not answer JSON") from exc
            token = body.get("owner_token") if isinstance(body, dict) else None
            if not isinstance(token, str) or not token:
                raise GatewayRefused("the gateway handed out no owner token")
            self._token = token
        return self._token

    async def state(self) -> GatewayState:
        base = self._cfg.gateway_url
        try:
            nodes = await self._client.get(f"{base}/api/nodes", timeout=10.0)
            deps = await self._client.get(f"{base}/api/deployments", timeout=10.0)
            net = await self._client.get(f"{base}/api/network", timeout=10.0)
            for r in (nodes, deps, net):
                r.raise_for_status()
            return GatewayState(True, _list(nodes.json()), _list(deps.json()), _dict(net.json()))
        except (httpx.HTTPError, ValueError) as exc:
            log.debug("gateway state unavailable: %s", type(exc).__name__)
            return GatewayState(False, [], [], {})

    async def chat(self, text: str) -> AsyncIterator[Delta | Done]:
        """Stream one room prompt. Raises GatewayBusy, GatewayLost or GatewayRefused; yields Deltas then one Done."""
        body = {
            "messages": [{"role": "system", "content": self._cfg.system_prompt}, {"role": "user", "content": text}],
            "stream": True,
            "max_tokens": self._cfg.max_tokens,
            "timings_per_token": True,  # llama-server: each chunk carries the engine's own running rates
        }
        for attempt in (0, 1):
            try:
                token = await self._owner_token(refresh=attempt == 1)
            except httpx.HTTPError as exc:
                raise GatewayBusy(f"gateway unreachable: {type(exc).__name__}") from exc
            try:
                async with self._client.stream(
                    "POST", f"{self._cfg.gateway_url}/v1/chat/completions", json=body,
                    headers={OWNER_HEADER: token},
                    timeout=httpx.Timeout(15.0, read=self._cfg.read_timeout_s),
                ) as r:
                    if r.status_code == 403 and attempt == 0:  # the gateway restarted with a new owner token
                        await r.aread()
                        continue
                    if r.status_code != 200:
                        await r.aread()
                        raise _refusal(r)
                    async for item in _fold(r.aiter_bytes()):
                        yield item
                    return
            except httpx.ConnectError as exc:  # nothing started: the gateway is down, not this prompt's fault
                raise GatewayBusy(f"gateway unreachable: {type(exc).__name__}") from exc
            except httpx.HTTPError as exc:
                raise GatewayLost(f"gateway connection: {type(exc).__name__}") from exc
        raise GatewayRefused("the gateway refused the owner token twice")


async def _fold(chunks: AsyncIterator[bytes]) -> AsyncIterator[Delta | Done]:
    parser = SSEParser()
    started = False
    finish: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    async for chunk in chunks:
        try:
            events = parser.feed(chunk)
        except ValueError as exc:  # one SSE event over the parser's size limit
            raise GatewayLost(f"unreadable gateway stream: {exc}") from exc
        for ev in events:
            data = ev.data.strip()
            if data == "[DONE]":
                yield _done(finish, usage, timings)
                return
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("error"):
                raise _stream_error(payload["error"], started)
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            if isinstance(payload.get("timings"), dict):
                timings = payload["timings"]
            content, reasoning = "", ""
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                content += _text(delta.get("content"))
                reasoning += _text(delta.get("reasoning_content"))
                finish = choice.get("finish_reason") or finish
            if content or reasoning:
                started = True
                yield Delta(content, reasoning, _rate(timings.get("predicted_per_second")))
    if finish or usage:  # engines that end without [DONE] but did finish
        yield _done(finish, usage, timings)
        return
    raise GatewayLost("the gateway stream ended without finishing")


def _done(finish: str | None, usage: dict[str, Any], timings: dict[str, Any]) -> Done:
    tokens = timings.get("predicted_n") if isinstance(timings.get("predicted_n"), int) else None
    prompt_tokens = timings.get("prompt_n") if isinstance(timings.get("prompt_n"), int) else None
    return Done(
        finish_reason=finish,
        tokens=tokens if tokens is not None else _int(usage.get("completion_tokens")),
        prompt_tokens=prompt_tokens if prompt_tokens is not None else _int(usage.get("prompt_tokens")),
        decode_tps=_rate(timings.get("predicted_per_second")),
        prefill_tps=_rate(timings.get("prompt_per_second")),
    )


def _stream_error(err: Any, started: bool) -> Exception:
    code = err.get("code") if isinstance(err, dict) else None
    message = (err.get("message") if isinstance(err, dict) else str(err)) or "engine error"
    if code in BUSY_CODES and not started:
        return GatewayBusy(str(message)[:200])
    if code in LOST_CODES or code in BUSY_CODES:
        return GatewayLost(f"{code}: {str(message)[:200]}")
    return GatewayRefused(f"{code or 'error'}: {str(message)[:200]}")


def _refusal(r: httpx.Response) -> Exception:
    try:
        body = r.json()
    except ValueError:
        body = {}
    detail = str(body.get("detail") or body.get("error") or r.reason_phrase)[:200] if isinstance(body, dict) else ""
    if r.status_code in (429, 503):
        return GatewayBusy(f"{r.status_code}: {detail}")
    if r.status_code >= 500:
        return GatewayLost(f"{r.status_code}: {detail}")
    return GatewayRefused(f"{r.status_code}: {detail}")


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _rate(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _list(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
