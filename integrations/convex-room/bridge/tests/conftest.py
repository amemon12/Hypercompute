from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from room_bridge.config import BridgeConfig
from room_bridge.gateway import Gateway
from room_bridge.room_api import RoomApi

SECRET = "room-secret-DO-NOT-LOG-0123456789abcdef"  # noqa: S105 - a test value
OWNER_TOKEN = "owner-token-DO-NOT-LOG-fedcba9876543210"  # noqa: S105 - a test value


def make_cfg(**over: Any) -> BridgeConfig:
    base: dict[str, Any] = dict(site_url="https://test-room.convex.site", room_code="calhacks", secret=SECRET,
                                gateway_url="http://127.0.0.1:8000", flush_s=0.01, keepalive_s=10.0)
    return BridgeConfig(**{**base, **over})


@dataclass
class FakeRoom:
    """Convex's /bridge/* routes, recording every call. `stale_after` appends make the claim go stale."""

    prompts: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stale_after: int | None = None
    auth_seen: set[str] = field(default_factory=set)
    fail_next: dict[str, list[httpx.Response]] = field(default_factory=dict)  # route -> canned replies, used first

    def routes(self) -> list[str]:
        return [r for r, _ in self.calls]

    def bodies(self, route: str) -> list[dict[str, Any]]:
        return [b for r, b in self.calls if r == route]

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth_seen.add(request.headers.get("authorization", ""))
        route = request.url.path.removeprefix("/bridge/")
        body = json.loads(request.content)
        self.calls.append((route, body))
        if self.fail_next.get(route):
            return self.fail_next[route].pop(0)
        if route == "claim":
            return httpx.Response(200, json={"prompt": self.prompts.pop(0) if self.prompts else None})
        if route == "append":
            n = len(self.bodies("append"))
            return httpx.Response(200, json={"ok": self.stale_after is None or n <= self.stale_after})
        if route == "pool":
            return httpx.Response(200, json={"roomId": "r1", "open": True})
        return httpx.Response(200, json={"ok": True})


def prompt(text: str = "how big is the pool?", claim_seq: int = 1) -> dict[str, Any]:
    return {"id": "p1", "text": text, "author": "Ada", "claimSeq": claim_seq, "attempt": 1, "leaseMs": 90000}


def sse(payload: Any) -> bytes:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n".encode()


def chunk(text: str, n: int, rate: float) -> bytes:
    return sse({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                "timings": {"predicted_n": n, "predicted_per_second": rate}})


def final(n: int, rate: float, prefill: float, prompt_n: int = 20) -> bytes:
    return sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": {"completion_tokens": n, "prompt_tokens": prompt_n},
                "timings": {"predicted_n": n, "predicted_per_second": rate, "prompt_n": prompt_n,
                            "prompt_per_second": prefill}})


Body = Callable[[], AsyncIterator[bytes]]


@dataclass
class FakeGateway:
    """The gateway's loopback API. `chat` decides each chat response; `bootstrap_calls` counts token reads."""

    chat: Callable[[httpx.Request], httpx.Response]
    token: str = OWNER_TOKEN
    bootstrap_calls: int = 0
    chats: list[dict[str, Any]] = field(default_factory=list)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/owner/bootstrap":
            self.bootstrap_calls += 1
            return httpx.Response(200, json={"owner_token": self.token})
        if request.url.path == "/v1/chat/completions":
            if request.headers.get("x-hypercompute-owner") != self.token:
                return httpx.Response(403, json={"detail": "missing or wrong owner token"})
            self.chats.append(json.loads(request.content))
            return self.chat(request)
        return httpx.Response(404)


Chat = Callable[[httpx.Request], httpx.Response]


def streamed(*parts: bytes, delay: float = 0.0, error: Exception | None = None) -> Chat:
    async def body() -> AsyncIterator[bytes]:
        for i, part in enumerate(parts):
            if delay and i == 0:
                await asyncio.sleep(delay)
            yield part
            await asyncio.sleep(0.02)
        if error is not None:
            raise error

    return lambda _req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())


def wire(cfg: BridgeConfig, room: FakeRoom, gw: FakeGateway) -> tuple[RoomApi, Gateway]:
    api = RoomApi(cfg, httpx.AsyncClient(transport=httpx.MockTransport(room.handler)), "bridge-test",
                  retry_delays=())
    gateway = Gateway(cfg, httpx.AsyncClient(transport=httpx.MockTransport(gw.handler)))
    return api, gateway


@pytest.fixture
def cfg() -> BridgeConfig:
    return make_cfg()
