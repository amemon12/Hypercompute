"""The Convex side: POST https://NAME.convex.site/bridge/<route> with the room secret as a bearer token."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from .config import BridgeConfig

log = logging.getLogger("room_bridge.convex")

RETRY_DELAYS_S = (0.5, 1.0, 2.0, 4.0)


class RoomApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"convex {status}: {message}")
        self.status = status


class RoomApi:
    def __init__(self, cfg: BridgeConfig, client: httpx.AsyncClient, bridge_id: str,
                 retry_delays: tuple[float, ...] = RETRY_DELAYS_S) -> None:
        self._cfg = cfg
        self._client = client
        self.bridge_id = bridge_id
        self._retry_delays = retry_delays

    async def _call(self, route: str, body: dict[str, Any]) -> dict[str, Any]:
        """One route call. Network errors and 5xx are retried; a 4xx is final. Headers are never logged."""
        url = f"{self._cfg.site_url}/bridge/{route}"
        headers = {"Authorization": f"Bearer {self._cfg.secret}"}
        last: Exception | None = None
        for delay in (0.0, *self._retry_delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                r = await self._client.post(url, json=body, headers=headers, timeout=15.0)
            except httpx.HTTPError as exc:
                last = exc
                log.warning("convex /bridge/%s unreachable (%s); retrying", route, type(exc).__name__)
                continue
            if r.status_code >= 500:
                last = RoomApiError(r.status_code, _error_text(r))
                log.warning("convex /bridge/%s answered %d; retrying", route, r.status_code)
                continue
            if r.status_code != 200:
                raise RoomApiError(r.status_code, _error_text(r))
            try:
                data = r.json()
            except ValueError as exc:
                raise RoomApiError(502, "convex answered with something that is not JSON") from exc
            if not isinstance(data, dict):
                raise RoomApiError(502, "convex answered with a non-object")
            return data
        if isinstance(last, RoomApiError):
            raise last
        raise RoomApiError(0, f"unreachable: {type(last).__name__ if last else 'unknown'}")

    async def claim(self) -> dict[str, Any] | None:
        """claimKey makes a retried claim (reply lost on the way back) return the same prompt, not a second one."""
        body = {"code": self._cfg.room_code, "bridgeId": self.bridge_id, "claimKey": uuid.uuid4().hex}
        data = await self._call("claim", body)
        return data.get("prompt")

    async def append(self, prompt_id: str, claim_seq: int, delta: str, offset: int, reasoning: str,
                     reasoning_offset: int, live_tps: float | None) -> bool:
        """False when the claim is stale (the prompt was requeued or finished elsewhere). The offsets (characters sent
        before this delta) let Convex place a retried delta over itself instead of appending it twice."""
        body: dict[str, Any] = {"promptId": prompt_id, "claimSeq": claim_seq, "delta": delta, "offset": offset}
        if reasoning:
            body = {**body, "reasoning": reasoning, "reasoningOffset": reasoning_offset}
        if live_tps is not None:
            body = {**body, "liveTps": live_tps}
        return bool((await self._call("append", body)).get("ok"))

    async def finish(self, prompt_id: str, claim_seq: int, tail: str, offset: int, reasoning_tail: str,
                     reasoning_offset: int, stats: dict[str, Any]) -> bool:
        body = {"promptId": prompt_id, "claimSeq": claim_seq, "tail": tail, "offset": offset,
                "reasoningTail": reasoning_tail, "reasoningOffset": reasoning_offset, **stats}
        return bool((await self._call("finish", body)).get("ok"))

    async def release(self, prompt_id: str, claim_seq: int, reason: str) -> bool:
        body = {"promptId": prompt_id, "claimSeq": claim_seq, "reason": reason}
        return bool((await self._call("release", body)).get("ok"))

    async def fail(self, prompt_id: str, claim_seq: int, error: str) -> bool:
        body = {"promptId": prompt_id, "claimSeq": claim_seq, "error": error[:300]}
        return bool((await self._call("fail", body)).get("ok"))

    async def pool(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        body = {"code": self._cfg.room_code, "bridgeId": self.bridge_id, "title": self._cfg.title, **snapshot}
        return await self._call("pool", body)

    async def admin(self, action: str) -> dict[str, Any]:
        return await self._call("admin", {"code": self._cfg.room_code, "action": action})


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:200]
    return str(body.get("error", body))[:200] if isinstance(body, dict) else str(body)[:200]
