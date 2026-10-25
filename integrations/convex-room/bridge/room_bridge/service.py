"""The bridge process: one pool-sync loop and one worker loop per inference slot."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import uuid
from dataclasses import dataclass

import httpx

from .config import BridgeConfig
from .gateway import Gateway
from .pool import PoolView, build_pool
from .room_api import RoomApi, RoomApiError
from .runner import Outcome, PromptRun

log = logging.getLogger("room_bridge")

BACKOFF_S = {Outcome.busy: 2.0, Outcome.lost: 3.0, Outcome.convex_down: 5.0}


@dataclass
class PoolStatus:
    view: PoolView | None = None
    sent_key: str | None = None
    sent_at: float = 0.0


def new_bridge_id() -> str:
    host = "".join(c for c in socket.gethostname().lower() if c.isalnum() or c == "-")[:24] or "host"
    return f"{host}-{uuid.uuid4().hex[:8]}"


class BridgeService:
    def __init__(self, cfg: BridgeConfig, api: RoomApi, gateway: Gateway) -> None:
        self._cfg, self.api, self.gateway = cfg, api, gateway
        self.status = PoolStatus()
        self._pool_ready = asyncio.Event()

    @classmethod
    def create(cls, cfg: BridgeConfig) -> tuple[BridgeService, list[httpx.AsyncClient]]:
        convex = httpx.AsyncClient(http2=False, follow_redirects=False)
        local = httpx.AsyncClient(trust_env=False)  # loopback only: never route the owner token through a proxy
        return cls(cfg, RoomApi(cfg, convex, new_bridge_id()), Gateway(cfg, local)), [convex, local]

    async def run(self) -> None:
        log.info("room %s: bridge %s, gateway %s, convex %s", self._cfg.room_code, self.api.bridge_id,
                 self._cfg.gateway_url, self._cfg.site_url)
        tasks = [asyncio.create_task(self.pool_loop(), name="pool")]
        tasks += [asyncio.create_task(self.worker_loop(i), name=f"worker-{i}") for i in range(16)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def pool_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                await self.sync_pool_once(loop.time())
            except Exception:  # noqa: BLE001 - one bad snapshot must not stop the room
                log.exception("pool snapshot failed; retrying")
            await asyncio.sleep(self._cfg.poll_s)

    async def sync_pool_once(self, now: float) -> PoolView:
        view = build_pool(await self.gateway.state(), self._cfg.slots)
        key = json.dumps(view.payload, sort_keys=True)
        previous = self.status.view
        self.status.view = view
        if previous is None or previous.ready != view.ready:
            log.info("gateway %s (%s, %d slot%s)", "ready" if view.ready else "not ready",
                     view.payload["deploymentState"], view.slots, "" if view.slots == 1 else "s")
        if key != self.status.sent_key or now - self.status.sent_at >= self._cfg.pool_heartbeat_s:
            try:
                await self.api.pool(view.payload)
                self.status.sent_key, self.status.sent_at = key, now
            except RoomApiError as exc:
                log.warning("pool snapshot not delivered: %s", exc)
        self._pool_ready.set()
        return view

    def may_claim(self, slot: int) -> bool:
        view = self.status.view
        return view is not None and view.ready and slot < view.slots

    async def worker_loop(self, slot: int) -> None:
        await self._pool_ready.wait()
        while True:
            if not self.may_claim(slot):
                await asyncio.sleep(1.0)
                continue
            try:
                outcome = await self.serve_one()
            except Exception:  # noqa: BLE001 - one bad prompt must not stop the other slots or the pool loop
                log.exception("worker %d: unexpected error; backing off", slot)
                await asyncio.sleep(BACKOFF_S[Outcome.convex_down])
                continue
            if outcome is None:
                await asyncio.sleep(self._cfg.idle_claim_s)
            elif outcome in BACKOFF_S:
                await asyncio.sleep(BACKOFF_S[outcome])

    async def serve_one(self) -> Outcome | None:
        """Claim and run one prompt. None when the queue is empty or Convex is out of reach."""
        try:
            prompt = await self.api.claim()
        except RoomApiError as exc:
            log.warning("claim failed: %s", exc)
            await asyncio.sleep(BACKOFF_S[Outcome.convex_down])
            return None
        if prompt is None:
            return None
        log.info("prompt %s: claimed (attempt %s)", prompt["id"], prompt.get("attempt"))
        return await PromptRun(self._cfg, self.api, self.gateway, prompt).run()
