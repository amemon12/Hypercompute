"""One claimed prompt, start to finish: stream from the gateway, batch deltas into appends, then finish, release
or fail. The outcome decides what the worker loop does next."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import aclosing
from enum import StrEnum
from typing import Any

from .config import BridgeConfig
from .gateway import Delta, Done, Gateway, GatewayBusy, GatewayLost, GatewayRefused
from .room_api import RoomApi, RoomApiError

log = logging.getLogger("room_bridge.runner")


class Outcome(StrEnum):
    done = "done"
    busy = "busy"  # released without spending an attempt; back off before claiming again
    lost = "lost"  # released as gateway_lost (counts); wait for the gateway to be ready again
    failed = "failed"
    stale = "stale"  # Convex says the claim moved on (lease expired and requeued): drop it
    convex_down = "convex_down"  # could not reach Convex; the lease will requeue the prompt


class StaleClaim(RuntimeError):
    pass


class PromptRun:
    def __init__(self, cfg: BridgeConfig, api: RoomApi, gateway: Gateway, prompt: dict[str, Any],
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg, self._api, self._gw = cfg, api, gateway
        self.id: str = prompt["id"]
        self.claim_seq: int = int(prompt["claimSeq"])
        self._text: str = prompt["text"]
        # A keepalive must land well inside the lease, whatever HC_LEASE_MS the deployment uses.
        lease_s = float(prompt.get("leaseMs") or 90_000) / 1000
        self._keepalive_s = min(cfg.keepalive_s, max(1.0, lease_s / 3))
        self._clock = clock
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._sent = 0  # characters of answer already handed to Convex: the next append's offset
        self._sent_reasoning = 0
        self._live: float | None = None
        self._last_sent = clock()
        self._lock = asyncio.Lock()

    async def run(self) -> Outcome:
        pump = asyncio.create_task(self._pump(), name=f"pump-{self.id}")
        stream = asyncio.create_task(self._stream(), name=f"stream-{self.id}")
        try:
            return await self._run(pump, stream)
        finally:  # whatever happened, nothing of this prompt keeps running (a stray pump would keep the lease alive)
            await _cancel(stream)
            await _cancel(pump)

    async def _run(self, pump: asyncio.Task[None], stream: asyncio.Task[Done]) -> Outcome:
        try:
            await asyncio.wait({pump, stream}, return_when=asyncio.FIRST_COMPLETED)
            if not stream.done():  # the pump stopped first: a stale claim or Convex out of reach
                await _cancel(stream)  # closes the gateway stream, so the engine stops generating
                raise pump.exception() or StaleClaim(self.id)
            done = stream.result()
        except GatewayBusy as exc:
            return await self._settle(pump, "release", "busy", str(exc), Outcome.busy)
        except GatewayLost as exc:
            return await self._settle(pump, "release", "gateway_lost", str(exc), Outcome.lost)
        except GatewayRefused as exc:
            return await self._settle(pump, "fail", str(exc), str(exc), Outcome.failed)
        except StaleClaim:
            log.info("prompt %s: claim is stale (requeued elsewhere); dropped", self.id)
            return Outcome.stale
        except RoomApiError as exc:
            log.warning("prompt %s: convex unreachable mid-answer (%s); the lease will requeue it", self.id, exc)
            return Outcome.convex_down
        except asyncio.CancelledError:
            await _cancel(stream)
            await _cancel(pump)
            await self._release_on_shutdown()
            raise
        except Exception as exc:  # noqa: BLE001 - an engine reply we cannot read: retry once, like a lost gateway
            log.exception("prompt %s: unexpected error while streaming", self.id)
            return await self._settle(pump, "release", "gateway_lost", type(exc).__name__, Outcome.lost)
        await self._halt(pump)
        return await self._finish(done)

    async def _halt(self, pump: asyncio.Task[None]) -> None:
        """Stop the pump between appends (never mid-request), so no delta it already took is lost."""
        async with self._lock:
            pump.cancel()
        await _cancel(pump)

    async def _stream(self) -> Done:
        async with aclosing(self._gw.chat(self._text)) as events:
            async for ev in events:
                if isinstance(ev, Delta):
                    self._content.append(ev.content)
                    self._reasoning.append(ev.reasoning)
                    self._live = ev.live_tps or self._live
                elif isinstance(ev, Done):
                    return ev
        raise GatewayLost("the gateway stream ended without a final event")

    async def _pump(self) -> None:
        """Every flush_s: send what arrived; send an empty append when idle for the keepalive (a long prefill)."""
        while True:
            await asyncio.sleep(self._cfg.flush_s)
            await self._flush(keepalive=self._clock() - self._last_sent >= self._keepalive_s)

    async def _flush(self, keepalive: bool) -> None:
        async with self._lock:
            taken_c, taken_r = len(self._content), len(self._reasoning)
            content, reasoning = "".join(self._content), "".join(self._reasoning)
            if not (content or reasoning or keepalive):
                return
            ok = await self._api.append(self.id, self.claim_seq, content, self._sent, reasoning,
                                        self._sent_reasoning, self._live)
            # Drop only what was sent: deltas the stream added during the request stay for the next append.
            self._content, self._reasoning = self._content[taken_c:], self._reasoning[taken_r:]
            self._sent += len(content)
            self._sent_reasoning += len(reasoning)
            self._last_sent = self._clock()
            if not ok:
                raise StaleClaim(self.id)

    async def _finish(self, done: Done) -> Outcome:
        async with self._lock:
            tail, reasoning = "".join(self._content), "".join(self._reasoning)
            stats = {"tokens": done.tokens, "promptTokens": done.prompt_tokens, "decodeTps": done.decode_tps,
                     "prefillTps": done.prefill_tps, "finishReason": done.finish_reason}
            try:
                ok = await self._api.finish(self.id, self.claim_seq, tail, self._sent, reasoning,
                                            self._sent_reasoning, stats)
            except RoomApiError as exc:
                log.warning("prompt %s: finish not delivered (%s); the lease will requeue it", self.id, exc)
                return Outcome.convex_down
        if not ok:
            return Outcome.stale
        rate = f"{done.decode_tps:.2f} tok/s" if done.decode_tps else "no engine timings"
        log.info("prompt %s: done, %s tokens, %s", self.id, done.tokens, rate)
        return Outcome.done

    async def _settle(self, pump: asyncio.Task[None], how: str, arg: str, why: str, outcome: Outcome) -> Outcome:
        await self._halt(pump)
        log.info("prompt %s: %s (%s)", self.id, outcome.value, why)
        try:
            ok = (await self._api.release(self.id, self.claim_seq, arg) if how == "release"
                  else await self._api.fail(self.id, self.claim_seq, arg))
        except RoomApiError as exc:
            log.warning("prompt %s: could not %s (%s); the lease will requeue it", self.id, how, exc)
            return Outcome.convex_down
        return outcome if ok else Outcome.stale

    async def _release_on_shutdown(self) -> None:
        try:
            await asyncio.wait_for(self._api.release(self.id, self.claim_seq, "shutdown"), timeout=5.0)
        except (RoomApiError, TimeoutError, asyncio.CancelledError):
            log.warning("prompt %s: not released on shutdown; its lease will requeue it", self.id)


async def _cancel(task: asyncio.Task[Any]) -> None:
    """Cancel a helper task and wait until it has stopped. asyncio.wait never raises the helper's own exception or
    cancellation into the caller, so cleanup during a shutdown still runs; a cancellation of the caller still does.
    What the helper raised was already handled through the other task."""
    if not task.done():
        task.cancel()
    await asyncio.wait({task})
    if not task.cancelled() and task.exception() is not None:
        log.debug("helper task %s ended with %s", task.get_name(), type(task.exception()).__name__)
