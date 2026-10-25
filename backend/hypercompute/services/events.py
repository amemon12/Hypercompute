"""In-process event bus feeding the dashboard's activity stream (SSE) and an audit JSONL."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator

from ..schemas import Event, utcnow
from ..storage import Store

log = logging.getLogger("hypercompute.events")


class EventBus:
    def __init__(self, store: Store | None = None, capacity: int = 500) -> None:
        self._store = store
        self._events: deque[Event] = deque(maxlen=capacity)
        self._seq = 0
        self._cond: asyncio.Condition | None = None

    def bind_loop(self) -> None:
        """Create the condition inside the running loop (call from app startup)."""
        self._cond = asyncio.Condition()

    def publish(self, type_: str, message: str, **data: object) -> Event:
        self._seq += 1
        event = Event(seq=self._seq, at=utcnow(), type=type_, message=message, data=dict(data))
        self._events.append(event)
        log.info("%s: %s", type_, message)
        if self._store is not None:
            self._store.append_jsonl("events", event.model_dump(mode="json"))
        cond = self._cond
        if cond is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(self._notify(cond))
        return event

    @staticmethod
    async def _notify(cond: asyncio.Condition) -> None:
        async with cond:
            cond.notify_all()

    def since(self, seq: int) -> list[Event]:
        return [e for e in self._events if e.seq > seq]

    async def stream(self, since: int, heartbeat_s: float = 15.0) -> AsyncIterator[Event | None]:
        """Yield events after `since`; yield None as a keep-alive tick when idle."""
        last = since
        while True:
            pending = self.since(last)
            for event in pending:
                last = event.seq
                yield event
            if pending:
                continue
            cond = self._cond
            if cond is None:
                await asyncio.sleep(heartbeat_s)
                yield None
                continue
            try:
                async with cond:
                    await asyncio.wait_for(cond.wait(), timeout=heartbeat_s)
            except TimeoutError:
                yield None
