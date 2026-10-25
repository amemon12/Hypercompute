"""Server-sent events from the gateway, split into whole events so the proxy can see the first token.

Raw bytes are passed to the buyer unchanged; the parsed copy only decides when to settle and what to record.
The gateway's stream is llama-server's OpenAI chunks, ending with `data: [DONE]`, or an error object
`{"error": {"code": ..., "message": ...}}` (busy, interrupted, upstream_error) in place of more chunks.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

MAX_EVENT_BYTES = 1024 * 1024  # one event this large means the upstream is not speaking SSE


class SSEError(ValueError):
    """The upstream sent something that is not a well-formed event stream."""


@dataclass(frozen=True)
class Event:
    raw: bytes  # exactly what the gateway sent, separator included
    payload: dict | None = None
    done: bool = False  # the [DONE] sentinel
    error: dict | None = None  # the gateway's error object, if this event is one

    @property
    def text(self) -> str:
        """Generated text in this chunk: content plus reasoning text."""
        if not self.payload:
            return ""
        parts = []
        for choice in self.payload.get("choices") or []:
            delta = choice.get("delta") or {}
            parts.append((delta.get("content") or "") + (delta.get("reasoning_content") or ""))
        return "".join(parts)

    @property
    def finish_reason(self) -> str | None:
        for choice in (self.payload or {}).get("choices") or []:
            if choice.get("finish_reason"):
                return str(choice["finish_reason"])
        return None

    @property
    def usage(self) -> dict | None:
        usage = (self.payload or {}).get("usage")
        return usage if isinstance(usage, dict) else None


def parse_event(raw: bytes) -> Event:
    data_lines = [line[5:].lstrip() for line in raw.decode("utf-8", errors="replace").splitlines()
                  if line.startswith("data:")]
    if not data_lines:
        return Event(raw=raw)  # a comment or keep-alive
    data = "\n".join(data_lines)
    if data.strip() == "[DONE]":
        return Event(raw=raw, done=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise SSEError(f"event is not JSON: {data[:80]!r}") from exc
    if not isinstance(payload, dict):
        raise SSEError("event is not a JSON object")
    error = payload.get("error")
    return Event(raw=raw, payload=payload, error=error if isinstance(error, dict) else None)


@dataclass
class EventSplitter:
    """Feeds byte chunks in, yields whole events out. Accepts \\n\\n and \\r\\n\\r\\n separators."""

    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> list[Event]:
        self._buf.extend(chunk.replace(b"\r\n", b"\n"))
        events = []
        while True:
            cut = self._buf.find(b"\n\n")
            if cut < 0:
                break
            raw = bytes(self._buf[:cut + 2])
            del self._buf[:cut + 2]
            if raw.strip():
                events.append(parse_event(raw))
        if len(self._buf) > MAX_EVENT_BYTES:
            raise SSEError("event over 1 MiB without a separator")
        return events

    def leftover(self) -> bytes:
        return bytes(self._buf)


async def events(chunks: AsyncIterator[bytes]) -> AsyncIterator[Event]:
    splitter = EventSplitter()
    async for chunk in chunks:
        for event in splitter.feed(chunk):
            yield event
    if splitter.leftover().strip():
        yield parse_event(splitter.leftover() + b"\n\n")


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()
