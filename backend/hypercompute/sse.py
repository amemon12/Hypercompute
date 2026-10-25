"""Incremental Server-Sent Events parser that tolerates arbitrary network fragmentation.

Bytes are buffered until a blank line ends an event (LF, CRLF or CR line endings). Multi-line `data:`
fields are joined with '\\n' per the SSE spec. One network chunk may carry zero, one or many events;
one event is not one token.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    raw: bytes  # exact bytes of the event including its terminating blank line
    event: str | None
    data: str
    id: str | None


class SSEParser:
    def __init__(self, max_event_bytes: int = 4 * 1024 * 1024) -> None:
        self._buf = bytearray()
        self._max = max_event_bytes

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf.extend(chunk)
        events: list[SSEEvent] = []
        while True:
            end, sep_len = self._find_boundary()
            if end < 0:
                if len(self._buf) > self._max:
                    raise ValueError("SSE event exceeds size limit")
                return events
            raw = bytes(self._buf[: end + sep_len])
            del self._buf[: end + sep_len]
            ev = self._parse(raw[:end])
            if ev is not None:
                events.append(SSEEvent(raw=raw, event=ev[0], data=ev[1], id=ev[2]))

    def flush(self) -> list[SSEEvent]:
        """Parse a trailing event that ended without a blank line (upstream closed early)."""
        if not self._buf.strip():
            self._buf.clear()
            return []
        raw = bytes(self._buf)
        self._buf.clear()
        ev = self._parse(raw)
        return [SSEEvent(raw=raw + b"\n\n", event=ev[0], data=ev[1], id=ev[2])] if ev else []

    def _find_boundary(self) -> tuple[int, int]:
        buf = self._buf
        best, best_len = -1, 0
        for sep in (b"\r\n\r\n", b"\n\n", b"\r\r"):
            i = buf.find(sep)
            if i >= 0 and (best < 0 or i < best):
                best, best_len = i, len(sep)
        return best, best_len

    @staticmethod
    def _parse(block: bytes) -> tuple[str | None, str, str | None] | None:
        text = block.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        event: str | None = None
        ev_id: str | None = None
        data: list[str] = []
        for line in text.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event = value
            elif field == "id":
                ev_id = value
        if not data and event is None:
            return None
        return event, "\n".join(data), ev_id
