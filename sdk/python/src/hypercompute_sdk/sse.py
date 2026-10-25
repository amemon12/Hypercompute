"""Incremental Server-Sent Events parser for the gateway's chat stream.

Bytes are buffered until a blank line ends an event (LF, CRLF or CR line endings). Lines starting with ``:`` are
comments. Multi-line ``data:`` fields are joined with a newline. One network chunk may carry zero, one or many events.
Only the ``data`` of each event is returned, since the chat stream uses nothing else.
"""

from __future__ import annotations

MAX_EVENT_BYTES = 4 * 1024 * 1024
_SEPARATORS = (b"\r\n\r\n", b"\n\n", b"\r\r")


class SSEParser:
    def __init__(self, max_event_bytes: int = MAX_EVENT_BYTES) -> None:
        self._buf = bytearray()
        self._max = max_event_bytes

    def feed(self, chunk: bytes) -> list[str]:
        """The data of every event this chunk completes."""
        self._buf.extend(chunk)
        out: list[str] = []
        while True:
            end, sep_len = self._boundary()
            if end < 0:
                if len(self._buf) > self._max:
                    raise ValueError("an SSE event is over the size limit")
                return out
            block = bytes(self._buf[:end])
            del self._buf[: end + sep_len]
            data = _data(block)
            if data is not None:
                out.append(data)

    def flush(self) -> list[str]:
        """The data of a last event that ended without a blank line (the stream closed early)."""
        block = bytes(self._buf)
        self._buf.clear()
        data = _data(block) if block.strip() else None
        return [] if data is None else [data]

    def _boundary(self) -> tuple[int, int]:
        best, best_len = -1, 0
        for sep in _SEPARATORS:
            i = self._buf.find(sep)
            if i >= 0 and (best < 0 or i < best):
                best, best_len = i, len(sep)
        return best, best_len


def _data(block: bytes) -> str | None:
    text = block.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    data: list[str] = []
    for line in text.split("\n"):
        if not line or line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    return "\n".join(data) if data else None
