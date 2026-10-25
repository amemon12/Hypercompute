"""Used only when the bridge runs outside the Hypercompute venv: the subset of hypercompute.sse.SSEParser it needs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    data: str


class SSEParser:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        events: list[SSEEvent] = []
        while "\n\n" in self._buf:
            block, self._buf = self._buf.split("\n\n", 1)
            data = [line[5:].removeprefix(" ") for line in block.split("\n") if line.startswith("data:")]
            if data:
                events.append(SSEEvent("\n".join(data)))
        return events
