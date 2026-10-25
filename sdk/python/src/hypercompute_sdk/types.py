"""The one type the SDK adds. Everything else is the gateway's JSON as plain dicts and lists."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatChunk:
    """One event of a streamed chat, as llama.cpp sends it through the gateway.

    ``content`` and ``reasoning`` are this event's text (either may be empty). ``timings`` is llama.cpp's timings
    object, usually on the last chunk. ``raw`` is the parsed event.
    """

    content: str
    reasoning: str
    finish_reason: str | None
    timings: dict[str, Any] | None
    usage: dict[str, Any] | None
    raw: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ChatChunk:
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
        delta = choice.get("delta")
        delta = delta if isinstance(delta, Mapping) else {}
        timings, usage = payload.get("timings"), payload.get("usage")
        finish = choice.get("finish_reason")
        return cls(content=str(delta.get("content") or ""),
                   reasoning=str(delta.get("reasoning_content") or ""),
                   finish_reason=None if finish is None else str(finish),
                   timings=dict(timings) if isinstance(timings, Mapping) else None,
                   usage=dict(usage) if isinstance(usage, Mapping) else None,
                   raw=dict(payload))
