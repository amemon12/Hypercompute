"""The SSE parser and ChatChunk."""

from __future__ import annotations

import pytest
from hypercompute_sdk import ChatChunk
from hypercompute_sdk.sse import SSEParser


def test_events_across_any_fragmentation() -> None:
    raw = b": keep-alive\n\ndata: one\n\nevent: x\nid: 3\ndata: two\n\ndata: [DONE]\n\n"
    for size in (1, 2, 5, len(raw)):
        parser, out = SSEParser(), []
        for i in range(0, len(raw), size):
            out += parser.feed(raw[i:i + size])
        assert out == ["one", "two", "[DONE]"], size


def test_crlf_cr_and_multiline_data() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: a\r\ndata: b\r\n\r\ndata:c\r\rdata: d\n\n") == ["a\nb", "c", "d"]


def test_flush_returns_a_trailing_event() -> None:
    parser = SSEParser()
    assert parser.feed(b"data: tail") == []
    assert parser.flush() == ["tail"]
    assert parser.flush() == []


def test_flush_of_whitespace_or_comment_is_empty() -> None:
    parser = SSEParser()
    parser.feed(b"\n")
    assert parser.flush() == []
    parser.feed(b": only a comment")
    assert parser.flush() == []


def test_event_without_data_is_skipped() -> None:
    assert SSEParser().feed(b"event: ping\n\nretry: 5\n\n") == []


def test_size_limit() -> None:
    parser = SSEParser(max_event_bytes=8)
    with pytest.raises(ValueError):
        parser.feed(b"data: 0123456789")


def test_chat_chunk_from_payload() -> None:
    payload = {"choices": [{"delta": {"content": "hi", "reasoning_content": "hm"}, "finish_reason": "stop"}],
               "timings": {"predicted_n": 2}, "usage": {"total_tokens": 9}}
    chunk = ChatChunk.from_payload(payload)
    assert (chunk.content, chunk.reasoning, chunk.finish_reason) == ("hi", "hm", "stop")
    assert chunk.timings == {"predicted_n": 2} and chunk.usage == {"total_tokens": 9} and chunk.raw == payload
    with pytest.raises(AttributeError):
        chunk.content = "x"  # type: ignore[misc]


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": ["x"]}, {"choices": [{"delta": None}]},
                                     {"choices": [{"delta": {"content": None}}], "timings": "x", "usage": 3}])
def test_chat_chunk_tolerates_odd_payloads(payload: dict) -> None:
    chunk = ChatChunk.from_payload(payload)
    assert (chunk.content, chunk.reasoning, chunk.finish_reason, chunk.timings, chunk.usage) == (
        "", "", None, None, None)
