"""SSE parsing under arbitrary fragmentation, using the real b11770 stream capture."""

import json

import pytest
from hypercompute.services.streaming import StreamStats
from hypercompute.sse import SSEParser


def _parse_all(chunks):
    p = SSEParser()
    out = []
    for c in chunks:
        out.extend(p.feed(c))
    out.extend(p.flush())
    return out


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 10_000])
def test_fragmentation_invariant(fixture_text, size):
    raw = fixture_text("llama_stream_b11770.sse").encode()
    whole = _parse_all([raw])
    pieces = _parse_all([raw[i:i + size] for i in range(0, len(raw), size)])
    assert [e.data for e in pieces] == [e.data for e in whole]
    assert b"".join(e.raw for e in pieces) == raw  # passthrough bytes are exact


def test_final_chunk_carries_usage_and_timings(fixture_text):
    events = _parse_all([fixture_text("llama_stream_b11770.sse").encode()])
    assert events[-1].data == "[DONE]"
    stats = StreamStats()
    for e in events[:-1]:
        stats.observe(json.loads(e.data))
    assert stats.finish_reason == "stop"
    assert stats.usage["completion_tokens"] == 9 and stats.usage["prompt_tokens"] == 35
    assert stats.timings["predicted_n"] == 9
    assert stats.ttft_ms is not None
    assert "".join(stats.text).strip()


def test_one_event_is_not_one_token(fixture_text):
    events = _parse_all([fixture_text("llama_stream_b11770.sse").encode()])
    content_events = [e for e in events if e.data != "[DONE]" and json.loads(e.data)["choices"]
                      and json.loads(e.data)["choices"][0]["delta"].get("content")]
    usage = json.loads(events[-2].data)["usage"]
    # token counts come from usage, which need not equal the number of content events
    assert usage["completion_tokens"] >= 1 and len(content_events) >= 1


def test_crlf_multiline_and_comments():
    events = _parse_all([b": keep-alive\r\n\r\n", b"event: hc\r\ndata: line1\r\ndata: line2\r\nid: 7\r\n\r\n"])
    assert len(events) == 1
    assert events[0].data == "line1\nline2" and events[0].event == "hc" and events[0].id == "7"


def test_unterminated_trailing_event_is_flushed():
    events = _parse_all([b'data: {"a": 1}'])
    assert events[0].data == '{"a": 1}'


def test_oversized_event_rejected():
    p = SSEParser(max_event_bytes=16)
    with pytest.raises(ValueError):
        p.feed(b"data: " + b"x" * 64)
