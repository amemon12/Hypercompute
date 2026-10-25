"""chat and stream: local checks, body size, SSE parsing, error events, early close, token rotation."""

from __future__ import annotations

import json

import httpx
import pytest
from fake_gateway import MODEL, SECOND_TOKEN, STREAMS, TIMINGS, FakeGateway, TrackedStream, delta, sse
from hypercompute_sdk import (
    BodyTooLarge,
    ChatChunk,
    ChatInterrupted,
    GatewayUnreachable,
    NotFoundError,
    OwnerAuthError,
    RequestError,
    StreamError,
    UnavailableError,
)
from hypercompute_sdk.client import CHAT_BODY_LIMIT, read_chunks

HI = [{"role": "user", "content": "hi"}]


def ready_gateway(**kwargs: object) -> FakeGateway:
    gw = FakeGateway(**kwargs)  # type: ignore[arg-type]
    gw.add_dep("ready")
    return gw


def chat_requests(gw: FakeGateway) -> list[httpx.Request]:
    return gw.calls("POST", "/v1/chat/completions")


# ----------------------------------------------------------------------------- chat


def test_chat_returns_the_completion() -> None:
    gw = ready_gateway()
    reply = gw.pool().chat(HI, model=MODEL, max_tokens=16, temperature=0, seed=7, cache_prompt=False)
    assert reply["choices"][0]["message"]["content"] == "Hello there"
    request = chat_requests(gw)[0]
    assert json.loads(request.content) == {"messages": HI, "stream": False, "model": MODEL, "max_tokens": 16,
                                           "temperature": 0, "seed": 7, "cache_prompt": False}
    assert request.headers["content-type"] == "application/json"
    assert request.extensions["timeout"] == {"connect": 30.0, "read": 1800.0, "write": 30.0, "pool": 30.0}


@pytest.mark.parametrize("messages", [[], "hi", None, [{"role": "user"}], [{"content": "x"}], ["hi"], ()])
def test_chat_refuses_bad_messages_without_sending(messages: object) -> None:
    gw = ready_gateway()
    with pytest.raises(ValueError):
        gw.pool().chat(messages)  # type: ignore[arg-type]
    assert gw.requests == []


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_tokens": True}, {"temperature": "hot"},
                                    {"temperature": True}, {"model": ""}, {"stream": True},
                                    {"seed": float("nan")}, {"seed": object()}])
def test_chat_refuses_bad_options_without_sending(kwargs: dict) -> None:
    gw = ready_gateway()
    with pytest.raises(ValueError):
        gw.pool().chat(HI, **kwargs)
    assert gw.requests == []


def test_chat_refuses_a_body_over_the_limit_without_sending() -> None:
    gw = ready_gateway()
    big = [{"role": "user", "content": "x" * CHAT_BODY_LIMIT}]
    with pytest.raises(BodyTooLarge) as info:
        gw.pool().chat(big)
    assert info.value.code == "body_too_large" and f"{CHAT_BODY_LIMIT // 1024} KiB" in str(info.value)
    with pytest.raises(BodyTooLarge):
        gw.pool().stream(big)
    assert gw.requests == []


def test_chat_just_under_the_limit_is_sent() -> None:
    gw = ready_gateway()
    overhead = len(json.dumps({"messages": [{"role": "user", "content": ""}], "stream": False},
                              separators=(",", ":")))
    gw.pool().chat([{"role": "user", "content": "x" * (CHAT_BODY_LIMIT - overhead)}])
    assert len(chat_requests(gw)[0].content) == CHAT_BODY_LIMIT


def test_chat_wrong_model_is_not_found() -> None:
    with pytest.raises(NotFoundError) as info:
        ready_gateway().pool().chat(HI, model="other-model")
    assert info.value.code == "model_not_found" and info.value.fix


def test_chat_with_no_ready_deployment_is_unavailable() -> None:
    with pytest.raises(UnavailableError) as info:
        FakeGateway().pool().chat(HI)
    assert info.value.code == "no_ready_deployment"


def test_chat_bad_request_from_gateway() -> None:
    gw = ready_gateway()
    gw.overrides[("POST", "/v1/chat/completions")] = lambda r: httpx.Response(
        400, json={"detail": "'messages' must be a non-empty list", "code": "bad_request"})
    with pytest.raises(RequestError):
        gw.pool(owner_token=gw.token).chat(HI)


def test_chat_interrupted_carries_partial_text() -> None:
    gw = ready_gateway(chat_mode="interrupted")
    with pytest.raises(ChatInterrupted) as info:
        gw.pool().chat(HI)
    err = info.value
    assert (err.status, err.run_id, err.partial_text, err.partial_truncated) == (502, "run1", "Hel", True)


def test_chat_retries_after_host_restart() -> None:
    gw = ready_gateway()
    pool = gw.pool()
    pool.chat(HI)
    gw.rotate()
    pool.chat(HI)
    assert [r.headers["x-hypercompute-owner"] for r in chat_requests(gw)][-1] == SECOND_TOKEN
    assert gw.bootstraps == 2


# ----------------------------------------------------------------------------- stream


def test_stream_yields_chunks_in_order_with_timings_last() -> None:
    gw = ready_gateway()
    chunks = list(gw.pool().stream(HI, max_tokens=8))
    assert [c.content for c in chunks] == ["", "Hel", "lo", ""]
    assert chunks[0].reasoning == "think"
    assert chunks[-1].finish_reason == "stop" and chunks[-1].timings == TIMINGS
    assert all(c.timings is None for c in chunks[:-1])
    assert all(isinstance(c, ChatChunk) for c in chunks)
    body = json.loads(chat_requests(gw)[0].content)
    assert body["stream"] is True and body["max_tokens"] == 8
    assert gw.streams[0].closed


def test_stream_is_lazy_but_checks_arguments_now() -> None:
    gw = ready_gateway()
    with pytest.raises(ValueError):
        gw.pool().stream([])
    iterator = gw.pool().stream(HI)
    assert gw.requests == []
    next(iterator)
    assert len(chat_requests(gw)) == 1


def test_stream_interrupted_raises_with_partial_text() -> None:
    gw = ready_gateway(chat_mode="interrupted")
    got: list[str] = []
    with pytest.raises(ChatInterrupted) as info:
        for chunk in gw.pool().stream(HI):
            got.append(chunk.content)
    err = info.value
    assert got == ["Hel"]
    assert (err.status, err.code, err.run_id, err.partial_text, err.partial_truncated) == (
        200, "interrupted", "run1", "Hel", False)
    assert gw.streams[0].closed


def test_stream_busy_raises_stream_error() -> None:
    gw = ready_gateway(chat_mode="busy")
    with pytest.raises(StreamError) as info:
        list(gw.pool().stream(HI))
    assert (info.value.code, info.value.detail) == ("busy", "the engine is busy")
    assert str(info.value) == "busy: the engine is busy"


def test_stream_cut_before_done_is_incomplete() -> None:
    gw = ready_gateway(chat_mode="cut")
    with pytest.raises(StreamError) as info:
        list(gw.pool().stream(HI))
    assert info.value.code == "incomplete"
    assert gw.streams[0].closed


def test_stream_with_finish_but_no_done_ends_cleanly() -> None:
    gw = ready_gateway(chat_mode="finish_no_done")
    chunks = list(gw.pool().stream(HI))
    assert [c.finish_reason for c in chunks] == ["length"]


def test_stream_event_that_is_not_json_is_an_error() -> None:
    gw = ready_gateway(chat_mode="not_json")
    with pytest.raises(StreamError) as info:
        list(gw.pool().stream(HI))
    assert info.value.code == "bad_event"


def test_stream_skips_payloads_that_are_not_objects() -> None:
    gw = ready_gateway(chat_mode="not_object")
    assert [c.content for c in gw.pool().stream(HI)] == ["ok"]


def test_stream_closes_the_response_on_early_break() -> None:
    gw = ready_gateway()
    for chunk in gw.pool().stream(HI):
        if chunk.content == "Hel":
            break
    stream = gw.streams[0]
    assert stream.closed
    assert stream.sent < len(STREAMS["normal"])


def test_stream_close_method_closes_the_response() -> None:
    gw = ready_gateway()
    iterator = gw.pool().stream(HI)
    next(iterator)
    iterator.close()  # type: ignore[attr-defined]
    assert gw.streams[0].closed


def test_stream_error_status_is_read_and_raised() -> None:
    with pytest.raises(UnavailableError):
        list(FakeGateway().pool().stream(HI))
    with pytest.raises(NotFoundError):
        list(ready_gateway().pool().stream(HI, model="other"))


def test_stream_retries_once_after_host_restart() -> None:
    gw = ready_gateway()
    pool = gw.pool()
    list(pool.stream(HI))
    gw.rotate()
    assert [c.content for c in pool.stream(HI)][1:3] == ["Hel", "lo"]
    assert [r.headers["x-hypercompute-owner"] for r in chat_requests(gw)][-1] == SECOND_TOKEN


def test_stream_with_argument_token_after_restart_raises() -> None:
    gw = ready_gateway()
    pool = gw.pool(owner_token=gw.token)
    gw.rotate()
    with pytest.raises(OwnerAuthError):
        list(pool.stream(HI))


def test_stream_connection_refused() -> None:
    gw = ready_gateway()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gw.overrides[("POST", "/v1/chat/completions")] = refuse
    with pytest.raises(GatewayUnreachable):
        list(gw.pool(owner_token=gw.token).stream(HI))


def test_stream_read_timeout_mid_stream_is_unreachable() -> None:
    gw = ready_gateway(stream_fail_after=3)
    got: list[str] = []
    with pytest.raises(GatewayUnreachable) as info:
        for chunk in gw.pool().stream(HI):
            got.append(chunk.content)
    assert got == ["", "Hel"]
    assert "ReadTimeout" in info.value.detail
    assert gw.streams[0].closed


def test_stream_error_body_that_fails_to_read() -> None:
    gw = ready_gateway()

    class Broken(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise httpx.ReadError("reset")
            yield b""  # pragma: no cover

    gw.overrides[("POST", "/v1/chat/completions")] = lambda r: httpx.Response(503, stream=Broken())
    with pytest.raises(GatewayUnreachable):
        list(gw.pool(owner_token=gw.token).stream(HI))


# ----------------------------------------------------------------------------- read_chunks directly


def test_read_chunks_error_shapes() -> None:
    with pytest.raises(StreamError) as info:
        list(read_chunks(sse({"error": "plain text"})))
    assert (info.value.code, info.value.detail) == ("stream_error", "plain text")
    with pytest.raises(StreamError) as info:
        list(read_chunks(sse({"error": {"code": "timeout"}})))
    assert (info.value.code, info.value.detail) == ("timeout", "the stream ended with an error")
    with pytest.raises(StreamError) as info:
        list(read_chunks(sse({"error": {"message": "lost"}})))
    assert info.value.code == "stream_error"


def test_read_chunks_trailing_event_without_blank_line() -> None:
    raw = b"data: " + json.dumps(delta("end", finish="stop")).encode()
    assert [c.finish_reason for c in read_chunks([raw])] == ["stop"]


def test_read_chunks_oversized_event() -> None:
    with pytest.raises(StreamError) as info:
        list(read_chunks([b"data: " + b"x" * (4 * 1024 * 1024 + 1)]))
    assert info.value.code == "bad_event"


def test_tracked_stream_helper_counts() -> None:
    stream = TrackedStream([b"data: 1\n\n"])
    assert b"".join(stream) == b"data: 1\n\n" and stream.sent == 1
