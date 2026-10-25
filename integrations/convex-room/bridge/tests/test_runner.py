"""The prompt runner against a fake Convex room and a fake gateway: route order, release paths, stale claims."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from room_bridge.runner import Outcome, PromptRun

from .conftest import OWNER_TOKEN, SECRET, FakeGateway, FakeRoom, chunk, final, make_cfg, prompt, sse, streamed, wire


def run(coro):  # noqa: ANN001, ANN201 - test helper
    return asyncio.run(coro)


async def serve(cfg, room: FakeRoom, gw: FakeGateway) -> Outcome:  # noqa: ANN001
    api, gateway = wire(cfg, room, gw)
    return await PromptRun(cfg, api, gateway, prompt()).run()


def test_claim_append_finish_order_and_engine_rates(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("Five", 1, 1.7), chunk(" machines", 2, 1.8), chunk(" hold it.", 3, 1.8),
                              final(3, 1.81, 3.3), sse("[DONE]")))
    assert run(serve(cfg, room, gw)) is Outcome.done

    routes = room.routes()
    assert routes[-1] == "finish" and set(routes[:-1]) <= {"append"}
    appended = "".join(b["delta"] for b in room.bodies("append"))
    finished = room.bodies("finish")[0]
    assert appended + finished["tail"] == "Five machines hold it."
    assert finished["decodeTps"] == 1.81 and finished["prefillTps"] == 3.3  # the engine's timings, not a count
    assert finished["tokens"] == 3 and finished["promptTokens"] == 20 and finished["finishReason"] == "length"
    assert all(b["claimSeq"] == 1 for _, b in room.calls)
    assert any(b.get("liveTps") == 1.8 for b in room.bodies("append"))

    body = gw.chats[0]
    assert body["stream"] is True and body["max_tokens"] == 48 and body["timings_per_token"] is True
    assert body["messages"][-1] == {"role": "user", "content": "how big is the pool?"}


def test_http_429_releases_without_spending_an_attempt(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(lambda _r: httpx.Response(429, json={"detail": "all slots busy", "code": "busy"}))
    assert run(serve(cfg, room, gw)) is Outcome.busy
    assert room.routes() == ["release"]
    assert room.bodies("release")[0]["reason"] == "busy"


def test_busy_error_event_before_output_releases_busy(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(sse({"error": {"code": "busy", "message": "all 1 inference slot(s) are busy"}})))
    assert run(serve(cfg, room, gw)) is Outcome.busy
    assert room.bodies("release")[0]["reason"] == "busy"


def test_no_ready_deployment_is_busy_not_a_failure(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(lambda _r: httpx.Response(503, json={"detail": "no ready deployment"}))
    assert run(serve(cfg, room, gw)) is Outcome.busy


def test_gateway_killed_mid_answer_releases_as_gateway_lost(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("Half", 1, 1.5), chunk(" an", 2, 1.5),
                              error=httpx.RemoteProtocolError("peer closed connection")))
    assert run(serve(cfg, room, gw)) is Outcome.lost
    assert room.routes()[-1] == "release"
    assert room.bodies("release")[0]["reason"] == "gateway_lost"
    assert "finish" not in room.routes()


def test_stream_that_stops_without_finishing_is_lost(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("Half", 1, 1.5)))
    assert run(serve(cfg, room, gw)) is Outcome.lost


def test_interrupted_error_event_after_output_is_lost(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("Half", 1, 1.5),
                              sse({"error": {"code": "interrupted", "message": "engine connection closed"}})))
    assert run(serve(cfg, room, gw)) is Outcome.lost


def test_gateway_refusal_fails_the_prompt(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(lambda _r: httpx.Response(400, json={"detail": "'messages' must be a non-empty list"}))
    assert run(serve(cfg, room, gw)) is Outcome.failed
    assert room.routes() == ["fail"]
    assert "messages" in room.bodies("fail")[0]["error"]


def test_stale_claim_stops_streaming_and_never_finishes(cfg) -> None:  # noqa: ANN001
    room = FakeRoom(stale_after=0)  # the very first append is refused: the lease moved on
    parts = [chunk(f" w{i}", i + 1, 2.0) for i in range(40)]
    gw = FakeGateway(streamed(*parts, final(40, 2.0, 5.0), sse("[DONE]")))
    assert run(serve(cfg, room, gw)) is Outcome.stale
    assert "finish" not in room.routes()
    assert len(room.bodies("append")) == 1


def test_keepalive_appends_during_a_long_prefill() -> None:
    cfg = make_cfg(keepalive_s=0.05)
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("late", 1, 1.0), final(1, 1.0, 2.0), sse("[DONE]"), delay=0.3))
    assert run(serve(cfg, room, gw)) is Outcome.done
    appends = room.bodies("append")
    assert any(b["delta"] == "" for b in appends), "an empty append keeps the lease alive before the first token"


def test_reasoning_is_streamed_separately(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    think = sse({"choices": [{"index": 0, "delta": {"reasoning_content": "The user asks"}}]})
    gw = FakeGateway(streamed(think, chunk("Yes.", 1, 3.0), final(2, 3.0, 9.0), sse("[DONE]")))
    assert run(serve(cfg, room, gw)) is Outcome.done
    streamed_part = "".join(b.get("reasoning", "") for b in room.bodies("append"))
    reasoning = streamed_part + room.bodies("finish")[0]["reasoningTail"]
    assert reasoning == "The user asks"


def test_owner_token_is_refetched_after_a_gateway_restart(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("ok", 1, 2.0), final(1, 2.0, 3.0), sse("[DONE]")))
    api, gateway = wire(cfg, room, gw)

    async def go() -> list[Outcome]:
        first = await PromptRun(cfg, api, gateway, prompt()).run()
        gw.token = "rotated-owner-token-0123456789"  # noqa: S105 - the gateway restarted with a new token
        second = await PromptRun(cfg, api, gateway, prompt(claim_seq=2)).run()
        return [first, second]

    assert run(go()) == [Outcome.done, Outcome.done]
    assert gw.bootstrap_calls == 2


def test_shutdown_mid_answer_releases_the_prompt(cfg) -> None:  # noqa: ANN001
    room = FakeRoom()
    gw = FakeGateway(streamed(*[chunk(f" w{i}", i + 1, 2.0) for i in range(100)]))
    api, gateway = wire(cfg, room, gw)

    async def go() -> None:
        task = asyncio.create_task(PromptRun(cfg, api, gateway, prompt()).run())
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(go())
    assert room.routes()[-1] == "release"
    assert room.bodies("release")[0]["reason"] == "shutdown"


def test_secrets_never_reach_logs_or_bodies(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = make_cfg()
    paths = [
        streamed(chunk("a", 1, 1.0), final(1, 1.0, 1.0), sse("[DONE]")),
        lambda _r: httpx.Response(429, json={"detail": "busy"}),
        streamed(chunk("a", 1, 1.0), error=httpx.RemoteProtocolError("peer closed")),
        lambda _r: httpx.Response(400, json={"detail": "bad"}),
    ]
    rooms = []
    for chat in paths:
        room = FakeRoom()
        run(serve(cfg, room, FakeGateway(chat)))
        rooms.append(room)
    assert SECRET not in caplog.text and OWNER_TOKEN not in caplog.text
    assert SECRET not in repr(cfg)
    for room in rooms:
        assert room.auth_seen == {f"Bearer {SECRET}"}  # the secret goes only in the Authorization header
        assert all(OWNER_TOKEN not in str(body) for _, body in room.calls)  # the owner token never leaves
