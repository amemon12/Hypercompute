"""Lost replies, retries and surprises: the bridge must neither double text nor leave anything running."""

from __future__ import annotations

import asyncio
import codecs
from pathlib import Path

import httpx
from room_bridge.config import BridgeConfig, read_env_file
from room_bridge.gateway import Gateway
from room_bridge.room_api import RoomApi
from room_bridge.runner import Outcome, PromptRun

from .conftest import SECRET, FakeGateway, FakeRoom, chunk, final, make_cfg, prompt, sse, streamed, wire


def run(coro):  # noqa: ANN001, ANN201 - test helper
    return asyncio.run(coro)


def retrying_api(cfg: BridgeConfig, room: FakeRoom) -> RoomApi:
    return RoomApi(cfg, httpx.AsyncClient(transport=httpx.MockTransport(room.handler)), "b", retry_delays=(0.01,))


def test_appends_carry_offsets_that_add_up_to_the_answer() -> None:
    cfg = make_cfg()
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("Five", 1, 1.0), chunk(" machines", 2, 1.0), chunk(" hold it.", 3, 1.0),
                              final(3, 1.0, 2.0), sse("[DONE]")))
    api, gateway = wire(cfg, room, gw)
    assert run(PromptRun(cfg, api, gateway, prompt()).run()) is Outcome.done
    sent = 0
    for body in room.bodies("append"):
        assert body["offset"] == sent
        sent += len(body["delta"])
    fin = room.bodies("finish")[0]
    assert fin["offset"] == sent and sent + len(fin["tail"]) == len("Five machines hold it.")


def test_a_retried_append_resends_the_same_offset() -> None:
    cfg = make_cfg()
    room = FakeRoom(fail_next={"append": [httpx.Response(500, json={"error": "temporary failure; retry"})]})
    gw = FakeGateway(streamed(chunk("Hello", 1, 1.0), final(1, 1.0, 2.0), sse("[DONE]"), delay=0.05))
    gateway = Gateway(cfg, httpx.AsyncClient(transport=httpx.MockTransport(gw.handler)))
    assert run(PromptRun(cfg, retrying_api(cfg, room), gateway, prompt()).run()) is Outcome.done
    first, retry = room.bodies("append")[:2]
    assert first == retry  # same delta, same offset: Convex writes it over itself


def test_a_retried_claim_reuses_its_claim_key() -> None:
    cfg = make_cfg()
    room = FakeRoom(prompts=[prompt()], fail_next={"claim": [httpx.Response(502)]})
    got = run(retrying_api(cfg, room).claim())
    assert got is not None and got["id"] == "p1"
    first, retry = room.bodies("claim")
    assert first["claimKey"] == retry["claimKey"] and len(first["claimKey"]) == 32


def test_an_unexpected_stream_error_releases_and_stops_the_pump() -> None:
    cfg = make_cfg()
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("Half", 1, 1.0), error=RuntimeError("engine sent garbage")))
    api, gateway = wire(cfg, room, gw)

    async def go() -> tuple[Outcome, int, int]:
        outcome = await PromptRun(cfg, api, gateway, prompt()).run()
        before = len(room.calls)
        await asyncio.sleep(0.2)  # a stray pump would keep appending (and keep the lease alive)
        return outcome, before, len(room.calls)

    outcome, before, after = run(go())
    assert outcome is Outcome.lost
    assert room.bodies("release")[0]["reason"] == "gateway_lost"
    assert before == after


def test_a_non_json_reply_from_convex_is_an_api_error_not_a_crash() -> None:
    cfg = make_cfg()
    room = FakeRoom(fail_next={"append": [httpx.Response(200, text="<html>proxy page</html>")]})
    gw = FakeGateway(streamed(chunk("Hello", 1, 1.0), final(1, 1.0, 2.0), sse("[DONE]"), delay=0.05))
    api, gateway = wire(cfg, room, gw)
    assert run(PromptRun(cfg, api, gateway, prompt()).run()) is Outcome.convex_down


def test_keepalive_follows_a_short_lease() -> None:
    cfg = make_cfg(keepalive_s=20.0)
    room = FakeRoom()
    gw = FakeGateway(streamed(chunk("late", 1, 1.0), final(1, 1.0, 2.0), sse("[DONE]"), delay=1.3))
    api, gateway = wire(cfg, room, gw)
    short = {**prompt(), "leaseMs": 3000}  # HC_LEASE_MS=3000: keepalive must come within 1 s, not 20 s
    assert run(PromptRun(cfg, api, gateway, short).run()) is Outcome.done
    assert any(b["delta"] == "" for b in room.bodies("append"))


def test_env_files_written_by_powershell_are_read(tmp_path: Path) -> None:
    text = f"HC_CONVEX_SITE_URL=https://x.convex.site\r\nHC_ROOM_SECRET={SECRET}\r\nHC_ROOM_CODE=calhacks\r\n"
    for name, data in [("utf16.env", codecs.BOM_UTF16_LE + text.encode("utf-16-le")),
                       ("bom.env", codecs.BOM_UTF8 + text.encode("utf-8"))]:
        f = tmp_path / name
        f.write_bytes(data)
        cfg = BridgeConfig.from_env(read_env_file(f))
        assert cfg is not None and cfg.room_code == "calhacks" and cfg.secret == SECRET
