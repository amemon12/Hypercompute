"""The bridge against the mock gateway app itself, so the mock and the bridge agree on the real wire shapes."""

from __future__ import annotations

import asyncio

import httpx
from mock_gateway import build_app
from room_bridge.gateway import Gateway
from room_bridge.pool import build_pool
from room_bridge.runner import Outcome, PromptRun

from .conftest import FakeRoom, make_cfg, prompt


def test_full_prompt_through_the_mock_gateway() -> None:
    cfg = make_cfg()
    room = FakeRoom()
    app = build_app(decode_tps=500.0, prefill_tps=5000.0, slots=1)

    async def go() -> Outcome:
        from room_bridge.room_api import RoomApi

        api = RoomApi(cfg, httpx.AsyncClient(transport=httpx.MockTransport(room.handler)), "b", retry_delays=())
        local = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 5000)))
        gw = Gateway(cfg, local)
        view = build_pool(await gw.state())
        assert view.ready and len(view.payload["machines"]) == 5
        assert view.payload["modelId"].startswith("mock-")
        return await PromptRun(cfg, api, gw, prompt()).run()

    assert asyncio.run(go()) is Outcome.done
    fin = room.bodies("finish")[0]
    assert fin["tokens"] and fin["decodeTps"] and fin["prefillTps"] == 5000.0
