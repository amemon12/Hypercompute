"""Settings validation, the pool snapshot, and the service's claim gate."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from room_bridge.config import BridgeConfig, ConfigError, read_env_file
from room_bridge.gateway import Gateway, GatewayState
from room_bridge.pool import build_pool, layer_range
from room_bridge.room_api import RoomApi
from room_bridge.service import BridgeService

from .conftest import SECRET, FakeRoom, make_cfg

ENV = {"HC_CONVEX_SITE_URL": "https://reliable-dogfish-865.convex.site", "HC_ROOM_SECRET": SECRET,
       "HC_ROOM_CODE": "calhacks"}


def test_nothing_runs_without_the_three_variables() -> None:
    assert BridgeConfig.from_env({}) is None
    with pytest.raises(ConfigError, match="missing HC_ROOM_SECRET"):
        BridgeConfig.from_env({**ENV, "HC_ROOM_SECRET": ""})


@pytest.mark.parametrize(("key", "value", "match"), [
    ("HC_CONVEX_SITE_URL", "http://reliable-dogfish-865.convex.site", "https"),
    ("HC_CONVEX_SITE_URL", "https://reliable-dogfish-865.convex.cloud", "HTTP actions URL"),
    ("HC_CONVEX_SITE_URL", "https://x.convex.site/bridge", "no path"),
    ("HC_ROOM_SECRET", "short", "24 characters"),
    ("HC_ROOM_CODE", "Cal Hacks!", "HC_ROOM_CODE"),
    ("HC_GATEWAY_URL", "http://10.0.0.5:8000", "127.0.0.1"),
    ("HC_GATEWAY_URL", "https://127.0.0.1:8000", "127.0.0.1"),
    ("HC_ROOM_MAX_TOKENS", "9000", "between"),
])
def test_unsafe_settings_are_refused_without_echoing_values(key: str, value: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match) as info:
        BridgeConfig.from_env({**ENV, key: value})
    assert SECRET not in str(info.value)


def test_good_settings_and_env_file(tmp_path: Path) -> None:
    f = tmp_path / "bridge.env"
    f.write_bytes(b"# comment\r\nHC_CONVEX_SITE_URL=https://reliable-dogfish-865.convex.site/\r\n"
                  b"export HC_ROOM_SECRET='" + SECRET.encode() + b"'\r\nHC_ROOM_CODE=calhacks\r\n"
                  b"HC_GATEWAY_URL=http://127.0.0.1:8799\r\nHC_ROOM_SLOTS=2\r\n")
    cfg = BridgeConfig.from_env(read_env_file(f))
    assert cfg is not None
    assert cfg.site_url == "https://reliable-dogfish-865.convex.site"
    assert cfg.secret == SECRET and cfg.gateway_url == "http://127.0.0.1:8799" and cfg.slots == 2
    assert SECRET not in repr(cfg)
    assert BridgeConfig.from_env({**ENV, "HC_CONVEX_SITE_URL": "http://127.0.0.1:3211"}) is not None


def test_layer_range() -> None:
    assert layer_range([]) is None
    assert layer_range([3]) == "3"
    assert layer_range([0, 1, 2, 5, 6, 9]) == "0-2, 5-6, 9"


NODES = [
    {"id": "local", "display_name": "gx10", "is_local": True, "state": "eligible",
     "inventory": {"os": {"system": "linux"}, "devices": [{"id": "CUDA0"}]}},
    {"id": "a" * 32, "display_name": "g14", "is_local": False, "state": "eligible",
     "inventory": {"os": {"system": "windows"}, "devices": [{"id": "CUDA0"}]}},
    {"id": "b" * 32, "display_name": "mac", "is_local": False, "state": "eligible",
     "inventory": {"os": {"system": "macos"}, "devices": [{"id": "MTL0"}]}, "contribution": {"devices": ["MTL0"]}},
    {"id": "c" * 32, "display_name": "gone", "is_local": False, "state": "left", "inventory": {}},
]
DEP = {"id": "d1", "state": "ready", "purpose": "run", "created_at": "2026-10-25T01:00:00Z",
       "plan": {"model_id": "kimi-k2.6", "tier": "gpu", "engine_options": {"parallel": 2},
                "assignments": [
                    {"node_id": "local", "device": "CUDA0", "layers": list(range(0, 40)), "weight_bytes": 9e10,
                     "holds_output": False},
                    {"node_id": "a" * 32, "device": "CUDA0", "layers": list(range(40, 62)), "weight_bytes": 1e10,
                     "holds_output": True}]},
       "placement": {"offloaded": "62/62"}}
NET = {"transport": "iroh", "peers": [{"node_id": "a" * 32, "connected": True, "path": {"mode": "relay"}},
                                      {"node_id": "b" * 32, "connected": False, "path": None}]}


def test_pool_snapshot_maps_layers_paths_and_slots() -> None:
    view = build_pool(GatewayState(True, NODES, [DEP], NET))
    assert view.ready and view.slots == 2
    p = view.payload
    assert (p["modelId"], p["deploymentState"], p["offloaded"]) == ("kimi-k2.6", "ready", "62/62")
    rows = [(m["name"], m["device"], m["layers"], m["layerRange"], m["path"], m["holdsOutput"]) for m in p["machines"]]
    assert rows == [("gx10", "CUDA0", 40, "0-39", "local", False),
                    ("g14", "CUDA0", 22, "40-61", "relay", True),
                    ("mac", "MTL0", 0, None, "connecting", False)]  # joined but left out of the plan; "gone" hidden
    assert build_pool(GatewayState(True, NODES, [DEP], NET), slots_override=1).slots == 1


def test_pool_snapshot_without_a_ready_model() -> None:
    loading = {**DEP, "state": "loading"}
    assert not build_pool(GatewayState(True, NODES, [loading], NET)).ready
    tune = {**DEP, "purpose": "tune"}
    view = build_pool(GatewayState(True, NODES, [tune], NET))
    assert not view.ready and view.payload["deploymentState"] == "no model loaded"
    offline = build_pool(GatewayState(False, [], [], {}))
    assert not offline.ready and offline.payload["deploymentState"] == "gateway offline"
    lan = build_pool(GatewayState(True, NODES, [DEP], {"transport": "lan"}))
    assert [m["path"] for m in lan.payload["machines"]][:2] == ["local", "lan"]


def test_service_claims_only_while_the_gateway_is_ready() -> None:
    cfg = make_cfg()
    state = {"deps": [DEP]}

    async def gw_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/nodes":
            return httpx.Response(200, json=NODES)
        if request.url.path == "/api/deployments":
            return httpx.Response(200, json=state["deps"])
        if request.url.path == "/api/network":
            return httpx.Response(200, json=NET)
        return httpx.Response(404)

    room = FakeRoom()

    async def go() -> list[bool]:
        api = RoomApi(cfg, httpx.AsyncClient(transport=httpx.MockTransport(room.handler)), "b", retry_delays=())
        gw = Gateway(cfg, httpx.AsyncClient(transport=httpx.MockTransport(gw_handler)))
        service = BridgeService(cfg, api, gw)
        await service.sync_pool_once(0.0)
        ready = [service.may_claim(0), service.may_claim(1), service.may_claim(2)]
        state["deps"] = [{**DEP, "state": "loading"}]
        await service.sync_pool_once(1.0)
        ready.append(service.may_claim(0))
        await service.sync_pool_once(2.0)  # unchanged and within the heartbeat: not resent
        await service.sync_pool_once(30.0)  # heartbeat: resent so the room shows the bridge online
        return ready

    assert asyncio.run(go()) == [True, True, False, False]
    assert room.routes() == ["pool", "pool", "pool"]
    assert room.bodies("pool")[1]["deploymentState"] == "loading"
