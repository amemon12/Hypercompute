"""The tools as Droid sees them: served by droid_sdk.mcp's in-process server (loopback HTTP, bearer token) and called
with the MCP client library, the same protocol the droid CLI speaks."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from droid_sdk.mcp import create_sdk_mcp_server
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from autotune.gateway import PlanOnlyGateway
from autotune.guard import ToolGuard
from autotune.pool import snapshot
from autotune.runlog import RunStore
from autotune.service import ready_deployment
from autotune.tools import RunContext, build_tools
from mock_gateway.app import PEERS, Joined
from tests.conftest import MAC_A


async def context(app, client, store: RunStore, **guard) -> RunContext:  # noqa: ANN001, ANN003
    app.state.mock.peers["mac-a"] = Joined(PEERS["mac-a"], time.monotonic() - 10)
    run = store.create(runner="droid", trigger={"reason": "test"})
    dep = ready_deployment(await client.deployments())
    snap = snapshot(await client.nodes(), await client.network())
    return RunContext(run_id=run.id, store=store, gateway=PlanOnlyGateway(client), guard=ToolGuard(**guard),
                      snap=snap, model_id=dep["plan"]["model_id"], ctx_size=4096, current_plan=dep["plan"],
                      deployment_id=dep["id"], measured=None)


async def test_tools_over_the_sdk_mcp_server(mock_g14, store) -> None:  # noqa: ANN001
    app, client = mock_g14
    ctx = await context(app, client, store)
    server = create_sdk_mcp_server("hypercompute", build_tools(ctx))
    config = await server.start()
    try:
        assert config.url.startswith("http://127.0.0.1:")
        headers = {h.name: h.value for h in config.headers}
        async with httpx.AsyncClient() as raw:  # no bearer token, no tools
            assert (await raw.post(config.url, json={})).status_code == 401
        async with streamablehttp_client(config.url, headers=headers) as (read, write, _), \
                ClientSession(read, write) as mcp:
            await mcp.initialize()
            listed = {t.name for t in (await mcp.list_tools()).tools}
            assert listed == {"pool_snapshot", "current_layout", "check_layout"}
            pool = (await mcp.call_tool("pool_snapshot", {})).structuredContent
            assert {n["name"] for n in pool["nodes"]} == {"G14 (RTX 5070 Ti laptop)", "mac-a"}
            current = (await mcp.call_tool("current_layout", {})).structuredContent
            assert current["stages"] == [{"node_id": "local", "device": "CUDA0", "layers": 36, "residency": None}]
            ok = await mcp.call_tool("check_layout", {"label": "mac-a 4", "stages": [
                {"node_id": MAC_A, "device": "MTL0", "layers": 4}, {"node_id": "local", "device": "CUDA0",
                                                                   "layers": 32}]})
            v = ok.structuredContent
            assert v["accepted"] and v["plan_id"] and v["hops"] == 1 and v["checks_left"] == 7
            assert v["predicted_source"] == "moe" and v["predicted_decode_ms"] > 0
            refused = (await mcp.call_tool("check_layout", {"stages": [
                {"node_id": MAC_A, "device": "MTL0", "layers": 8}, {"node_id": "local", "device": "CUDA0",
                                                                   "layers": 28}]})).structuredContent
            assert not refused["accepted"] and refused["code"] == "layout_memory" and refused["fix"]
            bad = await mcp.call_tool("check_layout", {"stages": [{"node_id": "local", "device": "CUDA0",
                                                                   "layers": 36, "evil": True}]})
            assert bad.isError  # extra fields are refused before the handler runs
            sized = await mcp.call_tool("check_layout", {"ctx_size": 256, "stages": [
                {"node_id": "local", "device": "CUDA0", "layers": 36}]})
            assert sized.isError  # Droid cannot choose the context size its evidence is checked at
            bad_device = await mcp.call_tool("check_layout", {"stages": [
                {"node_id": "local", "device": "CUDA0; rm -rf /", "layers": 36}]})
            assert bad_device.isError
    finally:
        await server.close()
    assert len(ctx.verdicts) == 2  # the two invalid calls never reached the planner
    assert [s.kind for s in store.get(ctx.run_id).steps].count("verdict") == 2


async def test_the_cap_is_enforced_inside_the_tool(mock_g14, store) -> None:  # noqa: ANN001
    app, client = mock_g14
    ctx = await context(app, client, store, max_checks=1, max_calls=2)
    handlers = {t.name: t.handler for t in build_tools(ctx)}
    body = {"stages": [{"node_id": "local", "device": "CUDA0", "layers": 36}]}
    first = json.loads((await handlers["check_layout"](body)).content)
    second = json.loads((await handlers["check_layout"](body)).content)
    assert first["accepted"] and second["code"] == "guard_refused" and "cap" in second["reason"]
    assert len(ctx.verdicts) == 1


async def test_tools_have_no_way_to_write() -> None:
    allowed = {"nodes", "network", "deployments", "benchmarks", "check_plan"}
    public = {n for n in dir(PlanOnlyGateway) if not n.startswith("_")}
    assert public == allowed


@pytest.mark.parametrize("layers", [0, 5000])
async def test_out_of_range_layers_are_refused_before_the_gateway(mock_g14, store, layers: int) -> None:  # noqa: ANN001
    app, client = mock_g14
    ctx = await context(app, client, store)
    handler = {t.name: t.handler for t in build_tools(ctx)}["check_layout"]
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError from the SDK's argument model
        await handler({"stages": [{"node_id": "local", "device": "CUDA0", "layers": layers}]})
    assert ctx.verdicts == []
