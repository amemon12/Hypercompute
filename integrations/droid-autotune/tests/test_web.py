from __future__ import annotations

import time

import httpx

from autotune.runner import ScriptedRunner, sweep_calls
from autotune.service import AutotuneService
from autotune.web import create_app
from mock_gateway.app import PEERS, Joined

TOKEN = "t" * 32


def app_for(settings, store, mock_g14):  # noqa: ANN001, ANN201
    mock, client = mock_g14
    mock.state.mock.peers["mac-a"] = Joined(PEERS["mac-a"], time.monotonic() - 10)
    service = AutotuneService(settings, client, lambda: ScriptedRunner(sweep_calls), store, runner_name="scripted")
    return create_app(service, token=TOKEN, watch=False), service


def http(app, host: str = "127.0.0.1:8787", client: tuple[str, int] = ("127.0.0.1", 5000)) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=client), base_url=f"http://{host}")


async def test_page_carries_the_token_and_strict_headers(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, _ = app_for(settings, store, mock_g14)
    async with http(app) as c:
        r = await c.get("/")
    assert r.status_code == 200 and TOKEN in r.text
    assert r.headers["x-frame-options"] == "DENY" and "frame-ancestors 'none'" in r.headers["content-security-policy"]


async def test_actions_need_the_owner_token(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, _ = app_for(settings, store, mock_g14)
    async with http(app) as c:
        assert (await c.post("/api/trigger")).status_code == 403
        assert (await c.post("/api/trigger", headers={"X-Autotune-Owner": "wrong"})).status_code == 403
        assert (await c.post("/api/runs/x/approve", json={"index": 0})).status_code == 403


async def test_non_loopback_clients_and_rebound_hosts_are_refused(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, _ = app_for(settings, store, mock_g14)
    async with http(app, client=("192.168.1.50", 5000)) as c:
        assert (await c.get("/api/state")).status_code == 403
    async with http(app, host="evil.example:8787") as c:
        assert (await c.get("/")).status_code == 403
        assert (await c.post("/api/trigger", headers={"X-Autotune-Owner": TOKEN})).status_code == 403


async def test_trigger_run_and_reject_through_the_api(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, service = app_for(settings, store, mock_g14)
    owner = {"X-Autotune-Owner": TOKEN}
    async with http(app) as c:
        run_id = (await c.post("/api/trigger", headers=owner)).json()["id"]
        await service._run_task
        run = (await c.get(f"/api/runs/{run_id}")).json()
        assert run["state"] == "awaiting_approval" and run["runner"] == "scripted"
        assert "current_plan" not in run and run["steps"]
        state = (await c.get("/api/state")).json()
        assert state["runner"] == "scripted" and state["runs"][0]["id"] == run_id
        assert {n["name"] for n in state["pool"]} >= {"mac-a"}
        bad = await c.post(f"/api/runs/{run_id}/approve", headers=owner, json={"index": 99})
        assert bad.status_code == 422
        assert (await c.post(f"/api/runs/{run_id}/reject", headers=owner)).json()["state"] == "rejected"
        assert (await c.post(f"/api/runs/{run_id}/reject", headers=owner)).status_code == 409
        assert (await c.get("/api/runs/nope")).status_code == 404
