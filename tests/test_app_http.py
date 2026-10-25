"""HTTP-level guards: owner-only endpoints, DNS-rebinding Host check, node bearer auth, 503 without a model."""

import asyncio
from types import SimpleNamespace

import pytest
from factories import contribution, engine_info, make_inventory
from fastapi.testclient import TestClient
from hypercompute.agent.supervisor import Supervisor
from hypercompute.app import BodyLimitMiddleware, create_app
from hypercompute.coordinator import Coordinator
from hypercompute.services.benchmarks import BenchmarkService
from hypercompute.services.deployments import DeploymentManager
from hypercompute.services.leases import LeaseManager
from hypercompute.services.models import ModelService
from hypercompute.services.nodes import NodeRegistry
from hypercompute.services.sessions import SessionService
from hypercompute.services.streaming import InferenceService
from test_sessions_and_join import join_request

OWNER = "owner-token-for-tests"


class NullAdapter:
    async def aclose(self):
        return None


def make_coord(store, events, tmp_path, monkeypatch) -> Coordinator:
    svc = SessionService(store, events, b"k" * 32, 8000, lambda: ["10.0.0.2"])
    reg = NodeRegistry(store, events, svc, b"k" * 32, engine_info(), make_inventory(), contribution(["CUDA0"]))

    async def probe(endpoints):
        from hypercompute.adapters.llama_rpc import ProbeResult

        return ProbeResult(False, [], {endpoints[0]: []}, "unreachable in tests")

    reg.probe = probe
    adapter = NullAdapter()
    sup = Supervisor(tmp_path / "m.json")
    models = ModelService(events, tmp_path / "models")
    deps = DeploymentManager(store, events, reg, models, LeaseManager(store), adapter, sup)
    inf = InferenceService(store, events, deps, models, adapter)
    coord = Coordinator(port=8000, owner_token=OWNER, store=store, events=events, sessions=svc, nodes=reg,
                        models=models, leases=LeaseManager(store), adapter=adapter, deployments=deps, inference=inf,
                        benchmarks=BenchmarkService(store, events, deps, inf), supervisor=sup,
                        engine_info=engine_info(), startup_notes=[])
    monkeypatch.setattr(Coordinator, "refresh_local", lambda self: _noop())
    monkeypatch.setattr(store, "close", lambda: None)
    return coord


def open_client(coord: Coordinator) -> TestClient:
    return TestClient(create_app(coord), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50123))


@pytest.fixture
def client(store, events, tmp_path, monkeypatch):
    with open_client(make_coord(store, events, tmp_path, monkeypatch)) as c:
        yield c


async def _noop():
    return None


def test_bootstrap_only_from_loopback_with_loopback_host(client):
    assert client.get("/api/owner/bootstrap").json()["owner_token"] == OWNER
    r = client.get("/api/owner/bootstrap", headers={"Host": "attacker.example:8000"})
    assert r.status_code == 403 and r.json()["code"] == "owner_only"


def test_owner_writes_need_token(client):
    assert client.post("/api/sessions", json={}).status_code == 403
    assert client.post("/api/sessions", json={}, headers={"X-Hypercompute-Owner": "wrong"}).status_code == 403
    r = client.post("/api/sessions", json={"name": "demo"}, headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 200 and r.json()["invite"]["join_command"].startswith("uv run hypercompute join")


def test_lan_health_is_minimal(client):
    body = client.get("/api/health", headers={"Host": "10.0.0.2:8000"}).json()
    assert body["engine"] == {"release": "b11770", "rpc_proto": "8.0.0"}


def test_chat_without_ready_model_is_503(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 503 and r.json()["code"] == "no_ready_deployment" and r.json()["fix"]


def test_chat_validates_messages(client):
    r = client.post("/v1/chat/completions", json={"messages": []}, headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 400


def test_node_endpoints_need_their_own_credential(client):
    code = client.post("/api/sessions", json={}, headers={"X-Hypercompute-Owner": OWNER}).json()["invite"]["code"]
    req = join_request(code, worker_ip="127.0.0.1").model_dump(mode="json")
    r = client.post("/api/sessions/join", json=req)
    assert r.status_code == 200 and r.json()["credential"]
    assert client.post("/api/sessions/join", json=req).json()["code"] == "code_used"
    hb = {"worker_state": "running", "sent_at": "2026-10-21T00:00:00Z"}
    r = client.post(f"/api/nodes/{'a' * 32}/heartbeat", json=hb)
    assert r.status_code in (401, 403)
    r = client.post(f"/api/nodes/{'a' * 32}/heartbeat", json=hb, headers={"Authorization": "Bearer forged"})
    assert r.status_code == 403


def test_agent_local_control_requires_token_and_loopback():
    from hypercompute.agent.control import TOKEN_HEADER, create_control_app

    left = []
    app = create_control_app("tok", lambda: {"state": "eligible"}, lambda: left.append(1), "10.0.0.2:8000")
    with TestClient(app, base_url="http://127.0.0.1:9", client=("127.0.0.1", 5)) as c:
        assert c.get("/status").json() == {"state": "eligible"}
        assert "Stop sharing" in c.get("/").text
        assert c.post("/leave").status_code == 403
        assert c.post("/leave", headers={TOKEN_HEADER: "nope"}).status_code == 403
        assert c.post("/leave", headers={TOKEN_HEADER: "tok"}).status_code == 200 and left == [1]
        assert c.get("/status", headers={"Host": "evil.example"}).status_code == 403


def test_agent_control_page_cannot_be_framed():
    from hypercompute.agent.control import create_control_app

    app = create_control_app("tok", lambda: {}, lambda: None, "10.0.0.2:8000")
    with TestClient(app, base_url="http://127.0.0.1:9", client=("127.0.0.1", 5)) as c:
        for r in (c.get("/"), c.get("/status"), c.post("/leave"), c.get("/", headers={"Host": "evil.example"})):
            assert r.headers["x-frame-options"] == "DENY"
            assert r.headers["content-security-policy"] == "frame-ancestors 'none'"


def test_agent_pause_resume_require_token():
    from hypercompute.agent.control import TOKEN_HEADER, create_control_app

    calls = []
    app = create_control_app("tok", lambda: {}, lambda: None, "h", pause=lambda: calls.append("p") or "paused",
                             resume=lambda: calls.append("r") or "resumed")
    with TestClient(app, base_url="http://127.0.0.1:9", client=("127.0.0.1", 5)) as c:
        assert c.post("/pause").status_code == 403 and calls == []
        assert c.post("/pause", headers={TOKEN_HEADER: "tok"}).json()["detail"] == "paused"
        assert c.post("/resume", headers={TOKEN_HEADER: "tok"}).json()["detail"] == "resumed"
        assert calls == ["p", "r"]


def test_health_has_time_and_bootstrap_uses_gateway_addresses(client):
    assert client.get("/api/health").json()["time"]
    assert client.get("/api/owner/bootstrap").json()["lan_addresses"] == ["10.0.0.2"]


# ---------------------------------------------------------------------- request body limits (finding 6)
def test_oversized_bodies_get_413_before_auth_or_parsing(client):
    big = b"x" * (64 * 1024 + 1)
    r = client.post("/api/sessions", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["code"] == "body_too_large"  # not 403: never parsed
    r = client.post("/api/sessions/join", content=b"{" + b" " * (256 * 1024) + b"}",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    r = client.post(f"/api/nodes/{'a' * 32}/heartbeat", content=b" " * (256 * 1024 + 1))
    assert r.status_code == 413

    def chunks():  # no Content-Length: the streamed size is counted
        for _ in range(5):
            yield b" " * (16 * 1024)

    assert client.post("/api/plans", content=chunks()).status_code == 413


def test_route_specific_limits_allow_normal_large_bodies(client):
    padding = "x" * (200 * 1024)  # over the 64 KiB default, under the 256 KiB node limit
    r = client.post("/api/sessions/join", json={"code": "ABCDE-FGHJK", "padding": padding})
    assert r.status_code == 422  # reached validation, not rejected for size
    messages = [{"role": "user", "content": "y" * (1024 * 1024)}]
    r = client.post("/v1/chat/completions", json={"messages": messages}, headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 503  # 1 MiB chat body passes the 16 MiB limit


def test_node_request_lists_and_version_are_capped():
    from hypercompute.schemas import ContributionLimits, Device, HeartbeatRequest, JoinRequest, utcnow
    from pydantic import ValidationError

    dev = Device(id="CUDA0", kind="gpu", name="g", pool_id="vram:CUDA0", source="t")
    with pytest.raises(ValidationError):
        HeartbeatRequest(worker_state="running", sent_at=utcnow(), devices=[dev] * 65)
    with pytest.raises(ValidationError):
        ContributionLimits(devices=["CUDA0"] * 65, max_bytes=1, reserve_bytes=0)
    req = join_request("ABCDE-FGHJK").model_dump()
    with pytest.raises(ValidationError):
        JoinRequest.model_validate({**req, "agent_version": "v" * 65})
    assert HeartbeatRequest(worker_state="running", sent_at=utcnow(), devices=[dev] * 8).devices


def test_gateway_server_limits_concurrent_connections():
    from fastapi import FastAPI
    from hypercompute import cli, config

    assert cli._gateway_config(FastAPI()).limit_concurrency == config.GATEWAY_MAX_CONNECTIONS


@pytest.mark.anyio
async def test_body_limit_stops_a_streamed_body_before_the_app_runs():
    called = []

    async def inner(scope, receive, send):
        called.append(scope["path"])

    chunks = [{"type": "http.request", "body": b"x" * 40_000, "more_body": True} for _ in range(3)]
    sent = []

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    await BodyLimitMiddleware(inner)({"type": "http", "path": "/api/plans", "headers": []}, receive, send)
    assert called == [] and sent[0]["status"] == 413


# ---------------------------------------------------------------------- anti-framing headers (finding 8)
def assert_security_headers(r):
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_security_headers_on_api_errors_and_dashboard(store, events, tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<!doctype html><script type="module" src="/assets/app.js"></script>'
                                     '<link rel="stylesheet" href="/assets/app.css">', encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    (dist / "assets" / "app.css").write_text("body{color:red}", encoding="utf-8")
    monkeypatch.setenv("HYPERCOMPUTE_FRONTEND_DIST", str(dist))
    with open_client(make_coord(store, events, tmp_path, monkeypatch)) as c:
        page = c.get("/")
        assert page.status_code == 200 and "app.js" in page.text
        assert_security_headers(page)
        js, css = c.get("/assets/app.js"), c.get("/assets/app.css")
        assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
        assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
        assert_security_headers(js)
        assert_security_headers(c.get("/api/health"))
        assert_security_headers(c.post("/api/sessions", json={}))  # 403
        assert_security_headers(c.post("/api/sessions", content=b"x" * (65 * 1024)))  # 413


# ---------------------------------------------------------------------- /v1 model name check
def ready_model(client, monkeypatch, model_id="qwen3-0.6b"):
    coord = client.app.state.coord
    dep = SimpleNamespace(id="dep1", plan=SimpleNamespace(model_id=model_id))
    monkeypatch.setattr(coord.deployments, "ready", lambda: (dep, None))

    async def complete(body):
        return 200, {"model": body.get("model"), "ok": True}

    monkeypatch.setattr(coord.inference, "chat_complete", complete)


def test_chat_rejects_a_model_that_is_not_loaded(client, monkeypatch):
    ready_model(client, monkeypatch)
    msgs = [{"role": "user", "content": "hi"}]
    r = client.post("/v1/chat/completions", json={"model": "llama-3-8b", "messages": msgs},
                    headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 404 and r.json()["code"] == "model_not_found"
    assert "qwen3-0.6b" in r.json()["detail"]
    for body in ({"messages": msgs}, {"model": "qwen3-0.6b", "messages": msgs}):
        r = client.post("/v1/chat/completions", json=body, headers={"X-Hypercompute-Owner": OWNER})
        assert r.status_code == 200 and r.json()["ok"]


# ---------------------------------------------------------------------- benchmark shutdown (finding 4)
def test_shutdown_records_a_running_benchmark_as_failed(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    dep = SimpleNamespace(id="dep1", placement=None, plan=SimpleNamespace(model_id="m", device_list=["CUDA0"],
                                                                          engine="llama_rpc", cpu_layers=0))
    monkeypatch.setattr(coord.deployments, "ready", lambda: (dep, None))

    async def hang(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(coord.inference, "collect", hang)
    with open_client(coord) as c:
        r = c.post("/api/benchmarks", json={}, headers={"X-Hypercompute-Owner": OWNER})
        assert r.status_code == 202 and r.json()["status"] == "running"
        bench_id = r.json()["id"]
    result = coord.benchmarks.get(bench_id)
    assert result.status == "failed" and "cancelled" in result.error
