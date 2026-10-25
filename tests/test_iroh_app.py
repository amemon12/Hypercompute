"""Owner invitation routes select the active transport without starting networking."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from hypercompute.app import create_app
from test_app_http import OWNER, make_coord

TICKET = "test-public-endpoint-ticket"
OWNER_HEADERS = {"X-Hypercompute-Owner": OWNER}


@pytest.fixture
def app(store, events, tmp_path, monkeypatch):
    result = create_app(make_coord(store, events, tmp_path, monkeypatch), transport="iroh")
    result.state.peer_service = SimpleNamespace(ticket=TICKET, status=lambda: {"transport": "iroh", "peers": []})
    return result


@pytest.fixture
def client(app):
    # Deliberately omit TestClient's context manager: no application lifespan, sockets, or background services.
    result = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50123))
    yield result
    result.close()


def assert_ticket_invite(invite):
    assert invite["join_address"] == TICKET
    assert invite["join_command"] == f"uv run hypercompute join --ticket {TICKET} --code {invite['code']}"
    assert invite["expires_at"]


def test_dashboard_create_renew_and_cli_invite_all_use_ticket(client):
    created = client.post("/api/sessions", json={"name": "test"}, headers=OWNER_HEADERS)
    assert created.status_code == 200
    session = created.json()
    assert_ticket_invite(session["invite"])
    renewed = client.post(f"/api/sessions/{session['id']}/invites", headers=OWNER_HEADERS)
    assert renewed.status_code == 200
    assert_ticket_invite(renewed.json()["invite"])
    direct = client.post("/api/network/invite", headers=OWNER_HEADERS)
    assert direct.status_code == 200
    assert_ticket_invite(direct.json())
    assert direct.json()["ticket"] == TICKET
    assert len({session["invite"]["code"], renewed.json()["invite"]["code"], direct.json()["code"]}) == 3


def test_first_cli_invitation_creates_session_and_uses_ticket(client):
    response = client.post("/api/network/invite", headers=OWNER_HEADERS)
    assert response.status_code == 200
    assert_ticket_invite(response.json())
    assert client.get("/api/sessions/current").json()["status"] == "active"


def test_lan_invitation_stays_compatible_when_no_peer_service(app, client):
    app.state.peer_service = None
    response = client.post("/api/network/invite", headers=OWNER_HEADERS)
    assert response.status_code == 200
    invite = response.json()
    assert invite["join_command"] == f"uv run hypercompute join --host 10.0.0.2:8000 --code {invite['code']}"
    assert "ticket" not in invite
    assert client.get("/api/network").json() == {"transport": "lan"}


@pytest.mark.parametrize("headers", [{}, {"X-Hypercompute-Owner": "wrong"}])
def test_network_invitation_requires_owner_token(client, headers):
    assert client.post("/api/network/invite", headers=headers).status_code == 403
    assert client.get("/api/sessions/current").json() is None


def test_network_routes_reject_rebinding_and_remote_clients(app, client):
    assert client.get("/api/network", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/network/invite", headers={**OWNER_HEADERS, "Host": "evil.example"}).status_code == 403
    remote = TestClient(app, base_url="http://127.0.0.1:8000", client=("192.0.2.5", 50123))
    try:
        assert remote.get("/api/network").status_code == 403
        assert remote.post("/api/network/invite", headers=OWNER_HEADERS).status_code == 403
    finally:
        remote.close()
