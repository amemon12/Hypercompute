"""The rehearsal server shares fixture state without accessing the real pool."""
from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo_server  # noqa: E402


@pytest.fixture
def client(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "index.html").write_text("<h1>Demo fixture</h1>")
    (tmp_path / "demo" / "style.css").write_text("body {color: black}")
    (tmp_path / "brand").mkdir()
    (tmp_path / "brand" / "font.woff2").write_bytes(b"font")
    (tmp_path / "private.txt").write_text("not served")
    server, private_url = demo_server.serve(port=0, static=tmp_path, allowed_hosts=["192.168.1.10:8792"])
    token = parse_qs(urlparse(private_url).fragment)["t"][0]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None, *, auth=True):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        data = json.dumps(body) if body is not None else None
        request_headers = {"Content-Type": "application/json", **({"X-Demo-Host": token} if auth else {}),
                           **(headers or {})}
        connection.request(method, path, data, request_headers)
        response = connection.getresponse()
        payload = response.read()
        result = json.loads(payload) if response.headers.get_content_type() == "application/json" else payload
        connection.close()
        return response.status, result

    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_host_join_and_reset_are_shared(client):
    status, off = client("GET", "/api/state")
    assert status == 200
    instance = off["instance"]
    assert isinstance(instance, str) and instance
    assert off == {"instance": instance, "hosting": False, "joined": False, "version": 0,
                   "session": None, "code": None, "share_url": None}
    status, hosting = client("POST", "/api/hosting", {"enabled": True})
    assert status == 200 and hosting["hosting"] and not hosting["joined"]
    assert hosting["share_url"].endswith("/join") and "?" not in hosting["share_url"]
    assert len(hosting["code"]) == 6
    status, joined = client("POST", "/api/join", {"code": hosting["code"]}, auth=False)
    assert status == 200 and joined["joined"]
    assert client("GET", "/api/state")[1]["joined"] is True
    assert client("GET", "/api/join-info", auth=False)[1] == joined
    status, stopped = client("POST", "/api/hosting", {"enabled": False})
    assert status == 200 and not stopped["hosting"] and not stopped["joined"]
    assert stopped["session"] is None and stopped["share_url"] is None and stopped["code"] is None
    assert client("POST", "/api/join", {"code": hosting["code"]}, auth=False)[0] == 409
    new = client("POST", "/api/hosting", {"enabled": True})[1]
    assert new["session"] != hosting["session"]
    assert all(state["instance"] == instance for state in (hosting, joined, stopped, new))
    assert client("POST", "/api/join", {"code": hosting["code"]}, auth=False)[0] == 409


def test_server_instances_differ_even_when_versions_match():
    first = demo_server.DemoSession("http://127.0.0.1:8792").state()
    restarted = demo_server.DemoSession("http://127.0.0.1:8792").state()
    assert first["version"] == restarted["version"] == 0
    assert first["instance"] != restarted["instance"]


def test_repeated_host_and_join_actions_are_idempotent(client):
    first = client("POST", "/api/hosting", {"enabled": True})[1]
    assert client("POST", "/api/hosting", {"enabled": True})[1] == first
    joined = client("POST", "/api/join", {"code": first["code"]}, auth=False)[1]
    assert client("POST", "/api/join", {"code": first["code"]}, auth=False)[1] == joined


@pytest.mark.parametrize("path", ["/", "/join", "/join/", "/style.css", "/brand/font.woff2"])
def test_serves_demo_and_brand_assets(client, path):
    assert client("GET", path, auth=False)[0] == 200


@pytest.mark.parametrize("path", ["/../private.txt", "/%2e%2e/private.txt", "/brand/../private.txt", "/hcapp.py"])
def test_static_paths_cannot_escape_assets(client, path):
    assert client("GET", path)[0] == 404


@pytest.mark.parametrize("headers", [{"Host": "evil.example"}, {"Origin": "https://evil.example"},
                                    {"Origin": "null"}])
def test_refuses_foreign_hosts_and_origins(client, headers):
    assert client("GET", "/api/state", headers=headers)[0] == 403
    assert client("POST", "/api/hosting", {"enabled": True}, headers=headers)[0] == 403
    assert client("GET", "/api/state")[1]["hosting"] is False


@pytest.mark.parametrize("body", [{"enabled": "true"}, {"enabled": 1}, {}, [], None])
def test_host_requires_a_boolean(client, body):
    assert client("POST", "/api/hosting", body)[0] == 400


def test_join_validation(client):
    assert client("POST", "/api/join", {})[0] == 400
    assert client("POST", "/api/join", {"code": "unavailable"}, auth=False)[0] == 409
    assert client("POST", "/api/join", {"code": "\u2603\u2603\u2603"}, auth=False)[0] == 409
    assert client("POST", "/api/real-gateway", {})[0] == 404
    assert client("POST", "/api/hosting", {"enabled": True}, {"Content-Type": "text/plain"})[0] == 415


def test_guest_cannot_read_code_or_control_host(client):
    assert client("GET", "/api/state", auth=False)[0] == 403
    assert client("GET", "/api/state", headers={"X-Demo-Host": "wrong"})[0] == 403
    assert client("POST", "/api/hosting", {"enabled": True}, auth=False)[0] == 403
    host = client("POST", "/api/hosting", {"enabled": True})[1]
    guest = client("GET", "/api/join-info", auth=False)[1]
    assert guest["hosting"] and guest["can_join"] and not guest["joined"]
    assert set(guest) == {"instance", "hosting", "joined", "version", "can_join"}
    assert host["code"] not in json.dumps(guest)
    reply = client("POST", "/api/join", {"code": host["code"]}, auth=False)[1]
    assert "code" not in reply and "session" not in reply and "share_url" not in reply
    assert client("POST", "/api/hosting", {"enabled": False}, auth=False)[0] == 403
    assert client("GET", "/api/state")[1]["hosting"] is True


def test_join_requires_code_and_never_happens_on_page_load(client):
    code = client("POST", "/api/hosting", {"enabled": True})[1]["code"]
    assert client("GET", "/join", auth=False)[0] == 200
    assert client("GET", "/api/join-info", auth=False)[1]["joined"] is False
    assert client("POST", "/api/join", {"code": " " + code.lower() + " "}, auth=False)[1]["joined"]


def test_code_guessing_is_limited_and_recovers(client, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(demo_server.time, "monotonic", lambda: clock[0])
    code = client("POST", "/api/hosting", {"enabled": True})[1]["code"]
    for _ in range(5):
        assert client("POST", "/api/join", {"code": "!!!!!!"}, auth=False)[0] == 409
    assert client("POST", "/api/join", {"code": code}, auth=False)[0] == 429
    assert client("GET", "/api/state")[1]["joined"] is False
    clock[0] += 61
    assert client("POST", "/api/join", {"code": code}, auth=False)[0] == 200


def test_explicit_lan_host_is_allowed(client):
    assert client("GET", "/api/join-info", headers={"Host": "192.168.1.10:8792"}, auth=False)[0] == 200


def test_lan_requires_an_explicit_share_address():
    with pytest.raises(ValueError, match="public-url"):
        demo_server.serve(bind="0.0.0.0", port=0)


@pytest.mark.parametrize("url", ["https://demo.example", "http://user:pass@host", "http://host/path",
                                "http://host/?query=1", "http://host/#fragment", "http://host:bad"])
def test_public_url_rejects_invalid_addresses(url):
    with pytest.raises(ValueError, match="public-url"):
        demo_server.serve(port=0, public_url=url)
