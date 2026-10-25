"""The demo app server against a fake pool: access checks, static files, actions and the chat stream."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hcapp  # noqa: E402

TICKET, CODE = "a" * 40, "ABCDE-FGHJK"


class FakeBackend:
    def __init__(self) -> None:
        self.job, self.calls = {"state": "idle"}, []

    def state(self) -> dict:
        return {"host_up": True, "job": self.job, "peers": [{"name": "mac", "state": "eligible", "devices": ["MTL0"],
                                                             "admissible_gib": 9.5}], "models": ["m1"], "sharing": "x"}

    def start_host(self) -> str:
        self.calls.append("start"); return "hosting"

    def stop_host(self) -> str:
        return "host stopped"

    def invite(self) -> str:
        return f"hypercompute join --ticket {TICKET} --code {CODE}"

    def join(self, invite: str) -> str:
        self.calls.append(("join", invite)); return "joining"

    def leave(self) -> str:
        return "stopped sharing"

    def run_model(self, model: str) -> None:
        self.job = {"state": "ready", "model": model, "stages": [{"node": "local", "device": "CUDA0", "layers": 4}]}

    def stop_model(self) -> str:
        return "stopped"

    def chat(self, prompt: str):
        yield {"text": "hello "}
        yield {"text": "pool"}
        yield {"tok_s": 12.5}


@pytest.fixture()
def app():
    backend = FakeBackend()
    httpd, token, url = hcapp.serve(backend, 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    yield backend, token, base
    httpd.shutdown()


def test_page_and_brand_are_served(app) -> None:
    _, _, base = app
    assert "hypercompute" in httpx.get(base + "/").text
    assert httpx.get(base + "/brand/favicon.svg").status_code == 200


def test_static_paths_cannot_escape(app) -> None:
    _, _, base = app
    for path in ("/../hcapp.py", "/%2e%2e/hcapp.py", "/brand/../../hcapp.py"):
        assert httpx.get(base + path).status_code == 404


def test_api_needs_the_token_and_a_loopback_host(app) -> None:
    _, token, base = app
    assert httpx.get(base + "/api/state").status_code == 403
    assert httpx.get(base + "/api/state", headers={"X-App-Token": "wrong"}).status_code == 403
    assert httpx.get(base + "/api/state", headers={"X-App-Token": token, "Host": "evil.example:80"}).status_code == 403
    assert httpx.post(base + "/api/host/start", headers={"X-App-Token": token, "Origin": "https://evil.example"},
                      json={}).status_code == 403
    assert httpx.get(base + "/api/state", headers={"X-App-Token": token}).json()["peers"][0]["name"] == "mac"


def test_actions_and_join_validation(app) -> None:
    backend, token, base = app
    h = {"X-App-Token": token}
    assert httpx.post(base + "/api/host/start", headers=h, json={}).json()["result"] == "hosting"
    assert "--ticket" in httpx.post(base + "/api/invite", headers=h, json={}).json()["result"]
    bad = httpx.post(base + "/api/join", headers=h, json={"invite": "hypercompute join --ticket x; rm -rf ~"})
    assert bad.status_code == 500 and "invite" in bad.json()["error"]
    ok = httpx.post(base + "/api/join", headers=h, json={"invite": f"  hypercompute join --ticket {TICKET} --code {CODE}\n"})
    assert ok.json()["result"] == "joining" and backend.calls[-1] == ("join", f"hypercompute join --ticket {TICKET} --code {CODE}")
    assert httpx.post(base + "/api/run", headers=h, json={"model": "m1; ls"}).status_code == 500
    assert httpx.post(base + "/api/run", headers=h, json={"model": "m1"}).json()["result"] == "started"


def test_chat_streams_text_and_speed(app) -> None:
    _, token, base = app
    r = httpx.post(base + "/api/chat", headers={"X-App-Token": token}, json={"prompt": "hi"})
    events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert "".join(e.get("text", "") for e in events) == "hello pool" and events[-1]["tok_s"] == 12.5
