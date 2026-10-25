"""Regression tests for the friend agent and CLI: stop paths, leave, host parsing, scoping guidance, link-local."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path

import httpx
import pytest
from factories import contribution, make_inventory
from hypercompute import cli, config
from hypercompute.agent import join
from hypercompute.agent.join import Agent, AgentError, parse_host
from hypercompute.agent.supervisor import StopOutcome, Supervisor

SLEEPER = ["-c", "import time; time.sleep(120)"]


def make_agent(system: str = "windows", host: str = "127.0.0.1") -> Agent:
    inv = make_inventory(system)
    return Agent(host=host, port=8000, code="code", display_name="friend", inv=inv,
                 contribution=contribution(["CUDA0"] if system != "macos" else ["MTL0"]),
                 advertise_ip="10.0.0.5", worker_port=47052)


# --- finding 14: parse_host ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("192.168.1.20", ("192.168.1.20", 8000)),
    ("192.168.1.20:8001", ("192.168.1.20", 8001)),
    ("192.168.1.20:8000/", ("192.168.1.20", 8000)),
    ("http://192.168.1.20:8000/", ("192.168.1.20", 8000)),
    (" operator-MBP.local:8000/ ", ("operator-mbp.local", 8000)),
])
def test_parse_host_accepts_common_forms(text, expected):
    assert parse_host(text) == expected


@pytest.mark.parametrize("text", ["192.168.1.20:80a0", "https://192.168.1.20:8000", "foo bar", "", ":8000",
                                  "http://", "192.168.1.20:0", "192.168.1.20:70000", "user@10.0.0.1:8000",
                                  "999.1.1.1", "192.168.1", "[::1]:8000"])
def test_parse_host_rejects_garbage_with_a_fix(text):
    with pytest.raises(AgentError) as err:
        parse_host(text)
    assert err.value.fix and "192.168.1.20:8000" in err.value.fix


def test_join_with_bad_host_fails_without_traceback(capsys):
    assert cli.main(["join", "--host", "192.168.1.20:8000x/", "--code", "abc"]) == 2
    assert "invalid --host" in capsys.readouterr().out


# --- findings 7/25/15: scoping guidance ----------------------------------------------------------------------------

def test_scoping_guidance_uses_resolved_host_ip_per_os():
    win = make_agent("windows", host="localhost")
    win.worker_exe = r"C:\hc\ggml-rpc-server.exe"
    text = join.scoping_guidance(win)
    assert "no authentication" in text and "localhost" not in text
    assert "-LocalPort 47052 -RemoteAddress 127.0.0.1 -Action Allow" in text
    assert 'Get-NetFirewallApplicationFilter -Program "C:\\hc\\ggml-rpc-server.exe"' in text

    linux = join.scoping_guidance(make_agent("linux", host="localhost"))
    assert "sudo ufw allow from 127.0.0.1 to any port 47052 proto tcp" in linux and "default deny" in linux

    mac = join.scoping_guidance(make_agent("macos", host="localhost"))
    assert ("printf 'pass in quick proto tcp from 127.0.0.1 to any port 47052\\nblock drop in quick proto tcp to any "
            "port 47052\\n' | sudo pfctl -a com.apple/hypercompute -f - && sudo pfctl -E") in mac
    assert "sudo pfctl -a com.apple/hypercompute -F all" in mac


def test_run_join_always_prints_scoping_guidance(monkeypatch, capsys):
    agent = make_agent("linux", host="localhost")

    async def serve() -> None:
        return None

    monkeypatch.setattr(agent, "start_worker", lambda: None)
    monkeypatch.setattr(agent, "join", lambda: None)
    monkeypatch.setattr(agent, "serve", serve)
    monkeypatch.setattr(agent, "leave", lambda: None)
    assert join.run_join(agent, fallback_cpu=False, assume_yes=True) == 0
    out = capsys.readouterr().out
    assert "no authentication" in out and "sudo ufw allow from 127.0.0.1 to any port 47052" in out
    assert "If a firewall is active" not in out


# --- finding 10: SIGTERM / SIGHUP / SIGBREAK stop sharing like Ctrl+C ----------------------------------------------

def test_stop_signals_trigger_graceful_stop(monkeypatch):
    agent = make_agent()
    installed: dict[int, object] = {}
    monkeypatch.setattr(join.signal, "signal", lambda sig, handler: installed.setdefault(sig, handler))
    join._install_stop_signals(agent)
    wanted = {signal.SIGBREAK} if sys.platform == "win32" else {signal.SIGTERM, signal.SIGHUP}
    assert set(installed) == wanted
    sig = next(iter(wanted))
    handler = installed[sig]
    with pytest.raises(KeyboardInterrupt):  # before serving: the same path as Ctrl+C (stop worker, tell host)
        handler(sig, None)

    async def serving() -> None:
        agent.loop = asyncio.get_running_loop()
        handler(sig, None)
        await asyncio.wait_for(agent.stop_event.wait(), timeout=2)

    asyncio.run(serving())
    agent.leaving = True
    handler(sig, None)  # a second signal during cleanup must not interrupt it


# --- findings 7 + 30: heartbeat loop ---------------------------------------------------------------------------------

def _run_heartbeats(agent: Agent, handler, monkeypatch) -> None:
    monkeypatch.setattr(join.inventory, "fresh_measurements", lambda inv: ([], []))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.wait_for(agent.heartbeat_loop(client), timeout=5)

    asyncio.run(run())


def test_assignment_is_not_reported_as_serving_layers(monkeypatch, capsys):
    agent = make_agent()
    agent.credential = "cred"

    def handler(request: httpx.Request) -> httpx.Response:
        agent.stop_event.set()
        return httpx.Response(200, json={"state": "eligible", "state_reason": None, "removed": False,
                                         "deployments": ["dep-1"]})

    _run_heartbeats(agent, handler, monkeypatch)
    out = capsys.readouterr().out
    assert "now serving model layers" not in out
    assert "a deployment on the host is using this machine (it may still be loading)" in out
    assert "serving_deployments" not in agent.status_doc()


def test_host_unreachable_too_long_stops_sharing(monkeypatch, capsys):
    agent = make_agent()
    agent.credential = "cred"
    monkeypatch.setattr(config, "HOST_UNREACHABLE_STOP_S", 0.05)
    monkeypatch.setattr(config, "HEARTBEAT_INTERVAL_S", 0.02)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("host is gone", request=request)

    _run_heartbeats(agent, handler, monkeypatch)
    assert agent.stop_event.is_set() and agent.host_lost
    assert "stopping sharing" in capsys.readouterr().out


def test_leave_after_host_lost_retries_goodbye_then_clears_state(monkeypatch):
    agent = make_agent()
    agent.credential, agent.host_lost = "cred", True
    join.write_state({"host": agent.host, "port": agent.port, "credential": "cred"})
    calls: list[str] = []

    def post(url: str, **kwargs: object) -> None:
        calls.append(url)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(join.httpx, "post", post)
    monkeypatch.setattr(join.time, "sleep", lambda s: None)
    agent.leave()
    assert len(calls) == 3 and join.read_state() is None


# --- findings 11/13/24: leave paths ---------------------------------------------------------------------------------

def test_agent_leave_keeps_state_when_worker_could_not_be_stopped(monkeypatch, capsys):
    agent = make_agent()
    agent.credential = "cred"
    join.write_state({"host": agent.host, "port": agent.port, "credential": "cred"})
    monkeypatch.setattr(agent.sup, "stop", lambda key, grace_s=8.0: StopOutcome(key, 4242, None, "still_running"))
    posts: list[str] = []
    monkeypatch.setattr(join.httpx, "post", lambda url, **kw: posts.append(url))
    agent.leave()
    assert join.read_state() is not None and posts == []
    assert "may still be running" in capsys.readouterr().out


def test_leave_without_state_stops_a_recorded_live_worker(capsys):
    records = config.paths().ensure().agent / "managed-processes.json"
    tmp = records.parent
    proc = Supervisor(records).start(join.WORKER_KEY, Path(sys.executable), SLEEPER, tmp, tmp / "w.log")
    try:
        assert join.read_state() is None  # the agent died before join() wrote state.json
        assert cli.main(["leave"]) == 0
        out = capsys.readouterr().out
        assert "orphaned pid" in out and "not currently sharing" not in out
        proc.popen.wait(timeout=10)
        assert not proc.alive() and json.loads(records.read_text()) == []
    finally:
        if proc.alive():
            proc.popen.kill()


def test_leave_keeps_state_when_a_worker_survives(monkeypatch, capsys):
    class StuckSupervisor:
        def __init__(self, path: Path) -> None:
            pass

        def reconcile_orphans(self) -> list[str]:
            return ["worker: could not stop pid 77 (AccessDenied); record kept"]

        def recorded(self) -> list[dict]:
            return [{"key": "worker", "pid": 77}]

    join.write_state({"host": "10.0.0.2", "port": 8000, "node_id": "n", "credential": "cred"})
    monkeypatch.setattr(cli, "Supervisor", StuckSupervisor)
    posts: list[str] = []
    monkeypatch.setattr(cli.httpx, "post", lambda url, **kw: posts.append(url))
    assert cli.main(["leave"]) == 1
    out = capsys.readouterr().out
    assert "still running" in out and "Administrator" in out
    assert join.read_state() is not None and posts == []


# --- finding 12: link-local advertise address ------------------------------------------------------------------------

def test_link_local_advertise_ip_is_accepted(monkeypatch):
    assert cli._advertise_ip("169.254.10.2", 8000, "169.254.10.3") == "169.254.10.3"
    monkeypatch.setattr(cli.netutil, "route_source_ip", lambda host, port: "169.254.10.3")
    assert cli._advertise_ip("169.254.10.2", 8000, None) == "169.254.10.3"
    with pytest.raises(AgentError):
        cli._advertise_ip("10.0.0.2", 8000, "8.8.8.8")
