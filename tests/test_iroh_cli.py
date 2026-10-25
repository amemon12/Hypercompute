"""Opt-in Iroh command boundaries, with no hardware collection or network services."""

from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest
from hypercompute import cli, config
from hypercompute.agent.join import AgentError
from hypercompute.transport import client as client_module


def test_default_host_and_legacy_join_remain_lan():
    parser = cli.build_parser()
    host = parser.parse_args(["host"])
    assert host.transport == "lan"
    assert host.relay_url is None
    join = parser.parse_args(["join", "--host", "192.168.1.20:8000", "--code", "ABCDE-FGHJK"])
    assert join.host == "192.168.1.20:8000"
    assert join.ticket is None


@pytest.mark.parametrize("arguments", [
    ["join", "--code", "ABCDE-FGHJK"],
    ["join", "--host", "127.0.0.1", "--ticket", "ticket", "--code", "ABCDE-FGHJK"],
    ["host", "--transport", "iroh", "--relay-only"],
])
def test_invalid_address_modes_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["host", "--transport", "iroh", "--bind", "192.168.1.20"],
    ["host", "--relay-url", "https://relay.example.com"],
])
def test_unsupported_host_flags_fail_before_startup(arguments, monkeypatch):
    collect = Mock(side_effect=AssertionError("must not collect hardware"))
    monkeypatch.setattr(cli.inventory, "collect", collect)
    with pytest.raises(AgentError):
        cli._run_host(cli.build_parser().parse_args(arguments))
    collect.assert_not_called()


@pytest.mark.parametrize("extra", [["--exo"], ["--advertise-ip", "192.168.1.21"]])
def test_peer_join_rejects_lan_only_features_before_connecting(extra, monkeypatch):
    create = Mock(side_effect=AssertionError("must not create peer transport"))
    monkeypatch.setattr(client_module, "PeerClient", create)
    args = cli.build_parser().parse_args(["join", "--ticket", "ticket", "--code", "ABCDE-FGHJK", *extra])
    with pytest.raises(AgentError, match="require LAN mode"):
        cli._run_join(args)
    create.assert_not_called()


def test_lan_join_rejects_ignored_relay_flag_before_connecting(monkeypatch):
    check = Mock(side_effect=AssertionError("must not query host"))
    monkeypatch.setattr(cli, "check_coordinator", check)
    args = cli.build_parser().parse_args([
        "join", "--host", "192.168.1.20", "--code", "ABCDE-FGHJK", "--relay-url", "https://relay.example.com",
    ])
    with pytest.raises(AgentError, match="requires --ticket"):
        cli._run_join(args)
    check.assert_not_called()


def test_peer_join_closes_transport_after_agent_failure(monkeypatch):
    peer = Mock()
    peer.request.return_value = httpx.Response(200, json={"engine": {}}, request=httpx.Request("POST", "http://fake"))
    peer.status.return_value = {"path": None}
    monkeypatch.setattr(client_module, "PeerClient", Mock(return_value=peer))
    monkeypatch.setattr(cli, "_run_join_on_host", Mock(side_effect=AgentError("worker failed")))
    args = cli.build_parser().parse_args(["join", "--ticket", "ticket", "--code", "ABCDE-FGHJK"])
    with pytest.raises(AgentError, match="worker failed"):
        cli._run_join(args)
    peer.request.assert_called_once_with("health")
    peer.close.assert_called_once()


def test_invite_uses_owner_local_bootstrap_and_prints_complete_command(monkeypatch, capsys):
    requests = []
    command = "uv run hypercompute join --ticket public-ticket --code ABCDE-FGHJK"

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "owner-test-token"})
        return httpx.Response(200, json={"join_command": command, "expires_at": "2026-10-21T12:15:00Z"})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    construct = Mock(return_value=client)
    monkeypatch.setattr(cli.httpx, "Client", construct)
    args = cli.build_parser().parse_args(["invite", "--port", "9123"])
    assert cli.cmd_invite(args) == 0
    construct.assert_called_once_with(trust_env=False, timeout=10)
    assert [(request.method, str(request.url)) for request in requests] == [
        ("GET", "http://127.0.0.1:9123/api/owner/bootstrap"),
        ("POST", "http://127.0.0.1:9123/api/network/invite"),
    ]
    assert requests[1].headers[config.OWNER_HEADER] == "owner-test-token"
    assert command in capsys.readouterr().out
    assert client.is_closed
