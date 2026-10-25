"""Read methods: health, status, nodes, peers, models, served_models, pools, explain."""

from __future__ import annotations

import pytest
from fake_gateway import MODEL, FakeGateway
from hypercompute_sdk import OwnerAuthError


def test_health() -> None:
    assert FakeGateway().pool().health()["engine"]["release"] == "b6000"


def test_simple_reads() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    assert [n["is_local"] for n in pool.nodes()] == [True, False]
    assert pool.peers() == [{"node_id": "n1"}]
    assert [m["artifact"]["id"] for m in pool.models()] == ["big-model", MODEL]
    assert pool.pools() == {"pools": []}
    assert gw.bootstraps == 0


def test_served_models() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    assert pool.served_models() == []
    gw.add_dep("loading")
    assert pool.served_models() == []
    gw.deps.clear()
    gw.add_dep("ready")
    assert pool.served_models() == [MODEL]


def test_status_without_deployment() -> None:
    status = FakeGateway().pool().status()
    assert status == {
        "version": "0.1.0", "engine_release": "b6000", "transport": "iroh",
        "connections": [{"node_id": "n1", "connected": True, "path": {"mode": "direct", "rtt_ms": 3.0}, "rpc": {}}],
        "nodes": [{"id": "local", "display_name": "gx10", "state": "ready", "heartbeat_age_s": None, "max_bytes": 100},
                  {"id": "n1", "display_name": "mac-a", "state": "ready", "heartbeat_age_s": 1.5, "max_bytes": 50}],
        "deployment": None, "ready_model": None}


def test_status_drops_iroh_ids() -> None:
    text = repr(FakeGateway().pool().status())
    assert "e" * 64 not in text and "p" * 64 not in text and "r" * 64 not in text


def test_status_with_latest_deployment_and_lan() -> None:
    import httpx

    gw = FakeGateway()
    gw.add_dep("stopped")
    latest = gw.add_dep("ready")
    gw.overrides[("GET", "/api/network")] = lambda r: httpx.Response(200, json={"transport": "lan"})
    status = gw.pool().status()
    assert status["deployment"]["id"] == latest["id"]
    assert status["ready_model"] == MODEL
    assert status["transport"] == "lan" and status["connections"] == []


def test_status_tolerates_missing_pieces() -> None:
    import httpx

    gw = FakeGateway()
    gw.overrides[("GET", "/api/health")] = lambda r: httpx.Response(200, json={"status": "ok", "version": "x"})
    gw.overrides[("GET", "/api/network")] = lambda r: httpx.Response(200, json={"peers": [
        "junk", {"node_id": "n2", "path": None}]})
    gw.overrides[("GET", "/api/nodes")] = lambda r: httpx.Response(200, json=[{"id": "n2"}, "junk"])
    status = gw.pool().status()
    assert status["engine_release"] is None and status["transport"] == "lan"
    assert status["connections"] == [{"node_id": "n2", "path": None}]
    assert status["nodes"] == [{"id": "n2", "display_name": None, "state": None, "heartbeat_age_s": None,
                                "max_bytes": None}]


def test_reads_off_loopback_are_owner_errors() -> None:
    with pytest.raises(OwnerAuthError):
        FakeGateway(loopback=False).pool().nodes()


def test_explain_sends_every_parameter() -> None:
    gw = FakeGateway()
    reply = gw.pool().explain(MODEL, ctx_size=8192, prompt_tokens=128, output_tokens=64, warm=True, slots=4)
    assert reply["params"] == {"model_id": MODEL, "ctx_size": "8192", "prompt_tokens": "128",
                               "output_tokens": "64", "warm": "true", "slots": "4"}
    assert gw.calls("GET", "/api/capacity/explain")[0].extensions["timeout"]["read"] == 180.0


def test_explain_defaults_match_the_gateway() -> None:
    reply = FakeGateway().pool().explain(MODEL)
    assert reply["params"] == {"model_id": MODEL, "ctx_size": "4096", "prompt_tokens": "512",
                               "output_tokens": "256", "warm": "false", "slots": "2"}


@pytest.mark.parametrize("kwargs", [
    {"ctx_size": 255}, {"ctx_size": 262145}, {"prompt_tokens": -1}, {"output_tokens": 0}, {"slots": 0},
    {"slots": 9}, {"warm": "yes"}, {"ctx_size": 1024, "prompt_tokens": 1000, "output_tokens": 25},
])
def test_explain_checks_locally(kwargs: dict) -> None:
    gw = FakeGateway()
    with pytest.raises(ValueError):
        gw.pool().explain(MODEL, **kwargs)
    assert gw.requests == []


@pytest.mark.parametrize("model_id", ["", "-bad", "a/b", "x" * 129, "ok\n", None])
def test_explain_checks_the_model_id(model_id: object) -> None:
    with pytest.raises(ValueError):
        FakeGateway().pool().explain(model_id)  # type: ignore[arg-type]


def test_explain_prompt_plus_output_equal_to_ctx_is_allowed() -> None:
    FakeGateway().pool().explain(MODEL, ctx_size=1024, prompt_tokens=1000, output_tokens=24)
