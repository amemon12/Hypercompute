"""plan, deploy, wait_ready, deploy_model and stop against the fake gateway."""

from __future__ import annotations

import json

import httpx
import pytest
from fake_gateway import MODEL, FakeClock, FakeGateway
from hypercompute_sdk import (
    ConflictError,
    DeploymentFailed,
    GatewayUnreachable,
    HypercomputeError,
    NotFoundError,
    Pool,
    ReadyTimeout,
    RequestError,
    UnavailableError,
)
from hypercompute_sdk.client import plan_body


def clocked(gw: FakeGateway, **kwargs: object) -> tuple[Pool, FakeClock]:
    clock = FakeClock()
    return gw.pool(sleep=clock.sleep, clock=clock.now, **kwargs), clock


def sent(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ----------------------------------------------------------------------------- plan


def test_plan_sends_only_given_keys() -> None:
    gw = FakeGateway()
    plan = gw.pool().plan(MODEL)
    assert plan["model_id"] == MODEL and plan["feasible"] is True
    assert sent(gw.calls("POST", "/api/plans")[0]) == {"model_id": MODEL, "strategy": "all_selected",
                                                        "ctx_size": 4096, "allow_ram_tier": True,
                                                        "allow_capacity_mode": True}


def test_plan_sends_every_option() -> None:
    gw = FakeGateway()
    layout = [{"node_id": "n1", "device": "MTL0", "layers": 4, "residency": "file_backed"}]
    gw.pool().plan(MODEL, strategy="auto", ctx_size=8192, node_ids=("n1",), allow_ram_tier=False,
                   allow_capacity_mode=False, engine="llama_rpc", objective="interactive", overflow="peers",
                   use_profiles=False, parallel=4, layout=layout)
    body = sent(gw.calls("POST", "/api/plans")[0])
    assert body == {"model_id": MODEL, "strategy": "auto", "ctx_size": 8192, "node_ids": ["n1"],
                    "allow_ram_tier": False, "allow_capacity_mode": False, "engine": "llama_rpc",
                    "objective": "interactive", "overflow": "peers", "use_profiles": False, "parallel": 4,
                    "layout": layout}


@pytest.mark.parametrize("kwargs", [
    {"strategy": "fastest"}, {"strategy": None}, {"ctx_size": 255}, {"ctx_size": 262145}, {"ctx_size": True},
    {"parallel": 0}, {"parallel": 17}, {"engine": "llama"}, {"objective": "speed"}, {"overflow": "disk"},
    {"allow_ram_tier": 1}, {"use_profiles": "yes"}, {"node_ids": "n1"}, {"node_ids": [""]}, {"layout": []},
    {"layout": "x"}, {"layout": [1]}, {"layout": [{}] * 65},
])
def test_plan_checks_locally(kwargs: dict) -> None:
    gw = FakeGateway()
    with pytest.raises(ValueError):
        gw.pool().plan(MODEL, **kwargs)
    assert gw.requests == []


def test_plan_needs_a_model_id() -> None:
    with pytest.raises(ValueError):
        plan_body("")


def test_plan_validation_error_from_gateway() -> None:
    gw = FakeGateway()
    gw.overrides[("POST", "/api/plans")] = lambda r: httpx.Response(422, json={"detail": [
        {"loc": ["body", "ctx_size"], "msg": "too big", "type": "less_than_equal"}]})
    with pytest.raises(RequestError) as info:
        gw.pool(owner_token=gw.token).plan(MODEL)
    assert info.value.locs == ("body.ctx_size",)


def test_plan_timeout_is_180_s() -> None:
    gw = FakeGateway()
    gw.pool().plan(MODEL)
    assert gw.calls("POST", "/api/plans")[0].extensions["timeout"]["read"] == 180.0


# ----------------------------------------------------------------------------- deploy


def test_deploy_always_sends_an_idempotency_key() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    dep = pool.deploy(pool.plan(MODEL)["id"])
    key = gw.calls("POST", "/api/deployments")[0].headers["idempotency-key"]
    assert len(key) == 32 and dep["idempotency_key"] == key and dep["state"] == "planned"


def test_deploy_uses_the_given_key_and_inline_plan() -> None:
    gw = FakeGateway()
    dep = gw.pool().deploy(plan=plan_body(MODEL, strategy="auto"), idempotency_key="run-2026-10-24:a")
    request = gw.calls("POST", "/api/deployments")[0]
    assert request.headers["idempotency-key"] == "run-2026-10-24:a"
    assert sent(request) == {"plan": {"model_id": MODEL, "strategy": "auto", "ctx_size": 4096,
                                      "allow_ram_tier": True, "allow_capacity_mode": True}}
    assert request.extensions["timeout"]["read"] == 180.0
    assert dep["plan"]["strategy"] == "auto"


@pytest.mark.parametrize(("args", "kwargs"), [((), {}), (("p",), {"plan": {"model_id": MODEL}}),
                                              (("",), {}), (("p",), {"idempotency_key": "has space"}),
                                              (("p",), {"idempotency_key": ""})])
def test_deploy_argument_checks(args: tuple, kwargs: dict) -> None:
    gw = FakeGateway()
    with pytest.raises(ValueError):
        gw.pool().deploy(*args, **kwargs)
    assert gw.requests == []


def test_deploy_timeout_finds_the_deployment_by_key() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    plan_id = pool.plan(MODEL)["id"]
    real = gw._create

    def slow(request: httpx.Request) -> httpx.Response:
        real(request)  # the gateway made it, but the reply is lost
        raise httpx.ReadTimeout("timed out", request=request)

    gw.overrides[("POST", "/api/deployments")] = slow
    dep = pool.deploy(plan_id, idempotency_key="key-1")
    assert dep["idempotency_key"] == "key-1"


def test_deploy_timeout_without_a_match_raises() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    pool.plan(MODEL)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    gw.overrides[("POST", "/api/deployments")] = slow
    with pytest.raises(GatewayUnreachable):
        pool.deploy("abc")
    gw.overrides[("GET", "/api/deployments")] = lambda r: httpx.Response(503, json={"detail": "x", "code": "y"})
    with pytest.raises(GatewayUnreachable):
        pool.deploy("abc")


def test_deploy_refused_connection_does_not_look_up() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    pool.plan(MODEL)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gw.overrides[("POST", "/api/deployments")] = refuse
    with pytest.raises(GatewayUnreachable):
        pool.deploy("abc")
    assert gw.calls("GET", "/api/deployments") == []


def test_deploy_unknown_plan_is_not_found() -> None:
    with pytest.raises(NotFoundError) as info:
        FakeGateway().pool().deploy("missing")
    assert info.value.code == "plan_not_found"


def test_deploy_retry_after_rotation_reuses_the_key() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    plan_id = pool.plan(MODEL)["id"]
    gw.rotate()
    pool.deploy(plan_id)
    keys = [r.headers["idempotency-key"] for r in gw.calls("POST", "/api/deployments")]
    assert len(keys) == 2 and keys[0] == keys[1]
    assert len(gw.deps) == 1


# ----------------------------------------------------------------------------- wait_ready


def test_wait_ready_reports_each_change_once() -> None:
    gw = FakeGateway(ready_after=3)
    pool, clock = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    seen: list[tuple[str, str | None]] = []
    ready = pool.wait_ready(dep["id"], poll=1.5, on_progress=lambda s, p: seen.append((s, p)))
    assert ready["state"] == "ready"
    assert seen == [("loading", "loading 1/3"), ("loading", "loading 2/3"), ("loading", "loading 3/3"),
                    ("ready", None)]
    assert clock.sleeps == [1.5, 1.5, 1.5]


def test_wait_ready_without_callback() -> None:
    gw = FakeGateway(ready_after=0)
    pool, _ = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    assert pool.wait_ready(dep["id"])["state"] == "ready"


def test_wait_ready_tolerates_four_failed_polls() -> None:
    gw = FakeGateway(ready_after=1, poll_failures=["503", "connect", "502", "connect"])
    pool, clock = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    assert pool.wait_ready(dep["id"])["state"] == "ready"
    assert len(gw.calls("GET", f"/api/deployments/{dep['id']}")) == 6
    assert len(clock.sleeps) == 5


def test_wait_ready_raises_on_the_fifth_failed_poll() -> None:
    gw = FakeGateway(poll_failures=["connect"] * 5)
    pool, clock = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    with pytest.raises(GatewayUnreachable):
        pool.wait_ready(dep["id"])
    assert len(clock.sleeps) == 4


def test_wait_ready_failure_count_resets_after_a_good_poll() -> None:
    gw = FakeGateway(ready_after=3, poll_failures=["503"] * 4 + ["ok"] + ["connect"] * 4)
    pool, _ = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    assert pool.wait_ready(dep["id"])["state"] == "ready"
    assert gw.poll_failures == []


def test_wait_ready_does_not_retry_a_404() -> None:
    gw = FakeGateway()
    pool, clock = clocked(gw)
    with pytest.raises(NotFoundError):
        pool.wait_ready("nope")
    assert clock.sleeps == []


def test_wait_ready_raises_deployment_failed() -> None:
    gw = FakeGateway(ready_after=1, fail_with="llama-server exited with code 1")
    pool, _ = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    with pytest.raises(DeploymentFailed) as info:
        pool.wait_ready(dep["id"])
    err = info.value
    assert err.code == "deployment_failed"
    assert err.deployment["error"] == "llama-server exited with code 1"
    assert [h["state"] for h in err.deployment["history"]] == ["planned", "loading", "failed"]
    assert "llama-server exited with code 1" in str(err)


def test_wait_ready_raises_on_stopped() -> None:
    gw = FakeGateway()
    dep = gw.add_dep("stopped")
    pool, _ = clocked(gw)
    with pytest.raises(DeploymentFailed) as info:
        pool.wait_ready(dep["id"])
    assert info.value.code == "deployment_stopped"
    assert "no error recorded" in str(info.value)


def test_wait_ready_timeout_keeps_the_last_deployment() -> None:
    gw = FakeGateway(ready_after=10_000)
    pool, clock = clocked(gw)
    dep = pool.deploy(pool.plan(MODEL)["id"])
    with pytest.raises(ReadyTimeout) as info:
        pool.wait_ready(dep["id"], timeout=10.0, poll=2.0)
    assert info.value.deployment is not None and info.value.deployment["state"] == "loading"
    assert clock.t > 10.0
    assert "last state loading" in str(info.value)


def test_wait_ready_timeout_with_no_good_poll() -> None:
    gw = FakeGateway(poll_failures=["503"] * 3)
    pool, _ = clocked(gw)
    with pytest.raises(ReadyTimeout) as info:
        pool.wait_ready("abc", timeout=1.0, poll=2.0)
    assert info.value.deployment is None


def test_wait_ready_checks_the_id() -> None:
    with pytest.raises(ValueError):
        FakeGateway().pool().wait_ready("../etc")


def test_default_timeouts_follow_run_big() -> None:
    import inspect

    sig = inspect.signature(Pool.wait_ready)
    assert sig.parameters["timeout"].default == 7200.0 and sig.parameters["poll"].default == 2.0


# ----------------------------------------------------------------------------- deploy_model


def test_deploy_model_plans_deploys_and_waits() -> None:
    gw = FakeGateway(ready_after=2)
    pool, _ = clocked(gw)
    states: list[str] = []
    dep = pool.deploy_model(MODEL, strategy="auto", on_progress=lambda s, p: states.append(s))
    assert dep["state"] == "ready" and states[-1] == "ready"
    assert sent(gw.calls("POST", "/api/plans")[0])["strategy"] == "auto"


def test_deploy_model_without_wait() -> None:
    gw = FakeGateway()
    dep = gw.pool().deploy_model(MODEL, wait=False)
    assert dep["state"] == "planned"
    assert gw.calls("GET", f"/api/deployments/{dep['id']}") == []


def test_deploy_model_plans_again_after_plan_stale() -> None:
    gw = FakeGateway(stale_deploys=1, ready_after=0)
    pool, _ = clocked(gw)
    dep = pool.deploy_model(MODEL)
    assert dep["state"] == "ready"
    assert len(gw.calls("POST", "/api/plans")) == 2
    assert len(gw.calls("POST", "/api/deployments")) == 2


def test_deploy_model_stale_twice_raises() -> None:
    gw = FakeGateway(stale_deploys=2)
    with pytest.raises(ConflictError) as info:
        gw.pool().deploy_model(MODEL)
    assert info.value.code == "plan_stale"


def test_deploy_model_never_stops_an_active_deployment() -> None:
    gw = FakeGateway()
    other = gw.add_dep("ready")
    with pytest.raises(ConflictError) as info:
        gw.pool().deploy_model(MODEL)
    assert info.value.code == "deployment_active"
    assert gw.deps[other["id"]]["state"] == "ready"
    assert gw.calls("DELETE", f"/api/deployments/{other['id']}") == []


def test_deploy_model_reply_without_id_is_a_bad_response() -> None:
    gw = FakeGateway()
    gw.overrides[("POST", "/api/plans")] = lambda r: httpx.Response(200, json={"feasible": False})
    with pytest.raises(HypercomputeError) as info:
        gw.pool(owner_token=gw.token).deploy_model(MODEL)
    assert info.value.code == "bad_response"
    gw.overrides[("POST", "/api/plans")] = lambda r: httpx.Response(200, json={"id": "p1"})
    gw.overrides[("POST", "/api/deployments")] = lambda r: httpx.Response(202, json={"state": "planned"})
    with pytest.raises(HypercomputeError) as info:
        gw.pool(owner_token=gw.token).deploy_model(MODEL)
    assert info.value.code == "bad_response"


# ----------------------------------------------------------------------------- stop and reads


def test_stop_with_no_active_deployment_returns_none() -> None:
    gw = FakeGateway()
    gw.add_dep("stopped")
    assert gw.pool().stop() is None
    assert [r for r in gw.requests if r.method == "DELETE"] == []


def test_stop_finds_the_active_deployment() -> None:
    gw = FakeGateway()
    gw.add_dep("failed")
    active = gw.add_dep("ready")
    stopped = gw.pool().stop()
    assert stopped is not None and stopped["id"] == active["id"] and stopped["state"] == "stopped"
    request = gw.calls("DELETE", f"/api/deployments/{active['id']}")[0]
    assert request.extensions["timeout"]["read"] == 300.0
    assert request.headers["x-hypercompute-owner"] == gw.token


def test_stop_by_id_and_unknown_id() -> None:
    gw = FakeGateway()
    dep = gw.add_dep("loading")
    pool = gw.pool()
    assert pool.stop(dep["id"])["state"] == "stopped"
    with pytest.raises(NotFoundError):
        pool.stop("abc")
    with pytest.raises(ValueError):
        pool.stop("a/b")


def test_deployments_and_deployment() -> None:
    gw = FakeGateway()
    first, second = gw.add_dep("stopped"), gw.add_dep("ready")
    pool = gw.pool()
    assert [d["id"] for d in pool.deployments()] == [second["id"], first["id"]]
    assert pool.deployment(first["id"])["state"] == "stopped"
    assert "polls" not in pool.deployment(first["id"])


def test_unavailable_maps_on_reads() -> None:
    gw = FakeGateway()
    gw.overrides[("GET", "/api/capacity/pools")] = lambda r: httpx.Response(503, json={
        "detail": "the capacity optimizer is not part of this gateway", "code": "capacity_unavailable"})
    with pytest.raises(UnavailableError):
        gw.pool().pools()


def test_wait_ready_skips_repeated_progress() -> None:
    gw = FakeGateway()
    replies = iter([{"id": "d1", "state": "loading", "progress": "10/40"}] * 3 + [{"id": "d1", "state": "ready"}])
    gw.overrides[("GET", "/api/deployments/d1")] = lambda r: httpx.Response(200, json=next(replies))
    pool, clock = clocked(gw)
    seen: list[tuple[str, str | None]] = []
    pool.wait_ready("d1", on_progress=lambda s, p: seen.append((s, p)))
    assert seen == [("loading", "10/40"), ("ready", None)]
    assert len(clock.sleeps) == 3
