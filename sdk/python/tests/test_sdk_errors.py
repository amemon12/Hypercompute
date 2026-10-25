"""Every error mapping row, the three body shapes, str() format, and that the owner token never leaks."""

# ruff: noqa: S105, S106 - fake owner tokens and source labels

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fake_gateway import FIRST_TOKEN, MODEL, FakeClock, FakeGateway, error
from hypercompute_sdk import (
    BodyTooLarge,
    ChatInterrupted,
    ConflictError,
    DeploymentFailed,
    EngineError,
    GatewayUnreachable,
    HypercomputeError,
    NotFoundError,
    OwnerAuthError,
    Pool,
    ReadyTimeout,
    RequestError,
    StreamError,
    UnavailableError,
)
from hypercompute_sdk.errors import from_body, parse_body

ROWS = [
    (400, "bad_json", RequestError),
    (400, "model_error", RequestError),
    (422, "plan_not_feasible", RequestError),
    (422, "workload_exceeds_context", RequestError),
    (403, "owner_only", OwnerAuthError),
    (404, "plan_not_found", NotFoundError),
    (404, "model_not_found", NotFoundError),
    (409, "deployment_active", ConflictError),
    (409, "plan_stale", ConflictError),
    (409, "host_ineligible", ConflictError),
    (413, "body_too_large", BodyTooLarge),
    (500, "engine_error", EngineError),
    (500, "candidate", EngineError),
    (502, "bad_engine_flags", EngineError),
    (503, "no_ready_deployment", UnavailableError),
    (503, "capacity_unavailable", UnavailableError),
    (418, "teapot", HypercomputeError),
    (429, "busy", HypercomputeError),
]


def canned(response: httpx.Response | Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(response if callable(response) else lambda r: response)


@pytest.mark.parametrize(("status", "code", "cls"), ROWS)
def test_status_maps_to_class(status: int, code: str, cls: type[HypercomputeError]) -> None:
    pool = Pool(transport=canned(error(status, code, "it went wrong", "do the fix")))
    with pytest.raises(cls) as info:
        pool.nodes()
    err = info.value
    assert type(err) is cls
    assert (err.status, err.code, err.detail, err.fix) == (status, code, "it went wrong", "do the fix")
    assert str(err) == f"HTTP {status} {code}: it went wrong\nfix: do the fix"


def test_str_without_fix_is_one_line() -> None:
    err = from_body(409, {"detail": "plan is older than 10 minutes; check the plan again", "code": "plan_stale"})
    assert str(err) == "HTTP 409 plan_stale: plan is older than 10 minutes; check the plan again"
    assert repr(err) == "ConflictError(status=409, code='plan_stale')"


def test_validation_list_keeps_locs() -> None:
    body = {"detail": [{"loc": ["body", "parallel"], "msg": "Input should be less than or equal to 16",
                        "type": "less_than_equal"}, {"loc": [], "msg": "bad", "type": "x"}]}
    err = from_body(422, body)
    assert isinstance(err, RequestError)
    assert err.locs == ("body.parallel", "")
    assert err.detail == "body.parallel: Input should be less than or equal to 16; bad"
    assert err.code is None
    assert str(err).startswith("HTTP 422 error: body.parallel")


def test_empty_validation_list_has_a_message() -> None:
    assert from_body(422, {"detail": []}).detail == "request validation failed"


def test_chat_error_shape() -> None:
    err = from_body(500, {"error": {"code": 500, "message": "llama-server said no", "type": "upstream_error"}})
    assert isinstance(err, EngineError)
    assert (err.code, err.detail) == ("500", "llama-server said no")


def test_string_error_and_odd_bodies() -> None:
    assert from_body(502, {"error": "upstream died"}).detail == "upstream died"
    assert from_body(500, ["not", "a", "dict"]).detail == "['not', 'a', 'dict']"
    assert from_body(500, {}).detail == "HTTP 500"
    assert parse_body("x" * 1000)["detail"] == "x" * 300


def test_non_json_error_body() -> None:
    pool = Pool(transport=canned(httpx.Response(502, text="<html>Bad Gateway</html>")))
    with pytest.raises(EngineError) as info:
        pool.health()
    assert info.value.detail == "<html>Bad Gateway</html>"


def test_interrupted_non_stream_is_chat_interrupted() -> None:
    err = from_body(502, {"error": {"code": "interrupted", "type": "upstream_error", "message": "deployment stopped",
                                    "run_id": "run1", "partial_text": "Hel", "partial_truncated": True}})
    assert isinstance(err, ChatInterrupted)
    assert (err.status, err.run_id, err.partial_text, err.partial_truncated) == (502, "run1", "Hel", True)


def test_interrupted_detail_shape() -> None:
    err = from_body(502, {"detail": "deployment stopped", "code": "interrupted"})
    assert isinstance(err, ChatInterrupted)
    assert (err.detail, err.run_id, err.partial_text) == ("deployment stopped", None, "")


def test_connection_refused_is_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    with pytest.raises(GatewayUnreachable) as info:
        Pool(transport=canned(refuse)).health()
    err = info.value
    assert (err.status, err.code) == (0, "unreachable")
    assert str(err).startswith("unreachable: ConnectError: [Errno 111] Connection refused\nfix: ")


def test_success_body_that_is_not_json() -> None:
    with pytest.raises(HypercomputeError) as info:
        Pool(transport=canned(httpx.Response(200, text="hello"))).health()
    assert info.value.code == "bad_response"


@pytest.mark.parametrize("call", ["health", "nodes", "served_models", "status"])
def test_success_body_of_wrong_shape(call: str) -> None:
    wrong = {"health": [], "nodes": {}, "served_models": [], "status": "x"}[call]
    with pytest.raises(HypercomputeError) as info:
        getattr(Pool(transport=canned(httpx.Response(200, json=wrong))), call)()
    assert info.value.code == "bad_response"


# ----------------------------------------------------------------------------- the token never leaks


def _echo(status: int, code: str) -> Callable[[httpx.Request], httpx.Response]:
    """A gateway that (wrongly) echoes the owner header in its error text."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": FIRST_TOKEN})
        token = request.headers.get("x-hypercompute-owner", FIRST_TOKEN)
        return error(status, code, f"saw {token}", f"fix {token}")
    return handler


def _leak_paths() -> list[tuple[str, Callable[[], Any]]]:
    paths: list[tuple[str, Callable[[], Any]]] = []
    for status, code, _ in ROWS:
        def plan(status: int = status, code: str = code) -> Any:
            return Pool(transport=canned(_echo(status, code))).plan(MODEL)
        paths.append((f"plan-{status}-{code}", plan))

    def unreachable() -> Any:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"refused {request.headers.get('x-hypercompute-owner')}", request=request)
        return Pool(owner_token=FIRST_TOKEN, transport=canned(refuse)).plan(MODEL)

    def stream_mode(mode: str) -> Callable[[], Any]:
        def run() -> Any:
            gw = FakeGateway()
            gw.add_dep("ready")
            gw.chat_mode = mode
            return list(gw.pool().stream([{"role": "user", "content": "hi"}]))
        return run

    def failed() -> Any:
        gw, clock = FakeGateway(fail_with=f"engine crashed near {FIRST_TOKEN}"), FakeClock()
        pool = gw.pool(owner_token=FIRST_TOKEN, sleep=clock.sleep, clock=clock.now)
        return pool.deploy_model(MODEL)

    def timeout() -> Any:
        gw, clock = FakeGateway(ready_after=10_000), FakeClock()
        pool = gw.pool(sleep=clock.sleep, clock=clock.now)
        return pool.deploy_model(MODEL, ready_timeout=5)

    def chat_interrupted() -> Any:
        gw = FakeGateway(chat_mode="interrupted")
        gw.add_dep("ready")
        return gw.pool().chat([{"role": "user", "content": "hi"}])

    def stale_argument_token() -> Any:
        return FakeGateway().pool(owner_token="tok-wrong-0123456789abcdefghijklmnop").plan(MODEL)

    paths += [("unreachable", unreachable), ("interrupted", stream_mode("interrupted")),
              ("busy", stream_mode("busy")), ("cut", stream_mode("cut")), ("bad_event", stream_mode("not_json")),
              ("failed", failed), ("timeout", timeout), ("chat_interrupted", chat_interrupted),
              ("stale_argument_token", stale_argument_token)]
    return paths


@pytest.mark.parametrize(("name", "run"), _leak_paths(), ids=[n for n, _ in _leak_paths()])
def test_token_never_in_error_text(name: str, run: Callable[[], Any]) -> None:
    with pytest.raises(HypercomputeError) as info:
        run()
    err = info.value
    for text in (str(err), repr(err), err.detail, err.fix or "", " ".join(map(str, err.args))):
        assert FIRST_TOKEN not in text, name
        assert "tok-wrong" not in text, name
    assert isinstance(err, (HypercomputeError, StreamError, DeploymentFailed, ReadyTimeout))
