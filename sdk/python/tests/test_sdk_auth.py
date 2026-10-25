"""Connection rules and the owner token: loopback URLs, token precedence, lazy bootstrap, rotation, repr."""

# ruff: noqa: S105, S106 - fake owner tokens and source labels

from __future__ import annotations

import httpx
import pytest
from fake_gateway import FIRST_TOKEN, MODEL, SECOND_TOKEN, FakeGateway
from hypercompute_sdk import DEFAULT_URL, HypercomputeError, OwnerAuthError, Pool

ENV_TOKEN = "tok-env-0123456789abcdefghijklmnopqrstuvwx"  # noqa: S105 - a fake owner token
ARG_TOKEN = "tok-arg-0123456789abcdefghijklmnopqrstuvwx"  # noqa: S105


@pytest.mark.parametrize("url", ["http://192.168.1.20:8001", "https://127.0.0.1:8001", "http://example.com",
                                 "ftp://127.0.0.1", "http://user:pw@127.0.0.1:8001"])
def test_non_loopback_url_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="ssh -L 8001:127.0.0.1:8001") as info:
        Pool(url, transport=httpx.MockTransport(FakeGateway()))
    assert "pw" not in str(info.value)


def test_blank_or_wrong_type_url_is_refused() -> None:
    for url in ("", "  ", None, 8001, "127.0.0.1:8001", "http://[::1"):
        with pytest.raises(ValueError):
            Pool(url)  # type: ignore[arg-type]


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001", "http://localhost:8001", "http://[::1]:8001",
                                 "http://LOCALHOST:9000/"])
def test_loopback_urls_are_accepted(url: str) -> None:
    with Pool(url, transport=httpx.MockTransport(FakeGateway())) as pool:
        assert pool.health()["status"] == "ok"


def test_default_url_is_loopback_8001() -> None:
    assert DEFAULT_URL == "http://127.0.0.1:8001"
    with FakeGateway().pool() as pool:
        assert pool.base_url.startswith("http://127.0.0.1:8001")


def test_client_ignores_proxy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    gw = FakeGateway()
    with gw.pool() as pool:
        pool.health()
    assert gw.calls("GET", "/api/health")[0].url.host == "127.0.0.1"
    assert pool._client.trust_env is False


def test_reads_send_no_token_and_no_bootstrap() -> None:
    gw = FakeGateway()
    with gw.pool() as pool:
        pool.nodes()
        pool.deployments()
    assert gw.bootstraps == 0
    assert all("x-hypercompute-owner" not in r.headers for r in gw.requests)
    assert pool.token_source == "none yet"


def test_bootstrap_is_lazy_and_runs_once() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    assert gw.bootstraps == 0
    pool.plan(MODEL)
    pool.plan(MODEL)
    assert gw.bootstraps == 1
    assert pool.token_source == "bootstrap"
    assert [r.headers["x-hypercompute-owner"] for r in gw.calls("POST", "/api/plans")] == [FIRST_TOKEN] * 2


def test_argument_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERCOMPUTE_OWNER_TOKEN", ENV_TOKEN)
    gw = FakeGateway(token=ARG_TOKEN)
    pool = gw.pool(owner_token=ARG_TOKEN)
    pool.plan(MODEL)
    assert pool.token_source == "argument"
    assert gw.bootstraps == 0


def test_environment_beats_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERCOMPUTE_OWNER_TOKEN", ENV_TOKEN)
    gw = FakeGateway(token=ENV_TOKEN)
    pool = gw.pool()
    pool.plan(MODEL)
    assert pool.token_source == "environment"
    assert gw.bootstraps == 0
    assert gw.calls("POST", "/api/plans")[0].headers["x-hypercompute-owner"] == ENV_TOKEN


def test_from_env_reads_url_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERCOMPUTE_URL", "http://localhost:8123")
    monkeypatch.setenv("HYPERCOMPUTE_OWNER_TOKEN", ENV_TOKEN)
    gw = FakeGateway(token=ENV_TOKEN)
    pool = Pool.from_env(transport=httpx.MockTransport(gw))
    pool.plan(MODEL)
    assert pool.base_url.startswith("http://localhost:8123")
    assert gw.requests[0].url.port == 8123
    assert pool.token_source == "environment"


def test_from_env_defaults_and_refuses_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Pool.from_env(transport=httpx.MockTransport(FakeGateway())).base_url.startswith(DEFAULT_URL)
    monkeypatch.setenv("HYPERCOMPUTE_URL", "http://10.0.0.5:8001")
    with pytest.raises(ValueError):
        Pool.from_env()


def test_rotated_bootstrap_token_is_fetched_again_and_retried_once() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    pool.plan(MODEL)
    gw.rotate()
    plan = pool.plan(MODEL)
    assert plan["model_id"] == MODEL
    assert gw.bootstraps == 2
    sent = [r.headers["x-hypercompute-owner"] for r in gw.calls("POST", "/api/plans")]
    assert sent == [FIRST_TOKEN, FIRST_TOKEN, SECOND_TOKEN]


def test_rotation_retries_only_once() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    pool.plan(MODEL)
    gw.token = "tok-never-handed-out-0123456789abcdefghij"  # noqa: S105 - bootstrap keeps returning a stale one
    gw.overrides[("GET", "/api/owner/bootstrap")] = lambda r: httpx.Response(200, json={"owner_token": FIRST_TOKEN})
    with pytest.raises(OwnerAuthError) as info:
        pool.plan(MODEL)
    assert info.value.code == "owner_only"
    assert len(gw.calls("POST", "/api/plans")) == 3  # first ok, then one try and one retry


def test_argument_token_is_never_replaced() -> None:
    gw = FakeGateway()
    pool = gw.pool(owner_token=FIRST_TOKEN)
    pool.plan(MODEL)
    gw.rotate()
    with pytest.raises(OwnerAuthError) as info:
        pool.plan(MODEL)
    assert info.value.status == 403
    assert gw.bootstraps == 0
    assert pool.token_source == "argument"


def test_environment_token_is_never_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERCOMPUTE_OWNER_TOKEN", ENV_TOKEN)
    gw = FakeGateway()
    with pytest.raises(OwnerAuthError):
        gw.pool().plan(MODEL)
    assert gw.bootstraps == 0


def test_not_loopback_bootstrap_raises_owner_auth() -> None:
    gw = FakeGateway(loopback=False)
    with pytest.raises(OwnerAuthError, match="only available from this machine"):
        gw.pool().plan(MODEL)


def test_bootstrap_without_token_is_an_error() -> None:
    gw = FakeGateway()
    gw.overrides[("GET", "/api/owner/bootstrap")] = lambda r: httpx.Response(200, json={"version": "0.1.0"})
    with pytest.raises(HypercomputeError, match="no owner token") as info:
        gw.pool().plan(MODEL)
    assert info.value.code == "no_owner_token"


def test_403_other_code_does_not_bootstrap_again() -> None:
    gw = FakeGateway()
    pool = gw.pool()
    pool.plan(MODEL)
    gw.overrides[("POST", "/api/plans")] = lambda r: httpx.Response(403, json={"detail": "nope", "code": "other"})
    with pytest.raises(OwnerAuthError):
        pool.plan(MODEL)
    gw.overrides[("POST", "/api/plans")] = lambda r: httpx.Response(403, text="forbidden")
    with pytest.raises(OwnerAuthError):
        pool.plan(MODEL)
    assert gw.bootstraps == 1


def test_owner_token_argument_must_be_text() -> None:
    with pytest.raises(ValueError):
        Pool(owner_token="  ")


def test_repr_shows_source_never_token(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = FakeGateway()
    pool = gw.pool()
    assert "none yet" in repr(pool)
    pool.plan(MODEL)
    assert "bootstrap" in repr(pool) and FIRST_TOKEN not in repr(pool) and FIRST_TOKEN not in str(pool)
    assert "argument" in repr(gw.pool(owner_token=ARG_TOKEN)) and ARG_TOKEN not in repr(gw.pool(owner_token=ARG_TOKEN))
    monkeypatch.setenv("HYPERCOMPUTE_OWNER_TOKEN", ENV_TOKEN)
    env_pool = gw.pool()
    assert "environment" in repr(env_pool) and ENV_TOKEN not in repr(env_pool)
