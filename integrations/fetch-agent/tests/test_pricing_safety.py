from decimal import Decimal

import pytest

from hc_agent.config import ConfigError, Pricing, Settings, check_gateway_url, parse_addresses
from hc_agent.planner import read_plan, snapshot
from hc_agent.pricing import CreditLedger, price_for, split
from hc_agent.safety import Gatekeeper, SeenMessages, SlidingWindow
from tests.conftest import GUEST, MEMBER


def test_price_scales_with_size_and_tokens() -> None:
    p = Pricing()
    assert price_for(1000, 256, p) == Decimal("0.02024")  # 0.01 + 0.004 * 0.256 * 10
    assert price_for(120, 256, p) == Decimal("0.011229")
    assert price_for(0.5, 16, p) == Decimal("0.01")  # floor
    assert price_for(1000, 1_000_000, p) == Decimal("5")  # cap


def test_split_is_by_layers_and_sums_exactly(recorded: dict) -> None:
    pool = snapshot(recorded["nodes"], recorded["network"], [], [])
    view = read_plan(recorded["plan"], pool)
    total = Decimal("0.02024")
    shares = split(total, view, 0.10)
    assert sum(s.amount for s in shares) == total
    by = {s.node_name: s for s in shares}
    assert by["gx10-49d1"].role == "host" and by["gx10-49d1"].layers == 50
    assert by["mac-a"].layers == 3 and by["g14"].layers == 2
    pool_part = total - total * Decimal("0.10")
    assert by["mac-a"].amount == (pool_part * 3 / 61).quantize(Decimal("0.000001"), rounding="ROUND_DOWN")
    assert by["mac-a"].amount > by["g14"].amount


def test_ledger_accumulates(tmp_path, recorded: dict) -> None:
    pool = snapshot(recorded["nodes"], recorded["network"], [], [])
    shares = split(Decimal("1"), read_plan(recorded["plan"], pool), 0.1)
    ledger = CreditLedger(tmp_path / "credits.json")
    ledger.record(request_id="r1", model_id="kimi", payer=GUEST, currency="FET", shares=shares, transaction_id="T1")
    ledger.record(request_id="r2", model_id="kimi", payer=GUEST, currency="FET", shares=shares, transaction_id="T2")
    totals = ledger.totals()
    assert sum(Decimal(v["earned"]) for v in totals.values()) == Decimal("2")
    assert totals["6dad6cf75b2e4cf197b7bc7f4d0a67e4"]["layers_served"] == 6


def test_gatekeeper_roles() -> None:
    g = Gatekeeper(allowed=frozenset({MEMBER}), guest_access="read", payments="guests", runs_per_window=2,
                   window_s=60, messages_per_minute=100)
    assert g.action(MEMBER, "run").allowed and not g.action(MEMBER, "run").needs_payment
    assert g.action(GUEST, "status").allowed
    assert not g.action(GUEST, "run").allowed
    assert not g.action(GUEST, "stop").allowed
    g.guest_access = "paid"
    d = g.action(GUEST, "run")
    assert d.allowed and d.needs_payment
    g.guest_access = "none"
    assert not g.action(GUEST, "status").allowed and g.action(GUEST, "help").allowed
    g.payments = "everyone"
    assert g.action(MEMBER, "run").needs_payment


def test_rate_limits_with_a_fake_clock() -> None:
    now = [0.0]
    w = SlidingWindow(2, 60, clock=lambda: now[0])
    assert w.check("a") is None and w.check("a") is None
    assert w.check("a") == pytest.approx(60)
    assert w.check("b") is None
    now[0] = 61
    assert w.check("a") is None
    w.refund("a")
    assert w.check("a") is None


def test_seen_messages_dedupes() -> None:
    s = SeenMessages(capacity=2)
    assert s.first_time("1") and not s.first_time("1")
    s.first_time("2"), s.first_time("3")
    assert s.first_time("1")  # evicted, so a very old id is new again


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({})
    with pytest.raises(ConfigError, match="loopback|this machine"):
        check_gateway_url("http://10.0.0.5:8000")
    assert check_gateway_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    with pytest.raises(ConfigError):
        parse_addresses("agent1qgood" + "a" * 50 + ", not-an-address")
    with pytest.raises(ConfigError, match="accept-test"):
        Settings.from_env({"HC_AGENT_SEED": "x" * 40, "HC_PAYMENT_VERIFY": "accept-test"})
    with pytest.raises(ConfigError, match="may not set"):
        Settings.from_env({"HC_AGENT_SEED": "x" * 40, "HC_CHAT_EXTRA_JSON": '{"messages": []}'})
    s = Settings.from_env({"HC_AGENT_SEED": "x" * 40, "HC_CHAT_EXTRA_JSON": '{"chat_template_kwargs": {"x": 1}}'})
    assert s.chat_extra == {"chat_template_kwargs": {"x": 1}}
    assert "x" * 40 not in repr(s)  # the seed never appears in logs
