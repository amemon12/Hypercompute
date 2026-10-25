"""Pricing, split, ledger, config and SSE parsing on their own."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from x402proxy.config import PUBLIC_FACILITATOR_URL, ConfigError, load, parse_payees
from x402proxy.ledger import Ledger, Payment
from x402proxy.pricing import (PriceError, Pricing, atomic_to_usd, quote_atomic, reserved_tokens,
                               usd_to_atomic)
from x402proxy.split import Contribution, SplitError, contributions, layer_ranges, split
from x402proxy.sse import EventSplitter, SSEError, parse_event

PLAN = json.loads((Path(__file__).resolve().parent.parent / "fixtures" / "kimi-k2.6-pooled-plan.json")
                  .read_text(encoding="utf-8"))["plan"]
HOST = "0x" + "11" * 20
PRICING = Pricing(base_atomic=1000, per_1k_atomic=20000, default_max_tokens=256, max_tokens_cap=1024)


# ---------------------------------------------------------------- pricing
def test_money_parses_to_exact_atomic_units():
    assert usd_to_atomic("$0.01") == 10_000
    assert usd_to_atomic("0.000001") == 1
    assert usd_to_atomic("$2") == 2_000_000
    for bad in ("$0.0000001", "abc", "-1", "$", "1e-3"):
        with pytest.raises(PriceError):
            usd_to_atomic(bad)


def test_atomic_formats_with_six_decimals():
    assert atomic_to_usd(6120) == "0.006120"
    assert atomic_to_usd(2_500_000) == "2.500000"
    with pytest.raises(PriceError):
        atomic_to_usd(-1)


def test_quote_is_base_plus_per_token_rounded_up():
    assert quote_atomic(256, PRICING) == 1000 + 5120
    tiny = Pricing(base_atomic=0, per_1k_atomic=1, default_max_tokens=1, max_tokens_cap=10)
    assert quote_atomic(1, tiny) == 1  # 0.001 atomic rounds up, never down to free
    with pytest.raises(PriceError):
        quote_atomic(0, PRICING)


def test_reserved_tokens_defaults_clamps_and_rejects():
    assert reserved_tokens(None, PRICING) == 256
    assert reserved_tokens(5000, PRICING) == 1024
    for bad in (0, -3, True, "12", 1.5):
        with pytest.raises(PriceError):
            reserved_tokens(bad, PRICING)


def test_pricing_refuses_free_or_inconsistent_settings():
    with pytest.raises(PriceError):
        Pricing(base_atomic=0, per_1k_atomic=0, default_max_tokens=1, max_tokens_cap=1)
    with pytest.raises(PriceError):
        Pricing(base_atomic=1, per_1k_atomic=1, default_max_tokens=10, max_tokens_cap=5)


# ---------------------------------------------------------------- split
def test_real_plan_units_match_llama_tensor_split():
    contribs = contributions(PLAN, {"local": "gx10"})
    assert contribs[0].is_host and contribs[0].display_name == "gx10" and contribs[0].layers == "11-60"
    assert sorted(c.units for c in contribs) == sorted(PLAN["tensor_split"])


@pytest.mark.parametrize("fee_bps", [0, 1000, 2500, 10_000])
def test_shares_always_sum_to_the_amount(fee_bps):
    contribs = contributions(PLAN, {})
    for amount in [0, 1, 7, 61, 62, 999, 6120, 123_457, 10**9 + 3]:
        shares = split(amount, contribs, fee_bps, {}, HOST)
        assert sum(s.amount_atomic for s in shares) == amount
        assert all(s.amount_atomic >= 0 for s in shares)


def test_host_takes_fee_and_remainder_peers_take_floor_of_their_units():
    contribs = contributions(PLAN, {"local": "gx10"})
    shares = {s.display_name: s for s in split(6120, contribs, 1000, {}, HOST)}
    pool = 6120 - 612
    assert shares["gx10"].fee_atomic == 612
    peers = [s for s in shares.values() if not s.is_host]
    assert [s.amount_atomic for s in peers] == [pool * s.units // 62 for s in peers]
    assert shares["gx10"].amount_atomic == 6120 - sum(s.amount_atomic for s in peers)
    assert shares["gx10"].payout_address == HOST


def test_payout_addresses_match_by_name_or_node_id():
    contribs = [Contribution("local", "host", 10, "0-9", True), Contribution("abc", "mac-a", 5, "10-14", False),
                Contribution("def", "mac-b", 5, "15-19", False)]
    shares = {s.display_name: s for s in split(100, contribs, 0, {"mac-a": "0xA", "def": "0xB"}, HOST)}
    assert (shares["mac-a"].payout_address, shares["mac-b"].payout_address) == ("0xA", "0xB")


def test_plan_without_a_host_layer_still_pays_the_host_fee():
    plan = {"assignments": [{"node_id": "peer1", "layers": [0, 1, 2], "holds_output": True}]}
    shares = split(1000, contributions(plan, {}), 1000, {}, HOST)
    host = next(s for s in shares if s.is_host)
    assert host.units == 0 and host.amount_atomic == 100


def test_split_rejects_bad_inputs():
    with pytest.raises(SplitError):
        contributions({"assignments": []}, {})
    contribs = contributions(PLAN, {})
    with pytest.raises(SplitError):
        split(100, contribs, 10_001, {}, HOST)
    with pytest.raises(SplitError):
        split(-1, contribs, 0, {}, HOST)
    with pytest.raises(SplitError):
        split(100, [c for c in contribs if not c.is_host], 0, {}, HOST)


def test_layer_ranges():
    assert layer_ranges([]) == "-"
    assert layer_ranges([3, 0, 1, 2, 7, 9, 8]) == "0-3,7-9"
    assert layer_ranges([5]) == "5"


# ---------------------------------------------------------------- ledger
def _payment(pid: str, amount: int) -> Payment:
    return Payment(id=pid, settle_point="first_token", tx="0x" + "a" * 64, network="eip155:84532", payer="0xP",
                   amount_atomic=amount, max_tokens=256, model_id="m", deployment_id="d", plan_id="p")


def test_ledger_totals_and_interruption_resplit(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite3")
    contribs = contributions(PLAN, {"local": "gx10"})
    ledger.record_settlement(_payment("p1", 6120), split(6120, contribs, 1000, {}, HOST))
    ledger.record_settlement(_payment("p2", 6120), split(6120, contribs, 1000, {}, HOST))
    ledger.mark_delivered("p1", 200)
    ledger.mark_interrupted("p2", 40, refund_owed_atomic=4000, note="peer lost", host_fee_bps=1000)
    summary = ledger.summary()
    assert (summary["payments"], summary["paid_atomic"], summary["refunds_owed_atomic"]) == (2, 12240, 4000)
    assert sum(t["earned_atomic"] for t in ledger.totals()) == 12240 - 4000
    p2 = next(p for p in ledger.recent() if p["id"] == "p2")
    assert p2["status"] == "interrupted" and sum(s["amount_atomic"] for s in p2["shares"]) == 2120
    assert ledger.totals()[0]["display_name"] == "gx10"
    ledger.close()


def test_ledger_refuses_shares_that_do_not_sum(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite3")
    shares = split(100, contributions(PLAN, {}), 0, {}, HOST)
    with pytest.raises(ValueError):
        ledger.record_settlement(_payment("p1", 101), shares)
    assert ledger.recent() == []
    with pytest.raises(KeyError):
        ledger.mark_interrupted("missing", 1, 1, "x", 0)
    ledger.close()


# ---------------------------------------------------------------- config
BASE = {"X402_PAY_TO": "0x" + "ab" * 20}


def test_config_defaults_to_base_sepolia_and_public_facilitator():
    config = load(BASE)
    assert config.network == "eip155:84532" and config.facilitator == PUBLIC_FACILITATOR_URL
    assert config.pricing.base_atomic == 1000 and config.pricing.per_1k_atomic == 20000
    assert config.bind_host == "127.0.0.1" and config.port == 8402
    assert config.gateway_url == "http://127.0.0.1:8001"
    assert load({**BASE, "HYPERCOMPUTE_GATEWAY": "http://127.0.0.1:18001/"}).gateway_url == "http://127.0.0.1:18001"


def test_config_picks_cdp_when_keys_exist_and_refuses_mainnet():
    assert load({**BASE, "CDP_API_KEY_ID": "id", "CDP_API_KEY_SECRET": "s"}).facilitator == "cdp"
    with pytest.raises(ConfigError):
        load({**BASE, "X402_FACILITATOR": "cdp"})
    with pytest.raises(ConfigError):
        load({**BASE, "X402_NETWORK": "eip155:8453"})  # Base mainnet
    with pytest.raises(ConfigError):
        load({"X402_PAY_TO": "0x123"})
    with pytest.raises(ConfigError):
        load({**BASE, "X402_FACILITATOR": "http://plain-http.example"})


def test_payees_parse_and_validate():
    assert dict(parse_payees("mac-a=0x" + "cd" * 20 + ", g14=0x" + "ef" * 20)) == {
        "mac-a": "0x" + "cd" * 20, "g14": "0x" + "ef" * 20}
    assert dict(parse_payees("")) == {}
    for bad in ("mac-a", "mac-a=0x12", "=0x" + "cd" * 20):
        with pytest.raises(ConfigError):
            parse_payees(bad)


# ---------------------------------------------------------------- sse
def test_splitter_reassembles_events_across_chunks_and_crlf():
    splitter = EventSplitter()
    first = b'data: {"choices":[{"delta":{"content":"Hel'
    assert splitter.feed(first) == []
    events = splitter.feed(b'lo"}}]}\r\n\r\n: keep-alive\n\ndata: [DONE]\n\n')
    assert [e.text for e in events] == ["Hello", "", ""]
    assert events[-1].done and events[1].payload is None


def test_parse_event_reads_errors_finish_and_usage():
    err = parse_event(b'data: {"error": {"code": "busy", "message": "slots"}}\n\n')
    assert err.error == {"code": "busy", "message": "slots"}
    end = parse_event(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":9}}\n\n')
    assert end.finish_reason == "stop" and end.usage["completion_tokens"] == 9
    with pytest.raises(SSEError):
        parse_event(b"data: not json\n\n")


# ---------------------------------------------------------------- payouts
def test_payout_reservation_counts_pending_and_blocks_double_pay(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite3")
    contribs = contributions(PLAN, {"local": "gx10"})
    ledger.record_settlement(_payment("p1", 100_000), split(100_000, contribs, 1000, {}, HOST))
    owed = {o["display_name"]: o for o in ledger.owed()}
    assert "gx10" not in owed and len(owed) == 4
    peer = owed["4c2991743c15"]
    node, name, total = peer["node_id"], peer["display_name"], peer["owed_atomic"]
    first = ledger.reserve_payout(node, name, "0x" + "cd" * 20, total - 10, "eip155:84532")
    assert {o["display_name"]: o["owed_atomic"] for o in ledger.owed()}[name] == 10  # pending already counts
    with pytest.raises(ValueError):
        ledger.reserve_payout(node, name, "0x" + "cd" * 20, 11, "eip155:84532")  # a second run cannot overpay
    ledger.complete_payout(first, "0x" + "b" * 64)
    second = ledger.reserve_payout(node, name, "0x" + "cd" * 20, 10, "eip155:84532")
    ledger.fail_payout(second)  # checked: never sent, so it is owed again
    assert {o["display_name"]: o["owed_atomic"] for o in ledger.owed()}[name] == 10
    row = next(t for t in ledger.totals() if t["display_name"] == name)
    assert (row["paid_out_atomic"], row["pending_payout_atomic"]) == (total - 10, 0)
    assert [p["status"] for p in ledger.payouts()] == ["failed", "sent"]
    with pytest.raises(ValueError):
        ledger.reserve_payout(node, name, "0x", 0, "eip155:84532")
    ledger.close()


def test_plan_payouts_needs_an_address_and_the_minimum(tmp_path):
    from payout import plan_payouts

    ledger = Ledger(tmp_path / "l.sqlite3")
    names = {"local": "gx10", "4c2991743c154be896d10a0b53ba4d2b": "mac-azim", "8f169e3549fc41fbb14f6876a19d704c": "mac-a"}
    ledger.record_settlement(_payment("p1", 100_000), split(100_000, contributions(PLAN, names), 1000, {}, HOST))
    pay, skip = plan_payouts(ledger, {"mac-azim": "0x" + "cd" * 20, "mac-a": "0x" + "ef" * 20}, min_atomic=3000)
    assert [p["display_name"] for p in pay] == ["mac-azim"]  # 4354 owed; mac-a has 2903, under the minimum
    reasons = {s["display_name"]: s["reason"] for s in skip}
    assert reasons["mac-a"].startswith("under") and reasons["6dad6cf75b2e"] == "no payout address"
    ledger.close()


def test_ledger_reopens_and_migrates_old_payouts_table(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE payouts (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, node_id TEXT "
               "NOT NULL, display_name TEXT NOT NULL, address TEXT NOT NULL, amount_atomic INTEGER NOT NULL, tx TEXT "
               "NOT NULL, network TEXT NOT NULL)")
    db.commit()
    db.close()
    ledger = Ledger(path)
    assert ledger.payouts() == [] and ledger.owed() == []
    ledger.close()


async def test_guarded_stream_runs_cleanup_when_the_client_vanishes_mid_stream():
    from x402proxy.relay import GuardedStream

    closed = []

    async def body():
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    async def on_close():
        closed.append(True)

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client went away")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    from starlette.requests import ClientDisconnect

    with pytest.raises(ClientDisconnect):  # Starlette turns the send error into ClientDisconnect
        await GuardedStream(body(), on_close)({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert closed == [True]


async def test_race_disconnect_cancels_work_when_the_buyer_leaves():
    import asyncio

    from x402proxy.relay import race_disconnect

    async def slow():
        await asyncio.sleep(30)

    async def quick():
        return 7

    async def gone():
        return True

    async def here():
        return False

    assert await race_disconnect(slow(), gone) == (True, None)
    assert await race_disconnect(quick(), here) == (False, 7)


def test_reconcile_replays_settlements_and_outcomes_once(tmp_path):
    import json
    from dataclasses import asdict

    from x402proxy.reconcile import replay

    ledger = Ledger(tmp_path / "l.sqlite3")
    shares = split(6120, contributions(PLAN, {"local": "gx10"}), 1000, {}, HOST)
    fallback = tmp_path / "unrecorded-settlements.jsonl"
    lines = [{"kind": "settlement", "payment": asdict(_payment("p9", 6120)), "shares": [asdict(s) for s in shares]},
             {"kind": "outcome", "payment_id": "p9", "outcome": "interrupted", "completion_tokens": 0,
              "refund_owed_atomic": 6120, "note": "buyer left", "host_fee_bps": 1000}]
    fallback.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    assert replay(ledger, fallback) == {"settlements": 1, "outcomes": 1, "skipped": 0}
    payment = ledger.recent()[0]
    assert payment["id"] == "p9" and payment["status"] == "interrupted" and payment["refund_owed_atomic"] == 6120
    assert not fallback.exists() and replay(ledger, fallback) == {"settlements": 0, "outcomes": 0, "skipped": 0}
    with pytest.raises(KeyError):
        ledger.mark_delivered("missing", 1)
    ledger.close()
