"""End to end through the proxy: real x402 client and server code, mock gateway, stub facilitator."""

from __future__ import annotations

import json

from x402.http.utils import decode_payment_required_header, decode_payment_response_header

from conftest import PAY_TO, make_rig
from mock_gateway import REPLY
from x402proxy.pricing import quote_atomic

WORDS = len(REPLY.split(" "))  # the mock sends one word per token
CHAT = {"messages": [{"role": "user", "content": "Explain MoE routing in two sentences"}], "max_tokens": 48}


def sse_events(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        data = block.removeprefix("data: ").strip()
        if data and data != "[DONE]":
            out.append(json.loads(data))
    return out


def streamed_text(text: str) -> str:
    return "".join((c.get("delta") or {}).get("content") or "" for e in sse_events(text) for c in e.get("choices", []))


async def test_unpaid_chat_gets_402_with_quote_and_requirements(rig):
    async with rig.plain() as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 402
    required = decode_payment_required_header(r.headers["payment-required"])
    accept = required.accepts[0]
    assert accept.network == "eip155:84532" and accept.scheme == "exact"
    assert accept.pay_to.lower() == PAY_TO
    assert int(accept.amount) == quote_atomic(48, rig.config.pricing)
    body = r.json()
    assert body["testnet"] is True and body["max_tokens_reserved"] == 48
    assert body["split"][0]["machine"] == "gx10" and body["split"][0]["host"] is True
    assert sum(round(float(s["earns_usdc"]) * 1e6) for s in body["split"]) == int(accept.amount)
    assert rig.facilitator.settled == []


async def test_paid_stream_settles_once_and_credits_by_layers(rig):
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 200, r.text
    receipt = decode_payment_response_header(r.headers["payment-response"])
    assert receipt.success and receipt.transaction.startswith("0x")
    assert streamed_text(r.text).startswith("Mixture-of-experts routing")
    assert len(rig.facilitator.settled) == 1
    payment = rig.proxy.state.ledger.recent()[0]
    assert payment["status"] == "delivered" and payment["settle_point"] == "first_token"
    assert payment["tx"] == receipt.transaction and payment["completion_tokens"] == WORDS
    shares = {s["display_name"]: s for s in payment["shares"]}
    assert set(shares) == {"gx10", "mac-azim", "mac-a", "g14", "mac-b"}
    assert (shares["gx10"]["units"], shares["mac-a"]["units"], shares["mac-azim"]["layers"]) == (51, 2, "0-2")
    assert shares["mac-a"]["payout_address"] == "0x" + "cd" * 20
    assert sum(s["amount_atomic"] for s in payment["shares"]) == payment["amount_atomic"]


async def test_paid_non_stream_returns_completion_and_settles_after_reply(rig):
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "kimi-k2.6-iq1_s"
    assert body["choices"][0]["message"]["content"].startswith("Mixture-of-experts")
    assert body["usage"]["completion_tokens"] == WORDS
    assert rig.proxy.state.ledger.recent()[0]["settle_point"] == "complete"
    assert len(rig.facilitator.settled) == 1


async def test_pool_not_ready_answers_503_before_asking_for_payment(rig):
    await rig.mode("not_ready")
    async with rig.plain() as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 503 and "payment-required" not in r.headers
    assert r.json()["charged"] is False


async def test_busy_or_failing_pool_before_first_token_is_never_charged(rig):
    for mode, status in (("busy", 503), ("error_before_output", 502)):
        await rig.mode(mode)
        for stream in (True, False):
            async with rig.paying() as c:
                r = await c.post("/v1/chat/completions", json={**CHAT, "stream": stream})
            assert r.status_code == status, (mode, stream, r.text)
            assert r.json()["charged"] is False
    assert rig.facilitator.settled == [] and len(rig.facilitator.verified) == 4
    assert rig.proxy.state.ledger.recent() == []


async def test_peer_lost_mid_stream_is_recorded_with_a_refund_owed(rig):
    await rig.mode("drop_mid_stream")
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 200
    events = sse_events(r.text)
    assert events[-1]["error"]["code"] == "interrupted"
    payment = rig.proxy.state.ledger.recent()[0]
    assert payment["status"] == "interrupted" and payment["completion_tokens"] == WORDS // 2
    expected_refund = payment["amount_atomic"] - quote_atomic(WORDS // 2, rig.config.pricing)
    assert payment["refund_owed_atomic"] == expected_refund > 0
    assert sum(s["amount_atomic"] for s in payment["shares"]) == payment["amount_atomic"] - expected_refund


async def test_peer_lost_in_non_stream_chat_is_not_charged(rig):
    await rig.mode("drop_mid_stream")
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 502 and r.json()["charged"] is False
    assert rig.facilitator.settled == []


async def test_failed_settlement_returns_402_and_records_nothing(tmp_path):
    rig = make_rig(tmp_path)
    rig.facilitator.fail_settle = True
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 402
    assert rig.proxy.state.ledger.recent() == []


async def test_bad_requests_are_refused_before_pricing(rig):
    async with rig.plain() as c:
        assert (await c.post("/v1/chat/completions", content=b"{not json")).status_code == 400
        r = await c.post("/v1/chat/completions", json={**CHAT, "resume_from": "run1"})
        assert r.status_code == 400 and "resume_from" in r.json()["detail"]
        assert (await c.post("/v1/chat/completions", json={"messages": []})).status_code == 400
        assert (await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 0})).status_code == 400
        r = await c.post("/v1/chat/completions", json={**CHAT, "model": "gpt-4o"})
        assert r.status_code == 404
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 100_000})
        assert r.status_code == 402 and r.json()["max_tokens_reserved"] == rig.config.pricing.max_tokens_cap
        for bad in ({"x": {"nested": 1}}, ["list"], {f"k{i}": i for i in range(9)}):
            r = await c.post("/v1/chat/completions", json={**CHAT, "chat_template_kwargs": bad})
            assert r.status_code == 400
        r = await c.post("/v1/chat/completions", json={**CHAT, "chat_template_kwargs": {"reasoning_effort": "low"}})
        assert r.status_code == 402


async def test_second_paid_chat_while_slot_is_held_gets_503(rig):
    slots = rig.proxy.state.slots
    await slots.acquire()
    try:
        async with rig.plain() as c:
            r = await c.post("/v1/chat/completions", json=CHAT)
        assert r.status_code == 503 and r.json()["error"] == "busy"
    finally:
        slots.release()


async def test_quote_models_ledger_and_health_are_free(rig):
    async with rig.plain() as c:
        q = (await c.get("/x402/quote", params={"max_tokens": 256})).json()
        assert q["price_usdc"] == "0.006120" and [s["units"] for s in q["split"]] == [51, 3, 3, 3, 2]
        assert (await c.get("/v1/models")).json()["data"][0]["id"] == "kimi-k2.6-iq1_s"
        assert "Hypercompute pool ledger" in (await c.get("/ledger")).text
        assert (await c.get("/api/ledger")).json()["summary"]["payments"] == 0
        assert (await c.get("/health")).json()["testnet"] is True


# ---------------------------------------------------------------- review fixes
async def test_empty_reply_is_never_charged(rig):
    await rig.mode("empty_reply")
    for stream in (True, False):
        async with rig.paying() as c:
            r = await c.post("/v1/chat/completions", json={**CHAT, "stream": stream})
        assert r.status_code == 502 and r.json()["charged"] is False, (stream, r.text)
    assert rig.facilitator.settled == []


async def test_buyer_gone_before_settlement_is_never_charged(rig, monkeypatch):
    from starlette.requests import Request

    async def gone(self):
        return True

    monkeypatch.setattr(Request, "is_disconnected", gone)
    for stream in (True, False):
        async with rig.paying() as c:
            await c.post("/v1/chat/completions", json={**CHAT, "stream": stream})
    assert rig.facilitator.settled == [] and len(rig.facilitator.verified) == 2
    assert rig.proxy.state.ledger.recent() == []


async def test_ledger_failure_after_settlement_still_delivers_and_keeps_a_record(rig, monkeypatch):
    import sqlite3

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(rig.proxy.state.ledger, "record_settlement", broken)
    for stream in (True, False):
        async with rig.paying() as c:
            r = await c.post("/v1/chat/completions", json={**CHAT, "stream": stream})
        assert r.status_code == 200 and "Mixture-of-experts" in r.text, stream
    fallback = rig.config.ledger_path.with_name("unrecorded-settlements.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in fallback.splitlines()]
    settlements = [r for r in records if r["kind"] == "settlement"]
    outcomes = [r for r in records if r["kind"] == "outcome"]
    assert [r["payment"]["tx"] for r in settlements] == [tx for tx, _ in rig.facilitator.settled]
    assert all(sum(s["amount_atomic"] for s in r["shares"]) == r["payment"]["amount_atomic"] for r in settlements)
    assert sorted(o["payment_id"] for o in outcomes) == sorted(r["payment"]["id"] for r in settlements)
    assert {o["outcome"] for o in outcomes} == {"delivered"}


async def test_hostile_bodies_are_refused_cleanly(tmp_path):
    rig = make_rig(tmp_path, X402_MAX_PROMPT_CHARS="100")
    async with rig.plain() as c:
        assert (await c.post("/v1/chat/completions", content=b"x" * (1024 * 1024 + 1))).status_code == 413
        nan = b'{"messages":[{"role":"user","content":"hi"}],"temperature":NaN}'
        assert (await c.post("/v1/chat/completions", content=nan)).status_code == 400
        assert (await c.post("/v1/chat/completions", content=b"[" * 100_000)).status_code == 400
        digits = b'{"max_tokens":' + b"9" * 5000 + b"}"
        assert (await c.post("/v1/chat/completions", content=digits)).status_code == 400
        long = {"messages": [{"role": "user", "content": "x" * 101}]}
        assert (await c.post("/v1/chat/completions", json=long)).status_code == 400
        for bad in ({"role": "tool", "content": "x"}, {"role": "user", "content": ["x"]}):
            assert (await c.post("/v1/chat/completions", json={"messages": [bad]})).status_code == 400
        for key, value in (("temperature", "hot"), ("top_k", 1.5), ("stop", ["x"] * 5), ("seed", True)):
            assert (await c.post("/v1/chat/completions", json={**CHAT, key: value})).status_code == 400, key


async def test_only_rebuilt_messages_and_checked_sampling_reach_the_gateway(rig):
    body = {"messages": [{"role": "user", "content": "hi", "name": "x", "tool_calls": [{"id": 1}]}],
            "max_tokens": 48, "temperature": 0.5, "stop": "END", "stream": True}
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    forwarded = rig.mock.state.mock["last_body"]
    assert forwarded["messages"] == [{"role": "user", "content": "hi"}]
    assert forwarded["temperature"] == 0.5 and forwarded["stop"] == ["END"] and forwarded["max_tokens"] == 48
    assert set(forwarded) == {"messages", "temperature", "stop", "max_tokens", "stream"}


async def test_rate_limit_answers_429(tmp_path):
    rig = make_rig(tmp_path, X402_RATE_PER_MIN="3")
    async with rig.plain() as c:
        codes = [(await c.post("/v1/chat/completions", json=CHAT)).status_code for _ in range(4)]
    assert codes == [402, 402, 402, 429]


async def test_facilitator_error_text_never_reaches_the_buyer(rig):
    async def leaky(payload, requirements):
        raise ValueError("facilitator HTTP 500: internal trace SECRET-BODY")

    rig.facilitator.verify = leaky
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 402
    assert "SECRET" not in r.headers.get("payment-required", "") + r.text
    assert "SECRET" not in decode_payment_required_header(r.headers["payment-required"]).model_dump_json()


async def test_facilitator_400_body_becomes_a_reason_code_only(rig):
    async def refusing(payload, requirements):
        raise ValueError('Facilitator verify failed (400): {"invalidMessage":"contract call failed: SECRET trace",'
                         '"invalidReason":"invalid_payload","isValid":false,"payer":"0x' + "12" * 20 + '"}')

    rig.facilitator.verify = refusing
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 402
    required = decode_payment_required_header(r.headers["payment-required"])
    assert required.error == "invalid_payload"
    assert "SECRET" not in required.model_dump_json() + r.text


# ---------------------------------------------------------------- second review
async def test_payer_whose_payments_keep_failing_to_settle_is_refused(tmp_path):
    from eth_account import Account

    rig = make_rig(tmp_path)
    rig.facilitator.fail_settle = True  # e.g. the buyer spends the USDC elsewhere right after verify
    payer = Account.create()
    for _ in range(3):
        async with rig.paying(payer) as c:
            assert (await c.post("/v1/chat/completions", json={**CHAT, "stream": True})).status_code == 402
    async with rig.paying(payer) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 429 and r.json()["error"] == "payer_blocked" and r.json()["charged"] is False
    assert rig.facilitator.settle_calls == 3 and rig.mock.state.mock["chats"] == 3  # no pool time for the 4th
    async with rig.paying() as c:  # another payer is unaffected
        assert (await c.post("/v1/chat/completions", json=CHAT)).status_code == 402


async def test_buyer_gone_during_settlement_gets_a_full_refund_booked(rig, monkeypatch):
    from starlette.requests import Request

    async def gone_after_settling(self):
        return bool(rig.facilitator.settled)

    monkeypatch.setattr(Request, "is_disconnected", gone_after_settling)
    async with rig.paying() as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 499
    payment = rig.proxy.state.ledger.recent()[0]
    assert payment["status"] == "interrupted" and payment["refund_owed_atomic"] == payment["amount_atomic"]
    assert sum(s["amount_atomic"] for s in payment["shares"]) == 0


async def test_surrogates_and_infinities_are_refused_not_500(rig):
    async with rig.plain() as c:
        surrogate = b'{"messages":[{"role":"user","content":"\\ud800"}]}'
        assert (await c.post("/v1/chat/completions", content=surrogate)).status_code == 400
        inf_kwargs = b'{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"a":1e999}}'
        assert (await c.post("/v1/chat/completions", content=inf_kwargs)).status_code == 400
        inf_temp = b'{"messages":[{"role":"user","content":"hi"}],"temperature":1e999}'
        assert (await c.post("/v1/chat/completions", content=inf_temp)).status_code == 400
