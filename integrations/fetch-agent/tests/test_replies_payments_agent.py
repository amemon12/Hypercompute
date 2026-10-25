import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from uagents_core.contrib.protocols.chat import ChatMessage, MetadataContent, TextContent
from uagents_core.contrib.protocols.chat.cards import (
    card_kind_for_payload,
    create_card_response_content,
    extract_card,
    validate_card_payload,
)
from uagents_core.contrib.protocols.payment import RequestPayment

from hc_agent.agent import build, chat_message, payment_request, read_chat
from hc_agent.payments import PaymentAsk, check_transfer_events, LedgerVerifier, to_atto
from hc_agent.planner import read_plan, snapshot
from hc_agent.replies import error_reply, help_reply, proposal_reply, result_reply, status_reply
from hc_agent.workflow import Proposal, RunOutcome, WorkflowError
from mock_gateway.server import MockState
from tests.conftest import GUEST, make_settings

WALLET = "fetch1" + "a" * 38


def _valid(reply) -> None:
    """Every card the agent sends passes the card protocol's own schema check, as ASI:One will apply it."""
    msg = chat_message(reply)
    meta = [c for c in msg.content if isinstance(c, MetadataContent)]
    assert len(meta) == 1
    card = extract_card(meta[0])
    assert card is not None and card.card_id == reply.card_id
    kind = card_kind_for_payload(reply.card)
    validate_card_payload(kind, json.loads(meta[0].metadata["card_payload"]))
    assert len(meta[0].metadata["card_payload"]) < 64 * 1024


def _recorded_view(recorded):
    pool = snapshot(recorded["nodes"], recorded["network"], MockState().model_views(), [])
    return pool, read_plan(recorded["plan"], pool)


def test_status_and_help_cards_validate() -> None:
    st = MockState()
    st.offline = {"mac-b"}
    pool = snapshot(st.node_views(), st.network(), st.model_views(), [])
    reply = status_reply(pool)
    assert "3 of 4" in reply.text and "stale" in reply.text or "no heartbeat" in reply.text
    _valid(reply)
    _valid(help_reply(member=False, address=GUEST, guest_access="read"))


def test_review_and_result_cards_validate(recorded: dict) -> None:
    pool, view = _recorded_view(recorded)
    kimi = next(m for m in pool.models if m.id == "kimi-k2.6-iq1_s")
    proposal = Proposal(pool=pool, model=kimi, view=view, strategy="all_selected", reuse=False)
    review = proposal_reply(proposal, prompt="a | pipe", max_tokens=256, ref="r1", price=Decimal("0.02024"))
    _valid(review)
    assert "0.02024 FET" in review.text and "| mac-azim |" in review.text
    assert review.card.approve_cta.selection == {"action": "approve", "ref": "r1"}
    outcome = RunOutcome(answer="Pooled answer.", reasoning=None, finish_reason="stop", model=kimi, view=view,
                         deployment_id="d", reused=False, decode_tok_s=1.54, prefill_tok_s=3.3, prompt_tokens=72,
                         output_tokens=64, total_s=44.0, load_s=366.0, participation_verified=True,
                         participation_detail=None)
    result = result_reply(outcome)
    _valid(result)
    assert "1.54 tok/s" in result.text and "loaded in 366 s" in result.text and "61" not in result.text.split("|")[0]


def test_reasoning_only_result_explains_itself(recorded: dict) -> None:
    pool, view = _recorded_view(recorded)
    kimi = next(m for m in pool.models if m.id == "kimi-k2.6-iq1_s")
    outcome = RunOutcome(answer="", reasoning="The user asks about RPC...", finish_reason="length", model=kimi,
                         view=view, deployment_id="d", reused=True, decode_tok_s=1.5, prefill_tok_s=2.8,
                         prompt_tokens=72, output_tokens=32, total_s=46.0, load_s=None, participation_verified=False,
                         participation_detail="not verified")
    text = result_reply(outcome).text
    assert "still thinking" in text and "The user asks about RPC" in text and "more tokens" in text


def test_error_reply_has_a_next_step() -> None:
    text = error_reply(WorkflowError("busy", "The pool is busy.", "ask again later")).text
    assert text == "**I couldn't finish that.** The pool is busy.\n\nNext: ask again later."


def test_read_chat_handles_text_card_responses_and_cancel() -> None:
    sel = create_card_response_content(card_id=uuid4(), selection={"action": "approve", "ref": "r"})
    assert read_chat(ChatMessage(content=[sel])) == ("", {"action": "approve", "ref": "r"}, False)
    assert read_chat(ChatMessage(content=[TextContent(text="hello")])) == ("hello", None, False)
    cancel = create_card_response_content(cancelled=True)
    assert read_chat(ChatMessage(content=[cancel]))[2] is True


def test_payment_request_shape() -> None:
    ask = PaymentAsk(amount=Decimal("0.020240"), currency="FET", recipient=WALLET, reference="ref1",
                     description="run")
    req = payment_request(ask)
    assert isinstance(req, RequestPayment)
    assert req.accepted_funds[0].amount == "0.02024" and req.accepted_funds[0].payment_method == "fet_direct"
    assert req.metadata["provider_agent_wallet"] == WALLET and req.metadata["fet_network"] == "stable-testnet"


def test_transfer_event_checks() -> None:
    amount = to_atto(Decimal("0.020653"))
    events = {"transfer": {"recipient": WALLET, "sender": "fetch1buyer", "amount": f"{amount}atestfet"}}
    kw = dict(recipient=WALLET, atto=amount, denom="atestfet")
    assert check_transfer_events(events, buyer_wallet="fetch1buyer", **kw).ok
    assert not check_transfer_events(events, buyer_wallet="fetch1other", **kw).ok
    assert not check_transfer_events(events, buyer_wallet=None, recipient=WALLET, atto=amount + 10**12,
                                     denom="atestfet").ok
    # the amount must match the quote to the micro-FET: another amount is someone else's payment (quotes are unique)
    assert not check_transfer_events(events, buyer_wallet=None, recipient=WALLET, atto=amount - 10**12,
                                     denom="atestfet").ok
    assert not check_transfer_events(events, buyer_wallet=None, recipient=WALLET, atto=amount, denom="afet").ok
    as_list = [{"type": "transfer", "attributes": [{"key": "recipient", "value": WALLET},
                                                   {"key": "amount", "value": f"5afet,{amount}atestfet"}]}]
    assert check_transfer_events(as_list, buyer_wallet=None, **kw).ok
    jitter = {"transfer": {"recipient": WALLET, "amount": f"{amount + 3}atestfet"}}  # a float round trip
    assert check_transfer_events(jitter, buyer_wallet=None, **kw).ok
    off = {"transfer": {"recipient": WALLET, "amount": f"{amount + 10**12}atestfet"}}  # another quote's tag
    assert not check_transfer_events(off, buyer_wallet=None, **kw).ok


async def test_ledger_verifier_rejects_replays(tmp_path, monkeypatch) -> None:
    amount = Decimal("0.02024")

    class Tx:
        events = {"transfer": {"recipient": WALLET, "sender": "fetch1buyer", "amount": f"{to_atto(amount)}atestfet"}}
        timestamp = None

        @staticmethod
        def is_successful() -> bool:
            return True

    v = LedgerVerifier(WALLET, "stable-testnet", tmp_path / "used.json", patience_s=0)
    monkeypatch.setattr(v, "_query", lambda tx_id: Tx())
    tx = "AB" * 32
    assert (await v.verify(tx, amount=amount, buyer_wallet="fetch1buyer")).ok
    again = await v.verify(tx, amount=amount, buyer_wallet="fetch1buyer")
    assert not again.ok and "already paid" in again.reason
    assert not (await v.verify("not-a-hash", amount=amount, buyer_wallet=None)).ok
    reloaded = LedgerVerifier(WALLET, "stable-testnet", tmp_path / "used.json", patience_s=0)
    assert not (await reloaded.verify(tx, amount=amount, buyer_wallet=None)).ok  # survives a restart

    class OldTx(Tx):
        timestamp = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)

    monkeypatch.setattr(v, "_query", lambda tx_id: OldTx())
    stale = await v.verify("EF" * 32, amount=amount, buyer_wallet=None,
                           not_before=datetime(2026, 10, 24, 13, 0, tzinfo=UTC))
    assert not stale.ok and "before" in stale.reason  # paid before this request was quoted
    fresh = await v.verify("EF" * 32, amount=amount, buyer_wallet=None,
                           not_before=datetime(2026, 10, 24, 12, 1, tzinfo=UTC))
    assert fresh.ok  # inside the clock-skew allowance

    def missing(_tx_id: str):
        raise RuntimeError("NotFoundError")

    monkeypatch.setattr(v, "_query", missing)
    lost = await v.verify("CD" * 32, amount=amount, buyer_wallet=None)
    assert "not found" in lost.reason and lost.retryable  # keep the request open, do not cancel


async def test_ledger_verifier_waits_for_indexing_and_reads_every_log(tmp_path, monkeypatch) -> None:
    amount = Decimal("0.020653")
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    class Log:
        def __init__(self, events):
            self.events = events

    class Tx:
        timestamp = None
        events = {"transfer": {"recipient": "fetch1other", "amount": "1atestfet"}}  # cosmpy keeps only the last
        logs = [Log({"transfer": {"recipient": WALLET, "amount": f"{to_atto(amount)}atestfet"}}),
                Log({"transfer": {"recipient": "fetch1other", "amount": "1atestfet"}})]

        @staticmethod
        def is_successful() -> bool:
            return True

    attempts = {"n": 0}

    def query(_tx_id: str):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("NotFoundError")
        return Tx()

    v = LedgerVerifier(WALLET, "stable-testnet", tmp_path / "used.json", sleep=fake_sleep)
    monkeypatch.setattr(v, "_query", query)
    assert (await v.verify("AB" * 32, amount=amount, buyer_wallet=None)).ok
    assert attempts["n"] == 3 and slept == [5.0, 5.0]


def test_corrupt_replay_file_fails_closed(tmp_path) -> None:
    from hc_agent.store import StoreError

    (tmp_path / "used.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreError):
        LedgerVerifier(WALLET, "stable-testnet", tmp_path / "used.json")


async def test_agent_builds_with_all_protocols(tmp_path) -> None:
    settings = make_settings(tmp_path, HC_MAILBOX="false", HC_AGENT_PORT="8799")
    agent, convo = build(settings)
    names = {p.name for p in agent.protocols.values()}
    assert {"AgentChatProtocol", "AgentPaymentProtocol", "HypercomputePool"} <= names
    assert agent.address.startswith("agent1")
    assert convo.wallet_address and convo.wallet_address.startswith("fetch1")


async def test_local_mode_publishes_nothing(tmp_path, monkeypatch) -> None:
    from hc_agent.agent import NoRegistration

    published = []

    async def spy(self, manifest):
        published.append(manifest)

    monkeypatch.setattr("uagents.agent.Agent.publish_manifest", spy)
    settings = make_settings(tmp_path, HC_MAILBOX="false", HC_AGENT_PORT="8798")
    agent, _ = build(settings)
    await __import__("asyncio").sleep(0)
    assert isinstance(agent._registration_policy, NoRegistration)  # no Almanac registration of a local rehearsal
    assert published == []  # no protocol manifests either


async def test_mailbox_mode_keeps_running_without_a_server(tmp_path, monkeypatch) -> None:
    """uAgents 0.26.1 drops its HTTP server when a mailbox agent has no REST routes, and the mailbox client's run()
    returns at once, so run_async would end immediately. The agent must stay up, still with no server."""
    import asyncio

    settings = make_settings(tmp_path, HC_MAILBOX="true", HC_AGENT_PORT="8797")
    agent, _ = build(settings)
    assert not agent._rest_handlers  # no inspector routes, so no local server
    started = []

    async def fake_original() -> None:  # stands in for the real client, which would poll Agentverse
        started.append(True)

    monkeypatch.setattr(agent._mailbox_client, "_check_mailbox_loop", fake_original)
    task = asyncio.ensure_future(agent._mailbox_client.run())
    await asyncio.sleep(0.2)
    assert started and not task.done()  # polling started, and run() keeps the agent alive
    task.cancel()
