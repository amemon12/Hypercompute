from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from uagents_core.contrib.protocols.chat.cards import FormCardPayload, ReviewCardPayload

from hc_agent.conversation import Conversation
from hc_agent.payments import Verdict
from hc_agent.workflow import Workflow
from mock_gateway.server import MockState
from tests.conftest import GUEST, MEMBER, Recorder, make_settings

WALLET = "fetch1" + "w" * 38


class FakeVerifier:
    def __init__(self, ok: bool = True) -> None:
        self.ok, self.calls = ok, []

    async def verify(self, tx_id: str, *, amount: Decimal, buyer_wallet: str | None, not_before=None) -> Verdict:
        self.calls.append((tx_id, amount, buyer_wallet, not_before))
        return Verdict(self.ok, "verified" if self.ok else "the transaction was not found on the ledger yet")


def convo(workflow: Workflow, verifier: FakeVerifier | None = None, **settings_changes) -> Conversation:
    workflow.settings = replace(workflow.settings, **settings_changes)
    return Conversation(workflow.settings, workflow, address="agent1qself", wallet_address=WALLET,
                        verifier=verifier or FakeVerifier())


async def test_member_plans_approves_and_gets_the_answer(workflow: Workflow, out: Recorder) -> None:
    c = convo(workflow)
    await c.on_text(MEMBER, "Run 'explain RPC in one line' on the biggest model my friends' laptops can hold", out)
    review = out.last
    assert isinstance(review.card, ReviewCardPayload)
    assert "Kimi K2.6" in review.text and "mac-a" in review.text and "free" in review.text
    ref = review.card.approve_cta.selection["ref"]
    await c.on_selection(MEMBER, {"action": "approve", "ref": ref}, out)
    result = out.last
    assert result.text.startswith("[mock] Kimi K2.6")
    assert "1.54 tok/s" in result.text and "| mac-a |" in result.text and "engine log" in result.text
    assert any("Deploying" in t for t in out.texts)


async def test_plain_yes_approves_and_cancel_drops(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow)
    await c.on_text(MEMBER, "run 'hi' on the biggest model", out)
    await c.on_text(MEMBER, "cancel", out)
    assert "Cancelled" in out.last.text and not mock_state.deployments
    await c.on_text(MEMBER, "yes", out)
    assert isinstance(out.last.card, FormCardPayload)  # nothing to approve: offer the form instead
    await c.on_text(MEMBER, "run 'hi' on the biggest model", out)
    await c.on_text(MEMBER, "run it", out)
    assert out.last.text.startswith("[mock]")


async def test_loaded_model_answers_without_a_confirmation(workflow: Workflow, out: Recorder,
                                                          mock_state: MockState) -> None:
    c = convo(workflow)
    plan = mock_state.plan("gpt-oss-120b-mxfp4", "all_selected", None, 4096)
    mock_state.create(plan, "owner", foreign=True, ready=True)
    await c.on_text(MEMBER, "run 'hi' on the biggest model", out)
    assert out.last.text.startswith("[mock] gpt-oss 120B") and "was already loaded" in out.last.text


async def test_guest_is_refused_with_their_address(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow)
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    assert "allowlist" in out.last.text and GUEST in out.last.text and not mock_state.plans
    await c.on_text(GUEST, "who is online?", out)
    assert "4 of 4 friends" in out.last.text and out.last.card is not None


async def test_paid_guest_flow(tmp_path, workflow: Workflow, out: Recorder) -> None:
    verifier = FakeVerifier()
    c = convo(workflow, verifier, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    review = out.last
    await c.on_selection(GUEST, review.card.approve_cta.selection, out)
    ask = out.payments[-1]
    # 0.01 + 0.004 * (256 / 1000) * (1000 / 100) = 0.02024, plus a unique tag of 1 to 999 micro-FET
    assert Decimal("0.020241") <= ask.amount <= Decimal("0.021239") and ask.amount.as_tuple().exponent >= -6
    assert f"{ask.amount.normalize():f} FET" in review.text
    assert ask.recipient == WALLET and ask.currency == "FET"
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet="fetch1buyer")
    tx, amount, buyer, not_before = verifier.calls[0]
    assert out.completed == ["AB" * 32] and buyer == "fetch1buyer" and amount == ask.amount
    assert not_before is not None  # the chain check rejects payments made before this quote
    assert f"Paid {ask.amount.normalize():f} FET" in out.last.text and "mac-a" in out.last.text
    totals = c.ledger.totals()
    assert sum(Decimal(v["earned"]) for v in totals.values()) == ask.amount


async def test_quotes_are_unique_per_request(tmp_path, gateway) -> None:
    settings = make_settings(tmp_path, HC_GUEST_ACCESS="paid")
    c = Conversation(settings, Workflow(gateway, settings), address="agent1qself", wallet_address=WALLET,
                     verifier=FakeVerifier())
    other = "agent1q" + "o" * 58
    await c.on_text(GUEST, "run 'hi' on the biggest model", Recorder())
    await c.on_text(other, "run 'hi' on the biggest model", Recorder())
    assert c._pending[GUEST].price != c._pending[other].price


async def test_commit_must_match_the_quote_exactly(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    await c.on_text(GUEST, "yes", out)
    ask = out.payments[-1]
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount + Decimal("1")), currency="FET",
                      reference=ask.reference, buyer_wallet=None)
    assert out.cancelled and not mock_state.deployments


async def test_unverified_payment_runs_nothing(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow, FakeVerifier(ok=False), guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    await c.on_text(GUEST, "yes", out)
    ask = out.payments[-1]
    await c.on_commit(GUEST, out, tx_id="CD" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert out.cancelled and "couldn't verify" in out.last.text and not mock_state.deployments
    await c.on_commit(GUEST, out, tx_id="EF" * 32, amount="0.000001", currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert len(out.cancelled) == 2  # not awaiting payment any more, or underpaid: refused either way


async def test_payment_for_someone_elses_request_is_refused(workflow: Workflow, out: Recorder) -> None:
    c = convo(workflow, guest_access="paid")
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount="1", currency="FET", reference="x", buyer_wallet=None)
    assert out.cancelled and "no payment was requested" in out.cancelled[0][1]


async def test_failed_paid_run_keeps_the_payment(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    await c.on_text(GUEST, "yes", out)
    ask = out.payments[-1]
    mock_state.fail_loads = 2
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert "payment is kept" in out.last.text
    assert c.credits.latest(GUEST) is not None  # persisted, so cancel, a new request or a restart cannot lose it
    await c.on_text(GUEST, "run it", out)  # a fresh plan, covered by the earlier payment
    assert "covered by your earlier payment" in out.last.text
    await c.on_text(GUEST, "run it", out)
    assert out.last.text.startswith("[mock]") and len(out.payments) == 1  # retried without a second charge
    assert c.credits.latest(GUEST) is None


async def test_rate_limit_and_busy(tmp_path, gateway, out: Recorder) -> None:
    settings = make_settings(tmp_path, HC_RUNS_PER_WINDOW="1", HC_CONFIRM="never")
    c = Conversation(settings, Workflow(gateway, settings, poll_s=0.01), address="agent1qself")
    await c.on_text(MEMBER, "run 'one' on the biggest model", out)
    assert out.last.text.startswith("[mock]")
    await c.on_text(MEMBER, "run 'two' on the biggest model", out)
    assert "limit" in out.last.text
    async with c.workflow.run_lock:
        await c.on_text(MEMBER, "run 'three' on the biggest model", out)
    assert "busy" in out.last.text


async def test_message_flood_is_cut_off(tmp_path, gateway, out: Recorder) -> None:
    settings = make_settings(tmp_path, HC_MESSAGES_PER_MINUTE="2")
    c = Conversation(settings, Workflow(gateway, settings), address="agent1qself")
    for _ in range(3):
        await c.on_text(GUEST, "help", out)
    assert "limit for messages" in out.last.text


async def test_form_when_no_prompt_and_stop(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow, confirm="never")
    await c.on_text(MEMBER, "run something on the biggest model my friends' laptops can hold", out)
    form = out.last.card
    assert isinstance(form, FormCardPayload) and form.fields[0].name == "prompt"
    await c.on_selection(MEMBER, {**form.submit_cta.selection, "prompt": "hello", "model": "gpt-oss-120b-mxfp4"}, out)
    assert out.last.text.startswith("[mock] gpt-oss 120B")
    await c.on_text(GUEST, "stop", out)
    assert "allowlist" in out.last.text or "only members" in out.last.text
    await c.on_text(MEMBER, "stop", out)
    assert "Unloaded gpt-oss-120b-mxfp4" in out.last.text and mock_state.active() is None


async def test_errors_become_plain_replies(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow)
    await c.on_text(MEMBER, "use llama to answer 'hi'", out)
    assert out.last.text.startswith("**I couldn't finish that.**") and "Next:" in out.last.text
    await c.on_text(MEMBER, "run '" + "x" * 5000 + "' on the biggest model", out)
    assert "limit is" in out.last.text


async def _paid_guest(workflow: Workflow, out: Recorder, verifier=None) -> tuple[Conversation, object]:
    c = convo(workflow, verifier, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    await c.on_text(GUEST, "yes", out)
    return c, out.payments[-1]


async def test_duplicate_commit_completes_once(workflow: Workflow, out: Recorder) -> None:
    import asyncio

    gate = asyncio.Event()

    class SlowVerifier(FakeVerifier):
        async def verify(self, tx_id, **kw):
            await gate.wait()
            return await super().verify(tx_id, **kw)

    c, ask = await _paid_guest(workflow, out, SlowVerifier())
    kw = dict(tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference, buyer_wallet=None)
    first = asyncio.create_task(c.on_commit(GUEST, out, **kw))
    await asyncio.sleep(0)
    await c.on_commit(GUEST, out, **kw)  # a redelivery while the first is verifying
    gate.set()
    await first
    await c.on_commit(GUEST, out, **kw)  # and one after it completed
    assert out.completed == ["AB" * 32] and out.cancelled == []
    assert sum(1 for t in out.texts if t.startswith("[mock]")) == 1


async def test_unexpected_crash_in_a_paid_run_keeps_a_credit(workflow: Workflow, out: Recorder,
                                                            monkeypatch) -> None:
    c, ask = await _paid_guest(workflow, out)

    async def boom(*_a, **_k):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(workflow, "execute", boom)
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert "payment is kept" in out.last.text and "socket" not in out.last.text
    assert c.credits.latest(GUEST).tx_id == "AB" * 32


async def test_open_payment_request_is_not_replaced(workflow: Workflow, out: Recorder) -> None:
    c, ask = await _paid_guest(workflow, out)
    await c.on_text(GUEST, "run 'something else entirely' on the biggest model", out)
    assert "open payment request" in out.last.text and c._pending[GUEST].price == ask.amount


async def test_cancel_closes_the_payment_and_late_commits_are_recorded(workflow: Workflow, out: Recorder,
                                                                       tmp_path) -> None:
    c, ask = await _paid_guest(workflow, out)
    await c.on_text(GUEST, "cancel", out)
    assert out.cancelled[-1] == (None, "you cancelled this request")
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    log = (Path(workflow.settings.data_dir) / "payment_claims.jsonl").read_text(encoding="utf-8")
    assert "AB" * 32 in log and "the owner has a record" in out.cancelled[-1][1]


async def test_unconfirmed_payment_keeps_the_request_open(workflow: Workflow, out: Recorder) -> None:
    class NotYet(FakeVerifier):
        async def verify(self, tx_id, **kw):
            return Verdict(False, "the transaction was not found on the ledger yet", retryable=True)

    c, ask = await _paid_guest(workflow, out, NotYet())
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert out.cancelled == [] and c._pending[GUEST].awaiting_payment and "stays open" in out.last.text


async def test_guests_never_see_engine_details(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c, ask = await _paid_guest(workflow, out)
    mock_state.fail_loads = 2
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert "llama-server" not in out.last.text and "internal error" in out.last.text
    assert all("llama-server" not in t for t in out.texts)


async def test_asi1_is_not_called_for_guests(tmp_path, gateway, out: Recorder, monkeypatch) -> None:
    import hc_agent.conversation as conv

    seen = []

    async def spy(text, **kw):
        seen.append(kw.get("asi1_key"))
        return conv.parse_intent.__wrapped__(text) if hasattr(conv.parse_intent, "__wrapped__") else \
            __import__("hc_agent.intent", fromlist=["parse_rules"]).parse_rules(text)

    monkeypatch.setattr(conv, "parse_intent", spy)
    settings = make_settings(tmp_path, ASI_ONE_API_KEY="k")
    c = Conversation(settings, Workflow(gateway, settings), address="agent1qself")
    await c.on_text(GUEST, "who is online?", out)
    await c.on_text(MEMBER, "who is online?", out)
    assert seen == [None, "k"]


async def test_a_credit_never_covers_a_pricier_run(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c = convo(workflow, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on gpt-oss", out)
    await c.on_text(GUEST, "yes", out)
    ask = out.payments[-1]
    mock_state.fail_loads = 2
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert c.credits.latest(GUEST) is not None
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    assert "covered by your earlier payment" not in out.last.text
    await c.on_text(GUEST, "yes", out)
    assert len(out.payments) == 2 and out.payments[-1].amount > ask.amount  # a full quote for the bigger model
    assert c.credits.latest(GUEST) is not None  # kept for a run it does cover


async def _failed_paid_run(workflow: Workflow, out: Recorder, mock_state: MockState) -> tuple[Conversation, object]:
    c, ask = await _paid_guest(workflow, out)
    mock_state.fail_loads = 2
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    mock_state.fail_loads = 0
    return c, ask


async def test_a_credit_is_spent_on_the_auto_run_path(workflow: Workflow, out: Recorder,
                                                     mock_state: MockState) -> None:
    c, ask = await _failed_paid_run(workflow, out, mock_state)
    assert c.credits.latest(GUEST) is not None
    plan = mock_state.plan("kimi-k2.6-iq1_s", "all_selected", None, 4096)
    mock_state.create(plan, "owner", foreign=True, ready=True)  # the paid model is loaded: no confirmation step
    await c.on_text(GUEST, "run it", out)
    assert out.last.text.startswith("[mock] Kimi K2.6") and len(out.payments) == 1
    assert c.credits.latest(GUEST) is None  # spent: the next request is quoted again
    await c.on_text(GUEST, "run 'again' on the biggest model", out)
    assert "covered by your earlier payment" not in out.last.text


async def test_a_verified_payment_survives_a_cancelled_run(workflow: Workflow, out: Recorder, monkeypatch) -> None:
    import asyncio

    import pytest

    c, ask = await _paid_guest(workflow, out)

    async def cancelled(*_a, **_k):
        raise asyncio.CancelledError

    monkeypatch.setattr(workflow, "execute", cancelled)
    with pytest.raises(asyncio.CancelledError):  # an agent shutdown in the middle of the load
        await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET",
                          reference=ask.reference, buyer_wallet=None)
    saved = c.credits.latest(GUEST)
    assert saved is not None and saved.tx_id == "AB" * 32 and saved.amount == str(ask.amount)
    reloaded = Conversation(workflow.settings, workflow, address="agent1qself", wallet_address=WALLET,
                            verifier=FakeVerifier())
    assert reloaded.credits.latest(GUEST).tx_id == "AB" * 32  # on disk, so a restart keeps it


async def test_cancel_is_refused_while_verifying(workflow: Workflow, out: Recorder) -> None:
    import asyncio

    gate = asyncio.Event()

    class SlowVerifier(FakeVerifier):
        async def verify(self, tx_id, **kw):
            await gate.wait()
            return await super().verify(tx_id, **kw)

    c, ask = await _paid_guest(workflow, out, SlowVerifier())
    commit = asyncio.create_task(c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET",
                                             reference=ask.reference, buyer_wallet=None))
    await asyncio.sleep(0)
    await c.on_text(GUEST, "cancel", out)
    assert "can't be cancelled" in out.last.text and out.cancelled == []
    await c.on_commit(GUEST, out, tx_id="CD" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)  # the buyer paid a second time
    gate.set()
    await commit
    assert out.completed == ["AB" * 32] and out.texts[-1].startswith("---") is False
    log = (Path(workflow.settings.data_dir) / "payment_claims.jsonl").read_text(encoding="utf-8")
    assert "CD" * 32 in log  # the second transfer is on record for a refund


async def test_rejected_payments_are_recorded(workflow: Workflow, out: Recorder) -> None:
    c, ask = await _paid_guest(workflow, out, FakeVerifier(ok=False))
    await c.on_commit(GUEST, out, tx_id="AB" * 32, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    log = (Path(workflow.settings.data_dir) / "payment_claims.jsonl").read_text(encoding="utf-8")
    assert "AB" * 32 in log and "not verified" in log


async def test_undelivered_payment_request_does_not_block(workflow: Workflow, out: Recorder) -> None:
    c = convo(workflow, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)
    out.deliver_payments = False
    await c.on_text(GUEST, "yes", out)
    assert "couldn't deliver" in out.last.text and not c._pending[GUEST].awaiting_payment
    out.deliver_payments = True
    await c.on_text(GUEST, "yes", out)
    assert c._pending[GUEST].awaiting_payment and len(out.payments) == 2


async def test_payment_claims_are_bounded(workflow: Workflow, out: Recorder) -> None:
    from hc_agent.conversation import UNSOLICITED_CLAIMS_PER_SENDER, claimed_amount

    c = convo(workflow, guest_access="paid")
    other = "agent1q" + "o" * 58
    for _ in range(3):  # the same hash from the same sender is recorded once
        await c.on_commit(GUEST, out, tx_id="AB" * 32, amount="1", currency="FET", reference=None, buyer_wallet=None)
    await c.on_commit(other, out, tx_id="AB" * 32, amount="1", currency="FET", reference=None, buyer_wallet=None)
    await c.on_commit(GUEST, out, tx_id="not-a-hash", amount="x" * 5000, currency="FET", reference=None,
                      buyer_wallet=None)
    await c.on_commit(GUEST, out, tx_id="CD" * 32, amount="1e999999", currency="FET", reference=None,
                      buyer_wallet=None)
    for i in range(10):  # unsolicited commits hit a small per-sender quota
        await c.on_commit(GUEST, out, tx_id=f"{i:064x}", amount="1", currency="FET", reference=None,
                          buyer_wallet=None)
    lines = (Path(workflow.settings.data_dir) / "payment_claims.jsonl").read_text(encoding="utf-8").splitlines()
    mine = [line for line in lines if GUEST in line]
    assert len(mine) == UNSOLICITED_CLAIMS_PER_SENDER and any(other in line for line in lines)
    assert "not-a-hash" not in "".join(lines) and all(len(line) < 1024 for line in lines)
    assert '"claimed_amount": "unparseable"' in mine[1]
    assert claimed_amount("1e999999") == "unparseable" and claimed_amount("0.020653") == "0.020653"
    c.record_unmatched(GUEST, "EF" * 32, "1", "verified, credit not saved", verified=True)  # never capped
    assert "EF" * 32 in (Path(workflow.settings.data_dir) / "payment_claims.jsonl").read_text(encoding="utf-8")
