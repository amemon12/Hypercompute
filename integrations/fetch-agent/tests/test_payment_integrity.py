"""Regression tests for the adversarial review of bd0de3d: one payment buys at most one answer, a payment that
bought something never lands on the refund list, and saved payments survive upgrades, failed writes and failed
deliveries."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hc_agent.conversation import Conversation
from hc_agent.intent import Intent
from hc_agent.store import Credit, Credits
from hc_agent.workflow import Workflow
from mock_gateway.server import MockState
from tests.conftest import GUEST, MEMBER, Recorder
from tests.test_conversation import WALLET, FakeVerifier, _failed_paid_run, _paid_guest, convo

AB = "AB" * 32


def _claims(workflow: Workflow) -> str:
    path = Path(workflow.settings.data_dir) / "payment_claims.jsonl"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _answers(out: Recorder) -> list[str]:
    return [t for t in out.texts if t.startswith("[mock]")]


async def test_a_stale_covered_plan_cannot_spend_a_used_credit(workflow: Workflow, mock_state: MockState) -> None:
    g, m = Recorder(), Recorder()
    c, ask = await _failed_paid_run(workflow, g, mock_state)
    assert [cr.tx_id for cr in c.credits.for_sender(GUEST)] == [AB]
    await c.on_text(GUEST, "use gpt-oss to answer 'first'", g)  # P1, covered by AB, waits for confirmation
    assert c._pending[GUEST].covered_tx == AB
    await c.on_text(MEMBER, "use gpt-oss to answer 'member'", m)
    await c.on_text(MEMBER, "yes", m)  # gpt-oss is loaded now
    await c.on_text(GUEST, "use gpt-oss to answer 'second'", g)  # reuse, no confirmation: spends AB
    assert len(_answers(g)) == 1 and c.credits.for_sender(GUEST) == []
    await c.on_text(GUEST, "yes", g)  # the stale plan
    assert len(_answers(g)) == 1 and len(g.payments) == 1  # still one answer for one payment
    entries = json.loads((Path(workflow.settings.data_dir) / "credits.json").read_text(encoding="utf-8"))["entries"]
    assert [e["transaction_id"] for e in entries].count(AB) == 1


async def test_a_resent_paid_hash_in_lowercase_is_not_a_refund(workflow: Workflow, out: Recorder) -> None:
    c, ask = await _paid_guest(workflow, out)
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert len(_answers(out)) == 1
    await c.on_commit(GUEST, out, tx_id=AB.lower(), amount=str(ask.amount), currency="FET",
                      reference=ask.reference, buyer_wallet=None)
    assert AB not in _claims(workflow) and out.cancelled == []


async def test_a_redelivered_commit_after_restart_keeps_the_credit(workflow: Workflow, out: Recorder,
                                                                   mock_state: MockState) -> None:
    c, ask = await _failed_paid_run(workflow, out, mock_state)
    restarted = Conversation(workflow.settings, workflow, address="agent1qself", wallet_address=WALLET,
                             verifier=FakeVerifier())
    await restarted.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET",
                              reference=ask.reference, buyer_wallet=None)
    assert AB not in _claims(workflow) and out.cancelled == []
    assert [cr.tx_id for cr in restarted.credits.for_sender(GUEST)] == [AB]


async def test_a_ledger_failure_keeps_a_durable_record(workflow: Workflow, out: Recorder, monkeypatch) -> None:
    c, ask = await _paid_guest(workflow, out)

    def broken(**_kw):
        raise RuntimeError("ledger unreadable")

    monkeypatch.setattr(c.ledger, "record", broken)
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert len(_answers(out)) == 1 and "mac-a" in out.last.text  # the split is still shown
    fallback = (Path(workflow.settings.data_dir) / "ledger_fallback.jsonl").read_text(encoding="utf-8")
    assert AB in fallback and "mac-a" in fallback


async def test_a_plan_made_around_my_held_model_is_not_kept_warm(workflow: Workflow, mock_state: MockState) -> None:
    async def noop(_t: str) -> None:
        return None

    await workflow.execute(await workflow.propose(Intent(action="run", prompt="x", model_hint="gpt-oss")), "q", 32,
                           noop)
    p = await workflow.propose(Intent(action="run", prompt="x"), may_unload=False)  # planned around gpt-oss
    r = await workflow.execute(p, "q", 32, noop)
    assert workflow._picks.get(r.deployment_id) is None


def test_old_credit_files_are_migrated(tmp_path) -> None:
    old = {GUEST: {"tx_id": AB, "amount": "0.020653", "prompt": "hi", "max_tokens": 256, "model_hint": None,
                   "preference": "biggest", "scope": "pool"}}
    (tmp_path / "paid_credits.json").write_text(json.dumps(old), encoding="utf-8")
    credits = Credits(tmp_path / "paid_credits.json")
    assert [c.tx_id for c in credits.for_sender(GUEST)] == [AB]
    assert AB in json.loads((tmp_path / "paid_credits.json").read_text(encoding="utf-8"))


def test_a_failed_spend_write_keeps_the_credit_everywhere(tmp_path, monkeypatch) -> None:
    import hc_agent.store as store

    credits = Credits(tmp_path / "paid_credits.json")
    credits.add(Credit(tx_id=AB, sender=GUEST, amount="1", prompt="hi", max_tokens=16))

    def full(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(store, "write_json", full)
    with pytest.raises(OSError):
        credits.spend(AB)
    assert [c.tx_id for c in credits.for_sender(GUEST)] == [AB]  # memory agrees with the file


async def test_an_undelivered_answer_is_resent_not_rerun(workflow: Workflow, out: Recorder,
                                                           mock_state: MockState) -> None:
    c, ask = await _paid_guest(workflow, out)
    out.undeliverable_prefix = "[mock]"  # the recipient controls this signal, so it must not decide about money
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert c.credits.for_sender(GUEST) == []  # spent: the answer was produced
    runs = mock_state.chats
    out.undeliverable_prefix = None
    await c.on_text(GUEST, "run it", out)
    assert out.last.text.startswith("[mock]") and mock_state.chats == runs  # the same answer, no second run
    await c.on_text(GUEST, "run it", out)
    assert not out.last.text.startswith("[mock]")  # resent once, not a source of free answers


async def test_retry_runs_the_request_that_failed(workflow: Workflow, out: Recorder, mock_state: MockState) -> None:
    c, ask = await _failed_paid_run(workflow, out, mock_state)  # credit AB for the prompt "hi"
    await c.on_text(GUEST, "run 'a different question' on the biggest model", out)
    assert "covered by your earlier payment" in out.last.text
    mock_state.fail_loads = 2
    await c.on_text(GUEST, "yes", out)
    assert "say **run it**" in out.last.text
    mock_state.fail_loads = 0
    assert c.credits.for_sender(GUEST)[0].prompt == "a different question"  # what "run it" will now retry


async def test_a_failed_credit_save_is_not_a_refund_when_the_run_succeeds(workflow: Workflow, out: Recorder,
                                                                         monkeypatch) -> None:
    c, ask = await _paid_guest(workflow, out)

    def full(_credit):
        raise OSError("disk full")

    monkeypatch.setattr(c.credits, "add", full)
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert len(_answers(out)) == 1 and AB not in _claims(workflow)


async def test_a_failed_credit_save_is_a_refund_when_the_run_fails(workflow: Workflow, out: Recorder,
                                                                  mock_state: MockState, monkeypatch) -> None:
    c, ask = await _paid_guest(workflow, out)

    def full(_credit):
        raise OSError("disk full")

    monkeypatch.setattr(c.credits, "add", full)
    mock_state.fail_loads = 2
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert '"verified_on_chain": true' in _claims(workflow) and AB in _claims(workflow)


async def test_a_verifier_crash_keeps_the_request_open(workflow: Workflow, out: Recorder) -> None:
    class Flaky(FakeVerifier):
        def __init__(self) -> None:
            super().__init__()
            self.first = True

        async def verify(self, tx_id, **kw):
            if self.first:
                self.first = False
                raise RuntimeError("grpc channel closed")
            return await super().verify(tx_id, **kw)

    c, ask = await _paid_guest(workflow, out, Flaky())
    kw = dict(tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference, buyer_wallet=None)
    await c.on_commit(GUEST, out, **kw)
    assert c._pending[GUEST].awaiting_payment and out.cancelled == []
    await c.on_commit(GUEST, out, **kw)  # the resend now verifies and buys the answer
    assert len(_answers(out)) == 1 and out.completed == [AB]
