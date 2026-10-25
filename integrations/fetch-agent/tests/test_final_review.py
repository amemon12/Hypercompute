"""Regression tests for the final adversarial review: a spent payment stays spent even when the spend write fails,
payment intake survives unreadable side files, paid hashes void earlier refund claims, a retry stays on the model
that was paid for, and a local rehearsal touches no Agentverse record."""

import json
from pathlib import Path

from hc_agent.conversation import Conversation
from hc_agent.workflow import Workflow
from mock_gateway.server import MockState
from tests.conftest import GUEST, Recorder, make_settings
from tests.test_conversation import WALLET, FakeVerifier, _failed_paid_run, _paid_guest, convo

AB = "AB" * 32


def _answers(out: Recorder) -> list[str]:
    return [t for t in out.texts if t.startswith("[mock]")]


def _claim_rows(workflow: Workflow) -> list[dict]:
    path = Path(workflow.settings.data_dir) / "payment_claims.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def test_a_failed_spend_write_does_not_buy_more_answers(workflow: Workflow, out: Recorder,
                                                              mock_state: MockState, monkeypatch) -> None:
    c, ask = await _failed_paid_run(workflow, out, mock_state)  # credit AB saved
    plan = mock_state.plan("kimi-k2.6-iq1_s", "all_selected", None, 4096)
    mock_state.create(plan, "owner", foreign=True, ready=True)  # the paid model is warm: no confirmation

    def locked(_tx):
        raise OSError("file in use")

    monkeypatch.setattr(c.credits, "spend", locked)
    for _ in range(3):
        await c.on_text(GUEST, "run it", out)
    assert len(_answers(out)) == 1 and len(out.payments) == 1
    entries = json.loads((Path(workflow.settings.data_dir) / "credits.json").read_text(encoding="utf-8"))["entries"]
    assert [e["transaction_id"] for e in entries].count(AB) == 1
    restarted = Conversation(workflow.settings, workflow, address="agent1qself", wallet_address=WALLET,
                             verifier=FakeVerifier())
    assert restarted._sender_credits(GUEST) == []  # the ledger already holds AB, so it is spent after a restart


async def test_an_unreadable_ledger_does_not_block_payments(workflow: Workflow, out: Recorder) -> None:
    c, ask = await _paid_guest(workflow, out)
    data = Path(workflow.settings.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    (data / "credits.json").write_text("{hand-edited, not json", encoding="utf-8")
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert len(_answers(out)) == 1 and out.completed == [AB]
    assert AB in (data / "ledger_fallback.jsonl").read_text(encoding="utf-8")  # the split is still kept


async def test_a_hash_that_pays_voids_earlier_claims(workflow: Workflow, out: Recorder) -> None:
    c, ask = await _paid_guest(workflow, out)
    thief = "agent1q" + "t" * 58
    await c.on_commit(thief, Recorder(), tx_id=AB, amount=str(ask.amount), currency="FET", reference=None,
                      buyer_wallet=None)  # front-runs the buyer's public transfer
    assert any(r.get("tx_id") == AB and not r.get("void") for r in _claim_rows(workflow))
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert len(_answers(out)) == 1
    assert any(r.get("tx_id") == AB and r.get("void") for r in _claim_rows(workflow))


async def test_retry_stays_on_the_model_that_was_paid_for(workflow: Workflow, out: Recorder,
                                                          mock_state: MockState) -> None:
    plan = mock_state.plan("gpt-oss-120b-mxfp4", "all_selected", None, 4096)
    foreign = mock_state.create(plan, "owner", foreign=True, ready=True)
    c = convo(workflow, guest_access="paid")
    c.gate.guest_access = "paid"
    await c.on_text(GUEST, "run 'hi' on the biggest model", out)  # quoted for the loaded gpt-oss
    assert c._pending[GUEST].proposal.model.id == "gpt-oss-120b-mxfp4"
    await c.on_text(GUEST, "yes", out)
    ask = out.payments[-1]
    mock_state.stop(foreign["id"])  # unloaded before the run, and the reload fails
    mock_state.fail_loads = 2
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    assert "say **run it**" in out.last.text
    mock_state.fail_loads = 0
    await c.on_text(GUEST, "run it", out)
    assert "covered by your earlier payment" in out.last.text and "gpt-oss" in out.last.text
    await c.on_text(GUEST, "yes", out)
    assert _answers(out)[-1].startswith("[mock] gpt-oss") and len(out.payments) == 1


async def test_a_covered_plan_waits_while_its_payment_is_running(workflow: Workflow, out: Recorder,
                                                                 mock_state: MockState) -> None:
    c, ask = await _failed_paid_run(workflow, out, mock_state)
    await c.on_text(GUEST, "run 'first' on the biggest model", out)  # P1 covered by AB, waits for confirmation
    c._in_use.add(AB)  # another of this guest's runs is spending AB right now
    await c.on_text(GUEST, "yes", out)
    assert "paying for another of your requests" in out.last.text and c._pending[GUEST].covered_tx == AB


async def test_a_reply_that_raises_after_the_spend_is_kept_for_resend(workflow: Workflow, out: Recorder,
                                                                      monkeypatch) -> None:
    c, ask = await _paid_guest(workflow, out)
    real = out.reply

    async def boom(to, reply):
        if reply.text.startswith("[mock]"):
            raise RuntimeError("card validation failed")
        return await real(to, reply)

    monkeypatch.setattr(out, "reply", boom)
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    monkeypatch.setattr(out, "reply", real)
    await c.on_text(GUEST, "run it", out)
    assert out.last.text.startswith("[mock]")  # resent, not lost


async def test_an_unsaved_credit_is_listed_for_refund_once(workflow: Workflow, out: Recorder, mock_state: MockState,
                                                          monkeypatch) -> None:
    c, ask = await _paid_guest(workflow, out)

    def full(_credit):
        raise OSError("disk full")

    monkeypatch.setattr(c.credits, "add", full)
    mock_state.fail_loads = 6
    await c.on_commit(GUEST, out, tx_id=AB, amount=str(ask.amount), currency="FET", reference=ask.reference,
                      buyer_wallet=None)
    await c.on_text(GUEST, "run it", out)
    await c.on_text(GUEST, "yes", out)
    rows = [r for r in _claim_rows(workflow) if r.get("tx_id") == AB and r.get("verified_on_chain")]
    assert len(rows) == 1


async def test_local_rehearsal_never_touches_the_almanac_status(tmp_path, monkeypatch) -> None:
    from hc_agent.agent import build

    calls = []

    async def spy(*_a, **_k):
        calls.append(True)

    monkeypatch.setattr("uagents.agent.update_agent_status", spy)
    agent, _ = build(make_settings(tmp_path, HC_MAILBOX="false", HC_AGENT_PORT="8796"))
    await agent._update_agent_status(active=True)
    await agent._update_agent_status(active=False)
    assert calls == []
