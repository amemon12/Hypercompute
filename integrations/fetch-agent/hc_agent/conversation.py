"""One conversation engine for every transport: ASI:One chat, card clicks, payments and agent-to-agent requests.

It holds per-sender state (a plan waiting for approval or payment), enforces the gatekeeper, runs the workflow and
writes replies through the Outbox of the message being handled (so replies stay in that message's session).
agent.py adapts it to uAgents; tests drive it directly.

Money rules: a quote is registered before it is shown, carries a unique tag, and stays open until it is paid,
rejected, cancelled or its deadline passes; an open quote is never silently replaced. A verified payment whose run
fails becomes a persisted credit that covers the sender's next run on a fresh plan. A commit nobody asked for, or
one that arrives after its request closed, is logged to agent_data/payment_claims.jsonl: the sender's unverified
claim (a valid hash, once, size-capped) for the owner to check on chain before refunding anything.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from .config import Settings
from .intent import Intent, parse_intent
from .payments import TX_RE, PaymentAsk, Verifier
from .pricing import CreditLedger, price_for, split
from .replies import (
    Reply,
    busy_reply,
    error_reply,
    help_reply,
    not_allowed_reply,
    progress_reply,
    prompt_form,
    proposal_reply,
    rate_limited_reply,
    result_reply,
    status_reply,
)
from .safety import Gatekeeper
from .store import Credit, Credits, StoreError, append_jsonl
from .workflow import Proposal, RunOutcome, Workflow, WorkflowError

log = logging.getLogger(__name__)
PENDING_TTL_S = 900.0
PAYMENT_DEADLINE_S = 300
PAYMENT_GRACE_S = 120.0  # a commit may arrive a little after the deadline the buyer saw
QUOTE_TAG = Decimal("0.000001")
MAX_OPEN_QUOTES = 400
MAX_UNDELIVERED = 200  # paid answers kept for one resend
MAX_CLAIMS = 2000  # unverified payment claims recorded per agent run
MAX_CLAIMS_BYTES = 2_000_000
CLAIMS_PER_SENDER = 20
UNSOLICITED_CLAIMS_PER_SENDER = 3


def claimed_amount(raw: object) -> str:
    """The amount a sender claims, bounded: at most 64 input characters and a sane exponent, so a crafted value
    such as 1e999999 cannot expand into a huge string."""
    text = str(raw)[:64]
    try:
        value = Decimal(text)
    except (ArithmeticError, ValueError, TypeError):
        return "unparseable"
    if not value.is_finite() or abs(value.adjusted()) > 30:
        return "unparseable"
    return f"{value.normalize():f}"[:64]
REFUNDABLE = frozenset({"busy", "no_match", "nothing_fits", "gateway_down", "gateway_auth", "no_models", "not_ready"})
GUEST_SAFE = frozenset({"busy", "no_match", "nothing_fits", "no_models", "load_timeout", "chat_timeout", "not_ready",
                        "gateway_down", "stopped_by_owner"})


def unique_quote(base: Decimal, taken: Iterable[Decimal]) -> Decimal:
    """base plus a random tag of 1 to 999 micro-FET that no open quote uses, so an on-chain transfer can match only
    one request (see payments.py)."""
    used = set(taken)
    for _ in range(64):
        quote = base + QUOTE_TAG * (secrets.randbelow(999) + 1)
        if quote not in used:
            return quote
    raise WorkflowError("busy", "Too many payment requests are open right now.", "try again in a few minutes")


def guest_safe(exc: WorkflowError) -> WorkflowError:
    """Engine and gateway errors can name friends' machines, ports and paths; guests get a plain version."""
    if exc.code in GUEST_SAFE:
        return exc
    return WorkflowError(exc.code, "The pool hit an internal error while running your request.",
                         "try again in a few minutes")


class Outbox(Protocol):
    async def reply(self, to: str, reply: Reply) -> bool: ...  # False when it was not delivered
    async def request_payment(self, to: str, ask: PaymentAsk) -> bool: ...  # False when it was not delivered
    async def complete_payment(self, to: str, tx_id: str) -> None: ...
    async def cancel_payment(self, to: str, tx_id: str | None, reason: str) -> None: ...


@dataclass(frozen=True)
class Pending:
    ref: str
    intent: Intent
    prompt: str
    max_tokens: int
    proposal: Proposal
    price: Decimal | None  # what this run costs (for the ledger); None when free
    created: float
    covered_tx: str | None = None  # paid by an earlier payment (a credit): no payment request
    awaiting_payment: bool = False
    verifying: bool = False
    verifying_tx: str | None = None
    quoted_at: datetime | None = None  # when the payment request went out; earlier transactions never count
    quote_expires: float | None = None  # monotonic deadline for the open payment request


class Conversation:
    def __init__(self, settings: Settings, workflow: Workflow, *, address: str,
                 wallet_address: str | None = None, verifier: Verifier | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.settings, self.workflow = settings, workflow
        self.address, self.wallet_address, self.verifier = address, wallet_address, verifier
        billing = bool(wallet_address and verifier)
        self.gate = Gatekeeper(allowed=settings.allowed_senders,
                               guest_access=settings.guest_access if billing or settings.guest_access != "paid"
                               else "read",  # paid access without a wallet would take requests it cannot bill
                               payments=settings.payments if billing else "off",
                               runs_per_window=settings.runs_per_window, window_s=settings.rate_window_s,
                               messages_per_minute=settings.messages_per_minute, clock=clock)
        data = Path(settings.data_dir)
        self.ledger = CreditLedger(data / "credits.json")
        self.credits = Credits(data / "paid_credits.json")
        self._claims_path = data / "payment_claims.jsonl"
        self._claims_seen: set[tuple[str, str]] = set()  # (hash, sender) pairs already recorded
        self._claims_by_sender: dict[str, int] = {}
        self._pending: dict[str, Pending] = {}
        self._completed: set[str] = set()  # transaction ids already completed (duplicate commits are ignored)
        self._in_use: set[str] = set()  # saved payments paying for a run in progress (never spent twice)
        self._unsaved: dict[str, Credit] = {}  # verified payments whose credit could not be written to disk
        self._fallback_path = data / "ledger_fallback.jsonl"
        self._undelivered: dict[str, Reply] = {}  # a paid answer whose send failed, resent once on "run it"
        self._spent: set[str] = set()  # payments that bought their answer, even if marking them spent failed
        self._refund_noted: set[str] = set()  # unsaved credits already put on the refund list
        self._clock = clock

    # ------------------------------------------------------------------ entry points
    async def on_text(self, sender: str, text: str, out: Outbox) -> None:
        gate = self.gate.message(sender)
        if not gate.allowed:
            await out.reply(sender, rate_limited_reply(gate.retry_after_s or 60, "messages"))
            return
        self._sweep()
        try:
            # asi1 runs on the owner's key, so only members' free text goes to it
            key = self.settings.asi1_api_key if gate.role == "member" else None
            intent = await parse_intent(text[: self.settings.max_prompt_chars * 2], asi1_key=key,
                                        asi1_model=self.settings.asi1_model, asi1_url=self.settings.asi1_url)
            await self._dispatch(sender, intent, out)
        except WorkflowError as exc:
            await out.reply(sender, error_reply(exc if gate.role == "member" else guest_safe(exc)))
        except Exception:  # never let one bad request kill the agent, and never leak internals into chat
            log.exception("unhandled error for %s", sender[:16])
            await out.reply(sender, Reply("Something went wrong on my side. The owner can check the agent "
                                          "log; please try again."))

    async def on_selection(self, sender: str, selection: dict, out: Outbox) -> None:
        """A card response (MetadataContent) or a JSON TextContent selection."""
        await self.on_text(sender, json.dumps(selection), out)

    def _sweep(self) -> None:
        """Drop expired plans and close payment requests whose deadline passed."""
        now = self._clock()
        kept = {}
        for sender, p in self._pending.items():
            if p.awaiting_payment and not p.verifying and p.quote_expires is not None and now > p.quote_expires:
                p = replace(p, awaiting_payment=False, quote_expires=None)
            if p.awaiting_payment or p.verifying or now - p.created <= PENDING_TTL_S:
                kept[sender] = p
        self._pending = kept

    # ------------------------------------------------------------------ dispatch
    async def _dispatch(self, sender: str, intent: Intent, out: Outbox) -> None:
        decision = self.gate.action(sender, intent.action)
        member = decision.role == "member"
        if intent.action == "help":
            await out.reply(sender, help_reply(member=member, address=sender,
                                               guest_access=self.gate.guest_access))
            return
        if not decision.allowed:
            await out.reply(sender, not_allowed_reply(sender, decision.reason or "not allowed"))
            return
        if intent.action == "status":
            await out.reply(sender, status_reply(await self.workflow.pool()))
        elif intent.action == "edit":
            await out.reply(sender, prompt_form(await self.workflow.pool()))
        elif intent.action == "cancel":
            await self._cancel(sender, out)
        elif intent.action == "stop":
            await self._stop(sender, out)
        elif intent.action == "approve":
            await self._approve(sender, intent, out, needs_payment=decision.needs_payment)
        elif intent.action == "run":
            await self._propose(sender, intent, out, needs_payment=decision.needs_payment)

    async def _cancel(self, sender: str, out: Outbox) -> None:
        pending = self._pending.get(sender)
        if pending is not None and pending.verifying:
            await out.reply(sender, Reply("I'm checking your payment on the ledger right now, so it can't be "
                                          "cancelled. I'll tell you as soon as it is verified."))
            return
        self._pending.pop(sender, None)
        if pending is not None and pending.awaiting_payment:
            await out.cancel_payment(sender, None, "you cancelled this request")
        await out.reply(sender, Reply("Cancelled. Nothing was loaded." if pending else
                                      "There was nothing waiting to cancel."))

    async def _stop(self, sender: str, out: Outbox) -> None:
        if self.workflow.run_lock.locked():
            await out.reply(sender, busy_reply())
            return
        async with self.workflow.run_lock:  # an unload never races a run
            stopped, failed = await self.workflow.stop_owned()
        if failed:
            text = (f"I asked the host to unload {', '.join(failed)} but it did not confirm. I'll keep trying; the "
                    f"owner can stop it from the dashboard.")
        elif stopped:
            text = f"Unloaded {', '.join(stopped)}. Friends' machines are free."
        else:
            text = "I have nothing loaded. I never unload models someone else started."
        await out.reply(sender, Reply(text))

    def _tokens(self, intent: Intent) -> int:
        asked = intent.max_tokens or self.settings.default_max_tokens
        return max(16, min(asked, self.settings.max_tokens_cap))

    def _open_quotes(self, except_sender: str) -> list[Decimal]:
        return [p.price for s, p in self._pending.items()
                if s != except_sender and p.price is not None and p.covered_tx is None]

    def _payment_open(self, sender: str) -> Pending | None:
        current = self._pending.get(sender)
        return current if current is not None and (current.awaiting_payment or current.verifying) else None

    async def _say_payment_open(self, sender: str, current: Pending, out: Outbox) -> None:
        await out.reply(sender, Reply(f"You have an open payment request for {current.price.normalize():f} "
                                      f"{self.settings.pricing.currency}. Pay it, reject it, or say **cancel** "
                                      f"before asking for something else."))

    def _credit_for(self, sender: str, price: Decimal) -> Credit | None:
        """An unspent payment of this sender that covers price and is not already paying for a run in progress."""
        fits = [c for c in self._sender_credits(sender)
                if c.tx_id not in self._in_use and Decimal(c.amount) >= price]
        return min(fits, key=lambda c: Decimal(c.amount)) if fits else None

    async def _propose(self, sender: str, intent: Intent, out: Outbox, *, needs_payment: bool) -> None:
        if not intent.prompt:
            await out.reply(sender, prompt_form(await self.workflow.pool()))
            return
        if len(intent.prompt) > self.settings.max_prompt_chars:
            await out.reply(sender, Reply(f"That prompt is {len(intent.prompt)} characters; the limit is "
                                          f"{self.settings.max_prompt_chars}. Please shorten it."))
            return
        if (current := self._payment_open(sender)) is not None:
            await self._say_payment_open(sender, current, out)
            return
        if self.workflow.run_lock.locked():
            await out.reply(sender, busy_reply())
            return
        wait = self.gate.propose(sender)
        if wait is not None:
            await out.reply(sender, rate_limited_reply(wait, "plans"))
            return
        if needs_payment and not self.credits.for_sender(sender) and \
                len(self._open_quotes(sender)) >= MAX_OPEN_QUOTES:
            raise WorkflowError("busy", "Too many payment requests are open right now.", "try again in a few minutes")
        await out.reply(sender, progress_reply("Checking which friends' machines are online and what fits..."))
        proposal = await self.workflow.propose(intent, may_unload=not needs_payment)
        if (current := self._payment_open(sender)) is not None:  # opened by another message while this planned
            await self._say_payment_open(sender, current, out)
            return
        max_tokens = self._tokens(intent)
        price, covered = None, None
        if needs_payment:
            base = price_for(proposal.model.params_b, max_tokens, self.settings.pricing)
            credit = self._credit_for(sender, base)  # an earlier payment covers a run that costs no more
            if credit is not None:
                price, covered = Decimal(credit.amount), credit.tx_id
            elif len(self._open_quotes(sender)) >= MAX_OPEN_QUOTES:
                raise WorkflowError("busy", "Too many payment requests are open right now.",
                                    "try again in a few minutes")
            else:
                price = unique_quote(base, self._open_quotes(sender))
        confirm = self.settings.confirm
        charge = price is not None and covered is None
        needs_confirmation = charge or confirm == "always" or (confirm == "auto" and not proposal.reuse)
        pending = Pending(ref=uuid.uuid4().hex[:12], intent=intent, prompt=intent.prompt, max_tokens=max_tokens,
                          proposal=proposal, price=price, created=self._clock(), covered_tx=covered)
        if needs_confirmation:
            self._pending[sender] = pending  # registered before it is shown, so a fast click always finds it
        await out.reply(sender, proposal_reply(proposal, prompt=intent.prompt, max_tokens=max_tokens, ref=pending.ref,
                                               price=price if charge else None,
                                               currency=self.settings.pricing.currency,
                                               needs_confirmation=needs_confirmation, covered=covered is not None))
        if not needs_confirmation:
            await self._run(sender, pending, out, paid_tx=covered)

    async def _approve(self, sender: str, intent: Intent, out: Outbox, *, needs_payment: bool) -> None:
        pending = self._pending.get(sender)
        if pending is None and sender in self._undelivered:  # a paid answer that did not get through: resend it
            kept = self._undelivered[sender]
            self._undelivered = {k: v for k, v in self._undelivered.items() if k != sender}
            await out.reply(sender, kept)
            return
        if pending is None:
            credit = next((c for c in reversed(self._sender_credits(sender)) if c.tx_id not in self._in_use), None)
            if credit is not None:  # a paid run did not finish: plan it again, covered by that payment
                retry = Intent(action="run", prompt=credit.prompt, max_tokens=credit.max_tokens,
                               model_hint=credit.model_hint, preference=credit.preference, scope=credit.scope)
                await self._propose(sender, retry, out, needs_payment=True)
                return
            await out.reply(sender, prompt_form(None, note="I have no plan waiting for approval."))
            return
        if intent.ref and intent.ref != pending.ref:
            await out.reply(sender, Reply("That card is from an older plan. Use the latest one, or ask again."))
            return
        if pending.verifying:
            await out.reply(sender, Reply("I'm checking your payment on the ledger now."))
            return
        if pending.awaiting_payment:
            await out.reply(sender, Reply("The payment request is open; complete it in the payment card and I'll "
                                          "start right away."))
            return
        if self._clock() - pending.created > PENDING_TTL_S:
            self._pending.pop(sender, None)
            await self._propose(sender, pending.intent, out, needs_payment=needs_payment)
            return
        if pending.price is not None and pending.covered_tx is None:
            await self._request_payment(sender, pending, out)
            return
        if pending.covered_tx is not None and pending.covered_tx in self._in_use:
            await out.reply(sender, Reply("That payment is paying for another of your requests right now. If that "
                                          "one fails, say **run it** to retry at no extra cost."))
            return
        self._pending.pop(sender, None)
        await self._run(sender, pending, out, paid_tx=pending.covered_tx)

    async def _request_payment(self, sender: str, pending: Pending, out: Outbox) -> None:
        assert self.wallet_address is not None and pending.price is not None
        if pending.quoted_at is not None:  # an earlier request for this plan lapsed: a new quote, a new tag
            base = price_for(pending.proposal.model.params_b, pending.max_tokens, self.settings.pricing)
            pending = replace(pending, price=unique_quote(base, self._open_quotes(sender)))
        ask = PaymentAsk(amount=pending.price, currency=self.settings.pricing.currency,
                         recipient=self.wallet_address, reference=pending.ref, network=self.settings.fet_network,
                         deadline_s=PAYMENT_DEADLINE_S,
                         description=f"Hypercompute: run your prompt on {pending.proposal.model.name} across "
                                     f"{len(pending.proposal.view.machines)} machines")
        self._pending[sender] = replace(pending, awaiting_payment=True, quoted_at=datetime.now(UTC),
                                        quote_expires=self._clock() + PAYMENT_DEADLINE_S + PAYMENT_GRACE_S)
        if not await out.request_payment(sender, ask):
            current = self._pending.get(sender)
            if current is not None and current.ref == pending.ref:
                self._pending[sender] = replace(current, awaiting_payment=False, quote_expires=None)
            await out.reply(sender, Reply("I couldn't deliver the payment request. Nothing was charged; reply "
                                          "**run it** to try again."))
            return
        await out.reply(sender, Reply(f"Sent a payment request for exactly {ask.amount.normalize():f} "
                                      f"{ask.currency} on {ask.network} (the last digits tag this request). I start "
                                      f"as soon as it confirms on chain."))

    # ------------------------------------------------------------------ payments
    def record_unmatched(self, sender: str, tx_id: str, amount: str, why: str, *, verified: bool = False,
                         solicited: bool = True) -> None:
        """A payment that may have moved without buying anything. Any agent can send a CommitPayment, so an
        unverified entry is only the sender's claim: a well-formed hash, once per (hash, sender), within a quota per
        sender (smaller for commits nobody asked for) and a global cap, for the owner to check on chain before
        refunding. A payment the ledger already verified (verified=True) is always written."""
        key = (tx_id or "").upper()
        if not TX_RE.match(key):
            log.warning("payment claim from %s with a malformed hash not recorded (%s)", sender[:20], why[:80])
            return
        if not verified and self._already_paid(key):  # it bought an answer or is a live credit: never a refund
            log.info("claim for %s skipped: already paid for an answer or saved as a credit", key[:16])
            return
        if not verified:
            mine = self._claims_by_sender.get(sender, 0)
            quota = CLAIMS_PER_SENDER if solicited else UNSOLICITED_CLAIMS_PER_SENDER
            if (key, sender) in self._claims_seen or mine >= quota or len(self._claims_seen) >= MAX_CLAIMS:
                log.warning("payment claim %s from %s not recorded (quota)", key, sender[:80])
                return
            self._claims_seen.add((key, sender))
            self._claims_by_sender = {**self._claims_by_sender, sender: mine + 1}
        record = {"sender": sender[:80], "tx_id": key, "claimed_amount": claimed_amount(amount),
                  "why": why[:200], "verified_on_chain": verified, "solicited": solicited}
        log.warning("payment claim %s from %s: %s", key[:16], sender[:20], why[:80])
        try:
            if not verified and self._claims_path.is_file() and \
                    self._claims_path.stat().st_size + 1024 > MAX_CLAIMS_BYTES:
                log.error("%s is full; claim %s not written", self._claims_path, key)
                return
            append_jsonl(self._claims_path, record)
        except OSError:
            log.exception("could not record payment claim: %s", record)

    async def on_commit(self, sender: str, out: Outbox, *, tx_id: str, amount: str, currency: str,
                        reference: str | None, buyer_wallet: str | None) -> None:
        try:
            await self._commit(sender, out, tx_id=tx_id, amount=amount, currency=currency, reference=reference,
                               buyer_wallet=buyer_wallet)
        except Exception:
            log.exception("payment handling failed for %s", str(tx_id)[:16])
            await out.reply(sender, Reply("Something went wrong while handling your payment. The owner can check "
                                          "the agent log; nothing ran on it."))

    def _close_quote(self, sender: str, ref: str) -> None:
        current = self._pending.get(sender)
        if current is not None and current.ref == ref:
            self._pending[sender] = replace(current, awaiting_payment=False, quote_expires=None, verifying=False,
                                            verifying_tx=None)

    def _stop_verifying(self, sender: str, ref: str) -> None:
        """Back to waiting for the payment: a resend of the same payment can still buy the answer."""
        current = self._pending.get(sender)
        if current is not None and current.ref == ref:
            self._pending[sender] = replace(current, verifying=False, verifying_tx=None)

    async def _commit(self, sender: str, out: Outbox, *, tx_id: str, amount: str, currency: str,
                      reference: str | None, buyer_wallet: str | None) -> None:
        key = str(tx_id or "").strip().upper()  # one transaction, whatever letter case it arrives in
        if self._already_paid(key):
            log.info("commit for %s already paid for an answer or is a saved credit; ignored", key[:16])
            if self._is_live(key):
                await out.reply(sender, Reply("I already have that payment; it covers your next run. Say **run it** "
                                              "to use it."))
            return
        pending = self._pending.get(sender)
        if pending is not None and pending.verifying:
            if pending.verifying_tx != key:  # the buyer paid twice: the second transfer buys nothing
                self.record_unmatched(sender, key, amount, "second payment while another was being verified")
                await out.reply(sender, Reply("I'm already verifying an earlier payment for this request. This one "
                                              "is recorded for the owner to check and refund."))
            return
        if pending is None or pending.price is None or not pending.awaiting_payment:
            self.record_unmatched(sender, key, amount, "no open payment request", solicited=False)
            await out.cancel_payment(sender, tx_id, "no payment was requested for this conversation; the owner has a "
                                                    "record of this transaction")
            return
        if reference and reference != pending.ref:
            self.record_unmatched(sender, key, amount, "reference of a different request", solicited=False)
            await out.cancel_payment(sender, tx_id, "that payment is for a different request")
            return
        try:
            offered = Decimal(str(amount)[:64])
            matches = offered.is_finite() and offered == pending.price
        except (ArithmeticError, ValueError):
            matches = False
        if currency != self.settings.pricing.currency or not matches:
            self.record_unmatched(sender, key, amount, f"amount or currency mismatch ({str(currency)[:16]})")
            await out.cancel_payment(sender, tx_id, f"expected exactly {pending.price.normalize():f} "
                                                    f"{self.settings.pricing.currency}")
            return
        assert self.verifier is not None
        self._pending[sender] = replace(pending, verifying=True, verifying_tx=key)  # before the first await
        try:
            verdict = await self.verifier.verify(key, amount=pending.price, buyer_wallet=buyer_wallet,
                                                 not_before=pending.quoted_at)
        except Exception:
            log.exception("payment check for %s failed", key[:16])
            self._stop_verifying(sender, pending.ref)
            await out.reply(sender, Reply("I couldn't reach the ledger to check that payment. The request stays "
                                          "open; send the payment confirmation again."))
            return
        except BaseException:
            self._stop_verifying(sender, pending.ref)
            raise
        if not verdict.ok:
            if verdict.retryable:  # not on the ledger yet: keep the request open so a later commit still counts
                self._stop_verifying(sender, pending.ref)
                await out.reply(sender, Reply(f"I can't see that payment on the ledger yet ({verdict.reason}). The "
                                              f"request stays open; if it confirms, send the payment confirmation "
                                              f"again."))
                return
            self.record_unmatched(sender, key, amount, f"not verified: {verdict.reason}")
            self._close_quote(sender, pending.ref)
            await out.cancel_payment(sender, tx_id, verdict.reason)
            await out.reply(sender, Reply(f"I couldn't verify that payment: {verdict.reason}. Nothing ran. Reply "
                                          f"**run it** for a new payment request."))
            return
        # write-ahead: the verified payment is saved before anything else can fail, and spent only once its
        # answer has been delivered (a crash, restart or failed load leaves it as a credit for the next run)
        self._completed.add(key)
        self._in_use.add(key)
        self._void_claims(key)
        i = pending.intent
        credit = Credit(tx_id=key, sender=sender, amount=str(pending.price), prompt=pending.prompt,
                        max_tokens=pending.max_tokens, model_hint=i.model_hint, preference=i.preference,
                        scope=i.scope)
        try:
            self.credits.add(credit)
        except (OSError, StoreError):  # kept in memory; if the run then fails it goes on the refund list
            log.exception("could not save the credit for verified payment %s", key)
            self._unsaved = {**self._unsaved, key: credit}
        current = self._pending.get(sender)
        if current is not None and current.ref == pending.ref:
            self._pending.pop(sender, None)
        try:
            await out.complete_payment(sender, tx_id)
            await self._run(sender, replace(pending, awaiting_payment=False, verifying=False, covered_tx=key), out,
                            paid_tx=key)
        finally:
            self._in_use.discard(key)

    async def on_reject(self, sender: str, reason: str | None, out: Outbox) -> None:
        pending = self._pending.get(sender)
        if pending is not None and pending.verifying:
            return  # a payment for this request is already being verified
        if pending is not None:
            self._close_quote(sender, pending.ref)
        await out.reply(sender, Reply("No problem, nothing ran. Reply **run it** if you change your mind."))

    # ------------------------------------------------------------------ saved payments
    def _sender_credits(self, sender: str) -> list[Credit]:
        mine = [*self.credits.for_sender(sender), *(c for c in self._unsaved.values() if c.sender == sender)]
        return [c for c in mine if not self._is_spent(c.tx_id)]

    def _is_spent(self, key: str) -> bool:
        """Bought its answer: marked here, or its split is in the ledger (after a restart whose spend write failed)."""
        return key in self._spent or self._in_ledger(key) or self._in_fallback(key)

    def _is_live(self, key: str) -> bool:
        """A verified payment that has not bought its answer yet (on disk, or in memory if the save failed)."""
        return (self.credits.has(key) or key in self._unsaved) and not self._is_spent(key)

    def _already_paid(self, key: str) -> bool:
        """The transaction already bought an answer or is a saved credit: never a refund, never a second run."""
        if key in self._completed or self._is_live(key):
            return True
        used = getattr(self.verifier, "is_used", None)
        return bool(used and used(key)) or self._is_spent(key)

    def _void_claims(self, key: str) -> None:
        """This hash paid for an answer or a saved credit: any earlier claim row for it (a front-runner's, or a
        second payment recorded while another was verifying) must not be refunded."""
        try:
            if self._claims_path.is_file() and key in self._claims_path.read_text(encoding="utf-8"):
                append_jsonl(self._claims_path, {"tx_id": key, "void": True,
                                                 "why": "this hash paid; do not refund any earlier claim for it"})
        except OSError:
            log.exception("could not void earlier claims for %s", key)

    def _in_ledger(self, key: str) -> bool:
        """Never raises: an unreadable ledger must not stop payment intake. Replays are refused by the verifier's
        own persisted used list either way."""
        try:
            return self.ledger.has_transaction(key)
        except (OSError, StoreError, ValueError):
            log.exception("credit ledger unreadable; relying on the verifier's used list for %s", key[:16])
            return False

    def _in_fallback(self, key: str) -> bool:
        try:
            return self._fallback_path.is_file() and key in self._fallback_path.read_text(encoding="utf-8")
        except OSError:
            log.exception("could not read %s; relying on the verifier's used list", self._fallback_path)
            return False

    def _retarget(self, key: str, pending: Pending) -> None:
        """A covered run failed: point the payment at this request, so "run it" retries what the user just asked."""
        i = pending.intent
        request = dict(prompt=pending.prompt, max_tokens=pending.max_tokens, model_hint=pending.proposal.model.id,
                       preference=i.preference, scope=i.scope)
        if key in self._unsaved:
            self._unsaved = {**self._unsaved, key: replace(self._unsaved[key], **request)}
            return
        try:
            self.credits.retarget(key, **request)
        except (OSError, StoreError):
            log.exception("could not update saved payment %s; run it retries its earlier request", key[:16])

    def _spend(self, key: str) -> None:
        self._spent = self._spent | {key}  # first, so a failed write below can never make it live again
        self._unsaved = {k: v for k, v in self._unsaved.items() if k != key}
        try:
            self.credits.spend(key)
        except (OSError, StoreError):
            log.exception("could not mark payment %s as spent; it still shows as a credit", key)

    def _record_split(self, sender: str, pending: Pending, outcome: RunOutcome, key: str,
                      shares: list) -> None:
        """The friends' shares of a paid run. If the ledger cannot take them, they go to a fallback log so the
        split is never lost; if that fails too, the full entry is in the error log."""
        try:
            self.ledger.record(request_id=outcome.request_id, model_id=outcome.model.id, payer=sender,
                               currency=self.settings.pricing.currency, shares=shares, transaction_id=key)
            return
        except Exception:
            log.exception("the credit ledger refused the split for %s; writing the fallback log", key[:16])
        entry = {"request_id": outcome.request_id, "model_id": outcome.model.id, "payer": sender,
                 "currency": self.settings.pricing.currency, "transaction_id": key, "amount": str(pending.price),
                 "shares": [{"node_id": s.node_id, "name": s.node_name, "layers": s.layers, "amount": str(s.amount),
                             "role": s.role} for s in shares]}
        try:
            append_jsonl(self._fallback_path, entry)
        except OSError:
            log.critical("credit split for %s could not be saved anywhere: %s", key, json.dumps(entry))

    # ------------------------------------------------------------------ run
    async def _run(self, sender: str, pending: Pending, out: Outbox, *, paid_tx: str | None = None) -> None:
        """paid_tx: the saved payment this run spends, only once its answer has been delivered."""
        member = self.gate.role(sender) == "member"
        kept = " Your payment is kept: say **run it** to retry this request at no extra cost." if paid_tx else ""
        if paid_tx and not self._is_live(paid_tx):
            self._drop_covered(paid_tx)
            await out.reply(sender, Reply("That payment already bought an answer, so this plan is no longer "
                                          "covered. Ask again for a new quote."))
            return
        if self.workflow.run_lock.locked():
            if paid_tx:
                self._retarget(paid_tx, pending)
            await out.reply(sender, Reply(busy_reply().text + kept))
            return
        wait = self.gate.start_run(sender) if not paid_tx else None
        if wait is not None:
            await out.reply(sender, rate_limited_reply(wait))
            return

        async def progress(text: str) -> None:
            await out.reply(sender, progress_reply(text))

        if paid_tx:
            self._in_use.add(paid_tx)
        try:
            async with self.workflow.run_lock:
                try:
                    outcome = await self.workflow.execute(pending.proposal, pending.prompt, pending.max_tokens,
                                                          progress)
                except Exception as exc:
                    if not isinstance(exc, WorkflowError):
                        log.exception("run failed for %s", sender[:16])
                        exc = WorkflowError("internal", "The run failed on my side.", "try again in a minute")
                    if exc.code in REFUNDABLE and not paid_tx:
                        self.gate.refund_run(sender)
                    if paid_tx:
                        self._retarget(paid_tx, pending)
                        if paid_tx in self._unsaved:  # not durable: if the agent restarts, the owner refunds it
                            self._persist_or_note(sender, paid_tx, pending)
                    shown = exc if member else guest_safe(exc)
                    await out.reply(sender, error_reply(WorkflowError(shown.code, shown.message + kept, shown.fix)))
                    return
            await self._deliver(sender, pending, outcome, out, paid_tx)
        finally:
            if paid_tx:
                self._in_use.discard(paid_tx)

    def _persist_or_note(self, sender: str, key: str, pending: Pending) -> None:
        try:
            self.credits.add(self._unsaved[key])
            self._unsaved = {k: v for k, v in self._unsaved.items() if k != key}
        except (OSError, StoreError):
            if key not in self._refund_noted:
                self._refund_noted = self._refund_noted | {key}
                self.record_unmatched(sender, key, str(pending.price), "verified, run failed, credit not saved",
                                      verified=True)

    def _drop_covered(self, key: str) -> None:
        """Plans waiting for confirmation that a now-spent payment was going to cover."""
        self._pending = {s: p for s, p in self._pending.items() if p.covered_tx != key}

    async def _deliver(self, sender: str, pending: Pending, outcome: RunOutcome, out: Outbox,
                       paid_tx: str | None) -> None:
        if paid_tx is None or pending.price is None:
            await out.reply(sender, result_reply(outcome))
            return
        try:
            shares = split(pending.price, outcome.view, self.settings.pricing.host_fee_share)
        except Exception:
            log.exception("could not split %s", paid_tx[:16])
            shares = []
        reply = (result_reply(outcome, shares=shares, paid=pending.price, currency=self.settings.pricing.currency)
                 if shares else result_reply(outcome))
        # the answer exists, so the payment is spent now: the delivery status comes from the recipient's side and
        # must never decide about money. An undelivered answer is kept and resent once on "run it", never rerun.
        self._record_split(sender, pending, outcome, paid_tx, shares)
        self._spend(paid_tx)
        self._drop_covered(paid_tx)
        try:
            delivered = await out.reply(sender, reply)
        except Exception:
            log.exception("answer for payment %s raised on send; kept for one resend", paid_tx[:16])
            delivered = False
        if not delivered:
            log.warning("answer for payment %s was not delivered; kept for one resend", paid_tx[:16])
            self._undelivered = {**{k: v for k, v in list(self._undelivered.items())[-MAX_UNDELIVERED + 1:]},
                                 sender: reply}
