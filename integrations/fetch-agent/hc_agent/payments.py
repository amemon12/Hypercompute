"""Verifying an AgentPaymentProtocol CommitPayment against the Fetch.ai ledger.

ASI:One pays the agent's wallet directly (payment_method fet_direct) and reports the transaction hash in
CommitPayment. The agent never trusts that report: it looks the transaction up on chain and checks that it succeeded,
paid this agent's wallet exactly the quoted amount in the right denomination, was made after the quote, came from
the buyer's wallet when the buyer named one, and has not been used for an earlier request.

Transfers to the agent's wallet are public, so a hash alone proves nothing about who paid. Each quote therefore
carries a unique tag in its last digits (conversation.unique_quote): a transfer counts only for the one open request
whose amount it matches exactly, so nobody can claim another person's payment as their own.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from .store import read_json, write_json

log = logging.getLogger(__name__)
ATTO = Decimal(10) ** 18
DENOMS = {"stable-testnet": "atestfet", "mainnet": "afet"}
TX_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
CLOCK_SKEW = timedelta(minutes=2)
MICRO_ATTO = 10**12  # quotes are whole micro-FET; on-chain amounts are compared at that precision
LOOKUP_TIMEOUT_S = 15.0  # one ledger query
LOOKUP_PATIENCE_S = 45.0  # how long a fresh transaction may take to be indexed
LOOKUP_EVERY_S = 5.0


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    retryable: bool = False  # the lookup failed or the tx is not indexed yet: keep the request open


@dataclass(frozen=True)
class PaymentAsk:
    amount: Decimal
    currency: str
    recipient: str  # the agent's fetch1... wallet address
    reference: str
    description: str
    deadline_s: int = 300
    network: str = "stable-testnet"


class Verifier(Protocol):
    async def verify(self, tx_id: str, *, amount: Decimal, buyer_wallet: str | None,
                     not_before: datetime | None = None) -> Verdict: ...


def to_atto(amount: Decimal) -> int:
    return int((amount * ATTO).to_integral_value())


def _transfers(events: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Mapping[str, str]]:
    """Transfer attribute maps from cosmpy's {type: {attr: value}} form (one per message log) or from a raw list of
    {type, attributes} events."""
    out: list[Mapping[str, str]] = []
    if isinstance(events, Mapping):
        t = events.get("transfer")
        if isinstance(t, Mapping):
            out.append(t)
        return out
    for ev in events:
        if isinstance(ev, Mapping) and ev.get("type") == "transfer":
            out.append({a.get("key"): a.get("value") for a in ev.get("attributes") or []})
        elif isinstance(ev, Mapping):
            out += _transfers(ev)
    return out


def check_transfer_events(events: Mapping[str, Any] | Iterable[Mapping[str, Any]], *, recipient: str, atto: int,
                          denom: str, buyer_wallet: str | None) -> Verdict:
    """A payment of exactly atto `denom` (to the micro-FET) to recipient among the transaction's transfers."""
    seen_to_us = None
    for t in _transfers(events):
        if t.get("recipient") != recipient:
            continue
        if buyer_wallet and t.get("sender") != buyer_wallet:
            seen_to_us = "the transfer came from a different wallet than the buyer named"
            continue
        total = 0
        for part in str(t.get("amount") or "").split(","):
            m = re.fullmatch(r"(\d+)([a-z]+)", part.strip())
            if m and m.group(2) == denom:
                total += int(m.group(1))
        if (total + MICRO_ATTO // 2) // MICRO_ATTO == (atto + MICRO_ATTO // 2) // MICRO_ATTO:
            return Verdict(True, "verified on chain")
        seen_to_us = f"the transfer paid {Decimal(total) / ATTO} instead of exactly {Decimal(atto) / ATTO}"
    return Verdict(False, seen_to_us or "no transfer to this agent's wallet in that transaction")


def tx_events(tx: Any) -> list[Mapping[str, Any]]:
    """cosmpy merges same-type events per scope, so read the transaction-level map and every message log's map."""
    out = [tx.events] if isinstance(getattr(tx, "events", None), Mapping) else []
    for entry in getattr(tx, "logs", None) or []:
        events = getattr(entry, "events", None)
        if isinstance(events, Mapping):
            out.append(events)
    return out


class LedgerVerifier:
    """Looks transactions up with cosmpy on the configured network."""

    def __init__(self, recipient: str, network: str = "stable-testnet", used_path: Path | None = None, *,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, patience_s: float = LOOKUP_PATIENCE_S
                 ) -> None:
        if network not in DENOMS:
            raise ValueError(f"unknown network {network!r}")
        self.recipient, self.network, self.denom = recipient, network, DENOMS[network]
        self._used_path = used_path
        self._used: set[str] = set(read_json(used_path, [])) if used_path is not None else set()  # fails closed
        self._inflight: set[str] = set()
        self._sleep, self._patience_s = sleep, patience_s

    def _query(self, tx_id: str) -> Any:
        from cosmpy.aerial.client import LedgerClient, NetworkConfig

        config = NetworkConfig.fetchai_stable_testnet() if self.network == "stable-testnet" else \
            NetworkConfig.fetchai_mainnet()
        return LedgerClient(config).query_tx(tx_id)

    def is_used(self, tx_id: str) -> bool:
        return tx_id.upper() in self._used

    def _remember(self, tx_id: str) -> None:
        self._used = self._used | {tx_id}  # in memory first: replays are refused even if the write below fails
        if self._used_path is not None:
            try:
                write_json(self._used_path, sorted(self._used))
            except OSError:
                log.exception("could not persist used transaction %s; it is refused until restart only", tx_id)

    async def _lookup(self, tx_id: str) -> Any | None:
        """The transaction, polling while it is not indexed yet; None if it never shows up in time."""
        waited = 0.0
        while True:
            try:
                return await asyncio.wait_for(asyncio.to_thread(self._query, tx_id), LOOKUP_TIMEOUT_S)
            except Exception as exc:  # cosmpy raises NotFoundError, grpc and HTTP errors; all mean "not yet"
                log.info("payment lookup for %s: %s", tx_id[:12], type(exc).__name__)
            if waited >= self._patience_s:
                return None
            await self._sleep(LOOKUP_EVERY_S)
            waited += LOOKUP_EVERY_S

    async def verify(self, tx_id: str, *, amount: Decimal, buyer_wallet: str | None,
                     not_before: datetime | None = None) -> Verdict:
        if not TX_RE.match(tx_id or ""):
            return Verdict(False, "that is not a transaction hash")
        key = tx_id.upper()
        if key in self._used:
            return Verdict(False, "that transaction already paid for an earlier request")
        if key in self._inflight:
            return Verdict(False, "that transaction is already being checked", retryable=True)
        self._inflight = self._inflight | {key}
        try:
            tx = await self._lookup(tx_id)
            if tx is None:
                return Verdict(False, "the transaction was not found on the ledger yet", retryable=True)
            if not tx.is_successful():
                return Verdict(False, "the transaction failed on chain")
            stamp = getattr(tx, "timestamp", None)
            if not_before is not None:
                if not isinstance(stamp, datetime):
                    return Verdict(False, "the transaction time could not be read")
                if stamp < not_before - CLOCK_SKEW:
                    return Verdict(False, "the transaction was made before this payment was requested")
            verdict = check_transfer_events(tx_events(tx), recipient=self.recipient, atto=to_atto(amount),
                                            denom=self.denom, buyer_wallet=buyer_wallet)
            if verdict.ok:
                if key in self._used:  # another check of the same hash finished first
                    return Verdict(False, "that transaction already paid for an earlier request")
                self._remember(key)
            return verdict
        finally:
            self._inflight = self._inflight - {key}
