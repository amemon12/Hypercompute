"""The proxy's ledger: one row per settled payment and one row per machine's share of it (SQLite, local file).

The ledger is bookkeeping, not a payout: the whole payment lands in the host's pay_to address on chain, and the
shares say what the host owes each friend. A stream cut after settlement (a peer lost mid-reply) keeps the payment
row, records the tokens actually delivered, books a refund owed to the buyer for the undelivered part, and re-splits
what is left, so friends are only credited for compute that reached the buyer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .split import Contribution, Share, split

SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,              -- settled | delivered | interrupted
    settle_point TEXT NOT NULL,        -- first_token (streamed) | complete (non-streamed)
    tx TEXT NOT NULL,
    network TEXT NOT NULL,
    payer TEXT,
    amount_atomic INTEGER NOT NULL,
    max_tokens INTEGER NOT NULL,
    completion_tokens INTEGER,
    refund_owed_atomic INTEGER NOT NULL DEFAULT 0,
    model_id TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS shares (
    payment_id TEXT NOT NULL REFERENCES payments(id),
    node_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    units INTEGER NOT NULL,
    layers TEXT NOT NULL,
    is_host INTEGER NOT NULL,
    amount_atomic INTEGER NOT NULL,
    fee_atomic INTEGER NOT NULL,
    payout_address TEXT,
    PRIMARY KEY (payment_id, node_id)
);
CREATE TABLE IF NOT EXISTS payouts (           -- on-chain settle-ups from the host's wallet to a friend
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    node_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    address TEXT NOT NULL,
    amount_atomic INTEGER NOT NULL,
    tx TEXT NOT NULL,                          -- empty while pending
    network TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'sent'        -- pending (reserved before the transfer) | sent | failed
);
"""


@dataclass(frozen=True)
class Payment:
    id: str
    settle_point: str
    tx: str
    network: str
    payer: str | None
    amount_atomic: int
    max_tokens: int
    model_id: str
    deployment_id: str
    plan_id: str


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        # WAL: the page, the CLI and payout.py can read while the proxy writes; busy_timeout waits out a lock
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(SCHEMA)  # executescript commits on its own, so not inside _tx
        columns = {r["name"] for r in self._db.execute("PRAGMA table_info(payouts)")}
        if "status" not in columns:  # a ledger made before payouts had a status
            self._db.execute("ALTER TABLE payouts ADD COLUMN status TEXT NOT NULL DEFAULT 'sent'")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            try:
                self._db.execute("COMMIT")
            except sqlite3.Error:
                if self._db.in_transaction:  # a failed COMMIT must not leave the connection stuck mid-transaction
                    self._db.execute("ROLLBACK")
                raise

    def close(self) -> None:
        self._db.close()

    def record_settlement(self, payment: Payment, shares: list[Share]) -> None:
        if sum(s.amount_atomic for s in shares) != payment.amount_atomic:
            raise ValueError("shares must sum to the amount paid")
        with self._tx() as db:
            db.execute("INSERT INTO payments (id, created_at, status, settle_point, tx, network, payer, amount_atomic, "
                       "max_tokens, model_id, deployment_id, plan_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (payment.id, _now(), "settled", payment.settle_point, payment.tx, payment.network,
                        payment.payer, payment.amount_atomic, payment.max_tokens, payment.model_id,
                        payment.deployment_id, payment.plan_id))
            _insert_shares(db, payment.id, shares)

    def mark_delivered(self, payment_id: str, completion_tokens: int | None) -> None:
        with self._tx() as db:
            cur = db.execute("UPDATE payments SET status='delivered', completion_tokens=? WHERE id=?",
                             (completion_tokens, payment_id))
            if cur.rowcount == 0:
                raise KeyError(payment_id)

    def has_payment(self, payment_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM payments WHERE id=?", (payment_id,)).fetchone() is not None

    def mark_interrupted(self, payment_id: str, completion_tokens: int, refund_owed_atomic: int, note: str,
                         host_fee_bps: int) -> None:
        """The reply was cut after settlement: book the refund and re-split what the pool still earned."""
        with self._tx() as db:
            row = db.execute("SELECT amount_atomic FROM payments WHERE id=?", (payment_id,)).fetchone()
            if row is None:
                raise KeyError(payment_id)
            refund = max(0, min(refund_owed_atomic, row["amount_atomic"]))
            old = db.execute("SELECT * FROM shares WHERE payment_id=?", (payment_id,)).fetchall()
            contribs = [Contribution(r["node_id"], r["display_name"], r["units"], r["layers"], bool(r["is_host"]))
                        for r in old]
            payees = {r["node_id"]: r["payout_address"] for r in old if r["payout_address"] and not r["is_host"]}
            host_address = next((r["payout_address"] for r in old if r["is_host"]), "") or ""
            shares = split(row["amount_atomic"] - refund, contribs, host_fee_bps, payees, host_address)
            db.execute("DELETE FROM shares WHERE payment_id=?", (payment_id,))
            _insert_shares(db, payment_id, shares)
            db.execute("UPDATE payments SET status='interrupted', completion_tokens=?, refund_owed_atomic=?, note=? "
                       "WHERE id=?", (completion_tokens, refund, note, payment_id))

    def totals(self) -> list[dict]:
        """What each machine has earned across all payments, and how much of it was paid out on chain."""
        rows = self._db.execute(
            "SELECT s.node_id, s.display_name, MAX(s.is_host) AS is_host, SUM(s.amount_atomic) AS earned_atomic, "
            "COUNT(*) AS payments, MAX(s.payout_address) AS payout_address, "
            "COALESCE((SELECT SUM(p.amount_atomic) FROM payouts p WHERE p.node_id = s.node_id "
            "AND p.status = 'sent'), 0) AS paid_out_atomic, "
            "COALESCE((SELECT SUM(p.amount_atomic) FROM payouts p WHERE p.node_id = s.node_id "
            "AND p.status = 'pending'), 0) AS pending_payout_atomic "
            "FROM shares s GROUP BY s.node_id ORDER BY earned_atomic DESC").fetchall()
        return [dict(r) for r in rows]

    def owed(self) -> list[dict]:
        """Peers whose earnings are not yet paid out or reserved for a payout. The host is never owed: payments
        already land in its wallet."""
        out = []
        for t in self.totals():
            owed = t["earned_atomic"] - t["paid_out_atomic"] - t["pending_payout_atomic"]
            if not t["is_host"] and owed > 0:
                out.append({**t, "owed_atomic": owed})
        return out

    def reserve_payout(self, node_id: str, display_name: str, address: str, amount_atomic: int,
                       network: str) -> int:
        """Books a pending payout before the transfer is sent, re-checking what is owed inside the same transaction.
        A crash after the transfer leaves it pending (not paid again); two runs at once cannot both reserve it."""
        if amount_atomic <= 0:
            raise ValueError("a payout must be positive")
        with self._tx() as db:
            earned = db.execute("SELECT COALESCE(SUM(amount_atomic), 0) FROM shares WHERE node_id=? AND is_host=0",
                                (node_id,)).fetchone()[0]
            committed = db.execute("SELECT COALESCE(SUM(amount_atomic), 0) FROM payouts WHERE node_id=? "
                                   "AND status IN ('pending', 'sent')", (node_id,)).fetchone()[0]
            if amount_atomic > earned - committed:
                raise ValueError(f"{display_name} is owed {earned - committed}, not {amount_atomic}")
            cur = db.execute("INSERT INTO payouts (created_at, node_id, display_name, address, amount_atomic, tx, "
                             "network, status) VALUES (?,?,?,?,?,?,?,'pending')",
                             (_now(), node_id, display_name, address, amount_atomic, "", network))
            return int(cur.lastrowid)

    def complete_payout(self, payout_id: int, tx: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE payouts SET status='sent', tx=? WHERE id=? AND status='pending'", (tx, payout_id))

    def fail_payout(self, payout_id: int) -> None:
        """Only for a transfer known not to have been sent; an unknown outcome stays pending for a person to check."""
        with self._tx() as db:
            db.execute("UPDATE payouts SET status='failed' WHERE id=? AND status='pending'", (payout_id,))

    def payouts(self, limit: int = 20) -> list[dict]:
        rows = self._db.execute("SELECT * FROM payouts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*) AS payments, COALESCE(SUM(amount_atomic),0) AS paid_atomic, "
            "COALESCE(SUM(refund_owed_atomic),0) AS refunds_owed_atomic FROM payments").fetchone()
        return dict(row)

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._db.execute("SELECT * FROM payments ORDER BY created_at DESC, rowid DESC LIMIT ?",
                                (limit,)).fetchall()
        out = []
        for r in rows:
            shares = self._db.execute("SELECT * FROM shares WHERE payment_id=? ORDER BY is_host DESC, units DESC",
                                      (r["id"],)).fetchall()
            out.append({**dict(r), "shares": [dict(s) for s in shares]})
        return out

    def export_json(self) -> str:
        return json.dumps({"summary": self.summary(), "totals": self.totals(), "recent": self.recent(100),
                           "payouts": self.payouts(100)}, indent=2)


def _insert_shares(db: sqlite3.Connection, payment_id: str, shares: list[Share]) -> None:
    db.executemany("INSERT INTO shares (payment_id, node_id, display_name, units, layers, is_host, amount_atomic, "
                   "fee_atomic, payout_address) VALUES (?,?,?,?,?,?,?,?,?)",
                   [(payment_id, s.node_id, s.display_name, s.units, s.layers, int(s.is_host), s.amount_atomic,
                     s.fee_atomic, s.payout_address) for s in shares])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
