"""Replays unrecorded-settlements.jsonl into the ledger: settlements the ledger could not take when the money moved,
and the outcomes (delivered, interrupted with a refund owed) that followed them. Safe to run twice: a payment already
in the ledger is skipped, and the file is renamed once it is replayed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .ledger import Ledger, Payment
from .split import Share


def replay(ledger: Ledger, fallback: Path) -> dict[str, int]:
    counts = {"settlements": 0, "outcomes": 0, "skipped": 0}
    if not fallback.is_file():
        return counts
    records = [json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()]
    for rec in records:
        if rec.get("kind", "settlement") == "settlement":
            payment = Payment(**rec["payment"])
            if ledger.has_payment(payment.id):
                counts["skipped"] += 1
                continue
            ledger.record_settlement(payment, [Share(**s) for s in rec["shares"]])
            counts["settlements"] += 1
    for rec in records:
        if rec.get("kind") != "outcome" or not ledger.has_payment(rec["payment_id"]):
            continue
        if rec["outcome"] == "interrupted":
            ledger.mark_interrupted(rec["payment_id"], rec["completion_tokens"], rec["refund_owed_atomic"],
                                    rec.get("note", ""), rec["host_fee_bps"])
        else:
            ledger.mark_delivered(rec["payment_id"], rec["completion_tokens"])
        counts["outcomes"] += 1
    fallback.rename(fallback.with_name(f"{fallback.stem}.replayed-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl"))
    return counts
