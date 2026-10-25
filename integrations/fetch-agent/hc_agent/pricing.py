"""Per-request price and the credit split across the machines that served it.

Price (FET) = base_fee + rate * (max_tokens / 1000) * (params_b / 100), clamped to [min_fee, max_fee].
It scales with what a request costs the pool: the model's size sets how much memory every machine gives up while it
is loaded and how much work each token is, and max_tokens bounds how long the pool is held.

Split: the coordinating host keeps host_fee_share; the rest is divided by layers served, so a friend whose laptop
held 3 of 61 layers earns 3/61 of it. Credits are an off-chain ledger (agent_data/credits.json) for the owner to
settle; the on-chain payment goes to the agent's wallet.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

from .config import Pricing
from .planner import PlanView
from .store import read_json, write_json

log = logging.getLogger(__name__)
QUANTUM = Decimal("0.000001")


@dataclass(frozen=True)
class Share:
    node_id: str
    node_name: str
    layers: int
    amount: Decimal
    role: str  # "host" | "peer"


def price_for(params_b: float, max_tokens: int, pricing: Pricing) -> Decimal:
    raw = (Decimal(str(pricing.base_fee))
           + Decimal(str(pricing.rate_per_1k_tokens_per_100b)) * Decimal(max_tokens) / Decimal(1000)
           * Decimal(str(params_b)) / Decimal(100))
    clamped = min(max(raw, Decimal(str(pricing.min_fee))), Decimal(str(pricing.max_fee)))
    return clamped.quantize(QUANTUM, rounding=ROUND_HALF_UP)


def split(total: Decimal, view: PlanView, host_fee_share: float) -> list[Share]:
    """Shares that sum exactly to total: rounding dust goes to the host."""
    layers = {}
    names = {}
    local = {}
    for s in view.stages:
        layers[s.node_id] = layers.get(s.node_id, 0) + s.layer_count
        names[s.node_id] = s.node_name
        local[s.node_id] = s.is_local
    host_id = next((nid for nid, is_local in local.items() if is_local), "local")
    names.setdefault(host_id, "host")
    total_layers = sum(layers.values())
    fee = (total * Decimal(str(host_fee_share))).quantize(QUANTUM, rounding=ROUND_DOWN)
    pool = total - fee
    shares: dict[str, Decimal] = {nid: Decimal(0) for nid in names}
    if total_layers:
        for nid, n in layers.items():
            shares[nid] += (pool * Decimal(n) / Decimal(total_layers)).quantize(QUANTUM, rounding=ROUND_DOWN)
    shares[host_id] += total - sum(shares.values())  # the fee plus rounding dust
    return [Share(node_id=nid, node_name=names[nid], layers=layers.get(nid, 0), amount=amount,
                  role="host" if nid == host_id else "peer")
            for nid, amount in sorted(shares.items(), key=lambda kv: (kv[0] != host_id, -layers.get(kv[0], 0)))]


class CreditLedger:
    """Append-only record of what each machine earned, persisted atomically as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        return read_json(self.path, {"nodes": {}, "entries": []})  # an unreadable ledger is never overwritten

    def record(self, *, request_id: str, model_id: str, payer: str, currency: str, shares: list[Share],
               transaction_id: str | None) -> dict:
        with self._lock:
            data = self._load()
            nodes = dict(data.get("nodes") or {})
            for s in shares:
                prev = nodes.get(s.node_id) or {"name": s.node_name, "earned": "0", "layers_served": 0}
                nodes[s.node_id] = {"name": s.node_name,
                                    "earned": str(Decimal(prev["earned"]) + s.amount),
                                    "layers_served": int(prev["layers_served"]) + s.layers}
            entry = {"at": datetime.now(UTC).isoformat(), "request_id": request_id, "model_id": model_id,
                     "payer": payer, "currency": currency, "transaction_id": transaction_id,
                     "shares": [{"node_id": s.node_id, "name": s.node_name, "layers": s.layers,
                                 "amount": str(s.amount), "role": s.role} for s in shares]}
            write_json(self.path, {"nodes": nodes, "entries": [*(data.get("entries") or []), entry]})
            return entry

    def has_transaction(self, tx_id: str) -> bool:
        with self._lock:
            return any(str(e.get("transaction_id") or "").upper() == tx_id.upper()
                       for e in self._load().get("entries") or [])

    def totals(self) -> Mapping[str, dict]:
        with self._lock:
            return dict(self._load().get("nodes") or {})
