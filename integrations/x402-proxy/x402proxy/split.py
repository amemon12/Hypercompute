"""Who earns what from one payment: the deployment plan says which machine served which layers.

Each machine's weight is its layer units, the same count llama.cpp gets in --tensor-split: one per repeating layer it
holds, plus one for the output head. The host first takes a fee (it runs the gateway, the embeddings, the sampler and
pays nothing extra to do so), then the rest is split by units. Shares are floored to whole atomic units and the
remainder goes to the host, so the shares always sum to exactly the amount paid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

HOST_NODE_ID = "local"  # how a plan names the gateway's own machine
BPS = 10_000


class SplitError(ValueError):
    """A plan or a fee that cannot be split."""


@dataclass(frozen=True)
class Contribution:
    node_id: str
    display_name: str
    units: int  # layers held, plus one for the output head
    layers: str  # e.g. "0-2" or "11-60", for people reading the ledger
    is_host: bool


@dataclass(frozen=True)
class Share:
    node_id: str
    display_name: str
    units: int
    layers: str
    is_host: bool
    amount_atomic: int  # everything this machine earns from the payment, fee included for the host
    fee_atomic: int  # the host fee part of amount_atomic (0 for peers)
    payout_address: str | None  # None until the friend registers a public address


def layer_ranges(layers: list[int]) -> str:
    """[0, 1, 2, 5] -> '0-2,5'; [] -> '-'."""
    if not layers:
        return "-"
    ordered = sorted(set(layers))
    runs: list[list[int]] = [[ordered[0], ordered[0]]]
    for n in ordered[1:]:
        if n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


def contributions(plan: Mapping, names: Mapping[str, str]) -> list[Contribution]:
    """One entry per machine in the plan, host first, then peers by units served. `names` maps node id to the
    display name the gateway shows (the host is HOST_NODE_ID in a plan)."""
    layers_by_node: dict[str, list[int]] = {}
    head_by_node: dict[str, int] = {}
    for a in plan.get("assignments") or []:
        node = str(a["node_id"])
        layers_by_node.setdefault(node, []).extend(int(n) for n in a.get("layers") or [])
        head_by_node[node] = head_by_node.get(node, 0) + (1 if a.get("holds_output") else 0)
    if not layers_by_node:
        raise SplitError("the plan has no device assignments")
    out = [Contribution(node_id=node, display_name=names.get(node, node[:12]),
                        units=len(set(layers)) + head_by_node[node], layers=layer_ranges(layers),
                        is_host=node == HOST_NODE_ID)
           for node, layers in layers_by_node.items()]
    if not any(c.is_host for c in out):
        out.append(Contribution(HOST_NODE_ID, names.get(HOST_NODE_ID, "host"), 0, "-", True))
    return sorted(out, key=lambda c: (not c.is_host, -c.units, c.display_name))


def split(amount_atomic: int, contribs: list[Contribution], host_fee_bps: int,
          payees: Mapping[str, str], host_address: str) -> list[Share]:
    """Shares of one payment. `payees` maps a display name or node id to a public payout address."""
    if amount_atomic < 0:
        raise SplitError("amount must not be negative")
    if not 0 <= host_fee_bps <= BPS:
        raise SplitError("host fee must be between 0 and 10000 basis points")
    hosts = [c for c in contribs if c.is_host]
    if len(hosts) != 1:
        raise SplitError("exactly one host contribution is required")
    fee = amount_atomic * host_fee_bps // BPS
    pool = amount_atomic - fee
    total_units = sum(c.units for c in contribs)
    raw = {c.node_id: (pool * c.units // total_units if total_units else 0) for c in contribs}
    remainder = amount_atomic - fee - sum(raw.values())
    host_id = hosts[0].node_id
    return [Share(node_id=c.node_id, display_name=c.display_name, units=c.units, layers=c.layers,
                  is_host=c.is_host,
                  amount_atomic=raw[c.node_id] + (fee + remainder if c.node_id == host_id else 0),
                  fee_atomic=fee if c.node_id == host_id else 0,
                  payout_address=host_address if c.is_host else payees.get(c.display_name) or payees.get(c.node_id))
            for c in contribs]
