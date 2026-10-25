"""The pool as the gateway reports it (GET /api/nodes + GET /api/network), reduced to what decides a layout.

Only eligible nodes count: a node the gateway has not verified, or one that is stale or left, cannot hold a stage, so a
join is seen when a peer turns eligible and a leave when it stops being eligible or disappears.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

LOCAL_ID = "local"  # services/nodes.py LOCAL_ID: the gateway's own machine
_UNSAFE = re.compile(r"[^A-Za-z0-9 ._()+:/-]")


def clean(text: object, limit: int = 40) -> str:
    """Peer-supplied text (display names, device descriptions) reaches Droid's prompt and the page. Keep it short
    and to plain name characters so a peer cannot write instructions into either."""
    return _UNSAFE.sub("", str(text)).strip()[:limit] or "?"


@dataclass(frozen=True)
class DeviceView:
    name: str  # as the node's worker names it (contribution.devices), or the local engine device
    description: str
    total_bytes: int | None
    free_bytes: int | None


@dataclass(frozen=True)
class NodeFacts:
    id: str
    name: str
    is_local: bool
    state: str
    devices: tuple[DeviceView, ...]
    max_bytes: int | None  # contribution.max_bytes: what Hypercompute may plan onto this node
    reserve_bytes: int | None
    path: str | None = None  # Iroh path to this peer: direct, relay, connecting; None on a LAN gateway
    rtt_ms: float | None = None
    rpc_count: int | None = None  # RPC devices the host's probe verified: the gateway numbers RPC names by this

    @property
    def eligible(self) -> bool:
        return self.state == "eligible"

    @property
    def fingerprint(self) -> tuple[str, tuple[str, ...]]:
        return self.id, tuple(d.name for d in self.devices)


@dataclass(frozen=True)
class PoolChange:
    joined: tuple[str, ...] = ()
    left: tuple[str, ...] = ()
    names: Mapping[str, str] = field(default_factory=dict, compare=False)

    @property
    def empty(self) -> bool:
        return not self.joined and not self.left

    def describe(self) -> str:
        parts = [f"{self.names.get(n, n[:8])} joined" for n in self.joined]
        parts += [f"{self.names.get(n, n[:8])} left" for n in self.left]
        return ", ".join(parts) or "no change"


@dataclass(frozen=True)
class PoolSnapshot:
    nodes: tuple[NodeFacts, ...]
    transport: str = "lan"

    @property
    def local(self) -> NodeFacts | None:
        return next((n for n in self.nodes if n.is_local), None)

    @property
    def peers(self) -> tuple[NodeFacts, ...]:
        return tuple(n for n in self.nodes if not n.is_local and n.eligible)

    @property
    def fingerprint(self) -> frozenset[tuple[str, tuple[str, ...]]]:
        return frozenset(p.fingerprint for p in self.peers)

    def name_of(self, node_id: str) -> str:
        if node_id == LOCAL_ID:
            local = self.local
            return local.name if local else "this machine"
        return next((n.name for n in self.nodes if n.id == node_id), node_id[:8])

    def node(self, node_id: str) -> NodeFacts | None:
        if node_id == LOCAL_ID:
            return self.local
        return next((n for n in self.nodes if n.id == node_id), None)


def diff(before: PoolSnapshot, after: PoolSnapshot) -> PoolChange:
    """Eligible peers that appeared or went away. A peer whose device list changed counts as leaving and joining."""
    old, new = before.fingerprint, after.fingerprint
    names = {n.id: n.name for n in (*before.nodes, *after.nodes)}
    joined = tuple(sorted({nid for nid, _ in new - old}))
    left = tuple(sorted({nid for nid, _ in old - new} - set(joined)))
    return PoolChange(joined=joined, left=left, names=names)


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _paths(network: Mapping[str, Any]) -> dict[str, tuple[str | None, float | None]]:
    out: dict[str, tuple[str | None, float | None]] = {}
    for peer in network.get("peers") or []:
        if not isinstance(peer, Mapping) or not isinstance(peer.get("node_id"), str):
            continue
        path = peer.get("path") if isinstance(peer.get("path"), Mapping) else {}
        rtt = path.get("rtt_ms")
        out[peer["node_id"]] = (path.get("mode") if isinstance(path.get("mode"), str) else None,
                                float(rtt) if isinstance(rtt, int | float) and not isinstance(rtt, bool) else None)
    return out


def _devices(node: Mapping[str, Any]) -> tuple[DeviceView, ...]:
    """A peer's worker devices in its -d order, sized by the RPC probe; the local node's contributed accelerators."""
    contribution = node.get("contribution") or {}
    names = [d for d in contribution.get("devices") or [] if isinstance(d, str)]
    verification = node.get("verification") or {}
    probed = [d for d in verification.get("rpc_devices") or [] if isinstance(d, Mapping)]
    inventory = {d.get("id"): d for d in (node.get("inventory") or {}).get("devices") or [] if isinstance(d, Mapping)}
    out = []
    for i, name in enumerate(names):
        probe = probed[i] if i < len(probed) else inventory.get(name, {})
        out.append(DeviceView(name=clean(name, 16), description=clean(probe.get("description") or probe.get("name")
                                                                          or name, 60),
                              total_bytes=_int(probe.get("total_bytes")), free_bytes=_int(probe.get("free_bytes"))))
    return tuple(out)


def snapshot(nodes: Sequence[Mapping[str, Any]], network: Mapping[str, Any]) -> PoolSnapshot:
    """Parse the gateway's JSON defensively: a malformed node is skipped, never trusted."""
    paths = _paths(network)
    facts = []
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str):
            continue
        is_local = bool(node.get("is_local"))
        contribution = node.get("contribution") or {}
        path, rtt = paths.get(node["id"], (None, None))
        probed = (node.get("verification") or {}).get("rpc_devices")
        facts.append(NodeFacts(
            id=LOCAL_ID if is_local else clean(node["id"], 64), name=clean(node.get("display_name") or node["id"][:8]),
            is_local=is_local, state=clean(node.get("state") or "unknown", 16), devices=_devices(node),
            max_bytes=_int(contribution.get("max_bytes")), reserve_bytes=_int(contribution.get("reserve_bytes")),
            path=path, rtt_ms=rtt, rpc_count=len(probed) if isinstance(probed, list) else None))
    return PoolSnapshot(nodes=tuple(facts), transport=str(network.get("transport") or "lan"))
