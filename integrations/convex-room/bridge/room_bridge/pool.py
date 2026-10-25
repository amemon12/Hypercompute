"""Turns the gateway's /api/nodes, /api/deployments and /api/network into the room's machine cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gateway import GatewayState

LIVE_STATES = ("ready", "warming", "loading", "preparing", "reserving", "planned", "draining")
MAX_MACHINES = 32


@dataclass(frozen=True)
class PoolView:
    ready: bool
    slots: int
    payload: dict[str, Any]  # the /bridge/pool body, minus code, bridgeId and title


def build_pool(state: GatewayState, slots_override: int | None = None) -> PoolView:
    if not state.reachable:
        return PoolView(False, 0, _payload(None, "gateway offline", None, None, 1, []))
    dep = active_deployment(state.deployments)
    nodes = {n.get("id"): n for n in state.nodes if isinstance(n.get("id"), str)}
    paths = peer_paths(state.network)
    machines: list[dict[str, Any]] = []
    used: set[str] = set()
    plan = (dep or {}).get("plan") or {}
    for a in plan.get("assignments") or []:
        node_id = str(a.get("node_id") or "")
        layers = [x for x in a.get("layers") or [] if isinstance(x, int)]
        machines.append(_machine(node_id, nodes.get(node_id), paths, str(a.get("device") or "?"), layers,
                                 int(a.get("weight_bytes") or 0), bool(a.get("holds_output"))))
        used.add(node_id)
    for node_id, node in nodes.items():  # joined machines the plan leaves out still show up, with 0 layers
        if node_id not in used and node.get("state") not in ("left", "removed"):
            device = _first_device(node)
            machines.append(_machine(node_id, node, paths, device, [], 0, False))
    slots = slots_override or _plan_slots(plan)
    state_name = str(dep.get("state")) if dep else "no model loaded"
    placement = (dep or {}).get("placement") or {}
    payload = _payload(plan.get("model_id"), state_name, plan.get("tier"), placement.get("offloaded"), slots,
                       machines[:MAX_MACHINES])
    return PoolView(state_name == "ready" and dep is not None and dep.get("purpose") != "tune", slots, payload)


def active_deployment(deployments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The ready serving deployment, else the newest one still on its way up or down."""
    serving = [d for d in deployments if d.get("purpose", "run") != "tune" and d.get("state") in LIVE_STATES]
    for d in serving:
        if d.get("state") == "ready":
            return d
    return max(serving, key=lambda d: str(d.get("created_at") or ""), default=None)


def peer_paths(network: dict[str, Any]) -> dict[str, str]:
    """node_id -> direct | relay | connecting, from the Iroh host status. A LAN gateway has no per-peer path."""
    if network.get("transport") != "iroh":
        return {}
    out: dict[str, str] = {}
    for p in network.get("peers") or []:
        if not isinstance(p, dict) or not isinstance(p.get("node_id"), str):
            continue
        mode = (p.get("path") or {}).get("mode") if p.get("connected") else "connecting"
        out[p["node_id"]] = mode if mode in ("direct", "relay", "connecting") else "unknown"
    return out


def layer_range(layers: list[int]) -> str | None:
    """[0,1,2,5,6] -> "0-2, 5-6"."""
    if not layers:
        return None
    runs: list[str] = []
    xs = sorted(set(layers))
    start = prev = xs[0]
    for x in xs[1:]:
        if x != prev + 1:
            runs.append(f"{start}-{prev}" if start != prev else str(start))
            start = x
        prev = x
    runs.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(runs)


def _machine(node_id: str, node: dict[str, Any] | None, paths: dict[str, str], device: str, layers: list[int],
             weight_bytes: int, holds_output: bool) -> dict[str, Any]:
    node = node or {}
    local = bool(node.get("is_local")) or node_id == "local"
    os_name = ((node.get("inventory") or {}).get("os") or {}).get("system")
    return {
        "nodeId": node_id or "?",
        "name": str(node.get("display_name") or node_id or "unknown")[:64],
        "device": device[:32],
        "layers": len(layers),
        "layerRange": layer_range(layers),
        "weightBytes": max(0, weight_bytes),
        "holdsOutput": holds_output,
        "path": "local" if local else paths.get(node_id, "lan" if not paths else "unknown"),
        "state": str(node.get("state") or "unknown")[:32],
        "os": os_name if isinstance(os_name, str) else None,
    }


def _first_device(node: dict[str, Any]) -> str:
    allowed = (node.get("contribution") or {}).get("devices") or []
    if allowed and isinstance(allowed[0], str):
        return allowed[0]
    devices = (node.get("inventory") or {}).get("devices") or []
    return str(devices[0].get("id")) if devices and isinstance(devices[0], dict) else "CPU"


def _plan_slots(plan: dict[str, Any]) -> int:
    raw = (plan.get("engine_options") or {}).get("parallel", 1)
    return max(1, min(int(raw), 16)) if isinstance(raw, int) else 1


def _payload(model_id: Any, state: str, tier: Any, offloaded: Any, slots: int,
             machines: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "modelId": model_id if isinstance(model_id, str) else None,
        "deploymentState": state,
        "tier": tier if isinstance(tier, str) else None,
        "offloaded": offloaded if isinstance(offloaded, str) else None,
        "slots": slots,
        "machines": machines,
    }
