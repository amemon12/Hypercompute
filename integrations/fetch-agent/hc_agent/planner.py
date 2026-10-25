"""Pure planning logic: summarize the pool, rank candidate models, read a gateway plan into stages.

Nothing here talks to the network; workflow.py calls the gateway and hands the JSON to these functions, so every
decision the agent makes about which model and which machines can be tested from recorded gateway output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .intent import Intent

GIB = 1024**3
TERMINAL = frozenset({"stopped", "failed"})
GPU_TIERS = frozenset({"gpu", "gpu_ram"})
STALE_HEARTBEAT_S = 60.0


class NoMatchingModel(LookupError):
    """The user named a model the pool does not have downloaded."""

    def __init__(self, hint: str, available: Sequence[str]) -> None:
        super().__init__(f"no downloaded model matches {hint!r}")
        self.hint, self.available = hint, list(available)


@dataclass(frozen=True)
class NodeSummary:
    id: str
    name: str
    is_local: bool
    state: str
    online: bool
    contribution_bytes: int
    memory_bytes: int
    devices: tuple[str, ...]
    path_mode: str | None = None  # direct | relay | connecting (Iroh), None on the LAN transport
    rtt_ms: float | None = None
    state_reason: str | None = None


@dataclass(frozen=True)
class ModelSummary:
    id: str
    name: str
    family: str
    params_b: float
    quantization: str
    size_bytes: int
    usable: bool
    role: str
    n_layer: int | None
    status: str


@dataclass(frozen=True)
class Stage:
    node_id: str
    node_name: str
    is_local: bool
    device: str
    layers: tuple[int, ...]
    holds_output: bool
    weight_bytes: int
    file_backed: bool
    endpoint: str | None = None
    observed_bytes: int | None = None  # from the engine's load log, once the deployment is up

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def layer_range(self) -> str:
        if not self.layers:
            return "experts only" if self.weight_bytes else "no layers"
        lo, hi = min(self.layers), max(self.layers)
        return f"layer {lo}" if lo == hi else f"layers {lo}-{hi}"


@dataclass(frozen=True)
class PlanView:
    plan_id: str
    model_id: str
    feasible: bool
    tier: str
    capacity_mode: bool
    stages: tuple[Stage, ...]
    predicted_tok_s: float | None  # an estimate from the gateway's priors, never a measurement
    warnings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    created_at: str | None = None

    @property
    def machines(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for s in self.stages:
            if s.layer_count or s.weight_bytes:
                seen.setdefault(s.node_name, None)
        return tuple(seen)

    @property
    def peer_count(self) -> int:
        return len({s.node_id for s in self.stages if not s.is_local and s.layer_count})

    def layers_by_node(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.stages:
            out[s.node_id] = out.get(s.node_id, 0) + s.layer_count
        return out


@dataclass(frozen=True)
class PoolSnapshot:
    nodes: tuple[NodeSummary, ...]
    models: tuple[ModelSummary, ...]
    deployments: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    transport: str = "lan"

    @property
    def host(self) -> NodeSummary | None:
        return next((n for n in self.nodes if n.is_local), None)

    @property
    def online_peers(self) -> tuple[NodeSummary, ...]:
        return tuple(n for n in self.nodes if not n.is_local and n.online)

    @property
    def offline_peers(self) -> tuple[NodeSummary, ...]:
        return tuple(n for n in self.nodes if not n.is_local and not n.online and n.state not in ("left", "removed"))

    @property
    def pooled_bytes(self) -> int:
        return sum(n.contribution_bytes for n in self.nodes if n.online)

    def active(self) -> Mapping[str, Any] | None:
        """The deployment occupying the pool (any state but stopped or failed), if one does."""
        return next((d for d in self.deployments if d.get("state") not in TERMINAL), None)

    def node_name(self, node_id: str) -> str:
        if node_id == "local":
            host = self.host
            return host.name if host else "host"
        return next((n.name for n in self.nodes if n.id == node_id), node_id[:8])


def _memory_bytes(node: Mapping[str, Any]) -> int:
    pools = node.get("memory_pools")
    if pools is None:
        pools = (node.get("inventory") or {}).get("memory_pools") or []
    return sum(int(p.get("total_bytes") or 0) for p in pools if isinstance(p, Mapping))


def _devices(node: Mapping[str, Any]) -> tuple[str, ...]:
    contribution = node.get("contribution") or {}
    return tuple(str(d) for d in contribution.get("devices") or [])


def clean_name(name: str) -> str:
    return re.sub(r"\s*\(this machine\)\s*$", "", name).strip() or name


def summarize_nodes(nodes: Iterable[Mapping[str, Any]], network: Mapping[str, Any] | None) -> tuple[NodeSummary, ...]:
    paths: dict[str, Mapping[str, Any]] = {}
    for peer in (network or {}).get("peers") or []:
        if isinstance(peer, Mapping) and peer.get("node_id"):
            paths[str(peer["node_id"])] = peer
    out = []
    for n in nodes:
        state = str(n.get("state") or "unknown")
        age = n.get("heartbeat_age_s")
        is_local = bool(n.get("is_local"))
        fresh = is_local or age is None or float(age) < STALE_HEARTBEAT_S
        peer = paths.get(str(n.get("id")))
        path = (peer or {}).get("path") or {}
        connected = peer is None or bool(peer.get("connected"))
        out.append(NodeSummary(
            id=str(n.get("id")), name=clean_name(str(n.get("display_name") or n.get("id"))), is_local=is_local,
            state=state, online=state == "eligible" and fresh and connected,
            contribution_bytes=int((n.get("contribution") or {}).get("max_bytes") or 0),
            memory_bytes=_memory_bytes(n), devices=_devices(n),
            path_mode=path.get("mode") if peer else None,
            rtt_ms=float(path["rtt_ms"]) if isinstance(path.get("rtt_ms"), (int, float)) else None,
            state_reason=n.get("state_reason")))
    return tuple(sorted(out, key=lambda s: (not s.is_local, s.name)))


def summarize_models(models: Iterable[Mapping[str, Any]]) -> tuple[ModelSummary, ...]:
    out = []
    for v in models:
        art = v.get("artifact") or {}
        status = str(v.get("status") or "unknown")
        usable = (status == "downloaded" and bool(v.get("verified_sha256"))) or status == "engine_managed"
        size = int(v.get("total_bytes") or art.get("size_bytes") or sum(int(f.get("size_bytes") or 0)
                                                                         for f in art.get("files") or []))
        out.append(ModelSummary(
            id=str(art.get("id")), name=str(art.get("display_name") or art.get("id")),
            family=str(art.get("family") or ""), params_b=float(art.get("params_b") or 0.0),
            quantization=str(art.get("quantization") or ""), size_bytes=size, usable=usable,
            role=str(art.get("role") or "standard"), n_layer=(v.get("gguf") or {}).get("n_layer"), status=status))
    return tuple(out)


def snapshot(nodes: Iterable[Mapping[str, Any]], network: Mapping[str, Any] | None,
             models: Iterable[Mapping[str, Any]], deployments: Iterable[Mapping[str, Any]]) -> PoolSnapshot:
    return PoolSnapshot(nodes=summarize_nodes(nodes, network), models=summarize_models(models),
                        deployments=tuple(deployments), transport=str((network or {}).get("transport") or "lan"))


def _matches(model: ModelSummary, hint: str) -> bool:
    hint = hint.lower().replace(" ", "")
    hay = f"{model.id} {model.name} {model.family}".lower().replace(" ", "")
    return hint in hay


def rank_candidates(models: Sequence[ModelSummary], intent: Intent) -> list[ModelSummary]:
    """Usable models in the order the agent will try them: the named one, else biggest first. Plumbing models
    (tiny test models) come last, so they are only picked when nothing else fits."""
    usable = [m for m in models if m.usable]
    if intent.model_hint:
        named = [m for m in usable if _matches(m, intent.model_hint)]
        if not named:
            raise NoMatchingModel(intent.model_hint, [m.name for m in usable])
        usable = named
    if intent.size_hint_b:
        target = intent.size_hint_b
        return sorted(usable, key=lambda m: (abs(m.params_b - target), -m.params_b))
    return sorted(usable, key=lambda m: (m.role == "plumbing", -m.params_b, -m.size_bytes))


def plan_ok(plan: Mapping[str, Any]) -> bool:
    return bool(plan.get("feasible")) and plan.get("fit_label") != "does_not_fit"


def _predicted_tok_s(plan: Mapping[str, Any]) -> float | None:
    pred = plan.get("prediction") or {}
    ms = pred.get("decode_ms")
    if not isinstance(ms, (int, float)) or ms <= 0:
        ms = (plan.get("moe") or {}).get("decode_ms_est")
    if not isinstance(ms, (int, float)) or ms <= 0:
        ms = (plan.get("stream") or {}).get("predicted_io_ms_per_token")
    return 1000.0 / float(ms) if isinstance(ms, (int, float)) and ms > 0 else None


def read_plan(plan: Mapping[str, Any], pool: PoolSnapshot) -> PlanView:
    stages = []
    host_ids = {"local"} | {n.id for n in pool.nodes if n.is_local}
    for a in plan.get("assignments") or []:
        node_id = str(a.get("node_id"))
        is_local = node_id in host_ids
        stages.append(Stage(
            node_id="local" if is_local else node_id, node_name=pool.node_name("local" if is_local else node_id),
            is_local=is_local, device=str(a.get("device")), layers=tuple(int(x) for x in a.get("layers") or []),
            holds_output=bool(a.get("holds_output")), weight_bytes=int(a.get("weight_bytes") or 0),
            file_backed=bool(a.get("file_backed")), endpoint=a.get("endpoint")))
    cpu_layers = int(plan.get("cpu_layers") or 0)
    if cpu_layers > 0 and not any(s.is_local and s.device == "CPU" and s.layers for s in stages):
        stages.insert(0, Stage(node_id="local", node_name=pool.node_name("local"), is_local=True, device="CPU",
                               layers=tuple(range(cpu_layers)), holds_output=False, weight_bytes=0,
                               file_backed=False))
    return PlanView(plan_id=str(plan.get("id")), model_id=str(plan.get("model_id")), feasible=plan_ok(plan),
                    tier=str(plan.get("tier") or "gpu"), capacity_mode=bool(plan.get("capacity_mode")),
                    stages=tuple(stages), predicted_tok_s=_predicted_tok_s(plan),
                    warnings=tuple(str(w) for w in plan.get("warnings") or []),
                    reasons=tuple(str(r) for r in plan.get("reasons") or []),
                    created_at=plan.get("created_at"))


def with_observed(view: PlanView, deployment: Mapping[str, Any]) -> PlanView:
    """Attach the bytes llama.cpp reported per device in its load log (RPC devices are named by endpoint)."""
    placement = deployment.get("placement") or {}
    observed = {str(d.get("name")): int(d.get("model_bytes") or 0) for d in placement.get("devices") or []}
    if not observed:
        return view
    stages = []
    for s in view.stages:
        got = None
        if s.endpoint:
            got = next((b for name, b in observed.items() if name.endswith(f"[{s.endpoint}]")), None)
        elif s.is_local:
            got = observed.get("CPU_Mapped" if s.device == "CPU" and not s.layers else s.device)
        stages.append(Stage(**{**s.__dict__, "observed_bytes": got}))
    return PlanView(**{**view.__dict__, "stages": tuple(stages)})


def choose(preference: str, tried: Sequence[tuple[ModelSummary, Mapping[str, Any] | None]]
           ) -> tuple[ModelSummary, Mapping[str, Any]] | None:
    """Pick from (model, plan or None) pairs in rank order. biggest: the first feasible plan. fastest: the biggest
    feasible plan on a GPU tier (no disk-backed capacity mode), else the first feasible one."""
    feasible = [(m, p) for m, p in tried if p is not None and plan_ok(p)]
    if not feasible:
        return None
    if preference == "fastest":
        quick = [(m, p) for m, p in feasible if not p.get("capacity_mode") and p.get("tier") in GPU_TIERS]
        if quick:
            return quick[0]
    return feasible[0]


def ready_deployment(pool: PoolSnapshot) -> Mapping[str, Any] | None:
    return next((d for d in pool.deployments if d.get("state") == "ready" and d.get("purpose", "run") == "run"), None)


def missing_peers(view: PlanView, pool: PoolSnapshot) -> list[str]:
    """Peers the plan gives layers to that are not online now (they left or went stale since planning)."""
    online = {n.id for n in pool.online_peers}
    return sorted({s.node_name for s in view.stages if not s.is_local and s.layer_count and s.node_id not in online})
