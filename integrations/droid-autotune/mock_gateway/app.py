"""A stand-in for the Hypercompute gateway, for building and rehearsing autotune without touching the real hosts.

It answers the routes autotune uses with the gateway's JSON shapes and checks explicit layouts in the order and with
the refusal codes of backend/hypercompute/services/layout.py. Its memory and decode figures are MOCK PRIORS chosen to
look like gpt-oss-120b on the G14 or the GX10; they are not measurements and never appear in results. The benchmark
it serves for the starting deployment copies the measured R2/R6 medians and says so in its label.

Scenario control (no owner token, loopback only): POST /mock/join/{name}, POST /mock/leave/{name}, GET /mock.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse

GiB, MiB = 1024**3, 1024**2
LOOPBACK = {"127.0.0.1", "::1"}
OVERHEAD = int(0.45 * GiB)
COMPUTE = int(0.15 * GiB)


@dataclass(frozen=True)
class MockModel:
    id: str = "gpt-oss-120b-mxfp4"
    n_layer: int = 36
    layer_bytes: int = int(1.62 * GiB)
    dense_bytes: int = int(0.08 * GiB)  # attention and norms per layer; the rest of a layer is routed experts
    head_bytes: int = int(0.9 * GiB)
    kv_per_layer: int = 8 * MiB  # at ctx 4096
    experts: int = 128
    experts_used: int = 4


@dataclass(frozen=True)
class MockNode:
    key: str
    name: str
    devices: tuple[tuple[str, str, int], ...]  # (worker device name, description, budget bytes)
    node_budget: int
    ms_per_layer: float  # mock compute prior
    host_ram_free: int = 0  # host only: RAM for the page cache of file-backed experts
    read_bytes_per_s: float = 1.2 * GiB  # host only: SSD page-fault rate prior

    @property
    def node_id(self) -> str:
        return hashlib.sha256(self.key.encode()).hexdigest()[:32]


HOSTS = {
    "g14": MockNode("host-g14", "G14 (RTX 5070 Ti laptop)", (("CUDA0", "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                                                              int(10.9 * GiB)),), int(10.9 * GiB), 2.2,
                    host_ram_free=17 * GiB),
    "gx10": MockNode("host-gx10", "gx10-49d1 (ASUS GX10, GB10)", (("CUDA0", "NVIDIA GB10", 84886034022),),
                     84886034022, 0.45, host_ram_free=30 * GiB),
}
MAC = (("MTL0", "Apple M4 (Metal)", int(10.7 * GiB)), ("CPU", "Apple M4 (CPU)", int(11.2 * GiB)))
PEERS = {
    "mac-a": MockNode("peer-mac-a", "mac-a", MAC, int(11.2 * GiB), 2.0),
    "mac-azim": MockNode("peer-mac-azim", "mac-azim", MAC, int(11.2 * GiB), 2.0),
    "mac-b": MockNode("peer-mac-b", "mac-b", MAC, int(11.2 * GiB), 2.0),
    "g14": MockNode("peer-g14", "G14 (RTX 5070 Ti laptop)", (("CUDA0", "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                                                              int(10.9 * GiB)),), int(10.9 * GiB), 1.6),
}
HOP_MS = 20.0  # mock prior per machine boundary a token crosses
ROUTING_SKEW = 0.27  # mock prior: share of a miss-rate-weighted expert read that really reaches the SSD


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fmt(n: int) -> str:
    return f"{n / GiB:.1f} GiB"


class Refusal(Exception):
    def __init__(self, status: int, code: str, detail: str, fix: str | None = None) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail, self.fix = status, code, detail, fix


@dataclass
class Joined:
    node: MockNode
    joined_at: float
    left: bool = False


@dataclass
class MockState:
    host_kind: str = "g14"
    verify_s: float = 2.0
    replace_delay_s: float = 4.0
    model: MockModel = field(default_factory=MockModel)
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    peers: dict[str, Joined] = field(default_factory=dict)
    plans: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)
    deployments: list[dict[str, Any]] = field(default_factory=list)
    benchmarks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def host(self) -> MockNode:
        return HOSTS[self.host_kind]

    def eligible(self) -> dict[str, MockNode]:
        now = time.monotonic()
        return {j.node.node_id: j.node for j in self.peers.values()
                if not j.left and now - j.joined_at >= self.verify_s}

    # ---------------------------------------------------------------- JSON views
    def node_json(self, node: MockNode, *, local: bool, state: str) -> dict[str, Any]:
        devices = [{"id": d, "kind": "cpu" if d == "CPU" else "gpu", "name": desc, "pool_id": "uma" if local else "ram",
                    "backend": "cpu" if d == "CPU" else ("metal" if d.startswith("MTL") else "cuda"),
                    "total_bytes": b + 2 * GiB, "free_bytes": b + GiB, "source": "mock"} for d, desc, b in node.devices]
        return {
            "id": "local" if local else node.node_id, "display_name": node.name, "is_local": local,
            "session_id": None if local else "mock-session", "state": state, "state_reason": None,
            "joined_at": _now(), "last_heartbeat_at": _now(), "heartbeat_age_s": 1.0,
            "inventory": {"hostname": node.name, "devices": devices, "memory_pools": [], "network": [],
                          "disk_free_bytes": 200 * GiB, "collected_at": _now(), "engine": {"installed": True}},
            "contribution": {"devices": [d for d, _, _ in node.devices], "max_bytes": node.node_budget,
                             "reserve_bytes": GiB, "enforcement": "soft_admission", "cache_enabled": True},
            "worker": None if local else {"host": "127.0.0.1", "port": 50052},
            "worker_state": None if local else "running",
            "verification": None if local else {
                "checked_at": _now(), "reachable": True, "engine_match": True, "detail": "mock probe",
                "rpc_devices": [{"name": f"RPC{i}", "description": desc, "total_bytes": b + GiB, "free_bytes": b}
                                for i, (_, desc, b) in enumerate(node.devices)]},
            "deployments": []}

    def nodes(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        out = [self.node_json(self.host, local=True, state="eligible")]
        for j in self.peers.values():
            state = "left" if j.left else ("verifying" if now - j.joined_at < self.verify_s else "eligible")
            out.append(self.node_json(j.node, local=False, state=state))
        return out

    def network(self) -> dict[str, Any]:
        peers = [{"node_id": nid, "peer_id": nid[:16], "connected": True,
                  "path": {"mode": "direct", "rtt_ms": 6.0, "paths": 2}, "rpc": {}} for nid in self.eligible()]
        return {"transport": "iroh", "endpoint_id": "mock-endpoint", "listening": True, "connections": len(peers),
                "peers": peers}

    # ---------------------------------------------------------------- the layout check (services/layout.py order)
    def plan(self, body: dict[str, Any]) -> dict[str, Any]:
        m = self.model
        if body.get("model_id") != m.id:
            raise Refusal(404, "unknown_model", f"model {body.get('model_id')!r} is not in this mock's catalog "
                          f"(it has {m.id})")
        stages = body.get("layout")
        if not isinstance(stages, list) or not stages:
            raise Refusal(422, "mock_layout_only", "this mock gateway only checks explicit layouts")
        for i, s in enumerate(stages):
            valid = (isinstance(s, dict) and isinstance(s.get("node_id"), str) and isinstance(s.get("device"), str)
                     and isinstance(s.get("layers"), int) and s["layers"] >= 1)
            if not valid:
                raise Refusal(422, "invalid_request", f"layout stage {i}: node_id, device and layers >= 1")
        total = sum(s["layers"] for s in stages)
        if total != m.n_layer:
            raise Refusal(422, "layout_layer_count", f"the stages hold {total} layer(s) and the model has "
                          f"{m.n_layer}; every layer goes to exactly one stage")
        last = stages[-1]
        if last["node_id"] != "local":
            raise Refusal(422, "layout_order", "the last stage must be this machine's GPU (node_id 'local'), which "
                          f"holds the output head, not {last['node_id']}:{last['device']}")
        if any(s["node_id"] == "local" for s in stages[:-1]):
            raise Refusal(422, "layout_order", "only the last stage is on this machine: peers come first and this "
                          "machine's GPU last, as llama.cpp orders them")
        host_dev = next((d for d in self.host.devices if d[0].upper() == last["device"].upper()), None)
        if host_dev is None:
            raise Refusal(422, "layout_device", f"this machine has no contributed accelerator {last['device']} (it "
                          f"has {', '.join(d[0] for d in self.host.devices)}); the last stage is its GPU")
        eligible = self.eligible()
        peer_rows, layer, last_index, endpoints, hop_chain = [], 0, {}, [], []
        for s in stages[:-1]:
            node = eligible.get(s["node_id"])
            if node is None:
                raise Refusal(422, "layout_node", f"{s['node_id']} is not an eligible peer of this request (joined, "
                              "verified, within node_ids, and a strategy other than local_only)",
                              fix="GET /api/nodes lists the peers with their ids and states")
            names = [d[0] for d in node.devices]
            idx = next((i for i, n in enumerate(names) if n.upper() == s["device"].upper()), None)
            if idx is None:
                raise Refusal(422, "layout_device", f"{node.name} ({node.node_id[:8]}) has no verified RPC device "
                              f"{s['device']}; its worker runs -d {','.join(names)}")
            if idx <= last_index.get(node.node_id, -1):
                raise Refusal(422, "layout_order", f"{node.name}'s stages must follow its worker's -d order "
                              f"({','.join(names)}), each device once")
            last_index[node.node_id] = idx
            if s.get("residency") == "file_backed":
                raise Refusal(422, "hcpeer_not_reported", f"{node.name}'s CPU stage cannot be file-backed: it is not "
                              "a managed peer (join --managed), so the host sees neither its cache nor hcpeer")
            endpoint = next((e[0] for e in endpoints if e[1] == node.node_id), None)
            if endpoint is None:
                endpoint = f"10.0.0.{len(endpoints) + 10}:50052"
                endpoints.append((endpoint, node.node_id, len(names)))
            layers = list(range(layer, layer + s["layers"]))
            layer += s["layers"]
            kv = m.kv_per_layer * len(layers)
            weight = m.layer_bytes * len(layers)
            budget = node.devices[idx][2]
            row = {"node_id": node.node_id, "device": f"{endpoint}#{idx}", "endpoint": endpoint, "pool_id": "ram",
                   "layers": layers, "holds_output": False, "weight_bytes": weight, "kv_bytes": kv,
                   "compute_bytes_est": COMPUTE, "overhead_bytes_est": OVERHEAD,
                   "total_bytes_est": weight + kv + COMPUTE + OVERHEAD, "budget_bytes": budget, "file_backed": False}
            row["fits"] = row["total_bytes_est"] <= budget
            if not row["fits"]:
                raise Refusal(422, "layout_memory", f"{node.name} {names[idx]} needs ~{_fmt(row['total_bytes_est'])} "
                              f"of RAM for {len(layers)} layer(s) (weights, KV and buffers) and allows "
                              f"{_fmt(budget)} (mock contribution limit)", fix="give that stage fewer layers")
            peer_rows.append(row)
            hop_chain.append(node.node_id)
        for nid in {r["node_id"] for r in peer_rows}:
            ram = sum(r["total_bytes_est"] for r in peer_rows if r["node_id"] == nid)
            node = eligible[nid]
            if ram > node.node_budget:
                raise Refusal(422, "layout_memory", f"{node.name}'s stages need ~{_fmt(ram)} of RAM together and it "
                              f"contributes {_fmt(node.node_budget)}", fix="give its stages fewer layers")
        return self._finish(body, stages, peer_rows, layer, host_dev, endpoints, hop_chain)

    def _finish(self, body: dict[str, Any], stages: list[dict[str, Any]], peer_rows: list[dict[str, Any]],
                first_host: int, host_dev: tuple[str, str, int], endpoints: list[tuple[str, str, int]],
                hop_chain: list[str]) -> dict[str, Any]:
        m, host = self.model, self.host
        host_layers = list(range(first_host, m.n_layer))
        kv = m.kv_per_layer * len(host_layers)
        whole = m.layer_bytes * len(host_layers) + m.head_bytes
        budget = host_dev[2]
        moe, tier, warnings = None, "gpu", []
        if whole + kv + COMPUTE + OVERHEAD <= budget:
            weight = whole
        else:
            dense = m.dense_bytes * len(host_layers) + m.head_bytes + kv + COMPUTE + OVERHEAD
            if dense > budget:
                raise Refusal(422, "layout_memory", f"this machine's side does not fit: the dense parts of its "
                              f"{len(host_layers)} layer(s) need ~{_fmt(dense)} of {_fmt(budget)}",
                              fix="move layers to the peers")
            expert = m.layer_bytes - m.dense_bytes
            on_gpu = min(len(host_layers), (budget - dense) // expert)
            cpu_layers = host_layers[: len(host_layers) - on_gpu]
            cpu_bytes = expert * len(cpu_layers)
            miss = max(0.0, 1 - host.host_ram_free / cpu_bytes) if cpu_bytes else 0.0
            frac = m.experts_used / m.experts
            ssd = int(frac * ROUTING_SKEW * cpu_bytes * miss)
            peers_ms = sum(len(r["layers"]) for r in peer_rows) * 2.0
            decode = (len(host_layers) * host.ms_per_layer + peers_ms + ssd / host.read_bytes_per_s * 1000
                      + HOP_MS * len(hop_chain_boundaries(hop_chain)))
            weight = dense - kv - COMPUTE - OVERHEAD + expert * on_gpu
            tier = "moe_offload"
            moe = {"expert_count": m.experts, "expert_used_count": m.experts_used, "active_fraction": frac,
                   "cpu_expert_layers": cpu_layers, "cpu_expert_bytes": cpu_bytes, "gpu_expert_bytes": expert * on_gpu,
                   "override": "mock", "page_cache_bytes_est": host.host_ram_free, "miss_rate_est": round(miss, 3),
                   "ssd_bytes_per_token_est": ssd, "read_bytes_per_s": host.read_bytes_per_s,
                   "decode_ms_est": round(decode, 1), "provenance": "estimate"}
            warnings.append("mock gateway: moe_offload figures are mock priors, not the real planner's")
        host_row = {"node_id": "local", "device": host_dev[0], "endpoint": None, "pool_id": "uma",
                    "layers": host_layers, "holds_output": True, "weight_bytes": weight, "kv_bytes": kv,
                    "compute_bytes_est": COMPUTE, "overhead_bytes_est": OVERHEAD,
                    "total_bytes_est": weight + kv + COMPUTE + OVERHEAD, "budget_bytes": budget, "fits": True,
                    "file_backed": False}
        rpc_names, k = {}, 0
        for endpoint, _, count in endpoints:
            rpc_names[endpoint] = [f"RPC{k + i}" for i in range(count)]
            k += count
        for r in peer_rows:
            r["device"] = rpc_names[r["endpoint"]][int(r["device"].rsplit("#", 1)[1])]
        assignments = [*peer_rows, host_row]
        names = {n.node_id: n.name for n in self.eligible().values()}
        chain = [f"{'local' if s['node_id'] == 'local' else names.get(s['node_id'], s['node_id'][:8])} "
                 f"{s['device']} {s['layers']}" for s in stages]
        plan_id = uuid.uuid4().hex
        plan = {
            "id": plan_id, "model_id": m.id, "strategy": "all_selected", "ctx_size": int(body.get("ctx_size") or 4096),
            "feasible": True, "fit_label": "estimated_fit", "heuristic": True, "assignments": assignments,
            "device_list": [a["device"] for a in assignments], "tensor_split": [len(a["layers"]) for a in assignments],
            "rpc_endpoints": [e[0] for e in endpoints], "cpu_resident_bytes": int(0.6 * GiB),
            "reasons": [f"explicit layout as requested ({', '.join(chain)}); checked against the memory free now, "
                        "not chosen by the planner"],
            "warnings": warnings, "candidates": [{"label": "explicit layout", "devices": [
                f"{a['node_id'][:8]}:{a['device']}" for a in assignments], "feasible": True,
                "reason": "as requested; checked against the budgets free now (estimates)"}],
            "created_at": _now(), "engine": "llama_rpc", "tier": tier, "cpu_layers": 0, "moe": moe, "prediction": None,
            "revision": 0, "layout": stages, "capacity_mode": False, "stream": None}
        self.plans[plan_id] = (time.monotonic(), plan)
        return plan

    def start(self) -> None:
        plan = self.plan({"model_id": self.model.id, "layout": [{"node_id": "local", "device": "CUDA0",
                                                                 "layers": self.model.n_layer}]})
        dep_id = uuid.uuid4().hex
        self.deployments = [{"id": dep_id, "plan": plan, "state": "ready", "history": [], "created_at": _now(),
                             "purpose": "run", "revision": 0, "replaces": None}]
        decode, prefill = (3.296, 3.68) if self.host_kind == "g14" else (56.17, 382.0)
        self.benchmarks = [{"id": uuid.uuid4().hex, "deployment_id": dep_id, "status": "done", "created_at": _now(),
                            "label": "mock", "placement_label": f"mock copy of the measured {self.host.name} alone "
                            f"median ({'R2' if self.host_kind == 'g14' else 'R6'})", "summaries": [
                                {"prompt_label": "short", "decode_tokens_per_s": {"n": 9, "median": decode},
                                 "prompt_tokens_per_s": {"n": 9, "median": prefill}}]}]


def hop_chain_boundaries(chain: list[str]) -> list[int]:
    """Machine boundaries along peers then the host: one per change of node, plus the hand-off to the host."""
    full = [*chain, "local"]
    return [i for i in range(1, len(full)) if full[i] != full[i - 1]]


def create_app(host_kind: str = "g14", *, verify_s: float = 2.0, replace_delay_s: float = 4.0,
               peers: list[str] | None = None) -> FastAPI:
    if host_kind not in HOSTS:
        raise ValueError(f"host_kind is one of {sorted(HOSTS)}")
    state = MockState(host_kind=host_kind, verify_s=verify_s, replace_delay_s=replace_delay_s)
    for name in peers or []:
        state.peers[name] = Joined(PEERS[name], time.monotonic() - verify_s)
    state.start()
    app = FastAPI(title="Hypercompute mock gateway")
    app.state.mock = state

    def loopback(request: Request) -> None:
        if (request.client.host if request.client else "") not in LOOPBACK:
            raise Refusal(403, "owner_only", "owner endpoints are only available from this machine")

    def owner(request: Request, token: str | None) -> None:
        loopback(request)
        if not token or not hmac.compare_digest(token, state.token):
            raise Refusal(403, "owner_only", "missing or wrong owner token; reload the dashboard")

    @app.exception_handler(Refusal)
    async def _refusal(_: Request, exc: Refusal) -> JSONResponse:
        body = {"detail": exc.detail, "code": exc.code}
        if exc.fix:
            body["fix"] = exc.fix
        return JSONResponse(status_code=exc.status, content=body)

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "version": "mock", "time": _now(), "mock": True}

    @app.get("/api/owner/bootstrap")
    async def bootstrap(request: Request) -> dict:
        loopback(request)
        return {"owner_token": state.token, "version": "mock", "gateway_port": 0, "lan_addresses": []}

    @app.get("/api/nodes")
    async def nodes(request: Request) -> list:
        loopback(request)
        return state.nodes()

    @app.get("/api/network")
    async def network(request: Request) -> dict:
        loopback(request)
        return state.network()

    @app.get("/api/deployments")
    async def deployments(request: Request) -> list:
        loopback(request)
        return state.deployments

    @app.get("/api/benchmarks")
    async def benchmarks(request: Request) -> list:
        loopback(request)
        return state.benchmarks

    @app.get("/api/tune")
    async def tunes(request: Request) -> list:
        loopback(request)
        return []

    @app.post("/api/plans")
    async def plan(request: Request, x_hypercompute_owner: Annotated[str | None, Header()] = None) -> dict:
        owner(request, x_hypercompute_owner)
        return state.plan(await request.json())

    @app.post("/api/deployments/{dep_id}/replace")
    async def replace(dep_id: str, request: Request,
                      x_hypercompute_owner: Annotated[str | None, Header()] = None) -> dict:
        owner(request, x_hypercompute_owner)
        body = await request.json()
        stored = state.plans.get(str(body.get("plan_id")))
        if stored is None:
            raise Refusal(404, "plan_not_found", "plan not found; check the plan again")
        if time.monotonic() - stored[0] > 600:
            raise Refusal(409, "plan_stale", "plan is older than 10 minutes; check the plan again")
        old = next((d for d in state.deployments if d["id"] == dep_id and d["state"] == "ready"), None)
        if old is None:
            raise Refusal(409, "not_ready", "only a ready deployment can be replaced")
        old["state"] = "draining"
        await asyncio.sleep(state.replace_delay_s)
        old["state"] = "stopped"
        new = {"id": uuid.uuid4().hex, "plan": stored[1], "state": "ready", "history": [], "created_at": _now(),
               "purpose": "run", "revision": old["revision"] + 1, "replaces": dep_id}
        state.deployments.append(new)
        return new

    @app.post("/mock/join/{name}")
    async def join(name: str, request: Request) -> dict:
        loopback(request)
        if name not in PEERS or (state.host_kind == "g14" and name == "g14"):
            raise Refusal(404, "unknown_peer", f"mock peers: {sorted(PEERS)}")
        state.peers[name] = Joined(PEERS[name], time.monotonic())
        return {"joined": name, "node_id": PEERS[name].node_id, "eligible_in_s": state.verify_s}

    @app.post("/mock/leave/{name}")
    async def leave(name: str, request: Request) -> dict:
        loopback(request)
        joined = state.peers.get(name)
        if joined is None:
            raise Refusal(404, "unknown_peer", f"{name} has not joined")
        joined.left = True
        return {"left": name}

    @app.get("/mock", response_class=HTMLResponse)
    async def control(request: Request) -> str:
        loopback(request)
        names = [n for n in PEERS if not (state.host_kind == "g14" and n == "g14")]
        rows = "".join(f'<p><b>{n}</b> <button onclick="go(\'join\',\'{n}\')">join</button> '
                       f'<button onclick="go(\'leave\',\'{n}\')">leave</button></p>' for n in names)
        return (f"<!doctype html><meta charset=utf-8><title>Mock gateway</title><body style='font-family:system-ui;"
                f"background:#111;color:#eee;padding:16px'><h2>Mock gateway ({state.host.name})</h2>"
                "<p>Mock priors only. Joins turn eligible after the verify delay.</p>"
                f"{rows}<pre id=out></pre><script>async function go(a,n){{const r=await fetch(`/mock/${{a}}/${{n}}`,"
                "{method:'POST'});document.getElementById('out').textContent=await r.text()}</script>")

    return app
