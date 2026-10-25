"""A local stand-in for the Hypercompute gateway: the routes, shapes, owner-token rule and error bodies the agent
uses, with a simulated deployment lifecycle and switchable failure scenarios.

    .venv/Scripts/python -m mock_gateway.server --port 8765

Scenarios (POST /mock/scenario with JSON, GET /mock/state to read): offline (peer names marked stale),
fail_loads (fail the next N loads), drop_on_load (a peer that leaves while the next load runs), foreign (a model
someone else has loaded or is loading), reasoning_only (the model answers with reasoning and no content),
load_s and chat_delay_s (simulated time).
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .fixtures import GIB, HOST, MODELS, PEERS

TOKEN = "mock-owner-token-" + uuid.uuid4().hex
TERMINAL = {"stopped", "failed"}


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _error(status: int, code: str, detail: str, fix: str | None = None) -> JSONResponse:
    body = {"detail": detail, "code": code}
    if fix:
        body["fix"] = fix
    return JSONResponse(status_code=status, content=body)


@dataclass
class MockState:
    offline: set[str] = field(default_factory=set)
    fail_loads: int = 0
    drop_on_load: str | None = None
    reasoning_only: bool = False
    load_s: float = 3.0
    chat_delay_s: float = 0.0
    stop_fails: bool = False
    plans: dict[str, dict] = field(default_factory=dict)
    plan_times: dict[str, float] = field(default_factory=dict)
    deployments: dict[str, dict] = field(default_factory=dict)
    dep_meta: dict[str, dict] = field(default_factory=dict)
    chats: int = 0
    token: str = TOKEN

    # ------------------------------------------------------------------ views
    def peer_online(self, peer: dict) -> bool:
        return peer["display_name"] not in self.offline

    def node_views(self) -> list[dict]:
        now = _now()
        views = [self._node(HOST, is_local=True, online=True, now=now)]
        views += [self._node(p, is_local=False, online=self.peer_online(p), now=now) for p in PEERS]
        return views

    def _node(self, n: dict, *, is_local: bool, online: bool, now: datetime) -> dict:
        age = 0.9 if online else 95.0
        return {
            "id": n["id"], "display_name": n["display_name"], "is_local": is_local, "session_id": None if is_local
            else "mock-session", "state": "eligible" if online else "stale",
            "state_reason": None if online else "no heartbeat for 95 s",
            "joined_at": _iso(now - timedelta(hours=2)), "last_heartbeat_at": _iso(now - timedelta(seconds=age)),
            "heartbeat_age_s": 0.0 if is_local else age,
            "inventory": {"hostname": n["display_name"].split(" ")[0], "memory_pools": [
                {**n["pool"], "free_bytes": n["pool"]["total_bytes"] // 2, "measured_at": _iso(now), "source": "mock"}],
                "devices": [{"id": n["device"], "kind": "gpu", "name": n["device"], "pool_id": n["pool"]["id"],
                             "source": "mock"}]},
            "contribution": {"devices": [n["device"]], "max_bytes": n["max_bytes"], "reserve_bytes": 0,
                             "enforcement": "soft_admission", "cache_enabled": not is_local},
            "worker": None, "worker_state": None if is_local else ("running" if online else "unknown"),
            "worker_cpu_seconds": None, "verification": None, "deployments": [],
        }

    def network(self) -> dict:
        peers = [{"node_id": p["id"], "peer_id": "<mock>", "connected": self.peer_online(p),
                  "path": {"remote_id": "<mock>", "mode": p.get("mode", "direct"), "rtt_ms": p["rtt_ms"], "paths": 2}
                  if self.peer_online(p) else None, "rpc": {"streams": 0, "active": 0}} for p in PEERS]
        return {"transport": "iroh", "endpoint_id": "<mock>", "listening": True,
                "connections": sum(1 for p in peers if p["connected"]), "peers": peers}

    def model_views(self) -> list[dict]:
        out = []
        for mid, name, family, params, quant, size, n_layer, moe, *_rest, role in MODELS:
            out.append({
                "artifact": {"id": mid, "display_name": name, "family": family, "params_b": params,
                             "quantization": quant, "format": "gguf", "repo": f"mock/{mid}", "revision": "mock",
                             "license": "mock", "files": [], "role": role, "size_bytes": size},
                "status": "downloaded", "local_path": f"/mock/{mid}.gguf", "downloaded_bytes": size,
                "total_bytes": size, "verified_sha256": True,
                "gguf": {"n_layer": n_layer, "expert_count": 256 if moe else None}})
        return out

    # ------------------------------------------------------------------ planning
    def plan(self, model_id: str, strategy: str, node_ids: list[str] | None, ctx_size: int) -> dict:
        spec = next((m for m in MODELS if m[0] == model_id), None)
        if spec is None:
            raise KeyError(model_id)
        _, name, _, _, _, size, n_layer, moe, speed, _, _ = spec
        online = [p for p in PEERS if self.peer_online(p)]
        if node_ids is not None:
            online = [p for p in online if p["id"] in node_ids]
        host_budget = HOST["max_bytes"]
        fits_alone = size <= host_budget
        if strategy == "local_only" or (strategy in ("auto", "fastest_fit") and fits_alone):
            online = []
        budget = host_budget + sum(p["max_bytes"] for p in online)
        layer_bytes = size // n_layer
        assignments, start = [], 0
        for i, p in enumerate(online):
            n = max(1, min(int(p["max_bytes"] * 0.9 // max(layer_bytes, 1)), n_layer - 1 - start))
            if start + n >= n_layer:
                break
            assignments.append(self._assign(p["id"], f"RPC{i}", f"127.0.0.1:{33700 + i}", range(start, start + n),
                                            False, n * layer_bytes, p["max_bytes"]))
            start += n
        host_layers = range(start, n_layer)
        gpu_bytes = min(len(host_layers) * layer_bytes, host_budget)
        tier, capacity, feasible, reasons = "gpu", False, True, [f"strategy: {strategy}"]
        warnings: list[str] = []
        if size > budget:
            if moe and size <= budget + HOST["pool"]["total_bytes"]:
                tier, capacity = "moe_offload", True
                cpu_bytes = size - gpu_bytes - sum(a["weight_bytes"] for a in assignments)
                assignments.insert(0, self._assign("local", "CPU", None, range(0), False, cpu_bytes, host_budget))
                warnings.append("capacity mode (MoE): routed experts of host layers stay memory-mapped on the CPU")
            else:
                feasible = False
                reasons.append(f"does not fit: needs {size / GIB:.0f} GiB, the selected devices offer "
                               f"{budget / GIB:.0f} GiB")
        assignments.append(self._assign("local", HOST["device"], None, host_layers, True, gpu_bytes, host_budget))
        plan_id = uuid.uuid4().hex
        prior_tok_s = speed * (0.5 if capacity else 0.9)  # the gateway's priors are cautious; measured is faster
        plan = {
            "id": plan_id, "model_id": model_id, "strategy": strategy, "ctx_size": ctx_size, "feasible": feasible,
            "fit_label": "estimated_fit" if feasible else "does_not_fit", "heuristic": True,
            "assignments": assignments, "device_list": [a["device"] for a in assignments],
            "tensor_split": [len(a["layers"]) for a in assignments], "rpc_endpoints": [a["endpoint"] for a in
                                                                                        assignments if a["endpoint"]],
            "cpu_resident_bytes": 385351680, "reasons": reasons, "warnings": warnings, "candidates": [],
            "created_at": _iso(_now()), "engine": "llama_rpc", "tier": tier, "cpu_layers": 0, "capacity_mode": capacity,
            "stream": None, "moe": {"decode_ms_est": 1000.0 / prior_tok_s} if capacity else None,
            "prediction": None if capacity else {"decode_ms": 1000.0 / prior_tok_s, "prefill_ms": 50.0,
                                                 "load_ms": 1000.0, "objective_ms": 1.0, "bottleneck": "mock"},
            "revision": 0, "layout": None, "display": name}
        self.plans[plan_id] = plan
        self.plan_times[plan_id] = time.monotonic()
        return plan

    @staticmethod
    def _assign(node_id: str, device: str, endpoint: str | None, layers: range, head: bool, weight: int,
                budget: int) -> dict:
        return {"node_id": node_id, "device": device, "endpoint": endpoint, "pool_id": "uma", "layers": list(layers),
                "holds_output": head, "weight_bytes": int(weight), "kv_bytes": 0, "compute_bytes_est": 0,
                "overhead_bytes_est": 0, "total_bytes_est": int(weight), "budget_bytes": int(budget), "fits": True,
                "file_backed": False}

    # ------------------------------------------------------------------ deployments
    def active(self) -> dict | None:
        return next((self.deployment(d) for d in list(self.deployments) if self.deployment(d)["state"] not in
                     TERMINAL), None)

    def create(self, plan: dict, key: str | None, *, foreign: bool = False, ready: bool = False) -> dict:
        dep_id = uuid.uuid4().hex
        fail = None
        if not foreign and self.drop_on_load:
            fail = f"RPC peer {self.drop_on_load} unreachable during load: peer connection lost"
            self.offline.add(self.drop_on_load)
            self.drop_on_load = None
        elif not foreign and self.fail_loads > 0:
            self.fail_loads -= 1
            fail = "llama-server exited during load (mock failure)"
        self.deployments[dep_id] = {"id": dep_id, "plan": plan, "state": "planned", "history": [], "error": None,
                                    "cleanup": None, "progress": None, "engine_args": [], "upstream_port": None,
                                    "pid": 4242, "placement": None, "participation_verified": False,
                                    "participation_detail": None, "created_at": _iso(_now()),
                                    "idempotency_key": key, "purpose": "run", "revision": 0, "replaces": None}
        self.dep_meta[dep_id] = {"t0": time.monotonic() - (self.load_s + 3600 if ready else 0), "fail": fail,
                                 "stopped": False}
        return self.deployment(dep_id)

    def deployment(self, dep_id: str) -> dict:
        dep, meta = self.deployments[dep_id], self.dep_meta[dep_id]
        if meta["stopped"]:
            return {**dep, "state": "stopped", "cleanup": "stopped by the owner"}
        elapsed = time.monotonic() - meta["t0"]
        if elapsed < self.load_s * 0.3:
            return {**dep, "state": "preparing", "progress": "starting RPC workers"}
        if elapsed < self.load_s:
            pct = int(100 * elapsed / max(self.load_s, 1e-6))
            return {**dep, "state": "loading", "progress": f"loading tensors {pct}%"}
        if meta["fail"]:
            return {**dep, "state": "failed", "error": meta["fail"]}
        plan = dep["plan"]
        devices = [{"name": f"RPC0[{a['endpoint']}]" if a["endpoint"] else ("CPU_Mapped" if a["device"] == "CPU"
                                                                            else a["device"]),
                    "model_bytes": a["weight_bytes"], "kv_bytes": 0, "compute_bytes": 0,
                    "layers_est": len(a["layers"]) or None} for a in plan["assignments"]]
        return {**dep, "state": "ready", "progress": None,
                "placement": {"offloaded": f"{sum(len(a['layers']) for a in plan['assignments'])}/"
                                           f"{sum(len(a['layers']) for a in plan['assignments'])}",
                              "devices": devices, "evidence_lines": ["mock"], "parsed_at": _iso(_now())},
                "participation_verified": True,
                "participation_detail": "every planned device reported its buffers in the load log (mock)"}

    def stop(self, dep_id: str) -> dict:
        self.dep_meta[dep_id]["stopped"] = True
        return self.deployment(dep_id)


def create_app(state: MockState | None = None) -> FastAPI:
    app = FastAPI(title="Hypercompute mock gateway")
    st = state or MockState()
    app.state.mock = st

    def owner(token: str | None) -> JSONResponse | None:
        if not token or not hmac.compare_digest(token, st.token):
            return _error(403, "owner_only", "missing or wrong owner token; reload the dashboard")
        return None

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "version": "mock", "time": _iso(_now())}

    @app.get("/api/owner/bootstrap")
    async def bootstrap(request: Request) -> Any:
        if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
            return _error(403, "owner_only", "owner endpoints are only available from this machine")
        return {"owner_token": st.token, "version": "mock", "gateway_port": 0, "lan_addresses": []}

    @app.get("/api/nodes")
    async def nodes() -> list[dict]:
        return st.node_views()

    @app.get("/api/network")
    async def network() -> dict:
        return st.network()

    @app.get("/api/models")
    async def models() -> list[dict]:
        return st.model_views()

    @app.post("/api/plans")
    async def plans(body: dict, x_hypercompute_owner: str | None = Header(default=None)) -> Any:
        if (refused := owner(x_hypercompute_owner)) is not None:
            return refused
        try:
            return st.plan(str(body.get("model_id")), str(body.get("strategy") or "all_selected"),
                           body.get("node_ids"), int(body.get("ctx_size") or 4096))
        except KeyError:
            return _error(400, "model_error", f"unknown model {body.get('model_id')!r}")

    @app.get("/api/deployments")
    async def deployments() -> list[dict]:
        return [st.deployment(d) for d in st.deployments]

    @app.get("/api/deployments/{dep_id}")
    async def deployment(dep_id: str) -> Any:
        if dep_id not in st.deployments:
            return _error(404, "not_found", f"no deployment {dep_id}")
        return st.deployment(dep_id)

    @app.post("/api/deployments", status_code=202)
    async def create(body: dict, x_hypercompute_owner: str | None = Header(default=None),
                     idempotency_key: str | None = Header(default=None)) -> Any:
        if (refused := owner(x_hypercompute_owner)) is not None:
            return refused
        plan = st.plans.get(str(body.get("plan_id")))
        if plan is None:
            return _error(404, "plan_not_found", "unknown plan", "plan again")
        if time.monotonic() - st.plan_times[plan["id"]] > 600:
            return _error(409, "plan_expired", "the plan is older than 10 minutes", "plan again")
        if not plan["feasible"]:
            return _error(409, "plan_infeasible", "the plan does not fit")
        if st.active() is not None:
            return _error(409, "deployment_active", "another deployment is active", "stop it first")
        return JSONResponse(status_code=202, content=st.create(plan, idempotency_key))

    @app.delete("/api/deployments/{dep_id}")
    async def stop(dep_id: str, x_hypercompute_owner: str | None = Header(default=None)) -> Any:
        if (refused := owner(x_hypercompute_owner)) is not None:
            return refused
        if dep_id not in st.deployments:
            return _error(404, "not_found", f"no deployment {dep_id}")
        if st.stop_fails:
            return _error(500, "engine_error", "llama-server did not exit (mock failure)")
        return st.stop(dep_id)

    @app.post("/v1/chat/completions")
    async def chat(body: dict, x_hypercompute_owner: str | None = Header(default=None)) -> Any:
        if (refused := owner(x_hypercompute_owner)) is not None:
            return refused
        active = st.active()
        if active is None or active["state"] != "ready":
            return _error(503, "no_ready_deployment", "no deployment is ready", "deploy a model first")
        spec = next(m for m in MODELS if m[0] == active["plan"]["model_id"])
        speed, prefill = spec[8], spec[9]
        messages = body.get("messages") or []
        prompt = str(next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), ""))
        max_tokens = int(body.get("max_tokens") or 256)
        n_out = min(max_tokens, 40)
        n_in = max(8, len(prompt) // 4 + 20)
        if st.chat_delay_s:
            await asyncio.sleep(st.chat_delay_s)
        st.chats += 1
        text = (f"[mock] {spec[1]} would answer “{prompt[:80]}” here. On the real host this text "
                f"comes from the pooled model.")
        message = ({"role": "assistant", "content": "", "reasoning_content": f"[mock] thinking about {prompt[:60]}"}
                   if st.reasoning_only else {"role": "assistant", "content": text})
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()),
                "model": active["plan"]["model_id"],
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "length" if st.reasoning_only else "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
                "timings": {"prompt_n": n_in, "prompt_ms": n_in / prefill * 1000, "prompt_per_second": prefill,
                            "predicted_n": n_out, "predicted_ms": (n_out - 1) / speed * 1000,
                            "predicted_per_second": speed}}

    @app.get("/mock/state")
    async def mock_state() -> dict:
        return {"offline": sorted(st.offline), "fail_loads": st.fail_loads, "drop_on_load": st.drop_on_load,
                "reasoning_only": st.reasoning_only, "load_s": st.load_s, "chats": st.chats,
                "deployments": [{k: st.deployment(d)[k] for k in ("id", "state", "error")} for d in st.deployments]}

    @app.post("/mock/scenario")
    async def scenario(body: dict) -> dict:
        if "offline" in body:
            st.offline = set(body["offline"])
        for key in ("fail_loads", "drop_on_load", "reasoning_only", "load_s", "chat_delay_s", "stop_fails"):
            if key in body:
                setattr(st, key, body[key])
        if body.get("foreign"):
            ready = body.get("foreign_state", "ready") == "ready"
            plan = st.plan(body["foreign"], "all_selected", None, 4096)
            st.create(plan, "owner", foreign=True, ready=ready)
        if body.get("reset"):
            fresh = MockState(token=st.token)
            for name in ("offline", "fail_loads", "drop_on_load", "reasoning_only", "load_s", "chat_delay_s", "stop_fails",
                         "plans", "plan_times", "deployments", "dep_meta", "chats"):
                setattr(st, name, getattr(fresh, name))
        return await mock_state()

    return app


def use_generic_names() -> None:
    """Relabel the mock pool (gpu-host, laptop-1 to laptop-4, fresh node ids) for runs whose replies anyone can see,
    such as the public agent before it is pointed at the real gateway."""
    global HOST, PEERS
    HOST = {**HOST, "display_name": "gpu-host (this machine)"}
    PEERS = [{**peer, "display_name": f"laptop-{i}", "id": uuid.uuid4().hex} for i, peer in enumerate(PEERS, 1)]


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--load-s", type=float, default=20.0, help="simulated load time")
    parser.add_argument("--chat-delay-s", type=float, default=3.0, help="simulated generation time")
    parser.add_argument("--generic-names", action="store_true", help="label machines gpu-host and laptop-N")
    args = parser.parse_args()
    if args.generic_names:
        use_generic_names()
    state = MockState(load_s=args.load_s, chat_delay_s=args.chat_delay_s)
    uvicorn.run(create_app(state), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
