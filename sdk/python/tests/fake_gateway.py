"""A small in-memory gateway for httpx.MockTransport. Shapes follow backend/hypercompute/app.py and schemas.py."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from hypercompute_sdk import Pool

FIRST_TOKEN = "tok-first-0123456789abcdefghijklmnopqrstuv"  # noqa: S105 - a fake owner token
SECOND_TOKEN = "tok-second-0123456789abcdefghijklmnopqrstu"  # noqa: S105
MODEL = "qwen2.5-7b-instruct-q4_k_m"
PLAN_KEYS = {"model_id", "node_ids", "strategy", "ctx_size", "allow_ram_tier", "allow_capacity_mode", "engine",
             "objective", "use_profiles", "overflow", "layout", "parallel"}
TERMINAL = {"stopped", "failed"}


def error(status: int, code: str, detail: str, fix: str | None = None) -> httpx.Response:
    body = {"detail": detail, "code": code}
    if fix:
        body["fix"] = fix
    return httpx.Response(status, json=body)


def sse(*payloads: Any) -> list[bytes]:
    out = []
    for p in payloads:
        out.append(b": keep-alive\n\n" if p is None else f"data: {p if isinstance(p, str) else json.dumps(p)}\n\n"
                   .encode())
    return out


def delta(content: str = "", reasoning: str | None = None, finish: str | None = None, **top: Any) -> dict:
    d: dict[str, Any] = {"content": content}
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}], "object": "chat.completion.chunk", **top}


TIMINGS = {"prompt_n": 5, "prompt_ms": 40.0, "predicted_n": 3, "predicted_ms": 120.0, "predicted_per_second": 25.0}
STREAMS: dict[str, list[bytes]] = {
    "normal": sse(None, delta(reasoning="think"), delta("Hel"), delta("lo"),
                  delta("", finish="stop", timings=TIMINGS), "[DONE]"),
    "interrupted": sse(delta("Hel"), {"error": {"code": "interrupted", "type": "upstream_error",
                                                "message": "deployment stopped", "run_id": "run1",
                                                "partial_text": "Hel"}}),
    "busy": sse({"error": {"code": "busy", "message": "the engine is busy"}}),
    "cut": sse(delta("Hel"), delta("lo")),
    "finish_no_done": sse(delta("Hi", finish="length")),
    "not_json": sse("{oops"),
    "not_object": sse("[1, 2]", delta("ok", finish="stop"), "[DONE]"),
}


class TrackedStream(httpx.SyncByteStream):
    """A response body that records how far it was read and whether it was closed."""

    def __init__(self, chunks: list[bytes], fail_after: int | None = None) -> None:
        self.chunks, self.fail_after, self.sent, self.closed = chunks, fail_after, 0, False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            if self.fail_after is not None and self.sent >= self.fail_after:
                raise httpx.ReadTimeout("upstream went quiet")
            self.sent += 1
            # split each event to test fragmentation
            yield chunk[:3]
            yield chunk[3:]

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeClock:
    t: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@dataclass
class FakeGateway:
    token: str = FIRST_TOKEN
    loopback: bool = True
    ready_after: int = 2  # polls spent loading before ready
    fail_with: str | None = None  # the deployment fails with this error instead of getting ready
    stale_deploys: int = 0  # the next N deployment creates answer 409 plan_stale
    poll_failures: list[str] = field(default_factory=list)  # per poll, in order: "ok", "connect" or a status
    chat_mode: str = "normal"
    stream_fail_after: int | None = None
    requests: list[httpx.Request] = field(default_factory=list)
    bootstraps: int = 0
    plans: dict[str, dict] = field(default_factory=dict)
    deps: dict[str, dict] = field(default_factory=dict)
    streams: list[TrackedStream] = field(default_factory=list)
    overrides: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response]] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def rotate(self) -> None:
        """Imitate a host restart: a new owner token."""
        self.token = SECOND_TOKEN

    def pool(self, **kwargs: Any) -> Pool:
        return Pool(transport=httpx.MockTransport(self), **kwargs)

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def add_dep(self, state: str, **extra: Any) -> dict:
        dep = {"id": uuid.uuid4().hex, "plan": self._plan({"model_id": MODEL}), "state": state,
               "history": [{"state": state, "at": "2026-10-24T00:00:00Z"}], "error": None, "progress": None,
               "created_at": "2026-10-24T00:00:00Z", "idempotency_key": None, "purpose": "run", "polls": 0, **extra}
        self.deps[dep["id"]] = dep
        return dep

    def _plan(self, req: dict) -> dict:
        plan = {"id": uuid.uuid4().hex, "model_id": req["model_id"], "strategy": req.get("strategy", "all_selected"),
                "ctx_size": req.get("ctx_size", 4096), "feasible": True, "fit_label": "estimated_fit",
                "heuristic": True, "assignments": [], "device_list": ["CUDA0", "RPC0"], "tensor_split": [20, 20],
                "rpc_endpoints": ["127.0.0.1:47052"], "cpu_resident_bytes": 0, "reasons": [], "warnings": [],
                "candidates": [], "created_at": "2026-10-24T00:00:00Z", "engine": "llama_rpc", "tier": "gpu"}
        self.plans[plan["id"]] = plan
        return plan

    def _owner_read(self) -> httpx.Response | None:
        if not self.loopback:
            return error(403, "owner_only", "owner endpoints are only available from this machine (http://127.0.0.1)")
        return None

    def _owner_write(self, request: httpx.Request) -> httpx.Response | None:
        denied = self._owner_read()
        if denied is None and request.headers.get("x-hypercompute-owner") != self.token:
            denied = error(403, "owner_only", "missing or wrong owner token; reload the dashboard")
        return denied

    def _active(self) -> dict | None:
        return next((d for d in self.deps.values() if d["state"] not in TERMINAL), None)

    @staticmethod
    def _public(dep: dict) -> dict:
        return {k: v for k, v in dep.items() if k != "polls"}

    # ------------------------------------------------------------------ dispatch
    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key in self.overrides:
            return self.overrides[key](request)
        if request.url.path.startswith("/api/deployments/"):
            return self._deployment_route(request, request.url.path.rsplit("/", 1)[1])
        handler = {
            ("GET", "/api/health"): self._health,
            ("GET", "/api/owner/bootstrap"): self._bootstrap,
            ("GET", "/api/network"): self._network,
            ("GET", "/api/nodes"): self._nodes,
            ("GET", "/api/peers"): lambda r: self._owner_read() or httpx.Response(200, json=[{"node_id": "n1"}]),
            ("GET", "/api/models"): self._models,
            ("GET", "/api/capacity/pools"): lambda r: self._owner_read() or httpx.Response(200, json={"pools": []}),
            ("GET", "/api/capacity/explain"): self._explain,
            ("POST", "/api/plans"): self._plan_route,
            ("POST", "/api/deployments"): self._create,
            ("GET", "/api/deployments"): self._list,
            ("GET", "/v1/models"): self._v1_models,
            ("POST", "/v1/chat/completions"): self._chat,
        }.get(key)
        return handler(request) if handler else error(404, "not_found", "Not Found")

    def _health(self, request: httpx.Request) -> httpx.Response:
        body = {"status": "ok", "version": "0.1.0", "time": "2026-10-24T00:00:00Z",
                "engine": {"release": "b6000", "rpc_proto": "3.0.0", "variant": "cuda"}, "engines": []}
        return httpx.Response(200, json=body)

    def _bootstrap(self, request: httpx.Request) -> httpx.Response:
        denied = self._owner_read()
        if denied:
            return denied
        self.bootstraps += 1
        return httpx.Response(200, json={"owner_token": self.token, "version": "0.1.0", "gateway_port": 8001,
                                         "lan_addresses": ["192.168.1.20"]})

    def _network(self, request: httpx.Request) -> httpx.Response:
        peer = {"node_id": "n1", "peer_id": "p" * 64, "connected": True,
                "path": {"mode": "direct", "remote_id": "r" * 64, "rtt_ms": 3.0}, "rpc": {}}
        return self._owner_read() or httpx.Response(200, json={
            "transport": "iroh", "endpoint_id": "e" * 64, "listening": True, "connections": 1,
            "binding_speedups": {}, "peers": [peer]})

    def _nodes(self, request: httpx.Request) -> httpx.Response:
        nodes = [{"id": "local", "display_name": "gx10", "is_local": True, "state": "ready", "heartbeat_age_s": None,
                  "contribution": {"devices": ["CUDA0"], "max_bytes": 100, "reserve_bytes": 0}},
                 {"id": "n1", "display_name": "mac-a", "is_local": False, "state": "ready", "heartbeat_age_s": 1.5,
                  "contribution": {"devices": ["MTL0"], "max_bytes": 50, "reserve_bytes": 4}}]
        return self._owner_read() or httpx.Response(200, json=nodes)

    def _models(self, request: httpx.Request) -> httpx.Response:
        views = [{"artifact": {"id": "big-model"}, "status": "not_downloaded"},
                 {"artifact": {"id": MODEL}, "status": "downloaded"}]
        return self._owner_read() or httpx.Response(200, json=views)

    def _explain(self, request: httpx.Request) -> httpx.Response:
        return self._owner_read() or httpx.Response(200, json={"params": dict(request.url.params)})

    def _v1_models(self, request: httpx.Request) -> httpx.Response:
        dep = self._active()
        data = ([{"id": dep["plan"]["model_id"], "object": "model", "owned_by": "hypercompute",
                  "deployment_id": dep["id"]}] if dep and dep["state"] == "ready" else [])
        return self._owner_read() or httpx.Response(200, json={"object": "list", "data": data})

    def _plan_route(self, request: httpx.Request) -> httpx.Response:
        denied = self._owner_write(request)
        if denied:
            return denied
        req = json.loads(request.content)
        extra = sorted(set(req) - PLAN_KEYS)
        if extra:
            return httpx.Response(422, json={"detail": [{"loc": ["body", k], "msg": "Extra inputs are not permitted",
                                                         "type": "extra_forbidden"} for k in extra]})
        return httpx.Response(200, json=self._plan(req))

    def _create(self, request: httpx.Request) -> httpx.Response:
        denied = self._owner_write(request)
        if denied:
            return denied
        key, req = request.headers.get("idempotency-key"), json.loads(request.content)
        same = next((d for d in self.deps.values() if key and d["idempotency_key"] == key), None)
        if same:
            return httpx.Response(202, json=self._public(same))
        if self._active():
            return error(409, "deployment_active", "deployment 01234567 is ready; stop it first")
        if self.stale_deploys:
            self.stale_deploys -= 1
            return error(409, "plan_stale", "plan is older than 10 minutes; check the plan again")
        plan = self.plans.get(req.get("plan_id")) if "plan_id" in req else self._plan(req["plan"])
        if plan is None:
            return error(404, "plan_not_found", "plan not found; check the plan again")
        dep = self.add_dep("planned", plan=plan, idempotency_key=key)
        return httpx.Response(202, json=self._public(dep))

    def _list(self, request: httpx.Request) -> httpx.Response:
        rows = [self._public(d) for d in reversed(self.deps.values())]
        return self._owner_read() or httpx.Response(200, json=rows)

    def _deployment_route(self, request: httpx.Request, dep_id: str) -> httpx.Response:
        dep = self.deps.get(dep_id)
        if request.method == "DELETE":
            denied = self._owner_write(request)
            if denied:
                return denied
            if dep is None:
                return error(404, "not_found", "no such deployment")
            dep["state"] = "stopped"
            return httpx.Response(200, json=self._public(dep))
        failure = self.poll_failures.pop(0) if self.poll_failures else "ok"
        if failure == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if failure != "ok":
            return error(int(failure), "no_ready_deployment", "busy")
        if dep is None:
            return error(404, "not_found", "no such deployment")
        self._advance(dep)
        return httpx.Response(200, json=self._public(dep))

    def _advance(self, dep: dict) -> None:
        if dep["state"] in TERMINAL or dep["state"] == "ready":
            return
        dep["polls"] += 1
        if dep["polls"] <= self.ready_after:
            dep["state"], dep["progress"] = "loading", f"loading {dep['polls']}/{self.ready_after}"
        elif self.fail_with:
            dep["state"], dep["error"] = "failed", self.fail_with
        else:
            dep["state"], dep["progress"] = "ready", None
        dep["history"].append({"state": dep["state"], "at": "2026-10-24T00:00:01Z"})

    def _chat(self, request: httpx.Request) -> httpx.Response:
        denied = self._owner_write(request)
        if denied:
            return denied
        body = json.loads(request.content)
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            return error(400, "bad_request", "'messages' must be a non-empty list")
        dep = self._active()
        if dep is None or dep["state"] != "ready":
            return error(503, "no_ready_deployment", "no model is ready",
                         "choose a model and click 'Run together' in the dashboard")
        if body.get("model") not in (None, "") and body["model"] != dep["plan"]["model_id"]:
            return error(404, "model_not_found", f"model {body['model']!r} is not loaded",
                         "omit 'model' or use the ready model listed by /v1/models")
        if body.get("stream"):
            stream = TrackedStream(STREAMS[self.chat_mode], self.stream_fail_after)
            self.streams.append(stream)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        if self.chat_mode == "interrupted":
            return httpx.Response(502, json={"error": {"code": "interrupted", "type": "upstream_error",
                                                       "message": "deployment stopped", "run_id": "run1",
                                                       "partial_text": "Hel", "partial_truncated": True}})
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "model": dep["plan"]["model_id"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}, "timings": TIMINGS})
