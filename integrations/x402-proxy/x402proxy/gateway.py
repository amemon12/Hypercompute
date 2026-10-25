"""The Hypercompute gateway as the proxy sees it: who is serving, with which plan, and a streamed chat.

The proxy runs on the host and talks to the gateway over loopback, the only place the gateway hands out its owner
token (GET /api/owner/bootstrap). The token stays in this process's memory and is never logged or returned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType

import httpx

from .split import HOST_NODE_ID

NAMES_TTL_S = 30.0
CONNECT_TIMEOUT_S = 5.0
# A pooled 1T model can spend minutes in prefill before the first token, and the gateway bounds silence itself.
READ_TIMEOUT_S = 900.0


class GatewayError(Exception):
    """The gateway cannot serve a paid chat right now. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


@dataclass(frozen=True)
class PoolSnapshot:
    deployment_id: str
    model_id: str
    plan: MappingProxyType  # the deployment's plan as the gateway returned it
    names: MappingProxyType  # node id -> display name; HOST_NODE_ID is the host


class GatewayClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # trust_env=False: an HTTP(S)_PROXY in the environment must never see the owner token
        self._http = httpx.AsyncClient(base_url=base_url, transport=transport, follow_redirects=False, trust_env=False,
                                       timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S))
        self._token: str | None = None
        self._plans: dict[str, dict] = {}  # deployment id -> plan; a deployment's plan never changes
        self._names: tuple[float, dict[str, str]] = (0.0, {})

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str) -> object:
        try:
            r = await self._http.get(path, timeout=httpx.Timeout(15.0, connect=CONNECT_TIMEOUT_S))
        except httpx.HTTPError as exc:
            raise GatewayError(503, "gateway_unreachable", f"gateway did not answer {path}: {type(exc).__name__}") \
                from exc
        if r.status_code != 200:
            raise GatewayError(503, "gateway_error", f"gateway answered {path} with HTTP {r.status_code}")
        return r.json()

    async def owner_token(self, refresh: bool = False) -> str:
        if self._token is None or refresh:
            body = await self._get("/api/owner/bootstrap")
            token = body.get("owner_token") if isinstance(body, dict) else None
            if not isinstance(token, str) or not token:
                raise GatewayError(503, "no_owner_token", "gateway bootstrap returned no owner token; run the proxy "
                                   "on the host, next to the gateway")
            self._token = token
        return self._token

    async def _node_names(self) -> dict[str, str]:
        fetched_at, names = self._names
        if time.monotonic() - fetched_at < NAMES_TTL_S:
            return names
        nodes = await self._get("/api/nodes")
        names = {}
        for node in nodes if isinstance(nodes, list) else []:
            name = str(node.get("display_name") or node.get("id", "")[:12])
            names[str(node.get("id"))] = name
            if node.get("is_local"):
                names[HOST_NODE_ID] = name
        self._names = (time.monotonic(), names)
        return names

    async def snapshot(self) -> PoolSnapshot:
        """The served deployment, as /v1/models names it (ready and not a tuner trial), with its plan."""
        models = await self._get("/v1/models")
        data = models.get("data") if isinstance(models, dict) else None
        if not data:
            raise GatewayError(503, "no_model_ready", "no model is ready on the pool; nothing was charged")
        dep_id = str(data[0]["deployment_id"])
        plan = self._plans.get(dep_id)
        if plan is None:
            dep = await self._get(f"/api/deployments/{dep_id}")
            plan = dep.get("plan") if isinstance(dep, dict) else None
            if not isinstance(plan, dict):
                raise GatewayError(503, "no_plan", f"deployment {dep_id} has no plan")
            self._plans = {**self._plans, dep_id: plan}
        names = await self._node_names()
        return PoolSnapshot(deployment_id=dep_id, model_id=str(plan.get("model_id") or data[0]["id"]),
                            plan=MappingProxyType(plan), names=MappingProxyType(dict(names)))

    async def open_chat_stream(self, body: dict) -> httpx.Response:
        """POST the chat with stream=true; the caller reads and closes the response. Retries once on a stale token."""
        for attempt in (0, 1):
            token = await self.owner_token(refresh=attempt == 1)
            request = self._http.build_request("POST", "/v1/chat/completions", json={**body, "stream": True},
                                               headers={"X-Hypercompute-Owner": token,
                                                        "Accept": "text/event-stream"})
            try:
                response = await self._http.send(request, stream=True)
            except httpx.HTTPError as exc:
                raise GatewayError(503, "gateway_unreachable", f"chat request failed: {type(exc).__name__}") \
                    from exc
            if response.status_code != 403 or attempt == 1:
                return response
            await response.aclose()
        raise AssertionError("unreachable")
