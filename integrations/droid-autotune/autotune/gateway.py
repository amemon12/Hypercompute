"""HTTP client for the Hypercompute gateway (backend/hypercompute/app.py).

Owner endpoints answer only on loopback, so this service runs on the gateway host. Two views of the client exist on
purpose: PlanOnlyGateway is all the Droid tools ever receive (reads plus the plan-only check), and only the owner's
approve route in web.py holds a GatewayClient that can call replace.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

OWNER_HEADER = "X-Hypercompute-Owner"
# The gateway's own load budget tops out at 4 h (adapters/llama_rpc.py); replace waits for it, so this client waits
# a little longer and a timeout here means "unknown", never "failed".
REPLACE_TIMEOUT_S = 4.5 * 3600


class GatewayError(RuntimeError):
    def __init__(self, status: int, code: str, detail: str, fix: str | None = None) -> None:
        super().__init__(f"{status} {code}: {detail}")
        self.status, self.code, self.detail, self.fix = status, code, detail, fix


@dataclass(frozen=True)
class PlanOutcome:
    """POST /api/plans: a stored plan (accepted or infeasible) or the planner's refusal with its reason."""

    status: int
    plan: dict[str, Any] | None
    code: str | None = None
    detail: str = ""
    fix: str | None = None


def require_loopback(url: str) -> str:
    host = urlparse(url).hostname or ""
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(f"the gateway URL must be loopback (owner endpoints answer only on 127.0.0.1): {url}")
    return url.rstrip("/")


def _error(resp: httpx.Response) -> GatewayError:
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    detail = body.get("detail")
    if not isinstance(detail, str):  # FastAPI validation errors carry a list
        detail = str(detail) if detail is not None else resp.text[:300]
    return GatewayError(resp.status_code, str(body.get("code") or "http_error"), detail, body.get("fix"))


class GatewayClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0, transport: httpx.AsyncBaseTransport | None = None,
                 owner_token: str | None = None) -> None:
        self.base_url = require_loopback(base_url)
        # trust_env=False: never route the owner token via a proxy from the environment
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport,
                                       trust_env=False, follow_redirects=False)
        self._token = owner_token

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str) -> Any:
        resp = await self._http.get(path)
        if resp.status_code >= 400:
            raise _error(resp)
        return resp.json()

    async def _owner_headers(self) -> dict[str, str]:
        if self._token is None:
            boot = await self._get("/api/owner/bootstrap")
            self._token = str(boot["owner_token"])
        return {OWNER_HEADER: self._token}

    async def _owner_post(self, path: str, body: dict[str, Any], timeout: float | None = None) -> httpx.Response:
        """POST with the owner token. The gateway makes a new token every process, so a 403 owner_only after a
        gateway restart re-reads it once and retries."""
        kwargs: dict[str, Any] = {"json": body, "timeout": timeout} if timeout else {"json": body}
        resp = await self._http.post(path, headers=await self._owner_headers(), **kwargs)
        if resp.status_code == 403 and _error(resp).code == "owner_only":
            self._token = None
            resp = await self._http.post(path, headers=await self._owner_headers(), **kwargs)
        return resp

    # ---------------------------------------------------------------- reads (OwnerRead on the gateway)
    async def health(self) -> dict[str, Any]:
        return await self._get("/api/health")

    async def nodes(self) -> list[dict[str, Any]]:
        return await self._get("/api/nodes")

    async def network(self) -> dict[str, Any]:
        return await self._get("/api/network")

    async def deployments(self) -> list[dict[str, Any]]:
        return await self._get("/api/deployments")

    async def benchmarks(self) -> list[dict[str, Any]]:
        return await self._get("/api/benchmarks")

    async def tunes(self) -> list[dict[str, Any]]:
        try:
            return await self._get("/api/tune")
        except GatewayError as exc:
            if exc.status in (404, 503):  # a gateway without the tuner has no trials to wait for
                return []
            raise

    # ---------------------------------------------------------------- plan-only (OwnerWrite, never loads anything)
    async def check_plan(self, body: dict[str, Any]) -> PlanOutcome:
        """DeploymentManager.plan: checks an explicit layout and stores the plan row; it starts no engine."""
        resp = await self._owner_post("/api/plans", body)
        if resp.status_code == 200:
            return PlanOutcome(200, resp.json())
        err = _error(resp)
        if 400 <= resp.status_code < 500 and resp.status_code not in (401, 403):
            return PlanOutcome(resp.status_code, None, err.code, err.detail, err.fix)
        raise err

    # ---------------------------------------------------------------- the one write, owner approval only
    async def replace(self, dep_id: str, plan_id: str) -> dict[str, Any]:
        """POST /api/deployments/{id}/replace: reloads the served model onto the plan, with the gateway's rollback."""
        resp = await self._owner_post(f"/api/deployments/{dep_id}/replace", {"plan_id": plan_id},
                                      timeout=REPLACE_TIMEOUT_S)
        if resp.status_code >= 400:
            raise _error(resp)
        return resp.json()


class PlanOnlyGateway:
    """The gateway as the Droid tools see it: reads and the plan-only check. There is no replace, deploy, stop,
    download or chat method to reach, and the owner token stays inside the wrapped client."""

    def __init__(self, client: GatewayClient) -> None:
        self._client = client

    async def nodes(self) -> list[dict[str, Any]]:
        return await self._client.nodes()

    async def network(self) -> dict[str, Any]:
        return await self._client.network()

    async def deployments(self) -> list[dict[str, Any]]:
        return await self._client.deployments()

    async def benchmarks(self) -> list[dict[str, Any]]:
        return await self._client.benchmarks()

    async def check_plan(self, body: dict[str, Any]) -> PlanOutcome:
        return await self._client.check_plan(body)
