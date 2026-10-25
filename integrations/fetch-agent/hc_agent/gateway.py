"""Async client for the Hypercompute gateway HTTP API (backend/hypercompute/app.py in the main repo).

Reads are loopback-only; writes also carry the owner token from GET /api/owner/bootstrap in the
X-Hypercompute-Owner header. The token lives only in memory and never appears in a repr, log line or reply.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx

OWNER_HEADER = "X-Hypercompute-Owner"
log = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    """A refused or failed gateway call, with the gateway's own code and fix when it sent them."""

    def __init__(self, status: int, code: str | None, detail: str, fix: str | None = None) -> None:
        super().__init__(f"HTTP {status}{' ' + code if code else ''}: {detail}")
        self.status, self.code, self.detail, self.fix = status, code, detail, fix

    @property
    def unreachable(self) -> bool:
        return self.status == 0


def _refusal(r: httpx.Response) -> GatewayError:
    try:
        body = r.json()
    except ValueError:
        return GatewayError(r.status_code, None, r.text[:300])
    if not isinstance(body, dict):
        return GatewayError(r.status_code, None, str(body)[:300])
    detail = body.get("detail")
    if isinstance(detail, list):  # FastAPI request validation
        detail = "; ".join(str(d.get("msg")) for d in detail if isinstance(d, dict))
    if detail is None and isinstance(body.get("error"), dict):
        detail = body["error"].get("message")
    return GatewayError(r.status_code, body.get("code"), str(detail), body.get("fix"))


class Gateway:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 timeout: float = 30.0) -> None:
        # trust_env=False: proxy variables must never route the owner token or prompts off this machine
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout, trust_env=False)
        self._token: str | None = None

    def __repr__(self) -> str:
        return f"Gateway({str(self._client.base_url)!r})"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, *, timeout: float | None = None,
                       headers: Mapping[str, str] | None = None, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, path, headers=dict(headers or {}),
                                              timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
                                              **kwargs)
        except httpx.TimeoutException as exc:
            raise GatewayError(0, "timeout", f"the gateway did not answer {method} {path} in time") from exc
        except httpx.HTTPError as exc:
            raise GatewayError(0, "unreachable", f"cannot reach the gateway at {self._client.base_url} "
                                                 f"({type(exc).__name__})",
                               "start the host with `hypercompute host` on this machine") from exc

    async def _token_value(self, refresh: bool = False) -> str:
        if self._token is None or refresh:
            r = await self._request("GET", "/api/owner/bootstrap")
            if r.status_code != 200:
                raise _refusal(r)
            token = r.json().get("owner_token")
            if not isinstance(token, str) or not token:
                raise GatewayError(r.status_code, "bad_bootstrap", "the gateway sent no owner token")
            self._token = token
        return self._token

    async def call(self, method: str, path: str, *, owner: bool = False, timeout: float | None = None,
                   headers: Mapping[str, str] | None = None, **kwargs: Any) -> Any:
        merged = dict(headers or {})
        if owner:
            merged[OWNER_HEADER] = await self._token_value()
        r = await self._request(method, path, timeout=timeout, headers=merged, **kwargs)
        if r.status_code == 403 and owner:  # the host restarted with a new token: fetch it once and retry
            merged[OWNER_HEADER] = await self._token_value(refresh=True)
            r = await self._request(method, path, timeout=timeout, headers=merged, **kwargs)
        if r.status_code >= 400:
            raise _refusal(r)
        try:
            return r.json()
        except ValueError as exc:
            raise GatewayError(r.status_code, "bad_json", f"{method} {path} returned a non-JSON body") from exc

    # ------------------------------------------------------------------ reads
    async def health(self) -> dict:
        return await self.call("GET", "/api/health")

    async def nodes(self) -> list[dict]:
        return await self.call("GET", "/api/nodes")

    async def network(self) -> dict:
        return await self.call("GET", "/api/network")

    async def models(self) -> list[dict]:
        return await self.call("GET", "/api/models")

    async def deployments(self) -> list[dict]:
        return await self.call("GET", "/api/deployments")

    async def deployment(self, dep_id: str) -> dict:
        return await self.call("GET", f"/api/deployments/{dep_id}")

    # ------------------------------------------------------------------ writes
    async def plan(self, model_id: str, *, strategy: str, ctx_size: int = 4096,
                   node_ids: list[str] | None = None) -> dict:
        """POST /api/plans: a plan-only check. It stores a plan on the gateway and loads nothing."""
        body: dict[str, Any] = {"model_id": model_id, "strategy": strategy, "ctx_size": ctx_size}
        if node_ids is not None:
            body["node_ids"] = node_ids
        return await self.call("POST", "/api/plans", owner=True, json=body, timeout=180.0)

    async def deploy(self, plan_id: str, idempotency_key: str) -> dict:
        return await self.call("POST", "/api/deployments", owner=True, json={"plan_id": plan_id},
                               headers={"Idempotency-Key": idempotency_key}, timeout=60.0)

    async def stop(self, dep_id: str) -> dict:
        return await self.call("DELETE", f"/api/deployments/{dep_id}", owner=True, timeout=300.0)

    async def chat(self, messages: list[dict], *, max_tokens: int, timeout: float, temperature: float = 0.7,
                   extra: Mapping[str, Any] | None = None) -> dict:
        """A non-streamed completion; extra adds engine options (e.g. chat_template_kwargs) but never overrides
        messages, max_tokens or stream."""
        body = {**dict(extra or {}), "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "stream": False}
        return await self.call("POST", "/v1/chat/completions", owner=True, json=body, timeout=timeout)
