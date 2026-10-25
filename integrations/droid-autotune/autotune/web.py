"""The owner's page and API, on 127.0.0.1 only.

Reads are open to loopback; every action (trigger, approve, reject, cancel) also needs the per-process owner token
that the page carries, so another local page cannot drive approvals. The Host header must be loopback too, which keeps
a DNS-rebinding page from reaching this port.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .service import AutotuneService, ServiceError

STATIC = Path(__file__).resolve().parent / "static"
LOOPBACK_CLIENTS = {"127.0.0.1", "::1"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
TOKEN_HEADER = "X-Autotune-Owner"


class ApproveBody(BaseModel):
    index: int = Field(ge=0, le=7)


def _hostname(host: str) -> str:
    host = host.lower()
    return host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]


def pool_view(service: AutotuneService) -> list[dict[str, Any]]:
    snap = service.snap
    if snap is None:
        return []
    return [{"id": n.id, "name": n.name, "local": n.is_local, "state": n.state, "path": n.path, "rtt_ms": n.rtt_ms,
             "devices": [d.name for d in n.devices]} for n in snap.nodes]


def create_app(service: AutotuneService, *, token: str | None = None, watch: bool = True) -> FastAPI:
    owner_token = token or secrets.token_urlsafe(24)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(service.watch_forever()) if watch else None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await service.shutdown()
            await service.client.aclose()

    app = FastAPI(title="Hypercompute Droid autotune", lifespan=lifespan, docs_url=None, redoc_url=None)

    def local_only(request: Request) -> None:
        client = request.client.host if request.client else ""
        if client not in LOOPBACK_CLIENTS or _hostname(request.headers.get("host") or "") not in LOOPBACK_HOSTS:
            raise ServiceError(403, "autotune answers only on http://127.0.0.1")

    def owner(request: Request, x_autotune_owner: Annotated[str | None, Header()] = None) -> None:
        local_only(request)
        if not x_autotune_owner or not hmac.compare_digest(x_autotune_owner, owner_token):
            raise ServiceError(403, "missing or wrong owner token; reload the page")

    Local, Owner = Depends(local_only), Depends(owner)

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse, dependencies=[Local])
    async def index() -> HTMLResponse:
        html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__AUTOTUNE_TOKEN__", owner_token)
        return HTMLResponse(html, headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                                           "Content-Security-Policy": "default-src 'self'; script-src 'self' "
                                           "'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors "
                                           "'none'"})

    @app.get("/api/state", dependencies=[Local])
    async def state() -> dict[str, Any]:
        s = service.settings
        ready, why = s.droid_ready
        pending = service.watcher.pending
        return {
            "runner": service.runner_name, "droid": {"ready": ready, "detail": why, "mode": s.droid_mode},
            "gateway": {"url": s.gateway, **service.gateway_status},
            "limits": {"settle_s": s.settle_s, "max_checks": s.max_checks, "max_calls": s.max_calls,
                       "timeout_s": s.timeout_s},
            "pool": pool_view(service),
            "pending": pending.describe() if pending else None,
            "watch_log": list(service.store.watch_log)[-30:],
            "runs": [{"id": r.id, "state": r.state, "created_at": r.created_at, "runner": r.runner,
                      "reason": r.trigger.get("reason"), "model_id": r.model_id,
                      "recommendation": r.proposal.recommendation if r.proposal else None}
                     for r in service.store.list()],
        }

    @app.get("/api/runs/{run_id}", dependencies=[Local])
    async def run(run_id: str) -> dict[str, Any]:
        record = service.store.get(run_id)
        if record is None:
            raise ServiceError(404, "no such run")
        return record.model_dump(mode="json", exclude={"current_plan"})

    @app.post("/api/trigger", dependencies=[Owner])
    async def trigger() -> dict[str, Any]:
        return (await service.trigger("owner asked for a run")).model_dump(mode="json", include={"id", "state"})

    @app.post("/api/runs/{run_id}/approve", dependencies=[Owner])
    async def approve(run_id: str, body: ApproveBody) -> dict[str, Any]:
        return (await service.approve(run_id, body.index)).model_dump(mode="json", include={"id", "state"})

    @app.post("/api/runs/{run_id}/reject", dependencies=[Owner])
    async def reject(run_id: str) -> dict[str, Any]:
        return service.reject(run_id).model_dump(mode="json", include={"id", "state"})

    @app.post("/api/runs/{run_id}/cancel", dependencies=[Owner])
    async def cancel(run_id: str) -> dict[str, Any]:
        return (await service.cancel(run_id)).model_dump(mode="json", include={"id", "state"})

    return app
