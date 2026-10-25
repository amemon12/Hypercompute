"""A stand-in for the Hypercompute gateway, for developing and demoing the proxy without touching a real pool.

It serves the same endpoints the proxy uses, with the real plan of the R6 Kimi-K2.6 1T pooled run (GX10 host plus
four peers, fixtures/kimi-k2.6-pooled-plan.json). Node display names here are illustrative; against a real gateway
the proxy reads them from /api/nodes.

    uv run python mock_gateway.py            # listens on 127.0.0.1:8010
    curl -X POST 127.0.0.1:8010/mock/mode -d '{"mode": "drop_mid_stream"}'

Modes: ok, not_ready (no model, like a stopped deployment), busy (the gateway's busy error before any output),
error_before_output (an upstream error before the first token), drop_mid_stream (a peer lost mid-reply: the
gateway's interrupted error after some tokens), empty_reply (finishes without generating any text).
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kimi-k2.6-pooled-plan.json"
MODES = ("ok", "not_ready", "busy", "error_before_output", "drop_mid_stream", "empty_reply")
NAMES = {"local": "gx10", "4c2991743c15": "mac-azim", "8f169e3549fc": "mac-a", "6dad6cf75b2e": "g14",
         "d4e65f83c4be": "mac-b"}
REPLY = ("Mixture-of-experts routing sends each token to a few expert networks chosen by a small router, so only a "
         "fraction of the weights run per token. Here those experts live on five machines, and each one did part "
         "of the work for this answer.")


def _sse(payload: dict | str) -> bytes:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n".encode()


def _chunk(cid: str, model: str, delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    out = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        out["usage"] = usage
    return out


def create_mock_app(token_delay_s: float = 0.03) -> FastAPI:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    plan, dep_id = fixture["plan"], fixture["deployment_id"]
    token = secrets.token_urlsafe(24)
    state: dict = {"mode": "ok", "chats": 0, "last_body": None}
    app = FastAPI(title="mock hypercompute gateway")
    full_ids = {a["node_id"] for a in plan["assignments"] if a["node_id"] != "local"}

    def node_name(node_id: str) -> str:
        return NAMES.get(node_id[:12], node_id[:12])

    @app.get("/api/owner/bootstrap")
    async def bootstrap(request: Request) -> JSONResponse:
        if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
            return JSONResponse(status_code=403, content={"code": "owner_only"})
        return JSONResponse({"owner_token": token, "version": "mock", "gateway_port": 8010})

    @app.get("/v1/models")
    async def models() -> dict:
        if state["mode"] == "not_ready":
            return {"object": "list", "data": []}
        return {"object": "list", "data": [{"id": plan["model_id"], "object": "model", "owned_by": "hypercompute",
                                            "deployment_id": dep_id}]}

    @app.get("/api/deployments/{dep}")
    async def deployment(dep: str) -> JSONResponse:
        if dep != dep_id:
            return JSONResponse(status_code=404, content={"code": "not_found"})
        return JSONResponse({"id": dep_id, "state": "ready", "purpose": "run", "plan": plan})

    @app.get("/api/nodes")
    async def nodes() -> list[dict]:
        local = [{"id": "f" * 32, "display_name": node_name("local"), "is_local": True}]
        return local + [{"id": nid, "display_name": node_name(nid), "is_local": False} for nid in sorted(full_ids)]

    @app.post("/mock/mode")
    async def set_mode(request: Request) -> JSONResponse:
        mode = (await request.json()).get("mode")
        if mode not in MODES:
            return JSONResponse(status_code=400, content={"error": f"mode must be one of {MODES}"})
        state["mode"] = mode
        return JSONResponse({"mode": mode})

    @app.get("/mock/state")
    async def get_state() -> dict:
        return dict(state)

    app.state.mock = state  # tests read what the proxy forwarded

    @app.post("/v1/chat/completions")
    async def chat(request: Request, x_hypercompute_owner: str | None = Header(default=None)):
        if x_hypercompute_owner != token:
            return JSONResponse(status_code=403, content={"code": "owner_only", "detail": "missing or wrong token"})
        if state["mode"] == "not_ready":
            return JSONResponse(status_code=503, content={"code": "no_ready_deployment", "detail": "no model ready"})
        body = await request.json()
        state["chats"] += 1
        state["last_body"] = body
        return StreamingResponse(_stream(body, state["mode"]), media_type="text/event-stream")

    async def _stream(body: dict, mode: str) -> AsyncIterator[bytes]:
        cid, model = f"chatcmpl-mock{state['chats']}", plan["model_id"]
        if mode == "busy":
            yield _sse({"error": {"code": "busy", "message": "all 1 inference slot(s) of the deployment are busy"}})
            return
        if mode == "error_before_output":
            yield _sse({"error": {"code": "upstream_error", "message": "engine request failed (mock)"}})
            return
        words = REPLY.split(" ")[: int(body.get("max_tokens") or 64)]
        yield _sse(_chunk(cid, model, {"role": "assistant", "content": None}))
        if mode == "empty_reply":
            yield _sse(_chunk(cid, model, {}, finish="stop", usage={"prompt_tokens": 24, "completion_tokens": 0}))
            yield _sse("[DONE]")
            return
        for i, word in enumerate(words):
            await asyncio.sleep(token_delay_s)
            if mode == "drop_mid_stream" and i == len(words) // 2:
                partial = " ".join(words[:i])
                yield _sse({"error": {"code": "interrupted", "type": "upstream_error", "run_id": "mock-run",
                                      "message": "interrupted: deployment stopped or a peer was lost",
                                      "partial_text": partial}})
                return
            yield _sse(_chunk(cid, model, {"content": word if i == 0 else " " + word}))
        n = len(words)
        yield _sse(_chunk(cid, model, {}, finish="length" if n < len(REPLY.split(" ")) else "stop",
                          usage={"prompt_tokens": 24, "completion_tokens": n, "total_tokens": 24 + n}))
        yield _sse("[DONE]")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_app(float(os.environ.get("MOCK_TOKEN_DELAY_S", "0.03"))), host="127.0.0.1",
                port=int(os.environ.get("MOCK_PORT", "8010")))
