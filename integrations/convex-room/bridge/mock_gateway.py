"""A stand-in for the Hypercompute gateway's owner API, for developing the room without touching a real host.

    python bridge/mock_gateway.py --port 8799 [--decode-tps 6] [--prefill-tps 40] [--slots 1]

It serves the shapes the bridge reads (/api/owner/bootstrap, /api/nodes, /api/deployments, /api/network) and an
OpenAI-style streamed /v1/chat/completions with llama.cpp-style `timings`. Every number it reports is made up: the
model id starts with "mock-" so a screenshot of it can never pass for a measurement.

Fault switches, for rehearsing the failure paths (POST JSON to /mock/faults, GET to read them):
    {"busy_next": 2}     the next 2 chats answer with the gateway's SSE busy error before any output
    {"cut_after": 5}     the next chat's connection drops after 5 tokens, like a gateway killed mid-answer
    {"down": true}       no ready deployment: chats get 503, /api/deployments shows "loading"
    {"decode_tps": 1.8}  change the simulated speeds
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
MODEL_ID = "mock-kimi-k2.6-1t"

ANSWERS = {
    "split": "Each machine runs a llama.cpp RPC worker. The host sends activations layer by layer, so no single box "
             "needs the full trillion parameters in memory.",
    "convex": "Convex keeps the room: the queue, the leases, who is here, and every token, pushed to every screen "
              "the moment it is written.",
    "lease": "A lease is a claim with an expiry. The host keeps renewing it while it works; if it goes quiet, the "
             "lease runs out and a scheduled function puts the job back in the queue.",
    "haiku": "Borrowed GPUs hum, one model on many desks, tokens cross the room.",
    "": "Five machines hold one model here: the GX10 keeps the first layers, the laptops and Macs keep the rest, "
        "and every token walks through all of them over an encrypted QUIC tunnel.",
}
KEYWORDS = [("lease", "lease"), ("convex", "convex"), ("haiku", "haiku"), ("split", "split"), ("laptop", "split")]


def pick_answer(text: str) -> str:
    """A canned answer that fits the question, so a rehearsal or a recording reads sensibly."""
    lowered = text.lower()
    return next((ANSWERS[key] for word, key in KEYWORDS if word in lowered), ANSWERS[""])


def nodes() -> list[dict[str, Any]]:
    def node(node_id: str, name: str, system: str, device: str, local: bool = False) -> dict[str, Any]:
        return {"id": node_id, "display_name": name, "is_local": local, "state": "eligible",
                "inventory": {"os": {"system": system}, "devices": [{"id": device}]},
                "contribution": {"devices": [device]}}

    return [
        node("local", "gx10 (mock)", "linux", "CUDA0", local=True),
        node("a" * 32, "g14 (mock)", "windows", "CUDA0"),
        node("b" * 32, "macbook-air-1 (mock)", "macos", "MTL0"),
        node("c" * 32, "macbook-air-2 (mock)", "macos", "MTL0"),
        node("d" * 32, "macbook-air-3 (mock)", "macos", "MTL0"),
    ]


def deployment(state: str, slots: int) -> dict[str, Any]:
    split = [("local", "CUDA0", 0, 34, 300e9), ("a" * 32, "CUDA0", 34, 40, 50e9),
             ("b" * 32, "MTL0", 40, 47, 58e9), ("c" * 32, "MTL0", 47, 54, 58e9), ("d" * 32, "MTL0", 54, 61, 58e9)]
    assignments = [{"node_id": n, "device": d, "layers": list(range(a, b)), "weight_bytes": int(w),
                    "holds_output": b == 61} for n, d, a, b, w in split]
    return {"id": "mock-dep", "state": state, "purpose": "run", "created_at": "2026-10-25T00:00:00Z",
            "plan": {"model_id": MODEL_ID, "tier": "gpu", "engine_options": {"parallel": slots},
                     "assignments": assignments},
            "placement": {"offloaded": "62/62"}}


def network() -> dict[str, Any]:
    paths = {"a" * 32: "relay", "b" * 32: "direct", "c" * 32: "direct", "d" * 32: "direct"}
    return {"transport": "iroh", "peers": [{"node_id": n, "connected": True, "path": {"mode": m}}
                                           for n, m in paths.items()]}


def sse(payload: dict[str, Any] | str) -> bytes:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n".encode()


def build_app(decode_tps: float, prefill_tps: float, slots: int) -> FastAPI:
    app = FastAPI(title="mock hypercompute gateway", docs_url=None, redoc_url=None)
    token = secrets.token_urlsafe(24)
    faults: dict[str, Any] = {"busy_next": 0, "cut_after": None, "down": False,
                              "decode_tps": decode_tps, "prefill_tps": prefill_tps}
    free = {"slots": slots}

    def loopback(request: Request) -> None:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if (request.client.host if request.client else "") not in LOOPBACK or host not in LOOPBACK:
            raise HTTPException(403, "owner endpoints are only available from this machine")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": "mock"}

    @app.get("/api/owner/bootstrap")
    async def bootstrap(request: Request) -> dict[str, Any]:
        loopback(request)
        return {"owner_token": token, "version": "mock", "gateway_port": request.url.port}

    @app.get("/api/nodes")
    async def list_nodes(request: Request) -> list[dict[str, Any]]:
        loopback(request)
        return nodes()

    @app.get("/api/deployments")
    async def list_deployments(request: Request) -> list[dict[str, Any]]:
        loopback(request)
        return [deployment("loading" if faults["down"] else "ready", slots)]

    @app.get("/api/network")
    async def get_network(request: Request) -> dict[str, Any]:
        loopback(request)
        return network()

    @app.get("/mock/faults")
    async def get_faults() -> dict[str, Any]:
        return faults

    @app.post("/mock/faults")
    async def set_faults(request: Request) -> dict[str, Any]:
        loopback(request)
        body = await request.json()
        faults.update({k: v for k, v in body.items() if k in faults})
        return faults

    @app.post("/v1/chat/completions")
    async def chat(request: Request, x_hypercompute_owner: str | None = Header(default=None)) -> Any:
        loopback(request)
        if not x_hypercompute_owner or not secrets.compare_digest(x_hypercompute_owner, token):
            return JSONResponse(status_code=403,
                                content={"detail": "missing or wrong owner token", "code": "owner_only"})
        if faults["down"]:
            return JSONResponse(status_code=503, content={"detail": "no ready deployment", "code": "no_deployment"})
        body = await request.json()
        text = " ".join(str(m.get("content", "")) for m in body.get("messages", []) if m.get("role") == "user")
        answer = pick_answer(text)
        words = [w if i == 0 else f" {w}" for i, w in enumerate(answer.split(" "))]
        max_tokens = int(body.get("max_tokens") or 48)
        per_token = bool(body.get("timings_per_token"))
        prompt_n = max(8, len(text) // 4)

        async def stream() -> Any:
            if faults["busy_next"] > 0 or free["slots"] <= 0:
                faults["busy_next"] = max(0, faults["busy_next"] - 1)
                yield sse({"error": {"code": "busy", "message": "all 1 inference slot(s) of the deployment are busy"}})
                return
            free["slots"] -= 1
            cut, faults["cut_after"] = faults["cut_after"], None
            try:
                prompt_ms = prompt_n / faults["prefill_tps"] * 1000
                await asyncio.sleep(prompt_ms / 1000)
                started = time.perf_counter()
                n = min(max_tokens, len(words))
                for i, word in enumerate(words[:n]):
                    if cut is not None and i >= cut:
                        raise ConnectionResetError("mock gateway killed mid-answer")
                    await asyncio.sleep(1 / faults["decode_tps"])
                    chunk: dict[str, Any] = {"choices": [{"index": 0, "delta": {"content": word},
                                                          "finish_reason": None}]}
                    if per_token:
                        ms = (time.perf_counter() - started) * 1000
                        chunk["timings"] = {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "predicted_n": i + 1,
                                            "predicted_ms": ms, "predicted_per_second": (i + 1) / ms * 1000,
                                            "prompt_per_second": faults["prefill_tps"]}
                    yield sse(chunk)
                ms = (time.perf_counter() - started) * 1000
                reason = "length" if n < len(words) else "stop"
                yield sse({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                           "usage": {"prompt_tokens": prompt_n, "completion_tokens": n},
                           "timings": {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "predicted_n": n,
                                       "predicted_ms": ms, "predicted_per_second": n / ms * 1000,
                                       "prompt_per_second": faults["prefill_tps"]}})
                yield sse("[DONE]")
            finally:
                free["slots"] += 1

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--decode-tps", type=float, default=6.0)
    p.add_argument("--prefill-tps", type=float, default=40.0)
    p.add_argument("--slots", type=int, default=1)
    args = p.parse_args()
    uvicorn.run(build_app(args.decode_tps, args.prefill_tps, args.slots), host="127.0.0.1", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
