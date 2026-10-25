"""Test-day helper: fire N concurrent chat requests at the gateway and count outcomes.

Usage (G14, repo root):  uv run python scripts/stress/flood.py --port 8000 -n 20 --max-tokens 256 [--no-stream]
Long prompts:  uv run python scripts/stress/mkprompt.py 2000 p2k.json, then add --body p2k.json (every request posts
that body; --max-tokens and --no-stream still apply).
Outcome keys: (HTTP status, SSE/JSON error code or None). Streams that hit a limit still answer HTTP 200 and carry
`data: {"error": {"code": ...}}` (services/streaming.py:_sse_error); uvicorn's connection cap answers a plain 503.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter

import httpx


def error_code(raw: bytes) -> str:
    """The `code` of a gateway error body ({"detail", "code", "fix"}), or the start of the raw text."""
    try:
        data = json.loads(raw)
        err = data.get("error") or {}
        return data.get("code") or err.get("code") or err.get("type") or raw.decode(errors="replace")[:40]
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return raw.decode(errors="replace")[:40]


async def one(client: httpx.AsyncClient, i: int, headers: dict, max_tokens: int, stream: bool,
              base: dict | None = None) -> tuple:
    body = {"messages": [{"role": "user", "content": f"Request {i}: tell a long story about station {i}."}],
            "temperature": 0.7}
    body = {**(base or body), "max_tokens": max_tokens, "stream": stream}
    t0 = time.perf_counter()
    try:
        if not stream:
            r = await client.post("/v1/chat/completions", headers=headers, json=body)
            return r.status_code, None if r.status_code == 200 else error_code(r.content), time.perf_counter() - t0
        async with client.stream("POST", "/v1/chat/completions", headers=headers, json=body) as r:
            if r.status_code != 200:  # gateway refusals (503, 429, 4xx) are one JSON body, not an SSE stream
                return r.status_code, error_code(await r.aread()), time.perf_counter() - t0
            code = None
            async for line in r.aiter_lines():
                if line.startswith('data: {"error"'):
                    code = json.loads(line[6:])["error"]["code"]
            return r.status_code, code, time.perf_counter() - t0
    except httpx.HTTPError as exc:
        return "exception", exc.__class__.__name__, time.perf_counter() - t0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--body", help="chat body JSON to post for every request (from mkprompt.py)")
    args = ap.parse_args()
    base = None
    if args.body:
        with open(args.body, encoding="utf-8") as fh:
            base = json.load(fh)
        if not isinstance(base, dict) or not isinstance(base.get("messages"), list):
            ap.error(f"{args.body} is not a chat body with a messages list")
    limits = httpx.Limits(max_connections=args.n + 10, max_keepalive_connections=args.n + 10)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=900.0, limits=limits) as c:
        token = (await c.get("/api/owner/bootstrap")).json()["owner_token"]
        headers = {"X-Hypercompute-Owner": token}
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(c, i, headers, args.max_tokens, not args.no_stream, base)
                                     for i in range(args.n)))
    print(f"{args.n} requests in {time.perf_counter() - t0:.1f}s")
    for (status, code), k in sorted(Counter((s, e) for s, e, _ in res).items(), key=str):
        lat = [t for s, e, t in res if (s, e) == (status, code)]
        print(f"  status={status} code={code}: {k}  latency s median={statistics.median(lat):.1f} max={max(lat):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
