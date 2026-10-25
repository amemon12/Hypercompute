"""Failure-injection load helper for Hypercompute (catalogue failures.md, injections C1-C6 and N11).

Run from the repo root on the G14 so httpx comes from the project environment:
  uv run python <this file> chats --n 32 --max-tokens 128            # N concurrent streamed chats
  uv run python <this file> chats --n 110 --max-tokens 256           # C3: past the 100-connection cap
  uv run python <this file> idle --host <G14-LAN-IP> --n 100 --hold 60   # N11/C3: idle sockets only, no auth

`chats` uses the owner endpoint, so it must run on the G14 itself (loopback). `idle` opens raw TCP connections
that never send a byte; uvicorn counts them against limit_concurrency. It writes one JSON line per request to
--out with absolute timestamps (UTC), so outcomes can be lined up with /api/events and the agent's log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import time
from collections import Counter
from datetime import UTC, datetime

import httpx


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


async def one_chat(client: httpx.AsyncClient, headers: dict, i: int, max_tokens: int, stream: bool) -> dict:
    body = {"messages": [{"role": "user", "content": f"Request {i}: write a long story about weather station {i}."}],
            "max_tokens": max_tokens, "temperature": 0.7, "stream": stream}
    rec: dict = {"i": i, "start": now()}
    t0 = time.perf_counter()
    try:
        if not stream:
            r = await client.post("/v1/chat/completions", json=body, headers=headers)
            rec["http"] = r.status_code
            try:
                rec["outcome"] = "ok" if r.status_code == 200 else (r.json().get("code") or f"http_{r.status_code}")
            except json.JSONDecodeError:
                rec["outcome"] = f"http_{r.status_code}"
        else:
            async with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
                rec["http"] = r.status_code
                if r.status_code != 200:
                    rec["outcome"] = f"http_{r.status_code}"
                else:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            rec["outcome"] = "ok"
                            break
                        obj = json.loads(data)
                        if "error" in obj:
                            rec["outcome"] = str(obj["error"].get("code", "error"))
                            rec["msg"] = str(obj["error"].get("message", ""))[:160]
                            break
                        if "ttft_s" not in rec and any((c.get("delta") or {}).get("content")
                                                       for c in obj.get("choices") or []):
                            rec["ttft_s"] = round(time.perf_counter() - t0, 3)
                    rec.setdefault("outcome", "no_done")  # 200 that ended without [DONE] or an error event
    except httpx.HTTPError as exc:
        rec["outcome"] = f"transport_{exc.__class__.__name__}"
    rec["end"], rec["elapsed_s"] = now(), round(time.perf_counter() - t0, 3)
    return rec


async def run_chats(args: argparse.Namespace) -> list[dict]:
    limits = httpx.Limits(max_connections=args.n + 4, max_keepalive_connections=args.n + 4)
    async with httpx.AsyncClient(base_url=args.gw, timeout=httpx.Timeout(900.0, connect=10.0), limits=limits) as c:
        boot = await c.get("/api/owner/bootstrap")
        boot.raise_for_status()
        headers = {"X-Hypercompute-Owner": boot.json()["owner_token"]}
        return list(await asyncio.gather(*(one_chat(c, headers, i, args.max_tokens, not args.no_stream)
                                           for i in range(args.n))))


def run_idle(args: argparse.Namespace) -> list[dict]:
    socks, out = [], []
    for i in range(args.n):
        try:
            socks.append(socket.create_connection((args.host, args.port), timeout=5.0))
            out.append({"i": i, "outcome": "open", "at": now()})
        except OSError as exc:
            out.append({"i": i, "outcome": f"refused_{exc.__class__.__name__}", "at": now()})
    print(f"{now()} holding {len(socks)} idle connection(s) for {args.hold:.0f}s", flush=True)
    time.sleep(args.hold)
    for s in socks:
        s.close()
    print(f"{now()} released", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    ch = sub.add_parser("chats")
    ch.add_argument("--gw", default="http://127.0.0.1:8000")
    ch.add_argument("--n", type=int, default=32)
    ch.add_argument("--max-tokens", type=int, default=128)
    ch.add_argument("--no-stream", action="store_true")
    idle = sub.add_parser("idle")
    idle.add_argument("--host", required=True)
    idle.add_argument("--port", type=int, default=8000)
    idle.add_argument("--n", type=int, default=100)
    idle.add_argument("--hold", type=float, default=60.0)
    for p in (ch, idle):
        p.add_argument("--out", default="failures_load.jsonl")
    args = ap.parse_args()
    started = now()
    records = asyncio.run(run_chats(args)) if args.mode == "chats" else run_idle(args)
    with open(args.out, "a", encoding="utf-8", newline="\n") as fh:
        for rec in sorted(records, key=lambda r: r["i"]):
            fh.write(json.dumps({"mode": args.mode, "batch_start": started, **rec}) + "\n")
    print(json.dumps({"mode": args.mode, "n": args.n, "started": started, "finished": now(),
                      "outcomes": dict(Counter(r["outcome"] for r in records))}))


if __name__ == "__main__":
    main()
