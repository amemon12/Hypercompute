"""`hypercompute tune`: ask the running host gateway to measure candidate splits, then print each result.

Talks to the host on 127.0.0.1 only. The owner token comes from /api/owner/bootstrap, the same loopback-only
endpoint the dashboard uses, so no secret is stored or passed on the command line.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import httpx

from .agent.join import say

OWNER_HEADER = "X-Hypercompute-Owner"
POLL_S = 2.0
FINISHED = ("done", "failed", "skipped")


def _fail(message: str, fix: str | None = None, code: int = 2) -> int:
    say(f"error: {message}")
    if fix:
        say(f"fix:   {fix}")
    return code


def _detail(r: httpx.Response) -> tuple[str, str | None]:
    """The gateway's error body ({detail, code, fix}, or FastAPI's validation list) as a message and a fix."""
    try:
        body = r.json()
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:300]}", None
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, list):  # request validation: [{loc, msg}, ...]
        detail = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
    return f"HTTP {r.status_code}: {detail}", body.get("fix") if isinstance(body, dict) else None


def request_body(args: argparse.Namespace) -> dict:
    body: dict = {"model_id": args.model_id, "objective": args.objective, "ctx_size": args.ctx,
                  "max_candidates": args.max_candidates, "tune_flags": not args.no_flags}
    if args.nodes:
        body["node_ids"] = [n.strip() for n in args.nodes.split(",") if n.strip()]
    decode_tokens = getattr(args, "decode_tokens", None)
    if decode_tokens:
        body["decode_tokens"] = decode_tokens
    # the owner's plan limits; unset ones keep TuneRequest's defaults (llama.cpp, auto, RAM tier on, no capacity mode)
    for key in ("engine", "strategy"):
        if value := getattr(args, key, None):
            body[key] = value
    if getattr(args, "no_ram_tier", False):
        body["allow_ram_tier"] = False
    if getattr(args, "allow_capacity_mode", False):
        body["allow_capacity_mode"] = True
    return body


def _num(value: float | None, fmt: str, unit: str) -> str:
    return "-" if value is None else f"{value:{fmt}}{unit}"


def candidate_line(i: int, c: dict) -> str:
    if c["status"] != "done":
        return f"  {i + 1}. {c['label']}\n     {c['status']}" + (f": {c['error']}" if c.get("error") else "")
    return (f"  {i + 1}. {c['label']}\n     load {_num(c.get('load_s'), '.1f', ' s')}  "
            f"TTFT {_num(c.get('ttft_ms'), '.0f', ' ms')}  "
            f"decode {_num(c.get('decode_tokens_per_s'), '.1f', ' tok/s')}  "
            f"prefill {_num(c.get('prefill_tokens_per_s'), '.0f', ' tok/s')}  "
            f"score {_num(c.get('score'), '.2f', ' s')}")


def _summary(result: dict) -> int:
    if result["status"] == "done" and result.get("best_index") is not None:
        best = result["candidates"][result["best_index"]]
        saved = f"; saved as profile {result['profile_id'][:8]}" if result.get("profile_id") else ""
        say(f"Winner: #{result['best_index'] + 1} {best['label']} ({best['score']:.2f} s per typical turn){saved}")
        if result.get("note"):
            say(f"Note: {result['note']}")
        return 0
    say(f"Tune {result['status']}" + (f": {result['error']}" if result.get("error") else ""))
    return 1


def _owner_token(client: httpx.Client) -> str:
    r = client.get("/api/owner/bootstrap")
    r.raise_for_status()
    body = r.json()
    token = body.get("owner_token") if isinstance(body, dict) else None
    if not token:
        raise ValueError("the reply carried no owner token")
    return token


def _watch(client: httpx.Client, result: dict, sleep: Callable[[float], None]) -> dict:
    printed: set[int] = set()
    while True:
        for i, c in enumerate(result["candidates"]):
            if i not in printed and c["status"] in FINISHED:
                say(candidate_line(i, c))
                printed.add(i)
        if result["status"] != "running":
            return result
        sleep(POLL_S)
        r = client.get(f"/api/tune/{result['id']}")
        r.raise_for_status()
        result = r.json()


def _cancel(client: httpx.Client, tune_id: str, token: str) -> int:
    try:
        r = client.delete(f"/api/tune/{tune_id}", headers={OWNER_HEADER: token}, timeout=120.0)
        r.raise_for_status()
        say("Tune cancelled; the trial model was stopped.")
    except httpx.HTTPError as exc:
        say(f"Could not cancel tune {tune_id[:8]} ({exc}); cancel it from the dashboard.")
    return 130


def _tune(client: httpx.Client, args: argparse.Namespace, sleep: Callable[[float], None]) -> int:
    try:
        token = _owner_token(client)
    except httpx.TransportError as exc:  # connection refused, timeout: nothing is listening
        return _fail(f"no Hypercompute host is running on 127.0.0.1:{args.port} ({exc.__class__.__name__})",
                     "start one with `uv run hypercompute host` (or pass --port)")
    except (httpx.HTTPStatusError, ValueError) as exc:
        return _fail(f"the service on 127.0.0.1:{args.port} did not hand out an owner token: {exc}",
                     f"check that it is a Hypercompute host: `uv run hypercompute status --port {args.port}`")
    try:
        r = client.post("/api/tune", json=request_body(args), headers={OWNER_HEADER: token})
        if r.status_code != 202:
            return _fail(*_detail(r), code=1)
        result = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return _fail(f"the host did not accept the tune request: {exc}", "check `hypercompute status`")
    say(f"Tuning {args.model_id} ({args.objective}): {len(result['candidates'])} candidate(s), tune "
        f"{result['id'][:8]}. Each one loads the model, so this takes a few minutes.")
    try:
        return _summary(_watch(client, result, sleep))
    except KeyboardInterrupt:
        return _cancel(client, result["id"], token)
    except (httpx.HTTPError, ValueError) as exc:
        return _fail(f"lost contact with the host while tuning: {exc}",
                     f"the tune keeps running on the host; see `GET /api/tune/{result['id']}` or the dashboard")


def run_tune(args: argparse.Namespace, client: httpx.Client | None = None,
             sleep: Callable[[float], None] = time.sleep) -> int:
    own = client is None
    client = client or httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=30.0)
    try:
        return _tune(client, args, sleep)
    finally:
        if own:
            client.close()
