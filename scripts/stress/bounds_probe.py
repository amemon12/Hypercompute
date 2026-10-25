"""Test-day helper: probe request-schema bounds and body limits just under / at / just over, without starting a tune
or a benchmark (at-limit values for those are only sent when the gateway will refuse them for another reason).

Usage (G14, repo root, gateway running):  uv run python <this file> --port 8000 [--model qwen2.5-0.5b-instruct-q4_k_m]
Each line: case, HTTP status, expected status(es), first error location/code.
"""

from __future__ import annotations

import argparse

import httpx

KiB = 1024


def first_error(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:60]
    if isinstance(body, dict) and isinstance(body.get("detail"), list):  # FastAPI 422
        return ";".join(f"{'.'.join(map(str, e.get('loc', [])))}:{e.get('type')}" for e in body["detail"][:2])
    if isinstance(body, dict):
        return str(body.get("code") or body.get("detail"))[:80]
    return f"list[{len(body)}]" if isinstance(body, list) else str(body)[:60]


def check(name: str, r: httpx.Response, expect: tuple[int, ...]) -> None:
    mark = "ok " if r.status_code in expect else "BAD"
    print(f"{mark} {name:48s} -> {r.status_code} (expect {'/'.join(map(str, expect))})  {first_error(r)}")


def padded(size: int) -> bytes:
    """A JSON object of exactly `size` bytes."""
    head, tail = b'{"pad":"', b'"}'
    return head + b"x" * (size - len(head) - len(tail)) + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct-q4_k_m")
    a = ap.parse_args()
    c = httpx.Client(base_url=f"http://127.0.0.1:{a.port}", timeout=120.0)
    h = {"X-Hypercompute-Owner": c.get("/api/owner/bootstrap").json()["owner_token"]}
    deps = c.get("/api/deployments").json()
    active = any(d["state"] not in ("stopped", "failed") for d in deps)
    ready = any(d["state"] == "ready" for d in deps)
    print(f"deployment active={active} ready={ready}")

    # PlanRequest.ctx_size: Field(ge=256, le=262144) schemas.py:348
    for ctx, exp in ((255, (422,)), (256, (200,)), (262144, (200,)), (262145, (422,))):
        check(f"plan ctx_size={ctx}", c.post("/api/plans", headers=h, json={
            "model_id": a.model, "strategy": "local_only", "ctx_size": ctx}), exp)

    # TuneRequest: max_candidates 1..8, decode_tokens 8..512 (schemas.py:582,584); at-values only when a deployment is
    # active, so the tuner answers 409 deployment_active (tuner.py:326) instead of starting a tune
    for field, val, ok in (("max_candidates", 0, False), ("max_candidates", 1, True), ("max_candidates", 8, True),
                           ("max_candidates", 9, False), ("decode_tokens", 7, False), ("decode_tokens", 8, True),
                           ("decode_tokens", 512, True), ("decode_tokens", 513, False)):
        if ok and not active:
            print(f"--  tune {field}={val}: skipped (would start a tune; run again with a deployment active)")
            continue
        check(f"tune {field}={val}", c.post("/api/tune", headers=h, json={"model_id": a.model, field: val}),
              (409,) if ok else (422,))
    if not active:  # decode_tokens + 64 > ctx_size -> 422 bad_request (tuner.py:331); 193 + 64 > 256
        check("tune ctx=256 decode_tokens=193", c.post("/api/tune", headers=h, json={
            "model_id": a.model, "ctx_size": 256, "decode_tokens": 193}), (422,))

    # BenchmarkRequest bounds (schemas.py:528-531); at-values only when nothing is ready (503 before starting)
    for field, val, ok in (("repetitions", 0, False), ("repetitions", 20, True), ("repetitions", 21, False),
                           ("warmup", 5, True), ("warmup", 6, False), ("max_tokens", 0, False),
                           ("max_tokens", 2048, True), ("max_tokens", 2049, False),
                           ("label", "x" * 80, True), ("label", "x" * 81, False)):
        if ok and ready:
            print(f"--  bench {field}: skipped (would start a benchmark on the ready deployment)")
            continue
        check(f"bench {field}={val if not isinstance(val, str) else len(val)}",
              c.post("/api/benchmarks", headers=h, json={field: val}), (503,) if ok else (422,))

    # CreateSessionRequest.name max_length=80 (schemas.py:265); 80 issues a new invite for the current session
    check("session name len=81", c.post("/api/sessions", headers=h, json={"name": "x" * 81}), (422,))

    # /api/runs?limit clamps to 1..500 (app.py:367)
    for lim in (0, 1, 500, 501):
        r = c.get(f"/api/runs?limit={lim}")
        print(f"ok  runs limit={lim:<4d} -> {r.status_code} returned {len(r.json())} (cap {max(1, min(lim, 500))})")

    # JoinRequest.code min 4 / max 32 (schemas.py:158): rejected by the schema, before the rate limiter
    for code in ("abc", "x" * 33):
        check(f"join code len={len(code)}", c.post("/api/sessions/join", json={"code": code}), (422,))

    # Body limits (app.py:55-57): Content-Length precheck and the streamed count both use `> limit`
    for path, limit, hdr in (("/api/plans", 64 * KiB, h), ("/api/sessions/join", 256 * KiB, {}),
                             ("/v1/chat/completions", 4096 * KiB, h)):
        for size in (limit, limit + 1):
            r = c.post(path, headers={**hdr, "Content-Type": "application/json"}, content=padded(size))
            check(f"{path} body={size}", r, (413,) if size > limit else (400, 403, 422, 503))
        r = c.post(path, headers={**hdr, "Content-Type": "application/json"},
                   content=iter([padded(limit + 1)]))  # no Content-Length: chunked, counted while streaming
        check(f"{path} chunked body={limit + 1}", r, (413,))


if __name__ == "__main__":
    main()
