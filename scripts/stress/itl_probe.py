"""Inter-token latency through the Hypercompute gateway (run on the host, loopback only).

Usage (repo folder, gateway running, a deployment ready):
    uv run python <this file> --tokens 256 --reps 5 --label pair-14b-mac
Prints TTFT, decode tok/s and inter-token gap percentiles per rep, then pooled. The gateway passes unknown body
fields through to llama-server, so ignore_eos / cache_prompt work as in scripts/e2e_check.py benchmarks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx

PROMPT = "Write a long, detailed story about a lighthouse keeper and a storm."


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def one_run(client: httpx.Client, headers: dict[str, str], tokens: int) -> tuple[float, list[float]]:
    body = {"stream": True, "max_tokens": tokens, "temperature": 0, "ignore_eos": True, "cache_prompt": False,
            "messages": [{"role": "user", "content": PROMPT}]}
    t0 = time.perf_counter()
    stamps: list[float] = []
    with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                stamps.append(time.perf_counter())
    if len(stamps) < 2:
        raise RuntimeError(f"only {len(stamps)} content chunks arrived")
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)]
    return (stamps[0] - t0) * 1000, gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    client = httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=httpx.Timeout(300.0, connect=5.0))
    headers = {"X-Hypercompute-Owner": client.get("/api/owner/bootstrap").json()["owner_token"]}
    pooled: list[float] = []
    rates: list[float] = []
    for rep in range(args.reps):
        ttft, gaps = one_run(client, headers, args.tokens)
        rate = 1000 / statistics.mean(gaps)
        rates.append(rate)
        pooled += gaps
        print(f"{args.label} rep {rep + 1}: ttft {ttft:.0f} ms, decode {rate:.1f} tok/s, gap p50 {pct(gaps, .5):.1f} "
              f"p90 {pct(gaps, .9):.1f} p99 {pct(gaps, .99):.1f} max {max(gaps):.1f} ms")
    cv = statistics.stdev(rates) / statistics.mean(rates) * 100 if len(rates) > 1 else 0.0
    print(f"{args.label} pooled: decode {statistics.median(rates):.1f} tok/s (CV {cv:.1f}%), gap p50 "
          f"{pct(pooled, .5):.1f} p90 {pct(pooled, .9):.1f} p99 {pct(pooled, .99):.1f} max {max(pooled):.1f} ms, "
          f"gaps > 3x p50: {sum(g > 3 * pct(pooled, .5) for g in pooled)}/{len(pooled)}")


if __name__ == "__main__":
    main()
