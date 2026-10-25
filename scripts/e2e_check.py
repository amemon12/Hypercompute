"""Drive a running Hypercompute host through plan -> deploy -> stream -> (benchmark) -> stop.

Real hardware validation, not a protocol test: it talks to `uv run hypercompute host` on this machine
and saves the raw plan/deployment/runs/benchmark JSON under results/ as evidence.

  uv run python scripts/e2e_check.py --strategy local_only
  uv run python scripts/e2e_check.py --strategy all_selected --bench
  uv run python scripts/e2e_check.py --engine vllm --objective throughput --model <hf model id> --bench
  uv run python scripts/e2e_check.py --model <id> --strategy auto --plan-only     # the prediction, nothing loaded
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct-q4_k_m")
    ap.add_argument("--strategy", default="local_only", choices=["auto", "local_only", "fastest_fit", "all_selected"])
    ap.add_argument("--node", action="append", help="restrict to these peer node ids")
    ap.add_argument("--engine", choices=["llama_rpc", "vllm", "airllm", "exo"],
                    help="force an engine (default: the selector chooses)")
    ap.add_argument("--objective", default="interactive", choices=["interactive", "throughput"])
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prompt", default="Name three planets and one fact about each, briefly.")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="leave the deployment running")
    ap.add_argument("--plan-only", action="store_true", help="print and save the plan, do not deploy")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    c = httpx.Client(base_url=base, timeout=60.0)
    for _ in range(120):
        try:
            c.get("/api/health").raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(1)
    token = c.get("/api/owner/bootstrap").json()["owner_token"]
    h = {"X-Hypercompute-Owner": token}
    evidence: dict = {"started_at": datetime.now(UTC).isoformat(), "args": vars(args)}

    evidence["nodes"] = c.get("/api/nodes").json()
    for n in evidence["nodes"]:
        print(f"node {n['display_name']}: {n['state']} {n.get('state_reason') or ''}")
    req = {"model_id": args.model, "strategy": args.strategy, "ctx_size": args.ctx, "node_ids": args.node,
           "engine": args.engine, "objective": args.objective}
    plan = c.post("/api/plans", json=req, headers=h).json()
    evidence["plan"] = plan
    print(f"plan {plan.get('engine')}/{plan.get('tier')} {plan.get('fit_label')} feasible={plan.get('feasible')} "
          f"model={plan.get('model_id')} devices={plan.get('device_list')} split={plan.get('tensor_split')} "
          f"cpu_layers={plan.get('cpu_layers')} ngl={plan.get('n_gpu_units')}")
    gib = 1024**3
    for a in plan.get("assignments", []):
        print(f"  {a['device']}: {len(a['layers'])} layers{' + head' if a['holds_output'] else ''}, "
              f"{a['total_bytes_est'] / gib:.2f} of {a['budget_bytes'] / gib:.2f} GiB")
    for r in plan.get("reasons", []) + plan.get("warnings", []):
        print(f"  - {r}")
    if not plan.get("feasible") or args.plan_only:
        return save(evidence, args, 0 if plan.get("feasible") else 1)

    r = c.post("/api/deployments", json={"plan_id": plan["id"]}, headers={**h, "Idempotency-Key": uuid.uuid4().hex})
    if r.status_code >= 300:
        print("deployment refused:", r.text)
        evidence["deployment_error"] = r.json()
        return save(evidence, args, 1)
    dep = r.json()
    t0, seen = time.perf_counter(), ""
    while dep["state"] not in ("ready", "failed", "stopped"):
        time.sleep(0.5)
        dep = c.get(f"/api/deployments/{dep['id']}").json()
        line = f"{dep['state']}: {dep.get('progress') or ''}"
        if line != seen:
            print(f"  [{time.perf_counter() - t0:6.1f}s] {line}")
            seen = line
    evidence["load_seconds"] = round(time.perf_counter() - t0, 2)
    evidence["deployment"] = dep
    if dep["state"] != "ready":
        print("deployment failed:", dep.get("error"), "| cleanup:", dep.get("cleanup"))
        return save(evidence, args, 1)
    print("placement:", dep.get("participation_detail"))

    # the selector may serve a variant (e.g. the AWQ build for objective=throughput), so ask for what the plan serves
    body = {"model": plan["model_id"], "messages": [{"role": "user", "content": args.prompt}], "max_tokens": 128,
            "stream": True, "temperature": 0}
    t1, first, text, final = time.perf_counter(), None, [], {}
    with c.stream("POST", "/v1/chat/completions", json=body, headers=h, timeout=300.0) as resp:
        if resp.status_code != 200:
            err = resp.read().decode(errors="replace")
            evidence["chat_error"] = {"status": resp.status_code, "body": err[:2000]}
            print(f"chat refused: HTTP {resp.status_code} {err[:300]}")
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            if "error" in payload:  # busy, timeout, upstream_error arrive inside a 200 stream
                evidence["chat_error"] = payload["error"]
                print("chat error:", payload["error"])
            for ch in payload.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and first is None:
                    first = (time.perf_counter() - t1) * 1000
                text.append(piece)
            if payload.get("timings"):
                final = payload
    print("answer:", "".join(text).strip()[:400])
    print(f"client TTFT {first:.0f} ms" if first else "no tokens", "| timings:", final.get("timings"))

    if args.bench:
        b = c.post("/api/benchmarks", json={"prompt_set": "both", "repetitions": args.reps, "warmup": 1,
                                             "max_tokens": 128}, headers=h).json()
        while b["status"] == "running":
            time.sleep(2)
            b = c.get(f"/api/benchmarks/{b['id']}").json()
        evidence["benchmark"] = b
        for s in b["summaries"]:
            print(f"bench {s['prompt_label']}: ttft median {s['ttft_ms']['median']} ms "
                  f"[{s['ttft_ms']['min']}-{s['ttft_ms']['max']}], decode median "
                  f"{s['decode_tokens_per_s']['median']} tok/s, failures {s['failures']}")
    evidence["runs"] = c.get("/api/runs?limit=20").json()
    if not args.keep:
        stopped = c.delete(f"/api/deployments/{dep['id']}", headers=h).json()
        evidence["stopped"] = {"state": stopped["state"], "cleanup": stopped.get("cleanup")}
        print("stopped:", stopped["state"], "|", stopped.get("cleanup"))
    return save(evidence, args, 1 if "chat_error" in evidence else 0)


def save(evidence: dict, args: argparse.Namespace, code: int) -> int:
    out = ROOT / "results" / "plans" if args.plan_only else ROOT / "results"  # plans stay out of RESULTS.md
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    engine = f"-{args.engine}" if args.engine else ""
    name = f"{stamp}-{args.strategy}{engine}{('-' + args.label) if args.label else ''}.json"
    (out / name).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"evidence: {(out / name).relative_to(ROOT).as_posix()}")
    return code


if __name__ == "__main__":
    sys.exit(main())
