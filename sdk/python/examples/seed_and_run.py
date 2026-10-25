"""Run one big model across the host and every peer, end to end, through the SDK.

Usage, on the host:
    PYTHONPATH=sdk/python/src python sdk/python/examples/seed_and_run.py [MODEL_ID] [--allow-disk-tiers]
        [--layout=FILE] [--max-tokens=N] [--seed-timeout=SECONDS] [--keep-running]
--layout=FILE pins the stages (a JSON list of {node_id, device, layers}, peers first, the host's GPU last); the planner
still checks every stage against that machine's free memory.
--max-tokens=N sets each chat and stream response budget. Both must return visible answer text before
--keep-running leaves the model loaded; reasoning-only output is reported as a failed check.
By default the example refuses the disk tiers (gpu_disk, stream): they page weights from SSD through the host's
memory, and on a unified-memory host (a GB10) a 142 GiB model planned that way swamped the box until it rebooted.
Verified on 2026-10-25 with kimi-k2.6-iq1_s across a GX10 host, a Windows laptop and two Macs
(docs/E2E-CHECK-2026-10-25.md).
Plan with the SDK (every selected device gets layers), seed each peer's share from Hugging Face through the gateway's
seed API (the SDK does not wrap it), then deploy, chat, stream and stop with the SDK. Prints no token or invite."""
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx
from hypercompute_sdk import Pool  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = ARGS[0] if ARGS else "kimi-k2.6-iq1_s"
ALLOW_DISK = "--allow-disk-tiers" in sys.argv[1:]
LAYOUT_FILE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--layout=")), None)
MAX_TOKENS = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--max-tokens=")), "64"))
SEED_TIMEOUT = float(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seed-timeout=")), "3600"))
KEEP_RUNNING = "--keep-running" in sys.argv[1:]
DISK_TIERS = {"gpu_disk", "stream"}
BASE = "http://127.0.0.1:8001"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()), msg, flush=True)


def seed_outcome(job: str | None, row: dict) -> str:
    """Read this job's host result before consulting the peer's latest report."""
    req = row.get("seed_request") or {}
    if job is None or req.get("job_id") != job:
        return "failed (the host no longer tracks this seed job)"
    if req.get("state") in {"failed", "refused", "cancelled"}:
        # Older hosts expose nothing_to_seed only in detail. Match that complete response,
        # including an empty job, so unrelated failures never count as a successful seed.
        nothing_to_seed = re.fullmatch(
            r"layers \d+-\d+ of [A-Za-z0-9_.-]+ have no tensor over \d+(?:\.\d+)? [KMGTPE]?i?B; "
            r"the host sends smaller ones at load anyway", str(req.get("detail") or ""))
        if (req.get("state") == "failed" and nothing_to_seed
                and req.get("tensors") == 0 and req.get("parts") == 0):
            return "skipped (small tensors transfer during loading)"
        return f"failed (host seed request {req['state']})"
    seed = (row.get("report") or {}).get("seed") or {}
    if seed.get("job_id") == job:
        if seed.get("state") == "done":
            return "done"
        if seed.get("state") in {"failed", "cancelled"}:
            return f"failed (peer seed job {seed['state']})"
    return "pending"


def smallest_share(pool: Pool, failed: dict) -> dict | None:
    """When the planner finds no automatic layout across every machine, try the smallest share on each eligible peer
    (one layer each, the rest on the host's GPU) in each peer order, and keep the first plan the planner accepts."""
    total = sum(len(a.get("layers") or []) for a in failed["assignments"])
    host = next((a["device"] for a in failed["assignments"]
                 if a["node_id"] == "local" and a["device"] != "CPU"), "CUDA0")
    peers = [p for p in pool.peers() if p.get("state") == "eligible"]
    for order in itertools.permutations(peers):
        stages = [{"node_id": p["node_id"], "device": (p.get("contribution_devices") or ["CPU"])[0], "layers": 1}
                  for p in order]
        stages.append({"node_id": "local", "device": host, "layers": total - len(stages)})
        try:
            plan = pool.plan(MODEL, strategy="all_selected", allow_capacity_mode=True, layout=stages)
        except Exception as exc:  # noqa: BLE001 - a 422 means this order does not fit; try the next
            log(f"  smallest share {[p['display_name'] for p in order]}: {str(exc).splitlines()[0][:100]}")
            continue
        if plan["feasible"]:
            log(f"no automatic plan across every machine; the planner accepted the smallest share per peer "
                f"{[p['display_name'] for p in order]} on tier {plan.get('tier')}")
            return plan
    return None


def main() -> int:
    if not math.isfinite(SEED_TIMEOUT) or SEED_TIMEOUT <= 0:
        log("--seed-timeout must be a positive, finite number of seconds")
        return 1
    pool = Pool()
    layout = json.loads(Path(LAYOUT_FILE).read_text(encoding="utf-8")) if LAYOUT_FILE else None
    # capacity mode stays on so the planner may keep MoE experts on the host (moe_offload); the disk tiers it also
    # allows are refused below unless --allow-disk-tiers is given (the planner itself never pages from disk on a
    # unified-memory host)
    plan = pool.plan(MODEL, strategy="all_selected", allow_capacity_mode=True, layout=layout)
    stages = [(a["node_id"][:6], a["device"], len(a.get("layers") or []))
              for a in plan["assignments"] if a.get("layers")]
    log(f"plan {plan['id'][:8]}: feasible {plan['feasible']}, tier {plan.get('tier')}, stages {stages}")
    if not plan["feasible"] and layout is None:
        plan = smallest_share(pool, plan) or plan
    if not plan["feasible"]:
        log(f"reasons {plan.get('reasons')}")
        return 1
    if plan.get("tier") in DISK_TIERS and not ALLOW_DISK:
        log(f"refusing tier {plan.get('tier')}: it pages weights from disk; pass --allow-disk-tiers to run it anyway")
        return 1
    token = httpx.get(f"{BASE}/api/owner/bootstrap", timeout=10).json()
    token = token.get("token") or token.get("owner_token")
    hdr = {"X-Hypercompute-Owner": token}
    peers = sorted({a["node_id"] for a in plan["assignments"] if a.get("layers") and a["node_id"] != "local"})
    t0 = time.monotonic()
    jobs: dict[str, str | None] = {}
    for node in peers:  # a 409 means the peer is still busy with an earlier seed: wait for it, then ask again
        r = None
        for _ in range(40):
            remaining = SEED_TIMEOUT - (time.monotonic() - t0)
            if remaining <= 0:
                break
            r = httpx.post(f"{BASE}/api/peers/{node}/seed", headers=hdr,
                           json={"model_id": MODEL, "plan_id": plan["id"]}, timeout=min(60, remaining))
            if r.status_code != 409:
                break
            remaining = SEED_TIMEOUT - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(min(15, remaining))
        jobs[node] = r.json().get("job_id") if r is not None and r.status_code < 300 else None
        log(f"seed {node[:6]}: HTTP {r.status_code}" if r is not None else f"seed {node[:6]}: deadline reached")
    state: dict[str, str] = {n: ("pending" if j else "failed") for n, j in jobs.items()}
    while time.monotonic() - t0 < SEED_TIMEOUT and "pending" in state.values():
        rows = httpx.get(f"{BASE}/api/peers", headers=hdr, timeout=30).json()
        rows = {p["node_id"]: p for p in (rows if isinstance(rows, list) else rows.get("peers", []))}
        for node, job in jobs.items():
            if state[node] != "pending":
                continue
            state[node] = seed_outcome(job, rows.get(node) or {})
        if any(s.startswith("failed") for s in state.values()):
            break
        if "pending" in state.values():
            remaining = SEED_TIMEOUT - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(min(15, remaining))
    pending_failure = ("failed (another peer's seed failed)" if any(s.startswith("failed") for s in state.values())
                       else "failed (seed timed out)")
    state = {n: (pending_failure if s == "pending" else s) for n, s in state.items()}
    log(f"seeding finished after {time.monotonic() - t0:.0f} s: "
        f"{ {n[:6]: s for n, s in state.items()} }")
    if any(s.startswith("failed") for s in state.values()):
        log("seeding did not finish successfully; no deployment started")
        return 1
    t1 = time.monotonic()
    dep = pool.deploy(plan["id"])
    completed = False
    try:
        pool.wait_ready(dep["id"], on_progress=lambda s, p: log(f"  {s}" + (f": {p}" if p else "")))
        log(f"ready after {time.monotonic() - t1:.0f} s")
        question = [{"role": "user", "content": "In one sentence, what is pipeline parallelism?"}]
        msg = pool.chat(question, max_tokens=MAX_TOKENS,
                        temperature=0, chat_template_kwargs={"thinking": False})["choices"][0]["message"]
        text, thought = (msg.get("content") or "").strip(), msg.get("reasoning_content") or ""
        log(f"chat: {text[:300]!r} (reasoning {len(thought)} chars)")
        if not text:
            raise RuntimeError("chat returned no visible answer "
                               f"({len(thought)} reasoning chars); increase --max-tokens and try again")
        last = None
        out = []
        reasoning_chars = 0
        ask = [{"role": "user", "content": "Count from one to ten, separated by commas."}]
        for chunk in pool.stream(ask, max_tokens=MAX_TOKENS,
                                 temperature=0, chat_template_kwargs={"thinking": False}):
            out.append(chunk.content or "")
            reasoning_chars += len(chunk.reasoning or "")
            last = chunk
        streamed = ''.join(out).strip()
        log(f"stream: {streamed[:200]} (reasoning {reasoning_chars} chars)")
        if not streamed:
            raise RuntimeError("stream returned no visible answer "
                               f"({reasoning_chars} reasoning chars); increase --max-tokens and try again")
        if last is not None and last.timings:
            t = last.timings
            log(f"decode {t.get('predicted_per_second'):.3f} tok/s over {t.get('predicted_n')} tokens, "
                f"prefill {t.get('prompt_per_second'):.2f} tok/s (this run only)")
        d = pool.deployment(dep["id"])
        log(f"deployment devices {(d.get('plan') or {}).get('device_list')}")
        completed = True
    finally:
        if KEEP_RUNNING and completed:
            log(f"kept {dep['id'][:8]} ready for the demo")
            log('stop on the host: PYTHONPATH=sdk/python/src uv run python -c '
                f'"from hypercompute_sdk import Pool; Pool().stop(\'{dep["id"]}\')"')
        else:
            stopped = pool.stop(dep["id"])
            log(f"stopped {dep['id'][:8]}: {stopped.get('state') if stopped else 'gone'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
