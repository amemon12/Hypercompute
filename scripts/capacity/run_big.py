"""End-to-end big-model run through a Hypercompute gateway: plan, explain, deploy, chat, observe, stop.

  uv run --no-sync python scripts/capacity/run_big.py --model qwen3-next-80b-a3b-instruct-q3_k_m --port 8001 \
      --home C:/Users/operator/.hypercompute/iroh-session --strategy all_selected --ctx 4096

Every step goes through the gateway's HTTP API on 127.0.0.1 with the owner token from /api/owner/bootstrap (the
loopback-only endpoint the CLI uses; the token is never printed or recorded), so the run exercises the same planner,
optimizer, admission, deployment manager and llama adapter a user does:
  1. preflight: the gateway has the capacity routes, no deployment is active, the model is downloaded and verified
     (this runner never downloads: POST /api/models/<id>/download on the gateway is a separate, deliberate step).
  2. POST /api/plans (strategy, ctx) and GET /api/capacity/explain for the same workload: predictions.
     With --layout the plan request carries an explicit layout (PlanRequest.layout: every stage named, the gateway
     only checks it, services/layout.py), so a fixed placement is measured through the same path; a layout the
     gateway rejects ends the run as refused with its reason.
     With --seed-peers, each peer the plan gives layers (joined with --managed --max-disk) is then asked to fill its
     llama.cpp RPC cache with exactly those layers' tensors from Hugging Face (POST /api/peers/<node>/seed with the
     plan id), and GET /api/peers is polled until every seed is done or failed, or --seed-timeout passes (then the
     rest are cancelled). A failed seed is recorded and the load goes ahead; a plan that aged past what the gateway
     deploys is planned again first.
  3. POST /api/deployments with that plan id, then poll until ready, failed or --load-timeout, printing progress.
  4. Observed placement, participation, engine flags (MoE placement flags included) and the engine log's buffers.
  5. Streamed chats through /v1/chat/completions: warmups (which also size the prompt to --prompt-tokens from the
     engine's prompt_n), then --reps measured reps with exactly --output-tokens output tokens (ignore_eos,
     cache_prompt false, a unique prefix per rep). Decode is (predicted_n - 1) / predicted_ms from the engine timings.
     With --concurrency N (> 1) the plan request carries parallel N (llama-server -np N -kvu: N slots sharing one KV
     pool of --ctx cells, so N x (measured prompt_n + a few tokens of slack + output tokens) must fit --ctx); a run
     whose engine argv lacks --parallel N ends slots_short before any chat. After the sequential sizing warmups
     (tags shaped like the batch tags), one unmeasured warm batch of N chats, then --reps measured batches: N worker
     threads, a client each, released together by a barrier; requests are labelled b<k>r<j>. A measured prompt_n
     that overfills the pool ends pool_short before the batch it would overfill, which matrix does not retry. Each
     batch records its all-active window, the tokens inside it (from each text delta's arrival time), aggregate
     decode tok/s, the step time and SSD bytes once per window (batchlib.py); a batch whose window is too short is
     not_concurrent.
  6. hcstream stats (stream tier) and each peer's worker CPU seconds before and after the measured reps.
  7. DELETE the deployment unless --keep (also after a failure or Ctrl+C).
A background monitor (hostmon.py) samples host memory, system-wide disk reads, the system drive's free space (where
the Windows pagefile grows) and GPU memory for the whole run. Its guard stops the deployment when available RAM or
the system drive's free space falls below the floors, or when the commit limit grows (a pagefile is growing), so a
run cannot fill C: with pagefile growth. On macOS it watches the models volume and swap used instead (no commit
counters there), and available RAM is free plus file-backed pages from vm_stat, so mapped weights count as available;
run.json records the platform and which guards could act.

Evidence: results/capacity/<UTC stamp>-big-<model>/run.json. "requested" holds the inputs, "predicted" the gateway's
plan and explain output verbatim, "measured" everything observed during this run, each block naming its source, and
"comparison" puts predicted and measured numbers side by side under names that say which is which.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

from batchlib import Span, batch_metrics, decode_start, run_together  # noqa: E402
from batchlib import summarize as summarize_batches  # noqa: E402
from hostmon import (  # noqa: E402
    GIB,
    HostProfile,
    Monitor,
    Probes,
    Sample,
    counter_delta,
    engine_flags,
    host_profile,
    log_evidence,
    probes_for,
    window,
)
from hypercompute import config  # noqa: E402
from hypercompute.capacity import runtime as stream_runtime  # noqa: E402
from hypercompute.capacity.benchlib import BenchError, Timings, parse_timings  # noqa: E402
from hypercompute.sse import SSEParser  # noqa: E402

SCHEMA = "hypercompute.capacity.big/1"
RESULTS = ROOT / "results" / "capacity"
TERMINAL = frozenset({"stopped", "failed"})
STRATEGIES = ("all_selected", "fastest_fit", "auto", "local_only")
SEED_HOST_STOPPED = frozenset({"refused", "failed", "cancelled"})  # the host's seed job ended before the peer had it
SEED_PEER_STOPPED = frozenset({"failed", "cancelled"})
SEED_REPORT_S = 60.0  # a progress line per peer this often while seeding
LAYOUT_KEYS = frozenset({"node_id", "device", "layers", "residency"})  # schemas.LayoutStage
LAYOUT_REQUIRED = frozenset({"node_id", "device", "layers"})
RESIDENCIES = (None, "resident", "file_backed")
WORDS = ("the river carries small stones past a quiet mill where an old engineer writes careful notes about gears "
         "light rain the price of bread and the long road that leads over the hills to a busy market town").split()
CHAT_SETTINGS = {"stream": True, "temperature": 0, "seed": 7, "cache_prompt": False, "ignore_eos": True}
MAX_CONCURRENCY = 16  # PlanRequest.parallel's bound
BARRIER_TIMEOUT_S = 60.0  # the batch workers' wait for each other before sending (each opens its own client first)
POOL_SLACK_TOKENS = 4  # per slot: the extrapolated batch word count and the random hex tag move prompt_n a little


class RunRefused(RuntimeError):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


class PoolShort(RunRefused):
    """The KV pool cannot hold N of the measured prompts. Found after the load, and the same prompts would overfill it
    again, so retrying cannot help (matrix retries only infeasible and refused)."""


class GatewayError(RuntimeError):
    def __init__(self, status: int, code: str | None, detail: str, fix: str | None = None,
                 locs: tuple[str, ...] = ()) -> None:
        super().__init__(f"HTTP {status}{' ' + code if code else ''}: {detail}")
        self.status, self.code, self.fix = status, code, fix
        self.locs = locs  # FastAPI validation: each failing field, e.g. body.parallel


class LoadTimeout(RuntimeError):
    def __init__(self, dep: Mapping[str, Any], seconds: float) -> None:
        super().__init__(f"deployment {str(dep.get('id'))[:8]} still {dep.get('state')} after {seconds:.0f} s")
        self.dep = dict(dep)


@dataclass(frozen=True)
class Options:
    model: str
    port: int = 8001
    strategy: str = "all_selected"
    ctx: int = 4096
    nodes: tuple[str, ...] = ()
    engine: str | None = None
    allow_capacity_mode: bool = True
    prompt_tokens: int = 128
    output_tokens: int = 64
    warmup: int = 1
    reps: int = 3
    keep: bool = False
    plan_only: bool = False
    load_timeout_s: float = 7200.0
    chat_timeout_s: float = 1800.0
    poll_s: float = 2.0
    heartbeat_wait_s: float = 2.5 * config.HEARTBEAT_INTERVAL_S
    sample_interval_s: float = 0.5
    gpu_every: int = 4
    min_available_gib: float = 1.0
    min_drive_free_gib: float = 5.0
    max_commit_growth_gib: float = 0.25  # guard: the commit limit grows only when a pagefile grows
    max_swap_growth_gib: float = 2.0  # guard (macOS): swap used grows only when private memory is past RAM
    seed_peers: bool = False
    seed_timeout_s: float = 3600.0
    seed_poll_s: float = 5.0
    replan_after_s: float = 480.0  # the gateway deploys only plans under 10 minutes old (DeploymentManager.stored_plan)
    layout: tuple[dict[str, Any], ...] | None = None  # --layout: explicit stages for PlanRequest.layout
    concurrency: int = 1  # --concurrency: PlanRequest.parallel, and chats sent in batches of this many
    home: Path | None = None
    results_dir: Path = RESULTS


@dataclass(frozen=True)
class Env:
    clock: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep
    say: Callable[[str], None] = lambda msg: print(msg, flush=True)
    # a fresh client to the gateway for each batch worker (--concurrency N > 1); one client is not shared across threads
    new_client: Callable[[], httpx.Client] | None = None


# ----------------------------------------------------------------------------- gateway client


def _refusal(r: httpx.Response) -> GatewayError:
    try:
        body = r.json()
    except ValueError:
        return GatewayError(r.status_code, None, r.text[:300])
    if not isinstance(body, dict):
        return GatewayError(r.status_code, None, str(body)[:300])
    detail, locs = body.get("detail"), ()
    if isinstance(detail, list):  # FastAPI request validation: keep which field failed (body.parallel: ...)
        items = [d for d in detail if isinstance(d, dict)]
        locs = tuple(".".join(map(str, d.get("loc") or [])) for d in items)
        detail = "; ".join(f"{loc}: {d.get('msg')}" if loc else str(d.get("msg")) for loc, d in zip(locs, items,
                                                                                                    strict=True))
    if detail is None and isinstance(body.get("error"), dict):
        detail = body["error"].get("message")
    return GatewayError(r.status_code, body.get("code"), str(detail), body.get("fix"), locs)


@dataclass(frozen=True)
class Gateway:
    client: httpx.Client
    token: str = field(repr=False)  # the owner token never appears in a repr, a log line or the evidence

    def owner_headers(self) -> dict[str, str]:
        return {config.OWNER_HEADER: self.token}

    def call(self, method: str, path: str, *, owner: bool = False, headers: Mapping[str, str] | None = None,
             **kwargs: Any) -> Any:
        merged = {**(self.owner_headers() if owner else {}), **(headers or {})}
        try:
            r = self.client.request(method, path, headers=merged, **kwargs)
        except httpx.HTTPError as exc:
            raise GatewayError(0, "unreachable", f"{type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise _refusal(r)
        return r.json()


def connect(client: httpx.Client) -> Gateway:
    """The owner token, read the way the CLI reads it: GET /api/owner/bootstrap from loopback."""
    try:
        r = client.get("/api/owner/bootstrap")
    except httpx.HTTPError as exc:
        raise GatewayError(0, "unreachable", f"{type(exc).__name__}: {exc}") from exc
    if r.status_code != 200:
        raise _refusal(r)
    body = r.json()
    token = body.get("owner_token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise GatewayError(r.status_code, "no_owner_token", "the gateway handed out no owner token")
    return Gateway(client, token)


# ----------------------------------------------------------------------------- preflight, model, plan


def node_summaries(nodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What each node reported: state, devices and memory pools from its inventory, worker CPU seconds."""
    out = []
    for n in nodes:
        inv = n.get("inventory") or {}
        out.append({"id": n.get("id"), "display_name": n.get("display_name"), "is_local": n.get("is_local"),
                    "state": n.get("state"), "state_reason": n.get("state_reason"),
                    "worker_state": n.get("worker_state"), "worker_cpu_seconds": n.get("worker_cpu_seconds"),
                    "heartbeat_age_s": n.get("heartbeat_age_s"), "contribution": n.get("contribution"),
                    "memory_pools": inv.get("memory_pools"),
                    "engine_devices": (inv.get("engine") or {}).get("devices")})
    return out


def preflight(gw: Gateway) -> dict[str, Any]:
    health = gw.call("GET", "/api/health")
    try:
        pools: Any = gw.call("GET", "/api/capacity/pools")
    except GatewayError as exc:
        if exc.status == 404:
            raise RunRefused("this gateway has no /api/capacity routes, so it predates the capacity code (890f798?)",
                             "restart the host from the worktree: docs/research/big-model-runbook.md") from exc
        pools = {"error": str(exc)}
    busy = [d for d in gw.call("GET", "/api/deployments") if d.get("state") not in TERMINAL]
    if busy:
        raise RunRefused(f"deployment {busy[0]['id'][:8]} is {busy[0]['state']}; this runner never stops a deployment "
                         "it did not create", "stop it from the dashboard or DELETE /api/deployments/<id> first")
    return {"health": health, "pools": pools, "nodes": node_summaries(gw.call("GET", "/api/nodes")),
            "network": gw.call("GET", "/api/network")}


def check_model(gw: Gateway, model_id: str) -> dict[str, Any]:
    """The gateway's view of the model; refuses unless it is downloaded and its sha256 verified."""
    view = next((v for v in gw.call("GET", "/api/models") if (v.get("artifact") or {}).get("id") == model_id), None)
    if view is None:
        raise RunRefused(f"model {model_id!r} is not in the gateway's catalog",
                         "start the host with HYPERCOMPUTE_CONFIG_DIR set to the worktree's configs directory")
    if view.get("status") != "downloaded" or not view.get("verified_sha256"):
        done, total = int(view.get("downloaded_bytes") or 0), int(view.get("total_bytes") or 0)
        # Download through the gateway, so the file lands in the gateway's HYPERCOMPUTE_HOME (the CLI's pull writes
        # to its own shell's home, which on this host may be a different directory on the same small C: drive).
        raise RunRefused(f"model {model_id} is {view.get('status')} ({done / GIB:.2f} of {total / GIB:.2f} GiB, "
                         f"verified {bool(view.get('verified_sha256'))})",
                         f"POST /api/models/{model_id}/download on this gateway with the owner header (or the "
                         "dashboard's download), wait for downloaded + verified_sha256, then rerun")
    return view


def plan_body(opts: Options) -> dict[str, Any]:
    body: dict[str, Any] = {"model_id": opts.model, "strategy": opts.strategy, "ctx_size": opts.ctx,
                            "allow_ram_tier": True, "allow_capacity_mode": opts.allow_capacity_mode}
    if opts.nodes:
        body["node_ids"] = list(opts.nodes)
    if opts.engine:
        body["engine"] = opts.engine
    if opts.layout:
        body["layout"] = [dict(stage) for stage in opts.layout]
    if opts.concurrency != 1:
        body["parallel"] = opts.concurrency
    return body


def pool_problem(opts: Options, prompt_n: int, slack: int = 0) -> str | None:
    """Why N slots of prompt_n + slack + output_tokens do not fit the unified KV pool of --ctx cells, or None. Every
    slot keeps its cells until the batch ends (--cache-ram 0 also turns off idle-slot clearing), and a full pool makes
    llama-server halve its batch and purge slots, so the batch would not be N wide."""
    need = opts.concurrency * (prompt_n + slack + opts.output_tokens)
    if need <= opts.ctx:
        return None
    extra = f" + {slack} slack" if slack else ""
    return (f"{opts.concurrency} slots x ({prompt_n} prompt{extra} + {opts.output_tokens} output tokens) = {need} "
            f"cells do not fit the KV pool of --ctx {opts.ctx}")


def _check_pool(opts: Options, prompt_n: int, refusal: type[RunRefused] = RunRefused) -> None:
    """Refuses (refusal: PoolShort once the model is loaded) when N prompts of prompt_n, plus the slack, overfill the
    KV pool."""
    why = pool_problem(opts, prompt_n, POOL_SLACK_TOKENS)
    if why is not None:
        need = opts.concurrency * (prompt_n + POOL_SLACK_TOKENS + opts.output_tokens)
        raise refusal(why, f"raise --ctx to at least {need}, or lower --concurrency or --prompt-tokens")


def _measured_prompt_n(results: Iterable[ChatResult]) -> list[int]:
    return [r.timings.prompt_n for r in results if r.ok and r.timings]


def _layout_stage(i: int, stage: Any) -> dict[str, Any]:
    if not isinstance(stage, dict) or not LAYOUT_REQUIRED <= set(stage) or set(stage) - LAYOUT_KEYS:
        raise ValueError(f"--layout stage {i}: give node_id, device and layers (and optionally residency), nothing "
                         "else")
    layers = stage["layers"]
    if (not isinstance(stage["node_id"], str) or not isinstance(stage["device"], str) or isinstance(layers, bool)
            or not isinstance(layers, int) or layers < 1):
        raise ValueError(f"--layout stage {i}: node_id and device are strings, layers a whole number of at least 1")
    if stage.get("residency") not in RESIDENCIES:
        raise ValueError(f"--layout stage {i}: residency is resident or file_backed")
    return dict(stage)


def parse_layout(value: str) -> tuple[dict[str, Any], ...]:
    """--layout: a JSON list of stages, inline or in a file ({"stages": [...]} works too), each {"node_id": <peer id or
    "local">, "device": "MTL0", "layers": 4, "residency": "file_backed" (optional)}, in execution order. Only what
    cannot be a layout is refused here; the gateway checks the rest (devices, order, memory, disk, hcpeer)."""
    text = value.strip()
    if not text.startswith(("[", "{")):
        try:
            text = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"--layout is neither inline JSON nor a readable file ({exc})") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--layout is not valid JSON ({exc})") from exc
    stages = data.get("stages") if isinstance(data, dict) else data
    if not isinstance(stages, list) or not stages:
        raise ValueError('--layout must be a non-empty JSON list of stages, or {"stages": [...]}')
    return tuple(_layout_stage(i, stage) for i, stage in enumerate(stages))


def layout_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The explicit layout as the gateway planned it: the stages it was given and what each device holds."""
    keys = ("node_id", "device", "endpoint", "layers", "file_backed", "weight_bytes", "total_bytes_est", "budget_bytes")
    return {"source": "the gateway's plan for --layout (memory figures are its estimates)",
            "stages": plan.get("layout"), "tier": plan.get("tier"), "device_list": plan.get("device_list"),
            "tensor_split": plan.get("tensor_split"),
            "assignments": [{k: a.get(k) for k in keys} for a in plan.get("assignments") or []
                            if isinstance(a, Mapping)]}


def explain(gw: Gateway, opts: Options) -> dict[str, Any]:
    params = {"model_id": opts.model, "ctx_size": opts.ctx, "prompt_tokens": opts.prompt_tokens,
              "output_tokens": opts.output_tokens}
    try:
        return gw.call("GET", "/api/capacity/explain", params=params, timeout=180.0)
    except GatewayError as exc:
        return {"error": str(exc), "fix": exc.fix}


# ----------------------------------------------------------------------------- deployment


def create_deployment(gw: Gateway, plan_id: str, key: str) -> dict[str, Any]:
    return gw.call("POST", "/api/deployments", owner=True, json={"plan_id": plan_id}, headers={"Idempotency-Key": key})


def find_by_key(gw: Gateway, key: str) -> str | None:
    """The deployment a create with this idempotency key made, when its reply was lost (timeout, Ctrl+C)."""
    try:
        deps = gw.call("GET", "/api/deployments")
    except GatewayError:
        return None
    return next((str(d["id"]) for d in deps if d.get("idempotency_key") == key and d.get("id")), None)


def wait_ready(gw: Gateway, dep_id: str, opts: Options, monitor: Monitor, env: Env) -> tuple[dict[str, Any], float]:
    """Poll until ready, terminal or a guard trip; (deployment, seconds from create to that poll)."""
    t0, last, failures = env.clock(), None, 0
    while True:
        try:
            dep = gw.call("GET", f"/api/deployments/{dep_id}")
            failures = 0
        except GatewayError:
            failures += 1
            if failures >= 5:
                raise
            env.sleep(opts.poll_s)
            continue
        if dep.get("pid"):
            monitor.attach(int(dep["pid"]))
        line = (dep.get("state"), dep.get("progress"))
        if line != last:
            last = line
            env.say(f"[{env.clock() - t0:8.1f} s] {dep.get('state')}"
                    + (f": {dep['progress']}" if dep.get("progress") else ""))
        if dep.get("state") == "ready" or dep.get("state") in TERMINAL or monitor.tripped:
            return dep, env.clock() - t0
        if env.clock() - t0 > opts.load_timeout_s:
            raise LoadTimeout(dep, env.clock() - t0)
        env.sleep(opts.poll_s)


def placement_record(dep: Mapping[str, Any]) -> dict[str, Any]:
    plan = dep.get("plan") or {}
    return {"source": "the gateway's deployment record (placement parsed from the engine's load log)",
            "state": dep.get("state"), "error": dep.get("error"), "observed": dep.get("placement"),
            "participation_verified": dep.get("participation_verified"),
            "participation_detail": dep.get("participation_detail"), "engine_args": dep.get("engine_args"),
            "engine": engine_flags(dep.get("engine_args") or []), "pid": dep.get("pid"),
            "tier": plan.get("tier"), "history": dep.get("history")}


def stop_deployment(gw: Gateway, dep_id: str) -> dict[str, Any]:
    try:
        dep = gw.call("DELETE", f"/api/deployments/{dep_id}", owner=True, timeout=300.0)
        return {"state": dep.get("state"), "cleanup": dep.get("cleanup"), "error": dep.get("error")}
    except GatewayError as exc:
        return {"error": str(exc)}


# ----------------------------------------------------------------------------- peer cache seeding (--seed-peers)


def plan_peer_layers(plan: Mapping[str, Any]) -> dict[str, list[int]]:
    """The layers a plan gives each peer (every node but the host, "local"), in plan order."""
    out: dict[str, list[int]] = {}
    for a in plan.get("assignments") or []:
        node = a.get("node_id") if isinstance(a, Mapping) else None
        if isinstance(node, str) and node != "local" and a.get("layers"):
            out = {**out, node: sorted({*out.get(node, []), *a["layers"]})}
    return out


def start_seed(gw: Gateway, node_id: str, model_id: str, plan_id: str, now: float) -> dict[str, Any]:
    """Ask the gateway to seed this peer with its layers of the plan. A refusal fails that peer's seed, not the run."""
    try:
        job = gw.call("POST", f"/api/peers/{node_id}/seed", owner=True, json={"model_id": model_id, "plan_id": plan_id},
                      timeout=60.0)
    except GatewayError as exc:
        return {"state": "failed", "started": now, "seconds": 0.0, "error": str(exc), "fix": exc.fix}
    return {"state": "pending", "started": now, "job_id": job.get("job_id"), "host_job": job}


def _peer_progress(row: Mapping[str, Any], job_id: str | None) -> dict[str, Any]:
    """The peer's own report on this seed job (its later reports may be about a newer job)."""
    report = row.get("report") if isinstance(row.get("report"), Mapping) else {}
    seed = report.get("seed") if isinstance(report.get("seed"), Mapping) else {}
    return dict(seed) if job_id is not None and seed.get("job_id") == job_id else {}


def seed_outcome(job_id: str | None, row: Mapping[str, Any] | None) -> tuple[str, str | None]:
    """("pending" | "done" | "failed", reason) of one seed job, from the peer's GET /api/peers row: the host's job
    (hashing, queued, delivered or stopped) and the peer's progress report for the same job id."""
    if row is None:
        return "failed", "the peer is no longer listed (it left, or the host restarted)"
    req = row.get("seed_request") if isinstance(row.get("seed_request"), Mapping) else {}
    if job_id is None or req.get("job_id") != job_id:
        return "failed", "the host no longer tracks this seed job (the peer rejoined, or another seed replaced it)"
    if req.get("state") in SEED_HOST_STOPPED:
        return "failed", f"host job {req.get('state')}: {req.get('detail')}"
    progress = _peer_progress(row, job_id)
    if progress.get("state") == "done":
        return "done", None
    if progress.get("state") in SEED_PEER_STOPPED:
        return "failed", f"peer job {progress.get('state')}: {progress.get('error')}"
    return "pending", None


def _advance(entry: dict[str, Any], row: Mapping[str, Any] | None, now: float) -> dict[str, Any]:
    if entry["state"] != "pending":
        return entry
    state, reason = seed_outcome(entry.get("job_id"), row)
    req = row.get("seed_request") if row is not None and isinstance(row.get("seed_request"), Mapping) else None
    progress = _peer_progress(row, entry.get("job_id")) if row is not None else {}
    seen = {**entry, "host_job": req or entry.get("host_job"), "progress": progress or entry.get("progress")}
    if state == "pending":
        return seen
    return {**seen, "state": state, "error": reason, "seconds": round(now - entry["started"], 3)}


def _end_pending(entries: Mapping[str, dict[str, Any]], state: str, reason: str,
                 now: float) -> dict[str, dict[str, Any]]:
    return {n: ({**e, "state": state, "error": reason, "seconds": round(now - e["started"], 3)}
                if e["state"] == "pending" else e) for n, e in entries.items()}


def _cancel_pending(gw: Gateway, entries: Mapping[str, dict[str, Any]], opts: Options,
                    now: float) -> dict[str, dict[str, Any]]:
    """Seeds still running at --seed-timeout are cancelled, so their downloads do not share the peer's link with the
    load; finished tensors stay cached and part files resume on the next seed."""
    out: dict[str, dict[str, Any]] = {}
    for node, e in entries.items():
        if e["state"] != "pending":
            out = {**out, node: e}
            continue
        try:
            gw.call("POST", f"/api/peers/{node}/seed/cancel", owner=True)
            cancel = "cancel queued"
        except GatewayError as exc:
            cancel = f"cancel failed: {exc}"
        out = {**out, node: {**e, "state": "timeout", "seconds": round(now - e["started"], 3),
                             "error": f"not done after --seed-timeout {opts.seed_timeout_s:.0f} s; {cancel}"}}
    return out


def _say_seeds(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]], env: Env,
               every: bool) -> None:
    for node, e in after.items():
        p = e.get("progress") or {}
        done = (f"{int(p.get('bytes_done') or 0) / GIB:.1f} of {int(p.get('bytes_total') or 0) / GIB:.1f} GiB, "
                f"{p.get('tensors_done')}/{p.get('tensors_total')} tensors")
        if e["state"] != before[node]["state"]:
            env.say(f"seed {node[:8]}: {e['state']} after {e.get('seconds')} s ({done})"
                    + (f": {e['error']}" if e.get("error") else ""))
        elif every and e["state"] == "pending":
            env.say(f"seed {node[:8]}: {(e.get('host_job') or {}).get('state')}, {done}")


def wait_seeds(gw: Gateway, entries: dict[str, dict[str, Any]], opts: Options, env: Env) -> dict[str, dict[str, Any]]:
    """Poll GET /api/peers until every started seed is done or failed, or --seed-timeout passes."""
    t0, failures, reported = env.clock(), 0, env.clock()
    while any(e["state"] == "pending" for e in entries.values()):
        if env.clock() - t0 > opts.seed_timeout_s:
            return _cancel_pending(gw, entries, opts, env.clock())
        env.sleep(opts.seed_poll_s)
        try:
            rows = {str(r.get("node_id")): r for r in gw.call("GET", "/api/peers") if isinstance(r, Mapping)}
            failures = 0
        except GatewayError as exc:
            failures += 1
            if failures >= 5:
                return _end_pending(entries, "failed", f"stopped polling the gateway: {exc}", env.clock())
            continue
        after = {n: _advance(e, rows.get(n), env.clock()) for n, e in entries.items()}
        every = env.clock() - reported >= SEED_REPORT_S
        reported = env.clock() if every else reported
        _say_seeds(entries, after, env, every)
        entries = after
    return entries


def seed_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    p = entry.get("progress") or {}
    return {"state": entry["state"], "seconds": entry.get("seconds"), "job_id": entry.get("job_id"),
            **{k: p.get(k) for k in ("bytes_done", "bytes_total", "tensors_total", "tensors_done", "tensors_present",
                                     "tensors_failed")},
            "failures": p.get("failures") or [], "error": entry.get("error"), "fix": entry.get("fix"),
            "host_job": entry.get("host_job")}


def seed_section(plan: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]], seconds: float,
                 opts: Options) -> dict[str, Any]:
    peers = {n: seed_record(e) for n, e in entries.items()}
    return {"source": "POST /api/peers/<node>/seed with the plan id, then GET /api/peers every seed_poll_s: the host's "
                      "seed job (hashing, queued, delivered) and each peer's own progress report. seconds: client "
                      "clock from the POST to the first poll that saw the end, host-side hashing included; bytes_done "
                      "and tensors_done count tensors that were already cached (tensors_present)",
            "plan_id": plan.get("id"), "timeout_s": opts.seed_timeout_s, "seconds": round(seconds, 3),
            "done": sum(1 for p in peers.values() if p["state"] == "done"),
            "failed": sum(1 for p in peers.values() if p["state"] != "done"), "peers": peers}


def _fresh_plan(gw: Gateway, opts: Options, env: Env, plan: dict[str, Any], doc: dict[str, Any],
                age_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The gateway deploys only plans under 10 minutes old, so after a long seed the same request is planned again.
    A peer whose layers moved loads them over the network, as it would without seeding; run.json names it."""
    if age_s <= opts.replan_after_s:
        return doc, plan
    try:
        fresh = gw.call("POST", "/api/plans", owner=True, json=plan_body(opts), timeout=180.0)
    except GatewayError as exc:
        _keep(exc, doc)
        raise
    before, after = plan_peer_layers(plan), plan_peer_layers(fresh)
    moved = sorted(n for n in {*before, *after} if before.get(n) != after.get(n))
    env.say(f"plan {str(fresh.get('id'))[:8]} replaces {str(plan.get('id'))[:8]} ({age_s:.0f} s old after seeding); "
            f"peers whose layers moved: {', '.join(n[:8] for n in moved) or 'none'}")
    replan = {"reason": f"the seeded plan was {age_s:.0f} s old; the gateway deploys plans under 600 s old",
              "seeded_plan_id": plan.get("id"), "plan_id": fresh.get("id"), "peers_with_moved_layers": moved}
    doc = _put(_put(doc, "predicted", "seed_plan", plan), "predicted", "plan", fresh)
    return _put(doc, "measured", "seed", {**doc["measured"]["seed"], "replan": replan}), fresh


def _seed_phase(gw: Gateway, opts: Options, env: Env, monitor: Monitor, plan: dict[str, Any], doc: dict[str, Any],
                planned_at: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """--seed-peers: each peer of the plan fills its RPC cache with its layers before the deployment is created."""
    layers, t0 = plan_peer_layers(plan), env.clock()
    if not layers:
        env.say("--seed-peers: the plan places no layers on a peer; nothing to seed")
        return _put(doc, "measured", "seed", {**seed_section(plan, {}, 0.0, opts), "note": "no peer layers"}), plan
    env.say(f"seeding {len(layers)} peers' RPC caches from Hugging Face (timeout {opts.seed_timeout_s:.0f} s)")
    monitor.mark("seed:start")
    entries = {n: start_seed(gw, n, opts.model, str(plan.get("id")), env.clock()) for n in layers}
    try:
        entries = wait_seeds(gw, entries, opts, env)
    except BaseException as exc:  # Ctrl+C: the seeds started so far, then the run ends
        _keep(exc, _put(doc, "measured", "seed", {**seed_section(plan, entries, env.clock() - t0, opts),
                                                  "interrupted": True}))
        raise
    monitor.mark("seed:end")
    doc = _put(doc, "measured", "seed", seed_section(plan, entries, env.clock() - t0, opts))
    return _fresh_plan(gw, opts, env, plan, doc, env.clock() - planned_at)


# ----------------------------------------------------------------------------- chats


@dataclass(frozen=True)
class ChatResult:
    label: str
    ok: bool
    error: str | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    finish_reason: str | None = None
    content_chars: int = 0
    reasoning_chars: int = 0
    timings: Timings | None = None
    t_start: float | None = None  # monitor clock
    t_end: float | None = None
    # seconds after the send at which each text delta arrived (client clock); feeds the batch math, not the rep rows
    delta_s: tuple[float, ...] = ()

    def measured(self) -> dict[str, Any]:
        t = self.timings
        base = {k: v for k, v in asdict(self).items() if k not in ("timings", "delta_s")}
        if t is None:
            return base
        return {**base, "prompt_n": t.prompt_n, "prompt_ms": t.prompt_ms, "predicted_n": t.predicted_n,
                "predicted_ms": t.predicted_ms, "cache_n": t.cache_n, "prefill_tok_s": t.prefill_tok_s,
                "decode_tok_s": t.decode_tok_s, "decode_ms_per_token": t.decode_ms_per_token}


def build_prompt(tag: str, n_words: int) -> str:
    body = " ".join(WORDS[i % len(WORDS)] for i in range(max(n_words, 1)))
    return f"Run {tag}. Continue this text in plain English: {body}"


def next_words(points: Sequence[tuple[int, int]], target: int) -> int:
    """Words for the next prompt from (words, measured prompt_n) points: a line through the last two distinct points
    (template overhead + tokens per word), else proportional scaling from the last one."""
    distinct = dict(points)
    if len(distinct) >= 2:
        (w1, p1), (w2, p2) = sorted(distinct.items())[-2:]
        rate = (p2 - p1) / (w2 - w1)
        if rate > 0:
            return max(1, round(w2 + (target - p2) / rate))
    w, p = points[-1]
    return max(1, round(w * target / p)) if p > 0 else w


def read_stream(chunks: Iterable[bytes], t0: float, clock: Callable[[], float]) -> dict[str, Any]:
    """Fold a gateway chat SSE stream: first-text time, each text delta's arrival (seconds after t0), text sizes,
    finish reason, last engine timings, error."""
    parser = SSEParser()
    out: dict[str, Any] = {"ttft_s": None, "content": 0, "reasoning": 0, "finish": None, "timings": None,
                           "error": None, "done": False}
    arrivals: list[float] = []  # one per content or reasoning delta
    for chunk in chunks:
        for ev in parser.feed(chunk):
            if ev.data.strip() == "[DONE]":
                return {**out, "done": True, "deltas": tuple(arrivals)}
            payload = json.loads(ev.data)
            if not isinstance(payload, dict):
                continue
            if payload.get("error"):
                err = payload["error"]
                return {**out, "error": err.get("message") if isinstance(err, dict) else str(err),
                        "deltas": tuple(arrivals)}
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
                if content or reasoning:
                    arrivals.append(clock() - t0)
                    if out["ttft_s"] is None:
                        out = {**out, "ttft_s": arrivals[0]}
                out = {**out, "content": out["content"] + len(content), "reasoning": out["reasoning"] + len(reasoning),
                       "finish": choice.get("finish_reason") or out["finish"]}
            if payload.get("timings"):
                out = {**out, "timings": payload}
    return {**out, "deltas": tuple(arrivals)}


def chat_once(gw: Gateway, prompt: str, opts: Options, label: str, clock: Callable[[], float]) -> ChatResult:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": opts.output_tokens, **CHAT_SETTINGS}
    t0 = clock()
    try:
        with gw.client.stream("POST", "/v1/chat/completions", json=body, headers=gw.owner_headers(),
                              timeout=httpx.Timeout(30.0, read=opts.chat_timeout_s)) as r:
            if r.status_code != 200:
                r.read()
                return ChatResult(label, False, error=str(_refusal(r)), total_ms=(clock() - t0) * 1000)
            folded = read_stream(r.iter_bytes(), t0, clock)
    except (httpx.HTTPError, ValueError) as exc:
        return ChatResult(label, False, error=f"{type(exc).__name__}: {exc}", total_ms=(clock() - t0) * 1000)
    ttft = folded["ttft_s"] * 1000 if folded["ttft_s"] is not None else None
    common = {"ttft_ms": ttft, "total_ms": (clock() - t0) * 1000, "finish_reason": folded["finish"],
              "content_chars": folded["content"], "reasoning_chars": folded["reasoning"], "delta_s": folded["deltas"]}
    if folded["error"]:
        return ChatResult(label, False, error=folded["error"], **common)
    try:
        timings = parse_timings(folded["timings"] or {})
    except BenchError as exc:
        return ChatResult(label, False, error=str(exc), **common)
    return ChatResult(label, True, timings=timings, **common)


def _timed_chat(gw: Gateway, prompt: str, opts: Options, label: str, monitor: Monitor, env: Env) -> ChatResult:
    start = monitor.now()
    res = chat_once(gw, prompt, opts, label, env.clock)
    res = ChatResult(**{**asdict(res), "timings": res.timings, "t_start": start, "t_end": monitor.now()})
    if not res.ok:
        env.say(f"{label}: failed: {res.error}")
    elif res.timings is not None:
        t = res.timings
        env.say(f"{label}: prompt_n {t.prompt_n}, predicted_n {t.predicted_n}, ttft {res.ttft_ms or 0:.0f} ms, "
                f"prefill {t.prefill_tok_s or 0:.1f} tok/s, decode {t.decode_tok_s or 0:.2f} tok/s")
    return res


def size_prompt(gw: Gateway, opts: Options, monitor: Monitor, env: Env, warm: list[ChatResult],
                tag: str = "w") -> int:
    """Sequential warmups that size the prompt to --prompt-tokens (at most two extra when it is still off target),
    appended to warm as each finishes; the words for the measured prompts. Each prompt is tagged <tag><i>-<hex6>, so
    a batch run sizes with tags shaped like its batch tags. Stops early when the guard trips."""
    words, points = max(1, int(opts.prompt_tokens * 0.75)), []
    tolerance, base = max(4, opts.prompt_tokens // 20), max(opts.warmup, 1)
    for i in range(base + 2):
        if i >= base and points and abs(points[-1][1] - opts.prompt_tokens) <= tolerance:
            break
        res = _timed_chat(gw, build_prompt(f"{tag}{i}-{uuid.uuid4().hex[:6]}", words), opts, f"warmup{i}", monitor,
                          env)
        warm.append(res)
        if res.ok and res.timings:
            points.append((words, res.timings.prompt_n))
            words = next_words(points, opts.prompt_tokens)
        if monitor.tripped:
            break
    return words


def run_chats(gw: Gateway, opts: Options, monitor: Monitor, env: Env, warm: list[ChatResult],
              reps: list[ChatResult]) -> None:
    """Fills warm and reps as each chat finishes, so a Ctrl+C keeps the finished ones. Warmups size the prompt."""
    words = size_prompt(gw, opts, monitor, env, warm)
    if monitor.tripped:
        return
    for i in range(opts.reps):
        reps.append(_timed_chat(gw, build_prompt(f"r{i}-{uuid.uuid4().hex[:6]}", words), opts, f"rep{i}", monitor,
                                env))
        if monitor.tripped:
            return


@dataclass
class BatchLog:
    """The batch requests of a --concurrency N run, appended as each batch finishes (a Ctrl+C keeps those)."""

    warm: list[ChatResult] = field(default_factory=list)
    measured: list[list[ChatResult]] = field(default_factory=list)


def run_batches(gw: Gateway, opts: Options, monitor: Monitor, env: Env, warm: list[ChatResult],
                reps: list[ChatResult], batches: BatchLog) -> None:
    """--concurrency N: the sequential sizing warmups, then one unmeasured warm batch of N, then --reps measured
    batches; each batch request also goes to reps (labels b<k>r<j>), so the chat summary counts it. The KV pool must
    hold N of the prompts sizing measured (they can overshoot --prompt-tokens), plus the slack, before any batch is
    sent, and N of the warm batch's measured prompts before the measured batches; pool_short otherwise."""
    words = size_prompt(gw, opts, monitor, env, warm, tag="warmup-s")  # the shape of warmup-b<j>-<hex6>
    if monitor.tripped:
        return
    _check_pool(opts, max([opts.prompt_tokens, *_measured_prompt_n(warm)]), PoolShort)
    batches.warm.extend(run_batch(gw, opts, monitor, env, words, [f"warmup-b{j}" for j in range(opts.concurrency)]))
    if _measured_prompt_n(batches.warm):  # the batch template, measured: b<k>r<j> tags are no longer
        _check_pool(opts, max(_measured_prompt_n(batches.warm)), PoolShort)
    for k in range(opts.reps):
        if monitor.tripped:
            return
        group = run_batch(gw, opts, monitor, env, words, [f"b{k}r{j}" for j in range(opts.concurrency)])
        batches.measured.append(group)
        reps.extend(group)


def run_batch(gw: Gateway, opts: Options, monitor: Monitor, env: Env, words: int,
              labels: Sequence[str]) -> list[ChatResult]:
    """One batch: a worker thread per label, each with its own gateway client and unique prompt prefix, all sent
    together once every worker is ready (batchlib.run_together); results in label order."""
    new_client = env.new_client
    assert new_client is not None  # _prepare refuses a --concurrency run without one before anything loads

    def task(label: str) -> Callable[[Callable[[], None]], ChatResult]:
        prompt = build_prompt(f"{label}-{uuid.uuid4().hex[:6]}", words)

        def send(start: Callable[[], None]) -> ChatResult:
            with new_client() as client:
                start()
                return _timed_chat(Gateway(client, gw.token), prompt, opts, label, monitor, env)

        return send

    def failed(i: int, exc: Exception) -> ChatResult:
        env.say(f"{labels[i]}: failed: {type(exc).__name__}: {exc}")
        return ChatResult(labels[i], False, error=f"{type(exc).__name__}: {exc}")

    return run_together([task(label) for label in labels], failed, BARRIER_TIMEOUT_S)


# ----------------------------------------------------------------------------- evidence


def _stats(values: Iterable[float | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals)}


def _host_fields(r: ChatResult, samples: Sequence[Sample], shared: bool = False) -> dict[str, Any]:
    """The monitor window over one chat, and system-wide disk reads per token: over the whole chat (prefill included,
    per output token) and over decode only (end - predicted_ms to the end, per decode step: the batch decode span),
    which is what an SSD-per-token estimate predicts. A batch request (shared) gets no per-token reads: the N windows
    overlap, so each would count every byte of the batch; the batch record counts them once."""
    if r.t_start is None or r.t_end is None:
        return {}
    win, t = window(samples, r.t_start, r.t_end), r.timings
    if shared:
        return {"host": win, "disk_read_bytes_per_output_token": None, "decode_disk_read_bytes_per_token": None}
    reads = win["disk_read_bytes"]
    whole = reads / t.predicted_n if reads is not None and t is not None and t.predicted_n else None
    decode = None
    first = decode_start(r.t_start, r.t_end, t.predicted_ms if t else None, r.ttft_ms)
    if t is not None and t.decode_steps and r.ttft_ms is not None and first is not None:
        reads = counter_delta(samples, "disk_read_bytes", first, r.t_end)
        decode = reads / t.decode_steps if reads is not None else None
    return {"host": win, "disk_read_bytes_per_output_token": whole, "decode_disk_read_bytes_per_token": decode}


def _span(r: ChatResult) -> Span | None:
    """A request that finished and streamed text, on the monitor clock (decode start = end - engine predicted_ms,
    never before the send; delta arrivals = sent + each delta's offset), or None."""
    t = r.timings
    if not r.ok or t is None or r.ttft_ms is None or r.t_start is None or r.t_end is None:
        return None
    first = max(r.t_start, r.t_end - t.predicted_ms / 1000)
    return Span(r.label, r.t_start, first, r.t_end, t.predicted_n, tuple(r.t_start + d for d in r.delta_s) or None)


def batch_record(label: str, results: Sequence[ChatResult], samples: Sequence[Sample]) -> dict[str, Any]:
    """One batch from its requests' monitor-clock timestamps (batchlib.batch_metrics), disk reads from the samples
    over its all-active window; incomplete unless every request finished with engine timings."""
    spans = [_span(r) for r in results]
    done = [s for s in spans if s is not None]
    if not results or len(done) < len(results):
        return {"label": label, "requests": [r.label for r in results], "status": "incomplete",
                "failed": [r.label for r, s in zip(results, spans, strict=True) if s is None]}
    return {"label": label, **batch_metrics(done, lambda t0, t1: counter_delta(samples, "disk_read_bytes", t0, t1))}


def batch_section(batches: BatchLog, samples: Sequence[Sample], concurrency: int) -> dict[str, Any]:
    records = [batch_record(f"b{k}", group, samples) for k, group in enumerate(batches.measured)]
    return {"concurrency": concurrency,
            "batch_source": "client clock per request (sent after a barrier, each text delta's arrival, end; the "
                            "monitor's clock) and engine predicted_n and predicted_ms (scripts/capacity/batchlib.py): "
                            "a request's decode span is [end - predicted_ms, end], the engine's own first-token-to-"
                            "last-token duration ending at the client end (the first visible delta is not used, "
                            "because gpt-oss harmony tokens are not streamed); W is the window in which every request "
                            "decodes, tokens inside W are the text deltas that arrive in it (basis token_times; "
                            "even_spread spreads predicted_n - 1 over the decode span when delta times are missing), "
                            "and SSD reads are counted once over W; the per-request rows above are per sequence (each "
                            "request's engine timings), not throughput",
            "warm_batch": batch_record("warm", batches.warm, samples) if batches.warm else None,
            "batches": records, "batch_summary": summarize_batches(records)}


def chat_section(reps: Sequence[ChatResult], warm: Sequence[ChatResult], samples: Sequence[Sample],
                 batches: BatchLog | None = None, concurrency: int = 1) -> dict[str, Any]:
    shared = concurrency > 1
    rep_rows = [{**r.measured(), **_host_fields(r, samples, shared)} for r in reps]
    rows = [*({**r.measured(), **_host_fields(r, samples)} for r in warm),
            *({**r.measured(), **_host_fields(r, samples, shared)} for r in (batches.warm if batches else ())),
            *rep_rows]
    ok = [r for r in reps if r.ok and r.timings]
    measured_rows = [row for row in rep_rows if row.get("ok")]
    summary = {"ok": len(ok), "failed": len(reps) - len(ok),
               "ttft_ms": _stats(r.ttft_ms for r in ok), "prompt_n": _stats(r.timings.prompt_n for r in ok),
               "prefill_tok_s": _stats(r.timings.prefill_tok_s for r in ok),
               "decode_tok_s": _stats(r.timings.decode_tok_s for r in ok),
               "decode_ms_per_token": _stats(r.timings.decode_ms_per_token for r in ok),
               "disk_read_bytes_per_output_token": _stats(row.get("disk_read_bytes_per_output_token")
                                                          for row in measured_rows),
               "decode_disk_read_bytes_per_token": _stats(row.get("decode_disk_read_bytes_per_token")
                                                          for row in measured_rows)}
    section = {"source": "engine timings (decode = (predicted_n - 1) / predicted_ms) and the client clock through the "
                         "gateway (ttft_ms, total_ms, streaming); host fields from the monitor over each chat, disk "
                         "reads system wide, bracketed by samples every interval_s", "reps": rows, "summary": summary}
    return {**section, **batch_section(batches or BatchLog(), samples, concurrency)} if shared else section


def worker_cpu_delta(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prior = {n["id"]: n.get("worker_cpu_seconds") for n in before}
    out = {}
    for n in after:
        a, b = n.get("worker_cpu_seconds"), prior.get(n["id"])
        if not n.get("is_local"):
            out[str(n["id"])] = {"display_name": n.get("display_name"), "before": b, "after": a,
                                 "delta_s": round(a - b, 3) if a is not None and b is not None else None}
    return {"source": "rpc-server process CPU seconds reported by each peer's heartbeats, read before and after the "
                      "measured reps (heartbeats are every 3 s, so the window is approximate)", "nodes": out}


def comparison(plan: Mapping[str, Any], chats: Mapping[str, Any], load_s: float | None) -> dict[str, Any]:
    """Predicted next to measured. The optimizer's score (plan.prediction) wins; a moe_offload plan's own estimates
    (plan.moe, all labelled estimates by the planner) are used when the plan carries no optimizer score."""
    pred, moe = plan.get("prediction") or {}, plan.get("moe") or {}
    summary = chats.get("summary") or {}
    predicted_decode = pred.get("decode_ms", moe.get("decode_ms_est"))
    measured_decode = summary.get("decode_ms_per_token", {}).get("median")
    ratio = measured_decode / predicted_decode if measured_decode and predicted_decode else None
    out = {"predicted_decode_ms": predicted_decode,
           "predicted_decode_source": "plan.prediction" if "decode_ms" in pred else
           ("plan.moe (estimate)" if predicted_decode is not None else None),
           "measured_decode_ms_per_token_median": measured_decode,
           "measured_over_predicted": round(ratio, 3) if ratio else None,
           "predicted_load_ms": pred.get("load_ms"), "measured_load_s": load_s,
           "predicted_stream_bytes_per_token": pred.get("stream_bytes_per_token"),
           "predicted_moe_ssd_bytes_per_token_est": moe.get("ssd_bytes_per_token_est"),
           "predicted_moe_miss_rate_est": moe.get("miss_rate_est"),
           "measured_disk_read_bytes_per_output_token_median":
               summary.get("disk_read_bytes_per_output_token", {}).get("median"),
           # the like-for-like number for predicted_*_bytes_per_token: decode only, no prompt processing reads
           "measured_decode_disk_read_bytes_per_token_median":
               summary.get("decode_disk_read_bytes_per_token", {}).get("median"),
           "note": "predicted_* come from the gateway's plan; measured_* from this run"}
    concurrency = chats.get("concurrency", 1)
    if concurrency == 1:
        return out
    batch = chats.get("batch_summary") or {}
    return {**out, "concurrency": concurrency,
            "predicted_decode_basis": "single stream, per sequence: the plan predicts one sequence's decode step; "
                                      "measured_decode_ms_per_token_median is each request's own engine figure",
            "measured_batch_step_ms_median": (batch.get("step_ms") or {}).get("median"),
            "measured_aggregate_decode_tok_s_median": (batch.get("aggregate_decode_tok_s") or {}).get("median"),
            "measured_ssd_bytes_per_step_median": (batch.get("ssd_bytes_per_step") or {}).get("median")}


def runner_source() -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        return {"error": "git not found"}
    try:
        rev = subprocess.run([git, "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=15, check=True).stdout.strip()
        dirty = subprocess.run([git, "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=15, check=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"git_head": rev, "uncommitted_paths": len(dirty),
            "note": "the runner's checkout; the gateway runs whatever code it was started from"}


def out_dir(results: Path, model: str, started: datetime) -> Path:
    base = f"{started.strftime('%Y%m%dT%H%M%SZ')}-big-{re.sub(r'[^A-Za-z0-9._-]', '_', model)}"
    for n in range(1, 1000):
        path = results / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"no free results directory for {base}")


def write_doc(path: Path, doc: Mapping[str, Any]) -> Path:
    target = path / "run.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(target)
    return target


# ----------------------------------------------------------------------------- orchestration


def _put(doc: Mapping[str, Any], section: str, key: str, value: Any) -> dict[str, Any]:
    return {**doc, section: {**doc.get(section, {}), key: value}}


def _logs_dir(opts: Options) -> Path:
    return (opts.home or config.paths().home) / "coordinator" / "logs"


def _prepare(gw: Gateway, opts: Options, env: Env, doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if opts.concurrency > 1:  # before anything loads: N target prompts must fit the KV pool
        _check_pool(opts, opts.prompt_tokens)
        if env.new_client is None:
            raise RunRefused("--concurrency needs one gateway client per batch worker (Env.new_client)")
    doc = _put(doc, "measured", "gateway_before", {**preflight(gw), "source": "the gateway's own state (inventory "
                                                   "and heartbeats are measured by the nodes; admissible is derived)"})
    view = check_model(gw, opts.model)
    doc = _put(doc, "measured", "model", {k: view.get(k) for k in ("status", "verified_sha256", "local_path",
                                                                   "total_bytes", "gguf")})
    try:
        plan = gw.call("POST", "/api/plans", owner=True, json=plan_body(opts), timeout=180.0)
    except GatewayError as exc:
        # a gateway without PlanRequest.parallel forbids the field (FastAPI loc body.parallel); a layout's own 422
        # stays a layout refusal, and without --layout any 422 is taken as the parallel field
        stale = exc.status == 422 and "body.parallel" in exc.locs
        if opts.concurrency > 1 and (stale or (exc.status == 422 and not opts.layout)):
            raise RunRefused(f"the gateway rejected parallel {opts.concurrency}: {exc}", exc.fix or (
                "restart the gateway from a checkout with the parallel code, and check its commit")) from exc
        if opts.layout and 400 <= exc.status < 500:  # the gateway's reason the layout cannot run (services/layout.py)
            raise RunRefused(f"the gateway rejected --layout: {exc}", exc.fix) from exc
        raise
    doc = _put(doc, "predicted", "plan", plan)
    if opts.layout:
        doc = _put(doc, "predicted", "layout", layout_record(plan))
    doc = _put(doc, "predicted", "explain", explain(gw, opts))
    env.say(f"plan {str(plan.get('id'))[:8]}: feasible {plan.get('feasible')}, tier {plan.get('tier')}, devices "
            f"{plan.get('device_list')}, predicted decode {(plan.get('prediction') or {}).get('decode_ms')} ms/token")
    return doc, plan


def _soft(gw: Gateway, path: str, fallback: Any, **kwargs: Any) -> Any:
    """A read whose failure must not cost the measurements already taken: the fallback, with the error noted."""
    try:
        return gw.call("GET", path, **kwargs)
    except GatewayError as exc:
        return {"error": str(exc)} if fallback is None else fallback


def _measure(gw: Gateway, opts: Options, env: Env, monitor: Monitor, dep: Mapping[str, Any],
             doc: dict[str, Any]) -> dict[str, Any]:
    stats_path = stream_runtime.config_paths(_logs_dir(opts), dep["id"])[1]
    stream_before = stream_runtime.stats_summary(stream_runtime.read_stats(stats_path))
    nodes_before = node_summaries(_soft(gw, "/api/nodes", []))
    monitor.mark("chats:start")
    warm: list[ChatResult] = []
    reps: list[ChatResult] = []
    batches = BatchLog()
    try:
        if opts.concurrency == 1:
            run_chats(gw, opts, monitor, env, warm, reps)
        else:
            run_batches(gw, opts, monitor, env, warm, reps, batches)
    except BaseException as exc:  # Ctrl+C: keep the chats that finished (a refusal keeps them too, not interrupted)
        partial = chat_section(reps, warm, monitor.samples, batches, opts.concurrency)
        _keep(exc, _put(doc, "measured", "chats", partial if isinstance(exc, RunRefused) else
                        {**partial, "interrupted": True}))
        raise
    monitor.mark("chats:end")
    # each peer's next heartbeat carries the CPU seconds spent in the reps, and a later sample closes the last chat
    env.sleep(opts.heartbeat_wait_s)
    chats = chat_section(reps, warm, monitor.samples, batches, opts.concurrency)
    doc = _put(doc, "measured", "chats", chats)
    nodes_after = node_summaries(_soft(gw, "/api/nodes", []))
    # a batch run records N chats per batch; the gateway keeps up to 500 records
    runs = [r for r in _soft(gw, "/api/runs", [], params={"limit": 500}) if r.get("deployment_id") == dep["id"]]
    doc = _put(doc, "measured", "peer_worker_cpu", worker_cpu_delta(nodes_before, nodes_after))
    doc = _put(doc, "measured", "hcstream", {"source": "hcstream stats file", "path": str(stats_path),
                                             "before": stream_before, "after": stream_runtime.stats_summary(
                                                 stream_runtime.read_stats(stats_path))})
    doc = _put(doc, "measured", "gateway_runs", {"source": "the gateway's run records for this deployment",
                                                 "runs": runs})
    doc = _put(doc, "measured", "gateway_after", {"nodes": nodes_after,
                                                  "network": _soft(gw, "/api/network", None)})
    return {**doc, "comparison": comparison(doc["predicted"].get("plan") or {}, chats,
                                            doc["measured"].get("load", {}).get("seconds"))}


def _keep(exc: BaseException, doc: dict[str, Any]) -> None:
    """Pins the evidence gathered so far to an exception that ends the run; the innermost (fullest) one wins."""
    if not hasattr(exc, "partial_doc"):
        exc.partial_doc = doc  # type: ignore[attr-defined]


def _deploy_and_measure(gw: Gateway, opts: Options, env: Env, monitor: Monitor, plan: Mapping[str, Any],
                        doc: dict[str, Any], dep_ref: list[str]) -> tuple[dict[str, Any], str]:
    if monitor.tripped:  # the host is already below a floor: never start loading into it
        return doc, "aborted_by_guard"
    monitor.mark("load:start")
    key = uuid.uuid4().hex
    try:
        dep = create_deployment(gw, plan["id"], key)
    except BaseException:
        found = find_by_key(gw, key)  # the gateway may have created it before the reply was lost: stop that one too
        if found:
            dep_ref.append(found)
        raise
    dep_ref.append(dep["id"])
    env.say(f"deployment {dep['id'][:8]} created")
    return _load_and_measure(gw, opts, env, monitor, dep["id"], doc)


def _load_and_measure(gw: Gateway, opts: Options, env: Env, monitor: Monitor, dep_id: str,
                      doc: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        dep, seconds = wait_ready(gw, dep_id, opts, monitor, env)
    except LoadTimeout as exc:
        doc = _put(doc, "measured", "placement", placement_record(exc.dep))
        return _put(doc, "measured", "load", {"seconds": None, "error": str(exc)}), "load_timeout"
    monitor.mark("load:end")
    doc = _put(doc, "measured", "load", {"seconds": round(seconds, 3), "source": "client clock from create to the "
                                         "first poll that saw the final state (poll interval bounds the error)"})
    doc = _put(doc, "measured", "placement", placement_record(dep))
    doc = _put(doc, "measured", "engine_log", log_evidence(_logs_dir(opts) / f"llama-server-{dep['id']}.log"))
    if monitor.tripped:
        return doc, "aborted_by_guard"
    if dep.get("state") != "ready":
        return doc, "load_failed"
    if opts.concurrency > 1:
        doc = _put(doc, "measured", "slots", slots_record(dep, opts.concurrency))
        if doc["measured"]["slots"]["short"]:
            env.say(f"the engine runs {doc['measured']['slots']['engine_parallel']} slot(s), not "
                    f"{opts.concurrency}: no chats (is the gateway running the patched code?)")
            return doc, "slots_short"
    try:
        doc = _measure(gw, opts, env, monitor, dep, doc)
    except BaseException as exc:
        _keep(exc, doc)
        raise
    if monitor.tripped:
        return doc, "aborted_by_guard"
    return doc, chat_status(doc["measured"]["chats"])


def chat_status(chats: Mapping[str, Any]) -> str:
    """ok, partial or chat_failed from the measured chats (an N>1 batch with a request that never streamed text or
    timings counts as failed, and chat_failed when no batch is complete); not_concurrent when every batch finished but
    one's all-active window was too short to be a concurrent batch."""
    summary, batch = chats["summary"], chats.get("batch_summary") or {}
    if summary["failed"] or batch.get("incomplete"):
        none_left = summary["ok"] == 0 or bool(batch) and batch["incomplete"] == batch["batches"]
        return "chat_failed" if none_left else "partial"
    return "not_concurrent" if batch.get("not_concurrent") else "ok"


def slots_record(dep: Mapping[str, Any], n: int) -> dict[str, Any]:
    """Whether the engine really runs N slots, from its parsed argv: a gateway without the parallel code launches
    -np 1 and admits one chat at a time, so a batch would quietly run serially."""
    flags = engine_flags(dep.get("engine_args") or [])["flags"]
    return {"source": "the engine argv on the gateway's deployment record", "requested": n,
            "engine_parallel": flags.get("--parallel"), "kv_unified": bool(flags.get("--kv-unified")),
            "short": flags.get("--parallel") != str(n)}


def _after_plan(gw: Gateway, opts: Options, env: Env, monitor: Monitor, plan: dict[str, Any], doc: dict[str, Any],
                dep_ref: list[str], planned_at: float) -> tuple[dict[str, Any], str]:
    if not plan.get("feasible"):
        return doc, "infeasible"
    if opts.plan_only:
        return doc, "plan_only"
    if opts.seed_peers and not monitor.tripped:  # a tripped guard aborts before the load anyway
        doc, plan = _seed_phase(gw, opts, env, monitor, plan, doc, planned_at)
        if not plan.get("feasible"):
            return doc, "infeasible"
    return _deploy_and_measure(gw, opts, env, monitor, plan, doc, dep_ref)


def _requested(opts: Options) -> dict[str, Any]:
    return {**{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(opts).items()},
            "chat_settings": CHAT_SETTINGS}


def run(opts: Options, client: httpx.Client, probes: Probes, env: Env | None = None,
        profile: HostProfile | None = None) -> tuple[Path, str]:
    """One run; always writes run.json and stops the deployment it created unless --keep. (path, status)."""
    env = env or Env()
    profile = profile or host_profile(sys.platform, opts.home or config.paths().home)
    started = datetime.now(UTC)
    doc: dict[str, Any] = {"schema": SCHEMA, "started_at": started.isoformat(), "requested": _requested(opts),
                           "runner": runner_source(), "predicted": {}, "measured": {}}
    dep_ref: list[str] = []
    state: dict[str, Any] = {"gw": None, "guard_stop": None, "profile": profile}

    def on_trip(reason: str) -> None:
        env.say(f"guard: {reason}; stopping the deployment")
        if state["gw"] is not None and dep_ref:
            state["guard_stop"] = stop_deployment(state["gw"], dep_ref[0])

    monitor = Monitor(probes, interval_s=opts.sample_interval_s, gpu_every=opts.gpu_every,
                      min_available=int(opts.min_available_gib * GIB),
                      min_drive_free=int(opts.min_drive_free_gib * GIB), on_trip=on_trip, clock=env.clock,
                      max_commit_growth=int(opts.max_commit_growth_gib * GIB),
                      max_swap_growth=int(opts.max_swap_growth_gib * GIB), drive_label=profile.drive_label,
                      drive_note=profile.drive_note).start()
    status, error = "error", None
    try:
        state["gw"] = connect(client)
        planned_at = env.clock()  # before the plan is made: an upper bound on its age
        doc, plan = _prepare(state["gw"], opts, env, doc)
        doc, status = _after_plan(state["gw"], opts, env, monitor, plan, doc, dep_ref, planned_at)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - recorded in run.json, then the deployment stops
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else (
            "pool_short" if isinstance(exc, PoolShort) else "refused" if isinstance(exc, RunRefused) else "error")
        error = {"type": type(exc).__name__, "message": str(exc), "fix": getattr(exc, "fix", None)}
        doc = getattr(exc, "partial_doc", doc)  # load, placement and finished chats measured before the failure
    finally:
        monitor.stop()
    return _finish(state, opts, env, monitor, doc, dep_ref, status, error, started)


def _finish(state: Mapping[str, Any], opts: Options, env: Env, monitor: Monitor, doc: dict[str, Any],
            dep_ref: list[str], status: str, error: dict[str, Any] | None, started: datetime) -> tuple[Path, str]:
    stop = None
    if state["gw"] is not None and dep_ref and not opts.keep:
        stop = stop_deployment(state["gw"], dep_ref[0])
        env.say(f"deployment {dep_ref[0][:8]} stop: {stop.get('state') or stop.get('error')}")
    samples, profile, guards = monitor.samples, state["profile"], monitor.guard_status()
    guards = {**guards, "min_drive_free": {**guards["min_drive_free"], "path": profile.drive}}
    doc = _put(doc, "measured", "host", {
        "source": profile.source, "platform": profile.platform, "guards": guards,
        "interval_s": opts.sample_interval_s, "marks": monitor.marks, "error": monitor.error,
        "guard_tripped": monitor.tripped, "guard_stop": state["guard_stop"],
        "whole_run": window(samples, 0.0, samples[-1].t) if samples else {"samples": 0},
        "samples": [asdict(s) for s in samples]})
    doc = {**doc, "status": status, "error": error, "deployment_id": dep_ref[0] if dep_ref else None, "stop": stop,
           "kept": bool(dep_ref) and opts.keep, "finished_at": datetime.now(UTC).isoformat()}
    path = write_doc(out_dir(opts.results_dir, opts.model, started), doc)
    env.say(f"{status}: {path}")
    return path, status


def parse_args(argv: Sequence[str] | None) -> Options:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument("--model", required=True, help="catalog id, e.g. qwen3-next-80b-a3b-instruct-q3_k_m")
    p.add_argument("--port", type=int, default=8001, help="gateway port on 127.0.0.1 (default 8001, the Iroh host)")
    p.add_argument("--strategy", choices=STRATEGIES, default="all_selected")
    p.add_argument("--node", action="append", default=[], help="peer node id to plan onto (repeatable; default all)")
    p.add_argument("--engine", default=None, help="force an engine (default: the gateway's selector)")
    p.add_argument("--no-capacity-mode", action="store_true", help="forbid disk-backed tiers in the plan")
    for flag, default in (("--ctx", 4096), ("--prompt-tokens", 128), ("--output-tokens", 64), ("--warmup", 1),
                          ("--reps", 3)):
        p.add_argument(flag, type=int, default=default)
    p.add_argument("--keep", action="store_true", help="leave the deployment running afterwards")
    p.add_argument("--plan-only", action="store_true", help="plan and explain, deploy nothing")
    p.add_argument("--load-timeout", type=float, default=7200.0, help="seconds to wait for ready")
    p.add_argument("--chat-timeout", type=float, default=1800.0, help="read timeout per chat, seconds")
    p.add_argument("--min-available-gib", type=float, default=1.0, help="guard: stop below this available RAM")
    p.add_argument("--min-drive-free-gib", type=float, default=5.0,
                   help="guard: stop below this free space on the system drive (Windows) or models volume (macOS)")
    p.add_argument("--max-commit-growth-gib", type=float, default=0.25,
                   help="guard: stop when the commit limit grows more than this (a pagefile is growing)")
    p.add_argument("--max-swap-growth-gib", type=float, default=2.0,
                   help="guard (macOS): stop when swap used grows more than this (private memory is past RAM)")
    p.add_argument("--home", type=Path, default=None, help="the gateway's HYPERCOMPUTE_HOME (engine logs, hcstream)")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--seed-peers", action="store_true",
                   help="before deploying, have each peer of the plan (joined with --managed --max-disk) download its "
                        "layers' tensors from Hugging Face into its RPC cache; a failed seed is recorded, not fatal")
    p.add_argument("--seed-timeout", type=float, default=3600.0,
                   help="seconds to wait for every peer's seed; seeds still running then are cancelled")
    p.add_argument("--layout", default=None, metavar="JSON_OR_PATH",
                   help='explicit layout, inline JSON or a file: [{"node_id": "<peer id>", "device": "MTL0", '
                        '"layers": 4}, {"node_id": "<peer id>", "device": "CPU", "layers": 6, "residency": '
                        '"file_backed"}, {"node_id": "local", "device": "CUDA0", "layers": 26}]; peers first, this '
                        "machine's GPU last; the gateway checks it and plans nothing else")
    p.add_argument("--concurrency", type=int, default=1, metavar="N",
                   help="llama.cpp server slots (-np N, unified KV) and chats sent N at a time: one unmeasured warm "
                        "batch, then --reps measured batches (default 1: one chat at a time, as before)")
    a = p.parse_args(argv)
    try:
        layout = parse_layout(a.layout) if a.layout is not None else None
    except ValueError as exc:
        p.error(str(exc))
    if a.reps < 1 or a.warmup < 0 or a.prompt_tokens < 8 or a.output_tokens < 2:
        p.error("--reps >= 1, --warmup >= 0, --prompt-tokens >= 8 and --output-tokens >= 2")
    if a.seed_timeout <= 0:
        p.error("--seed-timeout must be positive")
    if a.prompt_tokens + a.output_tokens > a.ctx:
        p.error("--prompt-tokens + --output-tokens must fit in --ctx")
    if not 1 <= a.concurrency <= MAX_CONCURRENCY:
        p.error(f"--concurrency is 1 to {MAX_CONCURRENCY}")
    if a.concurrency * (a.prompt_tokens + a.output_tokens) > a.ctx:
        p.error("--concurrency x (--prompt-tokens + --output-tokens) must fit in --ctx (one unified KV pool)")
    return Options(model=a.model, port=a.port, strategy=a.strategy, ctx=a.ctx, nodes=tuple(a.node), engine=a.engine,
                   allow_capacity_mode=not a.no_capacity_mode, prompt_tokens=a.prompt_tokens,
                   output_tokens=a.output_tokens, warmup=a.warmup, reps=a.reps, keep=a.keep, plan_only=a.plan_only,
                   load_timeout_s=a.load_timeout, chat_timeout_s=a.chat_timeout,
                   min_available_gib=a.min_available_gib, min_drive_free_gib=a.min_drive_free_gib,
                   max_commit_growth_gib=a.max_commit_growth_gib, max_swap_growth_gib=a.max_swap_growth_gib,
                   seed_peers=a.seed_peers, seed_timeout_s=a.seed_timeout, layout=layout,
                   concurrency=a.concurrency, home=a.home, results_dir=a.results_dir)


def gateway_client(port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=60.0)


def main(argv: Sequence[str] | None = None) -> int:
    opts = parse_args(argv)
    profile = host_profile(sys.platform, opts.home or config.paths().home)
    env = Env(new_client=lambda: gateway_client(opts.port))
    with gateway_client(opts.port) as client:
        _, status = run(opts, client, probes_for(profile), env, profile=profile)
    return {"ok": 0, "plan_only": 0, "refused": 2, "pool_short": 2, "interrupted": 130}.get(status, 1)


if __name__ == "__main__":
    sys.exit(main())
