"""Measured split tuner: run a few candidate plans for real, score them, and save the fastest as a profile.

Candidates (split_optimization.md §5b): the planner's own pick; when part of the model must leave the local GPU,
the other overflow tier too (host RAM vs peers), because their break-even depends on the measured link and RAM
speed (§3.4); and with tune_flags, flash attention and micro-batch size on the planner's split. The local GPU keeps
the output head, so no device-order variant is tried. A model that fits the local GPU gets no split variants: every
layer moved off it only adds latency (§3.5).

Every candidate is planned with the owner's own limits (engine, strategy, RAM tier, capacity mode), and the engine is
llama.cpp unless the request names another. Each candidate runs as a normal deployment marked purpose="tune", one at
a time, most conservative memory use first (§5c), and is always stopped before the next. Score in seconds per typical
turn, lower is better (N_p = 1024, N_g = 256 as in §5b): interactive = TTFT + 256 / decode_tps; throughput =
1024 / prefill_tps + 256 / decode_tps. A winner that is a llama.cpp split of the requested model is saved as a
measured profile under the key the selector looks it up by, so later plans for the same setup replay it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..adapters.base import EngineAdapterError, artifact_engines
from ..adapters.llama_rpc import engine_flag_args
from ..schemas import (
    TERMINAL_STATES,
    CreateDeploymentRequest,
    Deployment,
    DeploymentState,
    NodeState,
    Plan,
    PlanRequest,
    PlanStrategy,
    RunRecord,
    TuneCandidate,
    TuneRequest,
    TuneResult,
    utcnow,
)
from ..storage import Store, dumps
from . import planner
from .benchmarks import SHORT_PROMPT
from .deployments import DeploymentError, DeploymentManager, local_budgets, remote_budgets, warmup_timeout
from .events import EventBus
from .models import ModelError, ModelService
from .nodes import NodeRegistry
from .profiles import ProfileStore, signature
from .selector import profile_devices
from .streaming import InferenceService

log = logging.getLogger("hypercompute.tuner")

TYPICAL_PROMPT = 1024  # N_p
TYPICAL_GEN = 256  # N_g
DEFAULT_FLAGS: dict[str, str | int] = {"flash_attn": "auto", "ubatch": 512}  # llama-server at b11770 (common.h)
# Flag variants in priority order: the ones that differ from llama-server's defaults first ('on' often equals auto).
FLAG_VARIANTS: tuple[tuple[str, dict[str, str | int]], ...] = (
    ("flash attention off", {"flash_attn": "off"}), ("micro-batch 1024", {"ubatch": 1024}),
    ("micro-batch 256", {"ubatch": 256}), ("micro-batch 512", {"ubatch": 512}),
    ("flash attention on", {"flash_attn": "on"}))
SEED = 42
PROMPT_MARGIN = 64  # tokens kept free for the chat template around the decode prompt
PREFILL_TARGET = 1000  # tokens in the prefill prompt (fewer on a small context)
FACT_TOKENS = 24  # rough tokens per weather-log line, for sizing the prefill prompt
READY_SLACK_S = 60.0  # past the engine's own load and warmup budgets, which DeploymentManager enforces
SLOT_TIMEOUT_S = 300.0


class TuneAborted(Exception):
    """The whole tune must stop (another deployment took over, or a trial could not be stopped)."""


@dataclass
class Measurement:
    ttft_ms: float | None = None
    decode_tps: float | None = None
    prefill_tps: float | None = None
    run_ids: list[str] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- candidates


def with_flags(plan: Plan, flags: dict[str, str | int]) -> Plan:
    return plan.model_copy(update={"engine_flags": {**plan.engine_flags, **flags}})


def with_ubatch(plan: Plan, ubatch: int) -> Plan | None:
    """The plan at another -ub, each device's compute-buffer estimate rescaled the way
    planner.ModelShape.with_ubatch scales it; None when a device no longer fits its budget."""
    old = int(plan.engine_flags.get("ubatch", planner.UBATCH))
    out = []
    for a in plan.assignments:
        compute = a.compute_bytes_est
        if compute > planner.COMPUTE_BASE:
            compute = planner.COMPUTE_BASE + (compute - planner.COMPUTE_BASE) * ubatch // old
        total = a.total_bytes_est - a.compute_bytes_est + compute
        out.append(a.model_copy(update={"compute_bytes_est": compute, "total_bytes_est": total,
                                        "fits": total <= a.budget_bytes}))
    if not all(a.fits for a in out):
        return None
    return with_flags(plan.model_copy(update={"assignments": out}), {"ubatch": ubatch})


def flag_variants(base: Plan) -> list[tuple[str, Plan]]:
    """Flash attention and micro-batch variants of the planner's split that fit and that llama-server accepts."""
    out = []
    for label, flags in FLAG_VARIANTS:
        ub = flags.get("ubatch")
        plan = with_ubatch(base, int(ub)) if ub is not None else with_flags(base, flags)
        if plan is None:
            continue
        try:
            engine_flag_args(plan.engine_flags)  # e.g. a micro-batch above the plan's batch size
        except EngineAdapterError as exc:
            log.info("tuner skips %s: %s", label, exc)
            continue
        out.append((f"{label}: {layout(base)}", plan))
    return out


def config_key(plan: Plan) -> tuple:
    """What llama-server would actually run: unset flags count as their defaults, so duplicates collapse."""
    flags = {k: str(v) for k, v in {**DEFAULT_FLAGS, **plan.engine_flags}.items()}
    return (plan.engine, tuple(plan.device_list), tuple(plan.tensor_split), plan.cpu_layers,
            tuple(plan.rpc_endpoints), tuple(sorted(flags.items())))


def memory_key(plan: Plan) -> tuple[float, int, int]:
    """Lower = more conservative: tightest device's estimated use / budget, then micro-batch, then flash attention."""
    worst = max((a.total_bytes_est / a.budget_bytes for a in plan.assignments if a.budget_bytes > 0), default=0.0)
    flags = {**DEFAULT_FLAGS, **plan.engine_flags}
    fa_rank = {"on": 0, "auto": 1, "off": 2}.get(str(flags["flash_attn"]), 1)  # FA avoids the score matrix buffer
    return round(worst, 3), int(flags["ubatch"]), fa_rank


def overflows(plan: Plan) -> bool:
    """Part of the model is off the local GPU (host RAM tier or peers)."""
    return plan.cpu_layers > 0 or bool(plan.rpc_endpoints)


def head_on_peer(plan: Plan) -> bool:
    """The output head sits on a peer although a local GPU is in the plan: logits would cross the network."""
    local_gpu = any(a.endpoint is None and a.device != "CPU" for a in plan.assignments)
    return local_gpu and any(a.holds_output and a.endpoint for a in plan.assignments)


def layout(plan: Plan) -> str:
    return " + ".join(f"{a.device} x{len(a.layers)}" + ("+head" if a.holds_output else "") for a in plan.assignments)


def describe(plan: Plan) -> str:
    flags = ", ".join(f"{k}={v}" for k, v in sorted(plan.engine_flags.items()) if k in DEFAULT_FLAGS)
    return layout(plan) + (f" [{flags}]" if flags else "")


def select_candidates(options: list[tuple[str, Plan]], max_candidates: int) -> list[tuple[str, Plan]]:
    """Drop duplicate configurations, keep the first max_candidates by priority, run the safest first."""
    seen: set[tuple] = set()
    unique = []
    for label, plan in options:
        key = config_key(plan)
        if key not in seen:
            seen.add(key)
            unique.append((label, plan))
    return sorted(unique[:max_candidates], key=lambda lp: memory_key(lp[1]))


def replayable(plan: Plan, model_id: str) -> bool:
    """Plans replay only llama.cpp splits of the requested model (selector._with_profile), so only they are saved."""
    return plan.engine == "llama_rpc" and plan.model_id == model_id


def load_budget(plan: Plan, adapter: Any) -> float:
    """The engine's own load budget: adapter.load_timeout_s(plan), else the wait_ready default the manager falls back
    to (EXO and AirLLM download for hours), never less than llama.cpp's."""
    hook = getattr(adapter, "load_timeout_s", None)
    if hook is not None:
        return float(hook(plan))
    param = inspect.signature(adapter.wait_ready).parameters.get("timeout_s")
    default = param.default if param is not None else None
    return max(config.LOAD_TIMEOUT_S, float(default) if isinstance(default, int | float) else 0.0)


def ready_timeout(plan: Plan, adapter: Any) -> float:
    """How long a trial may take to become ready: its engine's load and warmup budgets plus slack. The manager enforces
    both, so this only catches a deployment that never settles; it must never cut off a load the engine allows."""
    return load_budget(plan, adapter) + warmup_timeout(plan, adapter) + READY_SLACK_S


# --------------------------------------------------------------------------- measurement helpers


def prefill_prompt(ctx_size: int) -> str:
    target = min(PREFILL_TARGET, (ctx_size - PROMPT_MARGIN) // 2)
    lines = [f"Fact {i}: station {i} recorded {(i * 37) % 101} millimetres of rain and a peak wind of "
             f"{(i * 13) % 61} km/h." for i in range(1, max(4, target // FACT_TOKENS) + 1)]
    return "Read the weather log below, then name the station with the most rain.\n\n" + "\n".join(lines)


def request_bodies(plan: Plan, req: TuneRequest) -> list[tuple[str, dict]]:
    """TTFT on a short prompt, decode rate over decode_tokens tokens, prefill rate on a ~1000-token prompt."""
    common: dict[str, Any] = {"model": plan.model_id, "temperature": 0, "seed": SEED}
    if plan.engine == "llama_rpc":
        common["cache_prompt"] = False  # measure real prompt processing, not prefix-cache hits
    short = [{"role": "user", "content": SHORT_PROMPT}]
    decode = {**common, "messages": short, "max_tokens": req.decode_tokens}
    if plan.engine == "llama_rpc":
        decode["ignore_eos"] = True  # exactly decode_tokens tokens (tools/server/server-schema.cpp:423 at b11770)
    long = [{"role": "user", "content": prefill_prompt(plan.ctx_size)}]
    return [("tune-ttft", {**common, "messages": short, "max_tokens": 16}), ("tune-decode", decode),
            ("tune-prefill", {**common, "messages": long, "max_tokens": 4})]


def decode_rate(run: RunRecord) -> float | None:
    """Engine timings first; else generated tokens after the first over the client-side streaming time."""
    t = run.timings
    if t and t.predicted_per_second:
        return t.predicted_per_second
    if t and t.predicted_n and t.predicted_n > 1 and t.predicted_ms:
        return (t.predicted_n - 1) / t.predicted_ms * 1000
    n = run.completion_tokens
    if n and n > 1 and run.total_ms and run.ttft_ms is not None and run.total_ms > run.ttft_ms:
        return (n - 1) / ((run.total_ms - run.ttft_ms) / 1000)
    return None


def prefill_rate(run: RunRecord) -> float | None:
    """Engine timings first; else prompt tokens over the client-side time to first token."""
    t = run.timings
    if t and t.prompt_per_second:
        return t.prompt_per_second
    if t and t.prompt_n and t.prompt_ms:
        return t.prompt_n / t.prompt_ms * 1000
    if run.prompt_tokens and run.ttft_ms:
        return run.prompt_tokens / (run.ttft_ms / 1000)
    return None


def score(objective: str, ttft_ms: float | None, decode_tps: float | None, prefill_tps: float | None
          ) -> float | None:
    """Seconds for a typical turn (lower is better), or None when a needed rate was not measured."""
    if not decode_tps:
        return None
    gen = TYPICAL_GEN / decode_tps
    if objective == "interactive":
        return None if ttft_ms is None else ttft_ms / 1000 + gen
    return None if not prefill_tps else TYPICAL_PROMPT / prefill_tps + gen


def measurement_error(m: Measurement, objective: str) -> str | None:
    if m.error:
        return m.error
    if m.decode_tps is None:
        return "no decode rate could be measured (no engine timings and too few tokens)"
    if objective == "interactive" and m.ttft_ms is None:
        return "no first token was observed on the short prompt"
    if objective == "throughput" and m.prefill_tps is None:
        return "no prefill rate could be measured"
    return None


# --------------------------------------------------------------------------- service


class TunerService:
    def __init__(self, store: Store, events: EventBus, deployments: DeploymentManager, inference: InferenceService,
                 profiles: ProfileStore | None, models: ModelService, nodes: NodeRegistry) -> None:
        self._store, self._events, self._deps, self._inf = store, events, deployments, inference
        self._profiles, self._models, self._nodes = profiles, models, nodes
        self._tasks: dict[str, asyncio.Task] = {}
        self._live: dict[str, TuneResult] = {}
        self._signatures: dict[str, str] = {}  # tune id -> profile signature of the devices it was planned with
        self._cancel_reason: dict[str, str] = {}
        self._start_lock = asyncio.Lock()
        self.poll_s = 0.5

    # ------------------------------------------------------------------ persistence / queries
    def _save(self, result: TuneResult) -> TuneResult:
        self._live[result.id] = result
        self._store.execute("INSERT INTO tunes (id, status, created_at, doc) VALUES (?,?,?,?) "
                            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, doc=excluded.doc",
                            (result.id, result.status, result.created_at.isoformat(),
                             dumps(result.model_dump(mode="json"))))
        return result

    def list(self) -> list[TuneResult]:
        rows = self._store.query("SELECT doc FROM tunes ORDER BY created_at DESC LIMIT 50")
        return [TuneResult.model_validate(json.loads(r["doc"])) for r in rows]

    def get(self, tune_id: str) -> TuneResult:
        row = self._store.query_one("SELECT doc FROM tunes WHERE id=?", (tune_id,))
        if row is None:
            raise DeploymentError(404, "not_found", "no such tune")
        return TuneResult.model_validate(json.loads(row["doc"]))

    def _running(self) -> str | None:
        return next((tid for tid, t in self._tasks.items() if not t.done()), None)

    def _publish(self, result: TuneResult, message: str, **data: object) -> None:
        self._events.publish("tune.progress", f"Tune {result.id[:8]}: {message}", tune_id=result.id,
                             status=result.status, **data)

    def _store_plan(self, plan: Plan) -> Plan:
        """Insert or refresh a candidate's plan row so deployments.create(plan_id=...) accepts it (10-minute rule)."""
        plan = plan.model_copy(update={"created_at": utcnow()})
        self._store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE "
                            "SET created_at=excluded.created_at, doc=excluded.doc",
                            (plan.id, plan.created_at.isoformat(), dumps(plan.model_dump(mode="json"))))
        return plan

    # ------------------------------------------------------------------ start
    async def start(self, req: TuneRequest) -> TuneResult:
        async with self._start_lock:
            running = self._running()
            if running is not None:
                raise DeploymentError(409, "tune_running", f"tune {running[:8]} is still running",
                                      fix="wait for it to finish or cancel it first")
            current = self._deps.active()
            if current is not None:
                raise DeploymentError(409, "deployment_active",
                                      f"deployment {current.id[:8]} is {current.state.value}; tuning needs the "
                                      "GPUs to itself", fix="stop the running model first, then tune")
            if req.decode_tokens + PROMPT_MARGIN > req.ctx_size:
                raise DeploymentError(422, "bad_request", f"decode_tokens {req.decode_tokens} does not fit a "
                                      f"{req.ctx_size}-token context", fix="use a larger ctx_size or fewer tokens")
            options = await asyncio.to_thread(self.candidate_plans, req)
            sig = self.profile_signature(req)
            candidates = [TuneCandidate(label=label, plan=self._store_plan(plan.model_copy(
                update={"id": uuid.uuid4().hex})), status="pending") for label, plan in options]
            result = self._save(TuneResult(id=uuid.uuid4().hex, request=req, status="running",
                                           candidates=candidates, created_at=utcnow()))
            self._signatures[result.id] = sig
            self._publish(result, f"started with {len(candidates)} candidate(s)", model_id=req.model_id)
            self._tasks[result.id] = asyncio.get_running_loop().create_task(self._run(result))
            return result

    def profile_signature(self, req: TuneRequest) -> str:
        """The key the selector looks profiles up by for this request: every device it can use (local GPUs and the
        selected peers' devices, selector.profile_devices), so a later plan on the same setup finds the winner. A
        local_only plan sees no peers (DeploymentManager.plan), so neither does its key."""
        local_node = self._nodes.local
        local = local_budgets(local_node) if local_node.state == NodeState.eligible else []
        peers = [] if req.strategy == PlanStrategy.local_only else self._nodes.eligible_remote(req.node_ids)
        remote = [b for n in peers for b in remote_budgets(n)]
        return signature(req.model_id, req.ctx_size, profile_devices(local, remote))

    @staticmethod
    def plan_request(req: TuneRequest) -> PlanRequest:
        """The owner's own plan limits, so every candidate is one their plans would accept and replay."""
        return PlanRequest(model_id=req.model_id, node_ids=req.node_ids, ctx_size=req.ctx_size, strategy=req.strategy,
                           allow_ram_tier=req.allow_ram_tier, allow_capacity_mode=req.allow_capacity_mode,
                           engine=req.engine or "llama_rpc", objective=req.objective, use_profiles=False)

    def candidate_plans(self, req: TuneRequest) -> list[tuple[str, Plan]]:
        """Planner pick, the other overflow tier when the model spills off the local GPU, then flag variants."""
        base = self._deps.plan(self.plan_request(req))
        if not base.feasible:
            raise DeploymentError(422, "plan_not_feasible", "; ".join(base.reasons) or "no plan fits",
                                  fix=self._infeasible_fix(req.model_id, base))
        if base.capacity_mode:
            raise DeploymentError(422, "capacity_mode", "only the disk tier (capacity mode) fits this model; it "
                                  "takes seconds to minutes per token, so there is nothing to tune",
                                  fix="add a peer or free memory so a GPU tier fits, then tune")
        options = [(f"planner pick ({base.tier}): {describe(base)}", base)]
        if base.engine != "llama_rpc":
            return options  # splits and llama.cpp flags only exist for the llama.cpp engine
        if overflows(base):
            options += self._overflow_alternatives(req, base)
        if req.tune_flags:
            options += flag_variants(base)
        return select_candidates(options, req.max_candidates)

    def _infeasible_fix(self, model_id: str, base: Plan) -> str:
        """Downloading or a peer cannot help when llama.cpp was asked to tune an HF or MLX artifact: say what can."""
        try:
            art = self._models.get(model_id) if self._models is not None else None
        except ModelError:
            art = None
        if base.engine != "llama_rpc" or art is None or art.format == "gguf":
            return "check the plan in the dashboard: download the model or add a peer"
        engines = [e for e in artifact_engines(art) if e in ("vllm", "exo")]  # AirLLM is capacity mode: nothing to tune
        return f"llama.cpp needs a GGUF file: tune a GGUF variant of {model_id}" + "".join(
            f", or tune it with engine {e} (--engine {e})" for e in engines)

    def _overflow_alternatives(self, req: TuneRequest, base: Plan) -> list[tuple[str, Plan]]:
        out = []
        tiers = (("host RAM overflow", "host_ram"),) if req.allow_ram_tier else ()
        for label, overflow in (*tiers, ("peer overflow", "peers")):
            alt = self._deps.plan(self.plan_request(req).model_copy(update={
                "allow_ram_tier": overflow == "host_ram", "allow_capacity_mode": False, "engine": base.engine,
                "overflow": overflow}))
            if alt.feasible and alt.engine == base.engine and not alt.capacity_mode and not head_on_peer(alt):
                out.append((f"{label} ({alt.tier}): {describe(alt)}", alt))
        return out

    # ------------------------------------------------------------------ run
    async def _run(self, result: TuneResult) -> None:
        tune_id = result.id
        try:
            for i in range(len(result.candidates)):
                result = await self._run_candidate(result, i)
            self._finish(result)
        except asyncio.CancelledError:
            self._close(self._live[tune_id], "cancelled", self._cancel_reason.pop(tune_id, "cancelled"))
            raise
        except TuneAborted as exc:
            self._close(self._live[tune_id], "failed", str(exc))
        except Exception as exc:  # a crash still gets a final status instead of 'running' forever
            log.exception("tune %s crashed", tune_id)
            self._close(self._live[tune_id], "failed", f"internal error: {exc}")
        finally:
            self._signatures.pop(tune_id, None)

    def _set(self, result: TuneResult, i: int, **update: object) -> TuneResult:
        cands = list(result.candidates)
        cands[i] = cands[i].model_copy(update=update)
        result = self._save(result.model_copy(update={"candidates": cands}))
        if "status" in update:
            c = cands[i]
            detail = f" ({c.error})" if c.error else (f" score {c.score:.2f} s" if c.score is not None else "")
            self._publish(result, f"candidate {i + 1}/{len(cands)} {c.status}: {c.label}{detail}", index=i,
                          candidate_status=c.status)
        return result

    async def _run_candidate(self, result: TuneResult, i: int) -> TuneResult:
        plan = self._store_plan(result.candidates[i].plan)
        result = self._set(result, i, status="running", plan=plan)
        dep_id: str | None = None
        try:
            started = time.perf_counter()
            try:
                ready_s = ready_timeout(plan, self._deps.adapter_for(plan.engine))
                dep_id = (await self._deps.create(CreateDeploymentRequest(plan_id=plan.id, purpose="tune"), None)).id
            except DeploymentError as exc:
                if exc.code == "deployment_active":
                    raise TuneAborted(f"another deployment started while tuning, so tuning stopped: {exc}") from exc
                return self._set(result, i, status="failed", error=str(exc))
            result = self._set(result, i, deployment_id=dep_id)
            dep = await self._wait_settled(dep_id, ready_s)
            if dep.state != DeploymentState.ready:
                return self._set(result, i, status="failed", error=dep.error or f"deployment {dep.state.value}")
            result = self._set(result, i, load_s=time.perf_counter() - started)
            m = await self._measure(dep_id, plan, result.request, result.id)
            return self._record(result, i, m)
        finally:
            await self._stop(dep_id, plan.id)

    def _record(self, result: TuneResult, i: int, m: Measurement) -> TuneResult:
        objective = result.request.objective
        error = measurement_error(m, objective)
        return self._set(result, i, status="failed" if error else "done", error=error, ttft_ms=m.ttft_ms,
                         decode_tokens_per_s=m.decode_tps, prefill_tokens_per_s=m.prefill_tps, run_ids=m.run_ids,
                         score=None if error else score(objective, m.ttft_ms, m.decode_tps, m.prefill_tps))

    async def _wait_settled(self, dep_id: str, timeout_s: float) -> Deployment:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            dep = self._deps.get(dep_id)
            if dep.state == DeploymentState.ready or dep.state in TERMINAL_STATES:
                return dep
            if loop.time() > deadline:
                return dep.model_copy(update={"error": f"not ready after {timeout_s:.0f}s (still {dep.state.value})"})
            await asyncio.sleep(self.poll_s)

    async def _measure(self, dep_id: str, plan: Plan, req: TuneRequest, tune_id: str) -> Measurement:
        """Three requests with every inference slot held (exclusive), so no chat or benchmark shares the engine. The
        trial is addressed by id: as a purpose="tune" deployment it is never offered as the ready run."""
        dep, handle = self._deps.get(dep_id), self._deps.handle(dep_id)
        if dep.state != DeploymentState.ready or handle is None:
            return Measurement(error=f"trial deployment is no longer ready (it is {dep.state.value})")
        try:
            hold = await self._inf.acquire(dep, exclusive=True, timeout=SLOT_TIMEOUT_S)
        except DeploymentError as exc:
            return Measurement(error=str(exc))
        m = Measurement()
        try:
            for label, body in request_bodies(plan, req):
                run = await self._inf.collect(dep, handle, body, "benchmark", tune_id, label)
                m.run_ids.append(run.id)
                if run.status != "ok":
                    m.error = f"{label} request {run.status}: {run.error}"
                    break
                if label == "tune-ttft":
                    m.ttft_ms = run.ttft_ms
                elif label == "tune-decode":
                    m.decode_tps = decode_rate(run)
                else:
                    m.prefill_tps = prefill_rate(run)
        finally:
            hold.release()
        return m

    async def _stop(self, dep_id: str | None, plan_id: str) -> None:
        """Stop the candidate's deployment; a cancel that landed inside create() still finds it by plan id."""
        if dep_id is None:
            active = self._deps.active()
            if active is None or active.plan.id != plan_id:
                return
            dep_id = active.id
        try:
            await asyncio.shield(self._deps.stop(dep_id, "tuning: candidate finished"))
        except asyncio.CancelledError:
            raise
        except Exception:  # the next create() then reports the deployment that is still active
            log.exception("stopping trial deployment %s failed", dep_id)

    # ------------------------------------------------------------------ finish / cancel / restart
    def _close(self, result: TuneResult, status: str, error: str | None, **update: object) -> TuneResult:
        """Final status; candidates that never finished are marked skipped."""
        cands = [c if c.status in ("done", "failed", "skipped") else
                 c.model_copy(update={"status": "skipped", "error": c.error or error}) for c in result.candidates]
        result = self._save(result.model_copy(update={"status": status, "error": error, "candidates": cands,
                                                      **update}))
        self._publish(result, status + (f" ({error})" if error else ""))
        return result

    def _finish(self, result: TuneResult) -> TuneResult:
        done = [(i, c) for i, c in enumerate(result.candidates) if c.status == "done" and c.score is not None]
        if not done:
            why = "; ".join(f"{c.label}: {c.error}" for c in result.candidates if c.error)
            return self._close(result, "failed", f"no candidate finished ({why or 'nothing ran'})")
        best_i, best = min(done, key=lambda ic: ic[1].score or 0.0)  # every score in `done` is set
        profile_id, note, req = None, None, result.request
        if not replayable(best.plan, req.model_id):
            note = (f"the winner runs {best.plan.engine} on {best.plan.model_id}, "
                    f"and plans replay only llama.cpp splits of {req.model_id}")
            log.info("tune %s: %s", result.id, note)
        elif self._profiles is not None:
            sig = self._signatures.get(result.id) or self.profile_signature(req)
            profile_id = self._profiles.save(
                model_id=req.model_id, sig=sig, objective=req.objective, plan=best.plan, ttft_ms=best.ttft_ms,
                decode_tps=best.decode_tokens_per_s, prefill_tps=best.prefill_tokens_per_s,
                score=best.score, tune_id=result.id, run_ids=best.run_ids).id  # type: ignore[arg-type]
        return self._close(result, "done", None, best_index=best_i, profile_id=profile_id, note=note)

    async def cancel(self, tune_id: str) -> TuneResult:
        result = self.get(tune_id)
        task = self._tasks.get(tune_id)
        if result.status != "running" or task is None or task.done():
            return result
        self._cancel_reason.setdefault(tune_id, "cancelled by owner")
        task.cancel()
        await asyncio.wait({task})  # its handler stops the trial deployment and records the final status
        return self.get(tune_id)

    async def shutdown(self) -> None:
        """Cancel a running tune so it stops its trial deployment and records a final status."""
        running = [(tid, t) for tid, t in self._tasks.items() if not t.done()]
        for tid, task in running:
            self._cancel_reason.setdefault(tid, "gateway shutting down")
            task.cancel()
        await asyncio.gather(*(t for _, t in running), return_exceptions=True)

    def reconcile_on_start(self) -> list[str]:
        """A tune still 'running' in the store was interrupted by a gateway restart."""
        notes = []
        for row in self._store.query("SELECT doc FROM tunes WHERE status='running'"):
            result = TuneResult.model_validate(json.loads(row["doc"]))
            self._close(result, "failed", "gateway restarted")
            notes.append(f"tune {result.id[:8]} marked failed after restart")
        return notes
