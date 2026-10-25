"""Warm-run benchmarks: fixed prompts, warmup, >=3 repetitions, concurrency 1, median and range.

Decode and prompt rates come from llama.cpp's own `timings`; TTFT is measured at the gateway.
Failed and timed-out repetitions are kept as runs and counted, never dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import uuid

from ..adapters.base import ENGINE_PROCESS_LABEL
from ..schemas import BenchmarkRequest, BenchmarkResult, BenchmarkSummary, Deployment, RunRecord, StatSummary, utcnow
from ..storage import Store, dumps
from .deployments import DeploymentError, DeploymentManager
from .events import EventBus
from .streaming import InferenceService

log = logging.getLogger("hypercompute.benchmarks")

SHORT_PROMPT = "In two sentences, explain why the sky looks blue during the day."
_LONG_FACTS = [
    f"Fact {i}: station {i} recorded {(i * 37) % 101} millimetres of rain and a peak wind of {(i * 13) % 61} km/h."
    for i in range(1, 121)
]
LONG_PROMPT = ("Read the weather log below, then report the three stations with the most rain and one sentence "
               "on any pattern you notice.\n\n" + "\n".join(_LONG_FACTS))
PROMPTS = {"short": SHORT_PROMPT, "long": LONG_PROMPT}


def stat(values: list[float]) -> StatSummary:
    if not values:
        return StatSummary(n=0, median=None, min=None, max=None)
    return StatSummary(n=len(values), median=statistics.median(values), min=min(values), max=max(values))


def summarize(label: str, runs: list[RunRecord]) -> BenchmarkSummary:
    ok = [r for r in runs if r.status == "ok"]

    def pick(get) -> list[float]:  # type: ignore[no-untyped-def]
        return [v for r in ok if (v := get(r)) is not None]

    return BenchmarkSummary(
        prompt_label=label,
        ttft_ms=stat(pick(lambda r: r.ttft_ms)),
        decode_tokens_per_s=stat(pick(lambda r: r.timings.predicted_per_second if r.timings else None)),
        prompt_tokens_per_s=stat(pick(lambda r: r.timings.prompt_per_second if r.timings else None)),
        completion_tokens=stat(pick(lambda r: float(r.completion_tokens) if r.completion_tokens is not None else None)),
        failures=len(runs) - len(ok),
    )


def placement_label(dep: Deployment) -> str:
    """Where the benchmarked model ran, e.g. "CPU (4 layers) + CUDA0 (~44 layers)" or "vLLM: CUDA0 (RTX ...)".

    Engines that place layers themselves (vLLM, AirLLM, EXO) report no per-device layer count."""
    if dep.placement:
        devices = [d for d in dep.placement.devices if d.model_bytes and not d.name.startswith(("CPU", "CUDA_Host"))]
        parts = [f"{d.name} (~{d.layers_est} layers)" if d.layers_est is not None else d.name for d in devices]
    else:
        parts = list(dep.plan.device_list)
    if dep.plan.cpu_layers:
        parts.insert(0, f"CPU ({dep.plan.cpu_layers} layers)")
    label = " + ".join(parts) or "no device reported"
    engine = ENGINE_PROCESS_LABEL.get(dep.plan.engine, dep.plan.engine)
    return label if dep.plan.engine == "llama_rpc" else f"{engine}: {label}"


class BenchmarkService:
    def __init__(self, store: Store, events: EventBus, deployments: DeploymentManager,
                 inference: InferenceService) -> None:
        self._store, self._events, self._deps, self._inf = store, events, deployments, inference
        self._tasks: dict[str, asyncio.Task] = {}

    def _save(self, result: BenchmarkResult) -> BenchmarkResult:
        self._store.execute("INSERT INTO benchmarks (id, created_at, doc) VALUES (?,?,?) "
                            "ON CONFLICT(id) DO UPDATE SET doc=excluded.doc",
                            (result.id, result.created_at.isoformat(), dumps(result.model_dump(mode="json"))))
        return result

    def list(self) -> list[BenchmarkResult]:
        rows = self._store.query("SELECT doc FROM benchmarks ORDER BY created_at DESC LIMIT 50")
        return [BenchmarkResult.model_validate(json.loads(r["doc"])) for r in rows]

    def get(self, bench_id: str) -> BenchmarkResult:
        row = self._store.query_one("SELECT doc FROM benchmarks WHERE id=?", (bench_id,))
        if row is None:
            raise DeploymentError(404, "not_found", "no such benchmark")
        return BenchmarkResult.model_validate(json.loads(row["doc"]))

    def start(self, req: BenchmarkRequest) -> BenchmarkResult:
        dep, _ = self._deps.ready()
        if req.deployment_id and req.deployment_id != dep.id:
            raise DeploymentError(409, "not_ready", "that deployment is not the ready one")
        label = placement_label(dep)
        result = BenchmarkResult(id=uuid.uuid4().hex, deployment_id=dep.id, label=req.label, status="running",
                                 request=req, run_ids=[], summaries=[], placement_label=label, created_at=utcnow())
        self._save(result)
        self._events.publish("benchmark.state", f"Benchmark {result.id[:8]} started", benchmark_id=result.id)
        self._tasks[result.id] = asyncio.get_running_loop().create_task(self._run(result))
        return result

    async def _run(self, result: BenchmarkResult) -> None:
        req = result.request
        labels = ["short", "long"] if req.prompt_set == "both" else [req.prompt_set]
        runs: dict[str, list[RunRecord]] = {label: [] for label in labels}
        run_ids: list[str] = []
        status, error = "failed", None
        try:
            dep, handle = self._deps.ready()
            for label in labels:
                body = {"model": dep.plan.model_id, "messages": [{"role": "user", "content": PROMPTS[label]}],
                        "max_tokens": req.max_tokens, "temperature": 0, "seed": 42,
                        "cache_prompt": False}  # measure real prompt processing, not prefix-cache hits
                for i in range(req.warmup + req.repetitions):
                    kind = "warmup" if i < req.warmup else "benchmark"
                    hold = await self._inf.acquire(dep, exclusive=True, timeout=300)  # alone on the engine
                    try:
                        self._still_ready(dep.id)
                        run = await self._inf.collect(dep, handle, body, kind, result.id, label)
                    finally:
                        hold.release()
                    run_ids.append(run.id)
                    if kind == "benchmark":
                        runs[label].append(run)
                    if run.status != "ok":
                        self._still_ready(dep.id)  # cut off by a stop: end here rather than finish 'done'
            status = "done"
        except asyncio.CancelledError:
            error = "cancelled before it finished (gateway shutting down)"
            raise
        except (DeploymentError, OSError) as exc:
            error = str(exc)
        except Exception as exc:  # any other failure still gets a final status instead of 'running' forever
            log.exception("benchmark %s crashed", result.id)
            error = f"internal error: {exc}"
        finally:
            summaries = [summarize(label, runs[label]) for label in labels]
            result = result.model_copy(update={"status": status, "run_ids": run_ids, "summaries": summaries,
                                               "error": error})
            self._save(result)
            self._events.publish("benchmark.state", f"Benchmark {result.id[:8]} {result.status}",
                                 benchmark_id=result.id)

    def _still_ready(self, dep_id: str) -> None:
        """The benchmarked deployment must still be the ready one: a stopped engine (or a new one on its port) is
        never measured, and the benchmark fails instead of finishing 'done' with every repetition failed."""
        try:
            current = self._deps.ready()[0].id
        except DeploymentError as exc:
            raise DeploymentError(409, "not_ready", f"deployment {dep_id[:8]} is no longer ready ({exc}); "
                                  "the benchmark stopped") from exc
        if current != dep_id:
            raise DeploymentError(409, "not_ready", f"deployment {dep_id[:8]} is no longer ready (another one is); "
                                  "the benchmark stopped")

    async def shutdown(self) -> None:
        """Cancel running benchmarks so each records a final status before the deployment stops."""
        running = [t for t in self._tasks.values() if not t.done()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)

    def reconcile_on_start(self) -> list[str]:
        """A benchmark still 'running' in the store was interrupted by a gateway restart."""
        notes = []
        for row in self._store.query("SELECT doc FROM benchmarks WHERE json_extract(doc, '$.status') = 'running'"):
            result = BenchmarkResult.model_validate(json.loads(row["doc"]))
            self._save(result.model_copy(update={"status": "failed", "error": "gateway restarted"}))
            notes.append(f"benchmark {result.id[:8]} marked failed after restart")
        return notes
