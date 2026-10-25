"""AirLLM adapter: the `disk` tier. One local worker streams every layer from disk for every token.

Chosen only when nothing faster can serve the request (docs/ENGINE_TIERS.md): an HF-only checkpoint, full precision,
or no llama.cpp tier fits. workers/airllm_worker.py runs in the AirLLM venv under the Supervisor, listens on loopback
with a per-deployment key passed by environment (never argv) and speaks OpenAI-style SSE. Expect seconds to minutes
per token (docs/research/airllm.md section 7); a closed stream stops it after the token in progress, stop() kills it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .. import config, netutil
from ..installers import airllm as installer
from ..schemas import (
    CandidateSummary,
    DeviceAssignment,
    DeviceBudget,
    ExtraEngineInfo,
    ModelArtifact,
    NodeState,
    NodeView,
    ObservedDevice,
    ObservedPlacement,
    Plan,
    PlanRequest,
    PlanStrategy,
    utcnow,
)
from ..sse import SSEParser
from ..units import GiB, MiB, fmt_bytes
from ..workers.airllm_worker import KEY_ENV, SPLIT_DIR_NAME, check_checkpoint, expected_modules, split_done
from .base import (
    CancellationResult,
    Capabilities,
    CompletionResult,
    DeploymentHandle,
    DeploymentSpec,
    EngineAdapterError,
    EngineSnapshot,
    GenerationEvent,
    InferenceRequest,
    PlacementReport,
    StopResult,
    UnsupportedCapability,
    ValidationResult,
    artifact_engines,
)

if TYPE_CHECKING:
    from .registry import AdapterContext

log = logging.getLogger("hypercompute.airllm")
PROCESS_KEY_PREFIX = "deployment:"  # same as llama-server, so restart reconciliation finds the worker too
WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "workers" / "airllm_worker.py"
IDLE_TIMEOUT_S = 900.0  # one token can take minutes
LOAD_TIMEOUT_S = 4 * 3600.0  # the first start splits the whole checkpoint into per-layer files
COMPLETE_TIMEOUT_S = 6 * 3600.0
DISK_HEADROOM = 2 * GiB
CUDA_CONTEXT = 450 * MiB  # the llama planner's CUDA overhead estimate
PROCESS_RAM = 2 * GiB  # torch + transformers + CUDA libraries in the worker process (estimate)
ACTIVATION_SLACK = 256 * MiB
# Effective disk -> device throughput from field reports (research section 7): #364 Windows, #330 page cache.
SLOW_BPS, FAST_BPS = 0.21e9, 5e9
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
SAFE_SUFFIXES = (".json", ".safetensors", ".txt", ".model", ".tiktoken", ".jinja")
LOAD_FIX = "read the worker log; a broken split can be deleted and is rebuilt on the next start"
CAPABILITIES = Capabilities(
    engine_id="airllm", status="implemented", streaming=True, multi_host=False, disk_offload=True, cancel=True,
    platforms=["windows-x64 (CUDA/CPU)", "linux (CUDA/CPU)"],
    notes="disk capacity mode: every layer is read from disk for every token; cancel stops after the current token")


@dataclass(frozen=True)
class Footprint:
    """Per-forward-pass working set. AirLLM keeps one module on the device at a time (research section 4.3)."""

    n_layers: int | None
    module_bytes: int  # largest module streamed per pass
    resident_bytes: int  # a tied embedding stays on the device
    kv_bytes: int  # KV cache for ctx tokens over all layers
    compute_bytes: int  # activations of a ctx-long prefill
    known: bool  # False: guessed without config.json


def footprint(cfg: dict | None, ctx: int, weights: int) -> Footprint:
    """From config.json; without it, a rough guess that the plan marks as fit_unknown."""
    if cfg is None:
        return Footprint(None, max(weights // 10, 256 * MiB), 0, GiB, GiB, known=False)
    try:
        h, n, heads = int(cfg["hidden_size"]), int(cfg["num_hidden_layers"]), int(cfg["num_attention_heads"])
        kv = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or h // heads)
        inter, vocab = int(cfg.get("intermediate_size") or 4 * h), int(cfg["vocab_size"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return Footprint(None, max(weights // 10, 256 * MiB), 0, GiB, GiB, known=False)
    b = DTYPE_BYTES.get(str(cfg.get("torch_dtype") or cfg.get("dtype")), 2)
    attn = 2 * h * heads * head_dim + 2 * h * kv * head_dim + heads * head_dim + 2 * kv * head_dim
    layer, embed = (attn + 3 * h * inter + 2 * h) * b, vocab * h * b
    tied = bool(cfg.get("tie_word_embeddings"))
    return Footprint(n_layers=n, module_bytes=layer if tied else max(layer, embed),
                     resident_bytes=embed if tied else 0, kv_bytes=ctx * n * 2 * kv * head_dim * b,
                     compute_bytes=ctx * (4 * h + 2 * inter) * b + ACTIVATION_SLACK, known=True)


# ----------------------------------------------------------------------------- paths and local facts


def model_dir(paths: config.Paths, art: ModelArtifact) -> Path:
    return paths.models / art.id


def shards_dir(paths: config.Paths, art: ModelArtifact) -> Path:
    """Per model and revision: AirLLM's "already split" check looks only at layer file names (research 3.4)."""
    return paths.home / "airllm-shards" / art.id / art.revision[:12]


def read_config(mdir: Path) -> dict | None:
    try:
        cfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cfg if isinstance(cfg, dict) else None


def weights_bytes(art: ModelArtifact) -> int:
    total = sum(f.size_bytes for f in art.files if (f.path or f.name).endswith(".safetensors"))
    return total or art.size_bytes or 0


def _split_bytes(path: Path) -> int:
    try:
        with os.scandir(path) as it:
            return sum(e.stat().st_size for e in it if e.is_file() and e.name.endswith(".safetensors"))
    except OSError:
        return 0


def _free_disk(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _duration(s: float) -> str:
    return f"{s:.1f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


# ----------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class _Local:
    device: DeviceBudget | None  # where the forward pass runs: the local CUDA pool, else host RAM
    ram: DeviceBudget | None
    torch_device: str


def _local(local: NodeView, paths: config.Paths) -> _Local:
    from ..services.deployments import cpu_budget, local_budgets  # lazy: services import the adapters

    ram = cpu_budget(local)
    if installer.torch_device(paths).startswith("cuda"):
        gpu = next((b for b in local_budgets(local) if b.device.startswith("CUDA")), None)
        if gpu is not None:
            return _Local(gpu, ram, f"cuda:{gpu.device[4:] or 0}")
    return _Local(ram, ram, "cpu")


def _gate(req: PlanRequest, art: ModelArtifact, local: NodeView, info: ExtraEngineInfo, mdir: Path,
          cfg: dict | None) -> list[str]:
    reasons = []
    if info.status != "installed":
        reasons.append(f"AirLLM is not available on this host: {info.detail}")
    if art.format != "hf":
        reasons.append(f"AirLLM serves HF safetensors checkpoints; {art.id} is {art.format}")
    elif "airllm" not in artifact_engines(art):
        reasons.append(f"the model manifest does not list airllm for {art.id}")
    if not req.allow_capacity_mode:
        reasons.append("capacity mode is off for this request (allow_capacity_mode=false)")
    if local.state != NodeState.eligible:
        why = f" ({local.state_reason})" if local.state_reason else ""
        reasons.append(f"this machine is {local.state.value}{why}")
    unsafe = [f.path or f.name for f in art.files if not (f.path or f.name).endswith(SAFE_SUFFIXES)]
    if unsafe:
        reasons.append(f"AirLLM loads safetensors without custom code; drop {', '.join(unsafe[:3])} from the manifest")
    if cfg is not None:
        try:
            check_checkpoint(mdir)
        except ValueError as exc:
            reasons.append(str(exc))
    return reasons


@dataclass(frozen=True)
class _Disk:
    split_ready: bool
    done: int
    total: int | None
    need: int
    free: int | None


def _disk(paths: config.Paths, art: ModelArtifact, cfg: dict | None, weights: int) -> _Disk:
    """The split copies the weights once (about 1x, research 3.4); a missing download needs its own copy too."""
    shards = shards_dir(paths, art)
    total = expected_modules(cfg) if cfg else None
    done = split_done(shards)
    ready = total is not None and done >= total
    need = 0 if ready else max(0, weights - _split_bytes(shards / SPLIT_DIR_NAME)) + DISK_HEADROOM
    if cfg is None:
        need += weights  # not downloaded yet
    return _Disk(ready, done, total, need, _free_disk(shards))


def _speed(weights: int, fp: Footprint, device: str) -> tuple[int, float, float, str]:
    per_token = max(weights - fp.resident_bytes, fp.module_bytes)
    fast, slow = per_token / FAST_BPS, per_token / SLOW_BPS
    where = "" if device.startswith("cuda") else "; on the CPU it is slower still"
    text = (f"capacity mode: AirLLM reads about {fmt_bytes(per_token)} from disk for every token, so expect roughly "
            f"{_duration(fast)} to {_duration(slow)} per token and {_duration(100 * fast)} to "
            f"{_duration(100 * slow)} for a 100-token answer{where}")
    return per_token, fast, slow, text


def _assignment(local: NodeView, where: _Local, fp: Footprint, ram_need: int) -> DeviceAssignment | None:
    """One assignment for the pool the forward pass runs in, sized so the deployment's lease reserves it."""
    b = where.device
    if b is None:
        return None
    on_cpu = where.torch_device == "cpu"
    weights = fp.resident_bytes + fp.module_bytes
    overhead = PROCESS_RAM if on_cpu else CUDA_CONTEXT
    total = ram_need if on_cpu else weights + fp.kv_bytes + fp.compute_bytes + overhead
    if not on_cpu and where.ram is not None and where.ram.pool_id == b.pool_id:
        total += ram_need  # unified memory: the host-side copies come out of the same pool
    return DeviceAssignment(node_id=local.id, device=b.device, endpoint=None, pool_id=b.pool_id,
                            layers=list(range(fp.n_layers or 0)), holds_output=True, weight_bytes=weights,
                            kv_bytes=fp.kv_bytes, compute_bytes_est=fp.compute_bytes, overhead_bytes_est=overhead,
                            total_bytes_est=total, budget_bytes=b.budget_bytes, fits=total <= b.budget_bytes)


def _ram_need(fp: Footprint, art: ModelArtifact, disk: _Disk, on_cpu: bool) -> int:
    """Host RAM: each module is loaded, pinned and the next one prefetched; the first split holds 1-2 source shards."""
    run = PROCESS_RAM + 3 * fp.module_bytes
    if on_cpu:
        run += fp.resident_bytes + fp.kv_bytes + fp.compute_bytes
    shard = max((f.size_bytes for f in art.files if (f.path or f.name).endswith(".safetensors")), default=0)
    return run if disk.split_ready else max(run, PROCESS_RAM + 2 * shard)


def _memory_reasons(where: _Local, a: DeviceAssignment | None, ram_need: int) -> list[str]:
    if a is None:
        return []
    reasons = []
    if not a.fits:
        reasons.append(f"{a.device} needs about {fmt_bytes(a.total_bytes_est)} for one layer plus KV cache and "
                       f"activations; budget {fmt_bytes(a.budget_bytes)}")
    if where.ram is not None and where.ram.pool_id != a.pool_id and ram_need > where.ram.budget_bytes:
        reasons.append(f"host RAM needs about {fmt_bytes(ram_need)} for layer loading; budget "
                       f"{fmt_bytes(where.ram.budget_bytes)}")
    return reasons


def plan(req: PlanRequest, art: ModelArtifact, local: NodeView, remote: list[NodeView],
         ctx: AdapterContext) -> Plan:
    """Local node only, HF only, tier `disk`, capacity mode. Reads only local files; never starts a process."""
    paths, info = ctx.paths, installer.status(ctx.paths)
    mdir = model_dir(paths, art)
    cfg = read_config(mdir)
    weights = weights_bytes(art)
    reasons = _gate(req, art, local, info, mdir, cfg)
    fp = footprint(cfg, req.ctx_size, weights)
    where = _local(local, paths)
    disk = _disk(paths, art, cfg, weights)
    ram_need = _ram_need(fp, art, disk, where.torch_device == "cpu")
    a = _assignment(local, where, fp, ram_need)
    misfits = _memory_reasons(where, a, ram_need) if fp.known else []  # a guessed footprint never blocks a plan
    if a is None:
        misfits.append("free memory on this machine is unknown (no RAM measurement yet)")
    if disk.free is not None and disk.need > disk.free:
        misfits.append(f"not enough disk for the per-layer split: need about {fmt_bytes(disk.need)} under "
                       f"{shards_dir(paths, art)}, {fmt_bytes(disk.free)} free")
    reasons += misfits
    per_token, fast, slow, speed = _speed(weights, fp, where.torch_device)
    warnings = [speed, *_notes(req, remote, fp, disk, paths, art)]
    fit = "does_not_fit" if misfits else "estimated_fit" if fp.known else "fit_unknown"
    devices = [a.device] if a else []
    data = installer.read_installed(paths) or {}
    return Plan(
        id=uuid.uuid4().hex, model_id=art.id, strategy=req.strategy, ctx_size=req.ctx_size, feasible=not reasons,
        fit_label=fit, heuristic=True, assignments=[a] if a else [], device_list=[], tensor_split=[],  # type: ignore[arg-type]
        rpc_endpoints=[], cpu_resident_bytes=0, reasons=reasons, warnings=warnings,
        candidates=[CandidateSummary(label="AirLLM disk streaming on this machine", devices=devices,
                                     feasible=not reasons, reason="; ".join(reasons) or
                                     f"one layer at a time on {where.torch_device}")],
        created_at=utcnow(), engine="airllm", tier="disk", capacity_mode=True,
        engine_options={"shards_dir": str(shards_dir(paths, art)), "device": where.torch_device, "dtype": "auto",
                        "version": installer.AIRLLM_COMMIT, "pin": f"torch {data.get('torch', 'unknown')}",
                        "split_ready": disk.split_ready, "split_modules": [disk.done, disk.total],
                        "bytes_per_token": per_token, "s_per_token": [round(fast, 1), round(slow, 1)],
                        "host_ram_bytes": ram_need})


def _notes(req: PlanRequest, remote: list[NodeView], fp: Footprint, disk: _Disk, paths: config.Paths,
           art: ModelArtifact) -> list[str]:
    notes = []
    if remote and req.strategy != PlanStrategy.local_only:
        notes.append(f"AirLLM runs on this machine only; {len(remote)} peer(s) are not used")
    if not disk.split_ready:
        notes.append(f"the first start splits the checkpoint into per-layer files under {shards_dir(paths, art)}; "
                     "this can take many minutes and is reused afterwards")
    if not fp.known:
        notes.append("per-layer memory is a rough guess until the model (its config.json) is downloaded")
    return notes


# ----------------------------------------------------------------------------- adapter


def idle_timeout(engine_options: dict) -> float:
    """Longest silence to accept between tokens: the default, or 3x the plan's slow per-token estimate if longer."""
    estimate = engine_options.get("s_per_token")
    slow = estimate[1] if isinstance(estimate, list) and len(estimate) == 2 else 0
    return max(IDLE_TIMEOUT_S, 3.0 * float(slow)) if isinstance(slow, int | float) else IDLE_TIMEOUT_S


def worker_env(api_key: str) -> dict[str, str]:
    return {KEY_ENV: api_key, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false"}


def describe_phase(body: dict) -> str:
    phase = body.get("phase")
    split = body.get("split") if isinstance(body.get("split"), dict) else {}
    if phase == "splitting":
        total = split.get("total") or "?"
        return (f"splitting the checkpoint into per-layer files {split.get('done', 0)}/{total} "
                "(first start only; can take many minutes)")
    return {"starting": "starting the AirLLM worker", "importing": "importing torch and AirLLM",
            "loading": "building the model (the layers stay on disk)"}.get(str(phase), f"loading ({phase})")


def last_error(log_text: str) -> str:
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    errors = [ln for ln in lines if "Error" in ln or "error" in ln]
    return ((errors or lines or ["no output in the worker log"])[-1])[:400]


class AirLlmAdapter:
    engine_id = "airllm"
    idle_timeout_s = IDLE_TIMEOUT_S
    failure_reason = staticmethod(last_error)  # the deployment watcher's exit reason for a dead worker

    def __init__(self, ctx: AdapterContext) -> None:
        self._sup, self._logs, self._paths = ctx.supervisor, ctx.logs_dir, ctx.paths
        self._active: dict[str, httpx.Response] = {}
        self._idle: dict[str, float] = {}  # deployment -> longest wait for the next token
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(IDLE_TIMEOUT_S, connect=5.0))

    def idle_for(self, deployment_id: str) -> float:
        return self._idle.get(deployment_id, IDLE_TIMEOUT_S)

    def idle_timeout_for(self, plan: Plan) -> float:
        """The deployment's silence budget (stream reads, the gateway's warmup): grows with the plan's s/token."""
        return idle_timeout(plan.engine_options)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def inspect(self) -> EngineSnapshot:
        raise UnsupportedCapability("airllm", "inspect (EngineInfo describes llama.cpp; use installers.airllm.status)")

    async def validate(self, spec: DeploymentSpec) -> ValidationResult:
        problems = []
        if spec.plan.engine != "airllm" or spec.artifact is None:
            problems.append("this deployment is not an AirLLM plan with a model artifact")
        try:
            check_checkpoint(spec.model_dir or spec.model_path)
        except ValueError as exc:
            problems.append(str(exc))
        if not installer.venv_python(self._paths).is_file():
            problems.append(f"the AirLLM venv is missing; run: {installer.INSTALL_HINT}")
        if not WORKER_SCRIPT.is_file():
            problems.append(f"worker script missing from this build: {WORKER_SCRIPT}")
        return ValidationResult(ok=not problems, problems=problems)

    def build_args(self, spec: DeploymentSpec, port: int) -> list[str]:
        if spec.artifact is None:
            raise EngineAdapterError("validation_failed", "an AirLLM deployment needs its model artifact")
        opts = spec.plan.engine_options
        return ["-I", "-u", str(WORKER_SCRIPT), "--model-dir", str(spec.model_dir or spec.model_path),
                "--shards-dir", str(shards_dir(self._paths, spec.artifact)), "--host", "127.0.0.1",
                "--port", str(port), "--alias", spec.alias, "--max-seq-len", str(spec.plan.ctx_size),
                "--device", str(opts.get("device", "auto")), "--dtype", str(opts.get("dtype", "auto"))]

    async def prepare(self, spec: DeploymentSpec) -> DeploymentHandle:
        port = netutil.free_loopback_port()
        args = self.build_args(spec, port)
        api_key = secrets.token_urlsafe(24)
        log_path = self._logs / f"airllm-worker-{spec.deployment_id}.log"
        key = f"{PROCESS_KEY_PREFIX}{spec.deployment_id}"
        shards_dir(self._paths, spec.artifact).mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
        cwd = installer.engine_dir(self._paths)
        proc = await asyncio.to_thread(self._sup.start, key, installer.venv_python(self._paths), args, cwd,
                                       log_path, worker_env(api_key))
        self._idle[spec.deployment_id] = self.idle_timeout_for(spec.plan)
        return DeploymentHandle(deployment_id=spec.deployment_id, process_key=key, base_url=f"http://127.0.0.1:{port}",
                                log_path=log_path, pid=proc.pid, args=args, api_key=api_key)

    def read_log(self, handle: DeploymentHandle) -> str:
        proc = self._sup.get(handle.process_key)
        if proc:
            return proc.read_log(512 * 1024)
        try:
            return handle.log_path.read_text(encoding="utf-8", errors="replace")[-512 * 1024:]
        except OSError:
            return ""

    def process_alive(self, handle: DeploymentHandle) -> bool:
        proc = self._sup.get(handle.process_key)
        return bool(proc and proc.alive())

    @staticmethod
    def _auth(handle: DeploymentHandle) -> dict[str, str]:
        return {"Authorization": f"Bearer {handle.api_key}"}

    async def health(self, handle: DeploymentHandle) -> tuple[int, dict]:
        """(status, body) from the worker's /health; (0, {}) while it is not listening yet."""
        try:
            r = await self._client.get(f"{handle.base_url}/health", headers=self._auth(handle), timeout=5.0)
        except httpx.HTTPError:
            return 0, {}
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = {}
        return r.status_code, body if isinstance(body, dict) else {}

    async def wait_ready(self, handle: DeploymentHandle, timeout_s: float = LOAD_TIMEOUT_S,
                         on_progress: Callable[[str], None] | None = None) -> None:
        loop = asyncio.get_running_loop()
        deadline, last = loop.time() + timeout_s, ""
        while loop.time() < deadline:
            if not self.process_alive(handle):
                raise EngineAdapterError("engine_exited", "the AirLLM worker exited during load: "
                                         f"{last_error(self.read_log(handle))}", fix=LOAD_FIX)
            code, body = await self.health(handle)
            if code == 200:
                return
            if code == 500:
                raise EngineAdapterError("load_failed", f"AirLLM could not load the model: {body.get('error')}",
                                         fix=LOAD_FIX)
            if code == 401:
                raise EngineAdapterError("load_failed", "the AirLLM worker rejected this deployment's key")
            phase = describe_phase(body) if code == 503 else "starting the AirLLM worker"
            if phase != last and on_progress:
                on_progress(phase)
            last = phase
            await asyncio.sleep(1.0)
        raise EngineAdapterError("load_timeout", f"the AirLLM worker was not ready after {timeout_s / 60:.0f} min "
                                 f"(last phase: {last})", fix=LOAD_FIX)

    async def observe(self, handle: DeploymentHandle, spec: DeploymentSpec,
                      validation: ValidationResult) -> PlacementReport:
        """No peers to verify: the worker reports the device and what stays resident; everything else is on disk."""
        shards = shards_dir(self._paths, spec.artifact) if spec.artifact else None
        code, body = await self.health(handle)
        detail = (f"AirLLM streams every layer from {shards} to {spec.plan.engine_options.get('device', 'the device')}"
                  " for every token (disk tier, this machine only)")
        if code != 200:
            return PlacementReport(observed=None, verified=True, detail=detail)
        resident = body.get("resident_bytes")
        device = ObservedDevice(name=f"{body.get('device')} ({body.get('device_name')})",
                                model_bytes=resident if isinstance(resident, int) else 0, kv_bytes=0,
                                compute_bytes=0, layers_est=None)
        evidence = [f"worker ready on {body.get('device')} with torch {body.get('torch')}",
                    f"{body.get('split_modules')} per-layer shards in {shards}",
                    "resident on the device after load: " + (fmt_bytes(resident) if isinstance(resident, int)
                                                             else "not measured")]
        return PlacementReport(observed=ObservedPlacement(offloaded=None, devices=[device], evidence_lines=evidence,
                                                          parsed_at=utcnow()), verified=True, detail=detail)

    async def stop(self, handle: DeploymentHandle) -> StopResult:
        outcome = await asyncio.to_thread(self._sup.stop, handle.process_key, config.STOP_GRACE_S)
        self._idle.pop(handle.deployment_id, None)
        if outcome is None:
            return StopResult(clean=True, detail="process was not running")
        return StopResult(clean=outcome.clean, detail=outcome.describe())

    async def complete(self, handle: DeploymentHandle, req: InferenceRequest) -> CompletionResult:
        r = await self._client.post(f"{handle.base_url}/v1/chat/completions", json={**req.body, "stream": False},
                                    headers=self._auth(handle), timeout=httpx.Timeout(COMPLETE_TIMEOUT_S, connect=5.0))
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = {"error": {"message": r.text[:500]}}
        return CompletionResult(status_code=r.status_code, body=body)

    async def stream(self, handle: DeploymentHandle, req: InferenceRequest) -> AsyncIterator[GenerationEvent]:
        parser = SSEParser()
        timeout = httpx.Timeout(self.idle_for(handle.deployment_id), connect=5.0)
        async with self._client.stream("POST", f"{handle.base_url}/v1/chat/completions", timeout=timeout,
                                       json={**req.body, "stream": True}, headers=self._auth(handle)) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                code = "engine_busy" if resp.status_code == 429 else "upstream_error"
                raise EngineAdapterError(code, f"AirLLM worker HTTP {resp.status_code}: {text[:400]}")
            self._active[req.request_id] = resp
            try:
                async for chunk in resp.aiter_raw():
                    for ev in parser.feed(chunk):
                        yield _event(ev.raw, ev.data)
                for ev in parser.flush():
                    yield _event(ev.raw, ev.data)
            finally:
                self._active.pop(req.request_id, None)

    async def cancel(self, request_id: str) -> CancellationResult:
        resp = self._active.pop(request_id, None)
        if resp is None:
            return CancellationResult(request_id, False, "no active upstream request")
        await resp.aclose()
        return CancellationResult(request_id, True, "stream closed; the AirLLM worker stops after the token in "
                                  "progress (stop the deployment to end it at once)")


def _event(raw: bytes, data: str) -> GenerationEvent:
    if data.strip() == "[DONE]":
        return GenerationEvent(raw=raw, data=data, payload=None, done=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("error"):  # the worker failed after the headers, e.g. CUDA OOM
        err = payload["error"]
        message = err.get("message") if isinstance(err, dict) else err
        raise EngineAdapterError("upstream_error", f"AirLLM worker failed mid-stream: {message or 'generation failed'}")
    return GenerationEvent(raw=raw, data=data, payload=payload if isinstance(payload, dict) else None)


# ----------------------------------------------------------------------------- module contract


def build(ctx: AdapterContext) -> AirLlmAdapter | None:
    info = installer.status(ctx.paths)
    if info.status != "installed":
        log.info("AirLLM adapter not built: %s", info.detail)
        return None
    return AirLlmAdapter(ctx)


def status(ctx: AdapterContext) -> ExtraEngineInfo:
    return installer.status(ctx.paths)
