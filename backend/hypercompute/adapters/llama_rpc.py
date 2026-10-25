"""llama.cpp adapter: coordinator-side llama-server with remote ggml-rpc-server workers.

The coordinator owns the GGUF and the inference request; workers receive tensors over RPC.
llama-server listens on loopback only with a per-deployment API key (passed via environment, not
argv), so other local web pages cannot drive it directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import subprocess
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from .. import config, engine, netutil
from ..agent.supervisor import Supervisor
from ..capacity import runtime as stream_runtime
from ..gguf import expert_override_pattern
from ..schemas import EngineDevice, Plan
from ..sse import SSEParser
from .base import (
    ENGINE_ROADMAP,
    CancellationResult,
    CompletionResult,
    DeploymentHandle,
    DeploymentSpec,
    EngineAdapterError,
    EngineSnapshot,
    GenerationEvent,
    InferenceRequest,
    PlacementReport,
    StopResult,
    ValidationResult,
)
from .llama_output import (
    check_moe_placement,
    check_participation,
    detect_failure,
    parse_list_devices,
    parse_overrides,
    parse_placement,
)

log = logging.getLogger("hypercompute.llama_rpc")
PROCESS_KEY_PREFIX = "deployment:"  # supervisor key of a deployment's llama-server
LOAD_PHASES = (
    ("load_model: loading model", "reading model file"),
    ("llama_prepare_model_devices: using device RPC", "connected to peer devices"),
    ("load_tensors: offloaded", "assigning layers to devices"),
    ("model buffer size", "loading weights onto devices"),
    ("llama_kv_cache:", "allocating KV cache"),
    ("sched_reserve:", "reserving compute buffers"),
    ("llama_server: model loaded", "model loaded"),
)
# A first load sends every peer its weights over llama RPC before /health answers. The load budget assumes at least
# 1 MiB/s for that upload (the vLLM adapter's download floor; below the 1.59e6 B/s measured on the Iroh relay), on
# top of LOAD_TIMEOUT_S, and stops at the 4 h cap the other engines use.
PEER_UPLOAD_FLOOR_BPS = float(1 << 20)
MAX_LOAD_S = 4 * 3600.0


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    devices: list[EngineDevice]
    by_endpoint: dict[str, list[EngineDevice]]
    detail: str
    timed_out: bool = False


class LlamaRpcAdapter:
    engine_id = "llama_rpc"
    idle_timeout_s = config.UPSTREAM_IDLE_TIMEOUT_S
    engine_id = "llama_rpc"

    def __init__(self, installed: engine.InstalledEngine, lock: engine.EngineLock, supervisor: Supervisor,
                 logs_dir: Path) -> None:
        self._engine, self._lock, self._sup, self._logs = installed, lock, supervisor, logs_dir
        self._active: dict[str, httpx.Response] = {}
        self._stream_stats: dict[str, Path] = {}  # deployment id -> hcstream stats file
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(config.UPSTREAM_IDLE_TIMEOUT_S, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def load_timeout_s(self, plan: Plan) -> float:
        """wait_ready's budget: LOAD_TIMEOUT_S plus the planned peer weight bytes at PEER_UPLOAD_FLOOR_BPS, capped at
        MAX_LOAD_S. A plan without peers keeps LOAD_TIMEOUT_S."""
        sent = sum(a.weight_bytes for a in plan.assignments if a.endpoint)
        return min(MAX_LOAD_S, config.LOAD_TIMEOUT_S + sent / PEER_UPLOAD_FLOOR_BPS) if sent else config.LOAD_TIMEOUT_S

    # ------------------------------------------------------------------ inspection / probes
    async def inspect(self) -> EngineSnapshot:
        info = await asyncio.to_thread(engine.inspect, self._engine, self._lock)
        return EngineSnapshot(capabilities=ENGINE_ROADMAP[0], info=info)

    async def probe_workers(self, endpoints: list[str]) -> ProbeResult:
        """Real engine handshake: `llama-server --rpc <eps> --list-devices` (HELLO + device memory)."""
        args = ["--rpc", ",".join(endpoints), "--list-devices"]
        try:
            out = await asyncio.to_thread(engine.run_engine, self._engine, "llama-server", args,
                                          config.PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return ProbeResult(False, [], {}, f"probe timed out after {config.PROBE_TIMEOUT_S:.0f}s", timed_out=True)
        except (OSError, engine.EngineError) as exc:
            return ProbeResult(False, [], {}, f"probe could not run: {exc}")
        text = out.stdout + out.stderr
        devices = parse_list_devices(text)
        rpc = [d for d in devices if d.name.startswith("RPC")]
        by_ep: dict[str, list[EngineDevice]] = {ep: [d for d in rpc if ep in d.description] for ep in endpoints}
        missing = [ep for ep, devs in by_ep.items() if not devs]
        if out.returncode != 0 or missing:
            failure = detect_failure(text) or f"exit code {out.returncode}"
            return ProbeResult(False, devices, by_ep, f"{failure}; unreachable: {', '.join(missing) or 'none'}")
        return ProbeResult(True, devices, by_ep, "worker answered the llama.cpp RPC handshake")

    async def validate(self, spec: DeploymentSpec) -> ValidationResult:
        problems: list[str] = []
        if not spec.model_path.is_file():
            problems.append(f"model file missing: {spec.model_path}")
        endpoints = spec.plan.rpc_endpoints
        device_map: dict[str, list[str]] = {}
        if endpoints:
            probe = await self.probe_workers(endpoints)
            if not probe.ok:
                problems.append(f"worker probe failed: {probe.detail}")
            device_map = {ep: [d.name for d in devs] for ep, devs in probe.by_endpoint.items()}
            known = {d.name for d in probe.devices}
            for a in spec.plan.assignments:
                if a.endpoint and a.device not in known:
                    problems.append(f"planned device {a.device} not present in current device list {sorted(known)}")
        return ValidationResult(ok=not problems, problems=problems, rpc_device_map=device_map)

    # ------------------------------------------------------------------ lifecycle
    def build_args(self, spec: DeploymentSpec, port: int) -> list[str]:
        plan = spec.plan
        args = ["-m", str(spec.model_path), "--alias", spec.alias]
        if plan.rpc_endpoints:
            # First: llama.cpp parses arguments left to right and -ot resolves its buffer type names against the
            # devices registered so far, so an -ot naming an RPC buffer (RPC0[host:port]) needs --rpc before it.
            args += ["--rpc", ",".join(plan.rpc_endpoints)]
        if (engine.host_os() == "windows" and plan.feasible and plan.tier == "gpu_ram"
                and plan.cpu_layers > 0 and not plan.capacity_mode):
            # This tier budgets CPU weights in RAM. Avoid Windows whole-file mmap prefetch and
            # pinned/repacked CPU weights; the disk tier still needs file-backed weight storage.
            args += ["--load-mode", "none", "--no-host", "--no-repack"]
        if plan.tier == "stream":
            # hcstream needs file-backed CPU weights (it rebinds them to resident copies or slots), no repacked copy,
            # no weight copies to an accelerator during prompt batches, and no host prompt cache outside its bound.
            args += ["--load-mode", "mmap", "--no-repack", "--no-op-offload", "--cache-ram", "0"]
        elif plan.tier == "moe_offload":
            args += moe_offload_args(plan)
        elif plan.tier in ("gpu_ram", "gpu_disk"):
            # Admission budgets host RAM for weights, KV and buffers. llama-server's host prompt cache (up to 8 GiB by
            # default) is outside that budget, so it stays off where host RAM is the constraint.
            args += ["--cache-ram", "0"]
        if plan.device_list:
            args += ["--device", ",".join(plan.device_list), "--split-mode", "layer",
                     "--tensor-split", ",".join(str(n) for n in plan.tensor_split),
                     "-ngl", str(plan.n_gpu_units) if plan.cpu_layers else "all"]
        else:  # every unit on the host CPU
            args += ["--device", "none", "-ngl", "0"]
        if spec.mmproj_path is not None and plan.tier != "stream":
            # hcstream bounds the host RAM of CPU weights; a projector it does not account for stays off that tier
            args += projector_args(plan, spec.mmproj_path)
        args += ["--fit", "off", "-c", str(plan.ctx_size), *slot_args(parallel_slots(plan)),
                 *engine_flag_args(plan.engine_flags),
                 "--host", "127.0.0.1", "--port", str(port), "--metrics", "--no-ui", "-lv", "4",
                 "--log-colors", "off"]
        return args

    def max_concurrency(self, plan: Plan) -> int:
        """The -np slots the plan launches with: the gateway's cap on requests llama-server batches at once. Never
        raises (the gateway asks for every new deployment); build_args refuses a plan whose slot count is bad."""
        try:
            return parallel_slots(plan)
        except EngineAdapterError:
            return 1

    async def prepare(self, spec: DeploymentSpec) -> DeploymentHandle:
        port = netutil.free_loopback_port()
        args = self.build_args(spec, port)
        api_key = secrets.token_urlsafe(24)
        log_path = self._logs / f"llama-server-{spec.deployment_id}.log"
        key = f"{PROCESS_KEY_PREFIX}{spec.deployment_id}"
        exe, cwd, env = self._engine.binary("llama-server"), self._engine.bin_dir, {"LLAMA_API_KEY": api_key}
        if spec.plan.tier == "stream":
            if spec.plan.stream is None:
                raise EngineAdapterError("bad_plan", "a stream-tier plan carries no stream layout")
            try:
                launch = await asyncio.to_thread(stream_runtime.prepare_stream, self._engine.bin_dir, spec.model_path,
                                                 spec.plan.stream, self._logs, spec.deployment_id)
            except stream_runtime.StreamRuntimeError as exc:
                raise EngineAdapterError(exc.code, str(exc), fix=exc.fix) from exc
            exe, cwd = launch.bin_dir / engine.exe_name("llama-server"), launch.bin_dir
            env = {**env, **launch.env}
            self._stream_stats[spec.deployment_id] = launch.stats_path
        proc = await asyncio.to_thread(self._sup.start, key, exe, args, cwd, log_path, env)
        return DeploymentHandle(deployment_id=spec.deployment_id, process_key=key,
                                base_url=f"http://127.0.0.1:{port}", log_path=log_path, pid=proc.pid, args=args,
                                api_key=api_key)

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

    def load_phase(self, handle: DeploymentHandle) -> str:
        text = self.read_log(handle)
        phase = "starting llama-server"
        for needle, label in LOAD_PHASES:
            if needle in text:
                phase = label
        if phase == "loading weights onto devices" and "--rpc" in handle.args:
            phase += " (peers receive their layers over the network; progress is not observable)"
        return phase

    async def wait_ready(self, handle: DeploymentHandle, timeout_s: float = config.LOAD_TIMEOUT_S,
                         on_progress: Callable[[str], None] | None = None) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_phase = ""
        while loop.time() < deadline:
            if not self.process_alive(handle):
                text = self.read_log(handle)
                raise EngineAdapterError("engine_exited",
                                         detect_failure(text) or "llama-server exited during load")
            phase = self.load_phase(handle)
            if phase != last_phase and on_progress:
                on_progress(phase)
                last_phase = phase
            try:
                r = await self._client.get(f"{handle.base_url}/health", timeout=3.0,
                                           headers=self._auth(handle))
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        raise EngineAdapterError("load_timeout", f"llama-server was not ready after {timeout_s:.0f}s "
                                 "(a timeout does not prove workers released memory)")

    async def observe(self, handle: DeploymentHandle, spec: DeploymentSpec,
                      validation: ValidationResult) -> PlacementReport:
        """llama.cpp's own load log says which buffers each device got; every planned peer must hold weights."""
        text = self.read_log(handle)
        placement = parse_placement(text)
        ok, detail = check_participation(spec.plan, placement, validation.rpc_device_map)
        if spec.plan.tier == "stream" and spec.plan.stream is not None:
            stats = stream_runtime.read_stats(self.stream_stats_path(spec.deployment_id))
            stream_ok, stream_detail = stream_runtime.verify_stats(stats, spec.plan.stream)
            ok = stream_ok and (ok or not spec.plan.rpc_endpoints)
            detail = f"{stream_detail}; {detail}"
        elif spec.plan.tier == "moe_offload":  # experts in CPU_Mapped, dense parts on the GPU, from llama.cpp's log
            moe_ok, moe_detail = check_moe_placement(spec.plan, placement, parse_overrides(text))
            ok = moe_ok and (ok or not spec.plan.rpc_endpoints)
            detail = f"{moe_detail}; {detail}"
        return PlacementReport(observed=placement, verified=ok, detail=detail)

    def stream_stats_path(self, deployment_id: str) -> Path:
        """The hcstream stats file of a deployment (stable across gateway restarts: derived from the id)."""
        return self._stream_stats.get(deployment_id) or stream_runtime.config_paths(self._logs, deployment_id)[1]

    def capacity_stats(self, deployment_id: str) -> dict:
        """Live hcstream counters of a stream-tier deployment, straight from the extension's stats file."""
        return stream_runtime.stats_summary(stream_runtime.read_stats(self.stream_stats_path(deployment_id)))

    async def stop(self, handle: DeploymentHandle) -> StopResult:
        outcome = await asyncio.to_thread(self._sup.stop, handle.process_key, config.STOP_GRACE_S)
        if outcome is None:
            return StopResult(clean=True, detail="process was not running")
        return StopResult(clean=outcome.clean, detail=outcome.describe())

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _auth(handle: DeploymentHandle) -> dict[str, str]:
        return {"Authorization": f"Bearer {handle.api_key}"} if handle.api_key else {}

    async def complete(self, handle: DeploymentHandle, req: InferenceRequest) -> CompletionResult:
        r = await self._client.post(f"{handle.base_url}/v1/chat/completions", json={**req.body, "stream": False},
                                    headers=self._auth(handle))
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = {"error": {"message": r.text[:500]}}
        return CompletionResult(status_code=r.status_code, body=body)

    async def stream(self, handle: DeploymentHandle, req: InferenceRequest) -> AsyncIterator[GenerationEvent]:
        body = {**req.body, "stream": True}
        body.setdefault("stream_options", {"include_usage": True})
        parser = SSEParser()
        async with self._client.stream("POST", f"{handle.base_url}/v1/chat/completions", json=body,
                                       headers=self._auth(handle)) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                raise EngineAdapterError("upstream_error", f"llama-server HTTP {resp.status_code}: {text[:400]}")
            self._active[req.request_id] = resp
            try:
                async for chunk in resp.aiter_raw():
                    for ev in parser.feed(chunk):
                        yield _to_generation_event(ev.raw, ev.data)
                for ev in parser.flush():
                    yield _to_generation_event(ev.raw, ev.data)
            finally:
                self._active.pop(req.request_id, None)

    async def cancel(self, request_id: str) -> CancellationResult:
        resp = self._active.pop(request_id, None)
        if resp is None:
            return CancellationResult(request_id, False, "no active upstream request")
        await resp.aclose()  # llama-server stops generating when the connection closes
        return CancellationResult(request_id, True, "upstream connection closed")


# Plan.engine_flags -> llama-server flags (common/arg.cpp at b11770: -fa on|off|auto, -b, -ub, -t, -tb). An empty
# dict keeps llama.cpp's defaults: -fa auto, -b 2048, -ub 512, threads = physical cores (llama_cpu_tier.md §4.1).
ENGINE_FLAGS: dict[str, str] = {"flash_attn": "-fa", "batch": "-b", "ubatch": "-ub", "threads": "-t",
                                "threads_batch": "-tb"}
FLAG_RANGES: dict[str, tuple[int, int]] = {"batch": (1, 65536), "ubatch": (1, 65536), "threads": (1, 256),
                                           "threads_batch": (1, 256)}  # sanity bounds, not llama.cpp limits
DEFAULT_BATCH = 2048  # llama.cpp's n_batch default; it clamps -ub to the batch size without a word
MAX_PARALLEL = 16  # a sanity bound on server slots (-np), not llama.cpp's LLAMA_MAX_SEQ (256)


def parallel_slots(plan: Plan) -> int:
    """Server slots of a plan, from engine_options["parallel"] (DeploymentManager.plan sets it; absent means 1).

    Refuses bools, non-integers and values outside 1..MAX_PARALLEL, and more than one slot on the stream tier, where
    hcstream's bound has not been checked with several sequences. moe_offload plans, capacity_mode included, may batch.
    """
    value = plan.engine_options.get("parallel", 1)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_PARALLEL:
        raise EngineAdapterError("bad_plan", f"parallel must be a whole number from 1 to {MAX_PARALLEL}, not {value!r}",
                                 fix=f"plan again with parallel between 1 and {MAX_PARALLEL}")
    if value > 1 and plan.tier == "stream":
        raise EngineAdapterError("bad_plan", f"{value} server slots on the stream tier: hcstream is not verified with "
                                 "more than one sequence", fix="plan with parallel 1, or without the stream tier")
    return value


def slot_args(n: int) -> list[str]:
    """-np n; several slots share one unified KV pool of -c cells (-kvu), so -c and the KV bytes stay what the plan
    budgets (llama-context.cpp: n_ctx_seq = n_ctx with kv_unified) and one decode step stays one ubatch."""
    return ["-np", str(n), "-kvu"] if n > 1 else ["-np", "1"]


def moe_offload_args(plan: Plan) -> list[str]:
    """llama-server flags of a moe_offload plan (-ngl all and --fit off come with every GPU plan).

    The CPU experts stay file-backed: --load-mode mmap, no repacked copy (--no-repack) and no pinned host copy
    (--no-host); either copy is private commit, which grows the pagefile on Windows. --no-op-offload keeps prompt
    batches from copying CPU experts into the GPU compute buffer, which the GPU budget does not hold, and --cache-ram 0
    keeps the host prompt cache out of the RAM the file cache needs. The -ot value is rebuilt from the plan's layers,
    which must all be host-side GPU layers, and must equal the plan's own.
    """
    moe = plan.moe
    if moe is None or not moe.cpu_expert_layers or plan.cpu_layers:
        raise EngineAdapterError("bad_plan", "a moe_offload plan needs CPU expert layers and every layer offloaded")
    host = {i for a in plan.assignments if a.endpoint is None and a.device != "CPU" for i in a.layers}
    outside = sorted(set(moe.cpu_expert_layers) - host)
    override = expert_override_pattern(moe.cpu_expert_layers) + "=CPU"
    if outside or moe.override != override:
        raise EngineAdapterError("bad_plan", f"the moe_offload -ot value {moe.override!r} does not name exactly its "
                                 f"host-side CPU expert layers (not on the local GPU: {outside[:8]})")
    return ["--load-mode", "mmap", "--no-repack", "--no-host", "--no-op-offload", "--cache-ram", "0", "-ot", override]


def projector_args(plan: Plan, path: Path) -> list[str]:
    """--mmproj and the device the projector runs on (common/arg.cpp at b11770). Without -mmdev llama.cpp puts the
    projector on the first --device entry, and peers come first there, so it would run on a peer over RPC. It goes to
    this machine's last GPU in the plan (the one with the output head), or the host CPU when the plan has none."""
    local = [a.device for a in plan.assignments if a.endpoint is None and a.device != "CPU"]
    return ["--mmproj", str(path), "-mmdev", local[-1] if local else "none"]


def engine_flag_args(flags: Mapping[str, object]) -> list[str]:
    """llama-server arguments for a plan's engine_flags; unknown keys and bad values raise EngineAdapterError."""
    unknown = sorted(set(flags) - set(ENGINE_FLAGS))
    if unknown:
        raise EngineAdapterError("bad_engine_flags", f"unknown llama.cpp engine flag(s): {', '.join(unknown)}",
                                 fix=f"use only {', '.join(ENGINE_FLAGS)}")
    args = [part for key, flag in ENGINE_FLAGS.items() if key in flags for part in (flag, _flag_value(key, flags[key]))]
    batch, ubatch = flags.get("batch", DEFAULT_BATCH), flags.get("ubatch")
    if isinstance(ubatch, int) and isinstance(batch, int) and ubatch > batch:
        raise EngineAdapterError("bad_engine_flags", f"ubatch {ubatch} is larger than the batch size {batch}; "
                                 "llama.cpp would silently use the batch size instead",
                                 fix=f"set ubatch <= batch (llama.cpp's default batch is {DEFAULT_BATCH})")
    return args


def _flag_value(key: str, value: object) -> str:
    if key == "flash_attn":
        if not isinstance(value, str) or value not in ("on", "off", "auto"):
            raise EngineAdapterError("bad_engine_flags", f"flash_attn must be on, off or auto, not {value!r}",
                                     fix="use flash_attn: on, off or auto")
        return value
    low, high = FLAG_RANGES[key]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        hint = " (llama.cpp reads 0 or less as every logical CPU)" if key.startswith("threads") else ""
        raise EngineAdapterError("bad_engine_flags", f"{key} must be a whole number from {low} to {high}, not "
                                 f"{value!r}", fix=f"pass {key} as an integer in that range{hint}")
    return str(value)


def _to_generation_event(raw: bytes, data: str) -> GenerationEvent:
    if data.strip() == "[DONE]":
        return GenerationEvent(raw=raw, data=data, payload=None, done=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        payload = None
    return GenerationEvent(raw=raw, data=data, payload=payload if isinstance(payload, dict) else None)
