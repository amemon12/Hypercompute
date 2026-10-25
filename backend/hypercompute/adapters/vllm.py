"""vLLM adapter: one NVIDIA GPU holds the whole model, served by `vllm serve` from its own Linux venv.

On Windows the server runs inside the WSL2 distro recorded at install time and binds 127.0.0.1 there; WSL's NAT
localhost relay makes it reachable from the gateway (probed: a loopback-bound server inside WSL answers on Windows
127.0.0.1). The per-deployment API key travels as VLLM_API_KEY (shared into WSL with WSLENV `/u`), never in argv.
Facts and pins: docs/research/vllm.md (vLLM v0.32.0 = 5df63a7); source lines cited inline are at that commit.

Process model on Windows (probed on WSL 2, kernel 6.6.87): killing wsl.exe from Windows does NOT stop the Linux
process, so the launch wrapper records the vLLM session leader in a pidfile and starts a small watchdog that takes
the whole process group down when the wsl.exe side goes away. stop() signals inside the distro first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import shutil
import socket
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil

from .. import config, engine, netutil
from ..installers import vllm as installer
from ..installers.vllm import InstallError, VllmInstall
from ..schemas import (
    CandidateSummary,
    Deployment,
    DeviceAssignment,
    EngineInfo,
    ExtraEngineInfo,
    ModelArtifact,
    NodeState,
    NodeView,
    ObservedDevice,
    ObservedPlacement,
    Plan,
    PlanRequest,
    utcnow,
)
from ..units import GiB, MiB, fmt_bytes
from . import vllm_output
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
    artifact_engines,
)
from .registry import AdapterContext
from .vllm_output import StreamNormalizer, exit_detail, exit_reason, load_phase, parse_startup, startup_failure

log = logging.getLogger("hypercompute.vllm")

CAPABILITIES = next(c for c in ENGINE_ROADMAP if c.engine_id == "vllm")
PROCESS_KEY_PREFIX = "deployment:"  # same prefix as llama-server, so restart reconciliation keeps the leases
CLI_MODULE = "vllm.entrypoints.cli.main"  # the `vllm` console script (pyproject [project.scripts]); has __main__
MAX_UTILIZATION = 0.90  # research §2.4: cap; the 0.92 default fails on a shared laptop GPU
# vLLM compares total * utilization with the free memory it measures after its own CUDA context exists
# (v1/worker/utils.py request_memory), so the headroom covers that context plus the desktop. Research said 512 MiB;
# 1 GiB is a deliberate margin until the context size is measured on hardware.
VRAM_HEADROOM = 1 * GiB
RUNTIME_OVERHEAD = 1536 * MiB  # estimate: activations + CUDA graphs (~1 GiB, research §2.3) + non-torch allocations
DISK_HEADROOM = 2 * GiB  # beyond the checkpoint: torch.compile and FlashInfer caches, logs, a full disk breaking WSL
KV_FALLBACK_PER_B = 20 * 1024  # bytes of fp16 KV per token per billion params when the shape is unknown (generous)
KV_SHAPES: dict[str, tuple[int, int, int]] = {  # logical_id -> (layers, KV heads, head dim), from config.json
    "qwen2.5-7b-instruct": (28, 4, 128),  # Qwen/Qwen2.5-7B-Instruct-AWQ@b250375: 3584 hidden / 28 heads
}
MAX_NUM_SEQS = {"interactive": 8, "throughput": 16}
HEALTH_PROBE_S = 2.0  # pre-listen probes from Windows time out rather than refuse (probed through the WSL relay)
POLL_S = 1.0
STOP_GRACE_S = 30  # research §3.5: TERM, wait up to ~30 s, then KILL the process group
WSL_REAP_S = 10.0
NVIDIA_SMI_TIMEOUT_S = 10.0
SMI_QUERY = ["--query-gpu=index,name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]


# ----------------------------------------------------------------------------- GPU memory


@dataclass(frozen=True)
class GpuMemory:
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    used_bytes: int


def parse_nvidia_smi(text: str) -> list[GpuMemory]:
    """Rows of `nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits`.
    """
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not all(p.isdigit() for p in (parts[0], *parts[-3:])):
            continue  # "[N/A]" or a malformed row
        total, free, used = (int(p) * MiB for p in parts[-3:])
        out.append(GpuMemory(int(parts[0]), ", ".join(parts[1:-3]), total, free, used))
    return out


def query_gpus(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[GpuMemory]:
    """Fresh free/total VRAM from the host's nvidia-smi (on Windows it sees the same GPU WSL uses)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise EngineAdapterError("gpu_query_failed", "nvidia-smi not found", fix="install the NVIDIA driver")
    try:
        out = run([exe, *SMI_QUERY], capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EngineAdapterError("gpu_query_failed", f"nvidia-smi failed: {exc}",
                                 fix="check the NVIDIA driver with `nvidia-smi`") from exc
    rows = parse_nvidia_smi(out.stdout or "")
    if out.returncode != 0 or not rows:
        raise EngineAdapterError("gpu_query_failed", f"nvidia-smi reported no GPU memory (exit {out.returncode}): "
                                 f"{(out.stderr or out.stdout or '').strip()[:200]}", fix="run `nvidia-smi` to check")
    return rows


def gpu_holders() -> list[str]:
    """Hypercompute's own llama.cpp processes on this host, named in fix hints when VRAM is short."""
    names = {"llama-server", "ggml-rpc-server", "rpc-server"}
    out = []
    for p in psutil.process_iter(["pid", "name"]):
        name = str(p.info.get("name") or "")
        if name.lower().removesuffix(".exe") in names:
            out.append(f"{name} (pid {p.info['pid']})")
    return out


def utilization(budget: int, total: int) -> float:
    """--gpu-memory-utilization: the share of total VRAM vLLM may reserve, floored to 0.01 and capped at 0.90."""
    if total <= 0 or budget <= 0:
        return 0.0
    return math.floor(min(MAX_UTILIZATION, budget / total) * 100) / 100


def reserved_bytes(total: int, util: float) -> int:
    return math.ceil(total * util)  # vLLM: requested_memory = ceil(total_memory * gpu_memory_utilization)


def kv_bytes_per_token(art: ModelArtifact) -> tuple[int, str]:
    shape = KV_SHAPES.get(art.logical_id or "")
    if shape:
        layers, heads, dim = shape
        return 2 * layers * heads * dim * 2, f"{layers} layers x {heads} KV heads x {dim} dims, fp16 K and V"
    est = math.ceil(max(art.params_b, 1.0) * KV_FALLBACK_PER_B)
    return est, f"estimate ({KV_FALLBACK_PER_B // 1024} KiB per billion parameters; shape unknown)"


def weight_bytes(art: ModelArtifact) -> int | None:
    return art.size_bytes or (sum(f.size_bytes for f in art.files) if art.files else None)


@dataclass(frozen=True)
class Sizing:
    total: int
    free: int
    budget: int
    util: float
    weights: int
    kv_per_token: int
    ctx: int

    @property
    def reserved(self) -> int:
        return reserved_bytes(self.total, self.util)

    @property
    def need(self) -> int:
        return self.weights + self.kv_per_token * self.ctx + RUNTIME_OVERHEAD

    @property
    def fits(self) -> bool:
        return self.util > 0 and self.need <= self.reserved

    @property
    def kv_room(self) -> int:
        return max(0, self.reserved - self.weights - RUNTIME_OVERHEAD)

    def explain(self) -> str:
        return (f"needs ~{fmt_bytes(self.need)} ({fmt_bytes(self.weights)} weights + "
                f"{fmt_bytes(self.kv_per_token * self.ctx)} KV for {self.ctx} tokens + {fmt_bytes(RUNTIME_OVERHEAD)} "
                f"runtime) of {fmt_bytes(self.reserved)} reserved at --gpu-memory-utilization {self.util:.2f} "
                f"({fmt_bytes(self.free)} of {fmt_bytes(self.total)} free now)")


@dataclass(frozen=True)
class Download:
    """The checkpoint vLLM fetches on its first start (size_bytes), against free space where its cache lives."""

    need: int  # 0 when an earlier deployment already downloaded this repo@revision
    free: int | None  # None when the free space could not be read cheaply
    where: str

    @property
    def fits(self) -> bool:
        return self.need == 0 or self.free is None or self.need + DISK_HEADROOM <= self.free

    def shortfall(self, hf_home: str) -> str:
        return (f"does not fit on disk: the first start downloads ~{fmt_bytes(self.need)} into {hf_home}, and "
                f"{self.where} has {fmt_bytes(self.free)} free; free at least "
                f"{fmt_bytes(self.need + DISK_HEADROOM - (self.free or 0))} there first")


# ----------------------------------------------------------------------------- plan


def serve_args(model: str, revision: str, alias: str, port: int, util: float, max_len: int, max_seqs: int) -> list[str]:
    """`vllm serve` arguments (flags verified in arg_utils.py / cli_args.py at v0.32.0). No secret goes here."""
    return ["serve", model, "--revision", revision, "--served-model-name", alias, "--host", "127.0.0.1",
            "--port", str(port), "--gpu-memory-utilization", f"{util:.2f}", "--max-model-len", str(max_len),
            "--max-num-seqs", str(max_seqs), "--generation-config", "vllm", "--disable-uvicorn-access-log"]


def artifact_problems(art: ModelArtifact) -> list[str]:
    problems = []
    if art.format != "hf":
        problems.append(f"vLLM serves Hugging Face safetensors checkpoints (AWQ, GPTQ, FP8); {art.id} is "
                        f"{art.format}" + (": use llama.cpp for GGUF" if art.format == "gguf" else ""))
    elif "vllm" not in artifact_engines(art):
        problems.append(f"the model manifest does not list vllm among the engines for {art.id}")
    if weight_bytes(art) is None:
        problems.append(f"size of {art.id} is unknown: add size_bytes to its manifest entry")
    return problems


def _local_gpu(local: NodeView, warnings: list[str],
               query: Callable[[], list[GpuMemory]]) -> tuple[str, GpuMemory] | str:
    """(inventory device id, fresh memory) of the local NVIDIA GPU with the most free memory, or why there is none."""
    cuda = [d for d in local.inventory.devices if d.backend == "cuda" and d.kind == "gpu"]
    if not cuda:
        return "this machine has no NVIDIA GPU, and vLLM's CUDA engine needs one"
    usable = [d for d in cuda if d.id in local.contribution.devices]
    if not usable:
        return (f"{', '.join(d.id for d in cuda)} is not among this machine's shared devices "
                "(hypercompute host --local-devices)")
    try:
        rows = query()
    except EngineAdapterError as exc:
        warnings.append(f"{exc}; using the last inventory reading of free VRAM")
        rows = [GpuMemory(i, d.name, d.total_bytes or 0, d.free_bytes or 0, 0) for i, d in enumerate(cuda)]
    picks: list[tuple[str, GpuMemory]] = []
    seen: dict[str, int] = {}
    shared = {d.id for d in usable}
    for d in cuda:  # the k-th inventory GPU with a given name is the k-th nvidia-smi row with that name
        same = [r for r in rows if r.name == d.name]
        k = seen[d.name] = seen.get(d.name, -1) + 1
        if d.id in shared and k < len(same):
            picks.append((d.id, same[k]))
    if not picks:
        return f"nvidia-smi does not list {', '.join(d.name for d in usable)}"
    return max(picks, key=lambda p: p[1].free_bytes)


def _base(req: PlanRequest) -> dict:
    return dict(id=uuid.uuid4().hex, model_id=req.model_id, strategy=req.strategy, ctx_size=req.ctx_size,
                heuristic=True, created_at=utcnow(), device_list=[], tensor_split=[], rpc_endpoints=[],
                cpu_resident_bytes=0, engine="vllm", tier="gpu")


def _infeasible(req: PlanRequest, reasons: list[str], warnings: list[str], options: dict) -> Plan:
    return Plan(**_base(req), feasible=False, fit_label="fit_unknown", assignments=[], reasons=reasons,
                warnings=warnings, candidates=[], engine_options=options)


def plan(req: PlanRequest, art: ModelArtifact, local: NodeView, remote: list[NodeView], ctx: AdapterContext,
         query: Callable[[], list[GpuMemory]] = query_gpus,
         disk: Callable[[VllmInstall], tuple[int | None, str]] = installer.cache_disk) -> Plan:
    """vLLM on this machine's NVIDIA GPU, whole model, sized against free VRAM read from nvidia-smi now and, for the
    first start's download (size_bytes), against free disk where vLLM keeps its model cache."""
    warnings: list[str] = []
    if remote:
        warnings.append(f"vLLM runs on this machine's GPU only; the {len(remote)} selected peer(s) are not used "
                        "(vLLM cannot split one model across this mix of machines; llama.cpp RPC can)")
    try:
        rec = installer.read_install(ctx.paths)
        problem = installer.install_problem(rec, engine.host_os()) if rec else None
    except InstallError as exc:
        rec, problem = None, f"{exc} (fix: {exc.fix})"
    reasons = artifact_problems(art)
    if rec is None or problem:
        reasons.insert(0, problem or f"vLLM is not installed on this machine; run: {installer.INSTALL_HINT}")
    if local.state != NodeState.eligible:
        reasons.append(f"this machine is {local.state.value}, not ready to serve")
    gpu = _local_gpu(local, warnings, query)
    if isinstance(gpu, str):
        reasons.append(gpu)
    if reasons or rec is None or isinstance(gpu, str):
        return _infeasible(req, reasons, warnings, {"engine": "vllm"})
    cached = installer.is_served(ctx.paths, rec, art.repo, art.revision)
    free, where = disk(rec)
    return _fit_plan(req, art, local, rec, gpu, Download(0 if cached else weight_bytes(art) or 0, free, where),
                     warnings)


def _fit_plan(req: PlanRequest, art: ModelArtifact, local: NodeView, rec: VllmInstall, gpu: tuple[str, GpuMemory],
              dl: Download, warnings: list[str]) -> Plan:
    device, row = gpu
    budget = max(0, min(local.contribution.max_bytes, row.free_bytes - VRAM_HEADROOM))
    kv_tok, kv_basis = kv_bytes_per_token(art)
    size = Sizing(total=row.total_bytes, free=row.free_bytes, budget=budget, util=utilization(budget, row.total_bytes),
                  weights=weight_bytes(art) or 0, kv_per_token=kv_tok, ctx=req.ctx_size)
    max_seqs = MAX_NUM_SEQS[req.objective]
    reasons = [f"vLLM {rec.vllm} on {device} ({row.name}): {size.explain()}"]
    if not size.fits:
        holders = gpu_holders()
        reasons.append("does not fit: " + (f"{', '.join(holders)} holds GPU memory; stop the llama.cpp deployment "
                                           "first" if holders else "free GPU memory or lower ctx_size"))
    if not dl.fits:
        reasons.append(dl.shortfall(rec.hf_home))
    feasible = size.fits and dl.fits
    warnings += _warnings(art, rec, size, dl, kv_basis, max_seqs, req)
    options = {**_options(art, rec, req, device, row, size, max_seqs, kv_basis),
               "download_bytes": dl.need, "disk_free_bytes": dl.free}
    a = DeviceAssignment(node_id=local.id, device=device, endpoint=None, pool_id=_pool_id(local, device),
                         layers=list(range(KV_SHAPES[art.logical_id][0])) if art.logical_id in KV_SHAPES else [],
                         holds_output=True, weight_bytes=size.weights, kv_bytes=size.kv_room,
                         compute_bytes_est=RUNTIME_OVERHEAD, overhead_bytes_est=0,
                         total_bytes_est=min(size.reserved, budget),  # vLLM holds the whole reservation
                         budget_bytes=budget, fits=size.fits)
    misses = reasons if not size.fits else reasons[1:]  # a VRAM miss keeps its breakdown (reasons[0])
    cand = CandidateSummary(label="vLLM on this machine's GPU", devices=[f"{local.id}:{device}"], feasible=feasible,
                            reason="estimated fit" if feasible else "; ".join(misses))
    return Plan(**_base(req), feasible=feasible, fit_label="estimated_fit" if feasible else "does_not_fit",
                assignments=[a], reasons=reasons, warnings=warnings, candidates=[cand], engine_options=options)


def _pool_id(local: NodeView, device: str) -> str:
    dev = next((d for d in local.inventory.devices if d.id == device), None)
    return dev.pool_id if dev else f"vram:{device}"


def _warnings(art: ModelArtifact, rec: VllmInstall, size: Sizing, dl: Download, kv_basis: str, max_seqs: int,
              req: PlanRequest) -> list[str]:
    where = f"inside WSL2 ({rec.distro})" if rec.platform == "wsl2" else "on this machine"
    if dl.need:
        free = f"; {dl.where} has {fmt_bytes(dl.free)} free" if dl.free is not None else ""
        out = [f"vLLM downloads {art.repo} at revision {art.revision[:12]} itself, {where}, into {rec.hf_home} on the "
               f"first start (~{fmt_bytes(dl.need)}{free}); later starts reuse it. The first start also compiles "
               "CUDA graphs and kernels, so expect several minutes."]
        if dl.free is None:
            out.append(f"free disk space of {dl.where} could not be read, so the download was not checked against it: "
                       f"make sure at least {fmt_bytes(dl.need + DISK_HEADROOM)} is free there")
    else:
        out = [f"vLLM reuses the copy of {art.repo}@{art.revision[:12]} it downloaded {where} on an earlier start"]
    if art.files:
        out.append(f"the manifest lists files for {art.id}, but vLLM never reads Hypercompute's copy: it downloads "
                   f"{art.repo} by revision itself. List files: [] (with size_bytes) to skip the extra download.")
    if art.logical_id not in KV_SHAPES:
        out.append(f"KV cache size per token is an {kv_basis}")
    full = size.kv_room // max(1, size.kv_per_token * size.ctx)
    if size.fits and full < max_seqs:
        out.append(f"the KV cache holds about {full} full-length request(s) at once (--max-num-seqs {max_seqs})")
    if req.objective == "interactive":
        out.append("for one user vLLM is about as fast as llama.cpp; its advantage is many concurrent requests")
    return out


def _options(art: ModelArtifact, rec: VllmInstall, req: PlanRequest, device: str, row: GpuMemory, size: Sizing,
             max_seqs: int, kv_basis: str) -> dict:
    port = netutil.free_loopback_port()
    return {"engine": "vllm", "version": rec.vllm, "pin": f"vllm=={rec.vllm} torch=={rec.torch}",
            "platform": rec.platform, "distro": rec.distro, "model": art.repo, "revision": art.revision,
            "served_model_name": req.model_id, "device": device, "gpu_index": row.index, "gpu_name": row.name,
            "port": port, "gpu_memory_utilization": size.util, "max_model_len": req.ctx_size,
            "max_num_seqs": max_seqs, "need_bytes": size.need, "reserved_bytes": size.reserved,
            "kv_bytes_per_token": size.kv_per_token, "kv_basis": kv_basis, "hf_home": rec.hf_home,
            "engine_managed": not art.files,
            "args": serve_args(art.repo, art.revision, req.model_id, port, size.util, req.ctx_size, max_seqs),
            "env": {**launch_env(rec, row.index), "VLLM_API_KEY": "<per-deployment key, set at launch>"}}


def launch_env(rec: VllmInstall, gpu_index: int) -> dict[str, str]:
    """Non-secret environment of the vLLM server (the API key is added separately)."""
    return {"HF_HOME": rec.hf_home, "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_index)}


def _arg_after(args: list[str], flag: str) -> str:
    i = args.index(flag) if flag in args else -1
    return args[i + 1] if 0 <= i < len(args) - 1 else ""


def served_name(handle: DeploymentHandle) -> str:
    return _arg_after(handle.args, "--served-model-name")


# ----------------------------------------------------------------------------- adapter


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wslenv_with(name: str, current: str | None) -> str:
    """WSLENV that also shares `name` from Windows into WSL (`/u`: only Win32 -> WSL), keeping existing entries."""
    entries = [e for e in (current or "").split(":") if e and e.split("/")[0] != name]
    return ":".join([*entries, f"{name}/u"])


@dataclass(frozen=True)
class Launch:
    exe: Path
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    display: list[str]


def launch_command(rec: VllmInstall, wsl: str | None, pidfile: str, args: list[str], gpu_index: int, api_key: str,
                   logs_dir: Path) -> Launch:
    server = [f"{rec.venv}/bin/python", "-m", CLI_MODULE, *args]
    base = launch_env(rec, gpu_index)
    if rec.platform == "linux":
        env = {**base, "VLLM_API_KEY": api_key, "PYTHONPATH": ""}  # never inherit the gateway venv's path
        return Launch(Path(server[0]), server[1:], Path(rec.root), env, ["vllm", *args])
    if not wsl or not rec.distro:
        raise EngineAdapterError("wsl_missing", "wsl.exe or the WSL distro recorded at install time is missing",
                                 fix=f"{installer.INSTALL_HINT} --force")
    argv = ["-d", rec.distro, "--cd", "~", "--exec", "setsid", "-w", "env", *(f"{k}={v}" for k, v in base.items()),
            "bash", "-c", vllm_output.LAUNCH, pidfile, *server]
    env = {"VLLM_API_KEY": api_key, "WSLENV": wslenv_with("VLLM_API_KEY", os.environ.get("WSLENV")),
           "WSL_UTF8": "1"}
    return Launch(Path(wsl), argv, logs_dir, env, ["wsl.exe", "-d", rec.distro, "--", "vllm", *args])


class VllmAdapter:
    engine_id = "vllm"
    idle_timeout_s = config.UPSTREAM_IDLE_TIMEOUT_S
    failure_reason = staticmethod(exit_reason)  # the deployment watcher's message when a ready server exits

    def __init__(self, rec: VllmInstall, ctx: AdapterContext, query: Callable[[], list[GpuMemory]] = query_gpus,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 disk: Callable[[VllmInstall], tuple[int | None, str]] | None = None) -> None:
        self._rec, self._sup, self._logs, self._query, self._run = rec, ctx.supervisor, ctx.logs_dir, query, runner
        self._paths, self._disk = ctx.paths, disk or installer.cache_disk
        self._wsl = installer.wsl_exe() if rec.platform == "wsl2" else None
        self._util: dict[str, float] = {}  # deployment -> utilization re-checked by validate()
        self._downloads: dict[str, int] = {}  # deployment -> bytes its first start downloads (set by prepare)
        self._active: dict[str, httpx.Response] = {}
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(config.UPSTREAM_IDLE_TIMEOUT_S, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def inspect(self) -> EngineSnapshot:
        rec = self._rec
        info = EngineInfo(engine="vllm", installed=True, release=f"vllm {rec.vllm}",
                          variant=f"{rec.platform}:{rec.distro or 'host'}",
                          accelerator="cuda", bin_dir=f"{rec.venv}/bin", smoke="not_run",
                          smoke_detail=f"torch {rec.torch}, {rec.device} capability {rec.capability}")
        return EngineSnapshot(capabilities=CAPABILITIES, info=info)

    def pidfile(self, deployment_id: str) -> str:
        return f"{self._rec.run_dir}/{deployment_id}.pid"

    # ------------------------------------------------------------------ validate / prepare
    def _static_problems(self, spec: DeploymentSpec) -> list[str]:
        opts = spec.plan.engine_options
        problems = [] if spec.plan.engine == "vllm" else [f"plan is for {spec.plan.engine}, not vllm"]
        if not isinstance(opts.get("args"), list) or not isinstance(opts.get("gpu_index"), int):
            problems.append("plan has no vLLM launch arguments; check the plan again")
        if spec.artifact is not None:
            problems += artifact_problems(spec.artifact)
        if self._rec.platform == "wsl2" and self._wsl is None:
            problems.append("wsl.exe not found; vLLM on Windows runs inside WSL2 (wsl --install -d Ubuntu)")
        return problems

    def _fresh_util(self, spec: DeploymentSpec) -> tuple[float | None, list[str]]:
        """Re-read free VRAM right before launch: anything that took the GPU since planning fails here, with a fix."""
        opts = spec.plan.engine_options
        try:
            rows = self._query()
        except EngineAdapterError as exc:
            return None, [f"{exc} (fix: {exc.fix})"]
        row = next((r for r in rows if r.index == opts["gpu_index"]), None)
        if row is None:
            return None, [f"GPU {opts['gpu_index']} ({opts.get('gpu_name')}) is no longer listed by nvidia-smi"]
        util = min(float(opts["gpu_memory_utilization"]), utilization(row.free_bytes - VRAM_HEADROOM, row.total_bytes))
        need = int(opts.get("need_bytes", 0))
        if util <= 0 or reserved_bytes(row.total_bytes, util) < need:
            holders = gpu_holders()
            who = f"{', '.join(holders)} holds GPU memory" if holders else "another process holds GPU memory"
            return None, [f"only {fmt_bytes(row.free_bytes)} of {fmt_bytes(row.total_bytes)} VRAM is free now and "
                          f"vLLM needs ~{fmt_bytes(need + VRAM_HEADROOM)} ({who}); fix: stop the llama.cpp "
                          "deployment or close GPU apps, or a stale vLLM in WSL: wsl -- pkill -TERM -f "
                          f"'{CLI_MODULE} serve'"]
        return util, []

    def _disk_problems(self, spec: DeploymentSpec) -> list[str]:
        """Re-check disk room for the first start's download: a stored plan can be deployed long after planning."""
        opts = spec.plan.engine_options
        if installer.is_served(self._paths, self._rec, str(opts.get("model")), str(opts.get("revision"))):
            return []
        need = (weight_bytes(spec.artifact) if spec.artifact else None) or int(opts.get("download_bytes") or 0)
        dl = Download(need, *self._disk(self._rec))
        if dl.need and dl.free is None:
            log.warning("free disk space of %s unknown; the ~%s download was not checked", dl.where,
                        fmt_bytes(dl.need))
        return [] if dl.fits else [dl.shortfall(self._rec.hf_home)]

    async def validate(self, spec: DeploymentSpec) -> ValidationResult:
        problems = self._static_problems(spec)
        if not problems:
            util, gpu_problems = await asyncio.to_thread(self._fresh_util, spec)
            problems += gpu_problems + await asyncio.to_thread(self._disk_problems, spec)
            if util is not None and not problems:
                self._util[spec.deployment_id] = util
        return ValidationResult(ok=not problems, problems=problems)

    def max_concurrency(self, plan: Plan) -> int:
        """--max-num-seqs the plan launches with: how many requests vLLM batches at once (the gateway's cap)."""
        try:
            return max(1, int(plan.engine_options.get("max_num_seqs") or 1))
        except (TypeError, ValueError):
            return 1

    def load_timeout_s(self, plan: Plan) -> float:
        """wait_ready's budget: LOAD_TIMEOUT_S plus the first start's in-distro download, capped at 4 h."""
        return vllm_output.load_budget(vllm_output.download_need(self._paths, self._rec, plan))

    def _launch_args(self, spec: DeploymentSpec) -> list[str]:
        opts = spec.plan.engine_options
        port = int(opts["port"])
        if not _port_free(port):
            port = netutil.free_loopback_port()
        util = self._util.pop(spec.deployment_id, float(opts["gpu_memory_utilization"]))
        return serve_args(str(opts["model"]), str(opts["revision"]), spec.alias, port, util,
                          int(opts["max_model_len"]), self.max_concurrency(spec.plan))

    async def prepare(self, spec: DeploymentSpec) -> DeploymentHandle:
        args = self._launch_args(spec)
        api_key = secrets.token_urlsafe(24)
        launch = launch_command(self._rec, self._wsl, self.pidfile(spec.deployment_id), args,
                                int(spec.plan.engine_options["gpu_index"]), api_key, self._logs)
        key = f"{PROCESS_KEY_PREFIX}{spec.deployment_id}"
        log_path = self._logs / f"vllm-{spec.deployment_id}.log"
        proc = await asyncio.to_thread(self._sup.start, key, launch.exe, launch.argv, launch.cwd, log_path,
                                       launch.env)
        self._downloads[spec.deployment_id] = vllm_output.download_need(self._paths, self._rec, spec.plan)
        port = args[args.index("--port") + 1]
        return DeploymentHandle(deployment_id=spec.deployment_id, process_key=key, base_url=f"http://127.0.0.1:{port}",
                                log_path=log_path, pid=proc.pid, args=launch.display, api_key=api_key)

    # ------------------------------------------------------------------ readiness
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

    async def _health(self, handle: DeploymentHandle) -> int | None:
        try:
            r = await self._client.get(f"{handle.base_url}/health", timeout=HEALTH_PROBE_S)
        except httpx.TransportError:  # refused, reset or (through the WSL relay) a connect timeout before listen()
            return None
        return r.status_code

    def _check_startup(self, handle: DeploymentHandle) -> str:
        text = self.read_log(handle)
        failure = startup_failure(text)
        if failure:
            raise EngineAdapterError("engine_failed", failure[0], fix=failure[1])
        if not self.process_alive(handle):
            proc = self._sup.get(handle.process_key)
            where = f"; check WSL with: wsl -d {self._rec.distro} -- true" if self._rec.platform == "wsl2" else ""
            raise EngineAdapterError("engine_exited", exit_detail(text, proc.exit_code() if proc else None),
                                     fix=f"read {handle.log_path}{where}")
        return load_phase(text, self._rec)

    async def wait_ready(self, handle: DeploymentHandle, timeout_s: float | None = None,
                         on_progress: Callable[[str], None] | None = None) -> None:
        """Poll /health until vLLM serves. No timeout_s: load_timeout_s of the plan prepare() launched."""
        need = self._downloads.pop(handle.deployment_id, 0)
        timeout_s = vllm_output.load_budget(need) if timeout_s is None else timeout_s
        dl = vllm_output.DownloadProgress(self._rec, self._wsl, self._run, _arg_after(handle.args, "serve"), need)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last = ""
        while True:
            phase = await dl.phase(loop.time(), self._check_startup(handle))
            if phase != last and on_progress:
                on_progress(phase)
                last = phase
            status = await self._health(handle)
            if status == 200:  # the socket is served only after load, KV allocation and warmup (research §3.3)
                break
            if status == 503:
                raise EngineAdapterError("engine_dead", "vLLM answered 503 on /health: its engine core died",
                                         fix=f"read {handle.log_path}")
            if loop.time() >= deadline:
                raise EngineAdapterError("load_timeout", f"vLLM was not ready after {timeout_s:.0f}s ({last})",
                                         fix=f"read {handle.log_path}; the first start can take many minutes")
            await asyncio.sleep(POLL_S)
        await self._check_models(handle)
        self._remember_cached(handle)

    def _remember_cached(self, handle: DeploymentHandle) -> None:
        """The served checkpoint is now in vLLM's cache: later plans need no disk room for its download."""
        repo, revision = _arg_after(handle.args, "serve"), _arg_after(handle.args, "--revision")
        if not repo or not revision:
            return
        try:
            installer.remember_served(self._paths, self._rec, repo, revision)
        except OSError as exc:
            log.warning("could not record that %s@%s is in vLLM's cache: %s", repo, revision, exc)

    async def _check_models(self, handle: DeploymentHandle) -> None:
        alias = served_name(handle)
        try:
            r = await self._client.get(f"{handle.base_url}/v1/models", headers=self._auth(handle), timeout=10.0)
            ids = [m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict)] if r.status_code == 200 \
                else []
        except (httpx.HTTPError, json.JSONDecodeError, AttributeError) as exc:
            raise EngineAdapterError("models_check_failed", f"GET /v1/models failed: {exc}",
                                     fix=f"read {handle.log_path}") from exc
        if r.status_code == 401:
            raise EngineAdapterError("auth_failed", "vLLM rejected the deployment's API key on /v1/models",
                                     fix="the key reaches WSL through WSLENV; check that WSLENV is not overridden")
        if alias not in ids:
            raise EngineAdapterError("alias_missing", f"vLLM serves {ids or 'nothing'} (HTTP {r.status_code}), "
                                     f"not {alias!r}", fix="stop the deployment and check the plan again")

    async def observe(self, handle: DeploymentHandle, spec: DeploymentSpec,
                      validation: ValidationResult) -> PlacementReport:
        values, evidence = parse_startup(self.read_log(handle))
        opts = spec.plan.engine_options
        device = ObservedDevice(name=f"{opts.get('device', 'CUDA0')} ({opts.get('gpu_name', 'GPU')})",
                                model_bytes=int(values.get("weights", 0) * GiB),
                                kv_bytes=int(values.get("kv", 0) * GiB),
                                compute_bytes=int(values.get("graphs_gib", 0) * GiB), layers_est=None)
        placement = ObservedPlacement(offloaded=None, devices=[device], evidence_lines=evidence, parsed_at=utcnow())
        if "weights" not in values:
            return PlacementReport(observed=placement, verified=True,
                                   detail="vLLM serves the whole model on one GPU (memory breakdown not in its log)")
        detail = (f"vLLM holds the whole model on {device.name}: weights {values['weights']:.2f} GiB, KV cache "
                  f"{values.get('kv', 0):.2f} GiB")
        if "kv_tokens" in values:
            detail += (f" ({values['kv_tokens']:,.0f} tokens, {values['concurrency']:.1f}x concurrency at "
                       f"{values['max_len']:,.0f} tokens)")
        if "init" in values:
            detail += f"; engine init {values['init']:.1f} s, weights load {values['load_s']:.1f} s"
        return PlacementReport(observed=placement, verified=True, detail=detail)

    # ------------------------------------------------------------------ stop
    def _stop_in_distro(self, distro: str, pidfile: str) -> tuple[str, str]:
        """TERM the vLLM session inside the distro, wait for its process group, else KILL it."""
        argv = [self._wsl or "wsl.exe", "-d", distro, "--exec", "bash", "-c", vllm_output.STOP, pidfile,
                str(STOP_GRACE_S)]
        try:
            out = self._run(argv, capture_output=True, timeout=STOP_GRACE_S + 30, stdin=subprocess.DEVNULL,
                            env=installer.wsl_env(), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return "unknown", f"could not signal vLLM inside WSL: {exc}"
        text = out.stdout.decode("utf-8", errors="replace") if isinstance(out.stdout, bytes) else str(out.stdout)
        words = text.split()
        how = words[-1] if words else "unknown"
        return how, f"vLLM in WSL ({distro}): {how}"

    async def stop(self, handle: DeploymentHandle) -> StopResult:
        inner = await asyncio.to_thread(self._stop_in_distro, self._rec.distro or "",
                                        self.pidfile(handle.deployment_id)) if self._rec.platform == "wsl2" else None
        grace = WSL_REAP_S if inner else float(STOP_GRACE_S)
        outcome = await asyncio.to_thread(self._sup.stop, handle.process_key, grace)
        notes = [inner[1]] if inner else []
        notes.append(outcome.describe() if outcome else "supervised process was not running")
        clean = (outcome is None or outcome.clean) and (inner is None or inner[0] in vllm_output.INNER_CLEAN)
        return StopResult(clean=clean, detail="; ".join(notes))

    async def cleanup_orphan(self, dep: Deployment) -> tuple[bool, str]:
        """After a gateway crash: the server a deployment left in its WSL distro (the restart kills wsl.exe, which does
        not stop it once its watchdog is gone) is stopped through stop()'s TERM-then-KILL path. A distro that is not
        running holds nothing, so WSL is never started for this. True once nothing of it runs."""
        return await asyncio.to_thread(self._cleanup_orphan, dep)

    def _cleanup_orphan(self, dep: Deployment) -> tuple[bool, str]:
        opts, refuse = dep.plan.engine_options, f"cannot clean up vLLM for deployment {dep.id[:8]}"
        if dep.plan.engine != "vllm":
            return False, f"{refuse}: it is a {dep.plan.engine} deployment"
        if opts.get("platform", self._rec.platform) != "wsl2":  # no pidfile: the server was the supervised process
            return True, ("not checked further: on native Linux vLLM was the supervised process, and the restart "
                          "stopped its leader (children in its process group are not checked)")
        distro, pidfile = vllm_output.orphan_pidfile(self._rec, opts, dep.id)
        running = vllm_output.distro_running(self._run, self._wsl, distro)
        if isinstance(running, str):
            return False, f"{refuse}: {running}; check with: wsl -l --running"
        if not running:
            return True, f"nothing left: WSL distro {distro} is not running"
        how, detail = self._stop_in_distro(distro, pidfile)
        if how in vllm_output.INNER_GONE:
            return True, f"nothing left: {detail}"
        if how in vllm_output.INNER_CLEAN:
            return True, f"stopped the vLLM server it left running: {detail}"
        return False, f"{detail}; fix: wsl -d {distro} -- pkill -KILL -f '{CLI_MODULE} serve'"

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _auth(handle: DeploymentHandle) -> dict[str, str]:
        return {"Authorization": f"Bearer {handle.api_key}"} if handle.api_key else {}

    async def complete(self, handle: DeploymentHandle, req: InferenceRequest) -> CompletionResult:
        body = vllm_output.chat_body(req.body, served_name(handle), stream=False)
        r = await self._client.post(f"{handle.base_url}/v1/chat/completions", json=body, headers=self._auth(handle))
        try:
            payload = r.json()
        except json.JSONDecodeError:
            payload = {"error": {"message": r.text[:500]}}
        return CompletionResult(status_code=r.status_code, body=payload)

    async def stream(self, handle: DeploymentHandle, req: InferenceRequest) -> AsyncIterator[GenerationEvent]:
        body = vllm_output.chat_body(req.body, served_name(handle), stream=True)
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        norm = StreamNormalizer()
        async with self._client.stream("POST", f"{handle.base_url}/v1/chat/completions", json=body,
                                       headers=self._auth(handle)) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                raise EngineAdapterError("upstream_error", f"vLLM HTTP {resp.status_code}: {text[:400]}")
            self._active[req.request_id] = resp
            try:
                async for chunk in resp.aiter_raw():
                    for ev in norm.feed(chunk):
                        yield ev
                for ev in norm.flush():
                    yield ev
            finally:
                self._active.pop(req.request_id, None)

    async def cancel(self, request_id: str) -> CancellationResult:
        resp = self._active.pop(request_id, None)
        if resp is None:
            return CancellationResult(request_id, False, "no active upstream request")
        await resp.aclose()  # vLLM's with_cancellation aborts the request when the client disconnects
        return CancellationResult(request_id, True, "upstream connection closed")


# ----------------------------------------------------------------------------- module contract (adapters.registry)


def _usable_install(paths: config.Paths) -> VllmInstall | None:
    try:
        rec = installer.read_install(paths)
    except InstallError as exc:
        log.warning("vLLM install record unusable: %s", exc)
        return None
    if rec is None:
        return None
    problem = installer.install_problem(rec, engine.host_os())
    if problem:
        log.warning("vLLM not loaded: %s", problem)
        return None
    return rec


def build(ctx: AdapterContext) -> VllmAdapter | None:
    rec = _usable_install(ctx.paths)
    return VllmAdapter(rec, ctx) if rec else None


def status(ctx: AdapterContext) -> ExtraEngineInfo:
    return installer.status(ctx.paths)

