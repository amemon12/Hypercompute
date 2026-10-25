"""Engine adapter contract. One model instance has exactly one engine owner.

Engines: `llama_rpc` (llama.cpp layer split, the backbone), `vllm` (single NVIDIA Linux/WSL2 node), `exo`
(Apple-only MLX cluster) and `airllm` (disk-streamed capacity mode). Each optional engine lives in its own
environment and module and is built through adapters.registry; ENGINE_ROADMAP reports what each supports.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from ..schemas import EngineId, EngineInfo, ModelArtifact, ObservedPlacement, Plan


class Capabilities(BaseModel):
    engine_id: str
    status: Literal["implemented", "planned"]
    streaming: bool
    multi_host: bool
    disk_offload: bool
    cancel: bool
    platforms: list[str]
    notes: str


ENGINE_ROADMAP: list[Capabilities] = [
    Capabilities(engine_id="llama_rpc", status="implemented", streaming=True, multi_host=True, disk_offload=False,
                 cancel=True, platforms=["macos-arm64 (Metal)", "windows-x64 (CUDA/CPU)", "linux (CUDA/CPU)"],
                 notes="llama.cpp layer split over ggml RPC; unauthenticated RPC, trusted private LAN only"),
    Capabilities(engine_id="vllm", status="implemented", streaming=True, multi_host=False, disk_offload=False,
                 cancel=True, platforms=["linux x86_64/aarch64 (CUDA)", "windows-x64 via WSL2 (CUDA)"],
                 notes="one NVIDIA GPU holds the whole model; continuous batching for concurrent requests"),
    Capabilities(engine_id="exo", status="implemented", streaming=True, multi_host=True, disk_offload=False,
                 cancel=True, platforms=["macos-arm64"],
                 notes="Apple-only MLX pipeline over the LAN; unauthenticated API, trusted private LAN only"),
    Capabilities(engine_id="airllm", status="implemented", streaming=True, multi_host=False, disk_offload=True,
                 cancel=True, platforms=["windows-x64 (CUDA/CPU)", "linux (CUDA/CPU)"],
                 notes="disk capacity mode: every layer is read from disk for every token"),
    Capabilities(engine_id="exo_paged", status="planned", streaming=False, multi_host=False, disk_offload=False,
                 cancel=False, platforms=[], notes="phase 4 research: layer offload inside EXO"),
]


# What each engine's serving process is called in progress and failure messages.
ENGINE_PROCESS_LABEL: dict[str, str] = {"llama_rpc": "llama-server", "vllm": "vLLM server",
                                        "exo": "EXO instance", "airllm": "AirLLM worker"}


def artifact_engines(art: ModelArtifact) -> list[str]:
    """Engines that can serve an artifact: the manifest's list, else the default for its format."""
    if art.engines:
        return list(art.engines)
    return {"gguf": ["llama_rpc"], "hf": ["airllm", "vllm"], "mlx": ["exo"]}[art.format]


class EngineAdapterError(RuntimeError):
    def __init__(self, code: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.code, self.fix = code, fix


class UnsupportedCapability(EngineAdapterError):
    def __init__(self, engine_id: str, capability: str) -> None:
        super().__init__("unsupported_capability", f"{engine_id} does not support {capability}")


@dataclass(frozen=True)
class EngineSnapshot:
    capabilities: Capabilities
    info: EngineInfo


@dataclass(frozen=True)
class DeploymentSpec:
    deployment_id: str
    plan: Plan
    model_path: Path  # GGUF: first shard; HF: the model directory; MLX/EXO: unused (EXO downloads itself)
    alias: str
    artifact: ModelArtifact | None = None
    model_dir: Path | None = None
    mmproj_path: Path | None = None  # verified vision projector (llama.cpp --mmproj), when the model has one here


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    problems: list[str]
    rpc_device_map: dict[str, list[str]] = field(default_factory=dict)  # endpoint -> coordinator device names


@dataclass(frozen=True)
class DeploymentHandle:
    deployment_id: str
    process_key: str
    base_url: str
    log_path: Path
    pid: int
    args: list[str]  # redacted argument list for display
    api_key: str = field(repr=False, default="")


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    body: dict
    stream: bool


@dataclass(frozen=True)
class CompletionResult:
    status_code: int
    body: dict


@dataclass(frozen=True)
class GenerationEvent:
    """One SSE event from the engine. `raw` is forwarded verbatim; `payload` is parsed JSON or None."""

    raw: bytes
    data: str
    payload: dict | None
    done: bool = False


@dataclass(frozen=True)
class CancellationResult:
    request_id: str
    cancelled: bool
    detail: str


@dataclass(frozen=True)
class StopResult:
    clean: bool
    detail: str


@dataclass(frozen=True)
class PlacementReport:
    """What the engine itself reports about where the model landed, checked against the plan."""

    observed: ObservedPlacement | None
    verified: bool  # every planned peer/device really holds work (or the engine has no peers to check)
    detail: str


class EngineAdapter(Protocol):
    """Required surface. Optional hooks the gateway uses when an adapter defines them:
    load_timeout_s(plan) -> float          load budget for wait_ready (first starts that download need longer)
    idle_timeout_for(plan) -> float        per-deployment silence budget (warmup of slow engines)
    max_concurrency(plan) -> int           requests the engine serves at once (default 1)
    failure_reason(log_text) -> str|None   why a dead deployment died, in the engine's own terms
    async cleanup_orphan(dep) -> (bool, str)  after a gateway crash: remove what the engine left running for a
                                           deployment that has no local process (True when confirmed gone)
    """

    engine_id: EngineId
    idle_timeout_s: float  # longest silence allowed between streamed events (capacity mode is minutes)

    async def inspect(self) -> EngineSnapshot: ...
    async def validate(self, spec: DeploymentSpec) -> ValidationResult: ...
    async def prepare(self, spec: DeploymentSpec) -> DeploymentHandle: ...
    async def wait_ready(self, handle: DeploymentHandle, timeout_s: float = ...,
                         on_progress: Callable[[str], None] | None = None) -> None: ...
    async def complete(self, handle: DeploymentHandle, req: InferenceRequest) -> CompletionResult: ...
    def stream(self, handle: DeploymentHandle, req: InferenceRequest) -> AsyncIterator[GenerationEvent]: ...
    async def cancel(self, request_id: str) -> CancellationResult: ...
    async def stop(self, handle: DeploymentHandle) -> StopResult: ...
    async def observe(self, handle: DeploymentHandle, spec: DeploymentSpec,
                      validation: ValidationResult) -> PlacementReport: ...
    def read_log(self, handle: DeploymentHandle) -> str: ...
    def process_alive(self, handle: DeploymentHandle) -> bool: ...
    async def aclose(self) -> None: ...
