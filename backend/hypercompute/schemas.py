"""Typed contracts shared by the gateway, node agent, planner and dashboard.

Conventions: sizes are bytes (int), times are timezone-aware UTC datetimes, and an unavailable
measurement is ``None`` (never 0). Hardware capacity, backend-visible memory, a free-memory
observation and an allowed contribution are deliberately separate fields.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# One model instance has exactly one engine owner; engines never share a model's layers with each other.
EngineId = Literal["llama_rpc", "exo", "airllm", "vllm"]
# gpu: every layer on accelerators; gpu_ram: the first layers run on the host CPU from RAM;
# gpu_disk: like gpu_ram but the CPU part exceeds free RAM and is paged from the GGUF on disk (slow);
# stream: the CPU part is split into a resident subset (private copies) and units read from the SSD every pass into
#   a fixed number of slot buffers by the hcstream extension: bounded memory, measured disk reads (Plan.stream);
# disk: AirLLM streams every layer from disk for every token (capacity mode, very slow).
# moe_offload: a mixture-of-experts model with every layer on the accelerators (-ngl all) except the routed-expert
#   tensors of some host layers, which stay on the host CPU file-backed (mmap, no repack): only the experts a token
#   routes to are read, from the file cache or the SSD (Plan.moe).
Tier = Literal["gpu", "gpu_ram", "gpu_disk", "stream", "disk", "moe_offload"]


# --------------------------------------------------------------------------- inventory


class Diagnostic(Model):
    level: Literal["info", "warning", "error"]
    code: str
    message: str
    fix: str | None = None


class OSInfo(Model):
    system: Literal["windows", "macos", "linux", "other"]
    release: str
    version: str
    machine: str  # normalized arch: x86_64 | arm64 | ...


class CpuInfo(Model):
    model: str | None
    physical_cores: int | None
    logical_cores: int | None


class MemoryPool(Model):
    """One physical memory pool. Apple and GB10 CPU/GPU share one `unified` pool, counted once."""

    id: str  # "ram" | "uma" | "vram:<device>"
    kind: Literal["system_ram", "unified", "discrete_vram"]
    total_bytes: int | None
    free_bytes: int | None
    measured_at: datetime | None
    source: str
    note: str | None = None


class Device(Model):
    id: str  # engine device name when known (CUDA0, MTL0, CPU), else a vendor-tool name
    kind: Literal["cpu", "gpu", "igpu", "accel"]
    name: str
    vendor: str | None = None
    pool_id: str
    backend: Literal["cuda", "metal", "vulkan", "cpu", "rocm", "unknown"] = "unknown"
    total_bytes: int | None = None  # what the vendor tool / engine reports as capacity
    free_bytes: int | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    source: str


class EngineDevice(Model):
    """A device line from `llama-server --list-devices`, i.e. what the engine itself can use."""

    name: str  # CUDA0 | MTL0 | RPC0 | CPU ...
    description: str
    total_bytes: int | None
    free_bytes: int | None


class EngineInfo(Model):
    engine: EngineId = "llama_rpc"
    installed: bool
    release: str | None = None  # b11770
    build_number: int | None = None
    commit: str | None = None
    rpc_proto: str | None = None  # 8.0.0 (from ggml-rpc-server startup banner)
    variant: str | None = None  # windows-x64-cuda13.4 | macos-arm64 | ...
    accelerator: Literal["cuda", "metal", "vulkan", "cpu", "rocm", "unknown"] = "unknown"
    bin_dir: str | None = None
    smoke: Literal["passed", "failed", "not_run"] = "not_run"
    smoke_detail: str | None = None
    devices: list[EngineDevice] = Field(default_factory=list)


class NetworkInterface(Model):
    name: str
    ipv4: list[str]
    is_up: bool
    speed_mbps: int | None
    kind: Literal["wifi", "ethernet", "virtual", "loopback", "other"]


class ExtraEngineInfo(Model):
    """An optional engine installed in its own environment (AirLLM, vLLM, EXO), never in the gateway's."""

    engine: EngineId
    installed: bool
    status: Literal["not_installed", "installed", "running", "failed", "unsupported"]
    version: str | None = None  # pinned commit or release
    endpoint: str | None = None  # e.g. an EXO API host:port the agent started
    detail: str | None = None


class HostInventory(Model):
    hostname: str
    os: OSInfo
    cpu: CpuInfo
    memory_pools: list[MemoryPool]
    devices: list[Device]
    disk_free_bytes: int | None
    network: list[NetworkInterface]
    engine: EngineInfo
    collected_at: datetime
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    extra_engines: list[ExtraEngineInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- join / nodes


class ContributionLimits(Model):
    """Soft admission budget. Not an enforced process/GPU cap (llama.cpp offers none for RPC workers)."""

    devices: list[str] = Field(max_length=64)  # allowlisted engine device ids exposed through the RPC worker
    max_bytes: int = Field(ge=0)  # total bytes Hypercompute may plan onto this node
    reserve_bytes: int = Field(ge=0)  # headroom kept for the owner's OS/apps
    enforcement: Literal["soft_admission"] = "soft_admission"
    cache_enabled: bool = False  # ggml-rpc-server --cache (tensors persist on disk until purged)


class WorkerEndpoint(Model):
    host: str
    port: int = Field(ge=1, le=65535)


class JoinRequest(Model):
    code: str = Field(min_length=4, max_length=32)
    node_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    display_name: str = Field(min_length=1, max_length=64)
    inventory: HostInventory
    contribution: ContributionLimits
    worker: WorkerEndpoint
    agent_version: str = Field(max_length=64)


class CoordinatorEngineIdentity(Model):
    release: str | None
    commit: str | None
    rpc_proto: str | None


class JoinResponse(Model):
    session_id: str
    node_id: str
    credential: str
    heartbeat_interval_s: float
    coordinator_engine: CoordinatorEngineIdentity
    state: NodeState
    # Optional host capabilities (MANAGED_PEERS_FEATURE). An agent sends the fields a feature adds only when its
    # host listed that feature here, so an older host never sees them.
    features: list[str] = Field(default_factory=list, max_length=32)


class WorkerState(StrEnum):
    starting = "starting"
    running = "running"
    exited = "exited"
    stopped = "stopped"
    paused = "paused"


class HeartbeatRequest(Model):
    worker_state: WorkerState
    worker_exit_code: int | None = None
    memory_pools: list[MemoryPool] = Field(default_factory=list, max_length=64)
    devices: list[Device] = Field(default_factory=list, max_length=64)
    worker_cpu_seconds: float | None = None  # rpc-server process CPU time, used as execution evidence
    sent_at: datetime
    extra_engines: list[ExtraEngineInfo] | None = Field(default=None, max_length=8)  # e.g. EXO crashed after join
    managed: ManagedReport | None = None  # only from an agent joined with --managed, to a host with the feature


class NodeState(StrEnum):
    verifying = "verifying"
    eligible = "eligible"
    ineligible = "ineligible"
    stale = "stale"
    left = "left"
    removed = "removed"


class HeartbeatResponse(Model):
    state: NodeState
    state_reason: str | None
    removed: bool
    deployments: list[str]
    commands: list[PeerCommand] | None = Field(default=None, max_length=16)  # host-managed peers only


class VerificationResult(Model):
    checked_at: datetime
    reachable: bool
    engine_match: bool
    rpc_devices: list[EngineDevice]
    detail: str
    fix: str | None = None


class NodeView(Model):
    id: str
    display_name: str
    is_local: bool
    session_id: str | None
    state: NodeState
    state_reason: str | None
    joined_at: datetime
    last_heartbeat_at: datetime | None
    heartbeat_age_s: float | None
    inventory: HostInventory
    contribution: ContributionLimits
    worker: WorkerEndpoint | None
    worker_state: WorkerState | None
    worker_cpu_seconds: float | None = None
    verification: VerificationResult | None
    deployments: list[str] = Field(default_factory=list)
    # Set only by the authenticated peer transport, never accepted in JoinRequest.
    transport_peer_id: str | None = Field(default=None, max_length=128)


# --------------------------------------------------------------------------- host-managed peers (optional)
# A peer that joined with --managed lets the host restart its RPC worker with devices from the list it consented to,
# and download catalog model shards into its llama.cpp RPC cache. Every field is optional on the existing messages:
# an old agent never sends `managed` and an old host never sends `commands`.

MANAGED_PEERS_FEATURE = "managed_peers.v1"
# ManagedReport.hcpeer_mmap: an agent sends it only to a host that lists this feature (older hosts forbid the field)
HCPEER_REPORT_FEATURE = "hcpeer_report.v1"
MAX_SEED_TENSORS_PER_COMMAND = 256  # keeps a heartbeat reply far below the Iroh 64 KiB control frame
_HEX32 = r"^[a-f0-9]{32}$"
_FNV_HEX = r"^[0-9a-f]{16}$"
_DEVICE_NAME = r"^[A-Za-z][A-Za-z0-9_]{0,15}$"
DeviceName = Annotated[str, Field(pattern=_DEVICE_NAME)]


class SeedTensor(Model):
    """Exactly the bytes [offset, offset + nbytes) of one catalog file: one weight tensor as llama.cpp loads it. Its
    FNV-1a 64 (lowercase %016x) is the name the RPC client asks the peer's cache for (ggml-rpc.cpp fnv_hash)."""

    file: str = Field(min_length=1, max_length=255)
    offset: int = Field(ge=0)
    nbytes: int = Field(gt=0)
    fnv64: str = Field(pattern=_FNV_HEX)


class SeedSpec(Model):
    job_id: str = Field(pattern=_HEX32)
    model_id: str = Field(min_length=1, max_length=128)
    repo: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=64)
    part: int = Field(ge=1)
    parts: int = Field(ge=1)
    tensors: list[SeedTensor] = Field(min_length=1, max_length=MAX_SEED_TENSORS_PER_COMMAND)


class PeerCommand(Model):
    """One host request to a managed peer. Delivered in heartbeat replies until the agent acknowledges its id;
    `epoch` names the host process and `seq` orders commands within it."""

    id: str = Field(pattern=_HEX32)
    epoch: str = Field(pattern=_FNV_HEX)
    seq: int = Field(ge=1)
    version: int = Field(default=1, ge=1)
    kind: Literal["worker", "seed", "seed_cancel"]
    devices: list[DeviceName] | None = Field(default=None, min_length=1, max_length=8)
    threads: int | None = Field(default=None, ge=1, le=256)
    seed: SeedSpec | None = None


class CommandAck(Model):
    id: str = Field(pattern=_HEX32)
    kind: str = Field(max_length=32)
    status: Literal["accepted", "applied", "refused", "failed"]
    detail: str | None = Field(default=None, max_length=300)


class SeedFailure(Model):
    fnv64: str = Field(pattern=_FNV_HEX)
    error: str = Field(max_length=200)


class SeedProgress(Model):
    job_id: str = Field(pattern=_HEX32)
    model_id: str = Field(max_length=128)
    state: Literal["running", "done", "failed", "cancelled"]
    parts_received: int = Field(ge=0)
    parts: int = Field(ge=1)
    tensors_total: int = Field(ge=0)
    tensors_done: int = Field(ge=0)  # written by this job or already in the cache
    tensors_present: int = Field(ge=0)  # of tensors_done: already in the cache, nothing downloaded
    tensors_failed: int = Field(ge=0)
    bytes_total: int = Field(ge=0)
    bytes_done: int = Field(ge=0)
    failures: list[SeedFailure] = Field(default_factory=list, max_length=16)
    error: str | None = Field(default=None, max_length=300)
    updated_at: datetime


class ManagedReport(Model):
    """What a managed peer allows and runs now, sent with every heartbeat to a host that supports it."""

    version: int = Field(default=1, ge=1)
    devices_allowed: list[str] = Field(max_length=8)
    max_disk_bytes: int = Field(ge=0)  # 0: the contributor did not allow cache downloads
    min_free_bytes: int = Field(ge=0)  # seeding never leaves less than this free on the cache volume
    max_threads: int = Field(ge=1)
    cache_enabled: bool
    worker_devices: list[str] = Field(max_length=8)  # the running worker's -d list, in order
    worker_threads: int | None = None  # the running worker's -t (None: rpc-server's default)
    worker_generation: int = Field(ge=0)  # +1 on every host-requested restart
    worker_error: str | None = Field(default=None, max_length=300)
    cache_bytes: int | None = None  # bytes under <LLAMA_CACHE>/rpc
    disk_free_bytes: int | None = None
    acks: list[CommandAck] = Field(default_factory=list, max_length=64)
    seed: SeedProgress | None = None
    # the running worker's hcpeer banner line (native/hcpeer): True "on", False "off" or no line (stock library);
    # None: not reported (sent only to a host with HCPEER_REPORT_FEATURE)
    hcpeer_mmap: bool | None = None


class PeerWorkerRequest(Model):
    devices: list[DeviceName] = Field(min_length=1, max_length=8)
    threads: int | None = Field(default=None, ge=1, le=256)


class PeerSeedRequest(Model):
    model_id: str = Field(min_length=1, max_length=128)
    layers: str | None = Field(default=None, pattern=r"^\d{1,4}(-\d{1,4})?$")  # "A-B", inclusive
    plan_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")  # the node's layers in this plan


# --------------------------------------------------------------------------- sessions


class InviteView(Model):
    code: str  # plaintext is returned exactly once, at creation
    expires_at: datetime
    join_address: str
    join_command: str


class SessionView(Model):
    id: str
    name: str
    status: Literal["active", "closed"]
    created_at: datetime
    lan_addresses: list[str]
    gateway_port: int
    invite: InviteView | None = None


class CreateSessionRequest(Model):
    name: str = Field(default="Hypercompute session", max_length=80)


# --------------------------------------------------------------------------- models


class ModelFile(Model):
    name: str
    url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int
    path: str | None = None  # relative path inside the artifact dir (HF repos keep subdirectories)


class GgufSummary(Model):
    architecture: str | None
    name: str | None
    file_type: str | None
    n_layer: int | None
    n_embd: int | None
    n_head: int | None
    n_head_kv: int | None
    context_length: int | None
    vocab_size: int | None
    tensor_count: int
    total_tensor_bytes: int
    layer_bytes: list[int]  # weight bytes per repeating block, index = layer
    token_embd_bytes: int
    output_bytes: int  # explicit output.weight, or tied embedding duplicated onto the output device
    output_tied: bool
    other_bytes: int
    chat_template_present: bool
    key_length: int | None = None  # <arch>.attention.key_length (per-head K dim), when the GGUF sets it
    value_length: int | None = None  # <arch>.attention.value_length (per-head V dim)
    expert_count: int | None = None  # <arch>.expert_count of a mixture-of-experts model
    expert_used_count: int | None = None  # experts each token is routed to
    expert_shared_count: int | None = None  # always-active shared experts (dense bytes)
    layer_expert_bytes: list[int] = Field(default_factory=list)  # routed-expert bytes inside layer_bytes; [] if dense


class ModelArtifact(Model):
    id: str
    display_name: str
    family: str
    params_b: float
    quantization: str
    format: Literal["gguf", "hf", "mlx"] = "gguf"
    engines: list[EngineId] = Field(default_factory=list)  # empty = default for the format (see artifact_engines)
    logical_id: str | None = None  # groups the GGUF / HF / MLX variants of one model
    size_bytes: int | None = None  # for artifacts an engine downloads itself (EXO/MLX), where files is empty
    repo: str
    revision: str
    license: str
    files: list[ModelFile]
    # vision projector (llama.cpp --mmproj): runs on the coordinator next to the first layers, never sent to peers
    mmproj: ModelFile | None = None
    default_ctx: int = 4096
    max_output_tokens: int = 1024
    role: Literal["plumbing", "capacity_candidate", "standard"] = "standard"
    notes: str | None = None

    @model_validator(mode="after")
    def _projector_needs_gguf(self) -> ModelArtifact:
        if self.mmproj is not None and self.format != "gguf":
            raise ValueError("an mmproj vision projector is a llama.cpp file and needs a gguf artifact")
        return self


class ModelView(Model):
    artifact: ModelArtifact
    # engine_managed: the engine fetches the weights itself on every node (EXO/MLX), nothing to download here
    status: Literal["not_downloaded", "downloading", "downloaded", "corrupt", "error", "engine_managed"]
    local_path: str | None
    downloaded_bytes: int
    total_bytes: int
    verified_sha256: bool
    gguf: GgufSummary | None
    error: str | None = None
    projector_status: Literal["none", "not_downloaded", "downloading", "downloaded"] = "none"  # artifact.mmproj


# --------------------------------------------------------------------------- planning


class PlanStrategy(StrEnum):
    auto = "auto"  # fastest tier that fits: GPUs, then GPUs + host RAM, then capacity mode
    all_selected = "all_selected"  # every selected peer gets layers (participation)
    fastest_fit = "fastest_fit"  # fewest devices that fit, local first
    local_only = "local_only"


class LayoutStage(Model):
    """One stage of an explicit layout: the next `layers` whole layers on one device. Stages run in list order, peers
    first and this machine's GPU last, which also holds the output head."""

    node_id: str = Field(min_length=1, max_length=64)  # a peer's node id, or "local" for this machine
    device: str = Field(pattern=_DEVICE_NAME)  # as that node's worker names it: MTL0, CPU, CUDA0
    layers: int = Field(ge=1, le=4096)
    # file_backed: a peer CPU device maps these weights from its SSD (hcpeer). None: file_backed for a peer CPU stage
    # whose worker reports hcpeer on, else resident
    residency: Literal["resident", "file_backed"] | None = None


class PlanRequest(Model):
    model_id: str
    node_ids: list[str] | None = None  # None = all eligible remote nodes
    strategy: PlanStrategy = PlanStrategy.all_selected
    ctx_size: int = Field(default=4096, ge=256, le=262144)
    allow_ram_tier: bool = True  # may keep the first layers in host RAM when GPUs alone do not fit
    allow_capacity_mode: bool = True  # auto may fall back to disk-backed tiers when nothing faster fits
    engine: EngineId | None = None  # force an engine; None lets the selector choose
    objective: Literal["interactive", "throughput"] = "interactive"
    use_profiles: bool = True  # prefer a measured profile for this model + device set when one exists
    # where layers the local GPU cannot hold may go; auto picks the cheaper by the decode-cost prior (the tuner
    # forces each alternative to measure it)
    overflow: Literal["auto", "peers", "host_ram"] = "auto"
    # an explicit layout (services/layout.py): every stage named, only checked by the planner; None lets it choose
    layout: list[LayoutStage] | None = Field(default=None, min_length=1, max_length=64)
    # llama.cpp server slots (-np, unified KV): concurrent requests the engine batches; stored in engine_options
    parallel: int = Field(default=1, ge=1, le=16)


class DeviceBudget(Model):
    node_id: str
    device: str  # coordinator-visible engine device name (CUDA0, RPC1, ...)
    description: str
    endpoint: str | None
    pool_id: str
    pool_kind: Literal["system_ram", "unified", "discrete_vram"]
    budget_bytes: int
    budget_basis: str  # human explanation of which limit bound it


class DeviceAssignment(Model):
    node_id: str
    device: str
    endpoint: str | None
    pool_id: str
    layers: list[int]  # repeating block indices
    holds_output: bool
    weight_bytes: int
    kv_bytes: int
    compute_bytes_est: int
    overhead_bytes_est: int
    total_bytes_est: int
    budget_bytes: int
    fits: bool
    # a peer CPU stage whose weights hcpeer maps from that peer's SSD: weight_bytes are on its disk, and
    # total_bytes_est holds only what stays in its RAM (dense parts, KV, buffers)
    file_backed: bool = False


class StreamSpec(Model):
    """The CPU units of a stream-tier plan. Unit n_layer is the output head (CPU-only plans)."""

    resident_units: list[int]
    streamed_units: list[int]
    slots: int  # slot buffers of slot_bytes each; 1 means serial read then compute
    slot_bytes: int
    resident_bytes: int
    stream_bytes_per_pass: int  # SSD bytes read for every forward pass (each decoded token)
    read_bytes_per_s: float | None = None  # storage rate the prediction used
    read_provenance: Literal["measured", "prior"] = "prior"
    predicted_io_ms_per_token: float | None = None

    @property
    def committed_bytes(self) -> int:
        return self.slots * self.slot_bytes + self.resident_bytes


class MoeSpec(Model):
    """Tier moe_offload. Every *_est number is an estimate from priors, never a measurement."""

    expert_count: int
    expert_used_count: int
    active_fraction: float  # expert_used_count / expert_count: the share of a layer's expert bytes one token reads
    cpu_expert_layers: list[int]  # host-side layers whose routed-expert tensors stay on the host CPU
    cpu_expert_bytes: int  # moe_cpu_expert_bytes: file-backed (mmap, no repack), backed by the GGUF on disk, not RAM
    gpu_expert_bytes: int  # expert bytes of the other host layers, on the local GPU
    override: str  # llama.cpp -ot value: exactly the expert tensors of cpu_expert_layers, buffer type CPU
    page_cache_bytes_est: int  # host RAM left for the file cache that holds CPU experts
    miss_rate_est: float  # max(0, 1 - page_cache_bytes_est / cpu_expert_bytes)
    ssd_bytes_per_token_est: int  # active_fraction * cpu_expert_bytes * miss_rate_est
    read_bytes_per_s: float  # page-fault read rate the I/O estimate assumes (a prior)
    decode_ms_est: float  # compute priors on the bytes a token reads + SSD reads + peer round trips
    provenance: Literal["estimate"] = "estimate"


class PlanLimits(Model):
    """The limits of the PlanRequest a plan was made for. The replan evaluator keeps them: it never proposes a tier the
    request turned off or a peer it did not select. Plans made before this field existed carry None."""

    allow_ram_tier: bool = True
    allow_capacity_mode: bool = True
    node_ids: list[str] | None = None  # None = every eligible remote node
    strategy: PlanStrategy = PlanStrategy.all_selected

    @classmethod
    def of(cls, req: PlanRequest) -> PlanLimits:
        return cls(allow_ram_tier=req.allow_ram_tier, allow_capacity_mode=req.allow_capacity_mode,
                   node_ids=list(req.node_ids) if req.node_ids is not None else None, strategy=req.strategy)


class CandidateSummary(Model):
    label: str
    devices: list[str]
    feasible: bool
    reason: str


class StagePrediction(Model):
    """One stage of a scored placement, in execution order. Times are per decoded token unless named otherwise."""

    node_id: str
    device: str
    units: list[int]
    resident_units: list[int] = Field(default_factory=list)
    streamed_units: list[int] = Field(default_factory=list)
    compute_ms: float = 0.0
    exposed_io_ms: float = 0.0
    hop_ms: float = 0.0
    upload_bytes: int = 0  # weights that must cross a link before a cold start
    path: str = "local"


class PlanPrediction(Model):
    """The capacity optimizer's score for a plan. Predictions, not measurements; `assumptions` names each prior used."""

    decode_ms: float  # per token: sum of stage compute + exposed I/O + hops (never the max stage)
    prefill_ms: float
    load_ms: float  # missing-weight uploads + resident reads + setup
    objective_ms: float  # J for the workload the plan was scored for
    bottleneck: str
    stream_bytes_per_token: int = 0
    stages: list[StagePrediction] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # devices left out, with the reason
    assumptions: list[str] = Field(default_factory=list)
    explain: list[str] = Field(default_factory=list)  # plain lines for the CLI and dashboard
    manifest_fingerprint: str | None = None


class Plan(Model):
    id: str
    model_id: str
    strategy: PlanStrategy
    ctx_size: int
    feasible: bool
    fit_label: Literal["estimated_fit", "does_not_fit", "fit_unknown"]
    heuristic: bool  # True until a matching measured configuration exists
    assignments: list[DeviceAssignment]
    device_list: list[str]  # passed to --device, in order
    tensor_split: list[int]  # layer units per device, passed to --tensor-split
    rpc_endpoints: list[str]
    cpu_resident_bytes: int  # token embeddings stay in host RAM on the coordinator
    reasons: list[str]
    warnings: list[str]
    candidates: list[CandidateSummary]
    created_at: datetime
    engine: EngineId = "llama_rpc"
    tier: Tier = "gpu"
    cpu_layers: int = 0  # layers 0..cpu_layers-1 run on the host CPU (llama.cpp keeps the input side there)
    n_gpu_units: int | None = None  # -ngl value when cpu_layers > 0 (offloaded layers + output head)
    engine_flags: dict[str, str | int] = Field(default_factory=dict)  # llama.cpp: flash_attn, ubatch, batch, threads
    engine_options: dict = Field(default_factory=dict)  # engine-specific plan data (EXO preview, vLLM args, AirLLM)
    profile_id: str | None = None  # measured profile this plan reuses (then heuristic is False)
    capacity_mode: bool = False  # disk-backed tier: expect seconds or more per token
    stream: StreamSpec | None = None  # tier "stream": which CPU units are resident and which stream from the SSD
    moe: MoeSpec | None = None  # tier "moe_offload": which host layers keep their experts on the CPU, and estimates
    revision: int = 0  # increases with every replacement of the same served model (replan controller)
    prediction: PlanPrediction | None = None  # capacity optimizer score, when the optimizer planned it
    limits: PlanLimits | None = None  # the request's limits; None on plans stored before the field existed
    layout: list[LayoutStage] | None = None  # the explicit layout this plan was made from (PlanRequest.layout)


# --------------------------------------------------------------------------- deployments


class DeploymentState(StrEnum):
    planned = "planned"
    reserving = "reserving"
    preparing = "preparing"
    loading = "loading"
    warming = "warming"
    ready = "ready"
    draining = "draining"
    stopped = "stopped"
    failed = "failed"


TERMINAL_STATES = {DeploymentState.stopped, DeploymentState.failed}


class StateTransition(Model):
    state: DeploymentState
    at: datetime
    detail: str | None = None


class ObservedDevice(Model):
    name: str  # as llama.cpp printed it, e.g. RPC0[10.0.0.5:47052]
    model_bytes: int
    kv_bytes: int
    compute_bytes: int
    layers_est: int | None  # derived from KV share; None when KV is absent


class ObservedPlacement(Model):
    offloaded: str | None  # "25/25"
    devices: list[ObservedDevice]
    evidence_lines: list[str]
    parsed_at: datetime


class Deployment(Model):
    id: str
    plan: Plan
    state: DeploymentState
    history: list[StateTransition]
    error: str | None = None
    cleanup: str | None = None
    progress: str | None = None
    engine_args: list[str] = Field(default_factory=list)
    upstream_port: int | None = None
    pid: int | None = None
    placement: ObservedPlacement | None = None
    participation_verified: bool = False
    participation_detail: str | None = None
    created_at: datetime
    idempotency_key: str | None = None
    purpose: Literal["run", "tune"] = "run"  # "tune": a tuner trial, never offered to chat or benchmarks
    revision: int = 0  # equals plan.revision; leases carry it so a stale release is rejected
    replaces: str | None = None  # the deployment this one replaced (replan), if any


class ReplanDecision(Model):
    """One evaluation by the replan controller, taken or not, with the numbers behind it."""

    at: datetime
    trigger: str  # "node.verified", "node.left", "pressure", "drift", "manual", ...
    deployment_id: str | None
    revision: int
    action: Literal["keep", "replace", "fallback", "defer", "failed"]
    reason: str
    current_decode_ms: float | None = None
    proposed_decode_ms: float | None = None
    relative_gain: float | None = None
    reload_ms: float | None = None
    replay_ms: float | None = None
    break_even_tokens: float | None = None
    proposed_plan_id: str | None = None
    outcome: str | None = None  # filled when a replacement finishes: "activated", "rolled_back", "fell_back", ...


class CreateDeploymentRequest(Model):
    plan_id: str | None = None
    plan: PlanRequest | None = None
    purpose: Literal["run", "tune"] = "run"


# --------------------------------------------------------------------------- runs / benchmarks


class EngineTimings(Model):
    prompt_n: int | None = None
    prompt_ms: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None
    predicted_per_second: float | None = None
    prompt_per_second: float | None = None
    cache_n: int | None = None


class RunRecord(Model):
    id: str
    attempt_id: str
    deployment_id: str | None
    kind: Literal["chat", "benchmark", "warmup"]
    status: Literal["ok", "error", "cancelled", "timeout"]
    simulated: bool = False
    model: dict
    engine: dict
    placement: list[dict]
    transport: str = "tcp"
    ctx_size: int | None
    stream: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_ms: float | None  # request -> first real model output, measured at the gateway
    total_ms: float | None
    timings: EngineTimings | None
    finish_reason: str | None
    sampling: dict
    cache_state: Literal["warm", "cold", "unknown"] = "unknown"
    output_text: str | None
    error: str | None = None
    started_at: datetime
    benchmark_id: str | None = None
    prompt_label: str | None = None
    resumed_from: str | None = None  # the interrupted run this one continues; its output_text was already delivered
    messages_sha256: str | None = None  # chat runs: hash of the client's messages, so a resume can prove they match
    output_truncated: bool = False  # output_text kept only its first RECORD_TEXT_MAX_CHARS characters
    non_content_output: bool = False  # the reply held reasoning text or tool calls that output_text does not keep


class BenchmarkRequest(Model):
    deployment_id: str | None = None
    prompt_set: Literal["short", "long", "both"] = "both"
    repetitions: int = Field(default=3, ge=1, le=20)
    warmup: int = Field(default=1, ge=0, le=5)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    label: str | None = Field(default=None, max_length=80)


class StatSummary(Model):
    n: int
    median: float | None
    min: float | None
    max: float | None


class BenchmarkSummary(Model):
    prompt_label: str
    ttft_ms: StatSummary
    decode_tokens_per_s: StatSummary
    prompt_tokens_per_s: StatSummary
    completion_tokens: StatSummary
    failures: int


class BenchmarkResult(Model):
    id: str
    deployment_id: str
    label: str | None
    status: Literal["running", "done", "failed"]
    request: BenchmarkRequest
    run_ids: list[str]
    summaries: list[BenchmarkSummary]
    placement_label: str
    created_at: datetime
    error: str | None = None


# --------------------------------------------------------------------------- events


class Event(Model):
    seq: int
    at: datetime
    type: str
    message: str
    data: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- tuning / measured profiles


class TuneRequest(Model):
    model_id: str
    node_ids: list[str] | None = None
    ctx_size: int = Field(default=4096, ge=256, le=262144)
    objective: Literal["interactive", "throughput"] = "interactive"
    max_candidates: int = Field(default=4, ge=1, le=8)
    tune_flags: bool = True  # also try flash attention and micro-batch size, not only the layer split
    decode_tokens: int = Field(default=64, ge=8, le=512)
    # the owner's own plan limits: tuning measures only configurations their plans would accept
    engine: EngineId | None = None  # tuning measures llama.cpp splits; None means llama_rpc
    strategy: PlanStrategy = PlanStrategy.auto
    allow_ram_tier: bool = True
    allow_capacity_mode: bool = False


class TuneCandidate(Model):
    label: str
    plan: Plan
    status: Literal["pending", "running", "done", "failed", "skipped"]
    deployment_id: str | None = None
    load_s: float | None = None
    ttft_ms: float | None = None  # short prompt, prompt cache off
    decode_tokens_per_s: float | None = None  # llama.cpp timings
    prefill_tokens_per_s: float | None = None  # long prompt, prompt cache off
    score: float | None = None  # lower is better: estimated seconds for a typical turn
    run_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class TuneResult(Model):
    id: str
    request: TuneRequest
    status: Literal["running", "done", "failed", "cancelled"]
    candidates: list[TuneCandidate]
    best_index: int | None = None
    profile_id: str | None = None
    created_at: datetime
    error: str | None = None
    note: str | None = None  # e.g. why the winner was not saved as a measured profile


class MeasuredProfile(Model):
    """A configuration that was actually run and measured, reused by the planner for the same setup."""

    id: str
    model_id: str
    signature: str  # model + ctx + sorted (node, device) set the plan used; see services.profiles.signature
    objective: Literal["interactive", "throughput"]
    plan_fields: dict  # device_list, tensor_split, cpu_layers, n_gpu_units, engine_flags, rpc_endpoints, tier
    ttft_ms: float | None
    decode_tokens_per_s: float | None
    prefill_tokens_per_s: float | None
    score: float
    tune_id: str | None
    run_ids: list[str]
    created_at: datetime


JoinResponse.model_rebuild()
