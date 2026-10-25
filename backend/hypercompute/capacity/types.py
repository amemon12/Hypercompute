"""Shared value types of the capacity layer: model units, memory pools, measured coefficients and placements.

Every number carries where it came from. `Coefficient.provenance` distinguishes a measurement on this hardware from a
prior or an estimate, and a placement keeps its per-stage breakdown so a report can say which term dominated. These
types are immutable; modules return new values instead of editing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

PoolKind = Literal["system_ram", "unified", "discrete_vram", "storage"]
Provenance = Literal["measured", "prior", "estimate", "reported"]
PathClass = Literal["local", "loopback", "lan", "direct", "relay", "unknown"]
UnitKind = Literal["block", "head"]
GIB = 1 << 30


# --------------------------------------------------------------------------------------------------- model


@dataclass(frozen=True)
class UnitSpec:
    """One placement unit: a transformer block, or the output head (index n_layer, llama.cpp's extra GPU unit)."""

    index: int
    kind: UnitKind
    weight_bytes: int  # exact sum of ggml_nbytes over the unit's tensors
    buffer_bytes: int  # 4 KiB-aligned bytes a streaming slot or private resident copy needs (hcstream layout)
    types: tuple[tuple[str, int], ...] = ()  # (ggml type name, bytes): the unit's quantization layout
    expert_bytes: int = 0  # routed-expert tensors (gguf.EXPERT_TENSOR_RE) inside weight_bytes; 0 for a dense block

    @property
    def dense_bytes(self) -> int:
        """Attention, norms, router and shared experts: the bytes every token reads."""
        return self.weight_bytes - self.expert_bytes


@dataclass(frozen=True)
class ShardInfo:
    path: str
    size: int
    sha256: str | None  # None when not verified (hashing 40 GB is not free); never invented


@dataclass(frozen=True)
class ModelManifest:
    """Exact per-unit metadata of one GGUF artifact. A different quantization is a different manifest."""

    model_id: str
    revision: str | None
    shards: tuple[ShardInfo, ...]
    architecture: str
    file_type: str | None
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    key_length: int
    value_length: int
    context_length: int | None
    rope: tuple[tuple[str, str], ...]
    tokenizer_model: str | None
    vocab_size: int | None
    chat_template_sha256: str | None
    units: tuple[UnitSpec, ...]  # n_layer blocks in order, then the head
    token_embd_bytes: int
    other_bytes: int
    tied_embeddings: bool
    streamable: bool  # every block/head tensor has a ggml type whose size is known
    fingerprint: str  # sha256 over shard identities and the tensor table
    expert_count: int = 0  # <arch>.expert_count; 0 for a dense model
    expert_used_count: int = 0  # experts each token is routed to
    expert_shared_count: int = 0  # always-active shared experts, counted in dense bytes

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 0 and any(u.expert_bytes for u in self.units)

    @property
    def active_expert_fraction(self) -> float:
        """Share of a block's routed-expert bytes one token reads (expert_used_count / expert_count); 1.0 if dense."""
        return self.expert_used_count / self.expert_count if self.is_moe else 1.0

    @property
    def head(self) -> UnitSpec:
        return self.units[self.n_layer]

    def block(self, index: int) -> UnitSpec:
        return self.units[index]

    def kv_bytes_per_layer(self, ctx_size: int, bytes_per_elem: int = 2) -> int:
        """Standard K+V cache for one layer at batch one: ctx * n_head_kv * (k_len + v_len) * elem size."""
        return ctx_size * self.n_head_kv * (self.key_length + self.value_length) * bytes_per_elem

    @property
    def weight_bytes(self) -> int:
        return sum(u.weight_bytes for u in self.units) + self.token_embd_bytes + self.other_bytes


# --------------------------------------------------------------------------------------------------- pools


@dataclass(frozen=True)
class Pool:
    """A physical allocation pool. Mac CPU and Metal share one `unified` pool and must never be counted twice."""

    node_id: str
    pool_id: str  # "ram", "uma", "vram:CUDA0", "ssd"
    kind: PoolKind
    total_bytes: int | None
    free_bytes: int  # fresh observation: available RAM, free VRAM, or free volume space
    observed_at: datetime
    source: str
    reserve_bytes: int = 0  # what admission must leave free (user or contributor reserve)
    cap_bytes: int | None = None  # contribution cap for this pool, None for no cap
    devices: tuple[str, ...] = ()  # engine devices that allocate from this pool


@dataclass(frozen=True)
class PoolLease:
    """An admitted allocation. Once the engine has loaded, its bytes are already inside the observed free memory."""

    lease_id: str
    deployment_id: str
    revision: int
    node_id: str
    pool_id: str
    bytes: int
    materialized: bool


@dataclass(frozen=True)
class PoolDemand:
    """What one placement needs from one pool, split by purpose so reports and admission agree."""

    node_id: str
    pool_id: str
    weights: int = 0  # resident weights (private copies or device buffers)
    kv: int = 0
    workspace: int = 0  # compute buffers
    staging: int = 0  # streaming slots, RPC staging, upload buffers
    overhead: int = 0  # runtime and allocator margin
    peak_extra: int = 0  # transient conversion/dequantization/repacking peak above steady state

    @property
    def steady(self) -> int:
        return self.weights + self.kv + self.workspace + self.staging + self.overhead

    @property
    def peak(self) -> int:
        return self.steady + self.peak_extra


@dataclass(frozen=True)
class PoolCheck:
    node_id: str
    pool_id: str
    demand_peak: int
    admissible: int
    ok: bool
    reason: str


@dataclass(frozen=True)
class Admission:
    ok: bool
    checks: tuple[PoolCheck, ...]

    @property
    def failures(self) -> tuple[PoolCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)


# --------------------------------------------------------------------------------------------------- coefficients


@dataclass(frozen=True)
class Coefficient:
    value: float
    unit: str
    provenance: Provenance
    measured_at: datetime | None = None
    samples: int = 0
    note: str = ""


@dataclass(frozen=True)
class LinkProfile:
    path: PathClass
    rtt_ms: Coefficient
    goodput_bytes_per_s: Coefficient  # sustained bulk transfer (weight uploads), not idle latency
    round_trips_per_token: Coefficient  # RPC exchanges per decoded token for one remote stage
    cached_bytes: int = 0  # model bytes already in the peer's upload cache


@dataclass(frozen=True)
class DeviceProfile:
    node_id: str
    device: str  # engine device name: "CPU", "CUDA0", "MTL0"
    backend: str  # "cpu", "cuda", "metal"
    pool_id: str
    decode_ms_per_gib: Coefficient  # batch-one decode time per GiB of weights computed on this device
    prefill_ms_per_gib_token: Coefficient  # prompt processing time per GiB of weights per prompt token
    link: LinkProfile | None = None  # None for devices inside the host process


@dataclass(frozen=True)
class StorageProfile:
    node_id: str
    read_bytes_per_s: Coefficient  # unbuffered sustained reads of a model file
    stream_efficiency: Coefficient  # fraction of read_bytes_per_s hcstream achieves end to end


# --------------------------------------------------------------------------------------------------- placement


@dataclass(frozen=True)
class Workload:
    ctx_size: int
    prompt_tokens: int
    output_tokens: int
    warm: bool = False  # already loaded: do not charge uploads that were paid
    horizon_tokens: int = 2048  # output tokens a replacement has to pay itself back over
    objective: Literal["interactive", "throughput"] = "interactive"


@dataclass(frozen=True)
class StagePlan:
    """One contiguous span of units on one device, in execution order."""

    node_id: str
    device: str
    pool_id: str
    units: tuple[int, ...]
    resident_units: tuple[int, ...]
    streamed_units: tuple[int, ...] = ()
    slots: int = 0
    demand: PoolDemand | None = None
    compute_ms: float = 0.0
    exposed_io_ms: float = 0.0
    hop_ms: float = 0.0  # activation exchange cost attributed to this stage per token
    upload_bytes: int = 0  # weights that must cross a link before the stage can run (cold start)
    path: PathClass = "local"


@dataclass(frozen=True)
class PlacementCandidate:
    stages: tuple[StagePlan, ...]
    feasible: bool
    reasons: tuple[str, ...]
    decode_ms: float  # T_decode = sum(stage compute + exposed I/O + hops); never max(stage)
    prefill_ms: float
    load_ms: float  # missing-weight upload + resident reads + setup
    objective_ms: float  # J for the workload
    bottleneck: str
    stream_bytes_per_token: int
    skipped: tuple[str, ...] = ()  # devices left out, with the reason
    assumptions: tuple[str, ...] = field(default=())  # every prior or estimate the score used
