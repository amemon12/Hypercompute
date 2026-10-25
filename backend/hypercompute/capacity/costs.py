"""Cost priors and per-stage time formulas for the placement optimizer.

Every prior here is a `Coefficient(provenance="prior")` with a note saying where the number comes from. A measured
profile replaces a prior through `merge`, never the other way round. The formulas are the ones in
docs/research/hybrid-capacity-implementation.md: compute is linear in the weight bytes a stage reads per token,
CPU streaming overlaps with compute only when there are at least two slots, and a remote stage pays its round trips
plus the activation bytes over the link's goodput. The CPU residency rule (`split_cpu`) lives here too, because it
decides how many bytes a CPU stage streams per token.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TypeVar

from .types import GIB, Coefficient, DeviceProfile, LinkProfile, PathClass, Provenance, StorageProfile, UnitSpec

GB = 1_000_000_000
MB = 1_000_000
MIB = 1 << 20
GIB_PER_GB = GIB / GB  # DECODE_MS_PER_GB is per 1e9 bytes; coefficients here are per GiB

# The planner's priors (services/planner.py), repeated here so the capacity layer does not import the service layer;
# tests/test_capacity_optimizer.py checks that the two copies agree.
DECODE_MS_PER_GB = {"cuda": 2.64, "metal": 10.85, "cpu": 12.5}  # ms per 1e9 weight bytes read per decoded token
UBATCH = 512
COMPUTE_BASE_BYTES = 96 * MIB
OVERHEAD_BYTES = {"cuda": 450 * MIB, "metal": 256 * MIB, "vulkan": 300 * MIB, "cpu": 64 * MIB}
RPC_STAGING_BYTES = 64 * MIB
ACTIVATION_BYTES_PER_ELEM = 4  # f32 hidden state crossing a stage boundary
LOGIT_BYTES_PER_ELEM = 4  # f32 logits, copied from the device holding the head (docs/research/llama_cpu_tier.md A2)

# Prefill time per GiB per prompt token, as a fraction of the decode prior of the same backend.
# cuda: 9.6 us per Qwen2.5-7B layer-token against 2.64 ms/GB decode, with a 0.139 GB layer (split_optimization.md
# section 1, derived from docs/RESULTS.md). metal: 163 us per layer-token against 10.85 ms/GB (same section, derived
# from llama.cpp discussion #4167). cpu: no prefill measurement exists on this hardware; 0.3 is an assumption.
PREFILL_FRACTION = {"cuda": 9.6e-3 / 0.139 / 2.64, "metal": 163e-3 / 0.139 / 10.85, "cpu": 0.3}
_PREFILL_NOTE = {
    "cuda": "derived: 9.6 us per Qwen2.5-7B layer-token (split_optimization.md section 1)",
    "metal": "derived: 163 us per Qwen2.5-7B layer-token on an M4 (split_optimization.md section 1)",
    "cpu": "assumed: 0.3 of the CPU decode prior; no CPU prefill measurement on this hardware",
}
_DECODE_NOTE = {
    "cuda": "derived: RTX 5070 Ti Laptop two-point fit, 2.64 ms/GB (split_optimization.md section 1)",
    "metal": "derived: M4 llama.cpp discussion #4167 row, 10.85 ms/GB (split_optimization.md section 1)",
    "cpu": "assumed: 80 GB/s effective, 12.5 ms/GB (split_optimization.md section 3.4), unverified",
}
ROUND_TRIPS_PER_TOKEN = 2.0  # RPC exchanges per remote stage per decoded token (implementation plan, costs.py)


@dataclass(frozen=True)
class _PathPrior:
    rtt_ms: float
    goodput_bytes_per_s: float
    note: str


_PATH_PRIORS: dict[str, _PathPrior] = {
    "loopback": _PathPrior(0.1, 3 * GB, "assumed: loopback socket, 0.1 ms and 3 GB/s"),
    "lan": _PathPrior(2.0, 60 * MB, "assumed: Wi-Fi LAN through one access point, 2 ms and 60 MB/s"),
    "direct": _PathPrior(8.0, 30 * MB, "assumed: direct internet path, 8 ms and 30 MB/s"),
    "relay": _PathPrior(20.0, 1 * MIB, "assumed rtt 20 ms; 1 MiB/s matches the relay bulk rate seen in the 32B "
                        "handover (hybrid-capacity-architecture.md)"),
    "unknown": _PathPrior(20.0, 1 * MIB, "unknown path class: priced as a relay path"),
}

STORAGE_READ_BYTES_PER_S = 3 * GIB  # below the 3.45 GiB/s QD1 storage probe on this host
STREAM_EFFICIENCY = 0.7
SETUP_MS = 2000.0  # engine start, graph build and warmup on a cold load
SETUP_MS_PER_REMOTE_STAGE = 500.0  # RPC connect and remote buffer allocation per peer stage
RELAY_RISK_FRACTION = 0.10  # margin for relay paths: share of the stage's per-token cost, times output tokens


def prior(value: float, unit: str, note: str) -> Coefficient:
    return Coefficient(value=value, unit=unit, provenance="prior", note=note)


def setup_prior() -> Coefficient:
    return prior(SETUP_MS, "ms", "assumed: engine start, graph build and warmup on a cold load")


def setup_per_remote_prior() -> Coefficient:
    return prior(SETUP_MS_PER_REMOTE_STAGE, "ms", "assumed: RPC connect and remote allocation per peer stage")


def relay_risk_prior() -> Coefficient:
    return prior(RELAY_RISK_FRACTION, "fraction", "assumed margin: 10 percent of a relay stage's per-token cost, "
                 "for stalls and reconnects on relayed paths")


def link_prior(path: PathClass, *, cached_bytes: int = 0) -> LinkProfile:
    """Link profile for a path class from documented priors. In-process devices have no link."""
    if path not in _PATH_PRIORS:
        raise ValueError(f"no link prior for path class {path!r}; in-process devices have link=None")
    if cached_bytes < 0:
        raise ValueError("cached_bytes must be non-negative")
    p = _PATH_PRIORS[path]
    return LinkProfile(path=path, rtt_ms=prior(p.rtt_ms, "ms", p.note),
                       goodput_bytes_per_s=prior(p.goodput_bytes_per_s, "bytes/s", p.note),
                       round_trips_per_token=prior(ROUND_TRIPS_PER_TOKEN, "round trips",
                                                   "assumed: 2 RPC exchanges per remote stage per token"),
                       cached_bytes=cached_bytes)


def device_prior(backend: str, node_id: str, device: str, pool_id: str,
                 link: LinkProfile | None = None) -> DeviceProfile:
    """Device profile with decode and prefill priors for a backend family (cpu, cuda or metal)."""
    if backend not in PREFILL_FRACTION:
        raise ValueError(f"no cost prior for backend {backend!r}")
    decode = DECODE_MS_PER_GB[backend] * GIB_PER_GB
    return DeviceProfile(
        node_id=node_id, device=device, backend=backend, pool_id=pool_id,
        decode_ms_per_gib=prior(decode, "ms/GiB", _DECODE_NOTE[backend] + ", converted to ms/GiB"),
        prefill_ms_per_gib_token=prior(decode * PREFILL_FRACTION[backend], "ms/GiB/token", _PREFILL_NOTE[backend]),
        link=link)


def cpu_device(node_id: str, *, device: str = "CPU", pool_id: str = "ram",
               link: LinkProfile | None = None) -> DeviceProfile:
    return device_prior("cpu", node_id, device, pool_id, link)


def cuda_device(node_id: str, *, device: str = "CUDA0", pool_id: str | None = None,
                link: LinkProfile | None = None) -> DeviceProfile:
    return device_prior("cuda", node_id, device, pool_id or f"vram:{device}", link)


def metal_device(node_id: str, *, device: str = "MTL0", pool_id: str = "uma",
                 link: LinkProfile | None = None) -> DeviceProfile:
    return device_prior("metal", node_id, device, pool_id, link)


def storage_prior(node_id: str) -> StorageProfile:
    return StorageProfile(
        node_id=node_id,
        read_bytes_per_s=prior(float(STORAGE_READ_BYTES_PER_S), "bytes/s",
                               "assumed: 3.0 GiB/s, below the 3.45 GiB/s unbuffered QD1 probe on this host"),
        stream_efficiency=prior(STREAM_EFFICIENCY, "fraction", "assumed: share of the read rate hcstream "
                                "achieves end to end"))


# --------------------------------------------------------------------------------------------------- memory


def workspace_bytes(n_embd: int, ubatch: int = UBATCH) -> int:
    """Compute-buffer prior for one stage: the planner's base plus ubatch * n_embd * 4 bytes * 8 live tensors."""
    if n_embd <= 0 or ubatch <= 0:
        raise ValueError("n_embd and ubatch must be positive")
    return COMPUTE_BASE_BYTES + ubatch * n_embd * 4 * 8


def overhead_bytes(backend: str) -> int:
    """Runtime and allocator margin prior for one stage on a backend family."""
    if backend not in OVERHEAD_BYTES:
        raise ValueError(f"no memory overhead prior for backend {backend!r}")
    return OVERHEAD_BYTES[backend]


def activation_bytes(n_embd: int) -> int:
    """Bytes of one token's hidden state at a stage boundary, one way."""
    return n_embd * ACTIVATION_BYTES_PER_ELEM


def logits_bytes(vocab_size: int) -> int:
    """Bytes llama.cpp copies back from the device holding the head per decoded token (n_outputs = 1)."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return vocab_size * LOGIT_BYTES_PER_ELEM


def _check(c: Coefficient, name: str, *, positive: bool) -> None:
    bad = not math.isfinite(c.value) or c.value < 0 or (positive and c.value == 0)
    if bad:
        kind = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {kind} number, got {c.value!r} ({c.provenance})")


def check_device(d: DeviceProfile) -> None:
    """Raise ValueError when a coefficient the cost formulas use is NaN, infinite, negative or a zero rate."""
    key = f"{d.node_id}:{d.device}"
    _check(d.decode_ms_per_gib, f"{key} decode_ms_per_gib", positive=False)
    _check(d.prefill_ms_per_gib_token, f"{key} prefill_ms_per_gib_token", positive=False)
    if d.link is not None:
        _check(d.link.rtt_ms, f"{key} rtt_ms", positive=False)
        _check(d.link.goodput_bytes_per_s, f"{key} goodput_bytes_per_s", positive=True)
        _check(d.link.round_trips_per_token, f"{key} round_trips_per_token", positive=False)
        if d.link.cached_bytes < 0:
            raise ValueError(f"{key} cached_bytes must be non-negative")


def check_storage(s: StorageProfile) -> None:
    _check(s.read_bytes_per_s, f"{s.node_id} storage read_bytes_per_s", positive=True)
    _check(s.stream_efficiency, f"{s.node_id} stream_efficiency", positive=True)


_PROVENANCE_RANK: dict[str, int] = {"measured": 0, "reported": 1, "estimate": 2, "prior": 3}


def weakest(*coefficients: Coefficient) -> Provenance:
    """The least trustworthy provenance among the coefficients a figure was computed from."""
    if not coefficients:
        raise ValueError("weakest needs at least one coefficient")
    return max((c.provenance for c in coefficients), key=lambda p: _PROVENANCE_RANK[p])


# --------------------------------------------------------------------------------------------------- merge

T = TypeVar("T", Coefficient, DeviceProfile, LinkProfile, StorageProfile)


def _usable(measured: Coefficient | None, unit: str, now: datetime | None, max_age_s: float | None) -> bool:
    if measured is None or measured.provenance != "measured":
        return False
    if measured.unit != unit:
        raise ValueError(f"cannot merge a measurement in {measured.unit!r} into a coefficient in {unit!r}")
    if not math.isfinite(measured.value) or measured.value < 0:
        raise ValueError(f"measured coefficient in {unit!r} is not a finite non-negative number: {measured.value!r}")
    if max_age_s is None:
        return True
    if measured.measured_at is None or now is None:
        return False  # freshness was asked for and cannot be shown
    age = (now - measured.measured_at).total_seconds()
    return 0 <= age <= max_age_s  # a measurement dated after `now` is not fresh (as in measurements.latest)


def _coef(p: Coefficient, m: Coefficient | None, now: datetime | None, max_age_s: float | None) -> Coefficient:
    return m if _usable(m, p.unit, now, max_age_s) and m is not None else p


def _merge_link(p: LinkProfile, m: LinkProfile, now: datetime | None, age: float | None) -> LinkProfile:
    if p.path != m.path:
        raise ValueError(f"link identity differs: {p.path} vs {m.path}")
    return replace(p, rtt_ms=_coef(p.rtt_ms, m.rtt_ms, now, age),
                   goodput_bytes_per_s=_coef(p.goodput_bytes_per_s, m.goodput_bytes_per_s, now, age),
                   round_trips_per_token=_coef(p.round_trips_per_token, m.round_trips_per_token, now, age),
                   cached_bytes=m.cached_bytes)


def _merge_device(p: DeviceProfile, m: DeviceProfile, now: datetime | None, age: float | None) -> DeviceProfile:
    if (p.node_id, p.device, p.backend) != (m.node_id, m.device, m.backend):
        raise ValueError(f"device identity differs: {p.node_id}:{p.device}/{p.backend} vs "
                         f"{m.node_id}:{m.device}/{m.backend}")
    link = p.link if m.link is None else (m.link if p.link is None else _merge_link(p.link, m.link, now, age))
    return replace(p, decode_ms_per_gib=_coef(p.decode_ms_per_gib, m.decode_ms_per_gib, now, age),
                   prefill_ms_per_gib_token=_coef(p.prefill_ms_per_gib_token, m.prefill_ms_per_gib_token, now, age),
                   link=link)


def _merge_storage(p: StorageProfile, m: StorageProfile, now: datetime | None, age: float | None) -> StorageProfile:
    if p.node_id != m.node_id:
        raise ValueError(f"storage identity differs: {p.node_id} vs {m.node_id}")
    return replace(p, read_bytes_per_s=_coef(p.read_bytes_per_s, m.read_bytes_per_s, now, age),
                   stream_efficiency=_coef(p.stream_efficiency, m.stream_efficiency, now, age))


_MERGERS = {Coefficient: _coef, DeviceProfile: _merge_device, LinkProfile: _merge_link,
            StorageProfile: _merge_storage}


def merge(prior_value: T, measured: T | None, *, now: datetime | None = None, max_age_s: float | None = None) -> T:
    """`prior_value` with every coefficient replaced by a usable measurement of the same identity.

    A measurement is usable when its provenance is `measured` and (when `max_age_s` is given) it was taken at most
    that long before `now`, which is then required. Identity is the coefficient unit, the device's (node, device,
    backend), the link's path class or the storage node; a mismatch raises ValueError, and so does a measured value
    that is NaN, infinite or negative (a broken measurement is reported, not replaced by the prior).
    """
    if max_age_s is not None and (now is None or not math.isfinite(max_age_s) or max_age_s < 0):
        raise ValueError("max_age_s must be a finite non-negative number and needs `now`")
    if measured is None:
        return prior_value
    if type(prior_value) is not type(measured):
        raise TypeError(f"cannot merge {type(measured).__name__} into {type(prior_value).__name__}")
    merger = _MERGERS.get(type(prior_value))
    if merger is None:
        raise TypeError(f"merge does not support {type(prior_value).__name__}")
    return merger(prior_value, measured, now, max_age_s)  # type: ignore[operator]


# --------------------------------------------------------------------------------------------------- CPU residency


@dataclass(frozen=True)
class CpuSplit:
    """Residency of a CPU prefix: resident units stay in memory, streamed units are read per pass into slots."""

    resident: tuple[int, ...]
    streamed: tuple[int, ...]
    slots: int
    slot_bytes: int
    resident_bytes: int  # private buffer copies when anything streams, else the mapped weight bytes
    stream_bytes: int  # storage bytes read per decoded token

    @property
    def committed_bytes(self) -> int:
        return self.slots * self.slot_bytes + self.resident_bytes


def all_mapped(units: Sequence[UnitSpec]) -> CpuSplit:
    return CpuSplit(resident=tuple(u.index for u in units), streamed=(), slots=0, slot_bytes=0,
                    resident_bytes=sum(u.weight_bytes for u in units), stream_bytes=0)


def _split_for_slot(units: Sequence[UnitSpec], budget: int, slots: int, slot: int,
                    extra: int | None = None) -> CpuSplit | None:
    """Units larger than `slot` resident, then the largest others while they fit (at most `extra` of them)."""
    forced = [u for u in units if u.buffer_bytes > slot]
    room = budget - slots * slot - sum(u.buffer_bytes for u in forced)
    if room < 0:
        return None
    chosen = {u.index for u in forced}
    added = 0
    for u in sorted((u for u in units if u.buffer_bytes <= slot), key=lambda u: (-u.buffer_bytes, u.index)):
        if (extra is None or added < extra) and u.buffer_bytes <= room:
            chosen.add(u.index)
            room -= u.buffer_bytes
            added += 1
    streamed = [u for u in units if u.index not in chosen]
    if not streamed:
        return None  # nothing streams: the mapped split covers this case
    return CpuSplit(resident=tuple(sorted(chosen)), streamed=tuple(u.index for u in streamed), slots=slots,
                    slot_bytes=max(u.buffer_bytes for u in streamed),
                    resident_bytes=sum(u.buffer_bytes for u in units if u.index in chosen),
                    stream_bytes=sum(u.buffer_bytes for u in streamed))


def split_cpu(units: Sequence[UnitSpec], budget: int, *, slots: int, stream: bool) -> CpuSplit | None:
    """CPU residency under `budget` bytes for weights and slots (admissible minus KV, workspace and overhead).

    All units stay mapped when they fit. Otherwise, with streaming allowed, each distinct unit buffer size is tried as
    the slot size: larger units must be resident, then the largest others become resident while the budget allows,
    and the rest stream. The fewest streamed bytes win, ties going to the smaller committed memory. None: no fit.
    """
    mapped = all_mapped(units)
    if mapped.resident_bytes <= budget:
        return mapped
    if not stream or not units:
        return None
    options = [o for s in sorted({u.buffer_bytes for u in units}) if (o := _split_for_slot(units, budget, slots, s))]
    return min(options, key=lambda o: (o.stream_bytes, o.committed_bytes), default=None)


def cpu_split_options(units: Sequence[UnitSpec], budget: int, *, slots: int, stream: bool) -> tuple[CpuSplit, ...]:
    """Every split `split_cpu` returns for some budget up to `budget`, the first being its choice for `budget`.

    A CPU whose pool later stages share gets `split_cpu` for the bytes those stages leave, which the search does not
    know when it places the CPU, so it tries each of these. Committed bytes fall and streamed bytes grow along the
    tuple. Stepping to one byte below each split's commitment finds them all because a greedy pass whose picks fit a
    smaller budget makes the same picks there; tests compare the tuple with `split_cpu` at every budget of small cases.
    """
    out: list[CpuSplit] = []
    room = budget
    while (o := split_cpu(units, room, slots=slots, stream=stream)) is not None:
        out.append(o)
        room = o.committed_bytes - 1
    return tuple(out)


def smallest_split(units: Sequence[UnitSpec], *, slots: int, stream: bool) -> CpuSplit:
    """The split that commits the least memory (everything mapped when streaming is off), for overflow reports."""
    if not stream or not units:
        return all_mapped(units)
    unbounded = sum(u.buffer_bytes for u in units) + slots * max(u.buffer_bytes for u in units)
    options = [o for s in sorted({u.buffer_bytes for u in units})
               if (o := _split_for_slot(units, unbounded, slots, s, extra=0))]
    return min([all_mapped(units), *options], key=lambda o: (o.committed_bytes, o.stream_bytes))


# --------------------------------------------------------------------------------------------------- formulas


def compute_ms(ms_per_gib: Coefficient, weight_bytes: int) -> float:
    """Time to compute `weight_bytes` of weights once: coefficient times GiB."""
    return ms_per_gib.value * weight_bytes / GIB


def stream_io_ms(stream_bytes: int, storage: StorageProfile) -> float:
    """Unbuffered read time of streamed weights: bytes / (read rate * stream efficiency)."""
    if stream_bytes <= 0:
        return 0.0
    rate = storage.read_bytes_per_s.value * storage.stream_efficiency.value
    if rate <= 0:
        raise ValueError("storage read rate times stream efficiency must be positive to stream")
    return stream_bytes / rate * 1000.0


def read_ms(nbytes: int, storage: StorageProfile) -> float:
    """Cold read of resident weights from the model file (no streaming efficiency factor)."""
    if nbytes <= 0:
        return 0.0
    if storage.read_bytes_per_s.value <= 0:
        raise ValueError("storage read rate must be positive")
    return nbytes / storage.read_bytes_per_s.value * 1000.0


def cpu_stage_ms(compute: float, io: float, slots: int) -> float:
    """CPU stage time: I/O overlaps compute with two or more slots, and adds to it with one."""
    return max(io, compute) if slots >= 2 else io + compute


def transfer_ms(nbytes: int, link: LinkProfile) -> float:
    if nbytes <= 0:
        return 0.0
    if link.goodput_bytes_per_s.value <= 0:
        raise ValueError(f"{link.path} link goodput must be positive")
    return nbytes / link.goodput_bytes_per_s.value * 1000.0


def hop_ms(link: LinkProfile, up_bytes: int, down_bytes: int, exchanges: float = 1.0) -> float:
    """Remote stage exchange: round_trips * rtt per exchange plus the bytes both ways over the goodput."""
    return exchanges * link.round_trips_per_token.value * link.rtt_ms.value + transfer_ms(up_bytes + down_bytes, link)
