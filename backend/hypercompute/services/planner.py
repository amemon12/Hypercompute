"""Conservative placement planner for llama.cpp layer split, down the gpu -> gpu_ram -> gpu_disk ladder.

Per device:  weights(assigned layers [+ output head]) + KV(assigned layers) + compute_est + overhead_est
             <= budget.
llama.cpp assigns layer unit `il` (0..n_layer, where unit n_layer is the output head) to the first
device j whose cumulative tensor-split share exceeds il/(n_layer+1). Passing integer unit counts per
device therefore reproduces the plan exactly; the observed load log is still checked afterwards.

Device order is llama.cpp's own default: RPC peers first and the local GPU last, so the local GPU holds the
output head and only activations, never logits, cross the network (docs/research/split_optimization.md §0).
For one user a split is a sequential chain: fill the local GPU first and send only the overflow off it, to the
cheaper of the peers and the host CPU tier by a decode-cost prior (§3.4). objective=throughput splits by
capacity instead. Token embeddings stay in host RAM on the coordinator. All memory numbers are estimates until a
load is observed, and the plan says so. A mixture-of-experts model the accelerators cannot hold whole tries the
moe_offload rung first (services/moe_tier.py): dense parts on the GPU, routed experts of the fewest host layers on the
CPU, file-backed. On a unified-memory host (GB10, Apple) the GPU, the CPU tier and the file cache share one pool, so a
host part larger than that pool does not run from disk: paged CPU layers are charged to the pool like resident ones and
CPU units are not streamed; an MoE model larger than the pool runs there as moe_offload or not at all.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Literal

from ..schemas import (
    CandidateSummary,
    DeviceAssignment,
    DeviceBudget,
    GgufSummary,
    MoeSpec,
    Plan,
    PlanRequest,
    PlanStrategy,
    StreamSpec,
    utcnow,
)
from ..units import GiB, MiB, fmt_bytes

UBATCH = 512
COMPUTE_BASE = 96 * MiB
KV_BYTES_PER_ELEM = 2  # f16 K and V (llama.cpp default cache types)
OVERHEAD_BY_KIND = {"cuda": 450 * MiB, "metal": 256 * MiB, "vulkan": 300 * MiB, "cpu": 64 * MiB}
RPC_STAGING = 64 * MiB
# Decode cost priors (split_optimization.md §1, §3.4), ms per GB (1e9 bytes) read per token. CUDA and Metal are
# derived from real runs; the CPU figure is assumed (80 GB/s effective); Vulkan has no measurement and borrows the
# CUDA figure. HOP_MS is the blocking network round trip each RPC device adds per token (assumed Wi-Fi). Measured
# profiles (hypercompute tune) replace all of them.
DECODE_MS_PER_GB = {"cuda": 2.64, "metal": 10.85, "vulkan": 2.64, "cpu": 12.5}
HOP_MS = 4.0
UNIFIED_NO_STREAM = ("refused: the host part fits only by streaming CPU units from the SSD, and this machine's CPU and "
                     "GPU share one unified pool, where a host part larger than the pool is not run from disk")

Overflow = Literal["auto", "peers", "host_ram"]  # where the layers the local GPU cannot hold may go


@dataclass(frozen=True)
class StreamCapability:
    """What the hcstream extension can do for this model on this host.

    unit_buffer_bytes are the exact aligned buffers per unit from the model manifest (blocks, then the head at index
    n_layer). The storage rate is a measurement of unbuffered reads when one exists, otherwise a labelled prior;
    efficiency is the fraction of that rate the extension achieved end to end (0.7 on the 7B canary).
    """

    unit_buffer_bytes: tuple[int, ...]
    slots: int = 2
    read_bytes_per_s: float = 3.0 * GiB
    read_provenance: Literal["measured", "prior"] = "prior"
    efficiency: float = 0.7

    def io_ms(self, stream_bytes: int) -> float:
        return 1000.0 * stream_bytes / (self.read_bytes_per_s * self.efficiency) if stream_bytes else 0.0


@dataclass(frozen=True)
class ModelShape:
    n_layer: int
    layer_bytes: tuple[int, ...]
    output_bytes: int
    token_embd_bytes: int
    kv_per_layer: int
    compute_est: int
    expert_bytes: tuple[int, ...] = ()  # MoE: routed-expert bytes inside each layer_bytes entry; () for a dense model
    expert_count: int = 0
    expert_used_count: int = 0

    @classmethod
    def from_gguf(cls, s: GgufSummary, ctx_size: int) -> ModelShape | None:
        if not (s.n_layer and s.n_embd and s.n_head and s.n_head_kv) or len(s.layer_bytes) != s.n_layer:
            return None
        # llama.cpp sizes K/V from <arch>.attention.key_length / value_length, defaulting each to n_embd/n_head.
        # An MLA GGUF (deepseek2 with key_length_mla) declares 1 KV head of key_length 576 and value_length 512, and
        # b11770 caches only that K row (no V cache), so this K+V sum over-counts its cache ~1.9x: kept, conservative.
        k_len = s.key_length or s.n_embd // s.n_head
        v_len = s.value_length or s.n_embd // s.n_head
        kv = ctx_size * s.n_head_kv * (k_len + v_len) * KV_BYTES_PER_ELEM
        compute = COMPUTE_BASE + UBATCH * s.n_embd * 4 * 8
        n_exp, used = s.expert_count or 0, s.expert_used_count or 0
        moe = len(s.layer_expert_bytes) == s.n_layer and 0 < used <= n_exp  # else planned as dense, as before
        return cls(n_layer=s.n_layer, layer_bytes=tuple(s.layer_bytes), output_bytes=s.output_bytes,
                   token_embd_bytes=s.token_embd_bytes, kv_per_layer=kv, compute_est=compute,
                   expert_bytes=tuple(s.layer_expert_bytes) if moe else (), expert_count=n_exp if moe else 0,
                   expert_used_count=used if moe else 0)

    @property
    def is_moe(self) -> bool:
        return any(self.expert_bytes) and 0 < self.expert_used_count <= self.expert_count

    @property
    def active_fraction(self) -> float:
        """Share of a layer's expert bytes one token reads; 1.0 for a dense model."""
        return self.expert_used_count / self.expert_count if self.is_moe else 1.0

    def experts(self, i: int) -> int:
        return self.expert_bytes[i] if self.expert_bytes else 0

    def dense(self, i: int) -> int:
        return self.layer_bytes[i] - self.experts(i)

    def with_ubatch(self, ubatch: int) -> ModelShape:
        """The same model with the compute-buffer estimate scaled to another -ub (it grows with the micro-batch)."""
        return replace(self, compute_est=COMPUTE_BASE + (self.compute_est - COMPUTE_BASE) * ubatch // UBATCH)


def _kind(budget: DeviceBudget) -> str:
    """Backend family of a device (cuda, metal, vulkan or cpu); an RPC device is known by its description."""
    desc = (budget.description + " " + budget.device).lower()
    if budget.endpoint is None:
        if budget.device.startswith("CUDA"):
            return "cuda"
        if budget.device.startswith("MTL"):
            return "metal"
        return "vulkan" if budget.device.lower().startswith("vulkan") else "cpu"
    if desc.startswith("cpu "):  # a peer's CPU device ("<backend> <name> via <endpoint>"), e.g. "cpu Apple M4 via .."
        return "cpu"
    if "metal" in desc or "apple" in desc:
        return "metal"
    if "nvidia" in desc or "cuda" in desc or budget.pool_kind == "discrete_vram":
        return "cuda"
    return "cpu"


def _overhead(budget: DeviceBudget) -> int:
    return OVERHEAD_BY_KIND[_kind(budget)] + (RPC_STAGING if budget.endpoint else 0)


def _assignment(shape: ModelShape, b: DeviceBudget, layers: list[int], holds_output: bool,
                paged: bool = False, kept_off: int = 0) -> DeviceAssignment:
    """paged: the layers' weights stay in the memory-mapped GGUF and page in from disk, so only KV, compute and
    overhead must be resident (and leased). kept_off: expert bytes of these layers held elsewhere (moe_offload)."""
    weights = sum(shape.layer_bytes[i] for i in layers) + (shape.output_bytes if holds_output else 0) - kept_off
    kv = shape.kv_per_layer * len(layers)
    overhead = _overhead(b)
    total = (0 if paged else weights) + kv + shape.compute_est + overhead
    return DeviceAssignment(node_id=b.node_id, device=b.device, endpoint=b.endpoint, pool_id=b.pool_id,
                            layers=layers, holds_output=holds_output, weight_bytes=weights, kv_bytes=kv,
                            compute_bytes_est=shape.compute_est, overhead_bytes_est=overhead, total_bytes_est=total,
                            budget_bytes=b.budget_bytes, fits=total <= b.budget_bytes)


def _layer_capacity(shape: ModelShape, b: DeviceBudget, holds_output: bool) -> int:
    avg = sum(shape.layer_bytes) / shape.n_layer + shape.kv_per_layer
    free = b.budget_bytes - shape.compute_est - _overhead(b) - (shape.output_bytes if holds_output else 0)
    return max(0, int(free // avg)) if avg > 0 else 0


def _counts_greedy(shape: ModelShape, devices: list[DeviceBudget], floor: int = 0,
                   from_back: bool = True) -> list[int]:
    """Fill one device to capacity at a time; the device filled last takes whatever is left.

    from_back (the normal order): the last device, which holds the output head (the local GPU), fills first, then
    each earlier one, so the peers at the front only get the overflow. floor keeps that many layers back for every
    device not filled yet (run-together gives each one at least one). from_back=False is the head-on-a-peer
    fallback: the front device (the local GPU) fills first and the last one takes the rest and the head.
    """
    n = len(devices)
    counts = [0] * n
    remaining = shape.n_layer
    for step, i in enumerate(range(n - 1, -1, -1) if from_back else range(n)):
        later = n - 1 - step  # devices still to fill after this one
        if later == 0:
            counts[i] = remaining
            break
        reserve = floor * later + (0 if from_back or floor else 1)  # the head's device needs a layer of its own
        cap = _layer_capacity(shape, devices[i], holds_output=i == n - 1)
        counts[i] = max(1 if i == n - 1 else floor, min(cap, remaining - reserve))
        remaining -= counts[i]
    return counts


def _counts_proportional(shape: ModelShape, devices: list[DeviceBudget]) -> list[int]:
    caps = [max(1, _layer_capacity(shape, b, holds_output=i == len(devices) - 1)) for i, b in enumerate(devices)]
    total_cap = sum(caps)
    counts = [max(1, int(shape.n_layer * c / total_cap)) for c in caps]
    while sum(counts) > shape.n_layer:
        reducible = [k for k in range(len(counts)) if counts[k] > 1]
        if not reducible:  # more devices than layers: trailing devices get none, so run-together is infeasible
            counts[max(k for k in range(len(counts)) if counts[k] > 0)] = 0
            continue
        j = max(reducible, key=lambda k: counts[k] - caps[k])
        counts[j] -= 1
    i = 0
    while sum(counts) < shape.n_layer:
        spare = [k for k in range(len(counts)) if counts[k] < caps[k]]
        j = spare[i % len(spare)] if spare else len(counts) - 1
        counts[j] += 1
        i += 1
    return counts


def _layout(shape: ModelShape, devices: list[DeviceBudget], counts: list[int]) -> list[DeviceAssignment]:
    pairs = [(b, c) for b, c in zip(devices, counts, strict=True) if c > 0]
    out, start = [], 0
    for idx, (b, c) in enumerate(pairs):
        out.append(_assignment(shape, b, list(range(start, start + c)), holds_output=idx == len(pairs) - 1))
        start += c
    return out


def _index(devices: list[DeviceBudget], a: DeviceAssignment) -> int:
    return next(i for i, b in enumerate(devices) if b.device == a.device and b.node_id == a.node_id)


def _move(shape: ModelShape, devices: list[DeviceBudget], counts: list[int], layout: list[DeviceAssignment],
          src: int) -> list[int] | None:
    """counts with one layer moved off src onto the first fitting device that still fits after the move, or None.

    The move is laid out, not sized from one layer: a count shift moves the layers at the boundaries between src
    and the destination (and shifts every device in between), so only a fresh layout knows what each one holds.
    A device in between may overflow after the shift; the next pass moves layers off it.
    """
    for dst in (_index(devices, a) for a in layout if a.fits):
        trial = [c - (k == src) + (k == dst) for k, c in enumerate(counts)]
        if all(a.fits for a in _layout(shape, devices, trial) if _index(devices, a) == dst):
            return trial
    return None


def _rebalance(shape: ModelShape, devices: list[DeviceBudget], counts: list[int], keep_all: bool) -> list[int]:
    """Move layers off overflowing devices onto devices with room (bounded passes)."""
    counts = list(counts)
    for _ in range(shape.n_layer * 2):
        layout = _layout(shape, devices, counts)
        over = [a for a in layout if not a.fits]
        if not over:
            return counts
        src = _index(devices, over[0])
        if counts[src] <= (1 if keep_all else 0):
            return counts
        moved = _move(shape, devices, counts, layout, src)
        if moved is None:
            return counts
        counts = moved
    return counts


def _evaluate(shape: ModelShape, devices: list[DeviceBudget], keep_all: bool, proportional: bool = False,
              head_local: bool = True) -> list[DeviceAssignment]:
    # more devices than layers cannot all get a nonzero share: keep the end that holds the head
    devices = devices[-shape.n_layer:] if head_local else devices[: shape.n_layer]
    if proportional:
        counts = _counts_proportional(shape, devices)
    else:
        counts = _counts_greedy(shape, devices, floor=1 if keep_all else 0, from_back=head_local)
    return _layout(shape, devices, _rebalance(shape, devices, counts, keep_all))


def _tail(shape: ModelShape, k: int) -> ModelShape:
    """The model without its first k layers (what is left for the accelerators)."""
    return replace(shape, n_layer=shape.n_layer - k, layer_bytes=shape.layer_bytes[k:],
                   expert_bytes=shape.expert_bytes[k:])


def _shift(assignments: list[DeviceAssignment], k: int) -> list[DeviceAssignment]:
    return [a.model_copy(update={"layers": [i + k for i in a.layers]}) for a in assignments]


def _cpu_side(shape: ModelShape, cpu: DeviceBudget, k: int, paged: bool) -> DeviceAssignment:
    """The host CPU's first k layers. Token embeddings stay resident in host RAM (Plan.cpu_resident_bytes) on every
    tier, so they come out of the CPU budget first."""
    cpu = cpu.model_copy(update={"budget_bytes": max(0, cpu.budget_bytes - shape.token_embd_bytes)})
    return _assignment(shape, cpu, list(range(k)), holds_output=False, paged=paged)


def _pool_ok(cpu_a: DeviceAssignment, accel: list[DeviceAssignment], file_backed: int = 0) -> bool:
    """A unified-memory host shares one pool between CPU and GPU: the GPU's share is charged to the CPU check, and so
    are `file_backed` bytes, CPU weights held in that same pool's file cache (_file_backed)."""
    shared = sum(a.total_bytes_est for a in accel if a.node_id == cpu_a.node_id and a.pool_id == cpu_a.pool_id)
    return cpu_a.total_bytes_est + shared + file_backed <= cpu_a.budget_bytes


def _file_backed(cpu: DeviceBudget, cpu_a: DeviceAssignment, paged: bool) -> int:
    """Paged CPU weights a unified pool must hold. Their pages live in the file cache, which on a GB10 or a Mac is the
    memory the GPU allocates from, so they compete with it like resident layers (a gpu_disk plan larger than the GB10's
    pool thrashed it until a reboot, docs/research/PAPER-RUNS-2026-10-22.md R6.x). Elsewhere they page against RAM."""
    return cpu_a.weight_bytes if paged and cpu.pool_kind == "unified" else 0


def ram_tier_layout(shape: ModelShape, accel: list[DeviceBudget], cpu: DeviceBudget, keep_all: bool, *,
                    proportional: bool = False, paged: bool = False,
                    head_local: bool = True) -> tuple[list[DeviceAssignment], bool]:
    """Fewest host-CPU layers that let the rest fit on the accelerators.

    llama.cpp with -ngl N keeps units 0..(n_layer - N) on the CPU, i.e. the first layers (input side), and
    spreads the remaining layers plus the output head over --device by --tensor-split (llama_cpu_tier.md §1).
    paged (the gpu_disk tier): the CPU part is larger than free RAM, so its weights stay in the memory-mapped GGUF
    and page in from disk on every token. llama.cpp neither checks nor warns (llama_cpu_tier.md §3), and the GGUF is
    on disk already, so only the CPU layers' KV cache, compute buffer and overhead are charged to RAM. On a unified
    pool the paged weights are charged to the pool as well (_file_backed).
    """
    accel = accel[-shape.n_layer:] if head_local else accel[: shape.n_layer]
    if not accel or shape.n_layer < 2:
        return [], False
    for k in range(1, shape.n_layer):
        cpu_a = _cpu_side(shape, cpu, k, paged)
        if not cpu_a.fits:
            break
        tail = _shift(_evaluate(_tail(shape, k), accel, keep_all, proportional, head_local), k)
        if not _pool_ok(cpu_a, tail, _file_backed(cpu, cpu_a, paged)):
            continue
        if tail and all(a.fits for a in tail) and (not keep_all or len(tail) == len(accel)):
            return [cpu_a, *tail], True
    return [], False


def stream_split(units: list[int], buffer_bytes: tuple[int, ...], budget: int,
                 slots: int) -> tuple[list[int], list[int], int] | None:
    """(resident, streamed, slot_bytes) for CPU `units` within `budget` bytes of private weights plus slots.

    Every candidate slot size is a distinct unit size: units larger than the slot must stay resident, then the
    largest remaining units become resident while the budget allows, and the rest stream. The split with the fewest
    streamed bytes wins (then the smaller commitment). None when even the smallest slots do not fit.
    """
    size = {u: buffer_bytes[u] for u in units}
    if sum(size.values()) <= budget:
        return sorted(units), [], 0
    best: tuple[tuple[int, int], tuple[list[int], list[int], int]] | None = None
    for slot in sorted(set(size.values())):
        resident = [u for u in units if size[u] > slot]
        room = budget - slots * slot - sum(size[u] for u in resident)
        if room < 0:
            continue
        for u in sorted((u for u in units if size[u] <= slot), key=lambda u: (-size[u], u)):
            if size[u] <= room:
                resident.append(u)
                room -= size[u]
        streamed = [u for u in units if u not in resident]
        if not streamed:
            continue
        actual = max(size[u] for u in streamed)
        key = (sum(size[u] for u in streamed), slots * actual + sum(size[u] for u in resident))
        if best is None or key < best[0]:
            best = (key, (sorted(resident), sorted(streamed), actual))
    return best[1] if best else None


def _stream_cpu(shape: ModelShape, cpu: DeviceBudget, cap: StreamCapability, units: list[int],
                holds_output: bool) -> tuple[DeviceAssignment, StreamSpec] | None:
    """The host CPU assignment for a stream-tier layout: resident private copies plus slots, all of it leased."""
    layers = [u for u in units if u < shape.n_layer]
    kv = shape.kv_per_layer * len(layers)
    fixed = kv + shape.compute_est + OVERHEAD_BY_KIND["cpu"] + shape.token_embd_bytes
    split = stream_split(units, cap.unit_buffer_bytes, cpu.budget_bytes - fixed, cap.slots)
    if split is None:
        return None
    resident, streamed, slot_bytes = split
    slots = cap.slots if streamed else 0
    resident_bytes = sum(cap.unit_buffer_bytes[u] for u in resident)
    stream_bytes = sum(cap.unit_buffer_bytes[u] for u in streamed)
    weights = sum(shape.layer_bytes[i] for i in layers) + (shape.output_bytes if holds_output else 0)
    total = resident_bytes + slots * slot_bytes + kv + shape.compute_est + OVERHEAD_BY_KIND["cpu"]
    spec = StreamSpec(resident_units=resident, streamed_units=streamed, slots=slots, slot_bytes=slot_bytes,
                      resident_bytes=resident_bytes, stream_bytes_per_pass=stream_bytes,
                      read_bytes_per_s=cap.read_bytes_per_s, read_provenance=cap.read_provenance,
                      predicted_io_ms_per_token=cap.io_ms(stream_bytes))
    budget = max(0, cpu.budget_bytes - shape.token_embd_bytes)
    a = DeviceAssignment(node_id=cpu.node_id, device=cpu.device, endpoint=None, pool_id=cpu.pool_id, layers=layers,
                         holds_output=holds_output, weight_bytes=weights, kv_bytes=kv,
                         compute_bytes_est=shape.compute_est, overhead_bytes_est=OVERHEAD_BY_KIND["cpu"],
                         total_bytes_est=total, budget_bytes=budget, fits=total <= budget)
    return a, spec


def stream_tier_layout(shape: ModelShape, accel: list[DeviceBudget], cpu: DeviceBudget, keep_all: bool,
                       cap: StreamCapability, *, proportional: bool = False,
                       head_local: bool = True) -> tuple[list[DeviceAssignment], StreamSpec | None, bool]:
    """Fewest host-CPU layers that let the rest fit on the accelerators, with the CPU part resident where RAM allows
    and streamed from the SSD otherwise. Without accelerators every unit, the head included, runs on the CPU."""
    if len(cap.unit_buffer_bytes) != shape.n_layer + 1:
        return [], None, False
    if not accel:
        got = _stream_cpu(shape, cpu, cap, list(range(shape.n_layer + 1)), holds_output=True)
        return ([got[0]], got[1], got[0].fits) if got else ([], None, False)
    accel = accel[-shape.n_layer:] if head_local else accel[: shape.n_layer]
    for k in range(1, shape.n_layer):
        tail = _shift(_evaluate(_tail(shape, k), accel, keep_all, proportional, head_local), k)
        if not (tail and all(a.fits for a in tail) and (not keep_all or len(tail) == len(accel))):
            continue
        got = _stream_cpu(shape, cpu, cap, list(range(k)), holds_output=False)
        if got is None or not got[0].fits:
            continue
        if _pool_ok(got[0], tail):
            return [got[0], *tail], got[1], True
    return [], None, False


def layout_from_units(shape: ModelShape, devices: list[DeviceBudget], units: list[int], cpu: DeviceBudget | None,
                      cpu_layers: int, paged: bool = False) -> tuple[list[DeviceAssignment], bool]:
    """Assignments for an explicit split (a measured profile), checked like a fresh plan.

    `units` are --tensor-split counts in --device order; the last device's count includes the output head. The
    first `cpu_layers` layers run on the host CPU (-ngl = sum(units)).
    """
    if (not devices or len(units) != len(devices) or min(units) < 1 or cpu_layers < 0
            or sum(units) != shape.n_layer + 1 - cpu_layers or (cpu_layers and cpu is None)):
        return [], False
    out, start = [], cpu_layers
    for i, (b, u) in enumerate(zip(devices, units, strict=True)):
        n = u - (1 if i == len(devices) - 1 else 0)
        out.append(_assignment(shape, b, list(range(start, start + n)), holds_output=i == len(devices) - 1))
        start += n
    ok = all(a.fits for a in out)
    if not cpu_layers:
        return out, ok
    cpu_a = _cpu_side(shape, cpu, cpu_layers, paged)  # type: ignore[arg-type]
    return [cpu_a, *out], ok and cpu_a.fits and _pool_ok(cpu_a, out, _file_backed(cpu, cpu_a, paged))  # type: ignore[arg-type]


def order_devices(local: list[DeviceBudget], remote: list[DeviceBudget]) -> list[DeviceBudget]:
    """--device order: peers first and the local accelerators last, as llama.cpp orders them by default.

    The last device holds the output head, so logits stay on this machine and only n_embd-sized activations cross
    the network. Peers go by ascending budget, so the back-to-front fill reaches the largest peer first.
    """
    return [*sorted(remote, key=lambda b: b.budget_bytes), *local]


def decode_ms(layout: list[DeviceAssignment], budgets: list[DeviceBudget]) -> float:
    """Predicted single-stream decode time per token from the priors (split_optimization.md §3.2, weights only):
    the devices read their weights one after another, and every RPC device adds one blocking round trip."""
    by = {(b.node_id, b.device): b for b in budgets}
    total = 0.0
    for a in layout:
        b = by.get((a.node_id, a.device))
        total += DECODE_MS_PER_GB[_kind(b) if b else "cpu"] * a.weight_bytes / 1e9 + (HOP_MS if a.endpoint else 0.0)
    return total


# --------------------------------------------------------------------------- plan search


@dataclass(frozen=True)
class _Choice:
    layout: list[DeviceAssignment]
    ok: bool
    paged: bool = False
    stream: StreamSpec | None = None
    moe: MoeSpec | None = None


_NONE = _Choice([], False)


@dataclass
class _Search:
    """The candidate layouts tried for one request; each one is recorded in the plan's candidate list."""

    shape: ModelShape
    cpu: DeviceBudget | None
    proportional: bool
    head_local: bool
    allow_ram: bool
    allow_disk: bool
    stream_cap: StreamCapability | None = None
    candidates: list[CandidateSummary] = field(default_factory=list)
    budgets: list[DeviceBudget] = field(default_factory=list)

    def cost(self, c: _Choice) -> float:
        if c.moe is not None:  # decode_ms charges every expert byte; a token reads only the routed share
            return c.moe.decode_ms_est
        io = (c.stream.predicted_io_ms_per_token or 0.0) if c.stream else 0.0
        return decode_ms(c.layout, self.budgets) + io

    def _note(self, label: str, devs: list[DeviceBudget], c: _Choice, why: str) -> _Choice:
        self.candidates.append(CandidateSummary(label=label, devices=[f"{b.node_id}:{b.device}" for b in devs],
                                                feasible=c.ok, reason=why))
        return c

    def gpu(self, label: str, devs: list[DeviceBudget], keep_all: bool) -> _Choice:
        if not devs:
            return self._note(label, [], _NONE, "no devices")
        layout = _evaluate(self.shape, devs, keep_all, self.proportional, self.head_local)
        c = _Choice(layout, all(a.fits for a in layout) and (not keep_all or len(layout) == len(devs)))
        worst = max(layout, key=lambda a: a.total_bytes_est - a.budget_bytes)
        self.budgets += devs
        why = (f"estimated fit; predicted decode ~{self.cost(c):.1f} ms/token (priors)" if c.ok else
               f"{worst.device} needs ~{fmt_bytes(worst.total_bytes_est)} of {fmt_bytes(worst.budget_bytes)}")
        return self._note(label, devs, c, why)

    @property
    def unified(self) -> bool:
        """This machine's CPU tier is a unified pool (GB10, Apple) that its GPU and the file cache allocate from too."""
        return self.cpu is not None and self.cpu.pool_kind == "unified"

    def host(self, label: str, devs: list[DeviceBudget], keep_all: bool, paged: bool = False) -> _Choice:
        layout, ok = ram_tier_layout(self.shape, devs, self.cpu, keep_all, proportional=self.proportional,  # type: ignore[arg-type]
                                     paged=paged, head_local=self.head_local)
        c = _Choice(layout, ok, paged)
        where = "paged from the GGUF on disk" if paged else "in host RAM"
        self.budgets += [self.cpu, *devs]  # type: ignore[list-item]
        why = (f"does not fit even with layers {where}" if not ok else
               f"estimated fit with {len(layout[0].layers)} layer(s) {where}"
               + ("; disk bound" if paged else f"; predicted decode ~{self.cost(c):.1f} ms/token (priors)"))
        if paged and not ok and self.unified:
            why += ("; on this unified-memory host paged layers live in the file cache of the pool the GPU shares, so "
                    "they count like resident layers")
        return self._note(label, [self.cpu, *devs], c, why)  # type: ignore[list-item]

    def stream(self, label: str, devs: list[DeviceBudget], keep_all: bool) -> _Choice:
        layout, spec, ok = stream_tier_layout(self.shape, devs, self.cpu, keep_all, self.stream_cap,  # type: ignore[arg-type]
                                              proportional=self.proportional, head_local=self.head_local)
        self.budgets += [self.cpu, *devs]  # type: ignore[list-item]
        if self.unified and ok and spec is not None and spec.streamed_units:
            return self._note(label, [self.cpu, *devs], _NONE, UNIFIED_NO_STREAM)  # type: ignore[list-item]
        c = _Choice(layout, ok, stream=spec)
        if not ok or spec is None:
            why = "does not fit even with CPU units streamed from the SSD through bounded slots"
        else:
            why = (f"estimated fit: {len(spec.resident_units)} CPU unit(s) resident "
                   f"({fmt_bytes(spec.resident_bytes)}), {len(spec.streamed_units)} streamed through "
                   f"{spec.slots} slot(s) of {fmt_bytes(spec.slot_bytes)}, {fmt_bytes(spec.stream_bytes_per_pass)} "
                   f"read per token; predicted decode ~{self.cost(c):.0f} "
                   f"ms/token ({spec.read_provenance} storage rate)")
        return self._note(label, [self.cpu, *devs], c, why)  # type: ignore[list-item]

    def moe(self, label: str, devs: list[DeviceBudget], keep_all: bool) -> _Choice:
        from .moe_tier import moe_offload_layout  # imported here: moe_tier builds on this module's helpers

        layout, spec, ok, why = moe_offload_layout(self.shape, devs, self.cpu, keep_all,  # type: ignore[arg-type]
                                                   allow_disk=self.allow_disk)
        self.budgets += [self.cpu, *devs]  # type: ignore[list-item]
        return self._note(label, [self.cpu, *devs], _Choice(layout, ok, moe=spec), why)  # type: ignore[list-item]

    def ladder(self, c: _Choice, devs: list[DeviceBudget], keep_all: bool, label: str, ram: bool = True,
               moe: bool = True) -> _Choice:
        """gpu -> moe_offload (MoE models: experts of the fewest host layers on the CPU) -> gpu_ram -> stream
        (bounded, measured reads) or else gpu_disk (OS paging) on one device set: the first rung that fits, else the
        GPU-only attempt."""
        if c.ok:
            return c
        if moe and self.shape.is_moe and self.allow_ram and devs:
            if (m := self.moe(f"{label} + MoE experts on the host CPU", devs, keep_all)).ok:
                return m
        if ram and self.allow_ram and devs and (r := self.host(f"{label} + host RAM", devs, keep_all)).ok:
            return r
        if self.allow_ram and self.stream_cap is not None:
            if (st := self.stream(f"{label} + CPU units resident or streamed", devs, keep_all)).ok:
                return st
        if self.allow_disk and devs and (d := self.host(f"{label} + CPU layers paged from disk", devs, keep_all,
                                                        True)).ok:
            return d
        return c


def _head_fits(shape: ModelShape, local: list[DeviceBudget]) -> bool:
    return not local or _layer_capacity(shape, local[-1], holds_output=True) >= 1


def _moe_overflow(s: _Search, local: list[DeviceBudget], peers: list[DeviceBudget], warnings: list[str]) -> _Choice:
    """An MoE model off the local GPU: its experts to the host CPU on this machine alone, or with every peer taking
    whole layers first; the lower decode estimate wins and both stay in the candidates."""
    tried = [s.moe("MoE experts on the host CPU", local, keep_all=False)]
    if peers:
        tried.append(s.moe(f"{len(peers)} peer device(s) + MoE experts on the host CPU", order_devices(local, peers),
                           keep_all=False))
    fits = sorted((c for c in tried if c.ok), key=s.cost)
    if len(fits) == 2:
        warnings.append(f"MoE placement {'with' if fits[0] is tried[-1] else 'without'} the peers: estimated decode "
                        f"~{s.cost(fits[0]):.0f} ms/token against ~{s.cost(fits[1]):.0f} (priors, not measured); "
                        "the other layout is in the candidates")
    return fits[0] if fits else _NONE


def _overflow(s: _Search, local: list[DeviceBudget], peers: list[DeviceBudget], overflow: Overflow,
              warnings: list[str]) -> _Choice:
    """The local GPU alone does not fit: send only the overflow off it, to peers or to the host CPU tier, whichever
    the decode prior says is cheaper (split_optimization.md §3.4); both stay in the candidates for the tuner. An MoE
    model that peers cannot hold whole keeps its dense part on the GPU and moves experts first (moe_offload)."""
    via_peers = _NONE
    if overflow != "host_ram":
        for k in range(1, len(peers) + 1):
            devs = order_devices(local, peers[:k]) if s.head_local else [*local, *peers[:k]]
            if (via_peers := s.gpu(f"overflow to {k} peer device(s)", devs, keep_all=False)).ok:
                break
    if overflow == "auto" and s.shape.is_moe and s.allow_ram and local and not via_peers.ok:
        if (moe := _moe_overflow(s, local, peers, warnings)).ok:
            return moe
    via_ram = s.host("overflow to host RAM", local, False) if overflow != "peers" and s.allow_ram and local else _NONE
    if overflow != "auto":  # the tuner's alternative: exactly that overflow tier, no further fallback
        return via_peers if overflow == "peers" else via_ram
    if via_peers.ok and via_ram.ok:
        p, r = s.cost(via_peers), s.cost(via_ram)
        win, lose = ("host RAM", "the peers") if r <= p else ("the peers", "host RAM")
        warnings.append(f"overflow goes to {win}: predicted decode ~{min(p, r):.1f} ms/token against ~{max(p, r):.1f} "
                        f"for {lose} (priors: host CPU {DECODE_MS_PER_GB['cpu']} ms/GB, ~{HOP_MS:.0f} ms network round "
                        "trip per peer device per token). The other layout is in the candidates; `hypercompute tune` "
                        "measures both.")
        return via_ram if r <= p else via_peers
    if via_peers.ok or via_ram.ok:
        return via_peers if via_peers.ok else via_ram
    devs = order_devices(local, peers) if s.head_local else [*local, *peers]
    return s.ladder(_NONE, devs, False, "GPUs + peers" if peers else "this machine: GPU", ram=bool(peers), moe=False)


def _together(s: _Search, local: list[DeviceBudget], remote: list[DeviceBudget], local_ok: bool,
              reasons: list[str], warnings: list[str]) -> _Choice:
    if not remote:
        reasons.append("no eligible peer selected: 'run together' needs at least one joined, verified friend")
        return _NONE
    reasons.append("strategy: run together (every selected device gets at least one layer)")
    devs = order_devices(local, remote) if s.head_local else [*local, *remote]
    c = s.ladder(s.gpu("every selected device participates", devs, keep_all=True), devs, True, "every selected device")
    if local_ok:
        warnings.append("the model fits on this machine alone; splitting it adds network hops and is "
                        "expected to be slower. This run demonstrates participation, not necessity.")
    return c


def _base(req: PlanRequest) -> dict:
    return dict(id=uuid.uuid4().hex, model_id=req.model_id, strategy=req.strategy, ctx_size=req.ctx_size,
                heuristic=True, created_at=utcnow())


def make_plan(req: PlanRequest, shape: ModelShape | None, local: list[DeviceBudget], remote: list[DeviceBudget],
              host_ram_free: int | None, cpu: DeviceBudget | None = None, *, overflow: Overflow = "auto",
              stream: StreamCapability | None = None) -> Plan:
    """The llama.cpp plan for one request. overflow restricts where layers the local GPU cannot hold may go (the
    tuner uses it to build the alternative it measures); "auto" picks the cheaper by the decode prior. The request's
    own overflow field applies when the keyword is left at auto."""
    overflow = overflow if overflow != "auto" else req.overflow
    if shape is None:
        return Plan(**_base(req), feasible=False, fit_label="fit_unknown", assignments=[], device_list=[],
                    tensor_split=[], rpc_endpoints=[], cpu_resident_bytes=0, reasons=[
                        "model metadata unavailable (not downloaded or not a supported GGUF), so no fit estimate is "
                        "possible"], warnings=[], candidates=[])
    reasons: list[str] = []
    warnings: list[str] = []
    allow_ram = req.allow_ram_tier and cpu is not None
    s = _Search(shape, cpu, proportional=req.objective == "throughput",
                head_local=_head_fits(shape, local) or not remote, allow_ram=allow_ram,
                allow_disk=allow_ram and req.allow_capacity_mode,
                stream_cap=stream if allow_ram and req.allow_capacity_mode else None)
    if not s.head_local:
        warnings.append("this machine's GPU cannot hold the output head, so the last peer holds it and the logits "
                        "(vocabulary-sized) cross the network every token")
    if not local and req.strategy == PlanStrategy.local_only:
        reasons.append("this machine exposes no accelerator to llama.cpp")
    local_c = s.gpu("this machine only", local, keep_all=False)
    if req.strategy == PlanStrategy.local_only:
        reasons.append("strategy: this machine only")
        choice = s.ladder(local_c, local, False, "this machine: GPU")
    elif req.strategy in (PlanStrategy.fastest_fit, PlanStrategy.auto):
        reasons.append("strategy: " + ("best available (fastest tier that fits)" if req.strategy == PlanStrategy.auto
                                       else "fastest fit (fewest devices, local first; peers only when needed)"))
        peers = sorted(remote, key=lambda b: -b.budget_bytes)
        choice = local_c if local_c.ok else _overflow(s, local, peers, overflow, warnings)
    else:
        choice = _together(s, local, remote, local_c.ok, reasons, warnings)
    if req.objective == "throughput" and len([a for a in choice.layout if a.device != "CPU"]) > 1:
        reasons.append("objective: throughput (layers split in proportion to each device's capacity)")
    return finish_plan(req, shape, choice.layout, choice.ok, paged=choice.paged, reasons=reasons, warnings=warnings,
                       candidates=s.candidates, host_ram_free=host_ram_free,
                       total_budget=sum(b.budget_bytes for b in [*local, *remote]), stream=choice.stream,
                       moe=choice.moe)


def finish_plan(req: PlanRequest, shape: ModelShape, chosen: list[DeviceAssignment], ok: bool, *, paged: bool,
                reasons: list[str], warnings: list[str], candidates: list[CandidateSummary],
                host_ram_free: int | None, total_budget: int, stream: StreamSpec | None = None,
                moe: MoeSpec | None = None) -> Plan:
    """A Plan from a chosen layout: llama.cpp arguments, tier and the warnings that go with it."""
    reasons, warnings = list(reasons), list(warnings)
    cpu_bytes = shape.token_embd_bytes
    if host_ram_free is not None and cpu_bytes + 512 * MiB > host_ram_free:
        warnings.append(f"token embeddings need ~{fmt_bytes(cpu_bytes)} of host RAM on this machine; only "
                        f"{fmt_bytes(host_ram_free)} is free")
    if req.ctx_size > 32768:
        warnings.append("large context: KV cache estimate grows linearly with ctx_size")
    feasible = bool(chosen) and ok
    if chosen and not ok:
        reasons.extend(f"{a.node_id}:{a.device} over budget: ~{fmt_bytes(a.total_bytes_est)} needed, "
                       f"{fmt_bytes(a.budget_bytes)} allowed" for a in chosen if not a.fits)
    cpu_layers = len(chosen[0].layers) if chosen and chosen[0].device == "CPU" else 0
    accel = [a for a in chosen if a.device != "CPU"]
    if feasible and stream is not None:
        warnings.extend(_stream_warnings(shape, stream, cpu_layers, bool(accel)))
    if feasible and moe is not None:
        from .moe_tier import moe_warnings  # imported here: moe_tier builds on this module's helpers

        warnings.extend(moe_warnings(moe, accel, chosen[0] if chosen[0].device == "CPU" else None))
    if feasible and cpu_layers and not paged and stream is None:
        warnings.append(f"{cpu_layers} of {shape.n_layer} layers run on this machine's CPU from system RAM because "
                        "the GPU part cannot hold them; expect noticeably fewer tokens/s than all-GPU.")
    if feasible and paged:
        warnings.append(f"capacity mode: {cpu_layers} of {shape.n_layer} layers (~{fmt_bytes(chosen[0].weight_bytes)}) "
                        "run on this machine's CPU straight from the GGUF on disk, because they do not fit in the "
                        f"~{fmt_bytes(chosen[0].budget_bytes)} of free RAM left for CPU layers. Every token re-reads "
                        "the evicted part from disk, so expect seconds per token, far slower than the RAM tier.")
    reasons.append(f"contributed / estimated usable across selected devices: {fmt_bytes(total_budget)} "
                   f"(separate memory pools, not one fast global memory)")
    if shape.layer_bytes and max(shape.layer_bytes) > 2 * GiB:
        warnings.append("very large layers: per-layer granularity may waste budget")
    return Plan(
        **_base(req),
        feasible=feasible,
        fit_label="estimated_fit" if feasible else "does_not_fit",
        assignments=chosen,
        device_list=[a.device for a in accel],
        tensor_split=[len(a.layers) + (1 if a.holds_output else 0) for a in accel],
        rpc_endpoints=sorted({a.endpoint for a in chosen if a.endpoint}),
        cpu_resident_bytes=cpu_bytes,
        reasons=reasons,
        warnings=warnings,
        candidates=candidates,
        tier="moe_offload" if moe is not None else "stream" if stream is not None else (
            "gpu_disk" if paged and cpu_layers else ("gpu_ram" if cpu_layers else "gpu")),
        cpu_layers=cpu_layers,
        n_gpu_units=(shape.n_layer + 1 - cpu_layers if accel else 0) if cpu_layers else None,
        capacity_mode=(bool(paged and cpu_layers) or bool(stream and stream.streamed_units)
                       or bool(moe and moe.ssd_bytes_per_token_est)),
        stream=stream,
        moe=moe,
    )


def _stream_warnings(shape: ModelShape, spec: StreamSpec, cpu_layers: int, accel: bool) -> list[str]:
    where = "this machine's CPU" if accel else "this machine's CPU only (no accelerator in the plan)"
    out = [f"{cpu_layers} of {shape.n_layer} layers run on {where}: {len(spec.resident_units)} unit(s) "
           f"({fmt_bytes(spec.resident_bytes)}) are copied once into RAM and stay resident"]
    if spec.streamed_units:
        out.append(f"capacity mode (bounded): {len(spec.streamed_units)} unit(s) stream from the GGUF on the SSD "
                   f"through {spec.slots} slot(s) of {fmt_bytes(spec.slot_bytes)}, so this machine commits "
                   f"{fmt_bytes(spec.committed_bytes)} for CPU weights instead of "
                   f"{fmt_bytes(spec.resident_bytes + spec.stream_bytes_per_pass)} and reads "
                   f"{fmt_bytes(spec.stream_bytes_per_pass)} from the SSD for every token (about "
                   f"{spec.predicted_io_ms_per_token or 0:.0f} ms of reads per token at the {spec.read_provenance} "
                   "storage rate). Devices that join with free memory can take streamed units off the SSD.")
    return out
