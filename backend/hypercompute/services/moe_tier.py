"""The moe_offload tier of the llama.cpp planner: mixture-of-experts models larger than the GPU.

A token reads every layer's dense part (attention, norms, router, shared experts) but only expert_used_count of
expert_count routed experts. So the dense parts, the output head and the KV cache stay on the accelerators (-ngl all)
and only the routed-expert tensors of the fewest host layers move to the host CPU, through an explicit -ot pattern
over exactly those layers (never --n-cpu-moe, which counts from layer 0 and, with RPC peers first in --device order,
would pull a peer's experts onto this host). The CPU experts stay file-backed (--load-mode mmap, --no-repack), so on
Windows they occupy file cache instead of private commit that would grow the pagefile.

Peers take whole layers (dense part and experts) first, sized by their admissible budgets as the planner computes
them (contributor reserve and cap, services.deployments.remote_budgets). Every speed and SSD figure here is an estimate
from priors and is labelled so; nothing in this module is a measurement.
"""

from __future__ import annotations

from ..gguf import expert_override_pattern
from ..schemas import DeviceAssignment, DeviceBudget, MoeSpec
from ..units import GiB, fmt_bytes
from .planner import DECODE_MS_PER_GB, HOP_MS, ModelShape, _assignment, _cpu_side, _kind, _pool_ok

# CPU experts that miss the file cache are read by page faults on the memory-mapped GGUF: small reads, mostly one at a
# time, far below the unbuffered sequential rate the storage probe measures. A prior, never measured on this host.
MMAP_READ_BYTES_PER_S = 1.0 * GiB

MoeLayout = tuple[list[DeviceAssignment], MoeSpec | None, bool, str]  # layout, spec, fits, why


def _peers(shape: ModelShape, peers: list[DeviceBudget], keep_all: bool) -> tuple[list[DeviceAssignment], str]:
    """Peers in --device order take whole layers (dense part and experts) from the input side up to their budgets;
    the host keeps a layer, and with keep_all every later peer too. The text says why a peer cannot take part."""
    out, start = [], 0
    for k, b in enumerate(peers):
        n, stop = 0, shape.n_layer - 1 - (len(peers) - 1 - k if keep_all else 0)
        while start + n < stop and _assignment(shape, b, list(range(start, start + n + 1)), False).fits:
            n += 1
        if n == 0 and keep_all:
            return [], f"{b.device} cannot hold one whole layer"
        out += [_assignment(shape, b, list(range(start, start + n)), False)] if n else []
        start += n
    return out, ""


def cpu_expert_layers(shape: ModelShape, host: list[int], room: int) -> list[int] | None:
    """The fewest host layers whose experts must stay on the CPU so the other host experts fit in `room` bytes of
    the GPU: largest expert sets first, then lower index. None when the dense part alone overflows (room < 0)."""
    if room < 0:
        return None
    order = sorted(host, key=lambda i: (-shape.experts(i), i))
    left, k = sum(shape.experts(i) for i in host), 0
    while left > room:  # ends: with every expert moved, left is 0 <= room
        left, k = left - shape.experts(order[k]), k + 1
    return sorted(order[:k])


def gpu_cap(gpu: DeviceBudget, cpu_side: DeviceAssignment) -> int:
    """Bytes the local GPU may take. A discrete GPU: its own budget. A GPU in the CPU's pool (a Mac's MTL0 in `uma`):
    also no more than the pool bytes the CPU side leaves (its budget after the token embeddings, minus its compute
    buffer and overhead), so the Metal part and the CPU's private buffers come out of one budget, once; the file cache
    for the CPU experts gets whatever both leave (_spec). The GPU's own budget already holds the Metal working-set cap
    (services.deployments.local_budgets)."""
    if (gpu.node_id, gpu.pool_id) != (cpu_side.node_id, cpu_side.pool_id):
        return gpu.budget_bytes
    return min(gpu.budget_bytes, cpu_side.budget_bytes - cpu_side.total_bytes_est)


def _dense_overflow(gpu: DeviceBudget, cap: int, dense: DeviceAssignment, n_host: int) -> str:
    """Why the GPU cannot take even the dense part. On a unified pool the bytes the shared pool leaves after the CPU's
    buffers can bind before the GPU's own budget does; that remainder is named as such and never shown below zero."""
    need = (f"{gpu.device} cannot hold the dense part, head and KV of {n_host} layer(s): "
            f"~{fmt_bytes(dense.total_bytes_est)}")
    if cap == gpu.budget_bytes:
        return f"{need} of {fmt_bytes(cap)}"
    return (f"{need} of the {fmt_bytes(max(0, cap))} its {gpu.pool_id} memory pool leaves after this machine's CPU "
            f"buffers and token embeddings (free memory now minus the owner's reserve; its own budget is "
            f"{fmt_bytes(gpu.budget_bytes)}): free memory on this machine, then plan again")


def moe_offload_layout(shape: ModelShape, devs: list[DeviceBudget], cpu: DeviceBudget, keep_all: bool, *,
                       allow_disk: bool = True) -> MoeLayout:
    """Peers (first in `devs`) take whole layers; the local GPU (last) keeps every host layer's dense part and KV,
    the head, and the experts of all but the fewest host layers, whose experts stay on the host CPU file-backed.
    The host CPU holds no layer, so only its compute buffer and overhead are private RAM (and leased). On a unified
    pool the GPU's experts are sized against what the shared pool leaves (gpu_cap), not against the GPU budget alone."""
    gpu = devs[-1] if devs else None
    if not shape.is_moe or gpu is None or gpu.endpoint is not None or any(b.endpoint is None for b in devs[:-1]):
        return [], None, False, "moe_offload needs an MoE model and exactly one local GPU, after any peers"
    peers, why = _peers(shape, devs[:-1], keep_all)
    if why:
        return [], None, False, why
    return host_moe_layout(shape, peers, gpu, cpu, [cpu, *devs], allow_disk=allow_disk)


def host_moe_layout(shape: ModelShape, peers: list[DeviceAssignment], gpu: DeviceBudget, cpu: DeviceBudget,
                    budgets: list[DeviceBudget], *, allow_disk: bool = True) -> MoeLayout:
    """The host side after the peers' whole layers (layers 0.. in --device order): the local GPU keeps every later
    layer's dense part, KV and the head, and the experts of all but the fewest of them, whose experts stay on the host
    CPU file-backed. `budgets` holds every device of the layout, by (node_id, device), for the decode estimate. Also
    used for explicit layouts (services/layout.py), whose peers the owner sized."""
    host = list(range(sum(len(a.layers) for a in peers), shape.n_layer))
    cpu_side = _cpu_side(shape, cpu, 0, True)
    cap = gpu_cap(gpu, cpu_side)
    dense = _assignment(shape, gpu, host, True, kept_off=sum(shape.experts(i) for i in host))
    on_cpu = cpu_expert_layers(shape, host, cap - dense.total_bytes_est)
    if on_cpu is None:
        return [], None, False, _dense_overflow(gpu, cap, dense, len(host))
    gpu_a = _assignment(shape, gpu, host, True, kept_off=sum(shape.experts(i) for i in on_cpu))
    if not on_cpu:  # every expert fits on the accelerators: a plain GPU layout
        return [*peers, gpu_a], None, gpu_a.fits, "every expert fits on the accelerators"
    cpu_a = cpu_side.model_copy(update={"weight_bytes": sum(shape.experts(i) for i in on_cpu)})
    layout = [cpu_a, *peers, gpu_a]
    spec = _spec(shape, layout, on_cpu, budgets)
    if not (cpu_a.fits and _pool_ok(cpu_a, layout[1:])):
        return layout, spec, False, f"the host CPU's buffers do not fit {fmt_bytes(cpu_a.budget_bytes)} of RAM"
    if spec.ssd_bytes_per_token_est and not allow_disk:
        return layout, spec, False, "the CPU experts exceed the file cache and capacity mode is off"
    return layout, spec, True, (f"estimated fit: experts of {len(on_cpu)} of {len(host)} host layer(s) "
                                f"({fmt_bytes(spec.cpu_expert_bytes)}) stay on the CPU, file-backed; decode "
                                f"~{spec.decode_ms_est:.0f} ms/token (estimate)")


def _token_bytes(shape: ModelShape, a: DeviceAssignment, on_cpu: list[int]) -> float:
    """Weight bytes one decoded token reads on an accelerator: dense parts, the routed share of the experts it
    holds, and the head."""
    f = shape.active_fraction
    return (shape.output_bytes if a.holds_output else 0) + sum(
        shape.dense(i) + (0 if i in on_cpu else f * shape.experts(i)) for i in a.layers)


def _spec(shape: ModelShape, layout: list[DeviceAssignment], on_cpu: list[int], budgets: list[DeviceBudget]
          ) -> MoeSpec:
    """Estimates. The file cache is the host RAM budget the private allocations leave (and on a unified pool the
    GPU's share too); a token misses active_fraction * cpu_expert_bytes * miss_rate; decode adds the priors' compute
    on the bytes a token reads, the page-fault reads and one round trip per peer device."""
    cpu_a, accel, f = layout[0], layout[1:], shape.active_fraction
    by = {(b.node_id, b.device): b for b in budgets}
    shared = sum(a.total_bytes_est for a in accel if (a.node_id, a.pool_id) == (cpu_a.node_id, cpu_a.pool_id))
    cache = max(0, cpu_a.budget_bytes - cpu_a.total_bytes_est - shared)
    miss = max(0.0, 1.0 - cache / cpu_a.weight_bytes)
    ssd = int(f * cpu_a.weight_bytes * miss)
    ms = (DECODE_MS_PER_GB["cpu"] * f * cpu_a.weight_bytes / 1e9 + 1000.0 * ssd / MMAP_READ_BYTES_PER_S
          + sum(DECODE_MS_PER_GB[_kind(by[(a.node_id, a.device)])] * _token_bytes(shape, a, on_cpu) / 1e9
                + (HOP_MS if a.endpoint else 0.0) for a in accel))
    return MoeSpec(expert_count=shape.expert_count, expert_used_count=shape.expert_used_count, active_fraction=f,
                   cpu_expert_layers=on_cpu, cpu_expert_bytes=cpu_a.weight_bytes,
                   gpu_expert_bytes=sum(shape.experts(i) for i in accel[-1].layers if i not in on_cpu),
                   override=expert_override_pattern(on_cpu) + "=CPU", page_cache_bytes_est=cache, miss_rate_est=miss,
                   ssd_bytes_per_token_est=ssd, read_bytes_per_s=MMAP_READ_BYTES_PER_S, decode_ms_est=ms)


def _shared_pool_note(spec: MoeSpec, accel: list[DeviceAssignment], cpu: DeviceAssignment | None) -> list[str]:
    """On a unified pool (a Mac host) one line that shows the GPU part, the CPU buffers and the file cache adding up
    to the one RAM budget, so nothing is counted twice. Empty when the local GPU has its own memory."""
    gpu = next((a for a in accel if a.endpoint is None), None)
    if cpu is None or gpu is None or (gpu.node_id, gpu.pool_id) != (cpu.node_id, cpu.pool_id):
        return []
    return [f"this machine's {gpu.device} and CPU share the {gpu.pool_id} memory pool, counted once: the {gpu.device} "
            f"part (~{fmt_bytes(gpu.total_bytes_est)}: dense parts, head, KV, compute and "
            f"{fmt_bytes(spec.gpu_expert_bytes)} of experts, within its {fmt_bytes(gpu.budget_bytes)} budget, which "
            f"holds the GPU working-set limit), the CPU buffers (~{fmt_bytes(cpu.total_bytes_est)}) and the file cache "
            f"for the CPU experts (~{fmt_bytes(spec.page_cache_bytes_est)}) share {fmt_bytes(cpu.budget_bytes)} of "
            "free memory left after the owner's reserve and the token embeddings (estimates)"]


def moe_warnings(spec: MoeSpec, accel: list[DeviceAssignment], cpu: DeviceAssignment | None = None) -> list[str]:
    """What the tier keeps where and the speed class it is expected to reach, each number labelled an estimate."""
    tps = 1000.0 / spec.decode_ms_est if spec.decode_ms_est > 0 else 0.0
    speed = "interactive" if tps >= 10 else "readable" if tps >= 3 else "slow" if tps >= 0.5 else "very slow"
    peers = sum(a.weight_bytes for a in accel if a.endpoint)
    out = [f"moe_offload: attention, norms, router, shared experts, the head and the KV cache stay on the "
           f"accelerators; the routed experts of {len(spec.cpu_expert_layers)} host layer(s) "
           f"({fmt_bytes(spec.cpu_expert_bytes)}) stay on this machine's CPU, memory-mapped from the GGUF with no "
           "repacked or pinned copy (file cache, not committed RAM)"]
    out += _shared_pool_note(spec, accel, cpu)
    if peers:
        out.append(f"the peers hold whole layers ({fmt_bytes(peers)}), uploaded over the network before the first "
                   "token")
    if spec.ssd_bytes_per_token_est:
        out.append(f"capacity mode (MoE): ~{fmt_bytes(spec.page_cache_bytes_est)} of RAM left for the file cache "
                   f"against {fmt_bytes(spec.cpu_expert_bytes)} of CPU experts: an estimated {spec.miss_rate_est:.0%} "
                   f"of the {spec.active_fraction:.1%} of expert bytes a token routes to miss it, "
                   f"~{fmt_bytes(spec.ssd_bytes_per_token_est)} of SSD reads per token")
    out.append(f"expected speed ({speed}; an estimate from priors and an assumed page-fault read rate of "
               f"{fmt_bytes(int(spec.read_bytes_per_s))}/s, not a measurement): ~{spec.decode_ms_est:.0f} ms per "
               f"token, about {tps:.1f} tokens/s. Prompt processing of the CPU experts runs on the CPU")
    return out
