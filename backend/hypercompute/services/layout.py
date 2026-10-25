"""Explicit layouts: the owner names every stage of a llama.cpp split (PlanRequest.layout) and the planner only checks
it, so a measurement holds the placement fixed while admission, the deployment manager, the llama adapter, the
placement check and run_big all run their normal gateway path.

A layout is an ordered list of stages, each the next N whole layers on one device: peers first, each peer's stages in
its worker's -d order (MTL0 before CPU for -d MTL0,CPU) though not necessarily next to each other, and this machine's
GPU last, which also holds the output head. A mixture-of-experts model whose host layers do not fit the GPU keeps the
routed experts of the fewest of them on the host CPU, file-backed, exactly as the moe_offload tier does
(services/moe_tier.py, same -ot).

Memory is checked like a plan's, with the same estimates (nothing here is a measurement):
- a resident stage holds its weights, KV, compute buffer and overhead in that device's memory;
- a file-backed stage (a peer CPU device whose worker reports hcpeer on: native/hcpeer maps every cached weight over
  10 MiB from the peer's RPC cache) keeps its dense parts (read every token, so assumed hot), KV, compute buffer and
  overhead in RAM, and needs all of its weight bytes on the peer's cache volume;
- each stage fits its device (the node's whole contribution within that device's free memory), and a node's stages
  together fit the node (its contribution, within the largest device figure: a Mac's MTL0 and CPU share one pool);
- a peer with a file-backed stage can save every weight its worker is sent at load into its RPC cache, its resident
  Metal stage's included (rpc-server -c keeps every hash-cached tensor, and b11770 loads RPC0 before RPC1), within
  the room hcpeer v4 checks at save time: --max-disk in whole GiB minus every file the cache holds now, and the
  volume's free space above the floor. hcpeer copies a save past either into RAM, and the agent's disk guard frees
  room below the cap only a pass at a time (agent/managed.py HCPEER_SAVE_ROOM_BYTES), which a load can outrun, so
  another model's cached files are no room. Only a finished seed of these layers (or more) of this model is known to
  be in the cache already (those loads save nothing);
- a peer whose worker runs hcpeer maps every cached CPU weight (no per-stage opt-out), so its CPU stage is file-backed
  and a resident one is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..agent.hcpeer import cache_cap_gib
from ..schemas import (
    CandidateSummary,
    DeviceAssignment,
    DeviceBudget,
    LayoutStage,
    ManagedReport,
    MoeSpec,
    NodeView,
    Plan,
    PlanRequest,
)
from ..units import GiB, fmt_bytes
from .moe_tier import host_moe_layout
from .nodes import LOCAL_ID
from .peer_seed import SeededShard
from .planner import ModelShape, _assignment, finish_plan

HCPEER_FIX = ("on that Mac install hcpeer (scripts/build_hcpeer_mac.py install) and rejoin with HC_PEER_MMAP=1, "
              "--managed and --cache; or set the stage's residency to resident")
RESIDENT_FIX = "restart that Mac's agent without HC_PEER_MMAP=1 for a resident CPU stage"


class LayoutError(ValueError):
    """An explicit layout the planner refuses; the gateway answers 422 with its code, message and fix."""

    def __init__(self, code: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.code, self.fix = code, fix


@dataclass(frozen=True)
class PeerFacts:
    """What one peer offers an explicit layout."""

    node_id: str
    name: str  # display name, for messages
    devices: tuple[str, ...]  # its worker's -d list, one name per RPC device the host's probe verified
    caps: tuple[DeviceBudget, ...]  # per device: the node's whole contribution within that device's free memory
    node_budget: int  # RAM for all of this node's stages together
    disk_budget: int | None  # bytes hcpeer still lets its worker save; None: it can take no file-backed stage
    disk_basis: str  # what bound disk_budget, or why there is none
    hcpeer_on: bool = False  # its worker's banner shows hcpeer on: every cached CPU weight is mapped
    seeded: SeededShard | None = None  # a finished seed job: these layers are in its cache already


@dataclass(frozen=True)
class _PeerStage:
    peer: PeerFacts
    index: int  # the device's position in the worker's -d list: its RPC device index within that server
    layers: int
    file_backed: bool

    @property
    def label(self) -> str:
        return f"{self.peer.name} {self.peer.devices[self.index]}"

    @property
    def cap(self) -> DeviceBudget:
        return self.peer.caps[self.index]


def disk_budget(report: ManagedReport | None) -> tuple[int | None, str]:
    """Bytes hcpeer v4 still lets a peer's worker save into its RPC cache (native/hcpeer save_atomic): the cap
    HC_PEER_CACHE_MAX_GIB (--max-disk in whole GiB) minus every file the cache holds now, and the volume's free space
    above the floor. None with the reason it can take no file-backed stage."""
    if report is None:
        return None, "it is not a managed peer (join --managed), so the host sees neither its cache nor hcpeer"
    if report.hcpeer_mmap is None:
        return None, "its agent does not report hcpeer (an agent or host without hcpeer reports)"
    if not report.hcpeer_mmap:
        return None, "its worker's banner does not show 'hcpeer mmap : on' (no HC_PEER_MMAP=1, or the stock library)"
    if not report.cache_enabled:
        return None, "its worker runs without the RPC cache (-c) that hcpeer maps from"
    if report.disk_free_bytes is None or report.cache_bytes is None:
        return None, "it reported no free space or cache size for its cache volume"
    floor_room = report.disk_free_bytes - report.min_free_bytes
    cap = cache_cap_gib(report.max_disk_bytes)  # what the agent passes hcpeer as HC_PEER_CACHE_MAX_GIB
    if cap is not None and cap * GiB - report.cache_bytes < floor_room:
        return max(0, cap * GiB - report.cache_bytes), (f"its cache cap (--max-disk, {cap} GiB as hcpeer applies it) "
                                                        f"- {fmt_bytes(report.cache_bytes)} cached")
    return max(0, floor_room), (f"{fmt_bytes(report.disk_free_bytes)} free on its cache volume - "
                                f"{fmt_bytes(report.min_free_bytes)} floor")


def peer_facts(node: NodeView, caps: Sequence[DeviceBudget], report: ManagedReport | None,
               seeded: SeededShard | None = None) -> PeerFacts:
    """caps: services.deployments.remote_budgets(node, whole_contribution=True), one per verified RPC device.
    seeded: services.peer_seed.ManagedPeers.seeded(node.id)."""
    names = tuple(node.contribution.devices[i] if i < len(node.contribution.devices) else f"#{i}"
                  for i in range(len(caps)))
    node_budget = min(node.contribution.max_bytes, max((b.budget_bytes for b in caps), default=0))
    disk, basis = disk_budget(report)
    return PeerFacts(node.id, node.display_name, names, tuple(caps), node_budget, disk, basis,
                     hcpeer_on=report is not None and report.hcpeer_mmap is True, seeded=seeded)


# --------------------------------------------------------------------------- checks


def _check_chain(shape: ModelShape, stages: Sequence[LayoutStage]) -> None:
    total = sum(s.layers for s in stages)
    if total != shape.n_layer:
        raise LayoutError("layout_layer_count", f"the stages hold {total} layer(s) and the model has {shape.n_layer}; "
                          "every layer goes to exactly one stage")
    last = stages[-1]
    if last.node_id != LOCAL_ID:
        raise LayoutError("layout_order", f"the last stage must be this machine's GPU (node_id 'local'), which holds "
                          f"the output head, not {last.node_id}:{last.device}")
    if any(s.node_id == LOCAL_ID for s in stages[:-1]):
        raise LayoutError("layout_order", "only the last stage is on this machine: peers come first and this "
                          "machine's GPU last, as llama.cpp orders them. The host CPU is no stage: an MoE model keeps "
                          "the routed experts its GPU cannot hold there, as the moe_offload tier does")


def _local_gpu(stage: LayoutStage, local: Sequence[DeviceBudget]) -> DeviceBudget:
    if stage.residency == "file_backed":
        raise LayoutError("layout_residency", "this machine's GPU holds its layers resident; only a peer's CPU "
                          "device can be file-backed")
    gpu = next((b for b in local if b.device.upper() == stage.device.upper()), None)
    if gpu is None:
        raise LayoutError("layout_device", f"this machine has no contributed accelerator {stage.device} (it has "
                          f"{', '.join(b.device for b in local) or 'none'}); the last stage is its GPU")
    return gpu


def _device_index(peer: PeerFacts, device: str) -> int:
    found = [i for i, name in enumerate(peer.devices) if name.upper() == device.upper()]
    if not found:
        raise LayoutError("layout_device", f"{peer.name} ({peer.node_id[:8]}) has no verified RPC device {device}; "
                          f"its worker runs -d {','.join(peer.devices) or '(nothing verified)'}",
                          fix="restart its worker with that device (POST /api/peers/<node>/worker), then plan again")
    return found[0]


def _file_backed(peer: PeerFacts, stage: LayoutStage, allow_capacity_mode: bool) -> bool:
    is_cpu = stage.device.upper() == "CPU"
    if stage.residency == "file_backed" and not is_cpu:
        raise LayoutError("layout_residency", f"{stage.device} on {peer.name} cannot be file-backed: hcpeer maps only "
                          "CPU-buffer weights (Metal buffers are wired)")
    if stage.residency == "resident" and is_cpu and peer.hcpeer_on:
        raise LayoutError("layout_residency", f"{peer.name}'s CPU stage cannot be resident: its worker runs hcpeer, "
                          "which maps every cached CPU weight over 10 MiB from its SSD (there is no per-stage opt-out)",
                          fix=f"{RESIDENT_FIX}, or leave the stage's residency unset (file-backed)")
    wanted = stage.residency == "file_backed" or (stage.residency is None and is_cpu and peer.hcpeer_on)
    if wanted and peer.disk_budget is None:
        raise LayoutError("hcpeer_not_reported", f"{peer.name}'s CPU stage cannot be file-backed: {peer.disk_basis}",
                          fix=RESIDENT_FIX if peer.hcpeer_on else HCPEER_FIX)
    if wanted and not allow_capacity_mode:
        raise LayoutError("layout_capacity_mode", f"{peer.name}'s CPU stage is file-backed, so its weights are read "
                          "from that peer's SSD (capacity mode), which this request turns off",
                          fix=f"set allow_capacity_mode, or {RESIDENT_FIX}")
    return wanted


def _peer_stages(stages: Sequence[LayoutStage], peers: Mapping[str, PeerFacts],
                 allow_capacity_mode: bool) -> list[_PeerStage]:
    # A peer may be visited again later in the chain (every Metal stage first, then the file-backed CPU stages): each
    # handoff between servers is one blocking get, while a peer's own MTL0 to CPU handoff tries a same-server copy that
    # fails between Metal and CPU buffers and then does the get and set anyway. Within a peer, -d order still holds.
    out: list[_PeerStage] = []
    last_index: dict[str, int] = {}
    for stage in stages:
        peer = peers.get(stage.node_id)
        if peer is None:
            raise LayoutError("layout_node", f"{stage.node_id} is not an eligible peer of this request (joined, "
                              "verified, within node_ids, and a strategy other than local_only)",
                              fix="GET /api/nodes lists the peers with their ids and states")
        index = _device_index(peer, stage.device)
        if index <= last_index.get(peer.node_id, -1):
            raise LayoutError("layout_order", f"{peer.name}'s stages must follow its worker's -d order "
                              f"({','.join(peer.devices)}), each device once")
        last_index[peer.node_id] = index
        out.append(_PeerStage(peer, index, stage.layers, _file_backed(peer, stage, allow_capacity_mode)))
    return out


def _stage_assignment(shape: ModelShape, st: _PeerStage, layers: list[int]) -> DeviceAssignment:
    a = _assignment(shape, st.cap, layers, holds_output=False)
    if not st.file_backed:
        return a
    ram = a.total_bytes_est - a.weight_bytes + sum(shape.dense(i) for i in layers)
    return a.model_copy(update={"total_bytes_est": ram, "fits": ram <= st.cap.budget_bytes, "file_backed": True})


def _peer_assignments(shape: ModelShape, stages: Sequence[_PeerStage]) -> list[DeviceAssignment]:
    out, start = [], 0
    for st in stages:
        out.append(_stage_assignment(shape, st, list(range(start, start + st.layers))))
        start += st.layers
    return out


def _load_saves(peer: PeerFacts, mine: Sequence[DeviceAssignment], model_id: str) -> int:
    """Bytes this peer's worker saves into its RPC cache at load: with -c every weight it is sent, its resident stages'
    included; none when a finished seed of these layers of this model already put them there (cache hits)."""
    layers = {n for a in mine for n in a.layers}
    seeded = peer.seeded
    if seeded is not None and seeded.model_id == model_id and layers <= seeded.layers:
        return 0
    return sum(a.weight_bytes for a in mine)


def _check_disk(peer: PeerFacts, mine: Sequence[DeviceAssignment], model_id: str) -> None:
    mapped = sum(a.weight_bytes for a in mine if a.file_backed)
    if not mapped:
        return
    saves, room = _load_saves(peer, mine, model_id), peer.disk_budget or 0
    if saves > room:
        lo, hi = min(n for a in mine for n in a.layers), max(n for a in mine for n in a.layers)
        raise LayoutError("layout_disk", f"{peer.name}'s worker saves ~{fmt_bytes(saves)} of weights into its RPC "
                          f"cache at load (~{fmt_bytes(mapped)} for its file-backed stage to map) and hcpeer lets it "
                          f"save {fmt_bytes(room)} more ({peer.disk_basis}); it copies the rest into RAM",
                          fix=f"seed those layers first (POST /api/peers/<node>/seed with this model_id and layers "
                              f"{lo}-{hi}: a seed evicts least recently used cache files to make room), give it fewer "
                              "layers, free disk on that machine, or rejoin it with a larger --max-disk")


def _check_peers(stages: Sequence[_PeerStage], assigned: Sequence[DeviceAssignment], model_id: str) -> None:
    for st, a in zip(stages, assigned, strict=True):
        if not a.fits:
            what = "dense parts, KV and buffers; file-backed" if a.file_backed else "weights, KV and buffers"
            raise LayoutError("layout_memory", f"{st.label} needs ~{fmt_bytes(a.total_bytes_est)} of RAM for "
                              f"{len(a.layers)} layer(s) ({what}) and allows {fmt_bytes(a.budget_bytes)} "
                              f"({st.cap.budget_basis})", fix="give that stage fewer layers")
    for peer in {st.peer.node_id: st.peer for st in stages}.values():
        mine = [a for a in assigned if a.node_id == peer.node_id]
        ram = sum(a.total_bytes_est for a in mine)
        if ram > peer.node_budget:
            raise LayoutError("layout_memory", f"{peer.name}'s stages need ~{fmt_bytes(ram)} of RAM together and it "
                              f"contributes {fmt_bytes(peer.node_budget)}", fix="give its stages fewer layers")
        _check_disk(peer, mine, model_id)


def _host_side(shape: ModelShape, peers_a: list[DeviceAssignment], gpu: DeviceBudget, cpu: DeviceBudget | None,
               req: PlanRequest, budgets: list[DeviceBudget]) -> tuple[list[DeviceAssignment], MoeSpec | None]:
    """The local GPU's layers and head, all resident when they fit; else, for an MoE model, moe_offload's host side."""
    host = list(range(sum(len(a.layers) for a in peers_a), shape.n_layer))
    whole = _assignment(shape, gpu, host, holds_output=True)
    if whole.fits:
        return [*peers_a, whole], None
    if not shape.is_moe:
        raise LayoutError("layout_memory", f"{gpu.device} cannot hold its {len(host)} layer(s) and the head: "
                          f"~{fmt_bytes(whole.total_bytes_est)} of {fmt_bytes(gpu.budget_bytes)} "
                          f"({gpu.budget_basis})", fix="move layers to the peers")
    if cpu is None or not req.allow_ram_tier:
        why = "turned off (allow_ram_tier)" if cpu is not None else "not measured on this machine"
        raise LayoutError("layout_memory", f"{gpu.device} cannot hold every expert of its {len(host)} layer(s), and "
                          f"the host CPU that would keep the rest is {why}", fix="move layers to the peers")
    layout, spec, ok, why = host_moe_layout(shape, peers_a, gpu, cpu, budgets, allow_disk=req.allow_capacity_mode)
    if not ok:
        raise LayoutError("layout_memory", f"this machine's side does not fit: {why}", fix="move layers to the peers")
    return layout, spec


# --------------------------------------------------------------------------- plan


def _notes(stages: Sequence[_PeerStage], assigned: Sequence[DeviceAssignment]) -> list[str]:
    return [f"{st.label}: {len(a.layers)} layer(s) file-backed by hcpeer, ~{fmt_bytes(a.weight_bytes)} mapped from its "
            f"SSD (hcpeer may save {fmt_bytes(st.peer.disk_budget or 0)} more: {st.peer.disk_basis}) and ~"
            f"{fmt_bytes(a.total_bytes_est)} in its RAM. Pages its file cache cannot hold are read from its SSD on "
            "every token; the decode estimate prices this stage as CPU compute only (estimates, not measurements)"
            for st, a in zip(stages, assigned, strict=True) if a.file_backed]


def _finish(req: PlanRequest, shape: ModelShape, layout: list[DeviceAssignment], moe: MoeSpec | None,
            stages: Sequence[_PeerStage], gpu: DeviceBudget, host_ram_free: int | None) -> Plan:
    peers_a = [a for a in layout if a.endpoint]
    chain = [f"{st.label} {st.layers}" + (" file-backed" if st.file_backed else "") for st in stages]
    chain.append(f"{gpu.device} {shape.n_layer - sum(st.layers for st in stages)} + head")
    reasons = [f"explicit layout as requested ({', '.join(chain)}); checked against the memory free now, not chosen "
               "by the planner"]
    candidate = CandidateSummary(label="explicit layout", devices=[f"{a.node_id}:{a.device}" for a in layout],
                                 feasible=True, reason="as requested; checked against the budgets free now (estimates)")
    nodes = {st.peer.node_id: st.peer.node_budget for st in stages}
    plan = finish_plan(req, shape, layout, True, paged=False, reasons=reasons, warnings=_notes(stages, peers_a),
                       candidates=[candidate], host_ram_free=host_ram_free,
                       total_budget=sum(nodes.values()) + gpu.budget_bytes, moe=moe)
    capacity = plan.capacity_mode or any(a.file_backed for a in layout)
    return plan.model_copy(update={"layout": list(req.layout or []), "capacity_mode": capacity})


def layout_plan(req: PlanRequest, shape: ModelShape | None, local: Sequence[DeviceBudget], cpu: DeviceBudget | None,
                peers: Mapping[str, PeerFacts], host_ram_free: int | None) -> Plan:
    """The plan for req.layout, checked against the budgets free now; LayoutError says what is wrong with it. Peer
    devices keep the planner's provisional names (<endpoint>#<index>) for services.deployments.assign_rpc_names."""
    stages = list(req.layout or [])
    if not stages:
        raise LayoutError("layout_empty", "an explicit layout needs at least one stage")
    if req.engine not in (None, "llama_rpc"):
        raise LayoutError("layout_engine", f"an explicit layout is a llama.cpp split; this request forces {req.engine}")
    if shape is None:
        raise LayoutError("layout_needs_gguf", "an explicit layout needs the model's GGUF metadata (a downloaded "
                          "GGUF artifact)", fix="download the model through this gateway first")
    _check_chain(shape, stages)
    gpu = _local_gpu(stages[-1], local)
    peer_stages = _peer_stages(stages[:-1], peers, req.allow_capacity_mode)
    peers_a = _peer_assignments(shape, peer_stages)
    _check_peers(peer_stages, peers_a, req.model_id)
    budgets = [b for b in (cpu, *(st.cap for st in peer_stages), gpu) if b is not None]
    layout, moe = _host_side(shape, peers_a, gpu, cpu, req, budgets)
    return _finish(req, shape, layout, moe, peer_stages, gpu, host_ram_free)
