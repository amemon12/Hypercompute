"""Bridge between the gateway's live state and the capacity optimizer, in both directions.

Inputs. The local NodeView and the eligible remote NodeViews, the active leases and a fresh host memory snapshot become
pools (`pools.pools_report`) and admissible bytes (`pools.admissible_bytes`). Device, link and storage profiles are the
documented priors of `costs` merged (`costs.merge`) with fresh measurements of the same identity from the
MeasurementStore. The host is HostSide (its CPU on "ram" or "uma", one local GPU on its own pool); every contributed
device of a verified peer is a PeerSide reached over llama.cpp RPC, priced by the path class the transport reports.

Outputs. An optimizer PlacementCandidate becomes a Plan the llama.cpp adapter runs as it is: the same device names,
order, tensor split, -ngl, RPC endpoints and stream layout services.planner would write for that layout (the Plan is
built by `planner.finish_plan`), plus the prediction and every prior it used. A planner Plan can be translated back
into optimizer spans, so both plans are scored by one set of formulas.

Nothing here starts an engine, writes a file or mutates its inputs.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schemas import (
    CandidateSummary,
    DeviceAssignment,
    ModelArtifact,
    NodeView,
    Plan,
    PlanLimits,
    PlanPrediction,
    PlanRequest,
    StagePrediction,
    StreamSpec,
)
from ..services import planner
from ..services.planner import StreamCapability
from ..units import MiB, fmt_bytes
from . import costs, pools, runtime, storage_probe
from .manifest import build_manifest
from .measurements import MeasurementRecord, MeasurementStore, to_coefficient
from .optimizer import MAX_PEERS, RISK_PATHS, HostSide, OptimizeResult, PeerSide
from .types import (
    Coefficient,
    DeviceProfile,
    LinkProfile,
    ModelManifest,
    PathClass,
    PlacementCandidate,
    PoolLease,
    StagePlan,
    StorageProfile,
    Workload,
)

log = logging.getLogger("hypercompute.capacity.bridge")

# Freshness of measurements. Compute coefficients drift with drivers, builds and thermals over days; a network path
# changes within the hour (Wi-Fi, a relay replacing a direct route); storage rates follow runtime's calibration age.
COMPUTE_MAX_AGE_S = 7 * 24 * 3600.0
LINK_MAX_AGE_S = 3600.0
STORAGE_MAX_AGE_S = float(runtime.CALIBRATION_MAX_AGE_S)

# PlanRequest has no prompt or output length. These are the documented defaults plans are scored for: a chat turn.
DEFAULT_PROMPT_TOKENS = 512
DEFAULT_OUTPUT_TOKENS = 256
HOST_SAFETY_BYTES = 512 * MiB  # services.deployments.cpu_budget keeps this free on top of the owner's reserve

# MeasurementIdentity.kind values the bridge reads. Link kinds are the ones scripts/capacity/probe_link.py writes.
KIND_DECODE = "compute.decode_ms_per_gib"
KIND_PREFILL = "compute.prefill_ms_per_gib_token"
KIND_RTT = "link.rtt_ms"
KIND_GOODPUT = "link.goodput_bytes_per_s"
KIND_ROUND_TRIPS = "link.round_trips_per_token"
KIND_STORAGE_READ = "storage.read_bytes_per_s"
KIND_STREAM_EFFICIENCY = "storage.stream_efficiency"
_VOLUME_RE = re.compile(r"[A-Za-z]:|/.*")  # a storage record's device naming a volume (storage_probe.volume_of)

HOST_CPU = "CPU"
LINK_PATHS: frozenset[str] = frozenset({"loopback", "lan", "direct", "relay", "unknown"})
BACKENDS: frozenset[str] = frozenset({"cpu", "cuda", "metal"})  # backends with cost priors (costs.PREFILL_FRACTION)
_RPC_NAME = re.compile(r"RPC(\d+)")

Span = tuple[str, int, int]  # (node:device, first unit, end unit), the optimizer's layout element


class BridgeError(Exception):
    """The live state cannot be turned into optimizer inputs; `code` is the CapacityUnavailable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------------------------------- probes


def disk_free(path: Path) -> int | None:
    """Free bytes on the volume holding `path` (its nearest existing parent); None when the OS cannot say."""
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    try:
        return int(shutil.disk_usage(target).free)
    except OSError as exc:
        log.warning("free disk space of %s unavailable: %s", target, exc)
        return None


def read_manifest(artifact: ModelArtifact, shards: list[Path]) -> ModelManifest:
    """Manifest of a downloaded GGUF. ModelService verified every file's sha256 before marking it downloaded."""
    names = {p.name for p in shards}
    hashes = {f.name: f.sha256 for f in artifact.files if f.name in names}
    return build_manifest(artifact.id, shards, revision=artifact.revision, shard_hashes=hashes)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Probes:
    """The OS and file reads the bridge makes. Tests replace them; the defaults read this machine."""

    memory: Callable[[], pools.HostMemory] = pools.host_memory_snapshot
    disk_free: Callable[[Path], int | None] = disk_free
    manifest: Callable[[ModelArtifact, list[Path]], ModelManifest] = read_manifest
    stream: Callable[[list[Path]], StreamCapability | None] = runtime.stream_capability
    clock: Callable[[], datetime] = _utcnow


# --------------------------------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class PeerRef:
    """Where an optimizer peer device lives in llama.cpp terms."""

    key: str  # node:device, the optimizer's name
    node_id: str
    device: str  # engine device name the peer contributed (MTL0, CPU, CUDA0)
    endpoint: str  # host:port of the node's RPC worker
    index: int  # the device's position in the worker's RPC devices (services.deployments.remote_budgets numbering)
    endpoint_devices: int  # devices the worker exposes (services.deployments.rpc_device_counts)


@dataclass(frozen=True)
class Snapshot:
    """Pools, leases and admissible bytes observed together."""

    observed_at: datetime
    host_memory: pools.HostMemory
    report: pools.PoolsReport
    leases: tuple[PoolLease, ...]
    admissible: Mapping[tuple[str, str], int]
    released: str | None = None  # deployment whose leases count as released (a replacement stops it first)


@dataclass(frozen=True)
class BridgeInputs:
    manifest: ModelManifest
    workload: Workload
    host: HostSide
    peers: tuple[PeerSide, ...]
    refs: tuple[PeerRef, ...]  # aligned with peers
    local_gpu: str | None  # engine name of host.gpu
    stream: StreamCapability | None
    snapshot: Snapshot
    slots: int
    notes: tuple[str, ...] = ()  # assumptions the bridge made (they go into the prediction)
    warnings: tuple[str, ...] = ()  # devices left out of the search, for the plan's warnings
    measured: tuple[str, ...] = ()  # coefficients that came from measurements
    left_out: tuple[str, ...] = ()  # node:device of contributed peer devices the search does not place
    local_left_out: tuple[str, ...] = ()  # contributed local accelerators the search does not place

    def profile(self, key: str) -> DeviceProfile:
        if key == f"{self.host.node_id}:{self.host.cpu.device}":
            return self.host.cpu
        if self.host.gpu is not None and key == f"{self.host.node_id}:{self.host.gpu.device}":
            return self.host.gpu
        for p in self.peers:
            if key == f"{p.node_id}:{p.device.device}":
                return p.device
        raise KeyError(key)

    def admissible(self, key: str) -> int:
        if key == f"{self.host.node_id}:{self.host.cpu.device}":
            return self.host.cpu_pool_admissible
        if self.host.gpu is not None and key == f"{self.host.node_id}:{self.host.gpu.device}":
            return self.host.gpu_pool_admissible
        return next(p.pool_admissible for p in self.peers if key == f"{p.node_id}:{p.device.device}")

    def ref(self, key: str) -> PeerRef | None:
        return next((r for r in self.refs if r.key == key), None)

    @property
    def storage(self) -> StorageProfile:
        return self.host.storage or costs.storage_prior(self.host.node_id)


@dataclass(frozen=True)
class Request:
    """What build_inputs needs besides the live state."""

    workload: Workload
    slots: int = 2
    allow_cpu: bool = True  # False: the host CPU may hold no units (allow_ram_tier off)
    allow_stream: bool = True  # False: no bounded streaming (allow_capacity_mode off)
    releasing: str | None = None


def with_fresh_memory(node: NodeView, mem: pools.HostMemory) -> NodeView:
    """The node with its RAM or unified pool's free bytes replaced by a fresh snapshot (when the OS reported one)."""
    if mem.available is None:
        return node
    fresh = [p.model_copy(update={"free_bytes": mem.available, "measured_at": mem.observed_at,
                                  "source": f"{mem.source} (fresh snapshot)"}) if p.id in ("ram", "uma") else p
             for p in node.inventory.memory_pools]
    return node.model_copy(update={"inventory": node.inventory.model_copy(update={"memory_pools": fresh})})


def pool_leases(rows: Iterable[Mapping[str, Any]], ready: Callable[[str], bool]) -> tuple[PoolLease, ...]:
    """LeaseManager.active() rows as PoolLeases. A lease is materialized once its deployment is ready."""
    out = []
    for row in rows:
        try:
            dep = str(row["deployment_id"])
            out.append(PoolLease(lease_id=str(row["id"]), deployment_id=dep, revision=int(row.get("revision") or 0),
                                 node_id=str(row["node_id"]), pool_id=str(row["pool_id"]), bytes=int(row["bytes"]),
                                 materialized=ready(dep)))
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeError("lease_row", f"malformed lease row {dict(row)!r}: {exc}") from exc
    return tuple(out)


def _release(report: pools.PoolsReport, leases: Sequence[PoolLease], releasing: str | None) -> pools.PoolsReport:
    """Pools as they will be once `releasing` has stopped: its loaded bytes return to the free observation.

    The lease bytes are the plan's estimate of what the engine holds, not a measurement of it.
    """
    if releasing is None:
        return report
    back: dict[tuple[str, str], int] = {}
    for lease in leases:
        if lease.deployment_id == releasing and lease.materialized:
            key = (lease.node_id, lease.pool_id)
            back[key] = back.get(key, 0) + lease.bytes
    if not back:
        return report
    fresh = tuple(replace(p, free_bytes=p.free_bytes + back.get((p.node_id, p.pool_id), 0)) for p in report.pools)
    return replace(report, pools=fresh)


def take_snapshot(local: NodeView, remote: Sequence[NodeView], lease_rows: Iterable[Mapping[str, Any]],
                  ready: Callable[[str], bool], probes: Probes, storage_path: Path,
                  releasing: str | None = None) -> Snapshot:
    """Fresh pools of the host and the given peers with the admissible bytes left by the active leases."""
    now = probes.clock()
    mem = probes.memory()
    leases = pool_leases(lease_rows, ready)
    report = pools.pools_report([with_fresh_memory(local, mem), *remote],
                                host_reserve_bytes=local.contribution.reserve_bytes + HOST_SAFETY_BYTES,
                                storage_free_bytes=probes.disk_free(storage_path), now=now)
    report = _release(report, leases, releasing)
    kept = tuple(lease for lease in leases if lease.deployment_id != releasing)
    admissible = {(p.node_id, p.pool_id): pools.admissible_bytes(p, kept) for p in report.pools}
    return Snapshot(observed_at=now, host_memory=mem, report=report, leases=kept, admissible=admissible,
                    released=releasing)


# --------------------------------------------------------------------------------------------------- profiles


def load_records(store: MeasurementStore) -> tuple[tuple[MeasurementRecord, ...], str | None]:
    """Every stored measurement, or none with the reason when the store cannot be read (priors still apply)."""
    try:
        return store.records(), None
    except OSError as exc:
        log.warning("measurement store %s unreadable", store.path, exc_info=True)
        return (), f"measurement store {store.path} unreadable ({exc}); every coefficient is a prior"


def _freshest(records: Iterable[MeasurementRecord], now: datetime, max_age_s: float,
              match: Callable[[MeasurementRecord], bool],
              rank: Callable[[MeasurementRecord], object] = lambda r: 0) -> MeasurementRecord | None:
    fresh = [r for r in records if match(r) and 0 <= (now - r.measured_at).total_seconds() <= max_age_s]
    return max(fresh, key=lambda r: (rank(r), r.measured_at), default=None)


def _labelled(rec: MeasurementRecord, scope: str = "") -> Coefficient:
    """The record as a coefficient whose note says what it was measured on. Matching ignores the build, context and
    power state (the store has no records that differ only in those yet), so the label carries them."""
    i = rec.identity
    parts = [rec.note, scope, f"{i.device}, build {i.backend_build}", f"ctx {i.ctx}" if i.ctx else "",
             f"power {i.power_state}" if i.power_state else "", f"evidence {rec.evidence}" if rec.evidence else ""]
    return replace(to_coefficient(rec), note="; ".join(p for p in parts if p))


def _compute_record(records: Sequence[MeasurementRecord], node_id: str, device: str, kind: str, fingerprint: str,
                    now: datetime) -> Coefficient | None:
    """A fresh compute coefficient for the device: one taken on this model artifact first, else a model-agnostic one."""
    rec = _freshest(records, now, COMPUTE_MAX_AGE_S,
                    lambda r: (r.identity.node_id, r.identity.device, r.identity.kind) == (node_id, device, kind)
                    and r.identity.model_fingerprint in (fingerprint, None),
                    rank=lambda r: r.identity.model_fingerprint == fingerprint)
    if rec is None:
        return None
    return _labelled(rec, "on this model" if rec.identity.model_fingerprint == fingerprint else "model-agnostic")


def device_profile(backend: str, node_id: str, device: str, pool_id: str, link: LinkProfile | None,
                   records: Sequence[MeasurementRecord], manifest: ModelManifest, now: datetime) -> DeviceProfile:
    """Prior device profile with decode and prefill replaced by fresh measurements of the same device."""
    prior = costs.device_prior(backend, node_id, device, pool_id, link)
    decode = _compute_record(records, node_id, device, KIND_DECODE, manifest.fingerprint, now)
    prefill = _compute_record(records, node_id, device, KIND_PREFILL, manifest.fingerprint, now)
    if decode is None and prefill is None:
        return prior
    measured = replace(prior, decode_ms_per_gib=decode or prior.decode_ms_per_gib,
                       prefill_ms_per_gib_token=prefill or prior.prefill_ms_per_gib_token)
    return costs.merge(prior, measured, now=now, max_age_s=COMPUTE_MAX_AGE_S)


def link_profile(records: Sequence[MeasurementRecord], node_id: str, path: PathClass, now: datetime) -> LinkProfile:
    """The path-class prior merged with fresh link probes of this node over the same path class (any device)."""
    prior = costs.link_prior(path)

    def measured(kind: str) -> Coefficient | None:
        rec = _freshest(records, now, LINK_MAX_AGE_S, lambda r: r.identity.node_id == node_id
                        and r.identity.kind == kind and r.identity.path_class == path)
        return _labelled(rec, f"{path} path") if rec else None

    rtt, goodput, trips = measured(KIND_RTT), measured(KIND_GOODPUT), measured(KIND_ROUND_TRIPS)
    if rtt is None and goodput is None and trips is None:
        return prior
    probe = replace(prior, rtt_ms=rtt or prior.rtt_ms, goodput_bytes_per_s=goodput or prior.goodput_bytes_per_s,
                    round_trips_per_token=trips or prior.round_trips_per_token)
    return costs.merge(prior, probe, now=now, max_age_s=LINK_MAX_AGE_S)


def _on_volume(device: str, volume: str | None) -> bool:
    """Whether a storage record whose device is `device` describes `volume`: the same volume, or a device that names
    no volume (scripts/capacity/calibrate.py writes the volume, "C:"; a record for another volume does not count)."""
    return volume is None or not _VOLUME_RE.fullmatch(device) or device.upper() == volume.upper()


def storage_profile(records: Sequence[MeasurementRecord], node_id: str, stream: StreamCapability | None,
                    now: datetime, *, volume: str | None = None) -> StorageProfile:
    """Storage read rate and stream efficiency: fresh store records first (of `volume`, the one holding the model,
    when given), then runtime's measured volume calibration (the rate the stream tier itself uses), then the priors."""
    prior = costs.storage_prior(node_id)

    def measured(kind: str) -> Coefficient | None:
        rec = _freshest(records, now, STORAGE_MAX_AGE_S,
                        lambda r: r.identity.node_id == node_id and r.identity.kind == kind
                        and _on_volume(r.identity.device, volume),
                        rank=lambda r: volume is not None and r.identity.device.upper() == volume.upper())
        return _labelled(rec) if rec else None

    read, efficiency = measured(KIND_STORAGE_READ), measured(KIND_STREAM_EFFICIENCY)
    merged = prior
    if read is not None or efficiency is not None:
        merged = costs.merge(prior, replace(prior, read_bytes_per_s=read or prior.read_bytes_per_s,
                                            stream_efficiency=efficiency or prior.stream_efficiency),
                             now=now, max_age_s=STORAGE_MAX_AGE_S)
    if read is None and stream is not None and stream.read_provenance == "measured":
        calibrated = Coefficient(value=float(stream.read_bytes_per_s), unit="bytes/s", provenance="measured",
                                 note="unbuffered read calibration of the model's volume (runtime.storage_rate, at "
                                      "most 7 days old)")
        merged = replace(merged, read_bytes_per_s=calibrated)
    return merged


def coefficient_labels(inputs: BridgeInputs) -> list[tuple[str, Coefficient]]:
    """Every coefficient of every device the search may use, labelled node:device name."""
    out: list[tuple[str, Coefficient]] = []
    profiles = [inputs.host.cpu, *([inputs.host.gpu] if inputs.host.gpu else []), *(p.device for p in inputs.peers)]
    for p in profiles:
        key = f"{p.node_id}:{p.device}"
        out += [(f"{key} decode", p.decode_ms_per_gib), (f"{key} prefill", p.prefill_ms_per_gib_token)]
        if p.link is not None:
            out += [(f"{key} rtt", p.link.rtt_ms), (f"{key} goodput", p.link.goodput_bytes_per_s),
                    (f"{key} round trips per token", p.link.round_trips_per_token)]
    storage = inputs.storage
    return out + [("storage read", storage.read_bytes_per_s), ("stream efficiency", storage.stream_efficiency)]


def _measured_text(label: str, c: Coefficient) -> str:
    when = c.measured_at.isoformat() if c.measured_at else "date not recorded"
    return f"{label} {c.value:.4g} {c.unit} (measured {when}, {c.samples} samples){': ' + c.note if c.note else ''}"


# --------------------------------------------------------------------------------------------------- host and peers


def _local_gpu(local: NodeView) -> tuple[str | None, list[tuple[str, str]]]:
    """The one local GPU the optimizer places (it models a single GPU suffix): CUDA0 when contributed, as the planner
    orders it last and gives it the head; else the first contributed CUDA or Metal device. Also (name, reason) for
    every other contributed local accelerator, which the search does not place."""
    accel = [d.name for d in local.inventory.engine.devices
             if d.name.startswith(("CUDA", "MTL", "Vulkan", "ROCm")) and d.name in local.contribution.devices]
    usable = [n for n in accel if n.startswith(("CUDA", "MTL"))]
    left = [(n, "no cost prior for its backend") for n in accel if n not in usable]
    if not usable:
        return None, left
    chosen = "CUDA0" if "CUDA0" in usable else usable[0]
    left += [(n, f"the optimizer places one local GPU suffix and uses {chosen}, the device the planner orders last "
                 "with the output head") for n in usable if n != chosen]
    return chosen, left


def _host_gpu(local: NodeView, name: str, snap: Snapshot, records: Sequence[MeasurementRecord],
              manifest: ModelManifest, cpu_pool_id: str) -> tuple[DeviceProfile | None, int, str | None]:
    """The local GPU's profile and admissible bytes; or why it is left out.

    The figure is capped at the owner's contribution limit, as services.deployments.local_budgets caps it: lease
    admission holds this node's leases outside its CPU pool to that limit (lease_budgets), so leases already held there
    come off the cap.
    """
    inv = next((d for d in local.inventory.devices if d.id == name), None)
    pool_id = inv.pool_id if inv else f"vram:{name}"
    backend = inv.backend if inv and inv.backend in BACKENDS else ("cuda" if name.startswith("CUDA") else "metal")
    if (local.id, pool_id) not in snap.admissible:
        return None, 0, f"its pool {pool_id} has no free-memory observation"
    profile = device_profile(backend, local.id, name, pool_id, None, records, manifest, snap.observed_at)
    held = sum(lease.bytes for lease in snap.leases if lease.node_id == local.id and lease.pool_id != cpu_pool_id)
    return profile, min(snap.admissible[(local.id, pool_id)], max(0, local.contribution.max_bytes - held)), None


@dataclass(frozen=True)
class HostBuild:
    host: HostSide
    gpu_name: str | None  # engine name of host.gpu
    notes: tuple[str, ...]  # assumptions
    left_out: tuple[tuple[str, str], ...]  # (local device, reason) of contributed accelerators not placed


def host_side(local: NodeView, snap: Snapshot, manifest: ModelManifest, records: Sequence[MeasurementRecord],
              storage: StorageProfile, req: Request, stream_ok: bool) -> HostBuild:
    """HostSide, the local GPU's engine name, notes (assumptions) and the local accelerators left out.

    The host CPU's admissible bytes exclude the token embeddings, as services.deployments.cpu_budget plus the planner's
    _cpu_side do; on a unified-memory host the CPU and GPU then share the smaller figure (optimizer.HostSide). That
    sharing cannot express "the CPU holds nothing" (the RAM tier off): a zero CPU figure would starve the GPU, so such a
    request is left to the planner.
    """
    cpu_pool = next((p for p in snap.report.pools if p.node_id == local.id and p.pool_id in ("ram", "uma")), None)
    if cpu_pool is None:
        raise BridgeError("host_memory", "this machine's RAM pool has no free-memory observation")
    embd = manifest.token_embd_bytes
    cpu_adm = max(0, snap.admissible[(local.id, cpu_pool.pool_id)] - embd) if req.allow_cpu else 0
    notes = [f"host CPU admissible reduced by the token embeddings ({fmt_bytes(embd)}), which llama.cpp keeps in host "
             "memory on every tier (the planner charges them the same way)"] if embd and req.allow_cpu else []
    if not req.allow_cpu:
        notes.append("the host CPU holds no units: this request turns the host RAM tier off")
    cpu = device_profile("cpu", local.id, HOST_CPU, cpu_pool.pool_id, None, records, manifest, snap.observed_at)
    name, left = _local_gpu(local)
    gpu, gpu_adm = None, 0
    if name is not None:
        gpu, gpu_adm, why = _host_gpu(local, name, snap, records, manifest, cpu_pool.pool_id)
        left += [(name, why)] if why else []
        name = name if gpu is not None else None
    shared = gpu is not None and gpu.pool_id == cpu_pool.pool_id
    if shared and not req.allow_cpu:
        raise BridgeError("ram_tier_off_unified", f"this machine's CPU and {name} share the {cpu_pool.pool_id} pool "
                          "and the request turns the host RAM tier off; the optimizer cannot keep the CPU empty there")
    if shared and stream_ok:  # as the planner (services.planner.UNIFIED_NO_STREAM): no host part run from the SSD
        notes.append(f"bounded streaming not considered: this machine's CPU and {name} share the unified "
                     f"{cpu_pool.pool_id} pool, and a host part larger than it is not run from the SSD there")
    host = HostSide(node_id=local.id, cpu=cpu, cpu_pool_admissible=cpu_adm, storage=storage, gpu=gpu,
                    gpu_pool_admissible=gpu_adm, stream_supported=stream_ok and not shared)
    return HostBuild(host=host, gpu_name=name, notes=tuple(notes), left_out=tuple(left))


def _link_path(path_of: Callable[[str], PathClass], node_id: str) -> tuple[PathClass, str | None]:
    try:
        path = path_of(node_id)
    except Exception as exc:  # noqa: BLE001 - a transport that cannot report a route prices it as unknown
        log.warning("path lookup for node %s failed", node_id, exc_info=True)
        return "unknown", f"{node_id}: path class lookup failed ({exc}); priced as unknown (relay prior)"
    if path not in LINK_PATHS:
        return "unknown", f"{node_id}: path class {path!r} is not a network path; priced as unknown (relay prior)"
    return path, None


def _peer_devices(node: NodeView, snap: Snapshot, records: Sequence[MeasurementRecord], manifest: ModelManifest,
                  link: LinkProfile) -> tuple[list[tuple[PeerSide, PeerRef]], list[tuple[str, str]]]:
    """PeerSides of one node's contributed devices, and (node:device, reason) for each device left out."""
    if node.worker is None or node.verification is None:
        return [], [(f"{node.id}:{name}", "its RPC worker is not verified") for name in node.contribution.devices]
    endpoint = f"{node.worker.host}:{node.worker.port}"
    rpc = node.verification.rpc_devices
    out: list[tuple[PeerSide, PeerRef]] = []
    skipped: list[tuple[str, str]] = []
    for i, name in enumerate(node.contribution.devices):
        inv = next((d for d in node.inventory.devices if d.id == name), None)
        backend = inv.backend if inv else "unknown"
        if i >= len(rpc) or inv is None or backend not in BACKENDS:
            why = "no RPC device for it in the verification" if i >= len(rpc) else (
                "not in the inventory" if inv is None else f"no cost prior for backend {backend}")
            skipped.append((f"{node.id}:{name}", why))
            continue
        if (node.id, inv.pool_id) not in snap.admissible:
            skipped.append((f"{node.id}:{name}", f"pool {inv.pool_id} is not admissible (see the pool report)"))
            continue
        admissible = snap.admissible[(node.id, inv.pool_id)]
        if rpc[i].free_bytes is not None:  # what the device itself reported at verification also bounds it
            admissible = min(admissible, max(0, rpc[i].free_bytes))
        device = device_profile(backend, node.id, name, inv.pool_id, link, records, manifest, snap.observed_at)
        out.append((PeerSide(node_id=node.id, device=device, pool_admissible=admissible),
                    PeerRef(key=f"{node.id}:{name}", node_id=node.id, device=name, endpoint=endpoint, index=i,
                            endpoint_devices=max(1, len(rpc)))))
    return out, skipped


@dataclass(frozen=True)
class PeerSet:
    peers: tuple[PeerSide, ...]
    refs: tuple[PeerRef, ...]  # aligned with peers
    left_out: tuple[tuple[str, str], ...]  # (node:device, reason) of contributed devices the search does not place
    notes: tuple[str, ...]


def peer_sides(remote: Sequence[NodeView], snap: Snapshot, records: Sequence[MeasurementRecord],
               manifest: ModelManifest, path_of: Callable[[str], PathClass]) -> PeerSet:
    """Every contributed device of every verified peer, at most MAX_PEERS (largest admissible first)."""
    found: list[tuple[PeerSide, PeerRef]] = []
    notes: list[str] = []
    left_out: list[tuple[str, str]] = []
    for node in remote:
        path, why = _link_path(path_of, node.id)
        notes += [why] if why else []
        devices, skipped = _peer_devices(node, snap, records, manifest,
                                         link_profile(records, node.id, path, snap.observed_at))
        found += devices
        left_out += skipped
    found.sort(key=lambda pr: -pr[0].pool_admissible)
    left_out += [(ref.key, f"the exact search takes at most {MAX_PEERS} peer devices and this one has the least "
                  "admissible memory") for _, ref in found[MAX_PEERS:]]
    if found:
        notes.append("peer upload caches are assumed empty: a cold load uploads every weight byte a peer holds")
    kept = found[:MAX_PEERS]
    return PeerSet(peers=tuple(p for p, _ in kept), refs=tuple(r for _, r in kept), left_out=tuple(left_out),
                   notes=tuple(notes))


def build_inputs(manifest: ModelManifest, req: Request, local: NodeView, remote: Sequence[NodeView],
                 snap: Snapshot, store: MeasurementStore, path_of: Callable[[str], PathClass], probes: Probes,
                 shards: list[Path]) -> BridgeInputs:
    """Optimizer inputs for one model from one snapshot of the cluster."""
    records, store_note = load_records(store)
    stream = probes.stream(shards) if req.allow_stream else None
    storage = storage_profile(records, local.id, stream, snap.observed_at,
                              volume=storage_probe.volume_of(shards[0]) if shards else None)
    built = host_side(local, snap, manifest, records, storage, req, stream is not None)
    peer_set = peer_sides(remote, snap, records, manifest, path_of)
    notes = [*([store_note] if store_note else []), *built.notes, *peer_set.notes, *snap.report.notes]
    if not req.allow_stream:
        notes.append("bounded streaming not considered: this request turns capacity mode off")
    if snap.host_memory.error:
        notes.append(f"host memory snapshot: {snap.host_memory.error}")
    warnings = [*(f"local {name} not placed: {why}" for name, why in built.left_out),
                *(f"{key} not placed: {why}" for key, why in peer_set.left_out)]
    inputs = BridgeInputs(manifest=manifest, workload=req.workload, host=built.host, peers=peer_set.peers,
                          refs=peer_set.refs, local_gpu=built.gpu_name, stream=stream, snapshot=snap, slots=req.slots,
                          notes=tuple(notes), warnings=tuple(warnings),
                          left_out=tuple(key for key, _ in peer_set.left_out),
                          local_left_out=tuple(name for name, _ in built.left_out))
    measured = tuple(_measured_text(label, c) for label, c in coefficient_labels(inputs) if c.provenance == "measured")
    return replace(inputs, measured=tuple(dict.fromkeys(measured)))


# --------------------------------------------------------------------------------------------------- plans


def layout(c: PlacementCandidate) -> list[Span]:
    return [(f"{s.node_id}:{s.device}", s.units[0], s.units[-1] + 1) for s in c.stages]


def same_layout(plan: Plan, spans: Sequence[Span], c: PlacementCandidate) -> bool:
    """Whether `c` runs what `plan` runs: the same device spans (`spans` is plan_layout(plan), empty when the plan has
    none here) and the same host CPU units streamed from the SSD. Same spans with another residency is another plan."""
    running = frozenset(plan.stream.streamed_units) if plan.stream is not None else frozenset()
    streamed = frozenset(u for s in c.stages for u in s.streamed_units)
    return bool(spans) and list(spans) == layout(c) and running == streamed


def span_text(spans: Sequence[Span], head: int) -> str:
    def units(a: int, b: int) -> str:
        blocks = f"{a}-{min(b, head) - 1}" if a < min(b, head) else ""
        return (blocks + " + head" if blocks else "head") if b > head else blocks

    return ", ".join(f"{key} [{units(a, b)}]" for key, a, b in spans)


def stage_coefficients(c: PlacementCandidate, inputs: BridgeInputs) -> list[tuple[str, Coefficient]]:
    """The coefficients a candidate's score used, the way optimizer._build and _candidate read them: per stage the
    compute, link and storage terms, the relay risk margin on relayed or unknown paths, and the setup terms a cold load
    adds (no measurement kind exists for those yet, so a cold score always rests on them)."""
    w, storage = inputs.workload, inputs.storage
    out: dict[str, Coefficient] = {}
    for s in c.stages:
        key = f"{s.node_id}:{s.device}"
        p = inputs.profile(key)
        out[f"{key} decode"] = p.decode_ms_per_gib
        if w.prompt_tokens:
            out[f"{key} prefill"] = p.prefill_ms_per_gib_token
        if p.link is not None and inputs.ref(key) is not None:
            out |= {f"{key} rtt": p.link.rtt_ms, f"{key} goodput": p.link.goodput_bytes_per_s,
                    f"{key} round trips per token": p.link.round_trips_per_token}
            if s.path in RISK_PATHS and w.output_tokens:
                out[f"{key} relay risk"] = costs.relay_risk_prior()
            if not w.warm:
                out["setup per peer stage"] = costs.setup_per_remote_prior()
        elif s.streamed_units or not w.warm:
            out["storage read"] = storage.read_bytes_per_s
        if s.streamed_units:
            out["stream efficiency"] = storage.stream_efficiency
    if not w.warm:
        out["setup"] = costs.setup_prior()
    return list(out.items())


def confidence(c: PlacementCandidate, inputs: BridgeInputs) -> float:
    """Fraction of the coefficients behind the candidate's score that were measured on this hardware."""
    used = stage_coefficients(c, inputs)
    return sum(1 for _, k in used if k.provenance == "measured") / len(used) if used else 0.0


def priors_warning(c: PlacementCandidate, inputs: BridgeInputs) -> str | None:
    priors = [f"{label} {k.value:.4g} {k.unit} ({k.provenance})" for label, k in stage_coefficients(c, inputs)
              if k.provenance != "measured"]
    return ("predicted with priors, not measurements on this hardware: " + "; ".join(priors)) if priors else None


def model_shape(manifest: ModelManifest, ctx_size: int) -> planner.ModelShape:
    """The planner's view of the manifest, for planner.finish_plan's warnings and token-embedding figure."""
    blocks = manifest.units[: manifest.n_layer]
    moe = manifest.is_moe  # a dense manifest keeps the dense shape (no expert split)
    return planner.ModelShape(n_layer=manifest.n_layer, layer_bytes=tuple(u.weight_bytes for u in blocks),
                              output_bytes=manifest.head.weight_bytes, token_embd_bytes=manifest.token_embd_bytes,
                              kv_per_layer=manifest.kv_bytes_per_layer(ctx_size),
                              compute_est=costs.workspace_bytes(manifest.n_embd),
                              expert_bytes=tuple(u.expert_bytes for u in blocks) if moe else (),
                              expert_count=manifest.expert_count if moe else 0,
                              expert_used_count=manifest.expert_used_count if moe else 0)


def _device_name(s: StagePlan, inputs: BridgeInputs) -> tuple[str, str | None]:
    """(device name as the planner writes it before RPC naming, RPC endpoint or None)."""
    ref = inputs.ref(f"{s.node_id}:{s.device}")
    if ref is not None:
        return f"{ref.endpoint}#{ref.index}", ref.endpoint
    return s.device, None


def assignment(s: StagePlan, inputs: BridgeInputs) -> DeviceAssignment:
    """One stage as a DeviceAssignment. weight_bytes keeps the planner's meaning (all weights the device computes);
    total_bytes_est is the stage's PoolDemand peak, which is what gets leased.

    budget_bytes keeps the planner's meaning too: the pool's figure before leases. Lease admission (LeaseManager)
    subtracts every active lease on the pool from it, so it is the admissible bytes plus the leases the snapshot
    counted; a budget of only the admissible bytes would subtract those leases twice. `fits` compares with the
    admissible bytes, the optimizer's own check.
    """
    m = inputs.manifest
    d = s.demand
    if d is None:
        raise BridgeError("candidate", f"stage {s.node_id}:{s.device} carries no memory demand")
    name, endpoint = _device_name(s, inputs)
    admissible = inputs.admissible(f"{s.node_id}:{s.device}")
    held = sum(lease.bytes for lease in inputs.snapshot.leases
               if (lease.node_id, lease.pool_id) == (s.node_id, s.pool_id))
    return DeviceAssignment(node_id=s.node_id, device=name, endpoint=endpoint, pool_id=s.pool_id,
                            layers=[u for u in s.units if u != m.n_layer], holds_output=m.n_layer in s.units,
                            weight_bytes=sum(m.units[u].weight_bytes for u in s.units), kv_bytes=d.kv,
                            compute_bytes_est=d.workspace, overhead_bytes_est=d.overhead + d.staging + d.peak_extra,
                            total_bytes_est=d.peak, budget_bytes=admissible + held, fits=d.peak <= admissible)


def stream_spec(s: StagePlan, inputs: BridgeInputs) -> StreamSpec | None:
    """The hcstream layout of a host CPU stage that streams; buffer sizes match stream_config.stream_plan exactly."""
    if not s.streamed_units:
        return None
    size = {u.index: u.buffer_bytes for u in inputs.manifest.units}
    streamed = sorted(s.streamed_units)
    read = inputs.storage.read_bytes_per_s
    stream_bytes = sum(size[u] for u in streamed)
    provenance = "measured" if read.provenance == "measured" else "prior"
    return StreamSpec(resident_units=sorted(s.resident_units), streamed_units=streamed, slots=s.slots,
                      slot_bytes=max(size[u] for u in streamed),
                      resident_bytes=sum(size[u] for u in s.resident_units), stream_bytes_per_pass=stream_bytes,
                      read_bytes_per_s=read.value, read_provenance=provenance,
                      predicted_io_ms_per_token=costs.stream_io_ms(stream_bytes, inputs.storage))


def stage_predictions(c: PlacementCandidate) -> list[StagePrediction]:
    return [StagePrediction(node_id=s.node_id, device=s.device, units=list(s.units),
                            resident_units=list(s.resident_units), streamed_units=list(s.streamed_units),
                            compute_ms=s.compute_ms, exposed_io_ms=s.exposed_io_ms, hop_ms=s.hop_ms,
                            upload_bytes=s.upload_bytes, path=s.path) for s in c.stages]


def prediction(c: PlacementCandidate, inputs: BridgeInputs, explain: Sequence[str],
               context: Sequence[str] = ()) -> PlanPrediction:
    """The candidate's score as a PlanPrediction. Assumptions: the bridge's, the search context's, the candidate's."""
    assumptions = list(dict.fromkeys([*inputs.notes, *context, *c.assumptions]))
    return PlanPrediction(decode_ms=c.decode_ms, prefill_ms=c.prefill_ms, load_ms=c.load_ms,
                          objective_ms=c.objective_ms, bottleneck=c.bottleneck,
                          stream_bytes_per_token=c.stream_bytes_per_token, stages=stage_predictions(c),
                          skipped=list(c.skipped), assumptions=assumptions, explain=list(explain),
                          manifest_fingerprint=inputs.manifest.fingerprint)


def context_notes(result: OptimizeResult) -> tuple[str, ...]:
    """The search-wide notes of a result (its assumptions minus the best candidate's own)."""
    best = result.best.assumptions if result.best else ()
    if best and result.assumptions[-len(best):] == best:
        return result.assumptions[: -len(best)]
    return result.assumptions


def candidate_lines(c: PlacementCandidate, head: int, label: str) -> list[str]:
    """Plain lines for a candidate scored outside a search (the planner's layout)."""
    lines = [f"{label}: {span_text(layout(c), head)}",
             f"decode {c.decode_ms:.1f} ms/token, prefill {c.prefill_ms:.0f} ms, load {c.load_ms:.0f} ms, "
             f"J {c.objective_ms:.0f} ms; {fmt_bytes(c.stream_bytes_per_token)} streamed per token",
             f"bottleneck: {c.bottleneck}"]
    return lines + [f"overflow: {r}" for r in c.reasons]


def candidate_summary(c: PlacementCandidate, head: int, label: str) -> CandidateSummary:
    why = (f"J {c.objective_ms:.0f} ms, decode {c.decode_ms:.1f} ms/token (prediction); bottleneck {c.bottleneck}"
           if c.feasible else "; ".join(c.reasons[:2]) or "does not fit")
    return CandidateSummary(label=f"{label}: {span_text(layout(c), head)}"[:300],
                            devices=[f"{s.node_id}:{s.device}" for s in c.stages], feasible=c.feasible,
                            reason=why[:400])


def usable_total(inputs: BridgeInputs) -> int:
    """Admissible bytes over the searched devices with each physical pool counted once: devices on one (node, pool)
    share it, at the smallest figure among them, as the optimizer groups them (a Mac's CPU and MTL0 are one pool)."""
    host = inputs.host
    devices = [(host.node_id, host.cpu.pool_id, host.cpu_pool_admissible),
               *([(host.node_id, host.gpu.pool_id, host.gpu_pool_admissible)] if host.gpu is not None else []),
               *((p.node_id, p.device.pool_id, p.pool_admissible) for p in inputs.peers)]
    per_pool: dict[tuple[str, str], int] = {}
    for node_id, pool_id, admissible in devices:
        key = (node_id, pool_id)
        per_pool[key] = min(per_pool.get(key, admissible), admissible)
    return sum(per_pool.values())


def to_plan(req: PlanRequest, c: PlacementCandidate | None, inputs: BridgeInputs, *, reasons: Sequence[str],
            warnings: Sequence[str], candidates: Sequence[CandidateSummary], explain: Sequence[str],
            context: Sequence[str] = ()) -> Plan:
    """A Plan in the planner's conventions for a candidate (None: an infeasible plan carrying only reasons)."""
    m = inputs.manifest
    chosen = [assignment(s, inputs) for s in c.stages] if c is not None else []
    host_cpu = f"{inputs.host.node_id}:{inputs.host.cpu.device}"
    first = c.stages[0] if c is not None and c.stages else None
    stream = stream_spec(first, inputs) if first is not None and f"{first.node_id}:{first.device}" == host_cpu else None
    priors = priors_warning(c, inputs) if c is not None else None
    plan = planner.finish_plan(req, model_shape(m, req.ctx_size), chosen, c is not None and c.feasible, paged=False,
                               reasons=[*reasons, *(c.reasons if c is not None else ())],
                               warnings=[*warnings, *([priors] if priors else [])], candidates=list(candidates),
                               host_ram_free=inputs.snapshot.host_memory.available,
                               total_budget=usable_total(inputs), stream=stream)
    measured_only = c is not None and priors is None
    return plan.model_copy(update={"heuristic": not measured_only, "limits": PlanLimits.of(req),
                                   "prediction": prediction(c, inputs, explain, context) if c is not None else None})


# --------------------------------------------------------------------------------------------------- planner plans


def _peer_index(device: str, endpoint: str, plan: Plan, refs: Sequence[PeerRef]) -> int:
    """A peer device's index within its RPC worker, from its provisional (host:port#i) or llama.cpp (RPCn) name."""
    if "#" in device:
        return int(device.rsplit("#", 1)[1])
    m = _RPC_NAME.fullmatch(device)
    if m is None:
        raise ValueError(f"peer device {device!r} is neither host:port#index nor RPCn")
    counts = {r.endpoint: r.endpoint_devices for r in refs}
    first = 0
    for ep in plan.rpc_endpoints:
        if ep == endpoint:
            return int(m.group(1)) - first
        first += counts.get(ep, 1)
    raise ValueError(f"peer device {device} names endpoint {endpoint}, which the plan's --rpc list does not hold")


def _plan_key(a: DeviceAssignment, plan: Plan, inputs: BridgeInputs) -> str:
    host = inputs.host
    if a.endpoint is None:
        if a.node_id == host.node_id and a.device == HOST_CPU:
            return f"{host.node_id}:{host.cpu.device}"
        if a.node_id == host.node_id and host.gpu is not None and a.device == inputs.local_gpu:
            return f"{host.node_id}:{host.gpu.device}"
        raise ValueError(f"local device {a.device} is not one the optimizer places")
    index = _peer_index(a.device, a.endpoint, plan, inputs.refs)
    ref = next((r for r in inputs.refs if r.node_id == a.node_id and r.endpoint == a.endpoint and r.index == index),
               None)
    if ref is None:
        raise ValueError(f"peer device {a.node_id}:{a.device} is not an eligible device of the search now")
    return ref.key


def plan_layout(plan: Plan, inputs: BridgeInputs) -> list[Span]:
    """The optimizer spans of a llama.cpp plan; ValueError when the plan has no such layout."""
    if plan.engine != "llama_rpc":
        raise ValueError(f"a {plan.engine} plan has no llama.cpp layout")
    if plan.tier == "gpu_disk":
        raise ValueError("it pages CPU layers from the GGUF on disk (gpu_disk), which the optimizer does not model")
    if not plan.assignments:
        raise ValueError("it has no device assignments")
    spans: list[Span] = []
    start = 0
    for a in plan.assignments:
        if a.layers != list(range(start, start + len(a.layers))):
            raise ValueError(f"{a.node_id}:{a.device} does not hold the contiguous span after unit {start}")
        end = start + len(a.layers) + (1 if a.holds_output else 0)
        spans.append((_plan_key(a, plan, inputs), start, end))
        start = end
    if start != inputs.manifest.n_layer + 1:
        raise ValueError(f"its spans cover {start} units, the model has {inputs.manifest.n_layer + 1}")
    return spans


def workload_json(w: Workload) -> dict[str, Any]:
    return asdict(w)


def _when(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def pools_json(snap: Snapshot) -> dict[str, Any]:
    """The snapshot as plain JSON: every pool with its admissible bytes, what was left out, and the host memory."""
    mem = {**asdict(snap.host_memory), "observed_at": _when(snap.host_memory.observed_at)}
    rows = [{"node_id": p.node_id, "pool_id": p.pool_id, "kind": p.kind, "total_bytes": p.total_bytes,
             "free_bytes": p.free_bytes, "reserve_bytes": p.reserve_bytes, "cap_bytes": p.cap_bytes,
             "admissible_bytes": snap.admissible[(p.node_id, p.pool_id)], "devices": list(p.devices),
             "source": p.source, "observed_at": _when(p.observed_at),
             "age_s": round(max(0.0, (snap.observed_at - p.observed_at).total_seconds()), 1)}
            for p in snap.report.pools]
    leases = [{"deployment_id": lease.deployment_id, "node_id": lease.node_id, "pool_id": lease.pool_id,
               "bytes": lease.bytes, "materialized": lease.materialized, "revision": lease.revision}
              for lease in snap.leases]
    return {"observed_at": _when(snap.observed_at), "host_memory": mem, "pools": rows,
            "excluded": [asdict(x) for x in snap.report.excluded], "notes": list(snap.report.notes),
            "leases": leases, "released": snap.released}


def candidate_json(c: PlacementCandidate, head: int) -> dict[str, Any]:
    return {"layout": span_text(layout(c), head), "feasible": c.feasible, "decode_ms": c.decode_ms,
            "prefill_ms": c.prefill_ms, "load_ms": c.load_ms, "objective_ms": c.objective_ms,
            "bottleneck": c.bottleneck, "stream_bytes_per_token": c.stream_bytes_per_token,
            "stages": [s.model_dump(mode="json") for s in stage_predictions(c)], "skipped": list(c.skipped),
            "reasons": list(c.reasons), "assumptions": list(c.assumptions)}

