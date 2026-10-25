"""Memory pools of the cluster, admission against them, and host and process memory snapshots.

A pool is one physical allocation pool on one node: system RAM, a unified CPU and GPU pool, one discrete GPU's VRAM,
or the host's free volume space. Devices that allocate from the same pool share it, so a Mac's CPU and Metal device
are one `uma` pool and are counted once. Admission sums every demand on a pool and compares the sum with what the
pool can still give: the fresh free observation minus the reserve and the leases the engine has not loaded yet, and
the contribution cap minus every lease. A loaded (materialized) lease is already inside the free observation, so it
is not subtracted from free a second time.

The snapshot functions read the OS directly. A failing OS call never raises: the fields it would have filled are None
and the error is recorded with the observation.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime

import psutil

from ..schemas import MemoryPool, NodeView
from ..units import MiB, fmt_bytes
from .types import Admission, Pool, PoolCheck, PoolDemand, PoolLease

# Headroom kept free on a discrete GPU, the same margin as services.deployments.VRAM_HEADROOM. Defined here rather
# than imported so the capacity layer does not depend on the service layer.
VRAM_HEADROOM = 512 * MiB
HOST_NODE_ID = "local"  # services.nodes.LOCAL_ID; owner of the host storage pool when no local node is passed
STORAGE_POOL_ID = "ssd"

_PoolKey = tuple[str, str]


class ProcessMemoryError(RuntimeError):
    """The process exists but its memory counters could not be read (for example access denied)."""


@dataclass(frozen=True)
class PoolExclusion:
    """A pool left out of planning, with the reason (unknown free memory, not contributed, duplicate)."""

    node_id: str
    pool_id: str
    reason: str


@dataclass(frozen=True)
class PoolsReport:
    """Pools usable for admission, the pools left out with reasons, and facts no single pool can carry."""

    pools: tuple[Pool, ...]
    excluded: tuple[PoolExclusion, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HostMemory:
    """One observation of host memory, in bytes. A field the OS could not report is None, never 0."""

    total: int | None
    available: int | None  # Windows: free, zeroed and standby pages; elsewhere psutil's OS-specific estimate
    commit_charge: int | None  # Windows commit total; None on other OSes
    commit_limit: int | None  # Windows physical memory plus page files; None on other OSes
    system_cache: int | None  # Windows GetPerformanceInfo SystemCache (standby list plus system working set)
    observed_at: datetime
    source: str
    error: str | None = None


@dataclass(frozen=True)
class ProcessMemory:
    """Memory counters of one process, in bytes, as the OS reports them at `observed_at`."""

    pid: int
    working_set: int  # resident now
    private: int | None  # Windows PrivateUsage (private commit, including trimmed pages); None on other OSes
    peak_working_set: int | None
    page_faults: int | None  # cumulative soft plus hard faults
    observed_at: datetime
    source: str


# --------------------------------------------------------------------------------------------------- pools


def pools_from_nodes(nodes: Sequence[NodeView], *, host_reserve_bytes: int, storage_free_bytes: int | None = None,
                     now: datetime, storage_reserve_bytes: int = 0) -> tuple[Pool, ...]:
    """One Pool per (node, physical pool) usable for admission. See `pools_report` for what was left out and why."""
    return pools_report(nodes, host_reserve_bytes=host_reserve_bytes, storage_free_bytes=storage_free_bytes,
                        now=now, storage_reserve_bytes=storage_reserve_bytes).pools


def pools_report(nodes: Sequence[NodeView], *, host_reserve_bytes: int, storage_free_bytes: int | None = None,
                 now: datetime, storage_reserve_bytes: int = 0) -> PoolsReport:
    """Pools of the given nodes. The caller passes only nodes it may plan onto; node state is not filtered here.

    Local node: every pool with a known free observation; RAM and unified pools keep `host_reserve_bytes`, discrete
    VRAM keeps `VRAM_HEADROOM`, no cap. Peer: only pools a contributed device allocates from; RAM and unified pools
    keep the contributor's `reserve_bytes`, discrete VRAM keeps `VRAM_HEADROOM`. The cap is `max_bytes`, a node total:
    a peer with one usable pool gets all of it, a peer with several gets it split in proportion to free memory so the
    pools together never admit more than the contributor allowed. The host storage pool `ssd` is added when
    `storage_free_bytes` is given (an observation taken by the caller at `now`).
    """
    _require_non_negative(host_reserve_bytes=host_reserve_bytes, storage_reserve_bytes=storage_reserve_bytes)
    if storage_free_bytes is not None:
        _require_non_negative(storage_free_bytes=storage_free_bytes)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (UTC)")
    parts = [_local_pools(n, host_reserve_bytes) if n.is_local else _peer_pools(n) for n in nodes]
    parts.append(_storage_pools(nodes, storage_free_bytes, storage_reserve_bytes, now))
    return _merge(parts)


def _require_non_negative(**values: int) -> None:
    bad = [name for name, value in values.items() if value < 0]
    if bad:
        raise ValueError(f"must not be negative: {', '.join(bad)}")


def _local_pools(node: NodeView, host_reserve_bytes: int) -> PoolsReport:
    pools: list[Pool] = []
    excluded: list[PoolExclusion] = []
    for mp in node.inventory.memory_pools:
        if mp.free_bytes is None:
            excluded.append(_unknown_free(node, mp))
            continue
        reserve = VRAM_HEADROOM if mp.kind == "discrete_vram" else host_reserve_bytes
        pools.append(_pool(node, mp, mp.free_bytes, reserve=reserve, cap=None,
                           devices=_pool_devices(node, mp.id, None)))
    return PoolsReport(pools=tuple(pools), excluded=tuple(excluded))


def _peer_pools(node: NodeView) -> PoolsReport:
    limits = node.contribution
    contributed = tuple(dict.fromkeys(limits.devices))
    known = {d.id: d.pool_id for d in node.inventory.devices}
    used = tuple(dict.fromkeys(known[name] for name in contributed if name in known))
    usable: list[tuple[MemoryPool, int]] = []
    excluded: list[PoolExclusion] = []
    for mp in node.inventory.memory_pools:
        if mp.id not in used:
            excluded.append(PoolExclusion(node.id, mp.id, "not contributed: no contributed device allocates from it"))
        elif mp.free_bytes is None:
            excluded.append(_unknown_free(node, mp))
        else:
            usable.append((mp, mp.free_bytes))
    caps = _split_cap(limits.max_bytes, [free for _, free in usable])
    pools = tuple(_pool(node, mp, free, cap=cap, devices=_pool_devices(node, mp.id, contributed),
                        reserve=VRAM_HEADROOM if mp.kind == "discrete_vram" else limits.reserve_bytes)
                  for (mp, free), cap in zip(usable, caps, strict=True))
    listed = {mp.id for mp in node.inventory.memory_pools}
    excluded.extend(PoolExclusion(node.id, pool_id, "a contributed device allocates from it but the inventory "
                                  "lists no such pool") for pool_id in used if pool_id not in listed)
    notes = [f"{node.id}: contributed device {name} is not in the inventory, so it maps to no pool"
             for name in contributed if name not in known]
    if len(pools) > 1:
        split = ", ".join(f"{p.pool_id} {fmt_bytes(p.cap_bytes)}" for p in pools)
        notes.append(f"{node.id}: the contribution cap {fmt_bytes(limits.max_bytes)} is a node total; it is split "
                     f"across {len(pools)} pools in proportion to free memory ({split}), as remote_budgets does")
    return PoolsReport(pools=pools, excluded=tuple(excluded), notes=tuple(notes))


def _split_cap(cap: int, frees: Sequence[int]) -> tuple[int, ...]:
    """Per-pool caps that never sum to more than the node total `cap`.

    One pool gets the whole cap. Several pools share it in proportion to their free memory (the split
    services.deployments.remote_budgets applies across RPC devices); with no free memory anywhere, equally.
    """
    if len(frees) <= 1:
        return (cap,) * len(frees)
    weights = [max(0, free) for free in frees]
    total = sum(weights)
    if total == 0:
        return tuple(cap // len(frees) for _ in frees)
    return tuple(cap * weight // total for weight in weights)


def _unknown_free(node: NodeView, mp: MemoryPool) -> PoolExclusion:
    return PoolExclusion(node.id, mp.id, f"free memory unknown (source: {mp.source})")


def _pool_devices(node: NodeView, pool_id: str, allowed: tuple[str, ...] | None) -> tuple[str, ...]:
    """Engine devices allocating from the pool. Vendor-only entries (backend unknown, e.g. an iGPU) are left out."""
    engine_names = {d.name for d in node.inventory.engine.devices}
    ids = (d.id for d in node.inventory.devices
           if d.pool_id == pool_id and (d.backend != "unknown" or d.id in engine_names)
           and (allowed is None or d.id in allowed))
    return tuple(dict.fromkeys(ids))


def _pool(node: NodeView, mp: MemoryPool, free: int, *, reserve: int, cap: int | None,
          devices: tuple[str, ...]) -> Pool:
    observed = mp.measured_at or node.inventory.collected_at
    source = mp.source if mp.measured_at else f"{mp.source} (no measured_at: inventory collected_at used)"
    return Pool(node_id=node.id, pool_id=mp.id, kind=mp.kind, total_bytes=mp.total_bytes, free_bytes=free,
                observed_at=observed, source=source, reserve_bytes=reserve, cap_bytes=cap, devices=devices)


def _storage_pools(nodes: Sequence[NodeView], free: int | None, reserve: int, now: datetime) -> PoolsReport:
    node_id = next((n.id for n in nodes if n.is_local), HOST_NODE_ID)
    if free is None:
        return PoolsReport(pools=(), excluded=(PoolExclusion(node_id, STORAGE_POOL_ID,
                                                             "free volume space not observed"),))
    ssd = Pool(node_id=node_id, pool_id=STORAGE_POOL_ID, kind="storage", total_bytes=None, free_bytes=free,
               observed_at=now, source="free volume space (caller observation)", reserve_bytes=reserve)
    return PoolsReport(pools=(ssd,))


def _merge(parts: Sequence[PoolsReport]) -> PoolsReport:
    pools: list[Pool] = []
    excluded: list[PoolExclusion] = []
    seen: set[_PoolKey] = set()
    for part in parts:
        for pool in part.pools:
            key = (pool.node_id, pool.pool_id)
            if key in seen:
                excluded.append(PoolExclusion(*key, "duplicate pool: the first observation was kept"))
                continue
            seen.add(key)
            pools.append(pool)
        excluded.extend(part.excluded)
    return PoolsReport(pools=tuple(pools), excluded=tuple(excluded),
                       notes=tuple(note for part in parts for note in part.notes))


# --------------------------------------------------------------------------------------------------- admission


@dataclass(frozen=True)
class _Budget:
    free_side: int  # free - reserve - unmaterialized leases
    cap_side: int | None  # cap - all leases, None without a cap
    unmaterialized: int
    leased: int


def _budget(pool: Pool, leases: Sequence[PoolLease]) -> _Budget:
    if pool.reserve_bytes < 0 or (pool.cap_bytes is not None and pool.cap_bytes < 0):
        raise ValueError(f"pool {pool.pool_id} on {pool.node_id} has a negative reserve or cap "
                         f"(reserve {pool.reserve_bytes}, cap {pool.cap_bytes})")
    mine = [lease for lease in leases if lease.node_id == pool.node_id and lease.pool_id == pool.pool_id]
    negative = [lease.lease_id for lease in mine if lease.bytes < 0]
    if negative:
        raise ValueError(f"leases with negative bytes on {pool.pool_id}/{pool.node_id}: {', '.join(negative)}")
    unmaterialized = sum(lease.bytes for lease in mine if not lease.materialized)
    leased = sum(lease.bytes for lease in mine)
    return _Budget(free_side=pool.free_bytes - pool.reserve_bytes - unmaterialized,
                   cap_side=None if pool.cap_bytes is None else pool.cap_bytes - leased,
                   unmaterialized=unmaterialized, leased=leased)


def _admissible(budget: _Budget) -> int:
    limit = budget.free_side if budget.cap_side is None else min(budget.free_side, budget.cap_side)
    return max(0, limit)


def admissible_bytes(pool: Pool, leases: Sequence[PoolLease]) -> int:
    """`min(free - reserve - unmaterialized leases, cap - all leases)`, floored at zero.

    Only leases on this (node, pool) count. Materialized leases are already inside `free` and are subtracted only
    from the cap.
    """
    return _admissible(_budget(pool, leases))


def admit(demands: Sequence[PoolDemand], pools: Sequence[Pool], leases: Sequence[PoolLease]) -> Admission:
    """Sum `PoolDemand.peak` per (node, pool) and check each sum against the pool's admissible bytes.

    A demand on a pool that is not in `pools` fails with a reason. No demands means nothing to check: admitted.
    A pool listed twice is a counting error (a shared pool would be offered twice) and raises ValueError.
    """
    index = _index_pools(pools)
    totals = _demand_totals(demands)
    checks = tuple(_check(key, peak, index.get(key), leases) for key, peak in totals.items())
    return Admission(ok=all(c.ok for c in checks), checks=checks)


def _index_pools(pools: Sequence[Pool]) -> dict[_PoolKey, Pool]:
    index: dict[_PoolKey, Pool] = {}
    for pool in pools:
        key = (pool.node_id, pool.pool_id)
        if key in index:
            raise ValueError(f"pool {pool.pool_id} on {pool.node_id} is listed twice; a pool must be counted once")
        index[key] = pool
    return index


def _demand_totals(demands: Sequence[PoolDemand]) -> dict[_PoolKey, int]:
    totals: dict[_PoolKey, int] = {}
    for demand in demands:
        negative = [f.name for f in fields(demand) if isinstance(getattr(demand, f.name), int)
                    and getattr(demand, f.name) < 0]
        if negative:
            raise ValueError(f"demand on {demand.pool_id}/{demand.node_id} has negative {', '.join(negative)}")
        key = (demand.node_id, demand.pool_id)
        totals[key] = totals.get(key, 0) + demand.peak
    return totals


def _check(key: _PoolKey, peak: int, pool: Pool | None, leases: Sequence[PoolLease]) -> PoolCheck:
    node_id, pool_id = key
    if pool is None:
        return PoolCheck(node_id=node_id, pool_id=pool_id, demand_peak=peak, admissible=0, ok=False,
                         reason=f"{pool_id} on {node_id}: no such pool (unknown, not contributed, or without a "
                                "free-memory observation)")
    budget = _budget(pool, leases)
    admissible = _admissible(budget)
    ok = peak <= admissible
    cap = ("no cap" if pool.cap_bytes is None
           else f"cap {fmt_bytes(pool.cap_bytes)} - all leases {fmt_bytes(budget.leased)}")
    basis = (f"free {fmt_bytes(pool.free_bytes)} - reserve {fmt_bytes(pool.reserve_bytes)} - not yet loaded "
             f"{fmt_bytes(budget.unmaterialized)}; {cap}")
    verdict = "fits" if ok else f"short by {fmt_bytes(peak - admissible)}"
    return PoolCheck(node_id=node_id, pool_id=pool_id, demand_peak=peak, admissible=admissible, ok=ok,
                     reason=f"{pool_id} on {node_id}: peak {fmt_bytes(peak)} vs admissible "
                            f"{fmt_bytes(admissible)} ({basis}): {verdict}")


# --------------------------------------------------------------------------------------------------- host memory


class _PerformanceInformation(ctypes.Structure):
    """PERFORMANCE_INFORMATION from psapi.h. Counts are in pages of `PageSize` bytes."""

    _fields_ = [("cb", ctypes.c_uint32), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_uint32), ("ProcessCount", ctypes.c_uint32),
                ("ThreadCount", ctypes.c_uint32)]


def _performance_info() -> _PerformanceInformation:
    """Call GetPerformanceInfo (Windows only). Raises OSError on failure."""
    info = _PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    fn = psapi.GetPerformanceInfo
    fn.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_uint32]
    fn.restype = ctypes.c_int
    if not fn(ctypes.byref(info), info.cb):
        err = ctypes.get_last_error()
        raise OSError(err, f"GetPerformanceInfo failed (error {err})")
    if info.PageSize <= 0:
        raise OSError(f"GetPerformanceInfo returned page size {info.PageSize}")
    return info


def host_memory_snapshot() -> HostMemory:
    """Host memory now. Windows: GetPerformanceInfo (psutil if it fails). Other OSes: psutil. Never raises."""
    now = datetime.now(UTC)
    if sys.platform != "win32":
        return _psutil_snapshot(now, None)
    try:
        info = _performance_info()
    except Exception as exc:  # noqa: BLE001 - an OS query failure is reported in the snapshot, not raised
        return _psutil_snapshot(now, f"GetPerformanceInfo: {type(exc).__name__}: {exc}")
    page = int(info.PageSize)
    return HostMemory(total=info.PhysicalTotal * page, available=info.PhysicalAvailable * page,
                      commit_charge=info.CommitTotal * page, commit_limit=info.CommitLimit * page,
                      system_cache=info.SystemCache * page, observed_at=now, source="GetPerformanceInfo")


def _psutil_snapshot(now: datetime, prior_error: str | None) -> HostMemory:
    try:
        vm = psutil.virtual_memory()
    except Exception as exc:  # noqa: BLE001 - an OS query failure is reported in the snapshot, not raised
        error = f"psutil.virtual_memory: {type(exc).__name__}: {exc}"
        return HostMemory(total=None, available=None, commit_charge=None, commit_limit=None, system_cache=None,
                          observed_at=now, source="none",
                          error=error if prior_error is None else f"{prior_error}; {error}")
    return HostMemory(total=int(vm.total), available=int(vm.available), commit_charge=None, commit_limit=None,
                      system_cache=None, observed_at=now, source="psutil.virtual_memory", error=prior_error)


def process_memory(pid: int) -> ProcessMemory | None:
    """Memory counters of `pid`, or None if the process is gone. Raises ProcessMemoryError if it cannot be read."""
    if pid < 0:
        raise ValueError(f"pid must not be negative: {pid}")
    try:
        mem = psutil.Process(pid).memory_info()
    except psutil.NoSuchProcess:  # includes ZombieProcess
        return None
    except psutil.AccessDenied as exc:
        raise ProcessMemoryError(f"pid {pid}: access denied reading memory counters") from exc
    except psutil.Error as exc:
        raise ProcessMemoryError(f"pid {pid}: {type(exc).__name__}: {exc}") from exc
    return ProcessMemory(pid=pid, working_set=int(mem.rss), private=_optional_int(mem, "private"),
                         peak_working_set=_optional_int(mem, "peak_wset"),
                         page_faults=_optional_int(mem, "num_page_faults", "pfaults"),
                         observed_at=datetime.now(UTC), source="psutil.Process.memory_info")


def _optional_int(record: object, *names: str) -> int | None:
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return int(value)
    return None
