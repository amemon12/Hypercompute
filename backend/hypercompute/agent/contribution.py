"""Conservative default contribution limits, shared by the friend agent and the host's own machine.

These are soft admission budgets for the planner, not enforced process or GPU caps.
"""

from __future__ import annotations

from ..schemas import ContributionLimits, HostInventory
from ..units import GiB, MiB

ACCEL_PREFIXES = ("CUDA", "MTL", "Vulkan", "ROCm")
VRAM_HEADROOM = 512 * MiB


def eligible_devices(inv: HostInventory) -> tuple[list[str], bool]:
    """Engine-visible accelerators to share; falls back to the CPU device (explicitly labeled)."""
    accel = [d.name for d in inv.engine.devices if d.name.startswith(ACCEL_PREFIXES)]
    return (accel, False) if accel else (["CPU"], True)


def default_reserve(inv: HostInventory) -> int:
    ram = next((p for p in inv.memory_pools if p.id in ("ram", "uma")), None)
    total = ram.total_bytes if ram and ram.total_bytes else 16 * GiB
    return max(4 * GiB, int(total * 0.2))


def default_max_bytes(inv: HostInventory, devices: list[str]) -> int:
    total = 0
    pools = {p.id: p for p in inv.memory_pools}
    counted: set[str] = set()
    for dev_id in devices:
        dev = next((d for d in inv.devices if d.id == dev_id), None)
        engine_dev = next((d for d in inv.engine.devices if d.name == dev_id), None)
        pool = pools.get(dev.pool_id) if dev else pools.get("ram")
        if pool is None or pool.id in counted:
            continue  # a unified pool shared by several devices is counted once
        counted.add(pool.id)
        if pool.kind == "discrete_vram":
            cap = pool.total_bytes or (engine_dev.total_bytes if engine_dev else 0) or 0
            total += max(0, cap - VRAM_HEADROOM)
        elif pool.kind == "unified":
            cap = pool.total_bytes or 0
            total += max(0, min(int(cap * 0.65), cap - 6 * GiB))
        else:
            cap = pool.total_bytes or 0
            total += max(0, min(int(cap * 0.4), cap - 8 * GiB))
    return total


def default_contribution(inv: HostInventory, devices: list[str] | None = None, max_bytes: int | None = None,
                         reserve_bytes: int | None = None, cache_enabled: bool = False) -> ContributionLimits:
    chosen = devices or eligible_devices(inv)[0]
    return ContributionLimits(
        devices=chosen,
        max_bytes=max_bytes if max_bytes is not None else default_max_bytes(inv, chosen),
        reserve_bytes=reserve_bytes if reserve_bytes is not None else default_reserve(inv),
        cache_enabled=cache_enabled,
    )
