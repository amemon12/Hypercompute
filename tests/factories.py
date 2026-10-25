"""Sanitized builders for inventories, nodes and plans used by protocol tests."""

from __future__ import annotations

from hypercompute.schemas import (
    ContributionLimits,
    CpuInfo,
    Device,
    EngineDevice,
    EngineInfo,
    HostInventory,
    MemoryPool,
    OSInfo,
    utcnow,
)
from hypercompute.units import GiB, MiB


def engine_info(release: str = "b11770", rpc_proto: str = "8.0.0", smoke: str = "passed",
                devices: list[EngineDevice] | None = None, accelerator: str = "cuda") -> EngineInfo:
    if devices is None:
        devices = [EngineDevice(name="CUDA0", description="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                                total_bytes=12226 * MiB, free_bytes=11026 * MiB)]
    return EngineInfo(installed=True, release=release, build_number=int(release[1:]), commit="39da130fc",
                      rpc_proto=rpc_proto, variant="windows-x64-cuda13.4", accelerator=accelerator,  # type: ignore[arg-type]
                      smoke=smoke, smoke_detail="fixture", devices=devices)  # type: ignore[arg-type]


def make_inventory(system: str = "windows", engine: EngineInfo | None = None, ram_total: int = 32 * GiB,
                   ram_free: int = 16 * GiB, vram_total: int | None = 12226 * MiB,
                   vram_free: int | None = 11026 * MiB) -> HostInventory:
    engine = engine or engine_info()
    now = utcnow()
    if system == "macos":
        pools = [MemoryPool(id="uma", kind="unified", total_bytes=ram_total, free_bytes=ram_free, measured_at=now,
                            source="fixture")]
        devices = [Device(id="CPU", kind="cpu", name="Apple M4", pool_id="uma", backend="cpu", source="fixture"),
                   Device(id="MTL0", kind="gpu", name="Apple M4 (10-core GPU)", pool_id="uma", backend="metal",
                          total_bytes=int(ram_total * 0.67), source="fixture")]
    else:
        pools = [MemoryPool(id="ram", kind="system_ram", total_bytes=ram_total, free_bytes=ram_free, measured_at=now,
                            source="fixture")]
        devices = [Device(id="CPU", kind="cpu", name="AMD Ryzen AI 9 HX 370", pool_id="ram", backend="cpu",
                          source="fixture")]
        if vram_total is not None:
            pools.append(MemoryPool(id="vram:CUDA0", kind="discrete_vram", total_bytes=vram_total,
                                    free_bytes=vram_free, measured_at=now, source="fixture"))
            devices.append(Device(id="CUDA0", kind="gpu", name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
                                  pool_id="vram:CUDA0", backend="cuda", total_bytes=vram_total, free_bytes=vram_free,
                                  source="fixture"))
    return HostInventory(hostname=f"{system}-fixture", os=OSInfo(system=system, release="x", version="x",  # type: ignore[arg-type]
                                                                 machine="arm64" if system == "macos" else "x86_64"),
                         cpu=CpuInfo(model="fixture", physical_cores=8, logical_cores=16), memory_pools=pools,
                         devices=devices, disk_free_bytes=100 * GiB, network=[], engine=engine, collected_at=now)


def contribution(devices: list[str], max_gib: float = 8, reserve_gib: float = 4) -> ContributionLimits:
    return ContributionLimits(devices=devices, max_bytes=int(max_gib * GiB), reserve_bytes=int(reserve_gib * GiB))
