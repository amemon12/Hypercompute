"""Native hardware inventory normalized into HostInventory.

Probes run as fixed argument arrays with timeouts; a missing tool yields a diagnostic and null
fields, never a guessed value. Serial numbers and hardware UUIDs are not collected.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path

import psutil

from .. import config, engine, netutil
from ..schemas import (
    CpuInfo,
    Device,
    Diagnostic,
    EngineDevice,
    EngineInfo,
    HostInventory,
    MemoryPool,
    OSInfo,
    utcnow,
)
from ..units import GiB, MiB

log = logging.getLogger("hypercompute.inventory")


def _run(args: list[str], timeout: float = 20.0) -> str | None:
    exe = shutil.which(args[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=timeout, check=False,
                             errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("probe %s failed: %s", args[0], exc)
        return None
    return out.stdout if out.returncode == 0 else None


def _powershell_json(command: str) -> list[dict]:
    raw = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command + " | ConvertTo-Json -Depth 3"])
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def os_info() -> OSInfo:
    system = engine.host_os()
    release, version = platform.release(), platform.version()
    if system == "macos":
        release = platform.mac_ver()[0] or release
    return OSInfo(system=system, release=release, version=version, machine=engine.host_arch())  # type: ignore[arg-type]


def cpu_info(system: str) -> CpuInfo:
    model: str | None = None
    if system == "macos":
        model = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or None
    elif system == "windows":
        rows = _powershell_json("Get-CimInstance Win32_Processor | Select-Object Name")
        model = (rows[0].get("Name") or "").strip() if rows else None
    elif system == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            model = None
    return CpuInfo(model=model or platform.processor() or None, physical_cores=psutil.cpu_count(logical=False),
                   logical_cores=psutil.cpu_count(logical=True))


def nvidia_gpus() -> list[dict]:
    raw = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (raw or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        def mib_or_none(v: str) -> int | None:
            return int(float(v)) * MiB if v.replace(".", "", 1).isdigit() else None
        gpus.append({"index": int(parts[0]) if parts[0].isdigit() else len(gpus), "name": parts[1],
                     "total": mib_or_none(parts[2]), "free": mib_or_none(parts[3]),
                     "driver": parts[4], "cc": parts[5]})
    return gpus


def _engine_name_for(gpu_name: str, fallback: str, engine_devices: list[EngineDevice]) -> str:
    for d in engine_devices:
        if gpu_name.lower() in d.description.lower():
            return d.name
    return fallback


_VM_STAT_PAGE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_LINE = re.compile(r'^"?([^":\n]+)"?:\s+(\d+)\.\s*$', re.MULTILINE)
_VM_STAT_USED = ("Anonymous pages", "Pages purgeable", "Pages wired down", "Pages occupied by compressor")


def macos_available_bytes(vm_stat: str, total: int) -> int | None:
    """Memory macOS can hand to a new allocation without swapping: total minus app memory (anonymous pages that are not
    purgeable), wired and compressed memory, which is what Activity Monitor calls Memory Used.

    psutil's available on macOS is free + inactive only, so reclaimable file cache in active pages is missed: a 16 GiB
    M4 Air that had just read 8.5 GB of model files showed 7.1 GiB available to psutil while macOS reported 91% free."""
    page = _VM_STAT_PAGE.search(vm_stat)
    counts = {name.strip(): int(n) for name, n in _VM_STAT_LINE.findall(vm_stat)}
    if page is None or any(k not in counts for k in _VM_STAT_USED):
        return None
    used_pages = (max(0, counts["Anonymous pages"] - counts["Pages purgeable"]) + counts["Pages wired down"]
                  + counts["Pages occupied by compressor"])
    return max(0, total - used_pages * int(page.group(1)))


def available_ram_bytes() -> int:
    """psutil's available RAM, or on macOS the vm_stat measure above when vm_stat answers."""
    vm = psutil.virtual_memory()
    if platform.system() == "Darwin":
        out = _run(["vm_stat"], timeout=5.0)
        available = macos_available_bytes(out, int(vm.total)) if out else None
        if available is not None:
            return available
    return int(vm.available)


def _ram_pool(kind: str, pool_id: str, note: str | None = None) -> MemoryPool:
    vm = psutil.virtual_memory()
    source = "psutil + vm_stat" if platform.system() == "Darwin" else "psutil"
    return MemoryPool(id=pool_id, kind=kind, total_bytes=int(vm.total), free_bytes=available_ram_bytes(),  # type: ignore[arg-type]
                      measured_at=utcnow(), source=source, note=note)


def apple_devices(eng: EngineInfo, cpu_name: str | None) -> tuple[list[MemoryPool], list[Device]]:
    pool = _ram_pool("unified", "uma", note="CPU and GPU share this pool; counted once")
    total = _run(["sysctl", "-n", "hw.memsize"])
    if total and total.strip().isdigit():
        pool = pool.model_copy(update={"total_bytes": int(total.strip()), "source": "sysctl hw.memsize + psutil"})
    gpu_name = "Apple GPU"
    raw = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=40)
    if raw:
        try:
            displays = json.loads(raw).get("SPDisplaysDataType", [])
            if displays:
                cores = displays[0].get("sppci_cores")
                gpu_name = displays[0].get("sppci_model", gpu_name) + (f" ({cores}-core GPU)" if cores else "")
        except (json.JSONDecodeError, AttributeError):
            pass
    mtl = next((d for d in eng.devices if d.name.startswith("MTL")), None)
    devices = [
        Device(id="CPU", kind="cpu", name=cpu_name or "Apple CPU", vendor="Apple", pool_id="uma",
               backend="cpu", source="sysctl machdep.cpu.brand_string"),
        Device(id=mtl.name if mtl else "MTL0", kind="gpu", name=gpu_name, vendor="Apple", pool_id="uma",
               backend="metal", total_bytes=mtl.total_bytes if mtl else None,
               free_bytes=mtl.free_bytes if mtl else None,
               source="llama-server --list-devices (Metal working-set)" if mtl else "system_profiler"),
    ]
    return [pool], devices


def pc_devices(system: str, eng: EngineInfo, cpu_name: str | None) -> tuple[list[MemoryPool], list[Device]]:
    pools: list[MemoryPool] = []
    devices: list[Device] = [Device(id="CPU", kind="cpu", name=cpu_name or platform.processor() or "CPU",
                                    pool_id="ram", backend="cpu", source="cpu model probe")]
    unified_gpu = False
    for gpu in nvidia_gpus():
        is_gb10 = "gb10" in gpu["name"].lower()
        unified_gpu = unified_gpu or is_gb10
        dev_id = _engine_name_for(gpu["name"], f"CUDA{gpu['index']}", eng.devices)
        pool_id = "uma" if is_gb10 else f"vram:{dev_id}"
        if not is_gb10:
            pools.append(MemoryPool(id=pool_id, kind="discrete_vram", total_bytes=gpu["total"],
                                    free_bytes=gpu["free"], measured_at=utcnow(), source="nvidia-smi"))
        devices.append(Device(id=dev_id, kind="gpu", name=gpu["name"], vendor="NVIDIA", pool_id=pool_id,
                              backend="cuda", total_bytes=gpu["total"], free_bytes=gpu["free"],
                              driver_version=gpu["driver"], compute_capability=gpu["cc"], source="nvidia-smi"))
    if unified_gpu:
        pools.insert(0, _ram_pool("unified", "uma", note="GB10 CPU/GPU unified memory; counted once"))
        devices[0] = devices[0].model_copy(update={"pool_id": "uma"})
    else:
        pools.insert(0, _ram_pool("system_ram", "ram"))
    if system == "windows":
        for row in _powershell_json("Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion"):
            name = (row.get("Name") or "").strip()
            if not name or "nvidia" in name.lower():
                continue
            devices.append(Device(id=name, kind="igpu" if "radeon" in name.lower() or "intel" in name.lower()
                                  else "gpu", name=name, pool_id="ram", backend="unknown",
                                  driver_version=row.get("DriverVersion"),
                                  source="Win32_VideoController (AdapterRAM not trusted)"))
    return pools, devices


def windows_diagnostics() -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for row in _powershell_json("Get-NetConnectionProfile | Select-Object InterfaceAlias, NetworkCategory"):
        category = row.get("NetworkCategory")
        if category in (0, "Public"):
            out.append(Diagnostic(level="warning", code="windows_public_network",
                                  message=f"Network '{row.get('InterfaceAlias')}' uses the Windows Public profile; "
                                          "inbound connections to the worker are blocked by default.",
                                  fix="Allow ggml-rpc-server.exe when Windows asks, or add the scoped firewall rule "
                                      "that `hypercompute join` prints."))
    return out


def collect(include_engine: bool = True) -> HostInventory:
    osi = os_info()
    eng = engine.inspect(engine.find_installed()) if include_engine else EngineInfo(installed=False)
    cpu = cpu_info(osi.system)
    if osi.system == "macos":
        pools, devices = apple_devices(eng, cpu.model)
    else:
        pools, devices = pc_devices(osi.system, eng, cpu.model)
    diags: list[Diagnostic] = []
    if not eng.installed:
        diags.append(Diagnostic(level="error", code="engine_missing", message="llama.cpp engine is not installed.",
                                fix="uv run hypercompute engine install"))
    elif eng.smoke == "failed":
        diags.append(Diagnostic(level="error", code="engine_smoke_failed", message=eng.smoke_detail or "smoke failed",
                                fix="uv run hypercompute engine install --force"))
    if not netutil.lan_addresses():
        diags.append(Diagnostic(level="warning", code="no_private_lan",
                                message="No private LAN IPv4 address found on an active interface.",
                                fix="Connect to the same Wi-Fi/Ethernet as the host, or use an existing VPN overlay."))
    ram = next((p for p in pools if p.id in ("ram", "uma")), None)
    if ram and ram.free_bytes is not None and ram.free_bytes < 4 * GiB:
        diags.append(Diagnostic(level="warning", code="low_free_ram",
                                message=f"Only {ram.free_bytes / GiB:.1f} GiB host RAM is free right now.",
                                fix="Close large apps before contributing, or lower --max-memory."))
    if osi.system == "windows":
        diags.extend(windows_diagnostics())
    try:
        disk_free = shutil.disk_usage(config.paths().home if config.paths().home.exists() else Path.home()).free
    except OSError:
        disk_free = None
    from ..installers import status_all  # lazy: installers import config only, but keep inventory import-light

    return HostInventory(hostname=socket.gethostname(), os=osi, cpu=cpu, memory_pools=pools,
                         devices=devices, disk_free_bytes=disk_free, network=netutil.list_interfaces(), engine=eng,
                         collected_at=utcnow(), diagnostics=diags,
                         extra_engines=status_all() if include_engine else [])


def fresh_measurements(inv: HostInventory) -> tuple[list[MemoryPool], list[Device]]:
    """Cheap re-measurement for heartbeats: RAM via psutil and NVIDIA VRAM via nvidia-smi."""
    available = available_ram_bytes()
    pools = [p.model_copy(update={"free_bytes": available, "measured_at": utcnow()})
             if p.id in ("ram", "uma") else p for p in inv.memory_pools]
    gpus = {g["name"]: g for g in nvidia_gpus()} if any(d.backend == "cuda" for d in inv.devices) else {}
    devices = []
    for d in inv.devices:
        g = gpus.get(d.name)
        devices.append(d.model_copy(update={"free_bytes": g["free"]}) if g else d)
    pools = [p.model_copy(update={"free_bytes": gpus[d.name]["free"], "measured_at": utcnow()})
             if p.kind == "discrete_vram" and (d := next((x for x in devices if x.pool_id == p.id), None))
             and d.name in gpus else p for p in pools]
    return pools, devices
