"""Host monitor for scripts/capacity/run_big.py: a background sampler of this machine during a gateway-driven run.

Each sample keeps host memory from GetPerformanceInfo (available, commit charge and limit, system file cache),
system-wide physical disk reads (psutil.disk_io_counters, every process), free space on the system drive (where the
Windows pagefile grows), the engine process's working set, private bytes, ReadFile bytes and page faults, and GPU
memory used (nvidia-smi, every `gpu_every`-th sample because the query is slow). A guard calls `on_trip(reason)` once,
from the sampler thread, when available RAM or the system drive's free space falls below its floor.
Probes are injectable so the sampler runs in tests without hardware.

On macOS (MacProbes) available RAM is free plus file-backed pages from `vm_stat` (no sudo): the pages macOS hands out
without compressing or swapping, so the memory-mapped GGUF counts as available, as it should. psutil's macOS figure
(free + inactive) leaves out active file-backed pages, so mapped experts would look like used RAM there. The
file-backed pages are also the system_cache reading. Swap used, from psutil, is the macOS analogue of a growing
pagefile (private memory past RAM goes to swap files on the boot volume). macOS has no commit counters, no
per-process private or read bytes, and no GPU memory reading without sudo (powermetrics), so those fields stay None.
HostProfile names the platform, the volume the drive floor watches (the system drive on Windows, the models volume on
macOS) and the run.json source line.

It also reads what the engine itself reports about this host: the placement flags in llama-server's argv
(engine_flags) and the per-buffer sizes in its load log (log_evidence), which tell file-backed CPU weights
(CPU_Mapped, file cache) from private host copies (CPU, CPU_REPACK, CUDA_Host: commit).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from hypercompute.adapters.base import EngineAdapterError  # noqa: E402
from hypercompute.adapters.vllm import query_gpus  # noqa: E402
from hypercompute.capacity.pools import host_memory_snapshot  # noqa: E402

GIB = 1 << 30
T = TypeVar("T")
# llama-server b11770 flags that decide placement (checked against `llama-server --help`); value flags take one value.
VALUE_FLAGS = frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers", "-dev", "--device", "-ts", "--tensor-split",
                         "-sm", "--split-mode", "--rpc", "-ncmoe", "--n-cpu-moe", "-ot", "--override-tensor",
                         "-lm", "--load-mode", "-c", "--ctx-size", "-fa", "--flash-attn", "-ctk", "--cache-type-k",
                         "-ctv", "--cache-type-v", "-fit", "--fit", "-cram", "--cache-ram", "-np", "--parallel"})
SWITCH_FLAGS = frozenset({"-cmoe", "--cpu-moe", "-nr", "--no-repack", "--repack", "--no-host", "--no-op-offload",
                          "-kvu", "--kv-unified"})
CANONICAL = {"-ngl": "--n-gpu-layers", "--gpu-layers": "--n-gpu-layers", "-dev": "--device", "-ts": "--tensor-split",
             "-sm": "--split-mode", "-ncmoe": "--n-cpu-moe", "-ot": "--override-tensor", "-lm": "--load-mode",
             "-c": "--ctx-size", "-fa": "--flash-attn", "-ctk": "--cache-type-k", "-ctv": "--cache-type-v",
             "-cmoe": "--cpu-moe", "-nr": "--no-repack", "-fit": "--fit", "-cram": "--cache-ram", "-np": "--parallel",
             "-kvu": "--kv-unified"}
LOG_MARKERS = ("buffer size", "offloaded", "repack", "overrid", "hcstream:", "load_tensors:", "llama_kv_cache:")
BUFFER_RE = re.compile(r"(\S+) model buffer size\s*=\s*([\d.]+) MiB")
PRIVATE_HOST_BUFFERS = ("CPU", "CPU_REPACK", "CUDA_Host")  # private host copies: they count toward commit
# llama.cpp b11770 Metal buffer types (ggml-metal.cpp): MTLn_Mapped wraps the memory-mapped GGUF (file-backed); MTLn
# and MTLn_Private are Metal allocations, which on a unified-memory Mac are private RAM.
METAL_MAPPED_RE = re.compile(r"MTL\d+_Mapped")
METAL_PRIVATE_RE = re.compile(r"MTL\d+(_Private)?")
# server slots and KV layout as llama-server logs them at the pinned commit: server-context.cpp ("initializing,
# n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'"), llama-context.cpp (n_seq_max, n_ctx, n_ctx_seq) and
# llama-kv-cache.cpp ("size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs)": n_seq_max / n_stream)
SLOTS_RE = re.compile(r"n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(true|false)'")
CONTEXT_RE = re.compile(r"llama_context: (n_seq_max|n_ctx|n_ctx_seq)\s+=\s+(\d+)")
KV_SIZE_RE = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB "
                        r"\(\s*(\d+) cells,\s*(\d+) layers,\s*(\d+)/(\d+) seqs\)")
WINDOWS_SOURCE = ("hostmon: GetPerformanceInfo (available, commit, file cache), psutil disk_io_counters (system-wide "
                  "physical reads, every process), shutil.disk_usage (system drive), psutil engine process counters "
                  "(ReadFile bytes exclude mmap page-fault reads), nvidia-smi (all GPUs)")
MACOS_SOURCE = ("hostmon (macOS): vm_stat (available = free + file-backed pages, what macOS reclaims without "
                "compressing or swapping, so the memory-mapped GGUF counts as available; system_cache = file-backed "
                "pages), psutil swap_memory (swap used: macOS writes private memory past RAM to swap files on the boot "
                "volume), psutil disk_io_counters (system-wide physical reads, every process), shutil.disk_usage (the "
                "volume holding the models), psutil engine rss and page faults. Not recorded on macOS (None): commit "
                "charge and limit (no such counters), engine private and read bytes (psutil has no per-process private "
                "or I/O counters on macOS), GPU memory (no nvidia-smi; Metal usage needs powermetrics with sudo, so "
                "the engine log's MTL buffers are the record)")
VM_STAT = "/usr/bin/vm_stat"
VM_STAT_LINE = re.compile(r'^"?(?P<label>[^":]+)"?:\s+(?P<value>\d+)\.$')  # `printf("%-30s %16llu.\n", ...)`
VM_PAGE_SIZE = re.compile(r"\(page size of (\d+) bytes\)")


@dataclass(frozen=True)
class Sample:
    t: float  # seconds since the monitor started
    available: int | None = None
    commit_charge: int | None = None
    commit_limit: int | None = None
    system_cache: int | None = None
    disk_read_bytes: int | None = None
    drive_free: int | None = None
    engine_rss: int | None = None
    engine_private: int | None = None
    engine_read_bytes: int | None = None
    engine_page_faults: int | None = None
    gpu_used: int | None = None
    swap_used: int | None = None  # macOS only (MacProbes); Windows growth shows in the commit limit instead


class Probes(Protocol):
    def host_memory(self) -> Mapping[str, int | None]: ...
    def disk_read_bytes(self) -> int | None: ...
    def drive_free(self) -> int | None: ...
    def process(self, pid: int) -> Mapping[str, int] | None: ...
    def gpu_used(self) -> int | None: ...


def _try(fn: Callable[[], T], *errors: type[BaseException]) -> T | None:
    try:
        return fn()
    except errors:
        return None


@dataclass(frozen=True)
class SystemProbes:
    drive: str  # the volume the Windows pagefile grows on

    def host_memory(self) -> Mapping[str, int | None]:
        hm = host_memory_snapshot()  # never raises; a failed query leaves the fields None
        return {"available": hm.available, "commit_charge": hm.commit_charge, "commit_limit": hm.commit_limit,
                "system_cache": hm.system_cache}

    def disk_read_bytes(self) -> int | None:
        counters = _try(psutil.disk_io_counters, OSError, RuntimeError)
        return int(counters.read_bytes) if counters else None

    def drive_free(self) -> int | None:
        usage = _try(lambda: shutil.disk_usage(self.drive), OSError)
        return usage.free if usage else None

    def process(self, pid: int) -> Mapping[str, int] | None:
        try:
            proc = psutil.Process(pid)
            mem, io = proc.memory_info(), proc.io_counters()
        except psutil.Error:
            return None
        return {"rss": mem.rss, "private": int(getattr(mem, "private", mem.vms)), "read_bytes": io.read_bytes,
                "page_faults": int(getattr(mem, "num_page_faults", 0))}

    def gpu_used(self) -> int | None:
        rows = _try(query_gpus, EngineAdapterError)
        return sum(g.used_bytes for g in rows) if rows else None


PSUTIL_ERRORS = (OSError, RuntimeError, psutil.Error)
VM_STAT_ERRORS = (OSError, subprocess.SubprocessError, ValueError, KeyError)


def vm_stat_text() -> str:
    """One run of macOS `vm_stat` (no sudo): page counts and the kernel page size."""
    return subprocess.run([VM_STAT], capture_output=True, text=True, timeout=5, check=True).stdout


def parse_vm_stat(text: str) -> dict[str, int]:
    """vm_stat's counts by label ("Pages free", "File-backed pages", ...) plus "page size" in bytes."""
    size = VM_PAGE_SIZE.search(text)
    if size is None:
        raise ValueError("vm_stat printed no page size")
    matches = (VM_STAT_LINE.match(line.strip()) for line in text.splitlines())
    return {**{m["label"]: int(m["value"]) for m in matches if m}, "page size": int(size.group(1))}


def reclaimable(pages: Mapping[str, int]) -> tuple[int, int]:
    """(available, file cache) bytes. Available is free plus pageable file-backed pages, what XNU's memorystatus counts
    as available on jetsam systems (VM_CHECK_MEMORYSTATUS): pages macOS hands out without compressing or swapping.
    Wired and anonymous pages are not in it. KeyError when a line is missing."""
    page, cached = pages["page size"], pages["File-backed pages"]
    return (pages["Pages free"] + cached) * page, cached * page


@dataclass(frozen=True)
class MacProbes(SystemProbes):
    """macOS: `drive` is the volume holding the models. Memory from vm_stat (psutil's macOS available, free + inactive,
    leaves out active file-backed pages, so mapped experts would look like used RAM and trip the floor); swap from
    psutil. psutil has no Process.io_counters on macOS, so SystemProbes.process would raise AttributeError there and
    lose every sample once the engine is attached."""

    vm_stat: Callable[[], str] = vm_stat_text

    def host_memory(self) -> Mapping[str, int | None]:
        swap = _try(psutil.swap_memory, *PSUTIL_ERRORS)
        mem = _try(lambda: reclaimable(parse_vm_stat(self.vm_stat())), *VM_STAT_ERRORS)
        available, cache = mem if mem else (None, None)  # None: the floor cannot act, and guard_status says so
        return {"available": available, "commit_charge": None, "commit_limit": None, "system_cache": cache,
                "swap_used": int(swap.used) if swap else None}

    def process(self, pid: int) -> Mapping[str, int] | None:
        try:
            mem = psutil.Process(pid).memory_info()  # macOS: rss, vms, pfaults, pageins
        except psutil.Error:
            return None
        # vms counts the whole mapped GGUF and reserved address space, so it is no measure of private memory
        out = {"rss": int(mem.rss)}
        faults = getattr(mem, "pfaults", None)
        return {**out, "page_faults": int(faults)} if faults is not None else out

    def gpu_used(self) -> int | None:
        return None  # Metal memory needs powermetrics (sudo); the engine log's MTL buffers are the record


@dataclass(frozen=True)
class HostProfile:
    """What this OS lets the monitor watch: the volume the drive floor guards and the run.json source line."""

    platform: str  # "windows", "macos", or sys.platform elsewhere
    drive: str
    drive_label: str
    drive_note: str
    source: str


def _existing(path: Path) -> Path:
    """The path itself or its nearest existing parent, so disk_usage works before the models folder exists."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor or os.sep)


def host_profile(platform: str, home: Path, environ: Mapping[str, str] | None = None) -> HostProfile:
    """Windows: the system drive, where the pagefile grows. macOS: the volume holding the gateway's models (on the
    internal disk it shares free space with the swap files). Elsewhere: the root volume, as before."""
    env = os.environ if environ is None else environ
    if platform == "win32":
        return HostProfile("windows", env.get("SystemDrive", "C:") + os.sep, "system drive", "pagefile growth",
                           WINDOWS_SOURCE)
    if platform == "darwin":
        return HostProfile("macos", str(_existing(home / "models")), "models volume",
                           "the models and the swap files fill it", MACOS_SOURCE)
    return HostProfile(platform, "/", "system drive", "pagefile growth", WINDOWS_SOURCE)


def probes_for(profile: HostProfile) -> Probes:
    return MacProbes(profile.drive) if profile.platform == "macos" else SystemProbes(profile.drive)


class Monitor:
    """Background sampler with a memory/disk guard. Samples and marks share one clock (seconds since start)."""

    def __init__(self, probes: Probes, *, interval_s: float, gpu_every: int, min_available: int,
                 min_drive_free: int, on_trip: Callable[[str], None], clock: Callable[[], float] = time.perf_counter,
                 max_commit_growth: int | None = 256 << 20, max_swap_growth: int | None = None,
                 drive_label: str = "system drive", drive_note: str = "pagefile growth"):
        self._probes, self._interval, self._gpu_every = probes, interval_s, max(gpu_every, 1)
        self._min_available, self._min_drive_free, self._on_trip = min_available, min_drive_free, on_trip
        # Windows raises the commit limit only by growing a pagefile: growth past this means private memory went past
        # RAM and the pagefile is filling the system drive (the 2026-10-21 failure). None turns the check off.
        self._max_commit_growth, self._commit_limit0 = max_commit_growth, None
        # macOS has no commit limit: private memory past RAM is compressed, then written to swap files on the boot
        # volume, so swap used growing past this is the same failure. None turns the check off.
        self._max_swap_growth, self._swap0 = max_swap_growth, None
        self._drive_label, self._drive_note = drive_label, drive_note
        self._clock, self._t0 = clock, clock()
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = threading.Thread(target=self._run, name="hostmon", daemon=True)
        self._pid: int | None = None
        self._count = 0
        self.samples: list[Sample] = []
        self.marks: list[tuple[str, float]] = []
        self.tripped: str | None = None
        self.error: str | None = None

    def now(self) -> float:
        return round(self._clock() - self._t0, 3)

    def attach(self, pid: int) -> None:
        self._pid = pid

    def mark(self, label: str) -> None:
        with self._lock:
            self.marks.append((label, self.now()))

    def start(self) -> Monitor:
        """Takes the first sample (and runs the guard) before returning, then samples in the background."""
        self._safe_sample()
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stops the thread and takes a last sample, so windows that end now are bracketed."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        self._safe_sample()

    def sample_once(self) -> Sample:
        hm = self._probes.host_memory()
        proc = self._probes.process(self._pid) if self._pid else None
        gpu = self._probes.gpu_used() if self._count % self._gpu_every == 0 else None
        self._count += 1
        proc = proc or {}
        s = Sample(t=self.now(), available=hm.get("available"), commit_charge=hm.get("commit_charge"),
                   commit_limit=hm.get("commit_limit"), system_cache=hm.get("system_cache"),
                   disk_read_bytes=self._probes.disk_read_bytes(), drive_free=self._probes.drive_free(),
                   engine_rss=proc.get("rss"), engine_private=proc.get("private"),
                   engine_read_bytes=proc.get("read_bytes"), engine_page_faults=proc.get("page_faults"),
                   gpu_used=gpu, swap_used=hm.get("swap_used"))
        with self._lock:
            self.samples.append(s)
        self._guard(s)
        return s

    def _trip_reason(self, s: Sample) -> str | None:
        if s.commit_limit is not None and self._commit_limit0 is None:
            self._commit_limit0 = s.commit_limit  # the first reading is the baseline
        if s.swap_used is not None and self._swap0 is None:
            self._swap0 = s.swap_used
        if s.available is not None and s.available < self._min_available:
            return f"available RAM {s.available / GIB:.2f} GiB below {self._min_available / GIB:.2f} GiB"
        if s.drive_free is not None and s.drive_free < self._min_drive_free:
            return (f"{self._drive_label} free {s.drive_free / GIB:.2f} GiB below {self._min_drive_free / GIB:.2f} "
                    f"GiB ({self._drive_note})")
        return self._growth_reason(s)

    def _growth_reason(self, s: Sample) -> str | None:
        base, limit, cap = self._commit_limit0, s.commit_limit, self._max_commit_growth
        if cap is not None and base is not None and limit is not None and limit - base > cap:
            return (f"commit limit grew {(limit - base) / GIB:.2f} GiB to {limit / GIB:.2f} GiB (the pagefile is "
                    "growing on the system drive)")
        base, used, cap = self._swap0, s.swap_used, self._max_swap_growth
        if cap is not None and base is not None and used is not None and used - base > cap:
            return (f"swap used grew {(used - base) / GIB:.2f} GiB to {used / GIB:.2f} GiB (private memory is past "
                    "RAM and macOS is writing swap files to the boot volume)")
        return None

    def guard_status(self) -> dict[str, Any]:
        """Which guards could act in this run: a floor of 0 or a cap of None is off, and a guard whose counter this
        OS never reported (commit on macOS, swap on Windows) could not trip."""
        with self._lock:
            samples = list(self.samples)

        def entry(limit: int | None, attr: str, floor: bool) -> dict[str, Any]:
            on = (limit or 0) > 0 if floor else limit is not None
            seen = any(getattr(s, attr) is not None for s in samples)
            why = None if on and seen else ("turned off" if not on else f"this host reported no {attr} reading")
            return {"limit_bytes": limit, "active": on and seen, "inactive_because": why}

        return {"min_available": entry(self._min_available, "available", True),
                "min_drive_free": {**entry(self._min_drive_free, "drive_free", True), "volume": self._drive_label},
                "max_commit_growth": entry(self._max_commit_growth, "commit_limit", False),
                "max_swap_growth": entry(self._max_swap_growth, "swap_used", False)}

    def _guard(self, s: Sample) -> None:
        """Trips once even when the sampler thread and stop() sample at the same time."""
        reason = self._trip_reason(s)
        if reason is None:
            return
        with self._lock:
            if self.tripped:
                return
            self.tripped = reason
            self.marks.append((f"guard: {reason}", self.now()))
        self._on_trip(reason)

    def _safe_sample(self) -> None:
        try:
            self.sample_once()
        except Exception as exc:  # noqa: BLE001 - recorded in the evidence; the run itself must go on
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._safe_sample()


def _values(samples: Sequence[Sample], attr: str) -> list[int]:
    return [v for v in (getattr(s, attr) for s in samples) if v is not None]


def counter_delta(samples: Sequence[Sample], attr: str, t0: float, t1: float) -> int | None:
    """Counter growth over [t0, t1], bracketed by the last sample at or before t0 and the first at or after t1 (falls
    back to the samples inside the window when the run starts or ends inside it)."""
    have = sorted((s for s in samples if getattr(s, attr) is not None), key=lambda s: s.t)
    if not have:
        return None
    # the start is the last sample at or before t0, else the first one after it; the end is the first sample at or
    # after t1, else the last one before it (a window that ends after the newest sample, such as a chat that just
    # finished, is closed by the newest sample, never by the oldest)
    start = next((s for s in reversed(have) if s.t <= t0), have[0])
    end = next((s for s in have if s.t >= t1), have[-1])
    return int(getattr(end, attr) - getattr(start, attr)) if end.t >= start.t else 0


def window(samples: Sequence[Sample], t0: float, t1: float) -> dict[str, Any]:
    """Peaks and minima of the samples inside [t0, t1], and bracketed counter growth."""
    inside = [s for s in samples if t0 <= s.t <= t1]

    def ext(fn: Callable[[list[int]], int], attr: str) -> int | None:
        vals = _values(inside, attr)
        return fn(vals) if vals else None

    return {"t0": t0, "t1": t1, "samples": len(inside), "available_min": ext(min, "available"),
            "commit_charge_peak": ext(max, "commit_charge"), "commit_limit_peak": ext(max, "commit_limit"),
            "system_cache_peak": ext(max, "system_cache"), "drive_free_min": ext(min, "drive_free"),
            "engine_rss_peak": ext(max, "engine_rss"), "engine_private_peak": ext(max, "engine_private"),
            "gpu_used_peak": ext(max, "gpu_used"), "swap_used_min": ext(min, "swap_used"),
            "swap_used_peak": ext(max, "swap_used"),
            "disk_read_bytes": counter_delta(samples, "disk_read_bytes", t0, t1),
            "engine_read_bytes": counter_delta(samples, "engine_read_bytes", t0, t1),
            "engine_page_faults": counter_delta(samples, "engine_page_faults", t0, t1)}


# ----------------------------------------------------------------------------- engine evidence


def engine_flags(args: Sequence[str]) -> dict[str, Any]:
    """Placement flags from the engine argv, canonical names; repeated value flags (-ot) keep every value."""
    flags: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        name = CANONICAL.get(arg, arg)
        # a value flag takes the next argument unless that is itself a known flag (--fit's value is optional)
        if arg in VALUE_FLAGS and i + 1 < len(args) and args[i + 1] not in VALUE_FLAGS | SWITCH_FLAGS:
            value = args[i + 1]
            flags[name] = [*flags.get(name, []), value] if name == "--override-tensor" else value
            i += 2
            continue
        if arg in SWITCH_FLAGS:
            flags[name] = True
        i += 1
    # no_repack and no_host decide whether CPU experts get private copies (commit); fit "off" keeps the planned split
    moe = {"cpu_moe": bool(flags.get("--cpu-moe")), "n_cpu_moe": flags.get("--n-cpu-moe"),
           "override_tensor": flags.get("--override-tensor", []), "no_repack": bool(flags.get("--no-repack")),
           "no_host": bool(flags.get("--no-host")), "no_op_offload": bool(flags.get("--no-op-offload")),
           "load_mode": flags.get("--load-mode"), "fit": flags.get("--fit")}
    return {"flags": flags, "moe": moe}


def log_evidence(path: Path, limit: int = 160) -> dict[str, Any]:
    """Buffer lines from the engine's own load log: per-buffer MiB, and whether CPU weights stayed file-backed."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"available": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    lines = [ln.strip() for ln in text.splitlines() if any(m in ln for m in LOG_MARKERS)]
    buffers: dict[str, float] = {}
    for ln in lines:
        m = BUFFER_RE.search(ln)
        if m:
            buffers[m.group(1)] = round(buffers.get(m.group(1), 0.0) + float(m.group(2)), 2)
    private = sum(v for k, v in buffers.items() if k in PRIVATE_HOST_BUFFERS)
    # a Mac host: Metal buffers over the mapped GGUF stay file-backed; other Metal buffers are private unified RAM
    metal_mapped = sum(v for k, v in buffers.items() if METAL_MAPPED_RE.fullmatch(k))
    metal_private = sum(v for k, v in buffers.items() if METAL_PRIVATE_RE.fullmatch(k))
    return {"available": True, "path": str(path), "source": "the engine's load log", "model_buffers_mib": buffers,
            "file_backed_host_mib": buffers.get("CPU_Mapped", 0.0), "private_host_mib": round(private, 2),
            "metal_mapped_mib": round(metal_mapped, 2), "metal_private_mib": round(metal_private, 2),
            "slots": slot_evidence(text), "lines": lines[:limit], "lines_dropped": max(0, len(lines) - limit)}


def slot_evidence(text: str) -> dict[str, Any]:
    """Server slots and KV layout from the engine log (None where a line is missing): n_slots, n_ctx_slot and
    kv_unified from llama-server, n_seq_max / n_ctx / n_ctx_seq from the context, and each KV cache's cells and
    n_seq_max / n_stream (one stream under -kvu)."""
    m = SLOTS_RE.search(text)
    server = {"n_slots": int(m[1]), "n_ctx_slot": int(m[2]), "kv_unified": m[3] == "true"} if m else {
        "n_slots": None, "n_ctx_slot": None, "kv_unified": None}
    context = {name: int(value) for name, value in reversed(CONTEXT_RE.findall(text))}  # the first line of each wins
    caches = [{"mib": float(c[0]), "cells": int(c[1]), "layers": int(c[2]), "n_seq_max": int(c[3]),
               "n_stream": int(c[4])} for c in KV_SIZE_RE.findall(text)]
    return {**server, **{k: context.get(k) for k in ("n_seq_max", "n_ctx", "n_ctx_seq")}, "kv_caches": caches}
