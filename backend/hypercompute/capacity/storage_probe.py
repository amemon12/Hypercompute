"""Measured storage throughput for the streaming tier: unbuffered reads of a real model file.

The planner needs the SSD's sustained throughput as hcstream will see it (FILE_FLAG_NO_BUFFERING, aligned chunks,
several readers), not the drive's advertised peak and not a cached re-read. This probe reads byte ranges of an
existing GGUF with the same flags, so nothing is written and the file cache is neither used nor polluted.
Windows only; other hosts report `supported=False`.
"""

from __future__ import annotations

import ctypes
import os
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ALIGN = 4096
MAX_CALIBRATION_BYTES = 4 << 30  # a calibration reads at most this much in total, over all of its runs
GENERIC_READ = 0x80000000
FILE_SHARE_ALL = 0x7
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
MEM_COMMIT_RESERVE = 0x3000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04


class _Overlapped(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", ctypes.c_uint32),
                ("OffsetHigh", ctypes.c_uint32), ("hEvent", ctypes.c_void_p)]


@dataclass(frozen=True)
class StorageMeasurement:
    path: str
    readers: int
    chunk_bytes: int
    bytes_read: int
    seconds: float
    supported: bool = True
    error: str | None = None

    @property
    def bytes_per_s(self) -> float:
        return self.bytes_read / self.seconds if self.seconds > 0 else 0.0


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = ctypes.c_void_p
    k.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.c_void_p]
    k.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                           ctypes.POINTER(_Overlapped)]
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def measure(path: Path, *, total_bytes: int, readers: int = 4, chunk_bytes: int = 8 << 20,
            start_offset: int = 0) -> StorageMeasurement:
    """Read `total_bytes` of `path` from `start_offset` with `readers` threads and unbuffered aligned chunks."""
    if os.name != "nt":
        return StorageMeasurement(str(path), readers, chunk_bytes, 0, 0.0, supported=False,
                                  error="unbuffered probe is implemented for Windows only")
    if chunk_bytes % ALIGN or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive multiple of 4096")
    size = path.stat().st_size
    start = start_offset // ALIGN * ALIGN
    end = min(size, start + total_bytes) // ALIGN * ALIGN
    if end <= start:
        raise ValueError("nothing to read in the requested range")
    k = _kernel32()
    lock = threading.Lock()
    cursor = [start]
    done = [0]
    errors: list[str] = []

    def work() -> None:
        h = k.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
        if h in (None, ctypes.c_void_p(-1).value):
            errors.append(f"CreateFileW failed: {ctypes.get_last_error()}")
            return
        buf = k.VirtualAlloc(None, chunk_bytes, MEM_COMMIT_RESERVE, PAGE_READWRITE)
        try:
            while True:
                with lock:
                    off = cursor[0]
                    if off >= end:
                        return
                    cursor[0] = off + chunk_bytes
                want = min(chunk_bytes, end - off)
                ov = _Overlapped(0, 0, off & 0xFFFFFFFF, off >> 32, None)
                got = ctypes.c_uint32(0)
                if not k.ReadFile(h, buf, want, ctypes.byref(got), ctypes.byref(ov)):
                    errors.append(f"ReadFile failed at {off}: {ctypes.get_last_error()}")
                    return
                with lock:
                    done[0] += got.value
        finally:
            k.VirtualFree(buf, 0, MEM_RELEASE)
            k.CloseHandle(h)

    threads = [threading.Thread(target=work, daemon=True) for _ in range(readers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seconds = time.perf_counter() - t0
    return StorageMeasurement(str(path), readers, chunk_bytes, done[0], seconds,
                              error="; ".join(errors[:3]) if errors else None)


# --------------------------------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class StorageCalibration:
    """Several bounded unbuffered reads of one model file; the read rate is their median."""

    path: str
    volume: str
    runs: tuple[StorageMeasurement, ...]

    @property
    def bytes_read(self) -> int:
        return sum(r.bytes_read for r in self.runs)

    @property
    def median_bytes_per_s(self) -> float:
        return statistics.median(r.bytes_per_s for r in self.runs)

    @property
    def error(self) -> str | None:
        """Why the runs do not make a measurement (a failed, unsupported or empty run), else None."""
        bad = [r.error or "unsupported on this host" for r in self.runs if r.error or not r.supported]
        if bad:
            return "; ".join(bad[:3])
        if not self.runs or any(r.bytes_read <= 0 or r.seconds <= 0 for r in self.runs):
            return "a run read nothing"
        return None


def volume_of(path: Path) -> str:
    """The volume holding `path`: an upper-case drive ("C:") on Windows, else the mount anchor ("/")."""
    resolved = path.resolve()
    return (resolved.drive or resolved.anchor).upper()


def calibration_ranges(file_bytes: int, total_bytes: int, runs: int, chunk_bytes: int) -> tuple[tuple[int, int], ...]:
    """(start offset, bytes) per run: consecutive, chunk-aligned ranges of the file whose sum stays within
    `total_bytes`, MAX_CALIBRATION_BYTES and the file. Raises ValueError when a run would read less than one chunk."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if chunk_bytes <= 0 or chunk_bytes % ALIGN:
        raise ValueError("chunk_bytes must be a positive multiple of 4096")
    if total_bytes <= 0 or total_bytes > MAX_CALIBRATION_BYTES:
        raise ValueError(f"total_bytes must be 1..{MAX_CALIBRATION_BYTES} (4 GiB), not {total_bytes}")
    budget = min(total_bytes, file_bytes // ALIGN * ALIGN)
    per_run = budget // runs // chunk_bytes * chunk_bytes
    if per_run < chunk_bytes:
        raise ValueError(f"{runs} runs of at least one {chunk_bytes}-byte chunk do not fit in {budget} bytes")
    return tuple((i * per_run, per_run) for i in range(runs))


def calibrate(path: Path, *, total_bytes: int, runs: int = 3, readers: int = 4, chunk_bytes: int = 8 << 20,
              measure_fn: Callable[..., StorageMeasurement] | None = None) -> StorageCalibration:
    """`runs` unbuffered reads of consecutive ranges of `path`, at most `total_bytes` (and 4 GiB) in total.

    Unbuffered reads never come from the file cache, so each run reads the drive; distinct ranges also keep a drive's
    own read cache from serving a repeat. `measure_fn` replaces `measure` in tests.
    """
    probe = measure_fn or measure
    ranges = calibration_ranges(path.stat().st_size, total_bytes, runs, chunk_bytes)
    results: Sequence[StorageMeasurement] = [
        probe(path, total_bytes=n, readers=readers, chunk_bytes=chunk_bytes, start_offset=start)
        for start, n in ranges]
    return StorageCalibration(path=str(path), volume=volume_of(path), runs=tuple(results))
