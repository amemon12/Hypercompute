"""Physical observations of one engine process: working set, private commit, I/O and system memory over time.

Windows reports several different "memory" numbers and they are not interchangeable: the working set is what is
resident right now, private commit is what the process could need (including pages the OS trimmed to the page file
or compressed store), and file-cache pages of a mapped GGUF belong to neither process number. Disk reads are sampled
system wide (physical reads by every process) and per process (ReadFile bytes, which include unbuffered reads but not
page faults on a mapping). Each sample keeps all of them, with its timestamp, instead of collapsing to one figure.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import asdict, dataclass, field

import psutil


@dataclass(frozen=True)
class Sample:
    t: float  # seconds since the sampler started
    working_set: int
    private: int
    page_faults: int
    proc_read_bytes: int
    sys_available: int
    sys_disk_read_bytes: int


@dataclass
class ProcessSampler:
    """Background sampler; `stop()` returns the samples. Never raises into the caller's thread."""

    pid: int
    interval_s: float = 0.1
    min_available: int = 0  # 0 disables the pressure guard
    on_pressure: object = None  # callable() run once when available memory drops below min_available
    samples: list[Sample] = field(default_factory=list)
    error: str | None = None
    pressure_tripped: bool = False

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"sampler-{self.pid}", daemon=True)
        self._t0 = time.perf_counter()

    def start(self) -> ProcessSampler:
        self._thread.start()
        return self

    def stop(self) -> list[Sample]:
        self._stop.set()
        self._thread.join(timeout=5)
        return list(self.samples)

    def _run(self) -> None:
        try:
            proc = psutil.Process(self.pid)
            while not self._stop.is_set():
                try:
                    mem = proc.memory_info()
                    io = proc.io_counters()
                except psutil.Error:
                    break
                disk = psutil.disk_io_counters()
                avail = psutil.virtual_memory().available
                self.samples.append(Sample(
                    t=round(time.perf_counter() - self._t0, 3), working_set=mem.rss,
                    private=int(getattr(mem, "private", mem.vms)), page_faults=int(getattr(mem, "num_page_faults", 0)),
                    proc_read_bytes=io.read_bytes, sys_available=avail,
                    sys_disk_read_bytes=disk.read_bytes if disk else 0))
                if self.min_available and avail < self.min_available and not self.pressure_tripped:
                    self.pressure_tripped = True
                    if callable(self.on_pressure):
                        self.on_pressure()
                self._stop.wait(self.interval_s)
        except Exception as exc:  # noqa: BLE001 - recorded for the report, the sampler must not kill the run
            self.error = f"{type(exc).__name__}: {exc}"


def summarize(samples: list[Sample], t0: float = 0.0, t1: float | None = None) -> dict:
    """Peaks, minima and deltas over [t0, t1] of a sample list."""
    window = [s for s in samples if s.t >= t0 and (t1 is None or s.t <= t1)]
    if not window:
        return {"samples": 0}
    first, last = window[0], window[-1]
    return {
        "samples": len(window),
        "seconds": round(last.t - first.t, 3),
        "working_set_peak": max(s.working_set for s in window),
        "working_set_median": int(statistics.median(s.working_set for s in window)),
        "private_peak": max(s.private for s in window),
        "sys_available_min": min(s.sys_available for s in window),
        "page_faults": last.page_faults - first.page_faults,
        "proc_read_bytes": last.proc_read_bytes - first.proc_read_bytes,
        "sys_disk_read_bytes": last.sys_disk_read_bytes - first.sys_disk_read_bytes,
    }


def as_dicts(samples: list[Sample]) -> list[dict]:
    return [asdict(s) for s in samples]
