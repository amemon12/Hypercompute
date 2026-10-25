"""llama.cpp RPC cache seeding: FNV-1a 64 as ggml-rpc.cpp computes it, ranged downloads of one tensor into
<LLAMA_CACHE>/rpc/<%016x hash>, and the job runner that keeps a peer's cache within its disk limits.

ggml-rpc.cpp (b11770): the client sends SET_TENSOR_HASH (FNV-1a 64 of the tensor bytes) only for WEIGHTS tensors over
HASH_THRESHOLD; the server reads cache_dir/"%016" PRIx64 on a hit and takes the file's own size, so a short file
would silently corrupt a tensor. Files here are therefore written to a part file, verified, fsynced and renamed.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import shutil
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Literal

import httpx
import psutil

from .. import __version__
from ..schemas import SeedFailure, SeedProgress, SeedSpec, utcnow
from ..units import GiB, MiB, fmt_bytes

log = logging.getLogger("hypercompute.agent.seed")

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1
HASH_THRESHOLD = 10 * MiB  # ggml-rpc.cpp HASH_THRESHOLD: the client hashes only tensors with size > this
MIN_FREE_BYTES = 20 * GiB  # seeding never leaves less than this free on the cache volume
CHUNK = 4 * MiB
_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
_CACHE_FILE_RE = re.compile(r"[0-9a-f]{16}")  # a finished cache entry; part and stop files start with "."


class ManagedError(RuntimeError):
    """A host request this agent will not run; the text goes back to the host in the acknowledgement."""


@dataclass(frozen=True)
class SeedSource:
    file: str
    url: str
    size: int


# --------------------------------------------------------------------------- FNV-1a 64 (ggml-rpc.cpp fnv_hash)


def fnv1a64(data: bytes | bytearray | memoryview, h: int = FNV_OFFSET) -> int:
    """FNV-1a 64 of `data`, continued from `h`. The product is masked every 8 bytes only: its low 64 bits depend only
    on the low 64 bits of the factors and the xor changes only the low byte, so the result is exact (about 1.25x the
    speed of masking every byte in CPython)."""
    p, m = FNV_PRIME, _MASK64
    n8 = len(data) - len(data) % 8
    it = iter(data[:n8])
    for a, b, c, d, e, f, g, k in zip(it, it, it, it, it, it, it, it, strict=True):
        h = (((((((((((((((((h ^ a) * p) ^ b) * p) ^ c) * p) ^ d) * p) ^ e) * p) ^ f) * p) ^ g) * p) ^ k) * p) & m)
    for x in data[n8:]:
        h = ((h ^ x) * p) & m
    return h & m


def fnv1a64_file(path: str | Path, offset: int, nbytes: int, stop_file: str | None = None) -> int:
    """FNV-1a 64 over exactly [offset, offset + nbytes) of a file; a short file is an error, never a shorter hash."""
    h, left = FNV_OFFSET, nbytes
    with open(path, "rb") as fh:
        fh.seek(offset)
        while left > 0:
            _check_stop(stop_file)
            chunk = fh.read(min(CHUNK, left))
            if not chunk:
                raise ManagedError(f"{Path(path).name} ends {left} bytes before the tensor range does")
            h = fnv1a64(chunk, h)
            left -= len(chunk)
    return h


def cache_name(fnv: int) -> str:
    return f"{fnv:016x}"


def part_path(cache_dir: Path, fnv: int) -> Path:
    return cache_dir / f".seed-{cache_name(fnv)}.part"


def lock_path(cache_dir: Path, fnv: int) -> Path:
    return cache_dir / f".seed-{cache_name(fnv)}.lock"


# --------------------------------------------------------------------------- one tensor: ranged download -> cache


@dataclass(frozen=True)
class FetchTask:
    url: str
    offset: int
    nbytes: int
    fnv64: int
    cache_dir: str
    stop_file: str | None = None
    timeout_s: float = 60.0
    retries: int = 6


@dataclass(frozen=True)
class FetchResult:
    fnv64: int
    status: Literal["written", "present", "failed", "cancelled"]
    downloaded: int = 0
    error: str | None = None


@dataclass
class _Cursor:
    h: int
    have: int


class SeedStopped(Exception):
    pass


class _FetchError(Exception):
    def __init__(self, message: str, *, retry: bool) -> None:
        super().__init__(message)
        self.retry = retry


def _check_stop(stop_file: str | None) -> None:
    if stop_file is not None and os.path.exists(stop_file):
        raise SeedStopped


def short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def _complete(path: Path, nbytes: int) -> bool:
    """A cache file of the right size is complete: the name is the hash of the content, and seeding writes only
    whole verified files. A wrong size (an rpc-server write cut short) is replaced."""
    try:
        return path.stat().st_size == nbytes
    except FileNotFoundError:
        return False


def new_http_client(timeout_s: float = 60.0) -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(timeout_s, connect=20.0),
                        headers={"User-Agent": f"hypercompute/{__version__} rpc-cache-seed"})


def _check_response(r: httpx.Response, start: int, end: int) -> None:
    if r.url.scheme != "https":
        raise _FetchError("the download was redirected to a non-https URL", retry=False)
    if r.status_code in (408, 429) or r.status_code >= 500:
        raise _FetchError(f"HTTP {r.status_code}", retry=True)
    if r.status_code == 200:
        raise _FetchError("the server ignored the byte range (HTTP 200)", retry=False)
    if r.status_code != 206:
        raise _FetchError(f"HTTP {r.status_code}", retry=False)
    cr = (r.headers.get("content-range") or "").strip()
    m = _CONTENT_RANGE_RE.fullmatch(cr)
    if not m or int(m[1]) != start or int(m[2]) > end:
        raise _FetchError(f"unexpected Content-Range {cr[:80]!r} for bytes {start}-{end}", retry=False)


def _stream_range(client: httpx.Client, task: FetchTask, out: IO[bytes], cur: _Cursor) -> None:
    start, end = task.offset + cur.have, task.offset + task.nbytes - 1
    with client.stream("GET", task.url, headers={"Range": f"bytes={start}-{end}"}) as r:
        _check_response(r, start, end)
        for chunk in r.iter_bytes(CHUNK):
            _check_stop(task.stop_file)
            chunk = chunk[: task.nbytes - cur.have]
            if not chunk:
                break
            out.write(chunk)
            cur.h = fnv1a64(chunk, cur.h)
            cur.have += len(chunk)
    if cur.have < task.nbytes:
        raise _FetchError("the server ended the range early", retry=True)


def _download(client: httpx.Client, task: FetchTask, part: Path, cur: _Cursor) -> int:
    """Append the missing bytes to the part file (resuming where it ends), retrying transient failures."""
    fetched, failures = 0, 0
    with part.open("ab") as out:
        while cur.have < task.nbytes:
            _check_stop(task.stop_file)
            before = cur.have
            try:
                _stream_range(client, task, out, cur)
            except (httpx.TransportError, _FetchError) as exc:
                if isinstance(exc, _FetchError) and not exc.retry:
                    raise
                failures = 0 if cur.have > before else failures + 1
                if failures > task.retries:
                    raise
                if failures:
                    time.sleep(min(30.0, 2.0 ** failures))
            finally:
                fetched += cur.have - before
        out.flush()
        os.fsync(out.fileno())
    return fetched


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return  # Windows cannot open a directory for fsync; the rename itself is journaled by NTFS
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _resume(part: Path, task: FetchTask) -> _Cursor:
    try:
        have = part.stat().st_size
    except FileNotFoundError:
        return _Cursor(FNV_OFFSET, 0)
    if have > task.nbytes:
        part.unlink()
        return _Cursor(FNV_OFFSET, 0)
    return _Cursor(fnv1a64_file(part, 0, have, task.stop_file), have)


def _try_lock(fd: int) -> bool:
    """A non-blocking exclusive OS lock, which the OS drops when the process dies, so a killed agent leaves none."""
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:  # held by another writer (BlockingIOError on POSIX, EACCES on Windows)
        return False
    return True


def _same_file(fd: int, path: Path) -> bool:
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    held = os.fstat(fd)
    return (st.st_dev, st.st_ino) == (held.st_dev, held.st_ino)


def _acquire_part_lock(path: Path) -> int | None:
    """Hold the tensor's lock file, or None when another download of it (an orphaned pool worker of an earlier
    agent, say) holds it. On POSIX the holder unlinks the file before it lets go, so a lock taken on a file that is
    no longer at `path` is dropped and taken again on the new one; Windows cannot delete a file anyone has open."""
    for _ in range(3):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            locked = _try_lock(fd)
            current = locked and (os.name == "nt" or _same_file(fd, path))
        except OSError:
            os.close(fd)
            raise
        if current:
            return fd
        os.close(fd)
        if not locked:
            return None
    return None


def _release_part_lock(path: Path, fd: int) -> None:
    """POSIX: unlink the lock file while still holding it (see _acquire_part_lock). Windows: unlock and close first,
    since a file anyone has open cannot be deleted there."""
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("could not release the seed lock file %s: %s", path.name, exc)
    finally:
        os.close(fd)
    if os.name == "nt":
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # another download opened it meanwhile and removes it when it ends
            log.debug("could not remove the seed lock file %s: %s", path.name, exc)


def fetch_tensor(task: FetchTask, client: httpx.Client | None = None) -> FetchResult:
    """Download exactly [offset, offset + nbytes) into <cache>/.seed-<hash>.part, check its FNV-1a 64, fsync and
    rename it to <cache>/<hash>. rpc-server takes any file of that name as the tensor, so only a complete, verified
    file ever gets it. An OS lock on <cache>/.seed-<hash>.lock is held throughout: each download checks only the
    hash of the bytes it streamed itself, so a second writer on the same part file would publish a mixed file. Runs
    in a pool worker process; never raises."""
    lock = lock_path(Path(task.cache_dir), task.fnv64)
    try:
        fd = _acquire_part_lock(lock)
    except OSError as exc:
        return FetchResult(task.fnv64, "failed", error=short_error(exc))
    if fd is None:
        return FetchResult(task.fnv64, "failed", error="another download of this tensor is running (an earlier "
                                                       "agent's worker?); seed again once it ends")
    try:
        return _fetch_locked(task, client)
    finally:
        _release_part_lock(lock, fd)


def _fetch_locked(task: FetchTask, client: httpx.Client | None) -> FetchResult:
    cache = Path(task.cache_dir)
    final, part = cache / cache_name(task.fnv64), part_path(cache, task.fnv64)
    fetched = 0
    try:
        if _complete(final, task.nbytes):
            return FetchResult(task.fnv64, "present")
        cur = _resume(part, task)
        own = client is None
        http = client or new_http_client(task.timeout_s)
        try:
            fetched = _download(http, task, part, cur)
        finally:
            if own:
                http.close()
        if cur.h != task.fnv64:
            part.unlink(missing_ok=True)
            return FetchResult(task.fnv64, "failed", fetched,
                               f"hash mismatch: the bytes hash to {cache_name(cur.h)} (catalog file differs?)")
        os.replace(part, final)
        _fsync_dir(cache)
        return FetchResult(task.fnv64, "written", fetched)
    except SeedStopped:
        return FetchResult(task.fnv64, "cancelled", fetched)
    except (OSError, httpx.HTTPError, _FetchError, ManagedError) as exc:
        return FetchResult(task.fnv64, "failed", fetched, short_error(exc))


def init_pool_worker() -> None:
    """Pool initializer: hashing yields the CPU to the contributor's own work and the RPC worker."""
    try:
        if os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except (OSError, AttributeError, psutil.Error) as exc:
        log.debug("could not lower the seeding worker's priority: %s", exc)


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def default_executor(workers: int) -> Executor:
    """Processes, so FNV hashing (pure Python, about 20 MB/s per core) runs on several cores."""
    try:
        return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=init_pool_worker)
    except (OSError, ValueError, NotImplementedError) as exc:
        log.warning("process pool unavailable (%s); hashing on threads", exc)
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hypercompute-seed")


def _entry_size(e: os.DirEntry[str]) -> int:
    try:
        return e.stat(follow_symlinks=False).st_size if e.is_file(follow_symlinks=False) else 0
    except FileNotFoundError:  # removed while the directory was listed (an eviction, rpc-server replacing it)
        return 0


def cache_usage(cache_dir: Path, exclude: Collection[str] = ()) -> int:
    total = 0
    try:
        with os.scandir(cache_dir) as entries:
            for e in entries:
                if e.name not in exclude:
                    total += _entry_size(e)
    except FileNotFoundError:
        return 0
    return total


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def bytes_over_limits(used: int, free: int, max_bytes: int, min_free: int, extra: int = 0) -> int:
    """Bytes to free so the cache plus `extra` more stays within max_bytes and min_free stays free after it."""
    return max(0, used + extra - max_bytes, min_free - (free - extra))


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    size: int
    mtime: float
    last_used: float


def _cache_entry(e: os.DirEntry[str]) -> CacheEntry | None:
    """A finished cache entry (part and stop files start with "."). Its last use is the access time where the volume
    records reads, else the write time: max() falls back to mtime on noatime volumes, where atime stays at creation."""
    if not _CACHE_FILE_RE.fullmatch(e.name):
        return None
    try:
        if not e.is_file(follow_symlinks=False):
            return None
        st = e.stat(follow_symlinks=False)
    except FileNotFoundError:  # removed while the directory was listed
        return None
    return CacheEntry(Path(e.path), st.st_size, st.st_mtime, max(st.st_atime, st.st_mtime))


def _finished_cache_files(cache_dir: Path) -> list[CacheEntry]:
    """Every finished cache entry, least recently used first."""
    found: list[CacheEntry] = []
    try:
        with os.scandir(cache_dir) as entries:
            for e in entries:
                entry = _cache_entry(e)
                if entry is not None:
                    found.append(entry)
    except FileNotFoundError:
        return []
    return sorted(found, key=lambda f: f.last_used)


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:  # e.g. Windows while rpc-server has it open: try the next file
        log.debug("could not remove cache file %s: %s", path.name, exc)
        return False
    return True


def evict_cache(cache_dir: Path, need: int, keep: Collection[str] = (), used_before: float | None = None) -> int:
    """Delete the least recently used finished cache files, never part files (a running download writes those) or
    names in `keep`, until `need` bytes are freed. A file that vanished or cannot be deleted (Windows, while open) is
    skipped. rpc-server treats a missing file as a cache miss and is sent the tensor again, so this only costs a
    re-send; a file it has mapped (hcpeer) keeps its pages and its disk until unmapped. With `used_before` (a
    time.time() value) a file last used at or after it is left alone too: hcpeer sets the access time of every file
    it maps. Returns the bytes freed."""
    freed = 0
    for entry in _finished_cache_files(cache_dir) if need > 0 else []:
        if freed >= need:
            break
        if used_before is not None and entry.last_used >= used_before:
            continue
        if entry.path.name not in keep and _unlink(entry.path):
            freed += entry.size
    if freed:
        log.info("evicted %s of the least recently used RPC cache files to keep the disk limits", fmt_bytes(freed))
    return freed


def purge_recent(cache_dir: Path, since: float) -> int:
    """Delete finished cache files last written at or after `since` (a time.time() value). rpc-server writes its
    cache file in place with no check, so a worker that died mid-write leaves a short file under the tensor's hash
    name that the next load would take as the whole tensor. Returns how many were removed."""
    return sum(1 for f in _finished_cache_files(cache_dir) if f.mtime >= since and _unlink(f.path))


# --------------------------------------------------------------------------- seed jobs


@dataclass
class _Job:
    job_id: str
    model_id: str
    parts: int
    stop_file: Path
    parts_received: set[int] = field(default_factory=set)
    queue: deque[FetchTask] = field(default_factory=deque)
    nbytes: dict[int, int] = field(default_factory=dict)  # fnv64 -> size, every tensor of the job
    done: set[int] = field(default_factory=set)
    present: set[int] = field(default_factory=set)
    failed: OrderedDict[int, str] = field(default_factory=OrderedDict)
    inflight: dict[Future, FetchTask] = field(default_factory=dict)
    state: Literal["running", "done", "failed", "cancelled"] = "running"
    error: str | None = None
    updated_at: datetime = field(default_factory=utcnow)


class SeedRunner:
    """One seed job at a time (a new job id replaces the running one), on a pool of `workers` downloads. Disk limits
    are checked before each tensor starts; a job that would pass them stops with the reason."""

    def __init__(self, cache_dir: Path, max_disk_bytes: int, min_free_bytes: int = MIN_FREE_BYTES, *,
                 workers: int | None = None,
                 executor_factory: Callable[[int], Executor] = default_executor,
                 fetch: Callable[[FetchTask], FetchResult] = fetch_tensor,
                 disk_free: Callable[[Path], int] = disk_free, poll_s: float = 0.5) -> None:
        self.cache_dir = cache_dir
        self.max_disk_bytes, self.min_free_bytes = max_disk_bytes, min_free_bytes
        self._workers = workers or default_workers()
        self._executor_factory, self._fetch, self.disk_free, self._poll_s = executor_factory, fetch, disk_free, poll_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job: _Job | None = None
        self._retired: list[_Job] = []  # replaced or cancelled jobs whose downloads are still stopping
        self._thread: threading.Thread | None = None
        self._closed = False

    # ---- called from the agent's heartbeat loop
    def submit(self, spec: SeedSpec, sources: Mapping[str, SeedSource]) -> str:
        with self._lock:
            if self._closed:
                raise ManagedError("this agent is stopping")
            job = self._job
            if job is None or job.job_id != spec.job_id:
                if job is not None and job.state == "running":
                    self._retire(job, "cancelled", f"replaced by job {spec.job_id[:8]}")
                job = self._new_job(spec)
            if spec.part in job.parts_received:
                return f"part {spec.part}/{spec.parts} already received"
            if job.state != "running":  # cancelled, or a disk limit stopped it and dropped its queue: never resumed
                raise ManagedError(f"job {spec.job_id[:8]} stopped: {job.error or job.state}")
            job.parts_received.add(spec.part)
            added = self._enqueue(job, spec, sources)
            self._ensure_thread()
        self._wake.set()
        return f"part {spec.part}/{spec.parts}: {added} tensors queued"

    def cancel(self) -> bool:
        with self._lock:
            job = self._job
            if job is None or job.state != "running":
                return False
            self._retire(job, "cancelled", "cancelled by the host")
        self._wake.set()
        return True

    def progress(self) -> SeedProgress | None:
        with self._lock:
            job = self._job
            if job is None:
                return None
            partial = sum(_file_size(part_path(self.cache_dir, t.fnv64)) for t in job.inflight.values())
            done = sum(job.nbytes[f] for f in job.done)
            return SeedProgress(
                job_id=job.job_id, model_id=job.model_id, state=job.state, parts_received=len(job.parts_received),
                parts=job.parts, tensors_total=len(job.nbytes), tensors_done=len(job.done),
                tensors_present=len(job.present), tensors_failed=len(job.failed),
                bytes_total=sum(job.nbytes.values()), bytes_done=done + partial,
                failures=[SeedFailure(fnv64=cache_name(f), error=e) for f, e in list(job.failed.items())[-16:]],
                error=job.error, updated_at=job.updated_at)

    def protected(self) -> set[str]:
        """Cache names of the running job's tensors, which eviction must leave alone."""
        with self._lock:
            job = self._job
            return {cache_name(f) for f in job.nbytes} if job is not None and job.state == "running" else set()

    def current_job(self) -> str | None:
        """The id of the newest job, running or not."""
        with self._lock:
            return self._job.job_id if self._job is not None else None

    def close(self, timeout: float = 10.0) -> None:
        with self._lock:
            self._closed = True
            if self._job is not None and self._job.state == "running":
                self._retire(self._job, "cancelled", "the agent stopped")
            thread = self._thread
        self._wake.set()
        if thread is not None:
            thread.join(timeout)

    # ---- job bookkeeping (under self._lock)
    def _new_job(self, spec: SeedSpec) -> _Job:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stop_file = self.cache_dir / f".seed-stop-{spec.job_id}"
        stop_file.unlink(missing_ok=True)
        self._job = _Job(job_id=spec.job_id, model_id=spec.model_id, parts=spec.parts, stop_file=stop_file)
        return self._job

    def _enqueue(self, job: _Job, spec: SeedSpec, sources: Mapping[str, SeedSource]) -> int:
        added = 0
        for t in spec.tensors:
            fnv = int(t.fnv64, 16)
            if fnv in job.nbytes:
                continue
            job.nbytes[fnv] = t.nbytes
            job.queue.append(FetchTask(url=sources[t.file].url, offset=t.offset, nbytes=t.nbytes, fnv64=fnv,
                                       cache_dir=str(self.cache_dir), stop_file=str(job.stop_file)))
            added += 1
        job.updated_at = utcnow()
        return added

    def _retire(self, job: _Job, state: Literal["failed", "cancelled"], reason: str) -> None:
        """Stop a job for any reason. Its downloads in flight are told to stop and tracked until they end, so a new
        job never starts one of those tensors while a pool worker still writes its part file."""
        job.state, job.error, job.updated_at = state, reason, utcnow()
        job.queue.clear()
        if job.inflight:
            job.stop_file.touch()  # downloads in pool workers check for it between chunks
            self._retired.append(job)

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="hypercompute-seed", daemon=True)
            self._thread.start()

    def _admit(self, job: _Job, task: FetchTask) -> str | None:
        """None when the tensor fits the limits, after evicting the oldest cache files of other models if needed.

        The downloads in flight and this one count at their full size, and their own part and final files are left
        out of `used`: a download renaming its part file while the directory is listed can be seen under neither
        name, which once let a tensor in past the cap."""
        mine = [*job.inflight.values(), task]
        skip = {n for t in mine for n in (cache_name(t.fnv64), part_path(self.cache_dir, t.fnv64).name)}
        size = sum(t.nbytes for t in mine)
        to_write = sum(max(0, t.nbytes - _file_size(part_path(self.cache_dir, t.fnv64))) for t in mine)
        used, free, cap = cache_usage(self.cache_dir, skip), self.disk_free(self.cache_dir), self.max_disk_bytes
        over = max(0, used + size - cap, self.min_free_bytes - (free - to_write))
        if over and evict_cache(self.cache_dir, over, keep={cache_name(f) for f in job.nbytes}):
            used, free = cache_usage(self.cache_dir, skip), self.disk_free(self.cache_dir)
        if used + size > cap:
            return (f"the cache would pass --max-disk {fmt_bytes(cap)} ({fmt_bytes(used)} used, "
                    f"{fmt_bytes(size)} more needed)")
        if free - to_write < self.min_free_bytes:
            return (f"seeding would leave less than {fmt_bytes(self.min_free_bytes)} free on the cache disk "
                    f"({fmt_bytes(free)} free now)")
        return None

    def _fill(self, job: _Job, executor: Executor) -> None:
        with self._lock:
            stopping = {t.fnv64 for old in self._retired for t in old.inflight.values()}
            while job.state == "running" and job.queue and len(job.inflight) < self._workers:
                task = job.queue[0]
                if task.fnv64 in stopping:
                    return  # a replaced job still writes this part file; start it once that download stopped
                if _complete(self.cache_dir / cache_name(task.fnv64), task.nbytes):
                    job.queue.popleft()
                    job.done.add(task.fnv64)
                    job.present.add(task.fnv64)
                    continue
                reason = self._admit(job, task)
                if reason is not None:
                    self._retire(job, "failed", reason)
                    return
                job.queue.popleft()
                job.inflight[executor.submit(self._fetch, task)] = task

    @staticmethod
    def _record(job: _Job, task: FetchTask, fut: Future) -> None:
        try:
            result: FetchResult = fut.result()
        except Exception as exc:  # a pool worker died (BrokenProcessPool) or the fetch raised: one failed tensor
            result = FetchResult(task.fnv64, "failed", error=short_error(exc))
        if result.status in ("written", "present"):
            job.done.add(task.fnv64)
            if result.status == "present":
                job.present.add(task.fnv64)
        elif result.status == "failed":
            job.failed[task.fnv64] = (result.error or "failed")[:200]
        job.updated_at = utcnow()

    def _collect(self, jobs: list[_Job]) -> None:
        futures = [f for j in jobs for f in j.inflight]
        if not futures:
            return
        finished, _ = wait(futures, timeout=self._poll_s, return_when=FIRST_COMPLETED)
        with self._lock:
            for job in jobs:
                for fut in [f for f in job.inflight if f in finished]:
                    self._record(job, job.inflight.pop(fut), fut)

    def _settle(self) -> None:
        """Finish a job whose parts all arrived and ran; forget retired jobs once their downloads stopped."""
        with self._lock:
            job = self._job
            if job is not None and job.state == "running" and not job.queue and not job.inflight \
                    and len(job.parts_received) >= job.parts:
                job.state = "failed" if job.failed else "done"
                job.error = f"{len(job.failed)} tensors failed" if job.failed else None
                job.updated_at = utcnow()
            for old in [j for j in self._retired if not j.inflight]:
                self._retired.remove(old)
                old.stop_file.unlink(missing_ok=True)

    def _busy(self) -> list[_Job]:
        with self._lock:
            jobs = [j for j in self._retired if j.inflight]
            job = self._job
            if job is not None and job not in jobs and (job.inflight or (job.state == "running" and job.queue)):
                jobs.append(job)
            return jobs

    def _run(self) -> None:
        executor: Executor | None = None
        try:
            while True:
                jobs = self._busy()
                if not jobs:
                    self._settle()
                    if executor is not None:
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor = None
                    if self._closed:
                        return
                    self._wake.wait(timeout=5.0)
                    self._wake.clear()
                    continue
                executor = executor or self._executor_factory(self._workers)
                current = self._job
                if current is not None and not self._closed:
                    self._fill(current, executor)
                self._collect(self._busy())
                self._settle()
        except Exception:  # the thread's boundary: the job is marked failed and the cause logged
            log.exception("seed runner stopped")
            with self._lock:
                if self._job is not None and self._job.state == "running":  # its downloads stop too
                    self._retire(self._job, "failed", "the seed runner stopped on an internal error")
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
