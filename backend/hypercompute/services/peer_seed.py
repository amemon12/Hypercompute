"""Host side of managed peers: the command queue a managed peer drains through its heartbeats, and seed plans.

A seed plan lists the exact tensors a peer's layers put in its llama.cpp RPC cache: for each weight tensor over
HASH_THRESHOLD (the only ones ggml-rpc.cpp looks up by hash), the catalog file, absolute offset, exact nbytes and the
FNV-1a 64 of those bytes read from the host's own GGUF, so the names match what the RPC client will ask for at load.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .. import gguf
from ..agent.seed_cache import HASH_THRESHOLD, SeedStopped, default_executor, default_workers, fnv1a64_file
from ..schemas import (
    MAX_SEED_TENSORS_PER_COMMAND,
    ManagedReport,
    NodeView,
    PeerCommand,
    Plan,
    SeedSpec,
    SeedTensor,
    utcnow,
)
from ..units import fmt_bytes

if TYPE_CHECKING:
    from .deployments import DeploymentManager
    from .models import ModelService

log = logging.getLogger("hypercompute.peer_seed")
HISTORY = 20
SEED_PART_JSON_BUDGET = 40_000  # encoded tensor JSON per seed part: one part per reply, well below the 64 KiB frame
_LAYERS_RE = re.compile(r"(\d{1,4})(?:-(\d{1,4}))?")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
JobState = Literal["hashing", "queued", "delivered", "refused", "failed", "cancelled"]


class ManagedRefusal(Exception):
    """An owner request the host will not queue; status, code and fix map onto the gateway's error reply."""

    def __init__(self, status: int, code: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.fix = status, code, fix


# --------------------------------------------------------------------------- seed plans


@dataclass(frozen=True)
class SeedBuild:
    repo: str
    revision: str
    tensors: list[SeedTensor]
    bytes_total: int
    skipped: list[str]  # tensors over the threshold whose exact size this reader does not know


@dataclass(frozen=True)
class SeededShard:
    """A seed job the peer finished: every hash-cached tensor of these layers of this model is in its RPC cache, so
    a load of them hits the cache and saves nothing (services/layout.py credits it against the disk budget)."""

    model_id: str
    layers: frozenset[int]


def parse_layers(text: str) -> list[int]:
    m = _LAYERS_RE.fullmatch(text.strip())
    if not m:
        raise ValueError(f"layers must look like 30-35 or 7, not {text[:20]!r}")
    first, last = int(m[1]), int(m[2] if m[2] is not None else m[1])
    if last < first:
        raise ValueError(f"layer range {first}-{last} runs backwards")
    return list(range(first, last + 1))


def plan_layers(plan: Plan, node_id: str) -> list[int]:
    """Every layer the plan places on any device of this node."""
    layers = sorted({n for a in plan.assignments if a.node_id == node_id for n in a.layers})
    if not layers:
        raise ManagedRefusal(422, "no_layers_in_plan", f"plan {plan.id[:8]} gives node {node_id[:8]} no layers",
                             fix="pass --layers A-B, or plan with this peer selected")
    return layers


def load_plan(deployments: DeploymentManager, plan_id: str) -> Plan:
    """A stored plan of any age (seeding only reads its layer split; deployments refuse stale plans themselves)."""
    plan = deployments.find_plan(plan_id)
    if plan is None:
        raise ManagedRefusal(404, "plan_not_found", f"no stored plan {plan_id[:16]}")
    return plan


def select_tensors(table: Iterable[gguf.TensorInfo], layers: Iterable[int],
                   threshold: int = HASH_THRESHOLD) -> tuple[list[gguf.TensorInfo], list[str]]:
    """Tensors of these layers that the RPC client looks up by hash: exact nbytes over the threshold."""
    wanted = set(layers)
    chosen, skipped = [], []
    for t in table:
        if t.layer not in wanted:
            continue
        if t.nbytes is None:
            if t.span > threshold:
                skipped.append(t.name)
            continue
        if t.nbytes > threshold:
            chosen.append(t)
    return chosen, skipped


_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _cache_lock(path: Path) -> threading.Lock:
    """One lock per hash cache file: the seeds of one model's peers (run_big --seed-peers) hash at the same time."""
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(os.path.abspath(path), threading.Lock())


class HashCache:
    """FNV-1a 64 of tensor ranges already read from a model's files, kept per (file, size, mtime): hashing reads every
    byte (about 20 MB/s per core), so a second peer or a second request for the same layers costs nothing."""

    def __init__(self, path: Path | None, files: dict[str, Path]) -> None:
        self.path = path
        self._stamps = {name: self._stamp(p) for name, p in files.items()}
        self._hashes: dict[str, str] = {}
        if path is not None:
            with _cache_lock(path):
                self._hashes = self._read_valid(path)

    @staticmethod
    def _stamp(path: Path) -> list[int]:
        st = path.stat()
        return [st.st_size, st.st_mtime_ns]

    @staticmethod
    def key(file: str, offset: int, nbytes: int) -> str:
        return f"{file}:{offset}:{nbytes}"

    def _read_valid(self, path: Path) -> dict[str, str]:
        """The file's hashes of model files whose size and mtime still match (a re-download drops its entries)."""
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable hash cache %s: %s", path, exc)
            return {}
        stamps = doc.get("files") if isinstance(doc, dict) else None
        hashes = doc.get("hashes") if isinstance(doc, dict) else None
        if not isinstance(stamps, dict) or not isinstance(hashes, dict):
            return {}
        valid = {name for name, stamp in self._stamps.items() if stamps.get(name) == stamp}
        return {k: v for k, v in hashes.items()
                if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v) and k.split(":", 1)[0] in valid}

    def get(self, key: str) -> str | None:
        return self._hashes.get(key)

    def put(self, key: str, value: str) -> None:
        self._hashes[key] = value

    def save(self) -> None:
        """Merge into the file what other seeds of this model saved since this one loaded it, then replace it through
        a tmp file of its own, so concurrent seeds neither drop each other's hashes nor share a tmp file."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _cache_lock(self.path):
            merged = {**self._read_valid(self.path), **self._hashes}
            tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_text(json.dumps({"files": self._stamps, "hashes": merged}), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                tmp.unlink(missing_ok=True)
                raise
            self._hashes = merged


def _hash_item(item: tuple[str, int, int, str | None]) -> int:
    path, offset, nbytes, stop_file = item
    return fnv1a64_file(path, offset, nbytes, stop_file)


def hash_ranges(items: list[tuple[str, int, int]], workers: int, stop_file: str | None = None) -> list[int]:
    """FNV-1a 64 of each (path, offset, nbytes), on `workers` processes; touching stop_file aborts it."""
    jobs = [(p, o, n, stop_file) for p, o, n in items]
    if workers <= 1 or len(jobs) <= 1:
        return [_hash_item(j) for j in jobs]
    executor = default_executor(min(workers, len(jobs)))
    try:
        return list(executor.map(_hash_item, jobs))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def build_seed(models: ModelService, model_id: str, layers: list[int], *, cache_dir: Path | None,
               workers: int | None = None, threshold: int = HASH_THRESHOLD,
               stop_file: str | None = None) -> SeedBuild:
    """The seed tensors of `layers` of a downloaded, verified catalog GGUF, hashed from the host's files."""
    art = models.get(model_id)
    if art.format != "gguf":
        raise ManagedRefusal(422, "not_gguf", f"{model_id} is not a GGUF model")
    view = models.view(model_id)
    if view.status != "downloaded":
        raise ManagedRefusal(409, "model_not_downloaded", f"{model_id} is {view.status} on this host",
                             fix=f"download and verify it first: uv run hypercompute models pull {model_id}")
    paths = models.file_paths(model_id)
    chosen, skipped = select_tensors(gguf.read_tensor_table(paths), layers, threshold)
    if not chosen:
        raise ManagedRefusal(422, "nothing_to_seed", f"layers {layers[0]}-{layers[-1]} of {model_id} have no "
                             f"tensor over {fmt_bytes(threshold)}; the host sends smaller ones at load anyway")
    names = [f.name for f in art.files]
    cache_path = cache_dir / f"{model_id}.json" if cache_dir is not None and _SAFE_ID_RE.fullmatch(model_id) else None
    cache = HashCache(cache_path, dict(zip(names, paths, strict=True)))
    keys = [HashCache.key(names[t.shard], t.offset, t.nbytes or 0) for t in chosen]
    missing = [i for i, k in enumerate(keys) if cache.get(k) is None]
    if missing:
        fresh = hash_ranges([(str(paths[chosen[i].shard]), chosen[i].offset, chosen[i].nbytes or 0) for i in missing],
                            workers or default_workers(), stop_file)
        for i, value in zip(missing, fresh, strict=True):
            cache.put(keys[i], f"{value:016x}")
        cache.save()
    tensors = [SeedTensor(file=names[t.shard], offset=t.offset, nbytes=t.nbytes or 0, fnv64=cache.get(k) or "")
               for t, k in zip(chosen, keys, strict=True)]
    return SeedBuild(repo=art.repo, revision=art.revision, tensors=tensors,
                     bytes_total=sum(t.nbytes for t in tensors), skipped=skipped)


# --------------------------------------------------------------------------- per-peer command queue


def encoded_size(tensor: SeedTensor) -> int:
    """Bytes of one tensor entry as the Iroh transport frames it (json.dumps escapes every non-ASCII character to
    \\uXXXX, up to 12 bytes each), which is also an upper bound on the gateway's UTF-8 HTTP reply."""
    return len(json.dumps(tensor.model_dump(mode="json"), separators=(",", ":")))


def seed_chunks(tensors: list[SeedTensor], budget: int = SEED_PART_JSON_BUDGET) -> list[list[SeedTensor]]:
    """Split a seed list into command parts of at most MAX_SEED_TENSORS_PER_COMMAND tensors and `budget` bytes of
    encoded JSON, so a heartbeat reply stays under the Iroh 64 KiB control frame whatever the catalog file names are."""
    chunks: list[list[SeedTensor]] = []
    current: list[SeedTensor] = []
    size = 0
    for t in tensors:
        n = encoded_size(t) + 1  # and its comma
        if current and (len(current) >= MAX_SEED_TENSORS_PER_COMMAND or size + n > budget):
            chunks.append(current)
            current, size = [], 0
        current.append(t)
        size += n
    return [*chunks, current] if current else chunks


@dataclass
class HostSeedJob:
    job_id: str
    node_id: str
    model_id: str
    layers: list[int]
    stop_file: Path | None
    state: JobState = "hashing"
    detail: str | None = None
    tensors: int = 0
    bytes_total: int = 0
    parts: int = 0
    parts_accepted: int = 0
    skipped: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def set(self, state: JobState, detail: str | None = None) -> None:
        self.state, self.detail, self.updated_at = state, detail, utcnow()

    def view(self) -> dict[str, Any]:
        span = f"{self.layers[0]}-{self.layers[-1]}" if self.layers else "-"
        return {"job_id": self.job_id, "node_id": self.node_id, "model_id": self.model_id, "layers": span,
                "state": self.state, "detail": self.detail, "tensors": self.tensors, "bytes_total": self.bytes_total,
                "parts": self.parts, "parts_accepted": self.parts_accepted, "skipped": self.skipped[:20],
                "started_at": self.started_at.isoformat(), "updated_at": self.updated_at.isoformat()}


@dataclass
class _PeerState:
    report: ManagedReport | None = None
    reported_at: datetime | None = None
    generation: int | None = None
    pending: list[PeerCommand] = field(default_factory=list)
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=HISTORY))
    job: HostSeedJob | None = None


class ManagedPeers:
    """Commands for managed peers, delivered in heartbeat replies until acknowledged. In memory: a gateway restart
    forgets pending commands (re-issue them); the agent's own state is unaffected. At most one seed part goes out
    per reply so a reply stays far below the Iroh 64 KiB control frame."""

    def __init__(self) -> None:
        self.epoch = secrets.token_hex(8)
        self._seq = 0
        self._peers: dict[str, _PeerState] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ---- lifecycle
    def forget(self, node_id: str) -> None:
        """On join (a new agent run) and on leave: drop the queue and stop host-side hashing for this node."""
        peer = self._peers.pop(node_id, None)
        if peer is not None and peer.job is not None and peer.job.state == "hashing":
            self._stop_job(peer.job, "the peer left or rejoined")
        task = self._tasks.pop(node_id, None)
        if task is not None and not task.done():
            task.cancel()

    # ---- heartbeat
    def on_heartbeat(self, node_id: str, report: ManagedReport | None) -> tuple[list[PeerCommand] | None, bool]:
        """Commands to deliver now, and whether the worker restarted since the previous report. Nothing is ever
        delivered to an agent that sends no report (no --managed, or an agent or host without the feature)."""
        if report is None:
            return None, False
        peer = self._peers.setdefault(node_id, _PeerState())
        restarted = peer.generation is not None and report.worker_generation != peer.generation
        peer.generation, peer.report, peer.reported_at = report.worker_generation, report, utcnow()
        self._take_acks(peer, report)
        return self._deliverable(peer) or None, restarted

    def _take_acks(self, peer: _PeerState, report: ManagedReport) -> None:
        acks = {a.id: a for a in report.acks}
        refused_jobs: set[str] = set()
        kept: list[PeerCommand] = []
        for cmd in peer.pending:
            ack = acks.get(cmd.id)
            if ack is None:
                kept.append(cmd)
                continue
            peer.history.append({"id": cmd.id, "kind": cmd.kind, "seq": cmd.seq, "status": ack.status,
                                 "detail": ack.detail, "at": utcnow().isoformat()})
            job = peer.job
            if cmd.seed is not None and job is not None and job.job_id == cmd.seed.job_id:
                if ack.status == "accepted":
                    job.parts_accepted += 1
                    if job.parts_accepted >= job.parts:
                        job.set("delivered", "every part reached the peer; see its seed progress")
                else:
                    job.set("refused", ack.detail)
                    refused_jobs.add(job.job_id)
        peer.pending = [c for c in kept if c.seed is None or c.seed.job_id not in refused_jobs]

    @staticmethod
    def _deliverable(peer: _PeerState) -> list[PeerCommand]:
        out = [c for c in peer.pending if c.kind != "seed"]
        seed = next((c for c in peer.pending if c.kind == "seed"), None)
        return sorted([*out, *([seed] if seed is not None else [])], key=lambda c: c.seq)

    def _command(self, kind: Literal["worker", "seed", "seed_cancel"], **fields: Any) -> PeerCommand:
        self._seq += 1
        return PeerCommand(id=uuid.uuid4().hex, epoch=self.epoch, seq=self._seq, kind=kind, **fields)

    def _require(self, node_id: str) -> tuple[_PeerState, ManagedReport]:
        peer = self._peers.get(node_id)
        if peer is None or peer.report is None:
            raise ManagedRefusal(409, "not_managed", f"node {node_id[:8]} did not join with --managed (or its agent "
                                 "predates managed peers)",
                                 fix="on that machine: hypercompute leave, then join again with --managed")
        return peer, peer.report

    # ---- owner requests
    def request_worker(self, node_id: str, devices: list[str], threads: int | None, *, busy: bool) -> PeerCommand:
        peer, report = self._require(node_id)
        allowed = {d.upper(): d for d in report.devices_allowed}
        outside = [d for d in devices if d.upper() not in allowed]
        if outside:
            raise ManagedRefusal(409, "outside_consent", f"{','.join(outside)} is not among the devices node "
                                 f"{node_id[:8]} allowed ({','.join(report.devices_allowed)})")
        chosen = [allowed[d.upper()] for d in devices]
        if len(set(chosen)) != len(chosen):
            raise ManagedRefusal(422, "duplicate_device", "a device is listed twice")
        if threads is not None and threads > report.max_threads:
            raise ManagedRefusal(409, "outside_consent", f"node {node_id[:8]} allows at most {report.max_threads} "
                                 "threads")
        if busy:
            raise ManagedRefusal(409, "peer_busy", f"a deployment is using node {node_id[:8]}",
                                 fix="stop that deployment first; a restart would break it")
        peer.pending = [c for c in peer.pending if c.kind != "worker"]  # the newest worker request wins
        cmd = self._command("worker", devices=chosen, threads=threads)
        peer.pending.append(cmd)
        return cmd

    def request_cancel(self, node_id: str) -> PeerCommand:
        peer, _ = self._require(node_id)
        peer.pending = [c for c in peer.pending if c.kind not in ("seed", "seed_cancel")]
        if peer.job is not None and peer.job.state in ("hashing", "queued"):
            self._stop_job(peer.job, "cancelled by the owner")
        cmd = self._command("seed_cancel")
        peer.pending.append(cmd)
        return cmd

    def check_seed_allowed(self, node_id: str) -> ManagedReport:
        peer, report = self._require(node_id)
        if report.max_disk_bytes <= 0:
            raise ManagedRefusal(409, "seeding_not_allowed", f"node {node_id[:8]} did not allow cache downloads",
                                 fix="on that machine, rejoin with --managed --max-disk 80GiB")
        if not report.cache_enabled:
            raise ManagedRefusal(409, "cache_off", f"node {node_id[:8]} runs its worker without the RPC cache")
        if peer.job is not None and peer.job.state == "hashing":
            raise ManagedRefusal(409, "seed_in_progress", f"the host is still hashing job {peer.job.job_id[:8]}",
                                 fix="wait for it (capacity peers) or cancel it (capacity seed --cancel)")
        return report

    def start_seed(self, node_id: str, model_id: str, layers: list[int],
                   build: Callable[[str | None], SeedBuild], stop_dir: Path | None = None) -> HostSeedJob:
        """Hash the tensors on a worker thread, then queue them for the peer. `build(stop_file)` does the work."""
        report = self.check_seed_allowed(node_id)
        job_id = uuid.uuid4().hex
        stop_file = stop_dir / f".stop-{job_id}" if stop_dir is not None else None
        job = HostSeedJob(job_id=job_id, node_id=node_id, model_id=model_id, layers=layers, stop_file=stop_file)
        self._peers[node_id].job = job
        old = self._tasks.pop(node_id, None)
        if old is not None and not old.done():
            old.cancel()
        task = asyncio.get_running_loop().create_task(self._run_seed(node_id, job, build, report))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # failures are recorded on the job
        self._tasks[node_id] = task
        return job

    async def _run_seed(self, node_id: str, job: HostSeedJob, build: Callable[[str | None], SeedBuild],
                        report: ManagedReport) -> None:
        stop = str(job.stop_file) if job.stop_file is not None else None
        # A cancel (forget, a new request, the loop stopping) cannot stop the thread. The shielded executor future
        # (not a task, which the loop cancels when it stops) tells when the thread really returned.
        work = asyncio.get_running_loop().run_in_executor(None, build, stop)
        try:
            result = await asyncio.shield(work)
        except asyncio.CancelledError:
            self._cancel_hashing(job, work)
            raise
        except SeedStopped:
            self._stop_job(job, job.detail or "cancelled")
            return
        except ManagedRefusal as exc:
            job.set("failed", f"{exc}" + (f" (fix: {exc.fix})" if exc.fix else ""))
            return
        except (RuntimeError, gguf.GgufError, OSError, ValueError) as exc:  # ModelError, a short file, a dead pool
            log.warning("seed plan for %s failed: %s", node_id[:8], exc)
            job.set("failed", f"{type(exc).__name__}: {exc}"[:300])
            return
        finally:
            if work.done():  # else _cancel_hashing removes it once the thread returns
                self._clear_stop(job)
        self._queue_seed(node_id, job, result, report)

    def _cancel_hashing(self, job: HostSeedJob, work: asyncio.Future[SeedBuild]) -> None:
        """The request was cancelled while its thread may still hash: signal the hashing processes, whatever the
        job's state, and remove the stop file only after the thread returned, or they would never see it."""
        if not work.done():
            self._signal_stop(job)
            work.add_done_callback(lambda w: self._thread_returned(job, w))
        if job.state == "hashing":
            job.set("cancelled", "cancelled")

    @classmethod
    def _thread_returned(cls, job: HostSeedJob, work: asyncio.Future[SeedBuild]) -> None:
        """A cancelled request's thread returned. Nobody awaits its outcome any more (the shield dropped its own
        callback when the request was cancelled), so read it here or asyncio logs it as lost; then remove the stop
        file."""
        exc = None if work.cancelled() else work.exception()
        if exc is not None and not isinstance(exc, SeedStopped):
            log.debug("the hashing of cancelled job %s ended with %s", job.job_id[:8], exc)
        cls._clear_stop(job)

    def _queue_seed(self, node_id: str, job: HostSeedJob, result: SeedBuild, report: ManagedReport) -> None:
        peer = self._peers.get(node_id)
        if peer is None or peer.job is not job or job.state != "hashing":
            return  # the node left or rejoined, or the owner cancelled while hashing
        job.tensors, job.bytes_total, job.skipped = len(result.tensors), result.bytes_total, result.skipped
        if result.bytes_total > report.max_disk_bytes:
            job.set("refused", f"needs {fmt_bytes(result.bytes_total)} of cache but node {node_id[:8]} allows "
                               f"{fmt_bytes(report.max_disk_bytes)}; seed fewer layers")
            return
        chunks = seed_chunks(result.tensors)
        commands = [self._command("seed", seed=SeedSpec(job_id=job.job_id, model_id=job.model_id, repo=result.repo,
                                                        revision=result.revision, part=i, parts=len(chunks),
                                                        tensors=chunk))
                    for i, chunk in enumerate(chunks, 1)]
        peer.pending = [c for c in peer.pending if c.kind != "seed"] + commands
        job.parts = len(commands)
        job.set("queued", f"{len(commands)} parts wait for the peer's next heartbeats")

    @classmethod
    def _stop_job(cls, job: HostSeedJob, reason: str) -> None:
        if job.state == "hashing":
            cls._signal_stop(job)
        job.set("cancelled", reason)

    @staticmethod
    def _signal_stop(job: HostSeedJob) -> None:
        if job.stop_file is None:
            return
        try:
            job.stop_file.parent.mkdir(parents=True, exist_ok=True)
            job.stop_file.touch()  # the hashing processes check it between chunks
        except OSError as exc:
            log.warning("could not signal the hashing of job %s to stop: %s", job.job_id[:8], exc)

    @staticmethod
    def _clear_stop(job: HostSeedJob) -> None:
        if job.stop_file is None:
            return
        try:
            job.stop_file.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove the stop file of job %s: %s", job.job_id[:8], exc)

    # ---- views
    def report(self, node_id: str) -> ManagedReport | None:
        peer = self._peers.get(node_id)
        return peer.report if peer is not None else None

    def seeded(self, node_id: str) -> SeededShard | None:
        """The layers this host's last seed job put in the peer's RPC cache: delivered whole (no tensor skipped) and
        reported done by the peer with no tensor failed. None for anything else, or a report about another job."""
        peer = self._peers.get(node_id)
        job = peer.job if peer is not None else None
        seed = peer.report.seed if peer is not None and peer.report is not None else None
        if job is None or seed is None or seed.job_id != job.job_id or job.state != "delivered" or job.skipped:
            return None
        if seed.state != "done" or seed.tensors_failed or seed.tensors_done < job.tensors:
            return None
        return SeededShard(job.model_id, frozenset(job.layers))

    def snapshot(self, views: Iterable[NodeView], busy: Callable[[str], list[str]]) -> list[dict[str, Any]]:
        out = []
        for v in views:
            if v.is_local:
                continue
            peer = self._peers.get(v.id) or _PeerState()
            report = peer.report
            out.append({
                "node_id": v.id, "display_name": v.display_name, "state": v.state.value,
                "worker_state": v.worker_state.value if v.worker_state else None,
                "contribution_devices": list(v.contribution.devices),
                "rpc_devices": [d.name for d in v.verification.rpc_devices] if v.verification else [],
                "deployments": busy(v.id), "managed": report is not None,
                "report": report.model_dump(mode="json") if report is not None else None,
                "reported_at": peer.reported_at.isoformat() if peer.reported_at else None,
                "pending": [{"id": c.id, "kind": c.kind, "seq": c.seq} for c in peer.pending],
                "history": list(peer.history), "seed_request": peer.job.view() if peer.job else None})
        return out


def stop_dir_for(base: Path) -> Path:
    """Where host hashing jobs look for their stop file."""
    path = base / "rpc-hashes"
    path.mkdir(parents=True, exist_ok=True)
    return path
