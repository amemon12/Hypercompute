"""Host-managed peer: consented worker restarts and llama.cpp RPC cache pre-seeding.

Active only when the contributor joins with --managed. Through the authenticated heartbeat channel the host may then
ask for (1) a worker restart with devices from the consented list and a thread count, and (2) catalog model tensors
downloaded from huggingface.co into <LLAMA_CACHE>/rpc under the name llama.cpp's RPC client asks for (seed_cache.py).
Every command is a typed value checked against the consent and this machine's own model catalog: nothing the host
sends is run, used as a URL or used as a path.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ..schemas import CommandAck, HostInventory, ManagedReport, ModelArtifact, PeerCommand, SeedSpec
from ..units import GiB, fmt_bytes, parse_size
from .seed_cache import (
    MIN_FREE_BYTES,
    ManagedError,
    SeedRunner,
    SeedSource,
    bytes_over_limits,
    cache_usage,
    disk_free,
    evict_cache,
    purge_recent,
    short_error,
)

log = logging.getLogger("hypercompute.agent.managed")

HF_PREFIX = "https://huggingface.co/"
MAX_COMMANDS_PER_REPLY = 16
ACKS_REMEMBERED = 256  # handled command ids, so a re-delivered command is acknowledged again, never re-run
ACKS_SENT = 32
DISK_SNAPSHOT_TTL_S = 10.0
# the disk guard lists the cache directory at most this often from heartbeats (bounded cost); rpc-server writes at
# most a few GB in that time, well inside the free-space floor
GUARD_INTERVAL_S = 15.0
CRASH_PURGE_WINDOW_S = 120.0  # cache files rpc-server wrote this close to its crash may be cut short
# hcpeer v4 stops saving at its cache cap and never evicts (native/hcpeer under_cap), so while a worker runs with that
# cap the guard keeps this much room below it (at most an eighth of the cap) for the saves of the next pass: 2 GiB
# per GUARD_INTERVAL_S is ~140 MB/s, more than a Wi-Fi peer receives. A save that finds no room is copied into RAM.
HCPEER_SAVE_ROOM_BYTES = 2 * GiB
# a file used this long before the worker was last seen idle may belong to a load the agent has not heard of yet
# (the deployment list arrives with the heartbeat reply), so it counts as mapped
IN_USE_MARGIN_S = 120.0
_HEX32_RE = re.compile(r"[a-f0-9]{32}")
_DEVICE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,15}")
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")

__all__ = ["ManagedConsent", "ManagedError", "ManagedPeer", "build_consent", "consent_lines", "purge_after_crash",
           "resolve_sources"]


# --------------------------------------------------------------------------- consent


@dataclass(frozen=True)
class ManagedConsent:
    devices_allowed: tuple[str, ...]
    max_disk_bytes: int  # 0: no cache downloads
    max_threads: int
    min_free_bytes: int = MIN_FREE_BYTES

    @property
    def seeding_allowed(self) -> bool:
        return self.max_disk_bytes > 0


def parse_device_list(text: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    if not 1 <= len(names) <= 8:
        raise ValueError("give 1 to 8 device names, e.g. MTL0,CPU")
    bad = [n for n in names if not _DEVICE_RE.fullmatch(n)]
    if bad:
        raise ValueError(f"not a device name: {', '.join(repr(b[:20]) for b in bad)}")
    if len({n.upper() for n in names}) != len(names):
        raise ValueError("a device is listed twice")
    return names


def default_devices_allowed(system: str, current: list[str]) -> list[str]:
    """macOS: Metal and the CPU of the one rpc-server (-d MTL0,CPU). Elsewhere: what is shared now, plus the CPU."""
    if system == "macos":
        return list(dict.fromkeys(["MTL0", "CPU", *current]))
    return list(dict.fromkeys([*current, "CPU"]))


def build_consent(inv: HostInventory, current: list[str], devices_allowed: str | None,
                  max_disk: str | None) -> ManagedConsent:
    """The --managed consent, checked against the devices llama.cpp sees here. Raises ValueError with the reason."""
    explicit = devices_allowed is not None
    allowed = parse_device_list(devices_allowed) if explicit else default_devices_allowed(inv.os.system, current)
    engine_names = {d.name.upper(): d.name for d in inv.engine.devices}
    if engine_names:
        engine_names.setdefault("CPU", "CPU")  # --list-devices hides CPU, but rpc-server -d CPU selects it
        unknown = [n for n in allowed if n.upper() not in engine_names]
        if unknown and explicit:
            raise ValueError(f"not visible to llama.cpp on this machine: {', '.join(unknown)}")
        allowed = [engine_names[n.upper()] for n in allowed if n.upper() in engine_names]
    if len(allowed) > 8:
        raise ValueError(f"at most 8 devices can be allowed, not {len(allowed)}")
    upper = {a.upper() for a in allowed}
    outside = [n for n in current if n.upper() not in upper]
    if outside:
        raise ValueError(f"the shared devices ({','.join(current)}) must be within --devices-allowed "
                         f"({','.join(allowed) or 'none'})")
    disk = parse_size(max_disk) if max_disk else 0
    return ManagedConsent(tuple(allowed), disk, max(1, os.cpu_count() or 1))


def consent_lines(consent: ManagedConsent, cache_dir: Path) -> list[str]:
    lines = [f"Managed by the host (--managed): the host may restart this machine's worker with devices from "
             f"{','.join(consent.devices_allowed)} and set its thread count (1-{consent.max_threads})."]
    if consent.seeding_allowed:
        lines.append(f"  It may also download tensors of models in this machine's catalog, from huggingface.co only, "
                     f"into {cache_dir}: each one is checked by hash before use, the cache stays under "
                     f"{fmt_bytes(consent.max_disk_bytes)} and at least {fmt_bytes(consent.min_free_bytes)} stays "
                     "free on that disk (the least recently used cache files are deleted to keep both).")
    else:
        lines.append("  It may not download anything (add --max-disk SIZE to allow cache downloads).")
    lines.append("  It cannot run other programs or pick other files or addresses. Stop sharing to end this.")
    return lines


def purge_after_crash(cache_dir: Path) -> int:
    """rpc-server writes each cache file in place with no check, so a worker that died mid-write can leave a short
    file under the tensor's hash name, which the next load would take as the whole tensor. Drops what it wrote just
    before it died (the host sends those tensors again); returns how many files. Never raises."""
    try:
        return purge_recent(cache_dir, time.time() - CRASH_PURGE_WINDOW_S)
    except OSError as exc:
        log.warning("could not check the RPC cache after the worker exited: %s", exc)
        return 0


# --------------------------------------------------------------------------- catalog validation


def safe_file_name(name: str) -> bool:
    return bool(_FILE_RE.fullmatch(name)) and ".." not in name


def _pinned_resolve_url(url: str, art: ModelArtifact, rel: str) -> bool:
    """A resolve URL of the pinned repo and revision that ends in `rel`, optionally under safe repo folders.

    Repos such as unsloth's keep each quantization in a folder while the host stores the GGUF flat (no `path`)."""
    prefix = f"{HF_PREFIX}{art.repo}/resolve/{art.revision}/"
    if not url.startswith(prefix):
        return False
    remote = url[len(prefix):]
    return (remote == rel or remote.endswith("/" + rel)) and all(safe_file_name(seg) for seg in remote.split("/"))


def resolve_sources(spec: SeedSpec, catalog: Mapping[str, ModelArtifact]) -> dict[str, SeedSource]:
    """Each tensor's download URL, built from this machine's catalog entry, never from the host's request."""
    art = catalog.get(spec.model_id)
    if art is None:
        raise ManagedError(f"model {spec.model_id[:80]} is not in this machine's catalog")
    if art.format != "gguf":
        raise ManagedError(f"{art.id} is not a GGUF model")
    if (spec.repo, spec.revision) != (art.repo, art.revision):
        raise ManagedError(f"repo or revision differs from this machine's catalog entry for {art.id}")
    if not _REPO_RE.fullmatch(art.repo) or not _REVISION_RE.fullmatch(art.revision):
        raise ManagedError(f"the catalog entry for {art.id} is not pinned to a Hugging Face commit")
    files = {f.name: f for f in art.files}
    out: dict[str, SeedSource] = {}
    for t in spec.tensors:
        if not safe_file_name(t.file):
            raise ManagedError(f"unsafe file name {t.file[:60]!r}")
        f = files.get(t.file)
        if f is None:
            raise ManagedError(f"{t.file} is not a file of {art.id}")
        rel = f.path or f.name
        if not all(safe_file_name(seg) for seg in rel.split("/")):
            raise ManagedError(f"unsafe catalog path {rel[:60]!r}")
        if not _pinned_resolve_url(f.url, art, rel):
            raise ManagedError(f"the catalog URL of {t.file} is not a {HF_PREFIX} resolve URL of the pinned revision")
        url = f.url
        if t.offset + t.nbytes > f.size_bytes:
            raise ManagedError(f"a tensor range ends past the end of {t.file}")
        out[t.file] = SeedSource(t.file, url, f.size_bytes)
    return out


def load_catalog() -> dict[str, ModelArtifact]:
    from ..services.models import ModelError, load_manifest

    try:
        return {a.id: a for a in load_manifest()}
    except ModelError as exc:
        raise ManagedError(f"cannot read this machine's model catalog: {exc}") from exc


# --------------------------------------------------------------------------- the managed peer


class ManagedPeer:
    """Applies host commands within the consent and builds the report each heartbeat carries.

    Commands are re-delivered until acknowledged; each id runs at most once (a repeat is acknowledged again).
    Worker restarts run on their own thread so heartbeats continue; a seed command is acknowledged once queued and
    its progress is reported in `seed`.
    """

    def __init__(self, consent: ManagedConsent, cache_root: Path, *, cache_enabled: bool,
                 restart_worker: Callable[[list[str], int | None], None],
                 current_worker: Callable[[], tuple[list[str], int | None]],
                 is_busy: Callable[[], bool],
                 catalog: Callable[[], Mapping[str, ModelArtifact]] = load_catalog,
                 runner: SeedRunner | None = None) -> None:
        self.consent = consent
        self.cache_dir = cache_root / "rpc"  # rpc-server: $LLAMA_CACHE/rpc (rpc-server.cpp fs_get_cache_directory)
        self.cache_enabled = cache_enabled
        self.runner = runner or SeedRunner(self.cache_dir, consent.max_disk_bytes, consent.min_free_bytes)
        self._restart_worker, self._current_worker, self._is_busy = restart_worker, current_worker, is_busy
        self._catalog_loader = catalog
        self._catalog: Mapping[str, ModelArtifact] | None = None
        self._lock = threading.Lock()
        self._acks: OrderedDict[str, CommandAck] = OrderedDict()
        self._inflight: set[str] = set()
        self._last_worker: tuple[str, int] | None = None  # (epoch, seq) of the newest worker command taken
        self._worker_thread: threading.Thread | None = None
        self.worker_error: str | None = None
        self._disk: tuple[float, int | None, int | None] = (float("-inf"), None, None)
        self._next_guard = float("-inf")
        self._in_use_since: float | None = None  # with hcpeer's cap: files used since then may be mapped

    # ---- commands
    def handle(self, raw: object) -> None:
        """Take the `commands` of a heartbeat reply. Unknown or malformed commands are refused, never run."""
        if not isinstance(raw, list):
            return
        commands: list[PeerCommand] = []
        for item in raw[:MAX_COMMANDS_PER_REPLY]:
            cmd_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(cmd_id, str) or not _HEX32_RE.fullmatch(cmd_id) or self._seen(cmd_id):
                continue
            try:
                commands.append(PeerCommand.model_validate(item))
            except ValidationError:
                kind = item.get("kind")
                self._ack(cmd_id, kind[:32] if isinstance(kind, str) else "unknown", "refused",
                          "unsupported or malformed command")
        for cmd in sorted(commands, key=lambda c: c.seq):
            try:
                self._dispatch(cmd)
            except ManagedError as exc:
                self._ack(cmd.id, cmd.kind, "refused", str(exc))
            except (OSError, ValueError) as exc:  # e.g. the cache disk: reported, never fatal to the heartbeat loop
                log.warning("managed command %s failed", cmd.id[:8], exc_info=True)
                self._ack(cmd.id, cmd.kind, "failed", short_error(exc))

    def _seen(self, cmd_id: str) -> bool:
        with self._lock:
            if cmd_id in self._acks:
                self._acks.move_to_end(cmd_id)  # send its acknowledgement again: the host has not seen it
                return True
            return cmd_id in self._inflight

    def _ack(self, cmd_id: str, kind: str, status: Literal["accepted", "applied", "refused", "failed"],
             detail: str | None) -> None:
        with self._lock:
            self._acks[cmd_id] = CommandAck(id=cmd_id, kind=kind, status=status,
                                            detail=detail[:300] if detail else None)
            self._acks.move_to_end(cmd_id)
            while len(self._acks) > ACKS_REMEMBERED:
                self._acks.popitem(last=False)
        log.info("managed command %s %s: %s %s", kind, cmd_id[:8], status, detail or "")

    def _dispatch(self, cmd: PeerCommand) -> None:
        if cmd.version != 1:
            raise ManagedError(f"command version {cmd.version} is not supported by this agent")
        if cmd.kind == "worker":
            self._worker(cmd)
        elif cmd.kind == "seed":
            self._seed(cmd)
        else:
            stopped = self.runner.cancel()
            self._ack(cmd.id, cmd.kind, "applied", "seeding cancelled" if stopped else "no seeding was running")

    def _allowed_devices(self, devices: list[str] | None) -> list[str]:
        if not devices:
            raise ManagedError("a worker request must name its devices")
        allowed = {d.upper(): d for d in self.consent.devices_allowed}
        outside = [d for d in devices if d.upper() not in allowed]
        if outside:
            raise ManagedError(f"{','.join(outside)} is outside what this machine allowed "
                               f"({','.join(self.consent.devices_allowed)})")
        chosen = [allowed[d.upper()] for d in devices]
        if len(set(chosen)) != len(chosen):
            raise ManagedError("a device is listed twice")
        return chosen

    def _worker(self, cmd: PeerCommand) -> None:
        devices = self._allowed_devices(cmd.devices)
        if cmd.threads is not None and not 1 <= cmd.threads <= self.consent.max_threads:
            raise ManagedError(f"threads must be 1-{self.consent.max_threads} on this machine")
        last = self._last_worker
        if last is not None and last[0] == cmd.epoch and cmd.seq < last[1]:
            raise ManagedError("superseded by a newer worker request")
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return  # not acknowledged: the host re-sends it, and it runs once the current restart is over
        current_devices, current_threads = self._current_worker()
        threads = cmd.threads if cmd.threads is not None else current_threads
        self._last_worker = (cmd.epoch, cmd.seq)
        if (list(current_devices), current_threads) == (devices, threads):
            self._ack(cmd.id, cmd.kind, "applied", "the worker already runs with these settings")
            return
        if self._is_busy():
            raise ManagedError("a deployment is using this worker; stop it on the host first")
        with self._lock:
            self._inflight.add(cmd.id)
        self._worker_thread = threading.Thread(target=self._restart, args=(cmd, devices, threads),
                                               name="hypercompute-worker-restart", daemon=True)
        self._worker_thread.start()

    def _restart(self, cmd: PeerCommand, devices: list[str], threads: int | None) -> None:
        try:
            self._restart_worker(devices, threads)
        except ManagedError as exc:
            self._ack(cmd.id, cmd.kind, "refused", str(exc))
        except Exception as exc:  # the thread's boundary: any failure goes back to the host, with the cause logged
            log.exception("worker restart failed")
            self.worker_error = short_error(exc)
            self._ack(cmd.id, cmd.kind, "failed", self.worker_error)
        else:
            self.worker_error = None
            self._ack(cmd.id, cmd.kind, "applied",
                      f"worker restarted with -d {','.join(devices)}" + (f" -t {threads}" if threads else ""))
        finally:
            with self._lock:
                self._inflight.discard(cmd.id)

    def _seed(self, cmd: PeerCommand) -> None:
        if not self.consent.seeding_allowed:
            raise ManagedError("this machine did not allow cache downloads (join with --max-disk SIZE)")
        if not self.cache_enabled:
            raise ManagedError("the RPC tensor cache is off on this machine")
        if cmd.seed is None:
            raise ManagedError("a seed request must carry its tensors")
        if self._catalog is None:
            self._catalog = self._catalog_loader()
        sources = resolve_sources(cmd.seed, self._catalog)
        if self.runner.current_job() != cmd.seed.job_id:  # a new job starts within the limits, its own files kept
            self.guard_disk(force=True, keep={t.fnv64 for t in cmd.seed.tensors})
        self._ack(cmd.id, cmd.kind, "accepted", self.runner.submit(cmd.seed, sources))

    def wait_idle(self, timeout: float) -> bool:
        """Test and shutdown helper: wait for a running worker restart to finish."""
        thread = self._worker_thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    # ---- disk limits
    def worker_started(self, hcpeer_capped: bool) -> None:
        """The agent started a worker; `hcpeer_capped`: hcpeer on with HC_PEER_CACHE_MAX_GIB in its environment. That
        worker stops saving at the cap and never evicts (the host plans only the room below the cap that the report
        shows, services/layout.py disk_budget), so from now on the guard keeps room below the cap from the files this
        worker cannot have mapped."""
        self._in_use_since = time.time() if hcpeer_capped else None

    def _mapped_since(self) -> float | None:
        """With hcpeer's cap: the time from which a cache file may be mapped by the running worker (hcpeer sets the
        access time of every file it maps); None without the cap. A worker seen idle has no model loaded, so then only
        files used within IN_USE_MARGIN_S may be."""
        since = self._in_use_since
        if since is not None and not self._is_busy():
            since = self._in_use_since = max(since, time.time() - IN_USE_MARGIN_S)
        return since

    def guard_disk(self, *, force: bool = False, keep: Collection[str] = ()) -> int:
        """Hold the consented cache cap and free-space floor against what rpc-server saves itself (with -c it keeps
        every tensor it is sent, with no limit): the least recently used cache files go first, never part files, the
        running seed job's files or names in `keep`. With hcpeer's cap it keeps room below the cap for hcpeer's saves
        and never takes a file the worker may have mapped. Heartbeats call it every beat, host feature or not, and it
        runs at most every GUARD_INTERVAL_S; `force` runs it now (before a seed job). Never raises: a disk error is
        logged and the next pass tries again. Returns the bytes freed."""
        if not (self.consent.seeding_allowed and self.cache_enabled):
            return 0
        now = time.monotonic()
        if not force and now < self._next_guard:
            return 0
        self._next_guard = now + GUARD_INTERVAL_S
        try:
            return self._evict_over_limits({*self.runner.protected(), *keep})
        except OSError as exc:  # e.g. the cache directory cannot be listed; rpc-server keeps running regardless
            log.warning("cache disk guard skipped: %s", exc)
            return 0

    def _evict_over_limits(self, keep: Collection[str]) -> int:
        self._disk = (float("-inf"), None, None)  # measure afresh: the snapshot may predate an eviction by the runner
        used, free = self._disk_snapshot()
        if used is None or free is None:
            return 0
        cap, since = self.consent.max_disk_bytes, self._mapped_since()
        if since is not None:
            cap -= min(HCPEER_SAVE_ROOM_BYTES, cap // 8)
        over = bytes_over_limits(used, free, cap, self.consent.min_free_bytes)
        if not over:
            return 0
        freed = evict_cache(self.cache_dir, over, keep=keep, used_before=since)
        self._disk = (float("-inf"), None, None)  # measure again: a stale snapshot would evict twice
        return freed

    # ---- report
    def _disk_snapshot(self) -> tuple[int | None, int | None]:
        at, used, free = self._disk
        if time.monotonic() - at > DISK_SNAPSHOT_TTL_S:
            try:
                used, free = cache_usage(self.cache_dir), disk_free(self.cache_dir)
            except OSError as exc:
                log.debug("cache disk check failed: %s", exc)
                used, free = None, None
            self._disk = (time.monotonic(), used, free)
        return used, free

    def report(self, worker_devices: list[str], worker_threads: int | None, generation: int,
               hcpeer_mmap: bool | None = None) -> ManagedReport:
        """hcpeer_mmap: whether the running worker's banner shows hcpeer on (agent/hcpeer.py), None when not reported
        (a host without HCPEER_REPORT_FEATURE)."""
        used, free = self._disk_snapshot()
        with self._lock:
            acks = list(self._acks.values())[-ACKS_SENT:]
        return ManagedReport(
            devices_allowed=list(self.consent.devices_allowed), max_disk_bytes=self.consent.max_disk_bytes,
            min_free_bytes=self.consent.min_free_bytes, max_threads=self.consent.max_threads,
            cache_enabled=self.cache_enabled, worker_devices=list(worker_devices), worker_threads=worker_threads,
            worker_generation=generation, worker_error=self.worker_error, cache_bytes=used, disk_free_bytes=free,
            acks=acks, seed=self.runner.progress(), hcpeer_mmap=hcpeer_mmap)

    def summary(self) -> str:
        progress = self.runner.progress()
        seed = (f"; seed {progress.model_id} {progress.state} {fmt_bytes(progress.bytes_done)} of "
                f"{fmt_bytes(progress.bytes_total)}" if progress else "")
        return (f"on: devices within {','.join(self.consent.devices_allowed)}, cache downloads "
                f"{'up to ' + fmt_bytes(self.consent.max_disk_bytes) if self.consent.seeding_allowed else 'off'}"
                + seed)

    def close(self) -> None:
        self.runner.close()
