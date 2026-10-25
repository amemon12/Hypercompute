"""Owned-process supervision for engine processes (llama-server, ggml-rpc-server).

Every child is recorded with (pid, create_time, exe) in a state file so a restarted gateway or agent
can find and stop only the processes it started, never an unrelated process that reused a PID.

Children also die with their owner when it is killed outright: on Windows they join a kill-on-close
Job Object, on Linux they get PR_SET_PDEATHSIG. macOS has neither; the agent handles SIGTERM/SIGHUP.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil

from ..schemas import utcnow

log = logging.getLogger("hypercompute.supervisor")

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PR_SET_PDEATHSIG = 1
_job_handle: int | None = None  # process-wide and never closed: the OS closes it (killing the job) when we exit
_job_lock = threading.Lock()


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimitInformation), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _kernel32() -> ctypes.CDLL:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.SetInformationJobObject.restype = ctypes.c_int
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.AssignProcessToJobObject.restype = ctypes.c_int
    return k32


def _kill_on_close_job(k32: ctypes.CDLL) -> int:
    global _job_handle
    with _job_lock:
        if _job_handle is None:
            job = k32.CreateJobObjectW(None, None)
            if not job:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                               ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
            _job_handle = job
        return _job_handle


def bind_to_owner(popen: subprocess.Popen) -> None:
    """Put the child in our kill-on-close job: it dies when this process dies, however it dies."""
    try:
        k32 = _kernel32()
        if not k32.AssignProcessToJobObject(_kill_on_close_job(k32), int(popen._handle)):  # type: ignore[attr-defined]
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
    except OSError as exc:
        log.warning("could not tie pid %s to this process's lifetime (Job Object): %s", popen.pid, exc)


def _linux_pdeathsig() -> Callable[[], None] | None:
    """preexec_fn: SIGTERM the child when the thread that started it dies (prctl PR_SET_PDEATHSIG)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    except OSError as exc:
        log.warning("could not load libc for PR_SET_PDEATHSIG: %s", exc)
        return None
    owner = os.getpid()

    def preexec() -> None:
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if os.getppid() != owner:  # the owner died before prctl took effect
            os.kill(os.getpid(), signal.SIGTERM)

    return preexec


@dataclass
class ManagedProcess:
    key: str
    popen: subprocess.Popen
    exe: str
    args: list[str]
    log_path: Path
    started_at: datetime
    create_time: float | None = None
    _log_handle: object | None = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None

    def exit_code(self) -> int | None:
        return self.popen.poll()

    def cpu_seconds(self) -> float | None:
        try:
            t = psutil.Process(self.pid).cpu_times()
        except (psutil.Error, OSError):
            return None
        return float(t.user + t.system)

    def read_log(self, max_bytes: int = 256 * 1024) -> str:
        try:
            with self.log_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes))
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


@dataclass(frozen=True)
class StopOutcome:
    key: str
    pid: int
    exit_code: int | None
    how: str  # graceful | terminated (Windows TerminateProcess) | killed | already_exited | still_running

    @property
    def clean(self) -> bool:
        return self.how in ("graceful", "terminated", "killed", "already_exited")

    def describe(self) -> str:
        return f"{self.key} pid {self.pid}: {self.how} (exit code {self.exit_code})"


class Supervisor:
    def __init__(self, state_file: Path) -> None:
        self._state_file = state_file
        self._procs: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        state_file.parent.mkdir(parents=True, exist_ok=True)

    # -- persistence ---------------------------------------------------------------------------
    def _records(self) -> list[dict]:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write_records(self, records: list[dict]) -> None:
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
        tmp.replace(self._state_file)

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self, key: str, exe: Path, args: list[str], cwd: Path, log_path: Path,
              env: dict[str, str] | None = None) -> ManagedProcess:
        with self._lock:
            existing = self._procs.get(key)
            if existing and existing.alive():
                raise RuntimeError(f"managed process {key} is already running (pid {existing.pid})")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("wb")
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True
                kwargs["preexec_fn"] = _linux_pdeathsig()
            popen = subprocess.Popen(  # args are an allowlisted exe + argument array, never a shell string
                [str(exe), *args], cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})}, **kwargs,
            )
            if sys.platform == "win32":
                bind_to_owner(popen)
            try:
                create_time = psutil.Process(popen.pid).create_time()
            except psutil.Error:
                create_time = None
            proc = ManagedProcess(key=key, popen=popen, exe=str(exe), args=list(args), log_path=log_path,
                                  started_at=utcnow(), create_time=create_time, _log_handle=handle)
            self._procs[key] = proc
            records = [r for r in self._records() if r.get("key") != key]
            records.append({"key": key, "pid": popen.pid, "create_time": create_time, "exe": str(exe),
                            "started_at": proc.started_at.isoformat()})
            self._write_records(records)
            log.info("started %s pid %s", key, popen.pid)
            return proc

    def get(self, key: str) -> ManagedProcess | None:
        return self._procs.get(key)

    def recorded(self) -> list[dict]:
        """Records of processes this supervisor started and has not confirmed stopped."""
        with self._lock:
            return self._records()

    def _drop_record(self, key: str) -> None:
        with self._lock:
            self._write_records([r for r in self._records() if r.get("key") != key])

    def stop(self, key: str, grace_s: float = 8.0) -> StopOutcome | None:
        with self._lock:
            proc = self._procs.pop(key, None)
        if proc is None:
            self._drop_record(key)
            return None
        outcome = self._terminate(proc, grace_s)
        if not outcome.clean:
            with self._lock:
                self._procs.setdefault(key, proc)  # keep it and its record so stop/stop_all/reconcile can retry
            log.warning("could not stop %s", outcome.describe())
            return outcome
        self._drop_record(key)
        handle = proc._log_handle
        if handle is not None:
            try:
                handle.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        log.info("stopped %s", outcome.describe())
        return outcome

    def stop_all(self, grace_s: float = 8.0) -> list[StopOutcome]:
        return [o for key in list(self._procs) if (o := self.stop(key, grace_s)) is not None]

    @staticmethod
    def _terminate(proc: ManagedProcess, grace_s: float) -> StopOutcome:
        if not proc.alive():
            return StopOutcome(proc.key, proc.pid, proc.exit_code(), "already_exited")
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.Error:
            children = []
        try:
            if sys.platform == "win32":
                proc.popen.terminate()  # TerminateProcess; llama.cpp holds no state needing a flush
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            code = proc.popen.wait(timeout=grace_s)
            how = "terminated" if sys.platform == "win32" else "graceful"
        except subprocess.TimeoutExpired:
            try:
                if sys.platform == "win32":
                    proc.popen.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                code = proc.popen.wait(timeout=grace_s)
                how = "killed"
            except (subprocess.TimeoutExpired, OSError):
                return StopOutcome(proc.key, proc.pid, None, "still_running")
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        return StopOutcome(proc.key, proc.pid, code, how)

    def reconcile_orphans(self) -> list[str]:
        """Stop processes recorded by a previous run of this supervisor that are still ours.

        A record is dropped only when its process is confirmed gone (or is no longer ours). A process that
        could not be inspected or stopped keeps its record, so a later (e.g. elevated) run can retry.
        """
        notes: list[str] = []
        kept: list[dict] = []
        for rec in self.recorded():
            note, keep = self._reconcile_one(rec)
            notes.append(note)
            if keep:
                kept.append(rec)
        with self._lock:
            self._write_records(kept)
        return notes

    @staticmethod
    def _reconcile_one(rec: dict) -> tuple[str, bool]:
        """Returns (note, keep_record)."""
        pid, key = rec.get("pid"), rec.get("key")
        try:
            p = psutil.Process(int(pid))  # type: ignore[arg-type]
            same = (rec.get("create_time") is not None and abs(p.create_time() - rec["create_time"]) < 1.0
                    and Path(p.exe()).resolve() == Path(rec["exe"]).resolve())
        except (TypeError, ValueError):
            return f"{key}: invalid record (pid {pid}); dropped", False
        except psutil.NoSuchProcess:
            return f"{key}: pid {pid} no longer running", False
        except (psutil.Error, OSError) as exc:
            return f"{key}: could not inspect pid {pid} ({exc.__class__.__name__}); record kept", True
        if not same:
            return f"{key}: pid {pid} now belongs to another process; left untouched", False
        try:
            p.terminate()
            try:
                p.wait(timeout=8)
                return f"{key}: stopped orphaned pid {pid}", False
            except psutil.TimeoutExpired:
                p.kill()
                p.wait(timeout=8)
                return f"{key}: killed orphaned pid {pid}", False
        except psutil.NoSuchProcess:
            return f"{key}: orphaned pid {pid} exited", False
        except psutil.TimeoutExpired:
            return f"{key}: pid {pid} still running after kill; record kept", True
        except psutil.Error as exc:
            return f"{key}: could not stop pid {pid} ({exc.__class__.__name__}); record kept", True
