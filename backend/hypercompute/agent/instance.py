"""Single-instance lock per role, so a second host/agent never reconciles away the first one's processes."""

from __future__ import annotations

import json
from pathlib import Path

import psutil


class InstanceLocked(RuntimeError):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


def _holder(path: Path) -> int | None:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        pid, created = int(rec["pid"]), float(rec["create_time"])
        if abs(psutil.Process(pid).create_time() - created) < 1.0:
            return pid
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, psutil.Error):
        return None
    return None


def acquire(path: Path, role: str) -> None:
    pid = _holder(path)
    me = psutil.Process()
    if pid is not None and pid != me.pid:
        raise InstanceLocked(f"a Hypercompute {role} is already running on this machine (pid {pid})",
                             fix=f"use the running {role}, or stop it first (`hypercompute status`)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": me.pid, "create_time": me.create_time(), "role": role}), encoding="utf-8")


def release(path: Path) -> None:
    if _holder(path) == psutil.Process().pid:
        path.unlink(missing_ok=True)
