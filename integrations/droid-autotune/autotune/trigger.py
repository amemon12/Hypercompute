"""When a run starts: a peer joined or left, the pool then stayed the same for `settle_s`, and nothing else is moving.

The watcher compares each snapshot with a baseline (the pool the last run saw). A peer that joins and leaves inside one
settle window nets out to no change, and every further change restarts the window, so a flapping Wi-Fi peer cannot
start a run per heartbeat. A change that settles while the gateway is busy (a model loading, a tune trial, a run of
ours) waits and fires once the gateway is quiet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .pool import PoolChange, PoolSnapshot, diff

BUSY_DEPLOYMENT_STATES = frozenset({"planned", "reserving", "preparing", "loading", "warming", "draining"})


@dataclass(frozen=True)
class Tick:
    """What one observation decided, for the log."""

    action: Literal["baseline", "changed", "settling", "deferred", "fire", "netted_out", "idle"]
    change: PoolChange | None = None
    detail: str = ""


class PoolWatcher:
    def __init__(self, settle_s: float) -> None:
        if settle_s < 0:
            raise ValueError("settle_s must be >= 0")
        self.settle_s = settle_s
        self._baseline: PoolSnapshot | None = None
        self._latest: PoolSnapshot | None = None
        self._changed_at: float | None = None

    @property
    def baseline(self) -> PoolSnapshot | None:
        return self._baseline

    @property
    def pending(self) -> PoolChange | None:
        if self._baseline is None or self._latest is None:
            return None
        change = diff(self._baseline, self._latest)
        return None if change.empty else change

    def observe(self, snap: PoolSnapshot, now: float, busy: str | None = None) -> Tick:
        """Feed one snapshot. busy: why the gateway cannot take a run now (None when it can)."""
        if self._baseline is None:
            self._baseline, self._latest = snap, snap
            return Tick("baseline", detail=f"{len(snap.peers)} eligible peer(s)")
        moved = self._latest is None or snap.fingerprint != self._latest.fingerprint
        had_pending = self.pending is not None
        self._latest = snap
        if moved:
            self._changed_at = now
            change = self.pending
            if change is None:
                return Tick("netted_out" if had_pending else "idle", detail="the pool is back to the last baseline")
            return Tick("changed", change, f"waiting {self.settle_s:.0f} s for the pool to settle")
        change = self.pending
        if change is None:
            return Tick("idle")
        waited = now - (self._changed_at if self._changed_at is not None else now)
        if waited < self.settle_s:
            return Tick("settling", change, f"{self.settle_s - waited:.0f} s left")
        if busy:
            return Tick("deferred", change, busy)
        return Tick("fire", change)

    def consume(self) -> PoolChange | None:
        """A run started for the pending change: the pool it sees becomes the new baseline."""
        change = self.pending
        self._baseline = self._latest
        self._changed_at = None
        return change


def busy_reason(deployments: list[dict], tunes: list[dict], run_active: bool,
                benchmarks: list[dict] | None = None) -> str | None:
    """Why a run must wait, or None. A loading model, a tune trial or a benchmark is using the pool, so a run then
    would plan against memory that is about to change, and an apply would cut a measurement short."""
    if run_active:
        return "an autotune run is already in progress"
    for dep in deployments:
        if dep.get("state") in BUSY_DEPLOYMENT_STATES:
            return f"deployment {str(dep.get('id', ''))[:8]} is {dep.get('state')}"
    for tune in tunes:
        if tune.get("status") == "running":
            return f"tune {str(tune.get('id', ''))[:8]} is running"
    for bench in benchmarks or []:
        if bench.get("status") == "running":
            return f"benchmark {str(bench.get('id', ''))[:8]} is running"
    return None
