"""Who may do what, how often, and one run at a time.

Roles: owner/friends on the allowlist run prompts; everyone else is a guest whose access the owner sets
(none, read-only pool status, or paid runs). Limits are sliding windows per sender plus one global run slot, so a
stranger can never queue work on friends' machines and even allowed senders cannot flood the pool.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

READ_ACTIONS = frozenset({"status", "help", "cancel"})
RUN_ACTIONS = frozenset({"run", "approve", "edit"})
OWNER_ACTIONS = frozenset({"stop"})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    role: str  # "member" | "guest"
    needs_payment: bool = False
    reason: str | None = None
    retry_after_s: float | None = None


class SlidingWindow:
    MAX_KEYS = 10_000  # senders are free to create, so idle ones are evicted

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}

    def _evict(self, now: float) -> None:
        self._hits = {k: h for k, h in self._hits.items() if h and now - h[-1] < self.window_s}

    def _trim(self, key: str, now: float) -> deque[float]:
        if len(self._hits) > self.MAX_KEYS:
            self._evict(now)
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        return hits

    def check(self, key: str) -> float | None:
        """None when allowed (and counted), else seconds until the next slot frees."""
        now = self._clock()
        hits = self._trim(key, now)
        if len(hits) >= self.limit:
            return max(0.0, self.window_s - (now - hits[0]))
        hits.append(now)
        return None

    def refund(self, key: str) -> None:
        hits = self._hits.get(key)
        if hits:
            hits.pop()


class Gatekeeper:
    def __init__(self, *, allowed: frozenset[str], guest_access: str, payments: str, runs_per_window: int,
                 window_s: float, messages_per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.allowed, self.guest_access, self.payments = allowed, guest_access, payments
        self.runs = SlidingWindow(runs_per_window, window_s, clock)
        self.messages = SlidingWindow(messages_per_minute, 60.0, clock)
        self.guest_runs = SlidingWindow(max(1, runs_per_window // 2), window_s, clock)
        # plans cost gateway work before any payment, so guests' plans are limited per sender and in total
        self.guest_plans = SlidingWindow(max(2, runs_per_window), window_s, clock)
        self.all_guest_plans = SlidingWindow(max(10, runs_per_window * 10), window_s, clock)

    def role(self, sender: str) -> str:
        return "member" if sender in self.allowed else "guest"

    def message(self, sender: str) -> Decision:
        """Every inbound message, before any parsing: the cheap flood guard."""
        wait = self.messages.check(sender)
        role = self.role(sender)
        if wait is not None:
            return Decision(False, role, reason="too many messages", retry_after_s=wait)
        return Decision(True, role)

    def action(self, sender: str, action: str) -> Decision:
        role = self.role(sender)
        if action in READ_ACTIONS:
            if role == "guest" and self.guest_access == "none" and action == "status":
                return Decision(False, role, reason="guests may not see this pool")
            return Decision(True, role)
        if action in OWNER_ACTIONS:
            return Decision(role == "member", role, reason=None if role == "member" else "only members may stop it")
        if action in RUN_ACTIONS:
            if role == "member":
                return Decision(True, role, needs_payment=self.payments == "everyone")
            if self.guest_access == "paid":
                return Decision(True, role, needs_payment=True)
            return Decision(False, role, reason="not on this pool's allowlist")
        return Decision(False, role, reason=f"unknown action {action!r}")

    def propose(self, sender: str) -> float | None:
        """Count a plan (gateway plan-only calls). Members are bounded by runs; guests here. None when allowed."""
        if self.role(sender) == "member":
            return None
        wait = self.guest_plans.check(sender)
        if wait is not None:
            return wait
        total = self.all_guest_plans.check("*")
        if total is not None:
            self.guest_plans.refund(sender)
        return total

    def start_run(self, sender: str) -> float | None:
        """Count a run against the sender's window. None when allowed, else seconds to wait."""
        window = self.runs if self.role(sender) == "member" else self.guest_runs
        return window.check(sender)

    def refund_run(self, sender: str) -> None:
        """A run that never reached the pool (cancelled, refused) does not count."""
        window = self.runs if self.role(sender) == "member" else self.guest_runs
        window.refund(sender)


class SeenMessages:
    """Drops redelivered messages (ASI:One and the mailbox retry), keeping the last few thousand ids."""

    def __init__(self, capacity: int = 4096) -> None:
        self._order: deque[str] = deque()
        self._ids: set[str] = set()
        self._capacity = capacity

    def first_time(self, msg_id: str) -> bool:
        if msg_id in self._ids:
            return False
        self._ids.add(msg_id)
        self._order.append(msg_id)
        if len(self._order) > self._capacity:
            self._ids.discard(self._order.popleft())
        return True
