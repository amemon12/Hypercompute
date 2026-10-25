"""Atomic host-memory leases. One lease row per (deployment, node, physical pool).

Budgets are enforced per physical pool, so a unified-memory pool (Apple, GB10) is counted once even
when several devices draw from it, and per node against the contributor's total allowance.

Each lease carries the plan revision it was reserved for and its replacement chain (the first deployment of a chain
of replacements). Revisions only grow along a chain: a reservation or a revisioned release older than the newest
revision reserved in the chain is refused, so a late cleanup of an old revision cannot free a newer one's memory.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass

from ..schemas import Plan, utcnow
from ..storage import Store


class LeaseConflict(RuntimeError):
    pass


class _DryRun(Exception):
    """Rolls back an admission check's transaction."""


@dataclass(frozen=True)
class LeaseRequest:
    node_id: str
    pool_id: str
    bytes: int
    pool_budget: int
    node_budget: int


def requests_from_plan(plan: Plan, node_budgets: dict[str, int]) -> list[LeaseRequest]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    budgets: dict[tuple[str, str], int] = {}
    for a in plan.assignments:
        key = (a.node_id, a.pool_id)
        grouped[key] += a.total_bytes_est
        budgets[key] = max(budgets.get(key, 0), a.budget_bytes)
    return [LeaseRequest(node_id=n, pool_id=p, bytes=b, pool_budget=budgets[(n, p)],
                         node_budget=node_budgets.get(n, budgets[(n, p)]))
            for (n, p), b in grouped.items()]


def _admit(conn: sqlite3.Connection, r: LeaseRequest) -> None:
    pool_used = conn.execute(
        "SELECT COALESCE(SUM(bytes),0) FROM leases WHERE node_id=? AND pool_id=? AND released_at IS NULL",
        (r.node_id, r.pool_id)).fetchone()[0]
    if pool_used + r.bytes > r.pool_budget:
        raise LeaseConflict(f"{r.node_id}/{r.pool_id}: {pool_used + r.bytes} bytes requested over "
                            f"budget {r.pool_budget}")
    node_used = conn.execute(
        "SELECT COALESCE(SUM(bytes),0) FROM leases WHERE node_id=? AND released_at IS NULL",
        (r.node_id,)).fetchone()[0]
    if node_used + r.bytes > r.node_budget:
        raise LeaseConflict(f"{r.node_id}: {node_used + r.bytes} bytes requested over node allowance "
                            f"{r.node_budget}")


def _chain_of(conn: sqlite3.Connection, deployment_id: str) -> str | None:
    row = conn.execute("SELECT COALESCE(chain_id, deployment_id) FROM leases WHERE deployment_id=? LIMIT 1",
                       (deployment_id,)).fetchone()
    return row[0] if row else None


def _newest_revision(conn: sqlite3.Connection, chain: str) -> int | None:
    return conn.execute("SELECT MAX(revision) FROM leases WHERE COALESCE(chain_id, deployment_id)=?",
                        (chain,)).fetchone()[0]


class LeaseManager:
    def __init__(self, store: Store) -> None:
        self._store = store

    def reserve(self, deployment_id: str, requests: list[LeaseRequest], revision: int = 0,
                chain: str | None = None) -> list[str]:
        """All-or-nothing reservation inside one IMMEDIATE transaction. `chain` names the first deployment of the
        replacement chain this one belongs to (default: itself); a revision older than the chain's newest is refused."""
        with self._store.tx() as conn:
            return self._insert(conn, deployment_id, requests, revision, chain or deployment_id)

    def admits(self, requests: list[LeaseRequest], releasing: str | None = None) -> str | None:
        """None when the requests would be reserved right now, else the conflict. Nothing is kept. `releasing` names a
        deployment whose leases count as released (a replacement checks its plan before it stops the old one)."""
        try:
            with self._store.tx() as conn:
                if releasing is not None:
                    conn.execute("UPDATE leases SET released_at=? WHERE deployment_id=? AND released_at IS NULL",
                                 (utcnow().isoformat(), releasing))
                self._insert(conn, f"admission-check-{uuid.uuid4().hex}", requests, 0, None)
                raise _DryRun
        except _DryRun:
            return None
        except LeaseConflict as exc:
            return str(exc)

    @staticmethod
    def _insert(conn: sqlite3.Connection, deployment_id: str, requests: list[LeaseRequest], revision: int,
                chain: str | None) -> list[str]:
        if chain is not None:
            newest = _newest_revision(conn, chain)
            if newest is not None and revision < newest:
                raise LeaseConflict(f"stale reservation: revision {revision} of deployment {deployment_id[:8]} is "
                                    f"older than revision {newest} already reserved in its chain")
        ids: list[str] = []
        now = utcnow().isoformat()
        for r in requests:
            _admit(conn, r)
            lease_id = uuid.uuid4().hex
            conn.execute("INSERT INTO leases (id, deployment_id, node_id, pool_id, bytes, created_at, revision, "
                         "chain_id) VALUES (?,?,?,?,?,?,?,?)",
                         (lease_id, deployment_id, r.node_id, r.pool_id, r.bytes, now, revision,
                          None if chain == deployment_id else chain))
            ids.append(lease_id)
        return ids

    def release(self, deployment_id: str, revision: int | None = None) -> int:
        """Release the deployment's active leases; with `revision`, only that revision's. A revision older than the
        newest one reserved in the deployment's chain raises LeaseConflict and releases nothing."""
        now = utcnow().isoformat()
        with self._store.tx() as conn:
            if revision is None:
                return conn.execute("UPDATE leases SET released_at=? WHERE deployment_id=? AND released_at IS NULL",
                                    (now, deployment_id)).rowcount
            chain = _chain_of(conn, deployment_id)
            newest = _newest_revision(conn, chain) if chain is not None else None
            if newest is not None and revision < newest:
                raise LeaseConflict(f"stale release: revision {revision} of deployment {deployment_id[:8]} is older "
                                    f"than revision {newest} of its chain")
            return conn.execute("UPDATE leases SET released_at=? WHERE deployment_id=? AND revision=? "
                                "AND released_at IS NULL", (now, deployment_id, revision)).rowcount

    def active(self) -> list[dict]:
        return [dict(r) for r in self._store.query("SELECT * FROM leases WHERE released_at IS NULL")]

    def reconcile(self, live_deployment_ids: set[str]) -> int:
        stale = {r["deployment_id"] for r in self.active()} - live_deployment_ids
        return sum(self.release(d) for d in stale)
