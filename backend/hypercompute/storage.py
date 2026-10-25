"""SQLite document store plus append-only JSONL records for runs and events.

Rows hold a JSON document plus a few indexed columns. Writes go through one connection guarded by
a lock; every multi-row change runs in a single transaction so lease reservation stays atomic.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
  doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invites (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, code_hash TEXT NOT NULL UNIQUE,
  expires_at TEXT NOT NULL, used_at TEXT, failed_attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, session_id TEXT, credential_hash TEXT, state TEXT NOT NULL,
  doc TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS deployments (id TEXT PRIMARY KEY, state TEXT NOT NULL, idempotency_key TEXT UNIQUE,
  created_at TEXT NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, deployment_id TEXT NOT NULL, node_id TEXT NOT NULL,
  pool_id TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL, released_at TEXT,
  revision INTEGER NOT NULL DEFAULT 0, chain_id TEXT);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, deployment_id TEXT, kind TEXT NOT NULL, status TEXT NOT NULL,
  started_at TEXT NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS benchmarks (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, model_id TEXT NOT NULL, signature TEXT NOT NULL,
  objective TEXT NOT NULL, score REAL NOT NULL, created_at TEXT NOT NULL, doc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS profiles_lookup ON profiles (model_id, signature, objective);
CREATE TABLE IF NOT EXISTS tunes (id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
  doc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS leases_active ON leases (node_id, pool_id) WHERE released_at IS NULL;
"""

# Columns added after a table first shipped: (table, column, declaration). A database created by an older build gets
# them when it opens; CREATE TABLE above already has them for a new one.
# leases.revision: the plan revision the lease was reserved for. leases.chain_id: the first deployment of a replacement
# chain (NULL: the deployment is its own chain).
ADDED_COLUMNS = (
    ("leases", "revision", "INTEGER NOT NULL DEFAULT 0"),
    ("leases", "chain_id", "TEXT"),
)


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(doc: Any) -> str:
    return json.dumps(doc, default=_default, separators=(",", ":"))


class Store:
    def __init__(self, db_path: Path, jsonl_dir: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_dir = jsonl_dir
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._add_missing_columns()

    def _add_missing_columns(self) -> None:
        for table, column, declaration in ADDED_COLUMNS:
            present = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in present:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Serializable write transaction (BEGIN IMMEDIATE) under the process-wide lock."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self.tx() as conn:
            return conn.execute(sql, params).rowcount

    def append_jsonl(self, name: str, doc: Any) -> None:
        path = self._jsonl_dir / f"{name}.jsonl"
        line = dumps(doc) + "\n"
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
