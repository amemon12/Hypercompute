"""Small durable state in agent_data/: atomic JSON files and an append-only log.

Each file is written to a temp file and renamed over the old one, so a crash never leaves half a file. A file that
exists but cannot be read is a hard error (fail closed): silently starting empty would forget payments already used
or credits owed.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


class StoreError(RuntimeError):
    """A state file exists but is unreadable. The message names the file so the owner can inspect or move it."""


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise StoreError(f"{path} is unreadable ({type(exc).__name__}); move it aside to start fresh") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"at": datetime.now(UTC).isoformat(), **record}) + "\n")


@dataclass(frozen=True)
class Credit:
    """A verified payment that has not yet bought an answer. It is written the moment the payment verifies (before
    anything else can fail) and deleted only once the answer it paid for has been sent, so a crash, a restart or a
    failed load can never lose it. It covers one later request from the same sender that costs no more."""

    tx_id: str
    sender: str
    amount: str
    prompt: str
    max_tokens: int
    model_hint: str | None = None
    preference: str = "biggest"
    scope: str = "pool"


class Credits:
    """Unspent payments, keyed by transaction id (upper case). Every change is written to disk first and applied
    in memory only once the write succeeded, so memory and file never disagree about what is spent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        raw = dict(read_json(path, {}))
        migrated = self._migrate(raw)
        if migrated != raw:
            write_json(path, migrated)  # a failure here is a hard error, like any unreadable state file
        self._data: dict[str, dict] = migrated

    @staticmethod
    def _migrate(raw: dict[str, dict]) -> dict[str, dict]:
        """The first format (bd0de3d's parent) was keyed by sender, one credit each, with no sender field."""
        out: dict[str, dict] = {}
        for key, value in raw.items():
            if isinstance(value, dict) and "sender" not in value and value.get("tx_id"):
                value = {**value, "sender": key}
            if isinstance(value, dict) and value.get("tx_id"):
                out[str(value["tx_id"]).upper()] = {**value, "tx_id": str(value["tx_id"]).upper()}
        return out

    def _commit(self, data: dict[str, dict]) -> None:
        write_json(self.path, data)
        self._data = data

    def add(self, credit: Credit) -> None:
        self._commit({**self._data, credit.tx_id: asdict(credit)})

    def has(self, tx_id: str) -> bool:
        return tx_id in self._data

    def for_sender(self, sender: str) -> list[Credit]:
        return [Credit(**raw) for raw in self._data.values() if raw.get("sender") == sender]

    def latest(self, sender: str) -> Credit | None:
        mine = self.for_sender(sender)
        return mine[-1] if mine else None

    def retarget(self, tx_id: str, **request: object) -> None:
        """Point an unspent payment at the request that just failed, so "run it" retries that one."""
        if tx_id in self._data:
            self._commit({**self._data, tx_id: {**self._data[tx_id], **request}})

    def spend(self, tx_id: str) -> None:
        if tx_id in self._data:
            self._commit({k: v for k, v in self._data.items() if k != tx_id})
