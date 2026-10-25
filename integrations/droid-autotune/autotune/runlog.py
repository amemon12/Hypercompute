"""Every autotune run as a record the page shows and a JSONL audit trail on disk (runs/<id>.jsonl, one line per step,
plus runs/<id>.json with the final record). Records are frozen; the store swaps in an updated copy."""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import AutotuneProposal, DeviceRow, LayoutDiff, Verdict

log = logging.getLogger("autotune")

RunState = Literal["running", "awaiting_approval", "no_change", "failed", "cancelled", "superseded", "rejected",
                   "approved", "applying", "applied", "apply_failed"]
TERMINAL = frozenset({"no_change", "failed", "cancelled", "superseded", "rejected", "applied", "apply_failed",
                      "approved"})
OPEN = frozenset({"awaiting_approval"})


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    at: str
    kind: str  # trigger, context, session, guard, plan, droid, tool, permission, verdict, proposal, diff, owner, apply
    title: str
    detail: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created_at: str
    runner: str  # "droid" for a real Factory Droid session, "scripted" only in tests and the labelled offline demo
    trigger: dict[str, Any]
    state: RunState = "running"
    model_id: str | None = None
    deployment_id: str | None = None
    current_rows: list[DeviceRow] = Field(default_factory=list)
    node_names: dict[str, str] = Field(default_factory=dict)  # node id -> display name, for the page
    current_plan: dict[str, Any] | None = None
    plan_title: str | None = None
    plan_text: str | None = None
    steps: list[Step] = Field(default_factory=list)
    proposal: AutotuneProposal | None = None
    verdicts: list[Verdict] = Field(default_factory=list)
    diffs: dict[int, LayoutDiff] = Field(default_factory=dict)  # by index into proposal.candidates
    candidate_verdicts: dict[int, int] = Field(default_factory=dict)  # candidate index -> index into verdicts
    measured: dict[str, Any] | None = None  # the current deployment's last benchmark, if the gateway has one
    usage: dict[str, Any] | None = None
    error: str | None = None
    applied: dict[str, Any] | None = None
    finished_at: str | None = None


class RunStore:
    def __init__(self, runs_dir: Path | None, keep: int = 50) -> None:
        self.runs_dir = runs_dir
        if runs_dir is not None:
            runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, RunRecord] = {}
        self._order: deque[str] = deque(maxlen=keep)
        self.watch_log: deque[dict[str, Any]] = deque(maxlen=200)

    def create(self, *, runner: str, trigger: dict[str, Any]) -> RunRecord:
        run = RunRecord(id=uuid.uuid4().hex[:12], created_at=now_iso(), runner=runner, trigger=trigger)
        if len(self._order) == self._order.maxlen:
            self._runs.pop(self._order[0], None)
        self._order.append(run.id)
        self._runs[run.id] = run
        self._persist(run)
        return run

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list(self) -> list[RunRecord]:
        return [self._runs[i] for i in reversed(self._order) if i in self._runs]

    def update(self, run_id: str, **changes: Any) -> RunRecord:
        run = self._runs[run_id]
        if "state" in changes and changes["state"] in TERMINAL and run.finished_at is None:
            changes = {**changes, "finished_at": now_iso()}
        updated = run.model_copy(update=changes)
        self._runs[run_id] = updated
        self._persist(updated)
        return updated

    def step(self, run_id: str, kind: str, title: str, detail: dict[str, Any] | None = None) -> Step:
        run = self._runs[run_id]
        step = Step(seq=len(run.steps) + 1, at=now_iso(), kind=kind, title=title, detail=detail or {})
        self._runs[run_id] = run.model_copy(update={"steps": [*run.steps, step]})
        log.info("run %s [%s] %s", run_id, kind, title)
        if self.runs_dir is not None:
            with (self.runs_dir / f"{run_id}.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(step.model_dump(mode="json")) + "\n")
        return step

    def watch(self, action: str, detail: str) -> None:
        self.watch_log.append({"at": now_iso(), "action": action, "detail": detail})

    def _persist(self, run: RunRecord) -> None:
        if self.runs_dir is None:
            return
        path = self.runs_dir / f"{run.id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(run.model_dump(mode="json", exclude={"steps"}), indent=1), encoding="utf-8",
                       newline="\n")
        tmp.replace(path)
