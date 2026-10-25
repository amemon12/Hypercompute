"""Measured profiles: configurations that were actually run, reused by the planner for the same setup.

The handover's rule: conservative admission first, measured ranking second. A profile is keyed by a
signature of the model, context size and the exact set of (node, engine device) pairs it used, so it is only
reused when the same machines with the same devices are present again.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable

from ..schemas import MeasuredProfile, Plan, utcnow
from ..storage import Store, dumps

PLAN_FIELDS = ("device_list", "tensor_split", "cpu_layers", "n_gpu_units", "engine_flags", "rpc_endpoints",
               "tier", "engine")


def signature(model_id: str, ctx_size: int, devices: list[tuple[str, str]]) -> str:
    """Stable key: model, context and the sorted (node_id, device description or name) pairs available."""
    parts = sorted(f"{node}={dev}" for node, dev in devices)
    return f"{model_id}|ctx={ctx_size}|" + ",".join(parts)


class ProfileStore:
    def __init__(self, store: Store) -> None:
        self._store = store  # the profiles table is created with the rest of the schema in storage.SCHEMA

    def save(self, *, model_id: str, sig: str, objective: str, plan: Plan, ttft_ms: float | None,
             decode_tps: float | None, prefill_tps: float | None, score: float, tune_id: str | None,
             run_ids: list[str]) -> MeasuredProfile:
        profile = MeasuredProfile(
            id=uuid.uuid4().hex, model_id=model_id, signature=sig, objective=objective,  # type: ignore[arg-type]
            plan_fields={k: getattr(plan, k) for k in PLAN_FIELDS}, ttft_ms=ttft_ms, decode_tokens_per_s=decode_tps,
            prefill_tokens_per_s=prefill_tps, score=score, tune_id=tune_id, run_ids=run_ids, created_at=utcnow())
        self._store.execute(
            "INSERT INTO profiles (id, model_id, signature, objective, score, created_at, doc) VALUES (?,?,?,?,?,?,?)",
            (profile.id, model_id, sig, objective, score, profile.created_at.isoformat(),
             dumps(profile.model_dump(mode="json"))))
        return profile

    def best(self, model_id: str, sig: str, objective: str,
             accept: Callable[[MeasuredProfile], bool] | None = None) -> MeasuredProfile | None:
        """The lowest-score profile for this key that `accept` takes (any, without it). Profiles measured under other
        limits (RAM tier, capacity mode, strategy) share the key, so a refused one must not hide the next."""
        rows = self._store.query(
            "SELECT doc FROM profiles WHERE model_id=? AND signature=? AND objective=? ORDER BY score ASC, "
            "created_at DESC", (model_id, sig, objective))
        for row in rows:
            profile = MeasuredProfile.model_validate(json.loads(row["doc"]))
            if accept is None or accept(profile):
                return profile
        return None

    def list(self, model_id: str | None = None) -> list[MeasuredProfile]:
        if model_id:
            rows = self._store.query("SELECT doc FROM profiles WHERE model_id=? ORDER BY created_at DESC", (model_id,))
        else:
            rows = self._store.query("SELECT doc FROM profiles ORDER BY created_at DESC LIMIT 200")
        return [MeasuredProfile.model_validate(json.loads(r["doc"])) for r in rows]
