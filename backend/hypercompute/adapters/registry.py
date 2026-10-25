"""Builds the optional engine adapters (AirLLM, vLLM, EXO). Each is a separate module, imported lazily.

Each optional engine module (hypercompute.adapters.<engine>) exposes:
    build(ctx: AdapterContext) -> EngineAdapter | None
        None when the engine's environment is not installed on this host.
    status(ctx: AdapterContext) -> ExtraEngineInfo
        Whether the host can run it now, and why not.
    plan(req: PlanRequest, art: ModelArtifact, local: NodeView, remote: list[NodeView], ctx: AdapterContext) -> Plan
        A plan for this engine alone (feasible=False with reasons when it cannot serve the request). Pure apart from
        cheap local reads (e.g. nvidia-smi free memory); it never starts processes.
So the gateway runs even when these engines are absent, and reports why each one is unavailable.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .. import config
from ..agent.supervisor import Supervisor
from ..schemas import ExtraEngineInfo, ModelArtifact, NodeView, Plan, PlanRequest

log = logging.getLogger("hypercompute.registry")

OPTIONAL_ENGINES: dict[str, str] = {
    "airllm": "hypercompute.adapters.airllm",
    "vllm": "hypercompute.adapters.vllm",
    "exo": "hypercompute.adapters.exo",
}


@dataclass(frozen=True)
class AdapterContext:
    supervisor: Supervisor
    logs_dir: Path
    paths: config.Paths
    nodes: Any = None  # services.nodes.NodeRegistry: EXO finds the Mac peers' EXO APIs through it


def _module(engine_id: str) -> ModuleType | None:
    try:
        return importlib.import_module(OPTIONAL_ENGINES[engine_id])
    except ModuleNotFoundError as exc:
        log.info("engine module for %s not available: %s", engine_id, exc)
        return None


def build_optional(ctx: AdapterContext) -> dict[str, Any]:
    adapters: dict[str, Any] = {}
    for engine_id in OPTIONAL_ENGINES:
        mod = _module(engine_id)
        if mod is None:
            continue
        try:
            adapter = mod.build(ctx)
        except Exception:  # an optional engine must never stop the gateway from starting
            log.exception("could not build the %s adapter", engine_id)
            continue
        if adapter is not None:
            adapters[engine_id] = adapter
    return adapters


def engine_status(ctx: AdapterContext) -> list[ExtraEngineInfo]:
    out = []
    for engine_id in OPTIONAL_ENGINES:
        mod = _module(engine_id)
        if mod is None:
            out.append(ExtraEngineInfo(engine=engine_id, installed=False, status="not_installed",  # type: ignore[arg-type]
                                       detail="adapter module missing from this build"))
            continue
        try:
            out.append(mod.status(ctx))
        except Exception as exc:  # report, never crash the status page
            out.append(ExtraEngineInfo(engine=engine_id, installed=False, status="failed",  # type: ignore[arg-type]
                                       detail=str(exc)[:300]))
    return out


def plan_for(engine_id: str, req: PlanRequest, art: ModelArtifact, local: NodeView, remote: list[NodeView],
             ctx: AdapterContext) -> Plan | None:
    """The engine's own plan, or None when its module is missing from this build."""
    mod = _module(engine_id)
    return None if mod is None else mod.plan(req, art, local, remote, ctx)
