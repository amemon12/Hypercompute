"""Installers for the optional engines. Each lives in its own environment, never in the gateway's venv.

Every module here (airllm, vllm, exo) exposes:
    install(paths: config.Paths, *, yes: bool, force: bool, say: Callable[[str], None]) -> int   (exit code)
    status(paths: config.Paths) -> ExtraEngineInfo                                                 (cheap, no network)
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from .. import config
from ..schemas import ExtraEngineInfo

log = logging.getLogger("hypercompute.installers")
OPTIONAL_ENGINES = ("airllm", "vllm", "exo")


def load(engine_id: str) -> ModuleType | None:
    if engine_id not in OPTIONAL_ENGINES:
        raise ValueError(f"unknown engine {engine_id!r}; choose one of: llama, {', '.join(OPTIONAL_ENGINES)}")
    try:
        return importlib.import_module(f"{__name__}.{engine_id}")
    except ModuleNotFoundError as exc:
        log.info("installer for %s not available: %s", engine_id, exc)
        return None


def status_all(paths: config.Paths | None = None) -> list[ExtraEngineInfo]:
    """What this machine can run besides llama.cpp; reported in its inventory so the host can plan for it."""
    paths = paths or config.paths()
    out = []
    for engine_id in OPTIONAL_ENGINES:
        mod = load(engine_id)
        if mod is None:
            out.append(ExtraEngineInfo(engine=engine_id, installed=False, status="not_installed",  # type: ignore[arg-type]
                                       detail="installer missing from this build"))
            continue
        try:
            out.append(mod.status(paths))
        except Exception as exc:  # a broken optional engine must not break the inventory
            log.warning("status check for %s failed", engine_id, exc_info=True)
            out.append(ExtraEngineInfo(engine=engine_id, installed=False, status="failed",  # type: ignore[arg-type]
                                       detail=str(exc)[:300]))
    return out
