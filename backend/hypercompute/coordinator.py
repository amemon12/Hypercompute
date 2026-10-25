"""Builds the coordinator's service graph and keeps the host's own inventory fresh."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, engine, netutil
from .adapters import registry
from .adapters.llama_rpc import LlamaRpcAdapter
from .agent import inventory
from .agent.contribution import default_contribution
from .agent.supervisor import Supervisor
from .capacity.measurements import MeasurementStore
from .capacity.service import CapacityService
from .schemas import ContributionLimits, DeploymentState, EngineInfo
from .services.benchmarks import BenchmarkService
from .services.deployments import DeploymentError, DeploymentManager
from .services.events import EventBus
from .services.leases import LeaseManager
from .services.models import ModelService
from .services.nodes import NodeRegistry
from .services.profiles import ProfileStore
from .services.replan import ReplanController
from .services.sessions import SessionService
from .services.streaming import InferenceService
from .storage import Store

log = logging.getLogger("hypercompute.coordinator")


def load_secret(path: Path) -> bytes:
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="ascii").strip())
    key = secrets.token_bytes(32)
    path.write_text(key.hex(), encoding="ascii")
    if os.name != "nt":
        path.chmod(0o600)
    return key


@dataclass
class Coordinator:
    port: int
    owner_token: str
    store: Store
    events: EventBus
    sessions: SessionService
    nodes: NodeRegistry
    models: ModelService
    leases: LeaseManager
    adapter: LlamaRpcAdapter
    deployments: DeploymentManager
    inference: InferenceService
    benchmarks: BenchmarkService
    supervisor: Supervisor
    engine_info: EngineInfo
    startup_notes: list[str]
    adapter_ctx: registry.AdapterContext | None = None
    extra_adapters: dict[str, Any] = field(default_factory=dict)  # optional engines installed on this host
    profiles: ProfileStore | None = None
    tuner: Any = None  # services.tuner.TunerService when that module is part of the build
    replan: ReplanController | None = None  # follows node events only when HYPERCOMPUTE_REPLAN=1
    capacity: Any = None  # capacity.service.CapacityService: pools, optimizer plans and explanations

    async def aclose_adapters(self) -> None:
        for adapter in [self.adapter, *self.extra_adapters.values()]:
            try:
                await adapter.aclose()
            except Exception:  # closing one engine's client must not skip the others
                log.exception("closing an engine adapter failed")

    async def cleanup_orphans(self) -> None:
        """Once the gateway is up: engines remove what a crash left running with no local process (EXO instances on
        the Macs need HTTP, so this cannot run in the sync reconcile_on_start). Each outcome is a startup note."""
        try:
            notes = await self.deployments.cleanup_orphans()
        except Exception:  # the held leases stay held; the next start tries again
            log.exception("cleaning up engines left from before the restart failed")
            return
        for note in notes:
            self.events.publish("gateway.startup", note)

    async def refresh_local(self) -> None:
        """Re-measure host RAM / VRAM every few seconds so plans use fresh free-memory numbers."""
        while True:
            await asyncio.sleep(10.0)
            try:
                inv = self.nodes.local.inventory
                pools, devices = await asyncio.to_thread(inventory.fresh_measurements, inv)
                fresh = inv.model_copy(update={"memory_pools": pools, "devices": devices})
                self.nodes.update_local(fresh)
            except Exception:  # keep refreshing; the failure is logged with context
                log.exception("local inventory refresh failed")


def build(port: int, local_contribution: ContributionLimits | None = None,
          lan_override: list[str] | None = None) -> Coordinator:
    p = config.paths().ensure()
    lock = engine.load_lock()
    installed = engine.find_installed(lock)
    if installed is None:
        raise engine.EngineError("llama.cpp engine not installed", fix="uv run hypercompute engine install")
    store = Store(p.coordinator / "hypercompute.db", p.coordinator / "records")
    events = EventBus(store)
    inv = inventory.collect()
    if inv.engine.smoke != "passed":
        log.warning("engine smoke test did not pass: %s", inv.engine.smoke_detail)
    secret = load_secret(p.coordinator / "secret.key")

    def lan() -> list[str]:
        return lan_override if lan_override is not None else netutil.lan_addresses()

    sessions = SessionService(store, events, secret, port, lan)
    contribution = local_contribution or default_contribution(inv)
    nodes = NodeRegistry(store, events, sessions, secret, inv.engine, inv, contribution)
    supervisor = Supervisor(p.coordinator / "managed-processes.json")
    adapter = LlamaRpcAdapter(installed, lock, supervisor, p.coordinator / "logs")
    nodes.probe = adapter.probe_workers
    models = ModelService(events)
    for art in models.artifacts():  # adopt verified files already in the download cache
        if models.view(art.id).status == "not_downloaded":
            models.adopt_cached(art.id, [p.downloads / f.name for f in art.files])
    leases = LeaseManager(store)
    ctx = registry.AdapterContext(supervisor=supervisor, logs_dir=p.coordinator / "logs", paths=p, nodes=nodes)
    extra = registry.build_optional(ctx)
    deployments = DeploymentManager(store, events, nodes, models, leases, adapter, supervisor, extra_adapters=extra)
    deployments.adapter_ctx = ctx
    inference = InferenceService(store, events, deployments, models, adapter, nodes=nodes, extra_adapters=extra)
    inference.set_engine_info({"release": inv.engine.release, "commit": inv.engine.commit,
                               "variant": inv.engine.variant})
    capacity, replan = _build_capacity(nodes, models, leases, deployments, events, store)
    benchmarks = BenchmarkService(store, events, deployments, inference)
    profiles = ProfileStore(store)
    deployments.profiles = profiles
    tuner = _build_tuner(store, events, deployments, inference, profiles, models, nodes)
    notes = deployments.reconcile_on_start() + benchmarks.reconcile_on_start() + nodes.restore()
    if tuner is not None:
        notes += tuner.reconcile_on_start()
    return Coordinator(port=port, owner_token=secrets.token_urlsafe(32), store=store, events=events,
                       sessions=sessions, nodes=nodes, models=models, leases=leases, adapter=adapter,
                       deployments=deployments, inference=inference, benchmarks=benchmarks, supervisor=supervisor,
                       engine_info=inv.engine, startup_notes=notes, adapter_ctx=ctx, extra_adapters=extra,
                       profiles=profiles, tuner=tuner, replan=replan, capacity=capacity)


def _build_capacity(nodes: NodeRegistry, models: ModelService, leases: LeaseManager, deployments: DeploymentManager,
                    events: EventBus, store: Store) -> tuple[CapacityService, ReplanController]:
    """The capacity service (measurements in the runtime home), wired into planning and into the replan controller.
    The controller follows node events only when HYPERCOMPUTE_REPLAN=1."""
    capacity = CapacityService(nodes, models, leases, MeasurementStore(), deployment_ready=_ready_check(deployments))
    deployments.capacity = capacity
    replan = ReplanController(deployments, events, store, capacity.evaluate,
                              enabled=os.environ.get("HYPERCOMPUTE_REPLAN") == "1")
    return capacity, replan


# The engine has loaded its weights in these states (warming sends the first request after the load, draining stops a
# loaded engine), so the lease's bytes are inside the fresh free-memory observations and must not be subtracted again.
LOADED_STATES = frozenset({DeploymentState.warming, DeploymentState.ready, DeploymentState.draining})


def _ready_check(deployments: DeploymentManager) -> Callable[[str], bool]:
    """Whether a lease's deployment has loaded (its memory is then inside the fresh free-memory observations). While it
    loads, the lease counts as not loaded: the observation may hold part of it, so the reading errs on the low side."""
    def ready(deployment_id: str) -> bool:
        try:
            return deployments.get(deployment_id).state in LOADED_STATES
        except DeploymentError:  # no record: nothing is loaded for it
            return False
    return ready


def _build_tuner(*services: Any) -> Any:
    """TunerService(store, events, deployments, inference, profiles, models, nodes), when the module exists."""
    try:
        from .services.tuner import TunerService
    except ModuleNotFoundError:
        log.info("measured tuner not part of this build")
        return None
    return TunerService(*services)
