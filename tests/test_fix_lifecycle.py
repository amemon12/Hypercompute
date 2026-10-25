"""Deployment cleanup races: one cleanup per deployment, no orphaned llama-server, no early terminal state."""

import asyncio
import time

import pytest
from hypercompute.adapters.base import DeploymentHandle, StopResult
from hypercompute.schemas import CreateDeploymentRequest, DeploymentState
from test_deployments import FakeAdapter, _setup, _wait_terminal_or_ready

PEER = "a" * 32


class SlowStartAdapter(FakeAdapter):
    """prepare() starts llama-server from a worker thread, like the real adapter (a cancel cannot stop it)."""

    def __init__(self):
        super().__init__()
        self.preparing = asyncio.Event()

    async def prepare(self, spec):
        def start():
            time.sleep(0.3)
            self.started += 1
            return DeploymentHandle(deployment_id=spec.deployment_id, process_key="k", base_url="http://127.0.0.1:1",
                                    log_path=None, pid=1, args=[])  # type: ignore[arg-type]

        self.preparing.set()
        return await asyncio.to_thread(start)


class GatedStopAdapter(FakeAdapter):
    """stop() blocks until the test releases it, like a llama-server that takes the full grace period."""

    def __init__(self, clean=True, **kw):
        super().__init__(**kw)
        self.clean = clean
        self.stop_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stop(self, handle):
        self.stopped += 1
        self.stop_entered.set()
        await self.release.wait()
        return StopResult(clean=self.clean, detail="terminated" if self.clean else "pid 1: still_running")


async def _ready(mgr):
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.ready
    return dep


@pytest.mark.anyio
@pytest.mark.parametrize("how", ["stop", "lost"])
async def test_cancel_while_llama_server_is_starting_stops_the_started_process(store, events, tmp_path, how):
    adapter = SlowStartAdapter()
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)
    await asyncio.wait_for(adapter.preparing.wait(), 5)
    if how == "stop":
        await mgr.stop(dep.id)
    else:
        await mgr.on_node_lost(PEER, "no heartbeat for 16s")
    final = mgr.get(dep.id)
    assert adapter.started == 1 and adapter.stopped == 1  # the process that was starting got stopped
    assert final.state == (DeploymentState.stopped if how == "stop" else DeploymentState.failed)
    assert leases.active() == [] and mgr.handle(dep.id) is None


@pytest.mark.anyio
async def test_stop_and_node_loss_share_one_cleanup(store, events, tmp_path):
    adapter = GatedStopAdapter()
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _ready(mgr)
    stop_task = asyncio.create_task(mgr.stop(dep.id))
    await asyncio.wait_for(adapter.stop_entered.wait(), 5)
    lost_task = asyncio.create_task(mgr.on_node_lost(PEER, "paused by contributor"))
    await asyncio.sleep(0.05)
    # llama-server is still being stopped: not terminal yet, leases still held, a new deployment must wait
    assert mgr.active() is not None and mgr.get(dep.id).state == DeploymentState.draining
    assert mgr.busy(PEER) == [dep.id] and len(leases.active()) == 2
    adapter.release.set()
    final = await asyncio.wait_for(stop_task, 5)
    await asyncio.wait_for(lost_task, 5)
    assert final.state == DeploymentState.stopped and final.error is None  # the first caller's outcome wins
    assert [h.state for h in final.history][-2:] == [DeploymentState.draining, DeploymentState.stopped]
    assert mgr.get(dep.id).state == DeploymentState.stopped
    assert adapter.stopped == 1 and leases.active() == []


@pytest.mark.anyio
async def test_repeated_node_loss_transitions_once(store, events, tmp_path):
    adapter = GatedStopAdapter()
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _ready(mgr)
    first = asyncio.create_task(mgr.on_node_lost(PEER, "paused by contributor"))
    await asyncio.wait_for(adapter.stop_entered.wait(), 5)
    second = asyncio.create_task(mgr.on_node_lost(PEER, "paused by contributor"))  # the agent's retried heartbeat
    await asyncio.sleep(0.05)
    assert mgr.active() is not None and len(leases.active()) == 2
    adapter.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 5)
    final = mgr.get(dep.id)
    states = [h.state for h in final.history]
    assert final.state == DeploymentState.failed and states.count(DeploymentState.failed) == 1
    assert DeploymentState.stopped not in states and adapter.stopped == 1 and leases.active() == []


@pytest.mark.anyio
async def test_stop_during_startup_failure_cleanup_waits_for_it(store, events, tmp_path):
    adapter = GatedStopAdapter(warm_ok=False)
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)
    await asyncio.wait_for(adapter.stop_entered.wait(), 5)  # the failed warmup is stopping llama-server
    stop_task = asyncio.create_task(mgr.stop(dep.id))
    await asyncio.sleep(0.05)
    assert not stop_task.done() and mgr.active() is not None
    adapter.release.set()
    final = await asyncio.wait_for(stop_task, 5)
    assert final.state == DeploymentState.failed and "warmup" in final.error
    assert mgr.active() is None and leases.active() == [] and adapter.stopped == 1


@pytest.mark.anyio
async def test_unclean_stop_keeps_leases_and_says_so(store, events, tmp_path):
    adapter = GatedStopAdapter(clean=False)
    adapter.release.set()
    mgr, _, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _ready(mgr)
    final = await mgr.stop(dep.id)
    assert final.state == DeploymentState.failed and final.error
    assert "pid 1 may still be running" in final.cleanup and "leases are held" in final.cleanup
    assert "next gateway start" in final.cleanup and "released" not in final.cleanup
    assert len(leases.active()) == 2


@pytest.mark.anyio
async def test_restart_keeps_leases_of_a_llama_server_that_survived(store, events, tmp_path, monkeypatch):
    from hypercompute.agent.supervisor import Supervisor
    from hypercompute.services.deployments import DeploymentManager
    from test_deployments import FakeModels

    adapter = GatedStopAdapter(clean=False)
    adapter.release.set()
    mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
    dep = await _ready(mgr)
    await mgr.stop(dep.id)
    sup = Supervisor(tmp_path / "restart.json")
    sup._write_records([{"key": f"deployment:{dep.id}", "pid": 1, "create_time": None, "exe": "x"}])
    monkeypatch.setattr(sup, "reconcile_orphans", lambda: ["deployment: pid 1 could not be stopped; record kept"])
    notes = DeploymentManager(store, events, reg, FakeModels(), leases, adapter, sup).reconcile_on_start()
    assert len(leases.active()) == 2 and any("kept the leases of 1" in n for n in notes)
    sup._write_records([])  # once the process is confirmed gone, the next start releases them
    monkeypatch.setattr(sup, "reconcile_orphans", lambda: [])
    DeploymentManager(store, events, reg, FakeModels(), leases, adapter, sup).reconcile_on_start()
    assert leases.active() == []
