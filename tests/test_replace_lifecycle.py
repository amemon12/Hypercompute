"""Request-boundary replacement: revisions, drain, stale lease release, rollback and fallback (fake engine)."""

import asyncio
import sqlite3

import pytest
from hypercompute.adapters.base import EngineAdapterError, StopResult
from hypercompute.schemas import CreateDeploymentRequest, DeploymentState
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.leases import LeaseConflict, LeaseManager, LeaseRequest
from hypercompute.services.streaming import InferenceService
from hypercompute.storage import Store
from hypercompute.units import GiB
from test_deployments import FakeAdapter, FakeModels, _plan, _setup, _wait_terminal_or_ready

PEER = "a" * 32


class TrackingAdapter(FakeAdapter):
    """Logs engine starts and stops in order, counts engines alive at once, and fails the loads of chosen plans."""

    def __init__(self, fail_plans=(), clean_stop=True):
        super().__init__()
        self.log: list[tuple[str, str]] = []
        self.alive = self.max_alive = 0
        self.fail_plans, self.clean_stop = set(fail_plans), clean_stop
        self.plan_of: dict[str, str] = {}
        self.leases: LeaseManager | None = None
        self.leased_at_stop: list[list[str]] = []
        self.on_prepare = None
        self.gate: asyncio.Event | None = None  # when set, a load waits for it

    async def prepare(self, spec):
        handle = await super().prepare(spec)
        self.plan_of[spec.deployment_id] = spec.plan.id
        self.alive += 1
        self.max_alive = max(self.max_alive, self.alive)
        self.log.append(("start", spec.deployment_id))
        if self.on_prepare is not None:
            self.on_prepare(spec)
        return handle

    async def wait_ready(self, handle, timeout_s=0, on_progress=None):
        if self.gate is not None:
            await self.gate.wait()
        if self.plan_of.get(handle.deployment_id) in self.fail_plans:
            raise EngineAdapterError("load_timeout", f"plan {self.plan_of[handle.deployment_id]} did not load")
        await super().wait_ready(handle, timeout_s, on_progress)

    async def stop(self, handle):
        if self.leases is not None:
            self.leased_at_stop.append(sorted({r["deployment_id"] for r in self.leases.active()}))
        self.stopped += 1
        self.log.append(("stop", handle.deployment_id))
        if not self.clean_stop:
            return StopResult(clean=False, detail="pid 1: still_running")
        self.alive -= 1
        return StopResult(clean=True, detail="terminated")


def _variant(plan_id, remote=True, peer_bytes=GiB):
    plan = _plan(remote)
    if remote:
        peer = plan.assignments[1].model_copy(update={"total_bytes_est": peer_bytes})
        plan = plan.model_copy(update={"assignments": [plan.assignments[0], peer]})
    return plan.model_copy(update={"id": plan_id})


async def _manager(store, events, tmp_path, adapter):
    mgr, reg, leases = await _setup(store, events, adapter, tmp_path)
    inf = InferenceService(store, events, mgr, FakeModels(), adapter)
    mgr.drain = lambda dep, timeout_s: inf.acquire(dep, exclusive=True, timeout=timeout_s)
    adapter.leases = leases
    dep = await _wait_terminal_or_ready(mgr, (await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)).id)
    assert dep.state == DeploymentState.ready and dep.revision == 0
    return mgr, leases, inf, dep


def _lease_rows(store, dep_id):
    return [dict(r) for r in store.query("SELECT * FROM leases WHERE deployment_id=?", (dep_id,))]


def _outcomes(events):
    return [e.data.get("outcome") for e in events.since(0)
            if e.type in ("deployment.replaced", "deployment.replace_failed")]


@pytest.mark.anyio
async def test_replace_bumps_revision_and_hands_leases_over_after_the_old_engine_stops(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    new = await mgr.replace(old.id, _variant("plan2"), "two peers decode faster")
    assert new.state == DeploymentState.ready and new.plan.id == "plan2"
    assert new.revision == new.plan.revision == 1 and new.replaces == old.id
    assert mgr.get(old.id).state == DeploymentState.stopped and mgr.ready()[0].id == new.id
    # the old engine stopped while only its own leases were held; they were released after it was gone
    assert adapter.leased_at_stop == [[old.id]]
    assert all(r["released_at"] and r["revision"] == 0 for r in _lease_rows(store, old.id))
    fresh = _lease_rows(store, new.id)
    assert len(fresh) == 2 and all(r["revision"] == 1 and r["chain_id"] == old.id and r["released_at"] is None
                                   for r in fresh)
    assert adapter.log == [("start", old.id), ("stop", old.id), ("start", new.id)] and adapter.max_alive == 1
    with pytest.raises(LeaseConflict, match="stale"):  # a late cleanup of revision 0 cannot touch revision 1
        leases.release(old.id, revision=0)
    assert _outcomes(events) == ["activated"]
    steps = [(e.type, e.data["revision"]) for e in events.since(0) if e.type.startswith("deployment.")
             and e.type != "deployment.progress" and e.data.get("deployment_id") in (old.id, new.id)
             and e.data.get("revision") == 1]
    assert ("deployment.replacing", 1) in steps and ("deployment.drained", 1) in steps
    assert ("deployment.state", 1) in steps and ("deployment.replaced", 1) in steps
    progress = [e.data["revision"] for e in events.since(0)
                if e.type == "deployment.progress" and e.data["deployment_id"] == new.id]
    assert progress and set(progress) == {1}  # load progress names the revision too


@pytest.mark.anyio
async def test_a_plan_that_is_not_admitted_is_refused_before_the_drain(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    drained = []

    async def drain(dep, timeout_s):
        drained.append(dep.id)
        raise AssertionError("a plan that is not admitted must not drain the old deployment")

    mgr.drain = drain
    too_big = _variant("plan2", peer_bytes=9 * GiB)  # the peer's 8 GiB pool cannot take it even when emptied
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, too_big, "faster")
    assert e.value.code == "not_admitted" and "over budget" in str(e.value)
    assert drained == [] and adapter.stopped == 0 and mgr.ready()[0].id == old.id
    assert {r["deployment_id"] for r in leases.active()} == {old.id}
    refused = [e.data for e in events.since(0) if e.type == "deployment.replace_failed"]
    assert [(d["outcome"], d["revision"]) for d in refused] == [("refused", 1)]
    missing_engine = _variant("plan3").model_copy(update={"engine": "vllm"})
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, missing_engine, "faster")
    assert e.value.code == "engine_unavailable" and adapter.stopped == 0


@pytest.mark.anyio
async def test_the_admission_check_counts_the_old_leases_as_released(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    # old holds 1 GiB of the peer's 8 GiB pool; 7.5 GiB fits only once those are released, which the stop does first
    new = await mgr.replace(old.id, _variant("plan2", peer_bytes=int(7.5 * GiB)), "bigger share on the peer")
    assert new.state == DeploymentState.ready and new.revision == 1
    assert {r["deployment_id"] for r in leases.active()} == {new.id}


@pytest.mark.anyio
async def test_drain_is_awaited_before_the_old_engine_stops(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, _, _, old = await _manager(store, events, tmp_path, adapter)

    class Hold:
        def release(self):
            adapter.log.append(("release", old.id))

    async def drain(dep, timeout_s):
        adapter.log.append(("drain", dep.id))
        await asyncio.sleep(0.05)  # a request finishing
        return Hold()

    mgr.drain = drain
    new = await mgr.replace(old.id, _variant("plan2"), "manual")
    assert adapter.log == [("start", old.id), ("drain", old.id), ("stop", old.id), ("release", old.id),
                           ("start", new.id)]


@pytest.mark.anyio
async def test_drain_timeout_keeps_the_old_deployment_serving(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, leases, inf, old = await _manager(store, events, tmp_path, adapter)
    mgr.drain_timeout_s = 0.05
    chat = await inf.acquire(old)  # a request in flight that does not finish in time
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, _variant("plan2"), "manual")
    assert e.value.code == "drain_timeout"
    assert mgr.ready()[0].id == old.id and adapter.stopped == 0
    assert {r["deployment_id"] for r in leases.active()} == {old.id}
    chat.release()


@pytest.mark.anyio
async def test_failed_new_load_rolls_back_when_the_old_resources_still_admit(store, events, tmp_path):
    adapter = TrackingAdapter(fail_plans={"plan2"})
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    result = await mgr.replace(old.id, _variant("plan2"), "faster")
    assert result.state == DeploymentState.ready and result.plan.id == "plan1" and result.revision == 2
    failed = mgr.get(result.replaces)
    assert failed.state == DeploymentState.failed and failed.plan.id == "plan2"
    assert failed.revision == 1 and failed.replaces == old.id
    assert {(r["deployment_id"], r["revision"]) for r in leases.active()} == {(result.id, 2)}
    assert _outcomes(events) == ["rolled_back"] and adapter.max_alive == 1
    assert [kind for kind, _ in adapter.log] == ["start", "stop", "start", "stop", "start"]


def _ghost_on_peer(adapter, leases, gib):
    """While plan2 loads, another reservation takes most of the peer's pool, so plan1 no longer fits there."""
    def take(spec):
        if spec.plan.id == "plan2":
            leases.reserve("ghost", [LeaseRequest(PEER, "uma", int(gib * GiB), 8 * GiB, 8 * GiB)])
    adapter.on_prepare = take


@pytest.mark.anyio
async def test_rolls_to_the_fallback_when_the_previous_plan_is_not_admitted(store, events, tmp_path):
    adapter = TrackingAdapter(fail_plans={"plan2"})
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    _ghost_on_peer(adapter, leases, 7.25)  # plan2 needs 0.5 GiB there (fits), plan1 needs 1 GiB (does not)
    result = await mgr.replace(old.id, _variant("plan2", peer_bytes=GiB // 2), "faster",
                               fallback=_variant("plan3", remote=False))
    assert result.state == DeploymentState.ready and result.plan.id == "plan3" and result.revision == 2
    recovery = [e.data for e in events.since(0) if e.type == "deployment.recovery"]
    assert [(d["outcome"], d["admitted"]) for d in recovery] == [("rolled_back", False), ("fell_back", True)]
    assert _outcomes(events) == ["fell_back"] and adapter.max_alive == 1
    assert {r["deployment_id"] for r in leases.active()} == {result.id, "ghost"}


@pytest.mark.anyio
@pytest.mark.parametrize("fallback", [None, "fails"])
async def test_fails_with_the_reason_when_neither_rollback_nor_fallback_works(store, events, tmp_path, fallback):
    adapter = TrackingAdapter(fail_plans={"plan2", "plan3"})
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    _ghost_on_peer(adapter, leases, 7.25)
    alt = _variant("plan3", remote=False) if fallback else None
    result = await mgr.replace(old.id, _variant("plan2", peer_bytes=GiB // 2), "faster", fallback=alt)
    assert result.state == DeploymentState.failed and mgr.active() is None
    assert "previous plan not admitted" in result.error and "requested over budget" in result.error
    if fallback:
        assert result.plan.id == "plan3" and "fallback plan plan3 failed" in result.error
    else:
        assert result.plan.id == "plan2" and result.revision == 1
    assert {r["deployment_id"] for r in leases.active()} == {"ghost"}
    assert _outcomes(events) == ["failed"] and adapter.max_alive == 1


@pytest.mark.anyio
async def test_replace_is_refused_unless_ready_and_for_the_same_model(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, _, _, old = await _manager(store, events, tmp_path, adapter)
    other = _variant("plan9").model_copy(update={"model_id": "another-model"})
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, other, "manual")
    assert e.value.code == "model_mismatch"
    infeasible = _variant("plan8").model_copy(update={"feasible": False, "reasons": ["does not fit"]})
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, infeasible, "manual")
    assert e.value.code == "plan_not_feasible"
    await mgr.stop(old.id)
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, _variant("plan2"), "manual")
    assert e.value.code == "not_ready"
    # a deployment that is still loading is not ready either
    adapter.gate = asyncio.Event()
    loading = await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)
    for _ in range(500):
        if mgr.get(loading.id).state == DeploymentState.loading:
            break
        await asyncio.sleep(0.01)
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(loading.id, _variant("plan2"), "manual")
    assert e.value.code == "not_ready"
    adapter.gate.set()
    assert (await _wait_terminal_or_ready(mgr, loading.id)).state == DeploymentState.ready


@pytest.mark.anyio
async def test_create_is_refused_during_a_replacement_and_an_owner_stop_is_not_rolled_back(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, _, _, old = await _manager(store, events, tmp_path, adapter)
    adapter.gate = asyncio.Event()
    swap = asyncio.create_task(mgr.replace(old.id, _variant("plan2"), "manual"))
    for _ in range(500):
        if len(adapter.log) == 3:  # start old, stop old, start new
            break
        await asyncio.sleep(0.01)
    new_id = adapter.log[-1][1]
    with pytest.raises(DeploymentError) as e:
        await mgr.create(CreateDeploymentRequest(plan_id="plan1"), None)
    assert e.value.code == "deployment_replacing"
    await mgr.stop(new_id)
    result = await asyncio.wait_for(swap, 5)
    assert result.id == new_id and result.state == DeploymentState.stopped
    assert [kind for kind, _ in adapter.log] == ["start", "stop", "start", "stop"] and mgr.active() is None


@pytest.mark.anyio
async def test_a_deployment_stopped_while_it_drains_is_not_replaced(store, events, tmp_path):
    adapter = TrackingAdapter()
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    released = []

    class Hold:
        def release(self):
            released.append(True)

    async def drain(dep, timeout_s):
        await mgr.stop(dep.id)  # the owner clicks stop while the last request finishes
        return Hold()

    mgr.drain = drain
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, _variant("plan2"), "manual")
    assert e.value.code == "not_ready" and released == [True]
    assert [kind for kind, _ in adapter.log] == ["start", "stop"] and mgr.active() is None
    assert leases.active() == []


@pytest.mark.anyio
async def test_old_engine_that_does_not_stop_blocks_the_new_one(store, events, tmp_path):
    adapter = TrackingAdapter(clean_stop=False)
    mgr, leases, _, old = await _manager(store, events, tmp_path, adapter)
    with pytest.raises(DeploymentError) as e:
        await mgr.replace(old.id, _variant("plan2"), "manual")
    assert e.value.code == "replace_failed" and "its leases stay held" in str(e.value)
    assert mgr.get(old.id).state == DeploymentState.failed
    assert [kind for kind, _ in adapter.log] == ["start", "stop"]
    assert {r["deployment_id"] for r in leases.active()} == {old.id}  # its memory may still be in use


def test_stale_lease_release_and_reservation_are_rejected(store):
    leases = LeaseManager(store)
    req = [LeaseRequest("n", "ram", GiB, 4 * GiB, 4 * GiB)]
    leases.reserve("d1", req, revision=1)
    assert leases.release("d1", revision=1) == 1
    leases.reserve("d2", req, revision=2, chain="d1")
    with pytest.raises(LeaseConflict, match="stale release"):
        leases.release("d1", revision=1)
    with pytest.raises(LeaseConflict, match="stale release"):
        leases.release("d2", revision=1)
    with pytest.raises(LeaseConflict, match="stale reservation"):
        leases.reserve("d3", req, revision=1, chain="d1")
    assert [r["deployment_id"] for r in leases.active()] == ["d2"]
    assert leases.release("d2", revision=2) == 1 and leases.active() == []
    # a release without a revision keeps the old behavior (restart reconciliation)
    leases.reserve("d4", req)
    assert leases.release("d4") == 1


def test_admission_check_keeps_nothing(store):
    leases = LeaseManager(store)
    assert leases.admits([LeaseRequest("n", "ram", GiB, 4 * GiB, 4 * GiB)]) is None
    assert "over budget" in leases.admits([LeaseRequest("n", "ram", 5 * GiB, 4 * GiB, 8 * GiB)])
    assert leases.active() == []
    leases.reserve("old", [LeaseRequest("n", "ram", 3 * GiB, 4 * GiB, 4 * GiB)])
    want = [LeaseRequest("n", "ram", 2 * GiB, 4 * GiB, 4 * GiB)]
    assert "over budget" in leases.admits(want)
    assert leases.admits(want, releasing="old") is None  # as if old's leases were gone
    assert [r["deployment_id"] for r in leases.active()] == ["old"]  # ...but they are still held


def test_an_existing_database_gains_the_lease_revision_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE leases (id TEXT PRIMARY KEY, deployment_id TEXT NOT NULL, node_id TEXT NOT NULL, "
                 "pool_id TEXT NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL, released_at TEXT)")
    conn.execute("INSERT INTO leases VALUES ('l1', 'old', 'n', 'ram', 1, '2026-10-01T00:00:00+00:00', NULL)")
    conn.commit()
    conn.close()
    for _ in range(2):  # opening twice must not add the columns twice
        store = Store(path, tmp_path / "records")
        try:
            columns = {r["name"] for r in store.query("PRAGMA table_info(leases)")}
            assert {"revision", "chain_id"} <= columns
            row = LeaseManager(store).active()
            assert [(r["id"], r["revision"], r["chain_id"]) for r in row] == [("l1", 0, None)]
        finally:
            store.close()
    store = Store(path, tmp_path / "records")
    try:
        assert LeaseManager(store).release("old", revision=0) == 1
    finally:
        store.close()
