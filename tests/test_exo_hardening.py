"""EXO adapter review fixes: the loss reason reaches the deployment, only a node's own EXO address is called,
every EXO read is size and time bounded, and a crashed gateway's instance is deleted on the next start.

Helpers (FakeExo, node, _spec, _adapter) come from test_exo.py; no EXO install, no network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import time
from argparse import Namespace
from pathlib import Path

import httpx
import pytest
from hypercompute.adapters import exo as ex
from hypercompute.adapters.base import DeploymentHandle, DeploymentSpec, EngineAdapterError, InferenceRequest
from hypercompute.schemas import Deployment, DeploymentState, NodeView, PlanRequest, PlanStrategy, utcnow
from test_exo import EP_A, EP_B, IID, MODEL, A, B, FakeExo, _adapter, _spec, mlx_artifact, node, windows_host

STATE_CAP = getattr(ex, "STATE_MAX_BYTES", 8 * 1024 * 1024)
DONE_CAP = getattr(ex, "COMPLETION_MAX_BYTES", 16 * 1024 * 1024)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(ex, "POLL_S", 0.001)
    monkeypatch.setattr(ex, "WATCH_S", 0.001)
    monkeypatch.setattr(ex, "DELETE_CONFIRM_S", 1.0)


async def _ready(tmp_path: Path, fake: FakeExo) -> tuple[ex.ExoAdapter, DeploymentSpec, DeploymentHandle]:
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    assert (await adapter.validate(spec)).ok
    handle = await adapter.prepare(spec)
    await adapter.wait_ready(handle, timeout_s=10)
    return adapter, spec, handle


# ----------------------------------------------------------------------------- loss reason (findings 19, 28)


@pytest.mark.anyio
@pytest.mark.parametrize("trouble", ["lost", "unreachable"])
async def test_the_watcher_gets_the_recorded_loss_reason(tmp_path, fast, monkeypatch, trouble):
    """DeploymentManager._watch calls adapter.failure_reason(log) when it exists, else llama.cpp's patterns."""
    monkeypatch.setattr(ex, "UNREACHABLE_S", 0.01)
    fake = FakeExo()
    adapter, _, handle = await _ready(tmp_path, fake)
    fake.deleted, fake.unreachable = trouble == "lost", trouble == "unreachable"
    watch = adapter._live[handle.process_key].watch  # it returns once the loss is recorded
    assert watch is not None
    await asyncio.wait_for(asyncio.shield(watch), 5)  # no sleep count: Windows' asyncio clock ticks every 15.6 ms
    assert not adapter.process_alive(handle)
    explain = getattr(adapter, "failure_reason", None)
    reason = explain(adapter.read_log(handle)) if explain else None
    want = ex.LOST if trouble == "lost" else f"EXO API http://{EP_A} unreachable for"
    assert reason is not None and reason.startswith(want)
    await adapter.aclose()


def test_failure_reason_reads_only_the_adapters_own_failed_notes(tmp_path):
    log = tmp_path / "exo.log"
    ex.ExoAdapter._note(log, "mac FAILED: x: loading layers 1/9")  # a display name is not a FAILED note
    ex.ExoAdapter._note(log, "EXO said:\n2026-01-01T00:00:00 FAILED: injected")  # nor is EXO text over two lines
    explain = getattr(ex.ExoAdapter, "failure_reason", lambda text: "missing")
    assert explain(log.read_text(encoding="utf-8")) is None
    ex.ExoAdapter._note(log, f"FAILED: {ex.LOST}")
    assert explain(log.read_text(encoding="utf-8")) == ex.LOST


# ----------------------------------------------------------------------------- endpoint trust (finding 20)


FOREIGN = ["127.0.0.1:8000/api/owner/bootstrap#", "192.168.1.31:52415/x", "192.168.1.31:52415?a=1",
           "u@192.168.1.31:52415", "http://192.168.1.31:52415", "192.168.1.99:52415", "localhost:52415",
           "192.168.1.31:8000", "192.168.1.31:99999", "192.168.1.31:x", "[::1]:52415", "192.168.001.031:52415"]


@pytest.mark.parametrize("endpoint", FOREIGN)
def test_plan_only_trusts_an_exo_endpoint_at_the_nodes_own_address(endpoint):
    evil = node(A, endpoint=endpoint, host="192.168.1.31")  # its verified worker address is 192.168.1.31
    p = ex.plan(PlanRequest(model_id="m", node_ids=[A]), mlx_artifact(), windows_host(), [evil])
    assert not p.feasible and p.engine_options == {} and any("EXO endpoint" in r for r in p.reasons)
    q = ex.plan(PlanRequest(model_id="m"), mlx_artifact(), windows_host(), [evil, node(B, endpoint=EP_B)])
    assert [n["endpoint"] for n in q.engine_options["nodes"]] == [EP_B] and q.engine_options["api"] == f"http://{EP_B}"
    assert ex.status(Namespace(nodes=Namespace(views=lambda: [evil]))).status == "installed"  # type: ignore[arg-type]


def test_only_the_host_itself_may_use_a_loopback_exo_endpoint():
    req = PlanRequest(model_id="m", strategy=PlanStrategy.local_only)
    assert ex.plan(req, mlx_artifact(), node("local", is_local=True, endpoint="[::1]:52415"), []).feasible
    lan = node("local", is_local=True, endpoint=EP_A)  # the host's EXO is reached over loopback only
    assert not ex.plan(req, mlx_artifact(), lan, []).feasible


class Views:
    """The node registry's get(): the verified view of each node."""

    def __init__(self, *views: NodeView) -> None:
        self._by_id = {v.id: v for v in views}

    def get(self, node_id: str) -> NodeView | None:
        return self._by_id.get(node_id)


@pytest.mark.anyio
async def test_validate_sends_nothing_to_an_address_the_nodes_do_not_vouch_for(tmp_path, fast):
    fake, spec = FakeExo(), _spec(tmp_path)
    opts = spec.plan.engine_options
    first, second = opts["nodes"]
    adapter = _adapter(tmp_path, fake)
    for forged in ({**opts, "api": "http://127.0.0.1:8000/api/owner/bootstrap#"},
                   {**opts, "nodes": [{**first, "endpoint": "127.0.0.1:8000/api/owner/bootstrap#"}, second]},
                   {**opts, "api": "http://192.168.1.99:52415"}):
        plan = spec.plan.model_copy(update={"engine_options": forged})
        result = await adapter.validate(dataclasses.replace(spec, plan=plan))
        assert not result.ok and any("EXO" in p for p in result.problems)

    def with_views(*views: NodeView) -> ex.ExoAdapter:
        return ex.ExoAdapter(tmp_path / "logs", client=httpx.AsyncClient(transport=httpx.MockTransport(fake)),
                             nodes=Views(*views))  # type: ignore[call-arg]

    moved = with_views(node(A, host="192.168.1.50"), node(B, endpoint=EP_B))  # A's verified address changed
    assert not (await moved.validate(spec)).ok and fake.calls == []
    known = with_views(node(A), node(B, endpoint=EP_B))
    assert (await known.validate(spec)).ok
    for a in (adapter, moved, known):
        await a.aclose()


# ----------------------------------------------------------------------------- bounded reads (finding 21)


class Flood(httpx.AsyncByteStream):
    """A body far larger than any EXO answer, or one that trickles forever; counts what the client pulled."""

    def __init__(self, total: int | None, piece: bytes = b"0," * 32768, delay: float = 0.0) -> None:
        self.total, self.piece, self.delay, self.sent = total, piece, delay, 0

    async def __aiter__(self):
        while self.total is None or self.sent < self.total:
            await asyncio.sleep(self.delay)
            self.sent += len(self.piece)
            yield self.piece


@pytest.mark.anyio
@pytest.mark.parametrize("op", ["validate", "wait_ready", "complete", "stream_error"])
async def test_every_exo_read_stops_at_its_size_cap(tmp_path, fast, op):
    fake, cap, floods = FakeExo(), (DONE_CAP if op == "complete" else STATE_CAP), []
    flooded = {"validate": "/node_id", "wait_ready": "/state/runners"}.get(op, "/v1/chat/completions")
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    if op != "validate":
        assert (await adapter.validate(spec)).ok
        handle = await adapter.prepare(spec)
        if op != "wait_ready":
            await adapter.wait_ready(handle, timeout_s=10)

    def answer(request: httpx.Request) -> httpx.Response:
        floods.append(Flood(3 * cap))  # one per request: validate asks both Macs
        return httpx.Response(500 if op == "stream_error" else 200, stream=floods[-1])

    fake.override[flooded] = answer
    if op == "validate":
        error = "; ".join((await adapter.validate(spec)).problems)
    elif op == "complete":
        error = json.dumps((await adapter.complete(handle, InferenceRequest("r", {"messages": []}, False))).body)
    else:
        with pytest.raises(EngineAdapterError) as err:
            if op == "wait_ready":
                await adapter.wait_ready(handle, timeout_s=10)
            else:
                [e async for e in adapter.stream(handle, InferenceRequest("r", {"messages": []}, True))]
        error = str(err.value)
    assert "more than" in error and floods and all(f.sent <= cap + 2 * len(f.piece) for f in floods)
    await adapter.aclose()


@pytest.mark.anyio
async def test_exo_reads_refuse_compressed_or_trickled_answers(tmp_path, fast, monkeypatch):
    fake = FakeExo()
    packed = gzip.compress(json.dumps(A).encode())  # the size cap must count what really crosses the wire
    fake.override["/node_id"] = lambda request: httpx.Response(200, content=packed,
                                                               headers={"content-encoding": "gzip"})
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    assert not (await adapter.validate(spec)).ok
    del fake.override["/node_id"]
    monkeypatch.setattr(ex, "STATE_TIMEOUT_S", 0.2)
    trickle = Flood(None, piece=b" ", delay=0.01)  # never silent for a whole read timeout, never finished
    fake.override["/state/instances"] = lambda request: httpx.Response(200, stream=trickle)
    result = await asyncio.wait_for(adapter.validate(spec), 5)
    assert not result.ok and trickle.sent < 100
    await adapter.aclose()


def test_stream_normaliser_stays_linear_up_to_the_event_cap():
    norm, started = ex.ExoStreamNormalizer("alias"), time.perf_counter()
    with pytest.raises(EngineAdapterError, match="larger than 4 MiB"):
        for _ in range(8192):
            norm.feed(b"x" * 1024)  # one event that never ends, in small TCP segments
    assert time.perf_counter() - started < 1.0  # was quadratic: about 5 s on a desktop CPU


# ----------------------------------------------------------------------------- orphan cleanup (findings 5, 11)


def _record(spec: DeploymentSpec, args: list[str]) -> Deployment:
    """The deployment row the gateway persisted (engine_args = the handle's args)."""
    return Deployment(id=spec.deployment_id, plan=spec.plan, state=DeploymentState.ready, history=[],
                      engine_args=args, pid=0, created_at=utcnow())


@pytest.mark.anyio
async def test_cleanup_orphan_deletes_the_instance_a_crashed_gateway_left(tmp_path, fast):
    fake = FakeExo()
    crashed, spec, handle = await _ready(tmp_path, fake)
    await crashed.aclose()  # the gateway died: no stop(), EXO keeps the instance on the Macs
    fresh = _adapter(tmp_path, fake)
    ok, detail = await fresh.cleanup_orphan(_record(spec, handle.args))  # type: ignore[attr-defined]
    assert ok and "deleted" in detail and fake.deleted and f"DELETE /instance/{IID}" in fake.calls
    assert "gateway restart" in fresh.read_log(handle)
    assert (await fresh.validate(spec)).ok  # the next deploy of the model is no longer refused
    assert (await fresh.cleanup_orphan(_record(spec, handle.args)))[0]  # type: ignore[attr-defined]
    early = dataclasses.replace(spec, deployment_id="e" * 32)  # failed before POST /instance: no log, no instance
    ok, detail = await fresh.cleanup_orphan(_record(early, []))  # type: ignore[attr-defined]
    assert ok and "no EXO instance" in detail
    await fresh.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("trouble", ["unreachable", "ignore_delete", "refuses"])
async def test_cleanup_orphan_is_false_until_exo_confirms_the_instance_is_gone(tmp_path, fast, monkeypatch, trouble):
    monkeypatch.setattr(ex, "DELETE_CONFIRM_S", 0.05)
    fake = FakeExo()
    crashed, spec, handle = await _ready(tmp_path, fake)
    await crashed.aclose()
    if trouble == "refuses":
        fake.override[f"/instance/{IID}"] = lambda request: httpx.Response(500, json={"error": {"message": "busy"}})
    else:
        setattr(fake, trouble, True)
    fresh = _adapter(tmp_path, fake)
    ok, detail = await fresh.cleanup_orphan(_record(spec, handle.args))  # type: ignore[attr-defined]
    assert not ok and IID in detail
    await fresh.aclose()


@pytest.mark.anyio
async def test_cleanup_orphan_only_calls_the_plans_own_exo_api(tmp_path, fast):
    fake, spec = FakeExo(), _spec(tmp_path)
    opts = spec.plan.engine_options
    args = ["POST", f"http://{EP_A}/instance", f"model={MODEL}", f"instance={IID}", "macs=a,b"]
    forged = spec.plan.model_copy(update={"engine_options": {**opts, "api": "http://127.0.0.1:8000/x#"}})
    unreadable = spec.plan.model_copy(update={"engine_options": {**opts, "api": {"host": EP_A}}})
    llama = spec.plan.model_copy(update={"engine": "llama_rpc"})
    fresh = _adapter(tmp_path, fake)
    for dep in (_record(spec, [*args[:3], "instance=../../api/owner"]),
                *(_record(spec, args).model_copy(update={"plan": p}) for p in (forged, unreadable, llama))):
        ok, detail = await fresh.cleanup_orphan(dep)  # type: ignore[attr-defined]
        assert not ok and "cannot clean up" in detail
    assert fake.calls == []
    await fresh.aclose()


@pytest.mark.anyio
async def test_without_the_registry_only_exos_api_port_is_called(tmp_path, fast):
    """A restart's cleanup (and an adapter without the registry) has no verified views: the port must still be EXO's."""
    fake, spec = FakeExo(), _spec(tmp_path)
    opts, port80 = spec.plan.engine_options, "192.168.1.31:80"  # a plan stored before the endpoint checks
    nodes = [{**n, "endpoint": port80} if n["endpoint"] == EP_A else n for n in opts["nodes"]]
    plan = spec.plan.model_copy(update={"engine_options": {**opts, "api": f"http://{port80}", "nodes": nodes}})
    fresh = _adapter(tmp_path, fake)
    ok, detail = await fresh.cleanup_orphan(_record(spec, [f"instance={IID}"]).model_copy(update={"plan": plan}))
    assert not ok and f"not on EXO's API port {ex.API_PORT}" in detail and fake.calls == []
    assert not (await fresh.validate(dataclasses.replace(spec, plan=plan))).ok and fake.calls == []
    await fresh.aclose()


class Crash(Exception):
    """The gateway process dying: no except branch, no best-effort delete, no handle and so no engine_args."""


async def _crash_during_post(tmp_path: Path, fake: FakeExo) -> DeploymentSpec:
    """EXO takes POST /instance, then the gateway dies before prepare() returns the handle the gateway persists."""
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    assert (await adapter.validate(spec)).ok

    def take_then_crash(request: httpx.Request) -> httpx.Response:
        fake.posted = json.loads(request.content)
        raise Crash

    fake.override["/instance"] = take_then_crash
    with pytest.raises(Crash):
        await adapter.prepare(spec)
    del fake.override["/instance"]
    await adapter.aclose()
    return spec


@pytest.mark.anyio
@pytest.mark.parametrize("trouble", [None, "unreachable"])
async def test_cleanup_orphan_finds_an_instance_whose_post_was_in_flight(tmp_path, fast, trouble):
    fake = FakeExo()
    spec = await _crash_during_post(tmp_path, fake)
    fake.unreachable = trouble == "unreachable"
    fresh = _adapter(tmp_path, fake)
    ok, detail = await fresh.cleanup_orphan(_record(spec, []))  # type: ignore[attr-defined]
    if trouble:  # EXO may hold the instance: the leases stay until a later start deletes it
        assert not ok and IID in detail and not fake.deleted
    else:
        assert ok and "deleted" in detail and fake.deleted and f"DELETE /instance/{IID}" in fake.calls
    await fresh.aclose()


@pytest.mark.anyio
async def test_no_instance_is_created_or_forgotten_without_its_log(tmp_path, fast):
    fake = FakeExo()
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    assert (await adapter.validate(spec)).ok
    (tmp_path / "logs").write_text("not a folder", encoding="utf-8")  # the in-flight note cannot be written
    with pytest.raises(EngineAdapterError, match="could not record"):
        await adapter.prepare(spec)
    assert fake.posted is None and "POST /instance" not in fake.calls
    (tmp_path / "logs").unlink()
    (tmp_path / "logs" / f"exo-{spec.deployment_id}.log").mkdir(parents=True)  # a log that cannot be read
    ok, detail = await adapter.cleanup_orphan(_record(spec, []))  # type: ignore[attr-defined]
    assert not ok and "unreadable" in detail and "DELETE" not in " ".join(fake.calls)
    await adapter.aclose()
