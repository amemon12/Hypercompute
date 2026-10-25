"""Capacity and replan surface: the HTTP routes (fake CapacityService and ReplanController on the app tests'
coordinator), the lifespan's path lookup, and `hypercompute capacity` against the same app or a fake transport.

The fake service returns fixed dicts in the shape capacity/service.py documents. Its measured entries carry tonight's
numbers with their evidence (relay link probe, storage probe); every other number in the fixtures is a prior or a test
value and is labelled as one.
"""

import asyncio
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import hypercompute
import pytest
from fastapi.testclient import TestClient
from hypercompute import capacity_cli, cli
from hypercompute.app import CAPACITY_STATUS, create_app, iroh_path_mode, lan_path, read_on_loop
from hypercompute.capacity.service import CapacityUnavailable
from hypercompute.schemas import Deployment, DeploymentState, ReplanDecision, StateTransition, utcnow
from hypercompute.services.deployments import DeploymentError
from hypercompute.storage import dumps
from hypercompute.units import GiB, MiB, fmt_bytes
from test_app_http import OWNER, make_coord, open_client
from test_deployments import _plan

OWNER_HEADERS = {"X-Hypercompute-Owner": OWNER}
MODEL = "qwen2.5-7b-instruct-q4_k_m"
PEER = "a" * 32
OLD_DEP = "d" * 32
LINK_EVIDENCE = "results/capacity/20261022T040153Z-link-relay-127.0.0.1-52110/link.json"

POOLS = {
    "observed_at": "2026-10-22T04:10:00+00:00",
    "host_memory": {"total": 32 * GiB, "available": 20 * GiB, "commit_charge": 18 * GiB, "commit_limit": 40 * GiB,
                    "system_cache": 6 * GiB, "observed_at": "2026-10-22T04:10:00+00:00",
                    "source": "GetPerformanceInfo", "error": None},
    "pools": [
        {"node_id": "local", "pool_id": "ram", "kind": "system_ram", "total_bytes": 32 * GiB, "free_bytes": 20 * GiB,
         "reserve_bytes": 6 * GiB, "cap_bytes": None, "admissible_bytes": 14 * GiB, "devices": ["CPU"],
         "source": "inventory", "age_s": 3.2},
        {"node_id": PEER, "pool_id": "uma", "kind": "unified", "total_bytes": 16 * GiB, "free_bytes": 9 * GiB,
         "reserve_bytes": 4 * GiB, "cap_bytes": 8 * GiB, "admissible_bytes": 5 * GiB, "devices": ["MTL0"],
         "source": "heartbeat", "age_s": 7.0},
    ],
    "excluded": [{"node_id": "local", "pool_id": "vram:CUDA1", "reason": "unknown free memory"}],
    "notes": ["test note: no single pool carries this"],
    "leases": [],
}

EXPLAIN_LINES = [
    "best: J 41234 ms = load 9000 + prefill 1385 + 256 x decode 120.5 ms + risk 0; 0 B streamed per token",
    "stage 1 local:CPU (local) units 0-7: resident 1.5 GiB; demand 2.1 GiB of 14 GiB admissible; compute 18.3 ms "
    "(prior)",
    "stage 2 local:CUDA0 (local) units 8-28: resident 2.9 GiB; demand 3.4 GiB of 7.5 GiB admissible; compute 7.6 ms "
    "(prior)",
    "bottleneck: local:CPU compute 18.3 ms of 120.5 ms per token (15 percent)",
    f"skipped {PEER}:MTL0: not used: adds 76.1 ms hop per token, saves 6.0 ms compute and I/O per token",
    "assumption: local:CPU decode 12.5 ms/GiB (prior: planner figure, converted to ms/GiB)",
]
ASSUMPTIONS = [
    "local:CPU decode 12.5 ms/GiB (prior: planner figure, converted to ms/GiB)",
    "local:CUDA0 decode 2.64 ms/GiB (prior: planner figure, converted to ms/GiB)",
    "setup 4000 ms (estimate: engine start on this host, test value)",
]
MEASURED = [
    f"{PEER}:MTL0 rtt 36.9 ms (measured: relay median at RPC 127.0.0.1:52110, p90 62.9 ms, {LINK_EVIDENCE})",
    {"name": f"{PEER}:MTL0 goodput", "value": 1.59e6, "unit": "bytes/s", "evidence": LINK_EVIDENCE},
    "local storage read 3.45 GiB/s (measured: unbuffered QD1, storage probe)",
]


def _stage(node, device, units, path="local", streamed=()):
    """A StagePrediction dump (bridge.candidate_json)."""
    return {"node_id": node, "device": device, "units": list(units), "streamed_units": list(streamed),
            "resident_units": [u for u in units if u not in streamed], "compute_ms": 0.0, "exposed_io_ms": 0.0,
            "hop_ms": 0.0, "upload_bytes": 0, "path": path}


def _candidate(layout, j_ms, decode_ms, stages, feasible=True):
    """The shape of bridge.candidate_json; the times are test values, not predictions of this hardware."""
    return {"layout": layout, "feasible": feasible, "decode_ms": decode_ms, "prefill_ms": 1385.0, "load_ms": 9000.0,
            "objective_ms": j_ms, "bottleneck": "test", "stream_bytes_per_token": 0, "stages": stages, "skipped": [],
            "reasons": [], "assumptions": []}


EXPLAIN = {
    "model_id": MODEL,
    "workload": {"ctx_size": 4096, "prompt_tokens": 512, "output_tokens": 256, "warm": False,
                 "horizon_tokens": 2048, "objective": "interactive"},
    "feasible": True,
    "best": {"decode_ms": 120.5, "prefill_ms": 1385.0, "load_ms": 9000.0, "objective_ms": 41234.0,
             "bottleneck": "local:CPU compute", "skipped": [EXPLAIN_LINES[4].removeprefix("skipped ")],
             "assumptions": ASSUMPTIONS},
    "plan": None,
    "candidates": [
        _candidate("local:CPU [0-7], local:CUDA0 [8-27 + head]", 41234.0, 120.5,
                   [_stage("local", "CPU", range(8)), _stage("local", "CUDA0", range(8, 29))]),
        _candidate(f"local:CPU [0-3], {PEER}:MTL0 [4-11], local:CUDA0 [12-27 + head]", 52000.0, 160.2,
                   [_stage("local", "CPU", range(4), streamed=(2, 3)), _stage(PEER, "MTL0", range(4, 12), "relay"),
                    _stage("local", "CUDA0", range(12, 29))]),
        _candidate("local:CPU [0-27 + head]", 90000.0, 700.0, [_stage("local", "CPU", range(29))], feasible=False),
    ],
    "explain": EXPLAIN_LINES,
    "rejected": [],
    "assumptions": ASSUMPTIONS,
    "measured": MEASURED,
    "warnings": [f"{'b' * 32}:CUDA0: its RPC worker is not verified"],
}

DECISION = ReplanDecision(at=utcnow(), trigger="manual", deployment_id=OLD_DEP, revision=3, action="keep",
                          reason="gain below hysteresis", current_decode_ms=120.0, proposed_decode_ms=110.0,
                          relative_gain=1 / 12, reload_ms=9000.0, replay_ms=1400.0, break_even_tokens=1040.0,
                          proposed_plan_id="plan2")


class FakeCapacity:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.path_of = None  # the lifespan sets it

    def pools(self):
        return POOLS

    def explain(self, model_id, *, ctx_size=4096, prompt_tokens=512, output_tokens=256, warm=False, node_ids=None,
                slots=2):
        self.calls.append(dict(model_id=model_id, ctx_size=ctx_size, prompt_tokens=prompt_tokens,
                               output_tokens=output_tokens, warm=warm, node_ids=node_ids, slots=slots))
        if self.error is not None:
            raise self.error
        return EXPLAIN


class FakeReplan:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.triggered = 0
        self.decision = DECISION  # what trigger() answers

    def decisions(self):
        return [DECISION]

    async def trigger(self):
        self.triggered += 1
        return self.decision

    async def run(self):
        return None

    async def aclose(self):
        return None


class FakeReplace:
    """Stands in for DeploymentManager.replace: records the call, returns a result or raises an error.

    recovered=True answers as replace() does after the new plan failed and the previous plan came back: a ready
    deployment of the previous plan that replaces the failed attempt, not the deployment the owner named."""

    def __init__(self, state=DeploymentState.ready, error=None, recovered=False):
        self.calls, self.state, self.error, self.recovered = [], state, error, recovered

    async def __call__(self, dep_id, plan, reason, fallback=None):
        self.calls.append((dep_id, plan, reason, fallback))
        if self.error is not None:
            raise self.error
        if self.recovered:
            previous = plan.model_copy(update={"id": "oldplan", "revision": 5})
            detail = "rolled back after revision 4 failed: load_timeout"
            return Deployment(id="r" * 32, plan=previous, state=DeploymentState.ready, created_at=utcnow(),
                              history=[StateTransition(state=DeploymentState.planned, at=utcnow(), detail=detail)],
                              revision=5, replaces="f" * 32)
        new = plan.model_copy(update={"revision": 4})
        return Deployment(id="e" * 32, plan=new, state=self.state, history=[], created_at=utcnow(), revision=4,
                          replaces=dep_id, error=None if self.state == DeploymentState.ready else "load_timeout")


@pytest.fixture
def surface(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    coord.capacity, coord.replan = FakeCapacity(), FakeReplan()
    with open_client(coord) as client:
        yield SimpleNamespace(coord=coord, client=client, store=store, monkeypatch=monkeypatch)


def _store_plan(store, plan):
    store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?)",
                  (plan.id, plan.created_at.isoformat(), dumps(plan.model_dump(mode="json"))))
    return plan


def _args(*argv):
    return cli.build_parser().parse_args(["capacity", *argv])


# --------------------------------------------------------------------------- HTTP: capacity


def test_pools_route_returns_the_service_report_to_the_owner_only(surface):
    r = surface.client.get("/api/capacity/pools")
    assert r.status_code == 200 and r.json() == POOLS
    rebound = surface.client.get("/api/capacity/pools", headers={"Host": "evil.example"})
    assert rebound.status_code == 403 and rebound.json()["code"] == "owner_only"


def test_explain_passes_the_workload_through(surface):
    r = surface.client.get("/api/capacity/explain", params={"model_id": MODEL, "ctx_size": 8192, "prompt_tokens": 1024,
                                                            "output_tokens": 128, "warm": "true", "slots": 3})
    assert r.status_code == 200 and r.json() == EXPLAIN
    assert surface.coord.capacity.calls == [dict(model_id=MODEL, ctx_size=8192, prompt_tokens=1024, output_tokens=128,
                                                 warm=True, node_ids=None, slots=3)]
    assert surface.client.get("/api/capacity/explain", params={"model_id": MODEL}).status_code == 200
    assert surface.coord.capacity.calls[-1] == dict(model_id=MODEL, ctx_size=4096, prompt_tokens=512,
                                                    output_tokens=256, warm=False, node_ids=None, slots=2)


@pytest.mark.parametrize("params", [
    {}, {"model_id": "../configs/models"}, {"model_id": "qwen 7b"}, {"model_id": "x" * 129},
    {"model_id": MODEL, "ctx_size": 128}, {"model_id": MODEL, "ctx_size": 262145},
    {"model_id": MODEL, "prompt_tokens": -1}, {"model_id": MODEL, "output_tokens": 0},
    {"model_id": MODEL, "slots": 0}, {"model_id": MODEL, "slots": 9}, {"model_id": MODEL, "warm": "maybe"},
    {"model_id": MODEL, "ctx_size": "4k"},
])
def test_explain_rejects_bad_inputs_before_the_service(surface, params):
    assert surface.client.get("/api/capacity/explain", params=params).status_code == 422
    assert surface.coord.capacity.calls == []


def test_explain_rejects_a_workload_longer_than_its_context(surface):
    r = surface.client.get("/api/capacity/explain", params={"model_id": MODEL, "ctx_size": 1024,
                                                            "prompt_tokens": 1000, "output_tokens": 25})
    assert r.status_code == 422 and r.json()["code"] == "workload_exceeds_context" and r.json()["fix"]
    assert surface.coord.capacity.calls == []


def test_capacity_unavailable_is_422_with_its_code(surface):
    surface.coord.capacity.error = CapacityUnavailable("not_downloaded", f"{MODEL} is not downloaded")
    r = surface.client.get("/api/capacity/explain", params={"model_id": MODEL})
    assert r.status_code == 422
    assert r.json() == {"detail": f"{MODEL} is not downloaded", "code": "not_downloaded"}


SERVER_STATE_CODES = ("io", "pools", "host_memory", "model_files", "bridge")
REQUEST_CODES = ("not_downloaded", "not_gguf", "unknown_model", "bad_request", "participation", "workload")


@pytest.mark.parametrize("code, status", [*((c, 503) for c in SERVER_STATE_CODES), ("host_ineligible", 409),
                                          *((c, 422) for c in REQUEST_CODES)])
def test_capacity_unavailable_status_says_whether_the_host_or_the_request_is_at_fault(surface, code, status):
    surface.coord.capacity.error = CapacityUnavailable(code, f"test refusal ({code})")
    r = surface.client.get("/api/capacity/explain", params={"model_id": MODEL})
    assert r.status_code == status
    assert r.json() == {"detail": f"test refusal ({code})", "code": code}
    assert CAPACITY_STATUS[code] == status


def test_pools_route_reports_unreadable_host_memory_as_503(surface):
    def pools():
        raise CapacityUnavailable("host_memory", "this machine's RAM pool has no free-memory observation")

    surface.monkeypatch.setattr(surface.coord.capacity, "pools", pools)
    r = surface.client.get("/api/capacity/pools")
    assert r.status_code == 503 and r.json()["code"] == "host_memory"


def test_an_unclassified_capacity_code_keeps_422_and_is_logged(surface, caplog):
    surface.coord.capacity.error = CapacityUnavailable("brand_new_code", "a code nobody classified")
    with caplog.at_level("WARNING", logger="hypercompute.app"):
        r = surface.client.get("/api/capacity/explain", params={"model_id": MODEL})
    assert r.status_code == 422 and r.json()["code"] == "brand_new_code"
    assert any("brand_new_code" in rec.getMessage() and "CAPACITY_STATUS" in rec.getMessage()
               for rec in caplog.records)


def test_a_request_code_over_an_unreadable_file_is_503(surface, tmp_path):
    """A shard the gateway cannot read reaches the route as "manifest" (ManifestError over an OSError): host state."""
    from hypercompute.capacity.service import CapacityService

    with pytest.raises(CapacityUnavailable) as raised:  # the real service and manifest reader, shard gone
        CapacityService(None, None, None, None)._manifest(SimpleNamespace(id=MODEL, files=[], revision=None),
                                                          [tmp_path / "gone.gguf"])
    for error, status in ((raised.value, 503), (CapacityUnavailable("manifest", "the GGUF has no output head"), 422)):
        surface.coord.capacity.error = error
        r = surface.client.get("/api/capacity/explain", params={"model_id": MODEL})
        assert (r.status_code, r.json()["code"]) == (status, "manifest")


_RAISED_CODE = re.compile(r"(?:CapacityUnavailable|BridgeError)\(\s*\"([a-z_]+)\"")


def test_every_code_the_capacity_layer_raises_has_a_status():
    """A code added to capacity/service.py or capacity/bridge.py must be classified in app.CAPACITY_STATUS."""
    root = Path(hypercompute.__file__).parent
    raised = {code for p in root.rglob("*.py") for code in _RAISED_CODE.findall(p.read_text(encoding="utf-8"))}
    assert {"io", "host_memory", "host_ineligible", "not_downloaded"} <= raised  # the scan finds the raise sites
    assert sorted(raised - CAPACITY_STATUS.keys()) == []
    assert set(CAPACITY_STATUS.values()) <= {409, 422, 500, 503}


def test_real_capacity_service_behind_the_routes_and_the_cli(store, events, tmp_path, monkeypatch, capsys):
    """The real CapacityService on the app tests' coordinator: its JSON reaches the CLI's tables unchanged in shape.
    Host memory and disk are fixed test values so the run does not depend on this machine."""
    from hypercompute.capacity.bridge import Probes
    from hypercompute.capacity.measurements import MeasurementStore
    from hypercompute.capacity.pools import HostMemory
    from hypercompute.capacity.service import CapacityService

    memory = HostMemory(total=32 * GiB, available=20 * GiB, commit_charge=None, commit_limit=None, system_cache=None,
                        observed_at=utcnow(), source="test")
    coord = make_coord(store, events, tmp_path, monkeypatch)
    coord.capacity = CapacityService(coord.nodes, coord.models, coord.leases, MeasurementStore(tmp_path / "measure"),
                                     probes=Probes(memory=lambda: memory, disk_free=lambda _: 100 * GiB),
                                     models_dir=tmp_path / "models")
    with open_client(coord) as client:
        report = client.get("/api/capacity/pools").json()
        assert report["pools"] and all("admissible_bytes" in p for p in report["pools"])
        assert capacity_cli.cmd_capacity(_args("pools"), client=client) == 0
        out = capsys.readouterr().out
        assert f"host memory: {fmt_bytes(20 * GiB)} available of {fmt_bytes(32 * GiB)}" in out
        assert all(f"  local    {p['pool_id']}" in out for p in report["pools"] if p["node_id"] == "local")
        r = client.get("/api/capacity/explain", params={"model_id": "no-such-model"})
        assert r.status_code == 422 and r.json()["code"] == "unknown_model"


def test_capacity_routes_without_a_service_are_503(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)  # capacity None: the lifespan must still start
    with open_client(coord) as client:
        for path in ("/api/capacity/pools", f"/api/capacity/explain?model_id={MODEL}"):
            r = client.get(path)
            assert r.status_code == 503 and r.json()["code"] == "capacity_unavailable"


# --------------------------------------------------------------------------- HTTP: replan


def test_replan_decisions_and_an_owner_trigger(surface):
    r = surface.client.get("/api/replan/decisions")
    assert r.status_code == 200 and r.json() == [DECISION.model_dump(mode="json")]
    assert surface.client.post("/api/replan/trigger").status_code == 403
    assert surface.client.post("/api/replan/trigger", headers={"X-Hypercompute-Owner": "wrong"}).status_code == 403
    assert surface.coord.replan.triggered == 0
    r = surface.client.post("/api/replan/trigger", headers=OWNER_HEADERS)
    assert r.status_code == 200 and r.json() == DECISION.model_dump(mode="json")
    assert surface.coord.replan.triggered == 1


def test_trigger_on_a_disabled_controller_is_409_with_the_fix(surface):
    surface.coord.replan.enabled = False
    r = surface.client.post("/api/replan/trigger", headers=OWNER_HEADERS)
    assert r.status_code == 409 and r.json()["code"] == "replan_disabled"
    assert "HYPERCOMPUTE_REPLAN=1" in r.json()["fix"]
    assert surface.coord.replan.triggered == 0
    assert surface.client.get("/api/replan/decisions").status_code == 200  # reading stays possible


def test_replan_routes_without_a_controller_are_503(surface):
    surface.coord.replan = None
    assert surface.client.get("/api/replan/decisions").json()["code"] == "replan_unavailable"
    r = surface.client.post("/api/replan/trigger", headers=OWNER_HEADERS)
    assert r.status_code == 503 and r.json()["code"] == "replan_unavailable"


# --------------------------------------------------------------------------- HTTP: replace


def test_replace_loads_the_stored_plan_and_waits_for_the_manager(surface):
    plan = _store_plan(surface.store, _plan().model_copy(update={"id": "plan2"}))
    fake = FakeReplace()
    surface.monkeypatch.setattr(surface.coord.deployments, "replace", fake)
    r = surface.client.post(f"/api/deployments/{OLD_DEP}/replace", json={"plan_id": "plan2"}, headers=OWNER_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "ready" and body["revision"] == 4 and body["replaces"] == OLD_DEP
    [(dep_id, passed, reason, fallback)] = fake.calls
    assert dep_id == OLD_DEP and passed == plan and fallback is None and "plan2" in reason


def test_replace_needs_the_owner_token(surface):
    _store_plan(surface.store, _plan().model_copy(update={"id": "plan2"}))
    fake = FakeReplace()
    surface.monkeypatch.setattr(surface.coord.deployments, "replace", fake)
    for headers in ({}, {"X-Hypercompute-Owner": "wrong"}):
        r = surface.client.post(f"/api/deployments/{OLD_DEP}/replace", json={"plan_id": "plan2"}, headers=headers)
        assert r.status_code == 403
    assert fake.calls == []


@pytest.mark.parametrize("status, code", [(409, "not_admitted"), (409, "drain_timeout"), (422, "model_mismatch"),
                                          (500, "replace_failed")])
def test_replace_maps_the_managers_errors(surface, status, code):
    _store_plan(surface.store, _plan().model_copy(update={"id": "plan2"}))
    error = DeploymentError(status, code, f"refused: {code}", fix="a fix" if status == 409 else None)
    surface.monkeypatch.setattr(surface.coord.deployments, "replace", FakeReplace(error=error))
    r = surface.client.post(f"/api/deployments/{OLD_DEP}/replace", json={"plan_id": "plan2"}, headers=OWNER_HEADERS)
    assert r.status_code == status and r.json()["code"] == code and r.json()["detail"] == f"refused: {code}"


def test_replace_with_an_unknown_plan_or_deployment_uses_the_managers_errors(surface):
    r = surface.client.post(f"/api/deployments/{OLD_DEP}/replace", json={"plan_id": "missing"}, headers=OWNER_HEADERS)
    assert r.status_code == 404 and r.json()["code"] == "plan_not_found"
    _store_plan(surface.store, _plan().model_copy(update={"id": "plan2"}))  # real DeploymentManager.replace
    r = surface.client.post(f"/api/deployments/{OLD_DEP}/replace", json={"plan_id": "plan2"}, headers=OWNER_HEADERS)
    assert r.status_code == 404 and r.json()["code"] == "not_found"


@pytest.mark.parametrize("dep_id, body", [
    (OLD_DEP, {}), (OLD_DEP, {"plan_id": ""}), (OLD_DEP, {"plan_id": "../plans"}), (OLD_DEP, {"plan_id": "p" * 65}),
    (OLD_DEP, {"plan_id": "plan2", "reason": "extra fields are refused"}), ("dep.id", {"plan_id": "plan2"}),
])
def test_replace_validates_ids_and_body(surface, dep_id, body):
    fake = FakeReplace()
    surface.monkeypatch.setattr(surface.coord.deployments, "replace", fake)
    r = surface.client.post(f"/api/deployments/{dep_id}/replace", json=body, headers=OWNER_HEADERS)
    assert r.status_code == 422 and fake.calls == []


# --------------------------------------------------------------------------- path lookup


@pytest.mark.parametrize("peers, expected", [
    ([{"node_id": PEER, "connected": True, "path": {"mode": "relay", "rtt_ms": 36.9}}], "relay"),
    ([{"node_id": PEER, "connected": True, "path": {"mode": "direct"}}], "direct"),
    ([{"node_id": PEER, "connected": True, "path": {"mode": "connecting"}}], "unknown"),
    ([{"node_id": PEER, "connected": False, "path": {"mode": "relay"}}], "unknown"),
    ([{"node_id": PEER, "connected": True, "path": None}], "unknown"),
    ([{"node_id": "b" * 32, "connected": True, "path": {"mode": "direct"}}], "unknown"),
    ([], "unknown"), (None, "unknown"), (["not a peer"], "unknown"),
])
def test_iroh_path_mode(peers, expected):
    assert iroh_path_mode({"transport": "iroh", "peers": peers}, PEER) == expected


def test_lan_lifespan_binds_a_lan_path_lookup(surface):
    assert surface.coord.capacity.path_of is lan_path
    assert surface.coord.capacity.path_of(PEER) == "lan"


def test_iroh_lifespan_binds_each_peers_selected_path(store, events, tmp_path, monkeypatch):
    from hypercompute.transport.host import HostService

    status = {"transport": "iroh", "peers": [
        {"node_id": PEER, "connected": True, "path": {"mode": "relay"}},
        {"node_id": "b" * 32, "connected": True, "path": {"mode": "direct"}}]}
    peer = SimpleNamespace(ticket="test-ticket", status=lambda: status, close=_async_none)
    started = []

    async def start(coord, keypath, relay_url=None):
        started.append(coord.capacity.path_of)  # still unset: bound only once the transport is up
        return peer

    monkeypatch.setattr(HostService, "start", start)
    coord = make_coord(store, events, tmp_path, monkeypatch)
    coord.capacity = FakeCapacity()
    with TestClient(create_app(coord, transport="iroh"), base_url="http://127.0.0.1:8000",
                    client=("127.0.0.1", 50123)):
        lookup = coord.capacity.path_of
        assert started == [None]
        assert [lookup(n) for n in (PEER, "b" * 32, "c" * 32)] == ["relay", "direct", "unknown"]
        status["peers"][0] = {"node_id": PEER, "connected": False, "path": None}
        assert lookup(PEER) == "unknown"
        peer.status = _raise
        assert lookup("b" * 32) == "unknown"


def test_iroh_path_lookup_reads_the_transport_on_the_event_loop(store, events, tmp_path, monkeypatch):
    """explain, plan and the replan evaluator run in worker threads; HostService.status() walks dicts the event loop
    changes on join and leave, so a lookup from another thread must read it on the loop's thread."""
    from hypercompute.transport.host import HostService

    readers, loop_thread = [], []

    def status():
        readers.append(threading.get_ident())
        return {"transport": "iroh", "peers": [{"node_id": PEER, "connected": True, "path": {"mode": "direct"}}]}

    peer = SimpleNamespace(ticket="test-ticket", status=status, close=_async_none)

    async def start(coord, keypath, relay_url=None):
        loop_thread.append(threading.get_ident())
        return peer

    monkeypatch.setattr(HostService, "start", start)
    coord = make_coord(store, events, tmp_path, monkeypatch)
    coord.capacity = FakeCapacity()
    with TestClient(create_app(coord, transport="iroh"), base_url="http://127.0.0.1:8000",
                    client=("127.0.0.1", 50123)):
        lookup = coord.capacity.path_of
        seen = []
        worker = threading.Thread(target=lambda: seen.append(lookup(PEER)))
        worker.start()
        worker.join(10)
        assert seen == ["direct"] and lookup(PEER) == "direct"
        assert readers and set(readers) == set(loop_thread)
    assert lookup(PEER) == "unknown"  # the transport and its loop are gone after shutdown


def test_read_on_loop_runs_on_the_loop_thread_and_bounds_the_wait():
    assert read_on_loop(None, lambda: {"thread": threading.get_ident()}) == {"thread": threading.get_ident()}
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        assert read_on_loop(loop, lambda: {"thread": threading.get_ident()}) == {"thread": runner.ident}
        with pytest.raises(KeyError):
            read_on_loop(loop, lambda: {}["missing"])  # the reader's own error reaches the caller
        gate = threading.Event()
        loop.call_soon_threadsafe(gate.wait, 5)  # the loop is busy: the caller gives up after its timeout
        with pytest.raises(TimeoutError):
            read_on_loop(loop, dict, timeout_s=0.05)
        gate.set()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(5)
    with pytest.raises(RuntimeError):
        read_on_loop(loop, dict)  # stopped: nothing would ever run the read
    loop.close()


async def _async_none():
    return None


def _raise():
    raise RuntimeError("iroh path info unavailable")


# --------------------------------------------------------------------------- CLI against the app


def test_cli_explain_prints_lines_verbatim_then_candidates_skipped_and_labelled_assumptions(surface, capsys):
    args = _args("explain", MODEL, "--ctx", "8192", "--prompt-tokens", "1024", "--output-tokens", "64", "--warm",
                 "--slots", "3")
    assert capacity_cli.cmd_capacity(args, client=surface.client) == 0
    assert surface.coord.capacity.calls == [dict(model_id=MODEL, ctx_size=8192, prompt_tokens=1024,
                                                 output_tokens=64, warm=True, node_ids=None, slots=3)]
    out = capsys.readouterr().out.splitlines()
    assert all(line in out for line in EXPLAIN_LINES)
    assert out.index(EXPLAIN_LINES[-1]) < out.index("candidates (predicted from the assumptions below, not measured):")
    rows = [line for line in out if line.startswith(("  1 ", "  2 ", "  3 "))]
    assert rows[0].split() == ["1", "41234", "120.5", "local:CPU", "[0-7],", "local:CUDA0", "[8-27", "+", "head]"]
    assert rows[1].split()[:3] == ["2", "52000", "160.2"]
    assert rows[1].endswith(f"local:CPU [0-3], {PEER[:8]}:MTL0 [4-11], local:CUDA0 [12-27 + head] "
                            f"(local:CPU 2 streamed; {PEER[:8]}:MTL0 relay)")
    assert rows[2].endswith("local:CPU [0-27 + head]  (does not fit)")
    skipped = out.index("skipped devices:")
    assert out[skipped + 1:skipped + 3] == [f"  {EXPLAIN['best']['skipped'][0]}", f"  {EXPLAIN['warnings'][0]}"]
    labelled = out[out.index("assumptions:") + 1:]
    assert labelled[:3] == [f"  prior     {ASSUMPTIONS[0]}", f"  prior     {ASSUMPTIONS[1]}",
                            f"  estimate  {ASSUMPTIONS[2]}"]
    assert labelled[3] == f"  measured  {MEASURED[0]}"
    assert labelled[4].startswith("  measured  name ") and "value 1590000.0" in labelled[4]
    assert labelled[5] == f"  measured  {MEASURED[2]}"


def test_cli_explain_json_and_an_infeasible_model(surface, capsys):
    assert capacity_cli.cmd_capacity(_args("explain", MODEL, "--json"), client=surface.client) == 0
    assert json.loads(capsys.readouterr().out) == EXPLAIN
    infeasible = {**EXPLAIN, "feasible": False, "best": None, "candidates": [], "explain": ["no feasible layout"],
                  "assumptions": [], "measured": [], "warnings": []}
    surface.monkeypatch.setattr(surface.coord.capacity, "explain", lambda *a, **k: infeasible)
    assert capacity_cli.cmd_capacity(_args("explain", MODEL), client=surface.client) == 1
    out = capsys.readouterr().out
    assert "no feasible layout" in out and out.count("  none") == 3


def test_cli_explain_shows_the_gateway_refusal(surface, capsys):
    surface.coord.capacity.error = CapacityUnavailable("not_gguf", f"{MODEL} has no GGUF manifest")
    assert capacity_cli.cmd_capacity(_args("explain", MODEL), client=surface.client) == 1
    assert f"error: HTTP 422 not_gguf: {MODEL} has no GGUF manifest" in capsys.readouterr().out
    assert capacity_cli.cmd_capacity(_args("explain", MODEL, "--slots", "9"), client=surface.client) == 1
    assert "error: HTTP 422: slots: Input should be less than or equal to 8" in capsys.readouterr().out


def test_cli_explain_shows_a_host_state_refusal(surface, capsys):
    surface.coord.capacity.error = CapacityUnavailable("io", f"cannot read {MODEL}: disk gone")
    assert capacity_cli.cmd_capacity(_args("explain", MODEL), client=surface.client) == 1
    assert f"error: HTTP 503 io: cannot read {MODEL}: disk gone" in capsys.readouterr().out
    surface.coord.capacity.error = CapacityUnavailable("host_ineligible", "this machine is paused")
    assert capacity_cli.cmd_capacity(_args("explain", MODEL), client=surface.client) == 1
    assert "error: HTTP 409 host_ineligible: this machine is paused" in capsys.readouterr().out


def test_cli_pools_prints_one_row_per_pool(surface, capsys):
    assert capacity_cli.cmd_capacity(_args("pools"), client=surface.client) == 0
    out = capsys.readouterr().out
    assert f"host memory: {fmt_bytes(20 * GiB)} available of {fmt_bytes(32 * GiB)}" in out
    assert "via GetPerformanceInfo" in out
    local = next(line for line in out.splitlines() if line.startswith("  local    ram"))
    assert local.split()[2:] == ["system_ram", *fmt_bytes(32 * GiB).split(), *fmt_bytes(20 * GiB).split(),
                                 *fmt_bytes(6 * GiB).split(), "none", *fmt_bytes(14 * GiB).split(), "3", "CPU",
                                 "(inventory)"]
    assert f"  {PEER[:8]} uma" in out and "MTL0 (heartbeat)" in out
    assert "  excluded local vram:CUDA1: unknown free memory" in out
    assert "  note: test note: no single pool carries this" in out


def test_cli_decisions_label_the_numbers_as_predictions(surface, capsys):
    assert capacity_cli.cmd_capacity(_args("decisions"), client=surface.client) == 0
    out = capsys.readouterr().out
    assert f"manual: keep  deployment {OLD_DEP[:8]} revision 3" in out and "  reason: gain below hysteresis" in out
    assert ("  predicted by the optimizer, not measured: decode 120.0 -> 110.0 ms/token, gain +8.3%, reload 9000 ms, "
            "replay 1400 ms, break-even 1040 tokens") in out
    assert "  proposed plan: plan2" in out
    assert capacity_cli.cmd_capacity(_args("decisions", "--json"), client=surface.client) == 0
    assert json.loads(capsys.readouterr().out) == [DECISION.model_dump(mode="json")]
    surface.monkeypatch.setattr(surface.coord.replan, "decisions", lambda: [])
    assert capacity_cli.cmd_capacity(_args("decisions"), client=surface.client) == 0
    assert capsys.readouterr().out.strip() == "no replan decisions recorded"


def test_cli_replan_uses_the_owner_token_and_shows_a_disabled_controller(surface, capsys):
    assert capacity_cli.cmd_capacity(_args("replan"), client=surface.client) == 0
    assert surface.coord.replan.triggered == 1 and "manual: keep" in capsys.readouterr().out
    surface.coord.replan.enabled = False
    assert capacity_cli.cmd_capacity(_args("replan"), client=surface.client) == 1
    out = capsys.readouterr().out
    assert "error: HTTP 409 replan_disabled: the replan controller is disabled" in out
    assert "fix:   restart the host with HYPERCOMPUTE_REPLAN=1" in out


def test_cli_replace_sends_the_plan_id_and_reports_the_outcome(surface, capsys):
    _store_plan(surface.store, _plan().model_copy(update={"id": "plan2"}))
    fake = FakeReplace()
    surface.monkeypatch.setattr(surface.coord.deployments, "replace", fake)
    assert capacity_cli.cmd_capacity(_args("replace", OLD_DEP, "--plan-id", "plan2"), client=surface.client) == 0
    assert fake.calls[0][0] == OLD_DEP and fake.calls[0][1].id == "plan2"
    assert capsys.readouterr().out.strip() == f"deployment eeeeeeee revision 4: ready (replaces {OLD_DEP[:8]})"
    fake.state = DeploymentState.failed
    assert capacity_cli.cmd_capacity(_args("replace", OLD_DEP, "--plan-id", "plan2"), client=surface.client) == 1
    assert "failed" in capsys.readouterr().out
    fake.error = DeploymentError(409, "not_admitted", "plan plan2 is not admitted (no room); the deployment keeps "
                                 "serving")
    assert capacity_cli.cmd_capacity(_args("replace", OLD_DEP, "--plan-id", "plan2"), client=surface.client) == 1
    assert "error: HTTP 409 not_admitted: plan plan2 is not admitted" in capsys.readouterr().out


def test_cli_replace_that_recovered_onto_the_previous_plan_is_not_a_success(surface, capsys):
    """replace() answers with a ready deployment after a rollback too; the requested plan did not load."""
    _store_plan(surface.store, _plan().model_copy(update={"id": "plan2"}))
    surface.monkeypatch.setattr(surface.coord.deployments, "replace", FakeReplace(recovered=True))
    assert capacity_cli.cmd_capacity(_args("replace", OLD_DEP, "--plan-id", "plan2"), client=surface.client) == 1
    out = capsys.readouterr().out
    assert f"deployment rrrrrrrr revision 5: ready (replaces {'f' * 8})" in out
    assert "plan plan2 did not become ready; deployment rrrrrrrr serves plan oldplan instead" in out
    assert "rolled back after revision 4 failed: load_timeout" in out


@pytest.mark.parametrize("action, outcome, code", [
    ("keep", None, 0), ("defer", None, 0), ("replace", "activated", 0), ("failed", None, 1),
    ("replace", "rolled_back", 1), ("replace", "fell_back", 1), ("replace", "failed: load_timeout", 1),
    ("replace", "refused: drain_timeout", 1), ("replace", "stopped", 1),
])
def test_cli_replan_exit_code_follows_the_outcome(surface, capsys, action, outcome, code):
    surface.coord.replan.decision = DECISION.model_copy(update={"action": action, "outcome": outcome})
    assert capacity_cli.cmd_capacity(_args("replan"), client=surface.client) == code
    out = capsys.readouterr().out
    assert f"manual: {action}" in out and (outcome is None or f"  outcome: {outcome}" in out)


# --------------------------------------------------------------------------- CLI against a fake transport


def _client(handler):
    return httpx.Client(base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(handler))


def test_cli_reports_a_host_that_is_not_running(capsys):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    assert capacity_cli.cmd_capacity(_args("pools", "--port", "8123"), client=_client(handler)) == 2
    out = capsys.readouterr().out
    assert "no Hypercompute host is running on 127.0.0.1:8123 (ConnectError)" in out and "hypercompute host" in out


def test_cli_timeout_on_replace_says_the_replacement_keeps_running(capsys):
    seen = {}

    def handler(request):
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        seen["owner"], seen["body"] = request.headers.get("x-hypercompute-owner"), json.loads(request.content)
        seen["read_timeout"] = request.extensions["timeout"]["read"]
        raise httpx.ReadTimeout("timed out", request=request)

    assert capacity_cli.cmd_capacity(_args("replace", OLD_DEP, "--plan-id", "plan2"), client=_client(handler)) == 1
    out = capsys.readouterr().out
    assert seen == {"owner": "tok", "body": {"plan_id": "plan2"}, "read_timeout": capacity_cli.SWITCH_TIMEOUT_S}
    assert "did not answer in time" in out and "keeps running there" in out


def test_cli_reports_a_reply_that_is_not_the_gateways(capsys):
    def handler(request):
        return httpx.Response(200, text="<html>some other service</html>")

    assert capacity_cli.cmd_capacity(_args("decisions"), client=_client(handler)) == 2
    assert "unexpected reply from the host" in capsys.readouterr().out


def test_cli_without_an_owner_token_does_not_send_the_write(capsys):
    writes = []

    def handler(request):
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"version": "0.1.0"})
        writes.append(request)
        return httpx.Response(200, json={})

    assert capacity_cli.cmd_capacity(_args("replan"), client=_client(handler)) == 1
    assert writes == [] and "no owner token" in capsys.readouterr().out


# --------------------------------------------------------------------------- parser and formatting


def test_capacity_parser_defaults_and_id_checks():
    args = _args("explain", MODEL)
    assert (args.ctx, args.prompt_tokens, args.output_tokens, args.warm, args.slots, args.json) == (
        4096, 512, 256, False, 2, False)
    assert args.func is capacity_cli.cmd_capacity and args.port == cli.config.DEFAULT_GATEWAY_PORT
    assert _args("pools", "--port", "8123").port == 8123
    for argv in (["replace", "../api/sessions", "--plan-id", "p"], ["replace", OLD_DEP, "--plan-id", "a/b"],
                 ["replace", OLD_DEP], ["frobnicate"], []):
        with pytest.raises(SystemExit) as error:
            _args(*argv)
        assert error.value.code == 2


@pytest.mark.parametrize("units, text", [([0, 1, 2, 7], "0-2,7"), ([5], "5"), ([], "-"), (None, "-"),
                                         ([3, 1, 2, 2, 28], "1-3,28")])
def test_unit_spans(units, text):
    assert capacity_cli._span(units) == text


@pytest.mark.parametrize("line, label", [
    ("local:CPU decode 12.5 ms/GiB (prior: planner figure)", "prior"),
    ("setup 4000 ms (estimate: engine start)", "estimate"),
    ("MTL0 decode 10.85 ms/GiB (reported: by the peer)", "reported"),
    ("local:CPU decode 12.5 ms/GiB (prior: an older figure (estimate) was dropped)", "prior"),
    ("storage read 3.45 GiB/s (measured: storage probe)", "measured"),
    ("peer upload caches are assumed empty: a cold load uploads every weight byte a peer holds", "note"),
    ("no measurement replaced the prior for MTL0 decode", "note"),
])
def test_assumption_label_is_the_lines_own_provenance_marker(line, label):
    assert capacity_cli._label(line) == label


def test_explain_labels_an_unmarked_assumption_as_a_note_never_a_prior():
    unmarked = "peer upload caches are assumed empty: a cold load uploads every weight byte a peer holds"
    lines = capacity_cli.explain_lines({"model_id": MODEL, "assumptions": [unmarked, ASSUMPTIONS[0]]})
    assert lines[lines.index("assumptions:") + 1:] == [f"  note      {unmarked}", f"  prior     {ASSUMPTIONS[0]}"]


def test_main_dispatches_capacity_to_capacity_cli(monkeypatch):
    seen = []
    monkeypatch.setattr(capacity_cli, "cmd_capacity", lambda args: seen.append((args.action, args.port)) or 7)
    assert cli.main(["capacity", "pools", "--port", "8123"]) == 7
    assert seen == [("pools", 8123)]
    assert not hasattr(cli, "cmd_capacity") and not hasattr(cli, "_label")  # the group lives in capacity_cli only


def test_layout_without_the_services_text_is_built_from_the_stages():
    stages = [_stage("local", "CPU", range(8)), _stage(PEER, "MTL0", range(8, 29), "direct", streamed=(9,))]
    assert capacity_cli._layout({"stages": stages}) == (f"local:CPU [0-7], {PEER[:8]}:MTL0 [8-28] "
                                              f"({PEER[:8]}:MTL0 1 streamed direct)")
    assert capacity_cli._layout({}) == "-"


def test_pool_rows_tolerate_missing_fields():
    lines = capacity_cli.pools_lines({"host_memory": {"error": "GetPerformanceInfo failed"},
                             "pools": [{"node_id": "local", "pool_id": "ssd", "free_bytes": 512 * MiB}]})
    assert lines[0] == "host memory: not observed (GetPerformanceInfo failed)"
    assert lines[2].split()[:4] == ["local", "ssd", "None", "-"]
