"""Host-managed peers: consent, command refusal and idempotency, worker restarts, the heartbeat wire in both
directions (old agent with a new host, new agent with an old host), the host queue, owner routes and CLI."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute import capacity_cli, cli, config
from hypercompute.agent import join
from hypercompute.agent import managed as managed_module
from hypercompute.agent.join import Agent, AgentError
from hypercompute.agent.managed import ManagedConsent, ManagedPeer, build_consent, consent_lines
from hypercompute.agent.seed_cache import MIN_FREE_BYTES, ManagedError, SeedStopped, cache_name
from hypercompute.schemas import (
    MANAGED_PEERS_FEATURE,
    MAX_SEED_TENSORS_PER_COMMAND,
    CommandAck,
    EngineDevice,
    ExtraEngineInfo,
    HeartbeatRequest,
    HeartbeatResponse,
    ManagedReport,
    MemoryPool,
    Model,
    ModelArtifact,
    ModelFile,
    NodeState,
    PeerCommand,
    PlanRequest,
    PlanStrategy,
    SeedTensor,
    WorkerState,
    utcnow,
)
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.peer_seed import ManagedPeers, ManagedRefusal, SeedBuild, load_plan, plan_layers
from hypercompute.services.planner import make_plan
from hypercompute.units import GiB
from pydantic import Field
from test_app_http import OWNER, make_coord, open_client
from test_sessions_and_join import join_request, registry, sessions

EPOCH = "0123456789abcdef"
NODE = "a" * 32
REPO, REV = "org/model-GGUF", "a" * 40
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/model.gguf"


def mac_inventory():
    return make_inventory("macos", engine=engine_info(accelerator="metal").model_copy(update={"devices": [
        EngineDevice(name="MTL0", description="Apple M4", total_bytes=12 * GiB, free_bytes=11 * GiB),
        EngineDevice(name="BLAS", description="Accelerate", total_bytes=0, free_bytes=0)]}))


def consent(max_disk: int = 0, allowed=("MTL0", "CPU"), threads: int = 10) -> ManagedConsent:
    return ManagedConsent(tuple(allowed), max_disk, threads)


def command(seq: int, kind: str = "worker", cid: str | None = None, **fields) -> dict:
    cmd = PeerCommand(id=cid or f"{seq:032x}", epoch=EPOCH, seq=seq, kind=kind, **fields)
    return cmd.model_dump(mode="json")


class Worker:
    """Stands in for the agent's worker: records restarts, can be slow, busy or failing."""

    def __init__(self, devices=("MTL0",), threads=None) -> None:
        self.devices, self.threads = list(devices), threads
        self.restarts: list[tuple[list[str], int | None]] = []
        self.busy = False
        self.fail: Exception | None = None
        self.gate = threading.Event()
        self.gate.set()

    def restart(self, devices, threads) -> None:
        self.gate.wait(5)
        self.restarts.append((list(devices), threads))
        if self.fail is not None:
            raise self.fail
        self.devices, self.threads = list(devices), threads


def peer(tmp_path, worker: Worker, c: ManagedConsent | None = None, catalog=None) -> ManagedPeer:
    return ManagedPeer(c or consent(), tmp_path, cache_enabled=True, restart_worker=worker.restart,
                       current_worker=lambda: (worker.devices, worker.threads), is_busy=lambda: worker.busy,
                       catalog=catalog or (lambda: {}))


def acks(p: ManagedPeer) -> dict[str, CommandAck]:
    return {a.id: a for a in p.report(["MTL0"], None, 0).acks}


def settle(p: ManagedPeer) -> None:
    assert p.wait_idle(5)


# --------------------------------------------------------------------------- consent


def test_default_consent_on_macos_is_metal_and_cpu():
    c = build_consent(mac_inventory(), ["MTL0"], None, None)
    assert c.devices_allowed == ("MTL0", "CPU") and c.max_disk_bytes == 0 and not c.seeding_allowed
    assert c.min_free_bytes == MIN_FREE_BYTES == 20 * GiB
    assert build_consent(mac_inventory(), ["MTL0"], "mtl0,cpu", "80GiB").devices_allowed == ("MTL0", "CPU")
    assert build_consent(mac_inventory(), ["MTL0"], None, "80GiB").max_disk_bytes == 80 * GiB


@pytest.mark.parametrize("allowed,current,match", [
    ("MTL0,CUDA0", ["MTL0"], "not visible"),
    ("CPU", ["MTL0"], "must be within"),
    ("MTL0,MTL0", ["MTL0"], "twice"),
    ("rm -rf", ["MTL0"], "not a device name"),
])
def test_consent_refusals(allowed, current, match):
    with pytest.raises(ValueError, match=match):
        build_consent(mac_inventory(), current, allowed, None)


def test_cli_needs_managed_for_its_flags_and_turns_the_cache_on(monkeypatch):
    args = SimpleNamespace(managed=False, devices_allowed=None, max_disk="10GiB")
    with pytest.raises(AgentError, match="only apply with --managed"):
        cli._managed_consent(args, mac_inventory(), ["MTL0"])
    assert cli._managed_consent(SimpleNamespace(), mac_inventory(), ["MTL0"]) is None  # older Namespace: no change
    with pytest.raises(AgentError, match="--managed"):
        cli._managed_consent(SimpleNamespace(managed=True, devices_allowed="CUDA0", max_disk=None),
                             mac_inventory(), ["MTL0"])


def test_join_output_shows_the_consent(capsys):
    agent = make_agent(managed=consent(80 * GiB))
    join.print_plan(agent, False)
    out = capsys.readouterr().out
    assert "Managed by the host (--managed)" in out and "MTL0,CPU" in out
    assert "80.0 GiB" in out and "20.0 GiB" in out and "huggingface.co only" in out
    lines = consent_lines(consent(), config.paths().rpc_cache)
    assert any("may not download anything" in line for line in lines)


def test_without_managed_nothing_changes(capsys):
    agent = make_agent()
    join.print_plan(agent, False)
    assert "Managed" not in capsys.readouterr().out
    assert agent.managed is None and agent.managed_heartbeat() == {}
    assert "managed_by_host" not in agent.status_doc()


# --------------------------------------------------------------------------- commands on the agent


def test_worker_command_outside_consent_is_refused_and_never_runs(tmp_path):
    w = Worker()
    p = peer(tmp_path, w)
    p.handle([command(1, devices=["CUDA0"]), command(2, devices=["MTL0", "CPU"], threads=99)])
    a = acks(p)
    assert a[f"{1:032x}"].status == "refused" and "outside what this machine allowed" in a[f"{1:032x}"].detail
    assert a[f"{2:032x}"].status == "refused" and "threads" in a[f"{2:032x}"].detail
    assert w.restarts == []


def test_unknown_malformed_or_newer_commands_are_refused(tmp_path):
    w = Worker()
    p = peer(tmp_path, w)
    raw = [{"id": "c" * 32, "kind": "shell", "argv": ["rm", "-rf", "/"], "epoch": EPOCH, "seq": 1},
           {**command(2, devices=["MTL0"]), "devices": ["MTL0; reboot"]},
           {**command(3, devices=["MTL0", "CPU"]), "version": 2},
           {"id": "not-hex", "kind": "worker"}, "junk", None]
    p.handle(raw)
    a = acks(p)
    assert a["c" * 32].status == "refused" and a["c" * 32].kind == "shell"
    assert a[f"{2:032x}"].status == "refused"
    assert a[f"{3:032x}"].status == "refused" and "version 2" in a[f"{3:032x}"].detail
    assert len(a) == 3 and w.restarts == []
    p.handle({"not": "a list"})  # ignored


def test_worker_command_is_idempotent(tmp_path):
    w = Worker()
    p = peer(tmp_path, w)
    cmd = command(1, devices=["mtl0", "cpu"], threads=4)
    p.handle([cmd])
    settle(p)
    p.handle([cmd])  # re-delivered before the host saw the ack
    settle(p)
    assert w.restarts == [(["MTL0", "CPU"], 4)]
    assert acks(p)[cmd["id"]].status == "applied"
    p.handle([command(2, devices=["MTL0", "CPU"])])  # same settings (threads kept): nothing to restart
    assert w.restarts == [(["MTL0", "CPU"], 4)]
    assert "already runs" in acks(p)[f"{2:032x}"].detail


def test_restart_in_progress_defers_the_next_command(tmp_path):
    w = Worker()
    w.gate.clear()
    p = peer(tmp_path, w)
    p.handle([command(1, devices=["MTL0", "CPU"])])
    p.handle([command(1, devices=["MTL0", "CPU"]), command(2, devices=["CPU"])])
    assert f"{2:032x}" not in acks(p)  # not acknowledged: re-sent by the host, run after the restart
    w.gate.set()
    settle(p)
    p.handle([command(2, devices=["CPU"])])
    settle(p)
    assert w.restarts == [(["MTL0", "CPU"], None), (["CPU"], None)]


def test_older_worker_command_is_superseded_and_busy_worker_refused(tmp_path):
    w = Worker()
    p = peer(tmp_path, w)
    p.handle([command(5, devices=["MTL0", "CPU"])])
    settle(p)
    p.handle([command(4, devices=["CPU"])])
    assert "superseded" in acks(p)[f"{4:032x}"].detail
    w.busy = True
    p.handle([command(6, devices=["CPU"])])
    assert acks(p)[f"{6:032x}"].status == "refused" and "deployment" in acks(p)[f"{6:032x}"].detail
    assert len(w.restarts) == 1


def test_failed_restart_is_reported(tmp_path):
    w = Worker()
    w.fail = AgentError("ggml-rpc-server exited early (code 1)")
    p = peer(tmp_path, w)
    p.handle([command(1, devices=["MTL0", "CPU"])])
    settle(p)
    a = acks(p)[f"{1:032x}"]
    assert a.status == "failed" and "exited early" in a.detail
    assert "exited early" in p.report(["MTL0"], None, 0).worker_error


def catalog() -> dict[str, ModelArtifact]:
    f = ModelFile(name="model.gguf", url=URL, sha256="0" * 64, size_bytes=10**9)
    return {"m1": ModelArtifact(id="m1", display_name="m", family="llama", params_b=1.0, quantization="Q4",
                                repo=REPO, revision=REV, license="mit", files=[f])}


def seed_command(seq: int, file: str = "model.gguf", parts: int = 1, part: int = 1, job: str = "d" * 32) -> dict:
    tensors = [SeedTensor(file=file, offset=0, nbytes=1000, fnv64="1" * 16)]
    return command(seq, "seed", seed={"job_id": job, "model_id": "m1", "repo": REPO, "revision": REV,
                                      "part": part, "parts": parts, "tensors": [t.model_dump() for t in tensors]})


def test_seed_needs_disk_consent(tmp_path):
    p = peer(tmp_path, Worker(), consent(max_disk=0), lambda: catalog())
    p.handle([seed_command(1)])
    assert "--max-disk" in acks(p)[f"{1:032x}"].detail
    assert p.runner.progress() is None


def test_seed_with_path_traversal_is_refused(tmp_path):
    p = peer(tmp_path, Worker(), consent(max_disk=10 * GiB), lambda: catalog())
    p.handle([seed_command(1, file="../../../../etc/passwd")])
    a = acks(p)[f"{1:032x}"]
    assert a.status == "refused" and "unsafe file name" in a.detail
    assert p.runner.progress() is None


def test_seed_is_accepted_and_reported(tmp_path):
    p = peer(tmp_path, Worker(), consent(max_disk=10 * GiB), lambda: catalog())
    p.runner._workers = 1
    p.runner._fetch = lambda task: SimpleNamespace(fnv64=task.fnv64, status="failed", downloaded=0, error="offline")
    p.runner._executor_factory = lambda n: __import__("concurrent.futures").futures.ThreadPoolExecutor(n)
    p.handle([seed_command(1, parts=2)])
    assert acks(p)[f"{1:032x}"].status == "accepted"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and p.runner.progress().tensors_failed == 0:
        time.sleep(0.02)
    report = p.report(["MTL0"], None, 0)
    assert report.seed.job_id == "d" * 32 and report.seed.parts == 2 and report.seed.tensors_failed == 1
    p.handle([command(2, "seed_cancel")])
    assert acks(p)[f"{2:032x}"].status == "applied"
    p.close()


# --------------------------------------------------------------------------- the agent on the wire


def make_agent(managed: ManagedConsent | None = None, system: str = "macos") -> Agent:
    inv = mac_inventory() if system == "macos" else make_inventory(system)
    devices = ["MTL0"] if system == "macos" else ["CUDA0"]
    c = contribution(devices).model_copy(update={"cache_enabled": bool(managed and managed.seeding_allowed)})
    return Agent(host="127.0.0.1", port=8000, code="code", display_name="friend", inv=inv, contribution=c,
                 advertise_ip="10.0.0.5", worker_port=47052, managed_consent=managed)


class OldHeartbeatRequest(Model):
    """The heartbeat a host without managed peers accepts (extra fields forbidden), as at 70f8884."""

    worker_state: WorkerState
    worker_exit_code: int | None = None
    memory_pools: list[MemoryPool] = Field(default_factory=list, max_length=64)
    devices: list = Field(default_factory=list, max_length=64)
    worker_cpu_seconds: float | None = None
    sent_at: datetime
    extra_engines: list[ExtraEngineInfo] | None = Field(default=None, max_length=8)


def run_heartbeats(agent: Agent, handler, monkeypatch, timeout: float = 8.0) -> None:
    monkeypatch.setattr(join.inventory, "fresh_measurements", lambda inv: ([], []))
    monkeypatch.setattr(config, "HEARTBEAT_INTERVAL_S", 0.01)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.wait_for(agent.heartbeat_loop(client), timeout=timeout)

    asyncio.run(run())


def reply(**extra) -> httpx.Response:
    return httpx.Response(200, json={"state": "eligible", "state_reason": None, "removed": False,
                                     "deployments": [], **extra})


def test_new_agent_with_an_old_host_sends_nothing_new(monkeypatch):
    agent = make_agent(consent(80 * GiB))
    agent.credential = "cred"
    agent.host_features = []  # the join reply of an older host has no features
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        OldHeartbeatRequest.model_validate(body)  # the old host's strict model accepts it
        agent.stop_event.set()
        return reply()

    run_heartbeats(agent, handler, monkeypatch)
    assert bodies and "managed" not in bodies[0]
    assert not agent.managed_active


def test_old_agent_ignores_new_reply_fields(monkeypatch):
    agent = make_agent()  # no --managed: behaves like an agent from before the feature
    agent.credential = "cred"
    agent.host_features = [MANAGED_PEERS_FEATURE]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "managed" not in body
        OldHeartbeatRequest.model_validate(body)
        agent.stop_event.set()
        return reply(commands=[command(1, devices=["CPU"])], features=["future.v9"])

    run_heartbeats(agent, handler, monkeypatch)
    assert agent.state == "eligible"


def test_old_agent_heartbeat_validates_on_the_new_host():
    old = {"worker_state": "running", "worker_exit_code": None, "memory_pools": [], "devices": [],
           "worker_cpu_seconds": 1.0, "sent_at": "2026-10-22T00:00:00Z", "extra_engines": None}
    assert HeartbeatRequest.model_validate(old).managed is None


def test_new_agent_with_a_new_host_reports_and_acknowledges(monkeypatch):
    agent = make_agent(consent(80 * GiB))
    agent.credential = "cred"
    agent.host_features = [MANAGED_PEERS_FEATURE]
    seen: list[dict] = []
    bad = command(1, devices=["CUDA0"])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        HeartbeatRequest.model_validate(body)
        if len(seen) >= 2:
            agent.stop_event.set()
        return reply(commands=[bad])

    run_heartbeats(agent, handler, monkeypatch)
    first, second = (ManagedReport.model_validate(b["managed"]) for b in seen[:2])
    assert first.devices_allowed == ["MTL0", "CPU"] and first.max_disk_bytes == 80 * GiB and first.cache_enabled
    assert first.worker_devices == ["MTL0"] and first.acks == []
    assert [(a.id, a.status) for a in second.acks] == [(bad["id"], "refused")]
    assert agent.managed.cache_dir == config.paths().rpc_cache / "rpc"  # what rpc-server uses with LLAMA_CACHE
    agent.managed.close()


def test_agent_stops_reporting_when_the_host_refuses_the_field(monkeypatch, capsys):
    agent = make_agent(consent())
    agent.credential = "cred"
    agent.host_features = [MANAGED_PEERS_FEATURE]
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "managed" in body:
            return httpx.Response(422, json={"detail": "extra fields not permitted"})
        agent.stop_event.set()
        return reply()

    run_heartbeats(agent, handler, monkeypatch)
    assert "managed" in bodies[0] and "managed" not in bodies[1]
    assert "refused the managed-peer report" in capsys.readouterr().out


def test_join_reads_host_features(monkeypatch):
    agent = make_agent(consent())

    def post(url, json, timeout):
        return httpx.Response(200, json={"credential": "c", "state": "verifying",
                                         "features": [MANAGED_PEERS_FEATURE, 7]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(join.httpx, "post", post)
    agent.join()
    assert agent.host_features == [MANAGED_PEERS_FEATURE] and agent.managed_active


# --------------------------------------------------------------------------- worker restart on the agent

BANNER = ("Starting RPC server v8.0.0\n  endpoint       : 10.0.0.5:47052\n  local cache    : n/a\nDevices:\n"
          "  MTL0: Apple M4 (12124 MiB, 12123 MiB free)\n  CPU: Apple M4 (16384 MiB, 9000 MiB free)\n"
          "  transport      : TCP\n")


class FakeProc:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_log(self) -> str:
        return self.text

    def alive(self) -> bool:
        return True


def fake_engine(monkeypatch, agent: Agent, banners: list[str]) -> list[list[str]]:
    started: list[list[str]] = []
    installed = SimpleNamespace(binary=lambda name: "ggml-rpc-server", bin_dir=".")
    monkeypatch.setattr(join.engine, "find_installed", lambda: installed)

    def start(key, exe, args, cwd, log_path, env):
        started.append(list(args))
        return FakeProc(banners.pop(0) if banners else BANNER)

    monkeypatch.setattr(agent.sup, "start", start)
    monkeypatch.setattr(agent.sup, "stop", lambda key, grace_s=8.0: None)
    return started


def test_reconfigure_restarts_with_new_devices_and_threads(monkeypatch):
    agent = make_agent(consent())
    started = fake_engine(monkeypatch, agent, [BANNER])
    states = []
    monkeypatch.setattr(agent, "_revoke_peer_worker", lambda: states.append(agent.worker_status()[0]))
    agent.reconfigure_worker(["MTL0", "CPU"], 4)
    assert started == [["-H", "10.0.0.5", "-p", "47052", "-d", "MTL0,CPU", "-t", "4"]]
    assert states == [WorkerState.starting]
    assert agent.contribution.devices == ["MTL0", "CPU"] and agent.threads == 4 and agent.worker_generation == 1
    assert agent.worker_devices_seen == ["MTL0", "CPU"]
    agent.host_features = [MANAGED_PEERS_FEATURE]
    sent = agent.managed_heartbeat()["managed"]
    assert sent.worker_devices == ["MTL0", "CPU"] and sent.worker_threads == 4 and sent.worker_generation == 1
    agent.managed.close()


def test_reconfigure_restores_the_old_worker_when_the_new_one_differs(monkeypatch):
    agent = make_agent(consent())
    only_mtl = BANNER.replace("  CPU: Apple M4 (16384 MiB, 9000 MiB free)\n", "")
    started = fake_engine(monkeypatch, agent, [only_mtl, only_mtl, only_mtl, only_mtl])
    monkeypatch.setattr(join.time, "sleep", lambda s: None)
    with pytest.raises(AgentError, match="runs again"):
        agent.reconfigure_worker(["MTL0", "CPU"], None)
    assert agent.contribution.devices == ["MTL0"] and agent.threads is None
    assert started[-1] == ["-H", "10.0.0.5", "-p", "47052", "-d", "MTL0"]
    assert not agent.restarting
    agent.managed.close()


@pytest.mark.parametrize("stop", ["leave", "pause"])
def test_leave_or_pause_during_a_host_requested_restart_leaves_no_worker_running(monkeypatch, stop):
    """Regression: leave() and pause() stopped the worker without waiting for a restart thread, which could then
    start a new rpc-server after the agent had gone, or while the contributor had paused sharing."""
    agent = make_agent(consent())
    events: list[str] = []
    gate = threading.Event()

    def slow_start() -> None:
        events.append("start")
        gate.wait(5)

    monkeypatch.setattr(agent, "start_worker", slow_start)
    monkeypatch.setattr(agent.sup, "stop", lambda key, grace_s=8.0: events.append("stop"))
    monkeypatch.setattr(agent, "_revoke_peer_worker", lambda: None)
    monkeypatch.setattr(agent, "_say_goodbye", lambda: None)
    restart = threading.Thread(target=agent.reconfigure_worker, args=(["MTL0", "CPU"], None))
    restart.start()
    deadline = time.monotonic() + 5
    while events != ["stop", "start"] and time.monotonic() < deadline:
        time.sleep(0.01)
    leaver = threading.Thread(target=getattr(agent, stop))
    leaver.start()
    time.sleep(0.2)
    assert events == ["stop", "start"]  # it waits for the restart instead of racing it
    gate.set()
    restart.join(5)
    leaver.join(5)
    assert events == ["stop", "start", "stop"]  # the worker the restart started is the one leave stops
    events.clear()
    with pytest.raises(AgentError, match="stopping or paused"):  # and nothing starts after that
        agent._start_worker_checked(["MTL0"])
    assert "start" not in events
    agent.managed.close()


def test_reconfigure_refused_while_paused():
    agent = make_agent(consent())
    agent.paused = True
    with pytest.raises(ManagedError, match="paused"):
        agent.reconfigure_worker(["CPU"], None)
    agent.managed.close()


# --------------------------------------------------------------------------- the host queue


def report(**over) -> ManagedReport:
    base = {"devices_allowed": ["MTL0", "CPU"], "max_disk_bytes": 80 * GiB, "min_free_bytes": 20 * GiB,
            "max_threads": 10, "cache_enabled": True, "worker_devices": ["MTL0"], "worker_generation": 0}
    return ManagedReport(**{**base, **over})


def test_host_queue_delivers_until_acknowledged():
    mp = ManagedPeers()
    assert mp.on_heartbeat(NODE, None) == (None, False)
    with pytest.raises(ManagedRefusal) as exc:
        mp.request_worker(NODE, ["CPU"], None, busy=False)
    assert exc.value.code == "not_managed"
    mp.on_heartbeat(NODE, report())
    first = mp.request_worker(NODE, ["mtl0", "cpu"], 4, busy=False)
    newest = mp.request_worker(NODE, ["CPU"], None, busy=False)  # supersedes the pending one
    cmds, _ = mp.on_heartbeat(NODE, report())
    assert [c.id for c in cmds] == [newest.id] and first.seq < newest.seq
    assert mp.on_heartbeat(NODE, None) == (None, False)  # never to an agent that sends no report
    cmds, _ = mp.on_heartbeat(NODE, report(acks=[CommandAck(id=newest.id, kind="worker", status="applied")]))
    assert cmds is None
    snap = mp.snapshot([], lambda n: [])
    assert snap == []


@pytest.mark.parametrize("devices,threads,busy,code", [
    (["CUDA0"], None, False, "outside_consent"), (["CPU"], 64, False, "outside_consent"),
    (["CPU", "cpu"], None, False, "duplicate_device"), (["CPU"], None, True, "peer_busy")])
def test_host_refuses_outside_consent(devices, threads, busy, code):
    mp = ManagedPeers()
    mp.on_heartbeat(NODE, report())
    with pytest.raises(ManagedRefusal) as exc:
        mp.request_worker(NODE, devices, threads, busy=busy)
    assert exc.value.code == code


def tensors(n: int) -> list[SeedTensor]:
    return [SeedTensor(file="gpt-oss-120b-MXFP4.gguf", offset=10**10 + i * 10**9, nbytes=1_660_000_000 + i,
                       fnv64=f"{i:016x}") for i in range(n)]


@pytest.mark.anyio
async def test_seed_parts_go_out_one_per_reply_and_fit_an_iroh_frame():
    mp = ManagedPeers()
    mp.on_heartbeat(NODE, report())
    n = MAX_SEED_TENSORS_PER_COMMAND + 10
    job = mp.start_seed(NODE, "m1", [30, 35], lambda stop: SeedBuild(REPO, REV, tensors(n), 10**9, []))
    await asyncio.sleep(0.05)
    assert job.state == "queued" and job.parts == 2 and job.tensors == n
    cmds, _ = mp.on_heartbeat(NODE, report())
    assert [c.seed.part for c in cmds] == [1]
    body = HeartbeatResponse(state=NodeState.eligible, state_reason=None, removed=False, deployments=[],
                             commands=cmds).model_dump_json()
    assert len(body.encode()) < 64 * 1024 - 4096  # transport/iroh.py MAX_FRAME_BYTES, with room to spare
    accepted = CommandAck(id=cmds[0].id, kind="seed", status="accepted")
    cmds, _ = mp.on_heartbeat(NODE, report(acks=[accepted]))
    assert [c.seed.part for c in cmds] == [2]
    cmds, _ = mp.on_heartbeat(NODE, report(acks=[CommandAck(id=cmds[0].id, kind="seed", status="accepted")]))
    assert cmds is None and job.state == "delivered" and job.parts_accepted == 2


@pytest.mark.anyio
async def test_seed_parts_with_long_file_names_still_fit_an_iroh_frame():
    """Regression: parts were cut by tensor count only, so 256 tensors with long catalog file names made a reply
    over the 64 KiB frame, which failed every heartbeat until the agent gave up on the host."""
    mp = ManagedPeers()
    mp.on_heartbeat(NODE, report())
    long_name = [t.model_copy(update={"file": "x" * 250 + ".gguf"}) for t in tensors(MAX_SEED_TENSORS_PER_COMMAND)]
    job = mp.start_seed(NODE, "m1", [0], lambda stop: SeedBuild(REPO, REV, long_name, 10**9, []))
    await asyncio.sleep(0.05)
    assert job.state == "queued" and job.parts > 1
    seen = 0
    for _ in range(job.parts):
        cmds, _ = mp.on_heartbeat(NODE, report())
        body = HeartbeatResponse(state=NodeState.eligible, state_reason=None, removed=False, deployments=[],
                                 commands=cmds).model_dump_json()
        assert len(body.encode()) < 64 * 1024 - 4096
        seen += len(cmds[0].seed.tensors)
        mp.on_heartbeat(NODE, report(acks=[CommandAck(id=cmds[0].id, kind="seed", status="accepted")]))
    assert seen == MAX_SEED_TENSORS_PER_COMMAND and job.state == "delivered"


@pytest.mark.anyio
async def test_seed_parts_with_non_ascii_file_names_fit_the_frame_iroh_sends():
    """Regression: parts were measured in characters of pydantic's JSON, but transport/iroh.py frames json.dumps
    output, which escapes each non-ASCII character to 6 bytes (12 for a surrogate pair), so non-ASCII catalog file
    names made a reply over the 64 KiB frame."""
    mp = ManagedPeers()
    mp.on_heartbeat(NODE, report())
    names = ["模型" * 120 + ".gguf", "\U0001f600" * 120 + ".gguf"]
    wide = [t.model_copy(update={"file": names[i % 2]}) for i, t in enumerate(tensors(MAX_SEED_TENSORS_PER_COMMAND))]
    job = mp.start_seed(NODE, "m1", [0], lambda stop: SeedBuild(REPO, REV, wide, 10**9, []))
    await asyncio.sleep(0.05)
    assert job.state == "queued" and job.parts > 1
    seen: list[SeedTensor] = []
    for _ in range(job.parts):
        cmds, _ = mp.on_heartbeat(NODE, report())
        reply = HeartbeatResponse(state=NodeState.eligible, state_reason=None, removed=False, deployments=[],
                                  commands=cmds).model_dump(mode="json")
        framed = json.dumps(reply, separators=(",", ":"), allow_nan=False).encode("utf-8")  # as transport/iroh.py
        assert len(framed) < 64 * 1024 - 4096
        seen += cmds[0].seed.tensors
        mp.on_heartbeat(NODE, report(acks=[CommandAck(id=cmds[0].id, kind="seed", status="accepted")]))
    assert seen == wide and job.state == "delivered"


@pytest.mark.anyio
async def test_refused_seed_part_drops_the_rest_and_disk_cap_is_checked():
    mp = ManagedPeers()
    mp.on_heartbeat(NODE, report())
    n = MAX_SEED_TENSORS_PER_COMMAND * 2
    job = mp.start_seed(NODE, "m1", [0], lambda stop: SeedBuild(REPO, REV, tensors(n), 10**9, []))
    await asyncio.sleep(0.05)
    cmds, _ = mp.on_heartbeat(NODE, report())
    refused = CommandAck(id=cmds[0].id, kind="seed", status="refused", detail="not in this machine's catalog")
    cmds, _ = mp.on_heartbeat(NODE, report(acks=[refused]))
    assert cmds is None and job.state == "refused" and "catalog" in job.detail
    big = mp.start_seed(NODE, "m1", [0], lambda stop: SeedBuild(REPO, REV, tensors(1), 81 * GiB, []))
    await asyncio.sleep(0.05)
    assert big.state == "refused" and "allows" in big.detail
    no_disk = ManagedPeers()
    no_disk.on_heartbeat(NODE, report(max_disk_bytes=0))
    with pytest.raises(ManagedRefusal) as exc:
        no_disk.check_seed_allowed(NODE)
    assert exc.value.code == "seeding_not_allowed"


@pytest.mark.anyio
@pytest.mark.parametrize("how", ["forget", "loop_shutdown"])
async def test_stopping_a_seed_request_stops_the_host_hashing(tmp_path, how):
    """Regression: forget() (the peer left or rejoined) and the loop cancelling the request on shutdown removed the
    stop file at once, while the hashing thread, which a cancel cannot stop, read the rest of the model for minutes
    on the host."""
    mp = ManagedPeers()
    mp.on_heartbeat(NODE, report())
    chunks, saw_stop, ended = [0], threading.Event(), threading.Event()

    def build(stop: str | None) -> SeedBuild:
        try:
            for _ in range(100):  # like fnv1a64_file: look for the stop file between chunks
                if stop is not None and os.path.exists(stop):
                    saw_stop.set()
                    raise SeedStopped
                chunks[0] += 1
                time.sleep(0.02)
            return SeedBuild(REPO, REV, tensors(1), 10**9, [])
        finally:
            ended.set()

    job = mp.start_seed(NODE, "m1", [0], build, stop_dir=tmp_path)
    await asyncio.sleep(0.05)
    if how == "forget":
        mp.forget(NODE)
    else:
        mp._tasks[NODE].cancel()  # what asyncio.run does to the pending tasks when the gateway stops
    assert await asyncio.to_thread(ended.wait, 5)
    await asyncio.sleep(0.05)
    assert saw_stop.is_set() and chunks[0] < 100
    assert job.state == "cancelled" and not job.stop_file.exists()  # removed once the hashing thread returned


@pytest.mark.anyio
async def test_registry_join_lists_the_feature_and_restart_triggers_reverify(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    joined = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", "10.0.0.2:8000")
    assert MANAGED_PEERS_FEATURE in joined.features
    await reg.verify(joined.node_id)
    assert reg.get(joined.node_id).state == NodeState.eligible
    hb = HeartbeatRequest(worker_state=WorkerState.running, sent_at=utcnow(), managed=report())
    resp = await reg.heartbeat(joined.node_id, hb)
    assert resp.commands is None and reg.get(joined.node_id).state == NodeState.eligible
    cmd = reg.managed.request_worker(joined.node_id, ["MTL0", "CPU"], None, busy=False)
    resp = await reg.heartbeat(joined.node_id, hb)
    assert [c.id for c in resp.commands] == [cmd.id]
    done = report(worker_devices=["MTL0", "CPU"], worker_generation=1,
                  acks=[CommandAck(id=cmd.id, kind="worker", status="applied")])
    resp = await reg.heartbeat(joined.node_id, hb.model_copy(update={"managed": done}))
    view = reg.get(joined.node_id)
    assert resp.commands is None and view.contribution.devices == ["MTL0", "CPU"]
    assert view.state == NodeState.verifying and "restarted by the host" in view.state_reason
    await asyncio.sleep(0.05)
    assert reg.get(joined.node_id).state == NodeState.eligible
    await reg.leave(joined.node_id, by_owner=False)
    assert reg.managed.report(joined.node_id) is None


@pytest.mark.anyio
async def test_old_agent_heartbeat_gets_no_commands(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    joined = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", "10.0.0.2:8000")
    resp = await reg.heartbeat(joined.node_id, HeartbeatRequest(worker_state=WorkerState.running, sent_at=utcnow()))
    data = resp.model_dump(mode="json")
    assert data["commands"] is None and set(data) == {"state", "state_reason", "removed", "deployments", "commands"}


# --------------------------------------------------------------------------- owner routes and CLI


def test_peer_routes(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    with open_client(coord) as client:
        owner = {"X-Hypercompute-Owner": OWNER}
        assert client.get("/api/peers").json() == []
        assert client.post(f"/api/peers/{NODE}/worker", json={"devices": ["CPU"]}).status_code == 403
        r = client.post(f"/api/peers/{NODE}/worker", json={"devices": ["CPU"]}, headers=owner)
        assert r.status_code == 404
        code = client.post("/api/sessions", json={}, headers=owner).json()["invite"]["code"]
        joined = client.post("/api/sessions/join", json=join_request(code, worker_ip="127.0.0.1").model_dump(
            mode="json")).json()
        r = client.post(f"/api/peers/{NODE}/worker", json={"devices": ["CPU"]}, headers=owner)
        assert r.status_code == 409 and r.json()["code"] == "not_managed"
        hb = HeartbeatRequest(worker_state=WorkerState.running, sent_at=utcnow(), managed=report())
        auth = {"Authorization": f"Bearer {joined['credential']}"}
        assert client.post(f"/api/nodes/{NODE}/heartbeat", json=hb.model_dump(mode="json"),
                           headers=auth).status_code == 200
        r = client.post(f"/api/peers/{NODE}/worker", json={"devices": ["rm -rf"]}, headers=owner)
        assert r.status_code == 422
        r = client.post(f"/api/peers/{NODE}/worker", json={"devices": ["MTL0", "CPU"], "threads": 4}, headers=owner)
        assert r.status_code == 202 and r.json()["command"]["devices"] == ["MTL0", "CPU"]
        r = client.post(f"/api/nodes/{NODE}/heartbeat", json=hb.model_dump(mode="json"), headers=auth)
        assert [c["kind"] for c in r.json()["commands"]] == ["worker"]
        r = client.post(f"/api/peers/{NODE}/seed", json={"model_id": "gpt-oss-120b-mxfp4", "layers": "1-2",
                                                         "plan_id": "p1"}, headers=owner)
        assert r.status_code == 422 and r.json()["code"] == "layers_or_plan"
        r = client.post(f"/api/peers/{NODE}/seed", json={"model_id": "no-such-model", "layers": "1-2"},
                        headers=owner)
        assert r.status_code == 400
        assert client.post(f"/api/peers/{NODE}/seed/cancel", headers=owner).status_code == 202
        peers = client.get("/api/peers").json()
        assert peers[0]["managed"] and peers[0]["report"]["devices_allowed"] == ["MTL0", "CPU"]
        assert [p["kind"] for p in peers[0]["pending"]] == ["worker", "seed_cancel"]
        lines = capacity_cli.peer_lines(peers)
        assert any(NODE[:8] in line and "MTL0,CPU" in line for line in lines)


def test_capacity_cli_resolves_node_prefixes_and_posts():
    calls = []
    peers = [{"node_id": NODE, "display_name": "mac", "state": "eligible", "managed": True,
              "report": report().model_dump(mode="json"), "pending": [], "history": []}]

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "t"})
        if request.url.path == "/api/peers":
            return httpx.Response(200, json=peers)
        return httpx.Response(202, json={"command": {"seq": 3, "devices": ["MTL0", "CPU"], "threads": None}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8000")
    args = SimpleNamespace(action="worker", node="aaaa", devices="MTL0,CPU", threads=None, port=8000)
    assert capacity_cli.cmd_capacity(args, client) == 0
    assert ("POST", f"/api/peers/{NODE}/worker", b'{"devices":["MTL0","CPU"]}') in calls
    bad = SimpleNamespace(action="worker", node="zz", devices="CPU", threads=None, port=8000)
    assert capacity_cli.cmd_capacity(bad, client) == 1
    seed = SimpleNamespace(action="seed", node=NODE, model=None, layers=None, plan=None, cancel=False, port=8000)
    assert capacity_cli.cmd_capacity(seed, client) == 1


# --------------------------------------------------------------------------- unattended: crashed worker, frame size


def exiting_worker(monkeypatch, agent: Agent, exits: int) -> list[int]:
    """worker_status reports `exits` crashes, each followed by one running heartbeat; restarts are recorded."""
    statuses = []
    for _ in range(exits):
        statuses += [(WorkerState.exited, 1, None), (WorkerState.running, None, 1.0)]
    revived: list[int] = []
    monkeypatch.setattr(agent, "worker_status",
                        lambda: statuses.pop(0) if statuses else (WorkerState.running, None, 1.0))
    monkeypatch.setattr(agent, "_start_worker_checked", lambda devices: revived.append(1))
    monkeypatch.setattr(agent.sup, "stop", lambda key, grace_s=8.0: None)
    return revived


def test_managed_agent_restarts_a_crashed_worker(monkeypatch, capsys):
    agent = make_agent(consent())
    agent.credential = "cred"
    agent.host_features = [MANAGED_PEERS_FEATURE]
    revived = exiting_worker(monkeypatch, agent, 1)
    beats = []

    def handler(request: httpx.Request) -> httpx.Response:
        beats.append(json.loads(request.content)["worker_state"])
        if len(beats) >= 3:
            agent.stop_event.set()
        return reply()

    run_heartbeats(agent, handler, monkeypatch)
    assert beats[:2] == ["exited", "running"] and revived == [1] and agent.worker_generation == 1
    assert "starting it again" in capsys.readouterr().out
    agent.managed.close()


def test_crash_loop_still_stops_and_unmanaged_agent_stops_at_once(monkeypatch, capsys):
    agent = make_agent(consent())
    agent.credential = "cred"
    agent.host_features = [MANAGED_PEERS_FEATURE]
    revived = exiting_worker(monkeypatch, agent, 10)
    run_heartbeats(agent, lambda request: reply(), monkeypatch)
    assert len(revived) == join.REVIVE_LIMIT and agent.stop_event.is_set()
    assert "The local worker exited" in capsys.readouterr().out
    agent.managed.close()
    plain = make_agent()
    plain.credential = "cred"
    revived = exiting_worker(monkeypatch, plain, 1)
    run_heartbeats(plain, lambda request: reply(), monkeypatch)
    assert revived == [] and plain.stop_event.is_set()


def test_largest_report_fits_an_iroh_frame():
    from hypercompute.schemas import SeedFailure, SeedProgress

    ack = CommandAck(id="f" * 32, kind="worker", status="refused", detail="x" * 300)
    fail = SeedFailure(fnv64="0" * 16, error="e" * 200)
    seed = SeedProgress(job_id="e" * 32, model_id="m" * 128, state="failed", parts_received=9, parts=9,
                        tensors_total=2000, tensors_done=1000, tensors_present=10, tensors_failed=990,
                        bytes_total=10**12, bytes_done=10**11, failures=[fail] * 16, error="z" * 300,
                        updated_at=utcnow())
    big = report(acks=[ack] * 32, seed=seed, worker_error="w" * 300)
    hb = HeartbeatRequest(worker_state=WorkerState.running, sent_at=utcnow(), managed=big,
                          memory_pools=[MemoryPool(id="uma", kind="unified", total_bytes=16 * GiB,
                                                   free_bytes=8 * GiB, measured_at=utcnow(), source="x")])
    assert len(hb.model_dump_json().encode()) < 40 * 1024


def test_disk_errors_and_bad_reports_never_end_the_heartbeat(tmp_path, monkeypatch):
    p = peer(tmp_path, Worker(), consent(max_disk=10 * GiB), lambda: catalog())
    monkeypatch.setattr(p.runner, "submit", lambda spec, sources: (_ for _ in ()).throw(OSError("disk gone")))
    p.handle([seed_command(1)])
    a = acks(p)[f"{1:032x}"]
    assert a.status == "failed" and "disk gone" in a.detail
    agent = make_agent(consent())
    agent.host_features = [MANAGED_PEERS_FEATURE]
    monkeypatch.setattr(agent.managed, "report", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    assert agent.managed_heartbeat() == {}
    agent.managed.close()
    with pytest.raises(ValueError, match="8"):  # the parser and the default list both cap at 8
        build_consent(mac_inventory(), ["MTL0"], "MTL0,CPU,A1,A2,A3,A4,A5,A6,A7", None)


def test_host_and_agent_round_trip_over_json(tmp_path):
    """Host queue -> HeartbeatResponse JSON -> agent applies -> ManagedReport JSON -> host clears the command."""
    w = Worker()
    agent_side = peer(tmp_path, w)
    host = ManagedPeers()
    sent = agent_side.report(w.devices, w.threads, 0)
    host.on_heartbeat(NODE, ManagedReport.model_validate_json(sent.model_dump_json()))
    cmd = host.request_worker(NODE, ["MTL0", "CPU"], 2, busy=False)
    for _ in range(3):  # a lost reply is simply delivered again: the restart still happens once
        cmds, _ = host.on_heartbeat(NODE, ManagedReport.model_validate_json(sent.model_dump_json()))
        wire = HeartbeatResponse(state=NodeState.eligible, state_reason=None, removed=False, deployments=[],
                                 commands=cmds).model_dump(mode="json")
        agent_side.handle(json.loads(json.dumps(wire))["commands"])
        settle(agent_side)
    assert w.restarts == [(["MTL0", "CPU"], 2)]
    sent = agent_side.report(w.devices, w.threads, 1)
    cmds, restarted = host.on_heartbeat(NODE, ManagedReport.model_validate_json(sent.model_dump_json()))
    assert cmds is None and restarted
    assert host.snapshot([], lambda n: []) == [] and host.report(NODE).worker_devices == ["MTL0", "CPU"]
    assert [h["status"] for h in host._peers[NODE].history] == ["applied"] and cmd.kind == "worker"


def hash_file(cache, fill: bytes, size: int, age_s: float):
    """A cache file under a hash-shaped name (the content need not hash to it), last written age_s ago."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / cache_name(int.from_bytes(fill * 8, "big"))
    path.write_bytes(fill * size)
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_disk_guard_holds_the_cap_against_what_the_worker_saves(tmp_path):
    """Regression: rpc-server -c saves every tensor it is sent with no limit, so --max-disk and the free-space
    floor only bounded seeding and an unattended night of loads could fill the Mac's disk."""
    c = ManagedConsent(("MTL0", "CPU"), 50_000, 10, min_free_bytes=0)
    p = peer(tmp_path, Worker(), c)
    oldest = hash_file(p.cache_dir, b"a", 20_000, 3000)
    middle = hash_file(p.cache_dir, b"b", 20_000, 2000)
    newest = hash_file(p.cache_dir, b"c", 20_000, 1000)
    assert p.guard_disk() == 20_000
    assert not oldest.exists() and middle.exists() and newest.exists()
    assert p.guard_disk() == 0 and p.report(["MTL0"], None, 0).cache_bytes == 40_000
    p.close()
    floor = ManagedConsent(("MTL0", "CPU"), 10**9, 10, min_free_bytes=10**18)  # more free space than any disk has
    q = peer(tmp_path / "q", Worker(), floor)
    kept = hash_file(q.cache_dir, b"d", 1000, 10)
    q.runner.protected = lambda: {kept.name}  # the running seed job's own files are never evicted
    gone = hash_file(q.cache_dir, b"e", 1000, 20)
    assert q.guard_disk() == 1000 and kept.exists() and not gone.exists()
    q.close()
    no_disk = peer(tmp_path / "n", Worker(), consent(max_disk=0))
    untouched = hash_file(no_disk.cache_dir, b"f", 1000, 10)
    assert no_disk.guard_disk() == 0 and untouched.exists()
    no_disk.close()


def test_agent_guards_the_disk_even_when_the_host_lacks_managed_peers(monkeypatch):
    agent = make_agent(consent(80 * GiB))
    agent.host_features = []
    calls = []
    monkeypatch.setattr(agent.managed, "guard_disk", lambda: calls.append(1) or 0)
    assert agent.managed_heartbeat() == {} and calls == [1]
    monkeypatch.setattr(agent.managed, "guard_disk", lambda: (_ for _ in ()).throw(OSError("disk gone")))
    agent.host_features = [MANAGED_PEERS_FEATURE]
    assert "managed" in agent.managed_heartbeat()  # a failing guard never costs the heartbeat
    agent.managed.close()


def test_disk_guard_lists_the_cache_at_most_every_interval_unless_forced(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(managed_module.time, "monotonic", lambda: clock[0])
    p = peer(tmp_path, Worker(), ManagedConsent(("MTL0", "CPU"), 10_000, 10, min_free_bytes=0))
    first = hash_file(p.cache_dir, b"a", 8_000, 300)
    assert p.guard_disk() == 0 and first.exists()  # within the cap
    second = hash_file(p.cache_dir, b"b", 8_000, 200)  # rpc-server saved another tensor: over the cap now
    clock[0] += managed_module.GUARD_INTERVAL_S - 1
    assert p.guard_disk() == 0 and first.exists()  # bounded cost: no pass before the interval is up
    assert p.guard_disk(force=True) == 8_000 and not first.exists() and second.exists()
    third = hash_file(p.cache_dir, b"c", 8_000, 100)
    clock[0] += 1
    assert p.guard_disk() == 0 and second.exists()  # the forced pass restarted the interval
    clock[0] += managed_module.GUARD_INTERVAL_S
    assert p.guard_disk() == 8_000 and not second.exists() and third.exists()
    p.close()


def test_disk_guard_evicts_by_last_use_so_a_tensor_every_load_reads_stays(tmp_path):
    p = peer(tmp_path, Worker(), ManagedConsent(("MTL0", "CPU"), 2_500, 10, min_free_bytes=0))
    hot = hash_file(p.cache_dir, b"h", 1_000, 3000)
    now = time.time()
    os.utime(hot, (now - 5, now - 3000))  # written long ago, read by rpc-server at the last load
    cold = hash_file(p.cache_dir, b"c", 1_000, 2000)
    noatime = hash_file(p.cache_dir, b"n", 1_000, 1000)
    os.utime(noatime, (now - 9000, now - 1000))  # a noatime volume: atime stays at creation, mtime decides
    assert p.guard_disk() == 1_000 and hot.exists() and not cold.exists() and noatime.exists()
    p.close()


def test_with_the_hcpeer_cap_the_guard_makes_room_below_it_from_files_the_worker_has_not_used(tmp_path):
    """Regression: hcpeer v4 stops saving at HC_PEER_CACHE_MAX_GIB and never evicts, so the guard, which acted only
    past --max-disk, never ran: stale files of earlier models stayed, and every later save of a load was copied into
    the Mac's RAM while the host's plan counted those files as room."""
    w = Worker()
    p = peer(tmp_path, w, ManagedConsent(("MTL0", "CPU"), 80_000, 10, min_free_bytes=0))
    stale = [hash_file(p.cache_dir, bytes([i + 1]), 10_000, 5000 - i) for i in range(7)]  # earlier models
    older = hash_file(p.cache_dir, b"o", 9_000, 4000)  # 79_000 in all, under --max-disk
    p.worker_started(hcpeer_capped=False)  # stock or v3 saves past the cap, so the guard acts there
    assert p.guard_disk(force=True) == 0
    p.worker_started(hcpeer_capped=True)
    w.busy = True
    assert p.guard_disk(force=True) == 10_000  # 80_000 // 8 of room below the cap, from the least recently used
    assert not stale[0].exists() and stale[1].exists() and older.exists()
    mapped = hash_file(p.cache_dir, b"m", 70_000, 0)  # saved and mapped by this worker: 139_000 now
    assert p.guard_disk(force=True) == 69_000  # every file it has not used goes; one it may have mapped never does
    assert mapped.exists() and not older.exists() and not any(f.exists() for f in stale)
    p.close()


def test_with_the_hcpeer_cap_a_worker_seen_idle_maps_nothing_older_than_the_margin(tmp_path, monkeypatch):
    w = Worker()
    p = peer(tmp_path, w, ManagedConsent(("MTL0", "CPU"), 80_000, 10, min_free_bytes=0))
    started = time.time() - 3000
    with monkeypatch.context() as m:
        m.setattr(managed_module.time, "time", lambda: started)
        p.worker_started(hcpeer_capped=True)
    unloaded = hash_file(p.cache_dir, b"u", 50_000, 1000)  # a model this worker loaded earlier
    recent = hash_file(p.cache_dir, b"r", 40_000, 10)  # maybe the first tensors of a load the agent has not seen
    w.busy = True
    assert p.guard_disk(force=True) == 0 and unloaded.exists()  # busy since it started: it may still be mapped
    w.busy = False
    assert p.guard_disk(force=True) == 50_000 and not unloaded.exists() and recent.exists()
    p.close()


def test_disk_guard_never_raises_on_a_cache_it_cannot_list_or_measure(tmp_path, monkeypatch):
    p = peer(tmp_path, Worker(), ManagedConsent(("MTL0", "CPU"), 1_000, 10, min_free_bytes=0))
    kept = hash_file(p.cache_dir, b"a", 5_000, 100)

    def denied(*args, **kwargs):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(managed_module, "evict_cache", denied)
    assert p.guard_disk() == 0 and kept.exists()  # logged; the next pass tries again
    monkeypatch.setattr(managed_module, "cache_usage", lambda cache_dir: denied())
    assert p.guard_disk(force=True) == 0  # an unmeasurable disk evicts nothing
    p.close()


def test_a_new_seed_job_first_makes_room_and_keeps_its_own_tensors(tmp_path, monkeypatch):
    p = peer(tmp_path, Worker(), ManagedConsent(("MTL0", "CPU"), 50_000, 10, min_free_bytes=0), lambda: catalog())
    p.runner._workers = 1
    p.runner._executor_factory = lambda n: ThreadPoolExecutor(n)
    p.runner._fetch = lambda task: SimpleNamespace(fnv64=task.fnv64, status="failed", downloaded=0, error="offline")
    mine = hash_file(p.cache_dir, b"\x11", 1_000, 5000)  # the job's tensor (fnv64 1111...), least recently used
    other = hash_file(p.cache_dir, b"o", 60_000, 100)  # what rpc-server saved during earlier loads
    p.handle([seed_command(1, parts=2)])
    assert acks(p)[f"{1:032x}"].status == "accepted" and mine.exists() and not other.exists()
    forced: list[dict] = []
    monkeypatch.setattr(p, "guard_disk", lambda **kw: forced.append(kw) or 0)
    p.handle([seed_command(2, parts=2, part=2)])
    assert acks(p)[f"{2:032x}"].status == "accepted" and forced == []  # the same job: no second pass
    p.handle([seed_command(3, job="e" * 32)])
    assert forced == [{"force": True, "keep": {"1" * 16}}]  # a new job: a pass first
    p.close()


def test_seed_layers_come_from_a_stored_plan_of_any_age(store, events, tmp_path, monkeypatch):
    """Seeding reads a plan's layer split through DeploymentManager.find_plan, without the 10-minute limit that only
    a deployment needs (stored_plan keeps it)."""
    from test_split_order import EP5, cpu, gpu, mac, shape14

    coord = make_coord(store, events, tmp_path, monkeypatch)
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected), shape14(), [gpu(6)], [mac(4)],
                     32 * GiB, cpu(5))
    old = plan.model_copy(update={"created_at": utcnow() - timedelta(hours=2)})
    store.execute("INSERT INTO plans (id, created_at, doc) VALUES (?,?,?)",
                  (old.id, old.created_at.isoformat(), old.model_dump_json()))
    with pytest.raises(DeploymentError, match="older than 10 minutes"):
        coord.deployments.stored_plan(old.id)
    assert coord.deployments.find_plan(old.id).model_dump(mode="json") == old.model_dump(mode="json")
    assert coord.deployments.find_plan("missing") is None
    peer_layers = sorted(n for a in old.assignments if a.endpoint == EP5 for n in a.layers)
    assert peer_layers and plan_layers(load_plan(coord.deployments, old.id), f"mac-{EP5}") == peer_layers
    with pytest.raises(ManagedRefusal) as missing:
        load_plan(coord.deployments, "missing")
    assert missing.value.status == 404 and missing.value.code == "plan_not_found"


def test_revived_worker_does_not_reuse_a_cache_file_cut_short_by_the_crash(monkeypatch):
    """Regression: rpc-server writes its cache file in place with no check; a worker killed mid-write leaves a short
    file under the tensor's hash name, and the next load would take it as the whole tensor."""
    agent = make_agent(consent(80 * GiB))
    agent.host_features = [MANAGED_PEERS_FEATURE]
    cache = config.paths().rpc_cache / "rpc"
    before = hash_file(cache, b"o", 1000, 3600)
    cut = hash_file(cache, b"x", 10, 2)
    monkeypatch.setattr(agent, "_start_worker_checked", lambda devices: None)
    monkeypatch.setattr(agent.sup, "stop", lambda key, grace_s=8.0: None)
    assert agent._revive_worker(9) is True
    assert before.exists() and not cut.exists()
    agent.managed.close()


def test_unattended_iroh_peer_waits_longer_for_the_host(monkeypatch):
    plain = make_agent()
    managed = make_agent(consent())
    assert plain.unreachable_limit_s() == managed.unreachable_limit_s() == config.HOST_UNREACHABLE_STOP_S  # LAN
    managed.peer_client = SimpleNamespace()  # Iroh: the worker is reachable only through the paired host
    assert managed.unreachable_limit_s() == join.MANAGED_IROH_UNREACHABLE_STOP_S == 7200
    monkeypatch.setattr(join.time, "monotonic", lambda: 1000.0)
    assert not managed._host_gone_too_long(1000.0 - 600)  # 10 minutes: an unmanaged agent would have stopped
    assert managed._host_gone_too_long(1000.0 - 7300)
    managed.managed.close()
