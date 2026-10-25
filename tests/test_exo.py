"""EXO engine: installer, Mac sidecar, planning, preview/readiness parsing, SSE normalisation and the HTTP lifecycle.

No EXO install here: every EXO response comes from source-shaped fixtures (tests/fixtures/exo_*, field names from
exo@bc412e2's pydantic models). Hardware validation (real Macs, 2+ for a ring) is NOT RUN by these tests.
"""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from factories import contribution, make_inventory
from hypercompute import cli
from hypercompute.adapters import exo as ex
from hypercompute.adapters.base import DeploymentSpec, EngineAdapterError, InferenceRequest
from hypercompute.agent import exo_sidecar, join
from hypercompute.agent.join import Agent
from hypercompute.agent.supervisor import StopOutcome
from hypercompute.installers import exo as inst
from hypercompute.schemas import (
    ExtraEngineInfo,
    ModelArtifact,
    NodeState,
    NodeView,
    PlanRequest,
    PlanStrategy,
    WorkerEndpoint,
    WorkerState,
    utcnow,
)
from hypercompute.units import GiB

FIX = Path(__file__).parent / "fixtures"
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
A, B = "9b1f0c2e7d4a4e1fa3c5b7d9e1f2a3b4", "4c2d8e6f1a3b5c7d9e0f1a2b3c4d5e6"
IID = "5d0c6a3e-8f1b-4a52-9c3d-2e7f4b1a6c90"
RA, RB = "a1e5c3d2-7b4f-4e8a-9d1c-3f6b2a8e5c74", "b7f2d9e1-3c6a-4d5b-8e2f-1a9c7b3d6e05"
EP_A, EP_B = "192.168.1.31:52415", "192.168.1.32:52415"


def fixture(name: str) -> object:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def node(node_id: str, system: str = "macos", exo: str | None = "running", endpoint: str = EP_A,
         state: NodeState = NodeState.eligible, free_gib: float = 12, max_gib: float = 10,
         is_local: bool = False, host: str | None = None) -> NodeView:
    """A peer whose verified worker host is `host`, else the host of its EXO endpoint (as a real agent reports)."""
    inv = make_inventory(system, ram_total=16 * GiB, ram_free=int(free_gib * GiB))
    extra = [] if exo is None else [ExtraEngineInfo(engine="exo", installed=True, status=exo,  # type: ignore[arg-type]
                                                    endpoint=endpoint if exo == "running" else None)]
    worker = WorkerEndpoint(host=host or endpoint.rsplit(":", 1)[0], port=47052)
    return NodeView(id=node_id, display_name=f"{system}-{node_id[:4]}", is_local=is_local, session_id="s",
                    state=state, state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(),
                    heartbeat_age_s=0.0, inventory=inv.model_copy(update={"extra_engines": extra}),
                    contribution=contribution(["MTL0"], max_gib, 2), worker=worker,
                    worker_state=WorkerState.running, verification=None)


def windows_host() -> NodeView:
    return node("local", system="windows", exo=None, is_local=True)


def mlx_artifact(size: int | None = 729808896, fmt: str = "mlx") -> ModelArtifact:
    return ModelArtifact(id="llama-3.2-1b-mlx", display_name="Llama 3.2 1B (MLX 4-bit)", family="llama",
                         params_b=1.2, quantization="4bit", format=fmt, engines=["exo"],  # type: ignore[arg-type]
                         logical_id="llama-3.2-1b-instruct", size_bytes=size, repo=MODEL, revision="0" * 40,
                         license="llama3.2", files=[])


# ----------------------------------------------------------------------------- planning


def test_plan_uses_macs_running_exo_and_stays_off_llama_rpc_fields():
    macs = [node(A, endpoint=EP_A), node(B, endpoint=EP_B, free_gib=8)]
    p = ex.plan(PlanRequest(model_id="llama-3.2-1b-mlx"), mlx_artifact(), windows_host(), macs)
    assert p.feasible and p.engine == "exo" and p.tier == "gpu" and p.fit_label == "estimated_fit"
    assert {a.node_id for a in p.assignments} == {A, B}  # all_selected: every EXO Mac
    assert all(a.endpoint is None and a.device == "MLX" and a.pool_id == "uma" for a in p.assignments)
    assert p.device_list == [] and p.rpc_endpoints == [] and p.tensor_split == []
    opts = p.engine_options
    assert opts["api"] == f"http://{EP_A}" and opts["exo_model_id"] == MODEL and opts["version"] == inst.EXO_PIN
    assert [n["endpoint"] for n in opts["nodes"]] == [EP_A, EP_B]  # largest budget first = the API Mac
    assert sum(a.weight_bytes for a in p.assignments) <= 729808896
    assert any("main" in w for w in p.warnings)  # EXO does not pin HF revisions


def test_plan_fastest_fit_uses_one_mac_when_it_fits():
    macs = [node(A, endpoint=EP_A), node(B, endpoint=EP_B, free_gib=8)]
    p = ex.plan(PlanRequest(model_id="m", strategy=PlanStrategy.fastest_fit), mlx_artifact(), windows_host(), macs)
    assert p.feasible and [a.node_id for a in p.assignments] == [A]
    assert p.assignments[0].total_bytes_est == 729808896 + ex.MLX_OVERHEAD


def test_plan_infeasible_without_exo_macs():
    p = ex.plan(PlanRequest(model_id="m"), mlx_artifact(), windows_host(), [node(A, exo="installed")])
    assert not p.feasible and p.fit_label == "does_not_fit"
    assert any("no Mac in the session runs EXO" in r for r in p.reasons)
    assert any("EXO is installed" in w for w in p.warnings)


def test_plan_infeasible_when_a_selected_node_is_not_a_mac():
    pc = node("c" * 32, system="windows", exo=None)
    p = ex.plan(PlanRequest(model_id="m", node_ids=[A, pc.id]), mlx_artifact(), windows_host(), [node(A), pc])
    assert not p.feasible and any("not a Mac (windows)" in r for r in p.reasons)
    # not explicitly selected: the PC is skipped with a warning and the Mac serves alone
    q = ex.plan(PlanRequest(model_id="m"), mlx_artifact(), windows_host(), [node(A), pc])
    assert q.feasible and [a.node_id for a in q.assignments] == [A] and any("skipped" in w for w in q.warnings)


def test_plan_rejects_wrong_format_missing_size_and_ineligible_nodes():
    assert any("MLX models only" in r for r in
               ex.plan(PlanRequest(model_id="m"), mlx_artifact(fmt="hf"), windows_host(), [node(A)]).reasons)
    assert any("size_bytes" in r for r in
               ex.plan(PlanRequest(model_id="m"), mlx_artifact(size=None), windows_host(), [node(A)]).reasons)
    stale = node(A, state=NodeState.stale)
    assert not ex.plan(PlanRequest(model_id="m", node_ids=[A]), mlx_artifact(), windows_host(), [stale]).feasible


def test_plan_reports_memory_shortfall():
    p = ex.plan(PlanRequest(model_id="m"), mlx_artifact(size=40 * GiB), windows_host(), [node(A), node(B)])
    assert not p.feasible and any("they offer" in r for r in p.reasons)


def test_local_only_needs_the_host_to_be_an_exo_mac():
    p = ex.plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only), mlx_artifact(), windows_host(), [node(A)])
    assert not p.feasible and any("not a Mac" in r for r in p.reasons)
    mac_host = node("local", is_local=True, endpoint="127.0.0.1:52415")
    q = ex.plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only), mlx_artifact(), mac_host, [node(A)])
    assert q.feasible and [a.node_id for a in q.assignments] == ["local"]


# ----------------------------------------------------------------------------- preview / readiness / placement


def test_choose_preview_picks_the_ring_pipeline_on_exactly_the_planned_macs():
    preview, errors = ex.choose_preview(fixture("exo_previews_llama1b_two_macs.json"), {A, B})
    assert errors == [] and preview is not None and preview["sharding"] == "Pipeline"
    info = ex.InstanceInfo.parse(preview["instance"])
    assert info.instance_id == IID and info.model_id == MODEL and info.kind == "MlxRingInstance"
    assert {(s.node_id, s.start, s.end) for s in info.shards} == {(A, 0, 9), (B, 9, 16)}
    assert next(s for s in info.shards if s.node_id == A).weight_bytes == 729808896 * 9 // 16


def test_choose_preview_reports_exo_errors_when_nothing_matches():
    preview, errors = ex.choose_preview(fixture("exo_previews_llama1b_two_macs.json"), {A})
    assert preview is None
    assert "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available" in errors
    assert any("other sets of Macs" in e for e in errors)
    assert ex.choose_preview({"previews": []}, {A})[1][0].startswith("EXO sees no nodes")


def _info() -> ex.InstanceInfo:
    return ex.InstanceInfo.parse(fixture("exo_state_instances.json")[IID])  # type: ignore[index]


def test_readiness_tracks_download_load_ready_and_failure():
    info, names = _info(), {A: "mac-a", B: "mac-b"}
    loading = ex.readiness(info, True, True, 5.0, fixture("exo_state_runners_loading.json"),  # type: ignore[arg-type]
                           fixture("exo_state_downloads.json"), names)  # type: ignore[arg-type]
    assert loading.state == "waiting"
    assert loading.text == "mac-a: loading layers 4/9; mac-b: downloading 348 MiB of 696 MiB (50%)"
    ready = ex.readiness(info, True, True, 9.0, fixture("exo_state_runners_ready.json"), None, names)  # type: ignore[arg-type]
    assert ready.state == "ready"
    failed = ex.readiness(info, True, True, 9.0, fixture("exo_state_runners_failed.json"), None, names)  # type: ignore[arg-type]
    assert failed.state == "failed" and "ring transport error" in failed.text and failed.text.startswith("mac-a")


def test_readiness_when_the_instance_is_missing():
    info = _info()
    assert ex.readiness(info, False, False, 1.0, {}, None, {}).state == "waiting"
    assert ex.readiness(info, False, False, ex.ACK_TIMEOUT_S + 1, {}, None, {}).state == "failed"
    assert ex.readiness(info, False, True, 1.0, {}, None, {}).text == ex.LOST


def test_download_failure_fails_the_load():
    downloads = {B: [{"DownloadFailed": {"nodeId": B, "shardMetadata": {"PipelineShardMetadata": {
        "modelCard": {"modelId": MODEL}}}, "errorMessage": "disk full"}}]}
    r = ex.readiness(_info(), True, True, 1.0, {RA: {"RunnerReady": {}}, RB: {"RunnerIdle": {}}}, downloads, {})
    assert r.state == "failed" and "disk full" in r.text


def test_observed_placement_maps_runners_to_planned_nodes():
    hc_of = {A: "hc-a", B: "hc-b"}
    report = ex.observed_placement(_info(), fixture("exo_state_runners_ready.json"), {A: "mac-a", B: "mac-b"},  # type: ignore[arg-type]
                                   hc_of, {"hc-a", "hc-b"})
    assert report.verified and report.observed is not None and report.observed.offloaded == "16/16"
    assert [(d.name, d.layers_est) for d in report.observed.devices] == [("EXO mac-a", 9), ("EXO mac-b", 7)]
    loading = ex.observed_placement(_info(), fixture("exo_state_runners_loading.json"), {}, hc_of,  # type: ignore[arg-type]
                                    {"hc-a", "hc-b", "hc-c"})
    assert not loading.verified and "hc-c"[:8] in loading.detail and "RunnerIdle" in loading.detail


# ----------------------------------------------------------------------------- SSE normalisation


def _feed_all(norm: ex.ExoStreamNormalizer, raw: bytes, step: int) -> list:
    events = []
    for i in range(0, len(raw), step):
        events += norm.feed(raw[i:i + step])
    return events + norm.flush()


@pytest.mark.parametrize("crlf", [False, True])
@pytest.mark.parametrize("step", [1, 2, 7, 4096])
def test_stream_normalises_to_openai_chunks_with_usage_and_timings(step, crlf):
    raw = (FIX / "exo_chat_stream.sse").read_bytes()
    raw = raw.replace(b"\n", b"\r\n") if crlf else raw  # step 1 and 2 split every CRLF across two chunks
    norm = ex.ExoStreamNormalizer("llama-3.2-1b-mlx")
    events = _feed_all(norm, raw, step)
    assert events[0].raw == b": keep-alive\n\n" and events[0].payload is None
    data = [e.payload for e in events if e.payload]
    assert [d["choices"][0]["delta"].get("content") for d in data] == ["Hel", "lo", "!"]
    assert all(d["object"] == "chat.completion.chunk" and d["model"] == "llama-3.2-1b-mlx" for d in data)
    final = data[-1]
    assert final["choices"][0]["finish_reason"] == "stop" and final["usage"]["completion_tokens"] == 3
    assert final["timings"] == {"prompt_n": 12, "prompt_ms": 50.0, "prompt_per_second": 240.0, "predicted_n": 3,
                                "predicted_ms": 50.0, "predicted_per_second": 60.0, "cache_n": 0}
    assert events[-1].done and events[-1].raw == b"data: [DONE]\n\n"
    assert norm.command_id == "7c1e9b3a-4d2f-4a6e-8b5c-0d9f2e7a1c38"
    assert all(e.raw.startswith(b"data: ") for e in events[1:])  # prefill_progress / generation_stats dropped


def test_stream_error_object_becomes_an_adapter_error():
    norm = ex.ExoStreamNormalizer("alias")
    raw = (FIX / "exo_chat_stream_error.sse").read_bytes()
    with pytest.raises(EngineAdapterError, match="stopped before it finished"):
        _feed_all(norm, raw, 4096)


def test_stream_without_done_still_delivers_the_finish_chunk():
    norm = ex.ExoStreamNormalizer("alias")
    events = norm.feed(b'data: {"id":"x","object":"chat.completion","choices":[{"index":0,"delta":{},'
                       b'"finish_reason":"length"}]}\n\n')
    assert events == [] and norm.flush()[0].payload["choices"][0]["finish_reason"] == "length"  # type: ignore[index]


# ----------------------------------------------------------------------------- HTTP lifecycle against a fake EXO


class Chunks(httpx.AsyncByteStream):
    """A streamed body cut into small network fragments."""

    def __init__(self, data: bytes, step: int = 13) -> None:
        self.data, self.step = data, step

    async def __aiter__(self):
        for i in range(0, len(self.data), self.step):
            yield self.data[i:i + self.step]


class FakeExo:
    """Answers like EXO's API for one 2-Mac Llama 1B deployment; `ticks` advance the instance's lifecycle."""

    def __init__(self) -> None:
        self.ticks = 0
        self.deleted = False
        self.posted: dict | None = None
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.unreachable = False
        self.post_times_out = False
        self.ignore_delete = False  # EXO accepts DELETE but keeps the instance
        self.override: dict[str, Callable[[httpx.Request], httpx.Response]] = {}  # path -> custom answer

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("no route to host", request=request)
        path, host = request.url.path, request.url.host
        self.calls.append(f"{request.method} {path}")
        if path in self.override:
            return self.override[path](request)
        if path == "/node_id":
            return httpx.Response(200, json={"192.168.1.31": A, "192.168.1.32": B}[host])
        if path == "/state/instances":
            live = self.posted and not self.deleted
            return httpx.Response(200, json={IID: fixture("exo_state_instances.json")[IID]} if live  # type: ignore[index]
                                  else {})
        if path == "/instance/previews":
            assert request.url.params.get_list("node_ids") == [A, B] and request.url.params["model_id"] == MODEL
            return httpx.Response(200, json=fixture("exo_previews_llama1b_two_macs.json"))
        if path == "/instance" and request.method == "POST":
            if self.post_times_out:
                raise httpx.ReadTimeout("EXO is electing a master", request=request)
            self.posted = json.loads(request.content)
            return httpx.Response(200, json={"message": "Command received.", "command_id": "c1", "model_card": {}})
        return self._instance_and_state(request, path)

    def _instance_and_state(self, request: httpx.Request, path: str) -> httpx.Response:
        if path == f"/state/instances/{IID}":
            self.ticks += 1
            ok = self.ticks > 1 and not self.deleted
            return httpx.Response(200 if ok else 404, json={} if ok else {"error": {"message": "unable to find"}})
        if path == "/state/runners":
            return httpx.Response(200, json=fixture("exo_state_runners_loading.json" if self.ticks < 3
                                                    else "exo_state_runners_ready.json"))
        if path == "/state/downloads":
            return httpx.Response(200, json=fixture("exo_state_downloads.json"))
        if path == f"/instance/{IID}":
            if request.method == "DELETE":
                if self.deleted:
                    return httpx.Response(404, json={"error": {"message": "Instance not found"}})
                self.deleted = not self.ignore_delete
                return httpx.Response(200, json={"message": "Command received.", "command_id": "c2",
                                                 "instance_id": IID})
            return httpx.Response(404 if self.deleted else 200, json={})
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["model"] == MODEL and body["max_tokens"] == 1024
            if body["stream"]:
                return httpx.Response(200, stream=Chunks((FIX / "exo_chat_stream.sse").read_bytes()),
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=fixture("exo_chat_completion.json"))
        if path.startswith("/v1/cancel/"):
            self.cancelled.append(path.rsplit("/", 1)[1])
            return httpx.Response(200, json={"message": "ok", "command_id": "x"})
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(ex, "POLL_S", 0.001)
    monkeypatch.setattr(ex, "WATCH_S", 0.001)
    monkeypatch.setattr(ex, "DELETE_CONFIRM_S", 1.0)


def _spec(tmp_path: Path) -> DeploymentSpec:
    macs = [node(A, endpoint=EP_A), node(B, endpoint=EP_B, free_gib=8)]
    p = ex.plan(PlanRequest(model_id="llama-3.2-1b-mlx"), mlx_artifact(), windows_host(), macs)
    return DeploymentSpec(deployment_id="d" * 32, plan=p, model_path=tmp_path, alias="llama-3.2-1b-mlx",
                          artifact=mlx_artifact())


def _adapter(tmp_path: Path, fake: FakeExo) -> ex.ExoAdapter:
    return ex.ExoAdapter(tmp_path / "logs", client=httpx.AsyncClient(transport=httpx.MockTransport(fake)))


@pytest.mark.anyio
async def test_full_lifecycle_against_fake_exo(tmp_path, fast):
    fake = FakeExo()
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    validation = await adapter.validate(spec)
    assert validation.ok and validation.rpc_device_map == {}
    handle = await adapter.prepare(spec)
    assert fake.posted == {"instance": fixture("exo_previews_llama1b_two_macs.json")["previews"][0]["instance"]}  # type: ignore[index]
    assert handle.process_key == f"exo:{IID}" and handle.pid == 0 and handle.base_url == f"http://{EP_A}"
    progress: list[str] = []
    await adapter.wait_ready(handle, timeout_s=10, on_progress=progress.append)
    assert progress[0] == "waiting for EXO to place the instance" and "loading layers 4/9" in progress[1]
    assert adapter.process_alive(handle)
    report = await adapter.observe(handle, spec, validation)
    assert report.verified, report.detail
    events = [e async for e in adapter.stream(handle, InferenceRequest("r1", {"messages": []}, True))]
    assert events[-1].done and events[-2].payload["usage"]["total_tokens"] == 15  # type: ignore[index]
    done = await adapter.complete(handle, InferenceRequest("r2", {"messages": []}, False))
    assert done.status_code == 200 and done.body["model"] == "llama-3.2-1b-mlx"
    stopped = await adapter.stop(handle)
    assert stopped.clean and "deleted" in stopped.detail and not adapter.process_alive(handle)
    assert "created EXO instance" in adapter.read_log(handle)
    await adapter.aclose()


@pytest.mark.anyio
async def test_validate_refuses_a_leftover_instance_and_unreachable_macs(tmp_path, fast):
    fake = FakeExo()
    fake.posted = {"instance": "earlier"}  # /state/instances now lists IID for the model
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    result = await adapter.validate(spec)
    assert not result.ok and any(IID in p and "dashboard" in p for p in result.problems)
    fake.unreachable = True
    result = await adapter.validate(spec)
    assert not result.ok and "join --exo" in result.problems[0]
    with pytest.raises(EngineAdapterError, match="previewed"):
        await adapter.prepare(spec)
    await adapter.aclose()


@pytest.mark.anyio
async def test_watch_marks_the_deployment_lost_when_exo_deletes_the_instance(tmp_path, fast):
    fake = FakeExo()
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    await adapter.validate(spec)
    handle = await adapter.prepare(spec)
    await adapter.wait_ready(handle, timeout_s=10)
    fake.deleted = True  # a Mac went silent: EXO dropped the instance
    for _ in range(200):
        if not adapter.process_alive(handle):
            break
        await __import__("asyncio").sleep(0.005)
    assert not adapter.process_alive(handle) and ex.LOST in adapter.read_log(handle)
    stopped = await adapter.stop(handle)
    assert stopped.clean and "already gone" in stopped.detail
    await adapter.aclose()


@pytest.mark.anyio
async def test_prepare_timeout_deletes_the_instance_exo_may_still_create(tmp_path, fast):
    fake = FakeExo()
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    await adapter.validate(spec)
    fake.post_times_out = True
    with pytest.raises(EngineAdapterError, match="did not answer"):
        await adapter.prepare(spec)
    assert f"DELETE /instance/{IID}" in fake.calls
    await adapter.aclose()


@pytest.mark.anyio
async def test_stop_when_exo_is_unreachable_is_not_clean(tmp_path, fast):
    fake = FakeExo()
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    await adapter.validate(spec)
    handle = await adapter.prepare(spec)
    fake.unreachable = True
    stopped = await adapter.stop(handle)
    assert not stopped.clean and "dashboard" in stopped.detail
    await adapter.aclose()


@pytest.mark.anyio
async def test_cancel_posts_exo_cancel_and_closes_the_stream(tmp_path, fast):
    fake = FakeExo()
    adapter, spec = _adapter(tmp_path, fake), _spec(tmp_path)
    await adapter.validate(spec)
    handle = await adapter.prepare(spec)
    await adapter.wait_ready(handle, timeout_s=10)
    gen = adapter.stream(handle, InferenceRequest("r9", {"messages": []}, True))
    await gen.__anext__()  # keep-alive
    await gen.__anext__()  # first data chunk: EXO's command id is now known
    result = await adapter.cancel("r9")
    assert result.cancelled and fake.cancelled == ["7c1e9b3a-4d2f-4a6e-8b5c-0d9f2e7a1c38"]
    await gen.aclose()
    assert not (await adapter.cancel("r9")).cancelled
    await adapter.stop(handle)
    await adapter.aclose()


def test_module_status_and_build(tmp_path):
    class Nodes:
        def views(self):
            return [windows_host(), node(A)]

    ctx = Namespace(logs_dir=tmp_path, nodes=Nodes())
    info = ex.status(ctx)  # type: ignore[arg-type]
    assert info.status == "running" and info.endpoint == EP_A
    assert ex.status(Namespace(logs_dir=tmp_path, nodes=None)).status == "installed"  # type: ignore[arg-type]
    assert isinstance(ex.build(ctx), ex.ExoAdapter)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- installer


def test_unsupported_reasons():
    assert inst.unsupported_reason("darwin", "arm64") is None
    assert "Apple Silicon" in (inst.unsupported_reason("darwin", "x86_64") or "")
    assert "#2359" in (inst.unsupported_reason("linux", "x86_64") or "")
    assert "Windows" in (inst.unsupported_reason("win32", "AMD64") or "")


def test_status_reads_the_marker(monkeypatch, tmp_path):
    from hypercompute.config import Paths

    paths = Paths(home=tmp_path)
    monkeypatch.setattr(inst, "unsupported_reason", lambda *a: None)
    assert inst.status(paths).status == "not_installed"
    inst.engine_dir(paths).mkdir(parents=True)
    inst.marker_path(paths).write_text(json.dumps({"commit": "f" * 40}), encoding="utf-8")
    assert "--force" in (inst.status(paths).detail or "")
    inst.marker_path(paths).write_text(json.dumps({"commit": inst.EXO_PIN}), encoding="utf-8")
    assert not inst.status(paths).installed
    inst.exo_binary(paths).parent.mkdir(parents=True)
    inst.exo_binary(paths).write_text("#!", encoding="utf-8")
    st = inst.status(paths)
    assert st.installed and st.status == "installed" and st.version == inst.EXO_PIN
    monkeypatch.setattr(inst, "unsupported_reason", lambda *a: "EXO cannot run on Windows")
    assert inst.status(paths).status == "unsupported"


GOOD = {("xcode-select", "-p"): (0, "/Applications/Xcode.app/Contents/Developer"),
        ("xcodebuild", "-version"): (0, "Xcode 26.0\nBuild version 17A324"), ("git", "--version"): (0, "git 2.50.1"),
        ("rustc", "--version"): (0, "rustc 1.90.0 (1159e78c4 2025-09-14)"), ("cargo", "--version"): (0, "cargo 1.90.0"),
        ("node", "--version"): (0, "v22.11.0"), ("npm", "--version"): (0, "10.9.0"),
        ("uv", "--version"): (0, "uv 0.9.2 (Homebrew 2025-10-10)"),
        ("uv", "python", "find", "3.13"): (0, "/Users/f/.local/share/uv/python/cpython-3.13.7/bin/python3.13")}


def fake_runner(overrides: dict | None = None):
    table = {**GOOD, **(overrides or {})}

    def run(argv):
        key = tuple(argv)
        if key in table:
            return table[key]
        if argv[0] == "git" and "rev-parse" in argv:
            return 0, inst.EXO_PIN
        if argv[0].endswith("exo") and argv[1:] == ["--version"]:
            return 0, f"exo {inst.EXO_VERSION}"
        raise AssertionError(f"unexpected command {argv}")

    return run


def test_prerequisite_checks_report_exact_fixes():
    bad = fake_runner({("xcode-select", "-p"): (0, "/Library/Developer/CommandLineTools"),
                       ("node", "--version"): (0, "v16.20.0"),
                       ("rustc", "--version"): (127, "rustc: command not found"),
                       ("uv", "python", "find", "3.13"): (2, "")})
    checks = {c.name: c for c in inst.check_prerequisites(bad)}
    assert not checks["xcode"].ok and "xcode-select -s" in (checks["xcode"].fix or "")
    assert not checks["node"].ok and "older than 18.0.0" in checks["node"].detail
    assert not checks["rustc"].ok and "rustup" in (checks["rustc"].fix or "")
    assert not checks["python"].ok and checks["uv"].ok
    assert all(c.ok for c in inst.check_prerequisites(fake_runner()))


def test_install_clones_the_pin_builds_and_writes_the_marker(monkeypatch, tmp_path):
    from hypercompute.config import Paths

    paths = Paths(home=tmp_path).ensure()
    monkeypatch.setattr(inst, "unsupported_reason", lambda *a: None)
    streamed: list[list[str]] = []

    def stream(argv, cwd):
        streamed.append(list(argv))
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1]).mkdir(parents=True)
        if argv[:2] == ["uv", "sync"]:
            inst.exo_binary(paths).parent.mkdir(parents=True)
            inst.exo_binary(paths).write_text("#!", encoding="utf-8")
        return 0

    said: list[str] = []
    assert inst.install(paths, yes=True, force=False, say=said.append, run=fake_runner(), stream=stream) == 0
    assert streamed[0] == ["git", "clone", "--filter=blob:none", "--no-checkout", inst.EXO_REPO,
                           str(inst.src_dir(paths).with_name("src.partial"))]
    assert streamed[1][-3:] == ["checkout", "--detach", inst.EXO_PIN]
    assert streamed[2:] == [["npm", "ci"], ["npm", "run", "build"], ["uv", "sync", "--locked", "--extra", "mlx"]]
    assert json.loads(inst.marker_path(paths).read_text(encoding="utf-8"))["commit"] == inst.EXO_PIN
    assert inst.status(paths).installed
    assert inst.install(paths, yes=True, force=False, say=said.append, run=fake_runner(), stream=stream) == 0
    assert "already installed" in said[-1]


def test_install_stops_on_missing_prerequisites_and_unsupported_hosts(monkeypatch, tmp_path):
    from hypercompute.config import Paths

    paths = Paths(home=tmp_path).ensure()
    said: list[str] = []
    monkeypatch.setattr(inst, "unsupported_reason", lambda *a: "EXO cannot run on Windows")
    assert inst.install(paths, yes=True, force=False, say=said.append) == 2
    monkeypatch.setattr(inst, "unsupported_reason", lambda *a: None)
    bad = fake_runner({("uv", "--version"): (0, "uv 0.7.1")})
    assert inst.install(paths, yes=True, force=False, say=said.append, run=bad,
                        stream=lambda argv, cwd: pytest.fail("nothing may run")) == 1
    assert any("older than 0.8.6" in s for s in said)


# ----------------------------------------------------------------------------- sidecar and agent


class FakeProc:
    def __init__(self, alive: bool = True, log: str = "") -> None:
        self._alive, self._log, self.log_path = alive, log, Path("exo.log")

    def alive(self) -> bool:
        return self._alive

    def exit_code(self) -> int | None:
        return None if self._alive else 1

    def read_log(self, max_bytes: int = 0) -> str:
        return self._log


class FakeSup:
    def __init__(self, proc: FakeProc) -> None:
        self.proc, self.started, self.stopped = proc, [], []

    def start(self, key, exe, args, cwd, log_path, env=None):
        self.started.append((key, args, env))
        return self.proc

    def get(self, key):
        return None if self.stopped else self.proc

    def stop(self, key, grace_s=8.0):
        self.stopped.append(key)
        return StopOutcome(key, 77, 0, "graceful")


def test_namespace_is_per_host_and_pinned():
    ns = exo_sidecar.namespace_for("192.168.1.20", 8000)
    assert ns.startswith(f"hc-{inst.EXO_PIN[:7]}-") and ns == exo_sidecar.namespace_for("192.168.1.20", 8000)
    assert ns != exo_sidecar.namespace_for("192.168.1.21", 8000)


def test_sidecar_starts_exo_with_the_namespace_and_reports_running(monkeypatch, tmp_path):
    from hypercompute import netutil
    from hypercompute.config import Paths

    port = netutil.free_loopback_port()  # start() checks the port is free: a dev Mac with EXO open holds 52415
    monkeypatch.setattr(inst, "API_PORT", port)
    monkeypatch.setattr(inst, "status", lambda paths: ExtraEngineInfo(engine="exo", installed=True,
                                                                       status="installed"))
    sup = FakeSup(FakeProc())
    car = exo_sidecar.ExoSidecar(sup, Paths(home=tmp_path), "192.168.1.31", "hc-ns",  # type: ignore[arg-type]
                                 get=lambda url, timeout: httpx.Response(200, json=A))
    info = car.start(timeout_s=5)
    assert info.status == "running" and info.endpoint == f"192.168.1.31:{port}" and info.version == inst.EXO_PIN
    assert sup.started == [("exo", ["-m", "exo", "-v", "--api-port", str(port)], {"EXO_ZENOH_NAMESPACE": "hc-ns"})]
    assert car.stop() is not None and car.info().status == "installed"
    engines = exo_sidecar.with_exo([ExtraEngineInfo(engine="vllm", installed=False, status="unsupported"),
                                    ExtraEngineInfo(engine="exo", installed=True, status="installed")], info)
    assert [(e.engine, e.status) for e in engines] == [("vllm", "unsupported"), ("exo", "running")]


def test_sidecar_explains_an_early_exit(monkeypatch, tmp_path):
    from hypercompute import netutil
    from hypercompute.config import Paths

    monkeypatch.setattr(inst, "API_PORT", netutil.free_loopback_port())  # not the real 52415 an EXO app may hold
    monkeypatch.setattr(inst, "status", lambda paths: ExtraEngineInfo(engine="exo", installed=True,
                                                                       status="installed"))
    log = "Port 52415 is already in use - is exo already running? Use --api-port to choose another."
    car = exo_sidecar.ExoSidecar(FakeSup(FakeProc(alive=False, log=log)), Paths(home=tmp_path), "10.0.0.5",  # type: ignore[arg-type]
                                 "ns", get=lambda url, timeout: httpx.Response(503))
    with pytest.raises(exo_sidecar.ExoSidecarError) as err:
        car.start(timeout_s=5)
    assert "quit it first" in (err.value.fix or "")


def test_agent_reports_exo_at_join_and_stops_it_on_pause(monkeypatch):
    inv = make_inventory("macos")
    agent = Agent(host="192.168.1.20", port=8000, code="code", display_name="mac", inv=inv,
                  contribution=contribution(["MTL0"]), advertise_ip="192.168.1.31", worker_port=47052,
                  exo_enabled=True)
    running = ExtraEngineInfo(engine="exo", installed=True, status="running", endpoint=EP_A)

    class Car:
        endpoint, node_id, namespace = EP_A, A, "ns"

        def __init__(self, *a):
            self.state = running

        def start(self):
            return self.state

        def stop(self):
            self.state = ExtraEngineInfo(engine="exo", installed=True, status="installed")
            return StopOutcome("exo", 1, 0, "graceful")

        def info(self):
            return self.state

    monkeypatch.setattr(join, "ExoSidecar", Car)
    monkeypatch.setattr(agent.sup, "stop", lambda key, grace_s=8.0: None)
    agent.start_exo()
    assert [e for e in agent.inv.extra_engines if e.engine == "exo"] == [running]
    assert agent.exo_heartbeat() == {"extra_engines": [running]}  # live status reaches the host every heartbeat
    assert agent.pause() == "paused"
    assert next(e for e in agent.inv.extra_engines if e.engine == "exo").status == "installed"
    assert "exo" in agent.status_doc()


def test_join_with_exo_fails_early_on_a_machine_that_cannot_run_it(monkeypatch, capsys):
    monkeypatch.setattr(cli, "check_coordinator", lambda host, port: {})
    monkeypatch.setattr(cli.inventory, "collect", lambda: make_inventory("windows"))
    monkeypatch.setattr(cli, "_advertise_ip", lambda host, port, explicit: "10.0.0.5")
    monkeypatch.setattr(cli.netutil, "pick_bindable_port", lambda ip, port, scan: (47052, []))
    monkeypatch.setattr(inst, "status", lambda paths: ExtraEngineInfo(engine="exo", installed=False,
                                                                       status="unsupported", detail="EXO no"))
    args = Namespace(host="192.168.1.20:8000", code="ABCD-EFGH", name=None, max_memory=None, reserve=None,
                     devices=None, worker_port=47052, advertise_ip=None, cache=False, yes=True, exo=True)
    assert cli._run_join(args) == 2
    assert "join without --exo" in capsys.readouterr().out
