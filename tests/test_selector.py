"""Engine selector: forced engines, the AirLLM / vLLM / EXO rules across variants, and measured profile reuse."""

import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute.agent.supervisor import Supervisor
from hypercompute.schemas import (
    ExtraEngineInfo,
    GgufSummary,
    ModelArtifact,
    ModelFile,
    ModelView,
    NodeState,
    NodeView,
    Plan,
    PlanRequest,
    PlanStrategy,
    utcnow,
)
from hypercompute.services import deployments
from hypercompute.services.deployments import DeploymentManager
from hypercompute.services.leases import LeaseManager, requests_from_plan
from hypercompute.services.nodes import NodeRegistry
from hypercompute.services.profiles import ProfileStore, signature
from hypercompute.services.selector import Inputs, choose_plan, profile_devices
from hypercompute.services.sessions import SessionService
from hypercompute.units import GiB
from test_split_order import EP5, cpu, gpu, mac, shape05, shape14, summary

ALL = frozenset({"llama_rpc", "airllm", "vllm", "exo"})


def art(model_id: str, fmt: str = "gguf", logical: str | None = "q14", engines: list[str] | None = None,
        ) -> ModelArtifact:
    files = [] if fmt == "mlx" else [ModelFile(name=f"{model_id}.bin", url="https://example.invalid/x",
                                                sha256="0" * 64, size_bytes=1)]
    return ModelArtifact(id=model_id, display_name=model_id, family="qwen2", params_b=14, quantization="Q6_K",
                         format=fmt, engines=engines or [], logical_id=logical, repo="r", revision="x",  # type: ignore[arg-type]
                         license="apache-2.0", files=files, size_bytes=10 * GiB if fmt == "mlx" else None)


class FakeEngines:
    """Stands in for adapters.registry.plan_for: records calls, answers feasible unless told otherwise."""

    def __init__(self, infeasible: frozenset[str] = frozenset(), raises: str | None = None) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []
        self.infeasible, self.raises = infeasible, raises

    def __call__(self, engine: str, artifact: ModelArtifact, req: PlanRequest) -> Plan:
        self.calls.append((engine, artifact.id, req.model_id, req.engine))
        if engine == self.raises:
            raise RuntimeError("nvidia-smi exploded")
        ok = engine not in self.infeasible
        return Plan(id="e" * 32, model_id=artifact.id, strategy=req.strategy, ctx_size=req.ctx_size, feasible=ok,
                    fit_label="estimated_fit" if ok else "does_not_fit", heuristic=True, assignments=[],
                    device_list=[], tensor_split=[], rpc_endpoints=[], cpu_resident_bytes=0,
                    reasons=[f"{engine} test plan"], warnings=[], candidates=[], created_at=utcnow(),
                    engine=engine, tier="disk" if engine == "airllm" else "gpu",  # type: ignore[arg-type]
                    capacity_mode=engine == "airllm")


def inputs(artifact: ModelArtifact, artifacts: list[ModelArtifact], engines: FakeEngines, **kw) -> Inputs:
    base = dict(req=PlanRequest(model_id=artifact.id, strategy=PlanStrategy.auto), artifact=artifact,
                shape=shape14() if artifact.format == "gguf" else None, local=[gpu(6)], remote=[], cpu=None,
                host_ram_free=32 * GiB, artifacts=artifacts, available=ALL, plan_engine=engines)
    return Inputs(**{**base, **kw})


# ----------------------------------------------------------------------------- forced engine


def test_a_forced_engine_that_is_not_installed_names_the_install_command():
    hf, engines = art("q14-hf", "hf"), FakeEngines()
    plan = choose_plan(inputs(hf, [hf], engines, req=PlanRequest(model_id=hf.id, engine="vllm"),
                              available=frozenset({"llama_rpc"})))
    assert not plan.feasible and plan.engine == "vllm" and plan.model_id == hf.id
    assert any("hypercompute engine install --engine vllm" in r for r in plan.reasons) and engines.calls == []


def test_forcing_an_engine_on_a_gguf_uses_its_variant_and_says_the_id_changes():
    gguf, hf, engines = art("q14-gguf"), art("q14-hf", "hf", engines=["airllm"]), FakeEngines()
    plan = choose_plan(inputs(gguf, [gguf, hf], engines, req=PlanRequest(model_id=gguf.id, engine="airllm")))
    assert plan.engine == "airllm" and plan.model_id == hf.id and plan.feasible
    assert engines.calls == [("airllm", hf.id, hf.id, "airllm")]
    assert any("different model id" in w for w in plan.warnings)


def test_forcing_llama_on_an_hf_artifact_points_at_the_gguf_variant():
    gguf, hf = art("q14-gguf"), art("q14-hf", "hf")
    plan = choose_plan(inputs(hf, [gguf, hf], FakeEngines(), req=PlanRequest(model_id=hf.id, engine="llama_rpc")))
    assert not plan.feasible and plan.engine == "llama_rpc" and "q14-gguf" in plan.reasons[0]


def test_an_engine_planner_that_fails_is_reported_not_raised():
    hf = art("q14-hf", "hf", engines=["vllm"])
    plan = choose_plan(inputs(hf, [hf], FakeEngines(raises="vllm"), req=PlanRequest(model_id=hf.id, engine="vllm")))
    assert not plan.feasible and "planner failed" in plan.reasons[0] and "nvidia-smi exploded" in plan.reasons[0]


# ----------------------------------------------------------------------------- the ladder across engines


def test_airllm_serves_the_hf_variant_when_no_llama_tier_fits():
    gguf, hf, engines = art("q14-gguf"), art("q14-hf", "hf", engines=["airllm"]), FakeEngines()
    plan = choose_plan(inputs(gguf, [gguf, hf], engines))  # 6 GiB GPU, no peers, no RAM tier
    assert plan.engine == "airllm" and plan.model_id == hf.id and plan.feasible and plan.capacity_mode
    assert any("different model id" in w and "capacity mode" in w for w in plan.warnings)
    assert any(c.label == f"llama_rpc: {gguf.id}" and not c.feasible for c in plan.candidates)
    assert engines.calls == [("airllm", hf.id, hf.id, "airllm")]


def test_no_airllm_without_capacity_mode_or_without_the_engine():
    gguf, hf = art("q14-gguf"), art("q14-hf", "hf", engines=["airllm"])
    off = FakeEngines()
    plan = choose_plan(inputs(gguf, [gguf, hf], off, req=PlanRequest(model_id=gguf.id, strategy=PlanStrategy.auto,
                                                                      allow_capacity_mode=False)))
    assert plan.engine == "llama_rpc" and not plan.feasible and off.calls == []
    missing = choose_plan(inputs(gguf, [gguf, hf], FakeEngines(), available=frozenset({"llama_rpc"})))
    assert missing.engine == "llama_rpc" and not missing.feasible
    assert any("AirLLM could serve q14-hf" in r and "--engine airllm" in r for r in missing.reasons)


def test_disk_paging_in_llama_comes_before_airllm():
    gguf, hf, engines = art("q14-gguf"), art("q14-hf", "hf", engines=["airllm"]), FakeEngines()
    plan = choose_plan(inputs(gguf, [gguf, hf], engines, cpu=cpu(3), host_ram_free=4 * GiB))
    assert plan.engine == "llama_rpc" and plan.tier == "gpu_disk" and plan.feasible and engines.calls == []


def test_a_not_downloaded_gguf_never_falls_back_to_airllm():
    gguf, hf, engines = art("q14-gguf"), art("q14-hf", "hf", engines=["airllm"]), FakeEngines()
    plan = choose_plan(inputs(gguf, [gguf, hf], engines, shape=None))
    assert plan.engine == "llama_rpc" and plan.fit_label == "fit_unknown" and engines.calls == []


def test_throughput_prefers_vllm_for_a_downloaded_variant_that_fits():
    gguf, hf = art("q05-gguf", logical="q05"), art("q05-hf", "hf", logical="q05", engines=["vllm"])
    thr_req = PlanRequest(model_id=gguf.id, strategy=PlanStrategy.auto, objective="throughput")
    kw = dict(shape=shape05(), local=[gpu(10)])
    thr = choose_plan(inputs(gguf, [gguf, hf], FakeEngines(), req=thr_req, **kw))
    assert thr.engine == "vllm" and thr.model_id == hf.id and any("throughput" in w for w in thr.warnings)
    assert any(c.label == f"llama_rpc: {gguf.id}" and c.feasible for c in thr.candidates)
    inter = choose_plan(inputs(gguf, [gguf, hf], FakeEngines(), **kw))
    assert inter.engine == "llama_rpc" and inter.model_id == gguf.id
    not_ready = choose_plan(inputs(gguf, [gguf, hf], FakeEngines(), req=thr_req,
                                   status=lambda m: "not_downloaded" if m == hf.id else "downloaded", **kw))
    assert not_ready.engine == "llama_rpc" and any(c.label == f"vllm: {hf.id}" and not c.feasible
                                                   for c in not_ready.candidates)
    too_big = choose_plan(inputs(gguf, [gguf, hf], FakeEngines(infeasible=frozenset({"vllm"})), req=thr_req, **kw))
    assert too_big.engine == "llama_rpc" and too_big.feasible


def _mac_node(exo_status: str = "running") -> NodeView:
    inv = make_inventory("macos").model_copy(update={"extra_engines": [
        ExtraEngineInfo(engine="exo", installed=True, status=exo_status)]})  # type: ignore[arg-type]
    return NodeView(id="b" * 32, display_name="mac", is_local=False, session_id="s", state=NodeState.eligible,
                    state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(), heartbeat_age_s=0,
                    inventory=inv, contribution=contribution(["MTL0"]), worker=None, worker_state=None,
                    verification=None)


def test_exo_is_a_candidate_on_an_all_mac_cluster_and_llama_stays_preferred():
    gguf, mlx = art("q14-gguf"), art("q14-mlx", "mlx")
    kw = dict(remote=[mac(12)], remote_nodes=[_mac_node()])
    plan = choose_plan(inputs(gguf, [gguf, mlx], FakeEngines(), **kw))
    assert plan.engine == "llama_rpc" and plan.feasible
    assert any(c.label == f"exo: {mlx.id}" and c.feasible for c in plan.candidates)
    forced = choose_plan(inputs(gguf, [gguf, mlx], FakeEngines(), req=PlanRequest(model_id=gguf.id, engine="exo"),
                                **kw))
    assert forced.engine == "exo" and forced.model_id == mlx.id
    no_fit = choose_plan(inputs(gguf, [gguf, mlx], FakeEngines(), remote=[mac(2)], remote_nodes=[_mac_node()]))
    assert no_fit.engine == "exo" and no_fit.feasible and any("EXO serves" in w for w in no_fit.warnings)
    stopped = choose_plan(inputs(gguf, [gguf, mlx], FakeEngines(), remote=[mac(12)],
                                 remote_nodes=[_mac_node("installed")]))
    assert not any(c.label.startswith("exo") for c in stopped.candidates)


def test_an_hf_model_uses_its_own_engines_fastest_first():
    hf = art("q14-hf", "hf")
    plan = choose_plan(inputs(hf, [hf], FakeEngines(), available=frozenset({"llama_rpc", "airllm"})))
    assert plan.engine == "airllm" and plan.capacity_mode and any("capacity mode" in w for w in plan.warnings)
    assert any(c.label == "vllm: q14-hf" and not c.feasible for c in plan.candidates)
    fast = choose_plan(inputs(hf, [hf], FakeEngines()))
    assert fast.engine == "vllm"


# ----------------------------------------------------------------------------- measured profiles


def _profile_setup(store, peer=None):
    profiles = ProfileStore(store)
    local, remote = [gpu(6)], [peer or mac(12)]
    gguf = art("q14")
    req = PlanRequest(model_id="q14", strategy=PlanStrategy.fastest_fit)
    heuristic = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, local=local, remote=remote))
    measured = heuristic.model_copy(update={"device_list": ["RPC0", "CUDA0"], "tensor_split": [30, 19],
                                            "engine_flags": {"flash_attn": "on", "ubatch": 256},
                                            "rpc_endpoints": [EP5]})
    saved = profiles.save(model_id="q14", sig=signature("q14", 4096, profile_devices(local, remote)),
                          objective="interactive", plan=measured, ttft_ms=120.0, decode_tps=21.0, prefill_tps=600.0,
                          score=3.2, tune_id="t1", run_ids=["r1"])
    return profiles, gguf, req, heuristic, saved


def test_a_measured_profile_replaces_the_heuristic_split_and_leases_follow_it(store):
    profiles, gguf, req, heuristic, saved = _profile_setup(store)
    assert heuristic.tensor_split == [28, 21] and heuristic.profile_id is None
    plan = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, remote=[mac(12)], profile=profiles.best))
    assert plan.profile_id == saved.id and not plan.heuristic and plan.feasible and plan.tier == "gpu"
    assert plan.tensor_split == [30, 19] and plan.engine_flags == {"flash_attn": "on", "ubatch": 256}
    assert [len(a.layers) for a in plan.assignments] == [30, 18] and plan.assignments[-1].holds_output
    assert plan.assignments[0].endpoint == EP5 and plan.rpc_endpoints == [EP5]
    assert plan.assignments[-1].compute_bytes_est < heuristic.assignments[-1].compute_bytes_est  # -ub 256
    leased = sum(r.bytes for r in requests_from_plan(plan, {}))
    assert leased == sum(a.total_bytes_est for a in plan.assignments)
    assert DeploymentManager._llama_names(plan, {EP5: 1}).device_list == ["RPC0", "CUDA0"]
    assert any(c.label == "heuristic gpu plan" for c in plan.candidates)
    off = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req.model_copy(update={"use_profiles": False}),
                             remote=[mac(12)], profile=profiles.best))
    assert off.profile_id is None and off.heuristic


def test_a_profile_is_not_reused_when_the_devices_changed(store):
    profiles, gguf, req, _h, _saved = _profile_setup(store)
    other_peer = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, remote=[mac(12, "10.0.0.9:47052")],
                                    profile=profiles.best))
    assert other_peer.profile_id is None and other_peer.heuristic
    alone = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, local=[gpu(14)], remote=[],
                               profile=profiles.best))
    assert alone.profile_id is None and alone.device_list == ["CUDA0"]


def test_a_profile_that_no_longer_fits_is_ignored_with_a_warning(store):
    profiles, gguf, req, _h, saved = _profile_setup(store)
    plan = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, remote=[mac(6)], profile=profiles.best))
    assert plan.profile_id is None and plan.heuristic
    assert any(f"measured profile {saved.id[:8]} ignored: it no longer fits" in w for w in plan.warnings)


def test_a_profile_naming_a_device_that_is_gone_is_ignored(store):
    profiles, gguf, req, heuristic, _saved = _profile_setup(store)
    local, remote = [gpu(6)], [mac(12)]
    gone = heuristic.model_copy(update={"device_list": ["RPC3", "CUDA0"], "tensor_split": [30, 19],
                                        "rpc_endpoints": [EP5]})
    best = profiles.save(model_id="q14", sig=signature("q14", 4096, profile_devices(local, remote)),
                         objective="interactive", plan=gone, ttft_ms=1.0, decode_tps=99.0, prefill_tps=1.0, score=0.1,
                         tune_id=None, run_ids=[])
    plan = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, remote=remote, profile=profiles.best))
    assert plan.profile_id == _saved.id and not plan.warnings  # the next profile that can replay is used
    none_fit = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req, remote=[mac(6)], profile=profiles.best))
    assert none_fit.profile_id is None and none_fit.heuristic  # neither replays: the warning names the best one
    assert none_fit.warnings == [f"measured profile {best.id[:8]} ignored: its devices are not all present now; "
                                 "using the heuristic plan"]


@pytest.mark.parametrize("limit,shadow", [
    ({"allow_ram_tier": False}, {"device_list": ["CUDA0"], "tensor_split": [1], "cpu_layers": 20, "tier": "gpu_ram",
                                 "rpc_endpoints": []}),
    ({"strategy": PlanStrategy.all_selected}, {"device_list": ["CUDA0"], "tensor_split": [49], "rpc_endpoints": []})])
def test_a_faster_profile_the_request_refuses_does_not_hide_one_it_allows(store, limit, shadow):
    # two tunes on one setup: one with the RAM tier on (or strategy auto), one under the owner's tighter limits
    profiles, gguf, req, heuristic, saved = _profile_setup(store)
    sig = signature("q14", 4096, profile_devices([gpu(6)], [mac(12)]))
    faster = profiles.save(model_id="q14", sig=sig, objective="interactive", plan=heuristic.model_copy(update=shadow),
                           ttft_ms=50.0, decode_tps=40.0, prefill_tps=900.0, score=1.0, tune_id="t0", run_ids=[])
    assert profiles.best("q14", sig, "interactive").id == faster.id  # it ranks first for this key
    plan = choose_plan(inputs(gguf, [gguf], FakeEngines(), req=req.model_copy(update=limit), remote=[mac(12)],
                              profile=profiles.best))
    assert plan.profile_id == saved.id and not plan.heuristic and plan.tensor_split == [30, 19]
    assert not any("ignored" in w for w in plan.warnings)


# ----------------------------------------------------------------------------- DeploymentManager.plan wiring


class Models:
    def __init__(self, arts: list[ModelArtifact], status: dict[str, str], ggufs: dict[str, GgufSummary]) -> None:
        self._arts, self._status, self._ggufs = {a.id: a for a in arts}, status, ggufs

    def artifacts(self) -> list[ModelArtifact]:
        return list(self._arts.values())

    def get(self, model_id: str) -> ModelArtifact:
        return self._arts[model_id]

    def view(self, model_id: str) -> ModelView:
        return ModelView(artifact=self._arts[model_id], status=self._status[model_id], local_path=None,  # type: ignore[arg-type]
                         downloaded_bytes=0, total_bytes=0, verified_sha256=False, gguf=self._ggufs.get(model_id))


def _manager(store, events, tmp_path, models: Models, extra: dict | None = None) -> DeploymentManager:
    svc = SessionService(store, events, b"k" * 32, 8000, lambda: ["10.0.0.2"])
    reg = NodeRegistry(store, events, svc, b"k" * 32, engine_info(), make_inventory(), contribution(["CUDA0"]))
    return DeploymentManager(store, events, reg, models, LeaseManager(store), object(),  # type: ignore[arg-type]
                             Supervisor(tmp_path / "m.json"), extra_adapters=extra)


def test_manager_plans_an_hf_model_without_a_gguf_shape(store, events, tmp_path, monkeypatch):
    hf = art("q14-hf", "hf", engines=["airllm"])
    mgr = _manager(store, events, tmp_path, Models([hf], {hf.id: "downloaded"}, {}), extra={"airllm": object()})
    mgr.adapter_ctx = object()
    engines = FakeEngines()
    monkeypatch.setattr(deployments.registry, "plan_for",
                        lambda engine, req, a, local, remote, ctx: engines(engine, a, req))
    plan = mgr.plan(PlanRequest(model_id=hf.id, strategy=PlanStrategy.local_only))
    assert plan.engine == "airllm" and plan.feasible and plan.device_list == []
    assert mgr.stored_plan(plan.id).engine == "airllm"


def test_manager_gates_on_the_download_of_the_model_the_selector_chose(store, events, tmp_path, monkeypatch):
    gguf, hf = art("q14-gguf"), art("q14-hf", "hf", engines=["airllm"])
    big = summary(48, 225, 5120, 40, 8, 609)
    models = Models([gguf, hf], {gguf.id: "downloaded", hf.id: "not_downloaded"}, {gguf.id: big})
    mgr = _manager(store, events, tmp_path, models, extra={"airllm": object()})
    mgr.adapter_ctx = object()
    monkeypatch.setattr(deployments.registry, "plan_for",
                        lambda engine, req, a, local, remote, ctx: FakeEngines()(engine, a, req))
    plan = mgr.plan(PlanRequest(model_id=gguf.id, strategy=PlanStrategy.local_only, allow_ram_tier=False))
    assert plan.engine == "airllm" and plan.model_id == hf.id and not plan.feasible
    assert plan.reasons[0] == f"model {hf.id} is not downloaded; download it first"


def test_manager_keeps_the_cpu_tier_out_of_the_device_list(store, events, tmp_path):
    gguf = art("q14-gguf")
    mgr = _manager(store, events, tmp_path, Models([gguf], {gguf.id: "downloaded"},
                                                   {gguf.id: summary(48, 225, 5120, 40, 8, 609)}))
    plan = mgr.plan(PlanRequest(model_id=gguf.id, strategy=PlanStrategy.local_only))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.device_list == ["CUDA0"]
    assert plan.assignments[0].device == "CPU" and sum(plan.tensor_split) == plan.n_gpu_units


@pytest.mark.parametrize("status", ["not_downloaded", "downloading"])
def test_manager_still_reports_a_missing_download(store, events, tmp_path, status):
    gguf = art("q05", logical=None)
    mgr = _manager(store, events, tmp_path, Models([gguf], {gguf.id: status}, {}))
    plan = mgr.plan(PlanRequest(model_id=gguf.id, strategy=PlanStrategy.local_only))
    assert not plan.feasible and plan.reasons[0] == f"model is {status.replace('_', ' ')}; download it first"
