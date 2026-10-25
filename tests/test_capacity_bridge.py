"""Capacity bridge and CapacityService.plan on synthetic nodes and manifests: inputs from live state, candidate to Plan
to llama-server arguments, the planner-versus-optimizer choice and the selector's fallback. No hardware is touched."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute.adapters.base import DeploymentSpec
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.capacity import bridge, costs
from hypercompute.capacity.measurements import MeasurementIdentity, MeasurementRecord, MeasurementStore
from hypercompute.capacity.pools import HostMemory
from hypercompute.capacity.service import CapacityService, CapacityUnavailable
from hypercompute.capacity.types import ModelManifest, UnitSpec
from hypercompute.schemas import (
    EngineDevice,
    ModelArtifact,
    ModelFile,
    ModelView,
    NodeState,
    NodeView,
    Plan,
    PlanRequest,
    PlanStrategy,
    VerificationResult,
    WorkerEndpoint,
)
from hypercompute.services import planner
from hypercompute.services.deployments import (
    DeploymentManager,
    cpu_budget,
    local_budgets,
    remote_budgets,
)
from hypercompute.services.selector import Inputs, choose_plan

MIB, GIB = 1 << 20, 1 << 30
NOW = datetime(2026, 10, 22, 6, 0, tzinfo=UTC)
MODEL = "q7-test"
MAC = "a" * 32
BOX = "b" * 32
EP = "127.0.0.1:52110"  # the friend's M4 Air behind the local Iroh relay bridge (results/capacity/*-link-relay-*)
RELAY_RTT_MS = 36.9  # measured median, link.json of 2026-10-22T04:01:53Z
RELAY_GOODPUT = 1.59e6  # measured bulk goodput in bytes/s, same evidence file


# --------------------------------------------------------------------------------------------------- fixtures


def make_manifest(n_layer: int = 28, block_mib: float = 150, head_mib: float = 500, *, n_embd: int = 3584,
                  embd_mib: float = 0) -> ModelManifest:
    units = [UnitSpec(i, "block", int(block_mib * MIB), int(block_mib * MIB)) for i in range(n_layer)]
    units.append(UnitSpec(n_layer, "head", int(head_mib * MIB), int(head_mib * MIB)))
    return ModelManifest(
        model_id=MODEL, revision="r1", shards=(), architecture="qwen2", file_type="Q4_K_M", n_layer=n_layer,
        n_embd=n_embd, n_head=28, n_head_kv=4, key_length=128, value_length=128, context_length=32768, rope=(),
        tokenizer_model="gpt2", vocab_size=152064, chat_template_sha256=None, units=tuple(units),
        token_embd_bytes=int(embd_mib * MIB), other_bytes=0, tied_embeddings=False, streamable=True,
        fingerprint="f" * 64)


def local_node(*, vram_free_gib: float | None = 11.0, reserve_gib: float = 4.0) -> NodeView:
    gpu = vram_free_gib is not None
    inv = make_inventory(ram_total=32 * GIB, ram_free=16 * GIB, vram_total=12 * GIB if gpu else None,
                         vram_free=int(vram_free_gib * GIB) if gpu else None,
                         engine=engine_info() if gpu else engine_info(devices=[], accelerator="cpu"))
    return NodeView(id="local", display_name="win (this machine)", is_local=True, session_id=None,
                    state=NodeState.eligible, state_reason=None, joined_at=NOW, last_heartbeat_at=NOW,
                    heartbeat_age_s=0.0, inventory=inv,
                    contribution=contribution(["CUDA0"] if gpu else [], max_gib=64, reserve_gib=reserve_gib),
                    worker=None, worker_state=None, verification=None)


def mac_node(node_id: str = MAC, *, free_gib: float = 14.0, max_gib: float = 12.0, reserve_gib: float = 2.0,
             endpoint: str = EP) -> NodeView:
    host, port = endpoint.rsplit(":", 1)
    inv = make_inventory("macos", ram_total=16 * GIB, ram_free=int(free_gib * GIB))
    rpc = [EngineDevice(name="RPC0", description=endpoint, total_bytes=int(12.7 * GIB), free_bytes=int(12.7 * GIB))]
    return NodeView(id=node_id, display_name="m4 air", is_local=False, session_id="s", state=NodeState.eligible,
                    state_reason=None, joined_at=NOW, last_heartbeat_at=NOW, heartbeat_age_s=1.0, inventory=inv,
                    contribution=contribution(["MTL0"], max_gib=max_gib, reserve_gib=reserve_gib),
                    worker=WorkerEndpoint(host=host, port=int(port)), worker_state=None,
                    verification=VerificationResult(checked_at=NOW, reachable=True, engine_match=True,
                                                    rpc_devices=rpc, detail="fixture"))


def artifact() -> ModelArtifact:
    return ModelArtifact(id=MODEL, display_name=MODEL, family="qwen2", params_b=7, quantization="Q4_K_M",
                         repo="r", revision="r1", license="apache-2.0",
                         files=[ModelFile(name=f"{MODEL}.gguf", url="https://example.invalid/m", sha256="0" * 64,
                                          size_bytes=1)])


class FakeNodes:
    def __init__(self, local: NodeView, remote: list[NodeView] | None = None) -> None:
        self.local, self.remote = local, list(remote or [])

    def eligible_remote(self, node_ids: list[str] | None) -> list[NodeView]:
        return [n for n in self.remote if node_ids is None or n.id in node_ids]


class FakeModels:
    def __init__(self, status: str = "downloaded", fmt: str = "gguf") -> None:
        self.art = artifact().model_copy(update={"format": fmt})
        self.status = status

    def view(self, model_id: str) -> ModelView:
        from hypercompute.services.models import ModelError

        if model_id != MODEL:
            raise ModelError(f"unknown model {model_id!r}")
        return ModelView(artifact=self.art, status=self.status, local_path=None, downloaded_bytes=0,  # type: ignore[arg-type]
                         total_bytes=0, verified_sha256=True, gguf=None)

    def primary_path(self, model_id: str) -> Path:
        return Path("models") / model_id / f"{model_id}.gguf"


class FakeLeases:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = list(rows or [])

    def active(self) -> list[dict]:
        return list(self.rows)


def memory(available_gib: float) -> HostMemory:
    return HostMemory(total=32 * GIB, available=int(available_gib * GIB), commit_charge=None, commit_limit=None,
                      system_cache=None, observed_at=NOW, source="test")


def stream_cap(m: ModelManifest) -> planner.StreamCapability:
    return planner.StreamCapability(unit_buffer_bytes=tuple(u.buffer_bytes for u in m.units), slots=2)


def make_service(tmp_path: Path, m: ModelManifest, local: NodeView, remote: list[NodeView] | None = None, *,
                 ram_gib: float = 16.0, stream: bool = False, leases: FakeLeases | None = None,
                 ready: set[str] | None = None, path: str = "lan", models: FakeModels | None = None,
                 clock: datetime = NOW) -> CapacityService:
    cap = stream_cap(m) if stream else None
    probes = bridge.Probes(memory=lambda: memory(ram_gib), disk_free=lambda _p: 500 * GIB,
                           manifest=lambda _a, _s: m, stream=lambda _s: cap, clock=lambda: clock)
    svc = CapacityService(FakeNodes(local, remote), models or FakeModels(), leases or FakeLeases(),  # type: ignore[arg-type]
                          MeasurementStore(tmp_path / "measurements"),
                          deployment_ready=lambda dep: dep in (ready or set()), probes=probes,
                          models_dir=tmp_path / "models")
    svc.path_of = lambda _node: path  # type: ignore[assignment,return-value]
    return svc


def store_link(store_dir: Path, node_id: str, path: str, *, rtt: float = RELAY_RTT_MS,
               goodput: float = RELAY_GOODPUT, at: datetime = NOW) -> None:
    store = MeasurementStore(store_dir)
    for kind, value, unit in (("link.rtt_ms", rtt, "ms"), ("link.goodput_bytes_per_s", goodput, "bytes/s")):
        store.put(MeasurementRecord(identity=MeasurementIdentity(node_id=node_id, device="rpc0",
                                                                 backend_build="ggml-rpc-8.0.0", path_class=path,
                                                                 kind=kind),
                                    value=value, unit=unit, measured_at=at, samples=20, note="fixture"))


def req(strategy: PlanStrategy = PlanStrategy.fastest_fit, **kw: object) -> PlanRequest:
    return PlanRequest(model_id=MODEL, strategy=strategy, **kw)  # type: ignore[arg-type]


def args_for(plan: Plan, endpoints: dict[str, int] | None = None) -> list[str]:
    named = DeploymentManager._llama_names(plan, endpoints or {})
    adapter = object.__new__(LlamaRpcAdapter)  # build_args reads only the spec
    return adapter.build_args(DeploymentSpec(deployment_id="d", plan=named, model_path=Path("m.gguf"), alias="m"),
                              9000)


def flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


# --------------------------------------------------------------------------------------------------- conversions


def test_gpu_only_plan_puts_every_unit_and_the_head_on_cuda0(tmp_path):
    m = make_manifest()
    plan = make_service(tmp_path, m, local_node()).plan(req())
    assert plan.feasible and plan.tier == "gpu" and plan.cpu_layers == 0 and plan.n_gpu_units is None
    assert plan.device_list == ["CUDA0"] and plan.tensor_split == [29] and plan.rpc_endpoints == []
    a = plan.assignments[0]
    assert a.holds_output and a.layers == list(range(28)) and a.pool_id == "vram:CUDA0" and a.fits
    assert a.total_bytes_est == a.weight_bytes + a.kv_bytes + a.compute_bytes_est + a.overhead_bytes_est
    assert plan.prediction is not None and plan.prediction.decode_ms > 0 and plan.heuristic
    assert any(w.startswith("predicted with priors") and "local:CUDA0 decode" in w for w in plan.warnings)
    args = args_for(plan)
    assert flag(args, "--device") == "CUDA0" and flag(args, "--tensor-split") == "29" and flag(args, "-ngl") == "all"
    assert "--rpc" not in args


def test_cpu_prefix_and_gpu_become_ngl_and_a_gpu_ram_tier(tmp_path):
    m = make_manifest()
    plan = make_service(tmp_path, m, local_node(vram_free_gib=4.0)).plan(req())
    assert plan.feasible and plan.tier == "gpu_ram" and plan.stream is None and not plan.capacity_mode
    cpu_a, gpu_a = plan.assignments
    assert cpu_a.device == "CPU" and cpu_a.layers == list(range(plan.cpu_layers)) and not cpu_a.holds_output
    assert gpu_a.device == "CUDA0" and gpu_a.holds_output and gpu_a.layers[0] == plan.cpu_layers
    assert plan.n_gpu_units == 29 - plan.cpu_layers == sum(plan.tensor_split)
    args = args_for(plan)
    assert flag(args, "--device") == "CUDA0" and flag(args, "-ngl") == str(plan.n_gpu_units)
    assert flag(args, "--tensor-split") == str(29 - plan.cpu_layers) and "CPU" not in flag(args, "--device")


def test_streamed_cpu_prefix_carries_an_exact_stream_spec(tmp_path):
    m = make_manifest()
    # 6 GiB available - 4 GiB reserve - 512 MiB safety leaves 1.5 GiB for 13 CPU blocks of 150 MiB: some stream
    plan = make_service(tmp_path, m, local_node(vram_free_gib=4.0), ram_gib=6.0, stream=True).plan(req())
    assert plan.feasible and plan.tier == "stream" and plan.capacity_mode
    spec = plan.stream
    assert spec is not None and spec.streamed_units and spec.slots == 2
    assert sorted(spec.resident_units + spec.streamed_units) == list(range(plan.cpu_layers))
    assert spec.slot_bytes == 150 * MIB and spec.resident_bytes == len(spec.resident_units) * 150 * MIB
    assert spec.stream_bytes_per_pass == len(spec.streamed_units) * 150 * MIB
    assert spec.read_provenance == "prior" and spec.predicted_io_ms_per_token == pytest.approx(
        costs.stream_io_ms(spec.stream_bytes_per_pass, costs.storage_prior("local")))
    cpu_a = plan.assignments[0]
    assert cpu_a.total_bytes_est <= cpu_a.budget_bytes
    assert cpu_a.total_bytes_est == (spec.committed_bytes + cpu_a.kv_bytes + cpu_a.compute_bytes_est
                                     + costs.overhead_bytes("cpu"))
    args = args_for(plan)
    assert flag(args, "--load-mode") == "mmap" and "--no-op-offload" in args
    assert flag(args, "-ngl") == str(plan.n_gpu_units) and flag(args, "--device") == "CUDA0"


def test_cpu_mac_and_gpu_chain_names_the_peer_rpc0_before_cuda0(tmp_path):
    m = make_manifest()
    # the Mac admits 1 GiB (its cap), the host CPU 2 GiB: neither alone takes the 13 blocks the GPU cannot hold
    svc = make_service(tmp_path, m, local_node(vram_free_gib=4.0), [mac_node(free_gib=6.0, max_gib=1.0)],
                       ram_gib=6.5)
    plan = svc.plan(req())
    assert plan.feasible and [a.device for a in plan.assignments] == ["CPU", f"{EP}#0", "CUDA0"]
    cpu_a, mac_a, gpu_a = plan.assignments
    assert mac_a.endpoint == EP and mac_a.node_id == MAC and mac_a.pool_id == "uma" and mac_a.layers
    assert mac_a.layers[0] == len(cpu_a.layers) and gpu_a.layers[0] == mac_a.layers[-1] + 1 and gpu_a.holds_output
    assert plan.rpc_endpoints == [EP] and plan.tier == "gpu_ram"
    assert plan.tensor_split == [len(mac_a.layers), len(gpu_a.layers) + 1]
    assert sum(plan.tensor_split) == plan.n_gpu_units
    args = args_for(plan, {EP: 1})
    assert flag(args, "--rpc") == EP and flag(args, "--device") == "RPC0,CUDA0"
    assert flag(args, "--tensor-split") == ",".join(map(str, plan.tensor_split))
    assert flag(args, "-ngl") == str(29 - len(cpu_a.layers))
    stages = plan.prediction.stages  # type: ignore[union-attr]
    assert [s.device for s in stages] == ["CPU", "MTL0", "CUDA0"] and stages[1].path == "lan" and stages[1].hop_ms > 0


def test_cpu_only_host_runs_every_unit_on_the_cpu_with_device_none(tmp_path):
    m = make_manifest()
    plan = make_service(tmp_path, m, local_node(vram_free_gib=None)).plan(req())
    assert plan.feasible and plan.tier == "gpu_ram" and plan.device_list == [] and plan.tensor_split == []
    assert plan.cpu_layers == 28 and plan.n_gpu_units == 0 and plan.assignments[0].holds_output
    args = args_for(plan)
    assert flag(args, "--device") == "none" and flag(args, "-ngl") == "0" and "--tensor-split" not in args


def test_cpu_only_host_with_little_ram_streams_the_head_too(tmp_path):
    m = make_manifest()
    plan = make_service(tmp_path, m, local_node(vram_free_gib=None), ram_gib=7.0, stream=True).plan(req())
    assert plan.feasible and plan.tier == "stream" and plan.stream is not None
    assert sorted(plan.stream.resident_units + plan.stream.streamed_units) == list(range(29))  # head is unit 28
    args = args_for(plan)
    assert flag(args, "--device") == "none" and flag(args, "-ngl") == "0" and flag(args, "--load-mode") == "mmap"


def test_token_embeddings_come_out_of_the_cpu_pool_like_the_planner(tmp_path):
    m = make_manifest(embd_mib=1024)
    svc = make_service(tmp_path, m, local_node(vram_free_gib=4.0), ram_gib=8.0)
    plan = svc.plan(req())
    cpu_a = plan.assignments[0]
    # 8 GiB - 4 GiB reserve - 512 MiB safety - 1 GiB embeddings
    assert cpu_a.device == "CPU" and cpu_a.budget_bytes == int(2.5 * GIB)
    assert plan.cpu_resident_bytes == 1024 * MIB
    assert any("token embeddings" in a for a in plan.prediction.assumptions)  # type: ignore[union-attr]


def test_several_cuda_devices_use_cuda0_and_say_so(tmp_path):
    local = local_node()
    devices = [*local.inventory.engine.devices,
               EngineDevice(name="CUDA1", description="second GPU", total_bytes=8 * GIB, free_bytes=8 * GIB)]
    inv = local.inventory.model_copy(update={"engine": local.inventory.engine.model_copy(update={"devices": devices})})
    local = local.model_copy(update={"inventory": inv,
                                     "contribution": contribution(["CUDA0", "CUDA1"], max_gib=64, reserve_gib=4)})
    plan = make_service(tmp_path, make_manifest(), local).plan(req())
    assert plan.device_list == ["CUDA0"]
    assert any("local CUDA1 not placed" in w and "uses CUDA0" in w for w in plan.warnings)


# --------------------------------------------------------------------------------------------------- inputs


def test_loaded_leases_count_once_and_pending_leases_reduce_free_memory(tmp_path):
    m = make_manifest()
    rows = [{"id": "l1", "deployment_id": "ready", "node_id": "local", "pool_id": "vram:CUDA0", "bytes": 2 * GIB,
             "revision": 0},
            {"id": "l2", "deployment_id": "loading", "node_id": "local", "pool_id": "vram:CUDA0", "bytes": 3 * GIB,
             "revision": 0}]
    svc = make_service(tmp_path, m, local_node(), leases=FakeLeases(rows), ready={"ready"})
    report = svc.pools()
    vram = next(p for p in report["pools"] if p["pool_id"] == "vram:CUDA0")
    # free 11 GiB - 512 MiB headroom - 3 GiB not yet loaded; the loaded 2 GiB is already outside `free`
    assert vram["admissible_bytes"] == 11 * GIB - 512 * MIB - 3 * GIB
    ram = next(p for p in report["pools"] if p["pool_id"] == "ram")
    assert ram["free_bytes"] == 16 * GIB and "fresh snapshot" in ram["source"]
    assert any(p["pool_id"] == "ssd" for p in report["pools"]) and report["host_memory"]["source"] == "test"
    assert {lease["materialized"] for lease in report["leases"]} == {True, False}


def test_a_replacement_counts_the_old_deployments_memory_as_free(tmp_path):
    m = make_manifest()
    rows = [{"id": "l1", "deployment_id": "old", "node_id": "local", "pool_id": "vram:CUDA0", "bytes": 6 * GIB,
             "revision": 3}]
    svc = make_service(tmp_path, m, local_node(vram_free_gib=4.0), leases=FakeLeases(rows), ready={"old"})
    snap = svc._snapshot([], tmp_path, "old")
    assert snap.admissible[("local", "vram:CUDA0")] == 10 * GIB - 512 * MIB and snap.leases == ()


def test_measured_link_replaces_the_relay_prior_only_while_fresh(tmp_path):
    store_dir = tmp_path / "measurements"
    store_link(store_dir, MAC, "relay", at=NOW - timedelta(minutes=10))
    store = MeasurementStore(store_dir)
    fresh = bridge.link_profile(store.records(), MAC, "relay", NOW)
    assert fresh.rtt_ms.provenance == "measured" and fresh.rtt_ms.value == RELAY_RTT_MS
    assert fresh.goodput_bytes_per_s.value == RELAY_GOODPUT
    assert fresh.round_trips_per_token.provenance == "prior"
    stale = bridge.link_profile(store.records(), MAC, "relay", NOW + timedelta(hours=2))
    assert stale.rtt_ms.provenance == "prior" and stale.goodput_bytes_per_s.provenance == "prior"
    other_path = bridge.link_profile(store.records(), MAC, "direct", NOW)
    assert other_path.rtt_ms.provenance == "prior"  # a relay measurement never prices a direct path


def test_measured_compute_replaces_the_prior_only_while_fresh_and_for_this_device(tmp_path):
    m = make_manifest()
    store = MeasurementStore(tmp_path / "measurements")

    def put(device: str, value: float, at: datetime, fingerprint: str | None) -> None:
        store.put(MeasurementRecord(identity=MeasurementIdentity(node_id="local", device=device, backend_build="b11770",
                                                                 model_fingerprint=fingerprint,
                                                                 kind=bridge.KIND_DECODE),
                                    value=value, unit="ms/GiB", measured_at=at, samples=12))

    put("CUDA0", 3.1, NOW - timedelta(days=8), m.fingerprint)  # stale
    put("CPU", 9.0, NOW - timedelta(days=1), None)
    records = store.records()
    gpu = bridge.device_profile("cuda", "local", "CUDA0", "vram:CUDA0", None, records, m, NOW)
    cpu = bridge.device_profile("cpu", "local", "CPU", "ram", None, records, m, NOW)
    assert gpu.decode_ms_per_gib.provenance == "prior" and cpu.decode_ms_per_gib.value == 9.0
    put("CUDA0", 2.9, NOW - timedelta(hours=1), None)
    put("CUDA0", 2.7, NOW - timedelta(hours=2), m.fingerprint)  # older, but taken on this artifact: preferred
    gpu = bridge.device_profile("cuda", "local", "CUDA0", "vram:CUDA0", None, store.records(), m, NOW)
    assert gpu.decode_ms_per_gib.provenance == "measured" and gpu.decode_ms_per_gib.value == 2.7
    assert gpu.prefill_ms_per_gib_token.provenance == "prior"


def test_plans_with_measured_coefficients_list_them_and_keep_the_rest_as_priors(tmp_path):
    m = make_manifest()
    store_link(tmp_path / "measurements", MAC, "relay")
    svc = make_service(tmp_path, m, local_node(), [mac_node()], path="relay")
    out = svc.explain(MODEL)
    assert any(f"{MAC}:MTL0 rtt 36.9 ms (measured" in line for line in out["measured"])
    assert out["feasible"] and out["plan"]["heuristic"] is True
    assert out["workload"]["prompt_tokens"] == 512 and out["candidates"] and out["explain"]


# --------------------------------------------------------------------------------------------------- choices


def _legacy(local: NodeView, remote: list[NodeView], request: PlanRequest, m: ModelManifest) -> Plan:
    shape = bridge.model_shape(m, request.ctx_size)
    return planner.make_plan(request, shape, local_budgets(local), [b for n in remote for b in remote_budgets(n)],
                             16 * GIB, cpu_budget(local))


def test_optimizer_plan_wins_when_the_planner_overflows_onto_a_measured_relay_peer(tmp_path):
    m = make_manifest(n_layer=40)
    local, mac = local_node(vram_free_gib=4.0), mac_node()
    store_link(tmp_path / "measurements", MAC, "relay")
    legacy = _legacy(local, [mac], req(PlanStrategy.auto), m)
    assert legacy.feasible and legacy.rpc_endpoints == [EP]  # the planner's 4 ms hop prior picks the peer
    plan = make_service(tmp_path, m, local, [mac], path="relay").plan(req(PlanStrategy.auto), legacy)
    assert plan.id != legacy.id and plan.rpc_endpoints == [] and plan.tier == "gpu_ram"
    won = [w for w in plan.warnings if w.startswith("capacity optimizer: its plan won")]
    assert won and "lower" in won[0] and plan.prediction is not None
    assert any(c.label.startswith("planner gpu plan (scored)") for c in plan.candidates)
    assert any(s.startswith(f"{MAC}:MTL0: not used") for s in plan.prediction.skipped)


def test_equal_predictions_keep_the_planners_plan_and_attach_its_score(tmp_path):
    m = make_manifest()
    local = local_node()
    legacy = _legacy(local, [], req(PlanStrategy.auto), m)
    plan = make_service(tmp_path, m, local).plan(req(PlanStrategy.auto), legacy)
    assert plan.id == legacy.id and plan.device_list == legacy.device_list
    assert plan.prediction is not None and plan.prediction.decode_ms > 0
    assert any("equal predictions keep it" in w for w in plan.warnings)


def test_a_lower_decode_keeps_the_planners_lan_peer_layout_over_the_optimizers_cold_cheaper_one(tmp_path):
    m = make_manifest(n_layer=40)
    local, mac = local_node(vram_free_gib=4.0), mac_node()
    legacy = _legacy(local, [mac], req(PlanStrategy.auto), m)
    plan = make_service(tmp_path, m, local, [mac], path="lan").plan(req(PlanStrategy.auto), legacy)
    # J charges the cold upload to the Mac, so the optimizer keeps the overflow on the host CPU; the planner's Mac
    # layout decodes faster once loaded, and the comparison is by decode
    assert plan.id == legacy.id and plan.rpc_endpoints == [EP]
    kept = [w for w in plan.warnings if w.startswith("capacity optimizer: the planner's gpu plan was kept")]
    assert kept and "lower" in kept[0]
    assert plan.prediction is not None and [s.path for s in plan.prediction.stages] == ["lan", "local"]


def test_run_together_requires_every_selected_peer_in_the_optimizer_plan(tmp_path):
    m = make_manifest()
    svc = make_service(tmp_path, m, local_node(), [mac_node()], path="lan")
    plan = svc.plan(req(PlanStrategy.all_selected))
    assert plan.feasible and plan.rpc_endpoints == [EP] and any(r.startswith("strategy: run together")
                                                                for r in plan.reasons)
    with pytest.raises(CapacityUnavailable) as exc:
        make_service(tmp_path, m, local_node()).plan(req(PlanStrategy.all_selected))
    assert exc.value.code == "no_peers"


@pytest.mark.parametrize(("models", "request_kw", "code"), [
    (FakeModels(status="not_downloaded"), {}, "not_downloaded"),
    (FakeModels(fmt="hf"), {}, "not_gguf"),
    (FakeModels(), {"objective": "throughput"}, "objective"),
    (FakeModels(), {"overflow": "peers"}, "overflow_pinned"),
    (FakeModels(), {"engine": "exo"}, "engine"),
])
def test_requests_the_optimizer_cannot_plan_raise_capacity_unavailable(tmp_path, models, request_kw, code):
    svc = make_service(tmp_path, make_manifest(), local_node(), models=models)
    with pytest.raises(CapacityUnavailable) as exc:
        svc.plan(req(**request_kw))
    assert exc.value.code == code


def test_no_feasible_layout_gives_an_infeasible_plan_with_reasons(tmp_path):
    plan = make_service(tmp_path, make_manifest(), local_node(vram_free_gib=1.0), ram_gib=5.0).plan(req())
    assert not plan.feasible and plan.fit_label == "does_not_fit"
    assert any("no feasible layout" in r for r in plan.reasons)


# --------------------------------------------------------------------------------------------------- selector


def _inputs(capacity: object) -> Inputs:
    m = make_manifest()
    local = local_node()
    return Inputs(req=req(PlanStrategy.local_only), artifact=artifact(), shape=bridge.model_shape(m, 4096),
                  local=local_budgets(local), remote=[], cpu=cpu_budget(local), host_ram_free=16 * GIB,
                  capacity=capacity)  # type: ignore[arg-type]


def test_selector_without_capacity_returns_the_planners_plan_unchanged():
    plan = choose_plan(_inputs(None))
    assert plan.feasible and plan.prediction is None and not any("capacity" in w for w in plan.warnings)


@pytest.mark.parametrize(("error", "text"), [
    (CapacityUnavailable("not_downloaded", "q7 is not downloaded"), "not used (not_downloaded): q7 is not downloaded"),
    (ValueError("bad measurement unit"), "not used (invalid input): bad measurement unit"),
    (RuntimeError("bug"), "capacity optimizer failed (RuntimeError: bug)"),
])
def test_selector_falls_back_to_the_planners_plan_with_the_reason(error, text):
    def failing(_req: PlanRequest, _legacy: Plan) -> Plan:
        raise error

    plan = choose_plan(_inputs(failing))
    assert plan.feasible and plan.device_list == ["CUDA0"] and any(text in w for w in plan.warnings)


def test_selector_returns_what_the_capacity_hook_chose():
    seen: list[str] = []

    def chosen(request: PlanRequest, legacy: Plan) -> Plan:
        seen.append(legacy.id)
        return legacy.model_copy(update={"warnings": [*legacy.warnings, "hook"]})

    plan = choose_plan(_inputs(chosen))
    assert seen == [plan.id] and plan.warnings[-1] == "hook"


def test_deployment_manager_passes_its_capacity_service_to_the_selector(store, events, tmp_path):
    from test_selector import Models, _manager, art, summary

    gguf = art("q14-gguf")
    mgr = _manager(store, events, tmp_path, Models([gguf], {gguf.id: "downloaded"},
                                                   {gguf.id: summary(48, 225, 5120, 40, 8, 609)}))
    calls: list[tuple[str, str]] = []

    class Capacity:
        def plan(self, request: PlanRequest, legacy: Plan) -> Plan:
            calls.append((request.model_id, legacy.model_id))
            raise CapacityUnavailable("manifest", "fixture has no GGUF file")

    plan = mgr.plan(PlanRequest(model_id=gguf.id, strategy=PlanStrategy.local_only))
    assert calls == [] and not any("capacity" in w for w in plan.warnings)
    mgr.capacity = Capacity()
    plan = mgr.plan(PlanRequest(model_id=gguf.id, strategy=PlanStrategy.local_only))
    assert calls == [(gguf.id, gguf.id)] and any("not used (manifest)" in w for w in plan.warnings)
    assert plan.feasible and plan.device_list == ["CUDA0"]


# --------------------------------------------------------------------------------------------------- review regressions


def test_the_local_contribution_limit_caps_the_gpu_as_the_planner_and_lease_admission_do(tmp_path):
    from hypercompute.services.deployments import lease_budgets
    from hypercompute.services.leases import requests_from_plan

    m = make_manifest()
    local = local_node().model_copy(update={"contribution": contribution(["CUDA0"], max_gib=2, reserve_gib=4)})
    legacy = _legacy(local, [], req(), m)
    assert legacy.tier == "gpu_ram" and legacy.assignments[-1].budget_bytes == 2 * GIB
    plan = make_service(tmp_path, m, local).plan(req(), legacy)
    gpu_a = next(a for a in plan.assignments if a.device == "CUDA0")
    assert gpu_a.budget_bytes == 2 * GIB and gpu_a.total_bytes_est <= 2 * GIB and plan.cpu_layers > 0

    class Nodes:
        def get(self, node_id: str) -> NodeView | None:
            return local if node_id == "local" else None

    # the reservation create() makes: every lease within the node allowance it is checked against
    requests = requests_from_plan(plan, lease_budgets(plan, Nodes()))  # type: ignore[arg-type]
    assert sum(r.bytes for r in requests) <= requests[0].node_budget
    assert all(r.bytes <= r.pool_budget for r in requests)


def test_a_plan_that_fits_beside_held_leases_passes_lease_admission(tmp_path, store):
    from hypercompute.services.deployments import lease_budgets
    from hypercompute.services.leases import LeaseManager, LeaseRequest, requests_from_plan

    m = make_manifest()
    local = local_node()
    leases = LeaseManager(store)
    # a failed deployment whose cleanup could not confirm its engine stopped still holds 3 GiB of VRAM
    leases.reserve("stuck", [LeaseRequest(node_id="local", pool_id="vram:CUDA0", bytes=3 * GIB,
                                          pool_budget=11 * GIB, node_budget=64 * GIB)])
    plan = make_service(tmp_path, m, local, leases=leases).plan(req())  # type: ignore[arg-type]
    gpu_a = next(a for a in plan.assignments if a.device == "CUDA0")
    # the optimizer admits 11 GiB - 512 MiB headroom - the 3 GiB lease; the GPU alone fits in that
    assert plan.feasible and plan.tier == "gpu" and gpu_a.fits and gpu_a.total_bytes_est <= int(7.5 * GIB)

    class Nodes:
        def get(self, node_id: str) -> NodeView | None:
            return local if node_id == "local" else None

    # LeaseManager subtracts the held 3 GiB from the plan's budget itself: it must not have been taken off already
    assert leases.admits(requests_from_plan(plan, lease_budgets(plan, Nodes()))) is None  # type: ignore[arg-type]


def test_measured_coefficients_say_what_they_were_measured_on(tmp_path):
    m = make_manifest()
    store = MeasurementStore(tmp_path / "measurements")
    identity = MeasurementIdentity(node_id="local", device="CPU", backend_build="b11770", ctx=4096,
                                   kind=bridge.KIND_DECODE)
    store.put(MeasurementRecord(identity=identity, value=9.0, unit="ms/GiB", measured_at=NOW - timedelta(hours=1),
                                samples=12, evidence="results/capacity/x/bench.json"))
    store_link(tmp_path / "measurements", MAC, "relay")
    records = store.records()
    cpu = bridge.device_profile("cpu", "local", "CPU", "ram", None, records, m, NOW).decode_ms_per_gib
    assert cpu.provenance == "measured" and cpu.note.startswith("model-agnostic; CPU, build b11770; ctx 4096")
    assert cpu.note.endswith("evidence results/capacity/x/bench.json")
    rtt = bridge.link_profile(records, MAC, "relay", NOW).rtt_ms
    assert rtt.note == "fixture; relay path; rpc0, build ggml-rpc-8.0.0"


def _mac_with_cpu_and_metal(max_gib: float = 8.0) -> NodeView:
    mac = mac_node(free_gib=10.0, max_gib=max_gib)
    rpc = [EngineDevice(name="RPC0", description=EP, total_bytes=12 * GIB, free_bytes=9 * GIB),
           EngineDevice(name="RPC1", description=EP, total_bytes=12 * GIB, free_bytes=9 * GIB)]
    ver = mac.verification.model_copy(update={"rpc_devices": rpc})  # type: ignore[union-attr]
    return mac.model_copy(update={"contribution": contribution(["MTL0", "CPU"], max_gib=max_gib, reserve_gib=2),
                                  "verification": ver})


def test_a_unified_peer_pool_is_counted_once_in_the_usable_total(tmp_path):
    svc = make_service(tmp_path, make_manifest(n_layer=40), local_node(vram_free_gib=4.0), [_mac_with_cpu_and_metal()],
                       path="lan", ram_gib=5.0)
    plan = svc.plan(req())
    total = next(r for r in plan.reasons if r.startswith("contributed / estimated usable"))
    # host CPU 0.5 GiB + CUDA0 3.5 GiB + the Mac's one uma pool 8 GiB, although MTL0 and its CPU both offer it
    assert "12.0 GiB" in total
    assert {a.pool_id for a in plan.assignments if a.node_id == MAC} <= {"uma"}


def test_a_plan_whose_compute_and_storage_are_measured_still_names_the_setup_prior(tmp_path):
    m = make_manifest()
    store = MeasurementStore(tmp_path / "measurements")
    rows = [("CUDA0", bridge.KIND_DECODE, 2.6, "ms/GiB"), ("CUDA0", bridge.KIND_PREFILL, 0.02, "ms/GiB/token"),
            ("ssd", bridge.KIND_STORAGE_READ, 3.45 * GIB, "bytes/s"),
            ("ssd", bridge.KIND_STREAM_EFFICIENCY, 0.7, "fraction")]
    for device, kind, value, unit in rows:
        identity = MeasurementIdentity(node_id="local", device=device, backend_build="b11770", kind=kind)
        store.put(MeasurementRecord(identity=identity, value=value, unit=unit, measured_at=NOW - timedelta(hours=1),
                                    samples=5))
    plan = make_service(tmp_path, m, local_node()).plan(req())
    priors = [w for w in plan.warnings if w.startswith("predicted with priors")]
    # J and the load charge a 2 s setup that nothing measures: the plan is still a heuristic and says why
    assert plan.heuristic and priors and "setup 2000 ms (prior)" in priors[0]
    assert "local:CUDA0 decode" not in priors[0] and "storage read" not in priors[0]


def test_a_relayed_peer_stage_names_the_relay_risk_prior(tmp_path):
    m = make_manifest()
    plan = make_service(tmp_path, m, local_node(vram_free_gib=4.0), [mac_node(free_gib=6.0, max_gib=1.0)],
                        ram_gib=6.5, path="relay").plan(req(PlanStrategy.all_selected))
    assert plan.feasible and plan.rpc_endpoints == [EP]
    priors = next(w for w in plan.warnings if w.startswith("predicted with priors"))
    assert f"{MAC}:MTL0 relay risk 0.1 fraction (prior)" in priors and "setup per peer stage" in priors


def _mac_host() -> NodeView:
    mtl = [EngineDevice(name="MTL0", description="Apple M4", total_bytes=10 * GIB, free_bytes=10 * GIB)]
    inv = make_inventory("macos", ram_total=32 * GIB, ram_free=20 * GIB,
                         engine=engine_info(devices=mtl, accelerator="metal"))
    return NodeView(id="local", display_name="mac host", is_local=True, session_id=None, state=NodeState.eligible,
                    state_reason=None, joined_at=NOW, last_heartbeat_at=NOW, heartbeat_age_s=0.0, inventory=inv,
                    contribution=contribution(["MTL0"], max_gib=16, reserve_gib=4), worker=None, worker_state=None,
                    verification=None)


def test_a_unified_memory_host_with_the_ram_tier_off_is_left_to_the_planner(tmp_path):
    m = make_manifest()
    local = _mac_host()
    svc = make_service(tmp_path, m, local, ram_gib=20.0)
    assert svc.plan(req(PlanStrategy.local_only)).device_list == ["MTL0"]  # the RAM tier on: shared pool, fits
    request = req(PlanStrategy.local_only, allow_ram_tier=False)
    with pytest.raises(CapacityUnavailable) as exc:  # a zero CPU figure would starve MTL0 on the shared pool
        svc.plan(request)
    assert exc.value.code == "ram_tier_off_unified"
    inp = Inputs(req=request, artifact=artifact(), shape=bridge.model_shape(m, 4096), local=local_budgets(local),
                 remote=[], cpu=cpu_budget(local), host_ram_free=20 * GIB, capacity=svc.plan)
    plan = choose_plan(inp)
    assert plan.feasible and plan.device_list == ["MTL0"]
    assert any("capacity optimizer not used (ram_tier_off_unified)" in w for w in plan.warnings)


def test_run_together_refuses_a_layout_that_leaves_a_local_gpu_out(tmp_path):
    local = local_node()
    devices = [*local.inventory.engine.devices,
               EngineDevice(name="CUDA1", description="second GPU", total_bytes=8 * GIB, free_bytes=8 * GIB)]
    inv = local.inventory.model_copy(update={"engine": local.inventory.engine.model_copy(update={"devices": devices})})
    local = local.model_copy(update={"inventory": inv,
                                     "contribution": contribution(["CUDA0", "CUDA1"], max_gib=64, reserve_gib=4)})
    svc = make_service(tmp_path, make_manifest(), local, [mac_node()], path="lan")
    with pytest.raises(CapacityUnavailable) as exc:
        svc.plan(req(PlanStrategy.all_selected))
    assert exc.value.code == "participation" and "local CUDA1" in str(exc.value)


def test_an_unsafe_catalog_file_path_is_capacity_unavailable(tmp_path):
    from hypercompute.services.models import ModelError

    class UnsafeModels(FakeModels):
        def primary_path(self, model_id: str) -> Path:
            raise ModelError(f"unsafe file path in manifest for {model_id}: ../x.gguf")

    svc = make_service(tmp_path, make_manifest(), local_node(), models=UnsafeModels())
    with pytest.raises(CapacityUnavailable) as exc:
        svc.explain(MODEL)
    assert exc.value.code == "model_files"


def test_loaded_states_count_a_lease_as_materialized():
    from types import SimpleNamespace

    from hypercompute.coordinator import _ready_check
    from hypercompute.schemas import DeploymentState
    from hypercompute.services.deployments import DeploymentError

    states = {"warm": DeploymentState.warming, "ready": DeploymentState.ready, "drain": DeploymentState.draining,
              "load": DeploymentState.loading, "plan": DeploymentState.planned, "dead": DeploymentState.failed}

    class Deps:
        def get(self, dep_id: str) -> SimpleNamespace:
            if dep_id not in states:
                raise DeploymentError(404, "not_found", "no such deployment")
            return SimpleNamespace(state=states[dep_id])

    ready = _ready_check(Deps())  # type: ignore[arg-type]
    assert {d for d in [*states, "gone"] if ready(d)} == {"warm", "ready", "drain"}


def _storage_record(device: str, gib_per_s: float, age: timedelta) -> MeasurementRecord:
    identity = MeasurementIdentity(node_id="local", device=device, backend_build="storage_probe-4r-8MiB",
                                   kind=bridge.KIND_STORAGE_READ)
    return MeasurementRecord(identity=identity, value=gib_per_s * GIB, unit="bytes/s", measured_at=NOW - age,
                             samples=3, note=f"fixture {device}")


def test_a_storage_record_counts_only_for_the_volume_holding_the_model():
    # calibrate.py keys storage.read_bytes_per_s by volume; the newest record here is of another volume
    records = (_storage_record("D:", 1.0, timedelta(minutes=5)), _storage_record("C:", 5.0, timedelta(hours=2)),
               _storage_record("ssd", 2.0, timedelta(hours=1)))  # names no volume: counts for any, after its own
    read = bridge.storage_profile(records, "local", None, NOW, volume="c:").read_bytes_per_s
    assert read.provenance == "measured" and read.value == 5.0 * GIB
    assert bridge.storage_profile(records, "local", None, NOW, volume="E:").read_bytes_per_s.value == 2.0 * GIB
    alone = bridge.storage_profile(records[:1], "local", None, NOW, volume="E:").read_bytes_per_s
    assert alone.provenance == "prior"  # D:'s rate is not E:'s
    # without a volume (scripts/capacity/calibrate.py's stream terms) the freshest record wins, as before
    assert bridge.storage_profile(records, "local", None, NOW).read_bytes_per_s.value == 1.0 * GIB


def test_plans_read_the_storage_rate_of_the_model_volume(tmp_path):
    from hypercompute.capacity import storage_probe

    here = storage_probe.volume_of(FakeModels().primary_path(MODEL))
    other = "Z:" if here != "Z:" else "Y:"
    store = MeasurementStore(tmp_path / "measurements")
    store.put(_storage_record(other, 1.0, timedelta(minutes=5)))
    store.put(_storage_record(here, 5.0, timedelta(hours=2)))
    svc = make_service(tmp_path, make_manifest(), local_node(vram_free_gib=4.0), ram_gib=6.0, stream=True)
    out = svc.explain(MODEL)
    [line] = [m for m in out["measured"] if m.startswith("storage read")]
    assert f"fixture {here}" in line and f"fixture {other}" not in line
