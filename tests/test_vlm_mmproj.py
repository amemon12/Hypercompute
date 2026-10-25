"""Vision projectors (llama.cpp --mmproj): catalog entries, verified downloads, engine flags and the host budget.

The projector runs on the coordinator only, so it is downloaded with the model, never seeded to peers, and llama-server
gets it with -mmdev naming a local device: at b11770 the projector otherwise follows the first --device entry, and
Hypercompute lists the RPC peers first (common/arg.cpp, docs/VISION.md).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from hypercompute import engine
from hypercompute.adapters.base import DeploymentSpec
from hypercompute.adapters.llama_output import detect_failure
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.agent.supervisor import Supervisor
from hypercompute.schemas import GgufSummary, ModelArtifact, ModelFile, PlanRequest, PlanStrategy
from hypercompute.services import models
from hypercompute.services.deployments import (
    PROJECTOR_COMPUTE_EST,
    DeploymentManager,
    projector_warnings,
    reserve_projector,
)
from hypercompute.services.events import EventBus
from hypercompute.services.leases import LeaseManager
from hypercompute.services.models import PROJECTOR_MARKER, ModelError, ModelService, load_manifest
from hypercompute.services.planner import ModelShape, make_plan
from hypercompute.units import GiB, MiB
from pydantic import ValidationError
from test_explicit_layout import MAC_A, MODEL, Nodes, report
from test_explicit_layout import Models as ExplicitModels
from test_explicit_layout import mac as explicit_mac
from test_split_order import EP5, EP6, cpu, gpu, mac, shape14
from test_stream_tier import cap14

QWEN_VL = "qwen2.5-vl-72b-instruct-q4_k_m"
GEMMA3 = "gemma-3-27b-it-q4_k_m"


# ----------------------------------------------------------------------------- catalog


@pytest.mark.parametrize(("model_id", "name", "size", "sha"), [
    (QWEN_VL, "mmproj-F16.gguf", 1410223552, "add333fada915332370b62d3fb1b6fc21e3615002f667e96f241d85213e1956d"),
    (GEMMA3, "mmproj-google_gemma-3-27b-it-f16.gguf", 857739168,
     "54cb61c842fe49ac3c89bc1a614a2778163eb49f3dec2b90ff688b4c0392cb48"),
])
def test_catalog_vlms_list_their_projector_at_the_entry_revision(model_id, name, size, sha):
    art = {a.id: a for a in load_manifest()}[model_id]
    assert art.mmproj is not None
    assert (art.mmproj.name, art.mmproj.size_bytes, art.mmproj.sha256) == (name, size, sha)
    assert art.mmproj.url == f"https://huggingface.co/{art.repo}/resolve/{art.revision}/{name}"
    assert name not in {f.name for f in art.files}  # the projector is not a shard of the language model


def test_only_gguf_artifacts_take_a_projector():
    proj = ModelFile(name="mmproj.gguf", url="https://example.invalid/mmproj.gguf", sha256="0" * 64, size_bytes=1)
    with pytest.raises(ValidationError, match="mmproj"):
        ModelArtifact(id="hf", display_name="hf", family="t", params_b=1, quantization="bf16", format="hf",
                      repo="t/m", revision="r", license="mit", files=[], mmproj=proj)


# ----------------------------------------------------------------------------- model service


def use_transport(monkeypatch, payloads: dict[str, bytes], requests: list[str]) -> None:
    client_type = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        requests.append(name)
        return httpx.Response(200, content=payloads[name])

    monkeypatch.setattr(engine.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * GiB))


def file_of(name: str, payload: bytes, sha: str | None = None) -> ModelFile:
    return ModelFile(name=name, url=f"https://example.invalid/{name}",
                     sha256=sha or hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))


def vlm_service(tmp_path, weights: bytes, proj: bytes | None, proj_sha: str | None = None):
    art = ModelArtifact(id="tiny-vlm", display_name="Tiny VLM", family="test", params_b=0.001, quantization="Q4_K_M",
                        repo="test/vlm", revision="pinned", license="mit", files=[file_of("model.gguf", weights)],
                        mmproj=file_of("mmproj.gguf", proj, proj_sha) if proj is not None else None)
    service = ModelService(EventBus(), tmp_path / "models")
    service._artifacts = {art.id: art}
    return service, art


def test_download_fetches_and_verifies_the_projector_with_the_model(tmp_path, monkeypatch):
    weights, proj = b"w" * 4096, b"p" * 1024
    service, art = vlm_service(tmp_path, weights, proj)
    requests: list[str] = []
    use_transport(monkeypatch, {"model.gguf": weights, "mmproj.gguf": proj}, requests)
    assert service.view(art.id).projector_status == "not_downloaded" and service.projector_path(art.id) is None
    asyncio.run(service.download(art.id))
    view = service.view(art.id)
    assert view.status == "downloaded" and view.projector_status == "downloaded"
    assert requests == ["model.gguf", "mmproj.gguf"]
    path = service.projector_path(art.id)
    assert path == service.model_dir(art.id) / "mmproj.gguf" and path.read_bytes() == proj
    assert service.file_paths(art.id) == [service.model_dir(art.id) / "model.gguf"]  # the GGUF shards stay the model's
    assert (service.model_dir(art.id) / PROJECTOR_MARKER).is_file()


def test_a_model_downloaded_before_its_projector_stays_usable_and_fetches_only_the_projector(tmp_path, monkeypatch):
    weights, proj = b"w" * 4096, b"p" * 1024
    service, art = vlm_service(tmp_path, weights, proj)
    service.model_dir(art.id).mkdir(parents=True)
    (service.model_dir(art.id) / "model.gguf").write_bytes(weights)
    service._write_marker(art.id)  # what a download before projector support left behind
    before = service.view(art.id)
    assert before.status == "downloaded" and before.projector_status == "not_downloaded"
    requests: list[str] = []
    use_transport(monkeypatch, {"model.gguf": weights, "mmproj.gguf": proj}, requests)
    asyncio.run(service.download(art.id))
    assert requests == ["mmproj.gguf"]
    assert service.view(art.id).projector_status == "downloaded"
    asyncio.run(service.download(art.id))  # everything verified: no request at all
    assert requests == ["mmproj.gguf"]


def test_a_projector_with_the_wrong_hash_is_not_used_and_the_model_stays_downloaded(tmp_path, monkeypatch):
    weights, proj = b"w" * 4096, b"p" * 1024
    service, art = vlm_service(tmp_path, weights, proj, proj_sha="1" * 64)
    use_transport(monkeypatch, {"model.gguf": weights, "mmproj.gguf": proj}, [])
    with pytest.raises(ModelError):
        asyncio.run(service.download(art.id))
    view = service.view(art.id)
    assert view.status == "downloaded" and view.projector_status == "not_downloaded" and view.error
    assert service.projector_path(art.id) is None


def test_a_changed_projector_file_is_no_longer_verified(tmp_path, monkeypatch):
    weights, proj = b"w" * 4096, b"p" * 1024
    service, art = vlm_service(tmp_path, weights, proj)
    use_transport(monkeypatch, {"model.gguf": weights, "mmproj.gguf": proj}, [])
    asyncio.run(service.download(art.id))
    (service.model_dir(art.id) / "mmproj.gguf").write_bytes(b"q" * 2048)
    assert service.projector_path(art.id) is None and service.view(art.id).projector_status == "not_downloaded"


def test_disk_preflight_counts_the_projector(tmp_path, monkeypatch):
    weights, proj = b"w" * 4096, b"p" * MiB
    service, art = vlm_service(tmp_path, weights, proj)
    use_transport(monkeypatch, {"model.gguf": weights, "mmproj.gguf": proj}, [])
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * GiB + len(weights)))
    with pytest.raises(ModelError, match="not enough disk"):
        asyncio.run(service.download(art.id))


def test_a_model_without_a_projector_reports_none(tmp_path):
    service, art = vlm_service(tmp_path, b"w" * 16, None)
    assert service.view(art.id).projector_status == "none" and service.projector_path(art.id) is None


# ----------------------------------------------------------------------------- engine flags


def peer_plan():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected), shape14(), [gpu(6)], [mac(8)],
                     32 * GiB, cpu(16))
    plan = DeploymentManager._llama_names(plan, {EP5: 1})
    assert plan.device_list[0].startswith("RPC") and plan.device_list[-1] == "CUDA0"
    return plan


def cpu_plan():  # a host without a GPU in the pool: its CPU and one peer
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected), shape14(), [], [mac(8)],
                     32 * GiB, cpu(16))
    plan = DeploymentManager._llama_names(plan, {EP5: 1})
    assert plan.feasible and plan.device_list == ["RPC0"] and plan.assignments[0].device == "CPU"
    return plan


def stream_plan():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only), shape14(), [gpu(8)], [], 6 * GiB,
                     cpu(4), stream=cap14())
    assert plan.tier == "stream"
    return plan


def args_for(plan, mmproj: Path | None) -> list[str]:
    adapter = object.__new__(LlamaRpcAdapter)  # build_args reads only the spec
    spec = DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m", mmproj_path=mmproj)
    return adapter.build_args(spec, 9000)


def test_the_projector_runs_on_the_local_gpu_not_the_first_peer():
    args = args_for(peer_plan(), Path("mmproj.gguf"))
    assert args[args.index("--mmproj") + 1] == "mmproj.gguf"
    assert args[args.index("-mmdev") + 1] == "CUDA0"
    assert args.index("--rpc") < args.index("-mmdev")


def test_the_projector_runs_on_the_host_cpu_when_the_plan_has_no_local_gpu():
    args = args_for(cpu_plan(), Path("mmproj.gguf"))
    assert args[args.index("-mmdev") + 1] == "none" and "--mmproj" in args


def test_text_only_args_are_unchanged_without_a_projector():
    for plan in (peer_plan(), cpu_plan()):
        args = args_for(plan, None)
        assert "--mmproj" not in args and "-mmdev" not in args
        assert args == object.__new__(LlamaRpcAdapter).build_args(
            DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("m.gguf"), alias="m"), 9000)


def test_the_stream_tier_never_loads_a_projector():
    assert "--mmproj" not in args_for(stream_plan(), Path("mmproj.gguf"))


def gemma3_27b() -> ModelShape:
    """Gemma 3 27B Q4_K_M: 62 blocks, n_embd 5376, 32 heads over 16 KV heads of 128, tied 262208-token embedding
    (Q6_K, ~1.16 GB). Block bytes are the rest of the 16546404992-byte file split evenly: an approximation."""
    embd = 1_156_000_000
    layer = (16_546_404_992 - embd) // 62
    s = GgufSummary(architecture="gemma3", name="gemma-3-27b-it", file_type="Q4_K_M", n_layer=62, n_embd=5376,
                    n_head=32, n_head_kv=16, context_length=131072, vocab_size=262208, tensor_count=808,
                    total_tensor_bytes=16_546_404_992, layer_bytes=[layer] * 62, token_embd_bytes=embd,
                    output_bytes=embd, output_tied=True, other_bytes=0, chat_template_present=True, key_length=128,
                    value_length=128)
    return ModelShape.from_gguf(s, 4096)


def test_the_demo_pool_keeps_the_projector_on_the_g14_while_both_macs_hold_layers():
    """The demo: Gemma 3 27B on the G14 (RTX 5070 Ti Laptop, 12 GB) and two MacBook Air M4 peers, its 0.86 GB f16
    projector reserved and loaded on the G14's GPU, never on RPC0 (the first --device entry)."""
    need = 857_739_168 + PROJECTOR_COMPUTE_EST
    g14, airs, host = [gpu(11)], [mac(9, EP5), mac(9, EP6)], cpu(16)
    local, reserved_cpu = reserve_projector(g14, host, need)
    plan = make_plan(PlanRequest(model_id="gemma-3-27b-it-q4_k_m", strategy=PlanStrategy.all_selected), gemma3_27b(),
                     local, airs, 32 * GiB, reserved_cpu)
    plan = DeploymentManager._llama_names(plan, {EP5: 1, EP6: 1})
    assert plan.feasible and plan.tier == "gpu" and sorted(plan.rpc_endpoints) == [EP5, EP6]
    on = {a.device: a for a in plan.assignments}
    assert set(on) == {"RPC0", "RPC1", "CUDA0"} and all(on[d].layers for d in on) and on["CUDA0"].holds_output
    assert on["CUDA0"].total_bytes_est + need <= 11 * GiB  # the layers left the projector its room on the GPU
    assert projector_warnings(plan, g14, host, need) == []
    args = args_for(plan, Path("models/gemma-3-27b-it-q4_k_m/mmproj-google_gemma-3-27b-it-f16.gguf"))
    assert args[args.index("--device") + 1].split(",")[-1] == "CUDA0"
    assert args[args.index("--device") + 1].split(",")[0].startswith("RPC")
    assert args[args.index("-mmdev") + 1] == "CUDA0"


def test_a_projector_load_failure_is_named():
    log = ("srv load_model: failed to load multimodal model, 'C:\\m\\mmproj-F16.gguf'\n"
           "cudaMalloc failed: out of memory\n")
    assert "vision projector" in detect_failure(log)


# ----------------------------------------------------------------------------- host budget


def test_reserve_takes_the_projector_off_the_last_local_gpu():
    need = 2 * GiB
    local, host = reserve_projector([gpu(4), gpu(10)], cpu(16), need)
    assert [b.budget_bytes for b in local] == [4 * GiB, 8 * GiB] and host.budget_bytes == 16 * GiB
    assert "vision projector" in local[-1].budget_basis


def test_reserve_takes_the_projector_off_the_cpu_without_a_local_gpu():
    local, host = reserve_projector([], cpu(16), 2 * GiB)
    assert local == [] and host.budget_bytes == 14 * GiB


def test_reserve_without_a_projector_changes_nothing():
    budgets, host = [gpu(10)], cpu(16)
    assert reserve_projector(budgets, host, 0) == (budgets, host)


def test_a_plan_from_reserved_budgets_leaves_room_for_the_projector():
    need = int(1.41 * GiB) + PROJECTOR_COMPUTE_EST
    local, host = reserve_projector([gpu(6)], cpu(16), need)
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected), shape14(), local, [mac(8)],
                     32 * GiB, host)
    on_gpu = next(a for a in plan.assignments if a.device == "CUDA0")
    assert on_gpu.total_bytes_est + need <= 6 * GiB
    assert projector_warnings(plan, [gpu(6)], cpu(16), need) == []


def test_a_plan_that_fills_the_projector_device_gets_a_warning():
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected), shape14(), [gpu(6)], [mac(8)],
                     32 * GiB, cpu(16))  # planned without the reservation (a measured profile, say)
    warnings = projector_warnings(plan, [gpu(6)], cpu(16), 3 * GiB)
    assert len(warnings) == 1 and "CUDA0" in warnings[0] and "vision projector" in warnings[0]


def test_the_stream_tier_plan_says_images_are_off():
    warnings = projector_warnings(stream_plan(), [], cpu(16), GiB)
    assert len(warnings) == 1 and "stream tier" in warnings[0]


# ----------------------------------------------------------------------------- gateway planning path


PROJ = ModelFile(name="mmproj.gguf", url="https://example.invalid/mmproj.gguf", sha256="0" * 64, size_bytes=GiB)


class VisionModels(ExplicitModels):
    def __init__(self, status: str) -> None:
        self.status = status

    def view(self, model_id: str):
        v = super().view(model_id)
        return v.model_copy(update={"artifact": v.artifact.model_copy(update={"mmproj": PROJ}),
                                    "projector_status": self.status})


def plan_with(store, events, tmp_path, models):
    mgr = DeploymentManager(store, events, Nodes([explicit_mac()], {MAC_A: report()}), models, LeaseManager(store),
                            object(), Supervisor(tmp_path / "m.json"))  # type: ignore[arg-type]
    return mgr.plan(PlanRequest(model_id=MODEL))


def gpu_budget(plan) -> int:
    return next(a.budget_bytes for a in plan.assignments if a.device == "CUDA0")


def test_the_gateway_reserves_a_downloaded_projector_on_the_host_gpu(store, events, tmp_path):
    plain = plan_with(store, events, tmp_path, ExplicitModels())
    vision = plan_with(store, events, tmp_path, VisionModels("downloaded"))
    missing = plan_with(store, events, tmp_path, VisionModels("not_downloaded"))
    assert gpu_budget(vision) == gpu_budget(plain) - GiB - PROJECTOR_COMPUTE_EST
    assert gpu_budget(missing) == gpu_budget(plain)  # not loaded: no reservation
    assert "mmproj" not in plain.engine_options
    assert vision.engine_options["mmproj"] is True and missing.engine_options["mmproj"] is False
    assert any("~2.0 GiB of CUDA0 held back for the vision projector" in r for r in vision.reasons)
    assert not any("vision projector" in r for r in [*missing.reasons, *plain.reasons])


class SpecModels:
    def projector_path(self, model_id: str) -> Path:
        return Path("models") / model_id / "mmproj.gguf"


@pytest.mark.parametrize(("options", "loads"), [({"mmproj": True}, True), ({"mmproj": False}, False), ({}, True)])
def test_the_plan_decides_whether_the_projector_loads(options, loads):
    mgr = object.__new__(DeploymentManager)
    mgr._models = SpecModels()  # _projector_for reads only the model service
    plan = peer_plan().model_copy(update={"engine_options": options})
    assert (mgr._projector_for(plan) is not None) is loads  # a plan made before the projector came stays text-only
