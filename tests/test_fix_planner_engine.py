"""Regressions: proportional split overshoot, greedy output-head placement, GGUF K/V head length,
engine variant resolution, verified model adoption and complete-.part download resume."""

import dataclasses
import hashlib
import io
import json
import zipfile

import httpx
import pytest
from hypercompute import config, engine
from hypercompute.schemas import DeviceBudget, GgufSummary, ModelArtifact, ModelFile, PlanRequest, PlanStrategy
from hypercompute.services import planner
from hypercompute.services.models import ModelService
from hypercompute.services.planner import ModelShape, make_plan
from hypercompute.units import GiB, MiB


def summary(n_layer=24, layer_bytes=int(10.5 * MiB), n_embd=896, n_head=14, n_head_kv=2, output_bytes=89 * MiB,
            **extra) -> GgufSummary:
    return GgufSummary(architecture="qwen2", name="m", file_type="Q4_K_M", n_layer=n_layer, n_embd=n_embd,
                       n_head=n_head, n_head_kv=n_head_kv, context_length=32768, vocab_size=151936, tensor_count=1,
                       total_tensor_bytes=n_layer * layer_bytes, layer_bytes=[layer_bytes] * n_layer,
                       token_embd_bytes=output_bytes, output_bytes=output_bytes, output_tied=False, other_bytes=0,
                       chat_template_present=True, **extra)


def dev(node, device, nbytes, endpoint=None, pool="vram:CUDA0", kind="discrete_vram") -> DeviceBudget:
    return DeviceBudget(node_id=node, device=device, description=device, endpoint=endpoint, pool_id=pool,
                        pool_kind=kind, budget_bytes=int(nbytes), budget_basis="test")


def peer(i: int, nbytes: int) -> DeviceBudget:
    ep = f"10.0.0.{i}:47052"
    return dev(f"n{i}", f"{ep}#0", nbytes, ep, "uma", "unified")


# ----------------------------------------------------------------------------- 16: proportional overshoot


@pytest.mark.parametrize("caps", [[100, 1, 1], [100, 0, 0], [0, 0, 0, 100], [5, 5, 5], [1, 1]])
def test_proportional_counts_never_exceed_n_layer(monkeypatch, caps):
    shape = ModelShape.from_gguf(summary(), 4096)
    devs = [dev(f"n{i}", f"D{i}", GiB) for i in range(len(caps))]
    by_dev = {d.device: c for d, c in zip(devs, caps, strict=True)}
    monkeypatch.setattr(planner, "_layer_capacity", lambda _s, b, holds_output: by_dev[b.device])
    counts = planner._counts_proportional(shape, devs)
    assert sum(counts) == shape.n_layer and min(counts) >= 1


def test_proportional_counts_with_more_devices_than_layers_leave_trailing_devices_empty():
    shape = ModelShape.from_gguf(summary(n_layer=2), 4096)
    counts = planner._counts_proportional(shape, [dev(f"n{i}", f"D{i}", GiB) for i in range(3)])
    assert counts == [1, 1, 0]


def test_two_zero_budget_peers_make_run_together_infeasible_not_crash():
    shape = ModelShape.from_gguf(summary(), 4096)
    local, peers = [dev("local", "CUDA0", 10 * GiB)], [peer(5, 0), peer(6, 0)]
    together = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected), shape, local, peers, 16 * GiB)
    assert not together.feasible and together.fit_label == "does_not_fit"
    assert len(together.assignments) == 3 and sum(together.tensor_split) == 25
    assert any("over budget" in r for r in together.reasons)
    fastest = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.fastest_fit), shape, local, peers, 16 * GiB)
    assert fastest.feasible and fastest.device_list == ["CUDA0"]


# ----------------------------------------------------------------------------- 21: greedy output head


def test_fastest_fit_keeps_the_head_local_and_sends_only_the_overflow_to_the_peer():
    shape = ModelShape.from_gguf(summary(n_layer=28, layer_bytes=150 * MiB, n_embd=3584, n_head=28, n_head_kv=4,
                                         output_bytes=420 * MiB), 4096)
    overhead = planner.OVERHEAD_BY_KIND["cuda"]
    no_head = sum(shape.layer_bytes) + shape.n_layer * shape.kv_per_layer + shape.compute_est + overhead
    local = [dev("local", "CUDA0", no_head + shape.output_bytes // 2)]  # every layer fits, the head does not
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.fastest_fit), shape, local,
                     [peer(5, 16 * GiB)], 16 * GiB)
    assert plan.feasible and plan.fit_label == "estimated_fit"
    # peers go first and the local GPU last, so the head (and the logits) stay on CUDA0; the peer gets the 2 layers
    # the head displaced
    assert [len(a.layers) for a in plan.assignments] == [2, 26] and plan.tensor_split == [2, 27]
    assert plan.assignments[0].endpoint == "10.0.0.5:47052" and not plan.assignments[0].holds_output
    assert plan.assignments[-1].holds_output and plan.assignments[-1].endpoint is None


# ----------------------------------------------------------------------------- 22: GGUF K/V head length


def test_kv_estimate_uses_gguf_key_and_value_length():
    assert ModelShape.from_gguf(summary(), 4096).kv_per_layer == 2 * MiB  # observed for Qwen2.5-0.5B
    qwen3 = summary(n_layer=28, n_embd=1024, n_head=16, n_head_kv=8)  # n_embd/n_head = 64, real head dim 128
    assert ModelShape.from_gguf(qwen3, 4096).kv_per_layer == 4096 * 8 * (64 + 64) * 2
    real = qwen3.model_copy(update={"key_length": 128, "value_length": 128})
    assert ModelShape.from_gguf(real, 4096).kv_per_layer == 4096 * 8 * (128 + 128) * 2 == 16 * MiB


# ----------------------------------------------------------------------------- 18: installed variant choice


@pytest.fixture
def windows_host(monkeypatch):
    monkeypatch.delenv(engine.BIN_DIR_ENV, raising=False)
    monkeypatch.setattr(engine, "host_os", lambda: "windows")
    monkeypatch.setattr(engine, "host_arch", lambda: "x86_64")
    return engine.load_lock()


def fake_install(lock: engine.EngineLock, key: str, installed_at: str) -> None:
    d = config.paths().engines / f"llama-{lock.release}-{key}"
    d.mkdir(parents=True)
    meta = {"release": lock.release, "variant": key, "bin_subdir": "", "installed_at": installed_at}
    (d / engine.MARKER).write_text(json.dumps(meta), encoding="utf-8")


def set_active(lock: engine.EngineLock, key: str) -> None:
    (config.paths().engines / engine.ACTIVE_FILE).write_text(
        json.dumps({"release": lock.release, "variant": key}), encoding="utf-8")


def test_find_installed_prefers_accelerator_build_over_cpu(windows_host):
    lock = windows_host
    fake_install(lock, "windows-x64-cuda13.4", "2026-10-20T10:00:00+00:00")
    fake_install(lock, "windows-x64-cpu", "2026-10-20T11:00:00+00:00")  # newer, still loses to the GPU build
    assert engine.find_installed(lock).variant == "windows-x64-cuda13.4"
    set_active(lock, "windows-x64-vulkan")  # active names a variant that is not installed: ignored
    assert engine.find_installed(lock).variant == "windows-x64-cuda13.4"
    set_active(lock, "windows-x64-cpu")  # an explicit cpu install stays the one in use
    found = engine.find_installed(lock)
    assert found.variant == "windows-x64-cpu" and found.accelerator == "cpu"


def test_find_installed_prefers_newest_among_accelerator_builds(monkeypatch):
    monkeypatch.delenv(engine.BIN_DIR_ENV, raising=False)
    monkeypatch.setattr(engine, "host_os", lambda: "linux")
    monkeypatch.setattr(engine, "host_arch", lambda: "x86_64")
    lock = engine.load_lock()
    fake_install(lock, "linux-x64-cpu", "2026-10-20T12:00:00+00:00")
    fake_install(lock, "linux-x64-cuda12.8", "2026-10-20T10:00:00+00:00")
    fake_install(lock, "linux-x64-cuda13.4", "2026-10-20T11:00:00+00:00")
    assert engine.find_installed(lock).variant == "linux-x64-cuda13.4"


def test_install_activates_the_variant_it_installed(windows_host, monkeypatch, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("build/bin/llama-server.exe", "x")
    archive = buf.getvalue()

    def fake_download(url, dest, sha256, size, progress=None):
        dest.write_bytes(archive)
        return dest

    monkeypatch.setattr(engine, "download_verified", fake_download)
    lock = windows_host
    tiny = (engine.Asset(name="a.zip", url="https://example.invalid/a.zip", sha256="0" * 64, size=len(archive)),)
    variants = {k: dataclasses.replace(v, assets=tiny) for k, v in lock.variants.items()}
    lock = dataclasses.replace(lock, variants=variants)
    for key in ("windows-x64-cpu", "windows-x64-cuda13.4", "windows-x64-cpu"):
        installed = engine.install(lock.variants[key], lock)
        found = engine.find_installed(lock)
        assert found.variant == key and found.bin_dir == installed.bin_dir
        assert (found.bin_dir / "llama-server.exe").is_file()


# ----------------------------------------------------------------------------- 19: adopt_cached


@pytest.fixture
def tiny_model(tmp_path):
    payload = bytes(range(256)) * 16 + b"tail"  # 4100 bytes
    art = ModelArtifact(id="tiny", display_name="Tiny", family="t", params_b=0.001, quantization="Q4_K_M",
                        repo="x/y", revision="main", license="mit",
                        files=[ModelFile(name="tiny.gguf", url="https://example.invalid/tiny.gguf",
                                         sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))])
    svc = ModelService(models_dir=tmp_path / "models")
    svc._artifacts = {art.id: art}
    src = tmp_path / "downloads" / "tiny.gguf"
    src.parent.mkdir()
    src.write_bytes(payload)
    dst = svc.model_dir("tiny") / "tiny.gguf"
    dst.parent.mkdir(parents=True)
    return svc, src, dst, payload


def test_adopt_cached_replaces_truncated_destination(tiny_model):
    svc, src, dst, payload = tiny_model
    dst.write_bytes(payload[:100])  # remnant of an interrupted copy
    assert svc.adopt_cached("tiny", [src])
    assert dst.read_bytes() == payload and svc._verified("tiny")
    assert not list(dst.parent.glob("*.adopting"))


def test_marker_over_wrong_size_file_is_not_verified(tiny_model):
    svc, _src, dst, payload = tiny_model
    dst.write_bytes(payload[:100])
    svc._write_marker("tiny")  # a marker written by the old adopt_cached over a truncated file
    assert not svc._verified("tiny")


# ----------------------------------------------------------------------------- 20: complete .part resume


def test_complete_part_is_finalized_without_a_range_request(tmp_path):
    payload = b"model-bytes" * 100
    good = hashlib.sha256(payload).hexdigest()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "range" in request.headers:
            return httpx.Response(416)  # RFC 7233: bytes=<size>- is unsatisfiable
        return httpx.Response(200, content=payload)

    def client() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    dest, part = tmp_path / "m.gguf", tmp_path / "m.gguf.part"
    part.write_bytes(payload)  # interrupted after the last chunk, before the rename
    assert engine.download_verified("https://example.invalid/m", dest, good, len(payload), client=client()) == dest
    assert dest.read_bytes() == payload and not part.exists() and seen == []

    dest.unlink()
    part.write_bytes(b"x" * len(payload))  # full size but corrupt: discarded and fetched fresh
    engine.download_verified("https://example.invalid/m", dest, good, len(payload), client=client())
    assert dest.read_bytes() == payload and not part.exists()
    assert len(seen) == 1 and "range" not in seen[0].headers
