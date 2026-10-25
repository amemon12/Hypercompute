"""vLLM adapter: plan sizing, launch command, readiness, streaming and stop (the installer: test_vllm_install.py).

Portable: no WSL, GPU or network. nvidia-smi rows and `wsl -l -v` bytes are captured from the G14; the SSE streams
and startup logs are synthetic but follow vLLM v0.32.0's formats (chat_completion/serving.py, the log format strings
cited in adapters/vllm.py).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from factories import contribution, make_inventory
from hypercompute import config, schemas
from hypercompute.adapters import registry
from hypercompute.adapters import vllm as adapter_mod
from hypercompute.adapters import vllm_output as vout
from hypercompute.adapters.base import DeploymentHandle, DeploymentSpec, EngineAdapterError, InferenceRequest
from hypercompute.installers import vllm as installer
from hypercompute.schemas import ModelArtifact, ModelFile, NodeState, NodeView, PlanRequest, utcnow
from hypercompute.services import models as models_mod
from hypercompute.services.models import load_manifest
from hypercompute.units import GiB, MiB

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]
AWQ_SHA = "b25037543e9394b818fdfca67ab2a00ecc7dd641"  # HF API main of Qwen/Qwen2.5-7B-Instruct-AWQ, 2026-10-21
AWQ_BYTES = 3996422976 + 1574406784  # its two safetensors shards (HF API ?blobs=true)


def rows(name: str) -> list[adapter_mod.GpuMemory]:
    return adapter_mod.parse_nvidia_smi((FIXTURES / name).read_text(encoding="utf-8"))


def sse(name: str) -> bytes:
    """An SSE fixture as vLLM sends it (LF); a Windows checkout with core.autocrlf may have turned it into CRLF."""
    return (FIXTURES / name).read_bytes().replace(b"\r\n", b"\n")


IDLE = "vllm_nvidia_smi_g14_idle.txt"
BUSY = "vllm_nvidia_smi_g14_llama_server.txt"


def roomy_disk(rec) -> tuple[int, str]:
    return 100 * GiB, "C: (test)"


def tight_disk(rec) -> tuple[int, str]:
    return AWQ_BYTES, "C: (holds WSL2 distro Ubuntu's virtual disk)"  # the checkpoint fits, its headroom does not


def awq(**over) -> ModelArtifact:
    base = dict(id="qwen2.5-7b-instruct-awq", display_name="Qwen2.5 7B Instruct (AWQ)", family="qwen2",
                params_b=7.6, quantization="AWQ-4bit", format="hf", engines=["vllm"],
                logical_id="qwen2.5-7b-instruct", size_bytes=AWQ_BYTES, repo="Qwen/Qwen2.5-7B-Instruct-AWQ",
                revision=AWQ_SHA, license="apache-2.0", files=[])
    return ModelArtifact(**{**base, **over})


def plan_of(art: ModelArtifact | None = None, req: PlanRequest | None = None, remote=(), query=None,
            disk=roomy_disk, local: NodeView | None = None):
    return adapter_mod.plan(req or PlanRequest(model_id="m"), art or awq(), local or local_node(), list(remote),
                            ctx(), query=query or (lambda: rows(IDLE)), disk=disk)


def record(platform: str = "wsl2", **over) -> installer.VllmInstall:
    base = dict(platform=platform, distro="Ubuntu" if platform == "wsl2" else None,
                root="/home/u/.hypercompute-vllm", venv="/home/u/.hypercompute-vllm/venv", vllm="0.32.0",
                torch="2.13.0+cu130", torch_cuda="13.0", capability="12.0",
                device="NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    return installer.VllmInstall(**{**base, **over})


def local_node(state: NodeState = NodeState.eligible, **inventory) -> NodeView:
    return NodeView(id="local", display_name="g14", is_local=True, session_id=None, state=state, state_reason=None,
                    joined_at=utcnow(), last_heartbeat_at=utcnow(), heartbeat_age_s=0.0,
                    inventory=make_inventory(**inventory), contribution=contribution(["CUDA0"], max_gib=12),
                    worker=None, worker_state=None, verification=None)


class FakeProc:
    def __init__(self, log: str = "", alive: bool = True, pid: int = 4242) -> None:
        self.log, self._alive, self.pid = log, alive, pid

    def alive(self) -> bool:
        return self._alive

    def exit_code(self) -> int | None:
        return None if self._alive else 1

    def read_log(self, max_bytes: int = 0) -> str:
        return self.log


class FakeSupervisor:
    def __init__(self) -> None:
        self.started: list[tuple] = []
        self.procs: dict[str, FakeProc] = {}
        self.stopped: list[str] = []

    def start(self, key, exe, args, cwd, log_path, env=None):
        self.started.append((key, exe, list(args), cwd, log_path, dict(env or {})))
        self.procs[key] = FakeProc()
        return self.procs[key]

    def get(self, key):
        return self.procs.get(key)

    def stop(self, key, grace_s=8.0):
        self.stopped.append(key)
        self.procs.pop(key, None)
        return None


def ctx(sup: FakeSupervisor | None = None) -> registry.AdapterContext:
    paths = config.paths()
    return registry.AdapterContext(supervisor=sup or FakeSupervisor(), logs_dir=paths.home / "logs", paths=paths)


def installed(monkeypatch, platform: str = "wsl2") -> installer.VllmInstall:
    monkeypatch.setattr(adapter_mod.engine, "host_os", lambda: "windows" if platform == "wsl2" else "linux")
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows" if platform == "wsl2" else "linux")
    rec = record(platform)
    installer.write_install(config.paths(), rec)
    return rec


# ----------------------------------------------------------------------------- sizing


def test_parse_nvidia_smi_fixture_and_bad_rows():
    (gpu,) = rows(IDLE)
    assert (gpu.index, gpu.total_bytes, gpu.free_bytes) == (0, 12227 * MiB, 11944 * MiB)
    assert adapter_mod.parse_nvidia_smi("0, GPU, [N/A], 1, 2\n1, A, B, 100, 50, 50\n")[0].name == "A, B"


def test_utilization_from_free_memory():
    total = 12227 * MiB
    assert adapter_mod.utilization(11944 * MiB - adapter_mod.VRAM_HEADROOM, total) == 0.89
    assert adapter_mod.utilization(31 * GiB, 32 * GiB) == 0.90  # capped, never vLLM's 0.92 default
    assert adapter_mod.utilization(731 * MiB - adapter_mod.VRAM_HEADROOM, total) == 0.0
    assert adapter_mod.reserved_bytes(total, 0.89) == -(-total * 89 // 100)


def test_kv_bytes_per_token():
    assert adapter_mod.kv_bytes_per_token(awq())[0] == 56 * 1024  # 28 x 2 x 4 x 128 x 2 B (research §2.3)
    est, basis = adapter_mod.kv_bytes_per_token(awq(logical_id="other", params_b=14.0))
    assert est == 14 * 20 * 1024 and basis.startswith("estimate")


def test_serve_args_are_exact_and_carry_no_key():
    args = adapter_mod.serve_args("Qwen/Qwen2.5-7B-Instruct-AWQ", AWQ_SHA, "alias", 51234, 0.89, 4096, 8)
    assert args == ["serve", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--revision", AWQ_SHA, "--served-model-name", "alias",
                    "--host", "127.0.0.1", "--port", "51234", "--gpu-memory-utilization", "0.89",
                    "--max-model-len", "4096", "--max-num-seqs", "8", "--generation-config", "vllm",
                    "--disable-uvicorn-access-log"]
    assert not any("api" in a.lower() for a in args)


# ----------------------------------------------------------------------------- plan


def test_plan_fits_7b_awq_on_an_idle_12gib_gpu(monkeypatch):
    rec = installed(monkeypatch)
    req = PlanRequest(model_id="qwen2.5-7b-instruct-awq", objective="throughput")
    plan = plan_of(req=req)
    assert plan.feasible and plan.engine == "vllm" and plan.tier == "gpu" and plan.fit_label == "estimated_fit"
    assert plan.device_list == [] and plan.rpc_endpoints == [] and plan.tensor_split == []
    opts = plan.engine_options
    assert opts["gpu_memory_utilization"] == 0.89 and opts["max_num_seqs"] == 16 and opts["engine_managed"]
    assert opts["args"][:2] == ["serve", "Qwen/Qwen2.5-7B-Instruct-AWQ"] and opts["revision"] == AWQ_SHA
    assert opts["env"]["HF_HOME"] == rec.hf_home and opts["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    (a,) = plan.assignments
    assert a.device == "CUDA0" and a.pool_id == "vram:CUDA0" and len(a.layers) == 28 and a.fits
    assert a.total_bytes_est == opts["reserved_bytes"] <= a.budget_bytes
    assert opts["need_bytes"] == AWQ_BYTES + 4096 * 56 * 1024 + adapter_mod.RUNTIME_OVERHEAD  # size_bytes = weights
    assert opts["download_bytes"] == AWQ_BYTES and opts["disk_free_bytes"] == 100 * GiB
    assert any("inside WSL2 (Ubuntu)" in w and AWQ_SHA[:12] in w and "C: (test) has 100.0 GiB free" in w
               for w in plan.warnings)


def test_plan_does_not_fit_while_llama_server_holds_the_gpu(monkeypatch):
    installed(monkeypatch)
    monkeypatch.setattr(adapter_mod, "gpu_holders", lambda: ["llama-server.exe (pid 68572)"])
    plan = plan_of(query=lambda: rows(BUSY))
    assert not plan.feasible and plan.fit_label == "does_not_fit"
    assert any("llama-server.exe (pid 68572)" in r and "stop the llama.cpp deployment" in r for r in plan.reasons)


def test_vram_miss_candidate_keeps_the_memory_breakdown(monkeypatch):
    installed(monkeypatch)
    monkeypatch.setattr(adapter_mod, "gpu_holders", lambda: [])
    (cand,) = plan_of(query=lambda: rows(BUSY)).candidates
    assert not cand.feasible and cand.reason.startswith("vLLM 0.32.0 on CUDA0")
    assert "weights + " in cand.reason and "reserved at --gpu-memory-utilization 0.00" in cand.reason
    assert cand.reason.endswith("does not fit: free GPU memory or lower ctx_size")


def test_plan_reasons_when_it_cannot_serve(monkeypatch):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    monkeypatch.setattr(adapter_mod.engine, "host_os", lambda: "windows")
    plan = plan_of()
    assert not plan.feasible and "engine install --engine vllm" in plan.reasons[0]
    installer.write_install(config.paths(), record())
    plan = plan_of(awq(format="gguf", engines=[], size_bytes=None))
    assert any("use llama.cpp for GGUF" in r for r in plan.reasons) and any("size_bytes" in r for r in plan.reasons)
    plan = plan_of(local=local_node(system="macos"))
    assert any("no NVIDIA GPU" in r for r in plan.reasons)


def test_plan_ignores_peers_and_falls_back_to_inventory(monkeypatch):
    installed(monkeypatch)

    def broken():
        raise EngineAdapterError("gpu_query_failed", "nvidia-smi failed: boom")

    plan = plan_of(remote=[local_node()], query=broken)
    assert plan.feasible  # the inventory reading (11026 MiB free) still fits
    assert any("peer(s) are not used" in w for w in plan.warnings) and any("inventory" in w for w in plan.warnings)


def test_plan_needs_disk_room_for_the_first_download_only(monkeypatch):
    rec = installed(monkeypatch)
    plan = plan_of(disk=tight_disk)
    assert not plan.feasible and plan.fit_label == "does_not_fit" and not plan.candidates[0].feasible
    assert any(r.startswith("does not fit on disk") and "C: (holds WSL2" in r and "free at least 2.0 GiB" in r
               for r in plan.reasons)
    assert plan.candidates[0].reason.startswith("does not fit on disk")  # VRAM fits: no VRAM breakdown there
    assert plan.engine_options["download_bytes"] == AWQ_BYTES and plan.assignments[0].fits  # VRAM still fits
    installer.remember_served(config.paths(), rec, "Qwen/Qwen2.5-7B-Instruct-AWQ", AWQ_SHA)
    plan = plan_of(disk=tight_disk)
    assert plan.feasible and plan.engine_options["download_bytes"] == 0
    assert any("reuses the copy" in w for w in plan.warnings)
    other = plan_of(awq(revision="0" * 40), disk=lambda rec: (None, "WSL2 distro Ubuntu's virtual disk"))
    assert other.feasible and other.engine_options["download_bytes"] == AWQ_BYTES  # unknown free space never blocks


def test_plan_warns_when_free_disk_cannot_be_read(monkeypatch):
    rec = installed(monkeypatch)
    unknown = (None, "WSL2 distro Ubuntu's virtual disk")
    plan = plan_of(disk=lambda r: unknown)
    assert plan.feasible and plan.engine_options["disk_free_bytes"] is None
    assert any("was not checked" in w and "WSL2 distro Ubuntu's virtual disk" in w and "7.2 GiB" in w
               for w in plan.warnings)  # 5.2 GiB checkpoint + 2 GiB headroom
    installer.remember_served(config.paths(), rec, "Qwen/Qwen2.5-7B-Instruct-AWQ", AWQ_SHA)
    assert not any("not checked" in w for w in plan_of(disk=lambda r: unknown).warnings)  # nothing to download


def awq_handle() -> DeploymentHandle:  # prepare()'s display args for the AWQ plan, read back by wait_ready
    return DeploymentHandle("dep1", "deployment:dep1", "http://127.0.0.1:1", Path("vllm-dep1.log"), 1,
                            ["wsl.exe", "-d", "Ubuntu", "--", "vllm", "serve", awq().repo, "--revision", AWQ_SHA,
                             "--served-model-name", "m"])


@pytest.mark.anyio
async def test_served_record_is_per_distro_and_cache_dir(monkeypatch):
    installed(monkeypatch)
    adapter = adapter_mod.VllmAdapter(record(), ctx())
    adapter._remember_cached(awq_handle())  # what wait_ready records once /health and /v1/models passed
    assert plan_of(disk=tight_disk).engine_options["download_bytes"] == 0  # same distro and HF_HOME: reused
    for moved in (record(distro="Ubuntu-24.04"), record(root="/srv/hc-vllm")):  # another distro, another cache dir
        monkeypatch.setattr(installer, "read_install", lambda paths, rec=moved: rec)
        plan = plan_of(disk=tight_disk)
        assert not plan.feasible and plan.engine_options["download_bytes"] == AWQ_BYTES
        assert not any("reuses the copy" in w for w in plan.warnings)
    await adapter.aclose()


def test_manifest_entry_is_engine_managed_and_sized_by_size_bytes(monkeypatch, tmp_path):
    (art,) = load_manifest(FIXTURES / "vllm_manifest.yaml")
    assert (art.format, art.engines, art.files, art.logical_id) == ("hf", ["vllm"], [], "qwen2.5-7b-instruct")
    assert (art.repo, art.revision, art.size_bytes, art.license) == ("Qwen/Qwen2.5-7B-Instruct-AWQ", AWQ_SHA,
                                                                     AWQ_BYTES, "apache-2.0")
    monkeypatch.setattr(models_mod, "load_manifest", lambda path=None: [art])
    view = models_mod.ModelService(models_dir=tmp_path).view(art.id)
    assert view.status == "engine_managed" and view.total_bytes == AWQ_BYTES  # nothing for Hypercompute to download
    installed(monkeypatch)
    plan = plan_of(art, PlanRequest(model_id=art.id, objective="throughput"))
    assert plan.feasible and plan.engine_options["engine_managed"] and plan.assignments[0].weight_bytes == AWQ_BYTES
    assert not any("manifest lists files" in w for w in plan.warnings)
    shard = ModelFile(name="model.safetensors", url="https://example.invalid/m", sha256="a" * 64, size_bytes=1)
    plan = plan_of(art.model_copy(update={"files": [shard]}))
    assert plan.assignments[0].weight_bytes == AWQ_BYTES and not plan.engine_options["engine_managed"]
    assert any("manifest lists files" in w and "files: []" in w for w in plan.warnings)


# ----------------------------------------------------------------------------- launch / validate / stop


def spec_for(plan) -> DeploymentSpec:
    return DeploymentSpec(deployment_id="dep1", plan=plan, model_path=Path("unused"), alias=plan.model_id,
                          artifact=awq())


def test_windows_launch_keeps_the_key_out_of_argv(monkeypatch):
    rec = installed(monkeypatch)
    monkeypatch.setenv("WSLENV", "WT_SESSION:VLLM_API_KEY/p")
    args = adapter_mod.serve_args("repo", AWQ_SHA, "alias", 5000, 0.89, 4096, 8)
    launch = adapter_mod.launch_command(rec, r"C:\Windows\System32\wsl.exe", "/home/u/.hypercompute-vllm/run/d.pid",
                                        args, 0, "SECRET-KEY", Path("logs"))
    assert launch.argv[:8] == ["-d", "Ubuntu", "--cd", "~", "--exec", "setsid", "-w", "env"]
    assert "HF_HOME=/home/u/.hypercompute-vllm/hf" in launch.argv and "CUDA_VISIBLE_DEVICES=0" in launch.argv
    i = launch.argv.index("-c")
    assert launch.argv[i + 1] == vout.LAUNCH and launch.argv[i + 2].endswith("/run/d.pid")
    assert launch.argv[i + 3:i + 6] == ["/home/u/.hypercompute-vllm/venv/bin/python", "-m", adapter_mod.CLI_MODULE]
    assert all("SECRET" not in a for a in launch.argv + launch.display)
    assert launch.env["VLLM_API_KEY"] == "SECRET-KEY" and launch.env["WSLENV"] == "WT_SESSION:VLLM_API_KEY/u"
    assert launch.display[:5] == ["wsl.exe", "-d", "Ubuntu", "--", "vllm"]


def test_linux_launch_is_a_plain_process(monkeypatch):
    rec = installed(monkeypatch, "linux")
    launch = adapter_mod.launch_command(rec, None, "/x.pid", ["serve", "m"], 1, "K", Path("logs"))
    assert launch.exe == Path("/home/u/.hypercompute-vllm/venv/bin/python")
    assert launch.argv == ["-m", adapter_mod.CLI_MODULE, "serve", "m"] and launch.cwd == Path(rec.root)
    assert launch.env["VLLM_API_KEY"] == "K" and launch.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert launch.env["PYTHONPATH"] == ""


@pytest.mark.anyio
async def test_validate_rereads_free_memory_and_prepare_launches(monkeypatch):
    installed(monkeypatch)
    monkeypatch.setattr(installer, "cache_disk", roomy_disk)  # validate re-reads free disk too
    plan = plan_of()
    sup = FakeSupervisor()
    busy = adapter_mod.VllmAdapter(record(), ctx(sup), query=lambda: rows(BUSY))
    monkeypatch.setattr(adapter_mod, "gpu_holders", lambda: ["llama-server.exe (pid 1)"])
    result = await busy.validate(spec_for(plan))
    assert not result.ok and "llama-server.exe (pid 1)" in result.problems[0] and "fix:" in result.problems[0]
    adapter = adapter_mod.VllmAdapter(record(), ctx(sup), query=lambda: rows(IDLE))
    assert (await adapter.validate(spec_for(plan))).ok
    handle = await adapter.prepare(spec_for(plan))
    key, exe, argv, _, log_path, env = sup.started[0]
    assert key == "deployment:dep1" and Path(exe).name.lower().startswith("wsl") and "--served-model-name" in argv
    assert env["VLLM_API_KEY"] == handle.api_key and handle.api_key not in " ".join(argv)
    assert handle.base_url.startswith("http://127.0.0.1:") and adapter_mod.served_name(handle) == "m"
    assert log_path.name == "vllm-dep1.log"
    await adapter.aclose()
    await busy.aclose()


@pytest.mark.anyio
async def test_validate_rechecks_disk_room_for_a_stored_plan(monkeypatch):
    installed(monkeypatch)
    plan = plan_of()  # planned while the disk had room
    monkeypatch.setattr(installer, "cache_disk", tight_disk)  # it filled up before the deploy
    adapter = adapter_mod.VllmAdapter(record(), ctx(), query=lambda: rows(IDLE))
    (problem,) = (await adapter.validate(spec_for(plan))).problems  # the GPU is idle: only the disk is short
    assert problem.startswith("does not fit on disk") and "free at least 2.0 GiB" in problem
    adapter._remember_cached(awq_handle())  # once the checkpoint is in vLLM's cache a full disk does not matter
    cached = plan_of()
    assert cached.engine_options["download_bytes"] == 0 and (await adapter.validate(spec_for(cached))).ok
    installer.write_install(config.paths(), record())  # a reinstall forgets the cache: that plan's reuse is stale
    assert not (await adapter.validate(spec_for(cached))).ok
    await adapter.aclose()


@pytest.mark.anyio
async def test_max_concurrency_is_the_launched_max_num_seqs(monkeypatch):
    installed(monkeypatch)
    sup = FakeSupervisor()
    adapter = adapter_mod.VllmAdapter(record(), ctx(sup), query=lambda: rows(IDLE))
    for objective, seqs in (("throughput", 16), ("interactive", 8)):
        plan = plan_of(req=PlanRequest(model_id="m", objective=objective))
        assert adapter.max_concurrency(plan) == seqs
        await adapter.prepare(spec_for(plan))
        assert adapter_mod._arg_after(sup.started[-1][2], "--max-num-seqs") == str(seqs)
    for bad in [{"max_num_seqs": v} for v in (0, -3, None, "x")] + [{}]:
        assert adapter.max_concurrency(plan.model_copy(update={"engine_options": bad})) == 1
    await adapter.aclose()


@pytest.mark.anyio
async def test_stop_signals_inside_wsl_then_reaps(monkeypatch):
    installed(monkeypatch)
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"graceful\n", stderr=b"")

    sup = FakeSupervisor()
    adapter = adapter_mod.VllmAdapter(record(), ctx(sup), runner=runner)
    handle = DeploymentHandle("dep1", "deployment:dep1", "http://127.0.0.1:1", Path("x.log"), 1, ["vllm"])
    result = await adapter.stop(handle)
    assert result.clean and "graceful" in result.detail and sup.stopped == ["deployment:dep1"]
    assert calls[0][calls[0].index("-c") + 1] == vout.STOP and calls[0][-1] == "30"
    adapter._run = lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout=b"still-running\n", stderr=b"")
    assert not (await adapter.stop(handle)).clean
    await adapter.aclose()


def crashed(plan, dep_id: str = "dep1abcd") -> schemas.Deployment:
    """A deployment the gateway marked failed at restart (what reconcile_on_start hands to cleanup_orphan)."""
    return schemas.Deployment(id=dep_id, plan=plan, state=schemas.DeploymentState.failed, history=[],
                              created_at=utcnow())


class WslRunner:
    """subprocess.run stand-in: `wsl -l --running -q` answers `running`, the in-distro STOP script answers `stop`."""

    def __init__(self, running: tuple[int, bytes] | Exception, stop: tuple[int, bytes] = (0, b"killed\n")) -> None:
        self.running, self.stop, self.calls = running, stop, []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        answer = self.running if "--running" in argv else self.stop
        if isinstance(answer, Exception):
            raise answer
        return subprocess.CompletedProcess(argv, answer[0], stdout=answer[1], stderr=b"")


@pytest.mark.anyio
async def test_cleanup_orphan_stops_the_server_a_crashed_gateway_left_in_wsl(monkeypatch):
    """Killing wsl.exe at restart does not stop vLLM inside the distro once its watchdog is gone: the hook must."""
    installed(monkeypatch)
    plan, run = plan_of(), WslRunner((0, b"docker-desktop\r\nUbuntu\r\n"))
    adapter = adapter_mod.VllmAdapter(record(), ctx(), runner=run)
    adapter._wsl = r"C:\Windows\System32\wsl.exe"
    ok, detail = await adapter.cleanup_orphan(crashed(plan))  # type: ignore[attr-defined]
    assert ok and detail == "stopped the vLLM server it left running: vLLM in WSL (Ubuntu): killed"
    listing, stop = run.calls
    assert listing == [r"C:\Windows\System32\wsl.exe", "-l", "--running", "-q"]  # lists without starting the VM
    assert stop[:3] == [r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu"] and stop[stop.index("-c") + 1] == vout.STOP
    assert stop[-2:] == ["/home/u/.hypercompute-vllm/run/dep1abcd.pid", "30"]  # stop()'s TERM, 30 s, KILL path
    for word in ("no-pidfile", "already-exited", "not-vllm", "bad-pidfile"):  # no vLLM server of it is left
        run.stop = (0, f"{word}\n".encode())
        ok, detail = await adapter.cleanup_orphan(crashed(plan))  # type: ignore[attr-defined]
        assert ok and detail.startswith("nothing left:") and word in detail
    run.stop = (1, b"still-running\n")
    ok, detail = await adapter.cleanup_orphan(crashed(plan))  # type: ignore[attr-defined]
    assert not ok and "still-running" in detail and f"pkill -KILL -f '{adapter_mod.CLI_MODULE} serve'" in detail
    run.stop = subprocess.TimeoutExpired("wsl.exe", 60)
    assert not (await adapter.cleanup_orphan(crashed(plan)))[0]  # type: ignore[attr-defined]
    await adapter.aclose()


@pytest.mark.anyio
async def test_cleanup_orphan_never_starts_wsl_and_holds_leases_when_unsure(monkeypatch):
    installed(monkeypatch)
    plan, adapter = plan_of(), adapter_mod.VllmAdapter(record(), ctx())
    for out in (b"", "docker-desktop\r\n".encode("utf-16-le")):  # -q prints nothing (exit 0) when none runs
        adapter._run = run = WslRunner((0, out))
        ok, detail = await adapter.cleanup_orphan(crashed(plan))  # type: ignore[attr-defined]
        assert ok and detail == "nothing left: WSL distro Ubuntu is not running" and len(run.calls) == 1
    for trouble in ((1, b"Wsl/Service/E_UNEXPECTED"), subprocess.TimeoutExpired("wsl.exe", 30), OSError("gone")):
        adapter._run = run = WslRunner(trouble)
        ok, detail = await adapter.cleanup_orphan(crashed(plan))  # type: ignore[attr-defined]
        assert not ok and "wsl -l --running" in detail and len(run.calls) == 1  # unknown: keep the leases
    ok, detail = await adapter.cleanup_orphan(crashed(plan.model_copy(update={"engine": "exo"})))  # type: ignore
    assert not ok and "exo deployment" in detail
    await adapter.aclose()
    assert vout.distro_running(lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, stdout="Ubuntu\r\n".encode("utf-16-le")), None, "ubuntu") is True  # wsl.exe without WSL_UTF8


@pytest.mark.anyio
async def test_cleanup_orphan_targets_the_plans_own_distro_and_root(monkeypatch):
    """A reinstall into another distro between the crash and the restart must not hide the old server."""
    installed(monkeypatch)
    plan, run = plan_of(), WslRunner((0, b"Ubuntu\nDebian\n"), stop=(0, b"graceful\n"))
    moved = record(distro="Debian", root="/home/d/.hc", venv="/home/d/.hc/venv")
    adapter = adapter_mod.VllmAdapter(moved, ctx(), runner=run)
    ok, detail = await adapter.cleanup_orphan(crashed(plan))  # type: ignore[attr-defined]
    stop = run.calls[1]
    assert ok and "graceful" in detail and stop[1:3] == ["-d", "Ubuntu"]
    assert stop[-2] == "/home/u/.hypercompute-vllm/run/dep1abcd.pid"
    assert vout.orphan_pidfile(moved, {}, "x") == ("Debian", "/home/d/.hc/run/x.pid")  # an old plan: the record's
    await adapter.aclose()
    linux = WslRunner((0, b""))
    adapter = adapter_mod.VllmAdapter(record("linux"), ctx(), runner=linux)
    lplan = plan.model_copy(update={"engine_options": {**plan.engine_options, "platform": "linux"}})
    ok, detail = await adapter.cleanup_orphan(crashed(lplan))  # type: ignore[attr-defined]
    assert ok and "native Linux" in detail and linux.calls == []  # the supervisor's restart check owns that process
    await adapter.aclose()


@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("setsid") is None,
                    reason="runs the real launch/stop scripts with bash + setsid on Linux")
def test_launch_and_stop_scripts_on_linux(tmp_path):
    pidfile = str(tmp_path / "run" / "d.pid")
    proc = subprocess.Popen(["setsid", "-w", "bash", "-c", vout.LAUNCH, pidfile, sys.executable, "-c",
                             "import time; time.sleep(60)", "vllm-probe"])
    for _ in range(50):
        if Path(pidfile).exists():
            break
        time.sleep(0.1)
    out = subprocess.run(["bash", "-c", vout.STOP, pidfile, "10"], capture_output=True, text=True, timeout=30)
    assert out.stdout.split()[-1] == "graceful"
    assert proc.wait(timeout=10) is not None and not Path(pidfile).exists()


# ----------------------------------------------------------------------------- readiness and observe


def test_log_and_stream_parsing_live_in_vllm_output():
    """adapters/vllm.py stays under the 800-line limit and runs vllm_output's parsers unchanged."""
    assert len(Path(adapter_mod.__file__).read_text(encoding="utf-8").splitlines()) <= 800
    assert adapter_mod.VllmAdapter.failure_reason is vout.exit_reason
    assert adapter_mod.StreamNormalizer is vout.StreamNormalizer and adapter_mod.parse_startup is vout.parse_startup


def test_failure_reason_for_a_server_that_died_after_ready(monkeypatch):
    installed(monkeypatch)
    adapter = adapter_mod.VllmAdapter(record(), ctx())
    explain = getattr(adapter, "failure_reason", None)  # how DeploymentManager._watch looks it up
    log = ("(EngineCore pid=9) ERROR 10-21 10:01:02 [core.py:1102] vllm.v1.engine.exceptions.EngineDeadError: "
           "EngineCore encountered an issue.\n(APIServer pid=8) INFO [shutdown] API server: exiting\n")
    assert explain(log) == ("vLLM server exited unexpectedly: vllm.v1.engine.exceptions.EngineDeadError: "
                            "EngineCore encountered an issue.")
    assert explain("") == "vLLM server exited unexpectedly"


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def handle_for(port: int = 1) -> DeploymentHandle:
    return DeploymentHandle("dep1", "deployment:dep1", f"http://127.0.0.1:{port}", Path("vllm-dep1.log"), 1,
                            ["vllm", "serve", "repo", "--revision", AWQ_SHA, "--served-model-name", "alias"],
                            api_key="KEY")


@pytest.mark.anyio
async def test_wait_ready_polls_health_then_checks_the_alias(monkeypatch):
    installed(monkeypatch)
    monkeypatch.setattr(adapter_mod, "POLL_S", 0.0)
    sup = FakeSupervisor()
    sup.procs["deployment:dep1"] = FakeProc(log=(FIXTURES / "vllm_startup_synthetic.log").read_text())
    adapter = adapter_mod.VllmAdapter(record(), ctx(sup))
    health = iter([None, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if next(health) is None:
                raise httpx.ConnectTimeout("pre-listen", request=request)
            return httpx.Response(200)
        assert request.headers["authorization"] == "Bearer KEY"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "alias", "object": "model"}]})

    await adapter.aclose()
    adapter._client = mock_client(handler)
    phases: list[str] = []
    assert installer.served_models(config.paths()) == set()
    await adapter.wait_ready(handle_for(), timeout_s=5, on_progress=phases.append)
    assert phases == ["HTTP server up"]
    assert installer.is_served(config.paths(), record(), "repo", AWQ_SHA)  # later plans skip the download room
    report = await adapter.observe(handle_for(), spec_for(plan_of()), None)
    assert report.verified and report.observed.devices[0].model_bytes == int(5.21 * GiB)
    assert "72,464 tokens" in report.detail
    await adapter.aclose()


@pytest.mark.anyio
async def test_wait_ready_fails_fast_on_a_fatal_marker_and_on_exit(monkeypatch):
    installed(monkeypatch)
    sup = FakeSupervisor()
    sup.procs["deployment:dep1"] = FakeProc(log=(FIXTURES / "vllm_startup_free_memory_error.log").read_text())
    adapter = adapter_mod.VllmAdapter(record(), ctx(sup))
    await adapter.aclose()
    adapter._client = mock_client(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)))
    with pytest.raises(EngineAdapterError) as exc:
        await adapter.wait_ready(handle_for(), timeout_s=5)
    assert exc.value.code == "engine_failed" and exc.value.fix and "llama.cpp" in exc.value.fix
    sup.procs["deployment:dep1"] = FakeProc(log="Traceback\nOSError: boom\n", alive=False)
    with pytest.raises(EngineAdapterError, match="OSError: boom") as exc:
        await adapter.wait_ready(handle_for(), timeout_s=5)
    assert exc.value.code == "engine_exited"
    await adapter.aclose()


@pytest.mark.anyio
async def test_wait_ready_rejects_a_wrong_alias(monkeypatch):
    installed(monkeypatch)
    sup = FakeSupervisor()
    sup.procs["deployment:dep1"] = FakeProc()
    adapter = adapter_mod.VllmAdapter(record(), ctx(sup))
    await adapter.aclose()
    adapter._client = mock_client(lambda r: httpx.Response(200) if r.url.path == "/health" else
                                  httpx.Response(200, json={"data": [{"id": "other"}]}))
    with pytest.raises(EngineAdapterError, match="not 'alias'"):
        await adapter.wait_ready(handle_for(), timeout_s=5)
    assert installer.served_models(config.paths()) == set()  # only a server that passed both checks counts
    await adapter.aclose()


@pytest.mark.anyio
async def test_load_budget_covers_the_first_start_download(monkeypatch):
    installed(monkeypatch)
    adapter = adapter_mod.VllmAdapter(record(), ctx())
    plan = plan_of()  # nothing cached: vLLM downloads the checkpoint inside WSL before it listens
    assert adapter.load_timeout_s(plan) == config.LOAD_TIMEOUT_S + AWQ_BYTES / MiB  # ~2 h at the ~8 Mbit/s floor
    huge = plan.model_copy(update={"engine_options": {**plan.engine_options, "download_bytes": 100 * GiB}})
    assert adapter.load_timeout_s(huge) == 4 * 3600  # capped like AirLLM and EXO
    adapter._remember_cached(awq_handle())
    cached = plan_of()
    assert adapter.load_timeout_s(plan) == adapter.load_timeout_s(cached) == config.LOAD_TIMEOUT_S
    installer.write_install(config.paths(), record())  # a reinstall forgets the cache: the stored plan downloads
    assert adapter.load_timeout_s(cached) == config.LOAD_TIMEOUT_S + AWQ_BYTES / MiB
    await adapter.aclose()


class SlowFirstStart(FakeProc):
    """vLLM's log over a first start on a 20 Mbit/s link: silent while it downloads, then its milestones."""

    RATE = 20e6 / 8
    DONE = AWQ_BYTES / RATE  # ~2228 s, past the fixed 30 min deadline wait_ready used to have
    READY = DONE + 240  # weights load, KV profiling and CUDA graph capture

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self.clock = clock

    def read_log(self, max_bytes: int = 0) -> str:
        t, out = self.clock[0], "(EngineCore pid=9) INFO [gpu_model_runner.py] Starting to load model\n"
        if t >= self.DONE:
            out += ("(EngineCore pid=9) INFO [weight_utils.py] Time spent downloading weights for "
                    f"{awq().repo}: 2228.395712 seconds\n")
        if t >= self.DONE + 30:
            out += "(EngineCore pid=9) INFO [gpu_model_runner.py] Model loading took 5.21 GiB memory and 30.0 seconds\n"
        return out


@pytest.mark.anyio
async def test_first_start_on_a_slow_link_gets_its_download_budget_and_progress(monkeypatch):
    """DeploymentManager._run calls wait_ready without timeout_s: the prepared plan's budget applies, not 30 min."""
    installed(monkeypatch)
    monkeypatch.setattr(installer, "wsl_exe", lambda: r"C:\Windows\System32\wsl.exe")
    clock, du, sup = [0.0], [], FakeSupervisor()

    def runner(argv, **kw):  # `du -sk` of the repo's folder in vLLM's HF cache, inside the distro
        du.append((clock[0], argv))
        kib = int(min(AWQ_BYTES, clock[0] * SlowFirstStart.RATE)) // 1024
        return subprocess.CompletedProcess(argv, 0, stdout=f"{kib}\t{argv[-1]}\n".encode(), stderr=b"")

    def handler(request: httpx.Request) -> httpx.Response:
        if clock[0] < SlowFirstStart.READY:
            raise httpx.ConnectTimeout("pre-listen", request=request)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    adapter = adapter_mod.VllmAdapter(record(), ctx(sup), runner=runner)
    handle = await adapter.prepare(spec_for(plan_of()))
    sup.procs[handle.process_key] = SlowFirstStart(clock)
    await adapter.aclose()
    adapter._client = mock_client(handler)
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock[0])
    real_sleep = asyncio.sleep

    async def sleep(seconds: float, *args, **kwargs) -> None:
        clock[0] += seconds
        await real_sleep(0)

    monkeypatch.setattr(adapter_mod.asyncio, "sleep", sleep)
    phases: list[str] = []
    await adapter.wait_ready(handle, on_progress=phases.append)
    assert clock[0] >= SlowFirstStart.READY > config.LOAD_TIMEOUT_S
    downloading = [p for p in phases if p.startswith(f"downloading {awq().repo} (first start): ")]
    assert phases[0] == downloading[0] and len(downloading) > 20 and downloading[-1].endswith(" of 5.2 GiB")
    assert phases[len(downloading):] == ["weights downloaded; loading them", "weights loaded; profiling memory"]
    assert {tuple(argv[:5]) for _, argv in du} == {(r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu", "--exec", "du")}
    assert du[0][1][-1] == "/home/u/.hypercompute-vllm/hf/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ"
    assert all(b - a >= vout.DOWNLOAD_POLL_S for (a, _), (b, _) in zip(du, du[1:]))  # one wsl.exe call per 15 s
    assert du[-1][0] < SlowFirstStart.DONE  # sizing stops once vLLM logs that the download finished
    assert installer.is_served(config.paths(), record(), awq().repo, AWQ_SHA)
    await adapter.aclose()


# ----------------------------------------------------------------------------- streaming


class Tick:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        self.t += 0.05
        return self.t


def test_stream_normalizer_adds_timings_to_the_usage_chunk():
    data = sse("vllm_stream_ok.sse")
    norm = vout.StreamNormalizer(vout.StreamClock(now=Tick()))
    events = [ev for i in range(0, len(data), 37) for ev in norm.feed(data[i:i + 37])] + norm.flush()
    assert events[-1].done and sum(1 for e in events if e.payload) == 5
    usage = events[-2].payload
    t = usage["timings"]
    assert usage["usage"]["completion_tokens"] == t["predicted_n"] == 3 and t["prompt_n"] == 31
    # clock ticks 50 ms per read: request at 100.05, first token 100.10, last token 100.20
    assert t["prompt_ms"] == pytest.approx(50.0) and t["predicted_ms"] == pytest.approx(100.0)
    assert t["predicted_per_second"] == pytest.approx(30.0)
    assert events[-2].raw.startswith(b"data: {") and b'"timings"' in events[-2].raw and events[-2].raw.endswith(b"\n\n")
    assert events[1].raw == data.split(b"\n\n")[1] + b"\n\n"  # other events pass through byte for byte


def test_stream_normalizer_raises_on_an_error_inside_a_200_stream():
    norm = vout.StreamNormalizer()
    with pytest.raises(EngineAdapterError, match="Internal server error") as exc:
        norm.feed(sse("vllm_stream_error.sse"))
    assert exc.value.code == "upstream_error"


@pytest.mark.anyio
async def test_stream_end_to_end_with_mock_transport(monkeypatch):
    installed(monkeypatch)
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer KEY"
        return httpx.Response(200, stream=httpx.ByteStream(sse("vllm_stream_ok.sse")),
                              headers={"content-type": "text/event-stream"})

    adapter = adapter_mod.VllmAdapter(record(), ctx())
    await adapter.aclose()
    adapter._client = mock_client(handler)
    body = {"model": "whatever", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    events = [ev async for ev in adapter.stream(handle_for(), InferenceRequest("r1", body, True))]
    assert sent["model"] == "alias" and sent["stream"] and sent["stream_options"] == {"include_usage": True}
    assert events[-1].done and events[-2].payload["timings"]["predicted_n"] == 3
    error_sse = sse("vllm_stream_error.sse")
    adapter._client = mock_client(lambda r: httpx.Response(200, stream=httpx.ByteStream(error_sse)))
    with pytest.raises(EngineAdapterError):
        _ = [ev async for ev in adapter.stream(handle_for(), InferenceRequest("r2", body, True))]
    adapter._client = mock_client(lambda r: httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(EngineAdapterError, match="HTTP 400"):
        _ = [ev async for ev in adapter.stream(handle_for(), InferenceRequest("r3", body, True))]
    await adapter.aclose()


@pytest.mark.anyio
async def test_cache_prompt_false_keeps_each_request_off_the_prefix_cache(monkeypatch):
    """Benchmarks send one prompt for warmup and every repetition with llama.cpp's cache_prompt: false."""
    installed(monkeypatch)
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if sent[-1]["stream"]:
            return httpx.Response(200, stream=httpx.ByteStream(sse("vllm_stream_ok.sse")))
        return httpx.Response(200, json={"choices": []})

    adapter = adapter_mod.VllmAdapter(record(), ctx())
    await adapter.aclose()
    adapter._client = mock_client(handler)
    bench = {"messages": [{"role": "user", "content": "long prompt"}], "max_tokens": 8, "cache_prompt": False}
    for i in range(2):
        _ = [ev async for ev in adapter.stream(handle_for(), InferenceRequest(f"s{i}", bench, True))]
        await adapter.complete(handle_for(), InferenceRequest(f"c{i}", bench, False))
    salts = [body.get("cache_salt") for body in sent]
    assert all(salts) and len(set(salts)) == 4 and not any("cache_prompt" in body for body in sent)
    assert all(len(s) <= 128 and not set(s) & set("@/\\\0") for s in salts)  # vLLM's validate_cache_salt
    assert bench["cache_prompt"] is False  # the caller's body is not modified
    for chat in ({**bench, "cache_prompt": True}, {k: v for k, v in bench.items() if k != "cache_prompt"}):
        await adapter.complete(handle_for(), InferenceRequest("chat", chat, False))
        assert "cache_salt" not in sent[-1] and "cache_prompt" not in sent[-1]  # chats keep prefix caching
    await adapter.aclose()


# ----------------------------------------------------------------------------- module contract


@pytest.mark.anyio
async def test_build_and_status_follow_the_record(monkeypatch):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    monkeypatch.setattr(adapter_mod.engine, "host_os", lambda: "windows")
    assert adapter_mod.build(ctx()) is None and adapter_mod.status(ctx()).status == "not_installed"
    installer.write_install(config.paths(), record())
    built = adapter_mod.build(ctx())
    assert built is not None and built.engine_id == "vllm" and built.idle_timeout_s == config.UPSTREAM_IDLE_TIMEOUT_S
    await built.aclose()
    snap = await built.inspect()
    assert snap.capabilities.engine_id == "vllm" and snap.info.engine == "vllm" and snap.info.release == "vllm 0.32.0"
    installer.write_install(config.paths(), replace(record(), vllm="0.31.0"))
    assert adapter_mod.build(ctx()) is None
