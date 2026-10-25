"""AirLLM sizing: install disk, the torch device the smoke test picks, the per-deployment time budget.

Split from test_airllm.py (800-line limit); reuses its helpers. Nothing here imports torch or loads a model.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import test_airllm
from hypercompute import config
from hypercompute.adapters import airllm as adapter_mod
from hypercompute.installers import airllm as inst
from hypercompute.units import GiB, MiB
from test_airllm import FIXTURES, LARGE, Usage, ctx_with, mark_installed, plan_of, ready

fake_install = test_airllm.fake_install  # the fixture: pytest finds it by name in this module


def test_install_counts_the_cuda_wheels_linux_torch_pulls_in(monkeypatch):
    ran: list[tuple] = []
    monkeypatch.setattr(inst.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(inst.engine, "nvidia_driver_major", lambda: 591)
    monkeypatch.setattr(inst.sys, "platform", "linux")
    monkeypatch.setattr(inst.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(inst, "remote_size", lambda url: 764 * MiB)  # the thin manylinux torch wheel alone
    monkeypatch.setattr(inst.shutil, "disk_usage", lambda p: Usage(0, 0, 8 * GiB))  # torch + CUDA wheels: 6.4 GiB
    monkeypatch.setattr(inst, "_run", lambda *a: ran.append(a) or 0)
    said: list[str] = []
    assert inst.install(config.paths(), yes=True, force=False, say=said.append) == 1
    assert ran == [] and any("not enough disk" in s for s in said) and any("CUDA wheels" in s for s in said)


BUILD_ARCHS = ("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")  # torch 2.11.0+cu128: no sm_89, no PTX


def run_smoke(monkeypatch, capsys, kernel_error: str | None, cap: tuple[int, int] = (8, 9),
              archs: tuple[str, ...] = BUILD_ARCHS) -> dict:
    """The installer's SMOKE source against a fake torch: by default an 8.9 card (RTX 40, L4) and the pinned build's
    arch list, which has no sm_89 although torch runs that card on its sm_86 code."""
    def ones(n, device):
        if kernel_error:
            raise RuntimeError(kernel_error)
        return NS(add_=lambda x: NS(item=lambda: 2.0))

    cuda = NS(is_available=lambda: True, get_device_capability=lambda i: cap, get_device_name=lambda i: "NVIDIA L4",
              get_arch_list=lambda: list(archs))
    torch = NS(__version__="2.11.0+cu128", version=NS(cuda="12.8"), cuda=cuda, ones=ones)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "airllm", NS(AirLLMBaseModel=object))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0")
    exec(inst.SMOKE, {})
    return inst.parse_smoke(capsys.readouterr().out)


def test_smoke_keeps_cuda_for_a_card_the_build_runs_without_its_exact_arch(monkeypatch, capsys):
    data = run_smoke(monkeypatch, capsys, None)
    assert (data["capability"], data["arch_ok"], data.get("kernel_error")) == ("8.9", True, None)
    mark_installed(config.paths(), **data)
    assert inst.torch_device(config.paths()) == "cuda:0"  # plans and the worker's --device follow this


@pytest.mark.parametrize("error", ["CUDA error: no kernel image is available for execution on the device",
                                   "CUDA error: invalid device function"])
def test_smoke_falls_back_to_the_cpu_when_a_cuda_kernel_fails(monkeypatch, capsys, error):
    data = run_smoke(monkeypatch, capsys, error)  # sm_86 is listed, but the failed kernel is the stronger evidence
    assert data["arch_ok"] is False and data.get("kernel_error") == f"RuntimeError: {error}"


TRANSIENT = "CUDA error: out of memory"  # another engine holds the VRAM while the install runs


def test_smoke_keeps_the_gpu_when_a_kernel_fails_for_another_reason(monkeypatch, capsys):
    data = run_smoke(monkeypatch, capsys, TRANSIENT)
    assert data["arch_ok"] is True and data["kernel_error"] == f"RuntimeError: {TRANSIENT}"
    mark_installed(config.paths(), **data)
    assert inst.torch_device(config.paths()) == "cuda:0"  # not pinned to the CPU until a --force reinstall


@pytest.mark.parametrize(("cap", "archs", "ok"), [
    ((8, 9), BUILD_ARCHS, True),  # sm_86 machine code runs on 8.9
    ((12, 1), ("sm_120",), True),
    ((8, 6), ("sm_75", "sm_89"), False),  # never on a lower minor
    ((6, 1), BUILD_ARCHS, False),
    ((8, 9), ("sm_75", "compute_75"), True),  # PTX is JIT-compiled for any later card
    ((9, 0), ("sm_80", "sm_90a"), True), ((10, 3), ("sm_100a",), False),  # arch-specific code: that card only
])
def test_smoke_uses_the_same_major_rule_when_the_kernel_test_is_inconclusive(monkeypatch, capsys, cap, archs, ok):
    assert run_smoke(monkeypatch, capsys, TRANSIENT, cap, archs)["arch_ok"] is ok


@pytest.mark.usefixtures("fake_install")
def test_install_warns_but_keeps_the_gpu_after_a_transient_kernel_failure(monkeypatch):
    info = {**json.loads((FIXTURES / "airllm_installed.json").read_text(encoding="utf-8")), "capability": "8.9",
            "arch_ok": True, "kernel_error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB"}
    monkeypatch.setattr(inst, "_capture", lambda cmd, cwd, timeout: subprocess.CompletedProcess(
        cmd, 0, stdout=inst.SMOKE_MARKER + json.dumps(info), stderr=""))
    said: list[str] = []
    assert inst.install(config.paths(), yes=True, force=False, say=said.append) == 0
    assert any(s.startswith("warning:") and "OutOfMemoryError" in s and "cuda:0" in s for s in said)
    assert inst.torch_device(config.paths()) == "cuda:0"


def test_status_does_not_claim_the_gpu_when_the_install_pinned_the_cpu():
    paths = config.paths()
    mark_installed(paths, arch_ok=False, kernel_error="AcceleratorError: CUDA error: no kernel image is available")
    detail = inst.status(paths).detail
    assert inst.torch_device(paths) == "cpu" and "cuda:0" not in detail and "CPU" in detail


def test_idle_timeout_for_is_the_per_deployment_budget():
    ready(LARGE)
    p, a = plan_of(LARGE), adapter_mod.AirLlmAdapter(ctx_with())  # the gateway's warmup budget, not the class 900 s
    assert a.idle_timeout_for(p) == 3 * p.engine_options["s_per_token"][1] > a.idle_timeout_s
