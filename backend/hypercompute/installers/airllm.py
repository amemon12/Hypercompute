"""AirLLM installer: a uv-created Python 3.12 venv under ~/.hypercompute/engines/airllm, never the gateway's.

Pins come from docs/research/airllm.md: AirLLM from git at bc47e0d (62 commits past the 4.0.0 release), torch
2.11.0 from the PyTorch cu128 index on NVIDIA hosts (cu128 carries sm_120 kernels; cu126 does not) and
transformers 5.16.1 (reported working with the generic streaming path on Windows + Blackwell, issue #364).
The other pins satisfy transformers 5.16.1's declared requirements. The gateway never imports torch: status()
only reads installed.json, which the smoke import inside the venv writes.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import config, engine
from ..schemas import ExtraEngineInfo, utcnow
from ..units import GiB, MiB, fmt_bytes

AIRLLM_COMMIT = "bc47e0d474abf03f491f7547085bd85a2c6f0b1b"
AIRLLM_SPEC = f"airllm @ git+https://github.com/lyogavin/airllm@{AIRLLM_COMMIT}#subdirectory=air_llm"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.11.0"
TORCH_INDEX = "https://download.pytorch.org/whl/{flavor}"
PINS = ("transformers==5.16.1", "accelerate==1.15.0", "safetensors==0.8.0", "huggingface-hub==1.33.0",
        "tokenizers==0.23.3")
MIN_CU128_DRIVER = 570  # same threshold engine.choose_variant uses for CUDA 12.8 builds
OTHER_DOWNLOAD_BYTES = 400 * MiB  # transformers, accelerate, tokenizers, airllm and their dependencies (estimate)
# Linux cu128 torch is a thin wheel: cuda-toolkit, nvidia-cudnn/cublas/nccl/cusparselt/nvshmem and triton come as
# separate wheels its HEAD does not count (torch 2.11.0: 2.9 GiB of them on x86_64 and on aarch64).
LINUX_CUDA_WHEELS_BYTES = 3 * GiB
# Disk during install is 3x every download (archive, unpacked copy, venv). Measured installs stay well below it: the
# Windows cu128 venv is 4.4 GiB for a 2.6 GiB torch wheel, the Linux x86_64 torch set 6.4 GiB for 3.7 GiB downloaded.
DISK_FACTOR = 3
SMOKE_MARKER = "HC_SMOKE "
STEP_TIMEOUT_S = 3600.0
SMOKE_TIMEOUT_S = 600.0
INSTALL_HINT = "uv run hypercompute engine install --engine airllm"

# Runs inside the AirLLM venv: imports the class the worker uses and reports what torch can see.
SMOKE = r"""
import importlib.metadata as md, json, re, sys
import torch
from airllm import AirLLMBaseModel  # noqa: F401  the generic streaming class the worker constructs
cuda = torch.cuda.is_available()
cap = tuple(torch.cuda.get_device_capability(0)) if cuda else None

def arch_listed(cap, archs):
    for arch in archs:
        m = re.fullmatch(r"(sm|compute)_(\d+)(\d)([a-z]?)", arch)
        if m is None:
            continue
        have = (int(m[2]), int(m[3]))
        if m[4]:  # sm_90a style arch-specific code runs on that exact card only
            ok = have == cap
        elif m[1] == "sm":  # CUDA's machine-code rule: same major, card minor >= the build's
            ok = have[0] == cap[0] and have[1] <= cap[1]
        else:  # PTX is JIT-compiled for any later card
            ok = have <= cap
        if ok:
            return True
    return False

kernel_error, arch_ok = None, None
if cuda:  # run one kernel: an 8.9 card runs this build's sm_86 code although get_arch_list() has no sm_89
    try:
        torch.ones(1, device="cuda:0").add_(1).item()
        arch_ok = True
    except Exception as exc:
        kernel_error = f"{type(exc).__name__}: {exc}"[:300]
        # only a missing kernel proves the build cannot run this card; out of memory or a busy GPU proves nothing,
        # so those keep the arch-list verdict instead of pinning AirLLM to the CPU until a --force reinstall
        missing = ("no kernel image", "invalid device function", "unsupported toolchain")
        arch_ok = not any(s in str(exc).lower() for s in missing) and arch_listed(cap, torch.cuda.get_arch_list())
try:
    commit = json.loads(md.distribution("airllm").read_text("direct_url.json") or "{}").get("vcs_info", {}).get(
        "commit_id")
except Exception:
    commit = None
names = ("airllm", "transformers", "accelerate", "safetensors", "huggingface-hub", "tokenizers")
info = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": cuda, "device": "cuda:0" if cuda else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
        "capability": f"{cap[0]}.{cap[1]}" if cap else None,
        "arch_ok": arch_ok, "kernel_error": kernel_error,
        "airllm_commit": commit, "packages": {n: md.version(n) for n in names}}
print("HC_SMOKE " + json.dumps(info), flush=True)
"""

Say = Callable[[str], None]


class InstallError(RuntimeError):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True)
class TorchChoice:
    flavor: str  # cu128 | cpu (both from download.pytorch.org)
    reason: str


def engine_dir(paths: config.Paths) -> Path:
    return paths.engines / "airllm"


def venv_dir(paths: config.Paths) -> Path:
    return engine_dir(paths) / "venv"


def venv_python(paths: config.Paths) -> Path:
    v = venv_dir(paths)
    return v / "Scripts" / "python.exe" if sys.platform == "win32" else v / "bin" / "python"


def installed_file(paths: config.Paths) -> Path:
    return engine_dir(paths) / "installed.json"


def read_installed(paths: config.Paths) -> dict | None:
    """installed.json, or None when it is missing or unreadable."""
    try:
        data = json.loads(installed_file(paths).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------------- status


def status(paths: config.Paths) -> ExtraEngineInfo:
    """Cheap and offline: reads installed.json and checks the venv python exists. Never imports torch."""
    if sys.platform == "darwin":
        return ExtraEngineInfo(engine="airllm", installed=False, status="unsupported",
                               detail="AirLLM's macOS path is MLX-only (Llama/Mistral, greedy); Macs serve through EXO")
    if not installed_file(paths).exists():
        return ExtraEngineInfo(engine="airllm", installed=False, status="not_installed",
                               detail=f"not installed; run: {INSTALL_HINT}")
    data = read_installed(paths)
    if data is None:
        return _failed(f"{installed_file(paths)} is unreadable")
    if not venv_python(paths).is_file():
        return _failed(f"the AirLLM venv python is missing at {venv_python(paths)}")
    if data.get("commit") != AIRLLM_COMMIT:
        return _failed(f"installed AirLLM commit {str(data.get('commit'))[:12]} is not the pinned {AIRLLM_COMMIT[:12]}")
    return ExtraEngineInfo(engine="airllm", installed=True, status="installed", version=AIRLLM_COMMIT,
                           detail=describe(data))


def _failed(problem: str) -> ExtraEngineInfo:
    return ExtraEngineInfo(engine="airllm", installed=False, status="failed",
                           detail=f"{problem}; fix: {INSTALL_HINT} --force")


def describe(data: dict) -> str:
    if (device := _device(data)) != "cpu":
        where = f"{device} ({data.get('device_name')})"
    elif data.get("cuda_available"):
        where = f"CPU only (a test CUDA kernel failed on {data.get('device_name')}): much slower"
    else:
        where = "CPU only (no CUDA in this torch build): much slower"
    return f"torch {data.get('torch')} on {where}; disk capacity mode, seconds to minutes per token"


def _device(data: dict) -> str:
    return "cuda:0" if data.get("cuda_available") and data.get("arch_ok") is not False else "cpu"


def torch_device(paths: config.Paths) -> str:
    """The torch device the installed build can use: cpu when the smoke test saw no CUDA or no kernel for the card."""
    return _device(read_installed(paths) or {})


# ----------------------------------------------------------------------------- install


def choose_torch(system: str, driver_major: int | None) -> TorchChoice:
    if system == "darwin":
        raise InstallError("AirLLM's macOS path is MLX-only (Llama/Mistral, greedy) and the generic streaming "
                           "class is not exported there", "serve models on Macs through EXO "
                           "(uv run hypercompute engine install --engine exo)")
    if driver_major is None:
        return TorchChoice("cpu", "no NVIDIA driver found: CPU torch (works, but every token is much slower)")
    if driver_major < MIN_CU128_DRIVER:
        return TorchChoice("cpu", f"NVIDIA driver {driver_major} is older than {MIN_CU128_DRIVER}, which CUDA 12.8 "
                                  "torch needs: installing CPU torch. Update the driver, then reinstall with --force")
    return TorchChoice("cu128", f"NVIDIA driver {driver_major}: CUDA 12.8 torch (has RTX 50-series sm_120 kernels)")


def torch_wheel_url(flavor: str) -> str | None:
    tag = {("win32", "AMD64"): "win_amd64", ("linux", "x86_64"): "manylinux_2_28_x86_64",
           ("linux", "aarch64"): "manylinux_2_28_aarch64"}.get((sys.platform, platform.machine()))
    if tag is None or flavor not in ("cu128", "cpu"):
        return None
    return f"{TORCH_INDEX.format(flavor=flavor)}/torch-{TORCH_VERSION}%2B{flavor}-cp312-cp312-{tag}.whl"


def cuda_wheels_bytes(flavor: str) -> int:
    """Download the torch wheel's HEAD does not see: the CUDA wheels Linux cu128 torch depends on."""
    return LINUX_CUDA_WHEELS_BYTES if flavor == "cu128" and sys.platform.startswith("linux") else 0


def remote_size(url: str | None) -> int | None:
    if url is None:
        return None
    req = urllib.request.Request(url, method="HEAD")  # noqa: S310 (fixed https URL built above)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            return int(resp.headers.get("Content-Length") or 0) or None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _run(cmd: list[str], cwd: Path, timeout: float) -> int:
    """Runs a step with the console attached, so uv's own progress and errors stay visible."""
    try:
        return subprocess.run(cmd, cwd=str(cwd), timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        raise InstallError(f"{cmd[0]} {cmd[1]} did not finish in {timeout / 60:.0f} min",
                           "check the network and re-run the install") from None
    except OSError as exc:
        raise InstallError(f"could not run {cmd[0]}: {exc}") from exc


def _capture(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd), timeout=timeout, check=False, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        raise InstallError(f"the smoke import did not finish in {timeout:.0f}s") from None
    except OSError as exc:
        raise InstallError(f"could not run the venv python: {exc}") from exc


def _confirm(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input("Download and install? [y/N] ").strip().lower() in ("y", "yes")


def install(paths: config.Paths, *, yes: bool, force: bool, say: Say) -> int:
    try:
        return _install(paths, yes=yes, force=force, say=say)
    except InstallError as exc:
        say(f"error: {exc}")
        if exc.fix:
            say(f"fix:   {exc.fix}")
        return 1


def _install(paths: config.Paths, *, yes: bool, force: bool, say: Say) -> int:
    current = status(paths)
    if current.status == "installed" and not force:
        say(f"AirLLM {AIRLLM_COMMIT[:10]} already installed: {current.detail}")
        return 0
    choice = choose_torch(sys.platform, engine.nvidia_driver_major())
    uv = shutil.which("uv")
    if uv is None:
        raise InstallError("uv was not found on PATH", "install uv (https://docs.astral.sh/uv/) and re-run")
    if shutil.which("git") is None:
        raise InstallError("git was not found on PATH (AirLLM is installed from git at a pinned commit)",
                           "install git and re-run")
    root = engine_dir(paths)
    root.mkdir(parents=True, exist_ok=True)
    wheel, cuda_wheels = remote_size(torch_wheel_url(choice.flavor)), cuda_wheels_bytes(choice.flavor)
    need = ((wheel or 3 * GiB) + cuda_wheels + OTHER_DOWNLOAD_BYTES) * DISK_FACTOR + GiB
    free = shutil.disk_usage(root).free
    say(f"AirLLM {AIRLLM_COMMIT[:10]} into {venv_dir(paths)} (Python {PYTHON_VERSION})")
    say(f"torch {TORCH_VERSION} [{choice.flavor}]: {choice.reason}")
    say(f"  torch wheel  {fmt_bytes(wheel) if wheel else 'size unknown'}  {TORCH_INDEX.format(flavor=choice.flavor)}")
    if cuda_wheels:
        say(f"  CUDA wheels torch needs on Linux (cuDNN, cuBLAS, NCCL, triton, ...): about {fmt_bytes(cuda_wheels)}")
    say(f"  {AIRLLM_SPEC}")
    say(f"  {' '.join(PINS)} and dependencies: about {fmt_bytes(OTHER_DOWNLOAD_BYTES)}")
    say(f"Needs about {fmt_bytes(need)} free disk during install; {fmt_bytes(free)} free on that drive")
    if free < need:
        raise InstallError(f"not enough disk: need about {fmt_bytes(need)}, {fmt_bytes(free)} free",
                           f"free space on the drive holding {root}")
    if not _confirm(yes):
        raise InstallError("not installed", "re-run with --yes to accept the download")
    _create_venv(uv, paths, root, say)
    _install_packages(uv, paths, root, choice, say)
    data = _smoke(paths, root, choice, say)
    _write_installed(paths, data)
    say(f"Installed: {describe(data)}")
    return 0


def _create_venv(uv: str, paths: config.Paths, root: Path, say: Say) -> None:
    installed_file(paths).unlink(missing_ok=True)  # a half-finished install must never look installed
    say("Creating the venv")
    code = _run([uv, "venv", "--no-project", "--clear", "--python", PYTHON_VERSION, str(venv_dir(paths))], root,
                STEP_TIMEOUT_S)
    if code != 0 or not venv_python(paths).is_file():
        raise InstallError(f"uv venv failed (exit {code})",
                           f"make sure uv can find or download Python {PYTHON_VERSION} (uv python install "
                           f"{PYTHON_VERSION}), then re-run")


def _install_packages(uv: str, paths: config.Paths, root: Path, choice: TorchChoice, say: Say) -> None:
    py = str(venv_python(paths))
    torch_cmd = [uv, "pip", "install", "--python", py, "--no-cache", f"torch=={TORCH_VERSION}",
                 "--default-index", TORCH_INDEX.format(flavor=choice.flavor)]
    say(f"Installing torch {TORCH_VERSION} [{choice.flavor}] first, so AirLLM's torch>=2.4 is already satisfied")
    if (code := _run(torch_cmd, root, STEP_TIMEOUT_S)) != 0:
        raise InstallError(f"installing torch failed (exit {code})", "check the network and re-run with --force")
    say("Installing AirLLM and the pinned transformers stack")
    if (code := _run([uv, "pip", "install", "--python", py, "--no-cache", AIRLLM_SPEC, *PINS], root,
                     STEP_TIMEOUT_S)) != 0:
        raise InstallError(f"installing AirLLM failed (exit {code})", "check the network and git, then re-run "
                           "with --force")


def parse_smoke(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(SMOKE_MARKER):
            try:
                data = json.loads(line[len(SMOKE_MARKER):])
            except json.JSONDecodeError:
                return None
            return data if isinstance(data, dict) else None
    return None


def _smoke(paths: config.Paths, root: Path, choice: TorchChoice, say: Say) -> dict:
    say("Smoke test: importing torch and AirLLM inside the venv (no model is loaded)")
    out = _capture([str(venv_python(paths)), "-I", "-c", SMOKE], root, SMOKE_TIMEOUT_S)
    data = parse_smoke(out.stdout)
    if out.returncode != 0 or data is None:
        tail = (out.stderr or out.stdout).strip().splitlines()[-5:]
        raise InstallError(f"smoke import failed (exit {out.returncode}): {' | '.join(tail)[:600]}",
                           "re-run the install with --force; if it persists, report the error above")
    if data.get("airllm_commit") not in (None, AIRLLM_COMMIT):
        raise InstallError(f"the venv has AirLLM {data['airllm_commit']}, not the pinned {AIRLLM_COMMIT}",
                           f"{INSTALL_HINT} --force")
    torch_version = str(data.get("torch", ""))
    if not torch_version.endswith(f"+{choice.flavor}"):
        raise InstallError(f"the venv has torch {torch_version}, not {TORCH_VERSION}+{choice.flavor} "
                           "(a dependency replaced it)", f"{INSTALL_HINT} --force")
    if choice.flavor == "cu128" and not data.get("cuda_available"):
        say("warning: torch cannot see the NVIDIA GPU; AirLLM will run on the CPU")
    if data.get("arch_ok") is False:
        say(f"warning: a test CUDA kernel failed on compute capability {data.get('capability')} "
            f"({data.get('kernel_error')}); AirLLM will run on the CPU")
    elif data.get("kernel_error"):
        say(f"warning: a test CUDA kernel failed ({data['kernel_error']}), but this torch build has kernels for "
            f"compute capability {data.get('capability')}, so AirLLM stays on cuda:0; if chats fail the same way, "
            "free GPU memory (stop other engines) and retry")
    return {**data, "engine": "airllm", "commit": AIRLLM_COMMIT, "torch_index": choice.flavor,
            "venv": str(venv_dir(paths)), "installed_at": utcnow().isoformat()}


def _write_installed(paths: config.Paths, data: dict) -> None:
    target = installed_file(paths)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(target)
