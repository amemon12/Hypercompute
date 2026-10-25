"""Pinned llama.cpp engine: variant selection, verified install, binary allowlist and smoke test.

Only binaries named in ALLOWED_BINARIES, resolved inside the installed engine directory, are ever
executed, always with argument arrays (never a shell string).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import config
from .adapters.llama_output import parse_list_devices, parse_version
from .schemas import EngineInfo, utcnow

log = logging.getLogger("hypercompute.engine")

ALLOWED_BINARIES = ("llama-server", "ggml-rpc-server")
MARKER = ".hypercompute-engine.json"
ACTIVE_FILE = "active.json"  # in the engines dir: the release+variant the last install() activated
BIN_DIR_ENV = "HYPERCOMPUTE_LLAMA_BIN_DIR"  # escape hatch for a source build (scripts/build_llama_*.sh)
EXPECTED_DEVICE_PREFIX = {"cuda": "CUDA", "metal": "MTL", "vulkan": "Vulkan", "rocm": "ROCm"}


class EngineError(RuntimeError):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Variant:
    key: str
    os: str
    arch: str
    accelerator: str
    requires: str
    assets: tuple[Asset, ...]

    @property
    def download_bytes(self) -> int:
        return sum(a.size for a in self.assets)


@dataclass(frozen=True)
class EngineLock:
    release: str
    build_number: int
    commit: str
    rpc_proto: str
    variants: dict[str, Variant]


def load_lock(path: Path | None = None) -> EngineLock:
    path = path or config.configs_dir() / "engines.lock.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["llama_cpp"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise EngineError(f"cannot read engine lock {path}: {exc}") from exc
    variants = {
        key: Variant(key=key, os=v["os"], arch=v["arch"], accelerator=v["accelerator"], requires=v["requires"],
                     assets=tuple(Asset(**a) for a in v["assets"]))
        for key, v in raw["variants"].items()
    }
    return EngineLock(release=raw["release"], build_number=int(raw["build_number"]), commit=raw["commit"],
                      rpc_proto=raw["rpc_proto"], variants=variants)


def host_os() -> str:
    s = platform.system()
    return {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(s, "other")


def host_arch() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(m, m)


def nvidia_driver_major() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    try:
        return int(first.split(".")[0])
    except ValueError:
        return None


def choose_variant(lock: EngineLock, os_name: str, arch: str, nvidia_major: int | None) -> tuple[Variant, str]:
    """Pick the variant for this host. Returns (variant, reason)."""
    key: str | None = None
    reason = ""
    if os_name == "macos" and arch == "arm64":
        key, reason = "macos-arm64", "Apple Silicon: Metal backend"
    elif os_name == "windows" and arch == "x86_64":
        if nvidia_major is not None and nvidia_major >= 580:
            key, reason = "windows-x64-cuda13.4", f"NVIDIA driver {nvidia_major} supports CUDA 13"
        else:
            key = "windows-x64-cpu"
            reason = ("no NVIDIA driver >= 580 found; using the explicitly labeled CPU fallback "
                      "(Vulkan build exists but is not validated: --variant windows-x64-vulkan)")
    elif os_name == "linux" and arch == "x86_64":
        if nvidia_major is not None and nvidia_major >= 580:
            key, reason = "linux-x64-cuda13.4", f"NVIDIA driver {nvidia_major} supports CUDA 13"
        elif nvidia_major is not None and nvidia_major >= 570:
            key, reason = "linux-x64-cuda12.8", f"NVIDIA driver {nvidia_major}: CUDA 12.8 build"
        else:
            key, reason = "linux-x64-cpu", "no usable NVIDIA driver; explicitly labeled CPU fallback"
    elif os_name == "linux" and arch == "arm64" and nvidia_major is not None:
        key, reason = "linux-arm64-cuda13.4", "Arm64 + NVIDIA (GX10 candidate, not yet validated)"
    if key is None or key not in lock.variants:
        raise EngineError(
            f"no prebuilt llama.cpp {lock.release} variant for {os_name}/{arch}",
            fix="build from source with scripts/build_llama_*.sh and set " + BIN_DIR_ENV,
        )
    return lock.variants[key], reason


def exe_name(binary: str) -> str:
    if binary not in ALLOWED_BINARIES:
        raise EngineError(f"binary {binary!r} is not allowlisted")
    return binary + (".exe" if host_os() == "windows" else "")


def install_dir(lock: EngineLock, variant: Variant) -> Path:
    return config.paths().engines / f"llama-{lock.release}-{variant.key}"


@dataclass(frozen=True)
class InstalledEngine:
    bin_dir: Path
    release: str | None
    variant: str | None
    accelerator: str
    source: str  # "pinned-release" | "env-override"

    def binary(self, name: str) -> Path:
        path = self.bin_dir / exe_name(name)
        if not path.is_file():
            raise EngineError(f"{path} is missing from the engine install",
                              fix="reinstall: uv run hypercompute engine install")
        return path


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def find_installed(lock: EngineLock | None = None) -> InstalledEngine | None:
    """The engine install() last activated; else accelerator builds before cpu, then the newest install."""
    override = os.environ.get(BIN_DIR_ENV)
    if override:
        return InstalledEngine(bin_dir=Path(override), release=None, variant="source-build",
                               accelerator="unknown", source="env-override")
    lock = lock or load_lock()
    found: dict[str, tuple[InstalledEngine, str]] = {}  # variant key -> (engine, installed_at)
    for marker in sorted(config.paths().engines.glob(f"llama-{lock.release}-*/{MARKER}")):
        meta = _read_json(marker)
        if meta.get("release") != lock.release or meta.get("variant") not in lock.variants:
            continue
        variant = lock.variants[meta["variant"]]
        if variant.os == host_os() and variant.arch == host_arch():
            eng = InstalledEngine(bin_dir=marker.parent / meta.get("bin_subdir", ""), release=lock.release,
                                  variant=variant.key, accelerator=variant.accelerator, source="pinned-release")
            found[variant.key] = (eng, str(meta.get("installed_at", "")))
    active = _read_json(config.paths().engines / ACTIVE_FILE)
    if active.get("release") == lock.release and active.get("variant") in found:
        return found[active["variant"]][0]
    if not found:
        return None
    return max(found.values(), key=lambda e: (e[0].accelerator != "cpu", e[1]))[0]


# ----------------------------------------------------------------------------- install


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(url: str, dest: Path, sha256: str, size: int,
                      progress: Callable[[int, int], None] | None = None,
                      client: httpx.Client | None = None) -> Path:
    """Resumable download to dest with SHA256 verification. Reuses a verified cached file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == size and _sha256(dest) == sha256:
        return dest
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have and have == size:  # complete but never finalized: a Range request would only get HTTP 416
        if _sha256(part) == sha256:
            part.replace(dest)
            return dest
        part.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    http = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0))
    try:
        with http, http.stream("GET", url, headers=headers) as r:
            if r.status_code == 200 and have:
                have = 0  # server ignored the range; start over
            elif r.status_code not in (200, 206):
                raise EngineError(f"download failed: HTTP {r.status_code} for {url}")
            response_end = size
            if r.status_code == 206:
                content_range = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                                             r.headers.get("Content-Range", ""))
                if not content_range:
                    raise EngineError(f"invalid Content-Range for {dest.name}; partial download preserved")
                start, end, total = map(int, content_range.groups())
                if start != have or total != size or not start <= end < size:
                    raise EngineError(f"unexpected Content-Range for {dest.name}; partial download preserved")
                response_end = end + 1
            with part.open("ab" if have else "wb") as fh:
                done = have
                for chunk in r.iter_bytes(1024 * 1024):
                    if done + len(chunk) > response_end:
                        raise EngineError(f"download exceeds expected size or range for {dest.name}; "
                                          "partial download preserved")
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, size)
    except httpx.HTTPError as exc:
        detail = str(exc).strip()
        raise EngineError(f"download interrupted for {dest.name} ({type(exc).__name__})"
                          + (f": {detail}" if detail else "")
                          + "; retry to resume the partial download") from exc
    if done != size:
        raise EngineError(f"incomplete download for {dest.name}: received {done} of {size} bytes; "
                          "retry to resume the partial download")
    digest = _sha256(part)
    if digest != sha256:
        part.unlink(missing_ok=True)
        raise EngineError(f"SHA256 mismatch for {dest.name}: expected {sha256}, got {digest}")
    part.replace(dest)
    return dest


def _safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise EngineError(f"unsafe path in archive: {member.filename}")
            zf.extractall(dest)  # noqa: S202 - every member path validated above
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    else:
        raise EngineError(f"unsupported archive type: {archive.name}")


def _locate_bin_dir(root: Path) -> Path:
    server = exe_name("llama-server")
    for path in sorted(root.rglob(server), key=lambda p: len(p.parts)):
        return path.parent
    raise EngineError(f"{server} not found after extracting into {root}")


def install(variant: Variant, lock: EngineLock, progress: Callable[[str, int, int], None] | None = None,
            ) -> InstalledEngine:
    target = install_dir(lock, variant)
    downloads = config.paths().ensure().downloads
    free = shutil.disk_usage(downloads).free
    need = variant.download_bytes * 3  # archive + extracted + slack
    if free < need:
        raise EngineError(f"not enough disk: need ~{need // 2**20} MiB free, have {free // 2**20} MiB")
    archives = [
        download_verified(a.url, downloads / a.name, a.sha256, a.size,
                          (lambda d, t, n=a.name: progress(n, d, t)) if progress else None)
        for a in variant.assets
    ]
    staging = target.with_name(target.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    for archive in archives:
        _safe_extract(archive, staging)
    bin_dir = _locate_bin_dir(staging)
    if host_os() != "windows":
        for name in ALLOWED_BINARIES:
            p = bin_dir / name
            if p.exists():
                p.chmod(p.stat().st_mode | 0o111)
    # cudart archives extract next to the binaries on Windows; on Linux they may land elsewhere.
    meta = {"release": lock.release, "commit": lock.commit, "variant": variant.key,
            "bin_subdir": str(bin_dir.relative_to(staging)), "installed_at": utcnow().isoformat(),
            "assets": [{"name": a.name, "sha256": a.sha256} for a in variant.assets]}
    (staging / MARKER).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(target, ignore_errors=True)
    staging.replace(target)
    (target.parent / ACTIVE_FILE).write_text(json.dumps({"release": lock.release, "variant": variant.key}),
                                             encoding="utf-8")
    return InstalledEngine(bin_dir=target / meta["bin_subdir"], release=lock.release, variant=variant.key,
                           accelerator=variant.accelerator, source="pinned-release")


# ----------------------------------------------------------------------------- inspect / smoke


def run_engine(engine: InstalledEngine, binary: str, args: list[str], timeout: float,
               ) -> subprocess.CompletedProcess[str]:
    exe = engine.binary(binary)
    return subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=str(engine.bin_dir), check=False, errors="replace")


def inspect(engine: InstalledEngine | None, lock: EngineLock | None = None) -> EngineInfo:
    """Version + device smoke test. A failed accelerator check is reported, never hidden."""
    lock = lock or load_lock()
    if engine is None:
        return EngineInfo(installed=False, smoke="not_run",
                          smoke_detail="llama.cpp engine not installed (uv run hypercompute engine install)")
    info: dict = {"installed": True, "variant": engine.variant, "bin_dir": str(engine.bin_dir),
                  "accelerator": engine.accelerator}
    try:
        ver = run_engine(engine, "llama-server", ["--version"], timeout=60)
        dev = run_engine(engine, "llama-server", ["--list-devices"], timeout=120)
    except (OSError, subprocess.TimeoutExpired, EngineError) as exc:
        return EngineInfo(**info, smoke="failed", smoke_detail=f"engine did not run: {exc}")
    v = parse_version(ver.stdout + ver.stderr)
    devices = parse_list_devices(dev.stdout + dev.stderr)
    info.update(devices=devices)
    if v:
        info.update(build_number=v.build_number, commit=v.commit_short,
                    release=f"b{v.build_number}")
    problems: list[str] = []
    if v is None:
        problems.append("could not parse llama-server --version")
    elif engine.source == "pinned-release" and v.build_number != lock.build_number:
        problems.append(f"build {v.build_number} != pinned {lock.build_number}")
    if v and v.build_number == lock.build_number:
        info["rpc_proto"] = lock.rpc_proto
    prefix = EXPECTED_DEVICE_PREFIX.get(engine.accelerator)
    if prefix and not any(d.name.startswith(prefix) for d in devices):
        problems.append(f"expected a {prefix} device from the {engine.accelerator} build, found "
                        f"{[d.name for d in devices] or 'none'}")
    if problems:
        return EngineInfo(**info, smoke="failed", smoke_detail="; ".join(problems))
    names = ", ".join(d.name for d in devices) or "CPU only"
    return EngineInfo(**info, smoke="passed", smoke_detail=f"{v.line if v else ''}; devices: {names}")
