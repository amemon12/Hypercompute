"""vLLM installer: the pinned vLLM in its own Linux venv, inside a WSL2 distro on Windows or natively on Linux.

The gateway never imports vLLM. scripts/install_vllm_wsl.sh does the work on the Linux file system
(~/.hypercompute-vllm/venv) and prints one HC_VLLM_RESULT line, recorded here in engines/vllm/installed.json.
status() only reads that file, so it never starts the WSL VM. Facts: docs/research/vllm.md (v0.32.0 = 5df63a7).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .. import config, engine
from ..schemas import ExtraEngineInfo, utcnow
from ..units import GiB, fmt_bytes

VLLM_VERSION = "0.32.0"  # must match VLLM_VERSION in scripts/install_vllm_wsl.sh (tests check)
TORCH_BACKEND = "cu130"
INSTALL_HINT = "uv run hypercompute engine install --engine vllm"
SCRIPT_NAME = "install_vllm_wsl.sh"
RESULT_MARKER = "HC_VLLM_RESULT "
DISTRO_ENV = "HYPERCOMPUTE_WSL_DISTRO"  # pick a WSL distro by name instead of the default choice
ROOT_ENV = "HC_VLLM_ROOT"  # the script's install root override (native Linux; WSL does not pass it through)
# Measured on the G14: ~8 GB of packages from uv plus ~9.3 GB of FlashInfer kernels (download-kernels), and room
# for the distro itself, which breaks when its disk fills.
INSTALL_NEED = 20 * GiB
NOT_DISTROS = ("docker-desktop", "docker-desktop-data")  # Docker's utility VMs, never a place to install
# Runs inside the distro (or on native Linux): store the script streamed on stdin in the Linux home, then run it from
# there (a script read from stdin would have its own commands eat the rest of it).
WSL_COPY_AND_RUN = ('mkdir -p "$HOME/.hypercompute-vllm" && cat > "$HOME/.hypercompute-vllm/install_vllm_wsl.sh" '
                    '&& exec bash "$HOME/.hypercompute-vllm/install_vllm_wsl.sh" "$@"')
LEAKY_ENV = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "PYTHONHOME")  # the gateway venv's, not vLLM's
LXSS_KEY = r"Software\Microsoft\Windows\CurrentVersion\Lxss"  # HKCU: one subkey per WSL distro (DistributionName)
# `bash -c RECLAIM <venv>`: bytes deleting the venv frees, when it is the one the script's --force deletes. Only files
# with no other link count: an install from before $ROOT/uv-cache shares ~8 GB of packages with ~/.cache/uv hardlinks.
RECLAIM = r"""[ "$0" = "${HC_VLLM_ROOT:-$HOME/.hypercompute-vllm}/venv" ] && [ -d "$0" ] || { echo none; exit 0; }
find "$0" -xdev -type f -links 1 -printf '%b\n' | awk '{ s += $1 } END { printf "%.0f\n", s * 512 }'"""

log = logging.getLogger("hypercompute.installers.vllm")

Streamer = Callable[[Sequence[str], bytes | None, Callable[[str], None]], int]  # argv, stdin, on_line -> exit code
Runner = Callable[[Sequence[str]], tuple[int, bytes]]  # argv -> (exit code, raw stdout)


class InstallError(Exception):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True)
class WslDistro:
    name: str
    state: str
    version: int
    default: bool


@dataclass(frozen=True)
class VllmInstall:
    """What installed.json records. Paths are Linux paths (inside the distro on Windows)."""

    platform: Literal["wsl2", "linux"]
    distro: str | None
    root: str
    venv: str
    vllm: str
    torch: str
    torch_cuda: str | None
    capability: str | None
    device: str | None
    python: str | None = None
    uv: str | None = None
    kernels: str | None = None
    installed_at: str | None = None

    @property
    def vllm_bin(self) -> str:
        return f"{self.venv}/bin/vllm"

    @property
    def hf_home(self) -> str:
        return f"{self.root}/hf"  # vLLM downloads engine-managed checkpoints here (Linux file system)

    @property
    def run_dir(self) -> str:
        return f"{self.root}/run"

    def describe(self) -> str:
        where = f"WSL2 distro {self.distro}" if self.platform == "wsl2" else "native Linux"
        cap = f"sm_{self.capability.replace('.', '')}" if self.capability else "GPU not seen"
        gpu = self.device or "unknown GPU"
        return f"vLLM {self.vllm} (torch {self.torch}) in {where}, {gpu} {cap}, venv {self.venv}"


# ----------------------------------------------------------------------------- record


def engine_dir(paths: config.Paths) -> Path:
    return paths.engines / "vllm"


def marker_path(paths: config.Paths) -> Path:
    return engine_dir(paths) / "installed.json"


def read_install(paths: config.Paths) -> VllmInstall | None:
    """The recorded install, None when absent. A corrupt record raises InstallError."""
    try:
        raw = marker_path(paths).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise InstallError(f"cannot read {marker_path(paths)}: {exc}", f"{INSTALL_HINT} --force") from exc
    try:
        data = json.loads(raw)
        fields = set(VllmInstall.__dataclass_fields__)
        rec = VllmInstall(**{k: v for k, v in data.items() if k in fields})
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise InstallError(f"{marker_path(paths)} is not a valid vLLM install record ({exc})",
                           f"{INSTALL_HINT} --force") from exc
    if rec.platform not in ("wsl2", "linux") or not rec.venv.startswith("/") or not rec.root.startswith("/"):
        raise InstallError(f"{marker_path(paths)} has unexpected values", f"{INSTALL_HINT} --force")
    return rec


def write_install(paths: config.Paths, rec: VllmInstall) -> Path:
    """Record a fresh install, forgetting first which checkpoints the previous one had cached (see served_key)."""
    forget_served(paths)
    path = marker_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(rec), indent=2), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


def record_from_result(result: dict, distro: str | None) -> VllmInstall:
    """Validate the script's HC_VLLM_RESULT payload (external data) and build the record."""
    def text(key: str, required: bool = True) -> str | None:
        value = result.get(key)
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value or len(value) > 512:
            raise InstallError(f"install script reported no valid {key!r}", "re-run the install and read its output")
        return value

    venv, root = text("venv"), text("root")
    if not (venv or "").startswith("/") or not (root or "").startswith("/") or (root or "").startswith("/mnt/"):
        raise InstallError(f"install script reported a bad venv/root ({venv}, {root})", "re-run the install")
    if result.get("cuda_available") is not True:
        raise InstallError("PyTorch in the vLLM venv cannot see the GPU",
                           "WSL2: update the Windows NVIDIA driver and run `wsl --update`")
    return VllmInstall(platform="wsl2" if distro else "linux", distro=distro, root=root or "", venv=venv or "",
                       vllm=text("vllm") or "", torch=text("torch") or "", torch_cuda=text("torch_cuda", False),
                       capability=text("capability", False), device=text("device", False),
                       python=text("python", False), uv=text("uv", False), kernels=text("kernels", False),
                       installed_at=utcnow().isoformat())


def parse_result(lines: Sequence[str]) -> dict | None:
    for line in reversed(lines):
        if line.startswith(RESULT_MARKER):
            try:
                data = json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                return None
            return data if isinstance(data, dict) else None
    return None


# ----------------------------------------------------------------------------- platform / WSL


def unsupported_reason(system: str | None = None) -> str | None:
    system = system or engine.host_os()
    if system in ("windows", "linux"):
        return None
    if system == "macos":
        return ("vLLM's CUDA engine does not run on macOS; Apple Silicon needs the separate vllm-metal plugin "
                "(future work). This Mac serves through llama.cpp Metal or EXO.")
    return f"vLLM needs Linux or Windows with WSL2 and an NVIDIA GPU, not {system}"


def wsl_exe() -> str | None:
    found = shutil.which("wsl.exe") or shutil.which("wsl")
    if found:
        return found
    fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "wsl.exe"
    return str(fallback) if fallback.is_file() else None


def decode_wsl(raw: bytes) -> str:
    """wsl.exe prints UTF-16LE unless WSL_UTF8=1 took effect; accept both."""
    if b"\x00" in raw:
        return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return raw.decode("utf-8", errors="replace")


def parse_wsl_list(text: str) -> list[WslDistro]:
    """Rows of `wsl.exe -l -v`: an optional '*' (default), NAME, STATE, VERSION."""
    out = []
    for line in text.replace("\x00", "").splitlines():
        default = line.lstrip().startswith("*")
        parts = line.replace("*", " ", 1).split() if default else line.split()
        if len(parts) != 3 or parts[0] == "NAME" or not parts[2].isdigit():
            continue
        out.append(WslDistro(name=parts[0], state=parts[1], version=int(parts[2]), default=default))
    return out


def wsl_env() -> dict[str, str]:
    return {**os.environ, "WSL_UTF8": "1"}


def run_capture(argv: Sequence[str], timeout: float = 30.0) -> tuple[int, bytes]:
    try:
        out = subprocess.run(list(argv), capture_output=True, timeout=timeout, env=wsl_env(), stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found".encode()
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]}: timed out after {timeout:.0f}s".encode()
    except OSError as exc:
        return 126, f"{argv[0]}: {exc}".encode()
    return out.returncode, out.stdout + out.stderr


def list_distros(runner: Runner = run_capture) -> list[WslDistro]:
    exe = wsl_exe()
    if exe is None:
        raise InstallError("WSL is not installed (wsl.exe not found)",
                           "run `wsl --install -d Ubuntu` in an elevated terminal, reboot, then retry")
    code, raw = runner([exe, "-l", "-v"])
    if code != 0:
        raise InstallError(f"`wsl -l -v` failed: {decode_wsl(raw).strip()[:300]}",
                           "run `wsl --install -d Ubuntu`, or `wsl --update` if WSL is installed")
    return parse_wsl_list(decode_wsl(raw))


def choose_distro(distros: Sequence[WslDistro], wanted: str | None = None) -> WslDistro:
    """The named distro, else the default Ubuntu, else any Ubuntu, else the default; never Docker's VMs."""
    if wanted:
        hit = next((d for d in distros if d.name.lower() == wanted.lower()), None)
        if hit is None:
            have = ", ".join(d.name for d in distros) or "none"
            raise InstallError(f"WSL distro {wanted!r} not found (have: {have})",
                               f"unset {DISTRO_ENV} or set it to one of the names in `wsl -l -v`")
        usable = [hit]
    else:
        usable = [d for d in distros if d.name.lower() not in NOT_DISTROS]
        usable.sort(key=lambda d: (not d.name.lower().startswith("ubuntu"), not d.default))
    if not usable:
        raise InstallError("no Linux distro is installed in WSL (Docker's own VMs do not count)",
                           "run `wsl --install -d Ubuntu-24.04`, open it once to create your user, then retry")
    pick = usable[0]
    if pick.version != 2:
        raise InstallError(f"WSL distro {pick.name} runs under WSL {pick.version}; CUDA needs WSL 2",
                           f"run `wsl --set-version {pick.name} 2`")
    return pick


# ----------------------------------------------------------------------------- model cache (disk)


def wsl_base_path(distro: str) -> Path | None:
    """Windows folder holding the distro's virtual disk (ext4.vhdx), read from the registry; never starts WSL."""
    try:
        import winreg
    except ImportError:  # not Windows
        return None
    try:
        root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, LXSS_KEY)
    except OSError:
        return None
    with root:
        for i in range(winreg.QueryInfoKey(root)[0]):
            try:
                with winreg.OpenKey(root, winreg.EnumKey(root, i)) as key:
                    if str(winreg.QueryValueEx(key, "DistributionName")[0]).lower() == distro.lower():
                        return Path(str(winreg.QueryValueEx(key, "BasePath")[0]).removeprefix("\\\\?\\"))
            except OSError:
                continue  # a half-registered distro; keep looking
    return None


def disk_free(wsl_distro: str | None, linux_path: str) -> tuple[int | None, str]:
    """(free bytes, where) of the disk that receives writes to linux_path; None if unknown.

    WSL2 (wsl_distro set) keeps the Linux file system in a virtual disk that grows on a Windows drive, so that drive's
    free space bounds the writes (the disk's own size cap, 1 TB by default, is far above it, and `df` inside WSL
    reports that cap). Native Linux: linux_path's own file system."""
    if wsl_distro is not None:
        base = wsl_base_path(wsl_distro)
        if base is None:
            return None, f"WSL2 distro {wsl_distro}'s virtual disk"
        probe, where = base, f"{base.drive or base} (holds WSL2 distro {wsl_distro}'s virtual disk)"
    else:
        probe, where = Path(linux_path), linux_path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free, where
    except OSError as exc:
        log.warning("free disk space of %s unknown: %s", probe, exc)
        return None, where


def cache_disk(rec: VllmInstall) -> tuple[int | None, str]:
    """(free bytes, where) of the file system that receives vLLM's model downloads (rec.hf_home); None if unknown."""
    return disk_free((rec.distro or "") if rec.platform == "wsl2" else None, rec.hf_home)


def install_disk(distro: str | None) -> tuple[int | None, str]:
    """(free bytes, where) for the install itself: the drive holding the distro's virtual disk, else the install root
    on this Linux host (the script's default, $HOME/.hypercompute-vllm, unless HC_VLLM_ROOT moves it)."""
    return disk_free(distro, os.environ.get(ROOT_ENV) or str(Path.home() / ".hypercompute-vllm"))


def served_path(paths: config.Paths) -> Path:
    return engine_dir(paths) / "served.json"


def served_key(rec: VllmInstall, repo: str, revision: str) -> str:
    """Where a checkpoint sits in vLLM's cache: the same repo@revision in another distro or HF_HOME is not there."""
    return f"{rec.distro or rec.platform}|{rec.hf_home}|{repo}@{revision}"


def served_models(paths: config.Paths) -> set[str]:
    """served_key of every checkpoint a vLLM deployment served here, so it is already in vLLM's model cache."""
    try:
        data = json.loads(served_path(paths).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable %s: %s", served_path(paths), exc)
        return set()
    return {s for s in data if isinstance(s, str)} if isinstance(data, list) else set()


def is_served(paths: config.Paths, rec: VllmInstall, repo: str, revision: str) -> bool:
    return served_key(rec, repo, revision) in served_models(paths)


def remember_served(paths: config.Paths, rec: VllmInstall, repo: str, revision: str) -> None:
    """Record that vLLM downloaded repo@revision for this install (raises OSError when it cannot be written)."""
    seen = served_models(paths)
    key = served_key(rec, repo, revision)
    if key in seen:
        return
    path = served_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(seen | {key}), indent=2), encoding="utf-8", newline="\n")
    tmp.replace(path)


def forget_served(paths: config.Paths) -> None:
    """A (re)install may sit on a new distro or a wiped cache: plans check disk room again until a server proves it.
    Raises OSError when the record exists but cannot be removed."""
    served_path(paths).unlink(missing_ok=True)


def script_path() -> Path:
    """The checkout's scripts/ copy, else one bundled into the wheel under hypercompute/_scripts (if packaged)."""
    checkout = config.repo_root() / "scripts" / SCRIPT_NAME
    return checkout if checkout.is_file() else Path(__file__).resolve().parents[1] / "_scripts" / SCRIPT_NAME


def script_bytes() -> bytes:
    path = script_path()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InstallError(f"installer script missing: {path} ({exc})",
                           "run the install from a Hypercompute source checkout") from exc
    return data.replace(b"\r", b"")  # a CRLF checkout breaks bash ($'\r': command not found); the script has no CR


# ----------------------------------------------------------------------------- running


def _feed(stream: object, data: bytes) -> None:
    try:
        stream.write(data)  # type: ignore[attr-defined]
    except OSError:
        pass  # the child exited early; its exit code and output explain why
    finally:
        try:
            stream.close()  # type: ignore[attr-defined]
        except OSError:
            pass


def stream_command(argv: Sequence[str], stdin: bytes | None, on_line: Callable[[str], None]) -> int:
    env = {k: v for k, v in wsl_env().items() if k not in LEAKY_ENV}
    try:
        proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except OSError as exc:
        on_line(f"could not start {argv[0]}: {exc}")
        return 127
    feeder = None
    if stdin is not None:
        feeder = threading.Thread(target=_feed, args=(proc.stdin, stdin), daemon=True)
        feeder.start()
    assert proc.stdout is not None
    # Universal newlines: a progress redraw ends in a lone CR, and the result marker after it starts a new line.
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
        on_line(line.rstrip("\n"))
    if feeder:
        feeder.join(timeout=5)
    return proc.wait()


def install_command(system: str, force: bool,
                    runner: Runner = run_capture) -> tuple[list[str], bytes | None, str | None]:
    """argv, stdin and WSL distro (None on native Linux) that run the install script.

    Both paths stream the CR-stripped script on stdin, so a checkout with CRLF endings still installs."""
    extra = ["--force"] if force else []
    script = script_bytes()
    if system == "linux":
        return ["bash", "-c", WSL_COPY_AND_RUN, "hc-vllm-install", *extra], script, None
    distro = choose_distro(list_distros(runner), os.environ.get(DISTRO_ENV))
    exe = wsl_exe() or "wsl.exe"
    return [exe, "-d", distro.name, "--cd", "~", "--exec", "bash", "-c", WSL_COPY_AND_RUN, "hc-vllm-install",
            *extra], script, distro.name


def _confirm(prompt: str) -> bool:
    return sys.stdin.isatty() and input(prompt).strip().lower() in ("y", "yes")


def install(paths: config.Paths, *, yes: bool, force: bool, say: Callable[[str], None],
            stream: Streamer = stream_command, runner: Runner = run_capture) -> int:
    system = engine.host_os()
    reason = unsupported_reason(system)
    if reason:
        say(f"error: {reason}")
        return 2
    if not force:
        info = status(paths)
        if info.installed:
            say(f"{info.detail}; already installed (use --force to reinstall)")
            return 0
    try:
        argv, stdin, distro = install_command(system, force, runner)
    except InstallError as exc:
        return _fail(say, str(exc), exc.fix)
    target = f"WSL2 distro {distro}" if distro else "this Linux host"
    say(f"vLLM {VLLM_VERSION} with PyTorch 2.13.0 ({TORCH_BACKEND}) into {target} at ~/.hypercompute-vllm/venv "
        "(~18 GB of packages and kernels; the model is downloaded later, on the first deployment)")
    if not _room_to_install(paths, distro, force, say, runner):
        return 1
    if not yes and not _confirm("Install? [y/N] "):
        return _fail(say, "not installed", "re-run with --yes to accept the download")
    if force and not _forget_install(paths, say):
        return 1
    lines: list[str] = []

    def on_line(line: str) -> None:
        lines.append(line)
        say(f"  {line}")

    code = stream(argv, stdin, on_line)
    result = parse_result(lines)
    if code != 0 or result is None:
        return _fail(say, f"vLLM install script failed (exit code {code})",
                     "read the [hypercompute-vllm] ERROR/fix lines above, fix them, then re-run the install")
    try:
        rec = record_from_result(result, distro)
    except InstallError as exc:
        return _fail(say, str(exc), exc.fix)
    path = write_install(paths, rec)
    say(f"Installed {rec.describe()}; recorded in {path}")
    if rec.vllm != VLLM_VERSION:
        say(f"warning: the venv reports vLLM {rec.vllm}, this build pins {VLLM_VERSION}")
    if rec.kernels == "failed":
        say("warning: FlashInfer kernels were not downloaded; the first start compiles them (several minutes)")
    return 0


def _room_to_install(paths: config.Paths, distro: str | None, force: bool, say: Callable[[str], None],
                     runner: Runner = run_capture) -> bool:
    """Free space where the install lands. A --force reinstall over this target's own recorded venv also counts what
    deleting that venv frees (the script removes it first; a WSL2 virtual disk reuses those blocks), measured."""
    free, where = install_disk(distro)
    if free is None:
        say(f"warning: free space of {where} could not be read; make sure {fmt_bytes(INSTALL_NEED)} is free there")
        return True
    if free >= INSTALL_NEED:
        return True
    short = f"the install needs ~{fmt_bytes(INSTALL_NEED)} free and {where} has {fmt_bytes(free)}"
    freed = _old_venv_frees(paths, distro, runner) if force else None
    if freed is not None and free + freed >= INSTALL_NEED:
        say(f"warning: {short}; continuing because --force first deletes the old venv, which frees "
            f"~{fmt_bytes(freed)} (not counting files uv's cache still links to)")
        return True
    _fail(say, short + (f" plus ~{fmt_bytes(freed)} from deleting the old venv" if freed else ""),
          f"free at least {fmt_bytes(INSTALL_NEED - free - (freed or 0))} on {where}, then re-run the install")
    return False


def _old_venv_frees(paths: config.Paths, distro: str | None, runner: Runner) -> int | None:
    """Bytes the script's --force frees by deleting the recorded venv, measured in the target; None unless the record
    is this target's (same WSL2 distro, or native Linux) and its venv is there."""
    try:
        rec = read_install(paths)
    except InstallError:
        return None
    if rec is None or rec.platform != ("wsl2" if distro else "linux") or \
            (rec.distro or "").lower() != (distro or "").lower():
        return None
    argv = ["bash", "-c", RECLAIM, rec.venv]
    if distro:
        argv = [wsl_exe() or "wsl.exe", "-d", distro, "--cd", "~", "--exec", *argv]
    code, raw = runner(argv)
    words = decode_wsl(raw).split()
    if code != 0 or not words or not words[-1].isdigit():
        if words[-1:] != ["none"]:
            log.warning("could not measure %s (exit %s): %s", rec.venv, code, decode_wsl(raw).strip()[:200])
        return None
    return int(words[-1])


def _forget_install(paths: config.Paths, say: Callable[[str], None]) -> bool:
    """--force deletes the venv before rebuilding it: drop the old record first, so a reinstall that fails reads as
    not installed instead of as the venv that is gone (AirLLM does the same)."""
    try:
        marker_path(paths).unlink(missing_ok=True)
        forget_served(paths)
    except OSError as exc:
        _fail(say, f"cannot remove the old install record: {exc}", f"delete {marker_path(paths)}, then re-run")
        return False
    return True


def _fail(say: Callable[[str], None], message: str, fix: str | None) -> int:
    say(f"error: {message}")
    if fix:
        say(f"fix:   {fix}")
    return 1


# ----------------------------------------------------------------------------- status


def install_problem(rec: VllmInstall, system: str) -> str | None:
    if rec.vllm != VLLM_VERSION:
        return f"installed vLLM {rec.vllm} differs from the pinned {VLLM_VERSION}"
    if (rec.platform == "wsl2") != (system == "windows"):
        return f"the record is for {rec.platform} but this host runs {system}"
    if not rec.capability:
        return "PyTorch could not see the GPU when vLLM was installed"
    return None


def status(paths: config.Paths) -> ExtraEngineInfo:
    """Cheap and offline: reads installed.json only (never starts the WSL VM)."""
    system = engine.host_os()
    reason = unsupported_reason(system)
    if reason:
        return ExtraEngineInfo(engine="vllm", installed=False, status="unsupported", detail=reason)
    try:
        rec = read_install(paths)
    except InstallError as exc:
        return ExtraEngineInfo(engine="vllm", installed=False, status="failed", detail=f"{exc}; fix: {exc.fix}")
    if rec is None:
        where = " (installs into WSL2)" if system == "windows" else ""
        return ExtraEngineInfo(engine="vllm", installed=False, status="not_installed",
                               detail=f"vLLM is not installed; run: {INSTALL_HINT}{where}")
    problem = install_problem(rec, system)
    if problem:
        return ExtraEngineInfo(engine="vllm", installed=False, status="failed", version=rec.vllm,
                               detail=f"{problem}; run: {INSTALL_HINT} --force")
    return ExtraEngineInfo(engine="vllm", installed=True, status="installed", version=rec.vllm, detail=rec.describe())
