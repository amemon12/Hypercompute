"""EXO installer: the pinned source build in its own uv environment under ~/.hypercompute/engines/exo/src.

Apple Silicon Macs only. The packaged EXO app (v1.0.71) predates the zenoh rewrite and cannot join a source node
at the pin, so friends run this source build under the Hypercompute agent. Facts: docs/research/exo.md (bc412e2).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..schemas import ExtraEngineInfo, utcnow

EXO_REPO = "https://github.com/exo-explore/exo"
EXO_PIN = "bc412e2230f395e652c23426a5eb88dfa515769b"
EXO_VERSION = "0.3.70"  # pyproject version at the pin
API_PORT = 52415  # every node must use the same API port: peers probe it to build topology edges
PYTHON = "3.13"  # requires-python ==3.13.*
MIN_NODE = (18, 0, 0)
MIN_UV = (0, 8, 6)  # [tool.uv] required-version
MIN_RUST = (1, 85, 0)  # exo-rs uses Cargo edition 2024
INSTALL_HINT = "uv run hypercompute engine install --engine exo"

Runner = Callable[[Sequence[str]], tuple[int, str]]  # argv -> (exit code, combined output)
Streamer = Callable[[Sequence[str], Path], int]  # argv, cwd -> exit code; output goes to the terminal


def engine_dir(paths: config.Paths) -> Path:
    return paths.engines / "exo"


def src_dir(paths: config.Paths) -> Path:
    return engine_dir(paths) / "src"


def exo_binary(paths: config.Paths) -> Path:
    return src_dir(paths) / ".venv" / "bin" / "exo"


def venv_python(paths: config.Paths) -> Path:
    """The agent runs `python -m exo`: a supervised process whose exe is the interpreter, so orphan
    reconciliation (pid + create time + exe) recognises it; the `exo` entry-point script would not match."""
    return src_dir(paths) / ".venv" / "bin" / "python"


def marker_path(paths: config.Paths) -> Path:
    return engine_dir(paths) / "installed.json"


def unsupported_reason(system: str | None = None, machine: str | None = None) -> str | None:
    """Why this machine cannot run EXO at the pin, or None on an Apple Silicon Mac."""
    system = system or sys.platform
    machine = machine or platform.machine()
    if system == "darwin":
        return None if machine == "arm64" else f"EXO needs an Apple Silicon Mac (MLX on Metal); this Mac is {machine}"
    if system.startswith("linux"):
        return ("EXO's Linux CPU build is broken at the pinned commit (mlx 0.32.0 fork wheel vs mlx-cpu 0.31.2, "
                "upstream issue #2359) and Linux GPUs are unsupported; this machine serves through llama.cpp")
    if system == "win32":
        return ("EXO cannot run on Windows (it imports the Unix-only `resource` module and supports only "
                "darwin/linux environments); this machine serves through llama.cpp")
    return f"EXO does not support {system}"


def read_marker(paths: config.Paths) -> dict | None:
    try:
        data = json.loads(marker_path(paths).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def status(paths: config.Paths) -> ExtraEngineInfo:
    """Cheap and offline: installed.json at the pinned commit plus the environment's `exo` entry point."""
    reason = unsupported_reason()
    if reason:
        return ExtraEngineInfo(engine="exo", installed=False, status="unsupported", detail=reason)
    rec = read_marker(paths)
    if rec is None:
        return ExtraEngineInfo(engine="exo", installed=False, status="not_installed",
                               detail=f"EXO is not installed; run: {INSTALL_HINT}")
    if rec.get("commit") != EXO_PIN:
        return ExtraEngineInfo(engine="exo", installed=False, status="not_installed", version=rec.get("commit"),
                               detail=f"installed EXO is commit {str(rec.get('commit'))[:10]}, this build pins "
                                      f"{EXO_PIN[:10]}; run: {INSTALL_HINT} --force")
    if not exo_binary(paths).is_file():
        return ExtraEngineInfo(engine="exo", installed=False, status="not_installed", version=EXO_PIN,
                               detail=f"EXO environment is incomplete ({exo_binary(paths)} missing); run: "
                                      f"{INSTALL_HINT} --force")
    return ExtraEngineInfo(engine="exo", installed=True, status="installed", version=EXO_PIN,
                           detail=f"EXO {EXO_VERSION} source build at {src_dir(paths)}")


# ----------------------------------------------------------------------------- prerequisites


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None


def _clean_env() -> dict[str, str]:
    """The gateway's own venv must not leak into EXO's uv project (uv run sets VIRTUAL_ENV)."""
    return {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH")}


def run_capture(argv: Sequence[str], timeout: float = 60.0) -> tuple[int, str]:
    try:
        out = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, env=_clean_env())
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]}: timed out after {timeout:.0f}s"
    except OSError as exc:
        return 126, f"{argv[0]}: {exc}"
    return out.returncode, (out.stdout + out.stderr).strip()


def run_streaming(argv: Sequence[str], cwd: Path) -> int:
    try:
        return subprocess.run(list(argv), cwd=str(cwd), env=_clean_env()).returncode
    except FileNotFoundError:
        return 127
    except OSError:
        return 126


def parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    return (int(m[1]), int(m[2]), int(m[3] or 0)) if m else None


def _fmt(v: tuple[int, ...]) -> str:
    return ".".join(str(x) for x in v)


def _versioned(run: Runner, name: str, argv: list[str], minimum: tuple[int, int, int] | None, fix: str) -> Check:
    code, out = run(argv)
    if code != 0:
        return Check(name, False, out.splitlines()[0] if out else f"{argv[0]} failed (exit {code})", fix)
    found = parse_version(out)
    if minimum and (found is None or found < minimum):
        have = _fmt(found) if found else "unknown"
        return Check(name, False, f"{name} {have} is older than {_fmt(minimum)}", fix)
    return Check(name, True, out.splitlines()[0] if out else name)


def _xcode(run: Runner) -> list[Check]:
    fix = ("install Xcode from the App Store (the Command Line Tools alone lack the Metal toolchain MLX compiles "
           "with), then: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer")
    code, out = run(["xcode-select", "-p"])
    if code != 0:
        return [Check("xcode", False, out or "xcode-select failed", fix)]
    if "CommandLineTools" in out or ".app/" not in out:
        return [Check("xcode", False, f"developer directory is {out}, not a full Xcode.app", fix)]
    code, ver = run(["xcodebuild", "-version"])
    if code != 0:
        return [Check("xcode", False, ver.splitlines()[0] if ver else "xcodebuild failed",
                      "open Xcode once and accept the license: sudo xcodebuild -license accept")]
    return [Check("xcode", True, f"{ver.splitlines()[0] if ver else 'Xcode'} at {out}")]


def check_prerequisites(run: Runner = run_capture) -> list[Check]:
    """Everything the pinned source build needs. The `python` check is fixable by the installer itself."""
    rust_fix = ("install stable Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh, then open a "
                "new terminal (or: rustup update stable)")
    checks = [*_xcode(run),
              _versioned(run, "git", ["git", "--version"], None, "xcode-select --install, or brew install git"),
              _versioned(run, "rustc", ["rustc", "--version"], MIN_RUST, rust_fix),
              _versioned(run, "cargo", ["cargo", "--version"], None, rust_fix),
              _versioned(run, "node", ["node", "--version"], MIN_NODE, "brew install node"),
              _versioned(run, "npm", ["npm", "--version"], None, "brew install node"),
              _versioned(run, "uv", ["uv", "--version"], MIN_UV, "brew install uv (or: brew upgrade uv)")]
    if next(c for c in checks if c.name == "uv").ok:
        code, out = run(["uv", "python", "find", PYTHON])
        checks.append(Check("python", code == 0, out.splitlines()[0] if code == 0 and out else
                            f"no Python {PYTHON} found by uv", f"uv python install {PYTHON} (the installer does this)"))
    return checks


# ----------------------------------------------------------------------------- install


def _checkout_at_pin(run: Runner, repo: Path) -> bool:
    code, out = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    return code == 0 and out.strip() == EXO_PIN


def _clone(paths: config.Paths, run: Runner, stream: Streamer, say: Callable[[str], None]) -> bool:
    """Clone into src.partial, check out the pin, verify HEAD, then swap it in as src."""
    src = src_dir(paths)
    if src.is_dir() and _checkout_at_pin(run, src):
        say(f"Reusing the checkout at {src} (commit {EXO_PIN[:10]})")
        return True
    partial = src.with_name("src.partial")
    shutil.rmtree(partial, ignore_errors=True)
    engine_dir(paths).mkdir(parents=True, exist_ok=True)
    if stream(["git", "clone", "--filter=blob:none", "--no-checkout", EXO_REPO, str(partial)], engine_dir(paths)):
        say(f"error: git clone of {EXO_REPO} failed\nfix:   check the network, then re-run: {INSTALL_HINT}")
        return False
    if stream(["git", "-C", str(partial), "checkout", "--detach", EXO_PIN], partial) or not _checkout_at_pin(
            run, partial):
        say(f"error: could not check out {EXO_PIN}\nfix:   re-run: {INSTALL_HINT} --force")
        return False
    shutil.rmtree(src, ignore_errors=True)
    partial.rename(src)
    return True


def _build(paths: config.Paths, stream: Streamer, say: Callable[[str], None]) -> bool:
    src = src_dir(paths)
    steps = [(["npm", "ci"], src / "dashboard", "dashboard dependencies (npm ci)"),
             (["npm", "run", "build"], src / "dashboard", "dashboard build (the EXO API will not start without it)"),
             (["uv", "sync", "--locked", "--extra", "mlx"], src, "Python environment (builds MLX and exo_rs)")]
    for argv, cwd, label in steps:
        say(f"==> {label}: {' '.join(argv)}")
        code = stream(argv, cwd)
        if code:
            fix = (f"if the MLX build failed on the Metal compiler, run `xcodebuild -downloadComponent "
                   f"MetalToolchain` (Xcode 26+); then re-run: {INSTALL_HINT}") if argv[0] == "uv" else \
                f"check the output above, then re-run: {INSTALL_HINT}"
            say(f"error: {label} failed (exit code {code})\nfix:   {fix}")
            return False
    return True


def _verify(paths: config.Paths, run: Runner, say: Callable[[str], None]) -> bool:
    exe = exo_binary(paths)
    if not exe.is_file():
        say(f"error: {exe} is missing after uv sync\nfix:   re-run: {INSTALL_HINT} --force")
        return False
    code, out = run([str(exe), "--version"])
    if code != 0 or EXO_VERSION not in out:
        say(f"error: `exo --version` did not report {EXO_VERSION}: {out[-300:]}\nfix:   re-run: {INSTALL_HINT} --force")
        return False
    return True


def _confirm(yes: bool, say: Callable[[str], None]) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        say("error: confirmation needed\nfix:   re-run with --yes")
        return False
    return input("Clone and build EXO now? [y/N] ").strip().lower() in ("y", "yes")


def install(paths: config.Paths, *, yes: bool, force: bool, say: Callable[[str], None],
            run: Runner = run_capture, stream: Streamer = run_streaming) -> int:
    reason = unsupported_reason()
    if reason:
        say(f"error: {reason}")
        return 2
    if status(paths).installed and not force:
        say(f"EXO {EXO_VERSION} (commit {EXO_PIN[:10]}) is already installed at {src_dir(paths)}")
        return 0
    checks = check_prerequisites(run)
    for c in checks:
        fix = "" if c.ok or not c.fix else f"\n          fix: {c.fix}"
        say(f"[{'OK' if c.ok else 'MISSING':7}] {c.name}: {c.detail}{fix}")
    if any(not c.ok and c.name != "python" for c in checks):
        say(f"Fix the items above, then re-run: {INSTALL_HINT}")
        return 1
    say(f"Will clone {EXO_REPO} at {EXO_PIN} into {src_dir(paths)}, build its dashboard and run "
        "`uv sync --locked --extra mlx` (compiles MLX from source: expect several GB of disk and 10-30 minutes).")
    if not _confirm(yes, say):
        say("Not installed.")
        return 1
    needs_python = any(c.name == "python" and not c.ok for c in checks)
    if needs_python and stream(["uv", "python", "install", PYTHON], paths.engines):
        say(f"error: uv could not install Python {PYTHON}\nfix:   uv python install {PYTHON}")
        return 1
    if not (_clone(paths, run, stream, say) and _build(paths, stream, say) and _verify(paths, run, say)):
        return 1
    marker_path(paths).write_text(json.dumps({"engine": "exo", "commit": EXO_PIN, "version": EXO_VERSION,
                                              "repo": EXO_REPO, "src": str(src_dir(paths)),
                                              "exe": str(exo_binary(paths)), "installed_at": utcnow().isoformat()},
                                             indent=2), encoding="utf-8")
    say(f"EXO {EXO_VERSION} installed. Share it with: uv run hypercompute join --host <host> --code <code> --exo")
    return 0
