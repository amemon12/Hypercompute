"""Tests for scripts/build_llama_windows.ps1, scripts/build_llama_linux.sh and scripts/build_llama_macos.sh.

Nothing here compiles llama.cpp. A stub cmake on PATH answers --version and fails any configure (leaving
a marker file), so a run that gets through the checkout stops at "cmake configure failed". The macOS script
runs under bash with uname, sysctl, xcode-select and clang++ stubbed to look like an Apple Silicon Mac. The lock
points at a github URL that git rewrites (url.<local>.insteadOf) to a throwaway local repo built with
plumbing, so its tree can hold paths this host cannot write. The Windows checkout tests also need the
Visual Studio C++ build tools, because the prerequisite phase finds them through vswhere before any
checkout; they skip without them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FAKE_REPO_URL = "https://github.com/hc-test/llama.cpp"
GIT = shutil.which("git")


def _find_bash() -> str | None:
    if os.name == "nt":
        # System32\bash.exe is the WSL launcher and would run the script inside a Linux VM.
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(candidate).exists():
                return candidate
        return None
    return shutil.which("bash")


BASH = _find_bash()
POWERSHELL = shutil.which("powershell") if os.name == "nt" else None
needs_git = pytest.mark.skipif(GIT is None, reason="git not available")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash (Git Bash on Windows) not available")
needs_powershell = pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell not available")


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _git(git_dir: Path, *args: str, stdin: bytes | None = None) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    proc = subprocess.run([GIT, "--git-dir", str(git_dir), *args], input=stdin, capture_output=True,
                          env=env, check=True)
    return proc.stdout.decode().strip()


def _tree(git_dir: Path, node: dict) -> str:
    lines = []
    for name, value in node.items():
        if isinstance(value, dict):
            lines.append(f"040000 tree {_tree(git_dir, value)}\t{name}")
        else:
            lines.append(f"100644 blob {_git(git_dir, 'hash-object', '-w', '--stdin', stdin=value)}\t{name}")
    return _git(git_dir, "mktree", stdin=("\n".join(lines) + "\n").encode())


def _fake_upstream(tmp_path: Path, files: dict[str, bytes]) -> tuple[dict[str, str], str]:
    """Bare repo holding FILES at one commit; returns (env for the script, commit sha)."""
    git_dir = tmp_path / "upstream.git"
    subprocess.run([GIT, "init", "-q", "--bare", str(git_dir)], check=True)
    root: dict = {}
    for path, content in files.items():
        *dirs, leaf = path.split("/")
        node = root
        for d in dirs:
            node = node.setdefault(d, {})
        node[leaf] = content
    commit = _git(git_dir, "commit-tree", _tree(git_dir, root), "-m", "fake llama.cpp")
    _git(git_dir, "update-ref", "refs/heads/master", commit)
    _git(git_dir, "symbolic-ref", "HEAD", "refs/heads/master")
    configs = tmp_path / "configs"
    configs.mkdir()
    lock = {"schema": 1, "llama_cpp": {"repo": FAKE_REPO_URL, "build_number": 1,
                                       "source_build": {"commit": commit}}}
    (configs / "engines.lock.json").write_text(json.dumps(lock), encoding="ascii")
    env = dict(os.environ, HYPERCOMPUTE_CONFIG_DIR=_posix(configs), GIT_CONFIG_COUNT="1",
               GIT_CONFIG_KEY_0=f"url.{git_dir.as_uri()}.insteadOf", GIT_CONFIG_VALUE_0=FAKE_REPO_URL)
    return env, commit


def _case_insensitive(dir_: Path) -> bool:
    (dir_ / "probe").write_text("x")
    return (dir_ / "PROBE").exists()


# ----------------------------------------------------------------------------- Windows PowerShell

WIN_SCRIPT = SCRIPTS / "build_llama_windows.ps1"
STUB_CMAKE_BAT = ("@echo off\r\nif \"%~1\"==\"--version\" (\r\n  echo cmake version 3.30.0\r\n  exit /b 0\r\n)\r\n"
                  "echo called>\"%~dp0configure-called\"\r\necho stub cmake: no configure here 1>&2\r\nexit /b 1\r\n")
FAKE_NVCC_BAT = ("@echo off\r\nif \"%~1\"==\"--version\" (\r\n  echo Cuda compilation tools, release 12.8, V12.8.93\r\n"
                 "  exit /b 0\r\n)\r\nif \"%~1\"==\"--list-gpu-arch\" (\r\n  echo compute_89\r\n  echo compute_120\r\n"
                 "  exit /b 0\r\n)\r\nexit /b 1\r\n")


def _ps(args: list[str], env: dict[str, str], stub_dir: Path | None = None) -> tuple[int, str]:
    if stub_dir is not None:
        env = dict(env, PATH=str(stub_dir) + os.pathsep + env.get("PATH", ""))
    cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WIN_SCRIPT), *args]
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=600)
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


@pytest.fixture
def win_stubs(tmp_path: Path) -> Path:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "cmake.bat").write_bytes(STUB_CMAKE_BAT.encode())
    code, out = _ps(["-Cpu", "-CheckOnly"], dict(os.environ), stubs)
    if code != 0:
        pytest.skip(f"Windows build prerequisites missing on this host: {out[-300:]}")
    return stubs


@needs_powershell
def test_windows_rejects_malformed_cuda_architectures() -> None:
    code, out = _ps(["-CudaArchitectures", "sm_120", "-CheckOnly"], dict(os.environ))
    assert code == 1
    assert "is not a CMAKE_CUDA_ARCHITECTURES list" in out


@needs_powershell
def test_windows_rejects_arch_that_nvcc_does_not_list(tmp_path: Path) -> None:
    bin_dir = tmp_path / "cuda" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "nvcc.bat").write_bytes(FAKE_NVCC_BAT.encode())
    (tmp_path / "cuda" / "extras" / "visual_studio_integration" / "MSBuildExtensions").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k.upper() != "CUDA_PATH"}
    code, out = _ps(["-CudaArchitectures", "120;121", "-CheckOnly"], env, bin_dir)
    assert code == 2, out
    assert "nvcc does not list compute_121" in out
    assert "does not list compute_120" not in out


def _path_without(*names: str) -> str:
    """os.environ PATH minus every directory holding one of NAMES (so a real cmake/uv cannot be found)."""
    kept = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        try:
            hit = any((Path(entry) / name).exists() for name in names)
        except OSError:
            hit = False
        if entry and not hit:
            kept.append(entry)
    return os.pathsep.join(kept)


NO_CMAKE_NO_UV = ("cmake.exe", "cmake.bat", "cmake.cmd", "uv.exe", "uv.bat", "uv.cmd")


def _env_without_cmake(home: Path) -> dict[str, str]:
    # PowerShell 5.1 derives $HOME from USERPROFILE; point it at an empty dir so a real ~\.local\bin is not seen.
    home.mkdir(parents=True, exist_ok=True)
    return dict(os.environ, PATH=_path_without(*NO_CMAKE_NO_UV), USERPROFILE=str(home))


VS_MISSING = "Visual Studio 2022 C++ build tools missing:"


def _assert_prereqs_ok_or_only_vs_missing(code: int, out: str) -> None:
    """Prerequisites pass, or the one problem listed is the missing VS C++ tools (a host without them)."""
    if code == 0:
        assert "prerequisites ok" in out, out
        return
    problems = [line.removeprefix("  - ") for line in out.splitlines() if line.startswith("  - ")]
    assert code == 2 and "cannot build yet. Fix these first:" in out, out
    assert len(problems) == 1 and problems[0].startswith(VS_MISSING), out


def test_prereq_helper_accepts_only_a_missing_visual_studio() -> None:
    header = "[hypercompute] cannot build yet. Fix these first:\n"
    _assert_prereqs_ok_or_only_vs_missing(2, f"{header}  - {VS_MISSING} winget install ...\n")
    # "Visual Studio" elsewhere in the output must not excuse an unrelated problem.
    unrelated = f"[hypercompute] using the CMake bundled with Visual Studio: C:\\vs\\cmake.exe\n{header}  - git x\n"
    for code, out in ((2, unrelated), (2, f"{header}  - {VS_MISSING} x\n  - git missing\n"), (1, VS_MISSING)):
        with pytest.raises(AssertionError):
            _assert_prereqs_ok_or_only_vs_missing(code, out)


def _uv_stub(stubs: Path, tool_bin: Path, exit_code: int = 0) -> None:
    """uv.bat: `uv tool dir --bin` prints TOOL_BIN and exits EXIT_CODE (leaving a uv-called marker)."""
    stubs.mkdir(exist_ok=True)
    bat = (f'@echo off\r\necho called>"%~dp0uv-called"\r\nif "%~1 %~2 %~3"=="tool dir --bin" (\r\n'
           f"  echo {tool_bin}\r\n  exit /b {exit_code}\r\n)\r\nexit /b 1\r\n")
    (stubs / "uv.bat").write_bytes(bat.encode())


@needs_powershell
@pytest.mark.parametrize("dir_name", ["uv-tools-bin", "uv tools [x64] bin"])  # [x64] is a wildcard to Get-Command
def test_windows_finds_cmake_in_uv_tool_bin_dir_off_path(tmp_path: Path, dir_name: str) -> None:
    tool_bin = tmp_path / dir_name
    tool_bin.mkdir()
    (tool_bin / "cmake.bat").write_bytes(STUB_CMAKE_BAT.encode())
    _uv_stub(tmp_path / "stubs", tool_bin)
    code, out = _ps(["-Cpu", "-CheckOnly"], _env_without_cmake(tmp_path / "home"), tmp_path / "stubs")
    assert f"using the one in uv's tool bin dir: {tool_bin / 'cmake.bat'}" in out, out
    _assert_prereqs_ok_or_only_vs_missing(code, out)


@needs_powershell
def test_windows_finds_cmake_in_home_local_bin_without_uv(tmp_path: Path) -> None:
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    (local_bin / "cmake.bat").write_bytes(STUB_CMAKE_BAT.encode())
    code, out = _ps(["-Cpu", "-CheckOnly"], _env_without_cmake(home))
    assert f"using the one in $HOME\\.local\\bin: {local_bin / 'cmake.bat'}" in out, out
    _assert_prereqs_ok_or_only_vs_missing(code, out)


@needs_powershell
def test_windows_falls_back_to_home_local_bin_when_uv_tool_dir_fails(tmp_path: Path) -> None:
    """A failing `uv tool dir --bin` is not trusted: what it printed is skipped, $HOME\\.local\\bin is not."""
    printed = tmp_path / "printed-bin"
    printed.mkdir()
    (printed / "cmake.bat").write_bytes(STUB_CMAKE_BAT.encode())
    _uv_stub(tmp_path / "stubs", printed, exit_code=1)
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    (local_bin / "cmake.bat").write_bytes(STUB_CMAKE_BAT.encode())
    code, out = _ps(["-Cpu", "-CheckOnly"], _env_without_cmake(home), tmp_path / "stubs")
    assert (tmp_path / "stubs" / "uv-called").exists(), "the uv stub was never asked"
    assert f"using the one in $HOME\\.local\\bin: {local_bin / 'cmake.bat'}" in out, out
    assert str(printed) not in out, out
    _assert_prereqs_ok_or_only_vs_missing(code, out)


@needs_powershell
def test_windows_missing_cmake_hint_offers_no_admin_route(tmp_path: Path) -> None:
    code, out = _ps(["-Cpu", "-CheckOnly"], _env_without_cmake(tmp_path / "home"))
    if "using the CMake bundled with Visual Studio" in out:
        pytest.skip("Visual Studio's bundled CMake is installed on this host")
    assert code == 2, out
    assert "cmake missing: winget install --id Kitware.CMake -e" in out, out
    assert "without admin rights: uv tool install cmake" in out, out


@needs_powershell
@needs_git
def test_windows_checks_out_paths_longer_than_max_path(tmp_path: Path, win_stubs: Path) -> None:
    src = tmp_path / "src" / "llama.cpp"
    # A directory Windows can still create (< 248 chars) holding a file whose full path is over 260:
    # without core.longpaths git reports "Filename too long" for the file yet exits 0.
    dir_len = 230 - len(str(src)) - 1
    if dir_len < 1:
        pytest.skip("temp dir path too long to build the fixture")
    long_rel = "d" * dir_len + "/" + "f" * 60 + ".txt"
    env, _ = _fake_upstream(tmp_path, {"CMakeLists.txt": b"project(x)\n", long_rel: b"long\n"})
    code, out = _ps(["-Cpu", "-SourceDir", str(src)], env, win_stubs)
    assert "Filename too long" not in out, out
    assert Path("\\\\?\\" + str(src / long_rel)).exists(), out
    assert code == 1 and "cmake configure failed" in out, out  # got past the checkout to the stub
    assert (win_stubs / "configure-called").exists()


@needs_powershell
@needs_git
def test_windows_stops_on_incomplete_checkout(tmp_path: Path, win_stubs: Path) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive file system to force a lossy checkout")
    env, commit = _fake_upstream(tmp_path, {"README": b"upper\n", "readme": b"lower\n"})
    code, out = _ps(["-Cpu", "-SourceDir", str(tmp_path / "src" / "llama.cpp")], env, win_stubs)
    assert code == 1, out
    assert f"checkout of {commit} left tracked files missing or modified" in out, out
    assert not (win_stubs / "configure-called").exists(), "cmake ran on an incomplete source tree"


# ----------------------------------------------------------------------------- Linux bash

LINUX_SCRIPT = SCRIPTS / "build_llama_linux.sh"


def _sh_stub(dir_: Path, name: str, body: str) -> None:
    path = dir_ / name
    path.write_bytes(("#!/bin/sh\n" + body + "\n").encode())
    path.chmod(0o755)


@pytest.fixture
def linux_stubs(tmp_path: Path) -> Path:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    marker = _posix(stubs / "configure-called")
    _sh_stub(stubs, "cmake", f'[ "$1" = --version ] && {{ echo "cmake version 3.30.0"; exit 0; }}\n'
                             f'echo called > "{marker}"; echo "stub cmake: no configure here" >&2; exit 1')
    _sh_stub(stubs, "c++", "exit 0")
    _sh_stub(stubs, "make", "exit 0")
    _sh_stub(stubs, "python3", f'exec "{_posix(sys.executable)}" "$@"')
    return stubs


def _bash(args: list[str], env: dict[str, str], stub_dir: Path, script: Path = LINUX_SCRIPT) -> tuple[int, str]:
    # Prepend the stubs inside bash: Git Bash converts the inherited Windows PATH at startup.
    prelude = 'd=$1; if command -v cygpath >/dev/null 2>&1; then d=$(cygpath -u "$d"); fi; PATH="$d:$PATH"; '
    cmd = [BASH, "-c", prelude + 'shift; exec bash "$@"', "bash", _posix(stub_dir), _posix(script), *args]
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=600)
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


@needs_bash
def test_linux_rejects_malformed_cuda_arch(linux_stubs: Path) -> None:
    code, out = _bash(["--cuda-arch", "sm_120", "--check-only"], dict(os.environ), linux_stubs)
    assert code == 1
    assert "is not a CMAKE_CUDA_ARCHITECTURES list" in out


@needs_bash
def test_linux_rejects_arch_that_nvcc_does_not_list(linux_stubs: Path) -> None:
    _sh_stub(linux_stubs, "nvcc", 'case $1 in --version) echo "Cuda compilation tools, release 12.8, V12.8.93" ;;\n'
                                  "  --list-gpu-arch) printf 'compute_89\\ncompute_120\\n' ;;\n  *) exit 1 ;; esac")
    code, out = _bash(["--cuda-arch", "120;121", "--check-only"], dict(os.environ), linux_stubs)
    assert code == 2, out
    assert "does not list compute_121" in out
    assert "does not list compute_120" not in out


@needs_bash
@needs_git
def test_linux_stops_on_incomplete_checkout(tmp_path: Path, linux_stubs: Path) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive file system to force a lossy checkout")
    env, commit = _fake_upstream(tmp_path, {"README": b"upper\n", "readme": b"lower\n"})
    code, out = _bash(["--cpu", _posix(tmp_path / "src" / "llama.cpp")], env, linux_stubs)
    assert code == 1, out
    assert f"checkout of {commit} left tracked files missing or modified" in out, out
    assert not (linux_stubs / "configure-called").exists(), "cmake ran on an incomplete source tree"


# ----------------------------------------------------------------------------- macOS bash

MACOS_SCRIPT = SCRIPTS / "build_llama_macos.sh"


@pytest.fixture
def mac_stubs(linux_stubs: Path) -> Path:
    """The Linux stubs (cmake, python3) plus the Darwin tools the macOS script asks: an arm64 Mac with the CLT."""
    _sh_stub(linux_stubs, "uname", "case $1 in -m) echo arm64 ;; *) echo Darwin ;; esac")
    _sh_stub(linux_stubs, "sysctl", "case $2 in sysctl.proc_translated) echo 0 ;; hw.logicalcpu) echo 4 ;;\n"
                                    "  *) exit 1 ;; esac")
    _sh_stub(linux_stubs, "xcode-select", "echo /Library/Developer/CommandLineTools")
    _sh_stub(linux_stubs, "clang++", "exit 0")
    return linux_stubs


@needs_bash
@needs_git
def test_macos_builds_from_a_complete_checkout(tmp_path: Path, mac_stubs: Path) -> None:
    env, _ = _fake_upstream(tmp_path, {"CMakeLists.txt": b"project(x)\n"})
    code, out = _bash([_posix(tmp_path / "src" / "llama.cpp")], env, mac_stubs, MACOS_SCRIPT)
    assert code == 1 and "cmake configure failed" in out, out  # got past the checkout to the stub
    assert (mac_stubs / "configure-called").exists()


@needs_bash
@needs_git
def test_macos_stops_on_incomplete_checkout(tmp_path: Path, mac_stubs: Path) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive file system to force a lossy checkout")
    env, commit = _fake_upstream(tmp_path, {"README": b"upper\n", "readme": b"lower\n"})
    code, out = _bash([_posix(tmp_path / "src" / "llama.cpp")], env, mac_stubs, MACOS_SCRIPT)
    assert code == 1, out
    assert f"checkout of {commit} left tracked files missing or modified" in out, out
    assert not (mac_stubs / "configure-called").exists(), "cmake ran on an incomplete source tree"
