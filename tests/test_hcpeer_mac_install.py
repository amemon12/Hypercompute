"""scripts/capacity/hcpeer_mac_install.sh under bash, with uname, codesign and xattr stubbed as on an Apple Silicon
Mac and a fake engine. The fake ggml-rpc-server prints the banner the real one would: an hcpeer line, with the
version read from the libggml-rpc.0.dylib next to it, the name dyld loads ("hcpeer-v<N>" in a fake library), only for
an hcpeer library. Nothing here runs macOS code; the real smoke has to run at a Mac (docs/research/hcpeer-macos.md,
"Running it").
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from test_build_scripts import BASH, _posix, _sh_stub, needs_bash

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capacity" / "hcpeer_mac_install.sh"
STOCK_LIB = b"stock libggml-rpc 0.27.0\n"
RPC_REAL = "libggml-rpc.0.27.0.dylib"  # the release's file; libggml-rpc.0.dylib and libggml-rpc.dylib link to it
# Git Bash cannot make native symlinks without a privilege; with this, ln -s and cp -R make links it reads back as
# links. Other platforms ignore it.
MSYS_LINKS = {"MSYS": "winsymlinks:sys"}
FAKE_SERVER = r"""d=$(dirname "$0")
echo "Starting RPC server v8.0.0"
echo "  endpoint       : $2:$4"
echo "  local cache    : $LLAMA_CACHE/rpc"
lib="$d/libggml-rpc.0.dylib"  # what dyld loads (@rpath/libggml-rpc.0.dylib)
[ -e "$lib" ] || lib="$d/libggml-rpc.dylib"  # the plain-file engines have only this name
v=$(sed -n 's/^hcpeer-v\([0-9]*\).*/\1/p' "$lib")
if [ -n "$v" ]; then
  if [ "$HC_PEER_MMAP" = 1 ]; then s="on (v$v, page 16384 B, no cache size cap)"; else s="off (v$v, set it)"; fi
  echo "  hcpeer mmap    : $s"
fi
echo "Devices:"
echo "  CPU: Apple M4 (16384 MiB, 9000 MiB free)"
echo "  transport      : TCP"
exec sleep 60"""

pytestmark = needs_bash


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    d = tmp_path / "stubs"
    d.mkdir()
    calls, unsigned, signed = (_posix(d / n) for n in ("calls.log", "unsigned", "signed"))
    _sh_stub(d, "uname", "case $1 in -m) echo arm64 ;; *) echo Darwin ;; esac")
    _sh_stub(d, "codesign", f'echo "codesign $*" >> "{calls}"\n'
                            f'if [ "$1" = --verify ] && [ -f "{unsigned}" ] && [ ! -f "{signed}" ]; then exit 1; fi\n'
                            f'if [ "$1" = -s ]; then touch "{signed}"; fi')
    _sh_stub(d, "xattr", f'echo "xattr $*" >> "{calls}"\nexit 1')  # nothing is quarantined
    return d


def bash(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([BASH, "-c", script, "bash", *args], capture_output=True, text=True, errors="replace",
                          env={**os.environ, **MSYS_LINKS}, timeout=60)


def symlink(d: Path, name: str, target: str) -> None:
    """d/name -> target, made by bash so that the script under test reads it back as a link; skips where it cannot."""
    r = bash('cd "$1" && ln -s "$2" "$3" && [ -L "$3" ]', _posix(d), target, name)
    if r.returncode != 0:
        pytest.skip(f"this bash cannot make a symlink: {r.stderr.strip()}")


def link_target(path: Path) -> str | None:
    """What the symlink at path points to, None for anything else (read through bash, as the script reads it)."""
    r = bash('[ -L "$1" ] && readlink "$1"', _posix(path))
    return r.stdout.strip() if r.returncode == 0 else None


def make_engine(tmp_path: Path, rpc_lib: bytes = STOCK_LIB, *, links: bool = False) -> Path:
    """links: the release layout, libggml-rpc.dylib -> libggml-rpc.0.dylib -> libggml-rpc.0.27.0.dylib."""
    bin_dir = tmp_path / "home" / "engines" / "llama-b11770-macos-arm64" / "llama-b11770"
    bin_dir.mkdir(parents=True)
    _sh_stub(bin_dir, "ggml-rpc-server", FAKE_SERVER)
    _sh_stub(bin_dir, "llama-server", "exit 0")
    if links:
        (bin_dir / RPC_REAL).write_bytes(rpc_lib)
        symlink(bin_dir, "libggml-rpc.0.dylib", RPC_REAL)
        symlink(bin_dir, "libggml-rpc.dylib", "libggml-rpc.0.dylib")
    else:
        (bin_dir / "libggml-rpc.dylib").write_bytes(rpc_lib)
    (bin_dir / "libggml-base.0.dylib").write_bytes(b"base\n")
    (bin_dir.parent / ".hypercompute-engine.json").write_text(json.dumps({"bin_subdir": "llama-b11770"}))
    return bin_dir


def make_build(tmp_path: Path, version: int = 4, original: bytes = STOCK_LIB, listed: str | None = None) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    lib = build / "libggml-rpc.dylib"
    lib.write_bytes(f"hcpeer-v{version} fake library\n".encode())
    digest = listed or hashlib.sha256(lib.read_bytes()).hexdigest()
    (build / "libggml-rpc.dylib.sha256").write_bytes(f"{digest}  libggml-rpc.dylib\n".encode())
    (build / "build.json").write_text(json.dumps({"sha256": digest,
                                                  "release_original_sha256": hashlib.sha256(original).hexdigest()}))
    return lib


def run(stubs: Path, tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    # the stubs go first on PATH inside bash: Git Bash converts the inherited Windows PATH at startup
    prelude = 'd=$1; if command -v cygpath >/dev/null 2>&1; then d=$(cygpath -u "$d"); fi; PATH="$d:$PATH"; '
    cmd = [BASH, "-c", prelude + 'shift; exec bash "$@"', "bash", _posix(stubs), _posix(SCRIPT), *args]
    full = {k: v for k, v in os.environ.items() if not k.startswith(("HC_PEER_", "HYPERCOMPUTE_"))}
    full.update(TMPDIR=_posix(tmp_path), **env)
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=full, timeout=180)


def install(stubs: Path, tmp_path: Path, engine: Path, lib: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return run(stubs, tmp_path, "--engine-dir", _posix(engine), "--dylib", _posix(lib), **env)


def quarantine_everything(stubs: Path, *, removable: bool = True) -> None:
    """xattr as on a Mac where every file and directory carries com.apple.quarantine until `xattr -d` removes it;
    with removable False, `xattr -d` fails."""
    calls, cleared = _posix(stubs / "calls.log"), _posix(stubs / "cleared.log")
    remove = f'echo "$3" >> "{cleared}"' if removable else "exit 1"
    _sh_stub(stubs, "xattr", f'echo "xattr $*" >> "{calls}"\n'
                             'case $1 in\n'
                             f'  -p) if [ -f "{cleared}" ] && grep -qxF "$3" "{cleared}"; then exit 1; fi; echo q ;;\n'
                             f'  -d) {remove} ;;\n'
                             '  *) exit 2 ;;\n'
                             'esac')


def snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_install_makes_a_checked_clone_and_leaves_the_stock_engine_alone(tmp_path, stubs):
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    before = snapshot(engine)
    r = install(stubs, tmp_path, engine, lib)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 1 and lines[0].endswith("/llama-b11770-hcpeer"), r.stdout
    clone = engine.parent / "llama-b11770-hcpeer"
    assert (clone / "libggml-rpc.dylib").read_bytes() == lib.read_bytes()
    assert (clone / "ggml-rpc-server").read_bytes() == before["ggml-rpc-server"]
    assert snapshot(engine) == before
    assert sorted(p.name for p in engine.parent.iterdir()) == [".hypercompute-engine.json", "llama-b11770",
                                                               "llama-b11770-hcpeer"]  # no staging left
    record = (clone / ".hcpeer-clone").read_text()
    assert f"hcpeer_sha256={hashlib.sha256(lib.read_bytes()).hexdigest()}" in record
    assert "smoke passed: hcpeer v4 on" in r.stderr
    calls = (stubs / "calls.log").read_text()
    assert "codesign --verify" in calls and "codesign -s" not in calls


def test_a_second_run_rechecks_the_clone_and_a_tampered_clone_is_rebuilt(tmp_path, stubs):
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    first = install(stubs, tmp_path, engine, lib)
    clone = engine.parent / "llama-b11770-hcpeer"
    record = (clone / ".hcpeer-clone").read_text()
    again = install(stubs, tmp_path, engine, lib)
    assert again.returncode == 0 and again.stdout == first.stdout, again.stderr
    assert "is up to date" in again.stderr and "smoke passed" in again.stderr
    assert (clone / ".hcpeer-clone").read_text() == record
    (clone / "libggml-rpc.dylib").write_bytes(STOCK_LIB)
    fixed = install(stubs, tmp_path, engine, lib)
    assert fixed.returncode == 0 and "cloning" in fixed.stderr, fixed.stderr
    assert (clone / "libggml-rpc.dylib").read_bytes() == lib.read_bytes()


@pytest.mark.parametrize(("case", "message"), [
    ("sha", "does not match"),
    ("release", "is not the release library hcpeer was built against"),
    ("in_place", "already has hcpeer in place"),
    ("old_version", "v4 or later is needed"),
])
def test_refusals_leave_no_clone_and_the_stock_engine_unchanged(tmp_path, stubs, case, message):
    lib = make_build(tmp_path, version=3 if case == "old_version" else 4,
                     original=b"another release\n" if case == "release" else STOCK_LIB,
                     listed="0" * 64 if case == "sha" else None)
    engine = make_engine(tmp_path, lib.read_bytes() if case == "in_place" else STOCK_LIB)
    before = snapshot(engine)
    r = install(stubs, tmp_path, engine, lib)
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert r.stdout == ""
    assert sorted(p.name for p in engine.parent.iterdir()) == [".hypercompute-engine.json", "llama-b11770"]
    assert snapshot(engine) == before


def test_re_signs_the_library_only_when_its_signature_does_not_verify(tmp_path, stubs):
    (stubs / "unsigned").write_text("")
    r = install(stubs, tmp_path, make_engine(tmp_path), make_build(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "re-signing libggml-rpc.dylib ad hoc" in r.stderr
    assert "codesign -s - -f" in (stubs / "calls.log").read_text()


def test_finds_the_installed_engine_without_engine_dir(tmp_path, stubs):
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    r = run(stubs, tmp_path, "--dylib", _posix(lib), HYPERCOMPUTE_HOME=_posix(tmp_path / "home"))
    assert r.returncode == 0, r.stderr
    assert (engine.parent / "llama-b11770-hcpeer" / ".hcpeer-clone").is_file()


def test_never_replaces_a_directory_it_did_not_make(tmp_path, stubs):
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    mine = engine.parent / "llama-b11770-hcpeer"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep")
    r = install(stubs, tmp_path, engine, lib)
    assert r.returncode == 1 and "was not made by this script" in r.stderr, r.stderr
    assert (mine / "notes.txt").read_text() == "keep"


def test_runs_only_on_an_apple_silicon_mac(tmp_path, stubs):
    _sh_stub(stubs, "uname", "echo Linux")
    r = install(stubs, tmp_path, make_engine(tmp_path), make_build(tmp_path))
    assert r.returncode == 1 and "runs on the Mac peer" in r.stderr, r.stderr


def test_the_release_symlink_layout_gets_hcpeer_in_the_file_dyld_loads_and_keeps_its_links(tmp_path, stubs):
    """The release ships libggml-rpc.0.27.0.dylib behind the links libggml-rpc.0.dylib (what dyld loads) and
    libggml-rpc.dylib: the clone's real file is replaced by rename, both links stay links to it, and the stock real
    file is unchanged. The smoke passing shows the fake server read the hcpeer library through libggml-rpc.0.dylib."""
    engine, lib = make_engine(tmp_path, links=True), make_build(tmp_path)
    before = snapshot(engine)
    r = install(stubs, tmp_path, engine, lib, **MSYS_LINKS)
    assert r.returncode == 0, r.stderr
    assert "smoke passed: hcpeer v4 on" in r.stderr
    clone = engine.parent / "llama-b11770-hcpeer"
    assert (clone / RPC_REAL).read_bytes() == lib.read_bytes()
    assert link_target(clone / "libggml-rpc.0.dylib") == RPC_REAL
    assert link_target(clone / "libggml-rpc.dylib") == "libggml-rpc.0.dylib"
    assert (engine / RPC_REAL).read_bytes() == STOCK_LIB and snapshot(engine) == before
    assert link_target(engine / "libggml-rpc.0.dylib") == RPC_REAL
    assert f"installed_file={RPC_REAL}\n" in (clone / ".hcpeer-clone").read_text()
    again = install(stubs, tmp_path, engine, lib, **MSYS_LINKS)
    assert again.returncode == 0 and "is up to date" in again.stderr, again.stderr


@pytest.mark.parametrize(("layout", "message"), [
    ("loads_another_file", "libggml-rpc.0.dylib (libggml-rpc.0.dylib) is not libggml-rpc.dylib"),
    ("links_outside", "libggml-rpc.dylib is missing or links outside its directory"),
])
def test_an_engine_whose_loaded_library_is_not_the_one_replaced_is_refused(tmp_path, stubs, layout, message):
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    if layout == "loads_another_file":  # dyld would load libggml-rpc.0.dylib, which the clone would never replace
        (engine / "libggml-rpc.0.dylib").write_bytes(b"another library\n")
    else:
        (engine.parent / RPC_REAL).write_bytes(STOCK_LIB)
        (engine / "libggml-rpc.dylib").unlink()
        symlink(engine, "libggml-rpc.dylib", f"../{RPC_REAL}")
    before = snapshot(engine)
    r = install(stubs, tmp_path, engine, lib, **MSYS_LINKS)
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert r.stdout == "" and snapshot(engine) == before
    assert not (engine.parent / "llama-b11770-hcpeer").exists()


def test_clears_the_quarantine_attribute_of_every_file_and_directory_in_the_clone_only(tmp_path, stubs):
    quarantine_everything(stubs)
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    r = install(stubs, tmp_path, engine, lib)
    assert r.returncode == 0, r.stderr
    cleared = (stubs / "cleared.log").read_text().splitlines()
    staging = [p for p in cleared if Path(p).name.startswith("llama-b11770-hcpeer.staging.")]
    assert len(staging) == 1, cleared  # the clone's directory, before it was moved into place
    assert all(p == staging[0] or p.startswith(staging[0] + "/") for p in cleared), cleared  # never the stock engine
    assert sorted(Path(p).name for p in cleared if p != staging[0]) == [
        "ggml-rpc-server", "libggml-base.0.dylib", "libggml-rpc.dylib", "llama-server"]


def test_a_quarantine_attribute_that_cannot_be_cleared_fails_the_install(tmp_path, stubs):
    """xattr -d fails inside the find | while pipeline: its die ends only that subshell, so this checks that pipefail
    and errexit stop the script before the smoke and that no clone or staging directory is left."""
    quarantine_everything(stubs, removable=False)
    engine, lib = make_engine(tmp_path), make_build(tmp_path)
    before = snapshot(engine)
    r = install(stubs, tmp_path, engine, lib)
    assert r.returncode == 1 and "cannot clear the quarantine attribute of" in r.stderr, r.stderr
    assert r.stdout == "" and "smoke" not in r.stderr
    assert sorted(p.name for p in engine.parent.iterdir()) == [".hypercompute-engine.json", "llama-b11770"]
    assert snapshot(engine) == before
