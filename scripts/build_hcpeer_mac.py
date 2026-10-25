"""Build, check, install and roll back the hcpeer libggml-rpc.dylib for macOS peers (native/hcpeer).

hcpeer is llama.cpp b11770's libggml-rpc with a small server-side patch (native/hcpeer/hcpeer.patch): with
HC_PEER_MMAP=1 and `ggml-rpc-server -c`, weights that the peer's CPU device receives through the RPC tensor cache are
mapped from their cache files (clean pages macOS can drop and re-read) instead of copied into anonymous memory.
Details, install and rollback: docs/research/hcpeer-macos.md. Standard library only; every command prints JSON.

  uv run --no-project --with ziglang==0.16.0 python scripts/build_hcpeer_mac.py cross [--archive <release .tar.gz>]
  uv run --no-sync python scripts/build_hcpeer_mac.py native        # on a Mac with clang++ (xcode-select --install)
  uv run --no-sync python scripts/build_hcpeer_mac.py verify  [--engine-dir <bin dir>] [--dylib <file>]
  uv run --no-sync python scripts/build_hcpeer_mac.py install [--engine-dir <bin dir>] [--dylib <file>]
  uv run --no-sync python scripts/build_hcpeer_mac.py rollback|status [--engine-dir <bin dir>]
  uv run --no-sync python scripts/build_hcpeer_mac.py smoke [--dylib <file>]   # on the Mac: start a CPU rpc-server
  uv run --no-project --with ziglang==0.16.0 python scripts/build_hcpeer_mac.py selftest   # macOS or Linux
  uv run --no-sync python scripts/build_hcpeer_mac.py inspect <mach-o> [...]

Without --engine-dir the installed pinned engine is used: $HYPERCOMPUTE_HOME (default ~/.hypercompute)
/engines/llama-b11770-macos-arm64/<bin_subdir from its marker>.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native" / "hcpeer"))

from hcpeer_macho import (  # noqa: E402
    CPU_TYPE_ARM64,
    EXPORT_WEAK,
    FLAT_LOOKUP,
    MH_DYLIB,
    MachO,
    MachOError,
    describe,
    is_macho,
    parse_macho,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "native" / "hcpeer"
THIRD = SRC / "third_party" / "llama.cpp"
BUILD = SRC / "build"
OUT = BUILD / "libggml-rpc.dylib"
RELEASE_DIR = BUILD / "release"
SOURCES = (SRC / "src" / "ggml-rpc.cpp", SRC / "src" / "hcpeer.cpp", THIRD / "transport.cpp", THIRD / "log.cpp")
HEADERS = (SRC / "src" / "hcpeer.h",)
SELFTEST = SRC / "test" / "hcpeer_posix_test.cpp"
LOCK = ROOT / "configs" / "engines.lock.json"
VARIANT = "macos-arm64"
ENGINE_MARKER = ".hypercompute-engine.json"

# The release ships libggml-rpc.0.27.0.dylib (libggml-rpc.0.dylib and libggml-rpc.dylib are symlinks to it), and its
# one importer, libggml.0.dylib, loads it as @rpath/libggml-rpc.0.dylib. ggml-rpc-server is ad-hoc signed without the
# hardened runtime, so it accepts a dylib with a different ad-hoc signature (`verify` re-checks this).
LIB_REAL = "libggml-rpc.0.27.0.dylib"
INSTALL_NAME = "@rpath/libggml-rpc.0.dylib"
CURRENT_VERSION = "0.27.0"
COMPAT_VERSION = "0.0.0"
MACOS_MIN = "13.3"
BASE_LIB = "libggml-base.0.27.0.dylib"
SERVER = "ggml-rpc-server"
RELEASE_FILES = (LIB_REAL, BASE_LIB, "libggml.0.27.0.dylib", SERVER)
ORIGINAL_SHA256 = "547ac08b705a96e82ddb9fcfc2529472f97f5da1957984b3a7652cd5875d452d"
BACKUP_SUFFIX = ".hcpeer-orig"
TMP_SUFFIX = ".hcpeer-tmp"
MARKER = ".hcpeer.json"
SYMLINKS_FILE = "_symlinks.json"
SYSTEM_LIBS = ("/usr/lib/libSystem.B.dylib", "/usr/lib/libc++.1.dylib")
DEFINES = ("-DNDEBUG", "-DGGML_SHARED", "-DGGML_BUILD", "-DGGML_BACKEND_SHARED", "-DGGML_BACKEND_BUILD")
WARNINGS = ("-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-function", "-Wno-nullability-completeness")


class HcpeerError(RuntimeError):
    """A build, install or rollback step that cannot proceed safely."""


@dataclass(frozen=True)
class Report:
    ok: bool
    problems: tuple[str, ...]
    notes: tuple[str, ...]
    facts: dict[str, object]


# ----------------------------------------------------------------------------- engine directory


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_sha256(path: Path) -> str:
    """sha256 of a text source with LF line endings, so a Windows (CRLF) and a Mac checkout record the same value."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def engine_names(engine_dir: Path) -> dict[str, Path]:
    """File name -> real file, following symlinks and an extracted copy's _symlinks.json."""
    names = {p.name: p.resolve() for p in engine_dir.iterdir() if p.is_file()}
    links_file = engine_dir / SYMLINKS_FILE
    links = json.loads(links_file.read_text(encoding="utf-8")) if links_file.is_file() else {}
    for _ in range(len(links)):
        for link, target in links.items():
            if link not in names and target in names:
                names[link] = names[target]
    return names


def find_engine_dir(home: Path | None) -> Path:
    home = home or Path(os.environ.get("HYPERCOMPUTE_HOME") or Path.home() / ".hypercompute").expanduser()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))["llama_cpp"]
    root = home / "engines" / f"llama-{lock['release']}-{VARIANT}"
    marker = root / ENGINE_MARKER
    if not marker.is_file():
        raise HcpeerError(f"no installed {VARIANT} engine at {root}; pass --engine-dir or install the engine")
    return root / json.loads(marker.read_text(encoding="utf-8")).get("bin_subdir", "")


def original_lib(engine_dir: Path) -> Path:
    backup = engine_dir / (LIB_REAL + BACKUP_SUFFIX)
    return backup if backup.is_file() else engine_dir / LIB_REAL


# ----------------------------------------------------------------------------- static compatibility check


def _version_tuple(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in (v or "0").split("."))


def _check_layout(new: MachO, orig: MachO, server: MachO) -> list[str]:
    problems = []
    if new.cputype != CPU_TYPE_ARM64 or new.filetype != MH_DYLIB:
        problems.append("not an arm64 dylib")
    if new.dylib_id is None or orig.dylib_id is None or asdict(new.dylib_id) != asdict(orig.dylib_id):
        problems.append(f"install name/versions {new.dylib_id} differ from the release's {orig.dylib_id}")
    if _version_tuple(new.minos) > _version_tuple(server.minos):
        problems.append(f"needs macOS {new.minos}, newer than the release's {server.minos}")
    if "@loader_path" not in new.rpaths:
        problems.append("no @loader_path rpath, so @rpath/libggml-base.0.dylib would not resolve")
    return problems


def _check_signatures(new: MachO, server: MachO) -> list[str]:
    problems = []
    if new.signature is None or not new.signature.slots_match:
        problems.append("missing or inconsistent code signature (arm64 macOS kills unsigned code)")
    sig = server.signature
    if sig is not None:
        validates = "library-validation" in sig.flags or (
            "runtime" in sig.flags and "disable-library-validation" not in (sig.entitlements or ""))
        if validates:
            problems.append(f"{SERVER} has hardened runtime/library validation (team {sig.team_id or 'none'}): "
                            "it would refuse a dylib with a different signature")
    return problems


def _check_exports(new: MachO, orig: MachO, importers: dict[str, set[str]]) -> list[str]:
    problems = [f"{name} imports {sorted(syms - new.exports.keys())} which the new dylib does not export"
                for name, syms in importers.items() if syms - new.exports.keys()]
    api = {s for s in orig.exports if s.startswith("_ggml_")}
    if api - new.exports.keys():
        problems.append(f"missing release API exports {sorted(api - new.exports.keys())}")
    return problems


def _check_imports(new: MachO, libs: dict[str, MachO]) -> tuple[list[str], dict[str, int]]:
    problems: list[str] = []
    counts = {"engine": 0, "system": 0}
    every_export = set().union(*(m.exports.keys() for m in libs.values())) if libs else set()
    for sym, ordinal in new.imports:
        if ordinal == FLAT_LOOKUP:
            if sym not in every_export:
                problems.append(f"flat-lookup import {sym} is exported by no engine library")
            continue
        if not 0 < ordinal <= len(new.loads):
            problems.append(f"import {sym} has library ordinal {ordinal}")
            continue
        lib = new.loads[ordinal - 1].name
        if lib in SYSTEM_LIBS:
            counts["system"] += 1
        elif not lib.startswith("@rpath/") or lib.removeprefix("@rpath/") not in libs:
            problems.append(f"import {sym} binds to {lib}, which is not in the engine directory")
        elif sym not in libs[lib.removeprefix("@rpath/")].exports:
            problems.append(f"import {sym} is not exported by {lib}")
        else:
            counts["engine"] += 1
    return problems, counts


def _importers(engine_dir: Path, names: dict[str, Path], skip: set[Path]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, path in sorted(names.items()):
        if path in skip or name != path.name or not is_macho(path):
            continue
        m = parse_macho(path)
        ordinals = {i + 1 for i, d in enumerate(m.loads) if d.name == INSTALL_NAME}
        if ordinals:
            out[name] = {s for s, o in m.imports if o in ordinals}
    return out


def _engine_libs(names: dict[str, Path], exclude: Path) -> dict[str, MachO]:
    """Every libggml* name (symlink aliases included) -> its parsed file, each file parsed once."""
    parsed: dict[Path, MachO] = {}
    libs: dict[str, MachO] = {}
    for name, path in names.items():
        if name.startswith("libggml") and name.endswith(".dylib") and path != exclude and is_macho(path):
            if path not in parsed:
                parsed[path] = parse_macho(path)
            libs[name] = parsed[path]
    return libs


def check(engine_dir: Path, dylib: Path) -> Report:
    """Static checks that `dylib` can replace the release libggml-rpc in engine_dir. Nothing is executed."""
    names = engine_names(engine_dir)
    orig_path = original_lib(engine_dir)
    for needed in (SERVER, BASE_LIB):
        if needed not in names:
            raise HcpeerError(f"{engine_dir} has no {needed}; is this the {VARIANT} engine bin directory?")
    new, orig, server = parse_macho(dylib), parse_macho(orig_path), parse_macho(names[SERVER])
    libs = _engine_libs(names, orig_path.resolve())
    skip = {orig_path.resolve(), dylib.resolve()} | ({names[LIB_REAL]} if LIB_REAL in names else set())
    importers = _importers(engine_dir, names, skip)
    import_problems, counts = _check_imports(new, libs)
    problems = (_check_layout(new, orig, server) + _check_signatures(new, server)
                + _check_exports(new, orig, importers) + import_problems)
    orig_sha = sha256(orig_path)
    notes = [] if orig_sha == ORIGINAL_SHA256 else [f"original {orig_path.name} is not the pinned b11770 file"]
    weak = sorted(s for s, f in new.exports.items() if f & EXPORT_WEAK)
    if weak:
        notes.append(f"exports {len(weak)} weak definitions from a static libc++ (operator new/delete); they call "
                     "malloc/free like the system ones, so dyld coalescing them is harmless")
    facts = {"dylib": str(dylib), "sha256": sha256(dylib), "original": str(orig_path), "original_sha256": orig_sha,
             "importers": {k: sorted(v) for k, v in importers.items()}, "imports_checked": counts,
             "new": describe(new), "server_signature": asdict(server.signature) if server.signature else None}
    return Report(ok=not problems and orig_sha == ORIGINAL_SHA256, problems=tuple(problems), notes=tuple(notes),
                  facts=facts)


# ----------------------------------------------------------------------------- release fetch and build


def release_asset() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))["llama_cpp"]
    return lock["variants"][VARIANT]["assets"][0]


def fetch_release(archive: Path | None, dest: Path = RELEASE_DIR) -> Path:
    """Extract the release files a cross build links against and checks, from the pinned sha256-verified archive."""
    asset = release_asset()
    if archive is None:
        archive = BUILD / asset["name"]
        if not archive.is_file() or sha256(archive) != asset["sha256"]:
            part = archive.with_name(archive.name + ".part")
            BUILD.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(asset["url"], part)  # noqa: S310 - pinned https URL from the engine lock
            part.replace(archive)
    if sha256(archive) != asset["sha256"]:
        raise HcpeerError(f"{archive} does not match the pinned sha256 {asset['sha256']}")
    dest.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            name = Path(member.name).name
            if member.issym() and name.startswith("libggml"):
                links[name] = Path(member.linkname).name
            elif member.isfile() and name in RELEASE_FILES:
                (dest / name).write_bytes(tf.extractfile(member).read())  # type: ignore[union-attr]
    (dest / SYMLINKS_FILE).write_text(json.dumps(links, indent=1), encoding="utf-8")
    return dest


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def compile_args() -> list[str]:
    """Paths relative to the repo root (the build runs there), so no host path ends up in the binary."""
    return ["-std=c++17", "-O2", "-fPIC", "-g0", *DEFINES, *WARNINGS, f"-I{_rel(SRC / 'src')}", f"-I{_rel(THIRD)}",
            *(_rel(s) for s in SOURCES)]


def link_args(base_lib: Path, out: Path) -> list[str]:
    return [f"-Wl,-install_name,{INSTALL_NAME}", f"-Wl,-compatibility_version,{COMPAT_VERSION}",
            f"-Wl,-current_version,{CURRENT_VERSION}", "-Wl,-rpath,@loader_path", str(base_lib), "-o", str(out)]


def cross_command(base_lib: Path, out: Path) -> list[str]:
    """zig c++ for arm64 macOS: libc++ linked statically, stripped (no host paths), ad-hoc signed by zig's linker."""
    return [sys.executable, "-m", "ziglang", "c++", "-target", f"aarch64-macos.{MACOS_MIN}", "-shared", "-s",
            *compile_args(), *link_args(base_lib, out)]


def native_command(base_lib: Path, out: Path, cxx: str = "clang++") -> list[str]:
    """Apple clang on the Mac: links the system libc++ like the release, and ld64 ad-hoc signs arm64 output."""
    return [cxx, "-arch", "arm64", f"-mmacosx-version-min={MACOS_MIN}", "-dynamiclib", *compile_args(),
            *link_args(base_lib, out)]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace", check=False)


def build(cmd: list[str], engine_dir: Path, out: Path, toolchain: str) -> Report:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    result = _run(cmd)
    if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        tail = (result.stdout + result.stderr)[-4000:]
        raise HcpeerError(f"build failed (exit {result.returncode}):\n{tail}")
    report = check(engine_dir, out)
    meta = {"sha256": sha256(out), "toolchain": toolchain, "built_at": datetime.now(UTC).isoformat(),
            "sources": {str(s.relative_to(ROOT).as_posix()): source_sha256(s) for s in (*SOURCES, *HEADERS)},
            "release_original_sha256": ORIGINAL_SHA256, "verify_ok": report.ok}
    out.with_name(out.name + ".sha256").write_text(f"{meta['sha256']}  {out.name}\n", encoding="utf-8")
    out.with_name("build.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return report


def _tool_version(cmd: list[str]) -> str:
    result = _run(cmd)
    return (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else "unknown"


# ----------------------------------------------------------------------------- smoke test (runs on the Mac only)

CMD_ALLOC_BUFFER, CMD_GET_ALIGNMENT, CMD_BUFFER_GET_BASE, CMD_FREE_BUFFER = 0, 1, 3, 4
CMD_SET_TENSOR, CMD_SET_TENSOR_HASH, CMD_GET_TENSOR, CMD_GET_DEVICE_MEMORY, CMD_HELLO = 6, 7, 8, 11, 14
RPC_TENSOR = struct.Struct("<QIQ4I4II16ii10QQQQ64si")  # packed rpc_tensor of protocol 8.0.0: 296 bytes
GGML_TYPE_I8 = 24
CONN_CAPS_SIZE = 24
FNV_OFFSET, FNV_PRIME = 0xCBF29CE484222325, 0x100000001B3
SMOKE_SLOT = 1 << 20  # each smoke tensor sits at its own MiB, a multiple of every page size
BANNER_ON = "hcpeer mmap    : on"


def fnv1a64(data: bytes) -> int:
    """The RPC cache key: FNV-1a 64 over the tensor bytes (ggml-rpc.cpp fnv_hash)."""
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def encode_tensor(buffer: int, data: int, nbytes: int) -> bytes:
    """A 1-D GGML_TYPE_I8 rpc_tensor of nbytes at remote address data (the same layout linkprobe uses)."""
    return RPC_TENSOR.pack(1, GGML_TYPE_I8, buffer, nbytes, 1, 1, 1, 1, nbytes, nbytes, nbytes, 0, *([0] * 16), 0,
                           *([0] * 10), 0, 0, data, b"hcpeer_smoke", 0)


class RpcClient:
    """Just enough of RPC protocol 8 for the smoke test. Every socket call has the socket's timeout."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def send(self, cmd: int, payload: bytes) -> None:
        self.sock.sendall(struct.pack("<BQ", cmd, len(payload)) + payload)

    def _exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise HcpeerError("the rpc-server closed the connection (it drops a client on any rejected request)")
            buf += chunk
        return bytes(buf)

    def call(self, cmd: int, payload: bytes, reply_size: int) -> bytes:
        self.send(cmd, payload)
        (size,) = struct.unpack("<Q", self._exact(8))
        if size != reply_size:
            raise HcpeerError(f"command {cmd}: reply of {size} bytes, expected {reply_size}")
        return self._exact(size)


def short_file_probe(rpc: RpcClient, ptr: int, addr: int, nbytes: int, cache_dir: Path) -> dict[str, object]:
    """Plant what an interrupted stock save leaves (the first half of a payload under the payload's hash) and ask for
    it: hcpeer v3 must answer a miss and remove the file, leaving the tensor at addr as it was."""
    data = os.urandom(nbytes)
    short = cache_dir / f"{fnv1a64(data):016x}"
    short.write_bytes(data[: nbytes // 2])
    hit = rpc.call(CMD_SET_TENSOR_HASH, encode_tensor(ptr, addr, nbytes) + struct.pack("<QQ", 0, fnv1a64(data)), 1)
    return {"short_hit": hit[0], "short_file_left": short.exists()}


def smoke_exchange(rpc: RpcClient, nbytes: int, cache_dir: Path | None = None) -> dict[str, object]:
    """Cache save (mapped), aligned cache hit (mapped), unaligned cache hit (stock copy), with cache_dir a short cache
    file asked for at the aligned tensor (a miss), then read all three back."""
    major = rpc.call(CMD_HELLO, bytes(CONN_CAPS_SIZE), 28)[0]
    (alignment,) = struct.unpack("<Q", rpc.call(CMD_GET_ALIGNMENT, struct.pack("<I", 0), 8))
    ptr, _size = struct.unpack("<QQ", rpc.call(CMD_ALLOC_BUFFER, struct.pack("<IQ", 0, 4 * SMOKE_SLOT), 16))
    if ptr == 0:
        raise HcpeerError("the rpc-server could not allocate the 4 MiB smoke buffer")
    (base,) = struct.unpack("<Q", rpc.call(CMD_BUFFER_GET_BASE, struct.pack("<Q", ptr), 8))
    data = os.urandom(nbytes)
    addrs = (base, base + SMOKE_SLOT, base + 2 * SMOKE_SLOT + 64)
    rpc.send(CMD_SET_TENSOR, encode_tensor(ptr, addrs[0], nbytes) + struct.pack("<BQ", 1, 0) + data)
    hash_tail = struct.pack("<QQ", 0, fnv1a64(data))
    hits = [rpc.call(CMD_SET_TENSOR_HASH, encode_tensor(ptr, a, nbytes) + hash_tail, 1)[0] for a in addrs[1:]]
    short = short_file_probe(rpc, ptr, addrs[1], nbytes, cache_dir) if cache_dir is not None else {}
    get_tail = struct.pack("<QQ", 0, nbytes)
    same = [rpc.call(CMD_GET_TENSOR, encode_tensor(ptr, a, nbytes) + get_tail, nbytes) == data for a in addrs]
    rpc.send(CMD_FREE_BUFFER, struct.pack("<Q", ptr))
    rpc.call(CMD_GET_DEVICE_MEMORY, struct.pack("<I", 0), 16)  # answered only after FREE_BUFFER was handled
    return {"proto_major": major, "alignment": alignment, "base_page_aligned": alignment > 0 and base % alignment == 0,
            "hash": f"{fnv1a64(data):016x}", "nbytes": nbytes, "hits": hits, "readback_equal": same, **short}


def cache_listing(cache_dir: Path, name: str) -> dict[str, object]:
    """The saved cache file's size and any hcpeer temporary file left next to it (there must be none)."""
    entries = sorted(p.name for p in cache_dir.iterdir()) if cache_dir.is_dir() else []
    target = cache_dir / name
    return {"cache_file_bytes": target.stat().st_size if target.is_file() else None,
            "tmp_files": [e for e in entries if e.startswith(".hcpeer-")]}


def smoke_verdict(log: str, exchange: dict[str, object], page: int) -> dict[str, object]:
    mapped = log.count("[hcpeer] mapped ")
    fallbacks = log.count("copying instead")
    rejected = log.count("answering a cache miss")
    checks = {"banner_on": f"{BANNER_ON} (v" in log and f"page {page} B" in log,
              "alignment_is_page": exchange.get("alignment") == page, "hits": exchange.get("hits") == [1, 1],
              "readback_equal": exchange.get("readback_equal") == [True, True, True],
              "mapped_two": mapped == 2, "one_unaligned_fallback": fallbacks == 1,
              "short_file_missed": exchange.get("short_hit") == 0,
              "short_file_removed": exchange.get("short_file_left") is False, "one_rejection": rejected == 1,
              "cache_file_complete": exchange.get("nbytes") is not None
              and exchange.get("cache_file_bytes") == exchange.get("nbytes"),
              "no_tmp_left": exchange.get("tmp_files") == []}
    return {"ok": all(checks.values()), "checks": checks, "mapped_lines": mapped, "fallback_lines": fallbacks,
            "rejected_lines": rejected}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_banner(proc: subprocess.Popen, log_path: Path, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "transport" in text and ":" in text.split("transport", 1)[1]:
            return text
        if proc.poll() is not None:
            raise HcpeerError(f"{SERVER} exited with code {proc.returncode} before listening:\n{text[-2000:]}")
        time.sleep(0.2)
    raise HcpeerError(f"{SERVER} printed no banner within {timeout_s:.0f}s")


def _connect(port: int, timeout_s: float) -> socket.socket:
    """The banner is printed just before the server binds, so retry a refused connection until the deadline."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=timeout_s)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def _smoke_session(proc: subprocess.Popen, log_path: Path, port: int, page: int, timeout_s: float,
                   cache_dir: Path) -> tuple[dict[str, object], str | None]:
    """Run the exchange against a started server, then always stop it. Returns (exchange, error)."""
    try:
        _wait_banner(proc, log_path, timeout_s)
        with _connect(port, timeout_s) as sock:
            exchange = smoke_exchange(RpcClient(sock), 3 * page + 1000, cache_dir)
        time.sleep(0.5)  # let the server flush its log lines
        return exchange, None
    except (HcpeerError, OSError, struct.error) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def smoke(engine_dir: Path, dylib: Path | None, timeout_s: float = 30.0) -> dict[str, object]:
    """Start the engine's ggml-rpc-server on 127.0.0.1 with -d CPU -c and HC_PEER_MMAP=1, push three small tensors
    through the cache paths, ask for a planted short cache file (must be a miss) and read the tensors back. With
    --dylib the candidate library is loaded through
    DYLD_LIBRARY_PATH, so it can be tested before it is installed. Mac only; the server is always stopped."""
    if sys.platform != "darwin":
        raise HcpeerError("smoke starts the macOS ggml-rpc-server, so it only runs on the Mac")
    page = os.sysconf("SC_PAGE_SIZE")
    with tempfile.TemporaryDirectory(prefix="hcpeer-smoke-") as tmp:
        # floor 0: the smoke saves 50 KB and must not depend on how full this Mac's disk is
        env = {**os.environ, "HC_PEER_MMAP": "1", "HC_PEER_CACHE_MIN_FREE_GIB": "0",
               "LLAMA_CACHE": str(Path(tmp) / "cache")}
        if dylib is not None:
            (Path(tmp) / "lib").mkdir()
            shutil.copyfile(dylib, Path(tmp) / "lib" / INSTALL_NAME.removeprefix("@rpath/"))
            env["DYLD_LIBRARY_PATH"] = str(Path(tmp) / "lib")
        log_path, port = Path(tmp) / "server.log", _free_port()
        cmd = [str(engine_dir / SERVER), "-H", "127.0.0.1", "-p", str(port), "-d", "CPU", "-c"]
        with log_path.open("wb") as log_fh:
            proc = subprocess.Popen(cmd, cwd=engine_dir, env=env, stdin=subprocess.DEVNULL, stdout=log_fh,
                                    stderr=subprocess.STDOUT)
        cache_dir = Path(tmp) / "cache" / "rpc"
        exchange, error = _smoke_session(proc, log_path, port, page, timeout_s, cache_dir)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        exchange = {**exchange, **cache_listing(cache_dir, str(exchange.get("hash")))}
    verdict = smoke_verdict(log, exchange, page)
    if error is not None:
        verdict = {**verdict, "ok": False, "error": error}
    return {**verdict, "page": page, "exchange": exchange, "log_tail": log[-3000:]}


# ----------------------------------------------------------------------------- selftest (macOS or Linux)


def selftest_compiler() -> list[str]:
    """zig c++ when the ziglang wheel is importable (no Xcode needed on the Mac), else a working system c++. On macOS
    zig gets the same explicit target as the cross build, so it uses its bundled SDK stubs and never looks for Xcode."""
    if importlib.util.find_spec("ziglang") is not None:
        target = ["-target", f"aarch64-macos.{MACOS_MIN}"] if sys.platform == "darwin" else []
        return [sys.executable, "-m", "ziglang", "c++", *target]
    if sys.platform == "darwin" and _run(["xcode-select", "-p"]).returncode != 0:
        raise HcpeerError("no C++ compiler: run it as `uv run --no-project --with ziglang==0.16.0 python "
                          "scripts/build_hcpeer_mac.py selftest`")
    for cxx in ("c++", "clang++", "g++"):
        if shutil.which(cxx):
            return [cxx]
    raise HcpeerError("no C++ compiler found: add `--with ziglang==0.16.0` to the uv command")


def selftest_command(cxx: list[str], out: Path) -> list[str]:
    defines = [] if sys.platform == "darwin" else ["-DHCPEER_POSIX_TEST"]
    return [*cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function", *defines,
            f"-I{_rel(SRC / 'src')}", f"-I{_rel(THIRD)}", _rel(SRC / "src" / "hcpeer.cpp"), _rel(SELFTEST),
            "-o", str(out)]


def selftest(timeout_s: float = 120.0) -> dict[str, object]:
    """Build hcpeer.cpp with native/hcpeer/test/hcpeer_posix_test.cpp for this host and run it with hcpeer on and off:
    real files, mmap over page-aligned memory (vm_allocate on macOS, as ggml uses), rename and fork. POSIX only."""
    if os.name != "posix":
        raise HcpeerError("selftest runs the mmap code natively, so it needs macOS or Linux")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HC_PEER_")}
    runs: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="hcpeer-selftest-") as tmp:
        exe = Path(tmp) / "hcpeer_posix_test"
        built = _run(selftest_command(selftest_compiler(), exe))
        if built.returncode != 0:
            raise HcpeerError(f"selftest build failed:\n{(built.stdout + built.stderr)[-4000:]}")
        for mode in ("on", "off"):
            (Path(tmp) / mode).mkdir()
            r = subprocess.run([str(exe), str(Path(tmp) / mode), mode], capture_output=True, text=True,
                               errors="replace", env=env, timeout=timeout_s, check=False)
            runs[mode] = {"exit": r.returncode, "output": (r.stdout + r.stderr)[-6000:]}
    return {"ok": all(r["exit"] == 0 for r in runs.values()), "runs": runs}


# ----------------------------------------------------------------------------- install, rollback, status


def _write_marker(engine_dir: Path, dylib: Path) -> dict[str, object]:
    meta = {"installed": True, "sha256": sha256(engine_dir / LIB_REAL), "source": str(dylib),
            "backup": LIB_REAL + BACKUP_SUFFIX, "installed_at": datetime.now(UTC).isoformat()}
    (engine_dir / MARKER).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def install(engine_dir: Path, dylib: Path) -> dict[str, object]:
    """Replace the release libggml-rpc with `dylib`, keeping the pinned original as <name>.hcpeer-orig.

    The new file is written beside the old one and renamed over it, so it gets a new inode: rewriting a loaded,
    signed Mach-O in place can get the next process that maps it killed by the code-signing check. A running
    ggml-rpc-server keeps the old library until it restarts."""
    real, backup = engine_dir / LIB_REAL, engine_dir / (LIB_REAL + BACKUP_SUFFIX)
    if real.is_symlink() or not real.is_file():
        raise HcpeerError(f"{real} is missing or a symlink; refusing to replace it")
    report = check(engine_dir, dylib)
    if not report.ok:
        raise HcpeerError("static check failed: " + "; ".join(report.problems or report.notes))
    if not backup.exists():
        if sha256(real) != ORIGINAL_SHA256:
            raise HcpeerError(f"{real} is not the pinned release file and there is no backup; refusing")
        part = engine_dir / (LIB_REAL + BACKUP_SUFFIX + ".part")
        shutil.copy2(real, part)
        if sha256(part) != ORIGINAL_SHA256:  # a short backup (full disk) would make rollback impossible later
            part.unlink(missing_ok=True)
            raise HcpeerError("the backup copy of the release library is incomplete; nothing was replaced")
        os.replace(part, backup)
    elif sha256(backup) != ORIGINAL_SHA256:
        raise HcpeerError(f"{backup} is not the pinned release file; refusing to overwrite the library")
    tmp = engine_dir / (LIB_REAL + TMP_SUFFIX)
    shutil.copyfile(dylib, tmp)
    if sha256(tmp) != report.facts["sha256"]:
        tmp.unlink(missing_ok=True)
        raise HcpeerError(f"the copy of {dylib} differs from the checked file; nothing was replaced")
    tmp.chmod(0o755)
    os.replace(tmp, real)
    return _write_marker(engine_dir, dylib)


def rollback(engine_dir: Path) -> dict[str, object]:
    """Put the pinned original back (rename, new inode) and drop the marker. Idempotent."""
    real, backup = engine_dir / LIB_REAL, engine_dir / (LIB_REAL + BACKUP_SUFFIX)
    if not backup.is_file():
        if real.is_file() and sha256(real) == ORIGINAL_SHA256:
            (engine_dir / MARKER).unlink(missing_ok=True)
            return {"rolled_back": False, "stock": True, "reason": "already the release library"}
        raise HcpeerError(f"no backup {backup.name} and {real.name} is not the release file: reinstall the engine")
    if sha256(backup) != ORIGINAL_SHA256:
        raise HcpeerError(f"{backup} is not the pinned release file; not restoring it")
    os.replace(backup, real)
    (engine_dir / MARKER).unlink(missing_ok=True)
    return {"rolled_back": True, "stock": True, "sha256": ORIGINAL_SHA256}


def status(engine_dir: Path) -> dict[str, object]:
    real, backup, marker = engine_dir / LIB_REAL, engine_dir / (LIB_REAL + BACKUP_SUFFIX), engine_dir / MARKER
    digest = sha256(real) if real.is_file() else None
    meta = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
    return {"engine_dir": str(engine_dir), "library_sha256": digest, "stock": digest == ORIGINAL_SHA256,
            "hcpeer": bool(meta) and meta.get("sha256") == digest, "backup_present": backup.is_file(),
            "marker": meta}


# ----------------------------------------------------------------------------- CLI


def _engine_dir(args: argparse.Namespace) -> Path:
    return args.engine_dir if args.engine_dir else find_engine_dir(args.home)


def _cmd_cross(args: argparse.Namespace) -> dict:
    engine_dir = args.engine_dir or fetch_release(args.archive)
    version = _tool_version([sys.executable, "-m", "ziglang", "version"])
    report = build(cross_command(engine_dir / BASE_LIB, args.out), engine_dir, args.out, f"zig {version}")
    return asdict(report)


def _cmd_native(args: argparse.Namespace) -> dict:
    engine_dir = _engine_dir(args)
    # /usr/bin/clang++ exists even without the command line tools (it is a stub that opens an install dialog)
    no_tools = sys.platform == "darwin" and _run(["xcode-select", "-p"]).returncode != 0
    if no_tools or shutil.which(args.cxx) is None:
        raise HcpeerError(f"{args.cxx} is not usable: run `xcode-select --install` (needs someone at the Mac), or "
                          "use the `cross` command, which needs no Xcode")
    version = _tool_version([args.cxx, "--version"])
    report = build(native_command(engine_dir / BASE_LIB, args.out, args.cxx), engine_dir, args.out, version)
    return asdict(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cross", "native", "verify", "install", "rollback", "status", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--engine-dir", type=Path, help="the engine bin dir holding ggml-rpc-server")
        p.add_argument("--home", type=Path, help="Hypercompute home (default $HYPERCOMPUTE_HOME or ~/.hypercompute)")
        if name in ("cross", "native"):
            p.add_argument("--out", type=Path, default=OUT)
        if name in ("verify", "install"):
            p.add_argument("--dylib", type=Path, default=OUT)
        if name == "cross":
            p.add_argument("--archive", type=Path, help="local copy of the pinned macOS release archive")
        if name == "native":
            p.add_argument("--cxx", default="clang++")
        if name == "smoke":
            p.add_argument("--dylib", type=Path, default=None, help="test this library before installing it")
            p.add_argument("--timeout", type=float, default=30.0)
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("files", nargs="+", type=Path)
    sub.add_parser("selftest")
    return parser


def run(args: argparse.Namespace) -> tuple[dict | list, bool]:
    if args.command == "inspect":
        return [describe(parse_macho(f)) for f in args.files], True
    if args.command == "selftest":
        result = selftest()
        return result, bool(result["ok"])
    if args.command == "cross":
        result = _cmd_cross(args)
        return result, bool(result["ok"])
    if args.command == "native":
        result = _cmd_native(args)
        return result, bool(result["ok"])
    engine_dir = _engine_dir(args)
    if args.command == "verify":
        report = check(engine_dir, args.dylib)
        return asdict(report), report.ok
    if args.command == "smoke":
        result = smoke(engine_dir, args.dylib, args.timeout)
        return result, bool(result["ok"])
    actions = {"install": lambda: install(engine_dir, args.dylib), "rollback": lambda: rollback(engine_dir),
               "status": lambda: status(engine_dir)}
    return actions[args.command](), True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result, ok = run(args)
    except (HcpeerError, MachOError, OSError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
