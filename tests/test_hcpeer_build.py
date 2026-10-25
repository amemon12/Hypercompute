"""hcpeer (native/hcpeer): the server-side ggml-rpc patch, the Mach-O reader, the static compatibility check, the
install/rollback file operations and the smoke-test protocol client.

Everything here is portable: Mach-O files are synthesized byte by byte (load commands, export trie, symbol table and
an ad-hoc code signature with real page hashes), the RPC smoke exchange runs against an in-process fake server, and
the one test that reads the real cross-built dylib skips unless `scripts/build_hcpeer_mac.py cross` has been run.
Nothing here can show runtime behaviour on macOS; `build_hcpeer_mac.py smoke` on a Mac does that.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HCPEER = ROOT / "native" / "hcpeer"
SELFTEST_SRC = HCPEER / "test" / "hcpeer_posix_test.cpp"
sys.path.insert(0, str(ROOT / "scripts"))

import build_hcpeer_mac as b  # noqa: E402
import hcpeer_macho as m  # noqa: E402  (build_hcpeer_mac puts native/hcpeer on sys.path)

PAGE = 16384
# sha256 of the vendored upstream files (LF line endings) from ggml-org/llama.cpp tag b11770
# (commit 39da130fcd1b6ca47d3c0522a40eccff41ea3dfb); ggml-rpc.cpp is the unpatched original.
PINNED = {
    "LICENSE": "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d",
    "ggml-alloc.h": "d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d",
    "ggml-backend-impl.h": "4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7",
    "ggml-backend.h": "0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2",
    "ggml-cpp.h": "bc15ba002c68685df23faae42df0d2b3d58032011f88b649021ab92485862c72",
    "ggml-impl.h": "43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3",
    "ggml-rpc.cpp": "583cf1ee149c5b94c215eca07a293564d1b755427b57b1a7846f0328e724eacd",
    "ggml-rpc.h": "a0bff02aa0a3c246683a0918ead7fdcc9cbbee38926bac47ea01cc7b68fbd66a",
    "ggml.h": "475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3",
    "gguf.h": "e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4",
    "log.cpp": "8a8d787eec2e4e8bbeb31d169fffd871a3ccf298be502a23aaaff790dfc77d2b",
    "log.h": "a295b4ba734c10d036d10c733f7c785bfee6296184197e45cb9697045d8d8664",
    "transport.cpp": "a5a6774d87398344a09da643de6ac175ad8188c91d8def25f6dad9c0e8957e34",
    "transport.h": "5859ae198de5c8df1d7184eb7eca13adab5cf2d5548910052756ad6d0ef53cc7",
}
EXPECTED_ADDED = [
    '#include "hcpeer.h"',
    "    alignment = hcpeer_alignment(buft, alignment, cache_dir != nullptr);",
    "        const int hc = hcpeer_save_cached(tensor, offset, data, size, cache_dir, hash_str);",
    "        if (hc == HCPEER_DONE) {",
    "            return true;",
    "        }",
    "        if (hc == HCPEER_COPY) {",
    "            ggml_backend_tensor_set(tensor, data, offset, size);",
    "            return true;",
    "        }",
    "    const int hc = hcpeer_map_cached(tensor, request.offset, cache_dir, request.hash);",
    "    if (hc != HCPEER_STOCK) {",
    "        response.result = hc == HCPEER_DONE ? 1 : 0;",
    "        return true;",
    "    }",
    "    hcpeer_log_banner(cache_dir);",
]


# ----------------------------------------------------------------------------- synthetic Mach-O


def _ver(text: str) -> int:
    major, minor, patch = (int(x) for x in text.split("."))
    return major << 16 | minor << 8 | patch


def _string_cmd(header: bytes, text: str) -> bytes:
    raw = text.encode() + b"\0"
    raw += b"\0" * (-(len(header) + 4 + len(raw)) % 8)
    return header + raw


def _dylib_cmd(cmd: int, name: str, current: str = "0.27.0", compat: str = "0.0.0") -> bytes:
    raw = name.encode() + b"\0"
    raw += b"\0" * (-(24 + len(raw)) % 8)
    return struct.pack("<IIIIII", cmd, 24 + len(raw), 24, 2, _ver(current), _ver(compat)) + raw


def _rpath_cmd(path: str) -> bytes:
    raw = path.encode() + b"\0"
    raw += b"\0" * (-(12 + len(raw)) % 8)
    return struct.pack("<III", m.LC_RPATH, 12 + len(raw), 12) + raw


def _trie(exports: dict[str, int]) -> bytes:
    """Root with one edge per full symbol name; child offsets as 2-byte ULEB128 (valid, not minimal)."""
    names = sorted(exports)
    root_len = 2 + sum(len(n) + 1 + 2 for n in names)
    children, edges = b"", b""
    for name in names:
        offset = root_len + len(children)
        edges += name.encode() + b"\0" + bytes([0x80 | (offset & 0x7F), offset >> 7])
        children += bytes([2, exports[name], 0, 0])  # terminal size 2: flags, address 0; no children
    return bytes([0, len(names)]) + edges + children


def _signature(body: bytes, code_limit: int, flags: int, team: str | None) -> bytes:
    ident, team_b = b"synthetic\0", (team.encode() + b"\0") if team else b""
    nslots = -(-code_limit // 4096)
    hashes = b"".join(hashlib.sha256(body[i * 4096:min((i + 1) * 4096, code_limit)]).digest() for i in range(nslots))
    cd_len = 88 + len(ident) + len(team_b) + len(hashes)
    team_off = 88 + len(ident) if team else 0
    hash_off = 88 + len(ident) + len(team_b)
    cd = struct.pack(">IIIIIIIIIBBBBIIIIQQQQ", m.CS_CODEDIRECTORY, cd_len, 0x20400, flags, hash_off, 88, 0, nslots,
                     code_limit, 32, 2, 0, 12, 0, 0, team_off, 0, 0, 0, 0, 0) + ident + team_b + hashes
    return struct.pack(">IIIII", m.CS_SUPERBLOB, 20 + len(cd), 1, 0, 20) + cd


def make_macho(*, filetype: int = 6, id_name: str | None = None, loads: tuple[str, ...] = (),
               rpaths: tuple[str, ...] = ("@loader_path",), minos: str = "13.3.0",
               exports: dict[str, int] | None = None, imports: tuple[tuple[str, int], ...] = (),
               flags: int = 0x20002, team: str | None = None, sign: bool = True) -> bytes:
    cmds = ([_dylib_cmd(m.LC_ID_DYLIB, id_name)] if id_name else []) + [_dylib_cmd(m.LC_LOAD_DYLIB, n) for n in loads]
    cmds += [_rpath_cmd(r) for r in rpaths]
    cmds.append(struct.pack("<IIIIII", m.LC_BUILD_VERSION, 24, 1, _ver(minos), _ver(minos), 0))
    sizeofcmds = sum(map(len, cmds)) + 24 + 16 + (16 if sign else 0)
    strtab, nlist = b"\0", b""
    for name, ordinal in imports:
        nlist += struct.pack("<IBBHQ", len(strtab), m.N_EXT | m.N_UNDF, 0, ordinal << 8, 0)
        strtab += name.encode() + b"\0"
    trie = _trie(exports or {})
    symoff = 32 + sizeofcmds
    stroff, trie_off = symoff + len(nlist), symoff + len(nlist) + len(strtab)
    code_limit = trie_off + len(trie) + (-(trie_off + len(trie)) % 16)
    cmds.append(struct.pack("<IIIIII", m.LC_SYMTAB, 24, symoff, len(imports), stroff, len(strtab)))
    cmds.append(struct.pack("<IIII", m.LC_DYLD_EXPORTS_TRIE, 16, trie_off, len(trie)))
    sig_len = len(_signature(b"\0" * code_limit, code_limit, flags, team))
    if sign:
        cmds.append(struct.pack("<IIII", m.LC_CODE_SIGNATURE, 16, code_limit, sig_len))
    header = struct.pack("<IiiIIIII", m.MH_MAGIC_64, m.CPU_TYPE_ARM64, 0, filetype, len(cmds), sizeofcmds, 0, 0)
    body = header + b"".join(cmds) + nlist + strtab + trie
    body += b"\0" * (code_limit - len(body))
    return body + _signature(body, code_limit, flags, team) if sign else body


RPC_ID = "@rpath/libggml-rpc.0.dylib"
BASE_ID = "@rpath/libggml-base.0.dylib"
SYSTEM = "/usr/lib/libSystem.B.dylib"


def new_lib(**overrides: object) -> bytes:
    spec: dict[str, object] = {
        "id_name": RPC_ID, "loads": (BASE_ID, SYSTEM),
        "exports": {"_ggml_backend_rpc_reg": 0, "_ggml_backend_rpc_start_server": 0, "__Znwm": m.EXPORT_WEAK},
        "imports": (("_ggml_nbytes", 1), ("_malloc", 2)),
    }
    return make_macho(**{**spec, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A synthetic release bin dir laid out like the extracted b11770 macOS archive."""
    d = tmp_path / "llama-b11770"
    d.mkdir()
    original = make_macho(id_name=RPC_ID, loads=(BASE_ID, SYSTEM), imports=(("_ggml_nbytes", 1),),
                          exports={"_ggml_backend_rpc_reg": 0, "_ggml_backend_rpc_start_server": 0})
    (d / b.LIB_REAL).write_bytes(original)
    (d / b.BASE_LIB).write_bytes(make_macho(id_name=BASE_ID, loads=(SYSTEM,),
                                            exports={"_ggml_nbytes": 0, "_ggml_backend_cpu_buffer_type": 0}))
    (d / "libggml.0.27.0.dylib").write_bytes(make_macho(id_name="@rpath/libggml.0.dylib", loads=(RPC_ID, BASE_ID),
                                                        imports=(("_ggml_backend_rpc_reg", 1),)))
    (d / b.SERVER).write_bytes(make_macho(filetype=2, loads=("@rpath/libggml.0.dylib", RPC_ID, BASE_ID, SYSTEM)))
    (d / b.SYMLINKS_FILE).write_text(json.dumps({"libggml-base.0.dylib": b.BASE_LIB, "libggml.0.dylib":
                                                 "libggml.0.27.0.dylib", "libggml-rpc.0.dylib": b.LIB_REAL}))
    monkeypatch.setattr(b, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    return d


def write(tmp_path: Path, data: bytes, name: str = "candidate.dylib") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ----------------------------------------------------------------------------- patch and sources


def test_vendored_sources_are_the_pinned_b11770_files() -> None:
    third = HCPEER / "third_party" / "llama.cpp"
    actual = {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in third.iterdir()}
    assert actual == PINNED


def test_patch_file_reproduces_the_patched_source() -> None:
    pristine = (HCPEER / "third_party" / "llama.cpp" / "ggml-rpc.cpp").read_text(encoding="utf-8")
    patched = (HCPEER / "src" / "ggml-rpc.cpp").read_text(encoding="utf-8")
    diff = "".join(difflib.unified_diff(pristine.splitlines(keepends=True), patched.splitlines(keepends=True),
                                       fromfile="a/ggml/src/ggml-rpc/ggml-rpc.cpp",
                                       tofile="b/ggml/src/ggml-rpc/ggml-rpc.cpp"))
    assert diff == (HCPEER / "hcpeer.patch").read_text(encoding="utf-8")


def test_patch_only_adds_the_five_server_side_hooks() -> None:
    lines = (HCPEER / "hcpeer.patch").read_text(encoding="utf-8").splitlines()
    added = [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")]
    removed = [line for line in lines if line.startswith("-") and not line.startswith("---")]
    assert removed == []
    assert added == EXPECTED_ADDED
    assert sum(line.startswith("@@") for line in lines) == 5


def test_mapping_code_is_private_cpu_only_and_env_gated() -> None:
    code = (HCPEER / "src" / "hcpeer.cpp").read_text(encoding="utf-8")
    assert "MAP_FIXED | MAP_PRIVATE, fd, 0" in code
    assert "MAP_SHARED" not in code
    assert "ggml_backend_buffer_get_type(buffer) == ggml_backend_cpu_buffer_type()" in code
    assert 'std::getenv("HC_PEER_MMAP")' in code and 'std::strcmp(value, "1") == 0' in code
    # every other platform compiles the stock behaviour; the test define is only for the Linux self-test
    assert "#if defined(__APPLE__) || defined(HCPEER_POSIX_TEST)" in code


def test_cache_saves_are_atomic_and_bounded_by_a_disk_floor() -> None:
    """Regression: the stock save truncates and rewrites <hash> in place, so a killed server or a full disk left a short
    file that the next load mapped as the weight, and a rewrite under a live mapping changes or kills it."""
    code = (HCPEER / "src" / "hcpeer.cpp").read_text(encoding="utf-8")
    save = code[code.index("bool save_atomic("):code.index("// The pid in a temporary file name")]
    assert 'dir + "/.hcpeer-" + name + "." + std::to_string((long) getpid()) + ".tmp"' in save
    assert "O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW" in save and "O_TRUNC" not in save
    assert save.index("write_all(") < save.index("fsync(fd)") < save.index("rename(tmp.c_str(), path.c_str())")
    assert "unlink(tmp.c_str())" in save and "keep_free && !has_room(dir, path, size)" in save
    assert 'std::getenv("HC_PEER_CACHE_MIN_FREE_GIB")' in code and "DEFAULT_MIN_FREE_GIB = 20" in code
    # atomic with hcpeer on or off: only an unsupported platform falls through to the stock in-place write
    hook_fn = code[code.index("int hcpeer_save_cached("):]
    assert "if (!HCPEER_SUPPORTED || cache_dir == nullptr || name == nullptr)" in hook_fn
    assert "save_atomic(cache_dir, name, data, size, on)" in hook_fn
    banner = code[code.index("void hcpeer_log_banner("):code.index("size_t hcpeer_alignment(")]
    assert banner.index("remove_stale_tmp(cache_dir)") < banner.index("hcpeer_enabled()")
    patched = (HCPEER / "src" / "ggml-rpc.cpp").read_text(encoding="utf-8")
    hook = patched.index("hcpeer_save_cached(tensor, offset, data, size, cache_dir, hash_str)")
    assert hook < patched.index("std::ofstream ofs(cache_file, std::ios::binary);")  # runs before the stock write


def test_cache_saves_keep_the_consented_cap_and_mapped_files_rank_as_used() -> None:
    """Regression: a mapped cache file keeps its disk blocks until it is unmapped, but the managed agent's guard_disk
    holds --max-disk by unlinking LRU files and counts each unlink as freed. hcpeer kept only the free-space floor, so
    it saved and mapped past the cap, and a mapped hit never refreshed the file's atime, so LRU could pick files in use.
    """
    code = (HCPEER / "src" / "hcpeer.cpp").read_text(encoding="utf-8")
    assert 'const char * value = std::getenv("HC_PEER_CACHE_MAX_GIB");' in code
    save = code[code.index("bool save_atomic("):code.index("// The pid in a temporary file name")]
    assert (save.index("keep_free && !has_room(dir, path, size)")
            < save.index("keep_free && !under_cap(dir, name, path, size)") < save.index("open(tmp.c_str()"))
    cap = code[code.index("bool under_cap("):code.index("// Write dir/name as a new file")]
    assert "dir_usage(dir, name, &used)" in cap and "used > cap || cap - used < size" in cap
    mapped = code[code.index("bool map_fd("):code.index("// SET_TENSOR: map the file just saved")]
    assert mapped.index("map_over(fd, dst, size, &why, &err)") < mapped.index("mark_used(tensor, fd, path)")
    assert "times[2] = {{0, UTIME_NOW}, {0, UTIME_OMIT}}" in code and "futimens(fd, times)" in code
    banner = code[code.index("void hcpeer_log_banner("):code.index("size_t hcpeer_alignment(")]
    assert "cap_clause()" in banner
    assert '#define HCPEER_VERSION "4"' in (HCPEER / "src" / "hcpeer.h").read_text(encoding="utf-8")
    assert "HC_PEER_CACHE_MAX_GIB" in SELFTEST_SRC.read_text(encoding="utf-8")
    doc = (ROOT / "docs" / "research" / "hcpeer-macos.md").read_text(encoding="utf-8")
    assert "HC_PEER_CACHE_MAX_GIB" in doc and "eviction cannot reclaim a mapped file" in doc


def test_cache_hits_reject_a_file_that_cannot_be_the_data() -> None:
    """Regression: stock SET_TENSOR_HASH trusts the cache file's size, so a short file left by an interrupted save was
    served as the whole weight. With hcpeer on, a file of the wrong size is removed and answered as a miss, for every
    buffer type (the size check runs before the CPU-only mapping), and the patch answers HCPEER_MISS with result 0."""
    code = (HCPEER / "src" / "hcpeer.cpp").read_text(encoding="utf-8")
    hit = code[code.index("int serve_hit("):code.index("// Write dir/name as a new file")]
    assert hit.index("plausible_size(tensor, offset, have)") < hit.index("is_cpu_buffer(tensor->buffer)")
    assert "if (err == ENOENT) {" in hit  # only a missing file is a silent stock miss
    reject = code[code.index("int reject_file("):code.index("// SET_TENSOR_HASH with hcpeer on")]
    assert "unlink(path)" in reject and "return HCPEER_MISS;" in reject
    rule = code[code.index("bool plausible_size("):code.index("// A cache file that cannot be the data")]
    assert "size == nbytes - offset" in rule and "outer >= 2" in rule and "offset % slice == 0" in rule
    patched = (HCPEER / "src" / "ggml-rpc.cpp").read_text(encoding="utf-8")
    hook = patched.index("const int hc = hcpeer_map_cached(tensor, request.offset, cache_dir, request.hash);")
    assert hook < patched.index("if (!get_cached_file(request.hash, cached_file)) {")
    assert "HCPEER_MISS  = 3" in (HCPEER / "src" / "hcpeer.h").read_text(encoding="utf-8")


def test_log_lines_are_rate_limited() -> None:
    code = (HCPEER / "src" / "hcpeer.cpp").read_text(encoding="utf-8")
    reporters = ("void report_fallback(", "void report_rejected(", "void report_mapped(", "void report_save_problem(")
    for reporter in reporters:
        body = code[code.index(reporter):]
        assert "if (!take_log_line())" in body[:body.index("\n}\n")], reporter
    assert "LOG_BURST       = 512.0" in code and "LOG_PER_SECOND  = 1.0" in code


def test_posix_selftest_harness_drives_every_hook() -> None:
    harness = SELFTEST_SRC.read_text(encoding="utf-8")
    for hook in ("hcpeer_save_cached(", "hcpeer_map_cached(", "hcpeer_alignment(", "hcpeer_log_banner("):
        assert hook in harness
    assert "vm_allocate(mach_task_self()" in harness  # the macOS run uses ggml's own allocator


def test_owned_files_have_no_em_dashes() -> None:
    owned = [HCPEER / "src" / "hcpeer.cpp", HCPEER / "src" / "hcpeer.h", HCPEER / "hcpeer_macho.py", SELFTEST_SRC,
             ROOT / "scripts" / "build_hcpeer_mac.py", ROOT / "docs" / "research" / "hcpeer-macos.md",
             Path(__file__)]
    for path in owned:
        assert chr(0x2014) not in path.read_text(encoding="utf-8"), path


# ----------------------------------------------------------------------------- Mach-O reader


def test_reader_parses_load_commands_trie_imports_and_signature(tmp_path: Path) -> None:
    lib = m.parse_macho(write(tmp_path, new_lib(team="TEAM123456")))
    assert lib.cputype == m.CPU_TYPE_ARM64 and lib.filetype == m.MH_DYLIB
    assert lib.dylib_id == m.Dylib("id", RPC_ID, "0.27.0", "0.0.0")
    assert [d.name for d in lib.loads] == [BASE_ID, SYSTEM]
    assert lib.rpaths == ("@loader_path",) and lib.minos == "13.3.0"
    assert lib.exports == {"_ggml_backend_rpc_reg": 0, "_ggml_backend_rpc_start_server": 0,
                           "__Znwm": m.EXPORT_WEAK}
    assert lib.imports == (("_ggml_nbytes", 1), ("_malloc", 2))
    sig = lib.signature
    assert sig is not None and sig.slots_match and sig.flags == ("adhoc", "linker-signed")
    assert sig.team_id == "TEAM123456" and sig.hash_type == "sha256" and sig.page_size == 4096
    assert m.describe(lib)["weak_exports"] == ["__Znwm"]


def test_reader_detects_a_modified_code_page(tmp_path: Path) -> None:
    data = bytearray(new_lib())
    data[40] ^= 0xFF  # inside the load commands, which the first code page hashes
    sig = m.parse_macho(write(tmp_path, bytes(data))).signature
    assert sig is not None and not sig.slots_match


def test_reader_rejects_non_macho_and_truncated_files(tmp_path: Path) -> None:
    with pytest.raises(m.MachOError):
        m.parse_macho(write(tmp_path, b"MZ" + b"\0" * 64))
    with pytest.raises(m.MachOError):
        m.parse_macho(write(tmp_path, new_lib()[:60]))
    assert not m.is_macho(write(tmp_path, b"#!/bin/sh\n", "script"))


# ----------------------------------------------------------------------------- static compatibility check


def test_check_accepts_a_compatible_replacement(engine: Path, tmp_path: Path) -> None:
    report = b.check(engine, write(tmp_path, new_lib()))
    assert report.ok, report.problems
    assert report.facts["importers"] == {"ggml-rpc-server": [], "libggml.0.27.0.dylib": ["_ggml_backend_rpc_reg"]}
    assert report.facts["imports_checked"] == {"engine": 1, "system": 1}
    assert any("weak definitions" in note for note in report.notes)


@pytest.mark.parametrize(("overrides", "needle"), [
    ({"exports": {"_ggml_backend_rpc_start_server": 0}}, "libggml.0.27.0.dylib imports"),
    ({"id_name": "@rpath/libggml-rpc.dylib"}, "install name"),
    ({"imports": (("_ggml_not_there", 1),)}, "not exported by @rpath/libggml-base.0.dylib"),
    ({"imports": (("_ggml_not_there", 0xFE),)}, "flat-lookup import"),
    ({"loads": ("@rpath/libelsewhere.dylib",), "imports": (("_x", 1),)}, "not in the engine directory"),
    ({"minos": "15.0.0"}, "needs macOS 15.0.0"),
    ({"rpaths": ()}, "no @loader_path rpath"),
    ({"sign": False}, "code signature"),
])
def test_check_names_each_incompatibility(engine: Path, tmp_path: Path, overrides: dict, needle: str) -> None:
    report = b.check(engine, write(tmp_path, new_lib(**overrides)))
    assert not report.ok
    assert any(needle in p for p in report.problems), report.problems


@pytest.mark.parametrize(("flags", "team"), [(0x12000, "TEAM123456"), (0x10002, None), (0x2002, None)])
def test_check_refuses_a_hardened_server_with_library_validation(engine: Path, tmp_path: Path, flags: int,
                                                                  team: str | None) -> None:
    """Regression: an ad-hoc signed (no team) server with the hardened runtime also refuses a foreign dylib."""
    (engine / b.SERVER).write_bytes(make_macho(filetype=2, loads=(RPC_ID,), flags=flags, team=team))
    report = b.check(engine, write(tmp_path, new_lib()))
    assert any("hardened runtime" in p for p in report.problems)


def test_check_is_not_ok_against_an_unpinned_original(engine: Path, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b, "ORIGINAL_SHA256", "0" * 64)
    report = b.check(engine, write(tmp_path, new_lib()))
    assert not report.ok and report.problems == ()
    assert any("not the pinned b11770 file" in n for n in report.notes)


# ----------------------------------------------------------------------------- install, rollback, status


def test_install_backs_up_replaces_by_rename_and_rolls_back(engine: Path, tmp_path: Path) -> None:
    original = (engine / b.LIB_REAL).read_bytes()
    candidate = write(tmp_path, new_lib())
    inode_before = (engine / b.LIB_REAL).stat().st_ino
    meta = b.install(engine, candidate)
    assert (engine / b.LIB_REAL).read_bytes() == candidate.read_bytes()
    assert (engine / (b.LIB_REAL + b.BACKUP_SUFFIX)).read_bytes() == original
    assert not (engine / (b.LIB_REAL + b.TMP_SUFFIX)).exists()
    assert meta["sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    if inode_before:  # Windows reports 0 on some filesystems
        assert (engine / b.LIB_REAL).stat().st_ino != inode_before
    state = b.status(engine)
    assert state["hcpeer"] and not state["stock"] and state["backup_present"]
    b.install(engine, candidate)  # reinstalling keeps the pinned original as the backup
    assert (engine / (b.LIB_REAL + b.BACKUP_SUFFIX)).read_bytes() == original
    assert b.rollback(engine)["rolled_back"] is True
    assert (engine / b.LIB_REAL).read_bytes() == original
    assert not (engine / (b.LIB_REAL + b.BACKUP_SUFFIX)).exists() and not (engine / b.MARKER).exists()
    assert b.status(engine)["stock"] and b.rollback(engine)["rolled_back"] is False


def test_install_leaves_the_original_alone_when_the_check_fails(engine: Path, tmp_path: Path) -> None:
    original = (engine / b.LIB_REAL).read_bytes()
    with pytest.raises(b.HcpeerError, match="static check failed"):
        b.install(engine, write(tmp_path, new_lib(id_name="@rpath/other.dylib")))
    assert (engine / b.LIB_REAL).read_bytes() == original
    assert not (engine / (b.LIB_REAL + b.BACKUP_SUFFIX)).exists()


def test_install_refuses_a_copy_that_differs_from_the_checked_file(engine: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: install renamed whatever the copy produced; a short or changed copy must never replace the lib."""
    original = (engine / b.LIB_REAL).read_bytes()
    (engine / (b.LIB_REAL + b.BACKUP_SUFFIX)).write_bytes(original)

    def bad_copy(src: Path, dst: Path, **_: object) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:-1])

    monkeypatch.setattr(b.shutil, "copyfile", bad_copy)
    with pytest.raises(b.HcpeerError, match="differs from the checked file"):
        b.install(engine, write(tmp_path, new_lib()))
    assert (engine / b.LIB_REAL).read_bytes() == original
    assert not (engine / (b.LIB_REAL + b.TMP_SUFFIX)).exists()


def test_install_refuses_an_incomplete_backup(engine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: an unverified short backup (full disk) left rollback impossible after the swap."""
    original = (engine / b.LIB_REAL).read_bytes()

    def short_copy2(src: Path, dst: Path, **_: object) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:-1])

    monkeypatch.setattr(b.shutil, "copy2", short_copy2)
    with pytest.raises(b.HcpeerError, match="backup copy"):
        b.install(engine, write(tmp_path, new_lib()))
    assert (engine / b.LIB_REAL).read_bytes() == original
    assert sorted(p.name for p in engine.iterdir() if "hcpeer" in p.name) == []


def test_install_and_rollback_refuse_an_unknown_library(engine: Path, tmp_path: Path) -> None:
    (engine / b.LIB_REAL).write_bytes((engine / b.LIB_REAL).read_bytes() + b"\0")
    with pytest.raises(b.HcpeerError):
        b.install(engine, write(tmp_path, new_lib()))
    with pytest.raises(b.HcpeerError, match="reinstall the engine"):
        b.rollback(engine)


def test_cli_status_and_inspect_print_json(engine: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert b.main(["status", "--engine-dir", str(engine)]) == 0
    assert json.loads(capsys.readouterr().out)["stock"] is True
    assert b.main(["inspect", str(write(tmp_path, new_lib()))]) == 0
    assert json.loads(capsys.readouterr().out)[0]["id"]["name"] == RPC_ID
    assert b.main(["install", "--engine-dir", str(engine), "--dylib", str(tmp_path / "missing.dylib")]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_find_engine_dir_reads_the_engine_marker(tmp_path: Path) -> None:
    root = tmp_path / "engines" / "llama-b11770-macos-arm64"
    root.mkdir(parents=True)
    with pytest.raises(b.HcpeerError):
        b.find_engine_dir(tmp_path)
    (root / b.ENGINE_MARKER).write_text(json.dumps({"bin_subdir": "llama-b11770"}))
    assert b.find_engine_dir(tmp_path) == root / "llama-b11770"


# ----------------------------------------------------------------------------- build commands and release fetch


def test_cross_and_native_commands() -> None:
    cross = b.cross_command(Path("rel") / b.BASE_LIB, Path("out.dylib"))
    assert cross[1:5] == ["-m", "ziglang", "c++", "-target"] and cross[5] == "aarch64-macos.13.3"
    for cmd in (cross, b.native_command(Path("rel") / b.BASE_LIB, Path("out.dylib"))):
        assert "-Wl,-install_name,@rpath/libggml-rpc.0.dylib" in cmd
        assert "-Wl,-compatibility_version,0.0.0" in cmd and "-Wl,-current_version,0.27.0" in cmd
        assert "-Wl,-rpath,@loader_path" in cmd and "-g0" in cmd
        assert "native/hcpeer/src/hcpeer.cpp" in cmd and not any(":\\" in a or a.startswith("/") for a in cmd[2:])
        assert cmd[-2:] == ["-o", "out.dylib"]
    assert "-s" in cross
    native = b.native_command(Path("rel") / b.BASE_LIB, Path("out.dylib"))
    assert native[:5] == ["clang++", "-arch", "arm64", "-mmacosx-version-min=13.3", "-dynamiclib"]


def _release_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "llama-b11770-bin-macos-arm64.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name in (*b.RELEASE_FILES, "llama-server", "libllama.0.7.0.dylib"):
            info = tarfile.TarInfo(f"llama-b11770/{name}")
            info.size = len(name)
            tf.addfile(info, io.BytesIO(name.encode()))
        link = tarfile.TarInfo("llama-b11770/libggml-rpc.0.dylib")
        link.type, link.linkname = tarfile.SYMTYPE, b.LIB_REAL
        tf.addfile(link)
    return archive


def test_fetch_release_extracts_only_the_needed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _release_archive(tmp_path)
    monkeypatch.setattr(b, "release_asset", lambda: {"name": archive.name, "url": "https://invalid",
                                                     "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    dest = b.fetch_release(archive, tmp_path / "release")
    assert sorted(p.name for p in dest.iterdir()) == sorted([*b.RELEASE_FILES, b.SYMLINKS_FILE])
    assert json.loads((dest / b.SYMLINKS_FILE).read_text()) == {"libggml-rpc.0.dylib": b.LIB_REAL}
    monkeypatch.setattr(b, "release_asset", lambda: {"name": archive.name, "url": "x", "sha256": "0" * 64})
    with pytest.raises(b.HcpeerError, match="pinned sha256"):
        b.fetch_release(archive, tmp_path / "release2")


@pytest.mark.skipif(not (b.OUT.is_file() and (b.RELEASE_DIR / b.SERVER).is_file()),
                    reason="run `uv run --no-project --with ziglang python scripts/build_hcpeer_mac.py cross` first")
def test_cross_built_dylib_passes_the_check_against_the_real_release() -> None:
    report = b.check(b.RELEASE_DIR, b.OUT)
    assert report.ok, report.problems
    new = m.parse_macho(b.OUT)
    assert new.signature is not None and new.signature.slots_match
    assert {"_ggml_backend_rpc_reg", "_ggml_backend_rpc_start_server"} <= new.exports.keys()
    assert [d.name for d in new.loads] == [BASE_ID, SYSTEM]
    meta = json.loads((b.BUILD / "build.json").read_text())
    assert meta["sha256"] == b.sha256(b.OUT) and meta["verify_ok"] is True


# ----------------------------------------------------------------------------- smoke protocol client


def test_rpc_tensor_encoding_matches_linkprobe() -> None:
    from hypercompute.capacity import linkprobe

    ours = b.RPC_TENSOR.unpack(b.encode_tensor(7, 0x1000, 50152))
    theirs = linkprobe.RPC_TENSOR.unpack(linkprobe.encode_tensor_1d(buffer=7, data=0x1000, nbytes=50152))
    assert b.RPC_TENSOR.size == linkprobe.RPC_TENSOR_SIZE == 296
    assert ours[:-2] == theirs[:-2]  # all fields but the name and use_count trailer
    assert ours[-1] == theirs[-1] == 0


def test_fnv1a64_known_vectors() -> None:
    assert b.fnv1a64(b"") == 0xCBF29CE484222325
    assert b.fnv1a64(b"a") == 0xAF63DC4C8601EC8C


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def _fake_server(listener: socket.socket, base: int) -> None:
    """Protocol-8 subset with a byte store standing in for the buffer and a dict for the cache directory."""
    conn, _ = listener.accept()
    mem: dict[int, bytes] = {}
    cache: dict[int, bytes] = {}

    def reply(payload: bytes) -> None:
        conn.sendall(struct.pack("<Q", len(payload)) + payload)

    with conn:
        while True:
            try:
                cmd, size = struct.unpack("<BQ", _recv_exact(conn, 9))
                payload = _recv_exact(conn, size)
            except EOFError:
                return
            addr = b.RPC_TENSOR.unpack(payload[:296])[41] if size >= 296 else 0
            if cmd == b.CMD_HELLO:
                reply(bytes([8, 0, 0, 0]) + bytes(24))
            elif cmd == b.CMD_GET_ALIGNMENT:
                reply(struct.pack("<Q", PAGE))
            elif cmd == b.CMD_ALLOC_BUFFER:
                reply(struct.pack("<QQ", 0xABC, struct.unpack("<IQ", payload)[1]))
            elif cmd == b.CMD_BUFFER_GET_BASE:
                reply(struct.pack("<Q", base))
            elif cmd == b.CMD_SET_TENSOR:
                flag, _offset = struct.unpack("<BQ", payload[296:305])
                mem[addr] = payload[305:]
                if flag:
                    cache[b.fnv1a64(payload[305:])] = payload[305:]
            elif cmd == b.CMD_SET_TENSOR_HASH:
                hit = struct.unpack("<QQ", payload[296:312])[1] in cache
                if hit:
                    mem[addr] = cache[struct.unpack("<QQ", payload[296:312])[1]]
                reply(bytes([int(hit)]))
            elif cmd == b.CMD_GET_TENSOR:
                offset, nbytes = struct.unpack("<QQ", payload[296:312])
                reply(mem.get(addr, b"")[offset:offset + nbytes].ljust(nbytes, b"\0"))
            elif cmd == b.CMD_GET_DEVICE_MEMORY:
                reply(struct.pack("<QQ", 1, 2))


def test_smoke_exchange_against_a_fake_server(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        server = threading.Thread(target=_fake_server, args=(listener, 1 << 32), daemon=True)
        server.start()
        with socket.create_connection(listener.getsockname(), timeout=10) as sock:
            exchange = b.smoke_exchange(b.RpcClient(sock), 3 * PAGE + 1000, tmp_path)
        server.join(timeout=10)
    assert exchange["proto_major"] == 8 and exchange["alignment"] == PAGE and exchange["base_page_aligned"]
    assert exchange["hits"] == [1, 1] and exchange["readback_equal"] == [True, True, True]
    assert exchange["nbytes"] == 3 * PAGE + 1000
    # the planted short file: half a payload under a hash the fake server never saw, so it answers a miss; the fake
    # does not delete files, which the real hcpeer server must (smoke_verdict checks short_file_removed)
    planted = list(tmp_path.iterdir())
    assert exchange["short_hit"] == 0 and exchange["short_file_left"] is True
    assert len(planted) == 1 and planted[0].stat().st_size == (3 * PAGE + 1000) // 2


SMOKE_LOG = ("  hcpeer mmap    : on (v4, page 16384 B, CPU weights mapped from the cache, "
             "cache saves keep 0 GiB free, no cache size cap)\n"
             "[set_tensor] saved to '/tmp/c/rpc/x'\n"
             "[hcpeer] mapped hcpeer_smoke, 0.05 MiB from x (1 tensors)\n"
             "[hcpeer] mapped hcpeer_smoke, 0.05 MiB from x (2 tensors)\n"
             "[hcpeer] hcpeer_smoke: tensor data is not page aligned, copying instead (x, 1 fallbacks so far)\n"
             "[hcpeer] hcpeer_smoke: removed the cache file, 25076 bytes cannot be the data asked for at offset 0 "
             "of a 50152-byte tensor, answering a cache miss (y, 1 rejected so far, 0 log lines dropped)\n")
SMOKE_EXCHANGE = {"alignment": PAGE, "hits": [1, 1], "readback_equal": [True, True, True], "nbytes": 50152,
                  "cache_file_bytes": 50152, "tmp_files": [], "short_hit": 0, "short_file_left": False}


def test_smoke_verdict_needs_the_banner_two_mappings_and_one_fallback() -> None:
    assert b.smoke_verdict(SMOKE_LOG, SMOKE_EXCHANGE, PAGE)["ok"]
    stock = b.smoke_verdict("Starting RPC server v8.0.0\n", SMOKE_EXCHANGE, PAGE)
    assert not stock["ok"] and not stock["checks"]["banner_on"] and stock["mapped_lines"] == 0


@pytest.mark.parametrize(("change", "failed"), [
    ({"cache_file_bytes": 49152}, "cache_file_complete"),
    ({"cache_file_bytes": None}, "cache_file_complete"),
    ({"tmp_files": [".hcpeer-0011223344556677.42.tmp"]}, "no_tmp_left"),
    ({"short_hit": 1}, "short_file_missed"),  # a pre-v3 library serves the short file as the tensor
    ({"short_file_left": True}, "short_file_removed"),
])
def test_smoke_verdict_fails_on_a_short_cache_file_or_a_leftover_temp(change: dict, failed: str) -> None:
    verdict = b.smoke_verdict(SMOKE_LOG, {**SMOKE_EXCHANGE, **change}, PAGE)
    assert not verdict["ok"] and not verdict["checks"][failed]


def test_smoke_verdict_needs_the_rejection_logged() -> None:
    log = SMOKE_LOG[:SMOKE_LOG.index("[hcpeer] hcpeer_smoke: removed")]
    verdict = b.smoke_verdict(log, SMOKE_EXCHANGE, PAGE)
    assert not verdict["ok"] and not verdict["checks"]["one_rejection"] and verdict["rejected_lines"] == 0


def test_cache_listing_reports_size_and_temp_files(tmp_path: Path) -> None:
    (tmp_path / "00aa").write_bytes(b"x" * 10)
    (tmp_path / ".hcpeer-00aa.7.tmp").write_bytes(b"x")
    (tmp_path / ".seed-00aa.part").write_bytes(b"x")
    assert b.cache_listing(tmp_path, "00aa") == {"cache_file_bytes": 10, "tmp_files": [".hcpeer-00aa.7.tmp"]}
    assert b.cache_listing(tmp_path / "missing", "00aa") == {"cache_file_bytes": None, "tmp_files": []}


def test_source_hashes_ignore_line_endings(tmp_path: Path) -> None:
    crlf, lf = write(tmp_path, b"a\r\nb\r\n", "crlf.cpp"), write(tmp_path, b"a\nb\n", "lf.cpp")
    assert b.source_sha256(crlf) == b.source_sha256(lf) == hashlib.sha256(b"a\nb\n").hexdigest()


@pytest.mark.skipif(not (b.BUILD / "build.json").is_file(), reason="no prebuilt dylib")
def test_prebuilt_dylib_was_built_from_these_sources() -> None:
    """The committed dylib must come from the committed sources: rebuild with `cross` after any source edit."""
    meta = json.loads((b.BUILD / "build.json").read_text(encoding="utf-8"))
    expected = {s.relative_to(ROOT).as_posix(): b.source_sha256(s) for s in (*b.SOURCES, *b.HEADERS)}
    assert meta["sources"] == expected
    assert meta["sha256"] == b.sha256(b.OUT) and meta["verify_ok"] is True
    assert (b.BUILD / "libggml-rpc.dylib.sha256").read_text(encoding="utf-8").split() == [meta["sha256"],
                                                                                        "libggml-rpc.dylib"]


def test_prebuilt_dylib_is_not_git_ignored() -> None:
    """Regression: the root .gitignore `build/` dropped the prebuilt dylib from the push to the Macs."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = [f"native/hcpeer/build/{n}" for n in ("libggml-rpc.dylib", "libggml-rpc.dylib.sha256", "build.json")]
    ignored = ["native/hcpeer/build/release/ggml-rpc-server", "native/hcpeer/build/llama-b11770-bin-macos-arm64.tar.gz"]
    for path in tracked:
        assert subprocess.run([git, "check-ignore", "-q", path], cwd=ROOT, check=False).returncode == 1, path
    for path in ignored:
        assert subprocess.run([git, "check-ignore", "-q", path], cwd=ROOT, check=False).returncode == 0, path


def test_selftest_needs_a_posix_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b.os, "name", "nt")
    with pytest.raises(b.HcpeerError, match="macOS or Linux"):
        b.selftest()


def test_selftest_command_builds_the_harness_with_hcpeer() -> None:
    cmd = b.selftest_command(["c++"], Path("out"))
    assert cmd[0] == "c++" and "-Werror" in cmd and cmd[-2:] == ["-o", "out"]
    assert "native/hcpeer/src/hcpeer.cpp" in cmd and "native/hcpeer/test/hcpeer_posix_test.cpp" in cmd
    assert ("-DHCPEER_POSIX_TEST" in cmd) == (sys.platform != "darwin")


def _posix_compiler() -> bool:
    return os.name == "posix" and (importlib.util.find_spec("ziglang") is not None
                                   or any(shutil.which(c) for c in ("c++", "clang++", "g++")))


@pytest.mark.skipif(not _posix_compiler(), reason="needs macOS or Linux and a C++ compiler")
def test_selftest_runs_the_mmap_code_natively() -> None:
    result = b.selftest()
    assert result["ok"], json.dumps(result, indent=1)


def test_smoke_refuses_to_run_off_the_mac(engine: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b.sys, "platform", "win32")
    with pytest.raises(b.HcpeerError, match="only runs on the Mac"):
        b.smoke(engine, None)
