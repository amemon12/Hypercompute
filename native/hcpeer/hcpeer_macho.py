"""Read the parts of a thin arm64 Mach-O that decide whether a replacement dylib loads: the install name and
versions, dylib loads, rpaths, minimum macOS, dyld export trie, two-level imports, and the code signature (with every
code page hash re-checked). Standard library only, so it runs on any host; used by scripts/build_hcpeer_mac.py.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
MH_DYLIB = 6
CPU_TYPE_ARM64 = 0x0100000C
LC_SYMTAB, LC_LOAD_DYLIB, LC_ID_DYLIB, LC_CODE_SIGNATURE = 0x2, 0xC, 0xD, 0x1D
LC_BUILD_VERSION, LC_RPATH, LC_DYLD_INFO_ONLY, LC_DYLD_EXPORTS_TRIE = 0x32, 0x8000001C, 0x80000022, 0x80000033
DYLIB_LOADS = {LC_LOAD_DYLIB: "load", 0x80000018: "weak", 0x8000001F: "reexport", 0x20: "lazy", 0x80000023: "upward"}
N_STAB, N_TYPE, N_EXT, N_UNDF = 0xE0, 0x0E, 0x01, 0x0
FLAT_LOOKUP = 0xFE
EXPORT_WEAK = 0x04
CS_SUPERBLOB, CS_CODEDIRECTORY, CS_ENTITLEMENTS = 0xFADE0CC0, 0xFADE0C02, 0xFADE7171
CS_SLOT_CMS = 0x10000
CS_FLAGS = {0x2: "adhoc", 0x100: "hard", 0x200: "kill", 0x800: "restrict", 0x2000: "library-validation",
            0x10000: "runtime", 0x20000: "linker-signed"}
HASHES = {1: "sha1", 2: "sha256", 3: "sha256", 4: "sha384"}


class MachOError(ValueError):
    """The file is not a thin 64-bit Mach-O this reader understands."""


@dataclass(frozen=True)
class Dylib:
    kind: str
    name: str
    current: str
    compat: str


@dataclass(frozen=True)
class Signature:
    flags: tuple[str, ...]
    identifier: str
    team_id: str | None
    hash_type: str
    page_size: int
    code_slots: int
    code_limit: int
    slots_match: bool
    has_cms: bool
    entitlements: str | None


@dataclass(frozen=True)
class MachO:
    path: str
    cputype: int
    filetype: int
    dylib_id: Dylib | None
    loads: tuple[Dylib, ...]
    rpaths: tuple[str, ...]
    minos: str | None
    exports: dict[str, int] = field(default_factory=dict)  # dyld export trie: symbol -> export flags
    imports: tuple[tuple[str, int], ...] = ()  # (symbol, library ordinal: 1-based into loads, 0xFE flat lookup)
    signature: Signature | None = None


def _version(v: int) -> str:
    return f"{v >> 16}.{(v >> 8) & 0xFF}.{v & 0xFF}"


def _cstr(data: bytes, start: int, end: int | None = None) -> str:
    stop = data.find(b"\0", start, end)
    return data[start:stop if stop >= 0 else end].decode("utf-8", "replace")


def _uleb(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, pos


def export_trie(data: bytes, start: int, size: int) -> dict[str, int]:
    """Symbol -> flags from a dyld export trie: the symbols dyld can bind other images to."""
    out: dict[str, int] = {}
    stack, seen = [(0, "")], set()
    while stack:
        node, prefix = stack.pop()
        if node in seen or node >= size:
            raise MachOError(f"malformed export trie at node {node}")
        seen.add(node)
        terminal, pos = _uleb(data, start + node)
        if terminal:
            out[prefix] = _uleb(data, pos)[0]
        pos += terminal
        count, pos = data[pos], pos + 1
        for _ in range(count):
            end = data.index(b"\0", pos)
            label = data[pos:end].decode("utf-8", "replace")
            child, pos = _uleb(data, end + 1)
            stack.append((child, prefix + label))
    return out


def _imports(data: bytes, symoff: int, nsyms: int, stroff: int) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for i in range(nsyms):
        strx, ntype, _sect, desc, _value = struct.unpack_from("<IBBHQ", data, symoff + 16 * i)
        if not ntype & N_STAB and ntype & N_EXT and ntype & N_TYPE == N_UNDF:
            out.append((_cstr(data, stroff + strx), (desc >> 8) & 0xFF))
    return tuple(out)


def _code_slots_match(data: bytes, blob: bytes, cd: int, n_code: int, code_limit: int, hash_size: int,
                      algo: str, page: int, hash_off: int) -> bool:
    if not algo or page <= 0:
        return False
    for i in range(n_code):
        chunk = data[i * page:min((i + 1) * page, code_limit)]
        want = blob[cd + hash_off + i * hash_size:cd + hash_off + (i + 1) * hash_size]
        if hashlib.new(algo, chunk).digest()[:hash_size] != want:
            return False
    return True


def _code_directory(data: bytes, blob: bytes, cd: int, entitlements: str | None, has_cms: bool) -> Signature:
    (magic, _length, version, flags, hash_off, ident_off, _n_special, n_code, code_limit, hash_size, hash_type,
     _platform, page_log) = struct.unpack_from(">IIIIIIIIIBBBB", blob, cd)
    if magic != CS_CODEDIRECTORY:
        raise MachOError(f"code directory magic {magic:#x}")
    team_off = struct.unpack_from(">I", blob, cd + 48)[0] if version >= 0x20200 else 0
    algo = HASHES.get(hash_type, "")
    page = 1 << page_log if page_log else 0
    return Signature(
        flags=tuple(name for bit, name in CS_FLAGS.items() if flags & bit), identifier=_cstr(blob, cd + ident_off),
        team_id=_cstr(blob, cd + team_off) if team_off else None, hash_type=algo or str(hash_type), page_size=page,
        code_slots=n_code, code_limit=code_limit, has_cms=has_cms, entitlements=entitlements,
        slots_match=_code_slots_match(data, blob, cd, n_code, code_limit, hash_size, algo, page, hash_off))


def _signature(data: bytes, off: int, size: int) -> Signature:
    blob = data[off:off + size]
    magic, _length, count = struct.unpack_from(">III", blob, 0)
    if magic != CS_SUPERBLOB:
        raise MachOError(f"code signature superblob magic {magic:#x}")
    slots = dict(struct.unpack_from(">II", blob, 12 + 8 * i) for i in range(count))
    if 0 not in slots:
        raise MachOError("code signature has no code directory")
    entitlements = None
    for at in slots.values():
        blob_magic, length = struct.unpack_from(">II", blob, at)
        if blob_magic == CS_ENTITLEMENTS:
            entitlements = blob[at + 8:at + length].decode("utf-8", "replace")
    cms = slots.get(CS_SLOT_CMS)
    has_cms = cms is not None and struct.unpack_from(">I", blob, cms + 4)[0] > 8
    return _code_directory(data, blob, slots[0], entitlements, has_cms)


def _apply_command(fields: dict, data: bytes, cmd: int, off: int, size: int) -> None:
    if cmd == LC_ID_DYLIB or cmd in DYLIB_LOADS:
        name_off, _ts, cur, compat = struct.unpack_from("<IIII", data, off + 8)
        lib = Dylib(DYLIB_LOADS.get(cmd, "id"), _cstr(data, off + name_off, off + size), _version(cur),
                    _version(compat))
        if cmd == LC_ID_DYLIB:
            fields["dylib_id"] = lib
        else:
            fields["loads"].append(lib)
    elif cmd == LC_RPATH:
        fields["rpaths"].append(_cstr(data, off + struct.unpack_from("<I", data, off + 8)[0], off + size))
    elif cmd == LC_BUILD_VERSION:
        fields["minos"] = _version(struct.unpack_from("<I", data, off + 12)[0])
    elif cmd == LC_SYMTAB:
        symoff, nsyms, stroff, _strsize = struct.unpack_from("<IIII", data, off + 8)
        fields["imports"] = _imports(data, symoff, nsyms, stroff)
    elif cmd == LC_CODE_SIGNATURE:
        fields["signature"] = _signature(data, *struct.unpack_from("<II", data, off + 8))
    elif cmd in (LC_DYLD_EXPORTS_TRIE, LC_DYLD_INFO_ONLY):
        trie_off, trie_size = struct.unpack_from("<II", data, off + (8 if cmd == LC_DYLD_EXPORTS_TRIE else 40))
        if trie_size:
            fields["exports"] = export_trie(data, trie_off, trie_size)


def parse_macho(path: Path) -> MachO:
    data = path.read_bytes()
    if len(data) < 32:
        raise MachOError(f"{path}: too short for a Mach-O header")
    magic, cputype, _sub, filetype, ncmds, _sizeofcmds, _flags, _res = struct.unpack_from("<IiiIIIII", data, 0)
    if magic != MH_MAGIC_64:
        raise MachOError(f"{path}: not a thin 64-bit Mach-O (magic {magic:#x})")
    fields: dict = {"loads": [], "rpaths": [], "dylib_id": None, "minos": None}
    off = 32
    try:
        for _ in range(ncmds):
            cmd, size = struct.unpack_from("<II", data, off)
            if size < 8:
                raise MachOError(f"{path}: load command {cmd:#x} has size {size}")
            _apply_command(fields, data, cmd, off, size)
            off += size
    except (struct.error, IndexError) as exc:
        raise MachOError(f"{path}: truncated or malformed load commands: {exc}") from exc
    return MachO(path=str(path), cputype=cputype, filetype=filetype, loads=tuple(fields.pop("loads")),
                 rpaths=tuple(fields.pop("rpaths")), **fields)


def is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) == struct.pack("<I", MH_MAGIC_64)
    except OSError:
        return False


def describe(m: MachO) -> dict[str, object]:
    weak = sorted(s for s, f in m.exports.items() if f & EXPORT_WEAK)
    sig = asdict(m.signature) if m.signature else None
    return {"path": m.path, "arm64": m.cputype == CPU_TYPE_ARM64, "filetype": m.filetype,
            "id": asdict(m.dylib_id) if m.dylib_id else None, "loads": [asdict(d) for d in m.loads],
            "rpaths": list(m.rpaths), "minos": m.minos, "exports": len(m.exports), "weak_exports": weak,
            "imports": len(m.imports), "signature": sig}
