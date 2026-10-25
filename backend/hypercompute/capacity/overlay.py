"""Overlay engine directory: the pinned llama.cpp release with the hcstream proxy ggml.dll in front of the real one.

The overlay is content addressed (its name carries the proxy DLL's hash), built from hard links so it costs no
disk space beyond the 0.2 MB proxy, and never touches the installed engine that live deployments use. Release files
are linked, never copied: copying cuBLAS would cost ~550 MB on a nearly full disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import config, engine

PROXY_NAME = "ggml.dll"
ORIG_NAME = "ggml_hcorig.dll"
MARKER = ".hypercompute-overlay.json"
DLL_ENV = "HYPERCOMPUTE_HCSTREAM_DLL"
REPO_DLL = Path(__file__).resolve().parents[3] / "native" / "hcstream" / "build" / PROXY_NAME


class OverlayError(RuntimeError):
    pass


@dataclass(frozen=True)
class Overlay:
    bin_dir: Path
    proxy_sha256: str
    engine_dir: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_proxy_dll() -> Path | None:
    override = os.environ.get(DLL_ENV)
    candidate = Path(override) if override else REPO_DLL
    return candidate if candidate.is_file() else None


def supported() -> bool:
    """The extension hooks Windows import tables of the pinned release; other hosts use the stock engine."""
    return engine.host_os() == "windows"


def ensure_overlay(engine_dir: Path, proxy_dll: Path, root: Path | None = None) -> Overlay:
    if not supported():
        raise OverlayError("the hcstream extension is Windows-only")
    real = engine_dir / PROXY_NAME
    if not real.is_file():
        raise OverlayError(f"{real} is missing; the engine layout is not the pinned Windows release")
    digest = _sha256(proxy_dll)
    root = root or config.paths().home / "overlays"
    target = root / f"{engine_dir.name}-hcstream-{digest[:12]}"
    if _valid(target, engine_dir, digest):
        return Overlay(bin_dir=target, proxy_sha256=digest, engine_dir=engine_dir)
    tmp = root / f".{target.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for src in engine_dir.iterdir():
            if not src.is_file() or src.name in (PROXY_NAME, engine.MARKER):
                continue
            os.link(src, tmp / src.name)
        os.link(real, tmp / ORIG_NAME)
        shutil.copy2(proxy_dll, tmp / PROXY_NAME)
        (tmp / MARKER).write_text(json.dumps({"engine_dir": str(engine_dir), "proxy_sha256": digest,
                                              "orig": ORIG_NAME}, indent=2), encoding="utf-8")
        if target.exists():  # a concurrent builder won the race; theirs is identical by construction
            shutil.rmtree(tmp)
        else:
            tmp.rename(target)
    except OSError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise OverlayError(f"could not build the hcstream overlay (hard links need the same volume): {exc}") from exc
    if not _valid(target, engine_dir, digest):
        raise OverlayError(f"overlay {target} failed validation after it was built")
    return Overlay(bin_dir=target, proxy_sha256=digest, engine_dir=engine_dir)


def _valid(target: Path, engine_dir: Path, digest: str) -> bool:
    try:
        meta = json.loads((target / MARKER).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("proxy_sha256") != digest or Path(meta.get("engine_dir", "")) != engine_dir:
        return False
    if _sha256(target / PROXY_NAME) != digest:
        return False
    try:
        if not os.path.samefile(target / ORIG_NAME, engine_dir / PROXY_NAME):
            return False
        for src in engine_dir.iterdir():
            if src.is_file() and src.name not in (PROXY_NAME, engine.MARKER):
                if not os.path.samefile(target / src.name, src):
                    return False
    except OSError:
        return False
    return True
