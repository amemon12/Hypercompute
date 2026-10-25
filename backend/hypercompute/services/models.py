"""Curated model manifest, verified downloads and GGUF inspection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from .. import config, gguf
from ..engine import EngineError, download_verified
from ..schemas import GgufSummary, ModelArtifact, ModelFile, ModelView
from .events import EventBus

log = logging.getLogger("hypercompute.models")
VERIFIED_MARKER = ".verified.json"
# The vision projector has its own marker: a model verified before its projector was fetched stays downloaded.
PROJECTOR_MARKER = ".verified-mmproj.json"


class ModelError(RuntimeError):
    pass


def load_manifest(path: Path | None = None) -> list[ModelArtifact]:
    cfg = config.configs_dir()
    path = path or (cfg / "models.yaml" if (cfg / "models.yaml").exists() else cfg / "models.example.yaml")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ModelError(f"cannot read model manifest {path}: {exc}") from exc
    return [ModelArtifact.model_validate(m) for m in raw.get("models", [])]


def _plain_relative(text: str) -> bool:
    """No drive, no root and no '..' under both Windows and POSIX rules: on Windows '/x' and '\\x' are rooted on the
    current drive and 'D:x' is drive-relative, though none of them is_absolute(), and each escapes a join."""
    return all(not p.anchor and ".." not in p.parts for p in (PureWindowsPath(text), PurePosixPath(text)))


def _inside(path: Path, root: Path) -> bool:
    """The resolved path stays under root (a model id or a symlink cannot lead out of the models dir either)."""
    resolved, base = path.resolve(), root.resolve()
    return resolved != base and resolved.is_relative_to(base)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelService:
    def __init__(self, events: EventBus | None = None, models_dir: Path | None = None) -> None:
        self._events = events
        self._dir = models_dir or config.paths().models
        self._artifacts = {a.id: a for a in load_manifest()}
        self._progress: dict[str, int] = {}
        self._errors: dict[str, str] = {}
        self._downloading: set[str] = set()
        self._gguf_cache: dict[str, GgufSummary] = {}

    def artifacts(self) -> list[ModelArtifact]:
        return list(self._artifacts.values())

    def get(self, model_id: str) -> ModelArtifact:
        try:
            return self._artifacts[model_id]
        except KeyError:
            raise ModelError(f"unknown model {model_id!r}") from None

    def model_dir(self, model_id: str) -> Path:
        return self._dir / model_id

    def file_paths(self, model_id: str) -> list[Path]:
        art = self.get(model_id)
        return [self._local(model_id, f) for f in art.files]

    def _local(self, model_id: str, f: ModelFile) -> Path:
        """HF repos keep their subdirectories (f.path); GGUF files sit flat in the model dir."""
        text = f.path or f.name
        dest = self.model_dir(model_id) / text
        if not _plain_relative(text) or not _inside(dest, self._dir):
            raise ModelError(f"unsafe file path in manifest for {model_id}: {text}")
        return dest

    def primary_path(self, model_id: str) -> Path:
        """GGUF: the first shard (llama.cpp finds the rest). HF/MLX: the model directory."""
        art = self.get(model_id)
        if art.format != "gguf" or not art.files:
            return self.model_dir(model_id)
        return self.file_paths(model_id)[0]

    def _verified(self, model_id: str) -> bool:
        return self._marker_ok(model_id, VERIFIED_MARKER, self.get(model_id).files)

    def _projector_verified(self, model_id: str) -> bool:
        art = self.get(model_id)
        return art.mmproj is not None and self._marker_ok(model_id, PROJECTOR_MARKER, [art.mmproj])

    def projector_path(self, model_id: str) -> Path | None:
        """The verified vision projector (llama.cpp --mmproj), or None when the model has none or it is not here."""
        art = self.get(model_id)
        return self._local(model_id, art.mmproj) if art.mmproj and self._projector_verified(model_id) else None

    def _marker_ok(self, model_id: str, marker_name: str, files: list[ModelFile]) -> bool:
        marker = self.model_dir(model_id) / marker_name
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        for f in files:
            p = self._local(model_id, f)
            rec = meta.get(f.path or f.name) or {}
            if not p.is_file() or rec.get("sha256") != f.sha256 or rec.get("size") != p.stat().st_size \
                    or p.stat().st_size != f.size_bytes or rec.get("mtime") != int(p.stat().st_mtime):
                return False
        return True

    def _write_marker(self, model_id: str, marker_name: str = VERIFIED_MARKER,
                      files: list[ModelFile] | None = None) -> None:
        meta = {}
        for f in self.get(model_id).files if files is None else files:
            p = self._local(model_id, f)
            meta[f.path or f.name] = {"sha256": f.sha256, "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
        (self.model_dir(model_id) / marker_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def adopt_cached(self, model_id: str, candidates: list[Path]) -> bool:
        """Copy already-downloaded files (e.g. ~/.hypercompute/downloads) in after verifying SHA256."""
        art = self.get(model_id)
        by_name = {c.name: c for c in candidates if c.is_file()}
        if not all(f.name in by_name for f in art.files):
            return False
        self.model_dir(model_id).mkdir(parents=True, exist_ok=True)
        for f in art.files:
            src = by_name[f.name]
            if src.stat().st_size != f.size_bytes or _sha256(src) != f.sha256:
                return False
            dst = self.model_dir(model_id) / f.name
            if not (dst.is_file() and dst.stat().st_size == f.size_bytes and _sha256(dst) == f.sha256):
                tmp = dst.with_name(dst.name + ".adopting")  # a copy interrupted here never looks complete
                shutil.copy2(src, tmp)
                tmp.replace(dst)
        self._write_marker(model_id)
        return True

    def summary(self, model_id: str) -> GgufSummary | None:
        if model_id in self._gguf_cache:
            return self._gguf_cache[model_id]
        if self.get(model_id).format != "gguf" or not self._verified(model_id):
            return None
        try:
            s = gguf.summarize(self.file_paths(model_id))
        except (gguf.GgufError, OSError) as exc:
            self._errors[model_id] = f"GGUF inspection failed: {exc}"
            return None
        self._gguf_cache[model_id] = s
        return s

    def view(self, model_id: str) -> ModelView:
        art = self.get(model_id)
        if not art.files:  # EXO/MLX: every node's engine downloads the repo itself
            return ModelView(artifact=art, status="engine_managed", local_path=None, downloaded_bytes=0,
                             total_bytes=art.size_bytes or 0, verified_sha256=False, gguf=None)
        total = sum(f.size_bytes for f in art.files)
        verified = self._verified(model_id)
        if model_id in self._downloading and not verified:  # a verified model fetching its projector stays usable
            status = "downloading"
        elif verified:
            status = "downloaded"
        elif model_id in self._errors:
            status = "error"
        else:
            status = "not_downloaded"
        done = total if verified else self._progress.get(model_id, 0)
        return ModelView(artifact=art, status=status, local_path=str(self.primary_path(model_id)) if verified else None,
                         downloaded_bytes=done, total_bytes=total, verified_sha256=verified,
                         gguf=self.summary(model_id) if verified else None, error=self._errors.get(model_id),
                         projector_status=self._projector_status(model_id))

    def _projector_status(self, model_id: str) -> str:
        if self.get(model_id).mmproj is None:
            return "none"
        if self._projector_verified(model_id):
            return "downloaded"
        return "downloading" if model_id in self._downloading else "not_downloaded"

    def views(self) -> list[ModelView]:
        return [self.view(a.id) for a in self.artifacts()]

    async def download(self, model_id: str) -> None:
        art = self.get(model_id)
        if not art.files:
            raise ModelError(f"{art.display_name} is downloaded by its engine on each node, not by Hypercompute")
        projector_missing = art.mmproj is not None and not self._projector_verified(model_id)
        if model_id in self._downloading or (self._verified(model_id) and not projector_missing):
            return
        self._downloading.add(model_id)
        self._errors.pop(model_id, None)
        self._progress[model_id] = 0
        if self._events:
            self._events.publish("model.download", f"Downloading {art.display_name}", model_id=model_id)
        try:
            await asyncio.to_thread(self._download_files, art)
            if self._events:
                self._events.publish("model.download", f"Downloaded and verified {art.display_name}",
                                     model_id=model_id, done=True)
        except (EngineError, OSError) as exc:
            self._errors[model_id] = str(exc)
            if self._events:
                self._events.publish("model.download", f"Download failed: {exc}", model_id=model_id, error=True)
            raise ModelError(str(exc)) from exc
        finally:
            self._downloading.discard(model_id)

    def _download_files(self, art: ModelArtifact) -> None:
        """The model's files unless already verified, then its vision projector; each set gets its marker once its
        hashes match, so a failed projector leaves the model usable for text."""
        base = 0
        self.model_dir(art.id).mkdir(parents=True, exist_ok=True)
        files = [] if self._verified(art.id) else art.files
        projector = [art.mmproj] if art.mmproj is not None and not self._projector_verified(art.id) else []
        # Hashing completed shards stays in this worker thread, never on the gateway event loop.
        remaining = self._remaining_download_bytes(art, [*files, *projector])
        if shutil.disk_usage(self._dir).free < remaining + 2 * 1024**3:
            raise EngineError(f"not enough disk: need {remaining / 1024**3:.1f} GiB more + 2 GiB headroom")
        for f in files:
            def progress(done: int, _total: int, base: int = base) -> None:
                self._progress[art.id] = base + done
            dest = self._local(art.id, f)
            dest.parent.mkdir(parents=True, exist_ok=True)
            download_verified(f.url, dest, f.sha256, f.size_bytes, progress)
            base += f.size_bytes
        if files:
            self._write_marker(art.id)
        for f in projector:
            download_verified(f.url, self._local(art.id, f), f.sha256, f.size_bytes)
            self._write_marker(art.id, PROJECTOR_MARKER, projector)

    def _remaining_download_bytes(self, art: ModelArtifact, files: list[ModelFile]) -> int:
        remaining = 0
        for f in files:
            dest = self._local(art.id, f)
            if dest.is_file() and dest.stat().st_size == f.size_bytes and _sha256(dest) == f.sha256:
                continue
            part = dest.with_name(dest.name + ".part")
            have = part.stat().st_size if part.is_file() else 0
            if have == f.size_bytes and _sha256(part) == f.sha256:
                continue
            # Oversized or corrupt completed partials will be discarded, not resumed.
            remaining += f.size_bytes - have if 0 < have < f.size_bytes else f.size_bytes
        return remaining
