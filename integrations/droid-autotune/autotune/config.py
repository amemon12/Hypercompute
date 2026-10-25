"""Settings from the environment, with KEY=VALUE lines from this folder's git-ignored .env filling in what the
environment does not set. The Factory key is read here and handed to the SDK; it is never logged or shown."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        out[key] = value.strip().strip('"').strip("'")
    return out


def _num(env: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    value = float(raw)
    if not lo <= value <= hi:
        raise ValueError(f"{key}={raw} is outside {lo}..{hi}")
    return value


@dataclass(frozen=True)
class Settings:
    gateway: str = "http://127.0.0.1:8000"
    port: int = 8787
    settle_s: float = 20.0
    poll_s: float = 2.0
    max_checks: int = 8
    max_calls: int = 12
    timeout_s: float = 300.0
    ctx_size: int = 4096
    min_interval_s: float = 120.0  # between automatic runs; the owner's Run now is not limited
    max_runs_per_hour: int = 10  # automatic runs; each one is a paid Droid session
    runner: str = "droid"  # "droid" (Factory Droid SDK) or "scripted" (tests and the labelled offline demo only)
    droid_mode: str = "spec"  # "spec": Droid writes a plan first (published to the log), "auto": it starts at once
    # A model that calls MCP tools directly, so each call reaches ToolGuard.on_permission. droid's default
    # (gpt-6.1-sol) calls them from inside its Script code tool, which the lock disables (probed live 2026-10-24).
    droid_model: str | None = "claude-sonnet-5-5"
    droid_path: str | None = None
    fallback_model: str | None = None  # plan-only runs when no deployment is ready
    runs_dir: Path = ROOT / "runs"
    factory_api_key: str | None = field(default=None, repr=False)  # never printed

    @property
    def droid_ready(self) -> tuple[bool, str]:
        if not self.factory_api_key:
            return False, f"FACTORY_API_KEY is not set (put it in {ROOT / '.env'})"
        if not self.droid_path:
            return False, "the droid CLI is not on PATH (install it, then run `droid --version`)"
        return True, f"droid at {self.droid_path}"


def load(env: Mapping[str, str] | None = None, env_file: Path | None = None) -> Settings:
    merged = {**read_env_file(env_file or ROOT / ".env"), **(env if env is not None else os.environ)}
    runner = merged.get("AUTOTUNE_RUNNER", "droid")
    if runner not in ("droid", "scripted"):
        raise ValueError("AUTOTUNE_RUNNER is droid or scripted")
    mode = merged.get("AUTOTUNE_DROID_MODE", "spec")
    if mode not in ("spec", "auto"):
        raise ValueError("AUTOTUNE_DROID_MODE is spec or auto")
    max_checks = int(_num(merged, "AUTOTUNE_MAX_CHECKS", 8, 1, 32))
    return Settings(
        gateway=merged.get("AUTOTUNE_GATEWAY", "http://127.0.0.1:8000"),
        port=int(_num(merged, "AUTOTUNE_PORT", 8787, 1024, 65535)),
        settle_s=_num(merged, "AUTOTUNE_SETTLE_S", 20, 0, 600),
        poll_s=_num(merged, "AUTOTUNE_POLL_S", 2, 0.2, 60),
        max_checks=max_checks,
        max_calls=int(_num(merged, "AUTOTUNE_MAX_CALLS", max(12, max_checks + 4), max_checks, 64)),
        timeout_s=_num(merged, "AUTOTUNE_TIMEOUT_S", 300, 30, 1800),
        ctx_size=int(_num(merged, "AUTOTUNE_CTX", 4096, 256, 262144)),
        min_interval_s=_num(merged, "AUTOTUNE_MIN_INTERVAL_S", 120, 0, 3600),
        max_runs_per_hour=int(_num(merged, "AUTOTUNE_MAX_RUNS_PER_HOUR", 10, 1, 120)),
        runner=runner, droid_mode=mode,
        droid_model=merged.get("AUTOTUNE_DROID_MODEL") or "claude-sonnet-5-5",
        droid_path=merged.get("DROID_PATH") or shutil.which("droid"),
        fallback_model=merged.get("AUTOTUNE_MODEL") or None,
        runs_dir=Path(merged["AUTOTUNE_RUNS_DIR"]) if merged.get("AUTOTUNE_RUNS_DIR") else ROOT / "runs",
        factory_api_key=merged.get("FACTORY_API_KEY") or None,
    )
