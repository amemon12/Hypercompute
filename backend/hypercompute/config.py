"""Paths, ports and timing constants. Everything here is overridable by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GATEWAY_PORT = 8000
# 50052 is llama.cpp's conventional RPC port, but Windows hosts with Hyper-V/WSL/Docker
# commonly reserve 50000-50059 (see `netsh int ipv4 show excludedportrange protocol=tcp`).
# 47052 sits below the Windows dynamic range (49152+) that those reservations come from.
DEFAULT_WORKER_PORT = 47052
WORKER_PORT_SCAN = 24

HEARTBEAT_INTERVAL_S = 3.0
STALE_AFTER_S = 15.0
HOST_UNREACHABLE_STOP_S = 5 * 60  # a friend's worker stops sharing once the host has been gone this long
INVITE_TTL_S = 15 * 60
INVITE_MAX_FAILED_ATTEMPTS = 5
JOIN_RATE_LIMIT_PER_MIN = 10
GATEWAY_MAX_CONNECTIONS = 100  # the gateway listens on the LAN: past this uvicorn answers 503

PROBE_TIMEOUT_S = 25.0
LOAD_TIMEOUT_S = 30 * 60
WARMUP_TIMEOUT_S = 120.0
UPSTREAM_IDLE_TIMEOUT_S = 180.0
STOP_GRACE_S = 8.0

OWNER_HEADER = "X-Hypercompute-Owner"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def engines(self) -> Path:
        return self.home / "engines"

    @property
    def downloads(self) -> Path:
        return self.home / "downloads"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def coordinator(self) -> Path:
        return self.home / "coordinator"

    @property
    def agent(self) -> Path:
        return self.home / "agent"

    @property
    def rpc_cache(self) -> Path:
        """LLAMA_CACHE root handed to ggml-rpc-server so its tensor cache stays Hypercompute-managed."""
        return self.home / "rpc-cache"

    def ensure(self) -> Paths:
        for p in (self.home, self.engines, self.downloads, self.models, self.coordinator, self.agent):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    return Paths(home=_env_path("HYPERCOMPUTE_HOME", Path.home() / ".hypercompute"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    """Checkout `configs/` when running from source; the copy bundled into the wheel when installed."""
    checkout = repo_root() / "configs"
    bundled = Path(__file__).resolve().parent / "_configs"
    return _env_path("HYPERCOMPUTE_CONFIG_DIR", checkout if checkout.is_dir() else bundled)


def frontend_dist() -> Path:
    return _env_path("HYPERCOMPUTE_FRONTEND_DIST", repo_root() / "frontend" / "dist")
