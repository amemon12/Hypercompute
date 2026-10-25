"""hcpeer on a macOS peer: the opt-in, ggml-rpc-server's environment and its banner line.

hcpeer is a patched libggml-rpc (native/hcpeer, docs/research/hcpeer-macos.md). With HC_PEER_MMAP=1 and `-c` the
worker's CPU device maps the tensor cache files it holds instead of copying them into memory. The contributor opts in
by joining with HC_PEER_MMAP=1 in the agent's environment, on an engine selected with HYPERCOMPUTE_LLAMA_BIN_DIR
(scripts/capacity/hcpeer_mac_install.sh makes one without touching the stock engine).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..units import GiB

MMAP_ENV = "HC_PEER_MMAP"
CACHE_MAX_ENV = "HC_PEER_CACHE_MAX_GIB"
MIN_FREE_ENV = "HC_PEER_CACHE_MIN_FREE_GIB"  # never set by the agent: hcpeer's default is seed_cache.MIN_FREE_BYTES
CAP_MIN_VERSION = 4  # v3 ignores HC_PEER_CACHE_MAX_GIB
INSTALL_HINT = ("bash scripts/capacity/hcpeer_mac_install.sh, then join with HYPERCOMPUTE_LLAMA_BIN_DIR set to the "
                "path it prints")
# `  hcpeer mmap    : on (v4, page 16384 B, ...)` or `: off (v4, set HC_PEER_MMAP=1 to enable)`, before `Devices:`
_BANNER_RE = re.compile(r"^\s*hcpeer mmap\s*:\s*(?P<state>on|off)\s*\(v(?P<version>\d{1,4})\b", re.MULTILINE)


@dataclass(frozen=True)
class HcpeerBanner:
    on: bool
    version: int


def requested(system: str, environ: Mapping[str, str]) -> bool:
    """The contributor asked for hcpeer: HC_PEER_MMAP=1 (the value hcpeer itself accepts) on a Mac."""
    return system == "macos" and environ.get(MMAP_ENV) == "1"


def inherited_limits(environ: Mapping[str, str]) -> list[str]:
    """hcpeer's limit variables already in the agent's environment, which ggml-rpc-server inherits. Either would change
    the cap or floor hcpeer applies at save time while the agent's disk guard and the host's plan hold --max-disk in
    whole GiB (passed only once a v4 banner shows) and the 20 GiB floor, so a join with hcpeer refuses them."""
    return [name for name in (CACHE_MAX_ENV, MIN_FREE_ENV) if name in environ]


def parse_banner(text: str) -> HcpeerBanner | None:
    """The hcpeer line ggml-rpc-server prints at start; None for a stock library, which prints none."""
    m = _BANNER_RE.search(text)
    return HcpeerBanner(on=m["state"] == "on", version=int(m["version"])) if m else None


def cache_cap_gib(max_disk_bytes: int) -> int | None:
    """--max-disk in whole GiB, rounded down so hcpeer stops saving early rather than late; None without a limit."""
    return max_disk_bytes // GiB if max_disk_bytes > 0 else None


def consented_max_disk(max_disk_bytes: int) -> int:
    """--max-disk as the whole GiB hcpeer caps the cache at, so the agent's disk guard, its seed downloads and the
    host's plan (the report's max_disk_bytes) all hold the cap the worker enforces. 0 (no downloads) stays 0. Raises
    ValueError under 1 GiB: a cap of 0 GiB stops every save, so the CPU device would copy every weight into RAM."""
    if max_disk_bytes <= 0:
        return 0
    if max_disk_bytes < GiB:
        raise ValueError(f"with {MMAP_ENV}=1 --max-disk must be at least 1 GiB: hcpeer caps the cache in whole GiB "
                         "and a cap of 0 saves nothing")
    return max_disk_bytes // GiB * GiB


def worker_env(cache_root: Path | None, *, on: bool, cap_gib: int | None,
               banner: HcpeerBanner | None) -> dict[str, str]:
    """What ggml-rpc-server gets on top of the agent's environment. `cache_root` is None without -c; hcpeer needs -c.
    The cap is passed only once a banner showed a version that honours it."""
    if cache_root is None:
        return {}
    env = {"LLAMA_CACHE": str(cache_root)}
    if not on:
        return env
    env[MMAP_ENV] = "1"
    if cap_gib is not None and banner is not None and banner.version >= CAP_MIN_VERSION:
        env[CACHE_MAX_ENV] = str(cap_gib)
    return env


def consent_line(cache_dir: Path, cap_gib: int | None) -> str:
    """Shown before the contributor confirms."""
    cap = (f"the worker keeps those files under {cap_gib} GiB (--max-disk) once its banner shows hcpeer "
           f"v{CAP_MIN_VERSION} or later" if cap_gib is not None else
           "no size cap without --managed --max-disk")
    return (f"hcpeer ({MMAP_ENV}=1): the worker's CPU device maps cached tensor files from disk ({cache_dir}) "
            "instead of copying them into memory, so macOS can drop those pages and read them again from this disk. "
            f"The tensor cache is on for this; {cap}, and hcpeer's own floor (20 GiB free by default) holds.")


def status_line(banner: HcpeerBanner | None, cap_gib: int | None, env: Mapping[str, str]) -> str:
    """What the running worker reported, after it started with hcpeer requested."""
    if banner is None:
        return (f"{MMAP_ENV}=1, but this engine has no hcpeer: the CPU device copies weights into memory. "
                f"Install it: {INSTALL_HINT}.")
    if not banner.on:
        return f"hcpeer v{banner.version} is off in the worker: the CPU device copies weights into memory."
    if cap_gib is None:
        cap = ", no cache size cap"
    elif CACHE_MAX_ENV in env:
        cap = f", cache capped at {env[CACHE_MAX_ENV]} GiB"
    else:
        cap = f"; this version ignores the cache cap (v{CAP_MIN_VERSION} or later honours it)"
    return f"hcpeer v{banner.version} on: CPU weights are mapped from the cache{cap}."
