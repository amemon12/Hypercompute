"""Bridge settings from the environment (or a git-ignored env file). Nothing runs unless all three of
HC_CONVEX_SITE_URL, HC_ROOM_SECRET and HC_ROOM_CODE are set."""

from __future__ import annotations

import codecs
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOM_CODE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
REQUIRED = ("HC_CONVEX_SITE_URL", "HC_ROOM_SECRET", "HC_ROOM_CODE")
DEFAULT_SYSTEM = ("You are answering live on a projector at a hackathon. Reply in one or two short, plain "
                  "sentences.")


class ConfigError(ValueError):
    """A setting is missing or unsafe. Messages name the variable, never its value."""


@dataclass(frozen=True)
class BridgeConfig:
    site_url: str
    room_code: str
    secret: str = field(repr=False)
    gateway_url: str = "http://127.0.0.1:8000"
    title: str = "Hypercompute Pool Room"
    max_tokens: int = 48
    slots: int | None = None  # None: the deployment's own slot count (engine_options.parallel)
    system_prompt: str = DEFAULT_SYSTEM
    poll_s: float = 3.0  # how often the pool snapshot is read from the gateway
    pool_heartbeat_s: float = 10.0  # resend an unchanged snapshot this often, so the room shows the bridge online
    flush_s: float = 0.25  # token deltas are batched into one append per interval
    keepalive_s: float = 20.0  # an empty append during a long prefill keeps the 90 s lease alive
    idle_claim_s: float = 1.0
    read_timeout_s: float = 180.0  # silence from the gateway longer than this counts as a lost gateway

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BridgeConfig | None:
        """None when the room is not configured at all; ConfigError when it is configured wrongly."""
        values = {k: (env.get(k) or "").strip() for k in REQUIRED}
        if not all(values.values()):
            missing = [k for k, v in values.items() if not v]
            if len(missing) == len(REQUIRED):
                return None
            raise ConfigError(f"set all of {', '.join(REQUIRED)}; missing {', '.join(missing)}")
        site = _site_url(values["HC_CONVEX_SITE_URL"])
        secret = values["HC_ROOM_SECRET"]
        if len(secret) < 24:
            raise ConfigError("HC_ROOM_SECRET must be at least 24 characters (use a random 32-byte value)")
        code = values["HC_ROOM_CODE"]
        if not ROOM_CODE.match(code):
            raise ConfigError("HC_ROOM_CODE must be 3-32 chars of a-z, 0-9 and -, starting with a letter or digit")
        return cls(
            site_url=site,
            room_code=code,
            secret=secret,
            gateway_url=_gateway_url(env.get("HC_GATEWAY_URL") or "http://127.0.0.1:8000"),
            title=(env.get("HC_ROOM_TITLE") or cls.title).strip()[:64] or cls.title,
            max_tokens=_int(env, "HC_ROOM_MAX_TOKENS", 48, 1, 512),
            slots=_int(env, "HC_ROOM_SLOTS", 0, 0, 16) or None,
            system_prompt=(env.get("HC_ROOM_SYSTEM") or DEFAULT_SYSTEM).strip()[:500],
        )


def _site_url(raw: str) -> str:
    parts = urlsplit(raw.rstrip("/"))
    local = parts.hostname in LOOPBACK_HOSTS
    if parts.scheme != "https" and not (local and parts.scheme == "http"):
        raise ConfigError("HC_CONVEX_SITE_URL must be https://NAME.convex.site (http only for a local backend)")
    if not local and not (parts.hostname or "").endswith(".convex.site"):
        raise ConfigError("HC_CONVEX_SITE_URL must be the HTTP actions URL, https://NAME.convex.site")
    if parts.path or parts.query:
        raise ConfigError("HC_CONVEX_SITE_URL must be the bare deployment URL, with no path")
    return f"{parts.scheme}://{parts.netloc}"


def _gateway_url(raw: str) -> str:
    parts = urlsplit(raw.strip().rstrip("/"))
    if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS:
        raise ConfigError("HC_GATEWAY_URL must be http://127.0.0.1:<port>: the owner API only answers loopback")
    return f"http://{parts.netloc}"


def _int(env: Mapping[str, str], key: str, default: int, lo: int, hi: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a whole number") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{key} must be between {lo} and {hi}")
    return value


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; blank lines and # comments skipped; optional quotes stripped. CRLF tolerated."""
    out: dict[str, str] = {}
    raw = path.read_bytes()
    # PowerShell 5.1 writes UTF-16 or a UTF-8 BOM by default; accept those instead of failing on a decode error.
    encoding = "utf-16" if raw[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE) else "utf-8-sig"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path.name} is not UTF-8 or UTF-16 text") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out
