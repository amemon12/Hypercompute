"""Settings read once from the environment (and a git-ignored .env). Every value is validated here, at the boundary."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
ADDRESS_RE = re.compile(r"^(agent|test-agent)1[0-9a-z]{20,90}$")
ACCESS_MODES = ("none", "read", "paid")
PAYMENT_MODES = ("off", "guests", "everyone")
CONFIRM_MODES = ("auto", "always", "never")
STRATEGIES = ("all_selected", "auto", "fastest_fit", "local_only")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
RESERVED_CHAT_KEYS = frozenset({"messages", "max_tokens", "stream", "model"})


class ConfigError(ValueError):
    """A setting is missing or malformed. The message names the variable and how to fix it."""


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _choice(env: Mapping[str, str], name: str, default: str, choices: tuple[str, ...]) -> str:
    value = env.get(name, "").strip().lower() or default
    if value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _json_object(env: Mapping[str, str], name: str) -> dict[str, object]:
    raw = env.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a JSON object, got invalid JSON") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object")
    reserved = RESERVED_CHAT_KEYS & set(value)
    if reserved:
        raise ConfigError(f"{name} may not set {', '.join(sorted(reserved))}")
    return value


def parse_addresses(raw: str, name: str = "HC_ALLOWED_SENDERS") -> frozenset[str]:
    """Comma or whitespace separated agent addresses. A malformed entry is an error, never silently dropped."""
    items = [item for item in re.split(r"[\s,]+", raw.strip()) if item]
    bad = [item for item in items if not ADDRESS_RE.match(item)]
    if bad:
        raise ConfigError(f"{name} has entries that are not agent addresses: {', '.join(bad[:3])}")
    return frozenset(items)


def check_gateway_url(url: str) -> str:
    """The gateway's owner endpoints answer only on loopback, so anything else is a configuration mistake."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError(f"HC_GATEWAY_URL must be an http URL, got {url!r}")
    host = parsed.hostname.lower()
    if host not in LOOPBACK_HOSTS and f"[{host}]" not in LOOPBACK_HOSTS:
        raise ConfigError("HC_GATEWAY_URL must point at this machine (http://127.0.0.1:<port>): the gateway serves "
                          "owner endpoints on loopback only")
    return url.rstrip("/")


@dataclass(frozen=True)
class Pricing:
    """Per-request price in FET: base + rate * (max_tokens / 1000) * (params_b / 100). Credits split by layers."""

    currency: str = "FET"
    base_fee: float = 0.01
    rate_per_1k_tokens_per_100b: float = 0.004
    min_fee: float = 0.01
    max_fee: float = 5.0
    host_fee_share: float = 0.10  # the coordinating host keeps this share before the layer split


@dataclass(frozen=True)
class Settings:
    seed: str = field(repr=False)
    gateway_url: str = "http://127.0.0.1:8000"
    agent_name: str = "hypercompute"
    agent_port: int = 8460
    agent_handle: str | None = None
    allowed_senders: frozenset[str] = frozenset()
    guest_access: str = "read"  # none | read | paid
    payments: str = "guests"  # off | guests | everyone
    confirm: str = "auto"  # auto | always | never
    strategy: str = "all_selected"
    max_prompt_chars: int = 4000
    default_max_tokens: int = 256
    max_tokens_cap: int = 1024
    runs_per_window: int = 4
    rate_window_s: int = 600
    messages_per_minute: int = 20
    load_timeout_s: int = 2400
    chat_timeout_s: int = 900
    keep_warm_s: int = 900
    progress_every_s: int = 45
    plan_ttl_s: int = 480  # the gateway deploys only plans under 10 minutes old
    reuse_foreign_ready: bool = True  # chat on a ready deployment the agent did not start (never stop it)
    asi1_api_key: str | None = field(default=None, repr=False)
    asi1_model: str = "asi1"
    asi1_url: str = "https://api.asi1.ai/v1/chat/completions"
    agentverse_api_key: str | None = field(default=None, repr=False)
    fet_network: str = "stable-testnet"
    payment_verify: str = "ledger"  # ledger | accept-test (local rehearsal only, never with a mailbox)
    data_dir: str = "agent_data"
    mailbox: bool = True
    local_peers: Mapping[str, str] = field(default_factory=dict)  # address -> http endpoint, local testing only
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    chat_extra: Mapping[str, object] = field(default_factory=dict)  # extra engine options for every completion
    pricing: Pricing = Pricing()

    @property
    def payments_enabled(self) -> bool:
        return self.payments != "off" or self.guest_access == "paid"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        seed = env.get("HC_AGENT_SEED", "").strip()
        if len(seed) < 32:
            raise ConfigError("HC_AGENT_SEED must be set to a secret of at least 32 characters (it fixes the agent "
                              "address); see .env.example")
        local_peers: dict[str, str] = {}
        for pair in [p for p in re.split(r"[\s,]+", env.get("HC_LOCAL_PEERS", "").strip()) if p]:
            address, _, endpoint = pair.partition("=")
            if not ADDRESS_RE.match(address) or not endpoint.startswith("http://127.0.0.1:"):
                raise ConfigError("HC_LOCAL_PEERS entries look like agent1q...=http://127.0.0.1:<port>/submit")
            local_peers[address] = endpoint
        chat_extra = _json_object(env, "HC_CHAT_EXTRA_JSON")
        verify = _choice(env, "HC_PAYMENT_VERIFY", "ledger", ("ledger", "accept-test"))
        if verify == "accept-test" and _bool(env, "HC_MAILBOX", True):
            raise ConfigError("HC_PAYMENT_VERIFY=accept-test is for local rehearsal only; set HC_MAILBOX=false too")
        handle = env.get("HC_AGENT_HANDLE", "").strip() or None
        if handle is not None and not re.match(r"^[a-z0-9-]{3,40}$", handle):
            raise ConfigError("HC_AGENT_HANDLE must be 3 to 40 lowercase letters, digits or dashes")
        pricing = Pricing(
            base_fee=_float(env, "HC_PRICE_BASE_FET", 0.01, 0.0, 10.0),
            rate_per_1k_tokens_per_100b=_float(env, "HC_PRICE_RATE_FET", 0.004, 0.0, 10.0),
            min_fee=_float(env, "HC_PRICE_MIN_FET", 0.01, 0.0, 10.0),
            max_fee=_float(env, "HC_PRICE_MAX_FET", 5.0, 0.0, 100.0),
            host_fee_share=_float(env, "HC_HOST_FEE_SHARE", 0.10, 0.0, 0.5),
        )
        return cls(
            seed=seed,
            gateway_url=check_gateway_url(env.get("HC_GATEWAY_URL", "").strip() or "http://127.0.0.1:8000"),
            agent_name=env.get("HC_AGENT_NAME", "").strip() or "hypercompute",
            agent_port=_int(env, "HC_AGENT_PORT", 8460, 1024, 65535),
            agent_handle=handle,
            allowed_senders=parse_addresses(env.get("HC_ALLOWED_SENDERS", "")),
            guest_access=_choice(env, "HC_GUEST_ACCESS", "read", ACCESS_MODES),
            payments=_choice(env, "HC_PAYMENTS", "guests", PAYMENT_MODES),
            confirm=_choice(env, "HC_CONFIRM", "auto", CONFIRM_MODES),
            strategy=_choice(env, "HC_STRATEGY", "all_selected", STRATEGIES),
            max_prompt_chars=_int(env, "HC_MAX_PROMPT_CHARS", 4000, 100, 32000),
            default_max_tokens=_int(env, "HC_DEFAULT_MAX_TOKENS", 256, 16, 4096),
            max_tokens_cap=_int(env, "HC_MAX_TOKENS_CAP", 1024, 16, 8192),
            runs_per_window=_int(env, "HC_RUNS_PER_WINDOW", 4, 1, 1000),
            rate_window_s=_int(env, "HC_RATE_WINDOW_S", 600, 10, 86400),
            messages_per_minute=_int(env, "HC_MESSAGES_PER_MINUTE", 20, 1, 1000),
            load_timeout_s=_int(env, "HC_LOAD_TIMEOUT_S", 2400, 30, 14400),
            chat_timeout_s=_int(env, "HC_CHAT_TIMEOUT_S", 900, 10, 7200),
            keep_warm_s=_int(env, "HC_KEEP_WARM_S", 900, 0, 86400),
            progress_every_s=_int(env, "HC_PROGRESS_EVERY_S", 45, 5, 3600),
            plan_ttl_s=_int(env, "HC_PLAN_TTL_S", 480, 30, 590),
            reuse_foreign_ready=_bool(env, "HC_REUSE_FOREIGN_READY", True),
            asi1_api_key=(env.get("ASI_ONE_API_KEY", "") or env.get("ASI1_API_KEY", "")).strip() or None,
            asi1_model=env.get("HC_ASI1_MODEL", "").strip() or "asi1",
            agentverse_api_key=env.get("AGENTVERSE_API_KEY", "").strip() or None,
            fet_network=_choice(env, "HC_FET_NETWORK", "stable-testnet", ("stable-testnet", "mainnet")),
            payment_verify=verify,
            data_dir=env.get("HC_DATA_DIR", "").strip() or "agent_data",
            mailbox=_bool(env, "HC_MAILBOX", True),
            local_peers=local_peers,
            system_prompt=env.get("HC_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip(),
            chat_extra=chat_extra,
            pricing=pricing,
        )
