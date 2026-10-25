"""Proxy settings from the environment. Everything is validated here, once, at startup.

Testnet only: the network must be Base Sepolia. There is no switch for mainnet on purpose.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .pricing import Pricing, usd_to_atomic

ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")
BASE_SEPOLIA = "eip155:84532"
TESTNETS = frozenset({BASE_SEPOLIA})
PUBLIC_FACILITATOR_URL = "https://x402.org/facilitator"  # x402 Foundation's free testnet facilitator, no key
DEFAULT_GATEWAY = "http://127.0.0.1:8001"  # the demo host's gateway; override with HYPERCOMPUTE_GATEWAY
EXPLORERS = MappingProxyType({BASE_SEPOLIA: "https://sepolia.basescan.org/tx/"})
HERE = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """A setting that would make the proxy unsafe or wrong; the message names the variable."""


@dataclass(frozen=True)
class ProxyConfig:
    gateway_url: str  # the Hypercompute gateway on this machine
    pay_to: str  # public address that receives every payment
    network: str
    facilitator: str  # "cdp" (Coinbase's hosted facilitator, needs the CDP key) or a facilitator URL
    pricing: Pricing
    host_fee_bps: int
    payees: Mapping[str, str] = field(default_factory=dict)  # display name or node id -> public payout address
    ledger_path: Path = HERE / "data" / "ledger.sqlite3"
    max_inflight: int = 1  # paid chats served at once; more get 503 before anyone is charged
    bind_host: str = "127.0.0.1"
    port: int = 8402
    rate_per_min: int = 60  # requests per client address per minute on the chat and quote routes
    max_prompt_chars: int = 32_000  # the price covers max_tokens, so the prompt (prefill) is capped instead

    @property
    def explorer_tx_url(self) -> str:
        return EXPLORERS.get(self.network, "")


def parse_payees(text: str) -> Mapping[str, str]:
    """'mac-a=0xabc..,g14=0xdef..' to a read-only mapping of checked public addresses."""
    out: dict[str, str] = {}
    for pair in (p.strip() for p in text.split(",")):
        if not pair:
            continue
        name, sep, address = (part.strip() for part in pair.partition("="))
        if not sep or not name or not ADDRESS.match(address):
            raise ConfigError(f"X402_PAYEES entry {pair!r} must be name=0x followed by 40 hex characters")
        out[name] = address
    return MappingProxyType(out)


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name, "").strip() or str(default)
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _facilitator(env: Mapping[str, str]) -> str:
    choice = env.get("X402_FACILITATOR", "").strip() or "auto"
    has_cdp = bool(env.get("CDP_API_KEY_ID", "").strip() and env.get("CDP_API_KEY_SECRET", "").strip())
    if choice == "auto":
        return "cdp" if has_cdp else PUBLIC_FACILITATOR_URL
    if choice == "cdp" and not has_cdp:
        raise ConfigError("X402_FACILITATOR=cdp needs CDP_API_KEY_ID and CDP_API_KEY_SECRET")
    if choice != "cdp" and not choice.startswith("https://"):
        raise ConfigError("X402_FACILITATOR must be auto, cdp or an https:// URL")
    return choice


def load(env: Mapping[str, str] | None = None) -> ProxyConfig:
    env = os.environ if env is None else env
    pay_to = env.get("X402_PAY_TO", "").strip()
    if not ADDRESS.match(pay_to):
        raise ConfigError("X402_PAY_TO must be your public Base Sepolia address (0x followed by 40 hex characters)")
    network = env.get("X402_NETWORK", "").strip() or BASE_SEPOLIA
    if network not in TESTNETS:
        raise ConfigError(f"X402_NETWORK {network!r} is not a supported testnet; this proxy is testnet only")
    pricing = Pricing(base_atomic=usd_to_atomic(env.get("X402_PRICE_BASE", "$0.001")),
                      per_1k_atomic=usd_to_atomic(env.get("X402_PRICE_PER_1K", "$0.02")),
                      default_max_tokens=_int(env, "X402_DEFAULT_MAX_TOKENS", 256, 1, 8192),
                      max_tokens_cap=_int(env, "X402_MAX_TOKENS_CAP", 1024, 1, 8192))
    ledger = env.get("X402_LEDGER_PATH", "").strip()
    return ProxyConfig(gateway_url=(env.get("HYPERCOMPUTE_GATEWAY", "").strip() or DEFAULT_GATEWAY).rstrip("/"),
                       pay_to=pay_to, network=network, facilitator=_facilitator(env), pricing=pricing,
                       host_fee_bps=_int(env, "X402_HOST_FEE_BPS", 1000, 0, 10_000),
                       payees=parse_payees(env.get("X402_PAYEES", "")),
                       ledger_path=Path(ledger) if ledger else HERE / "data" / "ledger.sqlite3",
                       max_inflight=_int(env, "X402_MAX_INFLIGHT", 1, 1, 64),
                       bind_host=env.get("X402_BIND", "").strip() or "127.0.0.1",
                       port=_int(env, "X402_PORT", 8402, 1, 65535),
                       rate_per_min=_int(env, "X402_RATE_PER_MIN", 60, 1, 100_000),
                       max_prompt_chars=_int(env, "X402_MAX_PROMPT_CHARS", 32_000, 1, 4_000_000))


def safe_console() -> None:
    """cdp-sdk prints an emoji notice the first time cdp.x402 is imported; on a Windows console or a redirected
    stdout that is not UTF-8 that print raises UnicodeEncodeError and kills the process. Replace instead."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines; blank lines and # comments skipped. Values are never logged."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out
