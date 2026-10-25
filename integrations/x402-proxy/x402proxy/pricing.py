"""Price of one paid chat, in USDC atomic units (6 decimals). Integer math only, so every amount is exact.

A request reserves max_tokens completion tokens and pays base + per_1k * max_tokens / 1000, rounded up to the next
atomic unit. The x402 "exact" scheme fixes the amount when the buyer signs, so the buyer pays for what it reserves,
not for what the model happens to produce; the README explains why and what "upto" would change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

USDC_DECIMALS = 6
ATOMIC_PER_USDC = 10**USDC_DECIMALS
_MONEY = re.compile(r"^\$?(\d+(\.\d+)?)$")


class PriceError(ValueError):
    """A price setting or a request's max_tokens that cannot be priced."""


@dataclass(frozen=True)
class Pricing:
    base_atomic: int  # flat part of every request
    per_1k_atomic: int  # per 1,000 reserved completion tokens
    default_max_tokens: int  # used when the request names none
    max_tokens_cap: int  # a request asking for more is clamped to this

    def __post_init__(self) -> None:
        if self.base_atomic < 0 or self.per_1k_atomic < 0:
            raise PriceError("prices must not be negative")
        if self.base_atomic + self.per_1k_atomic == 0:
            raise PriceError("a paywall with a zero price is not a paywall")
        if not 1 <= self.default_max_tokens <= self.max_tokens_cap:
            raise PriceError("need 1 <= default_max_tokens <= max_tokens_cap")


def usd_to_atomic(text: str) -> int:
    """'$0.01' or '0.01' to atomic USDC units. More than 6 decimals is refused instead of silently rounded."""
    match = _MONEY.match(text.strip())
    if not match:
        raise PriceError(f"not a dollar amount: {text!r}")
    try:
        value = Decimal(match.group(1)) * ATOMIC_PER_USDC
    except InvalidOperation as exc:
        raise PriceError(f"not a dollar amount: {text!r}") from exc
    if value != value.to_integral_value():
        raise PriceError(f"{text!r} has more than {USDC_DECIMALS} decimals")
    return int(value)


def atomic_to_usd(amount: int) -> str:
    """Atomic units to a fixed 6-decimal string, e.g. 6120 -> '0.006120'."""
    if amount < 0:
        raise PriceError("amount must not be negative")
    return f"{amount // ATOMIC_PER_USDC}.{amount % ATOMIC_PER_USDC:0{USDC_DECIMALS}d}"


def reserved_tokens(requested: object, pricing: Pricing) -> int:
    """The max_tokens a request reserves: its own value clamped to the cap, or the default when it names none."""
    if requested is None:
        return pricing.default_max_tokens
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise PriceError("'max_tokens' must be a positive integer")
    return min(requested, pricing.max_tokens_cap)


def quote_atomic(max_tokens: int, pricing: Pricing) -> int:
    """base + ceil(per_1k * max_tokens / 1000)."""
    if max_tokens < 1:
        raise PriceError("max_tokens must be positive")
    return pricing.base_atomic + -(-pricing.per_1k_atomic * max_tokens // 1000)
