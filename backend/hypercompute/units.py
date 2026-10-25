"""Byte-size parsing and formatting. Internally everything is bytes; units are explicit at the edges."""

from __future__ import annotations

import re

MiB = 1024**2
GiB = 1024**3

_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": MiB,
    "gib": GiB,
    "tib": 1024**4,
}
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")


def parse_size(text: str) -> int:
    """Parse '8GiB', '512 MiB', '2.5gb' or a plain byte count. Bare 'G'/'M' are treated as binary units."""
    m = _SIZE_RE.match(text)
    if not m:
        raise ValueError(f"not a size: {text!r}")
    number, unit = float(m.group(1)), m.group(2).lower()
    unit = {"g": "gib", "m": "mib", "k": "kib", "t": "tib", "": "b"}.get(unit, unit)
    if unit not in _UNITS:
        raise ValueError(f"unknown size unit in {text!r}")
    value = int(number * _UNITS[unit])
    if value < 0:
        raise ValueError("size must be non-negative")
    return value


def fmt_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    if value >= GiB:
        return f"{value / GiB:.1f} GiB"
    if value >= MiB:
        return f"{value / MiB:.0f} MiB"
    return f"{value} B"


def mib(value: float) -> int:
    return int(round(value * MiB))
