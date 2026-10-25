"""Invite parsing, log redaction and command display for the one-click scripts (PACKAGING-SPEC 2.4 step 4, 2.6)."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # services/sessions.py: no 0, O, 1 or I
CODE_LEN = 10
MAX_INVITE_BYTES = 8 * 1024
TICKET_RE = re.compile(r"[A-Za-z0-9]{30,4096}")  # the 30 character floor keeps LONG_TOKEN_RE masking it
LONG_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{30,}")
CODE_SHAPE_RE = re.compile(r"\b[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}\b")
TICKET_TOKEN, CODE_TOKEN = "<ticket>", "<code>"


class InviteError(ValueError):
    """The pasted text is not an Iroh invite. `lan` marks a LAN invite, which gets no retry."""

    def __init__(self, reason: str, lan: bool = False) -> None:
        super().__init__(reason)
        self.lan = lan


@dataclass(frozen=True)
class Invite:
    ticket: str
    code: str  # XXXXX-XXXXX
    pasted_code: str

    @property
    def tail(self) -> str:
        return self.ticket[-4:]

    def code_forms(self) -> tuple[str, ...]:
        return tuple(sorted({self.pasted_code, self.code, self.code.replace("-", "")}))


def normalize_code(raw: str) -> str | None:
    """Uppercase, keep letters and digits, require 10 characters of the code alphabet, format XXXXX-XXXXX."""
    s = "".join(ch for ch in raw.upper() if ch.isalnum())
    if len(s) != CODE_LEN or any(ch not in CODE_ALPHABET for ch in s):
        return None
    return f"{s[:5]}-{s[5:]}"


def token_after(tokens: Sequence[str], flag: str) -> str | None:
    if flag not in tokens:
        return None
    i = list(tokens).index(flag)
    return tokens[i + 1] if i + 1 < len(tokens) else None


def parse_invite(text: str, ticket_check: Callable[[str], object] | None = None) -> Invite:
    """The full `uv run hypercompute join --ticket T --code C` line, or the two bare tokens in either order."""
    if len(text.encode("utf-8", "replace")) > MAX_INVITE_BYTES:
        raise InviteError("longer than 8 KiB")
    if "--host" in text:
        raise InviteError("LAN invite", lan=True)
    tokens = [t.strip("'\"`") for t in text.split()]
    if "--ticket" in tokens:
        ticket, pasted = token_after(tokens, "--ticket"), token_after(tokens, "--code")
    elif len(tokens) == 2:
        codes = [t for t in tokens if normalize_code(t)]
        if len(codes) != 1:
            raise InviteError("two tokens but not one code")
        pasted = codes[0]
        ticket = tokens[1] if tokens[0] == pasted else tokens[0]
    else:
        raise InviteError("not an invite line")
    code = normalize_code(pasted or "")
    if not ticket or not TICKET_RE.fullmatch(ticket):
        raise InviteError("bad ticket")
    if code is None or pasted is None:
        raise InviteError("bad code")
    if ticket_check is not None:
        try:
            ticket_check(ticket)
        except Exception as exc:  # the binding raises its own error types for a malformed ticket
            raise InviteError("ticket did not parse") from exc
    return Invite(ticket=ticket, code=code, pasted_code=pasted)


def iroh_ticket_check(ticket: str) -> None:
    """Local parse of the ticket when the Iroh binding is installed. It opens no connection."""
    try:
        import iroh
    except ImportError:
        return
    iroh.EndpointTicket.from_string(ticket)


def make_redactor(ticket: str | None = None, codes: Iterable[str | None] = ()) -> Callable[[str], str]:
    """Spec 2.6, in order: the exact ticket and code forms, then any 30+ character token, then the code shape."""
    exact = [(ticket, TICKET_TOKEN)] if ticket else []
    exact += [(c, CODE_TOKEN) for c in sorted({c for c in codes if c}, key=len, reverse=True)]

    def redact(line: str) -> str:
        for secret, token in exact:
            line = line.replace(secret, token)
        line = LONG_TOKEN_RE.sub("<r>", line)
        return CODE_SHAPE_RE.sub(CODE_TOKEN, line)

    return redact


redact_generic = make_redactor()


def show_cmd(argv: Sequence[str]) -> str:
    """A command for display. Composed commands carry <ticket> and <code> placeholders, never the values."""
    return " ".join(f'"{a}"' if (not a or any(ch.isspace() for ch in a)) else a for a in argv)
