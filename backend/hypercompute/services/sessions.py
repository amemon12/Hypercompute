"""Compute sessions and one-time, expiring, rate-limited invitation codes.

Codes and node credentials are stored only as keyed hashes and are never logged.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta

from .. import config
from ..schemas import InviteView, SessionView, utcnow
from ..storage import Store, dumps
from .events import EventBus

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I
CODE_LEN = 10  # 32^10 = 2^50


class JoinRejected(Exception):
    def __init__(self, status: int, code: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.fix = status, code, fix


def normalize_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def format_code(raw: str) -> str:
    return f"{raw[:5]}-{raw[5:]}"


def new_code() -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(CODE_LEN))


def hash_secret(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def new_credential() -> str:
    return secrets.token_urlsafe(32)


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit, self._window, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self._clock()
        q = self._hits[key]
        while q and now - q[0] > self._window:
            q.popleft()
        if len(q) >= self._limit:
            return False
        q.append(now)
        return True


class SessionService:
    def __init__(self, store: Store, events: EventBus, secret: bytes, gateway_port: int,
                 lan_addresses: Callable[[], list[str]], invite_ttl_s: int = config.INVITE_TTL_S,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._store, self._events, self._secret = store, events, secret
        self._port, self._lan = gateway_port, lan_addresses
        self._ttl = invite_ttl_s
        self._clock = clock
        self.limiter = RateLimiter(config.JOIN_RATE_LIMIT_PER_MIN)

    def _view(self, row: dict, invite: InviteView | None = None) -> SessionView:
        return SessionView(id=row["id"], name=row["name"], status=row["status"], created_at=row["created_at"],
                           lan_addresses=self._lan(), gateway_port=self._port, invite=invite)

    def lan_addresses(self) -> list[str]:
        """Addresses this gateway actually listens on for friends (respects `host --bind`)."""
        return self._lan()

    def current(self) -> SessionView | None:
        row = self._store.query_one("SELECT doc FROM sessions WHERE status='active' ORDER BY created_at DESC LIMIT 1")
        return self._view(_load(row["doc"])) if row else None

    def get(self, session_id: str) -> dict | None:
        row = self._store.query_one("SELECT doc FROM sessions WHERE id=?", (session_id,))
        return _load(row["doc"]) if row else None

    def create(self, name: str) -> SessionView:
        existing = self.current()
        if existing is None:
            doc = {"id": uuid.uuid4().hex, "name": name, "status": "active", "created_at": self._clock()}
            self._store.execute("INSERT INTO sessions (id, status, created_at, doc) VALUES (?,?,?,?)",
                                (doc["id"], "active", doc["created_at"].isoformat(), dumps(doc)))
            self._events.publish("session.created", f"Session '{name}' created", session_id=doc["id"])
            session_id = doc["id"]
        else:
            session_id = existing.id
        return self.issue_invite(session_id)

    def issue_invite(self, session_id: str) -> SessionView:
        doc = self.get(session_id)
        if not doc or doc["status"] != "active":
            raise JoinRejected(404, "session_not_active", "session is not active")
        raw = new_code()
        expires = self._clock() + timedelta(seconds=self._ttl)
        self._store.execute(
            "INSERT INTO invites (id, session_id, code_hash, expires_at, created_at) VALUES (?,?,?,?,?)",
            (uuid.uuid4().hex, session_id, hash_secret(self._secret, raw), expires.isoformat(),
             self._clock().isoformat()))
        lan = self._lan()
        address = f"{lan[0]}:{self._port}" if lan else f"<this-machine-LAN-IP>:{self._port}"
        code = format_code(raw)
        invite = InviteView(code=code, expires_at=expires, join_address=address,
                            join_command=f"uv run hypercompute join --host {address} --code {code}")
        self._events.publish("invite.created", "New invitation issued", session_id=session_id,
                             expires_at=expires.isoformat())
        return self._view(doc, invite)

    def close(self, session_id: str) -> SessionView:
        doc = self.get(session_id)
        if not doc:
            raise JoinRejected(404, "session_not_found", "no such session")
        doc = {**doc, "status": "closed"}
        with self._store.tx() as conn:
            conn.execute("UPDATE sessions SET status='closed', doc=? WHERE id=?", (dumps(doc), session_id))
            conn.execute("UPDATE invites SET used_at=? WHERE session_id=? AND used_at IS NULL",
                         (self._clock().isoformat(), session_id))
        self._events.publish("session.closed", "Session closed", session_id=session_id)
        return self._view(doc)

    def redeem(self, code: str, client_ip: str) -> str:
        """Consume a valid invite atomically and return its session id."""
        if not self.limiter.allow(client_ip):
            raise JoinRejected(429, "rate_limited", "too many join attempts from this address; wait a minute")
        raw = normalize_code(code)
        if len(raw) != CODE_LEN or any(ch not in ALPHABET for ch in raw):
            raise JoinRejected(403, "invalid_code", "invitation code is not valid",
                               fix="ask the host for a fresh code from the dashboard")
        digest = hash_secret(self._secret, raw)
        now = self._clock()
        with self._store.tx() as conn:
            row = conn.execute("SELECT id, session_id, expires_at, used_at, failed_attempts FROM invites "
                               "WHERE code_hash=?", (digest,)).fetchone()
            if row is None:
                raise JoinRejected(403, "invalid_code", "invitation code is not valid",
                                   fix="ask the host for a fresh code from the dashboard")
            if row["used_at"] is not None:
                raise JoinRejected(403, "code_used", "this invitation code was already used",
                                   fix="each friend needs their own code: the host clicks 'New invite'")
            if datetime.fromisoformat(row["expires_at"]) <= now:
                raise JoinRejected(403, "code_expired", "this invitation code expired",
                                   fix="ask the host for a fresh code")
            session = conn.execute("SELECT status FROM sessions WHERE id=?", (row["session_id"],)).fetchone()
            if session is None or session["status"] != "active":
                raise JoinRejected(403, "session_closed", "the session for this code is closed")
            conn.execute("UPDATE invites SET used_at=? WHERE id=?", (now.isoformat(), row["id"]))
        return row["session_id"]


def _load(doc: str) -> dict:
    import json

    d = json.loads(doc)
    if isinstance(d.get("created_at"), str):
        d["created_at"] = datetime.fromisoformat(d["created_at"])
    return d
