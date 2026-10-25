"""Errors raised by the SDK, and how a gateway reply becomes one.

The gateway answers errors in three shapes:

- ``{"detail": str, "code": str, "fix": str?}`` from ``_error`` in ``backend/hypercompute/app.py``;
- FastAPI validation, ``{"detail": [{"loc", "msg", "type"}, ...]}``;
- ``{"error": {"code", "message", "type"?}}`` from chat.

Every error keeps the HTTP status and the server's code. Classes are picked by HTTP status.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

REDACTED = "[redacted]"
_TEXT_LIMIT = 300  # a body that is not JSON is cut to this many characters


class HypercomputeError(Exception):
    """Base error. ``status`` is the HTTP status (0 when no reply arrived), ``code`` the server's code."""

    def __init__(self, status: int, code: str | None, detail: str, fix: str | None = None, *,
                 locs: tuple[str, ...] = ()) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.fix = fix
        self.locs = locs  # FastAPI validation: each failing field, its loc joined with dots

    def __str__(self) -> str:
        label = self.code or "error"
        head = f"HTTP {self.status} {label}: {self.detail}" if self.status >= 400 else f"{label}: {self.detail}"
        return f"{head}\nfix: {self.fix}" if self.fix else head

    def __repr__(self) -> str:
        return f"{type(self).__name__}(status={self.status}, code={self.code!r})"


class GatewayUnreachable(HypercomputeError):
    """No reply: connection refused, DNS, or a timeout."""


class RequestError(HypercomputeError):
    """400 or 422: the request itself is wrong."""


class OwnerAuthError(HypercomputeError):
    """403 owner_only: not on this machine, or a missing or wrong owner token."""


class NotFoundError(HypercomputeError):
    """404."""


class ConflictError(HypercomputeError):
    """409: valid, but it conflicts with the gateway's current state."""


class BodyTooLarge(HypercomputeError):
    """413 body_too_large, or the SDK's own size check before sending."""


class EngineError(HypercomputeError):
    """500 or 502 other than an interrupted chat."""


class ChatInterrupted(HypercomputeError):
    """A chat cut by a stop or a lost peer. Carries the run id and the text made before the cut."""

    def __init__(self, status: int, code: str | None, detail: str, fix: str | None = None, *,
                 run_id: str | None = None, partial_text: str = "", partial_truncated: bool = False) -> None:
        super().__init__(status, code, detail, fix)
        self.run_id = run_id
        self.partial_text = partial_text
        self.partial_truncated = partial_truncated


class UnavailableError(HypercomputeError):
    """503: no ready deployment, or the gateway's own state cannot be used right now."""


class StreamError(HypercomputeError):
    """A streamed chat ended with an error event (busy, timeout, upstream_error, ...) or closed early (incomplete)."""


class DeploymentFailed(HypercomputeError):
    """``wait_ready`` saw the deployment fail or stop. ``deployment`` holds its ``error`` and ``history``."""

    def __init__(self, deployment: Mapping[str, Any], *, secret: str | None = None) -> None:
        state = str(deployment.get("state"))
        dep_id = str(deployment.get("id", ""))[:8]
        error = scrub(str(deployment.get("error") or "no error recorded"), secret)
        super().__init__(0, f"deployment_{state}", f"deployment {dep_id} is {state}: {error}")
        self.deployment = dict(deployment)


class ReadyTimeout(HypercomputeError):
    """``wait_ready`` reached its deadline. ``deployment`` is the last one seen, or None."""

    def __init__(self, deployment: Mapping[str, Any] | None, waited_s: float) -> None:
        state = deployment.get("state") if deployment else "unknown"
        super().__init__(0, "ready_timeout", f"not ready after {waited_s:.0f} s (last state {state})",
                         "raise the timeout, or check the host's deployment card for progress")
        self.deployment = dict(deployment) if deployment else None


_BY_STATUS: Mapping[int, type[HypercomputeError]] = {
    400: RequestError,
    403: OwnerAuthError,
    404: NotFoundError,
    409: ConflictError,
    413: BodyTooLarge,
    422: RequestError,
    500: EngineError,
    502: EngineError,
    503: UnavailableError,
}


def scrub(text: str | None, secret: str | None) -> str | None:
    """``text`` with every copy of ``secret`` replaced."""
    if text is None or not secret:
        return text
    return text.replace(secret, REDACTED)


def _validation(items: list[Any]) -> tuple[str, tuple[str, ...]]:
    rows = [d for d in items if isinstance(d, Mapping)]
    locs = tuple(".".join(map(str, d.get("loc") or [])) for d in rows)
    detail = "; ".join(f"{loc}: {d.get('msg')}" if loc else str(d.get("msg")) for loc, d in zip(locs, rows,
                                                                                              strict=True))
    return detail or "request validation failed", locs


def parse_body(body: Any) -> dict[str, Any]:
    """The code, detail, fix and validation locs of an error body in any of the three shapes."""
    if not isinstance(body, Mapping):
        return {"code": None, "detail": str(body)[:_TEXT_LIMIT], "fix": None, "locs": (), "error": None}
    detail, locs = body.get("detail"), ()
    if isinstance(detail, list):
        detail, locs = _validation(detail)
    code, error = body.get("code"), body.get("error")
    if isinstance(error, Mapping):
        detail = detail if detail is not None else error.get("message")
        code = code if code is not None else error.get("code")
    elif error is not None and detail is None:
        detail = str(error)
    fix = body.get("fix")
    return {"code": None if code is None else str(code), "detail": "" if detail is None else str(detail),
            "fix": None if fix is None else str(fix), "locs": locs, "error": error}


def from_body(status: int, body: Any, *, secret: str | None = None) -> HypercomputeError:
    """The error for an HTTP status and its parsed body. ``secret`` is scrubbed from every message."""
    parts = parse_body(body)
    detail = scrub(parts["detail"], secret) or f"HTTP {status}"
    fix = scrub(parts["fix"], secret)
    if parts["code"] == "interrupted":
        error = parts["error"]
        return interrupted(status, error if isinstance(error, Mapping) else {"message": detail}, secret=secret)
    cls = _BY_STATUS.get(status, HypercomputeError)
    return cls(status, parts["code"], detail, fix, locs=parts["locs"])


def interrupted(status: int, error: Mapping[str, Any], *, secret: str | None = None) -> ChatInterrupted:
    """ChatInterrupted from the gateway's interrupted error object (streamed or not)."""
    run_id = error.get("run_id")
    return ChatInterrupted(status, "interrupted", scrub(str(error.get("message") or "chat interrupted"), secret) or "",
                           run_id=None if run_id is None else str(run_id),
                           partial_text=str(error.get("partial_text") or ""),
                           partial_truncated=bool(error.get("partial_truncated", False)))


def from_response(response: httpx.Response, *, secret: str | None = None) -> HypercomputeError:
    """The error for a gateway reply with status 400 or more. The body must already be read."""
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:_TEXT_LIMIT]
    return from_body(response.status_code, body, secret=secret)
