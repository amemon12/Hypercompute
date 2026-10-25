"""``Pool``: a client for the Hypercompute gateway on this machine.

Methods return the gateway's JSON as plain dicts and lists, so ``backend/hypercompute/schemas.py`` stays the only
schema. Owner routes answer only loopback clients, so the base URL must be ``http://`` on 127.0.0.1, localhost or
[::1]. Reads need no token. Writes send ``X-Hypercompute-Owner``; the token comes from the ``owner_token`` argument,
then ``HYPERCOMPUTE_OWNER_TOKEN``, then ``GET /api/owner/bootstrap`` (fetched once, before the first write).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import httpx

from . import errors
from .errors import (
    BodyTooLarge,
    ConflictError,
    DeploymentFailed,
    GatewayUnreachable,
    HypercomputeError,
    ReadyTimeout,
    StreamError,
)
from .sse import SSEParser
from .types import ChatChunk

DEFAULT_URL = "http://127.0.0.1:8001"
URL_ENV = "HYPERCOMPUTE_URL"
TOKEN_ENV = "HYPERCOMPUTE_OWNER_TOKEN"  # noqa: S105 - the variable name, not a token
OWNER_HEADER = "X-Hypercompute-Owner"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})  # httpx gives [::1] as ::1

KiB = 1024
# app.py BODY_LIMIT_CHAT, the loopback limit (one image plus its text); other clients get 4 MiB
CHAT_BODY_LIMIT = 16384 * KiB
MAX_TOKENS = 262144  # PlanRequest.ctx_size and the explain bounds
MAX_SLOTS = 8  # explain: hcstream accepts 1 to 8 slot buffers
MAX_PARALLEL = 16  # PlanRequest.parallel
STRATEGIES = frozenset({"auto", "all_selected", "fastest_fit", "local_only"})
ENGINES = frozenset({"llama_rpc", "exo", "airllm", "vllm"})
OBJECTIVES = frozenset({"interactive", "throughput"})
OVERFLOWS = frozenset({"auto", "peers", "host_ram"})
TERMINAL_STATES = frozenset({"stopped", "failed"})
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # app.py MODEL_ID_PATTERN
ID = re.compile(r"[A-Za-z0-9_-]{1,64}")  # app.py ID_PATTERN: deployment and plan ids
HEADER_VALUE = re.compile(r"[!-~]{1,200}")  # an Idempotency-Key the caller picks
MAX_POLL_FAILURES = 5  # wait_ready gives up on the 5th failed poll in a row

PLAN_TIMEOUT_S = 180.0
EXPLAIN_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 300.0
CHAT_TIMEOUT = httpx.Timeout(30.0, read=1800.0)
CHAT_PATH = "/v1/chat/completions"
JSON_HEADERS = {"Content-Type": "application/json"}

# Every (method, path) this SDK calls; tests check each one against backend/hypercompute/app.py.
ROUTES = (
    ("GET", "/api/health"),
    ("GET", "/api/owner/bootstrap"),
    ("GET", "/api/network"),
    ("GET", "/api/nodes"),
    ("GET", "/api/peers"),
    ("GET", "/api/models"),
    ("GET", "/api/capacity/pools"),
    ("GET", "/api/capacity/explain"),
    ("POST", "/api/plans"),
    ("POST", "/api/deployments"),
    ("GET", "/api/deployments"),
    ("GET", "/api/deployments/{dep_id}"),
    ("DELETE", "/api/deployments/{dep_id}"),
    ("GET", "/v1/models"),
    ("POST", CHAT_PATH),
)

ProgressCallback = Callable[[str, str | None], None]


# ----------------------------------------------------------------------------- argument checks


def check_base_url(base_url: str) -> str:
    """``base_url`` when it is ``http://`` on a loopback name, else ValueError with the fix."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"base_url must be a URL such as {DEFAULT_URL}")
    try:
        url = httpx.URL(base_url.strip())
    except httpx.InvalidURL as exc:
        raise ValueError(f"base_url is not a valid URL ({exc})") from exc
    if url.scheme != "http" or url.host.lower() not in LOOPBACK_HOSTS or url.userinfo:
        raise ValueError(
            f"owner routes only answer this machine, so base_url must be http:// on 127.0.0.1, localhost or [::1] "
            f"(got {url.scheme or '?'}://{url.host or '?'}). Run on the host, or tunnel with "
            f"`ssh -L 8001:127.0.0.1:8001 <host>` and keep base_url on {DEFAULT_URL}")
    return str(url)


def _int_in(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}, not {value!r}")
    return value


def _choice(name: str, value: Any, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}, not {value!r}")
    return value


def _flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False, not {value!r}")
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _ident(name: str, value: Any) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f"{name} must be 1 to 64 letters, digits, '_' or '-', not {value!r}")
    return value


def _header_value(name: str, value: Any) -> str:
    if not isinstance(value, str) or not HEADER_VALUE.fullmatch(value):
        raise ValueError(f"{name} must be 1 to 200 printable ASCII characters without spaces")
    return value


def _strings(name: str, value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of strings")
    return [_text(f"{name} item", v) for v in value]


def _layout(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not 1 <= len(value) <= 64:
        raise ValueError("layout must be a list of 1 to 64 stages")
    if not all(isinstance(stage, Mapping) for stage in value):
        raise ValueError("each layout stage must be a dict with node_id, device and layers")
    return [dict(stage) for stage in value]


def plan_body(model_id: str, *, strategy: str = "all_selected", ctx_size: int = 4096,
              node_ids: Sequence[str] | None = None, allow_ram_tier: bool = True, allow_capacity_mode: bool = True,
              engine: str | None = None, objective: str | None = None, overflow: str | None = None,
              use_profiles: bool | None = None, parallel: int = 1,
              layout: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """A ``PlanRequest`` body. Only the keys given are sent, because the gateway forbids extra fields."""
    body: dict[str, Any] = {
        "model_id": _text("model_id", model_id),
        "strategy": _choice("strategy", strategy, STRATEGIES),
        "ctx_size": _int_in("ctx_size", ctx_size, 256, MAX_TOKENS),
        "allow_ram_tier": _flag("allow_ram_tier", allow_ram_tier),
        "allow_capacity_mode": _flag("allow_capacity_mode", allow_capacity_mode),
    }
    if node_ids is not None:
        body["node_ids"] = _strings("node_ids", node_ids)
    if engine is not None:
        body["engine"] = _choice("engine", engine, ENGINES)
    if objective is not None:
        body["objective"] = _choice("objective", objective, OBJECTIVES)
    if overflow is not None:
        body["overflow"] = _choice("overflow", overflow, OVERFLOWS)
    if use_profiles is not None:
        body["use_profiles"] = _flag("use_profiles", use_profiles)
    if _int_in("parallel", parallel, 1, MAX_PARALLEL) != 1:
        body["parallel"] = parallel
    if layout is not None:
        body["layout"] = _layout(layout)
    return body


def _check_messages(messages: Any) -> list[Any]:
    if isinstance(messages, str) or not isinstance(messages, Sequence) or not messages:
        raise ValueError("messages must be a non-empty list of {'role': ..., 'content': ...} dicts")
    for i, message in enumerate(messages):
        if not isinstance(message, Mapping) or "role" not in message or "content" not in message:
            raise ValueError(f"messages[{i}] must be a dict with 'role' and 'content'")
    return list(messages)


def chat_body(messages: Any, *, stream: bool, model: str | None, max_tokens: int | None,
              temperature: float | None, extra: Mapping[str, Any]) -> bytes:
    """The chat request as JSON bytes. Raises BodyTooLarge over the gateway's loopback chat limit, before sending."""
    if {"messages", "stream"} & set(extra):
        raise ValueError("pass messages as the first argument, and use stream() for a streamed reply")
    body: dict[str, Any] = {"messages": _check_messages(messages), "stream": stream}
    if model is not None:
        body["model"] = _text("model", model)
    if max_tokens is not None:
        body["max_tokens"] = _int_in("max_tokens", max_tokens, 1, MAX_TOKENS)
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ValueError(f"temperature must be a number, not {temperature!r}")
        body["temperature"] = temperature
    body.update(extra)
    try:
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the chat request cannot be sent as JSON ({exc})") from exc
    if len(data) > CHAT_BODY_LIMIT:
        raise BodyTooLarge(413, "body_too_large",
                           f"the chat request is {len(data) // KiB} KiB, over the gateway's "
                           f"{CHAT_BODY_LIMIT // KiB} KiB limit; it was not sent", "send fewer or shorter messages")
    return data


# ----------------------------------------------------------------------------- reply helpers


def _bad_reply(status: int, path: str) -> HypercomputeError:
    return HypercomputeError(status, "bad_response", f"{path} answered with JSON of an unexpected shape")


def _dict(body: Any, path: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise _bad_reply(200, path)
    return body


def _list(body: Any, path: str) -> list[Any]:
    if not isinstance(body, list):
        raise _bad_reply(200, path)
    return body


def _reply_id(body: Mapping[str, Any], path: str) -> str:
    value = body.get("id")
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise _bad_reply(200, path)
    return value


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise HypercomputeError(response.status_code, "bad_response",
                                "the gateway answered with a body that is not JSON") from exc


def _public_peer(peer: Mapping[str, Any]) -> dict[str, Any]:
    """A network peer without its Iroh ids, as scripts/capacity/paper/matrix.py records it."""
    out = {k: v for k, v in peer.items() if k != "peer_id"}
    if isinstance(out.get("path"), Mapping):
        out["path"] = {k: v for k, v in out["path"].items() if k != "remote_id"}
    return out


def _node_summary(node: Mapping[str, Any]) -> dict[str, Any]:
    contribution = node.get("contribution")
    return {"id": node.get("id"), "display_name": node.get("display_name"), "state": node.get("state"),
            "heartbeat_age_s": node.get("heartbeat_age_s"),
            "max_bytes": contribution.get("max_bytes") if isinstance(contribution, Mapping) else None}


def _events(chunks: Iterable[bytes]) -> Iterator[str]:
    """The data of every SSE event, including a last one cut before its blank line."""
    parser = SSEParser()
    try:
        for raw in chunks:
            yield from parser.feed(raw)
        yield from parser.flush()
    except ValueError as exc:
        raise StreamError(200, "bad_event", str(exc)) from exc


def _chunk(data: str, secret: str | None) -> ChatChunk | None:
    """One event as a ChatChunk; None for a payload that is not an object. Raises on an error event."""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise StreamError(200, "bad_event", "the stream carried an event that is not JSON") from exc
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if error:
        if isinstance(error, Mapping) and error.get("code") == "interrupted":
            raise errors.interrupted(200, error, secret=secret)
        code = error.get("code") if isinstance(error, Mapping) else None
        message = error.get("message") if isinstance(error, Mapping) else error
        raise StreamError(200, "stream_error" if code is None else str(code),
                          errors.scrub(str(message or "the stream ended with an error"), secret) or "")
    return ChatChunk.from_payload(payload)


def read_chunks(chunks: Iterable[bytes], secret: str | None = None) -> Iterator[ChatChunk]:
    """Fold the gateway's chat SSE bytes into ChatChunks, ending at ``data: [DONE]``."""
    finished = False
    for data in _events(chunks):
        if data.strip() == "[DONE]":
            return
        chunk = _chunk(data, secret)
        if chunk is None:
            continue
        finished = finished or chunk.finish_reason is not None
        yield chunk
    if not finished:
        raise StreamError(200, "incomplete", "the stream closed before [DONE] and without a finish_reason",
                          "check the host's deployment card; the engine may have stopped")


# ----------------------------------------------------------------------------- Pool


class Pool:
    """A client for one Hypercompute gateway on this machine."""

    def __init__(self, base_url: str = DEFAULT_URL, *, owner_token: str | None = None, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._base_url = check_base_url(base_url)
        if owner_token is not None:
            self._token: str | None = _text("owner_token", owner_token)
            self._token_source: str | None = "argument"  # noqa: S105 - a source label
        else:
            env = os.environ.get(TOKEN_ENV, "").strip()
            self._token, self._token_source = (env, "environment") if env else (None, None)
        self._sleep, self._clock = sleep, clock
        self._client = httpx.Client(base_url=self._base_url, trust_env=False, timeout=timeout, transport=transport)

    @classmethod
    def from_env(cls, **kwargs: Any) -> Pool:
        """A Pool on ``HYPERCOMPUTE_URL`` (default http://127.0.0.1:8001) with ``HYPERCOMPUTE_OWNER_TOKEN``."""
        return cls(os.environ.get(URL_ENV, "").strip() or DEFAULT_URL, **kwargs)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def token_source(self) -> str:
        """Where the owner token came from: argument, environment, bootstrap, or none yet."""
        return self._token_source or "none yet"

    def __repr__(self) -> str:
        return f"Pool(base_url={self._base_url!r}, token_source={self.token_source!r})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport and the owner token

    def _unreachable(self, exc: httpx.RequestError) -> GatewayUnreachable:
        detail = errors.scrub(f"{type(exc).__name__}: {exc}", self._token) or "no reply"
        return GatewayUnreachable(0, "unreachable", detail,
                                  "check that the host is running on this machine and that base_url has its port")

    def _headers(self, owner: bool, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if owner:
            headers[OWNER_HEADER] = self._token if self._token is not None else self._bootstrap()
        return headers

    def _send(self, method: str, path: str, *, owner: bool = False, headers: Mapping[str, str] | None = None,
              **kwargs: Any) -> httpx.Response:
        request_headers = self._headers(owner, headers)
        try:
            return self._client.request(method, path, headers=request_headers, **kwargs)
        except httpx.RequestError as exc:
            raise self._unreachable(exc) from exc

    def _bootstrap(self) -> str:
        """Fetch the owner token from GET /api/owner/bootstrap (loopback only)."""
        response = self._send("GET", "/api/owner/bootstrap")
        if response.status_code >= 400:
            raise errors.from_response(response, secret=self._token)
        body = _json(response)
        token = body.get("owner_token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise HypercomputeError(response.status_code, "no_owner_token", "the gateway handed out no owner token")
        self._token, self._token_source = token, "bootstrap"  # noqa: S105 - a source label
        return token

    def _rotated(self, response: httpx.Response) -> bool:
        """After a 403 owner_only on a bootstrap token, fetch the restarted host's new token and say so."""
        if response.status_code != 403 or self._token_source != "bootstrap":  # noqa: S105 - a source label
            return False
        try:
            code = errors.parse_body(response.json())["code"]
        except ValueError:
            return False
        if code != "owner_only":
            return False
        self._bootstrap()
        return True

    def _call(self, method: str, path: str, *, owner: bool = False, **kwargs: Any) -> Any:
        response = self._send(method, path, owner=owner, **kwargs)
        if owner and self._rotated(response):
            response = self._send(method, path, owner=True, **kwargs)
        if response.status_code >= 400:
            raise errors.from_response(response, secret=self._token)
        return _json(response)

    # ------------------------------------------------------------------ reads

    def health(self) -> dict[str, Any]:
        return _dict(self._call("GET", "/api/health"), "/api/health")

    def nodes(self) -> list[dict[str, Any]]:
        """Every node, the host itself included (``is_local: true``)."""
        return _list(self._call("GET", "/api/nodes"), "/api/nodes")

    def peers(self) -> list[dict[str, Any]]:
        """Each peer's consent, worker devices, cache bytes and seed progress."""
        return _list(self._call("GET", "/api/peers"), "/api/peers")

    def models(self) -> list[dict[str, Any]]:
        """The model catalog as ModelView dicts; use ``artifact.id`` and ``status``."""
        return _list(self._call("GET", "/api/models"), "/api/models")

    def served_models(self) -> list[str]:
        """Ids of the model the ready deployment serves (never a tuner trial); empty when none is ready."""
        body = _dict(self._call("GET", "/v1/models"), "/v1/models")
        return [str(m["id"]) for m in body.get("data") or [] if isinstance(m, Mapping) and m.get("id")]

    def pools(self) -> dict[str, Any]:
        return _dict(self._call("GET", "/api/capacity/pools"), "/api/capacity/pools")

    def deployments(self) -> list[dict[str, Any]]:
        """The latest deployments, newest first."""
        return _list(self._call("GET", "/api/deployments"), "/api/deployments")

    def deployment(self, dep_id: str) -> dict[str, Any]:
        path = f"/api/deployments/{_ident('dep_id', dep_id)}"
        return _dict(self._call("GET", path), path)

    def status(self) -> dict[str, Any]:
        """The facts ``hypercompute status`` prints, without Iroh endpoint or peer ids."""
        health = self.health()
        network = _dict(self._call("GET", "/api/network"), "/api/network")
        deps, served = self.deployments(), self.served_models()
        engine = health.get("engine") if isinstance(health.get("engine"), Mapping) else {}
        return {"version": health.get("version"), "engine_release": engine.get("release"),
                "transport": network.get("transport", "lan"),
                "connections": [_public_peer(p) for p in network.get("peers") or [] if isinstance(p, Mapping)],
                "nodes": [_node_summary(n) for n in self.nodes() if isinstance(n, Mapping)],
                "deployment": deps[0] if deps else None, "ready_model": served[0] if served else None}

    def explain(self, model_id: str, *, ctx_size: int = 4096, prompt_tokens: int = 512, output_tokens: int = 256,
                warm: bool = False, slots: int = 2) -> dict[str, Any]:
        """The capacity optimizer's view of a model on this pool, without planning or loading it."""
        if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
            raise ValueError(f"model_id must be a catalog id such as qwen2.5-7b-instruct-q4_k_m, not {model_id!r}")
        params = {"model_id": model_id, "ctx_size": _int_in("ctx_size", ctx_size, 256, MAX_TOKENS),
                  "prompt_tokens": _int_in("prompt_tokens", prompt_tokens, 0, MAX_TOKENS),
                  "output_tokens": _int_in("output_tokens", output_tokens, 1, MAX_TOKENS),
                  "warm": _flag("warm", warm), "slots": _int_in("slots", slots, 1, MAX_SLOTS)}
        if prompt_tokens + output_tokens > ctx_size:
            raise ValueError(f"prompt_tokens + output_tokens ({prompt_tokens + output_tokens}) exceed ctx_size "
                             f"({ctx_size}); raise ctx_size or lower the token counts")
        path = "/api/capacity/explain"
        return _dict(self._call("GET", path, params=params, timeout=EXPLAIN_TIMEOUT_S), path)

    # ------------------------------------------------------------------ plans and deployments

    def plan(self, model_id: str, *, strategy: str = "all_selected", ctx_size: int = 4096,
             node_ids: Sequence[str] | None = None, allow_ram_tier: bool = True, allow_capacity_mode: bool = True,
             engine: str | None = None, objective: str | None = None, overflow: str | None = None,
             use_profiles: bool | None = None, parallel: int = 1,
             layout: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """POST /api/plans. Options left as None are not sent, and the gateway uses its defaults."""
        body = plan_body(model_id, strategy=strategy, ctx_size=ctx_size, node_ids=node_ids,
                         allow_ram_tier=allow_ram_tier, allow_capacity_mode=allow_capacity_mode, engine=engine,
                         objective=objective, overflow=overflow, use_profiles=use_profiles, parallel=parallel,
                         layout=layout)
        return _dict(self._call("POST", "/api/plans", owner=True, json=body, timeout=PLAN_TIMEOUT_S), "/api/plans")

    def deploy(self, plan_id: str | None = None, *, plan: Mapping[str, Any] | None = None,
               idempotency_key: str | None = None) -> dict[str, Any]:
        """POST /api/deployments with a stored plan id, or a PlanRequest dict (see ``plan_body``)."""
        if (plan_id is None) == (plan is None):
            raise ValueError("give exactly one of plan_id or plan")
        body = {"plan_id": _text("plan_id", plan_id)} if plan is None else {"plan": dict(plan)}
        key = uuid.uuid4().hex if idempotency_key is None else _header_value("idempotency_key", idempotency_key)
        extra = {} if plan is None else {"timeout": PLAN_TIMEOUT_S}  # an inline plan is planned before the reply
        try:
            reply = self._call("POST", "/api/deployments", owner=True, json=body, headers={"Idempotency-Key": key},
                               **extra)
        except GatewayUnreachable as err:
            found = self._find_by_key(key) if isinstance(err.__cause__, httpx.TimeoutException) else None
            if found is None:
                raise
            return found
        return _dict(reply, "/api/deployments")

    def _find_by_key(self, key: str) -> dict[str, Any] | None:
        """The deployment a create with this idempotency key made, when its reply was lost."""
        try:
            deps = self.deployments()
        except HypercomputeError:
            return None
        return next((d for d in deps if isinstance(d, dict) and d.get("idempotency_key") == key), None)

    def wait_ready(self, dep_id: str, *, timeout: float = 7200.0, poll: float = 2.0,
                   on_progress: ProgressCallback | None = None) -> dict[str, Any]:
        """Poll until the deployment is ready. Raises DeploymentFailed (failed or stopped) or ReadyTimeout."""
        _ident("dep_id", dep_id)
        start, last, seen, failures = self._clock(), None, None, 0
        while True:
            try:
                dep = self.deployment(dep_id)
            except HypercomputeError as err:
                if not (isinstance(err, GatewayUnreachable) or err.status >= 500):
                    raise
                failures += 1
                if failures >= MAX_POLL_FAILURES:
                    raise
            else:
                failures, seen = 0, dep
                state, progress = dep.get("state"), dep.get("progress")
                if (state, progress) != last:
                    last = (state, progress)
                    if on_progress is not None:
                        on_progress(str(state), progress)
                if state == "ready":
                    return dep
                if state in TERMINAL_STATES:
                    raise DeploymentFailed(dep, secret=self._token)
            waited = self._clock() - start
            if waited > timeout:
                raise ReadyTimeout(seen, waited)
            self._sleep(poll)

    def deploy_model(self, model_id: str, *, wait: bool = True, ready_timeout: float = 7200.0,
                     on_progress: ProgressCallback | None = None, **plan_options: Any) -> dict[str, Any]:
        """Plan, deploy and (by default) wait until ready. A stale plan is planned once more. A deployment that is
        already active raises ConflictError: this never stops someone else's deployment."""
        plan = self.plan(model_id, **plan_options)
        try:
            dep = self.deploy(_reply_id(plan, "/api/plans"))
        except ConflictError as err:
            if err.code != "plan_stale":
                raise
            plan = self.plan(model_id, **plan_options)
            dep = self.deploy(_reply_id(plan, "/api/plans"))
        if not wait:
            return dep
        return self.wait_ready(_reply_id(dep, "/api/deployments"), timeout=ready_timeout, on_progress=on_progress)

    def stop(self, dep_id: str | None = None) -> dict[str, Any] | None:
        """DELETE /api/deployments/{id}. None stops the one active deployment, and returns None when there is none."""
        if dep_id is None:
            active = next((d for d in self.deployments()
                           if isinstance(d, Mapping) and d.get("state") not in TERMINAL_STATES), None)
            if active is None:
                return None
            dep_id = active.get("id")
        path = f"/api/deployments/{_ident('dep_id', dep_id)}"
        return _dict(self._call("DELETE", path, owner=True, timeout=STOP_TIMEOUT_S), path)

    # ------------------------------------------------------------------ chat

    def chat(self, messages: Sequence[Mapping[str, Any]], *, model: str | None = None, max_tokens: int | None = None,
             temperature: float | None = None, **extra: Any) -> dict[str, Any]:
        """A non-streamed chat completion. ``extra`` passes llama.cpp fields such as seed or cache_prompt."""
        content = chat_body(messages, stream=False, model=model, max_tokens=max_tokens, temperature=temperature,
                            extra=extra)
        reply = self._call("POST", CHAT_PATH, owner=True, content=content, headers=JSON_HEADERS,
                           timeout=CHAT_TIMEOUT)
        return _dict(reply, CHAT_PATH)

    def stream(self, messages: Sequence[Mapping[str, Any]], *, model: str | None = None,
               max_tokens: int | None = None, temperature: float | None = None, **extra: Any) -> Iterator[ChatChunk]:
        """A streamed chat as ChatChunks. Arguments are checked now; the request is sent on the first ``next()``.
        Leaving the loop early closes the response, and the gateway then stops generating."""
        content = chat_body(messages, stream=True, model=model, max_tokens=max_tokens, temperature=temperature,
                            extra=extra)
        return self._stream_chunks(content)

    def _send_stream(self, content: bytes) -> httpx.Response:
        """An open streamed response; a non-200 reply is read in full and closed."""
        request = self._client.build_request("POST", CHAT_PATH, content=content,
                                             headers=self._headers(True, JSON_HEADERS), timeout=CHAT_TIMEOUT)
        try:
            response = self._client.send(request, stream=True)
        except httpx.RequestError as exc:
            raise self._unreachable(exc) from exc
        if response.status_code != 200:
            try:
                response.read()
            except httpx.RequestError as exc:
                raise self._unreachable(exc) from exc
            finally:
                response.close()
        return response

    def _stream_chunks(self, content: bytes) -> Iterator[ChatChunk]:
        response = self._send_stream(content)
        if response.status_code != 200 and self._rotated(response):
            response = self._send_stream(content)
        if response.status_code != 200:
            raise errors.from_response(response, secret=self._token)
        try:
            yield from read_chunks(response.iter_bytes(), self._token)
        except httpx.RequestError as exc:
            raise self._unreachable(exc) from exc
        finally:
            response.close()
