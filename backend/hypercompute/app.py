"""FastAPI gateway: owner dashboard API, node (agent) API and OpenAI-compatible inference."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import mimetypes
import re
from collections.abc import Callable, Mapping
from contextlib import aclosing, asynccontextmanager
from types import MappingProxyType
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Path, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__, config
from .adapters import registry
from .adapters.base import ENGINE_ROADMAP, EngineAdapterError
from .capacity.service import CapacityUnavailable
from .capacity.stream_config import MAX_SLOTS as MAX_STREAM_SLOTS  # hcstream.c accepts 1..8 slot buffers
from .capacity.types import PathClass
from .coordinator import Coordinator
from .engine import EngineError
from .multimodal import ImageInputError, accepts_images, count_images
from .schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    CreateDeploymentRequest,
    CreateSessionRequest,
    Deployment,
    HeartbeatRequest,
    HeartbeatResponse,
    JoinRequest,
    JoinResponse,
    MeasuredProfile,
    Model,
    ModelView,
    NodeView,
    PeerSeedRequest,
    PeerWorkerRequest,
    Plan,
    PlanRequest,
    ReplanDecision,
    RunRecord,
    SessionView,
    TuneRequest,
    TuneResult,
    utcnow,
)
from .services.deployments import DeploymentError
from .services.models import ModelError
from .services.nodes import AuthError
from .services.peer_seed import ManagedRefusal, build_seed, load_plan, parse_layers, plan_layers, stop_dir_for
from .services.replan import ReplanController
from .services.sessions import JoinRejected
from .services.streaming import ChatInterrupted

log = logging.getLogger("hypercompute.app")
LOOPBACK_CLIENTS = {"127.0.0.1", "::1"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

KiB = 1024
BODY_LIMIT_DEFAULT = 64 * KiB
BODY_LIMIT_NODE = 256 * KiB  # join (full inventory) and heartbeats (memory pools, devices)
BODY_LIMIT_CHAT = 16384 * KiB  # loopback: one image at multimodal.MAX_IMAGE_BYTES, base64 encoded, plus the text
BODY_LIMIT_CHAT_LAN = 4096 * KiB  # any other client, whose chat the owner check refuses anyway
_NODE_BODY_PATH = re.compile(r"^/api/(sessions/join|nodes/[^/]+/heartbeat)$")
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
_SECURITY_HEADER_PAIRS = [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"  # catalog ids: qwen2.5-7b-instruct-q4_k_m
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"  # deployment and plan ids (uuid hex in the gateway, short names in tests)
MAX_TOKENS = 262144  # PlanRequest.ctx_size's upper bound
PATH_READ_TIMEOUT_S = 2.0  # a worker thread's wait for the event loop to read the Iroh status; then "unknown"

# CapacityUnavailable.code -> HTTP status of the reply.
#   422  the request names something the optimizer cannot plan; change the request
#   503  the gateway's own state (model files, pools, host memory, stored records) cannot be read or used right now;
#        the same request can succeed once that state is fixed
#   409  the request is valid but conflicts with this host's current state
#   500  the optimizer broke one of its own invariants
# A code missing here answers CAPACITY_STATUS_DEFAULT (422, the status every code had before this table) and is
# logged, so a new code gets classified here. A 422 raised over an OSError (`raise ... from`, at any depth) answers 503:
# the gateway could not read its own files, as when "manifest" wraps a GGUF shard that cannot be opened.
CAPACITY_STATUS: Mapping[str, int] = MappingProxyType({
    # request problems
    "bad_request": 422,  # explain arguments the service refuses
    "unknown_model": 422,  # not in the model catalog
    "not_gguf": 422,  # not a llama.cpp GGUF artifact
    "not_downloaded": 422,  # the GGUF is not on disk yet; download it first
    "manifest": 422,  # the GGUF's tensor table does not map onto placement units
    "workload": 422,  # the workload does not fit the request's context
    "participation": 422,  # 'run together' cannot give every selected device a span
    "no_peers": 422,  # 'run together' without an eligible peer device
    "engine": 422,  # the request forces an engine other than llama.cpp
    "objective": 422,  # throughput objective; the optimizer scores single-stream latency
    "overflow_pinned": 422,  # the request pins the GPU overflow
    "ram_tier_off_unified": 422,  # RAM tier off on a host whose CPU and GPU share one pool
    "moe": 422,  # mixture-of-experts model: the planner's moe_offload tier plans it, not the layer optimizer
    # server state
    "io": 503,  # reading the downloaded GGUF failed
    "model_files": 503,  # the catalog entry's file path is unsafe, or its directory cannot be listed
    "pools": 503,  # the pool report refused inconsistent figures
    "host_memory": 503,  # no free-memory observation of this machine's RAM pool
    "bridge": 503,  # the live state could not be turned into optimizer inputs
    "lease_row": 503,  # a stored lease row is malformed
    "optimizer_input": 503,  # a stored measurement or profile is malformed
    # conflict with the host's current state
    "host_ineligible": 409,  # this machine is not eligible to run a model now
    # internal
    "candidate": 500,  # the optimizer produced a stage without a memory demand
})
CAPACITY_STATUS_DEFAULT = 422
# nosniff blocks assets served with a wrong type; a Windows registry can map .js/.css to text/plain.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

FALLBACK_PAGE = """<!doctype html><meta charset=utf-8><title>Hypercompute</title>
<body style="font:16px system-ui;background:#111;color:#ddd;padding:2rem">
<h1>Hypercompute gateway is running</h1><p>The dashboard has not been built yet. Run:</p>
<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>then reload. The API is live under /api.</p></body>"""


class OwnerAuthError(Exception):
    pass


def _error(status: int, code: str, detail: str, fix: str | None = None) -> JSONResponse:
    body = {"detail": detail, "code": code}
    if fix:
        body["fix"] = fix
    return JSONResponse(status_code=status, content=body)


def _over_os_error(cause: BaseException | None) -> bool:
    """Whether the explicit cause chain (`raise ... from`) holds an OSError."""
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        if isinstance(cause, OSError):
            return True
        seen.add(id(cause))
        cause = cause.__cause__
    return False


def capacity_status(code: str, cause: BaseException | None = None) -> int:
    """The HTTP status for a CapacityUnavailable code (CAPACITY_STATUS); an unclassified code is logged. `cause` is
    the exception's __cause__: a 422 over an OSError is the gateway's state, 503."""
    status = CAPACITY_STATUS.get(code)
    if status is None:
        log.warning("CapacityUnavailable code %r has no entry in CAPACITY_STATUS; answering %d", code,
                    CAPACITY_STATUS_DEFAULT)
        status = CAPACITY_STATUS_DEFAULT
    return 503 if status == 422 and _over_os_error(cause) else status


def body_limit(path: str, client: str = "") -> int:
    if path == "/v1/chat/completions":
        # bodies are buffered before auth and only this machine may chat, so only it gets room for an image
        return BODY_LIMIT_CHAT if client in LOOPBACK_CLIENTS else BODY_LIMIT_CHAT_LAN
    if _NODE_BODY_PATH.match(path):
        return BODY_LIMIT_NODE
    return BODY_LIMIT_DEFAULT


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _too_large(limit: int) -> JSONResponse:
    return _error(413, "body_too_large", f"request body is over the {limit // KiB} KiB limit")


def _replay(body: bytes, receive: Receive) -> Receive:
    sent = False

    async def replay() -> Message:
        nonlocal sent
        if sent:
            return await receive()  # later calls wait for the disconnect, as the app expects
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return replay


class BodyLimitMiddleware:
    """Rejects an oversized body with 413 before routing, parsing or auth: the gateway listens on the LAN."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = body_limit(scope["path"], (scope.get("client") or ("",))[0])
        if (_content_length(scope) or 0) > limit:
            await _too_large(limit)(scope, receive, send)
            return
        chunks: list[bytes] = []
        size, more = 0, True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client is gone; nobody is left to answer
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                await _too_large(limit)(scope, receive, send)
                return
            chunks.append(chunk)
            more = message.get("more_body", False)
        await self.app(scope, _replay(b"".join(chunks), receive), send)


class SecurityHeadersMiddleware:
    """Forbids framing (clickjacking of owner actions), MIME sniffing and referrer leaks on every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *_SECURITY_HEADER_PAIRS]}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class ReplaceRequest(Model):
    plan_id: str = Field(pattern=ID_PATTERN)  # a plan stored by POST /api/plans


def lan_path(node_id: str) -> PathClass:
    """The LAN transport has one path class for every peer."""
    return "lan"


def read_on_loop(loop: asyncio.AbstractEventLoop | None, read: Callable[[], dict[str, Any]],
                 timeout_s: float = PATH_READ_TIMEOUT_S) -> dict[str, Any]:
    """read() on the thread of `loop`, directly when called there. The capacity service looks paths up from worker
    threads (explain, plan and the replan evaluator run in asyncio.to_thread), and HostService.status() walks dicts the
    loop changes on every join and leave. Raises RuntimeError when the loop is not running, TimeoutError when it does
    not answer within timeout_s."""
    try:
        running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:  # a worker thread has no running loop
        running = None
    if loop is None or running is loop:
        return read()
    if loop.is_closed() or not loop.is_running():
        raise RuntimeError("the gateway event loop is not running")

    async def on_loop() -> dict[str, Any]:
        return read()

    future = asyncio.run_coroutine_threadsafe(on_loop(), loop)
    try:
        return future.result(timeout_s)
    except TimeoutError:
        future.cancel()
        raise


def iroh_path_mode(status: Mapping[str, Any], node_id: str) -> PathClass:
    """The selected Iroh route of a connected peer from HostService.status(): "relay" or "direct". "unknown" when the
    peer is missing, not connected, or its route is not selected yet ("connecting")."""
    for peer in status.get("peers") or []:
        if isinstance(peer, Mapping) and peer.get("node_id") == node_id:
            path = peer.get("path") if peer.get("connected") else None
            mode = path.get("mode") if isinstance(path, Mapping) else None
            if mode == "relay":
                return "relay"
            return "direct" if mode == "direct" else "unknown"
    return "unknown"


def create_app(coord: Coordinator, *, transport: str = "lan", relay_url: str | None = None) -> FastAPI:
    def transport_invite(view: SessionView) -> SessionView:
        peer = app.state.peer_service
        if peer is None or view.invite is None:
            return view
        invite = view.invite.model_copy(update={
            "join_address": peer.ticket,
            "join_command": f"uv run hypercompute join --ticket {peer.ticket} --code {view.invite.code}",
        })
        return view.model_copy(update={"invite": invite})

    def network_invite() -> dict:
        current = coord.sessions.current()
        view = transport_invite(coord.sessions.issue_invite(current.id) if current
                                else coord.sessions.create("Hypercompute mesh"))
        assert view.invite is not None
        invite = view.invite.model_dump(mode="json")
        peer = app.state.peer_service
        if peer is not None:
            invite["ticket"] = peer.ticket
        return invite

    def iroh_path(node_id: str) -> PathClass:
        peer = app.state.peer_service
        if peer is None:
            return "unknown"
        try:
            status = read_on_loop(app.state.loop, peer.status)
        except Exception:  # a route the transport cannot report is "unknown", never a guess
            log.exception("reading the Iroh path of node %s failed", node_id)
            return "unknown"
        return iroh_path_mode(status, node_id)

    def bind_paths() -> None:
        """Let the capacity layer price each peer's hop by its path class (called once the transport is up)."""
        if coord.capacity is not None:
            coord.capacity.path_of = iroh_path if transport == "iroh" else lan_path

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        coord.events.bind_loop()
        app.state.loop = asyncio.get_running_loop()  # the thread that owns the transport's state
        for note in coord.startup_notes:
            coord.events.publish("gateway.startup", note)
        tasks: list[asyncio.Task] = []
        try:
            if transport == "iroh":
                from .agent.join import say
                from .transport.host import HostService

                app.state.peer_service = await HostService.start(
                    coord, config.paths().ensure().coordinator / "iroh-device.key", relay_url=relay_url)
                invite = network_invite()
                say("Iroh ready. Send this one-use invitation to your friend:")
                say(invite["join_command"])
                say(f"Expires: {invite['expires_at']}; renew with: uv run hypercompute invite --port {coord.port}")
            bind_paths()
            tasks = [asyncio.create_task(coord.nodes.monitor()), asyncio.create_task(coord.refresh_local()),
                     asyncio.create_task(coord.cleanup_orphans())]
            if coord.replan is not None and coord.replan.enabled:
                tasks.append(asyncio.create_task(coord.replan.run()))
            yield
        finally:
            for t in tasks:
                t.cancel()
            if coord.replan is not None:
                await coord.replan.aclose()  # also cancels a replacement it started, before the manager shuts down
            if coord.tuner is not None:
                await coord.tuner.shutdown()  # stops its trial deployment through the deployment manager
            await coord.benchmarks.shutdown()  # record their final status while the deployment still exists
            await coord.deployments.shutdown()
            if app.state.peer_service is not None:
                await app.state.peer_service.close()
            await asyncio.to_thread(coord.supervisor.stop_all, config.STOP_GRACE_S)
            await coord.aclose_adapters()
            coord.store.close()

    app = FastAPI(title="Hypercompute", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)  # added last = outermost, so 413 replies carry the headers too
    app.state.coord = coord
    app.state.peer_service = None
    app.state.loop = None

    # ------------------------------------------------------------------ auth guards
    def owner_read(request: Request) -> None:
        client = request.client.host if request.client else ""
        host = (request.headers.get("host") or "").lower()
        hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if client not in LOOPBACK_CLIENTS or hostname not in LOOPBACK_HOSTS:
            raise OwnerAuthError("owner endpoints are only available from this machine (http://127.0.0.1)")

    def owner_write(request: Request, x_hypercompute_owner: Annotated[str | None, Header()] = None) -> None:
        owner_read(request)
        if not x_hypercompute_owner or not hmac.compare_digest(x_hypercompute_owner, coord.owner_token):
            raise OwnerAuthError("missing or wrong owner token; reload the dashboard")

    OwnerRead = Depends(owner_read)
    OwnerWrite = Depends(owner_write)

    # ------------------------------------------------------------------ error mapping
    @app.exception_handler(OwnerAuthError)
    async def _owner(_: Request, exc: OwnerAuthError) -> JSONResponse:
        return _error(403, "owner_only", str(exc))

    @app.exception_handler(JoinRejected)
    async def _join(_: Request, exc: JoinRejected) -> JSONResponse:
        return _error(exc.status, exc.code, str(exc), exc.fix)

    @app.exception_handler(AuthError)
    async def _auth(_: Request, exc: AuthError) -> JSONResponse:
        return _error(exc.status, "node_auth", str(exc))

    @app.exception_handler(DeploymentError)
    async def _dep(_: Request, exc: DeploymentError) -> JSONResponse:
        return _error(exc.status, exc.code, str(exc), exc.fix)

    @app.exception_handler(ModelError)
    async def _model(_: Request, exc: ModelError) -> JSONResponse:
        return _error(400, "model_error", str(exc))

    @app.exception_handler(EngineAdapterError)
    async def _engine_adapter(_: Request, exc: EngineAdapterError) -> JSONResponse:
        return _error(502, exc.code, str(exc), exc.fix)

    @app.exception_handler(EngineError)
    async def _engine(_: Request, exc: EngineError) -> JSONResponse:
        return _error(500, "engine_error", str(exc), exc.fix)

    @app.exception_handler(CapacityUnavailable)
    async def _capacity_unavailable(_: Request, exc: CapacityUnavailable) -> JSONResponse:
        return _error(capacity_status(exc.code, exc.__cause__), exc.code, str(exc))

    # ------------------------------------------------------------------ health / owner bootstrap
    @app.get("/api/health")
    async def health(request: Request) -> dict:
        body = {"status": "ok", "version": __version__, "time": utcnow().isoformat(),
                "engine": {"release": coord.engine_info.release, "rpc_proto": coord.engine_info.rpc_proto}}
        try:
            owner_read(request)
        except OwnerAuthError:
            return body
        return {**body, "engine": coord.engine_info.model_dump(mode="json"),
                "engines": [c.model_dump() for c in ENGINE_ROADMAP]}

    @app.get("/api/owner/bootstrap", dependencies=[OwnerRead])
    async def bootstrap() -> dict:
        current = coord.sessions.current()
        return {"owner_token": coord.owner_token, "version": __version__, "gateway_port": coord.port,
                "lan_addresses": current.lan_addresses if current else coord.sessions.lan_addresses()}

    @app.get("/api/network", dependencies=[OwnerRead])
    async def network_status() -> dict:
        peer = app.state.peer_service
        return peer.status() if peer is not None else {"transport": "lan"}

    @app.post("/api/network/invite", dependencies=[OwnerWrite])
    async def invite_network() -> dict:
        return network_invite()

    # ------------------------------------------------------------------ sessions
    @app.post("/api/sessions", dependencies=[OwnerWrite])
    async def create_session(req: CreateSessionRequest) -> SessionView:
        return transport_invite(coord.sessions.create(req.name))

    @app.get("/api/sessions/current", dependencies=[OwnerRead])
    async def current_session() -> SessionView | None:
        return coord.sessions.current()

    @app.post("/api/sessions/{session_id}/invites", dependencies=[OwnerWrite])
    async def new_invite(session_id: str) -> SessionView:
        return transport_invite(coord.sessions.issue_invite(session_id))

    @app.delete("/api/sessions/{session_id}", dependencies=[OwnerWrite])
    async def close_session(session_id: str) -> SessionView:
        view = coord.sessions.close(session_id)
        await coord.nodes.close_session(session_id)
        return view

    @app.post("/api/sessions/join")
    async def join(req: JoinRequest, request: Request) -> JoinResponse:
        client = request.client.host if request.client else ""
        return await coord.nodes.join(req, client, request.headers.get("host"))

    # ------------------------------------------------------------------ nodes
    @app.get("/api/nodes", dependencies=[OwnerRead])
    async def list_nodes() -> list[NodeView]:
        return coord.nodes.views()

    @app.post("/api/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str, req: HeartbeatRequest,
                        authorization: Annotated[str | None, Header()] = None) -> HeartbeatResponse:
        coord.nodes.authenticate(node_id, authorization)
        return await coord.nodes.heartbeat(node_id, req)

    @app.post("/api/nodes/{node_id}/leave")
    async def leave(node_id: str, request: Request, authorization: Annotated[str | None, Header()] = None,
                    x_hypercompute_owner: Annotated[str | None, Header()] = None) -> dict:
        if authorization:
            coord.nodes.authenticate(node_id, authorization)
            await coord.nodes.leave(node_id, by_owner=False)
            return {"ok": True}
        owner_write(request, x_hypercompute_owner)
        if coord.nodes.get(node_id) is None or node_id == "local":
            raise DeploymentError(404, "not_found", "no such peer")
        view = await coord.nodes.leave(node_id, by_owner=True)
        return {"ok": True, "node": view.model_dump(mode="json")}

    @app.post("/api/nodes/{node_id}/verify", dependencies=[OwnerWrite])
    async def verify(node_id: str) -> NodeView:
        if coord.nodes.get(node_id) is None or node_id == "local":
            raise DeploymentError(404, "not_found", "no such peer")
        return await coord.nodes.verify(node_id)

    # ------------------------------------------------------------------ managed peers (joined with --managed)
    def _managed_peer(node_id: str) -> str:
        if node_id == "local" or coord.nodes.get(node_id) is None:
            raise DeploymentError(404, "not_found", "no such peer")
        return node_id

    def _refused(exc: ManagedRefusal) -> DeploymentError:
        return DeploymentError(exc.status, exc.code, str(exc), exc.fix)

    @app.get("/api/peers", dependencies=[OwnerRead])
    async def list_peers() -> list[dict]:
        """Each peer with what it allowed (--managed), its worker devices, cache bytes and seed progress."""
        return coord.nodes.managed.snapshot(coord.nodes.views(), coord.nodes.busy)

    @app.post("/api/peers/{node_id}/worker", dependencies=[OwnerWrite], status_code=202)
    async def peer_worker(node_id: str, req: PeerWorkerRequest) -> dict:
        """Ask a managed peer to restart its worker with these devices (and threads); it re-verifies afterwards."""
        _managed_peer(node_id)
        try:
            cmd = coord.nodes.managed.request_worker(node_id, list(req.devices), req.threads,
                                                     busy=bool(coord.nodes.busy(node_id)))
        except ManagedRefusal as exc:
            raise _refused(exc) from exc
        return {"command": cmd.model_dump(mode="json")}

    @app.post("/api/peers/{node_id}/seed", dependencies=[OwnerWrite], status_code=202)
    async def peer_seed(node_id: str, req: PeerSeedRequest) -> dict:
        """Hash the peer's layers of a downloaded catalog GGUF here, then send the tensor list to the peer, which
        downloads exactly those byte ranges from Hugging Face into its RPC cache."""
        _managed_peer(node_id)
        coord.models.get(req.model_id)  # an unknown model is a 400 before anything starts
        try:
            if (req.layers is None) == (req.plan_id is None):
                raise ManagedRefusal(422, "layers_or_plan", "give exactly one of layers (e.g. 30-35) or plan_id")
            layers = (parse_layers(req.layers) if req.layers is not None else
                      plan_layers(load_plan(coord.deployments, req.plan_id or ""), node_id))
            hashes = stop_dir_for(config.paths().ensure().coordinator)
            job = coord.nodes.managed.start_seed(
                node_id, req.model_id, layers,
                lambda stop: build_seed(coord.models, req.model_id, layers, cache_dir=hashes, stop_file=stop),
                stop_dir=hashes)
        except ManagedRefusal as exc:
            raise _refused(exc) from exc
        except ValueError as exc:
            raise DeploymentError(422, "invalid_layers", str(exc)) from exc
        return job.view()

    @app.post("/api/peers/{node_id}/seed/cancel", dependencies=[OwnerWrite], status_code=202)
    async def peer_seed_cancel(node_id: str) -> dict:
        _managed_peer(node_id)
        try:
            cmd = coord.nodes.managed.request_cancel(node_id)
        except ManagedRefusal as exc:
            raise _refused(exc) from exc
        return {"command": cmd.model_dump(mode="json")}

    # ------------------------------------------------------------------ models / plans / deployments
    @app.get("/api/models", dependencies=[OwnerRead])
    async def list_models() -> list[ModelView]:
        return await asyncio.to_thread(coord.models.views)

    @app.post("/api/models/{model_id}/download", dependencies=[OwnerWrite], status_code=202)
    async def download_model(model_id: str) -> ModelView:
        coord.models.get(model_id)
        task = asyncio.create_task(coord.models.download(model_id))
        task.add_done_callback(lambda t: t.exception())  # failures are recorded on the model view
        await asyncio.sleep(0)
        return coord.models.view(model_id)

    @app.post("/api/plans", dependencies=[OwnerWrite])
    async def plan(req: PlanRequest) -> Plan:
        return await asyncio.to_thread(coord.deployments.plan, req)

    @app.post("/api/deployments", dependencies=[OwnerWrite], status_code=202)
    async def create_deployment(req: CreateDeploymentRequest,
                                idempotency_key: Annotated[str | None, Header()] = None) -> Deployment:
        return await coord.deployments.create(req, idempotency_key)

    @app.get("/api/deployments", dependencies=[OwnerRead])
    async def list_deployments() -> list[Deployment]:
        return coord.deployments.list()

    @app.get("/api/deployments/{dep_id}", dependencies=[OwnerRead])
    async def get_deployment(dep_id: str) -> Deployment:
        return coord.deployments.get(dep_id)

    @app.delete("/api/deployments/{dep_id}", dependencies=[OwnerWrite])
    async def stop_deployment(dep_id: str) -> Deployment:
        return await coord.deployments.stop(dep_id)

    @app.post("/api/deployments/{dep_id}/replace", dependencies=[OwnerWrite])
    async def replace_deployment(dep_id: Annotated[str, Path(pattern=ID_PATTERN)], req: ReplaceRequest) -> Deployment:
        """Reload the ready deployment onto a stored plan. Waits until the new deployment is ready or has failed (a
        rollback or fallback included); the reply is the deployment that ends up serving, or the failed attempt."""
        plan = coord.deployments.stored_plan(req.plan_id)
        return await coord.deployments.replace(dep_id, plan, f"requested by the owner (plan {plan.id[:8]})")

    # ------------------------------------------------------------------ capacity / replan
    def _capacity() -> Any:
        if coord.capacity is None:
            raise DeploymentError(503, "capacity_unavailable", "the capacity optimizer is not part of this gateway")
        return coord.capacity

    def _replan() -> ReplanController:
        if coord.replan is None:
            raise DeploymentError(503, "replan_unavailable", "the replan controller is not part of this gateway")
        return coord.replan

    @app.get("/api/capacity/pools", dependencies=[OwnerRead])
    async def capacity_pools() -> dict:
        return await asyncio.to_thread(_capacity().pools)

    @app.get("/api/capacity/explain", dependencies=[OwnerRead])
    async def capacity_explain(model_id: Annotated[str, Query(pattern=MODEL_ID_PATTERN)],
                               ctx_size: Annotated[int, Query(ge=256, le=MAX_TOKENS)] = 4096,
                               prompt_tokens: Annotated[int, Query(ge=0, le=MAX_TOKENS)] = 512,
                               output_tokens: Annotated[int, Query(ge=1, le=MAX_TOKENS)] = 256,
                               warm: bool = False,
                               slots: Annotated[int, Query(ge=1, le=MAX_STREAM_SLOTS)] = 2) -> dict:
        service = _capacity()
        if prompt_tokens + output_tokens > ctx_size:
            raise DeploymentError(422, "workload_exceeds_context",
                                  f"prompt_tokens + output_tokens ({prompt_tokens + output_tokens}) exceed ctx_size "
                                  f"({ctx_size})", fix="raise ctx_size or lower the token counts")
        return await asyncio.to_thread(service.explain, model_id, ctx_size=ctx_size, prompt_tokens=prompt_tokens,
                                       output_tokens=output_tokens, warm=warm, slots=slots)

    @app.get("/api/replan/decisions", dependencies=[OwnerRead])
    async def replan_decisions() -> list[ReplanDecision]:
        return _replan().decisions()

    @app.post("/api/replan/trigger", dependencies=[OwnerWrite])
    async def replan_trigger() -> ReplanDecision:
        """Evaluate the ready deployment now; a "replace" decision waits for the replacement to finish."""
        controller = _replan()
        if not controller.enabled:
            raise DeploymentError(409, "replan_disabled", "the replan controller is disabled",
                                  fix="restart the host with HYPERCOMPUTE_REPLAN=1 in its environment")
        return await controller.trigger()

    # ------------------------------------------------------------------ events / runs / benchmarks
    @app.get("/api/events", dependencies=[OwnerRead])
    async def events(request: Request, since: int = 0) -> StreamingResponse:
        last = request.headers.get("last-event-id")
        start = int(last) if last and last.isdigit() else since

        async def gen():  # type: ignore[no-untyped-def]
            async with aclosing(coord.events.stream(start)) as stream:
                async for ev in stream:
                    if ev is None:
                        yield b": keep-alive\n\n"
                    else:
                        yield f"id: {ev.seq}\nevent: hc\ndata: {ev.model_dump_json()}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs", dependencies=[OwnerRead])
    async def runs(limit: int = 50) -> list[RunRecord]:
        return coord.inference.runs(max(1, min(limit, 500)))

    @app.post("/api/benchmarks", dependencies=[OwnerWrite], status_code=202)
    async def benchmark(req: BenchmarkRequest) -> BenchmarkResult:
        return coord.benchmarks.start(req)

    @app.get("/api/benchmarks", dependencies=[OwnerRead])
    async def benchmarks() -> list[BenchmarkResult]:
        return coord.benchmarks.list()

    @app.get("/api/benchmarks/{bench_id}", dependencies=[OwnerRead])
    async def get_benchmark(bench_id: str) -> BenchmarkResult:
        return coord.benchmarks.get(bench_id)

    # ------------------------------------------------------------------ engines, tuning, measured profiles
    @app.get("/api/engines", dependencies=[OwnerRead])
    async def engines() -> dict:
        host = await asyncio.to_thread(registry.engine_status, coord.adapter_ctx) if coord.adapter_ctx else []
        return {"available": coord.deployments.engines(),
                "llama_rpc": coord.engine_info.model_dump(mode="json"),
                "host": [e.model_dump(mode="json") for e in host],
                "nodes": {n.id: [e.model_dump(mode="json") for e in n.inventory.extra_engines]
                          for n in coord.nodes.views()}}

    def _tuner():  # type: ignore[no-untyped-def]
        if coord.tuner is None:
            raise DeploymentError(501, "tuner_unavailable", "the measured tuner is not part of this build")
        return coord.tuner

    @app.post("/api/tune", dependencies=[OwnerWrite], status_code=202)
    async def tune(req: TuneRequest) -> TuneResult:
        return await _tuner().start(req)

    @app.get("/api/tune", dependencies=[OwnerRead])
    async def tunes() -> list[TuneResult]:
        return _tuner().list()

    @app.get("/api/tune/{tune_id}", dependencies=[OwnerRead])
    async def get_tune(tune_id: str) -> TuneResult:
        return _tuner().get(tune_id)

    @app.delete("/api/tune/{tune_id}", dependencies=[OwnerWrite])
    async def cancel_tune(tune_id: str) -> TuneResult:
        return await _tuner().cancel(tune_id)

    @app.get("/api/profiles", dependencies=[OwnerRead])
    async def profiles(model_id: str | None = None) -> list[MeasuredProfile]:
        return coord.profiles.list(model_id) if coord.profiles else []

    # ------------------------------------------------------------------ OpenAI-compatible
    @app.get("/v1/models", dependencies=[OwnerRead])
    async def v1_models() -> dict:
        dep = coord.deployments.active()
        data = []
        if dep is not None and dep.state.value == "ready" and dep.purpose != "tune":  # a tuner trial is not served
            data.append({"id": dep.plan.model_id, "object": "model", "owned_by": "hypercompute",
                         "deployment_id": dep.id})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions", dependencies=[OwnerWrite])
    async def chat(request: Request):  # type: ignore[no-untyped-def]
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "bad_json", "request body must be JSON")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return _error(400, "bad_request", "'messages' must be a non-empty list")
        try:
            images = count_images(body["messages"])  # checks parts only; the body is forwarded as sent
        except ImageInputError as exc:
            return _error(exc.status, exc.code, str(exc), exc.fix)
        dep, handle = coord.deployments.ready()  # 503 before any streaming starts
        if images and not accepts_images(dep, handle):
            return _error(400, "vision_unavailable", f"the ready model {dep.plan.model_id!r} was started without a "
                          "vision projector, so it cannot read images",
                          fix="deploy a model whose catalog entry lists an mmproj, after `uv run hypercompute models "
                              "pull <model>` has fetched it (docs/VISION.md)")
        requested = body.get("model")
        if requested not in (None, "") and requested != dep.plan.model_id:
            return _error(404, "model_not_found",
                          f"model {requested!r} is not loaded; the ready model is {dep.plan.model_id!r}",
                          fix="omit 'model' or use the ready model listed by /v1/models")
        if body.get("stream"):
            return StreamingResponse(coord.inference.chat_stream(body), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            status, payload = await coord.inference.chat_complete(body)
        except ChatInterrupted as exc:  # the same error object an interrupted stream ends with
            return JSONResponse(status_code=exc.status, content=exc.body)
        return JSONResponse(status_code=status, content=payload)

    # ------------------------------------------------------------------ dashboard
    dist = config.frontend_dist()
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    else:
        @app.get("/", include_in_schema=False)
        async def fallback() -> HTMLResponse:
            return HTMLResponse(FALLBACK_PAGE)

    return app
