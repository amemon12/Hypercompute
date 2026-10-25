"""Authenticated peer control and fixed loopback RPC bridges over Iroh.

This service never forwards HTTP to the owner gateway. A public endpoint ticket
only locates the host; a one-use invitation and authenticated device key admit a
node. Subsequent control operations require both that key and its node credential.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .. import __version__
from ..schemas import HeartbeatRequest, JoinRequest, NodeState, WorkerEndpoint, WorkerState
from ..services.nodes import AuthError
from ..services.sessions import JoinRejected, RateLimiter
from .iroh import BridgeStats, IrohConnection, IrohEndpoint, IrohStream, binding_speedups, bridge_tcp

if TYPE_CHECKING:
    from ..coordinator import Coordinator

log = logging.getLogger(__name__)
MAX_PEERS = 16
MAX_CONNECTIONS = 32
MAX_CONTROLS_PER_CONNECTION = 4
MAX_RPC_PER_NODE = 2
FIRST_STREAM_TIMEOUT = 15.0
ADMISSION_TIMEOUT = 90.0
CONTROL_IDLE_TIMEOUT = 45.0


@dataclass
class _Connection:
    connection: IrohConnection
    retired: bool = False
    tasks: set[asyncio.Task] = field(default_factory=set)
    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(120))


@dataclass
class _Peer:
    node_id: str
    peer_id: str
    connection: IrohConnection | None
    server: asyncio.AbstractServer | None = None
    port: int = 0
    rpc_tasks: set[asyncio.Task] = field(default_factory=set)
    # Cumulative over every RPC bridge of this peer, so a run can be priced in bytes and chunks per token.
    rpc_stats: BridgeStats = field(default_factory=BridgeStats)
    rpc_streams: int = 0

    def rpc_status(self) -> dict[str, int]:
        return {**self.rpc_stats.as_dict(), "streams": self.rpc_streams, "active": len(self.rpc_tasks)}


class HostService:
    def __init__(self, coord: Coordinator, endpoint: IrohEndpoint) -> None:
        self.coord = coord
        self.endpoint = endpoint
        self._connections: dict[int, _Connection] = {}
        self._peers: dict[str, _Peer] = {}
        self._admission = asyncio.Lock()
        self._join_limit = RateLimiter(10)
        self._closed = False
        if coord.nodes.on_transport_revoke is not None:
            raise RuntimeError("A peer transport is already attached to this coordinator")
        coord.nodes.on_transport_revoke = self._revoke

    @classmethod
    async def start(cls, coord: Coordinator, keypath: Path, relay_url: str | None = None,
                    relay_only: bool = False) -> HostService:
        endpoint = IrohEndpoint(Path(keypath), relay_url=relay_url, relay_only=relay_only,
                                max_connections=MAX_CONNECTIONS)
        service = cls(coord, endpoint)
        try:
            await endpoint.start(service._handle_connection)
        except BaseException:
            await service.close()
            raise
        return service

    @property
    def ticket(self) -> str:
        return self.endpoint.address

    def status(self) -> dict[str, Any]:
        return {"transport": "iroh", "endpoint_id": self.endpoint.endpoint_id,
                "listening": not self._closed, "connections": len(self._connections),
                "binding_speedups": binding_speedups(),
                "peers": [{"node_id": p.node_id, "peer_id": p.peer_id, "connected": p.connection is not None,
                           "path": p.connection.path_info() if p.connection else None, "rpc": p.rpc_status()}
                          for p in self._peers.values()]}

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for peer in list(self._peers.values()):
            await self._revoke(peer.node_id, True)
            await self.coord.nodes.transport_disconnected(peer.node_id, peer.peer_id)
        for state in list(self._connections.values()):
            state.retired = True
            state.connection.close(reason="host transport stopped")
        await self.endpoint.close()
        if self.coord.nodes.on_transport_revoke == self._revoke:
            self.coord.nodes.on_transport_revoke = None

    async def _handle_connection(self, connection: IrohConnection) -> None:
        if self._closed or len(self._connections) >= MAX_CONNECTIONS:
            connection.close(reason="peer capacity reached")
            return
        state = _Connection(connection)
        self._connections[id(connection)] = state
        admission_deadline = asyncio.get_running_loop().time() + ADMISSION_TIMEOUT
        try:
            while not self._closed and not state.retired:
                authenticated = any(p.connection is connection for p in self._peers.values())
                timeout = CONTROL_IDLE_TIMEOUT if authenticated else FIRST_STREAM_TIMEOUT
                if not authenticated:
                    # Health polling must not let an unpaired peer occupy a slot indefinitely.
                    remaining = admission_deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    timeout = min(timeout, remaining)
                stream = await asyncio.wait_for(connection.accept_bi(), timeout)
                if len(state.tasks) >= MAX_CONTROLS_PER_CONNECTION or not state.limiter.allow("control"):
                    await stream.abort()
                    continue
                task = asyncio.create_task(self._serve_control(state, stream))
                state.tasks.add(task)
                task.add_done_callback(state.tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Disconnection and malformed remote streams do not log request bodies or credentials.
            log.debug("Peer control disconnected (%s)", type(exc).__name__)
        finally:
            state.retired = True
            connection.close(reason="peer control disconnected")
            for task in list(state.tasks):
                task.cancel()
            await asyncio.gather(*state.tasks, return_exceptions=True)
            self._connections.pop(id(connection), None)
            for peer in list(self._peers.values()):
                if peer.connection is connection:
                    peer.connection = None
                    await self.coord.nodes.transport_disconnected(peer.node_id, peer.peer_id)

    async def _serve_control(self, state: _Connection, stream: IrohStream) -> None:
        try:
            request = await stream.receive_json()
            response = await self._dispatch(state.connection, request)
            await stream.send_json(response)
            await stream.finish()
        except asyncio.CancelledError:
            await stream.abort()
            raise
        except Exception:
            await stream.abort()

    async def _dispatch(self, connection: IrohConnection, request: dict[str, Any]) -> dict[str, Any]:
        try:
            if not isinstance(request, dict) or set(request) - {"op", "body", "node_id", "credential"}:
                raise JoinRejected(400, "invalid_request", "unsupported peer control fields")
            body = request.get("body", {})
            if not isinstance(body, dict):
                raise JoinRejected(400, "invalid_request", "peer control body must be an object")
            state = self._connections.get(id(connection))
            if self._closed or state is None or state.retired:
                raise AuthError(403, "peer connection is no longer active")
            op = request.get("op")
            if op == "health":
                engine = self.coord.engine_info
                return {"status": 200, "body": {"status": "ok", "version": __version__,
                        "engine": {"release": engine.release, "rpc_proto": engine.rpc_proto}}}
            if op not in {"join", "heartbeat", "leave"}:
                raise JoinRejected(404, "unsupported_operation", "peer operation is not available")
            async with self._admission:
                if state.retired or self._closed:
                    raise AuthError(403, "peer connection is no longer active")
                if op == "join":
                    result = await self._join(connection, body)
                else:
                    node_id, credential = request.get("node_id"), request.get("credential")
                    if not isinstance(node_id, str) or not isinstance(credential, str) or len(credential) > 512:
                        raise AuthError(401, "missing node identity or credential")
                    self.coord.nodes.authenticate(node_id, f"Bearer {credential}",
                                                  transport_peer_id=connection.remote_id)
                    if op == "heartbeat":
                        heartbeat = HeartbeatRequest.model_validate(body)
                        try:
                            await self._bind_peer(node_id, connection)
                            # Binding may yield to owner removal/session closure or service shutdown.
                            self.coord.nodes.authenticate(node_id, f"Bearer {credential}",
                                                          transport_peer_id=connection.remote_id)
                            if state.retired or self._closed:
                                raise AuthError(403, "peer connection is no longer active")
                        except BaseException:
                            await self._revoke(node_id, True)
                            raise
                        result = await self.coord.nodes.heartbeat(node_id, heartbeat,
                                                                 transport_peer_id=connection.remote_id)
                    else:
                        await self.coord.nodes.leave(node_id, by_owner=False)
                        result = {"ok": True}
                return {"status": 200, "body": result.model_dump(mode="json")
                        if hasattr(result, "model_dump") else result}
        except JoinRejected as exc:
            return {"status": exc.status, "body": {"error": exc.code, "code": exc.code,
                    "detail": str(exc), "fix": exc.fix}}
        except AuthError as exc:
            return {"status": exc.status, "body": {"error": "peer_auth_failed", "detail": str(exc)}}
        except (ValidationError, ValueError, TypeError):
            return {"status": 400, "body": {"error": "invalid_request", "detail": "invalid peer request"}}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Peer operation failed (%s)", type(exc).__name__)
            return {"status": 500, "body": {"error": "peer_operation_failed", "detail": "peer operation failed"}}

    async def _join(self, connection: IrohConnection, body: dict[str, Any]) -> Any:
        if not self._join_limit.allow("admission"):
            raise JoinRejected(429, "rate_limited", "too many peer admission attempts; retry later")
        req = JoinRequest.model_validate(body)
        # No endpoint supplied by a peer is ever used as a TCP destination or listener.
        req = req.model_copy(update={"worker": WorkerEndpoint(host="127.0.0.1", port=1)})
        if req.node_id not in self._peers and len(self._peers) >= MAX_PEERS:
            raise JoinRejected(503, "peer_capacity", "host peer capacity reached")
        allocated = False

        async def allocate() -> WorkerEndpoint:
            nonlocal allocated
            allocated = True
            return await self._bind_peer(req.node_id, connection, joining=True)

        try:
            return await self.coord.nodes.join(req, "127.0.0.1", None,
                                               transport_peer_id=connection.remote_id, transport_worker=allocate)
        except BaseException:
            if allocated:
                await self._revoke(req.node_id, True)
            raise

    async def _bind_peer(self, node_id: str, connection: IrohConnection, *,
                         joining: bool = False) -> WorkerEndpoint:
        peer = self._peers.get(node_id)
        if peer is not None and peer.peer_id != connection.remote_id:
            raise AuthError(403, "bridge belongs to another authenticated peer")
        if peer is None:
            if len(self._peers) >= MAX_PEERS:
                raise JoinRejected(503, "peer_capacity", "host peer capacity reached")
            peer = _Peer(node_id, connection.remote_id, connection)
            server = await asyncio.start_server(lambda reader, writer: self._serve_rpc(peer, reader, writer),
                                                "127.0.0.1", 0, limit=64 * 1024)
            peer.server = server
            peer.port = server.sockets[0].getsockname()[1]
            self._peers[node_id] = peer
        elif peer.connection is not connection or joining:
            old = peer.connection
            # Revoke old streams before assigning the replacement authenticated connection.
            await self._stop_rpc(peer)
            if old is not None and old is not connection:
                await self.coord.nodes.transport_disconnected(node_id, peer.peer_id)
                state = self._connections.get(id(old))
                if state:
                    state.retired = True
                old.close(reason="authenticated peer reconnected")
            peer.connection = connection
        endpoint = WorkerEndpoint(host="127.0.0.1", port=peer.port)
        if not joining:
            self.coord.nodes.attach_transport_worker(node_id, connection.remote_id, endpoint)
        return endpoint

    async def _serve_rpc(self, peer: _Peer, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        stream = None
        admitted = False
        try:
            view = self.coord.nodes.get(peer.node_id)
            connection = peer.connection
            state = self._connections.get(id(connection)) if connection else None
            session = self.coord.sessions.get(view.session_id) if view and view.session_id else None
            if (self._closed or self._peers.get(peer.node_id) is not peer or state is None or state.retired or
                    view is None or view.transport_peer_id != peer.peer_id or
                    view.state not in {NodeState.verifying, NodeState.eligible, NodeState.ineligible} or
                    view.worker_state != WorkerState.running or not session or session["status"] != "active" or
                    len(peer.rpc_tasks) >= MAX_RPC_PER_NODE):
                return
            peer.rpc_tasks.add(task)
            admitted = True
            stream = await connection.open_bi()
            await stream.send_json({"op": "rpc", "node_id": peer.node_id})
            response = await stream.receive_json()
            if response.get("status") != 200:
                return
            peer.rpc_streams += 1
            await bridge_tcp(stream, reader, writer, peer.rpc_stats)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("Peer RPC bridge closed (%s)", type(exc).__name__)
        finally:
            if admitted:
                peer.rpc_tasks.discard(task)
            if stream is not None:
                await stream.abort()
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 2.0)
            except (TimeoutError, OSError):
                writer.transport.abort()

    async def _stop_rpc(self, peer: _Peer) -> None:
        tasks = list(peer.rpc_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _revoke(self, node_id: str, permanent: bool) -> None:
        peer = self._peers.get(node_id)
        if peer is None:
            return
        if permanent:
            self._peers.pop(node_id, None)
            if peer.server is not None:
                peer.server.close()
        await self._stop_rpc(peer)
        if permanent and peer.server is not None:
            # Python 3.12 also waits for accepted clients; cancel bridges before waiting.
            try:
                await asyncio.wait_for(peer.server.wait_closed(), 2.0)
            except TimeoutError:
                log.debug("Peer listener close timed out")
