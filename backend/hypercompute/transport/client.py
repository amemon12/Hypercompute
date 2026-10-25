"""Guest-side authenticated Iroh control channel and fixed localhost RPC bridge.

The transport owns a background asyncio loop because the existing sharing agent has
both synchronous startup/cleanup and an asynchronous heartbeat loop. No HTTP proxy
or arbitrary destination forwarding is exposed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import httpx

from .iroh import BridgeStats, IrohConnection, IrohEndpoint, IrohStream, binding_speedups, bridge_tcp

T = TypeVar("T")
MAX_RPC_STREAMS = 64
CONTROL_TIMEOUT = 30.0
log = logging.getLogger(__name__)


class PeerClient:
    """Connect to the host named by a ticket and serve only our own RPC worker."""

    # Cumulative over every RPC bridge this client serves (diagnostics only); created on first use.
    _rpc_stats: BridgeStats | None = None
    _rpc_streams: int = 0

    def _bridge_stats(self) -> BridgeStats:
        if self._rpc_stats is None:
            self._rpc_stats = BridgeStats()
        return self._rpc_stats

    def __init__(self, ticket: str, key_path: Path, *, relay_url: str | None = None,
                 relays_enabled: bool = True) -> None:
        self.ticket = ticket
        self.key_path = key_path
        self.relay_url = relay_url
        self.relays_enabled = relays_enabled
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="hypercompute-iroh", daemon=True)
        self._thread.start()
        self._endpoint: IrohEndpoint | None = None
        self._connection: IrohConnection | None = None
        self._connect_lock: asyncio.Lock | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._rpc_tasks: set[asyncio.Task[Any]] = set()
        self._node_id: str | None = None
        self._worker_port: int | None = None
        self._rpc_allowed = False
        self._closed = False
        try:
            self._run(self._start())
        except BaseException:
            self.close()
            raise

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def _run(self, coroutine: Coroutine[Any, Any, T], timeout: float = CONTROL_TIMEOUT + 5) -> T:
        if self._closed or not self._thread.is_alive():
            coroutine.close()
            raise httpx.ConnectError("Iroh connection is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise httpx.ConnectTimeout("Iroh request timed out") from exc

    async def _start(self) -> None:
        self._connect_lock = asyncio.Lock()
        self._endpoint = IrohEndpoint(self.key_path, relay_url=self.relay_url,
                                      relays_enabled=self.relays_enabled)
        await self._endpoint.start(self._reject_connection)
        await self._connect()

    async def _reject_connection(self, connection: IrohConnection) -> None:
        # The guest only accepts streams on the connection it dialled to the ticket's host.
        connection.close()

    async def _connect(self) -> IrohConnection:
        assert self._connect_lock is not None and self._endpoint is not None
        async with self._connect_lock:
            if self._connection is None:
                self._connection = await self._endpoint.connect(self.ticket)
                task = asyncio.create_task(self._accept_rpc(self._connection))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            return self._connection

    def configure_worker(self, node_id: str, port: int) -> None:
        """Select the local worker before pairing; a peer can never change this destination."""
        if not node_id or not 1 <= port <= 65535:
            raise ValueError("A node identity and valid worker port are required")
        self._run(self._configure_worker(node_id, port))

    async def _configure_worker(self, node_id: str, port: int) -> None:
        if self._rpc_allowed:
            raise RuntimeError("Cannot replace a worker while sharing")
        self._node_id, self._worker_port = node_id, port

    def set_worker_active(self, active: bool) -> None:
        """Immediately stop remote access on pause/leave, and restore it after resume."""
        self._run(self._set_worker_active(active))

    async def _set_worker_active(self, active: bool) -> None:
        if active:
            if self._node_id is None or self._worker_port is None:
                raise ValueError("No configured worker")
            self._rpc_allowed = True
        else:
            await self._disable_rpc()

    async def _accept_rpc(self, connection: IrohConnection) -> None:
        try:
            while True:
                stream = await connection.accept_bi()
                if len(self._rpc_tasks) >= MAX_RPC_STREAMS:
                    await stream.abort()
                    continue
                task = asyncio.create_task(self._serve_rpc(connection, stream))
                self._rpc_tasks.add(task)
                task.add_done_callback(self._rpc_tasks.discard)
        except (Exception, asyncio.CancelledError) as exc:
            log.debug("Iroh peer connection ended: %s", type(exc).__name__)
        finally:
            if self._connection is connection:
                self._connection = None

    async def _serve_rpc(self, connection: IrohConnection, stream: IrohStream) -> None:
        try:
            message = await asyncio.wait_for(stream.receive_json(), 10)
            if (connection is not self._connection or not self._rpc_allowed or self._worker_port is None
                    or message != {"op": "rpc", "node_id": self._node_id}):
                await stream.send_json({"status": 403, "body": {"detail": "RPC access denied"}})
                return
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", self._worker_port), 5)
            try:
                await stream.send_json({"status": 200, "body": {}})
                self._rpc_streams += 1
                await bridge_tcp(stream, reader, writer, self._bridge_stats())
            finally:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=2)
                except (TimeoutError, OSError):
                    writer.transport.abort()
        except (Exception, asyncio.CancelledError) as exc:
            log.debug("Iroh worker stream ended: %s", type(exc).__name__)
        finally:
            await stream.abort()

    async def _request(self, op: str, body: dict | None, node_id: str | None,
                       credential: str | None) -> httpx.Response:
        stream: IrohStream | None = None
        connection: IrohConnection | None = None
        message: dict[str, Any] = {"op": op}
        if body is not None:
            message["body"] = body
        if node_id is not None:
            message["node_id"] = node_id
        if credential is not None:
            message["credential"] = credential
        if op == "join":
            if not body or body.get("node_id") != self._node_id or self._worker_port is None:
                raise ValueError("Configure the local worker before joining")
            # The host's asynchronous probe may arrive just before its join response.
            self._rpc_allowed = True
        try:
            async with asyncio.timeout(CONTROL_TIMEOUT):
                connection = await self._connect()
                stream = await connection.open_bi()
                await stream.send_json(message)
                await stream.finish()
                reply = await stream.receive_json()
            status, payload = reply.get("status"), reply.get("body")
            if type(status) is not int or not 100 <= status <= 599 or not isinstance(payload, dict):
                raise ValueError("Invalid peer response")
            if status in (401, 403) or (op == "join" and status != 200) or op == "leave":
                await self._disable_rpc()
            return httpx.Response(status, json=payload,
                                  request=httpx.Request("POST", f"http://iroh.invalid/{op}"))
        except asyncio.CancelledError:
            if op == "join":
                await self._disable_rpc()
            if connection is not None and self._connection is connection:
                connection.close()
                self._connection = None
            raise
        except Exception as exc:
            if op == "join":
                await self._disable_rpc()
            # Never retry join automatically: the one-use invite may already be consumed.
            if connection is not None and self._connection is connection:
                connection.close()
                self._connection = None
            if isinstance(exc, httpx.HTTPError):
                raise
            raise httpx.ConnectError(f"Iroh {op} failed ({type(exc).__name__})") from exc
        finally:
            if stream is not None:
                await stream.abort()

    async def _disable_rpc(self) -> None:
        self._rpc_allowed = False
        tasks = list(self._rpc_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def request(self, op: str, *, body: dict | None = None, node_id: str | None = None,
                credential: str | None = None) -> httpx.Response:
        """Issue one bounded authenticated control request from synchronous agent code."""
        return self._run(self._request(op, body, node_id, credential))

    async def arequest(self, op: str, *, body: dict | None = None, node_id: str | None = None,
                       credential: str | None = None) -> httpx.Response:
        """Issue a request without blocking the agent's heartbeat/control event loop."""
        if self._closed:
            raise httpx.ConnectError("Iroh connection is closed")
        future = asyncio.run_coroutine_threadsafe(self._request(op, body, node_id, credential), self._loop)
        return await asyncio.wrap_future(future)

    def status(self) -> dict:
        """Return non-secret connection diagnostics."""
        return self._run(self._status())

    async def _status(self) -> dict:
        connection = self._connection
        return {"transport": "iroh", "connected": connection is not None,
                "peer_id": connection.remote_id if connection else None,
                "path": connection.path_info() if connection else None,
                "binding_speedups": binding_speedups(),
                "rpc": {**self._bridge_stats().as_dict(), "streams": self._rpc_streams,
                        "active": len(self._rpc_tasks)}}

    async def _close(self) -> None:
        await self._disable_rpc()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self._endpoint is not None:
            await self._endpoint.close()

    def close(self) -> None:
        """Close all streams and the private event loop; safe to call repeatedly."""
        if self._closed:
            return
        try:
            self._run(self._close(), timeout=10)
        except (httpx.HTTPError, RuntimeError) as exc:
            log.debug("Iroh cleanup ended: %s", type(exc).__name__)
        finally:
            self._closed = True
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
