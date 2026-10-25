"""Guest peer security and lifecycle tests using fakes, without sockets or Iroh services."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from hypercompute.transport import client as client_module
from hypercompute.transport.client import PeerClient


class FakeStream:
    def __init__(self, response=None, *, pending=False):
        self.response = response if response is not None else {"status": 200, "body": {}}
        self.pending = pending
        self.read_started = asyncio.Event()
        self.sent = []
        self.finished = False
        self.aborted = False

    async def receive_json(self):
        self.read_started.set()
        if self.pending:
            await asyncio.Future()
        return self.response

    async def send_json(self, message):
        self.sent.append(message)

    async def finish(self):
        self.finished = True

    async def abort(self):
        self.aborted = True


class FakeConnection:
    def __init__(self, stream=None):
        self.stream = stream or FakeStream()
        self.closed = False
        self.remote_id = "ticket-authenticated-host"

    async def open_bi(self):
        return self.stream

    def close(self, **kwargs):
        self.closed = True


class FakeWriter:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


def make_client(stream=None):
    """Exercise async internals on one test loop, bypassing the production thread/endpoint."""
    client = PeerClient.__new__(PeerClient)
    client._node_id = "contributor-node"
    client._worker_port = 50052
    client._rpc_allowed = False
    client._rpc_tasks = set()
    client._tasks = set()
    client._connection = FakeConnection(stream)
    client._connect = AsyncMock(return_value=client._connection)
    client._closed = False
    return client


def test_cancelled_join_revokes_existing_rpc_and_closes_transport():
    async def scenario():
        stream = FakeStream(pending=True)
        client = make_client(stream)
        connection = client._connection
        rpc_cancelled = asyncio.Event()

        async def active_rpc():
            try:
                await asyncio.Future()
            finally:
                rpc_cancelled.set()

        rpc_task = asyncio.create_task(active_rpc())
        client._rpc_tasks.add(rpc_task)
        request = asyncio.create_task(client._request("join", {"node_id": client._node_id}, None, None))
        await asyncio.wait_for(stream.read_started.wait(), 1)
        assert client._rpc_allowed
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert not client._rpc_allowed
        assert rpc_cancelled.is_set()
        assert rpc_task.cancelled()
        assert connection.closed
        assert client._connection is None
        assert stream.aborted

    asyncio.run(scenario())


@pytest.mark.parametrize("message", [
    {"op": "rpc", "node_id": "contributor-node", "host": "192.0.2.1", "port": 22},
    {"op": "rpc", "node_id": "contributor-node", "url": "http://127.0.0.1:8000/api/bootstrap"},
    {"op": "rpc", "node_id": "different-node"},
    {"op": "owner", "node_id": "contributor-node"},
    {},
])
def test_rpc_rejects_noncanonical_requests_before_opening_any_tcp_socket(monkeypatch, message):
    async def scenario():
        client = make_client()
        client._rpc_allowed = True
        stream = FakeStream(message)
        open_connection = AsyncMock()
        monkeypatch.setattr(client_module.asyncio, "open_connection", open_connection)
        await client._serve_rpc(client._connection, stream)
        open_connection.assert_not_called()
        assert stream.sent == [{"status": 403, "body": {"detail": "RPC access denied"}}]
        assert stream.aborted

    asyncio.run(scenario())


@pytest.mark.parametrize("authorized", [False, True])
def test_rpc_requires_active_sharing_and_the_current_host_connection(monkeypatch, authorized):
    async def scenario():
        client = make_client()
        client._rpc_allowed = authorized
        connection = FakeConnection() if authorized else client._connection
        stream = FakeStream({"op": "rpc", "node_id": client._node_id})
        open_connection = AsyncMock()
        monkeypatch.setattr(client_module.asyncio, "open_connection", open_connection)
        await client._serve_rpc(connection, stream)
        open_connection.assert_not_called()
        assert stream.sent[0]["status"] == 403
        assert stream.aborted

    asyncio.run(scenario())


def test_pause_immediately_terminates_active_bridge_and_denies_new_rpc(monkeypatch):
    async def scenario():
        client = make_client()
        client._rpc_allowed = True
        stream = FakeStream({"op": "rpc", "node_id": client._node_id})
        writer = FakeWriter()
        open_connection = AsyncMock(return_value=(object(), writer))
        monkeypatch.setattr(client_module.asyncio, "open_connection", open_connection)
        bridge_started = asyncio.Event()
        bridge_cancelled = asyncio.Event()

        async def bridge(*args):
            bridge_started.set()
            try:
                await asyncio.Future()
            finally:
                bridge_cancelled.set()

        monkeypatch.setattr(client_module, "bridge_tcp", bridge)
        task = asyncio.create_task(client._serve_rpc(client._connection, stream))
        client._rpc_tasks.add(task)
        await asyncio.wait_for(bridge_started.wait(), 1)
        open_connection.assert_awaited_once_with("127.0.0.1", 50052)
        await client._set_worker_active(False)
        assert not client._rpc_allowed
        assert bridge_cancelled.is_set()
        assert task.done()
        assert stream.aborted and writer.closed
        subsequent = FakeStream({"op": "rpc", "node_id": client._node_id})
        await client._serve_rpc(client._connection, subsequent)
        assert subsequent.sent[0]["status"] == 403
        assert open_connection.await_count == 1

    asyncio.run(scenario())


def test_unsolicited_connection_is_closed_without_replacing_trusted_host():
    async def scenario():
        client = make_client()
        trusted = client._connection
        stranger = FakeConnection()
        await client._reject_connection(stranger)
        assert stranger.closed
        assert client._connection is trusted
        assert not trusted.closed

    asyncio.run(scenario())


def test_pause_completes_when_local_tcp_close_never_finishes(monkeypatch):
    async def scenario():
        client = make_client()
        client._rpc_allowed = True
        stream = FakeStream({"op": "rpc", "node_id": client._node_id})
        bridge_started = asyncio.Event()

        class StalledWriter(FakeWriter):
            async def wait_closed(self):
                await asyncio.Future()

        writer = StalledWriter()
        monkeypatch.setattr(client_module.asyncio, "open_connection", AsyncMock(return_value=(object(), writer)))

        async def bridge(*args):
            bridge_started.set()
            await asyncio.Future()

        monkeypatch.setattr(client_module, "bridge_tcp", bridge)
        task = asyncio.create_task(client._serve_rpc(client._connection, stream))
        client._rpc_tasks.add(task)
        await asyncio.wait_for(bridge_started.wait(), 1)
        # The caller's local control request must not hang behind a broken worker socket.
        await asyncio.wait_for(client._set_worker_active(False), 3)
        assert task.done()
        assert stream.aborted and writer.closed
        assert not client._rpc_allowed

    asyncio.run(scenario())


@pytest.mark.parametrize("response", [
    {"status": True, "body": {}},
    {"status": "200", "body": {}},
    {"status": 99, "body": {}},
    {"status": 600, "body": {}},
    {"status": 200, "body": []},
    {"status": 200},
    [],
])
def test_malformed_join_response_revokes_access_and_is_not_retried(response):
    async def scenario():
        stream = FakeStream(response)
        client = make_client(stream)
        connection = client._connection
        with pytest.raises(httpx.ConnectError, match="Iroh join failed"):
            await client._request("join", {"node_id": client._node_id}, None, None)
        assert not client._rpc_allowed
        assert connection.closed and stream.aborted
        assert client._connection is None
        client._connect.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize(("op", "status"), [("join", 409), ("heartbeat", 401), ("heartbeat", 403), ("leave", 500)])
def test_rejection_or_leave_revokes_rpc(op, status):
    async def scenario():
        stream = FakeStream({"status": status, "body": {"detail": "refused"}})
        client = make_client(stream)
        client._rpc_allowed = True
        response = await client._request(op, {"node_id": client._node_id}, client._node_id, "credential")
        assert response.status_code == status
        assert not client._rpc_allowed
        assert stream.aborted

    asyncio.run(scenario())


def test_worker_cannot_be_retargeted_while_sharing():
    async def scenario():
        client = make_client()
        client._rpc_allowed = True
        with pytest.raises(RuntimeError, match="Cannot replace"):
            await client._configure_worker("other-node", 8000)
        assert client._node_id == "contributor-node"
        assert client._worker_port == 50052

    asyncio.run(scenario())


def test_constructor_failure_closes_endpoint_and_background_loop(monkeypatch, tmp_path):
    endpoints = []
    clients = []

    class BrokenEndpoint:
        def __init__(self, *args, **kwargs):
            self.closed = False
            endpoints.append(self)

        async def start(self, handler):
            raise ValueError("invalid ticket setup")

        async def close(self):
            self.closed = True

    class RecordingClient(PeerClient):
        def __new__(cls, *args, **kwargs):
            instance = super().__new__(cls)
            clients.append(instance)
            return instance

    monkeypatch.setattr(client_module, "IrohEndpoint", BrokenEndpoint)
    with pytest.raises(ValueError, match="invalid ticket setup"):
        RecordingClient("unused-ticket", tmp_path / "unused.key")
    assert endpoints[0].closed
    assert clients[0]._closed
    assert not clients[0]._thread.is_alive()
    assert clients[0]._loop.is_closed()
    clients[0].close()
