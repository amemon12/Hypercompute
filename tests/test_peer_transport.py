"""Peer admission and revocation tests, with no public relay or engine process."""

import asyncio
from types import SimpleNamespace

import pytest
from factories import engine_info
from hypercompute.schemas import ExtraEngineInfo, HeartbeatRequest, NodeState, WorkerEndpoint, WorkerState, utcnow
from hypercompute.services.nodes import AuthError
from hypercompute.services.sessions import JoinRejected
from hypercompute.transport.host import HostService, _Connection
from test_sessions_and_join import join_request, registry, sessions


class FakeEndpoint:
    address = "test-ticket"
    endpoint_id = "host-public-key"

    async def close(self):
        return None


class EchoStream:
    def __init__(self):
        self.header = None
        self.data = asyncio.Queue()
        self.closed = asyncio.Event()

    async def send_json(self, data):
        self.header = data

    async def receive_json(self):
        return {"status": 200, "body": {}}

    async def read(self, _size):
        return await self.data.get()

    async def write(self, data):
        await self.data.put(data)

    async def finish(self):
        await self.data.put(b"")

    async def abort(self):
        self.closed.set()
        await self.data.put(b"")


class FakeConnection:
    def __init__(self, remote_id="peer-a"):
        self.remote_id = remote_id
        self.closed = False
        self.streams = []

    def close(self, **_kwargs):
        self.closed = True

    def path_info(self):
        return {"kind": "test"}

    async def open_bi(self):
        stream = EchoStream()
        self.streams.append(stream)
        return stream

    async def accept_bi(self):
        await asyncio.Future()


@pytest.fixture
async def service(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    host = HostService(SimpleNamespace(nodes=reg, sessions=svc, engine_info=engine_info()), FakeEndpoint())
    yield host
    await host.close()
    tasks = list(reg._verify_tasks.values()) + list(reg._lost_tasks.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def connect(service, peer_id="peer-a"):
    conn = FakeConnection(peer_id)
    service._connections[id(conn)] = _Connection(conn)
    return conn


async def join(service, conn, node_id="a" * 32, body=None):
    code = service.coord.sessions.create("test").invite.code
    req = join_request(code, node_id=node_id).model_dump(mode="json") if body is None else body
    result = await service._dispatch(conn, {"op": "join", "body": req})
    assert result["status"] == 200, result
    await asyncio.sleep(0)
    return result["body"]


async def heartbeat(service, conn, joined, worker_state=WorkerState.running, **extra):
    body = HeartbeatRequest(worker_state=worker_state, sent_at=utcnow(), **extra).model_dump(mode="json")
    return await service._dispatch(conn, {"op": "heartbeat", "node_id": joined["node_id"],
                                          "credential": joined["credential"], "body": body})


@pytest.mark.anyio
async def test_peer_surface_is_narrow_and_has_no_owner_bootstrap(service):
    conn = connect(service)
    response = await service._dispatch(conn, {"op": "health"})
    assert response["status"] == 200
    assert set(response["body"]) == {"status", "version", "engine"}
    for op in ("bootstrap", "owner", "http", "rpc", "connect", "/api/session"):
        assert (await service._dispatch(conn, {"op": op}))["status"] == 404
    assert (await service._dispatch(conn, {"op": "health", "target": "127.0.0.1:8000"}))["status"] == 400
    assert (await service._dispatch(conn, {"op": "health", "body": []}))["status"] == 400
    assert not service._peers


@pytest.mark.anyio
async def test_bad_invitation_never_allocates_proxy_and_rotation_is_rate_limited(service):
    for index in range(11):
        conn = connect(service, f"rotated-key-{index}")
        body = join_request("ZZZZZ-ZZZZZ").model_dump(mode="json")
        response = await service._dispatch(conn, {"op": "join", "body": body})
        assert response["status"] == (429 if index == 10 else 403)
        assert not service._peers


@pytest.mark.anyio
async def test_join_ignores_target_strips_other_engines_and_rejects_invite_reuse(service):
    conn = connect(service)
    code = service.coord.sessions.create("test").invite.code
    req = join_request(code, worker_ip="8.8.8.8")
    exo = ExtraEngineInfo(engine="exo", installed=True, status="running", endpoint="http://127.0.0.1:8000")
    req.inventory.extra_engines = [exo]
    joined = await join(service, conn, body=req.model_dump(mode="json"))
    node = service.coord.nodes.get(joined["node_id"])
    assert node.transport_peer_id == conn.remote_id
    assert node.worker.host == "127.0.0.1" and node.worker.port != req.worker.port
    assert not node.inventory.extra_engines
    assert (await heartbeat(service, conn, joined, extra_engines=[exo]))["status"] == 200
    assert not service.coord.nodes.get(node.id).inventory.extra_engines
    req.node_id = "b" * 32
    assert (await service._dispatch(conn, {"op": "join", "body": req.model_dump(mode="json")}))["status"] == 403
    assert set(service._peers) == {node.id}


@pytest.mark.anyio
async def test_bearer_requires_same_peer_and_cannot_be_used_over_lan(service):
    conn = connect(service)
    joined = await join(service, conn)
    intruder = connect(service, "peer-b")
    assert (await heartbeat(service, intruder, joined))["status"] == 403
    reg = service.coord.nodes
    with pytest.raises(AuthError):
        reg.authenticate(joined["node_id"], f"Bearer {joined['credential']}")
    with pytest.raises(AuthError):
        await reg.heartbeat(joined["node_id"], HeartbeatRequest(worker_state="running", sent_at=utcnow()))
    reg.authenticate(joined["node_id"], f"Bearer {joined['credential']}", transport_peer_id=conn.remote_id)


@pytest.mark.anyio
async def test_persisted_identity_prevents_takeover_even_after_leave(service, store, events):
    conn = connect(service)
    joined = await join(service, conn)
    await service.coord.nodes.leave(joined["node_id"], by_owner=True)
    restored = registry(store, events, service.coord.sessions)
    restored.restore()
    req = join_request(service.coord.sessions.create("again").invite.code, worker_ip="127.0.0.1")

    async def forbidden_allocation():
        pytest.fail("a different peer must be rejected before bridge allocation")

    for peer_id in (None, "peer-b"):
        with pytest.raises(JoinRejected) as exc:
            await restored.join(req, "127.0.0.1", None, transport_peer_id=peer_id,
                                transport_worker=forbidden_allocation if peer_id else None)
        assert exc.value.code == "node_id_in_use"
    assert restored.get(joined["node_id"]).transport_peer_id == conn.remote_id


@pytest.mark.anyio
async def test_reconnect_requires_credential_keeps_port_and_closes_old_rpc(service):
    conn = connect(service)
    joined = await join(service, conn)
    node = service.coord.nodes.get(joined["node_id"])
    reader, writer = await asyncio.open_connection(node.worker.host, node.worker.port)
    writer.write(b"first")
    await writer.drain()
    assert await asyncio.wait_for(reader.readexactly(5), 1) == b"first"
    replacement = connect(service)
    invalid = {**joined, "credential": "invalid"}
    assert (await heartbeat(service, replacement, invalid))["status"] == 403
    assert not conn.closed
    assert (await heartbeat(service, replacement, joined))["status"] == 200
    assert conn.closed
    assert conn.streams[0].closed.is_set()
    assert await asyncio.wait_for(reader.read(), 1) == b""
    assert service.coord.nodes.get(node.id).worker.port == node.worker.port
    assert (await heartbeat(service, conn, joined))["status"] == 403
    writer.close()
    await writer.wait_closed()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["pause", "leave", "remove", "close_session", "stale"])
async def test_revocation_closes_active_rpc(service, operation):
    conn = connect(service)
    joined = await join(service, conn)
    node = service.coord.nodes.get(joined["node_id"])
    reader, writer = await asyncio.open_connection(node.worker.host, node.worker.port)
    writer.write(b"payload")
    await writer.drain()
    assert await asyncio.wait_for(reader.readexactly(7), 1) == b"payload"
    assert conn.streams[0].header == {"op": "rpc", "node_id": node.id}
    if operation == "pause":
        assert (await heartbeat(service, conn, joined, WorkerState.paused))["status"] == 200
    elif operation == "leave":
        response = await service._dispatch(conn, {"op": "leave", "node_id": node.id,
                                                  "credential": joined["credential"]})
        assert response == {"status": 200, "body": {"ok": True}}
    elif operation == "remove":
        await service.coord.nodes.leave(node.id, by_owner=True)
    elif operation == "close_session":
        service.coord.sessions.close(node.session_id)
        await service.coord.nodes.close_session(node.session_id)
    else:
        await service.coord.nodes.transport_disconnected(node.id, conn.remote_id)
    assert await asyncio.wait_for(reader.read(), 1) == b""
    assert conn.streams[0].closed.is_set()
    writer.close()
    await writer.wait_closed()
    if operation in {"leave", "remove", "close_session"}:
        assert node.id not in service._peers
        assert (await heartbeat(service, conn, joined))["status"] == 403
    else:
        # Keep the reserved port for a valid resume, but reject RPC until it is eligible again.
        new_reader, new_writer = await asyncio.open_connection(node.worker.host, node.worker.port)
        assert await asyncio.wait_for(new_reader.read(), 1) == b""
        new_writer.close()
        await new_writer.wait_closed()


@pytest.mark.anyio
async def test_unauthenticated_idle_connection_has_bounded_admission(service, monkeypatch):
    monkeypatch.setattr("hypercompute.transport.host.FIRST_STREAM_TIMEOUT", 0.01)
    conn = FakeConnection()
    await asyncio.wait_for(service._handle_connection(conn), 1)
    assert conn.closed and id(conn) not in service._connections


@pytest.mark.anyio
async def test_health_polling_cannot_extend_absolute_admission_deadline(service, monkeypatch):
    monkeypatch.setattr("hypercompute.transport.host.ADMISSION_TIMEOUT", 0.08)
    monkeypatch.setattr("hypercompute.transport.host.FIRST_STREAM_TIMEOUT", 0.04)
    responses = []

    class HealthStream(EchoStream):
        async def receive_json(self):
            return {"op": "health"}

        async def send_json(self, data):
            responses.append(data)

    class HealthPollingConnection(FakeConnection):
        async def accept_bi(self):
            await asyncio.sleep(0.005)
            return HealthStream()

    conn = HealthPollingConnection()
    await asyncio.wait_for(service._handle_connection(conn), 1)
    assert len(responses) >= 2 and all(response["status"] == 200 for response in responses)
    assert conn.closed and id(conn) not in service._connections
    assert not service._peers


@pytest.mark.anyio
async def test_join_rechecks_session_after_bridge_allocation_yields(service):
    svc, reg = service.coord.sessions, service.coord.nodes
    invite = svc.create("test")

    async def allocation():
        svc.close(invite.id)
        return WorkerEndpoint(host="127.0.0.1", port=12345)

    req = join_request(invite.invite.code, worker_ip="127.0.0.1")
    with pytest.raises(JoinRejected) as exc:
        await reg.join(req, "127.0.0.1", None, transport_peer_id="peer-a", transport_worker=allocation)
    assert exc.value.code == "session_closed"
    assert reg.get(req.node_id) is None


@pytest.mark.anyio
async def test_join_rechecks_identity_after_bridge_allocation_yields(service):
    svc, reg = service.coord.sessions, service.coord.nodes
    first = join_request(svc.create("test").invite.code, worker_ip="127.0.0.1")
    second = join_request(svc.create("test").invite.code, worker_ip="127.0.0.1")

    async def allocation_b():
        return WorkerEndpoint(host="127.0.0.1", port=12346)

    async def allocation_a():
        await reg.join(second, "127.0.0.1", None, transport_peer_id="peer-b", transport_worker=allocation_b)
        return WorkerEndpoint(host="127.0.0.1", port=12345)

    with pytest.raises(JoinRejected) as exc:
        await reg.join(first, "127.0.0.1", None, transport_peer_id="peer-a", transport_worker=allocation_a)
    assert exc.value.code == "node_id_in_use"
    assert reg.get(first.node_id).transport_peer_id == "peer-b"
    assert reg.get(first.node_id).state in {NodeState.verifying, NodeState.eligible}


@pytest.mark.anyio
async def test_owner_removal_during_reconnect_cannot_restore_bridge(service, monkeypatch):
    conn = connect(service)
    joined = await join(service, conn)
    node_id = joined["node_id"]
    await service._revoke(node_id, True)  # gateway restart requires a newly allocated bridge
    start_server = asyncio.start_server

    async def remove_during_allocation(*args, **kwargs):
        await service.coord.nodes.leave(node_id, by_owner=True)
        return await start_server(*args, **kwargs)

    monkeypatch.setattr("hypercompute.transport.host.asyncio.start_server", remove_during_allocation)
    assert (await heartbeat(service, connect(service), joined))["status"] == 403
    assert service.coord.nodes.get(node_id).state == NodeState.removed
    assert node_id not in service._peers
