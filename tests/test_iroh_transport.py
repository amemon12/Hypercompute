"""Real local QUIC tests plus malformed-frame and key-persistence coverage."""

from __future__ import annotations

import asyncio
import os
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from hypercompute.transport.iroh import (
    MAX_FRAME_BYTES,
    IrohConnection,
    IrohEndpoint,
    bridge_tcp,
    load_device_key,
)


@asynccontextmanager
async def pair(
    tmp_path: Path, handler: Callable[[IrohConnection], Awaitable[None]],
) -> AsyncIterator[tuple[IrohEndpoint, IrohEndpoint, IrohConnection]]:
    """Bind real endpoints to loopback with no Internet or relay dependency."""
    server = IrohEndpoint(tmp_path / "server.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    client = IrohEndpoint(tmp_path / "client.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    try:
        await server.start(handler)
        await client.start(handler)
        connection = await client.connect(server.address)
        yield server, client, connection
    finally:
        await client.close()
        await server.close()


def test_device_key_persists_and_rejects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "private" / "device.key"
    first = load_device_key(path)
    assert len(first) == 32
    assert first == load_device_key(path)
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="refusing to replace"):
        load_device_key(path)
    assert path.read_bytes() == b"bad"


@pytest.mark.parametrize("url", ["ftp://example.com", "http://example.com", "https://u:p@example.com", "https://x/#a"])
def test_unsafe_relay_urls_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError):
        IrohEndpoint(tmp_path / "key", relay_url=url)


def test_relay_only_fails_honestly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot enforce"):
        IrohEndpoint(tmp_path / "key", relay_only=True)


def test_endpoint_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disabled"):
        IrohEndpoint(tmp_path / "key", relay_url="https://example.com", relays_enabled=False)
    with pytest.raises(ValueError, match="positive"):
        IrohEndpoint(tmp_path / "key", max_connections=0)


def test_invalid_sender_messages_and_connection_lifecycle(tmp_path: Path) -> None:
    async def run() -> None:
        endpoint = IrohEndpoint(tmp_path / "unstarted.key", relays_enabled=False)
        with pytest.raises(RuntimeError, match="not started"):
            _ = endpoint.address
        with pytest.raises(RuntimeError, match="not started"):
            _ = endpoint.endpoint_id
        with pytest.raises(RuntimeError, match="not started"):
            await endpoint.connect("invalid")

        async def handler(connection: IrohConnection) -> None:
            stream = await connection.accept_bi()
            await stream.receive_json()

        async with pair(tmp_path, handler) as (server, client, connection):
            with pytest.raises(RuntimeError, match="already started"):
                await server.start(handler)
            with pytest.raises(ValueError, match="ticket length"):
                await client.connect("")
            client.max_connections = 1
            with pytest.raises(RuntimeError, match="connection limit"):
                await client.connect(server.address)
            stream = await connection.open_bi()
            with pytest.raises(ValueError, match="JSON object"):
                await stream.send_json([])  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="limit"):
                await stream.send_json({"oversized": "x" * MAX_FRAME_BYTES})
            with pytest.raises(ValueError, match="read size"):
                await stream.read(0)
            await stream.abort()

    asyncio.run(run())


def test_real_quic_echo_and_authenticated_identity(tmp_path: Path) -> None:
    async def run() -> None:
        keep_open = asyncio.Event()
        seen_peer: list[str] = []

        async def handler(connection: IrohConnection) -> None:
            seen_peer.append(connection.remote_id)
            stream = await connection.accept_bi()
            await stream.send_json({"echo": await stream.receive_json()})
            await stream.finish()
            await keep_open.wait()

        async with pair(tmp_path, handler) as (server, client, connection):
            assert connection.remote_id == server.endpoint_id
            stream = await connection.open_bi()
            await stream.send_json({"request": "hello", "value": 42})
            assert await stream.receive_json() == {"echo": {"request": "hello", "value": 42}}
            assert await stream.read() == b""
            assert seen_peer == [client.endpoint_id]
            assert connection.path_info()["mode"] == "direct"
            assert connection.path_info()["rtt_ms"] is not None
            stats = connection.path_info()["selected_stats"]
            assert set(stats) == {"cwnd", "congestion_events", "lost_packets", "lost_bytes", "udp_tx_bytes",
                                  "udp_rx_bytes", "current_mtu"}
            assert stats["cwnd"] > 0 and stats["udp_tx_bytes"] > 0
            connection.close()
            assert connection not in client._connections
            keep_open.set()

    asyncio.run(run())


@pytest.mark.parametrize("payload", [struct.pack("!I", MAX_FRAME_BYTES + 1), struct.pack("!I", 0)])
def test_frame_length_rejected_before_body_read(tmp_path: Path, payload: bytes) -> None:
    async def run() -> None:
        keep_open = asyncio.Event()

        async def handler(connection: IrohConnection) -> None:
            stream = await connection.accept_bi()
            await stream.receive_json()
            await stream.write(payload)
            await keep_open.wait()

        async with pair(tmp_path, handler) as (_, _, connection):
            stream = await connection.open_bi()
            await stream.send_json({})
            with pytest.raises(ValueError, match="frame length"):
                await stream.receive_json()
            await stream.abort()
            keep_open.set()

    asyncio.run(run())


@pytest.mark.parametrize("body", [b"[]", b'{"number": NaN}', b"\xff"])
def test_invalid_control_json_rejected(tmp_path: Path, body: bytes) -> None:
    async def run() -> None:
        keep_open = asyncio.Event()

        async def handler(connection: IrohConnection) -> None:
            stream = await connection.accept_bi()
            await stream.receive_json()
            await stream.write(struct.pack("!I", len(body)) + body)
            await keep_open.wait()

        async with pair(tmp_path, handler) as (_, _, connection):
            stream = await connection.open_bi()
            await stream.send_json({})
            with pytest.raises(ValueError):
                await stream.receive_json()
            keep_open.set()

    asyncio.run(run())


def test_stalled_frame_times_out(tmp_path: Path) -> None:
    async def run() -> None:
        keep_open = asyncio.Event()

        async def handler(connection: IrohConnection) -> None:
            stream = await connection.accept_bi()
            await stream.receive_json()
            await stream.write(struct.pack("!I", 100) + b"partial")
            await keep_open.wait()

        async with pair(tmp_path, handler) as (_, _, connection):
            stream = await connection.open_bi()
            stream.control_timeout = 0.1
            await stream.send_json({})
            with pytest.raises(TimeoutError):
                await stream.receive_json()
            await stream.abort()
            keep_open.set()

    asyncio.run(run())


def test_real_tcp_bridge_preserves_half_close(tmp_path: Path) -> None:
    async def run() -> None:
        payload = b"streamed model bytes\0" * 20000
        completed = asyncio.Event()

        async def tcp_service(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            request = await reader.read()
            writer.write(request[::-1])
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        listener = await asyncio.start_server(tcp_service, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]

        async def handler(connection: IrohConnection) -> None:
            stream = await connection.accept_bi()
            assert await stream.receive_json() == {"op": "test"}
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await stream.send_json({"ready": True})
            await bridge_tcp(stream, reader, writer)
            await completed.wait()

        try:
            async with pair(tmp_path, handler) as (_, _, connection):
                stream = await connection.open_bi()
                await stream.send_json({"op": "test"})
                assert await stream.receive_json() == {"ready": True}
                await stream.write(payload)
                await stream.finish()
                chunks = []
                while chunk := await stream.read():
                    chunks.append(chunk)
                assert b"".join(chunks) == payload[::-1]
                completed.set()
        finally:
            listener.close()
            await listener.wait_closed()

    asyncio.run(run())


def test_endpoint_shutdown_cancels_active_bridge(tmp_path: Path) -> None:
    async def run() -> None:
        eof = asyncio.Event()
        bridged = asyncio.Event()

        async def tcp_service(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.read()
                eof.set()
            finally:
                writer.close()
                await writer.wait_closed()

        listener = await asyncio.start_server(tcp_service, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]

        async def handler(connection: IrohConnection) -> None:
            stream = await connection.accept_bi()
            await stream.receive_json()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            bridged.set()
            await bridge_tcp(stream, reader, writer)

        try:
            async with pair(tmp_path, handler) as (server, _, connection):
                stream = await connection.open_bi()
                await stream.send_json({})
                await asyncio.wait_for(bridged.wait(), 3)
                await server.close()
                await asyncio.wait_for(eof.wait(), 3)
        finally:
            listener.close()
            await listener.wait_closed()

    asyncio.run(run())
