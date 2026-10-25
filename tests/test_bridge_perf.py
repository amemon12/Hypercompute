"""Iroh RPC bridge hot path: binding speedups, byte accounting over real QUIC, and the loopback bench helpers."""

from __future__ import annotations

import argparse
import array
import asyncio
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

import pytest
from hypercompute.transport import iroh as transport
from hypercompute.transport.iroh import BridgeStats, IrohConnection, IrohEndpoint, bridge_tcp
from iroh import iroh_ffi as ffi

ROOT = Path(__file__).resolve().parents[1]
READY = ffi._UNIFFI_RUST_FUTURE_POLL_READY
WAKE = ffi._UNIFFI_RUST_FUTURE_POLL_WAKE


def load_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bridge_bench", ROOT / "scripts" / "capacity" / "bridge_bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bridge_bench"] = module
    spec.loader.exec_module(module)
    return module


def round_trip(value: bytes | bytearray | memoryview) -> bytes:
    converter = ffi._UniffiFfiConverterBytes
    return converter.lift(converter.lower(value))


def lowering_seconds(value: bytes, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        round_trip(value)
        best = min(best, time.perf_counter() - start)
    return best


def test_speedups_are_installed_at_import() -> None:
    assert transport.BINDING_SPEEDUPS == {"fast_bytes_lowering": True, "inline_ready_wakeups": True}
    assert transport.binding_speedups() == transport.BINDING_SPEEDUPS


@pytest.mark.parametrize("value", [b"", b"\x00", bytes(range(256)) * 3, bytearray(b"abc"),
                                   memoryview(b"xyz" * 1000), bytes(1024 * 1024 + 7)],
                         ids=["empty", "one", "all-bytes", "bytearray", "memoryview", "1mib-plus-7"])
def test_fast_lowering_round_trips_every_bytes_like(value: bytes | bytearray | memoryview) -> None:
    assert round_trip(value) == bytes(value)


def test_fast_lowering_matches_stock_for_multibyte_items() -> None:
    # The length prefix counts items and the generated code writes one byte per item; a raw memmove of the 12-byte
    # buffer under a prefix of 3 would hand Rust a corrupt RustBuffer.
    value = memoryview(array.array("i", [1, 2, 3]))
    try:
        transport.restore_stock_binding()
        stock = round_trip(value)
    finally:
        transport.install_binding_speedups()
    assert stock == b"\x01\x02\x03"
    assert round_trip(value) == stock
    assert transport.binding_speedups()["fast_bytes_lowering"]


def test_restore_and_reinstall_toggle_both_speedups() -> None:
    try:
        transport.restore_stock_binding()
        assert transport.binding_speedups() == {"fast_bytes_lowering": False, "inline_ready_wakeups": False}
        assert round_trip(b"stock path") == b"stock path"
    finally:
        assert transport.install_binding_speedups() == {"fast_bytes_lowering": True, "inline_ready_wakeups": True}
    assert transport.binding_speedups() == {"fast_bytes_lowering": True, "inline_ready_wakeups": True}


@pytest.mark.parametrize(("flag", "active"), [("1", False), ("0", True)])
def test_stock_bridge_env_skips_speedups_at_import(flag: str, active: bool) -> None:
    code = ("import json; from hypercompute.transport import iroh as t; "
            "print(json.dumps([t.BINDING_SPEEDUPS, t.binding_speedups()]))")
    env = {**os.environ, "HYPERCOMPUTE_STOCK_BRIDGE": flag}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, check=True)
    expected = {"fast_bytes_lowering": active, "inline_ready_wakeups": active}
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [expected, expected]


def test_lowering_cost_is_no_longer_per_byte() -> None:
    sample = bytes(256 * 1024)
    fast = lowering_seconds(sample)
    try:
        transport.restore_stock_binding()
        stock = lowering_seconds(sample, repeats=1)
    finally:
        transport.install_binding_speedups()
    # The stock binding copies byte by byte in Python (about 0.1 us per byte); one memmove is far cheaper.
    assert stock > 20 * fast, (stock, fast)


def test_inline_continuation_resolves_ready_in_place_and_defers_everything_else() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        handles = ffi._UniffiContinuationHandleMap
        callback = ffi._uniffi_continuation_callback

        ready = loop.create_future()
        callback(handles.insert((loop, ready)), READY)
        assert ready.done() and ready.result() == READY

        woken = loop.create_future()
        callback(handles.insert((loop, woken)), WAKE)
        assert not woken.done()  # a self-wake is re-polled on the next loop pass, never in a tight loop
        await asyncio.sleep(0)
        assert woken.result() == WAKE

        foreign = loop.create_future()
        await asyncio.to_thread(callback, handles.insert((loop, foreign)), READY)
        assert await asyncio.wait_for(foreign, 2) == READY

        cancelled = loop.create_future()
        cancelled.cancel()
        callback(handles.insert((loop, cancelled)), READY)
        assert cancelled.cancelled()

    asyncio.run(run())


@asynccontextmanager
async def bridged_echo(tmp_path: Path, stats: BridgeStats) -> AsyncIterator[int]:
    """A TCP port whose bytes cross host bridge, real Iroh QUIC and peer bridge to an echo server."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
        writer.close()

    echo_server = await asyncio.start_server(echo, "127.0.0.1", 0)
    echo_port = echo_server.sockets[0].getsockname()[1]
    accepted: asyncio.Future[IrohConnection] = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    async def on_peer_connection(connection: IrohConnection) -> None:
        stream = await connection.accept_bi()
        reader, writer = await asyncio.open_connection("127.0.0.1", echo_port)
        accepted.set_result(connection)
        await bridge_tcp(stream, reader, writer)
        await release.wait()

    peer = IrohEndpoint(tmp_path / "peer.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    host = IrohEndpoint(tmp_path / "host.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    await peer.start(on_peer_connection)
    await host.start(on_peer_connection)
    connection = await host.connect(peer.address)

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        stream = await connection.open_bi()
        await bridge_tcp(stream, reader, writer, stats)

    listener = await asyncio.start_server(serve, "127.0.0.1", 0, limit=64 * 1024)
    try:
        yield listener.sockets[0].getsockname()[1]
    finally:
        release.set()
        listener.close()
        await host.close()
        await peer.close()
        echo_server.close()


def test_real_bridge_moves_ggml_style_messages_and_counts_bytes(tmp_path: Path) -> None:
    async def run() -> None:
        stats = BridgeStats()
        payload = bytes(range(256)) * 8192  # 2 MiB, larger than one stream read or TCP read
        async with bridged_echo(tmp_path, stats) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            # ggml-rpc writes the command byte, the 8-byte size and the payload as separate sends.
            parts = (b"\x05", struct.pack("<Q", 16), b"0123456789abcdef")
            for part in parts:
                writer.write(part)
                await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(25), 5) == b"".join(parts)
            writer.write(payload)
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(len(payload)), 30) == payload
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.05)
        total = 25 + len(payload)
        assert stats.tcp_to_quic_bytes == total and stats.quic_to_tcp_bytes == total
        assert 1 <= stats.tcp_to_quic_chunks <= total and 1 <= stats.quic_to_tcp_chunks <= total
        assert set(stats.as_dict()) == {"tcp_to_quic_bytes", "tcp_to_quic_chunks",
                                        "quic_to_tcp_bytes", "quic_to_tcp_chunks"}

    asyncio.run(run())


def test_bench_helpers_parse_and_summarize() -> None:
    bench = load_bench()
    assert bench.parse_pairs("16:16,2048:8") == [(16, 16), (2048, 8)]
    for bad in ("8:16", "16:-1", "16"):
        with pytest.raises((ValueError, argparse.ArgumentTypeError)):
            bench.parse_pairs(bad)
    assert bench.parse_paths("tcp,iroh") == ("tcp", "iroh")
    with pytest.raises(argparse.ArgumentTypeError, match="paths"):
        bench.parse_paths("udp")
    summary = bench.summarize_rtt([float(v) for v in range(1, 101)])
    assert summary["median_us"] == 50.5 and summary["p90_us"] == 90.0 and summary["min_us"] == 1.0
    paths = {"tcp": {"rtt": {"16:16": {"median_us": 50.0, "p90_us": 60.0}}},
             "iroh": {"rtt": {"16:16": {"median_us": 450.0, "p90_us": 900.0}}}}
    assert bench.overhead(paths) == {"iroh": {"16:16": {"median_us": 400.0, "p90_us": 840.0}}}
    assert bench.summarize_bulk(64 << 20, [0.5, 0.25])["best_mib_s"] == 256.0


def test_bench_paths_carry_ggml_exchanges_in_process(tmp_path: Path) -> None:
    bench = load_bench()

    async def run() -> None:
        server = bench.FakeRpcServer()
        server.start()
        async with bench.tcp_proxy(server.port) as proxy_port:
            async with bench.iroh_bridge(server.port, tmp_path) as (iroh_port, path):
                assert path["mode"] == "direct"
                for port in (server.port, proxy_port, iroh_port):
                    samples = await asyncio.to_thread(bench._rtt, port, 16, 2048, 2, 10, False)
                    assert len(samples) == 10 and min(samples) > 0
                    seconds = await asyncio.to_thread(bench._bulk, port, 1 << 20, 1)
                    assert len(seconds) == 1 and seconds[0] > 0

    asyncio.run(run())
