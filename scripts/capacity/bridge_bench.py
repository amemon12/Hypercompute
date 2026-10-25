"""Loopback benchmark of the Iroh RPC bridge against a plain TCP floor, all on this machine.

  uv run --no-sync python scripts/capacity/bridge_bench.py --label after

Paths measured, each with fresh connections, interleaved round by round so machine noise hits all alike:
  tcp         requester -> fake rpc-server (the floor)
  tcp_proxy   requester -> one asyncio TCP forwarder -> fake rpc-server (Python stream overhead alone)
  iroh_stock  requester -> host bridge listener -> bridge_tcp -> Iroh stream (direct, relays off)
              -> bridge_tcp -> fake rpc-server, with the same JSON handshake the host and peer services use,
              and the iroh binding's generated copy and wakeup code restored (the code before the fix)
  iroh        the same path with the binding speedups from hypercompute.transport.iroh installed

The requester and the fake rpc-server run in a child process with blocking sockets and TCP_NODELAY, and write the way
ggml-rpc writes: command byte, 8-byte size and payload as three send calls, and the reply as a size send plus a data
send. The bridge therefore sees the same small segments llama.cpp produces. Both bridges share this process's event
loop because the Iroh FFI binds one uniffi event loop per process; in production each side runs on its own machine,
and Wi-Fi plus the Mac side are not part of this number.

Fake protocol: request | cmd u8 | size u64 LE | payload |, the first 8 payload bytes (u64 LE) name the reply size;
reply | size u64 LE | reply bytes |.

Output: results/capacity/<UTC stamp>-bridge-bench[-label]/bench.json (plus profile.txt with --profile) and a table.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import cProfile
import hashlib
import io
import json
import multiprocessing
import platform
import pstats
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from hypercompute.transport.iroh import (  # noqa: E402
    IrohConnection,
    IrohEndpoint,
    IrohStream,
    binding_speedups,
    bridge_tcp,
    install_binding_speedups,
    restore_stock_binding,
)
from iroh.iroh_ffi import IrohError  # noqa: E402

MIB = 1 << 20
DEFAULT_PAIRS = "16:16,256:256,2048:2048,16:2048,11520:11520"
PATHS = ("tcp", "tcp_proxy", "iroh_stock", "iroh")
MAX_REPLY_BYTES = 64 * MIB
SOCKET_TIMEOUT = 60.0
COMMAND_TIMEOUT = 600.0
HEADER = struct.Struct("<Q")


# ---------------------------------------------------------------- child process: fake rpc-server and requester


def _recv_exact(sock: socket.socket, view: memoryview) -> bool:
    """Fill view completely; False on a clean EOF before the first byte, error on a short read after it."""
    got = 0
    while got < len(view):
        n = sock.recv_into(view[got:])
        if n == 0:
            if got == 0:
                return False
            raise ConnectionError("peer closed mid-message")
        got += n
    return True


def _drain(sock: socket.socket, size: int, scratch: memoryview) -> None:
    while size > 0:
        n = sock.recv_into(scratch[: min(size, len(scratch))])
        if n == 0:
            raise ConnectionError("peer closed mid-payload")
        size -= n


def _serve_connection(conn: socket.socket) -> None:
    """Answer requests on one connection the way rpc-server does: size send, then data send."""
    header, first, scratch = bytearray(9), bytearray(8), memoryview(bytearray(MIB))
    replies: dict[int, bytes] = {}
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while _recv_exact(conn, memoryview(header)):
            size = HEADER.unpack_from(header, 1)[0]
            if size < 8:
                raise ValueError("fake rpc request needs at least 8 payload bytes")
            _recv_exact(conn, memoryview(first))
            _drain(conn, size - 8, scratch)
            reply_size = min(HEADER.unpack(first)[0], MAX_REPLY_BYTES)
            reply = replies.setdefault(reply_size, bytes(reply_size))
            conn.sendall(HEADER.pack(reply_size))
            if reply_size:
                conn.sendall(reply)


class FakeRpcServer:
    """A threaded loopback server; each accepted connection gets its own thread."""

    def __init__(self) -> None:
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port: int = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, name="fake-rpc", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _accept_loop(self) -> None:
        while True:
            conn, _ = self._listener.accept()
            threading.Thread(target=self._guarded, args=(conn,), daemon=True).start()

    @staticmethod
    def _guarded(conn: socket.socket) -> None:
        with contextlib.suppress(OSError, ConnectionError):
            _serve_connection(conn)


def _connect(port: int) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=SOCKET_TIMEOUT)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def _exchange(sock: socket.socket, payload: bytes, reply_view: memoryview, single: bool = False) -> None:
    """One ggml-rpc style request (three sends, or one with single) and its reply."""
    if single:
        sock.sendall(b"\x01" + HEADER.pack(len(payload)) + payload)
    else:
        sock.sendall(b"\x01")
        sock.sendall(HEADER.pack(len(payload)))
        sock.sendall(payload)
    if not _recv_exact(sock, reply_view[:8]):
        raise ConnectionError("bridge closed before the reply")
    size = HEADER.unpack(reply_view[:8])[0]
    if size and not _recv_exact(sock, reply_view[:size]):
        raise ConnectionError("bridge closed mid-reply")


def _rtt(port: int, request: int, response: int, warmup: int, iters: int, single: bool) -> list[float]:
    """Round trips in microseconds on one connection, after warmup."""
    payload = HEADER.pack(response) + bytes(request - 8)
    reply_view = memoryview(bytearray(max(response, 8)))
    samples: list[float] = []
    with _connect(port) as sock:
        for i in range(warmup + iters):
            start = time.perf_counter_ns()
            _exchange(sock, payload, reply_view, single)
            if i >= warmup:
                samples.append((time.perf_counter_ns() - start) / 1000)
    return samples


def _bulk(port: int, nbytes: int, repeats: int) -> list[float]:
    """Seconds to push nbytes one way and get an 8-byte acknowledgement, per repeat."""
    payload = HEADER.pack(8) + bytes(nbytes - 8)
    reply_view = memoryview(bytearray(8))
    seconds: list[float] = []
    with _connect(port) as sock:
        _exchange(sock, HEADER.pack(8) + bytes(8), reply_view)
        for _ in range(repeats):
            start = time.perf_counter()
            _exchange(sock, payload, reply_view)
            seconds.append(time.perf_counter() - start)
    return seconds


def _dispatch(command: tuple[Any, ...]) -> Any:
    kind = command[0]
    if kind == "rtt":
        return _rtt(*command[1:])
    if kind == "bulk":
        return _bulk(*command[1:])
    raise ValueError(f"unknown command {kind!r}")


def child_main(pipe: Connection) -> None:
    """Entry point of the spawned child: serve the fake rpc-server and run requester commands."""
    server = FakeRpcServer()
    server.start()
    pipe.send(("ready", server.port))
    while True:
        command = pipe.recv()
        if command[0] == "stop":
            return
        try:
            pipe.send(("ok", _dispatch(command)))
        except Exception as exc:  # reported to the parent, which fails the run loudly
            pipe.send(("error", f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------- parent process: bridges and orchestration


async def _ask(pipe: Connection, command: tuple[Any, ...]) -> Any:
    pipe.send(command)
    status, value = await asyncio.wait_for(asyncio.to_thread(pipe.recv), COMMAND_TIMEOUT)
    if status != "ok":
        raise RuntimeError(f"child command {command[0]} failed: {value}")
    return value


async def _forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    finally:
        writer.close()


@asynccontextmanager
async def tcp_proxy(target_port: int) -> AsyncIterator[int]:
    """A plain asyncio forwarder, to separate Python stream cost from Iroh cost."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", target_port)
        await asyncio.gather(_forward(reader, up_writer), _forward(up_reader, writer), return_exceptions=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0, limit=64 * 1024)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()


async def _peer_serve(stream: IrohStream, target_port: int) -> None:
    """Mirror PeerClient._serve_rpc: handshake, open the local worker, bridge."""
    try:
        await stream.receive_json()
        reader, writer = await asyncio.open_connection("127.0.0.1", target_port)
        await stream.send_json({"status": 200, "body": {}})
        await bridge_tcp(stream, reader, writer)
    except Exception as exc:  # a closed bench stream ends the bridge; report anything else
        if not isinstance(exc, ConnectionError | TimeoutError):
            print(f"peer bridge ended: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        await stream.abort()


async def _peer_accept(connection: IrohConnection, target_port: int, tasks: set[asyncio.Task[None]]) -> None:
    while True:
        stream = await connection.accept_bi()
        task = asyncio.create_task(_peer_serve(stream, target_port))
        tasks.add(task)
        task.add_done_callback(tasks.discard)


async def _host_serve(connection: IrohConnection, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
    """Mirror HostService._serve_rpc: open a stream, handshake, bridge."""
    stream = await connection.open_bi()
    try:
        await stream.send_json({"op": "rpc", "node_id": "bench"})
        if (await stream.receive_json()).get("status") != 200:
            raise RuntimeError("bench peer refused the RPC stream")
        await bridge_tcp(stream, reader, writer)
    except (IrohError, ConnectionError, TimeoutError):
        pass  # the requester closing its socket races the peer's stream reset; the samples are already taken
    finally:
        await stream.abort()


@asynccontextmanager
async def iroh_bridge(target_port: int, key_dir: Path) -> AsyncIterator[tuple[int, dict[str, Any]]]:
    """Two real endpoints on loopback (relays off); yields the host bridge port and the selected path."""
    host = IrohEndpoint(key_dir / "host.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    peer = IrohEndpoint(key_dir / "peer.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    accepted: asyncio.Future[IrohConnection] = asyncio.get_running_loop().create_future()
    release = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()

    async def on_host_connection(connection: IrohConnection) -> None:
        if not accepted.done():
            accepted.set_result(connection)
        await release.wait()

    async def reject(connection: IrohConnection) -> None:
        connection.close()

    listener: asyncio.AbstractServer | None = None
    try:
        await host.start(on_host_connection)
        await peer.start(reject)
        peer_connection = await peer.connect(host.address)
        accept_task = asyncio.create_task(_peer_accept(peer_connection, target_port, tasks))
        tasks.add(accept_task)
        host_connection = await asyncio.wait_for(accepted, 15)
        listener = await asyncio.start_server(lambda r, w: _host_serve(host_connection, r, w),
                                              "127.0.0.1", 0, limit=64 * 1024)
        yield listener.sockets[0].getsockname()[1], peer_connection.path_info()
    finally:
        if listener is not None:
            listener.close()
        release.set()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await peer.close()
        await host.close()


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize_rtt(samples: list[float]) -> dict[str, float]:
    """Median, p90, p99, min and mean of round trips in microseconds."""
    ordered = sorted(samples)
    return {"n": len(ordered), "median_us": round(statistics.median(ordered), 1),
            "p90_us": round(_percentile(ordered, 0.9), 1), "p99_us": round(_percentile(ordered, 0.99), 1),
            "min_us": round(ordered[0], 1), "mean_us": round(statistics.fmean(ordered), 1)}


def summarize_bulk(nbytes: int, seconds: list[float]) -> dict[str, Any]:
    rates = [nbytes / MIB / s for s in seconds]
    return {"bytes": nbytes, "seconds": [round(s, 4) for s in seconds],
            "median_mib_s": round(statistics.median(rates), 1), "best_mib_s": round(max(rates), 1)}


async def collect(pipe: Connection, port: int, args: argparse.Namespace, into: dict[str, Any]) -> None:
    """One round of every RTT pair and the bulk transfer, appended to into, with this thread's CPU time.

    The bridges run on this thread's event loop, so loop CPU over wall time shows how CPU bound they are
    (Windows updates thread CPU time at the 15.6 ms clock tick, so short rounds are coarse).
    """
    cpu, wall = time.thread_time(), time.perf_counter()
    for request, response in args.pairs:
        samples = await _ask(pipe, ("rtt", port, request, response, args.warmup, args.iters, args.single_send))
        into["rtt"].setdefault(f"{request}:{response}", []).extend(samples)
    rtt_cpu, rtt_wall = time.thread_time() - cpu, time.perf_counter() - wall
    into["bulk"].extend(await _ask(pipe, ("bulk", port, args.bulk_bytes, args.bulk_repeats)))
    into["rtt_loop_cpu_s"] = into.get("rtt_loop_cpu_s", 0.0) + rtt_cpu
    into["rtt_wall_s"] = into.get("rtt_wall_s", 0.0) + rtt_wall


def _select_binding(path: str) -> None:
    if path == "iroh_stock":
        restore_stock_binding()
    else:
        install_binding_speedups()


async def _collect_all(pipe: Connection, ports: dict[str, int], args: argparse.Namespace,
                       profiler: cProfile.Profile | None) -> dict[str, Any]:
    raw: dict[str, Any] = {path: {"rtt": {}, "bulk": []} for path in args.paths}
    try:
        for _ in range(args.rounds):
            for path in args.paths:
                _select_binding(path)
                active = profiler if path == "iroh" else None
                if active is not None:
                    active.enable()
                try:
                    await collect(pipe, ports[path], args, raw[path])
                finally:
                    if active is not None:
                        active.disable()
    finally:
        install_binding_speedups()
    return raw


def summarize(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    exchanges = (args.warmup + args.iters) * len(args.pairs) * args.rounds
    return {path: {"rtt": {pair: summarize_rtt(samples) for pair, samples in data["rtt"].items()},
                   "bulk": summarize_bulk(args.bulk_bytes, data["bulk"]),
                   "rtt_phase": {"loop_cpu_s": round(data.get("rtt_loop_cpu_s", 0.0), 3),
                                 "wall_s": round(data.get("rtt_wall_s", 0.0), 3),
                                 "loop_cpu_us_per_exchange": round(data.get("rtt_loop_cpu_s", 0.0) / exchanges * 1e6,
                                                                   1)}}
            for path, data in raw.items()}


async def run(args: argparse.Namespace, profiler: cProfile.Profile | None) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=child_main, args=(child,), daemon=True)
    process.start()
    try:
        status, server_port = await asyncio.wait_for(asyncio.to_thread(parent.recv), 60)
        if status != "ready":
            raise RuntimeError("child did not start the fake rpc-server")
        with tempfile.TemporaryDirectory(prefix="bridge-bench-") as key_dir:
            async with tcp_proxy(server_port) as proxy_port:
                async with iroh_bridge(server_port, Path(key_dir)) as (iroh_port, iroh_path):
                    ports = {"tcp": server_port, "tcp_proxy": proxy_port, "iroh_stock": iroh_port,
                             "iroh": iroh_port}
                    raw = await _collect_all(parent, ports, args, profiler)
        paths = summarize(raw, args)
        for name in ("iroh_stock", "iroh"):
            if name in paths:
                paths[name]["path"] = iroh_path
        return paths
    finally:
        with contextlib.suppress(OSError):
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()


def overhead(paths: dict[str, Any]) -> dict[str, Any]:
    """Median and p90 added over the TCP floor per pair."""
    floor = paths.get("tcp")
    out: dict[str, Any] = {}
    if floor is None:
        return out
    for name in (p for p in paths if p != "tcp"):
        out[name] = {pair: {"median_us": round(s["median_us"] - floor["rtt"][pair]["median_us"], 1),
                            "p90_us": round(s["p90_us"] - floor["rtt"][pair]["p90_us"], 1)}
                     for pair, s in paths[name]["rtt"].items()}
    return out


def _git_head() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,  # noqa: S607
                                text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def environment() -> dict[str, Any]:
    bridge_source = ROOT / "backend" / "hypercompute" / "transport" / "iroh.py"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "event_loop": type(asyncio.get_running_loop()).__name__, "git_head": _git_head(),
            "iroh_py_sha256": hashlib.sha256(bridge_source.read_bytes()).hexdigest(),
            "binding_speedups_after_run": binding_speedups()}


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for item in text.split(","):
        request, _, response = item.partition(":")
        pair = (int(request), int(response))
        if pair[0] < 16 or not 0 <= pair[1] <= MAX_REPLY_BYTES:
            raise argparse.ArgumentTypeError("each pair is request:response with request >= 16 bytes")
        pairs.append(pair)
    return pairs


def parse_paths(text: str) -> tuple[str, ...]:
    paths = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = set(paths) - set(PATHS)
    if not paths or unknown:
        raise argparse.ArgumentTypeError(f"paths must come from {','.join(PATHS)}")
    return paths


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=parse_pairs, default=parse_pairs(DEFAULT_PAIRS),
                   help=f"request:response byte pairs (default {DEFAULT_PAIRS})")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--bulk-bytes", type=int, default=64 * MIB)
    p.add_argument("--bulk-repeats", type=int, default=2, help="bulk transfers per round")
    p.add_argument("--rounds", type=int, default=2, help="interleaved rounds over all paths")
    p.add_argument("--paths", type=parse_paths, default=PATHS, help=f"comma list from {','.join(PATHS)}")
    p.add_argument("--single-send", action="store_true",
                   help="send each request in one call instead of ggml-rpc's three (shows fragmentation cost)")
    p.add_argument("--label", default="", help="suffix for the evidence directory, e.g. before or after")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "capacity")
    p.add_argument("--profile", action="store_true", help="cProfile the event loop during the iroh path")
    args = p.parse_args(argv)
    if args.iters < 10 or args.warmup < 0 or args.bulk_bytes < 16 or args.bulk_repeats < 1 or args.rounds < 1:
        p.error("need --iters >= 10, --warmup >= 0, --bulk-bytes >= 16, --bulk-repeats >= 1 and --rounds >= 1")
    return args


def print_table(report: dict[str, Any]) -> None:
    paths = report["paths"]
    print(f"{'pair':>14} " + " ".join(f"{name + ' med/p90 us':>24}" for name in paths))
    for pair in next(iter(paths.values()))["rtt"]:
        cells = [f"{p['rtt'][pair]['median_us']:>11.1f} / {p['rtt'][pair]['p90_us']:<10.1f}" for p in paths.values()]
        print(f"{pair:>14} " + " ".join(f"{c:>24}" for c in cells))
    print(f"{'bulk MiB/s':>14} " + " ".join(f"{p['bulk']['median_mib_s']:>24.1f}" for p in paths.values()))


def write_report(args: argparse.Namespace, report: dict[str, Any], profile_text: str | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{args.label}" if args.label else ""
    out_dir = args.results_dir / f"{stamp}-bridge-bench{suffix}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "bench.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if profile_text is not None:
        (out_dir / "profile.txt").write_text(profile_text, encoding="utf-8")
    return out_dir


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    profiler = cProfile.Profile() if args.profile else None
    started = datetime.now(UTC).isoformat(timespec="seconds")
    paths = await run(args, profiler)
    report = {"schema": "bridge-bench/1", "label": args.label, "started_utc": started, "env": environment(),
              "config": {"pairs": [f"{a}:{b}" for a, b in args.pairs], "warmup": args.warmup,
                         "iters": args.iters, "rounds": args.rounds, "bulk_bytes": args.bulk_bytes,
                         "bulk_repeats": args.bulk_repeats, "paths": list(args.paths),
                         "single_send": args.single_send},
              "paths": paths, "overhead_vs_tcp": overhead(paths)}
    if profiler is not None:
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(40)
        report["_profile_text"] = buffer.getvalue()
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(amain(args))
    profile_text = report.pop("_profile_text", None)
    out_dir = write_report(args, report, profile_text)
    print_table(report)
    print(f"evidence: {out_dir / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
