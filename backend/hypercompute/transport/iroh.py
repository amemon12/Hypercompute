"""Small Iroh adapter; application authorization belongs to the service layer.

Endpoint tickets identify a destination. They are not bearer credentials. This
module deliberately has no facility for forwarding to a peer-supplied address.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import logging
import os
import re
import stat
import struct
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from iroh import iroh_ffi as iroh

ALPN = b"hypercompute/peer/1"
MAX_FRAME_BYTES = 64 * 1024
MAX_TICKET_CHARS = 16 * 1024
COPY_CHUNK_BYTES = 64 * 1024  # QUIC read size: the binding reads into a fresh buffer this large
TCP_READ_BYTES = 256 * 1024  # fewer, larger stream writes for bulk uploads (no per-read allocation)
log = logging.getLogger(__name__)


# iroh 1.1.0 ships a uniffi-generated Python binding with two hot-path costs that the RPC bridge pays per
# chunk. Both are replaced at import with behaviour-identical versions; restore_stock_binding() undoes
# them so benchmarks and tests can compare against the stock binding in one process.
_STOCK_BUILDER_WRITE: Any = getattr(getattr(iroh, "_UniffiRustBufferBuilder", None), "write", None)
_STOCK_CONTINUATION: Any = getattr(iroh, "_uniffi_continuation_callback", None)
_INLINE_CONTINUATION_NAMES = ("_UNIFFI_RUST_FUTURE_CONTINUATION_CALLBACK", "_UniffiContinuationHandleMap",
                              "_uniffi_set_future_result", "_uniffi_continuation_callback",
                              "_UNIFFI_RUST_FUTURE_POLL_READY")
_inline_continuation: Any = None


def _memmove_write(self: Any, value: Any) -> None:
    """Copy a bytes-like value into the builder's RustBuffer with one memmove."""
    data = value if isinstance(value, bytes) else bytes(value)
    size = len(data)
    if size != len(value):
        # Items wider than one byte (e.g. a memoryview of an int array): the length prefix counts items and the
        # generated code writes one byte per item, so only the stock copy keeps that layout.
        _STOCK_BUILDER_WRITE(self, value)
        return
    with self._reserve(size):
        if size:
            address = ctypes.cast(self.rbuf.data, ctypes.c_void_p).value
            if address is None:
                raise MemoryError("Iroh RustBuffer has no storage after reserve")
            ctypes.memmove(address + self.rbuf.len, data, size)


def _bytes_round_trip_ok() -> bool:
    converter = iroh._UniffiFfiConverterBytes
    sample = bytes(range(256)) * 5 + b"tail"
    return bool(converter.lift(converter.lower(sample)) == sample)


def install_fast_bytes_lowering() -> bool:
    """Replace the binding's per-byte Python copy into Rust buffers with a single memmove.

    The generated code copies every byte passed to SendStream.write_all in a Python loop: about 0.11 us
    per byte on the G14, which capped the RPC bridge near 10 MiB/s. The buffer layout and reservation
    logic stay the generated ones; a round-trip self check keeps the stock copy if the binding does not
    behave as expected. Returns whether the fast copy is active.
    """
    builder = getattr(iroh, "_UniffiRustBufferBuilder", None)
    if (builder is None or not hasattr(builder, "_reserve") or not callable(_STOCK_BUILDER_WRITE)
            or not hasattr(iroh, "_UniffiFfiConverterBytes")):
        log.warning("Iroh binding layout changed; keeping its per-byte buffer copy")
        return False
    if builder.write is _memmove_write:
        return True
    builder.write = _memmove_write
    try:
        if _bytes_round_trip_ok():
            return True
        log.warning("Iroh fast buffer copy failed its self check; keeping the per-byte copy")
    except Exception as exc:
        log.warning("Iroh fast buffer copy self check raised %s; keeping the per-byte copy", type(exc).__name__)
    builder.write = _STOCK_BUILDER_WRITE
    return False


def _make_inline_continuation() -> Any:
    handles = iroh._UniffiContinuationHandleMap
    set_result = iroh._uniffi_set_future_result
    ready = iroh._UNIFFI_RUST_FUTURE_POLL_READY

    def continuation(future_handle: int, poll_code: int) -> None:
        eventloop, future = handles.remove(future_handle)
        if asyncio._get_running_loop() is not eventloop:
            eventloop.call_soon_threadsafe(set_result, future, poll_code)
        elif poll_code == ready:
            set_result(future, poll_code)
        else:
            # A future that woke itself while being polled is re-polled on the next loop pass, as before,
            # so a Rust-side yield can never turn into a busy loop that starves the event loop.
            eventloop.call_soon(set_result, future, poll_code)

    return iroh._UNIFFI_RUST_FUTURE_CONTINUATION_CALLBACK(continuation)


def install_inline_ready_wakeups() -> bool:
    """Resolve a uniffi future in place when Rust reports it on the event loop's own thread.

    The generated continuation always used call_soon_threadsafe, so even a write that completed on its
    first poll cost a self-pipe syscall and an extra event loop pass (about 37 us per write_all on
    Windows, 15 us after this change). Rust threads that wake a pending read keep the thread-safe path,
    and a non-ready poll on the loop thread is still deferred one loop pass. Setting a fresh future's
    result only schedules callbacks, so nothing re-enters Rust from inside the callback. Returns whether
    the inline path is active.
    """
    global _inline_continuation
    if not all(hasattr(iroh, name) for name in _INLINE_CONTINUATION_NAMES):
        log.warning("Iroh binding layout changed; keeping thread-safe future wakeups")
        return False
    if _inline_continuation is None:
        _inline_continuation = _make_inline_continuation()
    iroh._uniffi_continuation_callback = _inline_continuation
    return True


def install_binding_speedups() -> dict[str, bool]:
    """Install both binding speedups; returns which are active."""
    return {"fast_bytes_lowering": install_fast_bytes_lowering(),
            "inline_ready_wakeups": install_inline_ready_wakeups()}


def restore_stock_binding() -> None:
    """Put back the binding's generated copy and continuation (for A/B benchmarks and tests)."""
    builder = getattr(iroh, "_UniffiRustBufferBuilder", None)
    if builder is not None and _STOCK_BUILDER_WRITE is not None:
        builder.write = _STOCK_BUILDER_WRITE
    if _STOCK_CONTINUATION is not None:
        iroh._uniffi_continuation_callback = _STOCK_CONTINUATION


def binding_speedups() -> dict[str, bool]:
    """Which binding speedups are active right now."""
    builder = getattr(iroh, "_UniffiRustBufferBuilder", None)
    return {"fast_bytes_lowering": builder is not None and builder.write is _memmove_write,
            "inline_ready_wakeups": _inline_continuation is not None
            and getattr(iroh, "_uniffi_continuation_callback", None) is _inline_continuation}


# HYPERCOMPUTE_STOCK_BRIDGE=1 keeps the stock binding, so an end-to-end A/B across machines needs no code change.
BINDING_SPEEDUPS = ({"fast_bytes_lowering": False, "inline_ready_wakeups": False}
                    if os.environ.get("HYPERCOMPUTE_STOCK_BRIDGE") == "1" else install_binding_speedups())


def _restrict_key_permissions(path: Path) -> None:
    """Replace a key file's access list with access for its current user only."""
    if os.name != "nt":
        path.chmod(0o600)
        return
    # chmod does not restrict a Windows ACL. Set a protected DACL instead.
    whoami = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "whoami.exe"
    result = subprocess.run(
        [str(whoami), "/user", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    sid_match = re.search(r"S-1-\d+(?:-\d+)+", result.stdout)
    if sid_match is None:
        raise OSError("Cannot determine the Windows user SID for the device key")
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    descriptor = ctypes.c_void_p()
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = ctypes.c_int
    set_security = advapi.SetFileSecurityW
    set_security.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_void_p]
    set_security.restype = ctypes.c_int
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not convert(f"D:P(A;;FA;;;{sid_match.group(0)})", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION
        if not set_security(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


def load_device_key(path: Path) -> bytes:
    """Load or exclusively create a 32-byte identity; never replace corrupt keys.

    The empty file is protected before any key bytes are written on Windows.
    Concurrent creation fails closed if the other writer has not finished yet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Device key must be a regular file, not a symlink")
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Device key must be a regular file")
        _restrict_key_permissions(path)
        if created:
            key = cast(bytes, iroh.SecretKey.generate().to_bytes())
            if os.write(fd, key) != len(key):
                raise OSError("Could not persist the complete device key")
            os.fsync(fd)
            return key
        key = os.read(fd, 33)
        if len(key) != 32:
            raise ValueError("Invalid device key file; refusing to replace the device identity")
        return key
    finally:
        os.close(fd)


def _validate_relay_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or not parsed.hostname:
        raise ValueError("Relay URL must have a host and must not contain credentials or a fragment")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError("Relay URL must use HTTPS (HTTP is allowed only for a local test relay)")


@dataclass
class IrohStream:
    """A bidirectional stream with bounded control frames and streaming I/O."""

    _raw: Any
    control_timeout: float = 15.0
    # Each recv()/send() call on the binding is an FFI round trip that builds a new handle wrapper.
    _recv: Any = field(default=None, init=False, repr=False, compare=False)
    _send: Any = field(default=None, init=False, repr=False, compare=False)

    def _receiver(self) -> Any:
        if self._recv is None:
            self._recv = self._raw.recv()
        return self._recv

    def _sender(self) -> Any:
        if self._send is None:
            self._send = self._raw.send()
        return self._send

    async def read(self, size: int = COPY_CHUNK_BYTES) -> bytes:
        """Read a bounded chunk; an empty result indicates a clean half-close."""
        if not 0 < size <= 1024 * 1024:
            raise ValueError("Stream read size must be between 1 byte and 1 MiB")
        return cast(bytes, await self._receiver().read(size))

    async def write(self, data: bytes) -> None:
        """Write bytes with QUIC flow-control backpressure."""
        await self._sender().write_all(data)

    async def finish(self) -> None:
        """Half-close the sending direction, preserving the receiving direction."""
        await self._sender().finish()

    async def abort(self) -> None:
        """Stop both directions without allowing cleanup to wait indefinitely."""
        async def stop_recv() -> None:
            with contextlib.suppress(Exception):
                await self._receiver().stop(1)

        async def reset_send() -> None:
            with contextlib.suppress(Exception):
                await self._sender().reset(1)

        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(2):
                await asyncio.gather(stop_recv(), reset_send())

    async def send_json(self, value: dict[str, Any]) -> None:
        """Send one length-prefixed UTF-8 JSON object, bounded to 64 KiB."""
        if not isinstance(value, dict):
            raise ValueError("Control message must be a JSON object")
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if not 0 < len(body) <= MAX_FRAME_BYTES:
            raise ValueError("Control message exceeds the 64 KiB limit")
        async with asyncio.timeout(self.control_timeout):
            await self.write(struct.pack("!I", len(body)) + body)

    async def receive_json(self) -> dict[str, Any]:
        """Read a control object without buffering an unbounded or stalled frame."""
        async with asyncio.timeout(self.control_timeout):
            header = await self._receiver().read_exact(4)
            size = struct.unpack("!I", header)[0]
            if not 0 < size <= MAX_FRAME_BYTES:
                raise ValueError("Control frame length is outside the allowed range")
            body = await self._receiver().read_exact(size)
        value = json.loads(body.decode("utf-8"), parse_constant=_invalid_json_constant)
        if not isinstance(value, dict):
            raise ValueError("Control message must be a JSON object")
        return value


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


@dataclass
class IrohConnection:
    """A cryptographically identified connection, not yet application-authorized."""

    _raw: Any
    timeout: float = 15.0
    _on_close: Callable[[IrohConnection], None] | None = None

    @property
    def remote_id(self) -> str:
        """The TLS-authenticated remote public key."""
        return str(self._raw.remote_id())

    async def open_bi(self) -> IrohStream:
        """Open an outgoing stream with a bounded allocation wait."""
        async with asyncio.timeout(self.timeout):
            return IrohStream(await self._raw.open_bi(), self.timeout)

    async def accept_bi(self) -> IrohStream:
        """Wait for an incoming stream; connection closure wakes this wait."""
        return IrohStream(await self._raw.accept_bi(), self.timeout)

    def close(self, code: int = 0, reason: str = "closed") -> None:
        """Close all connection streams immediately."""
        self._raw.close(code, reason.encode("utf-8")[:256])
        if self._on_close is not None:
            self._on_close(self)

    def path_info(self) -> dict[str, Any]:
        """Return the selected route and measured RTT, without inferring throughput."""
        paths = self._raw.paths()
        selected = next((path for path in paths if path.is_selected), None)
        return {
            "remote_id": self.remote_id,
            "mode": "relay" if selected and selected.is_relay else "direct" if selected else "connecting",
            "rtt_ms": self._raw.rtt(),
            "paths": len(paths),
            "selected_stats": _path_stats(selected),
        }


def _path_stats(path: Any) -> dict[str, int] | None:
    """The selected path's QUIC counters (cumulative), so a run can show cwnd, loss and congestion on the link."""
    stats = getattr(path, "stats", None)
    if stats is None:
        return None
    return {name: int(getattr(stats, name)) for name in ("cwnd", "congestion_events", "lost_packets", "lost_bytes",
                                                         "udp_tx_bytes", "udp_rx_bytes", "current_mtu")}


ConnectionHandler = Callable[[IrohConnection], Awaitable[None]]


class IrohEndpoint:
    """Manage an Iroh endpoint, its device identity and its incoming handlers."""

    def __init__(
        self,
        key_path: Path,
        relay_url: str | None = None,
        relays_enabled: bool = True,
        relay_only: bool = False,
        connect_timeout: float = 15.0,
        *,
        bind_addr: str | None = None,
        max_connections: int = 32,
    ) -> None:
        if relay_only:
            raise ValueError("The Iroh Python 1.1 API cannot enforce relay-only mode")
        if relay_url:
            _validate_relay_url(relay_url)
        if relay_url and not relays_enabled:
            raise ValueError("A custom relay cannot be used with relays disabled")
        if connect_timeout <= 0 or max_connections < 1:
            raise ValueError("Timeout and connection limit must be positive")
        self.key_path = Path(key_path)
        self.relay_url = relay_url
        self.relays_enabled = relays_enabled
        self.connect_timeout = connect_timeout
        self.bind_addr = bind_addr
        self.max_connections = max_connections
        self._endpoint: Any = None
        self._accept_task: asyncio.Task[None] | None = None
        self._handlers: set[asyncio.Task[None]] = set()
        self._connections: list[IrohConnection] = []

    @property
    def endpoint_id(self) -> str:
        """The persistent public device identity."""
        if self._endpoint is None:
            raise RuntimeError("Endpoint has not started")
        return str(self._endpoint.id())

    @property
    def address(self) -> str:
        """A serializable ticket including current relay and direct addresses."""
        if self._endpoint is None:
            raise RuntimeError("Endpoint has not started")
        return str(iroh.EndpointTicket.from_addr(self._endpoint.addr()))

    async def start(self, handler: ConnectionHandler) -> None:
        """Bind, obtain a usable relay address and start accepting connections."""
        if self._endpoint is not None:
            raise RuntimeError("Endpoint is already started")
        iroh.uniffi_set_event_loop(asyncio.get_running_loop())
        key = await asyncio.to_thread(load_device_key, self.key_path)
        relay_mode = (
            iroh.RelayMode.custom_from_urls([self.relay_url])
            if self.relay_url
            else iroh.RelayMode.default_mode() if self.relays_enabled else iroh.RelayMode.disabled()
        )
        options = iroh.EndpointOptions(
            preset=iroh.preset_minimal(),
            secret_key=key,
            alpns=[ALPN],
            relay_mode=relay_mode,
            bind_addr=self.bind_addr,
        )
        async with asyncio.timeout(self.connect_timeout):
            self._endpoint = await iroh.Endpoint.bind(options)
        try:
            if self.relays_enabled:
                async with asyncio.timeout(self.connect_timeout):
                    await self._endpoint.online()
            self._accept_task = asyncio.create_task(self._accept_loop(handler))
        except BaseException:
            await self.close()
            raise

    async def _accept_loop(self, handler: ConnectionHandler) -> None:
        semaphore = asyncio.Semaphore(self.max_connections)
        while True:
            await semaphore.acquire()
            try:
                incoming = await self._endpoint.accept_next()
            except BaseException:
                semaphore.release()
                raise
            if incoming is None:
                semaphore.release()
                return
            task = asyncio.create_task(self._handle(incoming, handler, semaphore))
            self._handlers.add(task)
            task.add_done_callback(self._handlers.discard)

    async def _handle(self, incoming: Any, handler: ConnectionHandler, semaphore: asyncio.Semaphore) -> None:
        connection: IrohConnection | None = None
        try:
            async with asyncio.timeout(self.connect_timeout):
                accepting = await incoming.accept()
                raw = await accepting.connect()
            raw.set_max_concurrent_bi_streams(32)
            raw.set_max_concurrent_uni_streams(0)
            connection = IrohConnection(raw, self.connect_timeout, self._forget_connection)
            self._connections.append(connection)
            await handler(connection)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("Peer connection handler ended: %s", type(exc).__name__)
        finally:
            if connection is not None:
                connection.close()
            semaphore.release()

    def _forget_connection(self, connection: IrohConnection) -> None:
        with contextlib.suppress(ValueError):
            self._connections.remove(connection)

    async def connect(self, address: str) -> IrohConnection:
        """Connect using a ticket whose endpoint identity is checked by Iroh TLS."""
        if self._endpoint is None:
            raise RuntimeError("Endpoint has not started")
        self._connections[:] = [c for c in self._connections if c._raw.close_reason() is None]
        if len(self._connections) >= self.max_connections:
            raise RuntimeError("Endpoint connection limit reached")
        if not isinstance(address, str) or not 0 < len(address) <= MAX_TICKET_CHARS:
            raise ValueError("Invalid endpoint ticket length")
        addr = iroh.EndpointTicket.from_string(address).endpoint_addr()
        if addr.relay_url():
            _validate_relay_url(addr.relay_url())
        async with asyncio.timeout(self.connect_timeout):
            raw = await self._endpoint.connect(addr, ALPN)
        raw.set_max_concurrent_bi_streams(32)
        raw.set_max_concurrent_uni_streams(0)
        connection = IrohConnection(raw, self.connect_timeout, self._forget_connection)
        self._connections.append(connection)
        return connection

    async def close(self) -> None:
        """Close all streams and cancel handlers, including active TCP bridges."""
        if self._accept_task is not None:
            self._accept_task.cancel()
            await asyncio.gather(self._accept_task, return_exceptions=True)
            self._accept_task = None
        for connection in list(self._connections):
            connection.close()
        tasks = list(self._handlers)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connections.clear()
        if self._endpoint is not None:
            endpoint, self._endpoint = self._endpoint, None
            async with asyncio.timeout(5):
                await endpoint.close()


@dataclass
class BridgeStats:
    """Byte and chunk counters for one bridged TCP connection, updated in place while it runs."""

    tcp_to_quic_bytes: int = 0
    tcp_to_quic_chunks: int = 0
    quic_to_tcp_bytes: int = 0
    quic_to_tcp_chunks: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"tcp_to_quic_bytes": self.tcp_to_quic_bytes, "tcp_to_quic_chunks": self.tcp_to_quic_chunks,
                "quic_to_tcp_bytes": self.quic_to_tcp_bytes, "quic_to_tcp_chunks": self.quic_to_tcp_chunks}


def _idle_scope(idle_timeout: float | None) -> contextlib.AbstractAsyncContextManager[Any]:
    # The default (no idle timeout) path skips building a Timeout per chunk.
    return contextlib.nullcontext() if idle_timeout is None else asyncio.timeout(idle_timeout)


async def _copy_tcp_to_quic(reader: asyncio.StreamReader, stream: IrohStream, idle_timeout: float | None,
                            stats: BridgeStats) -> None:
    while True:
        async with _idle_scope(idle_timeout):
            data = await reader.read(TCP_READ_BYTES)
            if not data:
                await stream.finish()
                return
            await stream.write(data)
        stats.tcp_to_quic_bytes += len(data)
        stats.tcp_to_quic_chunks += 1


async def _copy_quic_to_tcp(stream: IrohStream, writer: asyncio.StreamWriter, idle_timeout: float | None,
                            stats: BridgeStats) -> None:
    while True:
        async with _idle_scope(idle_timeout):
            data = await stream.read(COPY_CHUNK_BYTES)
            if not data:
                if writer.can_write_eof():
                    writer.write_eof()
                    await writer.drain()
                return
            writer.write(data)
            await writer.drain()
        stats.quic_to_tcp_bytes += len(data)
        stats.quic_to_tcp_chunks += 1


async def bridge_tcp(
    stream: IrohStream,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    stats: BridgeStats | None = None,
    *,
    idle_timeout: float | None = None,
) -> None:
    """Bridge an already-authorized TCP connection with backpressure and half-close.

    The caller owns destination selection and authorization. Cancellation or I/O
    failure aborts both and closes the socket. Data streams are allowed to idle
    between inference requests; only control frames have a default timeout.
    Pass stats to observe the bytes and chunks moved in each direction.
    """
    if idle_timeout is not None and idle_timeout <= 0:
        raise ValueError("Bridge idle timeout must be positive")
    counters = stats if stats is not None else BridgeStats()
    tasks = [asyncio.create_task(_copy_tcp_to_quic(reader, stream, idle_timeout, counters)),
             asyncio.create_task(_copy_quic_to_tcp(stream, writer, idle_timeout, counters))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await stream.abort()
        raise
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                await writer.wait_closed()
