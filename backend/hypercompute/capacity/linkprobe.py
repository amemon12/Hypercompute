"""Link measurement over the llama.cpp RPC wire protocol, without llama-server.

Speaks protocol 8.0.0 as shipped in the pinned release b11770 (ggml/src/ggml-rpc/ggml-rpc.cpp). A request is
`| cmd (1 byte) | payload size (uint64) | payload |`, a response `| payload size (uint64) | payload |`, and every
struct is packed little-endian. HELLO must be the first command; this probe sends 24 zero capability bytes, so the
server stays on plain TCP. Only six commands are used: HELLO, GET_DEVICE_MEMORY, ALLOC_BUFFER, BUFFER_GET_BASE,
SET_TENSOR and FREE_BUFFER.

RTT is the round trip of GET_DEVICE_MEMORY (4-byte request, 16-byte answer). It includes the server's device memory
query, which is a cheap call on the CPU, CUDA and Metal backends. Goodput allocates a scratch buffer on the peer,
writes random bytes into it with SET_TENSOR (a command with no response), fences with one GET_DEVICE_MEMORY (the
server handles one connection's commands in order) and divides the bytes by the elapsed time minus one median RTT.
The payload comes from os.urandom, so the figure does not depend on compression; plain TCP and the Iroh QUIC bridge
(encrypted) do not compress anyway.

cache_flag in SET_TENSOR is always 0, so a server started with `-c` never writes the payload into its cache directory.

Cleanup: the scratch buffer is released with FREE_BUFFER in a finally block while the message stream is in sync, and
the release is confirmed by one more GET_DEVICE_MEMORY. After a timeout or a partial send the stream is out of sync
(a FREE_BUFFER would be parsed as payload), so the socket is closed instead: the server keeps one `rpc_server` per
connection, returns from `rpc_serve_client` on any error or EOF, and its destructor frees every buffer that connection
allocated. A second connection cannot free the buffer, because buffer handles are only valid on the connection that
allocated them. Through a relay bridge this relies on the bridge closing its side when the local socket closes.
"""

from __future__ import annotations

import logging
import math
import os
import socket
import statistics
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from .costs import link_prior
from .types import Coefficient, LinkProfile, PathClass

log = logging.getLogger("hypercompute.capacity.linkprobe")

PROTO_MAJOR = 8  # RPC_PROTO_MAJOR_VERSION in b11770 ggml-rpc.h (8.0.0)
PINNED_RELEASE = "b11770"

# enum rpc_cmd (ggml-rpc.cpp line 57): values are positions in the enum
CMD_ALLOC_BUFFER = 0
CMD_BUFFER_GET_BASE = 3
CMD_FREE_BUFFER = 4
CMD_SET_TENSOR = 6
CMD_GET_DEVICE_MEMORY = 11
CMD_HELLO = 14
_CMD_NAMES = {CMD_ALLOC_BUFFER: "ALLOC_BUFFER", CMD_BUFFER_GET_BASE: "BUFFER_GET_BASE",
              CMD_FREE_BUFFER: "FREE_BUFFER", CMD_SET_TENSOR: "SET_TENSOR",
              CMD_GET_DEVICE_MEMORY: "GET_DEVICE_MEMORY", CMD_HELLO: "HELLO"}

CONN_CAPS_SIZE = 24  # RPC_CONN_CAPS_SIZE in transport.h; all zero means no transport upgrade
GGML_MAX_DIMS = 4
GGML_MAX_SRC = 10
GGML_MAX_OP_PARAMS = 64
GGML_MAX_NAME = 64
GGML_TYPE_I8 = 24  # enum ggml_type in ggml.h: F32 = 0, ..., I8 = 24; block size 1, type size 1
GGML_OP_NONE = 0

# struct rpc_tensor, packed (ggml-rpc.cpp lines 36-53):
#   id u64 (8) + type u32 (4) + buffer u64 (8) + ne u32[4] (16) + nb u32[4] (16) + op u32 (4)
#   + op_params i32[64 / 4 = 16] (64) + flags i32 (4) + src u64[10] (80) + view_src u64 (8) + view_offs u64 (8)
#   + data u64 (8) + name char[64] (64) + use_count i32 (4)
#   = 8 + 4 + 8 + 16 + 16 + 4 + 64 + 4 + 80 + 8 + 8 + 8 + 64 + 4 = 296 bytes (a multiple of 8, as the C static_assert
#   requires)
RPC_TENSOR = struct.Struct(f"<QIQ{GGML_MAX_DIMS}I{GGML_MAX_DIMS}II{GGML_MAX_OP_PARAMS // 4}ii{GGML_MAX_SRC}QQQQ"
                           f"{GGML_MAX_NAME}si")
RPC_TENSOR_SIZE = 296
REQUEST_HEADER = struct.Struct("<BQ")  # cmd, payload size
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
U64_PAIR = struct.Struct("<QQ")  # alloc_buffer_rsp (remote_ptr, remote_size), get_device_memory_rsp (free, total)
HELLO_RSP = struct.Struct(f"<BBBB{CONN_CAPS_SIZE}s")  # major, minor, patch, padding, conn_caps: 28 bytes
ALLOC_REQ = struct.Struct("<IQ")  # device, size: 12 bytes
SET_TENSOR_TAIL = struct.Struct("<BQ")  # cache_flag, offset; the data follows
SET_TENSOR_HEADER_SIZE = RPC_TENSOR_SIZE + SET_TENSOR_TAIL.size  # 305
_CACHE_FLAG = 0  # never 1: the payload must not land in the peer's RPC cache directory
_TENSOR_ID = 1
_TENSOR_NAME = b"hc_linkprobe"

MIB = 1 << 20
MAX_GOODPUT_BYTES = 64 * MIB
MAX_FREE_FRACTION = 0.25
DEFAULT_CHUNK_BYTES = 4 * MIB
DEFAULT_TIMEOUT_S = 30.0
PROBE_PATH_CLASSES: tuple[str, ...] = ("loopback", "lan", "direct", "relay", "unknown")

Payload = Callable[[int], bytes]


class LinkProbeError(RuntimeError):
    """The probe could not complete. The message names the endpoint, the step and the cause."""


class ProtocolMismatch(LinkProbeError):
    """The peer does not speak RPC protocol major version 8, or answered with a malformed frame."""


class ProbeRefused(LinkProbeError):
    """A safety limit refused the goodput request before anything was allocated on the peer."""


# --------------------------------------------------------------------------------------------------- encoding


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """'host:port' or '[v6]:port' -> (host, port)."""
    host, sep, port_text = endpoint.rpartition(":")
    host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    if not sep or not host or not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise ValueError(f"endpoint must be host:port, got {endpoint!r}")
    return host, int(port_text)


def encode_request(cmd: int, payload: bytes) -> bytes:
    """One framed request: cmd byte, uint64 payload size, payload."""
    return REQUEST_HEADER.pack(cmd, len(payload)) + payload


def encode_tensor_1d(*, buffer: int, data: int, nbytes: int) -> bytes:
    """A packed rpc_tensor: 1-D GGML_TYPE_I8 view of `nbytes` bytes at remote address `data` inside `buffer`.

    ne = (nbytes, 1, 1, 1) and nb = (1, nbytes, nbytes, nbytes), so ggml_nbytes on the server equals `nbytes`, which
    keeps both the deserialize_tensor bounds check and ggml_backend_tensor_set's assertion satisfied.
    """
    if not 0 < nbytes < 1 << 32:
        raise ValueError(f"a 1-D rpc_tensor holds 1 to 2^32 - 1 bytes (ne is uint32), got {nbytes}")
    ne = (nbytes, 1, 1, 1)
    nb = (1, nbytes, nbytes, nbytes)
    op_params = (0,) * (GGML_MAX_OP_PARAMS // 4)
    src = (0,) * GGML_MAX_SRC
    return RPC_TENSOR.pack(_TENSOR_ID, GGML_TYPE_I8, buffer, *ne, *nb, GGML_OP_NONE, *op_params, 0, *src,
                           0, 0, data, _TENSOR_NAME, 0)


def encode_set_tensor_header(*, buffer: int, data: int, nbytes: int) -> bytes:
    """SET_TENSOR payload up to the data: rpc_tensor | cache_flag (always 0) | offset (always 0)."""
    return encode_tensor_1d(buffer=buffer, data=data, nbytes=nbytes) + SET_TENSOR_TAIL.pack(_CACHE_FLAG, 0)


# --------------------------------------------------------------------------------------------------- connection


class RpcConnection:
    """One TCP connection to a ggml rpc-server after a successful HELLO. Every socket operation has a timeout."""

    def __init__(self, sock: socket.socket, endpoint: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._sock: socket.socket | None = sock
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.in_sync = True  # False once a send or receive failed part way: the framing can no longer be trusted
        self.version: tuple[int, int, int] = (0, 0, 0)

    @classmethod
    def open(cls, endpoint: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> RpcConnection:
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError("timeout_s must be a finite positive number")
        host, port = parse_endpoint(endpoint)
        try:
            sock = socket.create_connection((host, port), timeout=timeout_s)
        except OSError as exc:
            raise LinkProbeError(f"connect to {endpoint} failed: {exc!r}") from exc
        conn = cls(sock, endpoint, timeout_s)
        try:
            try:
                sock.settimeout(timeout_s)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # the ggml client disables Nagle too
            except OSError as exc:
                raise LinkProbeError(f"configuring the socket to {endpoint} failed: {exc!r}") from exc
            conn.version = conn._hello()
        except BaseException:
            conn.close()
            raise
        return conn

    def __enter__(self) -> RpcConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        sock, self._sock = self._sock, None
        self.in_sync = False
        if sock is not None:
            try:
                sock.close()
            except OSError as exc:
                log.warning("closing the RPC socket to %s failed: %r", self.endpoint, exc)

    def _live(self, cmd: int) -> socket.socket:
        if self._sock is None or not self.in_sync:
            raise LinkProbeError(f"{_CMD_NAMES[cmd]} on {self.endpoint}: connection is closed or out of sync")
        return self._sock

    def _send(self, cmd: int, payload: bytes, data: bytes = b"") -> None:
        sock = self._live(cmd)
        try:
            sock.sendall(REQUEST_HEADER.pack(cmd, len(payload) + len(data)) + payload)
            if data:
                sock.sendall(data)
        except OSError as exc:  # TimeoutError is an OSError
            self.in_sync = False
            raise LinkProbeError(f"{_CMD_NAMES[cmd]} send to {self.endpoint} failed: {exc!r}") from exc
        except BaseException:  # an interrupt part way through a send also leaves a partial frame
            self.in_sync = False
            raise

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("peer closed the connection (the server drops a client on any rejected request)")
            got += k
        return bytes(buf)

    def _recv(self, cmd: int, size: int, timeout_s: float | None = None) -> bytes:
        sock = self._live(cmd)
        try:
            if timeout_s is not None:
                sock.settimeout(timeout_s)
            (announced,) = U64.unpack(self._recv_exact(sock, U64.size))
            if announced != size:
                self.in_sync = False
                raise ProtocolMismatch(f"{_CMD_NAMES[cmd]} from {self.endpoint}: response size {announced}, "
                                       f"expected {size}")
            reply = self._recv_exact(sock, size)
            if timeout_s is not None:
                sock.settimeout(self.timeout_s)
            return reply
        except OSError as exc:
            self.in_sync = False
            raise LinkProbeError(f"{_CMD_NAMES[cmd]} reply from {self.endpoint} failed: {exc!r}") from exc
        except BaseException:
            self.in_sync = False
            raise

    def _hello(self) -> tuple[int, int, int]:
        self._send(CMD_HELLO, bytes(CONN_CAPS_SIZE))
        major, minor, patch, _pad, _caps = HELLO_RSP.unpack(self._recv(CMD_HELLO, HELLO_RSP.size))
        if major != PROTO_MAJOR:
            raise ProtocolMismatch(f"{self.endpoint} speaks RPC protocol {major}.{minor}.{patch}; this probe speaks "
                                   f"{PROTO_MAJOR}.x ({PINNED_RELEASE})")
        return major, minor, patch

    def device_memory(self, device: int, *, reply_timeout_s: float | None = None) -> tuple[int, int]:
        """(free, total) bytes of a device on the peer. `reply_timeout_s` overrides the wait for this one answer."""
        self._send(CMD_GET_DEVICE_MEMORY, U32.pack(device))
        free, total = U64_PAIR.unpack(self._recv(CMD_GET_DEVICE_MEMORY, U64_PAIR.size, reply_timeout_s))
        return free, total

    def alloc_buffer(self, device: int, size: int) -> tuple[int, int]:
        """(remote_ptr, remote_size); remote_ptr 0 means the peer could not allocate."""
        self._send(CMD_ALLOC_BUFFER, ALLOC_REQ.pack(device, size))
        ptr, remote_size = U64_PAIR.unpack(self._recv(CMD_ALLOC_BUFFER, U64_PAIR.size))
        return ptr, remote_size

    def buffer_base(self, remote_ptr: int) -> int:
        self._send(CMD_BUFFER_GET_BASE, U64.pack(remote_ptr))
        (base,) = U64.unpack(self._recv(CMD_BUFFER_GET_BASE, U64.size))
        return base

    def free_buffer(self, remote_ptr: int) -> None:
        """FREE_BUFFER has no response in protocol 8; a later answered request shows it was handled."""
        self._send(CMD_FREE_BUFFER, U64.pack(remote_ptr))

    def set_tensor(self, *, buffer: int, data_ptr: int, data: bytes) -> None:
        """Write `data` at remote address `data_ptr` inside `buffer`. No response; cache_flag is always 0."""
        header = encode_set_tensor_header(buffer=buffer, data=data_ptr, nbytes=len(data))
        self._send(CMD_SET_TENSOR, header, data)


# --------------------------------------------------------------------------------------------------- results


@dataclass(frozen=True)
class RttResult:
    endpoint: str
    path_class: PathClass
    server_version: tuple[int, int, int]
    device: int
    samples_ms: tuple[float, ...]
    median_ms: float
    p90_ms: float  # nearest rank: sorted samples at index ceil(0.9 n) - 1
    free_bytes: int  # device free memory reported by the last sample
    total_bytes: int
    measured_at: datetime

    def to_coefficient(self) -> Coefficient:
        version = ".".join(map(str, self.server_version))
        return Coefficient(value=self.median_ms, unit="ms", provenance="measured", measured_at=self.measured_at,
                           samples=len(self.samples_ms),
                           note=f"median of {len(self.samples_ms)} GET_DEVICE_MEMORY round trips over llama.cpp "
                                f"RPC {version} to {self.endpoint}, path class {self.path_class}; "
                                f"p90 {self.p90_ms:.3f} ms")

    def to_link_profile(self, *, cached_bytes: int = 0) -> LinkProfile:
        return LinkMeasurement(rtt=self, goodput=None).to_link_profile(cached_bytes=cached_bytes)


@dataclass(frozen=True)
class GoodputResult:
    endpoint: str
    path_class: PathClass
    device: int
    total_bytes: int
    chunk_bytes: int
    chunks: int
    elapsed_s: float  # first SET_TENSOR sent -> fence GET_DEVICE_MEMORY answered
    rtt_subtracted_ms: float  # one median RTT: the fence's own round trip
    transfer_s: float
    bytes_per_s: float
    free_confirmed: bool  # FREE_BUFFER followed by an answered request on the same connection
    measured_at: datetime
    payload: str = "os.urandom, fresh bytes per chunk"

    def to_coefficient(self) -> Coefficient:
        return Coefficient(value=self.bytes_per_s, unit="bytes/s", provenance="measured",
                           measured_at=self.measured_at, samples=1,
                           note=f"{self.total_bytes} random bytes in {self.chunks} SET_TENSOR chunks to "
                                f"{self.endpoint}, path class {self.path_class}; bytes / (fence - start - "
                                f"{self.rtt_subtracted_ms:.3f} ms median RTT)")


@dataclass(frozen=True)
class LinkMeasurement:
    rtt: RttResult
    goodput: GoodputResult | None

    def to_link_profile(self, *, cached_bytes: int = 0) -> LinkProfile:
        """Measured rtt (and goodput when probed); round trips per token stays the documented prior."""
        base = link_prior(self.rtt.path_class, cached_bytes=cached_bytes)
        goodput = self.goodput.to_coefficient() if self.goodput is not None else base.goodput_bytes_per_s
        return replace(base, rtt_ms=self.rtt.to_coefficient(), goodput_bytes_per_s=goodput)


# --------------------------------------------------------------------------------------------------- probes


def _check_path_class(path_class: str) -> None:
    if path_class not in PROBE_PATH_CLASSES:
        raise ValueError(f"path_class must be one of {PROBE_PATH_CLASSES}, got {path_class!r}")


def _p90(samples: tuple[float, ...]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def measure_rtt(conn: RpcConnection, samples: int, *, path_class: PathClass, device: int = 0) -> RttResult:
    """`samples` GET_DEVICE_MEMORY round trips on an open connection, after one uncounted warmup exchange."""
    _check_path_class(path_class)
    if samples < 1:
        raise ValueError("samples must be at least 1")
    free, total = conn.device_memory(device)
    times: list[float] = []
    for _ in range(samples):
        t0 = time.perf_counter()
        free, total = conn.device_memory(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = tuple(times)
    return RttResult(endpoint=conn.endpoint, path_class=path_class, server_version=conn.version, device=device,
                     samples_ms=ms, median_ms=statistics.median(ms), p90_ms=_p90(ms), free_bytes=free,
                     total_bytes=total, measured_at=datetime.now(UTC))


def check_goodput_request(total_bytes: int, chunk_bytes: int) -> None:
    """The limits that need no answer from the peer, so a bad request is refused before connecting."""
    if total_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("total_bytes and chunk_bytes must be positive")
    if total_bytes > MAX_GOODPUT_BYTES:
        raise ProbeRefused(f"goodput probe of {total_bytes} bytes exceeds the {MAX_GOODPUT_BYTES} byte cap")


def check_goodput_limits(total_bytes: int, chunk_bytes: int, free_bytes: int) -> None:
    """Refuse before allocating: at most 64 MiB, and at most a quarter of the device's reported free memory."""
    check_goodput_request(total_bytes, chunk_bytes)
    if total_bytes > MAX_FREE_FRACTION * free_bytes:
        raise ProbeRefused(f"goodput probe of {total_bytes} bytes exceeds {MAX_FREE_FRACTION:.0%} of the "
                           f"{free_bytes} bytes the peer reports free")


def _release(conn: RpcConnection, remote_ptr: int, device: int) -> bool:
    """FREE_BUFFER confirmed by an answered GET_DEVICE_MEMORY; out of sync, close and let the server free it."""
    if not conn.in_sync:
        log.warning("RPC stream to %s is out of sync; closing it so the server frees buffer 0x%x on disconnect",
                    conn.endpoint, remote_ptr)
        conn.close()
        return False
    try:
        conn.free_buffer(remote_ptr)
        conn.device_memory(device)  # handled in order: an answer means the free ran without dropping the client
        return True
    except LinkProbeError as exc:
        log.warning("FREE_BUFFER 0x%x on %s not confirmed (%s); closing so the server frees it on disconnect",
                    remote_ptr, conn.endpoint, exc)
        conn.close()
        return False


def _timed_upload(conn: RpcConnection, remote_ptr: int, base: int, chunks: list[bytes], device: int) -> float:
    """Seconds from the first SET_TENSOR to the answered fence.

    A local bridge (the Iroh gateway's RPC port) accepts the chunks into its own buffers at once, so the whole
    transfer can land on the fence's wait. Each chunk's send is allowed one timeout, so the fence is allowed one per
    chunk plus one for itself: the same minimum rate either way.
    """
    start = time.perf_counter()
    offset = 0
    for chunk in chunks:
        conn.set_tensor(buffer=remote_ptr, data_ptr=base + offset, data=chunk)
        offset += len(chunk)
    # fence: answered only after every SET_TENSOR before it was applied
    conn.device_memory(device, reply_timeout_s=conn.timeout_s * (len(chunks) + 1))
    return time.perf_counter() - start


def _make_chunks(total_bytes: int, chunk_bytes: int, payload: Payload) -> list[bytes]:
    chunks: list[bytes] = []
    for offset in range(0, total_bytes, chunk_bytes):
        n = min(chunk_bytes, total_bytes - offset)
        data = payload(n)
        if len(data) != n:
            raise LinkProbeError(f"payload source returned {len(data)} bytes, expected {n}")
        chunks.append(data)
    return chunks


def measure_goodput(conn: RpcConnection, rtt: RttResult, total_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                    *, payload: Payload = os.urandom) -> GoodputResult:
    """Upload `total_bytes` into a scratch buffer on `rtt.device`; the buffer is released on every exit path."""
    check_goodput_limits(total_bytes, chunk_bytes, rtt.free_bytes)
    chunk_bytes = min(chunk_bytes, total_bytes)
    remote_ptr, remote_size = conn.alloc_buffer(rtt.device, total_bytes)
    if remote_ptr == 0:
        raise LinkProbeError(f"ALLOC_BUFFER of {total_bytes} bytes on device {rtt.device} of {conn.endpoint} failed")
    try:
        if remote_size < total_bytes:
            raise LinkProbeError(f"peer allocated {remote_size} bytes, asked for {total_bytes}")
        base = conn.buffer_base(remote_ptr)
        chunks = _make_chunks(total_bytes, chunk_bytes, payload)  # generated before the clock starts
        elapsed = _timed_upload(conn, remote_ptr, base, chunks, rtt.device)
    finally:
        freed = _release(conn, remote_ptr, rtt.device)
    transfer = elapsed - rtt.median_ms / 1000.0
    if transfer <= 0:
        raise LinkProbeError(f"upload of {total_bytes} bytes took {elapsed * 1000:.3f} ms, not more than one median "
                             f"RTT ({rtt.median_ms:.3f} ms); use more bytes")
    return GoodputResult(endpoint=conn.endpoint, path_class=rtt.path_class, device=rtt.device,
                         total_bytes=total_bytes, chunk_bytes=chunk_bytes, chunks=len(chunks), elapsed_s=elapsed,
                         rtt_subtracted_ms=rtt.median_ms, transfer_s=transfer, bytes_per_s=total_bytes / transfer,
                         free_confirmed=freed, measured_at=datetime.now(UTC))


def probe_rtt(endpoint: str, samples: int = 20, *, path_class: PathClass, device: int = 0,
              timeout_s: float = DEFAULT_TIMEOUT_S) -> RttResult:
    """Median and p90 GET_DEVICE_MEMORY round trip to `endpoint`, plus the device's free and total memory."""
    _check_path_class(path_class)
    with RpcConnection.open(endpoint, timeout_s=timeout_s) as conn:
        return measure_rtt(conn, samples, path_class=path_class, device=device)


def probe_goodput(endpoint: str, total_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES, *,
                  path_class: PathClass, device: int = 0, rtt_samples: int = 5,
                  timeout_s: float = DEFAULT_TIMEOUT_S, payload: Payload = os.urandom) -> GoodputResult:
    """Bulk upload rate to `endpoint`; the RTT subtracted is measured on the same connection first."""
    _check_path_class(path_class)
    check_goodput_request(total_bytes, chunk_bytes)
    with RpcConnection.open(endpoint, timeout_s=timeout_s) as conn:
        rtt = measure_rtt(conn, rtt_samples, path_class=path_class, device=device)
        return measure_goodput(conn, rtt, total_bytes, chunk_bytes, payload=payload)


def probe_link(endpoint: str, *, path_class: PathClass, total_bytes: int = 8 * MIB,
               chunk_bytes: int = DEFAULT_CHUNK_BYTES, samples: int = 20, device: int = 0,
               timeout_s: float = DEFAULT_TIMEOUT_S, payload: Payload = os.urandom) -> LinkMeasurement:
    """RTT, then goodput when `total_bytes` > 0, on one connection (the server serves one client at a time)."""
    _check_path_class(path_class)
    if total_bytes < 0:
        raise ValueError("total_bytes must be non-negative")
    if total_bytes > 0:
        check_goodput_request(total_bytes, chunk_bytes)
    with RpcConnection.open(endpoint, timeout_s=timeout_s) as conn:
        rtt = measure_rtt(conn, samples, path_class=path_class, device=device)
        if total_bytes == 0:
            return LinkMeasurement(rtt=rtt, goodput=None)
        goodput = measure_goodput(conn, rtt, total_bytes, chunk_bytes, payload=payload)
    return LinkMeasurement(rtt=rtt, goodput=goodput)
