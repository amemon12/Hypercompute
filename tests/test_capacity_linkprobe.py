"""Link probe over the llama.cpp RPC protocol 8 (b11770) against an in-process fake rpc-server.

The fake server mirrors the real one where it matters for safety: HELLO first, fixed request sizes, the
deserialize_tensor and set_tensor bounds checks, the ggml_backend_tensor_set assertion (which aborts the real server
process), FREE_BUFFER without a response, one rpc_server per connection that frees its buffers on disconnect, and a
dropped connection on any rejected request. It parses rpc_tensor with byte offsets computed by hand from the C layout,
independently of the struct format string under test.
"""

from __future__ import annotations

import importlib.util
import json
import math
import socket
import struct
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypercompute.capacity import bridge, linkprobe
from hypercompute.capacity.linkprobe import LinkProbeError, ProbeRefused, ProtocolMismatch
from hypercompute.capacity.measurements import MeasurementIdentity, MeasurementStore

MIB = 1 << 20
ROOT = Path(__file__).resolve().parents[1]

# rpc_tensor byte offsets, from the packed C struct in ggml-rpc.cpp:
# id 0 (u64) | type 8 (u32) | buffer 12 (u64) | ne 20 (4 x u32) | nb 36 (4 x u32) | op 52 (u32) | op_params 56 (64)
# | flags 120 (i32) | src 124 (10 x u64) | view_src 204 | view_offs 212 | data 220 (u64) | name 228 (64)
# | use_count 292 (i32) | end 296
OFF_TYPE, OFF_BUFFER, OFF_NE, OFF_NB, OFF_DATA, OFF_NAME, TENSOR_END = 8, 12, 20, 36, 220, 228, 296
TYPE_SIZE = {0: 4, 24: 1}  # F32, I8: block size 1


def _recv_exact(c: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        part = c.recv(n - len(buf))
        if not part:
            return None
        buf += part
    return buf


def _send_msg(c: socket.socket, payload: bytes) -> None:
    c.sendall(struct.pack("<Q", len(payload)) + payload)


class FakeRpcServer:
    """One client at a time, like ggml_backend_rpc_start_server."""

    def __init__(self, *, version=(8, 0, 0), free=8 << 30, total=16 << 30, n_devices=1, alloc_fail=False,
                 short_alloc=False, stall_fence_after_set=False, fence_delay_s=0.0, reject_free=False):
        self.version, self.free, self.total, self.n_devices = version, free, total, n_devices
        self.alloc_fail, self.short_alloc, self.stall = alloc_fail, short_alloc, stall_fence_after_set
        self.fence_delay_s, self.reject_free, self.fence_delayed = fence_delay_s, reject_free, False
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.endpoint = f"127.0.0.1:{self.listener.getsockname()[1]}"
        self.buffers: dict[int, tuple[int, int]] = {}  # remote_ptr -> (base, size)
        self.allocs = 0
        self.commands: list[int] = []
        self.hello_caps: bytes | None = None
        self.cache_flags: list[int] = []
        self.writes: list[tuple[int, int, int]] = []  # (remote_ptr, offset inside the buffer, size)
        self.explicit_frees: list[int] = []
        self.freed_on_disconnect: list[int] = []
        self.rejected: list[str] = []
        self.aborted: list[str] = []
        self.bytes_after_stall = 0
        self.sessions_done = threading.Semaphore(0)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.listener.close()

    def wait_session(self, timeout: float = 5.0) -> None:
        assert self.sessions_done.acquire(timeout=timeout), "fake server session did not end"

    def _serve(self) -> None:
        while True:
            try:
                c, _ = self.listener.accept()
            except OSError:
                return
            c.settimeout(10)
            try:
                self._session(c)
            except OSError:
                pass
            finally:
                self.freed_on_disconnect.extend(self.buffers)  # ~rpc_server frees what the connection allocated
                self.buffers.clear()
                c.close()
                self.sessions_done.release()

    def _session(self, c: socket.socket) -> None:
        head = _recv_exact(c, 9)
        if head is None or head[0] != linkprobe.CMD_HELLO or struct.unpack("<Q", head[1:])[0] != 24:
            self.rejected.append("hello")
            return
        self.hello_caps = _recv_exact(c, 24)
        _send_msg(c, struct.pack("<BBBB24s", *self.version, 0, bytes(24)))
        handlers = {0: self._alloc, 3: self._base, 4: self._free, 6: self._set_tensor, 11: self._device_memory}
        while True:
            cmd = c.recv(1)
            if not cmd:
                return
            self.commands.append(cmd[0])
            handler = handlers.get(cmd[0])
            if handler is None or not handler(c):
                return

    def _fixed(self, c: socket.socket, size: int, name: str) -> bytes | None:
        head = _recv_exact(c, 8)
        if head is None or struct.unpack("<Q", head)[0] != size:
            self.rejected.append(f"{name} size")
            return None
        return _recv_exact(c, size)

    def _device_memory(self, c: socket.socket) -> bool:
        req = self._fixed(c, 4, "device_memory")
        if req is None or struct.unpack("<I", req)[0] >= self.n_devices:
            return False
        if self.stall and self.writes:
            while (data := c.recv(65536)):
                self.bytes_after_stall += len(data)
            return False
        if self.fence_delay_s and self.writes and not self.fence_delayed:  # a relay still draining the upload
            self.fence_delayed = True
            time.sleep(self.fence_delay_s)
        _send_msg(c, struct.pack("<QQ", self.free, self.total))
        return True

    def _alloc(self, c: socket.socket) -> bool:
        req = self._fixed(c, 12, "alloc")
        if req is None:
            return False
        device, size = struct.unpack("<IQ", req)
        if device >= self.n_devices:
            return False
        self.allocs += 1
        if self.alloc_fail:
            _send_msg(c, struct.pack("<QQ", 0, 0))
            return True
        ptr, base = 0x5000 + 0x10 * self.allocs, 0x10_0000_0000 * self.allocs
        actual = size - 1 if self.short_alloc else size
        self.buffers[ptr] = (base, actual)
        _send_msg(c, struct.pack("<QQ", ptr, actual))
        return True

    def _base(self, c: socket.socket) -> bool:
        req = self._fixed(c, 8, "base")
        ptr = struct.unpack("<Q", req)[0] if req else None
        if ptr not in self.buffers:
            return False
        _send_msg(c, struct.pack("<Q", self.buffers[ptr][0]))
        return True

    def _free(self, c: socket.socket) -> bool:
        req = self._fixed(c, 8, "free")
        ptr = struct.unpack("<Q", req)[0] if req else None
        if ptr not in self.buffers or self.reject_free:
            return False
        del self.buffers[ptr]
        self.explicit_frees.append(ptr)
        return True  # FREE_BUFFER has no response

    def _set_tensor(self, c: socket.socket) -> bool:
        head = _recv_exact(c, 8)
        payload = _recv_exact(c, struct.unpack("<Q", head)[0]) if head else None
        if payload is None or len(payload) < TENSOR_END + 9:
            return False
        ttype = struct.unpack_from("<I", payload, OFF_TYPE)[0]
        buf = struct.unpack_from("<Q", payload, OFF_BUFFER)[0]
        ne = struct.unpack_from("<4I", payload, OFF_NE)
        nb = struct.unpack_from("<4I", payload, OFF_NB)
        data = struct.unpack_from("<Q", payload, OFF_DATA)[0]
        cache_flag, offset = struct.unpack_from("<BQ", payload, TENSOR_END)
        size = len(payload) - TENSOR_END - 9
        if ttype not in TYPE_SIZE or buf not in self.buffers:
            self.rejected.append("tensor")
            return False
        p0, length = self.buffers[buf]
        nbytes = 0 if min(ne) <= 0 else TYPE_SIZE[ttype] + sum((ne[i] - 1) * nb[i] for i in range(4))
        if nbytes and (data < p0 or data + nbytes > p0 + length):  # deserialize_tensor
            self.rejected.append("tensor bounds")
            return False
        if data + offset < p0 or data + offset >= p0 + length or size > p0 + length - data - offset:  # set_tensor
            self.rejected.append("data bounds")
            return False
        self.cache_flags.append(cache_flag)
        if offset + size > nbytes:  # GGML_ASSERT in ggml_backend_tensor_set: the real server process aborts
            self.aborted.append("tensor write out of bounds")
            return False
        self.writes.append((buf, data - p0 + offset, size))
        return True


@pytest.fixture
def fake():
    servers: list[FakeRpcServer] = []

    def make(**kw) -> FakeRpcServer:
        servers.append(FakeRpcServer(**kw))
        return servers[-1]

    yield make
    for s in servers:
        s.close()


# --------------------------------------------------------------------------------------------------- wire format


def test_rpc_tensor_matches_the_packed_c_layout():
    # 8 + 4 + 8 + 16 + 16 + 4 + 64 + 4 + 80 + 8 + 8 + 8 + 64 + 4 = 296 (see the offsets table above)
    assert linkprobe.RPC_TENSOR.size == linkprobe.RPC_TENSOR_SIZE == TENSOR_END == 296
    assert linkprobe.RPC_TENSOR_SIZE % 8 == 0  # static_assert in ggml-rpc.cpp
    assert linkprobe.SET_TENSOR_HEADER_SIZE == 296 + 1 + 8
    assert linkprobe.HELLO_RSP.size == 4 + 24
    assert linkprobe.ALLOC_REQ.size == 4 + 8
    assert linkprobe.REQUEST_HEADER.size == 1 + 8
    raw = linkprobe.encode_tensor_1d(buffer=0xABC, data=0x1234_5678_9000, nbytes=4096)
    assert struct.unpack_from("<I", raw, OFF_TYPE)[0] == linkprobe.GGML_TYPE_I8 == 24
    assert struct.unpack_from("<Q", raw, OFF_BUFFER)[0] == 0xABC
    assert struct.unpack_from("<4I", raw, OFF_NE) == (4096, 1, 1, 1)
    assert struct.unpack_from("<4I", raw, OFF_NB) == (1, 4096, 4096, 4096)
    assert struct.unpack_from("<Q", raw, OFF_DATA)[0] == 0x1234_5678_9000
    assert raw[OFF_NAME:OFF_NAME + 64].rstrip(b"\0") == b"hc_linkprobe"


def test_hello_frame_and_set_tensor_cache_flag():
    assert linkprobe.encode_request(14, bytes(24)) == b"\x0e" + (24).to_bytes(8, "little") + bytes(24)
    header = linkprobe.encode_set_tensor_header(buffer=1, data=2, nbytes=3)
    assert len(header) == 305
    assert header[296] == 0  # cache_flag
    assert struct.unpack_from("<Q", header, 297)[0] == 0  # offset


def test_encode_rejects_sizes_a_uint32_ne_cannot_hold():
    with pytest.raises(ValueError):
        linkprobe.encode_tensor_1d(buffer=1, data=2, nbytes=0)
    with pytest.raises(ValueError):
        linkprobe.encode_tensor_1d(buffer=1, data=2, nbytes=1 << 32)


@pytest.mark.parametrize("endpoint,expected", [("127.0.0.1:52110", ("127.0.0.1", 52110)),
                                               ("[::1]:47052", ("::1", 47052))])
def test_parse_endpoint(endpoint, expected):
    assert linkprobe.parse_endpoint(endpoint) == expected


@pytest.mark.parametrize("endpoint", ["127.0.0.1", ":80", "host:0", "host:70000", "host:x"])
def test_parse_endpoint_rejects(endpoint):
    with pytest.raises(ValueError):
        linkprobe.parse_endpoint(endpoint)


# --------------------------------------------------------------------------------------------------- rtt


def test_probe_rtt_reports_round_trips_and_device_memory(fake):
    srv = fake(free=3 << 30, total=8 << 30)
    r = linkprobe.probe_rtt(srv.endpoint, samples=7, path_class="loopback")
    srv.wait_session()
    assert srv.hello_caps == bytes(24)
    assert srv.commands == [11] * 8  # one uncounted warmup, then seven samples
    assert len(r.samples_ms) == 7 and r.median_ms <= r.p90_ms <= max(r.samples_ms)
    assert (r.free_bytes, r.total_bytes, r.server_version) == (3 << 30, 8 << 30, (8, 0, 0))
    coef = r.to_coefficient()
    assert (coef.provenance, coef.unit, coef.samples) == ("measured", "ms", 7)
    assert srv.endpoint in coef.note and "loopback" in coef.note
    profile = r.to_link_profile()
    assert profile.rtt_ms.provenance == "measured"
    assert profile.goodput_bytes_per_s.provenance == "prior"
    assert profile.round_trips_per_token.provenance == "prior"


@pytest.mark.parametrize("version", [(9, 0, 0), (7, 1, 0)])
def test_major_version_mismatch_is_refused(fake, version):
    srv = fake(version=version)
    with pytest.raises(ProtocolMismatch, match="speaks RPC protocol"):
        linkprobe.probe_rtt(srv.endpoint, samples=2, path_class="lan")
    srv.wait_session()
    assert srv.commands == []


def test_newer_minor_of_major_8_is_accepted(fake):
    srv = fake(version=(8, 2, 1))
    assert linkprobe.probe_rtt(srv.endpoint, samples=1, path_class="lan").server_version == (8, 2, 1)


def test_connection_refused_is_reported():
    probe = socket.create_server(("127.0.0.1", 0))
    endpoint = f"127.0.0.1:{probe.getsockname()[1]}"
    probe.close()
    with pytest.raises(LinkProbeError, match="connect to"):
        linkprobe.probe_rtt(endpoint, samples=1, path_class="relay", timeout_s=2)


def test_unknown_device_drops_the_connection(fake):
    srv = fake(n_devices=1)
    with pytest.raises(LinkProbeError, match="GET_DEVICE_MEMORY"):
        linkprobe.probe_rtt(srv.endpoint, samples=1, path_class="lan", device=3)


def test_path_class_is_required_and_checked(fake):
    srv = fake()
    with pytest.raises(ValueError, match="path_class"):
        linkprobe.probe_rtt(srv.endpoint, samples=1, path_class="local")
    with pytest.raises(TypeError):
        linkprobe.probe_rtt(srv.endpoint, samples=1)


# --------------------------------------------------------------------------------------------------- goodput


def test_goodput_uploads_every_byte_with_cache_flag_zero_and_frees(fake):
    srv = fake()
    total = 3 * MIB + 5
    m = linkprobe.probe_link(srv.endpoint, path_class="loopback", total_bytes=total, chunk_bytes=MIB, samples=3)
    srv.wait_session()
    (ptr,) = srv.explicit_frees
    assert sorted(w[1:] for w in srv.writes) == [(0, MIB), (MIB, MIB), (2 * MIB, MIB), (3 * MIB, 5)]
    assert {w[0] for w in srv.writes} == {ptr}
    assert srv.cache_flags == [0, 0, 0, 0]
    assert srv.aborted == [] and srv.rejected == [] and srv.freed_on_disconnect == []
    g = m.goodput
    assert g is not None and g.chunks == 4 and g.total_bytes == total and g.free_confirmed
    assert g.transfer_s == pytest.approx(g.elapsed_s - m.rtt.median_ms / 1000)
    assert g.bytes_per_s == pytest.approx(total / g.transfer_s)
    profile = m.to_link_profile(cached_bytes=7)
    assert profile.path == "loopback" and profile.cached_bytes == 7
    assert profile.goodput_bytes_per_s.provenance == "measured"
    assert profile.goodput_bytes_per_s.unit == "bytes/s" and profile.rtt_ms.unit == "ms"
    assert profile.round_trips_per_token.provenance == "prior"


def test_probe_goodput_alone(fake):
    srv = fake()
    g = linkprobe.probe_goodput(srv.endpoint, 2 * MIB, path_class="lan", rtt_samples=2)
    srv.wait_session()
    assert g.chunks == 1 and g.chunk_bytes == 2 * MIB and len(srv.explicit_frees) == 1


@pytest.mark.parametrize("probe", ["goodput", "link"])
def test_goodput_refuses_more_than_64_mib_before_connecting(fake, probe):
    srv = fake(free=1 << 40)
    with pytest.raises(ProbeRefused, match="cap"):
        if probe == "goodput":
            linkprobe.probe_goodput(srv.endpoint, 64 * MIB + 1, path_class="lan", rtt_samples=1)
        else:
            linkprobe.probe_link(srv.endpoint, path_class="lan", total_bytes=64 * MIB + 1, samples=1)
    assert not srv.sessions_done.acquire(timeout=0.3)  # the peer never saw a connection
    assert srv.hello_caps is None and srv.allocs == 0


def test_fence_wait_is_allowed_one_timeout_per_chunk(fake):
    # a local bridge takes every chunk at once, then the fence waits for the relay to drain the whole upload
    srv = fake(fence_delay_s=0.8)
    with linkprobe.RpcConnection.open(srv.endpoint, timeout_s=0.4) as conn:
        rtt = linkprobe.measure_rtt(conn, 2, path_class="relay")
        g = linkprobe.measure_goodput(conn, rtt, 3 * MIB, MIB)  # fence budget 0.4 s * (3 + 1)
        assert conn._sock is not None and conn._sock.gettimeout() == pytest.approx(0.4)  # restored
    srv.wait_session()
    assert g.free_confirmed and g.elapsed_s >= 0.8 and srv.freed_on_disconnect == []


def test_unconfirmed_free_closes_the_connection_and_says_so(fake):
    srv = fake(reject_free=True)
    g = linkprobe.probe_goodput(srv.endpoint, MIB, path_class="lan", rtt_samples=1)
    srv.wait_session()
    assert g.free_confirmed is False
    assert srv.explicit_frees == [] and len(srv.freed_on_disconnect) == 1


@pytest.mark.parametrize("timeout", [math.inf, math.nan, 0.0, -1.0])
def test_socket_timeout_must_be_finite_and_positive(fake, timeout):
    srv = fake()
    with pytest.raises(ValueError, match="timeout_s"):
        linkprobe.RpcConnection.open(srv.endpoint, timeout_s=timeout)


def test_goodput_refuses_more_than_a_quarter_of_free_memory(fake):
    srv = fake(free=16 * MIB)
    with pytest.raises(ProbeRefused, match="25%"):
        linkprobe.probe_goodput(srv.endpoint, 4 * MIB + 1, path_class="lan", rtt_samples=1)
    srv.wait_session()
    assert srv.allocs == 0
    linkprobe.probe_goodput(srv.endpoint, 4 * MIB, path_class="lan", rtt_samples=1)  # exactly a quarter is allowed
    srv.wait_session()
    assert srv.allocs == 1 and len(srv.explicit_frees) == 1


def test_buffer_is_freed_when_the_payload_fails(fake):
    srv = fake()

    def broken(n: int) -> bytes:
        raise RuntimeError("entropy source failed")

    with pytest.raises(RuntimeError, match="entropy"):
        linkprobe.probe_goodput(srv.endpoint, MIB, path_class="lan", rtt_samples=1, payload=broken)
    srv.wait_session()
    assert len(srv.explicit_frees) == 1 and srv.freed_on_disconnect == []


def test_buffer_is_freed_when_the_payload_is_short(fake):
    srv = fake()
    with pytest.raises(LinkProbeError, match="payload source"):
        linkprobe.probe_goodput(srv.endpoint, MIB, path_class="lan", rtt_samples=1, payload=lambda n: b"x")
    srv.wait_session()
    assert len(srv.explicit_frees) == 1 and srv.writes == []


def test_timeout_closes_the_out_of_sync_stream_and_the_server_frees_the_buffer(fake):
    srv = fake(stall_fence_after_set=True)
    with pytest.raises(LinkProbeError, match="GET_DEVICE_MEMORY reply"):
        linkprobe.probe_goodput(srv.endpoint, MIB, path_class="relay", rtt_samples=1, timeout_s=0.5)
    srv.wait_session()
    assert srv.explicit_frees == []  # no FREE_BUFFER written into a stream that is out of sync
    assert srv.bytes_after_stall == 0
    assert len(srv.freed_on_disconnect) == 1 and srv.buffers == {}


def test_failed_allocation_raises_and_frees_nothing(fake):
    srv = fake(alloc_fail=True)
    with pytest.raises(LinkProbeError, match="ALLOC_BUFFER"):
        linkprobe.probe_goodput(srv.endpoint, MIB, path_class="lan", rtt_samples=1)
    srv.wait_session()
    assert srv.explicit_frees == [] and srv.freed_on_disconnect == []


def test_short_allocation_is_freed(fake):
    srv = fake(short_alloc=True)
    with pytest.raises(LinkProbeError, match="allocated"):
        linkprobe.probe_goodput(srv.endpoint, MIB, path_class="lan", rtt_samples=1)
    srv.wait_session()
    assert len(srv.explicit_frees) == 1 and srv.writes == []


def test_fake_server_enforces_the_real_bounds_checks(fake):
    srv = fake()
    with linkprobe.RpcConnection.open(srv.endpoint, timeout_s=2) as conn:
        ptr, size = conn.alloc_buffer(0, 4096)
        base = conn.buffer_base(ptr)
        conn.set_tensor(buffer=ptr, data_ptr=base + size - 10, data=bytes(20))  # past the buffer end
        with pytest.raises(LinkProbeError):
            conn.device_memory(0)
    srv.wait_session()
    assert srv.rejected == ["tensor bounds"] and srv.freed_on_disconnect == [ptr]


def test_fake_server_reproduces_the_tensor_set_assertion(fake):
    srv = fake()
    with linkprobe.RpcConnection.open(srv.endpoint, timeout_s=2) as conn:
        ptr, _ = conn.alloc_buffer(0, 4096)
        base = conn.buffer_base(ptr)
        header = linkprobe.encode_set_tensor_header(buffer=ptr, data=base, nbytes=10)
        conn._send(linkprobe.CMD_SET_TENSOR, header, bytes(20))  # 20 bytes into a 10-byte tensor view
        with pytest.raises(LinkProbeError):
            conn.device_memory(0)
    srv.wait_session()
    assert srv.aborted == ["tensor write out of bounds"]


def test_an_interrupted_send_marks_the_stream_out_of_sync():
    class InterruptedSocket:
        def sendall(self, data):
            raise KeyboardInterrupt

        def close(self):
            pass

    conn = linkprobe.RpcConnection(InterruptedSocket(), "127.0.0.1:1")
    with pytest.raises(KeyboardInterrupt):
        conn.device_memory(0)
    assert not conn.in_sync
    with pytest.raises(LinkProbeError, match="out of sync"):
        conn.free_buffer(1)  # nothing more is written into a partial frame


# --------------------------------------------------------------------------------------------------- script


PEER = "4c2991743c154be896d10a0b53ba4d2b"  # a NodeView id: link records are keyed by the peer's node id


def _load_script():
    spec = importlib.util.spec_from_file_location("probe_link_script", ROOT / "scripts" / "capacity" / "probe_link.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_evidence_and_stores_measurements(fake, tmp_path, capsys):
    srv = fake()
    script = _load_script()
    code = script.main(["--endpoint", srv.endpoint, "--path-class", "loopback", "--bytes", str(2 * MIB),
                        "--chunk-bytes", str(MIB), "--samples", "3", "--results-dir", str(tmp_path / "res"),
                        "--store", "--store-dir", str(tmp_path / "store"), "--node-id", PEER])
    assert code == 0
    (link_json,) = (tmp_path / "res").glob("*-link-loopback-127.0.0.1-*/link.json")
    doc = json.loads(link_json.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["goodput"]["chunks"] == 2 and len(doc["rtt"]["samples_ms"]) == 3
    assert doc["link_profile"]["goodput_bytes_per_s"]["provenance"] == "measured"
    assert doc["link_profile"]["round_trips_per_token"]["provenance"] == "prior"
    store = MeasurementStore(tmp_path / "store")
    ident = MeasurementIdentity(node_id=PEER, device="rpc0", backend_build="ggml-rpc-8.0.0",
                                path_class="loopback", kind="link.goodput_bytes_per_s")
    rec = store.latest(ident, max_age_s=600)
    assert rec is not None and rec.value == pytest.approx(doc["goodput"]["bytes_per_s"])
    assert rec.evidence == str(link_json)
    assert "rtt" in capsys.readouterr().out


def test_script_records_a_failed_probe(tmp_path):
    probe = socket.create_server(("127.0.0.1", 0))
    endpoint = f"127.0.0.1:{probe.getsockname()[1]}"
    probe.close()
    script = _load_script()
    code = script.main(["--endpoint", endpoint, "--path-class", "relay", "--timeout", "2",
                        "--results-dir", str(tmp_path), "--label", "dead"])
    assert code == 2
    (link_json,) = tmp_path.glob("*-link-dead/link.json")
    doc = json.loads(link_json.read_text(encoding="utf-8"))
    assert doc["ok"] is False and "connect to" in doc["error"] and "rtt" not in doc


def test_script_store_requires_node_id():
    script = _load_script()
    with pytest.raises(SystemExit):
        script.parse_args(["--endpoint", "127.0.0.1:1", "--path-class", "lan", "--store"])


def test_script_rejects_bad_store_identity_before_probing(fake):
    srv = fake()
    script = _load_script()
    with pytest.raises(SystemExit):
        script.main(["--endpoint", srv.endpoint, "--path-class", "lan", "--store", "--node-id", PEER,
                     "--power-state", ""])
    assert not srv.sessions_done.acquire(timeout=0.3) and srv.hello_caps is None


def test_script_keeps_the_evidence_when_the_store_fails(fake, tmp_path):
    srv = fake()
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    script = _load_script()
    code = script.main(["--endpoint", srv.endpoint, "--path-class", "loopback", "--bytes", str(MIB),
                        "--samples", "2", "--results-dir", str(tmp_path / "res"), "--label", "lost",
                        "--store", "--store-dir", str(blocker), "--node-id", PEER])
    assert code == 3
    (link_json,) = (tmp_path / "res").glob("*-link-lost/link.json")
    doc = json.loads(link_json.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["goodput"]["total_bytes"] == MIB and "stored" not in doc
    assert doc["store_error"]  # the OSError subclass differs by platform


@pytest.mark.parametrize("node_id", ["mac", "node-a", "local", "4C2991743C154BE896D10A0B53BA4D2B", "a" * 31,
                                     PEER + "\n"])  # re's $ matches before a final newline; the id must not
def test_script_refuses_ids_that_are_not_peer_node_ids_before_probing(fake, node_id):
    srv = fake()
    script = _load_script()
    with pytest.raises(SystemExit):
        script.main(["--endpoint", srv.endpoint, "--path-class", "relay", "--store", "--node-id", node_id])
    assert not srv.sessions_done.acquire(timeout=0.3) and srv.hello_caps is None


def test_script_records_the_node_id_in_the_evidence(fake, tmp_path):
    srv = fake()
    script = _load_script()
    code = script.main(["--endpoint", srv.endpoint, "--path-class", "relay", "--bytes", str(MIB), "--samples", "2",
                        "--results-dir", str(tmp_path / "res"), "--store", "--store-dir", str(tmp_path / "store"),
                        "--node-id", PEER])
    assert code == 0
    (link_json,) = (tmp_path / "res").glob("*-link-relay-*/link.json")
    assert json.loads(link_json.read_text(encoding="utf-8"))["node_id"] == PEER
    profile = bridge.link_profile(MeasurementStore(tmp_path / "store").records(), PEER, "relay", datetime.now(UTC))
    assert profile.rtt_ms.provenance == "measured" and profile.goodput_bytes_per_s.provenance == "measured"
