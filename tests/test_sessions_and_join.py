"""Invitation codes, rate limiting, credential isolation and leave (protocol tests)."""

from datetime import timedelta

import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute.schemas import HeartbeatRequest, JoinRequest, NodeState, WorkerEndpoint, WorkerState, utcnow
from hypercompute.services.nodes import AuthError, NodeRegistry
from hypercompute.services.sessions import JoinRejected, RateLimiter, SessionService, normalize_code

SECRET = b"k" * 32


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def sessions(store, events, clock=None):
    return SessionService(store, events, SECRET, 8000, lambda: ["10.0.0.2"], invite_ttl_s=600,
                          clock=clock or utcnow)


def registry(store, events, svc, probe_ok=True):
    reg = NodeRegistry(store, events, svc, SECRET, engine_info(), make_inventory(), contribution(["CUDA0"]))

    async def probe(endpoints):
        from hypercompute.adapters.llama_rpc import ProbeResult
        from hypercompute.schemas import EngineDevice

        if not probe_ok:
            return ProbeResult(False, [], {endpoints[0]: []}, "a worker endpoint refused or dropped the RPC connection")
        dev = EngineDevice(name="RPC0", description=endpoints[0], total_bytes=16 << 30, free_bytes=10 << 30)
        return ProbeResult(True, [dev], {endpoints[0]: [dev]}, "worker answered the llama.cpp RPC handshake")

    reg.probe = probe
    return reg


def join_request(code, node_id="a" * 32, worker_ip="10.0.0.5", release="b11770", rpc_proto="8.0.0"):
    inv = make_inventory("macos", engine=engine_info(release=release, rpc_proto=rpc_proto, accelerator="metal"))
    return JoinRequest(code=code, node_id=node_id, display_name="friend-mac", inventory=inv,
                       contribution=contribution(["MTL0"]), worker=WorkerEndpoint(host=worker_ip, port=47052),
                       agent_version="0.1.0")


def test_invite_is_one_time(store, events):
    svc = sessions(store, events)
    code = svc.create("demo").invite.code
    assert svc.redeem(code, "10.0.0.5")
    with pytest.raises(JoinRejected) as e:
        svc.redeem(code, "10.0.0.6")
    assert e.value.code == "code_used"


def test_invite_expires(store, events, now):
    clock = Clock(now)
    svc = sessions(store, events, clock)
    code = svc.create("demo").invite.code
    clock.t = now + timedelta(seconds=601)
    with pytest.raises(JoinRejected) as e:
        svc.redeem(code, "10.0.0.5")
    assert e.value.code == "code_expired"


def test_wrong_code_and_normalization(store, events):
    svc = sessions(store, events)
    code = svc.create("demo").invite.code
    with pytest.raises(JoinRejected) as e:
        svc.redeem("ABCDE-FGHJK", "10.0.0.5")
    assert e.value.code == "invalid_code"
    assert svc.redeem(code.lower().replace("-", " "), "10.0.0.5")  # case and separators are forgiving


def test_codes_are_stored_hashed(store, events):
    svc = sessions(store, events)
    code = svc.create("demo").invite.code
    rows = store.query("SELECT code_hash FROM invites")
    assert rows and normalize_code(code) not in rows[0]["code_hash"]


def test_rate_limit_blocks_bruteforce(store, events):
    svc = sessions(store, events)
    svc.create("demo")
    for _ in range(10):
        with pytest.raises(JoinRejected):
            svc.redeem("ZZZZZ-ZZZZZ", "10.0.0.66")
    with pytest.raises(JoinRejected) as e:
        svc.redeem("ZZZZZ-ZZZZZ", "10.0.0.66")
    assert e.value.code == "rate_limited"


def test_rate_limiter_window():
    t = [0.0]
    rl = RateLimiter(2, window_s=60, clock=lambda: t[0])
    assert rl.allow("x") and rl.allow("x") and not rl.allow("x")
    t[0] = 61
    assert rl.allow("x")


def test_closed_session_rejects_codes(store, events):
    svc = sessions(store, events)
    view = svc.create("demo")
    svc.close(view.id)
    with pytest.raises(JoinRejected):
        svc.redeem(view.invite.code, "10.0.0.5")


@pytest.mark.anyio
async def test_join_verify_credential_isolation_and_leave(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    a = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", "10.0.0.2:8000")
    b = await reg.join(join_request(svc.issue_invite(svc.current().id).invite.code, node_id="b" * 32,
                                    worker_ip="10.0.0.6"), "10.0.0.6", "10.0.0.2:8000")
    await reg.verify(a.node_id)
    assert reg.get(a.node_id).state == NodeState.eligible
    reg.authenticate(a.node_id, f"Bearer {a.credential}")
    with pytest.raises(AuthError):  # node B's credential cannot act as node A
        reg.authenticate(a.node_id, f"Bearer {b.credential}")
    with pytest.raises(AuthError):
        reg.authenticate(a.node_id, None)
    await reg.leave(a.node_id, by_owner=False)
    assert reg.get(a.node_id).state == NodeState.left
    with pytest.raises(AuthError):  # credential revoked on leave
        reg.authenticate(a.node_id, f"Bearer {a.credential}")


@pytest.mark.anyio
async def test_join_rejects_spoofed_worker_address(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    with pytest.raises(JoinRejected) as e:
        await reg.join(join_request(svc.create("d").invite.code, worker_ip="10.0.0.99"), "10.0.0.5", None)
    assert e.value.code == "worker_address_mismatch"


@pytest.mark.anyio
async def test_join_rejects_public_worker_address(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    with pytest.raises(JoinRejected) as e:
        await reg.join(join_request(svc.create("d").invite.code, worker_ip="8.8.8.8"), "8.8.8.8", None)
    assert e.value.code == "worker_not_private"


@pytest.mark.anyio
async def test_join_accepts_link_local_worker_on_a_direct_cable(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    resp = await reg.join(join_request(svc.create("d").invite.code, worker_ip="169.254.10.3"), "169.254.10.3", None)
    assert resp.node_id == "a" * 32


@pytest.mark.anyio
async def test_engine_mismatch_is_rejected_before_any_probe(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    probed = []
    original = reg.probe

    async def spy(endpoints):
        probed.append(endpoints)
        return await original(endpoints)

    reg.probe = spy
    j = await reg.join(join_request(svc.create("d").invite.code, rpc_proto="7.1.0"), "10.0.0.5", None)
    view = await reg.verify(j.node_id)
    assert view.state == NodeState.ineligible and "RPC protocol" in view.state_reason
    assert probed == []


@pytest.mark.anyio
async def test_unreachable_worker_is_ineligible_with_fix(store, events):
    svc = sessions(store, events)
    reg = registry(store, events, svc, probe_ok=False)
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", "10.0.0.2:8000")
    view = await reg.verify(j.node_id)
    assert view.state == NodeState.ineligible
    assert view.verification and view.verification.fix
    assert "-RemoteAddress 10.0.0.2" not in view.verification.fix  # macOS friend: no Windows rule
    assert "allow incoming connections" in view.verification.fix and "client isolation" in view.verification.fix


@pytest.mark.anyio
async def test_stale_node_and_worker_exit(store, events, now):
    svc = sessions(store, events)
    reg = registry(store, events, svc)
    lost = []

    async def on_lost(node_id, reason):
        lost.append((node_id, reason))

    reg.on_lost = on_lost
    reg.busy = lambda node_id: ["dep1"]
    j = await reg.join(join_request(svc.create("d").invite.code), "10.0.0.5", None)
    await reg.verify(j.node_id)
    stale = await reg.check_staleness(utcnow() + timedelta(seconds=30))
    assert stale == [j.node_id] and reg.get(j.node_id).state == NodeState.stale
    assert lost and "no heartbeat" in lost[0][1]
    hb = HeartbeatRequest(worker_state=WorkerState.exited, worker_exit_code=3, sent_at=utcnow())
    resp = await reg.heartbeat(j.node_id, hb)
    assert resp.state == NodeState.ineligible and "exit code 3" in resp.state_reason
