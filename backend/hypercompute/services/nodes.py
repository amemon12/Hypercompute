"""Node registry: join, credentials, heartbeats, staleness, verification and leave."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from .. import config, netutil
from ..adapters.llama_rpc import ProbeResult
from ..schemas import (
    HCPEER_REPORT_FEATURE,
    MANAGED_PEERS_FEATURE,
    ContributionLimits,
    CoordinatorEngineIdentity,
    EngineInfo,
    HeartbeatRequest,
    HeartbeatResponse,
    HostInventory,
    JoinRequest,
    JoinResponse,
    ManagedReport,
    NodeState,
    NodeView,
    VerificationResult,
    WorkerEndpoint,
    WorkerState,
    utcnow,
)
from ..storage import Store, dumps
from .events import EventBus
from .peer_seed import ManagedPeers
from .sessions import JoinRejected, SessionService, hash_secret, new_credential

log = logging.getLogger("hypercompute.nodes")
LOCAL_ID = "local"
ACTIVE_STATES = {NodeState.verifying, NodeState.eligible, NodeState.ineligible}
HELD_STATES = ACTIVE_STATES | {NodeState.stale}  # the node id still belongs to a machine in the session
UNREACHABLE_REPROBE_S = 30.0  # re-probe a heartbeating but unreachable worker (its firewall may be fixed)

ProbeFn = Callable[[list[str]], Awaitable[ProbeResult]]
LostFn = Callable[[str, str], Awaitable[None]]
TransportRevokeFn = Callable[[str, bool], Awaitable[None]]


class AuthError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _log_task_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("background node task failed", exc_info=task.exception())


def firewall_fix(os_name: str, port: int, coordinator_ip: str | None) -> str:
    src = coordinator_ip or "<host LAN IP>"
    if os_name == "windows":
        rule = (f'New-NetFirewallRule -DisplayName "Hypercompute RPC worker" -Direction Inbound -Protocol TCP '
                f"-LocalPort {port} -RemoteAddress {src} -Action Allow")
        how = f"On the friend's Windows machine, in an Administrator PowerShell: {rule}"
    elif os_name == "macos":
        how = ("On the friend's Mac, allow incoming connections for ggml-rpc-server when macOS asks "
               "(System Settings > Network > Firewall > Options).")
    else:
        how = f"On the friend's Linux machine: sudo ufw allow from {src} to any port {port} proto tcp"
    return (how + " If both machines are on guest/campus Wi-Fi, client isolation can block peer traffic: "
            "use a hotspot, a travel router or Ethernet.")


class NodeRegistry:
    def __init__(self, store: Store, events: EventBus, sessions: SessionService, secret: bytes,
                 coordinator_engine: EngineInfo, local_inventory: HostInventory,
                 local_contribution: ContributionLimits) -> None:
        self._store, self._events, self._sessions, self._secret = store, events, sessions, secret
        self._engine = coordinator_engine
        self._nodes: dict[str, NodeView] = {}
        self._cred: dict[str, str] = {}
        self._coord_addr: dict[str, str] = {}
        self._verify_tasks: dict[str, asyncio.Task] = {}
        self._lost_tasks: dict[str, asyncio.Task] = {}
        self.probe: ProbeFn | None = None
        self.on_lost: LostFn | None = None
        self.on_transport_revoke: TransportRevokeFn | None = None
        self.busy: Callable[[str], list[str]] = lambda node_id: []
        self.managed = ManagedPeers()  # commands for peers that joined with --managed (in memory)
        self._local = self._make_local(local_inventory, local_contribution)

    # ------------------------------------------------------------------ local node
    def _make_local(self, inv: HostInventory, contribution: ContributionLimits) -> NodeView:
        has_accel = any(d.name.startswith(("CUDA", "MTL", "Vulkan", "ROCm")) for d in inv.engine.devices)
        if inv.engine.smoke != "passed":
            state, reason = NodeState.ineligible, f"engine smoke test: {inv.engine.smoke_detail}"
        elif not has_accel:
            state, reason = NodeState.eligible, "no local accelerator visible to llama.cpp; host is CPU only"
        else:
            state, reason = NodeState.eligible, None
        now = utcnow()
        return NodeView(id=LOCAL_ID, display_name=f"{inv.hostname} (this machine)", is_local=True, session_id=None,
                        state=state, state_reason=reason, joined_at=now, last_heartbeat_at=now, heartbeat_age_s=0.0,
                        inventory=inv, contribution=contribution, worker=None, worker_state=None,
                        verification=None)

    def update_local(self, inv: HostInventory) -> None:
        self._local = self._local.model_copy(update={"inventory": inv, "last_heartbeat_at": utcnow()})

    @property
    def local(self) -> NodeView:
        return self._local

    # ------------------------------------------------------------------ persistence
    def restore(self) -> list[str]:
        """Reload remote nodes after a gateway restart; they must heartbeat and re-verify before use."""
        notes = []
        for row in self._store.query("SELECT id, credential_hash, doc FROM nodes"):
            view = NodeView.model_validate(json.loads(row["doc"]))
            if view.state in HELD_STATES:
                view = view.model_copy(update={"state": NodeState.stale,
                                               "state_reason": "gateway restarted; waiting for heartbeat"})
                if row["credential_hash"]:
                    self._cred[view.id] = row["credential_hash"]
                notes.append(f"{view.display_name}: awaiting heartbeat after restart")
            self._nodes[view.id] = view
        return notes

    def _save(self, view: NodeView) -> None:
        self._nodes[view.id] = view
        self._store.execute(
            "INSERT INTO nodes (id, session_id, credential_hash, state, doc, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET session_id=excluded.session_id, credential_hash=excluded.credential_hash,"
            " state=excluded.state, doc=excluded.doc, updated_at=excluded.updated_at",
            (view.id, view.session_id, self._cred.get(view.id), view.state.value,
             dumps(view.model_dump(mode="json")), utcnow().isoformat()))

    def _set_state(self, view: NodeView, state: NodeState, reason: str | None, **extra: object) -> NodeView:
        changed = view.state != state or view.state_reason != reason
        view = view.model_copy(update={"state": state, "state_reason": reason, **extra})
        self._save(view)
        if changed:
            self._events.publish("node.state", f"{view.display_name}: {state.value}"
                                 + (f" ({reason})" if reason else ""), node_id=view.id, state=state.value)
        return view

    # ------------------------------------------------------------------ queries
    def views(self) -> list[NodeView]:
        now = utcnow()
        out = [self._local.model_copy(update={"heartbeat_age_s": 0.0})]
        for v in sorted(self._nodes.values(), key=lambda n: n.joined_at):
            if v.state in (NodeState.left, NodeState.removed):
                continue
            age = (now - v.last_heartbeat_at).total_seconds() if v.last_heartbeat_at else None
            out.append(v.model_copy(update={"heartbeat_age_s": age, "deployments": self.busy(v.id)}))
        return out

    def get(self, node_id: str) -> NodeView | None:
        return self._local if node_id == LOCAL_ID else self._nodes.get(node_id)

    def eligible_remote(self, node_ids: list[str] | None) -> list[NodeView]:
        pool = [v for v in self._nodes.values() if v.state == NodeState.eligible]
        if node_ids is not None:
            wanted = set(node_ids)
            pool = [v for v in pool if v.id in wanted]
        return pool

    # ------------------------------------------------------------------ join / auth
    async def join(self, req: JoinRequest, client_ip: str, host_header: str | None, *,
                   transport_peer_id: str | None = None,
                   transport_worker: Callable[[], Awaitable[WorkerEndpoint]] | None = None) -> JoinResponse:
        loopback_dev = client_ip in ("127.0.0.1", "::1") and req.worker.host == "127.0.0.1"
        if req.worker.host != client_ip and not loopback_dev:
            raise JoinRejected(400, "worker_address_mismatch",
                               f"advertised worker address {req.worker.host} does not match the address this "
                               f"request came from ({client_ip})",
                               fix=f"re-run join with --advertise-ip {client_ip}")
        if not loopback_dev and not netutil.is_lan_or_link_local(req.worker.host):
            raise JoinRejected(400, "worker_not_private", "worker must listen on a private LAN or link-local address",
                               fix="join over the local network or an existing private overlay")
        self._check_node_id_free(req, transport_peer_id)
        if transport_peer_id is not None and (not transport_peer_id or transport_worker is None):
            raise JoinRejected(400, "invalid_transport", "authenticated peer transport requires a local bridge")
        session_id = self._sessions.redeem(req.code, "iroh" if transport_peer_id else client_ip)
        if transport_worker is not None:
            if transport_peer_id is None:
                raise JoinRejected(400, "invalid_transport", "bridge requires an authenticated transport identity")
            # Allocate only after the one-use invitation was accepted. The caller's target is ignored.
            worker = await transport_worker()
            if worker.host != "127.0.0.1":
                raise JoinRejected(400, "invalid_transport", "peer bridge must listen on loopback")
            req = req.model_copy(update={"worker": worker,
                                         "inventory": req.inventory.model_copy(update={"extra_engines": []})})
            # The allocation yielded: a LAN join or owner session close may have run meanwhile.
            self._check_node_id_free(req, transport_peer_id)
            session = self._sessions.get(session_id)
            if session is None or session["status"] != "active":
                raise JoinRejected(403, "session_closed", "session was closed during peer admission")
        credential = new_credential()
        self._cred[req.node_id] = hash_secret(self._secret, credential)
        self.managed.forget(req.node_id)  # a new agent run: commands queued for the old one do not apply
        coord_ip = (host_header or "").rsplit(":", 1)[0] or None
        if coord_ip:
            self._coord_addr[req.node_id] = coord_ip
        now = utcnow()
        view = NodeView(id=req.node_id, display_name=req.display_name, is_local=False, session_id=session_id,
                        state=NodeState.verifying, state_reason="checking worker reachability and engine build",
                        joined_at=now, last_heartbeat_at=now, heartbeat_age_s=0.0, inventory=req.inventory,
                        contribution=req.contribution, worker=req.worker, worker_state=WorkerState.running,
                        verification=None, transport_peer_id=transport_peer_id)
        self._save(view)
        self._events.publish("node.joined", f"{req.display_name} joined", node_id=req.node_id,
                             os=req.inventory.os.system)
        self.schedule_verify(req.node_id)
        return JoinResponse(session_id=session_id, node_id=req.node_id, credential=credential,
                            heartbeat_interval_s=config.HEARTBEAT_INTERVAL_S,
                            coordinator_engine=CoordinatorEngineIdentity(release=self._engine.release,
                                                                         commit=self._engine.commit,
                                                                         rpc_proto=self._engine.rpc_proto),
                            state=view.state, features=[MANAGED_PEERS_FEATURE, HCPEER_REPORT_FEATURE])

    def _check_node_id_free(self, req: JoinRequest, transport_peer_id: str | None = None) -> None:
        """A node id held by a machine in the session may only rejoin from that machine (agent restart)."""
        existing = self._nodes.get(req.node_id)
        if existing is not None and existing.transport_peer_id is not None:
            if existing.transport_peer_id == transport_peer_id:
                return
            raise JoinRejected(409, "node_id_in_use", "this machine id belongs to another authenticated peer")
        if existing is None or existing.state not in HELD_STATES or req.node_id not in self._cred:
            return
        if transport_peer_id is None and existing.worker is not None and existing.worker.host == req.worker.host:
            return
        raise JoinRejected(409, "node_id_in_use", "this machine id already belongs to another machine in the session",
                           fix="the host removes this machine's old entry in the dashboard, then join again")

    def authenticate(self, node_id: str, authorization: str | None, *,
                     transport_peer_id: str | None = None) -> NodeView:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError(401, "missing node credential")
        expected = self._cred.get(node_id)
        given = hash_secret(self._secret, authorization[len("Bearer "):].strip())
        if expected is None or not hmac.compare_digest(expected, given):
            raise AuthError(403, "credential not valid for this node")
        view = self._nodes.get(node_id)
        if view is None or view.state in (NodeState.left, NodeState.removed):
            raise AuthError(403, "node is no longer part of the session")
        self._check_transport(view, transport_peer_id)
        session = self._sessions.get(view.session_id) if view.session_id else None
        if session is None or session["status"] != "active":
            raise AuthError(403, "session is no longer active")
        return view

    @staticmethod
    def _check_transport(view: NodeView, transport_peer_id: str | None) -> None:
        if view.transport_peer_id != transport_peer_id:
            raise AuthError(403, "node credential requires its authenticated transport identity")

    def attach_transport_worker(self, node_id: str, peer_id: str, worker: WorkerEndpoint) -> None:
        """Restore a loopback bridge after restart, after the service authenticates this peer."""
        view = self._nodes[node_id]
        self._check_transport(view, peer_id)
        if worker.host != "127.0.0.1":
            raise ValueError("transport worker must be loopback")
        self._save(view.model_copy(update={"worker": worker}))

    async def transport_disconnected(self, node_id: str, peer_id: str) -> None:
        view = self._nodes.get(node_id)
        if view is None or view.transport_peer_id != peer_id or view.state not in ACTIVE_STATES:
            return
        reason = "authenticated peer connection disconnected"
        self._set_state(view, NodeState.stale, reason)
        task = self._verify_tasks.get(node_id)
        if task is not None and not task.done():
            task.cancel()
        if self.on_transport_revoke is not None:
            await self.on_transport_revoke(node_id, False)
        self._start_lost(node_id, reason)

    # ------------------------------------------------------------------ heartbeat / staleness
    async def heartbeat(self, node_id: str, req: HeartbeatRequest, *,
                        transport_peer_id: str | None = None) -> HeartbeatResponse:
        view = self._nodes[node_id]
        self._check_transport(view, transport_peer_id)
        was_running = view.worker_state == WorkerState.running
        inv = view.inventory
        if req.memory_pools or req.devices:
            inv = inv.model_copy(update={"memory_pools": req.memory_pools or inv.memory_pools,
                                         "devices": req.devices or inv.devices})
        if transport_peer_id is not None:
            inv = inv.model_copy(update={"extra_engines": []})
        elif req.extra_engines is not None:  # EXO started, stopped or crashed since join
            inv = inv.model_copy(update={"extra_engines": req.extra_engines})
        view = view.model_copy(update={"last_heartbeat_at": utcnow(), "worker_state": req.worker_state,
                                       "worker_cpu_seconds": req.worker_cpu_seconds, "inventory": inv})
        if req.worker_state != WorkerState.running:
            reason = ("paused by contributor" if req.worker_state == WorkerState.paused else
                      f"worker {req.worker_state.value}" + (f" (exit code {req.worker_exit_code})"
                                                            if req.worker_exit_code is not None else ""))
            view = self._set_state(view, NodeState.ineligible, reason)
            if view.transport_peer_id and self.on_transport_revoke is not None:
                await self.on_transport_revoke(node_id, False)
            self._start_lost(node_id, reason)  # in the background: a slow cleanup must not stall this reply
        elif view.state == NodeState.stale:
            view = self._set_state(view, NodeState.verifying, "heartbeat resumed; re-checking worker")
            self.schedule_verify(node_id)
        elif view.state == NodeState.ineligible and not was_running:
            view = self._set_state(view, NodeState.verifying, "worker running again; re-checking")
            self.schedule_verify(node_id)
        else:  # unreachable workers are re-probed by the monitor on a backoff, not on every heartbeat
            self._save(view)
        commands, restarted = self.managed.on_heartbeat(node_id, req.managed)
        if req.managed is not None:
            view = self._apply_managed_worker(view, req.managed, restarted)
        return HeartbeatResponse(state=view.state, state_reason=view.state_reason, removed=False,
                                 deployments=self.busy(node_id), commands=commands)

    def _apply_managed_worker(self, view: NodeView, report: ManagedReport, restarted: bool) -> NodeView:
        """A host-requested restart changes the worker's -d list: the planner pairs the probe's RPC devices with
        contribution.devices by position, so both are refreshed together (re-verify) before the node is used."""
        devices = report.worker_devices
        if devices and devices != view.contribution.devices:
            view = view.model_copy(update={"contribution": view.contribution.model_copy(update={"devices": devices})})
            self._save(view)
            restarted = True
        if restarted and view.state == NodeState.eligible and not self.busy(view.id):
            view = self._set_state(view, NodeState.verifying, "worker restarted by the host; re-checking")
            self.schedule_verify(view.id)
        return view

    async def check_staleness(self, now: datetime | None = None) -> list[str]:
        now = now or utcnow()
        lost = []
        for view in list(self._nodes.values()):
            if view.state not in ACTIVE_STATES or view.last_heartbeat_at is None:
                continue
            age = (now - view.last_heartbeat_at).total_seconds()
            if age > config.STALE_AFTER_S:
                reason = f"no heartbeat for {age:.0f}s"
                self._set_state(view, NodeState.stale, reason)
                await self._lost(view.id, reason)
                lost.append(view.id)
        return lost

    def schedule_rechecks(self, now: datetime | None = None) -> list[str]:
        """Re-verify nodes a skipped or failed check left behind; returns the node ids scheduled."""
        now = now or utcnow()
        due = [v.id for v in list(self._nodes.values()) if self._recheck_due(v, now)]
        for node_id in due:
            self.schedule_verify(node_id)
        return due

    def _recheck_due(self, view: NodeView, now: datetime) -> bool:
        task = self._verify_tasks.get(view.id)
        if (task and not task.done()) or self.busy(view.id):
            return False
        if view.state == NodeState.verifying:
            return True  # verify skipped it while a deployment still held it; nothing else would retry
        v = view.verification
        return (view.state == NodeState.ineligible and view.worker_state == WorkerState.running
                and v is not None and v.engine_match and not v.reachable
                and (now - v.checked_at).total_seconds() >= UNREACHABLE_REPROBE_S)

    async def monitor(self) -> None:
        while True:
            try:
                await self.check_staleness()
                self.schedule_rechecks()
            except Exception:  # keep the monitor alive; the failure is logged with context
                log.exception("node monitor check failed")
            await asyncio.sleep(2.0)

    def _start_lost(self, node_id: str, reason: str) -> asyncio.Task | None:
        """Start loss handling for a node, or return the one already running (never two cleanups at once)."""
        task = self._lost_tasks.get(node_id)
        if task is not None and not task.done():
            return task
        if self.on_lost is None or not self.busy(node_id):
            return None
        task = asyncio.get_running_loop().create_task(self.on_lost(node_id, reason))
        task.add_done_callback(_log_task_failure)
        self._lost_tasks[node_id] = task
        return task

    async def _lost(self, node_id: str, reason: str) -> None:
        view = self._nodes.get(node_id)
        if view and view.transport_peer_id and self.on_transport_revoke is not None:
            await self.on_transport_revoke(node_id, False)
        task = self._start_lost(node_id, reason)
        if task is not None:
            await task

    # ------------------------------------------------------------------ verification
    def schedule_verify(self, node_id: str) -> None:
        task = self._verify_tasks.get(node_id)
        if task and not task.done():
            return
        task = asyncio.get_running_loop().create_task(self.verify(node_id))
        task.add_done_callback(_log_task_failure)
        self._verify_tasks[node_id] = task

    def _engine_match(self, view: NodeView) -> tuple[bool, str]:
        theirs = view.inventory.engine
        if theirs.rpc_proto and self._engine.rpc_proto and theirs.rpc_proto != self._engine.rpc_proto:
            return False, f"RPC protocol {theirs.rpc_proto} != host {self._engine.rpc_proto}"
        if theirs.release and self._engine.release and theirs.release != self._engine.release:
            return False, f"llama.cpp {theirs.release} != host {self._engine.release}"
        if theirs.smoke == "failed":
            return False, f"worker engine smoke test failed: {theirs.smoke_detail}"
        return True, f"llama.cpp {theirs.release or 'source build'}, RPC {theirs.rpc_proto or 'unknown'}"

    async def verify(self, node_id: str) -> NodeView:
        view = self._nodes[node_id]
        if self.busy(node_id):
            return view  # never probe a worker that is serving a deployment (its RPC server is single-client)
        if view.worker is None or self.probe is None:
            return self._set_state(view, NodeState.ineligible, "no worker endpoint")
        match, match_detail = self._engine_match(view)
        endpoint = f"{view.worker.host}:{view.worker.port}"
        if not match:
            result = VerificationResult(checked_at=utcnow(), reachable=False, engine_match=False, rpc_devices=[],
                                        detail=match_detail,
                                        fix=f"install the same release on both machines: uv run hypercompute engine "
                                            f"install (host runs {self._engine.release})")
            return self._set_state(view, NodeState.ineligible, match_detail, verification=result)
        probe = await self.probe([endpoint])
        devices = probe.by_endpoint.get(endpoint, []) if probe.ok else []
        fix = None if probe.ok else firewall_fix(view.inventory.os.system, view.worker.port,
                                                 self._coord_addr.get(node_id))
        result = VerificationResult(checked_at=utcnow(), reachable=probe.ok, engine_match=True,
                                    rpc_devices=devices, detail=f"{probe.detail}; {match_detail}",
                                    fix=fix)
        current = self._nodes.get(node_id, view)
        if current.state not in (NodeState.verifying, NodeState.ineligible, NodeState.eligible):
            return current
        if probe.ok:
            names = ", ".join(d.name for d in devices)
            view = self._set_state(current, NodeState.eligible, None, verification=result)
            self._events.publish("node.verified", f"{view.display_name} verified: {names}", node_id=node_id)
            return view
        return self._set_state(current, NodeState.ineligible, "worker unreachable from host", verification=result)

    # ------------------------------------------------------------------ leave
    async def leave(self, node_id: str, by_owner: bool) -> NodeView:
        view = self._nodes.get(node_id)
        if view is None:
            raise KeyError(node_id)
        state = NodeState.removed if by_owner else NodeState.left
        reason = "removed by host" if by_owner else "contributor stopped sharing"
        self._cred.pop(node_id, None)
        self.managed.forget(node_id)
        if view.transport_peer_id and self.on_transport_revoke is not None:
            await self.on_transport_revoke(node_id, True)
        await self._lost(node_id, reason)
        view = self._set_state(view, state, reason)
        self._events.publish("node.left", f"{view.display_name}: {reason}", node_id=node_id)
        return view

    async def close_session(self, session_id: str) -> None:
        for view in list(self._nodes.values()):
            if view.session_id == session_id and view.state not in (NodeState.left, NodeState.removed):
                await self.leave(view.id, by_owner=True)
