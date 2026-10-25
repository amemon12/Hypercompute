"""Friend agent: inventory -> start the RPC worker -> join -> heartbeat -> leave.

The agent only ever runs the allowlisted ggml-rpc-server binary with an argument array. The
coordinator cannot make it execute anything else. Heartbeats carry measurements; only with --managed
do heartbeat replies carry typed host requests (managed.py), each checked against what the
contributor allowed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import secrets
import signal
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import uvicorn

from .. import __version__, config, engine, netutil
from ..adapters.llama_output import parse_rpc_banner
from ..installers.exo import API_PORT as EXO_API_PORT
from ..schemas import (
    HCPEER_REPORT_FEATURE,
    MANAGED_PEERS_FEATURE,
    ContributionLimits,
    HeartbeatRequest,
    HostInventory,
    JoinRequest,
    WorkerEndpoint,
    WorkerState,
    utcnow,
)
from ..units import fmt_bytes
from . import hcpeer, inventory
from .control import create_control_app
from .exo_sidecar import ExoSidecar, ExoSidecarError, namespace_for, security_warning, with_exo
from .hcpeer import HcpeerBanner
from .managed import ManagedConsent, ManagedError, ManagedPeer, consent_lines, purge_after_crash
from .supervisor import Supervisor

if TYPE_CHECKING:
    from ..transport.client import PeerClient

log = logging.getLogger("hypercompute.agent")
WORKER_KEY = "worker"
REVIVE_LIMIT = 3  # --managed: restarts of a crashed worker allowed per REVIVE_WINDOW_S before sharing stops
REVIVE_WINDOW_S = 30 * 60
# --managed over Iroh: the worker listens on localhost and only the paired host reaches it, so an unattended machine
# waits this long (not config.HOST_UNREACHABLE_STOP_S) for a host that restarts overnight
MANAGED_IROH_UNREACHABLE_STOP_S = 2 * 3600


class AgentError(RuntimeError):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


def say(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except (OSError, ValueError):
        pass  # the terminal is gone (closed window / SIGHUP); cleanup must still run


def node_id() -> str:
    p = config.paths().ensure().agent / "node_id"
    if p.exists():
        value = p.read_text(encoding="ascii").strip()
        if len(value) == 32:
            return value
    value = uuid.uuid4().hex
    p.write_text(value, encoding="ascii")
    return value


def state_path() -> Path:
    return config.paths().ensure().agent / "state.json"


def write_state(state: dict) -> None:
    path = state_path()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)  # holds the node credential


def read_state() -> dict | None:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_state() -> None:
    state_path().unlink(missing_ok=True)


_HOSTNAME = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*")


def parse_host(value: str) -> tuple[str, int]:
    """'ip', 'ip:port', 'host:port/' or 'http://ip:port/' (as copied from a browser) -> (host, port)."""
    text = value.strip()
    bad = AgentError(f"invalid --host {value!r}",
                     fix=f"use the host's LAN address and port, e.g. 192.168.1.20:{config.DEFAULT_GATEWAY_PORT}")
    try:
        parts = urllib.parse.urlsplit(text if "://" in text else f"//{text}")
        host, port = parts.hostname, parts.port
    except ValueError:
        raise bad from None
    if parts.scheme not in ("", "http") or parts.username is not None or not host or port == 0:
        raise bad
    if re.fullmatch(r"[\d.]+", host):
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            raise bad from None
    elif not _HOSTNAME.fullmatch(host):
        raise bad  # also rejects IPv6 literals: the agent speaks IPv4 only
    return host, port or config.DEFAULT_GATEWAY_PORT


def check_coordinator(host: str, port: int) -> dict:
    url = f"http://{host}:{port}/api/health"
    try:
        r = httpx.get(url, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        ok, err = netutil.tcp_reachable(host, port, timeout=3.0)
        mine = netutil.lan_addresses()
        hint = ("TCP connection works but the gateway did not answer correctly." if ok else
                f"Cannot open TCP {host}:{port} ({err}).")
        raise AgentError(
            f"could not reach the Hypercompute host at {url}: {exc}. {hint}",
            fix=(f"Check the address on the host's dashboard. This machine's LAN addresses: {mine or 'none'}. "
                 "Both machines must be on the same network; guest/campus Wi-Fi often blocks device-to-device "
                 "traffic (client isolation). Try a phone hotspot, travel router or Ethernet."),
        ) from exc


def heartbeat_exclude(managed: dict) -> set[str] | dict[str, set[str]] | None:
    """Heartbeat fields a host must not see (hosts forbid unknown fields, even as null): `managed` for a host without
    managed peers, and an unreported `managed.hcpeer_mmap` for a host without HCPEER_REPORT_FEATURE."""
    if not managed:
        return {"managed"}
    return {"managed": {"hcpeer_mmap"}} if managed["managed"].hcpeer_mmap is None else None


def hcpeer_consent(consent: ManagedConsent) -> ManagedConsent:
    """The --managed consent with --max-disk as the whole GiB hcpeer caps the cache at (hcpeer.consented_max_disk)."""
    try:
        return replace(consent, max_disk_bytes=hcpeer.consented_max_disk(consent.max_disk_bytes))
    except ValueError as exc:
        raise AgentError(str(exc), fix=f"join with --max-disk 1GiB or more, or without {hcpeer.MMAP_ENV}=1") from exc


@dataclass
class Agent:
    host: str
    port: int
    code: str
    display_name: str
    inv: HostInventory
    contribution: ContributionLimits
    advertise_ip: str
    worker_port: int
    sup: Supervisor = field(init=False)
    credential: str | None = None
    node: str = field(default_factory=node_id)
    state: str = "starting"
    state_reason: str | None = None
    deployments: list[str] = field(default_factory=list)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    control_port: int | None = None
    rpc_proto: str | None = None
    paused: bool = False
    host_ip: str = ""  # the host's resolved IPv4 address (what firewall rules need), not the typed name
    worker_exe: str | None = None
    loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    leaving: bool = False
    host_lost: bool = False
    exo_enabled: bool = False  # --exo: also run the pinned EXO node so the host can serve MLX models here
    exo: ExoSidecar | None = field(default=None, init=False, repr=False)
    exo_status: str | None = None  # last EXO status shown to the user
    peer_client: PeerClient | None = field(default=None, repr=False)
    managed_consent: ManagedConsent | None = None  # --managed: what the host may change on this machine
    threads: int | None = None  # rpc-server -t; None keeps its default
    hcpeer_mmap: bool = False  # HC_PEER_MMAP=1 at join (macOS): the worker's CPU device maps cache files (hcpeer.py)
    hcpeer_banner: HcpeerBanner | None = None  # the hcpeer line the last started worker printed
    host_features: list[str] = field(default_factory=list)  # optional capabilities the host listed at join
    worker_devices_seen: list[str] = field(default_factory=list)  # device lines the running worker printed
    worker_generation: int = 0  # +1 on every host-requested restart
    restarting: bool = False
    managed: ManagedPeer | None = field(default=None, init=False, repr=False)
    _worker_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _revivals: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hcpeer_mmap and self.managed_consent is not None:  # every side holds the cap hcpeer enforces
            self.managed_consent = hcpeer_consent(self.managed_consent)
        self.sup = Supervisor(config.paths().ensure().agent / "managed-processes.json")
        self.host_ip = "127.0.0.1" if self.peer_client else self.host_ip or netutil.resolve_ipv4(self.host) or self.host
        if self.hcpeer_mmap and not self.contribution.cache_enabled:  # hcpeer maps the RPC cache's files: it needs -c
            self.contribution = self.contribution.model_copy(update={"cache_enabled": True})
        if self.managed_consent is not None:
            self.managed = ManagedPeer(self.managed_consent, config.paths().rpc_cache,
                                       cache_enabled=self.contribution.cache_enabled,
                                       restart_worker=self.reconfigure_worker,
                                       current_worker=lambda: (list(self.contribution.devices), self.threads),
                                       is_busy=lambda: bool(self.deployments))

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------ worker
    def reconcile_orphans(self) -> None:
        """Stop what an earlier run or an unclean stop left recorded. Only before starting anything: it stops every
        recorded process that is still ours, so after start_exo it would also kill the EXO this run just started."""
        for n in self.sup.reconcile_orphans():
            log.info("reconcile: %s", n)

    def cache_cap_gib(self) -> int | None:
        """--max-disk in whole GiB, the hcpeer cache cap; None without a consented limit."""
        return hcpeer.cache_cap_gib(self.managed_consent.max_disk_bytes) if self.managed_consent else None

    def worker_command(self) -> tuple[list[str], dict[str, str]]:
        """ggml-rpc-server's argument array, and what its environment adds to the agent's."""
        args = ["-H", self.advertise_ip, "-p", str(self.worker_port), "-d", ",".join(self.contribution.devices)]
        if self.threads is not None:
            args += ["-t", str(int(self.threads))]
        cache = config.paths().rpc_cache if self.contribution.cache_enabled else None
        if cache is not None:
            args.append("-c")
        env = hcpeer.worker_env(cache, on=self.hcpeer_mmap, cap_gib=self.cache_cap_gib(), banner=self.hcpeer_banner)
        return args, env

    def start_worker(self) -> None:
        installed = engine.find_installed()
        if installed is None:
            raise AgentError("llama.cpp engine is not installed", fix="uv run hypercompute engine install")
        exe = installed.binary("ggml-rpc-server")
        self.worker_exe = str(exe)
        args, env = self.worker_command()
        self._launch_worker(exe, installed.bin_dir, args, env)
        capped = self.worker_command()[1]
        if hcpeer.CACHE_MAX_ENV in capped and hcpeer.CACHE_MAX_ENV not in env:
            # only this start's banner could show an hcpeer that honours the cache cap: start again with it
            outcome = self.sup.stop(WORKER_KEY)
            if outcome is not None and not outcome.clean:
                raise AgentError(f"the worker could not be stopped to apply the hcpeer cache cap "
                                 f"({outcome.describe()})")
            log.info("restarting the worker with %s=%s", hcpeer.CACHE_MAX_ENV, capped[hcpeer.CACHE_MAX_ENV])
            self._launch_worker(exe, installed.bin_dir, args, capped)

    def _launch_worker(self, exe: Path, cwd: Path, args: list[str], env: dict[str, str]) -> None:
        log_path = config.paths().agent / "logs" / "ggml-rpc-server.log"
        proc = self.sup.start(WORKER_KEY, exe, args, cwd, log_path, env)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            text = proc.read_log()
            banner = parse_rpc_banner(text)
            if banner.bind_failed:
                self.sup.stop(WORKER_KEY)
                raise AgentError(f"the worker could not bind {self.advertise_ip}:{self.worker_port}",
                                 fix="pick another port with --worker-port, or check "
                                     "`netsh int ipv4 show excludedportrange protocol=tcp` on Windows")
            if banner.proto and banner.transport:
                self.rpc_proto = banner.proto
                self.worker_devices_seen = [d.name for d in banner.devices]
                self.hcpeer_banner = hcpeer.parse_banner(text)  # printed before the transport line
                if self.managed is not None:  # hcpeer with its cap never evicts: the disk guard makes room for it
                    on = bool(self.hcpeer_banner and self.hcpeer_banner.on)
                    self.managed.worker_started(on and hcpeer.CACHE_MAX_ENV in env)
                return
            if not proc.alive():
                raise AgentError(f"ggml-rpc-server exited early (code {proc.exit_code()}): {text[-600:]}")
            time.sleep(0.3)
        self.sup.stop(WORKER_KEY)
        raise AgentError("ggml-rpc-server did not report readiness within 45s")

    def start_exo(self) -> None:
        """Start EXO (--exo) and put its live status into the inventory the host reads at join."""
        if not self.exo_enabled:
            return
        if self.exo is None:
            self.exo = ExoSidecar(self.sup, config.paths().ensure(), self.advertise_ip,
                                  namespace_for(self.host_ip, self.port))
        try:
            info = self.exo.start()
        except ExoSidecarError as exc:
            raise AgentError(str(exc), fix=exc.fix) from exc
        self.inv = self.inv.model_copy(update={"extra_engines": with_exo(self.inv.extra_engines, info)})
        self.exo_status = info.status
        say(f"EXO running on {self.exo.endpoint} (node {self.exo.node_id}, namespace {self.exo.namespace}).")

    def stop_exo(self) -> str:
        if self.exo is None:
            return "not running"
        outcome = self.exo.stop()
        info = self.exo.info()
        self.inv = self.inv.model_copy(update={"extra_engines": with_exo(self.inv.extra_engines, info)})
        self.exo_status = info.status  # stopped on purpose: the heartbeat must not report EXO as lost
        return outcome.describe() if outcome else "not running"

    def exo_heartbeat(self) -> dict:
        """EXO's live status for the heartbeat, once HeartbeatRequest carries extra_engines (the host reads the
        inventory at join otherwise). Also tells the user when EXO stops on its own."""
        if self.exo is None:
            return {}
        info = self.exo.info()
        if info.status != self.exo_status:
            if self.exo_status == "running":
                say(f"[{time.strftime('%H:%M:%S')}] EXO is {info.status}: {info.detail}. The RPC worker keeps "
                    "sharing; the host cannot serve MLX models here until you rejoin with --exo.")
            self.exo_status = info.status
        if "extra_engines" not in HeartbeatRequest.model_fields:
            return {}
        return {"extra_engines": with_exo(self.inv.extra_engines, info)}

    def worker_status(self) -> tuple[WorkerState, int | None, float | None]:
        if self.restarting:
            return WorkerState.starting, None, None
        proc = self.sup.get(WORKER_KEY)
        if proc is None:
            return (WorkerState.paused if self.paused else WorkerState.stopped), None, None
        if proc.alive():
            return WorkerState.running, None, proc.cpu_seconds()
        return WorkerState.exited, proc.exit_code(), None

    def _stop_shared(self) -> str:
        """Stop the worker and EXO; returns what happened, for the user."""
        outcome = self.sup.stop(WORKER_KEY)
        detail = outcome.describe() if outcome else "not running"
        if self.exo is not None:
            detail += f"; EXO {self.stop_exo()}"
        return detail

    def pause(self) -> str:
        """Stop the worker but stay in the session; the host marks this node ineligible."""
        if self.paused:
            return "already paused"
        self.paused = True
        self._revoke_peer_worker()
        with self._worker_lock:  # a restart in progress (it gives up on seeing paused) must not outlive the pause
            detail = self._stop_shared()
        say(f"[{time.strftime('%H:%M:%S')}] paused: worker stopped ({detail})")
        return "paused"

    def resume(self) -> str:
        if not self.paused:
            return "not paused"
        try:
            self.reconcile_orphans()
            # EXO first: a worker up during EXO's start (up to 180 s) heartbeats as running while still paused, so
            # the host could re-verify it and place work that a failed EXO start then kills
            self.start_exo()
            self.start_worker()
            if self.peer_client:
                self.peer_client.set_worker_active(True)
        except Exception as exc:
            # all or nothing: a worker left running while paused is shared with the host and Pause skips it
            say(f"[{time.strftime('%H:%M:%S')}] resume failed, still paused ({self._stop_shared()}): {exc}")
            raise
        self.paused = False
        say(f"[{time.strftime('%H:%M:%S')}] resumed: worker running again; the host will re-check it")
        return "resumed"

    def _start_worker_checked(self, devices: list[str]) -> None:
        """Start the worker (retrying a port still held by the old process) and check the devices it printed."""
        for attempt in range(3):
            if self.leaving or self.paused:  # leave() and pause() stop the worker once this returns
                raise AgentError("sharing is stopping or paused on this machine")
            try:
                self.start_worker()
                break
            except AgentError:
                self.sup.stop(WORKER_KEY)  # an exited worker must not stay recorded: the heartbeat would stop sharing
                if attempt == 2:
                    raise
                time.sleep(2.0)
        seen = [d.upper() for d in self.worker_devices_seen]
        if seen and seen != [d.upper() for d in devices]:
            self.sup.stop(WORKER_KEY)
            raise AgentError(f"the worker started with {','.join(self.worker_devices_seen)}, not {','.join(devices)}")

    def reconfigure_worker(self, devices: list[str], threads: int | None) -> None:
        """Host-requested restart with new -d/-t (only within the --managed consent, checked by ManagedPeer). Runs on
        its own thread; heartbeats report the worker as starting meanwhile. A failed start restores the old settings."""
        if self.paused or self.leaving:
            raise ManagedError("sharing is paused or stopping on this machine")
        with self._worker_lock:
            self._reconfigure_locked(devices, threads)

    def _reconfigure_locked(self, devices: list[str], threads: int | None) -> None:
        previous = (self.contribution, self.threads)
        self.restarting = True
        try:
            self._revoke_peer_worker()
            outcome = self.sup.stop(WORKER_KEY)
            if outcome is not None and not outcome.clean:
                raise AgentError(f"the running worker could not be stopped ({outcome.describe()})")
            self.contribution = self.contribution.model_copy(update={"devices": list(devices)})
            self.threads = threads
            try:
                self._start_worker_checked(devices)
            except AgentError as exc:
                self.contribution, self.threads = previous
                self.worker_generation += 1
                try:
                    self._start_worker_checked(list(previous[0].devices))
                except AgentError as again:
                    raise AgentError(f"{exc}; restoring the previous worker also failed: {again}") from again
                raise AgentError(f"{exc}; the previous worker ({','.join(previous[0].devices)}) runs again") from exc
            self.worker_generation += 1
            say(f"[{time.strftime('%H:%M:%S')}] the host restarted the worker with {','.join(devices)}"
                + (f", {threads} threads" if threads else ""))
        finally:
            self.restarting = False
            if self.peer_client and not self.paused and self.sup.get(WORKER_KEY) is not None:
                self.peer_client.set_worker_active(True)

    def _revive_worker(self, code: int | None) -> bool:
        """--managed only: nobody may be at this machine, so a crashed worker is started again with the same
        settings (at most REVIVE_LIMIT times per REVIVE_WINDOW_S) instead of ending the session."""
        now = time.monotonic()
        self._revivals = [t for t in self._revivals if now - t < REVIVE_WINDOW_S]
        if not self.managed_active or self.paused or self.leaving or len(self._revivals) >= REVIVE_LIMIT:
            return False
        self._revivals.append(now)
        say(f"[{time.strftime('%H:%M:%S')}] the worker exited (code {code}); starting it again "
            f"({len(self._revivals)} of {REVIVE_LIMIT} restarts allowed per {REVIVE_WINDOW_S // 60} minutes)")
        with self._worker_lock:
            self.restarting = True
            try:
                self._revoke_peer_worker()
                self.sup.stop(WORKER_KEY)  # drops the exited process and closes its log
                log_path = config.paths().agent / "logs" / "ggml-rpc-server.log"
                try:  # the next start truncates the log: keep the crash for whoever looks in the morning
                    os.replace(log_path, log_path.with_name("ggml-rpc-server.crashed.log"))
                except OSError as exc:
                    log.debug("could not keep the crashed worker's log: %s", exc)
                if self.contribution.cache_enabled and (n := purge_after_crash(config.paths().rpc_cache / "rpc")):
                    say(f"[{time.strftime('%H:%M:%S')}] dropped {n} cache files written just before the exit "
                        "(one may be cut short); the host sends them again")
                self._start_worker_checked(list(self.contribution.devices))
            except AgentError as exc:
                say(f"[{time.strftime('%H:%M:%S')}] the worker did not start again: {exc}")
                return False
            finally:
                self.restarting = False
                if self.peer_client and not self.paused and self.sup.get(WORKER_KEY) is not None:
                    self.peer_client.set_worker_active(True)
            self.worker_generation += 1
        return True

    # ------------------------------------------------------------------ join / heartbeat / leave
    def join(self) -> None:
        eng = self.inv.engine.model_copy(update={"rpc_proto": self.rpc_proto or self.inv.engine.rpc_proto})
        req = JoinRequest(code=self.code, node_id=self.node, display_name=self.display_name,
                          inventory=self.inv.model_copy(update={"engine": eng}), contribution=self.contribution,
                          worker=WorkerEndpoint(host=self.advertise_ip, port=self.worker_port),
                          agent_version=__version__)
        if self.peer_client:
            self.peer_client.configure_worker(self.node, self.worker_port)
            r = self.peer_client.request("join", body=req.model_dump(mode="json"))
        else:
            r = httpx.post(f"{self.base}/api/sessions/join", json=req.model_dump(mode="json"), timeout=20.0)
        if r.status_code != 200:
            try:
                body = r.json()
            except json.JSONDecodeError:
                body = {"detail": r.text[:300]}
            raise AgentError(f"join refused: {body.get('detail')}", fix=body.get("fix"))
        data = r.json()
        self.credential = data["credential"]
        self.state = data["state"]
        features = data.get("features")
        self.host_features = [f for f in features if isinstance(f, str)] if isinstance(features, list) else []
        write_state({"host": self.host, "port": self.port, "node_id": self.node, "credential": self.credential,
                     "pid": os.getpid(), "control_port": None, "control_token": None,
                     **({"transport": "iroh", "ticket": self.peer_client.ticket} if self.peer_client else {})})

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.credential}"}

    @property
    def managed_active(self) -> bool:
        """--managed was given and the host understands managed peers (an older host never sees the new fields)."""
        return (self.managed is not None and MANAGED_PEERS_FEATURE in self.host_features
                and "managed" in HeartbeatRequest.model_fields)

    def managed_heartbeat(self) -> dict:
        if self.managed is None:
            return {}
        try:  # the consented disk limits hold whether or not this host can manage peers
            self.managed.guard_disk()
        except OSError as exc:
            log.warning("cache disk guard skipped: %s", exc)
        if not self.managed_active:
            return {}
        # hcpeer on in the running worker's banner, for a host that plans file-backed peer stages (services/layout.py)
        hcpeer_on = (bool(self.hcpeer_banner and self.hcpeer_banner.on)
                     if HCPEER_REPORT_FEATURE in self.host_features else None)
        try:
            report = self.managed.report(list(self.contribution.devices), self.threads, self.worker_generation,
                                         hcpeer_on)
        except (ValueError, OSError) as exc:  # a bad report must not end an unattended session: skip it this beat
            log.warning("managed report skipped: %s", exc)
            return {}
        return {"managed": report}

    def _host_refused_managed(self, status: int) -> None:
        self.host_features = [f for f in self.host_features if f != MANAGED_PEERS_FEATURE]
        say(f"[{time.strftime('%H:%M:%S')}] the host refused the managed-peer report (HTTP {status}); it can no longer "
            "change this machine's worker or cache in this session.")

    def _apply_heartbeat(self, data: dict) -> None:
        if data["state"] != self.state or data.get("state_reason") != self.state_reason:
            self.state, self.state_reason = data["state"], data.get("state_reason")
            say(f"[{time.strftime('%H:%M:%S')}] host sees this machine as: {self.state}"
                + (f" ({self.state_reason})" if self.state_reason else ""))
        if data["deployments"] != self.deployments:
            self.deployments = data["deployments"]
            # membership is not proof of model work: the deployment may still be probing or loading
            say(f"[{time.strftime('%H:%M:%S')}] "
                + ("a deployment on the host is using this machine (it may still be loading)" if self.deployments
                   else "not serving any model"))
        if self.managed_active and self.managed is not None:
            self.managed.handle(data.get("commands"))

    def unreachable_limit_s(self) -> float:
        if self.managed is not None and self.peer_client is not None:
            return max(config.HOST_UNREACHABLE_STOP_S, MANAGED_IROH_UNREACHABLE_STOP_S)
        return config.HOST_UNREACHABLE_STOP_S

    def _host_gone_too_long(self, since: float) -> bool:
        if time.monotonic() - since <= self.unreachable_limit_s():
            return False
        self.host_lost = True
        say(f"[{time.strftime('%H:%M:%S')}] the host has been unreachable for over "
            f"{self.unreachable_limit_s() / 60:g} minutes: stopping sharing so the worker is not left open "
            "on this network.")
        self.stop_event.set()
        return True

    async def heartbeat_loop(self, client: httpx.AsyncClient) -> None:
        failures = 0
        unreachable_since: float | None = None
        while not self.stop_event.is_set():
            ws, code, cpu = self.worker_status()
            pools, devices = await asyncio.to_thread(inventory.fresh_measurements, self.inv)
            managed = await asyncio.to_thread(self.managed_heartbeat)
            hb = HeartbeatRequest(worker_state=ws, worker_exit_code=code, memory_pools=pools, devices=devices,
                                  worker_cpu_seconds=cpu, sent_at=utcnow(), **self.exo_heartbeat(), **managed)
            # hosts forbid unknown fields: a host without managed peers must never see the key, not even as null
            body = hb.model_dump(mode="json", exclude=heartbeat_exclude(managed))
            try:
                if self.peer_client:
                    r = await self.peer_client.arequest("heartbeat", body=body, node_id=self.node,
                                                        credential=self.credential)
                else:
                    r = await client.post(f"{self.base}/api/nodes/{self.node}/heartbeat",
                                          json=body, headers=self._auth(), timeout=8.0)
                if r.status_code in (401, 403):
                    say("The host removed this machine from the session. Stopping the worker.")
                    self.credential = None
                    self.stop_event.set()
                    return
                if managed and r.status_code in (400, 422):  # e.g. the host was rolled back to a version without it
                    self._host_refused_managed(r.status_code)
                    continue
                r.raise_for_status()
                failures, unreachable_since = 0, None
                self._apply_heartbeat(r.json())
            except httpx.HTTPError as exc:
                failures += 1
                unreachable_since = time.monotonic() if unreachable_since is None else unreachable_since
                if self._host_gone_too_long(unreachable_since):
                    return
                if failures in (1, 5) or failures % 20 == 0:
                    say(f"[{time.strftime('%H:%M:%S')}] host unreachable ({exc.__class__.__name__}); worker keeps "
                        f"running for up to {self.unreachable_limit_s() / 60:g} minutes. Press Ctrl+C to stop "
                        "sharing.")
            if ws == WorkerState.exited and not await asyncio.to_thread(self._revive_worker, code):
                say(f"The local worker exited (code {code}). Stopping.")
                self.stop_event.set()
                return
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=config.HEARTBEAT_INTERVAL_S)
            except TimeoutError:
                pass

    def _say_goodbye(self) -> None:
        if not self.credential:
            return
        attempts = 3 if self.host_lost else 1  # the host was away: keep trying for a little while
        for attempt in range(1, attempts + 1):
            try:
                if self.peer_client:
                    self.peer_client.request("leave", node_id=self.node, credential=self.credential).raise_for_status()
                else:
                    httpx.post(f"{self.base}/api/nodes/{self.node}/leave", headers=self._auth(), timeout=5.0)
                say("Told the host this machine left.")
                return
            except httpx.HTTPError as exc:
                if attempt == attempts:
                    say(f"Could not reach the host to say goodbye ({exc.__class__.__name__}); the host will mark "
                        "this machine stale on its own.")
                else:
                    time.sleep(2)

    def _revoke_peer_worker(self) -> None:
        """A broken transport must never prevent the local worker from stopping."""
        if self.peer_client:
            try:
                self.peer_client.set_worker_active(False)
            except Exception as exc:
                say(f"Iroh access revocation failed ({type(exc).__name__}); stopping the local worker anyway.")

    def leave(self) -> None:
        self.leaving = True
        if self.managed is not None:
            self.managed.close()  # stop cache downloads first; finished files stay, part files resume next time
        self._revoke_peer_worker()
        if self.exo is not None:
            exo_outcome = self.exo.stop()
            if exo_outcome and not exo_outcome.clean:
                say(f"EXO could not be stopped and may still be running ({exo_outcome.describe()}); "
                    "`hypercompute leave` retries it.")
            elif exo_outcome:
                say(f"EXO stopped: {exo_outcome.describe()}")
        with self._worker_lock:  # a restart in progress gives up on seeing leaving; then stop what it started
            outcome = self.sup.stop(WORKER_KEY)
        if outcome and not outcome.clean:
            # keep state.json and the process record: a later `hypercompute leave` retries the stop
            say(f"The worker could not be stopped and may still be running ({outcome.describe()}). Run "
                "`hypercompute leave` again, from an Administrator (Windows) or sudo (macOS/Linux) shell if needed.")
            return
        if outcome:
            say(f"Worker stopped: {outcome.describe()}")
        self._say_goodbye()
        clear_state()
        if self.contribution.cache_enabled:
            say(f"RPC tensor cache kept at {config.paths().rpc_cache} (remove with: uv run hypercompute leave "
                "--purge-cache)")

    # ------------------------------------------------------------------ local control
    def status_doc(self) -> dict:
        ws, code, cpu = self.worker_status()
        return {"host": f"{self.host}:{self.port}", "state": self.state, "reason": self.state_reason,
                "worker": f"{ws.value} on {self.advertise_ip}:{self.worker_port}",
                "worker_cpu_seconds": round(cpu, 1) if cpu is not None else None,
                "sharing": ", ".join(self.contribution.devices),
                "budget": fmt_bytes(self.contribution.max_bytes) + " (soft admission limit)",
                "deployments_using_this_machine": self.deployments or "none",
                **(self.peer_client.status() if self.peer_client else {}),
                **({"managed_by_host": self.managed.summary() if self.managed_active else
                    "not supported by this host"} if self.managed is not None else {}),
                **({"exo": f"{(info := self.exo.info()).status} on {self.exo.endpoint}: {info.detail}"}
                   if self.exo is not None else {})}

    async def control_server(self) -> None:
        token = secrets.token_urlsafe(24)
        port = netutil.free_loopback_port()
        app = create_control_app(token, self.status_doc, self.stop_event.set, f"{self.host}:{self.port}",
                                 pause=self.pause, resume=self.resume)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.control_port = port
        st = read_state() or {}
        write_state({**st, "control_port": port, "control_token": token, "pid": os.getpid()})
        say(f"Local status and Stop button: http://127.0.0.1:{port}/")
        task = asyncio.create_task(server.serve())
        await self.stop_event.wait()
        server.should_exit = True
        await task

    async def serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        async with httpx.AsyncClient() as client:
            await asyncio.gather(self.heartbeat_loop(client), self.control_server())


def print_plan(agent: Agent, fallback_cpu: bool) -> None:
    inv = agent.inv
    say(f"Hypercompute agent {__version__} on {inv.hostname} ({inv.os.system} {inv.os.machine})")
    say(f"Engine: llama.cpp {inv.engine.release or 'unknown'} [{inv.engine.variant}] smoke={inv.engine.smoke}")
    for d in inv.devices:
        size = fmt_bytes(d.total_bytes)
        say(f"  device {d.id:<10} {d.name} [{d.backend}] total {size}, pool {d.pool_id}")
    for p in inv.memory_pools:
        say(f"  pool {p.id:<12} {p.kind:<14} total {fmt_bytes(p.total_bytes)}, free now {fmt_bytes(p.free_bytes)}")
    cpu_only = agent.contribution.devices == ["CPU"]
    note = (" (CPU fallback: no supported GPU found, expect slow work)" if fallback_cpu else
            " (CPU only, as requested: expect slow work)" if cpu_only else "")
    say(f"Will share: {', '.join(agent.contribution.devices)}{note}")
    say(f"Budget: up to {fmt_bytes(agent.contribution.max_bytes)} for model layers (a soft limit the host's planner "
        f"respects, not an enforced GPU cap); keeping {fmt_bytes(agent.contribution.reserve_bytes)} free for you.")
    if agent.peer_client:
        say(f"Worker: ggml-rpc-server on localhost:{agent.worker_port}, shared through encrypted Iroh streams "
            "with the host in your invitation. Only pair with a host you trust.")
    else:
        say(f"Worker: ggml-rpc-server on {agent.advertise_ip}:{agent.worker_port} (llama.cpp RPC is unauthenticated: "
            "only join hosts you trust, on a private network).")
    if agent.contribution.cache_enabled:
        say(f"Tensor cache ON at {config.paths().rpc_cache}: speeds up repeat loads, uses disk, kept after leaving.")
    if agent.hcpeer_mmap:
        say(hcpeer.consent_line(config.paths().rpc_cache / "rpc", agent.cache_cap_gib()))
    if agent.managed_consent is not None:
        for line in consent_lines(agent.managed_consent, config.paths().rpc_cache / "rpc"):
            say(line)
        if agent.peer_client:
            say(f"  Unattended: if the host goes away, this machine waits up to "
                f"{agent.unreachable_limit_s() / 3600:g} h for it (not {config.HOST_UNREACHABLE_STOP_S / 60:g} min), "
                "and a crashed worker is started again (at most 3 times per 30 minutes).")
    if agent.exo_enabled:
        say(f"EXO: also runs the pinned EXO node on {agent.advertise_ip}:{EXO_API_PORT} so the host can serve MLX "
            "models here. EXO downloads each model it serves from Hugging Face into ~/.exo/models (kept after "
            "leaving).")
        say(security_warning(agent.advertise_ip))  # before consent: EXO's API is open to the LAN once it starts
    for d in inv.diagnostics:
        say(f"  [{d.level}] {d.message}" + (f" Fix: {d.fix}" if d.fix else ""))


_RULE_NAME = "Hypercompute RPC worker"


def _windows_scoping(agent: Agent) -> list[str]:
    program = f' -Program "{agent.worker_exe}"' if agent.worker_exe else ""
    lines = ["  In an Administrator PowerShell:",
             f'  New-NetFirewallRule -DisplayName "{_RULE_NAME}" -Direction Inbound{program} -Protocol TCP '
             f"-LocalPort {agent.worker_port} -RemoteAddress {agent.host_ip} -Action Allow"]
    if agent.worker_exe:
        lines += ["  If Windows showed a firewall prompt for ggml-rpc-server, the rule it created admits every "
                  "address; remove it:",
                  f'  Get-NetFirewallApplicationFilter -Program "{agent.worker_exe}" | Get-NetFirewallRule | '
                  f'Where-Object DisplayName -ne "{_RULE_NAME}" | Remove-NetFirewallRule']
    return [*lines, f'  Remove when you stop sharing: Remove-NetFirewallRule -DisplayName "{_RULE_NAME}"']


def _macos_scoping(agent: Agent) -> list[str]:
    """One pf ruleset (pfctl -f replaces the anchor): the worker for the host only, EXO's API for the hc_exo table."""
    port, host_ip = agent.worker_port, agent.host_ip
    rules = (f"pass in quick proto tcp from {host_ip} to any port {port}\\nblock drop in quick proto tcp to any port "
             f"{port}\\n")
    if agent.exo_enabled:  # EXO's API also answers the other Macs' topology probes, so they go in the table too
        # lo0 passes first: macOS pf filters loopback, and the sidecar's readiness probe and EXO's dashboard use it
        rules += (f"table <hc_exo> persist {{ {host_ip} }}\\npass in quick on lo0 proto tcp to any port {EXO_API_PORT}"
                  f"\\npass in quick proto tcp from <hc_exo> to any port {EXO_API_PORT}\\nblock drop in quick proto "
                  f"tcp to any port {EXO_API_PORT}\\n")
    lines = ["  The macOS application firewall cannot scope by address. Optional pf rule:",
             f"  printf '{rules}' | sudo pfctl -a com.apple/hypercompute -f - && sudo pfctl -E"]
    if agent.exo_enabled:
        lines.append(f"  EXO's API (TCP {EXO_API_PORT}) must also admit every other Mac sharing with this host; add "
                     "each one: sudo pfctl -a com.apple/hypercompute -t hc_exo -T add <its IP>")
    return [*lines, "  Remove it when you stop sharing: sudo pfctl -a com.apple/hypercompute -F all"]


def scoping_guidance(agent: Agent) -> str:
    """Exact OS commands that let only the host reach the worker. Printed every join; never run by Hypercompute."""
    if agent.peer_client:
        return "Iroh: the worker listens on localhost; only the paired host can reach it through this connection."
    ip, port, host_ip = agent.advertise_ip, agent.worker_port, agent.host_ip
    head = (f"Security: the worker on {ip}:{port} has no authentication. Until you scope it, any device on this "
            f"network can connect to it and use this machine's memory. Allow only the host ({host_ip}):")
    system = agent.inv.os.system
    if system == "windows":
        lines = _windows_scoping(agent)
    elif system == "macos":
        lines = _macos_scoping(agent)
    else:
        lines = ["  ufw must be active with a default deny for incoming traffic (check: sudo ufw status verbose):",
                 f"  sudo ufw allow from {host_ip} to any port {port} proto tcp",
                 f"  Remove when you stop sharing: sudo ufw delete allow from {host_ip} to any port {port} proto tcp"]
    return "\n".join([head, *lines, "  Hypercompute never runs these commands for you."])


def _install_stop_signals(agent: Agent) -> dict[int, object]:
    """Closing the terminal (SIGHUP), `kill` (SIGTERM) or Ctrl+Break stop sharing the way Ctrl+C does."""
    if threading.current_thread() is not threading.main_thread():
        return {}

    def handler(signum: int, frame: object) -> None:
        if agent.leaving:
            return  # cleanup is already running; a second signal must not interrupt it
        loop = agent.loop
        if loop is not None and loop.is_running():
            say(f"\nReceived {signal.Signals(signum).name}: stopping sharing...")
            loop.call_soon_threadsafe(agent.stop_event.set)
        else:
            raise KeyboardInterrupt

    names = ("SIGBREAK",) if sys.platform == "win32" else ("SIGTERM", "SIGHUP")
    return {sig: signal.signal(sig, handler) for name in names if (sig := getattr(signal, name, None)) is not None}


def _restore_signals(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler if handler is not None else signal.SIG_DFL)  # type: ignore[arg-type]


def run_join(agent: Agent, fallback_cpu: bool, assume_yes: bool) -> int:
    print_plan(agent, fallback_cpu)
    if not assume_yes:
        if not sys.stdin.isatty():
            raise AgentError("confirmation needed", fix="re-run with --yes to accept the shown contribution")
        if input("Share these resources with the host? [y/N] ").strip().lower() not in ("y", "yes"):
            say("Not joining.")
            return 1
    previous = _install_stop_signals(agent)
    try:
        agent.reconcile_orphans()
        agent.start_worker()
        say(f"Worker running (RPC protocol v{agent.rpc_proto}).")
        if agent.hcpeer_mmap:
            say(hcpeer.status_line(agent.hcpeer_banner, agent.cache_cap_gib(), agent.worker_command()[1]))
        agent.start_exo()  # before join: the host reads this machine's engines from the join inventory
        agent.join()
        say("Joined. The host is verifying that it can reach this worker...")
        if agent.managed is not None and not agent.managed_active:
            say("This host does not support managed peers: it cannot change this machine's worker or cache.")
        say(scoping_guidance(agent))
        say("Press Ctrl+C (or use the local Stop button) to stop sharing.")
        asyncio.run(agent.serve())
    except KeyboardInterrupt:
        say("\nStopping sharing...")
    finally:
        agent.leave()
        _restore_signals(previous)
    return 0
