"""`hypercompute` command line: host, join, status, leave, doctor, engine, models, capacity, inventory."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import sys
import time
import webbrowser
from typing import TYPE_CHECKING, get_args

import httpx

from . import __version__, config, engine, netutil
from .agent import hcpeer, instance, inventory
from .agent.contribution import default_contribution, eligible_devices
from .agent.join import Agent, AgentError, check_coordinator, clear_state, parse_host, read_state, run_join, say
from .agent.managed import ManagedConsent, build_consent
from .agent.supervisor import Supervisor
from .capacity_cli import add_capacity_parser
from .image_cli import add_image_parser
from .schemas import EngineId, PlanStrategy
from .services.models import ModelError
from .units import GiB, fmt_bytes, parse_size

if TYPE_CHECKING:
    import uvicorn
    from fastapi import FastAPI

    from .schemas import HostInventory
    from .transport.client import PeerClient

log = logging.getLogger("hypercompute.cli")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _fail(message: str, fix: str | None = None, code: int = 2) -> int:
    say(f"error: {message}")
    if fix:
        say(f"fix:   {fix}")
    return code


# ----------------------------------------------------------------------------- engine


def cmd_engine(args: argparse.Namespace) -> int:
    if args.engine != "llama":
        return _cmd_optional_engine(args)
    lock = engine.load_lock()
    if args.action == "status":
        installed = engine.find_installed(lock)
        info = engine.inspect(installed, lock)
        say(json.dumps(info.model_dump(mode="json"), indent=2))
        return 0 if info.smoke == "passed" else 1
    if args.variant:
        if args.variant not in lock.variants:
            return _fail(f"unknown variant {args.variant}", f"choose one of: {', '.join(lock.variants)}")
        variant, reason = lock.variants[args.variant], "chosen with --variant"
    else:
        variant, reason = engine.choose_variant(lock, engine.host_os(), engine.host_arch(),
                                                engine.nvidia_driver_major())
    existing = engine.find_installed(lock)
    if existing and existing.variant == variant.key and not args.force:
        say(f"llama.cpp {lock.release} [{variant.key}] already installed at {existing.bin_dir}")
    else:
        say(f"llama.cpp {lock.release} (commit {lock.commit[:10]}) variant {variant.key}: {reason}")
        say(f"Requires: {variant.requires}")
        for a in variant.assets:
            say(f"  {a.name}  {a.size / 2**20:.0f} MiB  sha256 {a.sha256[:16]}...  {a.url}")
        if not args.yes and (not sys.stdin.isatty() or input("Download and install? [y/N] ").strip().lower()
                             not in ("y", "yes")):
            return _fail("not installed", "re-run with --yes to accept the download", 1)
        last = [0.0]

        def progress(name: str, done: int, total: int) -> None:
            now = time.monotonic()
            if now - last[0] > 2 or done >= total:
                last[0] = now
                say(f"  {name}: {done * 100 // max(total, 1)}%")

        existing = engine.install(variant, lock, progress)
        say(f"Installed to {existing.bin_dir}")
    info = engine.inspect(existing, lock)
    say(f"Smoke test: {info.smoke}: {info.smoke_detail}")
    return 0 if info.smoke == "passed" else 1


def _cmd_optional_engine(args: argparse.Namespace) -> int:
    """AirLLM / vLLM / EXO: each installs into its own environment under ~/.hypercompute/engines/<engine>."""
    from . import installers

    mod = installers.load(args.engine)
    if mod is None:
        return _fail(f"the {args.engine} installer is not part of this build")
    paths = config.paths().ensure()
    if args.action == "status":
        info = mod.status(paths)
        say(json.dumps(info.model_dump(mode="json"), indent=2))
        return 0 if info.installed else 1
    return int(mod.install(paths, yes=args.yes, force=args.force, say=say))


# ----------------------------------------------------------------------------- models


def cmd_models(args: argparse.Namespace) -> int:
    from .services.models import ModelService

    svc = ModelService()
    if args.action == "list":
        for v in svc.views():
            say(f"{v.artifact.id:<34} {v.status:<15} {fmt_bytes(v.total_bytes):>10}  {v.artifact.license}  "
                f"[{v.artifact.role}]")
        return 0
    view = svc.view(args.model_id)
    projector = view.artifact.mmproj if view.projector_status == "not_downloaded" else None
    if view.status == "downloaded" and projector is None:
        say(f"{args.model_id} already downloaded and verified at {view.local_path}")
        return 0
    if view.status != "downloaded":
        if svc.adopt_cached(args.model_id, [config.paths().downloads / f.name for f in view.artifact.files]):
            say(f"Adopted verified files from {config.paths().downloads}")
            if projector is None:
                return 0
        else:
            say(f"Downloading {view.artifact.display_name} ({fmt_bytes(view.total_bytes)}) from "
                f"{view.artifact.repo} @ {view.artifact.revision[:10]} (license {view.artifact.license})")
    if projector is not None:
        say(f"Downloading its vision projector {projector.name} ({fmt_bytes(projector.size_bytes)})")

    async def run() -> None:
        task = asyncio.create_task(svc.download(args.model_id))
        while not task.done():
            await asyncio.sleep(3)
            v = svc.view(args.model_id)
            say(f"  {v.downloaded_bytes * 100 // max(v.total_bytes, 1)}%" if v.status != "downloaded"
                else f"  vision projector {v.projector_status.replace('_', ' ')}")
        await task

    asyncio.run(run())
    say(f"Verified: {svc.view(args.model_id).local_path}")
    return 0


# ----------------------------------------------------------------------------- host


def _bind_sockets(ips: list[str], port: int) -> list[socket.socket]:
    socks = []
    for ip in ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform != "win32":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((ip, port))
        except OSError as exc:
            s.close()
            for other in socks:
                other.close()
            raise AgentError(f"cannot listen on {ip}:{port}: {exc}", fix="choose another --port") from exc
        s.listen(128)
        s.set_inheritable(True)
        socks.append(s)
    return socks


def _gateway_config(app: FastAPI) -> uvicorn.Config:
    import uvicorn

    return uvicorn.Config(app, log_level="warning", access_log=False, limit_concurrency=config.GATEWAY_MAX_CONNECTIONS)


def cmd_host(args: argparse.Namespace) -> int:
    lock = config.paths().ensure().coordinator / "host.lock"
    instance.acquire(lock, "host")  # before build(): restart reconciliation must never touch a live host
    try:
        return _run_host(args)
    finally:
        instance.release(lock)


def _run_host(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app
    from .coordinator import build

    transport = getattr(args, "transport", "lan")
    if transport == "iroh" and args.bind:
        raise AgentError("--bind is for LAN hosting; Iroh keeps the dashboard on localhost")
    if transport == "lan" and getattr(args, "relay_url", None):
        raise AgentError("--relay-url requires --transport iroh")
    lan = [] if transport == "iroh" else args.bind or netutil.lan_addresses()
    say(f"Hypercompute {__version__}: starting host on port {args.port} (this can take ~10 s: GPU probe)")
    local_contribution = None
    if args.local_max_memory or args.local_devices:
        inv = inventory.collect()
        local_contribution = default_contribution(
            inv, devices=args.local_devices.split(",") if args.local_devices else None,
            max_bytes=parse_size(args.local_max_memory) if args.local_max_memory else None)
    coord = build(args.port, local_contribution, lan_override=[] if transport == "iroh" else args.bind)
    app = create_app(coord, transport=transport, relay_url=getattr(args, "relay_url", None))
    socks = _bind_sockets(["127.0.0.1", *[ip for ip in lan if ip != "127.0.0.1"]], args.port)
    url = f"http://127.0.0.1:{args.port}/"
    say(f"Dashboard (this machine only): {url}")
    if lan:
        say(f"Friends join via: {', '.join(f'{ip}:{args.port}' for ip in lan)}")
    elif transport != "iroh":
        say("No private LAN address found: friends cannot reach this host yet (see `hypercompute doctor`).")
    for d in coord.nodes.local.inventory.diagnostics:
        say(f"[{d.level}] {d.message}" + (f" Fix: {d.fix}" if d.fix else ""))
    for note in coord.startup_notes:
        say(f"[startup] {note}")
    if not args.no_browser:
        webbrowser.open(url)
    server = uvicorn.Server(_gateway_config(app))
    server.run(sockets=socks)
    return 0


# ----------------------------------------------------------------------------- join / leave / status


def cmd_join(args: argparse.Namespace) -> int:
    lock = config.paths().ensure().agent / "agent.lock"
    instance.acquire(lock, "sharing agent")  # a second agent would reconcile away the first one's worker
    try:
        return _run_join(args)
    finally:
        instance.release(lock)


def _advertise_ip(host: str, port: int, explicit: str | None) -> str:
    """The worker address: --advertise-ip, else the local address the OS routes to the host through."""
    ip = explicit or netutil.route_source_ip(host, port)
    if not ip:
        raise AgentError(f"no route to {host}", fix="check that both machines are on the same network")
    loopback_dev = ip.startswith("127.") and host.startswith("127.")
    # link-local (169.254/16) is a direct cable or Thunderbolt Bridge: chosen explicitly or routed to the host
    if not loopback_dev and not netutil.is_lan_or_link_local(ip):
        raise AgentError(f"this machine would share via {ip}, which is not a private LAN address",
                         fix="join over a private LAN or an existing private overlay (Tailscale/ZeroTier)")
    return ip


def _run_join(args: argparse.Namespace) -> int:
    if getattr(args, "ticket", None):
        return _run_peer_join(args)
    if getattr(args, "relay_url", None):
        raise AgentError("--relay-url requires --ticket")
    host, port = parse_host(args.host)
    health = check_coordinator(host, port)
    return _run_join_on_host(args, host, port, health)


def _run_peer_join(args: argparse.Namespace) -> int:
    from .transport.client import PeerClient

    if args.exo or args.advertise_ip:
        raise AgentError("Iroh shares llama.cpp on localhost; --exo and --advertise-ip require LAN mode")
    peer = None
    try:
        say("Connecting to the host through Iroh...")
        peer = PeerClient(args.ticket, config.paths().ensure().agent / "iroh-device.key",
                          relay_url=getattr(args, "relay_url", None))
        response = peer.request("health")
        response.raise_for_status()
        say(f"Encrypted connection established: {peer.status().get('path')}")
        return _run_join_on_host(args, "127.0.0.1", 0, response.json(), peer)
    except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
        raise AgentError(f"Iroh connection failed: {exc}",
                         fix="ask the host for `uv run hypercompute invite`; check relay reachability") from exc
    finally:
        if peer is not None:
            peer.close()


def _run_join_on_host(args: argparse.Namespace, host: str, port: int, health: dict,
                      peer: PeerClient | None = None) -> int:
    inv = inventory.collect()
    if not inv.engine.installed or inv.engine.smoke != "passed":
        return _fail(f"local llama.cpp engine not ready: {inv.engine.smoke_detail}",
                     "uv run hypercompute engine install")
    host_release = (health.get("engine") or {}).get("release")
    if host_release and inv.engine.release and host_release != inv.engine.release:
        return _fail(f"engine mismatch: host runs llama.cpp {host_release}, this machine {inv.engine.release}",
                     "install the same pinned release on both machines: uv run hypercompute engine install")
    ip = "127.0.0.1" if peer else _advertise_ip(host, port, args.advertise_ip)
    port_ok, errors = netutil.pick_bindable_port(ip, args.worker_port, config.WORKER_PORT_SCAN)
    if port_ok is None:
        return _fail(f"no bindable worker port near {args.worker_port} on {ip}: {errors[:3]}",
                     "pass --worker-port with a free port")
    devices, fallback_cpu = eligible_devices(inv)
    if args.devices:
        devices, fallback_cpu = args.devices.split(","), False
    contribution = default_contribution(
        inv, devices=devices, max_bytes=parse_size(args.max_memory) if args.max_memory else None,
        reserve_bytes=parse_size(args.reserve) if args.reserve else None, cache_enabled=args.cache)
    if contribution.max_bytes < 1 * GiB:
        say(f"warning: contribution budget is only {fmt_bytes(contribution.max_bytes)}")
    consent = _managed_consent(args, inv, contribution.devices)
    if consent is not None and consent.seeding_allowed and not contribution.cache_enabled:
        contribution = contribution.model_copy(update={"cache_enabled": True})  # seeding fills the RPC cache
    if args.exo:
        from .installers import exo as exo_install

        exo_info = exo_install.status(config.paths())
        if not exo_info.installed:
            fix = ("join without --exo; this machine still shares through llama.cpp"
                   if exo_info.status == "unsupported" else exo_install.INSTALL_HINT)
            return _fail(f"--exo: {exo_info.detail}", fix)
    hcpeer_mmap = hcpeer.requested(inv.os.system, os.environ)
    inherited = hcpeer.inherited_limits(os.environ) if hcpeer_mmap else []
    if inherited:
        raise AgentError(f"{' and '.join(inherited)} set in this shell would reach the worker and change the cache "
                         f"limits hcpeer applies; with {hcpeer.MMAP_ENV}=1 they must be the ones this agent and the "
                         "host plan with (the cap from --max-disk, hcpeer's default 20 GiB free-space floor)",
                         fix=f"unset {' '.join(inherited)} and join again")
    agent = Agent(host=host, port=port, code=args.code, display_name=args.name or inv.hostname, inv=inv,
                  contribution=contribution, advertise_ip=ip, worker_port=port_ok, exo_enabled=args.exo,
                  peer_client=peer, managed_consent=consent, hcpeer_mmap=hcpeer_mmap)
    return run_join(agent, fallback_cpu, args.yes)


def _managed_consent(args: argparse.Namespace, inv: HostInventory, devices: list[str]) -> ManagedConsent | None:
    """--managed with --devices-allowed and --max-disk; None (nothing changes) without --managed."""
    allowed, max_disk = getattr(args, "devices_allowed", None), getattr(args, "max_disk", None)
    if not getattr(args, "managed", False):
        if allowed or max_disk:
            raise AgentError("--devices-allowed and --max-disk only apply with --managed",
                             fix="add --managed to let the host manage this machine, or drop those flags")
        return None
    try:
        return build_consent(inv, list(devices), allowed, max_disk)
    except ValueError as exc:
        raise AgentError(f"--managed: {exc}", fix="e.g. --managed --devices-allowed MTL0,CPU --max-disk 80GiB") from exc


def _ask_agent_to_leave(state: dict | None) -> bool:
    """True when a running agent acknowledged /leave (it then stops its own worker)."""
    if not state or not state.get("control_port") or not state.get("control_token"):
        return False
    try:
        httpx.post(f"http://127.0.0.1:{state['control_port']}/leave",
                   headers={"X-Hypercompute-Agent": state["control_token"]}, timeout=5.0).raise_for_status()
    except httpx.HTTPError:
        return False
    say("Asked the running agent to stop sharing.")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and read_state():
        time.sleep(0.5)
    say("Stopped." if not read_state() else "The agent is still shutting down; check `hypercompute status`.")
    return True


def _leave_without_agent(state: dict | None) -> bool:
    """No agent answered: stop any worker a dead agent left behind (even without state.json), then tell the host.

    Returns False when a recorded worker is still running, leaving state and records for a retry."""
    sup = Supervisor(config.paths().agent / "managed-processes.json")
    notes = sup.reconcile_orphans()
    for n in notes:
        say(n)
    if sup.recorded():
        return False
    if state and state.get("credential"):
        try:
            if state.get("transport") == "iroh":
                from .transport.client import PeerClient

                peer = PeerClient(state["ticket"], config.paths().agent / "iroh-device.key")
                try:
                    peer.request("leave", node_id=state["node_id"], credential=state["credential"]).raise_for_status()
                finally:
                    peer.close()
            else:
                httpx.post(f"http://{state['host']}:{state['port']}/api/nodes/{state['node_id']}/leave",
                           headers={"Authorization": f"Bearer {state['credential']}"}, timeout=5.0)
            say("Told the host this machine left.")
        except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
            say(f"Host unreachable ({exc.__class__.__name__}); it will mark this machine stale.")
    if state:
        clear_state()
    elif notes:
        say("Cleaned up after a sharing agent that did not exit cleanly.")
    else:
        say("This machine is not currently sharing.")
    return True


def cmd_leave(args: argparse.Namespace) -> int:
    state = read_state()
    if not _ask_agent_to_leave(state) and not _leave_without_agent(state):
        return _fail("a worker started by Hypercompute is still running and could not be stopped",
                     "run `hypercompute leave` again from an Administrator PowerShell (Windows) or with sudo "
                     "(macOS/Linux)", 1)
    if args.purge_cache:
        cache = config.paths().rpc_cache
        if cache.exists():
            size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
            shutil.rmtree(cache)
            say(f"Removed Hypercompute RPC tensor cache ({fmt_bytes(size)}) at {cache}")
    return 0


def cmd_pause_resume(args: argparse.Namespace) -> int:
    state = read_state()
    if not state or not state.get("control_port"):
        return _fail("this machine is not sharing (no running agent)", "start with `hypercompute join`", 1)
    try:
        r = httpx.post(f"http://127.0.0.1:{state['control_port']}/{args.command}",
                       headers={"X-Hypercompute-Agent": state["control_token"]}, timeout=60.0)
        say(r.json().get("detail", r.text))
        return 0 if r.status_code == 200 else 1
    except httpx.HTTPError as exc:
        return _fail(f"the agent did not answer: {exc}", "check `hypercompute status`")


def cmd_status(args: argparse.Namespace) -> int:
    state = read_state()
    if state and state.get("control_port"):
        try:
            s = httpx.get(f"http://127.0.0.1:{state['control_port']}/status", timeout=3.0).json()
            say("Sharing as a friend node:")
            for k, v in s.items():
                say(f"  {k}: {v}")
        except httpx.HTTPError:
            say("An agent state file exists but the agent is not answering (run `hypercompute leave`).")
    base = f"http://127.0.0.1:{args.port}"
    try:
        httpx.get(f"{base}/api/health", timeout=3.0).raise_for_status()
    except httpx.HTTPError:
        if not state:
            say("Not sharing and no host running on this machine.")
        return 0
    nodes = httpx.get(f"{base}/api/nodes", timeout=5.0).json()
    deps = httpx.get(f"{base}/api/deployments", timeout=5.0).json()
    say(f"Hosting on port {args.port}:")
    network = httpx.get(f"{base}/api/network", timeout=5.0)
    if network.status_code == 200:
        info = network.json()
        say(f"  transport: {info.get('transport', 'lan')}")
        for peer in info.get("peers", []):
            say(f"  connection: {peer}")
    for n in nodes:
        age = n.get("heartbeat_age_s")
        seen = f"{age:.0f}s ago" if age is not None else ""
        say(f"  {n['display_name']:<32} {n['state']:<10} {seen:>8}  "
            f"budget {fmt_bytes(n['contribution']['max_bytes'])}  {n.get('state_reason') or ''}")
    if deps:
        d = deps[0]
        say(f"  latest deployment {d['id'][:8]}: {d['state']} model {d['plan']['model_id']} devices "
            f"{','.join(d['plan']['device_list'])}")
    return 0


# ----------------------------------------------------------------------------- doctor / inventory


def cmd_doctor(args: argparse.Namespace) -> int:
    inv = inventory.collect()
    problems = 0

    def line(level: str, text: str, fix: str | None = None) -> None:
        nonlocal problems
        problems += level == "FAIL"
        say(f"[{level:4}] {text}" + (f"\n       fix: {fix}" if fix else ""))

    line("OK", f"Python {sys.version.split()[0]} on {inv.os.system} {inv.os.release} ({inv.os.machine})")
    if inv.engine.smoke == "passed":
        line("OK", f"llama.cpp engine: {inv.engine.smoke_detail}")
    else:
        line("FAIL", f"llama.cpp engine: {inv.engine.smoke_detail}", "uv run hypercompute engine install")
    lan = netutil.lan_addresses()
    line("OK" if lan else "WARN", f"private LAN addresses: {lan or 'none'}",
         None if lan else "connect to the same Wi-Fi/Ethernet as your friends")
    for ip in lan[:3]:
        port, errors = netutil.pick_bindable_port(ip, config.DEFAULT_WORKER_PORT, config.WORKER_PORT_SCAN)
        if port == config.DEFAULT_WORKER_PORT:
            line("OK", f"worker port {port} is bindable on {ip}")
        elif port:
            line("WARN", f"worker port {config.DEFAULT_WORKER_PORT} busy/reserved on {ip}; {port} works",
                 "join picks a free port automatically")
        else:
            line("FAIL", f"no bindable worker port on {ip}: {errors[:2]}", "pass --worker-port")
    for d in inv.diagnostics:
        line("WARN" if d.level != "error" else "FAIL", d.message, d.fix)
    if inv.disk_free_bytes is not None:
        line("OK" if inv.disk_free_bytes > 10 * GiB else "WARN", f"free disk: {fmt_bytes(inv.disk_free_bytes)}")
    if args.host:
        host, port = parse_host(args.host)
        try:
            h = check_coordinator(host, port)
            line("OK", f"host {host}:{port} reachable, engine {h.get('engine', {}).get('release')}")
            if h.get("engine", {}).get("release") != inv.engine.release:
                line("FAIL", "engine release differs from the host", "uv run hypercompute engine install")
        except AgentError as exc:
            line("FAIL", str(exc), exc.fix)
    say("All checks passed." if problems == 0 else f"{problems} check(s) failed.")
    return 0 if problems == 0 else 1


def cmd_inventory(args: argparse.Namespace) -> int:
    inv = inventory.collect(include_engine=not args.no_engine)
    say(inv.model_dump_json(indent=2))
    return 0


def cmd_invite(args: argparse.Namespace) -> int:
    """Print a fresh invitation from the locally running gateway."""
    base = f"http://127.0.0.1:{args.port}"
    try:
        with httpx.Client(trust_env=False, timeout=10) as client:
            owner = client.get(f"{base}/api/owner/bootstrap")
            owner.raise_for_status()
            response = client.post(f"{base}/api/network/invite",
                                   headers={config.OWNER_HEADER: owner.json()["owner_token"]})
            response.raise_for_status()
            invite = response.json()
            say(invite["join_command"])
            say(f"Expires: {invite['expires_at']}")
        return 0
    except (httpx.HTTPError, KeyError) as exc:
        raise AgentError(f"Could not create an invitation: {exc}",
                         fix="start this version with `uv run hypercompute host --transport iroh`") from exc


# ----------------------------------------------------------------------------- parser


def cmd_tune(args: argparse.Namespace) -> int:
    try:
        from .tune_cli import run_tune
    except ModuleNotFoundError:
        return _fail("the measured tuner is not part of this build")
    return int(run_tune(args))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hypercompute", description="Pool friends' computers to run one local model.")
    p.add_argument("--version", action="version", version=f"hypercompute {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    h = sub.add_parser("host", help="start the coordinator and dashboard")
    h.add_argument("--port", type=int, default=config.DEFAULT_GATEWAY_PORT)
    h.add_argument("--bind", action="append", help="LAN IP to listen on (repeatable; default: auto)")
    h.add_argument("--no-browser", action="store_true")
    h.add_argument("--transport", choices=["lan", "iroh"], default="lan",
                   help="LAN (default), or encrypted Iroh connections with relay fallback")
    h.add_argument("--relay-url", help="custom HTTPS Iroh relay (default: public Iroh relays)")
    h.add_argument("--local-max-memory", help="budget for this machine's own GPU, e.g. 8GiB")
    h.add_argument("--local-devices", help="local engine devices to use, e.g. CUDA0")
    h.set_defaults(func=cmd_host)

    j = sub.add_parser("join", help="share this computer with a host")
    address = j.add_mutually_exclusive_group(required=True)
    address.add_argument("--host", help="host LAN address, e.g. 192.168.1.20:8000")
    address.add_argument("--ticket", help="Iroh endpoint ticket from the host's invitation")
    j.add_argument("--relay-url", help="custom HTTPS Iroh relay (with --ticket)")
    j.add_argument("--code", required=True, help="one-time invitation code from the host's dashboard")
    j.add_argument("--name", help="display name shown to the host")
    j.add_argument("--max-memory", help="most memory the host may plan onto this machine, e.g. 8GiB")
    j.add_argument("--reserve", help="memory to keep free for yourself, e.g. 6GiB")
    j.add_argument("--devices", help="engine devices to share, e.g. CUDA0 or MTL0 (default: all GPUs)")
    j.add_argument("--worker-port", type=int, default=config.DEFAULT_WORKER_PORT)
    j.add_argument("--advertise-ip", help="LAN IP the host should connect to (default: auto)")
    j.add_argument("--cache", action="store_true", help="keep received tensors on disk to speed up reloads")
    j.add_argument("--yes", action="store_true", help="accept the shown contribution without prompting")
    j.add_argument("--managed", action="store_true",
                   help="let the host restart this machine's worker within --devices-allowed and, with --max-disk, "
                        "download catalog model tensors into its RPC cache (shown before you confirm)")
    j.add_argument("--devices-allowed", help="devices the host may switch the worker to, with --managed "
                                             "(default: MTL0,CPU on macOS, else the shared devices plus CPU)")
    j.add_argument("--max-disk", help="most disk the host may fill in the RPC cache, with --managed, e.g. 80GiB "
                                      "(default 0: no downloads); at least 20 GiB always stays free")
    j.add_argument("--exo", action="store_true",
                   help="macOS: also run EXO so the host can serve MLX models here (engine install --engine exo)")
    j.set_defaults(func=cmd_join)

    s = sub.add_parser("status", help="show sharing/hosting status")
    s.add_argument("--port", type=int, default=config.DEFAULT_GATEWAY_PORT)
    s.set_defaults(func=cmd_status)

    invite = sub.add_parser("invite", help="print a fresh one-use invitation from the local host")
    invite.add_argument("--port", type=int, default=config.DEFAULT_GATEWAY_PORT)
    invite.set_defaults(func=cmd_invite)

    for name, text in (("pause", "temporarily stop the worker but stay in the session"),
                       ("resume", "restart the worker after a pause")):
        pr = sub.add_parser(name, help=text)
        pr.set_defaults(func=cmd_pause_resume)

    lv = sub.add_parser("leave", help="stop sharing this computer")
    lv.add_argument("--purge-cache", action="store_true", help="also delete the Hypercompute RPC tensor cache")
    lv.set_defaults(func=cmd_leave)

    d = sub.add_parser("doctor", help="check prerequisites and connectivity")
    d.add_argument("--host", help="also check reachability of a host, e.g. 192.168.1.20:8000")
    d.set_defaults(func=cmd_doctor)

    e = sub.add_parser("engine", help="install or inspect an inference engine (default: the pinned llama.cpp)")
    e.add_argument("action", choices=["install", "status"])
    e.add_argument("--engine", choices=["llama", "airllm", "vllm", "exo"], default="llama",
                   help="llama (default), airllm (disk capacity mode), vllm (NVIDIA Linux/WSL2), exo (Apple MLX)")
    e.add_argument("--variant", help="override the auto-selected build variant")
    e.add_argument("--yes", action="store_true")
    e.add_argument("--force", action="store_true", help="reinstall even if present")
    e.set_defaults(func=cmd_engine)

    m = sub.add_parser("models", help="list or download curated models")
    m.add_argument("action", choices=["list", "pull"])
    m.add_argument("model_id", nargs="?")
    m.set_defaults(func=cmd_models)

    t = sub.add_parser("tune", help="measure candidate layer splits for a model and save the fastest")
    t.add_argument("model_id")
    t.add_argument("--port", type=int, default=config.DEFAULT_GATEWAY_PORT)
    t.add_argument("--nodes", help="comma-separated node ids to include (default: every ready peer)")
    t.add_argument("--objective", choices=["interactive", "throughput"], default="interactive")
    t.add_argument("--ctx", type=int, default=4096)
    t.add_argument("--max-candidates", type=int, default=4)
    t.add_argument("--no-flags", action="store_true", help="only vary the split, not flash attention / batch size")
    t.add_argument("--decode-tokens", type=int, default=None, help="tokens generated per decode measurement")
    # the owner's plan limits (tune_cli.request_body), so the winner is a plan their runs accept and replay
    t.add_argument("--engine", choices=list(get_args(EngineId)), default=None,
                   help="engine to measure (default: llama_rpc, the only one whose winner is saved as a profile)")
    t.add_argument("--strategy", choices=[s.value for s in PlanStrategy], default=None,
                   help="plan strategy, as in your runs (default: auto)")
    t.add_argument("--no-ram-tier", action="store_true", help="never keep layers in host RAM, as in your runs")
    t.add_argument("--allow-capacity-mode", action="store_true",
                   help="allow disk-paged tiers in the plans, as in your runs (default: off)")
    t.set_defaults(func=cmd_tune)
    add_capacity_parser(sub)
    add_image_parser(sub)

    i = sub.add_parser("inventory", help="print this machine's normalized hardware inventory")
    i.add_argument("--no-engine", action="store_true")
    i.set_defaults(func=cmd_inventory)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.command == "models" and args.action == "pull" and not args.model_id:
        return _fail("models pull needs a model id", "uv run hypercompute models list")
    try:
        return int(args.func(args) or 0)
    except AgentError as exc:
        return _fail(str(exc), exc.fix)
    except engine.EngineError as exc:
        return _fail(str(exc), exc.fix)
    except instance.InstanceLocked as exc:
        return _fail(str(exc), exc.fix)
    except ModelError as exc:
        return _fail(str(exc), "check configs/models.yaml (or models.example.yaml) and quote values that contain ': '")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
