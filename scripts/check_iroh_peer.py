"""Portable cross-process application smoke with fixture inventory and a TCP echo worker.

Uses real session/node services, real Iroh, and the real PeerClient. No engine or
GPU is required. Run: uv run python scripts/check_iroh_peer.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from factories import contribution, make_inventory  # noqa: E402
from hypercompute import __version__  # noqa: E402
from hypercompute.schemas import JoinRequest, WorkerEndpoint  # noqa: E402
from hypercompute.services.events import EventBus  # noqa: E402
from hypercompute.services.nodes import NodeRegistry  # noqa: E402
from hypercompute.services.sessions import SessionService  # noqa: E402
from hypercompute.storage import Store  # noqa: E402
from hypercompute.transport.client import PeerClient  # noqa: E402
from hypercompute.transport.host import HostService  # noqa: E402
from hypercompute.transport.iroh import IrohEndpoint  # noqa: E402

NODE_ID = "1" * 32
PAYLOAD = b"cross-process Hypercompute RPC echo\0" * 8192


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


async def server(directory: Path) -> None:
    store = Store(directory / "smoke.db", directory / "records")
    events = EventBus(store)
    secret = secrets.token_bytes(32)
    sessions = SessionService(store, events, secret, 8000, lambda: ["127.0.0.1"])
    inventory = make_inventory()
    nodes = NodeRegistry(store, events, sessions, secret, inventory.engine, inventory, contribution(["CUDA0"]))
    # The coordinator's unrelated model/deployment services are not part of this protocol smoke.
    coord = SimpleNamespace(nodes=nodes, sessions=sessions, engine_info=inventory.engine)
    endpoint = IrohEndpoint(directory / "host.key", relays_enabled=False, bind_addr="127.0.0.1:0")
    service = HostService(coord, endpoint)
    try:
        await endpoint.start(service._handle_connection)
        invitation = sessions.create("portable Iroh smoke").invite
        assert invitation is not None
        emit({"ticket": service.ticket, "code": invitation.code})
        while command := await asyncio.to_thread(sys.stdin.readline):
            if command.strip() == "probe":
                view = nodes.get(NODE_ID)
                assert view is not None and view.worker is not None
                assert view.worker.host == "127.0.0.1"
                reader, writer = await asyncio.open_connection(view.worker.host, view.worker.port)
                try:
                    writer.write(PAYLOAD)
                    await writer.drain()
                    writer.write_eof()
                    async with asyncio.timeout(5):
                        reply = await reader.read()
                    emit({"echo_ok": reply == PAYLOAD[::-1], "bytes": len(reply)})
                finally:
                    writer.close()
                    await writer.wait_closed()
            else:
                break
    finally:
        await service.close()
        store.close()


async def smoke() -> None:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await reader.read()
            writer.write(data[::-1])
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    with tempfile.TemporaryDirectory(prefix="hypercompute-peer-smoke-") as directory:
        worker = await asyncio.start_server(echo, "127.0.0.1", 0)
        worker_port = worker.sockets[0].getsockname()[1]
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), "--server", directory,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        client = None
        try:
            assert process.stdout is not None and process.stdin is not None
            async with asyncio.timeout(15):
                ready = await process.stdout.readline()
            if not ready:
                _, errors = await process.communicate()
                raise RuntimeError(errors.decode(errors="replace"))
            info = json.loads(ready)
            client = await asyncio.to_thread(PeerClient, info["ticket"], Path(directory) / "guest.key",
                                             relays_enabled=False)
            health = await client.arequest("health")
            assert health.status_code == 200
            owner = await client.arequest("owner/bootstrap")
            assert owner.status_code == 404
            unauthenticated = await client.arequest("heartbeat", node_id=NODE_ID, credential="invalid", body={})
            assert unauthenticated.status_code == 403
            await asyncio.to_thread(client.configure_worker, NODE_ID, worker_port)
            request = JoinRequest(
                code="invalid", node_id=NODE_ID, display_name="portable echo worker",
                inventory=make_inventory(), contribution=contribution(["CUDA0"]),
                worker=WorkerEndpoint(host="203.0.113.5", port=12345), agent_version=__version__,
            ).model_dump(mode="json")
            refused = await client.arequest("join", body=request)
            assert refused.status_code == 403
            request["code"] = info["code"]
            joined = await client.arequest("join", body=request)
            assert joined.status_code == 200, joined.text
            credential = joined.json()["credential"]
            process.stdin.write(b"probe\n")
            await process.stdin.drain()
            async with asyncio.timeout(10):
                probe = json.loads(await process.stdout.readline())
            assert probe["echo_ok"], probe
            left = await client.arequest("leave", node_id=NODE_ID, credential=credential)
            assert left.status_code == 200
            rejected = await client.arequest("heartbeat", node_id=NODE_ID, credential=credential, body={})
            assert rejected.status_code == 403
            emit({"ok": True, "checks": ["health", "owner_route_denied", "credential_required", "invite_required",
                                          "join", "fixed_worker_rpc_echo", "leave", "revoked_credential"],
                  "echo_bytes": probe["bytes"], "host_pid": process.pid, "guest_pid": os.getpid()})
        finally:
            if client is not None:
                await asyncio.to_thread(client.close)
            worker.close()
            await worker.wait_closed()
            if process.returncode is None:
                try:
                    async with asyncio.timeout(5):
                        _, errors = await process.communicate(b"stop\n")
                    if process.returncode:
                        raise RuntimeError(errors.decode(errors="replace"))
                except (TimeoutError, BrokenPipeError, ConnectionResetError):
                    process.terminate()
                    await process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path)
    args = parser.parse_args()
    asyncio.run(server(args.server) if args.server else smoke())


if __name__ == "__main__":
    main()
