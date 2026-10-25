"""Exercise Iroh between two fresh Python processes, using real QUIC sockets.

Run from the repository: uv run python scripts/check_iroh_transport.py
Use --local for an offline loopback test, or --relay-url for a private relay.
The default test bootstraps through a public relay and reports the observed path;
it does not claim to force relay-only traffic. No model or gateway is exposed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from hypercompute.transport.iroh import IrohConnection, IrohEndpoint


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


async def child(mode: str, directory: Path, local: bool, relay_url: str | None) -> None:
    async def echo(connection: IrohConnection) -> None:
        stream = await connection.accept_bi()
        message = await stream.receive_json()
        await stream.send_json({"echo": message, "authenticated_peer": connection.remote_id})
        await stream.finish()
        # Preserve the connection while the client consumes the response.
        await connection._raw.closed()

    endpoint = IrohEndpoint(
        directory / f"{mode}.key",
        relay_url=relay_url,
        relays_enabled=not local,
        bind_addr="127.0.0.1:0" if local else None,
        connect_timeout=25,
    )
    try:
        await endpoint.start(echo)
        if mode == "server":
            emit({"ticket": endpoint.address, "endpoint_id": endpoint.endpoint_id})
            await asyncio.to_thread(sys.stdin.readline)
        else:
            request = json.loads(await asyncio.to_thread(sys.stdin.readline))
            connection = await endpoint.connect(request["ticket"])
            initial_path = connection.path_info()
            stream = await connection.open_bi()
            payload = {"message": "hypercompute cross-process QUIC", "process": os.getpid()}
            await stream.send_json(payload)
            response = await stream.receive_json()
            assert response["echo"] == payload
            assert response["authenticated_peer"] == endpoint.endpoint_id
            assert connection.remote_id == request["endpoint_id"]
            emit({
                "ok": True,
                "client_pid": os.getpid(),
                "authenticated_host": connection.remote_id,
                "initial_path": initial_path,
                "final_path": connection.path_info(),
            })
            connection.close()
    finally:
        await endpoint.close()


async def launch(mode: str, directory: str, local: bool, relay_url: str | None) -> asyncio.subprocess.Process:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", mode, "--directory", directory]
    if local:
        command.append("--local")
    if relay_url:
        command.extend(["--relay-url", relay_url])
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


async def smoke(local: bool, relay_url: str | None) -> None:
    with tempfile.TemporaryDirectory(prefix="hypercompute-iroh-smoke-") as directory:
        server = await launch("server", directory, local, relay_url)
        client = None
        try:
            assert server.stdout is not None
            async with asyncio.timeout(35):
                line = await server.stdout.readline()
            if not line:
                _, errors = await server.communicate()
                raise RuntimeError(errors.decode(errors="replace"))
            ticket = json.loads(line)
            client = await launch("client", directory, local, relay_url)
            async with asyncio.timeout(40):
                output, errors = await client.communicate(json.dumps(ticket).encode() + b"\n")
            if client.returncode:
                raise RuntimeError(errors.decode(errors="replace"))
            result = json.loads(output)
            result["server_pid"] = server.pid
            result["public_relay_bootstrap"] = not local and relay_url is None
            emit(result)
        finally:
            for process in (client, server):
                if process is not None and process.returncode is None:
                    try:
                        async with asyncio.timeout(5):
                            await process.communicate(b"stop\n")
                    except (TimeoutError, BrokenPipeError, ConnectionResetError):
                        process.terminate()
                        await process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--relay-url")
    parser.add_argument("--child", choices=["server", "client"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.local and args.relay_url:
        parser.error("--local and --relay-url are mutually exclusive")
    if args.child:
        if args.directory is None:
            parser.error("--child requires --directory")
        asyncio.run(child(args.child, args.directory, args.local, args.relay_url))
    else:
        asyncio.run(smoke(args.local, args.relay_url))


if __name__ == "__main__":
    main()
