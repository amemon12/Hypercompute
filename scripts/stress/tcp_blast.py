"""Raw TCP throughput from the host to a friend (the direction model layers travel). Standard library only.

Friend (Mac):  uv run python tcp_blast.py --serve 47999        (click Allow if macOS asks)
Host (G14):    uv run python tcp_blast.py --send <MAC_IP>:47999 --mb 1000
Port 47999 is not Hypercompute's gateway (8000) or worker (47052) port.
"""

from __future__ import annotations

import argparse
import socket
import time

CHUNK = 1 << 20


def serve(port: int) -> None:
    with socket.create_server(("0.0.0.0", port)) as srv:
        conn, peer = srv.accept()
        with conn:
            received, t0 = 0, time.perf_counter()
            while chunk := conn.recv(CHUNK):
                received += len(chunk)
            elapsed = time.perf_counter() - t0
    print(f"received {received / 1e6:.0f} MB from {peer[0]} in {elapsed:.1f} s = {received / 1e6 / elapsed:.1f} MB/s")


def send(target: str, mib: int) -> None:
    host, port = target.rsplit(":", 1)
    payload = b"\0" * CHUNK
    with socket.create_connection((host, int(port)), timeout=30) as sock:
        t0 = time.perf_counter()
        for _ in range(mib):
            sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        sock.recv(1)  # returns once the receiver has read everything and closed
        elapsed = time.perf_counter() - t0
    print(f"sent {mib * CHUNK / 1e6:.0f} MB in {elapsed:.1f} s = {mib * CHUNK / 1e6 / elapsed:.1f} MB/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", type=int)
    ap.add_argument("--send")
    ap.add_argument("--mb", type=int, default=1000, help="MiB to send")
    args = ap.parse_args()
    if args.serve:
        serve(args.serve)
    elif args.send:
        send(args.send, args.mb)
    else:
        ap.error("--serve PORT or --send HOST:PORT")


if __name__ == "__main__":
    main()
