"""Test-day helper: hold N idle TCP connections to the gateway (no request bytes), then close them.

uvicorn counts open connections against limit_concurrency (100) from connection_made on, with no header timeout, so
N idle sockets make every later request on any connection answer 503 until they close (probe P-CONN, finding W1).
Run it from the G14 (127.0.0.1) or from the Mac against the G14's LAN address to show it needs no credentials.

Usage:  uv run python scripts/stress/hold.py N HOST PORT SECONDS
  e.g.  uv run python scripts/stress/hold.py 100 192.168.1.20 8000 40
Point it only at your own gateway.
"""

from __future__ import annotations

import socket
import sys
import time


def main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help") else 2)
    try:
        n, host, port, secs = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    except ValueError as exc:
        print(f"bad argument: {exc}\n{__doc__}", file=sys.stderr)
        sys.exit(2)
    socks: list[socket.socket] = []
    try:
        for _ in range(n):
            socks.append(socket.create_connection((host, port), timeout=5))
        print(f"{time.strftime('%H:%M:%S')} holding {len(socks)} idle connections to {host}:{port} for {secs:.0f}s",
              flush=True)
        time.sleep(secs)
    finally:
        for s in socks:
            s.close()
        print(f"{time.strftime('%H:%M:%S')} closed {len(socks)} connections", flush=True)


if __name__ == "__main__":
    main()
