"""Serve only the G14's hypercompute PUBLIC key at http://192.168.1.162:8765/k for the GX10 to append, then exit.

Logs the ?u= query (the GX10 username) so the operator can SSH in. Nothing else is served; exits after the first
successful fetch or after 45 minutes.
"""
from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PUB = Path.home() / ".ssh" / "hypercompute_ed25519.pub"
KEY = PUB.read_text(encoding="utf-8").strip()
if not KEY.startswith("ssh-ed25519 ") or "PRIVATE" in KEY:
    sys.exit("refusing: not a public key")
BODY = ("\n" + KEY + "\n").encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        if url.path != "/k":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)
        user = parse_qs(url.query).get("u", [""])[0]
        print(f"served key to {self.client_address[0]} user={user}", flush=True)
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.client_address[0]} {fmt % args}", flush=True)


def main() -> int:
    srv = HTTPServer(("192.168.1.162", 8765), Handler)
    threading.Timer(45 * 60, srv.shutdown).start()
    print("listening on 192.168.1.162:8765", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
