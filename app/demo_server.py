"""Serve the simulated Hypercompute demo without starting or stopping real compute.

Local: python app/demo_server.py
LAN: python app/demo_server.py --bind 0.0.0.0 --public-url http://192.168.1.10:8792

The shared session contains fixture state only. Its join link is not a compute invite,
and this server never reads credentials or connects to the Hypercompute gateway.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import threading
import time
import uuid
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

STATIC = Path(__file__).resolve().parent / "static"
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class JoinRefused(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class DemoSession:
    """One simulated host and one simulated friend, shared across browser windows."""

    def __init__(self, public_url: str) -> None:
        self.public_url = public_url
        self.instance = uuid.uuid4().hex
        self.host_token = secrets.token_urlsafe(32)
        self.hosting = False
        self.joined = False
        self.version = 0
        self.session: str | None = None
        self.code: str | None = None
        self.lock = threading.RLock()
        self.failures: dict[str, deque[float]] = {}
        self.all_failures: deque[float] = deque()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "instance": self.instance,
                "hosting": self.hosting,
                "joined": self.joined,
                "version": self.version,
                "session": self.session,
                "code": self.code,
                "share_url": f"{self.public_url}/join" if self.hosting else None,
            }

    def guest_state(self) -> dict[str, Any]:
        with self.lock:
            return {"instance": self.instance, "hosting": self.hosting, "joined": self.joined,
                    "version": self.version, "can_join": self.hosting and not self.joined}

    def host(self, enabled: bool) -> dict[str, Any]:
        with self.lock:
            if enabled != self.hosting:
                self.hosting = enabled
                self.joined = False
                self.version += 1
                self.session = uuid.uuid4().hex if enabled else None
                self.code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(6)) if enabled else None
            return self.state()

    def join(self, code: str, address: str) -> dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            cutoff = now - 60
            self.failures = {ip: deque(t for t in tries if t > cutoff)
                             for ip, tries in self.failures.items() if tries and tries[-1] > cutoff}
            while self.all_failures and self.all_failures[0] <= cutoff:
                self.all_failures.popleft()
            if len(self.failures.get(address, ())) >= 5 or len(self.all_failures) >= 30:
                raise JoinRefused(429, "Too many attempts. Wait a minute before trying again.")
            if not self.hosting or not secrets.compare_digest((self.code or "------").encode(),
                                                             code.strip().upper().encode()):
                self.failures.setdefault(address, deque()).append(now)
                self.all_failures.append(now)
                raise JoinRefused(409, "That code is not active. Ask the host for the current code.")
            if not self.joined:
                self.joined = True
                self.version += 1
            return self.guest_state()


def _public_url(value: str) -> str:
    parsed = urlparse(value)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("--public-url must be an http://host:port address without a path")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("--public-url has an invalid port") from exc
    return value.rstrip("/")


def make_handler(session: DemoSession, allowed_hosts: set[str], static: Path) -> type[BaseHTTPRequestHandler]:
    roots = {"demo": (static / "demo").resolve(), "brand": (static / "brand").resolve()}
    origins = {f"http://{host}" for host in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            if status == 429:
                self.send_header("Retry-After", "60")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status: int, body: Any) -> None:
            self.send(status, json.dumps(body).encode("utf-8"))

        def allowed(self) -> bool:
            origin = self.headers.get("Origin")
            return self.headers.get("Host", "").lower() in allowed_hosts and (not origin or origin in origins)

        def host_authorized(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Demo-Host", "").encode(), session.host_token.encode())

        def do_GET(self) -> None:  # noqa: N802
            if not self.allowed():
                return self.json(403, {"error": "This address is not allowed."})
            path = unquote(urlparse(self.path).path)
            if path == "/api/state":
                if not self.host_authorized():
                    return self.json(403, {"error": "Open the private host launch link on this computer."})
                return self.json(200, session.state())
            if path == "/api/join-info":
                return self.json(200, session.guest_state())
            if path in ("/", "/join", "/join/"):
                root, relative = roots["demo"], "index.html"
            elif path.startswith("/brand/"):
                root, relative = roots["brand"], path.removeprefix("/brand/")
            else:
                root, relative = roots["demo"], path.lstrip("/")
            target = (root / relative).resolve()
            if root not in target.parents or not target.is_file():
                return self.json(404, {"error": "Page not found."})
            content_type = {
                ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2",
            }.get(target.suffix, mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            return self.send(200, target.read_bytes(), content_type)

        def do_POST(self) -> None:  # noqa: N802
            # Consume bounded requests before replying. Closing a socket with an unread
            # body can reset the connection on Windows and discard the error response.
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 4096:
                    raise ValueError
                payload = self.rfile.read(size)
            except ValueError:
                return self.json(400, {"error": "Send a JSON object of at most 4096 bytes."})
            if not self.allowed():
                return self.json(403, {"error": "This address is not allowed."})
            path = urlparse(self.path).path
            if path not in ("/api/hosting", "/api/join"):
                return self.json(404, {"error": "Action not found."})
            if path == "/api/hosting" and not self.host_authorized():
                return self.json(403, {"error": "Open the private host launch link on this computer."})
            if self.headers.get_content_type() != "application/json":
                return self.json(415, {"error": "Send application/json."})
            try:
                body = json.loads(payload)
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, UnicodeError):
                return self.json(400, {"error": "Send a JSON object of at most 4096 bytes."})
            if path == "/api/hosting":
                if type(body.get("enabled")) is not bool:
                    return self.json(400, {"error": "enabled must be true or false."})
                return self.json(200, session.host(body["enabled"]))
            if not isinstance(body.get("code"), str) or len(body["code"]) > 32:
                return self.json(400, {"error": "Enter the six-character code from the host."})
            try:
                return self.json(200, session.join(body["code"], self.client_address[0]))
            except JoinRefused as exc:
                return self.json(exc.status, {"error": str(exc)})

    return Handler


def serve(bind: str = "127.0.0.1", port: int = 8792, public_url: str | None = None,
          static: Path = STATIC, *, allowed_hosts: list[str] | None = None) -> tuple[ThreadingHTTPServer, str]:
    if bind == "0.0.0.0" and not public_url:
        raise ValueError("Use --public-url http://LAN_IP:PORT when binding to 0.0.0.0")
    if public_url:
        public_url = _public_url(public_url)
    extra_hosts = {urlparse(_public_url(f"http://{host}")).netloc.lower() for host in allowed_hosts or []}
    server = ThreadingHTTPServer((bind, port), None)  # type: ignore[arg-type]
    actual_port = server.server_address[1]
    public_url = public_url or f"http://{bind}:{actual_port}"
    hosts = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}", urlparse(public_url).netloc.lower(), *extra_hosts}
    if bind != "0.0.0.0":
        hosts.add(f"{bind}:{actual_port}".lower())
    session = DemoSession(public_url)
    server.RequestHandlerClass = make_handler(session, hosts, static)
    local = "127.0.0.1" if bind == "0.0.0.0" else bind
    return server, f"http://{local}:{actual_port}/#t={session.host_token}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address; use 0.0.0.0 for a LAN demo")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--public-url", help="Address friends can reach, for example http://192.168.1.10:8792")
    parser.add_argument("--allowed-host", action="append", default=[], help="Additional allowed host:port; repeatable")
    parser.add_argument("--host-url-file", type=Path, help="Write the private host launch URL to an ignored local file")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server, url = serve(args.bind, args.port, args.public_url, allowed_hosts=args.allowed_host)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.host_url_file:
        args.host_url_file.parent.mkdir(parents=True, exist_ok=True)
        args.host_url_file.write_text(url, encoding="utf-8")
        args.host_url_file.chmod(0o600)
    print(f"Hypercompute simulated demo: {url.split('#')[0]}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
