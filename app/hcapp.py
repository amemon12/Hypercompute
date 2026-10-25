"""Hypercompute demo app: one local page to host a pool or join one, styled like hypercompute.co.

Run from the repo root: uv run --no-sync python app/hcapp.py [--port 8790] [--no-browser]

Security: it binds 127.0.0.1 only, every /api call needs the per-launch token from the URL it prints, the Host header must
be the loopback address it serves (blocks DNS rebinding) and a cross-origin Origin is refused. The owner token never
reaches the page; the invite does, only when you ask for it, so you can copy it to a friend.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "app" / "static"
GATEWAY = "http://127.0.0.1:8001"
DISK_TIERS = {"gpu_disk", "stream"}
TERMINAL_STATES = {"stopped", "failed"}
INVITE_RE = re.compile(r"--ticket\s+([A-Za-z0-9]{30,4096})\s+--code\s+([A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5})")
MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
UV = os.environ.get("HC_UV", "uv")


def hc(*args: str) -> list[str]:
    return [UV, "run", "--no-sync", "hypercompute", *args]


class Backend:
    """The real pool: the hypercompute CLI, the one-click join helper and the SDK. Tests swap in a fake."""

    def __init__(self) -> None:
        sys.path.insert(0, str(REPO / "sdk" / "python" / "src"))
        from hypercompute_sdk import Pool  # noqa: PLC0415
        self._pool_cls, self._pool = Pool, None
        self.host_proc: subprocess.Popen | None = None
        self.job: dict[str, Any] = {"state": "idle"}
        self.deployment: str | None = None
        self._observed_deployment: str | None = None
        self._peer_names: dict[str, str] = {}
        self._run_lock = threading.Lock()

    def pool(self) -> Any:
        if self._pool is None:
            self._pool = self._pool_cls()
        return self._pool

    def host_up(self) -> bool:
        try:
            import httpx  # noqa: PLC0415
            return httpx.get(f"{GATEWAY}/api/health", timeout=2, trust_env=False).status_code == 200
        except Exception:  # noqa: BLE001 - down is the answer, not an error
            return False

    def state(self) -> dict[str, Any]:
        up = self.host_up()
        out: dict[str, Any] = {"host_up": up, "job": self.job, "peers": [], "models": [], "sharing": self.sharing()}
        if not up:
            return out
        try:
            pool = self.pool()
            try:
                raw = pool.pools()
                rows = raw.get("pools", []) if isinstance(raw, dict) else raw
                pools = {p.get("node_id"): p for p in rows if isinstance(p, dict)}
            except Exception:  # noqa: BLE001 - memory is a nice-to-have on the page
                pools = {}
            peers = pool.peers()
            self._peer_names = {p["node_id"]: p.get("display_name") or "friend" for p in peers if p.get("node_id")}
            for p in peers:
                cap = pools.get(p.get("node_id"), {})
                out["peers"].append({"name": p.get("display_name"), "state": p.get("state"),
                                     "devices": p.get("contribution_devices"),
                                     "admissible_gib": round((cap.get("admissible_bytes") or 0) / 2**30, 1)})
            out["models"] = [m["artifact"]["id"] for m in pool.models() if m.get("status") == "downloaded"]
            deployments = pool.deployments()
            active = self._active_deployment(deployments)
            if active:
                self._show_deployment(active)
            elif self._observed_deployment:
                ended = next((d for d in deployments if d.get("id") == self._observed_deployment), None)
                if self.deployment == self._observed_deployment:
                    self.deployment = None
                self._observed_deployment = None
                self.job = ({"state": "failed", "model": (ended.get("plan") or {}).get("model_id"),
                             "detail": str(ended.get("error") or "model failed")[:300]}
                            if ended and ended.get("state") == "failed" else {"state": "idle"})
            out["job"] = self.job
        except Exception as exc:  # noqa: BLE001 - shown on the page
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    @staticmethod
    def _active_deployment(deployments: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((d for d in deployments if d.get("state") not in TERMINAL_STATES
                     and d.get("purpose", "run") == "run"), None)

    def _stages(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"node": "local" if a["node_id"] == "local" else self._peer_names.get(a["node_id"], "friend"),
                 "device": a["device"], "layers": len(a.get("layers") or [])}
                for a in plan.get("assignments", []) if a.get("layers")]

    def _show_deployment(self, dep: dict[str, Any]) -> None:
        plan = dep.get("plan") or {}
        self._observed_deployment = dep["id"]
        self.job = {"state": dep["state"], "model": plan.get("model_id"), "tier": plan.get("tier"),
                    "stages": self._stages(plan), "progress": dep.get("progress"),
                    "can_stop": dep["id"] == self.deployment}

    def sharing(self) -> str:
        log = Path.home() / ".hypercompute" / "oneclick" / "join.log"
        if not log.is_file():
            return "not sharing"
        lines = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        return lines[-1][-160:] if lines else "not sharing"

    def start_host(self) -> str:
        if self.host_up():
            return "already hosting"
        log_path = Path.home() / ".hypercompute" / "app-host.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as logf:
            self.host_proc = subprocess.Popen(hc("host", "--transport", "iroh", "--port", "8001", "--no-browser"),
                                              cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for _ in range(60):
            if self.host_up():
                return "hosting"
            time.sleep(1)
        return "starting (check ~/.hypercompute/app-host.log)"

    def stop_host(self) -> str:
        if self.host_proc is None or self.host_proc.poll() is not None:
            return "this app did not start a host"
        self.host_proc.terminate()
        return "host stopped"

    def invite(self) -> str:
        out = subprocess.run(hc("invite", "--port", "8001"), cwd=REPO, capture_output=True, text=True, timeout=60)
        m = INVITE_RE.search(out.stdout + out.stderr)
        if not m:
            raise ValueError("the host did not print an invite; is it running?")
        return f"hypercompute join --ticket {m.group(1)} --code {m.group(2)}"

    def join(self, invite: str) -> str:
        env = {**os.environ, "HC_INVITE": invite, "HC_ONECLICK_YES": "1", "HC_MANAGED": "1",
               "HC_NAME": re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname())[:32] or "friend"}
        subprocess.Popen([UV, "run", "--no-sync", "python", "join/lib/oneclick.py", "join", "--repo", str(REPO)],
                         cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return "joining"

    def leave(self) -> str:
        subprocess.run([UV, "run", "--no-sync", "python", "join/lib/oneclick.py", "leave", "--repo", str(REPO)],
                       cwd=REPO, env={**os.environ, "HC_ONECLICK_YES": "1"}, input="n\n\n", text=True,
                       capture_output=True, timeout=120)  # "n": keep the model cache
        return "stopped sharing"

    def run_model(self, model: str) -> None:
        if not self._run_lock.acquire(blocking=False):
            return
        try:
            self._run_model(model)
        finally:
            self._run_lock.release()

    def _run_model(self, model: str) -> None:
        pool = self.pool()
        active = self._active_deployment(pool.deployments())
        if active:
            self._show_deployment(active)
            return
        self._observed_deployment = None
        self.job = {"state": "planning", "model": model}
        plan = pool.plan(model, strategy="all_selected", allow_capacity_mode=True)
        stages = self._stages(plan)
        if not plan["feasible"]:
            self.job = {"state": "does not fit", "model": model, "detail": (plan.get("reasons") or ["no plan"])[-1]}
            return
        if plan.get("tier") in DISK_TIERS:
            self.job = {"state": "refused", "model": model, "detail": f"tier {plan.get('tier')} pages from disk"}
            return
        dep = pool.deploy(plan["id"])
        self.deployment = dep["id"]
        self.job = {"state": "loading", "model": model, "tier": plan.get("tier"), "stages": stages, "can_stop": True}
        try:
            pool.wait_ready(dep["id"], on_progress=lambda s, p: self.job.update(progress=f"{s} {p or ''}".strip()))
        except Exception:
            if self.deployment == dep["id"]:
                raise
            return  # A stop while loading is an intentional cancellation.
        if self.deployment == dep["id"]:
            self.job = {**self.job, "state": "ready"}

    def stop_model(self) -> str:
        if not self.deployment:
            return "this app did not start the active model; stop it from the SDK or CLI that loaded it"
        dep_id, self.deployment = self.deployment, None
        try:
            self.pool().stop(dep_id)
        except Exception:
            self.deployment = dep_id
            raise
        self._observed_deployment = None
        self.job = {"state": "idle"}
        return "stopped"

    def chat(self, prompt: str) -> Iterator[dict[str, Any]]:
        for chunk in self.pool().stream([{"role": "user", "content": prompt}], max_tokens=256, temperature=0,
                                        chat_template_kwargs={"thinking": False}):
            yield {"text": chunk.content or ""}
            if chunk.timings:
                yield {"tok_s": chunk.timings.get("predicted_per_second")}


def make_handler(backend: Any, token: str, port: int) -> type[BaseHTTPRequestHandler]:
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:  # no request logs: URLs carry the token
            return

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode())

        def _allowed(self) -> bool:
            if self.headers.get("Host") not in hosts:
                return False
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc not in hosts:
                return False
            return secrets.compare_digest(self.headers.get("X-App-Token", ""), token)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if self.headers.get("Host") not in hosts:
                return self._send(HTTPStatus.FORBIDDEN, b"{}")
            if path == "/api/state":
                return self._json(200, backend.state()) if self._allowed() else self._json(403, {"error": "forbidden"})
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            ctype = {".html": "text/html; charset=utf-8", ".svg": "image/svg+xml", ".css": "text/css",
                     ".woff2": "font/woff2", ".png": "image/png", ".js": "text/javascript"}.get(target.suffix,
                                                                                         "application/octet-stream")
            return self._send(200, target.read_bytes(), ctype)

        def do_POST(self) -> None:  # noqa: N802
            if not self._allowed():
                return self._json(403, {"error": "forbidden"})
            try:
                size = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(size, 65536)) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad json"})
            path = urlparse(self.path).path
            try:
                if path == "/api/chat":
                    return self._stream(backend.chat(str(body.get("prompt", ""))[:4000]))
                return self._json(200, {"result": self._act(path, body)})
            except Exception as exc:  # noqa: BLE001 - a friendly line for the page
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"[:300]})

        def _act(self, path: str, body: dict[str, Any]) -> Any:
            if path == "/api/host/start":
                return backend.start_host()
            if path == "/api/host/stop":
                return backend.stop_host()
            if path == "/api/invite":
                return backend.invite()
            if path == "/api/join":
                m = INVITE_RE.search(str(body.get("invite", "")))
                if not m:
                    raise ValueError("that does not look like a Hypercompute invite")
                return backend.join(f"hypercompute join --ticket {m.group(1)} --code {m.group(2)}")
            if path == "/api/leave":
                return backend.leave()
            if path == "/api/run":
                model = str(body.get("model", ""))
                if not MODEL_RE.match(model):
                    raise ValueError("pick a model from the list")
                threading.Thread(target=self._guarded, args=(backend.run_model, model), daemon=True).start()
                return "started"
            if path == "/api/stop-model":
                return backend.stop_model()
            raise ValueError("unknown action")

        @staticmethod
        def _guarded(fn: Callable[[str], None], arg: str) -> None:
            try:
                fn(arg)
            except Exception as exc:  # noqa: BLE001 - shown on the page through state()
                backend.job = {"state": "failed", "detail": f"{type(exc).__name__}: {exc}"[:300]}

        def _stream(self, events: Iterator[dict[str, Any]]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for ev in events:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

    return Handler


def serve(backend: Any, port: int = 0) -> tuple[ThreadingHTTPServer, str, str]:
    token = secrets.token_urlsafe(24)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), None)  # type: ignore[arg-type]
    real_port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(backend, token, real_port)
    return httpd, token, f"http://127.0.0.1:{real_port}/#t={token}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    httpd, _, url = serve(Backend(), a.port)
    print(f"Hypercompute app on {url.split('#')[0]} (open the link this window opened; it carries this session's key)",
          flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    else:
        print(url, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
