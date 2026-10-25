"""EXO sidecar for `hypercompute join --exo` on a Mac: the pinned EXO node, supervised like the RPC worker.

EXO discovers peers by IPv6 link-local multicast in one discovery namespace (EXO_ZENOH_NAMESPACE). The namespace
is derived from the host gateway the agent joins, so only Macs sharing with the same host form one cluster. It
isolates clusters; it does not authenticate them. The EXO API has no authentication: LAN only, like ggml RPC.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import socket
import time
from collections.abc import Callable

import httpx

from .. import config
from ..installers import exo as exo_install
from ..schemas import ExtraEngineInfo
from .supervisor import StopOutcome, Supervisor

log = logging.getLogger("hypercompute.agent.exo")
EXO_KEY = "exo"
START_TIMEOUT_S = 180.0  # first start imports MLX and torch; later starts are faster
STOP_GRACE_S = 15.0  # EXO shuts its runners down on SIGTERM
PORT_TAKEN_FIX = "another EXO is running on this Mac (the EXO app or a previous run): quit it first"
FAILURE_HINTS = (
    ("is already in use", PORT_TAKEN_FIX),
    ("Local Network", "allow Local Network access in System Settings > Privacy & Security > Local Network"),
    ("Unable to locate dashboard", f"the dashboard build is missing: {exo_install.INSTALL_HINT} --force"),
)


class ExoSidecarError(RuntimeError):
    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


def namespace_for(host_ip: str, port: int) -> str:
    """Same commit + same host gateway -> same namespace; another host or another EXO build -> another one."""
    digest = hashlib.sha256(f"hypercompute-exo|{host_ip}:{port}".encode()).hexdigest()[:16]
    return f"hc-{exo_install.EXO_PIN[:7]}-{digest}"


def security_warning(advertise_ip: str) -> str:
    return (f"Security: EXO's API on {advertise_ip}:{exo_install.API_PORT} has no authentication, and its cluster "
            "ports (UDP 52413, TCP 52414, TCP 49153-65535) accept any peer. Anyone on this network can start or "
            "delete models and trigger downloads on this Mac, and EXO's built-in model cards trust model code from "
            "Hugging Face (trust_remote_code), so a model started that way may run its repo's code here. Share only "
            "on a private LAN you trust, like the RPC worker; a firewall allowing only the host and the other Macs is "
            "the only real restriction.")


def explain_failure(log_text: str) -> str | None:
    return next((fix for needle, fix in FAILURE_HINTS if needle in log_text), None)


def api_port_free(port: int) -> bool:
    """Bind the way EXO does (api/listening_socket.py: 0.0.0.0 with SO_REUSEADDR), so only a live listener counts,
    not a port our last EXO left in TIME_WAIT. Windows SO_REUSEADDR would bind over a live listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError as exc:
            return exc.errno != errno.EADDRINUSE  # any other bind error is EXO's own to report
    return True


class ExoSidecar:
    def __init__(self, sup: Supervisor, paths: config.Paths, advertise_ip: str, namespace: str,
                 get: Callable[..., httpx.Response] = httpx.get) -> None:
        self._sup, self._paths, self._get = sup, paths, get
        self.advertise_ip, self.namespace = advertise_ip, namespace
        self.node_id: str | None = None  # EXO's own node id; new on every EXO start

    @property
    def endpoint(self) -> str:
        return f"{self.advertise_ip}:{exo_install.API_PORT}"

    def args(self) -> list[str]:
        return ["-m", "exo", "-v", "--api-port", str(exo_install.API_PORT)]  # src/exo/__main__.py

    def start(self, timeout_s: float = START_TIMEOUT_S) -> ExtraEngineInfo:
        st = exo_install.status(self._paths)
        if not st.installed:
            raise ExoSidecarError(f"EXO is not ready on this Mac: {st.detail}", fix=exo_install.INSTALL_HINT)
        self._require_free_port()
        log_path = self._paths.agent / "logs" / "exo.log"
        proc = self._sup.start(EXO_KEY, exo_install.venv_python(self._paths), self.args(),
                               exo_install.src_dir(self._paths), log_path, {"EXO_ZENOH_NAMESPACE": self.namespace})
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            node = self._probe()
            if not proc.alive():  # checked after the probe: an answer counts only while our EXO still runs
                text = proc.read_log()
                raise ExoSidecarError(f"EXO exited during startup (code {proc.exit_code()}): {text[-600:]}",
                                      fix=explain_failure(text) or f"see {log_path}")
            if node:
                self.node_id = node
                return self.info()
            time.sleep(1.0)
        text = proc.read_log()
        self._sup.stop(EXO_KEY, STOP_GRACE_S)
        raise ExoSidecarError(f"EXO's API did not answer on port {exo_install.API_PORT} within {timeout_s:.0f}s",
                              fix=explain_failure(text) or f"see {log_path}")

    def _require_free_port(self) -> None:
        """Any EXO already on the API port answers the readiness probe as if it were ours (our EXO only binds after
        seconds of imports), so start only on a free port and never adopt another EXO's node id."""
        port = exo_install.API_PORT
        if api_port_free(port):
            return
        node = self._probe()
        if node:
            raise ExoSidecarError(f"another EXO (node {node}) already serves port {port}", fix=PORT_TAKEN_FIX)
        raise ExoSidecarError(f"port {port}, EXO's API port, is already in use",
                              fix=f"quit what holds it (lsof -nP -iTCP:{port} -sTCP:LISTEN shows it)")

    def _probe(self) -> str | None:
        """GET /node_id returns this node's own id as a bare JSON string once the API is up."""
        try:
            r = self._get(f"http://127.0.0.1:{exo_install.API_PORT}/node_id", timeout=3.0)
            value = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
        return value if isinstance(value, str) and value else None

    def stop(self) -> StopOutcome | None:
        self.node_id = None
        return self._sup.stop(EXO_KEY, STOP_GRACE_S)

    def info(self) -> ExtraEngineInfo:
        """What the host plans with: running + the LAN endpoint, or why EXO is not serving."""
        proc = self._sup.get(EXO_KEY)
        if proc is None:
            return ExtraEngineInfo(engine="exo", installed=True, status="installed", version=exo_install.EXO_PIN,
                                   detail="EXO stopped (sharing paused or not started)")
        if not proc.alive():
            return ExtraEngineInfo(engine="exo", installed=True, status="failed", version=exo_install.EXO_PIN,
                                   detail=f"EXO exited (code {proc.exit_code()}); see {proc.log_path}")
        return ExtraEngineInfo(engine="exo", installed=True, status="running", version=exo_install.EXO_PIN,
                               endpoint=self.endpoint,
                               detail=f"EXO {exo_install.EXO_VERSION} node {self.node_id or '?'}, namespace "
                                      f"{self.namespace}")


def with_exo(engines: list[ExtraEngineInfo], info: ExtraEngineInfo) -> list[ExtraEngineInfo]:
    """The inventory's optional-engine list with the EXO entry replaced by the sidecar's live status."""
    return [*(e for e in engines if e.engine != "exo"), info]
