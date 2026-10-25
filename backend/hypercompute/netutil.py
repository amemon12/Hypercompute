"""Private-network helpers: interface selection, address classification and port bindability."""

from __future__ import annotations

import functools
import ipaddress
import json
import shutil
import socket
import subprocess
import sys

import psutil

from .schemas import NetworkInterface

_VIRTUAL_HINTS = ("vethernet", "virtualbox", "vmware", "hyper-v", "docker", "wsl", "veth", "br-", "virbr",
                  "utun", "awdl", "llw", "bridge", "tailscale", "zerotier", "loopback pseudo")
_WIFI_HINTS = ("wi-fi", "wifi", "wlan", "wireless", "airport")


def is_private_lan(ip: str) -> bool:
    """RFC1918 / CGNAT (100.64/10, used by overlays like Tailscale) / IPv6 ULA. Link-local is excluded."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return False
    if addr.version == 4 and addr in ipaddress.ip_network("100.64.0.0/10"):
        return True
    return addr.is_private


def is_link_local_v4(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and addr in ipaddress.ip_network("169.254.0.0/16")


def is_lan_or_link_local(ip: str) -> bool:
    """A private LAN address, or IPv4 link-local (169.254/16): a direct cable or Thunderbolt Bridge link."""
    return is_private_lan(ip) or is_link_local_v4(ip)


def resolve_ipv4(host: str) -> str | None:
    """The IPv4 address a name resolves to (firewall rules need an address, not a hostname)."""
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        return None
    return str(infos[0][4][0]) if infos else None


@functools.lru_cache(maxsize=1)
def _windows_adapter_descriptions() -> dict[str, str]:
    """Interface alias -> description (psutil only exposes aliases such as 'Ethernet' on Windows)."""
    exe = shutil.which("powershell")
    if sys.platform != "win32" or not exe:
        return {}
    try:
        out = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command",
                              "Get-NetAdapter | Select-Object Name, InterfaceDescription | ConvertTo-Json"],
                             capture_output=True, text=True, timeout=20, check=False)
        rows = json.loads(out.stdout or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}
    rows = rows if isinstance(rows, list) else [rows]
    return {r.get("Name", ""): r.get("InterfaceDescription") or "" for r in rows}


def classify_interface(name: str) -> str:
    low = (name + " " + _windows_adapter_descriptions().get(name, "")).lower()
    if low.startswith("lo") or "loopback" in low:
        return "loopback"
    if any(h in low for h in _VIRTUAL_HINTS):
        return "virtual"
    if any(h in low for h in _WIFI_HINTS) or low.startswith("en0"):
        return "wifi" if "wi" in low or low.startswith("wl") else "ethernet"
    if low.startswith(("eth", "en", "ethernet")):
        return "ethernet"
    return "other"


def list_interfaces() -> list[NetworkInterface]:
    stats = psutil.net_if_stats()
    out: list[NetworkInterface] = []
    for name, addrs in psutil.net_if_addrs().items():
        ipv4 = [a.address for a in addrs if a.family == socket.AF_INET]
        st = stats.get(name)
        speed = st.speed if st and st.speed else None
        out.append(NetworkInterface(name=name, ipv4=ipv4, is_up=bool(st and st.isup), speed_mbps=speed,
                                    kind=classify_interface(name)))  # type: ignore[arg-type]
    return out


def lan_addresses() -> list[str]:
    """Private IPv4 addresses on up interfaces: the default-route address first, virtual adapters last.

    IPv4 link-local addresses (a direct cable) are listed only when an up interface has nothing better.
    """
    rank = {"ethernet": 1, "wifi": 1, "other": 2, "virtual": 3}
    primary = route_source_ip("192.0.2.1")  # TEST-NET-1: only consults the routing table, sends nothing
    found: list[tuple[int, str]] = []
    link_local: list[tuple[int, str]] = []
    for iface in list_interfaces():
        if not iface.is_up or iface.kind == "loopback":
            continue
        for ip in iface.ipv4:
            entry = (0 if ip == primary else rank.get(iface.kind, 2), ip)
            if is_private_lan(ip):
                found.append(entry)
            elif is_link_local_v4(ip):
                link_local.append(entry)
    return [ip for _, ip in sorted(found or link_local)]


def route_source_ip(remote_host: str, remote_port: int = 80) -> str | None:
    """The local address the OS would use to reach `remote_host` (UDP connect sends no packets)."""
    try:
        infos = socket.getaddrinfo(remote_host, remote_port, socket.AF_INET, socket.SOCK_DGRAM)
    except socket.gaierror:
        return None
    for *_, sockaddr in infos:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.connect(sockaddr)
                return s.getsockname()[0]
            except OSError:
                continue
    return None


def can_bind(host: str, port: int) -> tuple[bool, str | None]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError as exc:
            return False, f"{exc.__class__.__name__}: {exc}"
    return True, None


def pick_bindable_port(host: str, preferred: int, scan: int) -> tuple[int | None, list[str]]:
    """First port in [preferred, preferred+scan) that binds. Returns attempts' errors for diagnostics."""
    errors: list[str] = []
    for port in range(preferred, preferred + scan):
        ok, err = can_bind(host, port)
        if ok:
            return port, errors
        errors.append(f"{port}: {err}")
    return None, errors


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def tcp_reachable(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
