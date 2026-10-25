"""Watch 192.168.1.0/24 for a new device (the GX10) and exit when one shows SSH, HTTP or a gx10-like name.

Usage: python -I lan_watch.py [max_minutes]
First pass records a baseline; later passes print and exit on a new host. Read-only probes (TCP connect, arp -a).
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

NET = "192.168.1."
SELF = "192.168.1.162"
PORTS = (22, 80, 443, 8080)
NAME_HINT = re.compile(r"gx10|spark|dgx|ubuntu|asus|promaxgb", re.I)


def open_ports(ip: str) -> list[int]:
    found = []
    for port in PORTS:
        try:
            with socket.create_connection((ip, port), timeout=0.4):
                found.append(port)
        except OSError:
            pass
    return found


def ssh_banner(ip: str) -> str:
    try:
        with socket.create_connection((ip, 22), timeout=2) as s:
            s.settimeout(2)
            return s.recv(256).decode(errors="replace").strip()
    except OSError:
        return ""


def looks_like_gx10(h: dict) -> bool:
    """macOS sshd banners carry no distro suffix; DGX OS (Ubuntu) ones end in Ubuntu-*."""
    if NAME_HINT.search(h["name"]):
        return True
    if 22 in h["ports"]:
        return "ubuntu" in h.get("banner", "").lower()
    return bool(h["ports"])


def arp_table() -> dict[str, str]:
    out = subprocess.run(["arp", "-a"], capture_output=True, text=True, timeout=20).stdout  # noqa: S607 - OS arp
    rows = re.findall(r"(192\.168\.1\.\d+)\s+([0-9a-f-]{17})\s+dynamic", out, re.I)
    return {ip: mac.lower() for ip, mac in rows}


def name_of(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except OSError:
        return ""


def sweep() -> dict[str, dict]:
    ips = [f"{NET}{i}" for i in range(1, 255) if f"{NET}{i}" != SELF]
    with ThreadPoolExecutor(max_workers=64) as pool:
        ports = dict(zip(ips, pool.map(open_ports, ips), strict=True))
    arp = arp_table()
    hosts = {}
    for ip in sorted(set(arp) | {ip for ip, p in ports.items() if p}, key=lambda s: int(s.rsplit(".", 1)[1])):
        hosts[ip] = {"mac": arp.get(ip, "?"), "ports": ports.get(ip, []), "name": name_of(ip)}
    return hosts


def main() -> int:
    max_min = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    base = sweep()
    print(f"{time.strftime('%H:%M:%S')} baseline {len(base)} hosts", flush=True)
    for ip, h in base.items():
        print(f"  {ip} {h['mac']} ports={h['ports']} name={h['name']}", flush=True)
        if NAME_HINT.search(h["name"]):
            print(f"FOUND (already present) {ip} {h}", flush=True)
            return 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_min * 60:
        time.sleep(15)
        now = sweep()
        for ip, h in now.items():
            old = base.get(ip)
            # an ARP entry that briefly drops out reads as mac "?"; that is the same host, not a new one
            is_new = old is None or ("?" not in (old["mac"], h["mac"]) and old["mac"] != h["mac"])
            if ip.endswith(".254"):
                is_new = False
            if is_new and 22 in h["ports"]:
                h["banner"] = ssh_banner(ip)
            if is_new:
                print(f"{time.strftime('%H:%M:%S')} new {ip} {h['mac']} ports={h['ports']} name={h['name']} "
                      f"banner={h.get('banner', '')}", flush=True)
            if is_new and looks_like_gx10(h):
                print(f"FOUND {ip} {h}", flush=True)
                return 0
        base.update({ip: h for ip, h in now.items() if ip not in base})
    print("no new device with ssh/http seen", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
