"""Exit with one line saying why, as soon as the paper runs need attention; the session wakes on exit and alerts.

Usage: python -I watchdog.py [DONE_LABEL]. Checks every 20 s: gateway health, every peer eligible, new run failures in
the matrix logs, the batch's last run done, the GPU answering, the G14 on AC, free disk. Never prints secrets.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUNS = Path(r"C:\Users\operator\AppData\Local\Temp\claude\C--\9e2c80d5-50ca-460d-8443-8db45359fcde\scratchpad\runs")
LOGS = ["matrix_235b_hs.log", "matrix_235b_hs_redo.log"]
PEER_GRACE_S = 300
GATEWAY_GRACE_S = 180
FAIL = re.compile(r"refused|exit=1 |exit=2 |exit=timeout|Traceback")


def api(path: str) -> object | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8001{path}", timeout=15) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - an unreachable gateway is a finding, not a crash
        return None


def log_lines() -> dict[str, list[str]]:
    out = {}
    for name in LOGS:
        p = RUNS / name
        out[name] = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
    return out


def on_ac() -> bool | None:
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-Command",  # noqa: S607 - Windows' own shell
                            "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | Select-Object -First 1)"
                            ".PowerOnline"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip().lower() == "true" if r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def gpu_ok() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],  # noqa: S607
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def check(seen: dict[str, int], down_since: dict[str, float], gw_down: list[float], done_label: str) -> str | None:
    now = time.monotonic()
    nodes = api("/api/nodes")
    if not isinstance(nodes, list):
        gw_down[0] = gw_down[0] or now
        return f"gateway unreachable for {int(now - gw_down[0])}s" if now - gw_down[0] > GATEWAY_GRACE_S else None
    gw_down[0] = 0.0
    for n in nodes:
        if n.get("is_local"):
            continue
        name = n.get("display_name", "?")
        if n.get("state") == "eligible":
            down_since.pop(name, None)
        else:
            down_since.setdefault(name, now)
            if now - down_since[name] > PEER_GRACE_S:
                return f"{name} {n.get('state')} for {int((now - down_since[name]) / 60)} min ({n.get('state_reason')})"
    for name, lines in log_lines().items():
        for line in lines[seen.get(name, 0):]:
            if FAIL.search(line):
                return f"run problem in {name}: {line.split('+00:00 ', 1)[-1][:120]}"
            if done_label and f"done {done_label} " in line:
                return f"batch finished: {line.split('+00:00 ', 1)[-1][:120]}"
        seen[name] = len(lines)
    if not gpu_ok():
        return "nvidia-smi failed: GPU or driver needs a look"
    if on_ac() is False:
        return "G14 is on battery (charger out?)"
    if shutil.disk_usage("C:\\").free < 10 * 1024**3:
        return "G14 C: under 10 GiB free"
    return None


def main() -> int:
    done_label = sys.argv[1] if len(sys.argv) > 1 else ""
    seen = {name: len(lines) for name, lines in log_lines().items()}
    down_since: dict[str, float] = {}
    gw_down = [0.0]
    while True:
        reason = check(seen, down_since, gw_down, done_label)
        if reason:
            print(time.strftime("%H:%M:%S", time.gmtime()) + " UTC " + reason, flush=True)
            return 0
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
