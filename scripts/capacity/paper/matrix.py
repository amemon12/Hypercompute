"""Run a list of run_big configs in order and index every result for the paper.

Usage: python -I matrix.py PLAN.json INDEX.jsonl
PLAN is a list of {"label", "model", "nodes": [...], "args": [...], "timeout_s"}.
Per run it snapshots the gateway's /api/network (per-peer bridge counters, path, RTT) and each Mac's memory,
swap, battery and thermal state before and after, then appends one JSON line to INDEX. Never prints secrets.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Defaults are the G14 host; a different host (e.g. the GX10) sets HC_PAPER_REPO / HC_PAPER_HOME / HC_PAPER_UV, and a
# unified-memory host also HYPERCOMPUTE_LLAMA_BIN_DIR so cooldown() can wait for the engine process.
LIVE = Path(os.environ.get("HC_PAPER_REPO", r"C:\Users\operator\.hypercompute\worktrees\live-host"))
HOME = os.environ.get("HC_PAPER_HOME", r"C:\Users\operator\.hypercompute\iroh-session")
UV = os.environ.get("HC_PAPER_UV", r"C:\Users\operator\.local\bin\uv.exe")
KEY = str(Path.home() / ".ssh" / "hypercompute_ed25519")
PROBE_MACS = os.environ.get("HC_PAPER_PROBE_MACS", "1") != "0"  # 0 on a host without SSH keys for the Macs
MACS = {"4c2991743c154be896d10a0b53ba4d2b": ("mac-azim", "azimmemon@100.101.19.13"),
        "6dad6cf75b2e4cf197b7bc7f4d0a67e4": ("mac-a", "operator@100.80.165.87"),
        "d4e65f83c4be4a41aad722e40983b08b": ("mac-b", "azim@100.90.34.93"),
        "8f169e3549fc41fbb14f6876a19d704c": ("g14", None)}
MAC_PROBE = ("sysctl -n vm.swapusage; vm_stat | head -9 | tail -8; pmset -g batt | tail -1; "
             "pmset -g therm 2>/dev/null | tail -2; memory_pressure | tail -1")
SECRETISH = re.compile(r"[A-Za-z0-9_-]{40,}")


def now() -> str:
    return datetime.now(UTC).isoformat()


def get(path: str) -> object:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:8001{path}", timeout=10) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001 - a missing snapshot is recorded, not fatal
        return {"error": repr(exc)}


def network() -> object:
    net = get("/api/network")
    if isinstance(net, dict):
        net.pop("endpoint_id", None)
        for p in net.get("peers", []):
            p.pop("peer_id", None)
            if isinstance(p.get("path"), dict):
                p["path"].pop("remote_id", None)
    return net


def mac_state(node_ids: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for nid in node_ids:
        name, host = MACS.get(nid, (nid[:8], None))
        if host is None or not PROBE_MACS:
            out[name] = "probe skipped"
            continue
        try:
            cmd = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, MAC_PROBE]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)  # noqa: S607 - the operator's ssh
            out[name] = SECRETISH.sub("<r>", r.stdout.strip()) or SECRETISH.sub("<r>", r.stderr.strip())
        except Exception as exc:  # noqa: BLE001
            out[name] = f"probe failed: {exc!r}"
    return out


def newest_result(before: set[str]) -> str | None:
    root = LIVE / "results" / "capacity"
    fresh = sorted(p.name for p in root.iterdir() if p.is_dir() and "-big-" in p.name and p.name not in before)
    return fresh[-1] if fresh else None


def gpu_used_mib() -> int | None:
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)  # noqa: S607 - the driver's own tool
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - no reading means no wait
        return None


def own_engine_running() -> bool:
    """Unified-memory hosts (GB10) report no GPU memory.used: look for a llama-server from our own bin dir instead.

    Needs HYPERCOMPUTE_LLAMA_BIN_DIR in this process's environment too (the gateway, not this driver, starts the
    engine). Matching the full path skips unrelated llama-servers, such as a vendor service under /snap.
    """
    bin_dir = os.environ.get("HYPERCOMPUTE_LLAMA_BIN_DIR")
    if not bin_dir or sys.platform == "win32":
        return False
    try:
        r = subprocess.run(["pgrep", "-f", str(Path(bin_dir) / "llama-server")],  # noqa: S607 - procps pgrep
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False  # no reading means no wait, as in gpu_used_mib()
    return r.returncode == 0


def cooldown_mode() -> str:
    """Which drain check cooldown() can use on this host: the GPU's memory.used, our engine's process, or neither."""
    if gpu_used_mib() is not None:
        return "gpu"
    if os.environ.get("HYPERCOMPUTE_LLAMA_BIN_DIR") and sys.platform != "win32":
        return "process"
    return "none"


def cooldown(max_s: float = 180.0) -> float:
    """Wait for the previous engine to release the GPU, then let heartbeats report the freed memory."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_s:
        used = gpu_used_mib()
        if (used is None or used < 1536) and not own_engine_running():
            break
        time.sleep(5)
    time.sleep(20)  # heartbeats are every 3 s; the planner reads the last reported free memory
    return round(time.monotonic() - t0, 1)


def wait_eligible(node_ids: list[str], max_s: float = 1200.0) -> float:
    """Wait until every node of the run is eligible (a flaky network drops peers for minutes at a time)."""
    t0 = time.monotonic()
    while node_ids and time.monotonic() - t0 < max_s:
        nodes = get("/api/nodes")
        states = {n.get("id"): n.get("state") for n in nodes} if isinstance(nodes, list) else {}
        if all(states.get(nid) == "eligible" for nid in node_ids):
            break
        time.sleep(10)
    return round(time.monotonic() - t0, 1)


def run_one(cfg: dict, attempt: int = 1) -> dict:
    root = LIVE / "results" / "capacity"
    before = {p.name for p in root.iterdir()}
    cmd = [UV, "run", "--no-sync", "python", "scripts/capacity/run_big.py", "--model", cfg["model"],
           "--min-available-gib", "0", "--home", HOME]
    for nid in cfg.get("nodes", []):
        cmd += ["--node", nid]
    if not cfg.get("nodes"):
        cmd += ["--strategy", "local_only"]
    cmd += cfg.get("args", [])
    rec: dict = {"label": cfg["label"], "model": cfg["model"],
                 "nodes": [MACS.get(n, (n[:8], None))[0] for n in cfg.get("nodes", [])],
                 "args": cfg.get("args", []), "started_at": now(), "network_before": network(),
                 "macs_before": mac_state(cfg.get("nodes", []))}
    t0 = time.monotonic()
    env = {**os.environ, "HYPERCOMPUTE_HOME": HOME}
    try:
        r = subprocess.run(cmd, cwd=LIVE, capture_output=True, text=True, timeout=cfg.get("timeout_s", 5400), env=env)
        rec["exit"] = r.returncode
        rec["tail"] = SECRETISH.sub("<r>", (r.stdout + r.stderr)[-1500:])
    except subprocess.TimeoutExpired as exc:
        rec["exit"] = "timeout"
        rec["tail"] = SECRETISH.sub("<r>", str(exc)[-800:])
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    rec["result_dir"] = newest_result(before)
    rec["network_after"] = network()
    rec["macs_after"] = mac_state(cfg.get("nodes", []))
    rec["finished_at"] = now()
    return rec


def run_status(rec: dict) -> str | None:
    path = LIVE / "results" / "capacity" / str(rec.get("result_dir")) / "run.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


def main() -> int:
    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    index = Path(sys.argv[2])
    mode = cooldown_mode()
    print(f"{now()} cooldown check: {mode}", flush=True)
    if mode == "none" and sys.platform != "win32":
        print("WARNING: no GPU memory reading and HYPERCOMPUTE_LLAMA_BIN_DIR unset: cooldown is a fixed 20 s",
              file=sys.stderr, flush=True)
    for cfg in plan:
        waited = cooldown()
        waited_nodes = wait_eligible(cfg.get("nodes", []))
        print(f"{now()} start {cfg['label']} (cooldown {waited}s, waited {waited_nodes}s for peers)", flush=True)
        rec = {**run_one(cfg), "cooldown_mode": mode, "cooldown_s": waited}
        status = run_status(rec)
        if status in ("infeasible", "refused"):
            print(f"{now()} {cfg['label']} {status}; retrying once after 60 s", flush=True)
            with index.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "label": cfg["label"] + "-refused"}) + "\n")
            time.sleep(60)
            cooldown()
            wait_eligible(cfg.get("nodes", []))
            rec = {**run_one(cfg, attempt=2), "cooldown_mode": mode, "cooldown_s": waited}
        with index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{now()} done {cfg['label']} exit={rec['exit']} wall={rec['wall_s']}s dir={rec['result_dir']}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
