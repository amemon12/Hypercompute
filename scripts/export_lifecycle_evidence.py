"""Export deployment lifecycle outcomes (including failures) from a running host as evidence JSON.

Failure-path tests (gateway crash + restart, peer loss, leave, pause) end deployments after
`scripts/e2e_check.py --keep` has already saved its file, so their final state lives only in the host's
records. This copies them, unedited, into results/lifecycle-<timestamp>.json.

  uv run python scripts/export_lifecycle_evidence.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
    deployments = c.get("/api/deployments").json()
    nodes = c.get("/api/nodes").json()
    home = Path(os.environ.get("HYPERCOMPUTE_HOME") or Path.home() / ".hypercompute")
    events_file = home / "coordinator" / "records" / "events.jsonl"
    events = []
    if events_file.exists():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev["type"] in ("deployment.state", "node.state", "node.left", "gateway.startup", "node.verified"):
                events.append(ev)
    lifecycle = [{
        "id": d["id"], "model": d["plan"]["model_id"], "strategy": d["plan"]["strategy"],
        "devices": d["plan"]["device_list"], "final_state": d["state"], "error": d.get("error"),
        "cleanup": d.get("cleanup"), "participation_verified": d.get("participation_verified"),
        "history": [(h["state"], h["at"], h.get("detail")) for h in d["history"]],
    } for d in deployments]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"lifecycle-{stamp}.json"
    path.write_text(json.dumps({"exported_at": stamp, "deployments": lifecycle,
                                "nodes": [{k: n[k] for k in ("display_name", "state", "state_reason", "is_local")}
                                          for n in nodes],
                                "events": events}, indent=2), encoding="utf-8")
    failed = [d for d in lifecycle if d["final_state"] == "failed"]
    print(f"wrote {path.relative_to(ROOT)}: {len(lifecycle)} deployments ({len(failed)} failed), {len(events)} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
