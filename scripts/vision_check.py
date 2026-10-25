"""Live check of image input on a running Hypercompute gateway: the ready deployment, its vision projector, one image
question and one text chat.

  uv run --no-sync python scripts/vision_check.py --port 8001 --home C:/Users/operator/.hypercompute/iroh-session

It never plans, deploys or stops anything; its only writes are the two chats. Like run_big.py it takes the owner token
from the loopback-only /api/owner/bootstrap and never prints it. One PASS / FAIL / INFO line per check, exit 0 only
when nothing failed:
  catalog     with --preflight MODEL, before deploying, and nothing else: the gateway's catalog lists MODEL's
              projector (it reads HYPERCOMPUTE_CONFIG_DIR at startup) and the model and projector are verified
  deployment  a ready deployment exists: its model and the peers in its plan
  projector   its engine argv has --mmproj, and -mmdev names a local GPU of the plan (or none with no local GPU),
              never an RPC peer
  log         with --home: llama-server's log says "CLIP using <that device> backend" and "loaded multimodal model",
              and the projector's compute buffer is shown next to the planner's prior (PROJECTOR_COMPUTE_EST)
  image       an image question answers 200 with text, with llama-server's timings and the wall time
  text        a text-only chat on the same deployment still answers 200
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import httpx
from hypercompute import config
from hypercompute.image_cli import request_body, summary_lines
from hypercompute.multimodal import ImageInputError, data_url
from hypercompute.services.deployments import PROJECTOR_COMPUTE_EST
from hypercompute.units import MiB

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "docs" / "screenshots" / "03-choose-model.png"
QUESTION = "What does this screenshot show? Answer in two sentences."
_BACKEND = re.compile(r"CLIP using (\S+) backend")
_BUFFER = re.compile(r"reserve_compute_meta:\s+(\S+) compute buffer size =\s+([\d.]+) MiB")


def log_facts(text: str) -> dict[str, Any]:
    """The projector's device, whether it loaded, and its last compute buffer per backend, from llama-server's log."""
    backends = _BACKEND.findall(text)
    return {"backend": backends[-1] if backends else None, "loaded": "loaded multimodal model" in text,
            "buffers": {name: float(mib) for name, mib in _BUFFER.findall(text)}}


class Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, status: str, label: str, detail: str) -> None:
        self.failed = self.failed or status == "FAIL"
        print(f"{status} {label:<12}{detail}", flush=True)


def check_projector(dep: dict[str, Any], report: Report) -> str | None:
    """-mmdev's value when the argv loads a projector on a local device (or on the CPU with no local GPU)."""
    args: list[str] = dep.get("engine_args") or []
    assignments = dep["plan"].get("assignments") or []
    local = {a["device"] for a in assignments if a.get("endpoint") is None and a["device"] != "CPU"}
    peers = [a["device"] for a in assignments if a.get("endpoint")]
    if "--mmproj" not in args:
        report.line("FAIL", "projector", "no --mmproj in the engine argv: the projector is not downloaded, or the "
                    "plan was made before it was (plan again)")
        return None
    mmdev = args[args.index("-mmdev") + 1] if "-mmdev" in args[:-1] else None
    if mmdev in local or (mmdev == "none" and not local):
        where = "local" if mmdev in local else "host CPU, no local GPU in the plan"
        report.line("PASS", "projector", f"-mmdev {mmdev} ({where}), peers {', '.join(peers) or 'none'}")
        return mmdev
    report.line("FAIL", "projector", f"-mmdev {mmdev} is not a local device of the plan ({', '.join(sorted(local))})")
    return None


def check_log(home: Path, dep_id: str, mmdev: str | None, report: Report) -> None:
    path = home / "coordinator" / "logs" / f"llama-server-{dep_id}.log"
    try:
        facts = log_facts(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        report.line("FAIL", "log", f"cannot read {path}: {exc}")
        return
    want = "CPU" if mmdev == "none" else mmdev
    if not facts["loaded"] or facts["backend"] is None or facts["backend"] != want:
        report.line("FAIL", "log", f"projector backend {facts['backend']}, wanted {want}; loaded multimodal model: "
                    f"{facts['loaded']} ({path})")
        return
    used = facts["buffers"].get(facts["backend"])
    measured = f"compute {used:.0f} MiB on {facts['backend']}" if used is not None else "no compute buffer line"
    report.line("PASS", "log", f"CLIP using {facts['backend']} backend; loaded multimodal model; {measured} against "
                f"the {PROJECTOR_COMPUTE_EST // MiB} MiB prior")


def chat(client: httpx.Client, base: str, token: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    r = client.post(f"{base}/v1/chat/completions", json=body, headers={config.OWNER_HEADER: token})
    wall = time.perf_counter() - started
    try:
        reply = r.json()
    except ValueError:
        reply = {"detail": r.text[:300]}
    return r.status_code, reply if isinstance(reply, dict) else {"detail": str(reply)[:300]}, wall


def check_chats(client: httpx.Client, base: str, token: str, url: str, opts: argparse.Namespace,
                report: Report) -> None:
    status, reply, wall = chat(client, base, token, request_body(url, opts.question, opts.max_tokens))
    if status == 200:
        answer, _, *timings = summary_lines(reply, wall)
        report.line("PASS", "image", answer.replace("\n", " ")[:300])
        for t in timings:
            print(f"{'':17}{t}", flush=True)
    else:
        report.line("FAIL", "image", f"HTTP {status}: {reply.get('detail') or reply}")
    status, reply, _ = chat(client, base, token, {"messages": [{"role": "user", "content": "Reply with: ready"}],
                                                 "max_tokens": 8, "temperature": 0})
    text = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") if status == 200 else None
    if text:
        report.line("PASS", "text", text.strip()[:120])
    else:
        report.line("FAIL", "text", f"HTTP {status}: {reply.get('detail') or reply}")


def check_catalog(views: list[dict[str, Any]], model_id: str, report: Report) -> None:
    """Before deploying: the gateway's catalog lists the model's projector and both files are verified here."""
    view = next((v for v in views if v["artifact"]["id"] == model_id), None)
    restart = ("restart the gateway from the merged checkout with HYPERCOMPUTE_CONFIG_DIR unset or set to that "
               "checkout's configs")
    if view is None or not view["artifact"].get("mmproj") or "projector_status" not in view:
        why = "is not in" if view is None else "has no projector in"
        report.line("FAIL", "catalog", f"{model_id} {why} the gateway's catalog or code: {restart}")
    elif view["status"] != "downloaded":
        report.line("FAIL", "catalog", f"{model_id} is {view['status']}: download the model first")
    elif view["projector_status"] != "downloaded":
        report.line("FAIL", "catalog", f"projector {view['projector_status']}: uv run --no-sync hypercompute models "
                    f"pull {model_id} (with the gateway's HYPERCOMPUTE_HOME)")
    else:
        report.line("PASS", "catalog", f"{model_id} downloaded; projector {view['artifact']['mmproj']['name']} "
                    "verified; plan after this so the plan loads it")


def run(opts: argparse.Namespace) -> int:
    report = Report()
    try:
        url = data_url(opts.image.read_bytes())
    except (OSError, ImageInputError) as exc:
        report.line("FAIL", "image", f"{opts.image}: {exc}")
        return 1
    base = f"http://127.0.0.1:{opts.port}"
    with httpx.Client(trust_env=False, timeout=httpx.Timeout(10.0, read=opts.timeout)) as client:
        try:
            token = client.get(f"{base}/api/owner/bootstrap").json()["owner_token"]
            if opts.preflight:
                check_catalog(client.get(f"{base}/api/models").json(), opts.preflight, report)
                return 1 if report.failed else 0
            deps = client.get(f"{base}/api/deployments").json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            report.line("FAIL", "gateway", f"{base} did not answer: {exc}")
            return 1
        dep = next((d for d in deps if d.get("state") == "ready"), None)
        if dep is None:
            report.line("FAIL", "deployment", "no ready deployment: deploy the vision model first")
            return 1
        peers = sorted({a["endpoint"] for a in dep["plan"].get("assignments") or [] if a.get("endpoint")})
        report.line("PASS", "deployment", f"{dep['plan']['model_id']} on this machine + {len(peers)} peer(s) "
                    f"{' '.join(peers)}".rstrip())
        mmdev = check_projector(dep, report)
        if opts.home is not None:
            check_log(opts.home, dep["id"], mmdev, report)
        else:
            report.line("INFO", "log", "pass --home <the gateway's HYPERCOMPUTE_HOME> to check llama-server's log")
        check_chats(client, base, token, url, opts, report)
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8001, help="gateway port on 127.0.0.1 (default 8001, the Iroh host)")
    p.add_argument("--home", type=Path, default=None, help="the gateway's HYPERCOMPUTE_HOME, to read its engine log")
    p.add_argument("--preflight", metavar="MODEL", default=None,
                   help="before deploying: only check that the catalog lists MODEL's projector and it is verified")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="PNG, JPEG or GIF (default: the model-selection dashboard shot)")
    p.add_argument("--question", default=QUESTION)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--timeout", type=float, default=1800.0, help="read timeout per chat, seconds")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
