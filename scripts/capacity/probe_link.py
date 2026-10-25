"""Measure one RPC link (round trip and bulk upload rate) over the llama.cpp RPC protocol, without llama-server.

  uv run python scripts/capacity/probe_link.py --endpoint 127.0.0.1:52110 --path-class relay \
      --bytes 8388608 --samples 20

The endpoint must be a ggml rpc-server (protocol 8, release b11770) or a bridge in front of one. The server serves one
client at a time, so do not probe an endpoint a running llama-server is connected to. The probe allocates one scratch
buffer of --bytes on device --device (at most 64 MiB and a quarter of the reported free memory), uploads random bytes
with cache_flag 0 (nothing reaches the peer's RPC cache directory) and frees the buffer before it exits.

Output: results/capacity/<UTC stamp>-link-<label>/link.json, a short table on stdout, and with --store the measured
coefficients in the measurement store (--node-id is then required). The evidence file is written before the store is
touched. Exit status: 0 measured, 2 probe failed, 3 measured but not stored.

Stored link records are keyed by the peer's NodeView id (32 lowercase hex characters, the id a contributor joins with)
and the path class; capacity/bridge.py prices a peer's devices with the freshest such record of the same path class
that is at most an hour old.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from hypercompute.capacity import linkprobe  # noqa: E402
from hypercompute.capacity.measurements import (  # noqa: E402
    MeasurementIdentity,
    MeasurementStore,
    from_coefficient,
)
from hypercompute.capacity.types import Coefficient  # noqa: E402

MIB = 1 << 20
NODE_ID_RE = re.compile(r"[a-f0-9]{32}")  # schemas.JoinRequest.node_id: the peer's NodeView id (fullmatch)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True, help="host:port of the rpc-server or its local bridge")
    p.add_argument("--path-class", required=True, choices=linkprobe.PROBE_PATH_CLASSES,
                   help="how the bytes travel; the caller knows this, the probe does not guess it")
    p.add_argument("--bytes", type=int, default=8 * MIB, help="goodput upload size; 0 skips the goodput probe")
    p.add_argument("--chunk-bytes", type=int, default=linkprobe.DEFAULT_CHUNK_BYTES)
    p.add_argument("--samples", type=int, default=20, help="counted RTT round trips")
    p.add_argument("--device", type=int, default=0, help="device index on the peer")
    p.add_argument("--timeout", type=float, default=60.0, help="seconds allowed for each socket operation")
    p.add_argument("--label", help="evidence directory label (default: <path class>-<host>-<port>)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "capacity")
    p.add_argument("--store", action="store_true", help="also write the measured coefficients to the store")
    p.add_argument("--store-dir", type=Path, help="measurement store directory (default: runtime home / capacity)")
    p.add_argument("--node-id", help="node id of the peer, required with --store")
    p.add_argument("--device-name", help="identity device name (default: rpc<device>)")
    p.add_argument("--backend-build", help="identity backend build (default: ggml-rpc-<version from HELLO>)")
    p.add_argument("--power-state", help="identity power state, for example ac or battery")
    args = p.parse_args(argv)
    if args.store and not args.node_id:
        p.error("--store needs --node-id: the probe cannot tell which node is behind the endpoint")
    if args.node_id is not None and not NODE_ID_RE.fullmatch(args.node_id):
        p.error(f"--node-id {args.node_id!r} is not a NodeView id (32 lowercase hex characters, the id "
                "GET /api/nodes lists); link records under any other key never reach the planner")
    if args.store:
        try:  # refuse bad identity fields before the peer is probed, not after
            MeasurementIdentity(kind="link.rtt_ms", **_identity_fields(args, "0.0.0"))
        except ValueError as exc:
            p.error(f"--store identity: {exc}")
    return args


def _identity_fields(args: argparse.Namespace, version: str) -> dict[str, Any]:
    return {"node_id": args.node_id, "device": args.device_name or f"rpc{args.device}",
            "backend_build": args.backend_build or f"ggml-rpc-{version}", "power_state": args.power_state,
            "path_class": args.path_class}


def default_label(endpoint: str, path_class: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{path_class}-{endpoint}").strip("-")


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _coef_json(c: Coefficient) -> dict[str, Any]:
    return _jsonable(asdict(c))


def evidence(args: argparse.Namespace, result: linkprobe.LinkMeasurement | None, error: str | None) -> dict:
    doc: dict[str, Any] = {
        "kind": "link_probe", "tool": "scripts/capacity/probe_link.py",
        "protocol": f"llama.cpp RPC {linkprobe.PROTO_MAJOR}.x ({linkprobe.PINNED_RELEASE})",
        "endpoint": args.endpoint, "path_class": args.path_class, "node_id": args.node_id,
        "requested": {"bytes": args.bytes, "chunk_bytes": args.chunk_bytes, "samples": args.samples,
                      "device": args.device, "timeout_s": args.timeout},
        "payload": "os.urandom, fresh bytes per chunk, generated before the clock starts; SET_TENSOR cache_flag 0",
        "ok": error is None, "error": error,
    }
    if result is not None:
        doc["rtt"] = _jsonable(asdict(result.rtt))
        doc["goodput"] = _jsonable(asdict(result.goodput)) if result.goodput is not None else None
        profile = result.to_link_profile()
        doc["link_profile"] = {"path": profile.path, "rtt_ms": _coef_json(profile.rtt_ms),
                               "goodput_bytes_per_s": _coef_json(profile.goodput_bytes_per_s),
                               "round_trips_per_token": _coef_json(profile.round_trips_per_token)}
    return doc


def store_results(args: argparse.Namespace, result: linkprobe.LinkMeasurement, evidence_path: Path) -> list[str]:
    common = _identity_fields(args, ".".join(map(str, result.rtt.server_version)))
    coefs = {"link.rtt_ms": result.rtt.to_coefficient()}
    if result.goodput is not None:
        coefs["link.goodput_bytes_per_s"] = result.goodput.to_coefficient()
    store = MeasurementStore(args.store_dir) if args.store_dir else MeasurementStore()
    for kind, coef in coefs.items():
        store.put(from_coefficient(MeasurementIdentity(kind=kind, **common), coef, evidence=str(evidence_path)))
    return [f"{kind} -> {store.path}" for kind in coefs]


def print_table(doc: dict) -> None:
    print(f"endpoint {doc['endpoint']}  path {doc['path_class']}  ok {doc['ok']}")
    if doc.get("error"):
        print(f"error    {doc['error']}")
    if doc.get("store_error"):
        print(f"store    not stored: {doc['store_error']}")
    rtt = doc.get("rtt")
    if rtt:
        print(f"server   RPC {'.'.join(map(str, rtt['server_version']))}  device {rtt['device']}  "
              f"free {rtt['free_bytes'] / MIB:.0f} MiB of {rtt['total_bytes'] / MIB:.0f} MiB")
        print(f"rtt      median {rtt['median_ms']:.3f} ms  p90 {rtt['p90_ms']:.3f} ms  "
              f"n {len(rtt['samples_ms'])}  (measured)")
    gp = doc.get("goodput")
    if gp:
        print(f"goodput  {gp['bytes_per_s'] / MIB:.3f} MiB/s  {gp['total_bytes'] / MIB:.1f} MiB in "
              f"{gp['elapsed_s']:.3f} s minus {gp['rtt_subtracted_ms']:.3f} ms  "
              f"freed {'confirmed' if gp['free_confirmed'] else 'on disconnect'}  (measured)")
    rtpt = doc.get("link_profile", {}).get("round_trips_per_token")
    if rtpt:
        print(f"rt/token {rtpt['value']}  ({rtpt['provenance']})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.results_dir / f"{stamp}-link-{args.label or default_label(args.endpoint, args.path_class)}"
    result: linkprobe.LinkMeasurement | None = None
    error: str | None = None
    try:
        result = linkprobe.probe_link(args.endpoint, path_class=args.path_class, total_bytes=args.bytes,
                                      chunk_bytes=args.chunk_bytes, samples=args.samples, device=args.device,
                                      timeout_s=args.timeout)
    except (linkprobe.LinkProbeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    doc = evidence(args, result, error)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "link.json"
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")  # evidence first: a store failure must not lose it
    code = 0 if error is None else 2
    if result is not None and args.store:
        try:
            doc["stored"] = store_results(args, result, path)
        except (OSError, ValueError) as exc:
            doc["store_error"] = f"{type(exc).__name__}: {exc}"
            code = 3
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print_table(doc)
    print(f"evidence {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
