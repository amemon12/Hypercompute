"""Pre-run optimizer predictions for the G14 + M4 Air relay pair, and the bench.py cases for layouts a, b, c.

uv run --no-sync python <this file> --nodes-json NODES --out-dir DIR

Read-only: no engine is started, no connection to 52110 is opened, nothing is written to the measurement store.
Inputs are CapacityService._search with this machine and the Mac from a gateway /api/nodes snapshot, the real store
(link records measured over the relay), path class relay for the Mac. Every number written here is a prediction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\Users\operator\.hypercompute\worktrees\iroh-transport")
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts" / "capacity"))

import ranklib as rl  # noqa: E402
from explain_host import NoLeases  # noqa: E402
from hypercompute.capacity import benchlib as bl  # noqa: E402
from hypercompute.capacity import bridge, runtime  # noqa: E402
from hypercompute.capacity.measurements import MeasurementStore  # noqa: E402
from hypercompute.capacity.optimizer import evaluate_layout  # noqa: E402
from hypercompute.capacity.service import CapacityService  # noqa: E402
from hypercompute.capacity.types import GIB, PlacementCandidate, Workload  # noqa: E402
from hypercompute.schemas import NodeView, PlanRequest, PlanStrategy  # noqa: E402
from hypercompute.services.deployments import assign_rpc_names  # noqa: E402
from hypercompute.services.models import ModelService  # noqa: E402

MAC = "4c2991743c154be896d10a0b53ba4d2b"
MODEL = "qwen2.5-0.5b-instruct-q4_k_m"


@dataclass(frozen=True)
class PairNodes:
    local: NodeView
    mac: NodeView

    def eligible_remote(self, node_ids: list[str] | None) -> list[NodeView]:
        return [self.mac]


def nodes(path: Path) -> PairNodes:
    rows = json.loads(path.read_text(encoding="utf-8"))
    local = [NodeView.model_validate(r) for r in rows if r.get("is_local")]
    mac = [NodeView.model_validate(r) for r in rows if r.get("id") == MAC]
    if len(local) != 1 or len(mac) != 1:
        raise SystemExit(f"{path}: need exactly one local node and the Mac {MAC}")
    return PairNodes(local=local[0], mac=mac[0])


def search(svc: CapacityService, w: Workload) -> Any:
    req = PlanRequest(model_id=MODEL, ctx_size=w.ctx_size, strategy=PlanStrategy.fastest_fit)
    return req, svc._search(req, bridge.Request(workload=w))


def stage_json(c: PlacementCandidate, inputs: bridge.BridgeInputs) -> list[dict[str, Any]]:
    out = []
    for s in c.stages:
        key = f"{s.node_id}:{s.device}"
        p = inputs.profile(key)
        weight = sum(inputs.manifest.units[u].weight_bytes for u in s.units)
        row = {"stage": key, "units": [s.units[0], s.units[-1] + 1], "weight_bytes": weight,
               "compute_ms": s.compute_ms, "decode_coefficient": rl_coef(p.decode_ms_per_gib),
               "prefill_coefficient": rl_coef(p.prefill_ms_per_gib_token), "hop_ms": s.hop_ms,
               "upload_bytes": s.upload_bytes, "path": s.path,
               "demand_peak_bytes": s.demand.peak if s.demand else None,
               "admissible_bytes": inputs.admissible(key)}
        if p.link is not None and s.path != "local":
            row["link"] = {"rtt": rl_coef(p.link.rtt_ms), "goodput": rl_coef(p.link.goodput_bytes_per_s),
                           "round_trips_per_token": rl_coef(p.link.round_trips_per_token)}
        out.append(row)
    return out


def rl_coef(c: Any) -> dict[str, Any]:
    return {"value": c.value, "unit": c.unit, "provenance": c.provenance,
            "measured_at": c.measured_at.isoformat() if c.measured_at else None, "samples": c.samples, "note": c.note}


def cand_json(c: PlacementCandidate, inputs: bridge.BridgeInputs) -> dict[str, Any]:
    return {"layout": bridge.span_text(bridge.layout(c), inputs.manifest.n_layer), "feasible": c.feasible,
            "reasons": list(c.reasons), "decode_ms": c.decode_ms, "prefill_ms": c.prefill_ms, "load_ms": c.load_ms,
            "objective_ms": c.objective_ms, "bottleneck": c.bottleneck, "stages": stage_json(c, inputs),
            "skipped": list(c.skipped), "assumptions": list(c.assumptions),
            "coefficients": [{"label": k, **rl_coef(v)} for k, v in bridge.stage_coefficients(c, inputs)]}


def bench_case(plan: Any, model_paths: tuple[str, ...], label: str, threads: int) -> tuple[bl.Case, list[str], list]:
    argv = rl.adapter_argv(plan, Path(model_paths[0]), label)
    args = dict(rl.argv_pairs(argv))
    flags, load_mode, notes = rl._added(args, threads)
    device, split, ngl = str(args["--device"]), str(args.get("--tensor-split") or ""), str(args["-ngl"])
    rpc = tuple(str(args["--rpc"]).split(",")) if args.get("--rpc") else ()
    eng = bl.EngineArgs(devices=() if device == "none" else tuple(device.split(",")),
                        tensor_split=tuple(int(n) for n in split.split(",")) if split else (),
                        ngl=ngl if ngl == "all" else int(ngl), ctx=int(str(args["-c"])),
                        threads=int(str(args.get("-t") or threads)), load_mode=load_mode, flags=tuple(flags))
    case = bl.Case(label=label, model_paths=model_paths, engine=eng, rpc_endpoints=rpc,
                   rpc_path_class="relay" if rpc else "local")
    return case, notes, argv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    pn = nodes(a.nodes_json)
    svc = CapacityService(pn, ModelService(), NoLeases(), MeasurementStore())  # type: ignore[arg-type]
    svc.path_of = lambda nid: "relay" if nid == MAC else "unknown"
    bench_w = Workload(ctx_size=2048, prompt_tokens=512, output_tokens=32, warm=True)
    cold_w = Workload(ctx_size=2048, prompt_tokens=512, output_tokens=32, warm=False)
    service_w = Workload(ctx_size=2048, prompt_tokens=bridge.DEFAULT_PROMPT_TOKENS,
                         output_tokens=bridge.DEFAULT_OUTPUT_TOKENS, warm=False)
    req, found = search(svc, bench_w)
    inputs, result = found.inputs, found.result
    n = inputs.manifest.n_layer
    gpu, mtl = f"{inputs.host.node_id}:CUDA0", f"{MAC}:MTL0"
    layouts = {"a": [(gpu, 0, n + 1)], "b": [(mtl, 0, 2), (gpu, 2, n + 1)], "c": [(mtl, 0, 6), (gpu, 6, n + 1)]}
    model_paths = tuple(str(p) for p in runtime.model_shards(Path(ModelService().primary_path(MODEL))))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "cases").mkdir(exist_ok=True)
    rows = {}
    for name, spans in layouts.items():
        per = {}
        for wname, w in (("bench_warm", bench_w), ("bench_cold", cold_w)):
            c = evaluate_layout(inputs.manifest, w, inputs.host, inputs.peers, spans, slots=inputs.slots)
            per[wname] = cand_json(c, inputs)
        c = evaluate_layout(inputs.manifest, bench_w, inputs.host, inputs.peers, spans, slots=inputs.slots)
        plan = bridge.to_plan(req, c, inputs, reasons=[f"pair layout {name}"], warnings=[], candidates=[], explain=[])
        plan = assign_rpc_names(plan, {r.endpoint: r.endpoint_devices for r in inputs.refs})
        label = f"pair-{MODEL}-{name}"
        case, notes, argv = bench_case(plan, model_paths, label, a.threads)
        cf = a.out_dir / "cases" / f"{name}.json"
        cf.write_text(json.dumps([asdict(case)], indent=1), encoding="utf-8")
        rows[name] = {"spans": [list(s) for s in spans], "prediction": per,
                      "plan": {"device_list": plan.device_list, "tensor_split": plan.tensor_split,
                               "rpc_endpoints": plan.rpc_endpoints, "n_gpu_units": plan.n_gpu_units,
                               "tier": plan.tier, "feasible": plan.feasible},
                      "case": {"file": str(cf), "adapter_argv": argv, "notes": notes,
                               "bench_argv": bl.build_argv(case, Path("llama-server"), 0)[1:]}}
    choices = {}
    for wname, w in (("bench_warm", bench_w), ("bench_cold", cold_w), ("service_default_cold", service_w)):
        _, f = search(svc, w)
        choices[wname] = {"workload": asdict(w), "best": cand_json(f.result.best, f.inputs) if f.result.best else None,
                          "candidates": [cand_json(c, f.inputs) for c in f.result.candidates[:5]],
                          "explain": f.result.explain(), "rejected": list(f.result.rejected)}
    doc = {"schema": "hypercompute.capacity.pair/1", "status": "predicted_only", "created_at": datetime.now(UTC).isoformat(),
           "model": MODEL, "fingerprint": inputs.manifest.fingerprint, "n_layer": n,
           "unit_weight_bytes": [u.weight_bytes for u in inputs.manifest.units],
           "nodes_snapshot": str(a.nodes_json), "mac_node": {"id": MAC, "state": pn.mac.state.value,
                                                             "state_reason": pn.mac.state_reason,
                                                             "heartbeat_age_s": pn.mac.heartbeat_age_s,
                                                             "last_heartbeat_at": pn.mac.last_heartbeat_at.isoformat()
                                                             if pn.mac.last_heartbeat_at else None},
           "measured_coefficients": list(inputs.measured), "notes": list(inputs.notes),
           "warnings": list(inputs.warnings), "pools": bridge.pools_json(inputs.snapshot),
           "layouts": rows, "optimizer_choice": choices,
           "provenance": "every number in this file is a prediction of optimizer formulas; coefficient provenance is "
                         "listed per term"}
    out = a.out_dir / "predictions.json"
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    shutil.copy2(__file__, a.out_dir / "pair_predict.py")
    for name, r in rows.items():
        p = r["prediction"]["bench_warm"]
        print(f"{name}: {p['layout']}: decode {p['decode_ms']:.2f} ms/token, prefill {p['prefill_ms']:.1f} ms, "
              f"load {p['load_ms']:.0f} ms, feasible {p['feasible']}; hops "
              f"{[round(s['hop_ms'], 2) for s in p['stages']]}; compute {[round(s['compute_ms'], 3) for s in p['stages']]}")
        print("   bench argv:", " ".join(r["case"]["bench_argv"]))
    for wname, ch in choices.items():
        b = ch["best"]
        print(f"choice {wname}: {b['layout'] if b else None}: decode {b['decode_ms']:.2f} J {b['objective_ms']:.0f} ms;"
              f" skipped {b['skipped'] if b else None}")
    print(f"measured coefficients: {len(inputs.measured)}; {GIB}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
