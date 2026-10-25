"""Planner gate: predicted versus measured decode ranking of local layouts for one model on this host.

  uv run --no-sync python scripts/capacity/rank_layouts.py --model qwen2.5-7b-instruct-q4_k_m \
      --layout a=CUDA0:0-29 --layout b=CPU:0-8,CUDA0:8-29 --layout c=CPU:0-16,CUDA0:16-29 \
      --layout "d=CPU:0-29;stream=8-28;slots=2"

A layout is a chain of half-open unit spans DEVICE:start-end over blocks 0..n_layer-1 and the head (unit n_layer): the
host CPU first, CUDA0 last with the head. A CPU span is resident unless ";stream=a-b[,c-d];slots=K" names the units
(half-open ranges inside the span) it streams from the GGUF through K hcstream slots. The optimizer's own best layout
is added when it differs from every listed one (--no-optimizer-best leaves it out).

Per layout, in this order:
  inputs  CapacityService's own search for this host alone: the live inventory (inventory.collect runs llama-server
          --version and --list-devices) or the local node of a gateway /api/nodes snapshot (--nodes-json), the default
          contribution, the measurement store, no peers, no leases, for the bench workload (ctx, prompt and output
          tokens, warm). A CPU-only layout is scored on the inputs the bridge builds when no accelerator is contributed.
  score   optimizer.evaluate_layout, then service.running_split with the layout's residency: evaluate_layout takes
          costs.split_cpu's residency for today's budget, running_split rescores decode for the residency that runs (as
          the replan evaluator does for a running plan).
  plan    bridge.to_plan, deployments.assign_rpc_names and the llama adapter's build_args; that argv becomes the bench
          case, with every difference recorded. A stream layout's hcstream config is also rendered by
          runtime.prepare_stream and compared with the one bench.py writes.
  run     scripts/capacity/bench.py --case-file as an isolated child, one layout at a time, each after a check that
          Windows reports the AC adapter online (root/wmi BatteryStatus.PowerOnline). The campaign stops at the first
          check that does not say True. --dry-run runs bench.py --dry-run (preflight only, no engine). A layout whose
          GPU stage the optimizer says overflows is not run (--run-infeasible runs it).

results/capacity/<UTC stamp>-ranking-<model>/ranking.json holds per layout the spans, the predicted decode with each
term and the coefficient behind it, the measured decode median and spread with its bench.json, the measured load,
prefill, SSD bytes per token and peak private memory, and the absolute percent error; then the predicted and measured
ranks, Kendall tau-b and Spearman rho over the layouts measured with at least 3 reps. Nothing is written to the
measurement store: coefficients are never refit to these results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

import ranklib as rl  # noqa: E402
from explain_host import ACCELERATORS, HostOnly, NoLeases, host_node  # noqa: E402
from hypercompute import engine  # noqa: E402
from hypercompute.agent import inventory  # noqa: E402
from hypercompute.agent.contribution import default_contribution  # noqa: E402
from hypercompute.capacity import benchlib as bl  # noqa: E402
from hypercompute.capacity import bridge, costs, runtime  # noqa: E402
from hypercompute.capacity.manifest import ManifestError  # noqa: E402
from hypercompute.capacity.measurements import MeasurementStore  # noqa: E402
from hypercompute.capacity.optimizer import OptimizeResult, evaluate_layout  # noqa: E402
from hypercompute.capacity.service import CapacityService, CapacityUnavailable, running_split  # noqa: E402
from hypercompute.capacity.types import GIB, Coefficient, PlacementCandidate, PoolDemand, Workload  # noqa: E402
from hypercompute.schemas import NodeView, Plan, PlanRequest, PlanStrategy  # noqa: E402
from hypercompute.services.deployments import assign_rpc_names  # noqa: E402
from hypercompute.services.models import ModelError, ModelService  # noqa: E402

SCHEMA = "hypercompute.capacity.ranking/1"
RESULTS = ROOT / "results" / "capacity"
BENCH = HERE / "bench.py"
BENCH_TIMEOUT_S = 3 * 3600.0
LLAMA_PROCESSES = ("llama-server", "llama-completion")


# --------------------------------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Host:
    local: NodeView
    models: Any  # ModelService (view, primary_path)
    store: MeasurementStore
    probes: bridge.Probes | None = None  # tests replace the OS and file reads; None reads this machine


def local_node(nodes_json: Path | None) -> NodeView:
    """This machine as the gateway builds it: from a fresh inventory, or the local row of an /api/nodes snapshot."""
    if nodes_json is None:
        inv = inventory.collect()
        return host_node(inv, default_contribution(inv))
    rows = json.loads(nodes_json.read_text(encoding="utf-8"))
    rows = rows.get("nodes", []) if isinstance(rows, dict) else rows
    local = [r for r in rows if isinstance(r, dict) and r.get("is_local")] if isinstance(rows, list) else []
    if len(local) != 1:
        raise rl.RankError(f"{nodes_json} holds {len(local)} local nodes, not one")
    return NodeView.model_validate(local[0])


def without_accelerators(node: NodeView) -> NodeView:
    """The node contributing no accelerator: the bridge then builds a CPU-only host."""
    kept = [d for d in node.contribution.devices if not d.startswith(ACCELERATORS)]
    return node.model_copy(update={"contribution": node.contribution.model_copy(update={"devices": kept})})


@dataclass(frozen=True)
class Search:
    request: PlanRequest
    inputs: bridge.BridgeInputs
    result: OptimizeResult


def search(host: Host, node: NodeView, model: str, workload: Workload) -> Search:
    """CapacityService._search for this host alone, the code path explain and plan use."""
    svc = CapacityService(HostOnly(node), host.models, NoLeases(), host.store, probes=host.probes)  # type: ignore[arg-type]
    req = PlanRequest(model_id=model, ctx_size=workload.ctx_size, strategy=PlanStrategy.local_only)
    found = svc._search(req, bridge.Request(workload=workload))
    return Search(request=req, inputs=found.inputs, result=found.result)


# --------------------------------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Scored:
    spec: rl.LayoutSpec
    inputs: bridge.BridgeInputs
    split_cpu: PlacementCandidate  # evaluate_layout as it scores the spans (split_cpu residency)
    candidate: PlacementCandidate  # with the layout's own residency
    plan: Plan  # named for llama.cpp, what the adapter runs


def with_residency(c: PlacementCandidate, spec: rl.LayoutSpec) -> PlacementCandidate:
    """`c` with its host CPU stage set to the spec's resident and streamed units (times unchanged)."""
    first = c.stages[0]
    if first.device != bridge.HOST_CPU:
        return c
    streamed = tuple(sorted(spec.streamed))
    stage = replace(first, resident_units=tuple(u for u in first.units if u not in set(streamed)),
                    streamed_units=streamed, slots=spec.slots if streamed else 0)
    return replace(c, stages=(stage, *c.stages[1:]))


def score(spec: rl.LayoutSpec, found: Search) -> Scored:
    inputs, req = found.inputs, found.request
    node = inputs.host.node_id
    spans = [(f"{node}:{dev}", a, b) for dev, a, b in spec.spans]
    base = evaluate_layout(inputs.manifest, inputs.workload, inputs.host, inputs.peers, spans,
                           slots=spec.slots or inputs.slots)
    shaped = bridge.to_plan(req, with_residency(base, spec), inputs, reasons=[], warnings=[], candidates=[],
                            explain=[])
    cand = running_split(base, shaped, inputs)
    lines = bridge.candidate_lines(cand, inputs.manifest.n_layer, f"layout {spec.name}")
    plan = bridge.to_plan(req, cand, inputs, reasons=[f"planner gate layout {spec.name}: {spec.text()}"],
                          warnings=list(inputs.warnings), candidates=[], explain=lines)
    if plan.stream != shaped.stream:
        raise rl.RankError(f"layout {spec.name}: the rescored plan streams other units than the layout")
    return Scored(spec=spec, inputs=inputs, split_cpu=base, candidate=cand, plan=assign_rpc_names(plan, {}))


def forced_cpu_demand(spec: rl.LayoutSpec, inputs: bridge.BridgeInputs) -> dict[str, Any] | None:
    """The host CPU stage's PoolDemand for the spec's residency, the terms optimizer._build charges, against the CPU
    pool's admissible bytes. running_split keeps split_cpu's demand, so the layout's own fit is computed here."""
    if spec.spans[0][0] != bridge.HOST_CPU:
        return None
    m, end = inputs.manifest, spec.spans[0][2]
    units, streamed = m.units[:end], set(spec.streamed)
    weights = (sum(u.buffer_bytes for u in units if u.index not in streamed) if streamed
               else sum(u.weight_bytes for u in units))
    slot = max((u.buffer_bytes for u in units if u.index in streamed), default=0)
    d = PoolDemand(node_id=inputs.host.node_id, pool_id=inputs.host.cpu.pool_id, weights=weights,
                   kv=m.kv_bytes_per_layer(inputs.workload.ctx_size) * sum(1 for u in units if u.kind == "block"),
                   workspace=costs.workspace_bytes(m.n_embd), staging=spec.slots * slot if streamed else 0,
                   overhead=costs.overhead_bytes(inputs.host.cpu.backend))
    admissible = inputs.host.cpu_pool_admissible
    return {"weights": d.weights, "kv": d.kv, "workspace": d.workspace, "staging": d.staging,
            "overhead": d.overhead, "peak_bytes": d.peak, "admissible_bytes": admissible,
            "fits": d.peak <= admissible, "provenance": "estimate (optimizer demand formula)"}


def _coef(label: str, c: Coefficient) -> dict[str, Any]:
    return {"label": label, "value": c.value, "unit": c.unit, "provenance": c.provenance,
            "measured_at": c.measured_at.isoformat() if c.measured_at else None, "samples": c.samples,
            "note": c.note}


def stage_terms(s: Scored) -> list[dict[str, Any]]:
    """Every term of the predicted decode per stage, with the coefficient each one multiplies."""
    out = []
    units, storage = s.inputs.manifest.units, s.inputs.storage
    for st in s.candidate.stages:
        key = f"{st.node_id}:{st.device}"
        p = s.inputs.profile(key)
        terms = [{"term": "compute", "ms": st.compute_ms, "weight_gib": sum(units[u].weight_bytes for u in st.units)
                  / GIB, "coefficients": [_coef(f"{key} decode", p.decode_ms_per_gib)]}]
        if st.streamed_units:
            terms.append({"term": "exposed I/O", "ms": st.exposed_io_ms, "slots": st.slots,
                          "stream_bytes": sum(units[u].buffer_bytes for u in st.streamed_units),
                          "coefficients": [_coef("storage read", storage.read_bytes_per_s),
                                           _coef("stream efficiency", storage.stream_efficiency)]})
        if st.path != "local" and p.link is not None:
            terms.append({"term": "hop", "ms": st.hop_ms, "coefficients": [_coef(f"{key} rtt", p.link.rtt_ms)]})
        out.append({"stage": key, "units": list(st.units), "resident_units": list(st.resident_units),
                    "streamed_units": list(st.streamed_units),
                    "stage_ms": st.compute_ms + st.exposed_io_ms + st.hop_ms, "terms": terms})
    return out


def prediction_json(s: Scored) -> dict[str, Any]:
    c, base = s.candidate, s.split_cpu
    first = base.stages[0]
    return {"provenance": "prediction (optimizer formulas on the coefficients listed per term)",
            "decode_ms": c.decode_ms, "prefill_ms": c.prefill_ms, "load_ms": c.load_ms,
            "stream_bytes_per_token": c.stream_bytes_per_token, "bottleneck": c.bottleneck, "stages": stage_terms(s),
            "coefficients": [_coef(label, k) for label, k in bridge.stage_coefficients(c, s.inputs)],
            "optimizer_feasible": base.feasible, "optimizer_reasons": list(base.reasons),
            "split_cpu_streamed_units": list(first.streamed_units) if first.device == bridge.HOST_CPU else [],
            "layout_cpu_demand": forced_cpu_demand(s.spec, s.inputs), "assumptions": list(c.assumptions)}


def plan_json(plan: Plan) -> dict[str, Any]:
    keep = ("id", "tier", "feasible", "device_list", "tensor_split", "cpu_layers", "n_gpu_units", "capacity_mode",
            "engine_flags", "heuristic", "warnings", "reasons")
    out = {k: v for k, v in plan.model_dump(mode="json").items() if k in keep}
    out["stream"] = plan.stream.model_dump(mode="json") if plan.stream else None
    out["assignments"] = [{"device": a.device, "layers": a.layers, "holds_output": a.holds_output,
                           "total_bytes_est": a.total_bytes_est, "budget_bytes": a.budget_bytes, "fits": a.fits}
                          for a in plan.assignments]
    return out


def gpu_overflows(s: Scored) -> list[str]:
    """The optimizer's overflow reasons for stages other than the host CPU (whose residency the layout sets)."""
    return [r for r in s.split_cpu.reasons if not r.startswith(f"{s.inputs.host.node_id}:{bridge.HOST_CPU} ")]


# --------------------------------------------------------------------------------------------------- processes


def runtime_stream(plan: Plan, model_path: Path, work_dir: Path, label: str) -> dict[str, Any] | None:
    """runtime.prepare_stream for the plan (the deployment path), its config kept for comparison with bench.py's."""
    if plan.tier != "stream" or plan.stream is None:
        return None
    stock = engine.find_installed()
    if stock is None:
        raise rl.RankError("no installed llama.cpp engine")
    launch = runtime.prepare_stream(stock.bin_dir, model_path, plan.stream, work_dir, label)
    return {"config": str(launch.config_path), "overlay": str(launch.bin_dir), "proxy_sha256": launch.proxy_sha256}


def power_online() -> tuple[bool | None, str]:
    if os.name != "nt":
        return None, "not Windows: no root/wmi BatteryStatus to read"
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           "Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | ForEach-Object { $_.PowerOnline }"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"PowerOnline unreadable: {bl._err(exc)}"
    return rl.parse_power(out.stdout, out.returncode, out.stderr)


def llama_pids() -> dict[int, str]:
    out = {}
    for p in psutil.process_iter(["name"]):
        name = (p.info.get("name") or "").lower()
        if name.startswith(LLAMA_PROCESSES):
            out[p.pid] = name
    return out


@dataclass(frozen=True)
class BenchRun:
    status: str | None
    path: Path | None
    exit_code: int | None
    tail: tuple[str, ...]


def run_bench(case_file: Path, args: argparse.Namespace) -> BenchRun:
    """bench.py for one case file as a child process; its own child dies with it (Job Object and tree kill)."""
    argv = [sys.executable, str(BENCH), "--case-file", str(case_file), "--prompt-tokens", str(args.prompt_tokens),
            "--output-tokens", str(args.output_tokens), "--reps", str(args.reps), "--warmup", str(args.warmup)]
    argv += ["--dry-run"] if args.dry_run else []
    lines: list[str] = []
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", cwd=str(ROOT))
    try:
        for line in proc.stdout or ():
            print(f"  | {line.rstrip()}", flush=True)
            lines.append(line)
        code = proc.wait(timeout=BENCH_TIMEOUT_S)
    except BaseException:
        _kill_tree(proc)
        raise
    status, path = rl.parse_bench_output(lines, ROOT)
    return BenchRun(status=status, path=path, exit_code=code, tail=tuple(x.rstrip() for x in lines[-15:]))


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        kids = psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []
    except psutil.Error:
        kids = []
    for p in kids:
        try:
            p.kill()
        except psutil.Error:
            pass
    proc.kill()
    proc.wait(timeout=30)


# --------------------------------------------------------------------------------------------------- campaign


@dataclass(frozen=True)
class Campaign:
    args: argparse.Namespace
    out_dir: Path
    model_paths: tuple[str, ...]
    doc: dict[str, Any]


def row_for(s: Scored, model_paths: Sequence[str], label: str, threads: int) -> tuple[dict[str, Any], rl.CaseBuild]:
    plan = s.plan if s.plan.feasible else s.plan.model_copy(update={"feasible": True})
    build = rl.bench_case(plan, model_paths, label, threads)
    notes = list(build.notes)
    if not s.plan.feasible:
        notes.append("the optimizer marks this layout infeasible (prediction.optimizer_reasons); its adapter "
                     "arguments are built as for a feasible plan so the tier's flags apply, and bench.py's RAM "
                     "preflight decides whether it runs")
    row = {"name": s.spec.name, "source": s.spec.source, "spans": [list(x) for x in s.spec.spans],
           "spans_text": s.spec.text(), "streamed_units": list(s.spec.streamed), "slots": s.spec.slots,
           "cpu_only_inputs": s.spec.cpu_only, "prediction": prediction_json(s), "plan": plan_json(s.plan),
           "case": {"label": label, "notes": notes, "adapter_argv": list(build.adapter_argv),
                    "bench_argv": list(build.bench_argv), **rl.argv_diff(build.adapter_argv, build.bench_argv)}}
    return row, build


def _compare_stream_config(row: dict[str, Any], bench_doc: Mapping[str, Any]) -> None:
    rt = row["case"].get("runtime_prepare_stream")
    if rt is None:
        return
    launch = bench_doc.get("launch") or {}
    if not launch.get("config"):
        rt["compared"] = "bench.py rendered no config (dry run, or the launch was not reached)"
        return
    same, detail = rl.same_config(Path(rt["config"]), Path(launch["config"]))
    rt["compared"] = {"identical": same, "detail": detail, "bench_config": launch["config"],
                      "same_overlay": launch.get("overlay") == rt["overlay"],
                      "same_proxy": launch.get("proxy_sha256") == rt["proxy_sha256"]}


def _gate(s: Scored, c: Campaign, row: dict[str, Any]) -> str | None:
    """Why this layout must not run now (row['skipped'] is set), and whether that stops the campaign."""
    overflow = gpu_overflows(s)
    if overflow and not c.args.run_infeasible:
        row["skipped"] = "not run: the optimizer says a non-CPU stage overflows its pool: " + "; ".join(overflow)
        return None
    if c.args.dry_run:
        return None
    online, why = power_online()
    c.doc["power_checks"].append({"before": s.spec.name, "at": datetime.now(UTC).isoformat(), "online": online,
                                  "detail": why})
    if online is not True:
        row["skipped"] = f"not run: {why}"
        return f"stopped before layout {s.spec.name}: {why}"
    return None


def run_layout(s: Scored, c: Campaign) -> str | None:
    """Plan, case and (after the AC check) bench.py for one layout. Returns why the campaign must stop, or None."""
    label = f"rank-{c.args.model}-{s.spec.name}"[:80]
    row, build = row_for(s, c.model_paths, label, c.args.threads)
    c.doc["layouts"].append(row)
    case_file = c.out_dir / "cases" / f"{s.spec.name}.json"
    case_file.parent.mkdir(exist_ok=True)
    case_file.write_text(json.dumps([asdict(build.case)], indent=1), encoding="utf-8")
    row["case"]["file"] = str(case_file)
    row["case"]["runtime_prepare_stream"] = runtime_stream(s.plan, Path(c.model_paths[0]),
                                                           c.out_dir / "runtime-prepare", label)
    stop = _gate(s, c, row)
    if stop or row.get("skipped"):
        return stop
    before = llama_pids()
    run = run_bench(case_file, c.args)
    leftover = {pid: name for pid, name in llama_pids().items() if pid not in before}
    row["bench"] = {"status": run.status, "exit_code": run.exit_code, "path": str(run.path) if run.path else None,
                    "output_tail": list(run.tail), "llama_processes_left": leftover}
    if run.path is not None and run.path.is_file():
        bench_doc = json.loads(run.path.read_text(encoding="utf-8"))
        row["measured"] = rl.measured_json(bench_doc, run.path)
        _compare_stream_config(row, bench_doc)
    if leftover:
        return f"llama processes still running after layout {s.spec.name}: {leftover}"
    return "interrupted by the operator" if run.status == "interrupted" else None


def prepare(args: argparse.Namespace, host: Host) -> tuple[list[Scored], tuple[str, ...], dict[str, Any]]:
    """Inputs, layouts (plus the optimizer's best) and their scores; the context document of the run."""
    workload = Workload(ctx_size=args.ctx, prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
                        warm=True)
    gpu = search(host, host.local, args.model, workload)
    specs = [rl.parse_layout(text, gpu.inputs.manifest.n_layer) for text in args.layout]
    if len({s.name for s in specs}) != len(specs):
        raise rl.RankError("layout names must be unique")
    best, same_as_best = gpu.result.best, None
    if best is not None and args.optimizer_best:
        b = rl.spec_from_candidate("opt-best", best, "optimizer best")
        same_as_best = next((s.name for s in specs if s.key() == b.key()), None)
        specs += [] if same_as_best else [b]
    cpu = (search(host, without_accelerators(host.local), args.model, workload)
           if gpu.inputs.host.gpu is not None and any(s.cpu_only for s in specs) else None)
    scored = [score(s, cpu if s.cpu_only and cpu is not None else gpu) for s in specs]
    model_paths = tuple(str(p) for p in runtime.model_shards(Path(host.models.primary_path(args.model))))
    head = gpu.inputs.manifest.n_layer
    context = {"model": args.model, "fingerprint": gpu.inputs.manifest.fingerprint, "model_paths": model_paths,
               "workload": bridge.workload_json(workload), "threads": args.threads,
               "inventory": str(args.nodes_json) if args.nodes_json else "live (inventory.collect)",
               "pools": bridge.pools_json(gpu.inputs.snapshot), "measured_coefficients": list(gpu.inputs.measured),
               "notes": list(gpu.inputs.notes), "warnings": list(gpu.inputs.warnings),
               "optimizer": {"best": bridge.candidate_json(best, head) if best else None,
                             "best_is_listed_layout": same_as_best, "explain": gpu.result.explain(),
                             "candidates": [bridge.candidate_json(c, head) for c in gpu.result.candidates]},
               "cpu_only_inputs": {"cpu_admissible_bytes": cpu.inputs.host.cpu_pool_admissible,
                                   "notes": list(cpu.inputs.notes)} if cpu else None}
    return scored, model_paths, context


def new_out_dir(model: str) -> Path:
    base = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-ranking-{model}"
    for n in range(1, 100):
        path = RESULTS / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise rl.RankError(f"no free results directory for {base}")


def write_doc(out_dir: Path, doc: dict[str, Any]) -> Path:
    for r in doc["layouts"]:
        r["abs_pct_error"] = rl.abs_pct_error(r)
        r["term_check"] = rl.term_check(r)
    doc["ranking"] = rl.ranking(doc["layouts"])
    doc["finished_at"] = datetime.now(UTC).isoformat()
    path = out_dir / "ranking.json"
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return path


def campaign(args: argparse.Namespace, scored: Sequence[Scored], model_paths: tuple[str, ...],
             context: dict[str, Any], power: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    out_dir = new_out_dir(args.model)
    doc: dict[str, Any] = {"schema": SCHEMA, "started_at": datetime.now(UTC).isoformat(), "dry_run": args.dry_run,
                           "status": "running", **context, "power_checks": power, "layouts": []}
    c = Campaign(args=args, out_dir=out_dir, model_paths=model_paths, doc=doc)
    stop = None
    for i, s in enumerate(scored):
        print(f"[{s.spec.name}] {s.spec.text()}: predicted decode {s.candidate.decode_ms:.2f} ms/token", flush=True)
        try:
            stop = run_layout(s, c)
        except BaseException as exc:  # the evidence must not keep saying "running" after a crash or an interrupt
            doc["status"] = ("interrupted by the operator" if isinstance(exc, KeyboardInterrupt)
                             else f"failed at layout {s.spec.name}: {bl._err(exc)}")
            doc["not_run"] = [x.spec.name for x in scored[i + 1:]]
            raise
        finally:
            write_doc(out_dir, doc)
        if stop:
            doc["not_run"] = [x.spec.name for x in scored[i + 1:]]
            break
    doc["status"] = f"stopped: {stop}" if stop else ("dry_run" if args.dry_run else "complete")
    path = write_doc(out_dir, doc)
    print(*rl.table(doc["layouts"], doc["ranking"]), sep="\n")
    return path, doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="catalog model id (a downloaded GGUF)")
    ap.add_argument("--layout", action="append", required=True,
                    help="NAME=DEV:a-b,DEV:b-c[;stream=x-y[,u-v];slots=K]")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=12, help="-t for every case (the CPU calibration used 12)")
    ap.add_argument("--nodes-json", type=Path, help="take this machine's node from a gateway /api/nodes snapshot "
                    "instead of running inventory.collect")
    ap.add_argument("--store-dir", type=Path, help="measurement store directory (default: runtime home / capacity)")
    ap.add_argument("--no-optimizer-best", dest="optimizer_best", action="store_false")
    ap.add_argument("--run-infeasible", action="store_true", help="also run layouts whose GPU stage overflows")
    ap.add_argument("--dry-run", action="store_true", help="score and plan, then bench.py --dry-run (no engine)")
    args = ap.parse_args(argv)
    if args.reps < bl.MIN_MEASURED_REPS or args.warmup < bl.MIN_WARMUP_REPS or args.threads < 1:
        ap.error(f"--reps must be >= {bl.MIN_MEASURED_REPS}, --warmup >= {bl.MIN_WARMUP_REPS}, --threads >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    power: list[dict[str, Any]] = []
    if args.nodes_json is None:  # inventory.collect starts llama-server (--version, --list-devices)
        online, why = power_online()
        power.append({"before": "inventory", "at": datetime.now(UTC).isoformat(), "online": online, "detail": why})
        if online is not True:
            print(f"stopped before collecting the inventory: {why}", file=sys.stderr)
            return 3
    try:
        store = MeasurementStore(args.store_dir) if args.store_dir else MeasurementStore()
        host = Host(local=local_node(args.nodes_json), models=ModelService(), store=store)
        scored, model_paths, context = prepare(args, host)
    except CapacityUnavailable as exc:
        print(f"error: {exc.code}: {exc}", file=sys.stderr)
        return 2
    except (rl.RankError, ModelError, ManifestError, OSError, ValueError) as exc:
        print(f"error: {bl._err(exc)}", file=sys.stderr)
        return 2
    path, doc = campaign(args, scored, model_paths, context, power)
    print(f"{doc['status']}: {path.relative_to(ROOT).as_posix()}", flush=True)
    if doc["status"].startswith("stopped"):
        return 3
    measured_all = all(rl.decode_median(r) is not None for r in doc["layouts"])
    return 0 if doc["status"] == "dry_run" or measured_all else 1


if __name__ == "__main__":
    sys.exit(main())
