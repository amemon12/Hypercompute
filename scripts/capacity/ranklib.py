"""Pure helpers of scripts/capacity/rank_layouts.py: layout specs, the adapter argv to bench case translation, the
measured fields of a bench.json, rank statistics and the plain table. Nothing here starts a process or reads hardware.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hypercompute.adapters.base import DeploymentSpec
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.capacity import benchlib as bl
from hypercompute.capacity.bridge import HOST_CPU
from hypercompute.capacity.stream_config import StreamOptions
from hypercompute.capacity.types import GIB, PlacementCandidate
from hypercompute.schemas import Plan

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,23}$")
RANGE_RE = re.compile(r"^(?P<a>\d+)-(?P<b>\d+)$")
BENCH_LINE_RE = re.compile(r"^\[(?P<label>[^\]]+)\] (?P<status>ok|partial|failed|refused|interrupted|dry_run): "
                           r"(?P<path>\S+bench\.json)\s*$")
# llama-server flags LlamaRpcAdapter.build_args writes, with their arity. Anything else has no bench.py equivalent.
ADAPTER_ARITY = {"-m": 1, "--alias": 1, "--load-mode": 1, "--no-host": 0, "--no-repack": 0, "--no-op-offload": 0,
                 "--cache-ram": 1, "--rpc": 1, "--device": 1, "--split-mode": 1, "--tensor-split": 1, "-ngl": 1,
                 "--fit": 1, "-c": 1, "-np": 1, "-fa": 1, "-b": 1, "-ub": 1, "-t": 1, "-tb": 1, "--host": 1,
                 "--port": 1, "--metrics": 0, "--no-ui": 0, "-lv": 1, "--log-colors": 1}
PASSED_FLAGS = ("--no-host", "--no-repack", "--no-op-offload", "--cache-ram", "-fa", "-b", "-ub", "-tb")
FIXED = {"--split-mode": "layer", "--fit": "off", "-np": "1", "-lv": "4", "--log-colors": "off"}
AC_POWER = frozenset({"ac", "no_battery"})  # bench.py power_source values that cannot be battery
DROPPED = {"--alias": "the served model name; bench.py talks to /completion and names nothing",
           "--host": "bench.py binds the child to 127.0.0.1 itself",
           "--port": "bench.py picks a free loopback port",
           "--metrics": "the Prometheus endpoint; bench.py reads the engine's own timings"}


class RankError(ValueError):
    """A layout, model or result the ranking will not use."""


# --------------------------------------------------------------------------------------------------- layouts


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    spans: tuple[tuple[str, int, int], ...]  # (engine device, first unit, end unit), half-open
    streamed: tuple[int, ...] = ()  # host CPU units read from the GGUF per pass
    slots: int = 0
    source: str = "listed"

    @property
    def cpu_only(self) -> bool:
        return all(dev == HOST_CPU for dev, _, _ in self.spans)

    def key(self) -> tuple[object, ...]:
        return self.spans, frozenset(self.streamed), self.slots if self.streamed else 0

    def text(self) -> str:
        spans = ",".join(f"{dev}:{a}-{b}" for dev, a, b in self.spans)
        if not self.streamed:
            return spans
        runs = _runs(self.streamed)
        return f"{spans};stream={','.join(f'{a}-{b}' for a, b in runs)};slots={self.slots}"


def _runs(units: Sequence[int]) -> list[tuple[int, int]]:
    """Half-open ranges covering sorted units."""
    out: list[tuple[int, int]] = []
    for u in sorted(units):
        if out and out[-1][1] == u:
            out[-1] = (out[-1][0], u + 1)
        else:
            out.append((u, u + 1))
    return out


def _range(text: str, what: str) -> tuple[int, int]:
    m = RANGE_RE.match(text.strip())
    if not m or int(m["a"]) >= int(m["b"]):
        raise RankError(f"{what} {text!r} must be a non-empty half-open unit range a-b")
    return int(m["a"]), int(m["b"])


def _stream_options(name: str, options: Sequence[str]) -> tuple[tuple[int, ...], int]:
    opts: dict[str, str] = {}
    for option in options:
        key, eq, value = option.partition("=")
        if not eq or key not in ("stream", "slots") or key in opts:
            raise RankError(f"layout {name}: option {option!r} is not stream=a-b or slots=K (each at most once)")
        opts[key] = value.strip()
    if "stream" not in opts:
        if "slots" in opts:
            raise RankError(f"layout {name}: slots without stream")
        return (), 0
    streamed: list[int] = []
    for part in opts["stream"].split(","):
        a, b = _range(part, f"layout {name}: stream")
        streamed += range(a, b)
    if len(set(streamed)) != len(streamed):
        raise RankError(f"layout {name}: stream ranges overlap")
    slots = opts.get("slots", "2")
    if not slots.isdigit() or int(slots) < 1:
        raise RankError(f"layout {name}: slots must be a whole number of at least 1, not {slots!r}")
    return tuple(sorted(streamed)), int(slots)


def parse_layout(text: str, n_layer: int) -> LayoutSpec:
    """NAME=DEV:a-b,DEV:b-c[;stream=x-y[,u-v];slots=K] as a LayoutSpec checked against the model's n_layer + 1
    units."""
    name, sep, body = text.partition("=")
    if not sep or not NAME_RE.match(name):
        raise RankError(f"layout {text!r}: give NAME=SPANS with a NAME of 1-24 letters, digits, '.', '_' or '-'")
    head, *options = [p.strip() for p in body.split(";")]
    spans = []
    for part in head.split(","):
        dev, colon, rng = part.strip().partition(":")
        if not colon or not dev.isalnum():
            raise RankError(f"layout {name}: span {part!r} is not DEVICE:a-b")
        spans.append((dev, *_range(rng, f"layout {name}: span")))
    streamed, slots = _stream_options(name, options)
    return check_layout(LayoutSpec(name=name, spans=tuple(spans), streamed=streamed, slots=slots), n_layer)


def check_layout(spec: LayoutSpec, n_layer: int) -> LayoutSpec:
    starts, ends = [a for _, a, _ in spec.spans], [b for _, _, b in spec.spans]
    if starts != [0, *ends[:-1]] or ends[-1] != n_layer + 1:
        raise RankError(f"layout {spec.name}: spans must be contiguous from unit 0 to {n_layer + 1} (the head is "
                        f"unit {n_layer})")
    devices = [dev for dev, _, _ in spec.spans]
    if len(set(devices)) != len(devices) or HOST_CPU in devices[1:]:
        raise RankError(f"layout {spec.name}: each device holds one span and the host CPU only the prefix")
    if spec.streamed and (devices[0] != HOST_CPU or not set(spec.streamed) <= set(range(spec.spans[0][2]))):
        raise RankError(f"layout {spec.name}: only units of the host CPU span can stream")
    return spec


def spec_from_candidate(name: str, c: PlacementCandidate, source: str) -> LayoutSpec:
    first = c.stages[0]
    streamed = tuple(first.streamed_units) if first.device == HOST_CPU else ()
    return LayoutSpec(name=name, spans=tuple((s.device, s.units[0], s.units[-1] + 1) for s in c.stages),
                      streamed=streamed, slots=first.slots if streamed else 0, source=source)


# --------------------------------------------------------------------------------------------------- bench case


@dataclass(frozen=True)
class CaseBuild:
    case: bl.Case
    notes: tuple[str, ...]  # every way the bench argv differs from the adapter's, and why
    adapter_argv: tuple[str, ...]
    bench_argv: tuple[str, ...]


def adapter_argv(plan: Plan, model_path: Path, label: str) -> list[str]:
    """LlamaRpcAdapter.build_args for the plan (it reads only the spec, as tests/test_capacity_bridge.py uses it)."""
    adapter = object.__new__(LlamaRpcAdapter)
    return adapter.build_args(DeploymentSpec(deployment_id=label, plan=plan, model_path=model_path, alias=label), 0)


def argv_pairs(argv: Sequence[str]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag not in ADAPTER_ARITY:
            raise RankError(f"llama-server argument {flag!r} has no bench.py equivalent")
        n = ADAPTER_ARITY[flag]
        if i + n >= len(argv) + (n == 0) or any(f == flag for f, _ in out):
            raise RankError(f"llama-server argument {flag!r} is missing its value or repeated")
        out.append((flag, argv[i + 1] if n else None))
        i += 1 + n
    return out


def _check_fixed(args: Mapping[str, str | None], first_shard: str) -> None:
    wrong = [f"{flag} {args[flag]}" for flag, want in FIXED.items() if flag in args and args[flag] != want]
    if args.get("-m") != first_shard or "--no-ui" not in args or "--rpc" in args or wrong:
        raise RankError(f"the adapter argv does not match what bench.py runs (model {args.get('-m')!r}, rpc "
                        f"{args.get('--rpc')!r}, fixed values {wrong})")


def _added(args: Mapping[str, str | None], threads: int) -> tuple[list[str], str, list[str]]:
    """Flags passed on, the load mode, and a note for each value bench.py sets that the adapter leaves out."""
    flags = [part for f in PASSED_FLAGS if f in args for part in ((f,) if args[f] is None else (f, str(args[f])))]
    notes = [f"{flag} dropped: {why}" for flag, why in DROPPED.items() if flag in args]
    if "--cache-ram" not in args:
        flags += ["--cache-ram", "0"]
        notes.append("--cache-ram 0 added: the adapter leaves the host prompt cache on for this tier; bench.py turns "
                     "it off for every case, as the calibration runs did")
    load_mode = args.get("--load-mode")
    if load_mode is None:
        load_mode = "mmap"
        notes.append("--load-mode mmap added: the adapter leaves llama-server's default (auto: mmap unless a device "
                     "does not support it); bench.py passes mmap or none, and the GPU calibration ran with mmap")
    if "-t" not in args:
        notes.append(f"-t {threads} added: the plan sets no threads (llama.cpp then picks its default); the CPU "
                     f"coefficients were measured with -t {threads}")
    return flags, str(load_mode), notes


def bench_case(plan: Plan, model_paths: Sequence[str], label: str, threads: int) -> CaseBuild:
    """The bench.py case that runs what LlamaRpcAdapter.build_args writes for the plan."""
    argv = adapter_argv(plan, Path(model_paths[0]), label)
    args = dict(argv_pairs(argv))
    _check_fixed(args, model_paths[0])
    flags, load_mode, notes = _added(args, threads)
    device, split, ngl = str(args["--device"]), str(args.get("--tensor-split") or ""), str(args["-ngl"])
    engine_args = bl.EngineArgs(devices=() if device == "none" else tuple(device.split(",")),
                                tensor_split=tuple(int(n) for n in split.split(",")) if split else (),
                                ngl=ngl if ngl == "all" else int(ngl), ctx=int(str(args["-c"])),
                                threads=int(str(args.get("-t") or threads)), load_mode=load_mode,  # type: ignore[arg-type]
                                flags=tuple(flags))
    case = bl.Case(label=label, model_paths=tuple(model_paths), engine=engine_args, stream=stream_layout(plan))
    bench = bl.build_argv(case, Path("llama-server"), 0)[1:]
    return CaseBuild(case=case, notes=tuple(notes), adapter_argv=tuple(argv), bench_argv=tuple(bench))


def stream_layout(plan: Plan) -> bl.StreamLayout | None:
    """The plan's StreamSpec with the StreamOptions runtime.prepare_stream renders by default."""
    if plan.tier != "stream" or plan.stream is None:
        return None
    s = plan.stream
    return bl.StreamLayout(resident_units=tuple(s.resident_units), streamed_units=tuple(s.streamed_units),
                           slots=max(s.slots, 1), options=StreamOptions())


def argv_diff(adapter: Sequence[str], bench: Sequence[str]) -> dict[str, list[str]]:
    def words(argv: Sequence[str]) -> set[str]:
        return {flag if value is None else f"{flag} {value}" for flag, value in argv_pairs(argv)}

    a, b = words(adapter), words(bench)
    return {"adapter_only": sorted(a - b), "bench_only": sorted(b - a)}


def same_config(a: Path, b: Path) -> tuple[bool, str]:
    """Whether two hcstream configs are identical apart from the stats file path."""
    def body(p: Path) -> list[str]:
        return [line for line in p.read_text(encoding="utf-8").splitlines() if not line.startswith("stats ")]

    try:
        x, y = body(a), body(b)
    except OSError as exc:
        return False, f"cannot compare: {bl._err(exc)}"
    if x == y:
        return True, f"identical apart from the stats path ({len(x)} lines)"
    diff = next(i for i, (p, q) in enumerate(zip([*x, None], [*y, None], strict=False)) if p != q)
    return False, f"first difference at line {diff + 1} (stats lines left out)"


# --------------------------------------------------------------------------------------------------- outputs


def parse_power(stdout: str, code: int, stderr: str = "") -> tuple[bool | None, str]:
    """PowerOnline per battery from `Get-CimInstance root/wmi BatteryStatus`: True only when every one says True."""
    values = [line.strip() for line in stdout.splitlines() if line.strip()]
    if code != 0 or not values:
        return None, f"PowerOnline unreadable (exit {code}){': ' + stderr.strip()[:200] if stderr.strip() else ''}"
    if any(v not in ("True", "False") for v in values):
        return None, f"unexpected PowerOnline output {values}"
    return all(v == "True" for v in values), f"PowerOnline {', '.join(values)}"


def parse_bench_output(lines: Iterable[str], root: Path) -> tuple[str | None, Path | None]:
    for line in lines:
        m = BENCH_LINE_RE.match(line.strip())
        if m:
            path = Path(m["path"])
            return m["status"], path if path.is_absolute() else root / path
    return None, None


def _spread(metrics: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = metrics.get(key)
    return dict(value) if isinstance(value, Mapping) else {"n": 0}


def stall_per_pass(reps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """hcstream stall_ns / passes of each measured rep's uncached phase, in ms (its prefill pass included)."""
    values = []
    for rep in reps:
        if rep.get("warmup"):
            continue
        for phase in rep.get("phases") or []:
            hc = (phase.get("io") or {}).get("hcstream") or {}
            if phase.get("phase") == "uncached" and phase.get("ok") and hc.get("passes"):
                values.append(hc["stall_ns"] / hc["passes"] / 1e6)
    return bl.spread(values)


def measured_json(doc: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """The measured fields of one bench.json (schema hypercompute.capacity.bench/1)."""
    summary = doc.get("summary") or {}
    m = summary.get("metrics") or {}
    reps = doc.get("reps") or []
    load = doc.get("load") or {}
    gpu = [{"process_used_mib": r["gpu"].get("process_used_mib"),
            "device_used_mib": [g.get("used_mib") for g in r["gpu"].get("gpus") or []]}
           for r in reps if not r.get("warmup") and isinstance(r.get("gpu"), dict)]
    return {"provenance": "measured on this host by bench.py", "evidence": str(path), "status": doc.get("status"),
            "refused": doc.get("refused") or [], "error": doc.get("error"), "n_reps": summary.get("n_reps"),
            "n_reps_ok": summary.get("n_reps_ok"), "failures": summary.get("failures") or [],
            "decode_ms_per_token": _spread(m, "decode_ms_per_token"), "decode_tok_s": _spread(m, "decode_tok_s"),
            "load_s": load.get("seconds"), "prefill_ms": _spread(m, "prefill_ms"),
            "prefill_tok_s": _spread(m, "prefill_tok_s"),
            "sys_disk_read_bytes_per_token": _spread(m, "sys_disk_read_bytes_per_token"),
            "hcstream_bytes_per_pass": _spread(m, "hcstream_bytes_per_pass"),
            "hcstream_stall_ms_per_pass": stall_per_pass(reps), "private_peak_bytes": _spread(m, "private_peak"),
            "whole_run_private_peak_bytes": ((doc.get("stop") or {}).get("memory_whole_run") or {}).get(
                "private_peak"), "gpu_snapshots_mib": gpu,
            "load_lines": [x for x in load.get("log_lines") or [] if "offloaded" in x or "model buffer size" in x],
            "power": sorted({str((r.get("power") or {}).get("source")) for r in reps}),
            "preflight": doc.get("preflight")}


def power_problem(row: Mapping[str, Any]) -> str | None:
    """Why the row's reps are not known to have run on AC (bench.py records the source per rep), or None. A battery
    or unknown source mid-run would otherwise enter the ranking unflagged; calibrate.py refuses such runs too."""
    sources = (row.get("measured") or {}).get("power") or []
    if not sources or not set(sources) <= AC_POWER:
        return f"not ranked: reps ran with power source {sorted(sources)}, not AC only"
    return None


def decode_median(row: Mapping[str, Any]) -> float | None:
    d = (row.get("measured") or {}).get("decode_ms_per_token") or {}
    if d.get("n", 0) < bl.MIN_MEASURED_REPS or not d.get("median") or power_problem(row):
        return None
    return float(d["median"])


def term_check(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Where the prediction is off, read against the stage terms (derived from one measured total; not stored)."""
    meas = decode_median(row)
    if meas is None:
        return None
    pred = row["prediction"]
    out: dict[str, Any] = {"provenance": "derived: measured decode median against predicted terms; not stored",
                           "residual_ms": meas - pred["decode_ms"], "measured_over_predicted": meas / pred["decode_ms"]}
    stages = {st["stage"].split(":")[-1]: st for st in pred["stages"]}
    cpu, gpu = stages.get(HOST_CPU), next((v for k, v in stages.items() if k != HOST_CPU), None)
    for name, st in (("cpu", cpu), ("gpu", gpu)):
        if st is None:
            continue
        io = next((t for t in st["terms"] if t["term"] == "exposed I/O"), None)
        if io is not None and io.get("slots", 0) >= 2 and io["ms"] > 0:
            # K >= 2: stage = max(io, compute); predicted I/O-bound, so the stage time says nothing about compute and
            # any I/O error would land on the compute coefficient (negative values included)
            out[f"implied_{name}_decode_ms_per_gib"] = None
            out[f"implied_{name}_not_derived"] = "predicted I/O-bound with 2+ slots: compute overlaps the reads"
            continue
        compute = st["terms"][0]
        rest = sum(x["stage_ms"] for x in pred["stages"]) - compute["ms"]
        out[f"implied_{name}_decode_ms_per_gib"] = (meas - rest) / compute["weight_gib"]
    if cpu is not None or gpu is not None:
        out["implied_basis"] = "that stage's compute coefficient if every other predicted term were exact"
    stall = (row.get("measured") or {}).get("hcstream_stall_ms_per_pass") or {}
    if cpu is not None and len(cpu["terms"]) > 1 and stall.get("n"):
        out["predicted_exposed_io_ms"] = cpu["terms"][1]["ms"]
        out["measured_stall_ms_per_pass"] = stall["median"]
    return out


# --------------------------------------------------------------------------------------------------- ranking


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, smallest first; ties share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b (ties in either sequence corrected); None when one sequence is constant."""
    if len(x) != len(y):
        raise ValueError("kendall_tau_b needs paired values")
    conc = disc = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = (x[i] > x[j]) - (x[i] < x[j]), (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else None


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation of average ranks; None for fewer than 2 pairs or a constant sequence."""
    if len(x) != len(y):
        raise ValueError("spearman_rho needs paired values")
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    var = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / var if var else None


def ranking(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Ranks and rank correlations over the layouts with a measured decode median of at least 3 reps."""
    used = [r for r in rows if decode_median(r) is not None]
    excluded = [{"name": r["name"], "why": r.get("skipped") or (power_problem(r) if r.get("measured") else None)
                 or (r.get("measured") or {}).get("status") or "not measured"}
                for r in rows if decode_median(r) is None]
    pred = [float(r["prediction"]["decode_ms"]) for r in used]
    meas = [float(decode_median(r) or 0.0) for r in used]
    names = [r["name"] for r in used]
    pr, mr = ranks(pred), ranks(meas)
    return {"basis": "decode ms/token: predicted (warm workload) against the measured median; rank 1 is fastest",
            "n": len(used), "predicted_rank": dict(zip(names, pr, strict=True)),
            "measured_rank": dict(zip(names, mr, strict=True)), "order_matches": pr == mr,
            "kendall_tau_b": kendall_tau_b(pred, meas) if len(used) >= 2 else None,
            "spearman_rho": spearman_rho(pred, meas), "excluded": excluded}


def abs_pct_error(row: Mapping[str, Any]) -> float | None:
    meas = decode_median(row)
    return abs(row["prediction"]["decode_ms"] - meas) / meas * 100 if meas else None


def _gib(n: object) -> str:
    return f"{n / GIB:.2f}" if isinstance(n, int | float) else "-"


def _num(value: object, fmt: str) -> str:
    return format(value, fmt) if isinstance(value, int | float) else "-"


def table(rows: Sequence[Mapping[str, Any]], rank: Mapping[str, Any]) -> list[str]:
    head = (f"{'layout':<10} {'spans':<36} {'pred ms':>8} {'p.rank':>6} {'meas ms (min-max)':>24} {'m.rank':>6} "
            f"{'err %':>6} {'load s':>6} {'prefill ms':>10} {'SSD B/tok':>10} {'priv GiB':>8}  status")
    out = [head, "-" * len(head)]
    for r in rows:
        m, p = r.get("measured") or {}, r["prediction"]
        d = m.get("decode_ms_per_token") or {}
        meas = f"{d['median']:.2f} ({d['low']:.2f}-{d['high']:.2f})" if d.get("n") else "-"
        ssd = (m.get("hcstream_bytes_per_pass") or {}).get("median") or (
            m.get("sys_disk_read_bytes_per_token") or {}).get("median")
        status = r.get("skipped") or m.get("status") or "-"
        if m and power_problem(r):
            status += f" (power {','.join(map(str, m.get('power') or ['unknown']))})"
        out.append(f"{r['name']:<10} {r['spans_text'][:36]:<36} {p['decode_ms']:>8.2f} "
                   f"{_num(rank['predicted_rank'].get(r['name']), '.1f'):>6} {meas:>24} "
                   f"{_num(rank['measured_rank'].get(r['name']), '.1f'):>6} {_num(r.get('abs_pct_error'), '.1f'):>6} "
                   f"{_num(m.get('load_s'), '.2f'):>6} {_num((m.get('prefill_ms') or {}).get('median'), '.1f'):>10} "
                   f"{_num(ssd, '.3g'):>10} {_gib((m.get('private_peak_bytes') or {}).get('high')):>8}  {status}")
    tau, rho = rank["kendall_tau_b"], rank["spearman_rho"]
    out.append(f"n {rank['n']}; order matches {rank['order_matches']}; Kendall tau-b {_num(tau, '.3f')}; "
               f"Spearman rho {_num(rho, '.3f')}")
    return out
