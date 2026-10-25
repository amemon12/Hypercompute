"""Turn benchmark and probe runs into measured coefficients the capacity optimizer reads.

  uv run --no-sync python scripts/capacity/calibrate.py storage --model qwen2.5-7b-instruct-q4_k_m
  uv run --no-sync python scripts/capacity/calibrate.py bench results/capacity/<run>/bench.json [...]

capacity/bridge.py is the consumer. It looks records up by node id, engine device name and kind, prefers a record
taken on the same model fingerprint over a model-agnostic one (fingerprint None), and ignores records older than its
max ages (compute 7 days, storage 7 days, links 1 hour). Records written here use exactly those keys: node id "local"
for this host (services.nodes.LOCAL_ID), device CPU or the engine's GPU name (CUDA0, MTL0), or the volume ("C:") for
storage, and model_fingerprint as bridge.read_manifest computes it (catalog sha256 values included), so a record
measured on a model is found when that model is planned.

bench: each bench.json (schema hypercompute.capacity.bench/1) is refused as a whole unless its status is ok, at least
3 measured reps succeeded, its shards are unchanged on disk, its model is a catalog GGUF and every rep ran on one known
power source (on battery or unknown power only with --allow-battery). Then:

  compute.decode_ms_per_gib (ms/GiB) = median over measured reps of predicted_ms / (predicted_n - 1), divided by the
      GiB of weights the device computes per token. Only single-device cases give it: stock-cpu (--device none, every
      unit on the CPU) and stock-gpu (every unit and the head on one GPU, confirmed by the engine's "offloaded N/N
      layers" log line), with no flag that overrides tensor or KV placement (PLACEMENT_FLAGS). The note records
      whether CPU weight repacking was on: the RAM and stream tiers run with --no-repack, stock-cpu does not.
      The GiB is the sum of the manifest's unit weight bytes (blocks and the head), the same bytes
      the optimizer multiplies the coefficient by. token_embd outside the head is excluded: llama.cpp reads it with
      get_rows, one row per token on the host, which is not a weight pass. With tied embeddings token_embd is the head
      matmul and is inside the head unit, so it counts.
  compute.prefill_ms_per_gib_token (ms/GiB/token) = median of prompt_ms / prompt_n of the uncached phase (cache_n 0),
      divided by the same GiB. The per-token cost depends on the prompt length (batching); the note records it.
  storage.stream_efficiency (fraction), from hcstream stream cases: B = median hcstream bytes_read / passes of the
      uncached phase, T = median decode ms per token, compute = the measured decode coefficients the bridge would use
      for the CPU (and the GPU suffix) times the GiB each computes, R = the measured storage.read_bytes_per_s of the
      volume holding the model (a rate probed on another volume does not count). With
      the CPU stage time T_cpu = T - gpu compute, the optimizer's stage model gives the exposed I/O time: K >= 2 slots
      overlap I/O with compute, T_cpu = max(io, compute), so io = T_cpu when I/O dominates; K = 1 adds them, io =
      T_cpu - compute. efficiency = B / (R * io). Limits: it is stored only when compute and I/O separate (with K >= 2
      T_cpu must exceed the CPU compute by SEPARATION_MARGIN, with K = 1 io must be at least that share of T_cpu),
      when B is within 10 percent of the planned streamed bytes, when every compute coefficient and R are measured and
      fresh, and when the result is in (0, 1]. Prefill passes are in B's pass count (one pass per ubatch), and the
      CPU-GPU activation copy is inside the GPU term.

storage: reads the model's largest shard unbuffered (FILE_FLAG_NO_BUFFERING, 4 readers, 8 MiB chunks: hcstream's
defaults) in --runs consecutive ranges of at most 4 GiB in total, writes results/capacity/<stamp>-storage-<volume>/
storage.json, then stores storage.read_bytes_per_s, the median run rate, for the model's volume. Refused on battery
power or when the power source cannot be read (Win32_Battery.BatteryStatus) unless --allow-battery.

Every record carries its evidence file, the sample count and the time it was measured. Exit status: 0 stored (or
--dry-run), 2 refused or nothing derived, 3 derived but the store write failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from hypercompute import config  # noqa: E402
from hypercompute.capacity import benchlib as bl  # noqa: E402
from hypercompute.capacity import bridge, costs, runtime, storage_probe  # noqa: E402
from hypercompute.capacity.manifest import ManifestError  # noqa: E402
from hypercompute.capacity.measurements import (  # noqa: E402
    MeasurementIdentity,
    MeasurementRecord,
    MeasurementStore,
)
from hypercompute.capacity.pools import HOST_NODE_ID  # noqa: E402
from hypercompute.capacity.types import GIB, Coefficient, ModelManifest  # noqa: E402
from hypercompute.schemas import ModelArtifact  # noqa: E402
from hypercompute.services.models import ModelError, load_manifest  # noqa: E402

MIB = 1 << 20
RESULTS = ROOT / "results" / "capacity"
SEPARATION_MARGIN = 0.25  # share by which exposed I/O must stand out from compute before it is read off the run
PLANNED_BYTES_TOLERANCE = 0.10  # measured bytes per pass vs the planned streamed bytes
NODE_ID_RE = re.compile(r"local|[a-f0-9]{32}")  # services.nodes.LOCAL_ID or a peer's NodeView id (fullmatch)
OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
# llama-server flags that move tensors or ops off the case's devices without changing its device list or -ngl, so the
# "offloaded N/N layers" line still reads as one device. A case passing any of them is not single-device.
PLACEMENT_FLAGS = frozenset({"-ot", "--override-tensor", "-cmoe", "--cpu-moe", "-ncmoe", "--n-cpu-moe", "-nkvo",
                             "--no-kv-offload"})
AC_STATUSES = frozenset({2, 3, 6, 7, 8, 9})  # Win32_Battery.BatteryStatus: on AC (2), full, or charging
BATTERY_STATUSES = frozenset({1, 4, 5})  # discharging, low, critical
POWER_SOURCES = {"ac": "ac", "no_battery": "ac", "battery": "battery"}  # bench.json host/rep power.source


class CalibrationError(ValueError):
    """A run, model or argument calibrate will not turn into coefficients."""


# --------------------------------------------------------------------------------------------------- bench runs


@dataclass(frozen=True)
class RunFacts:
    """What a bench.json says about the run, validated."""

    evidence: str
    label: str
    model_id: str
    backend_build: str
    ctx: int
    power_state: str | None
    measured_at: datetime
    case: bl.Case
    reps: tuple[Mapping[str, Any], ...]  # measured (not warmup) reps that succeeded
    load_lines: tuple[str, ...]
    proxy_sha256: str | None


def load_doc(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise CalibrationError(f"{path} is not a JSON object")
    return doc


def _obj(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def measured_reps(doc: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    reps = doc.get("reps") if isinstance(doc.get("reps"), list) else []
    return tuple(r for r in reps if isinstance(r, Mapping) and not r.get("warmup") and r.get("ok") is True)


def run_refusal(doc: Mapping[str, Any]) -> str | None:
    """Why the whole run gives no coefficient, or None."""
    if doc.get("schema") != bl.SCHEMA:
        return f"schema {doc.get('schema')!r} is not {bl.SCHEMA}"
    if doc.get("status") != "ok":
        return f"run status is {doc.get('status')!r}, not 'ok'"
    n = len(measured_reps(doc))
    if n < bl.MIN_MEASURED_REPS:
        return f"{n} measured reps succeeded; at least {bl.MIN_MEASURED_REPS} are required"
    return None


def run_power_state(doc: Mapping[str, Any], *, allow_battery: bool) -> str | None:
    """The one power state of the host and every measured rep. Mixed runs are refused; so are battery runs and runs
    whose power source was not recorded or unknown (it may have been battery), unless allow_battery."""
    sources = [_obj(_obj(doc.get("host")).get("power")).get("source")]
    sources += [_obj(r.get("power")).get("source") for r in measured_reps(doc)]
    states = {POWER_SOURCES.get(s) if isinstance(s, str) else None for s in sources}
    if len(states) != 1:
        raise CalibrationError(f"the power source changed or was unknown during the run ({sorted(map(str, sources))})")
    state = states.pop()
    if state != "ac" and not allow_battery:
        what = ("on battery power, where the CPU and GPU may be throttled" if state == "battery"
                else f"with no known power source ({sources[0]!r}), so it may have run on battery")
        raise CalibrationError(f"measured {what} (--allow-battery stores it with power_state {state})")
    return state


def _aware(text: object, what: str) -> datetime:
    try:
        value = datetime.fromisoformat(str(text))
    except ValueError as exc:
        raise CalibrationError(f"{what} {text!r} is not an ISO timestamp") from exc
    if value.tzinfo is None:
        raise CalibrationError(f"{what} {text!r} has no time zone")
    return value


def run_facts(doc: Mapping[str, Any], path: Path, *, allow_battery: bool) -> RunFacts:
    refusal = run_refusal(doc)
    if refusal:
        raise CalibrationError(refusal)
    try:
        case = bl.case_from_dict(doc.get("case"))
    except bl.BenchError as exc:
        raise CalibrationError(f"the recorded case is malformed: {exc}") from exc
    release = _obj(doc.get("llama_cpp")).get("release")
    if not isinstance(release, str) or not release:
        raise CalibrationError("the run records no engine release (llama_cpp.release)")
    model_id = _obj(doc.get("model")).get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise CalibrationError("the run records no model id")
    lines = _obj(doc.get("load")).get("log_lines") or []
    proxy = _obj(doc.get("launch")).get("proxy_sha256")
    return RunFacts(evidence=str(path.resolve()), label=case.label, model_id=model_id, backend_build=release,
                    ctx=case.engine.ctx, power_state=run_power_state(doc, allow_battery=allow_battery),
                    measured_at=_aware(doc.get("finished_at"), "finished_at"), case=case, reps=measured_reps(doc),
                    load_lines=tuple(x for x in lines if isinstance(x, str)),
                    proxy_sha256=proxy if isinstance(proxy, str) and proxy else None)


def _num(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CalibrationError(f"{what} is not a finite number: {value!r}")
    return float(value)


def _uncached(rep: Mapping[str, Any]) -> Mapping[str, Any]:
    phases = rep.get("phases") if isinstance(rep.get("phases"), list) else []
    phase = next((p for p in phases if isinstance(p, Mapping) and p.get("phase") == "uncached"), None)
    if phase is None or phase.get("ok") is not True or not isinstance(phase.get("timings"), Mapping):
        raise CalibrationError(f"rep {rep.get('index')} has no successful uncached phase with engine timings")
    return phase


def decode_ms_per_token(rep: Mapping[str, Any]) -> float:
    """predicted_ms over its predicted_n - 1 decode steps (the first token comes from the prefill)."""
    t = _uncached(rep)["timings"]
    steps = _num(t.get("predicted_n"), "predicted_n") - 1
    ms = _num(t.get("predicted_ms"), "predicted_ms")
    if steps < 1 or ms <= 0:
        raise CalibrationError(f"rep {rep.get('index')} decoded {steps + 1:.0f} tokens in {ms} ms: no decode rate")
    return ms / steps


def prefill_ms_per_token(rep: Mapping[str, Any]) -> tuple[float, int]:
    """(prompt_ms / prompt_n, prompt_n) of an uncached prefill."""
    t = _uncached(rep)["timings"]
    n, ms = _num(t.get("prompt_n"), "prompt_n"), _num(t.get("prompt_ms"), "prompt_ms")
    if _num(t.get("cache_n", 0), "cache_n"):
        raise CalibrationError(f"rep {rep.get('index')} reused cached prompt tokens: not an uncached prefill")
    if n < 1 or ms <= 0:
        raise CalibrationError(f"rep {rep.get('index')} has no prefill timing ({n} tokens in {ms} ms)")
    return ms / n, int(n)


def stream_bytes_per_pass(rep: Mapping[str, Any]) -> float:
    hc = _obj(_obj(_uncached(rep).get("io")).get("hcstream"))
    read, passes = _num(hc.get("bytes_read"), "hcstream bytes_read"), _num(hc.get("passes"), "hcstream passes")
    if passes < 1 or read <= 0:
        raise CalibrationError(f"rep {rep.get('index')} has no hcstream reads ({read} bytes over {passes} passes)")
    return read / passes


# --------------------------------------------------------------------------------------------------- model


def shard_paths(doc: Mapping[str, Any]) -> list[Path]:
    """The run's shards, refused when any changed size or mtime since the run."""
    shards = _obj(doc.get("model")).get("shards")
    if not isinstance(shards, list) or not shards:
        raise CalibrationError("the run records no model shards")
    out = []
    for s in shards:
        s = _obj(s)
        path = Path(str(s.get("path")))
        try:
            st = path.stat()
        except OSError as exc:
            raise CalibrationError(f"shard {path} is not readable now: {exc}") from exc
        if (st.st_size, st.st_mtime_ns) != (s.get("size"), s.get("mtime_ns")):
            raise CalibrationError(f"shard {path} changed since the run (size or mtime differ)")
        out.append(path)
    return out


def catalog_manifest(doc: Mapping[str, Any], artifacts: Sequence[ModelArtifact]) -> ModelManifest:
    """The manifest the bridge builds for this model (bridge.read_manifest, catalog sha256 values included), so the
    stored fingerprint is the one a plan of this model looks up."""
    model_id = _obj(doc.get("model")).get("model_id")
    art = next((a for a in artifacts if a.id == model_id), None)
    if art is None or art.format != "gguf":
        raise CalibrationError(f"{model_id!r} is not a GGUF in the model catalog; the bridge plans catalog artifacts "
                               "only and would never look its fingerprint up")
    try:
        manifest = bridge.read_manifest(art, shard_paths(doc))
    except ManifestError as exc:
        raise CalibrationError(f"cannot build the manifest of {model_id}: {exc}") from exc
    recorded = _obj(doc.get("model")).get("n_layer")
    if recorded is not None and recorded != manifest.n_layer:
        raise CalibrationError(f"the run recorded {recorded} blocks, the manifest has {manifest.n_layer}")
    return manifest


def computed_weight_bytes(manifest: ModelManifest, units: Iterable[int] | None = None) -> int:
    """Weight bytes computed per token by the given units (default all: blocks and the head). token_embd outside the
    head is a get_rows lookup on the host and never part of a unit."""
    chosen = range(len(manifest.units)) if units is None else units
    return sum(manifest.units[u].weight_bytes for u in chosen)


# --------------------------------------------------------------------------------------------------- compute


def single_device(case: bl.Case, n_layer: int) -> tuple[str | None, str]:
    """(engine device, why) when every unit and the head computed on one device, else (None, why not)."""
    devices = case.engine.devices
    if case.stream is not None:
        return None, "an hcstream case: its CPU time includes SSD reads"
    if case.rpc_endpoints or any(d.upper().startswith("RPC") for d in devices):
        return None, "RPC devices take part"
    moved = sorted({f.split("=", 1)[0] for f in case.engine.flags} & PLACEMENT_FLAGS)
    if moved:
        return None, f"its flags {', '.join(moved)} move tensors or ops to another device"
    if not devices:
        return bridge.HOST_CPU, "every unit and the head on the host CPU (--device none)"
    ngl = case.engine.ngl
    if len(devices) == 1 and (ngl == "all" or int(ngl) >= n_layer + 1):
        return devices[0], f"every unit and the head on {devices[0]} (-ngl {ngl})"
    return None, f"units are split between the host CPU and {', '.join(devices)} (-ngl {ngl})"


def offload_problem(facts: RunFacts, n_layer: int) -> str | None:
    """None when the engine log confirms all n_layer + 1 units were offloaded to the GPU."""
    found = [m for line in facts.load_lines if (m := OFFLOAD_RE.search(line))]
    if not found:
        return "the engine log has no 'offloaded N/M layers to GPU' line to confirm every unit ran on the GPU"
    done, total = int(found[-1][1]), int(found[-1][2])
    if done != total or total != n_layer + 1:
        return f"the engine offloaded {done}/{total} layers, not all {n_layer + 1} units"
    return None


def _record(facts: RunFacts, identity: MeasurementIdentity, value: float, unit: str, note: str) -> MeasurementRecord:
    return MeasurementRecord(identity=identity, value=value, unit=unit, measured_at=facts.measured_at,
                             samples=len(facts.reps), note=note, evidence=facts.evidence)


def compute_records(facts: RunFacts, manifest: ModelManifest, *, node_id: str,
                    agnostic: bool = False) -> tuple[tuple[MeasurementRecord, ...], tuple[str, ...]]:
    """Decode and prefill coefficients of a single-device run; (records, reasons nothing was derived)."""
    device, why = single_device(facts.case, manifest.n_layer)
    if device is None:
        return (), (f"no compute coefficient: {why}; decode_ms_per_gib comes only from single-device cases",)
    if device != bridge.HOST_CPU and (problem := offload_problem(facts, manifest.n_layer)):
        return (), (f"no compute coefficient for {device}: {problem}",)
    weight = computed_weight_bytes(manifest)
    gib = weight / GIB
    decode = statistics.median(decode_ms_per_token(r) for r in facts.reps)
    prefill_pairs = [prefill_ms_per_token(r) for r in facts.reps]
    prefill = statistics.median(ms for ms, _ in prefill_pairs)
    prompt_n = statistics.median(n for _, n in prefill_pairs)
    engine = facts.case.engine
    embd = "token_embd is the tied head, counted once" if manifest.tied_embeddings else "token_embd lookup excluded"
    repack = "repack off" if "--no-repack" in engine.flags else "CPU repack on"
    where = (f"{facts.model_id}, {why}, {engine.threads or 'default'} threads, {repack}, {gib:.3f} GiB of unit "
             f"weights ({embd})")
    identity = MeasurementIdentity(node_id=node_id, device=device, backend_build=facts.backend_build,
                                   model_fingerprint=manifest.fingerprint, ctx=facts.ctx,
                                   power_state=facts.power_state, kind=bridge.KIND_DECODE)
    records = [
        _record(facts, identity, decode / gib, "ms/GiB",
                f"median decode {decode:.3f} ms/token over {len(facts.reps)} reps; {where}"),
        _record(facts, replace(identity, kind=bridge.KIND_PREFILL), prefill / gib, "ms/GiB/token",
                f"median uncached prefill {prefill:.4f} ms/token at {prompt_n:g} prompt tokens; {where}"),
    ]
    if agnostic:
        records += [replace(r, identity=replace(r.identity, model_fingerprint=None),
                            note=f"{r.note}; model-agnostic copy of this {facts.model_id} measurement")
                    for r in list(records)]
    return tuple(records), ()


# --------------------------------------------------------------------------------------------------- stream


@dataclass(frozen=True)
class StreamTerms:
    decode_ms: float  # measured per token, whole pipeline
    bytes_per_pass: float  # measured hcstream bytes per forward pass
    planned_bytes: int  # streamed unit buffer bytes of the layout
    cpu_compute_ms: float  # measured CPU decode coefficient times the CPU units' GiB
    gpu_compute_ms: float  # same for the GPU suffix (0 without one)
    slots: int
    read_bytes_per_s: float  # measured storage.read_bytes_per_s


def derive_efficiency(t: StreamTerms) -> tuple[float | None, str]:
    """(efficiency, how) from the optimizer's CPU stage model, or (None, why it cannot be separated)."""
    if t.planned_bytes <= 0 or t.read_bytes_per_s <= 0:
        return None, "no planned streamed bytes or no read rate"
    drift = t.bytes_per_pass / t.planned_bytes - 1
    if abs(drift) > PLANNED_BYTES_TOLERANCE:
        return None, (f"hcstream read {t.bytes_per_pass / MIB:.1f} MiB per pass, {drift:+.0%} from the planned "
                      f"{t.planned_bytes / MIB:.1f} MiB: the run did not stream the planned layout")
    cpu_stage = t.decode_ms - t.gpu_compute_ms
    if t.slots >= 2:
        if cpu_stage < t.cpu_compute_ms * (1 + SEPARATION_MARGIN):
            return None, (f"not separable: the CPU stage took {cpu_stage:.2f} ms against {t.cpu_compute_ms:.2f} ms "
                          f"of compute, so with {t.slots} slots the I/O may be hidden under compute")
        io_ms, rule = cpu_stage, f"K={t.slots}: io = T_cpu (I/O dominates the overlapped stage)"
    else:
        io_ms, rule = cpu_stage - t.cpu_compute_ms, "K=1: io = T_cpu - compute"
        if io_ms < SEPARATION_MARGIN * cpu_stage:
            return None, (f"not separable: {io_ms:.2f} ms of exposed I/O in a {cpu_stage:.2f} ms CPU stage is within "
                          "the noise of the compute coefficient")
    efficiency = t.bytes_per_pass / (t.read_bytes_per_s * io_ms / 1000)
    if not 0 < efficiency <= 1:
        return None, (f"efficiency {efficiency:.3f} is outside (0, 1]: the compute or read-rate coefficients do not "
                      "describe this run")
    how = (f"{rule}; T {t.decode_ms:.2f} ms, GPU {t.gpu_compute_ms:.2f} ms, CPU compute {t.cpu_compute_ms:.2f} ms, "
           f"io {io_ms:.2f} ms for {t.bytes_per_pass / MIB:.1f} MiB/pass at a probed "
           f"{t.read_bytes_per_s / GIB:.2f} GiB/s")
    return efficiency, how


def _backend(device: str) -> str | None:
    return "cuda" if device.startswith("CUDA") else "metal" if device.startswith("MTL") else None


def _measured_decode(backend: str, node_id: str, device: str, records: Sequence[MeasurementRecord],
                     manifest: ModelManifest, now: datetime) -> Coefficient:
    return bridge.device_profile(backend, node_id, device, "ram" if backend == "cpu" else f"vram:{device}", None,
                                 records, manifest, now).decode_ms_per_gib


def stream_terms(facts: RunFacts, manifest: ModelManifest, records: Sequence[MeasurementRecord], node_id: str,
                 now: datetime) -> StreamTerms | str:
    """The terms of derive_efficiency read through the bridge's own lookups, or why they are not all measured."""
    case, layout = facts.case, facts.case.stream
    if layout is None or not layout.streamed_units:
        return "nothing streams in this case"
    if case.rpc_endpoints:
        return "RPC devices take part; their hops cannot be told apart from the I/O"
    cpu = bl.cpu_units(case, manifest.n_layer)
    gpu = [u for u in range(manifest.n_layer + 1) if u not in cpu]
    coefs = [("CPU", cpu, _measured_decode("cpu", node_id, bridge.HOST_CPU, records, manifest, now))]
    if gpu:
        device = case.engine.devices[0]
        backend = _backend(device)
        if backend is None:
            return f"no cost model for the GPU suffix device {device}"
        coefs.append((device, gpu, _measured_decode(backend, node_id, device, records, manifest, now)))
    missing = [name for name, _, c in coefs if c.provenance != "measured"]
    # the bridge matches read rates by node only; this run's efficiency is relative to its own volume's rate
    volume = storage_probe.volume_of(Path(case.model_paths[0]))
    same_volume = [r for r in records if r.identity.kind != bridge.KIND_STORAGE_READ or r.identity.device == volume]
    read = bridge.storage_profile(same_volume, node_id, None, now).read_bytes_per_s
    missing += [f"storage read rate of {volume}"] if read.provenance != "measured" else []
    if missing:
        return f"not separable: no fresh measurement of {', '.join(missing)} (calibrate those first)"
    ms = [costs.compute_ms(c, computed_weight_bytes(manifest, units)) for _, units, c in coefs]
    return StreamTerms(decode_ms=statistics.median(decode_ms_per_token(r) for r in facts.reps),
                       bytes_per_pass=statistics.median(stream_bytes_per_pass(r) for r in facts.reps),
                       planned_bytes=sum(manifest.units[u].buffer_bytes for u in layout.streamed_units),
                       cpu_compute_ms=ms[0], gpu_compute_ms=sum(ms[1:]), slots=layout.slots,
                       read_bytes_per_s=read.value)


def stream_record(facts: RunFacts, manifest: ModelManifest, records: Sequence[MeasurementRecord], *, node_id: str,
                  now: datetime) -> tuple[MeasurementRecord | None, str]:
    terms = stream_terms(facts, manifest, records, node_id, now)
    if isinstance(terms, str):
        return None, f"no stream efficiency: {terms}"
    efficiency, how = derive_efficiency(terms)
    if efficiency is None:
        return None, f"no stream efficiency: {how}"
    build = f"{facts.backend_build}+hcstream-{facts.proxy_sha256[:12]}" if facts.proxy_sha256 else facts.backend_build
    identity = MeasurementIdentity(node_id=node_id, device=storage_probe.volume_of(Path(facts.case.model_paths[0])),
                                   backend_build=build, model_fingerprint=manifest.fingerprint, ctx=facts.ctx,
                                   power_state=facts.power_state, kind=bridge.KIND_STREAM_EFFICIENCY)
    return _record(facts, identity, efficiency, "fraction", f"{facts.model_id} {facts.label}: {how}"), ""


# --------------------------------------------------------------------------------------------------- bench command


@dataclass(frozen=True)
class Outcome:
    evidence: str
    records: tuple[MeasurementRecord, ...] = ()
    skipped: tuple[str, ...] = ()  # coefficients not derived from an accepted run, with the reason
    refused: str | None = None  # the whole run was refused


ManifestFor = Callable[[Mapping[str, Any]], ModelManifest]


def _accepted(path: Path, manifest_for: ManifestFor,
              allow_battery: bool) -> tuple[RunFacts, ModelManifest] | Outcome:
    try:
        doc = load_doc(path)
        facts = run_facts(doc, path, allow_battery=allow_battery)
        return facts, manifest_for(doc)
    except CalibrationError as exc:
        return Outcome(evidence=str(path), refused=str(exc))


def calibrate_runs(paths: Sequence[Path], existing: Sequence[MeasurementRecord], manifest_for: ManifestFor, *,
                   node_id: str = HOST_NODE_ID, agnostic: bool = False, allow_battery: bool = False,
                   now: datetime | None = None) -> tuple[Outcome, ...]:
    """Compute coefficients from every single-device run first, then stream efficiency from stream runs, which reads
    the compute and storage coefficients through the bridge from `existing` plus the ones just derived."""
    now = now or datetime.now(UTC)
    accepted = [(p, _accepted(p, manifest_for, allow_battery)) for p in paths]
    outcomes: dict[Path, Outcome] = {p: a for p, a in accepted if isinstance(a, Outcome)}
    derived: list[MeasurementRecord] = []
    for path, item in accepted:
        if isinstance(item, Outcome) or item[0].case.stream is not None:
            continue
        try:
            records, skipped = compute_records(*item, node_id=node_id, agnostic=agnostic)
        except CalibrationError as exc:
            outcomes[path] = Outcome(evidence=item[0].evidence, refused=str(exc))
            continue
        derived += records
        outcomes[path] = Outcome(evidence=item[0].evidence, records=records, skipped=skipped)
    for path, item in accepted:
        if isinstance(item, Outcome) or item[0].case.stream is None:
            continue
        try:
            record, why = stream_record(*item, [*existing, *derived], node_id=node_id, now=now)
        except CalibrationError as exc:
            outcomes[path] = Outcome(evidence=item[0].evidence, refused=str(exc))
            continue
        outcomes[path] = Outcome(evidence=item[0].evidence, records=(record,) if record else (),
                                 skipped=(why,) if why else ())
    return tuple(outcomes[p] for p in paths)


# --------------------------------------------------------------------------------------------------- storage


def model_shards(spec: str) -> list[Path]:
    """Every shard of a model given a directory name under the models dir, a directory, or a (first) GGUF file."""
    path = Path(spec).expanduser()
    if not path.exists():
        path = config.paths().models / spec
    if path.is_dir():
        found = sorted(f for f in path.glob("*.gguf") if not f.name.startswith("mmproj"))
        if not found:
            raise CalibrationError(f"no GGUF in {path}")
        path = found[0]
    if not path.is_file():
        raise CalibrationError(f"model {spec!r} not found (looked in {config.paths().models})")
    shards = runtime.model_shards(path)
    missing = [p.name for p in shards if not p.is_file()]
    if missing:
        raise CalibrationError(f"missing shards of {spec}: {', '.join(missing)}")
    return shards


def power_from_battery_status(codes: Sequence[int]) -> str | None:
    """ac, battery or None (unknown) from Win32_Battery.BatteryStatus values; no battery at all is mains power."""
    if not codes:
        return "ac"
    if all(c in AC_STATUSES for c in codes):
        return "ac"
    if any(c in BATTERY_STATUSES for c in codes):
        return "battery"
    return None


def host_power(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, Any]:
    """This host's power state from Win32_Battery (Windows), else psutil."""
    if os.name != "nt":
        import psutil

        battery = psutil.sensors_battery()
        state = "ac" if battery is None or battery.power_plugged else "battery"
        return {"state": state, "via": "psutil.sensors_battery"}
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           "Get-CimInstance Win32_Battery | Select-Object -ExpandProperty BatteryStatus"]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"state": None, "via": "Win32_Battery", "error": f"{type(exc).__name__}: {exc}"}
    if out.returncode != 0:
        return {"state": None, "via": "Win32_Battery", "error": (out.stderr or "").strip()[:300]}
    codes = [int(x) for x in out.stdout.split() if x.isdigit()]
    return {"state": power_from_battery_status(codes), "via": "Win32_Battery.BatteryStatus", "battery_status": codes}


def storage_record(cal: storage_probe.StorageCalibration, *, node_id: str, power_state: str | None,
                   measured_at: datetime, evidence: str, readers: int, chunk_bytes: int) -> MeasurementRecord:
    rates = ", ".join(f"{r.bytes_per_s / GIB:.2f}" for r in cal.runs)
    identity = MeasurementIdentity(node_id=node_id, device=cal.volume,
                                   backend_build=f"storage_probe-{readers}r-{chunk_bytes // MIB}MiB",
                                   power_state=power_state, kind=bridge.KIND_STORAGE_READ)
    note = (f"unbuffered reads of {Path(cal.path).name}, {readers} readers x {chunk_bytes // MIB} MiB chunks, "
            f"median of {len(cal.runs)} runs ({rates} GiB/s), {cal.bytes_read / GIB:.2f} GiB read in total")
    return MeasurementRecord(identity=identity, value=cal.median_bytes_per_s, unit="bytes/s", measured_at=measured_at,
                             samples=len(cal.runs), note=note, evidence=evidence)


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")


def storage_command(args: argparse.Namespace) -> int:
    power = host_power()
    started = datetime.now(UTC)
    shards = model_shards(args.model)
    target = max(shards, key=lambda p: p.stat().st_size)
    out = args.results_dir / f"{started.strftime('%Y%m%dT%H%M%SZ')}-storage-{storage_probe.volume_of(target)[:1]}"
    doc: dict[str, Any] = {"kind": "storage_calibration", "tool": "scripts/capacity/calibrate.py storage",
                           "model": args.model, "file": str(target), "power": power, "started_at": started.isoformat(),
                           "requested": {"bytes": args.bytes, "runs": args.runs, "readers": args.readers,
                                         "chunk_bytes": args.chunk_bytes}}
    if power.get("state") != "ac" and not args.allow_battery:
        why = "on battery power" if power.get("state") == "battery" else "power source unknown"
        _write_json(out / "storage.json", {**doc, "refused": why})
        print(f"refused: {why} ({power}); evidence {out / 'storage.json'}")
        return 2
    cal = storage_probe.calibrate(target, total_bytes=args.bytes, runs=args.runs, readers=args.readers,
                                  chunk_bytes=args.chunk_bytes)
    doc |= {"volume": cal.volume, "runs": [asdict(r) | {"bytes_per_s": r.bytes_per_s} for r in cal.runs],
            "bytes_read": cal.bytes_read, "error": cal.error,
            "median_bytes_per_s": None if cal.error else cal.median_bytes_per_s}
    evidence = out / "storage.json"
    _write_json(evidence, doc)  # evidence first: a store failure must not lose it
    if cal.error:
        print(f"storage probe failed: {cal.error}; evidence {evidence}")
        return 2
    record = storage_record(cal, node_id=args.node_id, power_state=power.get("state"), measured_at=started,
                            evidence=str(evidence.resolve()), readers=args.readers, chunk_bytes=args.chunk_bytes)
    return _store_and_report([Outcome(evidence=str(evidence), records=(record,))], args)


# --------------------------------------------------------------------------------------------------- output


def prior_for(record: MeasurementRecord) -> Coefficient | None:
    """The documented prior this record replaces in the bridge (costs.py), for the report."""
    i = record.identity
    if i.kind in (bridge.KIND_DECODE, bridge.KIND_PREFILL):
        backend = "cpu" if i.device == bridge.HOST_CPU else _backend(i.device)
        if backend is None:
            return None
        p = costs.device_prior(backend, i.node_id, i.device, "ram")
        return p.decode_ms_per_gib if i.kind == bridge.KIND_DECODE else p.prefill_ms_per_gib_token
    storage = costs.storage_prior(i.node_id)
    return {bridge.KIND_STORAGE_READ: storage.read_bytes_per_s,
            bridge.KIND_STREAM_EFFICIENCY: storage.stream_efficiency}.get(i.kind)


def describe(record: MeasurementRecord) -> str:
    i, prior = record.identity, prior_for(record)
    was = f"prior {prior.value:.6g} {prior.unit}" if prior else "no prior"
    fp = i.model_fingerprint[:12] if i.model_fingerprint else "model-agnostic"
    return (f"{i.kind} {i.node_id}:{i.device} = {record.value:.6g} {record.unit} ({was}); samples {record.samples}; "
            f"{i.backend_build}, ctx {i.ctx}, power {i.power_state}, fingerprint {fp}; evidence {record.evidence}")


def _store_and_report(outcomes: Sequence[Outcome], args: argparse.Namespace) -> int:
    store = MeasurementStore(args.store_dir) if args.store_dir else MeasurementStore()
    records = [r for o in outcomes for r in o.records]
    for o in outcomes:
        print(f"{o.evidence}: {'REFUSED ' + o.refused if o.refused else f'{len(o.records)} coefficient(s)'}")
        for why in o.skipped:
            print(f"  skipped: {why}")
        for r in o.records:
            print(f"  {describe(r)}")
    if not records:
        return 2
    if args.dry_run:
        print(f"dry run: nothing written to {store.path}")
        return 0
    try:
        for r in records:
            store.put(r)
    except (OSError, ValueError) as exc:
        print(f"store write failed ({type(exc).__name__}: {exc}); {store.path}", file=sys.stderr)
        return 3
    print(f"stored {len(records)} record(s) in {store.path}")
    return 0


def bench_command(args: argparse.Namespace) -> int:
    store = MeasurementStore(args.store_dir) if args.store_dir else MeasurementStore()
    existing, note = bridge.load_records(store)
    if note:
        print(f"note: {note}")
    try:
        artifacts = load_manifest()
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    outcomes = calibrate_runs([Path(p) for p in args.bench], existing, lambda doc: catalog_manifest(doc, artifacts),
                              node_id=args.node_id, agnostic=args.model_agnostic, allow_battery=args.allow_battery)
    return _store_and_report(outcomes, args)


# --------------------------------------------------------------------------------------------------- CLI


def _node_id(text: str) -> str:
    if not NODE_ID_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a NodeView id ('local' or 32 lowercase hex characters)")
    return text


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--node-id", type=_node_id, default=HOST_NODE_ID, help="NodeView id the records belong to")
    p.add_argument("--store-dir", type=Path, help="measurement store directory (default: runtime home / capacity)")
    p.add_argument("--allow-battery", action="store_true",
                   help="store a run measured on battery or unknown power (power_state battery or None)")
    p.add_argument("--dry-run", action="store_true", help="derive and print, write nothing to the store")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench", help="coefficients from bench.json runs")
    b.add_argument("bench", nargs="+", help="bench.json paths")
    b.add_argument("--model-agnostic", action="store_true",
                   help="also store each compute coefficient without a model fingerprint, for models never measured")
    _common(b)
    s = sub.add_parser("storage", help="bounded unbuffered read probe of a model's volume")
    s.add_argument("--model", required=True, help="model dir name under ~/.hypercompute/models, a dir or a GGUF")
    s.add_argument("--bytes", type=int, default=storage_probe.MAX_CALIBRATION_BYTES,
                   help="total bytes read over all runs (at most 4 GiB)")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--readers", type=int, default=4)
    s.add_argument("--chunk-bytes", type=int, default=8 * MIB)
    s.add_argument("--results-dir", type=Path, default=RESULTS)
    _common(s)
    args = ap.parse_args(argv)
    if args.command == "storage" and not 0 < args.bytes <= storage_probe.MAX_CALIBRATION_BYTES:
        ap.error(f"--bytes must be 1..{storage_probe.MAX_CALIBRATION_BYTES} (4 GiB)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return bench_command(args) if args.command == "bench" else storage_command(args)
    except (CalibrationError, ValueError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
