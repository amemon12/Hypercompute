"""Capacity benchmark harness: case and workload specs, measured phases, summaries and the results document.

scripts/capacity/bench.py owns the llama-server child and the CLI presets; this module holds the specs, phases,
summaries and document, all testable without a child.
Phases are measured separately against the running child:

  load          process start to the first /health 200, plus the engine's own load log lines
  uncached      prefill with cache_prompt false and a prompt prefix unique to the rep, so nothing can be reused
  prefix reuse  one prompt sent twice, the second time with cache_prompt true; the engine reports cache_n
  decode        the engine's "timings": predicted_ms spans predicted_n - 1 decode steps (the first token comes from
                the prefill logits), so the rate is (predicted_n - 1) / predicted_ms; never a count of SSE chunks

Every rep is kept, failed ones with their error text. Summaries give the median with p10/p90 when a metric has at
least ten values and min/max otherwise, and name which. Numbers taken from a run are measurements; the predicted host
residency used to refuse a run is an estimate and is labelled as one.
"""

from __future__ import annotations

import json
import logging
import math
import re
import secrets
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

import httpx

from .stream_config import StreamOptions
from .types import GIB, PathClass

log = logging.getLogger("hypercompute.capacity.bench")

SCHEMA = "hypercompute.capacity.bench/1"
MIN_MEASURED_REPS = 3
MIN_WARMUP_REPS = 1
RAM_FLOOR_BYTES = 4 * GIB  # free RAM that must remain after the case's predicted residency
DISK_FLOOR_BYTES = 5 * GIB
PERCENTILE_MIN_N = 10  # below this a p10/p90 is just the extreme values, so min/max is reported and named
COMPUTE_PRIOR_BYTES = 512 << 20  # prior: compute buffers and runtime margin; not measured per model
KV_ELEM_BYTES = 2  # llama.cpp's default f16 K and V cache
STREAM_STATS_KEYS = ("bytes_read", "read_ns", "stall_ns", "layer_loads", "sync_loads", "prefetch_loads",
                     "prefetch_hits", "passes", "io_errors", "mmap_fallbacks")
LOAD_LOG_MARKERS = ("common_params_print_info: build", "system_info:", "load_model:", "load_tensors:",
                    "llama_kv_cache:", "sched_reserve:", "hcstream:", "llama_server: model loaded")
DEFAULT_PROMPT = ("You are a careful assistant. Explain, in plain sentences, why a laptop that reads part of a "
                  "language model from its SSD for every token is slower than one that keeps every layer in memory, "
                  "and what changes when a second computer with spare memory joins over a network.")
PROVENANCE = {
    "load, reps, summary": "measured on this host during this run",
    "preflight.predicted": "estimate: GGUF unit sizes plus KV sizing plus a compute prior, used only to refuse a run",
    "ttft_ms": "measured client side over loopback with streaming; no gateway in the path",
    "gpu": "nvidia-smi snapshot after the uncached phase, not a peak",
    "decode_tok_s": "(predicted_n - 1) / predicted_ms from engine timings, the engine's own predicted_per_second",
    "*_read_bytes_per_token": "all reads of the uncached phase (prefill and decode) over predicted_n",
    "hcstream_bytes_per_pass": "hcstream bytes_read over its forward-pass count (decode bytes per token)",
}
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_BUILD_RE = re.compile(r"\bbuild (?P<build>\d+) \((?P<commit>[0-9a-f]+)\)")
_PATH_CLASSES = get_args(PathClass)


class BenchError(ValueError):
    """A spec, response or environment the harness will not measure."""


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _check_strs(name: str, values: object) -> None:
    """A case file is a boundary: a bare string here would otherwise be iterated as characters."""
    if not isinstance(values, tuple) or not all(isinstance(v, str) and v for v in values):
        raise BenchError(f"{name} must be a list of non-empty strings, not {values!r}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_ints(name: str, values: object) -> None:
    if not isinstance(values, tuple) or not all(_is_int(v) and v >= 0 for v in values):
        raise BenchError(f"{name} must be a list of non-negative integers, not {values!r}")


# --------------------------------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class EngineArgs:
    devices: tuple[str, ...] = ()  # engine devices ("CUDA0", "RPC0[host:port]"); empty: every unit on the host CPU
    tensor_split: tuple[int, ...] = ()
    ngl: int | str = 0  # units on non-CPU devices, or "all"
    ctx: int = 4096
    threads: int | None = None
    load_mode: Literal["mmap", "none"] = "mmap"
    flags: tuple[str, ...] = ()  # extra llama-server flags, appended verbatim

    def __post_init__(self) -> None:
        _check_strs("devices", self.devices)
        _check_strs("flags", self.flags)
        _check_ints("tensor_split", self.tensor_split)
        if not _is_int(self.ctx) or self.ctx < 64:
            raise BenchError(f"ctx must be an integer of at least 64, not {self.ctx!r}")
        if not (self.ngl == "all" or (_is_int(self.ngl) and self.ngl >= 0)):
            raise BenchError(f"ngl must be a non-negative integer or 'all', not {self.ngl!r}")
        if self.threads is not None and (not _is_int(self.threads) or self.threads < 1):
            raise BenchError(f"threads must be an integer of at least 1, not {self.threads!r}")
        if self.load_mode not in ("mmap", "none"):
            raise BenchError(f"load_mode must be mmap or none, not {self.load_mode!r}")
        if self.tensor_split and len(self.tensor_split) != len(self.devices):
            raise BenchError("tensor_split needs one entry per device")


@dataclass(frozen=True)
class StreamLayout:
    """hcstream residency of the CPU units: copied once (resident) or read per pass into `slots` buffers."""

    resident_units: tuple[int, ...]
    streamed_units: tuple[int, ...]
    slots: int = 2
    options: StreamOptions = field(default_factory=lambda: StreamOptions(stats_interval_ms=100))

    def __post_init__(self) -> None:
        _check_ints("resident_units", self.resident_units)
        _check_ints("streamed_units", self.streamed_units)
        if set(self.resident_units) & set(self.streamed_units):
            raise BenchError("a unit cannot be both resident and streamed")
        if self.streamed_units and self.slots < 1:
            raise BenchError("streaming needs at least one slot")


@dataclass(frozen=True)
class Case:
    label: str
    model_paths: tuple[str, ...]  # every shard, first shard first
    engine: EngineArgs = field(default_factory=EngineArgs)
    stream: StreamLayout | None = None
    rpc_endpoints: tuple[str, ...] = ()
    rpc_path_class: PathClass = "local"

    def __post_init__(self) -> None:
        if not _LABEL_RE.match(self.label):
            raise BenchError(f"label {self.label!r} must be 1-80 characters of letters, digits, '.', '_' or '-'")
        if not self.model_paths:
            raise BenchError("a case needs one or more model paths")
        _check_strs("model_paths", self.model_paths)
        _check_strs("rpc_endpoints", self.rpc_endpoints)
        if self.rpc_path_class not in _PATH_CLASSES:
            raise BenchError(f"rpc_path_class must be one of {', '.join(_PATH_CLASSES)}")

    @property
    def uses_cuda(self) -> bool:
        return any(d.upper().startswith("CUDA") for d in self.engine.devices)


@dataclass(frozen=True)
class BenchWorkload:
    prompt: str = DEFAULT_PROMPT
    prompt_tokens: int | None = None  # token target: the prompt is tokenized by the engine and cut to this length
    output_tokens: int = 128
    reps: int = 3  # measured reps, after `warmup` unmeasured ones
    warmup: int = 1
    reuse_output_tokens: int = 1
    seed: int = 7
    temperature: float = 0.0
    top_k: int = 1
    request_timeout_s: float = 1800.0
    load_timeout_s: float = 900.0

    def __post_init__(self) -> None:
        if self.reps < MIN_MEASURED_REPS:
            raise BenchError(f"at least {MIN_MEASURED_REPS} measured reps are required, not {self.reps}")
        if self.warmup < MIN_WARMUP_REPS:
            raise BenchError(f"at least {MIN_WARMUP_REPS} warmup rep is required, not {self.warmup}")
        if self.output_tokens < 2:  # the first token comes from the prefill; a decode rate needs a second one
            raise BenchError(f"output_tokens must be at least 2 for a decode rate, not {self.output_tokens}")
        if self.reuse_output_tokens < 1:
            raise BenchError("reuse_output_tokens must be at least 1")
        if self.prompt_tokens is not None and self.prompt_tokens < 1:
            raise BenchError("prompt_tokens must be at least 1")
        if self.temperature != 0.0 or self.top_k != 1:
            raise BenchError("sampling is fixed for benchmarks: temperature 0, top_k 1")
        if not self.prompt.strip():
            raise BenchError("the prompt is empty")


def _tuplify(data: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {k: tuple(v) if k in keys and isinstance(v, list) else v for k, v in data.items()}


def case_from_dict(data: object) -> Case:
    """A Case from its JSON form (the shape `dataclasses.asdict(case)` writes). Raises BenchError otherwise."""
    if not isinstance(data, Mapping):
        raise BenchError("a case must be a JSON object")
    try:
        engine = EngineArgs(**_tuplify(data.get("engine") or {}, ("devices", "tensor_split", "flags")))
        stream = data.get("stream")
        if stream is not None:
            raw = _tuplify(stream, ("resident_units", "streamed_units"))
            raw["options"] = StreamOptions(**raw.get("options", {"stats_interval_ms": 100}))
            stream = StreamLayout(**raw)
        rest = _tuplify({k: v for k, v in data.items() if k not in ("engine", "stream")},
                        ("model_paths", "rpc_endpoints"))
        return Case(engine=engine, stream=stream, **rest)
    except (TypeError, AttributeError) as exc:
        raise BenchError(f"bad case {data.get('label')!r}: {exc}") from exc


def build_argv(case: Case, exe: Path, port: int) -> list[str]:
    e = case.engine
    argv = [str(exe), "-m", case.model_paths[0], "--load-mode", e.load_mode, *e.flags]
    if case.rpc_endpoints:
        argv += ["--rpc", ",".join(case.rpc_endpoints)]
    if e.devices:
        argv += ["--device", ",".join(e.devices), "--split-mode", "layer"]
        if e.tensor_split:
            argv += ["--tensor-split", ",".join(str(n) for n in e.tensor_split)]
        argv += ["-ngl", str(e.ngl)]
    else:
        argv += ["--device", "none", "-ngl", "0"]
    argv += ["--fit", "off", "-c", str(e.ctx), "-np", "1"]
    if e.threads:
        argv += ["-t", str(e.threads)]
    return argv + ["--host", "127.0.0.1", "--port", str(port), "--no-ui", "-lv", "4", "--log-colors", "off"]


# --------------------------------------------------------------------------------------------------- preflight


@dataclass(frozen=True)
class ModelFacts:
    n_layer: int
    unit_bytes: tuple[int, ...]  # aligned buffer bytes per unit (stream_config layouts): blocks, then the head
    token_embd_bytes: int
    tied_embeddings: bool
    kv_bytes_per_layer_token: int
    file_bytes: int


@dataclass(frozen=True)
class Residency:
    total_bytes: int
    parts: tuple[tuple[str, int], ...]
    provenance: str = "estimate"


def cpu_units(case: Case, n_layer: int) -> tuple[int, ...]:
    """Units the host CPU computes: everything without devices, else the first n_layer + 1 - ngl."""
    if not case.engine.devices:
        return tuple(range(n_layer + 1))
    ngl = case.engine.ngl
    if ngl == "all" or int(ngl) >= n_layer + 1:
        return ()
    return tuple(range(n_layer + 1 - int(ngl)))


def predict_host_bytes(case: Case, facts: ModelFacts) -> Residency:
    """Host RAM the case will hold: CPU unit weights (or the hcstream bound), token_embd, CPU-layer KV, a prior."""
    all_cpu = cpu_units(case, facts.n_layer)
    on_cpu = all_cpu
    parts: list[tuple[str, int]] = []
    if case.stream is not None:
        s = case.stream
        managed = set(s.resident_units) | set(s.streamed_units)
        if not managed <= set(on_cpu):
            raise BenchError("hcstream units must be CPU units of the case")
        slot = max((facts.unit_bytes[u] for u in s.streamed_units), default=0)
        parts.append(("hcstream resident", sum(facts.unit_bytes[u] for u in s.resident_units)))
        parts.append(("hcstream slots", (s.slots if s.streamed_units else 0) * slot))
        on_cpu = tuple(u for u in on_cpu if u not in managed)
    parts.append(("cpu unit weights", sum(facts.unit_bytes[u] for u in on_cpu)))
    if not (facts.tied_embeddings and facts.n_layer in all_cpu):  # tied: token_embd is already inside the head unit
        parts.append(("token_embd", facts.token_embd_bytes))
    cpu_blocks = sum(1 for u in all_cpu if u < facts.n_layer)
    parts.append(("kv cache (cpu blocks)", cpu_blocks * case.engine.ctx * facts.kv_bytes_per_layer_token))
    parts.append(("compute prior", COMPUTE_PRIOR_BYTES))
    return Residency(total_bytes=sum(b for _, b in parts), parts=tuple(parts))


@dataclass(frozen=True)
class Preflight:
    port: int
    port_listener: str | None  # "name (pid N)" of whatever already listens on the port
    available_ram_bytes: int
    predicted: Residency
    free_disk_bytes: int
    disk_path: str


def refusals(p: Preflight) -> tuple[str, ...]:
    """Every reason not to start the case; empty when it may run."""
    out: list[str] = []
    if p.port_listener is not None:
        what = "another llama-server" if "llama-server" in p.port_listener.lower() else p.port_listener
        out.append(f"{what} is already serving on 127.0.0.1:{p.port}")
    left = p.available_ram_bytes - p.predicted.total_bytes
    if left < RAM_FLOOR_BYTES:
        out.append(f"available RAM {p.available_ram_bytes / GIB:.2f} GiB minus the predicted "
                   f"{p.predicted.total_bytes / GIB:.2f} GiB (estimate) leaves {left / GIB:.2f} GiB, "
                   f"below the {RAM_FLOOR_BYTES / GIB:.0f} GiB floor")
    if p.free_disk_bytes < DISK_FLOOR_BYTES:
        out.append(f"free disk at {p.disk_path} is {p.free_disk_bytes / GIB:.2f} GiB, below the "
                   f"{DISK_FLOOR_BYTES / GIB:.0f} GiB floor")
    return tuple(out)


# --------------------------------------------------------------------------------------------------- observation


@dataclass(frozen=True)
class Counters:
    t: float  # seconds on the observer's clock (the memory sampler's)
    proc_read_bytes: int  # ReadFile bytes of the child: includes unbuffered hcstream reads, not mmap page faults
    sys_disk_read_bytes: int  # physical reads by every process on the host
    page_faults: int
    working_set: int
    private: int
    stream_stats: Mapping[str, Any] | None = None
    error: str | None = None  # the process counters could not be read; their zero placeholders are not values
    stats_error: str | None = None  # hcstream stats missing or stale (stream_stats is then None)


class Observer(Protocol):
    """What a rep needs to know about the child: I/O and memory counters, window peaks, GPU memory, power."""

    def counters(self, fresh_after_unix_ms: int | None = None) -> Counters: ...
    def peaks(self, t0: float, t1: float) -> tuple[int, int]: ...  # (working set, private) peaks in [t0, t1]
    def gpu_memory(self) -> dict[str, Any] | None: ...
    def power(self) -> dict[str, Any]: ...


def _delta(before: int, after: int, valid: bool) -> int | None:
    return after - before if valid else None


def _stream_delta(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> dict[str, int] | None:
    if not before or not after:
        return None
    try:
        return {k: int(after[k]) - int(before[k]) for k in STREAM_STATS_KEYS if k in after and k in before}
    except (TypeError, ValueError):
        return None


def window_io(before: Counters, after: Counters, peaks: tuple[int, int]) -> dict[str, Any]:
    """Counter deltas over one phase. A delta is None when either end's counters could not be read: a zero
    placeholder minus a cumulative counter is not a measurement."""
    valid = before.error is None and after.error is None
    out: dict[str, Any] = {
        "seconds": round(after.t - before.t, 4),
        "proc_read_bytes": _delta(before.proc_read_bytes, after.proc_read_bytes, valid),
        "sys_disk_read_bytes": _delta(before.sys_disk_read_bytes, after.sys_disk_read_bytes, valid),
        "page_faults": _delta(before.page_faults, after.page_faults, valid),
        "working_set_peak": max(peaks[0], before.working_set, after.working_set),
        "private_peak": max(peaks[1], before.private, after.private),
    }
    hcstream = _stream_delta(before.stream_stats, after.stream_stats)
    if hcstream is not None:
        out["hcstream"] = hcstream
    errors = [e for c in (before, after) for e in (c.error, c.stats_error) if e]
    if errors:
        out["errors"] = errors
    return out


# --------------------------------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class Timings:
    """llama-server's own `timings` for one request."""

    prompt_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float
    cache_n: int

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prompt_n / self.prompt_ms * 1000 if self.prompt_ms > 0 else None

    @property
    def decode_steps(self) -> int:
        """Decode steps inside predicted_ms. llama-server starts that clock when the first token is sampled from the
        prefill logits, so predicted_ms spans predicted_n - 1 steps (its own n_gen_steps / predicted_per_second)."""
        return max(self.predicted_n - 1, 0)

    @property
    def decode_tok_s(self) -> float | None:
        return self.decode_steps / self.predicted_ms * 1000 if self.predicted_ms > 0 and self.decode_steps else None

    @property
    def decode_ms_per_token(self) -> float | None:
        return self.predicted_ms / self.decode_steps if self.decode_steps else None


def parse_timings(payload: Mapping[str, Any]) -> Timings:
    t = payload.get("timings")
    if not isinstance(t, Mapping):
        raise BenchError("the response carried no engine timings")
    try:
        return Timings(prompt_n=int(t["prompt_n"]), prompt_ms=float(t["prompt_ms"]),
                       predicted_n=int(t["predicted_n"]), predicted_ms=float(t["predicted_ms"]),
                       cache_n=int(t.get("cache_n", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchError(f"engine timings are incomplete: {_err(exc)}") from exc


def completion_body(prompt: str | list[int], n_predict: int, workload: BenchWorkload, *, cache_prompt: bool,
                    stream: bool) -> dict[str, Any]:
    return {"prompt": prompt, "n_predict": n_predict, "temperature": workload.temperature, "top_k": workload.top_k,
            "seed": workload.seed, "cache_prompt": cache_prompt, "stream": stream, "ignore_eos": True}


def read_sse(lines: Iterable[str], t0: float, clock: Callable[[], float]) -> tuple[dict[str, Any], float | None]:
    """(final payload, seconds to the first chunk carrying text) from a llama-server /completion SSE stream."""
    ttft: float | None = None
    final: dict[str, Any] | None = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise BenchError("an SSE chunk is not a JSON object")
        if "error" in payload:
            raise BenchError(f"engine error: {payload['error']}")
        if ttft is None and (payload.get("content") or payload.get("tokens")):
            ttft = clock() - t0
        if payload.get("stop"):
            final = payload
    if final is None:
        raise BenchError("the stream ended without a final chunk")
    return final, ttft


def _post(client: httpx.Client, body: Mapping[str, Any], timeout: float, t0: float,
          clock: Callable[[], float]) -> tuple[dict[str, Any], float | None]:
    if body.get("stream"):
        with client.stream("POST", "/completion", json=body, timeout=timeout) as r:
            if r.status_code != 200:
                raise BenchError(f"HTTP {r.status_code}: {r.read().decode('utf-8', 'replace')[:300]}")
            return read_sse(r.iter_lines(), t0, clock)
    r = client.post("/completion", json=body, timeout=timeout)
    if r.status_code != 200:
        raise BenchError(f"HTTP {r.status_code}: {r.text[:300]}")
    payload = r.json()
    if not isinstance(payload, dict):
        raise BenchError("the response is not a JSON object")
    return payload, None


@dataclass(frozen=True)
class PhaseResult:
    phase: str  # "uncached", "prefix_prime", "prefix_reuse"
    ok: bool
    error: str | None = None
    wall_ms: float | None = None
    ttft_ms: float | None = None
    timings: Timings | None = None
    cache_prompt: bool | None = None
    prompt_head: str = ""  # first characters (or token ids) of the prompt, to show prefixes differ per rep
    io: dict[str, Any] = field(default_factory=dict)


def run_phase(client: httpx.Client, phase: str, body: Mapping[str, Any], observer: Observer, *, timeout: float,
              clock: Callable[[], float] = time.perf_counter) -> PhaseResult:
    before = observer.counters(fresh_after_unix_ms=int(time.time() * 1000))  # no stale snapshot from before the gap
    t0 = clock()
    timings, ttft, error = None, None, None
    try:
        payload, ttft = _post(client, body, timeout, t0, clock)
        timings = parse_timings(payload)
    except Exception as exc:  # recorded as the phase's error text (httpx.StreamError is not an HTTPError, for one)
        error = _err(exc)
    wall = clock() - t0
    after = observer.counters(fresh_after_unix_ms=int(time.time() * 1000))
    return PhaseResult(phase=phase, ok=error is None, error=error, wall_ms=round(wall * 1000, 3),
                       ttft_ms=round(ttft * 1000, 3) if ttft is not None else None, timings=timings,
                       cache_prompt=bool(body.get("cache_prompt")), prompt_head=str(body.get("prompt"))[:64],
                       io=window_io(before, after, observer.peaks(before.t, after.t)))


def tokenize(client: httpx.Client, text: str, timeout: float) -> list[int]:
    r = client.post("/tokenize", json={"content": text}, timeout=timeout)
    if r.status_code != 200:
        raise BenchError(f"/tokenize HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise BenchError("/tokenize returned no token list")
    return tokens


def prompt_for(client: httpx.Client, workload: BenchWorkload, tag: str) -> str | list[int]:
    """The tagged prompt text, or exactly `prompt_tokens` engine tokens of it (repeated as needed)."""
    text = tag + workload.prompt
    if workload.prompt_tokens is None:
        return text
    once = tokenize(client, workload.prompt, workload.request_timeout_s)  # per copy: the tag is prefixed only once
    if not once:
        raise BenchError("the prompt tokenizes to nothing")
    copies = math.ceil(workload.prompt_tokens / len(once)) + 1
    tokens = tokenize(client, tag + " ".join([workload.prompt] * copies), workload.request_timeout_s)
    if len(tokens) < workload.prompt_tokens:
        raise BenchError(f"could not build a {workload.prompt_tokens}-token prompt")
    return tokens[: workload.prompt_tokens]


def rep_tag(label: str, index: int, nonce: str, phase: str) -> str:
    """Unique per rep and phase; fixed-width digits keep the prompt's token count the same for every rep. The
    unique part comes first, so the 64-character prompt_head shows it whatever the label's length."""
    return f"[rep {index:04d} {nonce} {phase} bench {label}] "


# --------------------------------------------------------------------------------------------------- reps


@dataclass(frozen=True)
class RepResult:
    index: int
    warmup: bool
    ok: bool
    phases: tuple[PhaseResult, ...]
    gpu: dict[str, Any] | None
    power: dict[str, Any]
    rpc: dict[str, Any] | None
    started_at: str

    def phase(self, name: str) -> PhaseResult | None:
        return next((p for p in self.phases if p.phase == name), None)


def check_uncached(res: PhaseResult, wanted_tokens: int) -> PhaseResult:
    """An uncached phase that reused cache or decoded fewer tokens than asked is a failed measurement."""
    if not res.ok or res.timings is None:
        return res
    problems = []
    if res.timings.cache_n:
        problems.append(f"reused {res.timings.cache_n} cached prompt tokens, so this is not an uncached prefill")
    if res.timings.predicted_n < wanted_tokens:
        problems.append(f"decoded {res.timings.predicted_n} of {wanted_tokens} tokens")
    return replace(res, ok=False, error="; ".join(problems)) if problems else res


def _failed(phase: str, error: str) -> PhaseResult:
    return PhaseResult(phase=phase, ok=False, error=error)


def _uncached(client: httpx.Client, case: Case, workload: BenchWorkload, index: int, nonce: str,
              observer: Observer) -> PhaseResult:
    try:
        prompt = prompt_for(client, workload, rep_tag(case.label, index, nonce, "uncached"))
    except (httpx.HTTPError, BenchError, ValueError) as exc:
        return _failed("uncached", f"building the prompt: {_err(exc)}")
    body = completion_body(prompt, workload.output_tokens, workload, cache_prompt=False, stream=True)
    res = run_phase(client, "uncached", body, observer, timeout=workload.request_timeout_s)
    return check_uncached(res, workload.output_tokens)


def _reuse(client: httpx.Client, case: Case, workload: BenchWorkload, index: int, nonce: str,
           observer: Observer) -> tuple[PhaseResult, PhaseResult]:
    try:
        prompt = prompt_for(client, workload, rep_tag(case.label, index, nonce, "reuse"))
    except (httpx.HTTPError, BenchError, ValueError) as exc:
        error = f"building the prompt: {_err(exc)}"
        return _failed("prefix_prime", error), _failed("prefix_reuse", error)
    n, t = workload.reuse_output_tokens, workload.request_timeout_s
    prime = run_phase(client, "prefix_prime", completion_body(prompt, n, workload, cache_prompt=False, stream=False),
                      observer, timeout=t)
    if not prime.ok:
        return prime, _failed("prefix_reuse", f"not run: the priming request failed ({prime.error})")
    reuse = run_phase(client, "prefix_reuse", completion_body(prompt, n, workload, cache_prompt=True, stream=False),
                      observer, timeout=t)
    if reuse.ok and reuse.timings is not None and reuse.timings.cache_n == 0:
        reuse = replace(reuse, ok=False, error="cache_prompt true reused nothing (cache_n 0)")
    return prime, reuse


def run_rep(client: httpx.Client, case: Case, workload: BenchWorkload, index: int, *, warmup: bool,
            observer: Observer, nonce: str | None = None) -> RepResult:
    """One rep: uncached prefill + decode, then the prefix reuse pair. A failure is recorded, never raised."""
    started = datetime.now(UTC).isoformat()
    nonce = nonce or f"{secrets.randbelow(10**8):08d}"
    uncached = _uncached(client, case, workload, index, nonce, observer)
    gpu = observer.gpu_memory()
    prime, reuse = _reuse(client, case, workload, index, nonce, observer)
    phases = (uncached, prime, reuse)
    rpc = {"endpoints": list(case.rpc_endpoints), "path_class": case.rpc_path_class} if case.rpc_endpoints else None
    return RepResult(index=index, warmup=warmup, ok=all(p.ok for p in phases), phases=phases, gpu=gpu,
                     power=observer.power(), rpc=rpc, started_at=started)


# --------------------------------------------------------------------------------------------------- summaries


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]) of a non-empty sequence."""
    if not values:
        raise BenchError("percentile of no values")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def spread(values: Iterable[float | None]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {"n": 0}
    if len(vals) >= PERCENTILE_MIN_N:
        low, high, kind = percentile(vals, 10), percentile(vals, 90), "p10/p90"
    else:
        low, high, kind = vals[0], vals[-1], "min/max"
    return {"n": len(vals), "median": statistics.median(vals), "low": low, "high": high, "spread": kind}


def _per_token(num: float | None, tokens: int | None) -> float | None:
    return num / tokens if num is not None and tokens else None


def rep_metrics(rep: RepResult) -> dict[str, float | None]:
    """Per-rep values from successful phases only; a failed phase contributes nothing.

    The *_per_token read figures divide every byte read during the uncached phase (its prefill passes and its decode
    passes) by the decoded token count, so a long prompt raises them. hcstream_bytes_per_pass is bytes_read over the
    extension's own forward-pass count, which is the per-token figure for decode.
    """
    out: dict[str, float | None] = {}
    u = rep.phase("uncached")
    if u is not None and u.ok and u.timings is not None:
        t = u.timings
        hc = u.io.get("hcstream") or {}
        out["hcstream_bytes_per_pass"] = _per_token(hc.get("bytes_read"), hc.get("passes"))
        out.update(prefill_tokens=t.prompt_n, prefill_ms=t.prompt_ms, prefill_tok_s=t.prefill_tok_s,
                   ttft_ms=u.ttft_ms, decode_tokens=t.predicted_n, decode_steps=t.decode_steps,
                   decode_ms=t.predicted_ms, decode_tok_s=t.decode_tok_s, decode_ms_per_token=t.decode_ms_per_token,
                   uncached_wall_ms=u.wall_ms, proc_read_bytes=u.io.get("proc_read_bytes"),
                   sys_disk_read_bytes=u.io.get("sys_disk_read_bytes"),
                   proc_read_bytes_per_token=_per_token(u.io.get("proc_read_bytes"), t.predicted_n),
                   sys_disk_read_bytes_per_token=_per_token(u.io.get("sys_disk_read_bytes"), t.predicted_n),
                   hcstream_bytes_read=hc.get("bytes_read"),
                   hcstream_bytes_per_token=_per_token(hc.get("bytes_read"), t.predicted_n),
                   page_faults=u.io.get("page_faults"))
    r = rep.phase("prefix_reuse")
    if r is not None and r.ok and r.timings is not None:
        out.update(reuse_cache_n=r.timings.cache_n, reuse_prompt_n=r.timings.prompt_n,
                   reuse_prompt_ms=r.timings.prompt_ms, reuse_wall_ms=r.wall_ms)
    ios = [p.io for p in rep.phases if p.ok and p.io]
    if ios:
        out.update(working_set_peak=max(i.get("working_set_peak", 0) for i in ios),
                   private_peak=max(i.get("private_peak", 0) for i in ios))
    return out


def failures(reps: Iterable[RepResult]) -> list[dict[str, Any]]:
    """Every failed phase of every rep, warmups included, with its error text."""
    return [{"rep": r.index, "warmup": r.warmup, "phase": p.phase, "error": p.error}
            for r in reps for p in r.phases if not p.ok]


def summarize(reps: Sequence[RepResult]) -> dict[str, Any]:
    measured = [r for r in reps if not r.warmup]
    series: dict[str, list[float | None]] = {}
    for rep in measured:
        for key, value in rep_metrics(rep).items():
            series.setdefault(key, []).append(value)
    return {"n_reps": len(measured), "n_warmup": len(reps) - len(measured),
            "n_reps_ok": sum(1 for r in measured if r.ok), "metrics": {k: spread(v) for k, v in series.items()},
            "failures": failures(reps)}


# --------------------------------------------------------------------------------------------------- load


@dataclass(frozen=True)
class LoadResult:
    ok: bool
    seconds: float | None  # process start to the first /health 200
    error: str | None
    log_lines: tuple[str, ...] = ()
    build: dict[str, Any] | None = None
    memory: dict[str, Any] = field(default_factory=dict)
    log_tail: tuple[str, ...] = ()  # last engine log lines, kept when the load failed


def wait_healthy(client: httpx.Client, alive: Callable[[], bool], timeout_s: float, t_start: float, *,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
                 poll_s: float = 0.1) -> tuple[bool, float, str | None]:
    """(healthy, seconds since t_start, error) polling /health until 200, the process exits, or the timeout."""
    while True:
        if not alive():
            return False, clock() - t_start, "llama-server exited during load"
        try:
            if client.get("/health", timeout=2.0).status_code == 200:
                return True, clock() - t_start, None
        except httpx.HTTPError as exc:
            log.debug("health poll: %s", exc)
        if clock() - t_start > timeout_s:
            return False, clock() - t_start, f"llama-server was not healthy after {timeout_s:.0f}s"
        sleep(poll_s)


def load_log_lines(text: str, limit: int = 120) -> tuple[str, ...]:
    """Load lines of the engine log; past `limit` the middle is dropped so the last lines ("model loaded") stay."""
    lines = tuple(line.strip() for line in text.splitlines() if any(m in line for m in LOAD_LOG_MARKERS))
    if len(lines) <= limit:
        return lines
    head = limit // 2
    tail = limit - head - 1
    return (*lines[:head], f"... {len(lines) - head - tail} load lines omitted ...", *lines[-tail:])


def log_tail(text: str, n: int = 20) -> tuple[str, ...]:
    """The last non-empty lines of the engine log: the reason when a load fails."""
    return tuple(line.strip() for line in text.splitlines() if line.strip())[-n:]


def parse_build(text: str) -> dict[str, Any] | None:
    m = _BUILD_RE.search(text)
    return {"build": int(m["build"]), "commit": m["commit"], "build_source": "llama-server log"} if m else None


# --------------------------------------------------------------------------------------------------- document


def run_status(load: LoadResult | None, reps: Sequence[RepResult], expected_reps: int) -> str:
    """ok: every rep (warmups too) succeeded and all expected measured reps ran; partial: some measured rep did."""
    if load is None or not load.ok:
        return "failed"
    measured = [r for r in reps if not r.warmup]
    if len(measured) >= expected_reps and all(r.ok for r in reps):
        return "ok"
    return "partial" if any(r.ok for r in measured) else "failed"


def results_document(*, case: Case, workload: BenchWorkload, status: str, started_at: str, host: Mapping[str, Any],
                     model: Mapping[str, Any], preflight: Preflight | None, refused: Sequence[str] = (),
                     engine: Mapping[str, Any] | None = None, launch: Mapping[str, Any] | None = None,
                     load: LoadResult | None = None, reps: Sequence[RepResult] = (),
                     stop: Mapping[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "case": asdict(case),
        "workload": asdict(workload),
        "host": dict(host),
        "llama_cpp": dict(engine or {}),
        "model": dict(model),
        "preflight": asdict(preflight) if preflight else None,
        "refused": list(refused),
        "launch": dict(launch or {}),
        "load": asdict(load) if load else None,
        "reps": [asdict(r) for r in reps],
        "summary": summarize(reps),
        "stop": dict(stop or {}),
        "error": error,
        "provenance": PROVENANCE,
    }
