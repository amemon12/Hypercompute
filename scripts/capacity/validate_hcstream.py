"""Validation matrix for the hcstream native extension: real CPU-only llama.cpp children on the 0.5B model.

Every token case starts its own llama-server child (the stock engine for the baseline, the overlay engine for the
rest), asks for the same greedy completion (n_predict 24, temperature 0, top_k 1, seed 7, ignore_eos) and compares
token ids with the stock run. Fault cases check that the extension exits with code 86 and says why, or falls back
to the file mapping when the config says `on_io_error mmap`. The teardown case runs llama-completion from the
overlay to a normal exit and reads the final stats file.

  uv run python scripts/capacity/validate_hcstream.py
  uv run python scripts/capacity/validate_hcstream.py --only c-stream-k2,j-malformed

Children never see the GPU (--device none, -ngl 0, CUDA_VISIBLE_DEVICES=-1), never touch the live gateway, and are
killed on every exit path. Expected SSD reading is estimated from the stream plans before anything starts and the run
is refused above --max-ssd-gib. Output: results/capacity/<UTC stamp>-validate/summary.json, plus each child's
config, stats file and log next to it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import httpx

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import canary  # noqa: E402  (also puts backend/ on sys.path)
from hypercompute import config, engine  # noqa: E402
from hypercompute.capacity import overlay, probe  # noqa: E402
from hypercompute.capacity.stream_config import (  # noqa: E402
    ALIGN,
    StreamOptions,
    StreamPlan,
    block_count,
    layer_layouts,
    render_config,
    stream_plan,
)
from hypercompute.gguf import TensorInfo, read_tensor_table  # noqa: E402
from hypercompute.netutil import free_loopback_port  # noqa: E402

ROOT = canary.ROOT
GIB = 1 << 30
MIB = 1 << 20
FATAL_EXIT = 86  # hc_die() in hcstream.c
ERROR_READ_FAULT = 30  # winerror.h; hc_read_chunk reports an injected read fault with this code
FATAL_READ = re.compile(r"hcstream: FATAL: unbuffered read failed for layer (\d+) "
                        r"\(win32 error (\d+) at offset (-?\d+)\)")
MODEL_DIR = "qwen2.5-0.5b-instruct-q4_k_m"
FAULT_AFTER_BYTES = 50 * MIB
SLOW_READ_US = 2000
TEARDOWN_TOKENS = 8
EXTRA_PASSES = 2  # the loader's warmup run and the prompt batch, on top of one pass per generated token
STATS_SETTLE_S = 1.2  # > 2 stats intervals, so the stats file covers the finished completion
CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

Expect = Literal["tokens", "fatal", "teardown"]


class SetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    expect: Expect  # tokens: identical to stock; fatal: exit 86 with a reason; teardown: clean llama-completion exit
    engine: Literal["stock", "overlay"] = "overlay"
    resident: str = ""  # layer spec as in canary.py, 'head' is the output unit
    stream: str = ""
    slots: int = 2
    options: StreamOptions = StreamOptions()
    malformed: bool = False  # write the config without its tail and 'end' line
    completions: int = 1


@dataclass(frozen=True)
class Settings:
    tokens: int = 24
    ctx: int = 1024
    threads: int = 8
    load_timeout_s: float = 180.0
    request_timeout_s: float = 600.0
    exit_timeout_s: float = 120.0
    min_available_gib: float = 1.0
    prompt: str = canary.PROMPT


@dataclass(frozen=True)
class Target:
    model: Path
    shards: tuple[Path, ...]
    table: tuple[TensorInfo, ...]
    n_layer: int
    stock_bin: Path
    overlay_bin: Path
    proxy_sha256: str


@dataclass(frozen=True)
class Outcome:
    served: bool  # /health answered 200 (llama-server cases)
    exit_code: int | None  # the child's own exit code; None when it was still running and had to be stopped
    completions: tuple[dict, ...]
    stats: dict | None
    log_text: str
    seconds: float
    memory: dict
    error: str | None
    resident_peak: int = 0  # largest resident_bytes in a live stats snapshot (polled only for the teardown case)
    slot_peak: int = 0  # largest slot_bytes_committed in a live stats snapshot (same polling)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class CaseResult:
    name: str
    description: str
    expect: Expect
    passed: bool
    checks: tuple[Check, ...]
    exit_code: int | None
    token_match: bool | None
    tokens: tuple[int, ...]
    stats: dict | None
    plan: dict | None
    ssd_estimate_bytes: int
    seconds: float
    memory: dict
    error: str | None
    log: str | None
    hcstream_log: tuple[str, ...] = ()  # the extension's own stderr lines (attach, detach, FATAL reason)


# --------------------------------------------------------------------------------------------------- setup


def default_model() -> Path:
    return config.paths().models / MODEL_DIR


def missing_requirements(model: Path) -> list[str]:
    """Why the native matrix cannot run on this host; empty when it can."""
    missing: list[str] = []
    if os.name != "nt":
        missing.append("the hcstream extension is Windows-only")
    if overlay.find_proxy_dll() is None:
        missing.append(f"proxy DLL not built ({overlay.REPO_DLL})")
    if engine.find_installed() is None:
        missing.append("no installed llama.cpp engine")
    if not (model.is_file() or (model.is_dir() and any(model.glob("*.gguf")))):
        missing.append(f"model not found: {model}")
    return missing


def prepare(model: Path) -> Target:
    missing = missing_requirements(model)
    if missing:
        raise SetupError("; ".join(missing))
    stock = engine.find_installed()
    dll = overlay.find_proxy_dll()
    if stock is None or dll is None:  # checked above; keeps the type checker honest
        raise SetupError("engine or proxy DLL disappeared")
    ov = overlay.ensure_overlay(stock.bin_dir, dll)
    shards = tuple(canary.shards_of(model))
    table = tuple(read_tensor_table(shards))
    return Target(model=model, shards=shards, table=table, n_layer=block_count(table), stock_bin=stock.bin_dir,
                  overlay_bin=ov.bin_dir, proxy_sha256=ov.proxy_sha256)


def build_matrix(n_layer: int) -> tuple[Case, ...]:
    blocks = f"0-{n_layer - 1}"
    half = n_layer // 2
    lower, upper = f"0-{half - 1}", f"{half}-{n_layer - 1}"
    fault = StreamOptions(fault_read_error_after_bytes=FAULT_AFTER_BYTES)
    return (
        Case("a-stock", "stock engine, file-mapped weights, two completions", "tokens", engine="stock", completions=2),
        Case("b-stream-k1", f"blocks {blocks} streamed, 1 slot", "tokens", stream=blocks, slots=1),
        Case("c-stream-k2", f"blocks {blocks} streamed, 2 slots", "tokens", stream=blocks, slots=2),
        Case("d-stream-k3", f"blocks {blocks} streamed, 3 slots", "tokens", stream=blocks, slots=3),
        Case("e-mixed-k2", f"head and blocks {lower} resident, {upper} streamed, 2 slots", "tokens",
             resident=f"head,{lower}", stream=upper, slots=2),
        Case("f-noprefetch", "overlay with no managed layers: mapping prefetch off only", "tokens"),
        Case("g-fault-abort", "read error after 50 MiB, on_io_error abort", "fatal", stream=blocks, options=fault),
        Case("h-fault-mmap", "read error after 50 MiB, on_io_error mmap", "tokens", stream=blocks,
             options=replace(fault, on_io_error="mmap")),
        Case("i-slow-disk", f"{SLOW_READ_US} us delay per read chunk, 2 slots", "tokens", stream=blocks,
             options=StreamOptions(fault_read_delay_us=SLOW_READ_US)),
        Case("j-malformed", "config truncated before its 'end' line", "fatal", stream=blocks, malformed=True),
        Case("k-teardown", f"llama-completion -n {TEARDOWN_TOKENS} from the overlay, head and {lower} resident",
             "teardown", resident=f"head,{lower}", stream=upper, completions=0,
             options=StreamOptions(stats_interval_ms=100)),  # live snapshots must catch the resident copy
    )


def plan_for(target: Target, case: Case) -> StreamPlan | None:
    if case.engine == "stock":
        return None
    resident = canary.parse_layers(case.resident, target.n_layer)
    stream = canary.parse_layers(case.stream, target.n_layer)
    return stream_plan(layer_layouts(target.table), resident, stream, case.slots)


def estimate_ssd_bytes(plan: StreamPlan | None, case: Case, settings: Settings) -> int:
    """Upper estimate (not a measurement) of unbuffered SSD reads: resident copies plus one plan pass per pass."""
    if plan is None or case.malformed:
        return 0
    tokens = TEARDOWN_TOKENS if case.expect == "teardown" else settings.tokens
    passes = (tokens + EXTRA_PASSES) * max(case.completions, 1)
    total = plan.resident_bytes + passes * plan.stream_bytes_per_pass
    opts = case.options
    if opts.fault_read_error_after_bytes:
        # Every read after the threshold fails before touching the disk. Readers test it before each chunk against
        # bytes_read, which grows only when a chunk finishes, so each reader can have one whole chunk in flight.
        total = min(total, opts.fault_read_error_after_bytes + opts.readers * opts.chunk_bytes)
    return total


def truncate_config(text: str) -> str:
    """A config cut a third of the way from its end: whole lines, no 'end' line."""
    lines = text.splitlines()
    if not lines or lines[-1] != "end":
        raise ValueError("expected a rendered config ending in 'end'")
    keep = lines[: max(1, (len(lines) - 1) * 2 // 3)]
    return "\n".join(keep) + "\n"


def write_config(target: Target, case: Case, plan: StreamPlan, out_dir: Path) -> tuple[Path, Path]:
    stats = out_dir / f"{case.name}.stats.json"
    text = render_config(target.shards, target.table, plan, stats, case.options)
    cfg = out_dir / f"{case.name}.cfg"
    cfg.write_text(truncate_config(text) if case.malformed else text, encoding="utf-8")
    return cfg, stats


# --------------------------------------------------------------------------------------------------- children


def _common_flags(s: Settings) -> list[str]:
    return ["--load-mode", "mmap", "--no-repack", "--no-op-offload", "--device", "none", "-ngl", "0", "--fit", "off",
            "-c", str(s.ctx), "-t", str(s.threads), "--log-colors", "off"]


def server_argv(bin_dir: Path, shard: Path, port: int, s: Settings) -> list[str]:
    return [str(bin_dir / engine.exe_name("llama-server")), "-m", str(shard), *_common_flags(s), "--cache-ram", "0",
            "-np", "1", "--host", "127.0.0.1", "--port", str(port), "--no-ui"]


def completion_argv(bin_dir: Path, shard: Path, s: Settings) -> list[str]:
    # llama-completion is not in engine.ALLOWED_BINARIES (the product never runs it); this script is Windows-only.
    return [str(bin_dir / "llama-completion.exe"), "-m", str(shard), *_common_flags(s), "-p", s.prompt,
            "-n", str(TEARDOWN_TOKENS), "--temp", "0", "--top-k", "1", "--seed", "7", "--ignore-eos", "-no-cnv",
            "--simple-io"]


def child_env(cfg: Path | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "HC_STREAM_CONFIG"}
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # the CUDA backend still loads; it must not see a device
    if cfg is not None:
        env["HC_STREAM_CONFIG"] = str(cfg)
    return env


def hcstream_lines(text: str, limit: int = 20) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if "hcstream:" in line)[:limit]


def read_stats(path: Path | None) -> dict | None:
    return canary.read_stats({"stats_path": str(path)}) if path else None  # retries a file caught mid-rename


def wait_health(proc: subprocess.Popen, client: httpx.Client, timeout_s: float) -> str:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            return "exited"
        try:
            if client.get("/health", timeout=2).status_code == 200:
                return "ready"
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    return "timeout"


def _drive_server(proc: subprocess.Popen, port: int, completions: int, stats_path: Path | None,
                  s: Settings) -> tuple[bool, tuple[dict, ...], str | None, dict | None]:
    """Wait for /health, run the completions, then read the stats while the child is still alive."""
    args = SimpleNamespace(prompt=s.prompt, tokens=s.tokens, request_timeout=s.request_timeout_s)
    done: list[dict] = []
    with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
        state = wait_health(proc, client, s.load_timeout_s)
        if state != "ready":
            return False, (), None if state == "exited" else "load timed out", None
        for _ in range(completions):
            try:
                done.append(canary.complete(client, args))
            except httpx.HTTPError as exc:
                return True, tuple(done), f"completion failed: {type(exc).__name__}: {exc}", None
    if stats_path is not None:
        time.sleep(STATS_SETTLE_S)
    return True, tuple(done), None, read_stats(stats_path)


def _settle(proc: subprocess.Popen, wait_s: float, stats_path: Path | None, resident_target: int,
            slot_target: int = 0) -> tuple[int | None, int, int]:
    """The child's own exit code (None if still running after wait_s) and the peak live resident_bytes and
    slot_bytes_committed seen.

    Stats are polled only until snapshots reach both targets: while this process holds the file open the
    extension's rename fails, and the final 'shutdown' snapshot must not be lost that way."""
    deadline = time.perf_counter() + wait_s
    peak = slot_peak = 0

    def short() -> bool:
        return peak < resident_target or slot_peak < slot_target

    while short() and proc.poll() is None and time.perf_counter() < deadline:
        st = read_stats(stats_path) or {}
        peak = max(peak, int(st.get("resident_bytes", 0)))
        slot_peak = max(slot_peak, int(st.get("slot_bytes_committed", 0)))
        if short():
            time.sleep(0.05)
    try:
        code = proc.wait(timeout=max(0.0, deadline - time.perf_counter())) if wait_s > 0 else proc.poll()
    except subprocess.TimeoutExpired:
        code = None
    return code, peak, slot_peak


def run_child(argv: list[str], env: dict[str, str], cwd: Path, log_path: Path, stats_path: Path | None,
              s: Settings, *, port: int | None, completions: int, wait_s: float, resident_target: int = 0,
              slot_target: int = 0) -> Outcome:
    """Start one child, drive it, and stop it on every path (including KeyboardInterrupt) before returning."""
    t0 = time.perf_counter()
    served, done, error, stats, exit_code, peak, slot_peak = False, (), None, None, None, 0, 0
    sampler: probe.ProcessSampler | None = None
    with log_path.open("wb") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=str(cwd), creationflags=CREATE_FLAGS)
        try:
            sampler = probe.ProcessSampler(pid=proc.pid, min_available=int(s.min_available_gib * GIB),
                                           on_pressure=proc.kill).start()
            if port is not None:
                served, done, error, stats = _drive_server(proc, port, completions, stats_path, s)
            exit_code, peak, slot_peak = _settle(proc, wait_s, stats_path, resident_target, slot_target)
        finally:
            try:
                if proc.poll() is None:
                    canary.stop(proc)
            finally:
                samples = sampler.stop() if sampler else []
    if stats is None:
        stats = read_stats(stats_path)
    if sampler is not None and sampler.pressure_tripped:
        error = error or "memory guard killed the child"
    memory = {**probe.summarize(samples), "sampler_error": sampler.error if sampler else "sampler not started"}
    return Outcome(served=served, exit_code=exit_code, completions=done, stats=stats,
                   log_text=log_path.read_text(encoding="utf-8", errors="replace"),
                   seconds=round(time.perf_counter() - t0, 3), memory=memory, error=error, resident_peak=peak,
                   slot_peak=slot_peak)


# --------------------------------------------------------------------------------------------------- checks


def token_checks(outcome: Outcome, case: Case, baseline: tuple[int, ...] | None, s: Settings) -> list[Check]:
    runs = [tuple(c.get("tokens") or ()) for c in outcome.completions]
    checks = [Check("served", outcome.served, outcome.error or ""),
              Check("completions", len(runs) == case.completions, f"{len(runs)}/{case.completions}"),
              Check("token_count", bool(runs) and all(len(r) == s.tokens for r in runs),
                    f"{[len(r) for r in runs]} expected {s.tokens}"),
              Check("alive_after_completions", outcome.exit_code is None, f"exit code {outcome.exit_code}"),
              Check("no_fatal", "hcstream: FATAL" not in outcome.log_text)]
    if baseline is None:
        checks.append(Check("deterministic", len(set(runs)) == 1, f"{len(set(runs))} distinct token sequences"))
    else:
        checks.append(Check("tokens_match_stock", bool(runs) and all(r == baseline for r in runs)))
    return checks


def fatal_checks(outcome: Outcome, case: Case) -> list[Check]:
    checks = [Check("exit_86", outcome.exit_code == FATAL_EXIT, f"exit code {outcome.exit_code}"),
              Check("fatal_logged", "hcstream: FATAL" in outcome.log_text)]
    if case.malformed:
        checks += [Check("never_served", not outcome.served),
                   Check("reason_truncated", "truncated (no 'end' line)" in outcome.log_text)]
    if case.options.fault_read_error_after_bytes:
        stats = outcome.stats or {}
        checks += [*read_fault_checks(outcome.log_text),
                   Check("stats_state_fatal", stats.get("state") == "fatal", f"state {stats.get('state')}"),
                   Check("fault_injected", stats.get("fault_injected", 0) > 0, f"{stats.get('fault_injected')}")]
    return checks


def read_fault_checks(log_text: str) -> list[Check]:
    """The FATAL line must name the injected error code and the aligned file offset of the chunk that failed."""
    m = FATAL_READ.search(log_text)
    code, offset = (int(m.group(2)), int(m.group(3))) if m else (None, None)
    return [Check("reason_read_failed", m is not None, m.group(0) if m else "no FATAL read line"),
            Check("fatal_error_code", code == ERROR_READ_FAULT, f"win32 error {code} expected {ERROR_READ_FAULT}"),
            Check("fatal_offset", offset is not None and offset > 0 and offset % ALIGN == 0,
                  f"offset {offset} expected a positive multiple of {ALIGN}")]


def teardown_checks(outcome: Outcome) -> list[Check]:
    st = outcome.stats or {}
    planned = st.get("resident_bytes_planned", 0)
    slots_planned = int(st.get("slots", 0)) * int(st.get("slot_bytes", 0))
    return [Check("exit_0", outcome.exit_code == 0, f"exit code {outcome.exit_code}"),
            Check("state_shutdown", st.get("state") == "shutdown", f"state {st.get('state')}"),
            Check("resident_released", st.get("resident_bytes") == 0, f"resident_bytes {st.get('resident_bytes')}"),
            # bytes_read also counts streamed layers, so only a live snapshot proves the resident copy existed
            Check("resident_was_loaded", planned > 0 and outcome.resident_peak == planned,
                  f"live snapshot peak {outcome.resident_peak} planned resident {planned}"),
            # slot release is proven by the counter itself, not only by the detach log line
            Check("slots_were_committed", slots_planned > 0 and outcome.slot_peak == slots_planned,
                  f"live snapshot peak {outcome.slot_peak} expected slots x slot_bytes {slots_planned}"),
            Check("slots_released", st.get("slot_bytes_committed") == 0,
                  f"slot_bytes_committed {st.get('slot_bytes_committed')} at shutdown"),
            Check("detached_logged", "hcstream: detached" in outcome.log_text),
            Check("no_fatal", "hcstream: FATAL" not in outcome.log_text)]


def min_passes(plan: StreamPlan, case: Case, s: Settings) -> int:
    """Fewest streamed passes a case must show: the prompt pass plus one per generated token after the first.

    0 when there is no per-pass guarantee: `passes` counts binds of the first streamed layer, which recur per pass
    only when it is evicted in between (more streamed layers than slots); an injected read error stops streaming."""
    if case.options.fault_read_error_after_bytes or len(plan.stream_layers) <= plan.slots:
        return 0
    return TEARDOWN_TOKENS if case.expect == "teardown" else s.tokens * case.completions


def layout_checks(plan: StreamPlan, case: Case, st: dict | None, need_passes: int) -> list[Check]:
    """The extension's own view of the plan must equal the Python layout, byte for byte."""
    if st is None:
        return [Check("stats_present", False, "no stats file")]
    pairs = [("slot_bytes", plan.slot_bytes), ("stream_bytes_per_pass", plan.stream_bytes_per_pass),
             ("resident_bytes_planned", plan.resident_bytes), ("streamed_layers", len(plan.stream_layers)),
             ("verify_failures", 0)]
    if plan.stream_layers:
        pairs.append(("slots", plan.slots))
    if case.expect == "tokens":  # read while the child is alive, so every buffer is still committed
        pairs += [("resident_bytes", plan.resident_bytes), ("slot_bytes_committed", plan.slots * plan.slot_bytes)]
    if not case.options.fault_read_error_after_bytes:
        pairs.append(("io_errors", 0))
    checks = [Check(f"stats_{k}", st.get(k) == v, f"{st.get(k)} expected {v}") for k, v in pairs]
    if plan.stream_layers:
        passes = int(st.get("passes", 0))
        floor = max(need_passes, 1)
        checks.append(Check("stats_passes", passes >= floor, f"{passes} expected >= {floor}"))
        if need_passes:  # every streamed layer ran from a slot on every pass, none from the mapping
            want = passes * len(plan.stream_layers)
            checks.append(Check("stats_binds", st.get("binds") == want, f"{st.get('binds')} expected {want}"))
    return checks


def option_checks(plan: StreamPlan, case: Case, outcome: Outcome) -> list[Check]:
    st = outcome.stats or {}
    opts = case.options
    checks: list[Check] = []
    if opts.fault_read_error_after_bytes and opts.on_io_error == "mmap":
        checks += [Check("mmap_fallbacks", st.get("mmap_fallbacks", 0) > 0, f"{st.get('mmap_fallbacks')}"),
                   Check("fault_injected", st.get("fault_injected", 0) > 0, f"{st.get('fault_injected')}")]
    if opts.fault_read_delay_us:
        chunks = st.get("chunks", 0)
        mean_ns = st.get("read_ns", 0) / chunks if chunks else 0.0
        checks.append(Check("delay_applied", mean_ns >= opts.fault_read_delay_us * 1000,
                            f"measured mean chunk read {mean_ns / 1e6:.2f} ms over {chunks} chunks"))
    if not plan.stream_layers and not plan.resident_layers:
        checks += [Check("enabled_prefetch_off", "hcstream: enabled" in outcome.log_text
                         and "prefetch off" in outcome.log_text),
                   Check("not_attached", "hcstream: attached" not in outcome.log_text),
                   *passthrough_checks(st, opts)]
    elif case.expect != "fatal":
        checks.append(Check("attached", "hcstream: attached" in outcome.log_text))
    return checks


def passthrough_checks(st: dict, opts: StreamOptions) -> list[Check]:
    """A config without managed layers writes a 'passthrough' snapshot once the context exists; with no_prefetch
    its no-op counters must show the loader's PrefetchVirtualMemory call, and nothing may be committed or read."""
    calls, size = int(st.get("prefetch_noop_calls", 0)), int(st.get("prefetch_noop_bytes", 0))
    noop_ok = calls >= 1 and size > 0 if opts.no_prefetch else calls == 0
    idle = {k: st.get(k) for k in ("slot_bytes_committed", "resident_bytes", "bytes_read", "streamed_layers")}
    return [Check("stats_passthrough", st.get("state") == "passthrough", f"state {st.get('state')}"),
            Check("prefetch_noop_counted", noop_ok, f"{calls} no-op calls covering {size} bytes"),
            Check("passthrough_idle", all(v == 0 for v in idle.values()), f"{idle}")]


def evaluate(case: Case, plan: StreamPlan | None, outcome: Outcome, baseline: tuple[int, ...] | None,
             s: Settings) -> tuple[Check, ...]:
    checks: list[Check] = []
    if case.expect == "tokens":
        checks += token_checks(outcome, case, baseline if case.engine == "overlay" else None, s)
        if case.engine == "overlay" and baseline is None:
            checks.append(Check("stock_baseline", False, "no passing stock run to compare with"))
    elif case.expect == "fatal":
        checks += fatal_checks(outcome, case)
    else:
        checks += teardown_checks(outcome)
    if plan is not None and case.expect != "fatal":
        if plan.stream_layers or plan.resident_layers:
            checks += layout_checks(plan, case, outcome.stats, min_passes(plan, case, s))
        checks += option_checks(plan, case, outcome)
    return tuple(checks)


# --------------------------------------------------------------------------------------------------- matrix


def _plan_dict(plan: StreamPlan | None) -> dict | None:
    if plan is None:
        return None
    return {"resident_layers": list(plan.resident_layers), "stream_layers": list(plan.stream_layers),
            "slots": plan.slots, "slot_bytes": plan.slot_bytes, "resident_bytes": plan.resident_bytes,
            "stream_bytes_per_pass": plan.stream_bytes_per_pass, "committed_bytes": plan.committed_bytes}


def run_case(target: Target, case: Case, s: Settings, out_dir: Path, baseline: tuple[int, ...] | None) -> CaseResult:
    plan = plan_for(target, case)
    cfg, stats_path = write_config(target, case, plan, out_dir) if plan is not None else (None, None)
    bin_dir = target.stock_bin if case.engine == "stock" else target.overlay_bin
    log_path = out_dir / f"{case.name}.log"
    port = None if case.expect == "teardown" else free_loopback_port()
    if port is None:
        argv = completion_argv(bin_dir, target.shards[0], s)
    else:
        argv = server_argv(bin_dir, target.shards[0], port, s)
    wait_s = {"tokens": 0.0, "fatal": s.exit_timeout_s, "teardown": s.request_timeout_s}[case.expect]
    live = plan is not None and case.expect == "teardown"  # only teardown polls live snapshots
    outcome = run_child(argv, child_env(cfg), bin_dir, log_path, stats_path, s, port=port,
                        completions=case.completions, wait_s=wait_s,
                        resident_target=plan.resident_bytes if live and plan else 0,
                        slot_target=plan.slots * plan.slot_bytes if live and plan else 0)
    checks = evaluate(case, plan, outcome, baseline, s)
    tokens = tuple(outcome.completions[0].get("tokens") or ()) if outcome.completions else ()
    match = None if case.expect != "tokens" or case.engine == "stock" or baseline is None else tokens == baseline
    return CaseResult(name=case.name, description=case.description, expect=case.expect,
                      passed=all(c.ok for c in checks), checks=checks, exit_code=outcome.exit_code,
                      token_match=match, tokens=tokens, stats=outcome.stats, plan=_plan_dict(plan),
                      ssd_estimate_bytes=estimate_ssd_bytes(plan, case, s), seconds=outcome.seconds,
                      memory=outcome.memory, error=outcome.error, log=str(log_path),
                      hcstream_log=hcstream_lines(outcome.log_text))


def failed_result(case: Case, error: str) -> CaseResult:
    return CaseResult(name=case.name, description=case.description, expect=case.expect, passed=False,
                      checks=(Check("ran", False, error),), exit_code=None, token_match=None, tokens=(), stats=None,
                      plan=None, ssd_estimate_bytes=0, seconds=0.0, memory={}, error=error, log=None)


def run_matrix(target: Target, cases: Sequence[Case], s: Settings, out_dir: Path,
               max_ssd_bytes: int) -> tuple[CaseResult, ...]:
    """Run cases in order; the first stock case is the token baseline. Children never outlive their case."""
    results: list[CaseResult] = []
    baseline: tuple[int, ...] | None = None
    read = 0
    for case in cases:
        if read > max_ssd_bytes:
            results.append(failed_result(case, f"skipped: measured reads {read} exceed the budget"))
            continue
        try:
            result = run_case(target, case, s, out_dir, baseline)
        except Exception as exc:  # noqa: BLE001 - one broken case is evidence, not a reason to stop the matrix
            result = failed_result(case, f"{type(exc).__name__}: {exc}")
        if case.engine == "stock" and result.passed and baseline is None:
            baseline = result.tokens
        read += int((result.stats or {}).get("bytes_read", 0))
        results.append(result)
        print(format_row(result), flush=True)
    return tuple(results)


def format_row(r: CaseResult) -> str:
    st = r.stats or {}
    failed = ",".join(c.name for c in r.checks if not c.ok) or "-"
    exit_code = "stopped" if r.exit_code is None else str(r.exit_code)
    match = "-" if r.token_match is None else ("yes" if r.token_match else "NO")
    read = f"{st['bytes_read'] / MIB:.0f}" if "bytes_read" in st else "-"
    return (f"{r.name:<14} {r.expect:<9} {'PASS' if r.passed else 'FAIL':<5} {exit_code:<8} {match:<6} "
            f"{read:>8} {str(st.get('passes', '-')):>6} {str(st.get('mmap_fallbacks', '-')):>9} {r.seconds:>7.1f}  "
            f"{failed}")


def format_table(results: Sequence[CaseResult]) -> str:
    head = (f"{'case':<14} {'expect':<9} {'result':<5} {'exit':<8} {'match':<6} {'read MiB':>8} {'passes':>6} "
            f"{'fallbacks':>9} {'secs':>7}  failed checks")
    verdict = "PASS" if results and all(r.passed for r in results) else "FAIL"
    return "\n".join([head, *(format_row(r) for r in results), f"overall: {verdict}"])


def summary_doc(target: Target, s: Settings, results: Sequence[CaseResult], started: str, estimate: int) -> dict:
    measured = sum(int((r.stats or {}).get("bytes_read", 0)) for r in results)
    return {
        "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
        "passed": bool(results) and all(r.passed for r in results),
        "model": str(target.model), "shards": [str(p) for p in target.shards], "n_layer": target.n_layer,
        "stock_bin": str(target.stock_bin), "overlay_bin": str(target.overlay_bin),
        "proxy_sha256": target.proxy_sha256, "settings": asdict(s),
        "ssd_reads": {
            "planned_bytes": {"value": estimate, "provenance": "estimate",
                              "note": "plan pass bytes x (tokens + warmup + prompt) + resident copies; fault cases "
                                      "capped at the threshold plus one chunk per reader"},
            "hcstream_bytes_read": {"value": measured, "provenance": "measured",
                                    "note": "sum of the extension's bytes_read counters (unbuffered reads)"},
            "sys_disk_read_bytes": {"value": sum(int(r.memory.get("sys_disk_read_bytes", 0)) for r in results),
                                    "provenance": "measured", "note": "system-wide, includes other processes"}},
        "cases": [asdict(r) for r in results],
    }


def select_cases(matrix: Sequence[Case], only: str | None) -> tuple[Case, ...]:
    """The named cases in matrix order, always with the stock baseline first."""
    if not only:
        return tuple(matrix)
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    unknown = wanted - {c.name for c in matrix}
    if unknown:
        raise SetupError(f"unknown cases {sorted(unknown)}; known: {[c.name for c in matrix]}")
    return tuple(c for c in matrix if c.name in wanted or c.engine == "stock")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=f"GGUF file or directory (default models/{MODEL_DIR})")
    ap.add_argument("--only", help="comma-separated case names; the stock baseline always runs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-ssd-gib", type=float, default=40.0, help="refuse to run above this planned SSD reading")
    ap.add_argument("--out", help="output directory (default results/capacity/<UTC stamp>-validate)")
    return ap.parse_args(argv)


def output_dir(out: str | None, started: datetime) -> Path:
    """Absolute, because children run with the engine directory as their cwd and get the config and stats paths
    from this directory: a relative path would point into the engine directory."""
    path = Path(out).expanduser() if out else ROOT / "results" / "capacity" / f"{started:%Y%m%dT%H%M%SZ}-validate"
    return path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # a non-ASCII evidence path must not crash a cp1252 console at the end
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    settings = Settings(threads=args.threads)
    try:
        target = prepare(Path(args.model).expanduser() if args.model else default_model())
        cases = select_cases(build_matrix(target.n_layer), args.only)
    except (SetupError, overlay.OverlayError, OSError, ValueError) as exc:
        print(f"cannot validate: {exc}", file=sys.stderr)
        return 2
    estimate = sum(estimate_ssd_bytes(plan_for(target, c), c, settings) for c in cases)
    if estimate > args.max_ssd_gib * GIB:
        print(f"planned SSD reading {estimate / GIB:.1f} GiB is over --max-ssd-gib {args.max_ssd_gib}", file=sys.stderr)
        return 2
    started = datetime.now(UTC)
    out_dir = output_dir(args.out, started)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases on {target.model.name}, planned SSD reading {estimate / GIB:.1f} GiB (estimate)")
    results = run_matrix(target, cases, settings, out_dir, int(args.max_ssd_gib * GIB))
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary_doc(target, settings, results, started.isoformat(), estimate), indent=2),
                    encoding="utf-8")
    print(format_table(results))
    print(f"evidence: {path}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
