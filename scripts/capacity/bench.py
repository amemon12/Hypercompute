"""Reproducible capacity benchmark: one isolated llama-server child per case, with phases measured separately.

  uv run --no-sync python scripts/capacity/bench.py --preset stock-cpu --model qwen2.5-0.5b-instruct-q4_k_m \
      --output-tokens 16 --reps 3 --ctx 1024
  uv run --no-sync python scripts/capacity/bench.py --preset gpu-ram-8 --preset stream-r8-s20-k2 \
      --model qwen2.5-7b-instruct-q4_k_m
  uv run --no-sync python scripts/capacity/bench.py --case-file cases.json --output-tokens 128

Presets put blocks 0.. on the host CPU and the rest on --gpu: stock-cpu (every unit on the CPU, mmap weights),
stock-gpu (every unit on the GPU), gpu-ram-N (N CPU blocks in private RAM), stream-rR-sS-kK (R resident and S
streamed CPU blocks through K hcstream slots; the head stays resident when R + S covers every block).
A --case-file is a JSON list of cases in the shape bench.json records under "case".

Each case writes results/capacity/<UTC stamp>-bench-<label>/bench.json (schema hypercompute.capacity.bench/1) next to
the engine log and the memory samples. The child joins a kill-on-close Job Object on Windows and its process tree is
killed in a finally block, so it dies on every exit path. A case is refused, with bench.json saying why, when something
already listens on its port, when available RAM minus its predicted residency would drop below 4 GiB, or when the
results volume has less than 5 GiB free. Models are referenced by directory name under ~/.hypercompute/models or path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from hypercompute import config, engine, gguf  # noqa: E402
from hypercompute.capacity import benchlib as bl  # noqa: E402
from hypercompute.capacity import overlay, probe  # noqa: E402
from hypercompute.capacity.stream_config import (  # noqa: E402
    StreamConfigError,
    block_count,
    layer_layouts,
    render_config,
    stream_plan,
)
from hypercompute.netutil import can_bind, free_loopback_port  # noqa: E402

GIB = 1 << 30
RESULTS = ROOT / "results" / "capacity"
SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$")
GPU_RAM_RE = re.compile(r"^gpu-ram-(?P<n>\d+)$")
STREAM_RE = re.compile(r"^stream-r(?P<r>\d+)-s(?P<s>\d+)-k(?P<k>\d+)$")
NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
# Presets turn llama-server's host prompt cache off: with a unique prompt per rep it would keep every earlier prompt's
# KV state in RAM (up to 8 GiB by default), so memory peaks would drift upward from rep to rep.
NO_PROMPT_CACHE = ("--cache-ram", "0")


# --------------------------------------------------------------------------------------------------- presets


@dataclass(frozen=True)
class Preset:
    name: str
    kind: Literal["stock-cpu", "stock-gpu", "gpu-ram", "stream"]
    cpu_layers: int = 0
    resident: int = 0
    streamed: int = 0
    slots: int = 0


def parse_preset(name: str) -> Preset:
    """stock-cpu, stock-gpu, gpu-ram-N (N CPU blocks) or stream-rR-sS-kK (R resident, S streamed blocks, K slots)."""
    if name in ("stock-cpu", "stock-gpu"):
        return Preset(name=name, kind=name)  # type: ignore[arg-type]
    if m := GPU_RAM_RE.match(name):
        if int(m["n"]) < 1:
            raise bl.BenchError("gpu-ram-N needs N >= 1 CPU layers")
        return Preset(name=name, kind="gpu-ram", cpu_layers=int(m["n"]))
    if m := STREAM_RE.match(name):
        r, s, k = int(m["r"]), int(m["s"]), int(m["k"])
        if r + s < 1 or (s and k < 1):
            raise bl.BenchError(f"{name}: needs at least one CPU block and, when streaming, at least one slot")
        return Preset(name=name, kind="stream", resident=r, streamed=s, slots=k)
    raise bl.BenchError(f"unknown preset {name!r}: use stock-cpu, stock-gpu, gpu-ram-N or stream-rR-sS-kK")


def case_for_preset(preset: Preset, model_paths: Sequence[str], n_layer: int, *, label: str, ctx: int,
                    threads: int | None, windows: bool, gpu: str = "CUDA0") -> bl.Case:
    """llama_rpc.build_args' engine arguments for this tier (blocks 0.. on the CPU, the rest on `gpu`), plus
    NO_PROMPT_CACHE for every tier (llama_rpc only sets it for the stream tier)."""
    units = n_layer + 1
    paths = tuple(model_paths)
    base = bl.EngineArgs(ctx=ctx, threads=threads, flags=NO_PROMPT_CACHE)
    if preset.kind == "stock-cpu":
        return bl.Case(label=label, model_paths=paths, engine=base)
    if preset.kind == "stock-gpu":
        return bl.Case(label=label, model_paths=paths, engine=replace(base, devices=(gpu,), tensor_split=(units,),
                                                                       ngl="all"))
    if preset.kind == "gpu-ram":
        if preset.cpu_layers >= n_layer:
            raise bl.BenchError(f"{preset.name}: the model has {n_layer} blocks; use stock-cpu for all of them")
        gpu_units = units - preset.cpu_layers
        flags = ("--no-host", "--no-repack", *NO_PROMPT_CACHE) if windows else NO_PROMPT_CACHE
        return bl.Case(label=label, model_paths=paths, engine=replace(
            base, devices=(gpu,), tensor_split=(gpu_units,), ngl=gpu_units,
            load_mode="none" if windows else "mmap", flags=flags))
    return _stream_case(preset, paths, n_layer, label, base, gpu)


def _stream_case(preset: Preset, paths: tuple[str, ...], n_layer: int, label: str, base: bl.EngineArgs,
                 gpu: str) -> bl.Case:
    cpu_blocks = preset.resident + preset.streamed
    if cpu_blocks > n_layer:
        raise bl.BenchError(f"{preset.name}: {cpu_blocks} CPU blocks but the model has {n_layer}")
    resident = tuple(range(preset.resident))
    streamed = tuple(range(preset.resident, cpu_blocks))
    engine_args = replace(base, flags=("--no-repack", "--no-op-offload", *NO_PROMPT_CACHE))
    if cpu_blocks == n_layer:  # CPU only: the head is a CPU unit too and stays resident
        resident = (*resident, n_layer)
    else:
        gpu_units = n_layer + 1 - cpu_blocks
        engine_args = replace(engine_args, devices=(gpu,), tensor_split=(gpu_units,), ngl=gpu_units)
    layout = bl.StreamLayout(resident_units=resident, streamed_units=streamed, slots=preset.slots or 1)
    return bl.Case(label=label, model_paths=paths, engine=engine_args, stream=layout)


# --------------------------------------------------------------------------------------------------- model and host


def _split_set(first: Path) -> list[Path]:
    """Every shard name of the split set `first` belongs to, or just `first` when it is not split."""
    m = SPLIT_RE.match(first.name)
    if not m:
        return [first]
    return [first.with_name(f"{m['stem']}-{i:05d}-of-{m['n']}.gguf") for i in range(1, int(m["n"]) + 1)]


def model_shards(spec: str) -> tuple[Path, ...]:
    """Every shard of a model given a directory name under the models dir, a directory, or a (first) GGUF file."""
    path = Path(spec).expanduser()
    if not path.exists():
        path = config.paths().models / spec
    if path.is_dir():
        found = sorted(f for f in path.glob("*.gguf") if not f.name.startswith("mmproj"))
        files = _split_set(found[0]) if found else []
        if found and files != found:  # a partial split set or several models would be read as one model's shards
            raise bl.BenchError(f"model {spec!r}: the GGUF files in {path} are not one complete model "
                                f"({', '.join(f.name for f in found)})")
    elif path.is_file():
        files = _split_set(path)
    else:
        raise bl.BenchError(f"model {spec!r} not found (looked in {config.paths().models})")
    if not files or not all(f.is_file() for f in files):
        raise bl.BenchError(f"model {spec!r}: missing GGUF shards in {path}")
    return tuple(files)


def model_facts(paths: tuple[Path, ...]) -> bl.ModelFacts:
    """Exact unit sizes and KV sizing read from the GGUF headers (no tensor data is read)."""
    summary = gguf.summarize(paths)
    table = gguf.read_tensor_table(paths)
    layouts = layer_layouts(table)
    n_layer = block_count(table)
    if summary.n_layer is not None and summary.n_layer != n_layer:
        raise bl.BenchError(f"block_count {summary.n_layer} disagrees with the tensor table ({n_layer} blocks)")
    if any(i not in layouts for i in range(n_layer + 1)):
        raise bl.BenchError("the GGUF is missing tensors for some blocks or the output head")
    head_dim = summary.n_embd // summary.n_head if summary.n_embd and summary.n_head else 0
    k_len, v_len = summary.key_length or head_dim, summary.value_length or head_dim
    return bl.ModelFacts(n_layer=n_layer, unit_bytes=tuple(layouts[i].buffer_bytes for i in range(n_layer + 1)),
                         token_embd_bytes=summary.token_embd_bytes, tied_embeddings=summary.output_tied,
                         kv_bytes_per_layer_token=(summary.n_head_kv or 0) * (k_len + v_len) * bl.KV_ELEM_BYTES,
                         file_bytes=sum(p.stat().st_size for p in paths))


def model_identity(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Shard sizes and mtimes, plus the manifest fingerprint when capacity.manifest imports and succeeds."""
    model_id = paths[0].parent.name
    shards = [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths]
    out: dict[str, Any] = {"model_id": model_id, "shards": shards, "bytes": sum(s["size"] for s in shards)}
    try:
        from hypercompute.capacity.manifest import build_manifest
    except Exception as exc:  # optional module (being written elsewhere); the failure is recorded, not raised
        return {**out, "fingerprint": None, "fingerprint_error": f"manifest not importable: {bl._err(exc)}"}
    try:
        manifest = build_manifest(model_id, paths)
    except Exception as exc:  # a manifest failure must not cost the benchmark; it is recorded with its reason
        return {**out, "fingerprint": None, "fingerprint_error": bl._err(exc)}
    return {**out, "fingerprint": manifest.fingerprint, "architecture": manifest.architecture,
            "file_type": manifest.file_type, "n_layer": manifest.n_layer}


def power_source() -> dict[str, Any]:
    via = "psutil.sensors_battery"
    try:
        battery = psutil.sensors_battery()
    except (psutil.Error, OSError, NotImplementedError, AttributeError) as exc:
        return {"source": "unknown", "via": via, "error": bl._err(exc)}
    if battery is None:
        return {"source": "no_battery", "via": via}
    plugged = battery.power_plugged
    return {"source": "unknown" if plugged is None else ("ac" if plugged else "battery"), "percent": battery.percent,
            "via": via}


def parse_nvidia_smi(gpu_csv: str, apps_csv: str, pid: int) -> dict[str, Any]:
    """GPU rows and `pid`'s usage (MiB) from `--query-gpu=index,name,memory.used,memory.total` and
    `--query-compute-apps=pid,used_memory`, both with `--format=csv,noheader,nounits`. Bad rows are skipped."""
    gpus, used = [], None
    for line in gpu_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit() and NUMBER_RE.match(parts[2]) and NUMBER_RE.match(parts[3]):
            gpus.append({"index": int(parts[0]), "name": parts[1], "used_mib": int(float(parts[2])),
                         "total_mib": int(float(parts[3]))})
    for line in apps_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid and NUMBER_RE.match(parts[1]):
            used = int(float(parts[1]))
    return {"available": True, "unit": "MiB", "gpus": gpus, "process_used_mib": used}


def gpu_memory(pid: int) -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    queries = (["--query-gpu=index,name,memory.used,memory.total"], ["--query-compute-apps=pid,used_memory"])
    try:
        out = [subprocess.run([exe, *q, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                              check=True).stdout for q in queries]
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": bl._err(exc)}
    return parse_nvidia_smi(out[0], out[1], pid)


def host_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {"system": platform.system(), "release": platform.release(), "version": platform.version(),
            "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
            "ram_total_bytes": vm.total, "ram_available_bytes": vm.available, "power": power_source()}


# --------------------------------------------------------------------------------------------------- preflight


def port_listener(port: int) -> str | None:
    """'name (pid N)' of a process listening on `port`, or None. Uses the socket table, never a connection."""
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
                try:
                    name = psutil.Process(c.pid).name() if c.pid else "unknown"
                except psutil.Error:
                    name = "unknown"
                return f"{name} (pid {c.pid})"
    except (psutil.Error, OSError) as exc:
        print(f"note: socket table unavailable ({bl._err(exc)}); checking the port by binding it", flush=True)
    ok, err = can_bind("127.0.0.1", port)
    return None if ok else f"an unknown process ({err})"


def preflight(case: bl.Case, facts: bl.ModelFacts, port: int) -> bl.Preflight:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return bl.Preflight(port=port, port_listener=port_listener(port),
                        available_ram_bytes=psutil.virtual_memory().available,
                        predicted=bl.predict_host_bytes(case, facts), free_disk_bytes=shutil.disk_usage(RESULTS).free,
                        disk_path=str(RESULTS))


# --------------------------------------------------------------------------------------------------- child process


@dataclass(frozen=True)
class Launch:
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    stats_path: Path | None
    meta: dict[str, Any]


def prepare_stream(case: bl.Case, engine_dir: Path, out_dir: Path, port: int) -> tuple[Path, Path, dict[str, Any]]:
    """Overlay engine dir, stats path and launch metadata for an hcstream case (config written into out_dir)."""
    layout = case.stream
    if layout is None:
        raise bl.BenchError(f"case {case.label} has no stream layout")
    dll = overlay.find_proxy_dll()
    if dll is None:
        raise bl.BenchError("the hcstream proxy DLL is not built: "
                            "uv run --no-project --with ziglang python scripts/build_hcstream.py")
    try:
        ov = overlay.ensure_overlay(engine_dir, dll)
        shards = [Path(p) for p in case.model_paths]
        table = gguf.read_tensor_table(shards)
        plan = stream_plan(layer_layouts(table), layout.resident_units, layout.streamed_units, layout.slots)
        stats = out_dir / f"hcstream-stats-{port}.json"
        cfg = out_dir / f"hcstream-{port}.cfg"
        cfg.write_text(render_config(shards, table, plan, stats, layout.options), encoding="utf-8")
    except (overlay.OverlayError, StreamConfigError, gguf.GgufError, OSError) as exc:
        raise bl.BenchError(f"cannot prepare the stream tier: {bl._err(exc)}") from exc
    return ov.bin_dir, stats, {"overlay": str(ov.bin_dir), "proxy_sha256": ov.proxy_sha256, "config": str(cfg),
                               "stats_path": str(stats), "stream_plan": asdict(plan) | {
                                   "committed_bytes": plan.committed_bytes}}


def prepare_launch(case: bl.Case, out_dir: Path, port: int, api_key: str) -> tuple[Launch, dict[str, Any]]:
    stock = engine.find_installed()
    if stock is None:
        raise bl.BenchError("no installed llama.cpp engine (uv run hypercompute engine install)")
    # LLAMA_ARG_* variables are llama-server defaults for every flag the argv leaves out (batch, flash attention,
    # KV type...): inherited, they would change the measurement without appearing in bench.json.
    removed = sorted(k for k in os.environ if k.upper().startswith("LLAMA_ARG_"))
    env = {k: v for k, v in os.environ.items() if k != "HC_STREAM_CONFIG" and k not in removed}
    env["LLAMA_API_KEY"] = api_key  # the child is loopback only; the key keeps local web pages from driving it
    bin_dir, stats, meta = stock.bin_dir, None, {}
    if case.stream is not None:
        bin_dir, stats, meta = prepare_stream(case, stock.bin_dir, out_dir, port)
        env["HC_STREAM_CONFIG"] = meta["config"]
    argv = bl.build_argv(case, bin_dir / engine.exe_name("llama-server"), port)
    log = str(out_dir / f"llama-server-{port}.log")
    meta = {**meta, "argv": argv, "port": port, "bin_dir": str(bin_dir), "log": log, "env_removed": removed}
    info = {"release": stock.release, "variant": stock.variant, "accelerator": stock.accelerator,
            "engine_dir": str(stock.bin_dir), "source": stock.source}
    return Launch(argv=tuple(argv), cwd=bin_dir, env=env, stats_path=stats, meta=meta), info


def _job_binder() -> Callable[[subprocess.Popen], None] | None:
    """The supervisor's kill-on-close Job Object binder (Windows), or None; the finally-block tree kill remains."""
    if os.name != "nt":
        return None
    try:
        from hypercompute.agent.supervisor import bind_to_owner
    except Exception as exc:  # the binder is a second line of defence; its absence is reported, not fatal
        print(f"note: no Job Object binding ({bl._err(exc)}); relying on the process tree kill", flush=True)
        return None
    return bind_to_owner


def start_child(launch: Launch) -> tuple[subprocess.Popen, float]:
    """Start llama-server. Returns it with the perf_counter time taken just before Popen, the zero of the load time.

    Nothing that can raise runs between this return and the caller's try/finally; an interrupt while the child is
    being bound to the Job Object kills it before propagating.
    """
    bind = _job_binder()  # resolved before Popen so an import failure cannot leave an unowned child
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        with Path(launch.meta["log"]).open("wb") as log:
            t_start = time.perf_counter()
            proc = subprocess.Popen(list(launch.argv), stdout=log, stderr=subprocess.STDOUT, env=launch.env,
                                    cwd=str(launch.cwd), creationflags=flags)
    except OSError as exc:
        raise bl.BenchError(f"llama-server did not start: {bl._err(exc)}") from exc
    if bind is not None:
        try:
            bind(proc)  # logs and continues on OSError; never raises for a failed assignment
        except BaseException:
            kill_tree(proc)
            raise
    return proc, t_start


def _running_children(proc: subprocess.Popen) -> list[psutil.Process]:
    """The child's descendants, looked up only while the child runs: a reaped child's pid may belong to another
    process by now, and its descendants must not be killed."""
    if proc.poll() is not None:
        return []
    try:
        return psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        return []


def _stop_child(proc: subprocess.Popen) -> str | None:
    """Terminate, then kill, the child; the error text when it is still not gone."""
    if proc.poll() is not None:
        return None
    try:
        proc.terminate()
        proc.wait(timeout=30)
        return None
    except (OSError, subprocess.TimeoutExpired) as exc:
        first = bl._err(exc)
    try:
        proc.kill()
        proc.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{first}; then {bl._err(exc)}"
    return None


def kill_tree(proc: subprocess.Popen) -> dict[str, Any]:
    """Stop the child and every process it started; report what is still alive afterwards (should be nothing).
    Never raises for a process that refuses to die: that is reported."""
    t0 = time.perf_counter()
    children = _running_children(proc)
    errors = [e for e in (_stop_child(proc),) if e]
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            errors.append(f"pid {child.pid}: {bl._err(exc)}")
    _, alive = psutil.wait_procs(children, timeout=10)
    return {"exit_code": proc.returncode, "stop_s": round(time.perf_counter() - t0, 3), "children": len(children),
            "children_alive": [p.pid for p in alive], "child_alive_after_stop": proc.poll() is None,
            "error": "; ".join(errors) or None}


def read_stats(path: Path) -> dict[str, Any] | None:
    for _ in range(5):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            time.sleep(0.02)  # the extension replaces the file atomically; a read can still race the rename
    return None


def _updated_ms(stats: dict[str, Any]) -> int:
    try:
        return int(stats.get("updated_unix_ms", 0))
    except (TypeError, ValueError):
        return 0


class ChildObserver:
    """bl.Observer over the child: direct counters at phase boundaries, sampler peaks inside the window."""

    STATS_WAIT_S = 2.0

    def __init__(self, pid: int, sampler: probe.ProcessSampler, t0: float, stats_path: Path | None,
                 cuda: bool) -> None:
        self._pid, self._sampler, self._t0, self._stats_path, self._cuda = pid, sampler, t0, stats_path, cuda

    def now(self) -> float:
        """Seconds on the observer's clock, the one the sampler's samples use."""
        return round(time.perf_counter() - self._t0, 3)

    def counters(self, fresh_after_unix_ms: int | None = None) -> bl.Counters:
        stats, stats_error = self._stats(fresh_after_unix_ms)
        t = time.perf_counter() - self._t0
        try:
            proc = psutil.Process(self._pid)
            mem, io = proc.memory_info(), proc.io_counters()
            disk = psutil.disk_io_counters()
        except (psutil.Error, OSError) as exc:
            return bl.Counters(t=t, proc_read_bytes=0, sys_disk_read_bytes=0, page_faults=0, working_set=0, private=0,
                               stream_stats=stats, error=f"counters unavailable: {bl._err(exc)}",
                               stats_error=stats_error)
        return bl.Counters(t=t, proc_read_bytes=io.read_bytes, sys_disk_read_bytes=disk.read_bytes if disk else 0,
                           page_faults=int(getattr(mem, "num_page_faults", 0)), working_set=mem.rss,
                           private=int(getattr(mem, "private", mem.vms)), stream_stats=stats,
                           stats_error=stats_error)

    def _stats(self, fresh_after_unix_ms: int | None) -> tuple[dict[str, Any] | None, str | None]:
        """hcstream stats written at or after `fresh_after_unix_ms`. A missing or stale file gives no stats and an
        error, so a phase never reports a byte delta computed from an old snapshot."""
        if self._stats_path is None:
            return None, None
        wanted = fresh_after_unix_ms or 0
        deadline = time.monotonic() + self.STATS_WAIT_S
        while True:
            stats = read_stats(self._stats_path)
            if stats is not None and _updated_ms(stats) >= wanted:
                return stats, None
            if time.monotonic() > deadline:
                if stats is None:
                    return None, f"hcstream stats unreadable at {self._stats_path}"
                return None, (f"hcstream stats not refreshed within {self.STATS_WAIT_S:.0f}s "
                              f"(updated_unix_ms {_updated_ms(stats)} < {wanted})")
            time.sleep(0.02)

    def peaks(self, t0: float, t1: float) -> tuple[int, int]:
        s = probe.summarize(list(self._sampler.samples), t0, t1)
        return int(s.get("working_set_peak", 0)), int(s.get("private_peak", 0))

    def gpu_memory(self) -> dict[str, Any] | None:
        return gpu_memory(self._pid) if self._cuda else None

    def power(self) -> dict[str, Any]:
        return power_source()


# --------------------------------------------------------------------------------------------------- one case


@dataclass(frozen=True)
class Options:
    port: int
    min_available_gib: float
    dry_run: bool

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise bl.BenchError(f"--port must be 0 (pick a free one) to 65535, not {self.port}")
        if not (math.isfinite(self.min_available_gib) and self.min_available_gib >= 0):
            raise bl.BenchError(f"--min-available-gib must be a finite number >= 0, not {self.min_available_gib}")


def load_result(client: httpx.Client, proc: subprocess.Popen, workload: bl.BenchWorkload, observer: ChildObserver,
                t_start: float, log_path: Path) -> bl.LoadResult:
    """Process start to the first /health 200, the engine's load lines, and the log tail when the load failed."""
    ok, seconds, error = bl.wait_healthy(client, lambda: proc.poll() is None, workload.load_timeout_s, t_start)
    loaded_at = observer.now()  # sampler clock, so the load memory window matches samples.json
    if not ok and proc.poll() is not None:
        error = f"{error} (exit code {proc.returncode})"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return bl.LoadResult(ok=ok, seconds=round(seconds, 3), error=error, log_lines=bl.load_log_lines(text),
                         build=bl.parse_build(text), memory={"loaded_at_s": loaded_at},
                         log_tail=() if ok else bl.log_tail(text))


def measure(client: httpx.Client, proc: subprocess.Popen, case: bl.Case, workload: bl.BenchWorkload,
            observer: ChildObserver, t_start: float, log_path: Path) -> Iterator[bl.LoadResult | bl.RepResult]:
    """Yields the load result, then each rep as it finishes, so an interrupt or an error keeps what already ran."""
    load = load_result(client, proc, workload, observer, t_start, log_path)
    print(f"[{case.label}] load: {'ok' if load.ok else 'FAILED'} in {load.seconds:.2f}s"
          f"{'' if load.ok else ': ' + str(load.error)}", flush=True)
    yield load
    if not load.ok:
        return
    for i in range(workload.warmup + workload.reps):
        if proc.poll() is not None:
            raise bl.BenchError(f"llama-server exited during the reps (exit code {proc.returncode}) before rep {i}")
        rep = bl.run_rep(client, case, workload, i, warmup=i < workload.warmup, observer=observer)
        print(rep_line(case.label, rep), flush=True)
        yield rep


def rep_line(label: str, rep: bl.RepResult) -> str:
    m = bl.rep_metrics(rep)
    head = f"[{label}] rep {rep.index}{' (warmup)' if rep.warmup else ''}"
    errors = "; ".join(str(f["error"]) for f in bl.failures([rep]))
    return (f"{head}: {'ok' if rep.ok else 'FAILED ' + errors} | prefill {m.get('prefill_tokens')} tok in "
            f"{m.get('prefill_ms')} ms | decode {m.get('decode_tokens')} tok in {m.get('decode_ms')} ms | "
            f"ttft {m.get('ttft_ms')} ms | reuse cache_n {m.get('reuse_cache_n')}")


def _shutdown(proc: subprocess.Popen, client: httpx.Client | None,
              sampler: probe.ProcessSampler | None) -> tuple[dict[str, Any], list[probe.Sample]]:
    """Close the client, kill the child tree (even if closing failed), then stop the sampler."""
    try:
        if client is not None:
            client.close()
    finally:
        stop = kill_tree(proc)
    samples = sampler.stop() if sampler is not None else []
    return stop, samples


def _finish(load: bl.LoadResult | None, stop: dict[str, Any], samples: list[probe.Sample],
            sampler: probe.ProcessSampler | None, out_dir: Path) -> tuple[bl.LoadResult | None, dict[str, Any]]:
    """Load-window memory from the samples, the stop record, and samples.json (a write failure is recorded)."""
    if load is not None:
        load = replace(load, memory={**load.memory, **probe.summarize(samples, 0, load.memory.get("loaded_at_s"))})
    stop = {**stop, "pressure_guard_tripped": sampler.pressure_tripped if sampler else None,
            "sampler_error": sampler.error if sampler else "sampler not started",
            "memory_whole_run": probe.summarize(samples), "available_after_stop": psutil.virtual_memory().available}
    try:
        (out_dir / "samples.json").write_text(json.dumps(probe.as_dicts(samples)), encoding="utf-8")
    except OSError as exc:
        stop = {**stop, "samples_error": bl._err(exc)}
    return load, stop


def execute(case: bl.Case, workload: bl.BenchWorkload, pre: bl.Preflight, out_dir: Path,
            opts: Options) -> dict[str, Any]:
    """Launch, measure and always kill the child; the returned fields complete the results document.
    The load result and every finished rep are kept when a later step fails or the operator interrupts."""
    api_key = secrets.token_urlsafe(24)
    launch, engine_info = prepare_launch(case, out_dir, pre.port, api_key)
    proc, t_start = start_child(launch)
    client: httpx.Client | None = None
    sampler: probe.ProcessSampler | None = None
    load: bl.LoadResult | None = None
    reps: list[bl.RepResult] = []
    error, interrupted = None, False
    try:
        new_sampler = probe.ProcessSampler(pid=proc.pid, min_available=int(opts.min_available_gib * GIB),
                                           on_pressure=proc.kill)
        observer = ChildObserver(proc.pid, new_sampler, time.perf_counter(), launch.stats_path, case.uses_cuda)
        sampler = new_sampler.start()
        client = httpx.Client(base_url=f"http://127.0.0.1:{pre.port}", headers={"Authorization": f"Bearer {api_key}"})
        for item in measure(client, proc, case, workload, observer, t_start, Path(launch.meta["log"])):
            if isinstance(item, bl.LoadResult):
                load = item
            else:
                reps.append(item)
    except KeyboardInterrupt:
        error, interrupted = "interrupted by the operator", True
    except Exception as exc:  # every failure is evidence: recorded in bench.json, the child is still killed below
        error = bl._err(exc)
    finally:
        stop, samples = _shutdown(proc, client, sampler)
    load, stop = _finish(load, stop, samples, sampler, out_dir)
    status = "interrupted" if interrupted else bl.run_status(load, reps, workload.reps)
    if error and status == "ok":
        status = "failed"
    build = (load.build if load else None) or {}
    return {"status": status, "engine": {**engine_info, **build}, "launch": launch.meta, "load": load, "reps": reps,
            "stop": stop, "error": error}


def new_out_dir(label: str, started: datetime) -> Path:
    """results/capacity/<UTC stamp>-bench-<label>, created here; a suffix keeps an earlier run's evidence when two
    runs of one label start in the same second (refused and dry runs take no time at all)."""
    base = f"{started.strftime('%Y%m%dT%H%M%SZ')}-bench-{label}"
    for n in range(1, 1000):
        path = RESULTS / (base if n == 1 else f"{base}-{n}")
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            continue
        return path
    raise bl.BenchError(f"no free results directory for {base}")


def run_case(case: bl.Case, workload: bl.BenchWorkload, opts: Options) -> tuple[Path, str]:
    started = datetime.now(UTC)
    out_dir = new_out_dir(case.label, started)
    host = host_info()
    base: dict[str, Any] = {"case": case, "workload": workload, "started_at": started.isoformat(), "host": host}
    try:
        paths = tuple(Path(p) for p in case.model_paths)
        model = model_identity(paths)
        pre = preflight(case, model_facts(paths), opts.port or free_loopback_port())
    except (bl.BenchError, gguf.GgufError, OSError) as exc:
        doc = bl.results_document(**base, status="refused", model={}, preflight=None, refused=[bl._err(exc)])
        return write(out_dir, doc), "refused"
    refused = bl.refusals(pre)
    if refused or opts.dry_run:
        status = "refused" if refused else "dry_run"
        doc = bl.results_document(**base, status=status, model=model, preflight=pre, refused=refused)
        return write(out_dir, doc), status
    try:
        fields = execute(case, workload, pre, out_dir, opts)
    except Exception as exc:  # launch preparation failed before any child existed; recorded, not raised
        fields = {"status": "failed", "error": bl._err(exc)}
    doc = bl.results_document(**base, model=model, preflight=pre, **fields)
    return write(out_dir, doc), doc["status"]


def write(out_dir: Path, doc: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bench.json"
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------------- CLI


def _unique_labels(cases: list[bl.Case]) -> list[bl.Case]:
    labels = [c.label for c in cases]
    dupes = sorted({label for label in labels if labels.count(label) > 1})
    if dupes:
        raise bl.BenchError(f"case labels must be unique; repeated: {', '.join(dupes)}")
    return cases


def cases_from_args(args: argparse.Namespace) -> list[bl.Case]:
    if args.case_file:
        data = json.loads(Path(args.case_file).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise bl.BenchError("--case-file must hold a non-empty JSON list of cases")
        return _unique_labels([bl.case_from_dict(item) for item in data])
    if not args.preset or not args.model:
        raise bl.BenchError("give --case-file, or at least one --preset and one --model")
    cases = []
    for model in args.model:
        paths = model_shards(model)
        n_layer = model_facts(paths).n_layer
        for name in args.preset:
            label = f"{args.label + '-' if args.label else ''}{paths[0].parent.name}-{name}"
            cases.append(case_for_preset(parse_preset(name), [str(p) for p in paths], n_layer, label=label,
                                         ctx=args.ctx, threads=args.threads, windows=os.name == "nt", gpu=args.gpu))
    return _unique_labels(cases)


def workload_from_args(args: argparse.Namespace) -> bl.BenchWorkload:
    return bl.BenchWorkload(prompt=args.prompt or bl.DEFAULT_PROMPT, prompt_tokens=args.prompt_tokens,
                            output_tokens=args.output_tokens, reps=args.reps, warmup=args.warmup,
                            reuse_output_tokens=args.reuse_output_tokens, seed=args.seed,
                            request_timeout_s=args.request_timeout, load_timeout_s=args.load_timeout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-file", help="JSON list of cases (overrides --preset/--model)")
    ap.add_argument("--preset", action="append", help="stock-cpu, stock-gpu, gpu-ram-N, stream-rR-sS-kK (repeatable)")
    ap.add_argument("--model", action="append", help="model dir name under ~/.hypercompute/models, or a path")
    ap.add_argument("--label", default="", help="prefix for preset case labels")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gpu", default="CUDA0", help="engine device for GPU units in presets")
    ap.add_argument("--prompt", help="prompt text (default: a fixed English paragraph)")
    ap.add_argument("--prompt-tokens", type=int, help="token target: the prompt is repeated and cut to this length")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3, help=f"measured reps (at least {bl.MIN_MEASURED_REPS})")
    ap.add_argument("--warmup", type=int, default=1, help=f"unmeasured reps first (at least {bl.MIN_WARMUP_REPS})")
    ap.add_argument("--reuse-output-tokens", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=0, help="child port (default: a free loopback port)")
    ap.add_argument("--min-available-gib", type=float, default=2.0, help="kill the child below this much free RAM")
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="resolve cases and run the preflight only")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        opts = Options(port=args.port, min_available_gib=args.min_available_gib, dry_run=args.dry_run)
        workload = workload_from_args(args)
        cases = cases_from_args(args)
    except (bl.BenchError, gguf.GgufError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {bl._err(exc)}", file=sys.stderr)
        return 2
    codes = {"ok": 0, "dry_run": 0, "partial": 1, "failed": 1, "refused": 2, "interrupted": 130}
    worst = 0
    for case in cases:
        path, status = run_case(case, workload, opts)
        print(f"[{case.label}] {status}: {path.relative_to(ROOT).as_posix()}", flush=True)
        worst = max(worst, codes.get(status, 1))
        if status == "interrupted":
            break
    return worst


if __name__ == "__main__":
    sys.exit(main())
