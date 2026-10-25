"""Capacity canary: run an existing GGUF in an isolated, CPU-only llama-server child and measure what it really used.

Modes (same model, flags, prompt and token budget; only weight residency differs):
  stock     stock engine, file-backed weights with the loader's PrefetchVirtualMemory (OS decides residency)
  noprefetch  hcstream overlay with no managed layers: file-backed weights, mapping prefetch switched off
  resident  stock engine, --load-mode none: every CPU weight copied into private memory
  stream    hcstream overlay: a resident subset in private memory, the rest read per pass into bounded slots

  uv run python scripts/capacity/canary.py --model ~/.hypercompute/models/qwen2.5-0.5b-instruct-q4_k_m \
      --mode stream --stream-layers 0-23 --slots 2 --tokens 64 --reps 3

It never touches CUDA (--device none, -ngl 0), the live gateway, or the system file cache beyond what the chosen
mode itself reads. A memory guard kills the child if available RAM drops below --min-available-gib.
Output: one JSON evidence file under results/capacity/.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from hypercompute import engine  # noqa: E402
from hypercompute.capacity import overlay, probe  # noqa: E402
from hypercompute.capacity.stream_config import (  # noqa: E402
    StreamOptions,
    block_count,
    layer_layouts,
    render_config,
    stream_plan,
)
from hypercompute.gguf import read_tensor_table  # noqa: E402
from hypercompute.netutil import free_loopback_port  # noqa: E402

GIB = 1 << 30
PROMPT = ("You are a careful assistant. Explain, in plain sentences, why a laptop that reads part of a language model "
          "from its SSD for every token is slower than one that keeps every layer in memory, and what changes when "
          "a second computer with spare memory joins over a network.")


def parse_layers(spec: str | None, n_layer: int) -> list[int]:
    """'0-7,12,head' -> sorted unit ids; 'head' is the output unit (index n_layer)."""
    out: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part == "head":
            out.append(n_layer)
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def shards_of(model: Path) -> list[Path]:
    if model.is_file():
        return [model]
    files = sorted(p for p in model.glob("*.gguf") if not p.name.startswith("mmproj"))
    if not files:
        raise SystemExit(f"no GGUF files in {model}")
    return files


def launch(args: argparse.Namespace, out_dir: Path) -> tuple[subprocess.Popen, dict]:
    stock = engine.find_installed()
    if stock is None:
        raise SystemExit("no installed llama.cpp engine")
    shards = shards_of(Path(args.model).expanduser())
    port = free_loopback_port()
    meta: dict = {"shards": [str(s) for s in shards], "port": port, "mode": args.mode}
    env = {k: v for k, v in os.environ.items() if k != "HC_STREAM_CONFIG"}
    bin_dir = stock.bin_dir
    load_mode = "none" if args.mode == "resident" else "mmap"
    if args.mode in ("stream", "noprefetch"):
        dll = overlay.find_proxy_dll()
        if dll is None:
            raise SystemExit("build the proxy first: uv run --no-project --with ziglang python scripts/build_hcstream.py")
        ov = overlay.ensure_overlay(stock.bin_dir, dll)
        bin_dir = ov.bin_dir
        table = read_tensor_table(shards)
        layouts = layer_layouts(table)
        n_layer = block_count(table)
        stream = parse_layers(args.stream_layers, n_layer) if args.mode == "stream" else []
        resident = parse_layers(args.resident_layers, n_layer) if args.mode == "stream" else []
        plan = stream_plan(layouts, resident, stream, args.slots)
        stats_path = out_dir / f"hcstream-stats-{port}.json"
        cfg = render_config(shards, table, plan, stats_path,
                            StreamOptions(readers=args.readers, chunk_bytes=args.chunk_mib << 20,
                                          fault_read_error_after_bytes=args.fault_error_after_mib << 20,
                                          fault_read_delay_us=args.fault_delay_us))
        cfg_path = out_dir / f"hcstream-{port}.cfg"
        cfg_path.write_text(cfg, encoding="utf-8")
        env["HC_STREAM_CONFIG"] = str(cfg_path)
        meta.update(overlay=str(ov.bin_dir), proxy_sha256=ov.proxy_sha256, config=str(cfg_path),
                    stats_path=str(stats_path), stream_plan={
                        "resident_layers": list(plan.resident_layers), "stream_layers": list(plan.stream_layers),
                        "slots": plan.slots, "slot_bytes": plan.slot_bytes, "resident_bytes": plan.resident_bytes,
                        "stream_bytes_per_pass": plan.stream_bytes_per_pass, "committed_bytes": plan.committed_bytes})
    exe = bin_dir / engine.exe_name("llama-server")
    argv = [str(exe), "-m", str(shards[0]), "--load-mode", load_mode, "--no-repack", "--no-op-offload",
            "--cache-ram", "0", "--device", "none", "-ngl", "0", "--fit", "off", "-c", str(args.ctx), "-np", "1",
            "-t", str(args.threads), "--host", "127.0.0.1", "--port", str(port), "--no-ui", "--log-colors", "off"]
    log_path = out_dir / f"llama-server-{port}.log"
    meta.update(argv=argv, log=str(log_path), bin_dir=str(bin_dir))
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    log = log_path.open("wb")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(bin_dir), creationflags=flags)
    return proc, meta


def read_stats(meta: dict) -> dict | None:
    path = meta.get("stats_path")
    if not path:
        return None
    for _ in range(5):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)
    return None


def complete(client: httpx.Client, args: argparse.Namespace) -> dict:
    body = {"prompt": args.prompt, "n_predict": args.tokens, "temperature": 0, "top_k": 1, "seed": 7,
            "cache_prompt": False, "return_tokens": True, "stream": False, "ignore_eos": True}
    t0 = time.perf_counter()
    r = client.post("/completion", json=body, timeout=args.request_timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    timings = data.get("timings") or {}
    return {"wall_s": round(wall, 3), "tokens": data.get("tokens") or [], "content": data.get("content", ""),
            "prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms"),
            "predicted_n": timings.get("predicted_n"), "predicted_ms": timings.get("predicted_ms"),
            "predicted_per_second": timings.get("predicted_per_second")}


def cancel_probe(client: httpx.Client, args: argparse.Namespace) -> dict:
    """Disconnect a streaming request mid-generation, then prove the server still answers correctly."""
    body = {"prompt": args.prompt, "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 7, "cache_prompt": False,
            "stream": True, "ignore_eos": True}
    got = 0
    t0 = time.perf_counter()
    with client.stream("POST", "/completion", json=body, timeout=args.request_timeout) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                got += 1
                if got >= 4:
                    break
    disconnected_after = round(time.perf_counter() - t0, 3)
    time.sleep(1.0)
    health = client.get("/health", timeout=30).status_code
    return {"chunks_before_disconnect": got, "disconnected_after_s": disconnected_after, "health_after": health}


def stop(proc: subprocess.Popen) -> dict:
    t0 = time.perf_counter()
    proc.terminate()
    try:
        code = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = proc.wait(timeout=30)
    return {"exit_code": code, "stop_s": round(time.perf_counter() - t0, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GGUF file or directory of split shards")
    ap.add_argument("--mode", choices=["stock", "noprefetch", "resident", "stream"], required=True)
    ap.add_argument("--stream-layers", help="e.g. 0-23 or 4,6,8-11 (stream mode)")
    ap.add_argument("--resident-layers", help="layers copied once into private memory (stream mode)")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--readers", type=int, default=4)
    ap.add_argument("--chunk-mib", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--cancel-test", action="store_true")
    ap.add_argument("--min-available-gib", type=float, default=1.0)
    ap.add_argument("--load-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--fault-error-after-mib", type=int, default=0)
    ap.add_argument("--fault-delay-us", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / "capacity" / f"{stamp}-{args.mode}{('-' + args.label) if args.label else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"started_at": datetime.now(UTC).isoformat(), "args": vars(args),
                      "host": {"cpu_count": psutil.cpu_count(), "ram_total": psutil.virtual_memory().total,
                               "available_before": psutil.virtual_memory().available}}
    proc, meta = launch(args, out_dir)
    evidence["launch"] = meta
    sampler = probe.ProcessSampler(pid=proc.pid, min_available=int(args.min_available_gib * GIB),
                                   on_pressure=proc.kill).start()
    marks: dict[str, float] = {}
    client = httpx.Client(base_url=f"http://127.0.0.1:{meta['port']}")
    code = 0
    try:
        t_load = time.perf_counter()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during load with code {proc.returncode}")
            try:
                if client.get("/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - t_load > args.load_timeout:
                raise RuntimeError("load timed out")
            time.sleep(0.25)
        evidence["load_s"] = round(time.perf_counter() - t_load, 3)
        marks["loaded"] = sampler.samples[-1].t if sampler.samples else 0.0
        evidence["stats_after_load"] = read_stats(meta)
        runs = []
        for i in range(args.reps + 1):
            before = read_stats(meta)
            t_start = sampler.samples[-1].t if sampler.samples else 0.0
            result = complete(client, args)
            after = read_stats(meta)
            t_end = sampler.samples[-1].t if sampler.samples else 0.0
            result["warmup"] = i == 0
            result["memory"] = probe.summarize(sampler.samples, t_start, t_end)
            if before and after:
                passes = after["passes"] - before["passes"]
                result["hcstream"] = {k: after[k] - before[k] for k in (
                    "bytes_read", "read_ns", "stall_ns", "layer_loads", "sync_loads", "prefetch_loads",
                    "prefetch_hits", "passes", "io_errors")}
                result["hcstream"]["bytes_per_pass"] = (after["bytes_read"] - before["bytes_read"]) // passes if passes else None
            runs.append(result)
            print(f"rep {i}{' (warmup)' if i == 0 else ''}: {result['predicted_n']} tokens, "
                  f"{result['predicted_per_second']} tok/s decode, prompt {result['prompt_ms']} ms, "
                  f"ws peak {result['memory'].get('working_set_peak', 0) / GIB:.2f} GiB, "
                  f"disk {result['memory'].get('sys_disk_read_bytes', 0) / GIB:.2f} GiB", flush=True)
        evidence["runs"] = runs
        measured = [r for r in runs if not r["warmup"]]
        token_sets = {tuple(r["tokens"]) for r in runs}
        evidence["deterministic_tokens"] = len(token_sets) == 1
        evidence["tokens"] = runs[0]["tokens"]
        evidence["content"] = runs[0]["content"]
        evidence["summary"] = {
            "decode_tps_median": statistics.median(r["predicted_per_second"] or 0 for r in measured),
            "decode_tps_min": min(r["predicted_per_second"] or 0 for r in measured),
            "decode_tps_max": max(r["predicted_per_second"] or 0 for r in measured),
            "prompt_ms_median": statistics.median(r["prompt_ms"] or 0 for r in measured),
            "wall_s_median": statistics.median(r["wall_s"] for r in measured),
        }
        if args.cancel_test:
            evidence["cancel"] = cancel_probe(client, args)
            evidence["after_cancel"] = complete(client, args)
            evidence["after_cancel_matches"] = evidence["after_cancel"]["tokens"] == evidence["tokens"]
    except Exception as exc:  # noqa: BLE001 - every failure is evidence
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        client.close()
        evidence["stats_final"] = read_stats(meta)
        evidence["stop"] = stop(proc) if proc.poll() is None else {"exit_code": proc.returncode, "stop_s": 0}
        time.sleep(1.0)
        samples = sampler.stop()
        evidence["pressure_guard_tripped"] = sampler.pressure_tripped
        evidence["sampler_error"] = sampler.error
        evidence["memory_whole_run"] = probe.summarize(samples)
        evidence["memory_load"] = probe.summarize(samples, 0, marks.get("loaded"))
        evidence["available_after_stop"] = psutil.virtual_memory().available
        evidence["child_alive_after_stop"] = psutil.pid_exists(proc.pid)
        (out_dir / "samples.json").write_text(json.dumps(probe.as_dicts(samples)), encoding="utf-8")
        (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"evidence: {(out_dir / 'evidence.json').relative_to(ROOT).as_posix()}")
    return code


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sys.exit(main())
