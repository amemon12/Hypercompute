"""Parsers for llama.cpp's own output at release b11770.

Formats were taken from real runs on the pinned release (see tests/fixtures/*_b11770.*):
  --version       -> "version: 0.7.0-dev (build 11770, commit 39da130fc)"
  --list-devices  -> "  CUDA0: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12226 MiB, 11026 MiB free)"
                     "  RPC0: 127.0.0.1:47052 (31864 MiB, 4572 MiB free)"
  rpc banner      -> "Starting RPC server v8.0.0" / "  endpoint       : 127.0.0.1:47052"
  load log (-lv 4)-> "load_tensors: RPC0[127.0.0.1:47052] model buffer size =   244.33 MiB"
                     "llama_kv_cache:      CUDA0 KV buffer size =    26.00 MiB"
                     "sched_reserve: RPC0[127.0.0.1:47052] compute buffer size =    56.79 MiB"
  -ot overrides   -> "tensor blk.3.ffn_up_exps.weight (531 MiB mxfp4) buffer type overridden to CPU" (debug level;
                     the format string "tensor %s (%zu MiB %s) buffer type overridden to %s" in llama.dll at b11770,
                     not yet seen in a real MoE load log on this host). Debug is -lv 5 ("4: trace, 5: debug" in the
                     llama-server help); the adapter runs -lv 4, whose real logs carry no debug lines at all
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..gguf import EXPERT_TENSOR_RE
from ..schemas import DeviceAssignment, EngineDevice, ObservedDevice, ObservedPlacement, Plan, utcnow
from ..units import MiB

_VERSION_RE = re.compile(r"version:\s*(?P<ver>\S+)\s*\(build\s+(?P<build>\d+),\s*commit\s+(?P<commit>[0-9a-f]+)\)")
_DEVICE_RE = re.compile(
    r"^\s+(?P<name>[A-Za-z_]+\d*(?:\[[^\]]*\])?):\s+(?P<desc>.*?)\s+\((?P<total>\d+) MiB, (?P<free>\d+) MiB free\)\s*$"
)
_RPC_VERSION_RE = re.compile(r"Starting RPC server v(?P<proto>\d+\.\d+\.\d+)")
_RPC_ENDPOINT_RE = re.compile(r"^\s*endpoint\s*:\s*(?P<endpoint>\S+)", re.MULTILINE)
_RPC_TRANSPORT_RE = re.compile(r"^\s*transport\s*:\s*(?P<t>\S+)", re.MULTILINE)
_RPC_CACHE_RE = re.compile(r"^\s*local cache\s*:\s*(?P<c>.+?)\s*$", re.MULTILINE)

_BUFFER_RE = re.compile(
    r"(?P<prefix>load_tensors|llama_kv_cache|sched_reserve):\s+(?P<name>\S+?)\s+"
    r"(?P<kind>model|KV|compute) buffer size\s*=\s*(?P<mib>[\d.]+) MiB"
)
_OFFLOADED_RE = re.compile(r"offloaded (?P<n>\d+/\d+) layers to GPU")
_KV_TOTAL_RE = re.compile(r"llama_kv_cache: size =\s*(?P<mib>[\d.]+) MiB \(\s*\d+ cells,\s*(?P<layers>\d+) layers")
_OVERRIDE_RE = re.compile(r"tensor (?P<name>\S+) \(\d+ MiB [^)]*\) buffer type overridden to (?P<buft>\S+)")

FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Failed to connect to", "a worker endpoint refused or dropped the RPC connection"),
    ("RPC server version mismatch", "worker runs a different llama.cpp RPC protocol version"),
    ("RPC handshake failed", "RPC handshake failed (version or transport mismatch)"),
    ("failed to load multimodal model", "llama-server could not load the vision projector (--mmproj): usually out "
     "of memory on its device, or a projector that does not belong to the model"),
    ("failed to load model", "llama.cpp could not load the model"),
    ("error loading model", "llama.cpp could not load the model"),
    ("out of memory", "a device ran out of memory while loading"),
    ("failed to allocate", "a device buffer allocation failed"),
    ("cudaMalloc failed", "CUDA allocation failed"),
    ("GGML_ABORT", "llama.cpp aborted"),
    ("Failed to create server socket", "the worker could not bind its port"),
    ("hcstream: FATAL", "the hcstream weight-streaming extension stopped the engine"),
)

# Host-side buffers that are not accelerator allocations.
HOST_BUFFER_NAMES = ("CPU", "CPU_Mapped", "CPU_REPACK", "CUDA_Host", "Metal_Mapped", "CPU_AARCH64")
# Host weight buffers that are private copies (commit), not views of the memory-mapped GGUF.
PRIVATE_HOST_WEIGHTS = ("CPU", "CPU_REPACK", "CUDA_Host", "CPU_AARCH64")
# Buffer-type names a local accelerator's own buffers carry at b11770, after its device name. ggml-metal.cpp names
# MTLn (shared), MTLn_Private and MTLn_Mapped; with mmap a Metal model buffer is MTLn_Mapped (buffer_from_host_ptr wraps
# the mapped GGUF ranges), while its KV and compute buffers keep MTLn. CUDA logs CUDAn only, so its sums are unchanged.
ACCEL_BUFFER_SUFFIXES = ("", "_Mapped", "_Private")
MOE_PRIVATE_TOLERANCE = 256 * MiB  # private host weight bytes a moe_offload load may show (none expected)
MOE_GPU_SLACK = 64 * MiB  # MiB rounding and alignment between planned and logged accelerator weight bytes
MOE_GPU_LOW = 0.9  # least share of its planned weights the GPU must log (the 14B CPU-tier log shows 96 percent)


@dataclass(frozen=True)
class VersionInfo:
    version: str
    build_number: int
    commit_short: str
    line: str


@dataclass(frozen=True)
class RpcBanner:
    proto: str | None
    endpoint: str | None
    transport: str | None
    cache: str | None
    devices: list[EngineDevice] = field(default_factory=list)
    bind_failed: bool = False


def parse_version(text: str) -> VersionInfo | None:
    m = _VERSION_RE.search(text)
    if not m:
        return None
    return VersionInfo(version=m["ver"], build_number=int(m["build"]), commit_short=m["commit"], line=m.group(0))


def _device_from_match(m: re.Match[str]) -> EngineDevice:
    return EngineDevice(name=m["name"], description=m["desc"].strip(), total_bytes=int(m["total"]) * MiB,
                        free_bytes=int(m["free"]) * MiB)


def parse_list_devices(text: str) -> list[EngineDevice]:
    """Device lines printed after 'Available devices:' by `llama-server --list-devices`."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Available devices:")
    except StopIteration:
        return []
    out: list[EngineDevice] = []
    for line in lines[start + 1:]:
        m = _DEVICE_RE.match(line)
        if m:
            out.append(_device_from_match(m))
        elif line.strip() and not line.startswith(" "):
            break
    return out


def parse_rpc_banner(text: str) -> RpcBanner:
    proto = _RPC_VERSION_RE.search(text)
    endpoint = _RPC_ENDPOINT_RE.search(text)
    transport = _RPC_TRANSPORT_RE.search(text)
    cache = _RPC_CACHE_RE.search(text)
    devices: list[EngineDevice] = []
    in_devices = False
    for line in text.splitlines():
        if line.strip() == "Devices:":
            in_devices = True
            continue
        if in_devices:
            m = _DEVICE_RE.match(line)
            if m:
                devices.append(_device_from_match(m))
            elif not line.startswith("  ") or ":" not in line:
                in_devices = False
    return RpcBanner(
        proto=proto["proto"] if proto else None,
        endpoint=endpoint["endpoint"] if endpoint else None,
        transport=transport["t"] if transport else None,
        cache=cache["c"] if cache else None,
        devices=devices,
        bind_failed="Failed to create server socket" in text,
    )


def detect_failure(text: str) -> str | None:
    for needle, meaning in FAILURE_PATTERNS:
        idx = text.find(needle)
        if idx >= 0:
            line_start = text.rfind("\n", 0, idx) + 1
            line_end = text.find("\n", idx)
            line = text[line_start: line_end if line_end >= 0 else None].strip()
            return f"{meaning}: {line[:300]}"
    return None


def parse_placement(text: str) -> ObservedPlacement:
    model: dict[str, int] = {}
    kv: dict[str, int] = {}
    compute: dict[str, int] = {}
    evidence: list[str] = []
    order: list[str] = []
    for line in text.splitlines():
        m = _BUFFER_RE.search(line)
        if not m:
            continue
        name, size = m["name"], int(float(m["mib"]) * MiB)
        bucket = {"model": model, "KV": kv, "compute": compute}[m["kind"]]
        bucket[name] = bucket.get(name, 0) + size
        if name not in order:
            order.append(name)
        evidence.append(line[line.find(m["prefix"]):].strip())
    offloaded = _OFFLOADED_RE.search(text)
    kv_total = _KV_TOTAL_RE.search(text)
    per_layer_kv: float | None = None
    if kv_total and int(kv_total["layers"]) > 0:
        per_layer_kv = float(kv_total["mib"]) * MiB / int(kv_total["layers"])
    if offloaded:
        evidence.insert(0, f"load_tensors: offloaded {offloaded['n']} layers to GPU")
    devices = [
        ObservedDevice(
            name=name,
            model_bytes=model.get(name, 0),
            kv_bytes=kv.get(name, 0),
            compute_bytes=compute.get(name, 0),
            layers_est=(round(kv[name] / per_layer_kv) if per_layer_kv and name in kv else None),
        )
        for name in order
    ]
    return ObservedPlacement(offloaded=offloaded["n"] if offloaded else None, devices=devices,
                             evidence_lines=evidence, parsed_at=utcnow())


def rpc_device_endpoint(name: str) -> str | None:
    """'RPC0[10.0.0.5:47052]' -> '10.0.0.5:47052'."""
    m = re.match(r"^RPC\d*\[(?P<ep>[^\]]+)\]$", name)
    return m["ep"] if m else None


def _rpc_number(name: str) -> int | None:
    m = re.fullmatch(r"RPC(\d+)", name)
    return int(m[1]) if m else None


def _server_index(a: DeviceAssignment, plan: Plan, by_endpoint: dict[str, list[str]]) -> int:
    """Index of a planned peer device within its RPC server, which llama.cpp puts in its buffer names.

    Its position in the probe's device list for the endpoint; without one, its rank by RPC number among the plan's
    devices on that endpoint (exact when the plan uses all of that server's devices).
    """
    actual = by_endpoint.get(a.endpoint or "") or []
    if a.device in actual:
        return actual.index(a.device)
    same = sorted((x.device for x in plan.assignments if x.endpoint == a.endpoint),
                  key=lambda n: (_rpc_number(n) is None, _rpc_number(n) or 0, n))
    return same.index(a.device)


def accel_buffer_names(device: str, observed: Mapping[str, ObservedDevice]) -> list[str]:
    """The observed buffer names of a local accelerator: MTL0 also as MTL0_Mapped and MTL0_Private (Metal at b11770);
    the host CPU's own names (CPU, CPU_Mapped, ...) never count as an accelerator's."""
    if device == "CPU":
        return [device] if device in observed else []
    return [name for name in (device + s for s in ACCEL_BUFFER_SUFFIXES) if name in observed]


def accel_model_bytes(device: str, observed: Mapping[str, ObservedDevice]) -> int | None:
    """Model bytes llama.cpp logged for a local accelerator under any of its buffer names; None when it logged none."""
    names = accel_buffer_names(device, observed)
    return sum(observed[n].model_bytes for n in names) if names else None


def check_participation(plan: Plan, placement: ObservedPlacement,
                        by_endpoint: dict[str, list[str]] | None = None) -> tuple[bool, str]:
    """Every planned peer device must show a nonzero model buffer in llama.cpp's own load log.

    Peer buffers are named RPC<index within the server>[endpoint], so each planned device is matched by exactly
    that name and two devices of one peer cannot stand in for each other: a worker started with -d MTL0,CPU logs its
    Metal stage as RPC0[ep] and its CPU stage as RPC1[ep] (ggml-rpc.cpp buffer types at b11770), file-backed by hcpeer
    or not. A local Metal device's model buffer is MTL0_Mapped under mmap while its KV and compute buffers are MTL0
    (accel_model_bytes adds them up).
    """
    observed = {d.name: d for d in placement.devices}
    lines, ok = [], True
    for a in plan.assignments:
        key = f"RPC{_server_index(a, plan, by_endpoint or {})}[{a.endpoint}]" if a.endpoint else a.device
        names = [key] if a.endpoint else accel_buffer_names(key, observed)
        dev = observed.get(names[0]) if names else None
        if dev is None:
            ok = ok and not a.endpoint
            lines.append(f"{a.device}: no buffers observed")
            continue
        planned = len(a.layers)
        model = dev.model_bytes if a.endpoint else accel_model_bytes(key, observed) or 0
        lines.append(f"{dev.name}: model {model / MiB:.0f} MiB, KV {dev.kv_bytes / MiB:.0f} MiB, "
                     f"~{dev.layers_est} layers (planned {planned}" + (", file-backed)" if a.file_backed else ")"))
        if a.endpoint and dev.model_bytes <= 0:
            ok = False
    if not any(a.endpoint for a in plan.assignments):
        return False, "local-only deployment (no peers in the plan); " + "; ".join(lines)
    return ok, "; ".join(lines)


def parse_overrides(text: str) -> dict[str, str]:
    """Tensor name -> buffer type of every tensor llama.cpp moved by -ot. llama.cpp logs these at debug level, which
    is -lv 5 at b11770 (its help: 4 trace, 5 debug); the -lv 4 logs the adapter writes have none, so this is empty."""
    return {m["name"]: m["buft"] for m in _OVERRIDE_RE.finditer(text)}


def _override_problems(planned: set[int], overrides: Mapping[str, str]) -> tuple[list[str], str]:
    """The per-tensor evidence when the log has it (-lv 5): the expert tensors of exactly the planned layers, and
    nothing else, overridden to CPU. Without it the buffer sizes are the evidence (check_moe_placement)."""
    if not overrides:
        return [], ("no -ot override lines (debug level, -lv 5), so the expert placement is checked by buffer sizes: "
                    "CPU experts left on the GPU would push its model buffer over the plan")
    moved = {int(m[1]) for name, buft in overrides.items() if buft == "CPU" and (m := EXPERT_TENSOR_RE.match(name))}
    stray = sorted(name for name, buft in overrides.items() if buft != "CPU" or not EXPERT_TENSOR_RE.match(name))
    problems: list[str] = []
    if moved != planned:
        problems.append(f"experts of {len(moved)} layer(s) were overridden to CPU, planned {len(planned)} (missing "
                        f"{sorted(planned - moved)[:8]}, extra {sorted(moved - planned)[:8]})")
    if stray:
        problems.append(f"unexpected overrides: {', '.join(stray[:4])}")
    return problems, f"experts of {len(moved)} layer(s) overridden to CPU"


def check_moe_placement(plan: Plan, placement: ObservedPlacement, overrides: Mapping[str, str]) -> tuple[bool, str]:
    """A moe_offload load did what the plan says: the CPU_Mapped buffers span at least the planned CPU expert bytes and
    no private host weight buffer (repacked or pinned copy, which is commit) holds them; the local GPU holds its
    planned weights, never more than the plan plus less than half a CPU layer's experts, so no CPU expert landed
    there; with debug-level override lines, exactly the planned layers' experts moved. Peers: check_participation.
    The GPU's weights are its model buffers under every name it logs: CUDA0 on Windows, MTL0_Mapped on a Mac (mmap
    wraps the GGUF ranges of the dense parts and GPU experts; the CPU experts stay in CPU_Mapped as on Windows)."""
    moe = plan.moe
    gpu = next((a for a in plan.assignments if a.endpoint is None and a.holds_output), None)
    if moe is None or gpu is None or not moe.cpu_expert_layers:
        return False, "a moe_offload plan without an MoE layout or a local GPU holding the head"
    by = {d.name: d for d in placement.devices}
    planned = set(moe.cpu_expert_layers)
    problems, moved_note = _override_problems(planned, overrides)
    mapped = by["CPU_Mapped"].model_bytes if "CPU_Mapped" in by else 0
    private = sum(by[name].model_bytes for name in PRIVATE_HOST_WEIGHTS if name in by)
    on_gpu = accel_model_bytes(gpu.device, by)
    # under half of one CPU layer's experts, so one layer the -ot pattern missed shows; at least the MiB log rounding
    slack = min(MOE_GPU_SLACK, max(MiB, moe.cpu_expert_bytes // (2 * len(planned))))
    if mapped < 0.99 * moe.cpu_expert_bytes:
        problems.append(f"CPU_Mapped spans {mapped / MiB:.0f} MiB, less than the {moe.cpu_expert_bytes / MiB:.0f} "
                        "MiB of planned CPU experts")
    if private > MOE_PRIVATE_TOLERANCE:
        problems.append(f"{private / MiB:.0f} MiB of weights in private host buffers (repacked or pinned copies)")
    if placement.offloaded is not None and len(set(placement.offloaded.split("/"))) != 1:
        problems.append(f"llama.cpp offloaded {placement.offloaded} layers, planned all of them")
    # the dense parts on the GPU: the planned weight bytes match the logged buffer to within MiB rounding on real
    # b11770 loads (results/*local_only*: 4168.08 planned, 4168.09 logged), so the upper slack stays small
    if on_gpu is None or not MOE_GPU_LOW * gpu.weight_bytes <= on_gpu <= gpu.weight_bytes + slack:
        problems.append(f"{gpu.device} holds {'no' if on_gpu is None else f'{on_gpu / MiB:.0f} MiB of'} weights, "
                        f"planned {gpu.weight_bytes / MiB:.0f} MiB")
    summary = (f"{gpu.device} model {(on_gpu or 0) / MiB:.0f} MiB (planned {gpu.weight_bytes / MiB:.0f}), CPU_Mapped "
               f"{mapped / MiB:.0f} MiB, private host weights {private / MiB:.0f} MiB, {len(planned)} CPU expert "
               f"layer(s) planned; {moved_note}")
    if problems:
        return False, "moe_offload placement differs from the plan: " + "; ".join(problems) + "; " + summary
    return True, "moe_offload verified: " + summary
