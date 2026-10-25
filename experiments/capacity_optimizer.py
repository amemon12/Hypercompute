"""Offline contiguous-placement research prototype; never deploys or loads model weights.

Costs require calibrated inputs. Example fixtures are illustrative, NOT throughput measurements.
The search is optimal only inside its restricted CPU-prefix / peer-spans / local-head topology,
for the additive cost model supplied. It does not provide a physical memory limit, an engine,
live migration, tensor parallelism, or proof that a predicted paging configuration will run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class Model:
    layer_bytes: tuple[int, ...]
    embedding_bytes: int
    output_bytes: int
    kv_bytes_per_layer: int
    activation_bytes_per_token: int

    def __post_init__(self):
        if not self.layer_bytes or min(self.layer_bytes) <= 0:
            raise ValueError("positive layer sizes are required")
        if min(self.embedding_bytes, self.output_bytes, self.kv_bytes_per_layer,
               self.activation_bytes_per_token) < 0:
            raise ValueError("model byte counts must be nonnegative")


@dataclass(frozen=True)
class Device:
    name: str
    pool: str
    capacity_bytes: int  # fresh allocation budget AFTER OS/user reserve and other leases
    workspace_bytes: int
    effective_read_gib_s: float  # model-specific measured coefficient, not advertised peak
    rtt_ms: float = 0
    link_mib_s: float = 0
    cached_fraction: float = 0  # verified upload cache only; NOT an inference cache

    def __post_init__(self):
        numbers = (self.effective_read_gib_s, self.rtt_ms, self.link_mib_s, self.cached_fraction)
        if any(not math.isfinite(n) for n in numbers):
            raise ValueError("device coefficients must be finite")
        if (not self.name or not self.pool or self.capacity_bytes < 0 or self.workspace_bytes < 0
                or self.effective_read_gib_s <= 0 or self.rtt_ms < 0 or self.link_mib_s < 0
                or not 0 <= self.cached_fraction <= 1):
            raise ValueError("invalid device coefficient or allocation budget")


@dataclass(frozen=True)
class Assignment:
    device: str
    start_layer: int
    end_layer: int  # exclusive; empty CPU span still owns embeddings
    weight_bytes: int
    kv_bytes: int
    workspace_bytes: int
    resident_bytes_est: int
    disk_read_bytes_low: int
    disk_read_bytes_high: int
    compute_s: float
    network_s: float
    disk_s_low: float
    disk_s_high: float
    cold_upload_s: float

    @property
    def decode_s_high(self):
        return self.compute_s + self.network_s + self.disk_s_high


@dataclass(frozen=True)
class Candidate:
    assignments: tuple[Assignment, ...]
    score_s: float
    expected_decode_tokens: int
    include_cold_upload: bool

    def report(self):
        low = sum(a.compute_s + a.network_s + a.disk_s_low for a in self.assignments)
        high = sum(a.decode_s_high for a in self.assignments)
        return {
            "evidence": "cost-model estimate only; not a benchmark or executable deployment",
            "objective": "conservative decode cost times horizon, optionally plus remote cold upload",
            "scope": "CPU prefix, optional contiguous peer spans, local CUDA suffix/output head",
            "score_s": self.score_s,
            "expected_decode_tokens": self.expected_decode_tokens,
            "include_cold_upload": self.include_cold_upload,
            "estimated_decode_s_range": [low, high],
            "estimated_disk_bytes_per_token_range": [
                sum(a.disk_read_bytes_low for a in self.assignments),
                sum(a.disk_read_bytes_high for a in self.assignments),
            ],
            "remote_upload_s_est": sum(a.cold_upload_s for a in self.assignments),
            "assignments": [asdict(a) for a in self.assignments],
            "can_apply_to_running_engine": False,
            "excluded_costs": ["prefill", "thermal changes", "queueing", "host model loading",
                               "KV reconstruction", "shared-link contention", "kernel launch overhead"],
        }


def segment(model: Model, device: Device, start: int, end: int, *, cpu: bool = False,
            output: bool = False, allow_disk: bool = True, disk_gib_s: float = 1,
            cache_mode: str = "os") -> Assignment | None:
    """Admit one span. OS paging has a range: cyclic scans may miss on the entire working set.

    fixed_resident assumes an IMPLEMENTED explicitly retained subset; it is a proposed policy,
    not a description of today's mmap/AirLLM behavior. Cold OS caches can read more even on a
    resident plan. These estimates describe warm steady-state weight I/O, not initial loads.
    """
    if cache_mode not in {"os", "fixed_resident"}:
        raise ValueError("unknown cache mode")
    if not math.isfinite(disk_gib_s) or disk_gib_s <= 0:
        raise ValueError("positive finite disk throughput required")
    if not 0 <= start <= end <= len(model.layer_bytes):
        raise ValueError("invalid layer span")
    if cpu and start != 0:
        raise ValueError("CPU must own a prefix")
    layers = model.layer_bytes[start:end]
    weights = sum(layers) + (model.output_bytes if output else 0)
    embeddings = model.embedding_bytes if cpu else 0
    kv = len(layers) * model.kv_bytes_per_layer
    fixed = embeddings + kv + device.workspace_bytes
    room = device.capacity_bytes - fixed
    if room < 0:
        return None
    overflow = max(0, weights - room)
    if overflow and (not cpu or not allow_disk):
        return None
    high = weights if overflow and cache_mode == "os" else overflow
    # Embedding lookup does not scan the whole embedding matrix per decoded token.
    compute = (weights + (model.activation_bytes_per_token if cpu else 0)) / GIB / device.effective_read_gib_s
    network = upload = 0.0
    if not cpu and not output and layers:
        if device.link_mib_s <= 0:
            return None
        # Host-mediated RPC: send activation to peer and retrieve boundary activation.
        network = device.rtt_ms / 1000 + 2 * model.activation_bytes_per_token / MIB / device.link_mib_s
        upload = weights * (1 - device.cached_fraction) / MIB / device.link_mib_s
    return Assignment(device.name, start, end, weights + embeddings, kv, device.workspace_bytes,
                      fixed + min(weights, room), overflow, high, compute, network,
                      overflow / GIB / disk_gib_s, high / GIB / disk_gib_s, upload)


def optimize(model: Model, cpu: Device, local_gpu: Device, peers: list[Device], *,
             expected_decode_tokens: int = 128, include_cold_upload: bool = False,
             allow_disk: bool = True, disk_gib_s: float = 1, cache_mode: str = "os") -> Candidate | None:
    """Dynamic programming over layer cuts; enumerate peer order for at most four peers.

    Optional devices may receive no layers. Never double-count a UMA pool as CPU plus GPU:
    this prototype deliberately rejects duplicate pools; a production solver must share constraints.
    """
    devices = [cpu, local_gpu, *peers]
    if len({d.name for d in devices}) != len(devices) or len({d.pool for d in devices}) != len(devices):
        raise ValueError("device names and memory pools must be unique; do not double-count UMA")
    if len(peers) > 4 or expected_decode_tokens <= 0:
        raise ValueError("prototype supports at most four peers and a positive decode horizon")
    if cache_mode not in {"os", "fixed_resident"} or not math.isfinite(disk_gib_s) or disk_gib_s <= 0:
        raise ValueError("invalid cache mode or disk throughput")

    def score(a: Assignment):
        return a.decode_s_high * expected_decode_tokens + (a.cold_upload_s if include_cold_upload else 0)

    n = len(model.layer_bytes)
    best: Candidate | None = None
    # Cache all per-device spans across permutations: search is small for 80 layers/three Macs.
    spans: dict[tuple[str, int, int], Assignment | None] = {}
    for order in itertools.permutations(peers):
        states: dict[int, tuple[float, tuple[Assignment, ...]]] = {}
        for end in range(n + 1):
            a = segment(model, cpu, 0, end, cpu=True, allow_disk=allow_disk,
                        disk_gib_s=disk_gib_s, cache_mode=cache_mode)
            if a is not None:
                states[end] = (score(a), (a,))
        for peer in order:
            next_states = dict(states)  # skip peer entirely, without workspace/network cost
            for start, (cost, path) in states.items():
                for end in range(start + 1, n + 1):
                    key = (peer.name, start, end)
                    if key not in spans:
                        spans[key] = segment(model, peer, start, end)
                    a = spans[key]
                    if a is None:
                        break  # positive layer/KV sizes make larger spans infeasible too
                    proposed = cost + score(a)
                    if end not in next_states or proposed < next_states[end][0]:
                        next_states[end] = (proposed, (*path, a))
            states = next_states
        for start, (cost, path) in states.items():
            a = segment(model, local_gpu, start, n, output=True)
            if a is not None and (best is None or cost + score(a) < best.score_s):
                best = Candidate((*path, a), cost + score(a), expected_decode_tokens, include_cold_upload)
    return best


def switch_decision(*, old_decode_s: float, new_decode_s: float, remaining_tokens: int,
                    reload_and_replay_s: float, idle: bool, stable_samples: int,
                    minimum_stable_samples: int = 3, minimum_relative_gain: float = 0.15) -> dict:
    """Pure controller decision sketch. Caller separately validates freshness, leases and executability.

    Memory-pressure emergency handling is separate; a speed optimization never kills an active stream.
    Reload includes missing weights, engine startup, warmup and KV reconstruction. It is NOT migration.
    """
    if (any(not math.isfinite(x) for x in (old_decode_s, new_decode_s, reload_and_replay_s,
                                          minimum_relative_gain))
            or min(old_decode_s, new_decode_s) <= 0 or reload_and_replay_s < 0
            or remaining_tokens < 0 or stable_samples < 0 or minimum_stable_samples < 1
            or not 0 <= minimum_relative_gain < 1):
        raise ValueError("invalid replan inputs")
    saving = (old_decode_s - new_decode_s) * remaining_tokens
    gain = (old_decode_s - new_decode_s) / old_decode_s
    if stable_samples < minimum_stable_samples:
        action, reason = "observe", "capacity/link measurements are not stable"
    elif gain < minimum_relative_gain or saving <= reload_and_replay_s:
        action, reason = "keep", "expected savings do not justify reload/replay with hysteresis"
    elif not idle:
        action, reason = "defer", "wait for the active request to finish"
    else:
        action, reason = "prepare_replacement", "fresh-plan admission and transactional reload are still required"
    return {"action": action, "reason": reason, "expected_saved_s": saving,
            "reload_and_replay_s": reload_and_replay_s, "relative_decode_gain": gain}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    data = json.loads(args.scenario.read_text(encoding="utf-8"))
    model = Model(**{**data["model"], "layer_bytes": tuple(data["model"]["layer_bytes"])})
    reports = []
    for scenario in data["scenarios"]:
        candidate = optimize(model, Device(**scenario["cpu"]), Device(**scenario["local_gpu"]),
                             [Device(**p) for p in scenario["peers"]], **scenario.get("options", {}))
        reports.append({"name": scenario["name"], "result": candidate.report() if candidate else None})
    print(json.dumps({"input_evidence": data["evidence"], "scenarios": reports}, indent=2))


if __name__ == "__main__":
    main()
