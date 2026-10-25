"""EXO's wire formats, parsed without any network: tagged JSON state, readiness, placement and the SSE stream.

EXO unions serialize as {"ClassName": {...}}; field names follow EXO's pydantic models (docs/research/exo.md, bc412e2).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from ..schemas import ObservedDevice, ObservedPlacement, utcnow
from ..units import fmt_bytes
from .base import EngineAdapterError, GenerationEvent, PlacementReport

log = logging.getLogger("hypercompute.exo")
ACK_TIMEOUT_S = 60.0  # POST accepted but the instance never appeared (no upstream number exists; measure)
MAX_EVENT_BYTES = 4 * 1024 * 1024
READY = frozenset({"RunnerReady", "RunnerRunning"})
LOST = "EXO deleted the instance: a Mac in it went silent for 15 s or its runner failed 5 times"

# ----------------------------------------------------------------------------- EXO JSON shapes


def untag(value: object) -> tuple[str | None, dict]:
    """EXO unions serialize as {"ClassName": {...}}: -> ("ClassName", {...}); anything else -> (None, {})."""
    if isinstance(value, dict) and len(value) == 1:
        tag, body = next(iter(value.items()))
        if isinstance(body, dict):
            return str(tag), body
    return None, {}


def _bytes(memory: object) -> int:
    return int(memory.get("inBytes", 0)) if isinstance(memory, dict) else 0


@dataclass(frozen=True)
class Shard:
    runner_id: str
    node_id: str  # EXO's node id (new on every EXO start), not Hypercompute's
    kind: str  # PipelineShardMetadata | TensorShardMetadata | ...
    start: int  # layers [start, end) of n_layers
    end: int
    n_layers: int
    storage_bytes: int

    @property
    def weight_bytes(self) -> int:
        """Pipeline share of the weights, the way EXO's placement computes it."""
        return self.storage_bytes * (self.end - self.start) // self.n_layers if self.n_layers else 0


@dataclass(frozen=True)
class InstanceInfo:
    kind: str  # MlxRingInstance | MlxJacclInstance
    instance_id: str
    model_id: str
    shards: tuple[Shard, ...]

    @classmethod
    def parse(cls, instance: object) -> InstanceInfo:
        kind, body = untag(instance)
        if kind is None:
            raise ValueError("not a tagged EXO instance")
        sa = body.get("shardAssignments") or {}
        to_shard = sa.get("runnerToShard") or {}
        shards = []
        for node, runner in (sa.get("nodeToRunner") or {}).items():
            skind, sb = untag(to_shard.get(runner))
            shards.append(Shard(str(runner), str(node), skind or "unknown", int(sb.get("startLayer", 0)),
                                int(sb.get("endLayer", 0)), int(sb.get("nLayers", 0)),
                                _bytes((sb.get("modelCard") or {}).get("storageSize"))))
        if not body.get("instanceId") or not sa.get("modelId") or not shards:
            raise ValueError("instance lacks instanceId, modelId or runners")
        return cls(kind, str(body["instanceId"]), str(sa["modelId"]), tuple(shards))

    @property
    def node_ids(self) -> set[str]:
        return {s.node_id for s in self.shards}


def choose_preview(doc: object, wanted: set[str]) -> tuple[dict | None, list[str]]:
    """The first ring pipeline preview on exactly `wanted`, else EXO's reasons.

    Tags decide, not labels: EXO forces a one-Mac result to Pipeline + MlxRing but keeps the requested label."""
    previews = doc.get("previews") if isinstance(doc, dict) else None
    if not isinstance(previews, list) or not previews:
        return None, ["EXO sees no nodes yet (empty topology); wait a few seconds after the Macs start EXO"]
    errors: list[str] = []
    other_sets = 0
    for p in previews:
        if not isinstance(p, dict):
            continue
        if p.get("error"):
            errors.append(str(p["error"]))
            continue
        try:
            info = InstanceInfo.parse(p.get("instance"))
        except ValueError as exc:
            errors.append(f"unreadable preview: {exc}")
            continue
        if info.kind != "MlxRingInstance" or any(s.kind != "PipelineShardMetadata" for s in info.shards):
            continue
        if info.node_ids == wanted:
            return p, []
        other_sets += 1
    if other_sets:
        errors.append(f"EXO offered {other_sets} placement(s) on other sets of Macs only")
    return None, sorted(set(errors)) or ["EXO offered no ring pipeline placement"]


def instances_for(doc: object, model_id: str) -> list[str]:
    """Ids of EXO instances in /state/instances that already serve `model_id`."""
    out = []
    for iid, inst in (doc.items() if isinstance(doc, dict) else []):
        _, body = untag(inst)
        if (body.get("shardAssignments") or {}).get("modelId") == model_id:
            out.append(str(iid))
    return out


# ----------------------------------------------------------------------------- readiness


@dataclass(frozen=True)
class Readiness:
    state: Literal["waiting", "ready", "failed"]
    text: str


def _download_phase(items: object, model_id: str) -> Readiness | None:
    for item in items if isinstance(items, list) else []:
        tag, body = untag(item)
        _, shard = untag(body.get("shardMetadata"))
        if (shard.get("modelCard") or {}).get("modelId") != model_id:
            continue
        if tag == "DownloadFailed":
            return Readiness("failed", f"download failed: {body.get('errorMessage') or 'no message'}")
        if tag == "DownloadOngoing":
            prog = body.get("downloadProgress") or {}
            done, total = _bytes(prog.get("downloaded")), _bytes(prog.get("total"))
            pct = f" ({done * 100 // total}%)" if total else ""
            return Readiness("waiting", f"downloading {fmt_bytes(done)} of {fmt_bytes(total)}{pct}")
        if tag == "DownloadPending":
            return Readiness("waiting", "download queued")
        if tag == "DownloadCompleted":
            return Readiness("waiting", "downloaded, starting runner")
    return None


def runner_phase(status: object, downloads: object, model_id: str) -> Readiness:
    tag, body = untag(status)
    if tag in READY:
        return Readiness("ready", "ready")
    if tag == "RunnerFailed":
        return Readiness("failed", f"runner failed: {body.get('errorMessage') or 'no message'}")
    if tag in (None, "RunnerIdle"):
        return _download_phase(downloads, model_id) or Readiness("waiting", "starting runner")
    if tag in ("RunnerConnecting", "RunnerConnected"):
        return Readiness("waiting", "connecting to the other Macs")
    if tag == "RunnerLoading":
        return Readiness("waiting", f"loading layers {body.get('layersLoaded', 0)}/{body.get('totalLayers', 0)}")
    if tag in ("RunnerLoaded", "RunnerWarmingUp"):
        return Readiness("waiting", "warming up")
    if tag in ("RunnerShuttingDown", "RunnerShutdown"):
        return Readiness("waiting", "runner restarting")
    return Readiness("waiting", tag)


def readiness(info: InstanceInfo, present: bool, seen: bool, waited_s: float, runners: dict,
              downloads: dict | None, names: dict[str, str]) -> Readiness:
    """One poll of EXO state -> waiting/ready/failed, with a per-Mac progress line."""
    if not present:
        if seen:
            return Readiness("failed", LOST)
        if waited_s > ACK_TIMEOUT_S:
            return Readiness("failed", f"EXO did not create the instance within {ACK_TIMEOUT_S:.0f}s (the command "
                                       "was lost, or the Macs are still electing a master)")
        return Readiness("waiting", "waiting for EXO to place the instance")
    pending = []
    for s in info.shards:
        name = names.get(s.node_id, s.node_id[:8])
        phase = runner_phase(runners.get(s.runner_id), (downloads or {}).get(s.node_id), info.model_id)
        if phase.state == "failed":
            return Readiness("failed", f"{name}: {phase.text}")
        if phase.state != "ready":
            pending.append(f"{name}: {phase.text}")
    return Readiness("waiting", "; ".join(pending)) if pending else Readiness("ready", "every EXO runner is ready")


def observed_placement(info: InstanceInfo, runners: dict, names: dict[str, str], hc_of: dict[str, str],
                       planned: set[str]) -> PlacementReport:
    """Where EXO put the layers, and whether every planned Mac holds a ready runner."""
    devices, evidence, missing = [], [], []
    for s in info.shards:
        tag, _ = untag(runners.get(s.runner_id))
        name = names.get(s.node_id, s.node_id[:8])
        devices.append(ObservedDevice(name=f"EXO {name}", model_bytes=s.weight_bytes, kv_bytes=0, compute_bytes=0,
                                      layers_est=s.end - s.start))
        evidence.append(f"{name} (EXO node {s.node_id}): runner {s.runner_id} {tag or 'not reported'}, "
                        f"layers {s.start}-{s.end - 1} of {s.n_layers}")
        if tag not in READY:
            missing.append(f"{name}: runner is {tag or 'not reported'}")
    covered = {hc_of.get(s.node_id) for s in info.shards}
    missing += [f"planned node {n[:8]} has no EXO runner" for n in sorted(planned - covered)]
    layers = sum(s.end - s.start for s in info.shards)
    observed = ObservedPlacement(offloaded=f"{layers}/{info.shards[0].n_layers}", devices=devices,
                                 evidence_lines=evidence, parsed_at=utcnow())
    if missing:
        return PlacementReport(observed=observed, verified=False, detail="; ".join(missing))
    spans = ", ".join(f"{names.get(s.node_id, s.node_id[:8])} (layers {s.start}-{s.end - 1})" for s in info.shards)
    return PlacementReport(observed=observed, verified=True, detail=f"EXO runners ready on {spans}")


# ----------------------------------------------------------------------------- streaming


def stats_timings(stats: dict) -> dict:
    """EXO generation_stats -> llama.cpp-style timings, so run records compare across engines."""
    out: dict = {}
    for n_key, tps_key, prefix in (("prompt_tokens", "prompt_tps", "prompt"),
                                   ("generation_tokens", "generation_tps", "predicted")):
        n, tps = stats.get(n_key), stats.get(tps_key)
        if isinstance(n, int):
            out[f"{prefix}_n"] = n
        if isinstance(n, int) and isinstance(tps, int | float) and tps > 0:
            out[f"{prefix}_ms"] = n / tps * 1000
            out[f"{prefix}_per_second"] = float(tps)
    if stats.get("prefix_cache_hit") == "none":
        out["cache_n"] = 0
    return out


def _event(chunk: dict) -> GenerationEvent:
    data = json.dumps(chunk, separators=(",", ":"), ensure_ascii=False)
    return GenerationEvent(raw=f"data: {data}\n\n".encode(), data=data, payload=chunk)


class ExoStreamNormalizer:
    """EXO SSE -> OpenAI `chat.completion.chunk` events that keep `usage` and gain `timings`.

    EXO labels chunks `chat.completion`, names its own model id, sends comments (keep-alive, prefill_progress,
    generation_stats) and reports a mid-stream failure as `data: {"error": ...}` before [DONE]. The finish chunk is
    held until [DONE] because generation_stats arrives between them.
    """

    def __init__(self, alias: str) -> None:
        self._alias = alias
        self._buf = bytearray()
        self._cr = b""  # a chunk that ended inside a CRLF: its "\r" waits for the next chunk
        self._held: dict | None = None
        self._timings: dict | None = None
        self.command_id: str | None = None  # EXO's command id: POST /v1/cancel/<id> stops the generation

    def feed(self, chunk: bytes) -> list[GenerationEvent]:
        """Linear in the stream: only new bytes are normalised and searched, however small the segments."""
        data = self._cr + chunk
        self._cr = b"\r" if data.endswith(b"\r") else b""
        scan = max(0, len(self._buf) - 1)  # a blank line can straddle the old and the new bytes
        self._buf += data[:len(data) - len(self._cr)].replace(b"\r\n", b"\n")
        events: list[GenerationEvent] = []
        while (end := self._buf.find(b"\n\n", scan)) >= 0:
            block = bytes(self._buf[:end]).decode("utf-8", errors="replace")
            del self._buf[:end + 2]
            scan = 0
            events += self._block(block)
        if len(self._buf) > MAX_EVENT_BYTES:
            raise EngineAdapterError("exo_bad_stream", "EXO sent an SSE event larger than 4 MiB")
        return events

    def flush(self) -> list[GenerationEvent]:
        """The connection closed: parse what is left and still deliver a held finish chunk."""
        block = bytes(self._buf + self._cr).decode("utf-8", errors="replace")
        self._buf.clear()
        self._cr = b""
        return [*(self._block(block) if block.strip() else []), *self._release()]

    def _block(self, block: str) -> list[GenerationEvent]:
        out: list[GenerationEvent] = []
        data = []
        for line in block.split("\n"):
            if line.startswith(":"):
                out += self._comment(line[1:].strip())
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        return out + (self._data("\n".join(data)) if data else [])

    def _comment(self, text: str) -> list[GenerationEvent]:
        if text == "keep-alive":  # forwarded: keeps the client's connection open during a long prefill
            return [GenerationEvent(raw=b": keep-alive\n\n", data="", payload=None)]
        if text.startswith("generation_stats "):
            try:
                stats = json.loads(text.removeprefix("generation_stats "))
            except json.JSONDecodeError:
                log.warning("unreadable EXO generation_stats: %s", text[:200])
                return []
            if isinstance(stats, dict):
                self._timings = stats_timings(stats)
        return []  # prefill_progress and other comments are EXO-specific

    def _data(self, data: str) -> list[GenerationEvent]:
        if data.strip() == "[DONE]":
            return [*self._release(), GenerationEvent(raw=b"data: [DONE]\n\n", data="[DONE]", payload=None, done=True)]
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise EngineAdapterError("exo_bad_stream", f"EXO sent invalid JSON: {data[:200]}") from exc
        if not isinstance(payload, dict):
            raise EngineAdapterError("exo_bad_stream", f"EXO sent a non-object event: {data[:200]}")
        if "error" in payload:
            err = payload["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            raise EngineAdapterError("exo_generation_failed", f"EXO: {message or 'generation failed'}")
        self.command_id = self.command_id or payload.get("id")
        chunk = {**payload, "object": "chat.completion.chunk", "model": self._alias}
        if any(isinstance(c, dict) and c.get("finish_reason") for c in chunk.get("choices") or []):
            self._held = chunk
            return []
        return [_event(chunk)]

    def _release(self) -> list[GenerationEvent]:
        if self._held is None:
            return []
        chunk = {**self._held, "timings": self._timings} if self._timings else self._held
        self._held = None
        return [_event(chunk)]
