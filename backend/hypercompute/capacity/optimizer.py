"""Exact placement search over llama.cpp-expressible layouts: CPU prefix, peer spans, local GPU suffix.

A layout is a chain of stages over the model's units (blocks 0..n_layer-1, then the head at n_layer): an optional
host CPU prefix whose units are resident or streamed from the GGUF, then any subset of peers in any order with one
contiguous non-empty span each, then the local GPU suffix holding the head. Without a GPU the head goes with the
last peer stage, or with the CPU when no peer is used. Every such chain is a device list plus a tensor split.

For every peer subset and order a dynamic program over unit boundaries finds the layout with the smallest decode
time and the layout with the smallest workload objective J, with every stage's demand inside its pool's admissible
bytes. Devices that allocate from one pool (a Mac's CPU and Metal device, or a peer marked `shares_pool_with`) share
that pool: the program then keeps a Pareto set of (score, bytes used on each shared pool) per state, and a CPU on such
a pool tries every split the CPU rule can return, then keeps the one the rule returns for what the later stages leave.
On equal scores the layout streaming fewer bytes wins. Candidates are ranked by J. Every number that is not measured
is listed with its provenance.
"""

from __future__ import annotations

import bisect
import itertools
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

from ..units import fmt_bytes
from . import costs
from .types import (
    Coefficient,
    DeviceProfile,
    ModelManifest,
    PlacementCandidate,
    PoolDemand,
    StagePlan,
    StorageProfile,
    Workload,
)

MAX_PEERS = 4
RISK_PATHS = ("relay", "unknown")  # an unknown path is priced as a relay (costs.link_prior)
Role = Literal["cpu", "gpu", "peer"]


@dataclass(frozen=True)
class HostSide:
    """The coordinator: its CPU, optional local GPU and storage, with the admissible bytes of their pools.

    Admissible bytes come from pools.admissible_bytes upstream. When the CPU and GPU report the same pool_id (a Mac's
    unified memory) they share one budget, the smaller of the two figures.
    """

    node_id: str
    cpu: DeviceProfile
    cpu_pool_admissible: int
    storage: StorageProfile | None
    gpu: DeviceProfile | None
    gpu_pool_admissible: int
    stream_supported: bool


@dataclass(frozen=True)
class PeerSide:
    """One remote device reached over llama.cpp RPC; `device.link` describes the path to it.

    shares_pool_with names a node whose pool this device also allocates from: another peer's node_id, or the host's
    (its CPU pool, or its GPU pool when the pool_id matches). Devices with the same node and pool_id share a pool
    without it.
    """

    node_id: str
    device: DeviceProfile
    pool_admissible: int
    shares_pool_with: str | None = None


@dataclass(frozen=True)
class OptimizeResult:
    best: PlacementCandidate | None
    candidates: tuple[PlacementCandidate, ...]  # ranked by J, feasible first
    rejected: tuple[str, ...]
    assumptions: tuple[str, ...]
    workload: Workload
    head_unit: int
    budgets: tuple[tuple[str, int], ...]  # (node:device, admissible bytes of its pool)
    provenance: tuple[tuple[str, str, str, str], ...]  # (node:device, compute, storage I/O, hop)

    def explain(self) -> list[str]:
        """Plain lines: totals, each stage with bytes and ms terms, the bottleneck, skipped devices, priors."""
        tail = [f"rejected: {r}" for r in self.rejected] + [f"assumption: {a}" for a in self.assumptions]
        if self.best is None:
            lines = ["no feasible layout"]
            for c in self.candidates:
                lines.append("closest layout with memory ignored: " + _layout_text(c, self.head_unit))
                lines += [f"  overflow: {r}" for r in c.reasons]
            return lines + tail
        b, w = self.best, self.workload
        load = "load not charged (warm)" if w.warm else f"load {b.load_ms:.0f}"
        risk = max(0.0, b.objective_ms - (0.0 if w.warm else b.load_ms) - b.prefill_ms - w.output_tokens * b.decode_ms)
        lines = [f"best: J {b.objective_ms:.0f} ms = {load} + prefill {b.prefill_ms:.0f} + {w.output_tokens} x "
                 f"decode {b.decode_ms:.1f} ms + risk {risk:.0f}; {fmt_bytes(b.stream_bytes_per_token)} streamed "
                 "per token"]
        lines += [self._stage_line(i, s, b) for i, s in enumerate(b.stages, 1)]
        lines.append(f"bottleneck: {b.bottleneck}")
        lines += [f"skipped {s}" for s in b.skipped]
        lines += [f"alternative {i}: J {c.objective_ms:.0f} ms, decode {c.decode_ms:.1f} ms, "
                  + _layout_text(c, self.head_unit) for i, c in enumerate(self.candidates[1:], 2)]
        return lines + tail

    def _stage_line(self, i: int, s: StagePlan, cand: PlacementCandidate) -> str:
        key = f"{s.node_id}:{s.device}"
        budget = dict(self.budgets).get(key, 0)
        prov = {k: (c, io, h) for k, c, io, h in self.provenance}.get(key, ("?", "?", "?"))
        d = s.demand or PoolDemand(node_id=s.node_id, pool_id=s.pool_id)
        text = (f"stage {i} {key} ({s.path}) units {_span_text(s.units, self.head_unit)}: resident "
                f"{fmt_bytes(d.weights)}")
        if s.streamed_units:
            text += (f", streamed {fmt_bytes(cand.stream_bytes_per_token)} ({len(s.streamed_units)} units) in "
                     f"{s.slots} slots")
        text += (f"; demand {fmt_bytes(d.peak)} of {fmt_bytes(budget)} admissible; compute {s.compute_ms:.1f} ms "
                 f"({prov[0]})")
        if s.streamed_units:
            text += f", exposed I/O {s.exposed_io_ms:.1f} ms ({prov[1]})"
        if s.path != "local":
            text += f", hop {s.hop_ms:.1f} ms ({prov[2]})"
        return text


# --------------------------------------------------------------------------------------------------- internals


@dataclass(frozen=True)
class _Part:
    """A device that can hold one stage."""

    key: str  # node:device
    node_id: str
    role: Role
    profile: DeviceProfile
    group: str  # pool group after merging shared pools
    admissible: int  # what the pool group admits (the smallest member figure)


@dataclass(frozen=True)
class _Stage:
    part: _Part
    plan: StagePlan
    decode_ms: float
    prefill_ms: float
    load_ms: float
    risk_ms: float  # over the whole workload
    stream_bytes: int
    reasons: tuple[str, ...]  # memory overflow; empty when the stage fits its pool

    def j_ms(self, w: Workload) -> float:
        return (0.0 if w.warm else self.load_ms) + self.prefill_ms + w.output_tokens * self.decode_ms + self.risk_ms


@dataclass(frozen=True)
class _Ctx:
    manifest: ModelManifest
    workload: Workload
    storage: StorageProfile
    stream_ok: bool
    slots: int
    kv_per_layer: int
    workspace: int
    act_bytes: int
    head_down_bytes: int  # what a peer holding the head returns per token: f32 logits, or act_bytes if vocab unknown


@dataclass(frozen=True)
class _Run:
    """One search pass: the stage score it minimizes and whether memory is enforced. Caches are per optimize call."""

    ctx: _Ctx
    score: Callable[[_Stage], float]
    enforce: bool
    memo: dict[tuple[object, ...], object] = field(default_factory=dict)
    rejected: dict[str, tuple[int, str]] = field(default_factory=dict)


class _Label(NamedTuple):  # immutable like the dataclasses, and cheap to build in the inner loop
    score: float
    usage: tuple[int, ...]  # bytes used so far on each shared pool group of the sequence
    stream: int  # storage bytes streamed per token so far: on equal scores the label streaming less wins
    stages: tuple[_Stage, ...]


# --------------------------------------------------------------------------------------------------- stages


def _span_text(units: Sequence[int], head: int) -> str:
    blocks = [u for u in units if u != head]
    text = f"{blocks[0]}-{blocks[-1]}" if blocks else ""
    return (text + " + head" if text else "head") if head in units else text


def _layout_text(c: PlacementCandidate, head: int) -> str:
    return ", ".join(f"{s.node_id}:{s.device} [{_span_text(s.units, head)}]" for s in c.stages)


def _overflow(part: _Part, units: Sequence[int], d: PoolDemand, head: int) -> str:
    return (f"{part.key} units {_span_text(units, head)}: needs {fmt_bytes(d.peak)} (weights {fmt_bytes(d.weights)}, "
            f"KV {fmt_bytes(d.kv)}, workspace {fmt_bytes(d.workspace)}, staging {fmt_bytes(d.staging)}, overhead "
            f"{fmt_bytes(d.overhead)}), pool admits {fmt_bytes(part.admissible)}")


def _build(ctx: _Ctx, part: _Part, start: int, end: int, split: costs.CpuSplit | None) -> _Stage:
    """Demand and time terms of one stage; the decode, prefill, load and risk formulas of the contract."""
    w, p = ctx.workload, part.profile
    units = ctx.manifest.units[start:end]
    idx = tuple(u.index for u in units)
    weight = sum(u.weight_bytes for u in units)
    stream = split.stream_bytes if split else 0
    resident = split.resident_bytes if split else weight
    staging = costs.RPC_STAGING_BYTES if part.role == "peer" else (split.slots * split.slot_bytes if split else 0)
    demand = PoolDemand(node_id=p.node_id, pool_id=p.pool_id, weights=resident,
                        kv=ctx.kv_per_layer * sum(1 for u in units if u.kind == "block"), workspace=ctx.workspace,
                        staging=staging, overhead=costs.overhead_bytes(p.backend))
    compute = costs.compute_ms(p.decode_ms_per_gib, weight)
    stage_ms = costs.cpu_stage_ms(compute, costs.stream_io_ms(stream, ctx.storage), split.slots) if stream else compute
    hop = prefill_hop = risk = 0.0
    upload = 0
    path = "local"
    if part.role == "peer" and p.link is not None:
        link, n = p.link, w.prompt_tokens
        path = link.path
        head = ctx.manifest.n_layer in idx  # the host then reads the logits back instead of a hidden state
        down = ctx.head_down_bytes if head else ctx.act_bytes
        hop = costs.hop_ms(link, ctx.act_bytes, down)
        prefill_hop = costs.hop_ms(link, n * ctx.act_bytes, down if head else n * ctx.act_bytes) if n else 0.0
        upload = max(0, weight - link.cached_bytes)
        load = costs.transfer_ms(upload, link) + costs.setup_per_remote_prior().value
        if path in RISK_PATHS:
            risk = costs.relay_risk_prior().value * (stage_ms + hop) * w.output_tokens
    else:
        load = costs.read_ms(resident, ctx.storage)
    plan = StagePlan(node_id=part.node_id, device=p.device, pool_id=p.pool_id, units=idx,
                     resident_units=split.resident if split else idx, streamed_units=split.streamed if split else (),
                     slots=split.slots if split and split.streamed else 0, demand=demand, compute_ms=compute,
                     exposed_io_ms=stage_ms - compute, hop_ms=hop, upload_bytes=upload, path=path)
    over = demand.peak > part.admissible
    return _Stage(part=part, plan=plan, decode_ms=stage_ms + hop,
                  prefill_ms=w.prompt_tokens * costs.compute_ms(p.prefill_ms_per_gib_token, weight) + prefill_hop,
                  load_ms=load, risk_ms=risk, stream_bytes=stream,
                  reasons=(_overflow(part, idx, demand, ctx.manifest.n_layer),) if over else ())


def _stage(run: _Run, part: _Part, start: int, end: int) -> _Stage | str:
    key = (part.key, start, end, run.enforce)
    if key not in run.memo:
        st = _build(run.ctx, part, start, end, None)
        run.memo[key] = st.reasons[0] if run.enforce and st.reasons else st
    return run.memo[key]  # type: ignore[return-value]


def _cpu_budget(ctx: _Ctx, part: _Part, end: int) -> int:
    """Bytes a CPU prefix of `end` units has for weights and slots: admissible minus KV, workspace and overhead."""
    blocks = sum(1 for u in ctx.manifest.units[:end] if u.kind == "block")
    return part.admissible - ctx.kv_per_layer * blocks - ctx.workspace - costs.overhead_bytes(part.profile.backend)


def _cpu_stages(run: _Run, part: _Part, end: int, shared: bool) -> tuple[_Stage, ...] | str:
    key = (part.key, 0, end, run.enforce, shared)
    if key not in run.memo:
        ctx = run.ctx
        units = ctx.manifest.units[:end]
        budget = _cpu_budget(ctx, part, end)
        if shared:
            splits = costs.cpu_split_options(units, budget, slots=ctx.slots, stream=ctx.stream_ok)
        else:
            one = costs.split_cpu(units, budget, slots=ctx.slots, stream=ctx.stream_ok)
            splits = (one,) if one else ()
        stages = tuple(_build(ctx, part, 0, end, s) for s in splits)
        if not stages:  # report the least memory the prefix could use, streaming when allowed
            fallback = _build(ctx, part, 0, end, costs.smallest_split(units, slots=ctx.slots, stream=ctx.stream_ok))
            note = "" if ctx.stream_ok else " (streaming not available)"
            stages = fallback.reasons[0] + note if run.enforce else (fallback,)
        run.memo[key] = stages
    return run.memo[key]  # type: ignore[return-value]


def _refill(ctx: _Ctx, stages: tuple[_Stage, ...]) -> tuple[_Stage, ...]:
    """When the CPU shares its pool with later stages, its split is `split_cpu` for the bytes they leave.

    The search tries every split the rule can return; this sets the one the rule returns for the actual leftover, so
    nothing streams that the leftover could keep resident and `evaluate_layout` scores a shared pool the same way.
    """
    cpu = stages[0] if stages else None
    if cpu is None or cpu.part.role != "cpu":
        return stages
    others = sum(s.plan.demand.peak for s in stages[1:] if s.part.group == cpu.part.group and s.plan.demand)
    if not others:
        return stages
    end = len(cpu.plan.units)
    room = _cpu_budget(ctx, cpu.part, end) - others
    split = costs.split_cpu(ctx.manifest.units[:end], room, slots=ctx.slots, stream=ctx.stream_ok)
    return stages if split is None else (_build(ctx, cpu.part, 0, end, split), *stages[1:])


# --------------------------------------------------------------------------------------------------- search


def _note(run: _Run, part: _Part, n_units: int, reason: str) -> None:
    if part.key not in run.rejected or n_units < run.rejected[part.key][0]:
        run.rejected[part.key] = (n_units, reason)


def _extend(run: _Run, label: _Label, st: _Stage, tracked: Sequence[str], limits: dict[str, int]) -> _Label | None:
    usage = label.usage
    if st.part.group in tracked:
        i = tracked.index(st.part.group)
        used = usage[i] + (st.plan.demand.peak if st.plan.demand else 0)
        if used > limits[st.part.group]:
            run.rejected.setdefault(f"pool {st.part.group}", (0, f"pool shared by {st.part.group} admits "
                                                              f"{fmt_bytes(limits[st.part.group])} for all its stages"))
            return None
        usage = usage[:i] + (used,) + usage[i + 1:]
    return _Label(label.score + run.score(st), usage, label.stream + st.stream_bytes, (*label.stages, st))


def _dominates(a: _Label, b: _Label) -> bool:
    """`a` is at least as good in every future: no higher score, no more shared-pool usage, and on an equal score no
    more streaming (otherwise a split whose I/O hides under compute would win on memory although nothing needs it)."""
    return (a.score <= b.score and all(map(operator.le, a.usage, b.usage))
            and (a.score < b.score or a.stream <= b.stream))


def _usage0(label: _Label) -> int:
    return label.usage[0]


def _insert_1d(labels: list[_Label], new: _Label) -> list[_Label]:
    """`_insert` for one shared pool: labels sorted by usage with (score, stream) strictly falling, so bisection finds
    the one label that could dominate `new` and the run of labels `new` dominates."""
    key, used = (new.score, new.stream), new.usage[0]
    hi = bisect.bisect_right(labels, used, key=_usage0)
    if hi and (labels[hi - 1].score, labels[hi - 1].stream) <= key:
        return labels
    end = hi
    while end < len(labels) and (labels[end].score, labels[end].stream) >= key:
        end += 1
    return [*labels[:bisect.bisect_left(labels, used, key=_usage0)], new, *labels[end:]]


def _insert(labels: list[_Label], new: _Label) -> list[_Label]:
    """Labels with `new` added and every label it dominates (see `_dominates`) dropped."""
    if len(new.usage) == 1:
        return _insert_1d(labels, new)
    for old in labels:
        if _dominates(old, new):
            return labels
    return [old for old in labels if not _dominates(new, old)] + [new]


def _search(run: _Run, seq: tuple[_Part, ...]) -> tuple[_Stage, ...] | None:
    """Best chain for `seq` (CPU, peers in order, optional GPU) under the run's score; None when nothing fits."""
    n = len(run.ctx.manifest.units)
    cpu, required = seq[0], seq[1:]
    if len(required) > n:
        return None
    groups = [p.group for p in seq]
    tracked = sorted({g for g in groups if groups.count(g) > 1}) if run.enforce else []
    limits = {p.group: p.admissible for p in seq}
    root = _Label(0.0, (0,) * len(tracked), 0, ())
    frontier: dict[int, list[_Label]] = {0: [root]} if required else {}
    for end in range(1, n - len(required) + 1) if required else [n]:
        got = _cpu_stages(run, cpu, end, cpu.group in tracked)
        if isinstance(got, str):
            _note(run, cpu, end, got)
            continue
        for st in got:
            if (label := _extend(run, root, st, tracked, limits)) is not None:
                frontier[end] = _insert(frontier.get(end, []), label)
    for j, part in enumerate(required):
        last = j == len(required) - 1
        nxt: dict[int, list[_Label]] = {}
        for start, labels in frontier.items():
            for end in [n] if last else range(start + 1, n - (len(required) - j - 1) + 1):
                st = _stage(run, part, start, end)
                if isinstance(st, str):
                    _note(run, part, end - start, st)
                    break  # a longer span needs more memory, so it cannot fit either
                for label in labels:
                    if (new := _extend(run, label, st, tracked, limits)) is not None:
                        nxt[end] = _insert(nxt.get(end, []), new)
        frontier = nxt
    final = frontier.get(n, [])
    return min(final, key=lambda lb: (lb.score, lb.stream)).stages if final else None


# --------------------------------------------------------------------------------------------------- candidates


def _bottleneck(stages: Sequence[_Stage], decode: float) -> str:
    terms = [(t, f"{s.part.key} {name}") for s in stages
             for t, name in ((s.plan.compute_ms, "compute"), (s.plan.exposed_io_ms, "exposed storage I/O"),
                             (s.plan.hop_ms, "hop"))]
    value, label = max(terms, key=lambda t: t[0], default=(0.0, "none"))
    share = 100.0 * value / decode if decode > 0 else 0.0
    return f"{label} {value:.1f} ms of {decode:.1f} ms per token ({share:.0f} percent)"


def _assumptions(ctx: _Ctx, stages: Sequence[_Stage]) -> tuple[str, ...]:
    w, out = ctx.workload, []

    def add(label: str, c: Coefficient) -> None:
        if c.provenance != "measured":
            out.append(f"{label} {c.value:.4g} {c.unit} ({c.provenance}: {c.note})")

    for s in stages:
        p = s.part.profile
        add(f"{s.part.key} decode", p.decode_ms_per_gib)
        if w.prompt_tokens:
            add(f"{s.part.key} prefill", p.prefill_ms_per_gib_token)
        if s.part.role == "peer" and p.link is not None:
            for name, c in (("rtt", p.link.rtt_ms), ("goodput", p.link.goodput_bytes_per_s),
                            ("round trips per token", p.link.round_trips_per_token)):
                add(f"{s.part.key} {name}", c)
            if ctx.manifest.n_layer in s.plan.units:
                out.append(f"{s.part.key} holds the head: {fmt_bytes(ctx.head_down_bytes)} of f32 logits return per "
                           "token (llama_cpu_tier.md A2)" if ctx.manifest.vocab_size else
                           f"{s.part.key} holds the head, vocab size unknown: the logits it returns per token are "
                           "priced as one hidden state, an underestimate")
        if s.stream_bytes or (not w.warm and s.part.role != "peer"):
            add("storage read", ctx.storage.read_bytes_per_s)
        if s.stream_bytes:
            add("stream efficiency", ctx.storage.stream_efficiency)
        if s.risk_ms:
            add(f"{s.part.key} relay risk", costs.relay_risk_prior())
        out.append(f"{p.backend} overhead {fmt_bytes(costs.overhead_bytes(p.backend))} per stage (prior: runtime and "
                   "allocator margin, the planner's figure)")
    out.append(f"workspace {fmt_bytes(ctx.workspace)} per stage (prior: the planner's compute buffer base plus "
               f"ubatch {costs.UBATCH} term)")
    if any(s.part.role == "peer" for s in stages):
        out.append(f"RPC staging {fmt_bytes(costs.RPC_STAGING_BYTES)} per peer stage (prior, the planner's figure)")
        if not w.warm:
            add("setup per peer stage", costs.setup_per_remote_prior())
    if not w.warm:
        add("setup", costs.setup_prior())
    return tuple(dict.fromkeys(out))


def _pool_use(stages: Sequence[_Stage]) -> dict[str, tuple[int, int, int]]:
    """Per pool group: (bytes the stages need together, what the group admits, number of stages on it)."""
    use: dict[str, tuple[int, int, int]] = {}
    for s in stages:
        used, _, count = use.get(s.part.group, (0, 0, 0))
        use[s.part.group] = (used + (s.plan.demand.peak if s.plan.demand else 0), s.part.admissible, count + 1)
    return use


def _excess(stages: Sequence[_Stage]) -> int:
    return sum(max(0, used - limit) for used, limit, _ in _pool_use(stages).values())


def _candidate(ctx: _Ctx, stages: tuple[_Stage, ...]) -> PlacementCandidate:
    w = ctx.workload
    decode = sum(s.decode_ms for s in stages)
    prefill = sum(s.prefill_ms for s in stages)
    load = sum(s.load_ms for s in stages) + costs.setup_prior().value
    shared = tuple(f"pool {g}: stages together need {fmt_bytes(u)}, pool admits {fmt_bytes(limit)}"
                   for g, (u, limit, count) in _pool_use(stages).items() if u > limit and count > 1)
    reasons = tuple(r for s in stages for r in s.reasons) + shared
    return PlacementCandidate(
        stages=tuple(s.plan for s in stages), feasible=not reasons, reasons=reasons, decode_ms=decode,
        prefill_ms=prefill, load_ms=load,
        objective_ms=(0.0 if w.warm else load) + prefill + w.output_tokens * decode + sum(s.risk_ms for s in stages),
        bottleneck=_bottleneck(stages, decode), stream_bytes_per_token=sum(s.stream_bytes for s in stages),
        assumptions=_assumptions(ctx, stages))


def _keys(c: PlacementCandidate) -> set[str]:
    return {f"{s.node_id}:{s.device}" for s in c.stages}


def _hops(c: PlacementCandidate) -> float:
    return sum(s.hop_ms for s in c.stages)


def _skipped(c: PlacementCandidate, pool: Sequence[PlacementCandidate], peers: Sequence[_Part],
             rejected: dict[str, tuple[int, str]], warm: bool) -> tuple[str, ...]:
    out = []
    for part in peers:
        if part.key in _keys(c):
            continue
        with_it = [o for o in pool if part.key in _keys(o)]
        if not with_it:
            why = rejected.get(part.key, (0, "no feasible layout includes it"))[1]
            out.append(f"{part.key}: not used: {why}")
            continue
        alt = min(with_it, key=lambda o: o.objective_ms)
        added = _hops(alt) - _hops(c)
        saved = (c.decode_ms - _hops(c)) - (alt.decode_ms - _hops(alt))
        verb = "saves" if saved >= 0 else "adds"
        text = (f"{part.key}: not used: adds {added:.1f} ms hop per token, {verb} {abs(saved):.1f} ms compute and "
                "I/O per token")
        if not warm and alt.load_ms > c.load_ms:
            text += f", adds {(alt.load_ms - c.load_ms) / 1000:.1f} s load"
        out.append(text + f"; the best layout with it scores J {alt.objective_ms - c.objective_ms:+.0f} ms")
    return tuple(out)


# --------------------------------------------------------------------------------------------------- setup


def _validate(manifest: ModelManifest, workload: Workload, host: HostSide, peers: Sequence[PeerSide], slots: int,
              max_candidates: int) -> None:
    units = manifest.units
    if (len(units) != manifest.n_layer + 1 or any(u.index != i for i, u in enumerate(units))
            or units[-1].kind != "head" or any(u.kind != "block" for u in units[:-1])):
        raise ValueError("manifest units must be blocks 0..n_layer-1 followed by the head")
    if (any(u.weight_bytes < 0 or u.buffer_bytes < 0 for u in units) or manifest.n_embd <= 0
            or (manifest.vocab_size is not None and manifest.vocab_size <= 0)):
        raise ValueError("manifest unit bytes must be non-negative, n_embd and a known vocab_size positive")
    if workload.ctx_size <= 0 or workload.prompt_tokens < 0 or workload.output_tokens < 0:
        raise ValueError("workload needs a positive ctx_size and non-negative token counts")
    if slots < 1 or max_candidates < 1:
        raise ValueError("slots and max_candidates must be at least 1")
    if len(peers) > MAX_PEERS:
        raise ValueError(f"at most {MAX_PEERS} peers can be searched exactly; got {len(peers)}")
    figures = [host.cpu_pool_admissible, host.gpu_pool_admissible, *(p.pool_admissible for p in peers)]
    if any(f < 0 for f in figures):
        raise ValueError("admissible bytes must be non-negative (pools.admissible_bytes floors at zero)")
    for p in peers:
        if p.device.link is None:
            raise ValueError(f"peer {p.node_id}:{p.device.device} has no link profile")
    owners = [(host.node_id, host.cpu), *(((host.node_id, host.gpu),) if host.gpu else ()),
              *((p.node_id, p.device) for p in peers)]
    if wrong := [f"{node} holds {d.node_id}:{d.device}" for node, d in owners if d.node_id != node]:
        raise ValueError(f"device profiles must belong to the node that offers them: {wrong}")
    if unknown := sorted({d.backend for _, d in owners} - set(costs.OVERHEAD_BYTES)):
        raise ValueError(f"no memory overhead prior for backend(s) {unknown}")
    for _, d in owners:
        costs.check_device(d)
    if host.storage is not None:
        costs.check_storage(host.storage)


def _share_targets(own: str, share: str, host: HostSide, peers: Sequence[PeerSide], pool_id: str) -> list[str]:
    targets = [f"{p.node_id}:{p.device.device}" for p in peers if p.node_id == share]
    if share == host.node_id:
        on_gpu = host.gpu is not None and host.gpu.pool_id == pool_id
        targets.append(f"{host.node_id}:{(host.gpu if on_gpu and host.gpu else host.cpu).device}")
    targets = [t for t in targets if t != own]
    if not targets:
        raise ValueError(f"{own} shares_pool_with {share!r}, which names no other device")
    return targets


def _parts(host: HostSide, peers: Sequence[PeerSide]) -> tuple[_Part, tuple[_Part, ...], _Part | None]:
    raw: list[tuple[Role, str, DeviceProfile, int, str | None]] = [
        ("cpu", host.node_id, host.cpu, host.cpu_pool_admissible, None)]
    if host.gpu is not None:
        raw.append(("gpu", host.node_id, host.gpu, host.gpu_pool_admissible, None))
    raw += [("peer", p.node_id, p.device, p.pool_admissible, p.shares_pool_with) for p in peers]
    keys = [f"{node}:{prof.device}" for _, node, prof, _, _ in raw]
    if len(set(keys)) != len(keys):
        raise ValueError(f"device identities must be unique: {keys}")
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            k = parent[k]
        return k

    first: dict[tuple[str, str], str] = {}
    for key, (_, node, prof, _, share) in zip(keys, raw, strict=True):
        links = [first.setdefault((node, prof.pool_id), key)]
        links += _share_targets(key, share, host, peers, prof.pool_id) if share is not None else []
        for other in links:
            parent[find(key)] = find(other)
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(find(key), []).append(i)
    group_of = {i: ("+".join(keys[j] for j in idx), min(raw[j][3] for j in idx))
                for idx in members.values() for i in idx}
    parts = [_Part(key=keys[i], node_id=raw[i][1], role=raw[i][0], profile=raw[i][2], group=group_of[i][0],
                   admissible=group_of[i][1]) for i in range(len(raw))]
    gpu = parts[1] if host.gpu is not None else None
    return parts[0], tuple(p for p in parts if p.role == "peer"), gpu


def _context(manifest: ModelManifest, workload: Workload, host: HostSide, slots: int,
             stream_supported: bool | None) -> tuple[_Ctx, tuple[str, ...]]:
    supported = host.stream_supported if stream_supported is None else stream_supported
    notes = [why for ok, why in ((supported, "host does not support bounded streaming"),
                                 (host.storage is not None, "no storage profile for the host"),
                                 (manifest.streamable, "manifest has tensors of unknown size"))
             if not ok]
    out = [f"CPU streaming not considered: {n}" for n in notes]
    if host.storage is None:
        out.append("cold reads of local weights priced with the storage prior")
    if workload.objective != "interactive":
        out.append("the throughput objective is not modelled; scored as single-stream latency")
    out += ["decode is the sum of stage terms for one stream, never the slowest stage alone",
            "KV is f16 K and V per block on its stage at the workload context; the head has none",
            "token embeddings stay file-backed on the host and are not charged to any pool",
            "prefill is stage compute plus one exchange carrying the prompt activations per peer stage; weight "
            "streaming during prefill is not modelled"]
    act = costs.activation_bytes(manifest.n_embd)
    ctx = _Ctx(manifest=manifest, workload=workload, storage=host.storage or costs.storage_prior(host.node_id),
               stream_ok=not notes, slots=slots, kv_per_layer=manifest.kv_bytes_per_layer(workload.ctx_size),
               workspace=costs.workspace_bytes(manifest.n_embd), act_bytes=act,
               head_down_bytes=costs.logits_bytes(manifest.vocab_size) if manifest.vocab_size else act)
    return ctx, tuple(out)


def _provenance(part: _Part, storage: StorageProfile) -> tuple[str, str, str, str]:
    p = part.profile
    hop = (costs.weakest(p.link.rtt_ms, p.link.goodput_bytes_per_s, p.link.round_trips_per_token)
           if part.role == "peer" and p.link is not None else "-")
    io = costs.weakest(storage.read_bytes_per_s, storage.stream_efficiency) if part.role == "cpu" else "-"
    return part.key, p.decode_ms_per_gib.provenance, io, hop


# --------------------------------------------------------------------------------------------------- entry points


def _signature(c: PlacementCandidate) -> tuple[object, ...]:
    return tuple((s.node_id, s.device, s.units, s.streamed_units) for s in c.stages)


def optimize(manifest: ModelManifest, workload: Workload, host: HostSide, peers: Sequence[PeerSide], *,
             slots: int = 2, max_candidates: int = 5, stream_supported: bool | None = None) -> OptimizeResult:
    """Best llama.cpp-expressible layout for the workload, the next best distinct ones, and why others were left out.

    `stream_supported` overrides `host.stream_supported` when given. Raises ValueError on malformed inputs.
    """
    _validate(manifest, workload, host, peers, slots, max_candidates)
    cpu, peer_parts, gpu = _parts(host, peers)
    ctx, notes = _context(manifest, workload, host, slots, stream_supported)
    tail = (gpu,) if gpu else ()
    seqs = [(cpu, *order, *tail) for k in range(len(peer_parts) + 1) for order in itertools.permutations(peer_parts, k)]
    memo: dict[tuple[object, ...], object] = {}
    rejected: dict[str, tuple[int, str]] = {}
    found: dict[tuple[object, ...], PlacementCandidate] = {}
    for score in (lambda st: st.decode_ms, lambda st: st.j_ms(workload)):
        run = _Run(ctx=ctx, score=score, enforce=True, memo=memo, rejected=rejected)
        for seq in seqs:
            if (stages := _search(run, seq)) is not None:
                cand = _candidate(ctx, _refill(ctx, stages))
                found.setdefault(_signature(cand), cand)
    pool = sorted(found.values(), key=lambda c: (c.objective_ms, c.decode_ms))
    reasons = tuple(r for _, r in sorted(rejected.values(), key=lambda v: v[1]))
    if pool:
        ranked = tuple(replace(c, skipped=_skipped(c, pool, peer_parts, rejected, workload.warm))
                       for c in pool[:max_candidates])
        best: PlacementCandidate | None = ranked[0]
    else:  # per device order the layout with the least overflow, then the one overflowing least overall
        loose = _Run(ctx=ctx, score=lambda st: float(_excess((st,))), enforce=False)
        layouts = [_refill(ctx, s) for seq in seqs if (s := _search(loose, seq)) is not None]
        closest = min(layouts, key=lambda s: (_excess(s), sum(st.decode_ms for st in s)), default=None)
        ranked, best = ((_candidate(ctx, closest),) if closest else ()), None
        reasons = (f"no feasible layout among {len(seqs)} device orders",) + reasons
    parts = (cpu, *peer_parts, *tail)
    return OptimizeResult(best=best, candidates=ranked, rejected=reasons,
                          assumptions=notes + (best.assumptions if best else ()), workload=workload,
                          head_unit=manifest.n_layer, budgets=tuple((p.key, p.admissible) for p in parts),
                          provenance=tuple(_provenance(p, ctx.storage) for p in parts))


def evaluate_layout(manifest: ModelManifest, workload: Workload, host: HostSide, peers: Sequence[PeerSide],
                    layout: Sequence[tuple[str, int, int]], *, slots: int = 2,
                    stream_supported: bool | None = None) -> PlacementCandidate:
    """Score one given chain of (node:device, start unit, end unit) spans with the optimizer's formulas.

    The CPU split is `costs.split_cpu` for its pool's budget, minus what later stages on that pool need. A layout that
    overflows a pool comes back with feasible False and reasons; one llama.cpp cannot express raises ValueError.
    """
    _validate(manifest, workload, host, peers, slots, 1)
    cpu, peer_parts, gpu = _parts(host, peers)
    ctx, _ = _context(manifest, workload, host, slots, stream_supported)
    by_key = {p.key: p for p in (cpu, *peer_parts, *((gpu,) if gpu else ()))}
    n = len(manifest.units)
    keys = [k for k, _, _ in layout]
    bounds = [0] + [e for _, _, e in layout]
    if [s for _, s, _ in layout] != bounds[:-1] or bounds[-1] != n or any(e <= s for _, s, e in layout):
        raise ValueError("layout spans must be non-empty, contiguous and cover every unit")
    if any(k not in by_key for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("layout names an unknown or repeated device")
    if cpu.key in keys[1:] or (gpu is not None and keys[-1] != gpu.key):
        raise ValueError("the CPU may only hold the prefix and the local GPU must hold the suffix with the head")
    run = _Run(ctx=ctx, score=lambda st: st.decode_ms, enforce=False)
    stages = tuple(_cpu_stages(run, by_key[k], e, False)[0] if k == cpu.key else _stage(run, by_key[k], s, e)
                   for k, s, e in layout)
    return _candidate(ctx, _refill(ctx, stages))  # type: ignore[arg-type]
