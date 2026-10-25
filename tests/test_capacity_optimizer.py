"""Placement optimizer and cost priors on synthetic manifests: oracle agreement, head and pool rules, cost formulas."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from hypercompute.capacity import costs
from hypercompute.capacity.optimizer import HostSide, PeerSide, evaluate_layout, optimize
from hypercompute.capacity.types import Coefficient, ModelManifest, PlacementCandidate, UnitSpec, Workload
from hypercompute.services import planner

MIB = 1 << 20
GIB = 1 << 30
WARM = Workload(ctx_size=1024, prompt_tokens=16, output_tokens=16, warm=True)


def make_manifest(blocks_mib: Sequence[float], head_mib: float, *, n_embd: int = 1024,
                  vocab_size: int | None = None) -> ModelManifest:
    units = [UnitSpec(i, "block", int(b * MIB), int(b * MIB)) for i, b in enumerate(blocks_mib)]
    units.append(UnitSpec(len(blocks_mib), "head", int(head_mib * MIB), int(head_mib * MIB)))
    return ModelManifest(
        model_id="synthetic", revision=None, shards=(), architecture="llama", file_type=None,
        n_layer=len(blocks_mib), n_embd=n_embd, n_head=8, n_head_kv=2, key_length=64, value_length=64,
        context_length=4096, rope=(), tokenizer_model=None, vocab_size=vocab_size, chat_template_sha256=None,
        units=tuple(units), token_embd_bytes=0, other_bytes=0, tied_embeddings=False, streamable=True,
        fingerprint="synthetic")


def make_host(cpu_gib: float, gpu_gib: float | None, *, stream: bool = True, storage: bool = True,
              node: str = "win", uma: bool = False) -> HostSide:
    if uma:
        cpu, gpu = costs.cpu_device(node, pool_id="uma"), costs.metal_device(node)
    else:
        cpu, gpu = costs.cpu_device(node), costs.cuda_device(node)
    return HostSide(node_id=node, cpu=cpu, cpu_pool_admissible=int(cpu_gib * GIB),
                    storage=costs.storage_prior(node) if storage else None,
                    gpu=gpu if gpu_gib is not None else None, gpu_pool_admissible=int((gpu_gib or 0) * GIB),
                    stream_supported=stream)


def make_peer(node: str, gib: float, path: str = "lan", *, backend: str = "metal", cached: int = 0,
              share: str | None = None) -> PeerSide:
    link = costs.link_prior(path, cached_bytes=cached)  # type: ignore[arg-type]
    device = {"metal": costs.metal_device, "cuda": costs.cuda_device, "cpu": costs.cpu_device}[backend](node, link=link)
    return PeerSide(node_id=node, device=device, pool_admissible=int(gib * GIB), shares_pool_with=share)


def keys(c: PlacementCandidate) -> list[str]:
    return [f"{s.node_id}:{s.device}" for s in c.stages]


def layout_of(c: PlacementCandidate) -> list[tuple[str, int, int]]:
    return [(f"{s.node_id}:{s.device}", s.units[0], s.units[-1] + 1) for s in c.stages]


# --------------------------------------------------------------------------------------------------- oracle


def all_layouts(n: int, cpu: str, peers: Sequence[str], gpu: str | None) -> Iterator[list[tuple[str, int, int]]]:
    """Every chain the contract allows, enumerated directly: no dynamic program, no pruning."""
    for k in range(len(peers) + 1):
        for order in itertools.permutations(peers, k):
            tail = [*order, *([gpu] if gpu else [])]
            for c in range(n + 1):
                if not tail:
                    if c == n:
                        yield [(cpu, 0, n)]
                    continue
                if n - c < len(tail):
                    continue
                for cuts in itertools.combinations(range(c + 1, n), len(tail) - 1):
                    bounds = [c, *cuts, n]
                    spans = [(key, bounds[i], bounds[i + 1]) for i, key in enumerate(tail)]
                    yield ([(cpu, 0, c)] if c else []) + spans


BLOCKS = [300, 200, 450, 250, 350, 150, 400]
ORACLE_CASES = {
    "streaming cpu and gpu": (make_manifest(BLOCKS, 500), WARM, make_host(1.2, 1.3), []),
    "one lan peer": (make_manifest(BLOCKS, 500), WARM, make_host(1.2, 1.3), [make_peer("mac", 1.0)]),
    "two peers cold": (make_manifest(BLOCKS, 500), replace(WARM, warm=False, output_tokens=200),
                       make_host(1.0, 1.6), [make_peer("mac", 1.2), make_peer("box", 1.6, "relay", backend="cuda")]),
    "no gpu, no streaming": (make_manifest(BLOCKS[:5], 300), WARM, make_host(1.0, None, stream=False),
                             [make_peer("mac", 1.2), make_peer("box", 1.6, "direct", backend="cuda")]),
    "uma host": (make_manifest([300] * 6 + [250], 300), WARM,
                 make_host(2.5, 2.5, uma=True, node="m4", stream=False), [make_peer("mac", 1.2)]),
    "six units cached relay": (make_manifest([500, 120, 330, 260, 410], 350),
                               replace(WARM, warm=False, output_tokens=40),
                               make_host(1.2, 1.3), [make_peer("box", 1.5, "relay", backend="cuda", cached=GIB)]),
}


@pytest.mark.parametrize("case", list(ORACLE_CASES))
def test_dp_agrees_with_brute_force_oracle(case: str) -> None:
    manifest, workload, host, peers = ORACLE_CASES[case]
    result = optimize(manifest, workload, host, peers, max_candidates=10_000)
    peer_keys = [f"{p.node_id}:{p.device.device}" for p in peers]
    gpu = f"{host.node_id}:{host.gpu.device}" if host.gpu else None
    scored = [evaluate_layout(manifest, workload, host, peers, lay)
              for lay in all_layouts(len(manifest.units), f"{host.node_id}:CPU", peer_keys, gpu)]
    feasible = [c for c in scored if c.feasible]
    assert feasible, f"{case}: the fixture should admit at least one layout"
    assert result.best is not None
    assert result.best.objective_ms == pytest.approx(min(c.objective_ms for c in feasible), rel=1e-9)
    assert min(c.decode_ms for c in result.candidates) == pytest.approx(min(c.decode_ms for c in feasible), rel=1e-9)
    for cand in result.candidates:  # the search scores layouts exactly as an independent evaluation does
        again = evaluate_layout(manifest, workload, host, peers, layout_of(cand))
        assert again.feasible and again.objective_ms == pytest.approx(cand.objective_ms, rel=1e-9)
    assert [c.objective_ms for c in result.candidates] == sorted(c.objective_ms for c in result.candidates)


def random_case(rng: random.Random) -> tuple[ModelManifest, Workload, HostSide, list[PeerSide], int]:
    m = make_manifest([rng.choice([100, 150, 220, 300, 410, 500]) for _ in range(rng.randint(4, 7))],
                      rng.choice([100, 300, 600]))
    uma = rng.random() < 0.25
    host = make_host(rng.uniform(0.4, 2.2), None if rng.random() < 0.3 else rng.uniform(0.7, 2.0),
                     stream=not uma and rng.random() < 0.6, storage=rng.random() < 0.85, uma=uma)
    peers: list[PeerSide] = []
    for i in range(rng.randint(0, 2)):
        share = peers[0].node_id if i == 1 and rng.random() < 0.3 else None
        peers.append(make_peer(f"p{i}", rng.uniform(0.6, 2.0), rng.choice(["loopback", "lan", "direct", "relay"]),
                               backend=rng.choice(["metal", "cuda", "cpu"]), cached=rng.choice([0, 200 * MIB]),
                               share=share))
    w = Workload(ctx_size=rng.choice([256, 1024, 4096]), prompt_tokens=rng.choice([0, 16, 200]),
                 output_tokens=rng.choice([1, 16, 128, 400]), warm=rng.random() < 0.5)
    return m, w, host, peers, rng.choice([1, 2, 3])


def test_dp_agrees_with_oracle_on_seeded_random_cases() -> None:
    rng = random.Random(7)
    outcomes = {True: 0, False: 0}
    for _ in range(40):
        m, w, host, peers, slots = random_case(rng)
        result = optimize(m, w, host, peers, slots=slots, max_candidates=10_000)
        gpu = f"{host.node_id}:{host.gpu.device}" if host.gpu else None
        peer_keys = [f"{p.node_id}:{p.device.device}" for p in peers]
        feasible = [c for lay in all_layouts(len(m.units), f"{host.node_id}:CPU", peer_keys, gpu)
                    if (c := evaluate_layout(m, w, host, peers, lay, slots=slots)).feasible]
        outcomes[bool(feasible)] += 1
        assert (result.best is None) == (not feasible)
        if result.best is not None:
            assert result.best.objective_ms == pytest.approx(min(c.objective_ms for c in feasible), rel=1e-9)
            assert min(c.decode_ms for c in result.candidates) == pytest.approx(min(c.decode_ms for c in feasible))
    assert outcomes[True] >= 10 and outcomes[False] >= 5  # the sweep covers both outcomes


def test_oracle_cases_exercise_streaming_and_peers() -> None:
    """The oracle cases are only meaningful if the optimum streams in some and uses peers in others."""
    bests = {name: optimize(*args).best for name, args in ORACLE_CASES.items()}
    assert bests["streaming cpu and gpu"] and bests["streaming cpu and gpu"].stream_bytes_per_token > 0
    assert any(b and any(s.path != "local" for s in b.stages) for b in bests.values())


# --------------------------------------------------------------------------------------------------- layout rules


def test_head_goes_to_gpu_then_last_peer_then_cpu() -> None:
    m = make_manifest([400] * 6, 300)
    head = m.n_layer
    with_gpu = optimize(m, WARM, make_host(4.0, 1.0), [make_peer("mac", 4.0)], max_candidates=20)
    for c in with_gpu.candidates:
        assert c.stages[-1].device == "CUDA0" and head in c.stages[-1].units
        assert all(head not in s.units for s in c.stages[:-1])
    no_gpu = optimize(m, WARM, make_host(1.5, None, stream=False), [make_peer("mac", 2.2)], max_candidates=20)
    assert no_gpu.best is not None and keys(no_gpu.best) == ["win:CPU", "mac:MTL0"]
    for c in no_gpu.candidates:
        assert head in c.stages[-1].units and all(head not in s.units for s in c.stages[:-1])
    alone = optimize(m, WARM, make_host(4.0, None), [])
    assert alone.best is not None and keys(alone.best) == ["win:CPU"]
    assert alone.best.stages[0].units == tuple(range(head + 1))


def test_streaming_only_when_needed_and_only_on_cpu() -> None:
    m = make_manifest([512] * 8, 256, n_embd=4096)
    roomy = optimize(m, WARM, make_host(8.0, 1.0), [])
    assert roomy.best is not None and roomy.best.stream_bytes_per_token == 0
    assert all(not s.streamed_units and s.slots == 0 for s in roomy.best.stages)
    tight = optimize(m, WARM, make_host(2.5, 1.0), [make_peer("mac", 1.0)], max_candidates=50)
    assert tight.best is not None and tight.best.stream_bytes_per_token > 0
    for c in tight.candidates:
        for s in c.stages:
            assert not s.streamed_units or s.device == "CPU"
            if s.streamed_units:
                assert s.slots == 2 and s.demand and s.demand.staging == 2 * 512 * MIB
                assert set(s.resident_units) | set(s.streamed_units) == set(s.units)
    no_stream = optimize(m, WARM, make_host(2.5, 1.0, stream=False), [])
    assert no_stream.best is None
    assert any("streaming not available" in r for r in no_stream.rejected)
    no_storage = optimize(m, WARM, make_host(2.5, 1.0, storage=False), [])
    assert no_storage.best is None
    assert any("no storage profile" in a for a in no_storage.assumptions)


def brute_min_stream(units: Sequence[UnitSpec], budget: int, slots: int) -> int:
    best = None
    for r in range(1, len(units) + 1):
        for streamed in itertools.combinations(units, r):
            slot = max(u.buffer_bytes for u in streamed)
            resident = sum(u.buffer_bytes for u in units if u not in streamed)
            if slots * slot + resident <= budget:
                total = sum(u.buffer_bytes for u in streamed)
                best = total if best is None else min(best, total)
    assert best is not None
    return best


def test_slot_size_enumeration_picks_min_streamed_bytes() -> None:
    units = make_manifest([100] * 6, 300).units  # a head three times the block size
    budget = 850 * MIB
    split = costs.split_cpu(units, budget, slots=2, stream=True)
    assert split is not None
    assert split.stream_bytes == brute_min_stream(units, budget, 2) == 300 * MIB
    assert split.slot_bytes == 100 * MIB and 6 in split.resident  # head resident, block-sized slots
    assert split.committed_bytes <= budget
    largest = costs.split_cpu(units, budget, slots=2, stream=False)
    assert largest is None  # 900 MiB of weights cannot all stay mapped in 850 MiB
    assert costs.split_cpu(units, 2 * GIB, slots=2, stream=True) == costs.all_mapped(units)
    front = costs.cpu_split_options(units, budget, slots=2, stream=True)
    assert (front[0].stream_bytes, front[0].committed_bytes) == (split.stream_bytes, split.committed_bytes)
    assert all(a.committed_bytes > b.committed_bytes for a, b in itertools.pairwise(front))


def test_uma_host_shares_one_pool() -> None:
    m = make_manifest([300] * 8, 300)
    peer = [make_peer("mac2", 1.5)]
    shared = optimize(m, WARM, make_host(2.0, 2.0, uma=True, node="m4"), peer)
    assert shared.best is not None and "mac2:MTL0" in keys(shared.best)
    local = [s.demand.peak for s in shared.best.stages if s.node_id == "m4" and s.demand]
    assert sum(local) <= 2 * GIB
    split_pools = replace(make_host(2.0, 2.0, uma=True, node="m4"), cpu=costs.cpu_device("m4", pool_id="ram"))
    apart = optimize(m, WARM, split_pools, peer)
    assert apart.best is not None and "mac2:MTL0" not in keys(apart.best)
    assert any("mac2:MTL0: not used" in s for s in apart.best.skipped)


def test_shared_cpu_pool_streams_more_to_leave_room_for_the_gpu() -> None:
    """On one pool the CPU split that keeps the most resident can starve the GPU stage; the search trades it off."""
    m = make_manifest([300] * 6, 300)
    host = make_host(1.6, 1.6, uma=True, node="m4", stream=True)
    best = optimize(m, WARM, host, []).best
    assert best is not None and layout_of(best) == [("m4:CPU", 0, 6), ("m4:MTL0", 6, 7)]
    assert best.stages[0].resident_units == () and best.stages[0].streamed_units == tuple(range(6))
    assert sum(s.demand.peak for s in best.stages if s.demand) <= int(1.6 * GIB)
    # the rule for the CPU's own budget would keep two units resident and overflow the pool; evaluated with what the
    # GPU stage leaves, the same layout gets the optimizer's split and numbers
    again = evaluate_layout(m, WARM, host, [], layout_of(best))
    assert again.feasible and again.stages[0].streamed_units == best.stages[0].streamed_units
    assert again.objective_ms == pytest.approx(best.objective_ms)


def fast_cpu_uma_host(pool_gib: float) -> HostSide:
    """A unified-memory host whose CPU is measured faster than its GPU, so the CPU takes most units."""
    host = make_host(pool_gib, pool_gib, uma=True, node="m4", stream=True)
    cpu = replace(host.cpu, decode_ms_per_gib=Coefficient(20.0, "ms/GiB", "measured"))
    return replace(host, cpu=cpu, gpu=replace(host.gpu, decode_ms_per_gib=Coefficient(40.0, "ms/GiB", "measured")))


@pytest.mark.parametrize("warm", [True, False])
def test_shared_pool_streams_nothing_when_everything_fits(warm: bool) -> None:
    """Regression: streaming a few small units saves pool bytes at an unchanged score (their I/O hides under compute),
    and a cold J even rewards it with a shorter load. Neither may make a shared-pool CPU stream when all units fit."""
    m = make_manifest([400] * 12 + [10] * 6, 100)
    w = replace(WARM, warm=warm)
    result = optimize(m, w, fast_cpu_uma_host(16.0), [], max_candidates=10)
    assert result.best is not None and layout_of(result.best)[0] == ("m4:CPU", 0, 18)
    assert all(c.stream_bytes_per_token == 0 for c in result.candidates)
    tight = optimize(m, w, fast_cpu_uma_host(5.0), [])  # 5 GiB cannot map 4.8 GiB of CPU weights plus the GPU stage
    assert tight.best is not None and tight.best.stream_bytes_per_token > 0


def shared_case(rng: random.Random) -> tuple[ModelManifest, Workload, HostSide, list[PeerSide], int]:
    m = make_manifest([rng.choice([100, 150, 220, 300, 410]) for _ in range(rng.randint(4, 7))],
                      rng.choice([100, 300, 600]))
    host = make_host(rng.uniform(0.6, 2.4), rng.uniform(0.6, 2.4), uma=True, node="m4", stream=True)
    if rng.random() < 0.5:
        host = replace(host, cpu=replace(host.cpu, decode_ms_per_gib=Coefficient(rng.choice([2.0, 8.0]), "ms/GiB",
                                                                                 "measured")))
    peers = ([make_peer("p0", rng.uniform(0.6, 2.0), rng.choice(["lan", "relay"]), backend=rng.choice(["cpu", "metal"]),
                        share=rng.choice([None, "m4"]))] if rng.random() < 0.6 else [])
    w = Workload(ctx_size=1024, prompt_tokens=rng.choice([0, 16]), output_tokens=rng.choice([1, 16, 200]),
                 warm=rng.random() < 0.5)
    return m, w, host, peers, rng.choice([1, 2])


def test_shared_pool_with_streaming_agrees_with_oracle() -> None:
    """A CPU on a pool later stages share takes `split_cpu` for what they leave; the search must find the optimum of
    exactly those layouts, and `evaluate_layout` must score its picks the same way."""
    rng = random.Random(11)
    feasible_cases = streaming_bests = 0
    for _ in range(60):
        m, w, host, peers, slots = shared_case(rng)
        result = optimize(m, w, host, peers, slots=slots, max_candidates=10_000)
        peer_keys = [f"{p.node_id}:{p.device.device}" for p in peers]
        feasible = [c for lay in all_layouts(len(m.units), "m4:CPU", peer_keys, "m4:MTL0")
                    if (c := evaluate_layout(m, w, host, peers, lay, slots=slots)).feasible]
        assert (result.best is None) == (not feasible)
        if result.best is None:
            continue
        feasible_cases += 1
        streaming_bests += result.best.stream_bytes_per_token > 0
        assert result.best.objective_ms == pytest.approx(min(c.objective_ms for c in feasible), rel=1e-9)
        assert min(c.decode_ms for c in result.candidates) == pytest.approx(min(c.decode_ms for c in feasible))
        for cand in result.candidates:
            again = evaluate_layout(m, w, host, peers, layout_of(cand), slots=slots)
            assert again.feasible and again.objective_ms == pytest.approx(cand.objective_ms, rel=1e-9)
    assert feasible_cases >= 15 and streaming_bests >= 3  # the sample covers shared pools that do stream


def test_cpu_split_options_are_every_split_the_rule_returns() -> None:
    rng = random.Random(3)
    for _ in range(300):
        sizes = [rng.choice([1, 2, 3, 4, 5, 7, 9]) for _ in range(rng.randint(1, 7))]
        units = [UnitSpec(i, "block", s, s) for i, s in enumerate(sizes)]
        slots = rng.choice([1, 2, 3])
        top = sum(sizes) + slots * max(sizes)
        options = costs.cpu_split_options(units, top, slots=slots, stream=True)
        assert all(a.committed_bytes > b.committed_bytes and a.stream_bytes <= b.stream_bytes
                   for a, b in itertools.pairwise(options))
        every = {(o.resident, o.streamed) for b in range(top + 1)
                 if (o := costs.split_cpu(units, b, slots=slots, stream=True)) is not None}
        assert every == {(o.resident, o.streamed) for o in options}


def test_shares_pool_with_couples_two_peers() -> None:
    m = make_manifest([300] * 8, 300)
    host = make_host(1.0, 0.9, stream=False)
    a, b = make_peer("mac", 1.5), make_peer("mac-cpu", 1.5, backend="cpu")
    apart = optimize(m, WARM, host, [a, b])
    assert apart.best is not None and {"mac:MTL0", "mac-cpu:CPU"} <= set(keys(apart.best))
    coupled = optimize(m, WARM, host, [a, replace(b, shares_pool_with="mac")])
    assert coupled.best is None
    assert any("pool" in r and "mac:MTL0+mac-cpu:CPU" in r for r in coupled.rejected)
    assert coupled.candidates and not coupled.candidates[0].feasible


def test_infeasible_returns_reasons() -> None:
    m = make_manifest([400] * 6, 300)
    result = optimize(m, WARM, make_host(0.2, 0.5, stream=False), [])
    assert result.best is None
    assert result.rejected[0].startswith("no feasible layout")
    assert any("needs" in r and "pool admits" in r for r in result.rejected)
    closest = result.candidates[0]
    assert not closest.feasible and closest.reasons
    lines = result.explain()
    assert lines[0] == "no feasible layout" and any(line.startswith("  overflow: ") for line in lines)


# --------------------------------------------------------------------------------------------------- objective


def relay_case(cpu_gib: float) -> tuple[ModelManifest, HostSide, PeerSide]:
    return make_manifest([512] * 8, 256, n_embd=4096), make_host(cpu_gib, 1.0), make_peer("relay-mac", 6.0, "relay")


def test_slow_relay_skipped_when_warm_short_unless_it_removes_streaming() -> None:
    m, roomy_host, relay = relay_case(8.0)
    skipped = optimize(m, WARM, roomy_host, [relay])
    assert skipped.best is not None and "relay-mac:MTL0" not in keys(skipped.best)
    note = next(s for s in skipped.best.skipped if s.startswith("relay-mac:MTL0"))
    assert "not used: adds 71.2 ms hop per token" in note and "ms compute and I/O per token" in note
    assert any(line.startswith("skipped relay-mac:MTL0: not used") for line in skipped.explain())
    _, tight_host, _ = relay_case(1.5)
    used = optimize(m, WARM, tight_host, [relay])
    assert used.best is not None and "relay-mac:MTL0" in keys(used.best)
    alone = optimize(m, WARM, tight_host, [])
    assert alone.best is not None
    assert used.best.stream_bytes_per_token < alone.best.stream_bytes_per_token
    assert used.best.objective_ms < alone.best.objective_ms
    assert any("relay risk" in a for a in used.best.assumptions)


def test_cold_start_reverses_ranking_when_uploads_are_expensive() -> None:
    m, host, relay = relay_case(1.5)
    w = Workload(ctx_size=1024, prompt_tokens=16, output_tokens=64, warm=True)
    warm = optimize(m, w, host, [relay])
    cold = optimize(m, replace(w, warm=False), host, [relay])
    assert warm.best is not None and cold.best is not None
    assert "relay-mac:MTL0" in keys(warm.best) and "relay-mac:MTL0" not in keys(cold.best)
    assert any("relay-mac:MTL0" in keys(c) for c in cold.candidates[1:])
    assert any("s load" in s for s in cold.best.skipped)
    cached = optimize(m, replace(w, warm=False), host, [replace(relay, device=replace(
        relay.device, link=costs.link_prior("relay", cached_bytes=8 * GIB)))])
    assert cached.best is not None and "relay-mac:MTL0" in keys(cached.best)
    assert cached.best.decode_ms == pytest.approx(warm.best.decode_ms)


def test_explain_reports_stages_bottleneck_and_provenance() -> None:
    m, host, _ = relay_case(1.5)
    result = optimize(m, WARM, host, [])
    assert result.best is not None
    assert "win:CPU exposed storage I/O" in result.best.bottleneck
    lines = result.explain()
    assert f"bottleneck: {result.best.bottleneck}" in lines
    assert lines[0].startswith("best: J ") and "risk 0;" in lines[0]  # no relay stage: no risk, never "-0"
    cpu_line = next(line for line in lines if line.startswith("stage 1 win:CPU"))
    assert "streamed" in cpu_line and "exposed I/O" in cpu_line and "(prior)" in cpu_line and "admissible" in cpu_line
    assert any(line.startswith("assumption: win:CPU decode") and "prior" in line for line in lines)


def test_decode_is_a_sum_of_stage_terms() -> None:
    m, host, relay = relay_case(1.5)
    best = optimize(m, WARM, host, [relay]).best
    assert best is not None and len(best.stages) >= 2
    terms = [s.compute_ms + s.exposed_io_ms + s.hop_ms for s in best.stages]
    assert best.decode_ms == pytest.approx(sum(terms))
    assert best.decode_ms > max(terms)


def test_stage_costs_follow_the_contract_formulas() -> None:
    m = make_manifest([256] * 6, 128, n_embd=2048)
    w = Workload(ctx_size=512, prompt_tokens=10, output_tokens=7, warm=False)
    host = make_host(0.9, 1.0)
    peer = make_peer("mac", 2.0, "relay", cached=100 * MIB)
    cand = evaluate_layout(m, w, host, [peer], [("win:CPU", 0, 3), ("mac:MTL0", 3, 5), ("win:CUDA0", 5, 7)])
    assert cand.feasible
    cpu_stage, peer_stage, gpu_stage = cand.stages
    store, link, gpu = host.storage, peer.device.link, host.gpu
    assert store is not None and link is not None and gpu is not None
    read, eff = store.read_bytes_per_s.value, store.stream_efficiency.value
    # 0.9 GiB minus KV, workspace and overhead leaves 729 MiB: 768 MiB cannot stay mapped, two 256 MiB slots fit and
    # no block fits beside them, so all three blocks stream
    assert cpu_stage.streamed_units == (0, 1, 2) and cpu_stage.resident_units == () and cpu_stage.slots == 2
    assert cand.stream_bytes_per_token == 768 * MIB
    io = 768 * MIB / (read * eff) * 1000
    compute = host.cpu.decode_ms_per_gib.value * 0.75
    assert cpu_stage.compute_ms == pytest.approx(compute)
    assert cpu_stage.exposed_io_ms == pytest.approx(max(io, compute) - compute)
    act = 2048 * 4
    hop = 2 * 20.0 + 2 * act / MIB * 1000
    assert peer_stage.hop_ms == pytest.approx(hop)
    assert peer_stage.compute_ms == pytest.approx(peer.device.decode_ms_per_gib.value * 0.5)
    assert peer_stage.upload_bytes == 412 * MIB
    assert peer_stage.demand and peer_stage.demand.staging == costs.RPC_STAGING_BYTES
    assert peer_stage.demand.kv == 2 * m.kv_bytes_per_layer(512)
    assert gpu_stage.demand and gpu_stage.demand.kv == m.kv_bytes_per_layer(512)  # block 5 only; the head has none
    load = 384 * MIB / read * 1000 + 412 * 1000 + 500.0 + 2000.0  # GPU read, upload at 1 MiB/s, peer setup, setup
    assert cand.load_ms == pytest.approx(load)
    decode = max(io, compute) + peer_stage.compute_ms + hop + gpu_stage.compute_ms
    assert cand.decode_ms == pytest.approx(decode)
    prefill = 10 * (host.cpu.prefill_ms_per_gib_token.value * 0.75 + peer.device.prefill_ms_per_gib_token.value * 0.5
                    + gpu.prefill_ms_per_gib_token.value * 0.375)
    prefill += 2 * 20.0 + 2 * 10 * act / MIB * 1000
    assert cand.prefill_ms == pytest.approx(prefill)
    risk = 0.10 * (peer_stage.compute_ms + hop) * 7
    assert cand.objective_ms == pytest.approx(load + prefill + 7 * decode + risk)
    single = evaluate_layout(m, w, host, [peer], [("win:CPU", 0, 3), ("mac:MTL0", 3, 5), ("win:CUDA0", 5, 7)],
                             slots=1)
    assert single.stream_bytes_per_token == 512 * MIB  # one slot frees room for one resident block
    one_io = 512 * MIB / (read * eff) * 1000
    assert single.stages[0].exposed_io_ms == pytest.approx(one_io)  # one slot: I/O adds to compute
    assert single.decode_ms - single.stages[0].exposed_io_ms == pytest.approx(decode - (max(io, compute) - compute))


def test_peer_holding_the_head_returns_logits() -> None:
    """Regression: llama.cpp copies n_vocab * 4 bytes of logits back from the device holding the head every token
    (docs/research/llama_cpu_tier.md A2), not one hidden state."""
    vocab, n_embd = 152_064, 2048
    m = make_manifest([256] * 6, 300, n_embd=n_embd, vocab_size=vocab)
    w = Workload(ctx_size=512, prompt_tokens=10, output_tokens=7, warm=True)
    host, peer = make_host(4.0, None, stream=False), make_peer("mac", 4.0, "lan")
    link = peer.device.link
    assert link is not None
    cand = evaluate_layout(m, w, host, [peer], [("win:CPU", 0, 3), ("mac:MTL0", 3, 7)])
    act, logits = n_embd * 4, vocab * 4
    rtt, goodput = link.rtt_ms.value, link.goodput_bytes_per_s.value
    assert cand.stages[1].hop_ms == pytest.approx(2 * rtt + (act + logits) / goodput * 1000)
    compute = sum(s.compute_ms for s in cand.stages)
    assert cand.decode_ms == pytest.approx(compute + cand.stages[1].hop_ms)
    prefill_compute = 10 * (host.cpu.prefill_ms_per_gib_token.value * 768 / 1024
                            + peer.device.prefill_ms_per_gib_token.value * (768 + 300) / 1024)
    assert cand.prefill_ms == pytest.approx(prefill_compute + 2 * rtt + (10 * act + logits) / goodput * 1000)
    assert any("holds the head" in a and "logits" in a for a in cand.assumptions)
    middle = evaluate_layout(m, w, make_host(4.0, 1.0, stream=False), [peer],
                             [("win:CPU", 0, 2), ("mac:MTL0", 2, 4), ("win:CUDA0", 4, 7)])
    assert middle.stages[1].hop_ms == pytest.approx(2 * rtt + 2 * act / goodput * 1000)  # head local: hidden states
    unknown = evaluate_layout(replace(m, vocab_size=None), w, host, [peer], [("win:CPU", 0, 3), ("mac:MTL0", 3, 7)])
    assert unknown.stages[1].hop_ms == pytest.approx(2 * rtt + 2 * act / goodput * 1000)
    assert any("vocab size unknown" in a for a in unknown.assumptions)


def overflow_bytes(c: PlacementCandidate, budgets: dict[str, int]) -> int:
    return sum(max(0, s.demand.peak - budgets[f"{s.node_id}:{s.device}"]) for s in c.stages if s.demand)


def test_infeasible_reports_the_least_overflowing_layout() -> None:
    """Regression: the diagnostic layout was the fastest one with memory ignored but was called the closest."""
    m = make_manifest([400] * 6, 300)
    host = make_host(0.9, 0.9, stream=False)
    result = optimize(m, WARM, host, [])
    assert result.best is None and len(result.candidates) == 1
    budgets = dict(result.budgets)
    scored = [evaluate_layout(m, WARM, host, [], lay) for lay in all_layouts(len(m.units), "win:CPU", [], "win:CUDA0")]
    least = min(overflow_bytes(c, budgets) for c in scored)
    assert overflow_bytes(result.candidates[0], budgets) == least
    fastest = min(scored, key=lambda c: c.decode_ms)
    assert overflow_bytes(fastest, budgets) > least  # the old choice would have been a different, worse layout
    assert any(line.startswith("closest layout with memory ignored") for line in result.explain())


# --------------------------------------------------------------------------------------------------- validation


def test_inputs_are_validated() -> None:
    m = make_manifest([300] * 6, 300)
    host = make_host(2.0, 1.0)
    with pytest.raises(ValueError, match="at most 4 peers"):
        optimize(m, WARM, host, [make_peer(f"p{i}", 1.0) for i in range(5)])
    no_link = PeerSide(node_id="x", device=costs.metal_device("x"), pool_admissible=GIB)
    with pytest.raises(ValueError, match="no link profile"):
        optimize(m, WARM, host, [no_link])
    with pytest.raises(ValueError, match="unique"):
        optimize(m, WARM, host, [make_peer("mac", 1.0), make_peer("mac", 2.0)])
    with pytest.raises(ValueError, match="names no other device"):
        optimize(m, WARM, host, [make_peer("mac", 1.0, share="nowhere")])
    with pytest.raises(ValueError, match="non-negative"):
        optimize(m, WARM, replace(host, cpu_pool_admissible=-1), [])
    with pytest.raises(ValueError, match="backend"):
        optimize(m, WARM, replace(host, gpu=replace(costs.cuda_device("win"), backend="rocm")), [])
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_layout(m, WARM, host, [], [("win:CPU", 0, 3), ("win:CUDA0", 4, 7)])
    with pytest.raises(ValueError, match="local GPU must hold the suffix"):
        evaluate_layout(m, WARM, host, [], [("win:CPU", 0, 7)])


# --------------------------------------------------------------------------------------------------- costs


def test_priors_are_labelled_and_match_the_documented_values() -> None:
    expected = {"loopback": (0.1, 3e9), "lan": (2.0, 60e6), "direct": (8.0, 30e6), "relay": (20.0, float(MIB))}
    for path, (rtt, goodput) in expected.items():
        link = costs.link_prior(path)  # type: ignore[arg-type]
        assert (link.rtt_ms.value, link.goodput_bytes_per_s.value) == (rtt, goodput)
        assert link.round_trips_per_token.value == 2.0
        for c in (link.rtt_ms, link.goodput_bytes_per_s, link.round_trips_per_token):
            assert c.provenance == "prior" and c.note
    store = costs.storage_prior("win")
    assert (store.read_bytes_per_s.value, store.stream_efficiency.value) == (3 * GIB, 0.7)
    assert store.read_bytes_per_s.provenance == store.stream_efficiency.provenance == "prior"
    for make, backend in ((costs.cpu_device, "cpu"), (costs.cuda_device, "cuda"), (costs.metal_device, "metal")):
        d = make("n")
        assert d.decode_ms_per_gib.value == pytest.approx(costs.DECODE_MS_PER_GB[backend] * GIB / 1e9)
        assert d.decode_ms_per_gib.provenance == d.prefill_ms_per_gib_token.provenance == "prior"
    assert costs.relay_risk_prior().value == 0.10 and costs.relay_risk_prior().provenance == "prior"
    with pytest.raises(ValueError):
        costs.link_prior("local")


def test_cost_priors_match_the_planner() -> None:
    for backend, value in costs.DECODE_MS_PER_GB.items():
        assert planner.DECODE_MS_PER_GB[backend] == value
    assert costs.OVERHEAD_BYTES == planner.OVERHEAD_BY_KIND
    assert (costs.UBATCH, costs.COMPUTE_BASE_BYTES, costs.RPC_STAGING_BYTES) == (
        planner.UBATCH, planner.COMPUTE_BASE, planner.RPC_STAGING)
    assert costs.workspace_bytes(5120) == planner.COMPUTE_BASE + planner.UBATCH * 5120 * 4 * 8


def test_merge_prefers_fresh_measurements() -> None:
    now = datetime(2026, 10, 21, tzinfo=UTC)
    prior = costs.cuda_device("win")
    measured_decode = Coefficient(2.5, "ms/GiB", "measured", measured_at=now, samples=5)
    measured = replace(prior, decode_ms_per_gib=measured_decode)
    merged = costs.merge(prior, measured)
    assert merged.decode_ms_per_gib == measured_decode
    assert merged.prefill_ms_per_gib_token == prior.prefill_ms_per_gib_token  # the measurement had only a prior
    stale = replace(measured_decode, measured_at=now - timedelta(days=30))
    kept = costs.merge(prior, replace(prior, decode_ms_per_gib=stale), now=now, max_age_s=86400)
    assert kept.decode_ms_per_gib == prior.decode_ms_per_gib
    link = costs.link_prior("lan")
    rtt = Coefficient(3.1, "ms", "measured", measured_at=now, samples=20)
    assert costs.merge(link, replace(link, rtt_ms=rtt, cached_bytes=GIB)).rtt_ms == rtt
    with pytest.raises(ValueError, match="identity"):
        costs.merge(prior, costs.cuda_device("other"))
    with pytest.raises(ValueError, match="cannot merge a measurement"):
        costs.merge(prior.decode_ms_per_gib, Coefficient(2.5, "ms/GB", "measured"))
    with pytest.raises(TypeError):
        costs.merge(prior, link)  # type: ignore[type-var]


def test_merge_reports_broken_measurements_and_freshness_misuse() -> None:
    """Regression: a NaN or negative measurement, or max_age_s without `now`, fell back to the prior silently."""
    now = datetime(2026, 10, 21, tzinfo=UTC)
    prior = costs.cuda_device("win")
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="not a finite non-negative"):
            costs.merge(prior, replace(prior, decode_ms_per_gib=Coefficient(bad, "ms/GiB", "measured")))
    fresh = replace(prior, decode_ms_per_gib=Coefficient(2.5, "ms/GiB", "measured", measured_at=now))
    with pytest.raises(ValueError, match="needs `now`"):
        costs.merge(prior, fresh, max_age_s=60)
    with pytest.raises(ValueError, match="needs `now`"):
        costs.merge(prior, fresh, now=now, max_age_s=-1)
    future = replace(prior, decode_ms_per_gib=replace(fresh.decode_ms_per_gib, measured_at=now + timedelta(hours=1)))
    assert costs.merge(prior, future, now=now, max_age_s=86400).decode_ms_per_gib == prior.decode_ms_per_gib
    assert costs.merge(prior, fresh, now=now, max_age_s=60).decode_ms_per_gib == fresh.decode_ms_per_gib
    assert costs.merge(prior, None, max_age_s=None) is prior


def test_smallest_split_commits_the_least_memory() -> None:
    units = make_manifest([100, 100, 100, 100], 300).units
    small = costs.smallest_split(units, slots=2, stream=True)
    # head resident with two 100 MiB slots (500 MiB) beats two 300 MiB slots (600 MiB) and mapping all (700 MiB)
    assert small.committed_bytes == 500 * MIB and small.resident == (4,) and small.streamed == (0, 1, 2, 3)
    brute = min(2 * max(u.buffer_bytes for u in s) + sum(u.buffer_bytes for u in units if u not in s)
                for r in range(1, len(units) + 1) for s in itertools.combinations(units, r))
    assert small.committed_bytes == brute
    assert costs.smallest_split(units, slots=2, stream=False) == costs.all_mapped(units)


def test_coefficients_and_ownership_are_validated() -> None:
    """Regression: a NaN coefficient made every comparison in the search false, so a wrong layout won silently."""
    m = make_manifest([300] * 6, 300)
    host = make_host(2.0, 1.0)
    nan = Coefficient(float("nan"), "ms/GiB", "measured")
    with pytest.raises(ValueError, match="decode_ms_per_gib"):
        optimize(m, WARM, replace(host, cpu=replace(host.cpu, decode_ms_per_gib=nan)), [])
    zero = make_peer("mac", 2.0)
    assert zero.device.link is not None
    dead = replace(zero.device.link, goodput_bytes_per_s=Coefficient(0.0, "bytes/s", "measured"))
    with pytest.raises(ValueError, match="goodput"):
        optimize(m, WARM, host, [replace(zero, device=replace(zero.device, link=dead))])
    store = host.storage
    assert store is not None
    with pytest.raises(ValueError, match="stream_efficiency"):
        optimize(m, WARM, replace(host, storage=replace(store, stream_efficiency=Coefficient(-0.1, "fraction",
                                                                                              "measured"))), [])
    with pytest.raises(ValueError, match="belong to the node"):
        optimize(m, WARM, host, [replace(zero, node_id="elsewhere")])
    with pytest.raises(ValueError, match="vocab_size"):
        optimize(replace(m, vocab_size=0), WARM, host, [])
