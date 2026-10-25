"""The bounded stream tier: resident/streamed split, ladder position, plan fields, engine arguments and stats checks."""

from pathlib import Path

import pytest
from hypercompute.capacity import runtime
from hypercompute.schemas import Plan, PlanStrategy, StreamSpec
from hypercompute.services.planner import StreamCapability, make_plan, stream_split, stream_tier_layout
from hypercompute.units import GiB, MiB
from test_split_order import args_for, cpu, gpu, mac, req, shape05, shape14


def cap14(slots: int = 2) -> StreamCapability:
    # qwen2.5-14b q6_k shape: 48 blocks of 225 MiB plus a 609 MiB head, as aligned unit buffers
    return StreamCapability(unit_buffer_bytes=(225 * MiB,) * 48 + (609 * MiB,), slots=slots)


def cap05() -> StreamCapability:
    return StreamCapability(unit_buffer_bytes=(int(10.5 * MiB),) * 24 + (89 * MiB,), slots=2)


# ----------------------------------------------------------------------------- split


def test_split_keeps_everything_resident_when_it_fits():
    assert stream_split([0, 1, 2], (10, 20, 30), 60, 2) == ([0, 1, 2], [], 0)


def test_split_streams_the_fewest_bytes_and_forces_units_larger_than_the_slot_resident():
    sizes = (100, 100, 400, 100, 100)
    resident, streamed, slot = stream_split([0, 1, 2, 3, 4], sizes, 700, 2)
    # a 100-byte slot pair costs 200; the 400-byte unit must then be resident; 100 left for one more unit
    assert slot == 100 and 2 in resident and len(streamed) == 3
    assert sum(sizes[u] for u in resident) + 2 * slot <= 700


def test_split_is_none_when_even_the_smallest_slots_do_not_fit():
    assert stream_split([0, 1], (50, 60), 90, 2) is None


def test_split_with_one_slot_needs_half_the_slot_memory():
    resident, streamed, slot = stream_split([0, 1, 2, 3], (100,) * 4, 250, 1)
    assert slot == 100 and len(resident) == 1 and len(streamed) == 3


def test_split_never_mutates_and_lists_each_unit_once():
    units = [3, 1, 2]
    resident, streamed, _ = stream_split(units, (0, 10, 10, 10), 25, 1)
    assert units == [3, 1, 2] and sorted(resident + streamed) == [1, 2, 3]


# ----------------------------------------------------------------------------- layout and ladder


def test_cpu_only_layout_holds_every_unit_and_the_head_within_the_budget():
    shape, c = shape05(), cap05()
    layout, spec, ok = stream_tier_layout(shape, [], cpu(0.6), False, c)
    assert ok and len(layout) == 1 and layout[0].holds_output and layout[0].layers == list(range(24))
    assert sorted(spec.resident_units + spec.streamed_units) == list(range(25))
    assert layout[0].total_bytes_est <= layout[0].budget_bytes


def test_stream_tier_is_used_only_when_the_ram_tier_does_not_fit():
    shape = shape14()
    ram_ok = make_plan(req(PlanStrategy.local_only), shape, [gpu(8)], [], 40 * GiB, cpu(12), stream=cap14())
    assert ram_ok.tier == "gpu_ram" and ram_ok.stream is None
    tight = make_plan(req(PlanStrategy.local_only), shape, [gpu(8)], [], 6 * GiB, cpu(4), stream=cap14())
    assert tight.feasible and tight.tier == "stream" and tight.capacity_mode
    assert tight.stream.streamed_units and tight.stream.committed_bytes <= 4 * GiB
    cpu_a = tight.assignments[0]
    assert cpu_a.device == "CPU" and cpu_a.total_bytes_est <= cpu_a.budget_bytes
    assert any("stream from the GGUF" in w for w in tight.warnings)


def test_without_the_capability_the_old_disk_tier_remains():
    plan = make_plan(req(PlanStrategy.local_only), shape14(), [gpu(8)], [], 6 * GiB, cpu(4))
    assert plan.tier == "gpu_disk" and plan.stream is None


def test_capacity_mode_off_disables_the_stream_tier():
    plan = make_plan(req(PlanStrategy.local_only, allow_capacity_mode=False), shape14(), [gpu(8)], [], 6 * GiB, cpu(4),
                     stream=cap14())
    assert plan.tier != "stream" and not plan.feasible


def test_a_joined_peer_removes_streamed_units():
    shape = shape14()
    alone = make_plan(req(PlanStrategy.auto), shape, [gpu(8)], [], 6 * GiB, cpu(4), stream=cap14())
    together = make_plan(req(PlanStrategy.auto), shape, [gpu(8)], [mac(9)], 6 * GiB, cpu(4), stream=cap14())
    assert alone.tier == "stream"
    streamed_alone = alone.stream.stream_bytes_per_pass
    streamed_together = together.stream.stream_bytes_per_pass if together.stream else 0
    assert streamed_together < streamed_alone


def test_cpu_only_stream_plan_has_no_devices_and_zero_gpu_units():
    plan = make_plan(req(PlanStrategy.local_only), shape05(), [], [], 2 * GiB, cpu(0.6), stream=cap05())
    assert plan.feasible and plan.tier == "stream" and plan.device_list == [] and plan.n_gpu_units == 0
    args = args_for(plan)
    assert args[args.index("--device") + 1] == "none" and args[args.index("-ngl") + 1] == "0"
    for flag in ("--no-repack", "--no-op-offload"):
        assert flag in args
    assert args[args.index("--load-mode") + 1] == "mmap" and args[args.index("--cache-ram") + 1] == "0"


def test_stream_plan_with_a_gpu_keeps_the_device_split():
    plan = make_plan(req(PlanStrategy.local_only), shape14(), [gpu(8)], [], 6 * GiB, cpu(4), stream=cap14())
    args = args_for(plan)
    assert args[args.index("--device") + 1] == "CUDA0"
    assert int(args[args.index("-ngl") + 1]) == plan.n_gpu_units == 49 - plan.cpu_layers
    assert "--no-host" not in args and args[args.index("--load-mode") + 1] == "mmap"


def test_plan_with_stream_spec_survives_json():
    plan = make_plan(req(PlanStrategy.local_only), shape05(), [], [], 2 * GiB, cpu(0.6), stream=cap05())
    again = Plan.model_validate_json(plan.model_dump_json())
    assert again.stream == plan.stream and again.revision == 0


# ----------------------------------------------------------------------------- runtime helpers


def test_model_shards_lists_every_part_of_a_split_gguf(tmp_path: Path):
    first = tmp_path / "qwen-00001-of-00003.gguf"
    assert [p.name for p in runtime.model_shards(first)] == [
        "qwen-00001-of-00003.gguf", "qwen-00002-of-00003.gguf", "qwen-00003-of-00003.gguf"]
    single = tmp_path / "qwen.gguf"
    assert runtime.model_shards(single) == [single]


def _spec(**kw) -> StreamSpec:
    base = dict(resident_units=[0, 1], streamed_units=[2, 3], slots=2, slot_bytes=4096, resident_bytes=8192,
                stream_bytes_per_pass=8192)
    return StreamSpec(**{**base, **kw})


@pytest.mark.parametrize(("stats", "ok", "needle"), [
    (None, False, "did not attach"),
    ({"state": "running", "slots": 2, "slot_bytes": 4096, "resident_bytes": 8192, "bytes_read": 0}, True, "resident"),
    ({"state": "running", "slots": 2, "slot_bytes": 4096, "resident_bytes": 0, "bytes_read": 0}, True, "0 B loaded"),
    ({"state": "shutdown", "slots": 2, "slot_bytes": 4096}, False, "state shutdown"),
    ({"state": "running", "slots": 2, "slot_bytes": 8192}, False, "slot bytes"),
    ({"state": "running", "slots": 2, "slot_bytes": 4096, "verify_failures": 1}, False, "verify_failures"),
    ({"state": "running", "slots": 2, "slot_bytes": 4096, "resident_bytes": 4096}, False, "resident bytes"),
    ({"state": "running", "slots": 2, "slot_bytes": 4096, "slot_bytes_committed": 8192}, True, "streamed"),
    ({"state": "running", "slots": 2, "slot_bytes": 4096, "slot_bytes_committed": 4096}, False, "committed"),
    ({"state": "passthrough", "slots": 2, "slot_bytes": 4096}, False, "manages no layer"),
])
def test_verify_stats(stats, ok, needle):
    got, detail = runtime.verify_stats(stats, _spec())
    assert got is ok and needle in detail


def test_stats_summary_derives_per_pass_bytes():
    s = runtime.stats_summary({"passes": 4, "bytes_read": 400, "read_ns": 2_000_000_000, "stall_ns": 1_500_000_000,
                               "updated_unix_ms": 0})
    assert s["available"] and s["bytes_per_pass"] == 100 and s["stall_s"] == 1.5
    assert runtime.stats_summary(None) == {"available": False}


def test_storage_rate_prefers_a_fresh_measurement(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HYPERCOMPUTE_HOME", str(tmp_path))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    rate, prov = runtime.storage_rate(model)
    assert prov == "prior" and rate == runtime.PRIOR_READ_BYTES_PER_S
    volume = model.resolve().drive or model.resolve().anchor
    runtime.record_storage_rate(volume, 5.5 * GiB, readers=4, chunk_bytes=8 << 20, sample_bytes=2 << 30, source="test")
    rate, prov = runtime.storage_rate(model)
    assert prov == "measured" and rate == pytest.approx(5.5 * GiB)
