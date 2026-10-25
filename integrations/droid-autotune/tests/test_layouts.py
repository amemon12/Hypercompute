from __future__ import annotations

from autotune import layouts
from autotune.models import Stage
from autotune.pool import DeviceView, NodeFacts, PoolSnapshot

GiB = 1024**3


def peer(nid: str, name: str, devices: tuple[str, ...]) -> NodeFacts:
    return NodeFacts(id=nid, name=name, is_local=False, state="eligible",
                     devices=tuple(DeviceView(d, d, None, None) for d in devices), max_bytes=None, reserve_bytes=None)


SNAP = PoolSnapshot(nodes=(
    NodeFacts(id="local", name="gx10", is_local=True, state="eligible", devices=(DeviceView("CUDA0", "GB10", None,
                                                                                              None),),
              max_bytes=None, reserve_bytes=None),
    peer("g14id", "G14", ("CUDA0",)), peer("macid", "mac-a", ("MTL0", "CPU"))))


def a(node: str, device: str, layers: range, endpoint: str | None = None, **kw) -> dict:  # noqa: ANN003
    return {"node_id": node, "device": device, "endpoint": endpoint, "layers": list(layers),
            "total_bytes_est": kw.get("total", len(layers) * GiB), "budget_bytes": kw.get("budget", 20 * GiB),
            "fits": True, "holds_output": kw.get("holds_output", False), "file_backed": kw.get("file_backed", False)}


ALONE = {"id": "p0", "tier": "gpu", "rpc_endpoints": [], "assignments": [
    a("local", "CUDA0", range(0, 36), holds_output=True)], "prediction": None, "moe": None, "candidates": []}

# G14 first (RPC0), then mac-a (RPC1 = MTL0, RPC2 = CPU); the plan uses mac-a's CPU only
POOLED = {"id": "p1", "tier": "moe_offload", "rpc_endpoints": ["10.0.0.2:50052", "10.0.0.3:50052"], "assignments": [
    a("macid", "RPC2", range(3, 6), "10.0.0.3:50052", file_backed=True),
    a("g14id", "RPC0", range(0, 3), "10.0.0.2:50052"),
    a("local", "CUDA0", range(6, 36), holds_output=True)],
    "moe": {"decode_ms_est": 410.5, "ssd_bytes_per_token_est": 40 * 2**20}, "prediction": None, "candidates": []}


def test_rows_name_peer_devices_as_their_worker_does_and_run_in_layer_order() -> None:
    rows = layouts.rows(POOLED, SNAP)
    assert [(r.node_name, r.device, r.layers) for r in rows] == [("G14", "CUDA0", 3), ("mac-a", "CPU", 3),
                                                                ("gx10", "CUDA0", 30)]
    assert rows[1].file_backed and rows[2].holds_output


def test_stages_reproduce_the_plan_as_an_explicit_layout() -> None:
    stages = layouts.stages(layouts.rows(POOLED, SNAP))
    assert stages == [Stage(node_id="g14id", device="CUDA0", layers=3),
                      Stage(node_id="macid", device="CPU", layers=3, residency="file_backed"),
                      Stage(node_id="local", device="CUDA0", layers=30)]


def test_a_plan_with_host_cpu_layers_has_no_layout() -> None:
    plan = {**ALONE, "assignments": [a("local", "CPU", range(0, 4)), a("local", "CUDA0", range(4, 36),
                                                                       holds_output=True)]}
    assert layouts.stages(layouts.rows(plan, SNAP)) is None


def test_hops_count_machine_boundaries() -> None:
    assert layouts.hops(layouts.rows(ALONE, SNAP)) == 0
    assert layouts.hops(layouts.rows(POOLED, SNAP)) == 2


def test_prediction_sources_are_labelled() -> None:
    assert layouts.prediction(POOLED).decode_source == "moe"
    assert layouts.prediction(POOLED).decode_ms == 410.5
    optimizer = {**ALONE, "prediction": {"decode_ms": 17.9, "load_ms": 40000.0, "stream_bytes_per_token": 0}}
    assert layouts.prediction(optimizer).decode_source == "prediction"
    note = {**ALONE, "candidates": [{"feasible": True, "reason": "estimated fit; predicted decode ~165.7 ms/token "
                                     "(priors)"}]}
    assert layouts.prediction(note).decode_ms == 165.7
    assert layouts.prediction(ALONE).decode_ms is None


def test_diff_shows_layers_moving_per_machine_and_device() -> None:
    d = layouts.layout_diff(ALONE, POOLED, SNAP)
    by = {(r.node_name, r.device): r for r in d.rows}
    assert (by[("gx10", "CUDA0")].layers_before, by[("gx10", "CUDA0")].layers_after) == (36, 30)
    assert by[("gx10", "CUDA0")].range_after == "6-35"
    assert (by[("G14", "CUDA0")].layers_before, by[("G14", "CUDA0")].layers_after) == (0, 3)
    assert (d.hops_before, d.hops_after, d.machines_before, d.machines_after) == (0, 2, 1, 3)
    assert d.decode_ms_after == 410.5 and d.decode_ms_before is None  # different sources are never compared
    assert not d.unchanged


def test_diff_against_itself_is_unchanged() -> None:
    assert layouts.layout_diff(ALONE, ALONE, SNAP).unchanged


def test_diff_without_a_current_plan() -> None:
    d = layouts.layout_diff(None, ALONE, SNAP)
    assert d.hops_before is None and d.rows[0].layers_before == 0
