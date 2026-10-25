"""Memory pools: one pool per physical pool per node, lease math without double subtraction, admission, OS snapshots."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime

import psutil
import pytest
from factories import contribution, make_inventory
from hypercompute.capacity import pools as P
from hypercompute.capacity.types import Admission, Pool, PoolDemand, PoolLease
from hypercompute.schemas import Device, NodeState, NodeView, WorkerEndpoint, WorkerState, utcnow
from hypercompute.units import GiB, MiB

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
PEER = "b" * 32


def _node(system: str, *, local: bool, devices: list[str], max_gib: float = 8, reserve_gib: float = 2,
          **inventory) -> NodeView:
    return NodeView(id="local" if local else PEER, display_name=system, is_local=local,
                    session_id=None if local else "s", state=NodeState.eligible, state_reason=None,
                    joined_at=utcnow(), last_heartbeat_at=utcnow(), heartbeat_age_s=0.0,
                    inventory=make_inventory(system, **inventory),
                    contribution=contribution(devices, max_gib=max_gib, reserve_gib=reserve_gib),
                    worker=None if local else WorkerEndpoint(host="10.0.0.5", port=47052),
                    worker_state=None if local else WorkerState.running, verification=None)


def _pools(*nodes: NodeView, storage: int | None = None, reserve: int = 4 * GiB) -> tuple[Pool, ...]:
    return P.pools_from_nodes(nodes, host_reserve_bytes=reserve, storage_free_bytes=storage, now=NOW)


def _pool(free: int, reserve: int = 0, cap: int | None = None) -> Pool:
    return Pool(node_id="n", pool_id="ram", kind="system_ram", total_bytes=32 * GiB, free_bytes=free,
                observed_at=NOW, source="test", reserve_bytes=reserve, cap_bytes=cap)


def _lease(nbytes: int, materialized: bool, node_id: str = "n", pool_id: str = "ram") -> PoolLease:
    return PoolLease(lease_id=f"l{nbytes}{materialized}", deployment_id="d", revision=1, node_id=node_id,
                     pool_id=pool_id, bytes=nbytes, materialized=materialized)


# --------------------------------------------------------------------------------------------- pools_from_nodes


def test_local_mac_uma_is_one_pool_shared_by_cpu_and_metal():
    pools = _pools(_node("macos", local=True, devices=["MTL0"], ram_total=24 * GiB, ram_free=14 * GiB))
    assert [(p.pool_id, p.kind) for p in pools] == [("uma", "unified")]
    [uma] = pools
    assert uma.devices == ("CPU", "MTL0")
    assert uma.free_bytes == 14 * GiB and uma.total_bytes == 24 * GiB
    assert uma.reserve_bytes == 4 * GiB and uma.cap_bytes is None
    # CPU and Metal stages both draw from uma: their sum is checked once against one budget (14 - 4 = 10 GiB)
    both = P.admit([PoolDemand(node_id="local", pool_id="uma", weights=6 * GiB),
                    PoolDemand(node_id="local", pool_id="uma", weights=6 * GiB)], pools, [])
    assert not both.ok and len(both.checks) == 1
    assert both.checks[0].demand_peak == 12 * GiB and both.checks[0].admissible == 10 * GiB


def test_peer_mac_contributing_cpu_and_metal_still_has_one_capped_pool():
    pools = _pools(_node("macos", local=False, devices=["CPU", "MTL0"], max_gib=10, reserve_gib=4,
                         ram_free=20 * GiB), storage=None)
    assert [(p.node_id, p.pool_id) for p in pools] == [(PEER, "uma")]
    assert pools[0].devices == ("CPU", "MTL0")
    assert pools[0].reserve_bytes == 4 * GiB and pools[0].cap_bytes == 10 * GiB
    assert P.admissible_bytes(pools[0], []) == 10 * GiB  # min(20 - 4, cap 10)


def test_windows_ram_and_vram_are_separate_pools_with_their_own_reserves():
    node = _node("windows", local=True, devices=["CUDA0"], ram_free=16 * GiB, vram_free=11 * GiB)
    igpu = Device(id="AMD Radeon 890M", kind="igpu", name="AMD Radeon 890M", pool_id="ram", backend="unknown",
                  source="Win32_VideoController")
    node = node.model_copy(update={"inventory": node.inventory.model_copy(
        update={"devices": [*node.inventory.devices, igpu]})})
    by_id = {p.pool_id: p for p in _pools(node)}
    assert set(by_id) == {"ram", "vram:CUDA0"}
    assert by_id["ram"].kind == "system_ram" and by_id["ram"].devices == ("CPU",)  # the iGPU is not an engine device
    assert by_id["ram"].reserve_bytes == 4 * GiB
    assert by_id["vram:CUDA0"].kind == "discrete_vram" and by_id["vram:CUDA0"].devices == ("CUDA0",)
    assert by_id["vram:CUDA0"].reserve_bytes == P.VRAM_HEADROOM == 512 * MiB
    assert by_id["vram:CUDA0"].free_bytes == 11 * GiB


def test_peer_pools_are_limited_to_contributed_devices_and_report_the_rest():
    node = _node("windows", local=False, devices=["CUDA0"], max_gib=6, reserve_gib=3)
    report = P.pools_report([node], host_reserve_bytes=4 * GiB, storage_free_bytes=None, now=NOW)
    assert [(p.pool_id, p.cap_bytes, p.reserve_bytes) for p in report.pools] == [("vram:CUDA0", 6 * GiB, 512 * MiB)]
    assert (PEER, "ram") in {(e.node_id, e.pool_id) for e in report.excluded}
    assert "not contributed" in next(e.reason for e in report.excluded if e.pool_id == "ram")


def test_peer_contributing_two_pools_cannot_be_admitted_past_its_node_total():
    # max_bytes is the total a contributor allows on the node; two pools must not each get all of it
    node = _node("windows", local=False, devices=["CPU", "CUDA0"], max_gib=8, reserve_gib=2,
                 ram_free=16 * GiB, vram_free=8 * GiB)
    report = P.pools_report([node], host_reserve_bytes=0, storage_free_bytes=None, now=NOW)
    by_id = {p.pool_id: p for p in report.pools}
    assert set(by_id) == {"ram", "vram:CUDA0"}
    assert by_id["ram"].cap_bytes == 8 * GiB * 16 // 24 and by_id["vram:CUDA0"].cap_bytes == 8 * GiB * 8 // 24
    assert sum(P.admissible_bytes(p, []) for p in report.pools) <= 8 * GiB
    assert any("node total" in n and "split" in n for n in report.notes)
    # 6 GiB per pool fits each pool's free memory but is 12 GiB on an 8 GiB contribution
    result = P.admit([PoolDemand(node_id=PEER, pool_id="ram", weights=6 * GiB),
                      PoolDemand(node_id=PEER, pool_id="vram:CUDA0", weights=6 * GiB)], report.pools, [])
    assert not result.ok


def test_peer_node_total_is_split_equally_when_no_pool_has_free_memory():
    node = _node("windows", local=False, devices=["CPU", "CUDA0"], max_gib=8, ram_free=0, vram_free=0)
    pools = _pools(node)
    assert [p.cap_bytes for p in pools] == [4 * GiB, 4 * GiB]
    assert all(P.admissible_bytes(p, []) == 0 for p in pools)  # nothing free: the free side still rules


def test_pool_without_measured_at_says_its_time_is_the_inventory_collection_time():
    node = _node("windows", local=True, devices=["CUDA0"])
    stale = [mp.model_copy(update={"measured_at": None}) for mp in node.inventory.memory_pools]
    node = node.model_copy(update={"inventory": node.inventory.model_copy(update={"memory_pools": stale})})
    for pool in _pools(node):
        assert pool.observed_at == node.inventory.collected_at
        assert "no measured_at" in pool.source and "collected_at" in pool.source


def test_a_node_passed_twice_yields_each_pool_once_and_reports_the_duplicate():
    node = _node("macos", local=False, devices=["MTL0"])
    report = P.pools_report([node, node], host_reserve_bytes=0, storage_free_bytes=None, now=NOW)
    assert [(p.node_id, p.pool_id) for p in report.pools] == [(PEER, "uma")]
    assert any(e.pool_id == "uma" and "duplicate" in e.reason for e in report.excluded)


def test_contributed_device_missing_from_inventory_is_noted_not_guessed():
    report = P.pools_report([_node("windows", local=False, devices=["CUDA0", "CUDA1"])],
                            host_reserve_bytes=0, storage_free_bytes=None, now=NOW)
    assert [p.pool_id for p in report.pools] == ["vram:CUDA0"]
    assert any("CUDA1" in n for n in report.notes)


def test_pool_with_unknown_free_memory_is_excluded_with_a_reason():
    node = _node("windows", local=True, devices=["CUDA0"], vram_free=None)
    report = P.pools_report([node], host_reserve_bytes=0, storage_free_bytes=None, now=NOW)
    assert [p.pool_id for p in report.pools] == ["ram"]
    [why] = [e for e in report.excluded if e.pool_id == "vram:CUDA0"]
    assert "unknown" in why.reason
    assert P.pools_from_nodes([node], host_reserve_bytes=0, storage_free_bytes=None, now=NOW) == report.pools


def test_storage_pool_comes_from_the_given_free_space_on_the_local_node():
    local = _node("windows", local=True, devices=["CUDA0"])
    pools = _pools(local, _node("macos", local=False, devices=["MTL0"]), storage=50 * GiB)
    [ssd] = [p for p in pools if p.kind == "storage"]
    assert (ssd.node_id, ssd.pool_id, ssd.free_bytes, ssd.observed_at) == ("local", "ssd", 50 * GiB, NOW)
    assert ssd.total_bytes is None and ssd.cap_bytes is None
    report = P.pools_report([local], host_reserve_bytes=0, storage_free_bytes=None, now=NOW)
    assert not any(p.pool_id == "ssd" for p in report.pools)
    assert any(e.pool_id == "ssd" for e in report.excluded)
    peers_only = _pools(_node("macos", local=False, devices=["MTL0"]), storage=5 * GiB)
    assert [(p.node_id, p.pool_id) for p in peers_only if p.kind == "storage"] == [(P.HOST_NODE_ID, "ssd")]


def test_pools_are_unique_per_node_and_pool_id():
    pools = _pools(_node("windows", local=True, devices=["CUDA0"]), _node("macos", local=False, devices=["MTL0"]),
                   storage=1 * GiB)
    keys = [(p.node_id, p.pool_id) for p in pools]
    assert len(keys) == len(set(keys)) == 4


def test_inputs_are_validated():
    node = _node("windows", local=True, devices=["CUDA0"])
    with pytest.raises(ValueError, match="host_reserve_bytes"):
        P.pools_from_nodes([node], host_reserve_bytes=-1, storage_free_bytes=None, now=NOW)
    with pytest.raises(ValueError, match="timezone"):
        P.pools_from_nodes([node], host_reserve_bytes=0, storage_free_bytes=None, now=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="storage_free_bytes"):
        P.pools_from_nodes([node], host_reserve_bytes=0, storage_free_bytes=-5, now=NOW)


# --------------------------------------------------------------------------------------------- lease math


def test_unmaterialized_leases_come_out_of_free_memory():
    assert P.admissible_bytes(_pool(10 * GiB, reserve=1 * GiB), [_lease(2 * GiB, False)]) == 7 * GiB


def test_materialized_leases_are_already_inside_free_and_not_subtracted_again():
    assert P.admissible_bytes(_pool(10 * GiB, reserve=1 * GiB), [_lease(2 * GiB, True)]) == 9 * GiB


def test_cap_counts_every_lease_and_the_result_floors_at_zero():
    pool = _pool(20 * GiB, reserve=1 * GiB, cap=8 * GiB)
    both = [_lease(3 * GiB, True), _lease(2 * GiB, False)]
    assert P.admissible_bytes(pool, both) == 3 * GiB  # min(20 - 1 - 2, 8 - 5)
    assert P.admissible_bytes(pool, [_lease(10 * GiB, True)]) == 0
    assert P.admissible_bytes(_pool(1 * GiB, reserve=2 * GiB), []) == 0


def test_leases_on_other_pools_and_nodes_do_not_count():
    other = [_lease(5 * GiB, False, pool_id="vram:CUDA0"), _lease(5 * GiB, False, node_id="m")]
    assert P.admissible_bytes(_pool(10 * GiB), other) == 10 * GiB


def test_negative_lease_bytes_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        P.admissible_bytes(_pool(10 * GiB), [_lease(-1, False)])


def test_a_negative_reserve_or_cap_is_rejected_instead_of_raising_admissible_above_free():
    with pytest.raises(ValueError, match="negative reserve or cap"):
        P.admissible_bytes(_pool(10 * GiB, reserve=-2 * GiB), [])
    with pytest.raises(ValueError, match="negative reserve or cap"):
        P.admit([PoolDemand(node_id="n", pool_id="ram", weights=1)], [_pool(10 * GiB, cap=-1)], [])


# --------------------------------------------------------------------------------------------- admit


def test_admit_checks_two_pools_on_one_node_separately():
    pools = _pools(_node("windows", local=True, devices=["CUDA0"], ram_free=16 * GiB, vram_free=11 * GiB))
    demands = [PoolDemand(node_id="local", pool_id="vram:CUDA0", weights=6 * GiB, kv=1 * GiB),
               PoolDemand(node_id="local", pool_id="ram", weights=10 * GiB, staging=2 * GiB),
               PoolDemand(node_id="local", pool_id="vram:CUDA0", workspace=2 * GiB, peak_extra=1 * GiB)]
    result = P.admit(demands, pools, [_lease(1 * GiB, False, node_id="local", pool_id="vram:CUDA0")])
    by_pool = {c.pool_id: c for c in result.checks}
    assert [c.pool_id for c in result.checks] == ["vram:CUDA0", "ram"]
    assert by_pool["vram:CUDA0"].demand_peak == 10 * GiB  # peak, including peak_extra, summed over both demands
    assert by_pool["vram:CUDA0"].admissible == 11 * GiB - 512 * MiB - 1 * GiB
    assert not by_pool["vram:CUDA0"].ok and "short by" in by_pool["vram:CUDA0"].reason
    assert by_pool["ram"].ok and by_pool["ram"].admissible == 12 * GiB  # 16 free - 4 reserve
    assert not result.ok and [c.pool_id for c in result.failures] == ["vram:CUDA0"]


def test_admit_passes_when_every_pool_fits():
    pools = _pools(_node("windows", local=True, devices=["CUDA0"]))
    result = P.admit([PoolDemand(node_id="local", pool_id="ram", weights=1 * GiB)], pools, [])
    assert result.ok and result.failures == () and "fits" in result.checks[0].reason


def test_admit_fails_a_demand_for_an_unknown_pool_with_a_reason():
    pools = _pools(_node("windows", local=True, devices=["CUDA0"]))
    result = P.admit([PoolDemand(node_id="local", pool_id="vram:CUDA1", weights=1)], pools, [])
    [check] = result.checks
    assert not result.ok and check.admissible == 0 and "no such pool" in check.reason


def test_admit_with_no_demands_passes():
    pools = _pools(_node("windows", local=True, devices=["CUDA0"]))
    assert P.admit([], pools, []) == Admission(ok=True, checks=())


def test_admit_rejects_a_pool_listed_twice_and_negative_demands():
    pool = _pool(10 * GiB)
    with pytest.raises(ValueError, match="twice"):
        P.admit([PoolDemand(node_id="n", pool_id="ram", weights=1)], [pool, pool], [])
    with pytest.raises(ValueError, match="negative kv"):
        P.admit([PoolDemand(node_id="n", pool_id="ram", kv=-1)], [pool], [])


def test_storage_demand_is_admitted_against_the_ssd_pool():
    pools = _pools(_node("windows", local=True, devices=["CUDA0"]), storage=3 * GiB)
    ok = P.admit([PoolDemand(node_id="local", pool_id="ssd", weights=2 * GiB)], pools, [])
    too_big = P.admit([PoolDemand(node_id="local", pool_id="ssd", weights=4 * GiB)], pools, [])
    assert ok.ok and not too_big.ok


# --------------------------------------------------------------------------------------------- OS snapshots


def test_host_memory_snapshot_returns_sane_numbers_on_this_machine():
    mem = P.host_memory_snapshot()
    assert mem.error is None
    assert mem.total is not None and mem.available is not None
    assert 0 < mem.available <= mem.total
    assert mem.observed_at.tzinfo is not None
    if sys.platform == "win32":
        assert "GetPerformanceInfo" in mem.source
        assert mem.commit_charge is not None and mem.commit_limit is not None and mem.system_cache is not None
        assert 0 < mem.commit_charge <= mem.commit_limit
        assert abs(mem.total - psutil.virtual_memory().total) < 64 * MiB
    else:
        assert mem.commit_charge is None and mem.commit_limit is None and mem.system_cache is None


def test_host_memory_snapshot_never_raises_when_the_os_calls_fail(monkeypatch):
    def boom(*_args: object) -> object:
        raise OSError("simulated failure")

    monkeypatch.setattr(P, "_performance_info", boom)
    monkeypatch.setattr(P.psutil, "virtual_memory", boom)
    mem = P.host_memory_snapshot()
    assert (mem.total, mem.available, mem.commit_charge, mem.commit_limit, mem.system_cache) == (None,) * 5
    assert mem.error is not None and "simulated failure" in mem.error


@pytest.mark.skipif(sys.platform != "win32", reason="GetPerformanceInfo exists only on Windows")
def test_host_memory_snapshot_falls_back_to_psutil_when_windows_call_fails(monkeypatch):
    def boom() -> object:
        raise OSError("simulated failure")

    monkeypatch.setattr(P, "_performance_info", boom)
    mem = P.host_memory_snapshot()
    assert mem.source.startswith("psutil") and mem.total is not None and mem.commit_limit is None
    assert mem.error is not None and "GetPerformanceInfo" in mem.error


def test_process_memory_of_this_process():
    mem = P.process_memory(os.getpid())
    assert mem is not None and mem.pid == os.getpid() and mem.working_set > 0
    if sys.platform == "win32":
        assert mem.private is not None and mem.private > 0
        assert mem.peak_working_set is not None and mem.peak_working_set >= mem.working_set
        assert mem.page_faults is not None and mem.page_faults > 0


def test_process_memory_of_a_real_exited_child_is_none():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=60)
    # the Popen object still holds the process handle on Windows, so the pid cannot be reused yet
    assert P.process_memory(child.pid) is None


def test_process_memory_of_a_gone_process_is_none(monkeypatch):
    def gone(pid: int) -> object:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(P.psutil, "Process", gone)
    assert P.process_memory(123456) is None


def test_process_memory_access_denied_is_an_explicit_error(monkeypatch):
    def denied(pid: int) -> object:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(P.psutil, "Process", denied)
    with pytest.raises(P.ProcessMemoryError, match="access denied"):
        P.process_memory(4)
    with pytest.raises(ValueError):
        P.process_memory(-1)
