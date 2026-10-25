from __future__ import annotations

import pytest

from autotune.pool import DeviceView, NodeFacts, PoolSnapshot
from autotune.trigger import PoolWatcher, busy_reason


def node(nid: str, state: str = "eligible", devices: tuple[str, ...] = ("MTL0", "CPU")) -> NodeFacts:
    return NodeFacts(id=nid, name=nid, is_local=False, state=state,
                     devices=tuple(DeviceView(d, d, None, None) for d in devices), max_bytes=None, reserve_bytes=None)


HOST = NodeFacts(id="local", name="host", is_local=True, state="eligible", devices=(), max_bytes=None,
                 reserve_bytes=None)


def pool(*peers: NodeFacts) -> PoolSnapshot:
    return PoolSnapshot(nodes=(HOST, *peers))


def test_first_snapshot_is_a_baseline_and_never_fires() -> None:
    w = PoolWatcher(settle_s=20)
    assert w.observe(pool(node("a")), 0).action == "baseline"
    assert w.observe(pool(node("a")), 100).action == "idle"


def test_join_fires_once_after_the_settle_window() -> None:
    w = PoolWatcher(settle_s=20)
    w.observe(pool(), 0)
    tick = w.observe(pool(node("mac")), 1)
    assert tick.action == "changed" and tick.change.joined == ("mac",)
    assert w.observe(pool(node("mac")), 10).action == "settling"
    fire = w.observe(pool(node("mac")), 21)
    assert fire.action == "fire" and fire.change.joined == ("mac",)
    assert w.consume().joined == ("mac",)
    assert w.observe(pool(node("mac")), 60).action == "idle"


def test_every_further_change_restarts_the_window() -> None:
    w = PoolWatcher(settle_s=20)
    w.observe(pool(), 0)
    w.observe(pool(node("a")), 1)
    w.observe(pool(node("a"), node("b")), 15)  # second join restarts the window
    assert w.observe(pool(node("a"), node("b")), 30).action == "settling"
    fire = w.observe(pool(node("a"), node("b")), 36)
    assert fire.action == "fire" and fire.change.joined == ("a", "b")


def test_a_join_then_leave_inside_the_window_nets_out() -> None:
    w = PoolWatcher(settle_s=20)
    w.observe(pool(), 0)
    w.observe(pool(node("flaky")), 1)
    assert w.observe(pool(), 5).action == "netted_out"
    assert w.observe(pool(), 40).action == "idle"
    assert w.pending is None


def test_a_peer_that_stops_being_eligible_is_a_leave() -> None:
    w = PoolWatcher(settle_s=0)
    w.observe(pool(node("a")), 0)
    w.observe(pool(node("a", state="stale")), 1)
    fire = w.observe(pool(node("a", state="stale")), 2)
    assert fire.action == "fire" and fire.change.left == ("a",)


def test_a_verifying_peer_does_not_count_until_eligible() -> None:
    w = PoolWatcher(settle_s=0)
    w.observe(pool(), 0)
    assert w.observe(pool(node("m", state="verifying")), 1).action == "idle"
    assert w.observe(pool(node("m")), 2).action == "changed"


def test_a_settled_change_waits_while_the_gateway_is_busy() -> None:
    w = PoolWatcher(settle_s=5)
    w.observe(pool(), 0)
    w.observe(pool(node("a")), 1)
    tick = w.observe(pool(node("a")), 10, busy="deployment abc is loading")
    assert tick.action == "deferred" and "loading" in tick.detail
    assert w.observe(pool(node("a")), 30).action == "fire"


def test_a_device_list_change_counts_as_leave_and_join() -> None:
    w = PoolWatcher(settle_s=0)
    w.observe(pool(node("a", devices=("MTL0",))), 0)
    w.observe(pool(node("a", devices=("MTL0", "CPU"))), 1)
    fire = w.observe(pool(node("a", devices=("MTL0", "CPU"))), 2)
    assert fire.action == "fire" and fire.change.joined == ("a",) and fire.change.left == ()


def test_negative_settle_is_refused() -> None:
    with pytest.raises(ValueError):
        PoolWatcher(settle_s=-1)


@pytest.mark.parametrize(("deps", "tunes", "active", "expect"), [
    ([{"id": "d1", "state": "ready"}], [], False, None),
    ([{"id": "d1", "state": "loading"}], [], False, "loading"),
    ([{"id": "d1", "state": "warming"}], [], False, "warming"),
    ([], [{"id": "t1", "status": "running"}], False, "tune"),
    ([], [{"id": "t1", "status": "done"}], False, None),
    ([], [], True, "already in progress"),
])
def test_busy_reason(deps: list, tunes: list, active: bool, expect: str | None) -> None:
    reason = busy_reason(deps, tunes, active)
    assert (reason is None) if expect is None else (expect in reason)
