from __future__ import annotations

import time

from autotune.pool import diff, snapshot
from mock_gateway.app import PEERS, Joined, MockState


def mock_json(*peers: str) -> tuple[list, dict]:
    st = MockState(host_kind="gx10", verify_s=0)
    for p in peers:
        st.peers[p] = Joined(PEERS[p], time.monotonic())
    return st.nodes(), st.network()


def test_snapshot_reads_the_gateway_shapes() -> None:
    snap = snapshot(*mock_json("mac-a", "g14"))
    assert snap.transport == "iroh"
    assert snap.local is not None and snap.local.id == "local" and snap.local.devices[0].name == "CUDA0"
    mac = snap.node(PEERS["mac-a"].node_id)
    assert mac is not None and [d.name for d in mac.devices] == ["MTL0", "CPU"]
    assert mac.path == "direct" and mac.rtt_ms == 6.0
    assert mac.devices[0].free_bytes is not None
    assert len(snap.peers) == 2


def test_malformed_nodes_are_skipped() -> None:
    nodes, net = mock_json("mac-a")
    snap = snapshot([*nodes, "junk", {"no_id": True}, {"id": 5}], {"peers": ["x", {"node_id": 3}]})
    assert len(snap.nodes) == 2


def test_diff_reports_joins_and_leaves_by_name() -> None:
    before = snapshot(*mock_json("mac-a"))
    after = snapshot(*mock_json("mac-b"))
    change = diff(before, after)
    assert change.joined == (PEERS["mac-b"].node_id,) and change.left == (PEERS["mac-a"].node_id,)
    assert change.describe() == "mac-b joined, mac-a left"


def test_lan_gateway_network_is_fine() -> None:
    nodes, _ = mock_json("mac-a")
    snap = snapshot(nodes, {"transport": "lan"})
    assert snap.transport == "lan" and snap.peers[0].path is None
