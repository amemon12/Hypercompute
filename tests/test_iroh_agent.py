"""Local worker shutdown must remain reliable when the peer transport has already failed."""

from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest
from factories import contribution, make_inventory
from hypercompute.agent import join
from hypercompute.agent.join import Agent, AgentError
from hypercompute.agent.supervisor import StopOutcome


@pytest.fixture
def agent(monkeypatch):
    peer = Mock()
    result = Agent(host="127.0.0.1", port=0, code="ABCDE-FGHJK", display_name="Iroh contributor",
                   inv=make_inventory(), contribution=contribution(["CUDA0"]),
                   advertise_ip="127.0.0.1", worker_port=47052, peer_client=peer)
    monkeypatch.setattr(result.sup, "stop", Mock(return_value=StopOutcome(join.WORKER_KEY, 1234, 0, "graceful")))
    monkeypatch.setattr(result, "reconcile_orphans", Mock())
    monkeypatch.setattr(result, "start_exo", Mock())
    monkeypatch.setattr(result, "start_worker", Mock())
    monkeypatch.setattr(result, "_say_goodbye", Mock())
    return result


@pytest.mark.parametrize("error", [httpx.ConnectError("peer closed"), httpx.ConnectTimeout("peer stalled")])
def test_pause_stops_local_worker_even_if_peer_revocation_fails(agent, error):
    agent.peer_client.set_worker_active.side_effect = error
    assert agent.pause() == "paused"
    assert agent.paused
    agent.sup.stop.assert_called_once_with(join.WORKER_KEY)
    agent.peer_client.set_worker_active.assert_called_once_with(False)
    assert agent.pause() == "already paused"


def test_leave_stops_worker_and_clears_state_despite_peer_revocation_error(agent):
    join.write_state({"node_id": agent.node, "credential": "local-test-credential", "transport": "iroh"})
    agent.peer_client.set_worker_active.side_effect = httpx.ConnectError("peer closed")
    agent.leave()
    assert agent.leaving
    agent.sup.stop.assert_called_once_with(join.WORKER_KEY)
    agent._say_goodbye.assert_called_once()
    assert join.read_state() is None


def test_leave_preserves_recovery_state_if_worker_will_not_stop_even_when_peer_is_dead(agent):
    join.write_state({"node_id": agent.node, "credential": "local-test-credential", "transport": "iroh"})
    agent.peer_client.set_worker_active.side_effect = httpx.ConnectError("peer closed")
    agent.sup.stop.return_value = StopOutcome(join.WORKER_KEY, 1234, None, "still_running")
    agent.leave()
    agent.sup.stop.assert_called_once_with(join.WORKER_KEY)
    agent._say_goodbye.assert_not_called()
    assert join.read_state() is not None


def test_resume_rolls_worker_back_and_stays_paused_if_peer_activation_fails(agent):
    agent.paused = True
    agent.peer_client.set_worker_active.side_effect = httpx.ConnectError("peer closed")
    with pytest.raises((httpx.HTTPError, AgentError)):
        agent.resume()
    agent.start_worker.assert_called_once()
    agent.sup.stop.assert_called_once_with(join.WORKER_KEY)
    assert agent.paused
    assert agent.pause() == "already paused"
    # A later successful Resume can recover after the failed attempt.
    agent.peer_client.set_worker_active.side_effect = None
    assert agent.resume() == "resumed"
    assert not agent.paused
    assert agent.start_worker.call_count == 2
