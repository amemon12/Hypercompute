"""A model preloaded through the SDK must be visible without the app taking ownership."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hcapp  # noqa: E402


class Pool:
    def __init__(self):
        self.deps = []
        self.plan_calls = []
        self.stop_calls = []
        self.plan_result = {"id": "plan", "model_id": "large", "feasible": True, "tier": "moe_offload",
                            "assignments": [{"node_id": "local", "device": "CUDA0", "layers": [0, 1]},
                                            {"node_id": "mac-a", "device": "MTL0", "layers": [2]}]}

    def pools(self):
        return {}

    def peers(self):
        return [{"node_id": "mac-a", "display_name": "mac-a"}]

    def models(self):
        return [{"artifact": {"id": "large"}, "status": "downloaded"}]

    def deployments(self):
        return self.deps

    def plan(self, model, **options):
        self.plan_calls.append((model, options))
        return self.plan_result

    def deploy(self, plan_id):
        self.deps = [{"id": "app-owned", "state": "loading", "plan": self.plan_result}]
        return self.deps[0]

    def wait_ready(self, dep_id, **kwargs):
        self.deps[0]["state"] = "ready"
        return self.deps[0]

    def stop(self, dep_id):
        self.stop_calls.append(dep_id)
        self.deps[0]["state"] = "stopped"


@pytest.fixture
def backend(monkeypatch):
    backend = hcapp.Backend()
    backend._pool = Pool()
    monkeypatch.setattr(backend, "host_up", lambda: True)
    monkeypatch.setattr(backend, "sharing", lambda: "not sharing")
    return backend


def test_preloaded_model_shows_state_and_layer_bar_without_taking_ownership(backend):
    pool = backend.pool()
    pool.deps = [{"id": "preloaded", "state": "ready", "plan": pool.plan_result}]
    job = backend.state()["job"]
    assert job["state"] == "ready" and job["model"] == "large"
    assert [(s["node"], s["layers"]) for s in job["stages"]] == [("local", 2), ("mac-a", 1)]
    assert job["can_stop"] is False
    backend.stop_model()
    assert pool.stop_calls == []
    assert backend.deployment is None


def test_run_does_not_replace_an_existing_deployment(backend):
    pool = backend.pool()
    pool.deps = [{"id": "preloaded", "state": "warming", "plan": pool.plan_result}]
    backend.run_model("other")
    assert pool.plan_calls == []
    assert backend.job["model"] == "large" and backend.job["state"] == "warming"


def test_external_stop_clears_the_preloaded_layer_bar(backend):
    pool = backend.pool()
    pool.deps = [{"id": "preloaded", "state": "ready", "plan": pool.plan_result}]
    backend.state()
    pool.deps[0]["state"] = "stopped"
    assert backend.state()["job"] == {"state": "idle"}


def test_loading_failure_survives_state_refresh_and_clears_stale_layers(backend):
    pool = backend.pool()
    pool.deps = [{"id": "app-owned", "state": "loading", "plan": pool.plan_result}]
    backend.deployment = "app-owned"
    backend.state()
    pool.deps[0].update(state="failed", error="a peer disconnected")
    expected = {"state": "failed", "model": "large", "detail": "a peer disconnected"}
    assert backend.state()["job"] == expected
    assert backend.state()["job"] == expected
    assert backend.deployment is None


def test_safe_capacity_plan_is_loaded_and_owned_model_can_stop(backend):
    backend.run_model("large")
    assert backend.pool().plan_calls == [("large", {"strategy": "all_selected", "allow_capacity_mode": True})]
    assert backend.job["state"] == "ready" and backend.job["can_stop"] is True
    backend.stop_model()
    assert backend.pool().stop_calls == ["app-owned"]
    assert backend.job == {"state": "idle"}


@pytest.mark.parametrize("tier", ["gpu_disk", "stream"])
def test_disk_tiers_still_cannot_deploy(backend, tier):
    backend.pool().plan_result["tier"] = tier
    backend.run_model("large")
    assert backend.job["state"] == "refused"
    assert backend.pool().deps == []


def test_two_run_requests_cannot_plan_at_the_same_time(backend, monkeypatch):
    started, release = threading.Event(), threading.Event()
    plan = backend.pool().plan

    def slow_plan(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return plan(*args, **kwargs)

    monkeypatch.setattr(backend.pool(), "plan", slow_plan)
    worker = threading.Thread(target=backend.run_model, args=("large",))
    worker.start()
    try:
        assert started.wait(5)
        backend.run_model("second")
    finally:
        release.set()
        worker.join(5)
    assert [call[0] for call in backend.pool().plan_calls] == ["large"]


def test_stop_during_loading_does_not_turn_into_a_failure(backend, monkeypatch):
    def canceled_wait(*args, **kwargs):
        backend.stop_model()
        raise RuntimeError("deployment stopped")

    monkeypatch.setattr(backend.pool(), "wait_ready", canceled_wait)
    backend.run_model("large")
    assert backend.job == {"state": "idle"}


def test_tuning_trials_are_not_presented_as_the_chat_model(backend):
    pool = backend.pool()
    pool.deps = [{"id": "trial", "state": "ready", "purpose": "tune", "plan": pool.plan_result}]
    assert backend.state()["job"] == {"state": "idle"}


def test_first_host_launch_creates_its_log_directory(backend, monkeypatch, tmp_path):
    up = iter([False, True])
    monkeypatch.setattr(backend, "host_up", lambda: next(up))
    monkeypatch.setattr(hcapp.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hcapp.subprocess, "Popen", lambda *args, **kwargs: object())
    assert backend.start_host() == "hosting"
    assert (tmp_path / ".hypercompute" / "app-host.log").is_file()
