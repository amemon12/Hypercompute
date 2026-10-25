"""Seed examples stop on host failures and never mistake an older peer job for this one."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "seed_and_run.py"
EMPTY = "layers 0-0 of small-model have no tensor over 10 MiB; the host sends smaller ones at load anyway"


def load(monkeypatch):
    monkeypatch.setattr("sys.argv", [str(EXAMPLE), "small-model"])
    spec = importlib.util.spec_from_file_location("hc_sdk_seed_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(host="delivered", peer="done", detail=None, peer_job="current"):
    return {"node_id": "peer-1", "seed_request": {
        "job_id": "current", "state": host, "detail": detail, "tensors": 0, "parts": 0},
        "report": {"seed": {"job_id": peer_job, "state": peer}}}


@pytest.mark.parametrize("state", ["failed", "cancelled", "refused"])
def test_host_terminal_state_wins_over_old_peer_success(monkeypatch, state):
    module = load(monkeypatch)
    assert module.seed_outcome("current", row(host=state, peer_job="previous")).startswith("failed")


def test_empty_small_model_seed_is_skipped_only_for_the_exact_host_response(monkeypatch):
    module = load(monkeypatch)
    empty = row(host="failed", detail=EMPTY, peer_job="previous")
    assert module.seed_outcome("current", empty).startswith("skipped")
    empty["seed_request"]["detail"] = "could not read model weights"
    assert module.seed_outcome("current", empty).startswith("failed")
    empty["seed_request"].update(detail=EMPTY, tensors=1)
    assert module.seed_outcome("current", empty).startswith("failed")


def test_peer_outcome_must_match_the_requested_job(monkeypatch):
    module = load(monkeypatch)
    assert module.seed_outcome("current", row(peer_job="previous")) == "pending"
    assert module.seed_outcome("current", row()) == "done"
    assert module.seed_outcome("current", row(peer="cancelled")).startswith("failed")
    assert module.seed_outcome("replaced", row()).startswith("failed")
    assert module.seed_outcome("current", {}).startswith("failed")


def run_example(monkeypatch, peer_row, *, keep=False, chat_fails=False, chat_content="An answer",
                chat_reasoning="", stream_content="1, 2, 3", stream_reasoning="", empty_stream=False):
    module = load(monkeypatch)
    events, logs = [], []
    now = [0.0]
    monkeypatch.setattr(module, "log", logs.append)
    monkeypatch.setattr(module, "SEED_TIMEOUT", 20)
    monkeypatch.setattr(module, "KEEP_RUNNING", keep)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))

    class Pool:
        requests = {}

        def plan(self, *args, **kwargs):
            return {"id": "plan", "feasible": True, "tier": "gpu", "assignments": [
                {"node_id": "peer-1", "device": "CPU", "layers": [0]}]}

        def deploy(self, plan_id):
            events.append("deploy")
            return {"id": "deployment"}

        def wait_ready(self, *args, **kwargs):
            events.append("ready")

        def chat(self, *args, **kwargs):
            events.append("chat")
            self.requests["chat"] = kwargs
            if chat_fails:
                raise RuntimeError("chat failed")
            return {"choices": [{"message": {"content": chat_content, "reasoning_content": chat_reasoning}}]}

        def stream(self, *args, **kwargs):
            events.append("stream")
            self.requests["stream"] = kwargs
            return [] if empty_stream else [SimpleNamespace(
                content=stream_content, reasoning=stream_reasoning, timings=None)]

        def deployment(self, *args):
            return {}

        def stop(self, *args):
            events.append("stop")
            return {"state": "stopped"}

    def get(url, **kwargs):
        return httpx.Response(200, json={"token": "test-only"} if url.endswith("bootstrap") else [peer_row])

    monkeypatch.setattr(module, "Pool", Pool)
    monkeypatch.setattr(module.httpx, "get", get)
    monkeypatch.setattr(module.httpx, "post", lambda *a, **kw: httpx.Response(202, json={"job_id": "current"}))
    return module, events, logs, now


@pytest.mark.parametrize("state", ["failed", "refused", "cancelled"])
def test_example_does_not_wait_or_deploy_after_host_failure(monkeypatch, state):
    module, events, logs, now = run_example(monkeypatch, row(host=state, peer_job="previous"))
    assert module.main() == 1
    assert events == [] and now[0] == 0
    assert any("no deployment started" in line for line in logs)


def test_example_times_out_pending_seed_without_deploying(monkeypatch):
    module, events, logs, now = run_example(monkeypatch, row(peer_job="previous"))
    assert module.main() == 1
    assert events == [] and now[0] == 20
    assert any("seed timed out" in line for line in logs)


def test_seed_timeout_also_bounds_busy_peer_retries(monkeypatch):
    module, events, logs, now = run_example(monkeypatch, row())
    monkeypatch.setattr(module.httpx, "post", lambda *a, **kw: httpx.Response(409, json={}))
    assert module.main() == 1
    assert events == [] and now[0] == 20


def test_one_failed_peer_stops_waiting_for_other_pending_peers(monkeypatch):
    module, events, logs, now = run_example(monkeypatch, row(host="failed"))
    pool = module.Pool()
    original_plan = pool.plan

    def plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        result["assignments"].append({"node_id": "peer-2", "device": "CPU", "layers": [1]})
        return result

    pool.plan = plan
    monkeypatch.setattr(module, "Pool", lambda: pool)
    pending = row(peer_job="previous")
    pending["node_id"] = "peer-2"
    monkeypatch.setattr(module.httpx, "get", lambda url, **kw: httpx.Response(
        200, json={"token": "test-only"} if url.endswith("bootstrap") else [row(host="failed"), pending]))
    assert module.main() == 1
    assert events == [] and now[0] == 0
    assert any("another peer's seed failed" in line for line in logs)


@pytest.mark.parametrize("keep", [False, True])
def test_small_model_seed_can_deploy_and_keep_running_only_when_requested(monkeypatch, keep):
    module, events, logs, now = run_example(monkeypatch, row(host="failed", detail=EMPTY), keep=keep)
    assert module.main() == 0
    assert events == ["deploy", "ready", "chat", "stream"] + ([] if keep else ["stop"])
    assert now[0] == 0


def test_keep_running_still_stops_after_a_chat_failure(monkeypatch):
    module, events, logs, now = run_example(monkeypatch, row(), keep=True, chat_fails=True)
    with pytest.raises(RuntimeError, match="chat failed"):
        module.main()
    assert events == ["deploy", "ready", "chat", "stop"]


def test_max_tokens_sets_both_chat_and_stream_budgets(monkeypatch):
    module, events, logs, now = run_example(monkeypatch, row(), keep=True)
    monkeypatch.setattr(module, "MAX_TOKENS", 400)
    assert module.main() == 0
    assert module.Pool.requests["chat"]["max_tokens"] == 400
    assert module.Pool.requests["stream"]["max_tokens"] == 400
    assert "stop" not in events


@pytest.mark.parametrize("empty_stream", [False, True])
def test_keep_running_stops_without_a_visible_stream_answer(monkeypatch, empty_stream):
    module, events, logs, now = run_example(monkeypatch, row(), keep=True, stream_content="  ",
                                           stream_reasoning="Working it out", empty_stream=empty_stream)
    with pytest.raises(RuntimeError, match="stream returned no visible answer.*increase --max-tokens"):
        module.main()
    assert events == ["deploy", "ready", "chat", "stream", "stop"]
    assert not any(line.startswith("kept ") for line in logs)


def test_keep_running_stops_after_reasoning_only_chat(monkeypatch):
    module, events, logs, now = run_example(monkeypatch, row(), keep=True, chat_content="",
                                           chat_reasoning="Working it out")
    with pytest.raises(RuntimeError, match="chat returned no visible answer.*increase --max-tokens"):
        module.main()
    assert events == ["deploy", "ready", "chat", "stop"]
    assert not any(line.startswith("kept ") for line in logs)
