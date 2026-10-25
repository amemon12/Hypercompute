from dataclasses import replace

import pytest

from hc_agent.intent import Intent
from hc_agent.workflow import Workflow, WorkflowError
from mock_gateway.server import MockState

RUN = Intent(action="run", prompt="what is RPC?")


async def _noop(_text: str) -> None:
    return None


class Said:
    def __init__(self) -> None:
        self.lines: list[str] = []

    async def __call__(self, text: str) -> None:
        self.lines.append(text)


async def test_biggest_model_that_fits_is_planned_across_friends(workflow: Workflow, mock_state: MockState) -> None:
    p = await workflow.propose(RUN)
    assert p.model.id == "kimi-k2.6-iq1_s" and not p.reuse
    assert p.view.peer_count == 4 and p.view.capacity_mode
    assert set(p.view.machines) == {"gx10-49d1", "mac-a", "mac-b", "mac-azim", "g14"}
    assert not mock_state.deployments  # proposing only plans; nothing loads


async def test_execute_deploys_chats_and_reports(workflow: Workflow, mock_state: MockState) -> None:
    said = Said()
    p = await workflow.propose(RUN)
    r = await workflow.execute(p, "what is RPC?", 64, said)
    assert r.answer.startswith("[mock] Kimi K2.6") and r.decode_tok_s == pytest.approx(1.54, rel=1e-6)
    assert r.prefill_tok_s == pytest.approx(3.30) and r.load_s is not None and not r.reused
    assert r.participation_verified
    assert all(s.observed_bytes for s in r.view.stages if s.layer_count)  # engine-log bytes per machine
    assert any("Deploying Kimi K2.6" in s for s in said.lines)
    assert r.deployment_id in workflow.owned


async def test_second_request_reuses_the_loaded_model(workflow: Workflow, mock_state: MockState) -> None:
    await workflow.execute(await workflow.propose(RUN), "q1", 32, _noop)
    p = await workflow.propose(RUN)
    assert p.reuse and p.model.id == "kimi-k2.6-iq1_s"
    r = await workflow.execute(p, "q2", 32, _noop)
    assert r.reused and r.load_s is None and len(mock_state.deployments) == 1


async def test_named_and_local_requests(workflow: Workflow) -> None:
    p = await workflow.propose(Intent(action="run", prompt="x", model_hint="gpt-oss"))
    assert p.model.id == "gpt-oss-120b-mxfp4" and p.view.tier == "gpu"
    p = await workflow.propose(Intent(action="run", prompt="x", scope="local"))
    assert p.strategy == "local_only" and p.view.peer_count == 0
    assert p.model.id == "kimi-k2.6-iq1_s"  # capacity mode lets the host alone hold it, slowly
    p = await workflow.propose(Intent(action="run", prompt="x", preference="fastest"))
    assert p.model.id == "gpt-oss-120b-mxfp4" and not p.view.capacity_mode
    with pytest.raises(WorkflowError) as exc:
        await workflow.propose(Intent(action="run", prompt="x", model_hint="llama"))
    assert exc.value.code == "no_match" and "Kimi K2.6" in exc.value.message


async def test_busy_pool_and_foreign_models(workflow: Workflow, mock_state: MockState, gateway) -> None:
    plan = mock_state.plan("gpt-oss-120b-mxfp4", "all_selected", None, 4096)
    mock_state.create(plan, "owner", foreign=True)  # still loading
    mock_state.load_s = 1000
    with pytest.raises(WorkflowError) as exc:
        await workflow.propose(RUN)
    assert exc.value.code == "busy" and ("loading" in exc.value.message or "preparing" in exc.value.message)
    mock_state.load_s = 0.0  # now it is ready: someone else's model is used as is, never unloaded
    p = await workflow.propose(RUN)
    assert p.reuse and p.model.id == "gpt-oss-120b-mxfp4" and p.alternatives
    with pytest.raises(WorkflowError) as exc:
        await workflow.propose(Intent(action="run", prompt="x", model_hint="kimi"))
    assert exc.value.code == "busy" and "never unload" in exc.value.message
    assert await workflow.stop_owned() == ([], [])  # not ours
    assert mock_state.active() is not None


async def test_peer_that_leaves_before_deploy_is_planned_out(workflow: Workflow, mock_state: MockState) -> None:
    said = Said()
    p = await workflow.propose(RUN)
    mock_state.offline = {"mac-b"}
    r = await workflow.execute(p, "q", 32, said)
    assert "mac-b" not in r.view.machines
    assert any("mac-b went offline" in s for s in said.lines)
    assert any("Replanned without mac-b" in n for n in r.notes)


async def test_failed_load_recovers_once(workflow: Workflow, mock_state: MockState) -> None:
    said = Said()
    p = await workflow.propose(RUN)
    mock_state.drop_on_load = "mac-azim"  # the peer drops mid-load and the load fails
    r = await workflow.execute(p, "q", 32, said)
    assert "mac-azim" not in r.view.machines
    assert any("Recovering" in s and "dropped" in s for s in said.lines)
    states = [mock_state.deployment(d)["state"] for d in mock_state.deployments]
    assert states.count("failed") == 1 and states.count("ready") == 1


async def test_two_failed_loads_give_up_cleanly(workflow: Workflow, mock_state: MockState) -> None:
    mock_state.fail_loads = 2
    with pytest.raises(WorkflowError) as exc:
        await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    assert exc.value.code == "load_failed" and "twice" in exc.value.message
    assert not list(workflow.owned.idle(0))


async def test_load_timeout_stops_the_deployment(workflow: Workflow, mock_state: MockState) -> None:
    mock_state.load_s = 1000
    workflow.settings = replace(workflow.settings, load_timeout_s=0)
    with pytest.raises(WorkflowError) as exc:
        await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    assert exc.value.code == "load_timeout"
    assert mock_state.active() is None


async def test_nothing_fits_explains_why(workflow: Workflow, mock_state: MockState) -> None:
    mock_state.offline = {"mac-a", "mac-b", "mac-azim", "g14"}
    p = await workflow.propose(Intent(action="run", prompt="x", model_hint="gpt-oss"))
    assert any("No friends' machines are online" in n for n in p.notes)


async def test_gateway_down_is_a_clear_error(settings) -> None:
    import httpx

    from hc_agent.gateway import Gateway

    def refuse(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    gw = Gateway("http://127.0.0.1:1", transport=httpx.MockTransport(refuse))
    with pytest.raises(WorkflowError) as exc:
        await Workflow(gw, settings).propose(RUN)
    assert exc.value.code == "gateway_down" and "hypercompute host" in (exc.value.fix or "")


async def test_owner_token_refreshes_after_a_host_restart(workflow: Workflow, mock_state: MockState) -> None:
    await workflow.propose(RUN)
    mock_state.token = "rotated"
    p = await workflow.propose(RUN)  # the 403 triggers one bootstrap and a retry
    assert p.model.id == "kimi-k2.6-iq1_s"


async def test_reap_unloads_only_idle_models_we_loaded(workflow: Workflow, mock_state: MockState) -> None:
    r = await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    workflow.settings = replace(workflow.settings, keep_warm_s=0)
    assert await workflow.reap_idle() == []  # keep_warm 0 means never reap
    workflow.settings = replace(workflow.settings, keep_warm_s=1)
    workflow.owned._last_used[r.deployment_id] -= 5
    assert await workflow.reap_idle() == ["kimi-k2.6-iq1_s"]
    assert mock_state.active() is None


async def test_reasoning_only_answer_is_kept(workflow: Workflow, mock_state: MockState) -> None:
    mock_state.reasoning_only = True
    r = await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    assert r.answer == "" and r.reasoning and r.finish_reason == "length"


async def test_failed_stop_keeps_ownership(workflow: Workflow, mock_state: MockState) -> None:
    r = await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    mock_state.stop_fails = True
    assert await workflow.stop_owned() == ([], ["kimi-k2.6-iq1_s"])
    assert r.deployment_id in workflow.owned  # still ours: the reaper keeps trying
    mock_state.stop_fails = False
    assert await workflow.stop_owned() == (["kimi-k2.6-iq1_s"], [])
    assert r.deployment_id not in workflow.owned


async def test_reaper_never_unloads_during_a_run(workflow: Workflow, mock_state: MockState) -> None:
    r = await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    workflow.settings = replace(workflow.settings, keep_warm_s=1)
    workflow.owned._last_used[r.deployment_id] -= 5
    async with workflow.run_lock:
        assert await workflow.reap_idle() == []
    assert mock_state.active() is not None


async def test_stale_reuse_reloads_on_a_fresh_plan(workflow: Workflow, mock_state: MockState) -> None:
    plan = mock_state.plan("gpt-oss-120b-mxfp4", "all_selected", None, 4096)
    foreign = mock_state.create(plan, "owner", foreign=True, ready=True)
    p = await workflow.propose(RUN)
    assert p.reuse
    mock_state.stop(foreign["id"])  # the owner unloads it while the user decides
    said = Said()
    r = await workflow.execute(p, "q", 32, said)
    assert not r.reused and r.load_s is not None and r.answer.startswith("[mock]")
    assert any("unloaded since I checked" in s for s in said.lines)


async def test_my_idle_model_is_unloaded_before_planning_a_bigger_one(workflow: Workflow,
                                                                      mock_state: MockState) -> None:
    small = Intent(action="run", prompt="x", model_hint="gpt-oss")
    await workflow.execute(await workflow.propose(small), "q", 32, _noop)
    p = await workflow.propose(RUN)
    assert p.model.id == "kimi-k2.6-iq1_s" and not p.reuse
    assert any("unloaded gpt-oss-120b-mxfp4" in n for n in p.notes)
    assert mock_state.active() is None


async def test_someone_else_deploying_first_is_busy(workflow: Workflow, mock_state: MockState) -> None:
    p = await workflow.propose(RUN)
    plan = mock_state.plan("gpt-oss-120b-mxfp4", "all_selected", None, 4096)
    mock_state.load_s = 1000
    mock_state.create(plan, "owner", foreign=True)
    with pytest.raises(WorkflowError) as exc:
        await workflow.execute(p, "q", 32, _noop)
    assert exc.value.code == "busy"


async def test_chat_answer_survives_a_failed_pool_read(workflow: Workflow, mock_state: MockState,
                                                       monkeypatch) -> None:
    p = await workflow.propose(RUN)
    real = workflow.pool
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] >= 3:  # deploy reads the pool; the read after the chat fails
            raise WorkflowError("gateway_down", "down")
        return await real()

    monkeypatch.setattr(workflow, "pool", flaky)
    r = await workflow.execute(p, "q", 32, _noop)
    assert r.answer.startswith("[mock]")


async def test_an_unpaid_plan_never_unloads_my_idle_model(workflow: Workflow, mock_state: MockState) -> None:
    small = Intent(action="run", prompt="x", model_hint="gpt-oss")
    await workflow.execute(await workflow.propose(small), "q", 32, _noop)
    p = await workflow.propose(RUN, may_unload=False)
    assert p.model.id == "kimi-k2.6-iq1_s" and not any("unloaded" in n for n in p.notes)
    assert mock_state.active() is not None  # still loaded: only the approved run unloads it
    r = await workflow.execute(p, "q", 32, _noop)
    assert r.model.id == "kimi-k2.6-iq1_s" and r.answer.startswith("[mock]")


async def test_warm_model_is_kept_unless_a_friend_joined(workflow: Workflow, mock_state: MockState) -> None:
    mock_state.offline = {"mac-b"}
    first = await workflow.execute(await workflow.propose(RUN), "q", 32, _noop)
    pool = await workflow.pool()
    assert workflow._still_best(first.deployment_id, RUN, pool)  # same kind of request, same pool: keep it warm
    assert not workflow._still_best(first.deployment_id, Intent(action="run", prompt="x", model_hint="qwen"), pool)
    assert not workflow._still_best(first.deployment_id, Intent(action="run", prompt="x", preference="fastest"),
                                    pool)
    mock_state.offline = set()  # mac-b came back: a bigger model may fit now, so plan again
    assert not workflow._still_best(first.deployment_id, RUN, await workflow.pool())


async def test_agent_run_request_with_my_idle_model_loaded(tmp_path, gateway, settings) -> None:
    from hc_agent.agent import RunRequest, run_for_agent
    from hc_agent.conversation import Conversation
    from tests.conftest import MEMBER

    wf = Workflow(gateway, settings, poll_s=0.01)
    convo = Conversation(settings, wf, address="agent1qself")
    await wf.execute(await wf.propose(Intent(action="run", prompt="x", model_hint="gpt-oss")), "q", 32, _noop)
    reply = await run_for_agent(convo, wf, settings, MEMBER, RunRequest(prompt="what is RPC?"))
    assert reply.ok and reply.model == "kimi-k2.6-iq1_s" and reply.layers
