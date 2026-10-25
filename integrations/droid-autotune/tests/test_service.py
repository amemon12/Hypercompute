"""The whole loop against the mock gateway: join -> settle -> run -> planner verdicts -> diff -> owner approval ->
replace. The runner is the scripted one, which goes through the same tool handlers and permission gate as Droid."""

from __future__ import annotations

import json
import time

import pytest

from autotune.models import AutotuneProposal, Candidate, Stage
from autotune.runner import RunOutcome, ScriptCall, ScriptedRunner
from autotune.service import AutotuneService, ServiceError
from mock_gateway.app import PEERS, Joined
from tests.conftest import MAC_A

N = 36


def lay(*stages: tuple[str, str, int]) -> list[dict]:
    return [{"node_id": n, "device": d, "layers": k} for n, d, k in stages]


SCRIPT = [
    ScriptCall("pool_snapshot", {}), ScriptCall("current_layout", {}),
    ScriptCall("check_layout", {"label": "G14 alone", "stages": lay(("local", "CUDA0", N))}),
    ScriptCall("check_layout", {"label": "mac-a 7", "stages": lay((MAC_A, "MTL0", 7), ("local", "CUDA0", N - 7))}),
    ScriptCall("check_layout", {"label": "mac-a 4", "stages": lay((MAC_A, "MTL0", 4), ("local", "CUDA0", N - 4))}),
]


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def service_for(settings, client, store, runner, clock=None) -> AutotuneService:  # noqa: ANN001
    return AutotuneService(settings, client, lambda: runner, store, runner_name=runner.name,
                           clock=clock or Clock())


def join(app, name: str) -> None:  # noqa: ANN001
    app.state.mock.peers[name] = Joined(PEERS[name], time.monotonic() - 10)


async def finish(service: AutotuneService) -> None:
    assert service._run_task is not None
    await service._run_task


async def test_join_settles_runs_and_waits_for_the_owner(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    clock = Clock()
    service = service_for(settings, client, store, ScriptedRunner(SCRIPT), clock)
    assert (await service.poll_once()).action == "baseline"
    join(app, "mac-a")
    clock.t = 1
    assert (await service.poll_once()).action == "changed"
    clock.t = 10
    assert (await service.poll_once()).action == "settling"
    clock.t = 22
    assert (await service.poll_once()).action == "fire"
    await finish(service)
    run = store.list()[0]
    assert run.state == "awaiting_approval", run.error
    assert run.trigger["reason"] == "mac-a joined"
    assert run.plan_text.startswith("Scripted plan")
    refused = [v for v in run.verdicts if not v.accepted]
    assert [v.code for v in refused] == ["layout_memory"]
    assert "mac-a MTL0 needs" in refused[0].reason and refused[0].fix == "give that stage fewer layers"
    accepted = {v.label for v in run.verdicts if v.accepted}
    assert accepted == {"G14 alone", "mac-a 4"}
    # G14 alone is the current layout, so only mac-a 4 is a change
    i = next(i for i, c in enumerate(run.proposal.candidates) if c.label == "mac-a 4")
    d = run.diffs[i]
    assert (d.hops_before, d.hops_after) == (0, 1) and d.decode_ms_after < d.decode_ms_before
    kinds = {s.kind for s in run.steps}
    assert {"trigger", "context", "plan", "permission", "tool", "verdict", "proposal", "owner"} <= kinds
    # the audit trail on disk has every step
    lines = (settings.runs_dir / f"{run.id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(run.steps) and json.loads(lines[0])["kind"] == "trigger"


async def test_owner_approval_replaces_the_deployment(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    join(app, "mac-a")
    service = service_for(settings, client, store, ScriptedRunner(SCRIPT))
    await service.trigger()
    await finish(service)
    run = store.list()[0]
    old_dep = run.deployment_id
    i = next(i for i, c in enumerate(run.proposal.candidates) if c.label == "mac-a 4")
    assert (await service.approve(run.id, i)).state == "applying"
    await service._apply_task
    run = store.get(run.id)
    assert run.state == "applied", run.error
    deps = {d["id"]: d for d in app.state.mock.deployments}
    assert deps[old_dep]["state"] == "stopped"
    new = deps[run.applied["deployment_id"]]
    assert new["state"] == "ready" and new["replaces"] == old_dep
    assert [len(a["layers"]) for a in new["plan"]["assignments"]] == [4, 32]
    assert any(s.title.startswith("re-checked mac-a 4 before apply") for s in run.steps)


async def test_nothing_changes_without_approval(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    join(app, "mac-a")
    service = service_for(settings, client, store, ScriptedRunner(SCRIPT))
    before = [d["id"] for d in app.state.mock.deployments]
    await service.trigger()
    await finish(service)
    run = store.list()[0]
    assert service.reject(run.id).state == "rejected"
    assert [d["id"] for d in app.state.mock.deployments] == before
    with pytest.raises(ServiceError):
        await service.approve(run.id, 0)


class Liar:
    """Claims a layout the planner refuses, without ever checking it."""

    name = "droid"

    async def run(self, ctx, prompt):  # noqa: ANN001, ANN201
        bad = Candidate(label="mac-a 9", rationale="trust me", plan_id="made-up",
                        stages=[Stage(node_id=MAC_A, device="MTL0", layers=9),
                                Stage(node_id="local", device="CUDA0", layers=N - 9)])
        return RunOutcome(AutotuneProposal(summary="apply it", recommendation="apply", recommended_index=0,
                                           candidates=[bad]))

    async def interrupt(self) -> None:
        return None


async def test_a_refused_candidate_cannot_be_approved(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    join(app, "mac-a")
    service = service_for(settings, client, store, Liar())
    await service.trigger()
    await finish(service)
    run = store.list()[0]
    assert run.state == "no_change"
    v = run.verdicts[run.candidate_verdicts[0]]
    assert v.checked_by == "service" and not v.accepted and v.code == "layout_memory"
    assert run.diffs == {}
    with pytest.raises(ServiceError) as err:
        await service.approve(run.id, 0)
    assert err.value.status == 422


async def test_the_check_cap_holds_even_for_a_runaway_script(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    join(app, "mac-a")
    flood = [ScriptCall("check_layout", {"label": f"try {k}", "stages": lay(("local", "CUDA0", N))})
             for k in range(20)]
    service = service_for(settings, client, store, ScriptedRunner(flood))
    await service.trigger()
    await finish(service)
    run = store.list()[0]
    assert sum(1 for v in run.verdicts if v.checked_by == "droid") == settings.max_checks
    assert any(s.title == "denied check_layout" for s in run.steps)


async def test_a_disallowed_tool_never_runs(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    script = [ScriptCall("check_layout", {"stages": lay(("local", "CUDA0", N))}, server="filesystem"),
              ScriptCall("replace_deployment", {"plan_id": "x"})]
    service = service_for(settings, client, store, ScriptedRunner(script))
    await service.trigger()
    await finish(service)
    run = store.list()[0]
    assert run.verdicts == []
    assert [s.title for s in run.steps if s.kind == "permission"] == ["denied check_layout",
                                                                      "denied replace_deployment"]


async def test_a_new_pool_change_supersedes_an_unanswered_run(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    join(app, "mac-a")
    service = service_for(settings, client, store, ScriptedRunner(SCRIPT))
    first = await service.trigger()
    await finish(service)
    assert store.get(first.id).state == "awaiting_approval"
    await service.trigger()
    await finish(service)
    assert store.get(first.id).state == "superseded"


async def test_trigger_waits_while_a_model_loads(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    app.state.mock.deployments[0]["state"] = "loading"
    service = service_for(settings, client, store, ScriptedRunner(SCRIPT))
    with pytest.raises(ServiceError) as err:
        await service.trigger()
    assert err.value.status == 409 and "loading" in str(err.value)


async def test_no_deployment_and_no_fallback_model_is_a_clear_failure(settings, store, mock_g14) -> None:  # noqa: ANN001
    app, client = mock_g14
    app.state.mock.deployments.clear()
    service = service_for(settings, client, store, ScriptedRunner(SCRIPT))
    await service.trigger()
    await finish(service)
    run = store.list()[0]
    assert run.state == "failed" and "no ready deployment" in run.error


async def test_on_the_gx10_a_keep_recommendation_changes_nothing(settings, store) -> None:  # noqa: ANN001
    from mock_gateway.app import create_app
    from tests.conftest import make_client

    app = create_app("gx10", verify_s=0, replace_delay_s=0)
    client = make_client(app)
    try:
        join(app, "mac-a")
        script = [ScriptCall("check_layout", {"label": "GX10 alone", "stages": lay(("local", "CUDA0", N))}),
                  ScriptCall("check_layout", {"label": "mac-a 4", "stages": lay((MAC_A, "MTL0", 4),
                                                                               ("local", "CUDA0", N - 4))})]
        service = service_for(settings, client, store, ScriptedRunner(script, recommend="keep"))
        before = [d["id"] for d in app.state.mock.deployments]
        await service.trigger()
        await finish(service)
        run = store.list()[0]
        assert run.state == "no_change" and run.proposal.recommendation == "keep"
        assert all(v.accepted and v.tier == "gpu" for v in run.verdicts)  # all resident on the GB10
        pooled = next(d for i, d in run.diffs.items() if run.proposal.candidates[i].label == "mac-a 4")
        assert (pooled.hops_before, pooled.hops_after) == (0, 1) and pooled.decode_ms_after is None
        assert [d["id"] for d in app.state.mock.deployments] == before
    finally:
        await client.aclose()
