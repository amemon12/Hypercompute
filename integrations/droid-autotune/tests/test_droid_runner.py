"""DroidRunner end to end with the droid CLI replaced by a fake Session that behaves like it: it lists tools, takes the
restrict_tools update, sends the spec plan and each MCP call through the permission handler, calls the in-process
tools that were allowed, streams the SDK's own event types and returns structured output."""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import droid_sdk
import pytest

from autotune import runner as runner_mod
from autotune.models import AutotuneProposal
from autotune.runner import DroidRunner
from autotune.service import AutotuneService
from mock_gateway.app import PEERS, Joined
from tests.conftest import MAC_A

T = droid_sdk.ToolConfirmationOutcome
NOW = datetime.now(UTC)


@dataclasses.dataclass(frozen=True)
class FakeTool:
    id: str
    source: str
    allowed: bool = True


class FakeStream:
    def __init__(self, session: FakeSession, prompt: str) -> None:
        self.session, self.prompt, self.result = session, prompt, None
        self._events: list[Any] = []

    async def __aenter__(self) -> FakeStream:
        await self.session.play(self)
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class FakeSession:
    """Stands in for droid_sdk.Session. `script` is what the model would do."""

    script: list[tuple[str, dict[str, Any]]] = []
    answer: dict[str, Any] = {}
    instances: list[FakeSession] = []
    honours_lock = True  # False: a CLI that ignores both restrict_tools and disabled_tools

    def __init__(self, *, cwd, model, config, interactions, runtime, api_key) -> None:  # noqa: ANN001
        self.config, self.interactions, self.api_key = config, interactions, api_key
        self.restricted: frozenset[str] | None = None
        self.disabled: frozenset[str] = frozenset()
        self.interrupted = False
        self.id = "sess-1"
        self.settings = droid_sdk.SessionSettings(model="fake-model", reasoning_effort=droid_sdk.ReasoningEffort.LOW)
        self.tools = {t.name: t for t in config.mcp_servers[0].tools}
        FakeSession.instances.append(self)

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def list_tools(self) -> list[FakeTool]:
        """As droid 0.249 behaved live: restrict_tools leaves AskUser, Script, ToolSearch and WaitForScript on;
        disabled_tools turns them off except ToolSearch, which stays on regardless."""
        ids = [("Read", "native"), ("Execute", "native"), ("ExitSpecMode", "native"), ("AskUser", "native"),
               ("Script", "native"), ("ToolSearch", "native"), ("WaitForScript", "native"),
               *((f"hypercompute___{n}", "mcp") for n in self.tools)]
        sticky = {"AskUser", "Script", "ToolSearch", "WaitForScript"}

        def allowed(i: str) -> bool:
            if not self.honours_lock:
                return True
            if i == "ToolSearch":
                return True
            if i in self.disabled:
                return False
            return self.restricted is None or i in self.restricted or i in sticky

        return [FakeTool(i, src, allowed(i)) for i, src in ids]

    async def update_settings(self, *, restrict_tools: Any = None, disabled_tools: Any = None) -> None:
        self.restricted = frozenset(restrict_tools) if restrict_tools is not None else None
        self.disabled = frozenset(disabled_tools or ())

    async def interrupt(self) -> None:
        self.interrupted = True

    def stream(self, prompt: str, *, output: type, timeout: float) -> FakeStream:
        assert output is AutotuneProposal and timeout > 0
        return FakeStream(self, prompt)

    def _ask(self, action: Any) -> bool:
        options = [droid_sdk.PermissionOption("once", T.PROCEED_ONCE),
                   droid_sdk.PermissionOption("low", T.PROCEED_AUTO_RUN_LOW),
                   droid_sdk.PermissionOption("cancel", T.CANCEL)]
        reply = self.interactions.on_permission(droid_sdk.PermissionRequest(actions=[action], options=options))
        return reply.selected_option is not T.CANCEL

    async def play(self, stream: FakeStream) -> None:
        use = droid_sdk.ToolUseBlock(id="spec", name="ExitSpecMode", input={})
        self._ask(droid_sdk.ExitSpecModeAction(tool_use=use, plan="1. read the pool\n2. check 2 layouts"))
        events: list[Any] = [droid_sdk.AssistantMessage(id="m1", content=(), text="1. read the pool\n2. check 2",
                                                         created_at=NOW, updated_at=NOW)]
        for tool, args in self.script:
            tid = uuid.uuid4().hex
            server = "hypercompute" if tool in self.tools else "shell"
            use = droid_sdk.ToolUseBlock(id=tid, name=f"{server}___{tool}" if tool in self.tools else tool,
                                         input=args)
            events.append(droid_sdk.ToolCall(name=use.name, tool_use_id=tid, input=args))
            allowed = self._ask(droid_sdk.McpToolAction(tool_use=use, tool_name=use.name, impact_level="low",
                                                        server_name=server, actual_tool_name=tool))
            if allowed and tool in self.tools and (self.restricted is None or use.name in self.restricted):
                out = await self.tools[tool].handler(args)
                events.append(droid_sdk.ToolResult(tool_use_id=tid, tool_name=use.name, content=out.content,
                                                   is_error=out.is_error))
            else:
                events.append(droid_sdk.ToolResult(tool_use_id=tid, tool_name=use.name, content="denied",
                                                   is_error=True))
        events.append(droid_sdk.TokenUsageUpdate(input_tokens=1200, output_tokens=300, cache_creation_tokens=0,
                                                 cache_read_tokens=0, thinking_tokens=0, factory_credits=0.4))
        stream._events = events
        usage = droid_sdk.Usage(1200, 300, 0, 0, 0, 0.4)
        stream.result = droid_sdk.RunSuccess(text="done", messages=(), usage=usage, duration=timedelta(seconds=9),
                                             turn_count=1, session_id=self.id,
                                             output=AutotuneProposal.model_validate(self.answer))


def stages(*s: tuple[str, str, int]) -> list[dict[str, Any]]:
    return [{"node_id": n, "device": d, "layers": k} for n, d, k in s]


MAC4 = stages((MAC_A, "MTL0", 4), ("local", "CUDA0", 32))


@pytest.fixture
def fake_droid(monkeypatch: pytest.MonkeyPatch) -> type[FakeSession]:
    monkeypatch.setattr(runner_mod, "Session", FakeSession)
    FakeSession.instances = []
    FakeSession.honours_lock = True
    FakeSession.script = [("pool_snapshot", {}), ("current_layout", {}),
                          ("check_layout", {"label": "mac-a 8", "stages": stages((MAC_A, "MTL0", 8),
                                                                                 ("local", "CUDA0", 28))}),
                          ("check_layout", {"label": "mac-a 4", "stages": MAC4})]
    FakeSession.answer = {"summary": "mac-a takes 4 layers; 8 did not fit", "recommendation": "apply",
                          "recommended_index": 0,
                          "candidates": [{"label": "mac-a 4", "stages": MAC4, "rationale": "fits, fewer SSD reads"}]}
    return FakeSession


async def test_a_droid_run_through_the_sdk_session(settings, store, mock_g14, fake_droid) -> None:  # noqa: ANN001
    app, client = mock_g14
    app.state.mock.peers["mac-a"] = Joined(PEERS["mac-a"], time.monotonic() - 10)
    live = dataclasses.replace(settings, runner="droid", factory_api_key="fk-test", droid_path="droid")
    service = AutotuneService(live, client, lambda: DroidRunner(live), store, runner_name="droid")
    await service.trigger()
    await service._run_task
    run = store.list()[0]
    session = fake_droid.instances[0]
    assert session.api_key == "fk-test"
    assert session.restricted == {"ExitSpecMode", "ToolSearch", "hypercompute___pool_snapshot",
                                  "hypercompute___current_layout", "hypercompute___check_layout"}
    assert {"Read", "Execute", "AskUser", "Script", "WaitForScript"} <= session.disabled
    assert run.state == "awaiting_approval", run.error
    assert run.plan_text == "1. read the pool\n2. check 2 layouts"
    assert [v.accepted for v in run.verdicts] == [False, True]
    assert run.verdicts[1].plan_id and run.candidate_verdicts == {0: 1}
    assert run.usage == {"input_tokens": 1200, "output_tokens": 300, "factory_credits": 0.4}
    titles = [s.title for s in run.steps]
    assert "Droid session opened" in titles and "toolset locked" in titles
    assert "Droid: 1. read the pool" in titles
    assert "allowed check_layout" in titles and not session.interrupted
    lock = next(s for s in run.steps if s.title == "toolset lock read back")
    assert lock.detail["allowed_outside_lock"] == [] and "ToolSearch" in lock.detail["allowed"]
    assert any(t.startswith("Droid calls hypercompute___check_layout") for t in titles)
    dumped = json.dumps(run.model_dump(mode="json"))
    assert "fk-test" not in dumped and "fk-test" not in (settings.runs_dir / f"{run.id}.jsonl").read_text("utf-8")


async def test_a_run_without_our_tools_fails_clearly(settings, store, mock_g14, fake_droid, monkeypatch) -> None:  # noqa: ANN001
    app, client = mock_g14

    async def no_tools(self: FakeSession) -> list[FakeTool]:
        return [FakeTool("Read", "native")]

    monkeypatch.setattr(FakeSession, "list_tools", no_tools)
    live = dataclasses.replace(settings, runner="droid", factory_api_key="fk-test", droid_path="droid")
    runner = DroidRunner(live)
    runner.tool_wait_s = 0.0
    service = AutotuneService(live, client, lambda: runner, store, runner_name="droid")
    await service.trigger()
    await service._run_task
    run = store.list()[0]
    assert run.state == "failed" and "none of the hypercompute MCP tools" in run.error



async def test_a_cli_that_ignores_the_toolset_lock_stops_the_run(settings, store, mock_g14, fake_droid) -> None:  # noqa: ANN001
    app, client = mock_g14
    fake_droid.honours_lock = False
    live = dataclasses.replace(settings, runner="droid", factory_api_key="fk-test", droid_path="droid")
    service = AutotuneService(live, client, lambda: DroidRunner(live), store, runner_name="droid")
    await service.trigger()
    await service._run_task
    run = store.list()[0]
    assert run.state == "failed" and "lock did not hold: Read, Execute" in run.error
    assert run.verdicts == []  # it stopped before the model could call anything


async def test_a_call_outside_the_lock_ends_the_run(settings, store, mock_g14, fake_droid) -> None:  # noqa: ANN001
    app, client = mock_g14
    fake_droid.script = [("pool_snapshot", {}), ("Script", {"code": "print('outside the lock')"}),
                         ("check_layout", {"label": "mac-a 4", "stages": MAC4})]
    live = dataclasses.replace(settings, runner="droid", factory_api_key="fk-test", droid_path="droid")
    service = AutotuneService(live, client, lambda: DroidRunner(live), store, runner_name="droid")
    await service.trigger()
    await service._run_task
    run = store.list()[0]
    assert run.state == "failed" and "Script, which is outside the toolset lock" in run.error
    assert fake_droid.instances[0].interrupted
    titles = [s.title for s in run.steps]
    assert "denied Script" in titles and "stopping the session: Script is outside the lock" in titles


class _Interruptible:
    def __init__(self) -> None:
        self.interrupted = False

    async def interrupt(self) -> None:
        self.interrupted = True


def _log_ctx(store, **guard):  # noqa: ANN001, ANN003, ANN202
    from types import SimpleNamespace

    from autotune.guard import ToolGuard

    run = store.create(runner="droid", trigger={"reason": "test"})
    plans: list[str] = []
    return SimpleNamespace(store=store, run_id=run.id,
                           guard=ToolGuard(on_plan=lambda text, title: plans.append(text), **guard)), plans


def _runner(settings) -> DroidRunner:  # noqa: ANN001
    return DroidRunner(dataclasses.replace(settings, factory_api_key="fk-test", droid_path="droid"))


async def test_a_plan_sharing_its_message_with_a_check_is_still_the_plan(settings, store) -> None:  # noqa: ANN001
    """The SDK emits a message's ToolCall events before the message (streaming.py CreateMessageNotification), so the
    plan text arrives after the check it announces."""
    ctx, plans = _log_ctx(store)
    runner = _runner(settings)
    pool = droid_sdk.ToolUseBlock(id="a", name="hypercompute___pool_snapshot", input={})
    check = droid_sdk.ToolUseBlock(id="b", name="hypercompute___check_layout", input={"stages": MAC4})
    for event in (droid_sdk.ToolCall(name=pool.name, tool_use_id="a", input={}),
                  droid_sdk.ToolCall(name=check.name, tool_use_id="b", input={"stages": MAC4}),
                  droid_sdk.AssistantMessage(id="m1", content=(droid_sdk.TextBlock(text="1. read the pool\n2. check"),
                                                               pool, check),
                                             text="1. read the pool\n2. check", created_at=NOW, updated_at=NOW)):
        await runner._log(ctx, event)
    assert plans == ["1. read the pool\n2. check"] and ctx.guard.plan_published


async def test_text_after_an_earlier_check_is_not_the_plan(settings, store) -> None:  # noqa: ANN001
    ctx, plans = _log_ctx(store)
    runner = _runner(settings)
    check = droid_sdk.ToolUseBlock(id="b", name="hypercompute___check_layout", input={})
    await runner._log(ctx, droid_sdk.ToolCall(name=check.name, tool_use_id="b", input={}))
    await runner._log(ctx, droid_sdk.AssistantMessage(id="m1", content=(check,), text="", created_at=NOW,
                                                      updated_at=NOW))
    await runner._log(ctx, droid_sdk.AssistantMessage(id="m2", content=(droid_sdk.TextBlock(text="the verdict was"),),
                                                      text="the verdict was", created_at=NOW, updated_at=NOW))
    assert plans == []


async def test_a_toolsearch_loop_stops_at_its_cap(settings, store) -> None:  # noqa: ANN001
    """ToolSearch runs without a permission prompt or a handler of ours; the runner counts it and stops the session
    past the cap, so it cannot spin paid turns until the timeout."""
    ctx, _ = _log_ctx(store)
    runner = _runner(settings)
    runner._keep = frozenset({"ToolSearch", "hypercompute___check_layout"})
    session = _Interruptible()
    runner._session = session  # type: ignore[assignment]
    for i in range(ctx.guard.max_loader_calls + 1):
        await runner._log(ctx, droid_sdk.ToolCall(name="ToolSearch", tool_use_id=f"s{i}", input={"query": "deploy"}))
    assert session.interrupted and "ToolSearch cap reached (3 per run)" in (runner._violation or "")
    assert ctx.guard.total_calls == 3
    titles = [s.title for s in store.get(ctx.run_id).steps]
    assert "stopping the session: ToolSearch cap reached (3 per run)" in titles


async def test_toolsearch_counts_against_the_whole_call_cap(settings, store) -> None:  # noqa: ANN001
    ctx, _ = _log_ctx(store, max_checks=1, max_calls=2)
    runner = _runner(settings)
    runner._keep = frozenset({"ToolSearch"})
    runner._session = _Interruptible()  # type: ignore[assignment]
    await runner._log(ctx, droid_sdk.ToolCall(name="ToolSearch", tool_use_id="s0", input={}))
    assert ctx.guard.admit("pool_snapshot") is None
    assert "tool-call cap" in (ctx.guard.admit("check_layout") or "")  # 1 search + 1 call = the cap of 2
