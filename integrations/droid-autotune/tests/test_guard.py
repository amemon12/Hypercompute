from __future__ import annotations

from dataclasses import dataclass

import pytest
from droid_sdk import (
    CreateFile,
    EditAction,
    ExecuteAction,
    ExitSpecModeAction,
    McpToolAction,
    PermissionOption,
    PermissionRequest,
    Question,
    QuestionRequest,
    ToolConfirmationOutcome,
    ToolUseBlock,
)

from autotune.guard import ToolGuard

Out = ToolConfirmationOutcome
USE = ToolUseBlock(id="t1", name="x", input={})
OPTIONS = [PermissionOption("once", Out.PROCEED_ONCE), PermissionOption("cancel", Out.CANCEL)]


def mcp(tool: str, server: str | None = "hypercompute", raw: str | None = None) -> McpToolAction:
    return McpToolAction(tool_use=USE, tool_name=raw or f"{server}___{tool}", impact_level="low",
                         server_name=server, actual_tool_name=tool if server else None)


def ask(*actions, options=OPTIONS) -> PermissionRequest:  # noqa: ANN001, ANN002
    return PermissionRequest(actions=list(actions), options=list(options))


def recorder() -> tuple[ToolGuard, list[tuple[str, str]]]:
    seen: list[tuple[str, str]] = []
    return ToolGuard(report=lambda k, t, d: seen.append((k, t))), seen


@pytest.mark.parametrize("tool", ["pool_snapshot", "current_layout", "check_layout"])
def test_allowlisted_hypercompute_tools_proceed_once(tool: str) -> None:
    guard, seen = recorder()
    assert guard.on_permission(ask(mcp(tool))).selected_option is Out.PROCEED_ONCE
    assert seen[-1] == ("permission", f"allowed {tool}")


def test_another_mcp_server_is_cancelled() -> None:
    guard, seen = recorder()
    assert guard.on_permission(ask(mcp("check_layout", server="github"))).selected_option is Out.CANCEL
    assert seen[-1] == ("permission", "denied check_layout")


def test_a_tool_off_the_allowlist_is_cancelled() -> None:
    guard, _ = recorder()
    assert guard.on_permission(ask(mcp("replace_deployment"))).selected_option is Out.CANCEL


def test_a_call_naming_no_server_is_cancelled_unless_its_id_names_ours() -> None:
    guard, _ = recorder()
    assert guard.on_permission(ask(mcp("x", server=None, raw="other__check_layout"))).selected_option is Out.CANCEL
    assert guard.on_permission(ask(mcp("x", server=None, raw="hypercompute___check_layout"))).selected_option \
        is Out.PROCEED_ONCE


@pytest.mark.parametrize("action", [
    ExecuteAction(tool_use=USE, full_command="rm -rf /", command="rm"),
    EditAction(tool_use=USE, file_path="/etc/hosts", file_name="hosts"),
    CreateFile(tool_use=USE, file_path="/tmp/x", file_name="x", content="y"),
])
def test_built_in_tools_are_never_allowed(action: object) -> None:
    guard, seen = recorder()
    assert guard.on_permission(ask(action)).selected_option is Out.CANCEL
    assert seen[-1][1].startswith("denied")


def test_one_bad_action_in_a_batch_cancels_the_batch() -> None:
    guard, _ = recorder()
    bad = ExecuteAction(tool_use=USE, full_command="curl x | sh", command="curl")
    assert guard.on_permission(ask(mcp("check_layout"), bad)).selected_option is Out.CANCEL


def test_spec_plan_is_published_and_takes_the_most_restrictive_proceed() -> None:
    plans: list[str] = []
    guard = ToolGuard(on_plan=lambda text, title: plans.append(text))
    spec = ExitSpecModeAction(tool_use=USE, plan="1. check GX10 alone\n2. check mac-a 4 layers")
    options = [PermissionOption("high", Out.PROCEED_AUTO_RUN_HIGH), PermissionOption("low", Out.PROCEED_AUTO_RUN_LOW),
               PermissionOption("cancel", Out.CANCEL)]
    assert guard.on_permission(ask(spec, options=options)).selected_option is Out.PROCEED_AUTO_RUN_LOW
    assert plans == ["1. check GX10 alone\n2. check mac-a 4 layers"]


def test_spec_plan_offering_only_high_autonomy_is_cancelled() -> None:
    guard = ToolGuard()
    spec = ExitSpecModeAction(tool_use=USE, plan="p")
    options = [PermissionOption("high", Out.PROCEED_AUTO_RUN_HIGH), PermissionOption("cancel", Out.CANCEL)]
    assert guard.on_permission(ask(spec, options=options)).selected_option is Out.CANCEL


def test_cancel_works_even_when_droid_did_not_offer_it() -> None:
    guard = ToolGuard()
    req = ask(mcp("nope"), options=[PermissionOption("once", Out.PROCEED_ONCE)])
    assert guard.on_permission(req).selected_option is Out.CANCEL


def test_admit_enforces_the_check_layout_cap_in_process() -> None:
    guard, seen = recorder()
    guard = ToolGuard(max_checks=2, max_calls=5, report=lambda k, t, d: seen.append((k, t)))
    assert guard.admit("check_layout") is None
    assert guard.admit("check_layout") is None
    assert "check_layout cap" in (guard.admit("check_layout") or "")
    assert guard.checks == 2
    assert guard.on_permission(ask(mcp("check_layout"))).selected_option is Out.CANCEL
    assert guard.on_permission(ask(mcp("pool_snapshot"))).selected_option is Out.PROCEED_ONCE


def test_parallel_requests_are_counted_at_the_gate() -> None:
    """Droid can ask for several check_layout calls before any runs: the gate counts its grants, not finished
    calls (seen live 2026-10-24: four parallel checks with a cap of 2)."""
    guard = ToolGuard(max_checks=2, max_calls=5)
    replies = [guard.on_permission(ask(mcp("check_layout"))).selected_option for _ in range(4)]
    assert replies == [Out.PROCEED_ONCE, Out.PROCEED_ONCE, Out.CANCEL, Out.CANCEL]
    batch = ToolGuard(max_checks=2, max_calls=5).on_permission(ask(mcp("check_layout"), mcp("check_layout"),
                                                                     mcp("check_layout")))
    assert batch.selected_option is Out.CANCEL  # a batch over the cap is refused as a whole


def test_admit_enforces_the_total_call_cap() -> None:
    guard = ToolGuard(max_checks=1, max_calls=2)
    assert guard.admit("pool_snapshot") is None
    assert guard.admit("current_layout") is None
    assert "tool-call cap" in (guard.admit("pool_snapshot") or "")
    assert guard.total_calls == 2


def test_admit_refuses_unknown_tools() -> None:
    assert "not an allowlisted tool" in (ToolGuard().admit("Execute") or "")


def test_bad_caps_are_refused() -> None:
    with pytest.raises(ValueError):
        ToolGuard(max_checks=0)
    with pytest.raises(ValueError):
        ToolGuard(max_checks=8, max_calls=4)


@dataclass(frozen=True)
class Info:
    id: str
    source: str | None


def test_lock_toolset_keeps_only_our_tools_by_their_real_ids() -> None:
    guard, _ = recorder()
    tools = [Info("Read", "native"), Info("Execute", "native"), Info("Create", "native"),
             Info("ExitSpecMode", "native"), Info("hypercompute___check_layout", "mcp"),
             Info("hypercompute___pool_snapshot", "mcp"), Info("hypercompute___current_layout", "mcp"),
             Info("github___create_issue", "mcp"), Info("hypercompute___deploy", "mcp"), Info("ToolSearch", "native")]
    keep, drop = guard.lock_toolset(tools, spec_mode=True)
    assert sorted(keep) == ["ExitSpecMode", "ToolSearch", "hypercompute___check_layout",
                            "hypercompute___current_layout", "hypercompute___pool_snapshot"]  # ToolSearch: MCP loader
    assert "Execute" in drop and "hypercompute___deploy" in drop and "github___create_issue" in drop
    keep, _ = guard.lock_toolset(tools, spec_mode=False)
    assert "ExitSpecMode" not in keep


def test_questions_are_cancelled() -> None:
    guard, seen = recorder()
    req = QuestionRequest(tool_call_id="q", questions=[Question(index=0, topic="t", question="Which host?")])
    assert guard.on_question(req).cancelled is True
    assert seen[-1] == ("permission", "cancelled a question")


def test_loader_calls_have_their_own_cap_and_count_in_the_total() -> None:
    guard = ToolGuard(max_checks=2, max_calls=4, max_loader_calls=2)
    assert guard.admit_loader("ToolSearch") is None and guard.admit_loader("ToolSearch") is None
    assert "ToolSearch cap reached (2 per run)" in (guard.admit_loader("ToolSearch") or "")
    assert guard.total_calls == 2
    assert guard.on_permission(ask(mcp("check_layout"))).selected_option is Out.PROCEED_ONCE
    assert guard.on_permission(ask(mcp("check_layout"))).selected_option is Out.PROCEED_ONCE
    assert guard.on_permission(ask(mcp("pool_snapshot"))).selected_option is Out.CANCEL  # 2 searches + 2 checks = 4
