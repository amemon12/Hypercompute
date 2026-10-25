"""What Droid may touch, enforced in three places so no single one has to be right:

1. The toolset. After the session opens, every tool except this service's own MCP tools, ToolSearch (which loads
   them) and ExitSpecMode in spec mode is restricted away and disabled by id, so Read, Execute, Edit, Create, Script
   and web tools are not offered at all.
2. The permission gate. The session runs at Autonomy.OFF, so the droid CLI asks before each MCP tool call;
   on_permission allows only an McpToolAction from the "hypercompute" server naming an allowlisted tool, and cancels
   everything else. ToolSearch never asks (droid has no permission action for it).
3. The caps. Each MCP handler calls admit() first, and the runner calls admit_loader() for every ToolSearch call it
   sees, so all of them count against the per-run call cap in this process, whichever path a call took.

Every decision is reported to the run log, allowed or not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from droid_sdk import (
    ExitSpecModeAction,
    McpToolAction,
    PermissionRequest,
    PermissionResponse,
    QuestionRequest,
    QuestionResponse,
    ToolConfirmationOutcome,
)

SERVER_NAME = "hypercompute"
TOOL_NAMES = ("pool_snapshot", "current_layout", "check_layout")
CAPPED_TOOL = "check_layout"
SPEC_TOOL = "ExitSpecMode"
# droid 0.249 serves MCP tools as deferred tools that the model loads with ToolSearch (probed live 2026-10-24), so
# the lock keeps it. It only searches the tool catalog: everything else stays disabled, and on_permission cancels any
# action outside the allowlist.
LOADER_TOOLS = frozenset({"ToolSearch"})
MAX_LOADER_CALLS = 3  # loading three MCP tools takes one search; the live runs used one

# Proceed options for the spec plan, most restrictive first. Anything above low autonomy would stop built-in edits and
# commands from prompting, leaving only the toolset lock; then the plan is cancelled instead.
_SPEC_PROCEED_ORDER = (
    ToolConfirmationOutcome.PROCEED_ONCE,
    ToolConfirmationOutcome.PROCEED_AUTO_RUN_LOW,
)

Reporter = Callable[[str, str, dict[str, Any]], None]  # (kind, title, detail)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    tool: str
    reason: str


def _noop(kind: str, title: str, detail: dict[str, Any]) -> None:
    return None


class ToolGuard:
    def __init__(self, *, max_checks: int = 8, max_calls: int = 12, server: str = SERVER_NAME,
                 allowed: Iterable[str] = TOOL_NAMES, report: Reporter = _noop,
                 on_plan: Callable[[str, str | None], None] | None = None,
                 max_loader_calls: int = MAX_LOADER_CALLS) -> None:
        if max_checks < 1 or max_calls < max_checks:
            raise ValueError("need 1 <= max_checks <= max_calls")
        self.max_checks, self.max_calls, self.max_loader_calls = max_checks, max_calls, max_loader_calls
        self.server = server
        self.allowed = frozenset(allowed)
        self._report = report
        self._on_plan = on_plan
        self._calls: Counter[str] = Counter()  # calls the handlers ran
        # calls the permission gate let through: Droid can ask for several in parallel before any of them runs,
        # so the gate counts its own grants rather than finished calls
        self._granted: Counter[str] = Counter()
        self.locked_ids: tuple[str, ...] = ()
        self.plan_published = False

    # ---------------------------------------------------------------- counters
    @property
    def total_calls(self) -> int:
        return sum(self._calls.values())

    @property
    def checks(self) -> int:
        return self._calls[CAPPED_TOOL]

    def _over_cap(self, tool: str, counts: Counter[str] | None = None) -> str | None:
        counts = self._calls if counts is None else counts
        if sum(counts.values()) >= self.max_calls:
            return f"tool-call cap reached ({self.max_calls} per run)"
        if tool == CAPPED_TOOL and counts[CAPPED_TOOL] >= self.max_checks:
            return f"check_layout cap reached ({self.max_checks} per run)"
        return None

    def admit(self, tool: str) -> str | None:
        """Called by each tool handler before it does anything. Returns why the call is refused, or None and counts
        it."""
        if tool not in self.allowed:
            reason = f"{tool} is not an allowlisted tool"
        else:
            reason = self._over_cap(tool)
        if reason is not None:
            self._report("guard", f"refused {tool}", {"reason": reason})
            return reason
        self._calls[tool] += 1
        return None

    def admit_loader(self, tool: str) -> str | None:
        """Counts a ToolSearch call, which runs without a permission prompt or a handler of ours. It counts against
        the run's call cap and its own small cap; past either the runner stops the session."""
        if self._calls[tool] >= self.max_loader_calls:
            reason = f"{tool} cap reached ({self.max_loader_calls} per run)"
        elif self.total_calls >= self.max_calls:
            reason = f"tool-call cap reached ({self.max_calls} per run)"
        else:
            self._calls[tool] += 1
            return None
        self._report("guard", f"refused {tool}", {"reason": reason})
        return reason

    # ---------------------------------------------------------------- toolset lock
    def lock_toolset(self, tools: Iterable[Any], *, spec_mode: bool) -> tuple[list[str], list[str]]:
        """From session.list_tools(): (ids to keep, ids removed). Keeps this server's MCP tools by their real ids, so
        nothing depends on guessing the CLI's naming scheme."""
        keep, drop = [], []
        for info in tools:
            tool_id = str(getattr(info, "id", ""))
            source = getattr(info, "source", None)
            ours = (source in (None, "mcp") and self.server in tool_id
                    and any(tool_id.endswith(name) for name in self.allowed))
            if ours or tool_id in LOADER_TOOLS or (spec_mode and tool_id == SPEC_TOOL):
                keep.append(tool_id)
            else:
                drop.append(tool_id)
        self.locked_ids = tuple(keep)
        self._report("guard", "toolset locked", {"kept": keep, "removed": len(drop), "removed_ids": drop})
        return keep, drop

    # ---------------------------------------------------------------- permission gate
    def _actual_name(self, action: McpToolAction) -> str | None:
        if action.actual_tool_name:
            return action.actual_tool_name
        return next((n for n in self.allowed if action.tool_name.endswith(n)), None)

    def decide(self, action: Any, granted: Counter[str] | None = None) -> Decision:
        if isinstance(action, McpToolAction):
            name = self._actual_name(action) or action.tool_name
            server = action.server_name
            if server is not None and server != self.server:
                return Decision(False, name, f"MCP server {server!r} is not {self.server!r}")
            if server is None and self.server not in action.tool_name:
                return Decision(False, name, "the call names no server")
            if name not in self.allowed:
                return Decision(False, name, f"{name} is not on the allowlist {sorted(self.allowed)}")
            over = self._over_cap(name, granted if granted is not None else self._granted | self._calls)
            if over is not None:
                return Decision(False, name, over)
            return Decision(True, name, "plan-only tool on the allowlist")
        if isinstance(action, ExitSpecModeAction):
            return Decision(True, SPEC_TOOL, "the spec plan is published to the run log; it only describes checks")
        kind = type(action).__name__
        return Decision(False, kind, f"{kind} is never allowed in an autotune run")

    def _spec_option(self, request: PermissionRequest) -> ToolConfirmationOutcome | None:
        offered = {o.value for o in request.options}
        return next((o for o in _SPEC_PROCEED_ORDER if o in offered), None)

    def on_permission(self, request: PermissionRequest) -> PermissionResponse:
        # the larger of granted and run per tool (a call can also run without asking), and a batch counts as a whole
        pending = self._granted | self._calls
        decisions = []
        for action in request.actions:
            decision = self.decide(action, pending)
            decisions.append(decision)
            if decision.allowed and isinstance(action, McpToolAction):
                pending[decision.tool] += 1
        plan = request.plan
        if plan is not None:
            self.publish_plan(plan.text, plan.title)
        allowed = bool(decisions) and all(d.allowed for d in decisions)
        if allowed and plan is not None:
            option = self._spec_option(request)
        elif allowed:
            offered = {o.value for o in request.options}
            option = ToolConfirmationOutcome.PROCEED_ONCE if ToolConfirmationOutcome.PROCEED_ONCE in offered else None
        else:
            option = None
        for d in decisions:
            ok = allowed and option is not None
            self._report("permission", f"{'allowed' if ok else 'denied'} {d.tool}",
                         {"reason": d.reason if ok or not d.allowed else "no safe proceed option was offered",
                          "option": option.value if ok and option else "cancel"})
        if not allowed or option is None:
            # built directly: respond() refuses an option Droid did not list, and a cancel must never fail
            return PermissionResponse(selected_option=ToolConfirmationOutcome.CANCEL)
        self._granted = pending
        return request.respond(option)

    def publish_plan(self, text: str, title: str | None) -> None:
        self.plan_published = True
        if self._on_plan is not None:
            self._on_plan(text, title)

    def on_question(self, request: QuestionRequest) -> QuestionResponse:
        """Droid may not ask the owner questions mid-run; the owner answers once, at approval."""
        self._report("permission", "cancelled a question", {"questions": [q.question for q in request.questions]})
        return request.cancel()
