"""Runners turn a RunContext into a proposal.

DroidRunner is the product: one Factory Droid session per run through the Python Droid SDK (droid_sdk.Session over a
`droid exec` subprocess), with this service's MCP tools as its only tools. ScriptedRunner drives the very same tool
handlers through the very same permission gate from a fixed script; it exists for the tests and is labelled
"scripted" everywhere it shows, so a scripted run is never mistaken for Droid's work.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from droid_sdk import (
    AssistantMessage,
    Autonomy,
    ErrorEvent,
    InteractionHandlers,
    McpToolAction,
    Mode,
    PermissionOption,
    PermissionRequest,
    Runtime,
    Session,
    SessionConfig,
    ToolCall,
    ToolConfirmationOutcome,
    ToolResult,
    ToolUseBlock,
)
from droid_sdk.mcp import create_sdk_mcp_server

from . import layouts
from .config import Settings
from .guard import LOADER_TOOLS, SERVER_NAME, TOOL_NAMES
from .models import AutotuneProposal, Candidate
from .prompt import SYSTEM_APPEND
from .tools import RunContext, build_tools


@dataclass(frozen=True)
class RunOutcome:
    proposal: AutotuneProposal | None
    text: str = ""
    error: str | None = None
    usage: dict[str, Any] | None = None


class Runner(Protocol):
    name: str

    async def run(self, ctx: RunContext, prompt: str) -> RunOutcome: ...

    async def interrupt(self) -> None: ...


def _clip(value: Any, n: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= n else text[: n - 3] + "..."


class DroidRunner:
    name = "droid"
    tool_wait_s = 15.0  # how long to wait for the CLI to list this server's MCP tools

    def __init__(self, settings: Settings) -> None:
        ready, why = settings.droid_ready
        if not ready:
            raise RuntimeError(why)
        self.settings = settings
        self._session: Session | None = None
        self._keep: frozenset[str] = frozenset()
        self._violation: str | None = None
        self._layout_calls = 0  # check_layout calls seen in the stream, for telling the plan from later prose

    def _config(self, server: Any) -> SessionConfig:
        return SessionConfig(
            mode=Mode.SPEC if self.settings.droid_mode == "spec" else Mode.AUTO,
            autonomy=Autonomy.OFF,  # every tool call goes through ToolGuard.on_permission
            mcp_servers=[server],
            disable_builtin_skills=True,
            system_prompt={"type": "preset", "preset": "droid", "append": SYSTEM_APPEND},
        )

    async def run(self, ctx: RunContext, prompt: str) -> RunOutcome:
        server = create_sdk_mcp_server(SERVER_NAME, build_tools(ctx))
        cwd = self.settings.runs_dir / "droid-cwd" / ctx.run_id  # empty: there is nothing here for Droid to read
        cwd.mkdir(parents=True, exist_ok=True)
        session = Session(
            cwd=cwd, model=self.settings.droid_model, config=self._config(server),
            interactions=InteractionHandlers(on_permission=ctx.guard.on_permission,
                                             on_question=ctx.guard.on_question),
            runtime=Runtime(executable=Path(self.settings.droid_path or "droid")),
            api_key=self.settings.factory_api_key)
        self._session = session
        usage: dict[str, Any] | None = None
        try:
            async with session:
                settings = session.settings
                ctx.store.step(ctx.run_id, "session", "Droid session opened", {
                    "session_id": session.id, "model": settings.model, "mode": self.settings.droid_mode,
                    "autonomy": "off", "cwd": str(cwd)})
                keep, drop = await self._lock(ctx, session)
                if not any(t.endswith(name) for t in keep for name in TOOL_NAMES):
                    return RunOutcome(None, error="Droid's tool list has none of the hypercompute MCP tools")
                # restrict alone leaves AskUser, Script, ToolSearch and WaitForScript on in droid 0.249 (probed
                # live), so every other tool is also disabled by id
                await session.update_settings(restrict_tools=keep, disabled_tools=drop)
                allowed = [str(t.id) for t in await session.list_tools() if getattr(t, "allowed", True)]
                leaked = [t for t in allowed if t not in keep]
                ctx.store.step(ctx.run_id, "guard", "toolset lock read back",
                               {"allowed": allowed, "allowed_outside_lock": leaked})
                if leaked:
                    return RunOutcome(None, error=f"the toolset lock did not hold: {', '.join(leaked)[:300]}")
                self._keep = frozenset(keep)
                async with session.stream(prompt, output=AutotuneProposal, timeout=self.settings.timeout_s) as stream:
                    async for event in stream:
                        await self._log(ctx, event)
                result = stream.result
                if self._violation:
                    return RunOutcome(None, error=self._violation)
        finally:
            self._session = None
        if result.usage is not None:
            usage = {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
                     "factory_credits": result.usage.factory_credits}
        if not result.success:
            message = result.error.message if result.error else result.subtype
            return RunOutcome(None, text=result.text, error=f"Droid run ended: {message}", usage=usage)
        if result.output is None:
            why = result.output_validation_error or "no structured output"
            return RunOutcome(None, text=result.text, error=f"Droid's answer did not match the schema: {why}",
                              usage=usage)
        return RunOutcome(result.output, text=result.text, usage=usage)

    async def _lock(self, ctx: RunContext, session: Session) -> tuple[list[str], list[str]]:
        """(ids to keep, ids to disable). The CLI connects to the MCP server on its own schedule, so the list is
        read again for a few seconds until this server's tools show up."""
        tools: Any = []
        deadline = asyncio.get_running_loop().time() + self.tool_wait_s
        while True:
            tools = await session.list_tools()
            if any(SERVER_NAME in str(getattr(t, "id", "")) for t in tools):
                break
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.5)
        return ctx.guard.lock_toolset(tools, spec_mode=self.settings.droid_mode == "spec")

    async def _log(self, ctx: RunContext, event: Any) -> None:
        store, rid = ctx.store, ctx.run_id
        if isinstance(event, AssistantMessage) and event.text.strip():
            store.step(rid, "droid", "Droid: " + _clip(event.text.strip().splitlines()[0], 140),
                       {"text": _clip(event.text, 4000)})
            # the SDK emits a message's ToolCall events before the message itself, so checks carried by this same
            # message do not make its text "later"
            own = sum(isinstance(b, ToolUseBlock) and b.name.endswith("check_layout") for b in event.content)
            earlier = self._layout_calls - own
            if not ctx.guard.plan_published and earlier == 0 and not event.text.lstrip().startswith("{"):
                # outside spec mode, or when Droid skips ExitSpecMode: its first words before any check are the plan
                ctx.guard.publish_plan(_clip(event.text.strip(), 2000), "Droid's plan")
        elif isinstance(event, ToolCall):
            store.step(rid, "droid", f"Droid calls {event.name}", {"input": _clip(dict(event.input), 2000)})
            self._layout_calls += event.name.endswith("check_layout")
            if event.name in LOADER_TOOLS and self._violation is None:
                refused = ctx.guard.admit_loader(event.name)
                if refused:
                    self._violation = f"Droid called {event.name} past the cap: {refused}"
                    store.step(rid, "guard", f"stopping the session: {refused}", {})
                    if self._session is not None:
                        await self._session.interrupt()
            if self._keep and event.name not in self._keep and self._violation is None:
                # the permission gate cancels it anyway; a call outside the lock still ends the run
                self._violation = f"Droid called {event.name}, which is outside the toolset lock"
                store.step(rid, "guard", f"stopping the session: {event.name} is outside the lock", {})
                if self._session is not None:
                    await self._session.interrupt()
        elif isinstance(event, ToolResult):
            store.step(rid, "droid", f"{event.tool_name} returned" + (" an error" if event.is_error else ""),
                       {"content": _clip(event.content, 1500), "is_error": event.is_error})
        elif isinstance(event, ErrorEvent):
            store.step(rid, "droid", f"Droid error: {_clip(event.message, 200)}", {"type": str(event.error_type)})

    async def interrupt(self) -> None:
        if self._session is not None:
            await self._session.interrupt()


# ---------------------------------------------------------------- scripted (tests only)
@dataclass(frozen=True)
class ScriptCall:
    tool: str
    args: dict[str, Any]
    server: str = SERVER_NAME


class ScriptedRunner:
    """Replays tool calls through ToolGuard.on_permission and the real handlers, then proposes every layout the
    planner accepted. No model is involved."""

    name = "scripted"

    def __init__(self, calls: Sequence[ScriptCall] | Callable[[RunContext], Sequence[ScriptCall]],
                 plan_text: str = "Scripted plan: check the listed layouts.", recommend: str = "auto") -> None:
        self.calls = calls
        self.plan_text = plan_text
        self.recommend = recommend
        self.cancelled = False

    @staticmethod
    def _request(call: ScriptCall) -> PermissionRequest:
        use = ToolUseBlock(id=uuid.uuid4().hex, name=f"{call.server}___{call.tool}", input=call.args)
        action = McpToolAction(tool_use=use, tool_name=use.name, impact_level="low", server_name=call.server,
                               actual_tool_name=call.tool)
        return PermissionRequest(actions=[action], options=[
            PermissionOption("Allow once", ToolConfirmationOutcome.PROCEED_ONCE),
            PermissionOption("Cancel", ToolConfirmationOutcome.CANCEL)])

    async def run(self, ctx: RunContext, prompt: str) -> RunOutcome:
        calls = tuple(self.calls(ctx) if callable(self.calls) else self.calls)
        ctx.store.step(ctx.run_id, "session", "scripted runner (not Droid)", {"calls": len(calls)})
        ctx.guard.publish_plan(self.plan_text, "Scripted plan")  # the path a Droid spec plan takes
        handlers = {t.name: t.handler for t in build_tools(ctx)}
        for call in calls:
            if self.cancelled:
                return RunOutcome(None, error="cancelled")
            response = ctx.guard.on_permission(self._request(call))
            if response.selected_option is ToolConfirmationOutcome.CANCEL or call.tool not in handlers:
                continue
            await handlers[call.tool](call.args)
        current = layouts.stages(layouts.rows(ctx.current_plan, ctx.snap)) if ctx.current_plan else None
        accepted = sorted((v for v in ctx.verdicts if v.accepted), key=lambda v: v.stages == current)
        cands = [Candidate(label=v.label, stages=v.stages, rationale=v.reason[:600], plan_id=v.plan_id)
                 for v in accepted][:8]
        apply = bool(cands) and cands[0].stages != current and self.recommend != "keep"
        return RunOutcome(AutotuneProposal(
            summary=f"scripted: {len(accepted)} of {len(ctx.verdicts)} layouts accepted",
            recommendation="apply" if apply else "keep", recommended_index=0 if apply else None, candidates=cands))

    async def interrupt(self) -> None:
        self.cancelled = True


def sweep_calls(ctx: RunContext) -> list[ScriptCall]:
    """A fixed sweep for the scripted runner: the host alone, then each peer accelerator with 4 and with 7 layers.
    Deterministic, no model; only for rehearsing the page without Factory credits."""
    rows = layouts.rows(ctx.current_plan, ctx.snap) if ctx.current_plan else []
    n_layer = sum(r.layers for r in rows) or 36
    calls = [ScriptCall("pool_snapshot", {}), ScriptCall("current_layout", {}),
             ScriptCall("check_layout", {"label": "host alone", "stages": [
                 {"node_id": "local", "device": "CUDA0", "layers": n_layer}]})]
    for peer in ctx.snap.peers:
        accel = next((d.name for d in peer.devices if d.name != "CPU"), None)
        for k in (4, 7):
            if accel is not None and k < n_layer:
                calls.append(ScriptCall("check_layout", {"label": f"{peer.name} {accel} {k}", "stages": [
                    {"node_id": peer.id, "device": accel, "layers": k},
                    {"node_id": "local", "device": "CUDA0", "layers": n_layer - k}]}))
    return calls
