# Cal Hacks 13.0: Best Use of the Factory Droid SDK (Factory). Prize is $2000 cash, Factory credits and swag. Eligibility: the Droid SDK (TypeScript or Python) has to do real work in the running product, and the README has to say where and how Droid is used. What scores well: Droid runs on a trigger you designed, there is careful control over what Droid can touch, and its plans, steps, diffs and results are visible. What scores poorly: a chatbot wrapper, demos that only work in slides, and projects where nobody can tell what the agent did.

Fit 7/10, prize about $2000, effort about 6 h (estimate).

## Integration

VERDICT: real fit, with one weak spot. The gateway already has a deterministic replan controller (services/replan.py, which follows node.verified/node.state/node.left) and a measured tuner (services/tuner.py). So Droid must not re-decide what those two already decide. The real gap is the hand-done explicit-layout search. docs/research/deepseek-671b-layouts.md has hand-made candidates A/B/C with a "check before loading" list, and the CHECKPOINT-2026-10-24 TODO lists "fewer-hop pooled configs". PlanRequest.layout (schemas.py:470-496) plus DeploymentManager.plan (deployments.py:381-391) is already a plan-only verifier: it checks an explicit layout and never loads it. So Droid proposes layouts, the planner checks them, a short benchmark decides, and the owner applies. Droid never applies anything itself.

SMALLEST REAL INTEGRATION (Python SDK, because the backend is Python):
1) New file backend/hypercompute/services/droid_autotune.py (about 300 lines), class DroidAutotune.
   - Build it in coordinator.py with a _build_droid(), copying the optional-import pattern of _build_tuner (coordinator.py:180).
   - It is enabled only when HYPERCOMPUTE_DROID=1, `import droid_sdk` succeeds, and the droid binary is on PATH.
   - Start it in the app.py lifespan next to coord.replan.run() (app.py:358).
2) TRIGGER. run() follows EventBus.stream() exactly as ReplanController.run does (replan.py:256). It reacts to node.verified and node.left.
   - It waits 20 s for things to settle and runs one session at a time.
   - It skips while a tune trial or a deployment.replacing is in progress.
   - The owner can also trigger it with POST /api/autotune/trigger.
3) DROID SESSION. droid_sdk.Session with:
   - cwd = HYPERCOMPUTE_HOME/droid/<run_id>
   - SessionConfig: mode=Mode.SPEC, autonomy=Autonomy.OFF, mcp_servers=[hc], restrict_tools limited to the hc tools, and the built-in Execute/Edit/Create disabled
   - a system_prompt that carries the R6 numbers and the layout rules from layout.py's docstring
   - InteractionHandlers(on_permission=gate, on_question=cancel)
   - session.stream(prompt, output=AutotuneProposal (a pydantic model), timeout=300)
4) IN-PROCESS MCP SERVER. create_sdk_mcp_server("hypercompute", ...) from droid-sdk[mcp]. Its tools call the gateway's services directly, so nothing goes through HTTP and there is no owner token in the agent.
   - pool_snapshot(): nodes.views(), device budgets, Iroh direct/relay state and RTT.
   - current_layout(): the active deployment's plan assignments folded into LayoutStage stages, plus its prediction and last benchmark.
   - check_layout(stages, ctx_size): asyncio.to_thread(deployments.plan, PlanRequest(model_id, layout=stages)). Returns feasible, per-stage fits, bytes vs budget, the prediction and plan_id. It is plan-only and capped at 8 calls per run.
   - short_benchmark(plan_id): gated, see 5.
5) SAFETY GATE (on_permission):
   - It allows only McpToolAction with server_name == "hypercompute" and actual_tool_name on the allowlist (the SDK's McpToolAction carries both). Everything else gets CANCEL.
   - The spec-mode ExitSpecMode request: request.plan.text is published as a droid.plan event, then the turn proceeds.
   - short_benchmark waits up to 120 s for the owner's POST /api/autotune/{id}/approve, else CANCEL.
   - It refuses when the predicted load is over HYPERCOMPUTE_DROID_BENCH_MAX_LOAD_S (default 180 s). That keeps 671B and Kimi plan-only.
   - When approved, it benchmarks the current deployment (BenchmarkService, short prompt, 1 rep, 64 tokens), then calls DeploymentManager.replace(dep, plan), which already rolls back on failure (deployments.py:731). It benchmarks again and swaps back if the new layout is slower.
   - Droid has no tool that stops a deployment or applies a final replace.
6) VISIBILITY.
   - Each AssistantMessage, ToolCall and ToolResult becomes events.publish("droid.step", run_id=..., tool=..., summary=...). It then shows in the existing ActivityFeed SSE and in the events JSONL audit.
   - The run record (plan text, steps, proposal, diff) goes to Store.
   - A deterministic layout_diff(current, proposed) in the same module shows, per node and device: layers before and after, hops added or removed, bytes vs budget, predicted decode ms, and measured tok/s.
7) ROUTES in app.py:
   - GET /api/autotune (OwnerRead)
   - POST /api/autotune/trigger (OwnerWrite)
   - POST /api/autotune/{id}/approve (OwnerWrite)
   - DELETE /api/autotune/{id}, which calls session.interrupt() (OwnerWrite)
8) FRONTEND: frontend/src/components/AutotunePanel.tsx, plus types and a client in src/api. It shows the plan text, the steps with their state, and the diff table. Its Apply button calls the existing POST /api/deployments/{id}/replace with plan_id, so a human applies.
9) DEPENDENCIES AND TESTS:
   - pyproject optional extra: droid = ["droid-sdk[mcp]==0.5.0"].
   - tests/test_droid_autotune.py, using fakes with no network: gate allow/deny, the check_layout cap, layout_diff, the bench refusal over the load budget, and the module disabled when the SDK is missing.
   - A README section "Where Droid is used".

DATA FLOW: peer join/leave -> EventBus -> DroidAutotune waits 20 s -> droid CLI subprocess (Factory model) -> MCP calls over local HTTP back into in-process planner/benchmark services -> planner verdicts -> structured AutotuneProposal -> diff -> SSE -> dashboard -> owner Apply -> existing replace route.

## Demo moment

At the table, a MacBook runs `hypercompute join`. Within about 20 s the Autotune card wakes on the node.verified event.

1. It shows Droid's spec-mode plan, for example "try 3 layouts: GX10 alone, GX10 + mac-a Metal span, fewer-hop G14 + mac-a".
2. Three check_layout calls stream in with the planner's verdicts. One is refused with a byte-level reason, for example a Mac stage over its budget.
3. The owner clicks "Approve short bench". A 1-rep, 64-token bench runs on gpt-oss-120b, which loads in 38-50 s on the GX10 alone and 72-167 s pooled (R2/R6).
4. The diff appears. On the GX10 the honest likely result is "keep GX10 alone: the Mac adds a hop, predicted under 1x decode". R6 measured pooled decode at 0.83x on the 671B and 0.86x on Kimi.
5. The line to say: "a join is a reason to evaluate, not an order to use every device."

For a "yes, apply" moment, use the R2 setup instead (G14 host plus 1 Mac, 1.25x decode on 120B). There the diff shows 4 layers moving to MTL0 and the owner clicks Apply, which calls the existing replace route with rollback. Have a recorded clip as fallback.

## Accounts and keys (the user creates these)

The user does all of this; Claude never signs up or handles keys.
- A Factory account at app.factory.ai. Ask the Factory booth for hackathon credits.
- A Factory API key from https://app.factory.ai/settings/api-keys, exported as FACTORY_API_KEY in the host's environment.
  - On the GX10, put it in an env file outside the repo, for example ~/.hypercompute/droid.env with chmod 600, sourced by host.sh.
  - Never put it in the repo, the log or the invite output.
- The droid CLI installed by the user on the host machine (the GX10: glibc Ubuntu, arm64 binary per Factory's download paths). The documented command is `curl -fsSL https://app.factory.ai/cli | sh`, which installs to ~/.local/bin/droid. Verify that `droid --version` runs on aarch64.
- `uv pip install "droid-sdk[mcp]==0.5.0"` in the host venv.
- HYPERCOMPUTE_DROID=1 in host.sh.
- Opt into the Factory prize on Devpost.

The agent must run on the host machine, because owner endpoints accept only 127.0.0.1. That is fine here, since the MCP tools are in-process.

## Risks

1) Overlap with what exists. The deterministic replan controller and measured tuner already evaluate joins and leaves, so a judge may see an LLM around an optimizer. Mitigation: the pitch says Droid automates the layout search that was done by hand (deepseek-671b-layouts.md A/B/C, and the fewer-hop TODO), with the planner as verifier and the benchmark as judge. The predictions in that hand-made doc (B/C beat A) were later contradicted by measurement, which shows why the benchmark step matters.
2) Live benchmarks are only possible on small or mid models. 671B and Kimi loads take 239-517 s, and a cold start takes 17-32 min (R6). The load budget gate keeps them plan-only. Rehearse with gpt-oss-120b.
3) A benchmark by replace reloads the served model. Generation in flight is not migrated (replace docstring). Never auto-apply, and only on an owner click. Do not trigger it while judges are chatting.
4) The likely GX10 answer is "keep", which looks less exciting. Prepare the R2-style pooled-helps case or a recorded clip.
5) SDK unknowns to verify at runtime:
   - the exact MCP tool ids for restrict_tools (check with session.list_tools())
   - that MCP calls go through on_permission at Autonomy.OFF (the SDK example suggests yes)
   - the SDK MCP server runs a local Starlette HTTP server with a bearer token, so keep it on loopback
   - possible version conflicts between mcp/starlette and FastAPI
   - droid-sdk is 0.x (0.5.0, 2026-09-24) and the TS README warns of an API overhaul
6) The droid CLI on arm64 Linux is supported according to the docs' arch list but not tested on the GX10. It needs outbound internet and adds 10-60 s of model latency per run, plus Factory credit cost.
7) Time. About 6 h out of the roughly 15 left, competing with overnight benchmarks, pitch prep and other tracks. A minimal 4 h cut skips the approval UI and auto-approves the spec plan.
8) This subagent was read-only. Nothing is implemented, and the main session must write the code, tests and README section.
9) LLM layouts will often be infeasible. The 8-call cap and visible planner refusals handle that, and the refusals are part of the story.

## Sources

- https://hive.hackberkeley.org/prizes
- https://docs.factory.com/sdk/python
- https://docs.factory.com/sdk/typescript
- https://docs.factory.com/cli/droid-exec/overview
- https://pypi.org/pypi/droid-sdk/json
- https://github.com/Factory-AI/droid-sdk-python/blob/main/examples/sdk_mcp.py
- https://github.com/Factory-AI/droid-sdk-python/blob/main/src/droid_sdk/_high_level/config.py
- https://github.com/Factory-AI/droid-sdk-python/blob/main/src/droid_sdk/_high_level/interactions.py
- https://github.com/Factory-AI/droid-sdk-python/blob/main/src/droid_sdk/_high_level/enums.py
- https://docs.factory.ai/droid-cli/cli-reference.md
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\replan.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\tuner.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\layout.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\deployments.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\events.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\schemas.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\app.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\coordinator.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\research\deepseek-671b-layouts.md
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\research\PAPER-RUNS-2026-10-22.md
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\research\CHECKPOINT-2026-10-24.md
