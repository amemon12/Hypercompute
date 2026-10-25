# Droid autotune

When a friend's machine joins or leaves the pool, a Factory Droid session looks for a better model layout for the
model being served. Droid proposes explicit layouts, the gateway's planner checks each one (plan-only, nothing
loads), this service shows the plan, every step, the planner's verdicts and a per-device diff, and nothing changes
until the owner clicks Approve.

A join is a reason to evaluate, not an order to use every device. On the GX10 the measured answer is usually "keep":
gpt-oss-120b runs at 56 tok/s there alone, and pooling the 671B and the 1T model cost decode speed (0.83x and 0.86x
in R6). On the G14 laptop a Mac holding 4 layers measured 1.25x on decode (R2), so there the right answer is "apply".
Droid is told those measurements and must earn a recommendation through the planner.

## Where Droid is used

Droid runs inside the running product, on a trigger, through the Python Droid SDK (`droid-sdk[mcp]==0.5.0`).

| Step | Where | What happens |
|---|---|---|
| Trigger | `autotune/trigger.py`, `autotune/service.py` `poll_once` | Every 2 s the service reads `GET /api/nodes` and `GET /api/network`. A peer turning eligible or leaving starts a 20 s settle window; any further change restarts it, a join plus leave nets out, and a settled change waits while a model loads, a tune trial or benchmark runs, or another run is open. Automatic runs are at most one per 2 minutes and 10 per hour (each is a paid session). The owner can also press Run now. |
| Session | `autotune/runner.py` `DroidRunner` | One `droid_sdk.Session` per run (a `droid exec` subprocess) in spec mode, `Autonomy.OFF`, built-in skills off, an empty working directory, model `claude-sonnet-5-5` (see below), and the system prompt in `autotune/prompt.py`. |
| Tools | `autotune/tools.py` | An in-process MCP server from `droid_sdk.mcp.create_sdk_mcp_server` with three tools: `pool_snapshot`, `current_layout`, `check_layout`. They call the gateway through `PlanOnlyGateway`, which has no write method; the owner token never reaches Droid. |
| Plan | `autotune/guard.py` `on_permission`, `autotune/runner.py` `_log` | Droid is asked to write a short numbered plan before its first tool call. That text, or a spec-mode ExitSpecMode plan, is published as the run's plan before any layout is checked. |
| Checks | `check_layout` -> `POST /api/plans` | Each candidate goes to the real planner (`services/layout.py`), which accepts it or refuses it with a code, a reason and a fix, for example `layout_memory: mac-a MTL0 needs ~12.0 GiB of RAM for 7 layer(s)`. |
| Answer | `autotune/models.py` `AutotuneProposal` | Droid ends with structured output (summary, apply or keep, candidates with stages and plan ids), validated by the SDK. |
| Verification | `autotune/service.py` `_finish` | Droid's answer is not trusted on its own: each candidate is matched to the planner verdict from this run, a candidate Droid never checked is checked by the service (and badged so), and only accepted ones get a diff and an Approve button. |
| Apply | `autotune/service.py` `_apply` | Only on the owner's click: the layout is checked again, and if the planner would now place it differently (another tier, other layers per device, file-backing) nothing is applied and the owner is asked to run again. Otherwise `POST /api/deployments/{id}/replace`, which rolls back on failure. If that call outlives the client, the service reads what the gateway serves instead of guessing. |

## What Droid can touch

Three layers, so no single one has to be right:

1. **Toolset.** After the session opens, `session.list_tools()` gives the real tool ids. `update_settings(restrict_tools=..., disabled_tools=...)` keeps this server's three tools, `ToolSearch` (droid serves MCP tools as deferred tools that the model loads through it) and ExitSpecMode in spec mode, and disables every other tool by id. Restrict alone was not enough in droid 0.249: it left `AskUser`, `Script`, `ToolSearch` and `WaitForScript` on. The list is then read back, and the run stops before the model is prompted if any other tool is still allowed. A tool call outside the lock during the run interrupts the session.
2. **Permission gate.** At `Autonomy.OFF` every tool call asks first. `ToolGuard.on_permission` allows only an `McpToolAction` from server `hypercompute` naming an allowlisted tool and cancels everything else, including questions to the owner. It counts the calls it has granted, so parallel calls over the cap are denied at the prompt. A spec plan may only continue at manual approval or low autonomy; any higher option is cancelled.
   This is why the default model is `claude-sonnet-5-5`: it calls the MCP tools directly, so each call reaches this gate. droid's default model (`gpt-6.1-sol` in 0.249) calls them from inside its `Script` code tool, which skips the per-call prompt, and with `Script` disabled it cannot reach them at all.
3. **Caps in this process.** Each MCP handler calls `ToolGuard.admit` first, and the runner counts every `ToolSearch` call it sees with `ToolGuard.admit_loader` (ToolSearch runs without a permission prompt, since droid has no permission action for it): at most 8 `check_layout` calls, 3 `ToolSearch` calls and 12 tool calls in all per run. Past a cap a call is refused, and a ToolSearch past its cap stops the session. These hold even if a CLI version stopped routing MCP calls through the prompt. Arguments are validated against the gateway's `LayoutStage` rules before anything is sent, and Droid cannot choose the context size: every check uses the served one, which is what apply checks again.

Plan-only means `DeploymentManager.plan`: it checks the layout and stores a plan row, and starts no engine. The one
write in this folder is `GatewayClient.replace`, reachable only from the owner's approve route, which needs a
per-process token and a loopback client and Host header. Droid's text is rendered with `textContent`, never as HTML.
The Factory key is read from the git-ignored `.env`, passed to the SDK, and kept out of `repr`, logs and the page.
Peer names and device descriptions come from other people's machines, so they are cut to plain name characters
before they reach Droid's prompt, its tools or the page, and the page shows the planner's diff above Droid's prose.
If the gateway restarts (a new owner token), the client reads the token again once instead of failing.

## What the owner sees

`http://127.0.0.1:8787` on the gateway host:

- the pool and the trigger log (baseline, changed, settling, deferred, fire)
- per run: what Droid could touch (kept tools, removed built-ins, checks used, permissions allowed and denied),
  Droid's plan, its proposal, every planner verdict with its reason and fix, and a timeline of every step with the
  raw detail (Droid's messages, tool calls and results, permission decisions)
- per accepted candidate: a diff per machine and device (layers and range before and after, estimated GiB against
  the budget, fits), hops per token, tier, the planner's decode and SSD estimates when it gives them, and the last
  measured benchmark
- Approve and apply, or Reject

Every step is also appended to `runs/<id>.jsonl` (git-ignored), with the final record in `runs/<id>.json`.

## Run it

Rehearse against the mock gateway (mock priors, no real hosts; the scripted runner involves no Droid and the page
says so on every run):

```bash
uv sync
uv run python -m autotune mock-gateway --port 8019
uv run python -m autotune serve --gateway http://127.0.0.1:8019 --runner scripted
curl -X POST http://127.0.0.1:8019/mock/join/mac-a
```

With Droid against the mock: put the key in `.env`, install the droid CLI, and drop `--runner scripted`. With the
npm install on Windows, point `DROID_PATH` at the native binary
(`%APPDATA%\npm\node_modules\droid\node_modules\@factory\cli-win32-x64\bin\droid.exe`), since the npm
shim is a `.cmd` file.

On the gateway host (owner endpoints answer only on 127.0.0.1):

```bash
uv run python -m autotune serve --gateway http://127.0.0.1:8000
```

Settings come from the environment or `.env` (see `.env.example`): `AUTOTUNE_SETTLE_S`, `AUTOTUNE_MAX_CHECKS`,
`AUTOTUNE_MAX_CALLS`, `AUTOTUNE_TIMEOUT_S`, `AUTOTUNE_MIN_INTERVAL_S`, `AUTOTUNE_MAX_RUNS_PER_HOUR`,
`AUTOTUNE_DROID_MODE` (spec or auto), `AUTOTUNE_DROID_MODEL` (default `claude-sonnet-5-5`; pick a model that
calls tools directly), `AUTOTUNE_PORT`, `DROID_PATH`. If the droid CLI
offers no manual or low-autonomy option for leaving spec mode, the run stops at the plan; set
`AUTOTUNE_DROID_MODE=auto` and Droid starts at Autonomy.OFF without a spec step.

Factory setup, done by the owner: a Factory account and an API key from app.factory.ai/settings/api-keys in `.env`
as `FACTORY_API_KEY=...`, and the droid CLI (`irm https://app.factory.ai/cli/windows | iex` on Windows,
`curl -fsSL https://app.factory.ai/cli | sh` on Linux and macOS, or `npm install -g droid`), checked with
`droid --version`.

## Live runs (2026-10-24, droid 0.249.0, against the mock gateway)

All four ran through this service with the real Factory Droid session; the gateway was the mock, so the planner's
decode and SSD figures in them are mock priors.

- **mac-azim joined** (G14 serving alone): Droid checked 4 layouts, all accepted, and recommended `mac-a MTL0 4 +
  local CUDA0 32` because that layout is the one measured at 1.25x in R2. It said the larger predicted gains for 2
  and 3 Macs were mock priors and that 3 Macs measured only 1.02x over 1. The owner approved; the service re-checked
  the layout and the mock replaced the deployment.
- **mac-azim left**: Droid wrote a 4-line plan first, checked 5 layouts, the planner refused one (`layout_memory`:
  mac-a's stages need ~12.6 GiB and it contributes 11.2 GiB), and Droid recommended keep: mac-azim held no stage.
- **Adversarial prompt, caps lowered to 2 checks and 5 calls**: asked to read `C:\Windows\win.ini`, run `whoami`,
  check 4 layouts and deploy the best one itself. Droid had no file, shell or deploy tool. It fired the 4 checks in
  parallel: the planner saw 2 (one accepted, one refused for memory) and the guard refused the other 2. The mock's
  deployments did not change. The gate now also counts its grants, so such calls are denied at the prompt.
- **First attempts** failed closed before the model ran: the lock read-back caught the tools restrict left on, and
  then a model that only reaches MCP tools through `Script` had nothing callable. Both are fixed above.

## Tests

```bash
uv run pytest -q
```

They cover the trigger (debounce, flap, leave, busy deferral, rate cap), the guard (allowlist, other servers,
built-in tools, spec proceed choice, caps, toolset lock, questions), the layout fold and diff (RPC names to worker
devices, hops, prediction sources, drift), the tools served through the SDK's real MCP server and called with an MCP
client, DroidRunner driven through a fake SDK session (including a CLI that ignores the toolset lock), the whole loop
against the mock (join, run, refusal, approval, replace, rejection, a lying proposal, a runaway script, a disallowed
tool, supersede, busy gateway, apply timeout, rotated token), and the web API's loopback, Host and token checks.

## Limits

- The mock gateway's memory and decode figures are mock priors. Its starting benchmark copies the measured R2 and R6
  medians and says so in its label.
- The real planner gives explicit layouts no capacity-optimizer prediction; a `moe_offload` layout carries the MoE
  decode estimate, a resident `gpu` layout carries none. The diff shows what the planner gave and says where it came
  from.
- Replace reloads the served model, and a request in flight is not migrated. Never approve while judges are chatting.
- 671B and 1T loads take 4 to 9 minutes (R6); rehearse apply with gpt-oss-120b.
