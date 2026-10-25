# Demo app: the surfaces it calls

What the one-click demo UI must call on this branch (codex/iroh-transport), with file:line for each. Paths are
relative to the repo root. Line numbers were read on 2026-10-25 at HEAD 0a842e5.

The demo app is a small local server (stdlib + httpx + the repo's own packages) plus static HTML/CSS/JS. The browser
never talks to the gateway directly: the gateway sends no CORS headers (there is no CORSMiddleware in
`backend/hypercompute/app.py`), so a page on another port cannot read its replies. The demo server proxies through
`hypercompute_sdk.Pool` and, for the few routes the SDK does not wrap, plain `httpx` with `trust_env=False`.

## 0. Ground rules the code already enforces

| Rule | Where |
|---|---|
| Owner routes answer only a loopback client AND a loopback `Host` header (`127.0.0.1`, `localhost`, `[::1]`), else 403 `owner_only` | `backend/hypercompute/app.py:65-66`, `:386-391` |
| Owner writes also need `X-Hypercompute-Owner: <owner_token>` (constant-time compare) | `backend/hypercompute/app.py:393-396`, `backend/hypercompute/config.py:30` |
| The owner token comes from `GET /api/owner/bootstrap` (loopback only) | `backend/hypercompute/app.py:446-450` |
| The SDK refuses any base URL that is not `http://` on a loopback name | `sdk/python/src/hypercompute_sdk/client.py:88-101` |
| Ports: the CLI defaults to 8000, but the SDK, the one-click scripts and `seed_and_run.py` default to 8001 | `backend/hypercompute/config.py:9`; `sdk/python/src/hypercompute_sdk/client.py:34`; `join/lib/oneclick.py:36`; `sdk/python/examples/seed_and_run.py:28` |

So the demo must always pass the port explicitly (`--port`, `HC_PORT`, `Pool(base_url)`), and read it from one
config value. Tests must use fakes and never touch 127.0.0.1:8001 or a LAN address.

## 1. Host computer

### 1.1 Start hosting

CLI: `python -m hypercompute.cli host --transport iroh --port <P> --no-browser` (the console script is
`hypercompute = "hypercompute.cli:main"`, `pyproject.toml:17-18`).

| Flag | Default | Line |
|---|---|---|
| `--port` | 8000 | `backend/hypercompute/cli.py:557` |
| `--bind` (LAN only, refused with iroh) | auto | `:558`, refusal `:199-200` |
| `--no-browser` (else it opens the old dashboard) | off | `:559`, `:225-226` |
| `--transport lan\|iroh` (the demo uses `iroh`) | `lan` | `:560` |
| `--relay-url` (iroh only) | public relays | `:562`, refusal `:201-202` |
| `--local-max-memory 8GiB` (budget for the host's own GPU) | none | `:563` |
| `--local-devices CUDA0` | all | `:564` |

Code path: `cmd_host` takes `~/.hypercompute/coordinator/host.lock` (one host per machine, `InstanceLocked`
otherwise) `cli.py:184-190`; `_run_host` `cli.py:193-229`; `instance.acquire` `backend/hypercompute/agent/instance.py:28-35`.

What it prints (stdout, in order):

1. `Hypercompute <ver>: starting host on port <P> (this can take ~10 s: GPU probe)` `cli.py:205`
2. `Dashboard (this machine only): http://127.0.0.1:<P>/` `cli.py:215`
3. `[<level>] <diagnostic>` and `[startup] <note>` lines `cli.py:221-224`
4. From the gateway lifespan, with iroh: `Iroh ready. Send this one-use invitation to your friend:`, then the
   FULL join command with ticket and code, then `Expires: <iso>; renew with: ...` `backend/hypercompute/app.py:354-356`

Line 4 is a secret. The demo must never capture host stdout into a log or the UI unredacted. Two safe ways:

- Start it through the one-click supervisor, which redacts every line before it reaches the log:
  `python join/lib/oneclick.py _supervise --log <LOG> --pidfile <PID> -- python -m hypercompute.cli host ...`
  `join/lib/oneclick_supervise.py:101-132` (redactor `join/lib/oneclick_invite.py:94-108`, entry `:135-144`,
  dispatch `join/lib/oneclick.py:506-507`). It also keeps the machine awake and writes
  `{"supervisor_pid", "child_pid", "started_at"}` to the pidfile `oneclick_supervise.py:70-75`.
- Or `stdout=DEVNULL` and rely only on HTTP for state.

Whole one-click host action, for reference: `oneclick.py host` builds exactly the command above
(`host_command`, `join/lib/oneclick.py:352-364`, honours `HC_LOCAL_MAX_MEMORY`, `HC_LOCAL_DEVICES`), checks the
engine, launches under the supervisor, waits up to `WAIT_S = 120` s for health, then calls `ctx.open_url` and
`run_invite` (`oneclick.py:367-390`). If the demo reuses it in-process, pass a `Ctx` with `open_url` set to a no-op
or it opens the old dashboard.

### 1.2 Tell that the host is up

- `GET http://127.0.0.1:<P>/api/health` answers 200 `{"status": "ok", "version", "time", "engine": {...}}`
  `backend/hypercompute/app.py:435-444`. The one-click check uses a 3 s timeout, `trust_env=False`
  (`join/lib/oneclick_core.py:110-116`).
- With iroh, a 200 also means the Iroh endpoint is up: `HostService.start` runs inside the lifespan before the server
  accepts requests, and a failure there ends the process (`app.py:340-362`).
- `Pool.health()` `sdk/python/src/hypercompute_sdk/client.py:415-416`; `Pool.status()` `:446-456` returns
  `{version, engine_release, transport, connections, nodes, deployment, ready_model}` with Iroh ids stripped
  (`_public_peer` `:253-258`).
- Already running: the one-click host reuses a live host (`oneclick.py:376-379`). A second `host` fails with
  `error: a Hypercompute host is already running on this machine (pid N)` (`instance.py:32-33`).

### 1.3 Stop hosting

There is no stop command. `join/README.md:79` says so. The demo must stop the process it started: first
`Pool.stop()` for any deployment (section 3.7), then end the supervised child (`child_pid` from the pidfile). The
lifespan cleanup that stops deployments, closes Iroh and stops workers is `app.py:363-376`; it only runs on a
graceful shutdown (Ctrl+C / SIGINT on POSIX). On Windows a hard terminate skips it. Treat graceful Windows stop as
unverified.

### 1.4 Mint an invite

Each invite is one-use, expires in 15 minutes (`backend/hypercompute/config.py:19`), and minting a new one does not
cancel older ones (`backend/hypercompute/services/sessions.py:106-123`). The invite printed at startup (1.1 line 4)
is lost if stdout is discarded, so mint a fresh one for the UI.

HTTP (owner write): `POST /api/network/invite` with `X-Hypercompute-Owner` `app.py:457-459`, built by
`network_invite` `app.py:312-321` and `transport_invite` `app.py:302-310`. Reply (all secret except `expires_at`):

```json
{"code": "ABCDE-FGH23", "expires_at": "2026-10-25T12:00:00Z",
 "join_address": "<ticket>", "join_command": "uv run hypercompute join --ticket <ticket> --code ABCDE-FGH23",
 "ticket": "<ticket>"}
```

Schema `InviteView` `backend/hypercompute/schemas.py:370-374`. Code alphabet has no 0, O, 1 or I, shape
`XXXXX-XXXXX` (`sessions.py:36-41`, `join/lib/oneclick_invite.py:9-14`). The SDK does not wrap this route
(`ROUTES` `client.py:64-80`).

CLI: `python -m hypercompute.cli invite --port <P>` `cli.py:520-536`, parser `:596-598`. Stdout is exactly two lines:

```
uv run hypercompute join --ticket <ticket> --code <code>
Expires: <iso8601>
```

On failure: `error: Could not create an invitation: ...` and `fix:   start this version with ...` (exit 2,
`_fail` `cli.py:47-51`).

One-click: `oneclick.py invite` captures that output, never prints it, copies the `--ticket` line to the host
clipboard with `clip` / `pbcopy` / `wl-copy` / `xclip` / `xsel` (fed on stdin), or writes
`~/.hypercompute/oneclick/invite.txt` (0600) when no tool works, and prints only
`Invite copied. ... expires at HH:MM (15 minutes).` and `Ticket ...<last4>, code hidden.`
(`run_invite` `join/lib/oneclick.py:425-457`, `clipboard_tools` `:393-405`).

For "show an invite to copy" without embedding it in the page: the button calls the demo server, the server mints
and copies to the host clipboard (reuse `run_invite`, or `Runner.feed` `oneclick_core.py:73-81`), and the UI shows
only the last 4 ticket characters and the expiry. If the invite must reach the browser clipboard instead, return it
only in the response to that one click and write it with `navigator.clipboard.writeText` without putting it in the
DOM or any log.

### 1.5 Live peers joining

- `Pool.nodes()` `client.py:418-420` (`GET /api/nodes`, owner read, no token, `app.py:486-488`) returns every
  `NodeView` including the host (`is_local: true`, id `"local"`). `NodeView` `schemas.py:235-253`. Fields for the UI:
  `display_name`, `state` (`verifying`, `eligible`, `ineligible`, `stale`, `left`, `removed`; `schemas.py:209-215`),
  `state_reason`, `heartbeat_age_s`, `inventory.os.system`, `inventory.devices[]` (`id`, `name`, `kind`, `backend`,
  `total_bytes`, `free_bytes`; `schemas.py:73-84`), `inventory.memory_pools[]` (`kind` `system_ram|unified|discrete_vram`,
  `total_bytes`, `free_bytes`; `schemas.py:61-70`), `contribution.devices`, `contribution.max_bytes`
  (`schemas.py:147-154`), `verification.rpc_devices`. Do not show `inventory.network` or `worker` (LAN IPs and ports).
- `Pool.peers()` `client.py:422-424` (`GET /api/peers`, `app.py:524-527`) is remote peers only, per row
  (`backend/hypercompute/services/peer_seed.py:546-561`): `node_id`, `display_name`, `state`, `worker_state`,
  `contribution_devices`, `rpc_devices`, `deployments`, `managed`, `report` (`ManagedReport`, `schemas.py:334-353`,
  with `report.seed` progress), `reported_at`, `pending`, `history`, `seed_request`.
- Push instead of poll: `GET /api/events?since=<seq>` (owner read) is SSE, `id: <seq>`, `event: hc`,
  `data: <Event json>`, plus `: keep-alive` comments; resumes from `Last-Event-ID` (`app.py:656-669`). `Event` is
  `{seq, at, type, message, data}` (`schemas.py:833-838`). Useful types: `node.joined` (data `node_id`, `os`,
  `services/nodes.py:213-214`), `node.verified` (`:439`), `node.left` (`:456`), `node.state`,
  `deployment.state` (data `deployment_id`, `state`, `services/deployments.py:364-366`), `deployment.progress`
  (`:369-373`), `invite.created` (data `session_id`, `expires_at` only, `services/sessions.py:121-122`). Not wrapped by
  the SDK; the demo server relays it. Iroh path per peer (relay or direct, RTT) is `GET /api/network`
  `peers[].path.mode` / `rtt_ms` (`app.py:452-455`, `backend/hypercompute/transport/host.py:91-97`,
  `backend/hypercompute/transport/iroh.py:350-360`); `Pool.status()["connections"]` gives it without ids.

## 2. Friend computer

### 2.1 Join non-interactively (one-click helper)

Command (repo venv python, cwd = repo): `python join/lib/oneclick.py join --repo <repo> [--port <P>]`
(`join/lib/oneclick.py:481-488`, wrappers `join/hypercompute-join.ps1`, `.sh`). Extra positional words are refused
so a pasted invite is never echoed (`:508-511`).

Environment it reads (each one replaces a prompt; `ask` `join/lib/oneclick_core.py:186-197`):

| Variable | Meaning | Line |
|---|---|---|
| `HC_INVITE` | the full invite line or the two bare tokens; read once, never echoed, removed from the child env | `oneclick.py:52-72`, parse `oneclick_invite.py:55-82`, strip `oneclick.py:249` |
| `HC_ONECLICK_YES=1` | answers the final "Share this computer now?" | `oneclick.py:201-204` |
| `HC_NAME` | name shown to the host, 1 to 64 chars | `oneclick_core.py:234-243` |
| `HC_DEVICE` | override device (must be an engine device or `CPU`) | `oneclick_plan.py:107-129` |
| `HC_MAX_MEMORY` | GiB the host may plan onto this machine (whole number) | `oneclick_plan.py:146` |
| `HC_RESERVE` | GiB kept free | `oneclick_plan.py:150-152` |
| `HC_CACHE` | 1/0, keep received tensors on disk (default yes) | `oneclick_plan.py:214-218` |
| `HC_MANAGED=1` | let the host restart the worker and seed this machine's share from Hugging Face | `oneclick_plan.py:159-175` |
| `HC_MAX_DISK` | GiB the host may fill when managed | `oneclick_plan.py:165-167` |
| `HC_RELAY_URL` | custom https relay | `oneclick_plan.py:178-185` |
| `HC_PORT` | port for the status checks (default 8001) | `oneclick.py:491-494` |
| `HYPERCOMPUTE_HOME` | state root (default `~/.hypercompute`) | `oneclick.py:497-498` |

With no terminal (stdin closed) every prompt without a variable takes its default, because `ask` maps EOF to the
default (`oneclick_core.py:192-196`). That means managed is off, cache is on, device and memory come from the
inventory (`default_cap_bytes`: 80% of VRAM, 65% of unified memory minus 6 GiB, 40% of RAM minus 8 GiB;
`oneclick_plan.py:59-65`).

Flow (`run_join` `oneclick.py:188-215`):

1. `status --port P`; if it already shares, it asks to stop (default no, so it ends with
   `Still sharing. Nothing was changed.`, exit 0) `:101-114`.
2. `engine status`; if not ready it runs `engine install` attached to the terminal, which asks before each download
   `:133-150`. The demo must check `engine status` first and show a message instead, never trigger the install.
3. `doctor`, information only `:153-163`.
4. invite from `HC_INVITE` `:52-72` (bad invite: `Stop`; a LAN `--host` invite is refused `:62-64`).
5. `inventory` `:166-185`, then device, memory, name, cache, managed `oneclick_plan.py:221-234`. An NVIDIA box whose
   engine cannot see the GPU asks "Share the CPU instead?" and stops on the default no `oneclick_plan.py:118-127`.
6. consent block printed `oneclick.py:75-98` (lines like `Device:`, `Memory the host may use:`).
7. `HC_ONECLICK_YES=1`, then `launch` starts the supervisor in the background with the ticket and code only in its
   env (`HC_ONECLICK_TICKET`, `HC_ONECLICK_CODE`) `:241-255`; the composed CLI command carries `<ticket>` / `<code>`
   placeholders (`compose_join` `oneclick_plan.py:199-211`) that only the supervisor fills
   (`oneclick_supervise.py:27-41`).
8. `watch` waits up to 120 s for a log line starting `Joined.` (`oneclick.py:275-296`). Outcomes: joined (prints
   status, then `Sharing. You can close this window. To stop, open hypercompute-leave.`, exit 0); timeout
   (`Still connecting. ...`, exit 0); `error:` line or child exit (prints `error:` / `fix:` or the last 15 log lines,
   exit 1).

All output goes through `ctx.say`, which redacts tickets, any 30+ char token and code-shaped strings
(`oneclick_core.py:168-169`, `oneclick_invite.py:94-108`). Logs: `~/.hypercompute/oneclick/join.log` (rotated to
`join.log.1`) and `join.pid` (`oneclick.py:241-255`, `state_dir` `oneclick_core.py:177-179`).

In-process reuse: `Ctx` is a frozen dataclass (`oneclick_core.py:149-183`). The demo can build one with
`env={..., "HC_INVITE": invite, "HC_ONECLICK_YES": "1"}` (no change to `os.environ`), `out=<collector>`,
`read_line` raising `EOFError`, `open_url` a no-op, and call `run_join(ctx)` in a worker thread (it blocks up to
`WAIT_S`). Tests can inject `StubRunner` / `FakeProc` from `join/tests/oneclick_testkit.py:41-79` and
`make_ctx` `:91-110`, with inventories in `join/tests/fixtures/`.

### 2.2 Join with the CLI directly (what the helper ends up running)

`python -m hypercompute.cli join --ticket <T> --code <C> --name <N> --devices <D> --max-memory <N>GiB
--reserve <R>GiB [--cache] --yes [--managed --devices-allowed <D> --max-disk <N>GiB] [--relay-url <url>]`

Parser `backend/hypercompute/cli.py:567-590`: `--host` / `--ticket` mutually exclusive and required `:568-570`,
`--code` required `:572`, `--name` `:573`, `--max-memory` `:574`, `--reserve` `:575`, `--devices` `:576`,
`--worker-port` `:577`, `--cache` `:579`, `--yes` `:580` (without it and without a tty: `error: confirmation needed`,
`backend/hypercompute/agent/join.py:818-820`), `--managed` `:581`, `--devices-allowed` `:584`, `--max-disk` `:586`.
Code: `cmd_join` (one agent per machine, `agent.lock`) `cli.py:235-241`, Iroh path `_run_peer_join` `:267-286`,
`_run_join_on_host` `:289-333`, `run_join` `agent/join.py:816-845`. The success line is
`Joined. The host is verifying that it can reach this worker...` `agent/join.py:833`; the agent then serves its local
status page `Local status and Stop button: http://127.0.0.1:<port>/` `agent/join.py:686`.

Passing the ticket as an argument exposes it in the process list; the helper accepts that and says so
(`oneclick.py:41-42`). The demo should go through the helper rather than build this argv itself.

### 2.3 Status on the friend computer

- CLI `python -m hypercompute.cli status --port <P>` `cli.py:430-468`, parser `:592-594`. When sharing, the first
  line is `Sharing as a friend node:` (`:435`), then `  key: value` lines. With no host and no agent:
  `Not sharing and no host running on this machine.` (`:445`). The helper's own check is
  `stdout.startswith("Sharing as a friend node:")` (`oneclick.py:107`).
- JSON instead of text: read `control_port` from `~/.hypercompute/agent/state.json` (`agent/join.py:89-104`) and
  `GET http://127.0.0.1:<control_port>/status` (loopback only, no token; `backend/hypercompute/agent/control.py:61-76`).
  Keys (`status_doc` `agent/join.py:663-675`): `host`, `state` (`verifying`, `eligible`, ...), `reason`, `worker`,
  `worker_cpu_seconds`, `sharing` (devices), `budget`, `deployments_using_this_machine`, and with Iroh `transport`,
  `connected`, `peer_id`, `path` (`mode` relay/direct, `rtt_ms`, `remote_id`), `rpc`
  (`backend/hypercompute/transport/client.py:243-254`), plus `managed_by_host` when managed.
- `state.json` also holds the node `credential`, the `control_token` and, with Iroh, the `ticket`
  (`agent/join.py:495-497`, `:685`). The demo may read `control_port` only. Never forward the file, and drop `peer_id`
  and `path.remote_id` as the SDK does.

### 2.4 Stop sharing

- CLI `python -m hypercompute.cli leave [--purge-cache]` `cli.py:402-414`, parser `:605-607`. It POSTs the agent's
  control `/leave` with `X-Hypercompute-Agent: <control_token>` (`cli.py:350-364`, `control.py:81-86`) and waits up
  to 20 s; prints `Asked the running agent to stop sharing.` then `Stopped.`. With no agent it cleans up orphans and
  tells the host (`cli.py:367-399`), or prints `This machine is not currently sharing.`. Exit 1 with
  `error: a worker started by Hypercompute is still running and could not be stopped` when it cannot.
- One-click `oneclick.py leave` `join/lib/oneclick.py:314-329` asks about purging the cache (default no without
  stdin), runs `leave`, removes `join.pid`.
- Pause/resume without leaving: `hypercompute pause` / `resume` `cli.py:417-427`, `:600-603`.

## 3. SDK `Pool` (sdk/python/src/hypercompute_sdk)

`Pool(base_url, *, owner_token=None, timeout=30.0, transport=None, sleep=time.sleep, clock=time.monotonic)`
`client.py:320-331`. Token order: argument, `HYPERCOMPUTE_OWNER_TOKEN`, then bootstrap on the first write
(`:324-329`, `_bootstrap` `:380-390`). After a host restart it refetches once on 403 `owner_only` (`_rotated`
`:392-403`). Every error message is scrubbed of the token (`errors.scrub` `errors.py:125-129`). `repr` shows only
the token source (`client.py:347-348`). Returns are the gateway JSON as plain dicts. 251 tests pass with fakes
(`.venv/Scripts/python.exe -m pytest sdk/python/tests -q`, 2026-10-25, 0.48 s).

### 3.1 Reads (no token)

| Method | Route | Returns | Line |
|---|---|---|---|
| `health()` | GET /api/health | dict | `client.py:415` |
| `nodes()` | GET /api/nodes | list of NodeView | `:418` |
| `peers()` | GET /api/peers | list of peer rows (1.5) | `:422` |
| `models()` | GET /api/models | list of ModelView | `:426` |
| `served_models()` | GET /v1/models | list of ids of the ready model | `:430` |
| `deployments()` | GET /api/deployments | newest first | `:438` |
| `deployment(dep_id)` | GET /api/deployments/{id} | Deployment | `:442` |
| `status()` | health + network + nodes + deployments + v1/models | summary dict | `:446` |
| `pools()`, `explain(model_id, ...)` | capacity routes | dict | `:435`, `:458` |

### 3.2 Models: pick a downloaded one

`ModelView` `schemas.py:456-468`: `artifact` (`id`, `display_name`, `family`, `params_b`, `quantization`, `format`,
`default_ctx`, `max_output_tokens`, `role`; `schemas.py:428-447`), `status`
(`not_downloaded | downloading | downloaded | corrupt | error | engine_managed`), `total_bytes`,
`downloaded_bytes`, `verified_sha256`, `gguf`, `error`. Offer only `status == "downloaded"`. The demo must never call
`POST /api/models/{id}/download` (`app.py:576-582`) or `hypercompute models pull`.

### 3.3 Plan

`plan(model_id, *, strategy="all_selected", ctx_size=4096, node_ids=None, allow_ram_tier=True,
allow_capacity_mode=True, engine=None, objective=None, overflow=None, use_profiles=None, parallel=1, layout=None)`
`client.py:475-485`, body builder `plan_body` `:154-181` (only given keys are sent; the gateway forbids extras),
`POST /api/plans` with a 180 s timeout (`app.py:584-586`). Request schema `PlanRequest` `schemas.py:491-507`.
Strategies `auto | all_selected | fastest_fit | local_only` (`schemas.py:472-476`); `all_selected` gives every
selected device layers, which is the one that shows the pool. `layout` pins stages: list of
`{node_id ("local" for the host), device, layers (count), residency?}` `schemas.py:479-488`.

Reply `Plan` `schemas.py:629-662`. For "which machine holds which layers":

- `feasible`, `fit_label`, `reasons`, `warnings`, `tier`, `capacity_mode`, `device_list`, `tensor_split`, `id`.
- `assignments[]` (`DeviceAssignment` `schemas.py:521-537`): `node_id`, `device`, `layers` (list of block indices),
  `holds_output`, `weight_bytes`, `kv_bytes`, `total_bytes_est`, `budget_bytes`, `fits`, `file_backed`. Map
  `node_id` to `display_name` with `nodes()`; `"local"` is the host.
- `tier` is one of `gpu | gpu_ram | gpu_disk | stream | disk | moe_offload` (`schemas.py:35`). `seed_and_run.py`
  refuses `gpu_disk` and `stream` by default because they page weights from SSD and swamped a GB10
  (`sdk/python/examples/seed_and_run.py:8-9`, `:27`, `:48-50`). The demo should do the same.
- `prediction` (optional, `PlanPrediction` `schemas.py:613-626`): `decode_ms`, `prefill_ms`, `load_ms`, `bottleneck`,
  `stages`, `explain` lines. Predictions, not measurements; label them so.

### 3.4 Seed peers (optional, gateway only, not in the SDK)

Only for peers that joined managed (`HC_MANAGED=1`). It makes the peer download its layers from huggingface.co, so it
is off in the demo unless chosen on purpose. Without it, llama.cpp RPC sends the layers over the tunnel at load time.

- `POST /api/peers/{node_id}/seed` (owner write, 202) body `{"model_id": M, "plan_id": P}` or
  `{"model_id": M, "layers": "30-35"}`, exactly one of them (`app.py:540-560`, `PeerSeedRequest` `schemas.py:361-364`).
  Reply is the job view `{job_id, node_id, model_id, layers, state, detail, tensors, bytes_total, parts,
  parts_accepted, skipped, started_at, updated_at}` (`services/peer_seed.py:291-296`; host job states `hashing |
  queued | delivered | refused | failed | cancelled` `:48`).
- Refusals: 404 `not_found`; 409 `not_managed`, `seeding_not_allowed`, `cache_off`, `seed_in_progress`,
  `peer_busy`, `model_not_downloaded`; 422 `layers_or_plan`, `invalid_layers`, `no_layers_in_plan`, `not_gguf`,
  `nothing_to_seed` (`services/peer_seed.py:94-421`, `app.py:551-559`). `seed_and_run.py` retries 409 every 15 s up to
  40 times (`:57-65`).
- Progress: poll `GET /api/peers`; a job is finished only when `seed_request.job_id == job_id` and
  `report.seed.job_id == job_id` with `report.seed.state` in `done | failed | cancelled` (`seed_and_run.py:67-80`).
  `report.seed` is `SeedProgress` (`bytes_done`, `bytes_total`, `tensors_done`, `tensors_total`, `tensors_failed`,
  `error`; `schemas.py:317-331`).
- Cancel: `POST /api/peers/{node_id}/seed/cancel` (`app.py:562-569`).

### 3.5 Deploy and wait

- `deploy(plan_id)` or `deploy(plan=<PlanRequest dict>)`, exactly one (`client.py:487-503`); sends a fresh
  `Idempotency-Key` and recovers the deployment by that key if the reply times out (`:493-511`). Route
  `POST /api/deployments` 202 (`app.py:588-591`, `CreateDeploymentRequest` `schemas.py:741-744`).
- Errors (`ConflictError`, 409): `plan_stale` (plan older than 10 minutes), `deployment_active` (stop the current
  one first), `deployment_replacing`, `node_not_eligible`, `not_admitted`; 422 `plan_not_feasible`
  (`backend/hypercompute/services/deployments.py:529-556`, `:662-664`, `:814`).
- `deploy_model(model_id, wait=True, ready_timeout=7200, on_progress=None, **plan_options)` plans, deploys, replans
  once on `plan_stale`, waits; it never stops someone else's deployment (`client.py:543-557`).
- `wait_ready(dep_id, *, timeout=7200.0, poll=2.0, on_progress=None)` polls `deployment()` every `poll` seconds and
  calls `on_progress(state, progress)` only when either changes (`client.py:513-541`). Raises `DeploymentFailed`
  (state `failed` or `stopped`, carries `.deployment`) or `ReadyTimeout`; gives up after 5 failed polls in a row
  (`:54`). It blocks, so run it in a thread, or poll `deployment()` from the UI loop.
- `Deployment` `schemas.py:701-719`: `id`, `state`, `progress`, `error`, `history[]` (`state`, `at`, `detail`),
  `plan`, `placement` (llama.cpp's observed per-device bytes, `schemas.py:686-698`), `participation_verified`,
  `participation_detail`. States in order `planned, reserving, preparing, loading, warming, ready`, then
  `draining, stopped, failed` (`schemas.py:665-674`). Progress text for llama.cpp comes from
  `load_phase` (`backend/hypercompute/adapters/llama_rpc.py:222-230`), for example
  `loading weights onto devices (peers receive their layers over the network; progress is not observable)`, and
  `ready` at the end (`services/deployments.py:643`).

### 3.6 Chat and stream with tok/s

- `stream(messages, *, model=None, max_tokens=None, temperature=None, **extra)` `client.py:581-617` yields
  `ChatChunk` (`types.py:10-38`): `content`, `reasoning` (llama.cpp `reasoning_content`), `finish_reason`,
  `timings`, `usage`, `raw`. Arguments are checked at call time; the request goes out on the first `next()`.
  Breaking out of the loop closes the response and the gateway stops generating (`:583-584`).
- tok/s: the last chunk usually carries `timings` (`EngineTimings` `schemas.py:750-757`): `predicted_per_second`
  (decode), `prompt_per_second` (prefill), `predicted_n`, `prompt_n`, `predicted_ms`. `seed_and_run.py:99-102` prints
  exactly these. A live counter can count chunks over wall time until the final timings arrive.
- `chat(...)` non-streamed `client.py:572-579`, reply `choices[0].message.content` / `reasoning_content`.
- `extra` passes llama.cpp fields, e.g. `chat_template_kwargs={"thinking": False}` for Kimi (`seed_and_run.py:87-95`).
  `messages` and `stream` may not be in `extra` (`client.py:196-197`). Body limit 16 MiB, checked before sending
  (`:212-215`).
- Route `POST /v1/chat/completions` (owner write, `app.py:732-762`): 503 `no_ready_deployment` before any stream
  (`services/deployments.py:395-405`), 404 `model_not_found` when `model` names another model, 400
  `vision_unavailable` for images without a projector. A stream that ends badly raises `StreamError` (`incomplete`,
  `bad_event`, or the server code) or `ChatInterrupted` with `partial_text` (`client.py:279-311`).

### 3.7 Stop

`stop(dep_id=None)` `client.py:559-568`: DELETE /api/deployments/{id} with a 300 s timeout (`app.py:601-603`).
With no id it stops the one non-terminal deployment, and returns None when there is none.

### 3.8 Error classes for UI messages

Picked by HTTP status (`errors.py:112-122`): 400/422 `RequestError`, 403 `OwnerAuthError`, 404 `NotFoundError`,
409 `ConflictError`, 413 `BodyTooLarge`, 500/502 `EngineError`, 503 `UnavailableError`; transport failure is
`GatewayUnreachable` (status 0, `client.py:361-364`). Each has `.status`, `.code`, `.detail`, `.fix`
(`errors.py:23-41`). Show `detail` and `fix`; both are already scrubbed of the token.

## 4. Where secrets appear (never print, log or embed)

| Secret | Appears in | Line |
|---|---|---|
| owner token | `GET /api/owner/bootstrap` reply; every owner write header | `app.py:446-450`, `client.py:366-370` |
| invite ticket + code | `POST /api/network/invite` reply; `hypercompute invite` stdout; host startup stdout | `app.py:312-321`, `cli.py:530-532`, `app.py:354-356` |
| invite (friend side) | `HC_INVITE`, then `HC_ONECLICK_TICKET` / `HC_ONECLICK_CODE`, then the join argv | `oneclick.py:52-72`, `:249-251`, `oneclick_supervise.py:104-115` |
| node credential, control token, ticket | `~/.hypercompute/agent/state.json` | `agent/join.py:495-497`, `:685` |
| agent control token | embedded in the agent's own status page HTML | `agent/control.py:66-70` |

Rules for the demo server: never log request bodies (the friend's paste arrives in one), pass the invite to the helper
only through the `Ctx.env` mapping or the child env, run every helper output line through `redact_generic`
(`oneclick_invite.py:108`) before it reaches the UI, and keep the owner token server side (the SDK does).

## 5. Gaps the demo has to cover itself

1. No stop-host command (`join/README.md:79`); the demo owns the host process (1.3).
2. The SDK wraps neither invites, events, nor seeding; use `httpx.Client(base_url, trust_env=False)` with the owner
   header from the same bootstrap the SDK uses (section 3), as `cli.py:524-529` does for invites.
3. Port defaults differ (8000 CLI, 8001 SDK and helper); pass one explicit port everywhere.
4. The one-click join may try to install the engine and opens nothing in the browser, but the one-click host opens
   the old dashboard unless `open_url` is replaced (1.1).
5. Friend status has no JSON CLI flag; use the agent control `/status` (2.3) or parse the `key: value` lines.
6. The gateway sends no CORS headers; the browser must go through the demo server.
7. The demo server itself should bind 127.0.0.1 only and check the `Host` header and a same-origin custom header on
   POSTs, the way the gateway does (`app.py:386-391`), since it can start processes on the machine.

## 6. Testing without a gateway

- SDK fake: `sdk/python/tests/fake_gateway.py` (`FakeGateway`, `FakeClock`, `sse`, `delta`, `TIMINGS`;
  `FakeGateway.pool()` returns `Pool(transport=httpx.MockTransport(self))`). `sdk/python/tests/conftest.py` blocks
  every socket and clears the token env; copy that pattern.
- One-click fakes: `join/tests/oneclick_testkit.py` (`StubRunner`, `FakeProc`, `Clock`, `make_ctx`) and the inventory
  fixtures in `join/tests/fixtures/`.
- Run with the repo venv: `.venv/Scripts/python.exe -m pytest sdk/python/tests join/tests -q`.
