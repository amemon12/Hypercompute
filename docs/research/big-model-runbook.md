# Big MoE model on the Iroh pair: host restart, run and rollback

Runbook for moving the live Iroh host from the 890f798 build to this worktree's code without losing the Mac, then
running an 80-120B mixture-of-experts GGUF through the gateway with `scripts/capacity/run_big.py`. Written
2026-10-22. Nothing here has been executed yet: the live gateway was not restarted, and no MoE model has been
downloaded or run on this host. Facts below come from reading the code (file and function named) or from files on
this host (path named). Numbers marked "computed" are arithmetic on measured inputs, not measurements.

## 1. What is running now

| Item | Value | Source |
| --- | --- | --- |
| Live host | `127.0.0.1:8001`, Iroh transport, `HYPERCOMPUTE_HOME=C:\Users\operator\.hypercompute\iroh-session` | `iroh-session\start-host.ps1` |
| Process tree | `uv.exe` 40552 -> `hypercompute.exe` 7304 -> `python.exe` 53064 -> `python.exe` 42668; 42668 listens on 8001 and on the Mac's RPC bridge `127.0.0.1:52110` | `Get-CimInstance Win32_Process`, `Get-NetTCPConnection` (2026-10-22) |
| Code | started 2026-10-21 17:35 local from this worktree when HEAD was 890f798 | `iroh-session\CHECKPOINT.md` |
| Config dir | `HYPERCOMPUTE_CONFIG_DIR` = the OneDrive checkout's `configs` (its `models.yaml` overrides the example) | `start-host.ps1` |
| Mac | `Azims-MacBook-Air-5.local`, node `4c299174...`, 890f798, contribution 10 GiB max, 4 GiB reserve, `--cache` | `CHECKPOINT.md` |
| Mac ticket relay | `https://usw1-1.relay.n0.iroh.link./`, 4 direct addresses | parsed from `iroh-session\mac-join-command.txt` (ticket not printed) |
| Other host | `python.exe` 16528 on port 8000 (LAN host, separate). Leave it alone. | `Get-NetTCPConnection` |
| Standalone 32B helper | not running: nothing listens on 18001, no `llama-server` process | `Get-NetTCPConnection` (2026-10-22) |
| Free space C: | 42.89 GiB at about 00:35 local; 13.32 GiB at 01:22 local after the gpt-oss download below started | `Get-PSDrive C` (2026-10-22) |
| Free RAM | 10.22 GiB of 31.12 GiB at 00:35; 8.05 GiB at 01:22 | `Win32_OperatingSystem` FreePhysicalMemory (same times) |
| Models in this home | `qwen2.5-0.5b-instruct-q4_k_m` (verified marker present); `gpt-oss-120b-mxfp4`: the full-size 59.03 GiB file is being written in 64 MiB ranges by a separate ranged downloader (891 of 945 ranges listed in its `.ranges.json` at 01:22), no `.verified.json` yet. No Qwen2.5 32B in this home. | `Get-ChildItem iroh-session\models` (01:22) |

Python imports a module when it is first used. Any module the live process imported for the first time after the
worktree moved past 890f798 came from newer source, so treat the live process as 890f798 for the wire protocol, not
as a clean 890f798 build.

The Mac's heartbeats are not continuous. `iroh-session\host.err.log` shows gaps of 206 s (to 23:22:53), about 22
minutes (23:40:39 to 00:02:32) and two short flaps after 00:26, each followed by `verifying` then `eligible` and
`verified: RPC0` without a re-join. The cause was not established. Check that the Mac is eligible right before
every plan.

## 2. Will a 890f798 Mac work with the new host?

Checked with `git diff 890f798 HEAD -- backend/hypercompute/schemas.py backend/hypercompute/transport
backend/hypercompute/agent` plus `services/nodes.py`, `services/sessions.py`, `config.py`, `storage.py`.

| Area | Change 890f798 -> HEAD (d2bb8cd) | Effect on the Mac |
| --- | --- | --- |
| `transport/host.py`, `client.py`, `iroh.py` | none | same control ops (`health`, `join`, `heartbeat`, `leave`), envelope, ALPN and RPC bridge handshake (`{"op": "rpc", "node_id"}`) |
| `agent/` | `supervisor.py`: `_bind_to_owner_windows` renamed `bind_to_owner` | host-side only |
| Wire models in `schemas.py`: `JoinRequest`, `JoinResponse`, `HeartbeatRequest`, `HeartbeatResponse`, `HostInventory` (pools, devices, engine), `ContributionLimits`, `WorkerEndpoint`, `NodeState`, `WorkerState` | none | none |
| Other `schemas.py` models | `Tier` adds `stream`; new `StreamSpec`, `PlanLimits`, `StagePrediction`, `PlanPrediction`, `ReplanDecision`; optional fields on `Plan`, `Deployment`, `RunRecord` | host-internal |
| `services/nodes.py`, `services/sessions.py`, `config.py` | none | same 3 s heartbeat, 15 s stale, 5 min host-unreachable rule |
| `storage.py` | `leases` gains `revision` and `chain_id` (added by `ALTER TABLE` on first open) | host-internal |

Wire incompatibilities found: none. The engine must still be llama.cpp b11770 (RPC proto 8.0.0) on both sides; the
join and verify checks compare it.

Uncommitted work in this worktree tonight (the `moe_offload` tier: `schemas.py` adds `MoeSpec`, `Plan.moe` and
`GgufSummary` expert fields; `planner.py`, `selector.py`, `llama_rpc.py`, `services/moe_tier.py`) touches none of
the wire models as of this writing. Others are still editing, so repeat this right before the restart and read any
hunk in the wire models listed above:

```powershell
git -C C:\Users\operator\.hypercompute\worktrees\iroh-transport diff 890f798 -- backend/hypercompute/transport `
  backend/hypercompute/agent backend/hypercompute/services/nodes.py backend/hypercompute/services/sessions.py `
  backend/hypercompute/config.py backend/hypercompute/schemas.py
```

One incompatibility exists in the other direction (rollback), see section 6: the 890f798 code cannot read
deployment or run rows written by the new code.

## 3. Why the Mac reconnects by itself, and when it cannot

From the code:

1. Same identity. `HostService.start` (`app.py` lifespan) loads `HYPERCOMPUTE_HOME\coordinator\iroh-device.key`,
   so the same home gives the same endpoint id, which is what the Mac's ticket names and Iroh TLS checks.
2. Same membership. `NodeRegistry.restore()` reloads the Mac's row with its credential hash (an HMAC under
   `coordinator\secret.key`) and marks it `stale` ("gateway restarted; waiting for heartbeat"). The session row
   stays `active`. `authenticate()` also requires the Mac's own Iroh id (`transport_peer_id`), which lives on the
   Mac and does not change.
3. Automatic redial. `PeerClient._request` (`transport/client.py`) drops a failed connection and the next heartbeat
   (every 3 s) dials the ticket again. No new code, no join.
4. New bridge. On the first authenticated heartbeat `HostService._bind_peer` opens a new loopback RPC bridge on a new
   random port (52110 goes away) and `attach_transport_worker` records it; the host re-verifies RPC0 and the node
   goes `verifying` -> `eligible` (4 s in the 23:23 log entry).

The friend must re-run join when any of these happen:

- The host is unreachable for more than `HOST_UNREACHABLE_STOP_S` = 300 s counted from the Mac's first failed
  heartbeat (`Agent._host_gone_too_long` in `agent/join.py`): the Mac stops sharing and its agent exits.
- A heartbeat is answered 401 or 403 (wrong credential, removed node, closed session, changed key): the Mac prints
  "The host removed this machine from the session" and stops. A different `HYPERCOMPUTE_HOME`, a lost
  `secret.key` or a replaced database causes this.
- The Mac cannot reach the new process with its old ticket. `IrohEndpoint` binds an OS-chosen UDP port on every
  start (`bind_addr=None`), so the ticket's direct addresses go stale; I found no address discovery configured in
  `transport/iroh.py`, so the Mac depends on the ticket's relay URL. If the restarted host picks a different home
  relay than `usw1-1`, the Mac keeps logging "host unreachable" and stops after 5 minutes.

## 4. Restart procedure (old gateway -> worktree code)

Prepare every command first; the 5 minute window starts at the stop. Run from PowerShell on the host.

**4.1 Before stopping (read only)**

Windows PowerShell 5.1 passes a JSON array from `Invoke-RestMethod` down a pipeline as one object, so
`Invoke-RestMethod ... | Where-Object` tests the whole array at once (checked on this host with a local JSON server:
the node filter printed nothing and the deployment filter printed stopped deployments). Every array read below is
therefore wrapped in parentheses, which makes PowerShell enumerate it.

```powershell
$wt = 'C:\Users\operator\.hypercompute\worktrees\iroh-transport'
$hc = 'C:\Users\operator\.hypercompute\iroh-session'
# Nothing may be loaded: this must print nothing.
(Invoke-RestMethod http://127.0.0.1:8001/api/deployments) | Where-Object { $_.state -notin 'stopped','failed' }
# The Mac must be eligible now, so a later failure is attributable to the restart.
(Invoke-RestMethod http://127.0.0.1:8001/api/nodes) | Where-Object { -not $_.is_local } |
  Select-Object display_name, state, heartbeat_age_s, worker_state
# Record the code that will start (others may still be editing this worktree).
git -C $wt rev-parse HEAD; git -C $wt status --short
# The relay in the Mac's ticket (prints only the relay URL).
Set-Location -LiteralPath $wt
uv run --no-sync python -c "import iroh,re,sys; t=re.search(r'--ticket\s+(\S+)', open(sys.argv[1]).read()).group(1); print(iroh.EndpointTicket.from_string(t).endpoint_addr().relay_url())" "$hc\mac-join-command.txt"
```

Run the test suites for the code you are about to start and do not restart on a red suite.

**4.2 Stop the old gateway**

```powershell
$gw = (Get-NetTCPConnection -LocalPort 8001 -State Listen).OwningProcess   # 42668 at the time of writing
Stop-Process -Id $gw
Start-Sleep -Seconds 2
Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue   # must print nothing
Get-Process -Id 53064, 7304, 40552 -ErrorAction SilentlyContinue                   # launchers exit with it
```

This is a hard stop. Committed SQLite transactions survive (WAL). `host.lock` is reclaimed because its PID is dead
(`agent/instance.py`). Filter by port 8001 only: port 8000 is a different host.

**4.3 Back up the coordinator state (4.5 MB)**

```powershell
$bak = "$hc\coordinator-backup-890f798-$(Get-Date -Format yyyyMMddTHHmmss)"
Copy-Item -LiteralPath "$hc\coordinator" -Destination $bak -Recurse
```

**4.4 Start the worktree code with the same home and key**

```powershell
$env:HYPERCOMPUTE_HOME = $hc
$env:HYPERCOMPUTE_LLAMA_BIN_DIR = 'C:\Users\operator\.hypercompute\engines\llama-b11770-windows-x64-cuda13.4'
$env:HYPERCOMPUTE_CONFIG_DIR = "$wt\configs"          # the MoE catalog entries live here (models.example.yaml)
$env:HYPERCOMPUTE_FRONTEND_DIST = "$wt\frontend\dist"
Remove-Item Env:HYPERCOMPUTE_REPLAN -ErrorAction SilentlyContinue   # no automatic replacement during measurements
Set-Location -LiteralPath $wt
Start-Process -FilePath 'C:\Users\operator\.local\bin\uv.exe' -WindowStyle Hidden `
  -ArgumentList 'run','--no-sync','hypercompute','host','--transport','iroh','--port','8001','--no-browser' `
  -RedirectStandardOutput "$hc\host-worktree.out.log" -RedirectStandardError "$hc\host-worktree.err.log"
```

`--no-sync` keeps uv from rewriting the shared `.venv`. The worktree `configs` has no `models.yaml`, so the gateway
reads `models.example.yaml`; it lists `qwen2.5-0.5b-instruct-q4_k_m` (verified in this home) and the four MoE
entries. The Qwen2.5 32B entry is there too, but its file is no longer in `iroh-session\models` (section 1). The
OneDrive `models.yaml` also lists `mistral-small-3.2-24b-instruct-q4_k_m`, which is not downloaded in this home.

**4.5 Confirm the relay did not move**

`host-worktree.out.log` shows "Iroh ready" and a fresh one-use invitation (normal; do not send it unless section 5
fails). Compare its relay with 4.1 (prints only the relay URL):

```powershell
uv run --no-sync python -c "import iroh,re,sys; t=re.findall(r'--ticket\s+(\S+)', open(sys.argv[1]).read())[-1]; print(iroh.EndpointTicket.from_string(t).endpoint_addr().relay_url())" "$hc\host-worktree.out.log"
```

## 5. Verify the Mac is eligible again

Expect this within about 10-20 s of "Iroh ready" (3 s heartbeat plus the RPC check).

```powershell
Select-String -Path "$hc\host-worktree.err.log" -Pattern 'node\.(state|verified|joined|left)'
# want: "verifying (heartbeat resumed; re-checking worker)", "eligible", "verified: RPC0"
(Invoke-RestMethod http://127.0.0.1:8001/api/nodes) | Where-Object { -not $_.is_local } |
  Select-Object display_name, state, state_reason, heartbeat_age_s, worker_state,
    @{n='reachable';e={$_.verification.reachable}}, @{n='engine_match';e={$_.verification.engine_match}}
(Invoke-RestMethod http://127.0.0.1:8001/api/network).peers |
  Select-Object node_id, connected, @{n='mode';e={$_.path.mode}}
uv run --no-sync hypercompute status --port 8001
```

Eligible means `state` eligible, `heartbeat_age_s` under 15, `worker_state` running, `reachable` and `engine_match`
True, and the peer `connected` with a `relay` or `direct` path. Then prove the Mac executes layers through the same
runner, on the small model:

```powershell
uv run --no-sync python scripts/capacity/run_big.py --model qwen2.5-0.5b-instruct-q4_k_m --strategy all_selected `
  --reps 1 --home $hc
```

Pass: `status` ok, `measured.placement.participation_verified` true, a positive
`measured.peer_worker_cpu.nodes.<mac>.delta_s`.

**If the Mac is not eligible after 2 minutes:** ask the friend what the agent printed. "host unreachable" lines mean
it cannot reach the new process (usually a relay change); "removed this machine" means a 401/403. In both cases, and
whenever the agent has exited, the friend re-joins from the 890f798 checkout on the Mac with a new invitation:

```sh
uv run hypercompute leave            # only if the agent is still running or left a state file
uv run hypercompute join --ticket <NEW_TICKET> --code <NEW_CODE> --max-memory 10GiB --reserve 4GiB --cache --yes
caffeinate -dims &                   # keep the Mac awake for the run
```

Get `<NEW_TICKET>` and `<NEW_CODE>` from the invitation in `host-worktree.out.log` or a fresh one
(`uv run --no-sync hypercompute invite --port 8001`; one use, 15 minute expiry). The Mac keeps its node id, and a
re-join from the same machine (same Iroh key) is accepted for a node still held in the session (`NodeRegistry.join`).

## 6. Rollback to 890f798

The 890f798 schemas forbid unknown fields (`Model.model_config = ConfigDict(extra="forbid")`) and its
`DeploymentManager.reconcile_on_start` validates every `deployments` row at startup. Rows written by the new code carry
`revision`, `replaces` and plan fields (`stream`, `prediction`, `limits`, `revision`, and `moe` once that lands); run
records carry `resumed_from` and three more. One such row stops the old gateway from starting, and `/api/runs` would
fail. So a rollback restores the database from 4.3. Deployments and runs recorded by the new code disappear from the
database; the evidence stays in `results/capacity/*/run.json`.

```powershell
# 1. stop the new gateway exactly as in 4.2
# 2. restore the database ($bak is the folder made in 4.3); delete the live WAL first so newer transactions are
#    not replayed onto the old file
Remove-Item "$hc\coordinator\hypercompute.db-wal", "$hc\coordinator\hypercompute.db-shm" -ErrorAction SilentlyContinue
Copy-Item "$bak\hypercompute.db*" "$hc\coordinator\" -Force
# 3. a clean 890f798 checkout with its own venv (uv.lock is identical between 890f798 and HEAD)
git -C $wt worktree add --detach C:\Users\operator\.hypercompute\worktrees\rollback-890f798 890f798
Set-Location -LiteralPath C:\Users\operator\.hypercompute\worktrees\rollback-890f798
uv sync
# 4. start with the original environment from start-host.ps1, from the rollback checkout
$env:HYPERCOMPUTE_HOME = $hc
$env:HYPERCOMPUTE_LLAMA_BIN_DIR = 'C:\Users\operator\.hypercompute\engines\llama-b11770-windows-x64-cuda13.4'
$env:HYPERCOMPUTE_CONFIG_DIR = 'C:\Users\operator\OneDrive\Documents\ChatGPT\CalHacks 13.0\hypercompute\configs'
$env:HYPERCOMPUTE_FRONTEND_DIST = 'C:\Users\operator\OneDrive\Documents\ChatGPT\CalHacks 13.0\hypercompute\frontend\dist'
Start-Process -FilePath 'C:\Users\operator\.local\bin\uv.exe' -WindowStyle Hidden `
  -ArgumentList 'run','--no-sync','hypercompute','host','--transport','iroh','--port','8001','--no-browser' `
  -RedirectStandardOutput "$hc\host-rollback.out.log" -RedirectStandardError "$hc\host-rollback.err.log"
# 5. verify the Mac as in section 5
```

The keys (`iroh-device.key`, `secret.key`) are unchanged by either build, so the Mac reconnects to the rollback the
same way, under the same 5 minute rule, with one exception: if the friend re-joined while the new build ran (the
fallback in section 5), that join issued a new credential (`NodeRegistry.join` calls `new_credential()`), and the
restored database only holds the hash of the old one. The Mac's next heartbeat then gets 401/403, its agent stops,
and the friend re-joins the rollback host with a fresh invitation (`uv run --no-sync hypercompute invite --port 8001`
from the rollback checkout), exactly as in section 5.

Never `git checkout` or stash inside the shared worktree for a rollback: other sessions keep uncommitted work there.

## 7. Running the big model

**Catalog** (`configs/models.example.yaml`; sizes and sha256 from the Hugging Face tree API, revision = repo sha;
architecture facts from each file's GGUF header, fetched by byte range):

| id | repo | file size | experts (used) | blocks | free space a fresh gateway download needs on C: |
| --- | --- | --- | --- | --- | --- |
| `qwen3-next-80b-a3b-instruct-q3_k_m` | unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF | 35.67 GiB | 512 (10) | 48 | 37.67 GiB |
| `qwen3-next-80b-a3b-instruct-q4_k_m` | Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF | 45.09 GiB | 512 (10) | 48 | 47.09 GiB |
| `gpt-oss-120b-mxfp4` | ggml-org/gpt-oss-120b-GGUF | 59.03 GiB | 128 (4) | 36 | 61.03 GiB (already being placed, see section 1) |
| `glm-4.5-air-q4_k_m` | ggml-org/GLM-4.5-Air-GGUF | 59.26 GiB (2 parts) | 128 (8) | 46 | 61.26 GiB |

The last column is the file size plus the downloader's 2 GiB margin (`ModelService._download_files`); the download
goes to `<file>.part` next to the target and is renamed, so it never needs twice the space. With 13.32 GiB free on C:
at 01:22 none of the other three fits while the gpt-oss file is there. Sizes and sha256 were re-checked against the
HF API at review time (all match). The official Qwen file uses the legacy `blk.N.ssm_in` tensor layout; b11770 still
loads it (`src/models/qwen3next.cpp`, "ssm_in is used by legacy GGUF"). The GLM part 1 has 0 tensors; its sha256 was
checked against the catalog by a full download of that 8.9 MiB part.

**Steps**

1. Get the model verified by the gateway (the runner never downloads). Do not use `hypercompute models pull` for this
   home: the CLI downloads into its own shell's `HYPERCOMPUTE_HOME`, which is a different directory on the same C:
   unless that variable is set. The owner token stays in a variable:
   ```powershell
   $tok = (Invoke-RestMethod http://127.0.0.1:8001/api/owner/bootstrap).owner_token
   Invoke-RestMethod -Method Post http://127.0.0.1:8001/api/models/<id>/download `
     -Headers @{ 'X-Hypercompute-Owner' = $tok } | Select-Object status, downloaded_bytes, total_bytes
   ```
   Poll `(Invoke-RestMethod http://127.0.0.1:8001/api/models) | Where-Object { $_.artifact.id -eq '<id>' }` until
   `status` is downloaded and `verified_sha256` is true. For `gpt-oss-120b-mxfp4`, whose file a separate ranged
   downloader writes in place, send this POST only after that downloader has finished and checked the sha256. The
   gateway then hashes the existing file (twice: the disk check, then `download_verified`), downloads nothing and
   writes `.verified.json`. If the file is incomplete or wrong, the gateway wants the full 61.03 GiB free and
   refuses with "not enough disk", which leaves the file alone.
2. Plan only: `uv run --no-sync python scripts/capacity/run_big.py --model <id> --strategy all_selected --home $hc
   --plan-only`. Read `predicted.plan` (tier, `device_list`, `rpc_endpoints`, `moe` when the `moe_offload` tier is in
   the build) and `predicted.explain`. `all_selected` plans onto every eligible peer: if the Mac is not eligible the
   plan is host only and the run proves nothing about distribution, so check that `rpc_endpoints` is not empty.
3. Full run, Mac included: the same command without `--plan-only`, with `--load-timeout 10800`. Bytes sent to the
   Mac cross the relay at 1.58e6-1.59e6 B/s (measured, `results/capacity/*-link-relay-*`), about 680 s per GiB
   (computed); `--cache` on the Mac makes a reload of the same tensors local.
4. Host only for comparison: `--strategy fastest_fit` (or `local_only`).

The runner refuses when another deployment is active, when the gateway lacks `/api/capacity` (an old build), or when
the model is not downloaded and verified. It stops its own deployment at the end, after a failure and after Ctrl+C,
unless `--keep`.

**Reading `results/capacity/<stamp>-big-<model>/run.json`**

| Field | Question it answers |
| --- | --- |
| `status` | ok, partial, chat_failed, load_failed, load_timeout, aborted_by_guard, infeasible, refused, interrupted |
| `predicted.plan`, `predicted.explain` | what the planner and optimizer chose and expected (verbatim) |
| `measured.placement.engine.moe` | `-ot` patterns, `--n-cpu-moe`, `--no-repack`, `--no-host`, `--no-op-offload`, `--load-mode` and `--fit` actually passed |
| `measured.engine_log.file_backed_host_mib` vs `private_host_mib` | CPU expert weights stayed in `CPU_Mapped` (file cache) rather than private `CPU`, `CPU_REPACK` or `CUDA_Host` buffers that count toward commit |
| `measured.host.whole_run` | minimum available RAM, peak commit charge and limit, minimum free space on C:, peak GPU memory |
| `measured.chats.summary` | TTFT, prefill and decode (engine timings), prompt_n, system-wide disk reads per output token over the whole chat and per decode step over decode only |
| `measured.peer_worker_cpu` | the Mac's rpc-server CPU seconds during the reps (participation) |
| `comparison` | predicted decode and SSD bytes per token next to the measured ones (`measured_decode_disk_read_bytes_per_token_median` is the like-for-like SSD number: it leaves out the prompt's reads) |

After Ctrl+C or an error during the chats, `measured.chats` keeps the chats that finished (`interrupted` true) next to
the load and placement records.

**Safety.** The runner's monitor stops the deployment when C: falls below 5 GiB free, when the commit limit grows by
more than 0.25 GiB (Windows raises it only by growing a pagefile: private memory went past RAM, the failure that
filled the disk on 2026-10-21) or when available RAM drops below 1 GiB. Do not lower the drive floor or raise the
commit cap.
If the engine log shows CPU experts in private buffers (`private_host_mib` large, `file_backed_host_mib` near zero),
stop: that layout grows commit and the pagefile on C:.

The available-RAM floor can trip on a healthy `moe_offload` run. Windows counts file-backed pages in a process's
working set as in use, not available, so expert pages the engine touched can take available RAM toward zero without
any commit (an inference from how Windows reports memory, not observed on this host). If `guard_tripped` names
available RAM while `whole_run.commit_charge_peak` stayed well under `commit_limit_peak` and the commit limit did not
grow, that was file cache pressure; a rerun with `--min-available-gib 0.25` keeps the drive and commit guards.

## 8. Not covered yet

- No restart, reconnection, MoE download or MoE inference has been run; `run_big.py` is tested only against a mocked
  gateway (`tests/test_run_big.py`).
- `docs/iroh-networking.md` already lists physical restart/reconnection as unvalidated; section 3 is a code reading.
- Whether the host keeps the `usw1-1` home relay across a restart is not known until 4.5.
- Metal support for MXFP4 or Qwen3-Next tensors over RPC on the Mac was not checked.
- Disk reads in `run.json` are system wide and bracketed by 0.5 s samples; a sha256 pass over a model file, or any
  other process reading C: during a run, inflates them.
