# Failure-injection catalogue: G14 + M4 Air (+ WSL2)

> **Added after review (2026-10-21).** The friend's M4 MacBook Air had already run three deployments with this G14 over campus Wi-Fi on the MVP build (`results/pair/20261021-campus-wifi-g14-m4air.json`, summarized in [../TEST_PLAN.md](../TEST_PLAN.md) section 0). Where this file says the macOS binary, the Mac inventory or a real Mac split has never run, read it against that evidence: the binary ran, the Metal working set is 12,124 MiB, the Mac's planner budget was 2.9-4.4 GiB, and its splits decoded at about 12 tok/s (14B) and a median 9.1 tok/s (Mistral 24B) on the old local-first order, where the Mac held the output head and sent 0.5-0.6 MB of logits back per token. [../TEST_PLAN.md](../TEST_PLAN.md) cites the IDs in this file with a `FI-` prefix (for example `FI-N2`).

Basis: static read of `C:/Users/operator/hypercompute` at `dfd8cf9` (main). Nothing was executed against a gateway,
no model was loaded and no repo file was touched. Every "Expected" below is what the code says will happen, with
`file:line` (paths relative to `backend/hypercompute/` unless they start with `frontend/`, `docs/` or `.venv/`).
Where the code cannot decide the outcome (OS TCP behaviour, macOS power management, vLLM internals) the entry says
"verify" and names the evidence that settles it.

Hardware in scope: HOST = ASUS G14 (RTX 5070 Ti Laptop 12 GiB, ~11.4 GiB contributed, 32 GB RAM, Windows 11,
WSL2 Ubuntu). FRIEND = M4 MacBook Air 16 GB (Metal working set ~10.7 GiB, default contribution 10.0 GiB with
4.00 GiB kept free). Later: a second Mac, an RTX 5090 desktop, an ASUS GX10 (section 6).

Helper: `scripts/stress/failures_load.py`
(`chats` mode = N concurrent chats with timestamps; `idle` mode = N idle TCP sockets to the gateway). It compiles
and its `--help` runs in the project environment; it has not been run against a gateway.

---

## 1. The 10 injections most likely to expose a real bug

| # | Injection | What breaks, per the code | Why it is likely real | Cost |
|---|---|---|---|---|
| 1 | **N2/N3** Mac Wi-Fi off 20 s mid-load or mid-chat (also F1 sleep, N4, N9 hotspot) | The host kills llama-server while the Mac is offline, so the FIN never arrives. `ggml-rpc-server` is single-client, has no keepalive and a backlog of 1 (`docs/research/llama_rpc.md:26-31,280`), so it stays blocked on the dead connection. When the Mac returns, every re-probe times out (25 s, `services/nodes.py:339-354`) and the card says "worker unreachable" with **firewall** advice. Only Pause+Resume on the Mac recovers. Metal buffers from a half-finished load stay allocated | This is the most common real-world event on Wi-Fi, and nothing in the agent or gateway resets a wedged worker. In lifecycle run `725198fe` (`docs/RESULTS.md`), `%USERPROFILE%\.hypercompute\coordinator\logs\llama-server-725198fe….log` ends mid-request ("init sampler") with no abort; the 17 s heartbeat timeout ended that run, not llama-server | Pair, 5 min |
| 2 | **F9** `kill -STOP` the Mac worker (stands in for a Metal hang or swap thrash) | Heartbeats still report `running` (`agent/join.py:254-260`), so nothing marks the node lost. Mid-load, the load waits up to 1800 s (`adapters/llama_rpc.py:168-191`). Mid-chat, each request burns 180 s and ends `timeout` (`config.py:27`, `services/streaming.py:342-345`). The deployment stays `ready` forever because `_watch` checks only that the process is alive (`services/deployments.py:493-507`) | There is no token- or load-progress watchdog, even though the RPC research asked for one (`docs/research/llama_rpc.md:296-298`). Heartbeats already carry `worker_cpu_seconds` and nothing reads it | Pair, 5 min, deterministic |
| 3 | **N5** Mac IP changes (roam or a new DHCP lease) | `/heartbeat` ignores the source IP (`app.py:290-294`), and an eligible heartbeat only re-saves (`services/nodes.py:242-243`). The node stays "Ready to contribute" with a dead `worker.host`. Every new plan includes it and fails at *preparing*. Rejoining from the new IP gets 409 `node_id_in_use` until the host clicks Remove (`services/nodes.py:196-204`). Measured profiles keyed by `ip:port#i` stop matching (`services/deployments.py:160`) | Deterministic. Hotspots and mesh Wi-Fi hand out new leases | Pair, 5 min |
| 4 | **N11/C3** 100 idle TCP sockets from any LAN device, or 110 concurrent chats | uvicorn counts open connections, idle ones included, and answers 503 to every new request once there are ≥100 (`cli.py:164-167`, `.venv/Lib/site-packages/uvicorn/protocols/http/h11_impl.py:227-229`; idle sockets get no keep-alive timer, `h11_impl.py:326`). The Mac's heartbeats get 503, the agent counts them as failures (`agent/join.py:356-367`), and the node goes stale after 15 s, which fails the deployment. Unauthenticated LAN DoS | Deterministic, trivial, security-relevant | G14 + any LAN box, 2 min |
| 5 | **T2/T5** Click Stop on the "Tuning trial" card (`frontend/src/components/DeploymentPanel.tsx:55-66`), or lose the peer during the peer candidate | The tuner treats the stopped or lost trial as a failed candidate and starts the next one within ~0.5 s (`services/tuner.py:447-449,283`). If any candidate finished, `_finish` saves it as a **measured profile** (`services/tuner.py:526-543`) under a signature that includes the Mac (`services/tuner.py:345-353`). Later plans replay it, and the Mac sits idle although it was never measured | Deterministic. Stopping a trial is exactly what a user does to abort a tune. Wi-Fi flakiness on test day makes T5 likely | Pair, 15 min |
| 6 | **E9/E7** Pause or leave the Mac during an EXO run, or drop its Wi-Fi | The agent stops EXO first (`agent/join.py:262-268,393-401`). The host's DELETE then fails, `live.gone` is still None, and the stop is unclean (`adapters/exo.py:661-668`), so the leases are **held** (`services/deployments.py:534-540`). `cleanup_orphan` retries at every gateway start and fails while the Mac's EXO is gone (`adapters/exo.py:687-711`). The leak survives restarts and shrinks the Mac's lease allowance (`services/leases.py:52-63`) | Deterministic. The node goes stale at 15 s, before EXO's own 45 s watcher (`adapters/exo.py:90`) | Needs EXO on the Mac |
| 7 | **F8/F12** Worker exits on its own, or the host stops a deployment mid-token | The agent ends the whole session when the worker exits (`agent/join.py:368-371`), and every rejoin costs a new invite (`agent/join.py:296-313`, `services/sessions.py:153-155`). The worker can die of SIGPIPE when the host closes mid-response: llama.cpp sets neither MSG_NOSIGNAL nor SO_NOSIGPIPE (`docs/research/llama_rpc.md:290-295`), and Python's `subprocess` restores SIGPIPE to the default for the child (`agent/supervisor.py:203-206`) | A Stop mid-chat (10 tries) or a Metal abort under memory pressure. One event ends the friend's participation | Pair, 10 min |
| 8 | **F10** Two `resume` calls at once, or pause and resume at once | `pause`/`resume` run in separate threads with no lock (`agent/control.py:88-101`, `agent/join.py:270-293`). The second resume's `reconcile_orphans` kills the first one's new worker (`agent/join.py:282`, `agent/supervisor.py:313-336`). Both threads share one log file opened `wb` (`agent/join.py:193`, `agent/supervisor.py:196`). A failing resume's `_stop_shared()` stops whichever worker is registered (`agent/join.py:287-290`). One reachable end state: `paused=False` with no worker, so the host shows "worker stopped" indefinitely | Deterministic in unit form. The local page's buttons have no in-flight guard (`agent/control.py:43-45`) | Mac only, 5 min |
| 9 | **F7** `kill -9` the agent on the Mac | macOS has neither a Job Object nor PDEATHSIG, and the worker runs in its own session (`agent/supervisor.py:6-7,197-202`). The **unauthenticated** `ggml-rpc-server` keeps listening on the LAN indefinitely, because the 5-minute host-loss stop lives in the dead agent (`agent/join.py:330-338`) | Deterministic. The docstring acknowledges the platform gap, but nothing mitigates it | Mac only, 2 min |
| 10 | **E4b** Kill only vLLM's EngineCore inside WSL | `_watch` checks only `process_alive` (`services/deployments.py:500-501`, `adapters/vllm.py:578-580`). `/health` 503 is treated as engine death only during startup (`adapters/vllm.py:618-620`). If vLLM's API server survives its engine core, the deployment stays `ready` and every chat returns a 5xx | Depends on vLLM 0.32 behaviour, which this repo has not observed | G14 only, 5 min |

Runners-up: **C5/C6** (Stop with queued chats dispatches them to the dying engine; a chat that arrives during a
stop gets HTTP 200 with an empty body and no run record), **H6** (llama.cpp `validate` does not re-read free VRAM,
and plans stay valid for 10 min), **N6** (after a G14 IP change the dashboard advertises an address the gateway is
not bound to), **H4** (G14 sleep makes a healthy Mac look lost, and >5 min makes its agent quit), and **D2**
(Chrome's 6-connections-per-host limit with SSE plus streaming chats freezes the Stop button).

---

## 2. Setup and evidence kit (reuse in every entry)

### 2.1 G14 (PowerShell, repo root)

```powershell
# host with logs captured (cmd /c avoids PowerShell 5.1 wrapping native stderr)
cmd /c "uv run hypercompute host --no-browser > %TEMP%\hc-host.out.log 2> %TEMP%\hc-host.err.log"
# second window
$GW  = "http://127.0.0.1:8000"
$tok = (Invoke-RestMethod "$GW/api/owner/bootstrap").owner_token
$H   = @{ "X-Hypercompute-Owner" = $tok }
$TP  = "C:\Users\operator\hypercompute\scripts\stress"
curl.exe -sN "$GW/api/events" > "$env:TEMP\hc-events.sse"            # timestamped event log (third window)
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$env:TEMP\hc-nvsmi.csv"

function hc-snap($tag) {   # one folder per injection
  $d = "$env:TEMP\hc-snap\$tag-$(Get-Date -Format HHmmss)"; New-Item -ItemType Directory -Force $d | Out-Null
  Invoke-RestMethod "$GW/api/nodes"           | ConvertTo-Json -Depth 10 | Out-File -Encoding utf8 "$d\nodes.json"
  Invoke-RestMethod "$GW/api/deployments"     | ConvertTo-Json -Depth 12 | Out-File -Encoding utf8 "$d\deployments.json"
  Invoke-RestMethod "$GW/api/runs?limit=100"  | ConvertTo-Json -Depth 10 | Out-File -Encoding utf8 "$d\runs.json"
  Invoke-RestMethod "$GW/api/tune"            | ConvertTo-Json -Depth 12 | Out-File -Encoding utf8 "$d\tunes.json"
  Get-NetTCPConnection -RemotePort 47052 -ErrorAction SilentlyContinue | Out-File -Encoding utf8 "$d\rpc-conns.txt"
  Get-Process llama-server,wsl,python -ErrorAction SilentlyContinue | Select Id,ProcessName,WS | Out-File -Encoding utf8 "$d\procs.txt"
}
# leases still held (the gateway keeps the DB open in WAL mode; reading is safe)
uv run python -c "import sqlite3,pathlib;c=sqlite3.connect(pathlib.Path.home()/'.hypercompute/coordinator/hypercompute.db');[print(r) for r in c.execute('select deployment_id,node_id,pool_id,bytes,created_at from leases where released_at is null')]"
```

LONGCHAT is a long streamed chat used as the in-flight request. PowerShell 5.1 strips quotes passed to native
programs, so the body goes through a file:

```powershell
'{"stream":true,"max_tokens":1500,"ignore_eos":true,"messages":[{"role":"user","content":"Write a long story about a lighthouse keeper."}]}' | Set-Content -Encoding ascii "$env:TEMP\hc-long.json"
curl.exe -sN -H "X-Hypercompute-Owner: $tok" -H "Content-Type: application/json" --data-binary "@$env:TEMP\hc-long.json" "$GW/v1/chat/completions" > "$env:TEMP\hc-chat.sse"
```

Deployment driver: `uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-q4_k_m --strategy all_selected --keep --label inj-<id>`
(plan, deploy, one streamed chat, leave running; `scripts/e2e_check.py:40,135`). The dashboard works too.

Files: `%USERPROFILE%\.hypercompute\coordinator\records\{events,runs}.jsonl`, `...\coordinator\logs\llama-server-<dep>.log`,
`vllm-<dep>.log`, `airllm-worker-<dep>.log`, `exo-<dep>.log`, `...\coordinator\managed-processes.json`.

### 2.2 Mac (Terminal)

```bash
caffeinate -dims uv run hypercompute join --host <G14-IP>:8000 --code <CODE> --cache 2>&1 | tee -a ~/hc-agent.log
tail -f ~/.hypercompute/agent/logs/ggml-rpc-server.log      # "Accepted client connection" / "Client connection closed"
W=$(pgrep -x ggml-rpc-server); lsof -nP -a -p "$W" -iTCP    # LISTEN + ESTABLISHED sockets of the worker
netstat -anp tcp | grep 47052                               # an ESTABLISHED line with no host deployment = wedge
vm_stat 2; sysctl vm.swapusage; memory_pressure | tail -1
pmset -g batt; pmset -g | grep -i lowpowermode; pmset -g log | grep -iE "sleep|wake" | tail
cat ~/.hypercompute/agent/state.json ~/.hypercompute/agent/managed-processes.json
```

Activity Monitor: Memory tab (Memory Pressure graph, `ggml-rpc-server` Memory column), Window > GPU History.

### 2.3 Timing constants that define "expected"

| Constant | Value | Where |
|---|---|---|
| Heartbeat interval / HTTP timeout | 3 s / 8 s | `config.py:16`, `agent/join.py:350,373` |
| Stale after | 15 s, plus ≤2 s monitor tick (observed 16-17 s) | `config.py:17`, `services/nodes.py:247-259,280-287` |
| Agent stops sharing after host loss | 300 s | `config.py:18`, `agent/join.py:330-338` |
| Invite TTL / join rate limit | 900 s / 10 per min per IP | `config.py:19,21`, `services/sessions.py:137-163` |
| Gateway connection cap | 100 (idle sockets count) | `config.py:22`, `cli.py:164-167` |
| Probe timeout / unreachable re-probe | 25 s / ≥30 s | `config.py:24`, `services/nodes.py:37,269-278` |
| llama.cpp load / warmup / per-read idle | 1800 s / 120 s / 180 s | `config.py:25-27`, `adapters/llama_rpc.py:72,168` |
| Stop grace then kill | 8 s + 8 s | `config.py:28`, `agent/supervisor.py:260-293` |
| Inference slot wait (then 429/`busy`) | 30 s, FIFO | `services/streaming.py:43,93-131,272-281` |
| Stored plan validity | 600 s | `services/deployments.py:363-370` |
| Host VRAM/RAM refresh | every 10 s | `coordinator.py:84-94` |
| vLLM slots / stop grace + watchdog | 8 interactive, 16 throughput / 30 s | `adapters/vllm.py:86,89`, `adapters/vllm_output.py:210-232` |
| AirLLM idle / load budget | 900 s / 4 h | `adapters/airllm.py:68-69` |
| EXO watch / unreachable / command | 3 s / 45 s / 45 s | `adapters/exo.py:83-91` |
| Benchmark exclusive wait / tuner slot wait | 300 s / 300 s | `services/benchmarks.py:120`, `services/tuner.py:72` |
| Tuner per-trial ready budget | load + warmup + 60 s (1980 s for llama.cpp) | `services/tuner.py:192-195` |
| Event ring | 500 events | `services/events.py:17-19` |

---

## 3. Catalogue

Format per entry: **Trigger**, **Expected (code)**, **Evidence**, **Covered by**. "NONE" means no automated test
exercises the path, so it needs the hardware run (or a new unit test where noted).

### N. Network

**N1. Client isolation (campus or guest Wi-Fi)**
- Trigger: both laptops on eduroam or a guest SSID. On the Mac run `uv run hypercompute doctor --host <G14-IP>:8000`, then the join command.
- Expected: `check_coordinator` fails with "Cannot open TCP …" plus the isolation hint (`agent/join.py:121-137`). `_run_join` stops at `cli.py:240`, before the worker starts or the POST is sent, so the invite is not consumed. Asymmetric variant (Mac→G14 open, G14→Mac blocked): the join succeeds but verification fails. The node goes ineligible "worker unreachable from host" with the fix text (`services/nodes.py:54-66,339-354`) and is re-probed every ≥30 s (`services/nodes.py:269-278`).
- Evidence: doctor output; `GET /api/nodes` → `verification.detail/fix`; `node.state` events.
- Covered by: `test_join_with_bad_host_fails_without_traceback`, `test_unreachable_worker_is_ineligible_with_fix`, `test_unreachable_worker_is_reprobed_on_a_backoff_not_every_heartbeat`. Real network: hardware.

**N2. Mac Wi-Fi off for 20 s mid-load** (TOP-10 #1)
- Trigger: start the 7B `all_selected` run. When the Run card shows "loading weights onto devices …", run on the Mac `networksetup -setairportpower en0 off; sleep 20; networksetup -setairportpower en0 on` (check the device name with `networksetup -listallhardwareports`).
- Expected: heartbeats stop and the node goes stale at 15-17 s. `on_node_lost` (`services/deployments.py:570-574`) cancels `_run` (`services/deployments.py:442-445`), and cleanup TerminateProcess-es llama-server (`agent/supervisor.py:269-270`) and releases the leases. State `failed`, "peer xxxxxxxx lost: no heartbeat for 1Xs". The G14's FIN/RST is lost while the Mac is offline. When Wi-Fi returns, a heartbeat moves the node stale → verifying (`services/nodes.py:236-238`) and the probe runs. Predicted (single client, no keepalive): the worker is still in `recv` on the dead connection, so the probe hangs 25 s and the node goes ineligible "worker unreachable from host" with **firewall** advice. Re-probes every ~55 s keep failing, and the partial load's Metal buffers stay allocated. Recovery: `uv run hypercompute pause; uv run hypercompute resume` on the Mac.
- Evidence: G14 `Get-NetTCPConnection -RemotePort 47052` is empty while the Mac's `netstat -anp tcp | grep 47052` still shows ESTABLISHED to the G14's old port (that is the wedge). `ggml-rpc-server.log` has no "Client connection closed" after the drop. `GET /api/nodes` → `verification.detail = "probe timed out after 25s"`. The worker's memory in Activity Monitor stays high.
- Covered by: `test_node_loss_invalidates_ready_deployment` (ready only), `test_stale_node_and_worker_exit`. Mid-load cancel on real TCP and the wedge: NONE.

**N3. Mac Wi-Fi off for 20 s mid-chat** (TOP-10 #1)
- Trigger: LONGCHAT on the 7B split. After the first tokens, the same `networksetup` off/on as N2. Variant: off for 8 s (below the stale threshold).
- Expected: tokens stop (llama-server is blocked on RPC). At 15-17 s `interrupt` pops the request (`services/streaming.py:283-288`) and the SSE stream gets `upstream_error` with "interrupted: deployment stopped or a peer was lost". The run record is `cancelled` (`services/streaming.py:346-350`) and the deployment `failed`. The wedge follows as in N2. 8 s variant: the node never goes stale. Either TCP survives the blip and generation resumes, or llama-server reaches its retransmission limit and aborts. Then `_watch` fails it (`services/deployments.py:493-507`) while the node stays eligible. Verify which happens.
- Evidence: `hc-chat.sse` tail; runs; same as N2.
- Covered by: as N2.

**N4. G14 Wi-Fi drops mid-chat**
- Trigger: on the G14 run `netsh wlan disconnect`, wait 20 s, then `netsh wlan connect name="<profile>"` (`netsh wlan show profiles`).
- Expected: the Mac's heartbeats fail and the host marks it stale at 15-17 s. The host kills llama-server while it is offline, so the wedge from N2 applies. The agent keeps its worker for up to 300 s of host loss and then stops sharing (`agent/join.py:330-338`). Verify two points. First, whether uvicorn's socket bound to the G14's LAN IP at startup (`cli.py:185-195`) still accepts after the reconnect: from the Mac, `curl -m 3 http://<G14-IP>:8000/api/health`. Second, whether DHCP returned the same IP; if not, see N6.
- Evidence: agent log "host unreachable (ConnectError)"; host events; `Get-NetTCPConnection -LocalPort 8000 -State Listen | select LocalAddress`; `ipconfig`.
- Covered by: `test_host_unreachable_too_long_stops_sharing`, `test_leave_after_host_lost_retries_goodbye_then_clears_state` (agent side). Socket behaviour: NONE.

**N5. Mac IP changes while idle and while serving** (TOP-10 #3)
- Trigger: on the Mac run `sudo networksetup -setmanualwithdhcprouter "Wi-Fi" <unused IP in the same subnet>`, once with no deployment and once during LONGCHAT. Restore with `sudo networksetup -setdhcp "Wi-Fi"`.
- Expected: `/heartbeat` never compares `request.client.host` with `worker.host` (`app.py:290-294`). An eligible running heartbeat just saves (`services/nodes.py:242-243`), and eligible nodes are never re-probed (`services/nodes.py:269-278`). So the node stays "Ready to contribute" with the old address, while the worker is still bound to the old IP (`agent/join.py:188`). The next deployment's validate probe fails (`adapters/llama_rpc.py:102-117`) → `failed` at preparing, and the node stays eligible, so every later plan repeats the failure. If it changes during a chat, the host's unacked sends eventually reset (Windows retransmission limit, roughly 10-30 s, verify), llama-server aborts and `_watch` fails it, and the node is still eligible. If the friend restarts the agent from the new IP, the join gets 409 `node_id_in_use` until the host clicks Remove (`services/nodes.py:196-204`). Tune profiles keyed by `ip:port#i` (`services/deployments.py:160`, `services/selector.py:247-250`, `services/profiles.py:21-24`) no longer match.
- Evidence: `GET /api/nodes` (state `eligible`, `worker.host` = old IP, `heartbeat_age_s` 0-3); Mac `ifconfig en0 | grep inet` and `lsof -nP -iTCP:47052 -sTCP:LISTEN` (old IP); deployment error "worker probe failed".
- Covered by: `test_join_rejects_spoofed_worker_address` (join time only). Heartbeat-time: NONE (unit-testable).

**N6. G14 IP changes**
- Trigger: in an Administrator shell, `netsh interface ip set address name="Wi-Fi" static <new IP> 255.255.255.0 <router>`. Restore with `netsh interface ip set address name="Wi-Fi" dhcp`.
- Expected: the sockets stay bound to the startup addresses (`cli.py:185-195`), so the Mac cannot reach the gateway. The node goes stale, the deployment fails, and after 300 s the agent quits. The dashboard **advertises the new address anyway**: `SessionService.lan_addresses()` recomputes `netutil.lan_addresses()` on every call (`services/sessions.py:78-84`, `coordinator.py:111-112`), although its docstring says "addresses this gateway actually listens on". A Mac pf scoping rule pinned to the old host IP (`agent/join.py:493-508`) also blocks the new one.
- Evidence: `Get-NetTCPConnection -LocalPort 8000 -State Listen`; the join command shown in the dashboard; Mac `curl -m 3 http://<new IP>:8000/api/health`.
- Covered by: NONE.

**N7. Latency and loss on the RPC link**
- Trigger: on the Mac, using the built-in dummynet. Use an anchor other than `com.apple/hypercompute`, because `pfctl -a X -f -` replaces the agent's scoping anchor (`agent/join.py:503-508`).
  ```bash
  sudo dnctl pipe 1 config delay 40 plr 0.01        # steps: 0/20/40/80 ms one way, plr 0/0.01/0.05
  printf 'dummynet in quick proto tcp from any to any port 47052 pipe 1\ndummynet out quick proto tcp from any port 47052 to any pipe 1\n' | sudo pfctl -a com.apple/hc-netem -f -
  sudo pfctl -E
  # undo: sudo pfctl -a com.apple/hc-netem -F all; sudo dnctl -q flush
  ```
- Expected: the planner has no latency input (a prior only; `docs/IMPLEMENTATION_STATUS.md` known gaps), so the plan does not change. Split decode falls roughly as 1/(compute + RTT) per token, long-prompt TTFT grows and the load slows. Heartbeats go to port 8000, which is not shaped, so no node loss. Only `hypercompute tune` reacts.
- Evidence: per step, `timings.predicted_per_second` and `ttft_ms` in runs; `ping -c 50 <G14-IP>` on the Mac; load seconds from the deployment history.
- Covered by: NONE (performance; use these numbers to calibrate the cost prior).

**N8. Host on the Windows Public profile, and the firewall prompt**
- Trigger: (a) no gateway rule on a Public network; on the Mac run `doctor --host`. (b) Answer Cancel to Windows Defender's prompt for `python.exe` on the first `host`, then add the README allow rule and run `doctor --host` again.
- Expected: Hypercompute never edits firewall rules. (a) Mac doctor FAIL "Cannot open TCP". (b) Windows Firewall block rules beat allow rules, so it still fails with the "Hypercompute gateway" rule present. The host's diagnostic says "Public profile; inbound connections to **the worker** are blocked" (`agent/inventory.py:176-187`), which on a host is about port 8000, so the text misleads.
- Evidence: `Get-NetConnectionProfile`; `Get-NetFirewallApplicationFilter | ? Program -like '*python*' | Get-NetFirewallRule | ft DisplayName,Action,Profile,Enabled`; `Get-NetFirewallRule -DisplayName "Hypercompute gateway" | Get-NetFirewallAddressFilter`.
- Covered by: NONE.

**N9. Phone hotspot drops**
- Trigger: iPhone Personal Hotspot. Mid-chat, turn "Allow Others to Join" off for 20 s and back on (or an Android hotspot off/on).
- Expected: both machines lose the link, so N2 plus N4 apply and the wedge is likely. The hotspot may hand out new IPs (N5/N6). The iPhone's 172.20.10.0/28 counts as private, so the join is accepted.
- Evidence: as N2/N5/N6, plus `ipconfig`/`ifconfig` before and after.
- Covered by: `test_private_lan_classification`, `test_lan_or_link_local_classification`.

**N10. RPC port blocked while heartbeats keep flowing**
- Trigger: on the Mac, mid-chat, `printf 'block drop in quick proto tcp to any port 47052\n' | sudo pfctl -a com.apple/hc-block -f - && sudo pfctl -E`. Undo with `sudo pfctl -a com.apple/hc-block -F all`. The same thing happens when the agent's own scoping rule names an old host IP.
- Expected: heartbeats are unaffected, so the node stays eligible. With unacked data, llama-server blocks until Windows TCP gives up (~10-30 s, verify), aborts, and `_watch` fails the deployment. With nothing unacked, the deployment stays `ready` until the next chat. After that, every deployment fails at validate while the node stays eligible, because eligible nodes are never re-probed (`services/nodes.py:242-243,269-278`).
- Evidence: time to failure; deployment error; node state; llama-server log tail.
- Covered by: NONE.

**N11. Idle-socket DoS on the gateway** (TOP-10 #4)
- Trigger: while a 7B split is ready, run from any LAN machine (or the G14) `uv run python $TP\failures_load.py idle --host <G14-IP> --n 100 --hold 60`.
- Expected: `connection_made` adds every socket to `connections` before a byte arrives. With ≥100, every request gets 503 (`.venv/Lib/site-packages/uvicorn/protocols/http/h11_impl.py:227-229`, uvicorn 0.54.0). No keep-alive timer is armed until a response completes (`h11_impl.py:326`), so the sockets live as long as the attacker holds them. The Mac's heartbeats get 503, the node goes stale at 15-17 s, and the deployment fails. Dashboard polls get 503 too.
- Evidence: script timestamps; agent log "host unreachable (HTTPStatusError)"; host `node.state` stale; deployment error.
- Covered by: `test_gateway_server_limits_concurrent_connections` (asserts only the config value).

### F. Friend's Mac

**F1. Sleep or lid close**
- Trigger: mid-chat, `pmset sleepnow` or close the lid (`caffeinate -dims` does not prevent lid-close sleep), for 30 s and for 6 min.
- Expected: as N2 (stale at 15-17 s, deployment failed). On wake, heartbeats move the node to verifying and the probe runs, with the wedge risk. The agent's 300 s host-loss timer uses `time.monotonic` (`agent/join.py:331,361`); whether that clock counts sleep on this macOS/CPython is version-dependent, so verify that a 6-minute sleep does not make the agent quit. Doc mismatch: the README troubleshooting says a stale Mac must "rejoin with a new code", but the code recovers with no rejoin while the agent is alive (`services/nodes.py:236-238`).
- Evidence: `pmset -g log | grep -iE "sleep|wake"`; agent log; host events.
- Covered by: `test_stale_node_and_worker_exit`. Wake path on hardware: NONE.

**F2. Low battery**
- Trigger: there is no precise trigger. Run unplugged with a chat loop below 10 % and watch `pmset -g batt`; at the critical level the Mac sleeps (→ F1).
- Expected: heartbeats carry no power state (`agent/join.py:344-347`), so the host gives no warning before the drop.
- Evidence: `pmset -g batt` and `pmset -g log`; host events.
- Covered by: NONE.

**F3. Low Power Mode**
- Trigger: `sudo pmset -a lowpowermode 1` mid-benchmark; revert with `sudo pmset -a lowpowermode 0`.
- Expected: nothing reacts, and decode and prefill drop. A tuner profile measured in one mode replays in the other, because profiles are keyed by model, ctx and devices only (`services/profiles.py:21-24`).
- Evidence: benchmark results per mode; `sudo powermetrics --samplers gpu_power -i 1000 -n 10`.
- Covered by: NONE.

**F4. Memory pressure and swap while serving layers**
- Trigger: with the 14B or 7B split serving, run `memory_pressure -l critical` (check `man memory_pressure` for this macOS) or open about 30 heavy tabs.
- Expected: heartbeats report a shrinking `uma` pool (psutil `available`, `agent/inventory.py:227-231`), which only shrinks **new** plans (`services/deployments.py:135-163`). The running deployment has no slowness detection. Outcomes: a slower decode; or a Metal command-buffer timeout → worker abort → agent quits (F8); or a long stall → F9 behaviour.
- Evidence: Activity Monitor pressure graph and worker Memory; `vm_stat 2`; `sysctl vm.swapusage`; `log show --last 10m --predicate 'process == "ggml-rpc-server"'`; run rates.
- Covered by: NONE.

**F5. macOS firewall prompt (Allow, Deny, ignore)**
- Trigger: `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on` before joining. For Deny: `W=$(find ~/.hypercompute/engines -name ggml-rpc-server -type f | head -1); sudo /usr/libexec/ApplicationFirewall/socketfilterfw --blockapp "$W"`. Later run `--unblockapp "$W"`.
- Expected: the join succeeds but verification fails. The node goes ineligible "worker unreachable from host" with the macOS fix (`services/nodes.py:59-62,341-354`). A re-probe ≥30 s later picks up Allow with no rejoin (`services/nodes.py:37,269-278`). Every Pause/Resume starts a new worker process; check whether an ad-hoc-signed binary prompts again.
- Evidence: `socketfilterfw --listapps`; verification detail and timestamps.
- Covered by: `test_unreachable_worker_is_ineligible_with_fix`, `test_unreachable_worker_is_reprobed_on_a_backoff_not_every_heartbeat`.

**F6. Quit Terminal**
- Trigger: with the agent started through `caffeinate -dims uv run …`, quit Terminal (Cmd+Q, then Terminate). Variant: close the window.
- Expected: SIGHUP triggers the handler (`agent/join.py:528-544`), then `leave()`. The worker is in its own session, so SIGHUP does not reach it (`agent/supervisor.py:197-202`); `leave()` TERMs its group and sends the goodbye. The host marks the node `left: contributor stopped sharing` immediately, not stale (`services/nodes.py:357-367`), and `state.json` is removed (`agent/join.py:411`). Verify that `caffeinate`/`uv` do not take python down before the cleanup runs.
- Evidence: `left` versus `stale` in the host events tells which path ran; `pgrep -fl ggml-rpc-server` must be empty.
- Covered by: `test_stop_signals_trigger_graceful_stop` (simulated signal).

**F7. `kill -9` the agent** (TOP-10 #9)
- Trigger: `pkill -9 -f "hypercompute join"` mid-chat.
- Expected: the worker survives (`agent/supervisor.py:6-7,96-112,197-202`) and keeps serving until the host marks the node stale and kills llama-server. After that it listens on `<ip>:47052` **unauthenticated, indefinitely**. `state.json` and `managed-processes.json` remain. It is cleaned only by `uv run hypercompute leave` (`cli.py:292-300`) or the next join (`agent/join.py:178-182,562`).
- Evidence: `pgrep -fl ggml-rpc-server`; `lsof -nP -iTCP:47052 -sTCP:LISTEN`; from the G14 an hour later, `Test-NetConnection <mac-ip> -Port 47052`.
- Covered by: `test_leave_without_state_stops_a_recorded_live_worker`, `test_reconcile_stops_only_our_orphans`. The macOS orphan: NONE.

**F8. `kill -9` the worker (`ggml-rpc-server`)** (TOP-10 #7)
- Trigger: `pkill -9 -x ggml-rpc-server`, once mid-chat and once mid-load.
- Expected: llama-server aborts on the RPC failure and is caught by `_watch` within 2 s, or by `wait_ready` during load (`adapters/llama_rpc.py:174-177`). Within 3 s the agent's heartbeat reports `exited` (-9) (`agent/join.py:254-260`), and the host marks the node ineligible "worker exited (exit code -9)" and handles the loss (`services/nodes.py:230-235`). There is exactly one cleanup (`services/deployments.py:509-521`). Then the agent **exits by design** (`agent/join.py:368-371`) and says goodbye, so the node is `left` and a new invite is needed. The error text: the needle `GGML_ABORT` in `adapters/llama_output.py:46` is a macro name that `ggml_abort` does not print, so expect the generic "llama-server exited …" instead of a named cause.
- Evidence: agent output "The local worker exited (code -9). Stopping."; host events; deployment error; llama-server log tail.
- Covered by: `test_stale_node_and_worker_exit`, `test_stop_and_node_loss_share_one_cleanup`, `test_failure_patterns` (only "Failed to connect to"); lifecycle `7129a8d0` (loopback). macOS: hardware.

**F9. Stalled worker with no progress watchdog** (TOP-10 #2)
- Trigger: mid-load, `kill -STOP $(pgrep -x ggml-rpc-server)`, wait 120 s, then `kill -CONT …`. Mid-chat, the same with a 200 s stop (over the 180 s read timeout).
- Expected: `poll()` is None for a stopped process, so heartbeats say `running` (`agent/join.py:254-260`) and the node stays eligible. Mid-load: the TCP window fills and `wait_ready` keeps polling for up to 1800 s (`adapters/llama_rpc.py:168-191`). Mid-chat: no bytes for 180 s → SSE error `timeout` and run status `timeout` (`services/streaming.py:342-345`). llama-server stays alive and blocked, so `_watch` keeps the deployment `ready` (`services/deployments.py:500-501`). The next chat queues behind the stuck `-np 1` slot (`adapters/llama_rpc.py:128`) and times out as well. Heartbeats already carry `worker_cpu_seconds` (`agent/join.py:259`, `services/nodes.py:228-229`); a flat value while a request is in flight is the signal a watchdog could use.
- Evidence: `ps -o pid,stat,command -p $(pgrep -x ggml-rpc-server)` shows state `T`; the deployment's `progress`; runs; `worker_cpu_seconds` from `/api/nodes` over time.
- Covered by: NONE.

**F10. Pause/resume races** (TOP-10 #8)
- Trigger: `uv run hypercompute pause`, then `uv run hypercompute resume & uv run hypercompute resume & wait`, ten times. Also `uv run hypercompute pause & uv run hypercompute resume & wait`, and a double-click on Resume in the local page.
- Expected: no lock (`agent/control.py:88-101`, `agent/join.py:270-293`). The interleavings are listed in TOP-10 #8. A bad end state is `paused=False` with no worker, which heartbeats as `stopped` (`agent/join.py:256-257`) and leaves the host showing ineligible "worker stopped" until another pause+resume.
- Evidence: both commands' output; `pgrep -fl ggml-rpc-server | wc -l`; `uv run hypercompute status`; host `state_reason`; `managed-processes.json`.
- Covered by: `test_resume_that_fails_on_exo_stops_the_worker_again_and_stays_paused`, `test_resume_whose_worker_exits_early_stays_paused_instead_of_reading_as_exited`, `test_agent_pause_resume_require_token`. Concurrency: NONE (unit-testable without hardware).

**F11. Rejoin with an old code, and agent restarts**
- Trigger: Ctrl+C the agent and run the same join command again. Also use an unused code after 16 min, and make 11 join attempts within a minute.
- Expected: 403 `code_used` (`services/sessions.py:153-155`); after the TTL, 403 `code_expired` (`services/sessions.py:156-158`); more than 10 per minute, 429 `rate_limited` (`services/sessions.py:139-140`). The worker is started before the POST, so a refused join starts and then stops a worker (`agent/join.py:562-574`). A new code from the same Mac is accepted, because the persisted `node_id` (`agent/join.py:63-71`) matches the same worker IP (`services/nodes.py:196-204`); a new IP gets 409 (N5). Gap: there is no reconnect with the stored credential, so every agent restart costs an invite.
- Evidence: agent output; host events.
- Covered by: `test_invite_is_one_time`, `test_invite_expires`, `test_rate_limit_blocks_bruteforce`, `test_join_cannot_take_over_another_machines_node_id`.

**F12. Stop mid-token (SIGPIPE on the worker)** (TOP-10 #7)
- Trigger: ten times, run LONGCHAT on the 7B split and press Stop about 3 s into the tokens (or `Invoke-RestMethod -Method Delete -Headers $H "$GW/api/deployments/<id>"`).
- Expected (code): `interrupt`, then TerminateProcess on llama-server (`services/deployments.py:523-544`); the worker reads EOF, logs "Client connection closed" and accepts again. Predicted risk: a worker writing at that moment can die of SIGPIPE (`docs/research/llama_rpc.md:290-295`), exit -13, and the agent quits (`agent/join.py:368-371`).
- Evidence: number of survivals out of 10; agent output "code -13"; worker log tail.
- Covered by: NONE.

**F13. Mac disk fills up with `--cache` on**
- Trigger: `mkfile -n <size> ~/hc-fill`, leaving less free space than the Mac's layers; load the 14B split; `rm ~/hc-fill` afterwards.
- Expected: tensors over 10 MiB go to `$LLAMA_CACHE/rpc` (`agent/join.py:190-192`; no size cap and never deleted, per the research doc). What happens on ENOSPC (log and continue, or abort) is unknown.
- Evidence: worker log; `df -h ~`; `du -sh ~/.hypercompute/rpc-cache`.
- Covered by: NONE. Low priority.

### H. Host G14 (and WSL2)

**H1. Gateway `kill -9` during load**
- Trigger: `$gw=(Get-NetTCPConnection -LocalPort 8000 -State Listen | select -First 1).OwningProcess; Get-Process -Id $gw` (it must be `python`, not `uv`), then `Stop-Process -Id $gw -Force` during "loading weights onto devices". Restart the host.
- Expected: the kill-on-close Job Object kills llama-server (`agent/supervisor.py:70-93`), and the Mac sees an RST (link up) and frees its buffers. On restart, `reconcile_on_start` marks the deployment `failed` "gateway restarted while deployment was active" and releases its leases (`services/deployments.py:576-604`). Nodes come back as stale "gateway restarted; waiting for heartbeat" (`services/nodes.py:108-120`), and the next Mac heartbeat moves the node to verifying, then eligible (the credential survives in the DB). The owner token is new (`coordinator.py:139`), and the dashboard refetches it on the first 403 (`frontend/src/api/client.ts:147-151`).
- Evidence: startup notes in `hc-host.err.log` and as `gateway.startup` events; `GET /api/deployments`; `Get-Process llama-server` returns nothing; `managed-processes.json`; the Mac's worker memory drops.
- Covered by: `test_restart_reconciliation_marks_active_failed`, `test_child_dies_when_owner_is_hard_killed`, `test_restart_keeps_leases_of_a_llama_server_that_survived`; lifecycle `8a8576ac` (loopback plus WSL).

**H2. Gateway `kill -9` during chat**
- Trigger: as H1, during LONGCHAT.
- Expected: as H1. The in-flight chat leaves **no run record**, because the record is written in `stream_events`' `finally` (`services/streaming.py:351-356`), which never runs. The SIGPIPE risk from F12 applies.
- Evidence: request count versus `runs.jsonl` lines.
- Covered by: as H1.

**H3. Gateway `kill -9` during tune**
- Trigger: as H1, while `hypercompute tune` is measuring a candidate.
- Expected: after the restart the tune is "failed: gateway restarted" (`services/tuner.py:563-570`), the trial deployment fails as in H1, and no profile is saved.
- Evidence: `/api/tune`, `/api/profiles`.
- Covered by: `test_reconcile_marks_running_tunes_failed`.

**H4. Windows sleep**
- Trigger: run `powercfg /a` first (Modern Standby or S3). With a split deployment ready, choose Start → Power → Sleep (or close the lid) for 2 min, then for 6 min.
- Expected: the gateway freezes and the Mac sees the host as unreachable. On resume, the monitor compares wall-clock heartbeat age (`services/nodes.py:252-254`), so it marks the healthy Mac stale immediately and fails the deployment. The Mac's next heartbeat moves the node to verifying, with the wedge risk if the G14's FIN never arrived. After more than 300 s the agent stops sharing and its goodbye fails (the host is asleep), so the node stays stale forever and a new invite is needed. Under Modern Standby the network may stay up while the process is throttled; record which happened. IP change: see N6.
- Evidence: host event timestamps; agent log; `powercfg /sleepstudy`; `ipconfig`.
- Covered by: `test_host_unreachable_too_long_stops_sharing`. Host resume: NONE.

**H5. GPU used by a game or another CUDA process during a run**
- Trigger: with the 14B auto run ready (CPU 4L + CUDA0 44L), start a VRAM hog from the pinned engine, `& "<bin dir from 'uv run hypercompute engine status'>\llama-server.exe" -m <7B gguf> -ngl 99 -c 8192 --port 9999`, or launch a game.
- Expected: nothing reacts. `refresh_local` only feeds future plans (`coordinator.py:84-94`). WDDM spills one process into shared memory, giving the README's 27 → 18 tok/s class of slowdown, with no warning and no failure.
- Evidence: the nvidia-smi CSV; Task Manager GPU 1 Dedicated versus Shared; decode rates before, during and after.
- Covered by: NONE.

**H6. VRAM taken between Check plan and Run**
- Trigger: Check plan for the 14B auto, start the H5 hog (or a WebGL page with Chrome on High-performance GPU), then click Run together within 10 min.
- Expected: stored plans are accepted for 600 s (`services/deployments.py:363-370`). llama.cpp `validate` checks only the model file and peers (`adapters/llama_rpc.py:102-117`) and launches with `--fit off` (`adapters/llama_rpc.py:128`), so it loads, spills and runs slowly with no error. vLLM, by contrast, re-reads nvidia-smi and refuses with a fix (`adapters/vllm.py:492-511`). Variant: a stale vLLM in WSL holding about 10.6 GiB makes llama.cpp plans quietly shift layers to the CPU or the Mac with no hint.
- Evidence: plan `budget_bytes`/`budget_basis` versus nvidia-smi at launch; placement; decode.
- Covered by: `test_validate_rereads_free_memory_and_prepare_launches` (vLLM only). llama.cpp: NONE.

**H7. RAM pressure on the host-RAM tier**
- Trigger: with the 14B auto run ready, `uv run python -c "import time;n=22*2**30;b=bytearray(n);b[::4096]=b'\x01'*(n//4096);time.sleep(600)"`, then LONGCHAT.
- Expected: `cpu_budget` is computed only at plan time (`services/deployments.py:123-132`). The CPU layers are mapped GGUF pages and their KV sits in RAM; Windows trims them, hard faults follow, and decode sinks toward `gpu_disk` speed. No detection.
- Evidence: `Get-Counter '\Memory\Available MBytes','\Memory\Pages Input/sec' -SampleInterval 1 -MaxSamples 60`; Resource Monitor hard faults for `llama-server.exe`; decode rates.
- Covered by: `test_ram_tier_respects_the_host_ram_budget` (plan time only).

**H8. Disk full during a model download**
- Trigger: `uv run hypercompute models pull qwen2.5-14b-instruct-q8_0` (or any model not yet downloaded). At about 30 %, fill C: below the remaining size with `fsutil file createnew C:\hc-fill.bin <bytes>` (admin) or `uv run python -c "open(r'C:\hc-fill.bin','wb').truncate(<bytes>)"`. Delete the file afterwards.
- Expected: the pre-check passes because it only needs total + 2 GiB at the start (`services/models.py:183-186`). The write fails with an OSError, which is recorded on the model view along with a "Download failed" event (`services/models.py:198-202`). The `.part` file stays (`engine.py:222`), and a retry resumes with a Range request (`engine.py:227-233`); sha256 is verified at the end.
- Evidence: `GET /api/models` (`status`, `error`); `dir $env:USERPROFILE\.hypercompute\models\<id>\*.part`; the retry's progress starts at the part size.
- Covered by: `test_complete_part_is_finalized_without_a_range_request`, `test_download_verifies_sha256`, `test_adopt_cached_replaces_truncated_destination`. ENOSPC itself: NONE.

**H9. Disk full during vLLM's first start** (do this last)
- Trigger: force vLLM on a checkpoint that is not yet in the WSL cache. While the card shows the download phase, fill C: below the remaining download.
- Expected: disk room was checked only before launch (`adapters/vllm.py:513-532`). ENOSPC inside the VHDX makes vLLM exit. There is no FATAL marker for "No space left" (`adapters/vllm_output.py:43-62`), so the error is `exit_detail` with the last exception line (`adapters/vllm.py:594-598`). Risk: ext4 in the VHDX can go read-only when the host disk is full, and then WSL needs `wsl --shutdown`.
- Evidence: `vllm-<dep>.log`; `wsl -d Ubuntu -- df -h /`; `wsl -d Ubuntu -- dmesg | tail -20`.
- Covered by: `test_plan_needs_disk_room_for_the_first_download_only`, `test_validate_rechecks_disk_room_for_a_stored_plan`. Mid-download: NONE.

**H10. WSL shutdown during a vLLM run**
- Trigger: with vLLM 7B AWQ ready and LONGCHAT running, `wsl --shutdown`.
- Expected: `wsl.exe` exits, and `_watch` catches it within 2 s (`services/deployments.py:493-507`) using `exit_reason` (`adapters/vllm.py:453`). The cleanup's `stop()` runs `wsl.exe -d <distro> --exec bash -c STOP` (`adapters/vllm.py:676-698`), which **boots the distro again**. STOP finds a dead pid, reports already-exited or no-pidfile, the stop is clean and the leases are released.
- Evidence: `wsl -l --running -q` (Ubuntu running again); nvidia-smi shows the memory released; the cleanup text.
- Covered by: `test_stop_signals_inside_wsl_then_reaps`, `test_failure_reason_for_a_server_that_died_after_ready`. Hardware: NONE.

**H11. The WSL VM keeps holding RAM after vLLM**
- Trigger: stop a vLLM deployment and Check plan for the 14B auto at once. Then `wsl --shutdown` and Check plan again.
- Expected: `cpu_budget` comes from psutil `available` (`services/deployments.py:123-132`), so the two plans differ by whatever `vmmem` kept, and the tier may flip.
- Evidence: both plans' CPU `budget_bytes`; `Get-Process vmmem* | ft Name,WS`.
- Covered by: NONE. Low priority.

### E. Engines

**E1. RPC worker crash mid-load**: covered by F8 (mid-load). Check that exactly one terminal state is written and how the error is worded (`services/deployments.py:509-521`, `adapters/llama_output.py:37-48`).

**E2. Version mismatch**
- Trigger: on the Mac, build another tag with `scripts/build_llama_macos.sh` and `export HYPERCOMPUTE_LLAMA_BIN_DIR=<its bin dir>` before joining.
- Expected: the join is refused before any worker starts: "engine mismatch: host runs llama.cpp b11770 …" (`cli.py:245-248`). The host-side check (`services/nodes.py:315-323`) also marks the node ineligible without probing. A source build reports `b<build>` (`engine.py:341-343`), so it is caught.
- Evidence: join output.
- Covered by: `test_engine_mismatch_is_rejected_before_any_probe`. Low hardware value.

**E3. vLLM OOM at startup**
- Trigger: force vLLM (7B AWQ), click Run, and within about 5 s start the H5 hog (≥4 GiB).
- Expected: validate passed on the earlier free VRAM (`adapters/vllm.py:492-511`). The log then shows "on startup is less than" or "CUDA out of memory" (`adapters/vllm_output.py:43-62`), which raises `engine_failed` (`adapters/vllm.py:589-593`). Cleanup runs STOP in WSL and releases the leases.
- Evidence: `vllm-<dep>.log`; deployment error and cleanup; nvidia-smi.
- Covered by: `test_wait_ready_fails_fast_on_a_fatal_marker_and_on_exit`, `test_the_normal_free_memory_info_line_is_not_a_startup_failure`; a real run is lifecycle `14cb3362`.

**E4. vLLM killed inside WSL** (TOP-10 #10 is part b)
- Trigger (a), the API server: `wsl -d Ubuntu -- pkill -KILL -f "vllm.entrypoints.cli.main serve"`.
- Expected (a): `wsl.exe` exits and `_watch` reports "vLLM server exited unexpectedly: …" (`adapters/vllm_output.py:112-115`). The LAUNCH watchdog TERMs, then KILLs, the group within 30 s (`adapters/vllm_output.py:210-217`), and STOP reports already-exited.
- Trigger (b), the EngineCore only: get the pid from `Select-String -Path <vllm-<dep>.log> -Pattern 'EngineCore pid=(\d+)' | select -First 1`, then `wsl -d Ubuntu -- kill -KILL <pid>`.
- Expected (b): see TOP-10 #10. Verify whether vLLM 0.32's API server exits on its own; if it does, (a) applies.
- Evidence: `curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:<upstream_port>/health` (`upstream_port` from `GET /api/deployments/<id>`); `wsl -d Ubuntu -- pgrep -af vllm`; nvidia-smi; runs.
- Covered by: `test_failure_reason_for_a_server_that_died_after_ready`. (b): NONE.

**E5. Gateway `kill -9` during a vLLM run**
- Trigger: as H1, during a vLLM run.
- Expected: the Job Object kills `wsl.exe`. vLLM inside Linux keeps the GPU until the watchdog notices its waiter is gone, then TERMs, and KILLs after 30 s (`adapters/vllm_output.py:210-217`). On restart the leases are held for `cleanup_orphan` (`services/deployments.py:585-598`), which stops the pidfile's server and only then releases (`adapters/vllm.py:700-724`).
- Evidence: nvidia-smi timeline from the kill (the memory should be back within about 35 s); `wsl -d Ubuntu -- pgrep -af vllm`; startup notes.
- Covered by: `test_cleanup_orphan_stops_the_server_a_crashed_gateway_left_in_wsl`, `test_cleanup_orphan_never_starts_wsl_and_holds_leases_when_unsure`, `test_restart_holds_leases_until_the_engine_removes_its_orphan`. Hardware: NONE (`docs/RESULTS.md` shows only a graceful vLLM stop).

**E6. AirLLM worker killed mid-token**
- Trigger: with the AirLLM 0.5B ready and a chat running (about 5 s per token), `Stop-Process -Id <pid from GET /api/deployments/<id>> -Force`.
- Expected: the stream is reset and the record is `error`, or `cancelled` if `_watch`'s interrupt lands first (`services/streaming.py:339-350`). `_watch` catches it within 2 s with `failure_reason = last_error(log)` (`adapters/airllm.py:369`), and the cleanup frees the VRAM.
- Evidence: `airllm-worker-<dep>.log`; nvidia-smi; runs.
- Covered by: `test_adapter_raises_the_workers_mid_stream_error`, `test_wait_ready_reports_a_dead_worker_from_its_log`, `test_closing_the_stream_cancels_generation_and_frees_the_slot`. A kill after ready: NONE.

**E7. EXO API unreachable mid-run** (needs `engine install --engine exo` and `join --exo` on the Mac; TOP-10 #6)
- Trigger: an EXO deployment on the Air (an MLX 4-bit model), chat running, Mac Wi-Fi off for 20 s.
- Expected: see TOP-10 #6. The node goes stale at 15 s, before EXO's 45 s watcher (`adapters/exo.py:90,602-618`). The DELETE then fails (`adapters/exo.py:418-431,661-668`), the leases are held, and the instance stays loaded on the Mac. A later plan whose Mac share plus the leaked bytes exceeds the Mac's budget fails at *reserving* (`services/leases.py:52-63`) although the planner reported it feasible.
- Evidence: cleanup text; the lease query; Mac `curl -s http://127.0.0.1:52415/state/instances`; startup notes after a restart.
- Covered by: `test_stop_when_exo_is_unreachable_is_not_clean`, `test_cleanup_orphan_is_false_until_exo_confirms_the_instance_is_gone`.

**E8. EXO port taken**
- Trigger: `python3 -m http.server 52415` on the Mac before `join --exo`.
- Expected: `start_exo` refuses ("port 52415, EXO's API port, is already in use", `agent/exo_sidecar.py:113-122`) before the POST (`agent/join.py:565-566`), so the invite is not consumed. The worker that already started is stopped by `finally leave()`.
- Evidence: agent output; `pgrep -fl ggml-rpc-server` is empty.
- Covered by: `test_start_refuses_when_another_exo_already_serves_the_api_port`, `test_api_port_check_matches_a_live_listener_only`.

**E9. Pause or leave during an EXO run** (TOP-10 #6)
- Trigger: with an EXO deployment ready, run `uv run hypercompute pause` on the Mac (or Ctrl+C the agent).
- Expected: deterministic leaked leases that survive restarts (see the TOP-10 row).
- Evidence: the lease query before and after a gateway restart; cleanup text; startup note "leases held until a later gateway start confirms it is gone" (`services/deployments.py:615-626`).
- Covered by: `test_stop_when_exo_is_unreachable_is_not_clean` (asserts the unclean result), `test_agent_reports_exo_at_join_and_stops_it_on_pause`. Nothing checks that the Mac is still usable afterwards.

### T. Tuner

**T1. Deployment started from the dashboard mid-tune**
- Trigger: run `uv run hypercompute tune qwen2.5-14b-instruct-q6_k --max-candidates 4`. During a trial, click Run together and send a chat.
- Expected: create returns 409 `deployment_active` (`services/deployments.py:380-383`), chat returns 409 `tuning` (`services/deployments.py:289-292`), and `/v1/models` hides the trial (`app.py:417-424`). Between candidates the tuner re-creates within milliseconds (`services/tuner.py:408-409,433-441`), so a dashboard deploy practically never wins. If it does, the tune ends "another deployment started while tuning" (`services/tuner.py:443-444`).
- Evidence: UI errors; `GET /api/tune/<id>`.
- Covered by: `test_deployment_started_elsewhere_aborts_the_tune`, `test_a_tuner_trial_is_never_the_ready_run`, `test_v1_models_chat_and_benchmarks_skip_a_tuner_trial`.

**T2. Stop on the "Tuning trial" card** (TOP-10 #5)
- Trigger: during the tune, click Stop on the Run card once while candidate 2 loads and once while it is measured.
- Expected: see TOP-10 #5. During measurement, the cancelled request becomes `measurement_error` (`services/tuner.py:486-491`). The tune keeps going, and it saves a profile when any other candidate finished.
- Evidence: `GET /api/tune/<id>` (candidate status and error, `best_index`, `profile_id`); `GET /api/profiles?model_id=qwen2.5-14b-instruct-q6_k`; a fresh `POST /api/plans` with the Mac shows the replayed profile.
- Covered by: NONE.

**T3. Cancel mid-candidate**
- Trigger: Ctrl+C `hypercompute tune` (or `DELETE /api/tune/<id>`), once during the peer candidate's load and once during its measurement.
- Expected: the shielded `_stop` in `finally` stops the trial (`services/tuner.py:453-454,502-514`). The tune is "cancelled: cancelled by owner", unfinished candidates are `skipped` (`services/tuner.py:517-524`), and no profile is saved. The DELETE returns only after the trial has stopped (`services/tuner.py:545-553`).
- Evidence: `GET /api/tune/<id>`; `Get-Process llama-server`; the Mac logs "Client connection closed".
- Covered by: `test_cancel_stops_the_running_candidate`, `test_cli_ctrl_c_cancels_the_tune_on_the_host`.

**T4. A candidate that runs out of memory**
- Trigger: (a) on the G14, start the H5 hog right after a candidate begins loading. For a true CUDA OOM instead of a spill, set NVIDIA Control Panel → Manage 3D settings → "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" (a global driver setting; revert it after). (b) On the Mac, tune while its free memory is low (F4).
- Expected: a load failure marks the candidate `failed` with the deployment error (`services/tuner.py:447-449`), and the tune continues. With the default fallback the "OOM" candidate just runs slowly and scores badly. On the Mac, a swap stall can turn one candidate into a 1980 s wait (`services/tuner.py:192-195`) or kill the worker (F8). In that case every later peer candidate fails with `node_not_eligible` and T5 applies.
- Evidence: tune JSON; `llama-server-<dep>.log` per candidate; Task Manager Shared GPU memory.
- Covered by: `test_every_candidate_failing_fails_the_tune`, `test_ready_timeout_is_the_engines_own_load_and_warmup_budget`.

**T5. Peer lost mid-tune** (TOP-10 #5)
- Trigger: Mac Wi-Fi off for 20 s during the "peer overflow" candidate.
- Expected: that candidate fails, and later peer candidates fail at `_require_nodes` or validate (`services/deployments.py:453-462`, plus the N2 wedge). The RAM-tier candidate finishes, so the tune is `done` and **saves** the RAM split as the measured profile for a device set that includes the Mac (`services/tuner.py:335,345-353,526-543`).
- Evidence: as T2.
- Covered by: NONE.

**T6.** Gateway restart mid-tune is H3.

### D. Dashboard

**D1. Two tabs**
- Trigger: in two tabs, click Run together within 1 s. Then click Stop in tab A while tab B streams. Then click Create compute session in both.
- Expected: `_create_lock` serializes create, so one tab gets 202 and the other 409 (`services/deployments.py:373-383`). Stop in A interrupts B's stream (SSE error, run `cancelled`). Create reuses the active session and issues another valid invite (`services/sessions.py:94-104`).
- Evidence: both tabs' banners; runs.
- Covered by: `test_ready_requires_load_participation_and_warmup` (idempotency and 409).

**D2. Browser connection limit**
- Trigger: with vLLM 7B AWQ ready, open 3 tabs, run LONGCHAT-sized chats (2048 tokens) in each, then click Stop in a 4th tab.
- Expected: each tab holds one EventSource (`frontend/src/api/client.ts:282`) plus its chat fetch. uvicorn speaks HTTP/1.1 (h11), so 3 SSE plus 3 streams uses all 6 per-host connections. Chrome then queues every other request to 127.0.0.1:8000, Stop included, until a stream ends. Nothing in the code addresses this.
- Evidence: DevTools → Network → the Stop request's Timing shows "Stalled/Queueing".
- Covered by: NONE.

**D3. Reload during load, chat or tune**
- Trigger: reload the dashboard while a deployment loads, while a chat streams, and while a tune runs.
- Expected: load and tune are server tasks and are unaffected. A streaming chat's disconnect closes the upstream request and records `cancelled` (`services/streaming.py:339-341`). The page's chat transcript is lost (it is client state).
- Evidence: runs; deployment state continuity.
- Covered by: `test_client_disconnect_closes_upstream_and_records_cancel`.

**D4. SSE reconnect after a gateway restart**
- Trigger: Ctrl+C the host and start it again with the dashboard open.
- Expected: reconnect with 1-10 s backoff and no `since` (`frontend/src/api/client.ts:280-296`); the replay is deduplicated by `seq|at`, so the restarted counter is kept (`frontend/src/api/client.ts:225-246`). The first mutation gets a 403 and retries with a fresh token (`frontend/src/api/client.ts:147-151`).
- Evidence: the feed shows `gateway.startup` notes and the badge returns to live.
- Covered by: `client.test.ts` "accepts events from a restarted gateway that numbers from 1 again" and "drops an event replayed after a reconnect".

### C. Concurrency

**C1. 16 and 32 simultaneous chats on llama.cpp**
- Trigger: `uv run python $TP\failures_load.py chats --n 16 --max-tokens 128`, then `--n 32`, on (a) the 7B local, (b) the 7B with the Mac, (c) the 14B auto.
- Expected: 1 slot (`-np 1`, `adapters/llama_rpc.py:128`; no `max_concurrency`, so 1, `services/streaming.py:259-262`). FIFO waiters give up after 30 s. Streamed refusals arrive as **HTTP 200** with `data: {"error":{"code":"busy"}}` and no run record (`services/streaming.py:358-364`); non-streamed refusals are 429. From the `docs/RESULTS.md` rates: (a) about 17-18 of 32 served; (b) about 5-6; (c) about 6. The rest are `busy`.
- Evidence: the script summary and jsonl; the count of `ok` runs.
- Covered by: `test_capacity_one_keeps_one_request_at_a_time_and_429_after_the_wait`.

**C2. 16 and 32 chats on vLLM (G14 only)**
- Trigger: as C1 on vLLM 7B AWQ. Repeat with `--max-tokens 1024`.
- Expected: 8 or 16 slots (`adapters/vllm.py:86,534-539`). `docs/RESULTS.md` measured 32/32 ok at 8 and 16 users. With 1024 tokens, waiters past the first batches hit the 30 s wait and get `busy`.
- Evidence: as C1, plus nvidia-smi.
- Covered by: `test_capacity_four_runs_four_chats_at_once_and_the_fifth_waits`.

**C3. Past the connection cap: heartbeats starve** (TOP-10 #4)
- Trigger: with the 7B split (Mac) ready, `failures_load.py chats --n 110 --max-tokens 256`.
- Expected: 1 request runs and about 99 wait up to 30 s holding their sockets, so there are ≥100 connections and every new request gets 503. That includes the Mac's heartbeats (see N11), so the deployment fails in the middle of serving. Dashboard polls 503, and the page shows the gateway as unreachable (`frontend/src/App.tsx:73`).
- Evidence: `http_503` outcomes in the jsonl; agent log; host events; deployment error.
- Covered by: config check only.

**C4. Benchmark while chats run**
- Trigger: start `chats --n 32 --max-tokens 512`. Two seconds later, `Invoke-RestMethod -Method Post -Headers $H -ContentType application/json -Body '{"prompt_set":"both","repetitions":3}' "$GW/api/benchmarks"`.
- Expected: each benchmark repetition takes every slot, with a 300 s wait (`services/benchmarks.py:120`). Strict FIFO means chats queued behind an exclusive waiter wait too and get `busy` after 30 s (`services/streaming.py:93-131`). Repetitions and chats interleave and never overlap, so benchmark numbers stay clean.
- Evidence: `GET /api/benchmarks/<id>`; run timestamps by `kind`.
- Covered by: `test_benchmark_holds_every_slot_so_no_chat_overlaps_it`, `test_exclusive_waits_for_every_slot_and_blocks_new_chats`.

**C5. Stop while 8 streams are open**
- Trigger: vLLM `chats --n 16 --max-tokens 1024` (8 running, 8 queued), or llama.cpp `--n 8` (1 running, 7 queued). Press Stop after 3 s.
- Expected: `interrupt` cancels only the requests in `_inflight` (`services/streaming.py:283-288`); those runs are `cancelled`. **Queued waiters are not refused.** As holds are released they take slots and are sent to the stopping or stopped engine. They become `error` runs ("All connection attempts failed", or a reset) and the client sees `upstream_error`, not "deployment stopped". The DELETE returns after the vLLM STOP (up to 30 s grace, `adapters/vllm.py:89,676-698`).
- Evidence: statuses and errors of the 16 runs in `runs.jsonl`; DELETE latency.
- Covered by: `test_interrupt_cancels_every_inflight_request_of_the_deployment` (in-flight requests only). Waiters: NONE.

**C6. Chats arriving during a stop**
- Trigger: loop `chats --n 4` back to back while pressing Stop.
- Expected: a request passes `ready()` in `app.py:434`, then `chat_stream` calls `ready()` again inside the StreamingResponse generator, unguarded (`services/streaming.py:359`). The DeploymentError is raised after the 200 headers, so the client gets **HTTP 200 with no data** (outcome `no_done`, or a transport error) and no run record is written.
- Evidence: `no_done` outcomes; the request count exceeds the record count.
- Covered by: NONE.

---

## 4. Run order

1. **G14 alone** (no Mac needed): N11, C1(a), C2, C4, C5, C6, D1-D4, H1-H3 on loopback, H5, H6, H7, H10, H11, E3, E4(a/b), E5, E6. The cheapest way to expose TOP-10 #4, #10 and the runners-up.
2. **Mac alone** (agent running, host idle): F5, F6, F7, F10, F11, E2, E8.
3. **Pair, gentle**: N1, N7 (steps), F3, F8 mid-chat, F12 (×10), C1(b,c), C3, T1, T3.
4. **Pair, link-breaking**: N2, N3, N10, N5, F1, F9, T2, T5, H4, N4, N9. Put N2/N3 early in this block (TOP-10 #1) and keep Pause+Resume ready as the recovery.
5. **EXO** (if installed on the Air): E7, E8, E9.
6. **Destructive last**: H8, H9, F13, N6, N8(b). Revert firewall, IP, pf, dnctl and NVIDIA settings after each.

Take an `hc-snap <id>` before and after every injection, and keep `hc-events.sse`, the nvidia-smi CSV and
`~/hc-agent.log` running throughout.

---

## 5. What the next push needs to build (derived from the entries)

1. Worker liveness on the friend: a half-open connection while the host lists no deployment for the node means restart the worker (N2, F1). Have the agent restart the worker with backoff instead of quitting (F8).
2. A host-side progress watchdog: no SSE bytes or no load-phase change for X s while `worker_cpu_seconds` is flat fails the deployment (F9).
3. Endpoint drift: compare the heartbeat source IP with `worker.host`; let the agent re-bind and re-register with its credential (N5). Reconnect with the stored credential instead of burning an invite (F11).
4. Admission control that keeps node traffic alive: exempt heartbeats from the cap, put agents on a separate listener, time out idle sockets, and cap waiting chats (N11, C3).
5. Tuner: treat external candidate failures (peer lost, stopped by owner) as "inconclusive" and save no profile. Make Stop on a trial cancel the tune, or hide that Stop (T2, T5).
6. EXO: release the leases when the node itself stopped EXO (pause or leave), and treat "EXO process gone" as instance gone for a single Mac (E9, E7).
7. Agent pause/resume: one lock, a per-start log file, and `reconcile_orphans` must skip a worker this run started (F10). Start the worker with SIGPIPE ignored (`restore_signals=False` with SIG_IGN inherited) (F12).
8. macOS orphan worker: a parent-watch wrapper like vLLM's LAUNCH watchdog, or a kqueue NOTE_EXIT watcher (F7).
9. llama.cpp `validate` re-reads free VRAM, as vLLM's `_fresh_util` does (H6). Add real ggml abort formats to `detect_failure` (F8).
10. Inference slots: on interrupt, fail queued waiters with "deployment stopped"; guard `ready()` inside `chat_stream` with an SSE error plus a run record (C5, C6).
11. The vLLM watcher polls `/health` (a 503 means engine dead) and not only `process_alive` (E4b).
12. Gateway addresses: re-bind on a LAN change, or show only the bound addresses (N6). Fix the host diagnostic to name port 8000 and detect `python.exe` block rules (N8).
13. Friend telemetry in heartbeats: AC/battery %, Low Power Mode, thermal state and memory-pressure level, shown as warnings in plans (F2, F3, F4).
14. llama.cpp `-np N` with a `max_concurrency` hook, so a split serves more than one chat (C1).

---

## 6. Later hardware: what changes

- **Second Mac.** Two RPC workers make the `--rpc a,b` naming and remap paths live (`services/deployments.py:166-214`); lose one of two peers mid-chat. A real EXO ring: E7 and E9 with two Macs, plus the EXO pf table rules (`agent/join.py:493-508`). A Thunderbolt Bridge link-local join (169.254/16, `cli.py:225-235`). Both Macs on one AP contending for airtime (N7 with real contention). Profiles keyed by two peers (N5 doubles).
- **RTX 5090 desktop as a friend (Windows worker).** The Windows Firewall prompt for `ggml-rpc-server.exe` on Public; Hyper-V excluded port ranges (`config.py:9-13`, `cli.py:250-253`). The Job Object kills the worker with the agent (the opposite of F7). Discrete-VRAM headroom of 512 MiB (`services/deployments.py:154`). The friend gaming during a run (H5 on the peer). No SIGPIPE on Windows. As a host: vLLM on 32 GB runs C2 at 32-64 users.
- **ASUS GX10 (GB10, 128 GB unified, Linux arm64, sm_121).** The `linux-arm64-cuda13.4` variant is unvalidated (`docs/IMPLEMENTATION_STATUS.md` GX10 row). The Linux PDEATHSIG path has never run (`docs/IMPLEMENTATION_STATUS.md` supervision row). Repeat F7 there: PDEATHSIG fires when the **starting thread** exits (`agent/supervisor.py:96-112`), and `resume()` starts the worker from a `to_thread` executor thread (`agent/control.py:99`), so verify the worker neither dies early nor survives an agent kill. On unified memory, nvidia-smi may report memory as N/A, which `mib_or_none` turns into None (`agent/inventory.py:95-96`); check the llama.cpp budgets fallback and vLLM planning (`test_parse_nvidia_smi_fixture_and_bad_rows`). The UMA pool must be counted once in leases (`test_uma_pool_counted_once`).
