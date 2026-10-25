# Hypercompute stress limits: every cap, timeout and budget, with G14 + M4 Air probes

> **Added after review (2026-10-21).** The friend's M4 MacBook Air had already run three deployments with this G14 over campus Wi-Fi on the MVP build (`results/pair/20261021-campus-wifi-g14-m4air.json`, summarized in [../TEST_PLAN.md](../TEST_PLAN.md) section 0). Where this file says the macOS binary, the Mac inventory or a real Mac split has never run, read it against that evidence: the binary ran, the Metal working set is 12,124 MiB, the Mac's planner budget was 2.9-4.4 GiB, and its splits decoded at about 12 tok/s (14B) and a median 9.1 tok/s (Mistral 24B) on the old local-first order, where the Mac held the output head and sent 0.5-0.6 MB of logits back per token.

Repo `C:\Users\operator\hypercompute` at `dfd8cf9`. Read-only review on 2026-10-21. Paths are relative to
`backend/hypercompute/` unless they start with `docs/`, `scripts/` or `configs/`.

- **Where the expected outcomes come from.** They are read from the code. I did not run anything against a gateway
  or a GPU.
- **Where the planner numbers come from.** I ran the repo's own `make_plan` against the real GGUF headers on this
  machine, using synthetic budgets:
  - CUDA0 = 11,987,320,832 B (11.16 GiB) and host CPU = 8,015,210,087 B (7.46 GiB). Both are copied from the recorded
    plan in `results/20261021T104014Z-auto-gw-14b-auto.json`.
  - Mac budget = 6.67 / 4.0 / 2.0 GiB (section 1 explains these values).
- **Helper files.** The load tools are in `scripts/stress/`. All were syntax-checked; none was run against a
  gateway. The planner simulations (`predict.py`, `runs12.py`, `derive.py`, `matrix.py`, `sim_*.py`, `plan_sim*.py`) and their
  outputs (`rows.json`, `runs12.json`) were scratch work run against a 143 MB cache of real GGUF headers, so they are
  not in the repo. On test day the live prediction for any run is the plan `scripts/e2e_check.py` prints before it
  deploys.

| file | what it does |
|---|---|
| `sim_plans.py`, `sim_prior.py`, `sim_mac_bytes.py` | planner simulations (scratch, not in the repo). They read GGUF headers and cached summaries only |
| `flood.py` | sends N concurrent chats and counts the outcomes by (HTTP status, error code) |
| `mkprompt.py` | writes a chat body with about N prompt tokens, or a body of an exact byte size |
| `bounds_probe.py` | checks every request-schema bound and body limit just under, at and just over the limit. It never starts a tune or a benchmark |
| `hold.py` | holds N idle TCP connections to the gateway. This is the deterministic probe for W1 |

---

## 0. Test-day setup (use it in every probe below)

**G14 (PowerShell 5.1, repo root):**

```powershell
cd C:\Users\operator\hypercompute
$TP = "C:\Users\operator\hypercompute\scripts\stress"
$P = 8000; $B = "http://127.0.0.1:$P"          # test-day gateway port (default, config.py:9)
$T = (Invoke-RestMethod "$B/api/owner/bootstrap").owner_token
$H = @{ "X-Hypercompute-Owner" = $T }
function J($o) { $o | ConvertTo-Json -Depth 8 -Compress }
function Plan($r) { Invoke-RestMethod -Method Post "$B/api/plans" -Headers $H -ContentType "application/json" -Body (J $r) }
function Deploy($id) { Invoke-RestMethod -Method Post "$B/api/deployments" -ContentType "application/json" -Body (J @{ plan_id = $id }) -Headers ($H + @{ "Idempotency-Key" = [guid]::NewGuid().ToString() }) }
function Dep($id) { Invoke-RestMethod "$B/api/deployments/$id" }
function StopDep($id) { Invoke-RestMethod -Method Delete "$B/api/deployments/$id" -Headers $H }
function WaitDep($id) { do { Start-Sleep 1; $d = Dep $id; "{0:HH:mm:ss} {1} {2}" -f (Get-Date), $d.state, $d.progress } while ($d.state -notin "ready","failed","stopped"); $d | Select-Object state,error,cleanup,participation_detail }
function LastRun { (Invoke-RestMethod "$B/api/runs?limit=1")[0] | Select-Object kind,status,error,ttft_ms,total_ms,prompt_tokens,completion_tokens }
function Chat($file) { curl.exe -sN "$B/v1/chat/completions" -H "X-Hypercompute-Owner: $T" -H "Content-Type: application/json" --data-binary "@$file" -w "`nHTTP %{http_code} total %{time_total}s`n" }
$MAC = (Invoke-RestMethod "$B/api/nodes" | Where-Object { -not $_.is_local } | Select-Object -First 1).id
```

**Watchers on the G14.** Keep these open in their own windows for every test.

```powershell
# window 2: heartbeat age per friend, once a second (stale at age > 15 s, config.py:17)
while ($true) { Invoke-RestMethod "$B/api/nodes" | Where-Object { -not $_.is_local } | ForEach-Object { "{0:HH:mm:ss} {1,-11} age={2,5:n1}s {3}" -f (Get-Date), $_.state, $_.heartbeat_age_s, $_.state_reason }; Start-Sleep 1 }
# window 3: event stream (keep-alive every 15 s, services/events.py:52)
curl.exe -sN "$B/api/events"
```

**M4 Air (zsh):**

```bash
caffeinate -dims &                                   # keep the Air awake (except in the sleep tests)
pgrep -fl "hypercompute join"; pgrep -xl ggml-rpc-server
tail -f ~/.hypercompute/agent/logs/ggml-rpc-server.log   # the Mac's RPC allocations
```

**Baselines to record before the limit tests.** The results below cannot be read without them.

1. Network RTT (G14): `ping -n 50 <MAC_IP>`. The planner assumes `HOP_MS = 4.0` (services/planner.py:45). Record
   the throughput too, for example by timing a 1 GB copy. The load-time floors in W6 need it.
2. Mac memory: `(Invoke-RestMethod "$B/api/nodes" | ? { -not $_.is_local }).inventory.memory_pools` gives the
   uma `free_bytes` (psutil `available`). Read Metal free from `.verification.rpc_devices[0].free_bytes`.
   The planner's Mac budget is `min(10.0 GiB, min(metal_free, available) - 4.0 GiB)` (section 1).
3. G14 free disk: `Get-PSDrive C`. C: had **26 GiB free (98% full)** at review time. See W16.

---

## 1. Budget math for this pair (what the planner will actually allow)

| node | default from code | planner budget at plan time |
|---|---|---|
| G14 CUDA0 (RTX 5070 Ti, 12,227 MiB) | `max_bytes = total - 512 MiB = 11.44 GiB` (agent/contribution.py:38-40) | `min(11.44 GiB, free_VRAM - 512 MiB)` (services/deployments.py:112-115). Measured: **11.16 GiB** |
| G14 host CPU tier (32 GB RAM) | reserve = `max(4 GiB, 20% of RAM)` ≈ **6.3-6.4 GiB** (agent/contribution.py:21-24). The `host` command has no `--reserve` flag (cli.py:447-453) | `available - reserve - 512 MiB` (services/deployments.py:123-132). Measured: **7.46 GiB** |
| M4 Air MTL0 via RPC (16 GB unified) | `max_bytes = min(0.65×16, 16-6) = 10.0 GiB`, reserve `max(4, 3.2) = 4.0 GiB` (agent/contribution.py:24,43). The agent prints "10.0 GiB, keeps 4.00 GiB" | `min(10.0, min(metal_free≈10.7, available) - 4.0)` (services/deployments.py:148-156) |

The Mac's planner budget depends on how much memory is `available` at plan time:

| Mac `available` at plan time | Mac planner budget |
|---|---|
| ≥ 10.7 GiB (idle, nothing open) | **6.67 GiB** |
| 9 GiB | 5.0 GiB |
| 8 GiB | 4.0 GiB |
| 6 GiB (browser open) | **2.0 GiB** |

So the budget the agent advertises (10.0 GiB) is never what the planner uses. See W2.

Planner output on the real GGUF headers (`sim_plans.py`, ctx 4096). "L" means layers. The figures are estimated
bytes against the budget.

| model (weights) | strategy | Mac 6.67 GiB | Mac 4.0 GiB | Mac 2.0 GiB |
|---|---|---|---|---|
| 0.5B Q4_K_M (0.46 GiB) | all_selected | Mac 1L + CUDA0 23L | same | same |
| 7B Q4_K_M (4.36 GiB) | auto | CUDA0 28L 4.88 GiB (Mac unused) | same | same |
| 7B Q4_K_M | all_selected | Mac 1L 0.61 GiB + CUDA0 27L | same | same |
| 14B Q6_K (11.29 GiB) | **auto** | **CPU 4L + CUDA0 44L (Mac unused, W7)** | same | same |
| 14B Q6_K | all_selected / overflow=peers | Mac 4L 1.39 GiB + CUDA0 44L **11.15 of 11.16 GiB** | same | same |
| 14B Q8_0 (14.62 GiB, scaled) | auto | Mac 15L 4.80 GiB + CUDA0 33L | CPU 15L + CUDA0 33L | CPU 15L + CUDA0 33L |
| 14B Q8_0 | all_selected | as auto | CPU 3L + Mac 12L 3.94 GiB + CUDA0 33L | CPU 10L + Mac 5L + CUDA0 33L |
| 24B Mistral Q4_K_M (on disk, not in manifest) | auto | Mac 10L 3.80 GiB + CUDA0 30L | Mac 10L 3.80 of 4.00 GiB | CPU 10L + CUDA0 30L |
| 32B Q3_K_M (not in manifest) | auto | Mac 22L 5.66 GiB + CUDA0 42L | CPU 22L + CUDA0 42L | same as 4.0 |
| 32B Q4_K_M (18.49 GiB) | auto / all_selected | **CPU 9L + Mac 21L 6.43 GiB + CUDA0 34L** | CPU 18L + Mac 12L + CUDA0 34L | **gpu_disk**: CPU 25L paged + Mac 5L + CUDA0 34L |
| 32B Q4_K_M | overflow=peers | infeasible (Mac needs 9.15 GiB) | infeasible | infeasible |
| 32B Q4_K_M | local_only | **gpu_disk (capacity mode): CPU 30L paged + CUDA0 34L** | | |

How the 14B Q6_K plan changes with context (all_selected, Mac 6.67 GiB):

| ctx_size | placement |
|---|---|
| 8192 | Mac 7L + CUDA0 41L |
| 16384 | Mac 12L 3.76 GiB + CUDA0 36L |
| 32768 | CPU 1L + Mac 18L 6.52 GiB + CUDA0 29L |
| 65536 | CPU 14L + Mac 13L + CUDA0 21L |
| ≥ 131072 | infeasible |

7B local_only: ctx 131072 → gpu_ram (CPU 2L). ctx 262144 → **gpu_disk, capacity mode** (W8).

Bytes sent to the Mac over RPC, and the link speed below which `LOAD_TIMEOUT_S = 1800 s` expires
(`sim_mac_bytes.py`; an estimate, because the load also includes the local part):

| plan | bytes to the Mac | load fails below |
|---|---|---|
| 14B Q6_K all_selected | 0.90 GB | 0.50 MB/s |
| 24B auto | 3.39 GB | 1.89 MB/s |
| 14B Q8_0 auto | 4.39 GB | 2.44 MB/s |
| 32B Q3_K_M auto | 5.19 GB | 2.88 MB/s |
| 32B Q4_K_M 3-way | 6.03 GB | **3.35 MB/s** |
| 32B Q4_K_M overflow=peers (Mac budget 10.6) | 8.80 GB | **4.89 MB/s** |
| 14B Q6_K ctx 32768 | 4.07 GB | 2.26 MB/s |

---

## 2. Limits that look wrong for real hardware (the build list for the next push)

Most severe first. Each item names the probe in section 3 that proves it on hardware.

**W1. The 100-connection cap is shared between the owner's traffic and the friends' heartbeats, and any LAN host can
fill it without credentials.**
- Where: `config.py:22` and `cli.py:167`. uvicorn (h11, the installed protocol, since httptools is not installed)
  counts every open connection from `connection_made` on, with no header timeout (`.venv/.../uvicorn/protocols/http/h11_impl.py:92-95`).
  It answers 503 when `len(connections) >= 100 or len(tasks) >= 100` (`h11_impl.py:227-233`). That reply is a plain
  503, sent before the app and its auth, so it has no security headers.
- What breaks:
  - 100 idle TCP connections to `<G14_IP>:8000` from any device on the LAN make every later request get 503: the
    friends' heartbeats, joins and the dashboard.
  - The Mac's heartbeat failures (`agent/join.py:351-360`) turn into **stale** after 15-17 s, and every deployment that
    uses a friend **fails**.
  - The same happens with a local load test. Inference waiters have no queue cap
    (`services/streaming.py:100,110-113`), and each one holds its connection for up to `BUSY_WAIT_S = 30 s`.
- Recommendation:
  - Add a header-read timeout and a per-source-IP connection cap in front of the app.
  - Serve `/api/sessions/join` and `/api/nodes/*` on their own listener, or under their own limit.
  - Cap each deployment's waiting queue, for example 4 × capacity, and answer 429 at once past it.
- Probe: P-CONN.

**W2. The Mac's (and the GX10's) reserve is subtracted from memory that is already net of the OS.**
- Where: `services/deployments.py:148-156` subtracts `reserve_bytes` (4 GiB on a 16 GB Mac,
  `agent/contribution.py:21-24`) from `min(Metal working-set free, psutil available)`. Metal's working set is already
  about 67% of RAM, and `available` is already net of open apps. EXO does the same in `adapters/exo.py:179-189`.
- Effect: the advertised 10.0 GiB becomes 6.67 GiB at best, and about 2 GiB with a browser open.
  - 32B Q4_K_M overflow=peers (9.15 GiB) can never be planned with the defaults.
  - 14B Q8_0 needs `available ≥ 8.8 GiB` to use the Mac.
  - EXO 7B 4-bit (3.99 + 1 GiB `MLX_OVERHEAD`) needs `available ≥ 9.0 GiB`.
- Recommendation:
  - For unified pools, use `min(max_bytes, metal_free - 512 MiB, available - min(reserve, 1.5 GiB))`, or apply the
    reserve to total RAM instead.
  - Print the effective budget in the `join` output and on the dashboard card.
- Probe: P-MACBUDGET.

**W3. llama.cpp has one silence budget, 180 s, for every tier, context size and prompt length.**
- Where: `config.py:27`, `adapters/llama_rpc.py:65,72`. There is no `idle_timeout_for`.
- What breaks: llama-server sends nothing until the first token, so the whole prefill counts as silence. A
  30k-token prompt on 14B ctx 32768 (Mac 18L + CPU 1L), or on 32B 3-way, can pass 180 s. It is then reported as
  `timeout` even though the engine is healthy.
- Recommendation:
  - Add `idle_timeout_for(plan)` that grows with `ctx_size` and tier, for example 180 s plus ctx / (a 100 tok/s floor).
  - Or ask llama-server for prompt-progress events. Check whether b11770 supports `return_progress` before relying
    on it.
- Probe: P-PREFILL.

**W4. A hung friend worker is never detected.**
- Where: the agent reports `running` whenever the process exists (`agent/join.py:254-260`). The gateway's `_watch`
  checks only the local llama-server (`services/deployments.py:493-507`).
- What breaks:
  - A stopped or frozen `ggml-rpc-server` (Metal hang, `kill -STOP`, an App Nap-like stall) keeps the node
    `eligible`.
  - A load then waits the full **30 min** (`LOAD_TIMEOUT_S`).
  - Every chat waits **180 s**, and the deployment stays `ready` while it is broken.
- Recommendation:
  - Fail the deployment after 2 consecutive idle timeouts on a peer split.
  - Have the agent check its own worker: a TCP connect to its port, or `worker_cpu_seconds` advancing during
    requests. The value is already sent and is unused for liveness (`schemas.py:195`).
- Probe: P-HUNG.

**W5. The heartbeat cadence leaves little slack against the 15 s stale threshold under load.**
- How a cycle works: measure, then POST with an 8 s timeout, then wait 3 s (`agent/join.py:340-375`). Two slow
  heartbeats in a row pass 15 s (`config.py:17`).
- What breaks:
  - A friend with an NVIDIA GPU (the RTX 5090 later) runs `nvidia-smi` on every heartbeat with a **20 s** timeout
    (`agent/inventory.py:36,87-89,230`). One slow `nvidia-smi` can make that friend stale on its own.
  - On the M4 Air the risk is a multi-GB RPC upload saturating the same Wi-Fi.
- Recommendation:
  - Heartbeat on a fixed clock, and measure in the background.
  - Cap `nvidia-smi` at about 3 s in the heartbeat path.
  - Make staleness "N missed intervals", or about 30 s while a deployment is loading.
- Probes: P-HBLOAD, P-HBGAP.

**W6. The llama.cpp load budget is a fixed 30 min, whatever the bytes going to peers.**
- Where: `config.py:25`. `LlamaRpcAdapter` has no `load_timeout_s` (`adapters/llama_rpc.py:168`).
- Floors (section 1): 3.35 MB/s for 32B 3-way, 4.89 MB/s for overflow=peers. A phone hotspot, which RUNBOOK §4
  suggests, can fall below that. Every tune candidate pays the transfer again unless the friend joined with `--cache`.
- Recommendation: `load_timeout_s(plan) = 1800 + peer_weight_bytes / 1 MiB/s`, which mirrors vLLM's
  `DOWNLOAD_FLOOR_BPS` in `adapters/vllm_output.py:30,153-156`. Also show the bytes sent so far.
- Probe: P-LOAD.

**W7. `auto` never uses the Mac for 14B Q6_K, because the CPU decode prior is lower than the G14's measured cost.**
- Where: the prior is 12.5 ms/GB (`services/planner.py:44`). The G14 measured about 18 ms/GB
  (`docs/IMPLEMENTATION_STATUS.md`, Known gaps).
- Predicted decode at the prior:
  - RAM tier: 11.29 + 27.92 = **39.2 ms/token**.
  - Mac: 9.80 + 4.0 hop + 27.92 = **41.7 ms/token**.
- At 18 ms/GB `auto` picks the Mac (`sim_prior.py`). So the test day's most important pair test only happens with
  `all_selected`, `overflow=peers` or after a `tune`.
- Recommendation: set the CPU prior per host from a measurement (or tune once and persist it). Show both costs in the
  plan, as the overflow warning already does (`services/planner.py:392-398`).
- Probe: P-AUTO.

**W8. `ctx_size` is not capped at the model's trained context.**
- Where: the schema allows up to 262144 (`schemas.py:348`). The planner only warns above 32768
  (`services/planner.py:473-474`). vLLM gets `--max-model-len` unchecked (`adapters/vllm.py:229,383`).
- What breaks:
  - vLLM 7B AWQ at ctx 40960 plans as feasible, then fails at startup: Qwen2.5-7B's `max_position_embeddings` is
    32768. That is vLLM's behavior, not this code's, so confirm it.
  - llama.cpp 7B at ctx 262144 silently becomes a **capacity-mode gpu_disk** plan. The GGUF's `context_length` is
    131072.
- Recommendation: refuse or clamp above GGUF `context_length` or HF `max_position_embeddings`. Never let context alone
  push a plan into capacity mode without a warning that says so.
- Probe: P-CTX.

**W9. Fixed bind addresses: an IP change breaks the session.**
- Where: the gateway binds the LAN IPs it saw at start (`cli.py:143-159,195`). The worker binds the advertise IP
  (`agent/join.py:188`).
- What breaks:
  - A DHCP or hotspot reconnect that changes the G14's IP leaves friends heartbeating into nothing. They go stale
    after 15 s and stop sharing after 5 min.
  - A changed Mac IP keeps its heartbeats working, because the source IP is not checked, but the worker becomes
    unreachable. The node is re-probed every ~55 s forever.
- Recommendation: listen on 0.0.0.0 and filter in middleware (private sources only, as join already enforces), or
  re-bind on an address change. Let the agent re-advertise its worker address with its credential.
- Probe: P-IP.

**W10. llama.cpp plans do not re-check free VRAM at deploy time, and the 14B Q6_K plan has 15 MiB of estimate
margin.**
- Where: `adapters/llama_rpc.py:102-117` checks only the model file and the peer probe. vLLM does re-check, with
  `_fresh_util` in `adapters/vllm.py:495-511`.
- Margin: CUDA0 is planned at 11,971,186,688 of 11,987,320,832 B. A stored plan stays deployable for 10 min
  (`services/deployments.py:368`).
- What breaks: any app that takes VRAM after planning turns the deploy into a CUDA OOM during load.
- Recommendation: re-compute local budgets in `validate()`, and re-plan when free VRAM dropped.
- Probe: P-VRAM.

**W11. A warmup timeout is reported as the bare text "TimeoutError".**
- Where: `services/deployments.py:438-448`. `str(TimeoutError())` is empty, so `__class__.__name__` is used.
- Recommendation: say "warmup did not finish within {budget} s (a peer may be hung)".
- Probe: P-HUNG (the warming case).

**W12. Some config does not behave as it claims.**
- `INVITE_MAX_FAILED_ATTEMPTS = 5` (`config.py:20`) is never enforced. `services/sessions.py:148` reads
  `failed_attempts` and nothing writes it. A wrong code matches no row anyway, so per-invite counting needs a design
  change. The real protection is 10 joins per minute per IP, a 2^50 code space and the 15 min TTL.
- The `config.py:1` docstring says "Everything here is overridable by environment variables", but only paths are
  (`config.py:33-35`). Stress tests cannot change `STALE_AFTER_S` and similar constants without a code edit.
  Recommendation: `HC_*` env overrides for every timing constant.
- `JoinResponse.heartbeat_interval_s` (`services/nodes.py:190`) is ignored by the agent, which uses its own constant
  (`agent/join.py:373`).

**W13. An SSE event over 4 MiB raises `ValueError`, and `stream_events` does not catch it.**
- Where: `sse.py:22,32-33` raises it. `services/streaming.py:339-350` catches only httpx, `EngineAdapterError` and
  `CancelledError`.
- Effect: the client sees a torn connection, and the run is recorded as `error` with `error=None`.
- Coverage: a contract test only.

**W14. Later hardware.**
- GX10 / GB10: nvidia-smi reports memory as `[N/A]`. `parse_nvidia_smi` skips those rows (`adapters/vllm.py:107-117`),
  and the fallback rows have `total = 0`, so utilization is 0. **vLLM is always infeasible on GX10.**
- 5090:
  - `MAX_UTILIZATION 0.90` (`adapters/vllm.py:75`) leaves 3.2 GiB idle on a 32 GiB headless card.
  - `MAX_NUM_SEQS 8/16` (`adapters/vllm.py:86`) caps throughput far below what a 5090 or GX10 can batch.
- `KV_SHAPES` covers only qwen2.5-7b (`adapters/vllm.py:83-85`). The fallback of 20 KiB per token per billion
  parameters over-reserves KV about 2.5× for Qwen2.5-32B.

**W15. The runbook is wrong about stale nodes.**
- `docs/RUNBOOK.md` §5 says a stale node must "rejoin with a new code". In the code, a heartbeat that resumes within
  5 min auto-recovers it: stale → verifying → eligible (`services/nodes.py:236-238`).

**W16. The test-day disk is too small for two capacity models.**
- C: had 26 GiB free. A model download needs `total + 2 GiB` free (`services/models.py:183-184`):
  - 32B Q4_K_M needs 20.49 GiB and leaves about 5.5 GiB.
  - 14B Q8_0 needs 16.62 GiB.
- You cannot have both unless you delete something first. Mistral 24B is 14 GiB and 14B Q6_K is 12 GiB.
- The WSL vhdx (vLLM) and the pagefile also live on C:.

---

## 3. Inventory by subsystem: limit, location, behavior at the limit, probe, expected outcome

Format: **ID name**: `file:line` = value. *At the limit*: what the code does. *Probe*: under / at / over on the G14
+ M4 Air. *Expect*: the outcome predicted from the code.

### 3.1 Gateway HTTP surface

**L1 Chat body limit**: `app.py:57,88-93` = 4096 KiB (4,194,304 B).
- *At the limit*: the check is `> limit`. It runs on Content-Length (`app.py:134`) and again while counting a chunked
  body (`app.py:145`), before routing and auth. The answer is 413
  `{"detail":"request body is over the 4096 KiB limit","code":"body_too_large"}`, with the security headers.
- *Probe*: `uv run python $TP\bounds_probe.py --port $P`. It sends 4,194,304 and 4,194,305 bytes, plus a chunked
  4,194,305. To see the real engine path:
  `uv run python $TP\mkprompt.py 0 $env:TEMP\b4m.json --bytes 4194304 --no-stream; Chat $env:TEMP\b4m.json`
- *Expect*:
  - At the limit, the gateway passes the body. llama-server answers HTTP 400 (prompt longer than ctx), and the
    gateway returns **400** with `upstream_error`, keeping the engine's status (`services/streaming.py:426-429`).
    Streaming gets HTTP 200 plus an SSE `upstream_error`.
  - One byte over gives **413**, chunked or not.

**L2 Node body limit**: `app.py:56,58` = 256 KiB for `/api/sessions/join` and `/api/nodes/{id}/heartbeat`.
- *Probe*: `bounds_probe.py` (262,144 and 262,145 bytes). A real Mac join inventory is about 7 KB
  (`tests/fixtures/inventory_windows_g14_raw.json` is 7.3 KB raw).
- *Expect*: 422 at the limit (schema error), 413 over it. Not reachable by a real agent.

**L3 Default body limit**: `app.py:55` = 64 KiB for every other route, including `/api/plans` and `/api/tune`.
- *Probe*: `bounds_probe.py` (65,536 and 65,537 bytes).
- *Expect*: 422 at the limit, 413 over it. `node_ids` has no `max_length` (`schemas.py:346,579`), so this limit is
  its only bound: about 1,800 32-hex ids.

**L4 Connection cap**: `config.py:22`, `cli.py:167` = 100 (uvicorn `limit_concurrency`). The listen backlog is 128
(`cli.py:158`).
- *At the limit*: a complete request gets a plain-text `503 Service Unavailable` with `connection: close` once 100
  connections are open. Idle sockets that never sent a byte count too, and there is no header timeout. Keep-alives
  close 5 s after their last response. This happens before the app, so there are no security headers. Heartbeats are
  included (W1).
- *Probe P-CONN, deterministic version* (14B Q6_K all_selected ready, Mac in the plan, watchers on). Count the
  connections that are already open on the G14 (dashboard SSE, the Mac's keep-alive):
  `$O = (Get-NetTCPConnection -LocalPort $P -State Established).Count`. Then:
  - Under: `uv run python $TP\hold.py (98 - $O) 127.0.0.1 $P 40`
  - At: `uv run python $TP\hold.py (100 - $O) 127.0.0.1 $P 40`
  - Over: `python3 hold.py 100 <G14_IP> 8000 40`, run **from the Mac**, to show it needs no token. Copy `hold.py`
    over first.
- *Expect*:
  - Under: nothing changes, and the heartbeat age stays under 5 s.
  - At / over: `Invoke-RestMethod "$B/api/health"` throws 503 "Service Unavailable". The Mac prints
    "host unreachable (HTTPStatusError)". At heartbeat age 15-17 s the node becomes **stale**, and the deployment
    becomes **failed** with "peer xxxxxxxx lost: no heartbeat for 16s" (or 17s). A chat in flight gets SSE
    `upstream_error` "interrupted: deployment stopped or a peer was lost".
  - When `hold.py` closes the sockets, heartbeats resume, and the node goes through verifying to eligible.
- *Probe P-CONN, realistic version*: `uv run python $TP\flood.py --port $P -n 40 --max-tokens 512`, then `-n 120`.
- *Expect*:
  - `-n 40`: 1-2 requests answer `status=200 code=None` (about 20 s each at about 25 tok/s). The rest answer
    `status=200 code=busy` after 30 s. No 503s, and the deployment stays ready.
  - `-n 120`: some requests get `status=503`. Whether the Mac's heartbeats also get 503 depends on whether the open
    connections stay at 100 or more when a heartbeat arrives, so the result can go either way. Record the 503 count
    and the maximum heartbeat age.

**L5 Owner-only routes**: `app.py:199-209`. The client and the Host header must be loopback, and writes also need
the token.
- *Probe*: from the Mac, `curl -s -o /dev/null -w "%{http_code}\n" http://<G14_IP>:8000/api/nodes`. From the G14,
  call `/api/plans` without the token.
- *Expect*: 403 `owner_only` in both cases. This is not a capacity limit; it is here so the stress scripts are known
  to run on the G14 only.

**L6 Runs list clamp**: `app.py:366-367`, `limit` clamped to 1..500. Deployments, benchmarks and tunes lists are
fixed at 50 (`services/deployments.py:264`, `services/benchmarks.py:85`, `services/tuner.py:295`).
- *Probe*: `bounds_probe.py` (`limit` = 0, 1, 500, 501).
- *Expect*: at most 1, 1, 500 and 500 rows. The 51st oldest deployment drops off the list but is still readable by id
  (`services/deployments.py:267-274`).

**L7 Event bus**: `services/events.py:17` capacity 500 events in memory. Keep-alive every 15 s
(`services/events.py:52-69`).
- *At the limit*: a dashboard that reconnects with `Last-Event-ID` older than 500 events misses the gap. There is no
  error; the events are simply gone.
- *Probe*: a tune with `--max-candidates 8` publishes many events. Disconnect `curl.exe -sN $B/api/events` for its
  duration, then reconnect with `-H "Last-Event-ID: <last seq>"`.
- *Expect*: when more than 500 events happened in between, the first event you receive has `seq` greater than
  last + 1.

### 3.2 Sessions, invites, join

**L8 Invite TTL**: `config.py:19` = 900 s (`services/sessions.py:111,156`, `expires_at <= now` means expired).
- *Probe*: issue an invite at T0 (dashboard **New invite**, or
  `Invoke-RestMethod -Method Post "$B/api/sessions/<sid>/invites" -Headers $H`).
  - Under: on the Mac, `uv run hypercompute leave`, then `uv run hypercompute join --host <G14_IP>:8000 --code <code>`
    at T0+14:30.
  - Over: use a second invite issued at T0 after T0+15:05.
  - Reuse: join again with a code that was already used.
- *Expect*: under, join succeeds. Over, `join refused: this invitation code expired` (403 `code_expired`). Reuse gives
  403 `code_used`.
- *Note*: a first-time `engine install` on the Mac over slow Wi-Fi can outlast 15 min. Issue the code after the
  install finishes.

**L9 Join rate limit**: `config.py:21` = 10 per 60 s per client IP (`services/sessions.py:52-65,139-140`).
- *At the limit*: the counter is checked in `redeem()`, after the worker-address and node-id checks
  (`services/nodes.py:164-173`). Hits count whether the join succeeds or not. The 11th call in 60 s gets 429
  `rate_limited`.
- *Probe* (Mac, repo root, with the right IPs filled in):

  ```bash
  uv run python - <<'EOF'
  import httpx, uuid
  from hypercompute.agent import inventory
  from hypercompute.agent.contribution import default_contribution
  inv = inventory.collect()
  body = {"code": "AAAAA-AAAAA", "node_id": uuid.uuid4().hex, "display_name": "ratetest",
          "inventory": inv.model_dump(mode="json"), "contribution": default_contribution(inv).model_dump(mode="json"),
          "worker": {"host": "<MAC_IP>", "port": 47052}, "agent_version": "test"}
  for i in range(12):
      r = httpx.post("http://<G14_IP>:8000/api/sessions/join", json=body, timeout=10)
      print(i + 1, r.status_code, r.json().get("code"))
  EOF
  ```

- *Expect*: calls 1-10 get `403 invalid_code` and calls 11-12 get `429 rate_limited`. After 61 s the next call gets
  403 again. A worker address that does not match gives `400 worker_address_mismatch` and is **not** rate-limited.

**L10 Invite failed-attempt cap**: `config.py:20` = 5. **Never enforced** (W12). Nothing to probe; add a contract test
once it is designed.

**L11 Code format**: `services/sessions.py:22-23,142`. 10 characters from a 32-letter alphabet without 0/O/1/I. The
schema allows 4..32 characters (`schemas.py:158`).
- *Probe*: `bounds_probe.py` (lengths 3 and 33). Also try a code with an `O`.
- *Expect*: lengths 3 and 33 give 422 from the schema, which is not rate-limited. A well-formed length with an `O`
  gives 403 `invalid_code` and is rate-limited.

**L12 Join address rules**: `services/nodes.py:163-171`. The worker host must equal the source IP and must be a
private or link-local address.
- *Probe*: on the Mac, `join --advertise-ip 10.99.99.99`.
- *Expect*: 400 `worker_address_mismatch`, with a fix that suggests `--advertise-ip <source ip>`.

### 3.3 Heartbeats, staleness, probes (host and agent)

**L13 Heartbeat interval**: `config.py:16` = 3.0 s, counted after each POST returns (`agent/join.py:340-375`). The
POST timeout is 8.0 s (`agent/join.py:350`).
- *At the limit*: the real period is 3 s plus the measurement plus the RTT. Up to 11 s when the host times out.

**L14 Stale threshold**: `config.py:17` = 15.0 s. The host monitor ticks every 2.0 s (`services/nodes.py:280-287`)
and checks `age > 15` (`services/nodes.py:254`).
- *At the limit*: the node becomes stale at an age of 15-17 s. If it is busy, `on_lost` cancels the start task or
  fails the deployment, stops llama-server and releases leases (`services/deployments.py:570-574`). When heartbeats
  resume, stale goes to verifying, re-probes and becomes eligible (`services/nodes.py:236-238`).
- *Probe P-HBGAP* (14B Q6_K all_selected ready, a 1024-token stream running). On the Mac, freeze only the agent
  process. The worker keeps running.
  - Under: `pkill -STOP -f "hypercompute join"; sleep 10; pkill -CONT -f "hypercompute join"`
  - At: the same with `sleep 13`
  - Over: the same with `sleep 20`
- *Expect*:
  - Under: the maximum age is about 10-13 s, nothing changes, and the stream completes.
  - At: nondeterministic, because of the up-to-3 s phase plus the 2 s tick. Record it.
  - Over: stale at an age of 15-17 s. The deployment is `failed` with "peer xxxxxxxx lost: no heartbeat for 16s".
    The stream ends with SSE `upstream_error` "interrupted...". The cleanup reads "...terminated...; released 2
    lease(s)". After CONT the node passes through verifying ("heartbeat resumed; re-checking worker") and becomes
    eligible within the probe time, about 1-3 s.
- *Variants*:
  - Mac Wi-Fi off for 20 s: `networksetup -setairportpower en0 off; sleep 20; networksetup -setairportpower en0 on`.
  - Close the lid with no `caffeinate`.
  - Both give the same failure at 15-17 s. On wake, if the IP is unchanged, the node re-verifies.
  - Python's `time.monotonic` on macOS does not advance during sleep, so the agent's 5 min host-loss timer counts
    only awake time.

**L15 Stale under load (P-HBLOAD)**: the same constants as L13 and L14, measured while 6 GB cross the Wi-Fi.
- *Probe*: deploy 32B Q4_K_M all_selected (the Mac gets 6.03 GB). Log the heartbeat watcher to a file and take the
  maximum `age`.
- *Expect*: the code has no special handling. If any age goes over 15 s, the deployment fails during `loading` with
  "no heartbeat for Ns". Pass criterion: maximum age under 8 s. If it fails, that is evidence for W5.

**L16 Host-unreachable auto-stop (agent)**: `config.py:18` = 300 s (`agent/join.py:330-338,355-362`).
- *Probe*: press Ctrl+C on the G14 gateway at T0.
  - Under: restart the gateway with `uv run hypercompute host` at T0+4:00.
  - Over: leave it down past T0+5:15.
  - Variant: G14 Wi-Fi off instead of Ctrl+C.
- *Expect*:
  - The Mac prints "host unreachable (ConnectError)" at failures 1, 5 and every 20th.
  - Under: the next heartbeat succeeds. The credential survives a restart (`services/nodes.py:108-120`). The node
    goes stale, then verifying, then eligible.
  - Over: at T0+300..303 s the Mac prints "the host has been unreachable for over 5 minutes: stopping sharing...".
    The worker stops, and the goodbye tries 3 times (`agent/join.py:377-392`), printing "Could not reach the host to
    say goodbye".
  - Wi-Fi-off variant: each POST waits for the 8 s timeout, so the stop happens at T0+300..311 s.

**L17 G14 sleep blames the peer (P-G14SLEEP)**: the staleness check uses wall clock (`services/nodes.py:252-253`).
- *Probe*: with a Mac deployment ready, put the G14 to sleep (Start, then Sleep) for 60 s and wake it.
- *Expect*: on wake the Mac's age is about 60 s or more. It goes stale, and the deployment fails with
  "peer ... lost: no heartbeat for 6Xs", even though the host slept. Recommendation: detect a monitor gap larger than
  10 s and treat it as a host suspend.

**L18 Probe timeout**: `config.py:24` = 25 s. This is the handshake `llama-server --rpc <ep> --list-devices`
(`adapters/llama_rpc.py:82-100`). It also runs inside `validate()` at every deploy (`adapters/llama_rpc.py:108-116`).
- *Probe (P-PROBE)*: on the Mac, `pkill -STOP -x ggml-rpc-server`. The kernel still accepts TCP connections, but HELLO
  is never answered. Then on the G14:
  `Measure-Command { Invoke-RestMethod -Method Post "$B/api/nodes/$MAC/verify" -Headers $H }`. Afterwards run
  `pkill -CONT -x ggml-rpc-server`.
- *Expect*:
  - The call returns after about 25 s.
  - The node is `ineligible` with "worker unreachable from host", and the detail starts "probe timed out after 25s".
  - The suggested fix is the **firewall** text (`services/nodes.py:341-342`), which is misleading for a hung worker.
  - Under, with no STOP, verify returns in about 1-3 s and the node is eligible.

**L19 Re-probe backoff**: `services/nodes.py:37` = 30 s, measured from the end of the previous probe
(`services/nodes.py:269-278`).
- *Probe*: keep the worker STOPped. Every 5 s print
  `(Invoke-RestMethod "$B/api/nodes" | ? { -not $_.is_local }).verification.checked_at`. Then CONT the worker.
- *Expect*: `checked_at` moves about every 55-57 s (30 s backoff, up to a 2 s tick, the 25 s probe). After CONT the node
  is eligible within one cycle, at most about 57 s. A busy node is never re-probed (`services/nodes.py:271,327-328`).

**L20 Worker readiness at join**: `agent/join.py:197` = 45 s for `ggml-rpc-server` to print its banner.
- *Probe*: a normal join. Record the time from "Starting" to "Joined" in the join output.
- *Expect*: well under 45 s on the M4 Air. Over 45 s gives "ggml-rpc-server did not report readiness within 45s".

**L21 Join and agent HTTP timeouts**: coordinator check 5 s (`agent/join.py:124`), join POST 20 s
(`agent/join.py:302`), leave 5 s with 2 s retries (`agent/join.py:383-391`), `hypercompute leave` waits 20 s
(`cli.py:285`).
- *Probe*: join while the G14 runs a CPU-heavy model load (for example 32B gpu_disk).
- *Expect*: join finishes in under 20 s. Every request handler is non-blocking: plan runs in a thread (`app.py:330`),
  and so does the probe (`adapters/llama_rpc.py:86`).

**L22 Local inventory refresh**: `coordinator.py:84-94` every 10 s. The G14's free VRAM and RAM in plans can be up to
10 s old.
- *Probe*: covered by P-VRAM. Re-plan within 10 s of starting the hog, and again after 10 s.
- *Expect*: the first plan is unchanged. The second moves layers.

**L23 Friend measurement in the heartbeat**: `agent/inventory.py:225-238`. The Mac uses psutil only. NVIDIA friends
run `nvidia-smi` with a **20 s** timeout (`agent/inventory.py:36`) on every heartbeat.
- *Probe*: needs an NVIDIA friend, later (5090). Contract-test it now with a slow fake `nvidia-smi`.
- *Expect*: a single `nvidia-smi` stall over 12 s makes that friend stale (W5).

### 3.4 Deployment lifecycle

**L24 Stale plan**: `services/deployments.py:368` = 600 s (hard-coded).
- *Probe*: `$p = Plan @{ model_id = "qwen2.5-0.5b-instruct-q4_k_m"; strategy = "local_only" }`. Then
  `Deploy $p.id` at 9:30 (under) and with a second plan at 10:10 (over).
- *Expect*: under, the deployment is accepted (202). Over:
  `409 {"code":"plan_stale","detail":"plan is older than 10 minutes; check the plan again"}`.

**L25 One active deployment**: `services/deployments.py:380-383`.
- *Probe*: run `Deploy` while one is loading.
- *Expect*: `409 deployment_active`. Replaying the same `Idempotency-Key` returns the original deployment
  (`services/deployments.py:375-379`).

**L26 Load timeout**: `config.py:25` = 1800 s for llama.cpp (`adapters/llama_rpc.py:168`).
- *P-LOAD*: `uv run python scripts\e2e_check.py --port $P --model qwen2.5-32b-instruct-q4_k_m --strategy all_selected --keep --label 32b-3way`.
  Read the `[ t s] loading` lines. Effective link speed = 6.03 GB / (loading s - local load s). The local part
  alone is about 8 s (the 14B auto run loaded in 8.4 s).
- *Expect*: ready when the link is above 3.35 MB/s. Below that, at 1800 s:
  `failed: llama-server was not ready after 1800s (a timeout does not prove workers released memory)`.
  Do not try to force "over" on a good network; compute it (W6).
- Other engines have their own budgets (L50, L57, L63).

**L27 Warmup**: `config.py:26` = 120 s for normal tiers. `services/deployments.py:55-59` gives capacity-mode tiers
`max(120, 2 × idle)`: 360 s for llama.cpp gpu_disk, and 1800 s or more for AirLLM. The warmup asks for 8 tokens, or 1
in capacity mode (`services/streaming.py:443-446`).
- *Probe*: 32B Q4_K_M local_only, which is gpu_disk and capacity mode. This is the **first hardware run of gpu_disk**.
- *Expect*: the plan shows `capacity_mode=true` with the capacity-mode warning, the warmup finishes in seconds, and
  the token rate is disk or page-cache bound. A hung peer during warming fails after 120 s with the text
  "TimeoutError" (W11).

**L28 Engine watch**: `services/deployments.py:493-507` polls `process_alive` every 2.0 s.
- *Probe*: kill llama-server during a ready run:
  `Get-Process llama-server | Stop-Process -Force`.
- *Expect*: within 2 s the deployment is `failed`, with `detect_failure` text or "llama-server exited unexpectedly",
  and the leases are released.

**L29 Stop grace**: `config.py:28` = 8 s (`agent/supervisor.py:235-293`). On Windows, `TerminateProcess` is immediate.
vLLM uses 30 s (L48).
- *Probe*: `StopDep` during a 1024-token stream.
- *Expect*: the cleanup reads "terminated (exit code 1)" in under 1 s on Windows. The stream ends with SSE
  `upstream_error` "interrupted...". The run record is `cancelled`.

**L30 Lease rules**: `services/leases.py:46-68`, all-or-nothing per pool and per node.
`pool_budget = the plan's budget_bytes`. The node allowance is the node's **current** `contribution.max_bytes`
(`services/deployments.py:70-77`).
- *Probe P-LEASE*:
  1. `$p = Plan @{ model_id = "qwen2.5-32b-instruct-q4_k_m"; strategy = "all_selected" }` (the Mac gets 6.43 GiB).
  2. On the Mac: `uv run hypercompute leave`, then `uv run hypercompute join ... --max-memory 4GiB --yes`.
  3. Wait until the node is eligible, then `Deploy $p.id` within 10 min.
- *Expect*: the deployment fails at `reserving` with "<mac>: <N> bytes requested over node allowance 4294967296". The
  same id is kept because node_id is persistent (`agent/join.py:63-71`). If you deploy without the rejoin, it is
  under the limit and succeeds.

**L31 Host CPU safety**: `services/deployments.py:129` subtracts 512 MiB, and `services/planner.py:470-472` warns when
the embeddings plus 512 MiB exceed free RAM.
- *Probe*: plan 32B Q4_K_M auto while a RAM hog leaves about 2 GiB of CPU budget.
  - Size the hog at (available - 8.9 GiB), because budget = available - about 6.4 GiB reserve - 0.5 GiB.
  - Example: `uv run python -c "x=bytearray(<GiB>*2**30); input()"`.
  - Wait 10 s for `refresh_local`, then plan.
- *Expect*: with a CPU budget of about 2 GiB the plan becomes gpu_disk (`sim_plans.py` cpu sweep: CPU 9L paged +
  Mac 21L + CUDA0 34L) and shows the capacity-mode banner. With 4 GiB it stays gpu_ram. A budget of 0 makes it
  infeasible.

### 3.5 Inference slots and streaming (`services/streaming.py`)

**L32 Slots per deployment**: `services/streaming.py:259-270`. llama.cpp gets 1 (`-np 1`, `adapters/llama_rpc.py:128`),
vLLM gets 8 or 16 (L46), and AirLLM and EXO get 1. Waiters are FIFO with **no queue cap**
(`services/streaming.py:93-131`).

**L33 Busy wait**: `services/streaming.py:43` = 30.0 s (`services/streaming.py:272-281`).
- *Probe P-BUSY* (14B Q6_K, 1 slot):
  - A: `uv run python $TP\mkprompt.py 20 $env:TEMP\long.json --max-tokens 1024 --ignore-eos`, then `Chat $env:TEMP\long.json`.
    That is about 40 s at 25 tok/s.
  - B, in a second window 2 s later: `uv run python $TP\mkprompt.py 20 $env:TEMP\b.json --no-stream`, then
    `Chat $env:TEMP\b.json`.
  - Under: repeat with A at `--max-tokens 256` (about 10 s).
- *Expect*:
  - Over: B gets **HTTP 429** `{"code":"busy","detail":"all 1 inference slot(s) of the deployment are busy; try again shortly"}`
    after 30.0 s.
  - Streaming B gets HTTP 200 with `data: {"error":{"code":"busy",...}}`.
  - Under: B is served after about 8 s, and its TTFT includes the wait.

**L34 Exclusive waiters**: a benchmark takes **every** slot, but once per run: each of its 8 runs (2 prompts × (1
warmup + 3 reps)) acquires and releases separately (`services/benchmarks.py:114-125`). It waits up to 300 s for each
acquire (`services/benchmarks.py:120`). Tuner measurements behave the same way (`services/tuner.py:72,481`).
- *Probe*:
  1. Start A: `--max-tokens 512`, about 20 s.
  2. 2 s later, start a benchmark:
     `Invoke-RestMethod -Method Post "$B/api/benchmarks" -Headers $H -ContentType application/json -Body '{"repetitions":3}'`.
  3. 2 s after that, send chat C.
  4. Repeat with A at `--max-tokens 700` (about 28 s).
- *Expect*:
  - The FIFO order is A, then benchmark run 1, then C, then benchmark runs 2-8.
  - With A at 512 tokens, C waits for A's remaining time plus one benchmark run (about 18 + 6 s), which is under 30 s,
    so C is served. With A at 700 tokens, C gets 429 at 30 s.
- *Benchmark over-limit probe*: hold the slot for more than 300 s. On 14B ctx 16384, a chat with
  `--max-tokens 12000 --ignore-eos` takes about 8 min. The benchmark then ends `failed`, with the busy error as its
  `error` (`services/benchmarks.py:135`).

**L35 Engine-busy retry**: `services/streaming.py:48` = 0.5 s retries for up to 30 s (`services/streaming.py:291-316`).
Only AirLLM answers engine-busy (429). See L56.

**L36 Upstream idle timeout**: `config.py:27` = 180 s, used as httpx read timeouts (`adapters/llama_rpc.py:72`,
`adapters/vllm.py:464`). AirLLM uses 900 s or more and EXO 300 s.
- *Probe P-PREFILL* (W3): deploy 14B Q6_K **all_selected ctx 32768** (CPU 1L + Mac 18L + CUDA0 29L; the Mac needs
  `available ≥ 10.5 GiB`). Then:

  ```powershell
  foreach ($n in 8000,16000,31000) { uv run python $TP\mkprompt.py $n "$TP\p$n.json"; Chat "$TP\p$n.json"; LastRun }
  ```

- *Expect*: TTFT grows faster than linearly with prompt length. When TTFT passes 180 s, the client gets
  `data: {"error":{"code":"timeout","message":"no upstream data for 180s: ..."}}`. The run is `timeout`, and the
  deployment **stays ready**. Check the llama-server log to see whether the abandoned prompt keeps the slot busy, which
  would delay the next request.
- Repeat on 32B Q4_K_M 3-way with ctx 16384.

**L37 Non-stream reply cap**: `services/streaming.py:46` = 16 MiB of raw SSE bytes (`services/streaming.py:395-409`).
- One llama.cpp chunk is about 243-261 B (`tests/fixtures/llama_stream_b11770.sse`), so the cap trips at about
  **64,000 generated tokens**. It cannot be reached at ctx 4096.
- *Probe (optional, about 20 min)*: 7B local_only ctx 131072 (gpu_ram CPU 2L). Under:
  `mkprompt.py 20 x.json --no-stream --max-tokens 60000 --ignore-eos`. Over: `--max-tokens 70000`.
- *Expect*: under gives 200. Over gives **502** `response_too_large` "the engine's reply passed 16 MiB without
  finishing".

**L38 Run-record text cap**: `services/streaming.py:47` = 1,000,000 characters (about 200-250k tokens). Not reachable
at any ctx this hardware can plan. Contract test only.

**L39 SSE event cap**: `sse.py:22` = 4 MiB per event. Not reachable from llama.cpp or vLLM. W13 is a contract test.

**L40 Model name check**: `app.py:435-439`. A `model` that is not the ready model gives 404 `model_not_found`. A tune
in progress gives 409 `tuning` (`services/deployments.py:289-292`). No ready model gives 503 `no_ready_deployment`.

### 3.6 Benchmarks and tuner

**L41 Benchmark bounds**: `schemas.py:526-531`. Repetitions 1..20, warmup 0..5, max_tokens 1..2048, label at most 80
characters.
- *Probe*: `bounds_probe.py`. Run it with no ready deployment so at-limit values get 503 instead of starting a
  benchmark.
- *Expect*: 422 on every over-limit value.

**L42 Tune bounds**: `schemas.py:578-584`. max_candidates 1..8, decode_tokens 8..512, ctx 256..262144. Also
`decode_tokens + 64 > ctx_size` gives 422 `bad_request` (`services/tuner.py:331-333`). A deployment that is active
gives 409 first (`services/tuner.py:326-330`).
- *Probe*: `bounds_probe.py`, once with a deployment active (at-limit values give 409) and once with none (ctx 256 and
  decode 193 give 422).

**L43 Tune budgets**: `READY_SLACK_S = 60` (`services/tuner.py:71`). The per-trial budget is load + warmup + 60
(`services/tuner.py:180-195`). `SLOT_TIMEOUT_S = 300` (`services/tuner.py:72`).
- *P-TUNE*: `uv run hypercompute tune qwen2.5-14b-instruct-q6_k --nodes $MAC --max-candidates 8 --port $P`.
- *Expect*: candidates include RAM-overflow, peer-overflow and flag variants (`-fa off`, `-ub 1024/256`;
  `services/tuner.py:61-66`). Each candidate is a full load, and peer candidates resend 0.9 GB to the Mac unless it
  joined with `--cache`. Total time is roughly 8 × (load + about 30 s). The winner is saved, and a later
  `Plan @{model_id=...; strategy="auto"}` shows `heuristic=false`.

### 3.7 Planner constants (`services/planner.py`)

| ID | constant (line) | value | what it moves | hardware check |
|---|---|---|---|---|
| L44a | `UBATCH` (35) | 512 | compute estimate `96 MiB + 512 × n_embd × 32` = 176 MiB for 14B/24B/32B | compare with the "compute buffer size" lines in `%USERPROFILE%\.hypercompute\coordinator\logs\llama-server-<dep>.log`, and the Mac's `ggml-rpc-server.log` |
| L44b | `COMPUTE_BASE` (36) | 96 MiB | same | same |
| L44c | `KV_BYTES_PER_ELEM` (37) | 2 (f16) | KV = ctx × kv_heads × (k+v) × 2 B. 14B: 16 MiB per layer at 4096, 128 MiB per layer at 32768 | the `llama_kv_cache: size =` line, parsed into placement `kv_bytes` |
| L44d | `OVERHEAD_BY_KIND` (38) | cuda 450, metal 256, vulkan 300, cpu 64 MiB | per-device fixed cost | **the most likely source of error on Metal.** At the 32B 3-way plan the Mac is at 6.43 of 6.67 GiB |
| L44e | `RPC_STAGING` (39) | 64 MiB per RPC device | same | same |
| L44f | `DECODE_MS_PER_GB` (44) | cuda 2.64, metal 10.85, cpu 12.5 | the overflow choice (W7) | measured decode tok/s from P-AUTO and P-TUNE |
| L44g | `HOP_MS` (45) | 4.0 ms | the cost of each RPC device | `ping` RTT from the baselines |
| L44h | ctx warning (473-474) | ctx > 32768 | warning only | P-CTX |
| L44i | large-layer warning (491-492) | layer > 2 GiB | warning only | not reachable (the largest layer here is 341 MiB, Mistral 24B) |
| L44j | rebalance passes (191) | 2 × n_layer | bounded search | contract only |
| L44k | VRAM headroom (`services/deployments.py:83`) | 512 MiB (discrete) | CUDA0 budget | P-VRAM (W10) |

- *P-MARGIN* (estimate versus reality at the tightest plans):
  1. Deploy **14B Q6_K all_selected** (CUDA0 margin 15 MiB) and **32B Q4_K_M all_selected** (Mac at 6.43 of 6.67 GiB).
  2. For each device, compare the planned `total_bytes_est` (from `Dep $id`) with the observed placement and with the
     real usage (`nvidia-smi` on the G14; on the Mac, Activity Monitor GPU memory or `sudo powermetrics -s gpu_power`).
  3. Record llama.cpp's compute buffer on each device.
- *Expect*: both load. If the Mac load fails with "failed to allocate" or "out of memory", `OVERHEAD_BY_KIND["metal"]`
  is too low.

### 3.8 Agent, contribution, supervisor, ports

| ID | limit (`file:line`) | value | probe | expected |
|---|---|---|---|---|
| L45a | default reserve (`agent/contribution.py:21-24`) | max(4 GiB, 20% RAM) | P-MACBUDGET | M4 Air reserve 4.00 GiB |
| L45b | default max, unified (`agent/contribution.py:41-43`) | min(65%, total - 6 GiB) | same | 10.0 GiB |
| L45c | default max, discrete (`agent/contribution.py:38-40`) | total - 512 MiB | 5090 later | 31.5 GiB |
| L45d | default max, system RAM (`agent/contribution.py:45-46`) | min(40%, total - 8 GiB) | a CPU-only friend | |
| L45e | low-budget warning (`cli.py:260-261`) | < 1 GiB | Mac `join ... --max-memory 0.9GiB` | prints "warning: contribution budget is only 921.6 MiB". With `--max-memory 0` every plan that includes the Mac is infeasible ("over budget ... 0 B allowed") |
| L45f | worker port (`config.py:13`) | 47052 | | |
| L45g | port scan (`config.py:14`, `netutil.py:145-153`) | 24 ports, 47052..47075 | **P-PORTS**, below | |
| L45h | supervisor stop grace (`agent/supervisor.py:235,276,284`); reconcile wait (`agent/supervisor.py:331,335`) | 8 s; 8 s | `uv run hypercompute pause` on the Mac during a run | the worker stops, then within 3 s the deployment fails with "peer ... lost: paused by contributor" (`services/nodes.py:230-235`) |
| L45i | EXO sidecar start / stop (`agent/exo_sidecar.py:27-28`) | 180 s / 15 s | `join --exo` | the first EXO start finishes in under 180 s, else join fails |

**P-PORTS** (Mac). Hold all 24 ports, then try to join:

```bash
python3 - <<'EOF' &
import socket, time
s = [socket.socket() for _ in range(24)]
for i, x in enumerate(s):
    x.bind(("<MAC_IP>", 47052 + i)); x.listen()
time.sleep(600)
EOF
uv run hypercompute join --host <G14_IP>:8000 --code <code>
```

- *Expect*: with all 24 held, join fails with "no bindable worker port near 47052 on <ip>: [3 errors]".
- With only 47052 held (`range(1)`), the worker gets 47053, and the dashboard card shows `:47053`.

### 3.9 Engine adapters

**llama.cpp** (`adapters/llama_rpc.py`)

| ID | limit (line) | value | probe | expected |
|---|---|---|---|---|
| L46a | connect timeout (72) | 5 s | stop llama-server underneath a ready deployment (L28) | `_watch` catches it first |
| L46b | `/health` poll (183, 189) | 3 s timeout, every 0.5 s | | |
| L46c | log tail (148) | 512 KiB | | placement is parsed from the tail, so a very chatty `-lv 4` 3-device load could push the buffer lines out. Check `participation_detail` on the 32B 3-way run |
| L46d | flag ranges (251-253) | batch and ubatch 1..65536, threads 1..256; ubatch > batch rejected | contract only | |

**vLLM** (`adapters/vllm.py`, `adapters/vllm_output.py`). The G14 GPU must be free, with no llama deployment.

| ID | limit (line) | value | probe | expected |
|---|---|---|---|---|
| L47a | `MAX_UTILIZATION` (75) | 0.90 cap | P-VLLMVRAM, below | |
| L47b | `VRAM_HEADROOM` (79) | 1 GiB | same | |
| L47c | `RUNTIME_OVERHEAD` (80) | 1.5 GiB | same | |
| L47d | `MAX_NUM_SEQS` (86) | 8 interactive, 16 throughput | **P-VLLMSLOTS**, below | |
| L48a | stop grace (89); reap (90); in-distro watchdog | 30 s; 10 s; TERM, then KILL after 30 s | `StopDep` mid-stream; then deploy again and kill the gateway with `taskkill /F /PID <gateway pid>` | stop finishes in about 30 s or less. After the hard kill, `nvidia-smi` shows the VRAM freed within about 30-40 s, and the next gateway start reports `cleanup_orphan` |
| L48b | first-start load budget (`adapters/vllm_output.py:30-31,153-156`) | `min(4 h, 1800 s + download / 1 MiB/s)` | | 7B AWQ is cached (1800 s). Uncached it would be 7,113 s |
| L48c | health probe 2 s, poll 1 s, nvidia-smi 10 s (87-91) | | | |
| L48d | install needs (`installers/vllm.py:36`) | 20 GiB free in WSL | | already installed |
| L48e | disk headroom (81) | 2 GiB beyond the download | | |

**P-VLLMVRAM** (VRAM sizing):
- Fit math: need = 5.19 + 0.22 (KV at 4096, 56 KiB per token) + 1.5 = 6.91 GiB. It fits while
  `util = floor(min(0.9, (free - 1 GiB) / 11.94) × 100) / 100` reserves at least 6.91 GiB, that is, while free VRAM
  is at least about 7.9 GiB.
- Hog command:
  `& "$env:USERPROFILE\.hypercompute\engines\airllm\venv\Scripts\python.exe" -c "import torch; x = torch.empty(3500 * 2**20, dtype=torch.uint8, device='cuda'); input('hold')"`
  The torch CUDA context adds roughly 0.3-0.5 GiB on top of the tensor.
- Under: a 3.5 GiB hog gives feasible.
- Over: a 4.5 GiB hog gives "does not fit: ... free GPU memory or lower ctx_size".
- Plan-then-hog: plan with no hog, start a 5 GiB hog, then deploy. `validate` refuses with "only X of Y VRAM is free
  now" (`adapters/vllm.py:502-511`), and the leases are released. Compare this with W10.

**P-VLLMSLOTS** (slots):
- Command: `uv run python scripts\e2e_check.py --port $P --model qwen2.5-7b-instruct-awq --engine vllm --objective interactive --keep`,
  then `uv run python $TP\flood.py --port $P -n 32 --max-tokens 512`, then the same with `-n 40` and `-n 48`.
- Under: all 32 succeed (4 waves of about 7 s).
- At: all 40 succeed. The 5th wave starts at about 27.6 s, just under the 30 s wait.
- Over: about 8 of 48 get `busy`.
- With `--objective throughput` (16 slots), the boundary moves to about 80. Above 99 open connections, L4 applies.

**AirLLM** (`adapters/airllm.py`, `workers/airllm_worker.py`). Model: 0.5B HF on the G14.

| ID | limit (line) | value | probe | expected |
|---|---|---|---|---|
| L49a | idle (`airllm.py:68,337-341`) | `max(900, 3 × slow s/token)` | | 0.5B: 900 s. 32B: 915 s |
| L49b | warmup budget | 2 × idle | | 0.5B: 1800 s |
| L50 | load (69) | 4 h | | |
| L51 | complete (70) | 6 h | | |
| L52 | disk headroom (71) | 2 GiB beyond the split | 32B HF plan | needs about 131 GB; C: has 26 GiB, so expect "not enough disk for the per-layer split" |
| L53 | memory estimates (72-74) | CUDA context 450 MiB, process RAM 2 GiB, activation slack 256 MiB | | |
| L54 | speed priors (76) | slow 0.21 GB/s, fast 5 GB/s | | the plan shows `s_per_token`. 0.5B measured 0.2 tok/s |
| L55 | worker body (`airllm_worker.py:40`) | 4 MiB, the same as the gateway | a compact 4,194,304-byte body without `"stream"` | the gateway adds `"stream":true` (14 B), so the worker answers 413 and the gateway returns **413** (`services/streaming.py:426-429`). Probe it with a hand-built compact JSON; `mkprompt` output already contains `"stream"` |
| L56 | busy lock (`airllm_worker.py:388-389`) | 1 request at a time | P-AIRBUSY, below | |
| L56b | messages (`airllm_worker.py:41`) | at most 256 | send 257 messages | **400** `invalid_request` from the worker, passed through. llama.cpp accepts them, so the engines differ |
| L56c | stop strings (`airllm_worker.py:42`) | at most 4, each 1..64 characters | send 5 | 400 |
| L56d | default max_tokens (`airllm_worker.py:43`) | 256 | omit `max_tokens` | output ends by 256 tokens or the context. llama.cpp is unbounded up to ctx |

**P-AIRBUSY**: stream with `--max-tokens 64`. After 2 tokens, Ctrl+C the curl. At once, send a new request.
- *Expect*: the worker answers 429 while it finishes the token in progress (about 5 s on 0.5B). The gateway retries
  every 0.5 s and serves the new request within 30 s. Over (a token longer than 30 s) needs a 32B HF model and is a
  contract test only.

**EXO** (`adapters/exo.py`, `adapters/exo_state.py`). Single-Mac now, two Macs later.

| ID | limit (line) | value | probe | expected |
|---|---|---|---|---|
| L57 | load (89) | 4 h | | |
| L57b | stall (88) | no visible progress for 600 s | | |
| L58 | command (83) | 45 s | | |
| L58b | preview (84) | 60 s | | |
| L58c | state (85) | 10 s | | |
| L58d | ack (`exo_state.py:18`) | 60 s | | |
| L59 | API unreachable (90) | 45 s, checked every 3 s (87) | **P-EXOLOSS**, below | |
| L59b | delete confirm (91) | 30 s | | |
| L60 | stream idle (397) | 300 s | | EXO sends a keep-alive every 10 s |
| L61 | `MLX_OVERHEAD` (92) | 1 GiB per Mac | Mac `join --exo`; plan `qwen2.5-7b-instruct-mlx-4bit` with `engine=exo` | feasible only when Mac `available - 4 GiB ≥ 4.99 GiB` (W2), else "needs about 5.0 GiB on the EXO Macs; they offer X". The 0.6B always fits |
| L62 | size caps (93-94; `exo_state.py:19`) | state 8 MiB, completion 16 MiB, event 4 MiB | contract only | |
| L63 | EXO's own node drop (`exo_state.py:21`) | 15 s | two Macs later: Wi-Fi off on Mac 2 | instance deleted, then deployment failed with `LOST` |

**P-EXOLOSS**: with an EXO deployment ready on one Mac, kill only EXO on the Mac (`pkill -f "exo"`). The agent
keeps heartbeating.
- *Expect*: `failed` after about 45-48 s with "EXO API ... unreachable for 45s". If the whole Mac drops, the heartbeat
  path catches it in 15-17 s instead.

### 3.10 Installers and downloads (disk)

| ID | limit (`file:line`) | value | probe | expected |
|---|---|---|---|---|
| L64 | model download (`services/models.py:183-185`) | needs `total + 2 GiB` free, and does not subtract files already downloaded | `uv run hypercompute models pull qwen2.5-32b-instruct-q4_k_m` with 26 GiB free | starts (20.49 GiB needed). Pulling 14B Q8_0 afterwards (16.62 GiB needed, about 5.5 GiB free) fails with "not enough disk: need 14.6 GiB + 2 GiB headroom" |
| L65 | download read timeout (`engine.py:234`) | 60 s per read | | a 60 s stall aborts the download, with progress lost per file |
| L66 | engine install disk (`engine.py:283-286`) | need ≤ free | | |
| L66b | engine smoke (`engine.py:334-335`) | 60 s / 120 s | | |
| L67 | AirLLM install (`installers/airllm.py:41-44,278`) | disk = (wheels + 0.4 GiB) × 3 + 1 GiB; 3600 s per step; 600 s smoke | | |
| L67b | EXO prerequisites (`installers/exo.py:25-30`) | Python 3.13, uv ≥ 0.8.6, Rust ≥ 1.85, Node ≥ 18 | `uv run hypercompute engine install --engine exo` on the Mac | clear refusal naming the missing tool |
| L67c | RPC tensor cache (`agent/join.py:190-192`) | no disk check | `join --cache`, then deploy 32B 3-way twice | about 6 GB is written to `~/.hypercompute/rpc-cache` on the Mac. The second load is faster. Record both load times |

### 3.11 Schema `Field` bounds (`schemas.py`). All are probed by `bounds_probe.py` unless marked otherwise.

| field | line | bounds | under / at / over → expected |
|---|---|---|---|
| `PlanRequest.ctx_size` | 348 | 256..262144 | 255 → 422; 256 and 262144 → 200; 262145 → 422 |
| `TuneRequest.ctx_size` | 580 | 256..262144 | as above, plus the decode rule in L42 |
| `TuneRequest.max_candidates` | 582 | 1..8 | 0 → 422; 1 and 8 → 409 when a deployment is active; 9 → 422 |
| `TuneRequest.decode_tokens` | 584 | 8..512 | 7 → 422; 8 and 512 → 409 when a deployment is active; 513 → 422 |
| `BenchmarkRequest.repetitions` | 528 | 1..20 | 0 → 422; 20 → 503 with nothing ready; 21 → 422 |
| `BenchmarkRequest.warmup` | 529 | 0..5 | 5 → 503; 6 → 422 |
| `BenchmarkRequest.max_tokens` | 530 | 1..2048 | 0 → 422; 2048 → 503; 2049 → 422 |
| `BenchmarkRequest.label` | 531 | at most 80 | 80 → 503; 81 → 422 |
| `CreateSessionRequest.name` | 265 | at most 80 | 81 → 422 |
| `JoinRequest.code` | 158 | 4..32 | 3 or 33 → 422, which is not rate-limited |
| `JoinRequest.node_id` | 159 | `^[a-f0-9]{32}$` | uppercase → 422 (contract) |
| `JoinRequest.display_name` | 160 | 1..64 | `join --name <65 characters>` → 422 "join refused" |
| `JoinRequest.agent_version` | 164 | at most 64 | contract |
| `ContributionLimits.devices` | 145 | at most 64 | contract |
| `ContributionLimits.max_bytes` and `reserve_bytes` | 146-147 | ≥ 0 | `--max-memory 0` is accepted (L45e) |
| `WorkerEndpoint.port` | 154 | 1..65535 | `--worker-port 65536` → 422 at join |
| `HeartbeatRequest.memory_pools` and `devices` | 193-194 | at most 64 | contract |
| `HeartbeatRequest.extra_engines` | 197 | at most 8 | contract |
| `PlanRequest.node_ids` and `TuneRequest.node_ids` | 346, 579 | **unbounded** (only the 64 KiB body limit applies) | contract |

FastAPI's 422 body is its own `{"detail":[{"loc":[...],"type":...}]}`. `app.py` does not reformat validation errors,
so clients see a different shape for 422 than for the app's `{"detail","code","fix"}` errors.

---

## 4. Model-size test matrix (run in this order; each line names the limits it stresses)

| # | model (G14 disk) | how | expected plan (section 1) | limits under test | record |
|---|---|---|---|---|---|
| 1 | 0.5B Q4_K_M (on disk) | `e2e_check.py --strategy all_selected --bench` | Mac 1L + CUDA0 23L | probe (L18), join (L8-L12), placement parse | load s, TTFT, decode tok/s, RTT |
| 2 | 7B Q4_K_M (on disk) | all_selected and local_only `--bench`; then local_only ctx 32768/65536/131072/262144 (P-CTX) | Mac 1L; ctx 262144 → gpu_disk capacity mode | W8, L44h, L27 | per-ctx tier, VRAM used, warnings |
| 3 | 14B Q6_K (on disk) | P-AUTO: `Plan @{model_id="qwen2.5-14b-instruct-q6_k"; strategy="auto"}`, then `overflow="peers"`, then all_selected; P-TUNE | auto → CPU 4L (W7); peers → Mac 4L, CUDA0 15 MiB margin | W7, W10, L44, L43 | decode tok/s per layout, tune winner, actual VRAM |
| 3b | 14B Q6_K ctx 16384 / 32768 | all_selected; P-PREFILL | Mac 12L / CPU 1L + Mac 18L | W3, L36, W2 (needs Mac available ≥ 10.5 GiB) | TTFT against prompt length |
| 3c | 14B Q6_K all_selected | P-BUSY, P-CONN, P-HBGAP, P-HUNG (serving), P-PROBE, L16, L17, L29, L30 | | W1, W4, W5, L13-L19, L33-L34 | outcome per probe |
| 4 | 24B Mistral Q4_K_M (on disk, **add a `configs/models.yaml` entry**) | auto | Mac 10L 3.80 GiB (needs Mac budget ≥ 3.8) | W2 boundary at Mac `available` ≈ 7.8 GiB | where `auto` flips from the Mac to RAM |
| 5 | 32B Q4_K_M (download 18.49 GiB; W16) | all_selected (P-LOAD, P-HBLOAD, P-MARGIN, P-LEASE); local_only (first gpu_disk); overflow=peers (infeasible), then rejoin the Mac with `--reserve 1GiB --max-memory 12GiB` (P-MACBUDGET over) | CPU 9L + Mac 21L 6.43 GiB + CUDA0 34L; local_only gpu_disk | W2, W5, W6, L26, L27, L44d | effective MB/s, max heartbeat age, Metal allocation, swap (`sysctl vm.swapusage`), decode tok/s |
| 6 | 14B Q8_0 (download 14.62 GiB, only if disk allows) | auto at Mac available ≥ 9 and ≤ 8 GiB | Mac 15L 4.80 GiB, or CPU 15L | W2 boundary | which tier |
| 7 | vLLM 7B AWQ (WSL cache) | P-VLLMSLOTS, P-VLLMVRAM, ctx 32768 (ready) and 40960 (vLLM refuses, W8), L48a | 8/16 slots | L4, L32-L33, L47-L48 | throughput, 429 counts, refusal text |
| 8 | AirLLM 0.5B HF (on disk) | P-AIRBUSY, L56b-L56d | disk tier, 1 slot | L35, L49-L56 | s/token, retries |
| 9 | EXO qwen3-0.6b, then 7B 4-bit, on the Mac (`join --exo`) | single Mac; P-EXOLOSS | 0.6B fits; 7B needs Mac `available ≥ 9.0 GiB` | L57-L61, W2 | start s, load s, loss detection s |

**P-MACBUDGET (W2), step by step:**
1. Default join. Record `available` and Metal free, and plan 32B Q4_K_M with overflow=peers. Expect infeasible
   "...over budget: ~9.1 GiB needed, X allowed".
2. On the Mac: `uv run hypercompute leave`, then `uv run hypercompute join --host <G14_IP>:8000 --code <new> --reserve 1GiB --max-memory 12GiB --yes`.
3. Plan again. Expect feasible when `min(metal_free, available) - 1 GiB ≥ 9.15 GiB`.
4. Deploy. This is the real Metal-limit probe. Either:
   - it loads (record decode tok/s and swap), or
   - it fails at load with "a device buffer allocation failed" or "a device ran out of memory while loading"
     (`adapters/llama_output.py:37-48`) and the leases are released.

**P-HUNG (W4, W11):**
1. Serving: during a 512-token stream on 14B all_selected, run `pkill -STOP -x ggml-rpc-server` on the Mac.
   - Expect: the stream stalls. At 180 s the client gets SSE `timeout` and the run is `timeout`. The deployment stays
     `ready`, and the node stays `eligible`.
   - A second chat also waits 180 s.
   - Then `pkill -CONT -x ggml-rpc-server`. A third chat should work.
2. Loading: deploy 32B all_selected. When progress shows "loading weights onto devices (peers receive...)", STOP the
   worker.
   - Expect: the node stays eligible and the deployment stays `loading`.
   - After 3 min, CONT. The load should resume and finish, which proves only a 30 min timeout would have caught the
     hang.
3. Warming: optional and hard to time. If you hit it, it fails at 120 s with the text "TimeoutError".

**P-IP (W9).** These change network settings, so run them only on the test network.
- Mac: `sudo networksetup -setmanualwithdhcprouter Wi-Fi <MAC_IP+1>`, then revert with
  `sudo networksetup -setdhcp Wi-Fi`.
  - Expect: heartbeats continue from the new IP. A running RPC deployment fails with "llama-server exited
    unexpectedly" or "Failed to connect". The next verify reports "worker unreachable from host", and the node stays
    ineligible until the agent rejoins.
- G14: `netsh interface ip set address "Wi-Fi" static <new> <mask> <gw>`, then revert with
  `netsh interface ip set address "Wi-Fi" dhcp`.
  - Expect: the Mac goes stale at 15-17 s and stops sharing at about 300 s, because the gateway is still bound to the
    old IP.

---

## 5. Later hardware: limits that change

- **A second Mac.**
  - Both Macs get budgets of `min(10, min(metal, avail) - 4)` (W2), and the planner orders peers by ascending budget
    (`services/planner.py:287-294`).
  - EXO: each Mac downloads the full repo (STALL 600 s, LOAD 4 h). `MLX_OVERHEAD` is 1 GiB per Mac. EXO drops a
    silent node after 15 s, and the gateway's own unreachable timer is 45 s. Test Wi-Fi off on one Mac: expect EXO's
    404, then `LOST`, in about 15-20 s.
  - llama.cpp 3-device RPC: each device adds `HOP_MS`, and names come from `assign_rpc_names`
    (`services/deployments.py:172-185`).
- **RTX 5090 desktop.**
  - As a friend: W5 (`nvidia-smi` with a 20 s timeout in the heartbeat). The discrete budget is `total - 512 MiB`,
    which is 31.5 GiB. Windows needs a firewall rule and its excluded port ranges checked against the 24-port scan
    (L45g).
  - As host: vLLM is capped at 0.90 (28.8 GiB) and 16 sequences (W14). `refresh_local` and `nvidia-smi` time out
    at 10 s.
- **ASUS GX10 (GB10, 128 GB unified, sm_121).**
  - Unified pool: reserve = `max(4, 25.6) = 25.6 GiB` is subtracted from free memory (W2). `max_bytes` =
    `min(83.2, 122)` = 83.2 GiB.
  - vLLM is **infeasible** because nvidia-smi reports `[N/A]` memory (W14).
  - The llama.cpp `linux-arm64-cuda13.4` variant has never been validated.
  - AirLLM: the 0.21 GB/s slow prior is pessimistic when the page cache holds the model.

## 6. Contract-test only (not reachable on the G14 + M4 Air)

| limit | why not reachable | contract test to add |
|---|---|---|
| L37 non-stream cap | needs ≥ 64k generated tokens | optional 20 min hardware run |
| L38 record text cap | | |
| L39 SSE event cap | | add a test for W13 |
| L62 EXO size caps | | |
| L44i, L44j | | |
| L46d flag ranges | | |
| L10 failed-attempt cap | dead code | |
| tuner `SLOT_TIMEOUT_S` | a tune deployment refuses chats (409 `tuning`), so nothing else can hold its slot | |
| AirLLM token longer than 30 s | needs 32B HF; 131 GB of disk | |
| W1 | | heartbeats keep working while 100 owner connections are open |
| W4 | | a stopped worker with a live agent fails the deployment |
| W5 | | a slow `nvidia-smi` does not delay the heartbeat |
