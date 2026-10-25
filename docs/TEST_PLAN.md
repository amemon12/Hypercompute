# Test plan: find what breaks, stress every limit, decide the next push

This plan covers three phases:

* **Phase A** runs on the G14 alone, before the next session with the Mac (about 3 hours).
* **Phase B** is the next G14 + M4 MacBook Air session (about 4 hours).
* **Phase C** needs later hardware: a second Mac, an RTX 5090 desktop and an ASUS GX10.

Every test has an ID, a command, an **expected outcome taken from the code or the real planner**, the evidence to
keep and the decision it feeds. A test passes when what happens matches the prediction. Any mismatch is a finding.

The detailed sources behind this page were written by four independent agents and checked against the code at
`dfd8cf9`. A three-reviewer pass then checked this page for completeness, command accuracy and ID consistency.

| Detail file | Contents |
| --- | --- |
| [testplan/predictions.md](testplan/predictions.md) | The real planner and selector over 11 model sizes x 6 hardware setups x 3 context lengths, auto plus the throughput and run-together variants (396 plans), built from real GGUF headers, with flags for thin margins, coin-flip overflow choices and disk paging |
| [testplan/limits.md](testplan/limits.md) | About 70 hard limits, timeouts and budgets with `file:line`, what happens at each, and an under / at / over probe |
| [testplan/failures.md](testplan/failures.md) | 58 failure injections (network, Mac, host, engines, tuner, dashboard, concurrency), each with trigger, expected behaviour and evidence |
| [testplan/open_questions.md](testplan/open_questions.md) | Everything still unverified, the measurement that settles it, and the build item that depends on it |

**IDs on this page.** Every ID carries a prefix, so the same letter never means two things.

| Prefix | Meaning | Defined in |
| --- | --- | --- |
| `K1`-`K6` | Already measured (section 0) | this page |
| `W1`-`W16` | A limit that looks wrong | limits.md section 2 |
| `P-NAME` (P-CONN, P-HUNG, P-PREFILL, P-CTX, ...) | A limit probe | limits.md sections 3 and 4 |
| `limits.md L9` | A row of the limits inventory (L1-L67), always written with the file name | limits.md section 3 |
| `FI-` + ID (FI-N2, FI-F9, FI-C5, ...) | A failure injection | failures.md (which writes them without `FI-`) |
| `Q-` + ID (Q-L1, Q-C2, Q-M7, Q-F3, Q-P1, ...) | An open item | open_questions.md section 4 (which writes them without `Q-`) |
| `R1`-`R12`, `R2b` | Runs 1-12 of predictions.md section 9; R2b is added here | predictions.md section 9 |
| `S-1` | The soak test | section 4.4 of this page |
| `B0`-`B31` | A build item | section 6 of this page |

---

## 0. What is already known (measured, 2026-10-21)

**K1. The first real G14 + M4 Air session already happened** (2026-10-21 04:24-05:05 UTC, the evening of Oct 20
local). The gateway's records hold three deployments with the friend's M4 MacBook Air over UW campus Wi-Fi, run on
the MVP build before the engine-tier merge (old `--device CUDA0,RPC0`
order, strategy `fastest_fit` or `all_selected`). Raw evidence:
[results/pair/20261021-campus-wifi-g14-m4air.json](../results/pair/20261021-campus-wifi-g14-m4air.json).

| Deployment | Model | Placement (planned == observed) | Mac budget | Decode | Load to the Mac (end to end) | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| `725198fe` | 0.5B Q4_K_M, ctx 2048 | CUDA0 20 + Mac 4 layers | 2.87 GiB | warmup only: 1 decode step, 249 ms (4.0 tok/s); the chat was cut off | 176 MiB in 12.9 s (13.7 MiB/s) | Mac lost mid-request, failed after 17 s, 2 leases released; the raw llama-server log (`~/.hypercompute/coordinator/logs/llama-server-725198fe*.log`) ends mid-request |
| `c476506a` | 14B Q6_K, 4k | CUDA0 46 + Mac 2 layers | 4.35 GiB | 11.7-12.3 tok/s (2 chats) | 1,040 MiB in 87 s (11.9 MiB/s) | Stopped cleanly |
| `865929b9` | Mistral-Small 3.2 24B Q4_K_M, 4k | CUDA0 32 + Mac 8 layers | 4.12 GiB | 3.5-10.6 tok/s over 12 chats (11 at 8.1-10.6, median 9.1) | 3,166 MiB in 162 s (19.6 MiB/s) | Gateway stopped after the last chat (05:05Z); the next start found llama-server gone and marked it failed |

What this settles or moves, with the caveat that the G14's power state was not recorded:

* **Q-P1 answered:** the macOS arm64 b11770 binary runs and serves RPC (handshake, layers, KV). **Q-M1 answered:**
  Metal working set 12,124 MiB (12,712,935,424 B), 10-core GPU, 16 GiB unified memory counted once.
* **FI-N1 answered for that evening:** campus Wi-Fi did not isolate the two laptops (10.19.175.205 reached
  10.19.252.207:47052). It is a /15 Public network, so check again on the day (Q-L1).
* **W2 confirmed and worse than predicted:** the Mac's planner budget was 2.9-4.4 GiB, not the 10 GiB the join prints
  and not the 6 GiB this plan assumed. The friend's free memory was about 6.9-8.4 GiB, minus the 4 GiB reserve.
* **The old split order sent the logits over Wi-Fi every token.** In all three runs the Mac was the last device and
  held the output head, so each token shipped the full logits back to the G14: 152,064 x 4 B = 594 KiB for Qwen and
  131,072 x 4 B = 512 KiB for Mistral. The 14B with 2 layers + head on the Mac ran at 81-86 ms/token, against 37
  ms/token with 4 layers in the G14's RAM (`results/*gw-14b-auto.json`), a gap of about 48 ms. Moving 594 KiB at
  the 11.9 MiB/s the Mac's load ran at would take about 49 ms, so the logits are consistent with the gap. They are
  not proven to cause it: the runs differ in build, placement, time and possibly power state, the load rate was
  measured in the other direction, and the 0.5B's single decode step took 249 ms with the same logits. The current
  order keeps the head on CUDA0, so about 20 KiB crosses each way per token instead of 20 KiB out and about 594 KiB
  back. K1 is the "before" number for that change; R1 and R2 measure the "after".
* **Q-D1 triggered:** cold Mac loads of 1,040 MiB took 87 s and 3,166 MiB took 162 s (12-20 MiB/s end to end, which
  includes the G14's own load and the Mac's cache writes). Both are over the 30 s line for load progress (B9). Q-L3
  is not settled: that rate is only a lower bound on the link, so run `tcp_blast.py`. With the current order the
  Mac no longer receives the 609 MiB output head.
* **Q-M5 triggered:** on the 14B and 24B splits the CUDA0 compute buffers were 257 and 366 MiB (and Metal 268 MiB on
  the 24B) against the 176 MiB estimate, more than 64 MiB over (B3). The 0.5B split stayed under its 110 MiB
  estimate.
* **A real FI-N2-class event:** in `725198fe` the Mac dropped off mid-request and the run failed by the heartbeat rule
  at 17 s, as predicted.

**K2. W1 confirmed on the G14 (loopback).** `scripts/stress/hold.py 100 127.0.0.1 8000 12` made `/api/health` answer
503 three times in a row, then 200 after the sockets closed. Evidence:
[results/stress/20261021-g14-loopback.txt](../results/stress/20261021-g14-loopback.txt). Still to run: the same from
the Mac during a live split, to see the friend's heartbeats fail (FI-N11, the P-CONN over case).

**K3. Bounds, no-deployment half: 35 of 35 checks pass, 4 skipped** (`bounds_probe.py`, same file). ctx 262144 is accepted for
any model (W8 as coded). The tune at-limit values still need a run with a deployment active.

**K4. Q-M7 confirmed: free host RAM decides the tier.** With 8.32 GiB available as the planner measured it (PowerShell
read 8.19 GiB free just before), the 14B Q6_K `auto` plan kept the same 4 CPU layers but came out `gpu_disk` with a
"seconds per token" warning. The CPU budget was 1.0 GiB: 8,518 MiB available - 6,373 MiB reserve - 512 MiB safety -
609 MiB of token embeddings, which always stay in host RAM (`planner.py:229`). The 4 CPU layers need 1.14 GiB
([plan](../results/plans/20261021T203914Z-auto-verify-plan-only.json)). The section 3.2 tables assume 20 GiB free.
Record free RAM before every plan.

**K5. The selector swap is real.** `qwen2.5-7b-instruct-q4_k_m` with `--objective throughput` plans vLLM serving
`qwen2.5-7b-instruct-awq` ([plan](../results/plans/20261021T203916Z-auto-verify-swap.json)).

**K6. Two test-tool bugs were found and fixed while writing this plan.** `flood.py` crashed (httpx `StreamConsumed`)
on any streamed refusal, which is exactly the case it exists to count. `e2e_check.py` sent the requested model id in
its chat, which the gateway refuses with 404 once the selector swaps the model (K5); it now sends the plan's model id,
records refusals and in-stream errors in its evidence, and exits 1 on them. `e2e_check.py --plan-only` and `flood.py --body` are new (section 1).

---

## 1. How to run it

**Tools** (run from the repo root on the G14 with `uv run python ...`; they reach the gateway on loopback with the owner
token, except where noted):

| Tool | What it does |
| --- | --- |
| `scripts/e2e_check.py --plan-only` | Prints the full plan (tier, `--device`, `-ts`, CPU layers, `-ngl`, each device's estimate against its budget, warnings) and saves it under `results/plans/`. Nothing is loaded. **This is the prediction for a run** |
| `scripts/e2e_check.py` | Plans, deploys, chats, optionally benchmarks (`--bench`), then stops. Add `--keep` whenever another tool streams against the deployment afterwards; stop it from the dashboard when done |
| `scripts/stress/itl_probe.py` | Streams N tokens per rep; prints first-token time, decode tok/s and inter-token gap p50/p90/p99/max, plus run-to-run spread. G14 only |
| `scripts/stress/flood.py` | N concurrent chats; counts outcomes by (HTTP status, error code). `--body FILE` posts a long prompt from `mkprompt.py` |
| `scripts/stress/failures_load.py` | `chats` mode (concurrent chats with timestamps, JSONL) and `idle` mode (raw idle sockets, no credentials) |
| `scripts/stress/hold.py` | Holds N idle TCP connections (W1). Runs on the G14 or the Mac. Point it only at your own gateway |
| `scripts/stress/bounds_probe.py` | Request bounds and body limits just under / at / just over. Never starts a tune or a benchmark, so run it twice (section 4.7) |
| `scripts/stress/mkprompt.py` | Writes a chat body with about N prompt tokens, or a body of an exact byte size. It only writes the file |
| `scripts/stress/tcp_blast.py` | Raw TCP throughput G14 to Mac. `--serve 47999` on the Mac serves one connection and exits, so restart it before each `--send <MAC_IP>:47999` |

To send a `mkprompt.py` body once (PowerShell on the G14):

```powershell
$T = (Invoke-RestMethod http://127.0.0.1:8000/api/owner/bootstrap).owner_token
uv run python scripts/stress/mkprompt.py 30000 $env:TEMP\p30k.json --max-tokens 64
curl.exe -sN -H "X-Hypercompute-Owner: $T" -H "Content-Type: application/json" --data-binary "@$env:TEMP\p30k.json" http://127.0.0.1:8000/v1/chat/completions
```

**Watchers to keep open for every test** (G14): the gateway console, `curl.exe -sN http://127.0.0.1:8000/api/events`,
`nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1`, and Task Manager (GPU shared memory,
RAM). On the Mac: Activity Monitor (Memory Pressure), `memory_pressure`, `pmset -g therm`, and the agent log. Exact
commands: failures.md section 2 and limits.md section 0.

**Rules for every run**

1. Record free RAM first: `[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)` (K4).
   The planner reads available memory through psutil, which can differ slightly; the CPU budget in the
   `--plan-only` output is the number that decides the tier.
2. Run `e2e_check.py --plan-only` with the same arguments and write down the prediction: tier, `--device`, `-ts`, CPU
   layers, `-ngl`, each device's estimate against its budget. Compare it with section 3 and say why it differs.
3. Run the test. Afterwards compare with the observed placement (llama.cpp's own load log, saved in `results/*.json`)
   and the measured speed. Note any mismatch with its numbers.
4. After every failure test, confirm the cleanup: no `llama-server`, `ggml-rpc-server` or vLLM left, VRAM back to idle,
   leases released (the deployment's cleanup text says how many).
5. At the end: `uv run python scripts/export_lifecycle_evidence.py` and `uv run python scripts/summarize_results.py`,
   then commit `results/` and `docs/RESULTS.md`.

---

## 2. Breaks the code already predicts (confirm these first)

Reading the code and running the planner offline shows these. Each needs one real run to confirm, then a fix from
section 6.

| ID | What will go wrong | Trigger | Severity | Fix |
| --- | --- | --- | --- | --- |
| W1, FI-N11, FI-C3 | 100 idle TCP connections to the gateway, from any device on the LAN with no credentials, make every request 503. The friend's heartbeats fail, the Mac goes stale in 15-17 s and its deployment fails. **Confirmed on loopback (K2)** | From the Mac during a ready split: `uv run python scripts/stress/hold.py 100 <G14_IP> 8000 40`; or `failures_load.py chats --n 110` | High | B0 |
| W2 | A default `join` on a 16 GB Air contributes far less than the 10 GiB it prints (the 4 GiB reserve is taken from memory already net of the OS). **Confirmed: 2.9-4.4 GiB (K1)** | Read the Mac assignment's `budget_bytes` in any `--plan-only` output | High | B5 |
| FI-N2, FI-N3, FI-F1 | Mac Wi-Fi drop or sleep mid-run: the host kills llama-server, the friend's single-client worker stays stuck on the dead connection, and the Mac card says "worker unreachable" with firewall advice that does not apply. Only Pause then Resume recovers | `networksetup -setairportpower en0 off` for 20 s during a stream; or `pmset sleepnow` | High | B4 |
| W4, FI-F9 | A stalled worker is never detected: a load waits the full 30 min, every chat waits 180 s, the deployment stays `ready` | `pkill -STOP -x ggml-rpc-server` on the Mac during a stream, `pkill -CONT` after 200 s (past the 180 s read timeout; P-HUNG keeps it stopped through a second chat) | High | B4 |
| FI-N5 | The Mac's IP changes: heartbeats continue, the node stays eligible with a dead worker address (an eligible running heartbeat only re-saves, `services/nodes.py:242-243`), every plan fails, and rejoining from the new IP gets 409. W9 covers the G14 side | Switch the Mac between hotspot and router mid-session, idle and mid-chat | High | B4 |
| FI-F7, Q-P3 | `kill -9` of the agent on macOS leaves the unauthenticated RPC worker listening on the LAN with no time limit (macOS has no kill-on-close) | `pkill -9 -f "hypercompute join"`, then `lsof -nP -iTCP:47052 -sTCP:LISTEN` | High (security) | B4 |
| FI-T2, FI-T5 | Stop on a tuner trial card, or a peer lost mid-tune: the tune continues and still saves a "measured profile" that never measured the Mac | Click Stop on the trial card during the R2 tune; or kill the Mac worker mid-tune | Medium | B6 |
| FI-E9, FI-E7 | EXO pause, leave or Wi-Fi drop: unclean stop, leases stay held even across gateway restarts | `uv run hypercompute pause` on an EXO Mac during an EXO deployment | Medium | B22 |
| FI-F8, FI-F12 | The worker exits (including a possible SIGPIPE on Stop mid-token): the agent quits, and every rejoin needs a new invite code | Stop a deployment mid-token 10 times; `pkill -9 ggml-rpc-server` | Medium | B4, B23 |
| FI-F10 | Pause and resume at once: no lock, can end with no worker while the host shows "worker stopped" | On the Mac, 10 times each: `uv run hypercompute resume & uv run hypercompute resume & wait` and `uv run hypercompute pause & uv run hypercompute resume & wait` | Medium | B23 |
| FI-E4 (part b) | vLLM's EngineCore killed alone: the deployment can stay `ready` because only process liveness is watched | `wsl -- pkill -9 -f EngineCore` during a vLLM run | Medium | B24 |
| W3, P-PREFILL | One 180 s silence budget for llama.cpp: a 30k-token prompt over the Mac can pass it while the engine is healthy (prefill counts as silence) | The `mkprompt.py 30000` body (section 1) against a 14B ctx-32768 split | Medium | B25 |
| W5 | Heartbeats run 3 s after the previous one returns, with an 8 s request timeout; two slow ones in a row cross the 15 s stale threshold. NVIDIA friends also run `nvidia-smi` on every heartbeat | Count `no heartbeat` events over a 30-minute session with loads (Q-F1) | Medium | B29 |
| W6 | The 30 min llama.cpp load budget is fixed: at K1's end-to-end load rates (12-20 MiB/s), a peer share above about 21-35 GiB cannot finish (R10 sends about 29 GiB) | R10; P-LOAD | Medium | B28 |
| W8 | `ctx_size` is not capped at the model's trained context: vLLM at 40960 plans as feasible and then fails at startup; 7B at 262144 silently becomes a disk-paged plan (K3) | `--plan-only` 7B at ctx 262144; vLLM at 40960 | Medium | B19 |
| W10 | llama.cpp does not re-check free VRAM at deploy time; the 14B Q6_K plan has about 15 MiB of margin and plans stay deployable for 10 minutes | Plan the 14B, open a GPU-heavy app, deploy (FI-H6) | Medium | B3 |
| W12 | `INVITE_MAX_FAILED_ATTEMPTS` is never enforced (a wrong code matches no invite, so nothing can count it); `config.py` says every constant can be set by environment variable but only paths can; the agent ignores the heartbeat interval the host sends | No hardware probe for the invite cap (contract test, limits.md L10). Run limits.md L9 instead: 12 joins in 60 s, calls 11-12 get 429. Set `HYPERCOMPUTE_STALE_AFTER_S=30` and check the node still goes stale at 15-17 s | Medium | B19 |
| W11 | A warmup timeout is reported as the bare text "TimeoutError" | Hard to time; see P-HUNG step 3 | Low | B19 |
| W13 | An SSE event over 4 MiB raises `ValueError` that `stream_events` does not catch | A crafted engine stream (contract test) | Low | B19 |
| Planner gaps | `auto` never tries the Mac and host RAM together when RAM alone fits; `gpu_disk` warns "seconds per token" in 7 of 18 cases that run at RAM speed (and, per K4, even on the 14B); 124 plans at 16k/32k only fit if flash attention engages; compute buffers of full-offload splits are 1.5-2.1x the estimate on CUDA0 (K1); a 7B Q8_0 "Many users" request silently becomes the 4-bit AWQ on vLLM | R4 (Mac + RAM), R5 and K4 (gpu_disk label), R5/R6 (flash attention), Q-M5 on R1 and R3 (compute buffers), and a `--plan-only` 7B Q8_0 throughput plan once it has a manifest entry (selector swap) | Medium (speed and honesty) | B8, B15a, B26, B3, B19 |
| Q-P6 | The test suite has never run on macOS; `tests/test_vllm_install.py` likely fails there (GNU `find -printf`) | `uv run pytest -q` on the Mac | Low | B19 |

---

## 3. The model-size ladder

### 3.1 Models and what each size stresses

| Size | File | n_layer | On the G14 now | Manifest entry | What this size stresses |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Q4_K_M | 0.46 GiB | 24 | yes | yes | Plumbing and the network hop: 1 Mac layer is about 0.1 ms of work, so the rest of the time per token is the link |
| 1.5B Q4_K_M | 1.04 GiB | 28 | no | **no** | Small-model fixed cost per token |
| 3B Q4_K_M | 1.96 GiB | 36 | no | **no** | Same, more layers |
| 7B Q4_K_M | 4.36 GiB | 28 | yes | yes | Local baseline (87 tok/s), split cost, context sweep up to 262k, vLLM comparison |
| 7B Q8_0 | 7.54 GiB | 28 | no | **no** | Selector quality guard (silent swap to AWQ) |
| 14B Q4_K_M | 8.37 GiB | 48 | no | **no** | The edge of one GPU: all-GPU at 4k, 1-2 CPU layers at 16k, 12-13 at 32k |
| 14B Q6_K | 11.29 GiB | 48 | yes | yes | **The overflow decision** (Mac vs host RAM), the tuner, the 15 MiB margin |
| Mistral-Small 3.2 24B Q4_K_M | 13.35 GiB | 40 | yes | **no** (copy `configs/models.example.yaml` to `configs/models.yaml` first, then add the entry: the loader reads `models.yaml` instead of the example, `services/models.py:29`) | The Mac-wins branch of the cost model, and the W2 boundary |
| 14B Q8_0 | 14.62 GiB | 48 | no | yes | First real capacity split onto Metal (about 4.8 GiB on the Mac, more than the 2.9-4.4 GiB seen in K1) |
| 32B Q4_K_M | 18.49 GiB | 64 | no | yes | Peers + RAM together, 3-device chains, load time over Wi-Fi |
| 32B Q6_K | 25.04 GiB | 64 | no | **no** | Disk paging: RAM speed at 4k/16k, page-cache thrash at 32k |
| 72B Q4_K_M | 40.99 GiB | 80 | no | **no** | True capacity mode: cyclic disk re-reads at 5-22 s per token |

**Disk** (W16): C: has about 25 GiB free and `models pull` needs the file plus 2 GiB. So 32B Q6_K and 72B need
`HYPERCOMPUTE_HOME` on an external SSD, and 14B Q8_0 plus 32B Q4_K_M (33.1 GiB) do not fit together. **Do not delete
the Mistral 24B**: R2b and Q-G1 need it, and its repo and revision are not recorded, so it cannot be pulled again. Pull
the 32B in Phase A, run R5 and R4 on it, delete it, then pull the 14B Q8_0 for R3.

### 3.2 Predicted plans (strategy `auto`, single user, gateway budgets)

Budgets: CUDA0 11.16 GiB (free VRAM minus 512 MiB, measured idle), host CPU 13.28 GiB (assumes **20 GiB free RAM**),
Mac 6 GiB. Cell format: tier, CPU layers, `--device [-ts]`, predicted decode ms per token. **DISK** = capacity mode
(CPU layers paged from the GGUF). Rows with no gateway value in predictions.md section 4 are unchanged from its section
2. K1 and K4 say the real Mac and host budgets can be much smaller: see the next table, and always trust `--plan-only`.

| Model | G14 alone, 4k | G14 alone, 16k | G14 alone, 32k | G14 + Air 6 GiB, 4k | G14 + Air 6 GiB, 16k |
| --- | --- | --- | --- | --- | --- |
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | same | same | same (Mac idle) | same |
| 1.5B / 3B Q4_K_M | gpu CUDA0 only | same | same | same | same |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | same | same | same | same |
| 7B Q8_0 | gpu CUDA0 [29] 20 | same | same | same | same |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu2 [47] 26 | RAM cpu13 [36] 44 | same as alone | RAM cpu2 [47] 26 |
| 14B Q6_K | RAM cpu4 [45] 39 | RAM cpu12 [37] 57 | RAM cpu19 [30] 73 | RAM cpu4 [45] 39 (Mac alternative 41.7: a coin flip) | **Mac** RPC0,CUDA0 [12,37] 57 |
| 14B Q8_0 | RAM cpu15 [34] 83 | RAM cpu19 [30] 94 | RAM cpu24 [25] 109 | **Mac** RPC0,CUDA0 [15,34] 79 | RAM cpu19 [30] 94 |
| 32B Q4_K_M | RAM cpu30 [35] 138 | RAM cpu35 [30] 152 | **DISK** cpu40 [25] 166 | RAM cpu30 [35] 138 (Mac idle) | RAM cpu35 [30] 152 |
| 32B Q6_K | **DISK** cpu39 [26] 223 | **DISK** cpu42 [23] 235 | **DISK** cpu44 [21] 243 (thrash: 5.1-11.8 s/token) | RAM cpu25 + Mac [14,26] 218 | **DISK** cpu30 + Mac [12,23] 231 |
| 72B Q4_K_M | **DISK** cpu62 [19] 435 | **DISK** cpu64 [17] 445 | **DISK** cpu65 [16] 450 | **DISK** cpu52 + Mac [10,19] 430 | **DISK** cpu55 + Mac [9,17] 442 |

**At the budgets actually seen** (host CPU 7.46 GiB as in `results/*gw-14b-auto.json`, ctx 4096, limits.md section 1):

| Model | strategy | Mac 6.67 GiB | Mac 4.0 GiB (K1) | Mac 2.0 GiB (browser open) |
| --- | --- | --- | --- | --- |
| 14B Q6_K | auto | CPU 4 + CUDA0 44 (Mac unused) | same | same |
| 14B Q8_0 | auto | Mac 15 (4.80 GiB) + CUDA0 33 | CPU 15 + CUDA0 33 | CPU 15 + CUDA0 33 |
| 14B Q8_0 | all_selected | as auto | CPU 3 + Mac 12 + CUDA0 33 | CPU 10 + Mac 5 + CUDA0 33 |
| Mistral 24B Q4_K_M | auto | Mac 10 (3.80 GiB) + CUDA0 30 | Mac 10 (3.80 of 4.00) + CUDA0 30 | CPU 10 + CUDA0 30 |
| 32B Q4_K_M | auto | CPU 9 + Mac 21 (6.43 GiB) + CUDA0 34 | CPU 18 + Mac 12 + CUDA0 34 | **DISK**: CPU 25 paged + Mac 5 + CUDA0 34 |
| 32B Q4_K_M | local_only | **DISK**: CPU 30 paged + CUDA0 34 | | |

Things to check against reality in every run:

* The output head should never leave CUDA0 (all 396 predictions keep it local).
* Decode predictions use the planner's priors (CPU 12.5 ms/GB, Metal 10.85, hop 4 ms). The G14's measured CPU cost is
  about 18-20 ms/GB, so RAM-heavy rows will run slower than predicted, and the hop on a real Wi-Fi link is unmeasured
  (K1's Mac loads ran at only 12-20 MiB/s end to end). Disk-paged rows that fit the page cache should run near RAM speed, not "seconds per token". Rows bigger
  than the page cache (every 72B row on the G14 alone or with one Air, and 32B Q6_K at 32k on the G14 alone) will run
  at 5-22 s/token.
* Full tables for every setup (two Airs, RTX 5090, GX10, "Many users"), per-device memory and flags: predictions.md
  sections 2, 4, 5 and appendices B and C.

### 3.3 The most informative runs, in order

Command template: `uv run python scripts/e2e_check.py --model <id> --strategy <s> --ctx <n> --bench --label <label>`,
after the same command with `--plan-only` (section 1, rules 1-2). Add `--keep` when another tool runs against the
deployment afterwards. `R1`-`R7` and `R2b` need only the G14 and the Air; `R8`-`R12` need later hardware or more disk.

| Run | Model x setup x ctx | Expect | Measures / could break |
| --- | --- | --- | --- |
| R1 | 0.5B, G14 + Air, 4k, `all_selected` | `--device RPC0,CUDA0 -ts 1,24`, Mac 1 layer | The real hop: `hop = 1000/decode_tps - 1.9 ms`. First run of the new peers-first order with a Mac. Also Q-M5 (compute buffers) |
| R2 | 14B Q6_K, G14 + Air, 4k, `auto`, then `uv run hypercompute tune qwen2.5-14b-instruct-q6_k --ctx 4096 --max-candidates 4` twice (wrap each in `Measure-Command`, Q-D4), then once with `--ctx 16384` | RAM, 4 CPU layers, `-ngl 45`; peer alternative `-ts 4,45` | **The overflow decision.** K1's Mac split (old order, logits on the link) ran less than half as fast as the RAM tier; this is the first Mac-vs-RAM measurement with the head on CUDA0 (the Mac wins only if the real hop is under about 6.5 ms). FI-T2 and FI-T5 run during the second tune |
| R2b | Mistral 24B Q4_K_M, G14 + Air, 4k, `auto` (add the `configs/models.yaml` entry first) | Mac 9-10 layers at about 3.8 GiB if the Mac budget is >= 3.8 GiB; otherwise RAM. Prior 66 ms (Mac) vs 67 ms (RAM). K1 ran it at a median 9.1 tok/s (about 110 ms) with 8 layers + head on the Mac (old order) | The Mac-wins branch on a model already on disk. Repeat with a browser open on the Mac to find the W2 flip (Mac available about 7.8 GiB) |
| R3 | 14B Q8_0, G14 + Air, 4k, `auto` (pull 14.6 GiB first) | Mac 15 layers at 4.8/6.0 GiB, CUDA0 33 + head, **only if the Mac budget is >= 4.8 GiB**: K1 saw 2.9-4.4, so the friend joins with `--reserve 2GiB` and closes apps, and Q-M3 watches memory pressure. Otherwise CPU 15 + CUDA0 33 | First real capacity split onto Metal; about 4.1 GiB crosses Wi-Fi at load (about 4-6 minutes at K1's 12-20 MiB/s). Then close the Mac lid mid-stream: fail in 16-17 s, leases released |
| R4 | 32B Q4_K_M, G14 + Air, 4k, `auto` then `all_selected` | auto at CPU budget >= about 9.5 GiB: 30 CPU layers, Mac idle; at about 7.5 GiB: CPU 9 + Mac 21. Together (Mac 6 GiB): CPU 11 + Mac 19; at Mac 4 GiB: CPU 18 + Mac 12 | **Predicted: together is 20-25% faster, and auto picks the slower plan** (B8). Only shows when free RAM is high enough for auto to stay RAM-only (record it, Q-M7) |
| R5 | 32B Q4_K_M, G14 alone, 32k, `auto` | DISK, 40 CPU layers paged (assumes 20 GiB free) | Should run at about 166-230 ms/token after the first pass (page cache), not "seconds"; CUDA OOM if flash attention does not engage |
| R6 | 14B Q6_K, G14 alone, 32k, `auto` | RAM, 19 CPU layers (40%), 3.6 GiB KV | Compute buffer growth with context; long-prompt first token on a 40%-CPU split |
| R7 | 7B Q4_K_M, G14 alone, 32k, `--objective throughput` | vLLM `qwen2.5-7b-instruct-awq`, util 0.89, about 2 full-length requests of KV | Selector swap (K5). Then 16 concurrent long prompts with `flood.py -n 16 --body` on a `mkprompt.py 8000` body: expect queueing, not 16-way speed |
| R8 | 32B Q4_K_M, G14 + 2 Airs, 16k | auto RAM-only, Macs idle; together `RPC0,RPC1,CUDA0 -ts 16,16,30`, CPU 3 | 3-device chain (llama.cpp issue #28487, "deadlocks with 3+ endpoints"); run with Q-H1 (7B `all_selected` x 10, then the 32B) |
| R9 | 32B Q4_K_M, G14 + RTX 5090, 4k, interactive vs throughput | `-ts 30,35` vs `-ts 49,16` | Predicted: loading the faster 5090 more is 15-20% faster even for one user |
| R10 | 72B Q4_K_M, G14 + RTX 5090, 4k | RAM cpu3, `-ts 59,19`, 5090 at 30.4/30.7 GiB (1.1% margin) [brief budgets: cpu2, `-ts 60,19`, 0.3%] | About 29 GiB crosses the network at load: about 17 min on good Wi-Fi, past the 30 min budget at K1's rates (W6) |
| R11 | 72B Q4_K_M, GX10 alone, 32k | single device, 51 of 83 GiB | Unvalidated arm64 CUDA build for sm_121; GB10 inventory |
| R12 | 72B Q4_K_M, G14 alone, 4k (external SSD) | DISK, 62 CPU layers paged | 9-22 s per token; gaps near the 180 s idle budget; desktop stalls. **Run without `--bench`**: one 128-token chat already takes 19-47 min |

---

## 4. Stress dimensions beyond model size

### 4.1 Context length

| ID | Test | Expected |
| --- | --- | --- |
| P-CTX | 7B Q4_K_M `local_only` `--plan-only` at ctx 4096, 32768, 65536, 131072, 262144 | All-GPU up to 32k; 262144 silently becomes a disk-paged plan (W8). Record tier, VRAM, warnings per ctx |
| R6, Q-M8 | 14B Q6_K at 16k and 32k | KV estimate exact (`llama_kv_cache: size`); compute buffer vs the planner's flat 176 MiB |
| P-PREFILL, W3 | The 30k-token body (section 1), then 8000, 16000 and 31000 tokens, on a 14B ctx-32768 split over the Mac | First token time; whether the 180 s silence budget fires on a healthy engine |
| W8 | vLLM with ctx 40960 | Plans as feasible, fails at startup today: should be refused at plan time |

### 4.2 Concurrency

| ID | Test | Expected (from the code) |
| --- | --- | --- |
| FI-C2 | `flood.py -n 8 / 16 / 32 / 64` on vLLM 7B AWQ deployed with `e2e_check.py --model qwen2.5-7b-instruct-awq --engine vllm --objective throughput --keep` | 16 run at once (`--max-num-seqs 16`; the default interactive objective gives 8); the rest queue in the gateway; past `BUSY_WAIT_S` (30 s) streamed requests end as `status=200 code=busy` (HTTP 429 only with `--no-stream`). flood.py reports counts and latency, not tok/s: compare total time with the 546 / 862 tok/s rates |
| FI-C1 | Same on llama.cpp: (a) 7B local, (b) 7B split with the Mac, (c) 14B auto | One request at a time (`-np 1`); the rest wait up to 30 s, then `200 busy` (429 with `--no-stream`). failures.md predicts about 17-18 of 32 served on (a), 5-6 on (b), about 6 on (c) |
| FI-C3 | `failures_load.py chats --n 110` with the Mac split ready | Crosses the 100-connection cap: 503s, and the Mac's heartbeats can fail (W1) |
| FI-C5 | Stop while 8 streams are open (`failures_load.py chats --n 16 --max-tokens 1024` on vLLM, or `--n 8` on llama.cpp) | In-flight runs `cancelled`. Queued waiters are **not** refused: they reach the stopping engine and end as `error` runs with `upstream_error`, not "deployment stopped" (B30) |
| FI-C4, FI-C6 | Benchmark while 32 chats run; `failures_load.py chats --n 4` in a loop while pressing Stop | FI-C4: benchmark repetitions take every slot and never overlap a chat; chats queued behind it get `busy` after 30 s. FI-C6: HTTP 200 with no data (`no_done`) and no run record (B30) |
| Q-V3 | 32 and 64 users with 2k-token prompts on vLLM: `mkprompt.py 2000 $env:TEMP\p2k.json`, then `flood.py -n 32 --body $env:TEMP\p2k.json` and `-n 64` | KV exhaustion and preemption; check the gateway maps vLLM's mid-stream error correctly |
| Q-C8 | `e2e_check.py --model qwen2.5-7b-instruct-q4_k_m --engine llama_rpc --strategy all_selected --objective throughput --bench --label pair-7b-throughput` vs the same without `--objective` (without `--engine llama_rpc`, throughput picks vLLM and ignores the Mac) | llama.cpp runs `-np 1`, so a proportional split can only help prefill. **Prefill gain < 20%: B13** |

### 4.3 Network

| ID | Test | Expected / decision |
| --- | --- | --- |
| Q-L1 | RTT on each candidate network (campus, phone hotspot, a home router if possible): 300 x `curl -w "%{time_connect}"` to `/api/health` from the Mac; `ping -n 200 <MAC_IP>` from the G14 | Pick the network. **If p90 > 7 ms or p99/p50 > 5, build the link probe (B2)**. K1's 12-20 MiB/s end-to-end loads hint at a slow campus link |
| Q-L2 | Per-token hop through RPC: `llama-bench` 0.5B with 1-2 Mac layers vs local (open_questions.md section 5.1) | Replaces the 4 ms prior (B1) |
| Q-L3 | `tcp_blast.py` G14 to Mac, 1000 MiB, 3 times; restart `--serve 47999` on the Mac before each send | Predicts cold load time. K1's end-to-end loads ran at 12-20 MiB/s, a lower bound on the link. **Under 40 MB/s: the Mac is not a long-prompt target** |
| Q-L4 | Q-L1 loop during a cold 14B load | Bufferbloat. **Heartbeat RTT p99 > 5 s makes false "stale" real (B4, B29)** |
| Q-L5 | `GGML_RPC_DEBUG=3` llama-bench with 2 Mac layers (open_questions.md section 5.1) | How many RPC messages per token, and how much of the per-token cost is round trips vs bytes. **If the GET_TENSOR wait is much smaller than the hop, the cost is in small messages: watch upstream PR #24122, do not build our own** |
| Q-L6 | Optional: two USB-C Ethernet adapters and a cable (link-local works) | Separates Wi-Fi cost from RPC cost |
| Q-F3, FI-N3 | Wi-Fi off for 5 s, 20 s, 60 s during generation | Q-F3's bar: 5 s stalls and resumes; 20 and 60 s fail cleanly within about 20 s. The code predicts the FI-N2 wedge after 20 s and 60 s (Mac ineligible, "worker unreachable", until Pause + Resume). Any hang past 30 s: B4 |
| FI-N5 | Mac IP change mid-session | See section 2 |
| Q-F9 | WAN off (hotspot data off), then chat and start a new deployment | llama.cpp needs no internet; EXO's first start does |

### 4.4 Endurance

| ID | Test | Expected |
| --- | --- | --- |
| S-1 | 60 minutes of `itl_probe.py --tokens 512 --reps 150 --label soak` (about 24 s per rep) on the 14B Mac split, deployed with `e2e_check.py ... --strategy all_selected --keep` | Decode stays within 15%; gap p99 and stalls (gaps > 3x p50) recorded; zero false "no heartbeat" events (Q-F1) |
| Q-C6 | Sustained decode on the fanless Air: `llama-bench` 7B with 8 Mac layers, `-n 512 -r 10`, plus `pmset -g therm` | **If decode drops > 15% in 2 minutes, the tuner must measure after warm-up** |
| Q-C7, Q-T1 | The same tune twice (R2) | **If the winner flips, ship B6 (repetitions and a margin)** |

### 4.5 Memory pressure

| ID | Test | Expected |
| --- | --- | --- |
| Q-M7 | Before every 14B Q8_0, 32B or 72B plan (and any plan with CPU layers), record free RAM and the plan's CPU `budget_bytes` | Section 3.2 assumes 20 GiB free. K4 had 8.32 GiB available and a 1.0 GiB CPU budget (after the reserve, the 512 MiB safety and 609 MiB of token embeddings), which turned the 14B into `gpu_disk`. At about 7.5 GiB of CPU budget, 32B auto mixes Mac and RAM and 32B `local_only` is DISK |
| Q-M2, Q-M3 | Mac at 4-6.5 GiB of layers with the friend's usual apps open: `memory_pressure`, `vm_stat`, `sysctl vm.swapusage` before, during and after; also with `join --reserve 2GiB` | Decides the Mac budget formula (B5). K1: the default join gave 2.9-4.4 GiB |
| Q-F5, FI-F4 | Friend opens heavy apps during a stream | **If decode drops > 30% or the worker dies, warn per friend from the heartbeat (B21)** |
| Q-G1, Q-G2 | Hold RAM with a filler process so free RAM is below the 24B's CPU part, plan `local_only` | First real `gpu_disk` run; **> 5 s/token or a frozen desktop makes capacity mode opt-in (B15b)** |
| Q-M6 | `llama-bench` 14B with every layer on CUDA0 (does not fit); watch "Shared GPU memory" in Task Manager | Windows may spill to shared memory instead of failing; the engine-tier sessions saw a slowdown instead of an OOM. (The Mistral log once cited for this was the Mac, K1.) **Silent spill: add a post-load check (B3)** |
| W10, FI-H6 | Plan the 14B, start a GPU-heavy app, deploy within 10 minutes | Today nothing re-checks VRAM; expect an OOM or a silent spill |
| Q-M4 | `nvidia-smi -l 1` during loads vs the sum of CUDA0 buffers in the load log | Real CUDA overhead vs the 450 MiB estimate (B16) |

### 4.6 Failure injection (full catalogue in failures.md)

Run in this order, gentlest first. Take a snapshot before and after each (failures.md section 2).

1. **G14 alone:** FI-N11, FI-C1(a), FI-C2, FI-C4, FI-C5, FI-C6, FI-D1 to FI-D4 (dashboard: two tabs, browser
   connection limit, reload mid-run, SSE reconnect), FI-H1 to FI-H3 (gateway kill -9 during load, chat and tune),
   FI-H5 to FI-H7 (GPU used by another process, VRAM taken between plan and run, RAM pressure on the RAM tier), FI-H10,
   FI-H11 (WSL shutdown during vLLM, WSL RAM after vLLM), FI-E3 to FI-E6 (vLLM OOM at start, vLLM killed in WSL incl.
   EngineCore only, gateway kill -9 during vLLM, AirLLM killed mid-token).
2. **Mac alone** (agent running, host idle): FI-F5, FI-F6, FI-F7, FI-F10, FI-F11, FI-E2, FI-E8.
3. **Pair, gentle:** FI-N1, FI-N7 (steps), FI-F3, FI-F8 mid-chat, FI-F12 (10 times), FI-C1(b, c), FI-C3, FI-T1,
   FI-T3.
4. **Pair, link-breaking:** FI-N2, FI-N3, FI-N10, FI-N5, FI-F1, FI-F9, FI-T2, FI-T5, FI-H4, FI-N4, FI-N9. Keep Pause
   then Resume ready as the recovery for FI-N2 and FI-N3.
5. **EXO** (if installed on the Air): FI-E7, FI-E8, FI-E9.
6. **Destructive last:** FI-H8 (disk full during a pull), FI-H9, FI-F13, FI-N6, FI-N8. Revert firewall, IP, pf and
   NVIDIA settings after each.

### 4.7 Limits and bounds

`uv run python scripts/stress/bounds_probe.py --port 8000` checks every request bound and body limit just under / at /
just over. Run it twice: once with no deployment (done, K3: benchmark at-values give 503, plus the ctx-256 tune case),
and once while a deployment is ready (tune at-values `max_candidates` 1/8 and `decode_tokens` 8/512 give 409 instead
of starting a tune). The other limits (heartbeat and stale timing, probe timeouts, load budgets, invite TTL and rate
limit, inference slots, engine idle budgets, EXO timers, disk checks) each have a probe in limits.md section 3.

### 4.8 Engines

| ID | Engine | Test | Decision |
| --- | --- | --- | --- |
| Q-V1 | vLLM | Gateway kill -9 while vLLM serves; watch VRAM and `wsl -- pgrep -af vllm` | **GPU memory held > 60 s after the kill: fix before showing vLLM** |
| Q-V4 | vLLM | 14B AWQ (needs a pinned entry) | "Many users" for 14B on the laptop or not |
| Q-A1 | AirLLM | 7B bf16 (needs a pinned 15 GB entry) | **> 20 s/token: AirLLM stays behind explicit `engine=airllm`** |
| Q-A2 | AirLLM | Stop during a token | VRAM released within 10 s |
| Q-X1 | EXO | `time uv run hypercompute engine install --engine exo --yes` on the Air | **> 20 min, or needs Rust / Node the friend would not install: EXO stays experimental (B12)** |
| Q-X2, Q-X3 | EXO | `join --exo`, then `e2e_check.py --model qwen3-0.6b-mlx-4bit --engine exo --strategy all_selected --bench --label pair-exo-0.6b`, then the same with `qwen2.5-7b-instruct-mlx-4bit`; compare 7B decode with llama.cpp Metal on the same Mac (`local_only` would leave the Mac out and the plan infeasible) | **EXO >= 1.2x llama.cpp Metal: keep it for all-Mac sessions and plan the 2-Mac test; < 1.0x: demote** |
| Q-X4 | EXO | Gateway kill -9 during an EXO run, restart | Instance deleted on the next start, leases released |

### 4.9 First run on the friend's Mac

| ID | Test | Decision |
| --- | --- | --- |
| Q-P1 | Answered by K1 (the binary served RPC). Still run `uv run hypercompute doctor` after `git pull` | Fallback stays `scripts/build_llama_macos.sh` |
| Q-P2, Q-M1 | `uv run hypercompute inventory` vs `system_profiler SPDisplaysDataType`, and `"$(find ~/.hypercompute/engines -path '*macos-arm64*' -name llama-server -type f \| head -1)" --list-devices` (`llama-server` is not on the PATH) | K1 already shows 12,124 MiB working set and one unified pool; confirm on the day |
| Q-P5 | Stopwatch from a clean Mac to "Ready to contribute" (Xcode CLT, uv, private clone, engine install). Issue the invite only after `engine install` finishes (invite TTL 15 min) | **> 15 minutes, or the GitHub step blocks: single-download app moves up (B11)** |
| Q-P6 | `uv run pytest -q` on the Mac | Fix any macOS-only test failures (B19) |

---

## 5. The twelve answers that decide the next push

| # | Question | Today's assumption | Known now | Settled by |
| --- | --- | --- | --- | --- |
| 1 | Host-RAM decode cost on the G14 | 12.5 ms/GB prior | about 18-20 ms/GB (ngl sweep) | Q-C2 (G14 only, Phase A) |
| 2 | M4 Air Metal decode cost | 10.85 ms/GB from a third-party M4 row | not separable from the hop in K1 | Q-C1, Q-C6 |
| 3 | Per-token network hop on real Wi-Fi | 4 ms | Not measured for the current order; on the old order the Mac split cost about 48 ms per token more than the RAM tier, consistent with the logits transfer (K1) | Q-L1, Q-L2, Q-L5, R1 |
| 4 | Does the Mac beat G14 RAM at equal overflow, end to end | Planner picks RAM for the 14B at 4k by 2.9 ms: a coin flip | Not on the old order (11.7 vs 26.7 tok/s, K1); the current order removes the logits transfer | Q-C3, R2, `hypercompute tune` |
| 5 | Run-to-run noise of a decode measurement | Tuner takes one sample; minimum score wins | | Q-C7, Q-T1 |
| 6 | Real Mac memory budget | Join prints 10 GiB | 2.9-4.4 GiB with the default join (K1) | Q-M2, Q-M3 |
| 7 | Compute buffers of full-offload splits | Flat 176 MiB | 1.5-2.1x on CUDA0 with the Mac (K1) | Q-M5, Q-M6 |
| 8 | Cold and warm load time of Mac layers over Wi-Fi | No progress signal exists | Cold loads 12-20 MiB/s end to end: 87 s for 1,040 MiB (K1) | Q-D1, Q-D2, Q-L3 |
| 9 | How the system fails on real Wi-Fi | 15 s stale rule, 180 s idle timeout | One real drop failed cleanly at 17 s (K1) | FI-N2, FI-N3, FI-F9, FI-N5 |
| 10 | macOS first run and friend setup time | Never executed | Binary and inventory work (K1); setup time unknown | Q-P5, Q-P6 |
| 11 | EXO install cost and speed vs llama.cpp Metal | Never run; several timeouts are guesses | | Q-X1 to Q-X4 |
| 12 | Do profiles carry across networks | The profile key contains the friend's `ip:port` | | Q-T2, Q-L1 on two networks |

---

## 6. Next-push build list, gated on the results

Ranked by expected value. Each item names the result that triggers it and the result that cancels it. Sizes: S < 1
day, M 1-3 days, L > 3 days. **(confirmed)** = needed whatever the next session shows, from the code or from K1-K6.

| # | Build item | Triggered by | Cancelled by | Size |
| --- | --- | --- | --- | --- |
| B0 | **Admission control that keeps friends alive** (confirmed, K2): header-read timeout and per-IP connection cap; heartbeats and joins on their own listener or limit; cap each deployment's waiting queue | W1 | none | M |
| B1 | **Measured priors and a per-node calibration store** (confirmed: the CPU prior is 45-60% low): replace the CPU / Metal costs and the 4 ms hop with measured values; store per-device cost and per-link hop | Q-C1, Q-C2, Q-L2 differ from priors by > 15% | every coefficient within 15% | S constants, M store |
| B2 | **Per-session link probe** in the heartbeat (RTT p50/p90 per friend), used as the hop and shown on the friend's card | Q-L1 p90 > 7 ms or unstable across networks | p90 < 5 ms and stable everywhere | M |
| B3 | **VRAM safety for splits** (confirmed, K1): compute estimate including pipeline-parallel copies; a post-load check for Windows shared-memory spill; re-read free VRAM at deploy (W10) | Q-M5 > 64 MiB over estimate, Q-M6 silent spill, FI-H6 | estimates within 64 MiB and clean OOMs | S |
| B4 | **Real-network failure hardening**: progress watchdog for stalled workers (W4); heartbeat tolerance (two misses or 30 s); re-bind and re-register on IP change with the stored credential; the agent restarts its worker instead of quitting; worker liveness check for half-open connections; macOS parent-watch so the worker dies with the agent (confirmed for the macOS part) | FI-N2, FI-N3, FI-F9, FI-N5, FI-F7, FI-F8 | all pass within 20 s, no orphan | M |
| B5 | **Honest Mac budget** (confirmed, K1): apply the reserve to RAM only, cap by the Metal working set with a margin, print the effective budget at join | W2 | none (the Mac swaps at 6.5 GiB: keep the formula, fix only the message) | S |
| B6 | **Tuner robustness**: 3 repetitions per candidate, winner must beat the planner pick by 2 sigma; a layer-count sweep and mixed candidates; save no profile when a candidate failed for outside reasons (peer lost, stopped by owner); Stop on a trial card cancels the tune or is hidden | Q-C7 CV > 3%, Q-T1 winner flips, Q-T4 finds a better split, FI-T2, FI-T5, Q-D4 tune wall time > 10 min | CV < 2%, identical winners | M |
| B7 | **Inter-token latency in run records and benchmarks** (gap p50/p90/p99, stalls) | always useful | none | S |
| B8 | **Mac + host RAM together as an overflow option**, in the planner and the tuner | R4 confirms together is faster | Metal plus hop costs at least as much as CPU | M |
| B9 | **Load progress and `--cache` by default** (confirmed, K1: 87-162 s cold loads): the agent reports received bytes in the heartbeat | Q-D1 cold load > 30 s; Q-D2 warm < 10 s | Q-D1 < 15 s for the 14B | S-M |
| B10 | **Profile keys by hardware identity and link**, not `ip:port` | Q-T2 no replay after an IP change | Q-T2 replays | S |
| B11 | **Single-download friend app** (roadmap milestone 5) | Q-P5 > 15 min or the GitHub step blocks | Q-P5 < 10 min | L |
| B12 | **EXO: invest or demote** | Q-X3 >= 1.2x and Q-X1 < 20 min: invest | Q-X3 < 1.0x or heavy installs: demote | S / M |
| B13 | **llama.cpp "Many users"**: either `-np N` with a `max_concurrency` hook, or stop splitting proportionally | Q-C8 prefill gain < 20%; FI-C1 | gain >= 20% | S |
| B14 | **Peer and launch hygiene**: NVIDIA friends must expose a CUDA device; launch llama.cpp with an empty config dir; real ggml abort patterns in failure detection (FI-F8, Q-F4); show only bound addresses (FI-N6); the host diagnostic names port 8000 (FI-N8) | Q-H3, Q-G3 | none (cheap) | S |
| B15a | **Page-cache-aware `gpu_disk`** (confirmed by code, and K4): separate "paged but fits the page cache" (RAM speed) from "cyclic thrash" (seconds per token); fix the warning and the warmup budget | R5 at RAM speed | none | S |
| B15b | **Capacity mode opt-in** (`allow_capacity_mode` default false) | Q-G1 or R12 > 5 s/token or a desktop freeze | Q-G1 and R12 < 2 s/token and responsive | S |
| B16 | **CUDA overhead calibration** (450 MiB to measured + 150 MiB) | Q-M4 | Q-M4 >= 450 MiB real | S |
| B17 | **GX10 path**: validate the arm64 CUDA build for sm_121, GB10 inventory, a unified-pool budget default, a `parse_nvidia_smi` fallback for `[N/A]` memory so vLLM can plan on GB10 (W14), `KV_SHAPES` for 14B and 32B | Q-H6, Q-H7, W14 on the GX10 | build and vLLM plan work as is | S-M |
| B18 | **Cap RPC endpoints at 2 until validated** | Q-H1 or R8 hang with 2 Macs | 10/10 clean 3-device runs | S |
| B19 | **Correctness fixes found by reading the code** (confirmed): enforce `INVITE_MAX_FAILED_ATTEMPTS` (contract test); cap `ctx_size` at the model's trained context (W8, K3); a clear warmup-timeout message (W11); catch oversized SSE events (W13); honour the host's heartbeat interval; make `config.py` env overrides real or drop the claim (W12); the selector must not swap Q8_0 for 4-bit AWQ without asking; manifest entries for the missing sizes and a doctor disk check (W16); macOS test fixes (Q-P6) | W8, W11, W12, W13, selector rows | none | S |
| B20 | **Speed-aware fill for faster friends**: per-device cost so a faster friend GPU (5090, GB10) gets more layers | R9 | throughput split not faster | M |
| B21 | **Friend telemetry**: battery / AC, Low Power Mode, thermal state, memory pressure in heartbeats, shown as warnings | FI-F2, FI-F3, FI-F4, Q-F5, Q-C6 | none of these change speed | S |
| B22 | **EXO lease release**: when the node itself stopped EXO (pause or leave), release its leases; treat "EXO gone" as instance gone on a single Mac | FI-E9, FI-E7 | none | S |
| B23 | **Agent pause/resume lock**, a per-start log file, `reconcile_orphans` skips this run's worker, worker started with SIGPIPE ignored | FI-F10, FI-F12 | 10/10 clean | S |
| B24 | **vLLM watcher polls `/health`** (503 = engine dead), not only process liveness | FI-E4 part b: deployment stays `ready` | the API server exits with its EngineCore | S |
| B25 | **Prefill-aware idle budget**: `idle_timeout_for(plan)` that grows with `ctx_size` and tier, or llama-server progress events | W3 timeout on a healthy engine | a 30k prompt's first token < 180 s | S |
| B26 | **Flash-attention awareness**: pass `-fa on` for ctx >= 16k, or budget the no-FA KQ buffer (n_head x ctx x 512 x 4 B) | R5 or R6 CUDA OOM at `sched_reserve`, or FA not engaged on RPC / Metal | FA engages on every device | S |
| B27 | **Fill margin**, or re-plan one layer down on a load OOM (226 of 396 plans are within 3% of a budget) | Q-M5, W10 OOM on a tight plan | all tight plans load | S |
| B28 | **Load budget that scales with peer bytes**: 1800 s + peer bytes / 1 MiB/s, with bytes-sent progress (W6) | R10, P-LOAD near 30 min | loads well under budget | S |
| B29 | **Heartbeat on a fixed clock**, `nvidia-smi` capped at about 3 s in the heartbeat path (W5) | Q-F1 false stale; limits.md L23 | zero false stale | S |
| B30 | **Interrupt fails queued waiters** with "deployment stopped"; `ready()` inside `chat_stream` sends an SSE error and writes a run record | FI-C5, FI-C6 | none | S |
| B31 | **Host `--reserve` flag** so the owner keeps RAM for the desktop on the RAM tier | Q-G1, FI-H7 desktop stalls | none | S |

Suggested order given K1-K6: B0, B5, B1, B19, B3, B9, B4, B7, B23, B30, then B2 and B6 after Q-L1 and the R2
tunes, then B8, B10 and B25-B29 as their triggers land. B11 to B22 are each gated on one specific result.

---

## 7. Schedule

### Phase A: G14 alone, before the Mac session (about 3 hours)

1. Q-C2: `llama-bench` 14B Q6_K at `-ngl 46,45,44,41,37`, then `-t 6,8,10,12` at `-ngl 41` (host-RAM cost and best
   thread count). Commands: open_questions.md section 5.2.
2. Q-M4, Q-M6: CUDA overhead and the shared-memory spill check.
3. Q-G1 to Q-G4: first `gpu_disk` run with a RAM filler; llama.cpp config-file check; the Linux-only supervisor and
   vLLM tests inside WSL.
4. vLLM block: R7, FI-C2, Q-V3, W8 at ctx 40960, Q-V1, FI-E3, FI-E4 (a and b), FI-E5. AirLLM: Q-A2, FI-E6.
5. Host failures on loopback: FI-H1 to FI-H3, FI-H5 to FI-H7 (with W10), FI-H10, FI-H11. Then `bounds_probe.py` with a
   deployment ready (the second half of 4.7), FI-C1(a), FI-C4 to FI-C6, FI-D1 to FI-D4. W1 on loopback is done (K2).
6. Q-M7 then R5 and P-CTX. R5 needs the 32B pulled (section 3.1 disk order); skip it if the disk is short.
7. Commit and push `scripts/stress/`, then `git pull` on the Mac: it needs `tcp_blast.py` (the Q-L3 server) and
   `hold.py` (W1 from the Mac). `itl_probe.py` runs only on the G14.

### Phase B: G14 + M4 Air (about 4 hours)

1. **Network** (10 min per network): Q-L1 on each candidate network, keep the best; note band and BSSID. K1's Mac
   loads ran at only 12-20 MiB/s, so try a phone hotspot or a home router as well as campus Wi-Fi.
2. **Mac setup** (the friend): `git pull`, Q-P5 timings if starting clean, Q-P2, Q-M1, Q-P6. Issue the invite only after
   `engine install` finishes, then `join --cache` (and later `--reserve 2GiB` for R3).
3. **Cold load first:** Q-D1 (7B `all_selected`, empty `~/.hypercompute/rpc-cache` on the Mac) and Q-D2 right after
   the join, before any llama-bench, because llama-bench fills the cache.
4. **Calibration** (about 30 min, no deployment active, no Check plan or tune while llama-bench runs): Q-L3, Q-L2,
   Q-L5, Q-C1, Q-C4, Q-C6, Q-C3 llama-bench block.
5. **Model ladder through the gateway:** R1, 7B local vs split, R2 with the tunes (twice at 4k, once at 16k, each
   timed for Q-D4), R2b, Q-C8, R4 (if the 32B is still on disk, after Q-M7), R3 (if pulled), the 14B at 16k (Mac-wins
   branch), Q-M3 on the Mac during Mac splits, S-1 soak for 30-60 minutes (`--reps 75-150`).
6. **Failure paths:** W1 from the Mac during a ready 14B split (`hold.py 100 <G14_IP> 8000 40`) and FI-C3; Q-F7
   (Pause and Resume, then leave); Q-F10 (two users on a split: two `itl_probe.py --tokens 1024` at once); FI-F10
   (concurrent pause/resume, 10 times); FI-N2 and FI-N3 with Q-F3 (5 / 20 / 60 s Wi-Fi drops); FI-F1 (sleep); FI-F9
   (`kill -STOP` for 200 s); FI-F8 (worker crash, Q-F4); FI-F12 (Stop mid-token, 10 times); FI-H1 and FI-H2 with the
   Mac in the run (Q-F6); FI-F7 with Q-P3 (agent kill -9, check for an orphan worker); FI-N5 (IP change, idle and
   mid-chat); W3 and P-PREFILL; limits.md L9 (12 joins in 60 s); Q-F9 (WAN off). Keep counting false "no heartbeat"
   events the whole session (Q-F1).
7. **EXO last** (Q-X1 to Q-X4, then FI-E7, FI-E8, FI-E9), because the install may take long and nothing else should
   depend on it.
8. **Wrap up:** export lifecycle evidence, regenerate RESULTS, copy the Mac's calibration files next to `results/`,
   commit.

### Phase C: later hardware

| Hardware | Tests |
| --- | --- |
| Second Mac | R8 with Q-H1 (3-device chain: 7B x 10 runs, then the 32B), Q-H2 (EXO 2-Mac ring over Wi-Fi), FI-N7 with two Macs contending for airtime, losing one of two peers mid-chat, FI-E7 and FI-E9 with two Macs, profiles keyed by two peers (FI-N5 doubles) |
| RTX 5090 desktop | R9, R10, Q-H3 (its card must show a CUDA device with about 32 GiB, not RAM), Q-H4 (5090 as host vs G14 as host), Q-H5 (Ethernet hop), W5 / limits.md L23 (heartbeat age with the 5090 as a busy friend: one `nvidia-smi` stall over 12 s makes it stale), vLLM as host at 0.90 utilization and 16 sequences (W14), FI-H5 on the peer (the friend gaming) |
| ASUS GX10 | R11, Q-H6 (arm64 CUDA build, sm_121), Q-H7 (GB10 inventory), W14 (a vLLM plan on GB10: expect infeasible because nvidia-smi reports `[N/A]` memory) before Q-H8 (vLLM aarch64 wheels), FI-F7 repeated (PDEATHSIG fires when the starting thread exits, and `resume()` starts the worker from a `to_thread` thread) |

---

## 8. What to report after each phase

* `results/*.json` and `docs/RESULTS.md` (regenerated), plus the Mac's calibration JSONL files.
* For every test ID: prediction (the `--plan-only` output), observed, verdict (match / mismatch / broke), and the
  numbers.
* For every section 2 item: confirmed or not, with the evidence line.
* The twelve answers (section 5) filled in, and the build list (section 6) re-ranked from them.
* Doc fixes after the test (not before): RUNBOOK section 5 and the README stale-node text, which say a stale node must
  rejoin with a new code although it recovers on its own within 5 minutes (W15); the README's 32B and 14B Q8_0 claims
  from the real plans (Q-M7, Q-M2); the join message "up to 10.0 GiB" (B5); `configs/profiles.example.yaml` once the
  pair is measured.
