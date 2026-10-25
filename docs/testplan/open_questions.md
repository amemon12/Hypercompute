# Hypercompute: open questions and next-push inputs

> **Added after review (2026-10-21).** The friend's M4 MacBook Air had already run three deployments with this G14 over campus Wi-Fi on the MVP build (`results/pair/20261021-campus-wifi-g14-m4air.json`, summarized in [../TEST_PLAN.md](../TEST_PLAN.md) section 0). Where this file says the macOS binary, the Mac inventory or a real Mac split has never run, read it against that evidence: the binary ran, the Metal working set is 12,124 MiB, the Mac's planner budget was 2.9-4.4 GiB, and its splits decoded at about 12 tok/s (14B) and a median 9.1 tok/s (Mistral 24B) on the old local-first order, where the Mac held the output head and sent 0.5-0.6 MB of logits back per token. [../TEST_PLAN.md](../TEST_PLAN.md) cites the IDs in this file with a `Q-` prefix (for example `Q-C2`).

Repo `C:\Users\operator\hypercompute`, main at `dfd8cf9`, read 2026-10-21. Sources: README.md, docs/ENGINE_TIERS.md,
docs/IMPLEMENTATION_STATUS.md, docs/RESULTS.md, all of docs/research/*.md (UNVERIFIED / open-items / addendum
sections), a grep of `backend/` for estimates, guesses and "not checked", the planner/tuner/budget code, the skipped
and `hardware`-marked tests, and the raw logs under `~/.hypercompute/coordinator/logs`. Nothing was started on port
8000, no model was loaded on the GPU, no repo file was changed.

Labels: **MEASURED** = in this repo's evidence. **DERIVED** = arithmetic on measured numbers. **SIM** = output of the
real planner code run offline on real GGUF headers (scripts below). **INFERRED** = read from code or docs, not run.

Helper files:

| File | What it does |
|---|---|
| `plan_sim.py` (scratch, not in the repo) | runs `planner.make_plan` on the G14's downloaded GGUFs (7B, 14B Q6_K, Mistral 24B) for CUDA budgets 11.4/11.0/10.5 GiB and Mac budgets 0/2/4/6.7 GiB, overflow auto/peers/host_ram, ctx 4096 and 16384 |
| `plan_sim_synth.py` (scratch, not in the repo) | same for 14B Q8_0 and 32B Q4_K_M with shapes approximated from catalog sizes (not downloaded) |
| `scripts/stress/itl_probe.py` | streams N tokens through the gateway (owner token from `/api/owner/bootstrap`), prints TTFT, decode tok/s, inter-token gap p50/p90/p99/max and CV across reps |
| `scripts/stress/tcp_blast.py` | raw TCP throughput G14 -> Mac on port 47999 (stdlib only); smoke-tested on loopback |

`itl_probe.py` and `tcp_blast.py` reach the Mac with `git pull`. The planner simulations (`predict.py`, `runs12.py`, `derive.py`, `matrix.py`, `sim_*.py`, `plan_sim*.py`) and their
outputs (`rows.json`, `runs12.json`) were scratch work run against a 143 MB cache of real GGUF headers, so they are
not in the repo. On test day the live prediction for any run is the plan `scripts/e2e_check.py` prints before it
deploys.

---

## 0. The twelve answers that decide the next push

| # | Question | Today's assumption (code) | Settled by | What flips |
|---|---|---|---|---|
| 1 | Host-RAM decode cost `c_cpu` on the G14 | 12.5 ms/GB prior (`planner.py:44`), but the G14 already measured ~18-20 (DERIVED, see 3.2) | C2 | prior value; break-even overflow `X* = H/(c_cpu - c_mac)` drops from 2.4 GB to ~0.5 GB |
| 2 | M4 Air Metal decode cost `c_mac` (8 or 10 GPU cores, fanless) | 10.85 ms/GB from an old third-party M4 row (`planner.py:44`) | C1, C6 | whether the Mac ever beats host RAM |
| 3 | Per-token network hop `H` on real Wi-Fi | 4.0 ms (`planner.py:45`), assumed | L1, L2 | overflow choice; need for a link probe |
| 4 | Does the Mac beat G14 RAM at equal overflow, end to end | auto picks RAM for 14B Q6_K ctx 4096 by 2.9 ms (SIM) | C3, `hypercompute tune` | default overflow tier; three-way overflow |
| 5 | Run-to-run noise of a decode measurement | tuner takes one sample per candidate and the minimum score wins (`tuner.py:531`) | C7, T1 | tuner repetitions and a significance margin |
| 6 | Real Mac memory budget | agent prints "up to 10.0 GiB" (`join.py:462`), host plans at most Metal-free minus the 4 GiB reserve, about 6.7 GiB (`deployments.py:148-156`) | M1, M2, M3 | budget formula; which models are pair-feasible |
| 7 | Compute buffers of full-offload splits | `COMPUTE_BASE + ub*n_embd*32` (`planner.py:36,69`), but split runs measured 1.2-2.1x that (3.1) | M5, M6 | VRAM margin; OOM or silent WDDM spill at the k_min split |
| 8 | Cold and warm load time of Mac layers over Wi-Fi | no progress signal exists ("RPC tensor transfer progress is not observable", IMPLEMENTATION_STATUS) | D1, D2, L3 | load progress UI; `--cache` default |
| 9 | How the system fails on real Wi-Fi (sleep, drop, worker crash) | 15 s stale rule (`config.py:16-17`), 180 s upstream idle (`config.py:27`) | F1-F6 | stall watchdog, heartbeat tolerance |
| 10 | macOS first run: binary, inventory, orphan handling, friend setup time | macOS arm64 binary never executed (IMPLEMENTATION_STATUS) | P1-P6 | packaging priority, macOS supervision |
| 11 | EXO on a real Mac: install cost, speed vs llama.cpp Metal | never run; several timeouts are guesses (`exo_state.py:18`, `exo.py:83-92`) | X1-X4 | keep or demote EXO |
| 12 | Whether priors and profiles carry across networks | profile key contains the peer's `ip:port` (`selector.py:247-250`, `profiles.py:21-24`) | T2, L1 on two networks | profile keying, per-session link probe |

---

## 1. What the code assumes today

| Constant | Value | Where | Status |
|---|---|---|---|
| `DECODE_MS_PER_GB["cuda"]` | 2.64 | `services/planner.py:44` | DERIVED from G14 idle runs (0.5B 526.7 tok/s, 7B 80.6 tok/s; split_optimization.md §1) |
| `DECODE_MS_PER_GB["metal"]` | 10.85 | same | DERIVED from llama.cpp discussion #4167 (M4 10-core GPU, old commit 8e672ef); UNVERIFIED for the friend's Air |
| `DECODE_MS_PER_GB["cpu"]` | 12.5 | same | ASSUMED (80 GB/s effective); G14 measured ~18 (IMPLEMENTATION_STATUS "Known gaps") |
| `DECODE_MS_PER_GB["vulkan"]` | 2.64 | same | borrowed from CUDA, no measurement |
| `HOP_MS` | 4.0 | `services/planner.py:45` | ASSUMED Wi-Fi round trip per RPC device per token |
| `decode_ms()` cost model | weights only, no KV term, no fixed per-token cost, no per-message overhead | `services/planner.py:296-304` | simplification of split_optimization.md §3.2 |
| `OVERHEAD_BY_KIND` | cuda 450 MiB, metal 256, vulkan 300, cpu 64 | `services/planner.py:38` | estimates |
| `RPC_STAGING` | 64 MiB per RPC device | `services/planner.py:39` | estimate |
| `COMPUTE_BASE`, `UBATCH` | 96 MiB, 512; compute = 96 MiB + 512 x n_embd x 32 | `services/planner.py:35-36,69` | estimate; too low for full-offload splits (3.1) |
| KV bytes | f16, `ctx x n_head_kv x (k+v) x 2` per layer | `services/planner.py:37,66-68` | MEASURED exact (14B ctx 4096: 704 MiB CUDA0 + 64 MiB CPU = 48 x 16 MiB) |
| Local VRAM headroom | 512 MiB | `agent/contribution.py:12`, `services/deployments.py:113` | estimate |
| Friend reserve | max(4 GiB, 20% of RAM) | `agent/contribution.py:21-24` | policy; 4 GiB on a 16 GB Air, 6.4 GiB on the 32 GB G14 |
| Friend max on unified memory | min(65%, total - 6 GiB) | `agent/contribution.py:43` | policy; 10.0 GiB on a 16 GB Air |
| Host CPU-tier budget | free RAM - reserve - 512 MiB | `services/deployments.py:123-132` | policy; was 7.46 GiB in the 14B run (`results/20261021T104014Z-auto-gw-14b-auto.json`) |
| Heartbeat / stale | 3 s / 15 s | `config.py:16-17` | untested on Wi-Fi |
| Host-gone worker stop | 300 s | `config.py:18` | MEASURED only on loopback/WSL |
| Probe timeout / unreachable re-probe | 25 s / 30 s | `config.py:24`, `services/nodes.py:37` | untested on Wi-Fi |
| Load / warmup / upstream idle | 1800 s / 120 s / 180 s | `config.py:25-27` | untested with a cold Wi-Fi load |
| Tuner workload | N_p 1024, N_g 256, 64 decode tokens, 1 sample, min score wins | `services/tuner.py:59-61,531`, `schemas.py:582-584` | no noise handling |
| vLLM | util cap 0.90, VRAM headroom 1 GiB, runtime overhead 1.5 GiB | `adapters/vllm.py:75-80` | one MEASURED run says activations 1.06 + graphs 0.03 GiB (lifecycle `14cb3362`) |
| AirLLM | process RAM 2 GiB, CUDA context 450 MiB | `adapters/airllm.py:72-73` | estimates |
| EXO | ack 60 s ("no upstream number exists; measure"), command 45 s, stall 600 s, MLX overhead 1 GiB per Mac | `adapters/exo_state.py:18`, `adapters/exo.py:83-92` | guesses |

---

## 2. What the current planner will choose on test day (SIM)

`plan_sim.py` runs the real `make_plan` (strategy `auto`) on the GGUF headers already on the G14. Prior = the
planner's own `decode_ms` in ms/token. "off GPU" = weight bytes not on CUDA0.

| Model, ctx | CUDA0 budget | Mac budget | auto picks | off GPU | prior: RAM vs Mac (ms/token) |
|---|---|---|---|---|---|
| 7B Q4_K_M, 4096 or 16384 | 10.5-11.4 GiB | any | CUDA0 only | 0 | 11.5 |
| 14B Q6_K, 4096 | 11.4 | 2-6.7 | RAM, 3 layers | 0.68 GB | 37.0 vs 39.9 |
| 14B Q6_K, 4096 | 11.0 | 2-6.7 | RAM, 5 layers | 1.13 GB | 41.4 vs 43.6 |
| 14B Q6_K, 4096 | 10.5 | 4-6.7 | RAM, 7 layers | 1.58 GB | 45.9 vs 47.3 (Mac at 2 GiB: infeasible) |
| 14B Q6_K, 16384 | 11.4 | 4 or 6.7 | **Mac**, 11 layers | 2.48 GB | 54.8 vs 54.7 |
| 14B Q6_K, 16384 | 11.4 | 2 | RAM, 11 layers | 2.48 GB | 54.8 (Mac infeasible) |
| Mistral-Small 3.2 24B Q4_K_M, 4096 | 11.4 | 4 or 6.7 | **Mac**, 9 layers | 3.08 GB | 67.2 vs 66.1 |
| Mistral 24B, 16384 | 11.4 | 6.7 | **Mac**, 13 layers | 4.38 GB | 80.0 vs 76.8 (RAM at Mac 4 GiB) |
| 14B Q8_0 (approx), 4096 | 11.4 | 6.7 / 4 | Mac / RAM, 14 layers | 4.10 GB | 79.7 vs 76.9 |
| 32B Q4_K_M (approx), 4096, host CPU budget 13 GiB | 11.4 | any | RAM, 29 layers | 8.50 GB | 135.0 (peers infeasible) |
| 32B Q4_K_M (approx), 4096, host CPU budget 6 GiB | 11.4 | 6.7 | CPU 8 + Mac 21 layers | 8.51 GB | 128.9 (mixed only as a fallback) |

What this says:

* Every Mac-vs-RAM decision on the 14B and 24B is decided by 0.1 to 3 ms/token (under 5%) on priors that are
  themselves off by 40-60% (`c_cpu`). The auto choice is effectively a coin flip until C1-C3 run.
* The prior break-even overflow is `X* = HOP_MS / (c_cpu - c_mac) = 4 / (12.5 - 10.85) = 2.4 GB`, so the 14B at
  ctx 4096 goes to RAM and the 14B at ctx 16384 and the 24B go to the Mac. With the G14's measured `c_cpu` of ~18,
  `X*` is ~0.56 GB and the Mac would win every overflow above about 3 layers, if `c_mac` and `H` hold.
* The 14B at ctx 16384 is the cheapest way to exercise the "Mac wins" branch: no new download, already in the
  catalog. The Mistral 24B GGUF is on the G14 (`~/.hypercompute/models/mistral-small-3.2-24b-instruct-q4_k_m`,
  sha256 `80f5bda6...`, 14,333,915,264 bytes) but has no entry in `configs/models.example.yaml`; it needs a local
  `configs/models.yaml` entry (gitignored) whose repo and revision are not recorded anywhere in the repo (UNVERIFIED).
* Mixed overflow (Mac + RAM) only happens when neither tier fits alone (`planner.py:378-402`). For the 32B with
  plenty of host RAM the planner never uses the Mac, even if the Mac is faster per byte (gap, see build B8).

---

## 3. Evidence already in hand that closes or sharpens items

### 3.1 Compute buffers are underestimated for full-offload splits (new gap, MEASURED)

| Run | Device | Log `compute buffer size` | Plan `compute_bytes_est` |
|---|---|---|---|
| 7B local (`20261021T031812Z-local_only-final-7b-local.json`) | CUDA0 | 143.01 MiB | 152 MiB |
| 7B CUDA0 + WSL CPU peer (`20261021T031942Z-all_selected-final-7b-cuda+wsl.json`) | CUDA0 | 239.05 MiB | 152 MiB |
| same | RPC0 | 176.04 MiB | 152 MiB |
| 14B gpu_ram, 4 CPU layers (`20261021T104014Z-auto-gw-14b-auto.json`) | CUDA0 | 159.38 MiB | 176 MiB |
| Mistral 24B CUDA0 + RPC0 = the M4 Air over Wi-Fi (`results/pair/20261021-campus-wifi-g14-m4air.json`, see 3.3) | CUDA0 | 366.05 MiB | 176 MiB (computed) |
| same | RPC0 (Metal) | 268.04 MiB | 176 MiB |
| 14B Q6_K CUDA0 + RPC0 = the M4 Air (same file) | CUDA0 | 257.05 MiB | 176 MiB |
| same | RPC0 (Metal) | 157.04 MiB | 176 MiB |
| 0.5B CUDA0 + RPC0 = the M4 Air (same file) | CUDA0 | 64.55 MiB | 110 MiB |
| same | RPC0 (Metal) | 47.04 MiB | 110 MiB |

INFERRED cause: llama.cpp enables pipeline parallelism (4 scheduler copies) only with full offload across several
devices (split_optimization.md §2.4), and any CPU layer turns it off (llama_cpu_tier.md §5). The planner's estimate
ignores it. At the 14B ctx-16384 Mac split (SIM: CUDA0 37 layers + head, 9.0 GB weights + 2.31 GiB KV) the CUDA0
estimate is ~11.3 GiB against an 11.4 GiB budget, so an extra 100-190 MiB of compute buffer can push it over.

### 3.2 Host-RAM cost re-derived

The ngl sweep in README "Measured so far" (27.5 / 22.8 / 17.9 tok/s at 4 / 6 / 9 CPU layers of the 14B Q6_K,
0.2255 GB per layer from the GGUF) gives `c_cpu - c_cuda` = 16.6-17.3 ms/GB, so `c_cpu` is about **19-20 ms/GB**
(DERIVED; IMPLEMENTATION_STATUS rounds it to ~18). Either way the prior 12.5 is 45-60% low.

### 3.3 Mistral 24B log: a real pair number (corrected after this file was written)

The `865929b9` llama-server log shows `RPC0 : 10.19.252.207:47052 (12124 MiB, 12123 MiB free)`. This section first
read that as a CUDA worker on the same 5070 Ti. It is not: 10.19.252.207 is the friend's M4 MacBook Air (the
gateway's node record has macOS 26.6.2, an Apple M4 and an `MTL0` working set of 12,712,935,424 B = 12,124 MiB), and
the G14 is 10.19.175.205. So this is a real G14 CUDA0 + M4 Air Metal split over campus Wi-Fi: `CUDA0 model buffer
size = 10136.80 MiB` on the G14 plus `RPC0 ... = 3165.57 MiB` on the Mac, about 2 min 42 s from the buffer line to
`n_ctx` (19.6 MiB/s end to end) and decode at 107 ms/token in one chat (median 9.1 tok/s over 12 chats, one
outlier at 3.5). The Mac was the last device (MVP order) and held the output head, so each token also sent 512 KiB of logits back over Wi-Fi. MEASURED, with the G14's power state
not recorded. Two more runs from the same session (0.5B with 4 Mac layers, 14B Q6_K with 2 Mac layers at 11.7-12.3
tok/s, 1,040 MiB uploaded in 87 s) are exported with this one to
`results/pair/20261021-campus-wifi-g14-m4air.json`. It is not evidence of a WDDM spill; M6 still stands on the
engine-tier sessions, where an over-budget CUDA load slowed down instead of failing.

### 3.4 Items closed by existing evidence

| Item | Source | Closed by |
|---|---|---|
| Repack skipped with CUDA0 present | llama_cpu_tier.md §7 row 2, A3 | the 14B gpu_ram log has `CPU_Mapped model buffer size = 1470.68 MiB` and no `CPU_REPACK` line |
| TRACE volume at `-lv 4` during generation | llama_server.md open items | largest llama-server log is 46,189 bytes / 484 lines for a whole deployment |
| vLLM decode and startup on the G14 | vllm.md §7 item 1 | RESULTS.md: 80.9 tok/s, ready in 51 s |
| vLLM pre-listen behaviour through the WSL relay | vllm.md §7 item 5 | timeouts, not refusals (`adapters/vllm.py:87`) |
| CPU compute buffer, one data point | llama_cpu_tier.md A3 | `CUDA_Host compute buffer size = 44.02 MiB` with 4 CPU layers (14B, ub 512, ctx 4096) |

---

## 4. Open items by area

Each row: the unknown, the procedure that settles it (commands in section 5, keyed by ID) and the decision it feeds.
"G14 only" items can run before test day.

### L. Link between the G14 and the Mac

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| L1 | TCP RTT p50/p90/p99 on each candidate network (phone hotspot, home router 5 GHz, travel router); split_optimization.md §1 has no measured figure | curl loop from the Mac to `/api/health` (unauthenticated on the LAN, `app.py:244-253`), 300 samples; `ping -n 200` from the G14; record band/BSSID | Pick the test network. **If p90 > 7 ms or p99/p50 > 5 on any network we would use, build the per-session link probe (B2)** and stop using a fixed `HOP_MS`. If p90 < 5 ms everywhere, keep a constant |
| L2 | Per-token hop `H` through the real RPC path (includes small-message overhead, `TCP_NODELAY` 3-write commands, split_optimization.md §2.1) | llama-bench 0.5B, 1-2 Mac layers vs local: `H ~ T(1) - T_cuda - 0.15 ms` | Replaces `HOP_MS = 4.0` (`planner.py:45`). **If `H` > (c_cpu - c_mac) x 1 GB (~7-9 ms with measured c_cpu), the Mac loses the 14B ctx-4096 overflow: keep RAM default** |
| L3 | Raw throughput G14 -> Mac (the AP double hop, split_optimization.md §1 "UNVERIFIED") | `tcp_blast.py`, 1000 MiB, 3 times | Predicts cold load time (D1) and prefill cost per ubatch (~23 MB per ubatch for 14B, DERIVED from §2.4). **If < 40 MB/s, the Mac is not a prefill-heavy target; tune with objective interactive only** |
| L4 | RTT while layers are uploading (bufferbloat) | L1 loop during the D1 cold load | **If heartbeat RTT p99 during load > 5 s, F1 risk is real: raise `STALE_AFTER_S`** |
| L5 | Exact message count and timing per decode step | `GGML_RPC_DEBUG=3` llama-bench run (split_optimization.md §2.3, research cites `ggml-rpc.cpp:685-690`) | Whether per-message overhead, not RTT, dominates `H`. **If GET_TENSOR wait << `H`, overhead is in small messages: watch upstream PR #24122, do not build our own** |
| L6 | Wired baseline | Optional: two USB-C Ethernet adapters and a cable; link-local 169.254/16 is accepted by join (`netutil.py:44-45`) | Separates Wi-Fi cost from RPC cost; tells us whether "use Ethernet" belongs in the README |

### C. Device coefficients and the overflow choice

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| C1 | `c_mac` for this Air: GPU core count, current build (b11770), Metal | (a) slope of llama-bench 7B at 1/4/8/12 Mac layers; (b) optional direct `llama-bench -dev MTL0` on the Mac with the 0.5B and 7B GGUFs copied over | Replaces `DECODE_MS_PER_GB["metal"]`. **If `c_mac` >= `c_cpu`, the Mac never wins on decode: it is capacity-only for a G14 host, and Mac-speed features (B8) are cancelled** |
| C2 | `c_cpu` and the best thread count on the HX 370 (llama_cpu_tier.md §4.2, §7) | G14 only: llama-bench 14B at `-ngl 46,45,44,41,37` (3-12 CPU layers), then `-t 6,8,10,12` at `-ngl 41` | Replaces `DECODE_MS_PER_GB["cpu"]`. **If the best `-t` beats the default (physical cores) by > 5%, store `threads` in the host profile (the adapter already passes `-t`, `llama_rpc.py:249`)** |
| C3 | Mac vs RAM head to head at equal overflow, through the gateway | llama-bench 14B `-ts 4/45,8/41` vs C2 rows at `-ngl 45,41`; then `e2e_check` 14B `all_selected` (Mac overflow) vs `local_only` (RAM overflow), and `hypercompute tune` at ctx 4096 and 16384 | **If the Mac's decode is faster by > 2 sigma (C7) at the 14B ctx-4096 overflow, the corrected priors already flip auto to the Mac: ship B1. If it loses at 0.7-1.1 GB but wins at 2.5+ GB, keep auto with corrected priors (no default flip)** |
| C4 | Logits penalty of the head on the Mac (IMPLEMENTATION_STATUS "measure the logits saving on Wi-Fi with the Mac") | llama-bench 7B `-dev RPC0/CUDA0,CUDA0/RPC0 -ts 6/23,23/6` | Confirms the order rule. **If the penalty is < 2 ms/token, the "head local" constraint can be relaxed when the local GPU cannot hold the head** (`planner.py:440-443` warning path) |
| C5 | Prefill with a Mac stage: `q_mac`, per-ubatch hop `H_pp`, overlap `lambda` (split_optimization.md §3.3, §6) | pp512 rows of C1/C3 at `-d 0` and `-d 4096`; gateway long-prompt TTFT from the `--bench` rows | Interactive score uses only short-prompt TTFT (`tuner.py:247-255`). **If the Mac adds > 1 s TTFT per 1k-token prompt, tune with N_p-weighted scoring or keep RAM for long-prompt users** |
| C6 | Fanless throttling under sustained decode | llama-bench 7B with 8 Mac layers, `-n 512 -r 10`, look at the `samples_ts` drift; `pmset -g therm` on the Mac | **If decode drops > 15% within 2 minutes, a device profile needs a "sustained" coefficient and the tuner must measure after warm-up** |
| C7 | Run-to-run noise | `-r 5` rows from C1-C3; `itl_probe.py --reps 5` (CV) | **If CV > 3%, or the tune's Mac-vs-RAM gap is < 2 sigma, ship B6 (tuner repetitions + margin)** |
| C8 | Value of `objective=throughput` for llama.cpp | `e2e_check` 7B `all_selected --engine llama_rpc` with `--objective throughput` vs default (without `--engine llama_rpc` the selector picks vLLM) | llama-server always runs `-np 1` (`llama_rpc.py:128`), so proportional splits (`planner.py:140-160`, `438`) can only help prefill. **If prefill gain < 20%, drop the proportional branch for llama.cpp (B13)** |

### M. Memory budgets and estimates

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| M1 | Metal `recommendedMaxWorkingSetSize` and GPU cores on this 16 GB Air (split_optimization.md §6) | Mac: `llama-server --list-devices`, `system_profiler SPDisplaysDataType`, `uv run hypercompute inventory` | Input to M2 and C1 |
| M2 | Effective Mac budget. INFERRED from `deployments.py:148-156`: free = min(Metal free, psutil available), minus the 4 GiB reserve, so <= ~6.7 GiB even though join prints "up to 10.0 GiB" | Read `budget_bytes` and `budget_basis` of the Mac assignment in each `results/*pair*.json`; note `psutil` available on the Mac at that time | **If the effective budget is < 4 GiB on a normal friend setup (browser open), the 14B ctx-16384 and 24B Mac plans are infeasible on the day: say so in the join output and README** |
| M3 | Real memory pressure on the Mac at 5-6.5 GiB of layers | Mac: `memory_pressure`, `vm_stat`, `sysctl vm.swapusage` before, during and after the 24B or 14B-16k Mac run, with the friend's usual apps open | **If free% stays > 20% and swap does not grow, apply the reserve to RAM only, not on top of the Metal ceiling (B5). If the Mac swaps, keep the formula and fix only the message** |
| M4 | Real CUDA context + desktop VRAM on the day; `OVERHEAD_BY_KIND["cuda"] = 450 MiB` | G14 only: `nvidia-smi -l 1` during loads; compare peak used with the sum of CUDA0 buffers in the load log; llama-bench 14B `-ngl 46` (3 CPU layers) load success | **If (used - buffers) < 300 MiB in every run and `-ngl 46` loads, lower the CUDA overhead to measured + 150 MiB (one more 14B layer on the GPU, +1-4% decode)** |
| M5 | Compute buffer of full-offload splits (3.1) | every pair run: `compute buffer size` lines vs plan `compute_bytes_est` | **If any split device's real compute buffer exceeds the estimate by > 64 MiB, ship B3 (multiply by measured factor, or plan with the PP copies)** |
| M6 | Windows WDDM sysmem fallback hides OOM (seen in the engine-tier sessions; 3.3 was the Mac, not a spill) | G14 only: llama-bench 14B `-ngl -1` (does not fit) and watch `\GPU Adapter Memory(*)\Shared Usage`; check NVIDIA Control Panel "CUDA - Sysmem Fallback Policy" (INFERRED driver feature) | **If an over-budget load succeeds and runs 2x+ slower, add a post-load shared-memory check that fails or warns (B3) and document "Prefer No Sysmem Fallback" for llama-server.exe** |
| M7 | Host free RAM at plan time decides the 32B outcome (SIM: gpu_ram at 13 GiB CPU budget, Mac+RAM mix at 6 GiB) | `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory` before each plan; plan's CPU `budget_bytes` | README.md:265 says the 32B "does not fit"; SIM says it fits in gpu_ram. **Correct the README after one real 32B plan** |
| M8 | KV estimate at long context (exact at 4096) | `llama_kv_cache: size` lines in the 14B ctx-16384 runs | Keep or fix `KV_BYTES_PER_ELEM` path; expected exact |

### D. Load path

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| D1 | Cold load time of Mac layers over Wi-Fi (split_optimization.md §5c: 2 GB at 37.5 MB/s = 53 s, UNVERIFIED) | `e2e_check` 7B `all_selected` right after the first join (empty `~/.hypercompute/rpc-cache` on the Mac); MB/s = RPC0 model buffer / (load_s - local load_s) | **If a cold load takes > 30 s, build load progress (B9): the agent can report the worker's received bytes in its heartbeat** |
| D2 | Warm load with `--cache` (`join.py:190-192`, `LLAMA_CACHE` under `rpc-cache`) | repeat D1; `du -sh ~/.hypercompute/rpc-cache` on the Mac | **If warm load < 10 s and cache size equals the Mac's layers, make `--cache` the default in join** |
| D3 | Join-to-eligible time (probe over Wi-Fi, `PROBE_TIMEOUT_S = 25`) | timestamps of the node's events in `~/.hypercompute/coordinator/records/events.jsonl` | **If any probe takes > 10 s, lower nothing; if probes time out, the 25 s timeout and the single-client worker (llama_rpc.md C1) need a retry** |
| D4 | Wall time of `hypercompute tune` on the pair (split_optimization.md §5c "1-3 minutes", UNVERIFIED) | `Measure-Command` around the tune | **If > 10 minutes, move the tuner to llama-bench style scans (B6)** |

### F. Failure paths on a real network

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| F1 | False "no heartbeat" on Wi-Fi with 3 s / 15 s | count `no heartbeat` lines in `events.jsonl` before and after 30 idle minutes plus all loads | **Any false positive on a healthy pair: raise `STALE_AFTER_S` to 30 s or require two misses (B4)** |
| F2 | Mac sleeps mid-generation (`caffeinate -i` does not stop a forced sleep) | Mac `pmset sleepnow` while `itl_probe.py --tokens 512` streams a 14B Mac split | Pass = client gets an error and the deployment fails within ~20 s, llama-server is gone, Mac returns to eligible after wake. **If the client hangs to the 180 s idle timeout, build the stall watchdog (B4)** |
| F3 | Wi-Fi drop of 5 s / 20 s / 60 s | Mac `networksetup -setairportpower en0 off`, wait, `on`, during generation | 5 s should stall and resume; 20 s and 60 s should fail cleanly. **Any hang > 30 s past the drop: B4** |
| F4 | Worker crash on the Mac mid-generation | Mac `pkill -9 ggml-rpc-server` | What error text reaches the host; `FAILURE_PATTERNS` (`llama_output.py:37-48`) has no Metal or socket-reset pattern. **If the run record says only "exit code", add the pattern** |
| F5 | Friend memory spike during a run | friend opens heavy apps during `itl_probe.py`; watch decode and `memory_pressure` | **If decode drops > 30% or the worker dies, show a per-friend "memory pressure" warning from the heartbeat (macOS reports it)** |
| F6 | Gateway hard-kill with a real Mac in the deployment (MEASURED only with loopback/WSL peers, lifecycle `8a8576ac`) | `Stop-Process -Force` on the gateway's PID during a ready 14B Mac split, restart within 5 min and again after 5 min | Pass = llama-server gone within ~1.5 s, Mac worker reusable without the friend doing anything (< 5 min), worker self-stops after 300 s (`config.py:18`) |
| F7 | Pause and leave mid-run on the pair (README step 8) | friend clicks Pause, then Resume; later `hypercompute leave` | Pass = "paused by contributor", clean failure, re-eligible after Resume |
| F8 | Friend changes network or IP | Mac switches from hotspot to router mid-session | Join requires worker address == source IP. **If the node goes ineligible with no recovery, build re-join on address change (B4)** |
| F9 | WAN-off demo (IMPLEMENTATION_STATUS "not executed") | turn off cellular data on the hotspot phone, or unplug the router WAN, then chat and start a new deployment | Pass = no dependency on internet for llama.cpp; EXO first start needs HF downloads (X5), so not in the WAN-off demo |
| F10 | Two users on a split (`-np 1`, 30 s slot wait, then 429) | two `itl_probe.py --tokens 1024` at once | **If friends will chat at the same time, either raise `-np` for splits (KV x n on the Mac) or show a queue position** |

### P. macOS first run (the friend's machine)

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| P1 | macOS arm64 b11770 binary executes (never run) | `uv run hypercompute engine install --yes` then `uv run hypercompute doctor` (smoke test); if blocked, `xattr -l` on the binary (INFERRED: Python-downloaded files carry no quarantine attribute) | Gate for everything. Fallback: `scripts/build_llama_macos.sh` (needs Xcode CLT) |
| P2 | Mac inventory: UMA counted once, GPU core count, Metal working set (`agent/inventory.py:116-140`) | `uv run hypercompute inventory` vs `system_profiler` | Fixes the budget input (M1) |
| P3 | macOS has no kill-on-close or PDEATHSIG (`tests/test_supervisor.py:105` skips on darwin) | Mac: `pkill -9 -f "hypercompute join"`, then `pgrep -fl ggml-rpc-server` and `lsof -nP -iTCP:47052 -sTCP:LISTEN` | **If the unauthenticated worker keeps listening after the agent dies, build a parent-watch for the worker on macOS (B4 security part)**; the next join's `reconcile_orphans` (`join.py:178-182`) only helps if the friend rejoins |
| P4 | macOS firewall prompt and the optional pf anchor (`join.py:493-509`) | note whether the prompt appears; optionally apply the printed pf rules and re-verify | README accuracy |
| P5 | Friend time-to-ready from a clean Mac: CLT, uv, private-repo clone (needs a GitHub account and collaborator access), engine install | stopwatch per step | **If > 15 minutes or the GitHub step blocks, single-download packaging (roadmap milestone M5, build B11) moves up; the wheel path (RUNBOOK §0) is the interim** |
| P6 | The portable test suite on darwin (never run on a Mac) | Mac: `uv run pytest -q` | INFERRED failure: `tests/test_vllm_install.py:314` runs on macOS and the reclaim script uses GNU `find -printf` (`installers/vllm.py:47`), which BSD find lacks. **Fix the skip condition if it fails** |

### T. Tuner and profiles

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| T1 | Winner stability: one sample, 64 decode tokens, min score (`tuner.py:531`) | run the same 14B tune twice | **If the winner flips between identical tunes, ship B6** |
| T2 | Profile replay after an IP change: the key holds `<ip:port>#i` (`selector.py:247-250`) | after the tune, rejoin the Mac on another network or after a DHCP change; check whether the next 14B plan says "Measured profile" (`heuristic=false`) | **If it falls back to "Estimate", key peers by hardware identity + network class (B10)** |
| T3 | Profiles never expire when the link changes (split_optimization.md §5d says link profiles are short-lived) | compare the tune winner on two networks (L1) | **If the winner differs by network, profiles need a link fingerprint (B10)** |
| T4 | Candidates never include a k sweep or a mixed Mac + RAM split (`tuner.py:362-404`) | compare C1/C3 llama-bench best with the tune winner | **If llama-bench finds a split > 5% better than every tune candidate, add k and mixed candidates (B6, B8)** |

### X. EXO (needs the Mac; a real cluster needs two)

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| X1 | Install on the friend's Air: Python 3.13, node >= 18, Rust >= 1.85 (`installers/exo.py:27-31`), macOS minimum (exo.md §9: README says 26.2, cask says 15), Metal toolchain | `time uv run hypercompute engine install --engine exo --yes`; note every missing prerequisite | **If > 20 minutes or it needs Rust/Node installs the friend would not do, EXO stays experimental (B12 demote)** |
| X2 | Single-Mac instance through the gateway: POST-to-visible time (`ACK_TIMEOUT_S = 60`, "measure"), `generation_stats` on streams (exo.md E5) | join with `--exo`, `e2e_check --engine exo` with `qwen3-0.6b-mlx-4bit` then `qwen2.5-7b-instruct-mlx-4bit` | Set `ACK_TIMEOUT_S` to measured p100 x 2; decide whether client-side timing is needed |
| X3 | MLX vs llama.cpp Metal on the same Mac | X2's 7B decode vs C1(b) `llama-bench -dev MTL0` 7B Q4_K_M | **If EXO >= 1.2x llama.cpp Metal, keep EXO as the default for all-Mac sessions and plan the 2-Mac test; if < 1.0x, demote (B12)** |
| X4 | Orphan cleanup: instance deleted at the next gateway start (`cleanup_orphan`) | hard-kill the gateway during an EXO deployment, restart | Pass = instance gone, leases released |
| X5 | Internet dependence: every Mac downloads the repo's current `main` (`exo.py:268-270` warning) | note download time and that the revision is not pinned | EXO cannot be in the WAN-off demo; pinning needs upstream support |

### V. vLLM on the G14 (G14 only)

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| V1 | Gateway hard-kill while vLLM serves: watchdog TERM then KILL after 30 s, GPU memory freed, `cleanup_orphan` on restart (contract-tested only) | V1 commands (section 5) | **If GPU memory is still held > 60 s after the kill, or the restart does not report "killed"/"nothing left", fix before showing vLLM to anyone** |
| V2 | Headroom 1 GiB and runtime overhead 1.5 GiB (`vllm.py:78-80`) | already one measurement: activations 1.06 + graphs 0.03 GiB + ~0.26 GiB non-torch, about 1.35 GiB | Keep; revisit only if a 14B AWQ is wanted (V4) |
| V3 | KV exhaustion and preemption at 32-64 users or long prompts (vllm.md §7 item 8 mid-stream error shape unread) | concurrency run at 32 and 64 users with 2k-token prompts through the gateway | **If the gateway returns a malformed stream or a hang, map vLLM's mid-stream error event** |
| V4 | Does a 14B AWQ fit 12 GiB with useful KV | needs a pinned catalog entry; `uv run hypercompute` plan with `engine=vllm` | Product: "Many users" for 14B on the laptop or not |

### A. AirLLM on the G14 (G14 only)

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| A1 | s/token for a 7B bf16 (airllm.md §11; only the 0.5B is measured: 0.2 tok/s, about 5 s/token) | needs a pinned `Qwen/Qwen2.5-7B-Instruct` HF entry (~15 GB); `e2e_check --engine airllm` | **If > 20 s/token, AirLLM stays hidden behind explicit `engine=airllm`; compare with the llama.cpp gpu_disk tier (G1) on the same size** |
| A2 | Kill mid-forward releases VRAM (airllm.md §11 last row) | Stop the deployment during a token; `nvidia-smi` | Pass = VRAM released within 10 s |
| A3 | 32B bf16 needs ~131 GB of disk (IMPLEMENTATION_STATUS) | `Get-PSDrive C` | Only if A1 says AirLLM is worth it |

### G. llama.cpp tiers and hardening on the G14 (G14 only)

| ID | Unknown | Settle by | Decision it feeds |
|---|---|---|---|
| G1 | gpu_disk tier never run (IMPLEMENTATION_STATUS) | hold RAM with a filler process so real free RAM is below the CPU part (Mistral 24B: 3.1 GB CPU part), plan `local_only` | **If > 5 s/token or the desktop freezes, make `allow_capacity_mode` default false (`schemas.py:350`) (B15)** |
| G2 | Windows paging behaviour when the tier exceeds RAM (llama_cpu_tier.md §7) | same run, watch Task Manager hard faults | same |
| G3 | llama.cpp config files are applied before CLI flags (llama_placement.md P2); no `APPDATA`/`XDG_CONFIG_HOME` isolation in `backend/` (grep finds none) | `Test-Path $env:APPDATA\llama.cpp\config.ini`; Mac `ls ~/.config/llama.cpp` | cheap hardening (B14) regardless of result |
| G4 | Linux PDEATHSIG path never executed (IMPLEMENTATION_STATUS) | run `tests/test_supervisor.py` and `tests/test_vllm.py::test_launch_and_stop_scripts_on_linux` inside WSL (both skip on Windows) | first real Linux evidence before the GX10 |
| G5 | RESULTS rows with prompt cache ON predate the fix (IMPLEMENTATION_STATUS) | rerun `e2e_check` 0.5B loopback with `--bench` | doc hygiene only |

### H. Later hardware

| ID | Hardware | Unknown | Test | Decision it feeds |
|---|---|---|---|---|
| H1 | second Mac | 3+ device chain over Wi-Fi (llama.cpp issue #28487 "deadlocks with 3+ endpoints") | G14 + 2 Macs, 7B `all_selected` x 10 runs, then 32B Q4_K_M `all_selected` (GPU-only first: 11.4 + 6.7 + 6.7 GiB of budget vs ~21 GiB needed) and `auto` (which may still pick host RAM) | **Any hang: cap RPC endpoints at 2 (B18). Clean: roadmap milestone M5 "more peers" proceeds** |
| H2 | second Mac | EXO 2-Mac ring over the AP: `ff12::` discovery multicast and zenoh link-local dial (exo.md §9) | `join --exo` on both Macs, 7B MLX `all_selected` | EXO investment (B12) |
| H3 | RTX 5090 desktop | Windows CUDA worker may silently serve CPU when the cudart DLLs fail to load (llama_placement.md P1); no check in `services/nodes.py:315-354` | join the 5090 as a friend; its card must show a CUDA device with ~32 GiB, not system RAM | **If it shows CPU or RAM-sized memory, add the "NVIDIA node must expose CUDA; reject total == 0" check (B14)** |
| H4 | RTX 5090 desktop | The planner fills the local GPU first (`planner.py:116-138`, `order_devices` `planner.py:287-293`), assuming the host is the fastest device | 32B Q4_K_M with G14 host + 5090 friend vs 5090 host + G14 friend | **If the 5090-host layout is > 20% faster, the dashboard should recommend the fastest GPU as host, or the planner needs speed-aware fill (needs per-device `c`, B1)** |
| H5 | RTX 5090 desktop | First wired link: `H` on Ethernet vs Wi-Fi | L1/L2 over Ethernet | Calibrates the link model |
| H6 | GX10 (GB10, sm_121) | Release `linux-arm64-cuda13.4` asset "NOT validated" for sm_121 (`configs/engines.lock.json:156-160`) | `uv run hypercompute engine install --yes && uv run hypercompute doctor`; `llama-server --list-devices` must show CUDA0 (type IGPU, free ~ MemAvailable per llama_rpc.md C2) | **"no kernel image" -> `scripts/build_llama_linux.sh --cuda-arch 121` (supported, `build_llama_linux.sh:9`) and a source-build variant in the lock** |
| H7 | GX10 | GB10 inventory: pool keyed by the name containing "gb10" (`agent/inventory.py:148-157`); nvidia-smi memory fields may read N/A | `uv run hypercompute inventory` | fix the name match or memory source |
| H8 | GX10 | vLLM 0.32 wheels for aarch64 + sm_121 (vllm.md §7 item 7 for mixed clusters) | `uv run hypercompute engine install --engine vllm` on native Linux | GX10 as the "Many users" box or not |

### Doc inconsistencies to fix after the test (do not edit before)

* README.md:265: "32B Q4_K_M ... Expected **does not fit** for this pair". SIM: with >= ~9.5 GiB of host CPU budget the
  planner picks gpu_ram (29 layers, 8.5 GB in RAM, ~5-7 tok/s with the measured `c_cpu`); with less RAM it mixes the
  Mac in. Only "does not fit on the GPUs" is true.
* README.md:256 and :264: the 14B Q8_0 goes to the Mac only if the Mac's effective budget is >= ~4.6 GiB; otherwise
  all 14 overflow layers go to host RAM (SIM).
* `join.py:462` prints "up to 10.0 GiB" while the host plans with at most ~6.7 GiB on a 16 GB Air (M2).
* `configs/profiles.example.yaml` says "Not yet measured (needs hardware): G14 + M4 MacBook Air".

---

## 5. Commands, keyed by item ID

### 5.0 Conditions and variables

G14 on AC, Armoury Crate mode recorded, dashboard tab open (it uses VRAM), nothing else heavy. Mac plugged in, lid
open, joined with `--cache`. Do llama-bench runs only when no deployment or tune is active: llama-bench holds the
Mac worker exclusively, and a gateway probe that lands meanwhile hangs for 25 s and marks the Mac "worker
unreachable" until the 30 s re-probe (`config.py:24`, `nodes.py:37`). Do not click Check plan during llama-bench.

```powershell
# G14, PowerShell
$E   = "$HOME\.hypercompute\engines\llama-b11770-windows-x64-cuda13.4"
$M   = "$HOME\.hypercompute\models"
$Q05 = "$M\qwen2.5-0.5b-instruct-q4_k_m\qwen2.5-0.5b-instruct-q4_k_m.gguf"
$Q7  = "$M\qwen2.5-7b-instruct-q4_k_m\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
$Q14 = "$M\qwen2.5-14b-instruct-q6_k\qwen2.5-14b-instruct-q6_k-00001-of-00004.gguf"
$MAC = "<MAC_IP>:47052"          # the port shown on the Mac's node card
$OUT = "$HOME\hc-testday"; New-Item -ItemType Directory -Force $OUT | Out-Null
$S   = "C:\Users\operator\hypercompute\scripts\stress"
Push-Location $E                 # Windows llama.cpp builds load ggml-*.dll from the exe dir and the cwd (llama_placement.md P1)
```

```bash
# Mac, Terminal
LLB=$(find ~/.hypercompute/engines -path '*macos-arm64*' -name llama-bench -type f | head -1)
LLS=$(find ~/.hypercompute/engines -path '*macos-arm64*' -name llama-server -type f | head -1)
G14=<G14_IP>
```

### 5.1 Link (L1-L5)

```bash
# L1 / L4, on the Mac (gateway running). time_connect = one TCP round trip.
for i in $(seq 300); do curl -s -o /dev/null -w '%{time_connect}\n' http://$G14:8000/api/health; sleep 0.2; done \
  | sort -n | awk '{a[NR]=$1*1000} END {printf "n=%d p50=%.1f p90=%.1f p99=%.1f max=%.1f ms\n", NR, a[int(NR*.5)], a[int(NR*.9)], a[int(NR*.99)], a[NR]}'
/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I 2>/dev/null || system_profiler SPAirPortDataType | head -40   # band, channel, PHY rate
```

```powershell
# L1, on the G14 (macOS answers ICMP unless stealth mode is on)
ping -n 200 <MAC_IP> | Select-Object -Last 4
netsh wlan show interfaces | Select-String "SSID|BSSID|Band|Channel|Receive rate|Transmit rate|Signal"
```

```bash
# L3, Mac first:
uv run python scripts/stress/tcp_blast.py --serve 47999
```

```powershell
# L3, then on the G14 (3 times):
uv run python "$S\tcp_blast.py" --send <MAC_IP>:47999 --mb 1000
```

```powershell
# L2: per-token hop. 0.5B has 24 layers = 25 units; one Mac layer costs ~0.15 ms of compute, the rest is the hop.
.\llama-bench.exe -m $Q05 -dev CUDA0 -ngl -1 -fa on -p 0 -n 128 -r 5 -o jsonl > "$OUT\L2-0.5b-local.jsonl"
.\llama-bench.exe -m $Q05 -rpc $MAC -dev RPC0/CUDA0 -ngl -1 -fa on -ts 1/24,2/23 -p 0 -n 128 -r 5 -o jsonl > "$OUT\L2-0.5b-mac.jsonl"
# H = 1000/avg_ts(1/24) - 1000/avg_ts(local) - 0.15 ms

# L5: per-command timing from the RPC client (research: level 3 logs "<CMD> done in X ms")
$env:GGML_RPC_DEBUG = "3"
.\llama-bench.exe -m $Q7 -rpc $MAC -dev RPC0/CUDA0 -ngl -1 -ts 4/25 -p 0 -n 32 -r 1 *> "$OUT\L5-rpc-debug.txt"
Remove-Item Env:GGML_RPC_DEBUG
```

### 5.2 Coefficients (C1-C8)

```powershell
# C1 slope: 7B = 28 layers = 29 units. T(k) = 1000/avg_ts of the tg rows.
.\llama-bench.exe -m $Q7 -dev CUDA0 -ngl -1 -fa on -p 512 -n 128 -d 0,4096 -r 3 -o jsonl > "$OUT\C1-7b-local.jsonl"
.\llama-bench.exe -m $Q7 -rpc $MAC -dev RPC0/CUDA0 -ngl -1 -fa on -ts 1/28,4/25,8/21,12/17 -p 512 -n 128 -d 0,4096 -r 3 -o jsonl > "$OUT\C1-7b-mac-sweep.jsonl"
# b = slope of T over k;  c_mac = b / 0.1407 GB + 2.64 ms/GB;  H = T(1) - T_local - b   (7B mean layer 0.131 GiB from the GGUF)

# C2 (G14 only, can run before test day): 14B = 48 layers = 49 units, CPU layers = 49 - ngl.
foreach ($ngl in 46,45,44,41,37) { .\llama-bench.exe -m $Q14 -dev CUDA0 -ngl $ngl -fa on -p 512 -n 128 -r 3 -o jsonl >> "$OUT\C2-14b-cpu-sweep.jsonl" }
foreach ($t in 6,8,10,12)       { .\llama-bench.exe -m $Q14 -dev CUDA0 -ngl 41 -t $t -fa on -p 512 -n 128 -r 3 -o jsonl >> "$OUT\C2-14b-threads.jsonl" }
# c_cpu = (T(12 CPU) - T(3 CPU)) / (9 x 0.2255 GB) + 2.64 ms/GB.  One process per ngl so an OOM at 46 only loses that row (M4).

# C3 head to head at 4 and 8 overflow layers (compare with the C2 rows at ngl 45 and 41)
.\llama-bench.exe -m $Q14 -rpc $MAC -dev RPC0/CUDA0 -ngl -1 -fa on -ts 4/45,8/41 -p 512 -n 128 -r 5 -o jsonl > "$OUT\C3-14b-mac.jsonl"

# C4 head placement: keep rows (RPC0/CUDA0, 6/23) = 6 Mac layers, head local; (CUDA0/RPC0, 23/6) = 5 Mac layers + head
.\llama-bench.exe -m $Q7 -rpc $MAC -dev RPC0/CUDA0,CUDA0/RPC0 -ngl -1 -fa on -ts 6/23,23/6 -p 512 -n 128 -r 3 -o jsonl > "$OUT\C4-7b-order.jsonl"

# C6 sustained decode (watch samples_ts drift)
.\llama-bench.exe -m $Q7 -rpc $MAC -dev RPC0/CUDA0 -ngl -1 -fa on -ts 8/21 -p 0 -n 512 -r 10 -o jsonl > "$OUT\C6-7b-sustained.jsonl"
Pop-Location
```

```bash
# C1(b) optional, on the Mac with the GGUFs copied into ~/hc-models (both 7B shards in one folder)
"$LLB" -m ~/hc-models/qwen2.5-0.5b-instruct-q4_k_m.gguf -dev MTL0 -ngl -1 -fa on -p 512 -n 128 -r 3 -o jsonl > ~/C1-0.5b-mtl0.jsonl
"$LLB" -m ~/hc-models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -dev MTL0 -ngl -1 -fa on -p 512 -n 128 -r 3 -o jsonl > ~/C1-7b-mtl0.jsonl
pmset -g therm                                   # C6: thermal warnings without sudo
```

Gateway-level runs (repo folder on the G14, gateway running; these write `results/*.json`):

```powershell
uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-q4_k_m --strategy all_selected --bench --label pair-7b-g14+mac-cold   # D1
uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-q4_k_m --strategy all_selected --bench --label pair-7b-g14+mac-warm   # D2
uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-q4_k_m --strategy local_only   --bench --label pair-7b-local
uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-q4_k_m --strategy all_selected --objective throughput --bench --label pair-7b-throughput   # C8
uv run python scripts/e2e_check.py --model qwen2.5-14b-instruct-q6_k  --strategy local_only   --bench --label pair-14b-ram        # C3 RAM overflow
uv run python scripts/e2e_check.py --model qwen2.5-14b-instruct-q6_k  --strategy all_selected --bench --label pair-14b-mac        # C3 Mac overflow (run together = GPU-only first)
uv run python scripts/e2e_check.py --model qwen2.5-14b-instruct-q6_k  --strategy auto --ctx 16384 --bench --label pair-14b-16k-auto   # SIM says Mac if budget >= 4 GiB
Measure-Command { uv run hypercompute tune qwen2.5-14b-instruct-q6_k --max-candidates 4 }                  # C3, D4, T1 (run twice)
uv run hypercompute tune qwen2.5-14b-instruct-q6_k --ctx 16384 --max-candidates 4
uv run python scripts/e2e_check.py --model qwen2.5-14b-instruct-q6_k --strategy auto --keep --label pair-14b-itl
uv run python "$S\itl_probe.py" --tokens 256 --reps 5 --label pair-14b-auto                                   # C7, then stop the kept deployment in the dashboard
```

### 5.3 Memory (M1-M8)

```bash
# Mac: M1, P2, M3
"$LLS" --list-devices                                   # MTL0: Apple M4 (<total> MiB, <free> MiB free) = Metal working set
system_profiler SPDisplaysDataType | grep -iE "chipset|cores"
sysctl -n hw.memsize; uv run hypercompute inventory
memory_pressure | tail -1; vm_stat | grep -E "Pageouts|Swapouts"; sysctl vm.swapusage   # before, during, after each Mac split
```

```powershell
# G14: M4 / M6 / M7 in a second terminal during loads
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 1 > "$OUT\M4-vram.csv"
Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 1 -MaxSamples 60 | Out-File "$OUT\M6-shared.txt"
(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB      # GiB free before each plan (M7)
# M6 over-budget load: 49/49 units on CUDA0 (~12.2 GiB). Expect an OOM; a success that runs slowly means sysmem fallback.
Push-Location $E; .\llama-bench.exe -m $Q14 -dev CUDA0 -ngl -1 -fa on -p 0 -n 64 -r 1 -o jsonl > "$OUT\M6-14b-allgpu.jsonl"; Pop-Location
# M5 after the pair runs
Select-String -Path results\*pair*.json -Pattern "compute buffer size = [ 0-9.]+ MiB" -AllMatches | ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
Select-String -Path results\*pair*.json -Pattern '"compute_bytes_est": \d+' -AllMatches | ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
```

### 5.4 Failure paths (F1-F10, P3)

```powershell
# F1 before and after the session
(Select-String -Path "$HOME\.hypercompute\coordinator\records\events.jsonl" -Pattern "no heartbeat").Count
# F6 gateway hard-kill during a ready Mac split
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess -Force
Get-Process llama-server -ErrorAction SilentlyContinue      # expect nothing within ~1.5 s
uv run hypercompute host                                    # restart within 5 min, then run the 7B split again without touching the Mac
# F10 two users at once on a split (two terminals)
uv run python "$S\itl_probe.py" --tokens 1024 --reps 1 --label user-a
uv run python "$S\itl_probe.py" --tokens 1024 --reps 1 --label user-b
```

```bash
# Mac, while itl_probe.py --tokens 512 streams a 14B Mac split
pmset sleepnow                                                           # F2
networksetup -setairportpower en0 off; sleep 5;  networksetup -setairportpower en0 on   # F3 (then 20 and 60)
pkill -9 ggml-rpc-server                                                 # F4
# P3 orphan check
pkill -9 -f "hypercompute join"; sleep 2; pgrep -fl ggml-rpc-server; lsof -nP -iTCP:47052 -sTCP:LISTEN
```

### 5.5 EXO (X1-X4)

```bash
# Mac
time uv run hypercompute engine install --engine exo --yes
caffeinate -dims uv run hypercompute join --host $G14:8000 --code <CODE> --cache --exo
```

```powershell
# G14
uv run python scripts/e2e_check.py --model qwen3-0.6b-mlx-4bit --engine exo --strategy all_selected --bench --label pair-exo-0.6b
uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-mlx-4bit --engine exo --strategy all_selected --bench --label pair-exo-7b
```

### 5.6 G14-only items (before test day)

```powershell
# V1 vLLM hard-kill
uv run python scripts/e2e_check.py --model qwen2.5-7b-instruct-awq --engine vllm --strategy local_only --keep --label vllm-prekill
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess -Force
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1          # expect release within ~30-40 s (watchdog TERM, KILL after 30 s)
wsl -d Ubuntu -- pgrep -af "vllm"                                         # expect nothing after the watchdog
uv run hypercompute host                                                  # deployment record should show the orphan cleanup note
# G1 gpu_disk: hold RAM so real free RAM < the 24B's 3.1 GB CPU part (adjust 16 to free GiB - 2)
uv run python -c "b = b'\x01' * (16 * 2**30); input('holding 16 GiB; Enter to release')"
# G4 Linux-only tests inside WSL (uv at ~/.local/bin/uv, installed by scripts/install_vllm_wsl.sh:80-86), separate venv so the Windows .venv is untouched
wsl -d Ubuntu -- bash -lc 'cd /mnt/c/Users/operator/hypercompute && UV_PROJECT_ENVIRONMENT=$HOME/.hc-venv ~/.local/bin/uv run pytest -q tests/test_supervisor.py tests/test_vllm.py -k "hard_killed or launch_and_stop"'
```

---

## 6. Candidate build list for the next push, ranked by expected value

EV = likelihood the test triggers it x user-visible impact / cost. Sizes: S < 1 day, M 1-3 days, L > 3 days.

| Rank | Build item | Triggered by | Cancelled by | Size | Where |
|---|---|---|---|---|---|
| B1 | **Measured priors and a per-node calibration store.** Replace `DECODE_MS_PER_GB` cpu/metal and `HOP_MS` with measured values; store per-device `c` and per-link `H` (split_optimization.md §5d kinds `device` and `link`) from a llama-bench calibration run instead of full deployments | C1/C2/L2 differ from priors by > 15% (C2 is already ~50% off) | priors within 15% on every coefficient | S for constants, M for the store | `planner.py:40-45,296-304`, `services/profiles.py`, `tuner.py` |
| B2 | **Per-session link probe** in the heartbeat (agent echoes a gateway timestamp; p50/p90 RTT per node), used as `H` and shown on the node card with a "bad network" warning | L1 p90 > 7 ms, p99/p50 > 5, or `H` falls on opposite sides of the break-even on two networks (T3) | p90 < 5 ms and stable across every network tried | M | `agent/join.py` heartbeat, `services/nodes.py`, `planner.py:45` |
| B3 | **VRAM safety for splits.** Compute estimate that includes pipeline-parallel copies (or the measured factor 1.6-2.1x), plus a post-load check for WDDM shared-memory spill | M5 (already 1.2-2.1x in existing logs) or M6 shows a silent spill | M5 within 64 MiB on every pair run and M6 OOMs cleanly | S | `planner.py:36,69,96-107`, `adapters/llama_rpc.py` load parse |
| B4 | **Real-network failure hardening.** Stall watchdog (no token for k x expected ITL while a peer's heartbeat is late: abort and fail fast), heartbeat tolerance (two misses or 30 s), re-join on friend IP change, macOS worker parent-watch so the unauthenticated port closes when the agent dies | F1 false positive, F2/F3 client hang > 30 s, F8 no recovery, P3 orphan still listening | all F-items pass within 20 s and P3 shows no orphan | M | `config.py:16-17`, `services/streaming.py`, `services/nodes.py`, `agent/join.py` |
| B5 | **Mac budget formula and honest join message.** Apply the reserve to RAM-available only, cap by the Metal working set with a margin; print the effective budget, not `max_bytes` | M2 effective budget < max printed, and M3 shows no pressure at 6.5 GiB | M3 shows swap growth at <= 6.5 GiB (then fix only the message) | S | `services/deployments.py:135-158`, `agent/join.py:462`, `agent/contribution.py` |
| B6 | **Tuner robustness.** 3 repetitions per candidate, winner must beat the planner pick by 2 sigma else keep the pick with fewer off-GPU bytes (split_optimization.md §5c), k sweep and mixed candidates, optional llama-bench fast scan | C7 CV > 3%, T1 winner flips, T4 finds a better split, or D4 > 10 min | CV < 2% and identical winners across two tunes | M | `services/tuner.py:59-72,164-174,362-404,526-543` |
| B7 | **Inter-token latency in run records and benchmarks** (p50/p90/p99 gap, max stall, count of gaps > 3x p50) | always useful; required to judge Wi-Fi tails (L1, F3) | none | S | `services/streaming.py`, `services/benchmarks.py`, `scripts/summarize_results.py` |
| B8 | **Three-way overflow (Mac + host RAM) even when RAM alone fits.** SIM for the 32B: RAM-only 29 layers = ~182 ms/token with measured `c_cpu`; Mac 21 + RAM 8 = ~142 ms (-22%) if `c_mac` ~ 10.85 and `H` ~ 4 | C1/C3: `c_mac + H/X < c_cpu` for X >= 2 GB | `c_mac >= c_cpu` (Mac capacity-only) | M | `planner.py:378-402` (`_overflow`), tuner candidates |
| B9 | **Load progress and `--cache` by default.** Agent reports the worker's received bytes in the heartbeat; dashboard shows MB received of the planned Mac bytes; join defaults to `--cache` | D1 cold load > 30 s; D2 warm < 10 s | D1 < 15 s for the 14B | S-M | `agent/join.py:186-192`, heartbeat schema, dashboard stepper |
| B10 | **Profile keys by hardware identity + link fingerprint**, not `ip:port` | T2 profile not replayed after an IP change, or T3 winner differs by network | T2 replays and T3 winners agree | S | `services/selector.py:247-250`, `services/profiles.py:21-24` |
| B11 | **Single-download friend app (roadmap milestone M5)** | P5 > 15 min or the GitHub/collaborator step blocks | P5 < 10 min with no blocker (keep the wheel path from RUNBOOK §0) | L | packaging |
| B12 | **EXO: invest or demote.** Invest = 2-Mac test, measured `ACK_TIMEOUT_S`, `generation_stats` telemetry. Demote = label experimental and stop work | invest if X3 >= 1.2x llama.cpp Metal and X1 < 20 min | demote if X3 < 1.0x or X1 needs Rust/Node installs friends will not do | S (demote) / M (invest) | `adapters/exo*.py`, README |
| B13 | **Drop proportional splits for llama.cpp** (it runs `-np 1`); route "Many users" to vLLM or local-only | C8 prefill gain < 20% | C8 prefill gain >= 20% | S | `planner.py:438`, selector |
| B14 | **Peer and launch hygiene.** NVIDIA peer must expose a CUDA device, reject probe devices with total == 0 (llama_rpc.md C2); launch llama.cpp children with `APPDATA`/`PROGRAMDATA`/`XDG_CONFIG_HOME` pointed at an empty owned dir and warn on `using config file:` (llama_placement.md P2) | H3 shows CPU fallback, or G3 finds a config.ini on any machine; cheap enough to do anyway | none | S | `services/nodes.py:315-354`, `agent/supervisor.py:189-205`, `adapters/llama_rpc.py` |
| B15 | **gpu_disk opt-in** (`allow_capacity_mode` default false) | G1 > 5 s/token or desktop freeze | G1 < 2 s/token and responsive | S | `schemas.py:350`, dashboard toggle |
| B16 | **CUDA overhead calibration** (`OVERHEAD_BY_KIND["cuda"]` 450 MiB -> measured + 150 MiB) | M4: used - buffers < 300 MiB every run and the 3-CPU-layer 14B loads | M4 shows >= 450 MiB real overhead | S | `planner.py:38` |
| B17 | **GX10 path**: validate the arm64 CUDA asset for sm_121 or add a source-build variant; GB10 inventory | H6/H7 on the GX10 | asset works as is | S-M | `configs/engines.lock.json:156-175`, `agent/inventory.py:148-157` |
| B18 | **Cap RPC endpoints at 2 until validated** | H1 hang or abort with 2 Macs | 10/10 clean 3-device runs | S | planner device selection |

Suggested order for the push itself, if the test results land as the evidence so far predicts (c_cpu ~19, Mac
budget ~6.7 GiB, compute buffers under-estimated): B1 (constants first, store second), B3, B7, B5, B2, B4, B6, then
B8/B9/B10 depending on C1-C3, D1 and T2. B11-B18 are each gated on one specific result above.

---

## 7. Test-day order (about 3.5 hours, IDs refer to section 4)

Before the day, on the G14 alone: C2, M4, M6, G1-G4, V1, A2 (about 2 hours, no Mac needed).

1. Network choice: L1 on each candidate network (10 min each), keep the best; record band and BSSID.
2. Mac first run: P1, P2, M1, P5 timings (the friend), then join with `--cache`.
3. L3, then L2, C1, C4, C6, C3 llama-bench block (about 30 min, no deployment active).
4. Gateway runs: D1, D2, 7B local, C8, 14B RAM vs Mac, 14B ctx 16384 auto, M3 on the Mac during the Mac splits.
5. Tune twice at ctx 4096, once at 16384 (C3, D4, T1); C7 with `itl_probe.py`.
6. Failure paths: F7, F10, F2, F3 (5/20/60 s), F4, F6, P3, F9. Keep F1 counting across the whole session.
7. EXO last (X1-X4), because the install may take long and the session must not depend on it.
8. `uv run python scripts/export_lifecycle_evidence.py`, `uv run python scripts/summarize_results.py`, copy
   `$HOME\hc-testday` and the Mac's `~/C1-*.jsonl` next to `results/`.
