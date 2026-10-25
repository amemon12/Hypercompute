# Plan predictions for the next hardware test (real planner + selector)

> **Added after review (2026-10-21).** The friend's M4 MacBook Air had already run three deployments with this G14 over campus Wi-Fi on the MVP build (`results/pair/20261021-campus-wifi-g14-m4air.json`, summarized in [../TEST_PLAN.md](../TEST_PLAN.md) section 0). Where this file says the macOS binary, the Mac inventory or a real Mac split has never run, read it against that evidence: the binary ran, the Metal working set is 12,124 MiB, the Mac's planner budget was 2.9-4.4 GiB, and its splits decoded at about 12 tok/s (14B) and a median 9.1 tok/s (Mistral 24B) on the old local-first order, where the Mac held the output head and sent 0.5-0.6 MB of logits back per token.

Repo `C:/Users/operator/hypercompute` at `dfd8cf9`. Every plan below was produced by the repo's own code
(`hypercompute.services.selector.choose_plan` -> `planner.make_plan`, the real vLLM and AirLLM `plan()` functions,
`deployments.assign_rpc_names` for the `--device` names). Nothing was started: no gateway, no model load, no repo edit,
no commit. 396 grid rows + 24 selector rows + 30 budget-sensitivity rows, all in a scratch
`rows.json` that is not in the repo (see section 1).

## 0. The short version

1. **Every one of the 396 rows is feasible.** Nothing is refused, because the `gpu_disk` tier (mmap paging) catches
   everything the GPUs and RAM cannot hold. So "does it fit" is never the test. The real questions are speed,
   memory margins and whether the overflow choice is right.
2. **A 16 GB M4 Air will not contribute 10 GiB.** `remote_budgets()` subtracts the friend's 4 GiB reserve from
   `min(Metal working set 10.67 GiB, free unified RAM)` (`services/deployments.py:150-156`). The result is 6.0 GiB at
   10 GiB free, 6.67 GiB at 12 GiB free and 4.0 GiB at 8 GiB free. **S3 (6 GiB) is the real default pairing, and S2
   (10 GiB) only happens with `join --reserve ~0.7GiB`.** With gateway budgets, 6 plans change tier versus the
   brief's numbers. For example, 32B Q4_K_M at 4k goes from Mac split `-ts 29,36` to 30 CPU layers, and 14B Q8_0 at
   16k/32k goes from Mac split to RAM.
3. **The overflow choice is a coin flip under the priors.** 28 rows are within 10% (peers vs host RAM). 9 rows flip
   when the host-RAM cost is the G14's measured ~18-20 ms/GB instead of the 12.5 prior. The flip rows include the
   flagship 14B Q6_K @4k on the Air (RAM 37.0 vs Mac 39.9 ms by prior; 40.7 vs 39.9 at 18 ms/GB). The existing
   evidence also implies a per-token RPC hop of about 6 ms on the WSL virtual switch, against `HOP_MS = 4`.
4. **`auto` misses a plan its own cost model prefers.** When host RAM alone can take the overflow, `_overflow()` never
   tries "peers + RAM together" (`services/planner.py:388-402`), and the tuner does not either (`services/tuner.py:3-6`).
   On the realistic 6 GiB Mac this leaves the Mac idle and puts 30-35 layers on the CPU. Example: 32B Q4_K_M @4k on S3
   is 30 CPU layers, 138.0 ms (186.4 at 18 ms/GB), while run-together is 11 CPU + 19 Mac, 133.0 ms (151.5). At the measured CPU cost `auto` is 2-36% slower in the 11 grid rows of section 6 (39% with two default-join Airs, run 8).
5. **`gpu_disk` is mislabeled in 7 of its 18 auto rows.** llama.cpp mmaps the CPU layers in both RAM tiers (the 14B
   run logged `CPU_Mapped model buffer size = 1470.68 MiB` = token_embd 609 + 4 layers x 215.4). When the paged part
   fits the ~13-18 GiB page cache, decode runs at RAM speed (e.g. 32B Q4_K_M @32k on S1: ~163-230 ms/token). The
   capacity-mode warning still says "expect seconds per token" (`planner.py:484-488`). When the paged part is bigger
   than the page cache (all 72B rows on S1-S3), a cyclic scan defeats LRU and every token re-reads it: 5-22 s/token.
6. **Margins are razor thin by design.** Greedy fill-to-capacity leaves under 3% headroom on some device in 226/396
   rows. Examples: CUDA0 at 0.2% for 14B Q6_K @4k, the 5090 at 0.3% for 72B, the Mac at 0.0% for 72B @32k on S3.
   The estimate held for the one measured case (14B Q6_K: 11.149 of 11.164 GiB planned, loaded fine at 26.7 tok/s).
   But the compute-buffer estimate does not grow with context (`planner.py:67`), and **124 rows at 16k/32k only fit
   if flash attention engages on every device**. Without FA, llama.cpp's worst-case KQ buffer at 32k is 2.5 GiB
   (14B/32B) or 4 GiB (72B). Evidence that FA engaged on the local CUDA run: 159 MiB compute buffer at 4k, below the
   320 MiB no-FA KQ size. There is no evidence yet for RPC or Metal.
7. **The cost model has one number per backend kind** (`planner.py:44`, `_kind` at `planner.py:76-89`). The RTX 5090
   and the GB10 cost the same 2.64 ms/GB as the 5070 Ti Laptop. On S5 the local-first fill keeps the slower laptop GPU full (45 of 48 layers of 14B Q6_K, 35 of 64 of 32B Q4_K_M); bandwidth-adjusted, the throughput split that loads the 5090 is 10-40% faster even single-stream. On the GX10 the
   prior is about 1.8x optimistic (72B: 8.7 tok/s by prior, ~4.9 by bandwidth).
8. **Selector.** vLLM takes the AWQ variant for every `throughput` request of the 7B family, **including 7B Q8_0**.
   That is a silent 8-bit to 4-bit swap (`services/selector.py:101-105`). The plan reserves 10.63 GiB at util 0.89
   and its KV room is 3.94 GiB, so it holds about 2 full 32k requests at `--max-num-seqs 16`. AirLLM never serves:
   32B bf16 needs 124.1 GiB of disk (61 GiB download + 61 GiB split + 2 GiB) and C: has 25.2 GiB free. 72B has no
   HF variant in the manifest, so it has no AirLLM fallback at all. EXO is never chosen: it needs every selected peer to be a Mac running EXO, and the only MLX variants (7B, Qwen3 0.6B) are models llama.cpp already fits on CUDA0 (`selector.py:107-111`).
9. **Test-day blockers that are not planner bugs.** C: has 25.2 GiB free (98% full), and `models pull` needs file +
   2 GiB (`services/models.py:184`). So 32B Q6_K (25.04 GiB) and 72B Q4_K_M (40.99 GiB) cannot be downloaded, and
   14B Q8_0 + 32B Q4_K_M (33.1 GiB) cannot coexist. 6 of the 11 grid models have no manifest entry: 1.5B, 3B,
   7B Q8_0, 14B Q4_K_M, 32B Q6_K and 72B Q4_K_M.
10. **What never happens:** the Mac never gets the output head. In all 396 rows the head stays on CUDA0, because the
    G14 can always hold the 975 MiB 72B head. No row was infeasible.

## 1. Method (reproducible)

```
cd C:/Users/operator/hypercompute
uv run --no-sync python predict.py   # rows.json (grid, selector, sensitivity)
uv run --no-sync python runs12.py    # runs12.json (the 12 test-day runs, brief vs gateway budgets)
python -I derive.py                  # tables (stdlib only)
python -I matrix.py
```

The planner simulations (`predict.py`, `runs12.py`, `derive.py`, `matrix.py`, `sim_*.py`, `plan_sim*.py`) and their
outputs (`rows.json`, `runs12.json`) were scratch work run against a 143 MB cache of real GGUF headers, so they are
not in the repo. On test day the live prediction for any run is the plan `scripts/e2e_check.py` prints before it
deploys.

**Model shapes.** The three downloaded GGUFs use the real `ModelService().summary(id)`. For the other eight I did not
estimate. I read each shard's real tensor table:

- An HTTP Range GET took the first 10 MiB of shard 1 and the first 1 MiB of later shards from
  `Qwen/Qwen2.5-*-Instruct-GGUF` (43 shard headers in `testplan/hdr`).
- The repo's own `hypercompute.gguf.read_file` parsed them. A duck-typed path supplies the shard's real byte size
  from the HF tree API, which is only used for the last tensor's length.
- I validated the method on the three local files. It reproduces `ModelService().summary` byte for byte (layer_bytes,
  token_embd, output). n_layer, n_embd, heads and kv heads match `config.json` for all 7 sizes, and the vocab is
  151936 for 0.5B-3B and 152064 for 7B+.
- I also built the planner-test-style estimate (`tests/test_split_order.py:26-31`). It uses uniform layers =
  (file - 5.95 MB metadata - 2 x head) / n_layer, with head = vocab x n_embd x nominal bpw (4.85 / 6.5625 / 8.5), and
  token_embd = output. It gives the same plan in 89 of 99 cells. The other 10 cells differ by one layer, all 32B
  Q4_K_M or 72B Q4_K_M (mixed Q4_K/Q6_K layers, and a 975 MiB Q6_K head against the estimate's 720 MiB). Appendix A
  lists them.

**Hardware scenarios (the brief's numbers, helpers as in `tests/test_split_order.py:42-62`):**

| scenario | local | peers (RPC, `--device` first) | CPU tier | host free RAM |
|---|---|---|---|---|
| S1 | CUDA0 5070 Ti Laptop 11.40 GiB | none | 13.28 GiB = 20 - 6.22 reserve - 0.5 (`deployments.py:123-132`; G14 reserve_bytes 6,682,544,537 recorded in `results/*14b-auto.json`) | 20 GiB |
| S2 | same | M4 Air, metal, unified, 10.0 GiB | same | 20 GiB |
| S3 | same | M4 Air 6.0 GiB | same | 20 GiB |
| S4 | same | 2x M4 Air 10.0 GiB (192.168.1.21 = RPC0, .22 = RPC1) | same | 20 GiB |
| S5 | same | RTX 5090, cuda, discrete 31.0 GiB | same | 20 GiB |
| S6 | GX10 CUDA0, unified pool `uma`, 110 GiB | none | 110 GiB on the same `uma` pool | 119 GiB |

The CPU budget a row shows (12.5-12.9 GiB) is lower than 13.28 because token embeddings come off it first
(`planner.py:226-230`). Engines available: llama_rpc, vllm and airllm on the G14 (the real install records), and
llama_rpc only on the GX10. The vLLM planner got the G14's measured nvidia-smi row (11.94 GiB total, 11.66 GiB free)
and an assumed 200 GiB of WSL disk. AirLLM read the real local state (installed, 25.2 GiB free on C:). Profiles were
off (`use_profiles=False`). The only stored profile, 14B Q6_K, is keyed to the old loopback worker and would not
replay for any scenario here (`selector.py:253-276`, signature in `services/profiles.py:21-24`).

**Decode estimates.** "prior" is the planner's own `decode_ms` (`planner.py:296-304`). That is 2.64 ms/GB CUDA,
10.85 Metal, 12.5 CPU, plus 4 ms per RPC device. "(@18)" re-costs the same layout with CPU = 18 ms/GB, the G14's
measured figure (`docs/IMPLEMENTATION_STATUS.md:78`). The ngl sweep 45/43/40 = 27.5/22.8/17.9 tok/s
(`IMPLEMENTATION_STATUS.md:29`) gives a marginal 19.2-20.3 ms/GB, so @18 is if anything generous to RAM.

**Calibration against runs that already exist:**

| case | predicted | measured | verdict |
|---|---|---|---|
| 7B Q4_K_M local, S1 | `-ts 29` on CUDA0, 11.5 ms = 86.7 tok/s | 80.6-87.1 tok/s (`docs/RESULTS.md:63,69`) | prior OK |
| 14B Q6_K auto, S1 with the gateway's own budget (CUDA0 11.16 GiB) | gpu_ram, 4 CPU layers, `-ngl 45`, 39.2 ms = 25.5 tok/s | CPU 4L + CUDA0 44L, offloaded 45/49, 26.7 tok/s (`RESULTS.md:33,71`) | placement exact, speed within 5% |
| 0.5B local | 1.0 ms = 1000 tok/s | 1.9 ms = 526.7 tok/s (`RESULTS.md:59`) | ~1 ms fixed per-token cost missing from the prior |
| 0.5B CUDA0 18L + WSL CPU worker 6L | prior ~5.6 ms (0.9 CUDA + 0.8 CPU + 4 hop) | 8.6 ms = 116.6 tok/s (`RESULTS.md:61`) | after the ~1 ms fixed cost, the hop is ~6 ms on the virtual switch, not 4 |

**Flags.** `HR<3%` = estimate within 3% of a device's budget. `NEEDS_FA` = ctx >= 16k and a GPU's headroom is smaller
than llama.cpp's no-flash-attention KQ buffer (n_head x ctx x 512 x 4 B). `CPU>30%`, `GPU_DISK` and `MAC_HEAD`
(never fired) are self-explanatory. `FLIPS@18ms/GB` = the overflow winner changes at the measured CPU cost.
`CLOSE_CALL` = peers vs RAM within 10%. `THIN_PEER` = an auto plan sends a peer 2 or fewer layers (never fired).
Counts over 396 rows: HR<3% 226, NEEDS_FA 124, CPU>30% 56, GPU_DISK 32, CLOSE_CALL 28, FLIPS 9.

## 2. Predictions at a glance (auto)

**S1 G14 alone, auto, interactive** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 |
| 7B Q8_0 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | RAM cpu12 CUDA0 [37] 42 |
| 14B Q6_K | RAM cpu3 CUDA0 [46] 37 | RAM cpu11 CUDA0 [38] 55 | RAM cpu18 CUDA0 [31] 70 |
| 14B Q8_0 | RAM cpu14 CUDA0 [35] 80 | RAM cpu19 CUDA0 [30] 94 | RAM cpu23 CUDA0 [26] 106 |
| 32B Q4_K_M | RAM cpu29 CUDA0 [36] 135 | RAM cpu34 CUDA0 [31] 149 | **DISK** cpu39 CUDA0 [26] 163* |
| 32B Q6_K | **DISK** cpu38 CUDA0 [27] 219* | **DISK** cpu41 CUDA0 [24] 231* | **DISK** cpu44 CUDA0 [21] 243* |
| 72B Q4_K_M | **DISK** cpu62 CUDA0 [19] 435* | **DISK** cpu63 CUDA0 [18] 440* | **DISK** cpu65 CUDA0 [16] 450* |

**S2 G14 + M4 Air 10 GiB, auto, interactive** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 |
| 7B Q8_0 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | RAM cpu12 CUDA0 [37] 42 |
| 14B Q6_K | RAM cpu3 CUDA0 [46] 37 | gpu RPC0,CUDA0 [11,38] 55 | gpu RPC0,CUDA0 [18,31] 68 |
| 14B Q8_0 | gpu RPC0,CUDA0 [14,35] 77 | gpu RPC0,CUDA0 [19,30] 89 | gpu RPC0,CUDA0 [23,26] 98 |
| 32B Q4_K_M | gpu RPC0,CUDA0 [29,36] 125 | RAM cpu34 CUDA0 [31] 149 | RAM cpu15 RPC0,CUDA0 [24,26] 156 |
| 32B Q6_K | RAM cpu14 RPC0,CUDA0 [24,27] 207 | RAM cpu20 RPC0,CUDA0 [21,24] 221 | RAM cpu25 RPC0,CUDA0 [19,21] 234 |
| 72B Q4_K_M | **DISK** cpu43 RPC0,CUDA0 [19,19] 423* | **DISK** cpu46 RPC0,CUDA0 [17,18] 429* | **DISK** cpu50 RPC0,CUDA0 [15,16] 441* |

**S2 G14 + M4 Air 10 GiB, auto, throughput** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 |
| 7B Q8_0 | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | RAM cpu12 CUDA0 [37] 42 |
| 14B Q6_K | RAM cpu3 CUDA0 [46] 37 | RAM cpu11 CUDA0 [38] 55 | RAM cpu18 CUDA0 [31] 70 |
| 14B Q8_0 | RAM cpu14 CUDA0 [35] 80 | RAM cpu19 CUDA0 [30] 94 | gpu RPC0,CUDA0 [23,26] 98 |
| 32B Q4_K_M | gpu RPC0,CUDA0 [31,34] 130 | RAM cpu34 CUDA0 [31] 149 | RAM cpu15 RPC0,CUDA0 [24,26] 156 |
| 32B Q6_K | RAM cpu14 RPC0,CUDA0 [24,27] 207 | RAM cpu20 RPC0,CUDA0 [21,24] 221 | RAM cpu25 RPC0,CUDA0 [19,21] 234 |
| 72B Q4_K_M | **DISK** cpu43 RPC0,CUDA0 [19,19] 423* | **DISK** cpu46 RPC0,CUDA0 [17,18] 429* | **DISK** cpu50 RPC0,CUDA0 [15,16] 441* |

**S3 G14 + M4 Air 6 GiB, auto, interactive** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 |
| 7B Q8_0 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | RAM cpu12 CUDA0 [37] 42 |
| 14B Q6_K | RAM cpu3 CUDA0 [46] 37 | gpu RPC0,CUDA0 [11,38] 55 | RAM cpu18 CUDA0 [31] 70 |
| 14B Q8_0 | gpu RPC0,CUDA0 [14,35] 77 | RAM cpu19 CUDA0 [30] 94 | RAM cpu23 CUDA0 [26] 106 |
| 32B Q4_K_M | RAM cpu29 CUDA0 [36] 135 | RAM cpu34 CUDA0 [31] 149 | RAM cpu25 RPC0,CUDA0 [14,26] 161 |
| 32B Q6_K | RAM cpu24 RPC0,CUDA0 [14,27] 214 | **DISK** cpu29 RPC0,CUDA0 [12,24] 227* | **DISK** cpu33 RPC0,CUDA0 [11,21] 240* |
| 72B Q4_K_M | **DISK** cpu52 RPC0,CUDA0 [10,19] 430* | **DISK** cpu53 RPC0,CUDA0 [10,18] 435* | **DISK** cpu56 RPC0,CUDA0 [9,16] 447* |

**S4 G14 + 2x M4 Air, auto, interactive** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 |
| 7B Q8_0 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | RAM cpu12 CUDA0 [37] 42 |
| 14B Q6_K | RAM cpu3 CUDA0 [46] 37 | gpu RPC0,CUDA0 [11,38] 55 | gpu RPC0,CUDA0 [18,31] 68 |
| 14B Q8_0 | gpu RPC0,CUDA0 [14,35] 77 | gpu RPC0,CUDA0 [19,30] 89 | gpu RPC0,CUDA0 [23,26] 98 |
| 32B Q4_K_M | gpu RPC0,CUDA0 [29,36] 125 | gpu RPC0,RPC1,CUDA0 [6,28,31] 141 | gpu RPC0,RPC1,CUDA0 [16,23,26] 153 |
| 32B Q6_K | gpu RPC0,RPC1,CUDA0 [14,24,27] 202 | gpu RPC0,RPC1,CUDA0 [20,21,24] 212 | RAM cpu6 RPC0,RPC1,CUDA0 [19,19,21] 226 |
| 72B Q4_K_M | **DISK** cpu25 RPC0,RPC1,CUDA0 [19,18,19] 411* | **DISK** cpu29 RPC0,RPC1,CUDA0 [17,17,18] 419* | **DISK** cpu35 RPC0,RPC1,CUDA0 [15,15,16] 433* |

**S5 G14 + RTX 5090 31 GiB, auto, interactive** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 |
| 7B Q8_0 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | gpu RPC0,CUDA0 [12,37] 27 |
| 14B Q6_K | gpu RPC0,CUDA0 [3,46] 34 | gpu RPC0,CUDA0 [11,38] 34 | gpu RPC0,CUDA0 [18,31] 34 |
| 14B Q8_0 | gpu RPC0,CUDA0 [14,35] 43 | gpu RPC0,CUDA0 [19,30] 43 | gpu RPC0,CUDA0 [23,26] 43 |
| 32B Q4_K_M | gpu RPC0,CUDA0 [29,36] 55 | gpu RPC0,CUDA0 [34,31] 55 | gpu RPC0,CUDA0 [39,26] 55 |
| 32B Q6_K | gpu RPC0,CUDA0 [38,27] 73 | gpu RPC0,CUDA0 [41,24] 73 | gpu RPC0,CUDA0 [44,21] 73 |
| 72B Q4_K_M | RAM cpu2 RPC0,CUDA0 [60,19] 129 | RAM cpu8 RPC0,CUDA0 [55,18] 163 | RAM cpu16 RPC0,CUDA0 [49,16] 204 |

**S5 G14 + RTX 5090 31 GiB, auto, throughput** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 |
| 7B Q8_0 | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 | **vllm** 7b-instruct-awq 10.63/10.66 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | RAM cpu1 CUDA0 [48] 24 | gpu RPC0,CUDA0 [36,13] 27 |
| 14B Q6_K | gpu RPC0,CUDA0 [36,13] 34 | gpu RPC0,CUDA0 [36,13] 34 | gpu RPC0,CUDA0 [36,13] 34 |
| 14B Q8_0 | gpu RPC0,CUDA0 [37,12] 43 | gpu RPC0,CUDA0 [37,12] 43 | gpu RPC0,CUDA0 [37,12] 43 |
| 32B Q4_K_M | gpu RPC0,CUDA0 [48,17] 55 | gpu RPC0,CUDA0 [48,17] 55 | gpu RPC0,CUDA0 [49,16] 55 |
| 32B Q6_K | gpu RPC0,CUDA0 [48,17] 73 | gpu RPC0,CUDA0 [48,17] 73 | gpu RPC0,CUDA0 [48,17] 73 |
| 72B Q4_K_M | RAM cpu2 RPC0,CUDA0 [60,19] 129 | RAM cpu8 RPC0,CUDA0 [55,18] 163 | RAM cpu16 RPC0,CUDA0 [49,16] 204 |

**S6 GX10 110 GiB, auto, interactive** (cell: tier, CPU layers, --device, -ts, predicted decode ms; * = disk-bound, see section 5)

| model | ctx 4096 | ctx 16384 | ctx 32768 |
|---|---|---|---|
| 0.5B Q4_K_M | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 | gpu CUDA0 [25] 1 |
| 1.5B Q4_K_M | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 | gpu CUDA0 [29] 3 |
| 3B Q4_K_M | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 | gpu CUDA0 [37] 5 |
| 7B Q4_K_M | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 | gpu CUDA0 [29] 12 |
| 7B Q8_0 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 | gpu CUDA0 [29] 20 |
| 14B Q4_K_M | gpu CUDA0 [49] 23 | gpu CUDA0 [49] 23 | gpu CUDA0 [49] 23 |
| 14B Q6_K | gpu CUDA0 [49] 30 | gpu CUDA0 [49] 30 | gpu CUDA0 [49] 30 |
| 14B Q8_0 | gpu CUDA0 [49] 39 | gpu CUDA0 [49] 39 | gpu CUDA0 [49] 39 |
| 32B Q4_K_M | gpu CUDA0 [65] 51 | gpu CUDA0 [65] 51 | gpu CUDA0 [65] 51 |
| 32B Q6_K | gpu CUDA0 [65] 69 | gpu CUDA0 [65] 69 | gpu CUDA0 [65] 69 |
| 72B Q4_K_M | gpu CUDA0 [81] 114 | gpu CUDA0 [81] 114 | gpu CUDA0 [81] 114 |

The full per-row tables are in Appendix B (auto) and Appendix C (run together). They give the per-device estimate and
budget, both overflow options with their predicted ms, every flag and the planner's own reasons and warnings.

Things to notice in the matrices:

- **The 0.5B-7B models never leave CUDA0 under auto** in any scenario. The largest is 7B Q8_0 @32k at 9.34 of 11.40
  GiB. The only way a peer joins is `all_selected`, which gives the Mac exactly 1 layer (Appendix C).
- **14B Q4_K_M is the edge of one GPU.** All-GPU at 4k (9.32 GiB), 1 CPU layer at 16k, 12 CPU layers at 32k: KV
  grows from 0.75 to 6 GiB over 48 layers.
- **The Mac's use is not monotonic in context.** On S2, 32B Q4_K_M uses the Mac at 4k (`-ts 29,36`), drops it at 16k
  (34 CPU layers: the Mac alone cannot take 34 layers and peers+RAM is never tried), and uses it again at 32k
  (CPU 15 + Mac 24, because neither alone fits and the ladder then combines them). That is symptom 4 from section 0.
- **`objective=throughput` on llama.cpp only ever makes plans slower or the same.** It splits in proportion to
  capacity (`planner.py:438`), but llama-server runs `-np 1` (`adapters/llama_rpc.py:128`), so nothing batches. On S2
  the proportional peer layout (`-ts 24,25` for 14B) is costed as a single stream and loses to RAM. Throughput
  therefore moves 14B Q6_K @16k/32k and 14B Q8_0 @4k/16k from the Mac to host RAM. On S5 it moves up to 33 more layers onto the 5090 (14B Q6_K `-ts 3,46` becomes `-ts 36,13`) at an identical prior.
- **S6 GX10:** every model is a single-device `gpu` plan. 72B @32k uses 50.99 GiB, which fits both the brief's 110
  GiB and the 83.2 GiB a default contribution would give (0.65 x 128 GB, `agent/contribution.py:43`; reserve 25.6
  GiB, `contribution.py:21-24`).

## 3. Engine selector predictions

| case | ctx | engine | model served | fit | tier | device est/budget GiB | key numbers | reasons / warnings (short) |
|---|---|---|---|---|---|---|---|---|
| 7B Q4_K_M throughput -> vLLM AWQ variant (S1) | 4096 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 6.91 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu; vLLM 0.32.0 on CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU): needs ~6.9 GiB (5.2 GiB weights + 224 MiB KV for |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S2) | 4096 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 6.91 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu; vLLM 0.32.0 on CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU): needs ~6.9 GiB (5.2 GiB weights + 224 MiB KV for |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S5) | 4096 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 6.91 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu; vLLM 0.32.0 on CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU): needs ~6.9 GiB (5.2 GiB weights + 224 MiB KV for |
| qwen2.5-7b-instruct-awq forced engine=vllm, interactive (S1) | 4096 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 6.91 GiB, max-num-seqs 8, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; vLLM 0.32.0 on CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU): needs ~6.9 GiB (5.2 GiB weights + 224 MiB KV for |
| qwen2.5-32b-instruct-hf requested directly (AirLLM is its only engine) | 4096 | airllm | qwen2.5-32b-instruct-hf | N does_not_fit | disk | CUDA0 0L 8.54/11.16 | 61.0 GiB read/token, [13.1, 312.0] s/token, host RAM 20.3 GiB | capacity mode AirLLM reads about 61.0 GiB from disk for every token, so expect roughly 13.1 s to 5 min per tok; the first start splits the checkpoint into per-layer files under C:\Users\operator\.hypercompute\airllm-shards\qw; per-layer memory is a rough guess until the model (its config.json) is downloaded |
| 32B Q4_K_M, allow_ram_tier=false -> llama does_not_fit -> AirLLM 32B-hf fallback attempt | 4096 | llama_rpc | qwen2.5-32b-instruct-q4_k_m | N does_not_fit | gpu | - | - | AirLLM could serve qwen2.5-32b-instruct-hf in capacity mode, but: not enough disk for the per-layer split: nee |
| 72B Q4_K_M, allow_ram_tier=false -> does_not_fit; no HF variant so no AirLLM fallback | 4096 | llama_rpc | qwen2.5-72b-instruct-q4_k_m | N does_not_fit | gpu | - | - |  |
| 72B Q4_K_M, allow_capacity_mode=false (no gpu_disk) | 4096 | llama_rpc | qwen2.5-72b-instruct-q4_k_m | N does_not_fit | gpu | - | - |  |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S1) | 16384 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 7.56 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16); vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S2) | 16384 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 7.56 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16); vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S5) | 16384 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 7.56 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16); vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu |
| qwen2.5-7b-instruct-awq forced engine=vllm, interactive (S1) | 16384 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 7.56 GiB, max-num-seqs 8, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 8); vLLM 0.32.0 on CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU): needs ~7.6 GiB (5.2 GiB weights + 896 MiB KV for |
| qwen2.5-32b-instruct-hf requested directly (AirLLM is its only engine) | 16384 | airllm | qwen2.5-32b-instruct-hf | N does_not_fit | disk | CUDA0 0L 8.54/11.16 | 61.0 GiB read/token, [13.1, 312.0] s/token, host RAM 20.3 GiB | capacity mode AirLLM reads about 61.0 GiB from disk for every token, so expect roughly 13.1 s to 5 min per tok; the first start splits the checkpoint into per-layer files under C:\Users\operator\.hypercompute\airllm-shards\qw; per-layer memory is a rough guess until the model (its config.json) is downloaded |
| 32B Q4_K_M, allow_ram_tier=false -> llama does_not_fit -> AirLLM 32B-hf fallback attempt | 16384 | llama_rpc | qwen2.5-32b-instruct-q4_k_m | N does_not_fit | gpu | - | - | AirLLM could serve qwen2.5-32b-instruct-hf in capacity mode, but: not enough disk for the per-layer split: nee |
| 72B Q4_K_M, allow_ram_tier=false -> does_not_fit; no HF variant so no AirLLM fallback | 16384 | llama_rpc | qwen2.5-72b-instruct-q4_k_m | N does_not_fit | gpu | - | - |  |
| 72B Q4_K_M, allow_capacity_mode=false (no gpu_disk) | 16384 | llama_rpc | qwen2.5-72b-instruct-q4_k_m | N does_not_fit | gpu | - | - |  |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S1) | 32768 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 8.44 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16); vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S2) | 32768 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 8.44 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16); vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu |
| 7B Q4_K_M throughput -> vLLM AWQ variant (S5) | 32768 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 8.44 GiB, max-num-seqs 16, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16); vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for objective=throu |
| qwen2.5-7b-instruct-awq forced engine=vllm, interactive (S1) | 32768 | vllm | qwen2.5-7b-instruct-awq | Y estimated_fit | gpu | CUDA0 28L 10.63/10.66 | util 0.89, reserved 10.63 GiB, need 8.44 GiB, max-num-seqs 8, KV room 3.94 GiB | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded ; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 8); vLLM 0.32.0 on CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU): needs ~8.4 GiB (5.2 GiB weights + 1.8 GiB KV for |
| qwen2.5-32b-instruct-hf requested directly (AirLLM is its only engine) | 32768 | airllm | qwen2.5-32b-instruct-hf | N does_not_fit | disk | CUDA0 0L 8.54/11.16 | 61.0 GiB read/token, [13.1, 312.0] s/token, host RAM 20.3 GiB | capacity mode AirLLM reads about 61.0 GiB from disk for every token, so expect roughly 13.1 s to 5 min per tok; the first start splits the checkpoint into per-layer files under C:\Users\operator\.hypercompute\airllm-shards\qw; per-layer memory is a rough guess until the model (its config.json) is downloaded |
| 32B Q4_K_M, allow_ram_tier=false -> llama does_not_fit -> AirLLM 32B-hf fallback attempt | 32768 | llama_rpc | qwen2.5-32b-instruct-q4_k_m | N does_not_fit | gpu | - | - | AirLLM could serve qwen2.5-32b-instruct-hf in capacity mode, but: not enough disk for the per-layer split: nee |
| 72B Q4_K_M, allow_ram_tier=false -> does_not_fit; no HF variant so no AirLLM fallback | 32768 | llama_rpc | qwen2.5-72b-instruct-q4_k_m | N does_not_fit | gpu | - | - |  |
| 72B Q4_K_M, allow_capacity_mode=false (no gpu_disk) | 32768 | llama_rpc | qwen2.5-72b-instruct-q4_k_m | N does_not_fit | gpu | - | - |  |

Notes:
- `throughput` + 7B Q4_K_M **or 7B Q8_0** -> vLLM serves `qwen2.5-7b-instruct-awq`, identically in S1, S2 and S5. In S2/S5 the peers are ignored with a warning (`adapters/vllm.py:291`). The vLLM plan's `total_bytes_est` is the whole reservation (10.63 of
  10.66 GiB budget), which is why it shows 0.3% headroom; that is by design (`adapters/vllm.py:337`). The real
  risk is the startup free-memory check. The recorded run started with 10.77 GiB free against 10.63 GiB reserved,
  a 140 MiB margin (`RESULTS.md:82`).
- The vLLM KV room (3.94 GiB) / 57,344 B per token = ~73k tokens in total, which is 2 sequences at 32k and 4 at 16k,
  against `--max-num-seqs 16` (8 interactive).
- AirLLM 32B bf16: the footprint is a guess (config.json not downloaded, so `fit_unknown` becomes `does_not_fit`
  because of disk). The plan says 61.0 GiB read per token, 13 s to 5 min per token, and 20.3 GiB of host RAM.
- `allow_ram_tier=false` also disables `gpu_disk`, because `allow_disk = allow_ram and allow_capacity_mode`
  (`planner.py:440`). With it, 32B and 72B are simply `does_not_fit` on S1, and for 72B there is no AirLLM rescue.
- EXO: `_exo()` needs every selected peer to be a Mac with EXO running **and** an MLX variant, and it only wins when
  llama.cpp is capacity mode or `does_not_fit` (`selector.py:107-111`). The manifest has MLX only for 7B and Qwen3
  0.6B, which always fit CUDA0, so EXO is never auto-selected in this grid. A forced `--engine exo` run on two Airs
  could not be predicted: the EXO planner needs live EXO state.

## 4. Budget reality check (what the gateway would actually compute)

- M4 Air 16 GB, Metal working set 10.67 GiB, vm.available 12 GiB -> remote_budgets() = **6.67 GiB** (free memory minus headroom)
- M4 Air 16 GB, Metal working set 10.67 GiB, vm.available 10 GiB -> remote_budgets() = **6.00 GiB** (free memory minus headroom)
- M4 Air 16 GB, Metal working set 10.67 GiB, vm.available 8 GiB -> remote_budgets() = **4.00 GiB** (free memory minus headroom)
- G14 at the measured idle free VRAM (11.66 GiB): local_budgets() CUDA0 = **11.16 GiB** (free memory now minus 512 MiB headroom); cpu_budget() at 20 GiB free = **13.28 GiB** (free RAM now minus 6372 MiB reserve and 512 MiB safety)

| scenario | model | ctx | brief budgets: fit, tier, --device, -ts, cpuL, ms | gateway budgets: same | flags (gateway) |
|---|---|---|---|---|---|
| S1r | 14B Q4_K_M | 16384 | Y gpu_ram CUDA0 -ts 48 cpu 1 24.3 ms | Y gpu_ram CUDA0 -ts 47 cpu 2 26.0 ms | HR<3%:CUDA0(0.4%) NEEDS_FA |
| S1r | 14B Q4_K_M | 32768 | Y gpu_ram CUDA0 -ts 37 cpu 12 42.4 ms | Y gpu_ram CUDA0 -ts 36 cpu 13 44.0 ms | HR<3%:CUDA0(2.2%) NEEDS_FA |
| S2r | 14B Q4_K_M | 16384 | Y gpu_ram CUDA0 -ts 48 cpu 1 24.3 ms | Y gpu_ram CUDA0 -ts 47 cpu 2 26.0 ms | HR<3%:CUDA0(0.4%) NEEDS_FA |
| S2r | 14B Q4_K_M | 32768 | Y gpu_ram CUDA0 -ts 37 cpu 12 42.4 ms | Y gpu_ram CUDA0 -ts 36 cpu 13 44.0 ms | HR<3%:CUDA0(2.2%) FLIPS@18ms/GB CLOSE_CALL(<10%) NEEDS_FA |
| S1r | 14B Q6_K | 4096 | Y gpu_ram CUDA0 -ts 46 cpu 3 37.0 ms | Y gpu_ram CUDA0 -ts 45 cpu 4 39.2 ms | HR<3%:CUDA0(0.1%) |
| S1r | 14B Q6_K | 16384 | Y gpu_ram CUDA0 -ts 38 cpu 11 54.8 ms | Y gpu_ram CUDA0 -ts 37 cpu 12 57.0 ms | HR<3%:CUDA0(1.2%) NEEDS_FA |
| S1r | 14B Q6_K | 32768 | Y gpu_ram CUDA0 -ts 31 cpu 18 70.4 ms | Y gpu_ram CUDA0 -ts 30 cpu 19 72.6 ms | HR<3%:CUDA0(2.1%) CPU>30%(19/48) NEEDS_FA |
| S2r | 14B Q6_K | 4096 | Y gpu_ram CUDA0 -ts 46 cpu 3 37.0 ms | Y gpu_ram CUDA0 -ts 45 cpu 4 39.2 ms | HR<3%:CUDA0(0.1%) FLIPS@18ms/GB CLOSE_CALL(<10%) |
| S2r | 14B Q6_K | 16384 | Y gpu RPC0,CUDA0 -ts 11,38 cpu 0 54.7 ms | Y gpu RPC0,CUDA0 -ts 12,37 cpu 0 56.6 ms | HR<3%:CUDA0(1.2%) CLOSE_CALL(<10%) NEEDS_FA |
| S2r | 14B Q6_K | 32768 | Y gpu RPC0,CUDA0 -ts 18,31 cpu 0 67.7 ms | Y gpu_ram CUDA0 -ts 30 cpu 19 72.6 ms | HR<3%:CUDA0(2.1%) CPU>30%(19/48) NEEDS_FA |
| S1r | 14B Q8_0 | 4096 | Y gpu_ram CUDA0 -ts 35 cpu 14 79.6 ms | Y gpu_ram CUDA0 -ts 34 cpu 15 82.5 ms | HR<3%:CUDA0(2.5%) CPU>30%(15/48) |
| S1r | 14B Q8_0 | 32768 | Y gpu_ram CUDA0 -ts 26 cpu 23 105.6 ms | Y gpu_ram CUDA0 -ts 25 cpu 24 108.5 ms | HR<3%:CUDA0(2.2%) CPU>30%(24/48) NEEDS_FA |
| S2r | 14B Q8_0 | 4096 | Y gpu RPC0,CUDA0 -ts 14,35 cpu 0 76.9 ms | Y gpu RPC0,CUDA0 -ts 15,34 cpu 0 79.3 ms | HR<3%:CUDA0(2.5%) CLOSE_CALL(<10%) |
| S2r | 14B Q8_0 | 16384 | Y gpu RPC0,CUDA0 -ts 19,30 cpu 0 88.9 ms | Y gpu_ram CUDA0 -ts 30 cpu 19 94.1 ms | HR<3%:CUDA0(0.6%) CPU>30%(19/48) NEEDS_FA |
| S2r | 14B Q8_0 | 32768 | Y gpu RPC0,CUDA0 -ts 23,26 cpu 0 98.5 ms | Y gpu_ram CUDA0 -ts 25 cpu 24 108.5 ms | HR<3%:CUDA0(2.2%) CPU>30%(24/48) NEEDS_FA |
| S1r | 32B Q4_K_M | 4096 | Y gpu_ram CUDA0 -ts 36 cpu 29 135.3 ms | Y gpu_ram CUDA0 -ts 35 cpu 30 138.0 ms | HR<3%:CUDA0(1.3%) CPU>30%(30/64) |
| S1r | 32B Q4_K_M | 16384 | Y gpu_ram CUDA0 -ts 31 cpu 34 149.2 ms | Y gpu_ram CUDA0 -ts 30 cpu 35 152.3 ms | HR<3%:CUDA0(1.9%) CPU>30%(35/64) NEEDS_FA |
| S1r | 32B Q4_K_M | 32768 | Y gpu_disk CUDA0 -ts 26 cpu 39 163.4 ms | Y gpu_disk CUDA0 -ts 25 cpu 40 166.2 ms | HR<3%:CUDA0(3.0%) CPU>30%(40/64) GPU_DISK NEEDS_FA |
| S2r | 32B Q4_K_M | 4096 | Y gpu RPC0,CUDA0 -ts 29,36 cpu 0 125.2 ms | Y gpu_ram CUDA0 -ts 35 cpu 30 138.0 ms | HR<3%:CUDA0(1.3%) CPU>30%(30/64) |
| S2r | 32B Q4_K_M | 16384 | Y gpu_ram CUDA0 -ts 31 cpu 34 149.2 ms | Y gpu_ram CUDA0 -ts 30 cpu 35 152.3 ms | HR<3%:CUDA0(1.9%) CPU>30%(35/64) NEEDS_FA |
| S2r | 32B Q4_K_M | 32768 | Y gpu_ram RPC0,CUDA0 -ts 24,26 cpu 15 156.1 ms | Y gpu_ram RPC0,CUDA0 -ts 14,25 cpu 26 163.6 ms | HR<3%:Mac(0.8%) HR<3%:CUDA0(3.0%) CPU>30%(26/64) NEEDS_FA |
| S1r | 32B Q6_K | 4096 | Y gpu_disk CUDA0 -ts 27 cpu 38 219.2 ms | Y gpu_disk CUDA0 -ts 26 cpu 39 223.1 ms | HR<3%:CUDA0(2.3%) CPU>30%(39/64) GPU_DISK |
| S1r | 32B Q6_K | 16384 | Y gpu_disk CUDA0 -ts 24 cpu 41 231.0 ms | Y gpu_disk CUDA0 -ts 23 cpu 42 234.9 ms | CPU>30%(42/64) GPU_DISK NEEDS_FA |
| S2r | 32B Q6_K | 4096 | Y gpu_ram RPC0,CUDA0 -ts 24,27 cpu 14 207.3 ms | Y gpu_ram RPC0,CUDA0 -ts 14,26 cpu 25 217.9 ms | HR<3%:Mac(1.3%) HR<3%:CUDA0(2.3%) CPU>30%(25/64) |
| S2r | 32B Q6_K | 16384 | Y gpu_ram RPC0,CUDA0 -ts 21,24 cpu 20 221.1 ms | Y gpu_disk RPC0,CUDA0 -ts 12,23 cpu 30 231.0 ms | CPU>30%(30/64) GPU_DISK NEEDS_FA |
| S2r | 32B Q6_K | 32768 | Y gpu_ram RPC0,CUDA0 -ts 19,21 cpu 25 234.3 ms | Y gpu_disk RPC0,CUDA0 -ts 11,21 cpu 33 239.6 ms | HR<3%:Mac(0.7%) HR<3%:CUDA0(0.1%) CPU>30%(33/64) GPU_DISK NEEDS_FA |
| S1r | 72B Q4_K_M | 16384 | Y gpu_disk CUDA0 -ts 18 cpu 63 439.8 ms | Y gpu_disk CUDA0 -ts 17 cpu 64 445.3 ms | CPU>30%(64/80) GPU_DISK NEEDS_FA |
| S2r | 72B Q4_K_M | 4096 | Y gpu_disk RPC0,CUDA0 -ts 19,19 cpu 43 422.7 ms | Y gpu_disk RPC0,CUDA0 -ts 10,19 cpu 52 430.4 ms | HR<3%:CUDA0(1.6%) CPU>30%(52/80) GPU_DISK |
| S2r | 72B Q4_K_M | 16384 | Y gpu_disk RPC0,CUDA0 -ts 17,18 cpu 46 429.3 ms | Y gpu_disk RPC0,CUDA0 -ts 9,17 cpu 55 441.6 ms | CPU>30%(55/80) GPU_DISK NEEDS_FA |
| S2r | 72B Q4_K_M | 32768 | Y gpu_disk RPC0,CUDA0 -ts 15,16 cpu 50 441.4 ms | Y gpu_disk RPC0,CUDA0 -ts 9,16 cpu 56 446.5 ms | HR<3%:Mac(0.0%) HR<3%:CUDA0(0.3%) CPU>30%(56/80) GPU_DISK NEEDS_FA |

Reading: the brief's 11.40 GiB CUDA0 is the contribution cap (11.44 GiB = 12,284,067,840 B). At runtime the budget
is free VRAM - 512 MiB, which was 11.16 GiB on an idle G14 (`deployments.py:104-120`). Thirteen S1 rows shift by
one CPU layer, all toward the measured 14B result. With the Mac at its real 6 GiB, S2 behaves like S3. Expect test
day to look like the "gateway" column, not the brief's.

## 5. Where `gpu_disk` is chosen, and what will really happen

| scenario | model | ctx | CPU layers paged | paged weights GiB | CPU-side resident GiB | page cache room GiB | verdict | expected decode |
|---|---|---|---|---|---|---|---|---|
| S1 | 32B Q4_K_M | 32768 | 39/64 | 10.6 | 5.11 | 13.5 | fits the page cache: RAM speed after the first pass | 163.4-226.0 ms (not seconds; the capacity-mode warning overstates it) |
| S1 | 32B Q6_K | 4096 | 38/64 | 14.2 | 0.83 | 17.6 | fits the page cache: RAM speed after the first pass | 219.2-302.8 ms (not seconds; the capacity-mode warning overstates it) |
| S1 | 32B Q6_K | 16384 | 41/64 | 15.3 | 2.80 | 15.6 | fits the page cache: RAM speed after the first pass | 231.0-321.2 ms (not seconds; the capacity-mode warning overstates it) |
| S1 | 32B Q6_K | 32768 | 44/64 | 16.4 | 5.73 | 12.7 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 5.1-11.8 s/token (NVMe 3.5-1.5 GB/s effective) |
| S3 | 32B Q6_K | 16384 | 29/64 | 10.8 | 2.05 | 16.4 | fits the page cache: RAM speed after the first pass | 227.1-290.9 ms (not seconds; the capacity-mode warning overstates it) |
| S3 | 32B Q6_K | 32768 | 33/64 | 12.3 | 4.36 | 14.0 | fits the page cache: RAM speed after the first pass | 239.6-312.2 ms (not seconds; the capacity-mode warning overstates it) |
| S1 | 72B Q4_K_M | 4096 | 62/80 | 30.3 | 1.25 | 17.1 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 9.3-21.7 s/token (NVMe 3.5-1.5 GB/s effective) |
| S1 | 72B Q4_K_M | 16384 | 63/80 | 30.7 | 4.22 | 14.1 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 9.5-22.0 s/token (NVMe 3.5-1.5 GB/s effective) |
| S1 | 72B Q4_K_M | 32768 | 65/80 | 31.7 | 8.41 | 9.9 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 9.8-22.7 s/token (NVMe 3.5-1.5 GB/s effective) |
| S2 | 72B Q4_K_M | 4096 | 43/80 | 21.1 | 0.95 | 17.4 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 6.6-15.3 s/token (NVMe 3.5-1.5 GB/s effective) |
| S2 | 72B Q4_K_M | 16384 | 46/80 | 22.6 | 3.16 | 15.2 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 7.1-16.3 s/token (NVMe 3.5-1.5 GB/s effective) |
| S2 | 72B Q4_K_M | 32768 | 50/80 | 24.5 | 6.53 | 11.8 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 7.6-17.6 s/token (NVMe 3.5-1.5 GB/s effective) |
| S3 | 72B Q4_K_M | 4096 | 52/80 | 25.5 | 1.09 | 17.3 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 7.9-18.3 s/token (NVMe 3.5-1.5 GB/s effective) |
| S3 | 72B Q4_K_M | 16384 | 53/80 | 25.9 | 3.59 | 14.8 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 8.0-18.7 s/token (NVMe 3.5-1.5 GB/s effective) |
| S3 | 72B Q4_K_M | 32768 | 56/80 | 27.4 | 7.28 | 11.1 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 8.5-19.7 s/token (NVMe 3.5-1.5 GB/s effective) |
| S4 | 72B Q4_K_M | 4096 | 25/80 | 12.5 | 0.67 | 17.7 | fits the page cache: RAM speed after the first pass | 411.3-484.9 ms (not seconds; the capacity-mode warning overstates it) |
| S4 | 72B Q4_K_M | 16384 | 29/80 | 14.4 | 2.09 | 16.3 | fits the page cache: RAM speed after the first pass | 418.8-503.6 ms (not seconds; the capacity-mode warning overstates it) |
| S4 | 72B Q4_K_M | 32768 | 35/80 | 17.3 | 4.66 | 13.7 | cyclic scan larger than the page cache: LRU re-reads ~all of it every token | 5.5-12.6 s/token (NVMe 3.5-1.5 GB/s effective) |

"page cache room" = 20 GiB free - CPU-side resident estimate - token_embd - 1 GiB. It includes the owner's 6.2 GiB
reserve, which the page cache uses until the owner needs it. The "fits" verdicts therefore hold only on an otherwise
idle G14.

## 6. Where `auto` leaves a peer idle and the measured CPU cost says it should not

| scenario | model | ctx | auto plan | auto ms prior (@18) | run-together plan | run-together ms prior (@18) | auto slower by (@18) |
|---|---|---|---|---|---|---|---|
| S2 | 14B Q4_K_M | 32768 | CUDA0 -ts 37 cpu 12 | 42.4 (53.5) | RPC0,CUDA0 -ts 12,37 cpu 0 | 43.1 (43.1) | 24% |
| S3 | 14B Q4_K_M | 32768 | CUDA0 -ts 37 cpu 12 | 42.4 (53.5) | RPC0,CUDA0 -ts 12,37 cpu 0 | 43.1 (43.1) | 24% |
| S4 | 14B Q4_K_M | 32768 | CUDA0 -ts 37 cpu 12 | 42.4 (53.5) | RPC0,RPC1,CUDA0 -ts 1,11,37 cpu 0 | 47.1 (47.1) | 14% |
| S2 | 14B Q6_K | 4096 | CUDA0 -ts 46 cpu 3 | 37.0 (40.7) | RPC0,CUDA0 -ts 3,46 cpu 0 | 39.9 (39.9) | 2% |
| S3 | 14B Q6_K | 4096 | CUDA0 -ts 46 cpu 3 | 37.0 (40.7) | RPC0,CUDA0 -ts 3,46 cpu 0 | 39.9 (39.9) | 2% |
| S3 | 14B Q6_K | 32768 | CUDA0 -ts 31 cpu 18 | 70.4 (92.8) | RPC0,CUDA0 -ts 16,31 cpu 2 | 68.4 (70.9) | 31% |
| S3 | 14B Q8_0 | 16384 | CUDA0 -ts 30 cpu 19 | 94.1 (124.6) | RPC0,CUDA0 -ts 16,30 cpu 3 | 90.3 (95.2) | 31% |
| S3 | 14B Q8_0 | 32768 | CUDA0 -ts 26 cpu 23 | 105.6 (142.6) | RPC0,CUDA0 -ts 13,26 cpu 10 | 103.3 (119.4) | 19% |
| S2 | 32B Q4_K_M | 16384 | CUDA0 -ts 31 cpu 34 | 149.2 (203.8) | RPC0,CUDA0 -ts 28,31 cpu 6 | 139.9 (150.2) | 36% |
| S3 | 32B Q4_K_M | 4096 | CUDA0 -ts 36 cpu 29 | 135.3 (182.2) | RPC0,CUDA0 -ts 19,36 cpu 10 | 130.2 (147.0) | 24% |
| S3 | 32B Q4_K_M | 16384 | CUDA0 -ts 31 cpu 34 | 149.2 (203.8) | RPC0,CUDA0 -ts 16,31 cpu 18 | 145.6 (175.1) | 16% |

Fix direction: in `_overflow()` also evaluate the `ladder` "GPUs + peers + host RAM" layout when `via_ram` is
feasible, and compare all three by cost. Add the same candidate to the tuner. The S4 case with real 6 GiB Macs is
the worst: auto puts 35 CPU layers at 152.3 ms (208.6 @18), while run-together uses CPU 3 + 16 + 16 at 144.9 ms
(150.0 @18). Run-together wins even under the planner's own prior.

## 7. Bandwidth-blind cost model on faster/slower CUDA devices

| scenario | model | ctx | obj | --device -ts | prior ms (tok/s) | bandwidth-adjusted ms (tok/s) |
|---|---|---|---|---|---|---|
| S5 | 14B Q6_K | 4096 | inter | RPC0,CUDA0 -ts 3,46 | 34.3 (29.1) | 33.4 (29.9) |
| S5 | 14B Q6_K | 4096 | throu | RPC0,CUDA0 -ts 36,13 | 34.3 (29.1) | 23.6 (42.4) |
| S6 | 14B Q6_K | 4096 | inter | CUDA0 -ts 49 | 30.3 (33.0) | 54.5 (18.3) |
| S5 | 14B Q8_0 | 4096 | inter | RPC0,CUDA0 -ts 14,35 | 43.3 (23.1) | 37.8 (26.4) |
| S5 | 14B Q8_0 | 4096 | throu | RPC0,CUDA0 -ts 37,12 | 43.3 (23.1) | 29.0 (34.5) |
| S6 | 14B Q8_0 | 4096 | inter | CUDA0 -ts 49 | 39.3 (25.5) | 70.6 (14.2) |
| S5 | 32B Q4_K_M | 4096 | inter | RPC0,CUDA0 -ts 29,36 | 55.2 (18.1) | 44.0 (22.7) |
| S5 | 32B Q4_K_M | 4096 | throu | RPC0,CUDA0 -ts 48,17 | 55.2 (18.1) | 36.8 (27.2) |
| S6 | 32B Q4_K_M | 4096 | inter | CUDA0 -ts 65 | 51.2 (19.5) | 92.2 (10.8) |
| S5 | 32B Q6_K | 4096 | inter | RPC0,CUDA0 -ts 38,27 | 73.3 (13.6) | 53.2 (18.8) |
| S5 | 32B Q6_K | 4096 | throu | RPC0,CUDA0 -ts 48,17 | 73.3 (13.6) | 47.9 (20.9) |
| S6 | 32B Q6_K | 4096 | inter | CUDA0 -ts 65 | 69.3 (14.4) | 124.6 (8.0) |
| S5 | 72B Q4_K_M | 4096 | inter | RPC0,CUDA0 -ts 60,19 | 129.4 (7.7) | 87.9 (11.4) |
| S5 | 72B Q4_K_M | 4096 | throu | RPC0,CUDA0 -ts 60,19 | 129.4 (7.7) | 87.9 (11.4) |
| S6 | 72B Q4_K_M | 4096 | inter | CUDA0 -ts 81 | 114.3 (8.7) | 205.7 (4.9) |

The 5090 is costed at 1.32 ms/GB: 1,792 GB/s at the 5070 Ti Laptop's measured efficiency (2.64 ms/GB at 896 GB/s).
The GB10 is costed at 4.75 ms/GB: 273 GB/s at the M4 prior's ~77% efficiency. Both are assumptions, which the S5
and S6 runs replace with measurements.

## 8. Load time: weights that cross the network at startup

| scenario | model | ctx | peer weights GiB | load over Wi-Fi at 30 MB/s | at 1 GbE (110 MB/s) |
|---|---|---|---|---|---|
| S2 | 14B Q8_0 | 4096 | 3.8 | 2.3 min | 0.6 min |
| S3 | 14B Q8_0 | 4096 | 3.8 | 2.3 min | 0.6 min |
| S4 | 14B Q8_0 | 4096 | 3.8 | 2.3 min | 0.6 min |
| S5 | 14B Q8_0 | 4096 | 3.8 | 2.3 min | 0.6 min |
| S2 | 32B Q4_K_M | 4096 | 7.9 | 4.7 min | 1.3 min |
| S4 | 32B Q4_K_M | 4096 | 7.9 | 4.7 min | 1.3 min |
| S5 | 32B Q4_K_M | 4096 | 7.9 | 4.7 min | 1.3 min |
| S2 | 32B Q6_K | 4096 | 8.9 | 5.3 min | 1.5 min |
| S3 | 32B Q6_K | 4096 | 5.2 | 3.1 min | 0.8 min |
| S4 | 32B Q6_K | 4096 | 14.2 | 8.4 min | 2.3 min |
| S5 | 32B Q6_K | 4096 | 14.2 | 8.4 min | 2.3 min |
| S2 | 72B Q4_K_M | 4096 | 9.1 | 5.5 min | 1.5 min |
| S3 | 72B Q4_K_M | 4096 | 4.8 | 2.9 min | 0.8 min |
| S4 | 72B Q4_K_M | 4096 | 17.8 | 10.6 min | 2.9 min |
| S5 | 72B Q4_K_M | 4096 | 29.2 | 17.4 min | 4.8 min |

`LOAD_TIMEOUT_S = 30 min` (`config.py:25`). With `join --cache` (the recorded Mac had `cache_enabled: true`), later
loads read the tensors from the Mac's disk, so the first load is the one to time.

## 9. The 12 most informative real runs (ranked)

Each run lists the plan the gateway should produce with **its own budgets**: CUDA0 11.16 GiB, CPU 13.28 GiB, an M4
Air at 6.0 GiB (the default `join`), the 5090 at 30.7 GiB and the GX10 at 83.2 GiB. The brief's-budget plan is in
brackets when it differs. Numbers come from `runs12.py` (`runs12.json`, scratch). Test-day commands assume the G14 hosts on
`PORT` (8000 is the default) and the Mac runs `uv run hypercompute join --host <G14_IP>:$PORT --code <CODE> --cache --yes`.
A run is `uv run python scripts/e2e_check.py --port $PORT --model <id> --strategy <s> --ctx <n> --bench --label <l>`.
That writes `results/<ts>-<strategy>-<label>.json` with the plan, the observed placement and the benchmark. The check
on every run is plan vs observed: per-device layers from the load log (`placement`) and `participation_verified`.

**1. 0.5B Q4_K_M x S3 (real Mac, default join) x 4096, `all_selected`** (hop calibration + the whole Mac path)
- Expect `--device RPC0,CUDA0 -ts 1,24`: Mac 1 layer at 0.43/6.00 GiB, CUDA0 23 layers + head at 0.95/11.16. Prior
  5.1 ms (196 tok/s); measured local baseline 1.9 ms (526.7 tok/s).
- `hop = 1000/decode_tps - 1.9 ms` measures `HOP_MS` over Wi-Fi directly (1 layer = 10 MB of Metal work, ~0.1 ms).
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-0.5b-instruct-q4_k_m --strategy all_selected --ctx 4096 --bench --label s3-05b-hop`
- Could break:
  - The macOS arm64 engine has never been executed (`IMPLEMENTATION_STATUS.md:18`).
  - The Windows Public network profile blocks the Mac's join to the gateway (the `windows_public_network`
    diagnostic), or the macOS firewall blocks `ggml-rpc-server`.
  - Metal behind RPC.
  - The dashboard shows a Mac budget of ~6 GiB, not 10. That is expected (section 4), not a bug in the run.
- Break-even hops follow from this number: the 3-4 layer 14B Q6_K overflow (run 2) favours the Mac only below ~6.5 ms, while the 4+ GB overflows (14B Q8_0, 32B) favour it up to ~29 ms (14B Q8_0: 76.9 ms with a 4 ms hop vs 102.2 ms in RAM @18).

**2. 14B Q6_K x S3 x 4096, `auto`, then `hypercompute tune`** (the overflow decision itself)
- Expect gpu_ram, `--device CUDA0 -ngl 45 -ts 45`, 4 CPU layers (exactly the measured S1 plan), 39.2 ms
  [brief: `-ngl 46`, 3 CPU, 37.0 ms].
- The peers alternative is `RPC0,CUDA0 -ts 4,45` at 41.7 ms. At the measured CPU cost RAM = 44.2 ms, so **the Mac
  should win if the real hop is 6.5 ms or less**; the planner will still pick RAM. CUDA0 is at 11.15/11.16 GiB (0.1%).
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-14b-instruct-q6_k --strategy auto --ctx 4096 --bench --label s3-14bq6-auto`
- `uv run hypercompute tune qwen2.5-14b-instruct-q6_k --port $PORT --ctx 4096 --max-candidates 4`
- Could break:
  - The tuner's peer trial over Wi-Fi.
  - The saved profile is keyed with the Mac's node id (`selector.py:247-250`), so it only replays when the same Mac
    is joined.
  - The thin VRAM margin at `-ngl 45` (it held on 2026-10-21).

**3. 14B Q8_0 x S3 x 4096, `auto`** (the first real capacity split onto Metal; 14.62 GiB download)
- Expect gpu, `--device RPC0,CUDA0 -ts 15,34`: Mac 15 layers at 4.81/6.00 GiB, CUDA0 33L+H at 10.89/11.16. 79.3 ms
  (12.6 tok/s) [brief: `-ts 14,35`, 76.9 ms].
- The RAM alternative is `-ngl 34`, 15 CPU layers, at 82.5 ms (106.7 @18). About 4.1 GiB crosses Wi-Fi (~2.4 min at 30 MB/s).
- Measures the Metal ms/GB with a real share: `(1000/tps - CUDA part - hop) / 4.4 GB`.
- `uv run hypercompute models pull qwen2.5-14b-instruct-q8_0`, then
  `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-14b-instruct-q8_0 --strategy auto --ctx 4096 --bench --label s3-14bq8-auto --keep`.
- Then close the Mac lid mid-stream: expect `failed` with "no heartbeat" in ~16-17 s and the leases released (as in
  `RESULTS.md:84,86`).
- Could break:
  - Mac memory pressure: Metal 4.8 GiB plus macOS with only the 4 GiB reserve.
  - The real Metal cost is well above 10.85 ms/GB.
  - `--cache` disk on the Mac.

**4. 32B Q4_K_M x S3 x 4096, `auto` then `all_selected`** (the "peers + RAM never tried" gap; 18.49 GiB download)
- Auto: expect gpu_ram, `--device CUDA0 -ngl 35 -ts 35`, 30 CPU layers (47%), Mac idle, 138.0 ms (186.4 @18).
- Run-together: expect `--device RPC0,CUDA0 -ngl 54 -ts 19,35`, CPU 11, Mac 19 layers at 5.85/6.00, 133.0 ms
  (151.5 @18).
- Prediction: **run-together is ~20-25% faster in reality (186.4 vs 151.5 ms @18), and auto picks the slower plan.**
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-32b-instruct-q4_k_m --strategy auto --ctx 4096 --bench --label s3-32b-auto`,
  then the same with `--strategy all_selected --label s3-32b-together`.
- Could break:
  - ~8.2 GiB of mmap'd CPU weights compete with the 6.2 GiB reserve.
  - Pulling both 14B Q8_0 and 32B Q4_K_M overflows C: (33.1 GiB against 25.2 GiB free). Delete
    `mistral-small-3.2-24b-instruct-q4_k_m` (14 GiB) or point `HYPERCOMPUTE_HOME` at another drive first.

**5. 32B Q4_K_M x S1 x 32768, `auto`** (`gpu_disk` that should run at RAM speed + the 32k compute buffer)
- Expect gpu_disk/capacity_mode, `--device CUDA0 -ngl 25 -ts 25`, 40 CPU layers (10.9 GiB) paged, CUDA0 10.83/11.16
  [brief: `-ngl 26`, 39 CPU].
- The warning will say seconds per token. **Prediction: ~166-230 ms/token after the first pass**, because 10.9 GiB
  fits the ~13 GiB page cache.
- Also `NEEDS_FA`: CUDA0 has 0.33 GiB left, and the no-FA KQ buffer at 32k is 2.5 GiB.
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-32b-instruct-q4_k_m --strategy auto --ctx 32768 --bench --label s1-32b-32k`
- Could break:
  - CUDA OOM at `sched_reserve` if `-fa auto` does not engage.
  - A slow first token taking the warmup budget, which is max(120, 2 x 180) = 360 s for capacity mode
    (`deployments.py:55-59`).
  - A misleading UI banner.

**6. 14B Q6_K x S1 x 32768, `auto`** (KV/FA stress with an already-downloaded model)
- Expect gpu_ram, `--device CUDA0 -ngl 30 -ts 30`, 19 CPU layers (40%, `CPU>30%`), CUDA0 10.93/11.16 with 3.6 GiB
  of KV. 72.6 ms (96.2 @18) [brief: `-ngl 31`, 18 CPU].
- Compare the logged `CUDA0 compute buffer size` with the planner's flat 176 MiB.
- Add a ~30k-token prompt to time TTFT on a 40%-CPU split.
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-14b-instruct-q6_k --strategy auto --ctx 32768 --bench --label s1-14bq6-32k`
- Could break: compute buffer growth with ctx (`planner.py:67`), and the KV estimate if llama.cpp pads n_ctx.

**7. 7B Q4_K_M -> vLLM AWQ x S1 x 32768, `throughput`** (selector swap + vLLM KV limits)
- Expect engine vllm, `qwen2.5-7b-instruct-awq`, `--gpu-memory-utilization 0.89`, 10.63 GiB reserved, need 8.44
  GiB, `--max-model-len 32768 --max-num-seqs 16`.
- KV room 3.94 GiB, with the warning "about 2 full-length requests". Run 16 concurrent long prompts and expect queueing
  or preemption, not 16-way speed.
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-7b-instruct-q4_k_m --strategy auto --objective throughput --ctx 32768 --bench --label s1-vllm-32k`
- Could break: vLLM's startup free-memory check, which had a 140 MiB margin last time. A browser with GPU
  acceleration on the G14 is enough to fail it. A WSL restart.

**8. 32B Q4_K_M x S4 (two Airs, default join) x 16384, `auto` then `all_selected`** (second Mac; next hardware)
- Auto: expect RAM-only, `--device CUDA0 -ngl 30 -ts 30`, 35 CPU layers, both Macs idle, 152.3 ms (208.6 @18). The
  peers-only alternative does not fit: Mac 19 layers at 6.91/6.00.
- Run-together: expect `--device RPC0,RPC1,CUDA0 -ngl 62 -ts 16,16,30`, CPU 3, 144.9 ms (150.0 @18).
- With both Macs joined at `--reserve 0.7GiB` (10 GiB budgets), auto becomes `RPC0,RPC1,CUDA0 -ts 6,28,31` at
  140.8 ms.
- Exercises two endpoints, RPC numbering (`deployments.py:172-185`), three-device participation and two hops per
  token.
- Could break: the RPC name mapping when endpoints sort differently from the device order, and stale detection
  picking up one Mac.

**9. 32B Q4_K_M x S5 (RTX 5090) x 4096, `interactive` vs `throughput`** (the local-first assumption)
- Interactive: expect `--device RPC0,CUDA0 -ts 30,35`, 5090 9.34/30.70 GiB, 55.2 ms [brief `-ts 29,36`].
- Throughput: expect `-ts 49,16`, 55.2 ms (identical prior).
- Bandwidth-adjusted: 44.0 vs 36.8 ms. **Prediction: the throughput split is ~15-20% faster even for one user,**
  which would mean the local-first fill is wrong when the peer GPU is the faster one.
- `uv run python scripts/e2e_check.py --port $PORT --model qwen2.5-32b-instruct-q4_k_m --strategy auto --objective throughput --ctx 4096 --bench --label s5-32b-thr`
  (and `--objective interactive`).
- Could break: Windows-to-Windows RPC (both Public profile), and the CUDA rpc-server on the 5090 box.

**10. 72B Q4_K_M x S5 x 4096, `auto`** (biggest model on a pair; needs a manifest entry and ~43 GiB of free disk)
- Expect gpu_ram, `--device RPC0,CUDA0 -ngl 78 -ts 59,19`, CPU 3, 5090 at **30.36/30.70 GiB (1.1%)**. 134.9 ms by
  prior (7.4 tok/s), ~88-95 ms bandwidth-adjusted [brief: `-ngl 79 -ts 60,19`, CPU 2, 5090 30.89/31.00 = 0.3%].
- About 29 GiB crosses the network at load: ~17 min on Wi-Fi at 30 MB/s, ~5 min on 1 GbE, against
  `LOAD_TIMEOUT_S` 30 min.
- Could break:
  - 5090 OOM: a desktop on that GPU, or the CUDA context larger than the 450 + 64 MiB estimate.
  - The load timeout on a slow link.
  - G14 disk.

**11. 72B Q4_K_M x S6 (GX10 alone) x 32768** (new platform)
- Expect gpu, `--device CUDA0 -ts 81`, 50.99/83.2 GiB (or /110). Prior 114.3 ms (8.7 tok/s); GB10 bandwidth
  estimate ~206 ms (4.9 tok/s).
- Could break:
  - The `linux-arm64-cuda13.4` engine variant is unvalidated (`IMPLEMENTATION_STATUS.md:50`), and sm_121 kernels.
  - nvidia-smi reports `[N/A]` memory on GB10, which takes the `pool_id="uma"` path in `agent/inventory.py`
    `pc_devices`.
  - The default contribution caps the GX10 at 83.2 GiB with a 25.6 GiB reserve.
  - The decode prior is optimistic by ~1.8x.

**12. 72B Q4_K_M x S1 x 4096, `auto`** (true capacity mode end to end; needs ~43 GiB of free disk, e.g.
`HYPERCOMPUTE_HOME` on an external SSD)
- Expect gpu_disk, `--device CUDA0 -ngl 19 -ts 19`, 62 CPU layers (30.3 GiB) paged, CUDA0 10.99/11.16.
- The page cache holds ~17 GiB, so a cyclic re-read is expected: **~9-22 s/token**, 15-37 min for 100 tokens. The
  prior's 434.9 ms is meaningless here.
- Could break:
  - Gaps between tokens approaching `UPSTREAM_IDLE_TIMEOUT_S` = 180 s (`config.py:27`) on a slow disk.
  - Windows memory-manager behaviour under 30 GiB of mmap churn (the desktop stalls).
  - The UI banner and cancel path.

Not in the top 12 but worth a minute:
- **AirLLM 32B bf16** is blocked: it needs 124.1 GiB of disk.
- **vLLM forced** (`--engine vllm`) at 4k is the known-good path. The one regression check is `-ngl` vs vLLM's
  memory: a llama deployment must be stopped first, or vLLM refuses with the holder named.

## 10. What this says to build in the next push (ranked by how many rows it changes)

1. **Peers + RAM as a first-class overflow candidate** in `_overflow()` and in the tuner (`planner.py:388-402`,
   `tuner.py:3-6`). This affects the 11 grid rows of section 6: auto is up to 36% slower at the measured CPU cost (39% with two default-join Airs), and slower even under its own prior in 6 of them.
2. **Remote unified-memory budget** (`deployments.py:150-156`). Today a default `join` on a 16 GB Air yields 6.0
   GiB, not the 10.0 GiB it advertises. Either subtract the reserve from total unified RAM rather than from the Metal
   working set, or show the effective budget at join time. 17 of the 33 S2 rows change with the real budget, 6 of them in tier.
3. **Per-device decode cost and a calibrated hop.** Replace `DECODE_MS_PER_GB[kind]` with device bandwidth (or the
   tuner's measurement) so the 5090 and GB10 are not costed like a laptop 5070 Ti (`planner.py:44, 76-89`). Measure
   `HOP_MS` per peer at join: ~6 ms is already observed on a virtual switch. 28 close calls and 9 flips hinge on these.
4. **Context-dependent compute estimate / FA awareness** (`planner.py:67`). Either assert flash attention (`-fa on`)
   for ctx >= 16k, or budget the KQ buffer. This affects 124 `NEEDS_FA` rows.
5. **Page-cache-aware `gpu_disk`** (`planner.py:481-488`). Classify "paged but fits the page cache" (RAM speed)
   separately from "cyclic thrash" (seconds per token), and fix the warning and the warmup budget accordingly. 7 of
   18 auto `gpu_disk` rows are mislabeled.
6. **A safety margin on fill-to-capacity** (226 rows under 3%). Or keep it, but make "fits" depend on the first
   observed load and re-plan one layer down on OOM.
7. **`objective=throughput` on llama.cpp:** either raise `-np` (slots) or stop splitting proportionally. Today it only
   makes single-stream plans slower or equal.
8. **Selector quality guard:** do not swap Q8_0 for 4-bit AWQ without the user opting in (`selector.py:101-105`).
9. **Manifest and disk:** add entries (repo, revision, sha256 from the HF API) for 1.5B, 3B, 7B Q8_0, 14B Q4_K_M,
   32B Q6_K and 72B Q4_K_M. Add a doctor check for free disk against the largest planned GGUF. Optionally add a 72B
   HF entry if AirLLM is meant as its fallback.
10. **Host-side `--reserve`** for `hypercompute host` (only `--local-max-memory` exists, `cli.py:446-452`), so the
    owner can trade RAM-tier room for desktop headroom.
11. **GX10 profile (M6):** validate the arm64 CUDA engine, GB10 inventory and a unified-pool budget default that is
    not 0.65 x total.

## Appendix A. Model shapes used

| model | n_layer | n_embd | heads / kv | vocab | file GiB | layer MiB min/avg/max | token_embd / output MiB | KV/layer MiB @4k/16k/32k | compute est MiB | uniform-estimate layer / head MiB | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 24 | 896 | 14 / 2 | 151936 | 0.46 | 9.3 / 9.8 / 10.4 | 89 / 138 | 2 / 8 / 16 | 110 | 12.7 / 79 | ModelService().summary (local GGUF); header method matches byte-for-byte |
| 1.5B Q4_K_M | 28 | 1536 | 12 / 2 | 151936 | 1.04 | 25.1 / 26.9 / 28.6 | 125 / 183 | 4 / 16 / 32 | 120 | 28.2 / 135 | HF shard headers (Range GET) + HF API sizes |
| 3B Q4_K_M | 36 | 2048 | 16 / 2 | 151936 | 1.96 | 41.4 / 44.2 / 47.0 | 167 / 243 | 4 / 16 / 32 | 128 | 45.6 / 180 | HF shard headers (Range GET) + HF API sizes |
| 7B Q4_K_M | 28 | 3584 | 28 / 4 | 152064 | 4.36 | 125.1 / 133.6 / 142.2 | 292 / 426 | 8 / 32 / 64 | 152 | 136.8 / 315 | ModelService().summary (local GGUF); header method matches byte-for-byte |
| 7B Q8_0 | 28 | 3584 | 28 / 4 | 152064 | 7.54 | 236.2 / 236.2 / 236.2 | 552 / 552 | 8 / 32 / 64 | 152 | 236.2 / 552 | HF shard headers (Range GET) + HF API sizes |
| 14B Q4_K_M | 48 | 5120 | 40 / 8 | 152064 | 8.37 | 147.7 / 157.1 / 166.4 | 418 / 609 | 16 / 64 / 128 | 176 | 159.7 / 450 | HF shard headers (Range GET) + HF API sizes |
| 14B Q6_K | 48 | 5120 | 40 / 8 | 152064 | 11.29 | 215.4 / 215.4 / 215.4 | 609 / 609 | 16 / 64 / 128 | 176 | 215.4 / 609 | ModelService().summary (local GGUF); header method matches byte-for-byte |
| 14B Q8_0 | 48 | 5120 | 40 / 8 | 152064 | 14.62 | 279.0 / 279.0 / 279.0 | 789 / 789 | 16 / 64 / 128 | 176 | 279.0 / 789 | HF shard headers (Range GET) + HF API sizes |
| 32B Q4_K_M | 64 | 5120 | 40 / 8 | 152064 | 18.49 | 261.6 / 279.7 / 297.7 | 418 / 609 | 16 / 64 / 128 | 176 | 281.7 / 450 | HF shard headers (Range GET) + HF API sizes |
| 32B Q6_K | 64 | 5120 | 40 / 8 | 152064 | 25.04 | 381.5 / 381.5 / 381.5 | 609 / 609 | 16 / 64 / 128 | 176 | 381.5 / 609 | HF shard headers (Range GET) + HF API sizes |
| 72B Q4_K_M | 80 | 8192 | 64 / 8 | 152064 | 40.99 | 473.6 / 504.0 / 534.5 | 668 / 975 | 16 / 64 / 128 | 224 | 506.6 / 720 | HF shard headers (Range GET) + HF API sizes |

Uniform-estimate check (planner-test style shape vs the exact header shape, S1/S2/S5 x 3 ctx, auto): 10 of 99 cells differ, each by one layer:

- 32B Q4_K_M S2 ctx 32768: exact [True, 'gpu_ram', [24, 26], 15] vs uniform [True, 'gpu_ram', [23, 26], 16]
- 72B Q4_K_M S1 ctx 4096: exact [True, 'gpu_disk', [19], 62] vs uniform [True, 'gpu_disk', [20], 61]
- 72B Q4_K_M S1 ctx 16384: exact [True, 'gpu_disk', [18], 63] vs uniform [True, 'gpu_disk', [19], 62]
- 72B Q4_K_M S1 ctx 32768: exact [True, 'gpu_disk', [16], 65] vs uniform [True, 'gpu_disk', [17], 64]
- 72B Q4_K_M S2 ctx 4096: exact [True, 'gpu_disk', [19, 19], 43] vs uniform [True, 'gpu_disk', [18, 20], 43]
- 72B Q4_K_M S2 ctx 16384: exact [True, 'gpu_disk', [17, 18], 46] vs uniform [True, 'gpu_disk', [16, 19], 46]
- 72B Q4_K_M S2 ctx 32768: exact [True, 'gpu_disk', [15, 16], 50] vs uniform [True, 'gpu_disk', [15, 17], 49]
- 72B Q4_K_M S5 ctx 4096: exact [True, 'gpu_ram', [60, 19], 2] vs uniform [True, 'gpu_ram', [59, 20], 2]
- 72B Q4_K_M S5 ctx 16384: exact [True, 'gpu_ram', [55, 18], 8] vs uniform [True, 'gpu_ram', [54, 19], 8]
- 72B Q4_K_M S5 ctx 32768: exact [True, 'gpu_ram', [49, 16], 16] vs uniform [True, 'gpu_ram', [48, 17], 16]

## Appendix B. Full prediction grid (strategy auto)

Columns: est/budget per device in GiB (+H = holds the output head; a CPU row in gpu_disk excludes the paged weights); decode ms by the planner prior with the G14-measured CPU cost in parentheses (* = disk-bound, see section 5); overflow options = the two alternatives the tuner would measure. An empty notes cell means the planner gave only its strategy line and the contributed-budget line.

#### S1 auto

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 4.88/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 5.53/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 6.41/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 7.80/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 8.46/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 9.34/11.40 | 19.8 (19.8) |  |  |  |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | no peers / RAM 24.3 | HR<3%:CUDA0(0.5%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 37 | 12 / 37 | CPU 12L 3.61/12.87; CUDA0 36L+H 11.19/11.40 | 42.4 (53.5) | no peers / RAM 42.4 | HR<3%:CUDA0(1.8%) NEEDS_FA | 12 of 48 CPU layers (RAM) |
| 14B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 46 | 3 / 46 | CPU 3L 0.91/12.68; CUDA0 45L+H 11.38/11.40 | 37.0 (40.7) | no peers / RAM 37.0 | HR<3%:CUDA0(0.2%) | 3 of 48 CPU layers (RAM) |
| 14B Q6_K | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 38 | 11 / 38 | CPU 11L 3.24/12.68; CUDA0 37L+H 11.30/11.40 | 54.8 (68.5) | no peers / RAM 54.8 | HR<3%:CUDA0(0.9%) NEEDS_FA | 11 of 48 CPU layers (RAM) |
| 14B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 18 / 31 | CPU 18L 6.27/12.68; CUDA0 30L+H 11.27/11.40 | 70.4 (92.8) | no peers / RAM 70.4 | HR<3%:CUDA0(1.2%) CPU>30%(18/48) NEEDS_FA | 18 of 48 CPU layers (RAM) |
| 14B Q8_0 | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 35 | 14 / 35 | CPU 14L 4.27/12.51; CUDA0 34L+H 11.18/11.40 | 79.6 (102.2) | no peers / RAM 79.6 | HR<3%:CUDA0(2.0%) | 14 of 48 CPU layers (RAM) |
| 14B Q8_0 | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 30 | 19 / 30 | CPU 19L 6.60/12.51; CUDA0 29L+H 11.09/11.40 | 94.1 (124.6) | no peers / RAM 94.1 | HR<3%:CUDA0(2.7%) CPU>30%(19/48) NEEDS_FA | 19 of 48 CPU layers (RAM) |
| 14B Q8_0 | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 26 | 23 / 26 | CPU 23L 9.38/12.51; CUDA0 25L+H 11.32/11.40 | 105.6 (142.6) | no peers / RAM 105.6 | HR<3%:CUDA0(0.7%) CPU>30%(23/48) NEEDS_FA | 23 of 48 CPU layers (RAM) |
| 32B Q4_K_M | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 36 | 29 / 36 | CPU 29L 8.63/12.87; CUDA0 35L+H 11.29/11.40 | 135.3 (182.2) | no peers / RAM 135.3 | HR<3%:CUDA0(0.9%) CPU>30%(29/64) | 29 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 34 / 31 | CPU 34L 11.61/12.87; CUDA0 30L+H 11.31/11.40 | 149.2 (203.8) | no peers / RAM 149.2 | HR<3%:CUDA0(0.8%) CPU>30%(34/64) NEEDS_FA | 34 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 26 | 39 / 26 | CPU 39L 5.11/12.87; CUDA0 25L+H 11.21/11.40 | 163.4 (226.0)* | no peers / RAM no-fit -> disk paging | HR<3%:CUDA0(1.6%) CPU>30%(39/64) GPU_DISK NEEDS_FA | capacity mode 39 of 64 layers paged |
| 32B Q6_K | 4096 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 27 | 38 / 27 | CPU 38L 0.83/12.68; CUDA0 26L+H 11.30/11.40 | 219.2 (302.8)* | no peers / RAM no-fit -> disk paging | HR<3%:CUDA0(0.9%) CPU>30%(38/64) GPU_DISK | capacity mode 38 of 64 layers paged |
| 32B Q6_K | 16384 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 24 | 41 / 24 | CPU 41L 2.80/12.68; CUDA0 23L+H 11.21/11.40 | 231.0 (321.2)* | no peers / RAM no-fit -> disk paging | HR<3%:CUDA0(1.6%) CPU>30%(41/64) GPU_DISK NEEDS_FA | capacity mode 41 of 64 layers paged |
| 32B Q6_K | 32768 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 21 | 44 / 21 | CPU 44L 5.73/12.68; CUDA0 20L+H 11.16/11.40 | 242.8 (339.6)* | no peers / RAM no-fit -> disk paging | HR<3%:CUDA0(2.1%) CPU>30%(44/64) GPU_DISK NEEDS_FA | capacity mode 44 of 64 layers paged |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 19 | 62 / 19 | CPU 62L 1.25/12.62; CUDA0 18L+H 10.99/11.40 | 434.9 (613.7)* | no peers / RAM no-fit -> disk paging | CPU>30%(62/80) GPU_DISK | capacity mode 62 of 80 layers paged |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 18 | 63 / 18 | CPU 63L 4.22/12.62; CUDA0 17L+H 11.31/11.40 | 439.8 (621.4)* | no peers / RAM no-fit -> disk paging | HR<3%:CUDA0(0.8%) CPU>30%(63/80) GPU_DISK NEEDS_FA | capacity mode 63 of 80 layers paged |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | CUDA0 | 16 | 65 / 16 | CPU 65L 8.41/12.62; CUDA0 15L+H 11.14/11.40 | 450.2 (637.6)* | no peers / RAM no-fit -> disk paging | HR<3%:CUDA0(2.3%) CPU>30%(65/80) GPU_DISK NEEDS_FA | capacity mode 65 of 80 layers paged |

#### S2 auto

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 4.88/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 4096 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for  |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 5.53/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 16384 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16) |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 6.41/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 32768 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16) |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 7.80/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 4096 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q8_0) for ob |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 8.46/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 16384 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16) |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 9.34/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 32768 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16) |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | peers 28.0 / RAM 24.3 -> RAM (@18: 25.2 vs 28.0) | HR<3%:CUDA0(0.5%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 16384 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | peers 59.0 / RAM 24.3 -> RAM (@18: 25.2 vs 59.0) | HR<3%:CUDA0(0.5%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 37 | 12 / 37 | CPU 12L 3.61/12.87; CUDA0 36L+H 11.19/11.40 | 42.4 (53.5) | peers 43.1 / RAM 42.4 -> RAM (@18: 53.5 vs 43.1) | HR<3%:CUDA0(1.8%) FLIPS@18ms/GB CLOSE_CALL(<10%) NEEDS_FA | 12 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 32768 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 37 | 12 / 37 | CPU 12L 3.61/12.87; CUDA0 36L+H 11.19/11.40 | 42.4 (53.5) | peers 59.0 / RAM 42.4 -> RAM (@18: 53.5 vs 59.0) | HR<3%:CUDA0(1.8%) NEEDS_FA | 12 of 48 CPU layers (RAM) |
| 14B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 46 | 3 / 46 | CPU 3L 0.91/12.68; CUDA0 45L+H 11.38/11.40 | 37.0 (40.7) | peers 39.9 / RAM 37.0 -> RAM (@18: 40.7 vs 39.9) | HR<3%:CUDA0(0.2%) FLIPS@18ms/GB CLOSE_CALL(<10%) | 3 of 48 CPU layers (RAM) |
| 14B Q6_K | 4096 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 46 | 3 / 46 | CPU 3L 0.91/12.68; CUDA0 45L+H 11.38/11.40 | 37.0 (40.7) | peers 78.8 / RAM 37.0 -> RAM (@18: 40.7 vs 78.8) | HR<3%:CUDA0(0.2%) | 3 of 48 CPU layers (RAM) |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | Mac 11L 3.49/10.00; CUDA0 37L+H 11.30/11.40 | 54.7 (54.7) | peers 54.7 / RAM 54.8 -> peers (@18: 68.5 vs 54.7) | HR<3%:CUDA0(0.9%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q6_K | 16384 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 38 | 11 / 38 | CPU 11L 3.24/12.68; CUDA0 37L+H 11.30/11.40 | 54.8 (68.5) | peers 77.0 / RAM 54.8 -> RAM (@18: 68.5 vs 77.0) | HR<3%:CUDA0(0.9%) NEEDS_FA | 11 of 48 CPU layers (RAM) |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 18,31 | 0 | Mac 18L 6.52/10.00; CUDA0 30L+H 11.27/11.40 | 67.7 (67.7) | peers 67.7 / RAM 70.4 -> peers (@18: 92.8 vs 67.7) | HR<3%:CUDA0(1.2%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q6_K | 32768 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 18 / 31 | CPU 18L 6.27/12.68; CUDA0 30L+H 11.27/11.40 | 70.4 (92.8) | peers 78.8 / RAM 70.4 -> RAM (@18: 92.8 vs 78.8) | HR<3%:CUDA0(1.2%) CPU>30%(18/48) FLIPS@18ms/GB NEEDS_FA | 18 of 48 CPU layers (RAM) |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | Mac 14L 4.52/10.00; CUDA0 34L+H 11.18/11.40 | 76.9 (76.9) | peers 76.9 / RAM 79.6 -> peers (@18: 102.2 vs 76.9) | HR<3%:CUDA0(2.0%) CLOSE_CALL(<10%) |  |
| 14B Q8_0 | 4096 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 35 | 14 / 35 | CPU 14L 4.27/12.51; CUDA0 34L+H 11.18/11.40 | 79.6 (102.2) | peers 100.9 / RAM 79.6 -> RAM (@18: 102.2 vs 100.9) | HR<3%:CUDA0(2.0%) FLIPS@18ms/GB | 14 of 48 CPU layers (RAM) |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 19,30 | 0 | Mac 19L 6.85/10.00; CUDA0 29L+H 11.09/11.40 | 88.9 (88.9) | peers 88.9 / RAM 94.1 -> peers (@18: 124.6 vs 88.9) | HR<3%:CUDA0(2.7%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q8_0 | 16384 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 30 | 19 / 30 | CPU 19L 6.60/12.51; CUDA0 29L+H 11.09/11.40 | 94.1 (124.6) | peers 100.9 / RAM 94.1 -> RAM (@18: 124.6 vs 100.9) | HR<3%:CUDA0(2.7%) CPU>30%(19/48) FLIPS@18ms/GB CLOSE_CALL(<10%) NEEDS_FA | 19 of 48 CPU layers (RAM) |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 23,26 | 0 | Mac 23L 9.63/10.00; CUDA0 25L+H 11.32/11.40 | 98.5 (98.5) | peers 98.5 / RAM 105.6 -> peers (@18: 142.6 vs 98.5) | HR<3%:CUDA0(0.7%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q8_0 | 32768 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 23,26 | 0 | Mac 23L 9.63/10.00; CUDA0 25L+H 11.32/11.40 | 98.5 (98.5) | peers 98.5 / RAM 105.6 -> peers (@18: 142.6 vs 98.5) | HR<3%:CUDA0(0.7%) CLOSE_CALL(<10%) NEEDS_FA | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 29,36 | 0 | Mac 29L 8.88/10.00; CUDA0 35L+H 11.29/11.40 | 125.2 (125.2) | peers 125.2 / RAM 135.3 -> peers (@18: 182.2 vs 125.2) | HR<3%:CUDA0(0.9%) CLOSE_CALL(<10%) |  |
| 32B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 31,34 | 0 | Mac 31L 9.42/10.00; CUDA0 33L+H 10.75/11.40 | 129.7 (129.7) | peers 129.7 / RAM 135.3 -> peers (@18: 182.2 vs 129.7) | CLOSE_CALL(<10%) | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 34 / 31 | CPU 34L 11.61/12.87; CUDA0 30L+H 11.31/11.40 | 149.2 (203.8) | peers no-fit / RAM 149.2 | HR<3%:CUDA0(0.8%) CPU>30%(34/64) NEEDS_FA | 34 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 16384 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 34 / 31 | CPU 34L 11.61/12.87; CUDA0 30L+H 11.31/11.40 | 149.2 (203.8) | peers no-fit / RAM 149.2 | HR<3%:CUDA0(0.8%) CPU>30%(34/64) NEEDS_FA | 34 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 24,26 | 15 / 50 | CPU 15L 6.29/12.87; Mac 24L 9.90/10.00; CUDA0 25L+H 11.21/11.40 | 156.1 (180.8) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(1.0%) HR<3%:CUDA0(1.6%) NEEDS_FA | 15 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 32768 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 24,26 | 15 / 50 | CPU 15L 6.29/12.87; Mac 24L 9.90/10.00; CUDA0 25L+H 11.21/11.40 | 156.1 (180.8) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(1.0%) HR<3%:CUDA0(1.6%) NEEDS_FA | 15 of 64 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 24,27 | 14 / 51 | CPU 14L 5.67/12.68; Mac 24L 9.80/10.00; CUDA0 26L+H 11.30/11.40 | 207.3 (238.1) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(2.0%) HR<3%:CUDA0(0.9%) | 14 of 64 CPU layers (RAM) |
| 32B Q6_K | 4096 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 24,27 | 14 / 51 | CPU 14L 5.67/12.68; Mac 24L 9.80/10.00; CUDA0 26L+H 11.30/11.40 | 207.3 (238.1) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(2.0%) HR<3%:CUDA0(0.9%) | 14 of 64 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q6_K | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 21,24 | 20 / 45 | CPU 20L 8.94/12.68; Mac 21L 9.62/10.00; CUDA0 23L+H 11.21/11.40 | 221.1 (265.1) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:CUDA0(1.6%) CPU>30%(20/64) NEEDS_FA | 20 of 64 CPU layers (RAM) |
| 32B Q6_K | 16384 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 21,24 | 20 / 45 | CPU 20L 8.94/12.68; Mac 21L 9.62/10.00; CUDA0 23L+H 11.21/11.40 | 221.1 (265.1) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:CUDA0(1.6%) CPU>30%(20/64) NEEDS_FA | 20 of 64 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 19,21 | 25 / 40 | CPU 25L 12.67/12.68; Mac 19L 9.94/10.00; CUDA0 20L+H 11.16/11.40 | 234.3 (289.3) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.6%) HR<3%:CUDA0(2.1%) CPU>30%(25/64) NEEDS_FA | 25 of 64 CPU layers (RAM) |
| 32B Q6_K | 32768 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 19,21 | 25 / 40 | CPU 25L 12.67/12.68; Mac 19L 9.94/10.00; CUDA0 20L+H 11.16/11.40 | 234.3 (289.3) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.6%) HR<3%:CUDA0(2.1%) CPU>30%(25/64) NEEDS_FA | 25 of 64 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 19,19 | 43 / 38 | CPU 43L 0.95/12.62; Mac 19L 9.97/10.00; CUDA0 18L+H 10.99/11.40 | 422.7 (547.5)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.3%) CPU>30%(43/80) GPU_DISK | capacity mode 43 of 80 layers paged |
| 72B Q4_K_M | 4096 | throu | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 19,19 | 43 / 38 | CPU 43L 0.95/12.62; Mac 19L 9.97/10.00; CUDA0 18L+H 10.99/11.40 | 422.7 (547.5)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.3%) CPU>30%(43/80) GPU_DISK | capacity mode 43 of 80 layers paged; objective: throughput (layers split in proportion to each device's capacity) |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 17,18 | 46 / 35 | CPU 46L 3.16/12.62; Mac 17L 9.75/10.00; CUDA0 17L+H 11.31/11.40 | 429.3 (562.7)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(2.5%) HR<3%:CUDA0(0.8%) CPU>30%(46/80) GPU_DISK NEEDS_FA | capacity mode 46 of 80 layers paged |
| 72B Q4_K_M | 16384 | throu | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 17,18 | 46 / 35 | CPU 46L 3.16/12.62; Mac 17L 9.75/10.00; CUDA0 17L+H 11.31/11.40 | 429.3 (562.7)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(2.5%) HR<3%:CUDA0(0.8%) CPU>30%(46/80) GPU_DISK NEEDS_FA | capacity mode 46 of 80 layers paged; objective: throughput (layers split in proportion to each device's capacity) |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 15,16 | 50 / 31 | CPU 50L 6.53/12.62; Mac 15L 9.64/10.00; CUDA0 15L+H 11.14/11.40 | 441.4 (586.0)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:CUDA0(2.3%) CPU>30%(50/80) GPU_DISK NEEDS_FA | capacity mode 50 of 80 layers paged |
| 72B Q4_K_M | 32768 | throu | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 15,16 | 50 / 31 | CPU 50L 6.53/12.62; Mac 15L 9.64/10.00; CUDA0 15L+H 11.14/11.40 | 441.4 (586.0)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:CUDA0(2.3%) CPU>30%(50/80) GPU_DISK NEEDS_FA | capacity mode 50 of 80 layers paged; objective: throughput (layers split in proportion to each device's capacity) |

#### S3 auto

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 4.88/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 5.53/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 6.41/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 7.80/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 8.46/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 9.34/11.40 | 19.8 (19.8) |  |  |  |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | peers 28.0 / RAM 24.3 -> RAM (@18: 25.2 vs 28.0) | HR<3%:CUDA0(0.5%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 37 | 12 / 37 | CPU 12L 3.61/12.87; CUDA0 36L+H 11.19/11.40 | 42.4 (53.5) | peers 43.1 / RAM 42.4 -> RAM (@18: 53.5 vs 43.1) | HR<3%:CUDA0(1.8%) FLIPS@18ms/GB CLOSE_CALL(<10%) NEEDS_FA | 12 of 48 CPU layers (RAM) |
| 14B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 46 | 3 / 46 | CPU 3L 0.91/12.68; CUDA0 45L+H 11.38/11.40 | 37.0 (40.7) | peers 39.9 / RAM 37.0 -> RAM (@18: 40.7 vs 39.9) | HR<3%:CUDA0(0.2%) FLIPS@18ms/GB CLOSE_CALL(<10%) | 3 of 48 CPU layers (RAM) |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | Mac 11L 3.49/6.00; CUDA0 37L+H 11.30/11.40 | 54.7 (54.7) | peers 54.7 / RAM 54.8 -> peers (@18: 68.5 vs 54.7) | HR<3%:CUDA0(0.9%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 18 / 31 | CPU 18L 6.27/12.68; CUDA0 30L+H 11.27/11.40 | 70.4 (92.8) | peers no-fit / RAM 70.4 | HR<3%:CUDA0(1.2%) CPU>30%(18/48) NEEDS_FA | 18 of 48 CPU layers (RAM) |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | Mac 14L 4.52/6.00; CUDA0 34L+H 11.18/11.40 | 76.9 (76.9) | peers 76.9 / RAM 79.6 -> peers (@18: 102.2 vs 76.9) | HR<3%:CUDA0(2.0%) CLOSE_CALL(<10%) |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 30 | 19 / 30 | CPU 19L 6.60/12.51; CUDA0 29L+H 11.09/11.40 | 94.1 (124.6) | peers no-fit / RAM 94.1 | HR<3%:CUDA0(2.7%) CPU>30%(19/48) NEEDS_FA | 19 of 48 CPU layers (RAM) |
| 14B Q8_0 | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 26 | 23 / 26 | CPU 23L 9.38/12.51; CUDA0 25L+H 11.32/11.40 | 105.6 (142.6) | peers no-fit / RAM 105.6 | HR<3%:CUDA0(0.7%) CPU>30%(23/48) NEEDS_FA | 23 of 48 CPU layers (RAM) |
| 32B Q4_K_M | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 36 | 29 / 36 | CPU 29L 8.63/12.87; CUDA0 35L+H 11.29/11.40 | 135.3 (182.2) | peers no-fit / RAM 135.3 | HR<3%:CUDA0(0.9%) CPU>30%(29/64) | 29 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 31 | 34 / 31 | CPU 34L 11.61/12.87; CUDA0 30L+H 11.31/11.40 | 149.2 (203.8) | peers no-fit / RAM 149.2 | HR<3%:CUDA0(0.8%) CPU>30%(34/64) NEEDS_FA | 34 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 14,26 | 25 / 40 | CPU 25L 10.21/12.87; Mac 14L 5.99/6.00; CUDA0 25L+H 11.21/11.40 | 160.8 (201.2) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.2%) HR<3%:CUDA0(1.6%) CPU>30%(25/64) NEEDS_FA | 25 of 64 CPU layers (RAM) |
| 32B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 14,27 | 24 / 41 | CPU 24L 9.55/12.68; Mac 14L 5.92/6.00; CUDA0 26L+H 11.30/11.40 | 213.9 (266.7) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(1.3%) HR<3%:CUDA0(0.9%) CPU>30%(24/64) | 24 of 64 CPU layers (RAM) |
| 32B Q6_K | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 12,24 | 29 / 36 | CPU 29L 2.05/12.68; Mac 12L 5.71/6.00; CUDA0 23L+H 11.21/11.40 | 227.1 (290.9)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:CUDA0(1.6%) CPU>30%(29/64) GPU_DISK NEEDS_FA | capacity mode 29 of 64 layers paged |
| 32B Q6_K | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 11,21 | 33 / 32 | CPU 33L 4.36/12.68; Mac 11L 5.96/6.00; CUDA0 20L+H 11.16/11.40 | 239.6 (312.2)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.7%) HR<3%:CUDA0(2.1%) CPU>30%(33/64) GPU_DISK NEEDS_FA | capacity mode 33 of 64 layers paged |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 10,19 | 52 / 29 | CPU 52L 1.09/12.62; Mac 10L 5.49/6.00; CUDA0 18L+H 10.99/11.40 | 430.4 (580.8)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | CPU>30%(52/80) GPU_DISK | capacity mode 52 of 80 layers paged |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 10,18 | 53 / 28 | CPU 53L 3.59/12.62; Mac 10L 5.96/6.00; CUDA0 17L+H 11.31/11.40 | 435.3 (588.5)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.7%) HR<3%:CUDA0(0.8%) CPU>30%(53/80) GPU_DISK NEEDS_FA | capacity mode 53 of 80 layers paged |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 9,16 | 56 / 25 | CPU 56L 7.28/12.62; Mac 9L 6.00/6.00; CUDA0 15L+H 11.14/11.40 | 446.5 (608.3)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.0%) HR<3%:CUDA0(2.3%) CPU>30%(56/80) GPU_DISK NEEDS_FA | capacity mode 56 of 80 layers paged |

#### S4 auto

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 4.88/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 5.53/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 6.41/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 7.80/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 8.46/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 9.34/11.40 | 19.8 (19.8) |  |  |  |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | peers 28.0 / RAM 24.3 -> RAM (@18: 25.2 vs 28.0) | HR<3%:CUDA0(0.5%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 37 | 12 / 37 | CPU 12L 3.61/12.87; CUDA0 36L+H 11.19/11.40 | 42.4 (53.5) | peers 43.1 / RAM 42.4 -> RAM (@18: 53.5 vs 43.1) | HR<3%:CUDA0(1.8%) FLIPS@18ms/GB CLOSE_CALL(<10%) NEEDS_FA | 12 of 48 CPU layers (RAM) |
| 14B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 46 | 3 / 46 | CPU 3L 0.91/12.68; CUDA0 45L+H 11.38/11.40 | 37.0 (40.7) | peers 39.9 / RAM 37.0 -> RAM (@18: 40.7 vs 39.9) | HR<3%:CUDA0(0.2%) FLIPS@18ms/GB CLOSE_CALL(<10%) | 3 of 48 CPU layers (RAM) |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | Mac 11L 3.49/10.00; CUDA0 37L+H 11.30/11.40 | 54.7 (54.7) | peers 54.7 / RAM 54.8 -> peers (@18: 68.5 vs 54.7) | HR<3%:CUDA0(0.9%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 18,31 | 0 | Mac 18L 6.52/10.00; CUDA0 30L+H 11.27/11.40 | 67.7 (67.7) | peers 67.7 / RAM 70.4 -> peers (@18: 92.8 vs 67.7) | HR<3%:CUDA0(1.2%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | Mac 14L 4.52/10.00; CUDA0 34L+H 11.18/11.40 | 76.9 (76.9) | peers 76.9 / RAM 79.6 -> peers (@18: 102.2 vs 76.9) | HR<3%:CUDA0(2.0%) CLOSE_CALL(<10%) |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 19,30 | 0 | Mac 19L 6.85/10.00; CUDA0 29L+H 11.09/11.40 | 88.9 (88.9) | peers 88.9 / RAM 94.1 -> peers (@18: 124.6 vs 88.9) | HR<3%:CUDA0(2.7%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 23,26 | 0 | Mac 23L 9.63/10.00; CUDA0 25L+H 11.32/11.40 | 98.5 (98.5) | peers 98.5 / RAM 105.6 -> peers (@18: 142.6 vs 98.5) | HR<3%:CUDA0(0.7%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 29,36 | 0 | Mac 29L 8.88/10.00; CUDA0 35L+H 11.29/11.40 | 125.2 (125.2) | peers 125.2 / RAM 135.3 -> peers (@18: 182.2 vs 125.2) | HR<3%:CUDA0(0.9%) CLOSE_CALL(<10%) |  |
| 32B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 6,28,31 | 0 | Mac 6L 2.60/10.00; Mac 28L 9.74/10.00; CUDA0 30L+H 11.31/11.40 | 140.8 (140.8) | peers 140.8 / RAM 149.2 -> peers (@18: 203.8 vs 140.8) | HR<3%:Mac(2.6%) HR<3%:CUDA0(0.8%) CLOSE_CALL(<10%) NEEDS_FA |  |
| 32B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 16,23,26 | 0 | Mac 16L 6.92/10.00; Mac 23L 9.52/10.00; CUDA0 25L+H 11.21/11.40 | 152.7 (152.7) | peers 152.7 / RAM no-fit | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 14,24,27 | 0 | Mac 14L 5.92/10.00; Mac 24L 9.80/10.00; CUDA0 26L+H 11.30/11.40 | 202.1 (202.1) | peers 202.1 / RAM no-fit | HR<3%:Mac(2.0%) HR<3%:CUDA0(0.9%) |  |
| 32B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 20,21,24 | 0 | Mac 20L 9.19/10.00; Mac 21L 9.62/10.00; CUDA0 23L+H 11.21/11.40 | 211.9 (211.9) | peers 211.9 / RAM no-fit | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,RPC1,CUDA0 | 19,19,21 | 6 / 59 | CPU 6L 3.22/12.68; Mac 19L 9.94/10.00; Mac 19L 9.94/10.00; CUDA0 20L+H 11.16/11.40 | 225.7 (239.0) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.6%) HR<3%:Mac(0.6%) HR<3%:CUDA0(2.1%) NEEDS_FA | 6 of 64 CPU layers (RAM) |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | RPC0,RPC1,CUDA0 | 19,18,19 | 25 / 56 | CPU 25L 0.67/12.62; Mac 19L 9.97/10.00; Mac 18L 9.49/10.00; CUDA0 18L+H 10.99/11.40 | 411.3 (484.9)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(0.3%) CPU>30%(25/80) GPU_DISK | capacity mode 25 of 80 layers paged |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,RPC1,CUDA0 | 17,17,18 | 29 / 52 | CPU 29L 2.09/12.62; Mac 17L 9.81/10.00; Mac 17L 9.75/10.00; CUDA0 17L+H 11.31/11.40 | 418.8 (503.6)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:Mac(1.9%) HR<3%:Mac(2.5%) HR<3%:CUDA0(0.8%) CPU>30%(29/80) GPU_DISK NEEDS_FA | capacity mode 29 of 80 layers paged |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,RPC1,CUDA0 | 15,15,16 | 35 / 46 | CPU 35L 4.66/12.62; Mac 15L 9.64/10.00; Mac 15L 9.64/10.00; CUDA0 15L+H 11.14/11.40 | 432.6 (534.5)* | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:CUDA0(2.3%) CPU>30%(35/80) GPU_DISK NEEDS_FA | capacity mode 35 of 80 layers paged |

#### S5 auto

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/11.40 | 1.0 (1.0) |  |  |  |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/11.40 | 2.6 (2.6) |  |  |  |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/11.40 | 5.1 (5.1) |  |  |  |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 4.88/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 4096 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q4_k_m) for  |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 5.53/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 16384 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16) |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 6.41/11.40 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 32768 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16) |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 7.80/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 4096 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; vLLM serves qwen2.5-7b-instruct-awq (a different model id than qwen2.5-7b-instruct-q8_0) for ob |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 8.46/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 16384 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 4 full-length request(s) at once (--max-num-seqs 16) |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 9.34/11.40 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 32768 | throu | Y est | gpu | vllm | - | - | - | CUDA0 28L 10.63/10.66 | - |  | HR<3%:CUDA0(0.3%) | vLLM reuses the copy of Qwen/Qwen2.5-7B-Instruct-AWQ@b25037543e93 it downloaded; the KV cache holds about 2 full-length request(s) at once (--max-num-seqs 16) |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/11.40 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | peers 26.6 / RAM 24.3 -> RAM (@18: 25.2 vs 26.6) | HR<3%:CUDA0(0.5%) CLOSE_CALL(<10%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 16384 | throu | Y est | gpu_ram | llama_rpc | CUDA0 | 48 | 1 / 48 | CPU 1L 0.46/12.87; CUDA0 47L+H 11.34/11.40 | 24.3 (25.2) | peers 26.6 / RAM 24.3 -> RAM (@18: 25.2 vs 26.6) | HR<3%:CUDA0(0.5%) CLOSE_CALL(<10%) NEEDS_FA | 1 of 48 CPU layers (RAM) |
| 14B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 12,37 | 0 | GPU 12L 4.05/31.00; CUDA0 36L+H 11.19/11.40 | 26.6 (26.6) | peers 26.6 / RAM 42.4 -> peers (@18: 53.5 vs 26.6) | HR<3%:CUDA0(1.8%) NEEDS_FA |  |
| 14B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 36,13 | 0 | GPU 36L 10.66/31.00; CUDA0 12L+H 4.58/11.40 | 26.6 (26.6) | peers 26.6 / RAM 42.4 -> peers (@18: 53.5 vs 26.6) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 14B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 3,46 | 0 | GPU 3L 1.35/31.00; CUDA0 45L+H 11.38/11.40 | 34.3 (34.3) | peers 34.3 / RAM 37.0 -> peers (@18: 40.7 vs 34.3) | HR<3%:CUDA0(0.2%) CLOSE_CALL(<10%) |  |
| 14B Q6_K | 4096 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 36,13 | 0 | GPU 36L 8.81/31.00; CUDA0 12L+H 3.92/11.40 | 34.3 (34.3) | peers 34.3 / RAM 37.0 -> peers (@18: 40.7 vs 34.3) | CLOSE_CALL(<10%) | objective: throughput (layers split in proportion to each device's capacity) |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | GPU 11L 3.68/31.00; CUDA0 37L+H 11.30/11.40 | 34.3 (34.3) | peers 34.3 / RAM 54.8 -> peers (@18: 68.5 vs 34.3) | HR<3%:CUDA0(0.9%) NEEDS_FA |  |
| 14B Q6_K | 16384 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 36,13 | 0 | GPU 36L 10.50/31.00; CUDA0 12L+H 4.48/11.40 | 34.3 (34.3) | peers 34.3 / RAM 54.8 -> peers (@18: 68.5 vs 34.3) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 18,31 | 0 | GPU 18L 6.71/31.00; CUDA0 30L+H 11.27/11.40 | 34.3 (34.3) | peers 34.3 / RAM 70.4 -> peers (@18: 92.8 vs 34.3) | HR<3%:CUDA0(1.2%) NEEDS_FA |  |
| 14B Q6_K | 32768 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 36,13 | 0 | GPU 36L 12.75/31.00; CUDA0 12L+H 5.23/11.40 | 34.3 (34.3) | peers 34.3 / RAM 70.4 -> peers (@18: 92.8 vs 34.3) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | GPU 14L 4.71/31.00; CUDA0 34L+H 11.18/11.40 | 43.3 (43.3) | peers 43.3 / RAM 79.6 -> peers (@18: 102.2 vs 43.3) | HR<3%:CUDA0(2.0%) |  |
| 14B Q8_0 | 4096 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 37,12 | 0 | GPU 37L 11.33/31.00; CUDA0 11L+H 4.55/11.40 | 43.3 (43.3) | peers 43.3 / RAM 79.6 -> peers (@18: 102.2 vs 43.3) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 19,30 | 0 | GPU 19L 7.04/31.00; CUDA0 29L+H 11.09/11.40 | 43.3 (43.3) | peers 43.3 / RAM 94.1 -> peers (@18: 124.6 vs 43.3) | HR<3%:CUDA0(2.7%) NEEDS_FA |  |
| 14B Q8_0 | 16384 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 37,12 | 0 | GPU 37L 13.07/31.00; CUDA0 11L+H 5.07/11.40 | 43.3 (43.3) | peers 43.3 / RAM 94.1 -> peers (@18: 124.6 vs 43.3) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 23,26 | 0 | GPU 23L 9.81/31.00; CUDA0 25L+H 11.32/11.40 | 43.3 (43.3) | peers 43.3 / RAM 105.6 -> peers (@18: 142.6 vs 43.3) | HR<3%:CUDA0(0.7%) NEEDS_FA |  |
| 14B Q8_0 | 32768 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 37,12 | 0 | GPU 37L 15.38/31.00; CUDA0 11L+H 5.75/11.40 | 43.3 (43.3) | peers 43.3 / RAM 105.6 -> peers (@18: 142.6 vs 43.3) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 29,36 | 0 | GPU 29L 9.07/31.00; CUDA0 35L+H 11.29/11.40 | 55.2 (55.2) | peers 55.2 / RAM 135.3 -> peers (@18: 182.2 vs 55.2) | HR<3%:CUDA0(0.9%) |  |
| 32B Q4_K_M | 4096 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 48,17 | 0 | GPU 48L 14.43/31.00; CUDA0 16L+H 5.93/11.40 | 55.2 (55.2) | peers 55.2 / RAM 135.3 -> peers (@18: 182.2 vs 55.2) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 34,31 | 0 | GPU 34L 12.05/31.00; CUDA0 30L+H 11.31/11.40 | 55.2 (55.2) | peers 55.2 / RAM 149.2 -> peers (@18: 203.8 vs 55.2) | HR<3%:CUDA0(0.8%) NEEDS_FA |  |
| 32B Q4_K_M | 16384 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 48,17 | 0 | GPU 48L 16.68/31.00; CUDA0 16L+H 6.68/11.40 | 55.2 (55.2) | peers 55.2 / RAM 149.2 -> peers (@18: 203.8 vs 55.2) |  | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 39,26 | 0 | GPU 39L 16.15/31.00; CUDA0 25L+H 11.21/11.40 | 55.2 (55.2) | peers 55.2 / RAM no-fit | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q4_K_M | 32768 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 49,16 | 0 | GPU 49L 20.06/31.00; CUDA0 15L+H 7.30/11.40 | 55.2 (55.2) | peers 55.2 / RAM no-fit |  | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 38,27 | 0 | GPU 38L 15.43/31.00; CUDA0 26L+H 11.30/11.40 | 73.3 (73.3) | peers 73.3 / RAM no-fit | HR<3%:CUDA0(0.9%) |  |
| 32B Q6_K | 4096 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 48,17 | 0 | GPU 48L 19.31/31.00; CUDA0 16L+H 7.42/11.40 | 73.3 (73.3) | peers 73.3 / RAM no-fit |  | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 41,24 | 0 | GPU 41L 18.51/31.00; CUDA0 23L+H 11.21/11.40 | 73.3 (73.3) | peers 73.3 / RAM no-fit | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 16384 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 48,17 | 0 | GPU 48L 21.56/31.00; CUDA0 16L+H 8.17/11.40 | 73.3 (73.3) | peers 73.3 / RAM no-fit |  | objective: throughput (layers split in proportion to each device's capacity) |
| 32B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 44,21 | 0 | GPU 44L 22.57/31.00; CUDA0 20L+H 11.16/11.40 | 73.3 (73.3) | peers 73.3 / RAM no-fit | HR<3%:CUDA0(2.1%) NEEDS_FA |  |
| 32B Q6_K | 32768 | throu | Y est | gpu | llama_rpc | RPC0,CUDA0 | 48,17 | 0 | GPU 48L 24.56/31.00; CUDA0 16L+H 9.17/11.40 | 73.3 (73.3) | peers 73.3 / RAM no-fit | NEEDS_FA | objective: throughput (layers split in proportion to each device's capacity) |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 60,19 | 2 / 79 | CPU 2L 1.36/12.62; GPU 60L 30.89/31.00; CUDA0 18L+H 10.99/11.40 | 129.4 (135.5) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:GPU(0.3%) | 2 of 80 CPU layers (RAM) |
| 72B Q4_K_M | 4096 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 60,19 | 2 / 79 | CPU 2L 1.36/12.62; GPU 60L 30.89/31.00; CUDA0 18L+H 10.99/11.40 | 129.4 (135.5) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:GPU(0.3%) | 2 of 80 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 55,18 | 8 / 73 | CPU 8L 4.96/12.62; GPU 55L 30.73/31.00; CUDA0 17L+H 11.31/11.40 | 162.5 (187.2) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:GPU(0.9%) HR<3%:CUDA0(0.8%) NEEDS_FA | 8 of 80 CPU layers (RAM) |
| 72B Q4_K_M | 16384 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 55,18 | 8 / 73 | CPU 8L 4.96/12.62; GPU 55L 30.73/31.00; CUDA0 17L+H 11.31/11.40 | 162.5 (187.2) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:GPU(0.9%) HR<3%:CUDA0(0.8%) NEEDS_FA | 8 of 80 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 49,16 | 16 / 65 | CPU 16L 10.39/12.62; GPU 49L 30.46/31.00; CUDA0 15L+H 11.14/11.40 | 204.2 (252.1) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:GPU(1.7%) HR<3%:CUDA0(2.3%) NEEDS_FA | 16 of 80 CPU layers (RAM) |
| 72B Q4_K_M | 32768 | throu | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 49,16 | 16 / 65 | CPU 16L 10.39/12.62; GPU 49L 30.46/31.00; CUDA0 15L+H 11.14/11.40 | 204.2 (252.1) | peers no-fit / RAM no-fit -> ladder (peers+RAM, then disk) | HR<3%:GPU(1.7%) HR<3%:CUDA0(2.3%) NEEDS_FA | 16 of 80 CPU layers (RAM); objective: throughput (layers split in proportion to each device's capacity) |

#### S6 auto

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 0.96/110.00 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.10/110.00 | 1.0 (1.0) |  |  |  |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 25 | 0 | CUDA0 24L+H 1.29/110.00 | 1.0 (1.0) |  |  |  |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.58/110.00 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 1.91/110.00 | 2.6 (2.6) |  |  |  |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 2.34/110.00 | 2.6 (2.6) |  |  |  |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.50/110.00 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 2.92/110.00 | 5.1 (5.1) |  |  |  |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 37 | 0 | CUDA0 36L+H 3.48/110.00 | 5.1 (5.1) |  |  |  |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 4.88/110.00 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 5.53/110.00 | 11.5 (11.5) |  |  |  |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 6.41/110.00 | 11.5 (11.5) |  |  |  |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 7.80/110.00 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 8.46/110.00 | 19.8 (19.8) |  |  |  |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 29 | 0 | CUDA0 28L+H 9.34/110.00 | 19.8 (19.8) |  |  |  |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 9.32/110.00 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 11.57/110.00 | 22.6 (22.6) |  |  |  |
| 14B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 14.57/110.00 | 22.6 (22.6) |  |  |  |
| 14B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 12.05/110.00 | 30.3 (30.3) |  |  |  |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 14.30/110.00 | 30.3 (30.3) |  |  |  |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 17.30/110.00 | 30.3 (30.3) |  |  |  |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 15.21/110.00 | 39.3 (39.3) |  |  |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 17.46/110.00 | 39.3 (39.3) |  |  |  |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 49 | 0 | CUDA0 48L+H 20.46/110.00 | 39.3 (39.3) |  |  |  |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 65 | 0 | CUDA0 64L+H 19.69/110.00 | 51.2 (51.2) |  |  |  |
| 32B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 65 | 0 | CUDA0 64L+H 22.69/110.00 | 51.2 (51.2) |  |  |  |
| 32B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 65 | 0 | CUDA0 64L+H 26.69/110.00 | 51.2 (51.2) |  |  |  |
| 32B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 65 | 0 | CUDA0 64L+H 26.05/110.00 | 69.3 (69.3) |  |  |  |
| 32B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 65 | 0 | CUDA0 64L+H 29.05/110.00 | 69.3 (69.3) |  |  |  |
| 32B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 65 | 0 | CUDA0 64L+H 33.05/110.00 | 69.3 (69.3) |  |  |  |
| 72B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | CUDA0 | 81 | 0 | CUDA0 80L+H 42.24/110.00 | 114.3 (114.3) |  |  |  |
| 72B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | CUDA0 | 81 | 0 | CUDA0 80L+H 45.99/110.00 | 114.3 (114.3) |  |  |  |
| 72B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | CUDA0 | 81 | 0 | CUDA0 80L+H 50.99/110.00 | 114.3 (114.3) |  |  |  |

## Appendix C. Run together (strategy all_selected, interactive)

#### S2 all_selected

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | Mac 1L 0.43/10.00; CUDA0 23L+H 0.95/11.40 | 5.1 (5.1) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | Mac 1L 0.44/10.00; CUDA0 23L+H 1.08/11.40 | 5.1 (5.1) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | Mac 1L 0.45/10.00; CUDA0 23L+H 1.26/11.40 | 5.1 (5.1) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.46/10.00; CUDA0 27L+H 1.55/11.40 | 6.8 (6.8) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.47/10.00; CUDA0 27L+H 1.86/11.40 | 6.8 (6.8) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.49/10.00; CUDA0 27L+H 2.29/11.40 | 6.8 (6.8) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | Mac 1L 0.49/10.00; CUDA0 35L+H 2.45/11.40 | 9.5 (9.5) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | Mac 1L 0.50/10.00; CUDA0 35L+H 2.86/11.40 | 9.5 (9.5) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | Mac 1L 0.51/10.00; CUDA0 35L+H 3.40/11.40 | 9.5 (9.5) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.61/10.00; CUDA0 27L+H 4.73/11.40 | 16.8 (16.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.63/10.00; CUDA0 27L+H 5.36/11.40 | 16.8 (16.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.66/10.00; CUDA0 27L+H 6.21/11.40 | 16.8 (16.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.70/10.00; CUDA0 27L+H 7.57/11.40 | 25.9 (25.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.72/10.00; CUDA0 27L+H 8.20/11.40 | 25.9 (25.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.75/10.00; CUDA0 27L+H 9.04/11.40 | 25.9 (25.9) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,48 | 0 | Mac 1L 0.66/10.00; CUDA0 47L+H 9.14/11.40 | 28.0 (28.0) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,48 | 0 | Mac 1L 0.71/10.00; CUDA0 47L+H 11.34/11.40 | 28.0 (28.0) |  | HR<3%:CUDA0(0.5%) NEEDS_FA |  |
| 14B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 12,37 | 0 | Mac 12L 3.86/10.00; CUDA0 36L+H 11.19/11.40 | 43.1 (43.1) |  | HR<3%:CUDA0(1.8%) NEEDS_FA |  |
| 14B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 3,46 | 0 | Mac 3L 1.16/10.00; CUDA0 45L+H 11.38/11.40 | 39.9 (39.9) |  | HR<3%:CUDA0(0.2%) |  |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | Mac 11L 3.49/10.00; CUDA0 37L+H 11.30/11.40 | 54.7 (54.7) |  | HR<3%:CUDA0(0.9%) NEEDS_FA |  |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 18,31 | 0 | Mac 18L 6.52/10.00; CUDA0 30L+H 11.27/11.40 | 67.7 (67.7) |  | HR<3%:CUDA0(1.2%) NEEDS_FA |  |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | Mac 14L 4.52/10.00; CUDA0 34L+H 11.18/11.40 | 76.9 (76.9) |  | HR<3%:CUDA0(2.0%) |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 19,30 | 0 | Mac 19L 6.85/10.00; CUDA0 29L+H 11.09/11.40 | 88.9 (88.9) |  | HR<3%:CUDA0(2.7%) NEEDS_FA |  |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 23,26 | 0 | Mac 23L 9.63/10.00; CUDA0 25L+H 11.32/11.40 | 98.5 (98.5) |  | HR<3%:CUDA0(0.7%) NEEDS_FA |  |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 29,36 | 0 | Mac 29L 8.88/10.00; CUDA0 35L+H 11.29/11.40 | 125.2 (125.2) |  | HR<3%:CUDA0(0.9%) |  |
| 32B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 28,31 | 6 / 59 | CPU 6L 2.35/12.87; Mac 28L 9.74/10.00; CUDA0 30L+H 11.31/11.40 | 139.9 (150.2) |  | HR<3%:Mac(2.6%) HR<3%:CUDA0(0.8%) NEEDS_FA | 6 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 24,26 | 15 / 50 | CPU 15L 6.29/12.87; Mac 24L 9.90/10.00; CUDA0 25L+H 11.21/11.40 | 156.1 (180.8) |  | HR<3%:Mac(1.0%) HR<3%:CUDA0(1.6%) NEEDS_FA | 15 of 64 CPU layers (RAM) |
| 32B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 24,27 | 14 / 51 | CPU 14L 5.67/12.68; Mac 24L 9.80/10.00; CUDA0 26L+H 11.30/11.40 | 207.3 (238.1) |  | HR<3%:Mac(2.0%) HR<3%:CUDA0(0.9%) | 14 of 64 CPU layers (RAM) |
| 32B Q6_K | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 21,24 | 20 / 45 | CPU 20L 8.94/12.68; Mac 21L 9.62/10.00; CUDA0 23L+H 11.21/11.40 | 221.1 (265.1) |  | HR<3%:CUDA0(1.6%) CPU>30%(20/64) NEEDS_FA | 20 of 64 CPU layers (RAM) |
| 32B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 19,21 | 25 / 40 | CPU 25L 12.67/12.68; Mac 19L 9.94/10.00; CUDA0 20L+H 11.16/11.40 | 234.3 (289.3) |  | HR<3%:Mac(0.6%) HR<3%:CUDA0(2.1%) CPU>30%(25/64) NEEDS_FA | 25 of 64 CPU layers (RAM) |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 19,19 | 43 / 38 | CPU 43L 0.95/12.62; Mac 19L 9.97/10.00; CUDA0 18L+H 10.99/11.40 | 422.7 (547.5)* |  | HR<3%:Mac(0.3%) CPU>30%(43/80) GPU_DISK | capacity mode 43 of 80 layers paged |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 17,18 | 46 / 35 | CPU 46L 3.16/12.62; Mac 17L 9.75/10.00; CUDA0 17L+H 11.31/11.40 | 429.3 (562.7)* |  | HR<3%:Mac(2.5%) HR<3%:CUDA0(0.8%) CPU>30%(46/80) GPU_DISK NEEDS_FA | capacity mode 46 of 80 layers paged |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 15,16 | 50 / 31 | CPU 50L 6.53/12.62; Mac 15L 9.64/10.00; CUDA0 15L+H 11.14/11.40 | 441.4 (586.0)* |  | HR<3%:CUDA0(2.3%) CPU>30%(50/80) GPU_DISK NEEDS_FA | capacity mode 50 of 80 layers paged |

#### S3 all_selected

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | Mac 1L 0.43/6.00; CUDA0 23L+H 0.95/11.40 | 5.1 (5.1) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | Mac 1L 0.44/6.00; CUDA0 23L+H 1.08/11.40 | 5.1 (5.1) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | Mac 1L 0.45/6.00; CUDA0 23L+H 1.26/11.40 | 5.1 (5.1) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.46/6.00; CUDA0 27L+H 1.55/11.40 | 6.8 (6.8) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.47/6.00; CUDA0 27L+H 1.86/11.40 | 6.8 (6.8) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.49/6.00; CUDA0 27L+H 2.29/11.40 | 6.8 (6.8) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | Mac 1L 0.49/6.00; CUDA0 35L+H 2.45/11.40 | 9.5 (9.5) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | Mac 1L 0.50/6.00; CUDA0 35L+H 2.86/11.40 | 9.5 (9.5) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | Mac 1L 0.51/6.00; CUDA0 35L+H 3.40/11.40 | 9.5 (9.5) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.61/6.00; CUDA0 27L+H 4.73/11.40 | 16.8 (16.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.63/6.00; CUDA0 27L+H 5.36/11.40 | 16.8 (16.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.66/6.00; CUDA0 27L+H 6.21/11.40 | 16.8 (16.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.70/6.00; CUDA0 27L+H 7.57/11.40 | 25.9 (25.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.72/6.00; CUDA0 27L+H 8.20/11.40 | 25.9 (25.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | Mac 1L 0.75/6.00; CUDA0 27L+H 9.04/11.40 | 25.9 (25.9) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,48 | 0 | Mac 1L 0.66/6.00; CUDA0 47L+H 9.14/11.40 | 28.0 (28.0) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,48 | 0 | Mac 1L 0.71/6.00; CUDA0 47L+H 11.34/11.40 | 28.0 (28.0) |  | HR<3%:CUDA0(0.5%) NEEDS_FA |  |
| 14B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 12,37 | 0 | Mac 12L 3.86/6.00; CUDA0 36L+H 11.19/11.40 | 43.1 (43.1) |  | HR<3%:CUDA0(1.8%) NEEDS_FA |  |
| 14B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 3,46 | 0 | Mac 3L 1.16/6.00; CUDA0 45L+H 11.38/11.40 | 39.9 (39.9) |  | HR<3%:CUDA0(0.2%) |  |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | Mac 11L 3.49/6.00; CUDA0 37L+H 11.30/11.40 | 54.7 (54.7) |  | HR<3%:CUDA0(0.9%) NEEDS_FA |  |
| 14B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 16,31 | 2 / 47 | CPU 2L 0.91/12.68; Mac 16L 5.85/6.00; CUDA0 30L+H 11.27/11.40 | 68.4 (70.9) |  | HR<3%:Mac(2.5%) HR<3%:CUDA0(1.2%) NEEDS_FA | 2 of 48 CPU layers (RAM) |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | Mac 14L 4.52/6.00; CUDA0 34L+H 11.18/11.40 | 76.9 (76.9) |  | HR<3%:CUDA0(2.0%) |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 16,30 | 3 / 46 | CPU 3L 1.24/12.51; Mac 16L 5.84/6.00; CUDA0 29L+H 11.09/11.40 | 90.3 (95.2) |  | HR<3%:Mac(2.6%) HR<3%:CUDA0(2.7%) NEEDS_FA | 3 of 48 CPU layers (RAM) |
| 14B Q8_0 | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 13,26 | 10 / 39 | CPU 10L 4.21/12.51; Mac 13L 5.65/6.00; CUDA0 25L+H 11.32/11.40 | 103.3 (119.4) |  | HR<3%:CUDA0(0.7%) NEEDS_FA | 10 of 48 CPU layers (RAM) |
| 32B Q4_K_M | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 19,36 | 10 / 55 | CPU 10L 3.23/12.87; Mac 19L 5.88/6.00; CUDA0 35L+H 11.29/11.40 | 130.2 (147.0) |  | HR<3%:Mac(2.0%) HR<3%:CUDA0(0.9%) | 10 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 16,31 | 18 / 47 | CPU 18L 6.35/12.87; Mac 16L 5.75/6.00; CUDA0 30L+H 11.31/11.40 | 145.6 (175.1) |  | HR<3%:CUDA0(0.8%) NEEDS_FA | 18 of 64 CPU layers (RAM) |
| 32B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 14,26 | 25 / 40 | CPU 25L 10.21/12.87; Mac 14L 5.99/6.00; CUDA0 25L+H 11.21/11.40 | 160.8 (201.2) |  | HR<3%:Mac(0.2%) HR<3%:CUDA0(1.6%) CPU>30%(25/64) NEEDS_FA | 25 of 64 CPU layers (RAM) |
| 32B Q6_K | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 14,27 | 24 / 41 | CPU 24L 9.55/12.68; Mac 14L 5.92/6.00; CUDA0 26L+H 11.30/11.40 | 213.9 (266.7) |  | HR<3%:Mac(1.3%) HR<3%:CUDA0(0.9%) CPU>30%(24/64) | 24 of 64 CPU layers (RAM) |
| 32B Q6_K | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 12,24 | 29 / 36 | CPU 29L 2.05/12.68; Mac 12L 5.71/6.00; CUDA0 23L+H 11.21/11.40 | 227.1 (290.9)* |  | HR<3%:CUDA0(1.6%) CPU>30%(29/64) GPU_DISK NEEDS_FA | capacity mode 29 of 64 layers paged |
| 32B Q6_K | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 11,21 | 33 / 32 | CPU 33L 4.36/12.68; Mac 11L 5.96/6.00; CUDA0 20L+H 11.16/11.40 | 239.6 (312.2)* |  | HR<3%:Mac(0.7%) HR<3%:CUDA0(2.1%) CPU>30%(33/64) GPU_DISK NEEDS_FA | capacity mode 33 of 64 layers paged |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 10,19 | 52 / 29 | CPU 52L 1.09/12.62; Mac 10L 5.49/6.00; CUDA0 18L+H 10.99/11.40 | 430.4 (580.8)* |  | CPU>30%(52/80) GPU_DISK | capacity mode 52 of 80 layers paged |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 10,18 | 53 / 28 | CPU 53L 3.59/12.62; Mac 10L 5.96/6.00; CUDA0 17L+H 11.31/11.40 | 435.3 (588.5)* |  | HR<3%:Mac(0.7%) HR<3%:CUDA0(0.8%) CPU>30%(53/80) GPU_DISK NEEDS_FA | capacity mode 53 of 80 layers paged |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,CUDA0 | 9,16 | 56 / 25 | CPU 56L 7.28/12.62; Mac 9L 6.00/6.00; CUDA0 15L+H 11.14/11.40 | 446.5 (608.3)* |  | HR<3%:Mac(0.0%) HR<3%:CUDA0(2.3%) CPU>30%(56/80) GPU_DISK NEEDS_FA | capacity mode 56 of 80 layers paged |

#### S4 all_selected

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,23 | 0 | Mac 1L 0.43/10.00; Mac 1L 0.43/10.00; CUDA0 22L+H 0.93/11.40 | 9.2 (9.2) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,23 | 0 | Mac 1L 0.44/10.00; Mac 1L 0.44/10.00; CUDA0 22L+H 1.06/11.40 | 9.2 (9.2) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,23 | 0 | Mac 1L 0.45/10.00; Mac 1L 0.45/10.00; CUDA0 22L+H 1.24/11.40 | 9.2 (9.2) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.46/10.00; Mac 1L 0.46/10.00; CUDA0 26L+H 1.52/11.40 | 11.1 (11.1) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.47/10.00; Mac 1L 0.47/10.00; CUDA0 26L+H 1.82/11.40 | 11.1 (11.1) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.49/10.00; Mac 1L 0.49/10.00; CUDA0 26L+H 2.23/11.40 | 11.1 (11.1) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,35 | 0 | Mac 1L 0.49/10.00; Mac 1L 0.49/10.00; CUDA0 34L+H 2.40/11.40 | 13.9 (13.9) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,35 | 0 | Mac 1L 0.50/10.00; Mac 1L 0.50/10.00; CUDA0 34L+H 2.80/11.40 | 13.9 (13.9) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,35 | 0 | Mac 1L 0.51/10.00; Mac 1L 0.51/10.00; CUDA0 34L+H 3.33/11.40 | 13.9 (13.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.61/10.00; Mac 1L 0.61/10.00; CUDA0 26L+H 4.58/11.40 | 22.0 (22.0) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.63/10.00; Mac 1L 0.63/10.00; CUDA0 26L+H 5.19/11.40 | 22.0 (22.0) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.66/10.00; Mac 1L 0.66/10.00; CUDA0 26L+H 6.01/11.40 | 22.0 (22.0) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.70/10.00; Mac 1L 0.70/10.00; CUDA0 26L+H 7.33/11.40 | 31.9 (31.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.72/10.00; Mac 1L 0.72/10.00; CUDA0 26L+H 7.94/11.40 | 31.9 (31.9) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,27 | 0 | Mac 1L 0.75/10.00; Mac 1L 0.75/10.00; CUDA0 26L+H 8.75/11.40 | 31.9 (31.9) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,47 | 0 | Mac 1L 0.66/10.00; Mac 1L 0.66/10.00; CUDA0 46L+H 8.96/11.40 | 33.4 (33.4) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,1,47 | 0 | Mac 1L 0.71/10.00; Mac 1L 0.71/10.00; CUDA0 46L+H 11.12/11.40 | 33.4 (33.4) |  | HR<3%:CUDA0(2.5%) NEEDS_FA |  |
| 14B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,11,37 | 0 | Mac 1L 0.77/10.00; Mac 11L 3.57/10.00; CUDA0 36L+H 11.19/11.40 | 47.1 (47.1) |  | HR<3%:CUDA0(1.8%) NEEDS_FA |  |
| 14B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,2,46 | 0 | Mac 1L 0.71/10.00; Mac 2L 0.94/10.00; CUDA0 45L+H 11.38/11.40 | 43.9 (43.9) |  | HR<3%:CUDA0(0.2%) |  |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,10,38 | 0 | Mac 1L 0.76/10.00; Mac 10L 3.21/10.00; CUDA0 37L+H 11.30/11.40 | 58.7 (58.7) |  | HR<3%:CUDA0(0.9%) NEEDS_FA |  |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,17,31 | 0 | Mac 1L 0.82/10.00; Mac 17L 6.19/10.00; CUDA0 30L+H 11.27/11.40 | 71.7 (71.7) |  | HR<3%:CUDA0(1.2%) NEEDS_FA |  |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,13,35 | 0 | Mac 1L 0.77/10.00; Mac 13L 4.23/10.00; CUDA0 34L+H 11.18/11.40 | 80.9 (80.9) |  | HR<3%:CUDA0(2.0%) |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,18,30 | 0 | Mac 1L 0.82/10.00; Mac 18L 6.51/10.00; CUDA0 29L+H 11.09/11.40 | 92.9 (92.9) |  | HR<3%:CUDA0(2.7%) NEEDS_FA |  |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,22,26 | 0 | Mac 1L 0.88/10.00; Mac 22L 9.23/10.00; CUDA0 25L+H 11.32/11.40 | 102.5 (102.5) |  | HR<3%:CUDA0(0.7%) NEEDS_FA |  |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 1,28,36 | 0 | Mac 1L 0.79/10.00; Mac 28L 8.57/10.00; CUDA0 35L+H 11.29/11.40 | 129.2 (129.2) |  | HR<3%:CUDA0(0.9%) |  |
| 32B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 6,28,31 | 0 | Mac 6L 2.60/10.00; Mac 28L 9.74/10.00; CUDA0 30L+H 11.31/11.40 | 140.8 (140.8) |  | HR<3%:Mac(2.6%) HR<3%:CUDA0(0.8%) NEEDS_FA |  |
| 32B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 16,23,26 | 0 | Mac 16L 6.92/10.00; Mac 23L 9.52/10.00; CUDA0 25L+H 11.21/11.40 | 152.7 (152.7) |  | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 14,24,27 | 0 | Mac 14L 5.92/10.00; Mac 24L 9.80/10.00; CUDA0 26L+H 11.30/11.40 | 202.1 (202.1) |  | HR<3%:Mac(2.0%) HR<3%:CUDA0(0.9%) |  |
| 32B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,RPC1,CUDA0 | 20,21,24 | 0 | Mac 20L 9.19/10.00; Mac 21L 9.62/10.00; CUDA0 23L+H 11.21/11.40 | 211.9 (211.9) |  | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,RPC1,CUDA0 | 19,19,21 | 6 / 59 | CPU 6L 3.22/12.68; Mac 19L 9.94/10.00; Mac 19L 9.94/10.00; CUDA0 20L+H 11.16/11.40 | 225.7 (239.0) |  | HR<3%:Mac(0.6%) HR<3%:Mac(0.6%) HR<3%:CUDA0(2.1%) NEEDS_FA | 6 of 64 CPU layers (RAM) |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_disk | llama_rpc | RPC0,RPC1,CUDA0 | 19,18,19 | 25 / 56 | CPU 25L 0.67/12.62; Mac 19L 9.97/10.00; Mac 18L 9.49/10.00; CUDA0 18L+H 10.99/11.40 | 411.3 (484.9)* |  | HR<3%:Mac(0.3%) CPU>30%(25/80) GPU_DISK | capacity mode 25 of 80 layers paged |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_disk | llama_rpc | RPC0,RPC1,CUDA0 | 17,17,18 | 29 / 52 | CPU 29L 2.09/12.62; Mac 17L 9.81/10.00; Mac 17L 9.75/10.00; CUDA0 17L+H 11.31/11.40 | 418.8 (503.6)* |  | HR<3%:Mac(1.9%) HR<3%:Mac(2.5%) HR<3%:CUDA0(0.8%) CPU>30%(29/80) GPU_DISK NEEDS_FA | capacity mode 29 of 80 layers paged |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_disk | llama_rpc | RPC0,RPC1,CUDA0 | 15,15,16 | 35 / 46 | CPU 35L 4.66/12.62; Mac 15L 9.64/10.00; Mac 15L 9.64/10.00; CUDA0 15L+H 11.14/11.40 | 432.6 (534.5)* |  | HR<3%:CUDA0(2.3%) CPU>30%(35/80) GPU_DISK NEEDS_FA | capacity mode 35 of 80 layers paged |

#### S5 all_selected

| model | ctx | obj | fit | tier | engine | --device | -ts | cpuL / -ngl | per device est/budget GiB | decode ms prior (@18) | overflow options ms | flags | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | GPU 1L 0.62/31.00; CUDA0 23L+H 0.95/11.40 | 5.0 (5.0) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | GPU 1L 0.63/31.00; CUDA0 23L+H 1.08/11.40 | 5.0 (5.0) |  |  | fits alone, split only demonstrates participation |
| 0.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,24 | 0 | GPU 1L 0.64/31.00; CUDA0 23L+H 1.26/11.40 | 5.0 (5.0) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.65/31.00; CUDA0 27L+H 1.55/11.40 | 6.6 (6.6) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.66/31.00; CUDA0 27L+H 1.86/11.40 | 6.6 (6.6) |  |  | fits alone, split only demonstrates participation |
| 1.5B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.68/31.00; CUDA0 27L+H 2.29/11.40 | 6.6 (6.6) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | GPU 1L 0.68/31.00; CUDA0 35L+H 2.45/11.40 | 9.1 (9.1) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | GPU 1L 0.69/31.00; CUDA0 35L+H 2.86/11.40 | 9.1 (9.1) |  |  | fits alone, split only demonstrates participation |
| 3B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,36 | 0 | GPU 1L 0.70/31.00; CUDA0 35L+H 3.40/11.40 | 9.1 (9.1) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.80/31.00; CUDA0 27L+H 4.73/11.40 | 15.5 (15.5) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.82/31.00; CUDA0 27L+H 5.36/11.40 | 15.5 (15.5) |  |  | fits alone, split only demonstrates participation |
| 7B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.85/31.00; CUDA0 27L+H 6.21/11.40 | 15.5 (15.5) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.89/31.00; CUDA0 27L+H 7.57/11.40 | 23.8 (23.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.91/31.00; CUDA0 27L+H 8.20/11.40 | 23.8 (23.8) |  |  | fits alone, split only demonstrates participation |
| 7B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,28 | 0 | GPU 1L 0.94/31.00; CUDA0 27L+H 9.04/11.40 | 23.8 (23.8) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,48 | 0 | GPU 1L 0.85/31.00; CUDA0 47L+H 9.14/11.40 | 26.6 (26.6) |  |  | fits alone, split only demonstrates participation |
| 14B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 1,48 | 0 | GPU 1L 0.90/31.00; CUDA0 47L+H 11.34/11.40 | 26.6 (26.6) |  | HR<3%:CUDA0(0.5%) NEEDS_FA |  |
| 14B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 12,37 | 0 | GPU 12L 4.05/31.00; CUDA0 36L+H 11.19/11.40 | 26.6 (26.6) |  | HR<3%:CUDA0(1.8%) NEEDS_FA |  |
| 14B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 3,46 | 0 | GPU 3L 1.35/31.00; CUDA0 45L+H 11.38/11.40 | 34.3 (34.3) |  | HR<3%:CUDA0(0.2%) |  |
| 14B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 11,38 | 0 | GPU 11L 3.68/31.00; CUDA0 37L+H 11.30/11.40 | 34.3 (34.3) |  | HR<3%:CUDA0(0.9%) NEEDS_FA |  |
| 14B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 18,31 | 0 | GPU 18L 6.71/31.00; CUDA0 30L+H 11.27/11.40 | 34.3 (34.3) |  | HR<3%:CUDA0(1.2%) NEEDS_FA |  |
| 14B Q8_0 | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 14,35 | 0 | GPU 14L 4.71/31.00; CUDA0 34L+H 11.18/11.40 | 43.3 (43.3) |  | HR<3%:CUDA0(2.0%) |  |
| 14B Q8_0 | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 19,30 | 0 | GPU 19L 7.04/31.00; CUDA0 29L+H 11.09/11.40 | 43.3 (43.3) |  | HR<3%:CUDA0(2.7%) NEEDS_FA |  |
| 14B Q8_0 | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 23,26 | 0 | GPU 23L 9.81/31.00; CUDA0 25L+H 11.32/11.40 | 43.3 (43.3) |  | HR<3%:CUDA0(0.7%) NEEDS_FA |  |
| 32B Q4_K_M | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 29,36 | 0 | GPU 29L 9.07/31.00; CUDA0 35L+H 11.29/11.40 | 55.2 (55.2) |  | HR<3%:CUDA0(0.9%) |  |
| 32B Q4_K_M | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 34,31 | 0 | GPU 34L 12.05/31.00; CUDA0 30L+H 11.31/11.40 | 55.2 (55.2) |  | HR<3%:CUDA0(0.8%) NEEDS_FA |  |
| 32B Q4_K_M | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 39,26 | 0 | GPU 39L 16.15/31.00; CUDA0 25L+H 11.21/11.40 | 55.2 (55.2) |  | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 4096 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 38,27 | 0 | GPU 38L 15.43/31.00; CUDA0 26L+H 11.30/11.40 | 73.3 (73.3) |  | HR<3%:CUDA0(0.9%) |  |
| 32B Q6_K | 16384 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 41,24 | 0 | GPU 41L 18.51/31.00; CUDA0 23L+H 11.21/11.40 | 73.3 (73.3) |  | HR<3%:CUDA0(1.6%) NEEDS_FA |  |
| 32B Q6_K | 32768 | inter | Y est | gpu | llama_rpc | RPC0,CUDA0 | 44,21 | 0 | GPU 44L 22.57/31.00; CUDA0 20L+H 11.16/11.40 | 73.3 (73.3) |  | HR<3%:CUDA0(2.1%) NEEDS_FA |  |
| 72B Q4_K_M | 4096 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 60,19 | 2 / 79 | CPU 2L 1.36/12.62; GPU 60L 30.89/31.00; CUDA0 18L+H 10.99/11.40 | 129.4 (135.5) |  | HR<3%:GPU(0.3%) | 2 of 80 CPU layers (RAM) |
| 72B Q4_K_M | 16384 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 55,18 | 8 / 73 | CPU 8L 4.96/12.62; GPU 55L 30.73/31.00; CUDA0 17L+H 11.31/11.40 | 162.5 (187.2) |  | HR<3%:GPU(0.9%) HR<3%:CUDA0(0.8%) NEEDS_FA | 8 of 80 CPU layers (RAM) |
| 72B Q4_K_M | 32768 | inter | Y est | gpu_ram | llama_rpc | RPC0,CUDA0 | 49,16 | 16 / 65 | CPU 16L 10.39/12.62; GPU 49L 30.46/31.00; CUDA0 15L+H 11.14/11.40 | 204.2 (252.1) |  | HR<3%:GPU(1.7%) HR<3%:CUDA0(2.3%) NEEDS_FA | 16 of 80 CPU layers (RAM) |
