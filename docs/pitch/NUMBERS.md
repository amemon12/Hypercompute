# Hypercompute numbers sheet (Cal Hacks 13.0)

This is the one sheet the pitch drafts quote. Every value comes from a `run.json` listed in an index file under
`results/capacity/`, unless the row says it comes from paper text only. Every section points at the matching section
of `docs/research/PAPER-RUNS-2026-10-22.md` (PR). The headline ratios were re-run on 2026-10-24 (PDT) with the commands
at the end of this header. They agree with the paper except for the roundings and record differences in section 6,
where the run records win.

## How to read it

- **Ratio** is pooled / alone, the `B/A` of `stats.py`. For tok/s, above 1 means the pool is faster. For times,
  below 1 means the pool is sooner.
- **x sooner** is alone time / pooled time. It was bootstrapped with the two groups swapped, so it has its own
  interval.
- **95% CI** is a percentile bootstrap of the ratio of means, 10,000 resamples, seed 7. It resamples reps at N = 1
  and batches at N > 1. Small samples (2 vs 2, 3 vs 3 reps from one row per side) give tight intervals that show
  rep-to-rep spread inside one load, not run-to-run spread. Say "one pair of runs" when that is the case.
- An interval such as `[0.118, 0.118]` is not exact. It is the spread of 2 reps inside one row per side, which was
  under 0.2% for the 70B and 72B SSD reads. Quote it with "2 reps in one row per side".
- **n** is measured reps. Warmups are excluded.
- Tool output is printed as the tool gives it. Table values shortened from it (seconds from milliseconds, one or two
  decimals) are rounded normally. Anything spoken or pasted into a draft rounds toward the less impressive side and
  widens intervals outward. The "Say" lines and section 5 already do this.
- Common to every row: temperature 0, seed 7, `ignore_eos`, no prompt cache, fixed-length prompts. Decode is the
  engine's own (predicted_n - 1) / predicted_ms. SSD is system-wide physical reads per decoded token, in MiB.

### Condition keys

| Key | Host | Peers | Network | Section |
|---|---|---|---|---|
| **L7** | ROG Zephyrus G14 (RTX 5070 Ti Laptop GPU, 11.94 GiB VRAM (12 GB), 31.12 GiB RAM), Windows 11 | mac-a and mac-b (MacBook Air M4, 16 GiB, 10.5 GiB contribution cap each) | Home network. Both Macs direct over Iroh in every pooled row. RTT 5 to 8 ms in the snapshots before each pooled row. After it, 8 to 46 ms in the index `network_after` snapshots (8 to 42 ms in run.json `gateway_after`). Pooled rows wait until both peers are direct under 30 ms. | PR R7, 2026-10-25 UTC |
| **G6** | ASUS Ascent GX10 (GB10, 121.6 GiB unified memory) | G14 (CUDA, 9 GiB cap), mac-a and mac-b (11.5 GiB caps), mac-c (MacBook Air M4, 7.5 GiB cap) | Home Wi-Fi, one subnet, no wired link. 671B pooled rows: mac-c on the Iroh relay at 48 to 103 ms, the other three direct at 5 to 36 ms. 1T pooled rows: all four direct, 5 to 55 ms. Ranges cover the index and run.json snapshots before and after each pooled row. | PR R6, 2026-10-24 |
| **G3** | ASUS Ascent GX10 (GB10, 121.6 GiB unified memory) | G14 (CUDA0, 9 GiB cap), mac-a and mac-b (MTL0, 11.5 GiB caps). No mac-c. | Home network. All three peers direct over Iroh in every pooled row. RTT 5 to 16 ms in the index and run.json snapshots before each completed pooled row, up to 52 ms after it (13 to 21 ms before the failed 671B row). Every pooled row seeded its peers from Hugging Face (`--seed-peers`). | PR R8, 2026-10-25 UTC |
| **B1** | G14 as above | mac-c, mac-a, mac-b (7.5 GiB cap each). "1 Mac" is mac-c. | Building Wi-Fi, laptops on different subnets, Iroh direct through NAT. RTT 12 to 134 ms during setup and host-to-Mac 6.6 to 8 MiB/s (PR R1 setup text, not in run.json). The run.json snapshots of the pooled rows show 7 to 64 ms for the Macs in each pool (index snapshots 7 to 58). | PR R1, R2, 2026-10-22 to 23 |
| **G14A** | G14 alone | none | none | PR R4, 2026-10-23 |

"64/32" means a 64-token prompt and a 32-token answer requested.

### Reproduce the checks from this checkout

Run from the repository root after `uv sync`:

```sh
uv run --no-sync python bench/hcbench.py report --check
uv run --no-sync python scripts/capacity/paper/stats.py PREFIX_A PREFIX_B INDEX.jsonl
```

For concurrent runs, add `--batches` to the `stats.py` command. Replace the prefixes and index path with the
recorded pair you want to inspect. Index paths below are relative to `results/capacity/`.

The original audit also used local helpers named `ttft_stats.py`, `conc_ttft.py` and `dump_rows.py`. Those helpers
were not committed. Their first-token and answer calculations are implemented in the tracked benchmark reporter,
which reads the same records and checks this sheet's expected values. See [bench/README.md](../../bench/README.md)
for pair selection and report generation.

---

## 1. Headline laptop results (R7, laptops only, home network)

### 1a. Llama 3.3 70B Instruct (Q4_K_M, dense, about 40 GiB)

Conditions: L7, 64/32, 2 reps alone vs 2 reps pooled, one row per side, N = 1. Measured prompt_n 62 to 64.
Alone, the plan streams 61 of 80 layers from the file on the host CPU and keeps 19 on CUDA0. Pooled: 21 on the host
CPU, 20 on each Mac, 19 on CUDA0. Source for every row: PR R7.2, `paper-2026-10-24-laptop/index_d70.jsonl`.

| Metric | Alone | Pooled | Ratio pooled/alone [95% CI] | x sooner [95% CI] |
|---|---|---|---|---|
| Decode tok/s | 0.165 | 0.898 | 5.444 [5.423, 5.464] | |
| Prefill tok/s | 6.361 | 12.514 | 1.967 [1.896, 2.043] | |
| First token, s | 10.02 | 5.01 | 0.500 [0.480, 0.523] | 1.999 [1.913, 2.085] |
| 32-token answer, s | 197.91 | 39.53 | 0.200 [0.199, 0.201] | 5.007 [4.987, 5.027] |
| SSD MiB per token | 23454 | 2761 | 0.118 [0.118, 0.118], 2 reps in one row per side | |
| Load, s | 42.3 | 96.2 after 852 s of peer seeding | | |

Say: "On Llama 3.3 70B, the gaming laptop plus two MacBook Airs on a home network decoded at 0.898 tokens a second
against 0.165 alone, 5.44x [5.42, 5.47], one pair of runs with 2 reps each, 64-token prompt, 32-token answer.
Alone, the laptop rereads 23454 MiB (22.9 GiB) from its SSD per token. Pooled is still under 1 token a second, so the
32-token answer took 39.53 s (197.9 s alone)." In speech, say "about 5 times", not 5.44: see the caveat.

Caveats: only one interleaved pair. The second alone row (`d70-g14A-2`, run 20261025T014621Z, status
aborted_by_guard) was stopped by the commit guard after its warmups. Those decoded at 0.179 to 0.181 tok/s, about 9%
faster than row 1's measured 0.165 (row 1's own warmups were 0.148 to 0.156). Against 0.180 the pooled 0.898 would be
4.99x, outside the 5.444 interval, so 5.44x is one pair and likely the high end. The second pooled row was refused
(`index_d70_refused_layout_cpu_stage.jsonl`). A second d70 pair is still queued.

### 1b. gpt-oss-120b (MXFP4, 59.03 GiB MoE)

Conditions: L7, 128/64, 3 reps alone vs 3 reps pooled, one row per side, N = 1. Measured prompt_n 127 to 129.
Pooled layout: 6 layers on each Mac, 24 on CUDA0 (`--tensor-split 6,6,25`), the remaining experts file-backed on the
host. Source: PR R7.3, `paper-2026-10-24-laptop/index_g120.jsonl`.

| Metric | Alone | Pooled | Ratio pooled/alone [95% CI] | x sooner [95% CI] |
|---|---|---|---|---|
| Decode tok/s | 3.829 | 8.740 | 2.283 [2.000, 2.538] | |
| Prefill tok/s | 3.836 | 29.764 | 7.759 [6.945, 8.855] | |
| First token, s | 34.23 | 4.83 | 0.141 [0.120, 0.161] | 7.087 [6.204, 8.228] |
| 64-token answer, s | 49.88 | 11.67 | 0.234 [0.214, 0.262] | 4.276 [3.822, 4.671] |
| SSD MiB per token | 238.3 | 6.4 | 0.027 [0.017, 0.034] | |
| Load, s | 40.1 | 112.2 after 761 s of peer seeding | | |

Say: "On gpt-oss-120b, the gaming laptop plus two MacBook Airs on a home network (RTT 5 to 8 ms before the row) gave
the first token 7.08x sooner [6.20, 8.23], 34.22 s to 4.83 s, and decoded 2.28x faster [2.00, 2.54], 128-token
prompt, 64-token answer, one row per side with 3 reps each."

Caveat: R2 (section 4) ran the same model with three Macs on building Wi-Fi, 9 vs 9 reps, and got the first token
2.69x sooner [2.35, 3.08], prefill 2.99x [2.63, 3.35] and decode 1.27x [1.10, 1.47]. The 7.08x comes from the better
single pair. The two sessions differ in RTT, bandwidth and number of Macs, so the pair cannot say which one matters.

### 1c. Qwen2.5-VL 72B Instruct (Q4_K_M, dense, text only)

Conditions: L7, 64/32, 2 vs 2 reps, one row per side, N = 1. Measured prompt_n 63 to 66. Text only: the vision
projector is not loaded, and image input is not in this gateway. Alone: 64 of 80 layers streamed on the host CPU,
16 on CUDA0. Pooled: 28 on the host CPU, 18 on each Mac, 16 on CUDA0. Source: PR R7.4,
`paper-2026-10-24-laptop/index_vl72.jsonl`.

| Metric | Alone | Pooled | Ratio pooled/alone [95% CI] | x sooner [95% CI] |
|---|---|---|---|---|
| Decode tok/s | 0.141 | 0.365 | 2.592 [2.571, 2.613] | |
| Prefill tok/s | 4.642 | 7.221 | 1.555 [1.506, 1.608] | |
| First token, s | 13.70 | 9.10 | 0.664 [0.643, 0.686] | 1.506 [1.457, 1.555] |
| 32-token answer, s | 233.60 | 93.93 | 0.402 [0.398, 0.406] | 2.487 [2.463, 2.511] |
| SSD MiB per token | 26896 | 8956 | 0.333 [0.333, 0.333], 2 reps in one row per side | |
| Load, s | 34.1 | 100.2 after 847 s of peer seeding | | |

Say: "Qwen2.5-VL 72B, text only: the gaming laptop plus two MacBook Airs on a home network decoded at 0.365 tokens a
second against 0.141 alone, 2.59x [2.57, 2.62], 64-token prompt, 32-token answer, one pair of runs with 2 reps
each." Do not say it ran images.

### 1d. gpt-oss-120b with 1, 4 and 8 chats at once on the laptops (R7.5)

Conditions: L7, 128/64, N server slots with a unified KV cache (`-np N -kvu`), 3 batches per row, one alone row and
one pooled row per N. The pooled rows used the R7.3 layout (`--tensor-split 6,6,25`). Aggregate decode counts tokens
only in the window where all N chats generate. All 12 batches ended ok, with window coverage 0.9986 to 0.9994,
and no chat missed the first iteration. N = 1 is section 1b (3 reps). Sources: PR R7.5,
`paper-2026-10-24-laptop/index_g120n4.jsonl`, `index_g120n8.jsonl`, `index_g120.jsonl`.

| N | Metric | Alone | Pooled | Ratio pooled/alone [95% CI] |
|---|---|---|---|---|
| 1 | Aggregate decode tok/s | 3.829 | 8.740 | 2.283 [2.000, 2.538] |
| 1 | Step, ms | 261.638 | 115.550 | 0.442 [0.394, 0.499] |
| 1 | SSD MiB per step | 238.3 | 6.4 | 0.027 [0.017, 0.034] |
| 1 | First token, s | 34.23 | 4.83 | 0.141 [0.120, 0.161] |
| 4 | Aggregate decode tok/s | 8.095 | 11.854 | 1.464 [1.295, 1.674] |
| 4 | Step, ms | 494.299 | 340.621 | 0.689 [0.597, 0.769] |
| 4 | SSD MiB per step | 354.1 | 59.3 | 0.168 [0.135, 0.197] |
| 4 | First token per chat, s (12 chats per side) | 45.99 | 21.17 | 0.460 [0.390, 0.523] |
| 4 | 64-token answer per chat, s | 75.70 | 41.53 | 0.549 [0.469, 0.619] |
| 8 | Aggregate decode tok/s | 12.187 | 16.445 | 1.349 [1.246, 1.479] |
| 8 | Step, ms | 659.556 | 489.300 | 0.742 [0.677, 0.811] |
| 8 | SSD MiB per step | 438.0 | 75.0 | 0.171 [0.140, 0.201] |
| 8 | First token per chat, s (24 chats per side) | 59.80 | 30.80 | 0.515 [0.496, 0.535] |
| 8 | 64-token answer per chat, s | 99.62 | 59.77 | 0.600 [0.569, 0.633] |

First token and answer times at N > 1 come from `conc_ttft.py`. They include each chat's wait for the batch's prompts.
Pooled loads: 128.2 s after 80 s of seeding (N = 4), 122.2 s after 10 s (N = 8); alone 44.1 s and 40.1 s.

Claims check: "Serves many users" moved from the do-not-say list to the claims sheet on 2026-10-25, with these
numbers and conditions (laptops pooled; GX10 alone). The do-not-say item is now "Every user gets full speed". Say:

Say: "On gpt-oss-120b, laptops only (the G14 plus two Airs, home network), 128-token prompt, 64-token answer, one row
per side with 3 batches: with 8 chats at once the pool served 16.44 tokens a second in total against 12.19 alone,
1.34x [1.24, 1.48]. Each chat got about 2.05 tokens a second, slower than one chat on the laptop alone (3.83). The
pool's lead shrinks as chats are added: 2.28x at 1, 1.46x at 4, 1.34x at 8."

The lead shrinks as N grows: batching shares each expert read across N chats, which helps the SSD-bound host
alone more than the pool, whose every step includes a round trip to each Mac. Confound: the alone N = 4 row
overlapped about 30 s of an orphaned model download, during its load and before any measured batch.

---

## 2. Where pooling does not help

| Case | Metric | Alone | Pooled | Ratio pooled/alone [95% CI] | Conditions | Source |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B (Q3_K_M, MoE) | Decode tok/s | 21.373 | 10.005 | 0.468 [0.362, 0.611] | L7, 128/64, 6 alone reps (2 rows) vs 3 pooled reps (1 row), N = 1 | PR R7.1, `paper-2026-10-24-laptop/index_n80.jsonl` |
| same | Prefill tok/s | 84.348 | 29.513 | 0.350 [0.304, 0.400] | same | same |
| same | First token, s | 1.56 | 4.31 | 2.771 [2.447, 3.183] (pooled later) | same | same, ttft_stats.py |
| same | 64-token answer, s | 4.76 | 10.65 | 2.240 [1.869, 2.771] (pooled later) | same | same |
| same | SSD MiB per token | 9.974 | 5.398 | 0.541 [0.198, 1.383] | same | same |
| DeepSeek-V3.1 671B on the GX10 | Decode tok/s | 1.805 | 1.497 | 0.830 [0.769, 0.903] | G6, 64/32, 4 vs 4 reps (2 rows per side), 1 peer on relay, pooled reps mix two layouts | PR R6, `paper-2026-10-24/index_671b_gx.jsonl` |
| Kimi-K2.6 1T on the GX10 | Decode tok/s | 1.799 | 1.540 | 0.856 [0.818, 0.886] | G6, 64/32, 4 vs 4 reps, all peers direct | PR R6, `paper-2026-10-24/index_kimi_gx.jsonl` |
| Qwen3-235B, steady state, building Wi-Fi | Decode tok/s | 1.294 | 1.396 | 1.079 [0.976, 1.184], includes parity | B1, 3 Macs, 64/64, alone runs 3 to 5 and pooled runs 4 to 6 (`235b-3mac-3` infeasible), 9 vs 9 reps | PR R1, `paper-2026-10-22/index_235b.jsonl` + `index_235b_tight.jsonl` |
| gpt-oss-120b, 3 Macs vs 1 Mac | Decode tok/s | 4.130 (1 Mac) | 4.198 (3 Macs) | 1.016 [0.860, 1.199], includes parity | B1, 128/64, 6 vs 9 reps | PR R2, `paper-2026-10-22/index_120b.jsonl` |
| Pooled cold start, laptops | Load, s | 70B 42.3, 120B 40.1, 72B 34.1, 80B 22.0 and 18.0 | 70B 96.2 after 852 s of seeding, 120B 112.2 after 761 s, 72B 100.2 after 847 s, 80B 180.4 after 5 s (cache warmed by an aborted row's 691 s of seeding) | | L7, one pooled row per model | PR R7.1 to R7.4, run.json `measured.seed.seconds`, `measured.load.seconds` |
| Pooled cold start, GX10 | Load, s | 671B 239, 256; 1T 257, 266 | 671B 503 after 1416 s of seeding, 517 after 856 s; 1T 367 after 676 s, 369 after 5 s (warm cache) | | G6 | PR R6, same index files |
| Client-isolated apartment Wi-Fi | Path and outcome | | Relay only, host-to-Mac about 1 MiB/s, RTT 60 to 580 ms, a peer dropped mid-load. The 235B C1 run and four Mac runs failed or were refused. | | G14 on guest Wi-Fi, Macs on a phone hotspot, 2026-10-23 | PR R5 (paper text, no index) |
| Planner on the 671B, GX10 | Predicted vs measured | predicted pooled 1.77x ahead | measured 0.830x [0.769, 0.903] | | G6, plan.moe comparison block | PR R6, `index_671b_gx.jsonl` |
| Planner on the 1T, GX10 | Predicted vs measured | predicted pooled about 1.44x ahead (mean 2158 vs 1494 ms per token) | measured 0.856x [0.818, 0.886] | | G6, plan.moe comparison block (measured_over_predicted 0.254, 0.261, 0.426, 0.444) | PR R6, `index_kimi_gx.jsonl` |
| Planner on the 123B dense, GX10 alone | Plan vs outcome | plan marked feasible (estimated_fit, tier gpu_disk) although its own capacity optimizer reported "no feasible layout ... the planner's plan is kept" | load_failed after 2465 s, second row killed at 3031 s, box unreachable over SSH for about 50 minutes | | GX10 alone, 64/32 | PR R6.x, `paper-2026-10-24-gx10conc/index_gx10_ml123.jsonl` |

Reasons, one line each:

- **80B MoE.** Only 3B parameters are active per token and its experts stay in the G14's page cache (3.0 to 26.2 MiB
  of SSD reads per token per alone rep, mean 9.97 over 6 reps), so the Macs have no disk traffic to remove and each
  one adds a blocking round trip per token.
- **GX10 decode.** At batch size 1 each peer is one blocking network round trip per token, and on the GX10 the SSD
  reads the peers save cost less than four Wi-Fi stages.
- **235B on building Wi-Fi, and more Macs on the 120B.** Same cause: once the host's page cache is warm, extra
  stages cost about what they save.
- **Cold start.** Every pooled row seeds its peers before it loads, so a cold pooled start took about 15 to 16
  minutes on the laptops (873 to 948 s) and 17 to 32 minutes on the GX10 (1043 to 1919 s), against 34 to 42 s alone
  on the laptops and 239 to 266 s on the GX10.
- **Planner.** Its speed constants come from the G14 and overstate SSD cost on the GX10, so it ranked the pool ahead
  on both GX10 models. On the 123B it approved a plan its own optimizer had found infeasible.

The 80B alone rows vary a lot (14.75 to 34.47 tok/s per rep). The pooled row sits below every alone rep.

Aborted and refused R7 rows (kept in their own index files, never in a ratio):

- **80B.** The first four n80 rows were stopped by the commit guard at 0.25 GiB of allowed growth
  (`index_n80_aborted_commit_guard.jsonl`). The second pooled row of the interleave was refused because its pinned
  layout named RPC0 instead of MTL0 (`index_n80_refused_layout_device.jsonl`). That is why the 80B is 2 alone rows
  against 1 pooled row.
- **32B (d32).** Alone was refused as infeasible under `--no-capacity-mode`
  (`index_d32_refused_no_capacity_alone.jsonl`), then alone and pooled were stopped by the commit guard
  (`index_d32_aborted_commit_guard.jsonl`). d32 is still in the queue after failing twice.
- **70B.** Two pooled rows were refused as infeasible under `--no-capacity-mode` (`index_d70_refused_gpu_only.jsonl`).
  The second pooled row of the interleave was refused because a layout cannot name a host CPU stage
  (`index_d70_refused_layout_cpu_stage.jsonl`). The second alone row was stopped by the commit guard after its warmups
  (section 1a).

---

## 3. GX10 numbers (PR R6, R8)

### 3a. gpt-oss-120b on the GX10 alone

Conditions: GX10 alone, no peers, tier gpu (all 36 layers on CUDA0), 128/64, 1 warmup and 3 reps per row, 3 rows,
9 reps. Confound: the 671B download ran on the GX10 during these rows (about 30 MiB/s, read by hand). Source: PR R6,
`paper-2026-10-24/index_120b_gxA.jsonl`.

| Metric | Value |
|---|---|
| Decode tok/s | mean 56.00, median 56.17, per rep 55.21 to 56.37 |
| Prefill tok/s | mean 382.05 |
| First token, s | 0.42 (417.5 ms mean) |
| 64-token answer, s | 1.49 (1485.5 ms mean) |
| SSD MiB per token | 0.021 mean, median 0.002 (9 reps), effectively none |

This is a host change, not pooling. Do-not-say: the ULTRAPLAN list bans "56 tok/s on the 120B" outright ("Do not use
it"). Internal context only. Against the G14 alone in R2 (3.296 decode, 3.682 prefill) it is 16.99x decode and
103.76x prefill, a cross-session host swap with no interval. Do not pitch those ratios either.

### 3b. 671B and 1T: GX10 alone vs GX10 + G14 + 3 Macs, and the G14 alone

Conditions: G6, 64/32, 1 warmup and 2 reps per row, order A P A P, 4 vs 4 reps, N = 1, tier moe_offload. Measured
prompt_n 63 to 66 (671B), 58 to 66 (1T). Every pooled row used `--seed-peers`.

| Model | Metric | Alone | Pooled | Ratio pooled/alone [95% CI] | x sooner [95% CI] | Source |
|---|---|---|---|---|---|---|
| DeepSeek-V3.1 671B (UD-TQ1_0, 158.8 GiB) | Decode tok/s | 1.805 | 1.497 | 0.830 [0.769, 0.903] | | PR R6, `paper-2026-10-24/index_671b_gx.jsonl` |
| | Prefill tok/s | 2.631 | 6.362 | 2.418 [2.194, 2.721] | | same |
| | First token, s | 24.67 | 10.36 | 0.420 [0.373, 0.462] | 2.381 [2.159, 2.682] | same, ttft_stats.py |
| | 32-token answer, s | 41.84 | 31.20 | 0.746 [0.694, 0.797] | 1.341 [1.253, 1.439] | same |
| | SSD MiB per token | 683.6 | 37.1 | 0.054 [0.026, 0.097] | | same |
| | Load, s | 239, 256 | 503 after 1416 s seeding, 517 after 856 s seeding | | | same |
| Kimi-K2.6 1T (IQ1_S, 193.16 GiB) | Decode tok/s | 1.799 | 1.540 | 0.856 [0.818, 0.886] | | PR R6, `paper-2026-10-24/index_kimi_gx.jsonl` |
| | Prefill tok/s | 2.188 | 3.298 | 1.507 [1.412, 1.587] | | same |
| | First token, s | 28.25 | 19.08 | 0.675 [0.629, 0.728] | 1.481 [1.374, 1.592] | same, ttft_stats.py |
| | 32-token answer, s | 45.48 | 39.23 | 0.863 [0.818, 0.912] | 1.159 [1.097, 1.224] | same |
| | SSD MiB per token | 671.0 | 254.1 | 0.379 [0.365, 0.397] | | same |
| | Load, s | 257, 266 | 367 after 676 s seeding, 369 after 5 s seeding (warm cache) | | | same |

The 671B on the G14 alone (G14A, 64/32, 1 row, 1 warmup, 2 reps, tier moe_offload; PR R4,
`paper-2026-10-23/index_671b.jsonl`): decode 0.481 tok/s, prefill 0.747 tok/s, first token 88.68 s, 32-token
answer 153.13 s, SSD 2484.9 MiB per token. The GX10 alone decodes it 3.75 times faster (1.805 vs 0.481). That is a
host change, cross-session, with no interval: do not pitch it.

Say: "Host: the GX10, a rented box, with the G14 and three MacBook Airs as peers on home Wi-Fi, 64-token prompt,
32-token answer, 4 vs 4 reps from 2 rows per side. On the 671B, the pool read the prompt 2.41x faster [2.19, 2.73]
and gave the first token 2.38x sooner [2.15, 2.69], 24.67 s to 10.37 s, but decoded at 0.83x [0.76, 0.91]. One peer
was on the relay, and the pooled reps mix two layouts. On the 1T: prompt 1.50x faster [1.41, 1.59], first token
1.48x sooner [1.37, 1.60], decode 0.85x [0.81, 0.89]." The laptops-only pool never ran the 671B or the 1T.

### 3c. Many chats at once on the GX10 alone (R6.y)

Conditions: GX10 alone, no peers, `-np N -kvu`. 120B: 128/64, rows interleaved n1 n4 n8 n8 n4 n1, 3 batches per
row, so 6 batches per N. 1T and 671B: 64/32, one row per N, 2 batches per row. Ratios are N over N = 1 and
resample batches. Pooled concurrency on the GX10 was not measured in R6.y; R8 measured it (section 3e). Sources: PR R6.y,
`paper-2026-10-24-gx10conc/index_g120_conc.jsonl`, `paper-2026-10-24-gx10conc/index_gx10_conc2.jsonl`.

| Model | Metric | N = 1 | N = 4 [95% CI vs N = 1] | N = 8 [95% CI vs N = 1] |
|---|---|---|---|---|
| gpt-oss-120b | Aggregate decode tok/s | 54.621 | 111.861, 2.048x [1.973, 2.140] | 149.156, 2.731x [2.583, 2.901] |
| | Step, ms | 18.31 | 35.85 | 54.12, 2.956x [2.781, 3.112] |
| | First token per chat, s | 0.42 | 0.69 | 1.00 |
| | Full answer per chat, s | 1.52 | | 4.30, 2.833x [2.703, 2.950] |
| | SSD MiB per aggregate token | 0 | 0 | 0 |
| Kimi-K2.6 1T | Aggregate decode tok/s | 1.779 | 3.250, 1.826x [1.633, 2.021] | 4.640, 2.608x [2.372, 2.844] |
| | Step, ms | 562.0 | 1244.5, 2.214x [1.980, 2.450] | 1745.0, 3.105x [2.825, 3.387] |
| | First token per chat, s | 27.01 | 45.21 | 66.90 |
| | Full answer per chat, s | 44.42 | | 119.34, 2.686x [2.579, 2.794] |
| | SSD MiB per aggregate token | 657.3 | 359.7 | 235.5 |
| DeepSeek-V3.1 671B | Aggregate decode tok/s | 1.792 | 3.444, 1.922x [1.875, 1.970] | 5.547, 3.096x [2.525, 3.686] |
| | Step, ms | 558.2 | 1171.3, 2.098x [2.047, 2.151] | 1497.9, 2.683x [2.189, 3.195] |
| | First token per chat, s | 23.49 | 37.70 | 47.63 |
| | Full answer per chat, s | 40.77 | | 92.66, 2.273x [2.026, 2.532] |
| | SSD MiB per aggregate token | 686.3 | 318.5 | 186.4 |

Full answer per chat comes from `conc_ttft.py` (N = 4 not computed).

Say: "The GX10 alone, no peers, 64-token prompt, 32-token answer, one row per N with 2 batches each: eight chats of
the 671B at 5.54 tokens a second in total against 1.80 for one chat, 3.09x [2.52, 3.69]. Each chat is slower: a
decode step takes 2.69x as long [2.18, 3.20], and the full answer 92.66 s against 40.77 s. A Nemotron 30B service was
installed on the rented unit." Keep it off pooling slides. Never say this about the pool on the GX10.

Confound: a Nemotron 30B snap server on the rented unit held 30.5 GiB of GPU memory after a restart at 01:25 UTC.
The conc2 rows wait until no GPU compute process remains before their first row.

### 3d. Mistral Large 2411 123B dense (Q8_0, 121.33 GiB) did not load on the GX10 alone (R6.x)

Conditions: GX10 alone, 64/32, 2 reps requested, planner tier gpu_disk (56 layers on CUDA0, 32 file-backed on the
CPU). Source: PR R6.x, `paper-2026-10-24-gx10conc/index_gx10_ml123.jsonl`.

| Row | Result |
|---|---|
| ml123-gxA-1 | `load_failed` after 2465 s (run 20261024T235320Z). The plan was marked feasible (`estimated_fit`), although its warnings say "capacity optimizer: no feasible layout ... the planner's plan is kept". |
| ml123-gxA-2 | killed (exit 137) at 3031 s, no run.json |
| Side effect | From about 00:35 to 01:25 UTC the unit answered ping but not SSH, load average 120, 121 of 121 GiB in use. The gateway was gone after, so the six concurrency rows queued behind these failed at once (`index_gx10_chain_failed_gateway_down.jsonl`). |
| Confound | The rental's Nemotron 30B snap server was restarted at 01:25 UTC and then held 30.5 GiB of GPU memory. It may have held memory during these rows; nothing in run.json says. |

The model is 121.33 GiB against the GX10's 121.6 GiB, so it is about the size of the whole memory, not bigger.

Say: "A 121.33 GiB dense model, about the size of the GX10's whole 121.6 GiB memory, did not load on it alone. Our
planner approved a plan its own optimizer had found infeasible (gpu_disk, 32 layers paged from disk), and the box
stopped answering SSH for about 50 minutes. A Nemotron 30B service on the rented unit may have held memory." It was
not tried pooled, so do not say the pool fixes it.

### 3e. Every machine at once (GX10 host + G14 + mac-a + mac-b), 2026-10-25

Conditions: G3. The GX10 alone (no peers in the plan) against the GX10 with the G14, mac-a and mac-b as peers, one
row per side per N, tier moe_offload. N = 1 resamples reps; N > 1 (`-np N -kvu`) resamples batches. Maverick: 128/64,
`--warmup 1` (the pooled N = 1 row ran one extra prompt-sizing warmup; N > 1 rows also ran one unmeasured warm
batch) and 3 reps or 3 batches per row, measured prompt_n 124 to 130. 1T: 64/32, 2 reps or 2 batches per row,
measured prompt_n at N = 1 58 to 59 alone and 62 to 64 pooled. The 1T alone rows are R6.y's `kimi-n1-gxA-1` and
`kimi-n8-gxA-1` (section 3c), run on the same GX10 about 4 h before the pooled rows, not interleaved. The pooled
layout changed from row to row. Maverick N = 1: layers 0 to 2 on the G14, 3 to 4 on mac-a, 5 to 6 on mac-b, 7 to 47
on CUDA0, 46.1 GiB of expert tensors on the host CPU. 1T N = 1: 0 to 1 on mac-b, 2 to 3 on the G14, 4 to 6 on mac-a,
7 to 60 on CUDA0. The peers joined during the Maverick alone N = 4 row (none at its start, all three at its end) and stayed connected
through the N = 8 row; they held no layers in either. Sources: PR R8, `paper-2026-10-25-gx10sweep/index_mav.jsonl` (Maverick alone),
`paper-2026-10-25-gx10sweep/index_allpool.jsonl` (pooled), `paper-2026-10-24-gx10conc/index_gx10_conc2.jsonl` (1T
alone). bench pairs `mav-g3`, `mav-g3-n4`, `mav-g3-n8`, `kimi-g3`, `kimi-g3-n8`.

Llama 4 Maverick 400B (UD-Q2_K_XL, 142.17 GiB per PR R8.1, 17B active), N = 1:

| Metric | GX10 alone | All machines | Ratio pooled/alone [95% CI] | x sooner [95% CI] |
|---|---|---|---|---|
| Decode tok/s | 9.785 | 4.277 | 0.437 [0.400, 0.477] | |
| Prefill tok/s | 60.440 | 32.844 | 0.543 [0.353, 0.791] | |
| First token, s | 2.41 | 3.94 | 1.637 [1.264, 2.852] | 0.611 [0.351, 0.791] |
| 64-token answer, s | 8.86 | 18.71 | 2.111 [1.956, 2.280] | 0.474 [0.439, 0.511] |
| SSD MiB per token | 53.1 | 13.8 | 0.260 [0.166, 0.448] | |
| Load, s | 198.8 | 276.5 after 405 s of peer seeding | | |

Maverick, N chats at once:

| N | Metric | GX10 alone | All machines | Ratio pooled/alone [95% CI] |
|---|---|---|---|---|
| 4 | Aggregate decode tok/s | 29.859 | 9.826 | 0.329 [0.314, 0.348] |
| 4 | Step, ms | 134.372 | 407.790 | 3.035 [2.879, 3.184] |
| 4 | SSD MiB per step | 69.8 | 25.0 | 0.358 [0.246, 0.538] |
| 4 | First token per chat, s (12 chats per side) | 4.51 | 10.70 | 2.374 [1.884, 2.940] |
| 4 | 64-token answer per chat, s | 12.90 | 36.39 | 2.821 [2.607, 3.046] |
| 4 | Load, s | 210.4 | 242.4 after 10 s of peer seeding | |
| 8 | Aggregate decode tok/s | 43.362 | 15.071 | 0.348 [0.329, 0.370] |
| 8 | Step, ms | 185.158 | 533.290 | 2.880 [2.713, 3.044] |
| 8 | SSD MiB per step | 94.9 | 40.9 | 0.431 [0.143, 0.923] |
| 8 | First token per chat, s (24 chats per side) | 8.02 | 23.12 | 2.881 [2.352, 3.464] |
| 8 | 64-token answer per chat, s | 19.53 | 57.01 | 2.918 [2.569, 3.250] |
| 8 | Load, s | 214.4 | 254.4 after 10 s of peer seeding | |

Kimi-K2.6 1T (IQ1_S, 193.16 GiB, 32B active):

| N | Metric | GX10 alone | All machines | Ratio pooled/alone [95% CI] | x sooner [95% CI] |
|---|---|---|---|---|---|
| 1 | Decode tok/s | 1.779 | 1.774 | 0.997 [0.964, 1.030] | |
| 1 | Prefill tok/s | 2.178 | 3.132 | 1.438 [1.396, 1.480] | |
| 1 | First token, s | 27.01 | 20.22 | 0.748 [0.720, 0.777] | 1.336 [1.287, 1.389] |
| 1 | 32-token answer, s | 44.42 | 37.70 | 0.849 [0.819, 0.878] | 1.178 [1.139, 1.220] |
| 1 | SSD MiB per token | 657.3 | 333.8 | 0.508 [0.423, 0.598] | |
| 1 | Load, s | 274.5 | 308.8 after 255 s of peer seeding | | |
| 8 | Aggregate decode tok/s | 4.640 | 4.986 | 1.075 [0.975, 1.194] | |
| 8 | Step, ms | 1745.038 | 1615.040 | 0.926 [0.838, 1.031] | |
| 8 | SSD MiB per step | 1891.7 | 787.9 | 0.417 [0.331, 0.526] | |
| 8 | First token per chat, s (16 chats per side) | 66.90 | 60.66 | 0.907 [0.902, 0.911] | |
| 8 | 32-token answer per chat, s | 119.34 | 117.22 | 0.982 [0.873, 1.101] | |
| 8 | Load, s | 268.5 | 334.6 after 225 s of peer seeding | | |

First token and answer come from `ttft_stats.py` (N = 1) and `conc_ttft.py` (N > 1), with the two indexes of each
pair concatenated into one file for `conc_ttft.py`. "x sooner" is the swapped bootstrap; below 1 means later.

Batches: every N > 1 batch ended ok, window coverage 0.967 to 0.999. Chats that missed the server's first iteration
(their prompt ran in a later mixed batch), per batch: Maverick N = 4 alone 2, 2, 0 and pooled 2, 3, 3; N = 8 alone 0,
7, 0 and pooled 6, 0, 6; 1T N = 8 alone 0, 0 and pooled 5, 0. In the pooled 1T batch with 5 late chats the first
tokens spread from 33.0 s to 77.7 s, so the tight [0.902, 0.911] is two batch means per side, not a spread of chats.

DeepSeek-V3.1 671B (UD-TQ1_0), 64/32, 2 reps per side. The first pooled row `ds671-n1-gx3-1` is `load_failed`
(1236 s of seeding, load failed at 1077.4 s; run.json `stop.error` "peer d4e65f83 lost: worker exited (exit code -6)",
d4e65f83 is mac-b). The retry `ds671-n1-gx3-2` with both Macs at 10 GiB caps completed (seed 10.2 s, load 300.9 s;
layers G14 6, mac-a 3, mac-b 3, CUDA0 49). Against R6.y's `ds671-n1-gxA-1` (GX10 alone, about 5 h earlier, another
gateway session, not interleaved), the apparent speed gain was confounded by the separate sessions. The
interleaved comparison below supersedes that claim. The original records remain available for auditing. Sources: PR R8.3,
`paper-2026-10-25-gx10sweep/index_allpool.jsonl`, `paper-2026-10-24-gx10conc/index_gx10_conc2.jsonl`.

Interleaved check, same session, bench kit, A P A P, 2 reps per row, 4 vs 4 (`paper-2026-10-25-e2e/index_ds671e.jsonl`):
decode 1.761 vs 1.771 tok/s, 1.006 [0.926, 1.081]; prefill 2.654 vs 4.206, 1.585 [1.471, 1.706]; first token 25.02 vs
15.77 s, 0.630 [0.579, 0.681]; 32-token answer 42.63 vs 33.37 s, 0.783 [0.745, 0.844]; SSD 698.3 vs 260.6 MiB per
token, 0.373 [0.355, 0.393]. The earlier cross-session decode gain was not reproduced and is retracted.

Say (671B): "DeepSeek's 671B across the GX10, the gaming laptop and two MacBook Airs, home network, alone and pooled runs
alternating in one session: decode is even, 1.00 times [0.92, 1.09], and the first token comes 1.58 times sooner,
25.0 seconds down to 15.8."

Caveats. One row per side. The 1T alone rows ran 4 h earlier in another gateway session; the engine device line in
`gateway_before`, identical for every row of a session, shows 87.0 GiB of CUDA0 free in that session against 116.9
GiB in this one. Section 3c says the conc2 rows waited until no GPU compute process remained, so this may be a
startup snapshot only, but treat it as a possible memory confound for the 1T SSD and first-token ratios. In R6
(section 3b) the 1T pooled decode was 0.856 [0.818, 0.886] with four direct peers including mac-c; this session
had three. Cross-session, so do not pin the change on one cause. PR R8.1's explanation for Maverick (17B active, hot
expert pages cached) is a hypothesis; run.json only shows 53.1 MiB of SSD reads per token alone.

Say (Maverick): "Llama 4 Maverick 400B, 142 GiB with 17B active, is fastest on the GX10 alone: 9.79 tokens a second,
128-token prompt, 64-token answer, one row with 3 reps. Pooling made it slower. With the G14 and two MacBook Airs
added on the home network it decoded 4.27 tokens a second, 0.43x [0.40, 0.48], and the full answer took 2.12 times as
long [1.95, 2.29], 18.72 s against 8.86 s. With 8 chats at once the GX10 alone served 43.4 tokens a second in total
against 15.07 pooled, 0.34x [0.32, 0.37]." Run this model on the GX10 alone. Never pitch it as a pooling result.

Say (1T): "Kimi K2.6, a 1T model of 193 GiB, ran across every machine: the GX10 host with the G14 and two MacBook
Airs as peers, all direct on the home network, 64-token prompt, 32-token answer, one row per side with 2 reps. Decode
was at parity, 0.99x [0.96, 1.03], and the first token came 25% sooner [22%, 28%], 27.01 s to 20.22 s, because the
prompt read 1.43x faster [1.39, 1.48]. The full answer came 15% sooner [12%, 19%]." At 8 chats say only that
aggregate decode was 1.07x [0.97, 1.20], an interval that includes no gain. Mention that the alone row came 4 h
earlier.

### 3f. More models on the GX10 alone, 1, 4 and 8 chats (R8.1, R8.4)

Conditions: GX10 alone, no peers in the plan, 128/64, one row per N, 1 warmup and 3 reps at N = 1, 3 batches at
N = 4 and 8 (`-np N -kvu`). Ratios are N over N = 1 and resample batches. Maverick: tier moe_offload, 48 layers on
CUDA0 and 69.1 GiB of expert tensors on the host CPU. Gemma 3 27B IT and QwQ 32B (both Q4_K_M): tier gpu, the whole
model on CUDA0. Measured prompt_n 125 to 129 for both small models. Sources: PR R8.1, R8.4,
`paper-2026-10-25-gx10sweep/index_mav.jsonl`, `index_gm27.jsonl`, `index_qwq32.jsonl`. bench groups `gx10-gm27`,
`gx10-qwq32` (N = 1).

| Model | Metric | N = 1 | N = 4 [95% CI vs N = 1] | N = 8 [95% CI vs N = 1] |
|---|---|---|---|---|
| Llama 4 Maverick 400B (UD-Q2_K_XL) | Aggregate decode tok/s | 9.785 | 29.859, 3.051x [2.824, 3.300] | 43.362, 4.431x [4.087, 4.809] |
| | Step, ms | 102.515 | 134.372, 1.311x [1.213, 1.418] | 185.158, 1.806x [1.666, 1.959] |
| | Prefill tok/s | 60.440 | | |
| | First token per chat, s | 2.41 | 4.51 | 8.02 |
| | Full answer per chat, s | 8.86 | 12.90 | 19.53 |
| | SSD MiB per step | 53.1 | 69.8 | 94.9 |
| | Load, s | 198.8 | 210.4 | 214.4 |
| Gemma 3 27B IT (Q4_K_M) | Aggregate decode tok/s | 12.341 | 45.282, 3.669x [3.648, 3.685] | 85.044, 6.891x [6.883, 6.903] |
| | Step, ms | 81.029 | 88.341, 1.090x [1.081, 1.101] | 94.195, 1.162x [1.159, 1.165] |
| | Prefill tok/s | 396.670 | | |
| | First token per chat, s | 0.35 | 1.07 | 2.00 |
| | Full answer per chat, s | 5.46 | 6.61 | 7.90 |
| | Load, s | 14.1 | 4.0 | 4.0 |
| QwQ 32B (Q4_K_M, reasoning) | Aggregate decode tok/s | 10.643 | 39.125, 3.676x [3.658, 3.688] | 75.169, 7.063x [7.043, 7.077] |
| | Step, ms | 93.960 | 102.511, 1.091x [1.089, 1.093] | 106.928, 1.138x [1.136, 1.141] |
| | Prefill tok/s | 621.054 | | |
| | First token per chat, s | 0.23 | 0.89 | 1.77 |
| | Full answer per chat, s | 6.15 | 7.27 | 8.43 |
| | Load, s | 4.0 | 4.0 | 4.0 |

N = 1 is 3 reps, each N > 1 value 3 batches. The small models read no SSD to speak of (0.000 MiB per token at N = 1;
`stats.py` stops on the zero mean at N > 1). First token and full answer per chat come from `conc_ttft.py`. The
engine device line in `gateway_before` shows 103.9 GiB of CUDA0 free in the session of the small models and 116.9 GiB
in the Maverick session. QwQ is a reasoning model, but with `ignore_eos` and 64 output tokens the speed does not
depend on what it says.

Host numbers, not pooling. Maverick at N = 1 is the alone side of section 3e. Keep these off pooling slides.

### 3g. Kimi-K2.6 coherence spot check (R6)

Kimi-K2.6 1T IQ1_S, GX10 alone, temperature 0, thinking off, one prompt (an iterative Fibonacci function), run
20261024T215444Z (status ok, 1 rep). PR R6 says the answer was a correct, documented function with input validation.
The output text was not archived (run.json keeps only `content_chars`; ULTRAPLAN risk list). Quote it as a single spot
check, not a quality result. Used in DEMO V7 and Q7 and README-NEW.

---

## 4. Earlier laptop results (R1 and R2, building Wi-Fi)

Conditions: B1. Source for R2 rows: PR R2, `paper-2026-10-22/index_120b.jsonl`, paper set (the warm-up run
`120b-local-w` excluded; pass `--only-labels` to `stats.py` to match). Source for R1 rows: PR R1,
`paper-2026-10-22/index_235b.jsonl` + `index_235b_tight.jsonl`.

| Model, pool | Metric | Alone | Pooled | Ratio pooled/alone [95% CI] | Conditions |
|---|---|---|---|---|---|
| gpt-oss-120b, G14 + 3 Macs | Decode tok/s | 3.296 | 4.198 | 1.274 [1.105, 1.464] | 128/64, 9 vs 9 reps, N = 1 |
| | Prefill tok/s | 3.682 | 11.026 | 2.994 [2.639, 3.349] | same |
| | First token, s | 36.17 | 13.44 | 0.371 [0.325, 0.424]; 2.692x sooner, [2.35, 3.08] inverted | same |
| | 64-token answer, s | 54.44 | 27.93 | 0.513 [0.453, 0.577]; 1.949x sooner, [1.73, 2.21] inverted | same |
| | SSD MiB per token | 275.0 | 41.4 | 0.151 [0.117, 0.188] | same |
| gpt-oss-120b, G14 + 1 Mac | Decode tok/s | 3.296 | 4.130 | 1.253 [1.106, 1.407] | 128/64, 9 vs 6 reps |
| | Prefill tok/s | 3.682 | 5.454 | 1.481 [1.408, 1.567] | same |
| | First token, s | 36.17 | 23.97 | 0.663 [0.627, 0.696] | same |
| | 64-token answer, s | 54.44 | 38.81 | 0.713 [0.678, 0.750] | same |
| | SSD MiB per token | 275.0 | 147.6 | 0.537 [0.516, 0.558] | same |
| Qwen3-235B-A22B (UD-Q2_K_XL, 82.67 GiB), G14 + 3 Macs, steady state (alone runs 3 to 5, pooled runs 4 to 6; `235b-3mac-3` infeasible) | Decode tok/s | 1.294 | 1.396 | 1.079 [0.976, 1.184] | 64/64, 9 vs 9 reps |
| | Prefill tok/s | 1.450 | 2.333 | 1.609 [1.481, 1.742] | same |
| | First token, s | 43.50 | 27.86 | 0.641 [0.595, 0.695] | same |
| | 64-token answer, s | 92.91 | 73.25 | 0.788 [0.739, 0.845] | same |
| | SSD MiB per token | 646.8 | 295.8 | 0.457 [0.423, 0.489] | same |
| same, all runs | Decode tok/s | 1.194 | 1.337 | 1.119 [1.000, 1.258]: lower bound at parity, mixes 32- and 64-token answers. Quote the steady-state 1.079 [0.976, 1.184] instead. | 64/32 (runs 1 and 2) and 64/64, 13 vs 13 reps |
| | Prefill tok/s | 1.431 | 2.300 | 1.607 [1.509, 1.713] | same |
| | SSD MiB per token | 737.6 | 337.9 | 0.458 [0.391, 0.536] | same |

The "x sooner" intervals for the R2 rows are the pooled/alone intervals inverted (1/0.424, 1/0.325 and 1/0.577,
1/0.453), widened outward. The swapped `ttft_stats.py` run cannot drop the warm-up row from the command line.

Confound (R2): a code workflow shared the G14 from about 00:44 UTC, during six of the eight measured 120B runs. The
third 3-Mac run is likely pessimistic. Seeding the first 3-Mac cold start took 566 s.

---

## 5. Choosing numbers for the demo

Use the measured tables and their conditions above. Earlier pitch drafts contained pending placeholders and an
editing checklist; those are superseded by this sheet. Do not quote an unmeasured row or mix different sessions.

- Laptop example: section 1b, gpt-oss-120b on the G14 with two MacBook Airs. Quote the first-token and decode
  comparisons together with the prompt length, output length and sample count.
- Large pooled example: section 3e, DeepSeek 671B on the GX10 with the G14 and two MacBook Airs. The interleaved
  check found decode parity and an earlier first token. It supersedes the earlier cross-session decode claim.
- Network limit: section 3b records slower pooled decode in R6. The 671B pool had one relayed peer; the Kimi pool
  had four direct peers. These are separate conditions from R8, so they do not isolate the cause of the difference.
- GX10-only example: section 3e, Maverick 400B. Label it as a single-host result. Its pooled runs were slower.
- Concurrent requests: sections 1d, 3c, 3e and 3f. Aggregate throughput is not the speed of one chat.

The historical `ttft_stats.py` and `conc_ttft.py` names in the source notes identify how those tables were first
calculated. Use the tracked benchmark reporter for a fresh reproduction.

## 6. Flags: rounding and record differences

Rounded the favorable way, or interval narrowed, in the drafts or the paper. Use the right-hand form. This is a spot
check, not an exhaustive audit.

| Draft or paper says | Tool value | Say instead | Where |
|---|---|---|---|
| 671B first token "2.4x sooner" | 2.381 [2.159, 2.682] | 2.38x sooner [2.15, 2.69] | ULTRAPLAN one-liner, claims sheet; DEMO 48, 249, Q1 |
| 671B first token "24.7 s vs 10.4 s" | 24672 vs 10362 ms | 24.6 s vs 10.4 s, or 24.67 s vs 10.37 s | ULTRAPLAN 48, 54; DEMO 249 |
| 671B prompt "2.42x [2.19, 2.72]" | 2.418 [2.194, 2.721] | 2.41x [2.19, 2.73] | ULTRAPLAN, DEVPOST, DECK S4, DEMO, README; PR R6 |
| 671B SSD "95% lower", "95% fewer" | 0.054 [0.026, 0.097], 94.6% lower | 94% lower [90%, 98%] | ULTRAPLAN 48, 65; DEMO 51, 255; README 46; PR R6 ("remove 95%") |
| 671B SSD "684 vs 37 MiB" | 683.588 vs 37.083 | 683 vs 38 | ULTRAPLAN 55; DEMO 51, 255, V4, Q5; DECK S2; DEVPOST; README |
| 671B decode "0.83x [0.77, 0.90]" | 0.830 [0.769, 0.903] | 0.83x [0.76, 0.91] | DECK S5 row 1; DEVPOST; DEMO 52 and V5; README 62; PR R6 |
| 1T prompt "1.51x" | 1.507 [1.412, 1.587] | 1.50x [1.41, 1.59] | ULTRAPLAN, DEVPOST, DEMO, README, DECK; PR R6 |
| 1T decode "0.86x" | 0.856 [0.818, 0.886] | 0.85x [0.81, 0.89] | everywhere; rounding a slowdown toward 1.0 flatters the pool |
| 1T full answer "14% sooner" | 0.863 [0.818, 0.912], 13.7% sooner | 13% sooner [8%, 19%] | ULTRAPLAN claims sheet |
| 1T first token "about 19 s" | 19078.54 ms | about 19.1 s | ULTRAPLAN 53 |
| 1T first token "28.3 alone" | 28254.22 ms | 28.2 s | DEMO 239 |
| R2 120B first token "2.7x sooner" | 2.692, inverted [2.358, 3.077] | 2.69x sooner [2.35, 3.08] | ULTRAPLAN, DEMO V5 script |
| R2 120B full answer "about 1.95x sooner" | 1.949, inverted from 0.513 [0.453, 0.577] | 1.94x sooner [1.73, 2.21] | ULTRAPLAN claims sheet |
| R2 120B "36.2 seconds to 13.4" | 36170 vs 13436 ms | 36.1 to 13.5 seconds | DEMO Q4, Tier B 0:40 |
| R2 120B decode "1.27 [1.10, 1.46]" | 1.274 [1.105, 1.464] | 1.27 [1.10, 1.47] | ULTRAPLAN 69; DECK S5 row 5 and Q&A; DEVPOST; DEMO 62, V5; README 48, 80 |
| R2 120B prompt "2.99x [2.64, 3.35]" | 2.994 [2.639, 3.349] | 2.99x [2.63, 3.35] | ULTRAPLAN 69; DECK S4; DEVPOST 22; DEMO 63; README 80 |
| R2 1 Mac decode "1.25x [1.11, 1.41]" | 1.253 [1.106, 1.407] | 1.25x [1.10, 1.41] | DECK, DEMO, README |
| R2 Tier B "4.20 against 3.30" | 4.198 vs 3.296 | 4.19 against 3.30 | DEMO Tier B 2:20 |
| R1 235B steady state "1.08 [0.98, 1.18]" | 1.079 [0.976, 1.184] | 1.07 [0.97, 1.19] | DECK S5 row 3; DEMO 66; README 49, 82 |
| Hugging Face seeding "about 5x the host link" | 45 MB/s in one single-Mac probe; 13 to 24 MB/s per Mac while seeding; host push 8.6 MiB/s (9.0 MB/s) | "Peers pulled 13 to 24 MB/s each from Hugging Face while seeding, 1.4 to 2.6 times the 8.6 MiB/s host push (45 MB/s in one single-Mac probe)" | ULTRAPLAN 71; DECK S3 notes; PR R1.3 |
| R7.5 "about 2.1 tok/s per chat" | 16.445 / 8 = 2.056 | about 2.05 tok/s per chat | PR R7.5 |
| R7.5 "1.35 [1.25, 1.48]", "16.45", "16.4" | 1.349 [1.246, 1.479], 16.445 | 1.34x [1.24, 1.48], 16.44 | PR R7.5 |
| R7.5 "1.46 [1.30, 1.67]" | 1.464 [1.295, 1.674] | 1.46x [1.29, 1.68] | PR R7.5 |
| R7.1 "0.47 [0.36, 0.61]" | 0.468 [0.362, 0.611] | 0.46 [0.36, 0.62] | PR R7.1; a slowdown rounded toward 1 |
| R7.1 pooled decode "10.01", SSD "10.0" | 10.005, 9.974 | 10.00, 9.97 | PR R7.1 |
| R7.1 "6 to 14 MiB of SSD reads per token" | per-row medians; per rep 3.02 to 26.18, mean 9.97 | 3.0 to 26.2 MiB per token per alone rep | PR R7.1 |
| R7.2 "5.44 [5.42, 5.46]" | 5.444 [5.423, 5.464] | 5.44 [5.42, 5.47] | PR R7.2 |
| R7.2 prefill "1.97 [1.90, 2.04]" | 1.967 [1.896, 2.043] | 1.96x [1.89, 2.05] | PR R7.2 |
| R7.2 "23 GB per token", "2.8 GB" | 23454 and 2761 MiB | 23454 MiB (22.9 GiB) and 2761 MiB (2.7 GiB): units, not rounding | PR R7.2 |
| R7.3 "7.76 [6.95, 8.86]" | 7.759 [6.945, 8.855] | 7.75 [6.94, 8.86] | PR R7.3 |
| R7.4 "1.56 [1.51, 1.61]", "2.59 [2.57, 2.61]" | 1.555 [1.506, 1.608], 2.592 [2.571, 2.613] | 1.55 [1.50, 1.61], 2.59 [2.57, 2.62] | PR R7.4 |
| R6.y aggregates and ratios | 120B 54.621, 111.861 2.048 [1.973, 2.140], 149.156 2.731 [2.583, 2.901]; 1T 1.826 [1.633, 2.021], 2.608 [2.372, 2.844]; 671B 1.792, 1.922 [1.875, 1.970], 5.547 3.096 [2.525, 3.686] | 54.6 (baseline), 111.8 2.04x [1.97, 2.14], 149.1 2.73x [2.58, 2.91]; 1.82x [1.63, 2.03], 2.60x [2.37, 2.85]; 1.80 (baseline), 1.92x [1.87, 1.97], 5.54 3.09x [2.52, 3.69] | PR R6.y table |
| R6.y "step time 3.1 and 2.7 times longer" | 3.105 [2.825, 3.387], 2.683 [2.189, 3.195] | 3.11x [2.82, 3.39] and 2.69x [2.18, 3.20] | PR R6.y |
| R6.y "SSD per aggregate token falls by 2.8 and 3.7 times" | 657.3 / 235.5 = 2.79, 686.3 / 186.4 = 3.68 | 2.7 and 3.6 times lower | PR R6.y |
| R6 GX10 vs G14 on the 671B "3.8x" | 1.805 / 0.481 = 3.75 | host change, cross-session, no interval: do not pitch | PR R6 |
| R6 120B GX10 vs G14 "about 17x decode, 104x prefill" | 16.99x, 103.76x | host change, cross-session, no interval: do not pitch (ULTRAPLAN line 78 bans the 56 tok/s number) | PR R6 |
| R8.1 Maverick "0.44 [0.40, 0.48]", "0.33 [0.31, 0.35]", "0.35 [0.33, 0.37]" | 0.437 [0.400, 0.477], 0.329 [0.314, 0.348], 0.348 [0.329, 0.370] | 0.43x [0.40, 0.48], 0.32x [0.31, 0.35], 0.34x [0.32, 0.37] | PR R8.1 table; slowdowns rounded toward 1 |
| R8.1 Maverick N = 1 prefill "0.54 [0.35, 0.79]", first token "1.64 [1.26, 2.85]", answer "2.11 [1.96, 2.28]", SSD "0.26 [0.17, 0.45]" | 0.543 [0.353, 0.791], 1.637 [1.264, 2.852], 2.111 [1.956, 2.280], 0.260 [0.166, 0.448] | 0.54 [0.35, 0.80], 1.64 [1.26, 2.86], 2.12 [1.95, 2.28], 0.26 [0.16, 0.45] | PR R8.1 |
| R8.2 1T decode "1.00 [0.96, 1.03]" | 0.997 [0.964, 1.030] | 0.99x [0.96, 1.03], "at parity" | PR R8.2; rounding a slowdown up to 1.00 flatters the pool |
| R8.2 1T prefill "1.44 [1.40, 1.48]", answer "0.85 [0.82, 0.88]", N = 8 "1.08 [0.98, 1.19]" | 1.438 [1.396, 1.480], 0.849 [0.819, 0.878], 1.075 [0.975, 1.194] | 1.43x [1.39, 1.48], 0.85 [0.81, 0.88], 1.07x [0.97, 1.20] | PR R8.2 |
| R8.4 aggregates "45.35, 84.99" (Gemma), "39.20, 75.26" (QwQ) | batch medians; `stats.py --batches` means 45.282, 85.044, 39.125, 75.169 | the means, as everywhere else in this sheet (section 3f) | PR R8.4 table |

Run records vs the paper (run records win):

- **R8 RTT.** PR R8 says 5 to 16 ms "in the pooled rows' snapshots". That is the before snapshots (both sources) of
  the five completed pooled rows. After them, the index `network_after` snapshots show 6 to 50 ms and run.json
  `gateway_after` 5 to 52 ms. Before the failed 671B row: 13 to 21 ms. Quote "5 to 16 ms before each pooled row, up to
  52 ms after".
- **R8.2 relay.** PR R8.2 says R6 measured 0.86x on the 1T "with one peer on the relay". In R6 the relay peer
  (mac-c) was in the 671B rows; the 1T pooled rows had all four peers direct (condition G6, PR R6). R8 has three
  peers and no mac-c, so the change is not a relay path fixed.
- **R8.3 671B failure.** PR R8.3 says the row "loaded for 1071 s and then failed". Compute-buffer reservation started
  at 1071.4 s; run.json `load.seconds` is 1077.4. The `ggml-rpc.cpp:682` message and the peer buffer sizes are from
  the engine log, not run.json, whose `stop.error` is "peer d4e65f83 lost: worker exited (exit code -6)".

- **R7 RTT.** PR R7 says both Macs held 5 to 7 ms, and PR R7.3 says 5 to 6 ms. Before each pooled row both
  snapshot sources show 5 to 8 ms. After it, the index `network_after` snapshots show 8 to 46 ms (d70 15 and 30,
  g120 8 and 11, vl72 33 and 35, n80 9 and 8, g120n4 13 and 11, g120n8 16 and 46) and run.json `gateway_after` 8 to
  42 ms (d70 14 and 34, vl72 31 and 33, g120n8 18 and 42). Quote "5 to 8 ms before each pooled row, up to 46 ms
  after".
- **R6 RTT.** PR R6 gives 4 to 36 ms direct and 47 to 84 ms relay "in the 671B rows", which mixes alone and pooled
  rows and uses the index snapshots only. Pooled rows only, both sources: relay 48 to 103 ms (103 is run.json
  `gateway_after` of `671b-gx4-1`), direct 5 to 36 ms. For the 1T pooled rows PR R6 says 6 to 51 ms (index); with
  run.json it is 5 to 55 ms (`kimi-gx4-2` after, mac-c 55; `kimi-gx4-1` after, g14 5).
- **B1 RTT.** PR R1 "12 to 134 ms during setup" is paper text. The run.json snapshots of the pooled rows
  (`120b-3mac-1..3`, `120b-1mac-1..2`, `235b-3mac-1,2,4,5,6`; the Macs in each pool) show 7 to 64 ms, and the index
  snapshots 7 to 58 ms.
- **R7.5 step time, pooled N = 1.** PR R7.5 says 114 ms, which is 1000 / 8.74. `stats.py --batches` gives a mean
  step of 115.550 ms (median 110.245). This sheet uses 115.550. The alone figure is 261.638 by the tool, 261 in the
  paper.
- **R7 commit guard.** PR R7 says every row after the first aborts allowed 8 GiB of commit growth. In run.json
  `requested.max_commit_growth_gib`: n80, d70 and g120 (N = 1) requested 8; the vl72 rows (R7.4) and the N = 4 and
  N = 8 rows (R7.5) requested 24; the first aborted n80 and d32 rows 0.25. It is a guard, not a measured number.
- **R7.2 second alone row.** PR R7.2 calls its warmups (0.180, 0.179, 0.181 tok/s) "consistent with row 1". They are
  about 9% faster than row 1's measured 0.165 and 15 to 22% faster than row 1's warmups (section 1a).
- **R6 120B SSD.** PR R6 prints 0 MiB per token. The 9 reps average 0.021 (median 0.002).
- **R6.x model size.** PR R6.x says "a dense model larger than the GB10's unified memory". It is 121.33 GiB
  against 121.6 GiB, so about the same size, not larger.
- **R2 with `stats.py` and no label filter** includes the warm-up run and gives 1.262 [1.100, 1.444] decode. The
  paper set (`--only-labels`, warm-up excluded) gives 1.274 [1.105, 1.464], which matches PR R2.


## 7. Recorded rehearsal provenance, 2026-10-25

These are functional-demo observations, not controlled benchmark comparisons. Do not compare their token rates with the tables above. Source: [rehearsal record](../demo/REHEARSAL-2026-10-25.json) and the matching timestamped app captures retained with the local video.

- DeepSeek-V3.1 671B, UD-TQ1_0, GX10 host with three eligible peers: mac-a, mac-b and g14. The tier was moe_offload.
- The actual plan had 61 layers: mac-b 5, mac-a 2, g14 3, GX10 51. Those counts describe this rehearsal only.
- Deployment created at 11:43:40.925367 UTC; ready at 11:49:31.353791 UTC. Elapsed deployment-to-ready was 350.428424 seconds, shown as 350.4 seconds. Seeding preceded deployment and is separate.
- The loading edit shows 12 seconds, a 29.2x time compression after rounding (350.428424 / 12). It reconstructs the interval from actual loading and ready captures; it is not continuous loading footage.
- The peer-list capture at 11:41:07 UTC showed three eligible friends with admissible memory: mac-b 7.2 GiB, g14 9.0 GiB, mac-a 7.1 GiB. These are instantaneous planner readings, not hardware specifications. The loading capture showed 0 GiB admissible on each after reservations.
- The mac-a join command completed with exit code 0 and confirmed sharing. It was invoked remotely over SSH. A physical double-click was not filmed.
- Seeding took 180 seconds according to the example. Its readiness poll reported 351 seconds after deployment; the gateway history gives the more precise 350.428424-second transition above.
- The app displayed 2.23 tokens/s for its single recorded short answer. This is a functional spot check, not a benchmark or an improvement claim. The prompt and unedited answer are retained in the rehearsal record.
