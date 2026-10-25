# Paper runs, 2026-10-22 to 23

Measured runs of Hypercompute pooling one Windows laptop and three MacBook Airs to serve models larger than any one of
them. Every number in this file comes from a `run.json` written by `scripts/capacity/run_big.py`. Estimates are
labelled as estimates.

## Setup

| machine | role | chip | memory | storage | OS |
|---|---|---|---|---|---|
| ROG Zephyrus G14 (GA403WR) | host: llama-server, gateway | AMD Ryzen AI 9 HX 370, RTX 5070 Ti Laptop 12 GiB (driver 591.84) | 31.1 GiB | WD SN5000S 1 TB NVMe | Windows 11 Home 26200 |
| mac-azim (Mac16,12) | peer: ggml-rpc-server on Metal | Apple M4, 10 cores | 16 GiB unified | 52 GiB free | macOS 26.6.2 |
| mac-a (Mac16,12) | peer | Apple M4, 10 cores | 16 GiB unified | 183 GiB free | macOS 15.7.7 |
| mac-b (Mac16,12) | peer | Apple M4, 10 cores | 16 GiB unified | 183 GiB free | macOS 15.7.2 |

Software: llama.cpp b11770 (commit 39da130fc, RPC protocol 8.0.0) on every machine. Hypercompute gateway at db44c95
(branch `codex/iroh-transport`), Macs updated to the same commit. Peers reach the host over an Iroh QUIC tunnel; the
bridge speedups of 751f00b are on unless a run says otherwise (`HYPERCOMPUTE_STOCK_BRIDGE=1`).

Each Mac contributes at most 7.5 GiB with no reserve, pinned below its free memory so the planner gives the same
layers to the same Mac on every run.

Network: building Wi-Fi. The G14 sits on 10.2.144.0/x and the Macs on 10.3.80.0/x, with no LAN route between them, so
Iroh holds a direct path through NAT. Iroh RTT ranged 12 to 134 ms during setup; Mac to router pings averaged 15 ms
with spikes to 80 ms. Host to Mac bulk transfer measured 6.6 to 8 MiB/s over Iroh; an SSH stream to another Mac at the
same time got 3.3 MiB/s, so the link, not the tunnel, caps uploads.

Method: decode tok/s is the engine's own timing, (predicted_n - 1) / predicted_ms, over the measured reps only;
warmups are excluded. Prompts are fixed length with `ignore_eos`, temperature 0, seed 7, no prompt cache.
SSD bytes per token are system-wide physical reads bracketed around each chat. Per-peer bridge bytes come from the
gateway's `/api/network` counters before and after each run. Mac memory, swap, battery and thermal state are sampled
over SSH before and after each run. The driver and index live in the session scratchpad (`runs/index_*.jsonl`).

## Results

### R1. Qwen3-235B-A22B (UD-Q2_K_XL, 82.67 GiB): G14 alone vs G14 + 3 Macs

Placement with 3 Macs (planner, tier `moe_offload`): each Mac holds 7 or 8 whole layers (19.3 GiB over the three);
the G14 GPU holds attention, norms, router, shared experts, head and KV for the other 72 layers (10.2 GiB), and their
routed experts (53.1 GiB) stay memory-mapped from the GGUF on the G14. Alone, the G14 keeps the routed experts of all
94 layers file-backed (about 72 GiB against 31 GiB of RAM).

Runs 1 and 2 used 64 prompt and 32 output tokens with 2 measured reps; runs 3 to 6 used 64 and 64 with 3 reps.
Order: D1 A1 A2 D2 A3 (D3 refused, see R1.3) A4 D4 D5 A5 D6. "D" is 3 Macs, "A" is the G14 alone.

| run | Macs | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token | Iroh RTT ms |
|---|---|---|---|---|---|---|---|
| D1 | 3 | 2289 (cold) | 1.405, 1.271 | 1.338 | 2.247 | 402 | 26, 15, 58 |
| A1 | 0 | 50 | 0.888, 0.861 | 0.875 | 1.376 | 1013 | |
| A2 | 0 | 54 | 1.162, 0.965 | 1.064 | 1.397 | 871 | |
| D2 | 3 | 166 | 1.133, 1.003 | 1.068 | 2.200 | 463 | 10, 8, 33 |
| A3 | 0 | 52 | 1.177, 1.163, 1.069 | 1.163 | 1.395 | 702 | |
| D3 | 3 | refused | | | | | |
| A4 | 0 | 52 | 1.295, 1.262, 1.501 | 1.295 | 1.430 | 611 | |
| D4 | 3 | 152 | 1.399, 1.210, 1.275 | 1.275 | 2.357 | 310 | 12, 8, 10 |
| D5 | 3 | 130 | 1.458, 1.306, 1.531 | 1.458 | 2.498 | 299 | 8, 8, 8 |
| A5 | 0 | 44 | 1.284, 1.262, 1.635 | 1.284 | 1.574 | 612 | |
| D6 | 3 | 175 | 1.509, 1.486, 1.393 | 1.486 | 2.386 | 290 | 37, 19, 20 |

Pooled reps, ratio of means (3 Macs / alone) with a 95% percentile bootstrap interval (10,000 resamples, seed 7):

| metric | all runs (n = 13 reps each) | steady state, runs 3 to 6 (n = 9 each) |
|---|---|---|
| decode tok/s | 1.337 vs 1.194, ratio 1.12 [1.00, 1.26] | 1.396 vs 1.294, ratio 1.08 [0.98, 1.18] |
| prefill tok/s | 2.300 vs 1.431, ratio 1.61 [1.51, 1.71] | 2.333 vs 1.450, ratio 1.61 [1.48, 1.74] |
| SSD MiB per decoded token | 338 vs 738, ratio 0.46 [0.39, 0.54] | 296 vs 647, ratio 0.46 [0.42, 0.49] |

### R1.1 What the Macs buy

- Prefill (time to first token) is consistently faster with the Macs: about 2.2 to 2.4 tok/s against 1.4 alone, about
  1.6x. Prefill is batched, so the Macs' GPUs run the expert matmuls of their layers that the G14 would otherwise run
  on its CPU.
- SSD reads per generated token roughly halve with the Macs (the host keeps 53 instead of 72 GiB of experts on disk).
- Decode depends on how warm the G14's page cache is. Cold, the Macs gave 1.34 against 0.88 tok/s (run pair 1). As
  the G14's standby list filled over successive runs, the G14 alone rose from 0.88 to about 1.29 tok/s while its SSD
  reads fell from 1013 to 611 MiB per token. At steady state the Macs give 1.08x, with an interval that includes
  parity: on this Wi-Fi, three Macs do not reliably speed up warm decode of this model.

### R1.2 Why decode reaches parity: one blocking round trip per Mac per token

In llama.cpp b11770 the RPC client sends a token's graph to each peer asynchronously (`GRAPH_RECOMPUTE` reuses the
cached graph by uid), but the next stage needs that peer's output, which is a blocking `GET_TENSOR` (or a host-mediated
copy). Each Mac therefore costs at least one network round trip plus its own compute per token, in sequence, at batch
size 1. With 10 to 25 ms RTT on this Wi-Fi plus the Mac's compute for 7 or 8 layers, that is on the order of 30 to 50
ms per Mac, about 100 to 150 ms per token for three Macs. With a warm host cache, that is about what the Macs save in
SSD reads, so decode breaks even. Gateway bridge counters agree: each Mac returns about 16 KB of activations per token
(10.8 MB over 320 decoded plus 320 prompt tokens in D4) in several response chunks per token.

### R2. gpt-oss-120b (MXFP4, 59.03 GiB): G14 alone vs 1 Mac vs 3 Macs

Placement: with 3 Macs each holds 4 whole layers (layers 0-3, 4-7, 8-11, about 6.4 GiB each); the G14 GPU holds the
dense parts, head and KV of the other 24 layers, whose routed experts (about 30 GiB) stay file-backed on the G14. With
1 Mac (mac-azim) the Mac holds 4 layers and the G14 keeps the experts of 32. Every Mac run was preceded by
`run_big --seed-peers`: each Mac downloaded its own layers from Hugging Face before the load.
128 prompt and 64 output tokens, 1 warmup and 3 measured reps. Order: Aw D1 A1 D2 B1 A2 B2 D3 A3 (Aw is a warm-up
run after the 235B runs and is excluded from the pooled numbers; "B" is 1 Mac).

| run | Macs | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|---|
| Aw | 0 | 41 | 3.160, 3.505, 3.574 | 3.505 | 3.862 | 276 |
| D1 | 3 | 126 after 566 s seeding | 4.733, 4.634, 3.762 | 4.634 | 13.342 | 32 |
| A1 | 0 | 38 | 2.980, 2.893, 3.381 | 2.980 | 3.472 | 287 |
| D2 | 3 | 158 | 5.656, 3.375, 5.198 | 5.198 | 11.595 | 39 |
| B1 | 1 | 72 | 3.724, 4.775, 4.401 | 4.401 | 5.365 | 143 |
| A2 | 0 | 44 | 3.072, 3.721, 3.497 | 3.497 | 3.694 | 271 |
| B2 | 1 | 80 | 4.802, 3.558, 3.517 | 3.558 | 5.491 | 153 |
| D3 | 3 | 167 | 3.723, 3.152, 3.547 | 3.547 | 8.149 | 56 |
| A3 | 0 | 42 | 3.159, 3.649, 3.311 | 3.311 | 3.877 | 265 |

Pooled reps, ratio of means with a 95% bootstrap interval:

| comparison | decode | prefill | SSD MiB per decoded token |
|---|---|---|---|
| 3 Macs vs alone (9 vs 9 reps) | 4.198 vs 3.296, 1.27 [1.10, 1.46] | 11.03 vs 3.68, 2.99 [2.64, 3.35] | 41 vs 275, 0.15 [0.12, 0.19] |
| 1 Mac vs alone (6 vs 9) | 4.130 vs 3.296, 1.25 [1.11, 1.41] | 5.45 vs 3.68, 1.48 [1.41, 1.57] | 148 vs 275, 0.54 [0.52, 0.56] |
| 3 Macs vs 1 Mac (9 vs 6) | 1.02 [0.86, 1.20] | 2.02 [1.78, 2.27] | 0.28 [0.22, 0.35] |

Reading:

- Unlike the 235B, the 120B gains on decode with Macs: taking 4 to 12 layers off the G14 brings its file-backed
  expert set close to its RAM, so SSD reads per token fall 2x (1 Mac) to 7x (3 Macs).
- Decode does not improve from 1 to 3 Macs: the extra SSD savings are paid back in two more per-token stages.
  Prefill keeps scaling (2x from 1 to 3 Macs), because batched expert math moves from the G14 CPU to Metal.
- Confound: a code workflow ran on the G14 from about 00:44 UTC, so D2, B1, A2, B2, D3 and A3 shared the G14 with it.
  With 3 Macs the G14's expert set sits at the edge of its RAM, and D1 to D3 show SSD reads rising (32, 39, 56 MiB per
  token) and decode falling (4.63, 5.20, 3.55). D3 is therefore likely pessimistic. The alone runs are less sensitive
  (their set exceeds RAM by about 15 GiB either way).
- Seeding: the first 3-Mac cold start took 566 s of seeding (about 240 s of host hashing, then 13 to 24 MB/s per Mac
  from Hugging Face, 20.6 GB of which 5.1 GB was already cached on mac-azim) plus a 126 s load, against an estimated
  2400 s to push the same bytes over the 8.6 MiB/s host link.

### R3. hcpeer on a real Mac (S1 gate, 2026-10-23)

hcpeer v4 (`native/hcpeer/build/libggml-rpc.dylib`, sha256 8b582616...) was installed on mac-a and mac-b with
`scripts/capacity/hcpeer_mac_install.sh`, which clones the stock b11770 engine dir to `<dir>-hcpeer` and swaps the
library; the stock dir is unchanged and the clone's server printed `hcpeer mmap : on (v4, page 16384 B, ...)` on both.

Gate on mac-a (`scripts/capacity/paper/hcpeer_gate.sh`): one greedy completion (64 tokens, temperature 0, seed 7) with
llama-server on the Mac and every layer on an rpc-server CPU device, three ways: stock server; hcpeer with
`HC_PEER_MMAP=1 -c` on an empty cache (tensors are saved, then mapped); hcpeer again on the filled cache (mapped from
the start).

| model | stock | hcpeer first load | hcpeer mapped load | text | mapped |
|---|---|---|---|---|---|
| Qwen2.5 0.5B Q4_K_M | identical | identical | identical | match | 1 tensor (output.weight, 138 MiB) per load |
| Qwen2.5 7B Q4_K_M | 17.89 tok/s | 17.99 tok/s | 17.96 tok/s | match | 85 tensors, 3.63 GiB per load; 0 copy fallbacks; swap 0 |

So on a Mac the patch serves weights from mapped SSD files with byte-identical results and no decode cost while the
mapped set fits in page cache. What it costs when the set does not fit is the C1 question.

### R4. DeepSeek-V3.1 671B on the G14 alone (2026-10-23)

Model: unsloth DeepSeek-V3.1 UD-TQ1_0 (one 170.5 GB file, revision feb73eff, sha256 verified; deepseek2, 61 blocks,
first 3 dense, 256 routed experts with 8 used, shared experts, MLA; tensor types IQ1_S to Q8_0, no TQ1_0 tensors).
Tier moe_offload on the G14 alone: CUDA0 holds every block's dense parts plus the head (about 10.1 GiB), and the routed
experts of all 58 MoE blocks (about 149 GiB) stay memory-mapped from the NVMe against 31 GiB of RAM.

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| 671b-local-1 (64 prompt, 32 output, 1 warmup, 2 reps) | 50 | 0.482, 0.480 | 0.481 | 0.747 | 2485 |

A 671B-parameter model decodes on one gaming laptop at about half a token per second, paging about 2.5 GB of experts
per token from its SSD. The layout analysis (`deepseek-671b-layouts.md`) predicted 0.48 tok/s and 2002-2392 MiB per
token for this case before it ran. The Mac layouts (B: three Metal stages, 13 layers; C: B plus 8 hcpeer SSD layers on
mac-a and mac-b) plan feasibly (C needs 11.3 GiB on mac-b, so mac-a and mac-b joined at 11.5 GiB caps) but did not run:
see R5.

### R5. Network findings, 2026-10-23

- Apartment Wi-Fi (Trailside Guest and Trailside WiFi) puts each client in its own subnet (10.21.x, 10.157.x, 10.18.x)
  and the G14 sits behind an endpoint-dependent NAT (`tailscale netcheck`: MappingVariesByDestIP true, no port
  mapping). No direct path forms between laptops, so Tailscale and Iroh fall back to the Seattle relay.
- With the G14 on guest Wi-Fi and the Macs on a phone hotspot, every host-to-Mac byte crossed the internet: host to
  Mac uploads ran at about 1 MiB/s, Iroh RTT rose to 60-580 ms, and a peer dropped mid-load. The 235B SSD-tier run (C1)
  and four Mac runs of that batch failed or were refused for that reason; only its G14-alone runs are valid
  (1.24-1.30 tok/s, matching the earlier steady state).
- On Windows, `f.truncate(size)` physically zero-fills a new file, so pre-sizing a 170 GB download took about 50
  minutes of full disk bandwidth and stalled the gateway's event loop for about 70 s.
- Seeding peers from Hugging Face over a phone hotspot uses cellular data (about 300 KB/s per Mac); seed on Wi-Fi.

### R6. ASUS Ascent GX10 (GB10) as host, 2026-10-24

Setup change: a rented ASUS Ascent GX10 hosts (llama-server and gateway); the G14 joins as a CUDA peer for the first
time, next to the three Macs.

| machine | role | chip | memory | storage | OS |
|---|---|---|---|---|---|
| ASUS Ascent GX10 (gx10-49d1) | host | NVIDIA GB10 (Grace 10x Cortex-X925 + 10x A725, Blackwell GPU sm_121), driver 580.178.04, CUDA 13.0 | 121.6 GiB unified (one pool, CPU and GPU) | 916 GB NVMe | DGX OS 7.5.0 (Ubuntu 24.04.5), kernel 7.0.0-1019-nvidia |
| G14 | peer: ggml-rpc-server on CUDA0, at most 9 GiB | as above | as above | 50 GB free | as above |

Software: llama.cpp b11770 (commit 39da130fc) built on the GX10 with `scripts/build_llama_linux.sh --cuda-arch 121`
(version banner identical to the release); Hypercompute at bef2020 on the host, 75423bd on the Macs (code-identical).
Caps: mac-a and mac-b 11.5 GiB, mac-azim 7.5 GiB, G14 9 GiB. The host's planner budgets GB10 as one unified pool:
CUDA0 79.06 GiB (the 65% contribution limit), the CPU tier the rest less a 24.3 GiB reserve.

Network: home Wi-Fi, one subnet; the GX10 has no wired link. Per the `/api/network` snapshots in the index, mac-a,
mac-b and the G14 were direct in every run (RTT 4 to 36 ms in the 671B rows, 31 to 67 ms during the 120B rows while a
download ran), but mac-azim stayed on the Iroh relay in every R6 run (47 to 84 ms in the 671B rows, 95 to 148 ms in the
120B rows), although a spot check between runs had shown it direct after a rejoin. It is not the network: plain UDP
between the GX10 and mac-azim's LAN address worked in both directions from the same uv Python, and mac-azim's own join
log reports its connection to the host as direct (8 ms). Only the host's connection to it never gains a direct path
(one path, the relay), while mac-a's gains a second, direct one. mac-azim is the only peer on macOS 26 (26.6.2; the
others run 15.7); the cause is open. Reproduce the tables with
`HC_PAPER_REPO=<this repo> python -I scripts/capacity/paper/summarize.py` and `stats.py`.

**gpt-oss-120b on the GX10 alone** (128 prompt, 64 output, 1 warmup, 3 reps; tier gpu, all 36 layers on CUDA0):

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| 120b-gxA-1 | 50 | 56.353, 56.367, 56.166 | 56.353 | 383.3 | 0 |
| 120b-gxA-2 | 46 | 55.211, 55.610, 55.768 | 55.610 | 379.2 | 0 |
| 120b-gxA-3 | 48 | 56.273, 56.206, 56.044 | 56.206 | 386.5 | 0 |

Pooled 9 reps: decode mean 56.00 tok/s (median 56.17), prefill mean 382.0 tok/s. Against
the G14 alone in R2 (3.296 decode, 3.68 prefill) that is about 17x decode and 104x prefill, a host change, not pooling.
Confound: the 671B download ran on the GX10 during these rows (about 30 MiB/s, read by hand from the downloader's
progress, not recorded in run.json).

**DeepSeek-V3.1 671B: GX10 alone vs GX10 + G14 + 3 Macs** (64 prompt, 32 output, 1 warmup, 2 reps; tier moe_offload;
order A1 P1 A2 P2). Alone: CUDA0 holds 77.3 GiB (every block's dense parts, the head and part of the experts) and
81.0 GiB of routed experts stay file-backed in the same 121.6 GiB of memory. Pooled (gx4-1): mac-azim layers 0-4
(6.0 GiB), mac-b 5-7 (7.8), G14 8-10 (8.2), mac-a 11-13 (8.9), CUDA0 14-60 (76.5), CPU experts 51.0 GiB. gx4-2 kept
the same per-layer sizes but put different peers on layers 5-13 (G14 5-7, mac-a 8-10, mac-b 11-13), so it re-seeded;
the pooled reps below mix those two layouts. Every pooled run used `--seed-peers`: each peer fetched its layers from
Hugging Face before the load.

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| 671b-gxA-1 | 239 | 1.781, 1.868 | 1.825 | 2.644 | 686 |
| 671b-gx4-1 | 503 after 1416 s seeding | 1.407, 1.373 | 1.390 | 5.901 | 23 |
| 671b-gxA-2 | 256 | 1.789, 1.781 | 1.785 | 2.618 | 681 |
| 671b-gx4-2 | 517 after 856 s seeding | 1.690, 1.521 | 1.605 | 6.822 | 51 |

Pooled reps, ratio of means pooled vs alone with a 95% bootstrap interval (4 vs 4 reps):

| decode | prefill | SSD MiB per decoded token |
|---|---|---|
| 1.497 vs 1.805, 0.83 [0.77, 0.90] | 6.36 vs 2.63, 2.42 [2.19, 2.72] | 37 vs 684, 0.054 [0.026, 0.097] |

Reading:

- The GX10 alone decodes the 671B at 1.8 tok/s, 3.8x the G14 alone (R4: 0.481), reading 684 instead of 2485 MiB per
  token from its SSD: its unified memory keeps far more of the experts resident.
- Four peers remove 95% of the host's SSD reads and raise prefill 2.4x, but decode falls 17%. As with the 235B (R1.2),
  each peer is one blocking stage per token; on this host the SSD reads they save cost less than four Wi-Fi stages.
  One of the four stages (mac-azim, layers 0-4) ran over the relay at 47 to 84 ms, so the pooled decode is a lower
  bound for four direct peers. Fewer, larger, direct stages (the G14 alone, or mac-a and mac-b) are the next
  configurations to measure.
- Planner estimate (plan.moe, the comparison block of each run.json): 2147 and 2137 ms per token alone, 1212 ms pooled.
  Measured: 548 and 560 ms alone (measured_over_predicted 0.255, 0.262), 720 and 625 ms pooled (0.594, 0.515). The
  planner ranked pooled about 1.77x ahead of the GX10 alone, the reverse of the measured 0.83x: its speed constants come
  from the G14 and overstate SSD cost on the GX10.
**Kimi-K2.6 1T: GX10 alone vs GX10 + G14 + 3 Macs** (bartowski IQ1_S, 193.16 GiB in six parts, 1026B parameters, 32B
active; same workload as the 671B; tier moe_offload; order A1 P1 A2 P2). Alone: CUDA0 77.3 GiB, 115.5 GiB of routed
experts file-backed. Pooled, the same layout in both runs: mac-azim layers 0-2 (6.9 GiB), G14 3-4 (6.6), mac-a 5-7
(9.9), mac-b 8-10 (9.5), CUDA0 11-60 (76.0), CPU experts 83.9 GiB. All four peers were direct in both pooled runs
(RTT 6 to 51 ms): the host was restarted at 20:54 UTC and again at 21:14 UTC, and after each restart every peer,
mac-azim included, came back direct.

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| kimi-gxA-1 | 257 | 1.789, 1.858 | 1.823 | 2.212 | 662 |
| kimi-gx4-1 | 367 after 676 s seeding | 1.560, 1.585 | 1.573 | 3.414 | 255 |
| kimi-gxA-2 | 266 | 1.792, 1.759 | 1.775 | 2.165 | 680 |
| kimi-gx4-2 | 369 after 5 s seeding (cache warm) | 1.444, 1.570 | 1.507 | 3.181 | 253 |

Pooled vs alone, ratio of means with a 95% bootstrap interval (4 vs 4 reps):

| decode | prefill | SSD MiB per decoded token |
|---|---|---|
| 1.540 vs 1.799, 0.86 [0.82, 0.89] | 3.30 vs 2.19, 1.51 [1.41, 1.59] | 254 vs 671, 0.38 [0.37, 0.40] |

- A 1T-parameter model decodes at 1.8 tok/s on the GX10 alone, the same speed as the 671B, with 32B active parameters
  against 37B. Pooling repeats the 671B pattern with four direct peers: fewer SSD reads and faster prefill, slower
  single-request decode.
- Planner estimate: 2161 and 2156 ms per token alone, 1491 and 1496 pooled; measured 549 and 563 alone, 636 and 665
  pooled (measured_over_predicted 0.254, 0.261, 0.426, 0.444).
- Coherence (1-bit quantization): kept GX10-alone deployment, temperature 0, thinking off. Asked for an iterative
  Fibonacci function, it wrote a correct, documented function with input validation (run 20261024T215444Z).
- Operational: the first pooled attempt pushed every peer layer from the host at about 4 MiB/s because the peers'
  catalogs lacked the new model (each peer agent caches its catalog at the first seed command), so it was stopped,
  the catalogs were updated and the peers rejoined. Its runs are not in the index. Two earlier attempts were refused
  because the gateway itself had loaded its catalog before the entry existed
  (`index_kimi_gx_refused_stale_catalog.jsonl`).
- Confound: an idle Nemotron 30B snap service (root, sleeps when idle) is installed on the rented unit. A manual
  nvidia-smi check at 17:10 UTC, before the 120B rows, showed no compute process; nothing about it is in run.json.

#### R6.x Mistral Large 2411 (Q8_0, 121.33 GiB dense) on the GX10 alone: does not run

Two alone rows (64-token prompt, 32-token output, 2 reps), `index_gx10_ml123.jsonl`. The planner chose gpu_disk:
56 layers on CUDA0 and 32 file-backed on the CPU. The first row ended `load_failed` after 2465 s. The second was
killed (exit 137) at 3031 s. From about 00:35 to 01:25 UTC the unit answered ping but not SSH, with a load average
of 120 and 121 of 121 GiB in use. The Hypercompute gateway was gone afterwards, so the six concurrency rows queued
behind these rows failed at once with the gateway unreachable (`index_gx10_chain_failed_gateway_down.jsonl`). The
weights (121.33 GiB) are about the size of the GB10's whole unified memory (121.6 GiB), which the GPU, the page cache
and every process share, so the file-backed CPU layers kept evicting each other and the box did not survive it. The
planner approved this plan even so; that is a planner miss, not only an engine limit.

- Planner fix (2026-10-25, branch `fix/uma-planner-no-disk-paging`): on a unified-memory host the planner now charges
  CPU layers paged from the GGUF to the one pool, since their pages sit in the file cache the GPU allocates from, and
  it neither streams CPU units nor replays a measured gpu_disk profile there. The capacity service no longer keeps the
  planner's plan feasible on such a host when its optimizer finds no feasible layout. This request is now refused instead of planned as gpu_disk
  (`tests/test_uma_disk_tiers.py`); on a host with its own GPU memory gpu_disk is unchanged.

- Confound found after these rows: the rental's Nemotron 3.5 Lightning 30B snap server (root, `--sleep-idle-seconds
  600`) was restarted at 01:25 UTC and held 30.5 GiB of GPU memory. The rerun of the concurrency rows
  (`plan_gx10_conc2.json`) waits until no GPU compute process remains before its first row and logs what it saw.

#### R6.y Concurrent requests on the GX10 alone: 1, 4 and 8 chats

The engine runs N server slots with a unified KV cache (`-np N -kvu`, `PlanRequest.parallel`). Each N>1 rep is a
batch of N chats started together. Aggregate decode tok/s is counted over the window where all N are generating
(`scripts/capacity/batchlib.py`). The ratios resample batches, not chats, because the chats of one batch share every
decode step. Indexes: `index_g120_conc.jsonl` (120B, interleaved n1 n4 n8 n8 n4 n1, 128-token prompt, 64-token
output, 3 batches per row) and `index_gx10_conc2.jsonl` (1T and 671B, one row per N, 64/32 tokens, 2 batches).

| model | N=1 tok/s | N=4 aggregate | N=8 aggregate | step ms at N=1, 4, 8 | SSD MiB per aggregate token at N=1, 4, 8 |
|---|---|---|---|---|---|
| gpt-oss-120b | 54.6 | 111.9, 2.05x [1.97, 2.14] | 149.2, 2.73x [2.58, 2.90] | 18.3, 35.8, 54.1 | about 0 (median 0.0 at each N) |
| Kimi K2.6 1T | 1.78 | 3.25, 1.83x [1.63, 2.02] | 4.64, 2.61x [2.37, 2.84] | 562, 1244, 1745 | 657, 360, 236 |
| DeepSeek-V3.1 671B | 1.79 | 3.44, 1.92x [1.88, 1.97] | 5.55, 3.10x [2.53, 3.69] | 558, 1171, 1498 | 686, 319, 186 |

- One machine serves eight users of a 1T or 671B model at 4.6 to 5.5 tok/s in aggregate, against 1.8 for one user.
  Each user's own stream slows (step time 3.1 and 2.7 times longer at N=8), but the expert reads from SSD are shared
  across the batch, so SSD MiB per aggregate token falls by 2.8 and 3.7 times.
- The 120B fits in GPU memory (no SSD reads), so its gain comes from batching alone and flattens sooner.
- Not measured: the same rows pooled with peers. The laptops were running R7 as their own pool while these rows ran.

### R7. Laptop sweep: G14 host with mac-a and mac-b, 2026-10-25

Same G14 and Macs as the Setup table (mac-azim was not used), on a home network where both Macs held direct Iroh
paths at 5 to 7 ms RTT. Each pooled row waits until both peers are direct with RTT under 30 ms. Rows alternate alone
(A) and pooled (P): A P A P for headline models, one A P pair for the rest. Prompt and output lengths are in each
model's subsection. The driver is `scripts/capacity/paper/matrix.py` and the indexes are in
`results/capacity/paper-2026-10-24-laptop/`. After the first pooled row of a model, its placement is pinned
(`layout_<tag>.json`) so the later pooled rows reuse the same layout. Alone and pooled rows give the planner the
same freedom: either may fall back to a disk-backed tier when nothing faster fits.

Aborted and refused rows (kept, not in the tables):
- The first 80B and 32B rows (`index_n80_aborted_commit_guard.jsonl`, `index_d32_aborted_commit_guard.jsonl`) were
  stopped by the host guard because the Windows commit limit grew from 1.01 to 50.12 GiB as the pagefile expanded
  while other builds ran on the G14. From then on every row allows 8 GiB of commit growth (`--max-commit-growth-gib 8`),
  raised to 24 GiB from the Qwen2.5-VL 72B rows on (each row's `args` in the index records which).
- The 32B alone row with `--no-capacity-mode` was refused as infeasible (`index_d32_refused_no_capacity_alone.jsonl`);
  that flag now applies to pooled rows only.
- The second pooled 80B row was refused because its pinned layout named the host-side device (RPC0) for each Mac
  instead of the device the Mac's worker serves (MTL0) (`index_n80_refused_layout_device.jsonl`). Layouts are now
  written with MTL0 for peers.
- The first pooled 70B rows ran with `--no-capacity-mode`, which limits the plan to GPU memory. The G14's GPU plus
  two Macs hold 32.2 GiB against about 40 GiB of weights, so both attempts were refused as infeasible
  (`index_d70_refused_gpu_only.jsonl`). The campaign was restarted at 01:26 UTC without that flag on any row.

#### R7.1 Qwen3-Next-80B-A3B (Q3_K_M): 128-token prompt, 64-token output, 3 reps

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| n80-g14A-1 | 22.0 | 14.75, 24.42, 18.91 | 18.91 | 79.0 | 6.0 |
| n80-g14P2-1 | 180.4 | 8.92, 10.21, 10.89 | 10.21 | 29.7 | 5.1 |
| n80-g14A-2 | 18.0 | 15.53, 20.17, 34.47 | 20.17 | 84.8 | 14.2 |

Pooled vs alone, ratio of means with a 95% bootstrap interval (3 pooled vs 6 alone reps):

| decode | prefill | SSD MiB per decoded token |
|---|---|---|
| 10.01 vs 21.37, 0.47 [0.36, 0.61] | 29.5 vs 84.3, 0.35 [0.30, 0.40] | 5.4 vs 10.0, 0.54 [0.20, 1.38] |

- A 3B-active MoE runs fast on the G14 alone: its experts stay in the page cache (6 to 14 MiB of SSD reads per token),
  so there is little disk traffic for the Macs to remove. Each Mac adds a blocking round trip per token, so pooled
  decode and prefill are both slower. This model does not show a pooling benefit.

#### R7.2 Llama 3.3 70B Instruct (Q4_K_M, dense): 64-token prompt, 32-token output, 2 reps

Alone, the plan streams 61 of 80 layers from the file on the host CPU and keeps 19 on CUDA0. Pooled, it keeps 21 on
the host CPU, 20 on each Mac and 19 on CUDA0 (tier stream in both).

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| d70-g14A-1 | 42.3 | 0.165, 0.165 | 0.165 | 6.36 | 23454 |
| d70-g14P2-1 | 96.2 | 0.901, 0.895 | 0.898 | 12.51 | 2761 |

Pooled vs alone, ratio of means with a 95% bootstrap interval (2 vs 2 reps, one row each):

| decode | prefill | first token | 32-token answer | SSD MiB per decoded token |
|---|---|---|---|---|
| 0.898 vs 0.165, 5.44 [5.42, 5.46] | 12.5 vs 6.36, 1.97 [1.90, 2.04] | 5.0 vs 10.0 s, 0.50 [0.48, 0.52] | 39.5 vs 197.9 s, 0.20 [0.20, 0.20] | 2761 vs 23454, 0.12 |

- About 40 GiB of weights do not fit the G14's 12 GiB GPU and 31 GiB RAM together with everything else, so alone
  the host rereads most of the model from SSD every token (23 GB per token). With 40 layers on the Macs the host
  streams a quarter as many layers and reads 2.8 GB per token, and decode is 5.4 times faster.
- Only one interleaved pair is in the table. The second alone row (`d70-g14A-2`, kept in the index) was stopped by the
  commit guard after its warmups (0.180, 0.179, 0.181 tok/s, consistent with row 1). The second pooled row was
  refused because an explicit layout cannot name a host CPU stage (`index_d70_refused_layout_cpu_stage.jsonl`). A
  second pair is queued at the end of the sweep.

#### R7.3 gpt-oss-120b (MXFP4, 59.03 GiB MoE): 128-token prompt, 64-token output, 3 reps

Alone: moe_offload with all 36 layers on CUDA0 and the experts file-backed on the host CPU. Pooled: 6 layers on each
Mac, 24 on CUDA0, the remaining experts file-backed on the host.

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| g120-g14A-1 | 40.1 | 3.61, 3.88, 3.99 | 3.88 | 3.92 | 251 |
| g120-g14P2-1 | 112.2 | 7.59, 9.07, 9.56 | 9.07 | 28.6 | 7.5 |

Pooled vs alone (3 vs 3 reps, one row each):

| decode | prefill | first token | 64-token answer | SSD MiB per decoded token |
|---|---|---|---|---|
| 8.74 vs 3.83, 2.28 [2.00, 2.54] | 29.8 vs 3.84, 7.76 [6.95, 8.86] | 4.8 vs 34.2 s, 0.14 [0.12, 0.16] | 11.7 vs 49.9 s, 0.23 [0.21, 0.26] | 6.4 vs 238, 0.027 [0.017, 0.034] |

- Alone decode is close to R2 (G14 alone 2.89 to 3.72 tok/s per rep there, 3.61 to 3.99 here). Pooled decode is not:
  R2's three Macs on the building Wi-Fi (RTT 12 to 134 ms, 6.6 to 8 MiB/s) reached 3.15 to 5.66 tok/s, while two
  Macs on the home network (RTT 5 to 6 ms) reach 7.59 to 9.56. The pooled row read 7.5 MiB per token from SSD
  against 251 alone. The networks differ in both RTT and bandwidth, so this pair alone cannot say which one matters.

#### R7.4 Qwen2.5-VL 72B Instruct (Q4_K_M, dense, text only): 64-token prompt, 32-token output, 2 reps

Alone: 64 of 80 layers streamed on the host CPU, 16 on CUDA0. Pooled: 28 on the host CPU, 18 on each Mac, 16 on
CUDA0 (tier stream in both). These rows send text only; the projector is not loaded (image input is on the
`feat/vlm-mmproj` branch, not in this gateway).

| run | load s | decode tok/s per rep | decode median | prefill median | SSD MiB / token |
|---|---|---|---|---|---|
| vl72-g14A-1 | 34.1 | 0.141, 0.141 | 0.141 | 4.64 | 26896 |
| vl72-g14P2-1 | 100.2 | 0.367, 0.364 | 0.365 | 7.22 | 8956 |

Pooled vs alone (2 vs 2 reps, one row each):

| decode | prefill | first token | 32-token answer | SSD MiB per decoded token |
|---|---|---|---|---|
| 0.365 vs 0.141, 2.59 [2.57, 2.61] | 7.22 vs 4.64, 1.56 [1.51, 1.61] | 9.1 vs 13.7 s, 0.66 [0.64, 0.69] | 93.9 vs 233.6 s, 0.40 [0.40, 0.41] | 8956 vs 26896, 0.33 |

- Same shape as the 70B, smaller gain: this plan left 28 layers on the host CPU (21 for the 70B), so the host still
  read 9 GB from SSD per token, a third of the alone figure rather than an eighth.

#### R7.5 gpt-oss-120b with 4 and 8 concurrent chats on the laptops

Same model, prompt (128 tokens) and output (64 tokens) as R7.3, with N server slots (`-np N -kvu`), 3 batches per
row, one interleaved alone/pooled pair per N (`index_g120n4.jsonl`, `index_g120n8.jsonl`). The pooled rows used the
pinned R7.3 layout (6 layers on each Mac, 24 on CUDA0). Aggregate decode is counted over the all-active window; the
ratios resample batches. The N=1 column is R7.3.

| N | alone aggregate tok/s | pooled aggregate tok/s | pooled vs alone | step ms alone, pooled | SSD MiB per step alone, pooled |
|---|---|---|---|---|---|
| 1 | 3.83 | 8.74 | 2.28 [2.00, 2.54] | 262, 116 | 238, 6.4 |
| 4 | 8.10 | 11.85 | 1.46 [1.30, 1.67] | 494, 341 | 354, 59 |
| 8 | 12.19 | 16.45 | 1.35 [1.25, 1.48] | 660, 489 | 438, 75 |

- The pooled laptops stay ahead at every N, but the lead shrinks: batching shares each expert read across N chats,
  which helps the SSD-bound alone host more than the pooled one, whose per-step cost includes a round trip to each Mac.
- At N=8 the pool serves 16.4 tok/s in aggregate, about 2.1 tok/s per chat.
- The alone N=4 row overlapped about 30 s of an orphaned model download from the campaign restart (killed during its
  load, before any measured batch).

### R8. Every machine at once: the GX10 host with the G14, mac-a and mac-b, 2026-10-25

The GX10 hosts; the G14 (CUDA0, 9 GiB cap) and mac-a and mac-b (MTL0, 11.5 GiB caps) join as peers over Iroh, all
three direct on the home network (5 to 16 ms RTT before each pooled row, up to 52 ms after). Peers seed their own layers from Hugging Face before each pooled
row. Alone rows are the GX10 with no peers. Indexes: `paper-2026-10-25-gx10sweep/index_mav.jsonl` (Maverick alone),
`index_allpool.jsonl` (pooled), and for the 1T alone rows `paper-2026-10-24-gx10conc/index_gx10_conc2.jsonl` (R6.y,
same day). One row per side per N; ratios are pooled over alone with 95% bootstrap intervals; N > 1 resamples batches.

#### R8.1 Llama 4 Maverick 400B (UD-Q2_K_XL, 142.17 GiB, 17B active): 128-token prompt, 64-token answer

| N | GX10 alone, tok/s | all machines, tok/s | pooled vs alone |
|---|---|---|---|
| 1 | 9.79 (9.19, 9.65, 10.52) | 4.28 (4.05, 4.60, 4.19) | 0.44 [0.40, 0.48] |
| 4 (aggregate) | 29.86 | 9.83 | 0.33 [0.31, 0.35] |
| 8 (aggregate) | 43.36 | 15.07 | 0.35 [0.33, 0.37] |

At N = 1: prefill 60.4 vs 32.8 tok/s, 0.54 [0.35, 0.79]; first token 2.41 vs 3.94 s, 1.64 [1.26, 2.85]; full answer
8.86 vs 18.71 s, 2.11 [1.96, 2.28]; SSD 53.1 vs 13.8 MiB per token, 0.26 [0.17, 0.45].

- The model is larger than the GB10's 121.6 GiB, yet alone it read only 53 MiB per token from SSD: likely because
  only 17B are active and the hot expert pages stay cached, the GX10 alone is fast, and every peer adds a Wi-Fi round trip
  per token. Pooling made this model slower on every speed metric. For a 400B model at about 10 tok/s, run it on the
  GX10 alone.
- Not in the table: a run-together plan request for this model, with the peers allowed less memory than in these rows,
  got a gpu_disk plan (18 layers file-backed on the GX10's CPU, 26 on CUDA0, 1 to 2 on each peer), most likely because
  moe_offload could not give every peer a whole layer (that request's candidate list was not saved). Loading it swamped
  the GX10 (SSH stopped answering) until a manual reboot. After the R6.x planner fix that request is refused: on the
  GX10 a model larger than its pool runs as moe_offload or not at all. In the test with these rows' peer budgets (9,
  10.67 and 10.67 GiB) the planner still picks moe_offload with 3, 2 and 2 peer layers and 41 on CUDA0, as the
  05:39 UTC run loaded it.

#### R8.2 Kimi K2.6 1T (IQ1_S, 193.16 GiB, 32B active): 64-token prompt, 32-token answer

| metric | GX10 alone | all machines | pooled vs alone |
|---|---|---|---|
| decode tok/s, N = 1 | 1.779 | 1.774 | 1.00 [0.96, 1.03] |
| prefill tok/s, N = 1 | 2.18 | 3.13 | 1.44 [1.40, 1.48] |
| first token, N = 1 | 27.0 s | 20.2 s | 0.75 [0.72, 0.78] |
| 32-token answer, N = 1 | 44.4 s | 37.7 s | 0.85 [0.82, 0.88] |
| SSD MiB per token, N = 1 | 657 | 334 | 0.51 [0.42, 0.60] |
| aggregate decode tok/s, N = 8 | 4.64 | 4.99 | 1.08 [0.98, 1.19] |

- With every peer direct, the 1T decodes at parity pooled (R6 measured 0.86x with four peers, all direct) and answers
  15% sooner, because prompt reading is 1.44x faster and half the SSD reads go away. At 8 chats the interval includes
  no gain.
- Caveat: the alone rows ran about 4 h earlier in another gateway session (87.0 GiB of CUDA0 free at its start,
  116.9 GiB in the pooled session), not interleaved, which may inflate the first-token and SSD gains.

#### R8.3 DeepSeek-V3.1 671B (UD-TQ1_0, 158.8 GiB, 37B active): 64-token prompt, 32-token answer

First, a single pooled row against R6.y's alone row from another session:

The first pooled row (`ds671-n1-gx3-1`) loaded for 1071 s and then failed while reserving compute buffers: mac-b's
worker exited (code -6, `peer d4e65f83 lost` in run.json) and the host's engine reported `ggml-rpc.cpp:682: Remote RPC
server crashed or returned malformed response` during the warmup decode (peer buffers 9.66, 8.40 and 10.10 GiB of
weights plus 0.40 to 0.78 GiB of compute buffer each). The retry with both Macs at 10 GiB caps (`ds671-n1-gx3-2`)
completed: the planner put 6 layers on the G14, 3 on each Mac and 49 on CUDA0 (moe_offload), seeding took 10 s with
the caches warm and the load 301 s. The alone row is R6.y's `ds671-n1-gxA-1`, about 5 h earlier in another gateway
session, not interleaved.

| metric | GX10 alone | all machines | pooled vs alone |
|---|---|---|---|
| decode tok/s | 1.79 (1.82, 1.76) | 2.23 (2.16, 2.31) | 1.25 [1.18, 1.31] |
| prefill tok/s | 2.77 | 7.30 | 2.64 [2.22, 3.07] |
| first token | 23.5 s | 9.3 s | 0.40 [0.32, 0.47] |
| 32-token answer | 40.8 s | 23.2 s | 0.57 [0.51, 0.63] |
| SSD MiB per token | 686 | 145 | 0.21 [0.20, 0.23] |

- That single pair is across sessions. The interleaved check below does not reproduce the decode gain.

Interleaved check, same session, run with the bench kit (`bench/hcbench.py plan --order APAP`, then `run --yes`):
GX10 alone and pooled rows alternating, A P A P, 2 reps each, Macs at 10 GiB caps, all peers direct
(`paper-2026-10-25-e2e/index_ds671e.jsonl`).

| metric | GX10 alone | all machines | pooled vs alone |
|---|---|---|---|
| decode tok/s | 1.761 (1.733, 1.822, 1.712, 1.778) | 1.771 (1.837, 1.936, 1.743, 1.571) | 1.01 [0.93, 1.08] |
| prefill tok/s | 2.65 | 4.21 | 1.59 [1.47, 1.71] |
| first token | 25.0 s | 15.8 s | 0.63 [0.58, 0.68] |
| 32-token answer | 42.6 s | 33.4 s | 0.78 [0.75, 0.84] |
| SSD MiB per token | 698 | 261 | 0.37 [0.36, 0.39] |

- Same session, interleaved, the 671B decodes at parity across every machine and answers 22% sooner, from faster
  prompt reading and a third of the SSD reads. The earlier 1.25x was a session effect (the alone row ran about 5 h
  earlier, after a different gateway start). Loads: alone 243 and 252 s, pooled 313 and 317 s.

#### R8.4 Smaller models on the GX10 alone, 1, 4 and 8 chats (`index_gm27.jsonl`, `index_qwq32.jsonl`)

| model | N = 1 tok/s | N = 4 aggregate (batch median) | N = 8 aggregate (batch median) |
|---|---|---|---|
| Gemma 3 27B (Q4_K_M) | 12.34 | 45.35 | 84.99 |
| QwQ 32B (Q4_K_M, reasoning) | 10.64 | 39.20 | 75.26 |

### R1.3 Operational findings

- Back-to-back runs race the planner: D3 was refused because it planned 4 s after A3 ended, while the old llama-server
  still held the GPU (planner saw 0.8 GiB GPU and 281 MiB RAM free on the G14). The driver now waits for the GPU to
  drain plus 20 s of heartbeats before each run.
- Cold load over this Wi-Fi: 19.3 GiB to three Macs took 2289 s (8.6 MiB/s effective, link-bound). With the Macs'
  RPC caches warm the same load took 150 to 170 s, mostly the host hashing the peer tensors it would have sent.
- A Mac downloads from Hugging Face at 45 MB/s on the same Wi-Fi (300 MiB range in 7.0 s), about 5x the host-to-Mac
  link, which is why the next runs seed peer caches from the origin instead of the host.
