# Peer SSD tier: candidate layouts for Qwen3-235B-A22B tonight

2026-10-23 (UTC), branch `codex/iroh-transport`. Arithmetic only, nothing here has run, and hcpeer has never run on a
Mac (`hcpeer-macos.md`), so every Mac-side number is an estimate. Tags as in `peer-ssd-tier-design.md`: [V] read from
a file or source, [M] measured tonight (`run.json`), [E] estimate from section 2, [A] assumption. Source aliases
`RPC/` and `LL/` are the read-only b11770 copies named in the design doc.

Hypothesis under test: each Mac also holds whole layers on its CPU device, file-backed from its own SSD by hcpeer
(`HC_PEER_MMAP=1`), so the G14's file-backed expert set shrinks toward its RAM and each machine's free RAM caches its
own shard. Short answer from the arithmetic: a moderate version wins (C1, about 10 layers on each 200 GB Mac); the
strong version (G14 set under 20 GiB) does not, because each Mac's 3-6 GiB of page cache cannot hold a 12-18 GB shard.

## 1. Inputs

### 1.1 Bytes per layer (GGUF header) [V]

`gguf.read_tensor_table` over both shards (1131 tensors; script and output in
`C:\Users\operator\AppData\Local\Temp\claude\C--\9e2c80d5-50ca-460d-8443-8db45359fcde\scratchpad\candidates\`,
`qwen_layers.py` and `qwen_layers.json`). Each layer has 3 expert tensors (`ffn_{gate,up}_exps` Q2_K, `ffn_down_exps`
Q3_K or Q4_K), 2 dense tensors over 10 MiB (`attn_q`, `attn_output`, Q4_K or Q5_K, mapped by hcpeer) and 7 small ones
(`attn_k`, `attn_v`, `ffn_gate_inp` F32, norms; sent whole into anonymous memory). Every tensor over 10 MiB is a
multiple of 16 KiB, so hcpeer maps it with no partial page. 128 experts, 8 used: f = 0.0625.

| layers | n | experts W (B) | dense > 10 MiB, D (B) | small, d (B) |
|---|---|---|---|---|
| all others | 71 | 874,512,384 | 37,748,736 | 4,490,240 |
| 1, 3, 24, 58, 59, 71, 80, 90, 93 | 9 | 981,467,136 | 37,748,736 | 4,490,240 |
| 2, 11, 13, 23 | 4 | 981,467,136 | 46,137,344 | 5,014,528 |
| 35, 49, 57, 64, 70, 74, 77, 88 | 8 | 874,512,384 | 46,399,488 | 4,490,240 |
| 83, 89 | 2 | 981,467,136 | 46,399,488 | 4,490,240 |

Totals match `run.json` `layer_bytes` and `layer_expert_bytes`: experts 83.81 GB, dense 4.09 GB. A layer is about
0.92 GB, 95% experts.

### 1.2 Tonight's baseline [M]

Runs D4, D5, D6 (`live-host/results/capacity/20261022T233750Z-*`, `20261022T234753Z-*`, `20261023T000553Z-*`), all
`--device RPC0,RPC1,RPC2,CUDA0 --tensor-split 7,7,8,73`:

| device | node | layers | model buffer | KV | compute |
|---|---|---|---|---|---|
| RPC0 (MTL0) | mac-azim | 0-6 | 6434 MiB wired | 56 MiB | 172 MiB |
| RPC1 (MTL0) | mac-a | 7-13 | 6341 MiB wired | 56 MiB | 157 MiB |
| RPC2 (MTL0) | mac-b | 14-21 | 6994 MiB wired | 64 MiB | 157 MiB |
| CUDA0 | G14 | 22-93 + head; experts of 82, 84-88, 91, 92 | 10150 MiB | 576 MiB | 172 MiB |
| CPU_Mapped | G14 | experts of 22-81, 83, 89, 90, 93 (64 layers) | 54730 MiB (W0 = 57.04 GB + embd) | | 156 MiB |

Steady state (9 reps each): 3 Macs 1.396 tok/s, T0 = 716.2 ms/token, SSD 296 MiB per decoded token; G14 alone
(A3-A5) 1.294 tok/s (772.7 ms), 647 MiB with W = 78.56 GB. Iroh RTT in D4-D6: 8-37 ms.

## 2. Model and coefficients

Design section 6.3 per stage, anchored to the measured baseline so the unchanged Metal stages and GPU need no model:

```
S(W, C) = f * W * exp(-C / (kappa * W))                      SSD bytes per token of a file-backed expert set
dG14    = c_g14 * f * (W0 - W1) + r_g14 * (S(W0, C_g14) - S(W1, C_g14))
stage_i = h + c_mac * (D_i + d_i + f * W_i) + r_mac * S(W_i, C_i)
C_i     = P - (d_i + n_i * 8 MiB + 160 MiB) - D_i            Mac page cache left for experts (D assumed hot)
T       = T0 - dG14 + sum(stage_i)                           tok/s = 1000 / T (ms)
```

| coefficient | value | provenance |
|---|---|---|
| T0 | 716.2 ms | [M] D4-D6 mean |
| C_g14 | 19.54 GB | [M] whole-run system cache peak, run `20261022T232815Z`; design section 11 uses 19.5 GB |
| kappa | 0.140 | [M fit] solves S = 296 MiB at W0 = 57.04 GB; the alone runs give 0.126; the design's gpt-oss fit is 0.176 |
| r_g14 | 0.90 ms/MB | [M fit] G14 fault rate, gpt-oss (design 6.3) |
| r_mac | 0.90 ms/MB | [A] never measured on an M4 Air |
| c_g14 | 12.5 ms/GB | [A] planner prior (`run.json` explain assumptions) |
| c_mac | 25 ms/GB | [A] design 6.3 |
| P | 5 GiB (3-6 in section 4.3) | [A] given for tonight; see the caveat below |
| KV, compute | 8 MiB per layer, 160 MiB per device | [M] RPC buffers in 1.2 |
| h | 15 and 30 ms per added stage | [A] from the measured 8-37 ms RTT; 89 ms = [M fit] below |

Checks on the coefficients:

- kappa = 0.14 gives 294 MiB at the baseline (measured 296). It rose as W fell (0.126 at 78.6 GB, 0.140 at 57.0 GB);
  at 0.15 the G14 SSD in C1 is about 15 MiB higher and C1 about 15 ms slower.
- **Per Mac stage tonight is about 97 ms, not RTT plus compute.** dG14 from alone to 3 Macs is
  12.5 x 0.0625 x 21.52 + 0.90 x 368.0 = 16.8 + 331.2 = 348.1 ms, so the host part of the baseline is
  772.7 - 348.1 = 424.6 ms and the three Metal stages take 716.2 - 424.6 = 291.6 ms, 97 ms each. Metal compute for
  7.3 layers is about 8 ms at the planner's 10.85 ms/GB prior, which leaves about 89 ms per stage that is neither
  compute nor the 8-37 ms RTT. Its cause is open (design S0). The h = 89 ms column is the case where an added CPU stage
  pays what a Metal stage pays tonight.
- **P may be nearer 3 GiB.** The gateway read 10.29, 10.43 and 10.75 GB available on azim, a and b before D5's load;
  the Macs then wired 6.99, 6.87 and 7.57 GB (model + KV + compute), which leaves 3.0-3.3 GiB. psutil may undercount
  reclaimable file pages on macOS, so measure it on the Mac during a decode (section 5).

## 3. Common to all candidates

- **Peers.** Each Mac with a CPU span restarts its worker with `-d MTL0,CPU`, hcpeer v4 installed and
  `HC_PEER_MMAP=1` with `-c` (`hypercompute capacity worker --node <id> --devices MTL0,CPU --threads 4`; the agent
  takes `HC_PEER_MMAP` at join, `agent/join.py:185`). Gate before any load: the banner says `hcpeer mmap : on (v4`.
  Without it the span lands in anonymous memory, 8.5-18 GiB next to 7.5 GiB wired on a 16 GiB Mac, which is swap. Run
  `build_hcpeer_mac.py verify`, `selftest`, `smoke`, `install`, `smoke` first (`hcpeer-macos.md`): none has run on a
  Mac.
- **Threads.** rpc-server defaults to `hardware_concurrency()/2` = 5 on a 10-core M4 (design F13), one more than the 4
  performance cores; use 4 [A].
- **Seed first** (`hypercompute capacity seed --node <id> --model qwen3-235b-a22b-instruct-2507-ud-q2_k_xl --layers
  A-B`, inclusive), so the load maps instead of uploading over the 6.6-8 MiB/s host link. Mac to Hugging Face runs
  13-45 MB/s.
- **Device order: all Metal stages first, then the CPU stages, so no Mac's MTL0 sits next to its own CPU.** Names are
  `RPC0..` in `--rpc` order, then each server's `-d` order (design 6.2).
  - Across servers the client's `cpy_tensor` returns false without a message (`RPC/ggml-rpc.cpp:937-940`), so a
    boundary costs the one blocking `GET_TENSOR` it costs tonight (`LL/ggml_src_ggml-backend.cpp:628-649`).
  - Within one server it first sends a synchronous `COPY_TENSOR` (`RPC/ggml-rpc.cpp:947`), which fails both ways
    between Metal and CPU (Metal shared buffers need a Metal source, `RPC/ext/ggml_src_ggml-metal_ggml-metal.cpp:76-78`;
    CPU buffers need a host source, `LL/ggml_src_ggml-backend.cpp:2507-2513`), and then does the same get and set: 2
    RTT. The 30 ms column also covers a 15 ms RTT in that order.
  - `--device` keeps the given order (`RPC/ext/common_arg.cpp:1122-1142`) and an RPC device reports type GPU
    (`RPC/ggml-rpc.cpp:2851-2853`), so a remote CPU device is accepted anywhere in the list.
- **Host.** `-ot` keeps only the CUDA0 layers' experts on the G14 CPU (patterns from `gguf.expert_override_pattern`;
  the baseline pattern reproduces tonight's argv exactly). The GPU expert set {82, 84-88, 91, 92} is kept. The VRAM the
  moved layers free (dense + KV, 0.96-2.01 GiB) is left unused here; a planner would put 1-2 more expert layers there.
- **Hand layouts.** The planner does not model a file-backed peer CPU device yet (`hcpeer-macos.md`, requested
  changes), so these are hand-written argument sets.

## 4. Candidates

Metal stages are as in 1.2 in every candidate (mac-azim 0-6, mac-a 7-13, mac-b 14-21, 6.2-6.8 GiB of weights wired
each, nothing file-backed). Device names: a Mac with `-d MTL0,CPU` gets two consecutive names.

### 4.1 Summary [E]

| | baseline [M] | C1 conservative | C2 | C3 two-Mac aggressive | C4 aggressive |
|---|---|---|---|---|---|
| CPU spans (device order) | none | a 22-31, b 32-41 | a 22-35, b 36-49 | a 22-42, b 43-63 | azim 22-35, a 36-49, b 50-63 |
| `--device` | `RPC0,RPC1,RPC2,CUDA0` | `RPC0,RPC1,RPC3,RPC2,RPC4,CUDA0` | as C1 | as C1 | `RPC0,RPC2,RPC4,RPC1,RPC3,RPC5,CUDA0` |
| `--tensor-split` | 7,7,8,73 | 7,7,8,10,10,53 | 7,7,8,14,14,45 | 7,7,8,21,21,31 | 7,7,8,14,14,14,31 |
| G14 CPU expert layers | 64 | 44 (42-81, 83, 89, 90, 93) | 36 (50-81, ...) | 22 (64-81, ...) | 22 (64-81, ...) |
| G14 file-backed W1 | 57.04 GB | 39.33 GB (36.63 GiB) | 32.34 GB (30.12 GiB) | 19.88 GB (18.52 GiB) | 19.88 GB (18.52 GiB) |
| G14 miss, SSD MiB/token | 0.087, 296 | 0.029, 67.5 | 0.013, 25.8 | 0.001, 1.1 | 0.001, 1.1 |
| Mac miss | | 0.024, 0.022 | 0.077, 0.073 | 0.205, 0.205 | 0.077, 0.073, 0.076 |
| Mac SSD MiB/token | 0 | a 12.7, b 11.3 | 57.1, 53.4 | 227.2, 227.1 | 57.1, 53.4, 56.8 |
| all machines, MiB/token | 296 | 91.5 | 136.3 | 455.4 | 168.4 |
| added stages | | 2 | 2 | 2 | 3 |
| dG14 (CPU + faults) | | 13.8 + 214.1 = 228.0 ms | 19.3 + 253.5 = 272.8 | 29.0 + 276.9 = 305.9 | 29.0 + 276.9 = 305.9 |
| Mac compute + faults | | 71.9 ms | 173.1 | 532.0 | 261.2 |
| tok/s at h = 15 / 30 ms | 1.396 | **1.69 / 1.61** | 1.55 / 1.48 | 1.03 / 1.00 | 1.40 / 1.31 |
| tok/s at h = 89 ms (fit) | | 1.35 | 1.26 | 0.89 | 1.07 |
| break-even h | | 78 ms | 50 ms | none (-113) | 15 ms |

Break-even h = (dG14 - Mac compute and faults) / added stages: the per-stage cost at which the candidate only ties the
baseline. A sweep over equal spans (P = 5 GiB) peaks at 8-10 layers per Mac: two Macs give 1.68-1.69 tok/s at 15 ms,
three Macs x 8 layers give 1.74 at 15 ms but the same 1.61 at 30 ms, and 12 or more layers per Mac fall below C1.

### 4.2 Bytes per stage

C1, the arithmetic in full (mac-a 22-31 includes the 981 MB expert layers 23 and 24; mac-b 32-41 has none):

| stage | resident | file-backed (mapped) | page cache for it | miss | SSD/token | compute | faults |
|---|---|---|---|---|---|---|---|
| mac-a CPU (RPC2) | anonymous 283 MiB (d 43 + KV 80 + compute 160) | 9.34 GB (8.70 GiB): W 8.96 GB, D 368 MiB | 4.69 GB | 0.024 | 12.7 MiB | 24.8 ms | 12.0 ms |
| mac-b CPU (RPC4) | anonymous 283 MiB | 9.13 GB (8.50 GiB): W 8.75 GB, D 368 MiB | 4.69 GB | 0.022 | 11.3 MiB | 24.4 ms | 10.7 ms |
| G14 CUDA0 42-93 + head | VRAM 9328 + KV 416 MiB | 39.33 GB of experts (G14 CPU) | 19.54 GB | 0.029 | 67.5 MiB | | |

- mac-a: C = 5 GiB - 297 MB - 386 MB = 4686 MB; f W = 559.9 MB; miss = exp(-4686 / (0.14 x 8959)) = exp(-3.736) =
  0.024; S = 13.4 MB; compute = 25 x (0.386 + 0.045 + 0.560) = 24.8 ms; faults = 0.90 x 13.4 = 12.0 ms.
- G14: miss = exp(-19535 / (0.14 x 39330)) = exp(-3.548) = 0.0288; S1 = 2458 MB x 0.0288 = 70.8 MB, against the
  model's 308.7 MB at W0; dG14 = 12.5 x 0.0625 x 17.70 + 0.90 x (308.7 - 70.8) = 13.8 + 214.1 = 228.0 ms.
- T = 716.2 - 228.0 + 71.9 + 2h = 590.1 ms (h = 15), 620.1 ms (30), 738.1 ms (89).

Other candidates, per CPU stage (same formulas; script `calc.py` next to `qwen_layers.py`):

| candidate | per-Mac span | anonymous | file-backed (mapped) | page cache | VRAM model + KV on G14 |
|---|---|---|---|---|---|
| C2 | 14 layers | 332 MiB | 13.00 / 12.78 GB (12.11 / 11.90 GiB) | 4.47 GB | 8997 + 352 MiB |
| C3 | 21 layers | 418 MiB | 19.39 / 19.39 GB (18.06 GiB each) | 4.12 GB | 8425 + 240 MiB |
| C4 | 14 layers | 332 MiB | 13.00 / 12.78 / 12.99 GB (12.11 / 11.90 / 12.10 GiB) | 4.47 GB | 8425 + 240 MiB |

Disk: mac-a and mac-b (100 GiB cap, about 175 GiB free) hold any of these. mac-azim (C4 only) has
min(30 - 16 cap, 46 - 20 floor) = 14 GiB of headroom for 12.11 GiB; hcpeer v4 copies anything past the cap into
anonymous memory, which on azim is swap. Seeding at 45 / 13 MB/s per Mac: C1 3.5 / 12.0 min, C2 and C4 4.8 / 16.7
min, C3 7.2 / 24.9 min, if the shared uplink gives each Mac that rate. Load: tonight 19.3 GiB of peer bytes added
80-130 s of host hashing over the 44-54 s G14-alone load; C1's 17.2 GiB more adds about 70-115 s [E].

### 4.3 Sensitivity [E]

| case | C1 Mac SSD MiB | C1 tok/s at 15 / 30 / 89 ms | C1 break-even | C4 tok/s at 15 / 30 ms | C4 break-even |
|---|---|---|---|---|---|
| P = 3 GiB (gateway reading) | 71, 66 | 1.44 / 1.38 / 1.18 | 25 ms | 0.91 / 0.87 | none |
| P = 4 GiB | 30, 27 | 1.61 / 1.54 / 1.30 | 62 ms | 1.17 / 1.12 | none |
| P = 5 GiB (central) | 13, 11 | 1.69 / 1.61 / 1.35 | 78 ms | 1.40 / 1.31 | 15 ms |
| P = 6 GiB | 5, 5 | 1.73 / 1.65 / 1.38 | 85 ms | 1.55 / 1.45 | 39 ms |
| kappa_mac = 0.35 (design's pessimistic) | 120, 113 | 1.27 / 1.22 / 1.07 | none | 0.77 / 0.74 | none |

At P = 3 GiB the best two-Mac span is 6 layers each (a 22-27, b 28-33): 1.58 / 1.51 tok/s, break-even 56 ms.

Not modelled: prefill. A 64-token prompt routes to 1 - (120/128)^64 = 98.4% of each layer's experts, so in C1 each
Mac reads about W - C = 4.3 GB per prompt (about 3.8 s at 0.9 ms/MB) while the G14 set falls by 17.7 GB. Record it.

## 5. Run first, and what falsifies the hypothesis

**Run C1 first.** It has the widest margin (break-even 78 ms against 50, 15 and none), it beats the baseline's best
rep (1.53 tok/s) at h = 30 ms, it touches only the two 200 GB Macs and leaves the disk-tight mac-azim as it is, it seeds
the least (about 9.3 GB per Mac), and it is the smallest first exposure for hcpeer on a Mac. C3 is predicted to lose
outright, and C4 is at best parity: the strong form of the hypothesis is predicted false, because a 12-18 GB shard
does not fit a 3-6 GiB Mac page cache.

Before loading C1, read P on mac-a and mac-b with the Metal layers loaded (free plus file-backed pages from `vm_stat`,
times 16 KiB). If it is under 4 GiB, run a 22-27, b 28-33 instead, with `--tensor-split 7,7,8,6,6,61`.

Same protocol as D4-D6 (64 prompt and 64 output tokens, `ignore_eos`, temperature 0, seed 7, 1 warmup, 3 measured
reps), then judge in this order:

1. **Valid run.** `[hcpeer] mapped` totals about 8.70 and 8.50 GiB, only small tensors in `copying instead`, swap
   unchanged on both Macs, rpc-server footprint about the anonymous estimate plus the wired Metal span. If weights
   were copied, the run does not test the hypothesis.
2. **G14 side.** SSD per decoded token falls from 296 to about 68 MiB. Above about 150 MiB the G14 cache model is
   wrong; stop and do not run C2-C4.
3. **Mac side.** Pageins per decoded token (the `vm_stat` delta times 16 KiB) about 11-13 MiB per Mac. Above 60 MiB
   (kappa_mac about 0.24 or more, or P under about 3 GiB) the Mac caches do not hold their shards and every larger
   candidate is worse.
4. **Outcome.** The hypothesis is falsified for this network if the decode median is at or below 1.40 tok/s with
   checks 2 and 3 in range: then each added stage costs more than the 78 ms break-even, the limit is the per-stage
   cost (design S0), not the disk, and C2-C4 (break-even 50 ms or less) are not worth running tonight. Report the
   measured per-stage cost as h = (T_C1 - T0 + dG14_measured - Mac compute and faults) / 2, with dG14_measured from
   the measured G14 SSD drop.
5. **Next.** At 1.55 tok/s or more, try three Macs x 8 layers (1.74 / 1.61 tok/s at 15 / 30 ms) rather than C4.
   Between 1.40 and 1.55, keep C1 and measure the hop before adding stages.

Not done here: no Mac ran hcpeer, no layout ran, and P, r_mac, c_mac and the per-stage cost of an added CPU stage are
all unmeasured. The predictions are as good as those four numbers.
