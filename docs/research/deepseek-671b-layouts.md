# DeepSeek-V3.1 671B (UD-TQ1_0): candidate layouts for tonight

2026-10-23, branch `codex/iroh-transport` at 9fd2456. Arithmetic only: nothing has run on this model and its file is
still downloading. Tags: [V] read from the GGUF header (exact), [M] measured on this cluster (Qwen3-235B and
gpt-oss-120b runs in `PAPER-RUNS-2026-10-22.md` and `peer-ssd-tier-candidates.md`), [F] a fit to those measurements,
[E] estimate from section 1.3, [A] assumption or given. Scripts and outputs:
`C:\Users\operator\AppData\Local\Temp\claude\C--\9e2c80d5-50ca-460d-8443-8db45359fcde\scratchpad\ds671\` (`hdr.py` reads
the header, `calc.py` and `report.py` do the arithmetic below).

Short answer [E]: every candidate is bound by page faults on the G14. A (G14 alone) about 0.48 tok/s; B (three Metal
stages) 0.66 / 0.58 tok/s at 30 / 97 ms per stage; C (B plus 4-layer hcpeer CPU spans on mac-a and mac-b) 0.85 / 0.66.
Run A first (section 5).

## 1. Inputs

### 1.1 Bytes per layer [V]

`unsloth/DeepSeek-V3.1-GGUF` at feb73eff, `DeepSeek-V3.1-UD-TQ1_0.gguf`, 170,499,768,256 bytes, read from its first
16 MiB: GGUF v3, 1086 tensors, data from byte 5,308,352. Sizes come from dims and the ggml block table; the last tensor
ends exactly at the file size and no tensor has alignment padding, so every size is exact. Two independent readers
agree per layer and per tensor kind: `ds671/hdr.py` (pure Python, own type table) and `ds/v31_layers.py` (the repo's
`gguf.tensor_nbytes`, read-only).

- Routed experts 160.365 GB (149.35 GiB) in layers 3-60; the rest of the blocks 8.848 GB; output head 0.760 GB (Q6_K);
  token embeddings 0.521 GB (Q4_K, kept in G14 RAM).
- Non-expert part: layers 0-2 (dense FFN) 310.7 MB each; MoE layers 127.9-171.5 MB, of which 48-63 MB is tensors of
  10 MiB or less (attn_k_b, attn_v_b, attn_kv_a_mqa, attn_q_a, the F32 router, most shared-expert tensors). RPC always
  sends those whole: 3.333 GB over all blocks. The larger ones (attn_q_b, attn_output, some shexp) total 5.515 GB.
- 256 routed experts, 8 used: f = 1/32. One expert of an IQ1_S layer is 8.6 MB.

| routed experts per layer | gate, up / down | n | layers |
|---|---|---|---|
| 0 | dense FFN | 3 | 0-2 |
| 2.202 GB | IQ1_S / IQ1_S | 30 | 3, 5, 7, 9-11, 16, 17, 19, 24-27, 31, 32, 34, 36, 37, 39-41, 43, 44, 48-50, 52, 53, 55, 58 |
| 2.437 GB | IQ1_S / IQ2_XXS | 4 | 29, 47, 56, 57 |
| 3.083 GB | IQ1_S / IQ3_S | 5 | 4, 18, 20, 22, 45 |
| 3.376 GB | IQ2_XXS / IQ3_XXS | 7 | 28, 30, 38, 42, 46, 51, 54 |
| 3.464 GB | IQ1_S / IQ4_XS | 2 | 12, 13 |
| 3.553 GB | IQ2_XXS / IQ3_S | 2 | 6, 35 |
| 3.934 GB | IQ2_XXS / IQ4_XS | 8 | 8, 14, 15, 21, 23, 33, 59, 60 |

### 1.2 Coefficients

| coefficient | value | tag |
|---|---|---|
| G14 GPU budget; RAM; NVMe | 11.1 GiB; 31 GiB; about 3 GB/s | [A] given |
| KV per layer | 8.5 MiB: ctx 4096 x (576 + 512) x 2 B, the planner's formula with MLA's one KV head | [E] |
| compute buffer | 208 MiB per device: 96 MiB + 512 x 7168 x 32 B (`planner.py` compute_est) | [E] |
| overhead | CUDA 450 MiB; Metal 256 MiB + 64 MiB RPC staging (`planner.py` priors) | [E] |
| G14 file cache C_G14 | 19.54 GB, system-cache peak of Qwen run `20261022T232815Z`, assumed unchanged | [M], [A] here |
| kappa | 0.14 from Qwen3-235B (f = 1/16); 0.176 from gpt-oss-120b (also f = 1/32) | [F] |
| fault rate | G14 0.90 ms/MB [F]; Mac 0.90 ms/MB [A] | |
| compute | CUDA 2.64, G14 CPU 12.5, Mac CPU 25 ms per GB read | [A] priors |
| host residual R | 87.7 ms: Qwen alone measured 772.7 ms/token, the priors give 685.0 (13.0 + 61.4 + 610.6) | [F] |
| per Mac stage h | 97 ms, a whole Metal stage on tonight's Wi-Fi (RTT 8-37 ms, Metal compute about 8) [M]; 30 ms [A] | |
| Metal caps | mac-a and mac-b 10.67 GiB, mac-azim 11.84 GiB | [A] given |
| Mac page cache P | 4.5 GiB left beside a near-cap Metal stage (3 and 5 in section 3) | [A] |
| links | hotspot 10-30 MB/s [A]; Mac to Hugging Face 13-45 MB/s [M] | |

### 1.3 Model

```
S(W, C)  = f * W * exp(-C / (kappa * W))                 SSD bytes per token of a file-backed expert set
T_G14    = 2.64 * GPU_GB + 12.5 * f * W_GB + 0.90 * S_MB + R     GPU_GB = host dense + head + f * GPU experts
T_metal  = h                                             the 97 ms was measured as a whole stage
T_cpu    = h + 25 * (D + d + f * W)_GB + 0.90 * S_MB      C = P - (d + KV + compute) - D, D assumed hot
T        = T_G14 + sum(T_metal) + sum(T_cpu)             tok/s = 1000 / T
```

d is the stage's tensors of 10 MiB or less (anonymous memory), D its larger non-expert tensors (mapped by hcpeer).
T_cpu counts the Metal compute inside h once more, about 8 ms (conservative). The G14 keeps the planner's rule
(`moe_tier.cpu_expert_layers`): every host layer's dense part, KV and the head on CUDA0, then the experts of as many
small expert layers as the remaining VRAM holds; the other host layers' experts stay on the CPU, file-backed (-ot).

**The smaller f does not shrink the reads.** A token routes f * W = 5.01 GB of the G14's experts in A, the same as
Qwen alone (4.91 GB), because W doubled (160.4 vs 78.6 GB). What changes is C/W: 0.12 here against 0.25 for Qwen
alone, so the miss rate goes from 0.14 [M] to 0.42 [E] and SSD per token roughly triples. kappa was fitted at C/W
0.25-0.34; 0.12 is an extrapolation, which run A tests.

## 2. Candidates

### 2.1 Summary [E]

| | A: G14 alone (moe_offload) | B: + 3 Macs, Metal only | C: B + hcpeer CPU spans |
|---|---|---|---|
| stages in device order, layers | CUDA0 0-60 | azim MTL0 0-5, a MTL0 6-8, b MTL0 9-12, CUDA0 13-60 | B's three, a CPU 13-16, b CPU 17-20, CUDA0 21-60 |
| layers per stage | 61 | 6, 3, 4, 48 | 6, 3, 4, 4, 4, 40 |
| G14 VRAM of 11.1 GiB | 10.10 GiB, no expert layer | 9.92 GiB, experts of 58 | 10.86 GiB, experts of 55, 58 |
| G14 file-backed experts W vs 31 GiB RAM | 160.37 GB (149.35 GiB), 4.8x | 130.92 GB (121.93 GiB), 3.9x | 104.61 GB (97.43 GiB), 3.1x |
| C_G14 / W, miss | 0.122, 0.419 | 0.149, 0.344 | 0.187, 0.263 |
| G14 SSD per token | 2099 MB (2002 MiB) | 1409 MB (1344 MiB) | 861 MB (821 MiB) |
| G14 ms: GPU + CPU + faults + R | 25.4 + 62.6 + 1889.3 + 87.7 = 2065 | 19.5 + 51.1 + 1268.2 + 87.7 = 1427 | 16.8 + 40.9 + 775.1 + 87.7 = 920 |
| Mac SSD per token | none | none (all wired) | a 51.8 MB, b 21.2 MB |
| Mac ms beyond h | none | none | a 25.6 + 46.6, b 21.2 + 19.1 = 112.4 |
| added hops | 0 | 3 | 5 |
| T at h = 30 / 97 ms | 2065 | 1517 / 1718 | 1183 / 1518 |
| **tok/s at h = 30 / 97 ms** | **0.48** | **0.66 / 0.58** | **0.85 / 0.66** |
| break-even h | | 213 ms per stage against A | 197 ms per added stage against B |

Break-even h is the per-stage cost at which a candidate only ties the one before it; both are about twice tonight's
97 ms, so the ordering C > B > A holds unless a stage costs over 200 ms.

### 2.2 Bytes per stage

| stage | layers | weights [V] | of which experts [V] | d (sent whole) [V] | total vs cap [E] |
|---|---|---|---|---|---|
| mac-azim MTL0 | 0-5 | 8.805 GB (8.20 GiB) wired | 7.487 GB | 256 MB | 8.77 of 11.84 GiB (layer 6, 3.70 GB, does not fit) |
| mac-a MTL0 | 6-8 | 10.133 GB (9.44 GiB) wired | 9.689 GB | 171 MB | 9.98 of 10.67 GiB |
| mac-b MTL0 | 9-12 | 10.586 GB (9.86 GiB) wired | 10.071 GB | 217 MB | 10.41 of 10.67 GiB |
| mac-a CPU (C) | 13-16 | 13.904 GB (12.95 GiB) mapped + d | 13.535 GB | 230 MB | node 9.98 + 0.92 = 10.90 GiB |
| mac-b CPU (C) | 17-20 | 10.856 GB (10.11 GiB) mapped + d | 10.570 GB | 229 MB | node 10.41 + 0.84 = 11.25 GiB |

Every order of the three Macs ends their Metal stages at layer 12 (greedy fill is optimal for a fixed order and all
six orders give 13 layers); mac-azim first puts the cheap dense layers 0-2 on the largest cap. A CPU stage adds to its
node D + d + KV + compute + 128 MiB overhead (`services/layout.py` node check).

- A, G14: dense 8.848 + head 0.760 GB + KV 518.5 MiB + compute 208 MiB + overhead 450 MiB = 10.10 GiB, leaving
  1.00 GiB, under the smallest expert layer (2.202 GB, 2.05 GiB). So all 58 MoE layers' experts are file-backed:
  miss = exp(-19.535 / (0.14 x 160.365)) = exp(-0.870) = 0.419; S = 5.011 GB x 0.419 = 2099 MB; faults
  0.90 x 2099 = 1889 ms; CPU 12.5 x 5.011 = 62.6 ms; GPU 2.64 x 9.608 = 25.4 ms; T = 2065 ms.
- B, G14 13-60: fixed part 7.87 GiB leaves 3.23 GiB, one 2.202 GB expert layer (58) moves to VRAM; W = 160.365 -
  27.245 (Macs) - 2.202 = 130.92 GB.
- C, mac-a CPU 13-16: C = 4.5 GiB - (230 MB + 34 MiB KV + 208 MiB) - 369 MB D = 3979 MB; f W = 423 MB;
  miss = exp(-3979 / (0.14 x 13535)) = exp(-2.100) = 0.1225; S = 51.8 MB; compute 25 x (0.369 + 0.230 + 0.423) =
  25.6 ms; faults 46.6 ms. mac-b 17-20: C = 4062 MB, miss 0.064, S = 21.2 MB, 21.2 + 19.1 ms.
- C, G14 21-60: room 4.34 GiB takes expert layers 55 and 58; W = 104.61 GB, miss 0.263, S = 861 MB.

### 2.3 First load: bytes to each Mac

| Mac | B stage | C adds | unseeded, host link at 10 / 30 MB/s | seeded (d only) at 10 / 30 MB/s | Hugging Face seed at 13 / 45 MB/s |
|---|---|---|---|---|---|
| mac-azim | 8.805 GB | | 14.7 / 4.9 min | 256 MB: 26 / 9 s | 11.3 / 3.3 min |
| mac-a | 10.133 GB | 14.134 GB (13-16) | 16.9 / 5.6 min, C +23.6 / 7.9 | 171 MB, C +230 | 13.0 / 3.8 min, C +18.1 / 5.2 |
| mac-b | 10.586 GB | 11.086 GB (17-20) | 17.6 / 5.9 min, C +18.5 / 6.2 | 217 MB, C +229 | 13.6 / 3.9 min, C +14.2 / 4.1 |
| all | 29.52 GB | 54.74 GB in C | B 49 / 16 min, C 91 / 30 min | B 644 MB, C 1.10 GB | in parallel: B 14 / 4 min, C 31 / 9 min (mac-a) |

[V] bytes, [E] times. b11770 loads one RPC device after another, so unseeded times add up. Over a phone hotspot a
Hugging Face seed is cellular data (29.5 GB for B, 54.7 GB for C): seed on Wi-Fi before moving to the hotspot. Even
seeded, the G14 hashes every peer tensor over 10 MiB at load: tonight 19.3 GiB took 80-130 s [M], so B (27.5 GiB)
about 2-3 min and C (51.0 GiB) about 3.5-6 min [E]. Disk: mac-azim needs 8.2 GiB under a 30 GiB cache cap that holds
about 18 GB of older caches [M]; mac-a 22.6 and mac-b 20.2 GiB of 161 and 156 GiB free [E].

## 3. Sensitivity [E]

| case | A tok/s (G14 SSD MiB) | B at 30 / 97 | C at 30 / 97 (Mac SSD MiB a, b) | C vs B break-even |
|---|---|---|---|---|
| central: kappa 0.14, P 4.5 GiB, C_G14 19.5 GB | 0.48 (2002) | 0.66 / 0.58 | 0.85 / 0.66 (49, 20) | 197 ms |
| kappa 0.176 everywhere (gpt-oss fit, f = 1/32) | 0.41 (2392) | 0.55 / 0.49 | 0.68 / 0.56 (76, 35) | 210 ms |
| kappa_G14 0.10 | 0.66 (1414) | 0.93 / 0.78 | 1.16 / 0.84 | 137 ms |
| kappa_G14 0.20 | 0.38 (2599) | 0.50 / 0.46 | 0.64 / 0.53 | 245 ms |
| C_G14 15 GB | 0.40 (2450) | 0.53 / 0.48 | 0.68 / 0.56 | 234 ms |
| P 3 GiB / 5 GiB | as central | as central | 0.78 / 0.62 (116, 60); 0.86 / 0.67 (37, 14) | 147 / 206 ms |
| kappa_mac 0.35 (design's pessimistic) | as central | as central | 0.72 / 0.58 (174, 105) | 98 ms |
| kappa 0.20 everywhere, P 3 GiB, C_G14 15 GB | 0.33 (2994) | 0.43 / 0.40 | 0.49 / 0.42 (168, 99) | 177 ms |

C is at or above B in every row (a tie at 97 ms with kappa_mac 0.35), and B above A. A sweep of 1-7 layers per CPU
span peaks at 5 + 5 for P = 4.5 GiB (0.86 / 0.67) and at 3 + 4 for P = 3 GiB (0.78 / 0.62); 4 + 4 stays within 4% of
the best at every P from 3 to 5 GiB while keeping mac-a under 120 MiB of SSD per token at P = 3.

Prefill is not modelled. A 64-token prompt routes to 1 - (248/256)^64 = 86.9% of each layer's experts, so the G14
faults in about 0.869 W - C_G14: A 120 GB (108 s at 0.90 ms/MB), B 94 GB (85 s), C 71 GB (64 s) plus 7.8 and 5.1 GB
on mac-a and mac-b. Expect time to first token of 1-2 minutes; record it.

## 4. Check before loading

- **VRAM slack** is 1.00 GiB in A, 1.18 in B and 0.24 GiB in C on the 208 MiB compute estimate. Without flash
  attention the MLA score buffer at -ub 512 and ctx 4096 is about 128 heads x 512 x 4096 x 4 B = 1 GiB [E], which C
  cannot absorb. Read `CUDA0 compute buffer size` on the first load (A); if it is over about 0.4 GiB, plan C with a
  smaller GPU budget so expert layer 55 stays on the CPU (C then about 0.82 / 0.64 tok/s [E], one 2.2 GB layer more
  on the G14).
- **Mac contributions.** Tonight's 7.5 GiB caps are too small: rejoin with at least 8.8 (azim), 10.0 (a) and 10.4 GiB
  (b) for B, 10.9 (a) and 11.3 GiB (b) for C, and check the gateway's free-memory reading allows it. mac-b's Metal
  stage is 0.26 GiB under its working-set cap; close other apps on it. If mac-b cannot take 4 Metal layers, rerun
  `calc.py` with its real cap rather than guessing a smaller span.
- **hcpeer** is installed on mac-a and mac-b, S1 gate passed on mac-a (`PAPER-RUNS-2026-10-22.md` R3); a valid C run
  shows `[hcpeer] mapped` about 12.95 GiB on mac-a and 10.11 GiB on mac-b, only small tensors copied, swap unchanged.

## 5. Run first: A

1. It is the denominator of every speedup and needs no Mac and no bytes over the hotspot, so it can start as soon as
   the download's sha256 matches.
2. It measures the number all three predictions rest on: G14 SSD per decoded token at C/W = 0.12, predicted 2002 MiB
   (kappa 0.14) or 2392 MiB (0.176). Fit kappa = C / (W x -ln(S / (f W))) from it and rerun `calc.py` before sending
   30-55 GB to the Macs. The kappa 0.10 and 0.20 rows of section 3 bracket 1414 to 2599 MiB.
3. It shows the real CUDA compute buffer (section 4), which decides whether C's two GPU expert layers fit.
4. While it runs, seed the Macs from Hugging Face on Wi-Fi: azim 0-5, a 6-8 and 13-16, b 9-12 and 17-20.

Then run C alternating with B (B is C without the CPU spans and uses the same seeds), same protocol as D4-D6: 64 prompt
and 64 output tokens, temperature 0, seed 7, 1 warmup and 3 measured reps. At about 2 s per token each A rep takes
about 4 min [E]. If only one Mac run fits tonight, run C: it is predicted 28% faster than B at 30 ms and 13% at 97 ms.
Judge C by: G14 SSD per token near 821 MiB (B 1344 MiB); Mac pageins near 49 and 20 MiB per token (`vm_stat` delta
times 16 KiB); over 120 MiB on mac-a the span is too long for its page cache, so shrink to 3 + 3. If C's decode is at
or below B's with those checks in range, an added stage costs more than the 197 ms break-even and the limit is the
per-stage cost (design S0), not the disk.

Not done: nothing ran on this model; C_G14, kappa at C/W 0.12, P, the Mac fault rate and the per-stage cost of a CPU
stage are unmeasured for it, and the predictions are only as good as those five numbers.
