# Hypercompute deck outline (3-minute table pitch, 6 slides)

Source keys. **PR** = `docs/research/PAPER-RUNS-2026-10-22.md` (sections R1 to R6). **TT** = first-token and
full-answer bootstrap over the `run.json` reps in `results/capacity/paper-2026-10-2*` (ratio of means, 95% percentile
bootstrap, 10,000 resamples, seed 7; same method as PR). **CK** = `docs/research/CHECKPOINT-2026-10-24.md`.

Fixed conditions, so every number below can carry them:

- **GX10 rows**: host ASUS Ascent GX10 (GB10, 121.6 GiB unified). "Pooled" = GX10 + G14 (CUDA peer) + 3 M4 MacBook
  Airs, home Wi-Fi. 64-token prompt requested (measured prompt_n 58 to 66), 32-token answer, temperature 0, seed 7,
  4 measured reps per side. On the 671B one peer (mac-azim) was on the Iroh relay, and the pooled reps mix two peer
  layouts (gx4-1, gx4-2). On the 1T all four peers were direct. (PR R6)
- **G14 rows**: host ROG Zephyrus G14 (RTX 5070 Ti Laptop 12 GiB, 31.1 GiB RAM). "Pooled" = G14 + M4 MacBook Airs
  (16 GiB each), building Wi-Fi. 120B: 128-token prompt requested (measured prompt_n 128 to 132), 64-token answer,
  9 reps alone, 9 with 3 Airs, 6 with 1 Air. 235B: 64 / 64, steady state runs 3 to 6, 9 vs 9 reps. (PR R1, R2)
- **671B on the G14 alone** (Slide 2 only): no peers, tier moe_offload, 64-token prompt, 32-token answer, 1 run,
  1 warmup, 2 measured reps. (PR R4)
- Placeholders in `[SWEEP: ...]` and `[CONCURRENCY: ...]` are being measured tonight. Fill them only from the new
  `run.json` files, or cut them.

Timing: S1 0:00 to 0:20, S2 0:20 to 0:50, S3 0:50 to 1:30, S4 1:30 to 2:10, S5 2:10 to 2:40, S6 2:40 to 3:00.

---

## Slide 1. Multiplayer computing

- Kimi-K2.6 has 1026B parameters and is 193.16 GiB. The largest machine in our pool has 121.6 GiB. (PR R6)
- Friends join with a one-use code. Their laptops take whole layers over an encrypted Iroh QUIC tunnel.
- DeepSeek-V3.1 671B, 64-token prompt, 32-token answer: first token 10.36 s on GX10 + G14 + 3 Airs vs 24.67 s on the
  GX10 alone (mean of 4 vs 4 reps, home Wi-Fi, one peer on relay). (TT)

**Speaker notes.** This is Hypercompute. Friends' machines join one pool and run a model that none of them can hold
in memory. On a 671B model and a 64-token prompt, the pool starts answering in 10.36 seconds. The biggest box alone
takes 24.67.

---

## Slide 2. Big open models outgrow one machine's memory

- DeepSeek-V3.1 671B is a 170.5 GB (158.8 GiB) file and Kimi-K2.6 1T is 193.16 GiB. The GX10 has 121.6 GiB, the
  G14 31.1 GiB of RAM, each MacBook Air 16 GiB. (PR R4, R6)
- So the host pages experts off its SSD on every token: 684 MiB per token on the 671B, 671 on the 1T (GX10 alone;
  PR R6). The G14 alone reads 2485 MiB per token and decodes the 671B at 0.481 tok/s (G14 alone, 64 / 32 tokens,
  1 run, 2 reps; PR R4).
- The GX10 alone does run both, at about 1.8 tok/s (64-token prompt, 32-token answer, 4 reps). The cost is the wait:
  24.67 s to the first token on the 671B (mean of 4 reps). (PR R6, TT)

**Speaker notes.** The usual fix is a bigger machine. We had one, and it still runs these models by reading the SSD
for every token. In the GX10 runs we capped each laptop at 7.5 to 11.5 GiB (G14 9 GiB on its GPU, Macs 7.5 to
11.5 GiB unified). (PR R6 caps)

---

## Slide 3. How it works

- **Join.** The host makes a one-use invite and a friend runs `hypercompute join`. Traffic rides an encrypted,
  authenticated Iroh QUIC tunnel, and a peer counts only after a real llama.cpp RPC handshake.
- **Place.** The planner puts whole layers on peers and splits MoE experts across the host's GPU, RAM and SSD.
  1T example: four laptops hold layers 0 to 10 (6.6 to 9.9 GiB each), the GX10 GPU holds 11 to 60 (76.0 GiB). (PR R6)
- **Run.** Stock llama.cpp b11770 RPC on Linux GB10, Windows CUDA and macOS Metal in one pool. Each managed peer
  downloads its own slice from Hugging Face. In the G14-host runs on building Wi-Fi, each Mac seeded from Hugging Face
  at 13 to 24 MB/s (45 MB/s in a single range test), against 8.6 MiB/s (9.0 MB/s) pushed from the host. (PR R1.3, R2)

**Speaker notes.** The host runs the only llama-server. Every friend runs a small agent that serves layers over RPC.
The planner decides which machine holds which layers and which experts stay on the host's SSD. Peers fetch their
own layers from Hugging Face. In the G14-host runs, one range test from Hugging Face was about 5x faster than the
host pushing layers over Wi-Fi.

---

## Slide 4. Time to first token: pooled vs host alone

**Chart spec** (one chart).

- Type: grouped vertical bars, two bars per group.
- X axis label: `Model and pool (prompt / answer tokens)`.
- Categories:
  1. `DeepSeek-V3.1 671B, GX10 + G14 + 3 Airs (64 / 32)`
  2. `Kimi-K2.6 1T, GX10 + G14 + 3 Airs (64 / 32)`
  3. `gpt-oss-120b, G14 + 3 Airs, laptops only (128 / 64)`
- Y axis label: `Time to first token, s (mean of measured reps; lower is better)`.
- Series `Host alone`: 24.67, 28.25, 36.17.
- Series `Pooled`: 10.36, 19.08, 13.44.
- Whiskers, min to max rep (s): alone 24.2 to 25.7, 26.5 to 30.4, 33.8 to 39.5; pooled 8.6 to 11.7, 18.0 to 20.1,
  10.3 to 18.2.
- Label above each group, line 1, `pooled / alone [95% CI]`: 0.420 [0.373, 0.462]; 0.675 [0.629, 0.728];
  0.371 [0.325, 0.424].
- Label above each group, line 2, `prefill tok/s, pooled / alone [95% CI]`: 2.42x [2.19, 2.72]; 1.51x [1.41, 1.59];
  2.99x [2.64, 3.35].
- Footnote: `4 vs 4 reps (GX10 rows), 9 vs 9 reps (120B). 95% percentile bootstrap, 10,000 resamples. Home Wi-Fi
  (GX10 rows), building Wi-Fi (120B). 671B pooled: one peer on relay, reps mix two peer layouts. 120B: another job
  shared the G14 in 4 of its 6 runs.`
- Sources: TT for first token and whiskers; PR R6 (671B, 1T) and PR R2 (120B) for prefill.

**Speaker notes.** Prefill is batched, so peers take expert math off the host. On the 671B that read the prompt 2.42x
faster, and the first token came in 10.36 s instead of 24.67 s. The 1T gains less, 19.08 against 28.25 s. The
laptops-only 120B pool gains most, 13.44 against 36.17 s. These are short prompts, 64 and 128 tokens, with 4 and 9
reps per side. Every ratio carries its interval.

---

## Slide 5. The cost: decode, and when pooling wins

**Chart spec** (one chart).

- Type: horizontal forest plot, one point with a 95% CI bar per row, vertical reference line at 1.0 labelled
  `host alone`.
- X axis label: `Decode tok/s, pooled / host alone (ratio of means, 95% bootstrap CI)`.
- Y axis label: `Model, pool`.
- Rows (point [low, high]), measured:
  1. `DeepSeek-V3.1 671B, GX10 + G14 + 3 Airs`: 0.83 [0.77, 0.90] (PR R6)
  2. `Kimi-K2.6 1T, GX10 + G14 + 3 Airs`: 0.86 [0.82, 0.89] (PR R6)
  3. `Qwen3-235B, G14 + 3 Airs, steady state`: 1.08 [0.98, 1.18] (PR R1)
  4. `gpt-oss-120b, G14 + 1 Air`: 1.25 [1.11, 1.41] (PR R2)
  5. `gpt-oss-120b, G14 + 3 Airs`: 1.27 [1.10, 1.46] (PR R2)
- Rows, laptops-only pool (G14 + mac-a + mac-b, home network, PR R7); row 6 still being measured:
  6. `[SWEEP: 24B laptops-pool decode]`
  7. `Llama 3.3 70B dense, G14 + 2 Airs, home network`: 5.44 [5.42, 5.47] (PR R7.2)
  8. `Qwen2.5-VL 72B text, G14 + 2 Airs, home network`: 2.59 [2.57, 2.62] (PR R7.4)
  9. `gpt-oss-120b, G14 + 2 Airs, home network`: 2.28 [2.00, 2.54] (PR R7.3)
  10. `Qwen3-Next 80B MoE, G14 + 2 Airs, home network`: 0.46 [0.36, 0.62] (PR R7.1)
- Color: GX10-host rows in one color, laptops-only rows in a second, placeholder rows hollow grey.
- Footnote: `Batch size 1. Each peer is one blocking network round trip per token. 671B: 64 / 32, 4 vs 4, home
  Wi-Fi, 1 peer on relay. 1T: 64 / 32, 4 vs 4, home Wi-Fi, all peers direct. 235B: 64 / 64 steady state, 9 vs 9,
  building Wi-Fi. 120B: 128 / 64, 9 vs 9 (3 Airs), 6 vs 9 (1 Air), building Wi-Fi, G14 shared with another job in 4
  of 6 runs. Rows 7 to 10: G14 + mac-a + mac-b (10.5 GiB caps), home network, both Macs direct, RTT 5 to 8 ms
  before each pooled row, one pooled row per model; 70B and 72B 64 / 32, 2 vs 2 reps; 120B 128 / 64, 3 vs 3; 80B
  128 / 64, 6 alone (2 rows) vs 3 pooled. Decode tok/s alone vs pooled: 70B 0.165 vs 0.898 (a second alone row ran
  about 9% faster in its warmups), 72B 0.141 vs 0.365, 120B 3.829 vs 8.740, 80B 21.373 vs 10.005. No laptops-only
  pool of the 671B was run; the G14 alone runs it at 0.481 tok/s (R4). Concurrency with pooling was measured on the
  laptops only: gpt-oss-120b, G14 + 2 Airs, home network, 128 / 64, one row per side, 3 reps at N = 1 and 3 batches
  at N = 4 and 8; aggregate decode tok/s pooled vs alone 2.28x [2.00, 2.54] at N = 1, 1.46x [1.29, 1.68] at N = 4,
  1.34x [1.24, 1.48] at N = 8 (PR R7.5).`

**Speaker notes.** On a big host, pooling costs decode: 0.83x on the 671B and 0.86x on the 1T, because each peer
adds one blocking round trip per token. Decode wins only when the host is the weak link. On the 120B (128-token
prompt, 64-token answer, 9 vs 9 reps), with 3 Airs the G14 reads 41 instead of 275 MiB per token from its SSD and
decodes 1.27x faster. So we sell the wait for the first answer, not tokens per second on a big box.

---

## Slide 6. What is next, and the ask

- **Being measured tonight.** Laptops-only pool: 70B to 120B measured (slide 5, rows 7 to 10); the other rows
  (24B to 109B) are still queued, from `[SWEEP: 24B laptops-pool decode]` up. 235B, 671B and 1T were not in
  tonight's laptop sweep. Parallel requests at N = 1, 4, 8: GX10 alone, 64 / 32, one row per N, 2 batches each:
  671B 1.80, 3.44, 5.54 tok/s aggregate, 1T 1.78, 3.25, 4.64 (PR R6.y). Pooled concurrency was measured only on the
  laptops (gpt-oss-120b, slide 5 footnote). (CK)
- **Next.** Fewer, larger, direct peers (G14 + Mac mini; two Airs), and prompts longer than the 58 to 133 tokens
  measured so far. (CK; PR R6; prompt_n in run.json)
- **Ask.** Watch a laptop join with a one-use code, then send the pool a prompt.

**Speaker notes.** Every measured number comes from a run.json in the repo, and every ratio carries its interval.
Tonight's sweep fills in the laptops-only pool from 24B up. The concurrency test tells us whether the pool helps many
requests at once. Until then we do not claim it. You can send the pool a prompt now. Our timings come from short
prompts, so the time for your prompt is not from our runs.

---

## Q&A lines (not a slide)

- **"Is generation faster?"** "On the GX10 host, no: 0.83x and 0.86x (671B and 1T, 64-token prompt, 32-token
  answer). Prompt reading and first token are faster. The clear decode win is the laptops-only 120B pool, G14 + 3
  Airs: 1.27x [1.10, 1.46], 128-token prompt, 64-token answer."
- **"Couldn't one machine do this?"** "The GX10 alone runs both models at about 1.8 tok/s (64-token prompt, 32-token
  answer) by paging experts from its SSD. Neither model fits in its memory."
- **"How many users can it serve?"** "Being measured tonight. We have not measured concurrency with pooling yet."
- **"Is it private?"** "The tunnel is encrypted and authenticated. The host sees prompts and peers see activations."
- **"Full answer time?"** 671B: 41.84 s alone vs 31.20 s pooled, 0.746 [0.694, 0.797]. 1T: 45.48 vs 39.23 s,
  0.863 [0.818, 0.912] (64-token prompt, 32-token answer, GX10 + G14 + 3 Airs, 4 vs 4 reps). 120B laptops: 54.44 vs
  27.93 s, 0.513 [0.453, 0.577] (120B: G14 + 3 Airs, 128 / 64, 9 vs 9). (TT)
- **"Load time?"** 671B pooled: 503 to 517 s load after 856 to 1416 s of peer seeding; GX10 alone 239 to 256 s. We
  preload before judging. (PR R6)

## Do not put on any slide

- "Pooling makes generation faster", "first ever", "impossible on one machine", "serves many users", "private
  inference", the 56 tok/s 120B number (that is the GX10 alone, a host change), or "optimal" for the planner.
- Any other project by name.
