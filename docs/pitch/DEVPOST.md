# Hypercompute

## Elevator pitch

Hypercompute is multiplayer computing. Friends paste a one-use invite, and their laptops join one encrypted pool that runs open models too big for any single member's memory. A gaming laptop plus three MacBook Airs gets a 120B model to its first token in 13.44 s instead of 36.17 s (G14 host, 128-token prompt, 9 vs 9 reps), and the same four laptops helping a 121.6 GiB GX10 (NVIDIA GB10) box get DeepSeek-V3.1 671B there in 10.36 s instead of 24.67 s (64-token prompt, 4 vs 4 reps).

## Inspiration

Open models have outgrown the machines people own. Quantized, DeepSeek-V3.1 is 170.5 GB (158.8 GiB) and Kimi-K2.6 is 193.16 GiB. Our biggest box has 121.6 GiB, a MacBook Air 16 GiB. A machine that runs out of memory pages weights from its SSD, and you feel that as waiting: on the GX10 alone, the 671B's first token takes 24.67 s on a 64-token prompt. Every hackathon table has a few laptops. We wanted them to act as one.

## What it does

- A host starts a session. Friends run `hypercompute join` with a one-use invite, over an authenticated, encrypted Iroh QUIC tunnel (direct through NAT when possible, relay otherwise).
- A peer counts only after a real llama.cpp RPC handshake.
- A planner puts whole layers on peers. For the layers it keeps, the host GPU holds the dense weights and the KV cache, and their routed MoE experts stay in host RAM or memory-mapped from SSD.
- Managed peers download their own slice from Hugging Face. Windows CUDA, Linux GB10 and macOS Metal run in one pool.

**Measured speed** (means over measured reps; 95% bootstrap intervals on the ratio columns only):

| model, alone vs pooled | tokens in / out | first token | prompt reading | full answer | decode | SSD MiB per token |
|---|---|---|---|---|---|---|
| gpt-oss-120b: G14 laptop vs G14 + 3 Airs (9 vs 9 reps) | 128 / 64 | 36.17 vs 13.44 s | 2.99x [2.64, 3.35] | 54.44 vs 27.93 s | 1.27x [1.10, 1.46] | 275 vs 41 |
| DeepSeek-V3.1 671B: GX10 vs GX10 + G14 + 3 Airs (4 vs 4) | 64 / 32 | 24.67 vs 10.36 s | 2.42x [2.19, 2.72] | 41.84 vs 31.20 s | 0.83x [0.77, 0.90] | 684 vs 37 |
| Kimi-K2.6 1T: same pool (4 vs 4) | 64 / 32 | 28.25 vs 19.08 s | 1.51x [1.41, 1.59] | 45.48 vs 39.23 s | 0.86x [0.82, 0.89] | 671 vs 254 |

Ratio columns are pooled over alone throughput, so above 1x means the pool is faster. Prompt lengths are targets; tokenized prompts were 125 to 133 (120B) and 58 to 68 (671B, 1T). The 120B ran on building Wi-Fi, the GX10 rows on home Wi-Fi, and in the 671B pooled runs one Mac stayed on the relay (47 to 84 ms RTT). The two pooled 671B runs used two peer layouts (R6). A code workflow shared the G14 during most 120B runs (R2 confound). Pooling sped up prompt reading and first token in every row. Decode got faster only in the laptops-only pool.

**Measured tonight, laptops only** (G14 alone vs G14 + 2 MacBook Airs on a home network, one pooled run per model; decode pooled over alone): Llama 3.3 70B 0.165 to 0.898 tok/s, 5.44x [5.42, 5.47] (64 / 32, 2 vs 2 reps); Qwen2.5-VL 72B text 2.59x [2.57, 2.62] (64 / 32, 2 vs 2); gpt-oss-120b 2.28x [2.00, 2.54], first token 34.22 to 4.83 s (128 / 64, 3 vs 3); Qwen3-Next 80B 0.46x [0.36, 0.62], slower pooled (128 / 64, 6 vs 3). The laptops-only pool was not run on 235B, 671B or 1T. Still being measured: [SWEEP: 24B laptops-pool first token]. Concurrency: on the GX10 alone (64 / 32, one run per N, 2 batches each) the 671B served 1.80, 3.44 and 5.54 tok/s in total at N = 1, 4 and 8, and the 1T 1.78, 3.25 and 4.64. Pooled concurrency was measured only on the laptops, gpt-oss-120b, 128 / 64, one run per side, 3 batches: total decode 1.34x [1.24, 1.48] the laptop alone at N = 8.

## How we built it

- **Inference.** Stock llama.cpp b11770 on every machine (`llama-server` on the host, `ggml-rpc-server` on peers), sha256-pinned.
- **Gateway.** Python 3.12, FastAPI, SQLite and JSONL run records. Dashboard in React and TypeScript.
- **Planner.** An exact dynamic-programming placement optimizer over layers, experts, peers and SSD tiers. It agrees with a brute-force oracle on 3,000 random cases.
- **hcpeer.** An optional peer patch that serves weights from memory-mapped SSD files. On mac-a (Qwen2.5 7B Q4_K_M, every layer on an rpc-server CPU device, 64-token greedy run), it gave identical text at 17.96 tok/s mapped vs 17.89 stock, with the mapped set in page cache.

## Challenges we ran into

- **Decode.** In llama.cpp b11770 each peer adds one blocking round trip per token. On the GX10 those stages cost more than the SSD reads they save: 0.83x on the 671B, 0.86x on the 1T.
- **Networks.** Apartment Wi-Fi put every client in its own subnet, so laptops could only reach each other through the Iroh relay. With the Macs on a phone hotspot, host-to-Mac uploads fell to about 1 MiB/s, RTT rose to 60 to 580 ms, and a peer dropped mid-load.
- **Moving weights.** On building Wi-Fi, a cold push of the 235B's 19.3 GiB to three Macs took 2289 s (8.6 MiB/s). Peers now pull their own layers from Hugging Face: 13 to 24 MB/s per Mac while seeding the 120B, 45 MB/s on a single range test.
- **The planner on a new host.** On the 671B it predicted the pool 1.77x ahead of the GX10 alone on decode. We measured 0.83x, because its speed constants came from the G14.

## Accomplishments that we're proud of

- 671B and 1T models in one pool across Windows, Linux and macOS on stock llama.cpp.
- The 671B on the G14 alone at 0.48 tok/s (64 / 32 tokens, 2 reps), the speed our layout analysis predicted before the run.
- Every benchmark number traces to a run.json (temperature 0, seed 7, warmups excluded), and we show the losses next to the wins.

## What we learned

- Pooling buys memory and batched prompt math, so prompt reading improved in every pool in the table.
- Decode improved on the G14 host, where 3 Macs cut SSD reads from 275 to 41 MiB/token (120B, 1.27x [1.10, 1.46]). On the GX10 host it fell (0.83x on the 671B, 0.86x on the 1T): there, four Wi-Fi stages cost more than the SSD reads they saved. On the 120B, 3 Macs decoded no faster than 1 (1.02x [0.86, 1.20]) while prompt reading doubled (2.02x).
- Fewer, larger, direct stages are our hypothesis for decode, and fewer-hop pools are being measured. A cost model must be measured on each host: the planner predicted 1.77x on the 671B and we measured 0.83x.

## What's next for Hypercompute

- Concurrency with N parallel slots (`-np N -kvu`) and the laptops-only sweep, both being measured.
- Fewer-hop pools (a Mac mini plus the G14, then two Macs alone) and per-host planner calibration.
- Long prompts (so far only 58 to 133 tokens).

## Built with

llama.cpp, ggml-rpc, iroh, quic, python, fastapi, pydantic, sqlite, react, typescript, vite, cuda, metal, hugging-face, uv, pytest, vitest, windows, linux, macos, nvidia-gb10, apple-silicon
