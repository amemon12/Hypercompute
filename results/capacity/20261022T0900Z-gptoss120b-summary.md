# gpt-oss-120b through the gateway: laptop alone vs laptop + M4 Air

Model `gpt-oss-120b-mxfp4` (ggml-org/gpt-oss-120b-GGUF @ 7200b74b, 59.03 GiB, sha256 verified by the gateway), 116.8B
parameters, 128 experts with 4 active per token. Gateway built from this branch (moe_offload tier), llama.cpp b11770.
Host: RTX 5070 Ti Laptop 12 GiB, 31 GiB RAM with about 10 GiB free, AC power. Peer: friend's M4 Air 16 GiB
(contribution 10 GiB, reserve 4 GiB) over a direct Iroh path on a phone hotspot (7-11 ms RTT, uploads about
5-12 MB/s observed). Workload: ctx 4096, about 126-token prompts, 64 output tokens, 1 warmup plus 2 measured reps,
decode from engine timings. Both runs were driven by `scripts/capacity/run_big.py`.

| | Laptop alone | Laptop + M4 Air |
| --- | --- | --- |
| Evidence | `20261022T085541Z-big-gpt-oss-120b-mxfp4/run.json` | `20261022T085102Z-big-gpt-oss-120b-mxfp4/run.json` |
| Placement (engine load log) | CUDA0 9,697 MiB; CPU experts 50,741 MiB file-backed | RPC0 (Mac) 4,939 MiB (layers 0-2); CUDA0 9,612 MiB; CPU experts 45,888 MiB file-backed |
| Private host weight buffers | 0 MiB | 0 MiB |
| Load to ready | 41 s | 62 s (Mac layers from its RPC cache; the first upload took 521 s) |
| Decode, median (min-max) | 4.05 tok/s (3.77-4.33), 248 ms/token | 4.52 tok/s (4.48-4.56), 221 ms/token |
| First token, 125-126 token prompt | 30.8 s | 21.2 s |
| Prefill | 4.17 tok/s | 6.28 tok/s |
| SSD reads per decoded token (system wide) | 232 MB | 121 MB |
| Commit peak / limit | 42.1 / 55.0 GiB (limit unchanged) | 46.0 / 55.0 GiB (limit unchanged) |

What this shows, measured:
- A 117B model runs on this laptop through the Hypercompute gateway, with the planner's MoE placement verified from
  the engine's own buffers, and without growing the pagefile.
- With the Mac holding three whole layers, decode was 12% faster, the first token 31% faster and SSD reads per token
  48% lower than the same model on the laptop alone. The Mac's participation is in the load log (4,939 MiB on RPC0) and
  in its rpc-server CPU time during the reps.

Limits of the evidence:
- Two measured reps per configuration; the spread overlaps little but the sample is small.
- The runs were back to back with the laptop-alone run second, so its file cache was already warm from the pair run.
- The planner's estimate was 1,081 ms/token and 1.13 GB of SSD reads per token for the pair plan; measured 221 ms and
  121 MB. Its miss-rate model assumes no reuse of expert pages across tokens and is far too pessimistic.
- The runner's available-RAM guard had to be turned off (`--min-available-gib 0`): mapped expert pages take
  available RAM to zero while commit stays far below its limit. The commit-growth and drive-space guards stayed on.
- Two earlier attempts of the pair run were stopped by that guard and are kept as evidence
  (`20261022T083945Z-*`, `20261022T084916Z-*`).
