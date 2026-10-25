# Layer split optimisation for a CUDA laptop + M4 MacBook Air over Wi-Fi (llama.cpp b11770)

Scope: single-user latency (TTFT and decode tokens/s) for llama.cpp `-sm layer` splits across the host
(ASUS ROG Zephyrus G14 GA403WR: RTX 5070 Ti Laptop 12 GiB, Ryzen AI 9 HX 370, 32 GB LPDDR5X) and one RPC worker
(M4 MacBook Air 16 GB, Metal) over Wi-Fi. vLLM and MLC engines are out of scope for this file.

Pinned sources, read with `gh api -H "Accept: application/vnd.github.raw" "repos/<o>/<r>/contents/<path>?ref=<sha>"`.
Nothing from these repos was built, installed or run.

| Repo | Ref | License |
|---|---|---|
| ggml-org/llama.cpp | `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb` (tag b11770) | MIT |
| exo-explore/exo | `bc412e2230f395e652c23426a5eb88dfa515769b` (main, 2026-10-13); legacy tag `v0.0.15-alpha` = `af734f1b` | Apache-2.0 |
| bigscience-workshop/petals | `22afba627a7eb4fcfe9418c49472c6a51334b8ac` (main) | MIT |
| b4rtaz/distributed-llama | `4074c0f3fc02800346ccd6df3293915d4276c3d2` (main) | MIT |

Read first and not repeated here: `llama_placement.md` (device order, `-ts` mapping, `--fit`) and `llama_rpc.md`
(protocol, failure modes). Line numbers below are for the refs above. Labels: **VERIFIED** = read in source or an
official page; **DERIVED** = arithmetic on verified numbers; **UNVERIFIED** = assumption or third-party claim that
still needs a measurement.

---

## 0. TL;DR

1. **Each RPC device used costs exactly one blocking network round trip per decoded token.** Graph compute and
   `SET_TENSOR` are one-way messages. The only blocking call is the `GET_TENSOR` that reads a remote device's output
   back to the client, and the server synchronizes its backends before answering it. So that RTT also contains the
   remote compute time (§2, VERIFIED).
2. **Device order decides how many bytes cross the link, and Hypercompute's planner currently uses the expensive
   order.** `planner.py:198-200` puts local CUDA first and peers after it. `_layout` gives the output head to the last
   device (`planner.py:124-129`), so the head ends up on a peer. Evidence: in `results/20261021T032156Z-...-3machines.json`
   the order is `['CUDA0','RPC1','RPC0']` and `RPC0` has `holds_output: true`. Every token then pulls
   `n_vocab*4` bytes of logits over the network (608 KB for Qwen2.5) instead of `n_embd*4` (14 KB), and the largest
   tensor (about 0.45 GB of LM head) runs on the slower device. llama.cpp's own default puts RPC devices first and the
   local GPU last, "to minimize network transfers" (`src/llama.cpp` L275-285, llama_placement.md §1.3). Flip the order.
   Estimated gain: about 20 ms/token at 300 Mbit/s and 3 ms RTT (§3.4, DERIVED with an UNVERIFIED network).
3. **For one request stream, a layer split is a sequential chain, so the latency-optimal split puts as few bytes as
   possible on the slow device.** Per byte, the M4 is about 4x slower than the 5070 Ti for decode (92 vs 379 GB/s
   effective) and about 17x slower per layer for prefill (§1, §3). Use the Mac only for layers that do not fit in VRAM,
   and compare it with the host CPU tier for that overflow. The memory-proportional split (llama.cpp default without
   `-ts`, EXO's placement) and the stage-equalizing split (best for pipelined throughput) both put too much on the Mac
   for single-user latency.
4. **llama-bench can do the whole calibration and search at b11770.** It has `-rpc`, `-dev a/b,b/a`, `-ts x/y,...`,
   `-ngl`, `-d` (depth), `-pg`, `-r` and `-o jsonl`. `--fit` is off by default, and one invocation sweeps several
   splits (§5a).
5. Recommended optimiser: a GGUF-derived per-layer byte model, measured device and link coefficients, a closed-form
   cost `J = T_prefill(N_p) + N_g * T_decode`, exhaustive search over contiguous splits (at most a few thousand
   evaluations), a short llama-bench check of the top 3 candidates, and profiles kept in SQLite next to `benchmarks`
   and `plans` (§5).

---

## 1. Hardware facts

| Item | Value | Status / source |
|---|---|---|
| RTX 5070 Ti Laptop GPU memory | 12 GB GDDR7, **192-bit**, 5888 CUDA cores, 60-115 W | VERIFIED, NVIDIA laptop compare page https://www.nvidia.com/en-us/geforce/laptops/compare/ ("12 GB GDDR7", "192-bit") |
| RTX 5070 Ti Laptop peak bandwidth | **672 GB/s** nominal (192 bit / 8 x 28 Gbps) | UNVERIFIED: NVIDIA does not list it. LaptopMedia and Technical City give 672 GB/s. Notebookcheck lists "677 GB/s" next to "32000 effective", which does not match 192 bit (that would be 768 GB/s). https://www.notebookcheck.net/Nvidia-GeForce-RTX-5070-Ti-Laptop-Benchmarks-and-Specs.934945.0.html |
| G14 GA403WR GPU power | "Turbo mode: 1550MHz at 110W(...85W+25W Dynamic Boost)" | VERIFIED, ASUS spec https://rog.asus.com/us/laptops/rog-zephyrus/rog-zephyrus-g14-2025/spec/ |
| RTX 5070 Ti Laptop, **effective** decode bandwidth in llama.cpp | **about 379 GB/s** (56% of 672), fixed cost about 0.86 ms/token | DERIVED from Hypercompute's own idle runs (docs/RESULTS.md, `final-*`): Qwen2.5-7B Q4_K_M 80.6 tok/s with a 4168 MiB CUDA0 buffer, Qwen2.5-0.5B 526.7 tok/s with 374 MiB. Two-point fit `t = t0 + c*bytes` gives `c = 2.64 ms/GB`, `t0 = 0.86 ms` |
| RTX 5070 Ti Laptop prefill | 3728 tok/s for Qwen2.5-7B (long prompt), about **9.6 us per layer-token** | DERIVED from docs/RESULTS.md `final-7b-local` (28 layers) |
| Apple M4 (MacBook Air 13", 2025) | "120GB/s memory bandwidth", 10-core CPU, "8-core GPU, 10-core GPU", 16/24/32 GB | VERIFIED, Apple tech specs https://support.apple.com/en-au/122209 (apple.com/macbook-air/specs now shows the M5 model, 153 GB/s) |
| M4 in llama.cpp | 10-core GPU, Llama 2 7B Q4_0 (3.56 GiB): pp512 **221.29 t/s**, tg128 **24.11 t/s**. 8-core GPU: no data | VERIFIED as published, llama.cpp discussion #4167 table (commit 8e672ef, old) https://github.com/ggml-org/llama.cpp/discussions/4167 |
| M4, effective decode bandwidth | **about 92 GB/s** (77% of 120), `c_mac = 10.85 ms/GB` | DERIVED from the #4167 row. UNVERIFIED for the friend's machine (GPU core count, current build, fanless thermal throttling) |
| M4 prefill | about **163 us per Qwen2.5-7B layer-token** (141 us per Llama-2-7B layer-token, scaled by layer params 233M/202M) | DERIVED. About 17x the CUDA figure |
| M4 memory usable by Metal | total = `recommendedMaxWorkingSetSize` (below physical RAM); free = total minus this process's own allocations | VERIFIED code semantics (llama_rpc.md §5). The actual value on a 16 GB Air is UNVERIFIED: read it from the rpc-server `Devices:` line or `--list-devices` |
| AMD Ryzen AI 9 HX 370 memory | "DDR5 (FP8), LPDDR5X (FP8)", 2 channels, "DDR5-5600, LPDDR5x-8000", max 256 GB | VERIFIED, AMD product page https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html |
| G14 GA403WR RAM | "32GB LPDDR5X 8000 on board", "Support dual channel memory" | VERIFIED, ASUS spec page above |
| HX 370 peak bandwidth | **128 GB/s** if the bus is 128-bit (128/8 x 8000 MT/s) | DERIVED; the 128-bit width is UNVERIFIED (AMD lists only "2 channels"). Effective llama.cpp CPU decode bandwidth is UNVERIFIED (llama_cpu_tier.md §4.2) |
| Mac Wi-Fi | "Wi-Fi 6E (802.11ax)" | VERIFIED, Apple 122209 |
| Host Wi-Fi | "Wi-Fi 7(802.11be) (Triple band) 2*2" | VERIFIED, ASUS spec page. The link between the two machines is therefore 802.11ax at best |
| 2x2 802.11ax PHY | 2400 Mbps at 160 MHz (MCS 11) on recent Macs | VERIFIED for listed Macs, Apple "Wi-Fi and Ethernet specifications" https://support.apple.com/en-ie/guide/deployment/dep268652e6c/web. The MacBook Air M4 row was not captured, so UNVERIFIED for this machine |
| Real Wi-Fi 6E TCP | 1.7 Gbps downlink / 1.2 Gbps uplink at 160 MHz near the AP (Intel AX210 laptops, Asus AP) | VERIFIED as reported by the WBA/CableLabs/Intel trial, Lightwave, 2021-12-02 https://www.lightwaveonline.com/broadband/article/55031652/wba-field-trial-confirms-wi-fi-6e-6-ghz-benefits-for-home-networks. This is a best case for laptop to wired peer |
| Laptop to laptop through one AP | each frame crosses the air twice (Mac to AP, AP to host), so expect well under the figure above | UNVERIFIED (standard infrastructure-mode behaviour without TDLS; not measured) |
| Wi-Fi LAN RTT | no authoritative figure found. Planning assumption: 2-5 ms median, with tail spikes from power save and contention | UNVERIFIED. Must be measured per session (§5a). The examples below use **RTT 3 ms and 300 Mbit/s (37.5 MB/s)** and label them as assumed |

---

## 2. What crosses the network per token (ggml-rpc at b11770)

### 2.1 Message types (VERIFIED)

- Commands with no output are **one-way**. `send_rpc_cmd` without output writes cmd byte, size and payload and does not
  read anything (`ggml/src/ggml-rpc/ggml-rpc.cpp:372-386`). Commands with output read a response (`:388-405`).
- One dispatcher thread per endpoint serializes all commands on one TCP stream (`:660-697`). `send()` blocks only until
  the bytes are written (`:547-557`). `send()` with output blocks until the response arrives (`:569-579`). `send_async`
  only queues (`:559-567`, `:581-589`). `synchronize()` is a **local** barrier: it pushes `RPC_CMD_NONE`, which is
  never sent on the wire (`:621-626`, `:675`).
- Graph compute is fire-and-forget. `graph_compute` sends `GRAPH_RECOMPUTE {device, uid}` (a few bytes) if the server
  already holds the graph uid, otherwise the full serialized graph with `GRAPH_COMPUTE`, both through `send_async`
  (`:1242-1268`). The serialized graph costs `n_tensors * sizeof(rpc_tensor)`, and `rpc_tensor` is 296 bytes packed
  (`:36-53`).
- The server runs graphs asynchronously (`ggml_backend_graph_compute_async`, `:2191`, `:2210`), and every data
  command synchronizes first. Both `set_tensor` and `get_tensor` begin with `sync_all_backends()` (`:1698-1699`,
  `:1922-1923`, definition `:2215-2220`). **A `GET_TENSOR` reply therefore waits for the remote graph to finish.**
- The RPC backend has no `cpy_tensor_async` (`:1306`). The RPC buffer `cpy_tensor` succeeds only when both tensors are
  on the **same** server, otherwise it falls back (`:930-951`). CUDA's buffer `cpy_tensor` accepts only CUDA sources
  (`ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_backend_cuda_buffer_cpy_tensor`).
- Every TCP socket sets `TCP_NODELAY` (`ggml/src/ggml-rpc/transport.cpp:698-703`, `:716`, `:757`), and each command is
  written as 3 `send()` calls (cmd, size, payload; `ggml-rpc.cpp:374-385`), replies as 2 (`:323-331`). Each command can
  therefore become several small segments on the air. Open PR #24122 "rpc: reduce small-message overhead and tune
  cache probing" proposes coalescing these writes (https://github.com/ggml-org/llama.cpp/pull/24122). The per-message
  cost over Wi-Fi is UNVERIFIED.

### 2.2 How the scheduler moves activations (VERIFIED)

`ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp:1865-1940`) walks the splits in order and copies each
split's inputs with `ggml_backend_sched_copy_input` (`:1823-1863`):

- Graph inputs (`GGML_TENSOR_FLAG_INPUT`: positions, KQ mask, KV write indices, ...) are placed on the CPU backend
  (`:1093-1096`) and copied into **every** split that reads them, through `ggml_backend_tensor_copy` (`:1829-1838`).
  From host memory that is a `set_tensor`, so a one-way `SET_TENSOR` for an RPC split.
- Other inputs (the residual stream from the previous split) try `cpy_tensor_async` first. With RPC on either side it
  fails, so the scheduler synchronizes and calls `ggml_backend_tensor_copy` (`:1851-1861`). That function does
  `set_tensor` if the source is host memory, `get_tensor` if the destination is host memory, otherwise `cpy_tensor` or
  a host bounce (`get` then `set`) (`:628-649`).
- The token embedding stays on the CPU (llama_placement.md §3), so the first device always receives the embedded
  rows from the host.
- Logits: after the graph, `llama_context` calls `ggml_backend_tensor_get_async(backend_res, t_logits, ..., n_outputs*n_vocab*sizeof(float))`
  (`src/llama-context.cpp:2049-2060`) unless every sequence uses backend sampling (`needs_raw_logits`, `:1804-1818`).
  Backend sampling is off by default ("enable backend sampling (experimental) (default: disabled)", `common/arg.cpp:2310-2315`;
  `common/common.h:299`). If the output device is remote, this is a `GET_TENSOR` of `n_vocab*4` bytes per generated token.

### 2.3 Round trips per decoded token

| Transition | Network traffic | Blocking RTTs |
|---|---|---|
| host CPU -> RPC device (embedding rows, graph inputs) | one `SET_TENSOR` per input tensor, one-way | 0 |
| graph on RPC device | `GRAPH_RECOMPUTE` (graph reused) or `GRAPH_COMPUTE` (new uid), one-way | 0 |
| RPC device -> local GPU | `GET_TENSOR` of `n_embd*4*n_tokens` bytes, then a local H2D copy | **1** (includes the remote compute) |
| local GPU -> RPC device | local D2H, then `SET_TENSOR`, one-way | 0 |
| RPC server A -> RPC server B | `GET` from A, then `SET` to B (host bounce, `ggml-rpc.cpp:937-940`) | **1** |
| logits from an RPC device that holds the output | `GET_TENSOR` of `n_vocab*4*n_outputs` bytes | **1** |
| weights (load time only) | `SET_TENSOR_HASH` probe for weights > 10 MiB (`:118`, `:844-846`), then `SET_TENSOR` on a cache miss | 1 per large tensor, at load only |

**Rule (DERIVED from the table): blocking RTTs per token = number of RPC devices in the chain.** Each RPC device's
output has to be read back once, either to feed the next device or as logits. The order does not change the RTT
count, but it decides whether the last read carries `n_embd*4` bytes or `n_vocab*4` bytes.

For Qwen2.5-7B (`n_embd 3584`, `n_vocab 152064`, `n_head_kv 4`, `head_dim 128`; Hugging Face `config.json`):
`n_embd*4 = 14,336 B`, `n_vocab*4 = 608,256 B`.

Decode-step message list for RPC-first `[RPC0, CUDA0]` (DERIVED. Count and sizes UNVERIFIED until checked with
`GGML_RPC_DEBUG=2`, which logs every command with its in/out sizes, `ggml-rpc.cpp:676-677`):
`SET_TENSOR` x about 5 (embedding row 14 KB, `inp_pos` 4 B, KQ mask `n_kv*2` B with flash attention, `k_idxs`/`v_idxs`)
-> `GRAPH_RECOMPUTE` -> `GET_TENSOR` (14 KB back). The KQ mask is `n_kv x n_tokens` with no padding on the token axis
(`src/llama-graph.cpp:30-46`), and `n_kv` is the used cells rounded up to 256 (`src/llama-kv-cache.cpp:1282-1293`).

Graph re-upload: `llama_context` reuses the previous graph when `can_reuse` passes (`src/llama-context.cpp:1521`). The
KQ mask shape check fails whenever `n_kv` changes (`src/llama-graph.cpp:49-63`), which happens every 256 tokens, and
each scheduler split pass assigns new uids (`ggml-backend.cpp:1748-1751`). So roughly every 256 tokens the RPC device
gets a full `GRAPH_COMPUTE` (a few hundred bytes per tensor). The size per layer is UNVERIFIED; `GGML_RPC_DEBUG=2`
prints `serialized graph: <size>` (`ggml-rpc.cpp:1264`).

### 2.4 Prefill (per ubatch of `u` tokens)

Same message pattern per ubatch, with payloads scaled by `u`: up `u*n_embd*4` (embedding rows) + `u*n_kv*2` (mask,
f16 with flash attention, f32 without; `llama-graph.cpp:39-42`), down `u*n_embd*4`. For Qwen2.5-7B with `u = 512`
and `n_kv = 2048` that is 7.3 + 2.1 + 7.3 = **16.8 MB per ubatch per RPC device** (DERIVED).

Pipeline parallelism is enabled with RPC devices. It needs `n_devices > 1`, full offload, `-sm layer`, KV offload,
no tensor overrides and no MoE cache (`src/llama-context.cpp:517-523`), plus `async` and `events` caps on every non-CPU
device (`:525-541`), which RPC devices report (llama_placement.md §1.2). The scheduler then keeps
`GGML_SCHED_MAX_COPIES = 4` input copies (`ggml-backend.cpp:912`, `:1973`). The client thread still blocks in
`GET_TENSOR` while the remote device computes, so the overlap is "remote ubatch k+1 while CUDA runs ubatch k". How
much overlap this gives over Wi-Fi is UNVERIFIED. A comment in discussion #20252 claims RPC devices disable pipeline
parallelism; the source above contradicts that.

---

## 3. Cost model

### 3.1 Notation

- `L` repeating layers. `w_i` = bytes of `blk.i.*` (from GGUF, already in Hypercompute's `ModelShape.layer_bytes`).
  `w_out` = output head + `output_norm`. `kappa_i` = KV bytes per cell for layer i = `n_head_kv*(d_k + d_v)*bytes(type_kv)`
  (2048 B for Qwen2.5-7B with f16 KV). `P_i` = parameters of layer i.
- Devices in llama.cpp order `d = 1..D` (`-dev` order). `L_d` = contiguous layer range on d. The last device also
  holds the output head (llama_placement.md §3). Optional CPU prefix of `n_cpu` layers via `-ngl < all`.
- Per device: `c_d` decode seconds per byte, `t0` fixed per-token cost, `q_d` prefill seconds per parameter-token,
  `M_d` usable memory, `C_d` compute buffer, `R_d` safety margin.
- Per remote device r: `RTT_r`, `BW_r` (bytes/s per direction, client to worker), `m_r*delta_r` per-message overhead.
- Workload: prompt `N_p`, generated `N_g`, context `n_ctx`, ubatch `u` (default 512).

### 3.2 Decode, one token at depth t (single stream, layer split)

```
n_kv(t)   = min(n_ctx, max(256, ceil256(t + 1)))
W_d       = sum_{i in L_d} w_i  + [d == D] * w_out
K_d       = sum_{i in L_d} kappa_i
T_dec(t)  = t0 + sum_d c_d * (W_d + K_d * n_kv(t))                      # bandwidth-bound compute, all devices in sequence
               + sum_{r remote} ( RTT_r + (A_up_r(t) + A_dn_r) / BW_r + m_r * delta_r )
A_up_r(t) = 4*n_embd + 2*n_kv(t) + O(100 B)                               # residual or embedding row, f16 mask, pos, idx
A_dn_r    = 4*n_embd   if r does not hold the output
          = 4*n_vocab  if r holds the output (logits)
```

Terms are summed because one token visits the devices one after another. Nothing overlaps for a single stream.
KV reads matter at long context: at 32k, Qwen2.5-7B reads 64 MiB of KV per layer per token, about half a layer's
weights.

### 3.3 Prefill, prompt of N_p tokens

```
n_ub      = ceil(N_p / u)
S_d(u)    = u * q_d * sum_{i in L_d} P_i   (+ c_d * W_d, small for u >= 256 on a GPU)
H_r(u,t)  = RTT_r + (2*u*4*n_embd + u*n_kv(t)*bytes_mask) / BW_r
stage_r   = H_r + S_r       (remote stage: the client blocks on the GET)
stage_loc = S_local
T_serial  = sum over ubatches of (sum of all stages)
T_overlap = (sum of stages of one ubatch) + (n_ub - 1) * max(stage)
T_pp      = lambda * T_overlap + (1 - lambda) * T_serial      # fit lambda in [0,1] from measurements
```

### 3.4 Worked numbers (DERIVED; the network values are ASSUMED: RTT 3 ms, 37.5 MB/s)

Qwen2.5-7B Q4_K_M (fits on CUDA alone). Mid-layer size `w = 0.139 GB` (from RESULTS.md: CUDA0 24L/3173 MiB vs
17L/2246 MiB gives 927 MiB for 7 layers). LM head about 0.447 GB (Q6_K at 6.5625 bpw, assumed; it matches the
`RPC0 2L/711 MiB` row). `c_cuda = 2.64`, `c_mac = 10.85` ms/GB.

| Layout (k = Mac layers) | Decode ms/token | tok/s |
|---|---|---|
| local only (measured, RESULTS.md) | 12.4 | 80.6 |
| RPC-first `[RPC0 0..7, CUDA0 8..27 + head]` | 12.41 + 8 x 1.14 + **4.0** (RTT + 37 KB) = **25.5** | 39 |
| Hypercompute today `[CUDA0 0..19, RPC0 20..27 + head]` | 12.41 + 9.12 + 3.67 (head on Mac) + **19.8** (RTT + 608 KB logits) = **45.0** | 22 |

Prefill with `k = 8`, `N_p = 2048`, `u = 512`: Mac stage = 512 x 8 x 163 us = 666 ms compute + 447 ms network
(16.8 MB) + RTT, about 1.12 s. CUDA stage = 512 x 20 x 9.6 us = 98 ms. So `T_pp` is about 4.6 s with full overlap,
against 0.55 s local. **Conclusion: when the model fits in VRAM, putting layers on the Mac only adds latency.**

Overflow case (model does not fit in VRAM; X GB of layers must leave the GPU). The decode cost of the overflow is
`X*c_mac + H` on the Mac or `X*c_cpu` on the host CPU tier (no network hop; CPU layers are the first layers, and
llama_cpu_tier.md covers how llama.cpp handles them). Break-even overflow `X* = H / (c_cpu - c_mac)` when
`c_cpu > c_mac`. Example: with `c_cpu = 12.5 ms/GB` (80 GB/s effective, UNVERIFIED) and `H = 4 ms`, `X* = 2.4 GB`, so
the Mac wins for decode beyond 2.4 GB of overflow and the CPU wins below that. Prefill can go the other way, because
llama.cpp can run large CPU-resident batches on the GPU over PCIe (op offload, llama_cpu_tier.md) while Mac layers pay
Wi-Fi bytes for every ubatch. This has to be measured; do not hard-code the choice.

### 3.5 Objective and constraints

```
J(split)  = T_pp(N_p) + sum_{t = N_p}^{N_p + N_g - 1} T_dec(t)
          ~ T_pp(N_p) + N_g * T_dec(N_p + N_g/2)                          # one typical request, end-to-end latency
s.t. for every device d:
      W_d + K_d * n_ctx + C_d + R_d <= M_d                              # weights + KV at full context + compute buffer + margin
      L_d contiguous, in device order; the last device holds the output; counts sum to L
```

Moving layer i from the local GPU to a slower device changes J by
`dJ_i = N_g * (c_slow - c_gpu) * (w_i + kappa_i * n_kv) + n_ub * u * (q_slow - q_gpu) * P_i > 0`.
J therefore increases with every layer moved, plus a fixed `N_g*H + n_ub*H_pp` the moment any remote device is used.
**The optimum is the minimum number of off-GPU layers that satisfies memory, placed on the cheaper overflow tier.**
The open questions are only which tier takes the overflow (Mac, CPU, or both) and how much safety margin to keep,
and those need measurements.

For a different objective (several concurrent requests, maximum pipelined throughput), equalize stage times instead:
the share of the fast device is `s* = t_slow / (t_fast + t_slow)` (discussion #30671). Hypercompute should not use
that for single-user latency.

---

## 4. Prior art

### 4.1 EXO (memory-weighted, no speed term). VERIFIED

- Current main: candidate cycles must have total `ram_available >= model storage_size`
  (`src/exo/master/placement_utils.py:21-37`; `placement.py:229-233`). Only the **smallest** cycles are kept
  (`placement_utils.py:40-44`, `placement.py:277`), so a model that fits on one node stays on one node. Cycles with
  leaf nodes are preferred (`placement.py:323-332`). Among the rest, the winner maximizes (download progress,
  total free RAM) (`placement.py:334-345`).
- Layers are allocated **in proportion to `ram_available`** with largest-remainder rounding and at least 1 layer per
  node (`allocate_layers_proportionally`, `placement_utils.py:47-75`, called at `:102-107`). Each node is then checked
  with `storage * node_layers / n_layers <= ram_available` (`:109-120`). Pure pipeline shards are contiguous in cycle
  order (`:203-240`).
- `grep -i "flops|bandwidth|latency"` over placement.py and placement_utils.py finds nothing. Compute speed,
  bandwidth and link latency play no part in placement.
- Legacy EXO (`v0.0.15-alpha`): `RingMemoryWeightedPartitioningStrategy` sorts nodes by memory and gives each
  `memory/total_memory` of the layers (`exo/topology/ring_memory_weighted_partitioning_strategy.py:7-18`).
- Takeaway: the "smallest cycle" rule is right (do not split if you do not have to). The memory-proportional ratio is
  wrong for a 4x-slower node.

### 4.2 distributed-llama (equal tensor parallel). VERIFIED

- "You can run Distributed Llama only on 1, 2, 4... 2^n nodes", and the node count is capped by the number of KV
  heads (`README.md:48-49`). Root and workers each process "own slice of the neural network" (`README.md:67-70`).
- Slices are equal: `assert(d % nNodes == 0); s.d0 = d / nNodes;` (`src/nn/nn-core.cpp:220-230`, and `n0 = n / nNodes`
  at `:232-243`). There is no heterogeneity weighting, so every layer waits for the slowest node, and every layer
  synchronizes over the network.
- Takeaway for Hypercompute: tensor parallelism over Wi-Fi multiplies the per-token RTT count by the number of layers.
  llama.cpp's own `-sm tensor` RPC allreduce needs two RPC backends (llama_rpc.md §7) and does not fit a CUDA-host +
  Mac-worker pair anyway.

### 4.3 Petals (throughput- and latency-aware routing). VERIFIED

- Each server measures its compute per block once (`inference_rps` with 1-token steps, `forward_rps` with 1024-token
  batches) and its network (`hidden_size * 16` bits per token from a speedtest), and **caches** the result in
  `throughput_v5.json` keyed by model, device name, dtype and TP devices (`src/petals/server/throughput.py:55-92`,
  `:111-175`). Reported throughput is `min(forward_rps / ((num_blocks + 1) / 2), network_rps)` (`:96-103`).
- Servers choose which blocks to host so that the swarm's minimum per-block throughput is maximized
  (`src/petals/server/block_selection.py:23-33`, rebalancing at `:40-95`).
- The client picks the **minimum-latency path** with Dijkstra over a graph whose edges are `1/inference_rps` per block,
  `rtt/2` plus an empirically measured 18 ms overhead per hop, and a 10 s penalty if KV cache is short
  (`src/petals/client/routing/sequence_manager.py:217-279`, `_rtt_to_delay` `:281-292`).
- Takeaway: this is the same additive "per-block compute + per-hop network" model as §3, with measured and cached
  coefficients. Hypercompute only has a 2-3 node chain, so it can enumerate instead of running Dijkstra.

### 4.4 llama.cpp itself

- Default split is proportional to free memory: "distributes model weights and the KV cache ... in proportion to each
  device's available memory", which `--tensor-split` overrides (`tools/rpc/README.md:83-84`; code in
  llama_placement.md §1.5). `--fit` fills devices back to front (llama_placement.md §4). With RPC-first ordering
  that fills the local GPU first, which is close to the latency-optimal rule, but it uses reported free memory
  (Metal semantics differ) and a fixed 1 GiB margin, and its tensor overrides turn off pipeline parallelism.
- Discussion #30671 "Speed-Aware Asymmetric Tensor Split for RPC nodes" (2026-10-16): two loopback CPU rpc-servers
  (12 threads vs 1 thread), Qwen2.5-0.5B, build b10743. It reports that single-stream generation "pays the sum of node
  times plus a per-hop overhead", that the memory-proportional default sat 13-28% of layers away from the measured
  balance points, and proposes measuring per-device speed at load time with two objectives (minimum latency = fill the
  fastest device first, subject to memory; maximum throughput = equalize stage times). Third-party, CPU-only and
  loopback, not reproduced here. https://github.com/ggml-org/llama.cpp/discussions/30671
- Discussion #18873, maintainer reply (am17an) on adding a heterogeneous RPC node: "The performance most likely will be
  dominated by your network". https://github.com/ggml-org/llama.cpp/discussions/18873
- Discussion #20252 and PR #6017 ("llama : add pipeline parallelism support", merged 2024-03-13): ggerganov says PP
  "scales nearly linear with the number of devices, even for single request" when configured correctly. That is
  prompt processing, not decode.
- Open issue #28487 "llama.cpp RPC deadlocks with 3+ endpoints; 2 works reliably". Treat 3-device chains as risky.
- No llama.cpp feature for speed-aware `--tensor-split` exists at b11770 (VERIFIED by absence: only the memory
  default, `-ts`, and `--fit`).

---

## 5. Recommended optimiser for Hypercompute

### 5a. Calibration (all with llama.cpp itself, from the host)

llama-bench options at b11770 (VERIFIED, `tools/llama-bench/llama-bench.cpp`):
- `-rpc/--rpc` registers servers at parse time (`:738-749`, `register_rpc_server_list` `:176-196`). `-dev` resolves
  names at parse time (`parse_devices_arg` `:144-174`), so **`-rpc` must come before `-dev`**.
- `-dev RPC0/CUDA0,CUDA0/RPC0`: `/` separates devices in one test, `,` separates test values (`:675-692`,
  `split_delim = ','` `:530`). The same goes for `-ts 4/25,8/21` (`:919-940`; values within a combo split on `;` or `/`).
- `-ngl` default `-1` (`:399`), which means all layers plus output at the llama level. `-fitt`/`-fitc` default 0, so
  **fit is off** unless set (`:414`, `do_fit` `:2329-2330`). `-r` default 5 (`:417`). Also `-p`, `-n`, `-pg pp,tg`,
  `-d` (depth), `-ub`, `-fa on|off|auto`, `-o csv|json|jsonl|md|sql` (usage `:437-485`).
- The model is reloaded only when model params change (`equal_mparams` `:1296-1302`, reuse `:2365`). Every new `-ts`
  reloads, so run the Mac rpc-server with `-c` to make reloads hit its local weight cache (llama_rpc.md §4).
- The jsonl fields include `devices`, `tensor_split`, `n_prompt`, `n_gen`, `n_depth`, `avg_ns`, `stddev_ns`,
  `avg_ts`, `stddev_ts`, `samples_ns`, `samples_ts` (`:1604-1609`; example in `tools/llama-bench/README.md:330`).
- llama-bench holds the worker exclusively while it runs, like any client (llama_rpc.md §1, C1). Run it only on a
  leased, idle worker.

Steps (about 2-5 minutes per model once the Mac's RPC cache is warm; UNVERIFIED):

1. **Static shape** from GGUF: `w_i`, `w_out`, `kappa_i`, `P_i`, `n_embd`, `n_vocab` (extend `ModelShape` with `P_i`,
   `n_vocab`, `kappa_i`).
2. **Memory**: the existing `--rpc mac:50052 --list-devices` probe for RPC0 total/free (Metal working set) and CUDA0.
   Set `M_d = free - R_d` with `R_d` = max(1 GiB, 10%). On the Mac also respect macOS memory pressure (UNVERIFIED
   behaviour under swap).
3. **Local GPU coefficients** (no network):
   `llama-bench -m M.gguf -dev CUDA0 -ngl -1 -fa on -p 512 -n 128 -d 0,4096 -r 3 -o jsonl`
   - `T_cuda(t)` = `1/avg_ts` of the `n_gen` rows. Slope between `-d 0` and `-d 4096` gives the KV term, and with a
     second, smaller model of the same family, `c_cuda` and `t0` (as in §1).
   - `q_cuda` = (1/`avg_ts` of the pp512 row) / `sum P_i`.
4. **Mac + link coefficients through the real path** (no Mac-side tooling needed):
   `llama-bench -m M.gguf -rpc <mac-ip>:50052 -dev RPC0/CUDA0 -ngl -1 -fa on -ts 1/28,6/23 -p 512 -n 128 -d 0,4096 -r 3 -o jsonl`
   (units are `L + 1`, so 29 for a 28-layer model; the last unit is the head).
   With `T(k)` the per-token time at `k` Mac layers:
   ```
   slope  b   = (T(k2) - T(k1)) / (k2 - k1)          = wbar * (c_mac - c_cuda)  ->  c_mac
   offset H   = T(k1) - T_cuda - k1 * b                                          ->  per-token hop cost
   prefill: same two points on the pp rows give q_mac and H_pp (per ubatch)
   ```
   Cross-check by running `GGML_RPC_DEBUG=3` once: the host logs `"<CMD> done in X ms"` for every command
   (`ggml-rpc.cpp:685-690`), which gives the `GET_TENSOR` latency distribution directly.
   Optional cross-check on the Mac itself: `llama-bench -dev MTL0 -ngl -1 ...` if the agent can run it.
5. **Order check** (one run): `-dev RPC0/CUDA0,CUDA0/RPC0 -ts 6/23,23/6` measures the logits penalty of §2.3 directly.
   That is 4 combinations; only the 2 that keep 6 Mac layers are meaningful.
6. **Raw link** (cheap, every session and before every deployment): median and p90 RTT, and throughput, through
   Hypercompute's own agent channel. Never probe the rpc-server port itself, because it serves one client at a time.
7. **CPU tier** (only when overflow is possible): `-dev CUDA0 -ngl <L+1-k>` for two values of k gives `c_cpu` and
   `q_cpu` the same way (llama_cpu_tier.md lists the thread settings to sweep).

### 5b. Choosing candidates (cost model)

1. Enumerate structures: `{CUDA}`, `{CPU+CUDA}`, `{Mac,CUDA}`, `{CPU+Mac,CUDA}`. The local GPU is always last and
   holds the head; RPC devices come first (llama.cpp's own order). The Mac is never last.
2. For each structure enumerate the contiguous split points. Two devices is 1-D (k from `k_min` to `k_max`); with a CPU
   prefix it is 2-D, at most `(L+1)^2`, about 4,000 evaluations for 64 layers. Drop anything that breaks §3.5's
   memory constraint, using real `w_i` and not the average (Q4_K_M gives some layers more bits, so sizes vary).
3. Score with `J` from §3.5, using the request mix Hypercompute actually sees (defaults `N_p = 1024`, `N_g = 256`) or a
   user choice between "fast first token" (`N_g` small) and "fast streaming" (`N_g` large).
4. Keep the top 3 feasible candidates by `J`, plus `k_min + 2` as a margin check (CUDA OOM risk is highest at
   `k_min`). Expected result: `{CUDA}` whenever it fits; otherwise the smallest overflow on the cheaper tier.

### 5c. Empirical check (short benchmarks)

- One llama-bench process per candidate, so a load failure (OOM) stops only that candidate:
  `llama-bench -m M.gguf -rpc <mac>:50052 -dev RPC0/CUDA0 -ngl -1 -fa on -ts <k>/<L+1-k> -pg <N_p>,<N_g> -n 64 -d <N_p> -r 3 -o jsonl`
  (`-pg` measures one prompt+generate request; `-n 64 -d N_p` measures decode at realistic depth).
  Order the candidates from the most conservative memory use to the least.
- Measured `J = N_p/pp_rate + N_g/tg_rate` (from `avg_ns` / `samples_ns`). Pick the measured best only if it beats the
  model's pick by more than 2 standard deviations; otherwise keep the pick with fewer Mac layers (less network
  exposure).
- Feed the residuals back: refit `c_mac`, `H`, `lambda` by least squares on all (predicted, measured) pairs for this
  node pair, so the next plan needs fewer probes.
- Time budget: 3 candidates x (load from cache + about 20 s of tests), roughly 1-3 minutes (UNVERIFIED; the first
  uncached load of the Mac's layers over Wi-Fi takes `bytes_on_mac / BW`, for example 2 GB at 37.5 MB/s = 53 s).

### 5d. Profiles: storage and reuse

Store JSON docs in a new `profiles` table with the same `(id TEXT PRIMARY KEY, created_at TEXT, doc TEXT)` shape as
`benchmarks` and `plans` (`backend/hypercompute/storage.py:31-32`). Three kinds:

```jsonc
// device_profile: one per (node, device, engine build, model)
{ "kind": "device", "node_id": "...", "device": "CUDA0|MTL0|CPU", "engine": "llama.cpp b11770 39da130f",
  "driver": "591.84", "model_sha256": "...", "quant": "Q4_K_M", "fa": "on", "type_k": "f16", "type_v": "f16", "n_ubatch": 512,
  "c_decode_s_per_byte": 2.64e-12, "t0_s": 8.6e-4, "kv_s_per_cell_byte": ..., "q_prefill_s_per_param_token": ...,
  "mem_total": ..., "mem_free_at_measure": ..., "samples": [...], "measured_at": "...", "thermal_note": "fanless|ac|battery" }
// link_profile: one per (host node, worker node, network)
{ "kind": "link", "host": "...", "worker": "...", "ssid": "...", "bssid": "...", "band": "6GHz/160MHz",
  "rtt_p50_s": ..., "rtt_p90_s": ..., "bw_bytes_per_s": ..., "hop_decode_s": ..., "hop_prefill_s_per_ubatch": ...,
  "measured_at": "..." }
// split_result: one per measured candidate
{ "kind": "split", "model_sha256": "...", "devices": ["RPC0","CUDA0"], "tensor_split": [6,23], "ngl": "all",
  "n_ctx": 4096, "test": "pp1024+tg256", "avg_ts": ..., "stddev_ts": ..., "samples_ns": [...],
  "predicted_J_s": ..., "measured_J_s": ..., "link_profile_id": "...", "device_profile_ids": ["...","..."] }
```

Reuse rules (Petals keys its cache by model, device and dtype the same way, `throughput.py:65-70`):
- A device profile stays valid until the engine tag, the driver, the model hash, the KV type or the flash-attention
  setting changes. Re-measure if the newest decode sample is off by more than 15% (power mode, battery, thermal).
- A link profile is short-lived. Re-check RTT at every deployment, and redo step 5a.4 only if the BSSID or band changed
  or the RTT median moved by more than 30%.
- Split results are reused as warm starts. When the model, ctx and profiles match, launch the stored best split
  directly and skip 5c.

### 5e. Changes to make in Hypercompute's planner (in order)

1. `order_devices` (`planner.py:198-200`): put remote devices first and the local accelerator **last**, so `_layout`
   gives the head to the local GPU. Keep `--device` explicit; this matches llama.cpp's default order.
2. Replace capacity-proportional counts (`_counts_proportional`, `planner.py:104-121`) for single-user plans with
   "fill the local GPU to `M_d`, then send overflow to the cheaper tier by measured cost". `_counts_greedy` already
   fills front to back, so after the order flip it would fill the Mac first. It needs reversing.
3. Add the calibration (5a), the scorer (5b), the check (5c) and the `profiles` table (5d). Until profiles exist, use
   the §1 coefficients as priors.
4. Check every new plan's launch log (`-lv 4` is already passed) for which buffer holds the head, so an ordering
   regression shows up.

---

## 6. UNVERIFIED / open items

| Item | Why it matters | How to settle it |
|---|---|---|
| Wi-Fi RTT and throughput between these two laptops (and AP double-hop) | `H`, `H_pp`, logits penalty | step 5a.4 and 5a.6 on the real network |
| RTX 5070 Ti Laptop peak bandwidth (672 vs 677 GB/s; NVIDIA lists only the bus) | only the efficiency ratio; the optimiser uses measured `c_cuda` | not needed if measured |
| HX 370 bus width (128-bit) and llama.cpp CPU decode bandwidth | overflow tier choice (§3.4) | step 5a.7 |
| M4 Air GPU core count (8 vs 10), current-build speed, fanless throttling under sustained load | `c_mac`, `q_mac` | step 5a.4 with `-r 3` and a sustained run; record `thermal_note` |
| Metal `recommendedMaxWorkingSetSize` on a 16 GB M4 | Mac memory budget | rpc-server `Devices:` line / `--list-devices` |
| Exact `SET_TENSOR` count per decode step and graph re-upload size | small-message overhead over Wi-Fi | `GGML_RPC_DEBUG=2` on the host for one generation |
| How much prefill ubatches overlap with an RPC stage (`lambda`) | TTFT prediction | fit `lambda` from 5c pp rows |
| Qwen2.5-7B Q4_K_M head type (Q6_K assumed) | §3.4 example only | read the GGUF tensor types (gguf.py) |
| Coefficients from discussion #4167 (old commit) and #30671 (loopback CPU) | priors only | replaced by measurements |
| 3+ device chains (issue #28487 deadlock report) | multi-peer plans | test before enabling more than one RPC worker |
