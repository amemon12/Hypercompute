# llama.cpp host-RAM (CPU) tier audit (b11770)

Pinned source: ggml-org/llama.cpp tag `b11770` = commit `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb` (same ref as
`llama_placement.md`). Every file was read with
`gh api -H "Accept: application/vnd.github.raw" "repos/ggml-org/llama.cpp/contents/<path>?ref=39da130..."`.
No llama.cpp code was built or executed. Line numbers are for the pinned commit.

This file extends `llama_placement.md` (device order, `-ts` mapping, fit, KV sizing, log verbosity). It does not repeat
those sections. It adds what changes when some layers stay in host RAM.

Status words:
- VERIFIED: read directly in source at the pinned ref, or seen in a real Hypercompute log already on disk.
- INFERRED: follows from the code, but not exercised on hardware.
- UNVERIFIED: not confirmed, with the reason.

Real-log evidence used below comes from llama-server logs that earlier Hypercompute runs on this laptop saved inside
`C:/Users/operator/hypercompute/results/*.json` (engine `~/.hypercompute/engines/llama-b11770-windows-x64-cuda13.4`).
Nothing was run for this audit.

---

## 0. TL;DR for Hypercompute

1. `-ngl N` counts **units** = `n_layer + 1` (the +1 is the output head). With `1 <= N <= n_layer+1`, the **first**
   `k = n_layer + 1 - N` repeating layers (`blk.0 .. blk.k-1`) stay on the host CPU, and the last `N` units go to the
   `--device` list. The output head is always one of the offloaded units, so it stays on a GPU/RPC device unless `N = 0`.
   `token_embd` is always on the CPU. (VERIFIED, `src/llama-model.cpp:1667-1692`, `2418-2421`)
2. `--tensor-split` divides only the `N` offloaded units, not all layers. Integer splits that sum to `N` are reproduced
   exactly. Hypercompute can keep its current integer `-ts` and replace `-ngl all` with `-ngl <sum of -ts>`.
3. **`--no-mmap` and `--mlock` do not exist at b11770.** They were replaced by `-lm/--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`.
   Passing `--no-mmap` fails parsing with `error: invalid argument: --no-mmap` (exit 1). (VERIFIED, `common/arg.cpp:829-831`, `2694-2711`)
4. Default `-lm auto` resolves to **mmap** on this host (CUDA dGPU and RPC both report `mmap_support = true`). CPU-layer
   weights then live in **`CPU_Mapped`** buffers: file-backed pages, not a private copy. Real logs on this laptop already
   show `CPU_Mapped model buffer size` lines for `token_embd`. (VERIFIED)
5. **Repack (`CPU_REPACK`) is effectively off whenever CUDA0 is in the device list**, because the CPU buffer-type list
   puts `CUDA_Host` before `CPU_REPACK`, and under mmap `CUDA_Host` is then rewritten to plain `CPU`. Only `--no-host`
   lets CPU-layer weights reach `CPU_REPACK`. `-nr/--no-repack` disables repack completely. On this Zen 5 host, repack
   covers Q4_0, Q4_K, IQ4_NL, MXFP4 (AVX2 path) and Q2_K (AVX-512 path) only. Q5_K, Q6_K and Q8_0 have ARM-only repack
   kernels. (VERIFIED in source; the end result is INFERRED, not run)
6. Repacked weights are a real RAM allocation of the same size as the original tensor. They are never `CPU_Mapped`.
   They are **not host buffers**, so llama.cpp's prompt-time op offload to CUDA (batch >= 32) skips them. Repack trades
   prefill speed for CPU decode speed. (VERIFIED in source)
7. The KV cache of a CPU layer is allocated in the plain `CPU` buffer type (`llama_kv_cache: CPU KV buffer size = ...`).
   `-nkvo` moves **all** KV to the CPU and forces attention onto the CPU for every layer, including RPC layers. Do not use
   it with RPC. (VERIFIED)
8. A CPU tier larger than free RAM still loads and runs under mmap. Windows maps the whole GGUF read-only and re-reads
   evicted pages from disk on every token. llama.cpp prints **no warning** in this setup: fit checks host memory only when
   there are no devices at all, and Hypercompute runs `--fit off`. Hypercompute must do its own RAM check. (VERIFIED code
   path; the slowdown is INFERRED)
9. `--lazy-mode` is not a layer-streaming feature. It only applies to `per_layer_tok_embd` in the `gemma4` and `qwen4exp`
   architectures. Disk streaming of whole layers does not exist in llama.cpp, which is the gap AirLLM fills. (VERIFIED)
10. Threads: the default `-t` resolves to **12** on this laptop (physical cores from `GetLogicalProcessorInformationEx`), and
    `-tb` follows `-t`. Real log: `system_info: n_threads = 12 (n_threads_batch = 12) / 24`. `-t 0` or `-t -1` on the CLI
    means **24** (all logical CPUs), not "auto". The build uses OpenMP (`OPENMP = 1`), so `--poll` has no effect. (VERIFIED)
11. Host CPU layers always run on the **host's** CPU. A remote CPU joins only as an RPC device (`ggml-rpc-server -d CPU`).
    It counts as a "GPU" unit for `-ngl` and `-ts`. All three tiers can be combined. A host CPU tier adds **no network
    hop**, because the input embedding was already on the host CPU. (VERIFIED)
12. Any CPU layer (`N <= n_layer`) turns off pipeline parallelism. It is already off for `-np 1` single-stream decode in
    practice, so the main cost is the CPU layers themselves. (VERIFIED condition; impact INFERRED)

---

## 1. `-ngl N` below `n_layer+1`

### 1.1 Source

`src/llama-model.cpp:2418-2421`:

```cpp
uint32_t llama_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}
```

`src/llama-model.cpp:1667-1692` (inside `load_tensors`):

```cpp
const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
    const bool is_swa = il < n_layer_all && hparams.is_swa(il);
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
        return {cpu_dev, &pimpl->cpu_buft_list};
    }
    const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
    ...
};
// there is very little benefit to offloading the input layer, so always keep it on the CPU
pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
...
for (int il = 0; il < n_layer_all; ++il) { pimpl->dev_layer[il] = get_layer_buft_list(il); }
pimpl->dev_output = get_layer_buft_list(n_layer_all);
```

`splits` is built at `1634-1665`: cumulative `tensor_split` (or free memory when every value is 0), normalized to 1.
`cpu_dev` is `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)`, the **local** CPU (`1629-1632`).

The `-ngl` parser (`common/arg.cpp:2802-2818`): `auto` sets -1, `all` sets -2, anything else uses `std::stoi`. Negative values
mean "all units" (2420).

### 1.2 Placement formula

Let `L = n_layer` (`hparams.n_layer_all`), `U = L + 1` units (unit `L` is the output head), and `D` = the model device list
(`--device` order, or RPC-first when `--device` is absent; see `llama_placement.md` §1.3-1.4).

```
N    = clamp(ngl, 0, U)            # 'all'/'auto'(fit off)/negative -> U
k    = U - N                        # repeating layers kept on the host CPU (when N >= 1)
CPU  = token_embd + blk.0 .. blk.(k-1)          (+ output head only when N == 0)
unit u in [0, N) is virtual layer il = k + u    (il == L is the output head)
C_i  = float32(sum_{j<=i} ts_j) / float32(sum_j ts_j)        # cumulative share
dev(u) = D[min{ i : C_i > float32(u) / float32(N) }]         # strict upper_bound
```

- Each device gets one contiguous slice. `D[0]` gets the lowest offloaded layers.
- The output head is unit `N-1`. It goes to the device whose cumulative share first exceeds `(N-1)/N`. With integer
  splits that sum to `N`, that is the **last device with a non-zero split**. With `--device CUDA0,RPC0` (Hypercompute's
  current order, `backend/hypercompute/adapters/llama_rpc.py:122`) the head lands on **RPC0**.
- If every `ts_i` is integer and `sum ts = N`, device `i` gets exactly `ts_i` units. Both sides of the comparison are the
  same single-precision division of the same integers, so there is no rounding drift while `N < 2^24`. (INFERRED from
  IEEE rounding; matches `tests/test_planner_and_leases.py::test_split_counts_reproduce_llama_cpp_assignment`.)
- `-ngl 0` keeps everything, including the output head, on the CPU. `-ngl 1` offloads only the output head.
- `--device none` gives `devices.empty()`, so `act_gpu_layers = 0` and everything is on the CPU.

### 1.3 Worked examples

These come from a local 30-line re-implementation of the formula with float32 rounding
(`scratchpad/calc/place.py`). They are not llama.cpp runs.

| L | -ngl | --device | -ts | CPU | first device | second device |
|---|---|---|---|---|---|---|
| 64 | 55 | CUDA0,RPC0 | 30,25 | tok_embd + blk.0-9 (10) | CUDA0: blk.10-39 (30) | RPC0: blk.40-63 (24) + output |
| 64 | 55 | CUDA0,RPC0 | 1,1 | tok_embd + blk.0-9 | CUDA0: blk.10-37 (28) | RPC0: blk.38-63 (26) + output |
| 32 | 20 | CUDA0,RPC0 | 12,8 | tok_embd + blk.0-12 (13) | CUDA0: blk.13-24 (12) | RPC0: blk.25-31 (7) + output |
| 80 | 41 | CUDA0,RPC0 | 20,21 | tok_embd + blk.0-39 (40) | CUDA0: blk.40-59 (20) | RPC0: blk.60-79 (20) + output |
| 28 | 28 | RPC0,CUDA0 | 20,8 | tok_embd + blk.0 | RPC0: blk.1-20 (20) | CUDA0: blk.21-27 (7) + output |
| 28 | 1 | CUDA0,RPC0 | 1,0 | tok_embd + all 28 layers | CUDA0: output only | RPC0: none |
| 28 | 0 | any | any | everything incl. output | none | none |

Hypercompute rule: pick the per-device unit counts `g` (CUDA0) and `r` (RPC0, which includes the output unit when RPC0 is
last), then pass `-ts g,r -ngl (g+r)`. The CPU then holds `k = L + 1 - g - r` layers.

### 1.4 Load log lines for partial offload (INFO, need `-lv 4`)

`src/llama-model.cpp:1990-2004`:

```cpp
if (llama_supports_gpu_offload()) {
    const int n_gpu = std::min(n_gpu_layers, n_layer_all);
    int n_repeating = n_gpu;
    if (n_repeating > 0) {
        LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
        n_repeating--;
    }
    LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);
    ...
    LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
}
```

- For `L = 64, -ngl 55`: `offloading output layer to GPU`, `offloading 54 repeating layers to GPU`, `offloaded 55/65 layers to GPU`.
  CPU layers = `65 - 55 = 10`. Parse `offloaded (\d+)/(\d+)` and compute `k = M - N`.
- Quirk (INFERRED from the code): at full offload (`N = L+1`), `n_gpu = min(N, L) = L`, so the middle line reports
  `L-1` repeating layers even though all `L` are offloaded. Use only the `offloaded N/M` line for accounting.
- RPC devices count as "GPU" in these lines (`llama_supports_gpu_offload` is true when the RPC reg exists, `src/llama.cpp:101-115`).

### 1.5 Other ways to put weights on the CPU (tensor overrides)

| Flag | Lines (`common/arg.cpp`) | Effect |
|---|---|---|
| `-ot, --override-tensor <regex>=<buft>` | 2757-2762 | per-tensor buffer type. `=CPU` goes through the CPU list (`src/llama-model-loader.cpp:1241-1251`) |
| `-cmoe, --cpu-moe` | 2763-2769 | all `\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps` tensors to CPU (`common/common.h:1247`, `1255-1257`) |
| `-ncmoe, --n-cpu-moe N` | 2770-2779 | expert tensors of the first `N` layers to CPU |
| `-ncffn, --n-cpu-ffn N` | 2790-2800 | dense FFN tensors (`\.ffn_(up\|down\|gate)\.`, `common/common.h:1249`) of the first `N` layers to CPU |

- With mmap on, any `=CPU` override logs this WARN once (`src/llama-model-loader.cpp:1244-1248`), so it is visible at the
  default verbosity: `llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance`.
- Any tensor override disables pipeline parallelism (`!model.has_tensor_overrides()`, `src/llama-context.cpp:516-523`).
- Overrides keep attention and KV on the GPU while only FFN or expert weights sit in RAM. For **MoE** models
  `--n-cpu-moe` is the better CPU tier, because each token reads only the active experts. (INFERRED; not benchmarked)

---

## 2. CPU buffers, repack, mmap, KV

### 2.1 Buffer-type preference list for CPU tensors

`src/llama-model.cpp:1136-1194`, `make_cpu_buft_list(devices, use_extra_bufts, no_host)`, in this order:

1. ACCEL device buffer types, for example BLAS (`1140-1150`). The installed Windows CUDA engine has no `ggml-blas.dll`,
   so this host has none. (VERIFIED by listing `~/.hypercompute/engines/llama-b11770-windows-x64-cuda13.4`)
2. The host buffer type of the **first model device that has one**, unless `no_host` (`1152-1166`). RPC has none
   (`get_host_buffer_type = NULL`, see `llama_placement.md` §1.2). CUDA returns `CUDA_Host`
   (`ggml/src/ggml-cuda/ggml-cuda.cu:5535-5538`). So on this host the list starts with `CUDA_Host` for both
   `--device CUDA0,RPC0` and `RPC0,CUDA0`.
3. CPU "extra" buffer types, if `use_extra_bufts` (`1168-1185`). On x86 these are `AMX` (only when compiled with
   `__AMX_INT8__ && __AVX512VNNI__`, i.e. the sapphirerapids variant) and `CPU_REPACK` (`GGML_USE_CPU_REPACK`, default ON,
   `ggml/CMakeLists.txt:152`); see `ggml/src/ggml-cpu/ggml-cpu.cpp:42-73`. Zen 5 has no AMX, so on this host the list is just `CPU_REPACK`.
4. The plain `CPU` buffer type (`1187-1192`).

Every GPU/RPC device list ends with this CPU list as a fallback (`1621-1627`).

`select_weight_buft` (`src/llama-model-loader.cpp:1068-1079`) takes the **first** entry whose device `supports_op` the
tensor's op with the weight in that buffer type (`weight_buft_supported`, `928-1065`). CUDA's `supports_op` only rejects
CUDA *device* buffers of another GPU (`ggml-cuda.cu:5541-5553`), so `CUDA_Host` wins for any op/type CUDA supports.

Then comes the mmap rule (`src/llama-model-loader.cpp:1270-1278`):

```cpp
// avoid using a host buffer when using mmap
auto * buft_dev = ggml_backend_buft_get_device(buft);
if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    ...
    buft = ggml_backend_dev_buffer_type(cpu_dev);
}
```

`CUDA_Host`'s device is CUDA0 (`ggml-cuda.cu:1388`), so under mmap it becomes plain `CPU`. In `load_tensors`
(`src/llama-model.cpp:1930-1948`), a context whose buffer type is the CPU **default** type is wrapped around the mapped
file ranges with `buffer_from_host_ptr`, which gives buffers named `CPU_Mapped` (`ggml/src/ggml-backend.cpp:2603-2626`).

### 2.2 Resulting CPU-layer placement by mode (this host, CUDA0 in the device list)

| Launch flags | CPU-layer weights land in | Private RAM? | Prompt op offload to CUDA0 | Status |
|---|---|---|---|---|
| default (`-lm auto` = mmap) | `CPU_Mapped` (file-backed) | no (page cache) | yes | VERIFIED in source; `CPU_Mapped` lines seen in real logs |
| `--no-host` (mmap) | `CPU_REPACK` for repackable tensors, `CPU_Mapped` for the rest | repacked part yes | not for repacked tensors | VERIFIED in source |
| `-lm none` | `CUDA_Host` (pinned, chunks of at most 4 GiB; falls back to pageable `CPU` if pinning fails) | yes (pinned) | yes, from pinned memory | VERIFIED in source |
| `-lm none --no-host` | `CPU_REPACK` + `CPU` | yes | not for repacked tensors | VERIFIED in source |
| `-nr` (any) | as above, but `CPU_REPACK` is never used | | | VERIFIED |
| no CUDA in the device list (e.g. `--device RPC0`) | `CPU_REPACK` for repackable tensors, `CPU_Mapped` for the rest | repacked part yes | none (RPC has no `offload_op`) | VERIFIED in source |

`token_embd` uses GET_ROWS, which repack does not support. It always ends up in `CPU_Mapped` (mmap) or `CUDA_Host` (`-lm none`).

### 2.3 `CPU_REPACK` details

- Name `CPU_REPACK` (`ggml/src/ggml-cpu/repack.cpp:5162-5166`). `alloc_buffer` allocates an ordinary CPU buffer of the same
  byte size (`get_alloc_size = nullptr` defaults to `ggml_nbytes`; `5168-5180`, `5240-5247`). `set_tensor` repacks while copying
  (`5150-5160`). `is_host = nullptr`, so it is **not** a host buffer (`ggml-backend.cpp:221-227`).
- It is a real allocation, never mapped: `load_tensors` uses `buffer_from_host_ptr` only for the device's default buffer
  type (`is_default_buft`, `src/llama-model.cpp:1922-1930`). Under mmap the loader reads the source bytes through the mapping
  and calls `ggml_backend_tensor_set` (`src/llama-model-loader.cpp:1699-1735`).
- Memory duplication: the repacked tensor is private RAM equal to the original size. Its source pages stay in the OS file
  cache as clean, reclaimable pages. On Windows, `unmap_fragment` is a no-op (`src/llama-mmap.cpp:638-641`), so the whole
  file view stays mapped. That is address space plus evictable cache, not a second committed copy. Expect up to about 2x
  transient RAM use for repacked tensors during load, falling back to 1x under memory pressure. (INFERRED from Windows file-mapping semantics)
- Repack types (`ggml_repack_get_optimal_repack_type`, `repack.cpp:4925-5140`). These are runtime checks inside the loaded
  CPU variant DLL. Each also needs `ne[1] % 8 == 0` (x86):

  | Type | x86 condition | Line |
  |---|---|---|
  | Q4_0 | AVX2 | 4975-4979 |
  | Q4_K | AVX2 | 5007-5011 |
  | IQ4_NL | AVX2 | 5073-5077 |
  | MXFP4 | AVX2 | 5095-5099 |
  | Q2_K | AVX-512 | 5034-5038 |
  | Q5_K, Q6_K, Q8_0, Q1_0 | none on x86 (NEON/RVV only) | 5050-5071, 5105-5139 |

  A typical Q4_K_M GGUF mixes Q4_K (repackable) and Q6_K (not repackable), so `--no-host` gives a mix of `CPU_REPACK` and `CPU_Mapped` lines. (INFERRED)
- Supported ops: only 2-D `MUL_MAT` and 3-D `MUL_MAT_ID` with an F32, host-resident `src1` (`repack.cpp:5191-5228`).
- The CPU variant loaded on this laptop is **zen4**. INFERRED by elimination from the real `system_info` line, which shows
  `AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1` and no `AMX_INT8`. Among the installed variants only `zen4` has
  VBMI + BF16 without AMX (`ggml/src/CMakeLists.txt:504-520`). The same line shows `REPACK = 1`.
- Flags: `--repack` / `-nr, --no-repack`, env `LLAMA_ARG_REPACK`, default enabled (`common/arg.cpp:2425-2432`).
  `--no-host` (env `LLAMA_ARG_NO_HOST`, help "bypass host buffer allowing extra buffers to be used", `2433-2439`). They map to
  `mparams.use_extra_bufts = !no_extra_bufts` and `mparams.no_host` (`common/common.cpp:1686-1687`).

### 2.4 Prompt-time op offload of CPU-layer weights

`ggml/src/ggml-backend.cpp:1116-1124` (scheduler):

```cpp
if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {
        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
            SET_CAUSE(tensor, "1.off");
            return b;
```

- CUDA offloads when the op batch is at least `GGML_OP_OFFLOAD_MIN_BATCH` (default 32; `ggml-cuda.cu:180-183`, `6018-6037`).
  RPC's `offload_op` is `NULL` (`ggml-rpc.cpp:2934`), so ops are never offloaded to the Mac.
- Result: during prefill (ubatch of 32 or more tokens), matmuls of `CPU_Mapped`/`CUDA_Host` CPU-layer weights run on CUDA0,
  and the weights cross PCIe for each ubatch. Decode (1 token) runs on the CPU. `CPU_REPACK` weights are not host buffers, so
  their prefill also runs on the CPU. `--no-op-offload` (`common/arg.cpp:2960-2966`) turns this off.
- KV tensors are not WEIGHTS-usage buffers, so attention for CPU layers stays on the CPU during prefill too. (INFERRED from the
  same function)

### 2.5 `--no-mmap`, `--mlock`, `-lm`, `-lzm`

- At b11770 `common/arg.cpp` has no `--no-mmap`, `--mmap` or `--mlock` option (grep for `mmap|mlock` matches only the `-lm`
  help, 2697-2708). An unknown flag throws `error: invalid argument: <arg>` (829-831), and llama-server returns 1
  (`llama_placement.md` addendum P3). The server README at the same ref lists only `-lm` (line 75).
- `-lm, --load-mode MODE` (2694-2711, env `LLAMA_ARG_LOAD_MODE`): `auto` (default), `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`.
  The loader sets `use_mmap` for `mmap`, `mmap+mlock` and `auto` (`src/llama-model-loader.cpp:560`). `auto` falls back to
  no-mmap only if some model device lacks `mmap_support` (`src/llama-model.cpp:1590-1599`). CUDA reports
  `mmap_support = (type != IGPU)` (`ggml-cuda.cu:5520`), RPC reports true, so this host uses mmap. The load log prints the
  resolved mode: `load_tensors: loading model tensors, this can take a while... (load_mode = mmap)` (1613-1618).
- `mlock` and `mmap+mlock` (`use_mlock`, 1580): on Windows, `VirtualLock`, growing the working set with
  `SetProcessWorkingSetSize` and retrying once (`src/llama-mmap.cpp:749-774`). On failure it only warns:
  `warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s` and stops trying (`grow_to`,
  802-815). Under `mmap+mlock` the lock grows from the **start of the file** to the end of the last `CPU_Mapped` tensor
  (`src/llama-model-loader.cpp:1725-1729`), so any GPU tensor stored earlier in the file is locked as well. (INFERRED; tensor
  order inside the GGUF not checked)
- `-lzm, --lazy-mode {on,auto,off}` (2712-2724): see §3.3.

### 2.6 KV cache and compute buffer for CPU layers

`src/llama-kv-cache.cpp:212-222`:

```cpp
const char * dev_name = "CPU";
ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
if (offload) {
    auto * dev = model.dev_layer(il);
    buft = ggml_backend_dev_buffer_type(dev);
    dev_name = ggml_backend_dev_name(dev);
}
LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);
```

- With the default `offload = offload_kqv = true`, a CPU layer's `dev_layer(il)` is the CPU device, so its K/V go in the
  plain `CPU` buffer type. That is pageable private RAM, not mmap and not pinned. Per-layer size is as in
  `llama_placement.md` §5 (about `2 * n_ctx * n_head_kv * (head_dim_k + head_dim_v)` bytes at f16).
- `-nkvo, --no-kv-offload` (`common/arg.cpp:2417-2424`) sets `offload = false` for **every** layer, so all KV, including
  RPC layers' KV, lives in host RAM. It also pins the attention block to the CPU backend
  (`src/llama-graph.cpp:3026-3029`) and disables pipeline parallelism (`llama-context.cpp:521`). With RPC that means
  host-to-peer traffic in every layer (INFERRED). Do not use it in Hypercompute.
- Compute buffers: one per backend, in order `model.devices`, ACCEL, CPU (`src/llama-context.cpp:419-446`). The CPU backend
  uses the host buffer type of `model.devices[0]` when it has one (`495-506`; `no_host` is not consulted). So with
  `--device CUDA0,RPC0` the CPU compute buffer is `CUDA_Host`, and with RPC first it is `CPU`. Real logs confirm
  `sched_reserve:  CUDA_Host compute buffer size = 33.57 MiB` with `--device CUDA0,RPC0`. CPU layers make this buffer larger. (VERIFIED)

### 2.7 Log lines that reveal the CPU tier

The format strings are verbatim. `%12s` / `%10s` right-align the buffer name. At `-lv 4` (INFO maps to TRACE, see
`llama_placement.md` §6):

| Line | Format (file:line) |
|---|---|
| load mode | `"%s: loading model tensors, this can take a while... (load_mode = %s)\n"` (`llama-model.cpp:1617`) |
| offload counts | `offloading output layer to GPU`, `offloading %d repeating layers to GPU`, `offloaded %d/%d layers to GPU` (`1995`, `1998`, `2003`) |
| CPU weights | `"%s: %12s model buffer size = %8.2f MiB\n"` (`2009`) with name `CPU_Mapped`, `CPU_REPACK`, `CUDA_Host` or `CPU` |
| CPU KV | `"%s: %10s KV buffer size = %8.2f MiB\n"` (`llama-kv-cache.cpp:291`) with name `CPU` |
| KV total | `"%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): ..."` (`301`). `layers` counts all KV layers, not only CPU ones |
| CPU compute | `"%s: %10s compute buffer size = %8.2f MiB\n"` (`llama-context.cpp:842`) with name `CUDA_Host` or `CPU` |
| threads | `system_info: n_threads = %d (n_threads_batch = %d) / %d \| ... CPU : ... REPACK = 1 \|` (`common/common.cpp:421-437`, TRACE) |

Expected shape for `L = 64, -ngl 55, --device CUDA0,RPC0`, mmap (numbers elided):

```
load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
load_tensors: offloading output layer to GPU
load_tensors: offloading 54 repeating layers to GPU
load_tensors: offloaded 55/65 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   NNNN.NN MiB
load_tensors:        CUDA0 model buffer size =   NNNN.NN MiB
load_tensors: RPC0[<ip>:<port>] model buffer size =   NNNN.NN MiB
llama_kv_cache:        CPU KV buffer size =     NN.NN MiB
llama_kv_cache:      CUDA0 KV buffer size =     NN.NN MiB
llama_kv_cache: RPC0[<ip>:<port>] KV buffer size =     NN.NN MiB
sched_reserve:      CUDA0 compute buffer size =    NN.NN MiB
sched_reserve: RPC0[<ip>:<port>] compute buffer size =    NN.NN MiB
sched_reserve:  CUDA_Host compute buffer size =    NN.NN MiB
```

- Model buffer lines follow `ctx_map` order, which is sorted by buffer-type **name** with non-lazy contexts first
  (`src/llama-model-loader.h:149-163`): `CPU` < `CPU_REPACK` < `CUDA0` < `CUDA_Host` < `RPC0[...]`. KV lines are sorted the
  same way (`llama-kv-cache.cpp:104-110`).
- A `CPU_Mapped` context can print **several** lines, one per disjoint file range (`get_mapping_ranges`,
  `llama-model-loader.cpp:1491-1525`). Sum the lines by name, as `llama_placement.md` §2 already says.
- `CUDA_Host` model buffers larger than 4 GiB are allocated in chunks (`get_max_size`, `ggml-cuda.cu:1370-1374`) and printed
  as one multi-buffer line, named after the first chunk's type (`ggml-backend.cpp:858-872`). A failed pin is logged at DEBUG only
  (`failed to allocate %.2f MiB of pinned memory`, `ggml-cuda.cu:1347`).
- At `-lv 5`: `load_tensors: layer %3d assigned to device CPU, is_swa = %d` (1672), `llama_kv_cache: layer %3d: dev = CPU`
  (`llama-kv-cache.cpp:222`), and, under mmap with CUDA present,
  `done_getting_tensors: tensor '<name>' (<type>) (and N others) cannot be used with preferred buffer type CUDA_Host, using CPU instead`
  (`llama-model-loader.cpp:1404-1408`).

Parse rule for Hypercompute: CPU-tier weight bytes = the sum of `model buffer size` lines named `CPU_Mapped`, `CPU_REPACK`,
`CPU` or `CUDA_Host`. That sum includes `token_embd` (real logs show 89.26 MiB for the 0.5B test model and 292.36 MiB for the 7B one).
CPU KV = the `CPU KV buffer size` line.

---

## 3. Model or CPU tier larger than free RAM

### 3.1 What mmap does on Windows (VERIFIED code, INFERRED OS behavior)

- `init_mappings(true, ...)` is always called with `prefetch = true` (`src/llama-model.cpp:1890`), and
  `prefetch_size = prefetch && use_mmap ? -1 : 0` (`src/llama-model-loader.cpp:1430`), i.e. the whole file.
- Windows `llama_mmap::impl` (`src/llama-mmap.cpp:586-656`) maps the whole file read-only:
  `CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL)` (596) and `MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0)` (603).
  It then calls `PrefetchVirtualMemory` over the whole file (611-635; failure only warns
  `warning: PrefetchVirtualMemory failed: %s`). `unmap_fragment` does nothing (638-641), so ranges used by GPU/RPC tensors
  are never released on Windows. They are only clean, evictable file pages.
- A read-only file view of a 64-bit file does not count against commit, so a GGUF larger than RAM maps fine. CPU-layer
  weights are read from the mapping on every token. When the CPU-tier working set is larger than available RAM, Windows evicts
  pages and faults them back from disk each token. It works, but decode becomes disk-bound (roughly missing bytes per token ÷
  SSD read speed). GPU and RPC tensors are copied once at load and need no RAM afterwards. (INFERRED; not run)
- The Linux path differs: `MAP_POPULATE` when prefetching (`src/llama-mmap.cpp:493`) and real `munmap` of unused prefix and
  suffix ranges (`src/llama-model-loader.cpp:1841-1852`). It does not apply to this host.

### 3.2 Checks and warnings in llama.cpp: effectively none

- `--fit` compares projected host use against host memory **only when the model has no devices** (`nd == 0`,
  `common/fit.cpp:317-329`, TRACE level). With GPU/RPC devices it only budgets device memory. Hypercompute runs `--fit off`
  anyway (`llama_placement.md` §4).
- The CPU device reports Windows `ullAvailPhys` as free memory (`ggml/src/ggml-cpu/ggml-cpu.cpp:365-371`). On Linux/macOS it
  reports free = total (372-378). llama.cpp itself never compares this with the CPU-tier size.
- No warning text exists for "CPU weights larger than RAM" in `src/` or `common/` at this ref (grep for RAM/swap/host-memory
  wording found only the fit TRACE lines and a lazy-read message).
- `-lm none`: CPU weights go to pinned `CUDA_Host`, or to pageable `CPU` if pinning fails. Both count against commit. If commit
  is exhausted, allocation fails with `unable to allocate %s buffer` (`src/llama-model.cpp:1946`, `1962`) and the load aborts.
  `-lm none` is therefore worse than mmap for an oversized tier.
- `-lm mlock` / `mmap+mlock`: lock failures only warn (§2.5). Loading continues.

### 3.3 `--lazy-mode` and `--load-mode` relevance

- `-lzm` affects only tensors created with `TENSOR_READ_LAZY` (`src/llama-model-loader.cpp:1343-1346`). At this ref those are
  only `per_layer_tok_embd` in `src/models/gemma4.cpp:59` and `src/models/qwen4exp.cpp:213-214`. `auto` turns it on only for
  tensors over 4 GiB (`1093-1097`). Lazy tensors read rows from the mapping on demand (`1081-1115`).
  **It does not stream ordinary layer weights.** Layer-by-layer disk streaming is not available in llama.cpp.
- `-lm` matters as described above. Keep the default (mmap). `dio` (direct I/O) only changes how a non-mmap load reads the file.

### 3.4 Hypercompute RAM budget rule (recommendation)

Before launch, require:

```
cpu_weights(blk.0..k-1) + token_embd + kv_per_layer * k + cpu_compute_buffer + output_buffer
    <= ullAvailPhys (or psutil available) - margin
```

`cpu_weights` comes from GGUF tensor sizes, which the planner already reads. Use a margin of a few GiB on the 32 GB laptop. If
the check fails, do not rely on mmap paging for interactive use. Hand the model to the AirLLM or EXO path instead, or shrink
`k` by putting more units on GPU/RPC. After launch, confirm with the `CPU_Mapped`/`CPU KV` lines from §2.7.

---

## 4. Threads for CPU layers

### 4.1 Defaults (VERIFIED)

- `common_cpu_params::n_threads = -1` (`common/common.h:72`). After parsing, `postprocess_cpu_params(params.cpuparams, nullptr)`
  replaces -1 with `common_cpu_get_num_math()`, and the batch params copy the generation params when unset
  (`common/arg.cpp:886-890`; `common/common.cpp:291-313`).
- On Windows, `common_cpu_get_num_math()` falls through to `common_cpu_get_num_physical_cores()` (`common.cpp:205-231`). That
  counts `RelationProcessorCore` entries from `GetLogicalProcessorInformationEx` (`120-145`), i.e. physical cores. The
  hybrid-core probing (Intel `cpuid 0x1a`, skipping E-cores) runs only on Linux x86 (`152-201`).
- Real log on this laptop: `system_info: n_threads = 12 (n_threads_batch = 12) / 24 | ... CPU : ... AVX512 = 1 | ... OPENMP = 1 | REPACK = 1 |`.
  So the default is 12 for decode and for batches.
- Trap: the `-t`/`-tb` handlers map any value `<= 0` to `std::thread::hardware_concurrency()` (`common/arg.cpp:1519-1538`), so
  `-t -1` gives **24**, not the 12-thread default. The help text still says "(default: -1)". Hypercompute should pass an
  explicit positive number or omit the flag.
- Which count is used: `graph_compute(gf, ubatch.n_tokens > 1)` (`src/llama-context.cpp:1570`) uses `n_threads_batch` for
  batches of more than one token and `n_threads` for single-token decode (`2731-2748`).
- The build has `OPENMP = 1`. Under OpenMP, `ggml_graph_compute` runs `#pragma omp parallel num_threads(n_threads)` and applies
  `--prio` and the `-C/--cpu-mask` affinity per thread (`ggml/src/ggml-cpu/ggml-cpu.c:3419-3439`). The `--poll`
  busy-wait code exists only in the non-OpenMP build (`#ifndef GGML_USE_OPENMP`, 3176-3312), so `--poll` has no effect here. (INFERRED from the `#ifdef` layout)
- Remote `ggml-rpc-server` CPU workers use their own `-t` (default `hardware_concurrency()/2`, `tools/rpc/rpc-server.cpp:98`;
  already in `llama_rpc.md`).

### 4.2 Recommendation for the Ryzen AI 9 HX 370 (UNVERIFIED performance)

The HX 370 has 12 cores (4 Zen 5 + 8 Zen 5c) and 24 threads. That core mix comes from the vendor spec and was not checked
here. Decode through CPU layers is memory-bandwidth-bound. SMT siblings do not help, and the 12-core default already
avoids them. Start with:

- `-t 8 -tb 12`. Fewer decode threads leave cores for the HTTP thread, the RPC client socket work and the CUDA driver, and
  bandwidth usually saturates before all cores are busy. `-tb 12` matters mostly with `--no-host`, because otherwise large
  CPU-layer prefill matmuls are offloaded to CUDA0 (§2.4).
- Benchmark `-t` in {6, 8, 10, 12} on the real model and placement, keep the best tokens/s, and store it per host profile. Do
  not go above 12.
- Optional: `--prio 2` (high) to protect the CPU stage from desktop load. Its effect on a laptop is UNVERIFIED.

---

## 5. CPU layers and RPC devices

- Host CPU layers always use the **local** CPU device (`cpu_dev = ggml_backend_dev_by_type(CPU)`, `src/llama-model.cpp:1629-1632`).
  The CPU cannot appear in `--device`: `parse_device_list` throws `invalid device: CPU` for CPU-type devices
  (`common/arg.cpp:1122-1142`). The only ways to put weights in host RAM are `-ngl` below the unit count and the overrides in §1.5.
- A remote CPU can hold layers only as an RPC device. `ggml-rpc-server -d CPU` exposes it as `RPC<N>` with type hard-coded
  to GPU (`llama_placement.md` §1.2). It is a normal entry of `model->devices`. It takes `-ts` units and counts toward `-ngl`.
  This is what Hypercompute already does with `-d CPU` workers.
- Remote CPU weights are allocated with the server backend's **default** buffer type (`ggml-rpc.cpp:1502-1508`), i.e. plain
  `CPU` RAM on the peer. There is no mmap and no repack there, and the weights cross the network at load (or come from
  `-c` cache files).
- All tiers combine. Order per token with `--device CUDA0,RPC0` and `k > 0`:
  `host CPU (token_embd, blk.0..k-1)` → PCIe → `CUDA0` → network → `RPC0 (+ output head)` → network (logits) → host.
  Before the CPU tier the path was the same minus the CPU stage, so the **CPU tier adds no network crossing**. The input
  embedding was already on the host CPU (`dev_input`, 1683). (VERIFIED placement; traffic INFERRED)
- Pipeline parallelism needs `model.n_gpu_layers() > n_layer_all` (`src/llama-context.cpp:516-523`), so any CPU layer turns it
  off. It mainly helps multi-sequence or batched prefill across devices. With `-np 1`, losing it costs little. (INFERRED)
- Op offload never targets RPC (§2.4). Prefill of CPU-layer weights uses local CUDA0 only when CUDA0 is in the device list.
- Output head on RPC (INFERRED, worth measuring): logits are read from the backend that computed them,
  `ggml_backend_tensor_get_async(backend_res, t_logits, ..., n_outputs*n_vocab*sizeof(float))` (`src/llama-context.cpp:2049-2060`).
  That happens unless backend sampling makes raw logits unnecessary, and whether llama-server enables that by default is
  UNVERIFIED. With the head on RPC0, each generated token pulls `n_vocab * 4` bytes over the network (about 0.6 MB for a
  152k vocabulary). With `--device RPC0,CUDA0` the head is local, and the network carries two hidden-state vectors
  (`n_embd * 4` bytes each). For tokens/s on Wi-Fi, try `--device RPC0,CUDA0` (`-ts r,g`). The CPU tier is unaffected because it always comes first.

---

## 6. Recommended llama-server recipe: CUDA0 + RPC peer + host CPU layers

### 6.1 Dense model (default)

```
llama-server -m <model.gguf> --alias <alias> \
  --rpc <peer-ip>:<port> \
  --device CUDA0,RPC0 --split-mode layer \
  --tensor-split <g>,<r> -ngl <g+r> \
  --fit off -c <ctx> -np 1 \
  -t 8 -tb 12 \
  --host 127.0.0.1 --port <port> --metrics --no-ui -lv 4 --log-colors off
```

- `g` = CUDA0 units, `r` = RPC0 units including the output head, and `k = n_layer + 1 - g - r` layers on the host CPU.
  Keep `--rpc` before `--device` (`llama_placement.md` §1.4).
- Leave `-lm` at its default (`auto` = mmap). Never pass `--no-mmap` or `--mlock`. They no longer exist and abort startup.
- Fill order the planner should try (INFERRED, confirm by measurement): VRAM on CUDA0, then the peer, then host RAM. Moving
  layers to the peer adds no network hop. Host RAM decode costs about `bytes(CPU layers) / effective RAM bandwidth` per token.
  Use AirLLM only when `k` would exceed the RAM budget in §3.4.
- `-lv 4` is already in `build_args`. Read `offloaded N/M`, `CPU_Mapped ... model buffer size` and `CPU KV buffer size` to
  confirm placement (§2.7).

### 6.2 Variants

| Goal | Change | Trade-off |
|---|---|---|
| Faster CPU decode with Q4_0 / Q4_K / IQ4_NL / MXFP4 weights | add `--no-host` | Repacked weights take private RAM, not mmap. Their prefill stays on the CPU (no op offload). Benchmark both ways |
| MoE model (experts do not fit in VRAM) | `-ngl all -ts g,r --n-cpu-moe K` instead of a lower `-ngl` | Attention and KV stay on GPUs. Prints the mmap override WARN. Disables pipeline parallelism. `-lm none` would pin up to the full expert size in `CUDA_Host`, so stay on mmap on a 32 GB laptop |
| Dense model, keep attention on GPU | `--n-cpu-ffn K` | Same trade-offs as `--n-cpu-moe`. Not benchmarked |
| Lower per-token network bytes | `--device RPC0,CUDA0 -ts r,g` | Head and logits stay local (§5). Needs a planner change (output unit moves to CUDA0) |
| Prefill speed for CPU layers | keep the default (host buffers) and `-ub 512` | CPU-layer weights cross PCIe per ubatch. `GGML_OP_OFFLOAD_MIN_BATCH` (env, default 32) sets the threshold |

### 6.3 Do not

- `--no-mmap`, `--mlock`: removed (§2.5). Use `-lm none` / `-lm mlock` only on purpose.
- `-nkvo` with RPC (§2.6).
- `--fit on` together with an explicit `-ngl`/`-ts`: fit aborts and keeps the user's values (`llama_placement.md` §4). Keep `--fit off`.
- `-t 0` or `-t -1`: means 24 threads (§4.1).
- `-lm none` for a CPU tier near the RAM limit: it commits (and pins) the whole tier and can fail to allocate (§3.2).

---

## 7. UNVERIFIED / open items

| Item | Why not confirmed | How to close |
|---|---|---|
| Real tokens/s for CPU tier sizes and `-t` values on the HX 370 | nothing was run | benchmark `-t` 6/8/10/12, `--no-host` on/off, `k` = 0/4/8/16 |
| That repack is skipped by default with CUDA0 present | derived from buffer-type order + the mmap rule; the existing logs only show `token_embd`, which repack never takes | run once with `-lv 5` and look for `cannot be used with preferred buffer type CUDA_Host, using CPU instead` and the absence of `CPU_REPACK` lines |
| Windows paging behavior when the tier exceeds RAM | OS behavior, not llama.cpp source | controlled test with a tier about 1.2x free RAM |
| Whether llama-server reads full logits per token (backend sampling) | `needs_raw_logits` / sampler wiring in the server not traced | check `tools/server` sampling setup, or measure RPC traffic per token |
| Tensor order inside typical GGUFs (number of `CPU_Mapped` lines, `mmap+mlock` lock span) | depends on the converter | read a GGUF's tensor offsets with Hypercompute's own GGUF reader |
| HX 370 core layout and memory bandwidth | vendor spec, not in source | not needed for code. Calibrate from the benchmark above |
| `--prio` effect under OpenMP (vcomp/libomp) on Windows | runtime-specific | benchmark |

---

## Commands used

```
gh api -H "Accept: application/vnd.github.raw" "repos/ggml-org/llama.cpp/contents/<path>?ref=39da130fcd1b6ca47d3c0522a40eccff41ea3dfb"
  paths: src/llama-model.cpp, src/llama-model-loader.cpp, src/llama-model-loader.h, src/llama-mmap.cpp, src/llama.cpp,
         src/llama-context.cpp, src/llama-kv-cache.cpp, src/llama-graph.cpp, src/models/*.cpp, include/llama.h,
         common/arg.cpp, common/common.cpp, common/common.h, common/fit.cpp, tools/server/server.cpp, tools/server/README.md,
         tools/rpc/rpc-server.cpp, ggml/CMakeLists.txt, ggml/src/CMakeLists.txt, ggml/src/ggml-backend.cpp, ggml/src/ggml-alloc.c,
         ggml/src/ggml-cpu/ggml-cpu.cpp, ggml/src/ggml-cpu/ggml-cpu.c, ggml/src/ggml-cpu/repack.cpp,
         ggml/src/ggml-cpu/CMakeLists.txt, ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-rpc/ggml-rpc.cpp
gh api "repos/ggml-org/llama.cpp/git/trees/39da130...?recursive=1"     (list src/models)
gh api "repos/ggml-org/llama.cpp/releases/tags/b11770" --jq '.assets[].name'
local, read-only: ls ~/.hypercompute/engines/llama-b11770-windows-x64-cuda13.4 ; grep of results/*.json for existing log lines
local formula check: scratchpad/calc/place.py (own 30-line re-implementation, run with the system Python)
```

---

## Addendum (critic)

Completeness review for implementing the host-RAM tier with correct placement prediction. Two gaps were resolved by
reading source at the same pin (`39da130`, via `gh api ... ?ref=39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`); nothing was
built or run. Remaining gaps are listed at the end.

### A1. What a "unit" is: `block_count`, NextN/MTP blocks, and the output unit's bytes (VERIFIED)

§1.2 writes `L = n_layer (hparams.n_layer_all)` but does not say where that number comes from or what it contains. A
planner that gets this wrong mis-predicts every `-ts` split by one or more units.

- `n_layer_all` is GGUF `<arch>.block_count`, read verbatim: `ml.get_key(LLM_KV_BLOCK_COUNT, hparams.n_layer_all);`
  (`src/llama-model.cpp:1394-1395`; key string `"%s.block_count"`, `src/llama-arch.cpp:208`).
- It **includes** NextN/MTP blocks. `n_layer_nextn` comes from `<arch>.nextn_predict_layers` (`llama-model.cpp:1396-1397`,
  `llama-arch.cpp:235`) and the trunk is `n_layer() = n_layer_all - n_layer_nextn` (`src/llama-hparams.cpp:348-350`).
  Placement (`1667-1692`), `n_gpu_layers()` (`2420`) and `offloaded %d/%d` (`max_backend_supported_layers = n_layer_all + 1`,
  `2000-2003`) all use `n_layer_all`. So **U = block_count + 1** for every model, MTP or not.
- NextN blocks are the highest-numbered `blk.*`. Example loader: `const int flags = i < n_layer ? trunk_flags : mtp_flags;`
  (`src/models/glm4-moe.cpp:51`).
- llama-server does **not** load them by default. `nextn_flags()` adds `TENSOR_SKIP` when `!ml.load_mtp`
  (`llama-model.cpp:3884-3900`). `mparams.load_mtp` is true only for the imatrix `--nextn` option or a draft-MTP speculative
  type (`common/common.cpp:1706`; default `bool load_mtp = false;`, `common/common.h:591`). Skipped tensors print
  `model has unused tensor %s (size = %zu bytes) -- ignoring` at WARN and are subtracted from `size_data`
  (`src/llama-model-loader.cpp:1178-1185`).
- They have no KV in the main context: `filter = [&](uint32_t il) { return il < hparams.n_layer(); };` when
  `n_layer_nextn > 0 && n_layer() > 0 && router_layer < 0` (`llama-model.cpp:3233-3238`). SWA is forced off for them
  (`llama-hparams.cpp:19-21`).
- Planner rule: keep `U = block_count + 1`, but cost the trailing `nextn_predict_layers` units at **0 weight bytes and 0 KV**
  unless Hypercompute turns on MTP speculation. They are the first units counted by `-ngl` (just before the output unit), so
  they always land on the last device in `--device` order, and integer `-ts` values must include them. Example (formula of §1.2,
  INFERRED, not run): `block_count = 47` with 1 NextN block, `--device CUDA0,RPC0 -ts 20,13 -ngl 33` gives CPU `blk.0-14`,
  CUDA0 `blk.15-34`, RPC0 `blk.35-45` (11 real layers) + `blk.46` (NextN, empty) + output.
- Output unit bytes: when `output.weight` is absent (tied embeddings, e.g. Qwen2.5 <= 3B, Llama-3.2-1B/3B), loaders create
  the head from the embedding: `output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);`
  (`src/models/qwen2.cpp:26-30`). The loader treats that tensor as an OUTPUT-layer tensor (`llama-model-loader.cpp:1163-1169`),
  allocates a **separate copy** in the output device's buffer and adds its bytes to `size_data` (`1371-1387`); it reuses the
  existing tensor only when the same buffer-type context already holds it. So with any offload the output unit costs
  `bytes(output.weight)` if present, else `bytes(token_embd.weight)`, on the tail device, while `token_embd` also stays in
  host RAM (§0 item 1). Reuse in the `-ngl 0` case is INFERRED (same CPU context), not traced.

### A2. Does llama-server read full logits every token? Yes, by default (VERIFIED; closes §7 row 4)

- `bool backend_sampling = false;` in `common_params_sampling` (`common/common.h:299`). Only `-bs, --backend-sampling`
  turns it on, server-wide ("enable backend sampling (experimental) (default: disabled)", env `LLAMA_ARG_BACKEND_SAMPLING`,
  `common/arg.cpp:2310-2315`). Requests cannot turn it on: `tools/server/server-task.cpp` only serializes it (lines 84, 143).
- Even with `-bs`, the server attaches the sampler only if the request needs no pre-sampling probs
  (`need_pre_sample_logits = n_probs > 0 && !post_sampling_probs`, `tools/server/server-context.cpp:1933-1945`).
  `common_sampler_init` disables it for any grammar (JSON schema, tool-call grammars) and for a reasoning budget
  (`common/sampling.cpp:422-432`). `set_sampler` refuses under `--split-mode tensor` (`src/llama-context.cpp:1398-1408`).
- Without a backend sampler, `needs_raw_logits()` returns true (`llama-context.cpp:1804-1819`) and every decode copies
  `n_outputs * n_vocab * sizeof(float)` from the backend that computed the logits (`ggml_backend_tensor_get_async(backend_res,
  t_logits, ...)`, `2049-2060`). Single-stream decode has `n_outputs = 1`.
- Consequence for §5 and §6.2: with Hypercompute's current `--device CUDA0,RPC0` the head is on RPC0, so every generated
  token pulls `n_vocab * 4` bytes over the network: about 594 KiB for Qwen2.5 (vocab 152,064) and 501 KiB for Llama-3
  (128,256), roughly 40-50 ms per token at an effective 100 Mbit/s (INFERRED arithmetic). Make `--device RPC0,CUDA0 -ts r,g`
  (head local) the default when an RPC peer is used. Do not rely on `-bs`: it is experimental, is dropped for grammars and
  tool calls, and with the head on RPC0 its sampler buffers would live on the RPC device (`ggml_backend_dev_buffer_type(model.dev_output())`,
  `llama-context.cpp:1418`), whose support for the sampling ops is UNVERIFIED.

### A3. Open questions found by this review and not resolved here

| Gap | Why it matters | Status |
|---|---|---|
| CPU compute-buffer size as a function of `k`, `-ub`, `n_ctx` | the §3.4 RAM check uses it but no formula was traced; only the post-launch log line exists | UNVERIFIED |
| Non-`blk` tensors other than `token_embd`/`output*` (e.g. Gemma `per_layer_tok_embd`, vision/projector tensors) | they are neither a layer unit nor the output unit; which device gets them decides host-RAM bytes | UNVERIFIED (planner should sum GGUF bytes by `blk.<i>.` prefix and flag any other tensor) |
| Repack skipped with CUDA0 present (§7 row 2) | changes CPU decode speed for Q4_0/Q4_K weights | still needs one `-lv 5` run |
| Converted GGUFs that keep `nextn_predict_layers > 0` but drop the NextN tensors | `nextn_flags` treats them as trunk-only (`llama-model.cpp:3888`); unit count is still `block_count + 1` | INFERRED from the comment, not traced per converter |
