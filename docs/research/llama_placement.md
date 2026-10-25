# llama.cpp model loading and device placement audit (b11770)

Pinned source: ggml-org/llama.cpp tag `b11770` = commit `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`
(verified with `gh api repos/ggml-org/llama.cpp/git/ref/tags/b11770`, object type `commit`).

All files were read with `gh api -H "Accept: application/vnd.github.raw" "repos/ggml-org/llama.cpp/contents/<path>?ref=39da130..."`.
No external code was executed or built. Line numbers below are for the pinned commit.

Relation to the handover audit commit `9a3b91c6801e68f412fcf1b6f8e153aa562b8536`: `gh api .../compare/39da130...9a3b91c`
reports `ahead_by=12 behind_by=0`. No file under `ggml/src/ggml-rpc/` changed. `src/llama-model.cpp` and `common/common.cpp`
did change, but only in hunks at `@@ -1549` (load_hparams) and `@@ -3438` (cls label) of llama-model.cpp and decision-type
tables in common.cpp. None of the placement code below is touched. In the later commit, llama-model.cpp line numbers after
1549 shift by +3.

---

## 0. TL;DR for Hypercompute

1. With `--rpc` and no `--device`, the model device list is **all RPC devices first, in `--rpc` order, then local discrete GPUs
   (CUDA0, MTL0, ...)**. `-ts` index `i` is position `i` in that list. Integrated GPUs are appended only when no discrete GPU exists.
2. RPC device names are `RPC<N>`. N is a **process-global counter across all servers**. The description is the endpoint string
   (`host:port`). The **buffer and backend names** are different: `RPC<d>[host:port]`, where `d` is the device index **inside
   that server**. So two single-GPU servers give devices `RPC0` and `RPC1`, but buffers `RPC0[A:50052]` and `RPC0[B:50052]`.
3. Token embeddings always stay on the **CPU** (host). The output head goes to the device of virtual layer `n_layer_all`, which
   with full offload is the **last device** (the local GPU when RPC is present). Repeating layers are contiguous slices
   assigned by cumulative normalized `tensor_split` (or by free memory when `-ts` is unset).
4. `--fit` defaults to **on**. `-ngl` defaults to `auto` (-1). With fit on, fit may rewrite `n_ctx` (only if unset/0),
   `n_gpu_layers`, `tensor_split` and tensor overrides. With `--fit off`, `auto` means all layers. Fit treats RPC devices
   like any GPU and reads their remote free memory.
5. **At default verbosity (3) llama-server does NOT print** `offloaded N/M layers`, `model buffer size`, `KV buffer size`,
   `compute buffer size` or `using device` lines. They are libllama INFO logs, which the common logger maps to TRACE (4).
   Use `-lv 4` (or env `LLAMA_ARG_LOG_VERBOSITY=4`) to see them, and `-lv 5` / `-v` to also get per-layer device lines.
6. All non-output logs go to **stderr**. `--log-jsonl` sends everything to **stdout** as JSONL. `--log-file`,
   `--log-colors`, `--log-prefix/--no-log-prefix` and `--log-timestamps/--no-log-timestamps` all exist.
   Prefix and timestamps are **on** by default for llama-server.
7. Ordering trap: CLI args are applied in argv order, and `--rpc` connects at parse time. `--device RPC0,...` and
   `--list-devices` must come **after** `--rpc`. An unreachable RPC endpoint calls `GGML_ABORT` (the process dies).

---

## 1. Device list construction

### 1.1 How `--rpc` registers devices (client side)

`common/arg.cpp` L2683-2693, option `--rpc` (env `LLAMA_ARG_RPC`):

```cpp
{"--rpc"}, "SERVERS",
"comma-separated list of RPC servers (host:port)",
...
    if (!llama_supports_rpc()) { throw std::invalid_argument("RPC not supported in this build"); }
    add_rpc_devices(value);
```

`common/arg.cpp` L1169-1188, `add_rpc_devices`. It splits on `,`, calls `ggml_backend_rpc_add_server(server)` for each
endpoint **in list order**, then `ggml_backend_register(reg)`:

```cpp
for (const auto & server : rpc_servers) {
    auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
    ggml_backend_register(reg);
}
```

`ggml/src/ggml-backend-reg.cpp` L186-219: `register_backend` appends the reg, then `register_device` appends each of its
devices to the global `devices` vector (dedup by pointer). Local backends (CUDA L121, Metal L124, ..., RPC reg L163, CPU
L172) are registered first by `ggml_backend_load_all()`. So in the global list, the RPC server devices come **after** all
local devices. The static RPC reg has 0 devices: `ggml_backend_rpc_reg_get_device_count` returns 0 when `ctx == nullptr`
(L2952-2955).

### 1.2 RPC device naming (ggml/src/ggml-rpc/ggml-rpc.cpp)

`ggml_backend_rpc_add_server`, L3150-3191:

```cpp
static uint32_t dev_id = 0;                                   // L3153, process-global
...
uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);   // L3159, RPC_CMD_DEVICE_COUNT round trip
if (dev_count == 0) { return nullptr; }
ctx->name = "RPC[" + std::string(endpoint) + "]";             // L3165, per-server reg name
for (uint32_t ind = 0; ind < dev_count; ind++) {
    std::string dev_name = "RPC" + std::to_string(dev_id);    // L3167, device NAME
    std::string dev_desc = std::string(endpoint);             // L3168, device DESCRIPTION
    ... /* .device = */ ind, ...                              // index inside that server
    /* .reg = */ ggml_backend_rpc_reg(),                      // L3178, the static reg named "RPC"
    dev_id++;
}
```

The same endpoint is registered only once (`reg_map`, L3156-3158).

Buffer type name, `ggml_backend_rpc_buffer_type`, L1318-1321:

```cpp
std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
```

Backend name, `ggml_backend_rpc_init`, L1350-1351: `"RPC" + std::to_string(device) + "[" + endpoint + "]"`.

The device's default buffer type is `ggml_backend_rpc_buffer_type(ctx->endpoint, ctx->device)` (L2880-2884). `device`
here is the per-server index `ind`, not the global `dev_id`.

| Thing | Format | Example: `--rpc 10.0.0.2:50052,10.0.0.3:50052`, one GPU each |
|---|---|---|
| Device name (`--device`, `--list-devices`, `using device`) | `RPC<global id>` | `RPC0`, `RPC1` |
| Device description | `<endpoint as typed>` | `10.0.0.2:50052`, `10.0.0.3:50052` |
| Reg name (per server) | `RPC[<endpoint>]` | `RPC[10.0.0.2:50052]` |
| Reg returned by `ggml_backend_dev_backend_reg(dev)` | `RPC` (static reg, L2947-2950) | `RPC` |
| Buffer type / buffer name (model, KV, compute log lines) | `RPC<per-server idx>[<endpoint>]` | `RPC0[10.0.0.2:50052]`, `RPC0[10.0.0.3:50052]` |

Other RPC device properties (L2850-2869, L2920-2935):
- type is hard-coded to `GGML_BACKEND_DEVICE_TYPE_GPU` (L2853, comment `// TODO: obtain value from the server`)
- caps are `async=true`, `host_buffer=false`, `buffer_from_host_ptr=false`, `events=true`, `mmap_support=true`
- `get_host_buffer_type = NULL`, `buffer_from_host_ptr = NULL`
- memory is queried live from the server: `ggml_backend_rpc_get_device_memory` sends `RPC_CMD_GET_DEVICE_MEMORY` (L1374-1384).
  The server answers with `ggml_backend_dev_memory` of its own device (L2413-2426).
- proc addresses (L3098-3116) are only `ggml_backend_rpc_add_server`, `ggml_backend_rpc_start_server` and
  `ggml_backend_comm_*`. There is no `ggml_backend_split_buffer_type` and no `ggml_backend_dev_get_extra_bufts`.

Local device names, for comparison:
- CUDA: `GGML_CUDA_NAME` is `"CUDA"` (`ggml/include/ggml-cuda.h` L17; `"ROCm"` L11 and `"MUSA"` L14 for those builds).
  The device name is `GGML_CUDA_NAME + std::to_string(i)`, giving `CUDA0` (`ggml-cuda.cu` L6217). The pinned host buffer is
  `CUDA_Host` (L1323-1325). The device type is `IGPU` if `prop.integrated`, else `GPU` (L5488-5497).
- Metal: `snprintf(dev->props.name, ..., "%s%d", "MTL", device)` gives `MTL0` (`ggml-metal-device.m` L1302).
  `ggml_backend_metal_device_get_type` returns `GGML_BACKEND_DEVICE_TYPE_GPU` (ggml-metal.cpp L692-696), so at this commit
  Apple Silicon counts as a **discrete GPU** for llama's device selection. The buffer types are `MTL0`, `MTL0_Private` and
  `MTL0_Mapped` (L311, L388, L466). Caps: `buffer_from_host_ptr=true`, `mmap_support=true` (L704-710).

### 1.3 Default device list (no `--device`)

`src/llama.cpp` L158-312, `llama_prepare_model_devices`, called from `llama_model_load` L327. In the non-tensor-split path
(L220-273) it iterates `ggml_backend_dev_get(i)` in global registry order:
- `CPU` and `ACCEL` are skipped (L224-228)
- `GPU` whose reg name is `"RPC"` goes to `rpc_servers` (L230-233)
- other `GPU` goes to `gpus`, deduplicated by `props.device_id` (L234-257, log `skipping device %s (%s) with id %s - already using device %s (%s) with the same id`)
- `IGPU` goes to `igpus` (only the first backend's iGPUs, L260-266)

Then L275-285:

```cpp
// add RPC servers at the front of the list to minimize network transfers
model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());
// add GPUs
model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());
// add integrated GPUs only if no discrete GPUs were found
// (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
if (gpus.empty()) { model->devices.insert(model->devices.end(), igpus.begin(), igpus.end()); }
```

**Resulting order:** `[RPC devices in --rpc order (each server's devices consecutive)] + [local discrete GPUs in registry order] + [local iGPUs only if no discrete GPU]`.

Examples:
- Windows host with CUDA0, `--rpc mac:50052` (mac rpc-server exposing MTL0): `[RPC0, CUDA0]`
- Mac host with MTL0, `--rpc win:50052` (CUDA0 remote): `[RPC0, MTL0]`
- CPU-only host, `--rpc a,b`: `[RPC0, RPC1]`. The CPU is not in `model->devices`. CPU-assigned layers come only from `-ngl` < all.

`-sm none` (L289-301): keeps only `devices[main_gpu]`. `main_gpu` indexes the **RPC-first** list, so with RPC present the
default `-mg 0` selects `RPC0`, not the local GPU. `main_gpu < 0` clears the list (pure CPU).

After selection, each device is logged at L303-309 (INFO, hidden by default; see section 6):

```cpp
LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
        ggml_backend_dev_name(dev.dev), ggml_backend_dev_description(dev.dev),
        props.device_id ? props.device_id : "unknown id",
        props.memory_free/1024/1024);
```

Here `__func__` is `llama_prepare_model_devices`. Example:
`llama_prepare_model_devices: using device RPC0 (10.0.0.2:50052) (unknown id) - 23000 MiB free`.
The RPC device props do not set `device_id`, so it prints `unknown id`.

### 1.4 Explicit `--device`

`common/arg.cpp` L2741-2748 (`-dev/--device`, env `LLAMA_ARG_DEVICE`) calls `parse_device_list` (L1122-1142):
- `none` gives `[nullptr]`, which means an empty device list (all CPU)
- otherwise each name is looked up with `ggml_backend_dev_by_name`, which is **case-insensitive** (`striequals`,
  ggml-backend-reg.cpp L345-353). CPU devices are rejected with `invalid device: %s`. A `nullptr` terminator is appended.

`src/llama.cpp` L160-186: with `params.devices` set (and split mode not tensor), it does
`for (dev = params.devices; *dev; ++dev) model->devices.push_back({false, *dev});`. **The order is exactly the order given.**
There is no RPC-first reordering, no GPU dedup and no iGPU rule. RPC servers that are registered via `--rpc` but not named
in `--device` are unused (their TCP connection was still opened at parse time).

**Parse-order trap** (common/arg.cpp L768-860): environment variables are applied first, in option-definition order
(`--rpc` is defined at L2684, before `-dev` at L2742, so `LLAMA_ARG_RPC` then `LLAMA_ARG_DEVICE` works). Then CLI args
are applied in **argv order** (`for (int i = 1; i < argc; i++)`, L822). So `--device RPC0,CUDA0 --rpc h:p` fails with
`invalid device: RPC0`. Put `--rpc` first. The same applies to `--list-devices` (L2749-2756, which prints and `exit(0)`
immediately) and `-mmdev` (comment at L2613).

`--list-devices` format (`common_print_available_devices`, L1144-1167, `printf` to **stdout**, CPU excluded):

```
Available devices:
  %s: %s (%zu MiB, %zu MiB free)
```

For example `  RPC0: 10.0.0.2:50052 (24576 MiB, 23110 MiB free)`. This is a good probe for Hypercompute: it does real remote
`GET_DEVICE_MEMORY` calls and needs no model.

### 1.5 `--tensor-split` mapping

`common/arg.cpp` L2845-2869: splits on `,` or `/` (regex `[,/]+`), stores into `params.tensor_split[i]`, and zero-fills
the rest. It throws if `split_arg.size() >= llama_max_devices()` (L2854). `llama_max_devices()` returns 16 (`src/llama.cpp`
L86-88), so at most 15 values are accepted.

`src/llama-model.cpp` L1634-1665 (load_tensors):

```cpp
bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
...
if (all_zero) { /* default split, by free memory */ ... ggml_backend_dev_memory(dev, &free, &total); ... splits[i] = free; }
else { std::copy(tensor_split, tensor_split + n_devices(), splits.begin()); }
// sum and normalize the splits to get the split points (cumulative)
```

**`tensor_split[i]` applies to `model->devices[i]`**, the list from 1.3/1.4: RPC-first by default, or `--device` order.
Values beyond `n_devices()` are ignored. Missing values are 0, and a device with 0 gets no layers. When all values are 0,
the default weight is each device's **reported free memory** (live remote free memory for RPC). A device that reports 0/0
falls back to host memory (L1645-1651).

---

## 2. Load-time log lines that reveal placement (verbatim format strings)

All of these are `LLAMA_LOG_INFO` unless marked DEBUG. Section 6 explains why INFO is hidden by default in llama-server.

| Where | Format string (verbatim) | `%s` prefix value |
|---|---|---|
| `src/llama.cpp` L306 | `"%s: using device %s (%s) (%s) - %zu MiB free\n"` | `llama_prepare_model_devices` |
| `src/llama-model.cpp` L1617 | `"%s: loading model tensors, this can take a while... (load_mode = %s)\n"` | `load_tensors` |
| `src/llama-model.cpp` L1672, L1677 (DEBUG) | `"load_tensors: layer %3d assigned to device %s, is_swa = %d\n"` | literal |
| `src/llama-model.cpp` L1995 | `"%s: offloading output layer to GPU\n"` | `load_tensors` |
| `src/llama-model.cpp` L1998 | `"%s: offloading %d repeating layers to GPU\n"` | `load_tensors` |
| `src/llama-model.cpp` L2003 | `"%s: offloaded %d/%d layers to GPU\n"` | `load_tensors` |
| `src/llama-model.cpp` L2009 | `"%s: %12s model buffer size = %8.2f MiB\n"` | `load_tensors` |
| `src/llama-context.cpp` L395/397 | `"%s: n_seq_max             = %u\n"`, `"%s: n_ctx_seq             = %u\n"` | `llama_context` |
| `src/llama-context.cpp` L468 | `"%s: %10s  output buffer size = %8.2f MiB\n"` (two spaces before `output`) | `llama_context` |
| `src/llama-kv-cache.cpp` L223 (DEBUG) | `"%s: layer %3d: dev = %s\n"` | `llama_kv_cache` |
| `src/llama-kv-cache.cpp` L291 | `"%s: %10s KV buffer size = %8.2f MiB\n"` | `llama_kv_cache` |
| `src/llama-kv-cache.cpp` L301 | `"%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n"` | `llama_kv_cache` |
| `src/llama-context.cpp` L548 | `"%s: pipeline parallelism enabled\n"` | `llama_context` |
| `src/llama-context.cpp` L722 | `"%s: reserving ...\n"` | `sched_reserve` |
| `src/llama-context.cpp` L842 | `"%s: %10s compute buffer size = %8.2f MiB\n"` | `sched_reserve` |
| `src/llama-context.cpp` L856 | `"%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n"` | `sched_reserve` |

Notes:
- `model buffer size` iterates `pimpl->ctxs_bufs` and prints `ggml_backend_buffer_name(buf)`, which is the buffer type name
  (`ggml-backend.cpp` L252-254). Typical names: `CPU`, `CPU_Mapped` (mmapped CPU weights, ggml-backend.cpp L2604),
  `CUDA0`, `CUDA_Host`, `MTL0`, `MTL0_Mapped`, `RPC0[host:port]`. With mmap and several files or disjoint ranges, the
  **same name can appear on several lines**, so sum them per name.
- `offloaded %d/%d`: numerator `min(n_gpu_layers(), n_layer_all+1)`, denominator `n_layer_all+1` (the +1 is the output
  layer). It is printed whenever `llama_supports_gpu_offload()` is true, which includes "RPC backend present" (`src/llama.cpp`
  L101-109). The text always says "GPU", even when the device is an RPC device.
- The KV `size =` line: `layers.size()` counts layers that own a KV tensor; `%2u/%u seqs` is `n_seq_max/n_stream`.
- The compute buffer line is printed only when the expected size is `> 1` byte (L841). It uses the scheduler buffer type name.
- The per-layer device lines (`layer %3d assigned to device %s`, `layer %3d: dev = %s`) are DEBUG only. They need `-lv 5` or `-v`.

Log line shape on stderr with llama-server defaults (prefix and timestamps on, see 6):
`<M>.<SS>.<mmm>.<uuu> I load_tensors: offloaded 33/33 layers to GPU`. The timestamp is microseconds since logger start
(common/log.cpp L128-134, L290-292). There are also ANSI color codes if the output is a TTY.

Suggested parse regexes (apply to the message part, or to the `msg` field with `--log-jsonl`):

```
load_tensors: offloaded (\d+)/(\d+) layers to GPU
load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB
llama_kv_cache:\s+(\S+) KV buffer size =\s+([\d.]+) MiB
sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB
llama_context:\s+(\S+)\s+output buffer size =\s+([\d.]+) MiB
llama_prepare_model_devices: using device (\S+) \((.*)\) \((.*)\) - (\d+) MiB free
load_tensors: layer\s+(\d+) assigned to device (\S+), is_swa = (\d)        # -lv 5
```

---

## 3. What lives where (layer split mode, the default `-sm layer`)

`src/llama-model.cpp` load_tensors, L1667-1692:

```cpp
const int i_gpu_start    = std::max(n_layer_all + 1 - n_gpu_layers, 0);
const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) { /* CPU */ return {cpu_dev, &pimpl->cpu_buft_list}; }
    const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(),
                                           float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
    auto * dev = devices.at(layer_gpu).dev; ...
};
// there is very little benefit to offloading the input layer, so always keep it on the CPU
pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };
for (int il = 0; il < n_layer_all; ++il) { pimpl->dev_layer[il] = get_layer_buft_list(il); }
pimpl->dev_output = get_layer_buft_list(n_layer_all);
```

where `n_gpu_layers()` is (L2418-2421) `params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1`
(`// note: plus 1 for the "output" layer`).

Which tensor uses which list (`src/llama-model-loader.cpp` L1217-1229, classes from `src/llama-arch.cpp`):
- `LLM_TENSOR_TOKEN_EMBD` is `LLM_TENSOR_LAYER_INPUT` (arch L814), so it uses `dev_input` = **CPU** always. When mmap is on,
  host buffer types are avoided (loader L1270-1275), so it lands in `CPU_Mapped`.
- `LLM_TENSOR_OUTPUT` and `LLM_TENSOR_OUTPUT_NORM` are `LLM_TENSOR_LAYER_OUTPUT` (arch L819, L843), so they go to
  `dev_output` = the device of virtual layer index `n_layer_all`.
- `LLM_TENSOR_TOKEN_EMBD_NORM` is REPEATING (arch L818, "do the norms on the first layer"), so it sits with layer 0.
- Repeating `blk.N.*` tensors go to `dev_layer[N]`.
- `--override-tensor`/`-ot` and `--cpu-moe`/`--n-cpu-moe` patterns (loader L1236-1261) take precedence per tensor.

Rule: virtual layer index `il` in `[0, n_layer_all]` (the last one is output). If it is GPU-eligible, it goes to the first
device `i` with `cum_split[i] > (il - i_gpu_start)/act_gpu_layers`, where `cum_split` is the normalized cumulative
`tensor_split` or free memory (strict `upper_bound`). Layers below `i_gpu_start` stay on CPU, so partial offload
keeps the **first** layers on CPU and offloads the **last** layers plus output.

Consequences:
- Each device gets one **contiguous** slice. Device 0 gets the lowest layers. With RPC-first ordering, remote machines get the
  early layers and the **local GPU gets the last layers plus the LM head** (the source comment says this minimizes network
  transfers). Logits are then produced locally.
- A device with split 0 gets nothing (equal cumulative values are skipped by `upper_bound`).

Worked examples. These were computed by re-implementing the exact formula above in a 25-line local script. They are not
llama.cpp runs.

| n_layer_all | -ngl | -ts | dev0 | dev1 | CPU |
|---|---|---|---|---|---|
| 32 | all/auto(fit off) | 1,1 | layers 0-16 (17) | layers 17-31 + output (16 units) | tok_embd only |
| 32 | all | 3,1 | layers 0-24 (25) | layers 25-31 + output (8 units) | tok_embd only |
| 32 | 20 | 1,1 | layers 13-22 (10) | layers 23-31 + output (10 units) | tok_embd + layers 0-12 |
| 28 | all | 1,1 | layers 0-14 (15) | layers 15-27 + output (14 units) | tok_embd only |
| 32 | all | 1,0,1 (3 devs) | layers 0-16 | dev1: none; dev2: 17-31 + output | tok_embd only |

The KV cache of layer `il` is allocated on `model.dev_layer(il)` (section 5). So KV follows the layer when `offload_kqv`
is on (the default; `--no-kv-offload` disables it, `common/common.cpp` L1738).

Other split modes:
- `-sm row` with an RPC device **throws** `device %s does not support split buffers`. `make_gpu_buft_list` (L1199-1224)
  needs the reg proc `ggml_backend_split_buffer_type`, and the RPC reg does not provide it (ggml-rpc.cpp L3098-3116).
- `-sm tensor` builds a Meta device over all non-CPU-buffer devices (`src/llama.cpp` L193-219). The RPC reg exposes
  `ggml_backend_comm_*` allreduce. This is EXPERIMENTAL per the CLI help (arg.cpp L2826) and outside this audit.
  `--fit` refuses it (fit.cpp L183-185).

---

## 4. `--fit` and `-ngl`

Defaults (`common/common.h` L479-488):

```cpp
int32_t n_gpu_layers       = -1;    // number of layers to store in VRAM, -1 is auto, <= -2 is all
int32_t main_gpu           = 0;
float   tensor_split[128]  = {0};
bool    fit_params         = true;  // whether to fit unset model/context parameters to free device memory
bool    fit_params_print   = false;
int32_t fit_params_min_ctx = 4096;
std::vector<size_t> fit_params_target = std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024);   // 1 GiB margin per device
enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;
```

CLI (`common/arg.cpp`):
- `-ngl/--gpu-layers/--n-gpu-layers` (L2802-2818): `auto` gives -1, `all` gives -2, otherwise `stoi`. Help:
  `max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)`.
- `-fit/--fit [on|off]` (L2881-2894, env `LLAMA_ARG_FIT`), default `on`.
- `-fitp/--fit-print` (L2895-2908); `-fitt/--fit-target MiB0,MiB1,...` (L2909-2933, per device in model-device order;
  a single value is broadcast); `-fitc/--fit-ctx N` (L2934-2940, min ctx, default 4096).
- `-c 0` sets `fit_params_min_ctx = UINT32_MAX`, which disables ctx reduction (L1645-1649).

Meaning of -ngl at the llama level (`n_gpu_layers()`, llama-model.cpp L2418-2421): any negative value means all
`n_layer_all+1`. So:
- `--fit off -ngl auto` (or no -ngl) gives **all layers offloaded**.
- `-ngl all` (-2) also gives all layers. But because -2 differs from `llama_model_default_params().n_gpu_layers` (-1,
  llama-model.cpp L3351), fit treats it as user-set.

What fit does (`common/common.cpp` L1240-1276, `common_init_result` ctor; it runs before the real load whenever
`params.fit_params`):
- Fit calls `common_fit_params(...)` with `params.tensor_split` and `params.tensor_buft_overrides` as writable outputs.
- `common/fit.cpp` `common_params_fit_impl` (L178-882):
  1. Probe (`common_get_device_memory_data_impl`, L30-152): it loads the model **and** a context with
     `no_alloc = true, load_mode = NONE`, takes `llama_get_memory_breakdown`, and reads `ggml_backend_dev_memory` for each
     model device. This is the same device list as the real load, so it **includes RPC devices**, and their free memory
     comes from the remote server. A GPU or iGPU reporting 0/0 is skipped with the warning
     `"%s: device %s did not report memory; --fit will not use it\n"` (L120-127).
  2. If every device keeps at least its margin, it does nothing (`no changes needed`, L344-363).
  3. Step 2: if `n_ctx` was 0 (auto), it lowers the context, never below `fit_params_min_ctx * n_streams`, by linear
     interpolation (L366-458). A user-set `-c N` is never changed.
  4. Then it aborts (throws `common_params_fit_exception`) if: `n_gpu_layers` differs from the default (`"n_gpu_layers already
     set by user to " ... ", abort"`, L463-465); for more than one device, any `tensor_split[id] != 0` (`"model_params::tensor_split already set by user, abort"`,
     L470-474); `-sm row` (L477-479); or user tensor overrides (L484-486).
  5. Step 3: it fills devices **back to front** (`for (int id = nd - 1; id >= 0; id--)`, L675), using a false-position
     search per device against `free - margin`. For MoE models it first moves all expert tensors to CPU and may
     add per-layer overflow overrides. Results are written as `mparams.n_gpu_layers = sum`, `tensor_split[id] = n_layer of
     device id`, plus `tensor_buft_overrides` (L547-585).
- `common_fit_params` (L884-902) catches the exception and logs a WARN:
  `"%s: failed to fit params to free device memory: %s\n"`. Loading then **continues with the user's params**. Any `n_ctx`
  reduction already applied is kept, because cparams were modified in place. Hard errors log
  `"%s: encountered an error while trying to fit params to free device memory: %s\n"`. The caller ignores the status.

RPC interaction, verified from code:
- Fit uses RPC devices like any other device. Back-to-front filling means the **last** device (the local GPU in the default
  RPC-first order) is filled first, and remote devices receive the remaining layers.
- Each probe iteration rebuilds the model and context. For RPC devices it makes network calls (device memory, and
  `RPC_CMD_GET_ALLOC_SIZE` for some tensors, ggml-rpc.cpp L1034-1047). Zero-size dummy buffers do not hit the network
  (`ggml_backend_buft_alloc_buffer` returns a dummy for size 0, ggml-backend.cpp L39-44). Fit time therefore grows with RTT.
  The magnitude is UNVERIFIED (not measured).
- Recommendation for deterministic two-node runs: pass `--fit off -ngl all -ts a,b` (or `--fit off` and leave -ts empty to
  split by free memory). Otherwise fit can silently change `n_ctx`, the layer counts and `-ot` overrides. Fit's
  `tensor_buft_overrides` also disable pipeline parallelism (`!model.has_tensor_overrides()`, llama-context.cpp L516-523).

Pipeline parallelism (llama-context.cpp L516-549) turns on only when `n_devices() > 1`, `n_gpu_layers() > n_layer_all`
(full offload including output), `-sm layer`, `offload_kqv`, no tensor overrides, `moe_cache_size == 0`, and every non-CPU
backend has `async && events`. RPC, CUDA (unless `GGML_CUDA_NO_PEER_COPY`) and Metal all report both. When on, the scheduler
uses `GGML_SCHED_MAX_COPIES` (default 4) copies (ggml-backend.cpp L911-912, L1973), which increases compute buffer size.

---

## 5. KV cache size and compute buffer reservation

Context size (`src/llama-context.cpp`):
- L219: `cparams.n_ctx = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;` (fit may have already replaced 0)
- L377: `cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);`
- L379-391: unified KV gives `n_ctx_seq = n_ctx`. Otherwise `n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)` and `n_ctx` is
  rewritten to `n_ctx_seq * n_seq_max`.
- llama-server sets `n_parallel = -1` (auto) by default (arg.cpp L1405-1406). Auto becomes `n_parallel = 4, kv_unified = true`
  (tools/server/server.cpp L169-174). So the default server has **one stream of `n_ctx` cells**.

KV cache per layer (`src/llama-kv-cache.cpp`, ctor L65; `n_stream(unified ? 1 : n_seq_max)` L84; loop L164-249). Standard
models use `llama_kv_cache(..., kv_size = cparams.n_ctx_seq, ...)` (llama-model.cpp L3292-3308). SWA models use
`llama_kv_cache_iswa` (L3236-3289), whose SWA-layer size rule is not audited here (UNVERIFIED).

```cpp
const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);                                     // n_embd_head_k(il) * n_head_kv(il)
const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();
...
if (offload) { auto * dev = model.dev_layer(il); buft = ggml_backend_dev_buffer_type(dev); dev_name = ggml_backend_dev_name(dev); }
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;   // has_v = !is_mla
```

So, per layer that has KV:

```
K_bytes(il) = row_size(type_k, n_head_kv(il) * head_dim_k(il)) * kv_size * n_stream
V_bytes(il) = row_size(type_v, n_head_kv(il) * head_dim_v(il)) * kv_size * n_stream     (0 for MLA models)
with defaults type_k = type_v = f16 (2 bytes/elem; common.h L595-596),
kv_size * n_stream = n_ctx_seq * n_stream ~= padded n_ctx
=> KV_bytes(il) ~= 2 * n_ctx * n_head_kv * (head_dim_k + head_dim_v)
```

(`v_trans = !flash_attn`. When V widths vary across layers and FA is off, V is padded to `n_embd_v_gqa_max()`, L157-160.)
Totals are `size_k_bytes()` / `size_v_bytes()` = the sum of `ggml_nbytes` (L1922-1940). The device is `dev_layer(il)`, so
the KV buffer name equals that device's default buffer type (for example `RPC0[10.0.0.2:50052]`, `CUDA0`, `MTL0`, or `CPU`
when the layer is on CPU or `--no-kv-offload`). One `KV buffer size` line is printed per buffer type.

Illustrative arithmetic (hparams assumed, not read from a GGUF): 32 layers, `n_head_kv = 8`, `head_dim = 128`, f16,
`n_ctx = 8192`. That is 2 * 8192 * 8 * 256 = 32 MiB per layer and 1 GiB total, split between devices in proportion to
their layer counts.

Compute buffer:
- Backends (llama-context.cpp L418-445): one per `model.devices` entry **in the same order**, then ACCEL (for example BLAS),
  then CPU.
- Scheduler buffer types (L484-505): each backend's default buffer type, except CPU, which uses
  `ggml_backend_dev_host_buffer_type(model.devices[0])` when available (L499-506). With RPC first this is NULL (RPC has no
  host buffer type), so it stays `CPU`. With CUDA first it is `CUDA_Host`.
- Reserved in the ctor via `sched_reserve()` (call L555, definition L715). It creates the scheduler
  (`sched_new(max_nodes, cparams.pipeline_parallel)`, L741 → L707-709) and reserves the worst-case prompt-processing graph
  with `n_tokens = min(n_ctx, n_ubatch)` and `n_seqs = n_seq_max` (L728-729), plus the token-generation graph. It then prints
  one `compute buffer size` line per backend with expected size > 1 byte (L835-845) and the `graph` summary (L856).
  `sched_reserve()` is called again on later reconfiguration (L1625, L1915, L3773). It returns immediately unless
  `sched_need_reserve` is set.
- The output (logits) buffer is reserved separately: `output_reserve(n_seq_max)`, logged at L468.

---

## 6. Logging in llama-server

Init: `tools/server/server.cpp` L107 calls `common_init()`, which (common/common.cpp L387-398) runs
`common_log_set_prefix(common_log_main(), true); common_log_set_timestamps(common_log_main(), true);
llama_log_set(common_log_default_callback, NULL);`. `llama_log_set` also calls `ggml_log_set` (src/llama-impl.cpp L55-59),
so ggml backend logs (CUDA/Metal/RPC `GGML_LOG_*`) go through the same filter.

**Verbosity filter (the key finding):**
- `common_params::verbosity = 3; // LOG_LEVEL_INFO` (common.h L541). The threshold starts at `LOG_DEFAULT_LLAMA = LOG_LEVEL_INFO = 3`
  (log.h L26-32, log.cpp L17). `common_params_parse_ex` re-applies it (arg.cpp L772).
- `common_log_default_callback` (log.cpp L548-566) maps library levels: `GGML_LOG_LEVEL_INFO -> LOG_LEVEL_TRACE (4)`,
  `CONT -> TRACE (4)`, `DEBUG -> 5`, `WARN -> 2`, `ERROR -> 1`. It drops the message unless `verbosity <= threshold`.
- Result: **every `LLAMA_LOG_INFO`/`GGML_LOG_INFO` from section 2 is dropped at the default threshold 3.** WARN and ERROR
  still show. The server's own `SRV_INF` (= `LOG_INF`, level 3, server-common.h L36) still shows, for example
  `srv  llama_server: listening on http://127.0.0.1:8080` and `model loaded`.
- To see placement lines use `-lv 4` / `--verbosity 4` / `--log-verbosity 4` / env `LLAMA_ARG_LOG_VERBOSITY=4` (arg.cpp L3979-3993).
  For per-layer DEBUG lines use `-lv 5`, or `-v` / `--verbose` / `--log-verbose` (INT_MAX, L3964-3971).
- `common_params_print_info` (common.cpp L400-421) prints `verbosity = %d (adjust with the `-lv N` CLI arg)` at INFO, and the
  `device_info:` list (`  - %-8s: %s (%zu MiB, %zu MiB free)`) only when the threshold is at least 4.
- Fit's own report and the memory breakdown table (`common_memory_breakdown_print`, fit.cpp L1075-1079) are `LOG_TRC`, so they
  are also hidden at 3. During fit probes, library logs below ERROR are demoted to DEBUG unless verbosity is at least 5
  (common.cpp L1275; fit.cpp L50-54).

**Streams** (`common_log_entry::print`, log.cpp L91-157):

```cpp
fcur = stdout;
if (level != GGML_LOG_LEVEL_NONE && !jsonl) { fcur = stderr; }
```

So INF/WRN/ERR/DBG/TRC go to **stderr**. Only `LOG(...)` (level NONE, "output") goes to stdout. With `--log-jsonl`
**everything** goes to **stdout** as `{"type":"log","time":...,"level":...,"msg":...}`. DEBUG lines are suppressed on the
console unless the threshold is at least 5. Direct `printf` (for example `--list-devices`) goes to stdout. A few raw
`fprintf(stderr, ...)` warnings exist (for example the -ngl warnings at arg.cpp L2812-2816).

**Flags that exist at this commit** (common/arg.cpp):

| Flag | Lines | Env | Behavior |
|---|---|---|---|
| `--log-disable` | L3913-3919 | none | `common_log_pause` |
| `--log-file FNAME` | L3920-3926 | `LLAMA_ARG_LOG_FILE` | writes to the file **in addition to** the console (`print_entry`, log.cpp L207-216). Opened with `"w"` (truncates). |
| `--log-jsonl` / `--no-log-jsonl` | L3927-3934 | `LLAMA_ARG_LOG_JSONL` | JSONL to stdout, disables colors. Also enables `LOG_JSON(...)` records such as `fit_memory_breakdown` (fit.cpp L1081, log.h L139-147). |
| `--log-colors [on\|off\|auto]` | L3947-3963 | `LLAMA_ARG_LOG_COLORS` | default `auto` (TTY detection) |
| `-v`, `--verbose`, `--log-verbose` | L3964-3971 | none | threshold INT_MAX |
| `-lv`, `--verbosity`, `--log-verbosity N` | L3979-3993 | `LLAMA_ARG_LOG_VERBOSITY` | 0 output, 1 error, 2 warn, 3 info, 4 trace, 5 debug (default 3) |
| `--log-prefix` / `--no-log-prefix` | L3994-4001 | `LLAMA_ARG_LOG_PREFIX` | level letter prefix `I `/`W `/`E `/`D `; on by default via `common_init` |
| `--log-timestamps` / `--no-log-timestamps` | L4002-4009 | `LLAMA_ARG_LOG_TIMESTAMPS` | `M.SS.mmm.uuu ` prefix; on by default via `common_init` |
| `--log-prompts-dir PATH` | L3935-3946 | none | server/cli only, for debugging |

Server readiness: the HTTP server starts **before** the model loads (server.cpp L493-494, comment "start the HTTP
server before loading the model to be able to serve /health requests"). `model loaded` is logged after load (L517), and
`listening on %s` after that (L543-545).

At server shutdown, `common_memory_breakdown_print(ll_ctx)` runs (server.cpp L584-587). That is TRC level, plus a JSON record
if `--log-jsonl` is set.

---

## 7. rpc-server side facts relevant to placement

`tools/rpc/rpc-server.cpp`:
- defaults `host = "127.0.0.1"`, `port = 50052`, `use_cache = false`, `n_threads = max(1, hw_concurrency/2)` (L94-100)
- flags `-t/--threads`, `-d/--device <dev1,...>`, `-H/--host`, `-p/--port`, `-c/--cache` (L102-112)
- device choice (L170-211): `-d` names if given. Otherwise **all non-CPU, non-ACCEL devices**. If there are none, it uses the CPU.
  The number of devices it exposes is what `RPC_CMD_DEVICE_COUNT` returns, which sets how many `RPC<N>` devices the client
  creates.
- a non-loopback bind prints the `Never expose the RPC server to an open network!` warning (L224-232)
- startup log (ggml-rpc.cpp L2769-2781, `GGML_LOG_INFO`, so the ggml default logger writes it to stderr):
  `Starting RPC server v%d.%d.%d`, `  endpoint       : %s`, `  local cache    : %s`, `Devices:`, `  %s: %s (%zu MiB, %zu MiB free)`
- protocol is `RPC_PROTO_MAJOR_VERSION 8`, minor 0, patch 0 (ggml-rpc.h L9-11). The client rejects a server whose major
  differs or whose minor is greater (ggml-rpc.cpp L432). `GGML_RPC_MAX_SERVERS 16` (ggml-rpc.h L17).
- client connect failure is fatal: `rpc_dispatcher::start` calls `GGML_ABORT("Failed to connect to %s\n")` or
  `GGML_ABORT("RPC handshake failed for %s\n")` (ggml-rpc.cpp L637-656). This runs during argument parsing (`--rpc`), so a
  dead peer kills llama-server before any model work. Hypercompute should TCP-probe peers before launching.

---

## 8. UNVERIFIED / not covered

- Exact SWA KV sizing in `llama_kv_cache_iswa` (Gemma, etc.): not read.
- Whether the Metal backend on the rpc-server side reports accurate free memory for unified memory: not read
  (`ggml_metal_device_get_memory` not audited).
- Real-world fit duration over RPC, and actual buffer sizes: needs hardware runs. Nothing here was executed.
- `-sm tensor` with RPC allreduce: not audited beyond the existence of `ggml_backend_comm_*` procs.
- Behavior of `ggml_backend_load_all()` ordering under `GGML_BACKEND_DL` dynamic-loading builds (prebuilt release
  binaries): the static registration order (CUDA, Metal, ..., RPC, CPU) is from `ggml-backend-reg.cpp` L118-173. Dynamic
  search order was not read. RPC-first placement does not depend on it, but the relative order of several local GPUs might.

---

## Addendum (critic)

Critic pass 2026-10-20, same ref `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`, read with
`gh api -H "Accept: application/vnd.github.raw" ...`. Nothing built or run.

### Open items in this file

| Item | Matters for | Status after this pass |
|---|---|---|
| §8 "Behavior of `ggml_backend_load_all()` ordering under `GGML_BACKEND_DL` (prebuilt release binaries)" | (a), (d) | RESOLVED in P1 below. The Windows release binaries are DL builds. The macOS arm64 ones are not |
| Config files that inject options before env and CLI | (a) deterministic launch | NEW, verified in P2 below |
| Exact stderr text and exit code when `--rpc` / `--device` fail during parsing | (a) error mapping | RESOLVED in P3 below |
| `--log-jsonl` record shape for the (b) parser (§6 gives keys only) | (b) | RESOLVED in P4 below |
| §1.4 "their TCP connection was still opened at parse time" | (d) | CORRECTED. See llama_rpc.md "Addendum (critic)" C1: the connection is opened and closed during parsing |
| §4 fit duration over RPC: "magnitude is UNVERIFIED" | (a) | Partly resolved. In source, every remote memory query is a fresh TCP connect + HELLO (llama_rpc.md addendum C1). The time cost still needs hardware |
| §8 Metal free-memory semantics | (a), (d) | Already answered in llama_rpc.md §5 (`recommendedMaxWorkingSetSize` minus the process's own `currentAllocatedSize`) |
| §8 SWA KV sizing, `-sm tensor` | only for SWA models or tensor parallel | Still UNVERIFIED. Not needed for a layer-split MVP |

### P1. Backend loading in the release binaries (verified, one item inferred)

- Which builds are dynamic. `.github/workflows/release.yml`: the shared `CMAKE_ARGS` (line 38) does not set
  `GGML_BACKEND_DL`. The macOS arm64 job adds only `-DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3`
  (103-106), so the Mac binaries are **not DL builds**: backends are linked in rather than discovered at runtime. Backends register in the compiled order of `ggml-backend-reg.cpp:118-173`.
  The windows-cpu job uses `-DGGML_BACKEND_DL=ON` (915), and windows-cuda builds only `ggml-cuda.dll` with DL on
  (1242-1253). So the **Windows binaries are DL builds**.
- DL load order, `ggml_backend_load_all_from_path` (ggml-backend-reg.cpp:571-598): blas, zendnn, cann, cuda, hip, metal,
  **rpc**, sycl, vulkan, virtgpu, opencl, hexagon, musa, openvino, cpu, then `GGML_BACKEND_PATH`. This matches the static
  relative order for CUDA, RPC and CPU. The global list on a Windows CUDA host is therefore CUDA devices, then CPU, then
  the `--rpc` devices appended at parse time. The static RPC reg has 0 devices. RPC-first model placement (§1.3) is
  unchanged.
- Search path, `ggml_backend_load_best` (480-500): the compile-time `GGML_BACKEND_DIR` if defined, then the
  **executable's directory**, then the **current working directory**. The `ggml-<name>-*.dll` with the highest
  `ggml_backend_score` wins. Otherwise the base `ggml-<name>.dll` is loaded (548-560). Gateway rule: start `llama-server`
  and `ggml-rpc-server` with cwd set to their own bin directory or an empty directory. Never start them from a
  directory that holds another build's `ggml-*.dll`, because a backend from a different tag could be picked up.
- Failures are silent. `silent = true` under `NDEBUG` (571-576). INFERRED: CMake defines `NDEBUG` for the Release config,
  which is CMake's default and not llama.cpp source. When `ggml-cuda.dll` cannot load (for example `cudart64_*.dll`,
  `cublas64_*.dll` or `cublasLt64_*.dll` from the separate `cudart-...` zip are missing; release.yml:1261-1269), **no
  error is printed**. On a worker, `ggml-rpc-server` then falls back to the CPU device (`tools/rpc/rpc-server.cpp:202-208`)
  and starts normally. On the host, `--list-devices` simply omits `CUDA0`. The gateway should catch this by checking
  that a Windows/NVIDIA node's probe or the worker's `Devices:` startup block names a CUDA device, and by flagging an RPC
  device whose total equals system RAM.
- Successful loads log `load_backend: loaded %s backend from %s` (259) at GGML INFO. In llama-server, INFO maps to TRACE
  (4) and is hidden at the default verbosity (§6). In `ggml-rpc-server` the default ggml logger prints it to stderr. This
  answers llama_server.md §6's UNVERIFIED note about stderr noise from `ggml_backend_load_all()`.
- `--rpc` works in DL builds. `llama_supports_rpc()` loads all backends if the registry is empty and then checks
  `ggml_backend_reg_by_name("RPC")` (src/llama.cpp:110-115). `add_rpc_devices` calls `ggml_backend_load_all()` again
  (common/arg.cpp:1174). Repeated loads are harmless because `register_backend` de-duplicates by reg pointer
  (ggml-backend-reg.cpp:186-205). If `ggml-rpc.dll` is missing from the exe directory and the cwd, the error is
  `RPC not supported in this build` (P3).
- `ggml_backend_register(nullptr)` is a no-op (`if (!reg) return;`, 187-189). An endpoint that reports 0 devices
  (ggml-rpc.cpp:3161-3163 returns `nullptr`) therefore disappears from the device list without any error. In practice
  `ggml-rpc-server` refuses to start with no devices (llama_rpc.md §5).

### P2. Config files are applied before env and CLI (verified)

`common_params_parse_ex` calls `common_params_apply_system_config` first (common/arg.cpp:775; definition 721-766, which also applies the file's default preset section at 761-764):

```cpp
// apply config files (if present), a later file overrides an earlier one:
// 1. system-wide: /etc/llama.cpp/config.ini (%PROGRAMDATA%\llama.cpp\config.ini on windows)
// 2. user-level: ${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini (%APPDATA%\llama.cpp\config.ini on windows)
```

- Every key runs the normal option handler (`common_preset::apply_to_params`, common/preset.cpp:142-167), and unknown
  keys are ignored (`ignore_unknown_keys = true`, arg.cpp:755). The user-level directory comes from
  `fs_get_config_directory` (common/common.cpp:1010-1024). Each file found is logged at INFO, which **is** visible by
  default: `using config file: <path>` (arg.cpp:757).
- grep of arg.cpp and common.cpp at this ref finds no opt-out flag. The CLI overrides scalar options, but it cannot undo
  side-effect handlers. For example, a config key `rpc = ...` would register and contact extra RPC devices before the
  gateway's own `--rpc`, and they would then appear in the default device list.
- Gateway rule: launch children with `APPDATA` and `PROGRAMDATA` (Windows) or `XDG_CONFIG_HOME` (POSIX) pointing at an
  empty directory that Hypercompute owns. Treat any `using config file:` line as a launch warning.
  `/etc/llama.cpp/config.ini` on POSIX cannot be redirected through env, so detect it from that log line.

### P3. Parse-time failure text and exit codes (verified)

- An exception from an option handler is wrapped as
  `error while handling argument "<arg>": <msg>\n\nusage:\n...` (arg.cpp:874-879). `common_params_parse` prints it to
  stderr (1313-1316), and llama-server returns **1** (tools/server/server.cpp:113-115).
- Messages relevant to (a): `RPC not supported in this build` (2687-2689), `failed to find RPC backend` (1175-1177),
  `no RPC servers specified` (1171-1172), `invalid device: <name>` (1134-1135), `no devices specified` (1125-1126).
- These do not exit with 1. An unreachable endpoint and a failed HELLO call `GGML_ABORT`, which aborts the process
  (llama_rpc.md §3). `-ngl` on a build with no GPU and no RPC prints three `warning: no usable GPU found ...` lines to
  stderr (2812-2816) and continues.

### P4. `--log-jsonl` record shape (verified, common/log.cpp)

```cpp
// common/log.cpp:114-119
common_json obj = {
    {"type",  "log"},
    {"time",  timestamp},          // µs since logger start when timestamps are on, else 0 (:290-293)
    {"level", level_str(level)},   // "debug" | "info" | "warn" | "error" | "cont" | "none" (:66-75)
    {"msg",   msg.data()},
};
```

- `level` is the original ggml level. A libllama placement line such as `load_tensors: offloaded 33/33 layers to GPU`
  arrives as `"level":"info"`, even though it was filtered as TRACE (4). The filter maps INFO to TRACE
  (`common_log_get_verbosity`, log.cpp:548-559), but the entry keeps its own level (`entry.level = level`, 286).
  So `-lv 4` is still required. `msg` keeps the trailing `\n`.
- Records that are already JSON (`is_json`, for example `fit_memory_breakdown`) are printed as-is (107-112).
- With `--log-jsonl`, every record goes to **stdout** (§6). Do not pass it to a `--list-devices` probe, because WARN and
  ERROR records would mix with the `Available devices:` text.
