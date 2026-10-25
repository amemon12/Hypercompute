# llama.cpp, ggml and adjacent runtimes: source audit for a hybrid capacity engine

Audit: 2026-10-22 UTC (2026-10-21 Pacific). Read-only source research; no build, model reload, host restart, benchmark, or runtime modification was performed for this report. “Implemented” below means present in the audited source, not validated on our Windows CUDA + Mac Metal pair. Proposed patches are explicitly marked.

## Decision

Keep **GGUF + llama.cpp/ggml as the first execution engine**, and put an adaptive, measured placement controller above it. Add bounded weight movement and disk execution as explicit engine work. Do not equate an admission planner that counts SSD bytes with a runtime that can safely execute beyond RAM. Do not replace the working resident configuration with `--load-mode none` for an oversized CPU allocation.

The closest reusable ideas are:

1. llama.cpp's existing CUDA/Metal kernels, GGUF loading, remote ggml graphs, per-device layer assignment, and new MoE expert cache.
2. PRIMA's model of CPU/GPU/disk costs and repeating layer windows that overlap upcoming I/O with another machine's computation.
3. FastLLM's explicit disk operations and expert caches; ik_llama.cpp's deferred tensor residency and range prefetch; PowerInfer's measured hot/cold placement.
4. A Hypercompute controller that profiles actual devices and links, honors reserves, compares candidate plans, and changes placement at a safe lifecycle boundary. Arbitrary live migration or arbitrary mixed-engine tensors is not supplied by these components.

Adding Macs can make a disk-bound model much faster **when their resident memory replaces repeatedly read weights**. Layer splitting by itself does not make all machines compute the same next token simultaneously. Our current 32B proof is resident hybrid execution, not a verified 72B disk-capacity implementation.

## Sources and revision boundary

| Repository | Audited revision | License in repository | Relevance |
|---|---|---|---|
| ggml-org/llama.cpp, current master | `c7fc5094179bf8a9c73c18593db714f8da41a402` | [MIT](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/LICENSE#L1-L21) | Current loader, backend, RPC, MoE cache |
| llama.cpp, deployed b11770 | `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb` | [MIT](https://github.com/ggml-org/llama.cpp/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/LICENSE#L1-L21) | Actual binary used by this session |
| ikawrakow/ik_llama.cpp | `15e7ecacddd87455f3fbc4805bc0508df01fa8cc` | [MIT](https://github.com/ikawrakow/ik_llama.cpp/blob/15e7ecacddd87455f3fbc4805bc0508df01fa8cc/LICENSE#L1-L24) | Deferred sparse weights, CPU kernels, prefetch |
| ztxz16/fastllm | `76d4bfd89ad30e30c6faf4a36435fb3d71547738` | [Apache-2.0](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/LICENSE#L1-L20) | Disk operators and expert-cache implementation |
| Tiiny-AI/PowerInfer | `8bd56d69906c9d2dba4d3bf6899763401e01a9a4` | [MIT](https://github.com/Tiiny-AI/PowerInfer/blob/8bd56d69906c9d2dba4d3bf6899763401e01a9a4/LICENSE#L1-L21) | Profile-guided sparse placement; SmallThinker disk buffers |
| OpenCPIL/prima.cpp | `288df6a2cc2120387e12d6f40798f30cc525c9ea` | [MIT](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/LICENSE#L1-L21) | Distributed out-of-core scheduling |

These are separate implementations and revisions. Current-master features are not an instruction to upgrade the deployed binary. Preserve license notices when taking code; model licenses and dependency licenses are separate from the repository's top-level license. PowerInfer's former SJTU-IPADS address redirects to Tiiny-AI; PRIMA's current source is OpenCPIL/prima.cpp. The older PRIMA URL should not be used to pin implementation evidence.

## 1. What the llama.cpp execution path actually does

The useful architectural units are:

```text
GGUF metadata and tensor offsets
  → model architecture and placement choices
  → CPU mappings / CUDA, Metal, RPC backend buffers
  → context-owned KV and compute allocations
  → ggml graph split by backend
  → local execution or serialized remote graph + activation copies
```

This is a single graph ecosystem. GGUF tensor type, quantization, dimensions, operations, and backend support matter at every boundary.

### Layer assignment is constrained, but controllable

For the ordinary layer split, the model loader chooses a CPU prefix followed by GPU devices in order. It uses free-memory proportions when no split is specified, or cumulative normalized `tensor_split` values otherwise. `n_gpu_layers` counts the output unit as well as repeating blocks; input embeddings stay on the CPU. See [the actual mapping and output assignment](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model.cpp#L1641-L1699).

Consequences for an optimizing overlay:

- Search explicit integer boundaries, then convert them to the runtime's device order and proportions. Verify the resulting load log; float proportions and the output unit can otherwise create an off-by-one placement.
- Free memory is a capacity input, not a speed model. The default does not price CPU speed, link RTT, relay throughput, SSD misses, or a contributor's application reserve.
- Ordinary contiguous layer placement cannot express an arbitrary alternating CPU/Mac/CUDA layout. PRIMA's repeated windows are a different execution scheme.
- Placement is established while constructing/loading the model; changing the CLI proportions does not move a running model or its KV cache. An overlay should initially replan between deployments/requests and explicitly manage chat-state replay.

Current CLI offers `none`, `layer`, `row`, and experimental `tensor` split modes; therefore it is inaccurate to say llama.cpp only implements layer parallelism. [CLI definitions](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/common/arg.cpp#L2820-L2868). Tensor mode additionally rejects unsupported model architectures, and row mode asks a backend for a split buffer type. Their existence does not establish efficient CUDA-to-remote-Metal tensor parallelism. [Architecture guard](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model.cpp#L370-L385), [row buffer selection](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model.cpp#L1203-L1225).

### “Pipelined” does not mean concurrent layers of one dependent token

The ggml scheduler walks graph splits, handles input copies/synchronization, and then launches each split. Dependencies still require a later transformer stage to wait for the earlier stage's output. [Split execution](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-backend.cpp#L1865-L1930).

Current llama.cpp only enables its additional pipeline scheduling when all of these apply: multiple devices, more GPU units than repeating layers, layer split, KQV offload, no tensor overrides, no MoE cache, and async/event support from participating accelerator backends. [Exact conditions](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-context.cpp#L525-L558). A CPU-prefix deployment like the current 32B configuration does not satisfy the full-offload condition. Pipeline overlap can benefit suitable independent microbatches/work, but cannot remove the autoregressive dependency between successive tokens in one chat.

**Practical choice:** start the overlay with contiguous layer plans and measured end-to-end costs. Treat row/tensor modes, PRIMA-style windows, and simultaneous multi-request throughput as distinct experimental candidates.

## 2. Windows memory loading: what is and is not bounded

### Load modes and mappings

Current CLI exposes `auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, and `dio`; `--lazy-mode` separately handles certain architecture-marked tensors, such as large per-layer embeddings. It is not a general dense-layer pager. [Load and lazy options](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/common/arg.cpp#L2694-L2725).

The loader chooses full mapping prefetch when `prefetch && use_mmap`. If a CPU tensor can use the mapping, it points into that mapping; otherwise mapped bytes are copied to the target backend. The nonmapped CPU branch reads into an already allocated host tensor. [Mapping setup](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model-loader.cpp#L1418-L1452), [mapped versus allocated tensor branches](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model-loader.cpp#L1704-L1749). The model load call enables prefetch. [Call site](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model.cpp#L1887-L1905).

On Windows, `CreateFileMapping`/mapping plus `PrefetchVirtualMemory` operate over read-only file-backed regions. The implementation's partial unmap operation is currently a no-op on Windows. [Windows implementation](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-mmap.cpp#L589-L641). The deployed b11770 has the same relevant prefetch call and no-op fragment unmap at [its pinned lines](https://github.com/ggml-org/llama.cpp/blob/39da130fcd1b6ca47d3c0522a40eccff41ea3dfb/src/llama-mmap.cpp#L589-L641).

**Correct cache semantics:** Microsoft documents `PrefetchVirtualMemory` as fetching pages into physical memory without adding them to the process working set. It is a cache/performance hint, not a guarantee that all weight bytes become private resident allocations. Those pages are reclaimable; prefetch can still compete for physical memory and I/O. Disabling it does not impose a hard memory limit and does not by itself prove the cause of the earlier stalled load. [Microsoft API documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory).

The successful resident flags `--load-mode none --no-host --no-repack` address a different situation: CPU weights must fit their fully accounted allocation budget. Ordinary OS paging of private allocations is not an engineered file-backed capacity tier. `--no-host` controls host buffer selection; `--no-repack` removes repacking. Neither is a model-size cap. [Flag definitions](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/common/arg.cpp#L2427-L2441).

### Existing bounded staging is useful but limited

The nonmmap upload path creates four staging buffers, normally 1 MiB each, and uses backend async/event capabilities. It is disabled for mmap or tensor-validation loads. [Staging setup](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-model-loader.cpp#L1576-L1603). This is a useful pattern for bounded I/O, but does not bound final CPU/GPU weight allocations, all RPC serialization buffers, KV, compute buffers, or OS cache.

A truthful memory budget needs at least:

```text
resident CPU weights + KV + compute/workspace + staging + pending RPC bytes
+ runtime/allocator margin + contributor reserve
```

Account CUDA VRAM separately. Count Apple unified memory once, including Metal allocations and CPU use. Track process working set, private commitment, available physical memory, page-file activity, and disk reads separately: a large private-commit number is not itself evidence of the same amount of physical RAM residency.

### Smallest candidate patch, explicitly unvalidated

Expose a loader option that passes `false` to `init_mappings` for a capacity experiment, leaving CPU weights file-backed, logging the effective mode and avoiding mlock/repacking that forces additional residency:

```cpp
// Proposed interface, not upstream code or an implemented patch.
bool eager_mmap_prefetch = params.capacity_prefetch == PrefetchPolicy::Eager;
loader.init_mappings(eager_mmap_prefetch, locks);
```

This can isolate startup prefetch behavior on a small model. **It is not the final capacity implementation.** A strict runtime needs explicit bounded buffers or a designed pager, plus bounded transport and graph lifetimes. Windows working-set ceilings can be used in a controlled test, but are not a cap on the machine's file cache or total process commitment. [Working-set API semantics](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex).

## 3. RPC: resident remote buffers, cached uploads, graph reuse

RPC extends ggml devices. Serialized tensors carry operation/type/shape/stride and buffer/data references; commands allocate buffers, set/get/copy tensors, compute/recompute graphs, and support communication primitives. [Wire structures/commands](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L36-L82).

### Weight cache does not add inference capacity

For large weight writes, the client computes an FNV hash and asks the server whether the cache has it; otherwise it serializes and sends the bytes. The threshold is 10 MiB and eligibility excludes compute activations. [Threshold](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L117-L122), [weight transfer](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L829-L872).

A cache hit reads the whole cached file into a vector and then copies it into the allocated backend tensor. The target remote allocation still exists. Therefore `rpc-server -c` saves repeat network uploads, **not remote VRAM/RAM residency during inference**. [Cache reload](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L1818-L1881).

Capacity and latency risks visible in code:

- Set-tensor serialization copies the entire supplied payload into another contiguous allocation.
- The receiving vector is resized to the message's declared length before command-specific processing. [Receive allocation](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L344-L355).
- Disk-cache reload introduces another whole-tensor vector.
- Synchronous dispatcher calls wait for completion, while async calls enqueue messages; do not assume every operation has a byte-based in-flight bound. [Queue and dispatcher](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L444-L492), [sync/async difference](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L548-L578).

These explain why steady-state weight placement alone is an insufficient peak-memory model. They do not prove a particular past stall was caused by these copies.

### Graph cache is a separate optimization

The client caches graph IDs per device; a known ID uses GRAPH_RECOMPUTE, while a new graph sends the serialized graph. The client cache limit is 1,024 IDs per device. [Graph reuse](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/ggml/src/ggml-rpc/ggml-rpc.cpp#L1242-L1267). This avoids retransmitting static graph descriptions; activations and necessary synchronization remain. It does not provide a generic dynamic layer-migration API.

### Reusable changes, proposed

1. Bound upload chunks and total in-flight bytes at both ends; retain tensor offset/range checks.
2. Stream cached files through bounded buffers instead of loading an entire cached file into a vector.
3. Make a complete artifact manifest authoritative: model revision, tensor metadata, checksum, length, and an atomic completion state. FNV upload deduplication is not an adversarial integrity mechanism.
4. Preserve existing full-tensor cache semantics or negotiate a new protocol capability. Simply chopping transfers below the hash threshold can disable useful cache hits; changing the on-disk cache or protocol silently is unsafe.
5. Place a cold tensor near its intended worker before model startup when possible, and measure actual cache hits. Initial upload time and warm decode performance must be reported separately.

Upstream still explicitly labels RPC fragile and insecure and says not to expose it to an open network. [Upstream warning](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/tools/rpc/README.md#L1-L5). Authenticated Iroh and allowlisted services reduce exposure; they do not turn arbitrary untrusted ggml graphs into a safe public worker interface. Keep owner HTTP routes separate from the peer transport and retain contributor quotas.

## 4. A useful upstream addition: bounded GPU MoE expert cache

Current master already includes `--moe-cache-mib`, used with host-resident experts. The implementation groups expert tensors by assigned accelerator, divides the budget across devices/groups, allocates an integer number of slots, and rejects a cache too small for even one token's selected experts. [Option](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/common/arg.cpp#L2763-L2789), [host-weight eligibility](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-moe-cache.cpp#L179-L300), [budget/slot allocation](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-moe-cache.cpp#L344-L366).

At runtime it reads routing IDs, synchronizes, plans SLRU fills, uploads missing experts, and records hits/misses/bytes. [Runtime copy and counters](https://github.com/ggml-org/llama.cpp/blob/c7fc5094179bf8a9c73c18593db714f8da41a402/src/llama-moe-cache.cpp#L570-L614). This is a strong reference for accounting and observability. It still retains the host expert tensors and disables the additional pipeline scheduling described above. It is not a dense Qwen2.5-32B/72B layer cache and is not verified here over RPC/Metal. Do not promise the same benefit for a model whose dense weights are all used on every token.

## 5. Relevant alternatives: concrete reuse and incompatibilities

### PRIMA.cpp: most relevant distributed disk scheduling reference

PRIMA implements repeating per-rank layer windows. `this_layer_is_mine` cycles ranks and accumulates window sizes; local IDs include the cycle offset. This can assign several separated layer groups to the same device, unlike llama.cpp's ordinary single contiguous interval. [Window mapping](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/src/llama.cpp#L3838-L3882).

Its decode loop receives embeddings, computes a local subgraph, copies the result to host memory, synchronizes, sends it onward using ZeroMQ, then prefetches the next local subgraph while other ranks compute. [Actual execution/overlap](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/src/llama.cpp#L18503-L18575). The memory-management helper merges tensor ranges, calls `posix_madvise`, and optionally touches pages to force reads. [Prefetch implementation](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/src/llama.cpp#L18168-L18217). This is OS-backed paging with hints/optional page touching, not an intrinsic strict memory quota.

The scheduler uses device/model profiles, CPU/GPU and memory/disk costs, and integer variables for window/GPU-layer counts. It searches valid cycle counts and uses HiGHS to minimize its cost model, then may remove devices that receive at most one layer and solve again. [Variables/model constants](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L860-L887), [integer objective](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L1248-L1290), [solver](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L1379-L1419), [device pruning](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L1595-L1640).

**Important restrictions for our overlay:**

- The code assumes host/device transfers and communication are negligible and sets their cost term to zero. That assumption is unsuitable for our measured Iroh relay. [Explicit assumption](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L969-L971).
- The scheduler asserts that Windows is untested. The inference prefetch helper uses POSIX APIs. This is not a ready replacement for the Windows host. [Windows assertion](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L980-L984).
- Automatic profiling/scheduling happens in initialization; custom window sizes skip that path. The source inspected does not establish a live model/KV migration loop driven by changing background load. [Initialization and layer-plan broadcast](https://github.com/OpenCPIL/prima.cpp/blob/288df6a2cc2120387e12d6f40798f30cc525c9ea/common/common.cpp#L1717-L1803).
- Repeated windows create additional activation hops per token. Smaller windows may improve I/O overlap but hurt relay latency. Search the tradeoff; do not port a zero-network-cost objective unchanged.
- The project README's low-pressure and speed claims are measurements/claims for its configurations, not verified guarantees for this pair.

**Reuse first:** its per-stage cost decomposition, window ownership mapping, and “prefetch the next work while a remote stage runs” strategy. A Windows implementation, measured network term, explicit byte budgets, authenticated peer transport, and state-safe replanning are additional engineering.

### ik_llama.cpp: selective residency and prefetch, not a Windows dense pager

The loader explicitly distinguishes deferred expert and sparse per-layer-embedding data; mapping setup skips bulk prefetch for files that contain deferred PLE data. [Deferral decisions](https://github.com/ikawrakow/ik_llama.cpp/blob/15e7ecacddd87455f3fbc4805bc0508df01fa8cc/src/llama-model-loader.cpp#L672-L732), [mapping policy](https://github.com/ikawrakow/ik_llama.cpp/blob/15e7ecacddd87455f3fbc4805bc0508df01fa8cc/src/llama-model-loader.cpp#L1118-L1142). This is useful evidence for tensor-class-specific policies rather than one global “everything resident” switch.

The POSIX path implements range advice including dropping and prefetching pages. Windows implements range prefetch, but `dontneed_fragment`, random-access advice, and partial unmap are no-ops in the audited implementation. [Windows methods](https://github.com/ikawrakow/ik_llama.cpp/blob/15e7ecacddd87455f3fbc4805bc0508df01fa8cc/src/llama-mmap.cpp#L538-L582). Reusing the API names would not provide equivalent Windows eviction semantics.

**Reuse:** explicit sparse weight classification, bounded range-prefetch interfaces, and compatible quantized CPU kernels after compatibility/quality checks. **Do not assume:** expert-specific optimization accelerates a dense model, or the same pager behaves equivalently across operating systems.

### FastLLM: explicit disk operators and GPU expert-cache mechanics

FastLLM's disk device registers linear, embedding, and MoE operations. Its disk expert path can use direct I/O when a CPU/CUDA cache budget is present, attempting to avoid duplicating a large expert cache in the OS page cache. [Registered operations](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/src/devices/disk/diskdevice.cpp#L73-L96), [I/O policy](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/src/devices/disk/diskdevice.cpp#L153-L215). A small temporary buffer pool retains at most four buffers of up to 16 MiB each; this is workspace pooling, not a cap on all disk/backend memory. [Pool](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/src/devices/disk/diskdevice.cpp#L572-L608).

The CUDA expert-cache implementation maintains GPU-resident slot/recency/index metadata and hit/miss counts; query deduplication precedes cache placement. [Metadata](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/include/devices/cuda/fastllm-cuda-expert-cache.cuh#L10-L27), [lookup/selection kernel](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/include/devices/cuda/fastllm-cuda-expert-cache.cuh#L69-L125). Disk prefetch has its own two-worker queue. [Prefetch worker implementation](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/src/devices/disk/diskdevice.cpp#L2038-L2077). Queue bytes must also be bounded in a borrowed design.

The audited disk file includes POSIX mapping/I/O headers and uses `pread`/direct I/O; this needs a Windows implementation rather than an assumption that generic Windows packaging makes this exact path portable. [Platform includes](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/src/devices/disk/diskdevice.cpp#L1-L47). A compatible Metal and ggml-RPC bridge was not established by this audit. Its disk-MoE tests are a useful source of expected behavior, not proof of our deployment compatibility. [Tests](https://github.com/ztxz16/fastllm/blob/76d4bfd89ad30e30c6faf4a36435fb3d71547738/test/basic/test_disk_moe_cache.cpp#L1-L80).

**Reuse:** explicit tier dispatch, aligned bounded reads, hit/miss instrumentation, and eviction rules. Porting whole runtime/model formats is a larger undertaking than adopting those ideas within ggml.

### PowerInfer: learned sparsity and buffered I/O, model-specific

The original solver consumes per-layer activation profiles, sorts neuron importance, and solves a constrained placement problem. [Placement solver](https://github.com/Tiiny-AI/PowerInfer/blob/8bd56d69906c9d2dba4d3bf6899763401e01a9a4/powerinfer-py/powerinfer/solver.py#L20-L74). That is useful for a hot/cold policy, but requires compatible sparsity and profiling; it does not make arbitrary dense weights unnecessary. The original monolithic model architecture list does not include Qwen2. [Architecture list](https://github.com/Tiiny-AI/PowerInfer/blob/8bd56d69906c9d2dba4d3bf6899763401e01a9a4/llama.cpp#L227-L260).

The repository now also contains SmallThinker code. Its disk-buffer helper starts with two 32 MiB aligned buffers and grows a buffer to accommodate a larger requested read. [Buffers](https://github.com/Tiiny-AI/PowerInfer/blob/8bd56d69906c9d2dba4d3bf6899763401e01a9a4/smallthinker/powerinfer/src/disk_buffer.hpp#L12-L55). The two-buffer pattern is reusable, but its initial allocation size must not be described as a hard maximum. A Windows aligned-allocation branch alone is not evidence that the complete SmallThinker pipeline supports this Windows/Metal distributed setup.

## 6. Proposed optimizing overlay

This section is a design proposal, not implemented upstream behavior.

### Inputs and candidate plans

Profile the actual quantization/model shape, context size, prompt batch size, and backend combination. Collect:

- free/available RAM and VRAM with timestamp, reserve, committed leases, and reliability margin;
- warm and cold per-layer CPU/CUDA/Metal cost, KV and workspace size;
- I/O throughput and latency under a bounded read test; cache-hit ratio and bytes read per token;
- link RTT, goodput, jitter, direct/relay path, and current weight-cache availability;
- startup transfer and reconfiguration cost separately from warm generation.

Start with safe contiguous CPU-prefix + selected GPU intervals, including an option to omit a slow peer. Treat repeating windows as a later runtime capability, not a new set of proportions for the existing CLI.

A rough single-stream objective is:

```text
decode_latency ≈ sum(resident stage compute)
               + nonoverlapped disk miss time
               + sum(activation-hop RTT + activation_bytes / measured_goodput)

reconfiguration_cost ≈ uncached weight bytes / measured upload goodput
                     + load/allocation work + KV migration or prompt replay
```

Use measured overlap rather than blindly adding or subtracting theoretical peak bandwidth. Dense models may require reading most nonresident weights for each generated token. Additional Macs improve speed if avoided disk/CPU work exceeds their compute and added communication cost; more machines are not automatically faster.

### Reprofiling and safe actuation

Use a rolling window and hysteresis, not a new split on every noisy heartbeat. An example policy is to consider replanning after sustained memory pressure or material throughput change, then switch only if expected savings across the next useful work horizon exceed migration/replay cost plus a margin. Enforce cooldowns and a maximum number of retries.

For the initial implementation:

1. Record the active plan, observations, and reason for considering a replacement.
2. Validate capacities again immediately before creating the replacement.
3. Drain active inference; persist chat tokens or a supported compatible state artifact.
4. Release the old lease/model before loading another if both cannot fit simultaneously.
5. Load the new plan, verify actual buffers/placement, replay as needed, run a bounded health probe, then expose it.
6. On failure, retain a clear failure state and choose another feasible plan; do not repeatedly OOM or silently violate contributor reserves.

Do not assume arbitrary live KV migration: backend allocations, graph references and model ownership depend on placement. An engine-level migration protocol would require explicit tensor identity, sequence state, graph invalidation and completion barriers.

### A genuine bounded disk engine

Illustrative original pseudocode:

```python
for stage in execution_schedule:
    needed = manifest.byte_ranges_for(stage, selected_experts)
    lease = weight_cache.reserve(needed, max_inflight_bytes=budget)
    # Admission counts cache slots, staging, KV, compute and transport together.
    read_future = reader.read_into(lease.aligned_buffers, needed)
    read_future.wait()
    completion = backend.execute(stage, lease.buffers, kv_for_stage)
    prefetch_if_budget_remains(next_stage_ranges)
    completion.wait()          # CUDA event / Metal completion / remote acknowledgement
    weight_cache.unpin(lease)   # never evict memory referenced by unfinished work
```

The difficult integration is graph execution and tensor lifetimes, not just a file-read loop. A layer may need multiple tensors simultaneously, repacking may require another representation, a backend may retain pointers, and activation/KV dependencies cross stages. Chunk size and read-ahead must be charged against the same budget; a two-buffer label is insufficient if either buffer can grow without limit.

## 7. Validation with existing small models, before a larger download

These tests are proposed; none were run during this audit.

1. **Mapping A/B:** use the existing small GGUF with identical tokens, context and placement. Compare current mmap, candidate no-eager-prefetch mmap, and fully accounted resident loading. Record startup time, disk reads, working set, private commit, available RAM, hard faults, TTFT and decode time separately.
2. **Actual capacity test:** deliberately lower the child execution budget so the chosen model exceeds it. A planner-only fake cap tests routing but proves no bounded execution. Use a controlled child working-set ceiling and memory instrumentation, or a separately constrained Linux environment, without applying global memory pressure or flushing system-wide caches.
3. **Bounded transfer test:** use a tensor larger than the staging cap and a throttled link. Verify peak queued/staging bytes, checksum, cancellation, incomplete cache handling and reconnect behavior. Test cache hit and miss independently.
4. **Pager correctness:** compare logits/token output with the resident baseline; exercise eviction while kernels are in flight, oversized single tensors, full cache, allocation failure and cancellation. Use tolerance appropriate to the same quantized backend.
5. **Physical topology comparison:** identical prompts/context on one host, host+one Mac, and host+three Macs. Separate short-prompt decode, long-prompt prefill, initial uploads and warm cached loads. Show observed peak memory and every worker's actual model/KV buffers.
6. **Adaptive controller:** replay deterministic changing-capacity traces first. Verify reserve preservation, no oscillation, migration cost accounting, slow-peer exclusion, missing-worker cleanup and a coherent user-visible state.

The existing physical 32B evidence demonstrates real CUDA/CPU/Metal participation and correct answers under a resident plan. It does not demonstrate disk-overflow execution, dynamic migration, or a speedup from two additional Macs. Those claims need the tests above.

## Scope and unresolved questions

- No current alternative was established as a drop-in, bounded, secure Windows CUDA + Mac Metal disk-cluster engine.
- The strongest immediate improvements are placement economics, cache-aware startup, bounded transport, and a separately tested loader/pager path.
- Combining AirLLM, EXO and llama.cpp at the orchestration level is feasible in principle; cross-engine activation transfer needs an explicit architecture/tokenizer/quantization/KV contract. Their model objects and weight formats cannot be assumed interchangeable.
- The pinned b11770 and current-master audit should remain separate in implementation decisions. Every proposed cherry-pick needs compatibility, license, correctness and memory tests at the exact chosen revision.

