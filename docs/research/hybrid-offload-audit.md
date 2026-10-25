# Hybrid offload audit: AirLLM, Accelerate, FlexLLMGen, KTransformers

Research snapshot: 2026-10-22 UTC / 2026-10-21 Pacific. Hypercompute checkout inspected at `f21c42f9496d4cef631947b2c3beb1aa7a58ea2d`. This is a source audit and proposed interface, not an implementation or a new performance benchmark. No models, packages, or servers were started for this audit.

**Finding:** the components offer real, useful mechanisms for running weights outside accelerator memory. They do not together implement one distributed model. Hypercompute needs a shared placement controller plus a compatible execution backend that can retain, load, and release weight ranges. Adding Macs can remove repeated paging only after their memory is assigned useful weights and the executing model uses those assignments.

## Evidence and revisions

- **Source-confirmed** means the linked code was read at the exact revision.
- **Upstream claim** means a project's documentation says it; this audit did not reproduce it.
- **Inference/proposal** is engineering analysis, including all proposed interfaces and forecasts.
- **Local observation** is an already completed Hypercompute experiment, explicitly scoped below.

| Project | Audited revision and current status | License |
|---|---|---|
| AirLLM | Hypercompute pin `bc47e0d474abf03f491f7547085bd85a2c6f0b1b`; current head `22d72c46a67c9bf91da0615831fbb619e622c158`, October 21. The [comparison][airdiff] changes only two star-history images, so the audited runtime matches current head. | [Apache-2.0][airlicense] |
| Hugging Face Accelerate | Current main `abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8`, October 20. The latest change isolates compiler disabling to onload/offload hooks. This is an upstream source snapshot, not a claim that the installed 1.15.0 wheel has identical code. | [Apache-2.0][acclicense] |
| FlexLLMGen / FlexGen | `004ffef82b46e8dc8685c55d0cdda650bdaf1269`, October 28, 2024. Repository is archived; latest commit is the compliance rename. These names refer to the same project family, not two interoperable engines. | [Apache-2.0][flexlicense] |
| KTransformers | Current main `a694e718e8b3ec2e221277d674e369f3be38d2a4`, October 19. Current `kt-kernel` / SGLang path and the older `archive/` implementation must be distinguished. Latest commit adds an opt-in Intel SYCL expert backend. | [Apache-2.0][ktlicense] |

License entries cover repository code, not model weights, all transitive dependencies, or trademark rights. A reuse implementation must retain required notices and audit the particular copied dependencies.

## 1. AirLLM: actual layer streaming, with limited explicit retention

**Source-confirmed execution.** The PyTorch base creates a model on the meta device, attaches hooks to embedding, decoder layers, norm and head, and delegates generation to Transformers. Before a streamed module executes, its state is materialized on one `running_device`; after execution it is returned to meta. `generate()` does not provide a distributed execution loop. See [construction][airinit], [movement][airmove], and [hook lifecycle][airhooks].

**Prefetch and memory.** One background thread reads the next streamed module. The current module waits for its future when applicable; the future is not a general multi-layer cache, and the final layer does not prefetch the next token's first layer. CUDA prefetch can make an additional pinned CPU copy when a layer is at most 2 GiB. The current tensor, pinned copy, next module and device copy can overlap in lifetime; bounding only final GPU weights misses staging pressure. Compression disables prefetch. See [executor setup][airinit], [pinned copies][airpin], and [next-layer loading][airhooks].

**Retention exceptions matter.** Tied input/output embeddings stay resident and are re-tied; explicitly named auxiliary modules also stay resident. Large architecture-specific CPU embedding tables may use mmap. Thus “nothing stays resident” is incorrect. Ordinary dense decoder layers, however, have no configurable keep-the-hottest-N-layers policy in this implementation. KV state belongs to the underlying model generation/cache implementation; the weight post-hook does not itself offload KV. It must be budgeted independently. See [resident modules and tied weights][airresident].

**Cold versus warm.** The default persister calls safetensors `load_file(..., device="cpu")` for each load; the dense hooks repeat this logical file load on each forward. That is not proof that each byte reaches the SSD every token: file mappings and the OS page cache may satisfy reads. Evicting a PyTorch parameter or calling `torch.cuda.empty_cache()` is not an OS file-cache flush. With a working set larger than spare host memory, repeated dense sweeps can cause real disk reads again; this is an inference to measure with physical disk counters. See [persister][airpersist] and [cleanup][airclean]. Earlier local notes saying “every layer is read from disk every token” should be interpreted as repeated file-backed loading, not measured SSD traffic.

**Dense versus MoE.** A source-supported expert mode installs individual expert hooks when an architecture supplies `expert_prefix`, storage is safetensors, and compression is disabled. It reads selected expert tensors and releases them after use. Dense Qwen2.5-72B has no sparse expert routing benefit: all decoder layers participate in each token. This expert path also does not automatically make fused/packed expert architectures compatible. See [expert hooks][airexperts] and [subset reads][airsubset].

**Quantization boundary.** The code handles several HF quantizer and packed-tensor cases; its own compression is a different storage/decompression path. Neither implies direct execution of the existing GGUF Q3/Q4 model through PyTorch. Hypercompute currently launches AirLLM with `compression=None` and `prefetching=True` and uses HF artifacts. A compressed on-disk tensor can still expand during execution and require quantization metadata plus temporary storage. [Tensor materialization][airmove] preserves recognized prequantized payloads and delegates some reconstruction to the HF quantizer.

**First-start cost is separate from token cost.** The splitter builds per-layer files from the checkpoint index; completed output is reused on subsequent starts. Hypercompute preserves original artifacts. With no compression, retaining source and split tensors can approach twice the weight storage, apart from special links/layouts and metadata. For scale only, 72 billion BF16 parameters are approximately 144 GB of payload before a second split store; this is arithmetic, not the size of an audited Qwen72 artifact. Switching a downloaded quantized GGUF deployment to HF AirLLM can therefore increase disk requirements dramatically. [Split implementation][airsplit]

**Mac boundary.** Upstream's normal `AutoModel` route explicitly selects a separate MLX implementation on macOS and rejects architectures other than `LlamaForCausalLM` and `MistralForCausalLM`. Its MLX loop loads one block, evaluates it, preserves that layer's KV, then deletes the block. This is actual streaming, but it is not Qwen72 support and it is not EXO integration. The generic base's device utility also mentions MPS; that does not override the normal factory restriction or prove model compatibility. See [factory guard][airauto] and [MLX decode loop][airmlx].

**Best reuse:** the hook lifecycle and one-ahead prefetch as a prototype, extended with explicit retention decisions, cancellation-safe leases, and memory accounting. Do not assume its split-file layout, remote-code policy, executor cleanup, or optional model implementations are a universal backend.

## 2. Accelerate: a useful local residency contract, not an elastic cluster

Three distinct APIs avoid an important conflation:

- `cpu_offload` keeps a backing CPU state and materializes parameters for execution as needed.
- `cpu_offload_with_hook` retains the model on the execution device until an explicit offload call, optionally chaining the previous model's hook.
- `disk_offload` uses file-backed weights and hooks; `dispatch_model` accepts a static local device map containing accelerator devices, CPU, and disk. These are complementary local policies. [API definitions][accapis]

**Actual hook boundary.** `AlignDevicesHook.init_hook` leaves non-offloaded parameters resident or moves offloaded parameters to meta. `_onload_weights` obtains tensors from the backing mapping and installs them on the execution device; `_offload_weights` releases them after forward. Tied allocations are tracked and freed carefully. The examined standard path synchronously requests weights in pre-forward; it contains no next-layer prefetch scheduler. See [hook implementation][acchooks].

**File access.** `OffloadedWeightsLoader` can use existing safetensors indexes or per-tensor data files. The data-file path opens a NumPy mmap and constructs a torch tensor from it; the safetensors path retrieves a tensor and moves/casts it as required. These operations still depend on page-cache state and may allocate transient CPU storage. They are not an explicit bounded RAM cache. [Loader implementation][accload]

**Planning mechanism worth reusing.** `infer_auto_device_map` analyzes parameter sizes and dtypes without materializing the model, respects indivisible module classes, and reserves room for the largest layer that must return from CPU to GPU and from disk to CPU. This staging-space invariant is more valuable than a naive “sum weights <= sum memory” check. The inference planner is capacity-oriented; it does not model this Windows/Mac network, live peers, measured SSD throughput, or request latency. [Planner contract][accplan]

**Important semantics.** A map entry saying CPU is not necessarily CPU compute: with a GPU main device, CPU and disk entries are generally weight homes that execute on the main accelerator. The source treats CPU/MPS main devices differently. Read the map's *execution device* separately from its storage device. [Dispatch mapping][accdispatch]

**Best reuse:** an explicit `home / execution / retained` distinction, tied-tensor identities, largest-module staging reserves, and no-split groups. A custom weights mapping plus hooks can prototype a local bounded cache. It does not transport GGML buffers, MLX arrays or Transformers KV between hosts; compatible process/device APIs and model implementations remain necessary.

## 3. FlexLLMGen: scheduling ideas that trade latency for throughput

**Implemented storage policy.** Six percentages independently place weights, KV cache and hidden states on GPU or CPU, with disk as the remainder. Policy also controls pinned weights, CPU attention computation, compression and overlap. The implementation keeps separate home, read and write buffers. Weight loading occurs once for the first microbatch and can be reused by the remaining microbatches for that layer. [Policy and weight placement][flexpolicy], [weight reuse][flexreuse]

**Overlap mechanics.** Dedicated CUDA streams load weights, load KV and store KV. The single-batch loop schedules the next layer's weight/cache load before current-layer compute; the multi-batch loop also pipelines microbatch activations and cache writes. Explicit synchronization marks safe reuse points. This is actual scheduling code, not just a diagram. [Stream setup][flexstreams], [generation loops][flexloops]

**Disk backend and practical pitfall.** `TorchDisk` creates mmap-backed arrays and queues copy work to four threads by default. Each copy worker allocates `1 * GB` *elements* of pinned FP16 storage; `GB = 1 << 30`. That is 2 GiB per thread, approximately 8 GiB across four default workers, before useful model buffers. Copying those defaults into a low-memory laptop runtime would defeat the purpose. Replace them with bounded byte-sized chunks and explicit credits. Mmap reads also benefit from the OS page cache; creation/reuse of offload files is a different cost from repeatedly faulting their pages. [Disk and copy implementation][flexdisk], [copy worker][flexcopy], [unit constant][flexunits]

**Cost model.** The experimental LP separately models prompt and decode transfers, CPU/GPU compute, weights/KV/activations, and peak working memory. Its objective minimizes time divided by batch size, then searches microbatch counts for throughput. Default bandwidth/FLOP constants are calibrated for a Google Cloud T4 setup, not these laptops. The model is useful structure; its constants and optimal policy are not transferable measurements. [Cost configuration and objective][flexcost], [transfer equations][flexcostio], [batch search][flexsearch]

**Distributed support is real but bounded.** `DistOptLM` partitions OPT layers into fixed contiguous pipeline stages and sends hidden tensors with PyTorch distributed send/receive. Stages are determined at construction; this code is not device-arrival-aware placement, and it does not mix MLX or GGML operators. [Stage construction][flexdist], [activation transport][flexcomm]

**Model and platform fit.** The audited runtime explicitly builds OPT attention/MLP/embedding components and CUDA streams. Qwen2.5-72B's architecture, rotary positions, grouped-query attention, quantization representation and checkpoint names are not a drop-in OPT configuration. There is no demonstrated Metal execution route in these files, and native Windows execution was not validated here. Upstream frames the project as throughput-oriented and explicitly trades latency for batching; its published tokens/s must not be presented as one chat's decode speed. [OPT assembly][flexstreams], [workload goals][flexread]

**Best reuse:** separate storage budgets, a measured transfer graph, overlap fences, and one load reused across a batch. For the user's one-stream chat objective, optimize TTFT and inter-token latency with batch size fixed to one first. Expose batch-throughput mode separately if implemented.

## 4. KTransformers: strong MoE placement ideas, different workload and platform

The current main README points inference users toward `kt-kernel` integrated with SGLang; older implementations live in `archive/`. It would be misleading to combine an old Windows release statement with the current kernel's installation requirements and declare this laptop supported. [Current project layout][ktread]

**Scheduling and residency.** The expert factory selects CPU kernels for several quantization formats and instruction sets. A concrete helper ranks expert activation frequencies across layers and selects the top-k GPU expert mask. Current documentation describes dynamic updates after sufficiently large layerwise prefill, and a threshold choosing hybrid CPU/GPU versus layerwise GPU prefill. The mask helper is verified code; integration behavior and performance gains are upstream claims unless separately exercised. These methods benefit sparse routing, while every dense Qwen72 layer remains necessary on every token. [Factory][ktfactory], [frequency placement][ktplacement], [prefill choices][ktstrategy]

**Async boundary.** CPU expert execution has submit/sync operations and shared pinned input/output buffer management. These give useful examples of completion fences and buffer reuse, but “clear buffer cache” concerns temporary execution buffers, not a universal eviction API for model weights. [Execution and synchronization][ktexec]

**Current mmap behavior is nuanced.** The GGUF loader returns read-only views rather than eagerly duplicating each tensor. The current llamafile C++ expert loader has a single-NUMA identity-layout fast path that aliases those mappings directly and calls best-effort Linux `MADV_WILLNEED` over expert ranges. Multi-NUMA placement falls back to per-node copies/reshuffling. This is real file-backed residency and shared page-cache reuse; it is not an explicit byte-budgeted page-in/page-out policy. Do not infer that the entire model is always anonymous-RAM resident, or that an mmap guarantees the model avoids disk faults. [GGUF view][ktgguf], [alias/warm/copy paths][ktmmap]

**Platform limits.** Current prebuilt kernel documentation targets Linux x86-64 with AVX2+, with AMX/AVX512, BLIS and other specialized CPU routes, and CUDA wheels listing SM80/86/89/90. That does not establish native Windows, Apple Metal, or this Blackwell laptop's compatibility. The latest SYCL addition targets Intel, not M4 Metal. Legacy native Windows support is historical evidence only. No install or portability test was performed. [Current requirements][ktplatform], [historical announcement][ktread]

**Best reuse:** frequency-aware residency for future MoE workloads, async CPU/GPU completion contracts, and avoiding redundant mmap-to-anonymous copies when layouts already match. A dense Qwen72 plan should rank saved transfer/compute time per retained byte instead of expert frequency. KTransformers is not a ready fallback for a heterogeneous Windows+M4 dense model.

## 5. Fit to the actual Hypercompute implementation

| Existing component | What is implemented | What remains absent |
|---|---|---|
| Engine selector | Selects one engine and artifact variant; GGUF llama tiers, EXO candidate, then eligible AirLLM fallback. | A nominally feasible slow llama disk plan is not automatically replaced by AirLLM; no mixed-engine layer graph. |
| AirLLM adapter | One local CUDA/CPU assignment, HF artifact, layer-split storage, independent RAM/KV/staging checks, real worker. | Extra Macs do not add AirLLM memory or compute. No dense-layer retention cache controlled by the cluster. |
| Deployment lifecycle | Stores a plan, starts its selected adapter, stops an affected deployment on peer loss. | Node arrival does not rebalance a running deployment; loss does not transparently migrate KV and continue. |
| Existing llama/Iroh path | Authenticated transport carries llama RPC between physical Windows and Mac; an already completed resident Qwen32 run produced output. | That run is not a 72B out-of-core proof, and transport success does not implement a new paging scheduler. |

Local source anchors: [engine contract](../../backend/hypercompute/adapters/base.py) lines 1–5 and 33–46; [selector](../../backend/hypercompute/services/selector.py) lines 96–116 and 209 onward; [AirLLM planner](../../backend/hypercompute/adapters/airllm.py) lines 175, 259–329; [worker](../../backend/hypercompute/workers/airllm_worker.py) lines 573–584; [deployment planning/lifecycle](../../backend/hypercompute/services/deployments.py) lines 313–356, 401 onward and 570 onward; [node arrival](../../backend/hypercompute/services/nodes.py) line 207.

**Local observation boundary:** the earlier standalone Qwen32 experiment used one compatible llama.cpp graph, Windows CPU/CUDA plus one Mac RPC layer, and a resident-memory loading policy. It established actual cross-machine inference. This audit ran no new inference and establishes neither dense72B capacity nor a speedup from two additional Macs. Earlier AirLLM small-model validation establishes a working worker path, not large-model throughput.

## 6. Proposed bounded residency interface

This is an original design sketch for the overlay, not copied upstream code and not an implemented Hypercompute API.

A controller should choose assignments and retained ranges; each execution backend should own the actual allocations, tensor layout and fences. Use one common backend graph first. Cross-engine execution would additionally require exact hidden-state layout, normalization/rope semantics, dtype/quantization conversion, tokenizer identity and KV ownership/versioning. Sharing an OpenAI-compatible chat API is insufficient.

```python
@dataclass(frozen=True)
class WeightKey:
    model_digest: str
    revision: str
    tensor_group: str             # indivisible layer/range plus tied dependencies
    backend_abi: str
    quantization: str
    layout: str

class ResidencyBackend:
    async def reserve(self, epoch, budgets) -> Reservation: ...
    async def prefetch(self, key, destination, reservation) -> LoadTicket: ...
    async def acquire(self, ticket) -> Lease: ...
    async def execute(self, lease, activation, kv_handle) -> Completion: ...
    async def release(self, lease, after, retain: bool): ...
    async def drain_epoch(self, epoch): ...

async def run_range(plan, activation, kv):
    reservation = await backend.reserve(plan.epoch, plan.budgets)
    # Budget covers current weights + one next load + staging + KV + workspace.
    current = await backend.prefetch(plan.groups[0], plan.device, reservation)
    for index, group in enumerate(plan.groups):
        lease = await backend.acquire(current)
        next_load = None
        if index + 1 < len(plan.groups):
            next_load = await backend.prefetch(
                plan.groups[index + 1], plan.device, reservation)
        completion = await backend.execute(lease, activation, kv.for_group(group))
        activation = await completion.output()
        await backend.release(lease, after=completion, retain=group in plan.retained)
        current = next_load
```

The sketch omits error handling for readability; the following are required implementation invariants:

1. Immutable weights are keyed by content/revision and backend representation. A cache hit on the wrong quantization or layout is a correctness failure. Tied tensors share one allocation identity.
2. Budgets separately cover retained weights, current load, prefetched load, pageable staging, page-locked staging, GPU workspace and all live KV. Windows committed memory is an additional constraint for anonymous allocations; physical free memory alone is insufficient.
3. Leases are not evictable before all compute/copy fences complete. Cancellation drains or aborts work, then releases credits; it must not leave futures that retain an extra layer indefinitely.
4. An OS mmap/page-cache hit is tracked separately from an explicitly retained CPU/GPU allocation. Advertised retained capacity must be enforceable; page-cache capacity is opportunistic.
5. KV is mutable request state owned by its executing range and epoch. It cannot be treated as another immutable weight cache. A migration needs validated transfer or deliberate prompt replay.
6. Peer join/loss creates a new plan epoch. The first implementation drains at a request boundary, loads the new ranges, verifies placement, then atomically accepts new requests on that epoch. It does not rewrite a running graph's addresses.
7. Disk reads and deserialization use only the authorized verified artifact store. A remote peer supplies identity/capacity, not arbitrary local file paths or executable checkpoint code.

### Optimizing when devices arrive or disappear

For dense single-stream decode, all layers are visited once per token. Estimate each candidate range's compute time, expected physical-read time, host/device transfer and boundary activation transfer. Use measured context-dependent KV/workspace sizes. A new Mac can be beneficial by replacing repeated SSD reads with resident weights and local compute, even when its FLOP rate is lower; a slow relay and loading cost can outweigh that benefit for short requests.

A useful retained-range score is estimated saved time per token divided by additional retained bytes, constrained by contiguous placement and tied groups. Compare whole candidate plans, because an extra boundary changes transport cost. Do not rank dense layers by access frequency: that frequency is nearly uniform. Treat prefilling and decoding as separate objectives; use the user's expected prompt/output lengths to combine them.

Replan only when expected future savings exceed reload/migration cost plus a safety margin. Under ideal overlap, a streaming stage is bounded by the slowest overlapped component; without measured overlap, use a conservative sum. Never count bandwidth savings from prefetch as already achieved. FlexLLMGen supplies a useful cost-model pattern, while KTransformers supplies a sparse-expert placement pattern; neither supplies a device-arrival policy for this network.

## 7. Bounded next implementation and acceptance criteria

**Decision proposal:** use the existing compatible llama.cpp Windows/Metal graph as the first multi-host substrate, with an explicit file-backed capacity policy and measured placement. Develop the residency/controller interface as a backend-neutral contract. Use a small HF AirLLM/Accelerate prototype to validate lease and cache ideas only where useful; avoid requiring a duplicate full-precision 72B download merely to demonstrate scheduling.

The first milestone is deliberately narrower than cross-engine tensor interoperability:

- Profile a small supported dense model with fixed prompts under all-resident, one-window streaming, and bounded retained-layer policies. Test sequential versus overlapped loads and enforce the same byte ceiling.
- Record file bytes requested, actual device/disk read bytes where observable, explicit cache hits, page faults, peak RAM/commit/pinned/GPU memory, TTFT and generated tokens from the tokenizer/engine. SSE chunks are not tokens.
- Label fresh-process, warm-process and repeated-file-access runs separately. Restarting a process does not create a cold OS cache. Do not flush the user's global cache or disrupt live workloads to manufacture a cold result.
- Simulate more/less device capacity in the offline planner, then prove the committed plan changes the observed layer placement. On a real added peer, verify that physical paging/transfer falls and end-to-end decode improves; do not report only predicted allocation.
- Inject read failure, canceled prefetch and peer loss. Require no leaked leases, bounded memory, an actionable error or clean request-boundary restart, and unchanged output correctness within the chosen backend's numerical tolerance.

This sequence implements automatic behavior without asking the user to solve memory placement manually. It also avoids promising that two new Macs automatically accelerate an already running 72B deployment before a replan/reload mechanism exists.

## Source index

All upstream code links below are immutable source permalinks. Repository metadata and commit heads were read through the GitHub API. No third-party benchmark claims were adopted as local measurements.

[airdiff]: https://github.com/lyogavin/airllm/compare/bc47e0d474abf03f491f7547085bd85a2c6f0b1b...22d72c46a67c9bf91da0615831fbb619e622c158
[airlicense]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/LICENSE
[acclicense]: https://github.com/huggingface/accelerate/blob/abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8/LICENSE
[flexlicense]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/LICENSE
[ktlicense]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/LICENSE
[airinit]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_base.py#L73-L212
[airmove]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_base.py#L600-L632
[airhooks]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_base.py#L878-L930
[airpin]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_base.py#L430-L458
[airresident]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_base.py#L673-L775
[airpersist]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/persist/safetensor_model_persister.py#L32-L43
[airclean]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/utils.py#L77-L101
[airexperts]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_base.py#L779-L876
[airsubset]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/utils.py#L425-L459
[airsplit]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/utils.py#L544-L807
[airauto]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/auto_model.py#L29-L79
[airmlx]: https://github.com/lyogavin/airllm/blob/bc47e0d474abf03f491f7547085bd85a2c6f0b1b/air_llm/airllm/airllm_llama_mlx.py#L414-L432
[accapis]: https://github.com/huggingface/accelerate/blob/abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8/src/accelerate/big_modeling.py#L179-L304
[acchooks]: https://github.com/huggingface/accelerate/blob/abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8/src/accelerate/hooks.py#L298-L438
[accload]: https://github.com/huggingface/accelerate/blob/abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8/src/accelerate/utils/offload.py#L50-L194
[accplan]: https://github.com/huggingface/accelerate/blob/abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8/src/accelerate/utils/modeling.py#L1327-L1375
[accdispatch]: https://github.com/huggingface/accelerate/blob/abe3749c6274b2ffc9a99cc57ca72ff8bca72ab8/src/accelerate/big_modeling.py#L375-L417
[flexpolicy]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/flex_opt.py#L34-L131
[flexreuse]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/flex_opt.py#L309-L337
[flexstreams]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/flex_opt.py#L582-L637
[flexloops]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/flex_opt.py#L1011-L1059
[flexdisk]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/pytorch_backend.py#L621-L687
[flexcopy]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/pytorch_backend.py#L866-L906
[flexunits]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/utils.py#L17
[flexcost]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/experimental/cost_model.py#L49-L156
[flexcostio]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/experimental/cost_model.py#L182-L267
[flexsearch]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/experimental/cost_model.py#L495-L529
[flexdist]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/dist_flex_opt.py#L30-L80
[flexcomm]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/dist_flex_opt.py#L232-L260
[flexread]: https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/README.md#L3-L30
[ktread]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/README.md#L64-L102
[ktfactory]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/python/experts.py#L320-L405
[ktplacement]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/python/experts_base.py#L162-L213
[ktstrategy]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/README.md#L498-L557
[ktexec]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/python/experts_base.py#L739-L949
[ktgguf]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/python/utils/loader.py#L1040-L1078
[ktmmap]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/operators/llamafile/moe.hpp#L269-L374
[ktplatform]: https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/README.md#L27-L90

