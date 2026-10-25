# Adjacent distributed runtimes: Petals and Distributed Llama

Research date: 21 October 2026, America/Los_Angeles. Source inspection, not a hardware benchmark.
Repository metadata, immutable source files and licenses were read through the GitHub connector.

| Repository | Inspected commit | License | Useful contribution |
| --- | --- | --- | --- |
| [Petals](https://github.com/bigscience-workshop/petals) | `22afba627a7eb4fcfe9418c49472c6a51334b8ac` | MIT | Latency-aware routing, KV admission, failure recovery |
| [Distributed Llama](https://github.com/b4rtaz/distributed-llama) | `4074c0f3fc02800346ccd6df3293915d4276c3d2` | MIT | Tensor slicing, explicit synchronization, network counters |

## Petals: route blocks using measured links and available KV space

The [routing graph](https://github.com/bigscience-workshop/petals/blob/22afba627a7eb4fcfe9418c49472c6a51334b8ac/src/petals/client/routing/sequence_manager.py#L220)
contains compute edges for individual blocks and transfer edges between servers. It incorporates ping
measurements, inference-rate estimates, serialization overhead and a penalty for insufficient KV capacity.
Its throughput-oriented route is a different policy, favoring long spans while distributing requests;
it is not the same objective as minimum single-request latency.

**Reuse the method:** distinguish prefill, decode, network and admission costs. A count of parameters or
total free RAM is not a latency estimator. Our overlay should retain the existing private Iroh identity
and bridge rather than introduce a second discovery/control stack just to obtain this algorithm.

[Inference sessions](https://github.com/bigscience-workshop/petals/blob/22afba627a7eb4fcfe9418c49472c6a51334b8ac/src/petals/client/inference_session.py#L284)
run spans in sequence. On a server error they update the route and recreate sessions. The
[replacement logic](https://github.com/bigscience-workshop/petals/blob/22afba627a7eb4fcfe9418c49472c6a51334b8ac/src/petals/client/inference_session.py#L364)
retains input history so missing attention state can be regenerated. This is a concrete model for recovery:
a new worker needs the appropriate token history/cache, not just the next activation packet.

[MemoryCache](https://github.com/bigscience-workshop/petals/blob/22afba627a7eb4fcfe9418c49472c6a51334b8ac/src/petals/server/memory_cache.py#L26)
tracks persistent inference tensors, chiefly attention KV, with bounded allocation and timeouts. It is
not a disk-weight cache. Preserve that distinction in an implementation: weight residency, KV residency,
and temporary workspaces require separate reservations and lifetimes.

The inspected main commit dates to August 2024. The model directory at this revision registers Bloom,
Falcon, Llama and Mixtral families; no Qwen2 family appears there. Nothing in this audit establishes a
drop-in Qwen2.5-72B deployment across native Windows CUDA and M4 Metal. Treat it as an algorithm and
recovery reference, with MIT notices retained if code is adapted.

## Distributed Llama: parallel arithmetic requires frequent synchronization

The [README](https://github.com/b4rtaz/distributed-llama/blob/4074c0f3fc02800346ccd6df3293915d4276c3d2/README.md#L64)
describes one root plus workers and power-of-two total node counts. Its root loads and distributes model
weights, while every node executes a tensor slice. This is structurally different from assigning whole
consecutive layers to llama.cpp RPC workers.

The [executor builds compute and synchronization steps](https://github.com/b4rtaz/distributed-llama/blob/4074c0f3fc02800346ccd6df3293915d4276c3d2/src/nn/nn-executor.cpp#L45)
from segment descriptions. [Network synchronization](https://github.com/b4rtaz/distributed-llama/blob/4074c0f3fc02800346ccd6df3293915d4276c3d2/src/nn/nn-network.cpp#L542)
explicitly broadcasts from the root and exchanges node slices. The code also exposes
[sent/received byte counters](https://github.com/b4rtaz/distributed-llama/blob/4074c0f3fc02800346ccd6df3293915d4276c3d2/src/nn/nn-network.cpp#L525).
This supports studying actual synchronization volume rather than assuming more GPUs always improve latency.

Its [GPU instructions](https://github.com/b4rtaz/distributed-llama/blob/4074c0f3fc02800346ccd6df3293915d4276c3d2/docs/HOW_TO_RUN_GPU.md#L1)
use Vulkan. That is a separate executor/model-distribution path, not a direct reuse of Hypercompute's
CUDA/Metal llama RPC placements. The inspected network file includes Windows socket branches, but source
portability does not establish measured native Windows/M4 performance. No build or benchmark was run here.

**Reuse:** explicit synchronization plans and network counters; benchmark tensor parallelism on a good
direct link as an alternative. **Do not import blindly:** its executor, weight representation and worker
protocol do not become compatible with ggml RPC by wrapping their chat APIs.

## Architectural consequence

For batch-one interactive decoding, score a pipeline as the sum of each stage and boundary cost.
For an actually microbatched pipeline, the slowest stage limits sustained throughput after filling the
pipeline. Tensor parallelism can split one layer's arithmetic, but adds collectives repeatedly within
the model. The controller must model these as separate execution families with explicit capabilities.

For this hackathon, first prove capacity plus automatic replan/reload using the already working
llama.cpp CUDA/Metal executor. Keep Petals-style routing/KV recovery and tensor-parallel alternatives
as well-defined extensions. Do not describe process reload as live tensor or KV migration.
