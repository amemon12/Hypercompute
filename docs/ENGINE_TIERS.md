# Engines and tiers

Hypercompute serves one model at a time. Each deployment has exactly one engine owner. Engines never share one
model's layers with each other. This page is the contract for every engine and for the selector that chooses
between them. The research behind each decision is in `docs/research/`.

## The tier ladder

The selector picks the fastest tier that fits, in this order:

| Tier | Engine | Where the weights live | Expected speed (one user) |
|---|---|---|---|
| `gpu` | llama.cpp (+ RPC peers) | every layer on a GPU, the local one and/or peers | fastest; peers add one Wi-Fi round trip per token |
| `gpu_ram` | llama.cpp | the first layers on the host CPU from RAM, the rest on GPUs | a few tok/s lower per GB moved to RAM |
| `gpu_disk` | llama.cpp | like `gpu_ram`, but the CPU part is larger than free RAM and is paged from the GGUF (mmap) | slow, disk bound |
| `disk` | AirLLM | every layer is read from disk for every token | capacity mode: seconds to minutes per token |

Side engines, chosen only when they are the better fit:

- **vLLM** (`vllm`): one NVIDIA GPU on Linux or WSL2, the whole model on that GPU. About the same single-user speed as
  llama.cpp, several times the throughput with many concurrent requests. Picked for `objective=throughput` when the
  model fits one local NVIDIA GPU and vLLM is installed, or when forced with `engine=vllm`.
- **EXO** (`exo`): Apple-only clusters running MLX. Picked when every chosen node is a Mac with EXO running, or forced.

AirLLM comes in only when nothing faster can serve the request: the model exists only as an HF checkpoint (no GGUF),
the request asks for full precision, or no llama.cpp tier fits even with disk paging. `allow_capacity_mode=false`
turns off both disk-backed tiers, `disk` and `gpu_disk`.

On Windows, a feasible `gpu_ram` plan with CPU layers automatically uses llama.cpp's
`--load-mode none --no-host --no-repack`. This avoids whole-file memory-map prefetch during loading
and pinned/repacked CPU weights. The planner still charges the CPU weights, KV cache and working
buffers against available RAM. Disk capacity plans keep file-backed loading; the resident loader
must not be used to bypass their RAM accounting.

## Layer split rules (llama.cpp)

From `docs/research/split_optimization.md`:

1. RPC devices go first in `--device` and the local GPU goes last. The last device holds the output head, so logits
   (608 KB per token for Qwen2.5) never cross Wi-Fi; only 14 KB of activations do.
2. For one user, a split is a sequential chain. Fill the local GPU first, then send only the overflow off it, to the
   cheaper of the host CPU tier and the peers. Splitting a model that already fits only adds latency.
3. Measured profiles override heuristics. `hypercompute tune` tries a few candidates (peer vs RAM tier, head
   placement, `-fa`, `-ub`), scores each by `TTFT + N_g / decode_tps`, and saves the winner. Later plans for the same
   model, context and device set reuse it.

## Contracts

### Optional engine modules

Every optional engine has two modules.

`hypercompute/installers/<engine>.py`, used by `hypercompute engine install|status --engine <engine>` and by the
machine's inventory:

```python
def install(paths: config.Paths, *, yes: bool, force: bool, say: Callable[[str], None]) -> int: ...
def status(paths: config.Paths) -> ExtraEngineInfo: ...   # cheap, no network, never raises for "missing"
```

`hypercompute/adapters/<engine>.py`, loaded by `adapters/registry.py`:

```python
def build(ctx: AdapterContext) -> EngineAdapter | None: ...   # None when not installed on this host
def status(ctx: AdapterContext) -> ExtraEngineInfo: ...
def plan(req: PlanRequest, art: ModelArtifact, local: NodeView, remote: list[NodeView],
         ctx: AdapterContext) -> Plan: ...                    # feasible=False with reasons when it cannot serve
```

The adapter implements `EngineAdapter` from `adapters/base.py`: `engine_id`, `idle_timeout_s`, `inspect`,
`validate`, `prepare`, `wait_ready`, `observe`, `complete`, `stream`, `cancel`, `stop`, `read_log`,
`process_alive`, `aclose`. `stream` yields `GenerationEvent`s whose `raw` bytes are OpenAI-style SSE
(`data: {...}\n\n`, ending with `data: [DONE]\n\n`). The final chunk carries `usage`, and `timings` when the engine
can measure them (`prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`), so run records stay comparable across
engines.

Optional hooks. The gateway calls each one only when the adapter defines it (the list is also in the
`EngineAdapter` docstring):

| Hook | What the gateway uses it for | Without it |
|---|---|---|
| `load_timeout_s(plan) -> float` | `wait_ready`'s timeout at startup, and the tuner's budget for one trial (vLLM: a first start that downloads the model) | the adapter's own `wait_ready` default |
| `idle_timeout_for(plan) -> float` | this deployment's longest silence between streamed events: the capacity-mode warmup budget and the limit a timed-out run reports | `idle_timeout_s` |
| `max_concurrency(plan) -> int` | the deployment's inference slots (see [Inference concurrency](#inference-concurrency)) | 1 |
| `failure_reason(log_text) -> str \| None` | the error a deployment gets when its engine dies after it was ready | llama.cpp's log patterns (`detect_failure`) |
| `async cleanup_orphan(dep) -> (bool, str)` | at the next gateway start after a crash: remove what the engine left running for a deployment that has no local process; `True` only once nothing is left | the leases are released at restart |

What a crashed gateway can leave behind:

- Local engine processes (llama-server, the AirLLM worker, `wsl.exe` for vLLM) are started through `Supervisor`.
  They die with the gateway (kill-on-close Job Object on Windows, PDEATHSIG on Linux; macOS has neither). The next
  gateway start stops any recorded process that is still running, and keeps the leases of one it could not stop.
- vLLM in WSL2: killing `wsl.exe` does not stop the Linux process. The launch wrapper starts a watchdog inside the
  distro that sends TERM, then KILL after 30 s, to the server's process group once `wsl.exe` is gone. Until then
  the GPU memory is still in use. The next gateway start also stops any server a deployment's pidfile still names
  (`cleanup_orphan`) and holds that deployment's leases until vLLM is confirmed gone.
- EXO runs no local process. Its instance on the Macs is deleted at the next gateway start (`cleanup_orphan`), and
  the deployment's leases stay held until EXO confirms the instance is gone. If EXO cannot be reached, a later
  start tries again.

### Plans for non-llama engines

`Plan.engine` names the owner. `device_list`, `tensor_split` and `rpc_endpoints` stay empty unless the engine uses
llama.cpp RPC. `assignments` still list every node and pool the engine uses, so leases reserve their memory.
`engine_options` carries engine-specific data: the vLLM launch arguments, the EXO placement preview, the AirLLM
shard directory. `capacity_mode=true` marks both disk-backed tiers: `gpu_disk` (llama.cpp pages part of the model
from the GGUF) and `disk` (AirLLM). The UI then shows a banner. The warmup asks for one token, and its budget is the
larger of the normal warmup timeout and twice the deployment's idle budget (`warmup_timeout` in
`services/deployments.py`).

### Model artifacts

`ModelArtifact.format` is `gguf` (llama.cpp), `hf` (safetensors directory: AirLLM, vLLM) or `mlx` (EXO).
`artifact_engines()` in `adapters/base.py` gives the default engines for each format. HF files keep their repo
paths (`ModelFile.path`), and every file is pinned by revision and SHA256. An artifact with no files is
engine-managed, and the model view reports `engine_managed`:

- MLX (EXO): each node's EXO downloads the repo itself.
- vLLM HF checkpoints (`files: []`, for example `qwen2.5-7b-instruct-awq`): vLLM downloads `repo@revision` itself
  inside WSL2 on the first start, into its own Hugging Face cache on the Linux file system
  (`~/.hypercompute-vllm/hf`). The plan pins the revision (`vllm serve --revision`) and is infeasible when that
  file system lacks room for the download. Later deployments of the same `repo@revision` reuse the cache.

### Inference concurrency

The gateway gives each deployment a number of inference slots: `adapter.max_concurrency(plan)` when the engine's
adapter has it, else 1. A chat request holds one slot. Warmup, benchmarks and tuner measurements hold every slot
(exclusive), so their timings never include another request. Waiters are served in arrival order: a waiting
benchmark keeps new chats off the slots that free up, and a request that gets no slot within 30 s is refused with
429 `busy`. A new deployment starts with fresh slots sized by its own plan. Stopping a deployment cancels every
request in flight on it, and each request still leaves exactly one run record.

| Engine | Slots | Why |
|---|---|---|
| llama.cpp (`llama_rpc`) | the plan's `PlanRequest.parallel` (`engine_options["parallel"]`, 1 to 16, default 1) | 1 launches `llama-server -np 1`, one slot with the whole context for one sequence; N > 1 launches `-np N -kvu`, N slots sharing one unified KV pool of `-c` cells, so the KV budget is unchanged. The stream tier refuses N > 1, other engines refuse `parallel` > 1 (422), and the replan evaluator proposes nothing for a multi-slot deployment |
| vLLM (`vllm`) | the plan's `--max-num-seqs` (`MAX_NUM_SEQS` in `adapters/vllm.py`: 8 interactive, 16 throughput) | continuous batching: concurrent requests share each decode step |
| EXO (`exo`) | 1 | no `max_concurrency`; batched serving in EXO is not verified (`docs/research/exo.md`) |
| AirLLM (`airllm`) | 1 | the worker streams one request at a time and answers 429 `busy` itself (batch size 1 for streaming) |
