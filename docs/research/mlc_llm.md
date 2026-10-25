# MLC-LLM and MLX distributed: splitting one model across Mac Metal + NVIDIA CUDA

Audited 2026-10-20 (UTC). Sources were read only through `gh api -H "Accept: application/vnd.github.raw" repos/OWNER/REPO/contents/PATH?ref=SHA`,
`gh search issues`, `gh api repos/.../issues/N`, and the public Hugging Face and PyPI JSON APIs. Nothing was cloned, installed, built or run.
Line numbers refer to the file at the pinned SHA.

Scope: MLC-LLM and Apple MLX (+ mlx-lm) only. vLLM is not covered in this file.

| Repo | Pinned ref | License | Notes |
|---|---|---|---|
| mlc-ai/mlc-llm | `8978ea9aec626d64096aaa92fbc89bda2e3ea802` (main tip, 2026-10-06) | Apache-2.0 | No release tag since `v0.1.dev0` (2023). Nightly wheels only. |
| mlc-ai/relax (MLC's `3rdparty/tvm` submodule) | `201fa0001c81431ccce05419b442c47779ea917c` | Apache-2.0 | Disco runtime (socket session, NCCL). |
| ml-explore/mlx | `bff10220c742907f0c68e55ff174aaea56f74e3d` (main, 13 commits ahead of `v0.32.4`) | MIT | Latest release `v0.32.4`, 2026-10-14. |
| ml-explore/mlx-lm | `72be8c955d9a0d3099bd01865af288e0e947c3fa` (main tip, 2026-10-20) | MIT | Latest PyPI `mlx-lm` 0.32.2. |

Status words: VERIFIED (read in source, docs or API output at the stated ref), INFERRED (follows from the code, not exercised),
UNVERIFIED (not confirmed, reason given).

---

## 0. TL;DR for Hypercompute

1. **MLC-LLM cannot split one model across Metal and CUDA.** Any run with more than one worker must use NCCL or RCCL.
   Every other device aborts with "Multi-GPU on device ... is not supported" (`cpp/serve/engine.cc:826-835`). All workers share
   one device type (`engine.cc:836-840`). A maintainer said in #3018: "For now we don't support running on heterogeneous GPUs." (VERIFIED)
2. **MLC does have undocumented multi-node pipeline parallelism.** It uses a disco `SocketSession` driven by `MLC_SOCKET_HOST`/`MLC_SOCKET_PORT`,
   but tensors move between stages with **NCCL send/recv**, so every node must be CUDA, or every node ROCm. Only llama, llama4, nemotron,
   olmo and olmo2 implement `pipeline_parallel_stages`. **Qwen2/Qwen2.5 has tensor parallel only.** The layer split is an even ceil split fixed at compile time. (VERIFIED)
3. **MLC on Windows**: prebuilt wheels are CPU + Vulkan only. CUDA wheels exist only for Linux (`cu128`, `cu130`). (VERIFIED)
4. **MLX backends**: `ring` (TCP), `mpi`, `jaccl` (Thunderbolt RDMA, macOS >= 26.2) and `nccl` (CUDA). `ring` and `nccl` are excluded from
   **Windows** builds by CMake (`NOT WIN32`). `ring` is compiled into the macOS wheels and the Linux CPU/CUDA wheels. (VERIFIED)
5. **The MLX `ring` backend can put a Mac (Metal) and a Linux CUDA process in one group.** It runs on the CPU stream and moves raw bytes over TCP,
   with no device, OS or version handshake. MLX's docs do not cover mixed clusters, but EXO users run them: mlx#3862 reports a
   "heterogeneous 5-node ring (2× macOS/Metal, 3× Linux/CUDA", and mlx#3874 reports an RTX 5090 (sm_120) under **WSL2** in a mixed ring.
   VERIFIED in code and in field reports. UNVERIFIED on our two laptops.
6. **mlx-lm Qwen2 supports both pipeline and tensor sharding.** Pipeline uses `send`/`recv_like` + `all_gather`. It needs `ring` (or MPI),
   because the NCCL backend throws on send/recv and CUDA has no GPU Send/Recv. Tensor parallel runs 2 all-reduces per layer, which is not workable over Wi-Fi. (VERIFIED, last point INFERRED)
7. **Weighted split**: `PipelineMixin.pipeline(group, split=[...])` accepts any per-rank layer counts. But `sharded_load` and `mlx_lm.server --pipeline`
   never pass `split`, so the stock tools split evenly. **Rank 0 holds the LAST layers.** A weighted split needs a Hypercompute-owned loader. (VERIFIED)
8. **Windows host**: MLX has to run in **WSL2** (manylinux_2_35 wheel; Ubuntu 24.04 is fine). The native `win_amd64` mlx-cuda wheel has no ring or NCCL.
   Ring needs inbound TCP on both ranks, so WSL2 needs mirrored networking or a port proxy. (VERIFIED from code. Networking UNVERIFIED on this host)
9. **Verdict.** llama.cpp RPC is still the only engine with a documented Metal+CUDA layer split that runs natively from Windows and takes arbitrary ratios.
   MLX ring pipeline is the only credible second engine (native on the Mac, WSL2 + CUDA on the host). It is undocumented for mixed platforms and has open
   ring and Blackwell bugs. MLC-LLM cannot do this split.

---

## 1. MLC-LLM

### 1.1 Supported backends (VERIFIED)

- `README.md:31-60` lists Linux/Win NVIDIA as "Vulkan, CUDA", AMD as "Vulkan, ROCm" and macOS as "Metal". It also covers WebGPU, iOS Metal and Android OpenCL.
- `docs/install/mlc_llm.rst` prebuilt wheel tabs:
  - Linux (lines 24-70): CPU, `mlc-llm-nightly-cu128`, `mlc-llm-nightly-cu130`, ROCm 6.1/6.2. Vulkan is "Supported in all Linux packages".
  - macOS (lines 92-101): "CPU + Metal" only.
  - Windows (lines 118-127): **"CPU + Vulkan" only**. There is no Windows CUDA wheel.
- A model library is compiled for one target: `docs/compilation/compile_models.rst:120-121` (`--device cuda -o ...-cuda.so`) vs `135-136` (`--device metal`).
- `mlc_llm serve --device` takes a single device string: "one of `cuda`, `metal`, `vulkan`, `rocm`, `opencl`, `auto`" (`docs/deploy/rest.rst:98-102`).

### 1.2 Multi-GPU (disco) and multi-node (VERIFIED)

The engine creates the session in `cpp/serve/engine.cc` (MLC @ 8978ea9):

- L819: `int num_workers = num_shards * max_num_stages;`
- L826-835: collective library selection:
  ```cpp
  if (device.device_type == kDLCUDA) { ccl = "nccl"; }
  else if (device.device_type == kDLROCM) { ccl = "rccl"; }
  else { LOG(FATAL) << "ValueError: Multi-GPU on device " << ...
         << " is not supported. Currently, only NCCL and RCCL are integrated."; }
  ```
- L74-76: `GetEnvSocketHostPort()` reads `MLC_SOCKET_HOST` and `MLC_SOCKET_PORT`.
- L844-861: when `max_num_stages > 1` and a socket host is set, it creates `runtime.disco.SocketSession(max_num_stages, num_shards, num_groups=max_num_stages, host, port)`
  and logs the command remote nodes should run: `python -m mlc_llm.cli.disco_remote_socket_session <host> <port> <num_shards>`
  (`python/mlc_llm/cli/disco_remote_socket_session.py:24-33` calls `runtime.disco.RemoteSocketSession`).
- L862-875: otherwise it uses a local `ProcessSession`. If PP > 1 without the env vars, it prints a hint to set them.
- L877: `session.value()->InitCCL(ccl, Shape(device_ids));` runs for both session kinds.

TVM side (`mlc-ai/relax` @ 201fa00):
- `src/runtime/extra/disco/distributed/socket_session.cc:59-97`: the controller binds, listens (L76) and accepts `num_nodes - 1` remote nodes (L84-96),
  then sends each one `num_nodes, num_workers_per_node, num_groups, node_id`. `RemoteSocketSession` (L212-296) connects (L220) and forwards commands
  to its local workers (`MainLoop`, L235). **The socket is only the control plane.**
- Data plane `src/runtime/extra/disco/nccl/nccl.cc`:
  - L58-64: the controller calls `ncclGetUniqueId` and broadcasts it.
  - L111: `ncclCommInitRank(&ctx->global_comm, worker->num_workers, id, worker->worker_id)` builds one communicator across all nodes.
  - L115-116: `ncclCommSplit` per group.
  - L278-303: `SendToNextGroup` / `RecvFromPrevGroup` are `ncclSend` / `ncclRecv` on the global communicator.
- `nccl_context.h:59-63` hard-codes `TVM_DISCO_DEVICE_TYPE = kDLCUDA`. L77-81 is the ROCm/RCCL variant.
- `python/mlc_llm/compiler_pass/pipeline_parallel_rewrite.py:134,160` lowers the stage boundary op to `runtime.disco.recv_from_prev_group` / `send_to_next_group`.

So:
- Pipeline stages can span nodes over TCP control + NCCL data. (VERIFIED)
- Tensor-parallel groups have `num_shards` workers per node, so TP stays inside a node and PP stages map to nodes. (INFERRED from L859-860 and `socket_session_init_workers`, L316-326)
- None of the docs mention `pipeline_parallel_stages` or multi-node. A grep of `docs/` only finds `tensor_parallel_shards` (`deploy/cli.rst:55-62`,
  `deploy/rest.rst:56-63`, `deploy/python_engine.rst:102`). Issue #3198 (open since 2025-04-10) says: "I indeed found multi nodes support in source code, but it lacks of example".

### 1.3 Pipeline coverage and layer split (VERIFIED)

- `pipeline_parallel_stages` appears only in `model/llama`, `llama4`, `nemotron`, `olmo` and `olmo2` (grep over all `python/mlc_llm/model/` files).
- `QWen2Config` (`model/qwen2/qwen2_model.py:24-44`) has `tensor_parallel_shards: int = 1` (L40) and no pipeline field. Open issue #3302:
  "Qwen3MoeConfig does not support pipeline parallelism."
- The prebuilt `mlc-ai/Qwen2.5-14B-Instruct-q4f16_1-MLC` `mlc-chat-config.json` shows `model_config.tensor_parallel_shards = 1` and no `pipeline_parallel_stages` key.
- The split is even and fixed at compile time. `model/llama/llama_model.py:229-235`:
  `layers_per_stage = ceil(L / stages)`, `layer_partition = [i * layers_per_stage ...] + [L]`. There is no ratio or memory weighting.

### 1.4 Heterogeneous devices (VERIFIED)

- One engine has one `device`: `Model::Create(model_lib, model_str, model_configs[i], device, session, ...)` (`engine.cc:405`),
  and worker device ids are `device.device_id + i` (`engine.cc:836-840`).
- Issue #3018 (closed) asked to "Shard model on metal and vulkan simultaneously". The maintainer (MasterJH5574, 2024-11-14) replied: "For now we don't support running on
  heterogeneous GPUs. We will work on introducing REST API level protocols to bring different servers ... together."
- That REST protocol is MicroServing (`docs/microserving/tutorial.rst`). It does prefill/decode disaggregation between **whole engines**: each engine
  loads the full model, and KV transfer initializes NVSHMEM (`engine.cc:409-424`). It is CUDA-only and does not split layers.

### 1.5 Windows (VERIFIED, with one UNVERIFIED point)

- Native Windows: only the CPU + Vulkan wheel (`mlc_llm.rst:118-127`). There are open Windows bugs (#3384 "Windows CPU+Vulkan build error", #2979, #2983).
- Multi-GPU needs NCCL, and MLC only wires it for CUDA and ROCm. NCCL on native Windows is UNVERIFIED (not checked here).
- WSL2 + the Linux `cu130` wheel is plausible for single-GPU use. Whether the nightly works on sm_120 is UNVERIFIED.

### 1.6 Server API (VERIFIED)

- `mlc_llm serve MODEL [--model-lib] [--device] [--mode local|interactive|server] [--host] [--port]` (`rest.rst:78-104`). Default host is `127.0.0.1` and default port is `8000` (L159-160).
- OpenAI-style `GET /v1/models` (L175) and `POST /v1/chat/completions` (L199), with streaming (L322) and function calling (L334).
- Python `MLCEngine` / `AsyncMLCEngine` (`deploy/python_engine.rst`). Tensor parallel is set with `--overrides "tensor_parallel_shards=2"` (`rest.rst:59-63`).

### 1.7 Model format and Qwen2.5 availability (VERIFIED)

- A model needs two artifacts: converted weights (`params_shard_*.bin` + `ndarray-cache.json` + `mlc-chat-config.json`) and a **model library compiled per
  target** (`compile_models.rst:6-9`). The Python API JIT-compiles the library when `model_lib` is omitted (L20-21). The same `dist/...-MLC` weight dir
  is used for both the cuda and the metal library in the doc examples (L117-136). So weights are target-neutral and libraries are not. (INFERRED)
- The Hugging Face API lists 48 `mlc-ai` repos matching "Qwen2.5":
  - Instruct sizes 0.5B, 1.5B, 3B, 7B, 14B, 32B and 72B, in `q4f16_1`, `q4f32_1` and `q0f16`, plus some `q0f32`.
  - Coder and Math variants.
  - Example: `mlc-ai/Qwen2.5-14B-Instruct-q4f16_1-MLC` (sha `5afd957`, modified 2025-09-07) has `model_type` qwen2, `context_window_size` 32768 and 48 layers.

---

## 2. Apple MLX distributed

### 2.1 Backends and platforms (VERIFIED)

`docs/src/usage/distributed.rst:12-27` lists the backends:
- MPI
- RING: "Ring all reduce and all gather over TCP sockets"
- JACCL: "Low latency communication with RDMA over thunderbolt. Necessary for things like tensor parallelism."
- NCCL: "The backend of choice for CUDA environments."

Build gating (MLX @ bff1022):

| Backend | Compiled when | Source |
|---|---|---|
| ring | `MLX_BUILD_CPU AND NOT WIN32` | `mlx/distributed/ring/CMakeLists.txt:1-5` |
| jaccl | Darwin, macOS SDK and deployment target >= 26.2 | `mlx/distributed/jaccl/CMakeLists.txt:1-9`, `CMakeLists.txt:363-370` |
| nccl | `MLX_BUILD_CUDA AND NOT WIN32`, NCCL found | `mlx/distributed/nccl/CMakeLists.txt:1-12` |
| mpi | `MLX_BUILD_CPU` (any OS). It `dlopen`s `libmpi.dylib` / `libmpi.so` unless `MLX_MPI_LIBNAME` is set | `mpi/CMakeLists.txt`, `mpi/mpi.cpp:23-32` |

Backend selection:
- `init("any")` tries nccl when `cu::is_available()`, then ring, mpi, jaccl (`mlx/distributed/distributed.cpp:181-201`).
- NCCL init returns null unless `NCCL_HOST_IP`, `NCCL_PORT`, `MLX_RANK` and `MLX_WORLD_SIZE` are all set (`nccl/nccl.cpp:456-466`).
- Ring needs `MLX_HOSTFILE` + `MLX_RANK` (`ring/ring.cpp:824-845`).
- `mlx.launch` defaults to `nccl` if `mx.cuda.is_available()`, else `ring` (`python/mlx/_distributed_utils/launch.py:587-588`).

Packages:
- `docs/src/install.rst`:
  - macOS: Apple silicon, macOS >= 14 (L14-23).
  - CUDA: "Nvidia architecture >= SM 7.5", driver >= 550.54.14, "Linux distribution with glibc >= 2.35" (L35-42).
  - CUDA 13 needs "an Nvidia driver >= 580" (L44-45). The host driver is 591.84.
- PyPI `mlx-cuda-13` 0.32.4 (2026-10-14) ships `manylinux_2_35_x86_64`, `manylinux_2_35_aarch64` **and `win_amd64`** wheels.
  The release matrix builds Windows CUDA backends (`.github/workflows/release.yml:96-116`). Per the CMake gating, the Windows build has no ring or NCCL.
  The only distributed backend left there is MPI via `MLX_MPI_LIBNAME`, which is UNVERIFIED.
- sm_120:
  - CI falls back to `-DMLX_CUDA_ARCHITECTURES=80` when the build runner has no GPU (`.github/actions/setup/action.yml:248-251`).
  - The mlx#3874 reporter says "the published wheels lack sm_120 cubins entirely" and built from source with `CMAKE_CUDA_ARCHITECTURES=120`.
  - Whether the PyPI wheel runs on our sm_120 through PTX JIT is UNVERIFIED. The fallback is a source build with `-DMLX_CUDA_ARCHITECTURES=120`.

### 2.2 Can a Mac and a Linux CUDA box be in one MLX group?

From code (VERIFIED unless marked):
- The ring communication stream is the CPU on every build: `ring.cpp:431-433` `return to_stream(s, Device::cpu);`.
  Send and recv copy raw `data<char>()` bytes over sockets (`ring.cpp:505-545`).
- Ring init does no device, OS or MLX-version handshake (`ring.cpp:824-845`). A Metal rank and a CUDA rank should therefore interoperate when shapes and dtypes match.
  This assumes both hosts are little-endian and run the same MLX version. (INFERRED)
- On CUDA, CPU-stream access works because MLX allocates unified or pinned memory (`mlx/backend/cuda/allocator.cpp:60-68`) and moves device buffers there
  (L248-253). The code comment at L41-47 says that on "Windows (and WSL)" managed memory is off, so `cudaMallocHost` is used instead.
- An NCCL group cannot include a Mac (CUDA only), and JACCL is macOS only. **Ring is the only cross-platform option.**

Field reports (VERIFIED as reports, not reproduced):
- mlx#3862 (closed by PR #4060, merged 2026-08-09) was "Observed on a heterogeneous 5-node ring (2× macOS/Metal, 3× Linux/CUDA". Before the fix,
  a Wi-Fi reset killed the ring socket thread and every rank hung. `gh api compare` shows the fix commit `8c28c38` is an ancestor of `v0.32.4`.
- mlx#3874 (closed `not_planned`; the reporter retracted it as VRAM exhaustion): RTX 5090 sm_120, CUDA 12.8, "WSL2 (Ubuntu, mirrored networking)", in a
  "5-rank ring pipeline (2× Metal + sm_121 + sm_110 + sm_120)" run by EXO.
- mlx#3876 (open): in a mixed Metal + Blackwell (sm_121, sm_110) ring, the CPU-stream `all_sum` barrier hangs in `cu::AtomicEvent::wait` during warmup.
  This is a risk for our sm_120.
- mlx#4276 (open): ring `recv` fails with EFAULT during multi-node pipeline inference (Mac-only pairs, mlx 0.32.0).

**Answer:** yes in code and in the field (through EXO), not documented by MLX, and not yet verified on our pair.

### 2.3 Pipeline vs tensor sharding in mlx-lm (VERIFIED)

- `mlx_lm/models/pipeline.py:16-38` (`PipelineMixin.pipeline(group, split=None)`):
  - With no split, the layers are split evenly and "the low ranks get the extra layers".
  - With an explicit `split`, it must have one positive entry per rank and sum to the layer count.
  - The order is reversed: "rank=0 gets the last layers". Layers outside the rank's range are replaced with `None`.
- Models that use `PipelineMixin` at 72be8c9: bailing_hybrid, deepseek_v2, deepseek_v3, deepseek_v32, deepseek_v41, glm4_moe, glm4_moe_lite, minimax_m3_vl,
  ministral3, mistral4, muse_spark, **qwen2**, qwen3_5, qwen3_moe, qwen3_next. `llama.py` and `qwen3.py` have `shard()` but no pipeline support.
  24 model files define `shard()`.
- Qwen2.5 maps to `qwen2`: the `mlx-community/Qwen2.5-14B-Instruct-4bit` config has `model_type` qwen2, hidden 5120, 48 layers, 4-bit, group 64.
- `qwen2.py` forward (L149-171):
  1. `recv_like` from rank+1.
  2. Run the local layers.
  3. `send` to rank-1.
  4. `all_gather(h)` so every rank gets the final hidden state.
- `qwen2.py:206` is `shard()` (tensor parallel). `qwen2.py:238-239` makes `layers` return `pipeline_layers`, so the KV cache only covers local layers.
- **Pipeline does not work on the NCCL backend.**
  - `nccl.cpp:360-366`: "[nccl] Send not supported in NCCL backend."
  - `mlx/backend/cuda/primitives.cpp:58-61`: `NO_GPU_MULTI(Send)` and `NO_GPU_MULTI(Recv)`.
  - Metal has no GPU send/recv either (`backend/metal/distributed.cpp:25-31`).
  - With ring, send and recv run on the CPU stream, and only between direct neighbours (`ring.cpp:505-545`). That is enough for 2 ranks.
- Tensor parallel needs `all_sum` inside every attention and MLP block. The docs call JACCL "Necessary for things like tensor parallelism" (`distributed.rst:24-25`).
  Over Wi-Fi this is not viable. (INFERRED)
- `mlx_lm/utils.py:709-812` (`sharded_load`):
  - When neither group is given, it prefers tensor parallel (L753-757).
  - For pipeline, it reads `model.safetensors.index.json` (L766) and downloads only the files for the local layers.
  - "Every stage keeps the embeddings, the final norm and the head" (L775).
  - L805-808 calls `model.model.pipeline(pipeline_group)` **without `split`**.
  - Both `mlx-community/Qwen2.5-14B-Instruct-4bit` and `-7B-Instruct-4bit` include the index file.
- Server:
  - `mlx_lm.server --pipeline` (`server.py:2063-2066`) picks the pipeline group (L395-400). Rank 0 serves HTTP and the other ranks join the generator (L1901-1906).
  - Routes: `/v1/completions` and `/v1/chat/completions` (L1254-1255), `/v1/models` (L1779).
  - `SERVER.md:8-9`: "not recommended for production".
- Install: mlx-lm extras `cuda13`, `cuda12` and `cpu` (`pyproject.toml:44-46`). The base dependency is `mlx>=0.32.3; platform_system == 'Darwin'` (L25).
- Example launcher: `examples/sharded_generate.py:7-12, 47-63` (`--pipeline`).

### 2.4 Launching across the Mac and WSL2

- `mlx.launch` uses ssh to reach each host (`launch.py:54,96`). It uses one `--python` path for every host (L559-561), and ring requires IPs, not hostnames (L315).
  The alternative is to start each rank yourself with `MLX_RANK` + `MLX_HOSTFILE` (`distributed.rst:516-545`). (VERIFIED)
- In the ring connect order, the lower rank accepts first and then connects, while the last rank connects first (`ring.cpp:360-374`).
  **Both ranks must accept inbound TCP.** WSL2's default NAT blocks inbound LAN connections, so mirrored networking (used in mlx#3874) or a portproxy is needed.
  (Code VERIFIED. Host networking UNVERIFIED)
- The ring sockets have no authentication beyond TCP connect, the same as llama.cpp RPC. Bind them to the private interface only. (INFERRED)

---

## 3. Weight distribution: how each engine places layers

| Engine | Metal + CUDA in one model | Split control | Set at | Runs natively on the Windows host |
|---|---|---|---|---|
| llama.cpp RPC b11770 (`llama_rpc.md`, `llama_placement.md`) | Yes, documented | Any `-ts` ratio, or automatic by free memory (`--fit`, default on) | Runtime | Yes |
| MLX ring + mlx-lm pipeline | Yes in code and in EXO field reports, undocumented | The API takes any per-rank layer counts. The stock CLI and server split evenly | Load time | No (WSL2) |
| MLC-LLM disco PP | No (NCCL/RCCL only) | Even ceil split, llama-family only | Compile time | No (Windows has Vulkan single-GPU only) |

How to get a weighted MLX split (INFERRED, untested):
- Write a small Hypercompute launcher that mirrors `sharded_load`, but calls `model.model.pipeline(group, split=[n_rank0, n_rank1])` (rank 0 = last layers).
- Then filter the safetensors files by the resulting `start_idx`/`end_idx`, as `utils.py:765-792` does.
- Upstream MLX code needs no changes, and `PipelineMixin` already validates the split.

Per-token traffic for Qwen2.5-14B (INFERRED): each decode step sends one 5120-element activation (about 10 KiB at 2 bytes per element) from rank 1 to rank 0,
plus an `all_gather` of the same size. Decode is therefore bound by Wi-Fi latency, at roughly 2 hops per token. That is the same order as llama.cpp RPC's per-token round trips.
Not measured.

---

## 4. Conclusion

- **MLC-LLM: no.** It can split one model across machines only when every machine is CUDA (or every one is ROCm). The split is pipeline-only and only for llama-family
  models. Metal, Vulkan and mixed device types are rejected at engine start (`engine.cc:826-835`), and the maintainers confirmed it (#3018).
  Qwen2.5 can only use intra-node tensor parallelism.
- **MLX: yes, with caveats.**
  - The `ring` backend can join a native macOS Metal rank and a Linux CUDA rank. On our setup the Linux rank would run in WSL2 on the RTX 5070 Ti.
  - mlx-lm's Qwen2 pipeline can split layers between them.
  - The mixed setup is not in the MLX docs, and pipeline mode needs `--backend ring` (NCCL has no send/recv).
  - The stock tools use an even split.
  - There are open bugs: mlx#3876 (Blackwell barrier hang in a mixed ring) and mlx#4276 (ring recv EFAULT).
  - sm_120 support in the PyPI wheel is unconfirmed.
- **llama.cpp RPC already does all of this today.** It runs natively on Windows, the README shows CUDA + Metal + CPU together, and it takes arbitrary `-ts` ratios.
  It stays the primary engine.

Suggested next hardware checks before any MLX adapter work (not done):
1. Enable WSL2 mirrored networking. Confirm the Mac can reach a listener inside WSL.
2. Install `mlx[cuda13]==0.32.4` in WSL2 and check `mx.cuda.is_available()`, a bf16 matmul and a 4-bit `quantized_matmul` on sm_120. If they fail, build from source with
   `-DMLX_CUDA_ARCHITECTURES=120`.
3. Run a 2-rank ring `all_sum` and `send/recv` test (Mac `mlx==0.32.4` + WSL), launched by hand with `MLX_RANK`/`MLX_HOSTFILE`.
4. Run `mlx-community/Qwen2.5-7B-Instruct-4bit` with `examples/sharded_generate.py --pipeline` under `--backend ring`. Then try a custom split.
