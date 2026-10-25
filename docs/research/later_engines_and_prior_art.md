# Later engines and prior art audit

Audited 2026-10-20 (UTC) through the authenticated `gh` CLI (`gh api repos/OWNER/REPO/contents/PATH?ref=SHA`) and the public Hugging Face API. No external code was executed, cloned or installed. Line numbers refer to the file at the pinned SHA.

Status words used below:
- VERIFIED: read directly in source or API output at the stated ref.
- INFERRED: follows from reading the code, but not exercised on hardware.
- UNVERIFIED: could not be confirmed, with the reason.

---

## A. EXO (exo-explore/exo @ bc412e2230f395e652c23426a5eb88dfa515769b)

### Commit and license
- The commit exists (VERIFIED). Committer date 2026-10-13T20:11:46Z. Subject: "app: wait for the old exo to exit before starting a new one (#2427)". The repo's `pushed_at` is 2026-10-13T20:11:49Z, so at audit time this SHA is the tip of `main`.
- License: Apache-2.0. `LICENSE` lines 1-2 read "Apache License / Version 2.0, January 2004", and the GitHub API reports `spdx_id: Apache-2.0`.

### Python and MLX pins (`pyproject.toml`)
- `version = "0.3.70"` (line 3).
- `requires-python = "==3.13.*"` (line 6). basedpyright also uses `pythonVersion = "3.13"` (line 151). Python 3.13 is required exactly.
- The optional extra `mlx` (lines 50-60) contains `"mlx>=0.32.0"`, `"mlx-lm"` (no version), `"mlx-vlm>=0.3.11"`, `"mflux==0.17.5"` and `torch==2.10.0` / `torchvision==0.25.0` / `torchaudio==2.10.0` on darwin and linux.
- uv source overrides:
  - On darwin, `mlx` comes from git `ml-explore/mlx` at rev `0e3ff3643b1c3719f78814b98e0d222afbad867c` (line 84).
  - On linux, `mlx` comes from prebuilt `mlx-0.32.0-cp313` wheels in `rltakashige/mlx-jaccl-fix-small-recv` (lines 85-86).
  - `mlx-lm` comes from the git fork `rltakashige/mlx-lm`, branch `leo/deepseek-v4` (line 96). That is a branch, not a pinned rev.
  - `mflux` comes from the fork `evanev7/mflux`, branch `exo2`.
- Other extras:
  - `mlx-cpu`: `mlx-cpu==0.31.2`, linux only (line 62).
  - `mlx-cuda12`: `mlx-cuda-12==0.32.0`, linux only (line 65).
  - `mlx-cuda13`: `mlx-cuda-13==0.32.0`, linux only (line 70).
- Platforms: `environments = ["sys_platform == 'darwin'", "sys_platform == 'linux'"]` (line 188). README line 73 says "from source on macOS or Linux". **Windows is not supported**, so the Windows/NVIDIA machines cannot join an EXO cluster natively.
- Conflicting docs (UNVERIFIED): README line 247 says "Currently, exo runs on CPU on Linux. GPU support for Linux platforms is under development", but README line 238 and the pyproject both offer `--extra mlx-cuda13` / `mlx-cuda12`. Treat Linux+CUDA EXO as unproven until it is tested on hardware such as the GX10.
- Default API and dashboard: `http://localhost:52415` (README lines 71, 185).

### API routes (`src/exo/api/main.py`, registrations at lines 390-455)
OpenAI-compatible (VERIFIED):
- `POST /v1/chat/completions` (lines 408-410) maps to `chat_completions` (line 973). With `stream: true` it returns a `StreamingResponse` with `media_type="text/event-stream"` (lines 982-996). Otherwise it returns a JSON `CollectedResponse`.
- `GET /v1/models` and `GET /models` (lines 403-404).
- Also present: `POST /v1/responses` (OpenAI Responses), `POST /v1/messages` (Anthropic style), `POST /v1/cancel/{command_id}`, image endpoints, and Ollama-compatible routes under `/ollama/...` (lines 429-441).

Placement and instance management (VERIFIED):
| Route | Handler (line) | Notes |
|---|---|---|
| `GET /instance/placement` | `get_placement` (523) | Defaults are `sharding=Sharding.Pipeline`, `instance_meta=InstanceMeta.MlxRing` and `min_nodes=1`. Returns 400 with a reason when the model cannot be placed. |
| `GET /instance/previews` | `get_placement_previews` (564) | Candidate placements. |
| `POST /place_instance` | `place_instance` (475) | Runs `get_placement` first so it can refuse with a reason, then sends the `PlaceInstance` command. |
| `POST /instance` | `create_instance` (498) | Accepts an explicit `Instance`. Returns 400 when `storage_size` is larger than total available memory. |
| `GET /instance/await` | `await_instance` (695) | SSE wait, described below. |
| `GET /instance/{instance_id}` / `DELETE /instance/{instance_id}` | 690 / 737 | |
| `GET /state`, `GET /state/{path}`, `GET /events`, `GET /node_id` | 443-445, 390 | Cluster state, including runners. |
| `POST /download/start`, `DELETE /download/{node_id}/{model_id}`, `POST /download/cancel` | 446-448 | |

`InstanceMeta` has only `MlxRing` and `MlxJaccl` (`src/exo/shared/types/worker/instances.py` lines 15-17).

### "Acknowledged" vs "ready" (VERIFIED in code, INFERRED semantics)
- **Acknowledged.** `POST /place_instance`, `POST /instance` and `DELETE /instance/{id}` return immediately after `_send(command)` with
  `CreateInstanceResponse(message="Command received.", command_id=..., model_card=...)` (main.py lines 492-496, 517-521, 745-749). This only means the command was forwarded. README (around line 412) says: "This command is asynchronous."
- **"ready" as the API defines it.** `GET /instance/await?model_id=...&timeout_seconds=N` streams SSE. When any instance whose `shard_assignments.model_id` matches appears in `self.state.instances`, it emits `data: {"type":"ready","instance":{...}}` (main.py lines 707-712; `AwaitInstanceReadyMessage` has `type: Literal["ready"]` in `src/exo/api/types/api.py` lines 310-312). Otherwise it emits `{"type":"timeout","message":...}` (api.py lines 315-317). `timeout_seconds` defaults to 0, which waits forever, and the maximum is 300 (main.py line 698).
- **Caveat (INFERRED).** The await endpoint only checks that the instance exists in state. It does not check that weights are loaded. Runner lifecycle lives separately in `State.runners` (`src/exo/shared/types/state.py` line 47) with these statuses: `RunnerIdle`, `RunnerConnecting`, `RunnerConnected`, `RunnerLoading(layers_loaded, total_layers)`, `RunnerLoaded`, `RunnerWarmingUp`, `RunnerReady`, `RunnerRunning`, `RunnerShuttingDown`, `RunnerShutdown` and `RunnerFailed` (`src/exo/shared/types/worker/runners.py` lines 25-66). The Hypercompute EXO adapter should:
  - show "acknowledged" when it receives `Command received.`;
  - show "placed" on the await `ready` event;
  - show "ready" only once every runner of that instance reports `RunnerReady` or `RunnerRunning` via `GET /state`;
  - use `RunnerLoading.layers_loaded/total_layers` as the progress bar.
- **Chat acceptance.**
  - A chat request for a model with no instance returns 404 `"No instance found for model {model_id}"` (main.py lines 1044-1067).
  - After a chat request is sent, the API marks it accepted on `TaskCreated` or the first `ChunkGenerated` (lines 2052-2056).
  - If it is not accepted, the API resends after 10, 15 and 30 s (`REQUEST_RESEND_WAITS`, line 236). It then streams the ErrorChunk "The cluster didn't accept the request. Please check that the model is running and try again." (line 237).

### Implication
EXO fits a later Mac-only phase (MLX ring or JACCL), driven over HTTP on port 52415. Hypercompute should not vendor EXO code. Apache-2.0 would allow it, but the HTTP API is enough.

---

## B. AirLLM (lyogavin/airllm @ bc47e0d474abf03f491f7547085bd85a2c6f0b1b)

### Commit and license
- The commit exists (VERIFIED). Committer date 2026-10-20T06:19:55Z. Subject: "chore: refresh star history chart". It matches the repo's `pushed_at` of 2026-10-20T06:19:56Z, so it is the tip of `main`.
- License: Apache-2.0. Both `/LICENSE` and `/air_llm/LICENSE` are Apache 2.0 text, and the `air_llm/setup.py` classifier at line 45 says "License :: OSI Approved :: Apache Software License". Package version `4.0.0` (setup.py line 18).

### Architectures supported on the Mac path (`air_llm/airllm/auto_model.py`)
- On darwin, `from airllm import AirLLMLlamaMlx` (lines 10-11).
- `MLX_ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM"}` (line 31).
- `from_pretrained` on macOS raises `ValueError(f"{arch} is not supported on macOS yet. The macOS backend (MLX) only runs Llama-style models: ...")` for any other architecture (lines 65-74).
- **Qwen2.5 (`Qwen2ForCausalLM`) is not supported on the Mac path.** An AirLLM phase on Mac needs a Llama or Mistral checkpoint in HF safetensors format. AirLLM does not load GGUF.
- On other platforms, `ARCH_OVERRIDES` (lines 17-27) covers ChatGLM, QWen (v1), Baichuan, InternLM, KimiK3, Qwen3_5 and Qwen4Exp. Every other standard `*ForCausalLM` goes through the generic `AirLLMBaseModel` (lines 56-60).
- Security note (VERIFIED): `AutoConfig.from_pretrained(..., trust_remote_code=True)` is called in auto_model.py lines 44-47 and airllm_llama_mlx.py lines 254-256. `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` is called at lines 270-272. A model repo with custom code would execute that code. Hypercompute must restrict AirLLM to an allowlist of model repos.

### What `generate()` returns (`air_llm/airllm/airllm_llama_mlx.py` lines 275-287)
```python
def generate(self, x, temperature=0, max_new_tokens=None, **kwargs):
    x = _coerce_to_mlx_array(x)
    ...
        if len(tokens) >= max_new_tokens:
            break
    s = self.tokenizer.decode([t.item() for t in tokens])
    return s
```
- It returns **one decoded `str`**. It returns no token ids, no stream and no timing stats. Torch tensors are accepted as input and converted to `mx.array` (lines 192-205).
- Gotchas (INFERRED from code; not run):
  - `max_new_tokens` defaults to `None`, and `len(tokens) >= None` raises `TypeError` in Python 3. Callers must always pass `max_new_tokens`.
  - `temperature` is accepted, but `sample(y)` is called without it (lines 376, 452), so the MLX path always does greedy argmax.
  - There is no EOS check. `model_generate` loops `while True` (line 391), and generation stops only at `max_new_tokens`. The adapter must cut the output at EOS or stop strings itself.
  - `model_generate` is a generator that yields one `mx.array` token per step (lines 384, 461). Streaming would mean iterating it directly and decoding incrementally. That is an undocumented internal API (UNVERIFIED on hardware).
  - `setup.py` `install_requires` (lines 30-42) does not list `mlx` or `psutil`, but the MLX module imports both (lines 11-15). A Mac install needs `mlx` added separately. The exact install recipe is UNVERIFIED because the README was not read.

### Implication
AirLLM is a single-node fallback for "model larger than RAM" on Mac, limited to Llama and Mistral. It is slow by design because it loads layers from disk one at a time. Keep it in the later phase as specified. Wrap it in a subprocess with explicit `max_new_tokens`, stop-string trimming and a model allowlist.

---

## C. Prior art for the MVP (llama.cpp RPC across machines)

| Project | License | Last push | Native Windows + macOS workers? | Engine / model | Recommendation |
|---|---|---|---|---|---|
| gpustack/gpustack @ 22732bd | Apache-2.0 | 2026-10-20 | **No.** README line 83: "Only Linux is supported for GPUStack worker nodes. If you use Windows, consider using WSL2 ... macOS is not supported for GPUStack worker nodes." | At HEAD, `gpustack/worker/backends/` has only vllm, sglang, ascend_mindie, vox_box and custom. `docs/migration.md` line 159: "llama-box is no longer supported as of v2.0.0. Use llama.cpp via the custom inference backend instead." | Reference only. The useful ideas are memory and offload estimation (fixtures such as `tests/fixtures/estimates/llama3_8b_partial_offload_split_1main_1rpc.json`) and a server-plus-worker registration model. |
| gpustack/llama-box | MIT | 2025-11-24 (**archived**) | n/a | Former llama.cpp-based server with RPC | Do not use (archived). |
| gpustack/gguf-parser-go | MIT | 2026-10-08 | Go CLI, cross-platform (UNVERIFIED: release assets not listed) | Estimates GGUF memory and throughput | Optional later: call it as an external CLI for fit estimates. The MVP can read GGUF metadata in Python instead. |
| mudler/LocalAI @ d1684af | MIT | 2026-10-21 | **No Windows binary.** Release v4.11.0 (2026-10-02) assets are darwin-arm64, linux-amd64 and linux-arm64 only. | Worker mode uses llama.cpp RPC: `local-ai worker p2p-llama-cpp-rpc --llama-cpp-args="-m <memory>"` (docs `features/distributed_inferencing.md` line 122) or `LLAMACPP_GRPC_SERVERS="address1:port,address2:port"` (line 99). Discovery uses libp2p + EdgeVPN with a shared token (lines 155-157). "The weights are split proportional to the memory" (line 159). Documented limits: "Only a single model is supported currently" and "additional workers cannot be added once inference has begun" (lines 172-173). Labeled "still experimental" (line 44). | Reference ideas only: token-based join, a "How to join" panel that prints the exact command, and a per-worker memory cap. No code reuse: it is Go, libp2p-heavy and has no Windows build. |
| intentee/paddler | Apache-2.0 | 2026-10-20 | No Windows binary in release v4.1.0 (Linux libc2.35/2.39 and macos-arm). | Balancer plus agents with a built-in llama.cpp engine (README lines 11, 45, 81). Load-balances whole requests across agents and does **not** split one model's layers. | Reference only, for the "agents join a balancer dynamically" UX. |
| berkkarahan/kluster-llm @ f9af17f | MIT | 2026-10-20 (created 2026-09-17, 1 star) | **Yes, per its README**: "Distributed LLM inference across Windows, Linux, and macOS ... through llama.cpp RPC". Backend preference is Metal > CUDA > ROCm > Vulkan > CPU. | Python. The master holds the GGUF and workers run the RPC server on port 50052. Its README warns: "Upstream RPC has no authentication. Reachability is enough to join inference. The HTTP API token does not protect this port." | Closest match to the MVP. Use it as a design reference. Copy code only after a file-level review, with MIT attribution. Code quality is UNVERIFIED: only the README and file tree were read, and the project is very young. |
| arseniy0924/rpc_manager | MIT | 2026-03-08 (stale) | Python/JS web UI. CUDA builds mentioned. | mDNS/Zeroconf discovery; deploys binaries to nodes. | Reference only, for the mDNS discovery idea. |
| Camper-code/ClusterKit | MIT | 2026-05-24 (created and last pushed the same day) | README says "macOS or Windows" | Go. LAN discovery on UDP 47777; worker RPC server on 50052. README claims a Qwen 3.5 9B Q6 run at about 9 tok/s solo, decaying, vs about 3.5 tok/s steady on 2 nodes (unverified). | Reference only. |
| cramt/llama-mesh | **none** (all rights reserved) | 2026-07-22 | Rust | Workers join and leave over WebSocket. | Do not copy code. Idea reference only. |
| kjaiswal/llama-cpp-distributed-benchmarks | MIT | 2026-03-30 | n/a (benchmarks) | Real llama.cpp RPC runs with a Mac Studio M2 Ultra (Metal, llama-server host) and a DGX Spark GB10 (CUDA 13, RPC server) over 10GbE. Qwen2.5-7B-Instruct Q4_K_M: local Metal 76.1 prompt / 91.8 gen tok/s vs RPC 317.7 / 52.7. README line 52: "RPC's real value isn't speed — it's capacity". Lines 89, 93: build the same llama.cpp commit on both machines. | Cite as an expectation baseline, including for the GB10/GX10 phase. Do not reuse code. |
| b4rtaz/distributed-llama | MIT | 2026-10-14 | Own engine, not llama.cpp | | Reference only. |
| evilsocket/cake | "FAIR License (Free Access for Individuals and Researchers) Version 1.0.0" (not OSI; GitHub reports NOASSERTION) | 2026-04-24 | Own Rust engine | | Do not reuse code. |

Search queries used: `gh search repos` for "llama.cpp rpc distributed", "llama.cpp rpc-server", "rpc-server llama cluster", "distributed llama.cpp inference", "llama.cpp rpc worker" and "llama.cpp rpc" (sorted by stars). Every hit with more than about 5 stars that orchestrates llama.cpp RPC is listed above. Single-purpose guides (Strix Halo, Jetson, Android, Kubernetes Helm) were skipped.

**Verdict.** No widely adopted, maintained project gives native Windows + macOS workers, LAN join and llama.cpp RPC layer splitting together. GPUStack (Linux-only workers, llama-box dropped) and LocalAI (no Windows build, workers fixed once inference starts) both miss the MVP's core need. Hypercompute should:
- build its own thin orchestrator;
- reuse only the unmodified upstream llama.cpp binaries (MIT);
- take ideas from LocalAI (token join, memory-proportional split), kluster-llm (the unauthenticated RPC port warning, backend preference order), ClusterKit and rpc_manager (LAN discovery) and GPUStack/gguf-parser (fit estimation).

---

## llama.cpp facts the MVP depends on (ggml-org/llama.cpp @ 339f3242c761f16c27e5ffcde1d35dcc31933a55, pushed 2026-10-21)

- License: MIT, "Copyright (c) 2023-2026 The ggml authors" (`LICENSE` lines 1-3).
- **The RPC server binary was renamed.** `tools/rpc/CMakeLists.txt` line 1 now reads `set(TARGET ggml-rpc-server)`. The rename came in commit c299a92c38 (2026-06-27), "binaries : Improve rpc-server and export-graph-ops names. (#25045)". Older builds ship `rpc-server`. Hypercompute should look for `ggml-rpc-server[.exe]` first and fall back to `rpc-server[.exe]`.
- `tools/rpc/README.md`:
  - lines 4-5: "proof-of-concept development stage ... fragile and insecure. **Never run the RPC server on an open network or in a sensitive environment!**"
  - Default endpoint `127.0.0.1:50052`. Example: `ggml-rpc-server --device CUDA0 -p 50052` (lines 62, 70-71).
  - Main host: `llama-server ... -ngl 99 --rpc host1:50052,host2:50052`. Weights and the KV cache are split "in proportion to each device's available memory", and `--tensor-split` overrides that (lines 83-84).
  - `-c` enables a local tensor cache in `$HOME/.cache/llama.cpp/rpc`, which `LLAMA_CACHE` can override (lines 90-96).
  - RDMA is negotiated automatically; `GGML_RPC_NO_RDMA=1` forces TCP. `GGML_RPC_DEBUG` controls log verbosity.
- Prebuilt binaries:
  - Numbered builds (`bNNNNN`) are now published as **prereleases**. The newest is b11770 (2026-10-21). The "latest" release is `v0.7.0` (2026-10-15), whose only asset is `nightly-tag.txt`. A downloader that uses `releases/latest` will find no binaries. It must pin a `bNNNNN` tag.
  - b11770 assets include `llama-b11770-bin-macos-arm64.tar.gz`, `llama-b11770-bin-win-cuda-12.4-x64.zip`, `llama-b11770-bin-win-cuda-13.4-x64.zip` (plus matching `cudart-...` zips), `llama-b11770-bin-win-vulkan-x64.zip` and `llama-b11770-bin-win-cpu-x64.zip`.
  - `.github/workflows/release.yml` line 38 sets the shared `CMAKE_ARGS` with `-DGGML_RPC=ON`. The `macos-cpu` (line 151) and `windows-cpu` (line 919) jobs use it.
  - The `windows-cuda` job builds only `ggml-cuda.dll`. Its comment (lines 1191-1192) says "llama-server is injected from the windows-cpu zip during the release 'Merge artifacts' step". So `ggml-rpc-server.exe` should be inside the win-cuda zip. This is UNVERIFIED: no zip was downloaded, and the merge step lives in another workflow file.
- Split GGUF loading: `src/llama-model-loader.cpp` reads `LLM_KV_SPLIT_COUNT` and enumerates the parts with `llama_split_path` (lines 83-101, 596-611). Pass the `-00001-of-0000N.gguf` path; the other parts must sit next to it.
- Every peer needs the same llama.cpp build. The kjaiswal README (lines 89, 93) states this. The exact handshake and version-check behavior in `ggml-rpc` is UNVERIFIED because the C++ source was not read. Hypercompute should pin one `bNNNNN` tag for all platforms.

---

## Licenses for docs/ATTRIBUTION.md

| Component | License | Evidence |
|---|---|---|
| llama.cpp / ggml (ggml-org/llama.cpp) | MIT | `LICENSE` @ 339f324, "Copyright (c) 2023-2026 The ggml authors" |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF | Apache-2.0 | HF API tags include `license:apache-2.0`. `cardData.license = apache-2.0`, license link `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/LICENSE`. Repo sha `9217f5db79a29953eb74d5343926648285ec7e67`. `gated: false`. |
| Qwen/Qwen2.5-7B-Instruct-GGUF | Apache-2.0 | HF API tags include `license:apache-2.0`. `cardData.license = apache-2.0`. Repo sha `bb5d59e06d9551d752d08b292a50eb208b07ab1f`. `gated: false`. |
| EXO (later phase) | Apache-2.0 | `LICENSE` @ bc412e2 |
| AirLLM (later phase) | Apache-2.0 | `LICENSE` and `air_llm/LICENSE` @ bc47e0d |

Model files from the HF tree API, for downloader pinning and checksums:
| File | Bytes | sha256 (LFS oid) |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491400032 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3993201344 | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689872288 | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |

The 7B Q4_K_M is **split into two files**. The downloader must fetch both and give llama-server the `-00001-of-00002` path.

Suggested attribution text:
```
This software runs unmodified llama.cpp binaries (https://github.com/ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml authors.
Default models: Qwen2.5-0.5B-Instruct-GGUF and Qwen2.5-7B-Instruct-GGUF by the Qwen team (Alibaba Cloud), Apache License 2.0.
Later-phase engines (optional): EXO (exo-explore/exo), Apache License 2.0; AirLLM (lyogavin/airllm), Apache License 2.0.
```

---

## Addendum (critic)

Critic pass 2026-10-20. The llama.cpp items below were re-checked at the project pin `b11770` =
`39da130fcd1b6ca47d3c0522a40eccff41ea3dfb` with `gh api -H "Accept: application/vnd.github.raw" ...`. Nothing was
downloaded, built or run.

### Open items in this file that matter for the llama.cpp gateway

| Item | Matters for | Status after this pass |
|---|---|---|
| The "llama.cpp facts" section is pinned to `339f3242...` (master, 2026-10-21), not to the project pin `39da130` (b11770) used by the other three audits | all | Ref mismatch. The rename was re-checked at `39da130`: `tools/rpc/CMakeLists.txt:1` is `set(TARGET ggml-rpc-server)`, and `tools/CMakeLists.txt:32-33` builds it only `if (GGML_RPC)`. Cite b11770 for anything the gateway depends on |
| "The exact handshake and version-check behavior in `ggml-rpc` is UNVERIFIED" | (a), (d) | Answered in llama_rpc.md §2. The client rejects `major != 8 \|\| minor > 0`, the patch number is ignored, and the server never checks the client's version. So "same tag everywhere" must be enforced by Hypercompute |
| "`ggml-rpc-server.exe` should be inside the win-cuda zip ... UNVERIFIED" | worker launch on Windows/NVIDIA | Strongly INFERRED from both workflow files (below). Still UNVERIFIED by listing the actual asset, because that needs a download the user has not approved |
| "release assets not listed" for gguf-parser-go | optional | Not needed for the MVP. Not pursued |

### Windows release zip composition (INFERRED from workflow source at `39da130`)

- The `windows-cpu` job builds with `-DGGML_BACKEND_DL=ON` plus the shared `CMAKE_ARGS`
  (`-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON`; `.github/workflows/release.yml:38`, `915`). It then
  zips the whole output directory: `7z a -snl llama-bin-win-cpu-${{ matrix.arch }}.zip .\build\bin\Release\*` (925).
- The `windows-cuda` job builds only the `ggml-cuda` target and zips only `ggml-cuda.dll` (release.yml:1248, 1253). The
  CUDA runtime goes into a separate zip, `cudart-llama-bin-win-cuda-<ver>-x64.zip`, containing `cudart64_*.dll`,
  `cublas64_*.dll` and `cublasLt64_*.dll` (1261-1269).
- `release-publish.yml:77-104` ("Merge artifacts") unzips the windows-cpu package and adds its full contents to every
  other `llama-bin-win-*-<arch>.zip`, with this comment: "the windows-cpu zip contains the full toolset (llama-server
  with the embedded UI, ggml-cpu) - inject it into the other windows zips".
- Expected contents of `llama-b11770-bin-win-cuda-<ver>-x64.zip`: `llama-server.exe`, `ggml-rpc-server.exe`,
  `ggml-rpc.dll`, the `ggml-cpu-*.dll` variants and `ggml-cuda.dll`. The `cudart-...` zip must be extracted **into the
  same folder**. If it is missing, `ggml-cuda.dll` fails to load without any error in release builds, and
  `ggml-rpc-server` quietly serves the CPU (llama_placement.md "Addendum (critic)" P1). The installer should check
  that every file above is present and should verify the worker's startup `Devices:` block names `CUDA0`.
- The macOS arm64 tarball is not a `GGML_BACKEND_DL` build: backends are linked in, not discovered at runtime, and Metal is embedded (release.yml:103-106). It needs no backend library
  co-location.
