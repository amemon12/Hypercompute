# vLLM as a Hypercompute inference engine (audit of v0.32.0)

Audited 2026-10-20 (UTC). Read-only. No vLLM, plugin, uv or PyTorch code was installed, built or run.

Pinned sources:

- `vllm-project/vllm` tag `v0.32.0` = commit `5df63a7203cdb9328d824a27b682e77c7b9928be`. This is the latest release, published
  2026-10-18 (`gh api repos/vllm-project/vllm/releases/latest`). License Apache-2.0.
- `vllm-project/vllm-metal` tag `v0.32.0` = commit `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`. Apache-2.0.
- `vllm-project/vllm-gguf-plugin` `main` = `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. PyPI latest is `0.0.5`, uploaded 2026-08-10. Apache-2.0.
- `pytorch/pytorch` tag `v2.13.0` (`cf30153c4c13`). vLLM 0.32.0 pins `torch==2.13.0`.
- `astral-sh/uv` `main` @ `dce159bf3c49` (for `--torch-backend` detection).
- The PyPI JSON API was used for `vllm`, `torch` and `vllm-gguf-plugin`. The Hugging Face API was used for checkpoint sizes.
- Official docs fetched: NVIDIA CUDA on WSL user guide, Microsoft WSL networking and file-system docs, uv PyTorch guide.

Files were read with `gh api -H "Accept: application/vnd.github.raw" "repos/OWNER/REPO/contents/PATH?ref=REF"`. A citation
`path:Lx` refers to the file at the pinned ref. "Release notes" means the GitHub release body of v0.32.0.

Status words:

- VERIFIED: read directly in source, docs or API output at the stated ref.
- INFERRED: follows from the code, but was not exercised on hardware.
- UNVERIFIED: not confirmed, with the reason.

Local facts used (VERIFIED with read-only commands on the host, 2026-10-20):

- `nvidia-smi`: `NVIDIA GeForce RTX 5070 Ti Laptop GPU, 591.84, 12227 MiB total, compute_cap 12.0`.
- At audit time 11214 MiB was in use and 731 MiB was free, held by Hypercompute's own
  `llama-b11770-windows-x64-cuda13.4\llama-server.exe` (pid 68572).
- `wsl -l -v`: distro `Ubuntu`, WSL version 2, state Stopped. I did not start the VM, so the Ubuntu release (24.04 per the task
  brief) was not checked.
- Hypercompute's measured llama.cpp baseline (`docs/RESULTS.md`, `final-7b-local`): Qwen2.5-7B-Instruct Q4_K_M runs fully on
  CUDA0 (28 layers, 4168 MiB) at **80.6 tok/s decode** (short prompt) and 86.2 tok/s (long prompt), single stream, native Windows.

---

## 0. TL;DR for Hypercompute

1. **vLLM runs on Linux only.** The docs say "OS: Linux" and "vLLM does not support Windows natively". On Windows you run it
   through WSL2 (`docs/getting_started/installation/gpu.md:L27,L31`). vLLM detects WSL and changes behaviour there: pinned memory is
   off by default, the multiprocessing start method is `spawn`, and it warns that `--cpu-offload-gb` combined with CUDA graphs can
   hang the Windows host.
2. **Blackwell sm_120 (RTX 50-series) is supported out of the box in 0.32.0. No cu128 index is needed.**
   - The PyPI wheel is CUDA 13.0 (release notes L21).
   - PyPI `torch==2.13.0` depends on `cuda-toolkit==13.0.3`, and its CUDA 13.0 x86_64 wheel includes compute capability 120.
   - vLLM's CMake builds `12.0f` family kernels for Marlin, FP8 scaled-mm and NVFP4.
   - A `+cu129` wheel is attached to the release as an alternative. There is no `+cu128` asset for 0.32.0.
   - Driver 591.84 is newer than the R580 that CUDA 13 needs.
3. **macOS needs a separate plugin.**
   - Upstream vLLM on macOS is an experimental CPU build (FP32/FP16).
   - GPU on Apple Silicon is the separate **vllm-metal** plugin, which uses MLX as its backend.
4. **A Mac cannot join a CUDA vLLM cluster.**
   - Ray placement uses a single device key taken from the driver's platform: `"GPU"` for CUDA, `"mlx"` for vllm-metal, none for CPU.
   - The collective backends differ: NCCL for CUDA, the MLX ring for Metal.
   - vllm-metal's own pipeline parallelism (PP) is Mac-to-Mac only, over Thunderbolt. Its docs say Wi-Fi and Ethernet are too slow.
   - **Cross-vendor layer splits (CUDA laptop + M4 Mac) stay with llama.cpp RPC.**
5. **GGUF is a second-class format in vLLM.**
   - Support moved out of tree to `vllm-gguf-plugin`, which the docs call "highly experimental and under-optimized".
   - Sharded GGUF is not supported on vllm-metal.
   - For vLLM, use **`Qwen/Qwen2.5-7B-Instruct-AWQ`**: 5.19 GiB of safetensors, 4-bit, group size 128. It loads through Marlin on sm_120.
6. **Readiness = `GET /health` returns 200.**
   - The HTTP socket is bound before the engine starts but is only served once the model is loaded, the KV cache is allocated and
     warmup is done. `/health` returns 503 once the engine is dead.
   - `/health` is not behind `--api-key`.
   - SIGTERM or SIGINT triggers a clean shutdown. The default `--shutdown-timeout 0` aborts in-flight requests.
7. **Memory flag trap on a shared laptop GPU.**
   - `--gpu-memory-utilization` defaults to **0.92** of *total* memory, and startup fails if *free* memory is below that.
   - On the 12 GiB laptop that means 11,249 MiB must be free. Windows display use, or a running llama-server, will break it.
   - Pass about 0.80-0.85 (computed from free memory) plus `--max-model-len 8192..32768`.
8. **Expected single-stream decode is about the same as llama.cpp.**
   - On this GPU, AWQ 7B and Q4_K_M read almost the same bytes per token (about 4.5 vs 4.4 GB). Expect roughly 70-95 tok/s against
     llama.cpp's measured 80.6 (UNVERIFIED, must be measured).
   - The only third-party side-by-side found (Qwen3.8-27B INT4 vs UD-Q4_K_M, vLLM 0.31.0) shows vLLM about 10% faster
     single-stream on RTX 4090 and 5090. The large gains are under concurrency: 936 vs 369 aggregate tok/s at 32 requests on an RTX 5090.
9. **Where vLLM helps.** Concurrent serving on a single NVIDIA Linux box where the model fits (RTX 5090 workstation, GX10/GB10,
   the G14 under WSL2 for multi-user demos), and PP across several NVIDIA Linux nodes. It does **not** help the
   heterogeneous Windows+Mac pair, or single-user speed on one laptop.

---

## 1. Platform support

### 1.1 Linux, Windows, WSL2 (VERIFIED)

- `docs/getting_started/installation/gpu.md:L25-28`: "Requirements / - OS: Linux / - Python: 3.11 -- 3.14".
- `gpu.md:L31`: "vLLM does not support Windows natively. To run vLLM on Windows, you can use the Windows Subsystem for Linux (WSL)
  with a compatible Linux distribution, or use some community-maintained forks, e.g. https://github.com/SystemPanic/vllm-windows".
  That fork was not audited.
- `setup.py:L102-109`: on any other OS the build logs "vLLM only supports Linux platform (including WSL) and MacOS." and sets
  `VLLM_TARGET_DEVICE = "empty"`.
- `pyproject.toml:L34`: `requires-python = ">=3.11,<3.15"`. Release notes: "Python 3.10 is no longer supported".
  Ubuntu 24.04 ships Python 3.12, which is in range. The docs' own example uses `uv venv --python 3.12`
  (`python_env_setup.inc.md:L5`).
- `gpu.cuda.inc.md:L9`: CUDA GPUs need "compute capability 7.5 or higher".

WSL2-specific code paths at v0.32.0 (VERIFIED):

- WSL detection is `"microsoft" in " ".join(platform.uname()).lower()` (`vllm/platforms/interface.py:L65-68`).
- **Pinned memory is off by default in WSL2.** `vllm/platforms/cuda.py:L301-320` returns False for kernels below 4.19.121
  (it logs "Please run `wsl --update`"). On newer kernels it returns `envs.VLLM_WSL2_ENABLE_PIN_MEMORY`. `vllm/envs.py:L2178-2184`
  explains why: "On WSL2 with a compatible kernel (>= 4.19.121), pinned memory is supported but disabled by default due to a small
  performance regression. Set to 1 when pinned memory or UVA is required (e.g. CPU offloading or v2 model runner)."
- **`spawn` is forced.** `vllm/utils/system_utils.py:L149-150` adds the reason "WSL is detected and NVML is not compatible with
  fork" and overrides `VLLM_WORKER_MULTIPROC_METHOD` to `spawn`.
- **Host-hang warning.** `vllm/platforms/cuda.py:L345-364` warns when `--cpu-offload-gb > 0` is combined with CUDA graphs under WSL2:
  "Windows (WDDM) enforces a hard system-wide cap of roughly 50% of physical RAM on pinned memory ... can cause the entire host OS to
  become unresponsive and may require a hard reboot". **Never use `--cpu-offload-gb` on the G14 under WSL2.**
- Release notes L58: "non-blocking H2D copies on WSL2 (#60215)" (Model Runner V2).

NVIDIA's CUDA on WSL guide (https://docs.nvidia.com/cuda/wsl-user-guide/index.html, VERIFIED):

- "Do not install any Linux display driver in WSL."
- "The CUDA driver installed on Windows host will be stubbed inside the WSL 2 as libcuda.so".
- Known limitations (section 5.1):
  - pinned memory "availability for applications is limited"
  - unified memory: "WSL 2 will not support it for the foreseeable future"
  - "NVML (nvidia-smi) does not support all the queries yet"

Microsoft WSL docs (VERIFIED):

- https://learn.microsoft.com/en-us/windows/wsl/filesystems: "For the fastest performance speed, store your files in the WSL file
  system". Keep the HF cache under `/home/...`, not `/mnt/c/...`.
- https://learn.microsoft.com/en-us/windows/wsl/networking:
  - In the default NAT mode, Windows apps reach Linux servers via `localhost`.
  - LAN access to WSL2 needs `netsh interface portproxy` or `networkingMode=mirrored` (Windows 11 22H2+), plus a Hyper-V firewall rule.
  - This matters only if a *remote* node must reach the vLLM in WSL (section 4).

### 1.2 macOS / Apple Silicon (VERIFIED)

- Installation README `docs/getting_started/installation/README.md:L9`: "Apple Silicon (via vLLM-Metal)".
- Upstream CPU on Mac, `cpu.apple.inc.md`:
  - L4: "vLLM has experimental support for macOS with Apple Silicon. For now, users must build from source".
  - L6: "the CPU implementation for macOS supports FP32 and FP16 datatypes".
  - L43: "On macOS the `VLLM_TARGET_DEVICE` is automatically set to `cpu`, which is currently the only supported device".
  - Doc/asset conflict: L28 says "there are no pre-built Apple silicon CPU wheels", but the v0.32.0 release assets include
    `vllm-0.32.0+cpu-cp38-abi3-macosx_11_0_arm64.whl` (29.9 MB), and release notes L44 list "CPU Python wheels for x86_64, arm64,
    and macOS". The docs are stale. **Not useful for Hypercompute either way:** FP32/FP16 CPU decoding of a 7B model on an M4 Air
    would be far slower than Metal.
- GPU on Mac, `gpu.apple.inc.md:L4`: "use vLLM-Metal, a community-maintained hardware plugin that uses MLX as the compute backend".
- vllm-metal `README.md`:
  - L36: "macOS 15 (Sequoia) or later, on Apple Silicon"
  - L30: "Upstream vLLM supplies the API server, scheduler, and paged block manager; mlx_lm supplies the token-wise model layers"
  - Install: Homebrew (`brew tap vllm-project/vllm-metal ...; brew install vllm-project/vllm-metal/vllm-metal`) or the curl
    installer (`docs/installation.md:L41`: "`pip install vllm-metal` is not supported").
- Model support (vllm-metal `docs/supported_models.md`):
  - Llama 3, Mistral-7B and Qwen3 are supported (L137-146).
  - HF AWQ is "Verified for Qwen2.5, Llama 3, and Mistral" (L104-111).
  - mlx-community 4-bit checkpoints are preferred (`gpu.apple.inc.md:L39-41`).
  - Mac-side candidate: `mlx-community/Qwen2.5-7B-Instruct-4bit`, 3.99 GiB (HF API).
- 16 GB recipe (vllm-metal `docs/recipes.md:L29-42`): `vllm serve mlx-community/Qwen3.5-9B-MLX-4bit --host 127.0.0.1
  --max-model-len 65536 --max-num-batched-tokens 2048 ...`, with the note "Without it this model does not fit on 16 GB".
- Startup on Mac: `gpu.apple.inc.md:L55` says "Wait for 2-3 minutes until you see application startup complete".

### 1.3 Blackwell sm_120 (RTX 50-series): versions and wheels (VERIFIED unless marked)

| Item | Evidence |
|---|---|
| Default PyPI vLLM wheel is CUDA 13.0 | Release notes L21: `PyPI (CUDA 13.0) / pip install vllm`. PyPI `vllm==0.32.0` files are `vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl` (326.5 MB) and the aarch64 equivalent. `requires_dist` includes `nvidia-cutlass-dsl[cu13]==4.8.0`. |
| Alternative CUDA 12.9 wheel | Release assets: `vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` (559.4 MB). Docker `vllm/vllm-openai:v0.32.0-cu129`. **No `+cu128` asset exists for 0.32.0.** |
| Docs are stale on the default CUDA | `gpu.cuda.inc.md:L4` "pre-compiled C++ and CUDA (12.9) binaries" and L53 "compiled with CUDA 12.9 ... by default" contradict the release notes and the PyPI metadata. Trust the release notes and PyPI. |
| torch pin | `requirements/cuda.txt:L7` `torch==2.13.0`, `pyproject.toml:L10`, `CMakeLists.txt:L71` `TORCH_SUPPORTED_VERSION_CUDA "2.13.0"`. |
| torch 2.13.0 PyPI wheel is CUDA 13.0 | PyPI `torch==2.13.0` `requires_dist`: `cuda-toolkit[...]==13.0.3; platform_system == "Linux"`, `nvidia-nccl-cu13==2.29.7`. |
| torch 2.13 cu130 includes sm_120 | `pytorch .ci/manywheel/build_env_setup.py:L71-84`: `"13.0": {"x86_64": {75, 80, 86, 90, 100, 120}, ...}`. |
| vLLM kernels for sm_120 | `CMakeLists.txt:L121-126` CUDA ≥ 13.0 arch list `"7.5;...;12.0"`. Marlin `L615-637` (`12.0f` with CUDA ≥ 13.0, else `12.0a;12.1a`), comment "fp8 computation for SM89 (e.g. RTX 40x0) and 12.0 (e.g. RTX 50x0)". CUTLASS FP8 `scaled_mm_c3x_sm120` `L821-838`. NVFP4 SM12x `L995-1013`. |
| Blackwell minimum CUDA | `gpu.cuda.inc.md:L51`: "NVIDIA Blackwell GPUs (B200, GB200) require a minimum of CUDA 12.8". |
| Driver | `gpu.cuda.inc.md:L340`: "CUDA 13 requires an R580 or newer driver". The host has 591.84. |
| FlashInfer kernels | `gpu.cuda.inc.md:L36-42`: without `vllm download-kernels`, "vLLM downloads and compiles kernels at startup, which can add several minutes on Hopper and newer GPUs, especially Blackwell". |
| SM120 work in 0.32.0 | Release notes L75 "SM120 split-K attention (#58482)". L8 "NVFP4 KV cache now runs on SM8x and SM12x". L82 "ModelOpt IQ2_XS/IQ2_XXS/Q8_0 on SM120/SM121". |

So the old "nightly + cu128 index" recipe for RTX 50-series is obsolete. A plain install gets cu130 torch and a cu130 vLLM.

### 1.4 Install commands for vLLM 0.32.0 in WSL2 Ubuntu 24.04 (documentation only, NOT executed)

These come from `gpu.cuda.inc.md:L24-27,L48` and `python_env_setup.inc.md:L4-7`. The pinned version and explicit backend are my choices.

```bash
# inside WSL2 (Ubuntu 24.04), in the Linux file system, never /mnt/c
# prerequisite: uv (https://docs.astral.sh/uv/). Do NOT install any NVIDIA driver inside WSL.
nvidia-smi                                      # sanity: should list the RTX 5070 Ti via the Windows driver
uv venv ~/.venvs/vllm-0.32.0 --python 3.12 --seed --managed-python
source ~/.venvs/vllm-0.32.0/bin/activate
uv pip install "vllm==0.32.0" --torch-backend=cu130    # docs use --torch-backend=auto; explicit is safer under WSL
vllm download-kernels                           # FlashInfer precompiled kernels; avoids minutes of JIT on Blackwell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 2.13.0+cu130 13.0 (12, 0)     <- UNVERIFIED until run
```

- `--torch-backend=auto` under WSL (INFERRED). uv probes `/sys/module/nvidia/version`, then `/proc/driver/nvidia/version`, then
  `nvidia-smi` (`uv crates/uv-torch/src/accelerator.rs:L74-78`). WSL has only the `nvidia-smi` route, so `auto` probably works.
  Pass `cu130` explicitly to remove the guesswork. `UV_CUDA_DRIVER_VERSION` overrides detection (same file, L70-75).
- pip-only alternative from the docs: `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129` (L31-33). That
  is the cu129 route and needs the matching `+cu129` vLLM wheel. With plain `pip install vllm==0.32.0` the default cu130 torch from
  PyPI matches the cu130 vLLM wheel.
- Do not install vLLM into the Windows Python or into Hypercompute's own venv (harness rule: keep environments separate). The WSL
  venv is the engine's environment.

---

## 2. Model formats

### 2.1 GGUF (VERIFIED)

- Core docs `docs/features/quantization/gguf.md`:
  - L3-4: "GGUF support in vLLM is highly experimental and under-optimized at the moment, it might be incompatible with other features".
  - L6-7: "GGUF support has migrated to OOT vllm-gguf-plugin. Make sure you have GGUF plugin installed before serving a GGUF model."
  - Install: `uv pip install vllm-gguf-plugin` (L12).
- Usage: `vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B` (L19), or a local `./file.gguf` (L34). The docs
  recommend the base-model tokenizer because "tokenizer conversion from GGUF is time-consuming and unstable" (L38).
- vllm-metal `docs/supported_models.md:L114-116`: "vLLM 0.24 moved its in-tree GGUF support to the CUDA/ROCm-only vllm-gguf-plugin".
- Plugin `README.md`:
  - L3-5: support was moved there "after in-tree support deprecation (vllm#39583)". Prerequisite "CUDA toolkit or ROCm toolkit" (L11).
  - Tested coverage (L85-95) includes "Qwen 2.5 | Q6_K", "Qwen 3 | Q8_0", Gemma 3/4, Qwen 3.5/3.6 VLM. L99-102: other archs "may work
    when their GGUF tensor names map to the corresponding Hugging Face model", with no guarantee.
- Plugin wheels on PyPI: `vllm_gguf_plugin-0.0.5-cp310-abi3-manylinux_2_28_{x86_64,aarch64}.whl` (6.4 MB). The release workflow
  builds the x86_64 wheel on `pytorch/manylinux2_28-builder:cuda13.0` with `--torch-backend cu130 "torch>=2.9"`
  (`.github/workflows/release.yml:L141-173`). The build script sets `TORCH_CUDA_ARCH_LIST="7.5;...;12.0"` for CUDA ≥ 13
  (`scripts/build_release_wheel.sh:L18-19`), so sm_120 kernels are included.
- Torch ABI (INFERRED). The wheel was uploaded 2026-08-10, when the newest torch on PyPI was 2.13.0 (released 2026-07-08; 2.14.0 came
  2026-09-02). It was therefore most likely built against torch 2.13.0, which matches vLLM 0.32.0. If the import fails with
  undefined symbols, rebuild from source with `--no-build-isolation` as the README says (L35-43). UNVERIFIED until run.
- Single file vs sharded:
  - Core plugin: UNVERIFIED (not read in `gguf_files.py`).
  - vllm-metal is explicit (`docs/gguf.md`): "downloads exactly one matching `.gguf` file" (L35); "quants published only as shards,
    fail before model load" (L37-39); unsupported "IQ and T-quants, MoE, SSM or hybrid models, vision models, fused-QKV GGUFs,
    sharded GGUF files, and LoRA" (L66-67).
  - vllm-metal supports families "Qwen2, Qwen3, Llama, and Mistral" (L48) and qtypes Q8_0, Q4_0/1, Q5_0/1, Q2_K..Q6_K, so llama.cpp
    Q4_K_M loads (L49-54).
  - vllm-metal PP "rejects AWQ and GGUF" (`docs/distributed.md:L161`).
- **Verdict.** Hypercompute's GGUF catalog does not carry over to vLLM cleanly. Use HF safetensors checkpoints (AWQ/GPTQ/compressed-tensors)
  for the vLLM engine and keep GGUF for llama.cpp.

### 2.2 AWQ / GPTQ / FP8 / NVFP4 on sm_120

- The docs' hardware matrix (`docs/features/quantization/README.md:L67-79`) has no Blackwell column. Its columns stop at Hopper (L79).
  It is not evidence either way for sm_120.
- Code evidence (VERIFIED):
  - AWQ checkpoints (`quant_method: "awq"`) are routed to `auto_awq` on any non-CPU platform when the user passes nothing or
    `awq/awq_marlin/auto_awq/marlin` (`vllm/model_executor/layers/quantization/auto_awq.py:L262-285`). `get_min_capability`
    returns 75 (L233-234).
  - Kernels are chosen by `choose_mp_linear_kernel`, Marlin first when `check_marlin_supported` passes (L300-332, L436-452). The log
    line `Using %s for AutoAWQMarlinLinearMethod` names the kernel. Marlin is compiled for `12.0f` (`CMakeLists.txt:L615-628`).
    **AWQ and GPTQ 4-bit run on sm_120 via Marlin** (INFERRED from the arch lists; the kernel name in the log confirms it at runtime).
  - FP8 W8A8: CUTLASS `scaled_mm_c3x_sm120` + `scaled_mm_sm120_fp8` + blockwise (`CMakeLists.txt:L821-838`) and Marlin FP8
    (`L630-637`). **FP8 is supported on sm_120.**
  - NVFP4: "Building SM12x NVFP4" (`CMakeLists.txt:L995-1013`).
- `docs/features/quantization/auto_awq.md:L3-4`: "The `AutoAWQ` library is deprecated", replaced by llm-compressor. This applies only to
  *creating* AWQ checkpoints; serving existing AWQ checkpoints is unaffected.

### 2.3 Recommended checkpoint for Qwen2.5-7B-Instruct on a 12 GiB GPU

Sizes come from the HF API (`/api/models/<id>?blobs=true`, summed `.safetensors`). Config from `config.json`.

| Checkpoint | Format | Size | Notes |
|---|---|---|---|
| **`Qwen/Qwen2.5-7B-Instruct-AWQ`** (sha `b250375`) | AWQ 4-bit, `group_size 128`, `zero_point true`, `version gemm` | **5.19 GiB** (2 files) | Official Qwen, Apache-2.0. Recommended. |
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (`e9c932a`) | GPTQ 4-bit | 5.19 GiB | Equivalent alternative. |
| `RedHatAI/Qwen2.5-7B-Instruct-quantized.w4a16` (`92e7345`) | compressed-tensors W4A16 | 5.17 GiB | llm-compressor lineage, the path vLLM now recommends. |
| `RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic` (`a4f1d54`) | FP8 W8A8 | 8.11 GiB | Fits, but leaves about 1-2 GiB for KV and reads about 1.8x more bytes per token. Slower single-stream. |
| `Qwen/Qwen2.5-7B-Instruct` | BF16 | 14.19 GiB | Does not fit. |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | MLX 4-bit | 3.99 GiB | For vllm-metal on the Mac, not the CUDA box. |

Why the AWQ file is 5.19 GiB, not about 4 GiB: `tie_word_embeddings: false` and `vocab_size: 152064`, `hidden_size: 3584`, so
`embed_tokens` and `lm_head` are each 152064 × 3584 × 2 B ≈ 1.09 GB in fp16. AWQ does not quantize them.

KV cache math (INFERRED from `config.json`: 28 layers, 4 KV heads, head_dim 3584/28 = 128, fp16):

- 28 × 2 × 4 × 128 × 2 B = **56 KiB per token**.
- 8192 tokens ≈ 0.44 GiB. 32768 tokens (`max_position_embeddings`) ≈ 1.75 GiB.

Budget on the G14 (INFERRED; the activation and CUDA-graph reservation is UNVERIFIED and is logged at startup as `Available KV cache memory`):

- Total 12227 MiB × 0.85 ≈ 10.15 GiB.
- Minus 5.19 GiB of weights and about 0.5-1 GiB of activations and graphs.
- Leaves about 4 GiB of KV, roughly 70k tokens. That is enough for `--max-model-len 32768` with a few concurrent sequences.

### 2.4 Memory flags (VERIFIED)

- `--gpu-memory-utilization`, default **0.92** (`vllm/config/cache.py:L102-111`): "The fraction of GPU memory to be used for the model
  executor ... This is a per-instance limit".
  - It is checked against **free** memory at startup. `vllm/v1/worker/utils.py:L724-726` computes
    `requested_memory = ceil(total_memory * gpu_memory_utilization)`. L761-770 raise
    `ValueError("Free memory on device ... on startup is less than desired GPU memory utilization ... Decrease GPU memory utilization or reduce GPU memory used by other processes.")`.
  - On a Windows laptop with the desktop on the dGPU, or a llama-server holding 11 GiB as at audit time, the default fails.
    **The adapter must compute `util = (free_MiB - 512) / total_MiB`, capped at 0.90, from `nvidia-smi` just before launch.**
- `--kv-cache-memory-bytes` (`cache.py:L257-264`): "Size of KV Cache per GPU in bytes ... (when not-None) ignores
  gpu_memory_utilization". Whether it also skips the startup free-memory check is UNVERIFIED: `request_memory` is still called in
  `gpu_worker.py:L535-537`.
- `--max-model-len` (`vllm/config/model.py:L245-257`) takes k/m/g suffixes, and "-1 or 'auto' -> Automatically choose the maximum
  model length that fits in GPU memory". Set it explicitly (8192-32768) so KV sizing is predictable.
- `--enforce-eager` (`model.py:L272-279`): "This disables both `torch.compile` and CUDA graphs, and is equivalent to setting
  `-cc.mode=none -cc.cudagraph_mode=none`". It saves startup time and graph memory but costs decode speed.
  `docs/configuration/conserving_memory.md:L48-76` also shows the softer `cudagraph_capture_sizes=[1, 2, 4, 8, 16]`.
- `-O0..-O3` (`docs/design/optimization_levels.md`): "`-O0`: No optimization. Fastest startup time, but lowest performance."
  `-O2` is the default. `-O1` is "Fast optimization ... PIECEWISE cudagraphs".
- `--max-num-seqs` limits the batch (`conserving_memory.md:L39-46`).
- `--generation-config` defaults to `"auto"` (`model.py:L354-360`), which applies the checkpoint's `generation_config.json`. For
  Qwen2.5-7B-Instruct-AWQ that is `temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05` (HF raw file). Pass
  `--generation-config vllm` if Hypercompute wants llama.cpp-comparable defaults. Explicit request fields override either way.

---

## 3. Server

### 3.1 CLI (VERIFIED in `vllm/engine/arg_utils.py` and `vllm/entrypoints/launchers/cli_args.py`)

| Flag | Default | Source |
|---|---|---|
| `--host` | `None` (binds all interfaces, shown as `0.0.0.0`) | `cli_args.py:L283`, `launcher.py:L409` |
| `--port` | `8000` | `cli_args.py:L285` |
| `--api-key` / `VLLM_API_KEY` | none | `cli_args.py:L325` |
| `--served-model-name` | model id | `arg_utils.py:L1006` |
| `--gpu-memory-utilization` | 0.92 | `cache.py:L102` |
| `--max-model-len` | from model config | `arg_utils.py:L966` |
| `--enforce-eager` | False | `arg_utils.py:L975` |
| `--quantization`/`-q` | auto from checkpoint | `arg_utils.py:L967` |
| `--max-num-seqs` | set in `create_engine_config` | `arg_utils.py:L1717` |
| `--tensor-parallel-size`/`-tp`, `--pipeline-parallel-size`, `--distributed-executor-backend` | 1, 1, auto | `arg_utils.py:L1135-1177` |
| `--cpu-offload-gb`, `--offload-group-size` | 0 | `arg_utils.py:L1433-1438` |
| `--kv-offloading-size`, `--kv-offloading-backend` | None, `native` | `arg_utils.py:L1416-1419` |
| `--shutdown-timeout` | 0 ("0 = abort, >0 = wait") | `arg_utils.py:L1902-1905` |
| `--stream-interval` | 1 token | `vllm/config/scheduler.py:L213-217` |
| `--sse-keep-alive-interval` | 0 (off) | `cli_args.py:L156` |
| `--allowed-origins` | empty ("Cross-origin browser access is disabled by default") | `cli_args.py:L319-320` |

### 3.2 OpenAI-compatible streaming (VERIFIED)

- Route: `POST /v1/chat/completions` (`vllm/entrypoints/openai/chat_completion/api_router.py:L40-80`). Streaming returns
  `StreamingResponse(media_type="text/event-stream")` wrapped in `with_sse_keep_alive`.
- Framing (`chat_completion/serving.py`):
  - Each chunk is `yield f"data: {data}\n\n"`.
  - The stream always ends with `yield "data: [DONE]\n\n"` (L933).
  - Exceptions mid-stream become a `data: <error json>` event *before* `[DONE]` (L926-931). An HTTP 200 stream can therefore carry an
    error object, and the parser must check each payload for `"error"`.
- With `stream_options: {"include_usage": true}`, a final usage chunk is emitted (L839-893).
- Keep-alive: when `--sse-keep-alive-interval > 0`, idle periods emit the SSE comment `": keep-alive\n\n"`
  (`vllm/entrypoints/serve/utils/sse_keep_alive.py:L18`). The parser must ignore lines starting with `:`.
- Cancellation: `@with_cancellation` (`api_router.py:L51`; `vllm/entrypoints/serve/utils/api_utils.py:L51-60`) "allows a route handler
  to be cancelled by client disconnections". **To cancel, close the upstream HTTP connection.**
- **No llama.cpp-style `timings` object.** For decode tok/s, use `usage.completion_tokens` plus client-side first-token and last-token
  timestamps, or scrape Prometheus `/metrics` (`vllm/v1/metrics/loggers.py`):
  - `vllm:inter_token_latency_seconds` (L941)
  - `vllm:time_to_first_token_seconds` (L931)
  - `vllm:request_time_per_output_token_seconds` (L951)
  - `vllm:generation_tokens` (L799)
  - `vllm:kv_cache_usage_perc` (L620)
- Model name: requests must use the served name (`--served-model-name`). Env `VLLM_SKIP_MODEL_NAME_VALIDATION` exists (`envs.py:L322-323`).
- Auth caveat:
  - `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")` (`vllm/entrypoints/serve/middleware/authenticate.py:L11`).
  - `docs/serving/online_serving/openai_compatible_server.md:L5-14` warns that other endpoints such as `/invocations` are **not**
    authenticated: "Do not rely on `--api-key` alone".
  - Bind to `127.0.0.1` inside WSL. The default NAT localhost relay lets the Windows gateway reach it.

### 3.3 `/health` and readiness (VERIFIED)

- `vllm/entrypoints/serve/instrumentator/health.py:L22-33`: `GET /health` awaits `client.check_health()`. It returns 200, or **503 on
  `EngineDeadError`**. The body is empty.
- Startup ordering (`vllm/entrypoints/launchers/api_server/entry.py:L165-206`, `launcher.py:L378-411`, `L335-350`):
  1. `setup_server` creates the socket and **only `bind()`s it**. The comment reads "workaround to make sure that we bind the port
     before the engine is set up. This avoids race conditions with ray" (L391-398). There is no `listen()`.
  2. `build_async_engine_client(...)` loads the model, profiles memory, allocates the KV cache and captures graphs.
  3. `build_and_serve` logs `Supported tasks: ...` (L136) and `Starting vLLM server on <addr>` (L140), then `serve_http` starts uvicorn,
     which listens.
  - So **any HTTP 200 from `/health` means the engine is fully initialized.** Before step 3 a Linux client gets connection refused
    (INFERRED from bind without listen). Port conflicts fail fast at step 1.
  - Through the WSL2 localhost relay, the pre-listen behaviour seen from Windows may be a reset or a hang rather than a refusal
    (UNVERIFIED). Use short per-probe timeouts.
- Useful log milestones for a startup breakdown:
  - `Model loading took %s GiB memory and %.6f seconds` (`vllm/v1/worker/gpu_model_runner.py:L5150`)
  - `Available KV cache memory: %s GiB` (`gpu_worker.py:L759-761`)
  - `Graph capturing finished in %.0f secs, took %.2f GiB` (`gpu_model_runner.py:L6582`)
  - `init engine (profile, create kv cache, warmup model) took %.2f s` (`vllm/v1/engine/core.py:L403`)
  - uvicorn's `Application startup complete.` (also quoted in `gpu.apple.inc.md:L56`)
- Engine death after ready: the `watchdog_loop` checks every 5 s (`launcher.py:L313-317`) and exits the server when the engine has
  errored, unless `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` is set (L331). A dead engine therefore shows up as 503 and then process exit.

### 3.4 Startup time (mixed)

- Cold first start is slow:
  - Without `vllm download-kernels`, FlashInfer JIT "can add several minutes on Hopper and newer GPUs, especially Blackwell"
    (`gpu.cuda.inc.md:L36-42`, VERIFIED).
  - torch.compile artifacts are cached in `~/.cache/vllm/torch_compile_cache/...`. The docs say to copy that directory "to save a
    great amount of compilation time" (`docs/design/torch_compile.md:L10-18`, VERIFIED).
- Release notes highlight `vllm preload --preload-autotune` to keep the FlashInfer autotune table warm across restarts (L10, VERIFIED;
  not studied further).
- Expected numbers for a 7B AWQ on the G14 under WSL2 are **UNVERIFIED** (no hardware run):
  - Warm cache: roughly 30-90 s (model load of about 5 GB from ext4, KV profile, graph capture).
  - Cold first run: several minutes.
  - Compare llama.cpp's measured 3.7 s load. The adapter's readiness deadline should be about 600 s on the first start and about
    180 s afterwards. Record the four log milestones above to replace these guesses.
- `-O0` or `--enforce-eager` cut startup at the cost of decode speed (section 2.4).

### 3.5 Running detached and stopping cleanly

Verified behaviour:

- `serve_http` installs SIGINT and SIGTERM handlers on the event loop (`launcher.py:L260-261`) through `NoSignalServer`, which
  "never installs its own SIGINT/SIGTERM handlers" (L45-49).
- On the signal, `handle_shutdown` (L263-289) calls `engine_client.shutdown(timeout=...)`.
  - The mode is `"abort" if timeout == 0 else "drain"` (L268), and the default `--shutdown-timeout` is 0, so in-flight requests are aborted.
  - It then stops uvicorn. Log lines carry the `[shutdown] API server:` prefix.
- During engine init, SIGTERM raises `KeyboardInterrupt("terminated")` to interrupt start-up (`entry.py:L171-174`).

Recipe for Hypercompute on the Windows coordinator (INFERRED design; the WSL process-lifetime details are UNVERIFIED):

```text
launch:  wsl.exe -d Ubuntu --exec bash -c 'echo $$ > ~/.hypercompute/run/<dep>.pid; exec setsid -w \
           ~/.venvs/vllm-0.32.0/bin/vllm serve <args> >> ~/.hypercompute/logs/<dep>.log 2>&1'
         (keep the wsl.exe child under the existing Windows Job Object supervisor; `exec` keeps the
          recorded pid equal to the vllm API-server pid)
stop:    wsl.exe -d Ubuntu --exec bash -c 'kill -TERM $(cat ~/.hypercompute/run/<dep>.pid)'
         wait up to ~30 s for the wsl.exe child to exit (and /health to stop answering)
         fallback: wsl.exe -d Ubuntu --exec bash -c 'kill -KILL -- -$(cat ~/.hypercompute/run/<dep>.pid)'
         (negative pid = whole process group created by setsid, which includes the spawned EngineCore)
```

- Killing `wsl.exe` on the Windows side is **not** proven to terminate the Linux processes (UNVERIFIED). Always signal inside the distro.
- Whether WSL tears down an idle distro while a backgrounded process still runs is also UNVERIFIED. This is why the recipe keeps
  `wsl.exe` in the foreground as the supervised child instead of using `nohup ... &`.
- On a native Linux node, the same `setsid` + pidfile + `kill -TERM` / `kill -KILL -- -pgid` pattern applies without `wsl.exe`.

---

## 4. Distributed

### 4.1 Executors and requirements (VERIFIED from `docs/serving/parallelism_scaling.md`)

- Strategy guidance (L7-11): one GPU if the model fits; TP within a node; "TP + PP" across nodes. For uneven splits or GPUs without
  NVLink, "leverage pipeline parallelism instead of tensor parallelism" (L22-23).
- Executors (L33): "Ray for multi-node inference and native Python `multiprocessing` for single-node". Multi-node `mp` also exists without
  Ray (L139-159): `--nnodes 2 --node-rank 0 --master-addr <HEAD_IP>` on the head and `... --node-rank 1 --headless` on the worker.
- Homogeneity (L61): "Ensure that every node provides an identical execution environment, including the model path and Python
  packages. Using container images is recommended". In practice that means the same vLLM, torch, CUDA and NCCL on every node.
- Ray is optional: `pip install "ray[cgraph]"` (L71-75). Set `VLLM_HOST_IP` per node (L105).
  `docs/serving/distributed_troubleshooting.md:L11` explains that the "No available node types can fulfill resource request" error
  comes from IP mismatches.
- Security (L107-108): cluster traffic "is unencrypted, and the endpoints exchange data in a format that can be exploited to execute
  arbitrary code". Use a private segment only. This is the same trust model as ggml RPC.
- Network (L161-165, L213-219):
  - TP across nodes needs fast interconnect (InfiniBand, GPUDirect RDMA).
  - `[send] via NET/Socket` means NCCL used raw TCP, "which *is not* efficient for cross-node tensor parallelism".
  - PP sends one activation per stage boundary per step. For Qwen2.5-7B that is 3584 × 2 B = 7 KiB per token (INFERRED). Single-stream
    decode is therefore bound by **per-token round-trip latency**, not bandwidth, exactly as with llama.cpp RPC.
- WSL2 as a cluster member (INFERRED, UNVERIFIED): NCCL and Ray open many ports. NAT-mode WSL2 is not reachable from the LAN without
  per-port proxies, so mirrored networking would be required. Do not plan on WSL2 nodes in multi-node vLLM.

### 4.2 Can the Mac join a vLLM cluster? **No** (VERIFIED code facts, INFERRED conclusion)

- Placement uses one resource key from the *driver's* platform:
  - `device_str = current_platform.ray_device_key`, and an empty key raises "current platform ... does not support ray"
    (`vllm/v1/executor/ray_utils.py:L572-576`).
  - Every bundle is `{device_str: 1.0}` (L620-622).
- The keys differ by platform:
  - CUDA: `ray_device_key = "GPU"`, `dist_backend = "nccl"` (`vllm/platforms/cuda.py:L229-231`).
  - vllm-metal: `ray_device_key = "mlx"`, `device_control_env_var = "VLLM_METAL_VISIBLE_DEVICES"` (vllm-metal `vllm_metal/platform.py:L123-127`).
  - Upstream CPU (incl. macOS): no key (base default `""`, `interface.py:L150`) and `dist_backend = "gloo"` (`cpu.py:L105-108`).
- A CUDA-driven cluster places only on `GPU`, and a Metal-driven one only on `mlx`. There is also no shared collective: NCCL vs MLX ring.
- vllm-metal's own multi-Mac support (`docs/distributed.md`):
  - PP "data plane ... runs over MLX's own `mx.distributed` ring backend" (L137-139).
  - "Wi-Fi / Ethernet is too slow to serve over, so Thunderbolt is the supported transport" (L45).
  - "TP=1 only" (L164).
  - Validated "end-to-end on Qwen3-0.6B" across two Macs (L48).
  - DP across Macs is supported (L169-203), but "does **not** serve a model larger than one Mac".

**Consequence for the user's two-laptop CUDA + Mac layer sharing:** vLLM cannot split one model across the RTX 5070 Ti and the M4.
The options are:

- keep llama.cpp RPC for split placement; or
- run **whole-model replicas**: vLLM-CUDA in WSL2 on the G14 plus vllm-metal on the Mac, both OpenAI-compatible, with the Hypercompute
  gateway routing requests between them. That gives throughput scale-out for models that fit each machine, not more capacity.

### 4.3 Heterogeneous NVIDIA nodes and weight distribution (VERIFIED code, INFERRED implications)

- **KV capacity is set by the smallest worker.** `vllm/v1/core/kv_cache_utils.py:L2531-2553`, `get_kv_cache_configs`: "different workers
  may have different memory available ... 5. Change the num_blocks of each worker to the smallest among all workers". A 32 GB + 128 GB pair
  gets KV capacity sized to the tighter stage.
- **Layer split is even by default and overridable.** `vllm/distributed/utils.py:L126-166`, `get_pp_indices`:
  - With no override, layers are split evenly. The remainder goes to the middle partitions, never the last, "because it often contains
    an additional norm layer".
  - `VLLM_PP_LAYER_PARTITION="a,b,..."` overrides this. It must have `len == pp_size` and `sum == num_hidden_layers`, else
    `ValueError` (L142-151). Defined at `envs.py:L884`.
- **Stage contents** for Qwen2 (`vllm/model_executor/models/qwen2.py`):
  - `embed_tokens` lives only on the first rank (L367-378). `norm` and `lm_head` live only on the last rank (L394-397, L479-489).
  - The model is `SupportsPP` (L458-459).
  - For Qwen2.5-7B-AWQ each of those is about 1.09 GB fp16, so the first and last stages carry about 1 GB more than their layer count suggests.
- TP splits every layer evenly (Megatron, L31). On unequal GPUs it runs at the speed and memory of the smallest. Use PP for unequal nodes.
- Weight-distribution rule for the planner (INFERRED; applies equally to llama.cpp `-ts`, see `llama_placement.md` §1.5 and §4):
  - **Single-stream decode.** Stages run one after another for each token, so time per token is about Σ(layers_i × t_layer,i) +
    (stages − 1) × link RTT. The optimum is *not* proportional. Fill the fastest device up to its memory cap after embed/head/KV,
    put the remainder on the next fastest, and use as few stages as possible. Hypercompute's own data fits this: the 7B drops from
    80.6 to 25.8 tok/s with 9 of 28 layers on a CPU peer.
  - **Concurrent throughput with PP.** Stages overlap across requests, so balance stage time: layers_i ∝ speed_i (memory bandwidth for
    decode), subject to each node's free memory. Feed the result to `VLLM_PP_LAYER_PARTITION`.
- Mixed architectures (x86_64 RTX 5090 + aarch64 GX10/GB10, sm_120 + sm_121) in one cluster: both wheels exist (PyPI aarch64 and x86_64
  cu130), and `12.0f` family kernels cover 12.x. Release notes L75 mention DGX Spark-specific GEMM plans. One cluster across both
  architectures is **UNVERIFIED** (Ray/NCCL across architectures was not checked).

### 4.4 Offload options and their cost

- `--cpu-offload-gb` (UVA). `vllm/config/offload.py:L16-32`: "virtual way to increase the GPU memory size ... requires fast CPU-GPU
  interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model forward pass".
  - Cost (INFERRED): every offloaded GiB crosses PCIe once per forward step. `nvidia-smi` reports `pcie.link.gen.max=4, width.max=16`
    (about 25 GB/s practical). 1 GiB offloaded adds about 40 ms per token and would cut a 12 ms/token decode by roughly 4x.
  - On WSL2 it also needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` for pinned/UVA and triggers the host-hang warning (section 1.1).
  - Without pinned memory the offloader falls back to plain copies (`vllm/model_executor/offloader/uva.py:L51-54, L114-121`).
- Prefetch offload (`--offload-group-size`, `offload_num_in_group`, `offload_prefetch_step`; `offload.py:L48-77`) "uses explicit async
  prefetching to hide transfer latency". It overlaps transfers but does not remove the PCIe bound for single-stream decode (INFERRED).
- KV offloading (`--kv-offloading-size <GiB>`, backend `native|lmcache`; `cache.py:L266-275`, `docs/features/kv_offloading_usage.md`):
  - The `OffloadingConnector` "extends the prefix cache by offloading completed KV blocks to slower but larger tiers" (L3).
  - It is "CUDA, ROCm, and XPU only" (L6). `offload_prompt_only` defaults to `true` (L111).
  - It helps multi-turn prefix reuse and long shared prompts. **It does not add weight capacity** and does not raise `max_model_len` on
    its own (INFERRED).
- None of these make vLLM a capacity engine comparable to llama.cpp partial offload or AirLLM. For models larger than VRAM on the
  laptops, vLLM is the wrong tool.

---

## 5. Expected single-stream decode, 7B 4-bit, one consumer GPU

| Source | Setup | vLLM | llama.cpp | Status |
|---|---|---|---|---|
| Hypercompute `docs/RESULTS.md` | G14 RTX 5070 Ti Laptop, native Windows, Qwen2.5-7B Q4_K_M, b11770 | n/a | **80.6 tok/s** (short), 86.2 (long) | VERIFIED (local evidence file) |
| computingforgeeks "Ollama vs vLLM vs llama.cpp" (updated 2026-10-06), https://computingforgeeks.com/ollama-vs-vllm-vs-llama-cpp/ | Qwen3.8-**27B**, vLLM 0.31.0 INT4 W4A16 vs UD-Q4_K_M, 474-token prompt, 256 out, temp 0 | RTX 4090 **54.3**, RTX 5090 **85.7** | RTX 4090 **49.5**, RTX 5090 **77.8** | Third-party, not a 7B model. At 32 concurrent requests on the RTX 5090: vLLM 936 vs llama.cpp 369 aggregate tok/s. |
| techplained (search excerpt only) | A100 80GB, 7B/8B | AWQ 132 | Q4_K_M 105 | UNVERIFIED (page not read) |

Roofline estimate for the G14 (INFERRED; memory bandwidth about 672 GB/s per Tom's Hardware, 192-bit GDDR7; Notebookcheck lists 677 GB/s):

- llama.cpp reads about 4.37 GB per token (CUDA0 model buffer 4168 MiB; `token_embd` stays on CPU). Ceiling about 154 tok/s; measured
  80.6 = 52% efficiency.
- vLLM AWQ reads 5.57 GB − 1.09 GB embedding ≈ 4.48 GB per token (the fp16 `lm_head` is heavier than llama.cpp's Q6_K head, the int4
  layers lighter). Ceiling about 150 tok/s.
- At similar efficiency that is about **75-80 tok/s**. CUDA graphs and Marlin might push it to about 90, and WSL2 overhead might pull it
  to about 70. **Expected 70-95 tok/s single-stream. UNVERIFIED; must be benchmarked** with the same prompts as `RESULTS.md`
  (1 warm-up + N reps, concurrency 1).
- Bottom line: vLLM will not materially beat llama.cpp for one user on this laptop. Its advantage is batching and concurrency, and prefix
  caching across requests.

---

## 6. Recommendation and adapter recipe

### 6.1 Where vLLM fits in Hypercompute tiers

| Tier | Use vLLM? | Why |
|---|---|---|
| Single NVIDIA Linux box, model fits (RTX 5090 workstation; GX10/GB10, aarch64 cu130 wheel) | **Yes, for concurrent serving** | Continuous batching gives multiples of llama.cpp aggregate throughput at 8-32 concurrent requests (third-party data above). Same OpenAI API. |
| G14 alone (Windows), model fits | **Optional, demo of concurrency only** | Needs WSL2, about 5 GB AWQ in the Linux file system, a GPU lease exclusive of llama-server, and util computed from free memory. Single-stream gain about 0 (section 5). |
| Multiple NVIDIA Linux nodes, model too big for one | **Yes, PP across nodes** (`mp --nnodes` or Ray), `VLLM_PP_LAYER_PARTITION` from the planner, TP only within a node or with RDMA | Identical environments required. KV capacity is sized to the tightest stage. Over Ethernet, expect latency-bound single-stream like llama.cpp RPC. |
| Windows CUDA + Mac Metal (the current pair) | **No for splitting** | No cross-platform placement or collective (section 4.2). Keep llama.cpp RPC for layer splits. Optionally run whole-model replicas (vLLM-WSL2 + vllm-metal) behind the gateway for throughput. |
| Mac only | vllm-metal (separate engine id), not upstream vLLM | MLX checkpoints. Multi-Mac PP only over Thunderbolt. |
| Models bigger than total VRAM, CPU peers, GGUF-first catalog | **No** | Offload is PCIe-bound; GGUF is a plugin and experimental; no CPU-peer concept. |

### 6.2 Adapter recipe (`vllm_cuda`), mapped onto `backend/hypercompute/adapters/base.py` `EngineAdapter`

`inspect()`:

- Resolve a WSL distro, or a Linux host, and run
  `~/.venvs/vllm-0.32.0/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`.
- Capabilities: `streaming=True`, `cancel=True` (close the stream), `multi_host=False` for the first slice, `platforms=["linux (CUDA)", "windows-x64 via WSL2 (CUDA)"]`.

`validate(spec)`:

- The model must be an HF safetensors directory or repo with `quant_method` in {awq, gptq, compressed-tensors, fp8}. Reject `.gguf`
  unless the GGUF plugin is installed and the arch is in its tested list.
- Estimate weights + KV(`max_model_len` × 56 KiB for Qwen2.5-7B) + 1 GiB, and require it ≤ `util × total`.
- Query `nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits`, set
  `util = min(0.90, (free − 512) / total)`, and fail with a fix hint if util < needed. This is the vLLM startup check in
  `worker/utils.py:L761-770` done ahead of time.
- Require an exclusive GPU lease (no llama-server on that GPU).

`prepare(spec)`. Launch args:

```text
vllm serve <hf-id-or-/home/.../dir> \
  --served-model-name <alias> --host 127.0.0.1 --port <free port> \
  --api-key <per-deployment random key> \
  --gpu-memory-utilization <computed> --max-model-len 8192 --max-num-seqs 16 \
  --generation-config vllm --disable-uvicorn-access-log
env: HF_HOME=~/.cache/huggingface (Linux fs), VLLM_DISABLE_LOG_LOGO=1, no VLLM_WSL2_ENABLE_PIN_MEMORY, no --cpu-offload-gb
```

- Add `--enforce-eager` (or `-O0`) only for a fast-start "preview" mode.
- Start it with the WSL `exec setsid` + pidfile pattern from section 3.5.

`wait_ready(handle)`:

- Poll `GET http://127.0.0.1:<port>/health`, 2 s per-probe timeout, 1 s interval. Ready on the first 200.
- Deadline 600 s on a cold cache, 180 s on a warm one.
- Fail fast if the process exits or the log contains `Free memory on device`, `CUDA out of memory` or `Traceback`.
- After ready, `GET /v1/models` (with the API key) and assert that the alias is listed.
- Parse the log milestones (section 3.3) into the deployment record: load s, graph s, KV GiB.

`stream(handle, req)`:

- POST `/v1/chat/completions` with `"model": alias, "stream": true, "stream_options": {"include_usage": true}`.
- Read lines. Skip empty lines and lines starting with `:`. For a `data: ` prefix, `[DONE]` ends the stream; otherwise parse JSON.
  If the payload has `"error"`, raise an `EngineAdapterError` even though the HTTP status was 200.
- Tokens: `choices[0].delta.content`. Usage: the final chunk's `usage`.
- Compute TTFT and decode tok/s client-side, since there is no `timings` field.
- Forward the raw bytes verbatim to the gateway SSE (matching `GenerationEvent.raw`).

`cancel(request_id)`:

- Close the upstream response. vLLM's `with_cancellation` aborts the request on disconnect.

`stop(handle)`:

- `kill -TERM <pid>` inside the distro. With the default `--shutdown-timeout 0` the mode is abort.
- Wait up to 30 s for process exit, then `kill -KILL -- -<pgid>`.
- `StopResult.clean` = exited on TERM.

Health after ready:

- Poll `/health` every 5-10 s. 503 or connection refused means the deployment failed. vLLM's watchdog also exits on engine death.

Benchmark parity:

- Reuse `scripts/` bench prompts at concurrency 1 and 8 on the G14. Report vLLM AWQ vs llama.cpp Q4_K_M side by side, with startup breakdown.

### 6.3 What to tell the user about "layer sharing across MLC/MLX and CUDA on two laptops"

- vLLM's layer sharing (PP) works only *within* one vendor stack: CUDA with CUDA over NCCL, Metal with Metal over the MLX ring on
  Thunderbolt. The two laptops cannot share layers through vLLM.
- For the 7B class, both laptops fit the whole model. The fastest arrangement is one full replica per laptop: vLLM or llama.cpp on the
  G14 at about 80 tok/s, and vllm-metal or llama.cpp Metal on the M4. Hypercompute routes requests between them.
  Splitting a model that fits only loses speed (80.6 → 25.8 tok/s measured with 9 layers on a remote peer).
- For models that fit neither laptop alone, the only cross-vendor split today is llama.cpp RPC. The weight-distribution optimization
  belongs in Hypercompute's llama.cpp planner (`-ts`/`--device` order, `llama_placement.md`):
  - minimize Σ layers × per-layer time plus hop latency for single-stream use;
  - balance stage times only for concurrent use.
- The same rule, fed through `VLLM_PP_LAYER_PARTITION`, applies when the cluster is all NVIDIA Linux (5090s, GX10).

---

## 7. UNVERIFIED and open items

1. Actual vLLM 0.32.0 decode tok/s, startup time and `Available KV cache memory` on the G14 under WSL2. Requires installing into the
   WSL venv and running, which this audit was not allowed to do.
2. Whether `--torch-backend=auto` detects the driver under WSL (it should, via `nvidia-smi`). Recommend `cu130` explicitly.
3. Whether `vllm-gguf-plugin` 0.0.5 imports cleanly against torch 2.13.0 (ABI), and whether it accepts sharded GGUF.
4. WSL lifetime semantics: does terminating `wsl.exe` kill the Linux child, and does WSL keep a distro alive with a detached process?
5. Whether the pre-listen startup window looks like connection refused, or like a hang/reset through the WSL2 localhost relay.
6. Whether `--kv-cache-memory-bytes` bypasses the startup free-memory check.
7. Mixed x86_64 + aarch64 (RTX 5090 + GX10) vLLM clusters over Ray or `mp --nnodes`.
8. The exact JSON shape of the mid-stream error event (`create_streaming_error_response`); not read.
9. The third-party benchmarks were not reproduced. The only side-by-side found is a 27B model, not 7B.
10. vllm-metal throughput on an M4 Air vs llama.cpp Metal (SiliconBench, arXiv 2609.19169, is cited in the vllm-metal README but was not read).
