# AirLLM audit for the Windows/CUDA host path (lyogavin/airllm @ bc47e0d)

Scope: the non-Mac (torch/CUDA) path of AirLLM, which is what would run on the host (Windows 11, RTX 5070 Ti Laptop 12 GiB, sm_120, driver 591.84, 32 GB RAM). This extends `later_engines_and_prior_art.md` section B, which covered the license, the Mac/MLX path and the MLX `generate()` string return. Those facts are not repeated here.

Evidence labels:
- VERIFIED: read in source at the stated commit or tag (nothing was executed or installed).
- REPORTED: stated by a third party in a GitHub issue; not reproduced by us.
- INFERRED: follows from reading code or arithmetic, not run.
- UNVERIFIED: could not be confirmed; reason given.

Source was read with `gh api -H "Accept: application/vnd.github.raw" repos/lyogavin/airllm/contents/<path>?ref=bc47e0d474abf03f491f7547085bd85a2c6f0b1b`. Line numbers below refer to that commit unless stated.

---

## 0. TL;DR

1. **Pin stays at `bc47e0d474abf03f491f7547085bd85a2c6f0b1b`.** It is still the tip of `main` (committer date 2026-10-20T06:19:55Z, equal to the repo's `pushed_at`). It is 62 commits ahead of the PyPI/GitHub release `v4.0.0` (`ff35db2`). Install from git at that SHA with `#subdirectory=air_llm`, not from PyPI.
2. **Windows: installable and importable.** There are no `resource`, `fcntl`, `os.fork` or `signal` imports. The only Unix call, `ctypes.CDLL("libc.so.6").malloc_trim(0)`, sits inside `try/except`. MLX and `psutil` are imported only on darwin. VERIFIED.
3. **torch: AirLLM only requires `torch>=2.4`, with no upper bound.** sm_120 needs a cu128 or cu130 wheel. PyTorch's Windows build scripts put `12.0` in `CUDA_ARCH_LIST` for cu128/cu129/cu130 but not cu126. VERIFIED. `torch==2.11.0+cu128` (cp312 win wheel exists) is the reported-working combination on Windows 11 build 26200 with a Blackwell card (issue #364).
4. **The generic non-Mac path is a thin wrapper over a real transformers model.**
   - The model is built on `meta`. Forward hooks stream each module (embed, every decoder layer, norm, lm_head) disk -> GPU before it runs and back to `meta` after.
   - `generate()` is `return self.model.generate(*args, **kwargs)`. It returns a transformers `LongTensor`, or `GenerateDecoderOnlyOutput` with `.sequences`, and accepts `streamer=`, `stopping_criteria=`, sampling args and so on. VERIFIED.
5. **Every generated token re-reads every streamed layer from disk.** The KV cache (transformers `DynamicCache`) stays on the GPU, so prefill costs about one token's worth of IO.
   - Cost per token is roughly split size / effective disk->GPU throughput.
   - No option keeps layers resident in spare VRAM or RAM. INFERRED from hooks; feature request #109 is open.
6. **Qwen2.5 (`Qwen2ForCausalLM`) goes through the generic `AirLLMBaseModel`.** The layer names match exactly. It is REPORTED working with correct output on Windows + Blackwell (#364, Qwen2.5-3B) and on Linux (#330, Qwen2.5-32B).
7. **Performance is in the tens-of-seconds to minutes-per-token range.**
   - REPORTED: 28.6 s/token for a 3B model on Windows + RTX 5060 Ti, about 210 MB/s effective. 13.3 s/token for Qwen2.5-32B on Linux GB10.
   - `compression='4bit'/'8bit'` is 3.8x-8x *slower* (#330).
   - AirLLM is a batch/capacity engine, never an interactive one. Section 9 gives the routing gate.
8. **Cancellation:** in-process stop only works between tokens, via a transformers `StoppingCriteria`. A forward pass cannot be interrupted, so the supervisor must kill the process. Partial splits resume safely via `.done` markers.
9. **Do not use `delete_original=True` on Windows.** It crashes after deleting the first shard (`remove_real_and_linked_file`, #297).
10. **Always give each model and revision its own `layer_shards_saving_path`.** AirLLM appends a fixed `splitted_model` dir name, and its "already split" check is keyed only on layer file names.

---

## 1. Pin recommendation

| Item | Value | Evidence |
|---|---|---|
| `main` tip today | `bc47e0d474abf03f491f7547085bd85a2c6f0b1b` "chore: refresh star history chart", 2026-10-20T06:19:55Z | `gh api repos/lyogavin/airllm/commits` (first row); repo `pushed_at` 2026-10-20T06:19:56Z. VERIFIED |
| Previously pinned | same SHA | `later_engines_and_prior_art.md` B |
| Latest release | `v4.0.0`, tag -> `ff35db207a0c559af9aa95d686057c3fe84f1d40`, published 2026-09-05; PyPI `airllm 4.0.0` wheel uploaded 2026-09-05T16:00:15 | `gh api .../releases`, `.../tags`, PyPI JSON. VERIFIED |
| `v4.0.0...bc47e0d` | ahead_by=62, behind_by=0 | `gh api .../compare/ff35db2...bc47e0d`. VERIFIED |

Functional changes from v4.0.0 to bc47e0d, all VERIFIED from the compare patch:
- `airllm_base.py`:
  - `device` default changed from `"cuda:0"` to `None`, resolved by the new `utils.default_device()` (CUDA, then MPS, then CPU).
  - Per-expert MoE streaming is disabled when `compression` is set.
- `auto_model.py`: the macOS architecture pre-check (`MLX_ARCHITECTURES`).
- `airllm_llama_mlx.py`: MLX fixes for rope_theta, input types and an import regression.
- Everything else is docs, tests, LoRA BOM handling and star-history PNGs (about 50 of the 62 commits).

Recommendation: keep `bc47e0d`. The generic CUDA path is identical to v4.0.0 except for the device default, and we pass `device` explicitly anyway. Pinning the SHA rather than PyPI 4.0.0 keeps one source of truth with the prior audit and picks up the compression/expert guard.

Install spec:
```
airllm @ git+https://github.com/lyogavin/airllm@bc47e0d474abf03f491f7547085bd85a2c6f0b1b#subdirectory=air_llm
```
`setup.py` lives in `air_llm/` (VERIFIED in the tree listing). It only reads `README.md` from its own dir or the parent (lines 9-14).

Packaging gotcha (INFERRED from setuptools semantics): `packages=setuptools.find_packages()` (line 27) also picks up `air_llm/tests/` because it has an `__init__.py`. That installs a top-level `tests` package into site-packages. This is harmless in a dedicated venv, which is one more reason to give AirLLM its own venv.

Ignore the repo-root `requirements.txt`. It pins `bitsandbytes==0.39.0`, `peft@v0.3.0`, `accelerate@v0.20.3` and similar for the old Anima training project, not for the `airllm` package (VERIFIED, lines 1-9).

---

## 2. Install on the Windows host

### 2.1 Declared dependencies (`air_llm/setup.py` @ bc47e0d)
```python
install_requires=[
    'tqdm',
    'torch>=2.4',
    'transformers>=4.49,<6',
    'accelerate>=1.0',
    'safetensors',
    'huggingface-hub',
    'scipy',
    'sentencepiece',
    # 'bitsandbytes' is optional (used only for --compression); ...
    # 'compressed-tensors' is optional too: only checkpoints stored in that format (Kimi K3's ...
],
```
(lines 30-42, VERIFIED)

- **Extras:** there is no `extras_require`. There is no `python_requires`; the only Python classifier is `"Programming Language :: Python :: 3"` (line 44), and PyPI reports `requires_python: None`. VERIFIED.
- **Effective Python floor** comes from the dependencies, per PyPI JSON today (VERIFIED):
  - transformers 5.20.2 needs `>=3.10.0`.
  - torch 2.14.1 needs `>=3.10`.
  - accelerate 1.15.0 needs `>=3.10.0`.
  - Use Python 3.12. The host has `C:/Users/operator/AppData/Local/Programs/Python/Python312/python.exe`; do not use the WindowsApps alias. #364 ran 3.11.15.
- **Optional packages:**
  - `optimum` is not used anywhere. No module imports it. `get_use_better_transformer()` is defined in several subclasses (e.g. `airllm_qwen2.py` lines 17-18) but nothing calls it. VERIFIED by grep over all `air_llm/airllm/*.py`.
  - `bitsandbytes` is optional. It is imported in `try/except` (`airllm_base.py` lines 23-29; `utils.py` lines 31-36) and needed only for `compression=`. PyPI `bitsandbytes 0.50.2` ships a `win_amd64` wheel (VERIFIED); sm_120 support in it is UNVERIFIED. **Do not install it.** Compression is slower (section 7), it disables prefetching (`airllm_base.py` lines 188-190), and `compress_layer_state_dict` calls `v.cuda()` (`utils.py` lines 491, 498).
  - `compressed-tensors` and `flash-attn` are needed only for Kimi K3 (setup.py comment lines 40-41; README line 38). Skip them.
  - `scipy` is declared but no AirLLM module imports it (VERIFIED by import tally). It installs harmlessly.
- **Version pins from the README:** Qwen3.8-27B needs transformers 5.8+ (README line 36). Kimi K3 needs transformers 4.56.x (README line 38). Neither applies to Qwen2.5/Llama.

### 2.2 Windows compatibility of the code (VERIFIED by grep over every `.py` in `air_llm/airllm/` @ bc47e0d)
- There are no imports of `resource`, `fcntl`, `termios`, `pwd`, `grp`, `signal`, `multiprocessing` or `subprocess`. There is no `os.fork`, `os.setsid`, `/proc` or `/dev/shm`.
- `utils.py` lines 78-85: `clean_memory()` calls `ctypes.CDLL("libc.so.6").malloc_trim(0)` inside `try/except Exception: pass`. On Windows this raises and is swallowed, then `torch.cuda.empty_cache()` runs.
- `mlx` and `psutil` are imported only by `airllm_llama_mlx.py` (lines 11-15) and `persist/mlx_model_persister.py` (lines 5-10). Those load only when `sys.platform == "darwin"`:
  - `__init__.py` lines 5-11
  - `auto_model.py` lines 7-11
  - `persist/model_persister.py` lines 17-29
- `utils.link_or_copy_file` (lines 517-541) tries `os.link` (works on NTFS within one volume), then `os.symlink` (needs Developer Mode or admin on Windows), then `shutil.copyfile`. It is used only for "passthrough" shards that hold exactly one module.
- `__init__.py` lines 25-47 import the optional model families in `try/except` and only warn on failure, so `import airllm` does not break if a niche family fails to import.
- REPORTED (#364): `airllm 4.0.0` loads and generates correct output on "Windows 11 (build 26200)" (the same build as the host), Python 3.11.15, torch `2.11.0+cu128`, RTX 5060 Ti 16 GB, transformers 4.57.6 and 5.16.1.

### 2.3 torch for an RTX 50-series (sm_120)
- AirLLM sets only `torch>=2.4` (no upper bound), so it does not block a Blackwell-capable torch. VERIFIED.
- PyTorch's Windows CUDA build configs, VERIFIED in pytorch/pytorch:
  - `v2.11.0` `.ci/pytorch/windows/internal/cuda_config.bat` lines 9-23: `126` uses `CUDA_ARCH_LIST=5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0` (no 12.0). `128`, `129` and `130` use `7.5;8.0;8.6;9.0;10.0;12.0`.
  - `v2.14.1` `.ci/pytorch/windows/build_env_setup.py` lines 69-74 have the same table, plus `"132"`.
  - So the cu126 Windows wheels cannot run sm_120 kernels. cu128 and cu130 can.
- Wheels on the index (VERIFIED by listing `download.pytorch.org/whl/<cu>/torch/`):
  - `cu128` cp312 win_amd64: 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0. 2.11.0 is the newest cu128 build.
  - `cu130` goes up to 2.14.1, and `cu129` up to 2.9.0.
- Whether driver 591.84 runs cu130 wheels is UNVERIFIED (driver/CUDA compatibility tables were not re-read). cu128 with torch 2.11.0 is the combination with a field report on this exact Windows build (#364).

### 2.4 Recommended environment (do not run until the build phase)
```
C:/Users/operator/AppData/Local/Programs/Python/Python312/python.exe -m venv C:/Users/operator/hypercompute/.venvs/airllm
<venv>/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
<venv>/Scripts/python -m pip install "airllm @ git+https://github.com/lyogavin/airllm@bc47e0d474abf03f491f7547085bd85a2c6f0b1b#subdirectory=air_llm" "transformers==5.16.1"
```
- Install torch first from the cu128 index, so the resolver sees `torch>=2.4` as already satisfied. Otherwise pip may pull a different torch build from PyPI (INFERRED).
- `transformers==5.16.1` is suggested because #364 reports it working with the same generic path. 4.57.6 was also reported working.
- After the first good smoke test, freeze the exact `accelerate`, `huggingface-hub`, `safetensors` and `tokenizers` versions into a lock file. No compatibility matrix for those was found (UNVERIFIED).
- Keep this venv separate from the llama.cpp gateway's Python. AirLLM drags in the full torch/transformers stack.

---

## 3. Load API on the CUDA/CPU path

### 3.1 Dispatch (`air_llm/airllm/auto_model.py`)
- `AutoModel.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)` (line 63).
- On non-darwin it calls `get_module_class` (lines 53-60). That calls `get_architecture` (lines 42-50), which loads `AutoConfig.from_pretrained(path, trust_remote_code=True[, token=hf_token])` and takes `config.architectures[0]`.
- If the arch is in `ARCH_OVERRIDES` (lines 17-27: ChatGLM, QWen v1, Baichuan, InternLM, KimiK3, Qwen3_5, Qwen4Exp), the matching subclass is used. **Every other arch uses `AirLLMBaseModel`** (line 58 prints `using generic AirLLM streaming model for architecture: ...`).
- It then instantiates `class_(pretrained_model_name_or_path, *inputs, **kwargs)` (lines 76-79). VERIFIED.
- `AirLLMBaseModel` is exported directly on non-Mac (`__init__.py` line 14). Constructing it ourselves skips the `trust_remote_code=True` config probe in `get_architecture` (see 3.5).

### 3.2 Constructor (`airllm_base.py` lines 84-87, VERIFIED)
```python
def __init__(self, model_local_path_or_repo_id, device=None, dtype=None, max_seq_len=512,
             layer_shards_saving_path=None, profiling_mode=False, compression=None,
             hf_token=None, prefetching=True, delete_original=False,
             install_hooks=True, load_resident=True):
```

| Param | Behavior (VERIFIED unless marked) |
|---|---|
| `model_local_path_or_repo_id` | Local dir (must contain one of `pytorch_model.bin.index.json`, `model.safetensors.index.json`, `model.safetensors`, `pytorch_model.bin`) or an HF repo id. `utils.py` lines 834-846: a local dir without weights falls through to "treat it as a repo id". |
| `device` | `None` -> `default_device()` = `"cuda:0"` if `torch.cuda.is_available()`, else `"mps"`, else `"cpu"` (`utils.py` lines 88-101; `airllm_base.py` line 150). `self.device = torch.device(...)` (line 151). |
| `dtype` | `None` -> `config.torch_dtype` / `config.dtype` / `text_config.*`, else `torch.float16` (lines 170-181). Qwen2.5 and Llama-3 configs are bf16. |
| `max_seq_len` | **Stored and never used.** Line 207 `self.max_seq_len = max_seq_len` is the only reference in the package (grep). It does not truncate or size anything; the caller must bound prompt + new tokens. |
| `layer_shards_saving_path` | Split goes to `Path(layer_shards_saving_path) / "splitted_model"` (or `splitted_model.4bit` / `.8bit`). The default is `<checkpoint dir>/splitted_model` (`utils.py` lines 550-560). |
| `profiling_mode` | Records per-module disk load time into `model.profiler.profiling_time_dict['load_safe_tensor']` (lines 435-439; `profiler.py` lines 12-17). `model.profiler.print_profiling_time()` prints totals. Useful for calibration. |
| `compression` | `'4bit'` / `'8bit'` (bitsandbytes blockwise). Raises `ImportError` if bnb is missing (lines 131-133). Disables prefetching (lines 188-190). |
| `hf_token` | Passed to `AutoConfig`, `AutoTokenizer` and `snapshot_download`. |
| `prefetching` | Single-thread `ThreadPoolExecutor` that loads the next module while the current one computes (lines 191, 900-904). It also gates pinned-memory copies (line 443). |
| `delete_original` | Deletes original shards while splitting. **Broken on Windows** (section 3.4). |
| `install_hooks`, `load_resident` | Training knobs. Leave the defaults (`True`). |

- `self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True[, token])` (lines 224-228).
- `self.generation_config = GenerationConfig.from_pretrained(path)`, falling back to `GenerationConfig()` (lines 218-222). It is attached to the model at line 366.
- **There is no `revision` parameter anywhere.** `snapshot_download` is called without `revision` (`utils.py` lines 851-862), so a repo id always resolves to `main`. To pin a model revision, pre-download it ourselves and pass the local dir.

### 3.3 What it downloads (repo-id path; `utils.py` lines 848-886, VERIFIED)
1. `huggingface_hub.snapshot_download(repo, token=..., ignore_patterns=['*.safetensors', '*.bin'])`. This downloads **everything except** `.safetensors` and `.bin`: config, tokenizer, index, `generation_config.json`, README, and also any `*.py`, `*.gguf`, `*.pt`, `*.h5`, `*.msgpack` or `*.onnx` the repo ships.
2. If there is no index json, it also downloads `model.safetensors` / `pytorch_model.bin` (lines 857-862).
3. During the split, each shard is fetched on demand with `snapshot_download(repo, allow_patterns=<shard basename>)` (lines 712-713, 750-751, 769-770).
4. **Index preference:** `pytorch_model.bin.index.json` is checked *before* `model.safetensors.index.json` (lines 566-572). A repo shipping both is split from the pickled `.bin` files via `torch.load(..., map_location='cpu')` (lines 582, 754, 772). Qwen2.5 and Llama-3 repos ship safetensors only.

Recommendation: Hypercompute pre-downloads with its own `snapshot_download(repo, revision=<sha>, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model", "merges.txt", "vocab.json"])` into a managed dir (no `*.py`, no `*.bin`). It then runs the worker with `HF_HUB_OFFLINE=1`, as #364 did.

### 3.4 Split layout, size and reuse (VERIFIED in `utils.py` and `persist/safetensor_model_persister.py`)
- **Module list** (lines 590-607): `[embed] + [f"{layer_prefix}.{i}" for i in range(n_layers)] + [norm, lm_head] (+ resident, cpu-resident)`, each with a trailing `"."`. `n_layers` is counted from index keys (line 593).
  - Modules with no weights are dropped (lines 612-614). An example is a tied `lm_head`, as in Qwen2.5 0.5B/1.5B/3B.
- **Files:** `saving_path / (layer + "safetensors")` plus a zero-byte `... + "safetensors.done"` marker, written after `save_file` (persister lines 32-38).
  - For Qwen2.5 this gives `model.embed_tokens.safetensors`, `model.layers.0.safetensors` ... `model.layers.{N-1}.safetensors`, `model.norm.safetensors` and `lm_head.safetensors`, each with a `.done` file.
  - Load uses `load_file(Path(path) / (layer_name + ".safetensors"), device="cpu")` (persister lines 41-43).
- **Size:** tensors are saved as loaded, with no dtype change when `compression=None` (line 785 is a no-op without compression). So the split is about **1.0x** the original weights.
  - REPORTED (#330, Qwen2.5-32B): `none` 62G, `8bit` 31G, `4bit` 18G.
  - `check_space` (lines 463-484) requires free space >= original size (minus an existing partial split) before splitting. For 4bit it *divides* by 0.2813 (line 474), so it over-estimates the space needed by about 3.6x (bug, INFERRED).
- **Passthrough:** if a source shard holds exactly one module's tensors, it is linked instead of copied (lines 654-672, 708-716). Kimi K3 is the motivating case. Qwen2.5/Llama shards hold several layers, so they get copied (INFERRED).
- **Peak RAM while splitting:** whole source shards are `load_file`d into a `state_dict` that is trimmed per layer (lines 735-798). Peak is about 1-2 source shards, roughly 4-10 GB for Qwen2.5-72B's ~3.9 GB shards. INFERRED.
- **Reuse:** if `saving_path` exists and every expected module has both the `.safetensors` file and its `.done` marker, it returns immediately: `saved layers already found in ...` (lines 631-645). If only some exist, it re-reads all source shards but rewrites only modules without a marker (lines 789-791). An interrupted split therefore resumes correctly; a half-written file without `.done` is rewritten.
  - #364 confirms reuse in its log: `saved layers already found in D:\ai-models\Qwen2.5-3B-Instruct\splitted_model`. REPORTED.
- **Collision hazard (VERIFIED by lines 557-560 and 634-645):** the dir name is the constant `splitted_model`, and the completeness check only looks for the *new* model's layer file names.
  - Two models sharing one `layer_shards_saving_path` silently reuse each other's shards whenever the second model's layer names are a subset of the first's.
  - Fix: always use `layer_shards_saving_path = <store>/<org>__<name>@<revision>`.
- **Can the originals be deleted after a complete split?** INFERRED yes, not tested. On the local-path route the loader only needs `config.json`, the tokenizer files, `generation_config.json` and the index json:
  - `utils.py` lines 834-843 check for the index file.
  - Lines 563-572 read only the JSON.
  - Lines 634-645 return before any shard is opened.
  - Hypercompute can therefore delete `model-*.safetensors` itself after verifying every `.done` marker, which brings the disk need back to about 1x. Smoke-test this before relying on it.
- **`delete_original=True` is broken on Windows.** `remove_real_and_linked_file` (lines 507-513) compares `os.path.realpath(to_delete) != to_delete`, a `str` against a `Path`, which is always unequal. It removes the file and then tries to remove its realpath again.
  - On Windows without Developer Mode, `huggingface_hub` stores real files instead of symlinks, so the second remove raises `FileNotFoundError` after the shard is already gone.
  - REPORTED with a WinError 2 traceback in #297 (comment 2026-09-17, at `71ca1e8`). The helper is unchanged at bc47e0d.
  - If the paths were ever equal, `targetpath` would be unbound (`UnboundLocalError`).

### 3.5 Security notes (extends the prior audit)
- `auto_model.get_architecture` uses `trust_remote_code=True` (lines 44-47) and `get_tokenizer` always uses `trust_remote_code=True` (`airllm_base.py` lines 224-228).
- The base constructor tries `trust_remote_code=False` for `AutoConfig` first and only falls back to `True` on exception (lines 158-165). Meta instantiation reuses that decision (line 302).
- Mitigation:
  - Allowlist in-tree architectures (`Qwen2ForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`).
  - Check `config.json` ourselves for `"architectures"` and the absence of `"auto_map"`.
  - Download without `*.py`.
  - Construct `AirLLMBaseModel` directly instead of `AutoModel`.
  - With no `.py` files and no `auto_map`, `trust_remote_code=True` has nothing to execute (INFERRED from transformers semantics).

---

## 4. Generation on the non-Mac path

### 4.1 Signature and return type
```python
# airllm_base.py lines 923-930 (VERIFIED)
def generate(self, *args, **kwargs):
    return self.model.generate(*args, **kwargs)

def forward(self, *args, **kwargs):
    return self.model(*args, **kwargs)

def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)
```
- `self.model` is the real transformers `*ForCausalLM`, instantiated on `meta` (lines 297-345: `sdpa` first, then `eager`).
- The return type is whatever transformers `GenerationMixin.generate` returns. By default that is a `torch.LongTensor` `[batch, prompt_len + new_len]` (#364 uses `gen[0]`). With `return_dict_in_generate=True` it is a `GenerateDecoderOnlyOutput` with `.sequences`; the README quickstart, `inference_example.py` and `tests/test_streaming_gpu.py` line 53 all use `out.sequences[0]`.
- It is **not** a `str`. That is the MLX path only.
- `AirLLMBaseModel` itself does not subclass `GenerationMixin` (class at line 58 has no bases). The wrapped model does. If the model class lacks it, AirLLM mixes it in:
```python
# airllm_base.py lines 389-403 (VERIFIED)
extra_bases = () if isinstance(self.model, GenerationMixin) else (GenerationMixin,)
...
class _AirLLMRuntimeModel(base_cls, *extra_bases):
    @property
    def device(self):
        return running_device
    @property
    def dtype(self):
        return running_dtype
self.model.__class__ = _AirLLMRuntimeModel
```
`Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin)` already has it (transformers v5.20.2 `models/qwen2/modeling_qwen2.py` line 398, VERIFIED).

### 4.2 Streaming: real per-token streaming works
- `streamer=` passes straight through `**kwargs` to transformers. VERIFIED in transformers:
  - v5.20.2 `generation/utils.py` lines 2918-2919 call `streamer.put(input_ids.cpu())` once with the prompt.
  - The decode loop calls `stop_check(...)` once per step (loop at lines 3210-3290). `StopCheck.__call__` calls `self.streamer.put(tokens.cpu())` for every new token (lines 393-396).
  - `streamer.end()` is called at lines 3291-3292.
- The one-step-late `DeferredStopCheck` is only used when `device.type == "mps"` (`is_supported`, lines 441-456: `if device.type != "mps" or is_assistant: return False`), so on CUDA tokens are streamed immediately.
- v5.16.1 has the same behavior: `streamer.put(next_tokens.cpu())` at line 2934 and `streamer.end()` at line 2944 (VERIFIED by grep).
- `BaseStreamer` (`generation/streamers.py` lines 28-39) has two methods, `put(value)` and `end()`. `TextStreamer` / `TextIteratorStreamer` raise `ValueError("TextStreamer only supports batch size 1")` for batch > 1 (line 85).
- So `TextIteratorStreamer` (generate in a thread, iterate in the main thread) or a custom `BaseStreamer` yields real tokens as each full forward completes. With AirLLM that means one event every several seconds to minutes.
- No auto-compile: transformers auto-compiles only with a compileable non-`DynamicCache` cache (v5.20.2 `utils.py` lines 2424-2446: `using_compilable_cache = cache is not None and cache.is_compileable and type(decoder_cache) is not DynamicCache`). Do not pass `cache_implementation="static"`; torch.compile across meta-swapped weights is untested.

### 4.3 Per-token work and KV cache (VERIFIED code; behavior INFERRED)
```python
# airllm_base.py lines 759-775: every module except a tied embedding gets both hooks
if self.tie_word_embeddings:
    embed_state = self.load_layer_to_cpu(self.layer_names[0])
    self.move_layer_to_device(embed_state)
    self.model.tie_weights()
    self._streamed_indices = list(range(1, n - 1))  # decoder layers + final norm
else:
    self._streamed_indices = list(range(n))
...
    module.register_forward_pre_hook(self._pre_hook)
    module.register_forward_hook(self._post_hook)

# lines 889-904: load (or take the prefetched copy), move to GPU, prefetch the next module
def _pre_hook(self, module, args):
    idx = module._airllm_idx
    if self.prefetching and self._prefetch_future is not None and self._prefetched_idx == idx:
        state_dict = self._prefetch_future.result()
        ...
    else:
        state_dict = self._load_streamed_layer(idx)
    module._airllm_moved = self.move_layer_to_device(state_dict)
    if self.prefetching:
        nxt = self._next_streamed_idx(idx)
        if nxt is not None:
            self._prefetch_future = self._executor.submit(self._load_streamed_layer, nxt)

# lines 906-919: evict to meta and clean after the module runs
def _post_hook(self, module, args, output):
    ...
    else:
        module.to('meta')
    clean_memory()          # gc.collect(); (libc malloc_trim); torch.cuda.empty_cache()
    return output
```
- Every forward call, prefill or one decode step, re-loads every streamed module from its `.safetensors` file. `load_layer` calls `safetensors.torch.load_file(..., device="cpu")` (`utils.py` lines 444-459; persister lines 41-43). AirLLM has no RAM or VRAM layer cache. Only the OS file cache can help.
  - The last module (`lm_head`) has no next index (`_next_streamed_idx`, lines 878-880), so the embedding for the next token is loaded synchronously.
- **Pinned memory:** with `prefetching=True` and CUDA, each loaded tensor is copied into newly pinned memory if the module is <= 2 GiB (`max_pinned_layer_bytes`, line 75; lines 443-456). Bigger modules go pageable. For Qwen2.5-72B the bf16 `lm_head` is about 2.5 GB, so it goes pageable (INFERRED).
- **KV cache:** transformers creates its default `DynamicCache` and passes it into each decoder layer as an argument. The hooks only move module parameters, so K/V tensors stay on the GPU across steps (INFERRED from the hook code plus transformers' cache design).
  - `use_cache=True` saves attention recompute. It does not save weight IO (also stated in #364's 2026-09-27 comment).
  - KV size for Qwen2.5-72B (80 layers, 8 KV heads x 128 dims, bf16) is about 320 KiB/token, so 4k tokens is about 1.3 GB of VRAM (INFERRED arithmetic).
- **Cost model (INFERRED):**

  `t_request ≈ (1 + new_tokens) × (split_bytes / B_eff + N_modules × overhead_per_module)`

  - Prefill is one pass regardless of prompt length (plus compute), so long-prompt/short-answer jobs are the least bad use.
  - Batching several prompts in one `generate` call (batch > 1, left padding) amortizes the same IO over more sequences. TextStreamer cannot be used then.
- **VRAM use:** about one module plus KV plus activations. The README claims about 4 GB for a 70B and about 8 GB for 405B (README lines 289-298), so 12 GiB is enough. The remaining ~8 GiB sits idle; there is no option to pin K layers resident (no constructor parameter; issue #109 "Optimize for consumer GPU, eg 11GB or 16GB" is open).

---

## 5. Architectures

- **The generic path requires** `model.embed_tokens`, `model.layers.N`, `model.norm` and `lm_head` (`set_layer_names_dict`, lines 78-82). It verifies the instantiated class has them (`_model_matches_layer_names`, lines 252-265), then lets transformers run the forward pass.
- **Qwen2.5 (`Qwen2ForCausalLM`) works on the generic path.** VERIFIED layout plus REPORTED runs.
  - Not in `ARCH_OVERRIDES`, so `AirLLMBaseModel` is used. #364's log shows `using generic AirLLM streaming model for architecture: Qwen2ForCausalLM` and `built empty Qwen2ForCausalLM via AutoModelForCausalLM (attn=sdpa)`.
  - transformers v5.20.2 `modeling_qwen2.py` line 319 `self.embed_tokens`, line 320 `self.layers = nn.ModuleList(...)`, line 323 `self.norm`, line 324 `self.rotary_emb`, line 406 `self.model = Qwen2Model(config)`, line 408 `self.lm_head`. These match the generic names.
  - `rotary_emb.inv_freq` is a non-persistent buffer. It is materialized at construction because `include_buffers=False` (line 308) and moved to the device (lines 371-373).
  - Tied embeddings (Qwen2.5 <= 3B) keep the embedding resident and stream the rest (lines 757-765).
  - REPORTED correct output: Qwen2.5-3B-Instruct on Windows + RTX 5060 Ti (#364). Qwen2.5-32B-Instruct with none/8bit/4bit on GB10 (#330). `tests/test_streaming_gpu.py` line 14 uses `Qwen/Qwen2.5-7B-Instruct` as its example.
  - The class `AirLLMQWen2` (`airllm_qwen2.py`) still exists but `AutoModel` never selects it. It only overrides `get_use_better_transformer`, which is dead code.
- **Known-good list:**
  - There is no CI test workflow. `.github/workflows/` has only `release.yml` and `star-history.yml`.
  - `tests/test_automodel.py` is stale: it expects `AirLLMLlama2` for Llama and `AirLLMMistral` for Mistral, while `get_module_class` now returns `AirLLMBaseModel` for both.
  - "Known-good" therefore rests on the README and issues. The README "Supported Models" section (lines 281-298) claims Llama 2/3/3.1/3.3/4, Qwen 1/2/2.5/3/3.5/3.8, DeepSeek V2/V3/R1, Mistral/Mixtral, Phi, Gemma, ChatGLM, Baichuan, InternLM, Yi and Kimi K3.
  - Field evidence seen in this audit:
    - Qwen2.5 (#364, #330)
    - gpt-oss-20b (#374, slow but running)
    - Kimi-K2.5 with a user patch (#330 comment)
  - Known failure: `google/gemma-4-26B-A4B-it` breaks layer-prefix parsing on a nested `language_model` (#345, open).
- **Checkpoint formats:** HF safetensors or `.bin` only. **No GGUF** (#304 maintainer-side answer; there is no GGUF code path). Pre-quantized HF checkpoints go through `AutoHfQuantizer` (lines 347-362) and verbatim fp8/packed loading (lines 634-649). Whether GPTQ/AWQ/FP8 kernels work on Windows + sm_120 is UNVERIFIED; do not plan on it.

---

## 6. Cancellation

- **In-process:** `generate` is a synchronous call on the caller's thread. transformers checks `stopping_criteria` once per decode step (v5.20.2 `utils.py`, the `unfinished_sequences & ~stopping_criteria(input_ids, scores)` line inside the loop at 3210-3290). A `StoppingCriteria` that reads a `threading.Event` can therefore stop generation **after the current full forward pass**, which takes seconds to minutes here.
  - There is no hook to abort mid-forward.
  - The prefetch `ThreadPoolExecutor` worker (line 191) is non-daemon, so a Python-level exit waits for its in-flight layer read (INFERRED from `concurrent.futures` semantics).
- **Supervised subprocess (our plan):** `Popen.kill()` (TerminateProcess on Windows) ends it immediately, and the driver frees VRAM on process exit. Safe states:
  - During the split: `.done` markers mean the next run rewrites only unfinished modules (section 3.4).
  - During download: `huggingface_hub` resumes partial files.
  - During generation: nothing persistent is written.
- **Protocol advice:** send a soft cancel (the event), wait about 2 s, then kill. In practice the kill is the mechanism; the soft path only helps when a token is about to finish.

---

## 7. Performance and IO expectations

The README gives **no seconds-per-token figures**. It only gives VRAM figures (lines 285-298) and the 2023 claim "speed up the inference speed for up to 3x" for compression (line 149). Field data (all REPORTED):

| Source | Setup | Result |
|---|---|---|
| #364 (2026-09-10) | Windows 11 26200, torch 2.11.0+cu128, RTX 5060 Ti 16 GB, airllm 4.0.0, Qwen2.5-3B-Instruct, local NVMe, split already done | 4 tokens in 114.5 s = **28.6 s/token**, GPU util about 6%, CPU about 0%, no progress output ("looked like a hang"). The reporter estimates about **210 MB/s** effective. Output correct. |
| #330 (2026-08-04) | Linux GB10, Qwen2.5-32B-Instruct, 10 tokens, split fits in 76 GB page cache | none **13.3 s/token** (975 MB/layer, 62G split); 8bit 50.5 s/token; 4bit 106.7 s/token; 4bit split build took 361 s |
| #186 comment (2024-12-17) | RTX 3050 6 GB, 32 GB RAM, Qwen2.5-Coder-32B 4bit (older AirLLM) | HDD: just above 2 min/token. RAM disk via `layer_shards_saving_path`: 13 s/token |
| #374 (2026-10-11) | RTX 3090, gpt-oss-20b-bf16 | about 30 min for about 120 tokens (about 15 s/token) |
| #364 comment (2026-09-27) | M1 Pro, MLX path, TinyLlama-1.1B | 0.33 tok/s |

Implications for this host (INFERRED, must be measured):
- **Windows overhead looks like the bottleneck, not raw disk.**
  - A 3B bf16 split is about 6 GB and would sit in the OS file cache after the first token, yet #364 still saw about 0.75 s per module.
  - Likely suspects:
    - per-module `torch.cuda.empty_cache()` plus `gc.collect()` (`clean_memory`, `utils.py` lines 78-85, called from `_post_hook` line 918), which forces fresh CUDA allocations every module under WDDM
    - a fresh `pin_memory()` copy of every tensor of every module (lines 449-456)
  - These are hypotheses. A/B test `prefetching=True` vs `False`; `False` skips the pinning branch, line 443.
- **Bytes per token is about the split size**, so dtype dominates. Arithmetic at three effective throughputs (decode only, excluding the one-time split):

| Model (bf16 split, approx.) | 0.21 GB/s (#364-like) | 2 GB/s | 5 GB/s (#330-like) | Disk with originals kept |
|---|---|---|---|---|
| Qwen2.5-32B (about 65 GB) | about 310 s/token | about 33 s | about 13 s | about 130 GB |
| Qwen2.5-72B / Llama-3.3-70B (about 141-145 GB) | about 690 s/token | about 72 s | about 29 s | about 290 GB |
| Llama-3.1-405B (about 810 GB) | about 64 min/token | about 6.8 min | about 2.7 min | about 1.6 TB |

- **Compression is not a speed-up** on modern hardware (#330, measured 3.8x and 8x slower), and it disables prefetching (lines 188-190). Use `compression=None`.
- **Disk:** NVMe on the same volume as the split. Free space must be >= original size before the split starts (`check_space`). Keep originals until the split is complete; `delete_original` crashes on Windows.

---

## 8. Minimal worker recipe (verified APIs only)

Assumptions:
- Model pre-downloaded by Hypercompute to `MODEL_DIR` at a pinned revision, safetensors only.
- Arch allowlisted by reading `config.json`.
- One worker process per loaded model; requests one at a time over stdin JSON lines.
- Library prints go to stderr. AirLLM `print()`s a lot, e.g. `found_layers:{...}` and `saved as:`.

```python
# hypercompute airllm_worker.py  (runs in .venvs/airllm, launched by the supervisor)
import json, os, sys, threading, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never reach the Hub at runtime
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Protocol on the original stdout; everything the libraries print goes to stderr.
proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr

import torch
from transformers.generation.streamers import BaseStreamer            # put()/end() API
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from airllm import AirLLMBaseModel     # exported on non-Mac (airllm/__init__.py line 14)

MODEL_DIR = sys.argv[1]   # e.g. D:/hc-models/Qwen__Qwen2.5-72B-Instruct@<sha>
SPLIT_DIR = sys.argv[2]   # unique per model+revision; AirLLM appends "splitted_model"

def emit(obj):
    proto.write(json.dumps(obj, ensure_ascii=False) + "\n")

class JsonTokenStreamer(BaseStreamer):
    """First put() is the prompt; each later put() is one new token (CUDA path)."""
    def __init__(self, tokenizer):
        self.tok, self.prompt_seen, self.ids, self.sent = tokenizer, False, [], 0
        self.t_last = time.monotonic()
    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if value.dim() > 1:
            value = value[0]
        self.ids.extend(value.tolist())
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        now = time.monotonic()
        if not text.endswith("\ufffd"):               # hold back a split multi-byte char
            emit({"type": "token", "text": text[self.sent:], "n": len(self.ids),
                  "dt_s": round(now - self.t_last, 3)})
            self.sent = len(text)
        self.t_last = now
    def end(self):
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if len(text) > self.sent:
            emit({"type": "token", "text": text[self.sent:], "n": len(self.ids)})

class CancelCriteria(StoppingCriteria):
    def __init__(self, event):
        self.event = event
    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(),
                          dtype=torch.bool, device=input_ids.device)

def main():
    t0 = time.monotonic()
    model = AirLLMBaseModel(
        MODEL_DIR,
        device="cuda:0",
        dtype=torch.bfloat16,
        layer_shards_saving_path=SPLIT_DIR,
        prefetching=True,          # A/B against False on this host (section 7)
        compression=None,          # compression is slower (#330)
        delete_original=False,     # broken on Windows (#297)
        profiling_mode=False,
    )
    tok = model.tokenizer
    emit({"type": "ready", "load_s": round(time.monotonic() - t0, 1)})

    cancel = threading.Event()
    requests = []
    def reader():
        for line in sys.stdin:
            msg = json.loads(line)
            if msg.get("type") == "cancel":
                cancel.set()
            else:
                requests.append(msg)
    threading.Thread(target=reader, daemon=True).start()

    while True:
        while not requests:
            time.sleep(0.05)
        req = requests.pop(0)
        cancel.clear()
        try:
            enc = tok.apply_chat_template(req["messages"], add_generation_prompt=True,
                                          tokenize=True, return_dict=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            gen_kwargs = dict(
                attention_mask=enc["attention_mask"].to(model.device),
                max_new_tokens=int(req["max_new_tokens"]),        # always explicit
                use_cache=True,
                do_sample=bool(req.get("temperature", 0) > 0),
                streamer=JsonTokenStreamer(tok),
                stopping_criteria=StoppingCriteriaList([CancelCriteria(cancel)]),
                return_dict_in_generate=True,
            )
            if gen_kwargs["do_sample"]:
                gen_kwargs.update(temperature=float(req["temperature"]),
                                  top_p=float(req.get("top_p", 1.0)))
            t1 = time.monotonic()
            out = model.generate(input_ids, **gen_kwargs)
            new_ids = out.sequences[0, input_ids.shape[1]:]
            emit({"type": "done", "id": req.get("id"), "new_tokens": int(new_ids.numel()),
                  "cancelled": cancel.is_set(), "gen_s": round(time.monotonic() - t1, 1)})
        except Exception as e:                       # report, keep the worker alive
            emit({"type": "error", "id": req.get("id"), "error": f"{type(e).__name__}: {e}"})

if __name__ == "__main__":
    main()
```

Recipe notes:
- APIs used from AirLLM:
  - `AirLLMBaseModel(...)` with keywords from the constructor in section 3.2
  - `.tokenizer`, `.device` and `.generate()`
- APIs used from transformers: `BaseStreamer`, `StoppingCriteria`, `StoppingCriteriaList`, `apply_chat_template(..., return_dict=True)` (v5.16.1 `tokenization_utils_base.py` signature, `return_dict: bool = True`), and `generate` kwargs.
- **Always pass explicit sampling arguments.** Qwen2.5-Instruct's `generation_config.json` (loaded by AirLLM, `airllm_base.py` lines 218-222 and 366) defaults to sampling. EOS ids also come from that file, so generation stops at `<|im_end|>` without extra code.
- **For calibration**, construct with `profiling_mode=True`, generate 2 tokens, and read `sum(model.profiler.profiling_time_dict["load_safe_tensor"])`. Store measured s/token per model so the supervisor can set timeouts (for example `1.5 × (1 + max_new_tokens) × s_per_token`) and show an ETA.
- **First load of a new model performs the split.** That is minutes to hours of disk IO with no protocol output; `tqdm` goes to stderr. Run the split as a separate, explicitly requested "prepare model" job, not inside a chat request.
- **Batch > 1 is the throughput lever**, as the same weight IO serves every sequence. It needs left padding, a pad token, and no `TextStreamer`. Implement it only if a batch API is wanted.

---

## 9. Where AirLLM fits in Hypercompute

Requirement: AirLLM only for much larger models, or where necessary, so latency and tokens/s stay high. Proposed routing gate:

1. Default to llama.cpp, in order:
   - local CUDA0
   - then RPC workers
   - then the new host-RAM tier (CPU layers)
2. Use AirLLM only when **all** of these hold:
   - the model, at the smallest acceptable GGUF quant, does not fit in VRAM + RPC + host-RAM budget, or no GGUF / llama.cpp support exists for the architecture, or the user explicitly asks for unquantized bf16 output
   - the architecture is on the allowlist (`Qwen2ForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM` to start)
   - free NVMe space covers 2x the model size during the split
   - the request is marked batch/slow, with an ETA computed from the stored per-model s/token calibration
3. Never route an interactive chat to AirLLM silently. Show "about N s/token, about M min for this answer" before starting.
4. AirLLM is single-node, single-device (`device="cuda:0"`). It cannot use the M4 friend or RPC workers, and it does not use spare VRAM or RAM to cache layers. Run it exclusively on the host GPU; do not co-schedule it with a llama.cpp job.
5. INFERRED comparison to benchmark before finalizing the gate:
   - For a model that llama.cpp can run from a Q4 GGUF with weights spilling beyond RAM (mmap'd CPU layers, `CPU_Mapped` in `llama_placement.md` section 2), the per-token disk traffic is only the spilled portion, at about 0.56 bytes/param.
   - AirLLM reads the whole model at 2 bytes/param every token.
   - For a 70B on this host that is roughly 10x less IO for llama.cpp. AirLLM's real niche is "no GGUF/arch support" or "bf16 fidelity", not merely "big".
6. The Mac path (MLX, Llama/Mistral only, greedy, no EOS) is documented in `later_engines_and_prior_art.md` B. It is not part of this plan; the M4 Air goes through EXO.

---

## 10. Gotchas checklist

- `max_seq_len` does nothing (line 207 only). Bound prompt + `max_new_tokens` ourselves.
- No `revision` parameter. Pre-download pinned revisions.
- `.bin` index is preferred over safetensors when both exist (lines 566-572), which means a pickle load. Download safetensors only.
- `layer_shards_saving_path` collisions across models (section 3.4). Make it unique per model + revision.
- `delete_original=True` crashes on Windows (#297). Delete originals ourselves after verifying the `.done` markers, if at all.
- `check_space` for 4bit over-estimates by about 3.6x (line 474). Irrelevant with `compression=None`.
- The `prefetching` README note "For now, only AirLLMLlama2 supports this" (README line 179) is stale. The generic base implements it (lines 186-193, 900-904).
- The README compression "3x speed up" claim is contradicted by measurement (#330).
- There is no progress output during `generate`. #364 mistook slowness for a hang. The worker's per-token `dt_s` and a supervisor heartbeat ("layer N/M" is not exposed, so use a wall-clock heartbeat) are required.
- `import airllm` prints `>>>> bitsandbytes installed` when bnb is present (`airllm_base.py` line 27) and may warn about optional families. Keep stdout clean via fd redirection.

---

## 11. UNVERIFIED items and the smoke test that settles them

| Item | Why unverified | Settle by |
|---|---|---|
| Effective disk->GPU throughput and s/token on this laptop (NVMe model, WDDM, 5070 Ti Laptop) | Not run (audit rule: no execution) | `profiling_mode=True`, Qwen2.5-7B-Instruct bf16 (fits the file cache), 8 tokens, `prefetching` True vs False |
| Whether per-module `empty_cache` / `pin_memory` dominate on Windows | Hypothesis from code plus #364 | Same A/B; optionally compare against a WSL2 ext4 copy of the split |
| torch cu130 wheels on driver 591.84 | Driver table not re-read | Use cu128 2.11.0, which has a field report |
| `accelerate` / `huggingface-hub` exact versions with transformers 5.16.1 | No published matrix | Freeze after the first green run |
| Deleting original shards after a complete split is safe | Inferred from code paths | Split a small model, delete `model-*.safetensors`, reload offline |
| bitsandbytes on sm_120 / Windows | Not needed; not checked | n/a (do not install) |
| GPTQ/AWQ/FP8 pre-quantized checkpoints via `AutoHfQuantizer` on Windows + sm_120 | Kernel availability unknown | Out of scope for v1 |
| Mid-forward behaviour on `Popen.kill()` (GPU reset, no driver TDR) | Not run | Kill during a decode step in the smoke test and confirm VRAM is released (`nvidia-smi`) |

---

## Addendum (critic)

Completeness review for implementing an AirLLM subprocess worker on Windows + CUDA sm_120 **and on macOS MLX**. The
CUDA path above is well covered. The Mac path was deferred to `later_engines_and_prior_art.md` B, which left streaming,
install and failure modes UNVERIFIED. This addendum resolves that from source at `bc47e0d` (read with
`gh api -H "Accept: application/vnd.github.raw" "repos/lyogavin/airllm/contents/<path>?ref=bc47e0d474abf03f491f7547085bd85a2c6f0b1b"`;
files `air_llm/airllm/airllm_llama_mlx.py`, `persist/mlx_model_persister.py`, `persist/model_persister.py`, `utils.py`,
`auto_model.py`, `__init__.py`, `README.md`). Nothing was installed or run.

### M1. Install and constructor on macOS (VERIFIED unless marked)

- `setup.py` lists neither `mlx` nor `psutil` (lines 30-42). The README only says "make sure you installed mlx and torch"
  (line 186) and "only Llama-style models (`LlamaForCausalLM`, `MistralForCausalLM`) run on macOS" (line 189).
- torch is a hard runtime import on the Mac too: `mlx_model_persister.py:8` `import torch`, and line 79 converts tensors with
  it. A CPU torch wheel is enough (INFERRED: only `.to(torch.float16).numpy()` is used there).
- The MLX APIs the file uses exist in the current MLX release `v0.32.4` (2026-10-14):
  `create_additive_causal_mask(N, dtype=mx.float32)` (`python/mlx/nn/layers/transformer.py:96-100`) and
  `RoPE.__init__(dims, traditional=False, base=10000, scale=1.0)` (`positional_encoding.py`). Compatibility with any other
  MLX version is UNVERIFIED.
- Recipe: a dedicated venv on the Mac with `torch`, `mlx`, `psutil`, then airllm at the pinned SHA (§1 install spec).
- `AirLLMLlamaMlx(model_local_path_or_repo_id, device="cuda:0", dtype=None, max_seq_len=512, layer_shards_saving_path=None,
  profiling_mode=False, compression=None, hf_token=None, prefetching=True, test_nonlayered=False, show_memory_util=False,
  delete_original=False)` (`airllm_llama_mlx.py:233-236`). Only the path, `layer_shards_saving_path`, `compression`,
  `hf_token`, `delete_original`, `test_nonlayered` and `show_memory_util` are used (lines 238-265). `device`, `dtype`,
  `max_seq_len`, `profiling_mode` and `prefetching` are accepted and ignored. Always pass `compression=None`; the
  compression helper calls `.cuda()` (§2.1).
- `AutoModel.from_pretrained` on darwin probes the config with `trust_remote_code=True` before the arch check
  (`auto_model.py` `get_architecture` + `from_pretrained` darwin branch). Construct `AirLLMLlamaMlx` directly after our own
  `config.json` allowlist check, as §3.5 recommends for CUDA.

### M2. Split format and per-token IO on the Mac (VERIFIED)

- Each module is written as `np.savez(saving_path / (layer_name + 'mlx'), **weights)` with
  `weights = {k: v.to(torch.float16).numpy() ...}`, then a `<module>mlx.done` marker (`mlx_model_persister.py:77-88`), giving
  `model.layers.0.mlx.npz` + `model.layers.0.mlx.done`. Completeness is checked on those two names (lines 66-75). This split
  is not interchangeable with a CUDA split of the same model.
- **Weights are always stored as float16**, including bf16 checkpoints such as Llama 3.x. Numerical drift or overflow from
  bf16 to fp16 is INFERRED as a risk and not measured. `np.savez` is uncompressed, so the split is about 1x the bf16 size.
- Every token re-creates and reloads every module (embedding, each `TransformerBlock`, norm, lm_head) with `mx.load(...npz)`
  inside the decode loop (lines 396-461). There is no prefetch thread and no layer cache.

### M3. Streaming, stopping and cancellation for a worker (VERIFIED code, behaviour INFERRED)

- `model_generate(self, x, temperature=0, max_new_tokens=None)` (line 289) is a Python generator. It yields one `mx.array` of
  shape `[1]` per token (lines 384, 461). `generate()` (275-287) just collects it into a decoded `str`.
- A worker should iterate `model_generate` itself, call `.item()` on each token, decode incrementally (as `JsonTokenStreamer`
  does in §8) and stop on EOS. The generator never stops by itself (`while True`, line 391) and never checks EOS.
- Both keyword arguments of `model_generate` are ignored. `sample(y)` is called without a temperature (lines 376, 452), so
  decoding is always greedy, and there is no count check. Read EOS ids from `generation_config.json` / the tokenizer ourselves.
- Input must be a 2-D `[1, L]` id array. The mask is built from `x.shape[1]` (line 294) and decode feeds `y[:, None]`
  (line 394). No chat template is applied, so call `model.tokenizer.apply_chat_template(..., return_tensors="np")` first.
  Batch size 1 only (no attention mask or padding).
- Cancellation: stopping the iteration (or `gen.close()`) between tokens is clean. Inside a token the per-layer loop cannot be
  interrupted, so the supervisor's kill is still the hard stop.
- Free progress signal: both the prefill and each decode step wrap the layer loop in `tqdm(..., desc='running layers')`
  (lines 328, 414). That prints `running layers: k/N` to stderr, which the supervisor can parse as a per-layer heartbeat (the
  CUDA path has none, §10).

### M4. Memory and correctness traps on the Mac

- **The KV cache is GQA-expanded and grows by concatenation** (VERIFIED). K/V are repeated to `n_heads` before they are
  cached (`repeat`, lines 126-130), and every step runs `mx.concatenate` over the full cache (132-137).
  - Cost per token is `n_layers x 2 x n_heads x head_dim x 2 bytes`. For Llama-3.1-70B (80 layers, 64 heads, head_dim 128,
    8 KV heads) that is 2.5 MiB per token, 8x the GQA size. 2k tokens is about 5 GiB of unified memory, on top of one fp16
    layer of about 1.7 GB.
  - On the 16 GB M4 Air, bound prompt + new tokens from this formula (INFERRED arithmetic).
- **RoPE scaling is ignored** (VERIFIED). `get_model_args_from_config` reads only `rope_theta` (lines 64-69), and `nn.RoPE` is
  built from `base` alone (107-109).
  - Llama 3.1/3.2/3.3 `rope_scaling` (`rope_type: llama3`) is never applied. How far output then differs from transformers
    is UNVERIFIED, so keep contexts short.
  - The file has no sliding-window code either, so Mistral's window is not honoured (INFERRED).
- **Tied-embedding checkpoints fail on the Mac** (INFERRED from code, not run).
  - The shared split drops modules with no weights, with the comment "This happens for tied embeddings" (`utils.py:609-614`).
    So no `lm_head.mlx.npz` is written.
  - `model_generate` always loads `lm_head` (`airllm_llama_mlx.py:364-367`, `449-451`), so `mx.load` fails right after the
    first full prefill.
  - Llama-3.2-1B/3B (`tie_word_embeddings: true`) are therefore not valid Mac smoke-test models. Pick a checkpoint whose
    `config.json` has `tie_word_embeddings: false` and add that check to the Mac allowlist.

### M5. Open questions found by this review and not resolved here

| Gap | Status |
|---|---|
| Driver 591.84 with cu130 torch wheels (§2.3) | still UNVERIFIED; cu128 2.11.0 remains the safe pick |
| Whether `mx.load` of an `.npz` streams or materialises the whole module (peak unified memory) | UNVERIFIED |
| Quality impact of fp16 storage and missing `llama3` RoPE scaling on the Mac path | UNVERIFIED; needs a side-by-side run against transformers |
| Effective disk throughput and s/token on the M4 Air SSD | UNVERIFIED; measure with the tqdm per-layer timings |
