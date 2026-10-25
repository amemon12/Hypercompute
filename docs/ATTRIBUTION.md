# Attribution

Hypercompute downloads and runs unmodified llama.cpp release binaries
(https://github.com/ggml-org/llama.cpp, release b11770), MIT License,
Copyright (c) 2023-2026 The ggml authors. Binaries are fetched from the official GitHub release and
verified against the sha256 digests recorded in `configs/engines.lock.json`; they are not redistributed in
this repository.

Curated models (downloaded on demand from Hugging Face, never committed):

| Model | License | Pinned revision |
| --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct-GGUF (Qwen team, Alibaba Cloud) | Apache-2.0 | `9217f5db79a29953eb74d5343926648285ec7e67` |
| Qwen/Qwen2.5-7B-Instruct-GGUF | Apache-2.0 | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` |
| Qwen/Qwen2.5-32B-Instruct-GGUF (capacity candidate, not yet downloaded) | Apache-2.0 | `a15e3cc10f8bbb2c0af6f8f1f34a32e3b060c09d` |

Later-phase engines (not used by the MVP, nothing copied): EXO (exo-explore/exo) Apache-2.0;
AirLLM (lyogavin/airllm) Apache-2.0. dnet (firstbatchxyz/dnet) is a design reference only.

Python dependencies are listed in `pyproject.toml` / `uv.lock` (FastAPI, Uvicorn, Pydantic, httpx, PyYAML,
psutil); frontend dependencies in `frontend/package.json` / `package-lock.json` (React, Vite, TypeScript, Vitest).
