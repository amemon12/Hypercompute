# Source audit and pinned versions

Detailed, line-referenced audits live in `docs/research/` (read from source with the GitHub API, nothing
from external repos was executed). This page records the decisions that depend on them.

## Engine pin: llama.cpp b11770 (commit 39da130f)

| Item | Value | Evidence |
| --- | --- | --- |
| Handover audit commit | `9a3b91c6801e68f412fcf1b6f8e153aa562b8536` resolves on master (2026-10-20T23:19Z) | `gh api repos/ggml-org/llama.cpp/commits/9a3b91c6...` |
| Pinned release | `b11770` = `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb` | release `target_commitish`; `configs/engines.lock.json` |
| Why not 9a3b91c6 | no prebuilt release existed for it; the first host has no CMake/MSVC/CUDA toolkit | `uv run hypercompute doctor` on the G14 |
| Delta 39da130f..9a3b91c6 | 12 commits, files only under tools/server, tools/mtmd, src/models, conversion, hexagon; none under ggml-rpc, ggml-backend or common/arg.cpp | compare API (docs/research/llama_rpc.md) |
| RPC protocol | 8.0.0 (`RPC_PROTO_MAJOR/MINOR/PATCH`), binary target `ggml-rpc-server` | `ggml/include/ggml-rpc.h`, `tools/rpc/CMakeLists.txt` |
| Release build flags | `-DGGML_RPC=ON` for every platform build | `.github/workflows/release.yml` line 38 |
| Asset integrity | GitHub `digest` sha256 per asset, verified by the installer before extraction | `configs/engines.lock.json` |

Every RPC peer must run the same release: the client only rejects `major != 8 || minor > 0` and never compares
build tags, so Hypercompute checks release and RPC protocol at join and again in the worker probe.

## Facts the implementation depends on (verified against b11770)

| Fact | Where it is used | Verified by |
| --- | --- | --- |
| `--list-devices` prints `  NAME: DESC (TOTAL MiB, FREE MiB free)`; RPC devices print `RPCn: host:port`; `--rpc` must precede `--list-devices` | worker probe, engine smoke test | real output, `tests/fixtures/list_devices_rpc_b11770.txt` |
| An unreachable `--rpc` endpoint aborts llama-server at argument parsing | probe before launch; never launch with an unverified peer | source + real run (`Failed to connect to ...`) |
| Placement lines (`model/KV/compute buffer size`, `offloaded N/M layers`) are only printed at `-lv 4` | llama-server is launched with `-lv 4` | real log, `tests/fixtures/llama_server_rpc_split_b11770.log` |
| Buffer names are `RPC<index within server>[host:port]`; device names are `RPC<global index>` in `--rpc` order | endpoint-based matching of observed buffers; `assign_rpc_names` | docs/research/llama_placement.md |
| Layer unit `il` (0..n_layer, output = n_layer) goes to the first device whose cumulative split exceeds `il/(n_layer+1)` | planner passes integer unit counts as `--tensor-split` | every real run: planned layers == observed layers |
| `--device` order is honored when it follows `--rpc` | `build_args` | real runs |
| KV bytes per layer = 4 * n_ctx * n_head_kv * head_dim (f16) | planner | 2 MiB/layer observed for Qwen2.5-0.5B at ctx 4096 |
| `ggml-rpc-server` serves one client at a time, has no auth, and its `COMM_INIT` command can make it listen on `0.0.0.0:port+1000` or connect to any host | never probe a busy worker; firewall scoping documented as required | docs/research/llama_rpc.md |
| Any RPC send/recv failure aborts llama-server; a silent drop can hang it | heartbeat staleness fails the deployment and stops llama-server | real peer-loss run (16 s detection) |
| CPU RPC devices report free = total memory | remote budgets take the lower of probe free and heartbeat RAM free | docs/research/llama_rpc.md |
| `-c/--cache` stores tensors >10 MiB under `$LLAMA_CACHE/rpc`, never deleted, no size cap | cache is opt-in, LLAMA_CACHE points at `~/.hypercompute/rpc-cache`, `leave --purge-cache` | docs/research/llama_rpc.md |
| Bind failure prints `Failed to create server socket` and still exits 0 | agent parses the banner instead of trusting the exit code | real run on 50199 |
| Windows Hyper-V/WSL reserve 50000-50059 (includes the conventional 50052) | default worker port 47052, bind test with fallback scan | `netsh int ipv4 show excludedportrange` on the G14 |
| `/health` returns 503 while loading, 200 when ready; load failure exits the process | readiness = 200 plus warmup request | docs/research/llama_server.md |
| Streaming final chunk carries `finish_reason`; `usage` needs `stream_options.include_usage`; `timings` rides on the last chunk | run records | real stream, `tests/fixtures/llama_stream_b11770.sse` |
| `X-Conversation-Id` keeps generating after disconnect | gateway never forwards client headers | docs/research/llama_server.md |
| `cache_prompt` request field (default true) reuses prompt prefixes | benchmarks send `cache_prompt: false` | `tools/server/server-schema.cpp:31` |
| `LLAMA_API_KEY` env sets the server API key | per-deployment key, kept out of argv | `llama-server --help` |

## Later phases (not implemented, pins recorded)

| Engine | Pin | License | Notes | Source |
| --- | --- | --- | --- | --- |
| EXO | `bc412e2230f395e652c23426a5eb88dfa515769b` (2026-10-13, main tip) | Apache-2.0 | Python `==3.13.*`, darwin/linux only, API on 52415, mlx>=0.32 with forked mlx-lm | docs/research/later_engines_and_prior_art.md |
| AirLLM | `bc47e0d474abf03f491f7547085bd85a2c6f0b1b` | Apache-2.0 | Mac path supports Llama/Mistral layouts and returns final text | same |

## Prior art considered

GPUStack (Linux-only workers, dropped llama-box), LocalAI (no Windows build; workers fixed once inference
starts) and others were reviewed (docs/research/later_engines_and_prior_art.md section C). None offers native
Windows + macOS workers with LAN join and llama.cpp layer splitting, so Hypercompute reuses only the unmodified
upstream llama.cpp binaries and keeps its own small control plane.
