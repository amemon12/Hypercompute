# Hybrid capacity: implementation plan and module contracts

Working plan for turning [the architecture](hybrid-capacity-architecture.md) into code on branch
`codex/iroh-transport`. It records what is built and measured, what each new module must do, and the gates
between phases. Status words: **built** (code exists and its tests pass), **measured** (run on this hardware,
evidence file named), **planned** (not yet written).

## Decisions

**D1. Bounded residency is a native extension of the stock engine, not a rebuild.** The pinned b11770 Windows
release ships `ggml.dll` with 16 plain C exports. `native/hcstream/hcstream.c` builds a proxy `ggml.dll` (Zig C
toolchain from PyPI, no Visual Studio) that forwards those exports to the renamed original and, only when
`HC_STREAM_CONFIG` is set, patches import tables so that `llama_init_from_model` installs a ggml scheduler eval
callback. The overlay engine directory is hard links to the release plus the proxy, content addressed by the proxy
hash, outside `engines/` so the live gateway never sees it. CUDA, Metal over RPC and the CPU kernels are untouched.

**D2. Residency per placement unit.** A CPU unit (block or the output head) is `resident` (copied once into private
memory with unbuffered reads) or `streamed` (read from the GGUF with `FILE_FLAG_NO_BUFFERING` into one of K slot
buffers just before its first consumer runs). Streamed bytes never enter the file cache, so SSD reads are physical
and the bound is `K * slot_bytes + resident_bytes`. Weight tensors keep their `CPU_Mapped` buffer; only
`tensor->data` is rebound, and an evicted unit points back at the mapping, which always holds correct bytes.

**D3. Slot safety comes from the scheduler's own synchronization.** The callback sees each node before its pending
range runs and the scheduler synchronizes after every range. Layers referenced by pending nodes are pinned; with
K >= 2 the callback stops when a second layer enters the pending range, with K = 1 at every streamed consumer.
Prefetch only targets unpinned slots. Teardown joins the readers before freeing anything, and a slot epoch makes
queued reads for a recycled slot no-ops, so a cancelled prefetch cannot write into freed or reassigned memory.

**D4. Every number has provenance.** `Coefficient.provenance` is `measured`, `prior`, `estimate` or `reported`.
Placements keep their per-stage breakdown and the list of priors they relied on. Planner output never presents a
prior as a measurement.

**D5. Latency is a sum.** Single-stream decode is `sum(stage compute + exposed I/O + hops)`. `max(stage)` is a
filled-pipeline throughput bound and is not used for interactive latency.

**D6. Replacement is reload at a request boundary.** observe → plan → admit → drain → reload → verify → activate,
with revisioned leases and rollback only when the previous resources still exist. No live KV migration.

## Measured so far (CPU-only canaries, isolated child, live 32B baseline untouched)

| Evidence (results/capacity/) | Model | Residency | Result |
| --- | --- | --- | --- |
| `*-stock-q05b` | Qwen2.5 0.5B Q4_K_M | stock mmap | 86–106 tok/s, WS 0.64 GiB |
| `*-stream-q05b-all24-k2` | same | 24 blocks streamed, 2 slots of 10.4 MiB | tokens identical to stock, 11.5–12.5 tok/s, 14.5–14.9 GiB physical reads per 64-token rep, cancel + follow-up identical |
| `*-stream-q7b-r8-s20-k2` | Qwen2.5 7B Q4_K_M (4.36 GiB weights) | head + 8 blocks resident (1.48 GiB), 20 streamed, 2 slots | correct, deterministic, 1.34–1.37 tok/s, private peak 2.36 GiB, WS peak 2.09 GiB, 2.6–2.8 GiB read per token |
| `*-q7b-r8-s20-k3/k4` | same | 3 / 4 slots | 1.42–1.44 / 1.45–1.48 tok/s |
| storage probe | 7B shard | unbuffered, 1–16 readers | 3.45 GiB/s at QD1 8 MiB, 5.2–5.6 GiB/s with 2+ readers |

Single-slot streaming and teardown were checked with `llama-completion`: a `--fit` probe context attaches and
detaches, the real context streams, the final stats state is `shutdown` with resident and slot buffers released.

Later measurements (same night):

| Evidence (results/capacity/) | What | Result |
| --- | --- | --- |
| `20261022T050332Z-validate` | native gate, proxy sha256 `820dee92a8ee...` | 11 of 11 cases pass: tokens identical to stock for K = 1, 2, 3 and mixed residency; every streamed layer bound from a slot on every pass (624 = 26 x 24); read fault exits 86 with `win32 error 30` and its offset; `on_io_error mmap` keeps tokens identical; malformed config exits 86 before serving; teardown shows `slot_bytes_committed` 0 and `resident_bytes` 0 |
| `20261022T040153Z-link-relay-*` | M4 Air over the Iroh relay, raw llama RPC probe | RTT median 36.9 ms (p90 62.9), goodput 1.59e6 B/s (one 8 MiB upload) |
| `20261022T054244Z-link-relay-*` | same link, 1.6 h later | RTT median 17.5 ms (p90 25.1), goodput 1.58e6 B/s; stored in the measurement store |
| `20261022T040652Z-bench-smoke-*` | bench harness smoke, 0.5B stock CPU, machine busy | decode 115 tok/s median from engine timings (n-1 steps), not a benchmark |

Not measured yet: CPU and CUDA decode/prefill coefficients and the storage rate in the measurement store (calibration
refuses to run on battery, which is where the host was), stream efficiency, predicted versus measured ranking, and any
two- or four-device run with the new code.

## Build status (checkpoint commit 5574d33)

| Module | Status |
| --- | --- |
| `native/hcstream` | built, gate passed (above) |
| `capacity/stream_config.py`, `overlay.py`, `probe.py`, `storage_probe.py`, `runtime.py` | built; render_config refuses everything the native parser would kill the child for |
| `capacity/manifest.py`, `pools.py` | built; exact unit bytes, UMA counted once, leases not subtracted twice |
| `capacity/costs.py`, `optimizer.py` | built; exact DP plus Pareto sets for shared pools, brute-force oracle agreement on 3,000 random cases |
| `capacity/bridge.py`, `service.py`, `capacity_cli.py`, HTTP routes | built; the optimizer plan replaces the planner's only on a strictly lower predicted decode |
| `capacity/linkprobe.py`, `measurements.py`, `benchlib.py`, `scripts/capacity/{bench,calibrate,probe_link}.py` | built; calibration not yet run on AC |
| `services/leases.py`, `deployments.py` (replace), `replan.py` | built; revisioned leases, drain, rollback, fallback, dwell/hysteresis/break-even/cooldown |
| `services/streaming.py` (interrupt and resume) | built; resume continues by assistant prefill and never resends displayed text |

Corrections to the contracts below, found while building: the decode priors 2.64 / 10.85 / 12.5 are ms per GB
(1e9 bytes), 2.835 / 11.65 / 13.42 ms per GiB; a remote stage that holds the head returns `n_vocab * 4` bytes of
logits per token, not one activation; on a shared pool the CPU split is `split_cpu` for the bytes the other stages
on that pool leave.

## Module contracts (backend/hypercompute/capacity/)

Shared value types live in `types.py` (frozen dataclasses). Modules return new values; nothing mutates inputs.

### `stream_config.py` (built)
GGUF tensor table → per-unit aligned layouts (mirrors `hc_build_runs`), `stream_plan(...)`, `render_config(...)`.

### `overlay.py` (built), `probe.py` (built), `storage_probe.py` (built)
Overlay construction and validation; per-process memory/I/O sampler; unbuffered read throughput probe.

### `manifest.py` (built)
`build_manifest(model_id, shards, *, revision=None, shard_hashes=None) -> ModelManifest`. Uses
`gguf.read_tensor_table` and metadata. Units: blocks `0..n_layer-1` plus the head at `n_layer` (output tensors, or
token_embd when embeddings are tied). `buffer_bytes` must equal `stream_config.layer_layouts` for the same unit.
`fingerprint` hashes shard sizes/hashes and every tensor's name, type, dims and offset. Rope keys are any metadata key
containing `rope`; the chat template is hashed, never stored. `streamable` is false if any block/head tensor has an
unknown ggml size. Cached by (paths, sizes, mtimes).

### `pools.py` (built)
- `pools_from_nodes(nodes, *, host_reserve_bytes, now) -> tuple[Pool, ...]`: one pool per physical pool per node.
  A Mac's CPU and MTL0 share `uma`; Windows has `ram` and `vram:CUDAn`; the host adds `ssd` (free volume space,
  refreshed). Contributor `reserve_bytes`/`max_bytes` become `reserve_bytes`/`cap_bytes`.
- `admissible_bytes(pool, leases) -> int`: `min(free - reserve - unmaterialized leases, cap - all leases)`, floored
  at zero. Materialized leases are already inside `free` and are not subtracted twice.
- `admit(demands, pools, leases) -> Admission`: per-pool sum of `PoolDemand.peak` against `admissible_bytes`.
- `host_memory_snapshot() -> HostMemory`: available, total, commit charge/limit, system file cache (Windows
  `GetPerformanceInfo` via ctypes; psutil elsewhere), with `observed_at`.

### `costs.py` and `optimizer.py` (built)
- `costs.py`: documented priors (cuda 2.64, metal 10.85, cpu 12.5 ms/GiB decode; prefill as a fraction; RTT and
  goodput defaults per path class; 2 RPC round trips per remote stage per token; storage 3.0 GiB/s, stream
  efficiency 0.7) as `Coefficient(provenance="prior")`, and `merge(prior, measured)` that prefers fresh measurements
  for the same identity key.
- `optimize(manifest, workload, host, peers, *, stream_supported, slots=2, max_candidates=5) -> OptimizeResult`:
  exact search over llama.cpp-expressible layouts: CPU prefix (resident/streamed split) → zero or more peers in a
  chosen order, each a contiguous span, skipping any peer → local GPU suffix holding the head (or the head on the
  last stage when there is no GPU). Dynamic programming over unit boundaries per peer order; a brute-force oracle in
  tests for small models. Memory per stage uses `PoolDemand` with exact unit bytes, per-layer KV, workspace and
  staging; two devices on one pool share it. The CPU resident/streamed split enumerates the slot size over distinct
  streamed unit sizes and fills the remaining budget with the largest units resident.
- Stage time: compute = `decode_ms_per_gib * weight GiB`; CPU streaming: `io = stream_bytes / (read * efficiency)`,
  stage = `max(io, compute)` when K >= 2 else `io + compute`; remote hop = `round_trips * rtt +
  activation_bytes / goodput` (activation = `n_embd * 4` bytes each way). Cold load = uncached weight bytes /
  link goodput + resident read bytes / storage + setup prior. `J = (warm ? 0 : load) + prefill + output_tokens *
  decode + risk`, risk a documented margin for relay paths.
- `OptimizeResult.explain()` returns plain lines: per stage the units, resident/streamed bytes, demand vs
  admissible, ms terms with provenance, the bottleneck, and skipped devices with reasons.

## Lifecycle contracts (built)

- `Plan.revision`, `Deployment.revision`, lease `revision`: monotonically increasing per served model alias;
  release of a lease with a stale revision is rejected.
- `DeploymentManager.replace(dep_id, plan, reason)`: under the create lock; drain via exclusive inference
  acquisition with timeout; stop old; fresh admission; load; verify participation and a short probe; activate; on
  failure redeploy the previous plan if its pools still admit it, else the best alternative, else fail with the
  reason. Old leases are released only after old cleanup.
- `ReplanController`: triggers on `node.verified`, `node.state`, `node.left`, sustained pool pressure and measured
  drift. A decision needs stable inputs for `dwell_s`, relative gain above `hysteresis`, savings over
  `horizon_tokens` above reload + replay cost, and `cooldown_s` since the last switch. Every decision, taken or not,
  is recorded with its numbers for the CLI.
- Interruption: a stream cut by node loss ends with an `interrupted` error carrying the run id and the exact text
  already sent. Resume re-renders the conversation plus that partial assistant text and streams only new tokens.

## Gates

| Gate | Evidence required |
| --- | --- |
| Native tier | bit-identical tokens vs stock on the 0.5B for K = 1, 2, 3 and mixed residency; fault injection (read error → exit 86 with reason; `on_io_error mmap` → correct output and a fallback count); malformed config refused; teardown releases buffers |
| Planner | exact unit sizes; UMA counted once; infeasible plans give reasons; oracle agreement; predicted ranking vs measured on at least three local candidates |
| Replacement | join → evaluate → replace with revision bump; slow peer skipped; failed load rolls back or falls back; stale lease release rejected; no duplicate tokens on resume |
| Demo | same artifact, context, prompt, output length on 2 vs 4 devices with participation and SSD bytes/token per run |
