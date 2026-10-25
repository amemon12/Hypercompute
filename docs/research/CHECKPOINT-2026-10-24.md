# Checkpoint 2026-10-24 22:20 UTC (paused after the GX10 runs)

Resume from here. Results and method: `PAPER-RUNS-2026-10-22.md` section R6. Earlier state: `CHECKPOINT-2026-10-23.md`.

## Headline results (measured, R6)

| model | config | decode tok/s | prefill tok/s | SSD MiB / token |
|---|---|---|---|---|
| gpt-oss-120b | GX10 alone (9 reps) | 56.00 | 382.0 | 0 |
| DeepSeek-V3.1 671B | GX10 alone vs GX10 + G14 + 3 Macs (4 vs 4) | 1.805 vs 1.497, 0.83x [0.77, 0.90] | 2.63 vs 6.36, 2.42x [2.19, 2.72] | 684 vs 37, 0.054x |
| Kimi-K2.6 1T | GX10 alone vs GX10 + G14 + 3 Macs (4 vs 4) | 1.799 vs 1.540, 0.86x [0.82, 0.89] | 2.19 vs 3.30, 1.51x [1.41, 1.59] | 671 vs 254, 0.38x |

mac-azim was on the Iroh relay in the 671B pooled runs and direct in the Kimi pooled runs.

## Runtime state at pause

- **GX10** (`asus@192.168.1.221`, hostname gx10-49d1, rented):
  - Host stopped, no llama-server, matrix or run_big running. 387 GB free.
  - Models in `~/.hypercompute/models`, all sha256-verified: gpt-oss-120b-mxfp4, deepseek-v3.1-ud-tq1_0, kimi-k2.6-iq1_s, qwen2.5-0.5b.
  - Engine: llama.cpp b11770 CUDA sm_121 at `~/.hypercompute/src/llama.cpp/build-hypercompute/bin`.
  - Repo `~/hypercompute`: bef2020 plus the copied driver, catalog and summary files.
  - A root-owned Nemotron 30B snap service from the rental sleeps when idle. Sudo needs the rental's password, which we never enter.
- **mac-a, mac-b and the G14**: agents stopped (`hypercompute leave`). Their RPC caches are kept: mac-a 39 GB, mac-b 42 GB. The Macs' and live-host's `configs/models.example.yaml` carry the Kimi entry as an uncommitted copy.
- **Azim's MacBook (mac-azim)**: agent stopped, about to be replaced by a Mac mini. Its 24 GB Hypercompute RPC cache is still on disk, pending the owner's choice.

## Resume

Scripts are in `scripts/capacity/paper/gx10/`:
- `host.sh`, `matrix.sh`, `smoke.sh` and `pull_models.sh` live in `~/.hypercompute` on the GX10.
- `rejoin_mac.sh`, `join_g14.sh`, `lan_watch.py` and `key_server.py` run on the G14.
- `coherence.py` runs on the GX10, from the repo.

1. On the GX10: `setsid nohup ~/.hypercompute/host.sh > ~/.hypercompute/host.log 2>&1 &`.
   - The log holds a live invite, so never print it.
   - Restart the host after any catalog change: the gateway reads its catalog at start.
2. Join the peers:
   - `bash rejoin_mac.sh mac-a operator@100.80.165.87 11.5GiB 0GiB --managed --max-disk 100GiB`.
   - The same for mac-b (`azim@100.90.34.93`).
   - `join_g14.sh` in the background for the G14 (9 GiB on CUDA0).
   - If a peer stays on the relay, restart the host once: peers that dial back after a restart came back direct.
3. Run: `~/.hypercompute/matrix.sh PLAN.json INDEX.jsonl` (sets `HC_PAPER_*` and `HYPERCOMPUTE_LLAMA_BIN_DIR`). Then `summarize.py` and `stats.py` with `HC_PAPER_REPO` set.
4. Before pooled runs of a new model, make sure every peer's catalog lists it, then rejoin the peers. Each peer agent caches its catalog at the first seed command. A missing entry makes the host push layers itself at about 4 MiB/s.

## Next, in order

1. **Concurrency feature**: N parallel slots, `-np N -kvu`, a `run_big --concurrency` batch measurement counting tokens only inside the all-active window.
   - A workflow implemented it test-first with a review, verify and fix pass.
   - Read its report, rerun the full suite, review the diff, then commit.
2. **Mac mini**:
   - Set it up: Remote Login, the key via `key_server.py` and the one-line `curl` append, then `mac_bootstrap.sh`.
   - Join it in place of mac-azim, and record its memory size in R6's setup table.
3. **Concurrency rows**, with pooled placement pinned via `--layout` from the first pooled run:
   - {671B, Kimi} x {GX10 alone, pooled} x N in {1, 4, 8}, interleaved and replicated.
   - Plus gpt-oss-120b on the gpu tier.
4. **Fewer-hop pooled configs**: the Mac mini plus the G14 only, then mac-a and mac-b only.
5. **CalHacks 13.0 pitch**: the UltraPlan workflow output (to be saved next to this file), then the user's review.
