# Checkpoint 2026-10-22 (about 09:15 UTC)

Branch `codex/iroh-transport`, draft PR #3, pushed at 352b2e0 plus this file. Everything below is committed; nothing is
running except the gateway.

## Where things stand

| Area | State | Evidence |
| --- | --- | --- |
| 117B on the gateway | gpt-oss-120b MXFP4 served through the gateway, laptop alone 4.05 tok/s, laptop + M4 Air 4.52 tok/s | `results/capacity/20261022T0900Z-gptoss120b-summary.md` |
| MoE placement | `moe_offload` tier: dense parts, head and KV on the GPU; routed experts of host layers file-backed on the CPU; peers take whole layers | `services/moe_tier.py`, `tests/test_moe_placement.py` |
| Bounded streaming (dense models) | hcstream proxy DLL, 11-case native gate passes | `results/capacity/20261022T050332Z-validate/summary.json` |
| Placement optimizer | exact search with measured coefficients; predicted layout order matched measured on the 7B and 14B | `results/capacity/20261022T072404Z-ranking-*`, `*T072859Z-ranking-*` |
| Calibration | CPU 16.96 ms/GiB, CUDA0 2.93 ms/GiB, SSD 5.998e9 B/s, stored in `~/.hypercompute/capacity/measurements.json` | `results/capacity/20261022T0625*`, `*T0628*`, `*T0629*` |
| Replacement and replan | revisioned leases, drain, rollback, fallback, replan controller (off unless `HYPERCOMPUTE_REPLAN=1`) | `tests/test_replace_lifecycle.py`, `tests/test_replan_controller.py` |
| Interrupt and resume | interrupted streams carry run id and exact text; resume by assistant prefill, no duplicated text | `tests/test_stream_resume.py` |
| Tests | full non-hardware suite last run green (1586+ passed, 3 skipped); focused suites green after the last edits | run `uv run --no-sync pytest -q -p no:cacheprovider` |

## Runtime state on the G14

- Shut down at about 09:30 UTC at the user's request: the gateway on 8001 and its launchers were stopped, no engine
  or worker process is left, and the GPU is idle. To bring it back, use the start commands in
  `docs/research/big-model-runbook.md` section 4.4 (same `HYPERCOMPUTE_HOME`, so the Mac can rejoin with its existing
  credential if its agent is still running; if its agent stopped after 5 minutes without the host, the friend re-joins
  with a fresh invitation as in section 5).
- Before the shutdown, the live gateway on `127.0.0.1:8001` ran this branch's code (started from the worktree with
  `HYPERCOMPUTE_HOME=C:\Users\operator\.hypercompute\iroh-session`, logs `iroh-session\host-worktree2.*.log`). It has no
  deployment loaded. The friend's M4 Air (node `4c299174...`) is joined, eligible, on a direct Iroh path while both
  machines share the phone hotspot.
- The 890f798 gateway database is backed up in `iroh-session\coordinator-backup-890f798-20261022T013545`. Rolling back
  needs that restore because the old code rejects the new rows (`docs/research/big-model-runbook.md` section 6).
- Models in `iroh-session\models`: `gpt-oss-120b-mxfp4` (59.03 GiB, verified), `qwen2.5-0.5b-instruct-q4_k_m`.
  `~\.hypercompute\models` still holds the 0.5B and 7B. The Qwen2.5 32B, 14B Q6_K, Mistral 24B and two HF caches were
  deleted with the user's approval. C: has about 13 GiB free.

## How to run the 120B again

```powershell
cd C:\Users\operator\.hypercompute\worktrees\iroh-transport
# both machines on the same network without client isolation (the phone hotspot worked; the guest Wi-Fi did not)
uv run --no-sync python scripts/capacity/run_big.py --model gpt-oss-120b-mxfp4 --strategy all_selected `
  --home C:\Users\operator\.hypercompute\iroh-session --prompt-tokens 128 --output-tokens 64 --warmup 1 --reps 2 `
  --load-timeout 5400 --chat-timeout 1800 --min-available-gib 0
# laptop alone: --strategy local_only
```

`--min-available-gib 0` is required for MoE runs: mapped expert pages take available RAM to zero while commit stays
far below its limit. The commit-growth and drive-space guards stay on. The Mac keeps an RPC cache, so a second load of
the same layers took 40 s instead of 521 s.

## Next (the four-device demo)

1. Join the two extra M4 Airs to the same network; check each is eligible and above its reserve.
2. Run the 120B with `all_selected` on 2, 3 and 4 devices, same workload, and record decode, first token and SSD
   reads per token for each. Expect the planner to give each Mac whole layers.
3. Fix the planner's MoE miss-rate estimate (it predicted 1,081 ms/token and 1.13 GB SSD per token; measured 221 ms and
   121 MB) by modelling expert reuse across tokens, then calibrate it from these runs.
4. Run the replan controller live: join a Mac while a deployment runs and check it proposes and performs the
   replacement with the measured change recorded.
5. Rerun the 14B stream layout that was skipped (`s16`, power check read "unknown") and add a stream-efficiency
   calibration.

## Known gaps

- Two measured reps per configuration tonight; repeat with more before quoting percentages publicly.
- The replan controller and live KV migration have not run against real joins.
- `deployments.py`, `bridge.py` and `cli.py`-adjacent files are over the 800-line guideline.
- The frontend does not show the new tiers, predictions or resume.
