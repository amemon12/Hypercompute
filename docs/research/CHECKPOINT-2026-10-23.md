# Checkpoint 2026-10-23 11:00 UTC (second pause for travel)

## Since the first checkpoint

- Pushed: 212639c (macOS peers count reclaimable file cache as free memory), 9fd2456 (explicit layouts may visit a
  peer again), 2b5b0ac (DeepSeek-V3.1 deepseek2 planning, output_norm counted with the head), 75423bd (671B layout
  candidates, driver waits for peers), plus this commit's results. Full suite at 2b5b0ac: 1999 passed.
- hcpeer v4 passed the S1 gate on a real Mac (identical greedy text, 7B, 3.63 GiB mapped, no slowdown).
- The 235B and 120B GGUFs were deleted from the G14 (user approved) for DeepSeek-V3.1 UD-TQ1_0, now at
  `iroh-session/models/deepseek-v3.1-ud-tq1_0` (170.5 GB, sha256 OK, `.verified.json` written).
- 671B on the G14 alone: 0.481 tok/s decode, 0.747 prefill, 2485 MiB SSD per token (`PAPER-RUNS` R4).
- The 120B demo model must be re-downloaded on good Wi-Fi before the demo (59 GiB; G14 has about 48 GiB free with the
  671B on disk, so delete the 671B first or add an external drive).

## Next

1. All four laptops on one network with device-to-device traffic (phone hotspot or a travel router). The G14's saved
   hotspot profile fails with "cipher not found": rejoin from the Wi-Fi menu with the password.
2. Run 671B layouts B and C alternating with A (`results/capacity/paper-2026-10-23/layout_671b_*.json`; mac-a and
   mac-b need 11.5 GiB caps). First load sends about 30 GB (B) or 55 GB (C) over the local link; seed from Hugging
   Face only on Wi-Fi, never on the hotspot.
3. Rerun the 235B SSD-tier comparison (C1 vs D) on a local network; tonight's hotspot batch was invalid (R5).
4. Re-download gpt-oss-120b for the demo.

# Checkpoint 2026-10-23 01:50 UTC (paused for travel)

Resume from here. Results and method are in `PAPER-RUNS-2026-10-22.md`; candidate layouts for the next experiment are
in `peer-ssd-tier-candidates.md`.

## Headline results (measured)

| model | config | decode | prefill | SSD per decoded token |
|---|---|---|---|---|
| Qwen3-235B-A22B UD-Q2_K_XL | 3 Macs vs G14 alone, steady state | 1.08x [0.98, 1.18] | 1.61x [1.48, 1.74] | 0.46x |
| gpt-oss-120b MXFP4 | 3 Macs vs alone | 1.27x [1.10, 1.46] | 2.99x [2.64, 3.35] | 0.15x |
| gpt-oss-120b MXFP4 | 1 Mac vs alone | 1.25x [1.11, 1.41] | 1.48x [1.41, 1.57] | 0.54x |

Seeding peers from Hugging Face cut a 3-Mac 120B cold start to 566 s seeding + 126 s load, against about 2400 s over
the 8.6 MiB/s host link. The measured per-Mac stage cost is about 97 ms per token on this Wi-Fi, of which only 8-37 ms
is RTT and about 8 ms Metal compute (`peer-ssd-tier-candidates.md` section 2): the rest is the open S0 question.

## Commits (all pushed to `codex/iroh-transport`, PR #3)

- 751f00b perf: Iroh RPC bridge speedups, `HYPERCOMPUTE_STOCK_BRIDGE=1` A/B switch
- db44c95 feat: Qwen3-235B catalog entry
- 2b92117 feat: QUIC path stats in `/api/network`
- a073e8f feat: host-managed peers seeding their RPC cache from Hugging Face, `run_big --seed-peers`
- 9cf03db feat: hcpeer v4 (macOS rpc-server mapping cached tensors from the peer SSD; never run on a Mac)
- f14e58a docs: ultraplan + peer SSD tier design
- 6d5016b fix: accept repo subfolders in a pinned HF resolve URL (unsloth layout) when seeding
- this commit: tonight's results, run indexes, driver scripts, paper and checkpoint docs

Full suite at a073e8f..f14e58a: 1870 passed, 4 skipped.

## Uncommitted work in this worktree (do not lose)

Workflow `wf_b1c28bb0-6fd` (peer SSD tier prep) was stopped during its fixup stage:

- Explicit layouts through the gateway: `backend/hypercompute/services/layout.py` (new), edits to `planner.py`,
  `deployments.py`, `moe_tier.py`, `adapters/llama_output.py`, `scripts/capacity/run_big.py --layout`,
  `tests/test_explicit_layout.py`, `tests/test_run_big.py`. Review found a HIGH in `services/layout.py`; its fixup was
  running.
- hcpeer on the Mac agent: `backend/hypercompute/agent/hcpeer.py` (new), edits to `agent/join.py`, `agent/managed.py`,
  `cli.py`, `schemas.py`, `services/nodes.py`, `scripts/capacity/hcpeer_mac_install.sh`, `tests/test_hcpeer_agent.py`,
  `tests/test_hcpeer_mac_install.py`, `docs/research/hcpeer-macos.md`. Review found a MEDIUM about
  `HC_PEER_CACHE_MAX_GIB`; its fixup was running.

Resume: `Workflow({scriptPath: "C:\Users\operator\.claude\projects\C--\9e2c80d5-50ca-460d-8443-8db45359fcde\workflows\scripts\peer-ssd-tier-prep-wf_b1c28bb0-6fd.js", resumeFromRunId: "wf_b1c28bb0-6fd"})`
re-runs only the unfinished fixups. Then targeted tests, the full suite, review, and separate commits.

## Runtime state at pause

- Gateway on 127.0.0.1:8001 stopped; GPU idle; no llama-server, rpc-server, downloads or drivers running.
- `live-host` worktree detached at f14e58a (6d5016b is peer-side only). Start the gateway with
  `scripts/capacity/paper/gateway.ps1` (`HYPERCOMPUTE_HOME = iroh-session`, `CONFIG_DIR = live-host\configs`).
- Models on the G14 (verified): `iroh-session/models/gpt-oss-120b-mxfp4` (59.03 GiB),
  `iroh-session/models/qwen3-235b-a22b-instruct-2507-ud-q2_k_xl` (82.67 GiB, both shards sha256 OK, marker written).
- Macs at 6d5016b, agents stopped, caffeinate stopped, all at 100% battery. RPC caches kept:

| node | name | SSH (Tailscale) | cache | notes |
|---|---|---|---|---|
| 4c2991743c154be896d10a0b53ba4d2b | mac-azim | azimmemon@100.101.19.13 | 18 GB | Azim's personal Mac, 46 GiB free, cap 30 GiB |
| 6dad6cf75b2e4cf197b7bc7f4d0a67e4 | mac-a | operator@100.80.165.87 | 22 GB | 120B layers 4-7, 235B layers 7-13 and 22-31 (C1 seeded) |
| d4e65f83c4be4a41aad722e40983b08b | mac-b | azim@100.90.34.93 | 18 GB | 120B 8-11, 235B 14-21, partial 28-41 (5.2 of 12.8 GB, resumes) |

- Rejoin a Mac (fresh invite each time, never printed): `scripts/capacity/paper/rejoin.sh NAME USER@HOST 7.5GiB 0GiB
  --managed --max-disk 100GiB` (30GiB for mac-azim). Keep each Mac's cap below its free memory so placement is pinned.
- Run driver: `python -I scripts/capacity/paper/matrix.py PLAN.json INDEX.jsonl` (waits for the GPU to drain between
  runs), then `summarize.py` and `stats.py`. Paths inside them point at the G14's scratchpad and live-host; adjust if
  moved.

## Next, in order

1. Finish and commit the stopped workflow (above). Install hcpeer on mac-a and mac-b
   (`scripts/capacity/hcpeer_mac_install.sh`), gate on the `hcpeer mmap : on (v4` banner and a small-model token check.
2. Resume mac-b's seed (`hypercompute capacity seed --node d4e6 --model qwen3-235b-a22b-instruct-2507-ud-q2_k_xl
   --layers 28-41`), then run candidate C1 (`peer-ssd-tier-candidates.md` section 5) alternating with the 3-Mac Metal
   baseline. Falsified if decode stays at or below 1.40 tok/s with the G14 SSD reads near 68 MiB per token.
3. S0: find where the ~89 ms per Mac stage goes (bridge counters per message type, rpc-server timing on a Mac).
4. 235B long prompt (512 tokens) A vs D for prefill/TTFT; bridge stock vs fast A/B on 120B with 3 Macs.
5. Network: a private LAN (travel router or USB-C Ethernet + switch) should cut RTT to about 1 ms and lift host to Mac
   transfer well above 8.6 MiB/s; rerun the 120B 3-Mac config there.
