# Implementation status

Legend: YES = done with evidence; NOT RUN = needs hardware that was not reachable; PLANNED = later milestone,
intentionally not stubbed. "Contract-tested" = portable protocol tests (`uv run pytest`: 578 pass, no GPU).
"Local-runtime-tested" = exercised with real llama.cpp processes on the G14. "Multi-host-validated" requires two
physical machines. The first real pair (the G14 + a friend's M4 MacBook Air over campus Wi-Fi) ran three deployments
on 2026-10-21 UTC (the evening of Oct 20 local) on the MVP build, before the engine-tier merge (old local-first `--device` order). The rows it covers say
so; evidence is `results/pair/20261021-campus-wifi-g14-m4air.json`. Everything else in that column is NOT RUN.
Cross-OS runs used a WSL2 Ubuntu VM on the same G14 (separate kernel and network namespace, real TCP) and are labeled
as such.

Hardware that participated: ASUS ROG Zephyrus G14 GA403WR (Windows 11, Ryzen AI 9 HX 370, 32 GB RAM, NVIDIA RTX 5070
Ti Laptop 12 GiB sm_120, driver 591.84, Radeon 890M iGPU, Wi-Fi on a Public-profile /15 network), its WSL2 Ubuntu
24.04 VM, and for the pair runs an M4 MacBook Air 16 GB (macOS 26.6.2, 10-core GPU, Metal working set 12,124 MiB) on
the same campus Wi-Fi. RTX 5090 workstations, the Mac mini and the GX10 were not reachable.

Evidence files: `results/*.json` (per run, raw), `results/lifecycle-*.json` (deployment outcomes exported from the
host's records), `docs/RESULTS.md` (generated tables), `docs/screenshots/*.png` (dashboard flows).
Some archived captures were omitted for privacy; the recorded outcomes remain in the result files.

| Capability | Implemented | Contract-tested | Local-runtime-tested | Multi-host | Evidence | Blocker / next step |
| --- | --- | --- | --- | --- | --- | --- |
| Pinned engine install (variant choice, sha256, safe extract, active pointer, smoke test) | YES | YES | YES (Windows CUDA 13.4; Linux x64 CPU in WSL) | YES (macOS arm64 b11770 ran on the M4 Air and served RPC to the G14) | `engine install` output; tests; results/pair/*.json | time a clean-Mac install (TEST_PLAN Q-P5) |
| Native inventory (UMA counted once, AdapterRAM ignored, no serials) | YES | YES | YES (Windows, Linux/WSL) | YES (macOS: one 16 GiB unified pool, MTL0 working set 12,712,935,424 B) | `hypercompute inventory`; tests/fixtures/inventory_*_raw.json; results/pair/*.json | |
| Sessions, one-time expiring rate-limited invites, hashed codes and credentials | YES | YES | YES | YES (real join over campus Wi-Fi) | tests; results/pair/*.json | |
| Join: worker address must equal source IP, private or link-local only, node-id takeover refused | YES | YES | YES (loopback, WSL VM) | YES (Mac worker 10.19.252.207 accepted) | tests; results/pair/*.json | IP change mid-session not tested (TEST_PLAN FI-N5) |
| Engine identity + real RPC handshake probe before eligibility; 30 s re-probe of unreachable workers | YES | YES | YES | YES (b11770 / RPC 8.0.0 handshake over Wi-Fi) | node verification in results/*.json and results/pair/*.json | |
| Heartbeats, stale detection, deployment invalidation | YES | YES | YES (agent + worker hard-killed: failed after 16 s, llama-server stopped, leases released) | YES (`725198fe`: Mac lost mid-request, failed after 17 s, 2 leases released) | results/lifecycle-*.json (`7129a8d0`, `725198fe`) | |
| Leave / pause / resume / owner removal | YES | YES | YES (leave and pause mid-deployment fail it cleanly; resume re-verifies to eligible) | NOT RUN | results/lifecycle-*.json (`bc050115`, `77da9bc9`) | |
| Owned-process supervision: Windows Job Object kill-on-close, Linux parent-death signal, restart reconciliation | YES | YES (hard-kill child test on Windows) | YES (gateway hard-killed during a 3-machine deployment: llama-server gone within 1.5 s; restart reported "pid no longer running", released 3 leases) | n/a | results/lifecycle-*.json (`8a8576ac`); tests/test_supervisor.py | Linux PDEATHSIG path not executed (Windows host) |
| Single-instance locks (host, agent) | YES | YES | YES (second host and second agent refused) | n/a | session run log | |
| Planner (fit, local/fastest/all-selected/auto, UMA once, output-head aware, key/value head lengths) | YES | YES | YES (planned layers == llama.cpp observed layers in every split run, incl. 3 devices) | YES (MVP build: planned == observed on 3 G14 + M4 Air splits) | docs/RESULTS.md | |
| Split order: peers first, local GPU last holding the output head; interactive fill local-first, throughput proportional | YES | YES | YES (`--device RPC0,CUDA0`, CUDA0 holds the head, peer got the 1-layer minimum) | NOT RUN (Mac); the "before" number exists: on the old order the Mac held the head, and the 14B ran at 81-86 ms/token over campus Wi-Fi | results/*gw-0.5b-split-order.json; results/pair/*.json | measure the logits saving on Wi-Fi with the Mac (TEST_PLAN R1, R2) |
| Tier ladder gpu -> gpu_ram -> gpu_disk, overflow to peers vs host RAM by a decode-cost prior | YES | YES | YES (14B Q6_K: 4 CPU layers + 44 on CUDA0, both leases reserved, 27 tok/s; ngl sweep 45/43/40 = 27.5/22.8/17.9 tok/s confirms the pick) | NOT RUN (Mac overflow) | results/*gw-14b-auto.json | gpu_disk not run on hardware (needs free RAM below the CPU part) |
| Measured tuner + profiles (candidates, sequential trials, scoring, profile replay keyed by model/ctx/devices) | YES | YES | YES (14B: 4 candidates measured, RAM split won 27.2 tok/s vs peer 25.6, profile replayed with heuristic=false) | NOT RUN (Mac) | tune f8ae5fdc; `hypercompute tune` output | tune Mac vs RAM on test day |
| Engine selector and per-engine adapters (registry, optional modules, plan per engine) | YES | YES | YES (auto picked llama.cpp for 7B/14B; forced vLLM and AirLLM through the gateway) | n/a | docs/ENGINE_TIERS.md | |
| Per-engine inference slots (vLLM serves --max-num-seqs at once; benchmarks/warmup/tune take every slot) | YES | YES | YES (vLLM through the gateway: 8 users 546 tok/s, 16 users 862 tok/s, 96/96 ok) | n/a | docs/RESULTS.md | |
| Atomic leases; kept when a process could not be stopped | YES | YES | YES | n/a | tests | |
| GGUF reader for exact per-layer weights | YES | YES | YES (within 0.1 MiB of llama.cpp buffers, 0.5B and 7B) | n/a | `pytest -m hardware` | |
| llama.cpp adapter: launch, /health + warmup readiness, placement parse, per-device participation check | YES | YES | YES | YES (Mac participation verified: 2 and 8 layers, planned == observed) | `participation_detail` in results/*.json and results/pair/*.json | |
| Two-node and three-node RPC generation with verified nonzero peer work | YES | YES | YES (Windows CUDA + Windows CPU worker + Linux CPU worker in WSL) | YES, two-node (G14 CUDA + M4 Air Metal, campus Wi-Fi: 14B Q6_K with 2 Mac layers + head 11.7-12.3 tok/s; Mistral 24B with 8 Mac layers + head median 9.1 tok/s over 12 chats) | docs/RESULTS.md; results/pair/*.json; docs/screenshots/05-run-lifecycle.png | those runs used the old order (the Mac held the output head and sent the logits back every token); rerun peers-first (TEST_PLAN R1, R2) |
| Streaming proxy (fragmentation-safe SSE, usage/timings, disconnect closes upstream, model-name check) | YES | YES | YES | n/a | tests/test_sse.py, tests/test_deployments.py | |
| Run records with model/engine/host identity, failed and cancelled runs kept | YES | YES | YES | n/a | `~/.hypercompute/coordinator/records/runs.jsonl` | |
| Benchmarks (warmup, 3 reps, median/range, engine timings, `cache_prompt: false`) | YES | YES | YES | NOT RUN | docs/RESULTS.md | |
| Gateway hardening: owner loopback + Host + token, body-size limits, connection cap, anti-framing headers | YES | YES | YES (413 on a 300 KB join body; headers present) | n/a | tests/test_app_http.py | |
| Dashboard: session, peers, plan, run, chat, technical details, runs/benchmarks, activity | YES | YES (52 vitest) | YES (browser: successful 3-machine run + chat; failed deployment when a planned friend paused) | NOT RUN | docs/screenshots/*.png | |
| CLI `host/join/status/pause/resume/leave/doctor/engine/models/inventory` | YES | partial | YES | NOT RUN | session runs | |
| Worker scoping guidance (Windows rule, Linux ufw, macOS pf anchor), auto-stop after 5 min without host | YES | YES | YES (Windows text) | NOT RUN | `join` output | Hypercompute never applies firewall rules itself |
| Native build scripts (macOS/Linux/Windows) | YES | syntax-checked + script tests | YES (real CPU builds of b11770 on Windows MSVC with a user-level cmake, and in WSL Ubuntu with gcc 13.3; both report build 11770, commit 39da130) | NOT RUN (macOS) | scripts/build_llama_*; tests/test_build_scripts.py | CUDA build needs the CUDA toolkit (admin install) |
| Packaging step 1: ~92 KB wheel with bundled lock + manifest, `uv tool install` | YES | n/a | YES (Windows, isolated tool dir) | NOT RUN | RUNBOOK section 0 | still needs uv on the friend machine |
| Packaging step 2: single download app/agent | PLANNED (M5) | | | | | signed macOS/Windows binaries |
| vLLM engine (WSL2 install on uv-managed Python, Triton smoke, download-aware load budget, watchdog, orphan cleanup) | YES | YES | YES (0.32.0 on sm_120: 7B AWQ direct 1 user 78 tok/s, 16 users 1,048 tok/s; through the gateway ready in 51 s, graceful stop, nothing left in WSL) | n/a (one GPU) | docs/RESULTS.md | multi-node vLLM needs several NVIDIA Linux machines |
| AirLLM capacity mode (own venv, stdlib worker, disk tier plans, disconnect cancel) | YES | YES | YES (0.5B bf16: worker ready 62 s first start, 0.2 tok/s through the gateway, VRAM released on stop) | n/a | results/*airllm-gw-airllm-0.5b.json | 32B bf16 needs about 131 GB of disk |
| EXO/MLX adapter + Mac sidecar (`join --exo`), orphan cleanup on gateway start, endpoint pinned to the node | YES | YES (source-shaped fixtures from EXO bc412e2) | NOT RUN | NOT RUN | tests/test_exo*.py | needs Macs (2+ for a real cluster) |
| GX10 profile | PLANNED (M6) | | | | `linux-arm64-cuda13.4` variant listed, unvalidated | hardware |
| exo_paged distributed offload | PLANNED | | | | | |
| MLX ring pipeline between a Mac and a CUDA box | Research only | | | | docs/research/mlc_llm.md | undocumented upstream, open ring + Blackwell bugs |

## Review

MVP: an independent multi-lens review (lifecycle, security, agent/process, planner/engine, spec honesty) raised 34
findings; two skeptics per finding confirmed 31. All 31 plus one split-vote finding (`/v1` model mismatch) were
fixed with regression tests; the remaining two were refuted (one already fixed by the instance lock).

Engine tiers: six reviewers (lifecycle, planner, engines, security, frontend/backend contract, installers) raised 36
findings; a skeptic per finding confirmed 33 (4 high). All 33 were fixed with tests that fail before the fix, and a
second round of independent verifiers checked every fix and found 10 smaller follow-ups, also fixed and verified.
Hardware runs then found two bugs no test had caught: vLLM's Triton JIT needs Python headers that Ubuntu's Python
lacks (installer now uses uv's managed Python and test-compiles a kernel), and vLLM 0.32 logs a normal INFO line the
adapter mistook for its out-of-memory error. Both are fixed with regression tests.

## Known gaps

* One two-physical-machine session so far (three deployments, campus Wi-Fi, MVP build): the Mac joined, verified and
  served layers, but the new split order, the tier ladder, the tuner and EXO have not run with a Mac, and there is no
  hotspot, router or Ethernet measurement. The stress and break plan for the next session is
  [docs/TEST_PLAN.md](TEST_PLAN.md).
* Capacity claim not established: on the G14 the 7B fits the local GPU, so split runs show participation, not
  necessity. The 32B candidate is listed but not downloaded.
* WAN-disconnected demo step not executed.
* RPC tensor transfer progress is not observable from llama.cpp; the UI says so instead of inventing a bar.
* Benchmarks marked with prompt cache ON in RESULTS.md predate the `cache_prompt: false` fix.
* EXO has never run: it needs Macs, and a real EXO cluster needs at least two.
* The overflow choice between a friend's GPU and the host's RAM is a cost-model estimate until `hypercompute tune`
  measures it with the real Mac over Wi-Fi.
* The G14's measured host-RAM decode cost (about 18 ms per GB) is higher than the planner's prior (12.5 ms per GB),
  which favours a Metal peer. The three real Mac splits (one session) ran on the old local-first order, where the Mac
  held the output head and sent about 0.6 MB of logits back over campus Wi-Fi every token; its loads ran at 12-20
  MiB/s end to end. Separately, the 14B with 2 layers + head on the Mac ran at 11.7-12.3 tok/s, against 26.7 tok/s
  with 4 layers in host RAM (G14 power state not recorded); how much of that gap is the logits is not measured.
  The per-token cost with the head on the G14 is still unmeasured (TEST_PLAN R1, R2).
