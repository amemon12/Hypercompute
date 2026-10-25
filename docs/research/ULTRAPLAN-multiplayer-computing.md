# ULTRAPLAN: multiplayer computing (revision 2)

Revision 2, 2026-10-22 about 21:00 UTC, against `codex/iroh-transport` at a8e1364. It replaces revision 1, which was
written at d253f49. Inputs:
- revision 1 and a critique of it (answered in "Critique response" at the end);
- `CHECKPOINT-2026-10-22.md`, `big-model-runbook.md`, `peer-ssd-tier-design.md`, `mac-host-runbook.md`,
  `OVERNIGHT-SETUP.md`;
- the two gpt-oss run.json files and the bridge benches `results/capacity/20261022T203138Z-bridge-bench-before/` and
  `20261022T205040Z-bridge-bench-after/`;
- the code at the lines cited.

Every critique claim that cites code or a run.json was re-checked read-only before use. Citations carried over from
revision 1 that the critique did not touch were not all re-checked. Labels: MEASURED (a result file exists), ESTIMATE
(computed or fitted), INFERRED (reasoned), UNVERIFIED (claimed by a source, not checked).

**The shared worktree changed three times while this was written.** Between the first and last `git status` (about
20:30 to 20:57 UTC):
- the Mac-host edits were committed as a8e1364;
- a managed-peer and seeding change set appeared;
- a bridge "after" bench was recorded.

At the last read the uncommitted work was:
- modified: `agent/join.py`, `app.py`, `capacity_cli.py`, `cli.py`, `schemas.py`, `services/nodes.py`,
  `transport/client.py`, `transport/host.py`, `transport/iroh.py`;
- untracked: `agent/managed.py`, `agent/seed_cache.py`, `services/peer_seed.py`, `native/hcpeer/`,
  `scripts/build_hcpeer_mac.py`, `scripts/capacity/bridge_bench.py`, `tests/test_bridge_perf.py`,
  `tests/test_managed_peers.py`, `tests/test_peer_seed.py`.

Line numbers are for HEAD unless the file is in that list.

## Task

Make Hypercompute "multiplayer computing". Any laptop joins a friend's session with one click. The group then runs
models much larger than any single member can run usefully. Plan tonight's demo first. Then plan the peer SSD tier,
Apple-only and mixed clusters, the largest feasible model, and one-click join.

### Context

**What changed since revision 1**
- **Mac-host support is committed** (a8e1364, with `tests/test_mac_host.py`). It has never run on a Mac
  (mac-host-runbook.md:9-10).
- **Disk is no longer the G14's limit.** C: had 108.69 GiB free at 20:50 UTC (`Get-PSDrive`), not 13 GiB. A Qwen3-235B-A22B
  UD-Q2_K_XL download is writing into `iroh-session\models`: shard 1 of 2 was at 49,991,773,312 B with a `.ranges.json`
  at 13:49 local. The whole model is 88.8 GB.
- **Another session is building the peer SSD tier.** It has two parts:
  - a server-only mmap patch to rpc-server (`native/hcpeer/`, `HC_PEER_MMAP=1`, hcpeer.h:1-8);
  - Hugging Face range seeding behind `join --managed --max-disk` (`agent/managed.py:1-8`).

  Neither has run on a Mac.
- **The bridge is measured.** It was the upload bottleneck, and a fix measures 20x faster on loopback (below).
- **Placement is not stable from run to run** (below).

**Current state**
- **The core works.** The host runs llama-server, and each peer runs ggml rpc-server behind Iroh QUIC. It is the only
  engine in the repo that splits one model across Windows CUDA and Mac Metal.
- **gpt-oss-120b runs through the gateway.** MXFP4, 59.03 GiB, 4 of 128 experts active per token. MEASURED, 2 reps each:

  | | G14 alone | G14 + one M4 Air (layers 0-2) |
  | --- | --- | --- |
  | Decode | 4.05 tok/s (3.77-4.33) | 4.52 tok/s (4.48-4.56) |
  | First token, 126-token prompt | 30.8 s | 21.2 s |
  | SSD reads per token (system wide) | 232 MB | 121 MB |
  | Host CPU experts (engine log) | 50,741 MiB | 45,888 MiB |
  | Available RAM at run start | 19.70 GiB | 18.11 GiB |
  | Load to ready | 41 s | 62 s cached; first upload 521 s for 4,939 MiB (9.5 MiB/s) |

  Sources: summary:13-19, and run.json `measured.host.samples[0]` / `measured.gateway_before.pools`. The G14-alone run
  went second, so its file cache was warmer (summary:31).
- **The bridge capped uploads.** In the loopback bench at d253f49:
  - Iroh bulk was 10.1 MiB/s, against 1,129 MiB/s through a plain TCP proxy.
  - An 11,520-byte round trip (one gpt-oss activation each way, 2,880 x 4 B) took 3.25 ms median (p99 7.06 ms),
    against 0.10 ms through the proxy.
  - 19.3 s of the 23.4 s profile was the binding's per-byte `write` (`profile.txt`, `iroh_ffi.py:211`).

  So the 9.5 MiB/s first upload was bridge-bound, and real hotspot throughput is unknown (INFERRED).
- **The bridge fix measures well on loopback.** It is uncommitted in `transport/iroh.py`: a memmove copy with a
  round-trip self check, plus inline wakeups. The "after" bench at a8e1364 plus that diff, against the stock binding in
  the same run:
  - bulk 200.9 MiB/s against 10.0 MiB/s;
  - an 11,520-byte round trip 0.59 ms median (p99 2.86 ms) against 4.55 ms (p99 16.1 ms).

  It has not run between two machines.
- **The per-peer cost is not identified.** Two fits of the same two runs disagree:
  - Revision 1's fit gives about 15 ms per peer.
  - The design doc's fit gives 60 ms per peer stage (peer-ssd-tier-design.md:472).

  Two runs cannot pin three or more coefficients. The stock bridge explains at most about 3-4.5 ms per hop per token.
  Network RTT was 7-13 ms before runs and 65 ms after the pair run (run.json
  `measured.gateway_after.network.peers[0].path.rtt_ms`). The rest is unexplained.
- **The MoE estimate is about 5x too pessimistic.** It predicted 1,081 ms per token; measured was 221 ms. It assumes
  no expert reuse (moe_tier.py:128, `miss = 1 - cache / weight`) and charges a flat 4 ms per hop (planner.py:49).
- **Placement follows live free memory, so it drifts.** The chain:
  - Peers are ordered by ascending budget (planner.py:420; the MoE path calls this at planner.py:549).
  - Each peer then takes whole layers from layer 0 in that order (moe_tier.py:29-41).
  - A peer's budget is `min(share, free - reserve)` (deployments.py:192-193).
  - Free memory is re-measured on every heartbeat (agent/inventory.py:227-231).

  In the pair run the Air had 10.61e9 B free and a 4 GiB reserve, so its budget was 6.32e9 B. Free memory set it, not
  the 10 GiB contribution (run.json `gateway_before.nodes[1]`, `predicted.plan.assignments[1]`). The plan does not
  record the basis: `budget_basis` is null. When a peer's free memory moves, or a peer that is not last drops out, the
  later ranges shift. The content-addressed cache then misses for every layer that moved.
- **RAM.** Available RAM fell to 4.67 MB during the pair run with the guard off (`host.whole_run.available_min` =
  4,669,440). Revision 1's "about 10 GiB free" was `FreePhysicalMemory` (big-model-runbook.md:22), a different and
  smaller number. The page-cache model needs available RAM.
- **One heartbeat gap ends a run.** A node is stale after 15 s without a heartbeat (config.py:17; nodes.py:316-320).
  Losing it fails every deployment that uses it (deployments.py:278, :675-680).
  - The 15 s is a constant. config.py:1 says everything is overridable by environment variables, but only paths read
    the environment (config.py:33-35).
  - The Mac had gaps of 206 s and about 22 min, plus short flaps. The cause was not established
    (big-model-runbook.md:29-31).
- **Joining is all CLI.** Invites are one-use, expire after 15 min, and their failed-attempt cap is not enforced
  (config.py:19-20; TEST_PLAN.md:423). Replan is off by default (coordinator.py:160) and has only unit tests.
- **A clean second worktree is cheap.** The current `.venv` is 76 MB (`du`).

**Target state.** A friend opens a link and joins within minutes. The planner places weights across every device's
accelerator memory, RAM and SSD, and each node pages its own shard. The group runs a model that no member can run
usefully alone. A run.json backs every public number.

**Constraints**
- **The host holds the whole GGUF** (Mode A, peer-ssd-tier-design.md section 7). It hashes every tensor it places on a
  peer.
- **Weights upload once per tensor, at offset 0, at full size.** The design doc verifies this in source (F9,
  peer-ssd-tier-design.md:88); it was not re-checked here.
  - Only WEIGHTS tensors over 10 MiB use the hash cache (vendored ggml-rpc.cpp:118, :844-846). Smaller tensors are
    re-sent on every load.
  - This settles revision 1's UNVERIFIED pre-seeding premise, subject to the design doc's check.
- **Peers stay RAM-only** until hcpeer passes its gates.
- **Each peer adds at least one wait per token.** The b11770 client sends set-tensor and graph-compute asynchronously
  (vendored ggml-rpc.cpp:1173-1184, :1252-1266), but the host still needs each peer's output before the next stage.
  The exact number of synchronous commands per token is unmeasured. `GGML_RPC_DEBUG=2` logs every command (log.h:11-15).
- **3+ RPC endpoints may deadlock.** llama.cpp #28487 was closed as stale on 2026-10-22, and b11770 is untested
  (https://github.com/ggml-org/llama.cpp/issues/28487).
- **Paging works for MoE, not dense.** gpt-oss reads 121-232 MB per token; a dense 70B Q4 would read about 42.5 GB.
  The flip side: an MoE model degrades gracefully on one machine, so a dense model is the honest test of "bigger than
  any one laptop" (INFERRED).
- **RPC is unauthenticated.** CVE-2026-34159 (unauthenticated memory read and write in the RPC backend) is fixed in
  b8492 (https://www.sentinelone.com/vulnerability-database/cve-2026-34159/,
  https://www.cve.org/CVERecord?id=CVE-2026-34159). b11770 is a later build, so the fix is probably in (INFERRED). A
  read of the vendored `deserialize_tensor` did not settle it.
- **The G14 alone already runs the 120B at 4.05 tok/s.**

### Key decisions

| # | Chosen | Alternatives | Why |
| --- | --- | --- | --- |
| D1 | llama.cpp RPC b11770, stock binaries | EXO/MLX, AirLLM, prima.cpp | The only measured cross-OS 117B path. EXO is Mac-only with no SSD tier, and its multicast discovery fails under client isolation (exo.md:259-261). AirLLM ran at 0.2 tok/s on one host. prima.cpp has no Windows support |
| D2 | MoE for paging: gpt-oss-120b tonight, Qwen3-235B-A22B next (its download is under way). A dense model only for the pooled-memory proof (D13) | Dense 70B/72B paged | Dense paging reads the whole model per token |
| D3 | G14 hosts the demo tonight. The first Mac host is a 200 GB Air (peer-ssd-tier-design.md section 11). Azim hosts only with at least 70 GiB free (mac-host-runbook.md:32-40) | Azim hosting by default | A Mac host has never run, and Azim has about 60 GB |
| D4 | Phone hotspot, or a travel router without client isolation (OVERNIGHT-SETUP.md:16). Iroh only, and every peer `direct` before loading. G14 wired to the router when one exists | LAN mode; relay | LAN mode puts RPC on the LAN address (join.py:176 binds loopback only behind Iroh). The relay needs about 57 min per 5 GiB (ESTIMATE at 1.58e6 B/s). A wired host halves the radio crossings per hop (INFERRED) |
| D5 | 2 peers until 3 pass the gate, with a clock cutoff (Phase 0b.7) | 4 devices at once | #28487 |
| D6 | On stage, a join is `--plan-only` plus a reload from a warm cache, only if rehearsed under 90 s | Live `HYPERCOMPUTE_REPLAN=1` | Replan is unit-tested only and reloads anyway |
| D7 | Peer SSD tier = the design doc's route: the hcpeer server-only mmap patch plus HF range seeding (`--managed`) | mesh-llm stage runtime; a macOS interposer | Already in progress, keeps the wire protocol and the host binaries stock (peer-ssd-tier-design.md section 0). mesh-llm stays the fallback |
| D8 | "Pooled page cache" = each node caches its own shard in its own RAM | Host pulls expert pages from peers | A 9 ms round trip is about 100x a local NVMe read (INFERRED) |
| D9 | Stage claims come only from tonight's run.json files, and must beat the equal-headroom control (Phase 0b.6). No speed claim unless at least 1.2x with non-overlapping ranges | Quoting the 2-rep pair | The pair is 1.12x with n=2, under the plan's own bar |
| D10 | Long term, peers own their stages under the existing control plane | RPC forever | The RPC limits cannot be planned away |
| D11 | Stable placement tonight by pinned contribution caps, not code (Phase 0b.4). Matrix subsets are prefixes of the pinned order, chosen with `run_big --node` (run_big.py:757) | Join order (not used by the code); a code change tonight | A cap below free minus reserve makes the basis "contribution limit", which does not move with Safari. Distinct caps fix the order |
| D12 | The bridge fix goes into the demo only as one reviewed commit of the binding-speedup hunks, behind gates (Phase 0a.2) | Freeze without it; take the whole iroh.py diff | It cuts warm-up from bridge-bound to network-bound. For decode it saves an ESTIMATED 2-4 ms per peer per token (1-5% at 3 peers). The rest of the diff (managed peers) changes the protocol and stays out |
| D13 | "Bigger" is proven Apple-only in Phase 1, not after Phase 2, with a dense model held in pooled memory plus the runbook's 120B Mac-host run | Wait for the SSD tier | The vision needs this result, and it does not depend on engine work. The 235B on the G14 is a measurement probe, not a "bigger" claim, because the G14 alone may run it slowly |

### Risk Assessment

**HIGH**
- **Placement drift turns a warm cache into a 9-10 min upload per 5 GiB mid-demo** (at today's bridge).
  - Tonight: D11, with `--plan-only` checked right before every load. If any peer's basis or range differs from
    the cached one, do not reload on stage.
  - Phase 1: stable order and a cache-affinity tie-break.
- **A heartbeat gap ends the run, and the cause is unknown.** Three peers roughly triple the exposure, if gaps are
  independent.
  - Tonight: AC power and `caffeinate -dims`. These are cheap but not a known fix.
  - Tee each Mac's `join` output to a file. The agent prints a timestamped `host unreachable` on failures 1, 5 and
    every 20th (join.py, about :378-386 when read; the file is being edited).
  - After any gap, read `pmset -g log` for sleep and wake entries in that window. Output format UNVERIFIED.
  - Reading the evidence: unreachable lines mean the network or Iroh. A sleep entry with no lines means the Mac slept.
    Neither means the agent stalled.
  - Optional: Phase 0a.3.
- **Three peers may deadlock (#28487).** Follow D5.
- **The demo host could run untested code.** The shared worktree changed three times during this review. Run the demo
  gateway from a clean worktree at a tag. No stash or checkout in the shared worktree.
- **The headline could lose to "just close Chrome".** The pair moved 4.74 GiB of experts off the host and halved SSD
  reads. Page-cache headroom is the likely mechanism, and the G14 alone has not been run with equal headroom. Phase
  0b.6 adds that control.
- **Tracks compete for the Airs tonight.** The demo evidence (this plan), the design doc's S0-S2, the Azim-hosted
  runbook and the overnight 235B all want the same three Airs. Phase 0 sets one order (Open Question 4).
- **The 235B download competes with measurement.** It writes to the same SSD the G14 pages experts from. Finish or
  pause it before any measured run (INFERRED interference).
- **Friends' laptops are exposed.** RPC gives the host broad access to the peer. Iroh only (loopback bind,
  join.py:176). State the trust model.

**MEDIUM**
- **Metal runs out of memory on a peer** (cap 11.84 GiB). Quit apps and plan right before the load. Lower the cap and
  re-plan.
- **The hotspot degrades under load or caps clients** (about 5 on iPhones, UNVERIFIED). RTT went from 7 to 65 ms
  across the pair run. Count devices first, record each peer's RTT during the warm-up load, and carry a second hotspot.
  Never restart the host mid-demo (stale tickets).
- **The stage laptop is not the one that was measured.** Slides, screen sharing and a browser shrink the page cache.
  Measure a stage-condition config (Phase 0b.6), and drive slides from another device.
- **Disk fills.** Keep the drive guard on (5 GiB floor). macOS swap shares the model volume.
- **Planner predictions are wrong.** Quote none before Phase 1.
- **Peer code skew** (now low):
  - An 890f798 agent worked with this host in the pair run (MEASURED; big-model-runbook.md section 2).
  - a8e1364 changed no agent, transport or schema file (`git show --stat a8e1364`).
  - New Airs install the tag from a git bundle (mac-host-runbook.md:163-169).
- **Phase 2 slips.** hcpeer needs a Mac build, re-signing, and bit-identical tokens.

**LOW**
- **SSD wear.** Reads barely wear NAND; avoid swap.
- **Model license.** Confirm Apache-2.0 on the gpt-oss card.
- **Invite attempts are uncapped** (config.py:20).

**Irreversible, never without explicit approval**
- Deleting the 120B or 235B files.
- `leave --purge-cache` (cli.py:564), or any seeding cap that evicts the peers' gpt-oss cache files.
- A DB schema change, or dev code run against the demo home, without a fresh backup of `iroh-session\coordinator`.
- Force-pushing.

### Execution Phases

#### Phase 0a: pre-freeze changes (tonight, optional, 45 min cap)
Each change is its own commit on a demo branch from the base tag, reviewed, with the full non-hardware suite green
(`uv run --no-sync pytest -q -p no:cacheprovider`, CHECKPOINT:17). A change that has not passed its gate by T+0:45
stays out.

1. **Base: a8e1364 or d253f49.**
   - a8e1364 touched only Mac-host paths for a Windows host (deployments.py `_unified_budget`, Metal buffer names,
     macOS guards).
   - Use it if the first matrix run (config A) reproduces 4.05 tok/s within 15% and its load check passes.
   - Otherwise use d253f49, the code that was measured.
2. **Bridge speedups**, the binding-speedup hunks of iroh.py only (D12). Gates:
   - `tests/test_bridge_perf.py` and the Iroh transport tests pass.
   - `binding_speedups()` reports both active.
   - The after bench above stands as the loopback gate.
   - One 0.5B G14 + Air smoke over the hotspot passes.

   The host alone gets most of the upload gain, because the slow copy is in the sender's `write_all` (`tcp_to_quic`
   in `profile.txt`), and the host sends the uploads (INFERRED). Peers can keep their code.
3. **Stale window, 15 s to 60 s** (config.py:17). This is a code change, because there is no environment override.
   - It helps only short flaps, not the 206 s or 22 min gaps.
   - A truly dead peer already breaks the RPC stream, so a slower stale mark costs little (INFERRED).
   - Gate: suite green.

Not tonight: a code change to peer order. D11 handles it without code.

#### Phase 0b: tonight's demo-safe path (S)
1. **Freeze.**
   - Tag the base, plus any Phase 0a commits (proposed `demo-20261022`).
   - Create a clean `git worktree` with its own `uv sync`. Run the G14 gateway from it with the same
     `HYPERCOMPUTE_HOME` (`iroh-session`).
   - Record `git log -1` and the engine build.
   - Azim's agent may stay. New Airs install the tag from a bundle.
   - Create each invite right before its join (one-use, 15 min).
2. **Quiet the G14.**
   - The 235B download is finished or paused. Nothing else heavy runs on C:.
   - The same app set as on stage.
3. **Preflight.**
   - AC power, and `caffeinate -dims` on the Macs.
   - Run `doctor` and count hotspot clients.
   - Tee each Mac's `join` output to a file.
   - `status` shows every peer `eligible`, `direct`, with a heartbeat under 15 s.
4. **Pin placement (D11).**
   - Close apps on each Air, then rejoin it with a distinct `--max-memory`. Give Azim the smallest cap so he comes
     first and keeps layers 0-2, the range his cache already holds.
   - Example: 5.5, 5.6 and 5.7 GiB, if `--plan-only` shows 3 layers each and basis "contribution limit" (ESTIMATE).
   - Each Air needs free memory minus reserve at least 0.5 GiB above its cap. Azim had 9.89 GiB free with a 4 GiB
     reserve, so he must close more apps or accept a smaller reserve (Open Question 7).
   - Then run `--plan-only` for the B, C and D subsets. B's and C's ranges must be prefixes of D's.
5. **Warm caches.** One D load.
   - ESTIMATE: 20-25 min for two new Airs (about 10 GB at 9.5 MiB/s). Network-bound and faster if 0a.2 landed.
   - Read each peer's RTT from `status` during this load.
6. **Matrix.**
   - Order A B C D D C B A, each with `--warmup 1 --reps 2`, so 4 reps per config. Timeouts
     `--load-timeout 900 --chat-timeout 600`.
     - A = G14 alone (`--strategy local_only`).
     - B, C and D = 1, 2 and 3 Airs (`--node` subsets).
   - Two controls:
     - **A+**: G14 alone with at least 23 GiB available at start. That is the pair's 18.11 GiB start plus the 4.74 GiB
       of experts it moved. Close at least 3.2 GiB more than in the measured alone run.
     - **S**: the exact stage condition.
   - For D, about 14 GiB of experts move (ESTIMATE). Equal headroom would need about 32 GiB available, more than the
     G14's 31.12 GiB, so closing apps cannot reproduce D (ESTIMATE). Measured, this becomes the strongest honest
     claim.
7. **Gate 3 peers.** Five clean D runs: the two in the matrix plus 3 extra loads with one chat each.
   - A hang is a cached load over 3x its median, or no token within 120 s of ready.
   - **Cutoffs:**
     - No clean D load by T+1:45: the stage uses 2 Airs.
     - The gate not passed by T+2:50: the stage uses 2 Airs.
8. **Dress rehearsal** on the venue network, with the hotspot in the room. Run the stage sequence once from a cold
   terminal to the final table, and record RTT.

**Time budget** (ESTIMATE; one `run_big` invocation took 4.3 and 5.3 min in the measured runs, from run.json
`started_at` and `finished_at`):

| Step | Minutes | Ends at |
| --- | --- | --- |
| 0a and freeze, in parallel with Mac preflight | 45 | 0:45 |
| Pin placement and `--plan-only` | 15 | 1:00 |
| Warm caches (one D load) | 25-40 | 1:40 |
| Matrix, 8 invocations of about 5 min | 40 | 2:20 |
| Controls A+ and S | 12 | 2:32 |
| Gate, 3 extra loads | 10 | 2:42 |
| Slack | 18 | 3:00 |

The rehearsal is separate, at the venue.

- **Risk:** a peer drops mid-warm-up and its range moves. The cost is a new upload; D11 limits it to that peer.
- **Checkpoint:** a run.json for A, A+, B, C, D and S, with:
  - decode, prefill and first token;
  - MiB per device;
  - SSD MB per token, labelled system-wide;
  - available RAM at start;
  - each peer's free memory, budget and basis at plan time;
  - RTT before and after;
  - the run order, the tag and the engine build.
- **Rollback:** the largest clean device count, then the host-only deployment, then the recorded video.

#### Tonight after the freeze (Blocks 2 and 3)
These start only after Phase 0b's checkpoint is recorded. The demo tag stays untouched.
- **Dev code and the demo home.** Before dev code touches `iroh-session`, back up its `coordinator` directory
  (big-model-runbook.md section 6), and restore the backup before the demo.
- **Peer caches.** Overnight jobs must not delete or evict the peers' gpt-oss cache files. List each cache directory
  (names and sizes) before and after.
- **Block 2 (attended).**
  - Design doc S0 on the G14 and one Air: the per-hop breakdown with `GGML_RPC_DEBUG=2`, 0, 1 and 2 Mac layers on the
    0.5B, with and without the bridge speedups. This answers the 15 versus 60 ms question.
  - S1 (hcpeer build and 7B gate) on a 200 GB Air.
  - A Mac-host 0.5B smoke on the other 200 GB Air, per mac-host-runbook.md.
- **Block 3 (overnight, over SSH, per OVERNIGHT-SETUP.md).**
  - S2: seed one Air and run gpt-oss with it file-backed.
  - The 235B G14-host probe: alone, then with 2 Airs memory-only, 1 warmup and 2 reps, tight timeouts. The catalog
    entry is not in `configs/models.example.yaml`. The active config directory (big-model-runbook.md:16) must define
    it (UNVERIFIED).
  - The single-Air 120B baseline (mac-host-runbook.md section 4.3), if a 200 GB Air has the 63.4 GB file.
  - None of this is a stage claim until reviewed.

#### Phase 1: Foundation and the "bigger" proof (M, about 1 week)
1. **Land the in-flight work.**
   - Review, test and commit the transport, managed-peer and seeding changes as separate commits, with
     `tests/test_managed_peers.py`, `tests/test_peer_seed.py` and `tests/test_bridge_perf.py`.
   - Tag the result.
2. **The Apple-only "bigger" proof** (D13, moved up from Phase 3).
   - **Single-Air baselines first, each a speed or a refusal:**
     - the 120B host-only run (runbook 4.3);
     - a dense 32B Q4_K_M (about 20 GB, UNVERIFIED) on one Air.
   - **Then 3 Airs, Mac host, memory-only:**
     - the 120B: ESTIMATE 0.9-1.0 tok/s (peer-ssd-tier-design.md:54-56);
     - the 32B in pooled memory: ESTIMATE 3-4 tok/s, on the unmeasured Metal prior (planner.py:48).
   - Scored against the "Bigger" win below.
3. **Stable placement in code.**
   - Sort peers by (budget, node id).
   - Tie-break toward plans that reuse cached or seeded shards (peer-ssd-tier-design.md section 6.5).
   - Write the budget basis into run.json.
4. **Heartbeats.**
   - A run-time grace window as real configuration.
   - The agent's send time and a sequence number in every heartbeat, and the host logs each gap.
   - Replan without the lost peer (replan.py:416).
5. **Per-hop breakdown.** Bridge counters (frames, bytes, time per direction), plus RPC command counts per token per
   peer. Store `hop_ms` per peer with provenance.
6. **Estimator.** The design doc's miss model (section 6.3) with the measured `hop_ms`. Calibrate on every Phase 0b
   config (6 or more), never on 2.
7. **Fold MoE into the capacity optimizer** (capacity/service.py:15-16, 363-366). Stop ranking with
   `planner.decode_ms`.
8. **Fix the Windows available-RAM guard** (hostmon.py) to discount file-backed pages, so `--min-available-gib 0` is
   no longer needed.
9. **Join and doctor.**
   - `join` prints the admissible budget and its basis.
   - `doctor` checks Iroh direct versus relay.
   - Add Metal calibration and a macOS storage probe.
10. **Cleanup.**
    - Split deployments.py and bridge.py.
    - Update the frontend `Tier` type (types.ts:13) and the stale docs.
    - Enforce `INVITE_MAX_FAILED_ATTEMPTS`.
    - Refuse LAN mode by default.

- **Risk:** the estimator overfits, and the Mac host fails.
- **Checkpoint:**
  - every measured config, A+ and S included, predicted within ±25%;
  - the suite is green;
  - a Mac-host run.json exists;
  - the Apple-only result is recorded as a pass or a fail.
- **Rollback:** revert the estimator commit. The selector switches only on strictly faster predictions
  (selector.py:142-159).

#### Phase 2: Peer SSD tier and pooled page cache (L, 2-4 weeks)
This follows peer-ssd-tier-design.md sections 10 and 14, stages S1-S4.

1. **hcpeer** (`native/hcpeer/`, `scripts/build_hcpeer_mac.py`). Gates:
   - bit-identical tokens against stock;
   - mapped and copied tensor counts as expected;
   - the failure cases;
   - an unmap on disconnect.
2. **Seeding** (`agent/managed.py`, `join --managed --max-disk`). Gates are the design doc's S2.

   Alternative kept: copy cache files that the host hashed locally over USB or AirDrop. It needs a native FNV-1a,
   which the design doc assumes runs at about 1 GB/s (A11).
3. **Inventory, pools and contribution (S3), then the planner and optimizer (S4).**
4. **Settle one name.** The build script says hcpeer and `HC_PEER_MMAP`; the design doc says peerstream and
   `HC_PEERSTREAM`. Pick one before the planner refers to it.

- **Risk:** Metal buffers stay wired (design doc F3), so only peer-CPU experts page. A modified macOS binary needs
  re-signing.
- **Checkpoint:**
  - one Air holds at least 2x its lent memory in layers, with swap growth under 2 GiB;
  - a pre-seeded join is ready in 10 min or less;
  - a 3-Air SSD-tier run is at least as fast as memory-only.
- **Rollback:**
  - feature flags;
  - a hash mismatch falls back to a host upload;
  - the stock rpc-server stays the default;
  - `build_hcpeer_mac.py rollback`.

#### Phase 3: Integration and testing (L, 1-2 weeks after Phase 2)
1. **Mixed.** G14 host + 3 file-backed Airs (design doc S5a). Then a Mac host with the G14 as a stock CUDA peer (S5b).
2. **Largest model.** Qwen3-235B-A22B UD-Q2_K_XL, 88.8 GB
   (https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF).
   - The G14-host probe from Block 3 comes first. Then the 200 GB Air host with the SSD tier.
   - ESTIMATE: 0.3-1 tok/s memory-only, 1-2 tok/s with the SSD tier.
   - Stretch: DeepSeek-V3.1 UD-TQ1_0, 170.5 GB, 0.1-0.5 tok/s.
3. **The 3-peer gate.** 10/10 clean runs (TEST_PLAN.md:422).

- **Risk:** a 3-peer deadlock, Mac memory pressure, and an 89 GB download per host.
- **Checkpoint:**
  - at least 5 interleaved reps per config;
  - 10/10 clean 3-peer runs;
  - a 235B first token with tok/s recorded.
- **Rollback:** 2 peers on the G14 host with gpt-oss-120b.

#### Phase 4: One-click UX and demo polish (M-L, about 2 weeks)
1. **Installer.** A signed per-OS app opens a `hypercompute://join?ticket=...` link (proposed). It bundles uv and the
   pinned engine, and runs `join --cache`. macOS notarization needs an Apple Developer account.
2. **A reusable room invite** with an attempt cap.
3. **Dashboard.** Path and RTT, bytes per device, live tok/s and SSD MB per token, budget basis, and join and replan
   events. The frontend build goes into git.
4. **Live join.** Replan on by default, with a cached or pre-seeded reload. A lost peer degrades the run instead of
   failing it.
5. **Friend-side guard** for swap and battery, plus the existing Stop page (agent/control.py).

- **Risk:** install and notarization friction on a clean Mac.
- **Checkpoint:**
  - a clean Mac goes from install to joined in 5 min or less, with at most 2 clicks;
  - a live join replans and is ready again in 90 s or less.
- **Rollback:** the CLI path stays supported.

### Measurement plan
- **Per config:**
  - decode, prefill and first token from engine timings;
  - load to ready, cold and cached apart;
  - MiB per device from the engine load log;
  - SSD MB per token, system-wide tonight and per process from Phase 1 (today's figure is system wide, summary:19);
  - available RAM at start;
  - commit peak on Windows, swap growth on Macs;
  - per peer: free memory, budget and basis at plan time, path mode, and RTT before, during the load, and after;
  - power, thermal state where a probe exists, the tag and the engine build.
- **Per hop (from Block 2):** bridge time, network RTT, peer compute, and synchronous RPC commands per token.
- **Reps:** 1 warmup, then at least 4 tonight (2 interleaved invocations of 2) and at least 5 from Phase 1. Order
  A B C D D C B A, so cache warmth and heat balance out.
- **Controls:**
  - A+ (equal page-cache headroom);
  - S (stage conditions);
  - nothing else writing to the host SSD.
- **Warm-cache bias:** report steady state after a shared warmup, and each config's first-ever load separately. No
  third-party standby-list purges on Windows. `sudo purge` on macOS is the user's call. Always record the run order.
- **Wins (proposed):**
  - **Faster:** median decode at least 1.2x over the better of A and A+, with non-overlapping ranges. The 2-rep pair
    (1.12x) does not meet this.
  - **Bigger:** the group runs at 1 tok/s or more, first token in 60 s or less, on a model every member alone refuses
    or runs below 0.3 tok/s.
  - **Load:** a cached rejoin in 90 s or less.
  - **Planner:** within ±25%.

### Demo script
1. **Beforehand.**
   - Everything is warm and pinned, and on AC.
   - Every peer is `direct`, and `--plan-only` matches the cached ranges.
   - The stage deployment (D, or 2 Airs) is loaded and warmed.
   - The fallback deployment and the video are ready.
2. **Open with what is true.** "This model has 117B parameters and is 59 GiB. This laptop runs it alone by streaming
   experts from its SSD: [A's tok/s and SSD MB per token from tonight]. Each friend's laptop takes whole layers, and
   their memory becomes cache, so the SSD does less work."
3. **A friend joins.**
   - `invite`, then the friend runs `join` (pre-installed).
   - `status` shows a direct path and its RTT.
   - `--plan-only` shows the pinned layer ranges.
4. **Reload and compare.** Reload from warm caches only if rehearsed under 90 s. Then run the same prompt. Show A, A+,
   B, C and D from run.json.
5. **Pick the claim from the numbers.**
   - If "Faster" holds, say "faster as friends join".
   - If not, say what did improve (first token, SSD reads labelled system-wide).
   - If D beats A+, say "friends add memory this laptop does not have".
6. **The Apple-only result**, only if Phase 1.2 has landed: one Air alone next to three Airs together.

**Fallbacks**
- A hang: 2 peers.
- A dropped peer: the host-only deployment.
- The network fails: the video plus the run.json files.
- The dashboard breaks: CLI `status`.

Never claim "impossible on one laptop" for the G14.

### Rollback Strategy
- **Git.** Tag every phase. The demo runs from its own worktree. No push to main, no force-push, no stash in the
  shared worktree.
- **Engine.** b11770 stays the default. hcpeer is opt-in on the peer and can be rolled back by script.
- **Data.**
  - Back up `iroh-session\coordinator` before dev code or schema changes, and restore it before the demo
    (big-model-runbook.md section 6).
  - Keep the 120B file, the 235B file and the peer caches.
- **Features.** Replan, managed peers, seeding and hcpeer stay flagged until they pass their checkpoints. Any phase
  can fall back to the Phase 0b configuration.

### Dependencies

**External**
- llama.cpp b11770, and the CVE-2026-34159 fix in it (INFERRED).
- Iroh and its public relays.
- Hugging Face Range requests: `ggml-org/gpt-oss-120b-GGUF` @ 7200b74b, plus the unsloth GGUFs.
- The phone hotspot or a travel router, and the hotspot's mobile data.
- Tailscale and SSH, for Block 3 only (OVERNIGHT-SETUP.md).
- Apple notarization.
- uv.

**Internal**
- The demo tag and its worktree.
- The in-flight managed-peer, seeding, bridge and hcpeer work.
- `~/.hypercompute/capacity/measurements.json`.
- The `iroh-session` home and its DB backup.
- Azim's cached layers 0-2.
- Two 200 GB Air owners willing to build, host and run overnight.
- The frontend build.

### Open Questions (only the user can decide)
1. **The headline.** "Faster with friends" (G14) or "runs what no laptop can" (Apple only)? This decides how hard
   Phase 1.2 is pushed before the hackathon.
2. **What does "can run" mean?** Any speed through paging, or a floor such as 1 tok/s?
3. **Is a 2-peer demo acceptable** if 3 peers fail the gate?
4. **Who gets the Airs first tonight?** This plan's default is the demo evidence (Phase 0b), then the design doc's S0
   and S1, then overnight work. The Azim-hosted runbook moves to Phase 1 on a 200 GB Air.
5. **May the bridge speedups and the 60 s stale window go into the demo tag** (Phase 0a)?
6. **Will Azim close more apps or lower his reserve** so his cap can stay pinned?
7. **Which 200 GB Airs build and host,** and do their owners agree to SSH, Tailscale and overnight runs?
8. **Is the demo tonight, or after the overnight session?** If after, Block 3 must leave the demo home and caches
   exactly as frozen.
9. **Trust model:** friends only, or does untrusted-peer safety matter for the vision?
10. **Mobile data:** how much is acceptable if a peer falls back to relay?
11. **Release:** when does the repo go public, and is an Apple Developer account available?

### Critique response

| # | Critique point | Verdict | What changed, or why not |
| --- | --- | --- | --- |
| 1 | Warm caches not reused; placement follows live budget, not join order | Accepted, verified (planner.py:420, :549; moe_tier.py:29-41; deployments.py:192-193; inventory.py:227-231; the pair's basis was free memory) | D11, Phase 0b.4, Phase 1.3. Revision 1's "fixed join order" step is removed |
| 2 | The freeze locks in a slow bridge | Accepted, verified (bench before and after, profile.txt) | D12 and Phase 0a.2. Partly qualified: the fix mainly shortens warm-up. For decode it saves an ESTIMATED 2-4 ms per peer per token, not most of the per-peer cost |
| 3 | Heartbeat gaps have an unknown cause; caffeinate assumes sleep | Accepted (big-model-runbook.md:29-31) | Diagnosis steps in Risk HIGH, Phase 0a.3, Phase 1.4. caffeinate stays as a cheap measure, not a fix |
| 4 | The headline loses to "just close Chrome" | Accepted, with a nuance | Control A+ and win rule D9. The nuance: the pair started with 1.6 GiB less available RAM than the alone run and still won, and at 3 Airs equal headroom exceeds the G14's RAM. So the control can strengthen the claim as well as kill it |
| 5 | The deadlock gate eats the evening; no time budget | Accepted | Time budget and cutoffs. The gate reuses the matrix's D runs |
| 6 | Hotspot under load unmeasured | Accepted, one correction | RTT recorded during the warm-up load, rehearsal, wired host. Correction: traffic is host-to-peer (a star) and peers do not talk to each other. Each hop crosses the phone's radio twice |
| 7 | The stage laptop is not the measured one | Accepted (available_min 4,669,440 verified) | Control S. Slides from another device |
| 8 | Peer code skew | Accepted, lower severity | An 890f798 agent worked in the pair run, and a8e1364 changed no agent, transport or schema file. Bundle install and just-in-time invites |
| 9 | Disk | Accepted, premise outdated | C: had 108.69 GiB free at 20:50 UTC, and the venv is 76 MB. The risk is now the 235B download filling C: or disturbing measurements (Phase 0b.2) |
| 10 | "Bigger" only shows up in Phase 3 | Accepted | D13: the Apple-only proof moves to Phase 1.2, independent of Phase 2. The pre-seeding premise is now verified by the design doc (F9) |
| A1 | "About 5 tok/s with 3 Airs" rests on a 2-point fit | Accepted | Removed. The two fits disagree 4x on the per-peer cost (15 vs 60 ms). No prediction is quoted |
| A2 | "About 10 GiB free" vs available RAM | Accepted | Context now gives available RAM from run.json |
| A3 | "Latency matters, not bandwidth" only after the bridge fix | Rejected for decode, accepted for loads | Even with the stock bridge, an 11,520-byte activation is about 1 ms of transfer, and the 3-4.5 ms bridge round trip is itself latency. Bandwidth matters for warm-up, which D12 addresses |
| A4 | caffeinate fixes the gaps | Accepted | The claim is removed |
| A5 | A 2-peer plan reuses the 3-peer caches | Accepted (false as written) | It holds only for prefix subsets of a pinned order (D11) |
| A6 | 9 files, not 7 | Accepted | The file list is now at the top. It changed again during this review |
| M1 | Phase 0.5: pin ranges, bridge fix, stale window | Accepted with changes | Phase 0a. Ranges are pinned operationally, not in code. Each change has a gate and a 45 min cap |
| M2 | Per-hop breakdown and round trips per token | Accepted | Block 2 (design doc S0) and Phase 1.5 |
| M3 | G14-alone baseline with apps closed | Accepted | Control A+ |
| M4 | Peer free RAM at plan time in run.json | Partly rejected | Already recorded in `measured.gateway_before.nodes[].memory_pools` (10.61e9 B for the Air). What is missing is the budget basis (null), added in Phase 1.3 |
| M5 | Agent-side heartbeat timestamps | Accepted | The teed join log tonight. Sequence numbers in Phase 1.4 |
| M6 | A single-Air 120B result in Phase 0 | Partly rejected | It needs 63.4 GB on an Air, a Mac host that has never run, and Airs that Track A uses. Moved to Block 3 (stretch) and Phase 1.2 |
| M7 | Dress rehearsal on the venue network | Accepted | Phase 0b.8 |
| S1 | Pin ranges, or sort by node id | Accepted | Pinned caps tonight. Sort by (budget, node id) in Phase 1.3 |
| S2 | Seed caches over USB or AirDrop | Deferred | It needs a native FNV-1a and exact per-peer tensor lists. Tonight Azim's cache already covers layers 0-2, and the rest is a 20-25 min warm-up. Kept as a Phase 2 alternative |
| S3 | "Bigger" with pooled memory, no paging | Accepted | D13 with a dense 32B. The G14 disk objection no longer applies |
| O1 | 1.12x fails the plan's own 1.2x bar | Accepted | D9. Claims come from tonight's numbers |
| O2 | "No Air has more than 16 GB" frames a problem the G14 already solves | Accepted | The opening line is rewritten |
| O3 | SSD MB per token is system-wide | Accepted | Labelled tonight, per process in Phase 1 |
