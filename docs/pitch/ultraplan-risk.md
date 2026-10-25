# Hypercompute at Cal Hacks 13.0: risk register

PR = docs/research/PAPER-RUNS-2026-10-22.md, RM = README.md, FL = docs/testplan/failures.md

## 1. Rules and eligibility

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Prior work.** The first commit is dated 2026-10-20, and hacking opened Oct 23 at 6 PM PDT. The Hive FAQ bans projects "previously worked on" and promises detection and disqualification (hive.hackberkeley.org/faq, calhacks.io). The Kimi runs on Oct 24 still depend on pre-event code. | High | Critical | Do not submit PR #3 as the hack. Either show it outside judging, or build a new piece during the event on public OSS only (MLH rules 9 and 10, github.com/MLH/mlh-policies) and disclose the old repo to organizers in writing before 8:50 AM. |
| Registration is unconfirmed (only accepted, checked-in hackers can submit). Team size is unclear: Devpost says 2-4, the FAQ says 1-4. | Unknown | Critical | Confirm at the info desk first. |
| The rented GX10 is remote | Medium | Medium | Label it "rented NVIDIA GB10" on the slide and on Devpost. |
| The repo is private with no license (RM:639-640), and Devpost needs a GitHub link. The README is stale: it still says "not yet run between two physical machines" (RM:15). | High | Medium | Fix the README, sweep for secrets, then decide on visibility. |

## 2. Claims judges can break (likelihood High, impact High)
- **Pooled decode is slower on the GX10 host.** 671B is 0.83x [0.77, 0.90] and Kimi is 0.86x [0.82, 0.89] (PR:238, PR:271). Lead with first-token time instead: 671B goes from 24.7 s to 10.4 s, 2.4x faster, from run.json via ttft_stats.py. Then name the decode cost and its cause, one blocking round trip per peer per token (PR:79-87).
- **Host swap is not pooling.** The 120B at 56 tok/s on the GX10, against 3.3 on the G14 (PR:206-217), comes from changing the host. Always compare against the same host running alone.
- **"Impossible on one machine" is false.** The G14 alone runs the 671B at 0.48 tok/s (PR R4), and the GX10 alone runs Kimi at 1.80. Say instead that it fits in no single machine's memory and that pooling cuts SSD paging from 671 to 254 MiB/token (summary_kimi_gx.md).
- **The 1T claim needs caveats.**
  - It is an IQ1_S 1-bit quant with 32B active.
  - Peers hold 32.9 of 193 GiB, and the host still pages 83.9 GiB of experts from SSD (PR:253-277).
  - There is no quality benchmark, and the Fibonacci output was not archived (PR:278-279). Re-run the coherence check and save the text.
- **"First" claims are false.**
  - exo ran the 671B (blog.exolabs.net/day-2) and Kimi K2 (jeffgeerling.com).
  - Mesh-LLM already uses Iroh (github.com/Mesh-LLM/mesh-llm).
  - latentra pitched pooled laptops at Cal Hacks 12.0 (devpost.com/software/latentra).
  - Claim the combination only.
- **The planner was wrong.** It predicted 1.77x and measured 0.83x (PR:249-252). Never quote planner output as a result.
- **Small samples.** These are 4 vs 4 reps, and mac-azim was on the relay (PR:196-204, PR:246). Show the confidence intervals.
- **Concurrency is not measured** (CHECKPOINT-2026-10-24 "Next" items 1 and 3).

## 3. Live demo failure modes

| Failure | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Client isolation or NAT forces the relay: about 1 MiB/s, RTT 60-580 ms, and a peer dropped mid-load (PR:171-174). The 12.0 Wi-Fi failed (blog.dylanlu.com/cal-hacks-12). | High | High | Bring our own router or hotspot, test it with `hypercompute doctor` (docs/RUNBOOK.md:130), keep caches warm. |
| A Mac sleeps. The node goes stale after 15 s and the deployment fails (RUNBOOK:134). `caffeinate` does not stop lid-close sleep (FL:206). | Medium | High | Keep it on AC with the lid open, run `caffeinate -dims`, turn Low Power Mode off. |
| The host sleeps. Peers quit after 5 min without it (FL:39). | Medium | High | Keep the host on AC with sleep disabled. |
| GPU TDR on the G14. No incident is recorded in the repo and it is untested. Windows' default GPU timeout is 2 s (learn.microsoft.com, tdr-registry-keys). | Low to Medium | High | Use the G14 as a peer, not as the live host for big models. |
| Load takes 239-517 s, and a cold seed takes 676-1416 s (PR R6). Judges spend 5 minutes per table (Hive FAQ). | Certain if loaded live | Critical | Load before judging, keep it warm, never reload during a judging slot. |
| A Kimi answer takes about 39 s for 32 tokens | High | Medium | Run a short prompt live and keep a recorded side-by-side as backup. |

## 4. Security and privacy questions
- LAN-mode RPC has no authentication (RM:506). Demo Iroh mode only: encrypted QUIC with the worker on loopback (RM:509, docs/iroh-networking.md:1-25).
- The host gets broad access to the peer process, so the trust model is friends only (RM:503-520).
- The CVE-2026-34159 fix in b11770 is inferred, not verified. Check before claiming it.
- Consent: friends join with a one-time code that expires in 15 minutes and is stored as a keyed hash. A friend can stop, pause or leave locally (RM:72-81, RM:324-347).
- The host sees prompts and peers see activations. Do not claim private inference.

## 5. Irreversible decisions
1. Submitting prior work at the 9 AM PDT deadline risks disqualification and a record with Cal Hacks and MLH. Ask the organizers first.
2. Making the repo public turns its dated history into evidence. Choose a license and sweep for secrets first.
3. Devpost text and video are permanent. Every number in them must trace to a run.json.
4. Releasing the GX10 or deleting weights before the demo means re-downloading 170-193 GB.
