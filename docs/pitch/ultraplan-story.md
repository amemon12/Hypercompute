# Hypercompute at Cal Hacks 13.0: story and positioning

PR = docs/research/PAPER-RUNS-2026-10-22.md. TT = the scout's bootstrap of TTFT/total from run.json (ttft_stats.py).

## 0. Blocker first
The Hive FAQ says "all implementation must occur during the hacking period," and prior work means losing any prize (hive.hackberkeley.org/faq; cal-hacks-13-0.devpost.com). Every commit is dated Oct 20 to 24 (git log), and runs R1 to R6 were done before the event. Submissions close Oct 25 at 9:00 AM PDT, and only checked-in hackers can submit. So PR #3 and its numbers cannot be the entry. There are two honest paths. If you are checked in, rebuild a minimal pool on public open source (llama.cpp RPC, iroh) during the event and show only numbers measured there. If not, save this story for a venue that allows prior work.

## 1. One line and hook
One line: Hypercompute is multiplayer computing. Friends' laptops join one encrypted pool, and a 671B model starts answering 2.4x sooner than the big box can alone.
Hook: a friend types `hypercompute join <code>`, and the first token from DeepSeek-V3.1 671B arrives in 10.4 s instead of 24.7 s (TT; PR 238).

## 2. Arc
- Problem: frontier open models no longer fit in one box. The 671B is 158.8 GiB and Kimi-K2.6 is 193 GiB, while the GX10 has 121.6 GiB. So the box pages experts from SSD, 684 MiB per token (PR R4, R6, 238).
- Insight: friends' idle memory takes the place of SSD reads and shares the prompt work. SSD reads drop 95%, to 37 MiB per token (PR 238).
- Demo: do the join and a small-model answer live. Show the 671B side-by-side as a recorded clip, because a cold pooled start takes 17 to 32 min (PR R6). The 12.0 winner advised having a hardcoded fallback (blog.dylanlu.com/cal-hacks-12).
- Why now: open-weight models are now 671B to 1T MoE. Stock llama.cpp RPC works across CUDA and Metal. Iroh QUIC gets through NAT without a VPN (docs/iroh-networking.md 1-25).

## 3. Speed, said truthfully
"Problem speed" here means prompt (prefill) speed, time to first token and time to a full answer. It does not mean decode speed.

| Say | Number | Source |
|---|---|---|
| 671B reads your prompt 2.4x faster | 6.36 vs 2.63 tok/s, 2.42x [2.19, 2.72] | PR 238 |
| 671B first token 2.4x sooner, full answer 25% sooner | 24.7 to 10.4 s; 41.8 to 31.2 s | TT |
| 1T reads your prompt 1.5x faster | 3.30 vs 2.19, 1.51x [1.41, 1.59] | PR 271 |
| Laptop + 3 MacBook Airs, 120B: first token 2.7x sooner | 36.2 to 13.4 s; prefill 2.99x | TT; PR 114 |

Never say:
- That pooling speeds up generation. With the GX10 as host, pooled decode is 0.83x and 0.86x (PR 238, 271).
- That this is impossible on one machine. The GX10 alone runs both of the biggest models at about 1.8 tok/s (PR R6).
- That the 120B runs at 56 tok/s because of pooling. That number is the GX10 alone, a host swap (PR 214).
- That anything is a "first". exo ran the 671B on 8 Mac minis at 5.37 tok/s (blog.exolabs.net/day-2). prima.cpp ran a 770B across two machines (opencpil.github.io/prima.cpp). Mesh-LLM already pools machines over Iroh (github.com/Mesh-LLM/mesh-llm).

The strongest true claim about scale: our models are about 20x larger than the biggest model any pooled-device hackathon project we found has run, 671B and 1T against 35B (pooled.run; landscape scout).

## 4. Targets
- Main targets: Most Technically Complex, Hacker's Choice, and Best Solo if you enter alone. The Hive FAQ allows teams of 1 to 4 but Devpost says 2 to 4, so confirm.
- Long shots: 1st Overall and Best Hardware.
- Sponsor tracks, only if the sponsor's product is really used: MLH Vultr (a cloud GPU as a peer) and Spacetime (as the session backend).
- No Annapurna prize is listed yet (hive.hackberkeley.org/prizes).

## 5. Scorecard vs past winners
| Criterion | What winners won on | Us | How to cover |
|---|---|---|---|
| Application | one-sentence human story (Duet, FaceTimeOS) | weak, it is infra | "your club runs a private 120B model on laptops it already owns" |
| Functionality | smooth live demo | risky | preloaded pool, recorded clip |
| Technical Complexity | rarely decisive overall | strong | show the pool live across Windows, macOS and Linux on the dashboard |
| Creativity | novelty | medium: latentra, Airpool and HPC all pitched pooling and won nothing | lead on model size and time to first token |
| Spectacle | 11 of 28 top-3 places went to hardware | a table of real laptops | make each join visible |

Infrastructure has won overall once (Kanashi, 3.0). Pooled compute has never placed top-3 (Devpost galleries 1.0 to 12.0).

## 6. Weak spots and cover
- Decode is slower. Say so first: each peer adds one RPC round trip per token (PR R1.2).
- The GX10 is rented. Lead with the 120B result, which used only laptops.
- Prompts were 58 to 133 tokens long. The 512-token run has not been done (CHECKPOINT-2026-10-23, Next item 4).
- The 671B and 1T comparisons are only 4 runs against 4. Both models are 1-bit quants. The Fibonacci answer Kimi wrote was not saved (PR 278-279).
- The README is out of date (lines 11-17). Fix it before judges read it.
