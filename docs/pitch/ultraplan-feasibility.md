## Live demo feasibility at a Cal Hacks table

**Check this before anything else.** Cal Hacks 13.0 bans prior work, and its deadline is 9 AM PDT Oct 25 (hive.hackberkeley.org/faq; cal-hacks-13-0.devpost.com). Everything below only matters if organizers approve in writing or the work is rebuilt during the event. Otherwise it applies to a later event.

### Timings (all measured)

| config | load to ready | first token | full answer |
|---|---|---|---|
| Kimi 1T, GX10 + 4 peers | 367-369 s (seeding 5 s warm, 676 s cold) | 19.1 s | 39.2 s (32 tok) |
| Kimi 1T, GX10 alone | 257-266 s | 28.3 s | 45.5 s |
| 671B, GX10 + 4 peers | 503-517 s after 856-1416 s seeding | 10.4 s | 31.2 s |
| 671B, GX10 alone | 239-256 s | 24.7 s | 41.8 s |
| 120B, G14 + 3 Macs | 126-167 s warm (566 s seeding cold) | 13.4 s | 27.9 s (64 tok) |
| 120B, G14 alone | 38-44 s | 36.2 s | 54.4 s |
| 671B, G14 alone | 50 s | about 89 s | about 153 s |

Sources: PAPER-RUNS-2026-10-22.md lines 101, 128-129, 230-232, 263-265 and R6. First-token and answer times come from the inventory scout's bootstrap over run.json (scratchpad ttft_stats.py).

What this means:
- Nothing loads inside a 5-minute slot. Every live config has to be loaded beforehand and left running. Switching models costs 4 to 9 minutes.
- The GX10 cannot hold the alone and pooled deployments at the same time. Kimi alone uses 77.3 GiB of CUDA0 out of 121.6 GiB (PAPER-RUNS line 254). So the alone vs pooled comparison has to be recorded.
- The 671B on the G14 alone is too slow to show live.

### Hardware logistics
- **GX10.** It is rented and sits on home Wi-Fi at 192.168.1.221 (CHECKPOINT-2026-10-24.md line 17). Tier A needs it at the table with mains power, so check the rental terms. Leaving it at home and pooling over the internet has not been measured. The closest test ran at about 1 MiB/s, RTT 60-580 ms, and a peer dropped (PAPER-RUNS lines 173-174).
- **G14.** I checked it today. It has the 671B file (158.8 GiB), the 0.5B and 7B models, and only 23 GB free. The 120B (59.03 GiB) is gone. Tier B needs the 671B file moved or deleted first, which is your call.
- **Mac mini.** It replaces mac-azim and must be seeded at home. Wi-Fi gives 45 MB/s from Hugging Face; a hotspot gives about 300 KB/s (PAPER-RUNS lines 179, 295).

### Network plan
- Assume the venue blocks laptop-to-laptop traffic (client isolation; README lines 155-157). The 12.0 Wi-Fi failed (calhacks.io FAQ).
- Bring a 5 GHz travel router and put every machine on it. Use a phone tether as the uplink only for Iroh's relay bootstrap, which needs internet (iroh-networking.md line 182). Iroh with no uplink at all is untested.
- Fallback is the default LAN transport (iroh-networking.md lines 18-20). llama.cpp needs no internet (TEST_PLAN.md line 306), but LAN RPC is unauthenticated. That is acceptable only on your own router, and you should say so if asked.
- Never seed at the venue.

### Demo tiers
- **A (GX10 at the table).** Kimi 1T pooled, loaded in advance.
  - 0:20: send a short prompt.
  - First token at about 19 s, answer done at about 40 s. While it runs, show the per-peer layer lines in the llama.cpp log, /api/network direct paths, and SSD MiB/token.
  - Then play the recorded alone vs pooled clip.
- **B (laptops only).** 120B on G14 + 3 Macs. First token is 2.7x faster than the G14 alone, and decode is also 1.27x faster. This is the only config where pooled decode clearly wins.
- **C (minimum).** 120B on G14 + 1 Mac. Load 72-80 s, first token 24.0 s, answer 38.8 s against 54.4 s alone. If the 120B cannot come back, show the join flow on the 7B, call it a capacity demo, and play the big-model video.

Morning runbook:
- Be at the venue by 8:00 (required).
- Bring up the router, join peers under `caffeinate -dims` (README line 240), and load the model (about 6 minutes for Kimi pooled).
- Send one warm-up request and be ready well before 10:30 judging.
- Keep the host alive: friend workers stop on their own after 5 minutes without the host (README Security model section).

### Recorded backup must show
1. A split screen of the same prompt with timers: Kimi first token 28.3 s alone vs 19.1 s pooled, full answer 45.5 s vs 39.2 s.
2. The join flow: one-time code, then verifying, then eligible after a real RPC handshake, then layers landing on each peer.
3. SSD reads per token: 671 vs 254 MiB on Kimi, 684 vs 37 MiB on the 671B.
4. A correct coding answer. This has to be re-recorded because the Fibonacci text was never archived; that run.json holds only a 16-token benchmark.
5. A friend presses Stop and the deployment fails cleanly.

Keep it under 3 minutes. Devpost sets no general video limit.

### Packaging
- **Devpost.** A 2-3 sentence pitch, table number, team name, an image and the GitHub link are all required (Hive FAQ; Devpost). The repo is private and has no license (README lines 639-640), so make it public or grant judges access. Answer the AI-tools question honestly (MLH rules).
- **README.** Fix the stale claims at lines 15, 78, 144, 596 and 621 before judges read it.
- **Deck.** Four slides: hook, problem-speed chart with 95% intervals, architecture, and a slide stating that decode is slower pooled.
- **Join.** Today a friend needs to clone the repo, install uv, run `engine install`, then `join --host IP:8000 --code CODE` (README lines 201-205, 240), or install the wheel instead (README line 317). There is no zero-install one-liner, so do not promise judges they can join from their own laptops at the table.
