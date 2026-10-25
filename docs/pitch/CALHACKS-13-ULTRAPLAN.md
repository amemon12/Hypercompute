## Task: CalHacks 13.0 pitch for Hypercompute

Nothing below has been run. This is a plan only, and you asked to pause after packaging. Source keys: PR = docs/research/PAPER-RUNS-2026-10-22.md. TT = first-token and total-time bootstrap over run.json files, computed by C:\Users\operator\AppData\Local\Temp\claude\C--\33c9e5b2-ade6-449a-b9ce-e5e801ae75aa\scratchpad\ttft_stats.py. KIMI = results/capacity/paper-2026-10-24/summary_kimi_gx.md.

### Context (current state, target state, constraints)

**Current state**
- Branch codex/iroh-transport has 48 commits, all dated Oct 20 to 24 (git log). Draft PR #3 is open.
- Measured runs on 120B, 235B, 671B and 1T models across the GX10 (rented), the G14 and three M4 Airs (PR R1 to R6, KIMI).
- The README is stale: it still says "not yet run between two physical machines" (README.md:15) and "578 protocol tests" (README.md:144).
- The repo is private and has no license (README.md:639-640).

**Target state**
- A 3-minute pitch plus a Devpost entry that cannot be broken in Q&A.
- Lead with "problem speed": how fast the system reads your prompt (prefill), time to first token, and time to a finished answer. It is not decode speed, and the pitch says so.

**Hard constraints**
- **Prior-work ban.** The Hive FAQ says "all implementation must occur during the hacking period" and that cheating or prior work leads to disqualification (hive.hackberkeley.org/faq). Devpost says "All work must be done during the hackathon" (cal-hacks-13-0.devpost.com). MLH rule 10 bans open-sourcing earlier code just to reuse it (github.com/MLH/mlh-policies). **PR #3 and its numbers cannot be the Cal Hacks entry.**
- **Deadline.** Submissions close Oct 25 at 9:00 AM PDT (8:50 on the Hive schedule). You must be at the venue by 8 AM, and only accepted, checked-in hackers can submit.
- **Judging.** 3-minute pitch plus 2 minutes of Q&A at your table. The criteria are Application, Intentionality, Functionality, Creativity and Technical Complexity.
- **Honesty rule.** The strongest framing of true facts only.

**Three paths**
- **P1 (compliant).** If you are checked in, rebuild a minimal pool during the event on public open source (llama.cpp RPC and the iroh crate), and present only numbers measured at the event.
- **P2.** Ask the organizers in writing whether disclosed prior work can be shown.
- **P3.** Use this package at a venue that allows prior work.

The claims sheet below applies in full under P2 or P3. Under P1, only numbers re-measured at the event can be used.

### How we stand vs past Cal Hacks winners

| Criterion | What winners showed | What we have | Gap | Action |
|---|---|---|---|---|
| Application | A human story you can say in one sentence (Duet 11.0, FaceTimeOS 12.0) | Infrastructure | Weak | Story: "your club runs a private 120B model on laptops it already owns" |
| Functionality | A smooth live demo, plus a fallback (blog.dylanlu.com/cal-hacks-12) | Loads take 239 to 517 s (PR R6) | High risk | Preload before judging, keep a recorded split-screen ready |
| Technical Complexity | Rarely decides 1st overall, but has its own prize | Pool across Linux, Windows and macOS on stock llama.cpp, MoE-aware planner, about 2,000 test cases | Strong | Show per-peer layer lines and /api/network live |
| Creativity | Novelty | latentra, Airpool and HPC all pitched pooled compute and won nothing (devpost.com/software/latentra) | Medium | Lead on model size and time to first token, not on "pooling" |
| Spectacle | 11 of 28 top-3 places went to hardware (Devpost galleries 1.0 to 12.0) | A table of real laptops | Neutral | Make every friend's join visible on screen |
| Speed multiplier | Duck Duck Goose: about 1300x (45 s to 30 ms) won $10k (devpost.com/software/duck-duck-goose-jelxr9) | First token 2.4x sooner on the 671B (TT) | Smaller, but real | Show one before/after number with its 95% interval |
| Eligibility | Built during the event | Built before the event | Blocking | Resolve P1, P2 or P3 first |

Prize targets: Most Technically Complex, Hacker's Choice, Best Solo (if you enter alone; team-size rules conflict), with 1st Overall as a stretch. No Annapurna prize was posted at the time of the scout (hive.hackberkeley.org/prizes).

### The pitch

**One-liner.** Hypercompute is multiplayer computing. Your friends' laptops join one encrypted pool, and a 671B model starts answering 2.4x sooner than the big box can alone.

**30-second version.** Open frontier models no longer fit in one machine. DeepSeek-V3.1 is 158.8 GiB and Kimi-K2.6 is 193 GiB, while our biggest box has 121.6 GiB. So that box pages experts from its SSD, 684 MiB for every token. With Hypercompute, friends type one join code, and their laptops on Windows and macOS take whole layers over an encrypted Iroh tunnel. Each laptop pulls its own slice from Hugging Face. The 671B then reads your prompt 2.42x faster, its first token arrives in 10.4 s instead of 24.7 s, and SSD reads drop 95%.

**3-minute demo script** (Tier A: Kimi pooled, preloaded)
- **0:00 to 0:20, hook.** Point at the table: "This is a one-trillion-parameter model, and none of these machines can hold it in memory."
- **0:20 to 0:50, join.** A friend runs `hypercompute join` with a code. The dashboard goes from verifying to eligible after a real RPC handshake, and the per-peer layer lines appear. If live join is too risky, show a peer re-joining a running pool.
- **0:50 to 1:30, live prompt.** Send a short coding prompt to the pooled 1T model. First token arrives at about 19 s (TT 19.08 s). While it streams, show direct paths and RTT on /api/network.
- **1:30 to 2:10, recorded split-screen.** The 671B alone against the pool, with timers: first token 24.7 s vs 10.4 s, full answer 41.8 s vs 31.2 s.
- **2:10 to 2:35, SSD chart.** 684 vs 37 MiB per token on the 671B, and 671 vs 254 on Kimi.
- **2:35 to 3:00, honest cost and close.** "Pooling costs us decode on a big host, 0.83x, because each peer adds one network round trip per token. We win where people wait: reading the prompt and the first answer."

### Claims sheet

| Claim as we will say it | Exact number | Source |
|---|---|---|
| 671B reads your prompt 2.4x faster pooled | 6.36 vs 2.63 tok/s, 2.42x [2.19, 2.72] | PR:238 |
| 671B first token 2.4x sooner | 24.67 s to 10.36 s, ratio 0.420 [0.373, 0.462] | TT, index_671b_gx.jsonl |
| 671B full answer 25% sooner (64-token prompt, 32-token answer) | 41.84 s to 31.20 s, ratio 0.746 | TT |
| 671B SSD reads 95% lower | 684 to 37 MiB/token, 0.054x [0.026, 0.097] | PR:238 |
| 1T model reads your prompt 1.5x faster | 3.30 vs 2.19 tok/s, 1.51x [1.41, 1.59] | PR:271 |
| 1T first token 1.48x sooner, full answer 14% sooner | 28.25 s to 19.08 s; 45.48 s to 39.23 s | TT, index_kimi_gx.jsonl |
| 1T SSD reads 62% lower | 671 to 254 MiB/token, 0.38x | PR:271 |
| One gaming laptop plus 3 MacBook Airs: 120B first token 2.7x sooner | 36.17 s to 13.44 s; prefill 2.99x [2.64, 3.35]; decode 1.27x [1.10, 1.46] | TT; PR:114 |
| Serves many users: with 8 chats at once, the gaming laptop plus 2 MacBook Airs gives 1.34x the total output of the gaming laptop alone (gpt-oss-120b, G14 + 2 MacBook Airs, home network, 128-token prompt, 64-token answer) | 16.44 vs 12.19 tok/s in total, 1.34x [1.24, 1.48]; 4 chats 11.85 vs 8.10, 1.46x [1.29, 1.68]; 1 chat 2.28x [2.00, 2.54]. One row per side, 3 batches. Each chat is slower than one chat alone (about 2.05 tok/s each at 8) | PR R7.5; NUMBERS.md 1d |
| One box serves 8 users of a 671B model at 3.09x the total output of one user (GX10 alone, 64-token prompt, 32-token answer) | 5.54 vs 1.79 tok/s in total, 3.09x [2.52, 3.69]; 1T 4.64 vs 1.78, 2.60x [2.37, 2.85]. One row per N, 2 batches. Not pooled | PR R6.y |
| 120B laptops full answer about 1.95x sooner (128-token prompt, 64-token answer) | 54.44 s to 27.93 s, ratio 0.513 [0.453, 0.577] | TT |
| Peers fetch their own layers from Hugging Face, about 5x the host link | 13 to 45 MB/s from HF vs 8.6 MiB/s from the host | PR R2 lines 128-130 |
| Models about 20x larger than any pooled-device hackathon project we found | 671B and 1T vs 35B (pooled.run) | github.com/Nehanth/pooled; landscape scout |

**Do not say** (each with the strongest true replacement)
- **"Pooling makes generation faster."** False on the GX10 host: 0.83x on the 671B and 0.86x on the 1T (PR:238, PR:271). Say instead: "decode is faster when no single laptop can hold the model: on the laptops-only pool (home network) 5.44x
  [5.42, 5.47] on Llama 3.3 70B, 2.59x [2.57, 2.62] on Qwen2.5-VL 72B and 2.28x [2.00, 2.54] on gpt-oss-120b; it is
  slower when the model fits one machine (80B MoE 0.46x [0.36, 0.62]). On the GX10 host it depends: 0.83x and 0.85x
  with four peers (one on the relay), parity on the 671B with three direct peers (1.00x [0.92, 1.09], interleaved, PR R8.3), and 0.43x on
  Maverick 400B, which nearly fits the GX10." Each
  laptop ratio is one alone row against one pooled row (PR R7.1 to R7.4; NUMBERS.md).
- **"Impossible on one machine."** False: the GX10 alone runs both at about 1.8 tok/s, and the G14 alone runs the 671B at 0.48 tok/s (PR R4, R6). Say instead: "it fits in no single machine's memory."
- **"First ever" anything.** False: exo ran the 671B on 8 Mac minis (blog.exolabs.net/day-2), PRIMA ran a 770B across two machines, and Mesh-LLM already pools over Iroh. Say instead: "the only system we found that has shown this combination: three operating systems, stock llama.cpp, peers fetching their own layers, and measured 671B and 1T runs."
- **"56 tok/s on the 120B."** That is the GX10 alone, so it comes from swapping the host, not from pooling (PR:206-217). Do not use it.
- **"The planner finds the optimal layout."** It predicted 1.77x and measured 0.83x (PR:251). Say instead: "exact optimizer, agrees with a brute-force oracle on 3,000 cases."
- **"Every user gets full speed."** False: each chat slows as more join (at 8 chats each pooled chat gets about 2.05
  tok/s, and a 64-token answer takes 59.8 s against 11.7 s for one chat alone on the pool) (PR R7.5). Say instead: "more
  people at once get more total output; each answer takes longer." Pooled concurrency was measured on the laptops
  only, not with the GX10 as host.
- **"Private inference."** The host sees prompts and peers see activations. Say instead: "encrypted, authenticated tunnel."
- **"Fast on long documents."** Only 58 to 133-token prompts were measured. Say instead: "measured on short prompts; long prompts are next."

### Risk Assessment

**HIGH**
- **Prior-work disqualification.** The ban is explicit, with detection. Mitigation: P1, P2 or P3, decided before any submission.
- **Registration or check-in unknown.** No walk-ins are allowed. Confirm at the info desk.
- **Live load or seeding at the table.** Cold start takes 17 to 32 minutes (PR R6). Preload, and never reload during judging.
- **Venue Wi-Fi with client isolation.** The relay path ran at about 1 MiB/s and a peer dropped (PR:171-174). Bring your own 5 GHz router.
- **Decode question in Q&A.** Answer it first, in the script.

**MEDIUM**
- A Mac sleeps and the deployment fails (docs/RUNBOOK.md:134). Use AC power, `caffeinate -dims`, lid open.
- The GX10 is rented, so check whether the rental lets you bring it to the venue. If not, use Tier B.
- The 120B model is deleted from the G14, which has only 23 GB free. Re-downloading it needs the 671B file moved first.
- The Kimi coherence text was never saved (PR:279). Re-run the check and archive the output.
- Small samples: 4 vs 4 reps, and mac-azim was on the relay. Show the intervals.

**LOW**
- A GPU timeout (TDR) on the G14. Use it as a peer only.
- The CVE-2026-34159 fix being in llama.cpp b11770 is inferred, not verified. Do not claim it.

### Execution Phases

**Phase 1: Package (this step, then pause)**. Complexity: Low.
- [ ] Freeze the claims sheet above, with every number tied to a run.json or a PR line.
- [ ] Draft organizer email or desk question (P2): "We built a prototype Oct 20 to 24; may we demo it, or must we rebuild?"
- [ ] Draft README fixes for lines 15, 78, 144, 596 and 621 (do not commit yet).
- [ ] Draft Devpost text: 2 to 3 sentence pitch, the one-liner, and an honest AI-tools answer.
- Risk: packaging prior work as an entry before the rules question is answered.
- Checkpoint: you pick P1, P2 or P3. Nothing is sent, committed or published before that.

**Phase 2: Build the demo and assets**. Complexity: High under P1, Medium under P2 or P3.
- [ ] Under P1, rebuild a minimal pool during the event (llama.cpp RPC over an iroh tunnel, one host and one peer) and measure it fresh.
- [ ] Record the split-screen: alone vs pooled, first token and full answer, with on-screen timers.
- [ ] Record the join flow and the "friend presses Stop, deployment fails cleanly" clip.
- [ ] Re-run the Kimi coherence check and archive its text.
- [ ] Make a 4-slide deck: hook, problem-speed chart with 95% intervals, architecture, honest-cost slide.
- [ ] Set up the travel router, warm caches at home, and seed the Mac mini on home Wi-Fi.
- Risk: the G14 is short on disk for the 120B, and the GX10 rental terms are unknown.
- Checkpoint: the video plays offline, and every number on screen matches the claims sheet.

**Phase 3: Rehearse and submit**. Complexity: Medium.
- [ ] Rehearse 3 minutes, plus 2 minutes of Q&A against the "do not say" list, three times.
- [ ] Arrive by 8:00, start the router, join peers, load the model about 6 minutes ahead, send one warm-up prompt.
- [ ] Submit by 8:50 AM PDT with an image, the GitHub link, the table number and the team name. Do this only under P1, or P2 with written approval.
- Risk: missing the hard 9 AM deadline, or submitting work that breaks the rules.
- Checkpoint: Devpost confirmation, and the live pool answering before 10:30 judging.

### Rollback Strategy
- **Rules (P2 refused):** do not submit PR #3. Fall to P1 if time allows, otherwise P3.
- **Live demo fails:** play the recorded split-screen. If that fails too, show the 7B join demo and call it a capacity demo.
- **No GX10:** use Tier B (120B on the G14 plus 3 Macs), then Tier C (G14 plus 1 Mac).
- **Network fails:** move to LAN transport on your own router only, and say it is unauthenticated if asked.
- **Repo visibility:** keep it private until the license and secret sweep are done; give judges collaborator access instead.

### Dependencies
- Organizer ruling on prior work, and confirmation that you are registered and checked in.
- Team-size answer: Devpost says 2 to 4, the Hive FAQ says 1 to 4.
- GX10 rental terms and power at the table.
- A 5 GHz travel router, plus a phone uplink for the Iroh relay bootstrap.
- Warm Hugging Face caches on every peer, and the Mac mini replacing mac-azim.
- G14 disk space for the 120B (59.03 GiB).

### Open Questions (decisions only you can make)
1. P1, P2 or P3? Are you accepted and checked in for Cal Hacks 13.0?
2. Enter solo (Best Solo Hack) or with a team? Who is on it?
3. Bring the rented GX10 to the venue, or go laptops-only with Tier B?
4. Delete or move the 671B file on the G14 to make room for the 120B?
5. Make the repo public, and under which license, or give judges private access?
6. Approve the organizer question text before it is sent?
