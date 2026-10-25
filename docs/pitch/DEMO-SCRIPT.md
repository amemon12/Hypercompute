# Hypercompute live demo: runbook and 3-minute script

Written 2026-10-24 for Cal Hacks 13.0 judging.

Every number here comes from one of these:
- `docs/research/PAPER-RUNS-2026-10-22.md`, sections R1 to R6 ("PR").
- `results/capacity/paper-2026-10-2*/summary_*.md`.
- First-token and full-answer times computed from the run.json reps of the same index files ("TT"). This is the `ttft_stats.py` bootstrap, re-run on 2026-10-24.

Two things are not measured yet and are marked **being measured**:
- the 24B to 671B laptop-pool sweep;
- the concurrency test.

Both run tonight. Their numbers are bracketed placeholders, such as `[SWEEP: d70 laptops-pool decode]`. Section 9 lists every placeholder. Fill them in from run.json before judging, or cut the line.

Every pooled number below was measured with **mac-azim** (MacBook Air M4, 16 GiB, 7.5 GiB cap) as the fourth peer. If the Mac mini takes that seat, the live pool is a new configuration. Quote the old numbers as "measured with a MacBook Air in that seat".

---

## 0. The plan in one paragraph

**Live:** Kimi K2.6 (1T) is preloaded and pooled on the GX10 + G14 + 3 Macs. We send one short prompt and the judges watch the first token arrive.

**Recorded:** a split screen of DeepSeek V3.1 671B, GX10 alone against the pool, with timers. We cannot show that comparison live. The GX10's GPU budget is 79.06 GiB. The 671B uses 77.3 GiB of it alone and 76.5 GiB pooled, so the two cannot run side by side.

**Said by us, before anyone asks:** pooled generation is slower on the GX10. The win is reading the prompt and getting the first token.

If the GX10 is not at the table, use Tier B (section 5): laptops only, gpt-oss-120b, the one model where pooled decode is clearly faster (1.27x with three Macs, 1.25x with one).

---

## 1. Numbers card (print it, tape it to the table)

Conditions for all GX10 rows:
- Home Wi-Fi, Oct 24, temperature 0, seed 7, ignore_eos, no prompt cache.
- Prompt setting 64 tokens (58 to 66 actual), 32-token answer.
- 1 warmup plus 2 measured reps per run, 2 runs per config, so 4 vs 4 reps.
- Peers: G14 (CUDA, 9 GiB), mac-a and mac-b (11.5 GiB each), mac-azim (7.5 GiB).
- Brackets are 95% bootstrap intervals.

| Claim | Number | Extra condition | Source |
|---|---|---|---|
| Kimi 1T: first token sooner pooled | 28.25 s alone, 19.08 s pooled (1.48x sooner; ratio 0.675 [0.629, 0.728]) | all 4 peers direct, RTT 6 to 51 ms | TT, index_kimi_gx.jsonl |
| Kimi 1T: full answer | 45.48 s alone, 39.23 s pooled (ratio 0.863 [0.818, 0.912]) | 32-token answer | TT |
| Kimi 1T: prompt reading | 2.19 vs 3.30 tok/s, 1.51x [1.41, 1.59] | | PR R6 |
| Kimi 1T: SSD reads | 671 vs 254 MiB per token, 0.38x [0.37, 0.40] | | PR R6 |
| Kimi 1T: generation (the cost) | 1.799 vs 1.540 tok/s, 0.86x [0.82, 0.89] | | PR R6 |
| 671B: first token sooner pooled | 24.67 s alone, 10.36 s pooled (2.4x sooner; ratio 0.420 [0.373, 0.462]) | mac-azim on the Iroh relay at 47 to 84 ms | TT, index_671b_gx.jsonl |
| 671B: full answer | 41.84 s alone, 31.20 s pooled (ratio 0.746 [0.694, 0.797]) | 32-token answer | TT |
| 671B: prompt reading | 2.63 vs 6.36 tok/s, 2.42x [2.19, 2.72] | | PR R6 |
| 671B: SSD reads | 684 vs 37 MiB per token, 0.054x [0.026, 0.097] (95% fewer) | | PR R6 |
| 671B: generation (the cost) | 1.805 vs 1.497 tok/s, 0.83x [0.77, 0.90] | | PR R6 |
| One machine does run them | GX10 alone: 671B 1.805 and Kimi 1.799 tok/s. G14 alone: 671B 0.481 tok/s, 2485 MiB per token, first token 88.68 s, full answer 153.13 s | G14 row: 64/32, 2 reps, Oct 23 | PR R4, R6; TT |
| Model vs memory | 671B file 170.5 GB (158.8 GiB, converted from PR R4's 170.5 GB); Kimi 193.16 GiB (1026B params, 32B active, IQ1_S); GX10 121.6 GiB unified | | PR R4, R6 |

Laptops-only rows. Host: G14 (RTX 5070 Ti Laptop 12 GiB, 31.1 GiB RAM). Peers: up to three MacBook Air M4 16 GiB at 7.5 GiB each. Building Wi-Fi.

| Claim | Number | Condition | Source |
|---|---|---|---|
| 120B: first token sooner with 3 Macs | 36.17 s alone, 13.44 s pooled (2.7x sooner; ratio 0.371 [0.325, 0.424]) | Oct 22, 128-token prompt, 64-token answer, 9 vs 9 reps | TT, index_120b.jsonl |
| 120B: full answer | 54.44 s vs 27.93 s (ratio 0.513 [0.453, 0.577]) | same | TT |
| 120B: generation, a clear decode win | 3.296 vs 4.198 tok/s, 1.27x [1.10, 1.46] | same; a code workflow shared the G14 from 00:44 UTC, so D3 is likely pessimistic | PR R2 |
| 120B: prompt reading | 3.68 vs 11.03 tok/s, 2.99x [2.64, 3.35] | same | PR R2 |
| 120B: SSD reads | 275 vs 41 MiB per token, 0.15x [0.12, 0.19] | same | PR R2 |
| 120B with 1 Mac | first token 36.17 s vs 23.97 s; full answer 54.44 s vs 38.81 s; decode 1.25x [1.11, 1.41], also a clear decode win | mac-azim, same lengths, 9 vs 6 reps | TT; PR R2 |
| 235B with 3 Macs, steady state | first token 43.50 s vs 27.86 s (ratio 0.641 [0.595, 0.695]); decode 1.08x [0.98, 1.18], an interval that includes parity | runs 3 to 6, 64/64, 9 vs 9 reps | TT; PR R1 |
| Peers fetch their own layers | 13 to 24 MB/s per Mac from Hugging Face while seeding, 45 MB/s in a single-Mac probe, against 8.6 MiB/s pushed from the host | Oct 22 building Wi-Fi | PR R1.3, R2 |
| Laptop pool, G14 + mac-a + mac-b (10.5 GiB caps), home network, both Macs direct, RTT 5 to 8 ms before each pooled row | 70B (64/32, 2 vs 2): first token 10.01 vs 5.02 s, answer 197.90 vs 39.53 s, decode 5.44x [5.42, 5.47]. 120B (128/64, 3 vs 3): 34.22 vs 4.83 s, 49.87 vs 11.67 s, 2.28x [2.00, 2.54]. VL 72B text (64/32, 2 vs 2): 13.70 vs 9.11 s, 233.60 vs 93.94 s, 2.59x [2.57, 2.62]. 80B MoE (128/64, 6 vs 3): 1.55 vs 4.32 s, 4.75 vs 10.66 s, 0.46x [0.36, 0.62], slower pooled. Other models: `[SWEEP: per-model laptops-pool first token, full answer, decode vs host alone]` | Oct 25, alone vs pooled, one pooled row per model | PR R7.1 to R7.4, `paper-2026-10-24-laptop/index_{d70,g120,vl72,n80}.jsonl` |
| Several requests at once (laptops only) | gpt-oss-120b, aggregate decode tok/s alone vs pooled: N = 1 3.829 vs 8.740, 2.28x [2.00, 2.54]; N = 4 8.095 vs 11.854, 1.46x [1.29, 1.68]; N = 8 12.187 vs 16.445, 1.34x [1.24, 1.48] | G14 alone vs G14 + mac-a + mac-b, home network, 128/64, one row per side, 3 reps at N = 1, 3 batches at N = 4 and 8. Pooled concurrency on the GX10 (R8): Maverick 400B 0.35x at N = 8, the 1T 1.08x [0.98, 1.19] at N = 8 | PR R7.5, `index_g120n4.jsonl`, `index_g120n8.jsonl` |

---

## 2. Load times (measured, use these for the schedule)

| Config | Load to ready | Peer seeding before load | Source |
|---|---|---|---|
| Kimi 1T, GX10 + 4 peers | 367 to 369 s | 5 s with warm caches, 676 s cold | PR R6 |
| Kimi 1T, GX10 alone | 257 to 266 s | none | PR R6 |
| 671B, GX10 + 4 peers | 503 to 517 s | 856 to 1416 s (the second run changed layout and re-seeded) | PR R6 |
| 671B, GX10 alone | 239 to 256 s | none | PR R6 |
| 120B, G14 + 3 Macs | 126 to 167 s | 566 s cold | PR R2 |
| 120B, G14 + 1 Mac | 72 to 80 s | | PR R2 |
| 120B, G14 alone | 38 to 44 s | none | PR R2 |

The fastest big pooled load is Kimi with warm caches, and with warm-up it still takes over 6 minutes. A judging slot is 5 minutes. So nothing loads during a slot, and a failed deployment means at least 6 minutes of dead time before the next judge.

---

## 3. Pre-judging runbook (Tier A: GX10 at the table)

### 3.1 Tonight, before leaving home

- [ ] **Catalogs.** Every peer's catalog and the host's list `kimi-k2.6-iq1_s` (and `deepseek-v3.1-ud-tq1_0` for recording). After any catalog change, restart the host and rejoin the peers. Each peer agent caches its catalog at its first seed command. If an entry is missing, the host pushes the layers itself at about 4 MiB/s (PR R6, Operational).
- [ ] **Mac mini.** Seed it on home Wi-Fi. In a single-Mac probe on building Wi-Fi a Mac pulled 45 MB/s from Hugging Face; seeding three Macs ran 13 to 24 MB/s each. On a phone hotspot it fell to about 300 KB/s per Mac (PR R1.3, R2, R5). Record its memory and cap in `[MAC MINI: memory, cap]`.
- [ ] **Pin the layout.** After one pooled Kimi run with the morning's four peers, save its placement as a `--layout` file. Then the venue load does not re-plan, and every peer's cache matches. The measured layout was:
  - mac-azim, layers 0-2 (6.9 GiB)
  - G14, layers 3-4 (6.6 GiB)
  - mac-a, layers 5-7 (9.9 GiB)
  - mac-b, layers 8-10 (9.5 GiB)
  - GX10 CUDA0, layers 11-60 (76.0 GiB)
  - GX10 CPU experts, memory-mapped (83.9 GiB)

  If the Mac mini takes mac-azim's seat, record the new first-token time from that run: `[MAC MINI: Kimi pooled first token, 1 rep]`.
- [ ] **Rehearse on the travel router at home.** Bring up the host, all four peers, the load and one prompt, with the phone tether as the uplink. The travel router has not been measured. Fill in `[VENUE ROUTER: path per peer, RTT range]`. Iroh needs internet on both ends for relay bootstrap (`docs/iroh-networking.md`, Troubleshooting).
- [ ] **GX10 address.** `rejoin_mac.sh` and `join_g14.sh` hard-code `asus@192.168.1.221`, and the checkpoint's Mac targets are Tailscale 100.x addresses. On the travel router, either reserve 192.168.1.221 for the GX10 (only works if the router uses 192.168.1.0/24) or edit the `GX=` line and the G14 script's ssh target. The manual joins in 3.3 need neither.
- [ ] **Display.** The dashboard only answers on the host's 127.0.0.1 (owner_read checks loopback client and Host header). Use either:
  - a monitor on the GX10 at `http://127.0.0.1:8001`; or
  - an SSH local forward from the display laptop, `ssh -L 8001:127.0.0.1:8001 asus@<GX10-IP>`, then open `http://127.0.0.1:8001` there. The tunnel is untested, so try it tonight.
- [ ] **Thinking off.** Send a test prompt in the dashboard chat. It should start with answer text, not reasoning. `coherence.py` sends `"chat_template_kwargs": {"thinking": false}`. If the dashboard cannot turn thinking off, send the live prompt with the same request body coherence.py uses.
- [ ] **Network view.** On the GX10, `watch -n 2 uv run --no-sync hypercompute status --port 8001` (from `~/hypercompute`). It prints the transport, each peer connection (path mode and RTT) and each node's state and heartbeat age. Check tonight that it prints nothing secret. Note which node id is which machine.
- [ ] **Record the fallback video** (section 6).
- [ ] **Pack:** power strip, router, cables, phone for the tether, USB stick with the video. Charge everything.

### 3.2 Morning timeline

Clock times assume the first judge at 10:30. Confirm on the Hive schedule. Devpost closes at 9:00 AM PDT (8:50 on the Hive schedule), so one person submits while another runs the pool.

| Time | Step | Done when |
|---|---|---|
| 8:00 | Arrive and check in (required). | Badge in hand |
| 8:10 | Table: power strip, every machine on AC. Disable sleep on the GX10 and G14. Macs: lid open (`caffeinate` does not stop lid-close sleep), Low Power Mode off. | All plugged in |
| 8:15 | Router on 5 GHz, phone tether as its uplink. Every machine on the router's SSID. Never use venue Wi-Fi. On Oct 23, with the host on isolated guest Wi-Fi and the peers on a phone hotspot, traffic went over the relay at about 1 MiB/s, RTT 60 to 580 ms, and a peer dropped mid-load (R5). | Each machine has internet |
| 8:20 | GX10: start the host (3.3). Open the dashboard and the `status` watcher. | Dashboard loads |
| 8:25 | Join the peers one at a time, each with a fresh invite (3.3). Codes work once and expire in 15 minutes. | Each card shows Ready to contribute (real RPC handshake) |
| 8:40 | Check paths in the watcher. All four should be `direct`. If one stays on the relay, restart the host once and wait for re-verification (Oct 24: after both restarts every peer came back direct). If it still is not ready, rejoin it with a fresh invite. | 4 peers eligible, paths direct |
| 8:50 | Devpost submitted (other teammate). | Confirmation page |
| 8:55 | Load Kimi pooled (3.3). Expect about 5 s of seeding, then 367 to 369 s of load, then about 2 to 3 minutes after load for warmup plus 1 rep (estimate from kimi-gx4-2 wall time). If seeding is still moving bytes after 2 minutes, a peer's cache is cold: stop. Never seed over the tether. Drop that peer and re-plan (an unmeasured config, so quote no numbers for it), or go to Tier B. | Deployment `ready`, run.json written |
| 9:10 | Warm-up: run coherence.py and save its text (3.3). It asks two questions, the second up to 300 tokens; at about 1.5 tok/s budget 5 minutes. This also archives the coherence text that was never saved on Oct 24. | `[ARCHIVE: coherence output path]` filled |
| 9:20 | Rehearsal prompt in the dashboard with a prompt from the 3.4 list. Check the first-token timer, the decode rate and the per-device buffers. | Answer streams |
| 9:30 | Freeze. No reloads, catalog edits, downloads, `git pull` or OS updates. Video device on and cued. | |
| 9:30 to 10:30 | Run the 3-minute script twice and the Q&A once. | |

The venue load writes a run.json with 1 measured rep. You can cite it as "this morning at the table, one rep". Never mix it into the 4 vs 4 numbers.

### 3.3 Commands

Host, on the GX10. The log holds a live invite: never put `host.log` on screen.

```sh
setsid nohup ~/.hypercompute/host.sh > ~/.hypercompute/host.log 2>&1 &
```

Fresh invite, one per peer, on the GX10. The output contains the ticket and code, so keep it off the projector.

```sh
cd ~/hypercompute && ~/.local/bin/uv run --no-sync hypercompute invite --port 8001
```

Mac peers (from `mac_join.sh`). Run from the peer checkout (`~/hc-peer`). For mac-b, change `--name`. For the Mac mini, use `[MAC MINI: cap]`; mac-azim was measured at 7.5 GiB.

```sh
caffeinate -dimsu uv run hypercompute join --ticket '<TICKET>' --code '<CODE>' --name mac-a \
  --max-memory 11.5GiB --reserve 0GiB --devices MTL0 --cache --yes --managed --max-disk 100GiB
```

G14 peer (from `join_g14.sh`). Run in `C:\Users\operator\.hypercompute\worktrees\live-host`, with `HYPERCOMPUTE_HOME`, `HYPERCOMPUTE_CONFIG_DIR` and `HYPERCOMPUTE_LLAMA_BIN_DIR` unset.

```sh
uv run --no-sync hypercompute join --ticket '<TICKET>' --code '<CODE>' --name g14 --devices CUDA0 \
  --max-memory 9GiB --yes --cache --managed --devices-allowed CUDA0 --max-disk 24GiB
```

Load and keep it running, on the GX10. This is the same workload as the measured runs, with 1 rep. `--node` takes each peer id, as in `matrix.py`.

```sh
cd ~/hypercompute
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export HYPERCOMPUTE_LLAMA_BIN_DIR=$HOME/.hypercompute/src/llama.cpp/build-hypercompute/bin
uv run --no-sync python scripts/capacity/run_big.py --model kimi-k2.6-iq1_s --port 8001 \
  --min-available-gib 0 --home ~/.hypercompute \
  --node <peer1> --node <peer2> --node <peer3> --node <peer4> --layout <pinned-layout.json> \
  --seed-peers --seed-timeout 3600 --prompt-tokens 64 --output-tokens 32 --warmup 1 --reps 1 \
  --load-timeout 3600 --keep
```

Warm-up and archive, on the GX10, from the repo:

```sh
.venv/bin/python scripts/capacity/paper/gx10/coherence.py 300 | tee ~/.hypercompute/coherence-venue.txt
```

### 3.4 Live prompts (use a new one for each judge)

The measured runs had prompt cache off. llama-server may reuse a cached prompt, so never repeat a prompt from the warm-up or from the last judge. Short prompts, short answers:

1. In two sentences, why does a mixture-of-experts model use only part of its weights for each token?
2. In one sentence, what does QUIC add over TCP?
3. Write a Python one-liner that reverses a string, then explain it in one sentence.
4. In one sentence, why is reading from an SSD slower than reading from memory?
5. Name two uses for a GPU besides graphics, one short sentence each.

### 3.5 Between judges (30 seconds)

- Machines page: all 5 machines eligible, fresh heartbeats.
- Deployment state `ready`.
- Watcher: paths `direct`, RTT in the tens of ms.
- Video device: charged and cued to 0:00.
- Next prompt picked from 3.4.

### 3.6 If something breaks during a slot

| Event | Say | Do |
|---|---|---|
| A peer goes stale or the deployment shows `failed` | "A laptop just dropped. The run fails on purpose, records why and releases memory. Here is the recording." | Play the video at the matching segment. Reload after the judge leaves (6 to 7 minutes warm). |
| No first token by 45 s (more than twice the measured 19.08 s) | Keep talking from the script. | Go to the recorded segment at 1:25 as planned. Check the watcher afterwards. |
| Dashboard or display dies | "Same thing, recorded at home." | Play the full 3-minute cut (6.2) and narrate over it. |

---

## 4. The 3-minute script (Tier A)

Spoken lines are in quotes. About 435 words, so read at a steady pace and do not rush the numbers. The 2:20 cost segment is the densest; rehearse it.

**Screen setup.**
- Main display: the dashboard. Chat on the left. Technical details on the right, showing per-device buffers (layers per peer), with the `status` watcher (paths, RTT) below them.
- Second device: the recorded video, cued.
- Never play the video on a peer Mac. Each one contributes up to 11.5 GiB of its 16 GiB with no reserve.

### 0:00 to 0:15. Hook

Screen: the table, and the dashboard Machines page with all five machines eligible.

> "This is Kimi K2.6, a one-trillion-parameter model. In the 1-bit quantization we run, it is 193 GiB. The biggest box on this table has 121.6 GiB of memory. So it does not fit in any one machine's memory."

### 0:15 to 0:40. What it is

Screen: scroll the Machines cards: OS, chip, contributed memory, verified.

> "Hypercompute is multiplayer computing. A friend runs one join command with a one-use code. Their machine joins over an encrypted Iroh QUIC tunnel, and it only counts after a real llama.cpp RPC handshake. Our planner places whole layers and MoE experts across the machines and the SSD. Each peer downloads its own slice from Hugging Face. That is Linux, Windows and macOS in one pool, on stock llama.cpp."

### 0:40 to 1:25. Live prompt

Do at 0:40: send a prompt from 3.4. Screen: chat with its first-token timer; the per-device buffers; the watcher.

> "I just sent a prompt to the pooled model. The four peers hold 11 layers. The GX10 holds the other 50 layers and memory-maps the remaining 83.9 GiB of experts from its SSD."

Read the layer split off the screen if the Mac mini changed it.

> "Every peer is on a direct path, and the round-trip times are right there. At home yesterday, with a MacBook Air in the fourth seat and a 64-token prompt, the first token averaged 19.1 seconds pooled and 28.3 alone over four measurements each."

When the first token appears:

> "There it is. The screen shows llama.cpp's own generation rate. We measured about 1.5 tokens a second pooled."

### 1:25 to 1:55. Recorded 671B split screen

Screen: second device plays clip V3 (section 6). Left: GX10 alone. Right: the pool. Both timers start at send.

> "This is DeepSeek V3.1, 671 billion parameters, recorded at home. Same machines and lengths: a 64-token prompt and a 32-token answer, GX10 alone against GX10 plus the G14 and three Macs. One Mac was on a relay path here. Left is the GX10 alone, right is the pool. Measured four times each, the first token came at 24.7 seconds alone and 10.4 seconds pooled. That is 2.4 times sooner. The full answer took 41.8 seconds against 31.2."

### 1:55 to 2:20. Why it helps

Screen: clip V4, the SSD and prompt-reading bars.

> "Here is why. Alone, the GX10 reads 684 MiB from its SSD for every generated token of the 671B, mostly expert weights. With four peers holding layers, it reads 37. That is 95% fewer reads. Prompt reading goes from 2.63 to 6.36 tokens a second."

### 2:20 to 2:45. The cost, said first

Screen: clip V5, the cost card.

> "Here is the cost. On this host, generation after the first token is slower pooled: 0.83 times on the 671B and 0.86 on the 1T. Each peer adds one network round trip per token. The clear generation wins so far are laptops only, a gaming laptop and MacBook Airs on a 120B, with a 128-token prompt and a 64-token answer: 1.27 times faster with three Macs over nine measurements each, and 1.25 with one. With three Macs the first token also came 2.7 times sooner."

Then one sweep line, if tonight's sweep produced a clean number. Otherwise skip it:

> "On the same 120B on a home network, the gaming laptop plus two MacBook Airs gave the first token 7.08 times sooner [6.20, 8.23], 34.2 to 4.83 seconds, and generated 2.28 times faster [2.00, 2.54], with a 128-token prompt and a 64-token answer, one run each with three measurements. The building Wi-Fi run with three Macs was 1.27 times [1.10, 1.47]; the two runs differ in network and number of Macs."

### 2:45 to 3:00. Close

Screen: back to the live chat, finished or still streaming.

> "So your friends' idle memory replaces SSD reads, and the first answer comes sooner. Every number here comes from a run file with its conditions. Happy to take questions."

---

## 5. Tier B and C (laptops only, no GX10)

**Tier B.**
- Host: the G14, started with `scripts/capacity/paper/gateway.ps1` (Iroh, port 8001).
- Peers: mac-a, mac-b, and the Mac mini or mac-azim, 4 layers each.
- Model: `gpt-oss-120b-mxfp4` (59.03 GiB), or tonight's best laptops-pool config: gpt-oss-120b-mxfp4 on G14 + mac-a + mac-b, 6 layers per Mac (`--tensor-split 6,6,25`), home network, pooled load 112.2 s after 761 s of seeding (PR R7.3). That is not the 3-Mac, 4-layer layout above, which was not run tonight.
- Blockers:
  - The G14 has only 23 GB free while the 671B file is on it, so the 120B must be re-downloaded and something moved first. That is your call.
  - Seed the Macs at home (566 s cold for three Macs).
- Load: 126 to 167 s pooled.
- Risk: R1 and R2 ran with the G14 as host and no GPU timeout was recorded, but a timeout under load is untested.

Swap these lines into the Tier A script:

| Segment | Tier B line |
|---|---|
| 0:00 hook | "This is gpt-oss-120b, 59 GiB. This gaming laptop has 31.1 GiB of RAM and a 12 GiB GPU. These MacBook Airs have 16 GiB each, and the Mac mini has [MAC MINI: memory]. It fits in no single machine's memory." If mac-azim takes the third seat, say "These three MacBook Airs have 16 GiB each." |
| 0:15 what it is | Same, but "Windows and macOS in one pool". |
| 0:40 live | "Each Mac holds 4 whole layers. Measured with three MacBook Airs on building Wi-Fi, 128-token prompt, 9 reps each: first token 13.4 s pooled, 36.2 s on the laptop alone." If the Mac mini is in, add "with a MacBook Air in that seat". |
| 1:25 recorded | Clip V3b, the 120B split screen. Or keep V3 (671B) and say "and on a bigger host:". |
| 1:55 why | "Alone, the laptop reads 275 MiB from its SSD per generated token, mostly expert weights. With three Macs, 41. Prompt reading goes from 3.68 to 11.03 tokens a second." |
| 2:20 cost | "Here generation is faster pooled too: 4.20 against 3.30 tokens a second, 1.27 times, because the laptop alone pages from SSD. With a big host like a GB10 box it flips: 0.83 times on the 671B. Each peer adds one round trip per token." |

**Tier C.** 120B on the G14 + 1 MacBook Air (mac-azim) vs the G14 alone. 128-token prompt, 64-token answer, 6 vs 9 reps, building Wi-Fi. Both 1-Mac runs (B1, B2) ran after 00:44 UTC, while a code workflow shared the G14.
- Load: 72 to 80 s.
- First token: 23.97 s against 36.17 s alone.
- Full answer: 38.81 s against 54.44 s.
- Decode: 1.25x [1.11, 1.41].

**Last resort.** Show the join flow live on a small model and call it a capacity demo, with no speed numbers. Then play the full video cut.

---

## 6. Fallback recorded video

### 6.1 Clips (record tonight, on the travel router if possible, otherwise home Wi-Fi as measured)

| Clip | Length | Content | On-screen caption |
|---|---|---|---|
| V1 join | 20 s | Invite, join command (**blur the ticket and code**), card goes from verifying to Ready to contribute, plan with layers per peer | "One-use code, expires in 15 min. Counts only after a real RPC handshake." |
| V2 live pooled answer | 40 s | Dashboard chat on Kimi pooled, first-token timer, decode rate | "Kimi K2.6 1T, GX10 + G14 + 3 Macs. Measured mean first token 19.08 s, 4 reps, 64-token prompt, MacBook Air in the fourth seat." |
| V3 671B split screen | 45 s | Same prompt, GX10 alone left and pool right, timers aligned at send | "This recording: `[VIDEO: alone s]` vs `[VIDEO: pooled s]`. Mean of 4 measured reps: 24.67 s vs 10.36 s. 64-token prompt, 32-token answer." |
| V3b 120B split screen (Tier B) | 45 s | G14 alone vs G14 + 3 Macs | "Mean of 9 reps: 36.17 s vs 13.44 s. 128-token prompt, 64-token answer." |
| V4 why chart | 15 s | Bars: SSD MiB per token 684 vs 37 (671B), 671 vs 254 (Kimi); prompt tok/s 2.63 vs 6.36, 2.19 vs 3.30 | Intervals from the numbers card |
| V5 cost card | 15 s | Decode 0.83x [0.77, 0.90] and 0.86x [0.82, 0.89]; laptops 120B 1.27x [1.10, 1.46]; laptops-only decode, home network, G14 + 2 Airs: 80B MoE that fits the host 0.46x [0.36, 0.62]; 70B dense that does not 5.44x [5.42, 5.47], one pooled row each | "One blocking round trip per peer per token" |
| V6 friend stops | 20 s | A Mac runs `hypercompute leave`, its card goes to left, the deployment goes to failed with the reason, leases released | "No live migration. The run fails cleanly." |
| V7 answer quality | 10 s | The archived coherence text (Fibonacci function) | "IQ1_S 1-bit quant, temperature 0, thinking off" |

Machine time for V3: 671B alone has a 239 to 256 s load. Pooled has a 503 to 517 s load, plus 1416 s of seeding on the first pooled run and 856 s on the second, which changed layout and re-seeded (Oct 24). Record the two halves one after the other, because the GX10 cannot hold both. Put this ahead of the concurrency rows, or record the dashboard during those rows' N=1 deployments if they use `--keep`.

### 6.2 Cuts and storage

- **Segment cut:** V1 to V7 as separate files, named by script time (`1m25-V3.mp4`) so you can jump straight to one.
- **Full cut:** under 3:00, following section 4 exactly, so you can narrate over it if the live pool is down from the start.
- **Format:** H.264 MP4, 1080p, no audio needed. Keep copies on the video device, one spare laptop and a USB stick. No streaming links. Test playback in airplane mode.
- **Last fallback:** a printed numbers card (section 1) and one PNG of V4 on a phone.

---

## 7. Judge Q&A (short, true answers)

**1. Is generation faster pooled?**
"Not on the GX10. With a 64-token prompt and a 32-token answer, it is 0.83 times on the 671B and 0.86 on the 1T, four measurements each. At batch size one, each peer is a blocking network round trip per token. We win before the first token: prompt reading is 2.42 and 1.51 times faster, and the first token comes 2.4 and 1.48 times sooner. The clear generation wins so far are laptops only: a gaming laptop plus MacBook Airs on a 120B, with a 128-token prompt and a 64-token answer. That was 1.27 times with three Macs and 1.25 with one, because the laptop alone was paging from SSD." Then add: "Tonight, on a home network with two MacBook Airs, decode was 5.44 times [5.42, 5.47] on Llama 70B and 2.59 [2.57, 2.62] on the 72B vision model's text path, with a 64-token prompt and a 32-token answer, and 2.28 [2.00, 2.54] on gpt-oss-120b with 128 and 64, one pooled run per model. On the 80B mixture-of-experts model that fits the laptop it was 0.46 [0.36, 0.62], slower, 128 and 64, two alone runs against one pooled." Add `[SWEEP: laptops-pool decode by model]` for the queued models once measured.

**2. What if a laptop leaves?**
"The run fails cleanly. There is no live migration. Heartbeats go every 3 seconds. After 15 seconds without one, the node goes stale, the deployment moves to failed with the reason recorded, and the memory leases are released. Then we start a new run without that friend. Friends keep local control: Ctrl+C, Stop, pause or leave. If the host disappears for 5 minutes, the friend's worker stops on its own." Reload cost: 367 to 369 s for Kimi pooled with warm caches.

**3. Is it private?**
"Not from the host. The host sees your prompt, and each peer sees the activations for its layers. Between machines, traffic goes over an encrypted, authenticated Iroh QUIC tunnel, and the RPC worker stays on loopback. Invites are one-use, expire in 15 minutes and are stored only as a keyed hash. The trust model is friends: the host can use the memory and GPU you contribute, but your agent only runs the allowlisted ggml-rpc-server."

**4. Why not just rent a GPU?**
"If one rented box holds your model, rent it. It will generate faster than our pool. Hypercompute is for when the model is bigger than the machine you have. On hardware we already owned, a gaming laptop plus three MacBook Airs took the 120B's first token from 36.2 seconds to 13.4, with a 128-token prompt, nine measurements each."

**5. Couldn't one machine run this alone?**
"Yes. With a 64-token prompt and a 32-token answer, the GX10 alone runs both models at about 1.8 tokens a second, and the gaming laptop alone runs the 671B at 0.48. What none of them can do is hold it in memory. On the 671B the GX10 alone reads 684 MiB from its SSD per generated token, mostly expert weights. With the pool it reads 37."

**6. Can it handle several people at once?**
"Only on the laptops so far. With gpt-oss-120b, the gaming laptop alone against the laptop plus two Airs on a home network, 128-token prompt, 64-token answer, one run each with three batches: with four chats at once the laptop alone served 8.10 tokens a second in total and the pool 11.85, 1.46 times [1.29, 1.68]. At eight chats, 12.19 and 16.44, 1.34 times [1.24, 1.48]. Each chat is slower than one chat alone. With the GX10 as host it was the other way: pooling slowed the 400B at every count of chats, and the 1T came out about even at eight. Every number we showed is one request at a time."

**7. Is a 1-bit model any good?**
"It is IQ1_S, 32B of its 1026B parameters active per token. We have not run a quality benchmark. Our check is coherence: at temperature 0 with thinking off, it wrote a correct iterative Fibonacci function with input validation." Show V7 or `[ARCHIVE: coherence output path]`.

**8. What did you build, and what is llama.cpp?**
"The data plane is stock llama.cpp b11770 RPC on every machine. We built:
- the join flow, a one-use invite plus a real RPC handshake before a peer counts;
- the Iroh QUIC tunnel through NAT;
- a planner that places whole layers and MoE experts across GPUs, CPU RAM and SSD;
- managed peers that fetch their own layers from Hugging Face, at 13 to 24 MB/s per Mac while seeding, against 8.6 MiB/s pushed from the host, on building Wi-Fi;
- a deployment lifecycle that verifies each peer really received its layers."

If asked what "stock" means: "Unmodified llama.cpp b11770 on every machine. The GX10 build is compiled from the release source for its GPU."

**9. Hasn't this been done before?**
"People have pooled machines for inference before. We don't claim to be first. What we can show is what we measured: Linux, Windows and macOS in one pool, stock llama.cpp, peers fetching their own layers, and 671B and 1T runs with confidence intervals." If a judge names a project: "I don't know that one well enough to compare."

**10. What about venue Wi-Fi?**
"We don't use it. On client-isolated Wi-Fi, Iroh falls back to a relay. We measured it on Oct 23: with the host on isolated guest Wi-Fi and the peers on a phone hotspot, traffic went over the relay at about 1 MiB/s, RTT 60 to 580 ms, and a peer dropped mid-load. So we bring our own router and preload before judging. A pooled start takes about 6 minutes with warm caches and 17 to 32 minutes cold."

The 6 minutes is Kimi with warm caches (5 s of seeding plus 367 to 369 s of load). The 17 to 32 minutes is cold: Kimi at 676 s of seeding plus 367 s of load, up to the 671B at 1416 s of seeding plus 503 s of load.

---

## 8. Never say (and what to say instead)

| Do not say | Why | Say instead |
|---|---|---|
| That pooling makes generation faster on the GX10 | 0.83x and 0.86x (PR R6) | "Faster prompt reading and first token. The clear generation wins are the laptops-only 120B: 1.27x with three Macs, 1.25x with one." |
| That it is a first | Others have pooled machines | "Here is the combination we measured." Name no other project. |
| That it can't run on one machine | The GX10 alone runs both models; the G14 alone runs the 671B | "It fits in no single machine's memory." |
| That it handles lots of people at once | Concurrency is being measured | "One request at a time so far. Concurrency is being measured." |
| That the inference is private | The host sees prompts and peers see activations | "Encrypted, authenticated tunnel. Friends-only trust." |
| "56 tok/s" | That is the GX10 alone on the 120B, a host change, not pooling (PR R6) | Do not use it. Say "If one rented box holds your model, rent it." |
| That the planner finds the best layout | It predicted pooled 1.77x ahead and measured 0.83x | Say nothing about its predictions |
| That it is fast on long documents | Prompts were 58 to 133 tokens | "Measured on short prompts." |
| Any number without its condition | | Host, peers, prompt and answer lengths, number of reps |

---

## 9. Placeholders to fill before judging

| Placeholder | Where | Fill from |
|---|---|---|
| `[SWEEP: per-model laptops-pool first token, full answer, decode vs host alone]` | Numbers card | Tonight's sweep run.json files (24B to 671B), via ttft_stats.py and stats.py |
| `[SWEEP: d70 laptops-pool decode]` and the other per-model decode rows | Numbers card, Q1 | Same |
| `[SWEEP: one sentence, best laptops-pool result]` | Script 2:20, V5 | Same. Cut it if the interval includes parity. |
| `[SWEEP: best laptops-pool model]` | Tier B | Same |
| `[CONCURRENCY: N=1/4/8 aggregate tok/s, config]` | Numbers card, Q6 | Tonight's `run_big --concurrency` rows |
| `[MAC MINI: memory, cap]`, `[MAC MINI: memory]` | 3.1, 3.3, Tier B hook | Mac mini setup |
| `[MAC MINI: Kimi pooled first token, 1 rep]` | 3.1 | The layout-pinning run tonight |
| `[VENUE ROUTER: path per peer, RTT range]` | 3.1 | The rehearsal on the travel router |
| `[VIDEO: alone s]`, `[VIDEO: pooled s]` | V3 caption | The recorded clip's own timers |
| `[ARCHIVE: coherence output path]` | 3.2, Q7 | `~/.hypercompute/coherence-venue.txt` or tonight's run |
