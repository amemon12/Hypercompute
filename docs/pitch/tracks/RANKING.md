# Hypercompute at Cal Hacks 13.0: track ranking and build plan

**Step 0, before any build (15 min).** The rules say all implementation must happen during the event. The repo dates from Oct 20, and runs R1 to R5 happened before the event. Ask the info desk how to disclose the pre-event work. If it is ruled ineligible, every track below is worth zero. In the same visit, confirm three things: that there is no cap on tracks per submission (none was found in the guide, on Hive or on Devpost), which sponsors need their own opt-in forms, and whether the pitch is 2 minutes or Devpost's 3 plus 2 for questions.

## Ranking by expected value

The win odds are my estimates, not data.

| Rank | Track | Prize | Fit | Est. odds | EV | Build |
|---|---|---|---|---|---|---|
| 1 | Most Technically Complex | general | 8 | ~20% | high, zero build | 0 h |
| 2 | Convex Best Use | $5,000 | 6 | ~15% | ~$750 | 6 h (4.5 h cut) |
| 3 | Factory Droid SDK | $2,000 | 7 | ~22% | ~$440 | 6 h (4 h cut) |
| 4 | Spacetime | $2,048 / 1,024 / 512 | 6 | ~15% top 3 | ~$450 | 6.5 h |
| 5 | Anthropic Public Benefit | $4,998 credits | 5 | ~8% | ~$400 credits | 6 h |
| 6 | x402 Coinbase | $500 per member | 6.5 | ~20% | $100 to $400 | 5 h |
| 7 | Armadin Security | ~$1,556 | 4 now, 6.5 with guard | ~12% | ~$190 | 7 h |
| 8 | Meta, Hardware, Planet-First | low | 2 to 3 | low | ~0 | n/a |

## Recommended build set (about 9 to 10 h)

**1. Convex Pool Room, 5:00 PM to 11:00 PM (6 h).**
- Build it against a local G14 gateway running qwen2.5-0.5b, so the GX10 benchmark matrix is not interrupted.
- If time runs short, cut in this order: static hosting, then the projector view, then lease recovery. The queue, presence and rate limiter stay.
- What it adds to the pitch (about 40 s): judges scan a QR code and their faces appear on the projector. A judge's prompt streams onto their own phone, and the gateway sits behind NAT with no open port. Then kill the gateway mid-answer: when the 90 s lease expires the prompt is requeued and finishes. That last beat proves Convex is doing real work. It also gives the Technically Complex pitch a human hook: the friends lending machines can finally use the model.

**2. Factory Droid autotune, minimal cut, 11:00 PM to 3:00 AM (4 h).**
- Skip the approval UI and auto-approve the spec-mode plan. Keep the plan-only `check_layout` tools, the 8-call cap, the layout diff and the `droid.step` events.
- Rehearse on the R2 setup (G14 plus 1 Mac on 120B, which measured 1.25x), so there is a clear "apply" moment.
- What it adds (about 20 s): a Mac joins, Droid proposes layouts, and the planner refuses one with a byte-level reason. The owner clicks Apply. Line to say: "A join is a reason to evaluate, not an order to use every device."

**3. GX10 cutover, about 4:00 to 5:00 AM.**
- Once the matrix finishes, restart the GX10 gateway with the bridge env vars set.
- Test outbound HTTPS from the GX10 to `NAME.convex.site` on the venue Wi-Fi.
- Record backup clips of both demo beats.

**4. Devpost write-up, demo video and README, 5:00 to 8:30 AM.**
- The README needs a "Where Droid is used" section, which the Factory track requires.
- Opt into Most Technically Complex, Convex and Factory.

**Core pitch (about 60 s):** a 1T model running across the GB10, an RTX laptop and MacBook Airs over Iroh QUIC. The planner matches the brute-force oracle, and the decode rates in R6 are measured, not predicted.

## Accounts and keys the user must create first

The user does all of this. Claude signs up for nothing and handles no passwords. Keys stay out of the repo.

**Convex**
1. Create an account at dashboard.convex.dev with GitHub login.
2. In `room/`, run `npx convex dev`. It writes `VITE_CONVEX_URL` to `room/.env.local`. Add `room/.env.local` and `room/node_modules/` to `.gitignore`.
3. Generate a random `HC_ROOM_SECRET` and set it with `npx convex env set HC_ROOM_SECRET`.
4. On the GX10, export `HC_CONVEX_SITE_URL=https://NAME.convex.site`, `HC_ROOM_SECRET` (the same value) and `HC_ROOM_CODE` in the environment under `~/.hypercompute`.
5. On the existing Vercel account, set `VITE_CONVEX_URL` in the project env for `room/`.

**Factory**
1. Create an account at app.factory.ai and ask the booth for hackathon credits.
2. Create an API key at app.factory.ai/settings/api-keys. Put `FACTORY_API_KEY` in `~/.hypercompute/droid.env` with chmod 600, sourced by host.sh.
3. Install the droid CLI on the host yourself, then check that `droid --version` runs on arm64.
4. Run `uv pip install "droid-sdk[mcp]==0.5.0"` in the host venv.
5. Set `HYPERCOMPUTE_DROID=1` in host.sh.

## Tracks to skip

- **Spacetime.** It is nearly the same as the Convex room, and wiring both into one room looks bolted on. Its top prize is also smaller. Pick one, and Convex wins.
- **Anthropic Public Benefit.** Fit is 5. The product works without Claude, and the target community has no interviews on record yet. Revisit only if 2 to 3 real student teams are interviewed tonight.
- **x402.** A paywall in front of an LLM is the stock x402 demo. The middleware buffers the whole reply, so paid calls cannot stream. The Iroh-mode gateway binds to loopback only, and `x402[evm]` has to install on aarch64. With a solo team the prize is only $500. Treat it as a stretch only if the build set finishes early.
- **Armadin.** The criteria are still "Coming soon." The build is 7 h, the RPC firewall has to parse frames on the hot path, and the CVE patch status is inferred, not tested. The Iroh, invite and credential protections can still go in the write-up for free.
- **Meta.** The machines connect, not the people. Selling this infra as a social product would be bolted on.
- **Best Hardware.** Nothing physical was built.
- **Planet-First.** Pooling probably uses more energy per token than the GX10 alone (0.83x decode with extra laptops drawing power), and the GB10 and Mac power sensors are partly unavailable.
- **First Overall.** It is a long shot. Do not build anything specifically for it.
