# [Convex] Best use of Convex ($5,000 cash, 1 winner)

Fit 6/10, prize about $5000, effort about 6 h (estimate).

## Integration

Build a "Pool Room": a public, live page where anyone with the room link (friends who give compute, judges on their phones) can watch the pool and use it. Convex holds the shared state and the prompt queue. The gateway on the GX10 only makes outbound HTTPS calls, so the owner token and port stay private and no inbound port is opened. This gives the product something it lacks today. Right now /v1/chat/completions needs the owner token from 127.0.0.1, so friends who lend their machines cannot use the model they are helping run. Convex does the real work here (transactional queue claims, leases with recovery, rate limits, presence, live fan-out of streamed tokens). It is not a display mirror.

CONVEX BACKEND (new folder room/convex/, TypeScript):
- schema.ts has three tables. rooms holds code, title, modelId, deploymentState and updatedAt, indexed by_code. machines holds roomId, nodeId, name, device (CUDA0/MTL0/CPU), layers, weightBytes, path (direct/relay/local) and state, indexed by_room. prompts holds roomId, author, sessionId, text, status (queued, running, done, failed), answer, tokens, decodeTps, prefillTps, leaseUntil, attempts and createdAt, indexed by_room_status.
- convex.config.ts runs app.use(presence) from @convex-dev/presence and app.use(rateLimiter) from @convex-dev/rate-limiter.
- Public functions:
  - prompts.submit checks the room code, caps text at 500 chars and the author name at 32, limits each sessionId with a token bucket via rateLimiter.limit(ctx, "prompt", {key: sessionId, throws: true}), and caps queue depth at 8.
  - prompts.list, machines.list and rooms.get are the live queries.
  - presence heartbeat/list/disconnect wrap the Presence class.
- Internal mutations:
  - bridge.claim takes the oldest queued prompt, marks it running with leaseUntil = now + 90 s, and calls ctx.scheduler.runAfter(90000, internal.bridge.expireLease, ...). If the gateway dies mid-answer, the prompt is requeued once and then marked failed.
  - bridge.append patches answer and extends the lease.
  - bridge.finish writes the measured tokens and tok/s.
  - bridge.release requeues a prompt the engine refused (429 busy).
  - bridge.syncPool upserts the machines rows.
- http.ts uses httpRouter routes POST /bridge/claim, /append, /finish, /release and /pool, served at https://NAME.convex.site. Each one checks Authorization: Bearer against process.env.HC_ROOM_SECRET, then calls ctx.runMutation(internal...).

GATEWAY SIDE (one new module plus about 5 lines in app.py):
- backend/hypercompute/services/room_bridge.py, about 180 lines, uses httpx (already a dependency, so no new Python package and no aarch64 wheel risk on the GX10). It does nothing unless HC_CONVEX_SITE_URL, HC_ROOM_SECRET and HC_ROOM_CODE are set. It runs two asyncio tasks.
- (a) pool_sync reads coord.events.stream() and reacts to node.* and deployment.* events, debounced to 1 s. It builds a snapshot from coord.nodes.views(), the active Deployment.plan.assignments (node_id, device, layers, weight_bytes) and Deployment.placement (observed llama.cpp offload), plus the Iroh path class from the existing iroh_path closure, which app.py passes in. It POSTs this to /bridge/pool.
- (b) prompt_worker runs only while coord.deployments.ready() succeeds. It claims up to N prompts, where N is the deployment's slot count, so it lines up with tonight's -np N concurrency work. Each claimed prompt runs in process through coord.inference.chat_stream({messages, stream: True, max_tokens: 48}). The output bytes are parsed with the existing SSEParser (backend/hypercompute/sse.py), deltas are flushed to /bridge/append every 250 ms, and /bridge/finish is called with the llama.cpp usage/timings from the last chunk. That keeps the repo's rule that rates come from engine timings and never from event counts. Room prompts also create normal run records, so they appear in the owner dashboard's RunsPanel.
- app.py lifespan: if RoomBridge.from_env(coord, iroh_path) returns a bridge, add its tasks to the tasks list and await bridge.aclose() in the finally block.
- tests/test_room_bridge.py uses httpx.MockTransport and a fake chat_stream to check four things: the claim, append, finish order; a 429 leads to release; nothing runs when the env vars are missing; the secret never appears in logs.

ROOM PAGE (new folder room/, Vite + React + the convex npm package, about 250 lines; the owner dashboard in frontend/ is untouched):
- useQuery(api.machines.list) drives the machine cards: device, layer count, GiB held, and a direct/relay badge.
- useQuery(api.prompts.list) drives the shared queue, where answers grow token by token as patches land.
- usePresence(api.presence, code, name) feeds a FacePile.
- useMutation(api.prompts.submit) sends prompts and shows retryAfter when the rate limit hits.
- A /present view for the projector shows a big QR code of the room URL.
- Host it on Vercel with VITE_CONVEX_URL set. Convex static hosting (@convex-dev/static-hosting) is optional, only if its INTEGRATION.md turns out to be quick.

DATA FLOW: a phone calls prompts.submit, Convex runs the rate limiter and writes the row, every subscribed screen updates, the GX10 bridge claims the prompt (an atomic mutation plus a scheduled lease check), llama.cpp RPC runs it across the GX10, the G14 and the Macs, the token deltas come back through /bridge/append, and the live query pushes them to every phone and the projector.

## Demo moment

A QR code is on the projector. A judge scans it, types a name, and their face shows up in the presence pile on the big screen at once. They type a question on their phone. It appears in the shared queue on every screen, and the GX10 claims it. The GX10 sits behind venue Wi-Fi with no open port. Kimi K2.6 1T then answers across the GX10, the G14 and 3 MacBooks, and the tokens stream onto the judge's phone and the projector at the same time. The answer card shows the measured decode tok/s from llama.cpp timings, and the machine cards show which device holds how many layers and whether each link is direct or on the relay. For the Convex beat, kill the gateway mid-answer. After the 90 s lease expires, the scheduled function requeues the prompt, the restarted gateway picks it up again, and the live page reflects every step. The line to say: "Your friends' laptops are a shared 1T model now, and anyone in the room can use it from a phone."

## Accounts and keys (the user creates these)

Complete account setup manually. Keep passwords out of the repository and automation tools.
(1) Create a free Convex account at dashboard.convex.dev (GitHub login). In room/, run npx convex dev, which logs in through the browser, creates the project and dev deployment, and writes VITE_CONVEX_URL (https://NAME.convex.cloud) to room/.env.local. Add room/.env.local and room/node_modules/ to .gitignore, since the current .gitignore does not cover them. The dev deployment is enough for judging. npx convex deploy and a CONVEX_DEPLOY_KEY are only needed if Vercel builds against a prod deployment.
(2) Generate a random HC_ROOM_SECRET and set it with npx convex env set HC_ROOM_SECRET.
(3) On the GX10, export HC_CONVEX_SITE_URL=https://NAME.convex.site, HC_ROOM_SECRET (the same value) and HC_ROOM_CODE (a short room code) in the shell or host.sh environment under ~/.hypercompute, never in the repo. host.log already holds a live invite, so the bridge must never print the secret either.
(4) Use the existing Vercel account to host room/ with VITE_CONVEX_URL set in the project env.
No paid plan is needed, and no Python Convex client is needed because httpx plus HTTP actions covers it.

## Risks

1. ELIGIBILITY, which affects every track: the hacker guide says "All projects must be started during the event" and says entries get disqualified for this every year. This repo's first commit is 2026-10-20, and the R1-R5 runs predate Friday Oct 23. The new Convex Pool Room does not cure that. Decide how to disclose what existed before the event before submitting anywhere.
2. Fit is moderate (6/10). The core product (llama.cpp RPC, planner, Iroh) is Python and does not run on Convex, so Convex is the multiplayer front door, not the whole backend. Teams that built entirely on Convex will look more native. The answer if a judge asks "why not FastAPI websockets": the gateway sits behind NAT, so it needs a public, durable, reactive surface with lease recovery that survives gateway restarts. Keep the lease-kill moment in the demo, because it is what proves Convex is doing work.
3. Time: about 6 of the remaining 15 hours, competing with overnight benchmarks, the concurrency commit, the Mac mini setup, the Devpost write-up and the other sidetracks. Cut order if late: static hosting, then the projector view, then lease recovery (keep the queue, presence and rate limiter).
4. Live speed: pooled Kimi decodes at about 1.5 tok/s with about 3.3 tok/s prefill, so a judge waits roughly 25-35 s. Cap room answers at 48 tokens and short prompts, and the visible streaming makes the wait part of the show. Fallback: gpt-oss-120b on the GX10 at 56 tok/s (not pooled, so say so).
5. Venue Wi-Fi: the bridge needs outbound HTTPS from the GX10 to convex.site. Test it on the venue network early.
6. Public surface: anyone with the link can run prompts and the text is shown publicly. The room code, token-bucket limit, 500-char cap, queue cap of 8 and an owner clear/close switch keep this in check. The HTTP-action secret check uses a plain string compare, which is acceptable for a demo but should be noted.
7. If the owner chats at the same moment, the single slot returns 429. The bridge must release and requeue, and that path needs its test.
8. SpacetimeDB is an anchor sponsor with a near-identical live-shared-state track (1st place $2,048). Wiring both into the same room would look bolted on, so pick Convex for the bigger prize.
9. This agent was read-only. Nothing here has been built or run, and the exact Presence and Rate Limiter APIs come from their READMEs and must be checked against the installed versions.

## Sources

- https://cal-hacks-13-0.devpost.com/
- https://docs.google.com/document/d/1RQrueC9wm5u8uxtWSC2gadIchp05hVayhGUrIjxT9dE/edit?tab=t.0
- https://docs.convex.dev/functions/http-actions
- https://docs.convex.dev/http-api
- https://docs.convex.dev/quickstart/python
- https://github.com/get-convex/convex-py
- https://www.convex.dev/components
- https://github.com/get-convex/presence
- https://github.com/get-convex/rate-limiter
- https://www.convex.dev/components/static-hosting
- [backend/hypercompute/app.py](../../../backend/hypercompute/app.py)
- [backend/hypercompute/services/events.py](../../../backend/hypercompute/services/events.py)
- [backend/hypercompute/services/streaming.py](../../../backend/hypercompute/services/streaming.py)
- [backend/hypercompute/sse.py](../../../backend/hypercompute/sse.py)
- [backend/hypercompute/schemas.py](../../../backend/hypercompute/schemas.py)
- [frontend/src/App.tsx](../../../frontend/src/App.tsx)
- [docs/research/CHECKPOINT-2026-10-24.md](../../../docs/research/CHECKPOINT-2026-10-24.md)
- [docs/pitch/DEMO-SCRIPT.md](../../../docs/pitch/DEMO-SCRIPT.md)
