# Spacetime: Best Use of Spacetime (Cal Hacks 13.0), 1st $2,048, 2nd $1,024, 3rd $512. The separate Scrabblebot tournament is not relevant here.

Fit 6/10, prize about $2048, effort about 6.5 h (estimate).

## Integration

Verdict first. The fit is medium, and it is only real if SpacetimeDB becomes the front door for people while Iroh stays the back door for machines. Today only the owner can use the pool. /v1/chat/completions and every /api route need the owner token, so a friend who lends a Mac cannot send a single prompt. Putting the compute control plane itself (sessions, invites, leases, placement in SQLite via storage.py) into SpacetimeDB would be a rewrite of security-sensitive code with 15 hours left. That is not worth it. Build the multiplayer layer, and nothing more.

The smallest integration that does real work is "Pool Room". It is a SpacetimeDB database that owns the shared prompt queue, fair-share scheduling, job leases and the live pool board. It is the only way a non-owner reaches the pooled model.

(1) New SpacetimeDB TypeScript module at spacetime/hc-room/src/index.ts (spacetimedb/server: schema, table, t). It has these tables:
- room_config: admin_hex, saved in the init reducer, where ctx.sender is the publisher. It also holds worker_hex, paused and the caps.
- peer: node_id PK, display_name, platform, offered_gib, layer_units, path (direct or relay), heartbeat_age_s, seed_pct, state.
- pool_status: one row with model_id, deployment state and progress, slots, last_sync.
- job: id u64 autoInc PK, submitter Identity, prompt, status (queued, running, done or failed), slot, lease_until, attempts, output, tokens, ttft_ms, tok_s, run_id.
- reap_tick: a schedule table.

These are the reducers:
- submit_prompt(prompt) for anyone. It enforces at most 400 chars, one active job per identity, a queue cap of 30, and the room's paused flag.
- claim_jobs(free_slots, lease_s). Only the worker may call it (ctx.sender.toHexString() equals worker_hex). It picks queued jobs fairly, round robin by submitter's last-served time and then FIFO, inside one transaction, so two slots never get the same job.
- append_output(job_id, delta, tokens). Worker only. It also renews the lease.
- finish_job(job_id, status, ttft_ms, tok_s, run_id). Worker only.
- sync_pool(peers_json, status_json). Worker only. Each call upserts and deletes peer rows in one transaction.
- set_worker(hex), pause_room(bool) and delete_job(id). Admin only.
- reap, a scheduled reducer every 2 s. It requeues running jobs whose lease expired, or fails them after 2 attempts, and marks the gateway offline if last_sync is more than 10 s old. Lease liveness is therefore decided inside SpacetimeDB, next to the data.

Export reducers with snake_case names. The CLI converts sayHello to say_hello, and snake_case exports avoid that ambiguity over HTTP.

(2) New backend/hypercompute/services/room_bridge.py, about 200 lines. It uses httpx, which is already a dependency, because there is no Python SDK.
- The sync loop runs every 2 s. It builds peers from coord.nodes.views(), coord.nodes.managed.snapshot() and the ready deployment's plan.assignments. It then calls POST {HC_SPACETIME_URI}/v1/database/{HC_SPACETIME_DB}/call/sync_pool with a JSON array body and Authorization Bearer HC_SPACETIME_TOKEN. It syncs right away on node.joined, node.left and node.state, read from EventBus.since().
- The job loop runs every 400 ms. It calls claim_jobs with the number of free slots. The slot count is InferenceService capacity, that is adapter.max_concurrency / parallel_slots, from the uncommitted -np N -kvu feature. The loop reads its claimed rows with POST .../sql "SELECT * FROM job WHERE status = 'running'". Each job runs in process through coord.inference.chat_stream({messages, stream: true, max_tokens: 48}), so the owner token never leaves the host and every room job becomes a normal run record with TTFT. The loop parses deltas with the existing sse.SSEParser and batches append_output every 250 ms with ?confirmed=false. It ends with finish_job, or on an interrupted error it calls finish_job with failed.

(3) Config and app wiring.
- backend/hypercompute/config.py reads HC_SPACETIME_URI, HC_SPACETIME_DB and HC_SPACETIME_TOKEN. If they are absent, the bridge is off. If they are set partially, startup fails with a clear message, and the token is never logged.
- The app.py lifespan starts the two bridge tasks next to the existing transport tasks.

(4) New public page, frontend/room.html plus src/room/{main.tsx, RoomApp.tsx, PoolBoard.tsx, Queue.tsx}.
- vite.config.ts gets a second rollup input. package.json adds spacetimedb, pinned to the CLI's minor version. 2.11.0 is the newest that is at least two weeks old, and 2.13.0 is only 3 days old.
- Bindings come from spacetime generate.
- The page uses SpacetimeDBProvider with DbConnection.builder().withUri().withDatabaseName(). It reads useTable(job) and useTable(peer) and calls useReducer(submit_prompt).
- It never touches the gateway, and the static build is hosted publicly so phones can scan a QR code.

(5) tests/test_room_bridge.py uses httpx.MockTransport and a fake inference stream. It covers claim then stream then append then finish, lease renewal, the interrupted path, the bridge being disabled without env, and the token not appearing in logs.

Data flow. A phone with an anonymous identity holds a WebSocket subscription. It calls submit_prompt, every screen sees the queue update, and the gateway claims the job. The pooled llama.cpp RPC across GX10, G14 and Macs serves it. Deltas go back through append_output and fan out to all subscribers. The pool board updates from sync_pool.

## Demo moment

The projector shows the Pool Room. There is a live card for each machine (GX10, G14, mac-a, mac-b, Mac mini) with its layer bar, direct or relay path, heartbeat and seed progress, plus a QR code. Judges scan it and type prompts on their phones. The queue fills on every screen at once. The claim_jobs reducer hands jobs fairly to the N parallel slots, so three judges' answers stream side by side on everyone's phone. Each job card shows TTFT and tok/s and links to a run record.

Line to say: "Anyone who lends a machine gets a seat at the model. SpacetimeDB is the front door for people, Iroh is the back door for machines."

For the 1T beat, a judge's Kimi K2.6 prompt lands its first token on their own phone after about 19 s pooled. That is the measured 19.08 s TT number with all peers direct.

For a resilience beat, stop the bridge mid-answer. Within 2 s the scheduled reap reducer moves the job back to queued on every screen, and the job resumes when the bridge restarts.

Use gpt-oss-120b for the snappy multi-user part, since Kimi decodes at about 1.5 tok/s shared across slots.

## Accounts and keys (the user creates these)

All of these steps are done by the user. Claude never signs up or handles passwords.

1. Install the spacetime CLI from spacetimedb.com/install by running the installer yourself.
2. Run `spacetime login`, which signs in through a browser with GitHub or Google. This creates the SpacetimeDB and Maincloud account.
3. Run `spacetime publish hc-room --server maincloud`. The publisher identity becomes room admin through init.
4. Mint a worker identity with `curl -X POST https://maincloud.spacetimedb.com/v1/identity`. Put the returned token only in HC_SPACETIME_TOKEN on the gateway host, never in the repo. Then run `spacetime call --server maincloud hc-room set_worker <identity_hex>`.
5. Gateway env vars are HC_SPACETIME_URI=https://maincloud.spacetimedb.com, HC_SPACETIME_DB=hc-room and HC_SPACETIME_TOKEN, which is secret.
6. Frontend env vars are VITE_SPACETIME_URI and VITE_SPACETIME_DB. Neither is a secret.
7. Optionally, use a static host account to put the room page on the public internet for the QR code. Vercel works, and the user already deploys there.

No paid plan is needed. Maincloud Free auto-pauses idle databases and resumes in under 1 s.

## Risks

1. Fit perception. SpacetimeDB does not run the compute plane, which stays in SQLite and the coordinator. Spacetime judges may read it as a lobby added to an inference project. Mitigation: the room is the only non-owner path to the model, it holds the queue, the fair scheduling and the lease reaping, and the live demo runs through it.

2. No Python SDK. The worker side uses the HTTP reducer API plus about 400 ms SQL polling to claim jobs, and only viewers get true subscriptions. SATS-JSON quirks (u64, Timestamp, Identity encodings in SQL rows) can bite. Keep the fields the worker reads as strings or u32, and compare identities by hex string.

3. Version drift. npm spacetimedb 2.13.0 shipped Oct 21. Pin the module, the client and the generated bindings to the installed CLI's minor version.

4. Demo dependency on venue internet reaching Maincloud. The fallback is `spacetime start` on the presenter laptop (port 3000) with the room on the LAN, if venue Wi-Fi allows client-to-client traffic.

5. Wiring the bridge into the GX10 gateway needs a host restart. That would interrupt the overnight concurrency and sweep matrix, so develop against a local G14 gateway with qwen2.5-0.5b and restart the GX10 only after the matrix finishes. The bridge is off unless the env vars are set.

6. Slow 1T decode, about 1.5 tok/s split across slots, makes multi-user streams crawl. Use gpt-oss-120b for the interactive part. With 3 to 4 parallel slots on Kimi, also re-check the GX10's 79 GiB GPU budget for -kvu KV.

7. A public room means prompts are visible to everyone and open to abuse. Mitigate with caps (400 chars, 48 tokens, one active job per identity, queue of 30) and the admin pause_room and delete_job reducers.

8. Time. About 6.5 of the remaining roughly 15 hours, competing with pitch prep, the Devpost video and the main Most Technically Complex story. A cut-down MVP (queue plus streams plus the peer board, skipping reap and priority) is about 4.5 h.

9. Devpost says 3 minutes to pitch plus 2 for questions, not the 2-minute pitch in the task. It also requires a demo video for async sponsor judging, and the hard submission deadline is 9:00 AM PDT Oct 25.

## Sources

- https://cal-hacks-13-0.devpost.com/ (prize list: [Spacetime] Best Use of Spacetime 1st $2048, 2nd $1,024, 3rd $512; requirements and judging criteria)
- https://docs.google.com/document/d/1RQrueC9wm5u8uxtWSC2gadIchp05hVayhGUrIjxT9dE/edit?tab=t.0 (sponsor track doc: SpacetimeDB as core real-time backend, live shared state, multiplayer, live dashboards, AI agent coordination, must be meaningfully used, not just added on the side)
- https://spacetimedb.com/blog/spacetime-at-calhacks
- https://spacetimedb.com/docs/
- https://spacetimedb.com/docs/quickstarts/react
- https://spacetimedb.com/docs/clients/typescript
- https://spacetimedb.com/docs/http/database (POST /v1/database/:name/call/:reducer, POST /v1/database/:name/sql)
- https://spacetimedb.com/docs/http/identity (POST /v1/identity)
- https://spacetimedb.com/docs/tables/schedule-tables
- https://spacetimedb.com/docs/functions/reducers/lifecycle (init sender is the publisher)
- https://spacetimedb.com/docs/how-to/deploy/maincloud
- npm registry: spacetimedb versions (2.11.0 Oct 7, 2.12.0 Oct 13, 2.13.0 Oct 21 2026)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\app.py (owner-only /api and /v1/chat/completions, lifespan)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\events.py (EventBus)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\services\streaming.py (InferenceSlots, chat_stream)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\adapters\llama_rpc.py (parallel_slots, -np N -kvu, uncommitted)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\sse.py (SSEParser)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\backend\hypercompute\storage.py
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\pitch\DEMO-SCRIPT.md (Kimi 1T TTFT 19.08 s pooled, 120B timings)
- C:\Users\operator\.hypercompute\worktrees\iroh-transport\docs\research\CHECKPOINT-2026-10-24.md
