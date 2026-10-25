# Pool Room (Convex)

A live room for the Hypercompute pool. Anyone with the room link (friends who lend machines, judges on their phones)
sees who is in the room, which machine holds which layers of the served model, and every prompt streaming in, and
can send the pool a prompt of their own. The projector view adds a big QR code so people can join by scanning it.

Without this, only the owner can use the model: `/v1/chat/completions` needs the owner token, which the gateway hands
out on 127.0.0.1 only. The room gives everyone else a way in without opening a port on the host or sharing the token.

```
phone ──prompts.submit──▶ Convex ◀──POST /bridge/claim, /append, /finish (outbound HTTPS)── bridge ──loopback──▶ gateway
phone ◀──live queries──── Convex                                                          (next to the gateway, behind NAT)
projector ◀──live queries─┘
```

## What is in this folder

| Path | What |
|---|---|
| `convex/` | Convex backend: schema, the queue with leases, rate limits, presence, the bridge's HTTP routes |
| `src/` | React client: phone page (`?room=CODE`) and projector view (`?room=CODE&view=present`) |
| `bridge/room_bridge/` | Python bridge that runs next to the gateway (httpx only, already a gateway dependency) |
| `bridge/mock_gateway.py` | Stand-in for the gateway's owner API and streamed chat, with fault switches, for development |
| `tests/` | Convex function tests (vitest + convex-test) |
| `bridge/tests/` | Bridge tests (pytest, httpx.MockTransport and the mock gateway) |

## Setup

### 1. Convex (done once, by the account owner)

```bash
cd integrations/convex-room
npm install
npx convex dev --once          # logs in through the browser, creates the project, writes .env.local
```

Then set the room secret on the deployment. Generate a random value and keep it only in git-ignored files:

```bash
npx convex env set HC_ROOM_SECRET <random 32-byte value>
```

`npx convex env set HC_LEASE_MS 30000` shortens the 90 s lease for rehearsals (5 s to 10 min accepted).

No browser login on this machine? A Development deploy key from the Convex dashboard works too: put
`CONVEX_DEPLOY_KEY=...` in `.env.local` (git-ignored) instead of the `CONVEX_DEPLOYMENT` line, then run the same
`npx convex dev --once` and `npx convex env set` commands. Point `VITE_CONVEX_URL` and the bridge's
`HC_CONVEX_SITE_URL` at that key's deployment, and never paste the key into a shell history or a log.

### 2. The bridge, on the host that runs the gateway

Copy `bridge.env.example` to a git-ignored file outside the repo (for example `~/.hypercompute/room.env`, `chmod 600`)
and fill in `HC_CONVEX_SITE_URL=https://NAME.convex.site`, the same `HC_ROOM_SECRET`, and `HC_ROOM_CODE`. Then, with
the host's venv (it already has httpx and the `hypercompute` package):

```bash
cd integrations/convex-room/bridge
python -m room_bridge --env-file ~/.hypercompute/room.env check   # reads the pool, registers the room, prints it
python -m room_bridge --env-file ~/.hypercompute/room.env run     # claims prompts while the gateway is ready
```

Owner switches: `python -m room_bridge --env-file ... close` stops new prompts, `open` reopens, `clear` removes
everything that is not running. Ctrl-C hands any prompt in flight back to the queue.

The bridge makes only outbound calls: HTTPS to `NAME.convex.site`, and loopback HTTP to the gateway. It refuses a
gateway URL that is not 127.0.0.1/localhost and a Convex URL that is not https (except a local backend). It never
logs the room secret or the owner token, and its httpx client for the gateway ignores proxy settings.

### 3. The web page

```bash
npm run dev                      # http://localhost:5178/?room=calhacks  and  ...&view=present
npm run build                    # static files in dist/
npm run dev:lan                  # same, reachable from phones on the same Wi-Fi
```

For phones on another network, host `dist/` anywhere static (Vercel works) with `VITE_CONVEX_URL` set, and set
`VITE_PUBLIC_URL` to that address so the projector's QR code points there. Nothing is deployed publicly yet.

### Developing without a real host

```bash
python bridge/mock_gateway.py --port 8799 --decode-tps 4
HC_GATEWAY_URL=http://127.0.0.1:8799 HC_ROOM_CODE=mockroom python -m room_bridge --env-file ../bridge.env run
curl -X POST http://127.0.0.1:8799/mock/faults -d '{"busy_next":1}'    # next chat refused as busy
curl -X POST http://127.0.0.1:8799/mock/faults -d '{"cut_after":4}'    # next chat dies after 4 tokens
curl -X POST http://127.0.0.1:8799/mock/faults -d '{"down":true}'      # no ready model
```

Everything the mock reports is made up, and its model id starts with `mock-` so a screenshot cannot pass for a
measurement.

## Demo beats

The full table-demo script, setup checklist and failure drill are in [DEMO.md](DEMO.md).


1. The projector shows the QR code. A judge scans it, types a name, and their face appears in the pile on the big
   screen.
2. They ask something. The prompt shows up in the shared queue on every screen; the host claims it; the answer streams
   onto their phone and the projector at the same time. The layer ribbon pulses once per token across every machine.
   The finished card shows tokens, decode and prefill tok/s from the engine's own timings.
3. The Convex beat: kill the bridge (or pull the host's network) mid-answer. The card turns to "host quiet" with a
   countdown, the header says host offline. When the lease runs out, a scheduled Convex function puts the prompt back
   in the queue ("the host went quiet and its lease expired; requeued"). Restart the bridge and it claims the prompt
   again ("retry, attempt 2 of 2") and finishes it. On the mock, cutting the gateway mid-answer shows the other path:
   the bridge sees the broken stream and requeues at once ("the gateway connection dropped; requeued"). Do not do
   that to a real gateway at the table: it would have to reload the model.

## Which Convex features are used, and why

- **Transactional mutations as a work queue.** `bridge.claim` reads the oldest queued prompt and marks it running in
  one transaction, so two bridges (or two slots of one bridge) can never take the same prompt. A refused submit throws
  and rolls back, including the rate-limit tokens it would have spent.
- **Scheduled functions for leases.** Each claim arms `bridge.expireLease` for when its lease ends. Every append renews
  the lease; the scheduled call re-arms itself if the lease moved, and otherwise requeues the prompt once, then marks it
  failed. This is what survives a dead host: nothing on the host has to run for the recovery to happen.
- **Fencing with `claimSeq`.** Every claim bumps it and every bridge write must carry it, so a bridge that comes back
  after its prompt was requeued cannot write over the new attempt.
- **Idempotent writes.** Appends carry the character offset they start at, a claim carries a `claimKey`, and a retried
  finish, release or fail whose first try landed answers ok. A reply lost on venue Wi-Fi never doubles text or takes
  a second prompt.
- **An activity log written in the same transactions** (`events` table): claims, lease expiries, requeues and refused
  stale writes appear on the projector exactly when they commit, and never for a change that rolled back.
- **HTTP actions** (`convex/http.ts`): the host talks to Convex with plain HTTPS POSTs carrying a bearer secret, so it
  needs no inbound port, no WebSocket client and no new Python package. The secret check hashes both sides with SHA-256
  and compares in constant time.
- **Reactive queries.** `rooms.get`, `machines.list` and `prompts.list` take only the room code, so every viewer shares
  one cached subscription and each answer patch reaches every phone and the projector in the same push.
- **`@convex-dev/rate-limiter`**: a token bucket per phone session (3 at once, then one per 30 s), one for the whole
  room (so fresh session ids cannot flood the host), and one for presence heartbeats.
- **`@convex-dev/presence`**: the face pile. Heartbeats only, no polling; people drop off when their tab closes or goes
  quiet. New faces spend from a room-wide bucket and a room holds at most 60 online, so rotating names cannot flood
  the projector.
- **Validators** on every argument, plus server-side caps: 500 characters per prompt, 32 per name, 8 prompts queued or
  running per room, 6000 characters per answer.

## Tests

```bash
npx vitest run              # 37 tests: caps, rate limits, queue cap, claim order, lease expiry and requeue, lease
                            # renewal, fencing, busy release, retried writes, claim keys, presence abuse, HTTP auth
                            # and status codes, pool sync, the activity log
cd bridge && python -m pytest   # 36 tests: claim/append/finish order and offsets, 429 and SSE busy release, gateway
                            # killed mid-answer, stale claim, keepalive during prefill and with a short lease,
                            # shutdown release, unexpected errors, retried appends and claims, secret and owner
                            # token never logged, config and env-file checks
```

## Known limits

- The room is public to anyone with the link, by design. Names, prompts and answers are visible to everyone in it.
  The room code, the rate limits, the input caps, the queue cap and the owner's close/clear switches are the controls.
- The phone session id is generated by the browser, so a determined person can mint new ones. The room-wide bucket
  bounds what that buys them; it does not stop them from filling the room's share.
- No accounts: a name is whatever someone types. The presence `disconnect` mutation takes only the session token, as
  the component's README recommends for `sendBeacon`.
- Lease expiry takes up to the lease length (90 s by default) after the host goes silent. Kimi K2.6 on the pool
  decodes at about 1.5 tok/s, so answers are capped at 48 tokens and a judge waits roughly 25 to 35 s.
- "Live tok/s" on a running card comes from the engine's per-token `timings` (`timings_per_token`). An engine that does
  not send them shows the rate only when the answer finishes.
- A phone that locks its screen stops heartbeating and drops out of the face pile after about 25 s; it reappears when
  unlocked.
- One bridge per room is the intended setup. Two would share the queue correctly, but the machine cards show whichever
  bridge reported last.
- If the owner chats through the dashboard while the room has the only slot, the engine answers busy; the bridge hands
  the prompt back without counting it as an attempt and retries 2 s later.
