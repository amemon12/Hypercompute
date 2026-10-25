# Pool Room: 2-minute table demo

Everything runs from `integrations/convex-room/`. The host is the machine that runs the Hypercompute gateway.

## Setup (once, before judges arrive)

1. **Short lease for the table.** The default lease is 90 s; a judge will not wait that long.
   ```bash
   npx convex env set HC_LEASE_MS 20000        # back to 90 s afterwards: npx convex env remove HC_LEASE_MS
   ```
2. **Bridge on the host.** Keep the env file off the repo: `~/.hypercompute/room.env`, `chmod 600`, values from
   `bridge.env` (`HC_CONVEX_SITE_URL`, `HC_ROOM_SECRET`, `HC_ROOM_CODE=calhacks`, `HC_GATEWAY_URL`).
   ```bash
   cd integrations/convex-room/bridge
   python -m room_bridge --env-file ~/.hypercompute/room.env check    # prints the pool, registers the room
   python -m room_bridge --env-file ~/.hypercompute/room.env clear    # empty queue and log for the first judge
   python -m room_bridge --env-file ~/.hypercompute/room.env run      # leave running; it claims only while ready
   ```
   `check` must print the model as `ready` and `convex ... room calhacks open`. No outbound HTTPS to
   `*.convex.site` from the venue network means no room; test that first.
3. **A page phones can reach.** The QR code encodes `VITE_PUBLIC_URL` (else the page's own address). Pick one:
   - Vercel, with `VITE_CONVEX_URL` and `VITE_PUBLIC_URL` set (needs the user's yes, it is public).
   - The demo laptop: put `VITE_PUBLIC_URL=http://<laptop LAN IP>:5178` in `.env.local`, then `npm run dev:lan`, if
     the venue Wi-Fi lets phones reach other clients. A phone hotspot shared by the laptop and the judges' phones
     always works.

   The projector shows a red warning under the QR code if it still points at localhost.
4. **Projector:** open `<page>/?room=calhacks&view=present`, full screen. **Your phone:** open `<page>/?room=calhacks`,
   join, send one prompt, watch it finish. Then `clear` again.

## The two minutes

| Time | Do | Say |
|---|---|---|
| 0:00 | Point at the ribbon and machine cards | "One model, split layer by layer across these machines. Every token walks through all of them." |
| 0:15 | Judge scans the QR code, types a name | "Your face is on the projector now: Convex presence." |
| 0:30 | Judge taps an idea chip, sends | "It lands in a shared queue in Convex. The host sits behind NAT with no open port; it claims the prompt over outbound HTTPS, in one transaction, with a lease." |
| 0:45 | Answer streams on the phone and the projector | "Every token is a Convex write that every screen gets at once. The tok/s is llama.cpp's own timing." |
| 1:05 | **Failure drill**: freeze the host's bridge (below) during the next answer | "Now the host disappears mid-answer." |
| 1:10 | Card turns red, "host quiet", countdown | "Nothing on the host has to run for recovery. The lease lives in Convex." |
| 1:25 | Log: "lease expired; requeued" | "A scheduled Convex function expired the lease and put the prompt back." |
| 1:30 | Unfreeze the bridge | "The host comes back. Its stale write is refused, it reclaims the prompt as attempt 2, and finishes." |
| 1:50 | Point at the Convex log | "Transactional claims, leases, scheduled requeue, fencing, rate limits, presence: Convex is the multiplayer layer for a GPU pool behind NAT." |

### Failure drill commands

On a Linux or macOS host (the GX10), freeze and thaw the bridge. The process stays alive, which also shows fencing:
```bash
pkill -STOP -f room_bridge     # host goes silent; the card turns "host quiet" after about 6 s
pkill -CONT -f room_bridge     # after the log says "requeued": its late write is refused, then it reclaims
```
On a Windows host, hard-kill it and start it again (Ctrl-C would hand the prompt back at once, skipping the lease):
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? CommandLine -match 'room_bridge' | % { Stop-Process -Id $_.ProcessId -Force }
python -m room_bridge --env-file $HOME\.hypercompute\room.env run
```
**Do not kill the real gateway in front of judges.** The bridge would requeue at once ("the gateway connection
dropped"), but the gateway then has to reload the model, which takes minutes for a 1T model. Rehearse that path on the
mock only: `curl -X POST http://127.0.0.1:8799/mock/faults -d '{"cut_after":4}'`.

## Between judges

`python -m room_bridge --env-file ~/.hypercompute/room.env clear` empties the queue and the log. `close` and `open`
stop and resume new prompts without stopping the bridge.

## If something is off

| You see | Why | Do |
|---|---|---|
| "host offline" in the header | No bridge heartbeat for 30 s | Is `run` alive? Re-run `check`; it names the failing side |
| "model loading" or "no model loaded" | The gateway has no ready deployment | The bridge waits and claims as soon as it is ready |
| A prompt stays "queued" | Bridge down, or the engine is busy with the owner's own chat | The log says "Engine busy"; it retries every 2 s without spending an attempt |
| Phones cannot open the link | The QR code points somewhere phones cannot reach | `VITE_PUBLIC_URL`, `dev:lan`, or the hotspot |
| "Slow down a little" on a phone | The per-phone token bucket (3, then 1 per 30 s) | Working as designed; say so |
| Answers take 30 s | Pooled Kimi K2.6 decodes about 1.5 tok/s | Lower `HC_ROOM_MAX_TOKENS`, or serve gpt-oss-120b on the GX10 alone, and say it is not pooled |

## Backup

A recorded run against the mock gateway (the phone and the projector side by side, including the failure drill) is
kept outside the repo: `pool-room-backup-run.mp4` in the session notes folder. Every number in it is from the mock
and labelled so on screen; never present it as a measurement.
