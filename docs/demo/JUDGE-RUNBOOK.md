# Hypercompute judge runbook

Use this with the [numbers sheet](../pitch/NUMBERS.md), [shot list](VIDEO-SHOTLIST.md), and
[cross-check](../CROSS-CHECK-2026-10-25.md). The presentation is about three minutes. Setup and model loading happen
before the judges arrive.

## Setup order

1. Connect power, keep the computers awake, and put the GX10, G14 and both Macs on a stable network with internet
   access for Iroh and model downloads. The host is the GX10. Use `mac-a`, `mac-b`, and `g14` as peer names.
2. Use the reviewed checkout on each machine. From its root, run `uv sync`. Keep existing model and RPC caches.
   Do not reset someone else's checkout or delete caches to make the demo look fresh.
3. On the GX10, check the existing gateway before starting another one:

   ```bash
   uv run --no-sync hypercompute status --port 8001
   ```

   If it is not running, use the host's existing engine environment and start:

   ```bash
   uv run --no-sync hypercompute host --transport iroh --port 8001 --no-browser
   ```

   Leave that process running. Use key-only SSH. No sudo command or password is needed for this rehearsal.
4. On each friend, open `join/hypercompute-join.command` on macOS or `join/hypercompute-join.cmd` on Windows.
   Paste a fresh one-use invite privately. Set a generic peer name and enable host downloads for the large model.
   Retain the memory reserve. Confirm sharing through `hypercompute-status`. Invitations are pairing credentials;
   do this before screen capture or crop the entire input area.
5. On the host, verify the peer list with the SDK. This prints only names, state and devices:

   ```bash
   PYTHONPATH=sdk/python/src uv run --no-sync python - <<'PY'
   from hypercompute_sdk import Pool
   with Pool() as pool:
       for peer in pool.peers():
           print(peer['display_name'], peer['state'], peer['contribution_devices'])
   PY
   ```

   All three friends must be `eligible` before a four-machine claim. Check the actual layer bar after planning;
   joining alone does not prove a computer holds layers.
6. Check the SDK with the small cached model. Both examples stop the deployment they create by default:

   ```bash
   PYTHONPATH=sdk/python/src uv run --no-sync python sdk/python/examples/quickstart.py qwen2.5-0.5b-instruct-q4_k_m
   PYTHONPATH=sdk/python/src uv run --no-sync python sdk/python/examples/seed_and_run.py qwen2.5-0.5b-instruct-q4_k_m --seed-timeout=120
   ```

   The small model can skip pre-seeding because its small tensors transfer during loading. A genuine seed failure
   stops the example before deployment.

## Preload on the GX10

Choose one large model. DeepSeek is the default for this rehearsal. Run from the GX10 checkout:

```bash
PYTHONPATH=sdk/python/src uv run --no-sync python sdk/python/examples/seed_and_run.py deepseek-v3.1-ud-tq1_0 --max-tokens=400 --seed-timeout=1800 --keep-running
```

Alternatively, use Kimi:

```bash
PYTHONPATH=sdk/python/src uv run --no-sync python sdk/python/examples/seed_and_run.py kimi-k2.6-iq1_s --max-tokens=400 --seed-timeout=1800 --keep-running
```

Run these large-model commands only on the GX10. The example plans across every eligible peer, verifies each seed
job, deploys, waits for readiness, and checks chat and streaming. `--keep-running` retains the deployment only after
both calls return visible answer text. `--max-tokens=400` applies to each call. A reasoning-only response fails the
check and stops the deployment; increase the budget before retrying if that happens. Leave enough time for both
seeding and loading. A warm cache is not a cold-start measurement.

Accept a feasible `gpu`, `gpu_ram`, or `moe_offload` plan. Never pass `--allow-disk-tiers` for this demo. A `gpu_disk`
or `stream` plan on the GX10 is a stop condition, especially for a model larger than its unified memory.

Record the deployment identifier and exact stop command printed by the example. It contains no credential.
To stop that exact deployment later, replace the placeholder below with its identifier:

```bash
PYTHONPATH=sdk/python/src uv run --no-sync python -c "from hypercompute_sdk import Pool; Pool().stop('DEPLOYMENT_ID')"
```

The demo app displays models loaded through the SDK. Its Stop model button controls only models that the app
started. Do not start another model while the prepared one is ready.

## Open the host screen

The real app is `app/hcapp.py`. It binds loopback and requires a per-launch token. Run it on the GX10, then forward
the same port on the presenter computer. The same port is required by the app's Host-header check.

On the G14, in PowerShell, set `HC_GX10` privately to the existing SSH destination. Keep this terminal outside the
recorded window. The following uses the existing identity and the Git SSH client verified during the cross-check:

```powershell
& 'C:/Program Files/Git/usr/bin/ssh.exe' -i "$HOME/.ssh/hypercompute_ed25519" `
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=15 -L 127.0.0.1:8791:127.0.0.1:8791 $env:HC_GX10 `
  'cd ~/hypercompute && ~/.local/bin/uv run --no-sync python app/hcapp.py --port 8791 --no-browser'
```

The headless launcher emits its private launch URL. Open it in the presenter browser without recording the terminal,
copying it into a document, or logging it. The page removes the token from its address bar and keeps it in that tab.
A fresh tab without the stored launch token can load the page but cannot access the API. Keep the SSH session and
browser tab open.
Do not forward gateway port 8001 onto a presenter laptop that is also a peer.

Before the judges arrive, confirm:

- The Host tab says `hosting` and all expected friends are `eligible`.
- The deployment is `ready`, the model name is correct, and every intended machine has a nonzero segment in the
  layer bar. The segment counts are the current plan, not a fixed arrangement promised in the pitch.
- A short question produces visible text. Some reasoning models consume a short budget without displaying an
  answer. Use the already-checked model and prompt; do not change the model in front of the judges.
- The projector or OBS captures only the app or website. No terminal, clipboard, invite field or private launcher
  output is visible. Keep notifications out of the captured surface.

## What to click and say

| Time | Screen and action | Suggested words |
|---|---|---|
| 0:00 to 0:25 | Open hypercompute.co, then Benchmarks. Visit Laptops and Every machine. | “Hypercompute lets friends run one model together. Each computer holds part of it. These tabs show the comparisons we actually measured.” |
| 0:25 to 0:50 | Show a Mac opening the join command, then its entry in the host peer list. Keep the private invite off camera. | “A friend accepts an invitation and chooses what memory to share. Their computer appears here.” If the script was invoked remotely, say that instead of claiming a physical double-click. |
| 0:50 to 1:25 | Host tab, already-ready DeepSeek or Kimi, then the layer bar. | “We loaded this before you arrived. The planner assigns layers using the memory each machine can offer. Every machine shown here has layers in this plan.” |
| 1:25 to 2:05 | Enter a short question in Ask and submit. Watch the actual streamed response. | “This answer is coming from the pool. Generation still runs through the devices in sequence, so a slower link can slow the whole request.” |
| 2:05 to 2:45 | Return to benchmark tabs and the numbers sheet. | “The strongest gains were when a laptop otherwise reread weights from disk. On the 120B laptop test, the first token went from 34.22 seconds to 4.83, one row per side with three measurements. That result depends on the network and model.” |
| 2:45 to 3:00 | Close on the honest limit. | “Pooling is not always faster. The interleaved 671B test had even decode, with a sooner first token. Earlier GX10 runs decoded more slowly when pooled. We measure both.” |

The quoted 120B figures are NUMBERS section 1b. The 671B interleaved comparison is section 3e, with four measurements
per side across alternating runs. Do not present a functional rehearsal speed as a benchmark or quote the retracted
cross-session decode gain.

Maverick is optional. NUMBERS section 3e reports 9.79 tokens/s on the GX10 alone, a 128-token prompt and 64-token
answer, one row with three measurements. Pooling made that measured run slower. Switching models takes another
load, so omit the live switch from a short judge visit unless it is already prepared separately.

## Fallbacks

**A peer drops.** Stop the failed deployment through the owner that started it. Read the failure on the app and
check the peer list. Do not imply automatic recovery has kept the answer running. Either reconnect the missing peer
and replan, or use the GX10 alone. Keep the same authenticated browser tab.

**The network is unreliable.** On the GX10, stop the prepared deployment by its identifier. The following safe
local-only example refuses disk tiers and leaves a ready Maverick deployment for the app to display:

```bash
PYTHONPATH=sdk/python/src uv run --no-sync python - <<'PY'
from hypercompute_sdk import Pool
with Pool() as pool:
    plan = pool.plan('llama-4-maverick-17b-128e-instruct-ud-q2_k_xl', strategy='local_only')
    if not plan['feasible'] or plan.get('tier') in {'gpu_disk', 'stream'}:
        raise SystemExit('This plan is not safe for the demo. Use the small model.')
    dep = pool.deploy(plan['id'])
    try:
        pool.wait_ready(dep['id'], timeout=1800)
    except Exception:
        pool.stop(dep['id'])
        raise
    print('Ready. Deployment:', dep['id'])
PY
```

This is a fallback procedure, not a claim that Maverick was rerun in this cross-check. If it cannot fit safely,
use the cached small model locally. Keep the layer bar visible so the audience can see which machines participate.

**The GX10 is unavailable.** Use the small Qwen model with a laptop host and the other laptops as peers. First have
each laptop leave the old pool, retaining its cache. Start the laptop host and have the others join its invitation.
From the new host's checkout, download and preload the small model:

```bash
uv run --no-sync hypercompute models pull qwen2.5-0.5b-instruct-q4_k_m
PYTHONPATH=sdk/python/src uv run --no-sync python sdk/python/examples/seed_and_run.py qwen2.5-0.5b-instruct-q4_k_m --seed-timeout=120 --keep-running
uv run --no-sync python app/hcapp.py
```

Download before the venue loses internet. These commands use a macOS or Linux shell; on Windows set
`$env:PYTHONPATH = 'sdk/python/src'` in PowerShell, then run the Python command without the `PYTHONPATH=...` prefix.
The app opens locally on this laptop, so the GX10 SSH tunnel is unnecessary. Explain that this shows the joining,
planning and streaming flow at a smaller scale. Do not run the 671B or 1T model on the laptops.

**No usable network.** Show the local rehearsal recording and the measured comparisons, with its recorded-rehearsal
label visible. Do not claim the recording is live. Keep x402 on Base Sepolia testnet if that integration is shown.

## After the demo

Stop the deployment by its identifier, then let each friend choose when to stop sharing. Retain their model caches.
Stop the app and tunnel when finished. Keep the recording local until publication is approved.
