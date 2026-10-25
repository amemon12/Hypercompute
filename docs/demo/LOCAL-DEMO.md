# Local demo

This is a simulated walkthrough with a real browser-to-browser join. It does not
download a model, inspect a friend's hardware, start compute workers, or run inference.
The hardware, resource calculation, DeepSeek 671B layer split, loading, and replies
are fixed demo data. Use [JUDGE-RUNBOOK.md](JUDGE-RUNBOOK.md) for the separate live
hardware demo.

## Start the page

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe app/demo_server.py
```

The server listens on `127.0.0.1:8792` and opens the authenticated host page. A friend
cannot reach this loopback address from another computer. To share over a reachable
LAN or private network, replace the example address with this computer's address:

```powershell
.\.venv\Scripts\python.exe app/demo_server.py --bind 0.0.0.0 --public-url http://192.168.1.10:8792
```

`--public-url` sets the address placed in the copied invitation. It must be reachable
from the friend's browser. The server accepts that address and localhost. For an
additional address or resolvable hostname, add `--allowed-host demo-host.local:8792`.
This option is repeatable and accepts `host:port`, without `http://`.

For a launcher that needs the private host URL in a file, add:

```text
--host-url-file .hypercompute/demo-host-url --no-browser
```

`.hypercompute/` is ignored by Git. The file contains the private host launch URL;
keep it private. The launch token arrives in the URL fragment, is saved in that
browser tab's session storage, and is removed from the address bar. A new browser
tab opened at the bare server address does not receive host controls automatically.
The terminal prints only the bare address. Never share the private host launch URL.

## Walk through it

1. Turn **Hosting** on. The GX10 and GeForce RTX laptop become the two active demo
   computers. Invite and chat controls are unavailable while hosting is off.
2. Click **Copy invite**. It copies the public `/join` URL and a six-character
   session code. Send both to the friend. Copying the invitation does not join a
   computer automatically.
3. The friend opens the join page, enters the code, and clicks **Join session**.
   This changes the shared web session. The MacBook Air shown there is a sample
   Apple M4 with 16 GB unified memory, regardless of the browser's actual hardware.
4. The host page shows the MacBook joining, reads its demo specifications, calculates
   resources, fills the layer bar, and shows loading. This sequence takes about
   15 seconds after the join reaches the host page. Reduced-motion mode keeps the
   timing while removing the gradual layer animation.
5. When the model says **Ready**, enter a question and click **Ask**. A prepared
   response appears after a short pause and is revealed word by word. Suggested
   questions demonstrate the pool and layer assignments; this is not generated text.
6. Turn **Hosting** off to reset. Pending animation and response timers stop, the
   web session ends, and its code becomes invalid. Turning hosting back on creates
   a new code for another walkthrough.

## What the numbers mean

The active demo uses three computers and an illustrative 61-layer split:

| Computer | Demo layers |
|---|---:|
| GX10 | 56 |
| GeForce RTX laptop | 3 |
| MacBook Air | 2 |

These are presentation fixtures, not a planner result or benchmark. They differ
from the earlier four-computer, verified deployment recorded in
[REFERENCE-671B.json](REFERENCE-671B.json), which included a second MacBook and
assigned 54 layers to the GX10. The displayed memory figures are also fixed for
this walkthrough. Do not present the timed loading or scripted reply as a measured
DeepSeek run.

If a friend cannot open the join page, check that the public address is reachable
from their network and matches the server's allowed addresses. Keep the server
process running throughout the demo. A server restart creates a new private host
launch URL; reopen that URL to regain host controls.
