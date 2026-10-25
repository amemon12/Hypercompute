# Demo app topology

The demo app is `app/hcapp.py`. It serves `app/static/`, connects to the gateway on `127.0.0.1:8001`, and uses
the Python SDK for planning, loading and chat. This document describes the implementation, not the earlier design.

## Machines and processes

| Machine | Role | Processes |
|---|---|---|
| GX10 | Headless host | Gateway, demo app, model server |
| Windows laptop | Presenter and optional peer | Browser, SSH tunnel, peer agent and worker |
| MacBook Airs (`mac-a`, `mac-b`) | Peers | One-click join, peer agent and worker |

The host holds the model file. Peers receive assigned layers over encrypted Iroh connections, directly when the
network allows it or through a relay otherwise. Managed peers can download their assigned tensors into a local cache.
The presenter reaches the host's app through SSH. The gateway and app both stay on loopback.

## Ports

| Port | Where | Purpose |
|---|---|---|
| 8001 | Host loopback | Gateway; the SDK and demo app use this port |
| 8790 | Local machine loopback | Default demo app port |
| 8791 | GX10 and presenter loopback | Suggested alternate app port for the remote demo |
| 47052 and nearby free ports | Peer loopback | Workers reached through Iroh |

The app checks both the loopback hostname and the port in the HTTP Host header. Forward the same port on both
ends. Forwarding local 8791 to remote 8790 would fail this check.

## Local launch

From the repository root:

```sh
uv sync
uv run --no-sync python app/hcapp.py
```

The app opens a browser link with a per-launch key in the URL fragment. The page uses that key in the
`X-App-Token` header for every API call. Static files need no key. The gateway's separate owner token stays in the
server process.

The wrappers `join/hypercompute-app.command`, `.cmd` and `.sh` run this entry point. They expect `uv sync` to have
already completed. These app wrappers do not use the first-run bootstrap prompts of the join wrappers.

## Headless GX10 launch

Keep the existing gateway on port 8001. On the GX10, from its checkout:

```sh
uv run --no-sync python app/hcapp.py --port 8791 --no-browser
```

`--no-browser` prints a private launch link. Capture it only in a local file readable by its owner, transfer it
through the SSH connection, and open it privately. Do not paste the link into chat, a tracked file, a shared log or
a recording. Treat it like a session credential.

On the presenter laptop, use the configured SSH alias or key-only destination:

```sh
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:8791:127.0.0.1:8791 <gx10-ssh-alias>
```

Open the private launch link in the presenter browser. The app calls the gateway locally on the GX10, so the
owner token stays there. The tunnel binds only to `127.0.0.1` on the presenter. Leave the tunnel running while
presenting. Do not forward gateway port 8001 onto a laptop that also uses local host or join scripts, since those
scripts could mistake the remote gateway for a local host.

The app does not install the GX10's source-built engine or set its environment. Reuse the verified gateway startup
configuration on that machine. No sudo command is needed for the app or tunnel.

## Host and Join tabs

The Host tab lists downloaded models and eligible peers. Run plans the selected model across eligible devices,
shows layer counts, and keeps the deployment loaded for chat. The app refuses `gpu_disk` and `stream` plans.
A model loaded outside the app can be shown and used for chat; only a deployment started by this app can be stopped
with its Stop control. Start hosting reuses a running gateway. Stop hosting only stops a gateway started by that
app process.

The Join tab passes an invite to the one-click helper through `HC_INVITE`. It enables managed downloads and uses
the one-click consent path. Sharing runs separately from the browser. Stop sharing calls the leave helper and
keeps cached model pieces. The displayed sharing text is the last nonempty line of the one-click join log; it is
not a fresh worker health probe. Use the host's peer list to confirm the machine is eligible.

For the filmed MacBook join, use `join/hypercompute-join.command`, enter a generic name such as `mac-a`, and allow
host downloads. The invite input is hidden. Keep the host's invite panel and browser address bar outside the crop.

## Rehearsal checks

- Confirm the host page, styles, fonts and images load through the tunnel.
- Confirm an API request without `X-App-Token` is refused.
- Check each expected peer is eligible, then verify the layer bar assigns it work.
- Preload the large model and send a short prompt before the judges arrive.
- Keep the model loaded. Closing the browser does not stop a gateway or peer worker.
- If a peer drops, stop the failed deployment and load the planned fallback. There is no live migration.

See [JUDGE-RUNBOOK.md](../demo/JUDGE-RUNBOOK.md) for presentation steps and
[VIDEO-SHOTLIST.md](../demo/VIDEO-SHOTLIST.md) for recording.
