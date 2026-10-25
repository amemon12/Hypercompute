# Hypercompute runbook (MVP, private trusted LAN)

Everything below runs from a checkout of this repository with `uv run`. There is no hosted installer or
download URL yet; packaging is a later milestone (see IMPLEMENTATION_STATUS.md). No compiler is needed:
the pinned llama.cpp b11770 binaries are downloaded from the official GitHub release and sha256-verified.

## 0. Prerequisites on every machine

| OS | Needed | Install |
| --- | --- | --- |
| macOS (Apple Silicon) | git, uv | `xcode-select --install` (git), then `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv` |
| Windows 10/11 x64 | git, uv | `winget install --id Git.Git -e` and `winget install --id astral-sh.uv -e` (no WSL needed) |
| Linux x64 / arm64 | git, uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Host only | Node.js 18+ to build the dashboard once | https://nodejs.org or `winget install OpenJS.NodeJS.LTS` / `brew install node` |

uv installs the Python it needs (3.12, from `.python-version`) into the project; nothing touches the
system Python. NVIDIA machines need a driver that supports CUDA 13 (R580 or newer); no CUDA toolkit.

Get the code onto each machine by copying the `hypercompute` folder (USB, AirDrop, `git bundle create
hc.bundle --all` then `git clone hc.bundle hypercompute`), or push it to a private GitHub repository you own.

Lighter path for friends (verified on Windows): the host builds one small wheel that already contains the
pinned engine lock and model manifest, and friends install it as a uv tool, then use `hypercompute ...`
instead of `uv run hypercompute ...` (no checkout needed):

```bash
uv build --wheel                                   # host: dist/hypercompute-0.1.0-py3-none-any.whl
uv tool install ./hypercompute-0.1.0-py3-none-any.whl   # friend, after copying the file
hypercompute engine install --yes
hypercompute join --host <host-ip>:8000 --code <XXXXX-XXXXX>
```

## 1. Host (the machine that owns the model, ideally the biggest NVIDIA GPU)

```bash
cd hypercompute
uv run hypercompute engine install --yes            # pinned llama.cpp b11770 for this OS/GPU, sha256-checked
uv run hypercompute models pull qwen2.5-0.5b-instruct-q4_k_m   # 491 MB, Apache-2.0, sha256-checked
cd frontend && npm ci && npm run build && cd ..     # dashboard, once
uv run hypercompute doctor                          # engine smoke test, LAN address, ports, firewall profile
uv run hypercompute host                            # opens http://127.0.0.1:8000
```

The terminal prints the LAN address friends use (the default-route address comes first). The dashboard,
plans, deployments and chat are only reachable from the host itself (127.0.0.1 plus a per-start owner
token); friends only reach the join/heartbeat endpoints.

Windows host: if the network shows as Public (doctor warns), allow friends to reach the gateway, scoped to
their address, in an Administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Hypercompute gateway" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress <friend-ip> -Action Allow
```

## 2. Friend joins (any OS)

In the dashboard click **Create compute session**; it shows a one-time code that expires in 15 minutes and
the exact command. Each friend needs their own code (**New invite**).

macOS (Apple Silicon, Metal):

```bash
cd hypercompute
uv run hypercompute engine install --yes             # llama-b11770-bin-macos-arm64.tar.gz, 16 MB
uv run hypercompute doctor --host <host-ip>:8000
uv run hypercompute join --host <host-ip>:8000 --code <XXXXX-XXXXX>
```

Windows (PowerShell; NVIDIA uses the CUDA 13.4 build, other GPUs get an explicitly labeled CPU fallback):

```powershell
cd hypercompute
uv run hypercompute engine install --yes
uv run hypercompute doctor --host <host-ip>:8000
uv run hypercompute join --host <host-ip>:8000 --code <XXXXX-XXXXX>
```

Linux: same three commands as macOS.

`join` shows the detected hardware, what will be shared and the soft memory budget, asks for confirmation,
starts `ggml-rpc-server` bound to this machine's private LAN address, and prints a local status page with a
**Stop sharing** button. Useful options:

| Option | Meaning |
| --- | --- |
| `--max-memory 8GiB` | most memory the host may plan onto this machine (soft admission limit, not an enforced GPU cap) |
| `--reserve 6GiB` | RAM/unified memory kept free for the friend |
| `--devices MTL0` / `CUDA0` / `CPU` | which engine devices to share (default: all GPUs; CPU only when no GPU) |
| `--cache` | keep received tensors on disk under `~/.hypercompute/rpc-cache` to speed up reloads (kept after leaving) |
| `--worker-port 47052` | worker port; a busy or reserved port is skipped automatically |
| `--advertise-ip` | override the LAN address (must be the address the host sees) |

The friend stops sharing at any time with Ctrl+C, the Stop button, or `uv run hypercompute leave`
(`--purge-cache` also deletes the tensor cache).

## 3. Run a model together

Dashboard: pick the model, tick the friends, choose **Run together** (every selected friend gets layers),
**Fastest fit** (fewest devices, local first) or **This machine only**, click **Check plan**, then
**Run together**. The lifecycle shows planned, reserving, preparing, loading, warming, ready. Chat streams
real tokens; **Technical details** shows the layer/buffer placement parsed from llama.cpp's own log.

Command-line equivalent (also saves evidence JSON under `results/`):

```bash
uv run python scripts/e2e_check.py --strategy all_selected --bench
uv run python scripts/e2e_check.py --strategy local_only --bench          # same-model local baseline
uv run python scripts/summarize_results.py                                # regenerates docs/RESULTS.md
```

OpenAI-compatible endpoint for local tools (owner token from `GET /api/owner/bootstrap`):
`POST http://127.0.0.1:8000/v1/chat/completions` with header `X-Hypercompute-Owner`.

## 4. First Apple + NVIDIA pair (the next hardware validation)

1. Put the G14 and the Mac on the same network that allows device-to-device traffic (phone hotspot, travel
   router or Ethernet; campus/guest Wi-Fi often isolates clients).
2. G14: `uv run hypercompute host`, create a session.
3. Mac: `uv run hypercompute engine install --yes`, then the `join` command from the dashboard. Click
   **Allow** if macOS asks whether `ggml-rpc-server` may accept incoming connections.
4. G14 dashboard: the Mac card turns eligible after the RPC handshake probe. Run the 0.5B model with
   **Run together**, then the 7B. Check that Technical details shows a nonzero `RPC0[<mac-ip>:47052]` model
   buffer, then run `scripts/e2e_check.py --bench` for both strategies.
5. Record results with `scripts/summarize_results.py` and update IMPLEMENTATION_STATUS.md.

## 5. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `join` cannot reach the host | different subnet or client isolation; check `uv run hypercompute doctor --host ...` on both, use a hotspot/router; Windows host may need the gateway firewall rule above |
| Node stays `ineligible: worker unreachable from host` | friend's firewall blocks the worker port; the card shows the fix (Windows `New-NetFirewallRule ... -RemoteAddress <host-ip>`; macOS click Allow for `ggml-rpc-server` and, if the pf anchor from `join` is loaded, check it names the host's IP; Linux `ufw allow from <host-ip> ...`). The host re-probes every 30 s, so no rejoin is needed after the fix |
| `the worker could not bind ...` | port reserved or in use; on Windows see `netsh int ipv4 show excludedportrange protocol=tcp`; pass `--worker-port` |
| `engine mismatch` | both sides must run llama.cpp b11770: `uv run hypercompute engine install --yes` |
| Node `stale` | no heartbeat for 15 s (sleep, Wi-Fi drop, crash); deployments using it fail and stop; rejoin with a new code |
| Deployment `failed: a planned peer received no layers` | llama.cpp's own load log did not show buffers on that peer; check Technical details |
| Long first load with peers | weights travel over the network to each peer; RPC transfer progress is not observable, `--cache` helps repeat loads |

## 6. Security model (read before sharing)

* This is a private, trusted-LAN prototype. The join code and node credentials protect Hypercompute's own
  control plane only.
* llama.cpp RPC has no authentication and is labeled experimental/insecure upstream. Anyone who can reach a
  worker port can allocate memory and run graphs on it, and its `COMM_INIT` command can make the worker
  listen on `0.0.0.0:<port+1000>` or connect out. Workers bind only to the private LAN address, but the real
  protection is a firewall rule that allows the worker port only from the host's address.
* `join` prints the scoping step for the friend's OS: Windows `New-NetFirewallRule ... -RemoteAddress <host-ip>`;
  Linux `ufw allow from <host-ip> to any port <port> proto tcp` (ufw must be active with default deny); macOS an
  optional pf anchor (`printf '...' | sudo pfctl -a com.apple/hypercompute -f - && sudo pfctl -E`, removed with
  `sudo pfctl -a com.apple/hypercompute -F all`), because the macOS application firewall cannot filter by address.
  Without that step the worker is reachable by anyone on the same network.
* If the host stays unreachable for 5 minutes the agent stops sharing on its own. Closing the terminal, SIGTERM,
  SIGHUP or Ctrl+Break also stop the worker; on Windows and Linux the worker is additionally tied to the agent's
  lifetime by the OS (Job Object / parent-death signal), so even a hard-killed agent does not leave it running.
* Hypercompute never disables firewalls, never opens router ports and never runs commands sent by the host.
  The agent runs only the allowlisted `ggml-rpc-server` with an argument array.
* llama-server on the host listens on 127.0.0.1 with a random per-deployment API key passed by environment.

## 7. Engine-level commands (debugging without Hypercompute)

```bash
# worker (friend), bind to its LAN IP
~/.hypercompute/engines/llama-b11770-macos-arm64/<bin>/ggml-rpc-server -H <friend-ip> -p 47052 -d MTL0
# host: verify the handshake, then run a 2-device layer split. The friend goes first and CUDA0 last: the last device
# holds the output head, so the logits never cross the network. --tensor-split follows the --device order.
llama-server --rpc <friend-ip>:47052 --list-devices
llama-server -m <model.gguf> --alias hc --rpc <friend-ip>:47052 --device RPC0,CUDA0 --split-mode layer \
  --tensor-split 6,19 -ngl all --fit off -c 4096 -np 1 --host 127.0.0.1 --port 18080 --metrics --no-ui -lv 4
```

Source builds (only when no prebuilt variant fits, for example GX10/GB10 before validation):
`scripts/build_llama_macos.sh`, `scripts/build_llama_linux.sh [--cuda-arch 120|121]`,
`scripts/build_llama_windows.ps1`, then `export HYPERCOMPUTE_LLAMA_BIN_DIR=<bin dir>`.

## 8. Tests

```bash
uv run pytest                 # portable protocol tests (no GPU, no network)
uv run pytest -m hardware     # needs the engine and model on this machine
cd frontend && npm test && npm run typecheck
```

## 9. Where things live

`~/.hypercompute/` (override with `HYPERCOMPUTE_HOME`): `engines/` (pinned binaries), `downloads/` (verified
archives), `models/`, `coordinator/` (SQLite db, `records/runs.jsonl`, `records/events.jsonl`, llama-server
logs, managed-process records), `agent/` (node id, state with the node credential, worker log), `rpc-cache/`
(only with `--cache`). Removing that folder removes everything Hypercompute created.
