# Mac host runbook: gpt-oss-120b on three M4 MacBook Airs

For Azim. Your 16 GB M4 Air is the host: it runs the gateway and serves `gpt-oss-120b-mxfp4` (59.03 GiB GGUF,
36 blocks, 128 experts, 4 active per token). The two other 16 GB M4 Airs join it as peers over Iroh on a phone
hotspot and take whole layers. No Windows machine is involved.

Every command and flag below was checked against this branch's code (`backend/hypercompute/cli.py` argparse,
`scripts/capacity/run_big.py` argparse); `tests/test_run_big.py` parses every `uv run` line of this file with them.
Numbers marked "estimate" are not measurements. Nothing here has been run on
a Mac host yet; section 9 lists what is unverified.

Type the commands in Terminal. Lines starting with `#` are notes, not commands.

## 0. Rules for tonight

- Do not set any `HYPERCOMPUTE_*` environment variable. `models pull`, `host` and `run_big.py` must all use the
  default home `~/.hypercompute`, or the host will not see the model you downloaded.
- Download the model (step 1.5) on normal Wi-Fi, before you switch to the hotspot. It is 63.4 GB.
- All three Macs: power adapter in, lid open, `caffeinate -dims &` running, other apps quit (the planner sizes each
  Mac's share from the memory that is free when it plans).
- All three Macs on the same phone hotspot. A guest or office Wi-Fi with client isolation did not work last time; the
  phone hotspot did.

## 1. Host Mac (Azim)

### 1.1 Check the machine

```sh
uname -m                  # must print arm64
sw_vers -productVersion   # 13.3 or newer (the llama.cpp b11770 macOS build targets 13.3)
pmset -g batt             # must say "Now drawing from 'AC Power'"
df -h ~                   # "Avail" must be at least 70Gi before the model download
```

Why 70 GiB: the model file takes 59.03 GiB, and `run_big.py` stops the run when the volume holding the models has less
than 5 GiB free (`--min-drive-free-gib`, the guard that keeps swap files from filling the disk). It checks this before
it loads anything, so with only the downloader's minimum free (61.03 GiB: the file plus a 2 GiB margin,
`ModelService._download_files`) the run would end at once with `aborted_by_guard`. 70 GiB leaves about 11 GiB after
the download: the 5 GiB floor, plus room for swap and logs. If `Avail` is short, step 1.4 can free the RPC tensor
cache from last night (about 5 GiB); otherwise delete other files first. Do not lower the floor.

### 1.2 Update the code

```sh
cd ~/hypercompute          # your existing checkout; use its real path
git status --short         # must print nothing; if it lists files, stop and tell operator
git fetch origin
git checkout codex/iroh-transport
git pull --ff-only origin codex/iroh-transport
git log -1 --oneline       # put this line in your message back
uv sync
```

### 1.3 Engine

```sh
uv run hypercompute engine install --yes
uv run hypercompute doctor
```

`engine install` either downloads llama.cpp b11770 for macOS arm64 with Metal (16 MB) or says it is already
installed, and must end with `Smoke test: passed`. `doctor` must show `[OK  ] llama.cpp engine: ...`. `[WARN]` lines
about LAN addresses do not matter for Iroh; any `[FAIL]` line does.

### 1.4 Stop last night's join agent

```sh
uv run hypercompute leave
# or, to also delete last night's RPC tensor cache (about 5 GiB; a host Mac never reads it):
uv run hypercompute leave --purge-cache
```

Fine outputs: `This machine is not currently sharing.`, or `Host unreachable (...); it will mark this machine
stale.` (the Windows host is off; it can take a few seconds).

### 1.5 Download the model (on normal Wi-Fi)

```sh
uv run hypercompute models list | grep gpt-oss-120b
uv run hypercompute models pull gpt-oss-120b-mxfp4
```

- The `list` line must show `gpt-oss-120b-mxfp4`. If it does not, your checkout has its own `configs/models.yaml`,
  which replaces `configs/models.example.yaml`: run `mv configs/models.yaml configs/models.yaml.bak` and list again.
- `pull` writes to `~/.hypercompute/models/gpt-oss-120b-mxfp4/gpt-oss-120b-MXFP4.gguf` (the same home the host
  uses), prints a percentage every 3 seconds, checks the sha256 at the end and then prints `Verified: <path>`. Only
  after that line does the host count the model as downloaded.
- It refuses with `not enough disk: need ... GiB more + 2 GiB headroom` when the home volume is short.
- Any download error prints `error: ...` and then `fix: check configs/models.yaml ...`. That fix line is the CLI's
  generic text for every model error (`cli.py` `main`); ignore it for disk and network errors.
- If it stops (Wi-Fi drop, `download interrupted ... retry to resume`), run the same `pull` command again: it
  continues from the `.part` file next to the target with an HTTP Range request. To stop it yourself press Ctrl+C;
  if the prompt does not come back within a few seconds, close the terminal window (the download runs in a worker
  thread). The `.part` file stays either way. A finished download whose sha256 does not match is deleted and the
  command fails; running it again starts over.

Download time for 63.4 GB (computed, size divided by rate, plus one sha256 pass over the file at the end):

| Internet speed | Time |
| --- | --- |
| 100 Mbit/s (12.5 MB/s) | about 85 min |
| 200 Mbit/s (25 MB/s) | about 42 min |
| 400 Mbit/s (50 MB/s) | about 21 min |
| 800 Mbit/s (100 MB/s) | about 11 min |

After `Verified:`, run `df -h ~` again: `Avail` must still be at least 10Gi (see 1.1).

The peers do not need the model file: the host sends them their layers.

### 1.6 Start the host (now on the hotspot)

Join the phone hotspot first, then start the host in a new terminal (the invitations carry this Mac's addresses on
the network it is on when the host starts):

```sh
cd ~/hypercompute
caffeinate -dims &
uv run hypercompute host --transport iroh --local-max-memory 10GiB
```

- `--local-max-memory 10GiB` is the budget for this Mac's own GPU (MTL0). On a 16 GiB Mac that is also the default
  (`agent/contribution.py`: 65% of the unified pool, at most total minus 6 GiB), written out so it is visible. The
  host keeps a 4 GiB reserve for macOS by default (the larger of 4 GiB and 20% of RAM), and the planner also caps the
  Metal part at what is free when it plans minus that reserve (`services/deployments.py`, `local_budgets`). Your Air
  reported a Metal limit of 11.84 GiB last night (its MTL0 total), so a budget above about 11 GiB could not be used
  by Metal anyway, and every GiB given to Metal is one less GiB of file cache for the experts on the CPU. Do not
  raise it.
- It prints `Dashboard (this machine only): http://127.0.0.1:8000/`, then `Iroh ready. Send this one-use invitation
  to your friend:` and a `uv run hypercompute join --ticket ... --code ...` line with its expiry. Leave this terminal
  open for the whole session.
- The dashboard page may only say "Hypercompute gateway is running ... The dashboard has not been built yet": the
  frontend build is not in git. That is fine; use `uv run hypercompute status` (section 3).
- The port is 8000 (`--port` default). If it says `cannot listen on 127.0.0.1:8000`, start with `--port 8002` and add
  `--port 8002` to every `invite`, `status` and `run_big.py` command below.

### 1.7 One invitation per peer

In a second terminal, once per peer Mac, right before that Mac joins:

```sh
cd ~/hypercompute
uv run hypercompute invite
```

Each invitation works once and expires after 15 minutes (`config.INVITE_TTL_S`). Send the printed
`uv run hypercompute join --ticket ... --code ...` line to that Mac's owner (AirDrop or a message). The one printed
at host start is a valid first invitation too.

## 2. Each peer Mac (two of them)

### 2.1 Tools and code

```sh
git --version                                   # if macOS offers to install the command line tools, accept
curl -LsSf https://astral.sh/uv/install.sh | sh # or: brew install uv
source $HOME/.local/bin/env                     # or open a new terminal
git clone -b main https://github.com/amemon12/Hypercompute.git ~/hypercompute
cd ~/hypercompute
uv sync
```

The repository is private. If the clone asks for credentials the account has no access to, either ask operator to add
that GitHub account, or let Azim make a bundle on the host Mac and AirDrop it:

```sh
# on Azim's Mac
cd ~/hypercompute && git bundle create ~/Desktop/hypercompute.bundle codex/iroh-transport
# on the peer, after AirDrop put it in Downloads
git clone -b codex/iroh-transport ~/Downloads/hypercompute.bundle ~/hypercompute
cd ~/hypercompute && uv sync
```

A Mac that already has a checkout updates it like step 1.2 instead of cloning.

### 2.2 Engine

```sh
uv run hypercompute engine install --yes
uv run hypercompute doctor
```

Same checks as 1.3. The peer must run the same llama.cpp release as the host (b11770); `join` refuses with
`engine mismatch` otherwise.

### 2.3 Join

On the hotspot, with the line Azim sent, adding the four flags at the end:

```sh
cd ~/hypercompute
caffeinate -dims &
uv run hypercompute join --ticket <TICKET> --code <CODE> --max-memory 10GiB --reserve 4GiB --cache --yes
```

- `--max-memory 10GiB`: most memory the host may plan onto this Mac. `--reserve 4GiB`: memory kept free for macOS.
  `--cache`: keep the received layers on disk so a second load of the same layers is local. `--yes`: accept without
  a prompt. Same settings as Azim's Mac used as a peer last night.
- `--cache` needs disk: run `df -h ~` first, `Avail` at least 20Gi (the cache holds a copy of every layer this Mac
  receives, up to its 10 GiB budget). With less, leave `--cache` out. llama.cpp b11770 writes a cache file without
  checking that the write succeeded and later loads whatever the file holds (`ggml-rpc.cpp`, `set_tensor` and
  `get_cached_file`), so a cache written onto a full disk can load broken weights the next time.
- It prints `Connecting to the host through Iroh...` and `Encrypted connection established: ...`, then keeps
  running (`Press Ctrl+C ... to stop sharing.`). Leave the terminal open.
- If macOS asks whether Python or the llama.cpp worker may accept incoming network connections, allow it.

## 3. Check before running (host Mac)

```sh
uv run hypercompute status
```

Want:

- `Hosting on port 8000:` and `transport: iroh`.
- One `connection:` line per peer with `'connected': True` and a path whose mode is `direct`. A `relay` path works
  but goes over the phone's internet connection: slow (last time 1.6 MB/s) and it spends mobile data on gigabytes of
  layers. Fix it before running (same hotspot, no VPN on any Mac).
- Three node lines: this Mac and both peers, each `eligible`, last heartbeat under 15 s ago (`config.STALE_AFTER_S`).

## 4. Run (host Mac, from `~/hypercompute`)

### 4.1 Plan only (seconds)

```sh
sudo purge    # asks for this Mac's password; empties the file cache
sleep 15      # the host re-measures free memory every 10 s
uv run python scripts/capacity/run_big.py --model gpt-oss-120b-mxfp4 --port 8000 --strategy all_selected --plan-only
```

Why `purge`: the planner sizes this Mac's Metal and CPU shares from free memory as psutil reports it on macOS, free
plus inactive pages (`agent/inventory.py`). File pages the download, the sha256 check or an earlier run touched can sit
on the active list and count as used, so a plan made right after them gets less than the Mac really has. `purge`
drops those pages; nothing else changes. Do it again before 4.3 and before any rerun, on the host and, before a rerun,
on both peers too (the peers' `--cache` writes are file pages as well).

It prints `plan xxxxxxxx: feasible True, tier ..., devices [...], ...`. Want `feasible True`, tier `moe_offload` and
two RPC devices (one per peer) in `devices`. If an RPC device is missing, a peer was not eligible when the plan was
made: go back to section 3. `--port 8000` matters: the host listens on 8000 by default, and `run_big.py` defaults to
8001 (the old Windows host). This also writes a small `run.json` with status `plan_only`. If it also prints
`guard: models volume free ... below 5.00 GiB`, free disk space before 4.2: the full run would stop before loading.

### 4.2 The three-Mac run

Right after 4.1, so the plan it makes sees the same free memory:

```sh
uv run python scripts/capacity/run_big.py --model gpt-oss-120b-mxfp4 --port 8000 --strategy all_selected \
  --prompt-tokens 128 --output-tokens 64 --warmup 1 --reps 2 --load-timeout 5400 --chat-timeout 1800 \
  --min-available-gib 0
```

### 4.3 The host-only comparison

```sh
sudo purge
sleep 15
uv run python scripts/capacity/run_big.py --model gpt-oss-120b-mxfp4 --port 8000 --strategy local_only \
  --prompt-tokens 128 --output-tokens 64 --warmup 1 --reps 2 --load-timeout 5400 --chat-timeout 1800 \
  --min-available-gib 0
```

Leave the peers joined for 4.3; the plan simply does not use them. With `purge` before 4.1 and 4.3 both runs start
with an empty file cache, so they compare fairly.

Each run prints load progress, one line per chat, and ends with `<status>: <path of run.json>`, the path being
`~/hypercompute/results/capacity/<UTC stamp>-big-gpt-oss-120b-mxfp4/run.json`. `ok` is the good status. The runner
stops the deployment it created at the end, after an error, and on Ctrl+C.

**The safety guards on a Mac** (`scripts/capacity/hostmon.py`, recorded in `run.json` under `measured.host.guards`
with `platform: macos`):

| Guard | Default | Flag | On a Mac |
| --- | --- | --- | --- |
| Swap used grows by more than | 2 GiB | `--max-swap-growth-gib` | on: private memory past RAM goes to swap files on the boot volume |
| Free space on the models volume below | 5 GiB | `--min-drive-free-gib` | on: the models and the swap files share that volume |
| Available RAM below | 1 GiB | `--min-available-gib` | off tonight with `0`, still recorded (see below) |
| Commit limit grows | 0.25 GiB | `--max-commit-growth-gib` | Windows only; macOS has no commit counter (recorded inactive) |

On a Mac, available RAM is free plus file-backed pages from `vm_stat` (no sudo): the pages macOS can hand out without
compressing or swapping, so the memory-mapped experts count as available. psutil's macOS figure (free plus inactive
pages) would count the experts the engine touches as used RAM, the way the Windows figure did when it stopped two
healthy runs last night. The vm_stat reading has never been taken on a Mac during a run, so tonight
`--min-available-gib 0` keeps that floor from stopping anything while `run.json` still records it every 0.5 s
(`measured.host.samples[].available`, with the file-backed part as `system_cache`). Swap growth is the sign that the
Mac is out of memory, and that guard stays on, as does the disk floor.

## 5. Send the results back

Option A, if your GitHub account can push to the repository:

```sh
cd ~/hypercompute
git checkout -b results/mac-host-20261022
git add results/capacity
git commit -m "test: gpt-oss-120b with an M4 Air host and two M4 Air peers"
git push -u origin results/mac-host-20261022
```

Option B, a zip of only the new result folders:

```sh
cd ~/hypercompute
zip -r ~/Desktop/mac-host-results.zip $(git ls-files --others --exclude-standard results/capacity)
```

Send the zip, plus the `git log -1 --oneline` line from step 1.2. If a run failed, also send the engine log named in
that `run.json` under `measured.engine_log.path` (`~/.hypercompute/coordinator/logs/llama-server-<id>.log`).
`run.json` never contains the owner token or an invitation code.

## 6. What to expect (estimates)

- **Load time** is mostly the host sending each peer its layers over the hotspot. Last night one Mac received
  4,939 MiB (4.8 GiB) in about 521 s of loading, 531 s from create to ready
  (`results/capacity/20261022T083945Z-big-gpt-oss-120b-mxfp4`). At that rate (about 9.5 MiB/s, computed) 10 GiB takes
  about 18 min and 20 GiB about 36 min. How many bytes each peer takes is in the `run.json` from 4.1, under
  `predicted.plan.assignments` (`weight_bytes` of each `RPC` device). With `--cache` on the peers, a later load of the
  same layers took 59-62 s to ready.
- **Speed on a Mac host is unknown.** The Windows host did 4.05 tok/s alone and 4.52 tok/s with one Mac. This host
  has 16 GiB instead of 31 GiB, so far less file cache for the experts kept on the CPU and more SSD reads per token.
  The plan line shows the planner's own estimate; last night it predicted 1,081 ms per token against 221 ms
  measured, so treat it as rough.
- **First token**: 21-31 s for a 126-token prompt on the Windows host; on this Mac expect tens of seconds or more.
  The gateway gives up on a chat when llama-server sends nothing for 180 s (`config.UPSTREAM_IDLE_TIMEOUT_S`, not a
  command-line flag), so a first token slower than that fails the chat. The host-only run (4.3) is the likely one:
  its prompt pages in the experts of all 36 layers from the SSD.

## 7. Troubleshooting

**`guard: ...; stopping the deployment`.** The run ends `aborted_by_guard`; `measured.host.guard_tripped` says why.

- `swap used grew ...`: memory ran out. Check `measured.engine_log`: `metal_private_mib` (Metal allocations, private
  RAM on a Mac) and `private_host_mib` (CPU copies) should be small, `file_backed_host_mib` (CPU experts, file cache)
  and `metal_mapped_mib` (Metal over the mapped file) large. If they look like that, other apps were pushed to swap:
  quit them and run again. If `private_host_mib` is large and `file_backed_host_mib` near zero, stop and send the
  results: that layout copies the experts into RAM.
- `models volume free ... below 5.00 GiB`: the disk is nearly full. Free space (`leave --purge-cache` on this Mac
  frees last night's RPC cache) and run again. Do not lower the floor.
- `available RAM ... below ...`: you left out `--min-available-gib 0`. Send that `run.json` anyway: it is the first
  record of the vm_stat reading under load.

**A chat fails with `no upstream data for 180s`** (run status `chat_failed` or `partial`; the text is in that chat's
`error` under `measured.chats.reps`): the gateway's idle limit (section 6). Send the `run.json`; there is no flag for
it.

**`refused`.** The message says what to fix: another deployment is still running, or the model is not downloaded and
verified (finish 1.5 in this same home).

**A peer is stale or missing.** `status` shows it `stale`, or its last heartbeat over 15 s ago. Usually the Mac slept
or left the hotspot. Its agent keeps retrying for 5 minutes (`HOST_UNREACHABLE_STOP_S`) and then exits; after that,
make a new invitation (1.7) and run the join command again on that Mac. A peer that drops during a run makes chats
fail (`partial` or `chat_failed`): stop, check section 3, run again.

**Metal out of memory.** The run ends `load_failed` or `chat_failed`. Look in the host's engine log
(`~/.hypercompute/coordinator/logs/llama-server-<id>.log`) and on the peers in
`~/.hypercompute/agent/logs/ggml-rpc-server.log` for `failed to allocate Metal buffer of ... bytes (out of memory)`
(llama.cpp b11770, `ggml-metal.cpp`) or any line with `out of memory`. Quit other apps on that Mac and run again so
the plan is made from fresh free memory. If it repeats on a peer, have it `leave` and join again with
`--max-memory 8GiB`; on the host, restart it with `--local-max-memory 8GiB`.

**Anything else.** Keep the terminals as they are and send the `run.json` and the host terminal's last lines.

## 8. Stop everything

1. A running `run_big.py`: Ctrl+C. It stops its deployment and still writes `run.json` (status `interrupted`).
2. Host: Ctrl+C in the host terminal.
3. Each peer: Ctrl+C in the join terminal, then `uv run hypercompute leave`. A peer's agent also stops by itself
   5 minutes after the host is gone.
4. Every Mac: `kill %1` in the terminal that started `caffeinate` (or `killall caffeinate`).

## 9. Not verified

- Nothing in this runbook has run on a Mac host. The macOS monitor (`MacProbes`: vm_stat memory, swap and
  models-volume guards) is tested only with simulated `vm_stat` output and psutil readings (`tests/test_run_big.py`).
- How available RAM (free + file-backed pages) and swap behave on a 16 GiB Mac while tens of GiB of CPU experts page
  in from the GGUF, and whether other apps pushed to swap by that file cache trip the 2 GiB swap guard.
- How much the planner's free-memory input on a Mac (psutil: free + inactive pages) undercounts after a run; `purge`
  before each plan is the workaround until the inventory reads free + file-backed pages like the run monitor does.
- llama.cpp on the host Mac with Metal plus memory-mapped CPU experts at this size. The Mac did run three gpt-oss
  layers (MXFP4) on Metal as an RPC peer last night, so the Metal kernels exist; Metal as the host's own device with
  most experts file-backed on the CPU has not run.
- The hotspot throughput when one Mac sends layers to two others at once, and whether both peers get a direct path.
- The planner's Mac-host budgets (unified pool, Metal working-set cap) are being changed on this branch tonight; pull
  right before step 1.3 and use whatever `git log -1` says.
