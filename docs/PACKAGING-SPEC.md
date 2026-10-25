# Packaging spec: one-click join, Python SDK, bench CLI

This is the contract for three builders. Each builder owns one folder and follows the section for it:

| Folder | Owner | Section |
|---|---|---|
| `join/` | one-click scripts for friends and hosts | 2 |
| `sdk/python/` | `hypercompute-sdk`, import `hypercompute_sdk` | 3 |
| `bench/` | `bench/hcbench.py` and `bench/RESULTS.md` | 4 |

Everything here was read from the worktree `C:/Users/operator/.hypercompute/worktrees/iroh-transport`, branch
`codex/iroh-transport`, at HEAD `ad30dec`, on 2026-10-24. Section 5 lists every flag and route this spec relies on,
with the file and line it was read from. If the code moves, the code wins and the builder updates the line reference.

## 1. Rules for every builder

- A live benchmark campaign runs from this checkout. Do not edit anything under `backend/`, `frontend/`, `scripts/`,
  `configs/`, `integrations/` or `tests/`. Create files only in your own folder, plus a `README.md` in it for the
  integrator.
- Do not start a gateway, engine, join, model download or benchmark. Do not call `127.0.0.1:8001` or any LAN address.
  Tests must not open a socket.
- No network installs. Do not run `uv add`, `uv sync`, `pip install` or `uv pip install`. Use what the repo venv
  already has (httpx 0.28.1 per `uv.lock` line 222 to 223, pytest, ruff).
- Run tests with the repo venv:
  `cd C:/Users/operator/.hypercompute/worktrees/iroh-transport && uv run --no-sync python -m pytest <path> -q`.
  Passing a path overrides `testpaths = ["tests"]` in `pyproject.toml`, and `addopts = "-m 'not hardware'"` still
  applies. Test files need basenames that are unique across `tests/`, `join/tests/`, `sdk/python/tests/` and
  `bench/tests/` (no `__init__.py` in test folders, so pytest imports them by basename). Each test folder has its own
  `conftest.py` that puts its sources on `sys.path` and adds an autouse fixture that makes `socket.socket.connect` and
  `socket.create_connection` raise.
- Never print, log or embed an owner token, invite ticket or invite code. Scripts never echo an invite back in full.
  Show at most the last 4 characters of a ticket and nothing of a code.
- Docs use plain, short sentences. No em dashes, no hype. Every benchmark number comes from `docs/pitch/NUMBERS.md` or
  from the run indexes under `results/capacity/`, with its conditions next to it.
- Shell scripts (`.sh`, `.command`) are written with LF line endings. On this Windows host Python's text mode writes
  CRLF, so open files with `newline="\n"`. `.ps1` and `.cmd` files use CRLF. Do not commit; the integrator commits.

### Shared facts

| Fact | Value | Source |
|---|---|---|
| Gateway port used by packaging | `8001`, override with `HC_PORT` | `run_big.py:1243` and `matrix.py:42` assume 8001; the CLI default is 8000 (`config.py:9`), so packaging always passes `--port` |
| Worker port | 47052, scanned up to 24 ports | `config.py:13`, `config.py:14`, `cli.py:292` |
| Invite line printed by `hypercompute invite` | `uv run hypercompute join --ticket <TICKET> --code <CODE>`, then `Expires: <ISO time>` | `app.py:305`, `cli.py:523` to `524` |
| Invite lifetime | one use, 15 minutes | `config.py:19`, `sessions.py:111` |
| Code format | 10 characters from `ABCDEFGHJKLMNPQRSTUVWXYZ23456789`, shown as `XXXXX-XXXXX`; the host uppercases and drops non letters or digits before checking | `sessions.py:22`, `sessions.py:23`, `sessions.py:30` to `35` |
| Ticket | an Iroh `EndpointTicket` string; parsed by `iroh.EndpointTicket.from_string` | `transport/host.py:88` to `89`, `transport/iroh.py:416` to `420`, `transport/iroh.py:500` |
| Owner header | `X-Hypercompute-Owner` | `config.py:30` |
| Hypercompute home | `HYPERCOMPUTE_HOME`, default `~/.hypercompute`; agent state in `agent/`, tensor cache in `rpc-cache/` | `config.py:59` to `65`, `config.py:73` to `74` |
| uv location when not on PATH | `~/.local/bin/uv`, Windows `%USERPROFILE%\.local\bin\uv.exe` | `mac_join.sh:6`, `gx10/host.sh:3`, `gx10/join_g14.sh:14` |

## 2. `join/`: one-click scripts

### 2.1 Files

Exact names. Each OS has a thin wrapper per action. All logic after bootstrap lives in one Python helper, so the three
OSes behave the same and the logic is testable.

| Action | macOS (double-click) | Windows (double-click) | Linux |
|---|---|---|---|
| Share this computer | `join/hypercompute-join.command` | `join/hypercompute-join.cmd` calls `join/hypercompute-join.ps1` | `join/hypercompute-join.sh` |
| Stop sharing | `join/hypercompute-leave.command` | `join/hypercompute-leave.cmd` calls `join/hypercompute-leave.ps1` | `join/hypercompute-leave.sh` |
| Show status | `join/hypercompute-status.command` | `join/hypercompute-status.cmd` calls `join/hypercompute-status.ps1` | `join/hypercompute-status.sh` |
| Start the host | `join/hypercompute-host.command` | `join/hypercompute-host.cmd` calls `join/hypercompute-host.ps1` | `join/hypercompute-host.sh` |
| Make an invite (host) | `join/hypercompute-invite.command` | `join/hypercompute-invite.cmd` calls `join/hypercompute-invite.ps1` | `join/hypercompute-invite.sh` |

Shared files:

| File | Purpose |
|---|---|
| `join/lib/hc-common.sh` | bootstrap for `.command` and `.sh` (bash 3.2 compatible: macOS ships bash 3.2, so no `mapfile`, no `${x,,}`, no associative arrays) |
| `join/lib/hc-common.ps1` | bootstrap for `.ps1` (Windows PowerShell 5.1: no `&&`, no ternary, no `??`; do not use `2>&1` on native commands) |
| `join/lib/oneclick.py` | every step after bootstrap: `join`, `leave`, `status`, `host`, `invite`, and the internal `_supervise` |
| `join/.gitattributes` | `*.sh text eol=lf`, `*.command text eol=lf`, `*.ps1 text eol=crlf`, `*.cmd text eol=crlf` |
| `join/README.md` | the friend and host instructions of 2.8, plus a short note for the integrator |
| `join/tests/` | section 2.9 |

The integrator stages the `.command` and `.sh` files with `git add --chmod=+x`, because this Windows worktree cannot set
the executable bit and Finder will not run a `.command` without it.

### 2.2 Wrapper bootstrap (all OSes)

Every wrapper does these steps, then hands over to `oneclick.py`. The `.command` and `.sh` wrappers are three lines
plus the shebang: resolve their own folder, source `lib/hc-common.sh`, call `hc_main <action> "$@"`.

Before step 1, if any argument is `--ticket`, `--code`, a bare 30+ character token or a bare code, the bootstrap
prints "Run this with nothing after it, or only --dry-run. It asks you to paste the invite." and stops with exit
code 2, so an invite typed after the command never reaches uv or Python. `oneclick.py` also refuses any argument
it does not know without echoing it, and its argparse errors go through the redaction of 2.6.

1. **Find the checkout.** Use `HC_REPO` when set, else the parent of the wrapper's folder. Resolve the folder with
   `cd "$(dirname "$0")" && pwd` (Finder starts a `.command` with the home folder as the working directory; PowerShell
   uses `$PSScriptRoot`). Accept it only if `pyproject.toml` has the line `name = "hypercompute"` and
   `backend/hypercompute/cli.py` exists. Otherwise print "This folder is not inside a Hypercompute checkout" and stop.
2. **Find uv.** `command -v uv`, then `$HOME/.local/bin/uv`; on Windows `Get-Command uv`, then
   `$env:USERPROFILE\.local\bin\uv.exe`. If missing, print one line pointing at
   `https://docs.astral.sh/uv/getting-started/installation/` and stop. Never download or pipe an installer.
3. **Python environment.** If `<repo>/.venv` is missing, say that the first run downloads Python packages, and
   that uv also downloads Python 3.12 into its own folder outside the checkout when the computer has none (that
   copy stays after the folder is deleted). Then ask
   `Continue? [y/N]`, then run `uv sync --frozen` in the checkout. This is the only command that runs without
   `--no-sync`.
4. **Hand over.** `cd` into the checkout and run `uv run --no-sync python join/lib/oneclick.py <action> --repo <repo>`
   with the remaining arguments. Return its exit code.
5. **Keep the window open.** `.command` and `.cmd` end with "Press Return to close this window" (`read -r _`,
   `pause`), unless `HC_NO_PAUSE=1`. `.sh` pauses only when stdin is a terminal and `HC_NO_PAUSE` is unset.

`hypercompute-<action>.cmd` is exactly:

```bat
@echo off
setlocal
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0hypercompute-<action>.ps1" %*
set rc=%ERRORLEVEL%
if not "%HC_NO_PAUSE%"=="1" pause
exit /b %rc%
```

`-ExecutionPolicy Bypass` applies to this one process only. It changes no system setting. The full path to
`powershell.exe` matters: cmd.exe looks in the current folder before `PATH`, so a bare `powershell.exe` could run a
file of that name from wherever the terminal is.

### 2.3 `oneclick.py` conventions

- Standard library only, plus `httpx` from the venv for the two health checks. It runs the CLI as
  `[sys.executable, "-m", "hypercompute.cli", ...]` (the module runs `main()` under `__main__`, `cli.py:664` to `665`),
  so no PATH lookup is needed after bootstrap.
- It talks to the CLI through its public output only: `engine status` (JSON, exit 0 only when the smoke test passed),
  `inventory` (JSON), `doctor`, `join`, `leave`, `status`, `host`, `invite`. It does not import `hypercompute`
  internals.
- Its own files live in `<HYPERCOMPUTE_HOME>/oneclick/`: `join.log`, `join.log.1`, `host.log`, `host.log.1`,
  `join.pid`, `host.pid`, and the invite fallback file `invite.txt`. The folder is created with mode 0700 on POSIX.
- Options: `--repo PATH`, `--port N` (default `HC_PORT` or 8001), `--dry-run` (print the composed command with the ticket
  and code masked, launch nothing; tests use it).
- Environment overrides, for operators and tests. Each one replaces a prompt, and the consent screen still shows the
  value: `HC_INVITE` (the invite line; never echoed), `HC_NAME`, `HC_DEVICE`, `HC_MAX_MEMORY`, `HC_RESERVE`,
  `HC_CACHE` (0 or 1), `HC_MANAGED` (0 or 1), `HC_MAX_DISK`, `HC_RELAY_URL`, `HC_LOCAL_MAX_MEMORY`,
  `HC_LOCAL_DEVICES`. `HC_ONECLICK_YES=1` skips the final consent question; it is for scripted runs, and README says so.
- Every prompt has a default shown in brackets. Enter accepts it.

### 2.4 `oneclick.py join`, step by step

1. **Already sharing?** Run `hypercompute status --port <port>`. If its output starts with `Sharing as a friend node:`
   (`cli.py:427`), ask `This computer is already sharing. Stop and join again? [y/N]`. Yes runs `hypercompute leave`
   first. Never `pkill`; `leave` asks the running agent to stop its own worker (`cli.py:342` to `356`). A second agent is
   refused anyway by the agent lock (`cli.py:228` to `229`).
2. **Engine.** Run `hypercompute engine status`. Exit 0 means the pinned llama.cpp passed its smoke test
   (`cli.py:60` to `64`). Otherwise say what failed and run `hypercompute engine install` in the foreground with the
   terminal attached and without `--yes`, so the CLI's own prompt shows each file, size and sha256 and asks
   `Download and install? [y/N]` (`cli.py:76` to `82`). Check `engine status` again and stop if it still fails.
3. **Doctor, for information.** Run `hypercompute doctor` with output shown. Do not gate on its exit code: its LAN and
   worker port checks (`cli.py:477` to `488`) do not apply to Iroh, where the worker listens on 127.0.0.1
   (`cli.py:291`). The engine check that matters already passed in step 2.
4. **Invite.** Prompt `Paste the invite line from the host, then press Enter` and read it with
   `getpass.getpass` so the pasted text is not shown. Then parse:
   - Reject input over 8 KiB.
   - If it contains `--host`, stop with "This is a LAN invite. Ask the host for a new invite from hypercompute-invite."
     The LAN form is `uv run hypercompute join --host <ip:port> --code <code>` (`sessions.py:119` to `120`).
   - If it contains `--ticket`, the ticket is the next token and the code is the token after `--code`.
   - Else, if it is exactly two tokens, the one that matches the code shape is the code and the other is the ticket.
   - Code: uppercase, drop every character that is not a letter or digit, require exactly 10 characters from the code
     alphabet, format as `XXXXX-XXXXX`.
   - Ticket: require `^[A-Za-z0-9]{30,4096}$`. The 30 character floor also guarantees the log filter in 2.6 masks it.
     Then, if `import iroh` works, call `iroh.EndpointTicket.from_string(ticket)` as a local parse check. It opens no
     connection.
   - On failure print "That does not look like a Hypercompute invite" and ask again, at most 3 times.
   - On success print `Invite read: ticket ...<last 4> (<length> characters), code hidden.`
5. **Inventory.** Run `hypercompute inventory` and parse the JSON. Fields used: `hostname`, `os.system`
   (`windows`, `macos`, `linux`), `os.machine` (`arm64`, `x86_64`), `memory_pools[].{id, kind, total_bytes,
   free_bytes}`, `devices[].{id, pool_id, total_bytes}`, `disk_free_bytes`, `engine.{release, variant, devices[].name}`
   (`schemas.py:48` to `142`).
6. **Pick the device.**
   - `os.system == "macos"` and `os.machine == "arm64"` and `MTL0` is in `engine.devices[].name`: `MTL0`.
   - Else, `nvidia-smi -L` exits 0 within 15 s and prints a line starting with `GPU 0:`, and `CUDA0` is in
     `engine.devices[].name`: `CUDA0`.
   - Else `CPU`. Say "No supported GPU found. Sharing the CPU is slow." This matches the agent's own CPU fallback note
     (`join.py:708`).
   - If `nvidia-smi` works but the engine has no `CUDA0`, say the installed engine variant (`engine.variant`) cannot see
     the GPU, and offer CPU only after a yes. Before that, read the NVIDIA device's `driver_version` major. The CUDA
     builds need 580 or newer on Windows x86_64 and 570 or newer on Linux x86_64 (`engine.py` `choose_variant`; Linux
     arm64 has no minimum there). Below that minimum, say the driver is too old, name the minimum, and suggest
     `hypercompute engine install` after the update, because a forced reinstall picks the CPU build again. Otherwise,
     or when the version cannot be read, suggest `hypercompute engine install --force`.
   - `HC_DEVICE` overrides the choice (for example `Vulkan0`, or `CUDA1`). It must be in `engine.devices[].name` or be
     `CPU`.
   - One device only. The agent would share every GPU by default (`contribution.py:15` to `18`), but one device keeps
     the consent screen simple.
7. **Pick a safe memory cap.** Find the device's pool through `devices[].pool_id` (CPU uses the `ram` or `uma` pool).
   Work in bytes, then round down to whole GiB.
   - Unified pool (`MTL0`, and the CPU on a unified machine): `min(0.65 x total, total - 6 GiB)`. This is the agent's
     own default (`contribution.py:41` to `43`). A 16 GiB MacBook Air gets 10 GiB, the default of `mac_join.sh:7`.
   - Discrete VRAM (`CUDA0`): `0.80 x VRAM total`. This is lower than the agent default of total minus 512 MiB
     (`contribution.py:38` to `40`) because the laptop GPU also drives the display. The G14 (11.94 GiB VRAM, NUMBERS
     condition L7) gets 9 GiB, the cap it used in every GX10 run (`gx10/join_g14.sh:15`, NUMBERS condition G6).
   - System RAM (`CPU` on a non-unified machine): `min(0.40 x total, total - 8 GiB)`, the agent default
     (`contribution.py:44` to `46`).
   - Reserve: `max(4 GiB, 0.20 x RAM pool total)`, the agent default (`contribution.py:21` to `24`), rounded down.
   - Below 1 GiB: stop with "This computer does not have enough memory to share." (the CLI only warns,
     `cli.py:302` to `303`).
   - Prompt `Memory the host may use [10 GiB]`. Accept a whole number of GiB from 1 up to the pool total. Warn when the
     answer is above the computed default.
8. **Name.** Prompt `Name shown to the host [<hostname>]`. Strip control characters. 1 to 64 characters
   (`schemas.py:165`).
9. **Cache.** Prompt `Keep received model pieces on disk so the next load is quicker? [Y/n]`. Yes adds `--cache`
   (`cli.py:571`). Both campaign join scripts use it (`mac_join.sh:13`, `gx10/join_g14.sh:15`). Say the cache lives in
   `<HYPERCOMPUTE_HOME>/rpc-cache` and stays after leaving (`join.py:720`).
10. **Managed, opt in.** Prompt
    `Let the host download this computer's share of a model from huggingface.co into that cache (up to N GiB)? [y/N]`.
    N defaults to `min(80, floor(disk_free_gib - 30))`. If N would be under 10, skip the question. Yes adds
    `--managed --devices-allowed <DEVICE> --max-disk <N>GiB` (`cli.py:573` to `579`). Allow only the chosen device, the
    narrowest consent. The CLI keeps at least 20 GiB free on that disk (`cli.py:578` to `579`). Benchmark rows with
    `--seed-peers` need peers joined this way (`run_big.py:1264` to `1266`). Make no speed claim in the prompt.
    `--managed` grants more than downloads (`agent/managed.py` `consent_lines`, `agent/join.py` `print_plan`), so say
    so right before the question: the host may also restart this computer's worker on the chosen device and change
    its CPU thread count, from 1 up to all cores (`os.cpu_count()`); and if the host goes away, the computer keeps
    waiting for it for up to 2 hours, and a crashed worker is started again (at most 3 times in 30 minutes).
11. **Consent screen.** Print the name, device, memory cap, reserve, the pool's total and free memory, cache on or off and
    its path, managed on or off and the disk cap, and the ticket tail. With managed on, also print the two terms of
    step 10 (worker restarts and thread changes; the 2 hour wait and the crash restarts), because the CLI prints its
    own consent text only into `join.log`, after this question. Repeat the agent's warning: "Only pair with a host
    you trust" (`join.py:715`). Ask `Share this computer now? [y/N]`. Only a yes continues. The join itself gets `--yes`,
    because a background process has no terminal to prompt on (`join.py:818` to `820`).
12. **Compose the command**, in this order:

    ```
    <python> -m hypercompute.cli join --ticket <TICKET> --code <CODE> --name <NAME> --devices <DEVICE>
      --max-memory <N>GiB --reserve <R>GiB [--cache] --yes
      [--managed --devices-allowed <DEVICE> --max-disk <D>GiB] [--relay-url <HC_RELAY_URL>]
    ```

    Never add `--exo` or `--advertise-ip`: Iroh joins refuse them (`cli.py:262` to `263`). Never set `HC_PEER_MMAP`,
    `HC_PEER_CACHE_MAX_GIB` or `HC_PEER_CACHE_MIN_FREE_GIB`; the join refuses inherited limits (`cli.py:315` to `321`,
    `hcpeer.py:18` to `20`). With `--dry-run`, print the command with the ticket shown as `<ticket>` and the code as
    `<code>`, and stop.
13. **Run it in the background.** See 2.6. Rotate `join.log` to `join.log.1` first.
14. **Wait for the outcome.** Read the redacted log for up to 120 s.
    - A line starting `Joined.` (`join.py:833`) means success.
    - A line starting `error:` (`cli.py:47`) means failure. Print it and the next `fix:` line (`cli.py:49`), then exit 1.
      Typical ones: engine mismatch with the host (`cli.py:287` to `290`), Iroh connection failed (`cli.py:273` to
      `275`), invite expired or used.
    - If the supervisor exits first, print the last 15 log lines and exit 1.
    - On timeout, say it is still connecting and point at `hypercompute-status`.
15. **Show status.** Run `hypercompute status --port <port>` (`cli.py:422` to `457`) and print it. End with
    `Sharing. You can close this window. To stop, open hypercompute-leave.`

### 2.5 The other actions

**`leave`.** Compute the cache size of `<HYPERCOMPUTE_HOME>/rpc-cache` (sum of file sizes). Ask
`Also delete the model cache (X GiB)? [y/N]`. Run `hypercompute leave`, adding `--purge-cache` on yes (`cli.py:597` to
`599`). Print its output. If it exits 1 with the message about a worker that could not be stopped (`cli.py:396` to
`399`), print its fix line, which says to run it again as Administrator or with sudo. After a clean leave, remove
`join.pid`.

**`status`.** Run `hypercompute status --port <port>` and print it. When it says no host is running, add the port it
checked and that `HC_PORT` changes it, because the CLI's own default is 8000. Then print the last 10 lines of
`join.log` if it exists, and `host.log` if it exists. Both are already redacted.

**`host`.**
1. `GET http://127.0.0.1:<port>/api/health` with a 3 s timeout and `trust_env=False`. If it answers, say a host is
   already running, open `http://127.0.0.1:<port>/`, and go to the invite step. The host lock refuses a second host
   anyway (`cli.py:177` to `178`).
2. Engine check as in join step 2.
3. Compose `<python> -m hypercompute.cli host --transport iroh --port <port> --no-browser`, plus
   `--local-max-memory <HC_LOCAL_MAX_MEMORY>` and `--local-devices <HC_LOCAL_DEVICES>` when set (`cli.py:548` to `557`).
   `HC_LOCAL_MAX_MEMORY` needs a unit that `units.parse_size` knows (KB to TB, KiB to TiB, or a bare K, M, G or T),
   so `8Gi` or a bare `8` (8 bytes) stops with a message instead of a traceback in `host.log`.
   Do not pass `--bind` or `--relay-url` (`cli.py:192` to `195`). The environment passes through unchanged, so a
   source build of the engine on the GX10 keeps working through `HYPERCOMPUTE_LLAMA_BIN_DIR` (`gx10/host.sh:4`).
4. Run it in the background as in 2.6, with `host.log`. The host prints a live invite at startup (`app.py:351` to
   `353`), so the log filter matters here too.
5. Poll `/api/health` every second for up to 120 s. The CLI warns startup takes about 10 s for the GPU probe
   (`cli.py:197`), and the Iroh endpoint starts before the gateway serves (`app.py:344` to `353`).
6. Open the dashboard with `webbrowser.open("http://127.0.0.1:<port>/")`. The dashboard is local only (`cli.py:207` to
   `208`).
7. Run the invite step.

**`invite`.**
1. Run `hypercompute invite --port <port>` (`cli.py:588` to `590`) with stdout captured and never printed. It fetches the
   owner token from `/api/owner/bootstrap` and posts `/api/network/invite` (`cli.py:512` to `528`).
2. On a non-zero exit, print only its `error:` and `fix:` lines. They contain no secret. Then say which port was
   checked: `Looked for a host on 127.0.0.1:<port>. Start it with hypercompute-host, or set HC_PORT to the port your
   host uses.` The CLI's fix line starts a host on its default port 8000, which these scripts do not check.
3. Take the first line containing `--ticket` (as `gx10/join_g14.sh:8` does) and the `Expires:` line.
4. Copy the line to the clipboard by writing it to the tool's stdin, never as an argument: `pbcopy` on macOS;
   `clip.exe` on Windows; `wl-copy`, then `xclip -selection clipboard`, then `xsel --clipboard --input` on Linux.
5. With no clipboard tool, write it to `<HYPERCOMPUTE_HOME>/oneclick/invite.txt`, created with
   `os.open(..., O_WRONLY | O_CREAT | O_TRUNC, 0o600)`, and print the path only.
6. Print `Invite copied. Send it to one friend in a private message. It works once and expires at <time> (15 minutes).`
   and `Ticket ...<last 4>, code hidden.`

### 2.6 Background run, keep awake, log redaction

`oneclick.py` starts itself as a detached supervisor: `[sys.executable, oneclick.py, "_supervise", "--log", LOG,
"--pidfile", PID, "--", <command>...]`.

- POSIX: `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)`. The new session
  means closing Terminal does not hang it up, which is what `nohup` did in `mac_join.sh:12`.
- Windows: `creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP`, the same stdio.
- The ticket and code reach the supervisor in its environment (`HC_ONECLICK_TICKET`, `HC_ONECLICK_CODE`). The supervisor
  removes them from its own environment before it starts the child. The child gets them as arguments, because the CLI
  only takes flags, so other accounts on the same computer can read them in the process list while it runs. The
  code works once and is spent at join; if the join fails before that, it stays valid until the invite expires.
  The dry run and `join/README.md` say this. A follow-up outside `join/` could let `hypercompute join` read them
  from the environment or stdin.

The supervisor:

1. Writes `{"supervisor_pid", "child_pid", "started_at"}` to the pid file.
2. Keeps the machine awake while the child runs, without keeping the display on, so the screen can still turn
   off and lock. macOS: `caffeinate -ims -w <child_pid>` (no `-d` and no `-u`; `mac_join.sh:12` uses `-dimsu`, which
   suits its headless SSH user but not a friend's laptop). Linux: run the child under `systemd-inhibit --what=sleep
   --who=hypercompute --why=sharing --mode=block` when that tool works for this user (no idle inhibitor, which some
   desktops treat as "do not lock"). Windows: `ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 |
   0x00000001)` in the supervisor, with no display flag. It lasts only while the process lives and changes no power
   setting. The consent screen says "Keep awake: yes, while sharing (the screen can still turn off and lock)".
3. Starts the child with stdin closed and stdout and stderr merged into a pipe, reads it line by line, redacts each line
   (below), writes it to the log, and flushes after every line.
4. When the child exits, writes `hypercompute exited with code N`, removes the pid file and exits with that code.

Redaction, applied in this order to every line written to any log or printed by `oneclick.py`:

1. The exact ticket string, then the exact code in the forms pasted, `XXXXX-XXXXX` and `XXXXXXXXXX`, become `<ticket>`
   and `<code>`.
2. `[A-Za-z0-9_-]{30,}` becomes `<r>` (the pattern of `gx10/join_g14.sh:16`). This also masks node ids and hashes.
   That is accepted.
3. `\b[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}\b` becomes `<code>`. The script's version of the code pattern uses the real
   code alphabet so ordinary hyphenated words survive.

### 2.7 Security notes for README

- The llama.cpp RPC worker has no authentication. Over Iroh it listens on 127.0.0.1 and the host reaches it through
  the encrypted tunnel (`cli.py:291`, `join.py:712` to `715`). Only pair with a host you trust.
- An invite works once, for 15 minutes. Send it in a private message, one friend per invite.
- Windows may need an Administrator PowerShell to stop a stuck worker (`cli.py:398`).
- The folder is code that runs as the friend. The scripts are not signed, so Windows SmartScreen and macOS Gatekeeper
  warn about them, and the `.cmd` bypasses the PowerShell execution policy for its own run. Only use a folder that
  came straight from a host you trust. When the zip goes through a shared or public link, the host sends its SHA-256
  in a separate private message and the friend checks it (`Get-FileHash`, `shasum -a 256`) before unzipping.
- With host downloads on, the host may also restart the worker and change its CPU threads, and an unattended
  computer waits up to 2 hours for a host that went away (2.4 step 10).

### 2.8 How people use it (text for `join/README.md`)

For a friend:

1. Get the Hypercompute folder from a host you trust and put it anywhere. It holds programs that run as you.
2. Install uv once: https://docs.astral.sh/uv/getting-started/installation/
3. Open the `join` folder. On a Mac, double-click `hypercompute-join.command`. If macOS says it cannot open it,
   right-click it and choose Open once. On Windows, double-click `hypercompute-join.cmd`. On Linux, run
   `./join/hypercompute-join.sh` in a terminal.
4. The first run asks to download Python packages and the inference engine. Say yes.
5. Paste the invite line the host sent you and press Enter. The text stays hidden while you paste.
6. Check the name, memory and disk it shows. Press Enter to accept or type a smaller number. Type `y` to share.
7. When it says "Sharing", you can close the window. To stop, double-click `hypercompute-leave`.

For the host:

1. Double-click `hypercompute-host`. The dashboard opens in your browser.
2. Double-click `hypercompute-invite`. The invite is now on your clipboard.
3. Paste it to one friend. Make a new invite for each friend.

### 2.9 Tests (`join/tests/`)

Run `uv run --no-sync python -m pytest join/tests -q`. Suggested files: `test_oneclick_parse.py`,
`test_oneclick_device.py`, `test_oneclick_supervise.py`, `test_join_wrappers.py`.

- Invite parsing: the full `uv run hypercompute join --ticket T --code C` line, two bare tokens in either order, a
  lowercase code, a code without its dash, a missing code, a LAN `--host` invite, a 20 character ticket, a 9 KiB paste.
  Use synthetic tickets only. Never paste a real one into a test.
- Device and cap: inventory fixtures written by hand from the schema fields of step 5.
  - A 16 GiB M4 Mac with `MTL0` gives `MTL0`, 10 GiB cap, 4 GiB reserve.
  - A G14 (`CUDA0`, 11.94 GiB VRAM, 31.12 GiB RAM, NUMBERS L7) with `nvidia-smi` stubbed to succeed gives `CUDA0`,
    9 GiB cap, 6 GiB reserve.
  - A 32 GiB Linux box without a GPU gives `CPU`, 12 GiB cap, 6 GiB reserve.
  - `nvidia-smi` working while the engine lists no `CUDA0` gives the variant warning.
  - An 8 GiB CPU-only machine is refused.
- Command composition: `--dry-run` output for each branch (cache off, managed on, relay URL) matches 2.4 step 12 with
  the ticket and code masked.
- Redaction: lines holding the ticket, the code in each form, a 64 character hash and the word `multi-level`.
- Supervisor: run `_supervise` on a child `[sys.executable, "-c", "print(<synthetic ticket>)"]`. Check the log holds
  `<ticket>` and not the ticket, that the exit line is written and that the pid file is removed.
- Wrappers: no `\r` in any `.sh` or `.command` file. `bash -n` passes on each (skip when `bash` is not on PATH). On
  Windows, each `.ps1` parses with `[System.Management.Automation.Language.Parser]::ParseFile` with zero errors. Each
  `.cmd` matches the template of 2.2.
- No test runs `hypercompute join`, `host`, `engine install`, `uv sync` or a real clipboard tool. Stub them by
  injecting the command runner.

## 3. `sdk/python/`: `hypercompute-sdk`

### 3.1 Layout

```
sdk/python/pyproject.toml          name = "hypercompute-sdk", version 0.1.0, requires-python >=3.11,
                                   dependencies = ["httpx>=0.27"], build backend hatchling,
                                   [tool.hatch.build.targets.wheel] packages = ["src/hypercompute_sdk"]
sdk/python/README.md
sdk/python/src/hypercompute_sdk/__init__.py    exports Pool, ChatChunk, every error, __version__
sdk/python/src/hypercompute_sdk/client.py      Pool
sdk/python/src/hypercompute_sdk/errors.py
sdk/python/src/hypercompute_sdk/sse.py         SSE line parser
sdk/python/tests/conftest.py                   puts sdk/python/src on sys.path, blocks sockets
sdk/python/tests/fake_gateway.py               the MockTransport handler of 3.6
sdk/python/tests/test_sdk_*.py
```

The package is not installed into the repo venv (no network installs). Tests import it through `conftest.py`.
Methods return the gateway's JSON as plain dicts and lists, so `schemas.py` stays the only schema. The SDK adds one
small type, `ChatChunk`.

### 3.2 Connection and the owner token

- `Pool(base_url="http://127.0.0.1:8001", *, owner_token=None, timeout=30.0, transport=None, sleep=time.sleep,
  clock=time.monotonic)`. `transport` is passed to `httpx.Client` and is how tests inject `httpx.MockTransport`.
  `Pool.from_env()` reads `HYPERCOMPUTE_URL` and `HYPERCOMPUTE_OWNER_TOKEN`. `Pool` is a context manager and has
  `close()`.
- The client is `httpx.Client(base_url=..., trust_env=False, ...)`, as the CLI and runner do (`cli.py:516`,
  `run_big.py:1303`), so a proxy setting cannot capture loopback traffic.
- **Loopback only.** Owner routes check that the TCP client is `127.0.0.1` or `::1` and that the `Host` header's name
  is `127.0.0.1`, `localhost` or `[::1]` (`app.py:64` to `65`, `app.py:383` to `388`). The constructor raises
  `ValueError` when `base_url` is not `http://` on one of those names, with this fix: run on the host, or tunnel with
  `ssh -L 8001:127.0.0.1:8001 <host>` and keep `base_url` on 127.0.0.1. Through such a tunnel the gateway sees a
  loopback client and a loopback `Host`.
- **Reads** (GET owner routes, `OwnerRead`, `app.py:395`) need no token.
- **Writes** (`OwnerWrite`, `app.py:396`) need `X-Hypercompute-Owner: <token>`, compared with `hmac.compare_digest`
  (`app.py:390` to `393`). That covers `POST /api/plans`, `POST /api/deployments`, `DELETE /api/deployments/{id}` and
  `POST /v1/chat/completions`.
- **Where the token comes from**, first match wins:
  1. the `owner_token=` argument;
  2. `HYPERCOMPUTE_OWNER_TOKEN`;
  3. `GET /api/owner/bootstrap`, fetched lazily before the first write. The reply is
     `{"owner_token", "version", "gateway_port", "lan_addresses"}` (`app.py:443` to `447`).
- The gateway makes a new token every time the host starts (`secrets.token_urlsafe(32)`, `coordinator.py:146`). A
  restarted host answers 403 `owner_only` "missing or wrong owner token" (`app.py:393`, `app.py:399` to `401`). When the
  token came from bootstrap, the SDK bootstraps again and retries that request once. A token from the argument or the
  environment is never replaced; the SDK raises `OwnerAuthError`. Each retry is safe: the 403 comes from the auth
  dependency before the handler runs, and deployment creates carry an `Idempotency-Key`.
- The token never appears in `repr()`, `str()`, a log line or an exception message. `repr(pool)` shows the base URL and
  the token source (`argument`, `environment`, `bootstrap`, `none yet`).

### 3.3 `Pool` methods

| Method | Route(s) | Notes |
|---|---|---|
| `health()` | `GET /api/health` (`app.py:432`) | full engine info only from loopback (`app.py:436` to `441`) |
| `status()` | `GET /api/health`, `/api/network`, `/api/nodes`, `/api/deployments`, `/v1/models` (`app.py:432`, `449`, `483`, `590`, `720`) | returns `{"version", "engine_release", "transport", "connections", "nodes": [{id, display_name, state, heartbeat_age_s, max_bytes}], "deployment": latest or None, "ready_model": id or None}`, the same facts `hypercompute status` prints (`cli.py:439` to `456`). Drop `endpoint_id`, `peer_id` and `path.remote_id` from the network part, as `matrix.py:48` to `56` does |
| `nodes()` | `GET /api/nodes` (`app.py:483`) | includes the host itself with `is_local: true` |
| `peers()` | `GET /api/peers` (`app.py:521`) | each peer's consent, worker devices, cache bytes and seed progress |
| `models()` | `GET /api/models` (`app.py:569`) | `ModelView` list; use `artifact.id` and `status` |
| `served_models()` | `GET /v1/models` (`app.py:720`) | ids of the ready deployment, never a tuner trial (`app.py:724`) |
| `pools()` | `GET /api/capacity/pools` (`app.py:620`) | |
| `explain(model_id, *, ctx_size=4096, prompt_tokens=512, output_tokens=256, warm=False, slots=2)` | `GET /api/capacity/explain` (`app.py:624` to `637`) | check locally: ctx 256 to 262144, prompt 0 to 262144, output 1 to 262144, slots 1 to 8, prompt + output <= ctx (`app.py:625` to `635`). Timeout 180 s, as `run_big.py:372` |
| `plan(model_id, *, strategy="all_selected", ctx_size=4096, node_ids=None, allow_ram_tier=True, allow_capacity_mode=True, engine=None, objective=None, overflow=None, use_profiles=None, parallel=1, layout=None)` | `POST /api/plans` (`app.py:581`) | sends only the keys given, because `PlanRequest` forbids extra fields (`schemas.py:21` to `22`, `schemas.py:482` to `500`). Check locally: strategy in `auto, all_selected, fastest_fit, local_only` (`schemas.py:463` to `467`), ctx 256 to 262144, parallel 1 to 16, engine in `llama_rpc, exo, airllm, vllm` (`schemas.py:26`). Timeout 180 s (`run_big.py:1020`) |
| `deploy(plan_id=None, *, plan=None, idempotency_key=None)` | `POST /api/deployments` (`app.py:585` to `588`) | exactly one of `plan_id` or `plan` (`schemas.py:732` to `735`; the server says `plan_required` otherwise, `deployments.py:496`). Always sends an `Idempotency-Key` (uuid4 hex when not given). On a transport timeout, look the deployment up by that key in `GET /api/deployments` (`idempotency_key`, `schemas.py:707`), as `run_big.py:384` to `390` does |
| `deployments()`, `deployment(dep_id)` | `GET /api/deployments`, `GET /api/deployments/{id}` (`app.py:590`, `594`) | |
| `wait_ready(dep_id, *, timeout=7200.0, poll=2.0, on_progress=None)` | polls `GET /api/deployments/{id}` | returns the deployment at `ready`. Raises `DeploymentFailed` on `failed` or `stopped` (`schemas.py:656` to `668`) and `ReadyTimeout` at the deadline. Up to 5 failed polls in a row are tolerated (`run_big.py:396` to `404`). Defaults follow `run_big.py:152` and `run_big.py:1253`. `on_progress(state, progress)` is called when either changes |
| `deploy_model(model_id, *, wait=True, **plan_options)` | `plan`, then `deploy(plan_id)`, then `wait_ready` | on 409 `plan_stale` (plans expire after 10 minutes, `deployments.py:472` to `473`) it plans once more. A 409 `deployment_active` (`deployments.py:489`) is raised as is. It never stops someone else's deployment |
| `chat(messages, *, model=None, max_tokens=None, temperature=None, **extra)` | `POST /v1/chat/completions` with `stream: false` (`app.py:729` to `750`) | `messages` must be a non-empty list of dicts with `role` and `content` (`app.py:735` to `736`). `model`, when given, must be the ready model, else 404 `model_not_found` (`app.py:738` to `742`). `extra` passes llama.cpp fields such as `seed`, `cache_prompt`, `ignore_eos` (`run_big.py:104`). Refuse a body over 4096 KiB before sending (`app.py:70`, `app.py:167` to `168`). Read timeout 1800 s, as `run_big.py:1254` |
| `stream(messages, ...)` | the same route with `stream: true` (`app.py:743` to `745`) | returns an iterator of `ChatChunk`; see 3.4 |
| `stop(dep_id=None)` | `DELETE /api/deployments/{id}` (`app.py:598` to `600`) | `None` means the one non-terminal deployment from `GET /api/deployments`; with none, return `None`. Timeout 300 s, as `run_big.py:432` |

Out of scope for v1: sessions and invites (they hand out secrets; the host uses `hypercompute-invite`), node leave
and verify, managed peer worker and seed routes, model downloads, tuning, replan, benchmarks and events.

### 3.4 Streaming

- Open with `client.stream("POST", "/v1/chat/completions", json=body, headers=owner,
  timeout=httpx.Timeout(30.0, read=1800.0))`, as `run_big.py:705` to `706` does. A non-200 status is read in full and
  raised as the mapped error (`run_big.py:707` to `709`).
- The gateway forwards the engine's SSE events as they come (`streaming.py:457` to `462`). Each event is `data: <json>`
  followed by a blank line. Lines starting with `:` are comments. `data: [DONE]` ends the stream
  (`adapters/llama_rpc.py:409` to `410`).
- A payload with a top-level `error` ends the stream with an error. Code `interrupted` raises `ChatInterrupted`, which
  carries `run_id`, `partial_text` and `partial_truncated` when present (`streaming.py:682` to `689`). Other codes
  (`timeout`, `busy`, `upstream_error`, or a deployment error from slot acquisition) raise `StreamError`
  (`streaming.py:470` to `486`, `streaming.py:496` to `500`, `streaming.py:678` to `679`).
- `ChatChunk` is a frozen dataclass: `content` (from `choices[0].delta.content`, may be empty), `reasoning` (from
  `choices[0].delta.reasoning_content`), `finish_reason`, `timings` (llama.cpp's `timings` object when present),
  `usage`, and `raw` (the parsed payload).
- A stream that closes before `[DONE]` and without a `finish_reason` raises `StreamError` with code `incomplete`.
- Leaving the iterator early closes the response. The gateway then closes the engine stream so generation stops
  (`streaming.py:454` to `456`).

### 3.5 Errors

Error bodies come in three shapes. Parse them the way `run_big.py:182` to `198` does:

- `{"detail": str, "code": str, "fix": str?}` from `_error` (`app.py:137` to `141`);
- FastAPI validation, `{"detail": [{"loc", "msg", "type"}, ...]}`. Keep each `loc` joined with dots;
- `{"error": {"code", "message", "type"?}}` from chat (`streaming.py:575`, `app.py:748` to `749`).

Every error derives from `HypercomputeError(status, code, detail, fix)`. Map by HTTP status, keeping the server code:

| Class | When | Server codes seen in the code |
|---|---|---|
| `GatewayUnreachable` | connection refused, DNS, read timeout; `status = 0`, `code = "unreachable"` | |
| `RequestError` | 400 and 422 | `bad_json`, `bad_request` (`app.py:734`, `736`), `model_error` (`app.py:415` to `417`), validation lists, `plan_required`, `plan_not_feasible` (`deployments.py:496`, `498`), `workload_exceeds_context` (`app.py:633`), capacity request codes (`app.py:95` to `107`) |
| `OwnerAuthError` | 403 `owner_only` | not loopback, or a missing or wrong token (`app.py:388`, `393`) |
| `NotFoundError` | 404 | `not_found` (`deployments.py:340`), `plan_not_found` (`deployments.py:471`), `model_not_found` (`app.py:740`) |
| `ConflictError` | 409 | `deployment_active` (`deployments.py:489`), `plan_stale` (`deployments.py:473`), `deployment_replacing` (`deployments.py:479`), `tuning` (`deployments.py:358`), `host_ineligible` (`app.py:117`), `replan_disabled` (`app.py:648`) |
| `BodyTooLarge` | 413 `body_too_large` (`app.py:184` to `185`), or the local size check | |
| `EngineError` | 500 and 502 other than `interrupted` | `engine_error` (`app.py:423` to `425`), adapter codes (`app.py:419` to `421`), `candidate` (`app.py:119`) |
| `ChatInterrupted` | 502 `interrupted`, or the SSE event of 3.4 | `app.py:748` to `749`, `streaming.py:97` to `102` |
| `UnavailableError` | 503 | `no_ready_deployment` (`deployments.py:355`), `capacity_unavailable` (`app.py:612`), capacity state codes (`app.py:109` to `115`) |
| `StreamError` | an SSE error event other than `interrupted`, or `incomplete` | |
| `DeploymentFailed` | `wait_ready` saw `failed` or `stopped`; has `.deployment` with its `error` and `history` | |
| `ReadyTimeout` | `wait_ready` deadline; has `.deployment`, the last one seen | |

Any other status raises `HypercomputeError` itself. `str(err)` reads like `HTTP 409 plan_stale: plan is older than 10
minutes; check the plan again`, with the fix on a second line when present.

### 3.6 Tests (`sdk/python/tests/`)

Run `uv run --no-sync python -m pytest sdk/python/tests -q`. Everything goes through `httpx.MockTransport`. No socket
opens: the conftest guard fails the test if one does.

`fake_gateway.py` holds a small in-memory gateway as a handler function:

- It checks the `X-Hypercompute-Owner` header on the four write routes and answers 403 `owner_only` in the gateway's
  shape. It can rotate its token to imitate a host restart.
- It runs a deployment state machine: `planned` to `loading` to `ready` over a set number of polls, or to `failed` with
  an `error`.
- It returns plans that turn `plan_stale` on demand, and a FastAPI-style 422 list.
- It returns canned SSE bodies: normal chunks ending in `data: [DONE]`, an `interrupted` error event with
  `partial_text`, a `busy` error event, and a body cut before `[DONE]`.

Cases to cover:

- Token source precedence. Bootstrap is lazy and runs once. A rotated token is fetched again and the request retried
  once, but only when the token came from bootstrap. An argument token gives `OwnerAuthError`.
- The token never appears in `repr(pool)` or in `str()` of any raised error. Run every error path and search the text.
- `ValueError` for `http://192.168.1.20:8001`, `https://127.0.0.1:8001` and `http://example.com`.
- Each mapping row of 3.5, from canned responses.
- `deploy` always sends an `Idempotency-Key`. `deploy_model` plans again after `plan_stale`. `wait_ready` tolerates 4
  failed polls and raises on the 5th, raises `DeploymentFailed` and `ReadyTimeout` (with an injected clock and a no-op
  sleep, so no test waits), and calls `on_progress` once per change.
- `chat` refuses an empty `messages` list and a body over 4096 KiB without sending.
- `stream` yields chunks in order with `timings` on the last one, raises `ChatInterrupted` with `partial_text`, raises
  `StreamError` for `busy` and for a cut stream, and closes the response when the caller breaks early.
- `stop()` with no active deployment returns `None`.
- A route contract test reads `backend/hypercompute/app.py` as text and checks that every method and path the SDK uses
  appears as `@app.<method>("<path>"`. It does not import the app.

## 4. `bench/`: `hcbench.py` and `RESULTS.md`

### 4.1 Layout

```
bench/hcbench.py          the CLI (split helpers into bench/hcbench_report.py if it passes about 400 lines)
bench/pairs.json          the headline pairs of 4.5
bench/RESULTS.md          generated by `hcbench.py report`; never edited by hand
bench/README.md
bench/tests/conftest.py   puts bench/ and scripts/capacity/ on sys.path, blocks sockets
bench/tests/test_hcbench_*.py, bench/tests/fixtures/
```

Run every command with the repo venv from the repo root, for example
`uv run --no-sync python bench/hcbench.py report`. Run artifacts stay where they are today: plans and indexes in
`results/capacity/paper-<date>-<name>/`, run records in `results/capacity/<UTC stamp>-big-<model>/run.json`
(`run_big.py:43`).

`hcbench` imports `scripts/capacity/batchlib.py` (standard library only) for `batches_of`, `concurrency_of` and
`summarize`, the way `summarize.py:19` to `24` does. It does not import `summarize.py` or `stats.py`: both read
`HC_PAPER_REPO` when imported (`summarize.py:26` to `27`, `stats.py:32` to `33`). It reproduces their arithmetic and
tests prove the numbers match (4.6).

### 4.2 `hcbench plan`

Writes a matrix plan, the JSON list that `matrix.py` reads: one object per row with `label`, `model`, `nodes`, `args`
and `timeout_s` (`matrix.py:4`, `matrix.py:140` to `149`).

```
hcbench.py plan --tag TAG --model MODEL_ID --host HOST --peer NODE_ID [--peer NODE_ID ...]
  [--order AP|APAP|...] [--prompt-tokens 128] [--output-tokens 64] [--warmup 1] [--reps 3] [--ctx 4096]
  [--concurrency 1] [--seed-peers] [--seed-timeout 3600] [--layout PATH]
  [--no-capacity-mode none|pooled|all] [--max-commit-growth-gib X] [--load-timeout 3600]
  [--timeout-alone 7200] [--timeout-pooled 9000] [--extra-arg ARG ...] --out PLAN.json [--force]
```

- `--tag` and `--host` match `^[a-z0-9]+$`, with no dashes, so no group prefix can catch another group. `stats.py` groups
  by label prefix and warns about this (`stats.py:12` to `13`). Put concurrency into the tag, as the campaign did
  (`g120n4`).
- Labels: alone rows `{tag}-{host}A-{k}`, pooled rows `{tag}-{host}P{peer count}-{k}`, with `k` counting from 1 within
  each group (`d70-g14A-1`, `d70-g14P2-1` in `paper-2026-10-24-laptop/index_d70.jsonl`).
- `--order` is a string of `A` and `P`, at most 8 characters. `APAP` interleaves two pairs; R7 used it for headline
  models and one `AP` pair for the rest (PAPER-RUNS R7).
- `--peer` takes 32 hex character node ids (`schemas.py:164`). Find them on the dashboard or with `Pool().nodes()`.
  Pooled rows get every `--peer`; alone rows get none.
- Defaults are run_big's defaults: 128/64, 1 warmup, 3 reps, ctx 4096 (`run_big.py:1248` to `1250`). `--load-timeout`
  3600 and `timeout_s` of 7200 alone and 9000 pooled follow the campaign plans (`paper-2026-10-24/plan_kimi_gx.json`).
  `--seed-peers` adds `--seed-peers --seed-timeout <s>` to pooled rows only, as in
  `paper-2026-10-24-laptop/plan_g120n4_row.json`.
- `--layout PATH` adds `--layout PATH` to pooled rows. Warn when a peer stage names an `RPC` device, or when a `local`
  stage names `CPU`. Both were refused in R7 (`paper-2026-10-24-laptop/index_n80_refused_layout_device.jsonl`,
  `index_d70_refused_layout_cpu_stage.jsonl`, NUMBERS section 2).
- `--no-capacity-mode` defaults to `none`. The flag made both pooled 70B rows and the alone 32B row infeasible
  (`index_d70_refused_gpu_only.jsonl`, `index_d32_refused_no_capacity_alone.jsonl`, PAPER-RUNS R7).
- `--max-commit-growth-gib` is passed only when given. run_big's default guard is 0.25 GiB (`run_big.py:1258`). R7 rows
  requested 8 or 24 after the guard stopped rows (NUMBERS section 6, "R7 commit guard").
- **Validation.** For every row, build the argv `matrix.py` would pass (`--model M --min-available-gib 0 --home H`,
  one `--node` per peer, `--strategy local_only` when there are none, then `args`; `matrix.py:143` to `149`). Load
  `scripts/capacity/run_big.py` with `importlib` and call its `parse_args(argv)`. A `SystemExit` rejects the plan with
  run_big's own message. That covers every constraint in `run_big.py:1277` to `1291`. Loading run_big has no side
  effect beyond `sys.path` inserts and imports; it opens no connection.
- Refuse to overwrite `--out` without `--force`. Write JSON with `indent=1` and LF line endings, like the existing plans.
  Print a table of label, peer count and args.

### 4.3 `hcbench run`

```
hcbench.py run PLAN.json --index results/capacity/paper-<date>-<name>/index_<name>.jsonl
  [--preset g14|gx10] [--repo PATH] [--home PATH] [--uv PATH] [--no-probe-macs] [--llama-bin-dir PATH]
  [--append] [--dry-run] [--matrix PATH]
```

- It runs `[sys.executable, "-I", "<repo>/scripts/capacity/paper/matrix.py", PLAN, INDEX]` in the foreground with
  output streamed through. `matrix.py` uses only the standard library. It starts each row with
  `<uv> run --no-sync python scripts/capacity/run_big.py` in `HC_PAPER_REPO` (`matrix.py:143`, `matrix.py:157`).
- Flags map to matrix's environment: `--repo` to `HC_PAPER_REPO`, `--home` to `HC_PAPER_HOME`, `--uv` to `HC_PAPER_UV`,
  `--no-probe-macs` to `HC_PAPER_PROBE_MACS=0`, `--llama-bin-dir` to `HYPERCOMPUTE_LLAMA_BIN_DIR` (`matrix.py:20` to
  `26`, `matrix.py:96` to `113`). Without flags, matrix's own defaults apply, which are the G14 live-host checkout.
- `--preset gx10` sets exactly what `scripts/capacity/paper/gx10/matrix.sh:3` to `6` sets.
- Preflight, all local and read-only:
  - The plan parses and every row has the five keys.
  - The index name matches `^index_[a-z0-9_]+\.jsonl$` and holds none of `_refused`, `_aborted`, `_failed`. Those names
    are for archived bad rows (NUMBERS section 2).
  - An index that exists and is not empty is refused without `--append`.
  - An exclusive lock file `<index>.lock` (`O_CREAT | O_EXCL`) is held for the whole run.
- `--dry-run` prints each row's run_big argv and exits 0.
- Before starting, print two facts. matrix talks to the gateway on `127.0.0.1:8001` (`matrix.py:42`). A row that ends
  `infeasible` or `refused` is retried once after 60 s and leaves an extra `<label>-refused` record
  (`matrix.py:193` to `200`).
- run_big refuses when a deployment it did not create is active (`run_big.py:261` to `264`), so two campaigns cannot
  overlap on one gateway. `hcbench run` adds no gateway calls of its own.
- Ctrl+C reaches matrix and its run_big child in the same console. run_big stops its deployment on interrupt
  (`run_big.py:35`). `hcbench` waits for matrix to exit and returns its exit code.
- `--matrix PATH` swaps in another driver. Tests point it at a stub that appends one fake line to the index.
- Builders never run `hcbench run` without `--dry-run` or `--matrix <stub>`.

### 4.4 `hcbench report`

```
hcbench.py report [--glob "results/capacity/paper-2026-10-2*/index_*.jsonl"] [--results-root PATH]
  [--pairs bench/pairs.json] [--out bench/RESULTS.md] [--check]
```

- The glob is relative to the repo root. Files are read in sorted path order, lines in file order, blank lines
  skipped. The live campaign appends to these files, so a last line that is not valid JSON is skipped with a warning.
  A bad line anywhere else is an error.
- Run records are read from `--results-root`, else `$HC_PAPER_REPO/results/capacity`, else `<repo>/results/capacity`,
  at `<root>/<result_dir>/run.json` (`summarize.py:26` to `27`, `summarize.py:49`). On 2026-10-24, 95 of the 96 records
  in these files had a local `run.json`. The one without is `ml123-gxA-2` (exit 137, no result dir).
- Per-row values are computed exactly as `summarize.row` does (`summarize.py:46` to `74`):
  - status, and `load_s` from `measured.load.seconds`;
  - for N = 1 runs only, the ok reps whose label starts with `rep`, giving the decode and prefill medians and the SSD
    MiB per token median (`/ 2**20`, 1 decimal);
  - batch medians from `batchlib.summarize(batches_of(run))`;
  - RTT from the `network_before` peers that are connected (`summarize.py:39` to `43`);
  - the run error message (`summarize.py:73`).

  Added: `seed_s` from `measured.seed.seconds` (NUMBERS section 2, "Pooled cold start"), and the median first token
  `ttft_ms / 1000` over the same reps.
- Pairs are computed the way `stats.py` does:
  - group membership is label prefix, status `ok`, and the optional label list (`stats.py:39` to `52`);
  - reps come from N = 1 runs only (`stats.py:55` to `65`); batches mode uses one concurrency per group
    (`stats.py:68` to `82`);
  - the ratio is mean(B) / mean(A). The 95% interval is a percentile bootstrap with `random.Random(7)`, 10,000
    resamples, taking sorted ratios at index `int(0.025 n)` and `int(0.975 n) - 1` (`stats.py:94` to `98`).
- **Guard that `stats.py` lacks.** When a group's mean is 0, print `n/a (mean 0)` for that metric. `stats.py --batches
  g120-n1-gxA g120-n8-gxA paper-2026-10-24-gx10conc/index_g120_conc.jsonl` crashes with `ZeroDivisionError` on
  `ssd_bytes_per_step` (`stats.py:96`). Reproduced on 2026-10-24.
- Errors written into the report are cut to 200 characters and pass the secret filter `[A-Za-z0-9_-]{40,}` to `<r>`
  (`matrix.py:33`).
- Output is deterministic: no clock time, LF endings, numbers at the tools' precision (3 decimals for tok/s and ratios,
  1 for MiB and seconds), `n/a` for missing values. `--check` exits 1 when the regenerated text differs from the file
  on disk and writes nothing.

`bench/RESULTS.md` layout:

```
# Hypercompute benchmark results

Generated by `uv run --no-sync python bench/hcbench.py report` from <F> index files matching `<glob>`:
<M> records, <K> with a run.json. Do not edit by hand. docs/pitch/NUMBERS.md is the sheet to quote from.

## Method
(fixed text from NUMBERS "How to read it": ratio is pooled / alone; 95% CI is a percentile bootstrap of the ratio of
means, 10,000 resamples, seed 7, reps at N = 1 and batches at N > 1; n is measured reps, warmups excluded; temperature
0, seed 7, ignore_eos, no prompt cache, fixed-length prompts; decode is the engine's (predicted_n - 1) / predicted_ms;
SSD is system-wide physical reads per decoded token, in MiB; small samples from one row per side show rep-to-rep spread
inside one load, not run-to-run spread.)

## Condition keys
(the `conditions` map of pairs.json, one row per key)

## 1. Pairs
| Pair | Model | Condition | Prompt/answer | N | n A | n B | Metric | A mean | B mean | Ratio B/A [95% CI] | NUMBERS | Index files |

## 2. Every run
| Index | Label | Model | Peers | N | Status | Load s | Seed s | Decode tok/s median | Prefill tok/s median | First token s median | SSD MiB/token median | Aggregate decode tok/s median (N > 1) | RTT ms before | Wall s |

## 3. Not in any ratio
- <index> <label>: <status>, <reason or error>
```

Section 3 lists every record from an index file whose name holds `_refused`, `_aborted` or `_failed`, every label
ending in `-refused` (matrix's first attempts) or `-w` (warm-up runs such as `120b-local-w`), every status other than
`ok`, every record without a `run.json`, and every empty index file (such as
`paper-2026-10-24-gx10conc/index_gx10_chain.jsonl`).

Example rows, computed on 2026-10-24 from `paper-2026-10-24-laptop/index_d70.jsonl` (condition L7, 64/32, N = 1):

| Index | Label | Model | Peers | N | Status | Load s | Seed s | Decode tok/s median | Prefill tok/s median | First token s median | SSD MiB/token median | Aggregate decode tok/s median (N > 1) | RTT ms before | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper-2026-10-24-laptop/index_d70.jsonl | d70-g14A-1 | llama-3.3-70b... | 0 | 1 | ok | 42.3 | n/a | 0.165 | 6.361 | 10.02 | 23454.2 | n/a | 6, 6 | 1097.2 |
| paper-2026-10-24-laptop/index_d70.jsonl | d70-g14P2-1 | llama-3.3-70b... | 2 | 1 | ok | 96.2 | 851.7 | 0.898 | 12.514 | 5.01 | 2761.2 | n/a | 6, 6 | 1162.5 |

The model column prints the record's `model` field in full. The builder reads it from the index.

### 4.5 `bench/pairs.json` and the values it must reproduce

`pairs.json` holds a `conditions` map and a `pairs` list. Each pair has `id`, `kind` (`pooled_vs_alone` or
`n_vs_n1`), `condition`, `model`, `prompt_answer`, `n`, `a_prefix`, `b_prefix`, `indexes` (paths relative to
`results/capacity/`), `only_labels` (list or null), `mode` (`reps` or `batches`), `metrics`, and `numbers_ref` (the
NUMBERS.md section).

Conditions, shortened from NUMBERS "Condition keys":

- **L7**: G14 host (RTX 5070 Ti Laptop GPU, 11.94 GiB VRAM, 31.12 GiB RAM, Windows 11) with mac-a and mac-b (MacBook
  Air M4, 16 GiB, 10.5 GiB cap each), home network, both Macs direct over Iroh, RTT 5 to 8 ms before each pooled row.
- **G6**: ASUS Ascent GX10 (GB10, 121.6 GiB unified) with the G14 (CUDA, 9 GiB cap), mac-a and mac-b (11.5 GiB caps)
  and mac-azim (7.5 GiB cap), home Wi-Fi, one subnet. In the 671B pooled rows mac-azim was on the relay at 48 to
  103 ms; in the 1T pooled rows all four were direct at 5 to 55 ms.
- **B1**: G14 with mac-azim, mac-a and mac-b (7.5 GiB cap each), building Wi-Fi, laptops on different subnets, Iroh
  direct through NAT.
- **GXA**: the GX10 alone, no peers.

Every value below was reproduced on 2026-10-24 with `HC_PAPER_REPO=<this repo> python -I scripts/capacity/paper/stats.py`
on the local files. Each one matches NUMBERS.md. Ratio is B/A with the 95% CI.

| id | Condition | Model, prompt/answer, N | A vs B (index under `results/capacity/`) | Mode | Metric: ratio [95% CI] | NUMBERS |
|---|---|---|---|---|---|---|
| d70 | L7 | Llama 3.3 70B Instruct Q4_K_M, 64/32, N = 1, 2 vs 2 reps | `d70-g14A` vs `d70-g14P2` (`paper-2026-10-24-laptop/index_d70.jsonl`) | reps | decode 5.444 [5.423, 5.464]; prefill 1.967 [1.896, 2.043]; SSD 0.118 [0.118, 0.118] | 1a |
| g120 | L7 | gpt-oss-120b MXFP4, 128/64, N = 1, 3 vs 3 reps | `g120-g14A` vs `g120-g14P2` (`paper-2026-10-24-laptop/index_g120.jsonl`) | reps | decode 2.283 [2.000, 2.538]; prefill 7.759 [6.945, 8.855] | 1b |
| vl72 | L7 | Qwen2.5-VL 72B Instruct Q4_K_M, text only, 64/32, N = 1, 2 vs 2 reps | `vl72-g14A` vs `vl72-g14P2` (`paper-2026-10-24-laptop/index_vl72.jsonl`) | reps | decode 2.592 [2.571, 2.613] | 1c |
| g120n4 | L7 | gpt-oss-120b, 128/64, N = 4, 3 vs 3 batches | `g120n4-g14A` vs `g120n4-g14P2` (`paper-2026-10-24-laptop/index_g120n4.jsonl`) | batches | aggregate decode 1.464 [1.295, 1.674] | 1d |
| g120n8 | L7 | gpt-oss-120b, 128/64, N = 8, 3 vs 3 batches | `g120n8-g14A` vs `g120n8-g14P2` (`paper-2026-10-24-laptop/index_g120n8.jsonl`) | batches | aggregate decode 1.349 [1.246, 1.479]; step 0.742 [0.677, 0.811]; SSD per step 0.171 [0.140, 0.201] | 1d |
| n80 | L7 | Qwen3-Next-80B-A3B Q3_K_M, 128/64, N = 1, 6 vs 3 reps | `n80-g14A` vs `n80-g14P2` (`paper-2026-10-24-laptop/index_n80.jsonl`) | reps | decode 0.468 [0.362, 0.611] | 2 |
| 671b | G6 | DeepSeek-V3.1 671B UD-TQ1_0, 64/32, N = 1, 4 vs 4 reps | `671b-gxA` vs `671b-gx4` (`paper-2026-10-24/index_671b_gx.jsonl`) | reps | decode 0.830 [0.769, 0.903]; prefill 2.418 [2.194, 2.721] | 3b |
| kimi | G6 | Kimi-K2.6 1T IQ1_S, 64/32, N = 1, 4 vs 4 reps | `kimi-gxA` vs `kimi-gx4` (`paper-2026-10-24/index_kimi_gx.jsonl`) | reps | decode 0.856 [0.818, 0.886]; prefill 1.507 [1.412, 1.587]; SSD 0.379 [0.365, 0.397] | 3b |
| g120gx-n8 | GXA | gpt-oss-120b, 128/64, N = 8 vs N = 1, 6 vs 6 batches | `g120-n1-gxA` vs `g120-n8-gxA` (`paper-2026-10-24-gx10conc/index_g120_conc.jsonl`), kind `n_vs_n1` | batches | aggregate decode 2.731 [2.583, 2.901] | 3c |
| r2-3mac | B1 | gpt-oss-120b, 128/64, N = 1, 9 vs 9 reps | `120b-local` vs `120b-3mac` (`paper-2026-10-22/index_120b.jsonl`), only `120b-local-1..3`, `120b-3mac-1..3` | reps | decode 1.274 [1.105, 1.464]; prefill 2.994 [2.639, 3.349]; SSD 0.151 [0.117, 0.188] | 4 |
| r2-1mac | B1 | gpt-oss-120b, 128/64, N = 1, 9 vs 6 reps | `120b-local` vs `120b-1mac` (same index), only `120b-local-1..3`, `120b-1mac-1..2` | reps | decode 1.253 [1.106, 1.407] | 4 |
| r1-235b | B1 | Qwen3-235B-A22B UD-Q2_K_XL, 64/64, N = 1, 9 vs 9 reps, steady state | `235b-local` vs `235b-3mac` (`paper-2026-10-22/index_235b.jsonl` and `index_235b_tight.jsonl`), only `235b-local-3..5`, `235b-3mac-4..6` | reps | decode 1.079 [0.976, 1.184] | 2, 4 |

The `d70` alone group also matches `d70-g14A-2`, whose status is `aborted_by_guard`, so the status filter drops it.
Keep that behaviour. The pair is "one pair of runs with 2 reps each" (NUMBERS 1a caveat).

### 4.6 Tests (`bench/tests/`)

Run `uv run --no-sync python -m pytest bench/tests -q`. Suggested files: `test_hcbench_plan.py`,
`test_hcbench_run.py`, `test_hcbench_report.py`.

- **plan**: labels and order for `AP`, `APAP` and a concurrency tag; alone rows get no nodes; pooled rows get
  `--seed-peers` only when asked; bad tags and bad node ids are rejected; run_big's `parse_args` rejects
  `--concurrency 17` and 8 x (128 + 64) > 4096 with run_big's message; the layout warnings fire; overwrite needs
  `--force`.
- **run**: `--dry-run` prints matrix's argv per row; the index name rules, the `--append` rule and the lock work; the
  preset sets the env of `gx10/matrix.sh`; a stub `--matrix` script receives the plan and index paths and the mapped
  environment. Nothing reaches a real gateway.
- **report on fixtures**: a hand-made index of 4 records with small `run.json` files (N = 1 reps, an N = 4 batch run,
  a refused row, a missing run dir) gives the expected tables. A truncated last line is skipped with a warning. A
  zero-mean group prints `n/a (mean 0)`. `--check` detects a change.
- **parity with the real records** (skipped when `results/capacity/paper-2026-10-24-laptop` is missing):
  - for every record in the default glob, the per-row values equal `summarize.row(rec)` (load `summarize.py` with
    `importlib` after setting `HC_PAPER_REPO` with `monkeypatch`). Read each index once and pass the same records to
    both sides, since the live campaign may append between reads;
  - every value in the 4.5 table, to 3 decimals;
  - for `d70` and `kimi`, the output of `stats.py` run as a subprocess with `-I` matches `hcbench` line for line on
    n, mean, ratio and CI.

  These tests only read files.
- The real `bench/RESULTS.md` is generated once by the builder with `hcbench.py report` (local reads only) and checked
  in by the integrator.

## 5. Reference: every flag and route this spec relies on

### `backend/hypercompute/cli.py`

| Item | Line |
|---|---|
| global `--version`, `-v/--verbose` | 544, 545 |
| `host`: `--port` (default `config.DEFAULT_GATEWAY_PORT`, 8000), `--bind`, `--no-browser`, `--transport lan\|iroh` (default lan), `--relay-url`, `--local-max-memory`, `--local-devices` | 548 to 557 |
| host checks: `--bind` refused with iroh, `--relay-url` needs iroh | 192 to 195 |
| host lock, startup message, dashboard URL, browser open | 177 to 178, 197, 207 to 208, 217 to 218 |
| `join`: `--host` or `--ticket` (one required), `--relay-url`, `--code` (required), `--name`, `--max-memory`, `--reserve`, `--devices`, `--worker-port` (default 47052), `--advertise-ip`, `--cache`, `--yes`, `--managed`, `--devices-allowed`, `--max-disk`, `--exo` | 559 to 582 |
| join over Iroh: refuses `--exo` and `--advertise-ip`; engine release must match the host; worker on 127.0.0.1; `--devices` replaces the default; under 1 GiB warning; hcpeer env refusal; managed flag rules | 262 to 263, 287 to 290, 291, 297 to 298, 302 to 303, 315 to 321, 328 to 339 |
| join lock | 228 to 229 |
| `status --port` | 584 to 586; behaviour 422 to 457 |
| `invite --port` | 588 to 590; behaviour 512 to 528 |
| `pause`, `resume` | 592 to 595 |
| `leave --purge-cache` | 597 to 599; behaviour 342 to 406 |
| `doctor --host` | 601 to 603; behaviour 463 to 503 |
| `engine install\|status`, `--engine llama\|airllm\|vllm\|exo`, `--variant`, `--yes`, `--force` | 605 to 612; status exit code 60 to 64; install prompt 76 to 82 |
| `models list\|pull [model_id]` | 614 to 617 |
| `tune` and its flags (not used by packaging) | 619 to 636 |
| `inventory --no-engine` | 639 to 641 |
| `error:` and `fix:` output, exit codes | 46 to 50, 645 to 661 |

### `backend/hypercompute/app.py`

| Route | Auth | Line |
|---|---|---|
| `GET /api/health` | open; full body only from loopback | 432 |
| `GET /api/owner/bootstrap` | loopback | 443 |
| `GET /api/network` | loopback | 449 |
| `POST /api/network/invite` | owner | 454 |
| `GET /api/nodes` | loopback | 483 |
| `GET /api/peers` | loopback | 521 |
| `GET /api/models` | loopback | 569 |
| `POST /api/plans` | owner | 581 |
| `POST /api/deployments` (header `Idempotency-Key`) | owner | 585 to 588 |
| `GET /api/deployments` | loopback | 590 |
| `GET /api/deployments/{dep_id}` | loopback | 594 |
| `DELETE /api/deployments/{dep_id}` | owner | 598 |
| `GET /api/capacity/pools` | loopback | 620 |
| `GET /api/capacity/explain` and its query bounds | loopback | 624 to 637 |
| `GET /v1/models` | loopback | 720 |
| `POST /v1/chat/completions` | owner | 729 to 750 |
| auth guards `owner_read`, `owner_write`; loopback sets | | 383 to 396; 64 to 65 |
| error body `_error`; exception mapping | | 137 to 141; 399 to 429 |
| body limits 64 KiB, 256 KiB, 4096 KiB; 413 | | 67 to 71, 166 to 171, 184 to 185 |
| capacity code to status table | | 93 to 121 |
| invite join command over Iroh; startup invite print | | 299 to 318; 344 to 353 |

Routes that exist but that this spec does not use: sessions (459 to 480), node heartbeat, leave and verify (487 to
510), managed peer worker, seed and cancel (526 to 566), model download (573), replace (602), replan (639 to 650),
events, runs and benchmarks (653 to 682), engines (685), tune (699 to 713), profiles (715).

### `scripts/capacity/run_big.py`

| Item | Line |
|---|---|
| `--model` (required), `--port` (default 8001), `--strategy` (`all_selected`, `fastest_fit`, `auto`, `local_only`; default all_selected), `--node` (repeatable), `--engine`, `--no-capacity-mode` | 1242 to 1247; strategies 95 |
| `--ctx 4096`, `--prompt-tokens 128`, `--output-tokens 64`, `--warmup 1`, `--reps 3` | 1248 to 1250 |
| `--keep`, `--plan-only`, `--load-timeout 7200`, `--chat-timeout 1800` | 1251 to 1254 |
| guards `--min-available-gib 1.0`, `--min-drive-free-gib 5.0`, `--max-commit-growth-gib 0.25`, `--max-swap-growth-gib 2.0` | 1255 to 1261 |
| `--home`, `--results-dir`, `--seed-peers`, `--seed-timeout 3600`, `--layout`, `--concurrency 1` | 1262 to 1276 |
| argument checks | 1277 to 1291 |
| exit codes: ok 0, plan_only 0, refused 2, pool_short 2, interrupted 130, else 1 | 1312 |
| owner token from bootstrap; `Gateway.call`; error parsing | 220 to 234; 208 to 217; 182 to 198 |
| plan body; explain; deploy with `Idempotency-Key`; lookup by key; ready polling; stop | 286 to 297; 368 to 375; 380 to 381; 384 to 390; 393 to 417; 430 to 436 |
| streamed chat settings and call | 104; 701 to 722 |
| terminal states; results dir | 94; 93 |

### `scripts/capacity/paper/`

| Item | Line |
|---|---|
| `matrix.py PLAN.json INDEX.jsonl`; plan row keys | 4 |
| env `HC_PAPER_REPO`, `HC_PAPER_HOME`, `HC_PAPER_UV`, `HC_PAPER_PROBE_MACS` | 22 to 26 |
| gateway at `127.0.0.1:8001`; network snapshot redaction; secret pattern | 42; 48 to 56; 33 |
| run_big argv per row; cwd and env | 143 to 149; 155 to 157 |
| cooldown via `nvidia-smi` or `HYPERCOMPUTE_LLAMA_BIN_DIR` | 81 to 125 |
| retry after infeasible or refused, `-refused` record | 193 to 200 |
| `summarize.py OUT.md INDEX...`; results root; row values; RTT | 7, 26 to 27, 46 to 74, 39 to 43 |
| `stats.py [--batches] A B INDEX... [--only-labels]`; groups; bootstrap | 3 to 4, 39 to 82, 94 to 98 |
| `mac_join.sh TICKET CODE NAME [MAX_MEMORY] [RESERVE] [EXTRA]`, defaults 10GiB and 4GiB, flags `--devices MTL0 --cache --yes`, `caffeinate -dimsu` | 3, 7, 12 to 13 |
| `mac_bootstrap.sh`: bundle, branch, `uv sync`, `engine install --yes`, `doctor` | 8 to 10, 24 to 26 |
| `gx10/host.sh`: `host --transport iroh --port 8001 --no-browser`, `HYPERCOMPUTE_LLAMA_BIN_DIR` | 4, 6 |
| `gx10/join_g14.sh`: invite parsing, `--devices CUDA0 --max-memory 9GiB --yes --cache --managed --devices-allowed CUDA0 --max-disk 24GiB`, output redaction | 8 to 11, 14 to 16 |
| `gx10/rejoin_mac.sh`: same parsing and redaction | 10 to 15 |
| `gx10/matrix.sh`: GX10 env for matrix | 3 to 6 |

### Other files

| Item | File and line |
|---|---|
| `DEFAULT_GATEWAY_PORT = 8000`, `DEFAULT_WORKER_PORT = 47052`, `WORKER_PORT_SCAN = 24`, `INVITE_TTL_S = 15 * 60`, `OWNER_HEADER` | `config.py` 9, 13, 14, 19, 30 |
| home, agent and cache paths | `config.py` 59 to 65, 73 to 74 |
| owner token per host start | `coordinator.py` 146 |
| code alphabet, length, normalize, format, LAN join command | `services/sessions.py` 22 to 23, 30 to 35, 119 to 120 |
| ticket string | `transport/host.py` 88 to 89, `transport/iroh.py` 416 to 420, 500 |
| device eligibility and default caps | `agent/contribution.py` 15 to 47 |
| join consent text, `--yes` rule, `Joined.` line | `agent/join.py` 708, 715, 720, 818 to 820, 833, 837 |
| hcpeer env names | `agent/hcpeer.py` 18 to 20 |
| ready, stale plan, active deployment, not found | `services/deployments.py` 351 to 361, 468 to 474, 489, 340 |
| SSE passthrough, error events, interrupted body | `services/streaming.py` 443 to 486, 575, 678 to 689; `adapters/llama_rpc.py` 409 to 410 |
| `Model` forbids extra fields; `EngineId`; inventory; `JoinRequest`; `PlanStrategy`; `PlanRequest`; `DeploymentState`; `Deployment.idempotency_key`; `CreateDeploymentRequest` | `schemas.py` 21 to 22, 26, 48 to 142, 162 to 169, 463 to 467, 482 to 500, 656 to 668, 707, 732 to 735 |

## 6. Known issues found while writing this

- `stats.py --batches` divides by a zero mean and crashes (4.4). `hcbench` guards it; `stats.py` itself is under
  `scripts/` and is not changed here.
- The CLI defaults to port 8000 while `run_big.py` and `matrix.py` assume 8001. Packaging uses 8001 everywhere and
  always passes `--port`.
- The host prints a live invite into its own output at startup (`app.py:351` to `353`). Any log of a host, including
  the one-click `host.log`, needs the filter of 2.6.
