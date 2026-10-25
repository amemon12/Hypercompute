# Share your computer with Hypercompute

Hypercompute lets a few friends pool their computers so one AI model can run across all of them. One person is
the host. Everyone else lends part of their computer's memory while they share.

## For a friend: three steps

1. **Get ready (once).**
   - Get the Hypercompute folder from the host. If it came as a .zip file, unzip it first. Put the folder anywhere,
     for example in Downloads.
   - The folder is a program, not a document. Its scripts run with your account's rights, and step 2 tells you to
     click past the warning Windows or macOS shows for files from the internet. The Windows script also skips the
     PowerShell script policy for its own run. So only take the folder from someone you trust, and only from them
     directly.
   - Install uv once: https://docs.astral.sh/uv/getting-started/installation/
     uv is a small tool that sets up Python for this folder. These scripts never install it for you.

2. **Open the join script.** Open the `join` folder inside the Hypercompute folder.
   - Mac: double-click `hypercompute-join.command`. If macOS says it cannot open it, right-click it and choose
     Open once.
   - Windows: double-click `hypercompute-join.cmd`. If Windows warns you about the file, choose to run it. On the
     blue "Windows protected your PC" screen that means More info, then Run anyway.

   These warnings show because the scripts are not signed. Only skip them for a folder you got straight from the
   host. If the host sent the zip through a shared or public link, ask for its SHA-256 in a separate private message
   and check it before you unzip: `Get-FileHash file.zip` on Windows, `shasum -a 256 file.zip` on a Mac.
   - Linux: in a terminal, run `./join/hypercompute-join.sh` from the Hypercompute folder. Type nothing after it.
     Do not paste the invite on the same line. The script asks for it in step 3.

   The first run asks before each download. First it asks to download Python packages into the folder. If your
   computer has no Python 3.12, uv also downloads Python 3.12 into its own folder, outside Hypercompute, and that
   copy stays after you delete the Hypercompute folder. Then it asks to download the inference engine, and lists
   each file first. Type `y` and press Enter.

3. **Paste the invite and say yes.**
   - Paste the invite line the host sent you and press Enter. The text stays hidden while you paste. That is
     normal.
   - Check the name, memory and disk it shows. Press Enter to accept each one, or type a smaller number.
   - Type `y` to share. When it says "Sharing", you can close the window.

To stop sharing, double-click `hypercompute-leave` (`.command` on a Mac, `.cmd` on Windows, or run
`./join/hypercompute-leave.sh` on Linux). It asks whether to delete the model cache too.

To check whether you are still sharing, open `hypercompute-status` the same way.

## What sharing does on your computer

- A worker holds part of the model in your memory. The host plans within the memory number you accept. That
  number is a planning limit, not a hard cap on your GPU.
- The worker has no password of its own. It listens only on your computer, and the host reaches it through an
  encrypted Iroh tunnel. Only pair with a host you trust.
- Your computer is kept awake while it shares, but the screen can still turn off and lock as usual. No power
  setting changes, and this ends when sharing stops. On Linux this uses `systemd-inhibit` when your user is
  allowed to.
- With the cache on, model pieces stay on disk in `.hypercompute/rpc-cache` in your home folder after you stop, so
  the next load is quicker. `hypercompute-leave` can delete them.
- Host downloads are off unless you type `y`. With them on, the host can download your share of a model from
  huggingface.co into that cache, up to the amount shown. At least 20 GiB of disk always stays free.
- Host downloads on also lets the host restart your worker on the device you share and change how many CPU threads
  it uses, from 1 up to all your cores. If the host goes away, your computer keeps waiting for it for up to 2 hours,
  and a worker that crashed is started again, at most 3 times in 30 minutes. The question and the consent screen
  both say this.
- An invite works once, for 15 minutes.
- While sharing runs, other user accounts on this computer can see the invite in the list of running programs.
  This matters only on a computer that other people also log in to.

## For the host: three steps

1. Double-click `hypercompute-host` (`.command`, `.cmd`, or `./join/hypercompute-host.sh`). It checks the engine,
   starts the host in the background and opens the dashboard in your browser. The dashboard works only on this
   computer. When it opens, a first invite is already on your clipboard.
2. Paste the invite to one friend in a private message.
3. For the next friend, double-click `hypercompute-invite`. A new invite is now on your clipboard. Make a new invite
   for each friend.

If there is no clipboard tool (some Linux setups), the invite is saved in `.hypercompute/oneclick/invite.txt` in
your home folder, readable only by you. Delete it after you send it.

The host wrapper keeps the gateway running in the background. The demo app can stop a host that the app itself started; it does not stop a host started by the wrapper.

## Demo app

After running `uv sync` in the Hypercompute folder, open `hypercompute-app.command` on macOS,
`hypercompute-app.cmd` on Windows, or run `./join/hypercompute-app.sh` on Linux. The app wrappers start
`app/hcapp.py` directly. They do not run the first-use download prompts described above.

The app opens on port 8790 and talks to the gateway on port 8001. Its Host tab shows peers, model plans, layer
placement and chat. The Join tab asks for an invite with hidden text. The browser link contains a launch key;
keep the address bar and invites out of screenshots. Closing the browser tab does not stop a running gateway.

## If something goes wrong

| Message | What to do |
|---|---|
| This folder is not inside a Hypercompute checkout | The `join` folder was moved out of the Hypercompute folder. Keep the whole folder together. |
| uv is not installed | Install uv from the link above, close the window and open the script again. |
| Run this with nothing after it | You typed or pasted something after the command. Run it again with nothing after it and paste the invite when it asks. |
| That does not look like a Hypercompute invite | Copy the whole line again. If it is older than 15 minutes or was used, ask for a new one. |
| This is a LAN invite | Ask the host for a new invite from `hypercompute-invite`. |
| engine mismatch | Your folder and the host's are on different versions. Get a fresh copy of the folder from the host. |
| Your NVIDIA driver (N) is too old for the GPU engine | Update the NVIDIA driver to the version it names, then run `uv run hypercompute engine install` in the Hypercompute folder. Until then you can share the CPU, which is slow. |
| checked port 8001 / Looked for a host on 127.0.0.1:8001 | The scripts look for the host on port 8001. A host started by hand with `hypercompute host` uses 8000. Start it with `hypercompute-host`, or set `HC_PORT` to its port. |
| Still connecting | Open `hypercompute-status` in a minute. |
| could not be stopped (Windows) | Open PowerShell as Administrator, go to the Hypercompute folder and run `.\join\hypercompute-leave.cmd`. |

The logs are `.hypercompute/oneclick/join.log` and `host.log` in your home folder. They never hold the invite:
long tokens show as `<r>` and codes as `<code>`.

## Implementation notes

The one-click scripts implement section 2 of `docs/PACKAGING-SPEC.md`. Run the commands below from the repository root.

### Files

| File | Purpose |
|---|---|
| `hypercompute-{join,leave,status,host,invite}.command` | macOS double-click wrappers (three lines plus the shebang) |
| `hypercompute-{join,leave,status,host,invite}.sh` | Linux wrappers, same content |
| `hypercompute-{join,leave,status,host,invite}.cmd` | Windows double-click wrappers, the exact template of spec 2.2 (full path to `powershell.exe`) |
| `hypercompute-{join,leave,status,host,invite}.ps1` | Windows bootstrap and hand-over, called by the `.cmd` |
| `lib/hc-common.sh`, `lib/hc-common.ps1` | bootstrap: find the checkout, find uv, ask before `uv sync --frozen` |
| `lib/oneclick.py` | entry point and the five actions, plus the internal `_supervise` |
| `lib/oneclick_core.py` | command runner, context, prompts |
| `lib/oneclick_invite.py` | invite parsing, redaction, command display |
| `lib/oneclick_plan.py` | device, memory cap, reserve, cache, host downloads, the join command |
| `lib/oneclick_supervise.py` | the detached supervisor: redacted log, pid file, keep awake |
| `tests/` | Wrapper, invite, consent and lifecycle tests, with inventory fixtures in `tests/fixtures/` |
| `.gitattributes` | LF for `.sh` and `.command`, CRLF for `.ps1` and `.cmd` |

Stage the POSIX wrappers with the executable bit, because this Windows worktree cannot set it and Finder will not
run a `.command` without it:

```
git add --chmod=+x join/*.command join/*.sh
```

### Checks

```
# From the repository root:
uv run --no-sync python -m pytest join/tests -q
uv run --no-sync ruff check join
```

The tests run `bash -n` on every `.sh` and `.command` (Git Bash on Windows) and parse every `.ps1` with
`[System.Management.Automation.Language.Parser]::ParseFile`. They never run `hypercompute`, uv, a clipboard tool or
a background process, and they block sockets. `test_oneclick_argv.py` runs the two bootstraps with `HC_REPO` set to a
missing folder, so they stop before uv even if the invite check breaks.

### Dry run

The host, invite, join, leave and status wrappers take `--dry-run`. The app wrappers use `hcapp.py` options instead. The bootstrap prints `uv sync --frozen` instead of running it when `.venv` is
missing. Otherwise it hands over to `oneclick.py --dry-run`, which runs nothing: no `status`, `engine status`,
`doctor`, `inventory`, `nvidia-smi` or health check. It prints each command it would run, with the ticket shown
as `<ticket>` and the code as `<code>`.

```
HC_INVITE="uv run hypercompute join --ticket <a fake 30+ character ticket> --code ABCDE-FGHJK" \
  ./join/hypercompute-join.sh --dry-run --inventory-file join/tests/fixtures/inventory_g14.json
```

`--inventory-file` works only with `--dry-run`. Without it, device and memory show as placeholders. A dry run
skips the local Iroh ticket parse, so a fake invite line works.

### Environment overrides

Each one replaces a prompt, and the consent screen still shows the value: `HC_INVITE` (never echoed), `HC_NAME`,
`HC_DEVICE`, `HC_MAX_MEMORY`, `HC_RESERVE`, `HC_CACHE` (0 or 1), `HC_MANAGED` (0 or 1), `HC_MAX_DISK`,
`HC_RELAY_URL` (https only), `HC_LOCAL_MAX_MEMORY` (with a unit, for example `8GiB`) and `HC_LOCAL_DEVICES`
(host). Also `HC_PORT` (default 8001),
`HC_REPO`, `HC_NO_PAUSE=1` and `HYPERCOMPUTE_HOME`. `HC_ONECLICK_YES=1` skips the final "Share this computer now?"
question. It is for scripted runs only. Do not put it in front of a friend.

### Where this differs from the spec, and why

- The Python logic is one layer for all three OSes, as the spec asks, but split over five files in `lib/` so each
  stays under 800 lines. `oneclick.py` is still the only entry point and the supervisor still starts as
  `[python, oneclick.py, "_supervise", ...]`.
- Windows background start uses `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`, and tries `CREATE_BREAKAWAY_FROM_JOB`
  first, instead of `DETACHED_PROCESS`. `.venv\Scripts\python.exe` is the venv launcher, which starts the real
  interpreter as a child. Under `DETACHED_PROCESS` that child has no console, so Windows would open a new console
  window for it, and closing that window would stop sharing. `CREATE_NO_WINDOW` gives the supervisor its own hidden
  console, which no user window can close. The supervisor starts `hypercompute` with `CREATE_NO_WINDOW` too.
- The supervisor's child gets `PYTHONUNBUFFERED=1` and `PYTHONIOENCODING=utf-8` so the log fills line by line.
  Nothing else in the environment changes. `HC_INVITE` is removed before the supervisor starts.
- The supervisor fills `<ticket>` and `<code>` only right after `--ticket` and `--code`, so a name like `<code>`
  is never replaced.
- With host downloads on and the cache off, the cache is turned on and the script says so. The CLI does the same
  (`cli.py` 305 to 306), so the consent screen now matches what the join does.
- uv: the scripts never download or pipe an installer (spec 2.2 step 2). They print the official URL and stop.
  The first-run prompt says that `uv sync --frozen` may also download Python 3.12 into uv's own folder, because
  `.python-version` pins 3.12 and uv downloads a missing interpreter by default.
- An invite on the command line is refused. The bootstraps stop before uv when an argument is `--ticket`,
  `--code`, a bare ticket or a bare code. `oneclick.py` refuses any argument it does not know without echoing it,
  and its argparse errors are redacted.
- The ticket and code are in the arguments of the `hypercompute join` process (and of `systemd-inhibit` in front
  of it on Linux), because the CLI only takes them as flags. Other accounts on the same computer can read them
  with `ps` while it runs. The code is spent at join; if the join fails first, it stays valid until it expires.
  The dry run says this. The fix belongs in the CLI: let `hypercompute join` read them from the environment or
  stdin, then have the supervisor pass them that way.
- Keep awake does not keep the display on: `caffeinate -ims` on macOS (not the `-dimsu` of `mac_join.sh`, which
  runs as a headless SSH user), `systemd-inhibit --what=sleep` on Linux, and no display flag on Windows.
- The join runs with `--yes`, so the CLI's own `--managed` consent text only reaches `join.log`. The question and
  the consent screen repeat its terms (worker restarts, thread changes, the 2 hour wait, crash restarts) first.
- With an NVIDIA GPU and a CPU engine, the advice depends on the driver in the inventory. Below the CUDA minimum of
  `engine.choose_variant` (580 on Windows x86_64, 570 on Linux x86_64) it asks for a driver update, because
  `engine install --force` would install the CPU build again. Otherwise it suggests `--force`.
- `status` and `invite` name the port they checked, because the CLI defaults to 8000 and these scripts to 8001.
- After reading the invite with `getpass`, pending console input is dropped. If the host's `Expires:` line is pasted
  too, it cannot answer the next prompt.
- Printed paths go through the same redaction as everything else, so a folder name of 30 or more letters, digits,
  `_` or `-` shows as `<r>`. The default `~/.hypercompute` paths are not affected.
