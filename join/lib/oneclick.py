#!/usr/bin/env python3
"""One-click join, leave, status, host and invite for Hypercompute (docs/PACKAGING-SPEC.md section 2).

The wrappers in join/ find the checkout and uv, then run this file with the repo venv:

    uv run --no-sync python join/lib/oneclick.py <join|leave|status|host|invite> --repo <repo> [--port N] [--dry-run]

It talks to the `hypercompute` CLI only through its public output and never imports hypercompute internals.
It never prints an invite ticket or code. At most the last 4 characters of a ticket are shown.
The helper modules next to it (oneclick_*.py) hold the parts that tests check one by one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

_LIB = str(Path(__file__).resolve().parent)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)  # the oneclick_* modules, also when this file is imported by path

from oneclick_core import GIB, Ctx, Result, Runner, Stop, ask_yes  # noqa: E402
from oneclick_invite import Invite, InviteError, parse_invite, redact_generic, show_cmd, token_after  # noqa: E402
from oneclick_plan import JoinPlan, build_plan, compose_join, managed_terms  # noqa: E402
from oneclick_supervise import supervise_main  # noqa: E402

DEFAULT_PORT = 8001
WAIT_S = 120
LAN_INVITE = "This is a LAN invite. Ask the host for a new invite from hypercompute-invite."
BAD_INVITE = "That does not look like a Hypercompute invite"
TRUST = "Only pair with a host you trust."
ARGV_NOTE = ("the hypercompute join process gets them as command arguments, so other accounts on this computer "
             "can see them in the process list while it runs")
KEEP_AWAKE = "yes, while sharing (the screen can still turn off and lock)"
# A unit is required and must be one units.parse_size knows: KB to TB, KiB to TiB, or a bare K, M, G or T.
SIZE_RE = re.compile(r"\d{1,6}(?:\.\d{1,3})?\s?(?:[KMGT]i?B|[KMGT])", re.IGNORECASE)
DEVICES_RE = re.compile(r"[A-Za-z0-9]{1,16}(?:,[A-Za-z0-9]{1,16}){0,15}")


# ----------------------------------------------------------------------------- join steps


def read_invite(ctx: Ctx) -> Invite:
    """Step 4: hidden paste, at most 3 tries. HC_INVITE replaces the prompt and is never echoed."""
    from_env = ctx.env.get("HC_INVITE")
    check = None if ctx.dry_run else ctx.ticket_check
    for _ in range(1 if from_env is not None else 3):
        try:
            text = from_env if from_env is not None else ctx.read_secret(
                "Paste the invite line from the host, then press Enter")
            invite = parse_invite(text, check)
        except EOFError:
            raise Stop("No invite was pasted. Nothing was changed.") from None
        except InviteError as exc:
            if exc.lan:
                raise Stop(LAN_INVITE) from None
            ctx.say(BAD_INVITE)
            continue
        ctx.say(f"Invite read: ticket ...{invite.tail} ({len(invite.ticket)} characters), code hidden.")
        if ctx.dry_run:
            ctx.note("skipped the local ticket check (iroh.EndpointTicket.from_string)")
        return invite
    raise Stop("No valid invite. Ask the host for a fresh one from hypercompute-invite.")


def show_consent(ctx: Ctx, plan: JoinPlan, invite: Invite) -> None:
    """Step 11: everything the join will share, before the one question that starts it."""
    cache = f"on, in {ctx.home / 'rpc-cache'} (kept after you stop)" if plan.cache else "off"
    managed = (f"on, up to {plan.disk_gib} GiB from huggingface.co (at least 20 GiB of disk always stays free)"
               if plan.disk_gib is not None else "off")
    ctx.say("")
    ctx.say("Ready to share this computer:")
    ctx.say(f"  Name shown to the host:   {plan.name}")
    ctx.say(f"  Device:                   {plan.device}")
    ctx.say(f"  Memory the host may use:  {plan.max_gib} GiB (the host plans within it; it is not a hard cap)")
    ctx.say(f"  Memory kept free for you: {plan.reserve_gib} GiB")
    ctx.say(f"  Memory pool:              {plan.pool_text}")
    ctx.say(f"  Cache:                    {cache}")
    ctx.say(f"  Host downloads:           {managed}")
    if plan.disk_gib is not None:
        restart, unattended = managed_terms(plan.device)
        ctx.say(f"  With host downloads on:   {restart}")
        ctx.say(f"  Unattended:               {unattended}")
    ctx.say(f"  Keep awake:               {KEEP_AWAKE}")
    if plan.relay_url:
        ctx.say(f"  Relay:                    {plan.relay_url}")
    ctx.say(f"  Invite:                   ticket ...{invite.tail}, code hidden")
    ctx.say(f"{TRUST} The worker has no password. It listens only on this computer, and the host reaches it")
    ctx.say("through the encrypted Iroh tunnel.")


def check_not_sharing(ctx: Ctx) -> None:
    """Step 1. Never kills a process: `leave` asks the running agent to stop its own worker."""
    cmd = ctx.cli("status", "--port", str(ctx.port))
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(cmd)} (if this computer already shares, it offers to stop first)")
        return
    if not ctx.runner.capture(cmd, timeout=30).stdout.startswith("Sharing as a friend node:"):
        return
    if not ask_yes(ctx, "This computer is already sharing. Stop and join again? [y/N]", False):
        raise Stop("Still sharing. Nothing was changed.", 0)
    res = ctx.runner.capture(ctx.cli("leave"), timeout=180)
    ctx.show_lines(res)
    if res.returncode != 0:
        raise Stop("Could not stop the current sharing. Nothing else was changed.")


def first_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _engine_text(res: Result) -> str:
    info = first_json(res.stdout) or {}
    return f"llama.cpp {info.get('release') or '?'} [{info.get('variant') or '?'}], {info.get('smoke_detail') or ''}"


def ensure_engine(ctx: Ctx) -> None:
    """Step 2. The install runs in this window without --yes, so the CLI shows each file and asks first."""
    status, install = ctx.cli("engine", "status"), ctx.cli("engine", "install")
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(status)} (exit 0 means the engine passed its self-test)")
        ctx.note(f"if it is not ready: {show_cmd(install)} in this window. It lists each file and asks first.")
        return
    res = ctx.runner.capture(status, timeout=600)
    if res.returncode == 0:
        ctx.say(f"Engine ready: {_engine_text(res)}")
        return
    ctx.say(f"The inference engine is not ready: {_engine_text(res)}")
    ctx.say("Installing it now. It lists each file and asks before it downloads anything.")
    if ctx.runner.attached(install) != 0:
        raise Stop("The engine was not installed. Nothing else was changed.")
    res = ctx.runner.capture(status, timeout=600)
    if res.returncode != 0:
        raise Stop(f"The engine still fails its self-test: {_engine_text(res)}")


def run_doctor(ctx: Ctx) -> None:
    """Step 3, for information only: its LAN and worker port checks do not apply to Iroh."""
    cmd = ctx.cli("doctor")
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(cmd)} (shown for information, it does not block)")
        return
    ctx.say("Checking this computer (for information only):")
    res = ctx.runner.capture(cmd, timeout=600)
    ctx.show_lines(res, "  ")
    if res.returncode != 0:
        ctx.say("Checks about LAN addresses and worker ports do not apply to Iroh sharing.")


def load_inventory(ctx: Ctx) -> dict[str, Any] | None:
    """Step 5. A dry run reads --inventory-file instead, or plans with placeholders."""
    cmd = ctx.cli("inventory")
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(cmd)}")
        if ctx.inventory_file is None:
            ctx.note("no --inventory-file given, so device and memory are placeholders")
            return None
        ctx.note(f"using the inventory in {ctx.inventory_file}")
        text = ctx.inventory_file.read_text(encoding="utf-8")
    else:
        res = ctx.runner.capture(cmd, timeout=600)
        if res.returncode != 0:
            ctx.show_lines(res)
            raise Stop("Could not read this computer's hardware (hypercompute inventory failed).")
        text = res.stdout
    inv = first_json(text)
    if inv is None:
        raise Stop("Could not read this computer's hardware inventory.")
    return inv


def run_join(ctx: Ctx) -> int:
    if ctx.dry_run:
        ctx.note("Nothing is started, installed or sent. Commands are shown, not run.")
    check_not_sharing(ctx)
    ensure_engine(ctx)
    run_doctor(ctx)
    invite = read_invite(ctx)
    plan = build_plan(ctx, load_inventory(ctx))
    show_consent(ctx, plan, invite)
    cmd = compose_join(ctx.python, plan)
    status = ctx.cli("status", "--port", str(ctx.port))
    if ctx.dry_run:
        return dry_finish(ctx, "join", cmd, f"a line starting with 'Joined.', then run {show_cmd(status)}")
    if ctx.env.get("HC_ONECLICK_YES") == "1":
        ctx.say("Share this computer now? yes (HC_ONECLICK_YES=1)")
    elif not ask_yes(ctx, "Share this computer now? [y/N]", False):
        raise Stop("Not sharing. Nothing was changed.")
    proc, log = launch(ctx, "join", cmd, invite)
    ctx.say(f"Connecting to the host. The log is {log}")
    outcome = watch(ctx, proc, log, lambda lines: any(line.startswith("Joined.") for line in lines))
    if outcome == "timeout":
        ctx.say("Still connecting. Open hypercompute-status in a minute to check.")
        return 0
    if outcome != "ok":
        return 1
    ctx.show_lines(ctx.runner.capture(status, timeout=30))
    ctx.say("Sharing. You can close this window. To stop, open hypercompute-leave.")
    return 0


def dry_finish(ctx: Ctx, kind: str, cmd: list[str], done: str) -> int:
    if kind == "join":
        ctx.note("would ask: Share this computer now? [y/N]")
    ctx.note(f"would run in the background, logging to {ctx.state_dir / (kind + '.log')}:")
    ctx.say(f"  {show_cmd(cmd)}")
    ctx.note(f"then wait up to {WAIT_S} s for {done}")
    if kind == "join":
        ctx.note("the ticket and code reach the background helper in its environment and never reach a log")
        ctx.note(ARGV_NOTE)
    return 0


# ----------------------------------------------------------------------------- background run


def ensure_state_dir(ctx: Ctx) -> Path:
    d = ctx.state_dir
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        d.chmod(0o700)
    return d


def launch(ctx: Ctx, kind: str, cmd: list[str], invite: Invite | None = None) -> tuple[Any, Path]:
    """Spec 2.6: rotate the log, then start this file as a detached supervisor of `cmd`."""
    d = ensure_state_dir(ctx)
    log, pid = d / f"{kind}.log", d / f"{kind}.pid"
    if log.exists():
        os.replace(log, d / f"{kind}.log.1")
    argv = [ctx.python, str(Path(__file__).resolve()), "_supervise", "--log", str(log), "--pidfile", str(pid),
            "--", *cmd]
    env = {k: v for k, v in ctx.env.items() if k not in ("HC_INVITE", "HC_ONECLICK_TICKET", "HC_ONECLICK_CODE")}
    if invite is not None:
        env.update(HC_ONECLICK_TICKET=invite.ticket, HC_ONECLICK_CODE=invite.code)
    try:
        return ctx.runner.background(argv, env), log
    except OSError as exc:
        raise Stop(f"Could not start the background process: {exc}") from None


def read_lines(log: Path) -> list[str]:
    try:
        return log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def report_error(ctx: Ctx, lines: list[str]) -> bool:
    for i, line in enumerate(lines):
        if line.startswith("error:"):
            ctx.say(line)
            if i + 1 < len(lines) and lines[i + 1].startswith("fix:"):
                ctx.say(lines[i + 1])
            return True
    return False


def watch(ctx: Ctx, proc: Any, log: Path, done: Callable[[list[str]], bool]) -> str:
    """Spec 2.4 step 14: 'ok', 'error', 'exited' or 'timeout'."""
    deadline = ctx.clock() + WAIT_S
    while True:
        lines = read_lines(log)
        if done(lines):
            return "ok"
        if proc.poll() is not None or any(line.startswith("error:") for line in lines):
            end = ctx.clock() + 5
            while proc.poll() is None and ctx.clock() < end:  # let the fix: line and the exit line arrive
                ctx.sleep(0.5)
            lines = read_lines(log)
            if report_error(ctx, lines):
                return "error"
            ctx.say("hypercompute stopped. The last lines of its log:")
            for line in lines[-15:]:
                ctx.say("  " + line)
            return "exited"
        if ctx.clock() >= deadline:
            return "timeout"
        ctx.sleep(1.0)


# ----------------------------------------------------------------------------- leave, status, host, invite


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):  # a file removed while counting does not count
                total += os.path.getsize(os.path.join(root, name))
    return total


def size_text(size: int) -> str:
    return f"{size / GIB:.1f} GiB" if size >= GIB // 10 else f"{size / 2**20:.0f} MiB"


def run_leave(ctx: Ctx) -> int:
    size = dir_size(ctx.home / "rpc-cache")
    purge = size > 0 and ask_yes(ctx, f"Also delete the model cache ({size_text(size)})? [y/N]", False)
    cmd = ctx.cli("leave", *(["--purge-cache"] if purge else []))
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(cmd)}")
        ctx.note(f"after a clean leave: remove {ctx.state_dir / 'join.pid'}")
        return 0
    res = ctx.runner.capture(cmd, timeout=180)
    ctx.show_lines(res)
    if res.returncode != 0:
        if "could not be stopped" in res.stdout:
            ctx.say("Run hypercompute-leave again as Administrator on Windows, or with sudo on macOS and Linux.")
        return 1
    (ctx.state_dir / "join.pid").unlink(missing_ok=True)
    return 0


def run_status(ctx: Ctx) -> int:
    cmd = ctx.cli("status", "--port", str(ctx.port))
    rc = 0
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(cmd)}")
    else:
        res = ctx.runner.capture(cmd, timeout=30)
        ctx.show_lines(res)
        rc = res.returncode
        if "no host running" in res.stdout.lower():
            ctx.say(f"(checked port {ctx.port}; set HC_PORT if your host uses another port)")
    for kind in ("join", "host"):
        log = ctx.state_dir / f"{kind}.log"
        if log.exists():
            ctx.say(f"Last lines of {log}:")
            for line in read_lines(log)[-10:]:
                ctx.say("  " + line)
    return rc


def host_command(ctx: Ctx) -> list[str]:
    """Never --bind or --relay-url. The environment passes through, HYPERCOMPUTE_LLAMA_BIN_DIR included."""
    cmd = ctx.cli("host", "--transport", "iroh", "--port", str(ctx.port), "--no-browser")
    mem, devs = ctx.env.get("HC_LOCAL_MAX_MEMORY"), ctx.env.get("HC_LOCAL_DEVICES")
    if mem:
        if not SIZE_RE.fullmatch(mem):
            raise Stop("HC_LOCAL_MAX_MEMORY must look like 8GiB.")
        cmd += ["--local-max-memory", mem]
    if devs:
        if not DEVICES_RE.fullmatch(devs):
            raise Stop("HC_LOCAL_DEVICES must look like CUDA0 or MTL0,CPU.")
        cmd += ["--local-devices", devs]
    return cmd


def run_host(ctx: Ctx) -> int:
    url = f"http://127.0.0.1:{ctx.port}/"
    cmd = host_command(ctx)
    if ctx.dry_run:
        ctx.note("Nothing is started, installed or sent. Commands are shown, not run.")
        ctx.note(f"would check: GET {url}api/health (3 s). If a host answers, it opens {url} and makes an invite.")
        ensure_engine(ctx)
        dry_finish(ctx, "host", cmd, f"{url}api/health to answer, then open {url}")
        return run_invite(ctx)
    if ctx.health(ctx.port):
        ctx.say(f"A host is already running on this computer. Dashboard: {url}")
        ctx.open_url(url)
        return run_invite(ctx)
    ensure_engine(ctx)
    proc, log = launch(ctx, "host", cmd)
    ctx.say(f"Starting the host. This takes about 10 seconds. The log is {log}")
    outcome = watch(ctx, proc, log, lambda _lines: ctx.health(ctx.port))
    if outcome == "timeout":
        ctx.say(f"The host did not answer within {WAIT_S} s. Open hypercompute-status, or read {log}")
    if outcome != "ok":
        return 1
    ctx.say(f"Host running. Dashboard (this computer only): {url}")
    ctx.open_url(url)
    return run_invite(ctx)


def clipboard_tools() -> list[list[str]]:
    if sys.platform == "darwin":
        candidates = [["pbcopy"]]
    elif os.name == "nt":
        candidates = [["clip"]]
    else:
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    found = []
    for tool in candidates:
        exe = shutil.which(tool[0])
        if exe:
            found.append([exe, *tool[1:]])
    return found


def write_invite_file(ctx: Ctx, line: str) -> Path:
    path = ensure_state_dir(ctx) / "invite.txt"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    if os.name != "nt":
        path.chmod(0o600)
    return path


def local_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().strftime("%H:%M")
    except ValueError:
        return iso or "15 minutes from now"


def run_invite(ctx: Ctx) -> int:
    """The CLI's output is captured and never printed. Only error: and fix: lines, which hold no secret."""
    cmd = ctx.cli("invite", "--port", str(ctx.port))
    tools = clipboard_tools()
    if ctx.dry_run:
        ctx.note(f"would run: {show_cmd(cmd)} (its output is captured and never printed)")
        target = show_cmd(tools[0]) if tools else f"no clipboard tool found, so {ctx.state_dir / 'invite.txt'}"
        ctx.note(f"would copy the invite line with: {target}")
        return 0
    res = ctx.runner.capture(cmd, timeout=60)
    if res.returncode != 0:
        for line in res.lines:
            if line.startswith(("error:", "fix:")):
                ctx.say(line)
        ctx.say(f"Looked for a host on 127.0.0.1:{ctx.port}. Start it with hypercompute-host, or set HC_PORT to the "
                "port your host uses.")
        return 1
    lines = res.stdout.splitlines()
    line = next((x.strip() for x in lines if "--ticket" in x), None)
    expires = next((x.split(":", 1)[1].strip() for x in lines if x.startswith("Expires:")), "")
    if not line:
        ctx.say("The host answered, but it made no Iroh invite. Start the host with hypercompute-host.")
        return 1
    tail, when = (token_after(line.split(), "--ticket") or "")[-4:], local_time(expires)
    if any(ctx.runner.feed(tool, line) == 0 for tool in tools):
        ctx.say(f"Invite copied. Send it to one friend in a private message. It works once and expires at {when} "
                "(15 minutes).")
    else:
        path = write_invite_file(ctx, line)
        ctx.say(f"No clipboard tool worked, so the invite is saved in {path}")
        ctx.say(f"Send that line to one friend in a private message, then delete the file. It works once and "
                f"expires at {when} (15 minutes).")
    ctx.say(f"Ticket ...{tail}, code hidden.")
    return 0


# ----------------------------------------------------------------------------- entry point

ACTIONS: dict[str, Callable[[Ctx], int]] = {
    "join": run_join, "leave": run_leave, "status": run_status, "host": run_host, "invite": run_invite,
}


EXTRA_ARGS = ("Run this with nothing after it, or only --dry-run. It asks you to paste the invite. "
              "The extra words were not shown, in case they hold an invite.")


class RedactingParser(argparse.ArgumentParser):
    """argparse whose usage and error text go through the same redaction as every other printed line (spec 2.6)."""

    def error(self, message: str) -> NoReturn:
        sys.stderr.write(redact_generic(self.format_usage()))
        sys.stderr.write(f"{self.prog}: error: {redact_generic(message)}\n")
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    p = RedactingParser(prog="oneclick.py", description="Hypercompute one-click actions.")
    p.add_argument("action", choices=list(ACTIONS))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--port", type=int, default=None, help="gateway port (default HC_PORT or 8001)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run and launch nothing")
    p.add_argument("--inventory-file", type=Path, help="dry run only: an inventory JSON to plan with")
    return p


def make_ctx(args: argparse.Namespace) -> Ctx:
    port = args.port if args.port is not None else os.environ.get("HC_PORT", str(DEFAULT_PORT))
    if not str(port).isdigit() or not 1 <= int(port) <= 65535:
        raise Stop("The port (--port or HC_PORT) must be a number from 1 to 65535.", 2)
    if args.inventory_file is not None and not args.dry_run:
        raise Stop("--inventory-file only works with --dry-run.", 2)
    home_env = os.environ.get("HYPERCOMPUTE_HOME")
    home = Path(home_env).expanduser() if home_env else Path.home() / ".hypercompute"
    repo = args.repo.resolve()
    return Ctx(repo=repo, port=int(port), dry_run=args.dry_run, home=home, runner=Runner(repo),
               inventory_file=args.inventory_file)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_supervise"]:
        return supervise_main(argv[1:])
    args, extra = build_parser().parse_known_args(argv)
    if extra:  # an invite pasted after the command would land here; never echo it
        print(EXTRA_ARGS, file=sys.stderr)
        return 2
    try:
        return ACTIONS[args.action](make_ctx(args))
    except Stop as stop:
        if stop.message:
            print(redact_generic(stop.message))
        return stop.code
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
