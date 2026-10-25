"""The real (not dry-run) paths of join, leave, status, host and invite, with the command runner stubbed.

Nothing here runs `hypercompute`, uv, a clipboard tool or a background process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import oneclick
import pytest
from oneclick import run_host, run_invite, run_join, run_leave, run_status
from oneclick_core import Result
from oneclick_testkit import CODE, FIXTURES, INVITE_LINE, TICKET, FakeProc, StubRunner, make_ctx

ENGINE_OK = Result(0, json.dumps({"release": "b11770", "variant": "windows-x64-cuda13.4", "smoke": "passed"}))
G14 = Result(0, (FIXTURES / "inventory_g14.json").read_text(encoding="utf-8"))
JOIN_ENV = {"HC_INVITE": INVITE_LINE, "HC_ONECLICK_YES": "1", "PATH": "x"}


def writes_log(*lines: str, code: int | None = None):
    def start(argv, _env) -> FakeProc:
        Path(argv[argv.index("--log") + 1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return FakeProc(code)

    return start


def join_runner(**kw) -> StubRunner:
    answers = {"status": Result(0, "Not sharing and no host running on this machine.\n"), "engine status": ENGINE_OK,
               "inventory": G14, "doctor": Result(1, "[FAIL] no bindable worker port\n")}
    return StubRunner(answers=answers, **kw)


def test_join_happy_path(tmp_path) -> None:
    runner = join_runner(on_background=writes_log("Connecting...", "Joined. The host is verifying..."))
    ctx, out = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, nvidia=True)
    assert run_join(ctx) == 0
    kind, argv = next(c for c in runner.calls if c[0] == "background")
    assert argv[1] == str(Path(oneclick.__file__).resolve()) and argv[2] == "_supervise"
    child = argv[argv.index("--") + 1:]
    assert child[3:9] == ["join", "--ticket", "<ticket>", "--code", "<code>", "--name"]
    assert runner.background_env["HC_ONECLICK_TICKET"] == TICKET and runner.background_env["HC_ONECLICK_CODE"] == CODE
    assert "HC_INVITE" not in runner.background_env
    assert "Sharing. You can close this window. To stop, open hypercompute-leave." in out
    assert "Checks about LAN addresses and worker ports do not apply to Iroh sharing." in out
    assert all(TICKET not in line and CODE not in line for line in out)
    assert (ctx.state_dir).is_dir()


def test_join_rotates_the_previous_log(tmp_path) -> None:
    runner = join_runner(on_background=writes_log("Joined."))
    ctx, _ = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, nvidia=True)
    ctx.state_dir.mkdir(parents=True)
    (ctx.state_dir / "join.log").write_text("old run\n", encoding="utf-8")
    assert run_join(ctx) == 0
    assert (ctx.state_dir / "join.log.1").read_text(encoding="utf-8") == "old run\n"


def test_join_error_prints_error_and_fix(tmp_path) -> None:
    runner = join_runner(on_background=writes_log(
        "error: engine mismatch: host runs llama.cpp b1, this machine b2",
        "fix:   install the same pinned release on both machines", "hypercompute exited with code 2", code=2))
    ctx, out = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, nvidia=True)
    assert run_join(ctx) == 1
    assert out[-2:] == ["error: engine mismatch: host runs llama.cpp b1, this machine b2",
                        "fix:   install the same pinned release on both machines"]


def test_join_child_exit_prints_the_log_tail(tmp_path) -> None:
    runner = join_runner(on_background=writes_log("something odd", "hypercompute exited with code 1", code=1))
    ctx, out = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, nvidia=True)
    assert run_join(ctx) == 1
    assert "  hypercompute exited with code 1" in out


def test_join_timeout_points_at_status(tmp_path) -> None:
    runner = join_runner(on_background=writes_log("Connecting to the host through Iroh..."))
    ctx, out = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, nvidia=True)
    assert run_join(ctx) == 0
    assert out[-1] == "Still connecting. Open hypercompute-status in a minute to check."


def test_join_needs_a_yes(tmp_path) -> None:
    runner = join_runner()
    env = {k: v for k, v in JOIN_ENV.items() if k != "HC_ONECLICK_YES"}
    ctx, _ = make_ctx(tmp_path, runner=runner, env=env, nvidia=True)
    with pytest.raises(oneclick.Stop, match="Not sharing"):
        run_join(ctx)
    assert not any(c[0] == "background" for c in runner.calls)


def test_already_sharing_and_keep_it(tmp_path) -> None:
    runner = StubRunner(answers={"status": Result(0, "Sharing as a friend node:\n  state: ready\n")})
    ctx, _ = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, answers=["n"])
    with pytest.raises(oneclick.Stop) as stop:
        run_join(ctx)
    assert stop.value.code == 0
    assert [StubRunner.key(c[1]) for c in runner.calls] == ["status --port 8001"]


def test_already_sharing_then_leave_first(tmp_path) -> None:
    runner = join_runner(on_background=writes_log("Joined."))
    runner.answers["status"] = Result(0, "Sharing as a friend node:\n")
    ctx, _ = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, answers=["y"], nvidia=True)
    run_join(ctx)
    assert "leave" in [StubRunner.key(c[1]) for c in runner.calls]


def test_engine_install_runs_attached_then_rechecks(tmp_path) -> None:
    runner = join_runner(on_background=writes_log("Joined."))
    runner.answers["engine status"] = Result(1, json.dumps({"smoke": "failed", "smoke_detail": "not installed"}))
    ctx, out = make_ctx(tmp_path, runner=runner, env=JOIN_ENV, nvidia=True)
    with pytest.raises(oneclick.Stop, match="still fails"):
        run_join(ctx)
    assert ("attached", [ctx.python, "-m", "hypercompute.cli", "engine", "install"]) in runner.calls


def test_leave_asks_about_the_cache(tmp_path) -> None:
    runner = StubRunner(answers={"leave": Result(0, "Stopped.\n")})
    ctx, out = make_ctx(tmp_path, runner=runner, answers=["y"])
    (ctx.home / "rpc-cache").mkdir(parents=True)
    (ctx.home / "rpc-cache" / "t.bin").write_bytes(b"x" * 1024)
    ctx.state_dir.mkdir()
    (ctx.state_dir / "join.pid").write_text("{}", encoding="utf-8")
    assert run_leave(ctx) == 0
    assert StubRunner.key(runner.calls[-1][1]) == "leave --purge-cache"
    assert any("Also delete the model cache (0 MiB)? [y/N]" in line for line in out)
    assert not (ctx.state_dir / "join.pid").exists()


def test_leave_stuck_worker_says_administrator(tmp_path) -> None:
    stuck = Result(1, "error: a worker started by Hypercompute is still running and could not be stopped\n"
                      "fix:   run `hypercompute leave` again from an Administrator PowerShell\n")
    ctx, out = make_ctx(tmp_path, runner=StubRunner(answers={"leave": stuck}))
    assert run_leave(ctx) == 1
    assert "as Administrator on Windows" in out[-1]


def test_status_prints_log_tails(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, runner=StubRunner(answers={"status": Result(0, "Hosting on port 8001:\n")}))
    ctx.state_dir.mkdir(parents=True)
    (ctx.state_dir / "host.log").write_text("\n".join(f"line {i}" for i in range(30)) + "\n", encoding="utf-8")
    assert run_status(ctx) == 0
    assert "Hosting on port 8001:" in out and "  line 29" in out and "  line 19" not in out


def test_invite_copies_to_the_clipboard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oneclick, "clipboard_tools", lambda: [["/usr/bin/pbcopy"]])
    runner = StubRunner(answers={"invite": Result(0, f"{INVITE_LINE}\nExpires: 2026-10-24T12:15:00Z\n")})
    ctx, out = make_ctx(tmp_path, runner=runner)
    assert run_invite(ctx) == 0
    assert ("feed", ["/usr/bin/pbcopy"]) in runner.calls
    assert out[0].startswith("Invite copied. Send it to one friend in a private message. It works once")
    assert out[-1] == f"Ticket ...{TICKET[-4:]}, code hidden."
    assert all(TICKET not in line and CODE not in line for line in out)


def test_invite_without_clipboard_writes_a_private_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oneclick, "clipboard_tools", lambda: [])
    runner = StubRunner(answers={"invite": Result(0, f"{INVITE_LINE}\nExpires: 2026-10-24T12:15:00Z\n")})
    ctx, out = make_ctx(tmp_path, runner=runner)
    assert run_invite(ctx) == 0
    path = ctx.state_dir / "invite.txt"
    assert path.read_text(encoding="utf-8") == INVITE_LINE + "\n"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    assert any(line.endswith(os.path.join("oneclick", "invite.txt")) for line in out)  # printed through redaction
    assert all(TICKET not in line and CODE not in line for line in out)


def test_invite_failure_prints_only_error_and_fix(tmp_path) -> None:
    failed = Result(2, "some other output\n", "error: Could not create an invitation: refused\nfix:   start the host\n")
    ctx, out = make_ctx(tmp_path, runner=StubRunner(answers={"invite": failed}))
    assert run_invite(ctx) == 1
    assert out == ["error: Could not create an invitation: refused", "fix:   start the host",
                   "Looked for a host on 127.0.0.1:8001. Start it with hypercompute-host, or set HC_PORT to the port "
                   "your host uses."]


def test_status_names_the_port_when_no_host_answers(tmp_path) -> None:
    none = Result(0, "Not sharing and no host running on this machine.\n")
    ctx, out = make_ctx(tmp_path, runner=StubRunner(answers={"status": none}))
    assert run_status(ctx) == 0
    assert "(checked port 8001; set HC_PORT if your host uses another port)" in out
    hosting, out2 = make_ctx(tmp_path, runner=StubRunner(answers={"status": Result(0, "Hosting on port 8001:\n")}))
    run_status(hosting)
    assert not any("checked port" in line for line in out2)


def test_host_starts_waits_opens_and_invites(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oneclick, "clipboard_tools", lambda: [["/usr/bin/pbcopy"]])
    started: list[bool] = []
    runner = StubRunner(answers={"engine status": ENGINE_OK, "invite": Result(0, f"{INVITE_LINE}\nExpires: x\n")},
                        on_background=lambda argv, env: started.append(True) or FakeProc())
    ctx, out = make_ctx(tmp_path, runner=runner, env={"HC_LOCAL_DEVICES": "CUDA0"},
                        health=lambda _port: bool(started))
    assert run_host(ctx) == 0
    _, argv = next(c for c in runner.calls if c[0] == "background")
    assert argv[argv.index("--") + 1:][3:] == ["host", "--transport", "iroh", "--port", "8001", "--no-browser",
                                                "--local-devices", "CUDA0"]
    assert "OPEN http://127.0.0.1:8001/" in out


def test_host_already_running_goes_to_invite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oneclick, "clipboard_tools", lambda: [])
    runner = StubRunner(answers={"invite": Result(0, f"{INVITE_LINE}\n")})
    ctx, out = make_ctx(tmp_path, runner=runner, health=lambda _port: True)
    assert run_host(ctx) == 0
    assert not any(c[0] == "background" for c in runner.calls)
    assert out[0].startswith("A host is already running")


def test_background_start_failure_is_a_plain_message(tmp_path) -> None:
    def fail(_argv, _env) -> FakeProc:
        raise OSError("no such file")

    ctx, _ = make_ctx(tmp_path, runner=join_runner(on_background=fail), env=JOIN_ENV, nvidia=True)
    with pytest.raises(oneclick.Stop, match="Could not start the background process"):
        run_join(ctx)


def test_bad_local_memory_is_refused(tmp_path) -> None:
    ctx, _ = make_ctx(tmp_path, env={"HC_LOCAL_MAX_MEMORY": "8GiB; rm"}, dry_run=True)
    with pytest.raises(oneclick.Stop):
        run_host(ctx)


@pytest.mark.parametrize("size", ["8Gi", "8 Ti", "8", "8B", "8 GiBs", "GiB"])
def test_local_memory_needs_a_unit_parse_size_knows(tmp_path, size: str) -> None:
    ctx, _ = make_ctx(tmp_path, env={"HC_LOCAL_MAX_MEMORY": size}, dry_run=True)
    with pytest.raises(oneclick.Stop, match="HC_LOCAL_MAX_MEMORY"):
        oneclick.host_command(ctx)


@pytest.mark.parametrize("size", ["8GiB", "8 gb", "8g", "512MiB", "1.5TiB", "64 KB", "2T"])
def test_local_memory_accepts_parse_size_units(tmp_path, size: str) -> None:
    ctx, _ = make_ctx(tmp_path, env={"HC_LOCAL_MAX_MEMORY": size}, dry_run=True)
    cmd = oneclick.host_command(ctx)
    assert cmd[cmd.index("--local-max-memory") + 1] == size
