"""vLLM installer: the install script and its result record, WSL helpers, the install flow with its disk check and
forced reinstalls, and the record of checkpoints already in vLLM's model cache.

Portable: no WSL, GPU or network; `wsl -l -v` bytes are captured from the G14. Shared helpers live in test_vllm.py.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypercompute import config
from hypercompute.adapters import vllm as adapter_mod
from hypercompute.installers import vllm as installer
from hypercompute.units import GiB
from test_vllm import AWQ_BYTES, FIXTURES, REPO, awq_handle, ctx, installed, plan_of, record, tight_disk


@pytest.fixture
def roomy_drive(monkeypatch):
    """install() checks free disk before it runs the script: these installs never read the real drive."""
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * GiB))


RESULT = {"vllm": "0.32.0", "torch": "2.13.0+cu130", "torch_cuda": "13.0", "cuda_available": True, "device": "RTX",
          "capability": "12.0", "root": "/home/u/.hypercompute-vllm", "venv": "/home/u/.hypercompute-vllm/venv"}


def ok_stream(argv, stdin, on_line) -> int:
    on_line("HC_VLLM_RESULT " + json.dumps(RESULT))
    return 0


# ----------------------------------------------------------------------------- install script and record


def test_install_script_is_lf_and_pins_match_the_installer():
    data = (REPO / "scripts" / installer.SCRIPT_NAME).read_bytes()
    assert b"\r" not in data, "bash inside WSL rejects CR line endings"
    text = data.decode()
    assert f"VLLM_VERSION={installer.VLLM_VERSION} " in text and f"TORCH_BACKEND={installer.TORCH_BACKEND} " in text
    assert 'RESULT ' in installer.RESULT_MARKER and "HC_VLLM_RESULT " in text
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" not in text  # pinned, sha256-checked uv instead
    assert installer.script_bytes() == data


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="bash -n needs a POSIX bash")
def test_install_script_parses():
    subprocess.run(["bash", "-n", str(REPO / "scripts" / installer.SCRIPT_NAME)], check=True)


def test_a_crlf_checkout_is_streamed_without_cr(monkeypatch, tmp_path):
    lf = (REPO / "scripts" / installer.SCRIPT_NAME).read_bytes()
    crlf = tmp_path / installer.SCRIPT_NAME
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))  # core.autocrlf=true without the *.sh eol=lf rule
    monkeypatch.setattr(installer, "script_path", lambda: crlf)
    assert installer.script_bytes() == lf
    monkeypatch.setattr(installer, "wsl_exe", lambda: r"C:\Windows\System32\wsl.exe")
    utf16 = (FIXTURES / "vllm_wsl_list_v_utf16.bin").read_bytes()
    argv, stdin, distro = installer.install_command("windows", False, runner=lambda argv: (0, utf16))
    assert distro == "Ubuntu" and argv[-2:] == [installer.WSL_COPY_AND_RUN, "hc-vllm-install"] and stdin == lf
    argv, stdin, distro = installer.install_command("linux", True)  # native Linux streams the same LF bytes
    assert argv == ["bash", "-c", installer.WSL_COPY_AND_RUN, "hc-vllm-install", "--force"] and stdin == lf
    assert distro is None


def test_parse_result_and_record_validation():
    payload = {"vllm": "0.32.0", "torch": "2.13.0+cu130", "torch_cuda": "13.0", "cuda_available": True,
               "capability": "12.0", "device": "NVIDIA GeForce RTX 5070 Ti Laptop GPU", "python": "3.12.3",
               "root": "/home/u/.hypercompute-vllm", "venv": "/home/u/.hypercompute-vllm/venv", "uv": "uv 0.12.3",
               "wsl": True, "kernels": "ok"}
    lines = ["[hypercompute-vllm] installing", f"HC_VLLM_RESULT {json.dumps(payload)}"]
    rec = installer.record_from_result(installer.parse_result(lines), "Ubuntu")
    assert rec.platform == "wsl2" and rec.distro == "Ubuntu" and rec.capability == "12.0"
    assert rec.vllm_bin == "/home/u/.hypercompute-vllm/venv/bin/vllm" and rec.hf_home.endswith("/hf")
    with pytest.raises(installer.InstallError, match="cannot see the GPU"):
        installer.record_from_result({**payload, "cuda_available": False}, "Ubuntu")
    with pytest.raises(installer.InstallError, match="bad venv"):
        installer.record_from_result({**payload, "root": "/mnt/c/x"}, "Ubuntu")
    assert installer.parse_result(["no marker"]) is None and installer.parse_result(["HC_VLLM_RESULT {bad"]) is None


def test_status_reads_only_the_record(monkeypatch):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    paths = config.paths()
    st = installer.status(paths)
    assert st.status == "not_installed" and "engine install --engine vllm" in (st.detail or "")
    installer.write_install(paths, record())
    st = installer.status(paths)
    assert st.installed and st.status == "installed" and st.version == "0.32.0" and "sm_120" in (st.detail or "")
    installer.write_install(paths, record(vllm="0.31.0"))
    assert installer.status(paths).status == "failed"
    installer.marker_path(paths).write_text("{not json", encoding="utf-8")
    st = installer.status(paths)
    assert st.status == "failed" and "--force" in (st.detail or "")
    monkeypatch.setattr(installer.engine, "host_os", lambda: "macos")
    st = installer.status(paths)
    assert st.status == "unsupported" and "vllm-metal" in (st.detail or "")


def test_linux_record_on_windows_is_rejected(monkeypatch):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    installer.write_install(config.paths(), record("linux"))
    assert "this host runs windows" in (installer.status(config.paths()).detail or "")


# ----------------------------------------------------------------------------- WSL helpers


def test_wsl_list_utf16_capture_and_distro_choice(monkeypatch):
    text = installer.decode_wsl((FIXTURES / "vllm_wsl_list_v_utf16.bin").read_bytes())
    distros = installer.parse_wsl_list(text)
    assert [(d.name, d.version, d.default) for d in distros] == [("Ubuntu", 2, True), ("docker-desktop", 2, False)]
    assert installer.choose_distro(distros).name == "Ubuntu"
    docker_only = [d for d in distros if d.name.startswith("docker")]
    with pytest.raises(installer.InstallError, match="no Linux distro"):
        installer.choose_distro(docker_only)
    wsl1 = [installer.WslDistro("Debian", "Stopped", 1, True)]
    with pytest.raises(installer.InstallError, match="WSL 1") as exc:
        installer.choose_distro(wsl1)
    assert exc.value.fix == "run `wsl --set-version Debian 2`"
    with pytest.raises(installer.InstallError, match="not found"):
        installer.choose_distro(distros, "Fedora")
    assert installer.choose_distro(distros, "ubuntu").name == "Ubuntu"
    listed = installer.parse_wsl_list("  NAME   STATE   VERSION\n* Ubuntu-24.04   Running   2\n")
    assert listed[0].name == "Ubuntu-24.04"


def test_install_streams_the_lf_script_into_wsl(monkeypatch, roomy_drive):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    monkeypatch.setattr(installer, "wsl_exe", lambda: r"C:\Windows\System32\wsl.exe")
    utf16 = (FIXTURES / "vllm_wsl_list_v_utf16.bin").read_bytes()
    seen = {}

    def stream(argv, stdin, on_line):
        seen["argv"], seen["stdin"] = list(argv), stdin
        on_line("[hypercompute-vllm] installing vllm==0.32.0")
        return ok_stream(argv, stdin, on_line)

    said: list[str] = []
    code = installer.install(config.paths(), yes=True, force=False, say=said.append, stream=stream,
                             runner=lambda argv: (0, utf16))
    assert code == 0, said
    argv = seen["argv"]
    assert argv[:7] == [r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu", "--cd", "~", "--exec", "bash"]
    assert argv[8] == installer.WSL_COPY_AND_RUN and "--force" not in argv
    assert b"\r" not in seen["stdin"] and seen["stdin"].startswith(b"#!/usr/bin/env bash\n")
    rec = installer.read_install(config.paths())
    assert rec and rec.distro == "Ubuntu" and rec.platform == "wsl2"
    said.clear()
    assert installer.install(config.paths(), yes=True, force=False, say=said.append, stream=stream) == 0
    assert any("already installed" in s for s in said)


def test_install_failure_reports_exit_code(monkeypatch, roomy_drive):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "linux")
    said: list[str] = []
    code = installer.install(config.paths(), yes=True, force=True, say=said.append,
                             stream=lambda argv, stdin, on_line: (on_line("[hypercompute-vllm] ERROR: x"), 1)[1])
    assert code == 1 and any("exit code 1" in s for s in said) and installer.read_install(config.paths()) is None
    monkeypatch.setattr(installer.engine, "host_os", lambda: "macos")
    assert installer.install(config.paths(), yes=True, force=False, say=said.append) == 2


def test_install_stream_splits_lines_on_a_lone_cr(monkeypatch, roomy_drive):
    """Progress output redraws with a bare CR; the result marker after it must still start its own line."""
    monkeypatch.setattr(installer.engine, "host_os", lambda: "linux")
    out = b"Resolved 1/3\rResolved 3/3\rHC_VLLM_RESULT " + json.dumps(RESULT).encode() + b"\ndone\r\n"
    monkeypatch.setattr(installer, "install_command", lambda system, force, runner=None: (
        [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({out!r})"], None, None))
    said: list[str] = []
    assert installer.install(config.paths(), yes=True, force=True, say=said.append) == 0, said
    streamed = [s[2:] for s in said if s.startswith("  ")]  # through the real stream_command, a child process
    assert streamed == ["Resolved 1/3", "Resolved 3/3", f"HC_VLLM_RESULT {json.dumps(RESULT)}", "done"]
    assert not any("\r" in s for s in said) and installer.read_install(config.paths()) is not None


@pytest.mark.anyio
async def test_install_and_force_install_reset_the_served_record(monkeypatch, roomy_drive):
    """wsl --unregister, a distro switch or a wiped ~/.hypercompute-vllm all end in a reinstall: forget the cache."""
    installed(monkeypatch, "linux")
    adapter = adapter_mod.VllmAdapter(record("linux"), ctx())
    paths = config.paths()
    for force in (True, False):
        adapter._remember_cached(awq_handle())
        assert plan_of(disk=tight_disk).engine_options["download_bytes"] == 0
        if not force:
            installer.marker_path(paths).unlink()  # a plain install runs only when nothing is installed
        said: list[str] = []
        assert installer.install(paths, yes=True, force=force, say=said.append, stream=ok_stream) == 0, said
        assert installer.read_install(paths).hf_home == record("linux").hf_home  # same place: only the reset forgets
        plan = plan_of(disk=tight_disk)
        assert not plan.feasible and plan.engine_options["download_bytes"] == AWQ_BYTES, force
    await adapter.aclose()


def test_cache_disk_and_served_record(monkeypatch, tmp_path):
    linux = record("linux", root=str(tmp_path / "missing" / "root"))
    free, where = installer.cache_disk(linux)
    assert free is not None and free > 0 and where == linux.hf_home  # walks up to an existing parent
    monkeypatch.setattr(installer, "wsl_base_path", lambda distro: tmp_path)
    free, where = installer.cache_disk(record())
    assert free is not None and "WSL2 distro Ubuntu" in where
    monkeypatch.setattr(installer, "wsl_base_path", lambda distro: None)
    assert installer.cache_disk(record())[0] is None
    paths, rec = config.paths(), record()
    assert installer.served_models(paths) == set() and not installer.is_served(paths, rec, "org/repo", "abc")
    installer.remember_served(paths, rec, "org/repo", "abc")
    installer.remember_served(paths, rec, "org/repo", "abc")
    assert installer.served_models(paths) == {"Ubuntu|/home/u/.hypercompute-vllm/hf|org/repo@abc"}
    assert installer.is_served(paths, rec, "org/repo", "abc") and not installer.is_served(paths, rec, "org/repo", "x")
    assert not installer.is_served(paths, record("linux"), "org/repo", "abc")  # native Linux is another cache
    installer.served_path(paths).write_text(json.dumps(["org/repo@abc"]), encoding="utf-8")  # pre-keying record
    assert not installer.is_served(paths, rec, "org/repo", "abc")  # unknown location: check the disk again
    installer.served_path(paths).write_text("{bad", encoding="utf-8")
    assert installer.served_models(paths) == set()


def test_wsl_base_path_of_an_unknown_distro_is_none():
    assert installer.wsl_base_path("hypercompute-no-such-distro") is None  # registry read on Windows, else no winreg


# ----------------------------------------------------------------------------- disk room and forced reinstalls


def test_install_checks_room_on_the_drive_that_holds_the_virtual_disk(monkeypatch, tmp_path):
    """The distro's ext4.vhdx grows on a Windows drive that `df` inside WSL cannot see (it reports the 1 TB cap)."""
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    monkeypatch.setattr(installer, "wsl_exe", lambda: r"C:\Windows\System32\wsl.exe")
    monkeypatch.setattr(installer, "wsl_base_path", lambda distro: tmp_path if distro == "Ubuntu" else None)
    free, probed, streamed = [12 * GiB], [], []
    monkeypatch.setattr(installer.shutil, "disk_usage",
                        lambda p: probed.append(Path(p)) or SimpleNamespace(free=free[0]))
    utf16 = (FIXTURES / "vllm_wsl_list_v_utf16.bin").read_bytes()
    measured = f"{int(9.8 * GiB)}\n".encode()  # RECLAIM: what deleting the old venv frees

    def install(force: bool) -> tuple[int, list[str]]:
        said: list[str] = []
        code = installer.install(config.paths(), yes=True, force=force, say=said.append,
                                 stream=lambda *a: (streamed.append(a[0]), ok_stream(*a))[1],
                                 runner=lambda argv: (0, measured if installer.RECLAIM in argv else utf16))
        return code, said

    code, said = install(force=False)
    assert code == 1 and not streamed and probed == [tmp_path]  # refused before anything ran inside WSL
    assert any(s.startswith("error:") and "needs ~20.0 GiB free" in s and "has 12.0 GiB" in s
               and "WSL2 distro Ubuntu's virtual disk" in s for s in said), said
    assert any(s.startswith("fix:") and "free at least 8.0 GiB" in s for s in said)
    free[0] = 30 * GiB
    assert install(force=False)[0] == 0 and len(streamed) == 1
    free[0] = 12 * GiB  # --force over that install deletes its venv first: 12 + 9.8 GiB freed is room enough
    code, said = install(force=True)
    assert code == 0 and len(streamed) == 2
    assert any(s.startswith("warning:") and "has 12.0 GiB" in s and "frees ~9.8 GiB" in s for s in said), said
    assert not any("reused" in s for s in said)
    monkeypatch.setattr(installer, "wsl_base_path", lambda distro: None)  # not registered: unknown never blocks
    code, said = install(force=True)
    assert code == 0 and any(s.startswith("warning:") and "could not be read" in s for s in said)


LIST_UBUNTU_DEBIAN = b"  NAME   STATE     VERSION\r\n* Ubuntu Running   2\r\n  Debian Stopped   2\r\n"  # `wsl -l -v`


def test_force_reinstall_counts_only_its_own_venv_and_what_deleting_it_frees(monkeypatch, tmp_path):
    """A record for another distro frees nothing on the target's virtual disk; an old venv whose packages uv's cache
    still hardlinks frees less than its size. Both must be refused like a fresh install when the drive is short."""
    monkeypatch.setattr(installer.engine, "host_os", lambda: "windows")
    monkeypatch.setattr(installer, "wsl_exe", lambda: r"C:\Windows\System32\wsl.exe")
    monkeypatch.setattr(installer, "wsl_base_path", lambda distro: tmp_path)
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda p: SimpleNamespace(free=5 * GiB))
    installer.write_install(config.paths(), record())  # vLLM installed in Ubuntu
    calls: list[list[str]] = []
    measured = [b"none\n"]

    def runner(argv):
        calls.append(list(argv))
        return (0, measured[0]) if installer.RECLAIM in argv else (0, LIST_UBUNTU_DEBIAN)

    def install(distro: str) -> tuple[int, list[str], list]:
        monkeypatch.setenv(installer.DISTRO_ENV, distro)
        said, streamed = [], []
        calls.clear()
        code = installer.install(config.paths(), yes=True, force=True, say=said.append, runner=runner,
                                 stream=lambda *a: (streamed.append(a[0]), ok_stream(*a))[1])
        return code, said, streamed

    code, said, streamed = install("Debian")  # Debian has its own virtual disk: the Ubuntu venv frees nothing there
    assert code == 1 and not streamed and not any(installer.RECLAIM in c for c in calls), said
    assert any(s.startswith("fix:") and "free at least 15.0 GiB" in s for s in said)
    assert installer.read_install(config.paths()).distro == "Ubuntu"  # refused before the record is touched
    code, said, streamed = install("Ubuntu")  # same distro, but the recorded venv is not there ("none")
    reclaim = [c for c in calls if installer.RECLAIM in c]
    assert code == 1 and not streamed and reclaim == [[r"C:\Windows\System32\wsl.exe", "-d", "Ubuntu", "--cd", "~",
                                                       "--exec", "bash", "-c", installer.RECLAIM,
                                                       "/home/u/.hypercompute-vllm/venv"]]
    measured[0] = f"{int(9.8 * GiB)}\n".encode()  # 17 GB venv, ~8 GB of it hardlinked from ~/.cache/uv
    code, said, streamed = install("Ubuntu")
    assert code == 1 and not streamed and any("plus ~9.8 GiB from deleting the old venv" in s for s in said), said
    assert any(s.startswith("fix:") and "free at least 5.2 GiB" in s for s in said)
    measured[0] = b"bash: find: command not found\n"  # unmeasured counts as nothing freed
    assert install("Ubuntu")[0] == 1
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda p: SimpleNamespace(free=12 * GiB))
    measured[0] = f"{int(9.8 * GiB)}\n".encode()
    code, said, streamed = install("ubuntu")  # distro names are case-insensitive in WSL
    assert code == 0 and len(streamed) == 1 and any("frees ~9.8 GiB" in s for s in said), said


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("find") is None, reason="needs GNU find and a POSIX fs")
def test_reclaim_script_counts_only_files_no_other_path_links(tmp_path, monkeypatch):
    root, cache = tmp_path / "root", tmp_path / "cache"
    (root / "venv").mkdir(parents=True)
    cache.mkdir()
    (root / "venv" / "own.bin").write_bytes(b"x" * 1024 * 1024)
    (cache / "wheel.bin").write_bytes(b"y" * 4 * 1024 * 1024)
    (root / "venv" / "linked.bin").hardlink_to(cache / "wheel.bin")
    monkeypatch.setenv("HC_VLLM_ROOT", str(root))
    out = subprocess.run(["bash", "-c", installer.RECLAIM, str(root / "venv")], capture_output=True, text=True,
                         check=True).stdout.split()
    assert 1024 * 1024 <= int(out[-1]) < 2 * 1024 * 1024  # own.bin only: linked.bin stays in the cache
    out = subprocess.run(["bash", "-c", installer.RECLAIM, str(tmp_path / "other" / "venv")], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["none"]  # not the venv the script's --force deletes


def test_install_on_linux_checks_the_install_root(monkeypatch, tmp_path):
    monkeypatch.setattr(installer.engine, "host_os", lambda: "linux")
    monkeypatch.setenv("HC_VLLM_ROOT", str(tmp_path / "missing" / "root"))  # the script's root override
    probed: list[Path] = []
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda p: probed.append(Path(p)) or SimpleNamespace(free=GiB))
    said: list[str] = []
    assert installer.install(config.paths(), yes=True, force=False, say=said.append, stream=ok_stream) == 1
    assert probed == [tmp_path] and any(str(tmp_path / "missing" / "root") in s for s in said)
    assert installer.read_install(config.paths()) is None


def test_install_script_leaves_no_wheel_cache_behind():
    """In ~/.cache/uv the ~8 GB of wheels would outlive the venv, inside a virtual disk that never shrinks."""
    text = (REPO / "scripts" / installer.SCRIPT_NAME).read_text(encoding="utf-8")
    body = text[text.index("install_vllm() {"):text.index("report() {")]
    pip, rm = body.index('UV_CACHE_DIR="$UV_CACHE" "$UV" pip install'), body.index('rm -rf "$UV_CACHE"')
    default = re.search(r"^UV_CACHE=(\$\{HC_VLLM_UV_CACHE:-)?\$ROOT/uv-cache\}?\s", text, re.MULTILINE)
    assert default and pip < rm < body.index("download-kernels")


def test_a_failed_force_reinstall_reads_as_not_installed(monkeypatch, roomy_drive):
    """The script's --force deletes the venv before rebuilding it, so a later failure must not keep the old record."""
    monkeypatch.setattr(installer.engine, "host_os", lambda: "linux")
    monkeypatch.setattr(adapter_mod.engine, "host_os", lambda: "linux")
    monkeypatch.setattr(installer, "_confirm", lambda prompt: False)
    paths, said = config.paths(), []
    assert installer.install(paths, yes=True, force=False, say=said.append, stream=ok_stream) == 0
    installer.remember_served(paths, installer.read_install(paths), "org/repo", "abc")
    assert installer.install(paths, yes=False, force=True, say=said.append) == 1  # declined: nothing changes
    assert installer.status(paths).installed and installer.served_models(paths)

    def removed_then_failed(argv, stdin, on_line) -> int:
        on_line("[hypercompute-vllm] removing /home/u/.hypercompute-vllm/venv (--force)")
        on_line("error: Failed to fetch: network unreachable")
        return 1

    assert installer.install(paths, yes=True, force=True, say=said.append, stream=removed_then_failed) == 1
    st = installer.status(paths)
    assert not st.installed and st.status == "not_installed" and installer.served_models(paths) == set()
    assert adapter_mod.build(ctx()) is None  # plans and the selector stop offering the venv that is gone
    said.clear()  # and a plain re-run installs again instead of answering "already installed"
    assert installer.install(paths, yes=True, force=False, say=said.append, stream=ok_stream) == 0
    assert not any("already installed" in s for s in said) and installer.status(paths).installed
