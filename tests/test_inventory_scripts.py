"""Portable tests for scripts/inventory_{linux,macos}.sh and scripts/inventory_windows.ps1.

The shell tests run the real scripts under bash with stub probe tools on PATH, so they exercise the
JSON escaping, the python and pure-bash writers, redaction, timeouts and missing-tool handling on any
host with bash. The Windows tests run the real script under Windows PowerShell and check the bytes it
writes. They say nothing about real hardware; hardware fixtures live in tests/fixtures.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _find_bash() -> str | None:
    if os.name == "nt":
        # System32\bash.exe is the WSL launcher and would run the script inside a Linux VM.
        for candidate in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(candidate).exists():
                return candidate
        return None
    return shutil.which("bash")


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash (Git Bash on Windows) not available")
# Windows PowerShell 5.1 specifically: its ">" and native pipes re-encode text, which -OutFile avoids.
POWERSHELL = shutil.which("powershell") if os.name == "nt" else None
needs_powershell = pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell not available")

# Bytes deliberately hostile to a hand-rolled JSON escaper.
TRICKY_PRINTF = r'quote" back\\slash\ttab\rCR \001\033[1mesc\033[0m caf\303\251 \342\234\223 \177del %%s\n\n'
TRICKY_TEXT = 'quote" back\\slash\ttab\rCR \x01\x1b[1mesc\x1b[0m café ✓ \x7fdel %s\n\n'
# Three MACs in a row: each regex match consumes its trailing separator, so one pass misses the middle one.
MAC_RUN = "00:11:22:33:44:55 66:77:88:99:aa:bb cc:dd:ee:ff:00:11"


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _stub(dir_: Path, name: str, body: str) -> None:
    path = dir_ / name
    path.write_bytes(("#!/bin/sh\n" + body + "\n").encode())
    path.chmod(0o755)


def _run(script: str, stub_dir: Path, mode: str, timeout_s: int = 1) -> dict:
    env = dict(os.environ)
    env["HC_INVENTORY_JSON"] = mode
    env["HC_PYTHON"] = _posix(sys.executable)
    env["HC_PROBE_TIMEOUT_S"] = str(timeout_s)
    # Prepend the stubs inside bash: Git Bash converts the inherited Windows PATH at startup.
    # cygpath turns C:/x into /c/x, since a drive colon would split PATH.
    prelude = 'd=$1; if command -v cygpath >/dev/null 2>&1; then d=$(cygpath -u "$d"); fi; PATH="$d:$PATH"; '
    cmd = [BASH, "-c", prelude + 'exec bash "$2"', "bash", _posix(stub_dir), _posix(SCRIPTS / script)]
    proc = subprocess.run(cmd, capture_output=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    return json.loads(proc.stdout.decode("utf-8"))


@pytest.fixture
def linux_stubs(tmp_path: Path) -> Path:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _stub(stubs, "lscpu", f"printf '{TRICKY_PRINTF}'")
    _stub(stubs, "free", "sleep 4; echo never")  # exceeds the 1 s probe timeout
    _stub(stubs, "ip", f"printf 'eth0 UP 192.168.1.20/24 fe80::1/64\\nlink aa:bb:cc:dd:ee:ff\\nmacs {MAC_RUN}\\n'")
    _stub(stubs, "df", 'printf \'{"serial_number" : "C02ABC123", "platform_UUID": "1234-5678", "n": 7}\\n\'')
    _stub(stubs, "nvidia-smi", "echo 'name, memory.total [MiB]'; echo 'NVIDIA GB10, [N/A]'; echo warn >&2; exit 3")
    return stubs


@needs_bash
def test_linux_both_writers_agree_and_escape_correctly(linux_stubs: Path) -> None:
    by_python = _run("inventory_linux.sh", linux_stubs, "python")
    by_bash = _run("inventory_linux.sh", linux_stubs, "bash")
    for doc in (by_python, by_bash):
        doc.pop("collected_at")
    assert by_python == by_bash

    probes = by_bash["probes"]
    assert by_bash["platform"] == "linux"
    assert probes["lscpu"]["stdout"] == TRICKY_TEXT
    assert probes["lscpu"]["exit_code"] == 0
    assert probes["lscpu"]["cmd"] == ["lscpu"]
    assert probes["nvidia_smi"]["exit_code"] == 3
    assert probes["nvidia_smi"]["stderr"] == "warn\n"
    assert "[N/A]" in probes["nvidia_smi"]["stdout"]


@needs_bash
def test_linux_redaction_timeout_and_missing_tools(linux_stubs: Path) -> None:
    probes = _run("inventory_linux.sh", linux_stubs, "bash")["probes"]

    df_out = probes["df_home"]["stdout"]
    assert "C02ABC123" not in df_out and "1234-5678" not in df_out
    assert json.loads(df_out) == {"serial_number": "[redacted]", "platform_UUID": "[redacted]", "n": 7}

    ip_out = probes["ip_br_addr"]["stdout"]
    assert "aa:bb:cc:dd:ee:ff" not in ip_out
    for mac in MAC_RUN.split():
        assert mac not in ip_out
    assert "fe80::1/64" in ip_out and "192.168.1.20/24" in ip_out

    assert probes["free_b"]["error"].startswith("timed out after 1s")
    assert probes["free_b"]["exit_code"] != 0

    for name, probe in probes.items():
        assert set(probe) <= {"cmd", "exit_code", "stdout", "stderr", "error"}, name
        if probe["exit_code"] is None:
            assert probe["error"].startswith(("not found:", "timed out")), name


@needs_bash
def test_nul_and_invalid_utf8_are_dropped_identically_by_both_writers(linux_stubs: Path) -> None:
    """NUL and every kind of invalid UTF-8 are dropped before either writer sees the text.

    Otherwise the python writer would print U+FFFD while the bash writer passed raw bytes through,
    and a strict UTF-8 JSON reader rejects the bash document.
    """
    invalid_printf = (
        r"nul[\000] cont[\200] ff[\377] overlong[\300\257] surrogate[\355\240\200] above[\364\220\200\200]"
        r" five[\370\210\200\200\200] emoji[\360\237\230\200] ok[caf\303\251] trunc[\342\234"
    )
    _stub(linux_stubs, "lscpu", f"printf '{invalid_printf}'")

    by_python = _run("inventory_linux.sh", linux_stubs, "python")
    by_bash = _run("inventory_linux.sh", linux_stubs, "bash")  # _run decodes strictly as UTF-8
    for doc in (by_python, by_bash):
        doc.pop("collected_at")
    assert by_python == by_bash
    assert by_bash["probes"]["lscpu"]["stdout"] == (
        "nul[] cont[] ff[] overlong[] surrogate[] above[] five[] emoji[\U0001f600] ok[café] trunc["
    )


@needs_bash
def test_bash_writer_survives_bsd_sed_final_newline(linux_stubs: Path) -> None:
    """BSD/macOS sed ends its output with a newline even when the input has none.

    A wrapper with that behaviour goes first on PATH; probe names and error strings, which are
    written without a trailing newline, must come out unchanged.
    """
    real_sed = subprocess.run([BASH, "-c", "command -v sed"], capture_output=True, text=True).stdout.strip()
    wrapper = (
        "#!/usr/bin/env bash\n"
        f'out=$("{real_sed}" "$@"; rc=$?; printf x; exit $rc) || exit $?\n'
        "out=${out%x}\n"
        'if [ -n "$out" ] && [ "${out: -1}" != $\'\\n\' ]; then out="$out"$\'\\n\'; fi\n'
        "printf '%s' \"$out\"\n"
    )
    sed = linux_stubs / "sed"
    sed.write_bytes(wrapper.encode())
    sed.chmod(0o755)

    probes = _run("inventory_linux.sh", linux_stubs, "bash")["probes"]
    assert set(probes) == {
        "uname_m",
        "kernel",
        "os_release",
        "lscpu",
        "free_b",
        "df_home",
        "ip_br_addr",
        "nvidia_smi",
        "nvcc_version",
    }
    assert probes["free_b"]["error"] == "timed out after 1s"
    assert probes["lscpu"]["stdout"] == TRICKY_TEXT


@needs_bash
def test_macos_profiler_identifiers_are_redacted(tmp_path: Path) -> None:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    profiler = {
        "SPHardwareDataType": [
            {
                "chip_type": "Apple M3 Max",
                "physical_memory": "64 GB",
                "serial_number": "C02XYZ987",
                "platform_UUID": "00000000-1111-2222-3333-444444444444",
                "provisioning_UDID": "00008112-000A",
            }
        ],
        "SPDisplaysDataType": [{"sppci_model": "Apple M3 Max", "_spdisplays_display-serial-number": 12345}],
    }
    body = json.dumps(profiler, indent=2).replace("'", "")
    _stub(stubs, "system_profiler", f"cat <<'JSON'\n{body}\nJSON")
    _stub(stubs, "sw_vers", "printf 'ProductName:\\tmacOS\\nProductVersion:\\t15.5\\n'")
    _stub(stubs, "ifconfig", "printf 'en0: flags=8863<UP>\\n\\tether 3c:22:fb:01:02:03\\n\\tinet 192.168.1.9\\n'")
    _stub(stubs, "networksetup", "printf 'Hardware Port: Wi-Fi\\nDevice: en0\\nEthernet Address: 3c:22:fb:01:02:03\\n'")
    # Deterministic stand-ins so the two writer runs see identical input (real df output drifts).
    _stub(stubs, "df", "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'")
    _stub(stubs, "sysctl", "echo 68719476736")
    _stub(stubs, "vm_stat", "echo 'Mach Virtual Memory Statistics: (page size of 16384 bytes)'")

    by_python = _run("inventory_macos.sh", stubs, "python")
    by_bash = _run("inventory_macos.sh", stubs, "bash")
    for doc in (by_python, by_bash):
        doc.pop("collected_at")
    assert by_python == by_bash

    probes = by_bash["probes"]
    parsed = json.loads(probes["system_profiler"]["stdout"])
    hw = parsed["SPHardwareDataType"][0]
    assert hw["chip_type"] == "Apple M3 Max"
    assert hw["serial_number"] == hw["platform_UUID"] == hw["provisioning_UDID"] == "[redacted]"
    assert parsed["SPDisplaysDataType"][0]["_spdisplays_display-serial-number"] == "[redacted]"
    assert probes["system_profiler"]["cmd"] == ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"]
    for name in ("ifconfig", "hardware_ports"):
        assert "3c:22:fb:01:02:03" not in probes[name]["stdout"]
    assert "192.168.1.9" in probes["ifconfig"]["stdout"]


# ------------------------------------------------------------------ Windows

WINDOWS_SCRIPT = SCRIPTS / "inventory_windows.ps1"
MAC_RE = re.compile(r"(?<![0-9A-Fa-f:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])")
UUID_RE = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")


def _run_windows(args: list[str], cwd: Path, stub_dir: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    assert POWERSHELL is not None
    env = dict(os.environ)
    if stub_dir is not None:
        env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
    cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WINDOWS_SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, cwd=cwd, env=env, timeout=300)


def _check_windows_doc(raw: bytes) -> dict:
    """Bytes a strict UTF-8 JSON reader accepts: no BOM, pure ASCII, the documented probe shape."""
    assert raw[:1] == b"{", raw[:8]
    assert max(raw) < 0x80, "non-ASCII bytes in output"
    doc = json.loads(raw.decode("utf-8"))
    assert doc["schema"] == 1 and doc["platform"] == "windows" and doc["script"] == "inventory_windows.ps1"
    assert doc["probes"], "no probes"
    for name, probe in doc["probes"].items():
        assert {"cmd", "exit_code", "stdout", "stderr"} <= set(probe), name
        assert set(probe) <= {"cmd", "exit_code", "stdout", "stderr", "error", "note"}, name
        assert probe["cmd"] and all(isinstance(arg, str) for arg in probe["cmd"]), name
        if probe["exit_code"] is None:
            assert probe["error"], name
    text = json.dumps(doc, ensure_ascii=False)
    assert not MAC_RE.findall(text) and not UUID_RE.findall(text)
    assert not re.search(r'(?i)"[^"]*(serial|uuid|ssid|macaddress|permanentaddress)[^"]*"\s*:', text)
    return doc


@needs_powershell
def test_windows_outfile_is_bom_free_ascii_json(tmp_path: Path) -> None:
    """-OutFile is the documented way to save the inventory from Windows PowerShell 5.1.

    There, "> file" writes UTF-16LE and piping into a native program prepends a UTF-8 BOM for each of
    [Console]::InputEncoding and $OutputEncoding that carries one (console code page 65001 does), so
    neither is a safe way to produce a JSON file.
    """
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    # Raw strings: python -c sees the escapes, so the stub writes UTF-8 bytes for "café ✓".
    emit = (
        r"import sys; sys.stdout.buffer.write(b'name, memory.total [MiB]\r\n"
        r"NVIDIA caf\xc3\xa9 \xe2\x9c\x93, 8188 MiB\r\n');"
        r" sys.stderr.write('warn\n'); sys.exit(3)"
    )
    (stubs / "nvidia-smi.cmd").write_bytes(f'@echo off\r\n"{sys.executable}" -c "{emit}"\r\n'.encode())
    work = tmp_path / "work"
    work.mkdir()

    proc = _run_windows(["-TimeoutSeconds", "30", "-OutFile", "inventory.json"], cwd=work, stub_dir=stubs)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout == b""
    raw = (work / "inventory.json").read_bytes()
    doc = _check_windows_doc(raw)

    gpu = doc["probes"]["nvidia_smi"]
    assert gpu["exit_code"] == 3
    assert gpu["stderr"].strip() == "warn"
    assert gpu["stdout"].startswith("name, memory.total [MiB]")
    # Decoding follows the console code page; whatever it decoded to, it left the file as \u escapes.
    assert any(ord(ch) > 0x7F for ch in gpu["stdout"])
    assert rb"\u" in raw


@needs_powershell
@pytest.mark.parametrize("code_page", [None, 65001])
def test_windows_stdout_has_no_bom(tmp_path: Path, code_page: int | None) -> None:
    """The script's own stdout is BOM-free ASCII JSON, also when the console code page is UTF-8.

    A BOM seen after "powershell ... | python" is added by Windows PowerShell 5.1's native pipe, not here.
    """
    if code_page is None:
        proc = _run_windows(["-TimeoutSeconds", "30"], cwd=tmp_path)
    else:
        # CREATE_NO_WINDOW gives cmd a console of its own, so chcp cannot leak into the test runner's.
        line = (
            f'cmd /d /s /c "chcp {code_page} >nul & chcp 1>&2 & "{POWERSHELL}" -NoProfile -ExecutionPolicy Bypass'
            f' -File "{WINDOWS_SCRIPT}" -TimeoutSeconds 30"'
        )
        proc = subprocess.run(
            line, capture_output=True, cwd=tmp_path, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW
        )
        assert str(code_page).encode() in proc.stderr, proc.stderr  # chcp took effect
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    _check_windows_doc(proc.stdout)


@needs_powershell
def test_windows_in_process_call_returns_json_to_the_pipeline(tmp_path: Path) -> None:
    """An in-process call (& .\\inventory_windows.ps1) hands the JSON text to its caller, not the console."""
    host = tmp_path / "host.ps1"
    host.write_text(
        "param([string]$Script, [string]$Out)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$text = & $Script -TimeoutSeconds 30\n"
        "if ((ConvertFrom-Json -InputObject $text).platform -ne 'windows') { throw 'no inventory in the pipeline' }\n"
        "[IO.File]::WriteAllText($Out, $text, (New-Object System.Text.UTF8Encoding $false))\n",
        encoding="ascii",
    )
    out = tmp_path / "captured.json"
    cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(host)]
    proc = subprocess.run(
        [*cmd, "-Script", str(WINDOWS_SCRIPT), "-Out", str(out)], capture_output=True, cwd=tmp_path, timeout=300
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout == b""  # nothing went around the caller to the console
    _check_windows_doc(out.read_bytes())


@needs_powershell
def test_windows_outfile_missing_directory_fails_before_probing(tmp_path: Path) -> None:
    target = tmp_path / "no-such-dir" / "inventory.json"
    proc = _run_windows(["-OutFile", str(target)], cwd=tmp_path)
    assert proc.returncode != 0
    assert proc.stdout == b""
    assert b"no-such-dir" in proc.stderr
    assert not target.exists()


PROMPT = "PS C:\\> "


def _help_examples() -> list[str]:
    """The text of each .EXAMPLE section of the script's comment-based help."""
    return WINDOWS_SCRIPT.read_text(encoding="utf-8").split("#>", 1)[0].split(".EXAMPLE")[1:]


@needs_powershell
def test_windows_help_is_found_by_get_help(tmp_path: Path) -> None:
    """Comment-based help glued to the #requires line is one comment block that Get-Help ignores.

    Get-Help splits an example's first line at its first ">" into prompt and code, so a line without
    the PS C:\\> prompt that holds a redirection loses its head to the prompt field.
    """
    query = tmp_path / "help.ps1"
    query.write_text(
        "param([string]$Script)\n"
        "$help = Get-Help $Script -Full\n"
        "'synopsis: ' + ([string]$help.Synopsis).Trim()\n"
        "foreach ($example in @($help.examples.example)) { 'example: ' + $example.code }\n",
        encoding="ascii",
    )
    cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(query), "-Script", str(WINDOWS_SCRIPT)]
    proc = subprocess.run(cmd, capture_output=True, cwd=tmp_path, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    out = proc.stdout.decode("ascii")
    assert "synopsis: Hypercompute raw hardware inventory for Windows" in out
    codes = [line.removeprefix("example: ") for line in out.splitlines() if line.startswith("example: ")]
    expected = [example.strip().splitlines()[0].removeprefix(PROMPT) for example in _help_examples()]
    assert len(expected) == 3 and codes == expected, out


def test_windows_documented_command_uses_outfile() -> None:
    examples = _help_examples()
    assert r"-File scripts\inventory_windows.ps1 -OutFile inventory.json" in examples[0]
    for example in examples:
        assert example.strip().startswith(PROMPT), example
        # In-process the session's execution policy applies; the example must say how to get past it.
        if "& .\\scripts\\inventory_windows.ps1" in example:
            assert "execution policy" in example and "-ExecutionPolicy Bypass" in example, example
        # A PowerShell "> file" writes UTF-16LE in 5.1; redirection is only documented through cmd.exe.
        for line in example.splitlines():
            if "> inventory.json" in line:
                assert line.strip().removeprefix(PROMPT).startswith('cmd /c "'), line
