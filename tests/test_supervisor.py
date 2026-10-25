"""Owned-process supervision with real child processes (portable: uses this Python interpreter)."""

import json
import sys
import time
from pathlib import Path

import psutil
import pytest
from hypercompute.agent.supervisor import Supervisor

SLEEPER = ["-c", "import time; print('up', flush=True); time.sleep(120)"]


def test_start_stop_and_log(tmp_path):
    sup = Supervisor(tmp_path / "managed.json")
    proc = sup.start("worker", __import__("pathlib").Path(sys.executable), SLEEPER, tmp_path, tmp_path / "w.log")
    for _ in range(50):
        if "up" in proc.read_log():
            break
        time.sleep(0.1)
    assert proc.alive() and "up" in proc.read_log()
    outcome = sup.stop("worker", grace_s=5)
    assert outcome is not None and outcome.clean and not psutil.pid_exists(proc.pid) or not proc.alive()
    assert (tmp_path / "managed.json").read_text().strip() == "[]"


def test_reconcile_stops_only_our_orphans(tmp_path):
    from pathlib import Path

    state = tmp_path / "managed.json"
    first = Supervisor(state)
    proc = first.start("deployment:x", Path(sys.executable), SLEEPER, tmp_path, tmp_path / "d.log")
    # a second supervisor (as after a gateway crash) finds and stops the recorded child
    notes = Supervisor(state).reconcile_orphans()
    assert any("stopped orphaned" in n or "killed orphaned" in n for n in notes)
    proc.popen.wait(timeout=10)
    assert not proc.alive()


def test_reconcile_ignores_reused_pid(tmp_path):
    import json

    state = tmp_path / "managed.json"
    me = psutil.Process()
    # a record whose pid exists but whose create_time does not match must be left alone
    state.write_text(json.dumps([{"key": "deployment:y", "pid": me.pid, "create_time": me.create_time() - 999,
                                  "exe": me.exe()}]))
    notes = Supervisor(state).reconcile_orphans()
    assert any("left untouched" in n for n in notes) and psutil.pid_exists(me.pid)


def test_double_start_is_refused(tmp_path):
    from pathlib import Path

    import pytest

    sup = Supervisor(tmp_path / "managed.json")
    sup.start("w", Path(sys.executable), SLEEPER, tmp_path, tmp_path / "a.log")
    try:
        with pytest.raises(RuntimeError):
            sup.start("w", Path(sys.executable), SLEEPER, tmp_path, tmp_path / "b.log")
    finally:
        sup.stop_all(grace_s=5)


def test_instance_lock_blocks_second_live_instance(tmp_path):
    import json
    import subprocess

    import pytest
    from hypercompute.agent import instance

    lock = tmp_path / "host.lock"
    instance.acquire(lock, "host")
    instance.acquire(lock, "host")  # re-entrant for the same process
    other = subprocess.Popen([sys.executable, *SLEEPER])
    try:
        p = psutil.Process(other.pid)
        lock.write_text(json.dumps({"pid": other.pid, "create_time": p.create_time(), "role": "host"}))
        with pytest.raises(instance.InstanceLocked):
            instance.acquire(lock, "host")
    finally:
        other.kill()
        other.wait(timeout=10)
    instance.acquire(lock, "host")  # a dead holder no longer blocks
    instance.release(lock)
    assert not lock.exists()


# --- regression: owned children die with their owner (finding 10) ---------------------------------------------------

OWNER = """
import sys, time
from pathlib import Path
from hypercompute.agent.supervisor import Supervisor
work = Path(sys.argv[1])
proc = Supervisor(work / "managed.json").start(
    "worker", Path(sys.executable), ["-c", "import time; time.sleep(120)"], work, work / "child.log")
print(proc.pid, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS has no kill-on-close job or PDEATHSIG; join.py handles it")
def test_child_dies_when_owner_is_hard_killed(tmp_path):
    import os
    import subprocess

    script = tmp_path / "owner.py"
    script.write_text(OWNER, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "backend")}
    owner = subprocess.Popen([sys.executable, str(script), str(tmp_path)], stdout=subprocess.PIPE, text=True, env=env)
    child = None
    try:
        child = psutil.Process(int(owner.stdout.readline()))
        assert child.is_running()
        owner.kill()  # TerminateProcess / SIGKILL: no cleanup code runs in the owner
        owner.wait(timeout=10)
        _, alive = psutil.wait_procs([child], timeout=15)
        assert not alive, "the supervised child outlived its hard-killed owner"
    finally:
        owner.kill()
        if child is not None and child.is_running():
            child.kill()


# --- regression: a stop that did not finish keeps the process and its record (finding 24) --------------------------

def test_still_running_stop_keeps_process_and_record(tmp_path, monkeypatch):
    from hypercompute.agent.supervisor import StopOutcome

    state = tmp_path / "managed.json"
    sup = Supervisor(state)
    proc = sup.start("w", Path(sys.executable), SLEEPER, tmp_path, tmp_path / "a.log")
    with monkeypatch.context() as m:
        m.setattr(sup, "_terminate", lambda p, grace: StopOutcome(p.key, p.pid, None, "still_running"))
        outcome = sup.stop("w")
    assert outcome is not None and not outcome.clean
    assert sup.get("w") is proc  # stop_all / a later stop can still reach it
    assert [r["pid"] for r in json.loads(state.read_text())] == [proc.pid]
    assert [o.how for o in sup.stop_all(grace_s=5)] in (["terminated"], ["graceful"])
    assert json.loads(state.read_text()) == [] and sup.get("w") is None


# --- regression: reconcile keeps records it could not inspect or stop (finding 13) ---------------------------------

def test_reconcile_keeps_records_it_could_not_inspect_or_stop(tmp_path, monkeypatch):
    from hypercompute.agent import supervisor as supmod

    me = psutil.Process()
    created, exe = me.create_time(), me.exe()

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            if pid == 3:
                raise psutil.NoSuchProcess(pid)
            self.pid = pid

        def create_time(self) -> float:
            if self.pid == 1:
                raise psutil.AccessDenied(self.pid)
            return created

        def exe(self) -> str:
            return exe

        def terminate(self) -> None:
            raise psutil.AccessDenied(self.pid)  # e.g. an elevated worker seen from a normal shell

    records = [{"key": f"k{pid}", "pid": pid, "create_time": created, "exe": exe} for pid in (1, 2, 3)]
    state = tmp_path / "managed.json"
    state.write_text(json.dumps(records))
    monkeypatch.setattr(supmod.psutil, "Process", FakeProcess)
    sup = Supervisor(state)
    notes = sup.reconcile_orphans()
    assert [r["pid"] for r in sup.recorded()] == [1, 2]  # only the confirmed-gone pid 3 is dropped
    assert any("record kept" in n and "pid 1" in n for n in notes)
    assert any("could not stop pid 2" in n for n in notes)
    assert any("pid 3 no longer running" in n for n in notes)
