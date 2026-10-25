"""EXO on the friend agent: resume rolls back, a foreign EXO on the API port, the EXO warning before consent."""

from __future__ import annotations

import re
import socket
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from factories import contribution, make_inventory
from hypercompute.agent import exo_sidecar, join
from hypercompute.agent.join import Agent, AgentError
from hypercompute.agent.supervisor import StopOutcome
from hypercompute.config import Paths
from hypercompute.installers import exo as inst
from hypercompute.schemas import ExtraEngineInfo, WorkerState

BANNER = "Starting RPC server v3.0.0\n  endpoint       : 10.0.0.5:47052\n  transport      : TCP\n"
IN_USE = "Port 52415 is already in use - is exo already running? Use --api-port to choose another."
FOREIGN = "foreign-node-from-the-exo-app"


class Proc:
    def __init__(self, log: str = BANNER) -> None:
        self.live, self.log, self.log_path = True, log, Path("proc.log")

    def alive(self) -> bool:
        return self.live

    def exit_code(self) -> int | None:
        return None if self.live else 1

    def cpu_seconds(self) -> float:
        return 0.1

    def read_log(self, max_bytes: int = 0) -> str:
        return self.log


class Sup:
    """Supervisor's key semantics: start refuses a live key, stop pops the record, get reads it."""

    def __init__(self, log: str = BANNER, exits: bool = False) -> None:
        self.procs: dict[str, Proc] = {}
        self.started: list[str] = []
        self.log, self.exits = log, exits

    def reconcile_orphans(self) -> list[str]:
        for proc in self.procs.values():  # the real one terminates every recorded process still ours, this run's too
            proc.live = False
        return []

    def start(self, key, exe, args, cwd, log_path, env=None) -> Proc:
        if key in self.procs and self.procs[key].alive():
            raise RuntimeError(f"managed process {key} is already running")
        self.started.append(key)
        self.procs[key] = Proc(self.log)
        self.procs[key].live = not self.exits
        return self.procs[key]

    def get(self, key: str) -> Proc | None:
        return self.procs.get(key)

    def stop(self, key: str, grace_s: float = 8.0) -> StopOutcome | None:
        proc = self.procs.pop(key, None)
        if proc is None:
            return None
        proc.live = False
        return StopOutcome(key, 1, 0, "graceful")


class Installed:
    bin_dir = Path(".")

    def binary(self, name: str) -> Path:
        return Path(name)


def exo_agent(monkeypatch, exo_fails: list[bool], on_start: Callable[[], None] = lambda: None) -> Agent:
    """A Mac agent with --exo whose EXO start fails while exo_fails[0] is True (port 52415 taken). EXO runs under
    the agent's supervisor like the real sidecar's; on_start runs while EXO is starting."""

    class Car:
        endpoint, node_id, namespace = "10.0.0.5:52415", "ours", "ns"

        def __init__(self, sup: Sup, *a) -> None:
            self.sup = sup

        def start(self) -> ExtraEngineInfo:
            if exo_fails[0]:
                raise exo_sidecar.ExoSidecarError(f"EXO exited during startup (code 1): {IN_USE}",
                                                  fix=exo_sidecar.explain_failure(IN_USE))
            self.sup.start(exo_sidecar.EXO_KEY, Path("python"), [], Path("."), Path("exo.log"))
            on_start()
            return self.info()

        def stop(self) -> StopOutcome | None:
            return self.sup.stop(exo_sidecar.EXO_KEY)

        def info(self) -> ExtraEngineInfo:
            proc = self.sup.get(exo_sidecar.EXO_KEY)
            status = "installed" if proc is None else "running" if proc.alive() else "failed"
            return ExtraEngineInfo(engine="exo", installed=True, status=status,
                                   endpoint=self.endpoint if status == "running" else None)

    monkeypatch.setattr(join.engine, "find_installed", lambda: Installed())
    monkeypatch.setattr(join, "ExoSidecar", Car)
    agent = Agent(host="192.168.1.20", port=8000, code="code", display_name="mac", inv=make_inventory("macos"),
                  contribution=contribution(["MTL0"]), advertise_ip="10.0.0.5", worker_port=47052,
                  exo_enabled=True)
    agent.sup = Sup()  # type: ignore[assignment]
    return agent


# --- findings 1/9: a failed resume rolls back, so Pause keeps meaning "nothing is shared" ------------------------

def test_resume_that_fails_on_exo_stops_the_worker_again_and_stays_paused(monkeypatch, capsys):
    exo_fails = [False]
    agent = exo_agent(monkeypatch, exo_fails)
    agent.start_worker()
    agent.start_exo()
    assert agent.pause() == "paused"
    exo_fails[0] = True
    with pytest.raises(AgentError, match="already in use"):
        agent.resume()
    assert agent.paused and agent.worker_status()[0] == WorkerState.paused  # heartbeats say paused, not running
    assert "still paused" in capsys.readouterr().out
    assert agent.pause() == "already paused" and agent.sup.get(join.WORKER_KEY) is None
    exo_fails[0] = False  # the friend quit the EXO app: the next Resume works and Pause stops it again
    assert agent.resume() == "resumed" and agent.worker_status()[0] == WorkerState.running
    assert agent.pause() == "paused" and agent.worker_status()[0] == WorkerState.paused


def test_resume_whose_worker_exits_early_stays_paused_instead_of_reading_as_exited(monkeypatch):
    agent = exo_agent(monkeypatch, [False])
    agent.paused = True
    agent.sup = Sup(log="ggml_backend_load: no devices\n", exits=True)  # type: ignore[assignment]
    with pytest.raises(AgentError, match="exited early"):
        agent.resume()
    assert agent.paused and agent.worker_status()[0] == WorkerState.paused  # an exited worker would stop the agent


def test_pause_does_not_report_exo_as_lost_while_the_worker_keeps_sharing(monkeypatch, capsys):
    agent = exo_agent(monkeypatch, [False])
    agent.start_worker()
    agent.start_exo()
    agent.pause()
    capsys.readouterr()
    agent.exo_heartbeat()
    assert "keeps sharing" not in capsys.readouterr().out  # paused on purpose: nothing is sharing


def test_resume_starts_exo_before_the_worker_and_keeps_that_exo_running(monkeypatch):
    seen: list[WorkerState] = []
    exo_fails = [False]
    agent = exo_agent(monkeypatch, exo_fails, on_start=lambda: seen.append(agent.worker_status()[0]))
    agent.start_worker()
    agent.start_exo()
    agent.pause()
    seen.clear()
    assert agent.resume() == "resumed"
    assert seen == [WorkerState.paused]  # what heartbeats during EXO's start report: no running worker yet
    assert agent.exo is not None and agent.exo.info().status == "running"  # not reaped by the orphan cleanup
    agent.pause()
    agent.sup.started.clear()
    exo_fails[0] = True
    with pytest.raises(AgentError, match="already in use"):
        agent.resume()
    assert agent.sup.started == []  # a failed EXO start never brings the worker up


# --- findings 8/23: start() never takes another EXO's answer on 52415 for its own ---------------------------------

def _sidecar(monkeypatch, tmp_path, sup: Sup, port: int, get) -> exo_sidecar.ExoSidecar:
    monkeypatch.setattr(inst, "status", lambda paths: ExtraEngineInfo(engine="exo", installed=True,
                                                                       status="installed"))
    monkeypatch.setattr(inst, "API_PORT", port)
    monkeypatch.setattr(exo_sidecar.time, "sleep", lambda s: None)
    return exo_sidecar.ExoSidecar(sup, Paths(home=tmp_path), "10.0.0.5", "ns", get=get)  # type: ignore[arg-type]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return int(s.getsockname()[1])


def test_start_refuses_when_another_exo_already_serves_the_api_port(monkeypatch, tmp_path):
    with socket.create_server(("0.0.0.0", 0)) as foreign:  # the EXO app, listening on EXO's API port
        sup = Sup(log=IN_USE)
        car = _sidecar(monkeypatch, tmp_path, sup, foreign.getsockname()[1],
                       get=lambda url, timeout: httpx.Response(200, json=FOREIGN))
        with pytest.raises(exo_sidecar.ExoSidecarError) as err:
            car.start(timeout_s=5)
    assert "quit it first" in (err.value.fix or "") and FOREIGN in str(err.value)
    assert sup.started == [] and car.node_id is None  # ours never started, nothing foreign was adopted


def test_start_does_not_adopt_an_answer_that_arrives_after_our_exo_died(monkeypatch, tmp_path):
    sup = Sup(log=IN_USE)

    def foreign_answers_as_ours_exits(url: str, timeout: float) -> httpx.Response:
        sup.procs["exo"].live = False
        return httpx.Response(200, json=FOREIGN)

    car = _sidecar(monkeypatch, tmp_path, sup, _free_port(), get=foreign_answers_as_ours_exits)
    with pytest.raises(exo_sidecar.ExoSidecarError) as err:
        car.start(timeout_s=5)
    assert "quit it first" in (err.value.fix or "") and car.node_id is None


def test_api_port_check_matches_a_live_listener_only():
    with socket.create_server(("0.0.0.0", 0)) as listener:
        port = listener.getsockname()[1]
        assert exo_sidecar.api_port_free(port) is False
    assert exo_sidecar.api_port_free(port) is True


# --- finding 24: the friend reads the EXO warning before agreeing, and gets firewall commands for it ------------

class TtyStdin:
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("assume_yes", [False, True])
def test_join_with_exo_warns_before_consent_and_before_exo_starts(monkeypatch, capsys, assume_yes):
    agent = exo_agent(monkeypatch, [False])
    seen: dict[str, str] = {}

    def answer(prompt: str) -> str:
        seen["prompt"] = capsys.readouterr().out
        return "y"

    def start_exo() -> None:
        seen["start_exo"] = seen.get("prompt", "") + capsys.readouterr().out

    async def serve() -> None:
        return None

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(join.sys, "stdin", TtyStdin())
    monkeypatch.setattr(agent, "start_worker", lambda: None)
    monkeypatch.setattr(agent, "start_exo", start_exo)
    monkeypatch.setattr(agent, "join", lambda: None)
    monkeypatch.setattr(agent, "serve", serve)
    monkeypatch.setattr(agent, "leave", lambda: None)
    assert join.run_join(agent, fallback_cpu=False, assume_yes=assume_yes) == 0
    for before in ([seen["prompt"]] if not assume_yes else []) + [seen["start_exo"]]:
        assert "EXO's API on 10.0.0.5:52415 has no authentication" in before
        assert "Anyone on this network" in before and "model code" in before


def test_scoping_guidance_with_exo_also_scopes_the_api_port_on_a_mac(monkeypatch):
    agent = exo_agent(monkeypatch, [False])
    text = join.scoping_guidance(agent)
    assert ("printf 'pass in quick proto tcp from 192.168.1.20 to any port 47052\\nblock drop in quick proto tcp to "
            "any port 47052\\ntable <hc_exo> persist { 192.168.1.20 }\\npass in quick on lo0 proto tcp to any port "
            "52415\\npass in quick proto tcp from <hc_exo> to any port 52415\\nblock drop in quick proto tcp to any "
            "port 52415\\n' | sudo pfctl -a com.apple/hypercompute -f - && sudo pfctl -E") in text
    assert "sudo pfctl -a com.apple/hypercompute -t hc_exo -T add" in text
    agent.exo_enabled = False
    assert "52415" not in join.scoping_guidance(agent)


def _pf_verdict(rules: str, iface: str, src: str, port: int) -> str:
    """pf runs quick rules in order and the first match decides; enough of pf for the printed anchor."""
    table = re.search(r"table <hc_exo> persist \{ (.*?) \}", rules)
    members = set(table[1].split()) if table else set()
    for rule in rules.split("\\n"):
        m = re.fullmatch(r"(pass|block drop) in quick(?: on (\S+))? proto tcp(?: from (\S+))? to any port (\d+)", rule)
        if m and int(m[4]) == port and m[2] in (None, iface) and (
                m[3] in (None, src) or (m[3] == "<hc_exo>" and src in members)):
            return m[1]
    return "pass"  # nothing matched: pf's default


def test_mac_pf_rules_let_loopback_reach_exo_and_keep_the_lan_out(monkeypatch):
    agent = exo_agent(monkeypatch, [False])
    rules = re.search(r"printf '(.*?)' \| sudo pfctl", join.scoping_guidance(agent))
    assert rules is not None
    assert _pf_verdict(rules[1], "lo0", "127.0.0.1", 52415) == "pass"  # the readiness probe and EXO's dashboard
    assert _pf_verdict(rules[1], "en0", "192.168.1.20", 52415) == "pass"  # the host
    assert _pf_verdict(rules[1], "en0", "192.168.1.99", 52415) == "block drop"  # anyone else on the LAN
    assert _pf_verdict(rules[1], "en0", "192.168.1.20", 47052) == "pass"
    assert _pf_verdict(rules[1], "en0", "192.168.1.99", 47052) == "block drop"
