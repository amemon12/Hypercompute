"""hcpeer on the Mac agent: the HC_PEER_MMAP opt-in, ggml-rpc-server's arguments and environment, the banner, the
cache cap that waits for a v4 banner, and the consent text."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from factories import contribution, engine_info, make_inventory
from hypercompute import cli, config
from hypercompute.agent import hcpeer, join
from hypercompute.agent.hcpeer import HcpeerBanner
from hypercompute.agent.join import Agent, AgentError
from hypercompute.agent.managed import ManagedConsent
from hypercompute.agent.supervisor import StopOutcome
from hypercompute.schemas import EngineDevice
from hypercompute.units import GiB, MiB

HEAD = "Starting RPC server v8.0.0\n  endpoint       : 127.0.0.1:47052\n  local cache    : /Users/air/.hypercompute\n"
TAIL = ("Devices:\n  MTL0: Apple M4 (12124 MiB, 12123 MiB free)\n  CPU: Apple M4 (16384 MiB, 9000 MiB free)\n"
        "  transport      : TCP\n")
ON_V4 = ("  hcpeer mmap    : on (v4, page 16384 B, CPU weights mapped from the cache, cache saves keep 20 GiB free, "
         "no cache size cap)\n")
ON_V3 = "  hcpeer mmap    : on (v3, page 16384 B, CPU weights mapped from the cache, cache saves keep 20 GiB free)\n"
OFF_V4 = "  hcpeer mmap    : off (v4, set HC_PEER_MMAP=1 to enable)\n"


def banner(hcpeer_line: str = "") -> str:
    return HEAD + hcpeer_line + TAIL


def mac_inventory():
    return make_inventory("macos", engine=engine_info(accelerator="metal").model_copy(update={"devices": [
        EngineDevice(name="MTL0", description="Apple M4", total_bytes=12 * GiB, free_bytes=11 * GiB)]}))


def mac_agent(*, mmap: bool = True, max_disk: int | None = None, cache: bool = False, threads: int | None = None,
              devices: tuple[str, ...] = ("MTL0", "CPU")) -> Agent:
    managed = ManagedConsent(("MTL0", "CPU"), max_disk, 10) if max_disk is not None else None
    c = contribution(list(devices)).model_copy(update={"cache_enabled": cache})
    return Agent(host="127.0.0.1", port=0, code="code", display_name="air", inv=mac_inventory(), contribution=c,
                 advertise_ip="127.0.0.1", worker_port=47052, managed_consent=managed, threads=threads,
                 hcpeer_mmap=mmap)


def close(agent: Agent) -> None:
    if agent.managed is not None:
        agent.managed.close()


# --------------------------------------------------------------------------- the module


@pytest.mark.parametrize(("line", "expected"), [
    (ON_V4, HcpeerBanner(on=True, version=4)),
    (ON_V3, HcpeerBanner(on=True, version=3)),
    (OFF_V4, HcpeerBanner(on=False, version=4)),
    ("  hcpeer mmap    : off (v4, HC_PEER_MMAP=1 needs -c)\n", HcpeerBanner(on=False, version=4)),
    ("", None),  # the stock library prints no hcpeer line
])
def test_parse_banner(line, expected):
    assert hcpeer.parse_banner(banner(line)) == expected


@pytest.mark.parametrize(("system", "environ", "expected"), [
    ("macos", {"HC_PEER_MMAP": "1"}, True),
    ("macos", {"HC_PEER_MMAP": "0"}, False),
    ("macos", {"HC_PEER_MMAP": "yes"}, False),  # hcpeer itself turns on only for "1"
    ("macos", {}, False),
    ("windows", {"HC_PEER_MMAP": "1"}, False),  # the library is macOS only
])
def test_requested_only_for_hc_peer_mmap_1_on_a_mac(system, environ, expected):
    assert hcpeer.requested(system, environ) is expected


@pytest.mark.parametrize(("max_disk", "expected"), [
    (80 * GiB, 80), (80 * GiB + 900 * MiB, 80), (512 * MiB, 0), (0, None)])
def test_cache_cap_is_max_disk_in_whole_gib_rounded_down(max_disk, expected):
    assert hcpeer.cache_cap_gib(max_disk) == expected


V4, V3 = HcpeerBanner(on=True, version=4), HcpeerBanner(on=True, version=3)
CACHE = Path("/Users/air/.hypercompute/rpc-cache")


@pytest.mark.parametrize(("cache", "on", "cap", "seen", "expected"), [
    (None, True, 80, V4, {}),  # no -c: hcpeer has nothing to map
    (CACHE, False, 80, V4, {"LLAMA_CACHE": str(CACHE)}),
    (CACHE, True, None, V4, {"LLAMA_CACHE": str(CACHE), "HC_PEER_MMAP": "1"}),
    (CACHE, True, 80, None, {"LLAMA_CACHE": str(CACHE), "HC_PEER_MMAP": "1"}),  # no banner yet
    (CACHE, True, 80, V3, {"LLAMA_CACHE": str(CACHE), "HC_PEER_MMAP": "1"}),  # v3 ignores the cap
    (CACHE, True, 80, V4, {"LLAMA_CACHE": str(CACHE), "HC_PEER_MMAP": "1", "HC_PEER_CACHE_MAX_GIB": "80"}),
    (CACHE, True, 0, HcpeerBanner(on=False, version=5),
     {"LLAMA_CACHE": str(CACHE), "HC_PEER_MMAP": "1", "HC_PEER_CACHE_MAX_GIB": "0"}),
])
def test_worker_env(cache, on, cap, seen, expected):
    assert hcpeer.worker_env(cache, on=on, cap_gib=cap, banner=seen) == expected


def test_status_line_says_what_the_worker_reported():
    capped = {"HC_PEER_CACHE_MAX_GIB": "80"}
    assert "has no hcpeer" in hcpeer.status_line(None, 80, {})
    assert "is off" in hcpeer.status_line(HcpeerBanner(on=False, version=4), 80, {})
    expected = "hcpeer v4 on: CPU weights are mapped from the cache, cache capped at 80 GiB."
    assert hcpeer.status_line(V4, 80, capped) == expected
    assert "ignores the cache cap" in hcpeer.status_line(V3, 80, {})
    assert "no cache size cap" in hcpeer.status_line(V4, None, {})


# --------------------------------------------------------------------------- the agent's worker command


def test_worker_command_for_a_managed_hcpeer_mac():
    agent = mac_agent(max_disk=80 * GiB, threads=4)
    args, env = agent.worker_command()
    assert args == ["-H", "127.0.0.1", "-p", "47052", "-d", "MTL0,CPU", "-t", "4", "-c"]
    assert env == {"LLAMA_CACHE": str(config.paths().rpc_cache), "HC_PEER_MMAP": "1"}  # no cap before a banner
    agent.hcpeer_banner = V4
    assert agent.worker_command()[1]["HC_PEER_CACHE_MAX_GIB"] == "80"
    close(agent)


def test_hcpeer_turns_the_cache_on_since_it_maps_the_cache_files():
    agent = mac_agent(cache=False)
    assert agent.contribution.cache_enabled
    args, env = agent.worker_command()
    assert args[-1] == "-c" and env["HC_PEER_MMAP"] == "1" and "HC_PEER_CACHE_MAX_GIB" not in env  # no --max-disk


def test_without_hcpeer_the_worker_command_is_unchanged():
    plain = mac_agent(mmap=False, devices=("MTL0",))
    plain.hcpeer_banner = V4  # a v4 library alone changes nothing
    assert plain.worker_command() == (["-H", "127.0.0.1", "-p", "47052", "-d", "MTL0"], {})
    cached = mac_agent(mmap=False, cache=True, devices=("MTL0",))
    cached.hcpeer_banner = V4
    assert cached.worker_command() == (["-H", "127.0.0.1", "-p", "47052", "-d", "MTL0", "-c"],
                                       {"LLAMA_CACHE": str(config.paths().rpc_cache)})


# --------------------------------------------------------------------------- starting the worker


class Proc:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_log(self) -> str:
        return self.text

    def alive(self) -> bool:
        return True

    def exit_code(self) -> int | None:
        return None


class Sup:
    def __init__(self, logs: list[str], stop_how: str = "graceful") -> None:
        self.logs, self.stop_how = logs, stop_how
        self.started: list[tuple[list[str], dict[str, str]]] = []
        self.stopped: list[str] = []

    def start(self, key, exe, args, cwd, log_path, env=None) -> Proc:
        self.started.append((list(args), dict(env or {})))
        return Proc(self.logs.pop(0) if self.logs else banner(ON_V4))

    def stop(self, key: str, grace_s: float = 8.0) -> StopOutcome:
        self.stopped.append(key)
        return StopOutcome(key, 1, None if self.stop_how == "still_running" else 0, self.stop_how)


def with_engine(monkeypatch, agent: Agent, sup: Sup) -> Sup:
    installed = SimpleNamespace(binary=lambda name: Path(name), bin_dir=Path("."))
    monkeypatch.setattr(join.engine, "find_installed", lambda: installed)
    agent.sup = sup  # type: ignore[assignment]
    return sup


def test_a_v4_banner_restarts_the_first_worker_once_with_the_cache_cap(monkeypatch):
    agent = mac_agent(max_disk=80 * GiB)
    sup = with_engine(monkeypatch, agent, Sup([banner(ON_V4), banner(ON_V4)]))
    agent.start_worker()
    assert [e.get("HC_PEER_CACHE_MAX_GIB") for _, e in sup.started] == [None, "80"]
    assert sup.stopped == [join.WORKER_KEY] and sup.started[0][0] == sup.started[1][0]
    assert agent.hcpeer_banner == V4 and agent.worker_devices_seen == ["MTL0", "CPU"]
    agent.start_worker()  # a later start (host restart, revival) knows the version: one start, capped
    assert len(sup.started) == 3 and sup.started[2][1]["HC_PEER_CACHE_MAX_GIB"] == "80"
    assert sup.stopped == [join.WORKER_KEY]
    close(agent)


@pytest.mark.parametrize("line", [ON_V3, ""])
def test_an_older_or_stock_library_starts_once_without_the_cap(monkeypatch, line):
    agent = mac_agent(max_disk=80 * GiB)
    sup = with_engine(monkeypatch, agent, Sup([banner(line)]))
    agent.start_worker()
    assert len(sup.started) == 1 and sup.stopped == []
    assert sup.started[0][1]["HC_PEER_MMAP"] == "1" and "HC_PEER_CACHE_MAX_GIB" not in sup.started[0][1]
    assert agent.hcpeer_banner == hcpeer.parse_banner(banner(line))
    close(agent)


def test_without_hcpeer_a_v4_library_is_never_restarted_or_given_hc_peer_vars(monkeypatch):
    agent = mac_agent(mmap=False, max_disk=80 * GiB, cache=True)
    sup = with_engine(monkeypatch, agent, Sup([banner(OFF_V4)]))
    agent.start_worker()
    assert len(sup.started) == 1 and not any(k.startswith("HC_PEER_") for k in sup.started[0][1])
    close(agent)


@pytest.mark.parametrize(("line", "expected"), [(ON_V4, [False, True]), (ON_V3, [False]), ("", [False])])
def test_the_disk_guard_learns_whether_the_worker_runs_with_the_hcpeer_cap(monkeypatch, line, expected):
    agent = mac_agent(max_disk=80 * GiB)
    with_engine(monkeypatch, agent, Sup([banner(line), banner(line)]))
    seen: list[bool] = []
    monkeypatch.setattr(agent.managed, "worker_started", lambda hcpeer_capped: seen.append(hcpeer_capped))
    agent.start_worker()
    assert seen == expected  # the first start of a v4 library only reads the version; the restart has the cap
    close(agent)


def test_a_worker_that_cannot_be_stopped_for_the_cap_is_an_error(monkeypatch):
    agent = mac_agent(max_disk=80 * GiB)
    with_engine(monkeypatch, agent, Sup([banner(ON_V4)], stop_how="still_running"))
    with pytest.raises(AgentError, match="could not be stopped to apply the hcpeer cache cap"):
        agent.start_worker()
    close(agent)


# --------------------------------------------------------------------------- consent and the join command


def test_consent_says_the_cpu_device_maps_cached_files_from_disk(capsys):
    agent = mac_agent(max_disk=80 * GiB)
    join.print_plan(agent, fallback_cpu=False)
    out = capsys.readouterr().out
    assert "CPU device maps cached tensor files from disk" in out
    assert str(config.paths().rpc_cache / "rpc") in out and "under 80 GiB (--max-disk)" in out
    close(agent)
    join.print_plan(mac_agent(mmap=False, devices=("MTL0",)), fallback_cpu=False)
    assert "hcpeer" not in capsys.readouterr().out


def test_with_hcpeer_the_consented_max_disk_is_the_whole_gib_cap_the_worker_enforces():
    """Regression: hcpeer caps the cache at max_disk // GiB while the agent's guard, its seeding and the host's plan
    (the report's max_disk_bytes) held the raw --max-disk, so the host budgeted bytes hcpeer would not save."""
    agent = mac_agent(max_disk=80 * GiB + 900 * MiB)
    assert agent.managed is not None and agent.managed_consent is not None
    assert agent.managed_consent.max_disk_bytes == 80 * GiB and agent.managed.runner.max_disk_bytes == 80 * GiB
    assert agent.managed.report(["MTL0", "CPU"], None, 0).max_disk_bytes == 80 * GiB  # what the host budgets
    close(agent)
    plain = mac_agent(mmap=False, max_disk=80 * GiB + 900 * MiB)
    assert plain.managed_consent is not None and plain.managed_consent.max_disk_bytes == 80 * GiB + 900 * MiB
    close(plain)


def test_hcpeer_refuses_a_max_disk_under_1_gib_whose_cap_of_0_would_save_nothing():
    with pytest.raises(AgentError, match="at least 1 GiB") as exc:
        mac_agent(max_disk=900 * MiB)
    assert "HC_PEER_MMAP" in (exc.value.fix or "")
    no_downloads = mac_agent(max_disk=0)  # --managed without --max-disk: no cap and nothing to round
    assert no_downloads.managed_consent is not None and no_downloads.managed_consent.max_disk_bytes == 0
    close(no_downloads)


def join_args() -> SimpleNamespace:
    return SimpleNamespace(advertise_ip=None, worker_port=47052, devices="MTL0", max_memory=None, reserve=None,
                           cache=False, managed=True, devices_allowed="MTL0,CPU", max_disk="80GiB", exo=False,
                           code="code", name=None, yes=True)


def joined_agents(monkeypatch) -> list[Agent]:
    monkeypatch.setattr(cli.inventory, "collect", mac_inventory)
    monkeypatch.setattr(cli.netutil, "pick_bindable_port", lambda ip, port, scan: (port, []))
    agents: list[Agent] = []
    monkeypatch.setattr(cli, "run_join", lambda agent, fallback_cpu, yes: agents.append(agent) or 0)
    return agents


def run_join_on_host() -> int:
    return cli._run_join_on_host(join_args(), "127.0.0.1", 0, {"engine": {"release": "b11770"}},
                                 SimpleNamespace())  # type: ignore[arg-type]


@pytest.mark.parametrize(("value", "expected"), [("1", True), (None, False)])
def test_join_takes_hc_peer_mmap_from_the_agent_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("HC_PEER_MMAP", raising=False)
    else:
        monkeypatch.setenv("HC_PEER_MMAP", value)
    agents = joined_agents(monkeypatch)
    assert run_join_on_host() == 0
    agent = agents[0]
    assert agent.hcpeer_mmap is expected and agent.contribution.cache_enabled  # --max-disk turns the cache on too
    assert agent.managed_consent is not None and agent.managed_consent.devices_allowed == ("MTL0", "CPU")
    assert ("HC_PEER_MMAP" in agent.worker_command()[1]) is expected
    close(agent)


@pytest.mark.parametrize("name", ["HC_PEER_CACHE_MAX_GIB", "HC_PEER_CACHE_MIN_FREE_GIB"])
def test_join_refuses_an_hcpeer_limit_the_worker_would_inherit_from_the_shell(monkeypatch, name):
    """Regression: ggml-rpc-server inherits the agent's environment, so a cap or floor exported in the contributor's
    shell replaced the one the agent's disk guard and the host's plan hold (--max-disk in whole GiB, the 20 GiB floor):
    without --max-disk or before a v4 banner the worker capped the cache where the host planned no cap, and a floor
    of 0 saved and mapped past the consented free space."""
    monkeypatch.setenv("HC_PEER_MMAP", "1")
    monkeypatch.setenv(name, "0")
    agents = joined_agents(monkeypatch)
    with pytest.raises(AgentError, match=name) as exc:
        run_join_on_host()
    assert f"unset {name}" in (exc.value.fix or "") and agents == []


def test_without_hcpeer_an_inherited_hcpeer_limit_changes_nothing(monkeypatch):
    """hcpeer applies its cap and floor only while HC_PEER_MMAP=1: the stock path joins as before."""
    monkeypatch.delenv("HC_PEER_MMAP", raising=False)
    monkeypatch.setenv("HC_PEER_CACHE_MAX_GIB", "5")
    agents = joined_agents(monkeypatch)
    assert run_join_on_host() == 0
    assert agents[0].hcpeer_mmap is False
    close(agents[0])
