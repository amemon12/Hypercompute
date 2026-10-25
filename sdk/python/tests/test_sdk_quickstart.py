"""examples/quickstart.py runs end to end against the fake gateway, and stops only what it started."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
from fake_gateway import MODEL, FakeClock, FakeGateway

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hc_sdk_quickstart", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clocked(gw: FakeGateway):  # type: ignore[no-untyped-def]
    clock = FakeClock()
    return gw.pool(sleep=clock.sleep, clock=clock.now)


def test_quickstart_plans_deploys_chats_streams_and_stops() -> None:
    gw, lines, out = FakeGateway(ready_after=1), [], []
    pool = clocked(gw)
    code = load().run(pool, MODEL, strategy="auto", say=lines.append, write=out.append)
    assert code == 0
    assert lines[0].startswith("plan ") and "feasible True" in lines[0]
    assert "chat: Hello there" in lines
    assert "".join(out) == "Hello\n"
    assert any(line.startswith("decode 25.0 tok/s over 3 tokens") for line in lines)
    assert lines[-1].endswith(": stopped")
    (dep,) = gw.deps.values()
    assert dep["state"] == "stopped"


def test_quickstart_main_picks_the_downloaded_model() -> None:
    gw, lines = FakeGateway(ready_after=0), []
    assert load().main([], pool=clocked(gw), say=lines.append) == 0
    assert gw.plans and next(iter(gw.plans.values()))["model_id"] == MODEL


def test_quickstart_leaves_someone_elses_deployment_alone() -> None:
    gw, lines = FakeGateway(), []
    other = gw.add_dep("ready")
    assert load().main([MODEL], pool=clocked(gw), say=lines.append) == 1
    assert any("deployment_active" in line for line in lines)
    assert gw.deps[other["id"]]["state"] == "ready"


def test_quickstart_stops_after_a_failed_chat() -> None:
    gw, lines = FakeGateway(ready_after=0, chat_mode="interrupted"), []
    assert load().main([MODEL], pool=clocked(gw), say=lines.append) == 1
    assert any(line.startswith("error: ") and "interrupted" in line for line in lines)
    assert all(d["state"] == "stopped" for d in gw.deps.values())


def test_quickstart_infeasible_plan_and_no_model() -> None:
    gw, lines = FakeGateway(), []
    gw.overrides[("POST", "/api/plans")] = lambda r: httpx.Response(200, json={
        "id": "p1", "feasible": False, "reasons": ["needs 40 GiB, the pool has 20 GiB"], "device_list": []})
    assert load().main([MODEL], pool=clocked(gw), say=lines.append) == 1
    assert "  needs 40 GiB, the pool has 20 GiB" in lines
    gw = FakeGateway()
    gw.overrides[("GET", "/api/models")] = lambda r: httpx.Response(200, json=[])
    lines = []
    assert load().main([], pool=clocked(gw), say=lines.append) == 1
    assert "no downloaded model" in lines[0]


def test_quickstart_refuses_a_lan_url() -> None:
    lines: list[str] = []
    assert load().main(["--url", "http://192.168.1.20:8001", MODEL], say=lines.append) == 1
    assert lines[0].startswith("error: owner routes only answer this machine")
