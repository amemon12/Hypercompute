"""Windows RAM-tier loading avoids whole-file mmap prefetch without changing placement or disk tiers."""

from pathlib import Path

import pytest
from hypercompute import engine
from hypercompute.adapters.base import DeploymentSpec, EngineAdapterError
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.schemas import PlanRequest, PlanStrategy
from hypercompute.services.deployments import assign_rpc_names
from hypercompute.services.planner import make_plan
from hypercompute.units import GiB
from test_split_order import EP5, cpu, gpu, mac, shape14

LOADING_FLAGS = {"--load-mode", "--no-host", "--no-repack"}


def ram_plan(remote=False):
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.all_selected if remote
                                else PlanStrategy.local_only), shape14(), [gpu(6)],
                     [mac(4)] if remote else [], 32 * GiB, cpu(5 if remote else 16))
    assert plan.feasible and plan.tier == "gpu_ram" and plan.cpu_layers > 0
    return assign_rpc_names(plan, {EP5: 1}) if remote else plan


def args_for(plan):
    adapter = object.__new__(LlamaRpcAdapter)
    spec = DeploymentSpec(deployment_id="d", plan=plan, model_path=Path("weights.gguf"), alias="m")
    return adapter.build_args(spec, 9000)


@pytest.mark.parametrize("remote", [False, True])
def test_windows_ram_loading_preserves_placement_and_engine_settings(monkeypatch, remote):
    plan = ram_plan(remote).model_copy(update={"engine_flags": {"batch": 1024, "ubatch": 256}})
    before = plan.model_dump()
    monkeypatch.setattr(engine, "host_os", lambda: "linux")
    baseline = args_for(plan)
    monkeypatch.setattr(engine, "host_os", lambda: "windows")
    args = args_for(plan)

    assert args[args.index("--load-mode") + 1] == "none"
    assert args.count("--no-host") == args.count("--no-repack") == 1
    assert args[args.index("-ngl") + 1] == str(plan.n_gpu_units)
    assert args[args.index("--tensor-split") + 1] == ",".join(str(n) for n in plan.tensor_split)
    assert args[args.index("-c") + 1] == str(plan.ctx_size)
    assert args[args.index("--fit") + 1] == "off"
    if remote:
        assert args[args.index("--rpc") + 1] == EP5
        assert args[args.index("--device") + 1] == "RPC0,CUDA0"
    without_loading = args.copy()
    mode_index = without_loading.index("--load-mode")
    del without_loading[mode_index:mode_index + 2]
    without_loading.remove("--no-host")
    without_loading.remove("--no-repack")
    assert without_loading == baseline
    assert plan.model_dump() == before


@pytest.mark.parametrize("system", ["linux", "macos", "other"])
def test_other_platforms_keep_default_ram_loading(monkeypatch, system):
    monkeypatch.setattr(engine, "host_os", lambda: system)
    assert not LOADING_FLAGS.intersection(args_for(ram_plan()))


@pytest.mark.parametrize("changes", [
    {"tier": "gpu", "cpu_layers": 0, "n_gpu_units": None},
    {"feasible": False},
    {"capacity_mode": True},
    {"cpu_layers": 0},
])
def test_windows_policy_applies_only_to_feasible_resident_cpu_layers(monkeypatch, changes):
    monkeypatch.setattr(engine, "host_os", lambda: "windows")
    assert not LOADING_FLAGS.intersection(args_for(ram_plan().model_copy(update=changes)))


def test_windows_disk_tier_keeps_file_backed_loading_and_existing_budget(monkeypatch):
    monkeypatch.setattr(engine, "host_os", lambda: "windows")
    plan = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.local_only), shape14(),
                     [gpu(6)], [], 4 * GiB, cpu(3))
    before = plan.model_dump()
    assert plan.feasible and plan.tier == "gpu_disk" and plan.capacity_mode
    host = plan.assignments[0]
    assert host.weight_bytes > host.budget_bytes
    assert host.total_bytes_est == host.kv_bytes + host.compute_bytes_est + host.overhead_bytes_est
    assert not LOADING_FLAGS.intersection(args_for(plan))
    assert plan.model_dump() == before


def test_loading_policy_does_not_expand_user_engine_flag_allowlist(monkeypatch):
    monkeypatch.setattr(engine, "host_os", lambda: "windows")
    plan = ram_plan().model_copy(update={"engine_flags": {"load_mode": "none"}})
    with pytest.raises(EngineAdapterError, match="unknown llama.cpp engine flag"):
        args_for(plan)


@pytest.mark.parametrize("system", ["windows", "linux"])
def test_ram_tiers_turn_off_the_unbudgeted_host_prompt_cache(monkeypatch, system):
    monkeypatch.setattr(engine, "host_os", lambda: system)
    args = args_for(ram_plan())
    assert args[args.index("--cache-ram") + 1] == "0" and args.count("--cache-ram") == 1
    gpu_only = ram_plan().model_copy(update={"tier": "gpu", "cpu_layers": 0, "n_gpu_units": None})
    assert "--cache-ram" not in args_for(gpu_only)
