"""Cross-engine glue: --device never names the CPU, capacity-mode warmup budget, forced overflow tiers."""

from types import SimpleNamespace

from hypercompute import config
from hypercompute.schemas import DeviceAssignment, Plan, PlanRequest, PlanStrategy, utcnow
from hypercompute.services.deployments import assign_rpc_names, remap_rpc_names, warmup_timeout
from hypercompute.services.planner import make_plan
from hypercompute.units import GiB
from test_split_order import EP5, cpu, gpu, mac, shape14


def assignment(node_id: str, device: str, endpoint: str | None, layers: range, head: bool = False
               ) -> DeviceAssignment:
    return DeviceAssignment(node_id=node_id, device=device, endpoint=endpoint, pool_id="p", layers=list(layers),
                            holds_output=head, weight_bytes=1, kv_bytes=1, compute_bytes_est=1,
                            overhead_bytes_est=1, total_bytes_est=4, budget_bytes=8, fits=True)


def cpu_tier_plan(**extra: object) -> Plan:
    a = [assignment("local", "CPU", None, range(4)), assignment("mac", f"{EP5}#0", EP5, range(4, 10)),
         assignment("local", "CUDA0", None, range(10, 48), head=True)]
    return Plan(id="p", model_id="m", strategy=PlanStrategy.auto, ctx_size=4096, feasible=True,
                fit_label="estimated_fit", heuristic=True, assignments=a, device_list=["RPC0", "CUDA0"],
                tensor_split=[6, 39], rpc_endpoints=[EP5], cpu_resident_bytes=0, reasons=[], warnings=[],
                candidates=[], created_at=utcnow(), tier="gpu_ram", cpu_layers=4, n_gpu_units=45, **extra)


def test_rpc_naming_keeps_the_cpu_tier_out_of_the_device_list():
    named = assign_rpc_names(cpu_tier_plan(), {EP5: 1})
    assert named.device_list == ["RPC0", "CUDA0"]  # llama.cpp rejects `--device CPU`
    remapped = remap_rpc_names(named, {EP5: ["RPC0"]}, {EP5: 1})
    assert remapped.device_list == ["RPC0", "CUDA0"]
    assert [a.device for a in remapped.assignments] == ["CPU", "RPC0", "CUDA0"]  # leases still see the CPU tier


def test_capacity_mode_warmup_gets_the_adapters_idle_budget():
    slow = SimpleNamespace(idle_timeout_s=900.0)
    assert warmup_timeout(cpu_tier_plan(capacity_mode=True), slow) == 1800.0
    assert warmup_timeout(cpu_tier_plan(), slow) == config.WARMUP_TIMEOUT_S
    assert warmup_timeout(cpu_tier_plan(capacity_mode=True), SimpleNamespace()) == config.WARMUP_TIMEOUT_S * 2


def test_overflow_can_be_forced_to_either_tier_for_the_tuner():
    """The 14B Q6_K does not fit a 10.6 GiB GPU: auto picks one overflow tier, the tuner measures both."""
    args = (shape14(), [gpu(10.6)], [mac(9)], 16 * GiB, cpu(12))
    ram = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.auto, overflow="host_ram"), *args)
    peers = make_plan(PlanRequest(model_id="m", strategy=PlanStrategy.auto, overflow="peers"), *args)
    assert ram.feasible and ram.tier == "gpu_ram" and ram.cpu_layers > 0 and not ram.rpc_endpoints
    assert peers.feasible and peers.cpu_layers == 0 and peers.rpc_endpoints == [EP5]
    assert peers.assignments[-1].device == "CUDA0" and peers.assignments[-1].holds_output  # head stays local
