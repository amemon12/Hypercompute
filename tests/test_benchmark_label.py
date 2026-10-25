"""Benchmark placement labels: no "~None layers" for engines that place layers themselves, CPU tier shown."""

from hypercompute.schemas import ObservedDevice, ObservedPlacement, utcnow
from hypercompute.services.benchmarks import placement_label
from test_inference_slots import deployment


def observed(*devices: tuple[str, int | None]) -> ObservedPlacement:
    found = [ObservedDevice(name=n, model_bytes=1, kv_bytes=0, compute_bytes=0, layers_est=k) for n, k in devices]
    return ObservedPlacement(devices=found, offloaded=None, evidence_lines=[], parsed_at=utcnow())


def test_engines_that_place_layers_themselves_show_no_layer_count():
    dep = deployment()
    dep = dep.model_copy(update={"plan": dep.plan.model_copy(update={"engine": "vllm"}),
                                 "placement": observed(("CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU)", None))})
    assert placement_label(dep) == "vLLM server: CUDA0 (NVIDIA GeForce RTX 5070 Ti Laptop GPU)"


def test_llama_cpp_labels_name_the_cpu_tier_and_layer_counts():
    dep = deployment()
    dep = dep.model_copy(update={"plan": dep.plan.model_copy(update={"cpu_layers": 4}),
                                 "placement": observed(("CPU_Mapped", 4), ("CUDA0", 44))})
    assert placement_label(dep) == "CPU (4 layers) + CUDA0 (~44 layers)"
