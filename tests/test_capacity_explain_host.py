"""scripts/capacity/explain_host.py: the host node it builds and the coefficient rows it prints. No hardware."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from factories import contribution, engine_info, make_inventory
from hypercompute.capacity import costs
from hypercompute.capacity.measurements import MeasurementIdentity, MeasurementRecord
from hypercompute.capacity.types import ModelManifest, UnitSpec
from hypercompute.schemas import NodeState

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 22, 6, 30, tzinfo=UTC)
FP = "e" * 64
MIB = 1 << 20


def _script():
    name = "capacity_explain_host_script"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "capacity" / "explain_host.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _manifest() -> ModelManifest:
    units = [UnitSpec(i, "block", 150 * MIB, 150 * MIB) for i in range(4)] + [UnitSpec(4, "head", 500 * MIB,
                                                                                         500 * MIB)]
    return ModelManifest(model_id="m", revision="r1", shards=(), architecture="qwen2", file_type="Q4_K_M", n_layer=4,
                         n_embd=1024, n_head=8, n_head_kv=2, key_length=128, value_length=128, context_length=4096,
                         rope=(), tokenizer_model="gpt2", vocab_size=1000, chat_template_sha256=None,
                         units=tuple(units), token_embd_bytes=0, other_bytes=0, tied_embeddings=False,
                         streamable=True, fingerprint=FP)


def _rec(kind: str, device: str, value: float, unit: str, *, fingerprint: str | None = None,
         age: timedelta = timedelta(minutes=5)) -> MeasurementRecord:
    return MeasurementRecord(identity=MeasurementIdentity(node_id="local", device=device, backend_build="b11770",
                                                          model_fingerprint=fingerprint, kind=kind),
                             value=value, unit=unit, measured_at=NOW - age, samples=3, note="fixture",
                             evidence="results/capacity/x/bench.json")


def test_rows_mark_measured_coefficients_and_keep_priors_for_the_rest():
    s = _script()
    records = [_rec("compute.decode_ms_per_gib", "CPU", 20.0, "ms/GiB"),  # model-agnostic
               _rec("compute.decode_ms_per_gib", "CPU", 17.0, "ms/GiB", fingerprint=FP),  # this model wins
               _rec("compute.decode_ms_per_gib", "CUDA0", 2.9, "ms/GiB", age=timedelta(days=8)),  # stale
               _rec("storage.read_bytes_per_s", "C:", 6e9, "bytes/s")]
    rows = {r.label: r for r in s.coefficient_rows(_manifest(), records, gpu="CUDA0", stream=None, volume="C:",
                                                   now=NOW)}
    assert list(rows) == ["CPU decode", "CPU prefill", "CUDA0 decode", "CUDA0 prefill", "storage read",
                          "stream efficiency"]
    cpu = rows["CPU decode"]
    assert cpu.current.provenance == "measured" and cpu.current.value == 17.0
    assert cpu.ratio == 17.0 / costs.device_prior("cpu", "local", "CPU", "ram").decode_ms_per_gib.value
    assert rows["CUDA0 decode"].current.provenance == "prior" and rows["CUDA0 decode"].ratio is None
    assert rows["CPU prefill"].current.provenance == "prior"
    assert rows["storage read"].current.value == 6e9 and rows["storage read"].ratio == 6e9 / (3 << 30)
    assert rows["stream efficiency"].current.provenance == "prior"
    assert s.row_line(cpu).startswith("CPU decode: MEASURED 17 ms/GiB (prior 13.4218, measured/prior 1.267)")
    assert s.row_line(rows["CUDA0 decode"]).startswith("CUDA0 decode: PRIOR 2.83468 ms/GiB")


def test_rows_without_a_gpu_list_only_the_cpu_and_storage():
    rows = _script().coefficient_rows(_manifest(), [], gpu=None, stream=None, volume=None, now=NOW)
    assert [r.label for r in rows] == ["CPU decode", "CPU prefill", "storage read", "stream efficiency"]
    assert all(r.current.provenance == "prior" for r in rows)


def test_host_node_matches_the_registry_rules():
    s = _script()
    ok = s.host_node(make_inventory(), contribution(["CUDA0"]))
    assert ok.id == "local" and ok.is_local and ok.state == NodeState.eligible and ok.state_reason is None
    cpu_only = s.host_node(make_inventory(engine=engine_info(devices=[], accelerator="cpu")), contribution([]))
    assert cpu_only.state == NodeState.eligible and "CPU only" in (cpu_only.state_reason or "")
    broken = s.host_node(make_inventory(engine=engine_info(smoke="failed")), contribution([]))
    assert broken.state == NodeState.ineligible
    assert s.HostOnly(ok).eligible_remote(None) == [] and s.NoLeases().active() == []
