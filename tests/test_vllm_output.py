"""vLLM log reading (adapters/vllm_output.py): startup phases, fatal markers and startup metrics."""

from pathlib import Path

from hypercompute.adapters import vllm_output as vout
from test_vllm import record

FIXTURES = Path(__file__).parent / "fixtures"


def test_log_markers_phases_and_metrics():
    rec = record()
    good = (FIXTURES / "vllm_startup_synthetic.log").read_text(encoding="utf-8")
    bad = (FIXTURES / "vllm_startup_free_memory_error.log").read_text(encoding="utf-8")
    assert vout.startup_failure(good) is None and vout.load_phase(good, rec) == "HTTP server up"
    assert vout.load_phase("", rec).startswith("starting vLLM in WSL2")
    message, fix = vout.startup_failure(bad)
    assert "less GPU memory was free" in message and "0.71/11.94 GiB" in message and "llama.cpp" in fix
    values, evidence = vout.parse_startup(good)
    assert values["weights"] == 5.21 and values["kv"] == 3.87 and values["kv_tokens"] == 72464
    assert values["concurrency"] == 17.69 and values["init"] == 31.2 and values["graphs_gib"] == 0.41
    assert any("MarlinLinearKernel" in line for line in evidence)
    assert "RuntimeError: Engine core initialization failed" in vout.exit_detail(bad, 1)


def test_the_normal_free_memory_info_line_is_not_a_startup_failure():
    """vLLM 0.32 logs this INFO line on every healthy start (captured from the first gateway start on the RTX 5070 Ti
    Laptop); matching it as fatal killed a real deployment."""
    info = (FIXTURES / "vllm_startup_info_free_memory.log").read_text(encoding="utf-8")
    assert "Free memory on device" in info and vout.startup_failure(info) is None
