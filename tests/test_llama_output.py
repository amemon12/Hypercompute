"""Parsers against real llama.cpp b11770 output captured on the G14 (sanitized paths)."""

from hypercompute.adapters.llama_output import (
    detect_failure,
    parse_list_devices,
    parse_placement,
    parse_rpc_banner,
    parse_version,
    rpc_device_endpoint,
)
from hypercompute.units import MiB


def test_version_line():
    v = parse_version("version: 0.7.0-dev (build 11770, commit 39da130fc)\nbuilt with Clang 20.1.8 for Windows x86_64")
    assert v is not None and v.build_number == 11770 and v.commit_short == "39da130fc"


def test_list_devices_with_rpc_worker(fixture_text):
    devices = parse_list_devices(fixture_text("list_devices_rpc_b11770.txt"))
    names = [d.name for d in devices]
    assert names == ["CUDA0", "RPC0"]
    rpc = devices[1]
    assert rpc.description == "127.0.0.1:47052"
    assert rpc.total_bytes == 31864 * MiB and rpc.free_bytes == 4572 * MiB


def test_list_devices_absent_section_is_empty():
    assert parse_list_devices("ggml_cuda_init: found 1 CUDA devices\n") == []


def test_rpc_banner(fixture_text):
    banner = parse_rpc_banner(fixture_text("rpc_server_b11770.log"))
    assert banner.proto == "8.0.0"
    assert banner.endpoint == "127.0.0.1:47052"
    assert banner.transport == "TCP"
    assert [d.name for d in banner.devices] == ["CPU"]
    assert not banner.bind_failed


def test_rpc_bind_failure_detected():
    text = ("Starting RPC server v8.0.0\n  endpoint       : 127.0.0.1:50199\n  transport      : TCP\n"
            "Failed to create server socket\n")
    banner = parse_rpc_banner(text)
    assert banner.bind_failed
    assert "could not bind" in (detect_failure(text) or "")


def test_placement_from_real_split_log(fixture_text):
    p = parse_placement(fixture_text("llama_server_rpc_split_b11770.log"))
    assert p.offloaded == "25/25"
    by = {d.name: d for d in p.devices}
    assert by["CUDA0"].layers_est == 13 and by["RPC0[127.0.0.1:47052]"].layers_est == 11
    assert by["RPC0[127.0.0.1:47052]"].model_bytes > 200 * MiB  # 11 layers + the tied output head
    assert by["CPU_Mapped"].model_bytes > 0  # token embeddings stay on the host
    assert any("RPC0[127.0.0.1:47052] model buffer size" in line for line in p.evidence_lines)


def test_rpc_device_endpoint():
    assert rpc_device_endpoint("RPC0[10.0.0.5:47052]") == "10.0.0.5:47052"
    assert rpc_device_endpoint("CUDA0") is None


def test_failure_patterns():
    assert "refused or dropped" in detect_failure("...ggml-rpc.cpp:650: Failed to connect to 10.0.0.9:47052\n")
    assert detect_failure("all good") is None
