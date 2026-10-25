"""Device choice, memory cap, reserve and managed disk (PACKAGING-SPEC 2.4 steps 6, 7 and 10).

Fixtures are written by hand from the schema fields of step 5. The G14 numbers are NUMBERS.md condition L7.
"""

from __future__ import annotations

import pytest
from oneclick_core import Stop
from oneclick_plan import ask_managed, engine_fix, nvidia_driver_major, pick_device, plan_memory
from oneclick_testkit import inventory, make_ctx


def plan(tmp_path, name: str, nvidia: bool = False, answers=(), env=None):
    ctx, out = make_ctx(tmp_path, nvidia=nvidia, answers=answers, env=env)
    inv = inventory(name)
    device = pick_device(ctx, inv)
    return device, plan_memory(ctx, inv, device), out


def test_mac_m4_16gib(tmp_path) -> None:
    device, memory, out = plan(tmp_path, "inventory_mac_m4_16gib.json")
    assert (device, memory.max_gib, memory.reserve_gib) == ("MTL0", 10, 4)
    assert any("Memory the host may use [10 GiB]" in line for line in out)


def test_g14_with_nvidia_smi(tmp_path) -> None:
    device, memory, _ = plan(tmp_path, "inventory_g14.json", nvidia=True)
    assert (device, memory.max_gib, memory.reserve_gib) == ("CUDA0", 9, 6)
    assert memory.pool_text.startswith("vram:CUDA0 (discrete_vram), 11.9 GiB total")


def test_linux_cpu_32gib(tmp_path) -> None:
    device, memory, out = plan(tmp_path, "inventory_linux_cpu_32gib.json")
    assert (device, memory.max_gib, memory.reserve_gib) == ("CPU", 12, 6)
    assert "No supported GPU found. Sharing the CPU is slow." in out


def test_nvidia_without_cuda_engine_warns_then_asks(tmp_path) -> None:
    inv = inventory("inventory_g14.json")
    inv["engine"]["devices"] = []
    inv["engine"]["variant"] = "windows-x64-cpu"
    ctx, out = make_ctx(tmp_path, nvidia=True, answers=["n"])
    with pytest.raises(Stop, match="Not sharing"):
        pick_device(ctx, inv)
    assert any("installed engine (windows-x64-cpu) cannot see it" in line for line in out)
    assert any("engine install --force" in line for line in out)
    yes, _ = make_ctx(tmp_path, nvidia=True, answers=["y"])
    assert pick_device(yes, inv) == "CPU"


def cpu_engine_g14(system: str = "windows", machine: str = "x86_64", driver: str | None = None) -> dict:
    inv = inventory("inventory_g14.json")
    inv["engine"]["devices"] = []
    inv["engine"]["variant"] = f"{system}-cpu"
    inv["os"] = {**inv["os"], "system": system, "machine": machine}
    if driver is not None:
        next(d for d in inv["devices"] if d["id"] == "CUDA0")["driver_version"] = driver
    return inv


@pytest.mark.parametrize(("system", "machine", "driver", "need"), [
    ("windows", "x86_64", "560.94", 580), ("windows", "x86_64", "579.99", 580), ("linux", "x86_64", "565.57.01", 570),
])
def test_old_nvidia_driver_asks_for_a_driver_update_not_force(tmp_path, system, machine, driver, need) -> None:
    """choose_variant picks the CPU build again below the CUDA minimum, so --force would only loop."""
    inv = cpu_engine_g14(system, machine, driver)
    ctx, out = make_ctx(tmp_path, nvidia=True, answers=["y"])
    assert pick_device(ctx, inv) == "CPU"
    major = int(driver.split(".")[0])
    assert (f"Your NVIDIA driver ({major}) is too old for the GPU engine. Update it to {need} or newer, then close "
            "this window and run: uv run hypercompute engine install") in out
    assert not any("--force" in line for line in out)


@pytest.mark.parametrize(("system", "machine", "driver"), [
    ("windows", "x86_64", "581.57"), ("linux", "x86_64", "570.86"), ("linux", "arm64", "550.1"),
    ("windows", "x86_64", None), ("windows", "x86_64", "unknown"),
])
def test_new_or_unknown_driver_suggests_force(system, machine, driver) -> None:
    assert engine_fix(cpu_engine_g14(system, machine, driver)).endswith("engine install --force")


def test_driver_major_reads_only_nvidia_devices() -> None:
    inv = {"devices": [{"id": "Intel", "vendor": "Intel", "driver_version": "31.0.101"},
                       {"id": "CUDA0", "vendor": "NVIDIA", "backend": "cuda", "driver_version": "581.57"}]}
    assert nvidia_driver_major(inv) == 581
    assert nvidia_driver_major({"devices": []}) is None


def test_cpu_only_8gib_is_refused(tmp_path) -> None:
    with pytest.raises(Stop, match="does not have enough memory to share"):
        plan(tmp_path, "inventory_cpu_8gib.json")


def test_hc_device_override(tmp_path) -> None:
    ctx, _ = make_ctx(tmp_path, env={"HC_DEVICE": "CPU"}, nvidia=True)
    assert pick_device(ctx, inventory("inventory_g14.json")) == "CPU"
    bad, _ = make_ctx(tmp_path, env={"HC_DEVICE": "Vulkan0"})
    with pytest.raises(Stop, match="not a device the engine can use"):
        pick_device(bad, inventory("inventory_g14.json"))


def test_cpu_on_a_mac_uses_the_unified_pool(tmp_path) -> None:
    _, memory, _ = plan(tmp_path, "inventory_mac_m4_16gib.json", env={"HC_DEVICE": "CPU"})
    assert memory.max_gib == 10


def test_memory_answer_above_default_warns(tmp_path) -> None:
    _, memory, out = plan(tmp_path, "inventory_g14.json", nvidia=True, answers=["11 GiB"])
    assert memory.max_gib == 11
    assert any("above the safe default of 9 GiB" in line for line in out)


def test_memory_answer_out_of_range_asks_again(tmp_path) -> None:
    _, memory, out = plan(tmp_path, "inventory_g14.json", nvidia=True, answers=["50", "zero", "7"])
    assert memory.max_gib == 7
    assert sum("whole number of GiB from 1 to 11" in line for line in out) == 2


def test_bad_hc_max_memory_stops(tmp_path) -> None:
    with pytest.raises(Stop, match="HC_MAX_MEMORY"):
        plan(tmp_path, "inventory_g14.json", nvidia=True, env={"HC_MAX_MEMORY": "64GiB"})


def test_hc_reserve_override(tmp_path) -> None:
    _, memory, _ = plan(tmp_path, "inventory_g14.json", nvidia=True, env={"HC_RESERVE": "8GiB"})
    assert memory.reserve_gib == 8


def test_managed_limit_from_free_disk(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, answers=["y"])
    assert ask_managed(ctx, inventory("inventory_g14.json"), "CUDA0") == 24  # min(80, floor(54 - 30))
    assert any("(up to 24 GiB)? [y/N]" in line for line in out)
    no, _ = make_ctx(tmp_path)
    assert ask_managed(no, inventory("inventory_g14.json"), "CUDA0") is None  # the default is no
    big, _ = make_ctx(tmp_path, answers=["y"])
    assert ask_managed(big, inventory("inventory_mac_m4_16gib.json"), "MTL0") == 80


def test_managed_question_skipped_on_small_disk(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, answers=["y"])
    assert ask_managed(ctx, inventory("inventory_linux_cpu_32gib.json"), "CPU") is None  # floor(35 - 30) = 5 < 10
    assert not any("huggingface.co" in line for line in out)


def test_hc_max_disk_override(tmp_path) -> None:
    ctx, _ = make_ctx(tmp_path, env={"HC_MANAGED": "1", "HC_MAX_DISK": "30GiB"})
    assert ask_managed(ctx, inventory("inventory_g14.json"), "CUDA0") == 30
    bad, _ = make_ctx(tmp_path, env={"HC_MANAGED": "1", "HC_MAX_DISK": "40"})
    with pytest.raises(Stop, match="HC_MAX_DISK"):
        ask_managed(bad, inventory("inventory_g14.json"), "CUDA0")  # above floor(54 - 20)
