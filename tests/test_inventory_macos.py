"""macOS free memory for a peer's budget: what can be allocated without swapping, not psutil's free + inactive."""

from __future__ import annotations

import pytest
from hypercompute.agent import inventory

# vm_stat from a 16 GiB M4 MacBook Air (mac-a, 2026-10-23), right after its tensor test files were deleted.
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              487519.
Pages active:                            211877.
Pages inactive:                          228847.
Pages speculative:                         1503.
Pages throttled:                              0.
Pages wired down:                         69276.
Pages purgeable:                          14330.
"Translation faults":                   8762901.
Pages copy-on-write:                     363047.
File-backed pages:                       164434.
Anonymous pages:                         277793.
Pages stored in compressor:               30724.
Pages occupied by compressor:              9782.
Swapouts:                                     0.
"""
TOTAL = 16 * 1024**3


def test_available_is_total_minus_app_wired_and_compressed_memory():
    used = (277793 - 14330 + 69276 + 9782) * 16384  # app memory (anonymous minus purgeable) + wired + compressor
    assert inventory.macos_available_bytes(VM_STAT, TOTAL) == TOTAL - used


def test_cached_files_in_active_pages_count_as_available():
    # 9 GiB of file cache sitting in active pages: psutil (free + inactive) would leave it out
    heavy = VM_STAT.replace("Pages active:                            211877.",
                            "Pages active:                            801877.")
    assert inventory.macos_available_bytes(heavy, TOTAL) == inventory.macos_available_bytes(VM_STAT, TOTAL)


@pytest.mark.parametrize("drop", ["page size of 16384 bytes", "Anonymous pages", "Pages wired down",
                                  "Pages occupied by compressor", "Pages purgeable"])
def test_missing_fields_give_none_so_the_caller_falls_back(drop):
    text = "\n".join(line for line in VM_STAT.splitlines() if drop not in line)
    assert inventory.macos_available_bytes(text, TOTAL) is None


def test_never_negative():
    tiny = 1024**3
    assert inventory.macos_available_bytes(VM_STAT, tiny) == 0


def test_darwin_uses_vm_stat_and_other_systems_keep_psutil(monkeypatch):
    class VM:
        total = TOTAL
        available = 1234

    monkeypatch.setattr(inventory.psutil, "virtual_memory", lambda: VM)
    monkeypatch.setattr(inventory, "_run", lambda args, timeout=20.0: VM_STAT if args == ["vm_stat"] else None)
    monkeypatch.setattr(inventory.platform, "system", lambda: "Darwin")
    assert inventory.available_ram_bytes() == inventory.macos_available_bytes(VM_STAT, TOTAL)
    monkeypatch.setattr(inventory, "_run", lambda args, timeout=20.0: None)
    assert inventory.available_ram_bytes() == 1234  # vm_stat failed: psutil
    monkeypatch.setattr(inventory.platform, "system", lambda: "Windows")
    assert inventory.available_ram_bytes() == 1234
