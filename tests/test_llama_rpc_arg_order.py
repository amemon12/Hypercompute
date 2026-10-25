"""llama-server argument order: --rpc before any -ot.

llama.cpp parses arguments left to right and -ot resolves each buffer type name against the devices registered so far,
so an override naming an RPC buffer type (RPC0[host:port]) placed before --rpc fails with "unknown buffer type".
"""

from __future__ import annotations

import pytest
from hypercompute import engine
from test_moe_placement import args_for, moe_plan
from test_windows_loading import ram_plan


def test_rpc_comes_before_the_moe_override_tensor_flag():
    """Regression: build_args emitted the moe_offload -ot before --rpc. Harmless while the override names only CPU,
    fatal as soon as one names a peer's RPC buffer."""
    args = args_for(moe_plan(peer=True))
    assert args.count("--rpc") == 1 and "-ot" in args
    assert args.index("--rpc") < min(i for i, a in enumerate(args) if a in ("-ot", "--override-tensor"))


@pytest.mark.parametrize("system", ["windows", "macos", "linux"])
def test_rpc_directly_follows_the_model_on_every_tier(monkeypatch, system):
    monkeypatch.setattr(engine, "host_os", lambda: system)
    for plan in (moe_plan(peer=True), ram_plan(remote=True)):
        args = args_for(plan)
        assert args[4] == "--rpc" and args[5] == ",".join(plan.rpc_endpoints)  # after -m PATH --alias NAME
    assert "--rpc" not in args_for(moe_plan(peer=False))
