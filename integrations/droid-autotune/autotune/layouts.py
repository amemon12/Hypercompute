"""Gateway plans folded back into layout stages, and the deterministic diff the owner approves from.

A plan names peer devices the way llama.cpp registers them (RPC0, RPC1, ... in --rpc endpoint order, see
services/deployments.py assign_rpc_names), while a layout names them as the peer's worker does (MTL0, CPU). The worker
name is the peer's contribution.devices entry at the device's index within its own endpoint. Every byte figure here is
the planner's estimate; nothing is a measurement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .models import DeviceRow, DiffRow, LayoutDiff, Prediction, Stage, Verdict
from .pool import LOCAL_ID, PoolSnapshot

_RPC = re.compile(r"^RPC(\d+)$")
_CANDIDATE_DECODE = re.compile(r"predicted decode ~([0-9.]+) ms/token")


def _rpc_offsets(plan: Mapping[str, Any], snap: PoolSnapshot) -> dict[str, int]:
    """First RPC number of each endpoint: endpoints in --rpc order, each taking one number per verified device."""
    by_endpoint = {a.get("endpoint"): a.get("node_id") for a in plan.get("assignments") or [] if a.get("endpoint")}
    offsets, k = {}, 0
    for endpoint in plan.get("rpc_endpoints") or []:
        offsets[endpoint] = k
        node = snap.node(str(by_endpoint.get(endpoint, "")))
        # deployments.rpc_device_counts: one number per device the host's probe verified (len(rpc_devices))
        count = node.rpc_count if node is not None and node.rpc_count else (len(node.devices) if node else 1)
        k += max(1, count)
    return offsets


def _worker_device(a: Mapping[str, Any], snap: PoolSnapshot, offsets: Mapping[str, int]) -> str:
    device = str(a.get("device") or "?")
    endpoint = a.get("endpoint")
    if not endpoint:
        return device
    node = snap.node(str(a.get("node_id")))
    match = _RPC.match(device)
    if node is None or match is None or endpoint not in offsets:
        return device
    index = int(match.group(1)) - offsets[endpoint]
    return node.devices[index].name if 0 <= index < len(node.devices) else device


def rows(plan: Mapping[str, Any], snap: PoolSnapshot) -> list[DeviceRow]:
    """The plan's device assignments in execution order (by first layer), named as a layout names them."""
    offsets = _rpc_offsets(plan, snap)
    out = []
    for a in plan.get("assignments") or []:
        layers = [x for x in a.get("layers") or [] if isinstance(x, int)]
        node_id = LOCAL_ID if a.get("node_id") == LOCAL_ID else str(a.get("node_id"))
        out.append(DeviceRow(
            node_id=node_id, node_name=snap.name_of(node_id), device=_worker_device(a, snap, offsets),
            layers=len(layers), first_layer=min(layers) if layers else None, last_layer=max(layers) if layers else None,
            total_bytes_est=int(a.get("total_bytes_est") or 0), budget_bytes=int(a.get("budget_bytes") or 0),
            fits=bool(a.get("fits", True)), file_backed=bool(a.get("file_backed")),
            holds_output=bool(a.get("holds_output"))))
    return sorted(out, key=lambda r: (r.holds_output, r.first_layer if r.first_layer is not None else 1 << 30))


def stages(device_rows: Sequence[DeviceRow]) -> list[Stage] | None:
    """The explicit layout that reproduces these rows, or None when the plan uses a tier a layout cannot name
    (layers on the host CPU: the gpu_ram tier)."""
    if any(r.node_id == LOCAL_ID and r.device == "CPU" for r in device_rows):
        return None
    out = []
    for r in device_rows:
        if r.layers == 0:
            continue
        residency = "file_backed" if r.file_backed else None
        out.append(Stage(node_id=r.node_id, device=r.device, layers=r.layers, residency=residency))
    return out or None


def hops(device_rows: Sequence[DeviceRow]) -> int:
    """Machine boundaries a token crosses on its way through the stages. R2/R6 measured each extra peer stage as a
    per-token cost, so this is the number the decode side of a layout pays for."""
    chain = [r.node_id for r in device_rows if r.layers > 0 or r.holds_output]
    return sum(1 for a, b in zip(chain, chain[1:], strict=False) if a != b)


def prediction(plan: Mapping[str, Any]) -> Prediction:
    """The planner's numbers for this plan, labelled by source. Explicit layouts skip the capacity optimizer, so a
    dense layout often has none; a moe_offload layout carries the MoE estimate."""
    pred = plan.get("prediction")
    if isinstance(pred, Mapping) and pred.get("decode_ms") is not None:
        return Prediction(decode_ms=float(pred["decode_ms"]), decode_source="prediction",
                          ssd_bytes_per_token=int(pred.get("stream_bytes_per_token") or 0) or None,
                          load_ms=float(pred["load_ms"]) if pred.get("load_ms") is not None else None)
    moe = plan.get("moe")
    if isinstance(moe, Mapping) and moe.get("decode_ms_est") is not None:
        return Prediction(decode_ms=float(moe["decode_ms_est"]), decode_source="moe",
                          ssd_bytes_per_token=int(moe.get("ssd_bytes_per_token_est") or 0))
    for cand in plan.get("candidates") or []:
        match = _CANDIDATE_DECODE.search(str(cand.get("reason") or "")) if cand.get("feasible") else None
        if match:
            return Prediction(decode_ms=float(match.group(1)), decode_source="planner note (priors)")
    return Prediction()


def _range(r: DeviceRow | None) -> str:
    if r is None or r.first_layer is None:
        return "-"
    return f"{r.first_layer}" if r.first_layer == r.last_layer else f"{r.first_layer}-{r.last_layer}"


def layout_diff(before_plan: Mapping[str, Any] | None, after_plan: Mapping[str, Any], snap: PoolSnapshot) -> LayoutDiff:
    """Per node and device: layers and estimated bytes before and after, plus hops, tier and the planner's decode
    estimate on both sides when it gave one."""
    before = rows(before_plan, snap) if before_plan else []
    after = rows(after_plan, snap)
    keys: list[tuple[str, str]] = []
    for r in (*after, *before):
        if (r.node_id, r.device) not in keys:
            keys.append((r.node_id, r.device))
    b_map = {(r.node_id, r.device): r for r in before}
    a_map = {(r.node_id, r.device): r for r in after}
    diff_rows = []
    for key in keys:
        b, a = b_map.get(key), a_map.get(key)
        ref = a or b
        assert ref is not None
        diff_rows.append(DiffRow(
            node_id=key[0], node_name=ref.node_name, device=key[1], layers_before=b.layers if b else 0,
            layers_after=a.layers if a else 0, range_before=_range(b), range_after=_range(a),
            bytes_before=b.total_bytes_est if b else 0, bytes_after=a.total_bytes_est if a else 0,
            budget_bytes=ref.budget_bytes, fits_after=a.fits if a else True,
            file_backed_after=a.file_backed if a else False))
    p_before = prediction(before_plan) if before_plan else Prediction()
    p_after = prediction(after_plan)
    same_source = p_before.decode_source == p_after.decode_source
    return LayoutDiff(
        rows=diff_rows, hops_before=hops(before) if before else None, hops_after=hops(after),
        machines_before=len({r.node_id for r in before}), machines_after=len({r.node_id for r in after}),
        tier_before=before_plan.get("tier") if before_plan else None, tier_after=after_plan.get("tier"),
        decode_ms_before=p_before.decode_ms if same_source else None, decode_ms_after=p_after.decode_ms,
        decode_source=p_after.decode_source, ssd_bytes_before=p_before.ssd_bytes_per_token,
        ssd_bytes_after=p_after.ssd_bytes_per_token,
        unchanged=(all(r.layers_before == r.layers_after and r.range_before == r.range_after for r in diff_rows)
                   and [(r.node_id, r.device, r.file_backed) for r in before]
                   == [(r.node_id, r.device, r.file_backed) for r in after]
                   and (before_plan or {}).get("tier") == after_plan.get("tier")))


def placement_drift(approved: Verdict, fresh: Verdict) -> list[str]:
    """How the planner's plan for the same stages differs from the one the owner approved: the tier, or what a
    device holds and whether it is file-backed. Empty when it is the same placement."""
    out = []
    if approved.tier != fresh.tier:
        out.append(f"tier {approved.tier} -> {fresh.tier}")
    seen = {(r.node_id, r.device): (r.layers, r.file_backed) for r in approved.rows}
    now = {(r.node_id, r.device): (r.layers, r.file_backed) for r in fresh.rows}
    for key in sorted(set(seen) | set(now)):
        if seen.get(key) != now.get(key):
            before, after = seen.get(key, (0, False)), now.get(key, (0, False))
            out.append(f"{key[1]} on {key[0][:8]}: {before[0]} layers{' file-backed' if before[1] else ''} -> "
                       f"{after[0]} layers{' file-backed' if after[1] else ''}")
    return out
