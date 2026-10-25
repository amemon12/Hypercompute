"""Optimizer provenance for one model on this host alone: which coefficients are measured and which are priors.

  uv run --no-sync python scripts/capacity/explain_host.py --model qwen2.5-7b-instruct-q4_k_m

Builds this machine's NodeView the way coordinator.build does (inventory.collect, which runs `llama-server --version`
and `--list-devices` and waits for both, plus the default contribution), then calls CapacityService.explain with no
peers, the real measurement store and the model catalog, and prints its lines: the best layout, the measured
coefficients with their evidence, and the priors the score still rests on. It then lists every host coefficient the
bridge resolves (CPU and local GPU decode and prefill, storage read, stream efficiency) next to the costs.py prior it
replaces. Nothing is written and no engine is started. The live gateway is not contacted, so its leases are not counted:
the admissible bytes assume no deployment holds memory on this host.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from hypercompute.agent import inventory  # noqa: E402
from hypercompute.agent.contribution import default_contribution  # noqa: E402
from hypercompute.capacity import bridge, costs, runtime, storage_probe  # noqa: E402
from hypercompute.capacity.manifest import ManifestError  # noqa: E402
from hypercompute.capacity.measurements import MeasurementRecord, MeasurementStore  # noqa: E402
from hypercompute.capacity.service import CapacityService, CapacityUnavailable  # noqa: E402
from hypercompute.capacity.types import Coefficient, ModelManifest  # noqa: E402
from hypercompute.schemas import ContributionLimits, HostInventory, NodeState, NodeView  # noqa: E402
from hypercompute.services.models import ModelError, ModelService  # noqa: E402
from hypercompute.services.nodes import LOCAL_ID  # noqa: E402
from hypercompute.services.planner import StreamCapability  # noqa: E402

ACCELERATORS = ("CUDA", "MTL", "Vulkan", "ROCm")


def host_node(inv: HostInventory, contribution: ContributionLimits) -> NodeView:
    """This machine as NodeRegistry._make_local builds it at gateway start."""
    if inv.engine.smoke != "passed":
        state, reason = NodeState.ineligible, f"engine smoke test: {inv.engine.smoke_detail}"
    elif not any(d.name.startswith(ACCELERATORS) for d in inv.engine.devices):
        state, reason = NodeState.eligible, "no local accelerator visible to llama.cpp; host is CPU only"
    else:
        state, reason = NodeState.eligible, None
    now = datetime.now(UTC)
    return NodeView(id=LOCAL_ID, display_name=f"{inv.hostname} (this machine)", is_local=True, session_id=None,
                    state=state, state_reason=reason, joined_at=now, last_heartbeat_at=now, heartbeat_age_s=0.0,
                    inventory=inv, contribution=contribution, worker=None, worker_state=None, verification=None)


@dataclass(frozen=True)
class HostOnly:
    """The two NodeRegistry calls CapacityService makes, answered with this machine and no peers."""

    local: NodeView

    def eligible_remote(self, node_ids: list[str] | None) -> list[NodeView]:
        return []


class NoLeases:
    def active(self) -> list[dict[str, Any]]:
        return []


@dataclass(frozen=True)
class Row:
    label: str
    prior: Coefficient  # costs.py
    current: Coefficient  # what the bridge resolves now

    @property
    def ratio(self) -> float | None:
        """measured / prior, only when the current value is a measurement."""
        if self.current.provenance != "measured" or not self.prior.value:
            return None
        return self.current.value / self.prior.value


def coefficient_rows(manifest: ModelManifest, records: Sequence[MeasurementRecord], *, gpu: str | None,
                     stream: StreamCapability | None, volume: str | None, now: datetime) -> list[Row]:
    """Every host coefficient the optimizer reads, resolved by the bridge's own lookups, next to its prior."""
    devices = [("cpu", bridge.HOST_CPU, "ram")]
    if gpu is not None:
        devices.append(("cuda" if gpu.startswith("CUDA") else "metal", gpu, f"vram:{gpu}"))
    rows: list[Row] = []
    for backend, device, pool in devices:
        prior = costs.device_prior(backend, LOCAL_ID, device, pool)
        got = bridge.device_profile(backend, LOCAL_ID, device, pool, None, records, manifest, now)
        rows += [Row(f"{device} decode", prior.decode_ms_per_gib, got.decode_ms_per_gib),
                 Row(f"{device} prefill", prior.prefill_ms_per_gib_token, got.prefill_ms_per_gib_token)]
    storage_prior = costs.storage_prior(LOCAL_ID)
    storage = bridge.storage_profile(records, LOCAL_ID, stream, now, volume=volume)
    return rows + [Row("storage read", storage_prior.read_bytes_per_s, storage.read_bytes_per_s),
                   Row("stream efficiency", storage_prior.stream_efficiency, storage.stream_efficiency)]


def row_line(r: Row) -> str:
    c = r.current
    if r.ratio is None:
        return f"{r.label}: {c.provenance.upper()} {c.value:.6g} {c.unit}; {c.note}"
    when = c.measured_at.isoformat() if c.measured_at else "date not recorded"
    return (f"{r.label}: MEASURED {c.value:.6g} {c.unit} (prior {r.prior.value:.6g}, measured/prior {r.ratio:.3f}); "
            f"{c.samples} samples at {when}; {c.note}")


def explain_lines(out: dict[str, Any]) -> list[str]:
    """CapacityService.explain's answer as plain sections."""
    plan = out.get("plan") or {}
    w = out["workload"]
    lines = [f"workload: ctx {w['ctx_size']}, {w['prompt_tokens']} prompt tokens, {w['output_tokens']} output tokens, "
             f"{'warm' if w['warm'] else 'cold'}; feasible {out['feasible']}; plan heuristic "
             f"{plan.get('heuristic')} (true while any coefficient behind the score is not measured)", "",
             "optimizer explain (this host only):", *(f"  {x}" for x in out["explain"]), "",
             "measured coefficients the search read:", *(f"  {x}" for x in out["measured"] or ["(none)"]), ""]
    priors = [x for x in plan.get("warnings", []) if x.startswith("predicted with priors")]
    return lines + ["priors behind the chosen layout's score:", *(f"  {x}" for x in priors or ["(none)"]), ""]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="catalog model id (a downloaded GGUF)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=bridge.DEFAULT_PROMPT_TOKENS)
    ap.add_argument("--output-tokens", type=int, default=bridge.DEFAULT_OUTPUT_TOKENS)
    ap.add_argument("--warm", action="store_true", help="score without the cold load")
    ap.add_argument("--store-dir", type=Path, help="measurement store directory (default: runtime home / capacity)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inv = inventory.collect()
    local = host_node(inv, default_contribution(inv))
    store = MeasurementStore(args.store_dir) if args.store_dir else MeasurementStore()
    models = ModelService()
    svc = CapacityService(HostOnly(local), models, NoLeases(), store)  # type: ignore[arg-type]
    try:
        out = svc.explain(args.model, ctx_size=args.ctx, prompt_tokens=args.prompt_tokens,
                          output_tokens=args.output_tokens, warm=args.warm)
        shards = runtime.model_shards(models.primary_path(args.model))
        manifest = bridge.read_manifest(models.get(args.model), shards)
    except CapacityUnavailable as exc:
        print(f"error: {exc.code}: {exc}", file=sys.stderr)
        return 2
    except (ModelError, ManifestError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    records, note = bridge.load_records(store)
    gpu = bridge._local_gpu(local)[0]
    rows = coefficient_rows(manifest, records, gpu=gpu, stream=runtime.stream_capability(shards),
                            volume=storage_probe.volume_of(shards[0]), now=datetime.now(UTC))
    print(f"host {local.display_name}: state {local.state.value}, engine devices "
          f"{[d.name for d in inv.engine.devices]}, contributed {local.contribution.devices}, local GPU placed {gpu}")
    print(f"model {args.model}: fingerprint {manifest.fingerprint[:12]}, {manifest.n_layer} blocks + head; store "
          f"{store.path}{' (' + note + ')' if note else ''}")
    print(*explain_lines(out), sep="\n")
    print("host coefficients (bridge lookups) next to the costs.py priors they replace:")
    print(*(f"  {row_line(r)}" for r in rows), sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
