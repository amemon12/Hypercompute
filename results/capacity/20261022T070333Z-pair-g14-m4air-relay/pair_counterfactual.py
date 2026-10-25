"""Counterfactual optimizer choice for the pair: the Mac's owner reserve set to 0, so its pool admits memory and the
search has to weigh the Mac on time alone (hop versus compute). Read-only; every number is a prediction."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair_predict as pp  # noqa: E402

from hypercompute.capacity import costs  # noqa: E402
from hypercompute.capacity.measurements import MeasurementStore  # noqa: E402
from hypercompute.capacity.service import CapacityService  # noqa: E402
from hypercompute.capacity.types import Workload  # noqa: E402
from hypercompute.services.models import ModelService  # noqa: E402


def main() -> int:
    nodes_json, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    pn = pp.nodes(nodes_json)
    mac = pn.mac.model_copy(update={"contribution": pn.mac.contribution.model_copy(update={"reserve_bytes": 0})})
    svc = CapacityService(pp.PairNodes(local=pn.local, mac=mac), ModelService(), pp.NoLeases(),  # type: ignore[arg-type]
                          MeasurementStore())
    svc.path_of = lambda nid: "relay" if nid == pp.MAC else "unknown"
    out = {"what": "counterfactual: the Mac contribution reserve_bytes set from 4294967296 to 0 (the owner's real "
                   "setting is 4 GiB, which makes its pool admit 0 B at the 06:40:25Z inventory); predictions only",
           "setup_prior_ms": costs.setup_prior().value, "setup_per_remote_prior_ms": costs.setup_per_remote_prior().value,
           "relay_risk_fraction_prior": costs.relay_risk_prior().value, "choices": {}}
    for name, w in (("bench_warm", Workload(2048, 512, 32, True)), ("bench_cold", Workload(2048, 512, 32, False)),
                    ("service_default_cold", Workload(2048, 512, 256, False))):
        _, f = pp.search(svc, w)
        b = f.result.best
        out["choices"][name] = {"workload": asdict(w), "best": pp.cand_json(b, f.inputs) if b else None,
                                "candidates": [pp.cand_json(c, f.inputs) for c in f.result.candidates[:5]],
                                "explain": f.result.explain(),
                                "mac_admissible_bytes": f.inputs.admissible(f"{pp.MAC}:MTL0") if f.inputs.peers else None}
        print(f"{name}: best {b.layout if False else pp.cand_json(b, f.inputs)['layout']} decode {b.decode_ms:.3f} J "
              f"{b.objective_ms:.0f}; skipped {list(b.skipped)}")
        for c in f.result.candidates[:5]:
            j = pp.cand_json(c, f.inputs)
            print(f"   cand {j['layout']}: decode {c.decode_ms:.3f} prefill {c.prefill_ms:.1f} load {c.load_ms:.0f} "
                  f"J {c.objective_ms:.0f} feasible {c.feasible}")
    path = out_dir / "counterfactual_mac_reserve_0.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    shutil.copy2(__file__, out_dir / "pair_counterfactual.py")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
