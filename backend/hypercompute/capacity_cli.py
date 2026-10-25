"""`hypercompute capacity`: memory pools, optimizer explanations, replan decisions and replacement, asked of the running
host gateway on 127.0.0.1 and printed as plain text. `peers`, `seed` and `worker` manage peers that joined with
--managed (their worker devices and their llama.cpp RPC cache).

A write (replan, replace) carries the owner token from /api/owner/bootstrap, the same loopback-only endpoint the
dashboard uses, so no secret is stored or passed on the command line.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections.abc import Callable
from typing import Any

import httpx

from . import config
from .agent.join import say
from .units import fmt_bytes

CAPACITY_TIMEOUT_S = 120.0  # explain reads the GGUF tensor table when its manifest is not cached
SWITCH_TIMEOUT_S = 3600.0  # replace and replan answer once the new deployment is ready or has failed
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # deployment and plan ids, as the gateway accepts them
_NODE_ID_RE = re.compile(r"\b([0-9a-f]{8})[0-9a-f]{24}\b")  # peer node ids, printed by their first 8 characters
_FULL_NODE_RE = re.compile(r"[0-9a-f]{32}")
_NODE_PREFIX_RE = re.compile(r"[0-9a-f]{4,31}")
_PROVENANCE_RE = re.compile(r"\((prior|estimate|reported|measured)\b")  # Coefficient.provenance after a value
UNMARKED_LABEL = "note"  # an assumption line without a provenance marker: context, not a number's source


def _fail(message: str, fix: str | None = None, code: int = 2) -> int:
    say(f"error: {message}")
    if fix:
        say(f"fix:   {fix}")
    return code


class _Refused(Exception):
    """The gateway answered with an error: its status, code and detail as the message, and its fix."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


def _refusal(r: httpx.Response) -> _Refused:
    try:
        body = r.json()
    except ValueError:
        return _Refused(f"HTTP {r.status_code}: {r.text[:300]}")
    if not isinstance(body, dict):
        return _Refused(f"HTTP {r.status_code}: {body}")
    detail = body.get("detail")
    if isinstance(detail, list):  # FastAPI request validation: [{loc, msg}, ...]
        detail = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}"
                           for d in detail if isinstance(d, dict))
    code = f" {body['code']}" if body.get("code") else ""
    return _Refused(f"HTTP {r.status_code}{code}: {detail}", body.get("fix"))


def _gateway(client: httpx.Client, method: str, path: str, *, owner: bool = False, **kwargs: Any) -> Any:
    """One owner API call; a write carries the owner token from the loopback-only bootstrap, as the dashboard does."""
    headers = {}
    if owner:
        boot = client.get("/api/owner/bootstrap")
        if boot.status_code != 200:
            raise _refusal(boot)
        body = boot.json()
        token = body.get("owner_token") if isinstance(body, dict) else None
        if not token:
            raise _Refused("the host handed out no owner token", "check `uv run hypercompute status`")
        headers[config.OWNER_HEADER] = token
    r = client.request(method, path, headers=headers, **kwargs)
    if r.status_code >= 400:
        raise _refusal(r)
    return r.json()


def _b(value: Any, missing: str = "-") -> str:
    return fmt_bytes(value) if isinstance(value, int) else missing


def _num(value: Any, spec: str) -> str:
    return format(value, spec) if isinstance(value, int | float) else "-"


def _span(units: Any) -> str:
    """Unit indices as ranges: [0, 1, 2, 7] -> "0-2,7"."""
    nums = sorted({u for u in units or [] if isinstance(u, int)})
    runs = [[u for _, u in g] for _, g in itertools.groupby(enumerate(nums), key=lambda t: t[1] - t[0])]
    return ",".join(str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs) or "-"


def pools_lines(report: dict) -> list[str]:
    hm = report.get("host_memory") or {}
    if hm.get("error"):
        lines = [f"host memory: not observed ({hm['error']})"]
    else:
        lines = [f"host memory: {_b(hm.get('available'))} available of {_b(hm.get('total'))}, commit "
                 f"{_b(hm.get('commit_charge'))} of {_b(hm.get('commit_limit'))}, file cache "
                 f"{_b(hm.get('system_cache'))} (observed {hm.get('observed_at', '-')} via {hm.get('source', '-')})"]
    lines.append(f"  {'NODE':<9}{'POOL':<12}{'KIND':<14}{'TOTAL':>11}{'FREE':>11}{'RESERVE':>11}{'CAP':>11}"
                 f"{'ADMISSIBLE':>12}{'AGE s':>7}  DEVICES (SOURCE)")
    for p in report.get("pools") or []:
        lines.append(f"  {str(p.get('node_id'))[:8]:<9}{str(p.get('pool_id')):<12}{str(p.get('kind')):<14}"
                     f"{_b(p.get('total_bytes')):>11}{_b(p.get('free_bytes')):>11}{_b(p.get('reserve_bytes')):>11}"
                     f"{_b(p.get('cap_bytes'), 'none'):>11}{_b(p.get('admissible_bytes')):>12}"
                     f"{_num(p.get('age_s', p.get('age')), '.0f'):>7}  "
                     f"{','.join(map(str, p.get('devices') or [])) or '-'} ({p.get('source', '-')})")
    lines += [f"  excluded {str(e.get('node_id'))[:8]} {e.get('pool_id')}: {e.get('reason')}"
              for e in report.get("excluded") or []]
    return lines + [f"  note: {n}" for n in report.get("notes") or []]


def _label(line: str) -> str:
    """An assumption's provenance: the first "(prior: ...)", "(estimate: ...)", "(reported: ...)" or "(measured: ...)"
    marker the optimizer wrote after the value (a note may mention another word later). A line without a marker is
    context the optimizer added, not a number with a source, so it is labelled "note" and never "prior"."""
    marker = _PROVENANCE_RE.search(line)
    return marker.group(1) if marker else UNMARKED_LABEL


def _entry(item: Any) -> str:
    return ", ".join(f"{k} {v}" for k, v in item.items()) if isinstance(item, dict) else str(item)


def _layout(c: dict) -> str:
    """Devices and units in execution order (the service's layout text, else built from the stages), then the stages
    that stream units from the SSD or reach a peer over a network path. Node ids are cut to 8 characters."""
    stages = [s for s in c.get("stages") or [] if isinstance(s, dict)]
    text = c["layout"] if isinstance(c.get("layout"), str) else ", ".join(
        f"{s.get('node_id')}:{s.get('device')} [{_span(s.get('units'))}]" for s in stages)
    notes = [f"{s.get('node_id')}:{s.get('device')} " + " ".join(
                 ([f"{len(s['streamed_units'])} streamed"] if s.get("streamed_units") else [])
                 + ([str(s["path"])] if s.get("path") not in (None, "local") else []))
             for s in stages if s.get("streamed_units") or s.get("path") not in (None, "local")]
    return _NODE_ID_RE.sub(r"\1", (text or "-") + (f" ({'; '.join(notes)})" if notes else ""))


def _candidate_row(rank: int, c: Any) -> str:
    if not isinstance(c, dict):
        return f"  {rank:<5}{c}"
    fits = "" if c.get("feasible", True) else "  (does not fit)"
    return (f"  {c.get('rank', rank)!s:<5}{_num(c.get('objective_ms'), '.0f'):>10}"
            f"{_num(c.get('decode_ms'), '.1f'):>11}  {_layout(c)}{fits}")


def explain_lines(body: dict) -> list[str]:
    """The service's explain lines verbatim, then the candidates, skipped devices and each assumption's label."""
    workload = body.get("workload")
    lines = [f"{body.get('model_id')}: " + ", ".join(f"{k} {v}" for k, v in
                                                     (workload.items() if isinstance(workload, dict) else []))]
    lines += [str(line) for line in body.get("explain") or []]
    lines += ["", "candidates (predicted from the assumptions below, not measured):",
              f"  {'RANK':<5}{'J ms':>10}{'DECODE ms':>11}  DEVICES / UNITS"]
    lines += [_candidate_row(i, c) for i, c in enumerate(body.get("candidates") or [], 1)] or ["  none"]
    best = body.get("best") or {}
    skipped = [*(best.get("skipped") or []), *(body.get("warnings") or [])]  # by the optimizer, then before it
    lines += ["", "skipped devices:"] + ([f"  {s}" for s in skipped] or ["  none"])
    notes = [f"  {_label(str(a)):<9} {a}" for a in body.get("assumptions") or []]
    notes += [f"  {'measured':<9} {_entry(m)}" for m in body.get("measured") or []]
    return lines + ["", "assumptions:"] + (notes or ["  none"])


def decision_lines(d: dict) -> list[str]:
    head = f"{d.get('at')}  {d.get('trigger')}: {d.get('action')}"
    if d.get("deployment_id"):
        head += f"  deployment {str(d['deployment_id'])[:8]} revision {d.get('revision')}"
    numbers = []
    if d.get("current_decode_ms") is not None or d.get("proposed_decode_ms") is not None:
        numbers.append(f"decode {_num(d.get('current_decode_ms'), '.1f')} -> "
                       f"{_num(d.get('proposed_decode_ms'), '.1f')} ms/token")
    for key, text in (("relative_gain", "gain {:+.1%}"), ("reload_ms", "reload {:.0f} ms"),
                      ("replay_ms", "replay {:.0f} ms"), ("break_even_tokens", "break-even {:.0f} tokens")):
        if isinstance(d.get(key), int | float):
            numbers.append(text.format(d[key]))
    lines = [head, f"  reason: {d.get('reason')}"]
    if numbers:
        lines.append("  predicted by the optimizer, not measured: " + ", ".join(numbers))
    lines += [f"  {name}: {d[key]}" for key, name in (("proposed_plan_id", "proposed plan"), ("outcome", "outcome"))
              if d.get(key)]
    return lines


def _cap_pools(client: httpx.Client, args: argparse.Namespace) -> int:
    for line in pools_lines(_gateway(client, "GET", "/api/capacity/pools")):
        say(line)
    return 0


def _cap_explain(client: httpx.Client, args: argparse.Namespace) -> int:
    params = {"model_id": args.model_id, "ctx_size": args.ctx, "prompt_tokens": args.prompt_tokens,
              "output_tokens": args.output_tokens, "warm": "true" if args.warm else "false", "slots": args.slots}
    body = _gateway(client, "GET", "/api/capacity/explain", params=params)
    for line in [json.dumps(body, indent=2)] if args.json else explain_lines(body):
        say(line)
    return 0 if body.get("feasible") else 1


def _cap_decisions(client: httpx.Client, args: argparse.Namespace) -> int:
    decisions = _gateway(client, "GET", "/api/replan/decisions")
    if args.json:
        lines = [json.dumps(decisions, indent=2)]
    else:
        lines = [line for d in decisions for line in decision_lines(d)] or ["no replan decisions recorded"]
    for line in lines:
        say(line)
    return 0


def _cap_replan(client: httpx.Client, args: argparse.Namespace) -> int:
    decision = _gateway(client, "POST", "/api/replan/trigger", owner=True,
                        timeout=httpx.Timeout(CAPACITY_TIMEOUT_S, read=SWITCH_TIMEOUT_S))
    for line in decision_lines(decision):
        say(line)
    # a replacement that ran ended "activated", or rolled back, fell back, failed, was refused or was stopped
    outcome = decision.get("outcome")
    return 1 if decision.get("action") == "failed" or outcome not in (None, "activated") else 0


def _cap_replace(client: httpx.Client, args: argparse.Namespace) -> int:
    dep = _gateway(client, "POST", f"/api/deployments/{args.dep_id}/replace", owner=True,
                   json={"plan_id": args.plan_id}, timeout=httpx.Timeout(CAPACITY_TIMEOUT_S, read=SWITCH_TIMEOUT_S))
    short = str(dep.get("id"))[:8]
    replaces = f" (replaces {str(dep['replaces'])[:8]})" if dep.get("replaces") else ""
    say(f"deployment {short} revision {dep.get('revision')}: {dep.get('state')}{replaces}")
    if dep.get("error"):
        say(f"error: {dep['error']}")
    if dep.get("state") != "ready":
        return 1
    if dep.get("replaces") == args.dep_id:  # the first attempt, on the requested plan, is serving
        return 0
    # a recovery: the requested plan failed and the previous plan (or a fallback) was loaded in its place
    plan = dep.get("plan") if isinstance(dep.get("plan"), dict) else {}
    history = [h for h in dep.get("history") or [] if isinstance(h, dict)]
    detail = next((h.get("detail") for h in history if h.get("detail")), None)
    say(f"plan {args.plan_id[:8]} did not become ready; deployment {short} serves plan "
        f"{str(plan.get('id'))[:8]} instead" + (f" ({detail})" if detail else ""))
    return 1


def _seed_line(seed: dict) -> str:
    return (f"  seed on peer: {seed.get('model_id')} job {str(seed.get('job_id'))[:8]} {seed.get('state')}, "
            f"{_b(seed.get('bytes_done'))} of {_b(seed.get('bytes_total'))}, {seed.get('tensors_done')}/"
            f"{seed.get('tensors_total')} tensors ({seed.get('tensors_present')} already cached), "
            f"{seed.get('tensors_failed')} failed, parts {seed.get('parts_received')}/{seed.get('parts')}"
            + (f": {seed['error']}" if seed.get("error") else ""))


def peer_lines(peers: list) -> list[str]:
    """One row per peer, then what the host asked and what the peer reported."""
    lines = [f"  {'NODE':<9}{'NAME':<22}{'STATE':<12}{'MANAGED':<9}{'WORKER':<14}{'ALLOWED':<12}{'CACHE':>10}"
             f"{'MAX DISK':>11}{'DISK FREE':>11}"]
    for p in peers:
        if not isinstance(p, dict):
            continue
        r = p.get("report") if isinstance(p.get("report"), dict) else {}
        worker = ",".join(r.get("worker_devices") or p.get("contribution_devices") or []) or "-"
        if r.get("worker_threads"):
            worker += f" t{r['worker_threads']}"
        lines.append(f"  {str(p.get('node_id'))[:8]:<9}{str(p.get('display_name'))[:21]:<22}{str(p.get('state')):<12}"
                     f"{'yes' if p.get('managed') else 'no':<9}{worker:<14}"
                     f"{','.join(r.get('devices_allowed') or []) or '-':<12}{_b(r.get('cache_bytes')):>10}"
                     f"{_b(r.get('max_disk_bytes')):>11}{_b(r.get('disk_free_bytes')):>11}")
        if p.get("deployments"):
            lines.append(f"  in use by: {', '.join(str(d)[:8] for d in p['deployments'])}")
        if r.get("worker_error"):
            lines.append(f"  last worker error: {r['worker_error']}")
        req = p.get("seed_request")
        if isinstance(req, dict):
            lines.append(f"  seed request: {req.get('model_id')} layers {req.get('layers')} job "
                         f"{str(req.get('job_id'))[:8]} {req.get('state')}, {req.get('tensors')} tensors "
                         f"{_b(req.get('bytes_total'))}" + (f": {req['detail']}" if req.get("detail") else ""))
        if isinstance(r.get("seed"), dict):
            lines.append(_seed_line(r["seed"]))
        if p.get("pending"):
            lines.append("  pending: " + ", ".join(f"{c.get('kind')} #{c.get('seq')}" for c in p["pending"]))
        lines += [f"  {h.get('kind')} #{h.get('seq')} {h.get('status')}: {h.get('detail') or ''}"
                  for h in (p.get("history") or [])[-3:] if isinstance(h, dict)]
    return lines if len(lines) > 1 else ["no peers have joined"]


def _resolve_node(client: httpx.Client, text: str) -> str:
    """A full node id, or a unique prefix of one (the listing prints 8 characters)."""
    if _FULL_NODE_RE.fullmatch(text):
        return text
    if not _NODE_PREFIX_RE.fullmatch(text):
        raise _Refused(f"{text!r} is not a node id or a prefix of at least 4 hex digits")
    ids = [str(p.get("node_id")) for p in _gateway(client, "GET", "/api/peers") if isinstance(p, dict)]
    match = [i for i in ids if i.startswith(text)]
    if len(match) != 1:
        raise _Refused(f"{len(match)} peers match {text!r}", "run `hypercompute capacity peers` and use more digits")
    return match[0]


def _cap_peers(client: httpx.Client, args: argparse.Namespace) -> int:
    peers = _gateway(client, "GET", "/api/peers")
    for line in [json.dumps(peers, indent=2)] if args.json else peer_lines(peers):
        say(line)
    return 0


def _cap_seed(client: httpx.Client, args: argparse.Namespace) -> int:
    node = _resolve_node(client, args.node)
    if args.cancel:
        _gateway(client, "POST", f"/api/peers/{node}/seed/cancel", owner=True)
        say(f"cancel queued for {node[:8]}; the peer stops its downloads at its next heartbeat")
        return 0
    if not args.model or (args.layers is None) == (args.plan is None):
        raise _Refused("give --model and exactly one of --layers A-B or --plan PLAN_ID (or --cancel)")
    body = {"model_id": args.model, **({"layers": args.layers} if args.layers else {"plan_id": args.plan})}
    job = _gateway(client, "POST", f"/api/peers/{node}/seed", owner=True, json=body)
    say(f"seed job {str(job.get('job_id'))[:8]} for {node[:8]}: {job.get('model_id')} layers {job.get('layers')}, "
        f"{job.get('state')}. The host hashes the tensors from its own copy, then the peer downloads them; follow "
        "with `hypercompute capacity peers`.")
    return 0


def _cap_worker(client: httpx.Client, args: argparse.Namespace) -> int:
    node = _resolve_node(client, args.node)
    body = {"devices": [d.strip() for d in args.devices.split(",") if d.strip()],
            **({"threads": args.threads} if args.threads else {})}
    out = _gateway(client, "POST", f"/api/peers/{node}/worker", owner=True, json=body)
    cmd = out.get("command") or {}
    say(f"worker request #{cmd.get('seq')} queued for {node[:8]}: -d {','.join(cmd.get('devices') or [])}"
        + (f" -t {cmd['threads']}" if cmd.get("threads") else "")
        + ". The peer restarts its worker at its next heartbeat and the host re-verifies it.")
    return 0


_CAPACITY_ACTIONS: dict[str, Callable[[httpx.Client, argparse.Namespace], int]] = {
    "pools": _cap_pools, "explain": _cap_explain, "decisions": _cap_decisions, "replan": _cap_replan,
    "replace": _cap_replace, "peers": _cap_peers, "seed": _cap_seed, "worker": _cap_worker}


def cmd_capacity(args: argparse.Namespace, client: httpx.Client | None = None) -> int:
    """`hypercompute capacity ...`: owner calls on the gateway at 127.0.0.1:--port, printed as plain text."""
    own = client is None
    client = client or httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=CAPACITY_TIMEOUT_S,
                                    trust_env=False)
    try:
        return _CAPACITY_ACTIONS[args.action](client, args)
    except _Refused as exc:
        return _fail(str(exc), exc.fix, 1)
    except httpx.TimeoutException:  # before TransportError, its base class
        fix = ("a replacement the host started keeps running there; follow it with `uv run hypercompute status`"
               if args.action in ("replan", "replace") else "try again once the host is less busy")
        return _fail(f"the host on 127.0.0.1:{args.port} did not answer in time", fix, 1)
    except httpx.TransportError as exc:
        return _fail(f"no Hypercompute host is running on 127.0.0.1:{args.port} ({exc.__class__.__name__})",
                     "start one with `uv run hypercompute host` (or pass --port)")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:  # a reply without the gateway's JSON shape
        return _fail(f"unexpected reply from the host on 127.0.0.1:{args.port}: {exc!r}",
                     f"check that it is this Hypercompute version: `uv run hypercompute status --port {args.port}`")
    finally:
        if own:
            client.close()


def _ident(value: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not an id (letters, digits, '-' and '_', at most 64)")
    return value


def add_capacity_parser(sub: Any) -> None:
    """Register `capacity` and its actions on the top-level subparsers of cli.build_parser()."""
    port = argparse.ArgumentParser(add_help=False)
    port.add_argument("--port", type=int, default=config.DEFAULT_GATEWAY_PORT)
    c = sub.add_parser("capacity", help="memory pools, optimizer explanations, replan decisions and replacement")
    c.set_defaults(func=cmd_capacity)
    actions = c.add_subparsers(dest="action", required=True)
    actions.add_parser("pools", parents=[port], help="pools the optimizer may plan onto, with admissible bytes")
    ex = actions.add_parser("explain", parents=[port], help="the optimizer's view of a model on the current nodes")
    ex.add_argument("model_id")
    ex.add_argument("--ctx", type=int, default=4096)
    ex.add_argument("--prompt-tokens", type=int, default=512)
    ex.add_argument("--output-tokens", type=int, default=256)
    ex.add_argument("--warm", action="store_true", help="the model is loaded already: do not charge the load")
    ex.add_argument("--slots", type=int, default=2, help="slot buffers for units streamed from the SSD (1-8)")
    ex.add_argument("--json", action="store_true", help="print the gateway's JSON")
    dc = actions.add_parser("decisions", parents=[port], help="the replan controller's recent decisions")
    dc.add_argument("--json", action="store_true", help="print the gateway's JSON")
    actions.add_parser("replan", parents=[port], help="evaluate the ready deployment now (needs HYPERCOMPUTE_REPLAN=1)")
    rp = actions.add_parser("replace", parents=[port], help="reload the ready deployment onto a stored plan")
    rp.add_argument("dep_id", type=_ident, help="deployment id")
    rp.add_argument("--plan-id", required=True, type=_ident, help="plan id from POST /api/plans")
    pe = actions.add_parser("peers", parents=[port], help="peers with what they allowed (--managed), worker devices, "
                                                         "cache and seed progress")
    pe.add_argument("--json", action="store_true", help="print the gateway's JSON")
    sd = actions.add_parser("seed", parents=[port], help="fill a managed peer's RPC cache with its layers of a model")
    sd.add_argument("--node", required=True, help="node id or a unique prefix (see `capacity peers`)")
    sd.add_argument("--model", help="catalog model id, downloaded and verified on this host")
    sd.add_argument("--layers", help="inclusive layer range, e.g. 30-35")
    sd.add_argument("--plan", type=_ident, help="use the layers this stored plan gives the node")
    sd.add_argument("--cancel", action="store_true", help="stop the peer's seeding instead")
    wk = actions.add_parser("worker", parents=[port], help="restart a managed peer's worker with other devices")
    wk.add_argument("--node", required=True, help="node id or a unique prefix (see `capacity peers`)")
    wk.add_argument("--devices", required=True, help="devices in order, within what the peer allowed, e.g. MTL0,CPU")
    wk.add_argument("--threads", type=int, help="rpc-server -t (default: keep the current setting)")
