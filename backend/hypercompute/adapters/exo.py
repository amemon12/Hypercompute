"""EXO adapter: Apple-only MLX deployments served by Macs that run the pinned EXO node (`join --exo`).

The gateway runs no EXO process. It drives one Mac's EXO API over the LAN (unauthenticated, like ggml RPC):
GET /instance/previews?node_ids=.. places the model on exactly the planned Macs, POST /instance creates that
preview verbatim, and the deployment is ready only when every runner of the instance is RunnerReady or
RunnerRunning in /state/runners. EXO deletes an instance 15 s after one of its Macs goes silent and has no
failed-instance state, so a 404 on the instance means the deployment is lost. Every Mac downloads the whole
repo (EXO fetches the `main` revision itself). Facts and field names: docs/research/exo.md (bc412e2).
The host only calls a node's own verified address, and reads every answer under a size and time cap.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ..installers.exo import API_PORT, EXO_PIN, EXO_VERSION, INSTALL_HINT
from ..schemas import (
    CandidateSummary,
    Deployment,
    DeviceAssignment,
    EngineInfo,
    ExtraEngineInfo,
    ModelArtifact,
    NodeState,
    NodeView,
    Plan,
    PlanRequest,
    PlanStrategy,
    utcnow,
)
from ..units import GiB, fmt_bytes
from .base import (
    CancellationResult,
    Capabilities,
    CompletionResult,
    DeploymentHandle,
    DeploymentSpec,
    EngineAdapterError,
    EngineSnapshot,
    GenerationEvent,
    InferenceRequest,
    PlacementReport,
    StopResult,
    ValidationResult,
    artifact_engines,
)
from .exo_state import (  # the parsers stay importable from here, as before the split
    ACK_TIMEOUT_S,  # noqa: F401
    LOST,
    MAX_EVENT_BYTES,  # noqa: F401
    READY,  # noqa: F401
    ExoStreamNormalizer,
    InstanceInfo,
    Readiness,  # noqa: F401
    Shard,  # noqa: F401
    choose_preview,
    instances_for,
    observed_placement,
    readiness,
    runner_phase,  # noqa: F401
    stats_timings,  # noqa: F401
    untag,
)

if TYPE_CHECKING:
    from .registry import AdapterContext

log = logging.getLogger("hypercompute.exo")
PROCESS_KEY_PREFIX = "exo:"  # handle.process_key = exo:<EXO instance id>; handle.pid = 0 (no local process)
_INSTANCE_ID = re.compile(r"[0-9A-Za-z-]{1,64}")  # EXO instance ids are UUIDs; nothing else goes into a URL
IN_FLIGHT = "POST /instance in flight: instance="  # noted before the POST, so a crash during it leaves a trace
COMMAND_TIMEOUT_S = 45.0  # EXO holds POST/DELETE while it elects a new master
PREVIEW_TIMEOUT_S = 60.0  # a model id without a built-in card makes EXO fetch its config from Hugging Face
STATE_TIMEOUT_S = 10.0
POLL_S = 0.5
WATCH_S = 3.0
STALL_S = 600.0  # no visible load progress for this long fails the deployment
LOAD_TIMEOUT_S = 4 * 3600.0  # every Mac downloads the whole repo before it loads
UNREACHABLE_S = 45.0  # API silent this long = lost (EXO itself drops a silent node after 15 s)
DELETE_CONFIRM_S = 30.0
MLX_OVERHEAD = 1 * GiB  # per Mac, estimate: Metal runtime, activations, KV (EXO itself checks weights only)
STATE_MAX_BYTES = 8 * 1024 * 1024  # EXO state and command answers are KBs; a hostile endpoint could send GBs
COMPLETION_MAX_BYTES = 16 * 1024 * 1024  # one non-streamed chat completion
IDENTITY = {"Accept-Encoding": "identity"}  # the size caps count what crosses the wire, so no compression
LOOPBACK = frozenset({"127.0.0.1", "::1"})  # the host's own EXO only
UNREACHABLE_FIX = (f"check that `hypercompute join --exo` still runs on that Mac and that the host reaches TCP "
                   f"{API_PORT} on it (same LAN, no Wi-Fi client isolation)")
JOIN_HINT = f"on that Mac: {INSTALL_HINT}, then join with --exo"
CAPABILITIES = Capabilities(engine_id="exo", status="implemented", streaming=True, multi_host=True,
                            disk_offload=False, cancel=True, platforms=["macos-arm64"],
                            notes=f"EXO {EXO_VERSION} @ {EXO_PIN[:7]}: Apple-only MLX pipeline over the LAN; "
                                  "unauthenticated API, trusted private LAN only")


# ----------------------------------------------------------------------------- planning (pure: no network)


@dataclass(frozen=True)
class ExoNode:
    view: NodeView
    endpoint: str  # host:port of this Mac's EXO API
    budget: int
    pool_id: str
    basis: str


_ENDPOINT = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}|\[[0-9a-f:.]+\]):(\d{1,5})", re.ASCII)


def parse_endpoint(endpoint: object) -> tuple[str, int] | None:
    """(host, port) of a canonical `IPv4:port` or `[IPv6]:port`, else None: no names, scheme, user, path or query."""
    m = _ENDPOINT.fullmatch(endpoint) if isinstance(endpoint, str) else None
    if m is None:
        return None
    raw, port = m.group(1), int(m.group(2))
    host = raw[1:-1] if raw.startswith("[") else raw
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if str(ip) != host or (ip.version == 6) != raw.startswith("[") or not 0 < port < 65536:
        return None
    return host, port


def endpoint_problem(view: NodeView, endpoint: object) -> str | None:
    """Why the host must not call `endpoint` as this node's EXO API, else None. The friend's agent reports it, so
    only the node's own verified worker address (loopback for the host itself) on EXO's API port is trusted."""
    shown = str(endpoint)[:80]
    hp = parse_endpoint(endpoint)
    if hp is None:
        return f"EXO endpoint {shown!r} is not an IP:port"
    own = LOOPBACK if view.is_local else frozenset({view.worker.host} if view.worker else ())
    if hp[0] not in own:
        return f"EXO endpoint {shown} is not this node's own address ({', '.join(sorted(own)) or 'none known'})"
    if hp[1] != API_PORT:
        return f"EXO endpoint {shown} is not on EXO's API port {API_PORT}"
    return None


def _reported(view: NodeView) -> ExtraEngineInfo | None:
    return next((x for x in view.inventory.extra_engines if x.engine == "exo"), None)


def exo_entry(view: NodeView) -> ExtraEngineInfo | None:
    """The node's EXO entry when its agent reports EXO running at the node's own address."""
    e = _reported(view)
    return e if e is not None and e.status == "running" and endpoint_problem(view, e.endpoint) is None else None


def _unusable(view: NodeView) -> str | None:
    os_ = view.inventory.os
    if os_.system != "macos":
        return f"not a Mac ({os_.system}); EXO serves only from Apple Silicon Macs"
    if os_.machine != "arm64":
        return f"not Apple Silicon ({os_.machine})"
    e = _reported(view)
    if e is None or e.status != "running" or not e.endpoint:
        state = f"EXO is {e.status}" + (f" ({e.detail})" if e.detail else "") if e else "EXO is not running"
        return f"{state}; {JOIN_HINT}"
    if (bad := endpoint_problem(view, e.endpoint)) is not None:
        return f"{bad}; the host only calls a node's own EXO API"
    if not view.is_local and view.state != NodeState.eligible:
        return f"node is {view.state.value}" + (f" ({view.state_reason})" if view.state_reason else "")
    return None


def _budget(view: NodeView) -> tuple[int, str, str]:
    pools = view.inventory.memory_pools
    pool = next((p for p in pools if p.kind == "unified"), None) or next((p for p in pools if p.id in ("uma", "ram")),
                                                                         None)
    pool_id, limit, reserve = (pool.id if pool else "uma"), view.contribution.max_bytes, view.contribution.reserve_bytes
    if pool is None or pool.free_bytes is None:
        return limit, pool_id, "contribution limit (free memory not measured)"
    free_cap = pool.free_bytes - reserve
    if limit <= free_cap:
        return limit, pool_id, "contribution limit"
    return max(0, free_cap), pool_id, f"free unified memory minus {fmt_bytes(reserve)} reserve"


def exo_nodes(req: PlanRequest, local: NodeView, remote: list[NodeView]) -> tuple[list[ExoNode], list[str], list[str]]:
    """(usable Macs, problems that make the plan infeasible, warnings). Explicitly chosen nodes must all qualify."""
    explicit = set(req.node_ids or [])
    local_only = req.strategy == PlanStrategy.local_only
    pool = [local, *([] if local_only else [v for v in remote if req.node_ids is None or v.id in explicit])]
    nodes, problems, warnings = [], [], []
    for v in pool:
        why = _unusable(v)
        if why is None:
            budget, pool_id, basis = _budget(v)
            nodes.append(ExoNode(v, exo_entry(v).endpoint, budget, pool_id, basis))  # type: ignore[union-attr]
        elif v.id in explicit or (local_only and v.is_local):
            problems.append(f"{v.display_name}: {why}")
        elif not v.is_local:
            warnings.append(f"{v.display_name} skipped: {why}")
    return nodes, problems, warnings


def _fits(chosen: list[ExoNode], size: int) -> bool:
    total = sum(n.budget for n in chosen)
    return total > 0 and all(size * n.budget // total + MLX_OVERHEAD <= n.budget for n in chosen)


def _select(nodes: list[ExoNode], size: int, strategy: PlanStrategy) -> list[ExoNode]:
    """all_selected: every usable Mac. Otherwise the fewest Macs that fit (one Mac avoids ring hops)."""
    ranked = sorted(nodes, key=lambda n: (not n.view.is_local, -n.budget))
    if strategy == PlanStrategy.all_selected:
        return ranked
    chosen: list[ExoNode] = []
    for n in ranked:
        chosen.append(n)
        if _fits(chosen, size):
            break
    return chosen


def _assignments(chosen: list[ExoNode], size: int) -> list[DeviceAssignment]:
    """Weights split in proportion to each Mac's budget, as EXO splits layers by free memory. The real split comes
    from the preview at validate time. endpoint stays None: these are not llama.cpp RPC devices."""
    total = sum(n.budget for n in chosen) or 1
    out = []
    for n in chosen:
        share = size * n.budget // total
        out.append(DeviceAssignment(node_id=n.view.id, device="MLX", endpoint=None, pool_id=n.pool_id, layers=[],
                                    holds_output=False, weight_bytes=share, kv_bytes=0, compute_bytes_est=0,
                                    overhead_bytes_est=MLX_OVERHEAD, total_bytes_est=share + MLX_OVERHEAD,
                                    budget_bytes=n.budget, fits=share + MLX_OVERHEAD <= n.budget))
    return out


def _artifact_reasons(art: ModelArtifact) -> list[str]:
    reasons = []
    if art.format != "mlx":
        reasons.append(f"EXO serves MLX models only; {art.id} is {art.format}")
    elif "exo" not in artifact_engines(art):
        reasons.append(f"the manifest does not list exo among the engines for {art.id}")
    if not art.size_bytes:
        reasons.append(f"the manifest gives no size_bytes for {art.id}; EXO plans need the weight size")
    return reasons


def plan(req: PlanRequest, art: ModelArtifact, local: NodeView, remote: list[NodeView],
         ctx: AdapterContext | None = None) -> Plan:
    """An EXO plan over the Macs whose agents run EXO. Pure: EXO's own preview is fetched in validate()."""
    reasons = _artifact_reasons(art)
    nodes, problems, warnings = exo_nodes(req, local, remote)
    reasons += problems
    if not nodes and not problems:
        reasons.append(f"no Mac in the session runs EXO; {JOIN_HINT}")
    size = art.size_bytes or 0
    chosen = _select(nodes, size, req.strategy) if nodes and not reasons else []
    assignments = _assignments(chosen, size) if chosen else []
    if chosen and not all(a.fits for a in assignments):
        offer = sum(n.budget for n in chosen)
        reasons.append(f"{art.id} needs about {fmt_bytes(size + MLX_OVERHEAD * len(chosen))} on the EXO Macs; "
                       f"they offer {fmt_bytes(offer)}")
    if chosen and art.revision:
        warnings.append(f"EXO downloads the current `main` of {art.repo} on every Mac; the manifest revision "
                        f"{art.revision[:10]} is not enforced by EXO")
    feasible = bool(chosen) and not reasons
    primary = chosen[0] if chosen else None
    options = {} if primary is None else {
        "api": f"http://{primary.endpoint}", "endpoint": primary.endpoint, "exo_model_id": art.repo,
        "version": EXO_PIN, "sharding": "Pipeline", "instance_meta": "MlxRing",
        "nodes": [{"node_id": n.view.id, "endpoint": n.endpoint, "name": n.view.display_name} for n in chosen]}
    candidates = [CandidateSummary(label=n.view.display_name, devices=["MLX"], feasible=n in chosen,
                                   reason=f"budget {fmt_bytes(n.budget)} ({n.basis})") for n in nodes]
    return Plan(id=uuid.uuid4().hex, model_id=req.model_id, strategy=req.strategy, ctx_size=req.ctx_size,
                feasible=feasible, fit_label="estimated_fit" if feasible else "does_not_fit", heuristic=True,
                assignments=assignments, device_list=[], tensor_split=[], rpc_endpoints=[], cpu_resident_bytes=0,
                reasons=reasons, warnings=warnings, candidates=candidates, created_at=utcnow(), engine="exo",
                tier="gpu", engine_options=options)


# ----------------------------------------------------------------------------- adapter


def plan_endpoint_problems(api: object, nodes: object, view_of: Callable[[str], NodeView | None] | None) -> list[str]:
    """Checked before any request: every planned EXO endpoint is a strict IP:port (its node's own verified address
    when `view_of` reads the registry, else at least on EXO's API port) and the API is http:// one of them, so no
    stored plan aims the host elsewhere."""
    if not isinstance(nodes, list) or not nodes or not all(isinstance(n, dict) for n in nodes):
        return ["the plan's EXO node list is unreadable; check the plan again"]
    problems = []
    for n in nodes:
        ep, view = n.get("endpoint"), view_of(str(n.get("node_id"))) if view_of else None
        if (hp := parse_endpoint(ep)) is None:
            problems.append(f"{n.get('name')}: EXO endpoint {str(ep)[:80]!r} is not an IP:port")
        elif view_of is None:  # no registry (cleanup after a restart): at least EXO's own API port
            if hp[1] != API_PORT:
                problems.append(f"{n.get('name')}: EXO endpoint {ep} is not on EXO's API port {API_PORT}")
        else:
            why = endpoint_problem(view, ep) if view is not None else "the node is no longer in the session"
            problems += [f"{n.get('name')}: {why}"] if why else []
    if not isinstance(api, str) or api not in {f"http://{n.get('endpoint')}" for n in nodes}:
        problems.append(f"the plan's EXO API {str(api)[:80]!r} is not one of its Macs; check the plan again")
    return problems


@dataclass(frozen=True)
class Reply:
    """An EXO answer read under a size cap (httpx alone buffers a body of any size)."""

    status_code: int
    content: bytes
    url: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> object:
        return json.loads(self.content)


SUSPECT_FIX = "make sure that address runs the pinned EXO started by `hypercompute join --exo`"


async def read_capped(resp: httpx.Response, cap: int) -> Reply:
    """The whole body, refused past `cap` bytes or when compressed (the cap must count what really arrives)."""
    url = str(resp.request.url)
    encoding = resp.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in ("", "identity"):
        raise EngineAdapterError("exo_bad_response", f"EXO sent a {encoding[:20]}-encoded answer from {url}; refused",
                                 fix=SUSPECT_FIX)
    body = bytearray()
    async for chunk in resp.aiter_bytes():
        body += chunk
        if len(body) > cap:
            raise EngineAdapterError("exo_too_large", f"EXO sent more than {fmt_bytes(cap)} from {url}; refused",
                                     fix=SUSPECT_FIX)
    return Reply(resp.status_code, bytes(body), url)


def error_message(r: Reply) -> str:
    """EXO errors are {"error": {"message": ...}}."""
    try:
        body = r.json()
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:300]}"
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("message"):
        return f"HTTP {r.status_code}: {err['message']}"
    return f"HTTP {r.status_code}: {str(body)[:300]}"


def _json(r: Reply) -> object:
    try:
        return r.json()
    except ValueError as exc:
        raise EngineAdapterError("exo_bad_response", f"EXO sent non-JSON from {r.url}: {r.text[:200]}") from exc


@dataclass(frozen=True)
class _Chosen:
    api: str
    model: str
    preview: dict
    hc_of: dict[str, str]  # EXO node id -> Hypercompute node id
    names: dict[str, str]  # EXO node id -> display name


@dataclass
class _Live:
    base: str
    alias: str
    max_tokens: int
    info: InstanceInfo
    hc_of: dict[str, str]
    names: dict[str, str]
    log_path: Path
    seen: bool = False
    gone: str | None = None  # why the instance is lost; process_alive() is False from then on
    watch: asyncio.Task | None = field(default=None, repr=False)


@dataclass
class _Active:
    resp: httpx.Response
    base: str
    normalizer: ExoStreamNormalizer


class ExoAdapter:
    engine_id = "exo"
    idle_timeout_s = 300.0  # EXO sends a keep-alive every 10 s, so this much silence means EXO itself is stuck

    def __init__(self, logs_dir: Path, client: httpx.AsyncClient | None = None, nodes: Any = None) -> None:
        self._logs = logs_dir
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(self.idle_timeout_s, connect=5.0))
        self._nodes = nodes  # services.nodes.NodeRegistry: validate() rechecks each Mac's verified address
        self._chosen: dict[str, _Chosen] = {}  # deployment id -> preview picked by validate()
        self._live: dict[str, _Live] = {}  # process key -> instance being served
        self._active: dict[str, _Active] = {}  # request id -> open stream

    async def aclose(self) -> None:
        for live in self._live.values():
            if live.watch:
                live.watch.cancel()
        await self._client.aclose()

    async def inspect(self) -> EngineSnapshot:
        info = EngineInfo(installed=True, release=f"exo-{EXO_VERSION}", commit=EXO_PIN, variant="macos-arm64-mlx",
                          accelerator="metal")
        return EngineSnapshot(capabilities=CAPABILITIES, info=info)

    async def _call(self, method: str, url: str, timeout: float, cap: int = STATE_MAX_BYTES,
                    **kw: object) -> Reply:
        """One EXO request; `timeout` bounds each read and the whole exchange, `cap` the body."""
        try:
            async with asyncio.timeout(timeout):
                async with self._client.stream(method, url, timeout=timeout, headers=IDENTITY,
                                               **kw) as resp:  # type: ignore[arg-type]
                    return await read_capped(resp, cap)
        except TimeoutError as exc:
            raise EngineAdapterError("exo_unreachable", f"EXO API {url} sent no complete answer within "
                                     f"{timeout:.0f}s", fix=UNREACHABLE_FIX) from exc
        except httpx.HTTPError as exc:
            raise EngineAdapterError("exo_unreachable", f"EXO API {url} did not answer ({exc.__class__.__name__}: "
                                     f"{exc})", fix=UNREACHABLE_FIX) from exc

    @staticmethod
    def _note(path: Path, text: str) -> bool:
        """One line per note (EXO's own text included), so failure_reason reads only notes this adapter wrote."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"{utcnow().isoformat()} {' '.join(text.splitlines())}\n")
            return True
        except OSError:
            log.warning("could not write the EXO deployment log %s", path, exc_info=True)
            return False

    @staticmethod
    def failure_reason(log_text: str) -> str | None:
        """Why a served instance was lost (the deployment watcher's message): the last FAILED note, else None."""
        for line in reversed(log_text.splitlines()):
            note = line.partition(" ")[2]
            if note.startswith("FAILED: "):
                return note.removeprefix("FAILED: ").strip()[:400] or None
        return None

    # ------------------------------------------------------------------ validate / prepare
    async def _node_id(self, endpoint: str) -> str:
        r = await self._call("GET", f"http://{endpoint}/node_id", 5.0)
        value = _json(r) if r.status_code == 200 else None
        if not isinstance(value, str) or not value:
            raise EngineAdapterError("exo_bad_response", f"EXO at {endpoint} answered /node_id with "
                                     f"{error_message(r)}", fix=UNREACHABLE_FIX)
        return value

    async def _resolve_nodes(self, nodes: list[dict]) -> tuple[dict[str, str], dict[str, str], list[str]]:
        hc_of, names, problems = {}, {}, []
        for n in nodes:
            try:
                exo_id = await self._node_id(str(n.get("endpoint")))
            except EngineAdapterError as exc:
                problems.append(f"{n.get('name')}: {exc} Fix: {exc.fix}")
                continue
            hc_of[exo_id], names[exo_id] = str(n.get("node_id")), str(n.get("name") or exo_id[:8])
        return hc_of, names, problems

    async def validate(self, spec: DeploymentSpec) -> ValidationResult:
        """Map the planned Macs to EXO node ids and ask EXO for a placement on exactly those Macs."""
        opts = spec.plan.engine_options
        api, model, nodes = opts.get("api"), opts.get("exo_model_id"), opts.get("nodes") or []
        if not api or not model or not nodes:
            return ValidationResult(ok=False, problems=["the plan carries no EXO placement; check the plan again"])
        bad = plan_endpoint_problems(api, nodes, self._nodes.get if self._nodes is not None else None)
        if bad:
            return ValidationResult(ok=False, problems=bad)
        hc_of, names, problems = await self._resolve_nodes(nodes)
        if problems:
            return ValidationResult(ok=False, problems=problems)
        try:
            busy = instances_for(_json(await self._call("GET", f"{api}/state/instances", STATE_TIMEOUT_S)), model)
            r = await self._call("GET", f"{api}/instance/previews", PREVIEW_TIMEOUT_S,
                                 params=[("model_id", model), *(("node_ids", i) for i in hc_of)])
        except EngineAdapterError as exc:
            return ValidationResult(ok=False, problems=[f"{exc} Fix: {exc.fix}" if exc.fix else str(exc)])
        if busy:
            problems.append(f"EXO already runs instance {', '.join(busy)} for {model} (left by an earlier "
                            f"deployment?). Fix: delete it in the EXO dashboard at {api}/ and retry")
        if r.status_code != 200:
            problems.append(f"EXO could not preview {model}: {error_message(r)}")
            return ValidationResult(ok=False, problems=problems)
        preview, errors = choose_preview(_json(r), set(hc_of))
        if preview is None:
            problems.append(f"EXO found no placement of {model} on {', '.join(names.values())}: {'; '.join(errors)}")
        elif not problems:
            self._chosen[spec.deployment_id] = _Chosen(api, model, preview, hc_of, names)
        return ValidationResult(ok=not problems, problems=problems)

    async def prepare(self, spec: DeploymentSpec) -> DeploymentHandle:
        """POST the previewed instance verbatim; its instance id is known up front."""
        chosen = self._chosen.pop(spec.deployment_id, None)
        if chosen is None:
            raise EngineAdapterError("not_validated", "no EXO placement was previewed for this deployment",
                                     fix="check the plan again")
        info = InstanceInfo.parse(chosen.preview["instance"])
        log_path = self._logs / f"exo-{spec.deployment_id}.log"
        if not self._note(log_path, f"{IN_FLIGHT}{info.instance_id}"):  # engine_args are saved only after this
            raise EngineAdapterError("exo_log", f"could not record the instance in {log_path} before creating it",
                                     fix="free disk space or fix that folder's permissions, then deploy again")
        try:
            r = await self._call("POST", f"{chosen.api}/instance", COMMAND_TIMEOUT_S,
                                 json={"instance": chosen.preview["instance"]})
        except EngineAdapterError:  # EXO may still apply the command (master election): do not leave it behind
            result = await self._delete(chosen.api, info.instance_id, None)
            self._note(log_path, f"POST /instance failed; best-effort delete: {result.detail}")
            raise
        if r.status_code != 200:
            self._note(log_path, f"POST /instance refused: {error_message(r)}")
            raise EngineAdapterError("exo_rejected", f"EXO refused the instance: {error_message(r)}",
                                     fix="free memory on the Macs or pick a smaller model, then check the plan again")
        key = f"{PROCESS_KEY_PREFIX}{info.instance_id}"
        max_tokens = spec.artifact.max_output_tokens if spec.artifact else 1024
        self._live[key] = _Live(chosen.api, spec.alias, max_tokens, info, chosen.hc_of, chosen.names, log_path)
        spans = ", ".join(f"{chosen.names.get(s.node_id, s.node_id)} layers {s.start}-{s.end - 1} "
                          f"({fmt_bytes(s.weight_bytes)})" for s in info.shards)
        self._note(log_path, f"created EXO instance {info.instance_id} for {chosen.model} via {chosen.api}: {spans}")
        return DeploymentHandle(deployment_id=spec.deployment_id, process_key=key, base_url=chosen.api,
                                log_path=log_path, pid=0,
                                args=["POST", f"{chosen.api}/instance", f"model={chosen.model}",
                                      f"instance={info.instance_id}", f"macs={','.join(chosen.names.values())}"])

    # ------------------------------------------------------------------ readiness / watch
    def _require(self, handle: DeploymentHandle) -> _Live:
        live = self._live.get(handle.process_key)
        if live is None:
            raise EngineAdapterError("not_running", f"EXO deployment {handle.deployment_id[:8]} is not active")
        return live

    async def _snapshot(self, live: _Live) -> tuple[bool, dict, dict | None]:
        iid = live.info.instance_id
        r = await self._call("GET", f"{live.base}/state/instances/{iid}", STATE_TIMEOUT_S)
        if r.status_code not in (200, 404):
            raise EngineAdapterError("exo_state", f"EXO state read failed: {error_message(r)}")
        runners = _json(await self._call("GET", f"{live.base}/state/runners", STATE_TIMEOUT_S))
        runners = runners if isinstance(runners, dict) else {}
        idle = any(untag(runners.get(s.runner_id))[0] in (None, "RunnerIdle") for s in live.info.shards)
        downloads = None
        if r.status_code == 200 and idle:
            got = _json(await self._call("GET", f"{live.base}/state/downloads", STATE_TIMEOUT_S))
            downloads = got if isinstance(got, dict) else None
        return r.status_code == 200, runners, downloads

    async def wait_ready(self, handle: DeploymentHandle, timeout_s: float = LOAD_TIMEOUT_S,
                         on_progress: Callable[[str], None] | None = None) -> None:
        live = self._require(handle)
        loop = asyncio.get_running_loop()
        start = last_change = loop.time()
        last_text, unreachable_since = "", None
        while loop.time() - start < timeout_s:
            now = loop.time()
            try:
                present, runners, downloads = await self._snapshot(live)
                unreachable_since = None
            except EngineAdapterError as exc:
                unreachable_since = unreachable_since or now
                if exc.code != "exo_unreachable" or now - unreachable_since > UNREACHABLE_S:
                    self._fail(live, str(exc))
                    raise
                await asyncio.sleep(POLL_S)
                continue
            live.seen = live.seen or present
            r = readiness(live.info, present, live.seen, now - start, runners, downloads, live.names)
            if r.state == "failed":
                self._fail(live, r.text)
                raise EngineAdapterError("exo_failed", r.text, fix=f"see {live.log_path}")
            if r.state == "ready":
                self._note(live.log_path, r.text)
                live.watch = asyncio.create_task(self._watch(live))
                return
            if r.text != last_text:
                last_text, last_change = r.text, now
                self._note(live.log_path, r.text)
                if on_progress:
                    on_progress(r.text)
            elif now - last_change > STALL_S:
                self._fail(live, f"no progress for {STALL_S / 60:.0f} min: {r.text}")
                raise EngineAdapterError("load_stalled", f"EXO made no progress for {STALL_S / 60:.0f} min ({r.text})")
            await asyncio.sleep(POLL_S)
        raise EngineAdapterError("load_timeout", f"the EXO instance was not ready after {timeout_s / 60:.0f} min "
                                 f"({last_text})")

    def _fail(self, live: _Live, why: str) -> None:
        live.gone = live.gone or why
        self._note(live.log_path, f"FAILED: {why}")

    async def _watch(self, live: _Live) -> None:
        """While serving: the instance must stay in EXO's state; a 404 or a long-silent API means it is lost."""
        loop = asyncio.get_running_loop()
        since: float | None = None
        while live.gone is None:
            await asyncio.sleep(WATCH_S)
            try:
                r = await self._call("GET", f"{live.base}/instance/{live.info.instance_id}", STATE_TIMEOUT_S)
            except EngineAdapterError as exc:
                since = since or loop.time()
                if loop.time() - since > UNREACHABLE_S:
                    cause = exc.__cause__.__class__.__name__ if exc.__cause__ else exc.code
                    self._fail(live, f"EXO API {live.base} unreachable for {UNREACHABLE_S:.0f}s ({cause})")
                continue
            since = None
            if r.status_code == 404:
                self._fail(live, LOST)

    def process_alive(self, handle: DeploymentHandle) -> bool:
        live = self._live.get(handle.process_key)
        return live is not None and live.gone is None

    def read_log(self, handle: DeploymentHandle) -> str:
        try:
            return handle.log_path.read_text(encoding="utf-8", errors="replace")[-512 * 1024:]
        except OSError:
            return ""

    async def observe(self, handle: DeploymentHandle, spec: DeploymentSpec,
                      validation: ValidationResult) -> PlacementReport:
        live = self._require(handle)
        runners = _json(await self._call("GET", f"{live.base}/state/runners", STATE_TIMEOUT_S))
        report = observed_placement(live.info, runners if isinstance(runners, dict) else {}, live.names,
                                    live.hc_of, {a.node_id for a in spec.plan.assignments})
        self._note(live.log_path, f"placement: {report.detail}")
        return report

    # ------------------------------------------------------------------ stop
    async def _gone_within(self, base: str, iid: str, budget_s: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget_s
        while loop.time() < deadline:
            try:
                if (await self._call("GET", f"{base}/instance/{iid}", STATE_TIMEOUT_S)).status_code == 404:
                    return True
            except EngineAdapterError:
                pass
            await asyncio.sleep(POLL_S)
        return False

    async def stop(self, handle: DeploymentHandle) -> StopResult:
        live = self._live.pop(handle.process_key, None)
        if live and live.watch:
            live.watch.cancel()
        iid, base = handle.process_key.removeprefix(PROCESS_KEY_PREFIX), handle.base_url
        result = await self._delete(base, iid, live)
        self._note(handle.log_path, f"stop: {result.detail}")
        return result

    async def _delete(self, base: str, iid: str, live: _Live | None) -> StopResult:
        try:
            r = await self._call("DELETE", f"{base}/instance/{iid}", COMMAND_TIMEOUT_S)
        except EngineAdapterError as exc:
            if live is not None and live.gone:
                return StopResult(clean=True, detail=f"{live.gone}; nothing left to delete")
            return StopResult(clean=False, detail=f"{exc}; the instance may still hold memory on the Macs. Fix: "
                                                  f"delete instance {iid} in the EXO dashboard at {base}/")
        if r.status_code == 404:
            return StopResult(clean=True, detail=f"EXO instance {iid} was already gone")
        if r.status_code != 200:
            return StopResult(clean=False, detail=f"EXO refused to delete instance {iid}: {error_message(r)}")
        if await self._gone_within(base, iid, DELETE_CONFIRM_S):
            return StopResult(clean=True, detail=f"EXO instance {iid} deleted")
        return StopResult(clean=False, detail=f"EXO accepted the delete but still lists instance {iid} after "
                                              f"{DELETE_CONFIRM_S:.0f}s")

    def _in_flight(self, dep_id: str) -> list[str]:
        """The instance prepare() noted before POST /instance (the first such note), else [] (no log: no POST)."""
        try:
            text = (self._logs / f"exo-{dep_id}.log").read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        notes = (line.partition(" ")[2] for line in text.splitlines())
        return [n.removeprefix(IN_FLIGHT).strip() for n in notes if n.startswith(IN_FLIGHT)][:1]

    async def cleanup_orphan(self, dep: Deployment) -> tuple[bool, str]:
        """After a gateway crash: delete the instance a deployment left on the Macs, found from what was persisted
        (`instance=<id>` in engine_args, else prepare()'s in-flight note when it died during POST /instance; the API
        in the plan). True only once EXO confirms it is gone."""
        refuse = f"cannot clean up EXO for deployment {dep.id[:8]}"
        if dep.plan.engine != "exo":
            return False, f"{refuse}: it is a {dep.plan.engine} deployment"
        iids = [a.removeprefix("instance=") for a in dep.engine_args if a.startswith("instance=")]
        try:
            iids = iids or self._in_flight(dep.id)
        except OSError as exc:  # no proof that no POST /instance was in flight: keep the leases
            return False, f"{refuse}: its EXO log is unreadable ({exc.__class__.__name__}: {exc})"
        if not iids:
            return True, "no EXO instance was created for this deployment (it failed before POST /instance)"
        opts = dep.plan.engine_options
        problems = plan_endpoint_problems(opts.get("api"), opts.get("nodes"), None)
        if not _INSTANCE_ID.fullmatch(iids[-1]):
            problems.append(f"instance id {iids[-1][:80]!r} is not an EXO instance id")
        if problems:
            return False, f"{refuse}: {'; '.join(problems)}"
        result = await self._delete(str(opts["api"]), iids[-1], None)
        self._note(self._logs / f"exo-{dep.id}.log", f"cleanup after a gateway restart: {result.detail}")
        return result.clean, result.detail

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _body(live: _Live, body: dict, stream: bool) -> dict:
        """EXO matches `model` against its instance's model id exactly and needs max_tokens."""
        out = {**body, "model": live.info.model_id, "stream": stream}
        out.setdefault("max_tokens", live.max_tokens)
        return out

    async def complete(self, handle: DeploymentHandle, req: InferenceRequest) -> CompletionResult:
        live = self._require(handle)
        try:
            async with self._client.stream("POST", f"{handle.base_url}/v1/chat/completions", headers=IDENTITY,
                                           json=self._body(live, req.body, False)) as resp:
                r = await read_capped(resp, COMPLETION_MAX_BYTES)
        except EngineAdapterError as exc:  # recorded as a failed run, like any refused completion
            return CompletionResult(status_code=502, body={"error": {"message": str(exc), "type": exc.code}})
        try:
            body = r.json()
        except ValueError:
            body = {"error": {"message": r.text[:500]}}
        if isinstance(body, dict) and "model" in body:
            body = {**body, "model": live.alias}
        return CompletionResult(status_code=r.status_code, body=body if isinstance(body, dict) else {"data": body})

    async def stream(self, handle: DeploymentHandle, req: InferenceRequest) -> AsyncIterator[GenerationEvent]:
        live = self._require(handle)
        norm = ExoStreamNormalizer(live.alias)
        async with self._client.stream("POST", f"{handle.base_url}/v1/chat/completions", headers=IDENTITY,
                                       json=self._body(live, req.body, True)) as resp:
            if resp.status_code != 200:
                refused = await read_capped(resp, STATE_MAX_BYTES)
                raise EngineAdapterError("upstream_error", f"EXO {error_message(refused)}")
            self._active[req.request_id] = _Active(resp, handle.base_url, norm)
            try:
                async for chunk in resp.aiter_raw():
                    for ev in norm.feed(chunk):
                        yield ev
                for ev in norm.flush():
                    yield ev
            finally:
                self._active.pop(req.request_id, None)

    async def cancel(self, request_id: str) -> CancellationResult:
        active = self._active.pop(request_id, None)
        if active is None:
            return CancellationResult(request_id, False, "no active upstream request")
        command = active.normalizer.command_id
        if command:
            try:
                await self._call("POST", f"{active.base}/v1/cancel/{command}", 5.0)
            except EngineAdapterError as exc:  # closing the connection below also cancels in EXO
                log.info("EXO cancel of %s failed: %s", command, exc)
        await active.resp.aclose()
        return CancellationResult(request_id, True, "EXO generation cancelled and connection closed")


# ----------------------------------------------------------------------------- module contract


def build(ctx: AdapterContext) -> ExoAdapter:
    """Always available: the gateway only talks HTTP to the Macs' EXO APIs, nothing is installed here."""
    return ExoAdapter(ctx.logs_dir, nodes=ctx.nodes)


def status(ctx: AdapterContext) -> ExtraEngineInfo:
    views = ctx.nodes.views() if ctx.nodes is not None else []
    running = [(v, e) for v in views if (e := exo_entry(v)) is not None]
    if running:
        names = ", ".join(v.display_name for v, _ in running)
        return ExtraEngineInfo(engine="exo", installed=True, status="running", version=EXO_PIN,
                               endpoint=running[0][1].endpoint, detail=f"{len(running)} Mac(s) run EXO: {names}")
    return ExtraEngineInfo(engine="exo", installed=True, status="installed", version=EXO_PIN,
                           detail=f"no Mac in the session runs EXO yet; {JOIN_HINT}")
