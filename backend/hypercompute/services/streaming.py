"""OpenAI-compatible inference through the ready deployment, with run recording.

Streaming passes llama-server's SSE bytes through unchanged while a copy is parsed for metrics.
TTFT is request start -> first event carrying real model output (content or reasoning text).
Token counts and decode rates come from llama.cpp's `usage`/`timings`, never from event counts.
A client disconnect closes the upstream request so llama-server stops generating.
Each deployment has as many inference slots as its engine batches (adapter.max_concurrency, else 1): a chat holds
one, warmup/benchmark/tuner measurements hold all of them so their numbers are not shared with other requests.
A non-streamed chat is served from the engine's stream too, so the idle timeout applies per event, not to the answer.
An interrupted chat (deployment stopped, or the engine connection closed, as it does when llama-server aborts on a lost
RPC peer) is recorded first and then ends with an `interrupted` error that names the run and carries the exact text
generated before the cut. A later chat with "resume_from": <run id> and the same messages continues that text as an
assistant prefill and sends only new text (see Resume). The interruption is real: the continuation is a new request,
and with temperature > 0 a new sample.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from .. import config
from ..adapters.base import DeploymentHandle, EngineAdapterError, GenerationEvent, InferenceRequest
from ..adapters.llama_rpc import LlamaRpcAdapter
from ..schemas import Deployment, DeviceAssignment, EngineTimings, GgufSummary, RunRecord, utcnow
from ..storage import Store, dumps
from .deployments import DeploymentError, DeploymentManager, idle_budget
from .events import EventBus
from .models import ModelService
from .nodes import NodeRegistry

log = logging.getLogger("hypercompute.streaming")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "seed", "repeat_penalty",
                 "presence_penalty", "frequency_penalty", "stop", "reasoning_format", "chat_template_kwargs",
                 "cache_prompt")
BUSY_WAIT_S = 30.0
# A non-streamed reply is collected in memory before it is returned; an engine (or a friend's EXO endpoint) that
# never stops must not exhaust the gateway. Far above any max_tokens a model here can produce.
COMPLETION_MAX_BYTES = 16 * 1024 * 1024
RECORD_TEXT_MAX_CHARS = 1_000_000  # run records keep at most this much generated text
BUSY_RETRY_S = 0.5  # how often an engine that answered busy is asked again while a request waits for it
NO_FINISH = "stream ended without a finish reason (engine stopped or peer lost)"
INTERRUPTED_PREFIX = "interrupted:"  # every interrupted run's error starts so; only those runs can be resumed
INTERRUPTED = f"{INTERRUPTED_PREFIX} deployment stopped or a peer was lost"
CONNECTION_LOST = (f"{INTERRUPTED_PREFIX} the engine connection closed before the reply finished (llama-server "
                   "exited, as it does when an RPC peer is lost)")
# llama-server b11770 aborts when an RPC peer stops answering (ggml/src/ggml-rpc/ggml-rpc.cpp:31, 681-685), so a lost
# peer reaches a chat as a closed engine connection, before the deployment manager stops anything. httpx raises
# RemoteProtocolError for a server that dies mid-body, ReadError for a reset or a connection closed under the read.
CONNECTION_LOST_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError)
_HTTP_STATUS = re.compile(r"\bHTTP (\d{3})\b")  # adapters name the engine's status: "llama-server HTTP 400: ..."
RESUME_KEY = "resume_from"  # top-level chat body key; never forwarded to the engine
# Reserved RunRecord.sampling keys (schemas.py has no field for them): the chat's messages hash, which resume
# compares; a flag for an output_text that stops short of the text the client was sent; and a flag for a reply with
# reasoning text or tool calls, which output_text does not hold as a continuable assistant content.
RESUME_CHAIN_MAX = 64  # interrupted runs one resume may continue, counting earlier resumes of the same chat
# Engines verified to continue a final assistant message without sending it again. llama-server b11770:
# tools/server/server-common.cpp:1431-1438 turns a trailing assistant message into continue_final_message with
# add_generation_prompt off (--prefill-assistant is the default, common/common.h:666); common/chat.cpp:1734-1737
# starts the parse at that prefill so it is not streamed again unless the body says "echo": true
# (server-common.cpp:1491), which a resume strips. vLLM, EXO and AirLLM are not verified.
PREFILL_ENGINES = frozenset({"llama_rpc"})


@dataclass(frozen=True)
class Resume:
    """A chat that continues an interrupted run (its "resume_from").

    partial_text is all the client already has: the output_text of every run in the resumed_from chain, oldest first.
    It goes to the engine as a final assistant message that llama-server continues, so only new text comes back. The
    engine reads that text again as prompt: the new run's prompt_tokens include it, its completion_tokens (engine
    counts) cover only the new tokens, and max_tokens applies to the new tokens alone. The original run's sampling
    settings are reused, but with temperature > 0 the continuation is a new sample, not the text the cut request
    would have produced; a seed does not change that, since the sampler state is not carried over. Greedy decoding
    can also differ when the resumed run lands on another deployment, quantization or kernel.
    """

    run_id: str
    messages_sha256: str  # of the client's messages, without the prefill; every run of the chain stores the same
    partial_text: str
    sampling: dict  # the original run's sampling settings, reserved keys removed


class ChatInterrupted(DeploymentError):
    """A non-streamed chat cut by a stop or a lost peer. `body` is the error object a stream would end with."""

    def __init__(self, body: dict) -> None:
        super().__init__(502, "interrupted", body["error"]["message"])
        self.body = body


@dataclass
class StreamStats:
    started: float = field(default_factory=time.perf_counter)
    started_at: datetime = field(default_factory=utcnow)  # wall clock at request start, before the upstream call
    ttft_ms: float | None = None
    text: list[str] = field(default_factory=list)
    text_chars: int = 0  # characters kept in `text`; the client still gets everything, the record stops here
    truncated: bool = False  # the client got text the record does not keep
    non_content: bool = False  # the reply carried reasoning text or tool calls
    finish_reason: str | None = None
    usage: dict | None = None
    timings: dict | None = None

    def observe(self, payload: dict) -> None:
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if delta.get("reasoning_content") or delta.get("tool_calls") or delta.get("function_call"):
                self.non_content = True
            if piece:
                if self.ttft_ms is None:
                    self.ttft_ms = (time.perf_counter() - self.started) * 1000
                if len(piece) > RECORD_TEXT_MAX_CHARS - self.text_chars:
                    self.truncated = True
                if self.text_chars < RECORD_TEXT_MAX_CHARS:
                    self.text.append(piece[:RECORD_TEXT_MAX_CHARS - self.text_chars])
                    self.text_chars += len(self.text[-1])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if payload.get("usage"):
            self.usage = payload["usage"]
        if payload.get("timings"):
            self.timings = payload["timings"]

    def restart(self) -> None:
        """The engine was busy and the request is sent again: time it from the attempt that is served."""
        self.started, self.started_at = time.perf_counter(), utcnow()


def _timings(raw: dict | None) -> EngineTimings | None:
    if not raw:
        return None
    return EngineTimings(**{k: raw.get(k) for k in EngineTimings.model_fields})


class InferenceSlots:
    """One deployment's inference slots. Waiters are served strictly in arrival order, so an exclusive waiter keeps
    later chats off the slots that free up (no new chat starts while it waits) and is never starved by them."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.used = 0
        self._waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    @property
    def busy(self) -> bool:
        return self.used > 0 or bool(self._waiters)

    async def acquire(self, n: int) -> None:
        if not self._waiters and self.used + n <= self.capacity:
            self.used += n
            return
        entry = (n, asyncio.get_running_loop().create_future())
        self._waiters.append(entry)
        try:
            await entry[1]
        except asyncio.CancelledError:  # timeout or cancel: leave the queue, or give back slots granted meanwhile
            if entry[1].done() and not entry[1].cancelled():
                self.used -= n
            elif entry in self._waiters:
                self._waiters.remove(entry)
            self._wake()
            raise

    def release(self, n: int) -> None:
        self.used -= n
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.used + self._waiters[0][0] <= self.capacity:
            n, fut = self._waiters.popleft()
            if not fut.done():  # a cancelled waiter is skipped; its task cleans up after itself
                self.used += n
                fut.set_result(None)


@dataclass
class SlotHold:
    """Slots held by one request (1) or one exclusive measurement (all). release() is idempotent."""

    slots: InferenceSlots
    n: int
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.slots.release(self.n)


class InferenceService:
    def __init__(self, store: Store, events: EventBus, deployments: DeploymentManager, models: ModelService,
                 adapter: LlamaRpcAdapter, nodes: NodeRegistry | None = None,
                 extra_adapters: Mapping[str, Any] | None = None) -> None:
        self._store, self._events, self._deps, self._models, self._adapter = (store, events, deployments, models,
                                                                              adapter)
        self._adapters: dict[str, Any] = {"llama_rpc": adapter, **(extra_adapters or {})}
        self._nodes = nodes
        self._slots: dict[str, InferenceSlots] = {}  # deployment_id -> its slots
        self.busy_wait_s = BUSY_WAIT_S
        self.busy_retry_s = BUSY_RETRY_S
        self._inflight: dict[str, str] = {}  # request_id -> deployment_id
        self._engine_info: dict = {}
        deployments.warmup = self.warmup
        deployments.interrupt = self.interrupt
        # replace() drains by holding every inference slot of the old deployment
        deployments.drain = lambda dep, timeout_s: self.acquire(dep, exclusive=True, timeout=timeout_s)

    # ------------------------------------------------------------------ records
    def _identity(self, dep: Deployment) -> tuple[dict, dict, list[dict]]:
        art = self._models.get(dep.plan.model_id)
        shape = self._gguf(art.id)
        model = {"id": art.id, "repo": art.repo, "revision": art.revision, "quantization": art.quantization,
                 "format": art.format, "files": [{"name": f.name, "sha256": f.sha256} for f in art.files],
                 "architecture": shape.architecture if shape else None,
                 "chat_template": shape.chat_template_present if shape else None,
                 "context_length": shape.context_length if shape else None,
                 "max_output_tokens": art.max_output_tokens}
        info = self._engine_info
        if dep.plan.engine == "llama_rpc":
            engine = {"engine": "llama_rpc", "release": info.get("release"), "commit": info.get("commit"),
                      "variant": info.get("variant"), "hosts": self._hosts(dep)}
        else:
            engine = {"engine": dep.plan.engine, "tier": dep.plan.tier, "hosts": self._hosts(dep),
                      **{k: v for k, v in dep.plan.engine_options.items() if k in ("version", "pin", "endpoint")}}
        if dep.placement:
            placement = [{"device": d.name, "model_bytes": d.model_bytes, "kv_bytes": d.kv_bytes,
                          "layers_est": d.layers_est} for d in dep.placement.devices]
        else:
            placement = [{"device": a.device, "node_id": a.node_id, "layers": len(a.layers)}
                         for a in dep.plan.assignments]
        return model, engine, placement

    def _gguf(self, model_id: str) -> GgufSummary | None:
        try:
            return self._models.summary(model_id)
        except Exception:  # record() runs in a finally block: a failed lookup must not lose the run record
            log.warning("GGUF summary unavailable for %s", model_id, exc_info=True)
            return None

    def _hosts(self, dep: Deployment) -> list[dict]:
        """The machines that did the work: one entry per node in the plan, identified from the node registry."""
        by_node: dict[str, list[DeviceAssignment]] = {}
        for a in dep.plan.assignments:
            by_node.setdefault(a.node_id, []).append(a)
        return [self._host(node_id, assigned) for node_id, assigned in by_node.items()]

    def _host(self, node_id: str, assigned: list[DeviceAssignment]) -> dict:
        host: dict = {"node_id": node_id, "devices": [{"device": a.device, "endpoint": a.endpoint,
                                                       "layers": len(a.layers)} for a in assigned]}
        view = self._nodes.get(node_id) if self._nodes is not None else None
        if view is None:
            return host
        inv = view.inventory
        return {**host, "display_name": view.display_name, "hostname": inv.hostname,
                "os": {"system": inv.os.system, "release": inv.os.release, "machine": inv.os.machine},
                "contributed_devices": view.contribution.devices,
                "hardware": [{"id": d.id, "name": d.name, "backend": d.backend} for d in inv.devices],
                "engine": {"release": inv.engine.release, "commit": inv.engine.commit, "variant": inv.engine.variant,
                           "accelerator": inv.engine.accelerator, "rpc_proto": inv.engine.rpc_proto}}

    def _adapter_for(self, dep: Deployment) -> Any:
        return self._adapters.get(dep.plan.engine, self._adapter)

    def set_engine_info(self, info: dict) -> None:
        self._engine_info = dict(info)

    def record(self, dep: Deployment, kind: str, status: str, body: dict, stats: StreamStats | None,
               stream: bool, error: str | None = None, completion: dict | None = None,
               benchmark_id: str | None = None, prompt_label: str | None = None,
               started_at: datetime | None = None, resume: Resume | None = None) -> RunRecord:
        model, engine, placement = self._identity(dep)
        usage = (stats.usage if stats else None) or (completion or {}).get("usage") or {}
        timings = (stats.timings if stats else None) or (completion or {}).get("timings")
        text = "".join(stats.text) if stats else None
        finish = stats.finish_reason if stats else None
        if completion and not stats:
            choice = (completion.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content")
            finish = choice.get("finish_reason")
        total = (time.perf_counter() - stats.started) * 1000 if stats else None
        truncated, non_content = (stats.truncated, stats.non_content) if stats else (False, _non_content(completion))
        run = RunRecord(
            id=uuid.uuid4().hex, attempt_id=uuid.uuid4().hex, deployment_id=dep.id, kind=kind, status=status,  # type: ignore[arg-type]
            model=model, engine=engine, placement=placement, ctx_size=dep.plan.ctx_size, stream=stream,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            ttft_ms=stats.ttft_ms if stats else None, total_ms=total, timings=_timings(timings),
            finish_reason=finish, sampling={k: body[k] for k in SAMPLING_KEYS if k in body},
            messages_sha256=_messages_digest(kind, body, resume), output_truncated=truncated,
            non_content_output=non_content,
            cache_state=_cache_state(timings), output_text=text, error=error,
            started_at=stats.started_at if stats else (started_at or utcnow()), benchmark_id=benchmark_id,
            prompt_label=prompt_label, resumed_from=resume.run_id if resume else None)
        doc = run.model_dump(mode="json")
        self._store.execute("INSERT INTO runs (id, deployment_id, kind, status, started_at, doc) VALUES (?,?,?,?,?,?)",
                            (run.id, dep.id, kind, status, run.started_at.isoformat(), dumps(doc)))
        self._store.append_jsonl("runs", doc)
        self._events.publish("run.recorded", f"{kind} run {status}", run_id=run.id, deployment_id=dep.id)
        return run

    def runs(self, limit: int = 50) -> list[RunRecord]:
        # newest first; runs that start within one clock tick keep their recording order (rowid), not an arbitrary one
        rows = self._store.query("SELECT doc FROM runs ORDER BY started_at DESC, rowid DESC LIMIT ?", (limit,))
        return [RunRecord.model_validate(json.loads(r["doc"])) for r in rows]

    def run(self, run_id: str) -> RunRecord | None:
        row = self._store.query_one("SELECT doc FROM runs WHERE id=?", (run_id,))
        return RunRecord.model_validate(json.loads(row["doc"])) if row else None

    # ------------------------------------------------------------------ resume
    def prepare_resume(self, dep: Deployment, body: dict) -> tuple[dict, Resume | None]:
        """The body the engine gets for a chat and, when the chat names "resume_from", the run it continues.

        A resume is refused (422 resume_invalid, with the reason) unless the ready engine continues a prefill without
        sending it again, the request has one choice and does not end with its own assistant message, and every run of
        the chain is an interrupted chat (not one its client left) with plain content, its whole text recorded, and a
        stored messages hash equal to this request's messages."""
        if RESUME_KEY not in body:
            return body, None
        run_id, messages = body[RESUME_KEY], body.get("messages")
        if not isinstance(run_id, str) or not run_id:
            raise _resume_invalid("'resume_from' must be the id of an interrupted chat run")
        if dep.plan.engine not in PREFILL_ENGINES:
            raise _resume_invalid(f"the ready deployment runs {dep.plan.engine}; resume is verified only on "
                                  f"{', '.join(sorted(PREFILL_ENGINES))}, which continues a prefill without "
                                  "sending it again")
        if body.get("n") not in (None, 1):
            raise _resume_invalid("resume continues one choice; this request asks for n != 1")
        if not isinstance(messages, list) or not messages:
            raise _resume_invalid("'messages' must be the interrupted chat's non-empty message list")
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            raise _resume_invalid("the messages end with an assistant message; resume adds its own prefill")
        digest = messages_sha256(messages)
        chain = self._resume_chain(run_id, digest)
        sampling = {k: v for k, v in chain[-1].sampling.items() if not k.startswith("_")}
        resume = Resume(run_id, digest, "".join(r.output_text or "" for r in chain), sampling)
        return resume_body(body, resume), resume

    def _resume_chain(self, run_id: str, digest: str) -> list[RunRecord]:
        """The run to resume and every run it already continues, oldest first; each must be resumable."""
        chain: list[RunRecord] = []
        next_id: str | None = run_id
        while next_id is not None:
            if len(chain) >= RESUME_CHAIN_MAX:
                raise _resume_invalid(f"run {run_id} continues more than {RESUME_CHAIN_MAX} interrupted runs")
            run = self.run(next_id)
            if run is None:
                raise _resume_invalid(f"no run {next_id}" + (f", which run {run_id} continues" if chain else ""))
            _check_resumable(run, digest)
            chain.append(run)
            next_id = run.resumed_from
        return chain[::-1]

    # ------------------------------------------------------------------ concurrency
    def _capacity(self, dep: Deployment) -> int:
        """Requests the engine batches at once (vLLM: --max-num-seqs); an engine without max_concurrency gets 1."""
        max_concurrency = getattr(self._adapter_for(dep), "max_concurrency", None)
        return max(1, int(max_concurrency(dep.plan))) if max_concurrency is not None else 1

    def _slots_for(self, dep: Deployment) -> InferenceSlots:
        """A new deployment gets fresh slots; idle slots of other deployments are dropped then."""
        slots = self._slots.get(dep.id)
        if slots is None:
            slots = InferenceSlots(self._capacity(dep))
            self._slots = {k: s for k, s in self._slots.items() if s.busy} | {dep.id: slots}
        return slots

    async def acquire(self, dep: Deployment, *, exclusive: bool = False, timeout: float | None = None) -> SlotHold:
        """One slot of the deployment, or all of them when exclusive; 429 when none frees up within the wait."""
        slots = self._slots_for(dep)
        n = slots.capacity if exclusive else 1
        try:
            await asyncio.wait_for(slots.acquire(n), timeout=self.busy_wait_s if timeout is None else timeout)
        except TimeoutError:
            raise DeploymentError(429, "busy", f"all {slots.capacity} inference slot(s) of the deployment are busy; "
                                  "try again shortly") from None
        hold = SlotHold(slots, n)
        if not exclusive:  # exclusive holders (drain, warmup, benchmarks) act on this deployment by design
            self._require_current(dep, hold)
        return hold

    def _require_current(self, dep: Deployment, hold: SlotHold) -> None:
        """A request that queued behind a replacement drain must not reach the engine the drain is stopping."""
        try:
            current, _ = self._deps.ready()
        except DeploymentError:
            current = None
        if current is None or current.id != dep.id:
            hold.release()
            raise DeploymentError(503, "replacing", "the model was replaced or stopped while this request waited "
                                  "for a slot", fix="send the request again")

    async def interrupt(self, dep_id: str) -> None:
        for request_id, d in list(self._inflight.items()):
            if d == dep_id:
                self._inflight.pop(request_id, None)  # marks the request as interrupted, not failed
                for adapter in self._adapters.values():  # only the owning engine knows the id; others no-op
                    await adapter.cancel(request_id)

    # ------------------------------------------------------------------ streaming
    async def _engine_events(self, dep: Deployment, handle: DeploymentHandle, request_id: str, body: dict,
                             on_retry: Callable[[], None] | None = None) -> AsyncIterator[GenerationEvent]:
        """The engine's events for one request. An engine still finishing an earlier, cancelled request answers busy
        before any output (the AirLLM worker's 429): that is retried for up to busy_wait_s, like a wait for a slot."""
        adapter = self._adapter_for(dep)
        deadline = time.monotonic() + self.busy_wait_s
        served = False
        while True:
            try:
                async with aclosing(adapter.stream(handle, InferenceRequest(request_id, body, True))) as events:
                    async for ev in events:
                        served = True
                        yield ev
                return
            except EngineAdapterError as exc:
                left = deadline - time.monotonic()
                if exc.code != "engine_busy" or served or request_id not in self._inflight:
                    raise
                if left <= 0:
                    raise EngineAdapterError("engine_busy", f"the engine is still busy with an earlier request after "
                                             f"{self.busy_wait_s:.0f}s; try again shortly") from exc
            await asyncio.sleep(min(self.busy_retry_s, left))
            if request_id not in self._inflight:  # stopped during the wait: the engine gets no further request
                raise EngineAdapterError("interrupted", INTERRUPTED)
            if on_retry is not None:
                on_retry()

    async def stream_events(self, dep: Deployment, handle: DeploymentHandle, body: dict, kind: str,
                            benchmark_id: str | None = None, prompt_label: str | None = None,
                            on_record: Callable[[RunRecord], None] | None = None,
                            resume: Resume | None = None) -> AsyncIterator[bytes]:
        """Yield raw SSE bytes; always records exactly one run, before any error event so that event can name it.
        Caller must hold an inference slot."""
        request_id = uuid.uuid4().hex
        stats = StreamStats()
        status, error, failure = "error", None, None  # failure: code of the error event sent after the record
        self._inflight[request_id] = dep.id
        try:
            # aclosing() guarantees the upstream HTTP stream is closed (so llama-server stops generating)
            # even when this generator is finalized at a `yield` after the client went away.
            async with aclosing(self._engine_events(dep, handle, request_id, body, stats.restart)) as events:
                async for ev in events:
                    if ev.payload:
                        stats.observe(ev.payload)
                    yield ev.raw
                    if ev.done:
                        break
            status = "ok" if stats.finish_reason or stats.usage else "error"
            if status == "error":
                error = NO_FINISH
        except (asyncio.CancelledError, GeneratorExit):
            status, error = "cancelled", "client disconnected or request cancelled"
            raise
        except httpx.TimeoutException as exc:
            idle = idle_budget(dep.plan, self._adapter_for(dep), config.UPSTREAM_IDLE_TIMEOUT_S)
            status, error, failure = "timeout", f"no upstream data for {idle:.0f}s: {exc}", "timeout"
        except (httpx.HTTPError, httpx.StreamError, EngineAdapterError) as exc:
            cut = _interruption(request_id not in self._inflight, kind, exc)
            status, error = ("cancelled", cut) if cut else ("error", str(exc))
            failure = "interrupted" if cut else "busy" if _engine_busy(exc) else "upstream_error"
        finally:
            self._inflight.pop(request_id, None)
            run = self.record(dep, kind, status, body, stats, stream=True, error=error, benchmark_id=benchmark_id,
                              prompt_label=prompt_label, resume=resume)
            if on_record:
                on_record(run)
        if failure == "interrupted":
            yield _sse(interrupted_error(run))
        elif failure:
            yield _sse_error(failure, error or failure)

    def chat_stream(self, body: dict) -> AsyncIterator[bytes]:
        """A streamed chat. The deployment and any resume_from are checked here, before a byte is streamed, so a bad
        resume is an HTTP 422 and not an error event inside a 200 stream."""
        dep, handle = self._deps.ready()
        forwarded, resume = self.prepare_resume(dep, body)
        return self._chat_stream(dep, handle, forwarded, resume)

    async def _chat_stream(self, dep: Deployment, handle: DeploymentHandle, body: dict,
                           resume: Resume | None) -> AsyncIterator[bytes]:
        try:
            hold = await self.acquire(dep)
        except DeploymentError as exc:
            yield _sse_error(exc.code, str(exc))
            return
        try:
            async with aclosing(self.stream_events(dep, handle, body, "chat", resume=resume)) as chunks:
                async for raw in chunks:
                    yield raw
        finally:
            hold.release()

    async def chat_complete(self, body: dict) -> tuple[int, dict]:
        """A non-streamed chat, assembled from the engine's stream: a long answer on a slow engine is not cut off by
        the per-read idle timeout, which only bounds the silence between events. One run record, as before.
        A resumed chat answers with the new text only. An interrupted one is recorded, then raises ChatInterrupted
        carrying the same error object a stream ends with."""
        dep, handle = self._deps.ready()
        body, resume = self.prepare_resume(dep, body)
        hold = await self.acquire(dep)
        request_id = uuid.uuid4().hex
        started, started_at = time.perf_counter(), utcnow()
        chunks: list[dict] = []  # filled as events arrive, so a failure keeps what came before it
        self._inflight[request_id] = dep.id
        try:
            await self._chunks(dep, handle, request_id, body, chunks)
        except (httpx.HTTPError, httpx.StreamError, EngineAdapterError) as exc:
            cut = _interruption(request_id not in self._inflight, "chat", exc)
            if cut is None:
                return self._complete_failed(dep, body, exc, started, started_at, resume)
            run = self._record_complete(dep, body, "cancelled", started, started_at, assemble_completion(chunks),
                                        cut, resume)
            raise ChatInterrupted(interrupted_error(run)) from exc
        finally:
            self._inflight.pop(request_id, None)
            hold.release()
        completion = assemble_completion(chunks)
        finished = bool(any(c.get("finish_reason") for c in completion["choices"]) or completion.get("usage"))
        self._record_complete(dep, body, "ok" if finished else "error", started, started_at, completion,
                              None if finished else NO_FINISH, resume)
        if not finished:
            raise DeploymentError(502, "upstream_error", f"engine request failed: {NO_FINISH}")
        return 200, completion

    async def _chunks(self, dep: Deployment, handle: DeploymentHandle, request_id: str, body: dict,
                      out: list[dict]) -> None:
        """Append the engine's chunk payloads to `out` until the stream is done; raises on any failure."""
        size = 0
        async with aclosing(self._engine_events(dep, handle, request_id, body)) as events:
            async for ev in events:
                size += len(ev.raw)
                if size > COMPLETION_MAX_BYTES:
                    raise EngineAdapterError("response_too_large", f"the engine's reply passed "
                                             f"{COMPLETION_MAX_BYTES // (1024 * 1024)} MiB without finishing",
                                             fix="lower max_tokens, or use stream: true for very long replies")
                if ev.payload:
                    out.append(ev.payload)
                if ev.done:
                    break

    def _record_complete(self, dep: Deployment, body: dict, status: str, started: float, started_at: datetime,
                         completion: dict | None = None, error: str | None = None,
                         resume: Resume | None = None) -> RunRecord:
        run = self.record(dep, "chat", status, body, None, stream=False, completion=completion, error=error,
                          started_at=started_at, resume=resume)
        total = (time.perf_counter() - started) * 1000
        self._store.execute("UPDATE runs SET doc=json_set(doc, '$.total_ms', ?) WHERE id=?", (total, run.id))
        return run.model_copy(update={"total_ms": total})

    def _complete_failed(self, dep: Deployment, body: dict, exc: Exception, started: float, started_at: datetime,
                         resume: Resume | None) -> tuple[int, dict]:
        """A refused or failed non-streamed chat: the engine's own HTTP status when it named one, else 429 or 502."""
        error = str(exc) or exc.__class__.__name__
        self._record_complete(dep, body, "error", started, started_at, error=error, resume=resume)
        if _engine_busy(exc):
            raise DeploymentError(429, "busy", error) from exc
        match = _HTTP_STATUS.search(error) if isinstance(exc, EngineAdapterError) else None
        if match:
            status = int(match.group(1))
            return status, {"error": {"code": status, "message": error, "type": "upstream_error"}}
        raise DeploymentError(502, "upstream_error", f"engine request failed: {error}") from exc

    async def collect(self, dep: Deployment, handle: DeploymentHandle, body: dict, kind: str,
                      benchmark_id: str | None = None, prompt_label: str | None = None) -> RunRecord:
        box: list[RunRecord] = []
        async with aclosing(self.stream_events(dep, handle, body, kind, benchmark_id, prompt_label,
                                               box.append)) as chunks:
            async for _ in chunks:
                pass
        return box[0]

    async def warmup(self, dep: Deployment, handle: DeploymentHandle) -> None:
        hold = await self.acquire(dep, exclusive=True)
        try:
            body = {"model": dep.plan.model_id, "messages": [{"role": "user", "content": "Reply with: ready"}],
                    "max_tokens": 1 if dep.plan.capacity_mode else 8, "temperature": 0}
            run = await self.collect(dep, handle, body, "warmup")
        finally:
            hold.release()
        if run.status != "ok":
            raise EngineAdapterError("warmup_failed", f"warmup request failed: {run.error}")


def _engine_busy(exc: BaseException) -> bool:
    return isinstance(exc, EngineAdapterError) and exc.code == "engine_busy"


def _interruption(stopped: bool, kind: str, exc: BaseException) -> str | None:
    """The run error of a failed request that counts as interrupted, else None. A stop of the deployment always does.
    So does a chat whose engine connection closed: the text so far is exact and a resume can continue it. A
    measurement keeps reporting the closed connection as an error."""
    if stopped:
        return INTERRUPTED
    if kind == "chat" and isinstance(exc, CONNECTION_LOST_ERRORS):
        return f"{CONNECTION_LOST}: {exc}" if str(exc) else CONNECTION_LOST
    return None


def assemble_completion(chunks: list[dict]) -> dict:
    """chat.completion.chunk payloads -> the one chat.completion body a non-streamed request returns. Pieces are
    gathered in lists and joined once: a long reply arrives as thousands of fragments, and repeated string
    concatenation would make assembly quadratic."""
    head = next((c for c in chunks if c.get("id")), {})
    parts: dict[int, _ChoiceParts] = {}
    for chunk in chunks:
        for part in chunk.get("choices") or []:
            i = part.get("index", 0)
            parts.setdefault(i, _ChoiceParts(i)).add(part)
    usage = next((c["usage"] for c in reversed(chunks) if c.get("usage")), None)
    timings = next((c["timings"] for c in reversed(chunks) if c.get("timings")), None)
    return {**{k: head[k] for k in ("id", "created", "model", "system_fingerprint") if k in head},
            "object": "chat.completion", "choices": [parts[i].whole() for i in sorted(parts)],
            **({"usage": usage} if usage else {}), **({"timings": timings} if timings else {})}


class _ChoiceParts:
    """One choice's streamed fragments; local to assemble_completion, so mutation stays contained."""

    def __init__(self, index: int) -> None:
        self.index, self.role, self.finish_reason = index, None, None
        self.text: dict[str, list[str]] = {"content": [], "reasoning_content": []}
        self.tools: dict[int, dict] = {}  # stream index -> id/type plus name and argument fragments
        self.logprobs: list = []

    def add(self, part: dict) -> None:
        delta = part.get("delta") or {}
        self.role = delta.get("role") or self.role
        for key, pieces in self.text.items():
            if delta.get(key):
                pieces.append(delta[key])
        for call in delta.get("tool_calls") or []:
            slot = self.tools.setdefault(call.get("index", 0), {"type": "function", "name": [], "arguments": []})
            slot.update({k: call[k] for k in ("id", "type") if call.get(k)})
            fn = call.get("function") or {}
            slot["name"].append(fn.get("name") or "")
            slot["arguments"].append(fn.get("arguments") or "")
        self.logprobs.extend((part.get("logprobs") or {}).get("content") or [])
        self.finish_reason = part.get("finish_reason") or self.finish_reason

    def whole(self) -> dict:
        msg: dict = {"role": self.role or "assistant", "content": "".join(self.text["content"]) or None}
        if self.text["reasoning_content"]:
            msg["reasoning_content"] = "".join(self.text["reasoning_content"])
        if self.tools:  # a non-streamed reply lists tool calls without the stream's index
            msg["tool_calls"] = [{**{k: v for k, v in t.items() if k not in ("name", "arguments")},
                                  "function": {"name": "".join(t["name"]), "arguments": "".join(t["arguments"])}}
                                 for _, t in sorted(self.tools.items())]
        out = {"index": self.index, "message": msg, "finish_reason": self.finish_reason}
        return {**out, "logprobs": {"content": self.logprobs}} if self.logprobs else out


def _cache_state(timings: dict | None) -> str:
    """'warm' when llama.cpp reused cached prompt tokens (timings.cache_n > 0), else 'cold'."""
    if not timings or timings.get("cache_n") is None:
        return "unknown"
    return "warm" if timings["cache_n"] > 0 else "cold"


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_error(code: str, message: str) -> bytes:
    return _sse({"error": {"code": code, "message": message}})


def interrupted_error(run: RunRecord) -> dict:
    """The error object an interrupted chat ends with, streamed or not: the recorded run and the exact text generated
    before the cut (for a stream, what the client was already sent), equal to the run's output_text. A record that
    kept only the first RECORD_TEXT_MAX_CHARS says so with partial_truncated, and cannot be resumed. The message is the
    run's error, which says whether the deployment was stopped or the engine connection closed."""
    error = {"code": "interrupted", "type": "upstream_error", "message": run.error or INTERRUPTED, "run_id": run.id,
             "partial_text": run.output_text or ""}
    return {"error": {**error, "partial_truncated": True} if run.output_truncated else error}


def messages_sha256(messages: list) -> str:
    """Hash of a chat's messages, independent of key order and whitespace in the request JSON. The canonical form is
    ASCII (escaped), so an unpaired surrogate the request JSON may carry cannot make the hash, or the record, fail."""
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def resume_body(body: dict, resume: Resume) -> dict:
    """What the engine gets for a resumed chat: the request without resume_from, its own sampling settings or the keys
    that change how llama-server treats the prefill (echo sends it again; an explicit continue_final_message skips the
    prefill branch and, with add_generation_prompt at its default, is refused: b11770 server-common.cpp:1428-1441);
    the original run's sampling settings; and the messages plus the partial text as a final assistant message. With
    no partial text there is nothing to continue: a plain retry."""
    dropped = {RESUME_KEY, "echo", "continue_final_message", "add_generation_prompt", *SAMPLING_KEYS}
    prefill = [{"role": "assistant", "content": resume.partial_text}] if resume.partial_text else []
    return {**{k: v for k, v in body.items() if k not in dropped}, **resume.sampling,
            "messages": [*body["messages"], *prefill]}


def _messages_digest(kind: str, body: dict, resume: Resume | None) -> str | None:
    """A chat's messages hash: for a resumed chat the client's messages, not the prefilled ones."""
    if kind != "chat":
        return None
    messages = body.get("messages")
    return resume.messages_sha256 if resume else messages_sha256(messages) if isinstance(messages, list) else None


def _non_content(completion: dict | None) -> bool:
    """A non-streamed reply's first choice holds reasoning text or tool calls (its record keeps only the content)."""
    message = (((completion or {}).get("choices") or [{}])[0].get("message")) or {}
    return bool(message.get("reasoning_content") or message.get("tool_calls"))


def _check_resumable(run: RunRecord, digest: str) -> None:
    if run.kind != "chat":
        raise _resume_invalid(f"run {run.id} is a {run.kind} run; only a chat can be resumed")
    if run.status != "cancelled":
        raise _resume_invalid(f"run {run.id} ended with status {run.status}; only an interrupted chat can be resumed")
    if not (run.error or "").startswith(INTERRUPTED_PREFIX):  # the client left: it was never told what reached it
        raise _resume_invalid(f"run {run.id} was cancelled by its client ({run.error}), not interrupted, so the text "
                              "the client got is not known exactly; only a chat that ended with an 'interrupted' "
                              "error can be resumed")
    if run.non_content_output:
        raise _resume_invalid(f"run {run.id} replied with reasoning text or tool calls, which an assistant content "
                              "prefill cannot continue exactly")
    stored = run.messages_sha256
    if stored is None:
        raise _resume_invalid(f"run {run.id} has no stored messages hash (recorded before resume support)")
    if stored != digest:
        raise _resume_invalid(f"the messages differ from the ones run {run.id} answered; resend them unchanged")
    if run.output_truncated:
        raise _resume_invalid(f"run {run.id} kept only the first {RECORD_TEXT_MAX_CHARS} characters of its text, "
                              "so the exact partial text is not known")


def _resume_invalid(reason: str) -> DeploymentError:
    return DeploymentError(422, "resume_invalid", reason,
                           fix="send the interrupted chat's messages unchanged with 'resume_from' set to the run_id "
                               "of its 'interrupted' error")
