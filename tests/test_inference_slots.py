"""Per-deployment inference slots: chats share the engine's batch, warmup/benchmarks/tuner hold it alone."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from hypercompute.adapters.base import CompletionResult, GenerationEvent
from hypercompute.adapters.llama_rpc import LlamaRpcAdapter
from hypercompute.schemas import (
    BenchmarkRequest,
    Deployment,
    DeploymentState,
    ModelArtifact,
    ModelFile,
    Plan,
    PlanStrategy,
    utcnow,
)
from hypercompute.services.benchmarks import BenchmarkService
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.streaming import InferenceService

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
DONE = GenerationEvent(raw=b"data: [DONE]\n\n", data="[DONE]", payload=None, done=True)


class Models:
    def get(self, model_id):
        return ModelArtifact(id=model_id, display_name="m", family="qwen2", params_b=0.5, quantization="Q4_K_M",
                             repo="r", revision="x", license="apache-2.0",
                             files=[ModelFile(name="m.gguf", url="https://x", sha256="0" * 64, size_bytes=1)])

    def summary(self, model_id):
        return None


def event(text: str, finish: str | None = None) -> GenerationEvent:
    payload = {"choices": [{"delta": {"content": text}, "finish_reason": finish, "index": 0}]}
    if finish:
        payload["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
    return GenerationEvent(raw=b"data: x\n\n", data="x", payload=payload)


class GateAdapter:
    """Streams one token, then holds the request open until finish() or cancel(). No max_concurrency: 1 slot."""

    def __init__(self) -> None:
        self.open: dict[str, asyncio.Event] = {}  # request id -> gate, in arrival order
        self.cancelled: set[str] = set()
        self.started = self.peak = 0
        self.auto = False  # True: requests finish without waiting for finish()

    async def stream(self, handle, req):
        gate = self.open[req.request_id] = asyncio.Event()
        self.started += 1
        self.peak = max(self.peak, len(self.open))
        try:
            yield event("a")
            if not self.auto:
                await gate.wait()
            if req.request_id in self.cancelled:
                raise httpx.ReadError("upstream connection closed")
            yield event("b", "stop")
            yield DONE
        finally:
            self.open.pop(req.request_id, None)

    async def complete(self, handle, req):
        gate = self.open[req.request_id] = asyncio.Event()
        self.started += 1
        try:
            await gate.wait()
            return CompletionResult(200, {"choices": [{"message": {"content": "ab"}, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
        finally:
            self.open.pop(req.request_id, None)

    def finish(self, n: int | None = None) -> None:
        for gate in list(self.open.values())[:n]:
            gate.set()

    async def cancel(self, request_id):
        if request_id in self.open:
            self.cancelled.add(request_id)
            self.open[request_id].set()


class BatchedAdapter(GateAdapter):
    """Like the vLLM adapter: as many concurrent requests as the plan's --max-num-seqs."""

    def max_concurrency(self, plan: Plan) -> int:
        return plan.engine_options["max_num_seqs"]


def deployment(dep_id: str = "d1", seqs: int = 4) -> Deployment:
    plan = Plan(id=f"p-{dep_id}", model_id="m", strategy=PlanStrategy.auto, ctx_size=4096, feasible=True,
                fit_label="estimated_fit", heuristic=True, assignments=[], device_list=["CUDA0"], tensor_split=[1],
                rpc_endpoints=[], cpu_resident_bytes=0, reasons=[], warnings=[], candidates=[], created_at=utcnow(),
                engine_options={"max_num_seqs": seqs})
    return Deployment(id=dep_id, plan=plan, state=DeploymentState.ready, history=[], created_at=utcnow())


def service(store, events, adapter, dep: Deployment, busy_wait_s: float = 5.0):
    deps = SimpleNamespace(current=dep)
    deps.ready = lambda: (deps.current, None)
    inf = InferenceService(store, events, deps, Models(), adapter)
    inf.busy_wait_s = busy_wait_s
    return inf, deps


async def drain(gen) -> bytes:
    return b"".join([chunk async for chunk in gen])


def chat(inf: InferenceService) -> asyncio.Task:
    return asyncio.create_task(drain(inf.chat_stream(dict(BODY))))


async def until(cond, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "condition not reached"
        await asyncio.sleep(0.005)


@pytest.mark.anyio
async def test_capacity_one_keeps_one_request_at_a_time_and_429_after_the_wait(store, events):
    adapter = GateAdapter()
    inf, _ = service(store, events, adapter, deployment(), busy_wait_s=0.05)
    first = chat(inf)
    await until(lambda: len(adapter.open) == 1)
    assert b'"code": "busy"' in await drain(inf.chat_stream(dict(BODY)))
    with pytest.raises(DeploymentError) as err:
        await inf.chat_complete(dict(BODY))
    assert (err.value.status, err.value.code) == (429, "busy")
    assert adapter.started == 1
    adapter.finish()
    await asyncio.wait_for(first, 2)
    assert [r.status for r in inf.runs()] == ["ok"]  # refused requests leave no run record
    second = asyncio.create_task(inf.chat_complete(dict(BODY)))  # the slot is free again
    await until(lambda: len(adapter.open) == 1)
    adapter.finish()
    assert (await asyncio.wait_for(second, 2))[0] == 200
    assert len(inf.runs()) == 2 and adapter.peak == 1


@pytest.mark.anyio
async def test_capacity_four_runs_four_chats_at_once_and_the_fifth_waits(store, events):
    adapter = BatchedAdapter()
    inf, _ = service(store, events, adapter, deployment(seqs=4))
    tasks = [chat(inf) for _ in range(5)]
    await until(lambda: len(adapter.open) == 4)
    await asyncio.sleep(0.05)
    assert adapter.started == 4 and not any(t.done() for t in tasks)
    adapter.finish(1)
    await until(lambda: adapter.started == 5)  # the fifth takes the slot the first one gave back
    assert sum(t.done() for t in tasks) == 1
    adapter.auto = True
    adapter.finish()
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    runs = inf.runs()
    assert len(runs) == 5 and {r.status for r in runs} == {"ok"} and adapter.peak == 4


@pytest.mark.anyio
async def test_exclusive_waits_for_every_slot_and_blocks_new_chats(store, events):
    adapter = BatchedAdapter()
    dep = deployment(seqs=4)
    inf, _ = service(store, events, adapter, dep)
    chats = [chat(inf) for _ in range(2)]
    await until(lambda: len(adapter.open) == 2)
    exclusive = asyncio.create_task(inf.acquire(dep, exclusive=True, timeout=5))
    await asyncio.sleep(0.02)
    late = chat(inf)
    await asyncio.sleep(0.05)
    assert not exclusive.done() and adapter.started == 2  # two slots are free, but the late chat queues behind
    adapter.finish()
    hold = await asyncio.wait_for(exclusive, 2)
    await asyncio.wait_for(asyncio.gather(*chats), 2)
    await asyncio.sleep(0.05)
    assert adapter.started == 2 and not late.done()  # the exclusive holder has the engine alone
    hold.release()
    hold.release()  # idempotent: a second release must not free slots it no longer holds
    await until(lambda: adapter.started == 3)
    adapter.finish()
    await asyncio.wait_for(late, 2)
    assert len(inf.runs()) == 3


@pytest.mark.anyio
async def test_timed_out_exclusive_waiter_leaves_the_queue(store, events):
    adapter = BatchedAdapter()
    dep = deployment(seqs=2)
    inf, _ = service(store, events, adapter, dep)
    first = chat(inf)
    await until(lambda: len(adapter.open) == 1)
    with pytest.raises(DeploymentError) as err:
        await inf.acquire(dep, exclusive=True, timeout=0.05)
    assert err.value.status == 429
    second = chat(inf)  # would queue forever behind a leaked exclusive waiter
    await until(lambda: len(adapter.open) == 2)
    adapter.finish()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    hold = await inf.acquire(dep, exclusive=True, timeout=0.5)  # every slot is back
    hold.release()


@pytest.mark.anyio
async def test_interrupt_cancels_every_inflight_request_of_the_deployment(store, events):
    adapter = BatchedAdapter()
    dep = deployment(seqs=4)
    inf, _ = service(store, events, adapter, dep)
    tasks = [chat(inf) for _ in range(4)]
    await until(lambda: len(adapter.open) == 4)
    await inf.interrupt(dep.id)
    outs = await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert all(b'"code": "interrupted"' in out for out in outs) and len(adapter.cancelled) == 4
    runs = inf.runs()
    assert len(runs) == 4 and {r.status for r in runs} == {"cancelled"}  # exactly one record per request
    hold = await inf.acquire(dep, exclusive=True, timeout=0.5)  # and every slot was given back
    hold.release()


@pytest.mark.anyio
async def test_new_deployment_gets_fresh_slots_sized_by_its_own_plan(store, events):
    adapter = BatchedAdapter()
    inf, deps = service(store, events, adapter, deployment("a", seqs=1), busy_wait_s=0.05)
    old = chat(inf)
    await until(lambda: len(adapter.open) == 1)
    deps.current = deployment("b", seqs=2)
    new = [chat(inf) for _ in range(2)]
    await until(lambda: len(adapter.open) == 3)  # a's only slot is still held; b starts with two free ones
    assert b'"code": "busy"' in await drain(inf.chat_stream(dict(BODY)))  # b's capacity is its own (2)
    adapter.finish()
    await asyncio.wait_for(asyncio.gather(old, *new), 2)
    assert sorted(r.deployment_id for r in inf.runs()) == ["a", "b", "b"]


@pytest.mark.anyio
async def test_benchmark_holds_every_slot_so_no_chat_overlaps_it(store, events):
    adapter = BatchedAdapter()
    inf, deps = service(store, events, adapter, deployment(seqs=4))
    first = chat(inf)
    await until(lambda: len(adapter.open) == 1)
    bench = BenchmarkService(store, events, deps, inf)
    result = bench.start(BenchmarkRequest(prompt_set="short", repetitions=1, warmup=0, max_tokens=8))
    await asyncio.sleep(0.05)
    late = chat(inf)
    await asyncio.sleep(0.05)
    assert adapter.started == 1  # the benchmark waits for the chat; the late chat waits behind the benchmark
    adapter.auto = True
    adapter.finish()
    await until(lambda: bench.get(result.id).status != "running")
    await asyncio.wait_for(asyncio.gather(first, late), 2)
    assert bench.get(result.id).status == "done" and adapter.peak == 1
    assert [r.kind for r in reversed(inf.runs())] == ["chat", "benchmark", "chat"]


def test_runs_that_start_on_one_clock_tick_keep_their_recording_order(store, events):
    """Windows' wall clock can give back-to-back runs the same started_at: the newest recorded still comes first."""
    inf, _ = service(store, events, GateAdapter(), deployment())
    tick = utcnow()
    for kind in ("chat", "benchmark", "chat", "warmup"):
        inf.record(deployment(), kind, "ok", dict(BODY), None, stream=False, started_at=tick)
    assert [r.kind for r in inf.runs()] == ["warmup", "chat", "benchmark", "chat"]
    assert [r.kind for r in inf.runs(2)] == ["warmup", "chat"]


@pytest.mark.anyio
async def test_a_chat_queued_behind_a_replacement_drain_never_reaches_the_old_engine(store, events):
    adapter = GateAdapter()
    dep = deployment()
    inf, deps = service(store, events, adapter, dep)
    first = chat(inf)
    await until(lambda: len(adapter.open) == 1)
    queued = chat(inf)  # waits for the single slot
    await asyncio.sleep(0.05)
    deps.current = deployment("d2")  # the replacement became the ready deployment meanwhile
    adapter.finish()
    await asyncio.wait_for(first, 2)
    out = await asyncio.wait_for(queued, 2)
    assert b'"code": "replacing"' in out and adapter.started == 1
    hold = await inf.acquire(dep, exclusive=True, timeout=0.5)  # the refused request gave its slot back
    hold.release()


class LlamaSlotsAdapter(GateAdapter):
    """GateAdapter with the llama.cpp adapter's own max_concurrency (plan.engine_options["parallel"], -np)."""

    max_concurrency = LlamaRpcAdapter.max_concurrency


def llama_deployment(dep_id: str = "d1", parallel: int | None = None) -> Deployment:
    dep = deployment(dep_id)
    options = {} if parallel is None else {"parallel": parallel}
    return dep.model_copy(update={"plan": dep.plan.model_copy(update={"engine_options": options})})


@pytest.mark.anyio
async def test_llama_parallel_admits_n(store, events):
    adapter = LlamaSlotsAdapter()
    dep = llama_deployment(parallel=4)
    inf, _ = service(store, events, adapter, dep, busy_wait_s=0.05)
    tasks = [chat(inf) for _ in range(4)]
    await until(lambda: len(adapter.open) == 4)
    with pytest.raises(DeploymentError) as err:  # a fifth gets no slot: acquire and chat_complete answer 429
        await inf.acquire(dep, timeout=0.05)
    assert (err.value.status, err.value.code) == (429, "busy")
    with pytest.raises(DeploymentError) as err:
        await inf.chat_complete(dict(BODY))
    assert err.value.status == 429 and adapter.started == 4
    adapter.finish(1)
    await until(lambda: sum(t.done() for t in tasks) == 1)
    fifth = asyncio.create_task(inf.chat_complete(dict(BODY)))  # takes the slot the first one gave back
    await until(lambda: adapter.started == 5)
    adapter.auto = True
    adapter.finish()
    await asyncio.wait_for(asyncio.gather(*tasks, fifth), 2)
    assert adapter.peak == 4


@pytest.mark.anyio
async def test_llama_default_plan_admits_one(store, events):
    adapter = LlamaSlotsAdapter()
    inf, _ = service(store, events, adapter, llama_deployment(), busy_wait_s=0.05)
    first = chat(inf)
    await until(lambda: len(adapter.open) == 1)
    assert b'"code": "busy"' in await drain(inf.chat_stream(dict(BODY)))  # a streamed chat: the SSE busy event
    with pytest.raises(DeploymentError) as err:
        await inf.chat_complete(dict(BODY))
    assert (err.value.status, err.value.code) == (429, "busy") and adapter.started == 1
    adapter.finish()
    await asyncio.wait_for(first, 2)
