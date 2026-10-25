"""Interrupted chats and their resume: the cut is recorded before the error that names it, the error carries the
exact text already generated, and a resume continues that text as an assistant prefill and sends only new text.
The fake engine continues a trailing assistant message the way llama-server b11770 does, without sending it again."""

from __future__ import annotations

import asyncio
import json
from itertools import accumulate

import httpx
import pytest
from hypercompute.adapters.base import GenerationEvent
from hypercompute.schemas import Deployment
from hypercompute.services import streaming
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.streaming import (
    INTERRUPTED,
    ChatInterrupted,
    InferenceService,
    messages_sha256,
)
from hypercompute.sse import SSEParser
from test_app_http import OWNER, make_coord, open_client
from test_inference_slots import DONE, deployment, service

TOKENS = ("The ", "quick ", "brown ", "fox ", "jumps.")
WHOLE = "".join(TOKENS)
ENDS = list(accumulate(len(t) for t in TOKENS))
PROMPT_TOKENS = 11
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say the sentence."}]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "seed": 5, "max_tokens": 64}
BODY = {"model": "m", "messages": MESSAGES, **SAMPLING}
# What httpx raises when llama-server dies mid-body (measured against a socket server that aborts or closes)
DROPPED = "peer closed connection without sending complete message body (incomplete chunked read)"
TOOL_DELTA = {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                              "function": {"name": "f", "arguments": "{\"a\""}}]}


def chunk(payload: dict) -> GenerationEvent:
    data = json.dumps(payload)
    return GenerationEvent(raw=f"data: {data}\n\n".encode(), data=data, payload=payload)


def remaining(prefill: str) -> list[str]:
    assert WHOLE.startswith(prefill) and len(prefill) in (0, *ENDS), "a prefill is text this engine already wrote"
    return [t for t, end in zip(TOKENS, ENDS, strict=True) if end > len(prefill)]


class ScriptEngine:
    """Always writes WHOLE, one token per event. A trailing assistant message is continued, not repeated
    (b11770 tools/server/server-common.cpp:1431, common/chat.cpp:1734). Each request takes the next entry of `cuts`:
    after that many new tokens it waits until cancelled, the way a stop ends it. With drop=True it closes the
    connection there instead, the way llama-server does when it aborts on a lost RPC peer: nobody calls interrupt()."""

    def __init__(self, cuts: list[int | None], drop: bool = False) -> None:
        self.cuts = list(cuts)
        self.drop = drop
        self.bodies: list[dict] = []
        self.generated: list[str] = []  # every token produced, across requests
        self.waiting = asyncio.Event()
        self._gates: dict[str, asyncio.Event] = {}

    async def stream(self, handle, req):
        self.bodies.append(req.body)
        cut = self.cuts.pop(0) if self.cuts else None
        last = req.body["messages"][-1]
        rest = remaining(last["content"] if last["role"] == "assistant" else "")
        for i, token in enumerate(rest):
            if i == cut:
                if self.drop:
                    raise httpx.RemoteProtocolError(DROPPED)
                gate = self._gates[req.request_id] = asyncio.Event()
                self.waiting.set()
                await gate.wait()
                raise httpx.ReadError("upstream connection closed")
            self.generated.append(token)
            yield chunk({"choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]})
        usage = {"prompt_tokens": PROMPT_TOKENS + len(TOKENS) - len(rest), "completion_tokens": len(rest)}
        yield chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
        yield DONE

    async def cancel(self, request_id):
        gate = self._gates.pop(request_id, None)
        if gate is not None:
            gate.set()


class PartsEngine(ScriptEngine):
    """Sends the given deltas, then holds the request until it is cancelled."""

    def __init__(self, deltas: list[dict]) -> None:
        super().__init__([])
        self.deltas = deltas

    async def stream(self, handle, req):
        self.bodies.append(req.body)
        for delta in self.deltas:
            yield chunk({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        gate = self._gates[req.request_id] = asyncio.Event()
        self.waiting.set()
        await gate.wait()
        raise httpx.ReadError("upstream connection closed")


async def read_stream(inf: InferenceService, gen) -> tuple[list[dict], dict[str, bool]]:
    """What the client sees: the parsed events, and for each error naming a run whether that run was already
    recorded when the error arrived."""
    parser, seen, recorded = SSEParser(), [], {}
    async for raw in gen:
        for ev in parser.feed(raw):
            if ev.data == "[DONE]":
                continue
            payload = json.loads(ev.data)
            run_id = (payload.get("error") or {}).get("run_id")
            if run_id:
                recorded[run_id] = inf.run(run_id) is not None
            seen.append(payload)
    return seen, recorded


def displayed(seen: list[dict]) -> str:
    return "".join((c.get("delta") or {}).get("content") or "" for p in seen for c in p.get("choices") or [])


async def cut_stream(inf: InferenceService, engine: ScriptEngine, dep: Deployment,
                     body: dict) -> tuple[list[dict], dict[str, bool]]:
    task = asyncio.create_task(read_stream(inf, inf.chat_stream(dict(body))))
    await asyncio.wait_for(engine.waiting.wait(), 2)
    engine.waiting.clear()
    await inf.interrupt(dep.id)
    return await asyncio.wait_for(task, 2)


def setup(store, events, cuts: list[int | None],
          drop: bool = False) -> tuple[InferenceService, ScriptEngine, Deployment]:
    engine, dep = ScriptEngine(cuts, drop), deployment()
    inf, _ = service(store, events, engine, dep)
    return inf, engine, dep


# --------------------------------------------------------------------------- interruption
@pytest.mark.anyio
async def test_an_interrupted_stream_ends_with_its_recorded_run_and_the_exact_text_sent(store, events):
    inf, engine, dep = setup(store, events, [2])
    seen, recorded = await cut_stream(inf, engine, dep, BODY)
    run = inf.runs()[0]
    err = seen[-1]["error"]
    assert err["code"] == "interrupted" and err["message"] == INTERRUPTED and err["run_id"] == run.id
    assert recorded == {run.id: True}  # the run existed before its error event was sent
    assert err["partial_text"] == displayed(seen) == run.output_text == "The quick "
    assert run.status == "cancelled" and run.error == INTERRUPTED and "partial_truncated" not in err
    assert run.sampling == SAMPLING and run.messages_sha256 == messages_sha256(MESSAGES)


@pytest.mark.anyio
async def test_an_interrupted_non_streamed_chat_gives_the_same_error_object(store, events):
    inf, engine, dep = setup(store, events, [2])
    task = asyncio.create_task(inf.chat_complete(dict(BODY)))
    await asyncio.wait_for(engine.waiting.wait(), 2)
    await inf.interrupt(dep.id)
    with pytest.raises(ChatInterrupted) as err:
        await asyncio.wait_for(task, 2)
    run = inf.runs()[0]
    assert err.value.status == 502 and err.value.body == {"error": {
        "code": "interrupted", "type": "upstream_error", "message": INTERRUPTED, "run_id": run.id,
        "partial_text": "The quick "}}
    assert (run.status, run.stream, run.output_text) == ("cancelled", False, "The quick ")


@pytest.mark.anyio
async def test_a_record_that_kept_only_part_of_the_text_says_so_and_cannot_be_resumed(store, events, monkeypatch):
    monkeypatch.setattr(streaming, "RECORD_TEXT_MAX_CHARS", 6)
    inf, engine, dep = setup(store, events, [2])
    seen, _ = await cut_stream(inf, engine, dep, BODY)
    err = seen[-1]["error"]
    assert displayed(seen) == "The quick " and err["partial_text"] == "The qu" and err["partial_truncated"] is True
    with pytest.raises(DeploymentError) as refused:
        inf.chat_stream({**BODY, "resume_from": err["run_id"]})
    assert refused.value.status == 422 and "first 6 characters" in str(refused.value)


@pytest.mark.anyio
async def test_a_dropped_engine_connection_is_an_interruption_that_can_be_resumed(store, events):
    """llama-server b11770 aborts when an RPC peer is lost (ggml-rpc.cpp:31, 681-685), so the stream sees the
    connection close before the deployment manager stops anything. That is the node-loss case the contract names."""
    inf, engine, _ = setup(store, events, [2, None], drop=True)
    first, recorded = await read_stream(inf, inf.chat_stream(dict(BODY)))
    err, run = first[-1]["error"], inf.runs()[0]
    assert (err["code"], err["run_id"], recorded) == ("interrupted", run.id, {run.id: True})
    assert err["partial_text"] == displayed(first) == run.output_text == "The quick "
    assert run.status == "cancelled" and run.error == err["message"]
    assert run.error.startswith("interrupted:") and DROPPED in run.error
    second, _ = await read_stream(inf, inf.chat_stream({**BODY, "resume_from": run.id}))
    assert displayed(first) + displayed(second) == "".join(engine.generated) == WHOLE
    assert inf.runs()[0].resumed_from == run.id


@pytest.mark.anyio
async def test_a_dropped_connection_cuts_a_non_streamed_chat_the_same_way(store, events):
    inf, _, _ = setup(store, events, [3], drop=True)
    with pytest.raises(ChatInterrupted) as err:
        await inf.chat_complete(dict(BODY))
    run, body = inf.runs()[0], err.value.body["error"]
    assert (body["code"], body["run_id"], body["partial_text"]) == ("interrupted", run.id, "The quick brown ")
    assert (run.status, run.error) == ("cancelled", body["message"]) and DROPPED in str(err.value)


@pytest.mark.anyio
async def test_a_dropped_connection_still_fails_a_measurement(store, events):
    inf, _, dep = setup(store, events, [2], drop=True)
    run = await inf.collect(dep, None, dict(BODY), "benchmark")
    assert (run.status, run.error) == ("error", DROPPED)


@pytest.mark.anyio
async def test_messages_with_a_lone_surrogate_are_hashed_and_recorded(store, events):
    """JSON allows an unpaired \\ud83d escape (half an emoji); hashing it must not break the run record."""
    inf, _, _ = setup(store, events, [None])
    messages = [{"role": "user", "content": "half an emoji: \ud83d"}]
    seen, _ = await read_stream(inf, inf.chat_stream({**BODY, "messages": messages}))
    run = inf.runs()[0]
    assert displayed(seen) == WHOLE and run.status == "ok"
    assert run.messages_sha256 == messages_sha256(messages)


# --------------------------------------------------------------------------- refusals
@pytest.mark.anyio
async def test_resume_refuses_a_missing_finished_foreign_or_different_chat(store, events):
    inf, engine, dep = setup(store, events, [2])
    cut_id = (await cut_stream(inf, engine, dep, BODY))[0][-1]["error"]["run_id"]
    await read_stream(inf, inf.chat_stream(dict(BODY)))
    ok_id = inf.runs()[0].id
    bench_id = (await inf.collect(dep, None, dict(BODY), "benchmark")).id
    other = [MESSAGES[0], {"role": "user", "content": "Say another sentence."}]
    cases = [({**BODY, "resume_from": "nope"}, "no run nope"),
             ({**BODY, "resume_from": 7}, "must be the id"),
             ({**BODY, "resume_from": ok_id}, "status ok"),
             ({**BODY, "resume_from": bench_id}, "benchmark run"),
             ({**BODY, "messages": other, "resume_from": cut_id}, "messages differ"),
             ({**BODY, "messages": [*MESSAGES, {"role": "assistant", "content": "The "}], "resume_from": cut_id},
              "end with an assistant message"),
             ({**BODY, "n": 2, "resume_from": cut_id}, "one choice")]
    calls = len(engine.bodies)
    for body, reason in cases:
        with pytest.raises(DeploymentError) as streamed:
            inf.chat_stream(body)
        with pytest.raises(DeploymentError) as complete:
            await inf.chat_complete(body)
        for err in (streamed.value, complete.value):
            assert (err.status, err.code) == (422, "resume_invalid") and reason in str(err), body
    assert len(engine.bodies) == calls  # nothing refused reached the engine


@pytest.mark.anyio
async def test_resume_needs_an_engine_verified_to_continue_a_prefill(store, events):
    engine, dep = ScriptEngine([2]), deployment()
    inf, deps = service(store, events, engine, dep)
    cut_id = (await cut_stream(inf, engine, dep, BODY))[0][-1]["error"]["run_id"]
    deps.current = dep.model_copy(update={"plan": dep.plan.model_copy(update={"engine": "vllm"})})
    with pytest.raises(DeploymentError) as err:
        inf.chat_stream({**BODY, "resume_from": cut_id})
    assert err.value.status == 422 and "verified only on llama_rpc" in str(err.value)


@pytest.mark.anyio
async def test_resume_refuses_a_chat_its_client_left(store, events):
    """A client that disconnected was never told which text reached it: its record can hold a piece it never got,
    and a prefill of that piece would leave a gap."""
    inf, _, _ = setup(store, events, [None])
    gen = inf.chat_stream(dict(BODY))
    await gen.__anext__()
    await gen.aclose()
    run = inf.runs()[0]
    assert (run.status, run.output_text) == ("cancelled", "The ")
    with pytest.raises(DeploymentError) as err:
        inf.chat_stream({**BODY, "resume_from": run.id})
    assert err.value.status == 422 and "not interrupted" in str(err.value)


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "non_streamed"])
@pytest.mark.parametrize("deltas", [[{"reasoning_content": "Think. "}, {"content": "The "}], [TOOL_DELTA]],
                         ids=["reasoning", "tool_call"])
async def test_resume_refuses_a_reply_with_reasoning_or_tool_calls(store, events, deltas, stream):
    """output_text joins reasoning into the text (streamed) or drops it (non-streamed) and never holds tool calls,
    so a content prefill of it is not the reply that was cut."""
    engine, dep = PartsEngine(deltas), deployment()
    inf, _ = service(store, events, engine, dep)
    if stream:
        run_id = (await cut_stream(inf, engine, dep, BODY))[0][-1]["error"]["run_id"]
    else:
        task = asyncio.create_task(inf.chat_complete(dict(BODY)))
        await asyncio.wait_for(engine.waiting.wait(), 2)
        await inf.interrupt(dep.id)
        with pytest.raises(ChatInterrupted) as cut:
            await asyncio.wait_for(task, 2)
        run_id = cut.value.body["error"]["run_id"]
    with pytest.raises(DeploymentError) as err:
        inf.chat_stream({**BODY, "resume_from": run_id})
    assert err.value.status == 422 and "reasoning text or tool calls" in str(err.value)


# --------------------------------------------------------------------------- resume
@pytest.mark.anyio
async def test_resume_sends_the_conversation_plus_the_partial_as_a_prefill_with_the_original_sampling(store, events):
    inf, engine, dep = setup(store, events, [2])
    cut_id = (await cut_stream(inf, engine, dep, BODY))[0][-1]["error"]["run_id"]
    # echo would send the prefill again; an explicit continue_final_message skips llama-server's prefill branch and,
    # with add_generation_prompt left at its default, is refused (b11770 server-common.cpp:1428-1441)
    controls = {"echo": True, "continue_final_message": "reasoning_content", "add_generation_prompt": True}
    resume = {**BODY, "temperature": 0.0, "top_k": 3, **controls, "resume_from": cut_id}
    await read_stream(inf, inf.chat_stream(resume))
    sent = engine.bodies[-1]
    assert sent["messages"] == [*MESSAGES, {"role": "assistant", "content": "The quick "}]
    assert not {"resume_from", "top_k", *controls} & set(sent)  # never forwarded; top_k was not in the original
    assert {k: sent[k] for k in SAMPLING} == SAMPLING and sent["model"] == "m"


@pytest.mark.anyio
async def test_a_resumed_stream_sends_only_new_text_and_records_it(store, events):
    inf, engine, dep = setup(store, events, [2])
    first_seen, _ = await cut_stream(inf, engine, dep, BODY)
    cut_id = first_seen[-1]["error"]["run_id"]
    seen, _ = await read_stream(inf, inf.chat_stream({**BODY, "resume_from": cut_id}))
    partial, new = first_seen[-1]["error"]["partial_text"], displayed(seen)
    assert new == "brown fox jumps." and partial + new == "".join(engine.generated) == WHOLE  # nothing twice
    run, first = inf.runs()[0], inf.run(cut_id)
    assert (run.status, run.resumed_from, run.output_text) == ("ok", cut_id, new)
    assert (run.prompt_tokens, run.completion_tokens) == (PROMPT_TOKENS + 2, 3)  # the engine's counts
    assert run.sampling == first.sampling  # same settings and messages hash


@pytest.mark.anyio
async def test_a_resume_cut_again_is_resumed_from_the_whole_chain(store, events):
    inf, engine, dep = setup(store, events, [1, 1, None])
    first, _ = await cut_stream(inf, engine, dep, BODY)
    second, _ = await cut_stream(inf, engine, dep, {**BODY, "resume_from": first[-1]["error"]["run_id"]})
    second_id = second[-1]["error"]["run_id"]
    assert second[-1]["error"]["partial_text"] == displayed(second) == "quick "  # only what this stream sent
    third, _ = await read_stream(inf, inf.chat_stream({**BODY, "resume_from": second_id}))
    assert engine.bodies[-1]["messages"][-1] == {"role": "assistant", "content": "The quick "}
    assert displayed(first) + displayed(second) + displayed(third) == WHOLE
    assert [r.resumed_from for r in inf.runs()] == [second_id, first[-1]["error"]["run_id"], None]


@pytest.mark.anyio
async def test_resume_with_nothing_sent_yet_is_a_plain_retry(store, events):
    inf, engine, dep = setup(store, events, [0])
    first, _ = await cut_stream(inf, engine, dep, BODY)
    assert first[-1]["error"]["partial_text"] == ""
    seen, _ = await read_stream(inf, inf.chat_stream({**BODY, "resume_from": first[-1]["error"]["run_id"]}))
    assert engine.bodies[-1]["messages"] == MESSAGES and displayed(seen) == WHOLE


@pytest.mark.anyio
async def test_a_non_streamed_resume_answers_with_the_new_text_only(store, events):
    inf, engine, dep = setup(store, events, [3])
    cut_id = (await cut_stream(inf, engine, dep, BODY))[0][-1]["error"]["run_id"]
    status, body = await inf.chat_complete({**BODY, "resume_from": cut_id})
    run = inf.runs()[0]
    assert status == 200 and body["choices"][0]["message"]["content"] == "fox jumps."
    assert body["usage"] == {"prompt_tokens": PROMPT_TOKENS + 3, "completion_tokens": 2}
    assert (run.stream, run.resumed_from, run.output_text) == (False, cut_id, "fox jumps.")


# --------------------------------------------------------------------------- HTTP route
def test_the_chat_route_refuses_a_bad_streamed_resume_before_any_byte(store, events, tmp_path, monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    monkeypatch.setattr(coord.deployments, "ready", lambda: (deployment(), None))
    with open_client(coord) as c:
        r = c.post("/v1/chat/completions", json={**BODY, "stream": True, "resume_from": "nope"},
                   headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 422 and r.json()["code"] == "resume_invalid" and "no run nope" in r.json()["detail"]


def test_the_chat_route_returns_an_interrupted_non_streamed_chat_as_its_error_object(store, events, tmp_path,
                                                                                    monkeypatch):
    coord = make_coord(store, events, tmp_path, monkeypatch)
    monkeypatch.setattr(coord.deployments, "ready", lambda: (deployment(), None))
    error = {"error": {"code": "interrupted", "type": "upstream_error", "message": INTERRUPTED, "run_id": "r1",
                       "partial_text": "The quick "}}

    async def interrupted(body):
        raise ChatInterrupted(error)

    monkeypatch.setattr(coord.inference, "chat_complete", interrupted)
    with open_client(coord) as c:
        r = c.post("/v1/chat/completions", json=BODY, headers={"X-Hypercompute-Owner": OWNER})
    assert r.status_code == 502 and r.json() == error
