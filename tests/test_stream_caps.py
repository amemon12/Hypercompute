"""Bounded memory and CPU for replies the gateway collects: an engine (or a friend's EXO endpoint) may never stop."""

import time

import pytest
from hypercompute.adapters.base import GenerationEvent
from hypercompute.services import streaming
from hypercompute.services.deployments import DeploymentError
from hypercompute.services.streaming import assemble_completion
from test_inference_slots import DONE, deployment, drain, event, service

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class EndlessAdapter:
    """Streams content forever, the way a broken or hostile engine endpoint could."""

    def __init__(self) -> None:
        self.sent = 0

    async def stream(self, handle, req):
        piece = "x" * 1024
        payload = {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
        while True:
            self.sent += 1
            yield GenerationEvent(raw=b"data: " + piece.encode() + b"\n\n", data=piece, payload=payload)

    async def cancel(self, request_id):
        return None


@pytest.mark.anyio
async def test_a_non_streamed_reply_that_never_ends_is_cut_off_at_the_byte_cap(store, events, monkeypatch):
    monkeypatch.setattr(streaming, "COMPLETION_MAX_BYTES", 256 * 1024)
    adapter = EndlessAdapter()
    inf, _ = service(store, events, adapter, deployment())
    with pytest.raises(DeploymentError) as err:
        await inf.chat_complete(dict(BODY))
    assert err.value.status == 502 and "passed 0 MiB without finishing" in str(err.value)
    assert adapter.sent < 300  # stopped near the cap, not after the whole endless stream
    assert [r.status for r in inf.runs()] == ["error"]


@pytest.mark.anyio
async def test_a_streamed_reply_keeps_flowing_but_its_record_stores_bounded_text(store, events, monkeypatch):
    monkeypatch.setattr(streaming, "RECORD_TEXT_MAX_CHARS", 5)

    class Short:
        async def stream(self, handle, req):
            for piece in ("abc", "def", "ghi"):
                yield event(piece)
            yield event("", "stop")
            yield DONE

        async def cancel(self, request_id):
            return None

    inf, _ = service(store, events, Short(), deployment())
    out = await drain(inf.chat_stream(dict(BODY)))
    assert b"[DONE]" in out
    assert inf.runs()[0].output_text == "abcde"  # the client got every piece; the record keeps the first 5 chars


def test_assembly_is_linear_in_the_number_of_fragments():
    """200,000 eight-character fragments: repeated concatenation took 21.9 s on this laptop, list-and-join 0.1 s."""
    n = 200_000
    chunks = [{"id": "c", "choices": [{"index": 0, "delta": {"content": "xxxxxxxx"}, "finish_reason": None}]}] * n
    chunks.append({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    started = time.perf_counter()
    body = assemble_completion(chunks)
    assert time.perf_counter() - started < 5
    assert body["choices"][0]["message"]["content"] == "xxxxxxxx" * n
    assert body["choices"][0]["finish_reason"] == "stop"
