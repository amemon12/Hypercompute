"""Interrupted downloads surface failures and resume without reserving already-present bytes twice."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import httpx
import pytest
from hypercompute import engine
from hypercompute.schemas import ModelArtifact, ModelFile
from hypercompute.services import models
from hypercompute.services.events import EventBus
from hypercompute.services.models import ModelError, ModelService
from hypercompute.units import GiB, MiB


class InterruptedBody(httpx.SyncByteStream):
    def __init__(self, data, error):
        self.data, self.error = data, error

    def __iter__(self):
        yield self.data
        raise self.error


def use_transport(monkeypatch, handler):
    client_type = httpx.Client
    monkeypatch.setattr(engine.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(handler)))


def model_service(tmp_path, payloads):
    files = [ModelFile(name=name, url=f"https://example.invalid/{name}",
                       sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
             for name, payload in payloads.items()]
    artifact = ModelArtifact(id="tiny-download", display_name="Tiny download", family="test", params_b=0.001,
                             quantization="bf16", format="hf", engines=["airllm"], repo="test/model",
                             revision="pinned", license="mit", files=files)
    events = EventBus()
    service = ModelService(events, tmp_path / "models")
    service._artifacts = {artifact.id: artifact}
    service.model_dir(artifact.id).mkdir(parents=True)
    return service, artifact, events


@pytest.mark.parametrize("error", [httpx.ReadTimeout(""), httpx.RemoteProtocolError("peer closed connection")])
def test_interrupted_stream_preserves_partial_file_and_resumes_verified_download(tmp_path, monkeypatch, error):
    payload = b"a" * (2 * MiB) + b"complete"
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=InterruptedBody(payload[:MiB], error))
        return httpx.Response(206, content=payload[MiB:],
                              headers={"Content-Range": f"bytes {MiB}-{len(payload) - 1}/{len(payload)}"})

    use_transport(monkeypatch, handler)
    dest = tmp_path / "weights.bin"
    part = tmp_path / "weights.bin.part"
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(engine.EngineError, match=type(error).__name__):
        engine.download_verified("https://example.invalid/weights.bin", dest, digest, len(payload))
    assert part.read_bytes() == payload[:MiB]
    assert not dest.exists()
    engine.download_verified("https://example.invalid/weights.bin", dest, digest, len(payload))
    assert "range" not in requests[0].headers
    assert requests[1].headers["range"] == f"bytes={MiB}-"
    assert dest.read_bytes() == payload
    assert not part.exists()


def test_model_failure_is_visible_in_view_and_events_then_retry_clears_error(tmp_path, monkeypatch):
    payload = b"b" * (2 * MiB)
    service, artifact, events = model_service(tmp_path, {"weights.bin": payload})
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=InterruptedBody(payload[:MiB], httpx.ReadTimeout("")))
        return httpx.Response(206, content=payload[MiB:],
                              headers={"Content-Range": f"bytes {MiB}-{len(payload) - 1}/{len(payload)}"})

    use_transport(monkeypatch, handler)
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * GiB))
    with pytest.raises(ModelError, match="ReadTimeout"):
        asyncio.run(service.download(artifact.id))
    failed = service.view(artifact.id)
    assert failed.status == "error" and "ReadTimeout" in failed.error
    assert failed.downloaded_bytes == MiB
    failure_events = [event for event in events.since(0) if event.data.get("error")]
    assert len(failure_events) == 1 and "ReadTimeout" in failure_events[0].message
    asyncio.run(service.download(artifact.id))
    complete = service.view(artifact.id)
    assert complete.status == "downloaded" and complete.error is None
    assert complete.downloaded_bytes == len(payload) and complete.verified_sha256
    assert requests[1].headers["range"] == f"bytes={MiB}-"
    assert events.since(0)[-1].data["done"]


def test_disk_preflight_counts_only_missing_bytes_after_verified_shard_and_partial(tmp_path, monkeypatch):
    payloads = {"first.bin": b"first" * MiB, "second.bin": b"second" * MiB}
    service, artifact, _ = model_service(tmp_path, payloads)
    directory = service.model_dir(artifact.id)
    (directory / "first.bin").write_bytes(payloads["first.bin"])
    (directory / "second.bin.part").write_bytes(payloads["second.bin"][:MiB])
    remaining = len(payloads["second.bin"]) - MiB
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * GiB + remaining))
    requests = []

    def handler(request):
        requests.append(request)
        total = len(payloads["second.bin"])
        return httpx.Response(206, content=payloads["second.bin"][MiB:],
                              headers={"Content-Range": f"bytes {MiB}-{total - 1}/{total}"})

    use_transport(monkeypatch, handler)
    asyncio.run(service.download(artifact.id))
    assert service.view(artifact.id).status == "downloaded"
    assert len(requests) == 1 and requests[0].url.path == "/second.bin"
    assert requests[0].headers["range"] == f"bytes={MiB}-"
    assert all((directory / name).read_bytes() == payload for name, payload in payloads.items())


def test_complete_verified_part_can_finalize_with_only_headroom_free(tmp_path, monkeypatch):
    payload = b"ready" * MiB
    service, artifact, _ = model_service(tmp_path, {"weights.bin": payload})
    directory = service.model_dir(artifact.id)
    (directory / "weights.bin.part").write_bytes(payload)
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * GiB))

    def unexpected_request(request):
        raise AssertionError("a complete verified partial file must not make another network request")

    use_transport(monkeypatch, unexpected_request)
    asyncio.run(service.download(artifact.id))
    assert service.view(artifact.id).status == "downloaded"
    assert (directory / "weights.bin").read_bytes() == payload
    assert not (directory / "weights.bin.part").exists()


def test_corrupt_full_size_destination_does_not_reduce_disk_requirement(tmp_path, monkeypatch):
    payload = b"correct" * MiB
    service, artifact, _ = model_service(tmp_path, {"weights.bin": payload})
    (service.model_dir(artifact.id) / "weights.bin").write_bytes(b"x" * len(payload))
    monkeypatch.setattr(models.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * GiB))
    with pytest.raises(ModelError, match="not enough disk"):
        asyncio.run(service.download(artifact.id))
    assert service.view(artifact.id).status == "error"


@pytest.mark.parametrize("content_range", [None, "invalid", "bytes 0-7/8", "bytes 3-7/9",
                                          "bytes 3-8/8", "bytes 3-2/8"])
def test_invalid_resume_range_keeps_original_partial(tmp_path, monkeypatch, content_range):
    payload = b"abcdefgh"
    dest = tmp_path / "weights.bin"
    part = tmp_path / "weights.bin.part"
    part.write_bytes(payload[:3])
    headers = {"Content-Range": content_range} if content_range else {}
    use_transport(monkeypatch, lambda request: httpx.Response(206, content=payload, headers=headers))
    with pytest.raises(engine.EngineError, match="Content-Range"):
        engine.download_verified("https://example.invalid/weights.bin", dest,
                                 hashlib.sha256(payload).hexdigest(), len(payload))
    assert part.read_bytes() == payload[:3]
    assert not dest.exists()


@pytest.mark.parametrize("resuming", [False, True])
def test_oversized_response_cannot_write_beyond_expected_size(tmp_path, monkeypatch, resuming):
    payload = b"abcdefgh"
    dest = tmp_path / "weights.bin"
    part = tmp_path / "weights.bin.part"
    if resuming:
        part.write_bytes(payload[:3])
    headers = {"Content-Range": "bytes 3-7/8"} if resuming else {}
    body = payload[3:] + b"extra" if resuming else payload + b"extra"
    use_transport(monkeypatch, lambda request: httpx.Response(206 if resuming else 200,
                                                             content=body, headers=headers))
    with pytest.raises(engine.EngineError, match="exceeds"):
        engine.download_verified("https://example.invalid/weights.bin", dest,
                                 hashlib.sha256(payload).hexdigest(), len(payload))
    assert part.read_bytes() == (payload[:3] if resuming else b"")
    assert not dest.exists()


def test_short_valid_range_preserves_prefix_for_next_resume(tmp_path, monkeypatch):
    payload = b"abcdefgh"
    dest = tmp_path / "weights.bin"
    part = tmp_path / "weights.bin.part"
    part.write_bytes(payload[:3])
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(206, content=payload[3:6], headers={"Content-Range": "bytes 3-5/8"})
        return httpx.Response(206, content=payload[6:], headers={"Content-Range": "bytes 6-7/8"})

    use_transport(monkeypatch, handler)
    with pytest.raises(engine.EngineError, match="incomplete"):
        engine.download_verified("https://example.invalid/weights.bin", dest,
                                 hashlib.sha256(payload).hexdigest(), len(payload))
    assert part.read_bytes() == payload[:6]
    assert not dest.exists()
    engine.download_verified("https://example.invalid/weights.bin", dest,
                             hashlib.sha256(payload).hexdigest(), len(payload))
    assert requests[1].headers["range"] == "bytes=6-"
    assert dest.read_bytes() == payload


def test_manifest_size_mismatch_is_visible_even_when_checksum_matches(tmp_path, monkeypatch):
    payload = b"abcdefgh"
    dest = tmp_path / "weights.bin"
    use_transport(monkeypatch, lambda request: httpx.Response(200, content=payload))
    with pytest.raises(engine.EngineError, match="incomplete"):
        engine.download_verified("https://example.invalid/weights.bin", dest,
                                 hashlib.sha256(payload).hexdigest(), len(payload) + 1)
    assert not dest.exists()
    assert (tmp_path / "weights.bin.part").read_bytes() == payload
