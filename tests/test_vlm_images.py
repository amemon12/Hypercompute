"""Image input at the gateway: OpenAI-style image_url parts are checked at the boundary and passed through unchanged.

llama-server at b11770 would also fetch http(s) URLs, read raw base64 and decode audio or video parts
(tools/server/server-common.cpp), so the gateway lets only base64 data URLs of PNG, JPEG or GIF images through.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from hypercompute import image_cli
from hypercompute.app import BODY_LIMIT_CHAT, create_app
from hypercompute.multimodal import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGES,
    ImageInputError,
    accepts_images,
    count_images,
    data_url,
    image_size,
)
from test_app_http import OWNER, client, make_coord  # noqa: F401 (fixture)


def png(width: int, height: int) -> bytes:  # signature and an 8-bit RGB IHDR; nothing past the header is read here
    ihdr = struct.pack(">I4sII5s", 13, b"IHDR", width, height, b"\x08\x02\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + ihdr + b"\x00" * 32


def jpeg(width: int, height: int) -> bytes:  # SOI, a JFIF APP0 segment, then a baseline SOF0 frame header
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app0 + sof0 + b"\x00" * 32


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 32


PNG, JPEG, GIF = png(64, 48), jpeg(640, 480), gif(32, 32)
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 64


def url_of(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def image_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def user(*parts: dict) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": "What is in this picture?"}, *parts]}


# ----------------------------------------------------------------------------- boundary checks


def test_text_only_messages_have_no_images():
    assert count_images([{"role": "user", "content": "hi"}]) == 0
    assert count_images([{"role": "system", "content": "be brief"}, user()]) == 0
    assert count_images([{"role": "assistant", "content": None, "tool_calls": []}]) == 0


@pytest.mark.parametrize(("payload", "mime"), [(PNG, "image/png"), (JPEG, "image/jpeg"), (GIF, "image/gif")])
def test_png_jpeg_and_gif_data_urls_are_accepted(payload, mime):
    assert count_images([user(image_part(url_of(payload, mime)))]) == 1


def test_several_images_are_counted():
    msgs = [user(image_part(url_of(PNG, "image/png")), image_part(url_of(JPEG, "image/jpeg")))]
    assert count_images(msgs) == 2


@pytest.mark.parametrize(("url", "code"), [
    ("https://example.com/cat.png", "image_url_not_data"),  # llama-server would download it from the host
    ("file://cat.png", "image_url_not_data"),
    (base64.b64encode(PNG).decode(), "image_url_not_data"),  # raw base64, no data: scheme
    ("data:image/png,rawbytes", "image_not_base64"),
    (url_of(WEBP, "image/webp"), "image_type_unsupported"),  # stb_image cannot decode it; ffmpeg is not shipped
    ("data:image/svg+xml;base64,PHN2Zz4=", "image_type_unsupported"),
    ("data:image/png;base64,***", "image_not_base64"),
    ("data:image/png;base64,iVBORw0KGgo=\n", "image_not_base64"),  # llama.cpp would silently stop at the newline
    (url_of(JPEG, "image/png"), "image_type_mismatch"),
    (url_of(b"hello world", "image/png"), "image_type_mismatch"),
])
def test_bad_image_urls_are_refused(url, code):
    with pytest.raises(ImageInputError) as err:
        count_images([user(image_part(url))])
    assert err.value.code == code and err.value.status == 400


def test_an_image_over_the_size_cap_is_refused_before_decoding():
    big = "data:image/png;base64," + "A" * (MAX_IMAGE_BYTES // 3 * 4 + 8)
    with pytest.raises(ImageInputError) as err:
        count_images([user(image_part(big))])
    assert err.value.code == "image_too_large" and err.value.status == 413


def test_the_header_gives_the_size_of_each_format():
    assert image_size("image/png", PNG) == (64, 48)
    assert image_size("image/jpeg", JPEG) == (640, 480)
    assert image_size("image/gif", GIF) == (32, 32)
    real = Path(__file__).parents[1] / "docs" / "screenshots" / "08-overview.png"
    assert all(n > 0 for n in image_size("image/png", real.read_bytes()))


@pytest.mark.parametrize(("payload", "mime"), [
    (png(10_000, 10_000), "image/png"),  # a few bytes that would make stb_image allocate 300 MB
    (jpeg(65_535, 65_535), "image/jpeg"),
    (gif(65_535, 65_535), "image/gif"),
])
def test_an_image_declaring_too_many_pixels_is_refused(payload, mime):
    with pytest.raises(ImageInputError) as err:
        count_images([user(image_part(url_of(payload, mime)))])
    assert err.value.code == "image_too_large" and err.value.status == 413 and "pixels" in str(err.value)


def test_a_48_megapixel_photo_passes():
    assert 8064 * 6048 <= MAX_IMAGE_PIXELS
    assert count_images([user(image_part(url_of(jpeg(8064, 6048), "image/jpeg")))]) == 1


@pytest.mark.parametrize(("payload", "mime"), [
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png"),  # no IHDR chunk
    (b"\xff\xd8\xff\xda" + b"\x00" * 64, "image/jpeg"),  # a scan before any frame header
    (b"\xff\xd8\xff\xe0\x00\x01" + b"\x00" * 8, "image/jpeg"),  # a segment length below 2
    (b"GIF89a\x01", "image/gif"),  # cut inside the logical screen
])
def test_an_image_without_a_readable_size_is_refused(payload, mime):
    with pytest.raises(ImageInputError) as err:
        count_images([user(image_part(url_of(payload, mime)))])
    assert err.value.code == "image_unreadable"


def test_too_many_images_are_refused():
    parts = [image_part(url_of(PNG, "image/png"))] * (MAX_IMAGES + 1)
    with pytest.raises(ImageInputError) as err:
        count_images([user(*parts)])
    assert err.value.code == "too_many_images"


@pytest.mark.parametrize("message", [
    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]},
    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}]},
    {"role": "user", "content": ["just a string part"]},
    {"role": "user", "content": [{"type": "image_url", "image_url": "data:image/png;base64,AAAA"}]},
    {"role": "user", "content": [{"type": "image_url", "image_url": {}}]},
    {"role": "user", "content": [{"type": "text", "text": 5}]},
    {"role": "user", "content": 7},
    {"role": "assistant", "content": [image_part(url_of(PNG, "image/png"))]},
    "not a message",
])
def test_malformed_or_non_image_media_parts_are_refused(message):
    with pytest.raises(ImageInputError) as err:
        count_images([message])
    assert err.value.status == 400


def test_accepts_images_only_from_a_llama_server_started_with_a_projector():
    llama = SimpleNamespace(plan=SimpleNamespace(engine="llama_rpc"))
    assert accepts_images(llama, SimpleNamespace(args=["-m", "m.gguf", "--mmproj", "p.gguf", "-mmdev", "CUDA0"]))
    assert not accepts_images(llama, SimpleNamespace(args=["-m", "m.gguf"]))
    assert not accepts_images(llama, None)
    vllm = SimpleNamespace(plan=SimpleNamespace(engine="vllm"))
    assert not accepts_images(vllm, SimpleNamespace(args=["--mmproj"]))


def test_data_url_reads_the_type_from_the_bytes():
    assert data_url(JPEG) == url_of(JPEG, "image/jpeg")
    with pytest.raises(ImageInputError) as err:
        data_url(WEBP)
    assert err.value.code == "image_type_unsupported"


def test_the_chat_body_limit_carries_a_full_size_image():
    encoded = len(url_of(b"x" * MAX_IMAGE_BYTES, "image/png"))
    assert encoded + 64 * 1024 < BODY_LIMIT_CHAT


def test_only_loopback_clients_may_send_a_large_chat_body(store, events, tmp_path, monkeypatch):
    big = json.dumps({"messages": [{"role": "user", "content": "y" * (6 * 1024 * 1024)}]})
    headers = {"X-Hypercompute-Owner": OWNER, "Content-Type": "application/json"}
    app = create_app(make_coord(store, events, tmp_path, monkeypatch))
    with TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50123)) as local:
        assert local.post("/v1/chat/completions", content=big, headers=headers).status_code == 503  # no model yet
    with TestClient(app, base_url="http://127.0.0.1:8000", client=("10.0.0.9", 50123)) as lan:
        r = lan.post("/v1/chat/completions", content=big, headers=headers)
    assert r.status_code == 413 and r.json()["code"] == "body_too_large"  # buffered before auth: kept at 4 MiB


# ----------------------------------------------------------------------------- the route


def serve(client, monkeypatch, args):  # noqa: F811 (fixture)
    coord = client.app.state.coord
    dep = SimpleNamespace(id="dep1", plan=SimpleNamespace(model_id="vlm", engine="llama_rpc"))
    handle = SimpleNamespace(args=args) if args is not None else None
    monkeypatch.setattr(coord.deployments, "ready", lambda: (dep, handle))
    seen: list[dict] = []

    async def complete(body):
        seen.append(body)
        return 200, {"choices": [{"message": {"role": "assistant", "content": "a cat"}}]}

    monkeypatch.setattr(coord.inference, "chat_complete", complete)
    return seen


def post(client, body):  # noqa: F811 (fixture)
    return client.post("/v1/chat/completions", content=json.dumps(body), headers={
        "X-Hypercompute-Owner": OWNER, "Content-Type": "application/json"})


def test_an_image_request_reaches_the_engine_unchanged(client, monkeypatch):  # noqa: F811
    seen = serve(client, monkeypatch, ["-m", "m.gguf", "--mmproj", "p.gguf", "-mmdev", "CUDA0"])
    body = {"messages": [user(image_part(url_of(PNG, "image/png")))], "max_tokens": 32}
    r = post(client, body)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "a cat"
    assert seen == [body]


def test_a_text_only_request_is_forwarded_as_before_even_without_a_projector(client, monkeypatch):  # noqa: F811
    seen = serve(client, monkeypatch, None)  # the route never looks at the handle for text
    body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    assert post(client, body).status_code == 200 and seen == [body]


def test_an_image_for_a_model_without_a_projector_is_a_400(client, monkeypatch):  # noqa: F811
    seen = serve(client, monkeypatch, ["-m", "m.gguf"])
    r = post(client, {"messages": [user(image_part(url_of(PNG, "image/png")))]})
    assert r.status_code == 400 and r.json()["code"] == "vision_unavailable" and "vlm" in r.json()["detail"]
    assert r.json()["fix"] and seen == []


def test_a_remote_image_url_never_reaches_the_engine(client, monkeypatch):  # noqa: F811
    seen = serve(client, monkeypatch, ["--mmproj", "p.gguf"])
    r = post(client, {"messages": [user(image_part("http://169.254.169.254/latest/meta-data"))]})
    assert r.status_code == 400 and r.json()["code"] == "image_url_not_data" and seen == []


def test_streamed_image_requests_are_checked_too(client, monkeypatch):  # noqa: F811
    serve(client, monkeypatch, ["-m", "m.gguf"])
    r = post(client, {"stream": True, "messages": [user(image_part(url_of(PNG, "image/png")))]})
    assert r.status_code == 400 and r.json()["code"] == "vision_unavailable"


# ----------------------------------------------------------------------------- ask-image CLI


def test_cli_sends_one_image_and_prints_the_answer_and_timings(tmp_path: Path, monkeypatch, capsys):
    photo = tmp_path / "cat.png"
    photo.write_bytes(PNG)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        return httpx.Response(200, json={
            "model": "gemma-3-27b-it-q4_k_m", "choices": [{"message": {"role": "assistant", "content": "A cat."}}],
            "usage": {"prompt_tokens": 280, "completion_tokens": 4},
            "timings": {"prompt_n": 280, "prompt_ms": 1400.0, "prompt_per_second": 200.0, "predicted_n": 4,
                        "predicted_ms": 400.0, "predicted_per_second": 10.0}})

    real = httpx.Client
    monkeypatch.setattr(image_cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    rc = image_cli.main([str(photo), "What animal is this?", "--max-tokens", "16"])
    out = capsys.readouterr().out
    assert rc == 0 and "A cat." in out and "gemma-3-27b-it-q4_k_m" in out
    assert "280 prompt tokens" in out and "10.0 tok/s" in out
    chat = json.loads(sent[-1].content)
    assert sent[-1].headers["X-Hypercompute-Owner"] == "tok" and chat["max_tokens"] == 16
    parts = chat["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "What animal is this?"}
    assert parts[1] == image_part(url_of(PNG, "image/png"))


def test_cli_refuses_an_unsupported_file_before_sending(tmp_path: Path, capsys):
    photo = tmp_path / "cat.webp"
    photo.write_bytes(WEBP)
    assert image_cli.main([str(photo), "What is this?"]) == 2
    assert "PNG, JPEG or GIF" in capsys.readouterr().out


def test_cli_shows_the_gateway_error(tmp_path: Path, monkeypatch, capsys):
    photo = tmp_path / "cat.png"
    photo.write_bytes(PNG)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        return httpx.Response(400, json={"code": "vision_unavailable", "detail": "no projector", "fix": "redeploy"})

    real = httpx.Client
    monkeypatch.setattr(image_cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert image_cli.main([str(photo), "What is this?"]) == 2
    out = capsys.readouterr().out
    assert "no projector" in out and "fix:   redeploy" in out
