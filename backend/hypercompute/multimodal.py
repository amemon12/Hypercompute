"""Image input for /v1/chat/completions: OpenAI-style image_url parts, checked at the gateway before llama-server.

llama-server at b11770 (tools/server/server-common.cpp, handle_media) also downloads http(s) image URLs from the host,
reads raw base64 and file:// paths, and decodes audio and video parts. The gateway lets through only base64 data
URLs of PNG, JPEG or GIF images (what stb_image decodes; WebP needs an ffmpeg the hosts do not ship), in user
messages, at most MAX_IMAGES per request, MAX_IMAGE_BYTES and MAX_IMAGE_PIXELS (read from the header) each. It never
rewrites the body: a request that passes reaches the engine as the client sent it, and a text-only request is not
inspected beyond its content types.
"""

from __future__ import annotations

import base64
import binascii
import struct
from typing import Any

from .units import MiB

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * MiB  # decoded; the 16 MiB chat body limit holds one such image as base64 with room to spare
# stb_image decodes the whole image (3 bytes a pixel) before llama.cpp resizes it, so a small file that declares a
# huge size would allocate gigabytes on the host: the size in the header is capped (a 48 MP phone photo passes).
MAX_IMAGE_PIXELS = 64_000_000
IMAGE_MAGIC: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}
JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
SUPPORTED = "PNG, JPEG or GIF"


class ImageInputError(ValueError):
    def __init__(self, code: str, message: str, fix: str | None = None, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.fix, self.status = code, fix, status


def count_images(messages: list[Any]) -> int:
    """The number of images in a chat's messages, after checking every content part; ImageInputError otherwise."""
    images = 0
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ImageInputError("bad_request", f"messages[{i}] must be an object")
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ImageInputError("bad_request", f"messages[{i}].content must be a string or a list of parts")
        for j, part in enumerate(content):
            images += _check_part(message.get("role"), part, f"messages[{i}].content[{j}]")
    if images > MAX_IMAGES:
        raise ImageInputError("too_many_images", f"{images} images in one request; at most {MAX_IMAGES} are allowed",
                              fix="send fewer images per request")
    return images


def _check_part(role: object, part: object, where: str) -> int:
    """1 for a valid image part, 0 for a text part; anything else raises."""
    kind = part.get("type") if isinstance(part, dict) else None
    if kind == "text" and isinstance(part.get("text"), str):  # type: ignore[union-attr]
        return 0
    if kind != "image_url":
        raise ImageInputError("unsupported_content", f"{where} must be a text or image_url part",
                              fix="send text parts and image_url parts with base64 data URLs only")
    if role != "user":
        raise ImageInputError("bad_request", f"{where}: images go in user messages only")
    image = part.get("image_url")  # type: ignore[union-attr]
    if not isinstance(image, dict) or not isinstance(image.get("url"), str):
        raise ImageInputError("bad_request", f"{where}.image_url must be an object with a url")
    check_data_url(image["url"], where)
    return 1


def check_data_url(url: str, where: str = "image") -> bytes:
    """The decoded image of a base64 data URL whose declared type is supported and matches the bytes."""
    if not url.startswith("data:"):
        raise ImageInputError("image_url_not_data", f"{where}: only base64 data URLs are accepted, not "
                              f"{url[:24]!r}", fix="embed the image as data:image/png;base64,<data>")
    head, _, data = url.partition(",")
    mime, _, encoding = head[len("data:"):].partition(";")
    if encoding != "base64":
        raise ImageInputError("image_not_base64", f"{where}: the data URL must be base64 encoded")
    mime = mime.lower()
    if mime not in IMAGE_MAGIC:
        raise ImageInputError("image_type_unsupported", f"{where}: {mime or 'no type'} is not supported; "
                              f"send {SUPPORTED}", fix="convert the image to PNG or JPEG")
    if len(data) // 4 * 3 > MAX_IMAGE_BYTES + 3:  # base64 length bounds the decoded size: refuse before decoding
        raise _too_large(where)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ImageInputError("image_not_base64", f"{where}: the image data is not valid base64",
                              fix="send standard base64 without line breaks") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise _too_large(where)
    if not raw.startswith(IMAGE_MAGIC[mime]):
        raise ImageInputError("image_type_mismatch", f"{where}: the bytes are not a {mime} image",
                              fix="declare the image's real type in the data URL")
    size = image_size(mime, raw)
    if size is None:
        raise ImageInputError("image_unreadable", f"{where}: no {mime} header with the image size")
    if size[0] * size[1] > MAX_IMAGE_PIXELS:
        raise ImageInputError("image_too_large", f"{where} is {size[0]}x{size[1]} pixels; at most "
                              f"{MAX_IMAGE_PIXELS // 1_000_000} megapixels are allowed", status=413,
                              fix="downscale the image")
    return raw


def image_size(mime: str, raw: bytes) -> tuple[int, int] | None:
    """(width, height) from the header: PNG IHDR, GIF logical screen, JPEG frame header; None when absent."""
    if mime == "image/png":
        return struct.unpack(">II", raw[16:24]) if len(raw) >= 24 and raw[12:16] == b"IHDR" else None
    if mime == "image/gif":
        return struct.unpack("<HH", raw[6:10]) if len(raw) >= 10 else None
    return _jpeg_size(raw)


def _jpeg_size(raw: bytes) -> tuple[int, int] | None:
    """Walk the JPEG segments to the first SOFn frame header (its height and width follow the precision byte)."""
    i = 2
    while i + 4 <= len(raw):
        if raw[i] != 0xFF:
            return None
        marker = raw[i + 1]
        if marker == 0xFF:  # fill byte before a marker
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # markers without a length
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image or start of scan before any frame header
            return None
        length = int.from_bytes(raw[i + 2:i + 4], "big")
        if marker in JPEG_SOF:
            if i + 9 > len(raw):
                return None
            height, width = struct.unpack(">HH", raw[i + 5:i + 9])
            return width, height
        if length < 2:
            return None
        i += 2 + length
    return None


def _too_large(where: str) -> ImageInputError:
    return ImageInputError("image_too_large", f"{where} is over {MAX_IMAGE_BYTES // MiB} MiB", status=413,
                           fix="downscale or recompress the image")


def data_url(raw: bytes) -> str:
    """A base64 data URL for image bytes, typed from the bytes themselves (the ask-image CLI)."""
    mime = next((m for m, magic in IMAGE_MAGIC.items() if raw.startswith(magic)), None)
    if mime is None:
        raise ImageInputError("image_type_unsupported", f"the file is not a {SUPPORTED} image")
    url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    check_data_url(url)
    return url


def accepts_images(dep: Any, handle: Any) -> bool:
    """Whether the ready deployment's engine was started with a vision projector (llama-server --mmproj)."""
    return dep.plan.engine == "llama_rpc" and handle is not None and "--mmproj" in handle.args
