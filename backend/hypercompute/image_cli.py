"""`hypercompute ask-image IMAGE QUESTION`: one image and a question for the ready model on the local gateway.

Prints the answer, then the model and llama-server's own timings (its prompt tokens include the image's tokens) and
the wall time of the whole request. Like `hypercompute capacity`, it takes the owner token from the loopback-only
/api/owner/bootstrap, so no secret is passed on the command line. The image is checked here with the gateway's own
rules (multimodal.py) before anything is sent.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import httpx

from . import config
from .agent.join import say
from .multimodal import MAX_IMAGE_BYTES, SUPPORTED, ImageInputError, data_url
from .units import MiB

ASK_TIMEOUT_S = 1800.0  # a pooled 72B model reads a few hundred image tokens before its first word


def _fail(message: str, fix: str | None = None, code: int = 2) -> int:
    say(f"error: {message}")
    if fix:
        say(f"fix:   {fix}")
    return code


def request_body(url: str, question: str, max_tokens: int) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": question},
                                                      {"type": "image_url", "image_url": {"url": url}}]}],
            "max_tokens": max_tokens, "temperature": 0}


def summary_lines(reply: dict[str, Any], wall_s: float) -> list[str]:
    """The answer, then where the time went, from llama-server's timings when the reply carries them."""
    choice = (reply.get("choices") or [{}])[0]
    lines = [(choice.get("message") or {}).get("content") or "(no text in the reply)", ""]
    lines.append(f"model   {reply.get('model', '?')}")
    t = reply.get("timings") or {}
    if t.get("prompt_n") is not None:
        lines.append(f"prompt  {t['prompt_n']} prompt tokens in {t.get('prompt_ms', 0):.0f} ms "
                     f"({t.get('prompt_per_second', 0):.1f} tok/s)")
    if t.get("predicted_n") is not None:
        lines.append(f"decode  {t['predicted_n']} tokens in {t.get('predicted_ms', 0):.0f} ms "
                     f"({t.get('predicted_per_second', 0):.1f} tok/s)")
    lines.append(f"wall    {wall_s:.2f} s")
    return lines


def ask(image: Path, question: str, port: int, max_tokens: int, timeout_s: float = ASK_TIMEOUT_S) -> int:
    try:
        if image.stat().st_size > MAX_IMAGE_BYTES:
            return _fail(f"{image} is over {MAX_IMAGE_BYTES // MiB} MiB", "downscale or recompress the image")
        url = data_url(image.read_bytes())
    except OSError as exc:
        return _fail(f"cannot read {image}: {exc}")
    except ImageInputError as exc:
        return _fail(f"{image}: {exc}", f"send a {SUPPORTED} image")
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(10.0, read=timeout_s)) as client:
            owner = client.get(f"{base}/api/owner/bootstrap")
            owner.raise_for_status()
            started = time.perf_counter()
            r = client.post(f"{base}/v1/chat/completions", json=request_body(url, question, max_tokens),
                            headers={config.OWNER_HEADER: owner.json()["owner_token"]})
            wall = time.perf_counter() - started
            reply = r.json()
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return _fail(f"the gateway on port {port} did not answer: {exc}",
                     "start it with `uv run hypercompute host` and deploy a vision model")
    if r.status_code != 200:
        error = reply.get("error") if isinstance(reply.get("error"), dict) else reply
        return _fail(f"HTTP {r.status_code}: {error.get('detail') or error.get('message') or reply}", error.get("fix"))
    for line in summary_lines(reply, wall):
        say(line)
    return 0


def add_image_parser(sub: Any) -> None:
    """Register `ask-image` on the top-level subparsers of cli.build_parser()."""
    p = sub.add_parser("ask-image", help="ask the ready vision model about an image and print the answer and timings")
    _arguments(p)
    p.set_defaults(func=cmd_ask_image)


def _arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("image", type=Path, help=f"a {SUPPORTED} file of at most {MAX_IMAGE_BYTES // MiB} MiB")
    p.add_argument("question")
    p.add_argument("--port", type=int, default=config.DEFAULT_GATEWAY_PORT)
    p.add_argument("--max-tokens", type=int, default=256)


def cmd_ask_image(args: argparse.Namespace) -> int:
    return ask(args.image, args.question, args.port, args.max_tokens)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ask-image", description=__doc__.splitlines()[0])
    _arguments(p)
    return cmd_ask_image(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
