"""Test-day helper: write a chat body with a prompt of about N tokens (Qwen2.5: ' data' is one token), or a body of an
exact byte size for the gateway's body limits.

Usage:  uv run python <this file> N OUT.json [--max-tokens 16] [--no-stream] [--ignore-eos] [--bytes B]
--bytes B pads the prompt so the whole file is exactly B bytes (N is then ignored).
Python writes the file itself (PowerShell 5.1 `>` would write UTF-16). The run record's prompt_tokens is exact.
"""

from __future__ import annotations

import argparse
import json


def body_for(content: str, a: argparse.Namespace) -> dict:
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens,
            "stream": not a.no_stream, "temperature": 0, "cache_prompt": False}
    if a.ignore_eos:
        body["ignore_eos"] = True
    return body


ap = argparse.ArgumentParser()
ap.add_argument("tokens", type=int)
ap.add_argument("out")
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--no-stream", action="store_true")
ap.add_argument("--ignore-eos", action="store_true", help="llama.cpp only: generate exactly max_tokens")
ap.add_argument("--bytes", type=int, default=None, help="exact size of the written body in bytes")
a = ap.parse_args()
head = "Reply with the number of words below.\n"
if a.bytes is None:
    text = json.dumps(body_for(head + "data " * a.tokens, a))
else:
    empty = len(json.dumps(body_for(head, a)).encode())
    if a.bytes < empty:
        raise SystemExit(f"--bytes must be at least {empty}")
    text = json.dumps(body_for(head + "x" * (a.bytes - empty), a))
with open(a.out, "w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print(f"wrote {a.out}: {len(text.encode())} bytes")
