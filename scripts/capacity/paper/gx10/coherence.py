"""Ask the running deployment two short questions (temperature 0) and print the replies.

The owner token is never printed.

Usage (on the host, from the repo): .venv/bin/python coherence.py [MAX_TOKENS]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

sys.path.insert(0, "backend")
from hypercompute import config  # noqa: E402 - needs the repo's backend on the path

BASE = "http://127.0.0.1:8001"
QUESTIONS = ("What is the capital of Australia? Answer in one sentence.",
             "Write a Python function that returns the n-th Fibonacci number iteratively.")


def ask(question: str, max_tokens: int, headers: dict[str, str]) -> None:
    body = {"messages": [{"role": "user", "content": question}], "max_tokens": max_tokens, "temperature": 0,
            "seed": 7, "stream": False, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions",  # noqa: S310 - fixed loopback gateway URL
                                 data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:  # noqa: S310 - fixed loopback gateway URL
        d = json.load(r)
    choice = d["choices"][0]
    msg = choice["message"]
    print(f"\nQ: {question}\n  ({time.time() - t0:.0f} s, finish {choice.get('finish_reason')}, "
          f"usage {d.get('usage')})")
    if msg.get("reasoning_content"):
        print("  REASONING:", msg["reasoning_content"][:600].replace("\n", " | "))
    print("  ANSWER:", (msg.get("content") or "")[:900])


def main() -> int:
    with urllib.request.urlopen(BASE + "/api/owner/bootstrap", timeout=10) as r:  # noqa: S310 - loopback
        token = json.load(r)["owner_token"]
    headers = {config.OWNER_HEADER: token, "Content-Type": "application/json"}
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for q in QUESTIONS:
        ask(q, max_tokens, headers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
