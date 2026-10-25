"""Plan, deploy, chat, stream and stop, on the machine that runs ``hypercompute host``.

    python examples/quickstart.py                      # first downloaded model in the catalog
    python examples/quickstart.py qwen2.5-7b-instruct-q4_k_m --strategy auto

The gateway URL comes from --url or HYPERCOMPUTE_URL (default http://127.0.0.1:8001). The owner token comes from
HYPERCOMPUTE_OWNER_TOKEN, or is fetched from the gateway on first use. This script only stops the deployment it
started. If another deployment is active, it says so and exits.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence

from hypercompute_sdk import DEFAULT_URL, ConflictError, HypercomputeError, Pool

Say = Callable[[str], None]


def pick_model(pool: Pool) -> str | None:
    """The first catalog model that is already downloaded."""
    return next((m["artifact"]["id"] for m in pool.models() if m.get("status") == "downloaded"), None)


def run(pool: Pool, model_id: str, *, strategy: str = "all_selected", say: Say = print,
        write: Callable[[str], object] = sys.stdout.write) -> int:
    plan = pool.plan(model_id, strategy=strategy)
    say(f"plan {plan['id'][:8]}: feasible {plan['feasible']}, tier {plan.get('tier')}, "
        f"devices {', '.join(plan.get('device_list') or [])}")
    if not plan["feasible"]:
        for reason in plan.get("reasons") or []:
            say(f"  {reason}")
        return 1
    try:
        dep = pool.deploy(plan["id"])
    except ConflictError as err:
        say(f"not started: {err}")
        return 1
    try:
        pool.wait_ready(dep["id"], on_progress=lambda state, progress: say(
            f"  {state}" + (f": {progress}" if progress else "")))
        reply = pool.chat([{"role": "user", "content": "Say hello in five words."}], max_tokens=32)
        say(f"chat: {reply['choices'][0]['message']['content']}")
        last = None
        for chunk in pool.stream([{"role": "user", "content": "Count from one to five."}], max_tokens=64):
            write(chunk.content)
            last = chunk
        write("\n")
        if last is not None and last.timings:
            say(f"decode {last.timings.get('predicted_per_second')} tok/s over "
                f"{last.timings.get('predicted_n')} tokens (this run only)")
    finally:
        stopped = pool.stop(dep["id"])
        say(f"stopped {dep['id'][:8]}: {stopped.get('state') if stopped else 'gone'}")
    return 0


def main(argv: Sequence[str] | None = None, *, pool: Pool | None = None, say: Say = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id", nargs="?", help="catalog model id; default: the first downloaded one")
    parser.add_argument("--url", default=os.environ.get("HYPERCOMPUTE_URL") or DEFAULT_URL)
    parser.add_argument("--strategy", default="all_selected",
                        choices=["auto", "all_selected", "fastest_fit", "local_only"])
    args = parser.parse_args(argv)
    try:
        with pool or Pool(args.url) as client:
            model_id = args.model_id or pick_model(client)
            if model_id is None:
                say("no downloaded model; run `uv run hypercompute models pull <id>` on the host first")
                return 1
            return run(client, model_id, strategy=args.strategy, say=say)
    except (HypercomputeError, ValueError) as err:
        say(f"error: {err}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
