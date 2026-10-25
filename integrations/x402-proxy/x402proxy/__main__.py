"""python -m x402proxy serve | ledger | reconcile

serve      run the paywall proxy (reads integrations/x402-proxy/.env, then the process environment)
ledger     print what each machine has earned, from the local ledger file
reconcile  replay unrecorded-settlements.jsonl (settlements the ledger could not take) into the ledger
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import HERE, ConfigError, load, read_env_file, safe_console
from .pricing import atomic_to_usd


def _env() -> dict[str, str]:
    return {**read_env_file(HERE / ".env"), **os.environ}


def _serve(env: dict[str, str]) -> int:
    import uvicorn

    from .app import create_app

    config = load(env)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    facilitator = "CDP facilitator" if config.facilitator == "cdp" else config.facilitator
    print(f"x402 proxy on http://{config.bind_host}:{config.port}  ->  gateway {config.gateway_url}\n"
          f"network {config.network} (testnet), facilitator {facilitator}, pay_to {config.pay_to}\n"
          f"price ${atomic_to_usd(config.pricing.base_atomic)} + ${atomic_to_usd(config.pricing.per_1k_atomic)} "
          f"per 1k reserved tokens, host fee {config.host_fee_bps / 100:g}%\n"
          f"ledger page http://{config.bind_host}:{config.port}/ledger")
    try:
        app = create_app(config, env)  # asks the facilitator what it supports, so a bad key fails here
    except Exception as exc:  # unreachable facilitator, bad key, unsupported network: say so and exit
        print(f"facilitator check failed: {type(exc).__name__}: {exc}\n"
              "(CDP facilitator: check CDP_API_KEY_ID and CDP_API_KEY_SECRET; "
              "or leave them empty to use the public x402.org facilitator)", file=sys.stderr)
        return 2
    uvicorn.run(app, host=config.bind_host, port=config.port, log_level="info")
    return 0


def _ledger(env: dict[str, str]) -> int:
    from .ledger import Ledger

    config = load(env)
    ledger = Ledger(config.ledger_path)
    try:
        s = ledger.summary()
        print(f"{s['payments']} paid chats, {atomic_to_usd(s['paid_atomic'])} test USDC paid, "
              f"{atomic_to_usd(s['refunds_owed_atomic'])} refunds owed\n")
        print(f"{'machine':<18}{'role':<6}{'earned USDC':>14}  payout address")
        for t in ledger.totals():
            print(f"{t['display_name'][:17]:<18}{'host' if t['is_host'] else 'peer':<6}"
                  f"{atomic_to_usd(t['earned_atomic']):>14}  {t['payout_address'] or '(none yet)'}")
        for p in ledger.recent(10):
            tx = p["tx"] or "-"
            print(f"\n{p['created_at']}  {atomic_to_usd(p['amount_atomic'])} USDC  {p['status']}  "
                  f"{p['completion_tokens']}/{p['max_tokens']} tokens  tx {tx}")
            print("   " + ", ".join(f"{s['display_name']} {s['layers']} = {atomic_to_usd(s['amount_atomic'])}"
                                    for s in p["shares"]))
    finally:
        ledger.close()
    return 0


def _reconcile(env: dict[str, str]) -> int:
    from .ledger import Ledger
    from .reconcile import replay

    config = load(env)
    ledger = Ledger(config.ledger_path)
    try:
        counts = replay(ledger, config.ledger_path.with_name("unrecorded-settlements.jsonl"))
    finally:
        ledger.close()
    print(f"replayed {counts['settlements']} settlements and {counts['outcomes']} outcomes, "
          f"skipped {counts['skipped']} already in the ledger")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="x402proxy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("serve", "ledger", "reconcile"))
    args = parser.parse_args(argv)
    safe_console()
    commands = {"serve": _serve, "ledger": _ledger, "reconcile": _reconcile}
    try:
        return commands[args.command](_env())
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
