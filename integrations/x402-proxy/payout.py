"""Settle up: pay each friend what the ledger says their machine earned, from the host's CDP wallet. You run this.

    uv run python payout.py              # dry run: who is owed what, nothing is sent
    uv run python payout.py --execute    # send test USDC on Base Sepolia from the CDP seller wallet

Only works when X402_PAY_TO is the CDP server wallet setup_wallets.py created (hypercompute-seller), since that is
the wallet the payments landed in and the one this script can sign for without a private key on disk. The seller
needs a little Base Sepolia ETH for gas (setup_wallets.py --fund requests it). Friends without a payout address in
X402_PAYEES are skipped and stay owed. Each payout is booked as pending before its transfer is sent and marked sent
after, so a crash in between can never pay twice: it leaves a pending row to check on BaseScan, then
`payout.py --release ID` if nothing was sent.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from x402proxy.config import ADDRESS, BASE_SEPOLIA, EXPLORERS, HERE, ConfigError, load, read_env_file, safe_console
from x402proxy.ledger import Ledger
from x402proxy.pricing import atomic_to_usd, usd_to_atomic

SELLER = "hypercompute-seller"
CDP_NETWORK = "base-sepolia"  # testnet only; there is no option for another network


def plan_payouts(ledger: Ledger, payees: dict[str, str], min_atomic: int) -> tuple[list[dict], list[dict]]:
    """(to pay, skipped). A friend's address comes from X402_PAYEES now, else the one recorded with their shares."""
    pay, skip = [], []
    for row in ledger.owed():
        address = payees.get(row["display_name"]) or payees.get(row["node_id"]) or row["payout_address"]
        entry = {**row, "address": address}
        if not address or not ADDRESS.match(address):
            skip.append({**entry, "reason": "no payout address"})
        elif row["owed_atomic"] < min_atomic:
            skip.append({**entry, "reason": f"under the {atomic_to_usd(min_atomic)} USDC minimum"})
        else:
            pay.append(entry)
    return pay, skip


async def execute(pay: list[dict], env: dict[str, str], pay_to: str, ledger: Ledger) -> int:
    missing = [k for k in ("CDP_API_KEY_ID", "CDP_API_KEY_SECRET", "CDP_WALLET_SECRET") if not env.get(k)]
    if missing:
        print(f"--execute needs {', '.join(missing)} in .env")
        return 2
    from cdp import CdpClient

    async with CdpClient(api_key_id=env["CDP_API_KEY_ID"], api_key_secret=env["CDP_API_KEY_SECRET"],
                         wallet_secret=env["CDP_WALLET_SECRET"]) as cdp:
        seller = await cdp.evm.get_account(name=SELLER)
        if seller.address.lower() != pay_to.lower():
            print(f"X402_PAY_TO {pay_to} is not the CDP wallet {SELLER!r} ({seller.address}); pay friends from the "
                  "wallet that received the payments instead")
            return 2
        for entry in pay:
            payout_id = ledger.reserve_payout(entry["node_id"], entry["display_name"], entry["address"],
                                              entry["owed_atomic"], BASE_SEPOLIA)
            try:
                tx = await seller.transfer(to=entry["address"], amount=entry["owed_atomic"], token="usdc",
                                           network=CDP_NETWORK)
            except Exception as exc:  # sent or not is unknown: leave it pending, never retry blindly
                print(f"transfer to {entry['display_name']} failed or its outcome is unknown ({exc}). Payout "
                      f"#{payout_id} stays pending: check the seller wallet on BaseScan, then run "
                      f"payout.py --release {payout_id} only if nothing was sent.")
                return 1
            ledger.complete_payout(payout_id, str(tx))
            print(f"paid {entry['display_name']} {atomic_to_usd(entry['owed_atomic'])} USDC: "
                  f"{EXPLORERS[BASE_SEPOLIA]}{tx}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="actually send the transfers (testnet USDC)")
    parser.add_argument("--min-usdc", default="0.000100", help="skip friends owed less than this")
    parser.add_argument("--release", type=int, metavar="ID",
                        help="mark a pending payout as not sent (only after checking BaseScan), so it is owed again")
    args = parser.parse_args()
    safe_console()
    env = {**read_env_file(HERE / ".env"), **os.environ}
    try:
        config = load(env)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
    ledger = Ledger(config.ledger_path)
    try:
        if args.release is not None:
            ledger.fail_payout(args.release)
            print(f"payout #{args.release} released; its amount is owed again")
            return 0
        pay, skip = plan_payouts(ledger, dict(config.payees), usd_to_atomic(args.min_usdc))
        for e in pay:
            print(f"owe {e['display_name']:<12} {atomic_to_usd(e['owed_atomic'])} USDC -> {e['address']}")
        for e in skip:
            print(f"skip {e['display_name']:<11} {atomic_to_usd(e['owed_atomic'])} USDC ({e['reason']})")
        if not pay:
            print("nothing to pay out")
            return 0
        if not args.execute:
            print(f"dry run: {atomic_to_usd(sum(e['owed_atomic'] for e in pay))} test USDC would be sent on Base "
                  "Sepolia; add --execute to send it")
            return 0
        return asyncio.run(execute(pay, env, config.pay_to, ledger))
    finally:
        ledger.close()


if __name__ == "__main__":
    sys.exit(main())
