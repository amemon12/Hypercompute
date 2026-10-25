"""Create the demo's testnet wallets with your CDP key, and fund the buyer from the CDP faucet. You run this.

    uv run python setup_wallets.py            # create or reuse both wallets, print addresses and balances
    uv run python setup_wallets.py --fund     # also ask the CDP faucet for Base Sepolia test USDC for the buyer

Needs CDP_API_KEY_ID, CDP_API_KEY_SECRET and CDP_WALLET_SECRET in .env. Both wallets are CDP server wallets: their
keys stay in Coinbase's enclave and never touch this disk. Base Sepolia only; the faucet hands out test USDC that
has no value. If X402_PAY_TO in .env is empty, the seller wallet's PUBLIC address is written there; nothing secret
is ever written or printed.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from x402proxy.config import ADDRESS, BASE_SEPOLIA, EXPLORERS, read_env_file, safe_console

HERE = Path(__file__).resolve().parent
ENV = HERE / ".env"
SELLER, BUYER = "hypercompute-seller", "hypercompute-buyer"
CDP_NETWORK = "base-sepolia"  # the CDP API's name for eip155:84532
USDC_BASE_SEPOLIA = "0x036cbd53842c5426634e7929541ec2318f3dcf7e"  # Circle's test USDC on Base Sepolia


def _set_pay_to(address: str) -> None:
    """Fill X402_PAY_TO in .env when it is empty; leaves every other line untouched."""
    if not ADDRESS.match(address):
        raise ValueError("not an address")
    text = ENV.read_text(encoding="utf-8") if ENV.is_file() else ""
    if re.search(r"^X402_PAY_TO=\s*0x[0-9a-fA-F]{40}\s*$", text, flags=re.M):
        return
    if re.search(r"^X402_PAY_TO=.*$", text, flags=re.M):
        text = re.sub(r"^X402_PAY_TO=.*$", f"X402_PAY_TO={address}", text, count=1, flags=re.M)
    else:
        text += f"\nX402_PAY_TO={address}\n"
    ENV.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote X402_PAY_TO={address} to .env (public address only)")


async def _usdc(cdp, address: str) -> str:
    result = await cdp.evm.list_token_balances(address=address, network=CDP_NETWORK)
    for b in result.balances:
        if (b.token.symbol or "").upper() == "USDC" or b.token.contract_address.lower() == USDC_BASE_SEPOLIA:
            return f"{int(b.amount.amount) / 10 ** b.amount.decimals:.6f}"
    return "0.000000"


async def run(fund: bool) -> int:
    env = read_env_file(ENV)
    missing = [k for k in ("CDP_API_KEY_ID", "CDP_API_KEY_SECRET", "CDP_WALLET_SECRET") if not env.get(k)]
    if missing:
        print(f"fill {', '.join(missing)} in {ENV} first (CDP portal: API Keys, and Non-custodial Wallet > Security)")
        return 2
    from cdp import CdpClient

    async with CdpClient(api_key_id=env["CDP_API_KEY_ID"], api_key_secret=env["CDP_API_KEY_SECRET"],
                         wallet_secret=env["CDP_WALLET_SECRET"]) as cdp:
        seller = await cdp.evm.get_or_create_account(name=SELLER)
        buyer = await cdp.evm.get_or_create_account(name=BUYER)
        pay_to = env.get("X402_PAY_TO", "")
        if not ADDRESS.match(pay_to):
            _set_pay_to(seller.address)
            pay_to = seller.address
        print(f"seller (receives payments): {pay_to}" + ("  [CDP wallet]" if pay_to == seller.address else ""))
        print(f"buyer  (the paying agent):  {buyer.address}  [CDP wallet {BUYER!r}]")
        if fund:
            tx = await cdp.evm.request_faucet(address=buyer.address, network=CDP_NETWORK, token="usdc")
            print(f"faucet: test USDC requested for the buyer, tx {EXPLORERS[BASE_SEPOLIA]}{tx}")
            if pay_to == seller.address:  # gas for payout.py, which sends friends their shares from this wallet
                tx = await cdp.evm.request_faucet(address=seller.address, network=CDP_NETWORK, token="eth")
                print(f"faucet: test ETH requested for the seller's payout gas, tx {EXPLORERS[BASE_SEPOLIA]}{tx}")
            await asyncio.sleep(6)  # a Base Sepolia block or two, so the balance below includes it
        print(f"buyer USDC balance on Base Sepolia: {await _usdc(cdp, buyer.address)}")
        if pay_to == seller.address:
            print(f"seller USDC balance on Base Sepolia: {await _usdc(cdp, seller.address)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fund", action="store_true",
                        help="request test USDC for the buyer (and test ETH for the seller's payout gas)")
    fund = parser.parse_args().fund
    safe_console()
    return asyncio.run(run(fund))


if __name__ == "__main__":
    sys.exit(main())
