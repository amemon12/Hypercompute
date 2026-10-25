"""The buyer's signer, shared by buyer.py and pool_mcp.py. Testnet only.

cdp  a CDP server wallet (default name hypercompute-buyer). The key lives in Coinbase's enclave; this machine holds
     only the CDP API key and wallet secret.
key  a throwaway private key from X402_BUYER_KEY. Never a key that holds real funds.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from x402 import x402Client
from x402.mechanisms.evm.exact.client import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

from .config import BASE_SEPOLIA

BUYER_ACCOUNT = "hypercompute-buyer"
CDP_KEYS = ("CDP_API_KEY_ID", "CDP_API_KEY_SECRET", "CDP_WALLET_SECRET")


class WalletError(RuntimeError):
    """The signer cannot be made from the given settings; the message says what is missing."""


@dataclass(frozen=True)
class Wallet:
    signer: EthAccountSigner
    address: str
    label: str  # how to describe it to a person
    close: Callable[[], Awaitable[None]]


async def _noop() -> None:
    return None


def default_signer(env: Mapping[str, str]) -> str:
    """cdp when every CDP credential is set, else key: a setup without a wallet secret works without a flag."""
    return "cdp" if all(env.get(k, "").strip() for k in CDP_KEYS) else "key"


async def open_wallet(kind: str, env: Mapping[str, str]) -> Wallet:
    if kind == "key":
        key = env.get("X402_BUYER_KEY", "").strip()
        if not key:
            raise WalletError("X402_BUYER_KEY is empty (a throwaway testnet key in .env.buyer, never a real wallet)")
        from eth_account import Account

        account = Account.from_key(key)
        return Wallet(EthAccountSigner(account), account.address, "local throwaway key", _noop)
    if kind != "cdp":
        raise WalletError(f"unknown signer {kind!r}; use cdp or key")
    missing = [k for k in CDP_KEYS if not env.get(k, "").strip()]
    if missing:
        raise WalletError(f"the CDP signer needs {', '.join(missing)} in .env")
    from cdp import CdpClient
    from cdp.evm_local_account import EvmLocalAccount

    cdp = CdpClient(api_key_id=env["CDP_API_KEY_ID"], api_key_secret=env["CDP_API_KEY_SECRET"],
                    wallet_secret=env["CDP_WALLET_SECRET"])
    try:
        account = await cdp.evm.get_or_create_account(name=BUYER_ACCOUNT)
    except BaseException:
        await cdp.close()
        raise
    return Wallet(EthAccountSigner(EvmLocalAccount(account)), account.address,
                  f"CDP server wallet {BUYER_ACCOUNT!r}, key held by Coinbase", cdp.close)


def paying_client(signer: EthAccountSigner, max_usdc: str) -> x402Client:
    """An x402 client that pays on Base Sepolia only and refuses to sign above max_usdc."""
    client = x402Client()
    client.set_spend_controls({"max_amount_per_payment": f"${max_usdc}"})
    client.register(BASE_SEPOLIA, ExactEvmScheme(signer))
    return client
