"""The buyer's signer selection, without touching a network or a real key."""

from __future__ import annotations

import pytest
from eth_account import Account

from x402proxy.wallet import WalletError, open_wallet, paying_client


async def test_key_wallet_uses_the_given_throwaway_key():
    account = Account.create()  # fresh and unfunded
    wallet = await open_wallet("key", {"X402_BUYER_KEY": account.key.hex()})
    assert wallet.address == account.address and wallet.label == "local throwaway key"
    await wallet.close()


async def test_wallet_settings_errors_say_what_is_missing():
    with pytest.raises(WalletError, match="X402_BUYER_KEY"):
        await open_wallet("key", {})
    with pytest.raises(WalletError, match="CDP_WALLET_SECRET"):
        await open_wallet("cdp", {"CDP_API_KEY_ID": "id", "CDP_API_KEY_SECRET": "secret"})
    with pytest.raises(WalletError, match="unknown signer"):
        await open_wallet("ledger", {})


async def test_paying_client_refuses_prices_over_its_cap():
    from x402.schemas import PaymentRequired, PaymentRequirements, ResourceInfo

    wallet = await open_wallet("key", {"X402_BUYER_KEY": Account.create().key.hex()})
    client = paying_client(wallet.signer, "0.001")
    over = PaymentRequirements(scheme="exact", network="eip155:84532", asset="0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                               amount="1960", pay_to="0x" + "ab" * 20, max_timeout_seconds=60,
                               extra={"name": "USDC", "version": "2"})
    required = PaymentRequired(x402_version=2, resource=ResourceInfo(url="http://proxy/v1/chat/completions"),
                               accepts=[over])
    with pytest.raises(Exception, match="spend_controls"):
        await client.create_payment_payload(required)


def test_default_signer_needs_every_cdp_credential():
    from x402proxy.wallet import default_signer

    assert default_signer({"CDP_API_KEY_ID": "i", "CDP_API_KEY_SECRET": "s", "CDP_WALLET_SECRET": "w"}) == "cdp"
    assert default_signer({"CDP_API_KEY_ID": "i", "CDP_API_KEY_SECRET": "s", "X402_BUYER_KEY": "k"}) == "key"
