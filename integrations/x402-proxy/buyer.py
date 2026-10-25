"""A buyer agent: asks the pool a question, sees the 402 price, pays in test USDC, streams the answer.

    uv run python buyer.py "Explain MoE routing in two sentences"
    uv run python buyer.py --signer key --max-tokens 48 --url http://192.168.1.221:8402 "..."

Signers (testnet only):
  cdp  (the default when CDP_WALLET_SECRET is set) a CDP server wallet named hypercompute-buyer. Its key lives in Coinbase's enclave; this machine only
       holds the CDP API key and wallet secret from .env. Create and fund it with setup_wallets.py.
  key  (the default otherwise) a throwaway private key in .env.buyer (X402_BUYER_KEY). Never a key that holds
       real funds.

A spend cap (--max-usdc, default 0.05) is enforced by the x402 client before it signs anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from x402 import x402Client
from x402.http.clients.httpx import PaymentError, x402AsyncTransport
from x402.http.utils import decode_payment_required_header, decode_payment_response_header

from x402proxy.config import BASE_SEPOLIA, EXPLORERS, read_env_file, safe_console
from x402proxy.wallet import WalletError, default_signer, open_wallet, paying_client

HERE = Path(__file__).resolve().parent


def _say(text: str = "") -> None:
    print(text, flush=True)


async def _show_price(url: str, body: dict) -> bool:
    async with httpx.AsyncClient(timeout=60, trust_env=False) as plain:
        r = await plain.post(f"{url}/v1/chat/completions", json=body)
    if r.status_code != 402:
        _say(f"the proxy answered HTTP {r.status_code} before payment: {r.text[:400]}")
        return False
    req = decode_payment_required_header(r.headers["payment-required"]).accepts[0]
    quote = r.json()
    _say(f"HTTP 402 Payment Required: {quote['price_usdc']} USDC on {req.network} (Base Sepolia testnet) "
         f"to {req.pay_to}, for up to {quote['max_tokens_reserved']} tokens of {quote['model']}")
    for s in quote["split"]:
        _say(f"   {s['machine']:<12} layers {s['layers']:<8} earns {s['earns_usdc']} USDC"
             + ("  (host, incl. fee)" if s["host"] else ""))
    return True


async def _pay_and_read(url: str, body: dict, client: x402Client) -> int:
    async with httpx.AsyncClient(transport=x402AsyncTransport(client), timeout=httpx.Timeout(900, connect=10),
                                 trust_env=False) as c:
        async with c.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                reason = ""
                if "payment-required" in r.headers:  # a refused payment says why in the header, not the body
                    reason = decode_payment_required_header(r.headers["payment-required"]).error or ""
                body_text = (await r.aread())[:600].decode(errors="replace")
                _say(f"\npayment or chat failed: HTTP {r.status_code} {reason} {body_text}".rstrip())
                return 1
            receipt = decode_payment_response_header(r.headers["payment-response"])
            _say(f"paid: settled={receipt.success} tx {receipt.transaction}")
            link = EXPLORERS.get(receipt.network or BASE_SEPOLIA, "")
            if receipt.success and link and receipt.transaction:
                _say(f"      {link}{receipt.transaction}")
            _say(f"      ledger id {r.headers.get('x-hypercompute-payment')}\n")
            if not body.get("stream"):
                message = json.loads(await r.aread())["choices"][0]["message"]
                if message.get("reasoning_content"):
                    _say(f"\033[2m{message['reasoning_content']}\033[0m")
                _say(message.get("content") or "")
                return 0
            return await _print_stream(r)


async def _print_stream(r: httpx.Response) -> int:
    buf = ""
    async for chunk in r.aiter_text():
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            data = block.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if "error" in event:
                _say(f"\n[stream cut: {event['error'].get('message')}]")
                return 1
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):  # reasoning models think first; shown dimmed
                    print(f"\033[2m{delta['reasoning_content']}\033[0m", end="", flush=True)
                print(delta.get("content") or "", end="", flush=True)
    _say()
    return 0


async def run(args: argparse.Namespace) -> int:
    env = {**read_env_file(HERE / ".env"), **read_env_file(HERE / ".env.buyer"), **os.environ}
    body = {"messages": [{"role": "user", "content": args.prompt}], "max_tokens": args.max_tokens,
            "stream": not args.no_stream}
    if args.reasoning_effort:  # gpt-oss reads this from its chat template; other models ignore it
        body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    url = args.url.rstrip("/")
    if not await _show_price(url, body):
        return 1
    try:
        wallet = await open_wallet(args.signer or default_signer(env), env)
    except WalletError as exc:
        _say(f"no wallet: {exc}")
        return 2
    try:
        _say(f"buyer wallet {wallet.address} ({wallet.label})")
        _say(f"signing an EIP-3009 USDC authorization (cap {args.max_usdc} USDC) and retrying...")
        try:
            return await _pay_and_read(url, body, paying_client(wallet.signer, args.max_usdc))
        except PaymentError as exc:  # e.g. the price is over --max-usdc: nothing was signed or sent
            _say(f"did not pay: {exc}")
            return 1
    finally:
        await wallet.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--url", default="http://127.0.0.1:8402", help="the x402 proxy")
    parser.add_argument("--signer", choices=("cdp", "key"),
                        help="default: cdp when CDP_WALLET_SECRET is set, else key")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--max-usdc", default="0.05", help="refuse to sign for more than this")
    parser.add_argument("--no-stream", action="store_true", help="settle after the full reply instead")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"),
                        help="for gpt-oss: think less so a small max_tokens still reaches the answer")
    args = parser.parse_args()
    safe_console()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
