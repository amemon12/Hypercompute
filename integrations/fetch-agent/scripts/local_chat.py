"""Rehearse the ASI:One conversation locally: a stand-in user agent talks to the Hypercompute agent over the Agent
Chat Protocol on 127.0.0.1, clicks the cards' primary buttons, and pays test payments, printing every reply.

    .venv/Scripts/python scripts/local_chat.py --print-address            # the address to allowlist
    .venv/Scripts/python scripts/local_chat.py "Run 'what is RPC' on the biggest model my friends' laptops can hold"

The Hypercompute agent must run in local mode: HC_MAILBOX=false and HC_LOCAL_PEERS=<this address>=
http://127.0.0.1:8461/submit (see the README). No Agentverse account is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path

from uagents import Agent, Context
from uagents.resolver import RulesBasedResolver
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    MetadataContent,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.contrib.protocols.chat.cards import create_card_response_content, extract_card
from uagents_core.contrib.protocols.payment import (
    CancelPayment,
    CommitPayment,
    CompletePayment,
    Funds,
    RequestPayment,
    payment_protocol_spec,
)
from uagents_core.identity import Identity
from uagents import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hc_agent.agent import NoRegistration, silence_almanac_status  # noqa: E402  (a rehearsal publishes nothing)

CLIENT_PORT = 8461
SEED_FILE = Path(__file__).resolve().parent.parent / "agent_data" / "local_client_seed"


def client_seed() -> str:
    """A random seed made once per machine (git-ignored). A published seed would let anyone sign as this
    stand-in, which matters if its address is ever left on a live agent's allowlist."""
    if os.environ.get("HC_LOCAL_CLIENT_SEED"):
        return os.environ["HC_LOCAL_CLIENT_SEED"]
    if not SEED_FILE.is_file():
        SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
        SEED_FILE.write_text(secrets.token_hex(32), encoding="utf-8")
    return SEED_FILE.read_text(encoding="utf-8").strip()


def card_click(card) -> dict | None:
    payload = card.payload
    for attr in ("approve_cta", "submit_cta"):
        cta = getattr(payload, attr, None)
        if cta is not None:
            return dict(cta.selection)
    ctas = getattr(payload, "ctas", None) or []
    primary = next((c for c in ctas if c.primary), None)
    return dict(primary.selection) if primary else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("messages", nargs="*", help="messages to send in order (after each final reply)")
    parser.add_argument("--agent", default=None, help="Hypercompute agent address (default: derived from .env seed)")
    parser.add_argument("--agent-port", type=int, default=8460)
    parser.add_argument("--click", action="store_true", help="click the primary button of review and form cards")
    parser.add_argument("--form-prompt", default="Explain what RPC is in one sentence.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--print-address", action="store_true")
    args = parser.parse_args()

    seed = client_seed()
    me = Identity.from_seed(seed, 0).address
    if args.print_address:
        print(me)
        return
    target = args.agent
    if target is None:
        from dotenv import dotenv_values

        seed = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env")).get("HC_AGENT_SEED")
        if not seed:
            sys.exit("pass --agent or set HC_AGENT_SEED in .env")
        target = Identity.from_seed(seed, 0).address

    client = Agent(name="local-user", seed=seed, port=CLIENT_PORT,
                   endpoint=[f"http://127.0.0.1:{CLIENT_PORT}/submit"],
                   resolve=RulesBasedResolver({target: f"http://127.0.0.1:{args.agent_port}/submit"}),
                   enable_agent_inspector=False, publish_agent_details=False,
                   registration_policy=NoRegistration(), report_events=False, mark_inactive_on_shutdown=False)
    silence_almanac_status(client)
    queue = list(args.messages)
    done = asyncio.Event()
    chat = Protocol(spec=chat_protocol_spec)
    payments = Protocol(spec=payment_protocol_spec, role="buyer")

    async def send_next(ctx: Context) -> None:
        if not queue:
            done.set()
            return
        text = queue.pop(0)
        print(f"\n>>> {text}", flush=True)
        await ctx.send(target, ChatMessage(content=[TextContent(text=text)]))

    @chat.on_message(ChatMessage)
    async def on_reply(ctx: Context, sender: str, msg: ChatMessage) -> None:
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
        final = True
        for item in msg.content:
            if isinstance(item, TextContent):
                print(f"\n<<< {item.text}", flush=True)
                if item.text.startswith("_") and item.text.endswith("_"):
                    final = False  # a progress line: more is coming
            elif isinstance(item, MetadataContent):
                card = extract_card(item)
                if card is None:
                    continue
                kind = type(card.payload).__name__
                print(f"    [card {kind}: {getattr(card.payload, 'title', '')}]", flush=True)
                selection = card_click(card) if args.click else None
                if selection and kind in ("ReviewCardPayload", "FormCardPayload"):
                    if kind == "FormCardPayload":
                        selection = {**selection, "prompt": args.form_prompt}
                    print(f"    [click {json.dumps(selection)}]", flush=True)
                    await ctx.send(sender, ChatMessage(content=[create_card_response_content(
                        card_id=card.card_id, selection=selection)]))
                    final = False
        if final and "Sent a payment request" not in msg.text():
            await send_next(ctx)

    @chat.on_message(ChatAcknowledgement)
    async def on_ack(_ctx: Context, _sender: str, _msg: ChatAcknowledgement) -> None:
        return None

    @payments.on_message(RequestPayment)
    async def on_request(ctx: Context, sender: str, msg: RequestPayment) -> None:
        funds = msg.accepted_funds[0]
        print(f"    [payment request {funds.amount} {funds.currency} to {msg.recipient[:14]}...; paying (test)]",
              flush=True)
        await ctx.send(sender, CommitPayment(funds=Funds(amount=funds.amount, currency=funds.currency,
                                                         payment_method=funds.payment_method),
                                             recipient=msg.recipient, transaction_id=secrets.token_hex(32).upper(),
                                             reference=msg.reference, metadata={"buyer_fet_wallet": "fetch1test"}))

    @payments.on_message(CompletePayment)
    async def on_complete(_ctx: Context, _sender: str, msg: CompletePayment) -> None:
        print(f"    [payment complete {msg.transaction_id[:12]}...]", flush=True)

    @payments.on_message(CancelPayment)
    async def on_cancel(_ctx: Context, _sender: str, msg: CancelPayment) -> None:
        print(f"    [payment cancelled: {msg.reason}]", flush=True)

    client.include(chat)
    client.include(payments)

    @client.on_event("startup")
    async def start(ctx: Context) -> None:
        await asyncio.sleep(1.0)
        await send_next(ctx)

        async def stop_when_done() -> None:
            try:
                await asyncio.wait_for(done.wait(), timeout=args.timeout)
            except TimeoutError:
                print("\n[timed out waiting for replies]", flush=True)
            await asyncio.sleep(1.0)
            os._exit(0)

        asyncio.get_running_loop().create_task(stop_when_done())

    client.run()


if __name__ == "__main__":
    main()
