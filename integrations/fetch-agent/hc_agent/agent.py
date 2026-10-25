"""The uAgent: Agent Chat Protocol for ASI:One (with Interactive Cards), the Payment Protocol as seller, and a small
structured protocol so other agents can use the pool too. Connects to Agentverse through a mailbox, so the host
needs no inbound port.

    .venv/Scripts/python -m hc_agent.agent

With a mailbox the agent runs no HTTP server at all: the agent inspector (whose REST routes uAgents serves with open
CORS) is off, and the mailbox is registered in-process with the owner's Agentverse API key instead of through the
inspector's /connect route.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from uagents import Agent, Context, Model, Protocol
from uagents import asgi as uagents_asgi
from uagents.mailbox import AgentverseConnectRequest, register_in_agentverse
from uagents.resolver import RulesBasedResolver
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    MetadataContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from uagents_core.contrib.protocols.chat.cards import create_card_content, extract_card_response
from uagents_core.contrib.protocols.payment import (
    CancelPayment,
    CommitPayment,
    CompletePayment,
    Funds,
    RejectPayment,
    RequestPayment,
    payment_protocol_spec,
)
from uagents_core.identity import Identity
from uagents_core.registration import AgentRegistrationPolicy, RegistrationRequest
from uagents_core.types import AgentEndpoint, DeliveryStatus

from . import __version__
from .config import ConfigError, Settings
from .conversation import Conversation
from .gateway import Gateway
from .intent import Intent
from .payments import LedgerVerifier, PaymentAsk, Verdict
from .planner import GIB
from .replies import Reply
from .safety import SeenMessages
from .store import StoreError
from .workflow import Workflow, WorkflowError

log = logging.getLogger("hypercompute.agent")
HERE = Path(__file__).resolve().parent.parent
README = HERE / "README.md"
DESCRIPTION = ("Runs your prompt on the biggest model your friends' pooled laptops can hold (up to 1T parameters), "
               "over an encrypted Iroh tunnel, and reports which machine served which layers and the measured tok/s.")
PROFILE_METADATA = {"categories": ["compute", "llm-inference"],
                    "tags": ["llm", "distributed-inference", "llama.cpp", "pooled-compute", "innovationlab",
                             "hackathon"]}


# ---------------------------------------------------------------------- message building


def chat_message(reply: Reply) -> ChatMessage:
    content: list = [TextContent(text=reply.text)]
    if reply.card is not None:
        content.append(create_card_content(reply.card, card_id=reply.card_id, is_terminal=reply.terminal))
    if reply.end_session:
        content.append(EndSessionContent())
    return ChatMessage(content=content)


def read_chat(msg: ChatMessage) -> tuple[str, dict | None, bool]:
    """(text, card selection or None, cancelled) from an inbound ChatMessage."""
    texts, selection, cancelled = [], None, False
    for item in msg.content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        elif isinstance(item, MetadataContent):
            response = extract_card_response(item)
            if response is None:
                continue
            if response.cancelled:
                cancelled = True
            if response.selection:
                selection = dict(response.selection)
            elif response.text:
                texts.append(response.text)
    return "\n".join(t for t in texts if t).strip(), selection, cancelled


def payment_request(ask: PaymentAsk) -> RequestPayment:
    return RequestPayment(
        accepted_funds=[Funds(amount=f"{ask.amount.normalize():f}", currency=ask.currency, payment_method="fet_direct")],
        recipient=ask.recipient, deadline_seconds=ask.deadline_s, reference=ask.reference,
        description=ask.description,
        metadata={"provider_agent_wallet": ask.recipient, "fet_network": ask.network,
                  "mainnet": "true" if ask.network == "mainnet" else "false", "content": ask.description})


def is_progress(reply: Reply) -> bool:
    return reply.text.startswith("_") and reply.text.endswith("_") and reply.card is None


class ContextOutbox:
    """Sends through the uAgents context of the message being handled, so replies keep that message's session.
    ctx.send reports failures in its return value instead of raising; anything but a progress line gets one retry,
    and a final failure is logged."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    async def _send(self, to: str, msg: Model, *, retry: bool) -> bool:
        for attempt in (1, 2):
            status = await self.ctx.send(to, msg)
            if getattr(status, "status", None) != DeliveryStatus.FAILED:
                return True
            if not retry or attempt == 2:
                self.ctx.logger.error(f"could not deliver {type(msg).__name__} to {to[:20]}: "
                                      f"{getattr(status, 'detail', '')}")
                return False
        return False

    async def reply(self, to: str, reply: Reply) -> bool:
        """True unless uAgents reported the send failed. Advisory only: the recipient's side controls it, so it
        never decides about money."""
        return await self._send(to, chat_message(reply), retry=not is_progress(reply))

    async def request_payment(self, to: str, ask: PaymentAsk) -> bool:
        return await self._send(to, payment_request(ask), retry=True)

    async def complete_payment(self, to: str, tx_id: str) -> None:
        await self._send(to, CompletePayment(transaction_id=tx_id), retry=True)

    async def cancel_payment(self, to: str, tx_id: str | None, reason: str) -> None:
        await self._send(to, CancelPayment(transaction_id=tx_id, reason=reason), retry=True)


class NoRegistration(AgentRegistrationPolicy):
    """Local rehearsal mode: the agent registers nowhere. The default policy would publish its address and
    127.0.0.1 endpoint to the public Almanac registry."""

    async def register(self, agent_identifier: str, identity: Identity, protocols: list[str],
                       endpoints: list[AgentEndpoint], metadata: dict[str, Any] | None = None) -> None:
        return None


async def _no_status_update(active: bool) -> None:
    return None


def silence_almanac_status(agent: Agent) -> None:
    """uAgents 0.26.1 posts a signed active/inactive status to the Almanac on every start and stop, whatever the
    registration policy; a local rehearsal must not touch the public record."""
    agent._update_agent_status = _no_status_update


class AcceptTestPayments:
    """Local rehearsal only (HC_PAYMENT_VERIFY=accept-test with HC_MAILBOX=false): every commit verifies."""

    async def verify(self, tx_id: str, *, amount: Decimal, buyer_wallet: str | None,
                     not_before: datetime | None = None) -> Verdict:
        return Verdict(True, "accepted without a ledger lookup (local test mode)")


# ---------------------------------------------------------------------- agent-to-agent protocol


class PoolStatusRequest(Model):
    pass


class PoolStatusResponse(Model):
    online_peers: int
    machines: list[str]
    pooled_gib: float
    models: list[str]
    loaded: str | None = None
    error: str | None = None


class RunRequest(Model):
    prompt: str
    model: str | None = None
    max_tokens: int | None = None


class RunResponse(Model):
    ok: bool
    answer: str | None = None
    model: str | None = None
    decode_tok_s: float | None = None
    machines: list[str] = []
    layers: dict[str, int] = {}
    error: str | None = None


# ---------------------------------------------------------------------- wiring


def build(settings: Settings) -> tuple[Agent, Conversation]:
    resolver = RulesBasedResolver(dict(settings.local_peers)) if settings.local_peers else None
    local = not settings.mailbox  # a local rehearsal publishes nothing: no Almanac record, no manifests
    agent = Agent(
        name=settings.agent_name, seed=settings.seed, port=settings.agent_port, mailbox=settings.mailbox,
        endpoint=None if settings.mailbox else [f"http://127.0.0.1:{settings.agent_port}/submit"],
        resolve=resolver, readme_path=str(README) if README.is_file() else None, description=DESCRIPTION,
        handle=settings.agent_handle, publish_agent_details=not local, handle_messages_concurrently=True,
        version=__version__, metadata=PROFILE_METADATA, enable_agent_inspector=False,
        registration_policy=NoRegistration() if local else None, report_events=not local,
        mark_inactive_on_shutdown=not local)
    gateway = Gateway(settings.gateway_url)
    workflow = Workflow(gateway, settings)
    wallet = str(agent.wallet.address())
    verifier = (AcceptTestPayments() if settings.payment_verify == "accept-test" else
                LedgerVerifier(wallet, settings.fet_network, Path(settings.data_dir) / "used_transactions.json"))
    convo = Conversation(settings, workflow, address=agent.address,
                         wallet_address=wallet if settings.payments_enabled else None,
                         verifier=verifier if settings.payments_enabled else None)
    seen = SeenMessages()

    chat = Protocol(spec=chat_protocol_spec)

    @chat.on_message(ChatMessage)
    async def on_chat(ctx: Context, sender: str, msg: ChatMessage) -> None:
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
        if not seen.first_time(str(msg.msg_id)):
            return
        out = ContextOutbox(ctx)
        text, selection, cancelled = read_chat(msg)
        what = f"card {selection.get('action')}" if selection else f"text ({len(text)} chars)"
        ctx.logger.info(f"chat from {sender} ({convo.gate.role(sender)}): {what}")  # prompts are never logged
        if cancelled:
            await convo.on_selection(sender, {"action": "cancel"}, out)
        elif selection is not None:
            await convo.on_selection(sender, selection, out)
        elif text:
            await convo.on_text(sender, text, out)
        elif any(isinstance(c, StartSessionContent) for c in msg.content):
            await convo.on_text(sender, "help", out)

    @chat.on_message(ChatAcknowledgement)
    async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
        ctx.logger.debug(f"ack from {sender[:16]} for {msg.acknowledged_msg_id}")

    payments = Protocol(spec=payment_protocol_spec, role="seller")

    @payments.on_message(CommitPayment)
    async def on_commit(ctx: Context, sender: str, msg: CommitPayment) -> None:
        if not convo.gate.message(sender).allowed:  # a flood writes nothing to disk
            ctx.logger.warning(f"payment commit from {sender[:20]} dropped by the flood guard: "
                               f"{str(msg.transaction_id)[:16]!r}")
            return
        meta = msg.metadata or {}
        buyer = meta.get("buyer_fet_wallet") or meta.get("buyer_fet_address")
        await convo.on_commit(sender, ContextOutbox(ctx), tx_id=msg.transaction_id, amount=msg.funds.amount,
                              currency=msg.funds.currency, reference=msg.reference,
                              buyer_wallet=buyer if isinstance(buyer, str) else None)

    @payments.on_message(RejectPayment)
    async def on_reject(ctx: Context, sender: str, msg: RejectPayment) -> None:
        if convo.gate.message(sender).allowed:
            await convo.on_reject(sender, msg.reason, ContextOutbox(ctx))

    pool = Protocol(name="HypercomputePool", version="0.1.0")

    @pool.on_message(PoolStatusRequest, replies={PoolStatusResponse})
    async def on_status(ctx: Context, sender: str, _msg: PoolStatusRequest) -> None:
        if not (convo.gate.message(sender).allowed and convo.gate.action(sender, "status").allowed):
            await ctx.send(sender, PoolStatusResponse(online_peers=0, machines=[], pooled_gib=0, models=[],
                                                      error="not allowed or rate limited"))
            return
        try:
            snap = await workflow.pool()
        except WorkflowError as exc:
            await ctx.send(sender, PoolStatusResponse(online_peers=0, machines=[], pooled_gib=0, models=[],
                                                      error=exc.message))
            return
        active = snap.active()
        await ctx.send(sender, PoolStatusResponse(
            online_peers=len(snap.online_peers), machines=[n.name for n in snap.nodes if n.online],
            pooled_gib=round(snap.pooled_bytes / GIB, 1), models=[m.id for m in snap.models if m.usable],
            loaded=(active or {}).get("plan", {}).get("model_id")))

    @pool.on_message(RunRequest, replies={RunResponse})
    async def on_run(ctx: Context, sender: str, msg: RunRequest) -> None:
        await ctx.send(sender, await run_for_agent(convo, workflow, settings, sender, msg))

    if settings.mailbox:
        keep_mailbox_alive(agent)
    else:
        silence_almanac_status(agent)
    agent.include(chat, publish_manifest=not local)
    agent.include(payments, publish_manifest=not local)
    agent.include(pool, publish_manifest=not local)

    @agent.on_event("startup")
    async def startup(ctx: Context) -> None:
        ctx.logger.info(f"Hypercompute agent {__version__} address {agent.address}")
        ctx.logger.info(f"wallet {wallet} ({settings.fet_network}); payments {convo.gate.payments}, guests "
                        f"{convo.gate.guest_access}, {len(settings.allowed_senders)} allowlisted sender(s)")
        try:
            health = await gateway.health()
            ctx.logger.info(f"gateway {settings.gateway_url} is up (version {health.get('version')})")
        except Exception as exc:  # startup must not fail on a stopped host; requests report it to the user
            ctx.logger.warning(f"gateway {settings.gateway_url} not reachable yet: {exc}")
        if settings.mailbox and not settings.agentverse_api_key:
            ctx.logger.warning("no AGENTVERSE_API_KEY: the mailbox is not connected, so ASI:One cannot reach me")

    @agent.on_interval(period=60.0)
    async def reap(ctx: Context) -> None:
        try:
            stopped = await workflow.reap_idle()
        except WorkflowError as exc:
            ctx.logger.warning(f"idle check skipped: {exc.message}")
            return
        if stopped:
            ctx.logger.info(f"unloaded idle {', '.join(stopped)} to free friends' machines")

    @agent.on_event("shutdown")
    async def shutdown(_ctx: Context) -> None:
        await gateway.aclose()

    return agent, convo


def keep_mailbox_alive(agent: Agent) -> None:
    """uAgents 0.26.1 drops its HTTP server when a mailbox agent has no REST routes (ours has none, so nothing
    listens locally), and MailboxClient.run() only schedules the polling task and returns, so Agent.run_async
    finishes and shuts the agent down at once. Wrap run() so it stays pending until shutdown cancels it."""
    client = agent._mailbox_client
    if client is None:
        return
    original = client.run

    async def run_until_cancelled() -> None:
        await original()
        await asyncio.Event().wait()

    client.run = run_until_cancelled


async def run_for_agent(convo: Conversation, workflow: Workflow, settings: Settings, sender: str,
                        msg: RunRequest) -> RunResponse:
    """A RunRequest from another agent: allowlisted, rate limited, never charged (so refused when the owner
    charges members), no confirmation step."""
    gate = convo.gate
    if not gate.message(sender).allowed:
        return RunResponse(ok=False, error="rate limited")
    decision = gate.action(sender, "run")
    if decision.role != "member":
        return RunResponse(ok=False, error="not on this pool's allowlist")
    if decision.needs_payment:
        return RunResponse(ok=False, error="this pool charges per request; use the chat protocol to pay")
    if not msg.prompt.strip() or len(msg.prompt) > settings.max_prompt_chars:
        return RunResponse(ok=False, error="bad prompt")
    if workflow.run_lock.locked():
        return RunResponse(ok=False, error="busy")
    wait = gate.start_run(sender)
    if wait is not None:
        return RunResponse(ok=False, error=f"rate limited, retry in {wait:.0f} s")

    async def quiet(_text: str) -> None:
        return None

    model = msg.model if msg.model and msg.model.lower() != "auto" else None
    try:
        intent = Intent(action="run", prompt=msg.prompt, model_hint=model)
        proposal = await workflow.propose(intent)  # outside the lock: propose takes it itself to unload
        tokens = max(16, min(msg.max_tokens or settings.default_max_tokens, settings.max_tokens_cap))
        if workflow.run_lock.locked():
            gate.refund_run(sender)
            return RunResponse(ok=False, error="busy")
        async with workflow.run_lock:
            outcome = await workflow.execute(proposal, msg.prompt, tokens, quiet)
    except WorkflowError as exc:
        if exc.code in ("busy", "no_match", "nothing_fits", "gateway_down", "no_models"):
            gate.refund_run(sender)
        return RunResponse(ok=False, error=exc.message)
    except Exception:
        log.exception("agent run for %s failed", sender[:20])
        gate.refund_run(sender)
        return RunResponse(ok=False, error="internal error")
    layers = {s.node_name: s.layer_count for s in outcome.view.stages if s.layer_count}
    return RunResponse(ok=True, answer=outcome.answer or outcome.reasoning, model=outcome.model.id,
                       decode_tok_s=outcome.decode_tok_s, machines=list(outcome.view.machines), layers=layers)


async def connect_mailbox(agent: Agent, settings: Settings, logger: logging.Logger) -> bool:
    """Register the mailbox with the owner's Agentverse API key, in-process: the same identity proof and agent
    details the inspector's Connect button sends, without serving the open /connect route. Safe on every start (it
    also refreshes the profile and README). It runs before agent.run(): uAgents holds its startup handlers until the
    agent's Almanac status update succeeds, which retries with backoff while the agent is unknown to Agentverse.
    uAgents has no public call for this, so it reads the agent's own identity and details the way its /connect
    handler does (pinned uagents==0.26.1)."""
    details = RegistrationRequest(
        address=agent.address, name=agent.name, handle=agent._handle, url=None,
        profile=agent._build_registration_profile(), endpoints=agent._endpoints,
        protocols=list(agent.protocols.keys()), metadata=agent.metadata)
    request = AgentverseConnectRequest(user_token=settings.agentverse_api_key, agent_type="mailbox")
    try:
        result = await register_in_agentverse(request=request, identity=agent._identity, prefix=agent._prefix,
                                              agentverse=agent._agentverse, agent_details=details)
    except Exception as exc:  # network or Agentverse errors: the agent still starts and says why
        logger.error(f"mailbox registration failed: {type(exc).__name__}")
        return False
    if result.success:
        logger.info("mailbox connected on Agentverse")
    else:
        logger.error(f"mailbox registration refused: {result.detail}")
    return bool(result.success)


def main() -> None:
    load_dotenv(HERE / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        if not Path(settings.data_dir).is_absolute():
            settings = replace(settings, data_dir=str(HERE / settings.data_dir))
        Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
        # local rehearsal mode serves /submit; never beyond this machine (uAgents binds 0.0.0.0 by default)
        uagents_asgi.HOST = "127.0.0.1"
        agent, _ = build(settings)
    except (ConfigError, StoreError) as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    if settings.mailbox and settings.agentverse_api_key:
        agent._loop.run_until_complete(connect_mailbox(agent, settings, log))  # the loop agent.run() then uses
    agent.run()


if __name__ == "__main__":
    main()
