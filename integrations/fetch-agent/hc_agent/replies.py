"""What the agent says back: markdown text that stands on its own, plus an ASI1 Interactive Card when one helps.

ASI:One renders the card; any client that cannot shows the text, so every card's content is also in the text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID, uuid4

from uagents_core.contrib.protocols.chat.cards import (
    CtaAction,
    DetailCardPayload,
    DetailSummaryRow,
    FormCardPayload,
    FormField,
    FormFieldOption,
    ReviewCardPayload,
    ReviewSummaryRow,
)

from .planner import GIB, ModelSummary, PlanView, PoolSnapshot, Stage
from .pricing import Share
from .workflow import Proposal, RunOutcome, WorkflowError

MAX_ANSWER_CHARS = 6000
CardPayload = DetailCardPayload | FormCardPayload | ReviewCardPayload


@dataclass(frozen=True)
class Reply:
    text: str
    card: CardPayload | None = None
    card_id: UUID | None = None
    terminal: bool = False
    end_session: bool = False


def gib(n: int | None) -> str:
    if not n:
        return "0 GiB"
    value = n / GIB
    return f"{value:.1f} GiB" if value < 100 else f"{value:.0f} GiB"


def rate(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f} tok/s" if x < 10 else f"{x:.1f} tok/s"


def fet(x: Decimal, currency: str = "FET") -> str:
    text = f"{x.normalize():f}" if x != 0 else "0"
    return f"{text} {currency}"


def _md(text: str) -> str:
    """Neutralize table pipes in user or model text placed inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def model_label(m: ModelSummary) -> str:
    size = f"{m.params_b / 1000:g}T" if m.params_b >= 1000 else f"{m.params_b:g}B"
    return f"{m.name} ({size}, {m.quantization})" if m.quantization else f"{m.name} ({size})"


def stage_label(s: Stage) -> str:
    who = f"{s.node_name} (host)" if s.is_local else s.node_name
    bits = [s.layer_range() + (f" ({s.layer_count})" if s.layer_count > 1 else "")]
    if s.holds_output:
        bits.append("+ output head")
    if s.file_backed:
        bits.append("streamed from SSD")
    return f"{who} {s.device}: {', '.join(bits)}"


def _stage_rows(view: PlanView) -> list[tuple[str, str]]:
    rows = []
    for s in view.stages:
        if not s.layer_count and not s.weight_bytes:
            continue
        weights = gib(s.observed_bytes) + " observed" if s.observed_bytes else gib(s.weight_bytes) + " planned"
        name = f"{s.node_name} (host)" if s.is_local else s.node_name
        layers = s.layer_range() + (f" ({s.layer_count})" if s.layer_count > 1 else "")
        extra = " + head" if s.holds_output else ""
        rows.append((f"{name} {s.device}", f"{layers}{extra}, {weights}"))
    return rows


def _stage_table(view: PlanView) -> str:
    lines = ["| machine | device | layers | weights |", "|---|---|---|---|"]
    for s in view.stages:
        if not s.layer_count and not s.weight_bytes:
            continue
        name = f"{s.node_name} (host)" if s.is_local else s.node_name
        layers = s.layer_range() + (f" ({s.layer_count})" if s.layer_count > 1 else "")
        if s.holds_output:
            layers += " + output head"
        weights = f"{gib(s.observed_bytes)} (engine log)" if s.observed_bytes else f"{gib(s.weight_bytes)} (plan)"
        lines.append(f"| {_md(name)} | {s.device} | {layers} | {weights} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- help, access


def help_reply(*, member: bool, address: str, guest_access: str) -> Reply:
    lines = [
        "**Hypercompute** runs your prompt on a model too big for any one laptop, by pooling friends' machines "
        "over an encrypted tunnel (llama.cpp RPC over Iroh QUIC).",
        "",
        "Try:",
        "- \u201cRun \u2018explain how a transformer works in 3 sentences\u2019 on the biggest model my friends' laptops "
        "can hold\u201d",
        "- \u201cWho is online in my pool?\u201d",
        "- \u201cUse gpt-oss to answer: what is RPC?\u201d",
        "- \u201cStop\u201d (unloads the model I loaded)",
    ]
    if not member:
        access = {"none": "This pool is private.", "read": "You can see pool status; running prompts is for the "
                  "owner's allowlist.", "paid": "You can run prompts for a small per-request fee in FET."}[guest_access]
        lines += ["", access, f"Your address is `{address}` (the owner adds it to `HC_ALLOWED_SENDERS`)."]
    card = DetailCardPayload(
        title="Hypercompute pool agent",
        summary_rows=[DetailSummaryRow(label="What it does", value="Big-model inference on friends' pooled laptops"),
                      DetailSummaryRow(label="You", value="member" if member else "guest")],
        ctas=[CtaAction(label="Show my pool", selection={"action": "status"}),
              CtaAction(label="Run a prompt", selection={"action": "edit"}, primary=True)])
    return Reply("\n".join(lines), card=card, card_id=uuid4())


def not_allowed_reply(address: str, reason: str) -> Reply:
    return Reply(f"Sorry, I can't do that: {reason}.\n\nThis pool runs prompts only for its owner's friends. Your "
                 f"address is `{address}`; the owner can add it to `HC_ALLOWED_SENDERS`.")


def rate_limited_reply(seconds: float, what: str = "runs") -> Reply:  # what: runs, plans or messages
    minutes = max(1, round(seconds / 60))
    return Reply(f"You've hit the limit for {what}. Try again in about {minutes} minute{'s' if minutes != 1 else ''}.")


def busy_reply(holder: str | None = None) -> Reply:
    who = "your earlier request" if holder == "you" else "another request"
    return Reply(f"The pool is busy with {who} right now. It runs one prompt at a time so friends' laptops are not "
                 f"overloaded. Ask again in a minute.")


# ---------------------------------------------------------------------- pool status


def status_reply(pool: PoolSnapshot) -> Reply:
    host = pool.host
    peers = [n for n in pool.nodes if not n.is_local]
    online = pool.online_peers
    lines = [f"**Pool:** {len(online)} of {len(peers)} friends' machine{'s' if len(peers) != 1 else ''} online, "
             f"{gib(pool.pooled_bytes)} contributed in total (separate memories, not one big one).", "",
             "| machine | state | contributes | link |", "|---|---|---|---|"]
    rows = []
    for n in pool.nodes:
        state = "host" if n.is_local else ("online" if n.online else (n.state_reason or n.state))
        link = "this machine" if n.is_local else (
            f"{n.path_mode}{f', {n.rtt_ms:.0f} ms' if n.rtt_ms is not None else ''}" if n.path_mode else pool.transport)
        devices = "/".join(n.devices) or "none"
        lines.append(f"| {_md(n.name)} | {_md(str(state))} | {gib(n.contribution_bytes)} on {devices} | {link} |")
        rows.append(DetailSummaryRow(label=n.name + (" (host)" if n.is_local else ""),
                                     value=f"{state}, {gib(n.contribution_bytes)} on {devices}, {link}"))
    usable = [m for m in pool.models if m.usable]
    if usable:
        names = ", ".join(model_label(m) for m in sorted(usable, key=lambda m: -m.params_b))
        lines += ["", f"**Models on the host:** {names}"]
    active = pool.active()
    if active:
        lines += ["", f"**Loaded now:** {(active.get('plan') or {}).get('model_id')} ({active.get('state')})"]
    if host is None:
        lines += ["", "The host is not listed; is the gateway running?"]
    card = DetailCardPayload(
        title=f"Pool: {len(online)} friends online, {gib(pool.pooled_bytes)}",
        summary_rows=rows or [DetailSummaryRow(label="Machines", value="none")],
        ctas=[CtaAction(label="Run on the biggest model that fits", selection={"action": "edit"}, primary=True),
              CtaAction(label="Refresh", selection={"action": "status"})])
    return Reply("\n".join(lines), card=card, card_id=uuid4())


def prompt_form(pool: PoolSnapshot | None, *, note: str | None = None) -> Reply:
    options = [FormFieldOption(value="auto", label="Biggest that fits")]
    if pool is not None:
        options += [FormFieldOption(value=m.id, label=model_label(m))
                    for m in sorted((m for m in pool.models if m.usable), key=lambda m: -m.params_b)]
    card = FormCardPayload(
        title="Run a prompt on the pool",
        fields=[FormField(name="prompt", kind="text", label="Prompt", required=True,
                          placeholder="Explain RPC in two sentences"),
                FormField(name="model", kind="select", label="Model", options=options),
                FormField(name="max_tokens", kind="number", label="Max output tokens", placeholder="256")],
        submit_cta=CtaAction(label="Plan it", selection={"action": "run"}, primary=True))
    text = (note + "\n\n" if note else "") + ("What should the model answer? Reply with your prompt, for example "
                                              "\u201cRun \u2018write a haiku about GPUs\u2019\u201d.")
    return Reply(text, card=card, card_id=uuid4())


# ---------------------------------------------------------------------- plan review


def proposal_reply(p: Proposal, *, prompt: str, max_tokens: int, ref: str, price: Decimal | None,
                   currency: str = "FET", needs_confirmation: bool = True, covered: bool = False) -> Reply:
    v = p.view
    friends = v.peer_count
    machines = ", ".join(v.machines) or "the host"
    speed = f"~{rate(v.predicted_tok_s)} (estimate)" if v.predicted_tok_s else "not predicted"
    load = "already loaded, answers right away" if p.reuse else "needs loading first (minutes for big models)"
    tier = f"{v.tier}{', disk-backed capacity mode' if v.capacity_mode else ''}"
    cost = ("covered by your earlier payment" if covered else
            "free (you're on the allowlist)" if price is None else fet(price, currency))
    short = prompt if len(prompt) <= 160 else prompt[:157] + "..."
    lines = [f"**Plan:** {model_label(p.model)} on {len(v.machines)} machine{'s' if len(v.machines) != 1 else ''} "
             f"({friends} friend{'s' if friends != 1 else ''} + host): {machines}.", "", _stage_table(v), "",
             f"- Prompt: \u201c{short}\u201d (up to {max_tokens} tokens)", f"- Tier: {tier}",
             f"- Speed: {speed}", f"- Load: {load}", f"- Price: {cost}"]
    lines += [f"- Note: {n}" for n in (*p.notes, *p.alternatives)]
    if needs_confirmation:
        lines += ["", "Reply **run it** to go, or **cancel**."]
    rows = [ReviewSummaryRow(label="Model", value=model_label(p.model)),
            ReviewSummaryRow(label="Prompt", value=short),
            *[ReviewSummaryRow(label=k, value=val) for k, val in _stage_rows(v)],
            ReviewSummaryRow(label="Speed", value=speed),
            ReviewSummaryRow(label="Load", value=load),
            ReviewSummaryRow(label="Price", value=cost)]
    card = ReviewCardPayload(
        title=f"Run on {p.model.name} across {len(v.machines)} machines?",
        summary_rows=rows,
        approve_cta=CtaAction(label="Run it" if price is None else f"Pay {fet(price, currency)} and run",
                              selection={"action": "approve", "ref": ref}, primary=True),
        reject_cta=CtaAction(label="Cancel", selection={"action": "cancel", "ref": ref}))
    return Reply("\n".join(lines), card=card if needs_confirmation else None,
                 card_id=uuid4() if needs_confirmation else None)


# ---------------------------------------------------------------------- results


def result_reply(r: RunOutcome, *, shares: Sequence[Share] = (), paid: Decimal | None = None,
                 currency: str = "FET") -> Reply:
    answer = r.answer
    if not answer and r.reasoning:
        reasoning = r.reasoning.strip()
        excerpt = reasoning if len(reasoning) <= 1200 else reasoning[:1197] + "..."
        answer = (f"_The model was still thinking when it reached the token limit; here is its reasoning so far:_"
                  f"\n\n> {excerpt}\n\nAsk again with more tokens (\u201cin 600 tokens\u201d) for the final answer.")
    elif not answer:
        answer = "_The model returned an empty answer._"
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS - 3] + "..."
    v = r.view
    load = f"loaded in {r.load_s:.0f} s" if r.load_s is not None else "was already loaded"
    participation = ("every machine's share was confirmed in the engine's load log"
                     if r.participation_verified else (r.participation_detail or "participation not verified"))
    lines = [answer, "", "---",
             f"**Served by {model_label(r.model)}** across {len(v.machines)} machine"
             f"{'s' if len(v.machines) != 1 else ''} ({load}).", "", _stage_table(v), "",
             f"- Decode: **{rate(r.decode_tok_s)}** measured on this request ({r.output_tokens or '?'} tokens)",
             f"- Prompt processing: {rate(r.prefill_tok_s)} ({r.prompt_tokens or '?'} tokens)",
             f"- Time: {r.total_s:.1f} s for the answer",
             f"- Placement: {participation}"]
    if r.finish_reason == "length":
        lines.append("- Stopped at the token limit")
    lines += [f"- Note: {n}" for n in r.notes]
    if paid is not None and shares:
        split = ", ".join(f"{s.node_name} {fet(s.amount, currency)} ({s.layers} layers)" for s in shares)
        lines += ["", f"**Paid {fet(paid, currency)}.** Credited by layers served: {split}."]
    rows = [DetailSummaryRow(label="Model", value=model_label(r.model)),
            DetailSummaryRow(label="Decode (measured)", value=rate(r.decode_tok_s)),
            DetailSummaryRow(label="Prompt processing", value=rate(r.prefill_tok_s)),
            *[DetailSummaryRow(label=k, value=val) for k, val in _stage_rows(v)]]
    if paid is not None and shares:
        rows += [DetailSummaryRow(label=f"Credit: {s.node_name}", value=f"{fet(s.amount, currency)}, {s.layers} layers")
                 for s in shares]
    card = DetailCardPayload(
        title=f"{r.model.name}: {rate(r.decode_tok_s)} across {len(v.machines)} machines",
        summary_rows=rows,
        ctas=[CtaAction(label="Run another prompt", selection={"action": "edit"}, primary=True),
              CtaAction(label="Unload the model", selection={"action": "stop"})])
    return Reply("\n".join(lines), card=card, card_id=uuid4())


def error_reply(e: WorkflowError) -> Reply:
    text = f"**I couldn't finish that.** {e.message}"
    if e.fix:
        text += f"\n\nNext: {e.fix}."
    return Reply(text)


def progress_reply(text: str) -> Reply:
    return Reply(f"_{text}_")
