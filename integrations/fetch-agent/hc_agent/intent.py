"""What the user wants, from a chat message, a card response or ASI:One's rephrasing of either.

Rules handle the common shapes deterministically. With an ASI:One API key, asi1 extracts the same fields first and
the rules fill whatever it leaves out or gets malformed; any asi1 failure falls back to the rules alone.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any

import httpx

log = logging.getLogger(__name__)

ACTIONS = ("run", "status", "stop", "help", "approve", "cancel", "edit")
PREFERENCES = ("biggest", "fastest")
SCOPES = ("pool", "local")
FAMILIES = ("kimi", "deepseek", "gpt-oss", "qwen", "llama", "mistral", "gemma", "phi", "glm", "minimax")
_FAMILY_RE = re.compile(r"(?<![A-Za-z])(kimi|deepseek|gpt-oss|qwen|llama(?!\.cpp)|mistral|gemma|phi|glm|minimax)"
                        r"(?![A-Za-z])", re.I)
# a model family counts as a hint only inside an instruction ("use kimi", "on deepseek"), never in a question
_RUN_CUE = re.compile(r"\b(?:run|use|using|ask|via)\b|\b(?:on|with|through)\s+(?:the\s+|a\s+)?(?:kimi|deepseek|"
                      r"gpt-oss|qwen|llama|mistral|gemma|phi|glm|minimax)", re.I)
# words outside a quoted span that make it a wrapper ("run 'x' on the biggest model"), not part of a question
_INSTRUCTION = re.compile(r"\b(?:run|ask|answer|send|execute|use|using|prompt|query|model|llm|pool|laptops?|"
                          r"machines?|computers?|hypercompute)\b", re.I)

_QUOTED = re.compile(r"```(.+?)```|\"([^\"]{2,})\"|“([^”]{2,})”|‘([^’]{2,})’|"
                     r"(?<![A-Za-z])'([^']{2,})'(?![A-Za-z])", re.S)
_MARKER = re.compile(r"\b(?:prompt|question|query|task)\s*(?:is|:|-)\s*(.+)$", re.I | re.S)
_AFTER_COLON = re.compile(r"^[^:]{0,160}?\b(?:run|ask|answer|send|execute|use)\b[^:]{0,160}:\s*(.+)$", re.I | re.S)
_ASK_TO = re.compile(r"\b(?:ask|tell|get|have|use)\b.{0,120}?\b(?:model|llm|pool|laptops?|machines?|computers?|"
                     r"hold|fit|run)\b\s*(?:,\s*)?(?:to|and)\s+(.+)$", re.I | re.S)
_RUN_ON = re.compile(r"^\s*(?:please\s+)?(?:run|ask|answer|send)\s+(.+?)\s+(?:on|using|with|through|via)\s+(?:the\s+)?"
                     r"(?:biggest|largest|best|most|smartest|strongest|fastest|my|our|a|an|kimi|deepseek|gpt)\b.*$",
                     re.I | re.S)
_POOL_CUE = re.compile(r"\b(biggest|largest|friends?'?|laptops?|pool|machines|hypercompute|model\s+(?:that|which|my)|"
                       r"can hold|can fit)\b", re.I)
_STATUS = re.compile(r"\b(status|who'?s online|who is online|what'?s online|which (?:machines|peers|laptops|computers)"
                     r"|show (?:me )?(?:my |the )?pool|pool (?:status|health)|peers? online|is anyone online|"
                     r"how big|what can (?:my|the) pool|what models?)\b", re.I)
_STOP = re.compile(r"^\s*(?:please\s+)?(?:stop|unload|shut ?down|free (?:up )?the pool|release the pool)\b", re.I)
_HELP = re.compile(r"^\s*(?:help|\?|what can you do|how does this work|what do you do|hi|hello|hey)\s*[.!?]*\s*$", re.I)
_APPROVE = re.compile(r"^\s*(?:yes|yep|yeah|y|approve[d]?|confirm(?:ed)?|run it|go|go ahead|do it|ok(?:ay)?|sure|"
                      r"start|launch)\s*[.!]*\s*$", re.I)
_CANCEL = re.compile(r"^\s*(?:no|nope|n|cancel(?:led)?|reject(?:ed)?|never ?mind|abort|don'?t)\s*[.!]*\s*$", re.I)
_PLACEHOLDER = re.compile(r"^(?:(?:a|an|my|this|that|the|some|any)\s+)?(?:something|anything|prompt|question|query|"
                          r"task|request|text|stuff|it|this|that|one)(?:\s+(?:for me|please))?$", re.I)
_TOKENS = re.compile(r"\b(\d{2,5})\s*(tokens?|words?)\b", re.I)
_SIZE = re.compile(r"\b(\d+(?:\.\d+)?)\s*([bt])\b", re.I)


@dataclass(frozen=True)
class Intent:
    action: str
    prompt: str | None = None
    preference: str = "biggest"
    scope: str = "pool"
    model_hint: str | None = None  # a family or catalog id fragment the user named
    size_hint_b: float | None = None  # "120B", "1T"
    max_tokens: int | None = None
    ref: str | None = None  # card reference (plan ref) when the message is a card response
    source: str = "rules"


PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`"))


def _clean(text: str) -> str:
    """A prompt as the user wrote it: outer whitespace and one pair of wrapping quotes removed, nothing else (line
    breaks, indentation and lists are kept)."""
    s = text.strip()
    for left, right in PAIRS:
        if len(s) >= 2 and s[0] == left and s[-1] == right:
            s = s[1:-1].strip()
    return s


def extract_prompt(text: str) -> str | None:
    """The part of a request that is meant for the model, or None when the request carries no prompt."""
    m = _QUOTED.search(text)
    if m:
        outside = (text[:m.start()] + " " + text[m.end():]).strip()
        if not outside or _INSTRUCTION.search(outside):  # else the quote is part of a question: keep looking
            found = next(g for g in m.groups() if g)
            return _clean(found) or None
    for pattern in (_MARKER, _AFTER_COLON, _ASK_TO, _RUN_ON):
        m = pattern.search(text)
        if m and len(_clean(m.group(1))) >= 2:
            found = _clean(m.group(1))
            return None if _PLACEHOLDER.match(found) else found
    return None


def _model_hints(text: str) -> tuple[str | None, float | None]:
    m = _FAMILY_RE.search(text) if _RUN_CUE.search(text) else None
    family = m.group(1).lower() if m else None
    size = None
    for number, unit in _SIZE.findall(text):
        value = float(number) * (1000.0 if unit.lower() == "t" else 1.0)
        if 0.1 <= value <= 5000:
            size = value
    return family, size


def _max_tokens(text: str) -> int | None:
    m = _TOKENS.search(text)
    if not m:
        return None
    n = int(m.group(1))
    return int(n * 1.4) if m.group(2).lower().startswith("word") else n


def _from_selection(sel: dict[str, Any]) -> Intent | None:
    action = str(sel.get("action", "")).lower()
    if action not in ACTIONS:
        return None
    prompt = sel.get("prompt")
    prompt = _clean(prompt) if isinstance(prompt, str) and prompt.strip() else None
    preference = str(sel.get("preference", "biggest")).lower()
    scope = str(sel.get("scope", "pool")).lower()
    hint = sel.get("model")
    hint = str(hint).lower() if isinstance(hint, str) and hint.strip() and hint != "auto" else None
    tokens = sel.get("max_tokens")
    try:
        tokens = max(1, min(int(float(tokens)), 100_000)) if tokens not in (None, "") else None
    except (TypeError, ValueError, OverflowError):
        tokens = None
    ref = sel.get("ref")
    return Intent(action=action, prompt=prompt, preference=preference if preference in PREFERENCES else "biggest",
                  scope=scope if scope in SCOPES else "pool", model_hint=hint, max_tokens=tokens,
                  ref=str(ref) if isinstance(ref, str) else None, source="card")


def parse_selection_text(text: str) -> Intent | None:
    """ASI:One sends a card's selection as a JSON string when the user @mentions the agent."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        sel = json.loads(stripped)
    except ValueError:
        return None
    return _from_selection(sel) if isinstance(sel, dict) else None


def parse_rules(text: str) -> Intent:
    text = text.strip()
    card = parse_selection_text(text)
    if card is not None:
        return card
    if not text or _HELP.match(text):
        return Intent(action="help")
    if _APPROVE.match(text):
        return Intent(action="approve")
    if _CANCEL.match(text):
        return Intent(action="cancel")
    if _STOP.match(text):
        return Intent(action="stop")
    lowered = text.lower()
    preference = "fastest" if re.search(r"\b(fastest|quick(?:est|ly)?|fast)\b", lowered) else "biggest"
    scope = "local" if re.search(r"\b(only (?:on )?my (?:own )?(?:machine|laptop|computer)|locally|"
                                 r"just (?:on )?(?:my|this) (?:machine|laptop|computer)|no peers)\b", lowered) else "pool"
    prompt = extract_prompt(text)
    family, size = _model_hints(text.replace(prompt, " ") if prompt else text)  # hints live outside the prompt
    if prompt is None and _STATUS.search(text):
        return Intent(action="status", model_hint=family, size_hint_b=size)
    if prompt is None and not _POOL_CUE.search(text) and family is None:
        prompt, size = _clean(text), None  # a plain question routed to this agent: run it as is
    return Intent(action="run", prompt=prompt, preference=preference, scope=scope, model_hint=family,
                  size_hint_b=size, max_tokens=_max_tokens(text))


ASI1_SYSTEM = (
    "You extract a structured request for a compute-pool agent. The agent can: run a prompt on a large language "
    "model hosted on the user's friends' pooled machines (action run), show which machines are online (status), "
    "stop the model it loaded (stop), explain itself (help), approve or cancel a pending plan (approve, cancel). "
    "Reply with ONLY a JSON object with keys: action (run|status|stop|help|approve|cancel), prompt (the exact text "
    "the user wants the model to answer, copied verbatim without the instructions about which model or machines to "
    "use, or null), preference (biggest|fastest), scope (pool|local), model (a model family the user named such as "
    "kimi, deepseek, gpt-oss, qwen, or null), max_tokens (integer or null)."
)


async def parse_with_asi1(text: str, *, api_key: str, model: str, url: str,
                          client: httpx.AsyncClient | None = None) -> Intent | None:
    """asi1's reading of the request, or None on any failure (network, quota, malformed JSON)."""
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "system", "content": ASI1_SYSTEM}, {"role": "user", "content": text[:6000]}]}
    owns = client is None
    client = client or httpx.AsyncClient(timeout=20.0, trust_env=False)
    try:
        r = await client.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code != 200:
            log.warning("asi1 intent parse refused: HTTP %s", r.status_code)
            return None
        content = r.json()["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{.*\}", content, re.S)
        sel = json.loads(m.group(0)) if m else None
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        log.warning("asi1 intent parse failed: %s", type(exc).__name__)
        return None
    finally:
        if owns:
            await client.aclose()
    if not isinstance(sel, dict):
        return None
    parsed = _from_selection(sel)
    return replace(parsed, source="asi1") if parsed else None


async def parse_intent(text: str, *, asi1_key: str | None = None, asi1_model: str = "asi1",
                       asi1_url: str = "https://api.asi1.ai/v1/chat/completions",
                       client: httpx.AsyncClient | None = None) -> Intent:
    rules = parse_rules(text)
    if rules.source == "card" or rules.action in ("approve", "cancel", "help") or not asi1_key:
        return rules
    llm = await parse_with_asi1(text, api_key=asi1_key, model=asi1_model, url=asi1_url, client=client)
    if llm is None:
        return rules
    # asi1 decides the action and prompt; the rules keep hints asi1 did not return
    return replace(llm, prompt=llm.prompt or (rules.prompt if llm.action == "run" else None),
                   model_hint=llm.model_hint or rules.model_hint, size_hint_b=rules.size_hint_b,
                   max_tokens=llm.max_tokens or rules.max_tokens)
