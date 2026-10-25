import httpx
import pytest

from hc_agent.intent import extract_prompt, parse_intent, parse_rules


@pytest.mark.parametrize("text,prompt", [
    ("Run 'write a haiku about GPUs' on the biggest model my friends' laptops can hold", "write a haiku about GPUs"),
    ('run "explain quicksort in two lines" on the biggest model', "explain quicksort in two lines"),
    ("Please run the following prompt on the largest model in my pool: what is RPC?", "what is RPC?"),
    ("Ask the biggest model my friends' laptops can hold to write a limerick about Wi-Fi",
     "write a limerick about Wi-Fi"),
    ("run explain transformers simply on the biggest model you can", "explain transformers simply"),
    ("Prompt: summarize the history of Unix", "summarize the history of Unix"),
    ("use kimi to answer: why is the sky blue?", "why is the sky blue?"),
])
def test_extracts_the_prompt(text: str, prompt: str) -> None:
    assert extract_prompt(text) == prompt
    intent = parse_rules(text)
    assert intent.action == "run" and intent.prompt == prompt


def test_pool_request_without_a_prompt_asks_for_one() -> None:
    intent = parse_rules("run something on the biggest model my friends' laptops can hold")
    assert intent.action == "run" and intent.prompt is None


def test_plain_question_runs_as_is() -> None:
    intent = parse_rules("Explain how a CPU cache works")
    assert intent.action == "run" and intent.prompt == "Explain how a CPU cache works"


@pytest.mark.parametrize("text,action", [
    ("who is online in my pool?", "status"), ("status", "status"), ("show my pool", "status"),
    ("which machines are online", "status"), ("help", "help"), ("hi", "help"), ("", "help"),
    ("yes", "approve"), ("run it", "approve"), ("Go ahead!", "approve"), ("cancel", "cancel"), ("no", "cancel"),
    ("stop", "stop"), ("unload the model", "stop"),
])
def test_actions(text: str, action: str) -> None:
    assert parse_rules(text).action == action


def test_hints_preference_scope_tokens() -> None:
    i = parse_rules("use deepseek to answer 'what is entropy' in 300 tokens, fastest please")
    assert (i.model_hint, i.preference, i.max_tokens, i.prompt) == ("deepseek", "fastest", 300, "what is entropy")
    assert parse_rules("run 'hello there' only on my laptop").scope == "local"
    assert parse_rules("run 'hello there' on a 120B model").size_hint_b == 120
    assert parse_rules("run 'hello there' on a 1T model").size_hint_b == 1000
    assert parse_rules("answer 'hello there' in 100 words").max_tokens == 140


def test_card_selection_json() -> None:
    i = parse_rules('{"action": "run", "prompt": " hi there ", "model": "auto", "max_tokens": "128"}')
    assert (i.action, i.prompt, i.model_hint, i.max_tokens, i.source) == ("run", "hi there", None, 128, "card")
    i = parse_rules('{"action": "approve", "ref": "abc123"}')
    assert (i.action, i.ref) == ("approve", "abc123")
    assert parse_rules('{"action": "format_disk"}').action != "format_disk"


async def test_asi1_parse_used_when_key_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer k"
        content = 'Sure: {"action": "run", "prompt": "what is RPC", "preference": "biggest", "model": null}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        i = await parse_intent("hey could you get my pool's largest model to tell me what is RPC", asi1_key="k",
                               client=client)
    assert (i.action, i.prompt, i.source) == ("run", "what is RPC", "asi1")


async def test_asi1_failure_falls_back_to_rules() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        i = await parse_intent("run 'what is RPC' on the biggest model", asi1_key="k", client=client)
    assert (i.prompt, i.source) == ("what is RPC", "rules")


async def test_approve_never_goes_to_asi1() -> None:
    def boom(_r: httpx.Request) -> httpx.Response:
        raise AssertionError("asi1 must not be called for a plain approval")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        assert (await parse_intent("yes", asi1_key="k", client=client)).action == "approve"


@pytest.mark.parametrize("text", [
    'What does the phrase "break a leg" mean?',
    "Explain GLM regression in simple terms",
    "Is philosophy useful for engineers?",
    "How does llama.cpp split a model over RPC?",
])
def test_questions_with_quotes_or_model_words_run_whole(text: str) -> None:
    intent = parse_rules(text)
    assert (intent.action, intent.prompt, intent.model_hint) == ("run", text, None)


def test_prompt_formatting_is_kept() -> None:
    intent = parse_rules('run "Fix this:\n  def f():\n      return 1" on the biggest model')
    assert intent.prompt == "Fix this:\n  def f():\n      return 1"


def test_absurd_token_counts_are_clamped() -> None:
    assert parse_rules('{"action": "run", "prompt": "x", "max_tokens": 1e999}').max_tokens is None
    assert parse_rules('{"action": "run", "prompt": "x", "max_tokens": "99999999"}').max_tokens == 100_000
