# Hypercompute agent: run the biggest model your friends' laptops can hold

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

| | |
|---|---|
| Agent name | `hypercompute` |
| Agent address | `agent1q0kde2w45zr0fg2lczd08lp98ruq5tx08vlsh233nkxyf0fgx2a9x4eclg0` |
| Wallet (payments) | `fetch1fhm492sa5ytujehd5gpgeye76hsk50j3z32rxv` (Fetch.ai stable-testnet) |
| Protocols | AgentChatProtocol 0.3.0 (with ASI1 Interactive Cards), AgentPaymentProtocol 0.1.0 (seller), HypercomputePool 0.1.0 |
| Reach it | ASI:One chat, or any agent through Agentverse (mailbox, no inbound port) |

Say this in ASI:One:

> Run "explain how attention works in 3 sentences" on the biggest model my friends' laptops can hold

and the agent:

1. **Looks at the pool.** Which friends' machines are online, over which link (direct or relayed, round trip in ms), and how much memory each one contributes (`GET /api/nodes`, `GET /api/network`).
2. **Picks a model that fits.** It ranks the host's downloaded models biggest first and asks the gateway's planner for a plan-only check of each (`POST /api/plans`, nothing loads) until one fits the machines online right now.
3. **Shows you the plan** in an Interactive Card: which machine gets which layers, the tier, the gateway's speed estimate, and the price. You click **Run it**.
4. **Takes payment** if the owner charges (Payment Protocol, FET on stable-testnet), and checks the transaction on chain before doing anything.
5. **Deploys and runs.** It loads the model across the machines (`POST /api/deployments`), posts progress while it loads, then sends your prompt (`POST /v1/chat/completions`).
6. **Answers** with the model's reply plus who served it: every machine, its layers, the bytes the engine's own load log reported for it, and the decode and prompt speeds measured on your request. If you paid, the fee is credited to each machine by the layers it served.

The whole workflow happens inside one ASI:One conversation. There is no custom frontend.

## Why it exists

The tested 1T-parameter model uses a 193 GiB quantization. It ran across a GX10 host and laptop peers. Hypercompute connects them through encrypted Iroh tunnels and assigns part of the model to each machine. This agent makes that pool usable from a sentence: anyone the owner allows can ask for "the biggest model we can hold" without knowing which laptops are awake, how much memory each has, or how to plan a split.

Measured on the real pool on 2026-10-24 (a desktop GPU box as the host plus four friends' laptops; main repo `docs/research/PAPER-RUNS-2026-10-22.md`, section R6):

| model | decode, host alone | decode, pooled | prompt processing, pooled vs alone |
|---|---|---|---|
| Kimi K2.6, 1T (IQ1_S) | 1.80 tok/s | 1.54 tok/s | 1.50x faster |
| DeepSeek V3.1, 671B | 1.81 tok/s | 1.50 tok/s | 2.41x faster |
| gpt-oss 120B | 56.0 tok/s | not measured | |

Pooling these MoE models costs some decode speed (0.83x to 0.85x) and buys faster prompt processing and far fewer SSD reads per token (0.054x to 0.38x). The agent reports what it measured on each request, and labels the planner's numbers as estimates.

## Using it in ASI:One

| You say | The agent |
|---|---|
| "Who is online in my pool?" | Pool card: each machine, online or stale, memory contributed, link type and round trip, models on the host. |
| "Run 'write a haiku about GPUs' on the biggest model my friends' laptops can hold" | Plan review card, then the answer with layers per machine and tok/s. |
| "Use gpt-oss to answer: what is RPC?" | Same, on the model you named. |
| "Run 'hello' fastest" | Prefers the biggest model that fits on GPUs without disk-backed capacity mode. |
| "Run 'hello' only on my laptop" | Plans on the host alone. |
| "...in 600 tokens" | Raises the output limit (capped by the owner). |
| "Run it" / "cancel" | Approves or drops the plan waiting for you (the card buttons do the same). |
| "Stop" | Unloads the model this agent loaded. It never unloads a model someone else started. |
| "Run something on the biggest model" | A form card asks for the prompt and lets you pick a model. |

Free text is parsed by rules, and by ASI:One's `asi1` model when an `ASI_ONE_API_KEY` is set; card clicks arrive as structured selections.

## Safety

The pool is friends' personal laptops, so the agent is conservative by default:

- **Allowlist.** Only sender addresses in `HC_ALLOWED_SENDERS` run prompts. Anyone else gets pool status at most (`HC_GUEST_ACCESS=read`), or nothing (`none`), or paid runs (`paid`). A refused sender is told their own address, so the owner can add a friend in one step.
- **Rate limits.** A sliding window per sender for runs (default 4 per 10 minutes, guests half) and for messages (20 per minute), plus one run at a time across the whole pool. A second request while one runs gets a clear "busy" reply instead of queueing on friends' machines.
- **Limits on work.** Prompt length and output tokens are capped; load and chat time out, and a timed-out load is stopped.
- **Owner token stays local.** The agent runs on the host and reads the gateway's owner token from `GET /api/owner/bootstrap`, which the gateway serves on loopback only. The token is kept in memory and never logged or sent anywhere. `HC_GATEWAY_URL` must be a loopback address.
- **No inbound port, no local server.** The agent reaches Agentverse through its mailbox and registers that mailbox in-process with the owner's API key. The uAgents inspector is off, so the agent runs no HTTP server at all (its routes are served with open CORS, which would let any web page the owner visits rebind the agent or read its message history). Proxy environment variables are ignored, so the owner token and prompts never leave the machine through a proxy.
- **Never someone else's model.** The agent only stops or replaces deployments it started, and unloads its own after `HC_KEEP_WARM_S` of idle time (15 minutes) to give friends their memory back. Unloading takes the same lock as a run, so it can never cut off an answer, and a stop the host does not confirm is retried rather than forgotten.
- **Strangers cost nothing.** Guests' free text is parsed by rules only (never on the owner's ASI:One key), their plans are rate limited per sender and in total, and errors shown to guests never include engine or network details.
- **Payments are verified on chain and bound to one request.** Transfers to the agent's wallet are public, so a transaction hash alone proves nothing about who paid. Every quote ends in a random tag (1 to 999 micro-FET) that no other open request uses, and a `CommitPayment` is accepted only after the transaction is found on the ledger, succeeded, paid this agent's wallet exactly that amount in the right denomination, was made after the quote, came from the buyer's wallet when the buyer names one, and was never used before.

## Errors and recovery

Every failure becomes a plain reply with a next step:

| What happened | What the agent does |
|---|---|
| A friend's machine went offline between planning and loading | Replans without it and says so. |
| A load failed (for example a peer dropped mid-load) | Replans on the machines still online and retries once, then gives up with the engine's error. |
| The plan aged past the gateway's 10-minute limit while you decided | Plans again before deploying. |
| The pool is loading or serving someone else's model | Uses the loaded model if you did not insist on another, otherwise explains and never unloads it. |
| Nothing fits the machines online | Says how much memory is online and what each model needed. |
| The host restarted (new owner token) | Fetches the new token and retries once. |
| The host is down | "The pool's host is not answering", with the command that starts it. |
| A reasoning model used all its tokens thinking | Shows the reasoning so far and suggests more tokens. |
| A paid run failed, or the agent restarted mid-run | A verified payment is saved the moment it verifies and spent as soon as its answer exists (if sending it fails, the same answer is resent once on "run it", never recomputed), so a failed load or restart leaves it as a credit: "run it" retries the request that failed at no extra cost. A credit never covers a pricier request and pays for exactly one answer. |
| The payment is not on the ledger yet | Keeps checking for 45 s, then keeps the request open instead of cancelling it. |
| A payment arrives for a request that was cancelled or expired | Refuses it and records the claim (hash and claimed amount, unverified) in `agent_data/payment_claims.jsonl`; the owner checks the hash on chain before refunding. A hash that already bought an answer or is a saved credit, in any letter case and across restarts, is never listed, and when a listed hash later pays (someone claimed it first), a `"void": true` row is appended: never refund a hash that has a void row. |
| The ledger file cannot take a paid run's split | The split goes to `agent_data/ledger_fallback.jsonl` instead, so friends' shares are never lost. |
| The same payment confirmation arrives twice | Completes it once. |

Known limits: unverified payment claims are capped per agent run (2,000 rows) and by file size (2 MB). Sender addresses are free to create, so many throwaway senders can fill those caps and a later genuine claim is then only in the agent log (with its full hash). Payouts to friends are recorded, not sent.

## Monetization

Owners can charge per request (`HC_PAYMENTS=everyone`, or `HC_GUEST_ACCESS=paid` to charge only non-friends):

    price (FET) = 0.01 + 0.004 x (max_tokens / 1000) x (model parameters / 100B)

so a 256-token answer from the 1T model costs 0.02024 FET and from the 120B model 0.011229 FET, clamped to [0.01, 5], plus the per-request tag described under Safety (under 0.001 FET). The price grows with what a request costs the pool: model size sets how much memory every machine gives up and how much work each token is, and `max_tokens` bounds how long the pool is held. The coordinating host keeps 10%; the rest is credited to the machines by the layers they served (a laptop that held 3 of 61 layers earns 3/61 of it). Credits accumulate in `agent_data/credits.json` for the owner to pay out; the on-chain payment goes to the agent's wallet. All of it is tunable in `.env`.

## Other agents

Besides chat, the agent speaks a small structured protocol, `HypercomputePool`: `PoolStatusRequest` returns machines, contributed memory and models; `RunRequest {prompt, model, max_tokens}` returns `RunResponse {answer, model, decode_tok_s, machines, layers}`. Runs are for allowlisted agents only, so a planner or research agent of the owner's can call big-model inference on the pool directly.

## Running it

You need a Hypercompute host running on the same machine (`hypercompute host`, with friends joined) and Python 3.10 to 3.13.

    cd integrations/fetch-agent
    uv venv --python 3.12 .venv
    uv pip install --python .venv/Scripts/python.exe -e ".[dev]"     # .venv/bin/python on macOS and Linux
    cp .env.example .env                                              # then fill it in, see below
    .venv/Scripts/python -m hc_agent.agent

1. Create an Agentverse account (code CALHACKS26AV unlocks a free month) and an API key under Profile, API Keys. Put it in `.env` as `AGENTVERSE_API_KEY`. On every start the agent registers its mailbox with it (log line "mailbox connected on Agentverse"); there is no inspector step.
2. Set `HC_AGENT_SEED` to a long random secret. It fixes the agent's address; keep it private.
3. Optionally create an ASI:One API key (asi1.ai, code CALHACKS26) as `ASI_ONE_API_KEY` for better free-text parsing.
4. Message the agent from ASI:One once. It replies with your sender address; add it to `HC_ALLOWED_SENDERS` and restart.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HC_AGENT_SEED` | required | Secret that fixes the agent address and wallet |
| `AGENTVERSE_API_KEY` | | Connects the mailbox automatically |
| `ASI_ONE_API_KEY` | | asi1 parses free-text requests (rules otherwise) |
| `HC_GATEWAY_URL` | `http://127.0.0.1:8000` | The host's gateway, loopback only |
| `HC_ALLOWED_SENDERS` | | Addresses that may run prompts |
| `HC_GUEST_ACCESS` | `read` | `none`, `read` (pool status) or `paid` |
| `HC_PAYMENTS` | `guests` | `off`, `guests` (only paid guests pay) or `everyone` |
| `HC_CONFIRM` | `auto` | Review card before loading: `auto` (when a load or payment is needed), `always`, `never` |
| `HC_STRATEGY` | `all_selected` | Gateway plan strategy; `all_selected` gives every friend layers |
| `HC_DEFAULT_MAX_TOKENS` / `HC_MAX_TOKENS_CAP` | 256 / 1024 | Output tokens |
| `HC_RUNS_PER_WINDOW` / `HC_RATE_WINDOW_S` | 4 / 600 | Runs per sender per window |
| `HC_MESSAGES_PER_MINUTE` | 20 | Flood guard |
| `HC_LOAD_TIMEOUT_S` / `HC_CHAT_TIMEOUT_S` | 2400 / 900 | Give up after |
| `HC_KEEP_WARM_S` | 900 | Unload the agent's model after this idle time (0 keeps it) |
| `HC_REUSE_FOREIGN_READY` | `true` | Answer on a model someone else loaded (never unloading it) |
| `HC_CHAT_EXTRA_JSON` | | Extra engine options, for example `{"chat_template_kwargs": {"thinking": false}}` |
| `HC_SYSTEM_PROMPT` | short and helpful | System message for every request |
| `HC_PRICE_BASE_FET`, `HC_PRICE_RATE_FET`, `HC_PRICE_MIN_FET`, `HC_PRICE_MAX_FET`, `HC_HOST_FEE_SHARE` | 0.01, 0.004, 0.01, 5, 0.10 | Pricing |
| `HC_FET_NETWORK` | `stable-testnet` | Ledger for payment checks |
| `HC_AGENT_PORT` | 8460 | Local port (loopback) |

## Trying it without a pool or an account

A mock gateway reproduces the real API and the real pool's shape (memory budgets and layer counts from a recorded run, the R6 speeds), with switchable failures. A stand-in user agent plays ASI:One locally, clicks the card buttons and pays test payments:

    .venv/Scripts/python -m mock_gateway.server --port 8765 --load-s 8 --chat-delay-s 2
    .venv/Scripts/python scripts/local_chat.py --print-address        # the stand-in's address (random per machine)
    # in another shell, with HC_MAILBOX=false, HC_GATEWAY_URL=http://127.0.0.1:8765,
    # HC_LOCAL_PEERS=<that address>=http://127.0.0.1:8461/submit, HC_ALLOWED_SENDERS=<that address>,
    # HC_PAYMENT_VERIFY=accept-test and HC_PAYMENTS=everyone:
    .venv/Scripts/python -m hc_agent.agent
    .venv/Scripts/python scripts/local_chat.py --agent <agent address> --click "who is online in my pool?" "Run 'what is RPC' on the biggest model my friends' laptops can hold"

Failure scenarios: `POST /mock/scenario` with `{"offline": ["<peer name>"]}`, `{"drop_on_load": "<peer name>"}`, `{"fail_loads": 2}`, `{"foreign": "gpt-oss-120b-mxfp4"}` or `{"reasoning_only": true}`; `{"reset": true}` clears them. `--generic-names` labels the mock's machines gpu-host and laptop-1 to laptop-4.

## Tests

    .venv/Scripts/python -m pytest

The tests cover intent parsing (phrasings, quoted phrases inside questions, formatting kept, card selections, the asi1 path and its fallback), planning against a recorded 1T plan from the real pool, the workflow end to end against the mock (biggest-fit choice, reuse, a reused model unloaded meanwhile, busy pool, a peer leaving, a failed load and its recovery, timeouts, token refresh, idle unload that never interrupts a run, failed stops), the conversation (approval, cancel, forms, allowlist, rate limits, the paid flow, unique quotes, unverified, late, duplicate and replayed payments, credits after a failed or crashed paid run, guests never seeing engine errors), every card against the card protocol's own schema, and the uAgents adapter.

## Layout

| Path | What |
|---|---|
| `hc_agent/agent.py` | uAgent: chat, payment and pool protocols, mailbox connect, idle unload |
| `hc_agent/conversation.py` | Per-sender state machine: plan, approve, pay, run |
| `hc_agent/workflow.py` | The tool calls: pool snapshot, plan-only checks, deploy, recover, chat |
| `hc_agent/planner.py` | Pure logic: rank models, read a plan into per-machine stages |
| `hc_agent/intent.py` | Free text and card selections to an intent |
| `hc_agent/replies.py` | Markdown replies and Interactive Cards |
| `hc_agent/safety.py`, `pricing.py`, `payments.py`, `config.py` | Allowlist and limits, price and credits, on-chain checks, settings |
| `mock_gateway/` | Local stand-in for the gateway |
| `scripts/local_chat.py` | Local stand-in for an ASI:One user |
