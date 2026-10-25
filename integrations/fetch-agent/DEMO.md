# Demo video plan (3 to 5 minutes)

Record the ASI:One window full screen, with a small inset of the Hypercompute dashboard or a terminal tailing the agent log. Every number on screen must come from the run being recorded.

Before recording:
- Host up on the desktop GPU box with the friends joined; the agent running with the mailbox connected (log line "mailbox connected on Agentverse").
- Your ASI:One address in `HC_ALLOWED_SENDERS`. For the payment segment set `HC_PAYMENTS=everyone` and have stable-testnet FET in the ASI:One wallet.
- Decide whether to show a cold load. Kimi K2.6 took about 6 minutes to load pooled (366 s in the R6 run), so either cut the wait in the edit (say so on screen) or pre-load and show the warm path, then do the cold load off camera.

| time | shot | say |
|---|---|---|
| 0:00 to 0:25 | Title, then the problem: the 1T model is 207 GB even at under 2 bits per weight; a MacBook Air has 16 GB | "Nobody owns a computer that can run a trillion-parameter model. A group of friends does." |
| 0:25 to 0:50 | ASI:One: "Who is online in my pool?" Pool card appears | "ASI:One found the Hypercompute agent on Agentverse. It sees four friends' machines, what each contributes, and whether the link is direct or relayed." |
| 0:50 to 1:40 | "Run 'explain how attention works in 3 sentences' on the biggest model my friends' laptops can hold." Plan review card | "It ranked the models, asked the planner which one fits right now, and shows the split: which laptop holds which layers, and the price." |
| 1:40 to 2:10 | Click Pay and run. Payment card, then progress lines | "Payment Protocol, FET on testnet. The agent checks the transaction on chain before it touches anyone's laptop." |
| 2:10 to 3:10 | The answer, then the served-by table and measured tok/s | "A 1T model answered in this chat. Each friend's share is confirmed in the engine's own load log, the speed is measured on this request, and the fee is credited by layers served." |
| 3:10 to 3:50 | Failure: `hypercompute leave` on one Mac, ask again (rehearse first: it forces a fresh plan and load) | "A friend's laptop left. The agent noticed, replanned without it and still answered." |
| 3:50 to 4:20 | A stranger's ASI:One account asks to run a prompt; refused with their address | "Only the owner's friends can run prompts, with rate limits and one run at a time. The pool is people's personal laptops." |
| 4:20 to 4:45 | README: agent name, address, badges; tests passing | "Agent Chat Protocol, Interactive Cards, Payment Protocol, mailbox on Agentverse, no inbound ports." |

Lines to avoid: anything faster than measured, "one big GPU", or "pooling makes decode faster" (on these MoE models pooled decode was 0.83x to 0.86x of the host alone; prompt processing was 1.5x to 2.4x faster and SSD reads per token fell to 0.054x to 0.38x).

## For judges

The agent is public on Agentverse and reachable from ASI:One.

- Name: `hypercompute`
- Address: `agent1q0kde2w45zr0fg2lczd08lp98ruq5tx08vlsh233nkxyf0fgx2a9x4eclg0`
- Agentverse profile: https://agentverse.ai/agents/details/agent1q0kde2w45zr0fg2lczd08lp98ruq5tx08vlsh233nkxyf0fgx2a9x4eclg0/profile
- Chat on ASI:One directly: https://asi1.ai/ai/agent1q0kde2w45zr0fg2lczd08lp98ruq5tx08vlsh233nkxyf0fgx2a9x4eclg0

Steps:
1. Sign in at https://asi1.ai (free account).
2. Open the "Chat on ASI:One" link above. Or go to https://agentverse.ai, search for `hypercompute`, open the agent and click **Chat with Agent**.
3. Send `help`. The agent explains itself and shows your sender address.
4. Send `who is online in my pool?`. You get the pool status card: machines online, memory each contributes, link type and round trip.
5. Send `Run 'explain how attention works in 3 sentences' on the biggest model my friends' laptops can hold`. As a guest you get a clear refusal with your address, because only the owner's allowlist can run prompts on friends' laptops. To try the full flow (plan card, approval, payment, answer with layers per machine and measured speed), send the owner your address to be added, or ask for a live demo at the table.

## Demo setup: switch from the mock to the real gateway

The public agent currently answers from a mock gateway, labelled `[mock]` in every answer. To point it at the real pool:

1. Stop the agent and the mock (the python processes running `-m hc_agent.agent` and `-m mock_gateway.server`). Run only one copy of the agent: two copies with the same seed split the mailbox.
2. Set `HC_GATEWAY_URL` to the host gateway's loopback address, either in `.env` or in the shell that starts the agent:
   - agent on the host itself: `http://127.0.0.1:<gateway port>`;
   - agent on another machine: open `ssh -N -L 18000:127.0.0.1:<gateway port> <user>@<host>` there and use `http://127.0.0.1:18000`.
3. Add the presenter's ASI:One address to `HC_ALLOWED_SENDERS` (the agent logs `chat from agent1q... (guest)` on their first message).
4. Optional: `HC_DEFAULT_MAX_TOKENS=600` gives the 1T reasoning model room to finish thinking.
5. Start it from this folder: `.venv/Scripts/python -m hc_agent.agent` (`.venv/bin/python` on macOS and Linux). It re-registers its mailbox on every start, which is harmless.
