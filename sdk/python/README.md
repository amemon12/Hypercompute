# hypercompute-sdk

A small Python client for the Hypercompute gateway. It plans, deploys, chats, streams and stops a model on the pool
that `hypercompute host` runs. It depends on `httpx` only.

Methods return the gateway's JSON as plain dicts and lists. `backend/hypercompute/schemas.py` stays the only schema.
The SDK adds one type, `ChatChunk`, for streamed chat.

## Install

From the repo root:

```
uv pip install ./sdk/python
```

## Where it runs

Owner routes answer only this machine. The gateway checks that the TCP client is `127.0.0.1` or `::1` and that the
`Host` header is `127.0.0.1`, `localhost` or `[::1]`. So `Pool` accepts only `http://` on one of those names and
raises `ValueError` for anything else, such as a LAN address or `https://`.

To drive a host from another machine, tunnel to it and keep the URL on loopback:

```
ssh -L 8001:127.0.0.1:8001 <host>
```

The default URL is `http://127.0.0.1:8001`, the port the packaging scripts use. The CLI's own default is 8000, so pass
the URL when the host runs on 8000.

## Quickstart

```python
from hypercompute_sdk import Pool

with Pool() as pool:
    dep = pool.deploy_model("qwen2.5-7b-instruct-q4_k_m", strategy="auto",
                            on_progress=lambda state, progress: print(state, progress or ""))
    reply = pool.chat([{"role": "user", "content": "Say hello."}], max_tokens=32)
    print(reply["choices"][0]["message"]["content"])
    for chunk in pool.stream([{"role": "user", "content": "Count to five."}], max_tokens=64):
        print(chunk.content, end="", flush=True)
    pool.stop(dep["id"])
```

`examples/quickstart.py` does the same steps one by one (plan, deploy, wait, chat, stream, stop) and stops only the
deployment it started:

```
uv run --no-sync python sdk/python/examples/quickstart.py qwen2.5-7b-instruct-q4_k_m --strategy auto
```

## Seed peers before loading a large model

`seed_and_run.py` asks each managed peer to fetch its assigned weights, then deploys, chats, streams and stops.
It runs against the host gateway on `http://127.0.0.1:8001`. Start with a downloaded small model:

```sh
uv run --no-sync python sdk/python/examples/seed_and_run.py qwen2.5-0.5b-instruct-q4_k_m
```

On the GX10, with the required model downloaded and all peers eligible:

```sh
uv run --no-sync python sdk/python/examples/seed_and_run.py deepseek-v3.1-ud-tq1_0 --max-tokens=400
```

This example stops its deployment after the answers unless you pass `--keep-running`. That flag keeps a
successful deployment loaded so the demo app can show its layer bar and chat with it. Both chat and streaming
must return visible answer text. Reasoning-only or empty output fails the check and stops the deployment, even
with `--keep-running`. Increase `--max-tokens=N` if a reasoning model exhausts its budget before answering; the
same budget applies to each chat and stream call.

`--seed-timeout=SECONDS` changes the default 3600-second seeding deadline. Large loads take minutes.
The example refuses `gpu_disk` and `stream` tiers by default; keep that
protection on the GX10. Use `--layout=FILE` only for a layout the planner has checked against current free memory.
The example tracks each peer's own seed job and retries a busy peer before loading. Failed or refused jobs stop
the run. A small model with no tensors large enough for managed seeding loads through the normal transfer path.

## The owner token

Reads need no token. Writes (`plan`, `deploy`, `deploy_model`, `stop`, `chat`, `stream`) send
`X-Hypercompute-Owner`. The token comes from, first match wins:

1. the `owner_token=` argument;
2. `HYPERCOMPUTE_OWNER_TOKEN`;
3. `GET /api/owner/bootstrap`, fetched once, just before the first write.

The host makes a new token every time it starts. When the token came from bootstrap and a write gets 403
`owner_only`, the SDK fetches the new token and retries that request once. A token from the argument or the
environment is never replaced; you get `OwnerAuthError`.

The token never appears in `repr(pool)`, in an error message or in a log line. `repr(pool)` shows where it came from:
`argument`, `environment`, `bootstrap` or `none yet`.

`Pool.from_env()` reads `HYPERCOMPUTE_URL` and `HYPERCOMPUTE_OWNER_TOKEN`.

## Methods

| Method | Route |
|---|---|
| `health()` | `GET /api/health` |
| `status()` | health, network, nodes, deployments and `/v1/models`, folded into one dict without Iroh ids |
| `nodes()`, `peers()`, `models()`, `pools()` | `GET /api/nodes`, `/api/peers`, `/api/models`, `/api/capacity/pools` |
| `served_models()` | `GET /v1/models`, as a list of ids |
| `explain(model_id, ctx_size=4096, prompt_tokens=512, output_tokens=256, warm=False, slots=2)` | `GET /api/capacity/explain` |
| `plan(model_id, strategy="all_selected", ctx_size=4096, ...)` | `POST /api/plans` |
| `deploy(plan_id)` or `deploy(plan={...})` | `POST /api/deployments`, always with an `Idempotency-Key` |
| `deployments()`, `deployment(dep_id)` | `GET /api/deployments`, `GET /api/deployments/{id}` |
| `wait_ready(dep_id, timeout=7200, poll=2, on_progress=None)` | polls the deployment until `ready` |
| `deploy_model(model_id, wait=True, **plan_options)` | plan, deploy, wait; plans once more after `plan_stale` |
| `chat(messages, model=None, max_tokens=None, temperature=None, **extra)` | `POST /v1/chat/completions` |
| `stream(messages, ...)` | the same route with `stream: true`, as `ChatChunk`s |
| `stop(dep_id=None)` | `DELETE /api/deployments/{id}`; `None` stops the one active deployment |

Arguments are checked before anything is sent, with the gateway's own bounds. A bad argument raises `ValueError`. A
chat body over the gateway's 16384 KiB loopback limit (one image plus text; other clients get 4096 KiB) raises `BodyTooLarge` without being sent. `extra` passes engine-specific
fields such as `seed`, `cache_prompt` or `ignore_eos`.

`deploy_model` never stops a deployment it did not start. If one is active, it raises `ConflictError` with code
`deployment_active`.

`wait_ready` tolerates 4 failed polls in a row (no reply, or a 5xx) and raises on the 5th. A 4xx raises at once.

## Streaming

`stream()` checks its arguments at once and sends the request on the first `next()`. Each `ChatChunk` has `content`,
`reasoning`, `finish_reason`, `timings` (the engine's timings, usually on the last chunk), `usage` and `raw`.

The stream ends at `data: [DONE]`. An `interrupted` error event raises `ChatInterrupted` with `run_id`, `partial_text`
and `partial_truncated`. Any other error event (`busy`, `timeout`, `upstream_error`) raises `StreamError`. A stream that
closes before `[DONE]` and without a `finish_reason` raises `StreamError` with code `incomplete`.

Leaving the loop early closes the response. The gateway then closes the engine stream, so generation stops.

## Errors

Every error is a `HypercomputeError` with `status`, `code`, `detail` and `fix`. `str(err)` reads like
`HTTP 409 plan_stale: plan is older than 10 minutes; check the plan again`, with `fix: ...` on a second line when the
gateway sent one.

| Class | When |
|---|---|
| `GatewayUnreachable` | no reply: connection refused, DNS, timeout (`status` 0) |
| `RequestError` | 400 or 422; FastAPI validation keeps each field in `locs` |
| `OwnerAuthError` | 403 `owner_only` |
| `NotFoundError` | 404 |
| `ConflictError` | 409, such as `deployment_active` or `plan_stale` |
| `BodyTooLarge` | 413, or the SDK's own size check |
| `EngineError` | 500 or 502 |
| `ChatInterrupted` | 502 `interrupted`, or the stream's interrupted event |
| `UnavailableError` | 503, such as `no_ready_deployment` |
| `StreamError` | an error event in a stream, or `incomplete` |
| `DeploymentFailed` | `wait_ready` saw `failed` or `stopped`; `.deployment` has its `error` and `history` |
| `ReadyTimeout` | `wait_ready` ran out of time; `.deployment` is the last one seen |

Any other status raises `HypercomputeError` itself.

## Tests

```
# From the repository root:
uv run --no-sync python -m pytest sdk/python/tests -q
```

Everything goes through `httpx.MockTransport` and `tests/fake_gateway.py`, a small in-memory gateway. `conftest.py`
puts `sdk/python/src` on `sys.path` and makes every socket connect fail, so no test can reach a real gateway.
`test_sdk_routes.py` reads `backend/hypercompute/app.py` as text and checks that every route the SDK calls is there.

## Implementation notes

- Layout follows `docs/PACKAGING-SPEC.md` section 3 (`src/hypercompute_sdk/`, with `sse.py`). `ChatChunk` lives in
  `types.py`.
- The package is not installed into the repo venv. Tests import it through `conftest.py`.
- A non-streamed chat that the engine refuses as busy answers 429 `busy`. The spec's table has no row for 429, so it
  raises plain `HypercomputeError` with code `busy`.
- `Pool` is not meant to be shared across threads while the first bootstrap runs. Use one `Pool` per thread.
