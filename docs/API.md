# Hypercompute gateway API (v0.1)

Source of truth for payload shapes: `backend/hypercompute/schemas.py` (Pydantic). All sizes are
bytes, all times are ISO-8601 UTC, unavailable measurements are `null` (never 0).

## Auth model

| Caller | How it authenticates | Allowed from |
| --- | --- | --- |
| Owner dashboard / owner CLI | `GET /api/owner/bootstrap` returns `owner_token`; every non-GET owner request sends header `X-Hypercompute-Owner: <token>` | loopback only, and the `Host` header must be `127.0.0.1:<port>` or `localhost:<port>` (DNS-rebinding guard) |
| Friend node agent | `Authorization: Bearer <credential>` from `POST /api/sessions/join` | private LAN |
| Anyone | `POST /api/sessions/join` with a one-time code (rate limited per IP) | private LAN |

Owner GETs need loopback + Host check only (no header), so the dashboard can use `EventSource`.
Errors are JSON `{"detail": "...", "code": "...", "fix": "..."?}` with a 4xx/5xx status.

Limits applied before routing or auth: request bodies over 256 KiB (join, heartbeat), 16 MiB
(`/v1/chat/completions` from loopback; 4 MiB from other clients) or 64 KiB (everything else) get 413
`body_too_large`; more than 100 concurrent connections get 503 from the server. Every response carries
`X-Frame-Options: DENY`, `Content-Security-Policy: frame-ancestors 'none'`, `X-Content-Type-Options: nosniff` and
`Referrer-Policy: no-referrer`.

## Owner endpoints

- `GET /api/owner/bootstrap` -> `{owner_token, version, gateway_port, lan_addresses: [str]}`
- `GET /api/health` -> `{status: "ok", version, engine: EngineInfo, time}` (also reachable from LAN)
- `POST /api/sessions` body `CreateSessionRequest {name?}` -> `SessionView` (creates the session if none
  is active, always issues a fresh invite; `invite.code` plaintext appears only in this response)
- `POST /api/sessions/{id}/invites` -> `SessionView` with a new `invite`
- `GET /api/sessions/current` -> `SessionView | null` (no invite code)
- `DELETE /api/sessions/{id}` -> `SessionView` (closes session, nodes are told they were removed)
- `GET /api/nodes` -> `NodeView[]` (the coordinator itself is included with `is_local: true`)
- `POST /api/nodes/{id}/verify` -> `NodeView` (re-runs the engine reachability/identity probe)
- `POST /api/nodes/{id}/leave` -> `{ok: true, node: NodeView}` (owner removes a node; a node agent calling it with
  its own bearer credential gets `{ok: true}`)
- `GET /api/models` -> `ModelView[]`
- `POST /api/models/{id}/download` -> `ModelView` (202, progress visible via `GET /api/models`)
- `POST /api/plans` body `PlanRequest` -> `Plan`
- `POST /api/deployments` body `CreateDeploymentRequest {plan_id? | plan?}`, header `Idempotency-Key`
  optional -> `Deployment` (202). One active deployment at a time; 409 if another is active.
- `GET /api/deployments` -> `Deployment[]` (newest first); `GET /api/deployments/{id}` -> `Deployment`
- `DELETE /api/deployments/{id}` -> `Deployment` (draining -> stopped)
- `GET /api/events?since=<seq>` -> `text/event-stream`; each message `event: hc`, `id: <seq>`,
  `data: Event` JSON. Event `type` values: `session.created`, `invite.created`, `node.joined`,
  `node.verified`, `node.state`, `node.left`, `deployment.state`, `deployment.progress`,
  `run.recorded`, `benchmark.state`, `model.download`.
- `POST /api/benchmarks` body `BenchmarkRequest` -> `BenchmarkResult` (202, status `running`)
- `GET /api/benchmarks` -> `BenchmarkResult[]`; `GET /api/benchmarks/{id}` -> `BenchmarkResult`
- `GET /api/runs?limit=50` -> `RunRecord[]` (newest first; failed/cancelled runs are kept)

## Node endpoints (agent)

- `POST /api/sessions/join` body `JoinRequest` -> `JoinResponse` (credential returned once)
- `POST /api/nodes/{id}/heartbeat` body `HeartbeatRequest` -> `HeartbeatResponse`
- `POST /api/nodes/{id}/leave` -> `{ok: true}` (credential is revoked)

## OpenAI-compatible inference (owner, loopback)

- `GET /v1/models` -> `{object: "list", data: [{id, object: "model", owned_by: "hypercompute", deployment_id}]}`
- `POST /v1/chat/completions` -> proxied to the ready deployment's llama-server. With `stream: true`
  the response is the upstream SSE stream passed through unchanged (`data: {chunk}` ... `data: [DONE]`);
  the final chunk carries `usage` and llama.cpp `timings`. Hypercompute adds
  `stream_options.include_usage=true` when absent. Requires `X-Hypercompute-Owner`.
  503 `{code: "no_ready_deployment"}` when nothing is ready; 429 `{code: "busy"}` when the single
  inference slot stays busy for 30 s; 404 `{code: "model_not_found"}` when `model` is set and is not the
  ready deployment's model id (omit `model` to use whatever is ready). User messages may carry
  OpenAI-style `image_url` parts with base64 data URLs (PNG, JPEG or GIF; at most 4 images of 8 MiB and 64
  megapixels each, within the body limit) when the ready model was started with a vision projector; otherwise
  400 `{code: "vision_unavailable"}`. Remote image URLs, audio and video parts are refused at the gateway. See
  [VISION.md](VISION.md).

Run records (`GET /api/runs`) keep the `RunRecord` schema; `model` also carries `architecture`,
`chat_template_present`, `context_length` and `max_output_tokens`, `engine` carries `hosts` (node id, name, OS,
devices and llama.cpp release of every participating machine), and `started_at` is the request start time.

## Deployment lifecycle

`planned -> reserving -> preparing -> loading -> warming -> ready -> draining -> stopped`; any step can
go to `failed` with `error` and `cleanup` set. `participation_verified` is true only after the load log
shows nonzero model buffers on every planned RPC device. `placement` is parsed from llama.cpp's own log.
