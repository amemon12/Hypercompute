# llama-server audit for the Hypercompute gateway

Source pinned: ggml-org/llama.cpp tag `b11770` = commit `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`
(verified: `gh api repos/ggml-org/llama.cpp/git/ref/tags/b11770` -> that sha, type commit; commit count
at that sha = 11770 via the commits API `Link: ...page=11770; rel="last"`).

Method: every file below was read with `gh api -H "Accept: application/vnd.github.raw" "repos/ggml-org/llama.cpp/contents/<path>?ref=39da130..."`.
Nothing was built or executed. All JSON examples are constructed from the serializer code, not captured
from a live server. Paths are repo-relative; line numbers are at `39da130`.

Gap to the handover audit commit `9a3b91c6801e68f412fcf1b6f8e153aa562b8536` (12 commits ahead, 0 behind):
it does touch `tools/server/*` (adds `POST /tts`, `ws /v1/realtime`, a TTS-only "does not support
chat/completions" 501 guard, and sums `usage.completion_tokens` across choices for `n > 1` non-stream).
None of those change the behaviour documented below for single-choice chat streaming, `/health`,
disconnect handling, or the CLI flags.

---

## 0. Facts a gateway must not get wrong (summary)

| Item | Value at b11770 | Evidence |
|---|---|---|
| Default port | **9931** (not 8080) | `common/common.h:630` `int32_t port = 9931;` |
| Default bind | `127.0.0.1` | `common/common.h:652` |
| Default slots | `-np` = **-1 (auto)** -> 4 slots + unified KV | `common/arg.cpp:1405-1406`, `tools/server/server.cpp:169-174` |
| `/health` while loading | **503** `{"error":{"message":"Loading model","type":"unavailable_error","code":503}}` | `tools/server/server-http.cpp:307-329` |
| `/health` ready | **200** `{"status":"ok"}` | `tools/server/server-context.cpp:5288-5299` |
| Load failure | no HTTP error state; process exits with code 1 | `tools/server/server.cpp:507-512` |
| Stream terminator | `data: [DONE]\n\n` (chat + legacy completions only) | `tools/server/server-context.cpp:5070-5083` |
| `usage` in stream | only when `stream_options.include_usage: true` | `tools/server/server-task.cpp:494-506`, `server-schema.cpp:26-29` |
| `timings` in stream | always on the last chunk before `[DONE]` | `tools/server/server-task.cpp:508-510` |
| Disconnect | polled every 1 s, cancels the slot | `server-context.cpp:40`, `server-queue.cpp:551-622` |
| Jinja | **on** by default for the server | `common/common.h:653` |
| Web UI | on by default; `--no-ui` / `--no-webui` disables | `common/arg.cpp:3490-3497` |
| `/metrics` | off by default (`--metrics`) | `common/common.h:688` |
| `/slots` | **on** by default (`--no-slots`) | `common/common.h:686` |

---

## 1. `GET /health` (also `GET /v1/health`)

Routes: `tools/server/server.cpp:278-279`

```cpp
ctx_http.get ("/health",    ex_wrapper(routes.get_health)); // public endpoint (no API key check)
ctx_http.get ("/v1/health", ex_wrapper(routes.get_health)); // public endpoint (no API key check)
```

### Lifecycle (single-model mode, i.e. `-m` given)

`tools/server/server.cpp:493-517`: the HTTP listener starts **before** the model loads:

```cpp
// start the HTTP server before loading the model to be able to serve /health requests
if (!ctx_http.start()) { ... }
...
if (!ctx_server.load_model(params)) {
    clean_up();
    ctx_http.join();
    SRV_ERR("%s", "exiting due to model loading error\n");
    return 1;
}
routes.update_meta(ctx_server);
ctx_http.is_ready.store(true);
SRV_INF("%s", "model loaded\n");
```

| Phase | Result |
|---|---|
| Before listen (arg parsing, `--rpc` connect, model download) | TCP connection refused. No HTTP. |
| Loading (`is_ready == false`) | HTTP **503**, body below |
| Ready | HTTP **200**, `{"status":"ok"}` |
| Load error (bad GGUF, OOM, chat template parse error, ...) | **No HTTP error state.** `clean_up()` stops HTTP, logs `exiting due to model loading error`, `llama_server` returns 1 -> process exit code 1 |
| Sleeping (`--sleep-idle-seconds N`, default -1 = disabled) | still **200** `{"status":"ok"}` (`create_response(true)` bypasses the sleep wait) |

Loading-phase middleware, `tools/server/server-http.cpp:307-329` (runs for every path; only embedded UI
asset paths are let through):

```cpp
auto middleware_server_state = [this](const httplib::Request & req, httplib::Response & res) {
    if (!is_ready.load()) {
        if (frontend_paths.count(req.path)) {
            return true; // frontend asset, allow it to load and show "loading"
        }
        ...
        res.status = 503;
        res.set_content(safe_json_to_str(json {
            {"error", {
                {"message", "Loading model"},
                {"type", "unavailable_error"},
                {"code", 503}
            }}
        }), "application/json; charset=utf-8");
```

Order in the pre-routing handler (`server-http.cpp:331-362`): CORS headers -> `OPTIONS` short-circuit ->
`middleware_server_state` -> `middleware_validate_api_key`. So during load **every** API route
(`/health`, `/v1/models`, `/props`, `/v1/chat/completions`, ...) answers 503, regardless of API key.
JSON key order is insertion order (server `json` is `common_json` backed by `nlohmann::ordered_json`,
`common/json.cpp:14`), so the body is literally
`{"error":{"message":"Loading model","type":"unavailable_error","code":503}}`.

Ready handler, `tools/server/server-context.cpp:5288-5299`:

```cpp
this->get_health = [this](const server_http_req &) {
    // error and loading states are handled by middleware
    auto res = create_response(true);
    ...
    res->ok({{"status", "ok"}});
```

`/health` and `/v1/health` are in `get_public_endpoints` (`server-http.cpp:251-258`), so they skip the
`--api-key` check. README agrees (`tools/server/README.md:479-490`).

Other error shapes a gateway will see:
- Unknown route: 404 `{"error":{"message":"File Not Found","type":"not_found_error","code":404}}` (`server-http.cpp:199-212`).
- Bad API key: 401 `{"error":{"message":"Invalid API Key","type":"authentication_error","code":401}}` (`server-http.cpp:290-302`).
- Generic error envelope `{"error":{"code":N,"message":"...","type":"..."}}` with code mapping in
  `format_error_response` (`tools/server/server-common.cpp`): invalid_request_error 400,
  authentication_error 401, permission_error 403, not_found_error 404, server_error 500,
  not_supported_error 501, unavailable_error 503, exceed_context_size_error 400.

Router mode caveat: if no `-m`, no `-hf` and no docker repo is given, llama-server starts in **router
mode** (`server.cpp:153-155`, log `starting server in router mode. models will be automatically loaded on-demand`
at `server.cpp:441`) and `/health` is 200 immediately with no model. The gateway must always pass `-m`.

---

## 2. `POST /v1/chat/completions`

Route: `server.cpp:288-289` (`/chat/completions` and `/v1/chat/completions`). Handler
`server-context.cpp:5564-5582` -> `oaicompat_chat_params_parse` -> `handle_completions_impl`
(`server-context.cpp:4886-5148`) with `TASK_RESPONSE_TYPE_OAI_CHAT`.

The request's `model` field is **not** validated or used in single-model mode. Every response's
`model` is `meta->model_name` (`server-context.cpp:4954` `task.params.oaicompat_model = meta->model_name;`).

### 2a. Streaming (`"stream": true`)

HTTP framing (`server-http.cpp:635-666`): status 200, `Content-Type: text/event-stream`,
`X-Accel-Buffering: no`, `Server: llama.cpp`, chunked transfer encoding (`set_chunked_content_provider`).

**When the 200 headers go out.** When the slot starts processing the prompt, the server sends an
`is_begin` partial (`server-context.cpp:3924-3932`); its `to_json()` returns `nullptr`
(`server-task.cpp:1031-1033`), and `handle_completions_impl` then sets `res->data = ""; // simply send HTTP headers and status code`
(`server-context.cpp:5028-5029`). So **headers arrive before prompt processing finishes**. Time to headers
is not TTFT. Measure TTFT at the first chunk with non-empty `delta.content` or `delta.reasoning_content`.

**Errors before the first chunk** come back as an ordinary non-SSE JSON error with its HTTP status
(`server-context.cpp:5006-5017`, "in streaming mode, the first error must be treated as non-stream response").
Example: prompt too long -> 400 with `type: exceed_context_size_error` plus `n_prompt_tokens` and `n_ctx`
(`server-task.cpp:1525-1531`).

**Event framing**, `format_oai_sse` (`server-common.cpp:1737-1754`): each JSON object becomes
`data: <json>\n\n`. A result that is a JSON array produces several events in one write.

**Chunk sequence** (constructed from `server-task.cpp:1114-1166` partials and `454-517` final):

1. First generated token (`n_decoded == 1`): a role chunk, then the content diffs for that token:
   ```
   data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":1760000000,"id":"chatcmpl-<32 chars>","model":"<model id>","system_fingerprint":"b11770-<short sha>","object":"chat.completion.chunk"}

   data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Hello"}}],"created":1760000000,"id":"chatcmpl-...","model":"...","system_fingerprint":"b11770-...","object":"chat.completion.chunk"}
   ```
2. Each later token: zero or more chunks with `delta` holding any of `reasoning_content`, `content`,
   `tool_calls` (`server-chat.cpp:621-650`). A token that the parser holds back (partial UTF-8, possible
   tag start) yields an empty array and so **no event**.
3. Final result (`to_json_oaicompat_chat_stream`, `server-task.cpp:454-517`):
   - any remaining diffs as normal chunks with `finish_reason: null`,
   - then the finish chunk: `{"choices":[{"finish_reason":"stop","index":0,"delta":{}}], ...}`
     (`"stop"` for EOS or stop word, `"tool_calls"` if tool calls were parsed, else `"length"`;
     `server-task.cpp:455-459`),
   - then, only if `include_usage`, a usage chunk with **empty `choices`**:
     `{"choices":[],"created":...,"id":...,"model":...,"system_fingerprint":...,"object":"chat.completion.chunk","usage":{...}}`
     (`server-task.cpp:494-506`),
   - `timings` is attached to the **last** of these (`deltas.back()["timings"] = stats.to_json();`,
     `server-task.cpp:508-510`), i.e. the finish chunk without `include_usage`, the usage chunk with it.
4. Terminator: `data: [DONE]\n\n` (`server-context.cpp:5070-5083`). Note `[DONE]` is sent for
   chat and `/v1/completions` and `/completion`; it is **not** sent for `/v1/responses` and `/v1/messages`.
5. Keep-alive pings: if nothing arrives for `sse_ping_interval` seconds (default **30**,
   `common/common.h:634`; flag `--sse-ping-interval`, per-request field `sse_ping_interval`, `-1` disables;
   `server-schema.cpp:40-42`), the server writes an SSE comment `":\n\n"` (`server-context.cpp:5086-5105`).
   SSE parsers must ignore lines starting with `:`.
6. Mid-stream error: `data: {"error":{...}}\n\n` and the stream ends **without** `[DONE]`
   (`server-context.cpp:5040-5048, 5113-5119`).

`finish_reason` placement: always `null` in content chunks; the real value appears only in the dedicated
final chunk whose `delta` is `{}`.

`usage` object (`usage_json_oaicompat`, `server-task.cpp:357-364`):

```json
{"completion_tokens": <n_decoded>, "prompt_tokens": <n_prompt_tokens>, "total_tokens": <sum>,
 "prompt_tokens_details": {"cached_tokens": <n_prompt_tokens_cache>}}
```

`stream_options.include_usage` default is `false` (`server-task.h:53`; schema `server-schema.cpp:26-29`).
So yes, it is needed to get `usage` in streaming mode. `timings` does not need it.

Extra per-request stream knobs (schema `server-schema.cpp:17-42`): `timings_per_token` (adds `timings` to
every partial chunk, `server-context.cpp:2222-2225`), `return_progress` (adds `prompt_progress` chunks
during prompt processing), `verbose`.

### 2b. `timings` object

Serializer `server_slot_stats::to_json`, `tools/server/server-common.cpp:139-160`; struct
`tools/server/server-common.h:398-482`.

| Field | Meaning (from code) |
|---|---|
| `cache_n` | prompt tokens reused from KV cache (`n_prompt_cached`) |
| `prompt_n` | prompt tokens actually processed (**excludes** cached) |
| `prompt_ms` | `(t_prompt_last - t_start)/1000` |
| `prompt_per_token_ms` | `prompt_ms / prompt_n` (0 if unknown) |
| `prompt_per_second` | `1e3 / prompt_ms * prompt_n` |
| `predicted_n` | generated tokens (`n_gen`) |
| `predicted_ms` | `(t_gen_last - t_prompt_last)/1000`, clamped to >= 1 us |
| `predicted_per_token_ms` | `predicted_ms / (n_gen - 1)` |
| `predicted_per_second` | `(n_gen - 1) / predicted_ms * 1e3` |
| `draft_n`, `draft_n_accepted` | only when speculative decoding drafted tokens |

Measurement trap: `predicted_per_second` and `predicted_per_token_ms` use `n_gen_steps() = n_gen - 1`
("the first token is free, it comes from the logits of the last prompt batch", `server-common.h:454-458`).
`prompt_n` excludes cached tokens while `usage.prompt_tokens` includes them.

### 2c. Non-streaming

`to_json_oaicompat_chat`, `server-task.cpp:406-452`:

```json
{"choices":[{"finish_reason":"stop","index":0,
             "message":{"role":"assistant","content":"...","reasoning_content":"..."}}],
 "created":1760000000,"model":"<model id>","system_fingerprint":"b11770-<short sha>",
 "object":"chat.completion","usage":{...},"id":"chatcmpl-...","timings":{...}}
```

- `reasoning_content` is present only when non-empty (`common/chat.cpp:202-204`); `tool_calls` likewise.
- `timings` is present whenever the slot started (`stats.is_set()` = `t_start > 0`).
- `usage` is always present.
- `logprobs` only if requested.
- Non-stream wait also observes disconnect (`rd.wait_for_all(req.should_stop)`, `server-context.cpp:4977`).

---

## 3. Client disconnect detection and generation stop

Chain of evidence:

1. `server_http_req::should_stop` is bound to cpp-httplib's `req.is_connection_closed`
   (`server-http.cpp:677-688, 728-739`; struct field `server-http.h:56`).
2. httplib sets it as `!detail::is_socket_alive(sock)` (`vendor/cpp-httplib/httplib.cpp:10344-10347`).
   `is_socket_alive` does a zero-timeout `select` and then a 1-byte `MSG_PEEK`; a peer FIN (peek returns 0)
   counts as closed. A proxy that **half-closes** its upstream write side would be seen as disconnected.
3. The response reader polls with `HTTP_POLLING_SECONDS = 1` (`server-context.cpp:40`):
   `queue_results.recv_with_timeout(id_tasks, 1)`, then checks `should_stop()` on each timeout
   (`server-queue.cpp:551-579`).
4. Streaming path: each `next()` call first checks `res_this->should_stop()` and stops
   (`server-context.cpp:5051-5059, 5090-5111`). Also, when httplib's `sink.write()` fails, the content
   provider returns false (`server-http.cpp:648-653`), httplib calls `on_complete`, which drops the response.
5. Dropping the response destroys `server_response_reader`, whose destructor calls `stop()`
   (`server-queue.h:218-220`). `stop()` posts `SERVER_TASK_TYPE_CANCEL` for every unfinished task
   at the front of the queue (`server-queue.cpp:603-621`, log `cancel task, id_task = %d`).
6. The main loop releases the slot that owns the target task (`server-context.cpp:2901-2910`,
   "release slot linked with the task id").

So the worst case is about 1 s of extra generation after a disconnect, plus the current decode step.

**Exception: `X-Conversation-Id`.** If the request carries an `X-Conversation-Id` header (case-insensitive),
`set_req` attaches a resumable "spipe" (`server-stream.cpp:597-615`). Then
`server_res_spipe::should_stop()` returns `spipe->is_cancelled()` instead of the connection state
(`server-stream.cpp:622-629`), and `on_complete()` keeps draining generation into a 4 MiB buffer with a
300 s TTL (`server-stream.cpp:17-18, 631-652`). **Generation keeps running after the client leaves.** It is
only cancelled by `DELETE /v1/stream?conv_id=<id>` (204, `server-stream.cpp:560-575`), or replayed via
`GET /v1/stream?conv_id=<id>&from=N`. A gateway that wants "leave cleanly = stop compute" must strip this
header, or call the DELETE on disconnect.

---

## 4. Other endpoints

| Endpoint | Default | Gate | Evidence |
|---|---|---|---|
| `GET /v1/models`, `GET /models` | on | none | `server-context.cpp:5725-5735`, `5196-5227` |
| `GET /props` | on | none (`POST /props` needs `--props`) | `server-context.cpp:5438-5448`, `5450-5460` |
| `GET /slots` | **on** | `--no-slots` disables -> 501 "Start it with `--slots`" | `common/common.h:686`, `server-context.cpp:5363-5402` |
| `GET /metrics` | **off** | `--metrics` enables; otherwise 501 "This server does not support metrics endpoint. Start it with `--metrics`" | `common/common.h:688`, `server-context.cpp:5301-5361` |

`GET /v1/models` body (constructed, `server-context.cpp:5168-5227`): Ollama-style `models: [...]` plus
OpenAI-style `"object":"list","data":[{...}]`, where `data[0]` is:

```json
{"id":"<model_name>","aliases":[...],"tags":[...],"object":"model",
 "architecture":{...},"created":<now>,"owned_by":"llamacpp",
 "meta":{"vocab_type":..,"n_vocab":..,"n_ctx":<slot n_ctx>,"n_ctx_train":..,"n_embd":..,"n_params":..,"size":..,"ftype":".."}}
```

`created` is `std::time(0)` at request time, not load time.

`GET /props` keys (`server-context.cpp:5229-5277`): `default_generation_settings` (`params`, `n_ctx`),
`total_slots` (= effective `n_parallel`), `model_alias` (= model id), `model_ftype`, `model_path`,
`modalities`, `media_marker`, `endpoint_slots`, `endpoint_props`, `endpoint_metrics`, `ui`, `ui_settings`,
`chat_template`, `chat_template_caps`, `bos_token`, `eos_token`, `build_info` (= `b11770-<sha>`),
`is_sleeping`, `cors_proxy_enabled`; plus `chat_template_tool_use` when jinja is on and one exists, and
`connect_code` when `--connect` is used.

`GET /slots`: JSON array, one object per slot: `id`, `n_ctx`, `speculative`, `is_processing`, and when the
slot has had a task, `id_task`, `n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`,
`params`, `next_token[{has_next_token,has_new_line,n_remain,n_decoded}]` (`server-context.cpp:735-765`).
Prompt and generated text are omitted unless env `LLAMA_SERVER_SLOTS_DEBUG` is non-zero
(`server-context.cpp:1468-1469, 2981`). `?fail_on_no_slot=1` returns 503 `no slot available` when no slot
is idle (`server-context.cpp:5393-5399`). This is a cheap busy probe.

`GET /metrics`: Prometheus text, `Content-Type: text/plain; version=0.0.4`, header
`Process-Start-Time-Unix` (`server-context.cpp:5311-5316, 5354-5357`). Metric names
(`server-task.cpp:1547-1636`), all prefixed `llamacpp:`:
- counters: `prompt_tokens_total`, `prompt_tokens_cached_total`, `prompt_seconds_total`,
  `tokens_predicted_total`, `tokens_predicted_seconds_total`, `n_decode_total`, `n_tokens_max`,
  `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total`, `spec_decode_num_drafts_total`
- gauges: `prompt_tokens_seconds`, `predicted_tokens_seconds`, `requests_processing`, `requests_deferred`,
  `n_busy_slots_per_decode`
- labeled: `spec_decode_num_accepted_tokens_per_pos_total{position="i"}` (only if speculative data exists)
- The throughput gauges are averaged over the window between two scrapes, and a scrape resets that bucket
  (`metrics_reset_bucket = true`, `server-context.cpp:5329-5333`). If two scrapers share a server they will
  disturb each other.

Also present (not needed for the MVP): `/v1/completions`, `/completion(s)`, `/v1/responses`,
`/v1/messages`, `/v1/embeddings`, `/tokenize`, `/detokenize`, `/apply-template`,
`/v1/chat/completions/input_tokens`, `/v1/chat/completions/control`, `/lora-adapters`,
`POST /slots/:id_slot`, `/v1/stream` (`server.cpp:278-337`).

---

## 5. Flags

### `--alias` / model id
`common/arg.cpp:3024-3035`: `-a, --alias STRING`, "set model name aliases, comma-separated (to be used by
API)", env `LLAMA_ARG_ALIAS`, stored into `std::set<std::string> model_alias` (`common/common.h:514`).

- `server.cpp:189-193`: if no alias is given, the alias set gets `params.model.get_name()`, which is
  `hf_repo`, else `docker_repo`, else **the `-m` path string as typed** (`common/common.h:312-321`).
- `server-context.cpp:1521-1532`: `model_name = *params_base.model_alias.begin()`, which is the
  **lexicographically smallest** alias since it is a `std::set`. The filename-only fallback is reached only
  when the path is empty.
- So without `--alias`, `"model"` in responses and `data[0].id` in `/v1/models` is the full `-m` path (for
  example `C:\models\qwen.gguf`). With `--alias a,b`, the id is `a`, and `aliases` lists both.
- The request `model` is ignored in single-model mode (section 2), so the gateway can map its own names freely.

Recommendation: always pass exactly one `--alias <stable-id>`.

### `--parallel`
`common/arg.cpp:2545-2556`: `-np, --parallel N`, "number of server slots (default: -1, -1 = auto)",
`0` rejected, env `LLAMA_ARG_N_PARALLEL`. Server default is set to -1 at `arg.cpp:1405-1406`.
Auto resolves to `n_parallel = 4` and `kv_unified = true` (`server.cpp:169-174`).

With `-np 1`: one slot (`server-context.cpp:1392-1398`, log `initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'`),
and the slot context is the whole `llama_n_ctx_seq` capped by `n_ctx_train` (`n_ctx_slot()`).
Extra concurrent requests are **deferred (queued), not rejected** (`server-context.cpp:2852-2854`,
"no slot is available, defer task"). They show up in `llamacpp:requests_deferred`. Use
`/slots?fail_on_no_slot=1` or gateway-side admission control to reject instead.

### `--no-webui`
`common/arg.cpp:3490-3497`: `{"--ui", "--webui"}` / `{"--no-ui", "--no-webui"}`, default enabled, env
`LLAMA_ARG_UI`. When disabled the server logs `The UI is disabled` (`server-http.cpp:371-375`) and serves a
self-removing service worker at `/sw.js`.

### `--jinja` and chat templates
- `common/arg.cpp:3711-3718`: `--jinja` / `--no-jinja`, env `LLAMA_ARG_JINJA`. Default `use_jinja = true`
  (`common/common.h:653`); only the completion and mtmd examples flip it off (`arg.cpp:1400-1404`).
- The template comes from GGUF `tokenizer.chat_template` unless `--chat-template` or `--chat-template-file`
  is given (`server-context.cpp:1604`, `common/chat.cpp:729-757`). If the model has no template, it falls
  back to the `tool_use` template, else ChatML (`common/chat.cpp:752-757`).
- A template parse error makes `load_model` fail. The server logs a hint to try `--no-jinja --chat-template chatml`
  and exits 1 (`server-context.cpp:1615-1619`).
- `tools` or a non-`auto` `tool_choice` without jinja -> error "tools param requires --jinja flag"
  (`server-common.cpp:1341-1348`).
- A trailing assistant message is prefilled by default (`--prefill-assistant`, `server-common.cpp:1431-1438`).
- Per-request `chat_template_kwargs` are merged over CLI defaults (`server-common.cpp:1460-1475`).

---

## 6. `--version` and `--list-devices` output

### `--version`
`common/arg.cpp:1453-1459` calls `llama_print_build_info(llama_version())` then `exit(0)`.
`common/build-info.cpp.in:32-35`:

```cpp
void llama_print_build_info(const char * llama_version, FILE * stream) {
    fprintf(stream, "version: %s (build %d, commit %s)\n", llama_version, llama_build_number(), llama_commit());
    fprintf(stream, "built with %s for %s\n", llama_compiler(), llama_build_target());
}
```

- The default stream is **stderr** (`common/build-info.h:13`, `FILE * = stderr`). Capture stderr, not stdout.
- `llama_version()` returns `LLAMA_VERSION` (`src/llama.cpp:118-120`). That is `0.7.0` plus `-dev` unless
  configured with `-DLLAMA_BUILD_IS_DEV=OFF` (`CMakeLists.txt:5-21`). `.github/workflows/release.yml`
  `CMAKE_ARGS` (line 38) does not set it, so the release binaries print `0.7.0-dev`. Inferred from the
  workflow, not from running a binary.
- Build number = `git rev-list --count HEAD`. Commit = `git rev-parse --short HEAD` (`cmake/build-info.cmake`).
  The release jobs use `fetch-depth: 0`, so the number is 11770 for this tag.
- `BUILD_COMPILER` = `"${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}"`. `BUILD_TARGET` = the VS
  platform name on MSVC, else `"${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}"`.

Expected lines for a b11770 release binary (shape from code; values partly UNVERIFIED):

```
version: 0.7.0-dev (build 11770, commit 39da130)
built with AppleClang 17.0.0.17000013 for Darwin arm64
```

UNVERIFIED: the exact short-sha length. `git rev-parse --short` auto-scales with repository size and may
print more than 7 hex characters. The compiler and target strings depend on the build host. Parse with a
regex such as
`^version: (\S+) \(build (\d+), commit ([0-9a-f]+|unknown)\)$`. Build number 0 and commit `unknown` mean
no git was available at configure time.

`system_fingerprint` in responses and `build_info` in `/props` use a different format:
`"b" + build_number + "-" + commit` (`build-info.cpp.in:27-30`), for example `b11770-39da130`.

### `--list-devices`
`common/arg.cpp:2749-2756` calls `common_print_available_devices()` then `exit(0)`. Implementation at
`common/arg.cpp:1144-1167`:

```cpp
ggml_backend_load_all();
... skip GGML_BACKEND_DEVICE_TYPE_CPU ...
printf("Available devices:\n");
if (devices.empty()) { printf("  (none)\n"); return; }
printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / MiB, free / MiB);
```

- Output goes to **stdout** via `printf`. CPU devices are **never** listed; a CPU-only box prints `  (none)`.
- Device names come from the backends:
  - CUDA: `"CUDA" + i`, description = `cudaDeviceProp.name` plus ` (dev pX/vY)` when virtual devices exist
    (`ggml/src/ggml-cuda/ggml-cuda.cu:5282-5292, 6217-6218`; `GGML_CUDA_NAME` is `"CUDA"`, `"ROCm"` or
    `"MUSA"` in `ggml/include/ggml-cuda.h`).
  - Metal: `"MTL" + i`, description = the GPU name (`ggml/src/ggml-metal/ggml-metal-device.m:1302-1308`).
  - RPC: `"RPC" + global_dev_id`, description = the **endpoint string** as given to `--rpc`
    (`ggml/src/ggml-rpc/ggml-rpc.cpp:3165-3175`). An endpoint can expose more than one device.

Example (constructed; memory numbers illustrative):

```
Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24563 MiB, 23012 MiB free)
  RPC0: 192.168.1.20:50052 (24563 MiB, 22900 MiB free)
  RPC1: 192.168.1.31:50052 (98304 MiB, 90112 MiB free)
```

Ordering gotchas, verified:
- CLI args are handled **in argv order** (`common/arg.cpp:822-880`), and the `--list-devices` handler exits
  immediately. RPC devices are registered inside the `--rpc` handler (`arg.cpp:2684-2693` ->
  `add_rpc_devices`, `arg.cpp:1170-1189`). So `--rpc ... --list-devices` lists the RPC devices;
  `--list-devices --rpc ...` does not. Env `LLAMA_ARG_RPC` is applied before CLI args (`arg.cpp:787-810`),
  so it works in either case.
- An unreachable `--rpc` endpoint is **fatal**: `get_dispatcher` -> `rpc_dispatcher::start` calls
  `GGML_ABORT("Failed to connect to %s\n", ...)` or `GGML_ABORT("RPC handshake failed for %s\n", ...)`
  (`ggml/src/ggml-rpc/ggml-rpc.cpp:638-656`). This happens during argument parsing, before HTTP listens,
  so the gateway sees an abnormal exit and never gets a `/health` response. Preflight each RPC endpoint
  with a TCP connect before launch.
- UNVERIFIED: the exact stderr lines `ggml_backend_load_all()` emits while loading dynamic backends.
  Parse stdout only.

---

## 7. Reasoning / thinking defaults

- `-rea, --reasoning [on|off|auto]`, env `LLAMA_ARG_REASONING`, default `auto` (`common/arg.cpp:3730-3746`,
  `common/common.h:665` `enable_reasoning = -1`). `on` / `off` also set
  `default_template_kwargs["enable_thinking"] = "true"/"false"`.
- Effective thinking at load (`server-context.cpp:1609-1614`):
  `enable_thinking = enable_reasoning != 0 && use_jinja && common_chat_templates_support_enable_thinking(...)`.
  So with defaults (auto + jinja), thinking is **on whenever the template supports it**.
- `--reasoning-format`, env `LLAMA_ARG_THINK`: `none` | `auto` | `deepseek` | `deepseek-legacy`
  (`common/arg.cpp:3719-3729`, `common/chat.cpp:861-875`). The help text says "default: auto", but the
  struct default is `COMMON_REASONING_FORMAT_DEEPSEEK` (`common/common.h:664`). The enum comment says
  AUTO is "Same as deepseek" (`common/common.h:428`). Either way, thoughts are split out of `content`.
- Can be overridden per request (`server-common.cpp:1448-1485`): `reasoning_format`,
  `chat_template_kwargs.enable_thinking` (must be a JSON boolean; a string throws
  `invalid type for "enable_thinking"`), and OpenAI `reasoning_effort` (`"none"` disables thinking).
- Streaming deltas carry thoughts in `delta.reasoning_content`, separate from `delta.content`
  (`server-chat.cpp:621-628`). Non-stream responses use `message.reasoning_content` (`common/chat.cpp:202-204`).
- `--reasoning-budget N` (default -1 unrestricted, env `LLAMA_ARG_THINK_BUDGET`), `--reasoning-budget-message`,
  and `--reasoning-preserve` / `--no-reasoning-preserve` (default enabled when the template supports it)
  are at `arg.cpp:3760-3789`. `POST /v1/chat/completions/control {"id":"<chatcmpl id>","action":"reasoning_end"}`
  ends reasoning on a live completion (`server-context.cpp:5588-5623`). It only works if the original
  request set `"reasoning_control": true` and the template has thinking end tags. The field is read only
  inside `if (!out_session.thinking_end_tags().empty())` (`server-common.cpp:1496-1511`). Otherwise the
  result is `success:false`, "reasoning control not enabled for this completion"
  (`server-context.cpp:2926-2933`).

Gateway implication: forward both `delta.reasoning_content` and `delta.content`. If the UI shows only
`content`, a thinking model will look stalled. To get plain answers from thinking models, pass
`--reasoning off` at launch or `"chat_template_kwargs":{"enable_thinking":false}` per request.

---

## 8. Recommended launch line for the MVP gateway (derived from the above)

```
llama-server -m <model.gguf> --alias <id> --host 127.0.0.1 --port <port> -np 1 --no-webui \
  --metrics --rpc <host1:port,host2:port> -ngl 999
```

(`-ngl` was not audited here; see `docs/research/llama_rpc.md`.) Health-poll `/health` until 200, treat
process exit as a load error, strip `X-Conversation-Id` on proxy, and require `[DONE]` or a finish chunk
for success.

---

## Addendum (critic)

Critic pass 2026-10-20 at the same ref `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`. Nothing was built or run.

### Open items in this file

| Item | Matters for | Status after this pass |
|---|---|---|
| §8 launch line: `-ngl 999` with `--fit` left on, no `-c`, no `--device`, no `-ts`, no `-lv 4` | (a), (b) | SUPERSEDED by the combined line below. With fit on, any user `-ngl` makes fit throw `n_gpu_layers already set by user ... abort`. But the auto `n_ctx` reduction step runs before that check and is kept (common/fit.cpp step 2 at 373-458, user checks at 463-465; llama_placement.md §4). So the context could silently shrink, and load placement lines are hidden at the default verbosity (llama_placement.md §6) |
| §6 "UNVERIFIED: the exact stderr lines `ggml_backend_load_all()` emits" | (d) | RESOLVED in llama_placement.md "Addendum (critic)" P1. The only line on success is `load_backend: loaded <X> backend from <path>` at GGML INFO, hidden in llama-server unless `-lv` is 4 or more. Failures are silent in release builds. `--list-devices` output on stdout stays clean as long as `--log-jsonl` is not passed (P4) |
| §6 short-sha length printed by `--version` | worker/host version pinning | Still UNVERIFIED (depends on `git rev-parse --short` on the release runner). The regex given in §6 already tolerates any length |
| Default port | (a) | 9931 here (`common/common.h:630`). llama_rpc.md and llama_placement.md §6 show 8080 in examples. Always pass `--port` explicitly |
| (c) streaming, usage and timings | (c) | No blocking gaps. One reminder: with `include_usage`, the usage chunk has `"choices":[]`, so the proxy must not index `choices[0]` on every chunk. `timings` rides on that last chunk |
| Volume of TRACE output during generation at `-lv 4` | (b) log handling | UNVERIFIED. Server-side `LOG_TRC`/`SLT_*` call sites were not audited. Parse placement lines only up to `model loaded`, and write logs with `--log-file` |

### Combined launch line for the two-node MVP (derived from the three llama.cpp audits; not hardware-validated)

```
llama-server -m <model.gguf> --alias <id> --host 127.0.0.1 --port <port> -np 1 --no-webui --metrics \
  --rpc <workerIPv4>:50052 --device RPC0,<LOCAL0> --fit off -ngl all -ts <w_rpc>,<w_local> -c <ctx> \
  -lv 4 --log-file <session>/llama-server.log
```

- Order matters. `--rpc` must come before `--device`, because `--device` resolves names at parse time
  (llama_placement.md §1.4).
- `<LOCAL0>` is `CUDA0` on a Windows/NVIDIA host and `MTL0` on a Mac host. `RPC<n>` names come from a prior probe
  (`--rpc ... --list-devices`, run with a hard timeout; see llama_rpc.md addendum C1). `-ts` entries follow `--device`
  order.
- Child environment and cwd: cwd set to the binary directory (llama_placement.md addendum P1). `APPDATA` and
  `PROGRAMDATA`, or `XDG_CONFIG_HOME` on POSIX, point to an empty Hypercompute-owned directory (P2). Strip
  `X-Conversation-Id` on proxied requests (§3).
- Verify (b) by requiring a `load_tensors: RPC0[<workerIPv4>:50052] model buffer size = <x> MiB` line with x > 0, and the
  matching `KV buffer size` line, before reporting the session ready.
