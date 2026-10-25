"""AirLLM worker: serves one HF checkpoint through AirLLM, one request at a time, over loopback HTTP.

The gateway's Supervisor starts it with the AirLLM venv's python. It uses only the stdlib, airllm, torch and
transformers, and never imports hypercompute. Endpoints:
    GET  /health                503 while loading (with the phase), 200 when ready, 500 with the reason if load failed
    POST /v1/chat/completions   OpenAI-style; real SSE when "stream" is true, one JSON completion otherwise
Every request needs the bearer key from HYPERCOMPUTE_WORKER_KEY. AirLLM re-reads every layer from disk for every
token (docs/research/airllm.md section 4.3), so one token takes seconds to minutes. A closed stream stops generation
after the token in progress; only killing the process interrupts a forward pass.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import select
import signal
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

KEY_ENV = "HYPERCOMPUTE_WORKER_KEY"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
# In-tree transformers classes on AirLLM's generic path; nothing here runs checkpoint code (research section 3.5).
ALLOWED_ARCHITECTURES = ("Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM")
SPLIT_DIR_NAME = "splitted_model"  # AirLLM appends this to layer_shards_saving_path
DONE_SUFFIX = ".safetensors.done"  # AirLLM writes this marker once a module's shard is complete
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_MESSAGES = 256
MAX_STOP = 4
DEFAULT_MAX_TOKENS = 256
ROLES = ("system", "user", "assistant")
DTYPES = ("auto", "bfloat16", "float16", "float32")
DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d{1,2})?)$")
DISCONNECTS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)
DISCONNECT_POLL_S = 0.5
PHASE_IMPORT, PHASE_BUILD = "importing", "build"

log = logging.getLogger("airllm_worker")


class RequestError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


@dataclass(frozen=True)
class GenOptions:
    max_tokens: int
    temperature: float = 0.0  # 0 = greedy; Qwen's generation_config samples by default, so always explicit
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prompt:
    n_tokens: int
    data: Any  # backend-specific encoded prompt


@dataclass(frozen=True)
class Done:
    finish_reason: str  # stop | length
    prompt_n: int
    predicted_n: int
    prompt_ms: float | None
    predicted_ms: float | None


class Backend(Protocol):
    def load(self, on_phase: Callable[[str], None]) -> dict: ...
    def encode(self, messages: list[dict]) -> Prompt: ...
    def generate(self, prompt: Prompt, opts: GenOptions, cancel: threading.Event) -> Iterator[str | Done]: ...


# ----------------------------------------------------------------------------- request parsing


def _number(body: dict, key: str, lo: float, hi: float, default: Any, integer: bool) -> Any:
    value = body.get(key)
    if value is None:
        return default
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not lo <= value <= hi:
        kind = "an integer" if integer else "a number"
        raise RequestError(400, "invalid_request", f"{key} must be {kind} from {lo:g} to {hi:g}")
    return value if integer else float(value)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if not all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
                   for p in content):
            raise RequestError(400, "invalid_request", "only text content parts are supported")
        return "".join(p["text"] for p in content)
    raise RequestError(400, "invalid_request", "message content must be a string or a list of text parts")


def parse_messages(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not raw or len(raw) > MAX_MESSAGES:
        raise RequestError(400, "invalid_request", f"messages must be a list of 1 to {MAX_MESSAGES} messages")
    out = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            raise RequestError(400, "invalid_request", f"each message needs a role from {', '.join(ROLES)}")
        out.append({"role": m["role"], "content": _text(m.get("content"))})
    return out


def _stops(raw: Any) -> tuple[str, ...]:
    stops = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else [] if raw is None else None
    if stops is None or len(stops) > MAX_STOP or not all(isinstance(s, str) and 0 < len(s) <= 64 for s in stops):
        raise RequestError(400, "invalid_request", f"stop must be a string or a list of up to {MAX_STOP} "
                           "non-empty strings of at most 64 characters")
    return tuple(stops)


def parse_options(body: dict, max_seq_len: int) -> GenOptions:
    default_max = min(DEFAULT_MAX_TOKENS, max_seq_len)
    max_tokens = _number(body, "max_tokens", 1, max_seq_len, None, True)
    if max_tokens is None:
        max_tokens = _number(body, "max_completion_tokens", 1, max_seq_len, default_max, True)
    top_p = _number(body, "top_p", 0.0, 1.0, None, False)
    if top_p == 0.0:
        raise RequestError(400, "invalid_request", "top_p must be greater than 0")
    return GenOptions(max_tokens=max_tokens, temperature=_number(body, "temperature", 0.0, 5.0, 0.0, False),
                      top_p=top_p, top_k=_number(body, "top_k", 0, 1000, None, True) or None,
                      seed=_number(body, "seed", 0, 2**63 - 1, None, True), stop=_stops(body.get("stop")))


def fit_context(prompt: Prompt, opts: GenOptions, max_seq_len: int) -> GenOptions:
    """AirLLM's own max_seq_len is never used (research section 3.2), so the worker bounds prompt + output itself."""
    room = max_seq_len - prompt.n_tokens
    if room < 1:
        raise RequestError(400, "context_length_exceeded", f"the prompt is {prompt.n_tokens} tokens; this "
                           f"deployment allows {max_seq_len} tokens of prompt plus output")
    return replace(opts, max_tokens=min(opts.max_tokens, room))


class StopScanner:
    """Finds stop strings across streamed pieces, holding back text that could be the start of one."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self.stops, self.buf, self.hit = stops, "", False

    def feed(self, text: str) -> str:
        if not self.stops:
            return text
        self.buf += text
        found = [i for s in self.stops if (i := self.buf.find(s)) >= 0]
        if found:
            out, self.buf, self.hit = self.buf[:min(found)], "", True
            return out
        hold = max((k for s in self.stops for k in range(1, len(s)) if self.buf.endswith(s[:k])), default=0)
        cut = len(self.buf) - hold
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


# ----------------------------------------------------------------------------- checkpoint checks


def read_config(model_dir: Path) -> dict:
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {model_dir / 'config.json'}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{model_dir / 'config.json'} is not a JSON object")
    return cfg


def check_checkpoint(model_dir: Path) -> dict:
    """Refuses checkpoints that would run custom code or unpickle weights before AirLLM touches them."""
    if not model_dir.is_dir():
        raise ValueError(f"model directory not found: {model_dir}")
    cfg = read_config(model_dir)
    arch = (cfg.get("architectures") or [None])[0]
    if arch not in ALLOWED_ARCHITECTURES:
        raise ValueError(f"architecture {arch} is not on the AirLLM allowlist ({', '.join(ALLOWED_ARCHITECTURES)})")
    if "auto_map" in cfg or any(model_dir.glob("*.py")):
        raise ValueError("the checkpoint ships custom code (auto_map or .py files); refusing to load it")
    if any((model_dir / n).exists() for n in ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        raise ValueError("the checkpoint has pickled .bin weights, which AirLLM would prefer; use safetensors only")
    if not any((model_dir / n).is_file() for n in ("model.safetensors.index.json", "model.safetensors")):
        raise ValueError(f"no safetensors weights in {model_dir}")
    return cfg


def expected_modules(cfg: dict) -> int | None:
    """Per-module shards AirLLM writes: embed, each decoder layer, norm and lm_head (absent when tied)."""
    n = cfg.get("num_hidden_layers")
    if not isinstance(n, int) or n <= 0:
        return None
    return n + 2 + (0 if cfg.get("tie_word_embeddings") else 1)


def split_done(shards_dir: Path) -> int:
    try:
        with os.scandir(shards_dir / SPLIT_DIR_NAME) as it:
            return sum(1 for e in it if e.name.endswith(DONE_SUFFIX))
    except OSError:
        return 0


# ----------------------------------------------------------------------------- worker state


class Worker:
    """Load state and the one-request gate, shared by every HTTP handler thread."""

    def __init__(self, backend: Backend, *, alias: str, key: str, max_seq_len: int, model_dir: Path,
                 shards_dir: Path) -> None:
        self.backend, self.alias, self.max_seq_len = backend, alias, max_seq_len
        self.model_dir, self.shards_dir = model_dir, shards_dir
        self._key = key.encode()
        self.state, self.phase, self.error = "loading", "starting", None
        self.facts: dict = {}
        self.split_total: int | None = None
        self.busy = threading.Lock()
        self.active: threading.Event | None = None
        self.started = time.monotonic()

    def authorized(self, header: str | None) -> bool:
        scheme, _, token = (header or "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), self._key)

    def _on_phase(self, phase: str) -> None:
        if phase == PHASE_BUILD:
            done = split_done(self.shards_dir)
            phase = "loading" if self.split_total and done >= self.split_total else "splitting"
        self.phase = phase
        log.info("phase: %s", phase)

    def load(self) -> None:
        try:
            cfg = check_checkpoint(self.model_dir)
            self.split_total = expected_modules(cfg)
            self.facts = {**self.backend.load(self._on_phase), "split_modules": split_done(self.shards_dir)}
            self.state, self.phase = "ready", "ready"
            log.info("ready after %.1fs: %s", time.monotonic() - self.started, self.facts)
        except Exception as exc:  # reported through /health; the gateway reads it and stops this process
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")

    def health(self) -> tuple[int, dict]:
        if self.state == "ready":
            return 200, {"status": "ok", "alias": self.alias, **self.facts}
        if self.state == "failed":
            return 500, {"status": "failed", "error": self.error}
        body: dict = {"status": "loading", "phase": self.phase, "elapsed_s": round(time.monotonic() - self.started)}
        if self.phase in ("splitting", "loading"):
            done = split_done(self.shards_dir)
            if self.phase == "splitting" and self.split_total and done >= self.split_total:
                body["phase"] = "loading"
            body["split"] = {"done": done, "total": self.split_total}
        return 503, body

    def run(self, prompt: Prompt, opts: GenOptions, cancel: threading.Event) -> Iterator[str | Done]:
        """The backend's pieces with stop strings applied; the last item is always a Done."""
        scanner = StopScanner(opts.stop)
        with closing(self.backend.generate(prompt, opts, cancel)) as items:
            for item in items:
                if isinstance(item, Done):
                    if tail := scanner.flush():
                        yield tail
                    yield replace(item, finish_reason="stop") if scanner.hit else item
                    return
                if scanner.hit:
                    continue  # a stop string was seen: drop text until the backend notices the cancel
                text = scanner.feed(item)
                if scanner.hit:
                    cancel.set()
                if text:
                    yield text
        raise RuntimeError("generation ended without a final result")

    def shutdown(self) -> None:
        if self.active is not None:
            self.active.set()


def usage(done: Done) -> dict:
    return {"prompt_tokens": done.prompt_n, "completion_tokens": done.predicted_n,
            "total_tokens": done.prompt_n + done.predicted_n}


def timings(done: Done) -> dict:
    def rate(n: int, ms: float | None) -> float | None:
        return round(n * 1000.0 / ms, 4) if ms else None

    return {"prompt_n": done.prompt_n, "prompt_ms": done.prompt_ms, "predicted_n": done.predicted_n,
            "predicted_ms": done.predicted_ms, "prompt_per_second": rate(done.prompt_n, done.prompt_ms),
            "predicted_per_second": rate(done.predicted_n, done.predicted_ms)}


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "type": code}}


# ----------------------------------------------------------------------------- HTTP


def make_handler(worker: Worker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "hypercompute-airllm-worker"
        sys_version = ""
        timeout = 120  # a stalled client is treated as gone

        def log_message(self, format: str, *args: Any) -> None:
            log.debug("%s %s", self.address_string(), format % args)

        def _send(self, data: bytes) -> None:
            self.wfile.write(data)
            self.wfile.flush()

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self._send(data)
            except DISCONNECTS:
                log.info("client went away before the %d response", status)

        def _fail(self, exc: RequestError) -> None:
            self._json(exc.status, _error_body(exc.code, str(exc)))

        def _authorized(self) -> bool:
            if worker.authorized(self.headers.get("Authorization")):
                return True
            self._fail(RequestError(401, "unauthorized", "missing or wrong bearer key"))
            return False

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path.split("?", 1)[0] != "/health":
                self._fail(RequestError(404, "not_found", "unknown path"))
                return
            self._json(*worker.health())

        def do_POST(self) -> None:
            # Read the body before any reply: answering with unread bytes in the socket makes Windows reset the
            # connection, so the client sees a reset instead of the 401/404.
            raw = self._read_body()
            if not self._authorized():
                return
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._fail(RequestError(404, "not_found", "unknown path"))
                return
            try:
                if isinstance(raw, RequestError):
                    raise raw
                body = self._parse_json(raw)
                messages, opts = parse_messages(body.get("messages")), parse_options(body, worker.max_seq_len)
                stream = body.get("stream", False)
                if not isinstance(stream, bool):
                    raise RequestError(400, "invalid_request", "stream must be true or false")
                if worker.state != "ready":
                    raise RequestError(503, "not_ready", f"the model is {worker.state} ({worker.phase})")
            except RequestError as exc:
                self._fail(exc)
                return
            if not worker.busy.acquire(blocking=False):
                self._fail(RequestError(429, "busy", "the AirLLM worker serves one request at a time"))
                return
            try:
                self._serve(messages, opts, stream)
            finally:
                worker.busy.release()

        def _read_body(self) -> bytes | RequestError:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return RequestError(411, "length_required", "Content-Length is required")
            if not 0 < length <= MAX_BODY_BYTES:
                self.close_connection = True  # an oversized body is not read, so this connection cannot be reused
                return RequestError(413, "too_large", f"request body must be 1 to {MAX_BODY_BYTES} bytes")
            return self.rfile.read(length)

        @staticmethod
        def _parse_json(raw: bytes) -> dict:
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RequestError(400, "invalid_json", "request body is not valid JSON") from None
            if not isinstance(body, dict):
                raise RequestError(400, "invalid_request", "request body must be a JSON object")
            return body

        def _serve(self, messages: list[dict], opts: GenOptions, stream: bool) -> None:
            try:
                prompt = worker.backend.encode(messages)
                opts = fit_context(prompt, opts, worker.max_seq_len)
            except RequestError as exc:
                self._fail(exc)
                return
            except Exception as exc:  # e.g. a tokenizer without a chat template
                log.exception("could not encode the prompt")
                self._fail(RequestError(400, "invalid_request", f"could not apply the chat template: {exc}"))
                return
            cancel, finished = threading.Event(), threading.Event()
            worker.active = cancel
            log.info("request: prompt %d tokens, max_tokens %d, stream %s", prompt.n_tokens, opts.max_tokens, stream)
            threading.Thread(target=self._watch_disconnect, args=(cancel, finished), name="airllm-disconnect",
                             daemon=True).start()
            try:
                (self._stream if stream else self._complete)(prompt, opts, cancel)
            finally:
                finished.set()
                worker.active = None

        def _watch_disconnect(self, cancel: threading.Event, finished: threading.Event) -> None:
            """Cancel as soon as the client closes its socket. Waiting for the next write is too late: the streamer
            can hold back several tokens, and in capacity mode one token can take minutes."""
            while not finished.wait(DISCONNECT_POLL_S):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        log.info("client closed the connection; stopping after the current token")
                        cancel.set()
                        return
                except (OSError, ValueError):  # reset, or the socket was already closed
                    cancel.set()
                    return

        def _stream(self, prompt: Prompt, opts: GenOptions, cancel: threading.Event) -> None:
            cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

            def chunk(delta: dict, finish: str | None, **extra: Any) -> bytes:
                payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": worker.alias,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
                return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self._send(chunk({"role": "assistant", "content": None}, None))
                with closing(worker.run(prompt, opts, cancel)) as items:
                    for item in items:
                        if isinstance(item, Done):
                            self._send(chunk({}, item.finish_reason, usage=usage(item), timings=timings(item)))
                            break
                        self._send(chunk({"content": item}, None))
                self._send(b"data: [DONE]\n\n")
            except DISCONNECTS:
                cancel.set()
                log.info("client closed the stream; generation stopped")
            except Exception as exc:  # after the headers: report in-band, then end the stream
                log.exception("generation failed")
                try:
                    self._send(b"data: " + json.dumps(_error_body("generation_failed", str(exc))).encode() + b"\n\n")
                    self._send(b"data: [DONE]\n\n")
                except DISCONNECTS:
                    pass

        def _complete(self, prompt: Prompt, opts: GenOptions, cancel: threading.Event) -> None:
            pieces: list[str] = []
            try:
                with closing(worker.run(prompt, opts, cancel)) as items:
                    done = None
                    for item in items:
                        if isinstance(item, Done):
                            done = item
                            break
                        pieces.append(item)
            except Exception as exc:
                log.exception("generation failed")
                self._fail(RequestError(500, "generation_failed", str(exc)))
                return
            if done is None:
                self._fail(RequestError(500, "generation_failed", "generation ended without a final result"))
                return
            self._json(200, {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                             "created": int(time.time()), "model": worker.alias,
                             "choices": [{"index": 0, "finish_reason": done.finish_reason,
                                          "message": {"role": "assistant", "content": "".join(pieces)}}],
                             "usage": usage(done), "timings": timings(done)})

    return Handler


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"  # on Windows SO_REUSEADDR would let another socket share the port


def build_server(worker: Worker, host: str, port: int) -> WorkerServer:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"the worker listens on loopback only, not {host}")
    return WorkerServer(("127.0.0.1", port), make_handler(worker))


# ----------------------------------------------------------------------------- AirLLM backend


def _streaming_classes(torch: Any, streamer_base: type, criteria_base: type, criteria_list: type) -> tuple[Any, Any]:
    class TimedStreamer(streamer_base):  # type: ignore[misc,valid-type]
        """TextIteratorStreamer that notes when new tokens arrive. put() #1 is the prompt (transformers generate)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.puts, self.first_token_at, self.last_at = 0, None, time.perf_counter()

        def put(self, value: Any) -> None:
            self.puts += 1
            now = time.perf_counter()
            if self.puts >= 2:
                self.first_token_at = self.first_token_at or now
                log.info("token %d after %.1fs", self.puts - 1, now - self.last_at)
            self.last_at = now
            super().put(value)

    class CancelCriteria(criteria_base):  # type: ignore[misc,valid-type]
        """Checked by transformers once per decode step: stops after the token in progress."""

        def __init__(self, event: threading.Event) -> None:
            self.event = event

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)

    return TimedStreamer, lambda event: criteria_list([CancelCriteria(event)])


class AirLlmBackend:
    """AirLLM's generic streaming model: every module is read from disk to the device for each forward pass."""

    def __init__(self, model_dir: Path, shards_dir: Path, device: str, dtype: str, max_seq_len: int) -> None:
        self.model_dir, self.shards_dir, self.device, self.dtype = model_dir, shards_dir, device, dtype
        self.max_seq_len = max_seq_len
        self._torch: Any = None
        self._model: Any = None
        self._tok: Any = None
        self._streamer: Any = None
        self._criteria: Any = None

    def _resolve_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("torch cannot see a CUDA device; reinstall the AirLLM engine or plan for the CPU")
        return self.device

    def load(self, on_phase: Callable[[str], None]) -> dict:
        on_phase(PHASE_IMPORT)
        import torch
        from airllm import AirLLMBaseModel  # exported on Windows and Linux (airllm/__init__.py)
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        device = self._resolve_device(torch)
        on_phase(PHASE_BUILD)  # splits the checkpoint on first use (resumable via .done markers), then builds on meta
        model = AirLLMBaseModel(str(self.model_dir), device=device,
                                dtype=None if self.dtype == "auto" else getattr(torch, self.dtype),
                                max_seq_len=self.max_seq_len, layer_shards_saving_path=str(self.shards_dir),
                                prefetching=True, compression=None,  # compression is slower (#330)
                                delete_original=False)  # crashes on Windows (#297)
        self._torch, self._model, self._tok = torch, model, model.tokenizer
        self._streamer, self._criteria = _streaming_classes(torch, TextIteratorStreamer, StoppingCriteria,
                                                            StoppingCriteriaList)
        cuda = device.startswith("cuda")
        return {"device": device, "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
                "resident_bytes": int(torch.cuda.memory_allocated(device)) if cuda else None,
                "torch": torch.__version__}

    def encode(self, messages: list[dict]) -> Prompt:
        enc = self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt")
        return Prompt(n_tokens=int(enc["input_ids"].shape[1]), data=enc)

    def _kwargs(self, prompt: Prompt, opts: GenOptions, streamer: Any, cancel: threading.Event) -> dict:
        kwargs: dict = {"attention_mask": prompt.data["attention_mask"].to(self._model.device),
                        "max_new_tokens": opts.max_tokens, "use_cache": True, "streamer": streamer,
                        "stopping_criteria": self._criteria(cancel), "return_dict_in_generate": True,
                        "do_sample": opts.temperature > 0}
        if opts.temperature > 0:
            kwargs.update(temperature=opts.temperature, top_p=1.0 if opts.top_p is None else opts.top_p)
            if opts.top_k:
                kwargs["top_k"] = opts.top_k
        return kwargs

    def generate(self, prompt: Prompt, opts: GenOptions, cancel: threading.Event) -> Iterator[str | Done]:
        input_ids = prompt.data["input_ids"].to(self._model.device)
        streamer = self._streamer(self._tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = self._kwargs(prompt, opts, streamer, cancel)
        if opts.seed is not None:
            self._torch.manual_seed(opts.seed)
        box: dict = {}
        started = time.perf_counter()

        def run() -> None:
            try:
                box["out"] = self._model.generate(input_ids, **kwargs)
            except BaseException as exc:  # handed to the request thread, which reports it
                box["error"] = exc
                streamer.end()  # unblocks the iterator below

        thread = threading.Thread(target=run, name="airllm-generate", daemon=True)
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
            thread.join()
            if "error" in box:
                raise RuntimeError(f"generation failed: {box['error']}") from box["error"]
            n_new = int(box["out"].sequences.shape[1] - input_ids.shape[1])
            ended, first = time.perf_counter(), streamer.first_token_at
            yield Done(finish_reason="length" if n_new >= opts.max_tokens else "stop", prompt_n=prompt.n_tokens,
                       predicted_n=n_new, prompt_ms=round((first - started) * 1000, 3) if first else None,
                       predicted_ms=round((ended - first) * 1000, 3) if first else None)
        finally:
            cancel.set()
            thread.join()  # the GPU is free for the next request only once generate() has returned


# ----------------------------------------------------------------------------- entry point


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hypercompute AirLLM worker (loopback only)")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--shards-dir", required=True, type=Path, help="per-model dir; AirLLM adds splitted_model/")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--alias", required=True)
    p.add_argument("--max-seq-len", type=int, default=4096, help="prompt plus output tokens per request")
    p.add_argument("--device", default="auto", help="cuda:N, cpu or auto")
    p.add_argument("--dtype", default="auto", choices=DTYPES)
    args = p.parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        p.error(f"--host must be loopback ({', '.join(LOOPBACK_HOSTS)})")
    if not 1 <= args.port <= 65535:
        p.error("--port must be 1-65535")
    if not 16 <= args.max_seq_len <= 262144:
        p.error("--max-seq-len must be 16-262144")
    if not DEVICE_RE.match(args.device):
        p.error("--device must be auto, cpu, cuda or cuda:N")
    return args


def _setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):  # the log file is UTF-8 whatever the console code page is
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError) as exc:  # a replaced or already-used stream keeps its encoding
            print(f"airllm-worker: keeping the default stream encoding ({exc})", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s airllm-worker: %(message)s")


def _on_signals(stop: threading.Event) -> None:
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):  # SIGBREAK: Ctrl+Break to a Windows process group
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, lambda *_: stop.set())


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = parse_args(argv)
    key = os.environ.pop(KEY_ENV, "")
    if len(key) < 16:
        log.error("%s is missing or shorter than 16 characters; refusing to serve without a key", KEY_ENV)
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the gateway pre-downloads pinned files; never reach the Hub
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    backend = AirLlmBackend(args.model_dir, args.shards_dir, args.device, args.dtype, args.max_seq_len)
    worker = Worker(backend, alias=args.alias, key=key, max_seq_len=args.max_seq_len, model_dir=args.model_dir,
                    shards_dir=args.shards_dir)
    try:
        server = build_server(worker, args.host, args.port)
    except OSError as exc:
        log.error("cannot listen on %s:%d: %s", args.host, args.port, exc)
        return 3
    stop = threading.Event()
    _on_signals(stop)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    threading.Thread(target=worker.load, name="load", daemon=True).start()
    log.info("listening on http://127.0.0.1:%d for %s (model %s)", args.port, args.alias, args.model_dir)
    while not stop.wait(0.5):
        pass
    log.info("shutting down")
    worker.shutdown()
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    code = main()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # do not wait for library threads (AirLLM's prefetch pool) that may sit in a layer read
