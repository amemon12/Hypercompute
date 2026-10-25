"""vLLM v0.32.0 (5df63a7) as seen from outside its process: startup log milestones and failures, the first start's
download (budget and progress), the in-distro launch and stop scripts, the chat request it accepts and the SSE stream
it sends.

Log needles and format strings are cited in docs/research/vllm.md §3.3; the stream shape follows
chat_completion/serving.py. Fixtures: tests/fixtures/vllm_startup_*.log and vllm_stream_*.sse.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import config
from ..installers.vllm import VllmInstall, decode_wsl, is_served, wsl_env
from ..schemas import Plan
from ..sse import SSEParser
from ..units import MiB, fmt_bytes
from .base import EngineAdapterError, GenerationEvent

log = logging.getLogger("hypercompute.vllm")

DOWNLOAD_FLOOR_BPS = 1 * MiB  # the first start's budget assumes its in-distro download gets at least ~8 Mbit/s
MAX_LOAD_S = 4 * 3600.0  # the same cap as AirLLM and EXO, whose first starts also download
DOWNLOAD_POLL_S = 15.0  # how often a downloading first start sizes vLLM's HF cache (one wsl.exe call each)
PHASES = (  # log milestones (research §3.3), in startup order
    ("Time spent downloading weights", "weights downloaded; loading them"),  # weight_utils, only when it downloaded
    ("Model loading took", "weights loaded; profiling memory"),
    ("Available KV cache memory", "allocating the KV cache"),
    ("KV cache size:", "capturing CUDA graphs"),
    ("Graph capturing finished", "warming up"),
    ("init engine (profile, create kv cache, warmup model) took", "engine initialised"),
    ("Starting vLLM server on", "starting the HTTP server"),
    ("Application startup complete", "HTTP server up"),
)
FATAL = (  # (log needle, what happened, fix); the specific causes come before the generic engine-core failure
    # The error reads "Free memory on device ... on startup is less than ..."; vLLM 0.32 also logs a normal INFO line
    # "Free memory on device (x/y GiB) on startup. Desired GPU memory utilization is ...", which must not match.
    ("on startup is less than", "less GPU memory was free than --gpu-memory-utilization reserves",
     "stop the llama.cpp deployment or other GPU apps, then check the plan again"),
    ("is larger than the available KV cache memory", "the context does not fit the KV cache vLLM could allocate",
     "lower ctx_size (--max-model-len) or free GPU memory, then check the plan again"),
    ("Failed to load model - not enough GPU memory", "the weights did not fit in GPU memory",
     "free GPU memory or choose a smaller checkpoint"),
    ("CUDA out of memory", "the GPU ran out of memory during startup", "lower ctx_size or free GPU memory"),
    ("No CUDA GPUs are available", "PyTorch sees no GPU",
     "update the Windows NVIDIA driver and run `wsl --update`; never install a Linux NVIDIA driver inside WSL"),
    ("RevisionNotFoundError", "the pinned revision is not in the Hugging Face repo",
     "fix `revision` in the model manifest (full commit sha)"),
    ("RepositoryNotFoundError", "the Hugging Face repo was not found", "fix `repo` in the model manifest"),
    ("LocalEntryNotFoundError", "the model could not be downloaded",
     "check internet access inside WSL: wsl -- curl -sI https://huggingface.co"),
    ("Address already in use", "the chosen port was taken", "start the deployment again (a new port is chosen)"),
    ("Engine core initialization failed", "vLLM's engine core failed to start",
     "read the vLLM log above this line for the cause"),
)
METRICS = {
    "weights": re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
    "kv": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_tokens": re.compile(r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: "
                            r"([\d.]+)x"),
    "graphs": re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB"),
    "init": re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"),
    "kernel": re.compile(r"Using (\w+) for AutoAWQMarlinLinearMethod"),
}
# "ValueError: ..." anywhere in a line: vLLM prefixes its log lines with "(EngineCore pid=N) ERROR ... [file:line]"
EXCEPTION_LINE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception): .*)$", re.MULTILINE)


# ----------------------------------------------------------------------------- log reading


def load_phase(text: str, rec: VllmInstall) -> str:
    phase = ("starting vLLM in WSL2" if rec.platform == "wsl2" else "starting vLLM") + \
        " (the first start downloads the model and compiles kernels: several minutes)"
    for needle, label in PHASES:
        if needle in text:
            phase = label
    return phase


def startup_failure(text: str) -> tuple[str, str] | None:
    """(message, fix) for the first known fatal marker in the log, quoting the line that carries it."""
    for needle, what, fix in FATAL:
        at = text.find(needle)
        if at >= 0:
            start = text.rfind("\n", 0, at) + 1
            end = text.find("\n", at)
            line = text[start:end if end >= 0 else len(text)].strip()
            return f"vLLM failed to start: {what} ({line[:300]})", fix
    return None


def _last_error(text: str) -> str:
    lines = [m.group(1) for m in EXCEPTION_LINE.finditer(text)]
    last = lines[-1] if lines else next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
    return last.strip()[:300]


def exit_detail(text: str, code: int | None) -> str:
    last = _last_error(text)
    return f"vLLM server exited during startup (exit code {code})" + (f": {last}" if last else "")


def exit_reason(text: str) -> str:
    """Why a vLLM server that was serving went away (the deployment watcher's failure message)."""
    last = _last_error(text)
    return "vLLM server exited unexpectedly" + (f": {last}" if last else "")


def parse_startup(text: str) -> tuple[dict[str, float], list[str]]:
    """Memory and timing milestones vLLM logs at startup, with the lines they came from."""
    values: dict[str, float] = {}
    evidence: list[str] = []
    for key, rx in METRICS.items():
        matches = list(rx.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        evidence.append(text[line_start:line_end if line_end >= 0 else len(text)].strip()[:300])
        groups = [g.replace(",", "") for g in m.groups()]
        if key == "kernel":
            continue
        values[key] = float(groups[0])
        if key == "weights":
            values["load_s"] = float(groups[1])
        elif key == "kv_tokens":
            values["max_len"], values["concurrency"] = float(groups[1]), float(groups[2])
        elif key == "graphs":
            values["graphs_gib"] = float(groups[1])
    return values, evidence


# ----------------------------------------------------------------------------- first-start download


def download_need(paths: config.Paths, rec: VllmInstall, plan: Plan) -> int:
    """Bytes the plan's first start downloads: none once a server here served its repo@revision (validate's rule)."""
    opts = plan.engine_options
    if is_served(paths, rec, str(opts.get("model")), str(opts.get("revision"))):
        return 0
    return int(opts.get("download_bytes") or 0) or sum(a.weight_bytes for a in plan.assignments)


def load_budget(download: int) -> float:
    """Seconds a start may take: LOAD_TIMEOUT_S plus the download at DOWNLOAD_FLOOR_BPS, capped at MAX_LOAD_S."""
    return min(MAX_LOAD_S, config.LOAD_TIMEOUT_S + max(0, download) / DOWNLOAD_FLOOR_BPS)


def du_bytes(raw: bytes | str | None) -> int | None:
    """Bytes from `du -sk` output ("<KiB>\\t<path>"); None for anything else."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw or "")
    head = text.split()[:1]
    return int(head[0]) * 1024 if head and head[0].isdigit() else None


@dataclass
class DownloadProgress:
    """Progress of the first start's checkpoint download. vLLM logs nothing while it downloads (weight_utils runs
    snapshot_download with DisabledTqdm), so this sizes the repo's folder in vLLM's HF cache every DOWNLOAD_POLL_S."""

    rec: VllmInstall
    wsl: str | None
    run: Callable[..., subprocess.CompletedProcess]
    repo: str
    need: int  # bytes the start downloads; 0 turns this off
    next_at: float = 0.0
    line: str | None = None

    def size(self) -> int | None:
        folder = f"{self.rec.hf_home}/hub/models--{self.repo.replace('/', '--')}"  # huggingface_hub's repo folder
        du = ["du", "-sk", folder]
        argv = [self.wsl or "wsl.exe", "-d", self.rec.distro or "", "--exec", *du] if self.rec.platform == "wsl2" \
            else du
        try:
            out = self.run(argv, capture_output=True, timeout=10.0, stdin=subprocess.DEVNULL, env=wsl_env(),
                           check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.debug("could not size %s: %s", folder, exc)
            return None
        return du_bytes(out.stdout) if out.returncode == 0 else None  # 1 until the folder exists

    async def phase(self, now: float, logged: str) -> str:
        """The log's phase once vLLM logs a milestone; until then the download's progress, when it can be read."""
        if not self.need or logged != load_phase("", self.rec):
            return logged
        if now >= self.next_at:
            self.next_at = now + DOWNLOAD_POLL_S
            have = await asyncio.to_thread(self.size)
            if have is not None:
                self.line = (f"downloading {self.repo} (first start): {fmt_bytes(min(have, self.need))} of "
                             f"{fmt_bytes(self.need)}")
        return self.line or logged


# ----------------------------------------------------------------------------- the server's process in the distro

# Inside the distro, run under `setsid -w env ... bash -c LAUNCH <pidfile> <cmd...>`: record the session leader,
# start a detached watchdog (its own session, no inherited pipes) that TERMs then KILLs the group once the
# `setsid -w` waiter (alive exactly as long as wsl.exe) is gone, then exec the server so the pid stays the leader.
LAUNCH = r'''leader=$$; waiter=$PPID; pidfile=$0
mkdir -p "${pidfile%/*}" && echo "$leader" > "$pidfile" || exit 97
setsid -f bash -c 'while kill -0 "$0" 2>/dev/null && kill -0 "$1" 2>/dev/null; do sleep 2; done
kill -0 "$1" 2>/dev/null || exit 0
kill -TERM -- "-$1" 2>/dev/null
for _ in $(seq 30); do kill -0 -- "-$1" 2>/dev/null || exit 0; sleep 1; done
kill -KILL -- "-$1" 2>/dev/null' "$waiter" "$leader" </dev/null >/dev/null 2>&1
exec "$@"'''
# `bash -c STOP <pidfile> <grace_s>`: TERM the leader, wait for the whole group, else KILL it. Prints one outcome word.
STOP = r'''pidfile=$0; grace=$1
[ -f "$pidfile" ] || { echo no-pidfile; exit 0; }
pid=$(cat "$pidfile")
case $pid in ''|*[!0-9]*) rm -f "$pidfile"; echo bad-pidfile; exit 0 ;; esac
alive() { kill -0 -- "-$pid" 2>/dev/null; }
alive || { rm -f "$pidfile"; echo already-exited; exit 0; }
if kill -0 "$pid" 2>/dev/null && ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q vllm; then echo not-vllm; exit 0; fi
kill -TERM "$pid" 2>/dev/null
i=0
while [ "$i" -lt "$grace" ]; do alive || { rm -f "$pidfile"; echo graceful; exit 0; }; sleep 1; i=$((i + 1)); done
kill -KILL -- "-$pid" 2>/dev/null
sleep 1
alive && { echo still-running; exit 1; }
rm -f "$pidfile"; echo killed'''
INNER_GONE = {"already-exited", "no-pidfile", "bad-pidfile", "not-vllm"}  # STOP found no vLLM server to stop
INNER_CLEAN = {"graceful", "killed", *INNER_GONE}


def distro_running(run: Callable[..., subprocess.CompletedProcess], wsl: str | None, distro: str) -> bool | str:
    """Whether a WSL distro runs, from `wsl -l --running -q`, which lists without starting the VM: one name per line,
    and nothing with exit 0 when none runs (WslClient.cpp ListDistributionsHelper exits -1 only without -q). A str
    says why that is unknown."""
    try:
        out = run([wsl or "wsl.exe", "-l", "--running", "-q"], capture_output=True, timeout=30.0,
                  stdin=subprocess.DEVNULL, env=wsl_env(), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`wsl -l --running` failed: {exc}"
    text = (decode_wsl(out.stdout) if isinstance(out.stdout, bytes) else str(out.stdout or "")).strip()
    if out.returncode != 0:
        return f"`wsl -l --running` exited with {out.returncode}" + (f": {text[:200]}" if text else "")
    return distro.lower() in {ln.strip().lower() for ln in text.splitlines()}


def orphan_pidfile(rec: VllmInstall, opts: dict, deployment_id: str) -> tuple[str, str]:
    """(distro, pidfile) of the server a deployment launched, from its plan's own distro and install root (hf_home is
    <root>/hf), so a reinstall elsewhere since the crash still finds it; from the record when the plan lacks them."""
    distro, hf_home = opts.get("distro"), opts.get("hf_home")
    ok = isinstance(hf_home, str) and hf_home.startswith("/") and hf_home.endswith("/hf")
    root = hf_home.removesuffix("/hf") if ok else rec.root
    return (distro if isinstance(distro, str) and distro else rec.distro or ""), f"{root}/run/{deployment_id}.pid"


# ----------------------------------------------------------------------------- requests and streaming


def chat_body(body: dict, model: str, stream: bool) -> dict:
    """The chat request vLLM gets. llama.cpp's cache_prompt is dropped (vLLM ignores unknown fields); false becomes a
    fresh cache_salt, which vLLM hashes into the request's first KV block, so the request never reuses the prefix
    cache (on by default) and a benchmark repetition times a real prefill, as it does on llama.cpp."""
    out = {k: v for k, v in body.items() if k != "cache_prompt"}
    salt = {"cache_salt": uuid.uuid4().hex} if body.get("cache_prompt") is False else {}
    return {**out, **salt, "model": model, "stream": stream}


@dataclass
class StreamClock:
    """Client-side timings for runs (vLLM sends no llama.cpp `timings`). Same convention as llama.cpp: the first
    token counts in predicted_n, and generation time runs from the first token to the last."""

    now: Callable[[], float] = time.perf_counter
    start: float = field(init=False)
    first: float | None = None
    last: float | None = None

    def __post_init__(self) -> None:
        self.start = self.now()

    def observe(self, payload: dict) -> None:
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                t = self.now()
                self.first = self.first if self.first is not None else t
                self.last = t

    def timings(self, usage: dict) -> dict:
        prompt_n, predicted_n = usage.get("prompt_tokens"), usage.get("completion_tokens")
        prompt_ms = (self.first - self.start) * 1000 if self.first is not None else None
        predicted_ms = (self.last - self.first) * 1000 if self.first is not None and self.last is not None else None
        return {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "predicted_n": predicted_n,
                "predicted_ms": predicted_ms, "prompt_per_second": _rate(prompt_n, prompt_ms),
                "predicted_per_second": _rate(predicted_n, predicted_ms),
                "source": "gateway clock (vLLM reports no timings)"}


def _rate(n: object, ms: float | None) -> float | None:
    return n / ms * 1000 if isinstance(n, int) and n > 0 and ms else None


class StreamNormalizer:
    """vLLM SSE -> GenerationEvents: errors inside a 200 stream raise, the usage chunk gains synthesised timings."""

    def __init__(self, clock: StreamClock | None = None) -> None:
        self.clock = clock or StreamClock()
        self._parser = SSEParser()

    def feed(self, chunk: bytes) -> list[GenerationEvent]:
        return [self._event(ev.raw, ev.data) for ev in self._parser.feed(chunk)]

    def flush(self) -> list[GenerationEvent]:
        return [self._event(ev.raw, ev.data) for ev in self._parser.flush()]

    def _event(self, raw: bytes, data: str) -> GenerationEvent:
        if data.strip() == "[DONE]":
            return GenerationEvent(raw=raw, data=data, payload=None, done=True)
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return GenerationEvent(raw=raw, data=data, payload=None)
        if not isinstance(payload, dict):
            return GenerationEvent(raw=raw, data=data, payload=None)
        if payload.get("error"):  # chat_completion/serving.py: exceptions become a data event before [DONE]
            err = payload["error"] if isinstance(payload["error"], dict) else {"message": str(payload["error"])}
            raise EngineAdapterError("upstream_error", f"vLLM failed mid-stream: {err.get('message', err)} "
                                     f"({err.get('type', 'error')}, code {err.get('code')})",
                                     fix="read the vLLM log; an engine-core crash also fails the deployment")
        self.clock.observe(payload)
        if payload.get("usage"):
            payload = {**payload, "timings": self.clock.timings(payload["usage"])}
            data = json.dumps(payload, separators=(",", ":"))
            raw = f"data: {data}\n\n".encode()
        return GenerationEvent(raw=raw, data=data, payload=payload)
