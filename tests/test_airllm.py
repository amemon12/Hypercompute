"""AirLLM disk tier: installer status, worker HTTP/SSE protocol (fake model), adapter launch contract and plans.

Nothing here imports torch or AirLLM, loads a model or starts the real worker process.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import httpx
import pytest
import yaml
from factories import contribution, make_inventory
from hypercompute import config
from hypercompute.adapters import airllm as adapter_mod
from hypercompute.adapters import registry
from hypercompute.adapters.base import DeploymentHandle, DeploymentSpec, InferenceRequest, ValidationResult
from hypercompute.installers import airllm as inst
from hypercompute.schemas import ModelArtifact, NodeState, NodeView, PlanRequest, PlanStrategy, utcnow
from hypercompute.services.streaming import StreamStats
from hypercompute.sse import SSEParser
from hypercompute.units import GiB, MiB
from hypercompute.workers import airllm_worker as w

FIXTURES = Path(__file__).parent / "fixtures"
KEY = "k" * 32


def manifest() -> dict[str, ModelArtifact]:
    raw = yaml.safe_load((FIXTURES / "airllm_manifest.yaml").read_text(encoding="utf-8"))
    return {m["id"]: ModelArtifact.model_validate(m) for m in raw["models"]}


SMALL, LARGE = "qwen2.5-0.5b-instruct-hf", "qwen2.5-32b-instruct-hf"
Usage = namedtuple("Usage", "total used free")


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch):
    """Plans read free disk; keep it deterministic. Tests about a full disk patch it again."""
    monkeypatch.setattr(adapter_mod.shutil, "disk_usage", lambda p: Usage(0, 0, 500 * GiB))


def config_of(model_id: str) -> dict:
    size = "0.5b" if model_id == SMALL else "32b"
    name = f"airllm_qwen2.5-{size}-instruct_config.json"
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def mark_installed(paths: config.Paths, **overrides: object) -> dict:
    data = {**json.loads((FIXTURES / "airllm_installed.json").read_text(encoding="utf-8")), **overrides}
    py = inst.venv_python(paths)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_bytes(b"")
    inst.installed_file(paths).write_text(json.dumps(data), encoding="utf-8")
    return data


def write_model(dirpath: Path, model_id: str = SMALL) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(config_of(model_id)), encoding="utf-8")
    (dirpath / "model.safetensors").write_bytes(b"")
    return dirpath


# ============================================================================= installer


def test_status_not_installed_points_at_the_install_command():
    info = inst.status(config.paths())
    assert (info.installed, info.status) == (False, "not_installed")
    assert "engine install --engine airllm" in info.detail


def test_status_reads_installed_json_without_importing_torch():
    paths = config.paths()
    mark_installed(paths)
    info = inst.status(paths)
    assert (info.installed, info.status, info.version) == (True, "installed", inst.AIRLLM_COMMIT)
    assert "torch 2.11.0+cu128" in info.detail and "NVIDIA" in info.detail
    assert inst.torch_device(paths) == "cuda:0"


@pytest.mark.parametrize("problem", ["missing_python", "wrong_commit", "garbage"])
def test_status_reports_a_broken_install_with_a_fix(problem):
    paths = config.paths()
    mark_installed(paths, commit="0" * 40 if problem == "wrong_commit" else inst.AIRLLM_COMMIT)
    if problem == "missing_python":
        inst.venv_python(paths).unlink()
    if problem == "garbage":
        inst.installed_file(paths).write_text("{not json", encoding="utf-8")
    info = inst.status(paths)
    assert (info.installed, info.status) == (False, "failed")
    assert "--force" in info.detail


def test_status_on_macos_is_unsupported(monkeypatch):
    monkeypatch.setattr(inst.sys, "platform", "darwin")
    assert inst.status(config.paths()).status == "unsupported"


def test_torch_choice_follows_the_driver():
    assert inst.choose_torch("win32", 591).flavor == "cu128"
    assert inst.choose_torch("linux", None).flavor == "cpu"
    old = inst.choose_torch("win32", 551)
    assert old.flavor == "cpu" and "570" in old.reason
    with pytest.raises(inst.InstallError):
        inst.choose_torch("darwin", None)


def test_parse_smoke_takes_the_marker_line_after_library_noise():
    out = ">>>> bitsandbytes installed\nwarning\nHC_SMOKE {\"torch\": \"2.11.0+cu128\"}\n"
    assert inst.parse_smoke(out) == {"torch": "2.11.0+cu128"}
    assert inst.parse_smoke("no marker") is None


@pytest.fixture
def fake_install(monkeypatch):
    """uv, git, network and disk faked; returns the recorded commands."""
    calls: list[list[str]] = []
    smoke = {**json.loads((FIXTURES / "airllm_installed.json").read_text(encoding="utf-8"))}

    def run(cmd, cwd, timeout):
        calls.append(cmd)
        if cmd[1] == "venv":
            py = inst.venv_python(config.paths())
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_bytes(b"")
        return 0

    def capture(cmd, cwd, timeout):
        calls.append(cmd)
        line = inst.SMOKE_MARKER + json.dumps({k: smoke[k] for k in (
            "python", "torch", "torch_cuda", "cuda_available", "device", "device_name", "capability", "arch_ok",
            "airllm_commit", "packages")})
        return subprocess.CompletedProcess(cmd, 0, stdout=f"noise\n{line}\n", stderr="")

    monkeypatch.setattr(inst.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(inst.engine, "nvidia_driver_major", lambda: 591)
    monkeypatch.setattr(inst, "remote_size", lambda url: 2 * GiB)
    monkeypatch.setattr(inst.shutil, "disk_usage", lambda p: Usage(0, 0, 100 * GiB))
    monkeypatch.setattr(inst.sys, "platform", "win32")
    monkeypatch.setattr(inst, "_run", run)
    monkeypatch.setattr(inst, "_capture", capture)
    return calls


def test_install_requires_consent_before_downloading(fake_install, monkeypatch):
    monkeypatch.setattr(inst.sys.stdin, "isatty", lambda: False, raising=False)
    said: list[str] = []
    assert inst.install(config.paths(), yes=False, force=False, say=said.append) == 1
    assert fake_install == []
    assert any("2.0 GiB" in s for s in said) and any("--yes" in s for s in said)


def test_install_pins_torch_index_then_airllm_and_writes_installed_json(fake_install):
    paths = config.paths()
    assert inst.install(paths, yes=True, force=False, say=lambda s: None) == 0
    venv, torch_cmd, airllm_cmd, smoke = fake_install
    assert venv[1:3] == ["venv", "--no-project"] and "3.12" in venv
    assert "torch==2.11.0" in torch_cmd and torch_cmd[-2:] == ["--default-index", "https://download.pytorch.org/whl/cu128"]
    assert inst.AIRLLM_SPEC in airllm_cmd and "transformers==5.16.1" in airllm_cmd
    assert not any(a.startswith("torch") for a in airllm_cmd[3:])  # the cu128 build installed first stays
    assert smoke[1:3] == ["-I", "-c"]
    data = inst.read_installed(paths)
    assert data["commit"] == inst.AIRLLM_COMMIT and data["torch_index"] == "cu128"
    assert inst.status(paths).status == "installed"
    assert inst.install(paths, yes=True, force=False, say=lambda s: None) == 0 and len(fake_install) == 4


def test_install_rejects_a_torch_build_a_dependency_swapped_in(fake_install, monkeypatch):
    def capture(cmd, cwd, timeout):
        info = {"torch": "2.11.0", "cuda_available": False, "airllm_commit": inst.AIRLLM_COMMIT, "packages": {}}
        return subprocess.CompletedProcess(cmd, 0, stdout=inst.SMOKE_MARKER + json.dumps(info), stderr="")

    monkeypatch.setattr(inst, "_capture", capture)
    said: list[str] = []
    assert inst.install(config.paths(), yes=True, force=False, say=said.append) == 1
    assert any("not 2.11.0+cu128" in s for s in said)
    assert inst.status(config.paths()).status == "not_installed"


# ============================================================================= worker protocol


class FakeBackend:
    """Stands in for AirLLM: yields fixed text pieces and records how generation ended."""

    def __init__(self, pieces=("Hel", "lo", " world"), fail_load: str | None = None, delay: float = 0.0) -> None:
        self.pieces, self.fail_load, self.delay = pieces, fail_load, delay
        self.opts: list[w.GenOptions] = []
        self.cancel: threading.Event | None = None
        self.closed = threading.Event()

    def load(self, on_phase):
        on_phase(w.PHASE_IMPORT)
        on_phase(w.PHASE_BUILD)
        if self.fail_load:
            raise RuntimeError(self.fail_load)
        return {"device": "cuda:0", "device_name": "Fake GPU", "resident_bytes": 272 * MiB, "torch": "2.11.0+cu128"}

    def encode(self, messages):
        return w.Prompt(n_tokens=sum(len(m["content"].split()) for m in messages), data=messages)

    def generate(self, prompt, opts, cancel):
        self.opts.append(opts)
        self.cancel = cancel
        n = 0
        try:
            for piece in self.pieces[:opts.max_tokens]:
                if cancel.is_set():
                    break
                time.sleep(self.delay)
                n += 1
                yield piece
            yield w.Done("length" if n >= opts.max_tokens else "stop", prompt.n_tokens, n, 12.5, 40.0)
        finally:
            self.closed.set()


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(backend: FakeBackend, max_seq_len: int = 64, load: bool = True):
        mdir = write_model(tmp_path / "model")
        worker = w.Worker(backend, alias="qwen-hf", key=KEY, max_seq_len=max_seq_len, model_dir=mdir,
                          shards_dir=tmp_path / "shards")
        if load:
            worker.load()
        server = w.build_server(worker, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return worker, f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


AUTH = {"Authorization": f"Bearer {KEY}"}
CHAT = {"messages": [{"role": "user", "content": "say hello"}], "max_tokens": 8}


def sse_payloads(raw: bytes) -> list[str]:
    parser = SSEParser()
    return [e.data for e in [*parser.feed(raw), *parser.flush()]]


def test_worker_rejects_missing_or_wrong_keys(serve):
    _, url = serve(FakeBackend())
    assert httpx.get(f"{url}/health").status_code == 401
    assert httpx.get(f"{url}/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert httpx.post(f"{url}/v1/chat/completions", json=CHAT).status_code == 401
    assert httpx.get(f"{url}/health", headers=AUTH).status_code == 200


def test_health_reports_loading_ready_and_failure(serve):
    worker, url = serve(FakeBackend(), load=False)
    r = httpx.get(f"{url}/health", headers=AUTH)
    assert r.status_code == 503 and r.json()["phase"] == "starting"
    assert httpx.post(f"{url}/v1/chat/completions", json=CHAT, headers=AUTH).status_code == 503
    worker.load()
    ok = httpx.get(f"{url}/health", headers=AUTH).json()
    assert ok["status"] == "ok" and ok["device"] == "cuda:0" and ok["resident_bytes"] == 272 * MiB
    _, bad_url = serve(FakeBackend(fail_load="CUDA out of memory"))
    r = httpx.get(f"{bad_url}/health", headers=AUTH)
    assert r.status_code == 500 and "CUDA out of memory" in r.json()["error"]


def test_health_shows_split_progress_on_first_start(serve, tmp_path):
    worker, url = serve(FakeBackend(), load=False)
    worker.split_total = w.expected_modules(config_of(SMALL))
    split = tmp_path / "shards" / w.SPLIT_DIR_NAME
    split.mkdir(parents=True)
    for i in range(3):
        (split / f"model.layers.{i}.safetensors.done").write_bytes(b"")
    worker._on_phase(w.PHASE_BUILD)
    body = httpx.get(f"{url}/health", headers=AUTH).json()
    assert body["phase"] == "splitting" and body["split"] == {"done": 3, "total": 26}
    assert "3/26" in adapter_mod.describe_phase(body)


def test_stream_is_openai_sse_with_usage_and_timings(serve):
    backend = FakeBackend()
    _, url = serve(backend)
    r = httpx.post(f"{url}/v1/chat/completions", json={**CHAT, "stream": True}, headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
    data = sse_payloads(r.content)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    assert [c["choices"][0]["delta"].get("content") for c in chunks[1:-1]] == ["Hel", "lo", " world"]
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop" and final["model"] == "qwen-hf"
    assert final["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert final["timings"]["predicted_n"] == 3 and final["timings"]["prompt_ms"] == 12.5
    assert final["timings"]["predicted_per_second"] == 75.0
    stats = StreamStats()  # the gateway records runs from exactly these payloads
    for c in chunks:
        stats.observe(c)
    assert stats.finish_reason == "stop" and stats.usage["completion_tokens"] == 3 and stats.timings
    assert backend.opts[0].temperature == 0.0 and backend.opts[0].max_tokens == 8


def test_non_stream_returns_one_completion(serve):
    _, url = serve(FakeBackend())
    body = httpx.post(f"{url}/v1/chat/completions", json=CHAT, headers=AUTH).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert body["usage"]["completion_tokens"] == 3 and body["timings"]["prompt_n"] == 2


def test_one_request_at_a_time(serve):
    worker, url = serve(FakeBackend())
    worker.busy.acquire()
    try:
        r = httpx.post(f"{url}/v1/chat/completions", json=CHAT, headers=AUTH)
    finally:
        worker.busy.release()
    assert r.status_code == 429 and r.json()["error"]["code"] == "busy"


@pytest.mark.parametrize("patch", [
    {"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
    {"max_tokens": 0}, {"max_tokens": 65}, {"max_tokens": True}, {"temperature": -1}, {"top_p": 0},
    {"stop": ["a", "b", "c", "d", "e"]}, {"stop": [""]}, {"stream": "yes"},
])
def test_invalid_requests_are_400(serve, patch):
    _, url = serve(FakeBackend())
    r = httpx.post(f"{url}/v1/chat/completions", json={**CHAT, **patch}, headers=AUTH)
    assert r.status_code == 400, r.text


def test_malformed_bodies_are_rejected(serve):
    _, url = serve(FakeBackend())
    assert httpx.post(f"{url}/v1/chat/completions", content=b"{nope", headers=AUTH).status_code == 400
    assert httpx.post(f"{url}/v1/chat/completions", json=[1], headers=AUTH).status_code == 400
    assert httpx.post(f"{url}/v1/nope", json=CHAT, headers=AUTH).status_code == 404


def test_rejections_read_the_body_first_so_the_client_never_sees_a_reset(serve):
    """Replying with unread request bytes makes Windows reset the socket; a 1 MiB body makes that race likely."""
    _, url = serve(FakeBackend())
    big = b"x" * (1024 * 1024)
    for _ in range(20):
        assert httpx.post(f"{url}/v1/nope", content=big, headers=AUTH).status_code == 404
        assert httpx.post(f"{url}/v1/chat/completions", content=big, headers={}).status_code == 401


class BufferingBackend(FakeBackend):
    """Like the real streamer holding back text: nothing is written until generation ends or is cancelled."""

    def generate(self, prompt, opts, cancel):
        self.cancel = cancel
        started = time.monotonic()
        cancel.wait(20)
        self.waited = time.monotonic() - started
        yield w.Done("stop", prompt.n_tokens, 0, 1.0, 1.0)


def test_a_client_that_disconnects_cancels_generation_without_waiting_for_the_next_write(serve):
    backend = BufferingBackend()
    _, url = serve(backend)
    with httpx.stream("POST", f"{url}/v1/chat/completions", json={**CHAT, "stream": True}, headers=AUTH) as r:
        assert r.status_code == 200
        next(r.iter_lines())  # the role chunk: generation has started
    deadline = time.monotonic() + 5
    while backend.cancel is None or not backend.cancel.is_set():
        assert time.monotonic() < deadline, "generation was not cancelled after the client left"
        time.sleep(0.05)
    while not hasattr(backend, "waited"):
        time.sleep(0.05)
    assert backend.waited < 3  # well before the 20 s the backend would otherwise wait


def test_prompt_plus_output_is_bounded_by_max_seq_len(serve):
    backend = FakeBackend(pieces=tuple("abcdefgh"))
    _, url = serve(backend, max_seq_len=8)
    long = {"messages": [{"role": "user", "content": " ".join(["w"] * 8)}], "max_tokens": 4}
    r = httpx.post(f"{url}/v1/chat/completions", json=long, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    body = httpx.post(f"{url}/v1/chat/completions", headers=AUTH,
                      json={"messages": [{"role": "user", "content": "a b c"}], "max_tokens": 8}).json()
    assert backend.opts[-1].max_tokens == 5 and body["choices"][0]["finish_reason"] == "length"


def test_stop_strings_end_generation_and_are_not_emitted(serve):
    backend = FakeBackend(pieces=("Hello", " wor", "ld EN", "D tail", " more"))
    _, url = serve(backend)
    r = httpx.post(f"{url}/v1/chat/completions", json={**CHAT, "stop": "END", "stream": True}, headers=AUTH)
    chunks = [json.loads(d) for d in sse_payloads(r.content)[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == "Hello world " and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert backend.cancel.is_set()


def test_stop_scanner_holds_back_a_partial_match():
    s = w.StopScanner(("bc",))
    assert s.feed("ab") == "a" and s.feed("x") == "bx" and s.feed("b") == "" and s.feed("c") == ""
    assert s.hit and s.flush() == ""
    plain = w.StopScanner(("zz",))
    assert plain.feed("az") == "a" and plain.flush() == "z"


def test_closing_the_stream_cancels_generation_and_frees_the_slot(serve):
    backend = FakeBackend(pieces=tuple("x" * 40), delay=0.05)
    worker, url = serve(backend)
    with httpx.Client() as client, client.stream("POST", f"{url}/v1/chat/completions", headers=AUTH,
                                                 json={**CHAT, "max_tokens": 40, "stream": True}) as resp:
        next(resp.iter_raw())
    assert backend.closed.wait(5) and backend.cancel.is_set()
    deadline = time.monotonic() + 5
    while not worker.busy.acquire(blocking=False):
        assert time.monotonic() < deadline, "the request slot was never released"
        time.sleep(0.02)
    worker.busy.release()


def test_checkpoint_checks_refuse_custom_code_and_pickles(tmp_path):
    mdir = write_model(tmp_path / "m")
    assert w.check_checkpoint(mdir)["architectures"] == ["Qwen2ForCausalLM"]
    (mdir / "modeling_evil.py").write_text("x = 1", encoding="utf-8")
    with pytest.raises(ValueError, match="custom code"):
        w.check_checkpoint(mdir)
    (mdir / "modeling_evil.py").unlink()
    (mdir / "pytorch_model.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="pickled"):
        w.check_checkpoint(mdir)
    other = write_model(tmp_path / "o")
    (other / "config.json").write_text(json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="allowlist"):
        w.check_checkpoint(other)


def test_expected_modules_drop_a_tied_lm_head():
    assert w.expected_modules(config_of(SMALL)) == 26  # 24 layers + embed + norm; lm_head is tied
    assert w.expected_modules(config_of(LARGE)) == 67


def test_worker_cli_refuses_non_loopback_and_missing_key(monkeypatch, tmp_path):
    base = ["--model-dir", str(tmp_path), "--shards-dir", str(tmp_path), "--port", "5000", "--alias", "a"]
    with pytest.raises(SystemExit):
        w.parse_args([*base, "--host", "0.0.0.0"])
    monkeypatch.delenv(w.KEY_ENV, raising=False)
    assert w.main(base) == 2


def test_worker_imports_only_the_stdlib_at_module_level():
    tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
    top = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
           for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom) for a in n.names}
    assert "hypercompute" not in top and not top & {"torch", "airllm", "transformers"}
    assert top <= set(sys.stdlib_module_names) | {"__future__"}


# ============================================================================= adapter


class FakeProc:
    def __init__(self, alive: bool = True, log: str = "") -> None:
        self.pid, self._alive, self._log = 4242, alive, log

    def alive(self) -> bool:
        return self._alive

    def read_log(self, max_bytes: int = 0) -> str:
        return self._log


class FakeSupervisor:
    def __init__(self) -> None:
        self.started: list[tuple] = []
        self.procs: dict[str, FakeProc] = {}

    def start(self, key, exe, args, cwd, log_path, env=None):
        self.started.append((key, exe, list(args), cwd, log_path, dict(env or {})))
        self.procs[key] = FakeProc()
        return self.procs[key]

    def get(self, key):
        return self.procs.get(key)


def ctx_with(sup=None) -> registry.AdapterContext:
    paths = config.paths()
    return registry.AdapterContext(supervisor=sup or FakeSupervisor(), logs_dir=paths.home / "logs", paths=paths)


def local_node(**inventory) -> NodeView:
    return NodeView(id="local", display_name="g14 (this machine)", is_local=True, session_id=None,
                    state=NodeState.eligible, state_reason=None, joined_at=utcnow(), last_heartbeat_at=utcnow(),
                    heartbeat_age_s=0.0, inventory=make_inventory(**inventory),
                    contribution=contribution(["CUDA0"], max_gib=10, reserve_gib=4), worker=None, worker_state=None,
                    verification=None)


def spec_for(model_id: str = SMALL) -> DeploymentSpec:
    art = manifest()[model_id]
    plan = adapter_mod.plan(PlanRequest(model_id=model_id), art, local_node(), [], ctx_with())
    mdir = config.paths().models / model_id
    return DeploymentSpec(deployment_id="d" * 32, plan=plan, model_path=mdir, alias=model_id, artifact=art,
                          model_dir=mdir)


def test_build_needs_the_install():
    assert adapter_mod.build(ctx_with()) is None
    mark_installed(config.paths())
    a = adapter_mod.build(ctx_with())
    assert a is not None and a.engine_id == "airllm" and a.idle_timeout_s >= 600


@pytest.mark.anyio
async def test_prepare_starts_the_worker_with_the_key_in_env_not_argv():
    paths = config.paths()
    mark_installed(paths)
    sup = FakeSupervisor()
    a = adapter_mod.AirLlmAdapter(ctx_with(sup))
    try:
        handle = await a.prepare(spec_for())
    finally:
        await a.aclose()
    key, exe, args, cwd, log_path, env = sup.started[0]
    assert key == "deployment:" + "d" * 32 and exe == inst.venv_python(paths) and cwd == inst.engine_dir(paths)
    assert args[:3] == ["-I", "-u", str(adapter_mod.WORKER_SCRIPT)]
    assert args[args.index("--host") + 1] == "127.0.0.1" and args[args.index("--max-seq-len") + 1] == "4096"
    assert args[args.index("--device") + 1] == "cuda:0"
    shards = Path(args[args.index("--shards-dir") + 1])
    assert shards == paths.home / "airllm-shards" / SMALL / "7ae557604adf" and shards.is_dir()
    assert env[w.KEY_ENV] == handle.api_key and len(handle.api_key) >= 16
    assert handle.api_key not in " ".join(args) and env["HF_HUB_OFFLINE"] == "1"
    assert handle.base_url == f"http://127.0.0.1:{args[args.index('--port') + 1]}"
    assert log_path.name == f"airllm-worker-{'d' * 32}.log"


@pytest.mark.anyio
async def test_validate_checks_checkpoint_and_venv():
    a = adapter_mod.AirLlmAdapter(ctx_with())
    try:
        bad = await a.validate(spec_for())
        assert not bad.ok and any("model directory" in p for p in bad.problems)
        assert any("venv" in p for p in bad.problems)
        mark_installed(config.paths())
        write_model(config.paths().models / SMALL)
        assert (await a.validate(spec_for())).ok
    finally:
        await a.aclose()


def handle_for(url: str, key: str = KEY) -> DeploymentHandle:
    return DeploymentHandle(deployment_id="d" * 32, process_key="deployment:" + "d" * 32, base_url=url,
                            log_path=Path("worker.log"), pid=4242, args=[], api_key=key)


@pytest.mark.anyio
async def test_wait_ready_reports_phases_then_returns(serve):
    worker, url = serve(FakeBackend(), load=False)
    sup = FakeSupervisor()
    sup.procs["deployment:" + "d" * 32] = FakeProc()
    a = adapter_mod.AirLlmAdapter(ctx_with(sup))
    seen: list[str] = []
    threading.Timer(1.2, worker.load).start()
    try:
        await a.wait_ready(handle_for(url), timeout_s=15, on_progress=seen.append)
    finally:
        await a.aclose()
    assert seen and seen[0] == "starting the AirLLM worker"


@pytest.mark.anyio
async def test_wait_ready_raises_the_load_error(serve):
    _, url = serve(FakeBackend(fail_load="CUDA out of memory"))
    sup = FakeSupervisor()
    sup.procs["deployment:" + "d" * 32] = FakeProc()
    a = adapter_mod.AirLlmAdapter(ctx_with(sup))
    try:
        with pytest.raises(adapter_mod.EngineAdapterError, match="CUDA out of memory") as err:
            await a.wait_ready(handle_for(url), timeout_s=10)
        assert err.value.code == "load_failed" and err.value.fix
    finally:
        await a.aclose()


@pytest.mark.anyio
async def test_wait_ready_reports_a_dead_worker_from_its_log():
    sup = FakeSupervisor()
    crash = "loading\nModuleNotFoundError: No module named 'airllm'\n"
    sup.procs["deployment:" + "d" * 32] = FakeProc(alive=False, log=crash)
    a = adapter_mod.AirLlmAdapter(ctx_with(sup))
    try:
        with pytest.raises(adapter_mod.EngineAdapterError, match="No module named 'airllm'"):
            await a.wait_ready(handle_for("http://127.0.0.1:9"), timeout_s=5)
    finally:
        await a.aclose()


@pytest.mark.anyio
async def test_adapter_streams_observes_and_cancels(serve):
    backend = FakeBackend(pieces=tuple("abcdefghij" * 4), delay=0.05)
    _, url = serve(backend)
    a = adapter_mod.AirLlmAdapter(ctx_with())
    handle = handle_for(url)
    try:
        report = await a.observe(handle, spec_for(), ValidationResult(ok=True, problems=[]))
        assert report.verified and report.observed.devices[0].name == "cuda:0 (Fake GPU)"
        assert report.observed.devices[0].model_bytes == 272 * MiB
        req = InferenceRequest("r1", {**CHAT, "max_tokens": 40}, True)
        events = a.stream(handle, req)
        first = await events.__anext__()
        assert first.payload["choices"][0]["delta"]["role"] == "assistant"
        result = await a.cancel("r1")
        assert result.cancelled and "token in progress" in result.detail
        await events.aclose()
        assert backend.closed.wait(5) and backend.cancel.is_set()
        assert not (await a.cancel("r1")).cancelled
    finally:
        await a.aclose()


@pytest.mark.anyio
async def test_adapter_stream_reports_busy_worker(serve):
    worker, url = serve(FakeBackend())
    a = adapter_mod.AirLlmAdapter(ctx_with())
    worker.busy.acquire()
    try:
        with pytest.raises(adapter_mod.EngineAdapterError) as err:
            async for _ in a.stream(handle_for(url), InferenceRequest("r2", CHAT, True)):
                pass
        assert err.value.code == "engine_busy"
    finally:
        worker.busy.release()
        await a.aclose()


class OomBackend(FakeBackend):
    def generate(self, prompt, opts, cancel):
        yield "Hel"
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


@pytest.mark.anyio
async def test_adapter_raises_the_workers_mid_stream_error(serve):
    _, url = serve(OomBackend())
    a = adapter_mod.AirLlmAdapter(ctx_with())
    try:
        with pytest.raises(adapter_mod.EngineAdapterError, match="mid-stream: CUDA out of memory") as err:
            async for _ in a.stream(handle_for(url), InferenceRequest("r3", CHAT, True)):
                pass
        assert err.value.code == "upstream_error"
    finally:
        await a.aclose()


# ============================================================================= plan


def plan_of(model_id: str = SMALL, *, node: NodeView | None = None, remote: list | None = None, **req):
    return adapter_mod.plan(PlanRequest(model_id=model_id, **req), manifest()[model_id], node or local_node(),
                            remote or [], ctx_with())


def ready(model_id: str = SMALL, **installed: object) -> None:
    mark_installed(config.paths(), **installed)
    write_model(config.paths().models / model_id, model_id)


def test_plan_not_installed_is_infeasible_with_the_fix():
    write_model(config.paths().models / SMALL)
    p = plan_of()
    assert not p.feasible and any("engine install --engine airllm" in r for r in p.reasons)


def test_plan_only_serves_hf_artifacts():
    ready()
    gguf = manifest()[SMALL].model_copy(update={"format": "gguf", "engines": []})
    p = adapter_mod.plan(PlanRequest(model_id=SMALL), gguf, local_node(), [], ctx_with())
    assert not p.feasible and any("HF safetensors" in r for r in p.reasons)


def test_plan_fits_on_the_local_gpu_as_the_disk_tier():
    ready()
    p = plan_of()
    assert p.feasible, p.reasons
    assert (p.engine, p.tier, p.capacity_mode, p.fit_label) == ("airllm", "disk", True, "estimated_fit")
    assert p.device_list == [] and p.tensor_split == [] and p.rpc_endpoints == []
    [a] = p.assignments
    assert (a.node_id, a.device, a.pool_id, a.fits) == ("local", "CUDA0", "vram:CUDA0", True)
    assert a.weight_bytes == 151936 * 896 * 2 + 29_824_768  # resident tied embedding + one decoder layer
    assert a.layers == list(range(24)) and a.total_bytes_est < a.budget_bytes
    assert p.engine_options["device"] == "cuda:0" and p.engine_options["split_ready"] is False
    assert p.engine_options["shards_dir"].endswith(str(Path("airllm-shards") / SMALL / "7ae557604adf"))
    assert any("per token" in x for x in p.warnings) and any("splits the checkpoint" in x for x in p.warnings)


def test_plan_needs_disk_for_the_split(monkeypatch):
    ready()
    monkeypatch.setattr(adapter_mod.shutil, "disk_usage", lambda p: Usage(0, 0, GiB))
    p = plan_of()
    assert not p.feasible and p.fit_label == "does_not_fit"
    assert any("not enough disk" in r for r in p.reasons)


def test_a_finished_split_needs_no_more_disk(monkeypatch):
    ready()
    split = adapter_mod.shards_dir(config.paths(), manifest()[SMALL]) / w.SPLIT_DIR_NAME
    split.mkdir(parents=True)
    for i in range(26):
        (split / f"m{i}.safetensors.done").write_bytes(b"")
    monkeypatch.setattr(adapter_mod.shutil, "disk_usage", lambda p: Usage(0, 0, MiB))
    p = plan_of()
    assert p.feasible and p.engine_options["split_ready"] is True
    assert not any("splits the checkpoint" in x for x in p.warnings)


def test_plan_checks_vram_and_host_ram():
    ready()
    tiny_gpu = plan_of(node=local_node(vram_free=600 * MiB))
    assert not tiny_gpu.feasible and any("CUDA0 needs" in r for r in tiny_gpu.reasons)
    tiny_ram = plan_of(node=local_node(ram_free=5 * GiB))
    assert not tiny_ram.feasible and any("host RAM" in r for r in tiny_ram.reasons)


def test_large_model_footprint_and_speed_warning():
    fp = adapter_mod.footprint(config_of(LARGE), 4096, 65_539_393_631)
    assert fp.module_bytes == 152064 * 5120 * 2 and fp.resident_bytes == 0  # untied embedding is the largest module
    assert fp.kv_bytes == 4096 * 64 * 2 * 8 * 128 * 2
    ready(LARGE)
    p = plan_of(LARGE)
    assert any("min" in x and "per token" in x for x in p.warnings)
    assert p.engine_options["s_per_token"][1] > 60


def test_plan_respects_capacity_mode_and_local_only_scope():
    ready()
    off = plan_of(allow_capacity_mode=False)
    assert not off.feasible and any("allow_capacity_mode" in r for r in off.reasons)
    peers = plan_of(remote=[local_node()], strategy=PlanStrategy.all_selected)
    assert any("peer(s) are not used" in x for x in peers.warnings)
    stale = local_node().model_copy(update={"state": NodeState.stale, "state_reason": "no heartbeat"})
    assert not plan_of(node=stale).feasible


def test_plan_before_download_is_a_rough_guess():
    mark_installed(config.paths())
    p = plan_of()
    assert p.fit_label == "fit_unknown" and any("rough guess" in x for x in p.warnings)


def test_cpu_only_install_plans_on_host_ram():
    ready(cuda_available=False, device="cpu", device_name=None, arch_ok=None, torch="2.11.0+cpu")
    p = plan_of()
    [a] = p.assignments
    assert p.engine_options["device"] == "cpu" and (a.device, a.pool_id) == ("CPU", "ram")
    assert any("CPU it is slower" in x for x in p.warnings)


def test_unsafe_manifest_files_are_refused():
    ready()
    art = manifest()[SMALL]
    pickle = art.files[0].model_copy(update={"name": "x.bin", "path": "x.bin"})
    bad = art.model_copy(update={"files": [*art.files, pickle]})
    p = adapter_mod.plan(PlanRequest(model_id=SMALL), bad, local_node(), [], ctx_with())
    assert not p.feasible and any("x.bin" in r for r in p.reasons)


def test_registry_reaches_the_module():
    ready()
    assert registry.plan_for("airllm", PlanRequest(model_id=SMALL), manifest()[SMALL], local_node(), [],
                             ctx_with()).engine == "airllm"
    statuses = {s.engine: s for s in registry.engine_status(ctx_with())}
    assert statuses["airllm"].status == "installed"


def test_manifest_fixture_pins_every_file():
    for art in manifest().values():
        assert art.format == "hf" and art.engines == ["airllm"] and len(art.revision) == 40
        assert {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"} <= {
            f.path for f in art.files}
        for f in art.files:
            assert f.url == f"https://huggingface.co/{art.repo}/resolve/{art.revision}/{f.path}"


def test_idle_timeout_grows_with_the_slow_estimate():
    assert adapter_mod.idle_timeout({}) == adapter_mod.IDLE_TIMEOUT_S
    assert adapter_mod.idle_timeout({"s_per_token": [13.1, 312.1]}) == pytest.approx(936.3)
    assert adapter_mod.idle_timeout({"s_per_token": "garbage"}) == adapter_mod.IDLE_TIMEOUT_S
