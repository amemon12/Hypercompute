"""scripts/vision_check.py: the live check of image input on a running gateway (read-only apart from two chats)."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEP_ID = "d" * 32
EP5, EP6 = "10.0.0.5:47052", "10.0.0.6:47052"
MMPROJ = "C:/h/models/gemma-3-27b-it-q4_k_m/mmproj-google_gemma-3-27b-it-f16.gguf"
LOG = """0.01.000.000 I clip_model_loader: model name:   Gemma-3-27B-It
0.09.000.000 I sched_reserve:      CUDA0 compute buffer size =   522.00 MiB
0.10.000.000 I clip_ctx: CLIP using CUDA0 backend
0.10.500.000 I reserve_compute_meta:      CUDA0 compute buffer size =  1100.25 MiB
0.10.600.000 I reserve_compute_meta:        CPU compute buffer size =     9.19 MiB
0.11.000.000 I srv    load_model: loaded multimodal model, 'C:/h/models/gemma-3-27b-it-q4_k_m/mmproj.gguf'
"""


def _script():
    name = "vision_check_script"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "vision_check.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


vc = _script()


def assignment(device: str, endpoint: str | None) -> dict[str, Any]:
    return {"device": device, "endpoint": endpoint, "layers": [0]}


def deployment(mmdev: str | None = "CUDA0", state: str = "ready", pooled: bool = True) -> dict[str, Any]:
    peers = [assignment("RPC0", EP5), assignment("RPC1", EP6)] if pooled else []
    args = ["-m", "m.gguf", *(["--rpc", f"{EP5},{EP6}"] if pooled else []),
            "--device", ",".join([*(["RPC0", "RPC1"] if pooled else []), "CUDA0"])]
    if mmdev is not None:
        args += ["--mmproj", MMPROJ, "-mmdev", mmdev]
    return {"id": DEP_ID, "state": state, "engine_args": args,
            "plan": {"model_id": "gemma-3-27b-it-q4_k_m", "engine": "llama_rpc",
                     "assignments": [*peers, assignment("CUDA0", None)]}}


def gateway(monkeypatch, deps: list[dict], chat_status: int = 200) -> list[dict]:
    chats: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        if request.url.path == "/api/deployments":
            return httpx.Response(200, json=deps)
        body = json.loads(request.content)
        chats.append(body)
        if chat_status != 200:
            return httpx.Response(chat_status, json={"code": "vision_unavailable", "detail": "no projector"})
        has_image = isinstance(body["messages"][0]["content"], list)
        return httpx.Response(200, json={
            "model": "gemma-3-27b-it-q4_k_m",
            "choices": [{"message": {"content": "A dashboard." if has_image else "ready"}}],
            "timings": {"prompt_n": 290, "prompt_ms": 2900.0, "prompt_per_second": 100.0, "predicted_n": 8,
                        "predicted_ms": 1000.0, "predicted_per_second": 8.0}})

    real = httpx.Client
    monkeypatch.setattr(vc.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return chats


def png(tmp_path: Path) -> Path:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII5s", 13, b"IHDR", 64, 48, b"\x08\x02\x00\x00\x00"))
    return path


def home_with_log(tmp_path: Path, text: str = LOG) -> Path:
    logs = tmp_path / "home" / "coordinator" / "logs"
    logs.mkdir(parents=True)
    (logs / f"llama-server-{DEP_ID}.log").write_text(text, encoding="utf-8")
    return tmp_path / "home"


def run(tmp_path: Path, *extra: str) -> int:
    return vc.main(["--port", "8001", "--image", str(png(tmp_path)), *extra])


def test_a_pooled_deployment_with_the_projector_on_the_host_passes(tmp_path, monkeypatch, capsys):
    chats = gateway(monkeypatch, [deployment()])
    assert run(tmp_path, "--home", str(home_with_log(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "PASS deployment  gemma-3-27b-it-q4_k_m on this machine + 2 peer(s)" in out
    assert "PASS projector   -mmdev CUDA0 (local), peers RPC0, RPC1" in out
    assert "PASS log         CLIP using CUDA0 backend; loaded multimodal model" in out
    assert "1100 MiB" in out and "1024 MiB prior" in out
    assert "PASS image       A dashboard." in out and "290 prompt tokens" in out
    assert "PASS text        ready" in out
    assert isinstance(chats[0]["messages"][0]["content"], list)  # the image question first, then plain text
    assert chats[1]["messages"][0]["content"] == "Reply with: ready"


def test_a_projector_on_a_peer_fails(tmp_path, monkeypatch, capsys):
    gateway(monkeypatch, [deployment(mmdev="RPC0")])
    assert run(tmp_path) == 1
    assert "FAIL projector   -mmdev RPC0 is not a local device of the plan" in capsys.readouterr().out


def test_a_deployment_without_a_projector_fails_but_still_checks_text(tmp_path, monkeypatch, capsys):
    gateway(monkeypatch, [deployment(mmdev=None)], chat_status=400)
    assert run(tmp_path) == 1
    out = capsys.readouterr().out
    assert "FAIL projector   no --mmproj in the engine argv" in out


def test_no_ready_deployment_fails_without_chatting(tmp_path, monkeypatch, capsys):
    chats = gateway(monkeypatch, [deployment(state="loading")])
    assert run(tmp_path) == 1
    assert "FAIL deployment  no ready deployment" in capsys.readouterr().out and chats == []


def test_a_log_without_the_projector_lines_fails(tmp_path, monkeypatch, capsys):
    gateway(monkeypatch, [deployment()])
    assert run(tmp_path, "--home", str(home_with_log(tmp_path, "0.1 I sched_reserve: CUDA0 compute buffer\n"))) == 1
    assert "FAIL log" in capsys.readouterr().out


def test_without_home_the_log_check_is_skipped(tmp_path, monkeypatch, capsys):
    gateway(monkeypatch, [deployment(pooled=False)])
    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "INFO log" in out and "on this machine + 0 peer(s)" in out


def view(status: str = "downloaded", projector: str | None = "downloaded", mmproj: bool = True) -> dict[str, Any]:
    proj = {"name": "mmproj-google_gemma-3-27b-it-f16.gguf"} if mmproj else None
    art = {"id": "gemma-3-27b-it-q4_k_m", "mmproj": proj}
    out = {"artifact": art, "status": status}
    return {**out, "projector_status": projector} if projector is not None else out


@pytest.mark.parametrize(("views", "rc", "needle"), [
    ([view()], 0, "PASS catalog     gemma-3-27b-it-q4_k_m downloaded; projector mmproj-google_gemma-3-27b-it-f16.gguf"),
    ([view(projector="not_downloaded")], 1, "models pull gemma-3-27b-it-q4_k_m"),
    ([view(status="not_downloaded")], 1, "download the model first"),
    ([view(mmproj=False)], 1, "HYPERCOMPUTE_CONFIG_DIR"),  # an older catalog (start-host.ps1's OneDrive configs)
    ([view(projector=None)], 1, "HYPERCOMPUTE_CONFIG_DIR"),  # a gateway still on code without projector support
    ([], 1, "is not in the gateway's catalog"),
])
def test_preflight_checks_the_catalog_and_the_projector_before_any_deployment(tmp_path, monkeypatch, capsys, views,
                                                                              rc, needle):
    client_type = httpx.Client
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        return httpx.Response(200, json=views)

    monkeypatch.setattr(vc.httpx, "Client", lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw))
    assert run(tmp_path, "--preflight", "gemma-3-27b-it-q4_k_m") == rc
    assert needle in capsys.readouterr().out
    assert paths == ["/api/owner/bootstrap", "/api/models"]  # no deployment read, no chat


@pytest.mark.parametrize("text", [LOG])
def test_log_facts_reads_the_clip_backend_and_its_buffers(text):
    facts = vc.log_facts(text)
    assert facts["backend"] == "CUDA0" and facts["loaded"] is True
    assert facts["buffers"] == {"CUDA0": 1100.25, "CPU": 9.19}
