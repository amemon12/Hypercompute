"""`hypercompute tune`: the request body, the parser flags, and what the CLI prints (fake host, no gateway)."""

import argparse
import json

import httpx
import pytest
from hypercompute.cli import build_parser
from hypercompute.schemas import PlanStrategy, TuneCandidate, TuneRequest, TuneResult, utcnow
from hypercompute.tune_cli import request_body, run_tune
from test_tuner import LOCAL_FIT, MODEL, PEER


def cli_args(**kw):
    base = dict(model_id=MODEL, port=8123, nodes=None, objective="interactive", ctx=4096, max_candidates=4,
                no_flags=False)
    return argparse.Namespace(**{**base, **kw})


def _doc(status, cand_status, best=None, profile=None, note=None):
    cands = [TuneCandidate(label="planner pick (gpu): CUDA0 x28+head", plan=LOCAL_FIT, status=cand_status[0],
                           load_s=12.3, ttft_ms=85.0, decode_tokens_per_s=79.1, prefill_tokens_per_s=2450.0,
                           score=3.32),
             TuneCandidate(label="flash attention off: CUDA0 x28+head", plan=LOCAL_FIT, status=cand_status[1],
                           error="out of memory" if cand_status[1] == "failed" else None)]
    return TuneResult(id="f" * 32, request=TuneRequest(model_id=MODEL), status=status, candidates=cands,
                      best_index=best, profile_id=profile, created_at=utcnow(), note=note).model_dump(mode="json")


def test_cli_prints_each_candidate_and_the_winner(capsys):
    polls = [_doc("running", ["done", "running"]), _doc("done", ["done", "failed"], best=0, profile="p" * 32)]
    seen = {}

    def handler(request):
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        if request.method == "POST":
            seen["owner"] = request.headers.get("x-hypercompute-owner")
            seen["body"] = json.loads(request.content)
            return httpx.Response(202, json=_doc("running", ["pending", "pending"]))
        return httpx.Response(200, json=polls.pop(0))

    client = httpx.Client(base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    assert run_tune(cli_args(nodes=f"{PEER}, ", no_flags=True), client=client, sleep=lambda s: None) == 0
    out = capsys.readouterr().out
    assert seen["owner"] == "tok"
    assert seen["body"] == {"model_id": MODEL, "objective": "interactive", "ctx_size": 4096, "max_candidates": 4,
                            "tune_flags": False, "node_ids": [PEER]}
    assert "load 12.3 s  TTFT 85 ms  decode 79.1 tok/s  prefill 2450 tok/s  score 3.32 s" in out
    assert "failed: out of memory" in out
    assert out.count("planner pick") == 2 and "Winner: #1" in out and "profile pppppppp" in out


def test_cli_reports_a_host_that_is_not_running(capsys):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    assert run_tune(cli_args(), client=client, sleep=lambda s: None) == 2
    out = capsys.readouterr().out
    assert "no Hypercompute host is running on 127.0.0.1:8123" in out and "hypercompute host" in out


def test_cli_shows_the_gateway_refusal_and_fix(capsys):
    def handler(request):
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        return httpx.Response(409, json={"detail": "deployment 1234abcd is ready; tuning needs the GPUs to itself",
                                         "code": "deployment_active", "fix": "stop the running model first"})

    client = httpx.Client(base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    assert run_tune(cli_args(), client=client, sleep=lambda s: None) == 1
    out = capsys.readouterr().out
    assert "HTTP 409: deployment 1234abcd is ready" in out and "fix:   stop the running model first" in out


def test_cli_request_body_passes_decode_tokens_when_set():
    assert request_body(cli_args(decode_tokens=128))["decode_tokens"] == 128
    assert "node_ids" not in request_body(cli_args())


def test_cli_request_body_passes_the_owners_plan_limits():
    body = request_body(cli_args(engine="vllm", strategy="local_only", no_ram_tier=True, allow_capacity_mode=True))
    limits = {k: body[k] for k in ("engine", "strategy", "allow_ram_tier", "allow_capacity_mode")}
    assert limits == {"engine": "vllm", "strategy": "local_only", "allow_ram_tier": False, "allow_capacity_mode": True}
    assert TuneRequest.model_validate(body).strategy == PlanStrategy.local_only  # the gateway accepts it


def test_cli_ctrl_c_cancels_the_tune_on_the_host(capsys):
    deleted = []

    def handler(request):
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        if request.method == "POST":
            return httpx.Response(202, json=_doc("running", ["running", "pending"]))
        deleted.append((request.method, request.headers.get("x-hypercompute-owner")))
        return httpx.Response(200, json=_doc("cancelled", ["skipped", "skipped"]))

    def interrupt(_s):
        raise KeyboardInterrupt

    client = httpx.Client(base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    assert run_tune(cli_args(), client=client, sleep=interrupt) == 130
    assert deleted == [("DELETE", "tok")] and "Tune cancelled" in capsys.readouterr().out


def test_cli_prints_why_the_winner_was_not_saved_as_a_profile(capsys):
    note = (f"the winner runs vllm on q-awq, and plans replay only llama.cpp splits "
            f"of {MODEL}")

    def handler(request):
        if request.url.path == "/api/owner/bootstrap":
            return httpx.Response(200, json={"owner_token": "tok"})
        return httpx.Response(202, json=_doc("done", ["done", "failed"], best=0, note=note))

    client = httpx.Client(base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    assert run_tune(cli_args(engine="vllm"), client=client, sleep=lambda s: None) == 0
    out = capsys.readouterr().out
    assert "Winner: #1" in out and "saved as profile" not in out and f"Note: {note}" in out


def test_the_tune_command_has_a_flag_for_each_plan_limit_request_body_reads():
    args = build_parser().parse_args(["tune", MODEL, "--engine", "vllm", "--strategy", "local_only", "--no-ram-tier",
                                      "--allow-capacity-mode"])
    body = request_body(args)
    limits = {k: body[k] for k in ("engine", "strategy", "allow_ram_tier", "allow_capacity_mode")}
    assert limits == {"engine": "vllm", "strategy": "local_only", "allow_ram_tier": False, "allow_capacity_mode": True}
    assert TuneRequest.model_validate(body).strategy == PlanStrategy.local_only
    unset = request_body(build_parser().parse_args(["tune", MODEL]))  # TuneRequest's defaults apply
    assert not unset.keys() & {"engine", "strategy", "allow_ram_tier", "allow_capacity_mode"}
    with pytest.raises(SystemExit):
        build_parser().parse_args(["tune", MODEL, "--engine", "llama"])  # the API's engine ids only
