import pytest

from hc_agent.intent import Intent
from hc_agent.planner import (
    ModelSummary,
    NoMatchingModel,
    choose,
    missing_peers,
    rank_candidates,
    read_plan,
    snapshot,
    summarize_models,
    with_observed,
)
from mock_gateway.server import MockState


def _model(mid: str, params: float, role: str = "standard", usable: bool = True) -> ModelSummary:
    return ModelSummary(id=mid, name=mid, family=mid.split("-")[0], params_b=params, quantization="Q", size_bytes=1,
                        usable=usable, role=role, n_layer=10, status="downloaded" if usable else "not_downloaded")


def test_recorded_kimi_plan_reads_into_stages(recorded: dict) -> None:
    pool = snapshot(recorded["nodes"], recorded["network"], [], [])
    assert pool.host is not None and pool.host.name == "gx10-49d1"
    assert {n.name for n in pool.online_peers} == {"mac-a", "mac-b", "mac-azim", "g14"}
    assert all(n.path_mode == "direct" for n in pool.online_peers)
    view = read_plan(recorded["plan"], pool)
    assert view.tier == "moe_offload" and view.capacity_mode and view.feasible
    assert view.peer_count == 4 and len(view.machines) == 5
    layers = {s.node_name: s.layer_count for s in view.stages if s.layer_count}
    assert layers == {"mac-azim": 3, "g14": 2, "mac-a": 3, "mac-b": 3, "gx10-49d1": 50}
    assert sum(layers.values()) == 61
    head = [s for s in view.stages if s.holds_output]
    assert len(head) == 1 and head[0].is_local and head[0].device == "CUDA0"
    assert view.predicted_tok_s == pytest.approx(1000 / 1490.95, rel=1e-3)  # the gateway's estimate, not measured


def test_observed_bytes_attach_by_endpoint(recorded: dict) -> None:
    pool = snapshot(recorded["nodes"], recorded["network"], [], [])
    view = with_observed(read_plan(recorded["plan"], pool), recorded["deployment"])
    by_name = {s.node_name + "/" + s.device: s.observed_bytes for s in view.stages}
    assert by_name["mac-b/RPC0"] == 10191361802  # RPC0[127.0.0.1:33707] in the engine log
    assert by_name["g14/RPC3"] == 7087849472
    assert by_name["gx10-49d1/CUDA0"] == 81599576145
    assert by_name["gx10-49d1/CPU"] == 90491584512  # CPU_Mapped: the experts kept on the host CPU


def test_rank_biggest_first_plumbing_last() -> None:
    models = [_model("qwen-0.5b", 0.5, role="plumbing"), _model("gpt-oss-120b", 120), _model("kimi-1t", 1000),
              _model("deepseek-671b", 671), _model("big-missing", 2000, usable=False)]
    ranked = [m.id for m in rank_candidates(models, Intent(action="run"))]
    assert ranked == ["kimi-1t", "deepseek-671b", "gpt-oss-120b", "qwen-0.5b"]
    assert [m.id for m in rank_candidates(models, Intent(action="run", model_hint="deepseek"))] == ["deepseek-671b"]
    assert rank_candidates(models, Intent(action="run", size_hint_b=100))[0].id == "gpt-oss-120b"
    with pytest.raises(NoMatchingModel) as exc:
        rank_candidates(models, Intent(action="run", model_hint="llama"))
    assert "kimi-1t" in exc.value.available


def test_choose_biggest_and_fastest() -> None:
    kimi, gpt = _model("kimi", 1000), _model("gpt", 120)
    slow = {"feasible": True, "fit_label": "estimated_fit", "tier": "moe_offload", "capacity_mode": True}
    quick = {"feasible": True, "fit_label": "estimated_fit", "tier": "gpu", "capacity_mode": False}
    no = {"feasible": False, "fit_label": "does_not_fit"}
    assert choose("biggest", [(kimi, slow), (gpt, quick)])[0] is kimi
    assert choose("fastest", [(kimi, slow), (gpt, quick)])[0] is gpt
    assert choose("fastest", [(kimi, slow)])[0] is kimi  # nothing quick: still answer on what fits
    assert choose("biggest", [(kimi, no), (gpt, None)]) is None


def test_offline_and_stale_peers() -> None:
    st = MockState()
    st.offline = {"mac-b"}
    pool = snapshot(st.node_views(), st.network(), st.model_views(), [])
    assert {n.name for n in pool.offline_peers} == {"mac-b"}
    view = read_plan(st.plan("kimi-k2.6-iq1_s", "all_selected", None, 4096), pool)
    assert "mac-b" not in view.machines
    st.offline = {"mac-a"}
    later = snapshot(st.node_views(), st.network(), st.model_views(), [])
    assert missing_peers(view, later) == ["mac-a"]


def test_models_need_a_verified_download() -> None:
    views = [{"artifact": {"id": "a", "params_b": 7}, "status": "downloaded", "verified_sha256": False},
             {"artifact": {"id": "b", "params_b": 7}, "status": "downloaded", "verified_sha256": True},
             {"artifact": {"id": "c", "params_b": 7}, "status": "engine_managed"},
             {"artifact": {"id": "d", "params_b": 7}, "status": "downloading"}]
    assert [m.id for m in summarize_models(views) if m.usable] == ["b", "c"]
