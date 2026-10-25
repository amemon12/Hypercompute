"""Values shared by the watcher, the Droid tools, the run log and the page.

Stage mirrors the gateway's schemas.LayoutStage field for field (backend/hypercompute/schemas.py), so a layout Droid
proposes is validated here before it ever reaches POST /api/plans.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEVICE_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,15}$"  # schemas._DEVICE_NAME
NODE_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Stage(Frozen):
    """One stage of an explicit layout: the next `layers` whole layers on one device. Peers first, this machine's GPU
    ("local") last, which also holds the output head."""

    node_id: str = Field(pattern=NODE_PATTERN, description='a peer node id from pool_snapshot, or "local"')
    device: str = Field(pattern=DEVICE_PATTERN,
                        description="the device as that node's worker names it: MTL0, CPU, CUDA0")
    layers: int = Field(ge=1, le=4096)
    residency: Literal["resident", "file_backed"] | None = None


class Candidate(Frozen):
    label: str = Field(max_length=80, description="short name, e.g. 'GX10 alone' or 'mac-a 4 + CUDA0'")
    stages: list[Stage] = Field(min_length=1, max_length=64)
    rationale: str = Field(max_length=600, description="why this layout, citing planner verdicts and measured runs")
    plan_id: str | None = Field(default=None, max_length=64, description="plan_id check_layout returned, if any")


class AutotuneProposal(Frozen):
    """Droid's structured answer at the end of a run."""

    summary: str = Field(max_length=1200)
    recommendation: Literal["apply", "keep"]
    recommended_index: int | None = Field(default=None, ge=0, le=7,
                                          description="index into candidates when recommendation is apply")
    candidates: list[Candidate] = Field(default_factory=list, max_length=8)


class DeviceRow(Frozen):
    """What one stage puts on one device, from a gateway plan (estimates, never measurements)."""

    node_id: str
    node_name: str
    device: str
    layers: int
    first_layer: int | None
    last_layer: int | None
    total_bytes_est: int
    budget_bytes: int
    fits: bool
    file_backed: bool = False
    holds_output: bool = False


class Prediction(Frozen):
    """The planner's own numbers for a plan, with where each came from. None when the planner gave none."""

    decode_ms: float | None = None
    decode_source: str | None = None  # "prediction" (capacity optimizer) or "moe" (moe_offload estimate)
    ssd_bytes_per_token: int | None = None
    load_ms: float | None = None


class Verdict(Frozen):
    """The gateway planner's answer to one explicit layout (POST /api/plans, plan-only)."""

    accepted: bool
    label: str = ""
    stages: list[Stage] = Field(default_factory=list)
    code: str | None = None  # refusal code from services/layout.py, e.g. layout_memory
    reason: str = ""
    fix: str | None = None
    plan_id: str | None = None
    tier: str | None = None
    fit_label: str | None = None
    hops: int | None = None  # cross-machine boundaries a token crosses
    rows: list[DeviceRow] = Field(default_factory=list)
    prediction: Prediction = Field(default_factory=Prediction)
    warnings: list[str] = Field(default_factory=list)
    checked_by: Literal["droid", "service"] = "droid"


class DiffRow(Frozen):
    node_id: str
    node_name: str
    device: str
    layers_before: int
    layers_after: int
    range_before: str
    range_after: str
    bytes_before: int
    bytes_after: int
    budget_bytes: int
    fits_after: bool
    file_backed_after: bool = False


class LayoutDiff(Frozen):
    rows: list[DiffRow]
    hops_before: int | None
    hops_after: int | None
    machines_before: int
    machines_after: int
    tier_before: str | None
    tier_after: str | None
    decode_ms_before: float | None
    decode_ms_after: float | None
    decode_source: str | None
    ssd_bytes_before: int | None
    ssd_bytes_after: int | None
    unchanged: bool
