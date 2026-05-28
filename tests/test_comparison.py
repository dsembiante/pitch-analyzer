"""
Tests for the notable classification logic in app_comparison.compute_comparison.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics._types import MetricResult
from app_pipeline import PipelineResult
from app_comparison import (
    compute_comparison,
    NEUTRAL_NOTABLE_PCT_THRESHOLD,
    NEUTRAL_NOTABLE_ABS_THRESHOLD_BY_UNIT,
    UNCHANGED_PCT_THRESHOLD,
)


def _make_mr(name: str, value: float, unit: str = "degrees") -> MetricResult:
    return MetricResult(
        name=name,
        display_name=name,
        value=value,
        unit=unit,
    )


def _make_result(metrics: dict) -> PipelineResult:
    return PipelineResult(
        pose_df=pd.DataFrame(),
        phases={},
        metrics=metrics,
        time_series={},
        video_metadata={"fps": 30.0, "width": 1920, "height": 1080, "frame_count": 90, "duration_s": 3.0},
        annotated_video_bytes=b"",
    )


# ── neutral metric tests ────────────────────────────────────────────────────

def test_neutral_large_pct_change_is_notable():
    # arm_slot is neutral; a >10% change should be classified notable.
    mr_a = _make_mr("arm_slot", 30.0, "degrees")
    mr_b = _make_mr("arm_slot", 34.0, "degrees")   # +13.3%
    result = compute_comparison(
        _make_result({"arm_slot": mr_a}),
        _make_result({"arm_slot": mr_b}),
    )
    cmp = result.comparisons[0]
    assert cmp.classification == "notable"
    assert result.summary.get("notable", 0) == 1


def test_neutral_small_pct_change_is_unchanged():
    # arm_slot is neutral; a <2% change stays unchanged.
    mr_a = _make_mr("arm_slot", 30.0, "degrees")
    mr_b = _make_mr("arm_slot", 30.5, "degrees")   # +1.67%
    result = compute_comparison(
        _make_result({"arm_slot": mr_a}),
        _make_result({"arm_slot": mr_b}),
    )
    assert result.comparisons[0].classification == "unchanged"


def test_neutral_between_thresholds_is_unchanged():
    # A change between UNCHANGED (2%) and NEUTRAL_NOTABLE (10%) is still unchanged.
    mr_a = _make_mr("arm_slot", 30.0, "degrees")
    mr_b = _make_mr("arm_slot", 32.0, "degrees")   # +6.7% — above UNCHANGED, below NOTABLE
    result = compute_comparison(
        _make_result({"arm_slot": mr_a}),
        _make_result({"arm_slot": mr_b}),
    )
    assert result.comparisons[0].classification == "unchanged"


def test_neutral_notable_abs_fallback_when_baseline_near_zero():
    # When baseline is near zero, pct_change is None; use abs threshold fallback.
    # tempo_leg_lift_to_foot_strike is neutral with unit "seconds".
    # NEUTRAL_NOTABLE_ABS_THRESHOLD_BY_UNIT["seconds"] == 0.10
    name = "tempo_leg_lift_to_foot_strike"
    mr_a = _make_mr(name, 0.0, "seconds")          # baseline ~0 → pct_change is None
    mr_b = _make_mr(name, 0.15, "seconds")          # abs delta = 0.15 > 0.10
    result = compute_comparison(
        _make_result({name: mr_a}),
        _make_result({name: mr_b}),
    )
    assert result.comparisons[0].pct_change is None
    assert result.comparisons[0].classification == "notable"


def test_neutral_abs_fallback_below_threshold_is_unchanged():
    name = "tempo_leg_lift_to_foot_strike"
    mr_a = _make_mr(name, 0.0, "seconds")
    mr_b = _make_mr(name, 0.05, "seconds")          # abs delta = 0.05 < 0.10
    result = compute_comparison(
        _make_result({name: mr_a}),
        _make_result({name: mr_b}),
    )
    assert result.comparisons[0].classification == "unchanged"


# ── directional metrics are NOT affected ───────────────────────────────────

def test_directional_higher_better_large_change_is_improved_not_notable():
    # stride_length is higher_better; a large positive change should be improved.
    mr_a = _make_mr("stride_length", 80.0, "percent_body_height")
    mr_b = _make_mr("stride_length", 90.0, "percent_body_height")  # +12.5%
    result = compute_comparison(
        _make_result({"stride_length": mr_a}),
        _make_result({"stride_length": mr_b}),
    )
    assert result.comparisons[0].classification == "improved"


def test_directional_lower_better_large_change_is_regressed_not_notable():
    # head_path_length is lower_better; large increase should be regressed.
    mr_a = _make_mr("head_path_length", 20.0, "percent_body_height")
    mr_b = _make_mr("head_path_length", 23.0, "percent_body_height")  # +15%
    result = compute_comparison(
        _make_result({"head_path_length": mr_a}),
        _make_result({"head_path_length": mr_b}),
    )
    assert result.comparisons[0].classification == "regressed"


# ── incomparable cases are still correct ───────────────────────────────────

def test_none_value_is_incomparable():
    mr_a = _make_mr("arm_slot", None, "degrees")
    mr_b = _make_mr("arm_slot", 30.0, "degrees")
    mr_a.value = None
    result = compute_comparison(
        _make_result({"arm_slot": mr_a}),
        _make_result({"arm_slot": mr_b}),
    )
    assert result.comparisons[0].classification == "incomparable"


def test_error_value_is_incomparable():
    mr_a = _make_mr("arm_slot", 30.0, "degrees")
    mr_b = _make_mr("arm_slot", 34.0, "degrees")
    mr_a.error = "some error"
    result = compute_comparison(
        _make_result({"arm_slot": mr_a}),
        _make_result({"arm_slot": mr_b}),
    )
    assert result.comparisons[0].classification == "incomparable"


# ── summary dict reflects notable count ────────────────────────────────────

def test_summary_counts_notable_separately():
    # Two neutral metrics: one notable, one unchanged.
    mr_arm_a = _make_mr("arm_slot", 30.0, "degrees")
    mr_arm_b = _make_mr("arm_slot", 34.0, "degrees")    # +13.3% → notable
    mr_trunk_a = _make_mr("trunk_tilt_lateral", 10.0, "degrees")
    mr_trunk_b = _make_mr("trunk_tilt_lateral", 10.1, "degrees")  # +1% → unchanged

    result = compute_comparison(
        _make_result({"arm_slot": mr_arm_a, "trunk_tilt_lateral": mr_trunk_a}),
        _make_result({"arm_slot": mr_arm_b, "trunk_tilt_lateral": mr_trunk_b}),
    )
    assert result.summary.get("notable", 0) == 1
    assert result.summary.get("unchanged", 0) == 1
    assert result.summary.get("improved", 0) == 0
    assert result.summary.get("regressed", 0) == 0
