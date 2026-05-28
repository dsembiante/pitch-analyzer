"""
Orchestrator: run all implemented metrics against a single pitching delivery.

Each metric's compute() function is called inside a try/except so a failure
in one metric does not abort the pipeline. On exception a MetricResult with
error set is returned for that metric.

Two metrics return (MetricResult, pd.DataFrame) — hip_shoulder_separation and
front_knee_flex. The DataFrames are collected into the time_series return dict
keyed by the metric's name field. All other metrics return a bare MetricResult.
"""

import pandas as pd

from ._geometry import compute_body_height_pixels
from ._types import MetricResult
from . import (
    arm_slot, trunk_tilt, front_knee_flex, stride_length, tempo, balance_point,
    hip_shoulder_separation, head_movement, front_knee_extension,
)


def compute_all_metrics(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> tuple[dict[str, MetricResult], dict[str, pd.DataFrame]]:
    """Run all metrics and return (metrics_dict, time_series_dict).

    Args:
        pose_df: Full pose DataFrame from Phase 1.
        phases: Dict returned by detect_all_phases (or loaded from JSON).
        handedness: "right" or "left".
        video_metadata: Dict with keys: fps, width, height.
                        body_height_pixels is computed here and added.

    Returns:
        metrics:     Dict mapping metric name -> MetricResult.
        time_series: Dict mapping metric name -> per-frame DataFrame.
                     Present only for metrics that produce a series
                     (hip_shoulder_separation_max, front_knee_flex).
                     Absent if that metric errored.
    """
    # Compute body height once at ball_release and inject into metadata.
    reference_frame = int(phases["ball_release"])
    body_height = compute_body_height_pixels(
        pose_df,
        reference_frame,
        int(video_metadata["width"]),
        int(video_metadata["height"]),
    )
    video_metadata = {**video_metadata, "body_height_pixels": body_height}

    # --- Standard metrics: return a bare MetricResult ---
    standard = [
        ("arm_slot",                        arm_slot.compute),
        ("trunk_tilt_lateral",              trunk_tilt.compute_lateral),
        ("trunk_tilt_forward",              trunk_tilt.compute_forward),
        ("stride_length",                   stride_length.compute),
        ("tempo_leg_lift_to_foot_strike",   tempo.compute_leg_lift_to_foot_strike),
        ("tempo_foot_strike_to_release",    tempo.compute_foot_strike_to_release),
        ("tempo_total_motion",              tempo.compute_total_motion_time),
        ("balance_point",                   balance_point.compute),
        ("head_path_length",                head_movement.compute_path_length),
        ("head_max_deviation",              head_movement.compute_max_deviation),
        ("front_knee_extension_rate",       front_knee_extension.compute),
    ]

    # --- Tuple metrics: return (MetricResult, pd.DataFrame) ---
    # Inserted at the positions that match the original ordering:
    #   front_knee_flex  after trunk_tilt_forward (position 4)
    #   hip_shoulder_separation_max  after balance_point (position 10)
    tuple_metrics = [
        ("front_knee_flex",             front_knee_flex.compute),
        ("hip_shoulder_separation_max", hip_shoulder_separation.compute),
    ]

    results: dict[str, MetricResult] = {}
    time_series: dict[str, pd.DataFrame] = {}

    # Run standard metrics
    for name, fn in standard:
        try:
            result = fn(pose_df, phases, handedness, video_metadata)
        except Exception as exc:  # noqa: BLE001
            result = MetricResult(
                name=name,
                display_name=name.replace("_", " ").title(),
                value=None,
                unit="degrees",
                error=str(exc),
            )
        results[name] = result

    # Run tuple-returning metrics
    for name, fn in tuple_metrics:
        try:
            metric_result, series_df = fn(pose_df, phases, handedness, video_metadata)
            if not series_df.empty:
                time_series[name] = series_df
        except Exception as exc:  # noqa: BLE001
            metric_result = MetricResult(
                name=name,
                display_name=name.replace("_", " ").title(),
                value=None,
                unit="degrees",
                error=str(exc),
            )
        results[name] = metric_result

    # Restore the original display order expected by consumers (METRIC_DISPLAY_ORDER).
    # The dict insertion order matters for the CLI summary table.
    ordered = [
        "arm_slot",
        "trunk_tilt_lateral",
        "trunk_tilt_forward",
        "front_knee_flex",
        "stride_length",
        "tempo_leg_lift_to_foot_strike",
        "tempo_foot_strike_to_release",
        "tempo_total_motion",
        "balance_point",
        "hip_shoulder_separation_max",
        "head_path_length",
        "head_max_deviation",
        "front_knee_extension_rate",
    ]
    results = {k: results[k] for k in ordered if k in results}

    return results, time_series
