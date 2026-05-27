"""
Orchestrator: run all implemented metrics against a single pitching delivery.

Each metric's compute() function is called inside a try/except so a failure
in one metric does not abort the pipeline. On exception a MetricResult with
error set is returned for that metric.

The pose DataFrame is passed as-is to each metric. For single-frame lookups
(all current metrics) the pandas boolean filter is fast enough on ~370-frame
datasets; a wide-format pivot is not necessary at this scale.
"""

import pandas as pd

from ._geometry import compute_body_height_pixels
from ._types import MetricResult
from . import arm_slot, trunk_tilt, front_knee_flex


def compute_all_metrics(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> dict[str, MetricResult]:
    """Run all metrics and return a dict keyed by metric name.

    Args:
        pose_df: Full pose DataFrame from Phase 1.
        phases: Dict returned by detect_all_phases (or loaded from JSON).
        handedness: "right" or "left".
        video_metadata: Dict with keys: fps, width, height.
                        body_height_pixels is computed here and added.

    Returns:
        Dict mapping metric name -> MetricResult.
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

    # (metric_name, callable) pairs — trunk_tilt exposes two separate functions
    metrics = [
        ("arm_slot",            arm_slot.compute),
        ("trunk_tilt_lateral",  trunk_tilt.compute_lateral),
        ("trunk_tilt_forward",  trunk_tilt.compute_forward),
        ("front_knee_flex",     front_knee_flex.compute),
    ]

    results: dict[str, MetricResult] = {}
    for name, fn in metrics:
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

    return results
