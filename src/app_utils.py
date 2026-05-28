"""
Shared display helpers for the Streamlit app (src/app.py).
"""

import datetime
import re

from metrics._types import MetricResult
from app_pipeline import PipelineResult

# ── Phase display names ────────────────────────────────────────────────────
PHASE_DISPLAY_NAMES: dict[str, str] = {
    # Core delivery phases
    "start_of_motion":                   "Start of motion",
    "leg_lift_peak":                      "Leg lift peak",
    "foot_strike":                        "Foot strike",
    "max_layback":                        "Max layback",
    "ball_release":                       "Ball release",
    "end_of_motion":                      "End of motion",
    # Window phase strings used by metrics
    "between_leg_lift_and_foot_strike":   "Leg lift to foot strike",
    "leg_lift_to_release":                "Leg lift to release",
    "foot_strike_to_release":             "Foot strike to release",
}

# ── Metric display order ───────────────────────────────────────────────────
METRIC_DISPLAY_ORDER: list[str] = [
    # Release-frame snapshot
    "arm_slot",
    "trunk_tilt_lateral",
    "trunk_tilt_forward",
    "front_knee_flex",
    # Kinematic chain
    "stride_length",
    "hip_shoulder_separation_max",
    "balance_point",
    # Lead leg block
    "front_knee_extension_rate",
    # Tempo
    "tempo_leg_lift_to_foot_strike",
    "tempo_foot_strike_to_release",
    "tempo_total_motion",
    # Posture / head stability
    "head_path_length",
    "head_max_deviation",
]

# ── Improvement direction per metric ──────────────────────────────────────
IMPROVEMENT_DIRECTION: dict[str, str] = {
    "arm_slot":                        "neutral",
    "trunk_tilt_lateral":              "neutral",
    "trunk_tilt_forward":              "neutral",
    "front_knee_flex":                 "neutral",
    "stride_length":                   "higher_better",
    "tempo_leg_lift_to_foot_strike":   "neutral",
    "tempo_foot_strike_to_release":    "neutral",
    "tempo_total_motion":              "neutral",
    "balance_point":                   "neutral",
    "hip_shoulder_separation_max":     "higher_better",
    "head_path_length":                "lower_better",
    "head_max_deviation":              "lower_better",
    "front_knee_extension_rate":       "higher_better",
}


# ── User-facing hint appended to pipeline ValueError messages ─────────────
PIPELINE_ERROR_HINT = (
    "This usually means the clip is slow motion (which compresses real time so phase "
    "windows fall outside expected ranges), doesn't contain a complete pitching delivery, "
    "or has unreliable pose tracking. Try a real-time, side-angle clip with a complete "
    "delivery and the pitcher's full body visible throughout."
)


def get_direction(metric_name: str) -> str:
    """Return the improvement direction for a metric, defaulting to 'neutral'."""
    return IMPROVEMENT_DIRECTION.get(metric_name, "neutral")


_UNIT_LABELS: dict[str, str] = {
    "degrees":             "deg",
    "seconds":             "s",
    "percent_body_height": "%",
    "degrees_per_second":  "deg/s",
}

# Phase keys included in the export (delivery phases only; window/timestamp
# keys in the phases dict are excluded).
_EXPORT_PHASE_KEYS = [
    "start_of_motion",
    "leg_lift_peak",
    "foot_strike",
    "max_layback",
    "ball_release",
    "end_of_motion",
]


def format_metric_value(metric: MetricResult) -> str:
    """Format a metric value for table display.

    Returns "--" when value is None or an error is set.
    Tempo (seconds): 2 decimal places.
    Rate (deg/s): 1 decimal place with explicit sign.
    Everything else: 1 decimal place.
    """
    if metric.error is not None:
        return "⚠ —"
    if metric.value is None:
        return "--"

    v = metric.value
    unit_label = _UNIT_LABELS.get(metric.unit, metric.unit)

    if metric.unit == "seconds":
        return f"{v:.2f} {unit_label}"
    elif metric.unit == "degrees_per_second":
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f} {unit_label}"
    else:
        return f"{v:.1f} {unit_label}"


def slugify_for_filename(text: str) -> str:
    """Lowercase, replace whitespace with underscores, strip non-alphanumeric (except _)."""
    text = text.lower().strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def build_export_payload(
    pipeline_result: PipelineResult,
    athlete_name: str,
    pitch_type: str,
    session_date: datetime.date,
    handedness: str,
    video_filename: str | None,
) -> dict:
    """Build the JSON-serializable export payload for a session."""
    vm  = pipeline_result.video_metadata
    fps = float(vm["fps"])

    phases_out: dict = {}
    for key in _EXPORT_PHASE_KEYS:
        frame = pipeline_result.phases.get(key)
        if frame is not None:
            phases_out[key] = {"frame": int(frame), "time_s": round(frame / fps, 3)}

    return {
        "metadata": {
            "athlete_name":  athlete_name or None,
            "pitch_type":    pitch_type,
            "session_date":  session_date.isoformat(),
            "handedness":    handedness,
            "video_filename": video_filename,
            "processed_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "video": {
            "width":       int(vm["width"]),
            "height":      int(vm["height"]),
            "fps":         fps,
            "duration_s":  float(vm["duration_s"]),
            "frame_count": int(vm["frame_count"]),
        },
        "phases":  phases_out,
        "metrics": {
            name: metric.to_dict()
            for name, metric in pipeline_result.metrics.items()
        },
    }
