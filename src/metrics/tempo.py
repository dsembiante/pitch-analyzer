"""
Metric #6: Delivery timing — three time-interval components.

All durations are derived from the timestamp_ms values already embedded in
the phases dict (which were computed from the pose DataFrame timestamps at
detection time). Using these is equivalent to a DataFrame lookup and is robust
to variable frame rates. Time intervals are converted from ms to seconds.

Typical sanity ranges:
  leg_lift -> foot_strike:    0.4-0.8 s  (stride phase)
  foot_strike -> ball_release: 0.05-0.4 s (power/arm-acceleration phase;
                               faster for high-velocity pitchers, longer for
                               deliberate mechanics)
  total motion:               1.0-2.0 s
"""

import pandas as pd

from ._types import MetricResult

_PHASE_FS_TO_RELEASE = "ball_release"


def _duration_s(phases: dict, start_key: str, end_key: str) -> float:
    """Return the time difference in seconds between two phase timestamps."""
    return (phases[f"{end_key}_timestamp_ms"] - phases[f"{start_key}_timestamp_ms"]) / 1000.0


def compute_leg_lift_to_foot_strike(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Time from leg lift peak to foot strike (stride phase duration)."""
    duration = _duration_s(phases, "leg_lift_peak", "foot_strike")
    frame = int(phases["foot_strike"])
    return MetricResult(
        name="tempo_leg_lift_to_foot_strike",
        display_name="Tempo: Leg Lift to Foot Strike",
        value=round(duration, 3),
        unit="seconds",
        description="Duration of the stride phase from peak leg lift to front-foot landing. Typical: 0.4-0.8s.",
        frame=frame,
        phase="foot_strike",
        notes="Stride phase. Typical range: 0.4-0.8s.",
    )


def compute_foot_strike_to_release(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Time from foot strike to ball release (power/arm-acceleration phase)."""
    duration = _duration_s(phases, "foot_strike", "ball_release")
    frame = int(phases["ball_release"])
    return MetricResult(
        name="tempo_foot_strike_to_release",
        display_name="Tempo: Foot Strike to Release",
        value=round(duration, 3),
        unit="seconds",
        description="Power phase: time from front-foot landing to ball release. Faster pitchers compress this window.",
        frame=frame,
        phase="ball_release",
        notes="Power phase. Faster for higher-velocity pitchers; typically 0.05-0.4s.",
    )


def compute_total_motion_time(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Total delivery time from start of motion to ball release."""
    duration = _duration_s(phases, "start_of_motion", "ball_release")
    frame = int(phases["ball_release"])
    return MetricResult(
        name="tempo_total_motion",
        display_name="Tempo: Total Motion Time",
        value=round(duration, 3),
        unit="seconds",
        description="Total delivery time from first movement to ball release. Typical range: 1.0-2.0s.",
        frame=frame,
        phase="ball_release",
        notes="Start of motion to release. Typical range: 1.0-2.0s.",
    )
