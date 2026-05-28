"""
Metric #9: Front (lead) knee extension rate from foot_strike to ball_release.

Definition: rate of change of the lead knee angle (degrees/second) over the
foot_strike to ball_release window. Positive = extending (angle increasing
toward 180°); negative = flexing (angle decreasing).

The lead leg should act as a "blocking post" at release — extending to transfer
momentum from the lower body into the trunk. Flexing into release (negative rate)
is a mechanics flag indicating the front leg collapsed.

Algorithm:
  1. Compute lead knee angle (lead hip -> lead knee -> lead ankle, interior
     angle) in pixel space at each frame in the window.
  2. Smooth the angle series with Savitzky-Golay.
     - Preferred: window=7, polyorder=3 (requires >= 7 frames)
     - Fallback:  window=5, polyorder=2 (requires >= 5 frames)
     - Skip: < 5 frames (note the lack of smoothing)
  3. Angular velocity = (smoothed[-1] - smoothed[0]) / window_duration_s.
  4. Direction: "extending" (rate >= 50 deg/s), "flexing" (rate <= -50 deg/s),
     "stable" (abs(rate) < 50 deg/s).

Sanity ranges: 100-400+ deg/s for elite pitchers. <50 is "stable" (noise floor).
Flexing (negative) into release is a flag worth reviewing.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._landmarks import LEAD_HIP, LEAD_KNEE, LEAD_ANKLE
from ._pose_access import build_window_data
from ._geometry import angle_between_points
from ._types import MetricResult

_METRIC_NAME = "front_knee_extension_rate"
_DISPLAY_NAME = "Front Knee Extension Rate"
_PHASE = "foot_strike_to_release"
_VIS_THRESHOLD = 0.5
_MAX_EXCLUDED_FRACTION = 0.30
_STABLE_THRESHOLD = 50.0  # deg/s below which we call the rate "stable"


def _smooth(arr: np.ndarray) -> tuple[np.ndarray, str]:
    n = len(arr)
    if n >= 7:
        return savgol_filter(arr, 7, 3), ""
    elif n >= 5:
        return savgol_filter(arr, 5, 2), " (short window: SG w=5,p=2)"
    return arr.copy(), f" (smoothing skipped: only {n} frames)"


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Compute lead knee extension rate (deg/s) from foot strike to ball release."""
    fs_frame = int(phases["foot_strike"])
    br_frame = int(phases["ball_release"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    if fs_frame >= br_frame:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees_per_second",
            frame=fs_frame, phase=_PHASE,
            error=f"foot_strike ({fs_frame}) >= ball_release ({br_frame}); window has no duration",
        )

    # Window duration from phase timestamps (robust to variable frame rate)
    duration_s = (
        phases["ball_release_timestamp_ms"] - phases["foot_strike_timestamp_ms"]
    ) / 1000.0
    if duration_s <= 0:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees_per_second",
            frame=fs_frame, phase=_PHASE,
            error=f"Zero or negative window duration ({duration_s:.3f}s)",
        )

    hip_idx   = LEAD_HIP[handedness]
    knee_idx  = LEAD_KNEE[handedness]
    ankle_idx = LEAD_ANKLE[handedness]
    frames = list(range(fs_frame, br_frame + 1))
    n_total = len(frames)

    window_data = build_window_data(pose_df, fs_frame, br_frame)

    angles: list[float] = []
    frame_list: list[int] = []
    excluded = 0

    for f in frames:
        entries = {
            lm: window_data.get((f, lm), (float("nan"), float("nan"), float("nan"), 0.0))
            for lm in (hip_idx, knee_idx, ankle_idx)
        }
        if any(e[3] < _VIS_THRESHOLD for e in entries.values()):
            excluded += 1
            continue

        hip_px   = (entries[hip_idx][0]   * w, entries[hip_idx][1]   * h)
        knee_px  = (entries[knee_idx][0]  * w, entries[knee_idx][1]  * h)
        ankle_px = (entries[ankle_idx][0] * w, entries[ankle_idx][1] * h)

        angles.append(angle_between_points(hip_px, knee_px, ankle_px))
        frame_list.append(f)

    excl_frac = excluded / n_total
    if excl_frac > _MAX_EXCLUDED_FRACTION:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees_per_second",
            frame=fs_frame, phase=_PHASE,
            error=f"{excluded}/{n_total} frames excluded ({excl_frac:.0%} > {_MAX_EXCLUDED_FRACTION:.0%} limit)",
        )

    if len(angles) < 2:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees_per_second",
            frame=fs_frame, phase=_PHASE,
            error="Fewer than 2 valid frames in window",
        )

    smoothed, smooth_note = _smooth(np.array(angles))
    start_angle = float(smoothed[0])
    end_angle   = float(smoothed[-1])
    rate = (end_angle - start_angle) / duration_s

    if abs(rate) < _STABLE_THRESHOLD:
        direction = "stable"
    elif rate > 0:
        direction = "extending"
    else:
        direction = "flexing"

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(rate, 1),
        unit="degrees_per_second",
        description="Lead knee rotation rate from foot strike to release. Positive=extending (blocking post), negative=collapsing.",
        frame=br_frame,
        phase=_PHASE,
        notes=(
            f"direction={direction}, "
            f"start_angle={start_angle:.1f}deg, end_angle={end_angle:.1f}deg, "
            f"duration={duration_s:.3f}s, "
            f"window_frames={n_total}, excluded={excluded}"
            f"{smooth_note}. "
            "Positive=extending (good blocking), negative=flexing (mechanics flag). "
            "Elite typical: 100-400+ deg/s."
        ),
    )
