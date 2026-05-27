"""
Metric #3: Maximum hip-shoulder separation in the coiling window.

Definition: the maximum absolute angular difference between the hip line
(left hip -> right hip) and the shoulder line (left shoulder -> right shoulder)
over the window from leg_lift_peak to foot_strike.

Hip-shoulder separation is a key velocity predictor: a larger separation
(greater "torso coil") produces more elastic energy to be released into the arm.

IMPORTANT 2D CAVEAT: This is a 2D projection from a single side-facing camera.
True hip-shoulder separation is a rotation about the spine axis; the 2D
measurement captures only the apparent angle difference as projected onto the
camera plane. When the pitcher is exactly perpendicular to the camera this
approximates the 3D value; at other angles it underestimates. Values from this
metric should be interpreted as a relative indicator, not an absolute measurement.
See LIMITATIONS.md.

Sanity range: 30-60 deg typical; elite pitchers reach 45-60+. Under 20 deg
suggests insufficient torso coil.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._landmarks import LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER
from ._pose_access import build_window_data
from ._geometry import angle_of_line_2d, signed_angle_difference
from ._types import MetricResult

_METRIC_NAME = "hip_shoulder_separation_max"
_DISPLAY_NAME = "Hip-Shoulder Separation (Max)"
_PHASE = "between_leg_lift_and_foot_strike"
_LM_INDICES = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
_VIS_THRESHOLD = 0.5
_MAX_EXCLUDED_FRACTION = 0.30


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
    """Compute maximum hip-shoulder separation (degrees) over the coiling window."""
    ll_frame = int(phases["leg_lift_peak"])
    fs_frame = int(phases["foot_strike"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    frames = list(range(ll_frame, fs_frame + 1))
    if not frames:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees", frame=ll_frame, phase=_PHASE,
            error=f"Empty window: leg_lift_peak ({ll_frame}) >= foot_strike ({fs_frame})",
        )

    window_data = build_window_data(pose_df, ll_frame, fs_frame)

    # Per-frame angle computation with visibility filtering
    separations: list[float] = []
    frame_of_sep: list[int]  = []
    excluded = 0

    for f in frames:
        # Check visibility of all four landmarks
        lm_vals = {}
        bad_vis = False
        for lm in _LM_INDICES:
            entry = window_data.get((f, lm), (float("nan"), float("nan"), float("nan"), 0.0))
            if entry[3] < _VIS_THRESHOLD:
                bad_vis = True
                break
            lm_vals[lm] = entry
        if bad_vis:
            excluded += 1
            continue

        # Pixel-space coordinates
        lh_px = (lm_vals[LEFT_HIP][0]   * w, lm_vals[LEFT_HIP][1]   * h)
        rh_px = (lm_vals[RIGHT_HIP][0]  * w, lm_vals[RIGHT_HIP][1]  * h)
        ls_px = (lm_vals[LEFT_SHOULDER][0]  * w, lm_vals[LEFT_SHOULDER][1]  * h)
        rs_px = (lm_vals[RIGHT_SHOULDER][0] * w, lm_vals[RIGHT_SHOULDER][1] * h)

        hip_angle  = angle_of_line_2d(lh_px, rh_px)
        sho_angle  = angle_of_line_2d(ls_px, rs_px)
        sep = abs(signed_angle_difference(hip_angle, sho_angle))

        separations.append(sep)
        frame_of_sep.append(f)

    if not separations:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees", frame=ll_frame, phase=_PHASE,
            error="All frames in window excluded due to low landmark visibility",
        )

    excl_frac = excluded / len(frames)
    if excl_frac > _MAX_EXCLUDED_FRACTION:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees", frame=ll_frame, phase=_PHASE,
            error=f"{excluded}/{len(frames)} frames excluded ({excl_frac:.0%} > {_MAX_EXCLUDED_FRACTION:.0%} limit)",
        )

    arr = np.array(separations, dtype=float)
    smoothed, smooth_note = _smooth(arr)

    max_idx = int(np.argmax(smoothed))
    max_val = float(smoothed[max_idx])
    max_frame = frame_of_sep[max_idx]

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(max_val, 1),
        unit="degrees",
        frame=max_frame,
        phase=_PHASE,
        notes=(
            f"window_frames={len(frames)}, excluded={excluded}, max_at_frame={max_frame}"
            f"{smooth_note}. "
            "2D projection from side-facing camera underestimates true 3D separation; "
            "see LIMITATIONS.md."
        ),
    )
