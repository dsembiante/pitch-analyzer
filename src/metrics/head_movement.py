"""
Metric #10: Head movement / posture stability over the delivery window.

Window: leg_lift_peak to ball_release (the full delivery arc).

Head position is measured RELATIVE to the hip midpoint at each frame.
This removes whole-body stride translation from the signal so that the metrics
capture true head stability (head-over-hips posture) rather than the full
stride arc.

Two distinct measurements are exported as separate MetricResults:

  path_length — total path length of the smoothed (nose - hip_midpoint) vector
    trajectory through the window. Normalized to body height (%). Captures how
    much the head moved relative to the pelvis. High path length = significant
    head sway relative to the body.

  max_deviation — maximum euclidean distance from any single smoothed
    (nose - hip_midpoint) position to its mean over the window. Normalized to
    body height (%). Captures peak excursion from center, body-translation removed.

These measure different things:
  - A pitcher whose head oscillates back-and-forth has HIGH path length but
    LOW max deviation (the oscillations cancel out at the mean).
  - A pitcher who tilts hard in one direction has LOWER path length but
    HIGHER max deviation (one large excursion from center).

Both metrics are lower-is-better (less head sway relative to hips = better posture
stability). Specific numerical thresholds are not currently claimed; the metric
has not been calibrated against expert-labeled data.

All positions computed in pixel space; see LIMITATIONS.md for pixel convention.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ._landmarks import NOSE, LEFT_HIP, RIGHT_HIP
from ._pose_access import build_window_data
from ._geometry import euclidean_distance
from ._types import MetricResult

_VIS_THRESHOLD = 0.5
_MAX_EXCLUDED_FRACTION = 0.30


def _smooth(arr: np.ndarray) -> tuple[np.ndarray, str]:
    n = len(arr)
    if n >= 7:
        return savgol_filter(arr, 7, 3), ""
    elif n >= 5:
        return savgol_filter(arr, 5, 2), " (short window: SG w=5,p=2)"
    return arr.copy(), f" (smoothing skipped: only {n} frames)"


def _collect_nose_series(
    pose_df: pd.DataFrame,
    ll_frame: int,
    br_frame: int,
    w: int,
    h: int,
) -> tuple[list[float], list[float], list[int], int]:
    """Return (rel_x_list, rel_y_list, valid_frames, n_excluded) for nose relative to hip midpoint."""
    frames = list(range(ll_frame, br_frame + 1))
    window_data = build_window_data(pose_df, ll_frame, br_frame)
    _nan4 = (float("nan"), float("nan"), float("nan"), 0.0)

    xs, ys, valid_frames = [], [], []
    excluded = 0
    for f in frames:
        nose_e  = window_data.get((f, NOSE),      _nan4)
        lhip_e  = window_data.get((f, LEFT_HIP),  _nan4)
        rhip_e  = window_data.get((f, RIGHT_HIP), _nan4)
        if (nose_e[3] < _VIS_THRESHOLD
                or lhip_e[3] < _VIS_THRESHOLD
                or rhip_e[3] < _VIS_THRESHOLD):
            excluded += 1
            continue
        nose_x   = nose_e[0] * w
        nose_y   = nose_e[1] * h
        hip_mid_x = (lhip_e[0] + rhip_e[0]) / 2.0 * w
        hip_mid_y = (lhip_e[1] + rhip_e[1]) / 2.0 * h
        xs.append(nose_x - hip_mid_x)
        ys.append(nose_y - hip_mid_y)
        valid_frames.append(f)

    return xs, ys, valid_frames, excluded


def _check_exclusion(
    n_total: int,
    n_excluded: int,
    ll_frame: int,
    name: str,
    display_name: str,
) -> "MetricResult | None":
    excl_frac = n_excluded / max(n_total, 1)
    if excl_frac > _MAX_EXCLUDED_FRACTION:
        return MetricResult(
            name=name, display_name=display_name,
            value=None, unit="percent_body_height",
            frame=ll_frame, phase="leg_lift_to_release",
            error=f"{n_excluded}/{n_total} frames excluded ({excl_frac:.0%} > {_MAX_EXCLUDED_FRACTION:.0%} limit)",
        )
    return None


def compute_path_length(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Total smoothed nose trajectory path length, normalized to body height (%)."""
    ll_frame = int(phases["leg_lift_peak"])
    br_frame = int(phases["ball_release"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])
    body_height_px = float(video_metadata["body_height_pixels"])
    n_total = br_frame - ll_frame + 1

    name, display_name = "head_path_length", "Head Path Length"

    xs, ys, valid_frames, excluded = _collect_nose_series(pose_df, ll_frame, br_frame, w, h)

    err_result = _check_exclusion(n_total, excluded, ll_frame, name, display_name)
    if err_result:
        return err_result

    if len(xs) < 2:
        return MetricResult(
            name=name, display_name=display_name,
            value=None, unit="percent_body_height",
            frame=ll_frame, phase="leg_lift_to_release",
            error="Fewer than 2 valid nose frames in window",
        )

    arr_x, _ = _smooth(np.array(xs))
    arr_y, smooth_note = _smooth(np.array(ys))

    path_px = sum(
        euclidean_distance((arr_x[i], arr_y[i]), (arr_x[i + 1], arr_y[i + 1]))
        for i in range(len(arr_x) - 1)
    )
    pct = (path_px / body_height_px) * 100.0

    return MetricResult(
        name=name,
        display_name=display_name,
        value=round(pct, 1),
        unit="percent_body_height",
        description="Total head displacement relative to hip midpoint from leg lift to release, normalized to body height. Lower values indicate more stable head position relative to the torso.",
        frame=ll_frame,
        phase="leg_lift_to_release",
        notes=(
            f"window_frames={n_total}, excluded={excluded}, valid={len(xs)}"
            f"{smooth_note}. "
            "Measured as nose minus hip-midpoint vector; stride translation removed. "
            "Lower = more stable head relative to torso. "
            "Compare to max_deviation: oscillating head has high path but lower max_deviation."
        ),
    )


def compute_max_deviation(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Max nose displacement from its mean position (smoothed), normalized to body height (%)."""
    ll_frame = int(phases["leg_lift_peak"])
    br_frame = int(phases["ball_release"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])
    body_height_px = float(video_metadata["body_height_pixels"])
    n_total = br_frame - ll_frame + 1

    name, display_name = "head_max_deviation", "Head Max Deviation"

    xs, ys, valid_frames, excluded = _collect_nose_series(pose_df, ll_frame, br_frame, w, h)

    err_result = _check_exclusion(n_total, excluded, ll_frame, name, display_name)
    if err_result:
        return err_result

    if len(xs) < 2:
        return MetricResult(
            name=name, display_name=display_name,
            value=None, unit="percent_body_height",
            frame=ll_frame, phase="leg_lift_to_release",
            error="Fewer than 2 valid nose frames in window",
        )

    arr_x, _ = _smooth(np.array(xs))
    arr_y, smooth_note = _smooth(np.array(ys))

    mean_x = float(np.mean(arr_x))
    mean_y = float(np.mean(arr_y))
    max_dev_px = float(np.max([
        euclidean_distance((arr_x[i], arr_y[i]), (mean_x, mean_y))
        for i in range(len(arr_x))
    ]))
    pct = (max_dev_px / body_height_px) * 100.0

    return MetricResult(
        name=name,
        display_name=display_name,
        value=round(pct, 1),
        unit="percent_body_height",
        description="Peak head displacement (relative to hip midpoint) from its average position during the delivery, normalized to body height. Lower values indicate more stable head position relative to the torso.",
        frame=ll_frame,
        phase="leg_lift_to_release",
        notes=(
            f"window_frames={n_total}, excluded={excluded}, valid={len(xs)}"
            f"{smooth_note}. "
            "Max excursion from mean nose-minus-hip-midpoint position. "
            "Stride translation removed; reflects posture stability relative to torso. "
            "Compare to path_length: one-direction tilt has lower path but higher max_deviation."
        ),
    )
