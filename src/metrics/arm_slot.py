"""
Metric #1: Arm slot angle at ball release.

Definition: angle of the throwing arm (shoulder-to-wrist line) relative to
horizontal, measured at the frame of PEAK ARM ELEVATION within a short window
around ball release.

  +90 deg = straight overhead
    0 deg = sidearm (wrist level with shoulder)
  -90 deg = submarine (wrist below shoulder)

Why not ball_release directly: detect_ball_release returns the frame of peak
|horizontal wrist velocity|. For high-velocity pitchers the arm has often begun
its follow-through descent at that exact frame, placing the wrist below shoulder
level and producing an artificially negative (submarine-looking) reading. Searching
for peak wrist elevation within a small window around release recovers the
biomechanically correct "top of the arm arc" measurement.

Angles are computed in pixel space (not normalized coords) so that portrait-mode
video (e.g., 1080x1920) does not distort the aspect ratio.

The result is direction-agnostic: abs(dx_pixels) is used so the formula works
whether the pitcher faces left or right in the frame.
"""

import numpy as np
import pandas as pd

from ._landmarks import THROWING_SHOULDER, THROWING_WRIST
from ._pose_access import build_window_data, get_landmark_px, check_visibility
from ._types import MetricResult

_METRIC_NAME = "arm_slot"
_DISPLAY_NAME = "Arm Slot at Release"
_PHASE = "ball_release"

# Frames to search before / after ball_release for peak arm elevation.
# At 30 fps: 5 frames ≈ 165 ms before, 2 frames ≈ 65 ms after.
# Tune these if peak elevation consistently falls outside the window.
ARM_SLOT_SEARCH_BEFORE_RELEASE = 5
ARM_SLOT_SEARCH_AFTER_RELEASE  = 2

_VIS_THRESHOLD = 0.5


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Compute arm slot angle (degrees) at peak arm elevation near ball release."""
    br_frame     = int(phases[_PHASE])
    w            = int(video_metadata["width"])
    h            = int(video_metadata["height"])
    shoulder_idx = THROWING_SHOULDER[handedness]
    wrist_idx    = THROWING_WRIST[handedness]

    n_frames     = int(pose_df["frame"].max()) + 1
    search_start = max(0, br_frame - ARM_SLOT_SEARCH_BEFORE_RELEASE)
    search_end   = min(n_frames - 1, br_frame + ARM_SLOT_SEARCH_AFTER_RELEASE)

    window_data = build_window_data(pose_df, search_start, search_end)
    _nan4 = (float("nan"), float("nan"), float("nan"), 0.0)

    # Find the frame in the window where the wrist is highest relative to the
    # shoulder (maximum dy_up = shoulder_y - wrist_y in pixel coords).
    best_frame: int | None = None
    best_dy_up = float("-inf")
    for f in range(search_start, search_end + 1):
        s_e  = window_data.get((f, shoulder_idx), _nan4)
        wr_e = window_data.get((f, wrist_idx),    _nan4)
        if s_e[3] < _VIS_THRESHOLD or wr_e[3] < _VIS_THRESHOLD:
            continue
        dy_up = (s_e[1] * h) - (wr_e[1] * h)
        if dy_up > best_dy_up:
            best_dy_up = dy_up
            best_frame = f

    if best_frame is None:
        # No frame in the window met visibility — fall back to exact ball_release.
        ok, err = check_visibility(pose_df, br_frame, [shoulder_idx, wrist_idx])
        if not ok:
            return MetricResult(
                name=_METRIC_NAME, display_name=_DISPLAY_NAME,
                value=None, unit="degrees", frame=br_frame, phase=_PHASE,
                error=f"Low visibility in search window (frames {search_start}–{search_end}): {err}",
            )
        best_frame = br_frame

    shoulder_px = get_landmark_px(pose_df, best_frame, shoulder_idx, w, h)
    wrist_px    = get_landmark_px(pose_df, best_frame, wrist_idx,    w, h)

    # Elevation angle of wrist above shoulder in pixel space.
    # dy_up > 0 when wrist is higher in the frame (smaller pixel-y) than shoulder.
    # abs(dx) keeps result invariant to pitcher facing direction.
    dx    = wrist_px[0] - shoulder_px[0]
    dy_up = shoulder_px[1] - wrist_px[1]
    angle = float(np.degrees(np.arctan2(dy_up, abs(dx))))

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(angle, 1),
        unit="degrees",
        description="Throwing arm elevation angle at peak arm position near release: 90=overhead, 0=sidearm, negative=submarine.",
        frame=best_frame,
        phase=_PHASE,
        notes=(
            f"Measured at frame of peak arm elevation within "
            f"[ball_release-{ARM_SLOT_SEARCH_BEFORE_RELEASE}, "
            f"ball_release+{ARM_SLOT_SEARCH_AFTER_RELEASE}] "
            f"(frame {best_frame}, ball_release={br_frame}). "
            "90=overhead, 0=sidearm, negative=submarine. "
            "Computed in pixel space. "
            "2D approximation from single side-facing camera."
        ),
    )
