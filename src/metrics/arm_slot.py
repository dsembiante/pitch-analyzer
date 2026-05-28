"""
Metric #1: Arm slot angle at ball release.

Definition: angle of the throwing arm (shoulder-to-wrist line) relative to
horizontal at the moment of ball release.

  +90 deg = straight overhead (3/4 to overhand)
    0 deg = sidearm (wrist level with shoulder)
  -90 deg = submarine (wrist below shoulder)

Angles are computed in pixel space (not normalized coords) so that portrait-mode
video (e.g., 1080x1920) does not distort the aspect ratio.

The result is direction-agnostic: abs(dx_pixels) is used so the formula works
whether the pitcher faces left or right in the frame.
"""

import numpy as np
import pandas as pd

from ._landmarks import THROWING_SHOULDER, THROWING_WRIST
from ._pose_access import get_landmark_px, check_visibility
from ._types import MetricResult

_METRIC_NAME = "arm_slot"
_DISPLAY_NAME = "Arm Slot at Release"
_PHASE = "ball_release"


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Compute arm slot angle (degrees) at ball release."""
    frame = int(phases[_PHASE])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    shoulder_idx = THROWING_SHOULDER[handedness]
    wrist_idx    = THROWING_WRIST[handedness]

    ok, err = check_visibility(pose_df, frame, [shoulder_idx, wrist_idx])
    if not ok:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees", frame=frame, phase=_PHASE,
            error=f"Low visibility at frame {frame}: {err}",
        )

    shoulder_px = get_landmark_px(pose_df, frame, shoulder_idx, w, h)
    wrist_px    = get_landmark_px(pose_df, frame, wrist_idx,    w, h)

    # Elevation angle of wrist above shoulder in pixel space.
    # dy_up > 0 when wrist pixel-y is smaller than shoulder pixel-y (higher in frame).
    # abs(dx) keeps result invariant to pitcher facing direction.
    dx    = wrist_px[0] - shoulder_px[0]
    dy_up = shoulder_px[1] - wrist_px[1]
    angle = float(np.degrees(np.arctan2(dy_up, abs(dx))))

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(angle, 1),
        unit="degrees",
        description="Throwing arm elevation angle at release: 90=overhead, 0=sidearm, negative=submarine.",
        frame=frame,
        phase=_PHASE,
        notes=(
            "90=overhead, 0=sidearm, negative=submarine. "
            "Computed in pixel space. "
            "2D approximation from single side-facing camera."
        ),
    )
