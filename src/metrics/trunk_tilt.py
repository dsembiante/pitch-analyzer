"""
Metric #5: Trunk tilt at ball release — two components.

Angles are computed in pixel space to avoid aspect-ratio distortion on
non-square frames (e.g., 1080x1920 portrait video).

Lateral tilt:
  Angle of the shoulder line vs horizontal in pixel space.
  Positive = throwing-side shoulder is lower (larger pixel-y).
  Note: on a side-facing camera the two shoulders are nearly stacked
  (one in front of the other), so their apparent horizontal separation
  is small. This makes lateral tilt measurements noisy from a pure side
  angle; see LIMITATIONS.md.

Forward tilt:
  Angle of the shoulder-midpoint to hip-midpoint line vs downward vertical,
  measured in pixel space. Positive = hip midpoint is to the right of shoulder
  midpoint in the frame. Whether right = toward-the-plate depends on which
  direction the pitcher faces; see LIMITATIONS.md.
"""

import numpy as np
import pandas as pd

from ._landmarks import LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP
from ._pose_access import get_landmark_px, check_visibility
from ._types import MetricResult

_PHASE = "ball_release"
_ALL_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


def _get_px(pose_df: pd.DataFrame, frame: int, w: int, h: int) -> dict:
    return {
        "ls": get_landmark_px(pose_df, frame, LEFT_SHOULDER,  w, h),
        "rs": get_landmark_px(pose_df, frame, RIGHT_SHOULDER, w, h),
        "lh": get_landmark_px(pose_df, frame, LEFT_HIP,       w, h),
        "rh": get_landmark_px(pose_df, frame, RIGHT_HIP,      w, h),
    }


def compute_lateral(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Lateral trunk tilt at release: throwing-side shoulder lower = positive."""
    frame = int(phases[_PHASE])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    ok, err = check_visibility(pose_df, frame, _ALL_LANDMARKS)
    if not ok:
        return MetricResult(
            name="trunk_tilt_lateral", display_name="Trunk Tilt Lateral at Release",
            value=None, unit="degrees", frame=frame, phase=_PHASE,
            error=f"Low visibility at frame {frame}: {err}",
        )

    pts = _get_px(pose_df, frame, w, h)
    throwing_px = pts["rs"] if handedness == "right" else pts["ls"]
    lead_px     = pts["ls"] if handedness == "right" else pts["rs"]

    # positive = throwing shoulder has larger pixel-y (lower in frame)
    delta_y_px = throwing_px[1] - lead_px[1]
    delta_x_px = abs(throwing_px[0] - lead_px[0]) + 1e-6
    angle = float(np.degrees(np.arctan2(delta_y_px, delta_x_px)))

    return MetricResult(
        name="trunk_tilt_lateral",
        display_name="Trunk Tilt Lateral at Release",
        value=round(angle, 1),
        unit="degrees",
        description="Shoulder tilt at release: positive means throwing-side shoulder is lower in the frame.",
        frame=frame,
        phase=_PHASE,
        notes=(
            "Positive = throwing-side shoulder lower in frame. "
            "Side-angle cameras show minimal horizontal shoulder separation; "
            "see LIMITATIONS.md for measurement limitations."
        ),
    )


def compute_forward(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Forward trunk tilt: shoulder-mid to hip-mid line vs downward vertical (pixel space).

    Positive = hip midpoint is to the right of shoulder midpoint in frame.
    Sign interpretation depends on which way the pitcher faces; see LIMITATIONS.md.
    """
    frame = int(phases[_PHASE])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    ok, err = check_visibility(pose_df, frame, _ALL_LANDMARKS)
    if not ok:
        return MetricResult(
            name="trunk_tilt_forward", display_name="Trunk Tilt Forward at Release",
            value=None, unit="degrees", frame=frame, phase=_PHASE,
            error=f"Low visibility at frame {frame}: {err}",
        )

    pts = _get_px(pose_df, frame, w, h)
    shoulder_mid = (
        (pts["ls"][0] + pts["rs"][0]) / 2,
        (pts["ls"][1] + pts["rs"][1]) / 2,
    )
    hip_mid = (
        (pts["lh"][0] + pts["rh"][0]) / 2,
        (pts["lh"][1] + pts["rh"][1]) / 2,
    )

    # Line from shoulder_mid to hip_mid vs downward vertical (0 = perfectly upright).
    # Positive = hip_mid is to the right of shoulder_mid in pixel space.
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle = float(np.degrees(np.arctan2(dx, dy)))

    return MetricResult(
        name="trunk_tilt_forward",
        display_name="Trunk Tilt Forward at Release",
        value=round(angle, 1),
        unit="degrees",
        description="Trunk forward lean at release: angle of the shoulder-to-hip line vs vertical.",
        frame=frame,
        phase=_PHASE,
        notes=(
            "Angle of shoulder-mid-to-hip-mid vs downward vertical (pixel space). "
            "Positive = hips right of shoulders in frame. "
            "Toward-plate direction depends on pitcher facing; see LIMITATIONS.md."
        ),
    )
