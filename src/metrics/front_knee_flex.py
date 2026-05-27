"""
Metric #4: Front (lead) knee flexion angle at ball release.

Definition: interior angle at the lead knee formed by the lead hip -> lead
knee -> lead ankle in pixel space. Measured at the ball release frame.

  180 deg = fully extended (straight leg)
  ~150 deg = moderate flex (typical for most pitchers)
  ~90 deg = deeply flexed

Typical range at release: 130-170 degrees. Angles are computed in pixel space
to avoid aspect-ratio distortion on non-square frames.
"""

import pandas as pd

from ._landmarks import LEAD_HIP, LEAD_KNEE, LEAD_ANKLE
from ._pose_access import get_landmark_px, check_visibility
from ._geometry import angle_between_points
from ._types import MetricResult

_METRIC_NAME = "front_knee_flex"
_DISPLAY_NAME = "Front Knee Flex at Release"
_PHASE = "ball_release"


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Compute lead knee flexion angle (degrees) at ball release."""
    frame = int(phases[_PHASE])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    hip_idx   = LEAD_HIP[handedness]
    knee_idx  = LEAD_KNEE[handedness]
    ankle_idx = LEAD_ANKLE[handedness]

    ok, err = check_visibility(pose_df, frame, [hip_idx, knee_idx, ankle_idx])
    if not ok:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="degrees", frame=frame, phase=_PHASE,
            error=f"Low visibility at frame {frame}: {err}",
        )

    hip_px   = get_landmark_px(pose_df, frame, hip_idx,   w, h)
    knee_px  = get_landmark_px(pose_df, frame, knee_idx,  w, h)
    ankle_px = get_landmark_px(pose_df, frame, ankle_idx, w, h)

    angle = angle_between_points(hip_px, knee_px, ankle_px)

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(angle, 1),
        unit="degrees",
        frame=frame,
        phase=_PHASE,
        notes=(
            "Interior angle at lead knee: hip->knee->ankle (pixel space). "
            "180=fully extended, smaller=more flexed. "
            "Typical range at release: 130-170 deg."
        ),
    )
