"""
Metric #2: Stride length normalized to body height.

Definition: horizontal pixel distance between the back foot at start_of_motion
and the lead foot at foot_strike, expressed as a percentage of body height.

Horizontal-only distance is used (not diagonal/Euclidean) because stride length
is conventionally measured along the direction of pitch — the distance the front
foot advanced toward the plate. Vertical foot-height differences between setup
and landing would contaminate a diagonal measurement.

Typical elite range: 80-100%+ of body height.
Values under 50% or over 130% are suspect.
"""

import numpy as np
import pandas as pd

from ._landmarks import BACK_ANKLE, LEAD_ANKLE
from ._pose_access import get_landmark_px, check_visibility
from ._types import MetricResult

_METRIC_NAME = "stride_length"
_DISPLAY_NAME = "Stride Length"
_PHASE = "foot_strike"


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Compute stride length as a percentage of body height."""
    som_frame = int(phases["start_of_motion"])
    fs_frame  = int(phases["foot_strike"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])
    body_height_px = float(video_metadata["body_height_pixels"])

    if np.isnan(body_height_px) or body_height_px <= 0:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="percent_body_height",
            frame=fs_frame, phase=_PHASE,
            error="body_height_pixels is invalid (NaN or zero); check nose/ankle visibility at release frame",
        )

    back_ankle_idx = BACK_ANKLE[handedness]
    lead_ankle_idx = LEAD_ANKLE[handedness]

    # Ankle landmarks at the setup frame can have low MediaPipe confidence while
    # still providing usable position data; use a relaxed threshold.
    ok_back, err_back = check_visibility(pose_df, som_frame, [back_ankle_idx], threshold=0.25)
    if not ok_back:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="percent_body_height",
            frame=fs_frame, phase=_PHASE,
            error=f"Back ankle low visibility at start_of_motion frame {som_frame}: {err_back}",
        )

    ok_lead, err_lead = check_visibility(pose_df, fs_frame, [lead_ankle_idx])
    if not ok_lead:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="percent_body_height",
            frame=fs_frame, phase=_PHASE,
            error=f"Lead ankle low visibility at foot_strike frame {fs_frame}: {err_lead}",
        )

    back_ankle_px = get_landmark_px(pose_df, som_frame, back_ankle_idx, w, h)
    lead_ankle_px = get_landmark_px(pose_df, fs_frame,  lead_ankle_idx, w, h)

    # Horizontal distance only — stride length is measured along the plate direction.
    horiz_dist_px = abs(lead_ankle_px[0] - back_ankle_px[0])
    pct = (horiz_dist_px / body_height_px) * 100.0

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(pct, 1),
        unit="percent_body_height",
        description="Horizontal distance from back foot at setup to front foot at landing, as % of body height. Elite range: 80-100%+.",
        frame=fs_frame,
        phase=_PHASE,
        notes=(
            f"horizontal_distance_px={horiz_dist_px:.1f}, body_height_px={body_height_px:.1f}. "
            "Horizontal distance only (not diagonal); see LIMITATIONS.md."
        ),
    )
