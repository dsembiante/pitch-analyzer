"""
Metric #8: Balance point drift at leg lift peak.

Definition: horizontal displacement of the hip midpoint relative to the back
foot at the moment of maximum leg lift, expressed as a percentage of body height.

Positive drift = hip midpoint is toward the plate (good forward momentum).
Negative drift = hip midpoint is drifting away from the plate (timing leak).

Plate direction is inferred from the stride: the lead ankle at foot_strike
should be further toward the plate than the back ankle at start_of_motion.

Assumption: the back foot does not move meaningfully between start_of_motion
and leg_lift_peak. For IMG_8605 both phases are frame 172 so this is trivially
true. For other videos where they differ, back_ankle position is sampled at
leg_lift_peak (the measurement frame). See LIMITATIONS.md.

Typical range: 0-10% of body height. >20% or any negative value is unusual.
"""

import numpy as np
import pandas as pd

from ._landmarks import LEFT_HIP, RIGHT_HIP, BACK_ANKLE, LEAD_ANKLE
from ._pose_access import get_landmark_px, check_visibility
from ._geometry import midpoint
from ._types import MetricResult

_METRIC_NAME = "balance_point"
_DISPLAY_NAME = "Balance Point Drift"
_PHASE = "leg_lift_peak"
_REQUIRED_LMS = [LEFT_HIP, RIGHT_HIP]


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> MetricResult:
    """Compute balance point drift (% body height) at leg lift peak."""
    ll_frame  = int(phases["leg_lift_peak"])
    som_frame = int(phases["start_of_motion"])
    fs_frame  = int(phases["foot_strike"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])
    body_height_px = float(video_metadata["body_height_pixels"])

    if np.isnan(body_height_px) or body_height_px <= 0:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="percent_body_height",
            frame=ll_frame, phase=_PHASE,
            error="body_height_pixels is invalid; check nose/ankle visibility at release frame",
        )

    back_ankle_idx = BACK_ANKLE[handedness]
    lead_ankle_idx = LEAD_ANKLE[handedness]

    # Ankle landmarks at the leg lift frame can have low MediaPipe confidence
    # while still providing usable position data; use a relaxed threshold.
    ok, err = check_visibility(pose_df, ll_frame, _REQUIRED_LMS + [back_ankle_idx], threshold=0.25)
    if not ok:
        return MetricResult(
            name=_METRIC_NAME, display_name=_DISPLAY_NAME,
            value=None, unit="percent_body_height",
            frame=ll_frame, phase=_PHASE,
            error=f"Low visibility at leg_lift_peak frame {ll_frame}: {err}",
        )

    # --- Hip midpoint in pixel space at leg lift ---
    lh_px = get_landmark_px(pose_df, ll_frame, LEFT_HIP,  w, h)
    rh_px = get_landmark_px(pose_df, ll_frame, RIGHT_HIP, w, h)
    hip_mid_px = midpoint(lh_px, rh_px)

    # --- Back ankle at leg lift peak (the planted pivot foot) ---
    back_ankle_px = get_landmark_px(pose_df, ll_frame, back_ankle_idx, w, h)

    # --- Determine plate direction from stride ---
    # Back ankle at start_of_motion vs lead ankle at foot_strike.
    back_at_som  = get_landmark_px(pose_df, som_frame, back_ankle_idx, w, h)
    lead_at_fs   = get_landmark_px(pose_df, fs_frame,  lead_ankle_idx, w, h)
    plate_direction = "right" if lead_at_fs[0] > back_at_som[0] else "left"

    # --- Signed drift toward the plate ---
    if plate_direction == "right":
        drift_px = hip_mid_px[0] - back_ankle_px[0]
    else:
        drift_px = back_ankle_px[0] - hip_mid_px[0]

    pct = (drift_px / body_height_px) * 100.0

    return MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(pct, 1),
        unit="percent_body_height",
        description="Hip drift over back foot at peak leg lift, as % of body height. Positive = toward plate. Typical: 0-10%.",
        frame=ll_frame,
        phase=_PHASE,
        notes=(
            f"plate_direction={plate_direction}. "
            f"hip_mid_x={hip_mid_px[0]:.1f}px, back_ankle_x={back_ankle_px[0]:.1f}px, "
            f"drift_px={drift_px:.1f}, body_height_px={body_height_px:.1f}. "
            "Positive = hips drifted toward plate. Typical range: 0-10%."
        ),
    )
