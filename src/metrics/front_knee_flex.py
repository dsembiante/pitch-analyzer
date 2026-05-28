"""
Metric #4: Front (lead) knee flexion angle at ball release.

Definition: interior angle at the lead knee formed by the lead hip -> lead
knee -> lead ankle in pixel space. Measured at the ball release frame.

  180 deg = fully extended (straight leg)
  ~150 deg = moderate flex (typical for most pitchers)
  ~90 deg = deeply flexed

Typical range at release: 130-170 degrees. Angles are computed in pixel space
to avoid aspect-ratio distortion on non-square frames.

Return value: tuple[MetricResult, pd.DataFrame].
The MetricResult is the single-frame release angle.
The DataFrame has columns (frame, timestamp_s, knee_angle_deg) covering every
valid frame from foot_strike through end_of_motion, showing the leg stiffening
(or collapsing) arc. An empty DataFrame with the same columns is returned when
the release-frame metric fails.
"""

import pandas as pd

from ._landmarks import LEAD_HIP, LEAD_KNEE, LEAD_ANKLE
from ._pose_access import get_landmark_px, check_visibility, build_window_data
from ._geometry import angle_between_points
from ._types import MetricResult

_METRIC_NAME = "front_knee_flex"
_DISPLAY_NAME = "Front Knee Flex at Release"
_PHASE = "ball_release"
_VIS_THRESHOLD = 0.5

_EMPTY_SERIES = pd.DataFrame(columns=["frame", "timestamp_s", "knee_angle_deg"])


def compute(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    video_metadata: dict,
) -> tuple[MetricResult, pd.DataFrame]:
    """Compute lead knee flexion angle at ball release and per-frame series.

    The MetricResult captures the release-frame angle.
    The series DataFrame covers foot_strike through end_of_motion (the
    leg-block arc), with columns (frame, timestamp_s, knee_angle_deg).
    Returns an empty series DataFrame if the release-frame metric fails.
    """
    release_frame = int(phases[_PHASE])
    fs_frame      = int(phases["foot_strike"])
    eom_frame     = int(phases.get("end_of_motion", release_frame))
    w   = int(video_metadata["width"])
    h   = int(video_metadata["height"])
    fps = float(video_metadata.get("fps", 30.0))

    hip_idx   = LEAD_HIP[handedness]
    knee_idx  = LEAD_KNEE[handedness]
    ankle_idx = LEAD_ANKLE[handedness]

    # --- Single-frame release metric ---
    ok, err = check_visibility(pose_df, release_frame, [hip_idx, knee_idx, ankle_idx])
    if not ok:
        return (
            MetricResult(
                name=_METRIC_NAME, display_name=_DISPLAY_NAME,
                value=None, unit="degrees", frame=release_frame, phase=_PHASE,
                error=f"Low visibility at frame {release_frame}: {err}",
            ),
            _EMPTY_SERIES.copy(),
        )

    hip_px   = get_landmark_px(pose_df, release_frame, hip_idx,   w, h)
    knee_px  = get_landmark_px(pose_df, release_frame, knee_idx,  w, h)
    ankle_px = get_landmark_px(pose_df, release_frame, ankle_idx, w, h)
    release_angle = angle_between_points(hip_px, knee_px, ankle_px)

    metric = MetricResult(
        name=_METRIC_NAME,
        display_name=_DISPLAY_NAME,
        value=round(release_angle, 1),
        unit="degrees",
        description="Lead knee bend at release: 180=straight leg, lower=more flexed. Typical range: 130-170 deg.",
        frame=release_frame,
        phase=_PHASE,
        notes=(
            "Interior angle at lead knee: hip->knee->ankle (pixel space). "
            "180=fully extended, smaller=more flexed. "
            "Typical range at release: 130-170 deg."
        ),
    )

    # --- Per-frame series: foot_strike through end_of_motion ---
    window_data = build_window_data(pose_df, fs_frame, eom_frame)
    frame_list, angle_list = [], []

    for f in range(fs_frame, eom_frame + 1):
        entries = {
            lm: window_data.get((f, lm), (float("nan"), float("nan"), float("nan"), 0.0))
            for lm in (hip_idx, knee_idx, ankle_idx)
        }
        if any(e[3] < _VIS_THRESHOLD for e in entries.values()):
            continue
        h_px = (entries[hip_idx][0]   * w, entries[hip_idx][1]   * h)
        k_px = (entries[knee_idx][0]  * w, entries[knee_idx][1]  * h)
        a_px = (entries[ankle_idx][0] * w, entries[ankle_idx][1] * h)
        frame_list.append(f)
        angle_list.append(angle_between_points(h_px, k_px, a_px))

    series_df = pd.DataFrame({
        "frame":          frame_list,
        "timestamp_s":    [f / fps for f in frame_list],
        "knee_angle_deg": angle_list,
    })

    return metric, series_df
