"""
Phase visualization — skeleton overlay + delivery phase labels.

Combines the Phase 2 skeleton drawing with per-phase event annotations:
  - Colored text label in the upper-left, visible for ~10 frames
  - Highlighted circle on the relevant landmark for the same duration
"""

import cv2
import pandas as pd
from pathlib import Path
from visualization import POSE_CONNECTIONS

# -- handedness mappings (mirrored from phase_detection for self-containment) --
_WRIST_IDX      = {"right": 16, "left": 15}
_LEAD_ANKLE_IDX = {"right": 27, "left": 28}

# -- BGR colors --
_RED    = (0, 0, 255)
_GREEN  = (0, 255, 0)
_YELLOW = (0, 255, 255)

# (label_text, BGR_color, phases_dict_key)
_PHASE_DISPLAY = [
    ("BALL RELEASE",       (0,   0, 255), "ball_release"),
    ("MAX LAYBACK",        (0, 100, 255), "max_layback"),
    ("FOOT STRIKE",        (0, 255, 255), "foot_strike"),
    ("LEG LIFT PEAK",      (0, 255, 128), "leg_lift_peak"),
    ("START OF MOTION",    (0, 255,   0), "start_of_motion"),
    ("END OF MOTION",      (255, 0, 255), "end_of_motion"),
]

# Which landmark to highlight for each phase (None = no highlight)
_PHASE_HIGHLIGHT = {
    "ball_release":          lambda h: _WRIST_IDX[h],
    "max_layback":           lambda h: _WRIST_IDX[h],
    "foot_strike":           lambda h: _LEAD_ANKLE_IDX[h],
    "leg_lift_peak":         lambda h: _LEAD_ANKLE_IDX[h],
    "start_of_motion":       lambda h: None,
    "end_of_motion":         lambda h: None,
}


def _region_sets(handedness: str) -> tuple:
    if handedness == "right":
        return {12, 14, 16, 18, 20, 22}, {23, 25, 27, 29, 31}
    return {11, 13, 15, 17, 19, 21}, {24, 26, 28, 30, 32}


def _landmark_color(idx: int, throwing_arm: set, lead_leg: set) -> tuple:
    if idx in throwing_arm:
        return _RED
    if idx in lead_leg:
        return _GREEN
    return _YELLOW


def _connection_color(a: int, b: int, throwing_arm: set, lead_leg: set) -> tuple:
    if a in throwing_arm and b in throwing_arm:
        return _RED
    if a in lead_leg and b in lead_leg:
        return _GREEN
    return _YELLOW


def draw_phases_on_video(
    video_path: str,
    pose_data: pd.DataFrame,
    phases: dict,
    output_path: str,
    handedness: str = "right",
    label_duration: int = 10,
    visibility_threshold: float = 0.5,
) -> None:
    """Draw skeleton overlay with phase event labels on each video frame.

    Each phase label appears at the top-left for ``label_duration`` frames
    starting at the detected phase frame. Multiple simultaneous labels stack
    vertically. The relevant landmark is circled for the same duration.

    Args:
        video_path: Path to original video (.mp4/.mov).
        pose_data: DataFrame from Phase 1 extraction.
        phases: Dict returned by detect_all_phases.
        output_path: Output video path (.mp4).
        handedness: "right" or "left".
        label_duration: Frames to keep each label visible (~10 ≈ 0.33 s).
        visibility_threshold: Skip landmarks below this visibility score.
    """
    throwing_arm, lead_leg = _region_sets(handedness)

    # Build frame → [(label, color, highlight_lm)] lookup
    frame_labels: dict = {}
    for label, color, phase_key in _PHASE_DISPLAY:
        if phase_key not in phases:
            continue
        phase_frame = int(phases[phase_key])
        highlight_fn = _PHASE_HIGHLIGHT.get(phase_key)
        highlight_lm = highlight_fn(handedness) if highlight_fn else None
        for offset in range(label_duration):
            frame_labels.setdefault(phase_frame + offset, []).append(
                (label, color, highlight_lm)
            )

    # Pre-index landmark data by frame for O(1) lookup
    frame_landmarks: dict = {}
    for frame_idx, grp in pose_data.groupby("frame"):
        frame_landmarks[int(frame_idx)] = {
            int(r.landmark_idx): (r.x, r.y, r.visibility)
            for r in grp.itertuples()
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or "?"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")

    # Font parameters scaled to be readable on portrait 1080×1920 frames
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness  = 3
    shadow_th  = 6
    line_gap   = 80    # vertical spacing between stacked labels
    label_x    = 30
    label_y0   = 95    # y-position of first label

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- skeleton overlay ---
            lm_data = frame_landmarks.get(frame_idx, {})
            coords: dict = {}
            for lm_idx, (nx, ny, vis) in lm_data.items():
                if vis >= visibility_threshold:
                    coords[lm_idx] = (int(nx * width), int(ny * height))

            for (a, b) in POSE_CONNECTIONS:
                if a in coords and b in coords:
                    cv2.line(frame, coords[a], coords[b],
                             _connection_color(a, b, throwing_arm, lead_leg), 2)
            for lm_idx, pt in coords.items():
                cv2.circle(frame, pt, 4,
                           _landmark_color(lm_idx, throwing_arm, lead_leg), -1)

            # --- phase labels ---
            labels = frame_labels.get(frame_idx, [])
            for i, (label, color, highlight_lm) in enumerate(labels):
                y = label_y0 + i * line_gap
                # Shadow for readability on any background
                cv2.putText(frame, label, (label_x, y), font, font_scale, (0, 0, 0), shadow_th)
                cv2.putText(frame, label, (label_x, y), font, font_scale, color, thickness)
                # Landmark highlight circle
                if highlight_lm is not None and highlight_lm in coords:
                    cv2.circle(frame, coords[highlight_lm], 16, color, 3)

            writer.write(frame)

            if (frame_idx + 1) % 30 == 0:
                print(f"Processed {frame_idx + 1}/{total} frames")
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"Done - wrote {frame_idx} frames to {out_path}")
