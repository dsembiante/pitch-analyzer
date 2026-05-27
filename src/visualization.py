import cv2
import pandas as pd
from pathlib import Path

# Canonical MediaPipe Pose connections — 35 edges over 33 landmarks
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

_RED    = (0, 0, 255)    # throwing arm
_GREEN  = (0, 255, 0)    # lead leg
_YELLOW = (0, 255, 255)  # everything else


def _region_sets(handedness: str) -> tuple[set, set]:
    """Return (throwing_arm_indices, lead_leg_indices) for the given handedness."""
    if handedness == "right":
        # Throwing arm = right side; lead leg = left side (strides toward plate)
        return {12, 14, 16, 18, 20, 22}, {23, 25, 27, 29, 31}
    else:
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


def draw_pose_overlay(
    video_path: str,
    pose_data: pd.DataFrame,
    output_path: str,
    handedness: str = "right",
    visibility_threshold: float = 0.5,
) -> None:
    """
    Draw MediaPipe skeleton on each frame of the input video and write to output_path.

    Args:
        video_path: Path to original .mp4/.mov video
        pose_data: DataFrame from Phase 1 extraction (long format, one row per landmark per frame)
        output_path: Path for output video (.mp4)
        handedness: "right" or "left" — determines throwing arm and lead leg coloring
        visibility_threshold: Skip drawing landmarks below this visibility score
    """
    throwing_arm, lead_leg = _region_sets(handedness)

    # Pre-index landmarks by frame for O(1) per-frame lookup
    frame_landmarks: dict[int, dict[int, tuple]] = {}
    for frame_idx, grp in pose_data.groupby("frame"):
        frame_landmarks[int(frame_idx)] = {
            int(row.landmark_idx): (row.x, row.y, row.visibility)
            for row in grp.itertuples()
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
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lm_data = frame_landmarks.get(frame_idx, {})

            # Pixel coords for all landmarks that pass the visibility threshold.
            # Values outside [0,1] are kept — OpenCV clips off-screen drawing automatically.
            coords: dict[int, tuple[int, int]] = {}
            for lm_idx, (nx, ny, vis) in lm_data.items():
                if vis >= visibility_threshold:
                    coords[lm_idx] = (int(nx * width), int(ny * height))

            # Connections first so landmark circles sit on top
            for (a, b) in POSE_CONNECTIONS:
                if a in coords and b in coords:
                    cv2.line(frame, coords[a], coords[b],
                             _connection_color(a, b, throwing_arm, lead_leg), 2)

            for lm_idx, (px, py) in coords.items():
                cv2.circle(frame, (px, py), 4,
                           _landmark_color(lm_idx, throwing_arm, lead_leg), -1)

            writer.write(frame)

            if (frame_idx + 1) % 30 == 0:
                print(f"Processed {frame_idx + 1}/{total} frames")

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"Done - wrote {frame_idx} frames to {out_path}")
