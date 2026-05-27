"""Helpers for extracting landmark data from the long-format pose DataFrame."""

import numpy as np
import pandas as pd


def get_landmark(
    pose_df: pd.DataFrame,
    frame: int,
    landmark_idx: int,
) -> tuple[float, float, float, float]:
    """Return (x, y, z, visibility) for a landmark at a specific frame.

    Returns (nan, nan, nan, 0.0) if the landmark is absent in that frame.
    """
    mask = (pose_df["frame"] == frame) & (pose_df["landmark_idx"] == landmark_idx)
    rows = pose_df[mask]
    if rows.empty:
        return (float("nan"), float("nan"), float("nan"), 0.0)
    r = rows.iloc[0]
    return (float(r["x"]), float(r["y"]), float(r["z"]), float(r["visibility"]))


def get_landmark_xy(
    pose_df: pd.DataFrame,
    frame: int,
    landmark_idx: int,
) -> tuple[float, float]:
    """Convenience wrapper returning only (x, y) for a landmark at a frame."""
    x, y, _, _ = get_landmark(pose_df, frame, landmark_idx)
    return (x, y)


def get_landmark_px(
    pose_df: pd.DataFrame,
    frame: int,
    landmark_idx: int,
    video_width: int,
    video_height: int,
) -> tuple[float, float]:
    """Return pixel-space (x, y) for a landmark, converting from normalized coords.

    Necessary for aspect-ratio-correct angle calculations — never compute angles
    from raw normalized coordinates on non-square frames.
    """
    nx, ny = get_landmark_xy(pose_df, frame, landmark_idx)
    return (nx * video_width, ny * video_height)


def check_visibility(
    pose_df: pd.DataFrame,
    frame: int,
    landmark_indices: list[int],
    threshold: float = 0.5,
) -> tuple[bool, str]:
    """Check that all specified landmarks meet the visibility threshold.

    Returns:
        (True, "") if all landmarks pass.
        (False, message) naming the first failing landmark and its visibility.
    """
    for idx in landmark_indices:
        _, _, _, vis = get_landmark(pose_df, frame, idx)
        if vis < threshold:
            return (
                False,
                f"landmark {idx} has visibility {vis:.2f} (threshold {threshold:.2f})",
            )
    return (True, "")
