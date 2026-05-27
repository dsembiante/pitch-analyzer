"""
2D geometry helpers for biomechanics calculations.

All point arguments are (x, y) tuples of pixel coordinates unless otherwise
noted. Use pixel coordinates (norm_x * width, norm_y * height) rather than
normalized [0,1] values — mixing normalized x/y on non-square frames introduces
aspect-ratio distortion that corrupts angle calculations (e.g., on 1080x1920
portrait video a true 45-degree angle measures as ~29 degrees in normalized space).

Image coordinate convention: (0, 0) is top-left, x increases rightward,
y increases downward. Angle sign conventions are documented per function.
"""

import numpy as np
import pandas as pd


def angle_between_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    """Three-point angle in degrees, with p2 as the vertex.

    Computes the interior angle at p2 formed by rays p2->p1 and p2->p3.
    Uses arctan2(|cross|, dot) for numerical stability near 0 and 180 degrees.

    Returns:
        Angle in [0, 180] degrees.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot   = v1[0] * v2[0] + v1[1] * v2[1]
    return float(np.degrees(np.arctan2(abs(cross), dot)))


def angle_of_line_vs_horizontal(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Angle in degrees of the directed line from p1 to p2 relative to horizontal.

    Sign convention: positive = p2 is above p1.
    Because image y increases downward (y=0 is the top of the frame), "above"
    means p2.y < p1.y, so the raw dy is negated before arctan2.

    Returns:
        Angle in [-180, 180] degrees.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(-dy, dx)))


def angle_of_line_vs_vertical(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Angle in degrees of the directed line from p1 to p2 relative to vertical.

    Uses "downward vertical" as the zero reference (0° = p2 directly below p1).
    Positive = p2 is to the right of p1.

    Returns:
        Angle in [-180, 180] degrees.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dx, dy)))


def angle_of_line_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Raw image-plane angle in degrees of the directed line from p1 to p2.

    Uses arctan2(dy, dx) in pixel space with no y-flip — suitable for comparing
    two lines' angles against each other (e.g., hip line vs shoulder line) where
    consistency matters more than "up is positive" intuition.

    Returns:
        Angle in [-180, 180] degrees.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def signed_angle_difference(angle1: float, angle2: float) -> float:
    """Return angle1 - angle2 normalized to [-180, 180].

    Handles wraparound: e.g., 179° vs -179° correctly yields 2°, not 358°.
    Use abs() on the result to get the angular separation magnitude.
    """
    diff = angle1 - angle2
    return float(((diff + 180.0) % 360.0) - 180.0)


def midpoint(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> tuple[float, float]:
    """Midpoint of two 2D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def euclidean_distance(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))


def compute_body_height_pixels(
    pose_df: "pd.DataFrame",
    reference_frame: int,
    video_width: int,
    video_height: int,
) -> float:
    """Pixel distance from nose (landmark 0) to mid-ankle at a reference frame.

    Uses pixel coordinates (normalized x/y multiplied by video dimensions).
    The nose-to-ankle distance serves as a body-size normalizer for stride
    length and release height calculations.

    Args:
        pose_df: Full pose DataFrame from Phase 1.
        reference_frame: Frame number to sample landmark positions from.
        video_width: Video frame width in pixels.
        video_height: Video frame height in pixels.

    Returns:
        Distance in pixels. Returns NaN if required landmarks are missing.
    """
    from ._pose_access import get_landmark

    nose        = get_landmark(pose_df, reference_frame, 0)
    left_ankle  = get_landmark(pose_df, reference_frame, 27)
    right_ankle = get_landmark(pose_df, reference_frame, 28)

    if any(np.isnan(v) for pts in (nose, left_ankle, right_ankle) for v in pts[:2]):
        return float("nan")

    nose_px   = (nose[0] * video_width,  nose[1] * video_height)
    ankle_mid = (
        (left_ankle[0] + right_ankle[0]) / 2 * video_width,
        (left_ankle[1] + right_ankle[1]) / 2 * video_height,
    )
    return euclidean_distance(nose_px, ankle_mid)
