"""
Phase detection for baseball pitching motion analysis (Phase 3).

Detects six key delivery phases from MediaPipe pose data:
  1. Start of Motion — first significant movement from setup position
  2. Leg Lift Peak   — lead ankle at maximum height (min y)
  3. Foot Strike (FP) — lead foot contacts the ground
  4. Max Layback     — throwing wrist at its furthest layback position before acceleration
  5. Ball Release (BR) — throwing wrist at peak forward velocity
  6. End of Motion   — follow-through complete, wrist velocity decays

All detectors are designed around a right or left-handed pitcher.
Landmarks are referenced by MediaPipe Pose index (0–32).

Design notes:
  - The pitching window from detect_pitching_window is the "high-wrist-velocity"
    region. Early phases (max_layback, foot strike, leg lift) routinely occur BEFORE this
    window's start, so downstream detectors use fixed lookbacks from release_frame
    or foot_strike_frame rather than window[0] as their lower bound.
  - All series are linearly interpolated before smoothing to handle NaN gaps.
  - Bail-out checks raise ValueError with context when >20% of a critical
    landmark series is missing.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

# --- landmark index mappings ---
_WRIST_IDX      = {"right": 16, "left": 15}   # throwing wrist
_LEAD_ANKLE_IDX = {"right": 27, "left": 28}   # lead (stride) ankle
_HIP_SHOULDER_IDX = [11, 12, 23, 24]          # for total-motion calculation

# --- confidence-check thresholds (adjust here if video conditions differ) ---
_LEG_LIFT_MIN_ANKLE_VIS       = 0.65   # mean visibility of lead ankle in lift window; below → occluded
_LEG_LIFT_MIN_PEAK_PROMINENCE = 0.02   # normalized-y drop; leg must dip ≥2 % of frame height
_SOM_NEAR_FRAME_ZERO          = 5      # start_of_motion ≤ this many frames from 0 triggers extra check


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def smooth_series(series: "pd.Series | np.ndarray", window: int = 7, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a 1-D series.

    Falls back to raw values when the series is too short for the requested
    window (savgol requires window length < len(series)).

    Args:
        series: Input array or Series. Should be NaN-free before calling.
        window: Window length (must be odd; bumped up by 1 if even).
        polyorder: Polynomial order (must be < window).

    Returns:
        Smoothed numpy float array of the same length.
    """
    arr = np.asarray(series, dtype=float)
    if len(arr) < window:
        return arr
    w = window if window % 2 == 1 else window + 1
    return savgol_filter(arr, w, polyorder)


def get_landmark_series(pose_df: pd.DataFrame, landmark_idx: int, coord: str) -> np.ndarray:
    """Extract a coordinate time series for a single landmark across all frames.

    Frames where the landmark was not detected are represented as NaN so that
    callers can decide how to handle gaps (interpolate, bail, etc.).

    Args:
        pose_df: Full pose DataFrame from Phase 1.
        landmark_idx: MediaPipe landmark index (0–32).
        coord: Column to extract: 'x', 'y', 'z', or 'visibility'.

    Returns:
        float64 array of length ``max_frame + 1``, NaN where frame is absent.
    """
    sub = pose_df[pose_df.landmark_idx == landmark_idx].sort_values("frame")
    n_frames = int(pose_df.frame.max()) + 1
    out = np.full(n_frames, np.nan)
    out[sub.frame.values] = sub[coord].values
    return out


def compute_velocity(series: np.ndarray, fps: float) -> np.ndarray:
    """Per-second velocity via central differences (numpy.gradient).

    Args:
        series: 1-D array of positional values. Should be NaN-free.
        fps: Frames per second.

    Returns:
        Velocity array in [series_units / second].
    """
    return np.gradient(series, 1.0 / fps)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_nan_fraction(series: np.ndarray, name: str, threshold: float = 0.20) -> None:
    """Raise ValueError if NaN fraction exceeds threshold."""
    frac = float(np.isnan(series).mean())
    if frac > threshold:
        raise ValueError(
            f"Landmark '{name}' is missing in {frac:.0%} of frames (limit {threshold:.0%}). "
            "The landmark may be occluded. Check video quality and angle."
        )


def _interpolate(series: np.ndarray) -> np.ndarray:
    """Fill NaN gaps with linear interpolation; forward/back-fill at edges."""
    return pd.Series(series).interpolate(method="linear", limit_direction="both").values


def _leg_lift_confidence(
    pose_df: pd.DataFrame,
    handedness: str,
    foot_strike_frame: int,
    leg_lift_frame: int,
    lookback: int = 100,
) -> "tuple[bool, list[str]]":
    """Return (confident, notes) for leg_lift_peak.

    Two independent signals must BOTH pass; failing either trips the flag:
    1. Mean lead ankle visibility ≥ _LEG_LIFT_MIN_ANKLE_VIS over the search window.
    2. Prominence of the detected minimum ≥ _LEG_LIFT_MIN_PEAK_PROMINENCE (normalized y).
    """
    ankle_idx    = _LEAD_ANKLE_IDX[handedness]
    search_start = max(0, foot_strike_frame - lookback)
    notes: list[str] = []
    confident = True

    # ── Check 1: mean lead ankle visibility ──────────────────────────────
    vis_raw    = get_landmark_series(pose_df, ankle_idx, "visibility")
    window_vis = vis_raw[search_start:foot_strike_frame]
    # Treat NaN (landmark row absent) as 0 — not visible
    filled_vis = np.where(np.isnan(window_vis), 0.0, window_vis)
    mean_vis   = float(np.mean(filled_vis)) if len(filled_vis) > 0 else 0.0
    if mean_vis < _LEG_LIFT_MIN_ANKLE_VIS:
        notes.append(
            f"Leg lift not confidently detected: lead ankle occluded "
            f"(mean visibility {mean_vis:.2f} < {_LEG_LIFT_MIN_ANKLE_VIS})"
        )
        confident = False

    # ── Check 2: prominence of the detected y-minimum ────────────────────
    raw    = get_landmark_series(pose_df, ankle_idx, "y")
    y      = _interpolate(raw)
    y_smth = smooth_series(pd.Series(y))
    seg    = y_smth[search_start:foot_strike_frame]
    if len(seg) > 1:
        local_idx = max(0, min(leg_lift_frame - search_start, len(seg) - 1))
        y_at_min  = float(seg[local_idx])
        left_max  = float(np.max(seg[:local_idx]))   if local_idx > 0             else y_at_min
        right_max = float(np.max(seg[local_idx+1:])) if local_idx < len(seg) - 1 else y_at_min
        prominence = min(left_max, right_max) - y_at_min
        if prominence < _LEG_LIFT_MIN_PEAK_PROMINENCE:
            notes.append(
                f"Leg lift not confidently detected: peak prominence too small "
                f"({prominence:.4f} < {_LEG_LIFT_MIN_PEAK_PROMINENCE}) — signal may be flat or occluded"
            )
            confident = False

    return confident, notes


def _start_of_motion_confidence(
    pose_df: pd.DataFrame,
    start_of_motion_frame: int,
    leg_lift_frame: int,
    fps: float,
    lookback: int = 80,
) -> "tuple[bool, list[str]]":
    """Return (confident, notes) for start_of_motion.

    Trips when start_of_motion is within _SOM_NEAR_FRAME_ZERO frames of clip start
    AND the value equals the search boundary (detect_start_of_motion never found a
    still block and returned its initial fallback value). This distinguishes a genuine
    early-clip onset from a detector that fell back to the clip boundary.
    """
    notes: list[str] = []

    if start_of_motion_frame >= _SOM_NEAR_FRAME_ZERO:
        return True, notes  # well away from frame 0 — confident

    # Near frame 0: check whether the detector fell back to the search boundary
    # (last_start_of_motion == search_start means no still block was ever found).
    search_start = max(0, leg_lift_frame - lookback)
    if start_of_motion_frame <= search_start:
        notes.append(
            f"Start of motion not confidently detected: defaulted to frame {start_of_motion_frame} "
            f"— no still period found before leg lift (clip may not include full setup)"
        )
        return False, notes

    return True, notes


# ---------------------------------------------------------------------------
# Phase detectors
# ---------------------------------------------------------------------------

def detect_pitching_window(
    pose_df: pd.DataFrame,
    handedness: str,
    fps: float,
) -> tuple:
    """Locate the pitching delivery window using throwing-wrist horizontal velocity.

    Algorithm:
      1. Smooth wrist x-position; compute |velocity|.
      2. Find sustained velocity peaks (scipy find_peaks, width >= 5 frames at
         half-prominence). The 'width' filter separates the genuine throwing burst
         (wide, multi-frame peak) from brief arm-swing spikes during walking.
      3. Select the highest sustained peak as the release candidate.
      4. Expand backward and forward from that peak: stop when |velocity| stays
         below 15 % of peak for 5+ consecutive frames.
      5. Sanity check: window must be 0.5–3.0 s (reasonable delivery range).

    Args:
        pose_df: Full pose DataFrame.
        handedness: "right" or "left".
        fps: Frames per second.

    Returns:
        (start_frame, peak_frame, end_frame)

    Raises:
        ValueError: If no clear peak is found or the window fails the sanity check.
    """
    wrist_idx = _WRIST_IDX[handedness]
    raw = get_landmark_series(pose_df, wrist_idx, "x")
    _check_nan_fraction(raw, f"{handedness}_wrist_x")

    x = _interpolate(raw)
    x_smooth = smooth_series(pd.Series(x))
    vel = compute_velocity(x_smooth, fps)
    abs_vel = np.abs(vel)

    # width >= 5: the genuine throwing burst spans many frames at half-prominence;
    # brief walking arm-swings do not.
    peaks, _ = find_peaks(abs_vel, width=5, prominence=0.3)
    if len(peaks) == 0:
        peaks, _ = find_peaks(abs_vel, prominence=0.1)
    if len(peaks) == 0:
        raise ValueError(
            "No clear throwing-wrist velocity peak found. "
            "Confirm the video contains a full pitching delivery and the wrist is visible."
        )

    peak_frame = int(peaks[np.argmax(abs_vel[peaks])])
    peak_vel = float(abs_vel[peak_frame])
    threshold = 0.15 * peak_vel
    n = len(abs_vel)

    # Expand backward
    start_frame = 0
    consecutive = 0
    last_high = peak_frame
    for f in range(peak_frame - 1, -1, -1):
        if abs_vel[f] < threshold:
            consecutive += 1
            if consecutive >= 5:
                start_frame = last_high + 1
                break
        else:
            consecutive = 0
            last_high = f

    # Expand forward
    end_frame = n - 1
    consecutive = 0
    last_high = peak_frame
    for f in range(peak_frame + 1, n):
        if abs_vel[f] < threshold:
            consecutive += 1
            if consecutive >= 5:
                end_frame = last_high
                break
        else:
            consecutive = 0
            last_high = f

    duration_s = (end_frame - start_frame) / fps
    if not (0.5 <= duration_s <= 3.0):
        raise ValueError(
            f"Detected pitching window ({start_frame}–{end_frame}, {duration_s:.2f}s) "
            f"is outside the expected 0.5–3.0s range. "
            f"Wrist velocity peak was at frame {peak_frame} (|vel|={peak_vel:.3f}). "
            "The video may contain no full delivery, or the wrist landmark is unreliable."
        )

    return start_frame, peak_frame, end_frame


def detect_ball_release(
    pose_df: pd.DataFrame,
    handedness: str,
    fps: float,
    window: tuple,
) -> int:
    """Return the frame of peak throwing-wrist |horizontal velocity| within the window.

    Equivalent to the peak_frame from detect_pitching_window; exposed separately
    so it can be called independently or with a narrowed window.

    Args:
        pose_df: Full pose DataFrame.
        handedness: "right" or "left".
        fps: Frames per second.
        window: (start_frame, end_frame) search bounds.

    Returns:
        Frame number of ball release.
    """
    wrist_idx = _WRIST_IDX[handedness]
    raw = get_landmark_series(pose_df, wrist_idx, "x")
    x = _interpolate(raw)
    x_smooth = smooth_series(pd.Series(x))
    vel = compute_velocity(x_smooth, fps)
    abs_vel = np.abs(vel)

    s, e = int(window[0]), min(int(window[1]), len(abs_vel) - 1)
    return int(s + np.argmax(abs_vel[s:e + 1]))


def detect_max_layback(
    pose_df: pd.DataFrame,
    handedness: str,
    release_frame: int,
    window: tuple,
    foot_strike_frame: "int | None" = None,
    lookback: int = 40,
) -> int:
    """Detect max layback — throwing wrist at its furthest layback position before acceleration.

    This is a 2D proxy for true (3D) shoulder maximum external rotation.
    We measure when the throwing wrist reaches its most extreme horizontal position
    opposite to the release direction, which corresponds to the arm being fully
    cocked. Because this is a single-camera 2D measurement, the detected frame can
    occur slightly before foot strike due to camera projection; this is expected
    and documented in LIMITATIONS.md.

    Algorithm:
      Max layback is when the wrist is at its most extreme position *opposite* to
      the direction it travels during release. We determine the release direction
      from the sign of wrist x displacement over the 5 frames before release,
      then find the argmin (release rightward) or argmax (release leftward) of
      wrist x in the search window.

      When foot_strike_frame is supplied and valid, the search is constrained to
      [foot_strike_frame, release_frame] to enforce correct phase ordering
      (max external rotation is always after foot strike and before release).
      Falls back to release_frame - lookback when foot_strike_frame is absent or
      would produce an empty window.

    Args:
        pose_df: Full pose DataFrame.
        handedness: "right" or "left".
        release_frame: Detected ball release frame.
        window: Delivery window (used for context; search ignores window[0]).
        foot_strike_frame: When provided, constrains the search to start here.
        lookback: Fallback frames before release to search when foot_strike_frame
                  is absent or invalid.

    Returns:
        Frame number of max layback.
    """
    wrist_idx = _WRIST_IDX[handedness]
    raw = get_landmark_series(pose_df, wrist_idx, "x")
    x = _interpolate(raw)
    x_smooth = smooth_series(pd.Series(x))

    # Release direction: sign of positional change in the 5 frames before release.
    # Using positional delta avoids needing fps (only the sign matters).
    offset = min(5, release_frame)
    release_direction = float(x_smooth[release_frame]) - float(x_smooth[release_frame - offset])

    # Constrain search to [foot_strike, release] when foot_strike is valid.
    if foot_strike_frame is not None and 0 <= foot_strike_frame < release_frame:
        search_start = foot_strike_frame
    else:
        search_start = max(0, release_frame - lookback)

    segment = x_smooth[search_start:release_frame]
    if len(segment) == 0:
        return search_start

    local_idx = int(np.argmin(segment)) if release_direction > 0 else int(np.argmax(segment))
    return search_start + local_idx


def detect_foot_strike(
    pose_df: pd.DataFrame,
    handedness: str,
    fps: float,
    release_frame: int,
    window: tuple,
    lookback: int = 70,
) -> int:
    """Detect foot strike (FP) — lead foot contacts the ground.

    Algorithm:
      During the stride the lead ankle falls (y increases; y=0 is top of frame).
      At foot strike the ankle abruptly stops. We track lead-ankle y-velocity:
        1. Find the peak falling velocity (max vy) in the lookback window.
        2. Walk forward: the first frame where vy transitions from above 20 %
           of peak (still falling) to below 10 % of peak (landed) is foot strike.
        3. Fallback: frame of minimum |vy| in the lookback window.

      Search lower bound is ``release_frame - lookback``, not ``window[0]``,
      because foot strike often precedes the wrist-velocity window start.

    Args:
        pose_df: Full pose DataFrame.
        handedness: "right" or "left".
        fps: Frames per second.
        release_frame: Detected ball release frame.
        window: Delivery window (used for context; window[0] not used as bound).
        lookback: Frames before release to search (70 ≈ 2.3 s at 30 fps).

    Returns:
        Frame number of foot strike.
    """
    ankle_idx = _LEAD_ANKLE_IDX[handedness]
    raw = get_landmark_series(pose_df, ankle_idx, "y")
    _check_nan_fraction(raw, f"{handedness}_lead_ankle_y")

    y = _interpolate(raw)
    y_smooth = smooth_series(pd.Series(y))
    vy = compute_velocity(y_smooth, fps)

    search_start = max(0, release_frame - lookback)
    vy_seg = vy[search_start:release_frame]

    if len(vy_seg) == 0:
        return release_frame

    peak_falling = float(np.max(vy_seg))
    if peak_falling <= 0.01:
        # Ankle never clearly fell; return frame of minimum |vy|
        return int(np.argmin(np.abs(vy_seg))) + search_start

    high_thresh  = 0.20 * peak_falling   # "still falling fast"
    still_thresh = 0.10 * peak_falling   # "essentially stopped"

    prev_falling = False
    for i, v in enumerate(vy_seg):
        if v > high_thresh:
            prev_falling = True
        if prev_falling and abs(v) < still_thresh:
            return search_start + i

    # No clean transition found
    return int(np.argmin(np.abs(vy_seg))) + search_start


def detect_leg_lift_peak(
    pose_df: pd.DataFrame,
    handedness: str,
    foot_strike_frame: int,
    window: tuple,
    lookback: int = 100,
) -> int:
    """Detect leg lift peak — lead ankle at maximum height (minimum y).

    Algorithm:
      Lead ankle y is minimized (closest to frame top) at peak leg lift.
      We search from ``foot_strike_frame - lookback`` to ``foot_strike_frame``
      and return argmin of the smoothed y-series.

      The 100-frame lookback (3.3 s at 30 fps) is generous enough to capture
      the leg lift even when it occurs well before the wrist-velocity window.

    Args:
        pose_df: Full pose DataFrame.
        handedness: "right" or "left".
        foot_strike_frame: Detected foot strike frame.
        window: Delivery window (kept for API symmetry; not used as bound here).
        lookback: Frames before foot strike to search.

    Returns:
        Frame number of leg lift peak.
    """
    ankle_idx = _LEAD_ANKLE_IDX[handedness]
    raw = get_landmark_series(pose_df, ankle_idx, "y")
    y = _interpolate(raw)
    y_smooth = smooth_series(pd.Series(y))

    search_start = max(0, foot_strike_frame - lookback)
    segment = y_smooth[search_start:foot_strike_frame]
    if len(segment) == 0:
        return search_start

    return int(np.argmin(segment)) + search_start


def detect_start_of_motion(
    pose_df: pd.DataFrame,
    leg_lift_frame: int,
    window: tuple,
    fps: float,
    lookback: int = 80,
) -> int:
    """Detect start of motion — first significant movement from setup.

    Algorithm:
      Computes total body motion as sum of |velocity| across hips and shoulders
      (landmarks 11, 12, 23, 24 in both x and y). Walks backward from the leg
      lift peak to find the latest frame where this aggregate motion drops below
      20 % of its peak for 5+ consecutive frames — that sustained-still block is
      the "setup". The frame immediately after it is start of motion.

      The 80-frame lookback ensures the pre-delivery stillness is captured even
      when the pitcher walks in and pauses briefly before starting.

    Args:
        pose_df: Full pose DataFrame.
        leg_lift_frame: Detected leg lift peak frame.
        window: Delivery window (used for context only).
        fps: Frames per second.
        lookback: Frames before leg lift to search (80 ≈ 2.7 s at 30 fps).

    Returns:
        Frame number of start of motion.
    """
    n_frames = int(pose_df.frame.max()) + 1
    total_motion = np.zeros(n_frames)

    for lm_idx in _HIP_SHOULDER_IDX:
        for coord in ("x", "y"):
            raw = get_landmark_series(pose_df, lm_idx, coord)
            arr = _interpolate(raw)
            arr_smooth = smooth_series(pd.Series(arr))
            vel = compute_velocity(arr_smooth, fps)
            total_motion += np.abs(vel)

    search_start = max(0, leg_lift_frame - lookback)
    segment = total_motion[search_start:leg_lift_frame]

    if len(segment) == 0:
        return search_start

    peak = float(np.max(segment))
    if peak == 0:
        return search_start

    threshold = 0.20 * peak
    n = len(segment)
    still = total_motion[search_start:leg_lift_frame] < threshold

    # Forward scan: find the LAST 5+ consecutive still block before leg lift.
    # "Start of motion" = the frame immediately after that block ends.
    # Using a forward scan avoids the initialization ambiguity of a backward walk.
    last_start_of_motion = search_start
    i = 0
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < n and still[j]:
            j += 1
        if j - i >= 5:
            last_start_of_motion = search_start + j  # first frame after the still block
        i = j if j > i else i + 1

    return last_start_of_motion


def detect_end_of_motion(
    pose_df: pd.DataFrame,
    handedness: str,
    release_frame: int,
    window: tuple,
    fps: float,
) -> int:
    """Detect end of motion — follow-through complete.

    Algorithm:
      Walks forward from release. Returns the first frame where throwing-wrist
      |velocity| drops below 15 % of its post-release peak and stays there for
      5+ consecutive frames. Falls back to window[1] if no quiet period is found.

    Args:
        pose_df: Full pose DataFrame.
        handedness: "right" or "left".
        release_frame: Detected ball release frame.
        window: (start_frame, end_frame) delivery window; window[1] is the
                upper search bound and fallback.
        fps: Frames per second.

    Returns:
        Frame number of end of motion.
    """
    wrist_idx = _WRIST_IDX[handedness]
    raw = get_landmark_series(pose_df, wrist_idx, "x")
    x = _interpolate(raw)
    x_smooth = smooth_series(pd.Series(x))
    vel = compute_velocity(x_smooth, fps)
    abs_vel = np.abs(vel)

    search_end = min(int(window[1]), len(abs_vel) - 1)
    seg = abs_vel[release_frame:search_end + 1]

    if len(seg) == 0:
        return int(window[1])

    peak_vel = float(np.max(seg))
    if peak_vel == 0:
        return release_frame

    threshold = 0.15 * peak_vel
    consecutive = 0
    last_active = release_frame

    for f in range(release_frame, search_end + 1):
        if abs_vel[f] < threshold:
            consecutive += 1
            if consecutive >= 5:
                return last_active
        else:
            consecutive = 0
            last_active = f

    return int(window[1])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_all_phases(
    pose_df: pd.DataFrame,
    handedness: str,
    fps: float = None,
) -> dict:
    """Detect all six key delivery phases and return as a dictionary.

    Calls detectors in dependency order:
      window → ball_release → max_layback → foot_strike
      → leg_lift_peak → start_of_motion → end_of_motion

    Args:
        pose_df: Full pose DataFrame from Phase 1 extraction.
        handedness: "right" or "left".
        fps: Frames per second. If None, estimated from pose_df timestamps.

    Returns:
        Dict with keys:
          window_start, window_peak, window_end,
          start_of_motion, leg_lift_peak, foot_strike,
          max_layback, ball_release, end_of_motion,
          and a ``*_timestamp_ms`` entry for each phase frame.
    """
    if fps is None:
        fps = 1000.0 / float(np.median(np.diff(pose_df.timestamp_ms.unique())))

    win_start, win_peak, win_end = detect_pitching_window(pose_df, handedness, fps)
    window = (win_start, win_end)

    ball_release     = detect_ball_release(pose_df, handedness, fps, window)
    # foot_strike computed before max_layback so the ordering guard can be applied
    foot_strike      = detect_foot_strike(pose_df, handedness, fps, ball_release, window)
    max_ext_rot      = detect_max_layback(pose_df, handedness, ball_release, window, foot_strike_frame=foot_strike)
    leg_lift_peak    = detect_leg_lift_peak(pose_df, handedness, foot_strike, window)
    start_of_motion  = detect_start_of_motion(pose_df, leg_lift_peak, window, fps)
    end_of_motion    = detect_end_of_motion(pose_df, handedness, ball_release, window, fps)

    ll_confident,  ll_notes  = _leg_lift_confidence(pose_df, handedness, foot_strike, leg_lift_peak)
    som_confident, som_notes = _start_of_motion_confidence(pose_df, start_of_motion, leg_lift_peak, fps)
    confidence_notes = ll_notes + som_notes

    ts_map = (
        pose_df.drop_duplicates("frame")
        .set_index("frame")["timestamp_ms"]
        .to_dict()
    )

    def ts(frame: int) -> float:
        return float(ts_map.get(frame, frame / fps * 1000.0))

    return {
        "window_start":                     win_start,
        "window_peak":                      win_peak,
        "window_end":                       win_end,
        "start_of_motion":                  start_of_motion,
        "leg_lift_peak":                    leg_lift_peak,
        "foot_strike":                      foot_strike,
        "max_layback":                      max_ext_rot,
        "ball_release":                     ball_release,
        "end_of_motion":                    end_of_motion,
        "window_start_timestamp_ms":        ts(win_start),
        "window_end_timestamp_ms":          ts(win_end),
        "start_of_motion_timestamp_ms":     ts(start_of_motion),
        "leg_lift_peak_timestamp_ms":       ts(leg_lift_peak),
        "foot_strike_timestamp_ms":         ts(foot_strike),
        "max_layback_timestamp_ms":         ts(max_ext_rot),
        "ball_release_timestamp_ms":        ts(ball_release),
        "end_of_motion_timestamp_ms":       ts(end_of_motion),
        # ── confidence flags (additive; do not change existing keys) ──────
        "leg_lift_confident":               ll_confident,
        "start_of_motion_confident":        som_confident,
        "confidence_notes":                 confidence_notes,
    }
