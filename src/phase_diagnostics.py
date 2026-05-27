"""
Diagnostic signal plots for Phase 3 phase detection verification.

Produces a 4-subplot figure showing the raw kinematic signals each
detector operates on, with detected phase frames overlaid as vertical
lines. Use this to verify that detected frames sit on the correct
signal features before trusting any downstream metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from phase_detection import (
    get_landmark_series,
    smooth_series,
    compute_velocity,
    _interpolate,
    _WRIST_IDX,
    _LEAD_ANKLE_IDX,
)


def plot_phase_signals(
    pose_df: pd.DataFrame,
    phases: dict,
    handedness: str,
    fps: float,
    output_path: str,
) -> None:
    """Produce a 4-panel diagnostic plot of the kinematic signals used by each detector.

    Panels (shared x-axis in frame numbers):
      1. Throwing wrist x — smoothed position, release (red) and MER (orange) marked
      2. Throwing wrist x-velocity — release marked, ±15 % window threshold dashed
      3. Lead ankle y — smoothed position, leg lift (blue) and foot strike (green) marked
         (y-axis inverted so leg lift peak reads as a visual peak)
      4. Lead ankle y-velocity — foot strike marked, zero-line dashed

    The detected pitching window is shaded gray in all panels.

    Args:
        pose_df: Full pose DataFrame from Phase 1.
        phases: Dict returned by detect_all_phases.
        handedness: "right" or "left".
        fps: Frames per second (used for velocity scaling).
        output_path: Where to save the PNG file.
    """
    wrist_idx = _WRIST_IDX[handedness]
    ankle_idx = _LEAD_ANKLE_IDX[handedness]

    # --- build signals ---
    wrist_x_raw   = get_landmark_series(pose_df, wrist_idx, "x")
    wrist_x       = smooth_series(pd.Series(_interpolate(wrist_x_raw)))
    wrist_vel      = compute_velocity(wrist_x, fps)

    ankle_y_raw   = get_landmark_series(pose_df, ankle_idx, "y")
    ankle_y       = smooth_series(pd.Series(_interpolate(ankle_y_raw)))
    ankle_vel      = compute_velocity(ankle_y, fps)

    frames = np.arange(len(wrist_x))

    # --- x-axis window: span all detected phases with padding ---
    all_frames = [
        phases["window_start"], phases["window_end"],
        phases["start_of_motion"], phases["leg_lift_peak"],
        phases["foot_strike"], phases["max_external_rotation"],
        phases["ball_release"], phases["end_of_motion"],
    ]
    x_lo = max(0, min(all_frames) - 25)
    x_hi = min(len(frames) - 1, max(all_frames) + 25)

    # Wrist velocity window threshold lines (±15 % of peak |velocity| in full signal)
    abs_vel = np.abs(wrist_vel)
    peak_vel = float(abs_vel[phases["window_start"]:phases["window_end"] + 1].max())
    vel_threshold = 0.15 * peak_vel

    win_s = phases["window_start"]
    win_e = phases["window_end"]

    # --- phase markers ---
    release_frame = phases["ball_release"]
    mer_frame     = phases["max_external_rotation"]
    fs_frame      = phases["foot_strike"]
    ll_frame      = phases["leg_lift_peak"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Phase Detection Diagnostic  |  {handedness.capitalize()}-handed  |  "
        f"fps={fps:.0f}  |  window frames {win_s}-{win_e}",
        fontsize=12, fontweight="bold",
    )

    _WINDOW_ALPHA = 0.10
    _WINDOW_COLOR = "gray"

    def _shade_window(ax):
        ax.axvspan(win_s, win_e, color=_WINDOW_COLOR, alpha=_WINDOW_ALPHA,
                   label="Pitching window")

    def _vline(ax, frame, color, label, ls="-"):
        ax.axvline(frame, color=color, lw=1.6, ls=ls, label=f"{label} (f={frame})")

    # ------------------------------------------------------------------ #
    # Subplot 1: wrist x position
    # ------------------------------------------------------------------ #
    ax = axes[0]
    _shade_window(ax)
    ax.plot(frames, wrist_x, color="steelblue", lw=1.4, label="Wrist x (smoothed)")
    _vline(ax, release_frame, "red",    "Ball release")
    _vline(ax, mer_frame,     "darkorange", "Max ext rotation", ls="--")
    ax.set_ylabel("Normalized x (0–1)")
    ax.set_title("Throwing wrist x (smoothed)  —  release & MER")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------ #
    # Subplot 2: wrist x velocity
    # ------------------------------------------------------------------ #
    ax = axes[1]
    _shade_window(ax)
    ax.plot(frames, wrist_vel, color="steelblue", lw=1.4, label="Wrist x-vel (px/s)")
    _vline(ax, release_frame, "red", "Ball release")
    ax.axhline( vel_threshold, color="dimgray", lw=1.0, ls=":", label=f"+15% threshold ({vel_threshold:.2f})")
    ax.axhline(-vel_threshold, color="dimgray", lw=1.0, ls=":")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_ylabel("Velocity (units/s)")
    ax.set_title("Throwing wrist x-velocity  —  release & window threshold (dashed)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------ #
    # Subplot 3: lead ankle y position  (inverted so leg-lift is a peak)
    # ------------------------------------------------------------------ #
    ax = axes[2]
    _shade_window(ax)
    ax.plot(frames, ankle_y, color="mediumseagreen", lw=1.4, label="Lead ankle y (smoothed)")
    _vline(ax, ll_frame, "royalblue",    "Leg lift peak")
    _vline(ax, fs_frame, "forestgreen",  "Foot strike",  ls="--")
    ax.invert_yaxis()   # y=0 is top of frame; invert so higher foot = visual peak
    ax.set_ylabel("Normalized y (inverted: up = high foot)")
    ax.set_title("Lead ankle y (smoothed, y-inverted)  —  leg lift peak & foot strike")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------ #
    # Subplot 4: lead ankle y velocity
    # ------------------------------------------------------------------ #
    ax = axes[3]
    _shade_window(ax)
    ax.plot(frames, ankle_vel, color="mediumseagreen", lw=1.4, label="Lead ankle y-vel")
    _vline(ax, fs_frame, "forestgreen", "Foot strike", ls="--")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5, ls="--", label="zero")
    ax.set_ylabel("Velocity (units/s)")
    ax.set_xlabel("Frame number")
    ax.set_title("Lead ankle y-velocity  —  foot strike  (positive = ankle falling)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------ #
    # Shared x-axis range
    # ------------------------------------------------------------------ #
    for ax in axes:
        ax.set_xlim(x_lo, x_hi)

    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagnostic plot saved to: {out}")
