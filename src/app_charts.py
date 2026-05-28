"""
Matplotlib chart functions for the Streamlit app.

Each function returns a matplotlib Figure ready for st.pyplot(fig).
Decorated with @st.cache_data so charts are not re-rendered on every
Streamlit rerun — they are regenerated only when the underlying data changes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_BG       = "#0F172A"
_FG       = "#F1F5F9"
_SPINE    = "#94A3B8"
_LINE     = "#3B82F6"
_ACCENT   = "#EF4444"
_PHASE_LN = "#64748B"
_CONN     = "#475569"

# Session B color for overlay charts — amber, distinct from blue and from green/red
# classification semantics used in the comparison table.
SESSION_B_COLOR = "#F59E0B"


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE)
    ax.spines["bottom"].set_color(_SPINE)
    ax.tick_params(colors=_FG)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


def _phase_vline(ax: plt.Axes, t: float, label: str) -> None:
    """Draw a labeled vertical dashed line at time t."""
    ax.axvline(t, color=_PHASE_LN, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        t, 0.98, label,
        transform=ax.get_xaxis_transform(),
        ha="center", va="top",
        fontsize=9, color=_SPINE,
    )


# ── Single-session charts (used by app.py) ─────────────────────────────────

@st.cache_data
def plot_hip_shoulder_separation(
    series_df: pd.DataFrame,
    phases: dict,
    peak_frame: int,
    peak_value: float,
    fps: float,
) -> plt.Figure:
    """Line chart of hip-shoulder separation (smoothed) over the coiling window."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    xs = series_df["timestamp_s"].to_numpy()
    ys = series_df["separation_deg"].to_numpy()

    ax.plot(xs, ys, color=_LINE, linewidth=2)

    # Phase markers at window boundaries
    for key, label in [("leg_lift_peak", "Leg lift"), ("foot_strike", "Foot strike")]:
        if key in phases:
            _phase_vline(ax, phases[key] / fps, label)

    # Peak annotation
    peak_ts = peak_frame / fps
    y_range = float(ys.max() - ys.min()) if len(ys) > 1 else 1.0
    offset  = max(2.0, y_range * 0.25)
    ax.scatter([peak_ts], [peak_value], color=_ACCENT, s=60, zorder=5)
    ax.annotate(
        f"Peak: {peak_value:.1f} deg",
        xy=(peak_ts, peak_value),
        xytext=(peak_ts, peak_value + offset),
        ha="center", va="bottom",
        fontsize=10, color=_ACCENT,
        arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.2),
    )
    # Give the annotation text room above the peak
    ax.set_ylim(
        ys.min() - offset * 0.4,
        ys.max() + offset * 2.0,
    )

    ax.set_title("Hip-shoulder separation over time", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Separation (degrees)", fontsize=11)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


@st.cache_data
def plot_front_knee_angle(
    series_df: pd.DataFrame,
    phases: dict,
    release_frame: int,
    release_value: float,
    fps: float,
) -> plt.Figure:
    """Line chart of front knee angle from foot strike through end of motion."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    xs = series_df["timestamp_s"].to_numpy()
    ys = series_df["knee_angle_deg"].to_numpy()

    ax.plot(xs, ys, color=_LINE, linewidth=2)

    # Phase markers
    for key, label in [("foot_strike", "Foot strike"), ("ball_release", "Ball release")]:
        if key in phases:
            _phase_vline(ax, phases[key] / fps, label)

    # Release annotation
    release_ts = release_frame / fps
    y_range = float(ys.max() - ys.min()) if len(ys) > 1 else 1.0
    offset  = max(3.0, y_range * 0.20)
    ax.scatter([release_ts], [release_value], color=_ACCENT, s=60, zorder=5)
    ax.annotate(
        f"Release: {release_value:.1f} deg",
        xy=(release_ts, release_value),
        xytext=(release_ts, release_value + offset),
        ha="center", va="bottom",
        fontsize=10, color=_ACCENT,
        arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.2),
    )
    ax.set_ylim(
        ys.min() - offset * 0.4,
        ys.max() + offset * 2.0,
    )

    ax.set_title("Front knee angle over time", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Knee angle (degrees)", fontsize=11)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


@st.cache_data
def plot_head_trace(
    pose_df: pd.DataFrame,
    phases: dict,
    video_metadata: dict,
) -> plt.Figure:
    """2D scatter/line of nose position from leg lift to ball release, colored by frame."""
    ll_frame = int(phases["leg_lift_peak"])
    br_frame = int(phases["ball_release"])
    w = int(video_metadata["width"])
    h = int(video_metadata["height"])

    mask = (
        (pose_df["landmark_idx"] == 0) &
        (pose_df["frame"] >= ll_frame) &
        (pose_df["frame"] <= br_frame) &
        (pose_df["visibility"] >= 0.5)
    )
    sub = pose_df[mask].sort_values("frame")

    xs     = (sub["x"] * w).to_numpy()
    ys     = (sub["y"] * h).to_numpy()
    frames = sub["frame"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Light connector line underneath scatter
    ax.plot(xs, ys, color=_CONN, linewidth=0.8, zorder=1)

    # Scatter colored by frame number
    sc = ax.scatter(xs, ys, c=frames, cmap="viridis", s=25, zorder=2)
    cb = fig.colorbar(sc, ax=ax, label="Frame")
    cb.ax.tick_params(labelcolor=_FG, color=_FG)
    cb.outline.set_edgecolor(_SPINE)
    cb.set_label("Frame", color=_FG)

    # Image coords: y=0 is top of frame, invert so "up" is up
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_title("Head position trace", fontsize=12)
    ax.set_xlabel("X position (pixels)", fontsize=11)
    ax.set_ylabel("Y position (pixels)", fontsize=11)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


# ── Overlay / comparison helpers ───────────────────────────────────────────

def _norm_pct(frame: int, start: int, end: int) -> float:
    """Map a frame index to 0–100% of the delivery window [start, end]."""
    return (frame - start) / max(end - start, 1) * 100


def _overlay_phase_lines(
    ax: plt.Axes,
    phases: dict,
    start: int,
    end: int,
    color: str,
    y_label: float,
) -> None:
    """Colored dashed vlines for foot_strike and ball_release at the given label height.

    Two sessions use different y_label values (e.g. 0.95 vs 0.82) so their
    abbreviations don't collide when the phase positions differ.
    """
    for key, abbrev in [("foot_strike", "FS"), ("ball_release", "BR")]:
        if key not in phases:
            continue
        pct = _norm_pct(int(phases[key]), start, end)
        ax.axvline(pct, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(
            pct, y_label, abbrev,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=8, color=color, alpha=0.9,
        )


# ── Overlay charts (used by pages/2_Compare_Sessions.py) ──────────────────

@st.cache_data
def overlay_hip_shoulder_separation(
    ts_a: dict,
    ts_b: dict,
    phases_a: dict,
    phases_b: dict,
) -> plt.Figure:
    """Overlay hip-shoulder separation for two sessions on a shared 0–100% delivery window.

    Each session's frames are normalized to its own start_of_motion → end_of_motion
    span so deliveries of different durations can be compared by shape.
    Session A = blue (_LINE), Session B = amber (SESSION_B_COLOR).
    Dashed lines mark foot strike (FS) and ball release (BR) for each session.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    key = "hip_shoulder_separation_max"
    for ts, phases, color, session_label, y_label in [
        (ts_a, phases_a, _LINE,          "Session A", 0.95),
        (ts_b, phases_b, SESSION_B_COLOR, "Session B", 0.82),
    ]:
        if key not in ts:
            continue
        df    = ts[key]
        start = int(phases.get("start_of_motion", int(df["frame"].min())))
        end   = int(phases.get("end_of_motion",   int(df["frame"].max())))
        xs    = np.array([_norm_pct(int(f), start, end) for f in df["frame"]])
        ys    = df["separation_deg"].to_numpy()
        ax.plot(xs, ys, color=color, linewidth=2, label=session_label)
        _overlay_phase_lines(ax, phases, start, end, color, y_label)

    ax.set_xlim(0, 100)
    ax.set_title("Hip-shoulder separation over the delivery", fontsize=12)
    ax.set_xlabel("Delivery progression (%)", fontsize=11)
    ax.set_ylabel("Separation (degrees)", fontsize=11)
    ax.legend(facecolor=_BG, edgecolor=_SPINE, labelcolor=_FG, fontsize=10)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


@st.cache_data
def overlay_front_knee_angle(
    ts_a: dict,
    ts_b: dict,
    phases_a: dict,
    phases_b: dict,
) -> plt.Figure:
    """Overlay front knee angle for two sessions on a shared 0–100% delivery window.

    Each session's frames are normalized to its own start_of_motion → end_of_motion span.
    Session A = blue (_LINE), Session B = amber (SESSION_B_COLOR).
    Dashed lines mark foot strike (FS) and ball release (BR) for each session.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    key = "front_knee_flex"
    for ts, phases, color, session_label, y_label in [
        (ts_a, phases_a, _LINE,          "Session A", 0.95),
        (ts_b, phases_b, SESSION_B_COLOR, "Session B", 0.82),
    ]:
        if key not in ts:
            continue
        df    = ts[key]
        start = int(phases.get("start_of_motion", int(df["frame"].min())))
        end   = int(phases.get("end_of_motion",   int(df["frame"].max())))
        xs    = np.array([_norm_pct(int(f), start, end) for f in df["frame"]])
        ys    = df["knee_angle_deg"].to_numpy()
        ax.plot(xs, ys, color=color, linewidth=2, label=session_label)
        _overlay_phase_lines(ax, phases, start, end, color, y_label)

    ax.set_xlim(0, 100)
    ax.set_title("Front knee angle over the delivery", fontsize=12)
    ax.set_xlabel("Delivery progression (%)", fontsize=11)
    ax.set_ylabel("Knee angle (degrees)", fontsize=11)
    ax.legend(facecolor=_BG, edgecolor=_SPINE, labelcolor=_FG, fontsize=10)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


@st.cache_data
def overlay_head_trace(
    pose_df_a: pd.DataFrame,
    pose_df_b: pd.DataFrame,
    phases_a: dict,
    phases_b: dict,
    vm_a: dict,
    vm_b: dict,
) -> plt.Figure:
    """Overlay nose-position traces for two sessions, each translated to their own
    leg_lift_peak origin so paths are spatially comparable.

    Note: absolute spatial alignment across different videos is approximate —
    pixel-scale differences between sessions (e.g. different resolutions or camera
    distances) are not corrected.
    Session A = blue (_LINE), Session B = amber (SESSION_B_COLOR).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    def _nose_path(
        pose_df: pd.DataFrame,
        phases: dict,
        vm: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        ll_frame = int(phases["leg_lift_peak"])
        br_frame = int(phases["ball_release"])
        w = int(vm["width"])
        h = int(vm["height"])
        mask = (
            (pose_df["landmark_idx"] == 0) &
            (pose_df["frame"] >= ll_frame) &
            (pose_df["frame"] <= br_frame) &
            (pose_df["visibility"] >= 0.5)
        )
        sub = pose_df[mask].sort_values("frame")
        return (sub["x"] * w).to_numpy(), (sub["y"] * h).to_numpy()

    for pose_df, phases, vm, color, session_label in [
        (pose_df_a, phases_a, vm_a, _LINE,          "Session A"),
        (pose_df_b, phases_b, vm_b, SESSION_B_COLOR, "Session B"),
    ]:
        xs, ys = _nose_path(pose_df, phases, vm)
        if len(xs) == 0:
            continue
        # Translate so leg_lift_peak nose position is at the origin
        xs = xs - xs[0]
        ys = ys - ys[0]
        # label only the line so the legend shows one entry per session
        ax.plot(xs, ys, color=color, linewidth=0.8, zorder=1, alpha=0.6, label=session_label)
        ax.scatter(xs, ys, color=color, s=20, zorder=2, alpha=0.8)

    # Image coords: y=0 at frame top; invert so upward head movement plots upward
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Head position trace (translated to origin)", fontsize=12)
    ax.set_xlabel("X displacement (pixels)", fontsize=11)
    ax.set_ylabel("Y displacement (pixels)", fontsize=11)
    ax.legend(facecolor=_BG, edgecolor=_SPINE, labelcolor=_FG, fontsize=10)
    _clean_axes(ax)
    fig.tight_layout()
    return fig
