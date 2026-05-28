"""
Pitch Analyzer -- Streamlit entry point.

Run with:
    streamlit run src/app.py
"""

import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app_pipeline import run_full_pipeline, PipelineResult
from app_utils import (
    PHASE_DISPLAY_NAMES,
    METRIC_DISPLAY_ORDER,
    PIPELINE_ERROR_HINT,
    format_metric_value,
    build_export_payload,
    slugify_for_filename,
)
from app_charts import (
    plot_hip_shoulder_separation,
    plot_front_knee_angle,
    plot_head_trace,
)

st.set_page_config(page_title="Pitch Analyzer", layout="wide")
st.title("Pitch Analyzer")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    handedness = st.radio(
        "Pitcher handedness",
        options=["right", "left"],
        index=0,
        format_func=str.capitalize,
    )
    uploaded = st.file_uploader(
        "Upload pitch video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Single-pitch clip. Side-angle view recommended.",
    )
    athlete_name = st.text_input("Athlete name (optional)", value="")
    pitch_type = st.selectbox(
        "Pitch type",
        ["Fastball", "Curveball", "Slider", "Changeup", "Cutter", "Sinker", "Splitter", "Other"],
        index=0,
    )
    session_date = st.date_input("Session date", value=datetime.date.today())

if uploaded is None:
    st.info("Upload a pitch video in the sidebar to begin.")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────
video_bytes = uploaded.read()
try:
    result: PipelineResult = run_full_pipeline(video_bytes, handedness)
except ValueError as exc:
    st.error(f"Could not analyze this video.\n\n{exc}\n\n{PIPELINE_ERROR_HINT}")
    st.stop()

# ── Metadata header ────────────────────────────────────────────────────────
if athlete_name:
    st.markdown(f"## {athlete_name}")
st.caption(f"{pitch_type} | {session_date.strftime('%B %d, %Y')}")

# ── Summary banner ─────────────────────────────────────────────────────────
metrics = result.metrics
n_total   = len(metrics)
n_success = sum(1 for m in metrics.values() if m.error is None)
n_failed  = n_total - n_success

if n_failed == 0:
    st.success(f"All {n_total} metrics computed successfully.")
else:
    st.warning(f"{n_success} of {n_total} metrics computed -- {n_failed} failed.")

# ── Phase confidence warning ───────────────────────────────────────────────
# Maps each confidence flag to a human label and the metrics it affects.
_PHASE_CONFIDENCE_META: dict[str, dict] = {
    "leg_lift_confident": {
        "label":   "leg lift",
        "affects": "balance point, head movement, hip–shoulder separation, leg-lift tempo",
    },
    "start_of_motion_confident": {
        "label":   "start of motion",
        "affects": "stride length, balance point, start-to-release tempo",
    },
}

_conf_notes = result.phases.get("confidence_notes", [])
if _conf_notes:
    _affected_parts = [
        f"{meta['label']} (affects: {meta['affects']})"
        for flag, meta in _PHASE_CONFIDENCE_META.items()
        if not result.phases.get(flag, True)
    ]
    _note_lines = "\n".join(f"• {n}" for n in _conf_notes)
    _affected_str = "; ".join(_affected_parts)
    st.warning(
        f"Some delivery phases could not be confidently detected: {_affected_str}. "
        f"Reported values are best-guess estimates and may be unreliable.\n\n"
        f"{_note_lines}"
    )

# ── Main two-column layout ─────────────────────────────────────────────────
col_video, col_metrics = st.columns([1, 1], gap="large")

# --- Video panel ---
with col_video:
    st.subheader("Pose analysis")
    st.video(result.annotated_video_bytes)

    vm = result.video_metadata
    st.caption(
        f"{vm['width']}x{vm['height']} "
        f"| {vm['fps']:.0f} fps "
        f"| {vm['duration_s']:.1f}s "
        f"| {vm['frame_count']} frames"
    )

# --- Metrics panel ---
with col_metrics:
    st.subheader("Metrics")

    ordered_keys = [k for k in METRIC_DISPLAY_ORDER if k in metrics]
    remaining    = [k for k in metrics if k not in METRIC_DISPLAY_ORDER]
    display_keys = ordered_keys + remaining
    # Stable sort: failed metrics sink to the bottom, order within each group preserved.
    display_keys.sort(key=lambda k: 1 if metrics[k].error is not None else 0)

    rows = []
    for key in display_keys:
        mr = metrics[key]
        phase_label = PHASE_DISPLAY_NAMES.get(mr.phase, "--") if mr.phase else "--"
        description = (f"Error: {mr.error}") if mr.error else mr.description
        rows.append({
            "Metric":      mr.display_name,
            "Value":       format_metric_value(mr),
            "Phase":       phase_label,
            "Description": description,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Metric":      st.column_config.TextColumn("Metric",      width="medium"),
            "Value":       st.column_config.TextColumn("Value",       width="small"),
            "Phase":       st.column_config.TextColumn("Phase",       width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )

# ── Phases expander ────────────────────────────────────────────────────────
_PHASE_KEYS = [
    "start_of_motion",
    "leg_lift_peak",
    "foot_strike",
    "max_layback",
    "ball_release",
    "end_of_motion",
]

with st.expander("Detected phases", expanded=False):
    fps = result.video_metadata["fps"]
    phase_rows = []
    for key in _PHASE_KEYS:
        frame = result.phases.get(key)
        if frame is None:
            continue
        phase_rows.append({
            "Phase":    PHASE_DISPLAY_NAMES.get(key, key),
            "Frame":    frame,
            "Time (s)": round(frame / fps, 3),
        })
    phase_rows.sort(key=lambda r: r["Frame"])
    st.dataframe(
        pd.DataFrame(phase_rows),
        hide_index=True,
        use_container_width=True,
    )

# ── Time-series analysis ───────────────────────────────────────────────────
st.subheader("Time-series analysis")

fps    = result.video_metadata["fps"]
ts     = result.time_series
phases = result.phases

# --- Hip-shoulder separation ---
if "hip_shoulder_separation_max" in ts:
    try:
        hss_mr = metrics["hip_shoulder_separation_max"]
        fig = plot_hip_shoulder_separation(
            series_df  = ts["hip_shoulder_separation_max"],
            phases     = phases,
            peak_frame = int(hss_mr.frame),
            peak_value = float(hss_mr.value),
            fps        = fps,
        )
        st.pyplot(fig, clear_figure=False)
    except Exception as exc:
        st.info(f"Hip-shoulder separation chart unavailable: {exc}")
else:
    st.info("Hip-shoulder separation chart unavailable -- metric failed to compute.")

# --- Front knee angle ---
if "front_knee_flex" in ts:
    try:
        fkf_mr = metrics["front_knee_flex"]
        fig = plot_front_knee_angle(
            series_df     = ts["front_knee_flex"],
            phases        = phases,
            release_frame = int(fkf_mr.frame),
            release_value = float(fkf_mr.value),
            fps           = fps,
        )
        st.pyplot(fig, clear_figure=False)
        st.caption(
            "Lower angle = more knee flexion. "
            "Elite pitchers tend to extend (increase angle) into release."
        )
    except Exception as exc:
        st.info(f"Front knee chart unavailable: {exc}")
else:
    st.info("Front knee chart unavailable -- metric failed to compute.")

# --- Head position trace ---
try:
    fig = plot_head_trace(
        pose_df        = result.pose_df,
        phases         = phases,
        video_metadata = result.video_metadata,
    )
    st.pyplot(fig, clear_figure=False)
    st.caption("Tighter clustering indicates better posture stability through the delivery.")
except Exception as exc:
    st.info(f"Head trace chart unavailable: {exc}")

# ── Export ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Export")

try:
    payload = build_export_payload(
        pipeline_result = result,
        athlete_name    = athlete_name,
        pitch_type      = pitch_type,
        session_date    = session_date,
        handedness      = handedness,
        video_filename  = uploaded.name,
    )
    json_str = json.dumps(payload, indent=2)

    athlete_slug = slugify_for_filename(athlete_name)
    prefix       = athlete_slug if athlete_slug else "session"
    date_iso     = session_date.isoformat()
    video_stem   = Path(uploaded.name).stem
    filename     = f"{prefix}_{date_iso}_{video_stem}.json"

    st.download_button(
        label     = "Download session JSON",
        data      = json_str,
        file_name = filename,
        mime      = "application/json",
    )
except Exception as exc:
    st.error(f"Failed to build export payload: {exc}")

# ── About / Known limitations ──────────────────────────────────────────────
with st.expander("About this tool & known limitations", expanded=False):
    st.markdown("""
**About**

Pitch Analyzer uses MediaPipe pose estimation to extract 33-point skeletal landmarks
from a single-camera pitch video, then computes 13 pitching mechanics metrics across
six key delivery phases (start of motion through end of motion).

**Known limitations**

- **2D measurements only.** All metrics are derived from a single camera view.
  Rotational quantities (arm slot, trunk tilt, hip-shoulder separation) lose their
  depth component and underestimate true 3D values when the pitcher is not perfectly
  perpendicular to the camera.
- **Single side-view camera required.** The pitcher must be the dominant subject in
  frame, fully visible throughout the delivery, filmed from a side angle. Partial
  cropping or camera movement degrades detection.
- **Ball release timing is approximate.** Release is detected as peak wrist velocity,
  which may lag the true release by 1-2 frames due to finger drag and wrist snap.
- **Single pitch per clip.** The detector selects the highest-velocity wrist burst.
  Multi-pitch videos or bullpen sessions will produce unreliable results — clip to
  a single delivery before uploading.
""")
