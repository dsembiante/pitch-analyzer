"""
Compare Sessions -- Phase 6, Batch 4.

Navigate here via the Streamlit sidebar when running:
    streamlit run src/app.py
"""

import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# src/ siblings are not automatically on sys.path when loaded as a page.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app_pipeline import run_full_pipeline
from app_comparison import compute_comparison
from app_comparison_utils import (
    build_comparison_table_rows,
    style_comparison_dataframe,
    build_comparison_export_payload,
    direction_label,
    direction_tooltip,
)
from app_charts import (
    overlay_hip_shoulder_separation,
    overlay_front_knee_angle,
    overlay_head_trace,
)
from app_utils import slugify_for_filename, PIPELINE_ERROR_HINT

st.set_page_config(page_title="Compare Sessions", layout="wide")
st.title("Compare two sessions")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    handedness = st.radio(
        "Pitcher handedness",
        options=["right", "left"],
        index=0,
        format_func=str.capitalize,
    )
    st.subheader("Session A")
    uploaded_a = st.file_uploader(
        "Upload Session A video",
        type=["mp4", "mov", "avi", "mkv"],
        key="upload_a",
    )
    st.subheader("Session B")
    uploaded_b = st.file_uploader(
        "Upload Session B video",
        type=["mp4", "mov", "avi", "mkv"],
        key="upload_b",
    )
    athlete_name = st.text_input("Athlete name (optional)", value="")

# ── Empty / partial upload states ─────────────────────────────────────────
if uploaded_a is None and uploaded_b is None:
    st.info("Upload both Session A and Session B to begin.")
    st.stop()
elif uploaded_a is None:
    st.info("Session B uploaded — add Session A to compare.")
    st.stop()
elif uploaded_b is None:
    st.info("Session A uploaded — add Session B to compare.")
    st.stop()

if athlete_name:
    st.markdown(f"## {athlete_name}")

# ── Run pipelines ───────────────────────────────────────────────────────────
result_a = None
result_b = None

try:
    with st.spinner("Analyzing Session A..."):
        result_a = run_full_pipeline(uploaded_a.read(), handedness)
except ValueError as exc:
    st.error(f"Could not analyze Session A.\n\n{exc}\n\n{PIPELINE_ERROR_HINT}")

try:
    with st.spinner("Analyzing Session B..."):
        result_b = run_full_pipeline(uploaded_b.read(), handedness)
except ValueError as exc:
    st.error(f"Could not analyze Session B.\n\n{exc}\n\n{PIPELINE_ERROR_HINT}")

if result_a is None or result_b is None:
    st.stop()

# ── Comparison ─────────────────────────────────────────────────────────────
comparison = compute_comparison(result_a, result_b)

# ── Summary banner ─────────────────────────────────────────────────────────
st.subheader("Summary")
s = comparison.summary
_PILLS = [
    ("improved",     "🟢", "improved"),
    ("regressed",    "🔴", "regressed"),
    ("notable",      "🟡", "notable changes"),
    ("unchanged",    "⚪", "unchanged"),
    ("incomparable", "⚠",  "incomparable"),
]
parts = [f"{icon} {s[key]} {label}" for key, icon, label in _PILLS if s.get(key, 0) > 0]
st.markdown("  ".join(parts))
st.caption(
    "Improvement direction is metric-specific (e.g., more hip-shoulder separation is "
    "better; less head movement is better). Neutral metrics are flagged amber when "
    "they change by more than 10% between sessions."
)

# ── Metric-by-metric table ──────────────────────────────────────────────────
st.subheader("Metric-by-metric")
rows = build_comparison_table_rows(comparison)
classifications = [c.classification for c in comparison.comparisons]
styler = style_comparison_dataframe(rows, classifications)
st.dataframe(styler, hide_index=True, use_container_width=True)

# ── Color key expander ─────────────────────────────────────────────────────
with st.expander("What do the colors mean?", expanded=False):
    directional = [
        (cmp.name, cmp.display_name, cmp.direction)
        for cmp in comparison.comparisons
        if cmp.direction != "neutral"
    ]
    for name, display_name, direction in directional:
        st.markdown(
            f"**{display_name}** — *{direction_label(direction)}.*  \n"
            f"{direction_tooltip(name, direction)}"
        )
    st.markdown(
        "**Amber** — notable change in a neutral metric. Some metrics (trunk tilt, "
        "arm slot, tempo, balance point drift, etc.) have no universally agreed-upon "
        "better direction, so we don't classify them as improved or regressed. But "
        "when they change substantially between sessions (more than 10%), we flag them "
        "in amber to draw attention. A coach would still want to interpret whether the "
        "change is meaningful for the specific pitcher and context."
    )
    st.caption("Grey = incomparable (value missing or errored on one side).")

# ── Incomparable warning ───────────────────────────────────────────────────
incomparable = [
    c.display_name for c in comparison.comparisons
    if c.classification == "incomparable"
]
if incomparable:
    names = ", ".join(incomparable)
    st.warning(
        f"{len(incomparable)} metric(s) could not be compared because a value was "
        f"missing or errored on one or both sides: {names}. "
        "These rows show no delta or percent change."
    )

# ── Side by side ────────────────────────────────────────────────────────────
st.subheader("Side by side")
col_a, col_b = st.columns(2)

with col_a:
    st.caption("Session A")
    st.video(result_a.annotated_video_bytes)
    vm_a = result_a.video_metadata
    st.caption(
        f"{vm_a['width']}x{vm_a['height']} "
        f"| {vm_a['fps']:.0f} fps "
        f"| {vm_a['duration_s']:.1f}s "
        f"| {vm_a['frame_count']} frames"
    )

with col_b:
    st.caption("Session B")
    st.video(result_b.annotated_video_bytes)
    vm_b = result_b.video_metadata
    st.caption(
        f"{vm_b['width']}x{vm_b['height']} "
        f"| {vm_b['fps']:.0f} fps "
        f"| {vm_b['duration_s']:.1f}s "
        f"| {vm_b['frame_count']} frames"
    )

# ── Mechanics overlay charts ────────────────────────────────────────────────
st.subheader("Mechanics over the delivery")

ts_a     = result_a.time_series
ts_b     = result_b.time_series
phases_a = result_a.phases
phases_b = result_b.phases

if "hip_shoulder_separation_max" in ts_a or "hip_shoulder_separation_max" in ts_b:
    try:
        fig = overlay_hip_shoulder_separation(ts_a, ts_b, phases_a, phases_b)
        st.pyplot(fig, clear_figure=False)
        st.caption(
            "Hip-shoulder separation across the delivery window. "
            "Session A = blue, Session B = amber. "
            "Dashed lines mark foot strike (FS) and ball release (BR) for each session."
        )
    except Exception as exc:
        st.info(f"Hip-shoulder separation overlay unavailable: {exc}")
else:
    st.info("Hip-shoulder separation overlay unavailable — metric did not compute for either session.")

if "front_knee_flex" in ts_a or "front_knee_flex" in ts_b:
    try:
        fig = overlay_front_knee_angle(ts_a, ts_b, phases_a, phases_b)
        st.pyplot(fig, clear_figure=False)
        st.caption(
            "Front knee angle from foot strike through end of motion. "
            "Session A = blue, Session B = amber. "
            "Lower angle = more flexion; elite pitchers extend into release."
        )
    except Exception as exc:
        st.info(f"Front knee overlay unavailable: {exc}")
else:
    st.info("Front knee overlay unavailable — metric did not compute for either session.")

try:
    fig = overlay_head_trace(
        result_a.pose_df, result_b.pose_df,
        phases_a, phases_b,
        result_a.video_metadata, result_b.video_metadata,
    )
    st.pyplot(fig, clear_figure=False)
    st.caption(
        "Head position trace from leg lift to ball release, translated to origin. "
        "Session A = blue, Session B = amber. "
        "Tighter clustering = better posture stability. "
        "Absolute alignment across different videos is approximate."
    )
except Exception as exc:
    st.info(f"Head trace overlay unavailable: {exc}")

# ── Export ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Export")

try:
    today = datetime.date.today()
    metadata_a = {
        "athlete_name":  athlete_name,
        "pitch_type":    "",
        "session_date":  today,
        "handedness":    handedness,
        "video_filename": uploaded_a.name,
    }
    metadata_b = {
        "athlete_name":  athlete_name,
        "pitch_type":    "",
        "session_date":  today,
        "handedness":    handedness,
        "video_filename": uploaded_b.name,
    }
    payload  = build_comparison_export_payload(comparison, metadata_a, metadata_b)
    json_str = json.dumps(payload, indent=2)

    athlete_slug = slugify_for_filename(athlete_name)
    prefix       = athlete_slug if athlete_slug else "comparison"
    filename     = f"{prefix}_{today.isoformat()}_AvsB.json"

    st.download_button(
        label     = "Download comparison JSON",
        data      = json_str,
        file_name = filename,
        mime      = "application/json",
    )
except Exception as exc:
    st.error(f"Failed to build export payload: {exc}")

# ── About this comparison ──────────────────────────────────────────────────
with st.expander("About this comparison", expanded=False):
    st.markdown("""
**About**

This page runs two videos through the same analysis pipeline and compares the
resulting mechanics metrics side by side.

**What to know when interpreting results**

- **Videos are not time-synced.** Both sessions play at native speed; no attempt is
  made to align them frame-by-frame.
- **Delivery progression charts** normalize each session to a shared 0–100% window
  (start of motion → end of motion) so deliveries of different durations can be
  compared by shape rather than by timestamp.
- **Unchanged threshold.** A metric must change by at least 2% (relative) before it
  is classified as improved or regressed. Smaller differences are shown as unchanged
  to suppress noise from pose estimation variance.
- **Improvement direction applies to 5 of 13 metrics.** Stride length,
  hip-shoulder separation, head path length, head max deviation, and front knee
  extension rate have a clearly better direction. The remaining 8 metrics are
  technique-dependent and shown without directional color.
- All measurements are derived from a single side-view camera. See the
  "About this tool & known limitations" section on the main page for the full list
  of caveats that apply equally to both sessions.

**Confidence flags and the comparison**

Each session is analyzed independently, and confidence flags are surfaced on the
main page when you analyze a single clip. If either session's pipeline flagged a
phase as unreliable (occluded lead leg, clip starts mid-motion), the affected metrics
in the comparison table carry that same uncertainty — a delta between a reliable value
and a best-guess estimate is itself unreliable. Check both sessions individually on
the main page if you see unexpected comparison results.
""")
