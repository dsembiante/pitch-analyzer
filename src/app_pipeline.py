"""
Pipeline orchestrator for the Streamlit app.

  run_full_pipeline()  -- @st.cache_data: full pipeline result cached per (video, handedness)

Note on landmarker lifecycle: MediaPipe PoseLandmarker in RunningMode.VIDEO requires
strictly-increasing timestamps. A cached landmarker CANNOT be reused across different
videos because each video's timestamps restart from 0. A fresh landmarker is created
per run; since the model file is already on disk after the first download, creation
is fast (model parse only, no network I/O).
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from pose_extraction import extract_pose
from phase_detection import detect_all_phases
from metrics.compute_all import compute_all_metrics
from visualization import draw_pose_overlay


@dataclass
class PipelineResult:
    pose_df: pd.DataFrame
    phases: dict
    metrics: dict           # metric_name -> MetricResult
    time_series: dict       # metric_name -> pd.DataFrame (for series-capable metrics)
    video_metadata: dict    # fps, width, height, frame_count, duration_s
    annotated_video_bytes: bytes


@st.cache_data(show_spinner=False)
def run_full_pipeline(video_bytes: bytes, handedness: str) -> PipelineResult:
    """Run the full 4-stage pipeline and return a cached PipelineResult.

    Stages:
      1. Pose extraction  — MediaPipe landmark detection on every frame
      2. Phase detection  — detect the 6 key delivery phases
      3. Metrics          — compute all 13 mechanics metrics
      4. Annotation       — render skeleton overlay video

    Results are cached by (video_bytes, handedness) so re-uploading the same
    file returns instantly without reprocessing.
    """
    with st.status("Analyzing pitch...", expanded=True) as status:

        # --- Stage 1: pose extraction ---
        status.update(label="Stage 1 / 4: Extracting pose landmarks...")
        tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp_in.write(video_bytes)
            tmp_in.flush()
            tmp_in.close()

            # Create a fresh landmarker per video — VIDEO mode requires strictly-increasing
            # timestamps so the same landmarker cannot be shared across different videos.
            pose_df = extract_pose(tmp_in.name)

            cap = cv2.VideoCapture(tmp_in.name)
            fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration_s = frame_count / fps if fps > 0 else 0.0

            # --- Stage 2: phase detection ---
            status.update(label="Stage 2 / 4: Detecting delivery phases...")
            phases = detect_all_phases(pose_df, handedness, fps=fps)

            # --- Stage 3: metrics ---
            status.update(label="Stage 3 / 4: Computing mechanics metrics...")
            video_metadata = {
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "duration_s": duration_s,
            }
            metrics, time_series = compute_all_metrics(pose_df, phases, handedness, video_metadata)

            # --- Stage 4: annotated video ---
            status.update(label="Stage 4 / 4: Rendering annotated video...")
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_out.close()
            try:
                draw_pose_overlay(tmp_in.name, pose_df, tmp_out.name, handedness=handedness)
                annotated_bytes = Path(tmp_out.name).read_bytes()
            finally:
                Path(tmp_out.name).unlink(missing_ok=True)

        finally:
            Path(tmp_in.name).unlink(missing_ok=True)

        status.update(label="Done!", state="complete", expanded=False)

    return PipelineResult(
        pose_df=pose_df,
        phases=phases,
        metrics=metrics,
        time_series=time_series,
        video_metadata=video_metadata,
        annotated_video_bytes=annotated_bytes,
    )
