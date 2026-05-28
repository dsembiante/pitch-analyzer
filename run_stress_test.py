"""
Stress-test runner for Phase 7: pose extraction, overlay, phase detection,
phase-label video, and diagnostic PNG for each video in the test set.

Usage:
    python run_stress_test.py

All artifacts are written to outputs/stress_test/ with per-video naming.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_extraction import extract_pose, save_pose_data
from visualization import draw_pose_overlay
from phase_detection import detect_all_phases
from phase_visualization import draw_phases_on_video
from phase_diagnostics import plot_phase_signals

# ── Per-video config ───────────────────────────────────────────────────────
# Add or change entries here; handedness can be "right" or "left" per video.
VIDEOS = [
    {"path": "videos/IMG_8605.MOV", "handedness": "right"},
    {"path": "videos/IMG_8607.MOV", "handedness": "right"},
    {"path": "videos/IMG_8608.MOV", "handedness": "right"},
    {"path": "videos/IMG_8609.MOV", "handedness": "right"},
]

OUTPUT_DIR = Path("outputs/stress_test")

_PHASE_ORDER = [
    ("start_of_motion", "Start of Motion"),
    ("leg_lift_peak",   "Leg Lift Peak"),
    ("foot_strike",     "Foot Strike"),
    ("max_layback",     "Max Layback"),
    ("ball_release",    "Ball Release"),
    ("end_of_motion",   "End of Motion"),
]


def _print_phases(phases: dict) -> None:
    print(f"    {'Phase':<25} {'Frame':>6}  {'Time (s)':>8}")
    print(f"    {'-' * 43}")
    for key, label in _PHASE_ORDER:
        frame = phases.get(key)
        ts_ms = phases.get(f"{key}_timestamp_ms")
        if frame is not None and ts_ms is not None:
            print(f"    {label:<25} {frame:>6}  {ts_ms / 1000.0:>8.3f}s")
        else:
            print(f"    {label:<25} {'--':>6}  {'--':>8}")


def process_video(config: dict) -> None:
    video_path = Path(config["path"])
    handedness = config["handedness"]
    stem       = video_path.stem

    print(f"\n{'=' * 62}")
    print(f"  {video_path.name}  [{handedness}]")
    print(f"{'=' * 62}")

    pose_path    = OUTPUT_DIR / f"{stem}_pose.parquet"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay.mp4"
    phases_path  = OUTPUT_DIR / f"{stem}_phases.json"
    phase_vid    = OUTPUT_DIR / f"{stem}_phases.mp4"
    diag_path    = OUTPUT_DIR / f"{stem}_diagnostic.png"

    # ── Stage 1: Pose extraction ──────────────────────────────────────────
    # extract_pose() creates and closes its own landmarker — VIDEO mode requires
    # strictly-increasing timestamps, so a fresh landmarker is needed per video.
    print(f"\n  [1/5] Pose extraction...")
    t0 = time.time()
    df = extract_pose(str(video_path))
    save_pose_data(df, str(pose_path))
    n_frames = df["frame"].nunique()
    print(f"        {n_frames} frames, {len(df)} rows  ({time.time() - t0:.1f}s)")
    print(f"        -> {pose_path}")

    # ── Stage 2: Pose overlay video ───────────────────────────────────────
    print(f"\n  [2/5] Pose overlay video...")
    t0 = time.time()
    draw_pose_overlay(
        video_path  = str(video_path),
        pose_data   = df,
        output_path = str(overlay_path),
        handedness  = handedness,
    )
    size_mb = overlay_path.stat().st_size / 1_000_000
    print(f"        {size_mb:.1f} MB  ({time.time() - t0:.1f}s)")
    print(f"        -> {overlay_path}")

    # ── Stage 3: Phase detection ──────────────────────────────────────────
    print(f"\n  [3/5] Phase detection...")
    t0 = time.time()
    fps = 1000.0 / float(np.median(np.diff(df.timestamp_ms.unique())))
    phases = detect_all_phases(df, handedness=handedness, fps=fps)
    with open(phases_path, "w") as fh:
        json.dump(phases, fh, indent=2)
    print(f"        fps={fps:.1f}  ({time.time() - t0:.1f}s)")
    print(f"        -> {phases_path}")
    print()
    _print_phases(phases)

    # ── Stage 4: Phase-label video ────────────────────────────────────────
    print(f"\n  [4/5] Phase-label video...")
    t0 = time.time()
    draw_phases_on_video(
        video_path  = str(video_path),
        pose_data   = df,
        phases      = phases,
        output_path = str(phase_vid),
        handedness  = handedness,
    )
    size_mb = phase_vid.stat().st_size / 1_000_000
    print(f"        {size_mb:.1f} MB  ({time.time() - t0:.1f}s)")
    print(f"        -> {phase_vid}")

    # ── Stage 5: Phase diagnostics PNG ────────────────────────────────────
    print(f"\n  [5/5] Phase diagnostics PNG...")
    t0 = time.time()
    plot_phase_signals(
        pose_df     = df,
        phases      = phases,
        handedness  = handedness,
        fps         = fps,
        output_path = str(diag_path),
    )
    print(f"        ({time.time() - t0:.1f}s)")
    print(f"        -> {diag_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successes: list[str] = []
    failures:  list[dict] = []

    total_t0 = time.time()

    for config in VIDEOS:
        video_path = Path(config["path"])
        if not video_path.exists():
            print(f"\nSkipping {config['path']} -- file not found")
            failures.append({"video": config["path"], "error": "file not found"})
            continue
        try:
            process_video(config)
            successes.append(config["path"])
        except Exception as exc:
            print(f"\n  ERROR processing {config['path']}: {exc}")
            failures.append({"video": config["path"], "error": str(exc)})

    elapsed_total = time.time() - total_t0
    print(f"\n{'=' * 62}")
    print(f"  Stress test complete  ({elapsed_total:.0f}s total)")
    print(f"  Passed: {len(successes)} / {len(VIDEOS)}")
    if failures:
        print(f"  Failed: {len(failures)}")
        for f in failures:
            print(f"    {f['video']}: {f['error']}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
