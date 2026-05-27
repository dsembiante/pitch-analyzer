"""CLI: detect pitching phases from pose data and generate an annotated video."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from phase_detection import detect_all_phases
from phase_visualization import draw_phases_on_video


def main():
    parser = argparse.ArgumentParser(
        description="Detect pitching delivery phases from Phase 1 pose data."
    )
    parser.add_argument("--video",  required=True, help="Path to original video file")
    parser.add_argument("--pose",   required=True, help="Path to pose parquet file")
    parser.add_argument("--handedness", default="right", choices=["right", "left"])
    parser.add_argument("--output", default=None, help="JSON output path (default: outputs/<stem>_phases.json)")
    parser.add_argument("--annotated-video", default=None,
                        help="Annotated video path (default: outputs/<stem>_phases.mp4)")
    args = parser.parse_args()

    video_path = Path(args.video)
    pose_path  = Path(args.pose)

    for p, label in [(video_path, "video"), (pose_path, "pose data")]:
        if not p.exists():
            print(f"Error: {label} not found: {p}")
            sys.exit(1)

    stem = video_path.stem
    json_path  = Path(args.output) if args.output else Path("outputs") / f"{stem}_phases.json"
    video_out  = Path(args.annotated_video) if args.annotated_video else Path("outputs") / f"{stem}_phases.mp4"

    print(f"Video:      {video_path}")
    print(f"Pose data:  {pose_path}")
    print(f"Handedness: {args.handedness}")
    print()

    df = pd.read_parquet(str(pose_path))
    fps = 1000.0 / float(np.median(np.diff(df.timestamp_ms.unique())))
    print(f"Loaded {len(df)} landmark rows, {df.frame.nunique()} frames, ~{fps:.1f} fps")

    print(f"\nRunning phase detection...")
    phases = detect_all_phases(df, handedness=args.handedness, fps=fps)

    # --- summary table ---
    phase_order = [
        ("start_of_motion",       "Start of Motion"),
        ("leg_lift_peak",         "Leg Lift Peak"),
        ("foot_strike",           "Foot Strike"),
        ("max_external_rotation", "Max Ext Rotation"),
        ("ball_release",          "Ball Release"),
        ("end_of_motion",         "End of Motion"),
    ]
    print()
    print(f"{'Phase':<25} {'Frame':>6}  {'Time (s)':>8}")
    print("-" * 44)
    for key, label in phase_order:
        frame = phases[key]
        ts_s  = phases[f"{key}_timestamp_ms"] / 1000.0
        print(f"{label:<25} {frame:>6}  {ts_s:>8.3f}s")
    print()
    print(f"Pitching window:  frames {phases['window_start']}-{phases['window_end']}  "
          f"({(phases['window_end'] - phases['window_start']) / fps:.2f}s)")

    # --- save JSON ---
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(phases, f, indent=2)
    print(f"\nPhase data saved to: {json_path}")

    # --- annotated video ---
    print(f"\nGenerating annotated video: {video_out}")
    draw_phases_on_video(
        video_path=str(video_path),
        pose_data=df,
        phases=phases,
        output_path=str(video_out),
        handedness=args.handedness,
    )

    size_mb = video_out.stat().st_size / 1_000_000
    print(f"Saved to: {video_out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
