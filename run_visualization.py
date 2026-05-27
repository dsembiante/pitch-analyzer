"""CLI: draw MediaPipe pose overlay on a pitching video using extracted pose data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from visualization import draw_pose_overlay


def main():
    parser = argparse.ArgumentParser(
        description="Render skeleton overlay on a pitching video from Phase 1 pose data."
    )
    parser.add_argument("video", help="Path to original video file (.mp4, .mov, etc.)")
    parser.add_argument("pose_data", help="Path to pose parquet file from run_extraction.py")
    parser.add_argument(
        "--handedness", default="right", choices=["right", "left"],
        help="Pitcher handedness — controls throwing arm (red) and lead leg (green) colors (default: right)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: outputs/<video_stem>_overlay.mp4)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    pose_path  = Path(args.pose_data)

    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)
    if not pose_path.exists():
        print(f"Error: pose data not found: {pose_path}")
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else Path("outputs") / f"{video_path.stem}_overlay.mp4"
    )

    print(f"Video:      {video_path}")
    print(f"Pose data:  {pose_path}")
    print(f"Handedness: {args.handedness}")
    print(f"Output:     {output_path}")
    print()

    df = pd.read_parquet(str(pose_path))
    print(f"Loaded {len(df)} landmark rows across {df['frame'].nunique()} frames\n")

    draw_pose_overlay(
        video_path=str(video_path),
        pose_data=df,
        output_path=str(output_path),
        handedness=args.handedness,
    )

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"\nSaved to: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
