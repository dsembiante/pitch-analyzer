"""CLI script: extract pose landmarks from a pitching video and save to outputs/."""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_extraction import extract_pose, save_pose_data


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose data from a pitching video.")
    parser.add_argument("--video", required=True, help="Path to the input video file (.mp4, .mov, etc.)")
    parser.add_argument("--output", default=None, help="Output path (default: outputs/<video_stem>_pose.parquet)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path("outputs") / f"{video_path.stem}_pose.parquet"

    print(f"Processing: {video_path}")
    print("Running MediaPipe Pose extraction (model_complexity=2) — this may take a moment...")

    df = extract_pose(str(video_path))

    total_frames = df["frame"].nunique()
    total_rows = len(df)
    print(f"\n--- Extraction complete ---")
    print(f"Frames processed (with detected pose): {total_frames}")
    print(f"Total landmark rows: {total_rows}  ({total_rows // total_frames if total_frames else 0} landmarks/frame)")

    print("\n--- First 10 rows ---")
    print(df.head(10).to_string(index=False))

    print("\n--- x/y coordinate ranges (should be roughly 0.0–1.0) ---")
    print(f"  x:  min={df['x'].min():.4f}  max={df['x'].max():.4f}  mean={df['x'].mean():.4f}")
    print(f"  y:  min={df['y'].min():.4f}  max={df['y'].max():.4f}  mean={df['y'].mean():.4f}")
    print(f"  visibility:  min={df['visibility'].min():.4f}  max={df['visibility'].max():.4f}")

    low_vis = df[df["visibility"] < 0.5]
    pct_low = 100.0 * len(low_vis) / len(df)
    print(f"\n  Landmarks with visibility < 0.5: {len(low_vis)} ({pct_low:.1f}%) — high % may mean occlusion issues")

    save_pose_data(df, str(output_path))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
