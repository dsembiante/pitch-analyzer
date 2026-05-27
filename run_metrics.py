"""CLI: compute pitching mechanics metrics from pose data and phases JSON."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from metrics.compute_all import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute pitching mechanics metrics from Phase 1 pose data and Phase 3 phases."
    )
    parser.add_argument("--pose",       required=True, help="Path to pose parquet file")
    parser.add_argument("--phases",     required=True, help="Path to phases JSON file")
    parser.add_argument("--video",      required=True, help="Path to original video (for fps/dimensions)")
    parser.add_argument("--handedness", default="right", choices=["right", "left"])
    parser.add_argument("--output",     default=None,
                        help="JSON output path (default: outputs/<stem>_metrics.json)")
    args = parser.parse_args()

    pose_path   = Path(args.pose)
    phases_path = Path(args.phases)
    video_path  = Path(args.video)

    for p, label in [
        (pose_path,   "pose data"),
        (phases_path, "phases JSON"),
        (video_path,  "video"),
    ]:
        if not p.exists():
            print(f"Error: {label} not found: {p}")
            sys.exit(1)

    stem = video_path.stem
    out_path = Path(args.output) if args.output else Path("outputs") / f"{stem}_metrics.json"

    # --- read video metadata ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        sys.exit(1)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_metadata = {"fps": fps, "width": width, "height": height}

    # --- load inputs ---
    df = pd.read_parquet(str(pose_path))
    with open(phases_path) as f:
        phases = json.load(f)

    print(f"Pose data:  {pose_path}  ({df.frame.nunique()} frames)")
    print(f"Phases:     {phases_path}")
    print(f"Video:      {video_path}  ({width}x{height}, {fps:.0f} fps)")
    print(f"Handedness: {args.handedness}")
    print()

    # --- compute metrics ---
    results = compute_all_metrics(df, phases, args.handedness, video_metadata)

    # --- summary table ---
    col_w = 30
    print(f"{'Metric':<{col_w}} {'Value':>10}  {'Unit':<8}  Note / Error")
    print("-" * 75)
    for name, r in results.items():
        if r.error:
            val_str = "ERROR"
            note    = r.error
        else:
            val_str = f"{r.value:.1f}" if r.value is not None else "n/a"
            note    = r.notes[:55] + "..." if len(r.notes) > 55 else r.notes
        print(f"{r.display_name:<{col_w}} {val_str:>10}  {r.unit:<8}  {note}")

    # --- save JSON ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_dict = {name: r.to_dict() for name, r in results.items()}
    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    print()
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
