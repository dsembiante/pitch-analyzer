import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pandas as pd
from pathlib import Path

_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
_MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker_heavy.task"


def _ensure_model() -> Path:
    """Download the heavy pose landmarker bundle on first use (~27 MB)."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Download complete.")
    return _MODEL_PATH


def create_landmarker():
    """Create and return a PoseLandmarker; caller owns the lifecycle.

    Use this when you want to reuse a single landmarker across multiple calls
    (e.g., cached with @st.cache_resource). The returned object is NOT entered
    as a context manager — call .close() when done, or let the GC collect it.
    """
    model_path = _ensure_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _run_extraction(cap, landmarker) -> list:
    """Inner extraction loop; shared by extract_pose with or without external landmarker."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int(frame_idx / fps * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            for lm_idx, lm in enumerate(result.pose_landmarks[0]):
                rows.append({
                    "frame": frame_idx,
                    "timestamp_ms": round(frame_idx / fps * 1000, 2),
                    "landmark_idx": lm_idx,
                    "landmark_name": _LANDMARK_NAMES[lm_idx],
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                })
        frame_idx += 1
    return rows


def extract_pose(video_path: str, landmarker=None) -> pd.DataFrame:
    """Extract MediaPipe pose landmarks for every frame of a video.

    Uses the Tasks API (MediaPipe 0.10+), RunningMode.VIDEO so timestamps are
    strictly increasing integers required by the landmarker.

    Args:
        video_path: Path to the video file.
        landmarker: Optional pre-created PoseLandmarker (e.g., from create_landmarker()).
                    When provided it is NOT closed on exit — caller owns lifecycle.
                    When None (default) a landmarker is created and closed automatically.

    Returns a DataFrame with one row per (frame, landmark) pair.
    Columns: frame, timestamp_ms, landmark_idx, landmark_name, x, y, z, visibility
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    owned = landmarker is None
    if owned:
        landmarker = create_landmarker()

    try:
        rows = _run_extraction(cap, landmarker)
    finally:
        cap.release()
        if owned:
            landmarker.close()

    if not rows:
        raise RuntimeError(
            "No pose landmarks detected — check that the video contains a visible person "
            "and that the angle/lighting are reasonable."
        )

    return pd.DataFrame(rows)


def save_pose_data(df: pd.DataFrame, output_path: str) -> None:
    """Save pose DataFrame to parquet or CSV based on file extension."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
