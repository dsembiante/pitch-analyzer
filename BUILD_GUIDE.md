# Build Guide

Implementation reference for the pitch-analyzer pipeline. Each Phase section
describes what was built, the key design decisions, and deferred items.

---

## Phase 0 — Environment Setup
Python 3.11 venv, MediaPipe Tasks API (0.10.x), OpenCV, pandas, scipy, matplotlib.
Model: `pose_landmarker_heavy.task` (~27 MB, gitignored, auto-downloaded to `models/`).

## Phase 1 — Pose Extraction (`src/pose_extraction.py`, `run_extraction.py`)
MediaPipe `PoseLandmarker` in `RunningMode.VIDEO`. Outputs a long-format parquet:
`frame, timestamp_ms, landmark_idx, landmark_name, x, y, z, visibility`
(33 landmarks per frame, x/y normalized 0–1 relative to frame dimensions).

## Phase 2 — Pose Visualization (`src/visualization.py`, `run_visualization.py`)
OpenCV skeleton overlay: throwing arm in red, lead leg in green, everything else yellow.
No `mp.solutions.drawing_utils` — all drawing is manual via `POSE_CONNECTIONS`.

## Phase 3 — Phase Detection (`src/phase_detection.py`, `run_phase_detection.py`)
Six delivery phases detected from wrist/ankle velocity signals:

| Key | Description |
|-----|-------------|
| `start_of_motion` | Last sustained stillness block before delivery |
| `leg_lift_peak` | Lead ankle at minimum y (highest point) |
| `foot_strike` | Lead ankle velocity transitions from falling to stopped |
| `max_layback` | Throwing wrist at furthest horizontal layback (2D proxy for MER) |
| `ball_release` | Peak throwing-wrist horizontal velocity |
| `end_of_motion` | Wrist velocity decays below 15% of peak for 5+ frames |

Key design choice: all early phases (leg lift, foot strike, max layback) use
fixed lookbacks from `ball_release` rather than `window[0]`, because the
pitching-window search finds the wrist-velocity burst, which starts *after*
these earlier events.

Diagnostic tool: `src/phase_diagnostics.py` / `run_phase_detection.py --diagnostic-plot`
produces a 4-panel signal plot for visual verification.

## Phase 4 — Mechanics Metrics (`src/metrics/`, `run_metrics.py`)

### Architecture
- `src/metrics/_landmarks.py` — handedness-aware landmark index constants
- `src/metrics/_geometry.py` — pixel-space geometry helpers (angle, distance, midpoint)
- `src/metrics/_pose_access.py` — landmark extraction, visibility checks, window builder
- `src/metrics/_types.py` — `MetricResult` dataclass with `to_dict()`
- `src/metrics/compute_all.py` — orchestrator; runs all metrics in try/except isolation
- `run_metrics.py` — CLI: reads parquet + phases JSON, writes metrics JSON

**Critical design rule:** All geometric calculations use pixel coordinates
(`norm_x * width`, `norm_y * height`). On non-square frames (e.g., 1080×1920
portrait video) normalized-coordinate angles are distorted by the aspect ratio.

### Metrics implemented (9 of 10)

| # | Key | File | Window | Unit |
|---|-----|------|--------|------|
| 1 | `arm_slot` | `arm_slot.py` | single frame: ball_release | degrees |
| 2 | `stride_length` | `stride_length.py` | start_of_motion → foot_strike (horizontal only) | % body height |
| 3 | `hip_shoulder_separation_max` | `hip_shoulder_separation.py` | leg_lift → foot_strike | degrees |
| 4 | `front_knee_flex` | `front_knee_flex.py` | single frame: ball_release | degrees |
| 5a | `trunk_tilt_lateral` | `trunk_tilt.py` | single frame: ball_release | degrees |
| 5b | `trunk_tilt_forward` | `trunk_tilt.py` | single frame: ball_release | degrees |
| 6a | `tempo_leg_lift_to_foot_strike` | `tempo.py` | timestamp difference | seconds |
| 6b | `tempo_foot_strike_to_release` | `tempo.py` | timestamp difference | seconds |
| 6c | `tempo_total_motion` | `tempo.py` | timestamp difference | seconds |
| 8 | `balance_point` | `balance_point.py` | single frame: leg_lift_peak | % body height |
| 9 | `front_knee_extension_rate` | `front_knee_extension.py` | foot_strike → ball_release | deg/s |
| 10a | `head_path_length` | `head_movement.py` | leg_lift → ball_release | % body height |
| 10b | `head_max_deviation` | `head_movement.py` | leg_lift → ball_release | % body height |

### Deferred: Metric #7 — Release Point Consistency
Release point consistency (session-over-session repeatability of ball_release
position) requires multiple pitches to compute a meaningful variance, so it is
deferred to Phase 6 where multi-session comparison is implemented.

### Visibility policy
Default threshold 0.5 for all landmark checks. Exception: back ankle at the
setup frame (start_of_motion, leg_lift_peak) uses threshold 0.25 — the planted
back foot has reliably lower MediaPipe confidence at those frames but still
provides usable position data. Documented in LIMITATIONS.md #13.

---

## Phase 5 — Streamlit UI (planned)
## Phase 6 — Multi-session comparison + release point consistency (planned)
## Phase 7 — README, pitch deck, deployment (planned)
## Phase 8 — Testing and validation (planned)
