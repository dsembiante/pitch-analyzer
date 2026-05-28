# Pitch Analyzer

Automated pitching-mechanics analysis from a single side-angle video — pose extraction, phase detection, and 13 biomechanics metrics in one Streamlit interface.

---

## Why this matters

Pitching development has a measurement problem. A coach watching film can spot what's wrong in a few seconds, but extracting the actual numbers — arm slot, stride length, hip-shoulder separation — takes significant manual effort per clip. That time spent measuring is time not spent coaching.

This tool automates the measurement side. Upload a clip and you get 13 quantified mechanics metrics in seconds, with session-over-session comparison built in. The goal is to shift the coach's time from "what is the number?" to "what does this mean for this athlete?" — which is where their expertise actually lives.

---

## What it does

- **Pose extraction** — Runs Google MediaPipe PoseLandmarker (heavy model, 33 landmarks) on every frame of the input video. Outputs a frame-by-frame landmark dataset with visibility scores.
- **Automatic phase detection** — Identifies six key delivery phases from wrist and ankle velocity signals: start of motion, leg lift peak, foot strike, max layback, ball release, end of motion. No manual frame labeling required.
- **13 mechanics metrics** — Computed geometrically from pose data across the detected phases. Covers arm slot, trunk tilt, stride length, hip-shoulder separation, front knee flexion, front knee extension rate, balance point drift, three tempo measurements, and two head-stability measurements.
- **Annotated video output** — Overlays the MediaPipe skeleton and detected phase labels on the original footage for visual verification.
- **Session-over-session comparison** — Upload two clips and get a side-by-side metric table with delta, percent change, and improvement/regression classification for metrics that have a directionally better value.
- **Confidence flagging** — Detects when phase detection is unreliable (occluded lead ankle, clip starts mid-delivery) and surfaces a warning naming the affected metrics.
- **JSON export** — Every session and comparison can be downloaded as a structured JSON file for external analysis or record-keeping.

---

## Tech stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Pose estimation | MediaPipe 0.10.x (Tasks API) | 33-point skeletal landmark extraction |
| Video I/O | OpenCV | Frame-level read/write, skeleton overlay |
| Data wrangling | pandas, pyarrow | Long-format landmark storage (parquet) |
| Signal processing | scipy | Savitzky-Golay smoothing, peak detection |
| Charts | matplotlib | Time-series plots, head trace, comparison overlays |
| Web interface | Streamlit | Single-page app + multi-page navigation |
| Version control | GitHub | Source and CI |

Python 3.11. All dependencies pinned in `requirements.txt`.

---

## Screenshots

> Add real screenshots after the first demo run. Suggested captures:

```
[SCREENSHOT: app main page — video and metrics table side by side]
[SCREENSHOT: metrics table with phase column and description tooltips]
[SCREENSHOT: comparison view — metric-by-metric delta table with color coding]
[SCREENSHOT: confidence flag warning banner (use a clip with partial setup visible)]
```

---

## How to run

**Prerequisites:** Python 3.11, git.

```bash
# 1. Clone
git clone <repo-url>
cd pitch-analyzer

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run src/app.py
```

The MediaPipe pose model (`pose_landmarker_heavy.task`, ~27 MB) is downloaded automatically to `models/` on the first run. After that, startup is fast.

**Running the stress test** (verifies phase detection and metrics across a set of test clips):

```bash
python run_stress_test.py
```

Output is written to `outputs/stress_test/`. Requires test videos in `videos/` (gitignored).

**CLI tools** (for pipeline step-by-step inspection):

```bash
python run_extraction.py --video videos/mypitch.mp4 --output outputs/pose.parquet
python run_phase_detection.py --pose outputs/pose.parquet --video videos/mypitch.mp4
python run_metrics.py --pose outputs/pose.parquet --phases outputs/phases.json --video videos/mypitch.mp4
```

---

## Known limitations

The full list is in [LIMITATIONS.md](LIMITATIONS.md) (18 items). Key ones to know upfront:

- **2D measurements only.** Every metric is computed from a single camera view. Rotational quantities (arm slot, hip-shoulder separation, trunk tilt) lose their depth component. Values track relative changes well but shouldn't be compared against 3D motion-capture benchmarks.
- **Single pitch per clip required.** The phase detector finds the highest-sustained wrist-velocity burst. Multi-pitch videos or bullpen sessions produce unreliable results. Clip to one delivery before uploading.
- **Side-angle camera, pitcher perpendicular to lens.** The model assumes the pitcher's body plane is roughly parallel to the image plane. Off-angle footage degrades nearly every metric.
- **Full body must be visible throughout.** Partial cropping, early frame cuts, or the pitcher walking out of frame mid-delivery will corrupt phase detection.
- **Slow-motion video is rejected.** Phase detection relies on real-time velocity signals. Slow-motion footage compresses those signals below detectable thresholds; the pipeline raises an error rather than silently returning wrong values.

---

## Project structure

```
src/
├── app.py                     # Streamlit entry point — main analysis page
├── app_pipeline.py            # run_full_pipeline() orchestrator (pose → phases → metrics → video)
├── app_utils.py               # Display helpers, metric ordering, improvement directions, export
├── app_charts.py              # matplotlib chart builders (hip-shoulder, knee, head trace)
├── app_comparison.py          # Comparison data model and classification logic
├── app_comparison_utils.py    # Comparison table styling, tooltip text, export payload
├── pages/
│   └── 2_Compare_Sessions.py  # Streamlit multi-page: side-by-side session comparison
├── pose_extraction.py         # MediaPipe PoseLandmarker wrapper; outputs long-format parquet
├── visualization.py           # OpenCV skeleton overlay (colored by limb role)
├── phase_visualization.py     # Phase-label video overlay (annotates detected phase frames)
├── phase_detection.py         # Signal-processing phase detector (velocity heuristics + confidence flags)
├── phase_diagnostics.py       # 4-panel diagnostic plot for phase detection QA
└── metrics/
    ├── compute_all.py          # Metric orchestrator; runs all metrics in isolated try/except
    ├── _landmarks.py           # Handedness-aware landmark index constants (MediaPipe indices 0–32)
    ├── _geometry.py            # Pixel-space geometry helpers (angle, distance, midpoint)
    ├── _pose_access.py         # Landmark extraction, visibility checks, window data builder
    ├── _types.py               # MetricResult dataclass
    ├── arm_slot.py             # Arm elevation angle at peak arm position near release
    ├── trunk_tilt.py           # Lateral and forward trunk tilt at release
    ├── front_knee_flex.py      # Front knee flexion angle at release
    ├── front_knee_extension.py # Rate of front knee extension from foot strike to release
    ├── stride_length.py        # Horizontal stride length, normalized to body height
    ├── hip_shoulder_separation.py  # Peak hip-shoulder separation angle in the stride phase
    ├── balance_point.py        # Hip drift relative to back ankle at leg lift peak
    ├── tempo.py                # Three tempo measurements (leg lift→FS, FS→release, total)
    └── head_movement.py        # Head path length and max deviation relative to hip midpoint
```

Top-level CLI scripts (`run_*.py`) wrap each pipeline stage for standalone use outside the app.

---

## Architecture and design choices

The pipeline is a deterministic, hand-written signal-processing system layered on top of a pre-trained neural network. MediaPipe provides the 33-point pose landmarks; everything downstream — phase detection, metric computation, confidence flagging — is explicit geometric and kinematic logic with no learned components.

Phase detection works by computing velocity and position signals from specific landmarks (throwing wrist, lead ankle), smoothing them with a Savitzky-Golay filter, and applying threshold-based heuristics to locate the six delivery phases. This approach is interpretable and debuggable: the 4-panel diagnostic plot shows exactly what signals are being used and where the detector fired.

All geometric metric computations happen in pixel space (normalized MediaPipe coordinates × frame dimensions). This is not optional — on non-square video (e.g., 1080×1920 portrait), computing angles directly from normalized [0, 1] coordinates distorts them by the aspect ratio.

The system intentionally makes no probabilistic predictions and has no personalization — it measures, it does not advise. That boundary is a feature: it means every number the tool reports can be traced to a specific formula applied to specific pose landmarks at a specific frame.

---

## Future directions

This is a functional prototype built to demonstrate what automated measurement of pitching mechanics looks like in practice. A production version of this system would likely pursue several directions this prototype deliberately does not:

- **3D pose estimation.** Two cameras at known angles, or a depth sensor, would remove the single-camera projection limitation that affects every rotational metric. Several open-source multi-view pose systems exist; the pipeline architecture is designed to swap in a different landmark source without restructuring downstream logic.
- **Multi-pitch support and session aggregation.** Averaging metrics across 10–20 pitches per session is the standard in biomechanics research. The current single-pitch-per-clip constraint would need a segmentation step to identify individual pitches in a bullpen session.
- **Learned phase detection.** The current velocity-heuristic detector is robust on clean side-angle footage but degrades with unusual deliveries or marginal video quality. A small sequence model trained on labeled delivery clips could generalize more broadly.
- **Calibrated thresholds and population norms.** The tool reports values but makes no claims about what "good" looks like numerically, because those benchmarks require validated data. Building a reference database from labeled expert assessments would let the tool contextualize measurements against age/level cohorts.
- **Biomechanical injury risk signals.** Some mechanics patterns correlate with elevated UCL stress, shoulder impingement, etc. Connecting measurements to evidence-based risk factors is clinically valuable but requires careful validation before surfacing to coaches — out of scope for a demo prototype.
