"""
MediaPipe Pose landmark index constants, organized by handedness role.

MediaPipe numbers 33 landmarks 0-32. The mapping between body part and index
is fixed regardless of which way the person is facing:
  - 11 = LEFT_SHOULDER, 12 = RIGHT_SHOULDER
  - 13 = LEFT_ELBOW,    14 = RIGHT_ELBOW
  - 15 = LEFT_WRIST,    16 = RIGHT_WRIST
  - 23 = LEFT_HIP,      24 = RIGHT_HIP
  - 25 = LEFT_KNEE,     26 = RIGHT_KNEE
  - 27 = LEFT_ANKLE,    28 = RIGHT_ANKLE
  - 29 = LEFT_HEEL,     30 = RIGHT_HEEL
  - 31 = LEFT_FOOT_INDEX, 32 = RIGHT_FOOT_INDEX

Pitching convention (right-handed example):
  - "Throwing" side = right: wrist=16, elbow=14, shoulder=12
  - "Lead" side     = left:  ankle=27, knee=25, hip=23, shoulder=11
  - "Back" side     = right: ankle=28, hip=24

For a LEFT-HANDED pitcher everything mirrors: throwing wrist=15, lead ankle=28, etc.
"""

# --- Handedness-aware landmark dicts ---
# Each dict maps "right"/"left" (handedness, not body side) to MediaPipe index.

THROWING_WRIST    = {"right": 16, "left": 15}
THROWING_ELBOW    = {"right": 14, "left": 13}
THROWING_SHOULDER = {"right": 12, "left": 11}

LEAD_ANKLE        = {"right": 27, "left": 28}
LEAD_KNEE         = {"right": 25, "left": 26}
LEAD_HIP          = {"right": 23, "left": 24}
LEAD_SHOULDER     = {"right": 11, "left": 12}  # opposite side from throwing arm

BACK_ANKLE        = {"right": 28, "left": 27}
BACK_HIP          = {"right": 24, "left": 23}

# --- Handedness-independent landmarks ---
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
NOSE           = 0
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
