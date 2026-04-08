"""
Central location for all tunable parameters and constants.
Import from here instead of hardcoding magic numbers across files.
"""

# ──────────────────────────────────────────────
# Keypoint Dimensions (MediaPipe Holistic)
# ──────────────────────────────────────────────
POSE_LANDMARKS = 33
POSE_DIM = POSE_LANDMARKS * 4       # x, y, z, visibility

FACE_LANDMARKS = 468
FACE_DIM = FACE_LANDMARKS * 3       # x, y, z

HAND_LANDMARKS = 21
HAND_DIM = HAND_LANDMARKS * 3       # x, y, z (per hand)

TOTAL_KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2  # 1662

# ──────────────────────────────────────────────
# Inference Parameters
# ──────────────────────────────────────────────
SEQUENCE_WINDOW = 30                 # frames to accumulate before predicting
MAX_SEQ_LENGTH_CAP = 30              # MUST match SEQUENCE_WINDOW for consistency
CONFIDENCE_THRESHOLD = 0.7           # minimum confidence for webapp
CONFIDENCE_THRESHOLD_STRICT = 0.85   # stricter threshold for local detection
MOVEMENT_THRESHOLD = 0.001           # min movement magnitude to trigger prediction
RESET_FRAMES = 15                    # frames of no detection before resetting
CONSENSUS_REQUIRED_LOOSE = 2         # prediction consensus for webapp (2 of 3)
CONSENSUS_REQUIRED_STRICT = 4        # prediction consensus for local (4 of 5)
PREDICTION_HISTORY_LOOSE = 3         # history window for webapp
PREDICTION_HISTORY_STRICT = 5        # history window for local detection

# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────
MIN_SAMPLES_PER_CLASS = 10
DATA_AUGMENTATION_NOISE = 0.02       # σ for Gaussian noise augmentation
DATA_AUGMENTATION_COUNT = 2

# ──────────────────────────────────────────────
# Hardware / Training Acceleration
# ──────────────────────────────────────────────
ENABLE_MIXED_PRECISION = True      # float16 compute on Tensor Core GPUs
ENABLE_XLA = False                 # disabled to reduce VRAM compile spikes
GPU_MEMORY_GROWTH = True           # incremental VRAM allocation
DEFAULT_BATCH_SIZE = 32            # lowered to 32 to prevent RTX 4050 OOM
PREFETCH_BUFFER = "auto"           # tf.data.AUTOTUNE

# ──────────────────────────────────────────────
# MediaPipe Configuration
# ──────────────────────────────────────────────
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5
MP_MODEL_COMPLEXITY = 0
