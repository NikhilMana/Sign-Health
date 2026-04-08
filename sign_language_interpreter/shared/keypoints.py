"""
Single source of truth for keypoint extraction from MediaPipe Holistic results.

Includes body-centered normalization to make predictions
position-, scale-, and distance-invariant.
"""

import numpy as np
from shared.constants import (
    POSE_DIM, FACE_DIM, HAND_DIM, TOTAL_KEYPOINT_DIM,
    POSE_LANDMARKS, FACE_LANDMARKS, HAND_LANDMARKS,
)

# ── Pose landmark indices (MediaPipe Holistic) ───────────
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24


def normalize_keypoints(raw, dtype=np.float32):
    """
    Body-centered normalization.

    1. Compute body center = midpoint(left_hip, right_hip).
    2. Compute body scale  = ‖left_shoulder − right_shoulder‖.
    3. Subtract center, divide by scale for all xyz coords.
    4. Pose visibility channel is left untouched.

    Returns:
        np.ndarray, same shape as *raw*.
    """
    out = raw.copy().astype(dtype)

    # ── Unpack pose ──────────────────────────────────────
    pose = out[:POSE_DIM].reshape(POSE_LANDMARKS, 4)

    left_hip = pose[_LEFT_HIP, :3]
    right_hip = pose[_RIGHT_HIP, :3]
    center = (left_hip + right_hip) / 2.0

    left_shoulder = pose[_LEFT_SHOULDER, :3]
    right_shoulder = pose[_RIGHT_SHOULDER, :3]
    scale = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6

    # Normalise pose xyz (column 0-2), keep visibility (column 3)
    pose[:, :3] = (pose[:, :3] - center) / scale
    out[:POSE_DIM] = pose.flatten()

    # ── Face ─────────────────────────────────────────────
    face_start = POSE_DIM
    face_end = face_start + FACE_DIM
    face = out[face_start:face_end].reshape(FACE_LANDMARKS, 3)
    face = (face - center) / scale
    out[face_start:face_end] = face.flatten()

    # ── Left hand ────────────────────────────────────────
    lh_start = face_end
    lh_end = lh_start + HAND_DIM
    lh = out[lh_start:lh_end].reshape(HAND_LANDMARKS, 3)
    lh = (lh - center) / scale
    out[lh_start:lh_end] = lh.flatten()

    # ── Right hand ───────────────────────────────────────
    rh_start = lh_end
    rh_end = rh_start + HAND_DIM
    rh = out[rh_start:rh_end].reshape(HAND_LANDMARKS, 3)
    rh = (rh - center) / scale
    out[rh_start:rh_end] = rh.flatten()

    return out


def extract_keypoints(results, dtype=np.float32, normalize=True):
    """
    Extract, flatten, and optionally normalize keypoints from
    MediaPipe Holistic results.

    Returns:
        np.ndarray of shape (TOTAL_KEYPOINT_DIM,) — a 1-D vector of
        [pose | face | left_hand | right_hand] landmarks.
    """
    pose = np.zeros(POSE_DIM, dtype=dtype)
    face = np.zeros(FACE_DIM, dtype=dtype)
    lh = np.zeros(HAND_DIM, dtype=dtype)
    rh = np.zeros(HAND_DIM, dtype=dtype)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=dtype,
        ).flatten()

    if results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark],
            dtype=dtype,
        ).flatten()

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark],
            dtype=dtype,
        ).flatten()

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark],
            dtype=dtype,
        ).flatten()

    raw = np.concatenate([pose, face, lh, rh])

    if normalize and results.pose_landmarks:
        return normalize_keypoints(raw, dtype=dtype)
    return raw


def get_keypoint_info():
    """Print summary of the keypoint vector structure."""
    print("\nKeypoint Structure:")
    print(f"  Pose:  33×4 = {POSE_DIM}")
    print(f"  Face: 468×3 = {FACE_DIM}")
    print(f"  Hands: 21×3×2 = {HAND_DIM * 2}")
    print(f"  Total: {TOTAL_KEYPOINT_DIM} values/frame")
