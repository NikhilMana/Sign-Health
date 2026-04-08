"""
Rich data augmentation pipeline for sign language keypoint sequences.

Generates diverse training variants to improve real-world robustness:
  - Temporal scaling (speed variation)
  - Spatial jitter (position/offset shifts)
  - Landmark dropout (simulate MediaPipe failures)
  - Gaussian noise (at multiple scales)
  - Hand mirroring (left ↔ right swap)
"""

import numpy as np
from shared.constants import (
    POSE_DIM, FACE_DIM, HAND_DIM,
    POSE_LANDMARKS, HAND_LANDMARKS,
)


def temporal_scale(sequence, factor):
    """Resample sequence to simulate faster/slower signing."""
    n_frames = max(2, int(len(sequence) * factor))
    indices = np.linspace(0, len(sequence) - 1, n_frames).astype(int)
    indices = np.clip(indices, 0, len(sequence) - 1)
    return sequence[indices]


def spatial_jitter(sequence, max_offset=0.05):
    """Shift all xyz coordinates by a small random offset."""
    offset = np.random.uniform(-max_offset, max_offset, size=3).astype(np.float32)
    shifted = sequence.copy()

    # Pose: shape per frame = (33, 4) — shift xyz columns 0-2
    for i in range(len(shifted)):
        pose = shifted[i, :POSE_DIM].reshape(POSE_LANDMARKS, 4)
        pose[:, :3] += offset
        shifted[i, :POSE_DIM] = pose.flatten()

        # Face, left hand, right hand: shape per frame = (N, 3)
        kp_start = POSE_DIM
        for dim in [FACE_DIM, HAND_DIM, HAND_DIM]:
            block = shifted[i, kp_start:kp_start + dim]
            block_3d = block.reshape(-1, 3) + offset
            shifted[i, kp_start:kp_start + dim] = block_3d.flatten()
            kp_start += dim

    return shifted


def landmark_dropout(sequence, drop_prob=0.3):
    """Randomly zero out one hand's landmarks to simulate MediaPipe failures."""
    dropped = sequence.copy()
    lh_start = POSE_DIM + FACE_DIM
    rh_start = lh_start + HAND_DIM

    for i in range(len(dropped)):
        if np.random.random() < drop_prob:
            # Randomly choose which hand to drop
            if np.random.random() < 0.5:
                dropped[i, lh_start:lh_start + HAND_DIM] = 0
            else:
                dropped[i, rh_start:rh_start + HAND_DIM] = 0

    return dropped


def gaussian_noise(sequence, sigma=0.02):
    """Add Gaussian noise at a meaningful scale."""
    noise = np.random.normal(0, sigma, sequence.shape).astype(np.float32)
    return sequence + noise


def mirror_hands(sequence):
    """Swap left and right hand keypoints (simulate left-handed signers)."""
    mirrored = sequence.copy()
    lh_start = POSE_DIM + FACE_DIM
    lh_end = lh_start + HAND_DIM
    rh_start = lh_end
    rh_end = rh_start + HAND_DIM

    lh_copy = mirrored[:, lh_start:lh_end].copy()
    mirrored[:, lh_start:lh_end] = mirrored[:, rh_start:rh_end]
    mirrored[:, rh_start:rh_end] = lh_copy

    return mirrored


# All available augmentation transforms as (name, callable) pairs
_AUGMENTATION_POOL = [
    lambda seq: temporal_scale(seq, 0.7),
    lambda seq: temporal_scale(seq, 0.85),
    lambda seq: temporal_scale(seq, 1.15),
    lambda seq: temporal_scale(seq, 1.3),
    lambda seq: spatial_jitter(seq, max_offset=0.04),
    lambda seq: spatial_jitter(seq, max_offset=0.04),
    lambda seq: landmark_dropout(seq, drop_prob=0.25),
    lambda seq: landmark_dropout(seq, drop_prob=0.25),
    lambda seq: gaussian_noise(seq, sigma=0.01),
    lambda seq: gaussian_noise(seq, sigma=0.02),
    lambda seq: gaussian_noise(seq, sigma=0.03),
    lambda seq: mirror_hands(seq),
]


def augment_sequence(sequence, n_variants=3):
    """
    Generate *n_variants* random augmented variants for a single sequence.

    Randomly samples from the augmentation pool to keep memory bounded
    while still providing diversity across the dataset.
    """
    chosen = np.random.choice(len(_AUGMENTATION_POOL), size=n_variants, replace=False)
    return [_AUGMENTATION_POOL[i](sequence) for i in chosen]


def augment_dataset(sequences, labels, n_variants=3):
    """
    Augment an entire dataset of sequences.

    Args:
        sequences: list of np.ndarray, each (T, features)
        labels: list of str, same length as sequences
        n_variants: augmented copies per original sample (default 3)

    Returns:
        (augmented_sequences, augmented_labels) — combined original + augmented
    """
    n_orig = len(sequences)
    print(f"\nRich augmentation: {n_orig} sequences × {n_variants} variants...")
    aug_sequences = list(sequences)
    aug_labels = list(labels)

    for seq, label in zip(sequences, labels):
        for variant in augment_sequence(seq, n_variants):
            aug_sequences.append(variant)
            aug_labels.append(label)

    print(f"  → {len(aug_sequences)} sequences "
          f"(×{len(aug_sequences) / n_orig:.1f})")
    return aug_sequences, aug_labels

