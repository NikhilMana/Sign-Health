"""Tests for the shared constants and keypoints modules."""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.constants import (
    POSE_DIM, FACE_DIM, HAND_DIM, TOTAL_KEYPOINT_DIM,
    SEQUENCE_WINDOW, CONFIDENCE_THRESHOLD,
)
from shared.keypoints import extract_keypoints


class TestConstants:
    def test_keypoint_dimensions(self):
        assert POSE_DIM == 33 * 4
        assert FACE_DIM == 468 * 3
        assert HAND_DIM == 21 * 3
        assert TOTAL_KEYPOINT_DIM == POSE_DIM + FACE_DIM + HAND_DIM * 2

    def test_total_dim_value(self):
        assert TOTAL_KEYPOINT_DIM == 1662

    def test_sequence_window_is_positive(self):
        assert SEQUENCE_WINDOW > 0

    def test_confidence_threshold_range(self):
        assert 0 < CONFIDENCE_THRESHOLD < 1


class TestExtractKeypoints:
    def _make_mock_results(self, has_pose=False, has_face=False, has_lh=False, has_rh=False):
        """Create a mock MediaPipe results object."""
        results = MagicMock()
        results.pose_landmarks = None
        results.face_landmarks = None
        results.left_hand_landmarks = None
        results.right_hand_landmarks = None

        if has_pose:
            lm = MagicMock()
            lm.x, lm.y, lm.z, lm.visibility = 0.5, 0.5, 0.0, 1.0
            results.pose_landmarks = MagicMock()
            results.pose_landmarks.landmark = [lm] * 33

        if has_face:
            lm = MagicMock()
            lm.x, lm.y, lm.z = 0.5, 0.5, 0.0
            results.face_landmarks = MagicMock()
            results.face_landmarks.landmark = [lm] * 468

        if has_lh:
            lm = MagicMock()
            lm.x, lm.y, lm.z = 0.3, 0.3, 0.0
            results.left_hand_landmarks = MagicMock()
            results.left_hand_landmarks.landmark = [lm] * 21

        if has_rh:
            lm = MagicMock()
            lm.x, lm.y, lm.z = 0.7, 0.7, 0.0
            results.right_hand_landmarks = MagicMock()
            results.right_hand_landmarks.landmark = [lm] * 21

        return results

    def test_empty_results_returns_zeros(self):
        results = self._make_mock_results()
        kp = extract_keypoints(results)
        assert kp.shape == (TOTAL_KEYPOINT_DIM,)
        assert np.all(kp == 0)

    def test_correct_shape_with_all_landmarks(self):
        results = self._make_mock_results(has_pose=True, has_face=True, has_lh=True, has_rh=True)
        kp = extract_keypoints(results)
        assert kp.shape == (TOTAL_KEYPOINT_DIM,)
        assert np.any(kp != 0)

    def test_pose_only(self):
        results = self._make_mock_results(has_pose=True)
        kp = extract_keypoints(results)
        assert np.any(kp[:POSE_DIM] != 0)
        assert np.all(kp[POSE_DIM:] == 0)

    def test_output_dtype(self):
        results = self._make_mock_results(has_pose=True)
        kp = extract_keypoints(results, dtype=np.float32)
        assert kp.dtype == np.float32
