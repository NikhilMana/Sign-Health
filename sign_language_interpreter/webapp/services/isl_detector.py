"""
Indian Sign Language (ISL) real-time detector service.

Uses MediaPipe Holistic for landmark extraction and a trained LSTM model
for sign classification.  Designed for integration with the Flask-SocketIO
webapp — each connected patient gets their own ``ISLDetector`` instance.
"""

import logging
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
import pickle
from collections import deque, Counter
from pathlib import Path

from shared.keypoints import extract_keypoints
from shared.constants import (
    SEQUENCE_WINDOW,
    CONFIDENCE_THRESHOLD,
    CONSENSUS_REQUIRED_LOOSE,
    PREDICTION_HISTORY_LOOSE,
    RESET_FRAMES,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
)

logger = logging.getLogger(__name__)


class ISLDetector:
    """Stateful per-session sign language detector."""

    def __init__(self, model_path, encoder_path, max_length_path):
        self.model = keras.models.load_model(model_path)

        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(max_length_path, "r") as f:
            self.max_length = int(f.read())

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            model_complexity=MP_MODEL_COMPLEXITY,
        )

        # Rolling buffers
        self.sequence = deque(maxlen=SEQUENCE_WINDOW)
        self.prediction_history = deque(maxlen=PREDICTION_HISTORY_LOOSE)

        # State
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.last_sent_prediction = ""
        self.frames_since_last_detection = 0

    # ── public API ───────────────────────────────

    def process_frame(self, frame):
        """
        Process a single BGR video frame.

        Returns:
            dict with keys ``text``, ``confidence``, ``detected``.
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True

        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)

        if len(self.sequence) == SEQUENCE_WINDOW:
            self._run_prediction()
        else:
            self.frames_since_last_detection += 1

        # Reset after prolonged silence
        if self.frames_since_last_detection > RESET_FRAMES:
            self.last_sent_prediction = ""
            self.frames_since_last_detection = 0

        # Only emit genuinely new predictions
        should_send = (
            self.current_prediction
            and self.current_prediction != self.last_sent_prediction
        )
        if should_send:
            self.last_sent_prediction = self.current_prediction

        return {
            "text": self.current_prediction if should_send else "",
            "confidence": float(self.prediction_confidence),
            "detected": len(self.sequence) == SEQUENCE_WINDOW,
        }

    def reset(self):
        """Clear all accumulated state for a fresh session."""
        self.sequence.clear()
        self.prediction_history.clear()
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.last_sent_prediction = ""
        self.frames_since_last_detection = 0

    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()

    # ── internals ────────────────────────────────

    def _run_prediction(self):
        seq_array = np.array(self.sequence)
        padded = np.zeros(
            (self.max_length, seq_array.shape[1]), dtype=np.float32
        )
        padded[: len(seq_array), :] = seq_array

        prediction = self.model.predict(
            np.expand_dims(padded, axis=0), verbose=0
        )
        idx = np.argmax(prediction[0])
        confidence = prediction[0][idx]

        if confidence > CONFIDENCE_THRESHOLD:
            label = self.label_encoder.inverse_transform([idx])[0]
            self.prediction_history.append(label)

            if len(self.prediction_history) >= CONSENSUS_REQUIRED_LOOSE:
                most_common = Counter(self.prediction_history).most_common(1)[0]
                if most_common[1] >= CONSENSUS_REQUIRED_LOOSE:
                    self.current_prediction = most_common[0]
                    self.prediction_confidence = confidence
                    self.frames_since_last_detection = 0
        else:
            self.prediction_confidence = confidence
            self.frames_since_last_detection += 1
