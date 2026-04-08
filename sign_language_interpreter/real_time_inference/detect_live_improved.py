"""
Improved real-time ISL detection with movement filtering and strict consensus.

Run directly:
    python detect_live_improved.py
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from pathlib import Path
import pickle
from collections import deque, Counter

# Ensure project root is on path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.keypoints import extract_keypoints
from shared.constants import (
    SEQUENCE_WINDOW,
    CONFIDENCE_THRESHOLD_STRICT,
    CONSENSUS_REQUIRED_STRICT,
    PREDICTION_HISTORY_STRICT,
    MOVEMENT_THRESHOLD,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    GPU_MEMORY_GROWTH,
)
from shared.device_config import DeviceManager

# ── Hardware setup (memory growth only — no mixed precision for inference)
_device_mgr = DeviceManager(
    enable_mixed_precision=False,
    enable_xla=False,
    enable_memory_growth=GPU_MEMORY_GROWTH,
)
_device_mgr.initialize()


# ── Utility functions ──────────────────────────


def calculate_movement(sequence):
    """Calculate mean absolute movement across a keypoint sequence."""
    if len(sequence) < 2:
        return 0
    diffs = np.diff(np.array(sequence), axis=0)
    return np.mean(np.abs(diffs))


def draw_styled_landmarks(image, results):
    """Overlay pose and hand landmarks on the frame."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    pose_spec = mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2)
    hand_spec = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3)
    conn_spec = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_spec, pose_spec)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_spec, conn_spec)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_spec, conn_spec)


def draw_ui(image, prediction, confidence, threshold, movement, fps=0):
    """Draw a HUD overlay with detection status, confidence bar, and FPS."""
    height, width, _ = image.shape
    bar_height = 100

    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    # Status text
    if movement < MOVEMENT_THRESHOLD:
        text_color, display_text = (100, 100, 100), "NO MOVEMENT"
        status = "idle"
    elif prediction == "IDLE_NO_SIGN":
        text_color, display_text = (100, 100, 100), "IDLE"
        status = "idle"
    elif confidence > threshold:
        text_color, display_text = (0, 255, 0), prediction.upper()
        status = "detected"
    else:
        text_color, display_text = (200, 200, 200), "DETECTING..."
        status = "detecting"

    cv2.putText(image, display_text, (10, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
    cv2.putText(image, f"Confidence: {confidence:.1%}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"Movement: {movement:.4f}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if status != "idle":
        bar_width = int(confidence * (width - 20))
        bar_color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
        cv2.rectangle(image, (10, height - 75), (10 + bar_width, height - 70), bar_color, -1)

    cv2.putText(image, f"FPS: {fps:.1f} | Press 'q' to quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


# ── Main loop ──────────────────────────────────


def main():
    print("Starting IMPROVED sign language detection...")
    print(f"  • Device: {_device_mgr.device_tag}")
    print("  • Movement detection (no prediction when idle)")
    print(f"  • Confidence threshold: {CONFIDENCE_THRESHOLD_STRICT}")
    print(f"  • Consensus required: {CONSENSUS_REQUIRED_STRICT}/{PREDICTION_HISTORY_STRICT}\n")

    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / "sign_model.keras"
    encoder_path = models_dir / "label_encoder.pkl"
    max_length_path = models_dir / "max_length.txt"

    if not all(p.exists() for p in [model_path, encoder_path, max_length_path]):
        print("Error: Model files not found in", models_dir)
        return

    try:
        model = keras.models.load_model(model_path)
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        with open(max_length_path, "r") as f:
            max_length = int(f.read())
        print(f"Model loaded — {len(label_encoder.classes_)} classes\n")
    except Exception as e:
        print(f"Error loading model files: {e}")
        return

    # MediaPipe setup
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        model_complexity=MP_MODEL_COMPLEXITY,
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sequence = deque(maxlen=SEQUENCE_WINDOW)
    prediction_history = deque(maxlen=PREDICTION_HISTORY_STRICT)
    current_prediction = ""
    prediction_confidence = 0.0
    frame_count = 0
    fps = 0
    start_time = time.time()

    print("Detection started — move your hands to sign.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - start_time)
            start_time = time.time()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        movement = calculate_movement(sequence) if len(sequence) >= 2 else 0

        if len(sequence) == SEQUENCE_WINDOW and movement > MOVEMENT_THRESHOLD:
            seq_array = np.array(sequence)
            padded = np.zeros((max_length, seq_array.shape[1]), dtype=np.float32)
            padded[: len(seq_array), :] = seq_array

            prediction = model.predict(np.expand_dims(padded, axis=0), verbose=0)
            idx = np.argmax(prediction[0])
            confidence = prediction[0][idx]

            if confidence > CONFIDENCE_THRESHOLD_STRICT:
                pred_label = label_encoder.inverse_transform([idx])[0]
                if pred_label != "IDLE_NO_SIGN":
                    prediction_history.append(pred_label)
                    if len(prediction_history) >= CONSENSUS_REQUIRED_STRICT:
                        most_common = Counter(prediction_history).most_common(1)[0]
                        if most_common[1] >= CONSENSUS_REQUIRED_STRICT:
                            current_prediction = most_common[0]
                            prediction_confidence = confidence
                else:
                    current_prediction = "IDLE_NO_SIGN"
                    prediction_confidence = confidence
            else:
                prediction_confidence = confidence
        elif movement <= MOVEMENT_THRESHOLD:
            current_prediction = ""
            prediction_confidence = 0.0
            prediction_history.clear()

        image = draw_ui(image, current_prediction, prediction_confidence, CONFIDENCE_THRESHOLD_STRICT, movement, fps)
        cv2.imshow("ISL Interpreter — IMPROVED", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Detection stopped.")


if __name__ == "__main__":
    main()
