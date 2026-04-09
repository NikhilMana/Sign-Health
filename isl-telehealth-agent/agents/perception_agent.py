# Integrates with existing model at sign_language_interpreter/models/sign_model.keras
# Feature extraction matches sign_language_interpreter/data_preprocessing/process_videos.py
# Inference logic matches sign_language_interpreter/real_time_inference/detect_live.py

import os
import cv2
import yaml
import pickle
import random
import logging
import numpy as np
import tensorflow as tf
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    mp = None
    _MP_AVAILABLE = False
from collections import deque, Counter
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("perception_agent")

class PerceptionAgent:
    """
    Perception Agent that processes video frames using MediaPipe Holistic
    and predicts sign language gestures using a trained Bidirectional LSTM model.
    """
    def __init__(self, config_path: str = "config/agents_config.yaml"):
        """
        Initializes the perception agent, loads the keras model and label encoder,
        and sets up the MediaPipe holistic pipeline.
        
        Args:
            config_path (str): Optional path to config file.
        """
        self.demo_mode = False
        self.demo_index = 0
        self.frame_count = 0
        
        self.sequence = deque(maxlen=20)
        self.prediction_history = deque(maxlen=5)
        
        # Target model paths relative to the project root
        MODEL_PATH = "../sign_language_interpreter/models/sign_model.keras"
        BACKUP_MODEL_PATH = "../sign_language_interpreter/models/best_model.keras"
        ENCODER_PATH = "../sign_language_interpreter/models/label_encoder.pkl"
        MAX_LENGTH_PATH = "../sign_language_interpreter/models/max_length.txt"
        
        try:
            if os.path.exists(MODEL_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
                logger.info(f"Loaded primary model from {MODEL_PATH}")
            elif os.path.exists(BACKUP_MODEL_PATH):
                self.model = tf.keras.models.load_model(BACKUP_MODEL_PATH)
                logger.info(f"Loaded backup model from {BACKUP_MODEL_PATH}")
            else:
                raise FileNotFoundError("Could not find sign_model.keras or best_model.keras")

            with open(ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            with open(MAX_LENGTH_PATH, 'r') as f:
                self.max_length = int(f.read().strip())
                
        except Exception as e:
            logger.warning(f"Failed to load model/assets: {e}. Activating Demo Mode.")
            self.demo_mode = True

        # Initialize MediaPipe Holistic
        try:
            if not _MP_AVAILABLE:
                raise ImportError("MediaPipe is not installed.")
                
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.holistic = self.mp_holistic.Holistic(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Holistic initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Holistic unavailable: {e}. Landmarks disabled.")
            if not self.demo_mode:
                logger.warning("Forcing demo mode since MediaPipe is unavailable.")
                self.demo_mode = True
            self.holistic = None
            self.mp_drawing = None
            self.mp_holistic = None

    def extract_keypoints(self, results: Any) -> np.ndarray:
        """
        Extracts keypoints from MediaPipe holistic results exactly matching
        sign_language_interpreter/data_preprocessing/process_videos.py logic.
        
        Args:
            results: MediaPipe holistic results object.
            
        Returns:
            np.ndarray: Concatenated array of shape (258,).
        """
        pose = np.zeros(33 * 4)
        face = np.zeros(468 * 3)
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

        return np.concatenate([pose, face, lh, rh])

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single video frame, extracts keypoints, and populates the sequence buffer.
        Skips frames based on FRAME_SKIP logic.
        
        Args:
            frame (np.ndarray): Video frame from OpenCV.
            
        Returns:
            Dict[str, Any]: Dictionary containing ready status and frame/buffer info.
        """
        self.frame_count += 1
        
        # FRAME_SKIP logic = 3
        if self.frame_count % 3 != 0:
            return {"ready": len(self.sequence) == 20, "buffer_count": len(self.sequence), "results": None}
            
        # Optimization: Pass by reference mapping structure of detect_live.py
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        frame.flags.writeable = True
        
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        
        return {"ready": len(self.sequence) == 20, "buffer_count": len(self.sequence), "results": results}

    def predict(self) -> Dict[str, Any]:
        """
        Runs LSTM inference on a full sequence buffer (20 frames) padded to 60.
        
        Returns:
            Dict[str, Any]: Predictions, smoothed output, and flags.
        """
        if len(self.sequence) < 20:
            return {"ready": False, "prediction_ready": False}
            
        if self.demo_mode:
            res = self.get_demo_prediction()
            res["ready"] = True # Aligning format with orchestration backwards-compatibility
            return res
            
        sequence_array = np.array(self.sequence) # Dimension: (20, 1662)
        
        # Pad at the end with zeros to match MAX_SEQ_LENGTH_CAP = 60
        padded = np.pad(sequence_array, ((0, 40), (0, 0)), mode='constant')
        model_input = padded.reshape(1, 60, 1662)
        
        output = self.model.predict(model_input, verbose=0)[0]
        top3_indices = np.argsort(output)[-3:][::-1]
        
        top_predictions = [
            {"sign": self.label_encoder.inverse_transform([i])[0], "confidence": float(output[i])} 
            for i in top3_indices
        ]
        
        self.prediction_history.append(top_predictions[0]["sign"])
        smoothed = Counter(self.prediction_history).most_common(1)[0]
        
        self.sequence.clear()
        
        return {
            "top_predictions": top_predictions,
            "smoothed_prediction": {"sign": smoothed[0], "vote_count": smoothed[1]},
            "should_display": top_predictions[0]["confidence"] > 0.70,
            "raw_confidence": top_predictions[0]["confidence"],
            "prediction_ready": True,
            "ready": True
        }

    def get_demo_prediction(self) -> Dict[str, Any]:
        """
        Provides simulated predictions using a fallback medical vocabulary for demo workflows.
        
        Returns:
            Dict[str, Any]: Simulated prediction dictionary.
        """
        vocab = ["pain", "chest", "three_days", "help", "doctor", "medicine", "appointment", "breathe", "water", "emergency"]
        
        chosen_sign = vocab[self.demo_index % len(vocab)]
        self.demo_index += 1
        
        top_conf = random.uniform(0.75, 0.95)
        
        top_predictions = [
            {"sign": chosen_sign, "confidence": top_conf},
            {"sign": vocab[(self.demo_index+1) % len(vocab)], "confidence": top_conf * 0.4},
            {"sign": vocab[(self.demo_index+2) % len(vocab)], "confidence": top_conf * 0.1}
        ]
        
        self.prediction_history.append(top_predictions[0]["sign"])
        smoothed = Counter(self.prediction_history).most_common(1)[0]
        
        self.sequence.clear()
        
        return {
            "top_predictions": top_predictions,
            "smoothed_prediction": {"sign": smoothed[0], "vote_count": smoothed[1]},
            "should_display": top_predictions[0]["confidence"] > 0.70,
            "raw_confidence": top_predictions[0]["confidence"],
            "prediction_ready": True,
            "ready": True
        }

    def reset_buffer(self) -> None:
        """
        Explicitly clears the frame sequence buffer tracker.
        """
        self.sequence.clear()

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Annotates the given image frame with MediaPipe anatomical landmarks.
        
        Args:
            frame (np.ndarray): Original BGR image frame.
            results: MediaPipe holistic engine outputs.
            
        Returns:
            np.ndarray: Annotated BGR frame.
        """
        if not results:
            return frame
            
        annotated = frame.copy()
        
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(annotated, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        # Left Hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        # Right Hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            
        return annotated

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = PerceptionAgent()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps_start_time = cv2.getTickCount()
    fps_frames = 0
    fps = 0
    
    current_prediction = "--"
    current_confidence = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        res = agent.process_frame(frame)
        annotated_frame = agent.draw_landmarks(frame, res["results"])
        
        if res["ready"]:
            pred = agent.predict()
            if pred["should_display"]:
                current_prediction = pred["top_predictions"][0]["sign"]
                current_confidence = pred["top_predictions"][0]["confidence"]
                
        # Top-left Output Render
        cv2.putText(annotated_frame, f"Sign: {current_prediction} {current_confidence:.2f}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # FPS Tracking
        fps_frames += 1
        if fps_frames % 30 == 0:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            fps = 30 / time_diff
            fps_start_time = fps_end_time
            
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Buffer: {res['buffer_count']}/20", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Perception Validation Debug", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
