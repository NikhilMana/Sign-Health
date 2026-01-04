import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
import pickle
from collections import deque, Counter
from pathlib import Path

class ISLDetector:
    def __init__(self, model_path, encoder_path, max_length_path):
        self.model = keras.models.load_model(model_path)
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open(max_length_path, 'r') as f:
            self.max_length = int(f.read())
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        self.sequence = deque(maxlen=20)
        self.prediction_history = deque(maxlen=5)
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.PREDICTION_THRESHOLD = 0.7
    
    def extract_keypoints(self, results):
        pose = np.zeros(33 * 4)
        face = np.zeros(468 * 3)
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] 
                           for res in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[res.x, res.y, res.z] 
                           for res in results.face_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] 
                         for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] 
                         for res in results.right_hand_landmarks.landmark]).flatten()

        return np.concatenate([pose, face, lh, rh])
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        
        if len(self.sequence) == 20:
            seq_array = np.array(self.sequence)
            padded_sequence = np.zeros((self.max_length, seq_array.shape[1]), dtype=np.float32)
            padded_sequence[:len(seq_array), :] = seq_array
            
            prediction = self.model.predict(np.expand_dims(padded_sequence, axis=0), verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]

            if confidence > self.PREDICTION_THRESHOLD:
                pred_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
                self.prediction_history.append(pred_label)
                
                if len(self.prediction_history) >= 3:
                    most_common = Counter(self.prediction_history).most_common(1)[0]
                    if most_common[1] >= 2:
                        self.current_prediction = most_common[0]
                        self.prediction_confidence = confidence
            else:
                self.prediction_confidence = confidence
        
        return {
            'text': self.current_prediction,
            'confidence': float(self.prediction_confidence),
            'detected': len(self.sequence) == 20
        }
    
    def reset(self):
        self.sequence.clear()
        self.prediction_history.clear()
        self.current_prediction = ""
        self.prediction_confidence = 0.0
    
    def close(self):
        self.holistic.close()
