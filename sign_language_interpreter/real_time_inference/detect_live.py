import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from pathlib import Path
import pickle
from collections import deque, Counter

def extract_keypoints(results):
    """
    Extracts pose, face, and hand keypoints from MediaPipe results.
    Returns a flattened numpy array of all keypoints, ensuring consistent size.
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

def draw_styled_landmarks(image, results):
    """
    Draw landmarks and connections with custom styling.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    drawing_spec_pose = mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2)
    drawing_spec_hands = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3)
    drawing_spec_hand_connections = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, drawing_spec_pose, drawing_spec_pose)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_spec_hands, drawing_spec_hand_connections)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_spec_hands, drawing_spec_hand_connections)

def draw_ui(image, prediction, confidence, threshold, fps=0):
    """
    Draws the UI elements on the screen, including a status bar and confidence bar.
    """
    height, width, _ = image.shape
    bar_height = 80
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    if confidence > threshold:
        text_color = (0, 255, 0)
        display_text = f"{prediction.upper()}"
    else:
        text_color = (200, 200, 200)
        display_text = "Detecting..."
    
    cv2.putText(image, display_text, (10, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
    cv2.putText(image, f"{confidence:.1%}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    bar_width = int(confidence * (width - 20))
    cv2.rectangle(image, (10, height - 55), (10 + bar_width, height - 50), (0, 255, 0), -1)
    
    cv2.putText(image, f"FPS: {fps:.1f} | Press 'q' to quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def main():
    """
    Main function to run the real-time sign language detection.
    """
    print("Starting real-time sign language detection...")

    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "sign_model.keras"
    encoder_path = models_dir / "label_encoder.pkl"
    max_length_path = models_dir / "max_length.txt"

    if not all([model_path.exists(), encoder_path.exists(), max_length_path.exists()]):
        print("Error: Model, label encoder, or max_length file not found.")
        return

    try:
        model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(max_length_path, 'r') as f:
            max_length = int(f.read())
        print("Files loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sequence = deque(maxlen=20)
    prediction_history = deque(maxlen=5)
    current_prediction = ""
    prediction_confidence = 0.0
    PREDICTION_THRESHOLD = 0.7
    frame_count = 0
    fps = 0
    import time
    start_time = time.time()

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

        if len(sequence) == 20:
            seq_array = np.array(sequence)
            padded_sequence = np.zeros((max_length, seq_array.shape[1]), dtype=np.float32)
            padded_sequence[:len(seq_array), :] = seq_array
            
            prediction = model.predict(np.expand_dims(padded_sequence, axis=0), verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]

            if confidence > PREDICTION_THRESHOLD:
                pred_label = label_encoder.inverse_transform([predicted_class_idx])[0]
                prediction_history.append(pred_label)
                
                if len(prediction_history) >= 3:
                    most_common = Counter(prediction_history).most_common(1)[0]
                    if most_common[1] >= 2:
                        current_prediction = most_common[0]
                        prediction_confidence = confidence
            else:
                prediction_confidence = confidence

        image = draw_ui(image, current_prediction, prediction_confidence, PREDICTION_THRESHOLD, fps)
        cv2.imshow('ISL Interpreter', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Detection stopped.")

if __name__ == "__main__":
    main()