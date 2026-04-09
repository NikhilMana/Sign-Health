import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def process_single_video(video_path, output_file):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        refine_face_landmarks=False
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Could not open video"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0 or fps <= 0:
        cap.release()
        return False, "Corrupted video"
    
    video_sequence = []
    frame_skip = max(1, int(fps / 10))
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Extract keypoints
            pose = np.zeros(33 * 4, dtype=np.float32)
            face = np.zeros(468 * 3, dtype=np.float32)
            lh = np.zeros(21 * 3, dtype=np.float32)
            rh = np.zeros(21 * 3, dtype=np.float32)
            if results.pose_landmarks: pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark[:33]], dtype=np.float32).flatten()
            if results.face_landmarks: face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark[:468]], dtype=np.float32).flatten()
            if results.left_hand_landmarks: lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark[:21]], dtype=np.float32).flatten()
            if results.right_hand_landmarks: rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark[:21]], dtype=np.float32).flatten()
            
            video_sequence.append(np.concatenate([pose, face, lh, rh]))
        
        frame_idx += 1
    
    cap.release()
    holistic.close()
    
    if len(video_sequence) == 0:
        return False, "No frames processed"
    
    np.save(output_file, np.array(video_sequence, dtype=np.float32))
    return True, f"Saved {len(video_sequence)} frames"

def run_priority():
    target_signs = ['Patient', 'sick', 'bad', 'alive', 'Hello']
    videos_folder = Path('videos')
    output_folder = Path('MP_DATA')
    
    for sign in target_signs:
        sign_dir = videos_folder / sign
        if not sign_dir.exists():
            print(f"Skipping {sign}: folder not found")
            continue
            
        out_dir = output_folder / sign
        out_dir.mkdir(exist_ok=True, parents=True)
        
        exts = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']
        files = []
        for e in exts: files.extend(sign_dir.glob(e))
        
        if not files:
            continue
            
        print(f"\nProcessing Priority Class: {sign} ({len(files)} videos)")
        for idx, f in enumerate(files):
            out_file = out_dir / f"{idx}.npy"
            if not out_file.exists():
                success, msg = process_single_video(f, out_file)
                print(f"  [{idx+1}/{len(files)}] {f.name}: {msg}")
            else:
                print(f"  [{idx+1}/{len(files)}] {f.name}: Already processed")

if __name__ == '__main__':
    run_priority()
    print("\nPriority extraction finished!")
