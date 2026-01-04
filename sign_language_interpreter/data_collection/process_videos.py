import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mproc

def process_single_video(video_path, output_file):
    """
    Process a single video file and extract keypoints.
    """
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
            video_sequence.append(extract_keypoints(results))
        
        frame_idx += 1
    
    cap.release()
    holistic.close()
    
    if len(video_sequence) == 0:
        return False, "No frames processed"
    
    np.save(output_file, np.array(video_sequence, dtype=np.float32))
    return True, f"Saved {len(video_sequence)} frames"

def process_videos_for_sign_language(use_multiprocessing=True):
    """
    Process videos for sign language recognition using MediaPipe Holistic.
    Extracts pose, face, and hand landmarks from each video frame and saves
    the keypoint sequences as .npy files.
    """
    videos_folder = Path("videos")
    output_folder = Path("MP_DATA")
    output_folder.mkdir(exist_ok=True)
    
    sign_labels = [d.name for d in videos_folder.iterdir() if d.is_dir()]
    print(f"Found {len(sign_labels)} sign labels")
    
    tasks = []
    for sign_label in sign_labels:
        sign_output_dir = output_folder / sign_label
        sign_output_dir.mkdir(exist_ok=True)
        
        sign_videos_dir = videos_folder / sign_label
        video_files = list(sign_videos_dir.glob("*.mp4"))
        
        for video_idx, video_path in enumerate(video_files):
            output_file = sign_output_dir / f"{video_idx}.npy"
            if not output_file.exists():
                tasks.append((video_path, output_file, sign_label))
    
    print(f"Processing {len(tasks)} videos...")
    
    if use_multiprocessing and len(tasks) > 1:
        max_workers = min(mproc.cpu_count() - 1, 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_video, vp, of): (vp, sl) for vp, of, sl in tasks}
            
            for i, future in enumerate(as_completed(futures), 1):
                video_path, sign_label = futures[future]
                success, msg = future.result()
                status = "✓" if success else "✗"
                print(f"[{i}/{len(tasks)}] {status} {sign_label}/{video_path.name}: {msg}")
    else:
        for i, (video_path, output_file, sign_label) in enumerate(tasks, 1):
            success, msg = process_single_video(video_path, output_file)
            status = "✓" if success else "✗"
            print(f"[{i}/{len(tasks)}] {status} {sign_label}/{video_path.name}: {msg}")
    
    print("\nVideo processing completed!")

def extract_keypoints(results):
    """
    Extract keypoint coordinates from MediaPipe Holistic results.
    Returns a flattened numpy array of all keypoints.
    """
    pose = np.zeros(33 * 4, dtype=np.float32)
    face = np.zeros(468 * 3, dtype=np.float32)
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)
    
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark[:33]], dtype=np.float32).flatten()
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark[:468]], dtype=np.float32).flatten()
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark[:21]], dtype=np.float32).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark[:21]], dtype=np.float32).flatten()
    
    return np.concatenate([pose, face, lh, rh])

def get_keypoint_info():
    """
    Print information about the keypoint structure.
    """
    print("\nKeypoint Structure:")
    print(f"Pose: 33×4 = {33*4} | Face: 468×3 = {468*3}")
    print(f"Hands: 21×3×2 = {21*3*2} | Total: {33*4+468*3+21*3*2} values/frame")

if __name__ == "__main__":
    print("Sign Language Video Processing Script")
    print("=" * 50)
    get_keypoint_info()
    process_videos_for_sign_language(use_multiprocessing=True)
