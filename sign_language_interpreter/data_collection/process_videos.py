"""
Video processing pipeline for sign language recognition.

Extracts MediaPipe Holistic keypoints from video files and saves
them as ``.npy`` sequences in the ``MP_DATA`` directory.
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mproc

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.keypoints import extract_keypoints, get_keypoint_info
from shared.constants import MP_MODEL_COMPLEXITY

# ── Worker-level state (reused across videos in the same process) ──

_worker_holistic = None


def _init_worker():
    """Initialize MediaPipe Holistic once per worker process."""
    global _worker_holistic
    _worker_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )


def process_single_video(video_path, output_file):
    """
    Extract keypoints from a single video file.

    Uses the worker-level Holistic instance when running in a process pool,
    or creates a fresh one for single-threaded execution.
    """
    global _worker_holistic

    # Fallback: create holistic if not inside a process pool
    holistic = _worker_holistic
    own_holistic = False
    if holistic is None:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )
        own_holistic = True

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

    if own_holistic:
        holistic.close()

    if len(video_sequence) == 0:
        return False, "No frames processed"

    np.save(output_file, np.array(video_sequence, dtype=np.float32))
    return True, f"Saved {len(video_sequence)} frames"


def process_videos_for_sign_language(use_multiprocessing=True):
    """
    Batch-process all sign-language videos in ``videos/`` and save keypoint
    sequences to ``MP_DATA/``.
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

        for video_idx, video_path in enumerate((videos_folder / sign_label).glob("*.mp4")):
            output_file = sign_output_dir / f"{video_idx}.npy"
            if not output_file.exists():
                tasks.append((video_path, output_file, sign_label))

    print(f"Processing {len(tasks)} videos...")

    if use_multiprocessing and len(tasks) > 1:
        max_workers = min(mproc.cpu_count() - 1, 4)
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
            futures = {
                executor.submit(process_single_video, vp, of): (vp, sl)
                for vp, of, sl in tasks
            }
            for i, future in enumerate(as_completed(futures), 1):
                video_path, sign_label = futures[future]
                success, msg = future.result()
                icon = "✓" if success else "✗"
                print(f"[{i}/{len(tasks)}] {icon} {sign_label}/{video_path.name}: {msg}")
    else:
        for i, (video_path, output_file, sign_label) in enumerate(tasks, 1):
            success, msg = process_single_video(video_path, output_file)
            icon = "✓" if success else "✗"
            print(f"[{i}/{len(tasks)}] {icon} {sign_label}/{video_path.name}: {msg}")

    print("\nVideo processing completed!")


if __name__ == "__main__":
    print("Sign Language Video Processing Script")
    print("=" * 50)
    get_keypoint_info()
    process_videos_for_sign_language(use_multiprocessing=True)
