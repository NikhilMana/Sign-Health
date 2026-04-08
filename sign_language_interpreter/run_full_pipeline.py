"""
Full ISL Interpreter Training Pipeline.

Orchestrates feature extraction and model training in sequence.
"""

import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_videos_folder():
    """Verify that the videos directory contains sign-language clips."""
    videos_path = Path("videos")
    if not videos_path.exists():
        print("❌ Error: 'videos' folder not found!")
        print("Run auto_labeler.py first to create labeled video clips.")
        return False

    sign_folders = [d for d in videos_path.iterdir() if d.is_dir()]
    if not sign_folders:
        print("❌ Error: No sign folders found in 'videos' directory!")
        return False

    total_videos = sum(len(list(f.glob("*.mp4"))) for f in sign_folders)
    print(f"✓ Found {len(sign_folders)} sign classes with {total_videos} total videos")
    return True


def run_feature_extraction():
    """Step 1 — Extract MediaPipe keypoints from video files."""
    print("\n" + "=" * 60)
    print("STEP 1: FEATURE EXTRACTION")
    print("=" * 60)

    try:
        from data_collection.process_videos import process_videos_for_sign_language

        start = time.time()
        process_videos_for_sign_language(use_multiprocessing=True)
        print(f"\n✓ Feature extraction completed in {(time.time() - start) / 60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_training():
    """Step 2 — Train the LSTM model."""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    try:
        from model_training.train_model import train_model

        start = time.time()
        model, label_encoder, history = train_model()
        print(f"\n✓ Model training completed in {(time.time() - start) / 60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete pipeline: extract → train."""
    print("=" * 60)
    print("ISL INTERPRETER — FULL TRAINING PIPELINE")
    print("=" * 60)

    if not check_videos_folder():
        return

    print("\nThis will:")
    print("  1. Extract features from all videos  → MP_DATA/")
    print("  2. Train the LSTM model              → models/sign_model.keras")
    print("\nEstimated time: 20–60 minutes depending on dataset size")

    response = input("\nProceed? (y/n): ").strip().lower()
    if response != "y":
        print("Cancelled.")
        return

    total_start = time.time()

    if not run_feature_extraction():
        print("\n❌ Pipeline failed at feature extraction step")
        return

    if not run_model_training():
        print("\n❌ Pipeline failed at model training step")
        return

    total_time = (time.time() - total_start) / 60
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} minutes")
    print("\nNext steps:")
    print("  1. Check models/training_history.png for training curves")
    print("  2. Run real_time_inference/detect_live_improved.py to test")
    print("  3. Launch the webapp:  cd webapp && python app.py")


if __name__ == "__main__":
    import os
    os.chdir(PROJECT_ROOT)
    main()
