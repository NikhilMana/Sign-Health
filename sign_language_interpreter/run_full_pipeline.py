"""
Full ISL Interpreter Training Pipeline
Runs feature extraction and model training automatically
"""
import os
import sys
from pathlib import Path
import time

def check_videos_folder():
    """Check if videos folder exists and has content"""
    videos_path = Path("videos")
    if not videos_path.exists():
        print("❌ Error: 'videos' folder not found!")
        print("Run auto_labeler.py first to create labeled video clips.")
        return False
    
    sign_folders = [d for d in videos_path.iterdir() if d.is_dir()]
    if not sign_folders:
        print("❌ Error: No sign folders found in 'videos' directory!")
        return False
    
    total_videos = sum(len(list(folder.glob("*.mp4"))) for folder in sign_folders)
    print(f"✓ Found {len(sign_folders)} sign classes with {total_videos} total videos")
    return True

def run_feature_extraction():
    """Run the video processing script"""
    print("\n" + "="*60)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*60)
    
    sys.path.insert(0, str(Path("data_collection")))
    
    try:
        from data_collection.process_videos import process_videos_for_sign_language
        
        start_time = time.time()
        process_videos_for_sign_language(use_multiprocessing=True)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Feature extraction completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_model_training():
    """Run the model training script"""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    sys.path.insert(0, str(Path("model_training")))
    
    try:
        from model_training.train_model import train_model
        
        start_time = time.time()
        model, label_encoder, history = train_model()
        elapsed = time.time() - start_time
        
        print(f"\n✓ Model training completed in {elapsed/60:.1f} minutes")
        return True
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete pipeline"""
    print("="*60)
    print("ISL INTERPRETER - FULL TRAINING PIPELINE")
    print("="*60)
    
    # Check prerequisites
    if not check_videos_folder():
        return
    
    # Confirm before starting
    print("\nThis will:")
    print("1. Extract features from all videos (creates MP_DATA folder)")
    print("2. Train the LSTM model (creates models/sign_model.keras)")
    print("\nEstimated time: 20-60 minutes depending on dataset size")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    start_time = time.time()
    
    # Step 1: Feature Extraction
    if not run_feature_extraction():
        print("\n❌ Pipeline failed at feature extraction step")
        return
    
    # Step 2: Model Training
    if not run_model_training():
        print("\n❌ Pipeline failed at model training step")
        return
    
    # Success!
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print("\nNext steps:")
    print("1. Check models/training_history.png for training curves")
    print("2. Run real_time_inference/detect_live.py to test your model")
    print("3. Use your webcam to recognize ISL signs in real-time!")

if __name__ == "__main__":
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()
