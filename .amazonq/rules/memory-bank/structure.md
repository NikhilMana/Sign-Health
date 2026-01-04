# Project Structure

## Directory Organization

### Root Structure
```
sign_language_interpreter/
├── data_collection/          # Video processing and auto-labeling
├── model_training/           # LSTM model training scripts
├── real_time_inference/      # Live detection applications
├── webapp/                   # Telehealth web application
├── videos/                   # Raw input videos organized by sign
├── MP_DATA/                  # Extracted keypoint sequences (.npy files)
├── models/                   # Trained models and metadata
├── datasets/                 # Dataset management
├── downloaded_videos/        # Downloaded training videos
└── utility scripts           # Data analysis and pipeline tools
```

## Core Components

### 1. Data Collection Module (`data_collection/`)
**Purpose**: Process videos and extract sign language data

**Key Files**:
- `process_videos.py`: Extracts MediaPipe keypoints from videos
  - Uses multiprocessing for 6-12x speedup
  - Processes pose (33 landmarks) + hands (21 per hand)
  - Outputs .npy files with 1,662 features per frame
  - Skips already processed videos

**Data Flow**:
```
videos/ → process_videos.py → MP_DATA/
  ├── hello/video1.mp4          ├── hello/0.npy
  ├── hello/video2.mp4          ├── hello/1.npy
  └── thank_you/video1.mp4      └── thank_you/0.npy
```

### 2. Model Training Module (`model_training/`)
**Purpose**: Train LSTM models on extracted features

**Key Files**:
- `train_model.py`: Main training script
  - Loads sequences from MP_DATA/
  - Filters classes with <10 samples
  - Applies data augmentation (Gaussian noise)
  - Trains Bidirectional LSTM model
  - Saves model, encoder, and training history

**Outputs**:
- `models/sign_model.keras`: Trained model
- `models/best_model.keras`: Best checkpoint
- `models/label_encoder.pkl`: Class label encoder
- `models/max_length.txt`: Sequence length (60)
- `models/training_history.png`: Training curves

### 3. Real-Time Inference Module (`real_time_inference/`)
**Purpose**: Live sign language detection from webcam

**Key Files**:
- `detect_live.py`: Basic real-time detector
- `detect_live_improved.py`: Enhanced version with optimizations
  - Captures 20-frame sequences
  - Extracts keypoints using MediaPipe
  - Feeds to LSTM model
  - Uses voting from last 5 predictions
  - Displays prediction with confidence bar

**Performance**: 25-35 FPS, 300-500ms latency

### 4. Web Application Module (`webapp/`)
**Purpose**: Telehealth platform for doctor-patient communication

**Structure**:
```
webapp/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── models/
│   └── user.py              # User model and database
├── services/
│   ├── isl_detector.py      # ISL detection service
│   └── tts_service.py       # Text-to-speech service
├── routes/                   # HTTP route handlers
├── templates/                # HTML templates
│   ├── patient_dashboard.html
│   └── doctor_dashboard.html
├── static/
│   ├── css/style.css
│   └── js/
│       ├── patient.js       # Patient-side logic
│       └── doctor.js        # Doctor-side logic
└── database/
    └── telehealth.db        # SQLite database
```

**Architecture**:
- Flask backend with WebSocket support
- Role-based authentication (doctors vs patients)
- Real-time video frame processing
- Text-to-speech for detected signs
- Bidirectional messaging system

### 5. Data Storage

**videos/** - Raw input videos
- Organized by sign name (e.g., `videos/hello/`, `videos/thank_you/`)
- Supports MP4, AVI, MOV formats
- Minimum 5 videos per sign recommended

**MP_DATA/** - Processed keypoint sequences
- One .npy file per video
- Shape: (num_frames, 1662) - 1,662 features per frame
- Features: pose (132) + left hand (63) + right hand (63)

**models/** - Trained models and artifacts
- Keras model files (.keras)
- Label encoder (pickle)
- Sequence length metadata
- Training history visualizations

## Architectural Patterns

### Pipeline Architecture
```
1. DATA COLLECTION
   Video Input → auto_labeler.py → Labeled segments
   
2. FEATURE EXTRACTION
   Videos → process_videos.py → MediaPipe → .npy files
   
3. MODEL TRAINING
   .npy files → train_model.py → LSTM → Trained model
   
4. INFERENCE
   Webcam → detect_live.py → Model → Predictions
```

### Web Application Architecture
```
Patient Browser ←→ Flask Server ←→ Doctor Browser
     │                  │                │
     ├─ Video Frames ──▶│                │
     │                  ├─ ISL Detector  │
     │                  ├─ Translation ──▶│
     │◀─ Doctor Msg ────┤◀─ Text Msg ────┤
```

### Model Architecture Pattern
```
Input: (batch, 60, 1662)
  ↓
Bidirectional LSTM (64 units) → Dropout (0.3)
  ↓
Bidirectional LSTM (128 units) → Dropout (0.4)
  ↓
Dense (256, ReLU) → BatchNorm → Dropout (0.5)
  ↓
Dense (128, ReLU) → Dropout (0.4)
  ↓
Dense (num_classes, Softmax)
  ↓
Output: (batch, num_classes)
```

## Component Relationships

### Data Flow
1. **Collection**: Videos → Auto-labeler → Organized videos
2. **Processing**: Videos → MediaPipe → Keypoint sequences
3. **Training**: Sequences → LSTM → Trained model
4. **Inference**: Webcam → MediaPipe → Model → Predictions
5. **Web App**: Patient video → Server processing → Doctor interface

### Dependencies
- **Real-time inference** depends on trained model from training module
- **Web application** depends on trained model and detection services
- **Training** depends on processed data from collection module
- **All modules** depend on MediaPipe for keypoint extraction

## Utility Scripts

**Root-level tools**:
- `diagnose_data.py`: Analyze dataset distribution and statistics
- `evaluate_model.py`: Test model performance on validation data
- `filter_signs.py`: Filter dataset by sample count
- `auto_labeler.py`: Split videos by subtitle timestamps
- `run_full_pipeline.py`: Execute complete training pipeline
- `solution_1_reduce_classes.py`: Handle class imbalance
- `solution_2_add_idle_class.py`: Add idle/no-sign detection
- `test_auto_labeler.py`: Test auto-labeling functionality
- `test_idle_detection.py`: Test idle state detection

## Configuration Files

- `requirements.txt`: Main project dependencies
- `webapp/requirements_webapp.txt`: Web app specific dependencies
- `webapp/config.py`: Web application configuration
- `.gitignore`: Git ignore patterns
- `README.md`: Main project documentation
- `QUICK_START.md`: Quick start guide
- `TELEHEALTH_APP_PLAN.md`: Telehealth implementation plan
