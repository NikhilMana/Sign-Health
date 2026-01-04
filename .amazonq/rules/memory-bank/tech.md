# Technology Stack

## Programming Languages

### Python 3.8+
Primary language for entire project including:
- Deep learning model training
- Computer vision processing
- Web application backend
- Data processing pipelines
- Real-time inference

### JavaScript
Client-side logic for web application:
- WebSocket communication
- Video capture and streaming
- Real-time UI updates
- Patient and doctor dashboards

### HTML/CSS
Web application frontend:
- Responsive templates
- Dashboard interfaces
- Form handling

## Core Dependencies

### Deep Learning & ML
```
tensorflow>=2.15.0          # Deep learning framework, LSTM models
scikit-learn>=1.3.0         # Label encoding, train/test split
numpy>=1.24.0               # Numerical operations, array handling
```

### Computer Vision
```
opencv-python>=4.8.0        # Video capture, frame processing
mediapipe>=0.10.0           # Pose and hand keypoint extraction
```

### Data Processing & Visualization
```
pandas>=2.1.0               # Data manipulation and analysis
matplotlib>=3.8.0           # Training history plots
seaborn>=0.13.0             # Statistical visualizations
```

### Web Application
```
Flask>=2.3.0                # Web framework
Flask-SocketIO>=5.3.0       # WebSocket support
Flask-Login>=0.6.0          # User authentication
python-socketio>=5.9.0      # Socket.IO server
eventlet>=0.33.0            # Async networking
bcrypt>=4.0.0               # Password hashing
```

### Video Processing
```
yt-dlp                      # Video downloading
ffmpeg-python               # Video manipulation
```

## Build System

### Package Management
- **pip**: Python package installer
- **venv**: Virtual environment management

### Installation Commands
```bash
# Main project
pip install -r requirements.txt

# Web application
pip install -r webapp/requirements_webapp.txt
```

## Development Commands

### Data Processing
```bash
# Extract features from videos
cd data_collection
python process_videos.py

# Auto-label videos from subtitles
python auto_labeler.py

# Diagnose dataset
python diagnose_data.py
```

### Model Training
```bash
# Train LSTM model
cd model_training
python train_model.py

# Evaluate model
python evaluate_model.py
```

### Real-Time Inference
```bash
# Run live detection
cd real_time_inference
python detect_live.py

# Run improved version
python detect_live_improved.py
```

### Web Application
```bash
# Start Flask server
cd webapp
python app.py

# Verify setup
python verify_setup.py
```

### Full Pipeline
```bash
# Run complete pipeline
python run_full_pipeline.py
```

## System Requirements

### Hardware
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU optional (10-50x faster training)
- **Webcam**: Required for real-time detection
- **Storage**: 5GB+ for datasets and models

### Software
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.2+ (optional, for GPU acceleration)
- **Browser**: Chrome/Firefox (for web app)

## Key Technologies Explained

### TensorFlow 2.15+
- Deep learning framework for LSTM model
- Keras API for model building
- GPU acceleration support via CUDA
- Model serialization (.keras format)
- Training callbacks (EarlyStopping, ModelCheckpoint)

### MediaPipe 0.10+
- Google's ML solution for pose estimation
- Holistic model extracts:
  - 33 pose landmarks (x, y, z, visibility)
  - 21 left hand landmarks (x, y, z)
  - 21 right hand landmarks (x, y, z)
- Real-time performance (30+ FPS)
- Model complexity levels: 0 (fast), 1 (balanced), 2 (accurate)

### OpenCV 4.8+
- Video capture from webcam
- Frame processing and manipulation
- Video file reading/writing
- Image display and UI rendering
- Resolution: 640x480 default

### Flask + SocketIO
- Web framework for HTTP routes
- WebSocket support for real-time communication
- Session management with Flask-Login
- Event-driven architecture
- Supports multiple concurrent connections

### NumPy 1.24+
- Array operations for keypoint data
- Shape: (num_frames, 1662) for sequences
- Data type: float32 for memory efficiency
- Statistical operations (mean, std, etc.)

## Model Architecture Details

### LSTM Configuration
```python
# Bidirectional LSTM layers
Layer 1: 64 units, return_sequences=True
Layer 2: 128 units, return_sequences=False

# Regularization
Dropout: 0.3, 0.4, 0.5
L2 regularization: 0.001

# Optimization
Optimizer: Adam (lr=0.001)
Loss: Categorical crossentropy
Metrics: Accuracy
```

### Training Configuration
```python
Epochs: 150 (with early stopping)
Batch size: 64
Sequence length: 60 frames (capped)
Validation split: 20%
Early stopping patience: 15 epochs
LR reduction patience: 7 epochs
LR reduction factor: 0.5
```

### Data Augmentation
```python
# Applied when dataset < 1000 samples
Gaussian noise: mean=0, std=0.005
Augmentation factor: 3x (2 augmented per original)
```

## Performance Optimizations

### Multiprocessing
- Parallel video processing with 4 workers
- 6-12x speedup in feature extraction
- Process pool for CPU-bound tasks

### Frame Skipping
- Process every 3rd frame (10 FPS from 30 FPS)
- Reduces processing time by 66%
- Maintains gesture recognition quality

### Memory Optimization
- float32 instead of float64 (50% memory reduction)
- Batch processing for large datasets
- Efficient buffer management

### GPU Acceleration
- TensorFlow GPU support via CUDA
- 10-50x faster training
- Automatic GPU detection and usage

## Database

### SQLite (Web Application)
```python
# User table
- id: INTEGER PRIMARY KEY
- email: TEXT UNIQUE
- password_hash: TEXT
- role: TEXT (doctor/patient)
- created_at: TIMESTAMP
```

### File-Based Storage
- Models: .keras format (TensorFlow SavedModel)
- Encoders: .pkl format (pickle)
- Sequences: .npy format (NumPy arrays)
- Metadata: .txt format (plain text)

## Version Control

### Git
- `.gitignore` excludes:
  - `__pycache__/`
  - `*.pyc`
  - `venv/`
  - `models/*.keras`
  - `MP_DATA/`
  - `videos/`
  - `database/*.db`

## Testing

### Manual Testing
- `test_auto_labeler.py`: Test video splitting
- `test_idle_detection.py`: Test idle state detection
- `verify_setup.py`: Verify web app configuration

### Validation
- Train/validation split: 80/20
- Test accuracy reporting
- Top-3 accuracy metrics
- Confusion matrix analysis

## Deployment Considerations

### Production Server
```bash
# Use Gunicorn instead of Flask dev server
gunicorn -k eventlet -w 1 app:app
```

### Environment Variables
- SECRET_KEY: Flask session secret
- DATABASE_PATH: Database location
- MODEL_PATH: Model file location
- DOCTOR_EMAILS: Authorized doctor emails

### HTTPS Requirements
- Required for webcam access in browsers
- SSL certificate needed for production
- SESSION_COOKIE_SECURE = True

## Future Technology Plans

- **TensorFlow Lite**: Mobile deployment
- **WebRTC**: Video call integration
- **PostgreSQL**: Production database
- **Redis**: Session management and caching
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **AWS/GCP**: Cloud deployment
- **React Native**: Mobile app
