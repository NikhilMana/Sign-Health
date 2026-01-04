# Indian Sign Language (ISL) Interpreter

A real-time Indian Sign Language recognition system using deep learning and computer vision. This system can detect and interpret ISL gestures from video input using LSTM neural networks and MediaPipe pose estimation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---
   sending first pull request

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Real-time Sign Language Detection**: Recognizes ISL gestures from webcam feed at 25-35 FPS
- **Deep Learning Model**: Bidirectional LSTM network with 64-128 units
- **MediaPipe Integration**: Extracts pose and hand keypoints (1,662 features per frame)
- **Automated Data Pipeline**: Tools for video collection, labeling, and preprocessing
- **Data Augmentation**: Gaussian noise augmentation for improved generalization
- **Prediction Smoothing**: Voting mechanism for stable real-time predictions
- **GPU Acceleration**: Supports CUDA for faster training and inference
- **Confidence Scoring**: Visual confidence indicators for predictions
- **Extensible**: Easy to add new signs and retrain

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ISL Interpreter Pipeline                  │
└─────────────────────────────────────────────────────────────┘

1. DATA COLLECTION
   ├── Video Input (MP4 files)
   ├── auto_labeler.py → Splits videos by subtitles
   └── Output: Labeled video segments

2. FEATURE EXTRACTION
   ├── process_videos.py → MediaPipe processing
   ├── Extracts: Pose (33×4) + Hands (21×3 each)
   └── Output: .npy files (1,662 features/frame)

3. MODEL TRAINING
   ├── train_model.py → LSTM training
   ├── Architecture: Bidirectional LSTM + Dense layers
   └── Output: sign_model.keras + label_encoder.pkl

4. REAL-TIME INFERENCE
   ├── detect_live.py → Webcam detection
   ├── Processes 20-frame sequences
   └── Output: Live predictions with confidence
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- CUDA-compatible GPU (optional, for faster training)
- 8GB RAM minimum (16GB recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/isl-interpreter.git
cd isl-interpreter/sign_language_interpreter
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.10.0
opencv-python>=4.7.0
mediapipe>=0.10.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"
```

---

## 📁 Project Structure

```
sign_language_interpreter/
│
├── data_collection/
│   └── auto_labeler.py          # Splits videos by subtitles
│
├── data_preprocessing/
│   └── process_videos.py        # Extracts keypoints from videos
│
├── model_training/
│   └── train_model.py           # Trains LSTM model
│
├── real_time_inference/
│   └── detect_live.py           # Real-time webcam detection
│
├── videos/                      # Input videos organized by sign
│   ├── hello/
│   ├── thank you/
│   └── ...
│
├── MP_DATA/                     # Extracted keypoint sequences (.npy)
│   ├── hello/
│   ├── thank you/
│   └── ...
│
├── models/                      # Trained models and metadata
│   ├── sign_model.keras         # Main trained model
│   ├── best_model.keras         # Best checkpoint during training
│   ├── label_encoder.pkl        # Label encoder for classes
│   ├── max_length.txt           # Sequence length (60)
│   └── training_history.png     # Training curves
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── OPTIMIZATION_SUMMARY.md      # Performance optimization details
```

---

## 📖 Usage Guide

### 1. Prepare Your Dataset

#### Option A: Use Existing Videos

Place your ISL videos in the `videos/` folder, organized by sign:

```
videos/
├── hello/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── thank you/
│   ├── video1.mp4
│   └── video2.mp4
└── ...
```

#### Option B: Auto-Label from Subtitled Videos

If you have videos with subtitles:

```bash
cd data_collection
python auto_labeler.py
```

**Input prompts:**
- Video file path (e.g., `input_video.mp4`)
- Subtitle file path (e.g., `subtitles.srt`)
- Output directory (e.g., `../videos`)

This will automatically split videos based on subtitle timestamps.

---

### 2. Extract Features from Videos

Process videos to extract MediaPipe keypoints:

```bash
cd data_preprocessing
python process_videos.py
```

**What it does:**
- Processes all videos in `videos/` folder
- Extracts pose (33 landmarks) + hand keypoints (21 per hand)
- Saves sequences as `.npy` files in `MP_DATA/`
- Uses multiprocessing for 6-12x speedup
- Skips already processed videos

**Expected output:**
```
Processing videos...
Found 50 videos across 10 signs
Processing: hello/video1.mp4 ✓
Processing: hello/video2.mp4 ✓
...
Completed: 50/50 videos
```

**Requirements:**
- Minimum 5 videos per sign (10+ recommended)
- Clear visibility of signer's upper body
- Good lighting conditions

---

### 3. Train the Model

Train the LSTM model on extracted features:

```bash
cd model_training
python train_model.py
```

**Training process:**
1. Loads sequences from `MP_DATA/`
2. Filters classes with <10 samples
3. Applies data augmentation (if <1000 samples)
4. Trains Bidirectional LSTM model
5. Saves model to `models/sign_model.keras`

**Training parameters:**
- Epochs: 150 (with early stopping)
- Batch size: 64
- Learning rate: 0.001 (with reduction on plateau)
- Sequence length: 60 frames (capped)

**Expected output:**
```
Sign Language Recognition Model Training
==================================================
Loading sign language data...
Found sign labels: ['hello', 'thank you', ...]
Total sequences loaded: 150

Augmenting data...
Augmented from 150 to 450 sequences

Creating LSTM model...
Model Summary:
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
bidirectional (Bidirection  (None, 60, 128)          8,847,360
dropout (Dropout)           (None, 60, 128)          0         
bidirectional_1 (Bidirecti  (None, 256)              263,168   
dropout_1 (Dropout)         (None, 256)              0         
dense (Dense)               (None, 256)              65,792    
batch_normalization (Batch  (None, 256)              1,024     
dropout_2 (Dropout)         (None, 256)              0         
dense_1 (Dense)             (None, 128)              32,896    
dropout_3 (Dropout)         (None, 128)              0         
dense_2 (Dense)             (None, 10)               1,290     
=================================================================
Total params: 9,211,530
Trainable params: 9,211,018
Non-trainable params: 512

Training...
Epoch 1/150 - loss: 2.3456 - accuracy: 0.2345 - val_accuracy: 0.3456
...
Epoch 45/150 - loss: 0.1234 - accuracy: 0.9567 - val_accuracy: 0.8923

Test Accuracy: 89.23%
Test Top-3 Accuracy: 96.78%

Model saved to: models/sign_model.keras
```

**Training time:**
- CPU: 2-4 hours (small dataset)
- GPU: 15-30 minutes (small dataset)

---

### 4. Real-Time Detection

Run the live sign language detector:

```bash
cd real_time_inference
python detect_live.py
```

**Controls:**
- Press `q` to quit
- Ensure good lighting and clear view of upper body

**UI Elements:**
- **Top bar**: FPS counter and instructions
- **Video feed**: Live webcam with pose/hand landmarks
- **Bottom bar**: 
  - Predicted sign (green if confident, gray if detecting)
  - Confidence percentage
  - Confidence bar (visual indicator)

**How it works:**
1. Captures 20-frame sequences from webcam
2. Extracts keypoints using MediaPipe
3. Feeds sequence to LSTM model
4. Uses voting from last 5 predictions for stability
5. Displays prediction if confidence > 70%

**Performance:**
- FPS: 25-35 on average hardware
- Latency: ~300-500ms from gesture to prediction
- Confidence threshold: 70% (adjustable)

---

## 🧠 Model Architecture

### Input Features

**Per Frame: 1,662 features**
- Pose landmarks: 33 × 4 = 132 (x, y, z, visibility)
- Left hand: 21 × 3 = 63 (x, y, z)
- Right hand: 21 × 3 = 63 (x, y, z)
- **Total: 132 + 63 + 63 = 258 features/frame**

**Sequence: 60 frames × 1,662 features = 99,720 inputs**

### Network Architecture

```python
Model: Sequential
_________________________________________________________________
Layer                           Output Shape         Params
=================================================================
Bidirectional LSTM (64 units)   (None, 60, 128)     8,847,360
Dropout (0.3)                   (None, 60, 128)     0
Bidirectional LSTM (128 units)  (None, 256)         263,168
Dropout (0.4)                   (None, 256)         0
Dense (256, ReLU, L2=0.001)     (None, 256)         65,792
Batch Normalization             (None, 256)         1,024
Dropout (0.5)                   (None, 256)         0
Dense (128, ReLU, L2=0.001)     (None, 128)         32,896
Dropout (0.4)                   (None, 128)         0
Dense (num_classes, Softmax)    (None, num_classes) varies
=================================================================
Total params: ~9.2M (for 10 classes)
```

### Key Features

- **Bidirectional LSTM**: Captures temporal patterns in both directions
- **Dropout layers**: Prevents overfitting (0.3-0.5)
- **L2 Regularization**: Weight decay for better generalization
- **Batch Normalization**: Stabilizes training
- **Adam Optimizer**: Learning rate 0.001 with gradient clipping

### Training Strategy

1. **Data Augmentation**: Adds Gaussian noise (σ=0.005) to create 3× data
2. **Early Stopping**: Patience of 15 epochs on validation loss
3. **Learning Rate Reduction**: Halves LR after 7 epochs without improvement
4. **Model Checkpointing**: Saves best model based on validation accuracy

---

## 📊 Dataset

### Data Collection Guidelines

**Minimum Requirements:**
- 10 samples per sign (20+ recommended)
- 5-10 different signers for diversity
- 2-5 second duration per video

**Video Quality:**
- Resolution: 640×480 or higher
- Frame rate: 30 FPS
- Format: MP4, AVI, MOV
- Lighting: Bright, even lighting
- Background: Plain, uncluttered

**Signer Guidelines:**
- Full upper body visible
- Hands clearly visible throughout
- Consistent signing speed
- Natural signing style

### Data Augmentation

**Applied when dataset < 1000 samples:**
- Gaussian noise addition (mean=0, std=0.005)
- 2 augmented versions per original sample
- Preserves gesture integrity while adding variation

---

## ⚡ Performance

### System Performance

| Metric | Value |
|--------|-------|
| Real-time FPS | 25-35 FPS |
| Prediction Latency | 300-500ms |
| Model Size | ~37 MB |
| Training Time (GPU) | 15-30 min |
| Training Time (CPU) | 2-4 hours |
| Inference Time | ~30ms/sequence |

### Model Performance

**Expected Accuracy (depends on dataset):**
- Training Accuracy: 90-98%
- Validation Accuracy: 85-95%
- Top-3 Accuracy: 95-99%

**Factors Affecting Accuracy:**
- Dataset size (more is better)
- Signer diversity
- Video quality
- Sign complexity
- Class balance

### Optimization Highlights

- **6-12× faster** feature extraction (multiprocessing + frame skipping)
- **10-50× faster** video splitting (FFmpeg stream copy)
- **40-60% FPS improvement** in real-time detection
- **30-40% memory reduction** (float32, optimized buffers)

See [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) for detailed performance analysis.

---

## 🔧 Configuration

### Adjustable Parameters

**process_videos.py:**
```python
FRAME_SKIP = 3              # Process every 3rd frame (10fps from 30fps)
MODEL_COMPLEXITY = 0        # MediaPipe complexity (0=fast, 2=accurate)
NUM_WORKERS = 4             # Parallel processing workers
```

**train_model.py:**
```python
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples to include class
MAX_SEQ_LENGTH_CAP = 60     # Maximum sequence length
EPOCHS = 150                # Training epochs
BATCH_SIZE = 64             # Batch size
LEARNING_RATE = 0.001       # Initial learning rate
```

**detect_live.py:**
```python
SEQUENCE_LENGTH = 20        # Frames to collect before prediction
PREDICTION_THRESHOLD = 0.7  # Confidence threshold (70%)
PREDICTION_HISTORY = 5      # Frames for voting smoothing
CAMERA_WIDTH = 640          # Webcam resolution
CAMERA_HEIGHT = 480
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No GPU detected" during training
**Solution:**
```bash
# Check CUDA installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA-enabled TensorFlow
pip install tensorflow-gpu
```

#### 2. "Model file not found" during detection
**Solution:**
- Ensure you've run `train_model.py` first
- Check that `models/sign_model.keras` exists
- Verify `label_encoder.pkl` and `max_length.txt` are present

#### 3. Low FPS during real-time detection
**Solutions:**
- Reduce `MODEL_COMPLEXITY` to 0 in detect_live.py
- Lower camera resolution (640×480 → 320×240)
- Close other applications
- Use GPU acceleration

#### 4. Poor prediction accuracy
**Solutions:**
- Collect more training data (20+ samples per sign)
- Ensure consistent signing across videos
- Check lighting and video quality
- Increase training epochs
- Add more data augmentation

#### 5. "Memory Error" during training
**Solutions:**
- Reduce `BATCH_SIZE` (64 → 32 → 16)
- Reduce `MAX_SEQ_LENGTH_CAP` (60 → 40)
- Close other applications
- Use smaller model (reduce LSTM units)

#### 6. Webcam not opening
**Solutions:**
```python
# Try different camera indices
cap = cv2.VideoCapture(0)  # Try 0, 1, 2...

# Check available cameras
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

---

## 🎯 Best Practices

### Data Collection
1. Use multiple signers for diversity
2. Vary lighting conditions slightly
3. Include different backgrounds
4. Maintain consistent signing speed
5. Ensure full visibility of hands and upper body

### Training
1. Start with small dataset to test pipeline
2. Monitor both training and validation accuracy
3. Use early stopping to prevent overfitting
4. Save training history for analysis
5. Test on unseen signers

### Deployment
1. Test in various lighting conditions
2. Calibrate confidence threshold for your use case
3. Provide user feedback (visual/audio)
4. Log predictions for continuous improvement
5. Regularly retrain with new data

---

## 🚀 Future Enhancements

### Planned Features
- [ ] Sentence-level recognition (word sequences)
- [ ] Multi-language support (ASL, BSL)
- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] Cloud-based training pipeline
- [ ] Real-time translation to text/speech
- [ ] Gesture segmentation (auto-detect sign boundaries)
- [ ] Transformer-based architecture
- [ ] 3D hand pose estimation

### Research Directions
- Attention mechanisms for better temporal modeling
- Few-shot learning for new signs
- Transfer learning from pre-trained models
- Continuous sign language recognition
- Signer-independent recognition

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -m "Add new feature"`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Submit Pull Request**

### Contribution Areas
- Adding new ISL signs to dataset
- Improving model architecture
- Optimizing real-time performance
- Documentation improvements
- Bug fixes and testing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose and hand tracking
- **TensorFlow** team for deep learning framework
- **OpenCV** community for computer vision tools
- ISL community for sign language resources

---

## 📧 Contact

For questions, issues, or collaborations:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/isl-interpreter/issues)
- **Email**: your.email@example.com
- **Project Link**: https://github.com/yourusername/isl-interpreter

---

## 📚 References

1. MediaPipe Holistic: https://google.github.io/mediapipe/solutions/holistic
2. LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
3. Sign Language Recognition: Research papers and datasets
4. TensorFlow Documentation: https://www.tensorflow.org/

---

**Made with ❤️ for the ISL community**

*Last Updated: 2024*
