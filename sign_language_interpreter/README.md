<div align="center">

# 🤟 SignHealth

**Real-Time Indian Sign Language Interpreter for Telehealth**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)

*Bridging the communication gap between sign-language-speaking patients and healthcare providers — one gesture at a time.*

</div>

---

## ✨ What is SignHealth?

SignHealth is an end-to-end system that **recognizes Indian Sign Language (ISL)** in real time and translates it into spoken English during telehealth consultations. A patient signs into their webcam; the doctor sees a live translation with text and audio — no interpreter needed.

```
Patient signs "namaste" → Camera captures → MediaPipe extracts landmarks
    → LSTM model classifies → Doctor sees text + hears audio → Responds via text
```

### Why it matters

Over **18 million** people in India use sign language. Most cannot communicate directly with their doctors. SignHealth changes that by embedding ML-powered interpretation directly into the consultation workflow.

---

## 🏗️ Architecture

```
sign_language_interpreter/
│
├── shared/                     ← Shared utilities (constants, keypoints, registry)
│   ├── constants.py            # All tunable parameters in one place
│   ├── keypoints.py            # Single source of truth for landmark extraction
│   └── model_registry.py       # Model versioning and snapshotting
│
├── data_collection/
│   └── process_videos.py       # Video → keypoint sequences (parallelized)
│
├── model_training/
│   └── train_model.py          # LSTM training with augmentation & callbacks
│
├── real_time_inference/
│   └── detect_live_improved.py # Standalone webcam detection (no server needed)
│
├── webapp/                     ← Flask + SocketIO telehealth app
│   ├── app.py                  # Application factory
│   ├── config.py               # Dev / Production config hierarchy
│   ├── events/                 # SocketIO event handlers
│   ├── routes/                 # Flask Blueprints (auth, dashboard)
│   ├── services/               # ISL detector, TTS, phrase builder, session recorder
│   ├── models/                 # Database & User models (SQLite + bcrypt)
│   ├── templates/              # Jinja2 templates with dark mode support
│   └── static/                 # CSS, JS, images
│
├── models/                     ← Trained model artifacts
│   ├── sign_model.keras
│   ├── label_encoder.pkl
│   └── max_length.txt
│
├── auto_labeler.py             # SRT → labeled video clips
├── retrain_balanced.py         # Class-weighted retraining + confusion matrix
├── run_full_pipeline.py        # One-click: extract → train
└── tests/                      # Automated test suite
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/NikhilMana/Sign-Health.git
cd Sign-Health/sign_language_interpreter

# Core ML dependencies
pip install -r requirements.txt

# Webapp dependencies
pip install -r webapp/requirements_webapp.txt
```

### 2. Train Your Model *(skip if you already have `models/sign_model.keras`)*

```bash
# Option A: Full pipeline (extract features + train)
python run_full_pipeline.py

# Option B: Retrain with class balancing
python retrain_balanced.py
```

### 3. Test Locally (no server)

```bash
python real_time_inference/detect_live_improved.py
# Press 'q' to quit. Signs are shown on-screen with confidence scores.
```

### 4. Launch the Webapp

```bash
cd webapp
python app.py
# Open http://localhost:5000 in your browser
```

---

## 🧠 How the ML Pipeline Works

### Data Flow

```
Raw videos (ISL signs)
    │
    ▼
auto_labeler.py ── SRT subtitles → labeled video clips in videos/
    │
    ▼
process_videos.py ── MediaPipe Holistic → keypoint .npy files in MP_DATA/
    │
    ▼
train_model.py ── Bidirectional LSTM → sign_model.keras + label_encoder.pkl
    │
    ▼
isl_detector.py ── Real-time inference via webcam or SocketIO
```

### Keypoint Extraction

Each video frame produces a **1,662-dimensional** feature vector:

| Body Part | Landmarks | Values per Landmark | Total |
|-----------|-----------|---------------------|-------|
| Pose      | 33        | 4 (x, y, z, visibility) | 132   |
| Face      | 468       | 3 (x, y, z)         | 1,404 |
| Left Hand | 21        | 3 (x, y, z)         | 63    |
| Right Hand| 21        | 3 (x, y, z)         | 63    |
| **Total** |           |                      | **1,662** |

### Model Architecture

```
Input (sequence_length, 1662)
    → Bidirectional LSTM (64 units, return_sequences=True)
    → Dropout (0.3)
    → Bidirectional LSTM (128 units)
    → Dropout (0.4)
    → Dense (256, ReLU, L2) → BatchNorm → Dropout (0.5)
    → Dense (128, ReLU, L2) → Dropout (0.4)
    → Dense (num_classes, Softmax)
```

---

## 🌐 Webapp Architecture

The telehealth web application uses the **Flask application factory pattern** with:

- **Blueprints** for modular routing (`auth`, `dashboard`)
- **SocketIO** for real-time video frame streaming and translation
- **Lazy-loaded ML model** (loaded once, shared across sessions)
- **Cached TTS** (Google TTS results cached in-memory for instant response)
- **Session recording** for medical compliance and transcript export
- **Environment-aware config** (development/production with proper security settings)

### Key Services

| Service | Purpose |
|---------|---------|
| `ISLDetector` | Per-session sign language detection with sliding window and consensus voting |
| `TTSService` | Text-to-speech with in-memory caching for repeated signs |
| `PhraseBuilder` | Accumulates individual signs into coherent sentences |
| `SessionRecorder` | Timestamps and persists consultation transcripts |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_user.py -v
python -m pytest tests/test_phrase_builder.py -v
python -m pytest tests/test_shared.py -v
```

---

## ⚙️ Configuration

SignHealth uses an environment-aware configuration system:

```python
# Set environment (defaults to 'development')
export FLASK_ENV=production
export SECRET_KEY=your-production-secret
export ALLOWED_ORIGINS=https://signhealth.example.com
export DOCTOR_EMAILS=doctor1@hospital.com,doctor2@hospital.com
```

| Setting | Development | Production |
|---------|-------------|------------|
| Debug Mode | ✅ On | ❌ Off |
| HTTPS Cookies | ❌ Off | ✅ On |
| CORS Origins | localhost only | Explicit whitelist |
| Secret Key | Hardcoded (dev) | Environment variable |

---

## 📁 Model Versioning

Every training run is automatically snapshotted by the **Model Registry**:

```python
from shared.model_registry import ModelRegistry

registry = ModelRegistry("models")
registry.register_model(
    metrics={"accuracy": 0.95, "loss": 0.12},
    description="Balanced retrain with augmentation"
)
```

Versioned models are stored in `models/v_YYYYMMDD_HHMMSS/` and can be promoted to production via `registry.promote(version)`.

---

## 🗂️ Project Constants

All tunable parameters are centralized in `shared/constants.py`:

```python
SEQUENCE_WINDOW = 20          # Frames accumulated before prediction
CONFIDENCE_THRESHOLD = 0.7    # Minimum confidence for detection
MOVEMENT_THRESHOLD = 0.001    # Minimum movement to trigger prediction
CONSENSUS_REQUIRED_LOOSE = 2  # Votes needed for webapp consensus (2/3)
CONSENSUS_REQUIRED_STRICT = 4 # Votes needed for local detection (4/5)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `python -m pytest tests/ -v`
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

---

## 📄 License

This project is for educational and research purposes.

---

<div align="center">

*Built with ❤️ for accessibility in healthcare*

**[Report Bug](https://github.com/NikhilMana/Sign-Health/issues) · [Request Feature](https://github.com/NikhilMana/Sign-Health/issues)**

</div>
