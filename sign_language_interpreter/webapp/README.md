# ISL Telehealth Web Application

A web-based telehealth platform that enables real-time communication between deaf/mute patients and doctors using Indian Sign Language (ISL) interpretation.

## 🎯 Features

- **Role-Based Authentication**: Separate interfaces for doctors and patients
- **Real-Time ISL Detection**: Live sign language recognition from webcam
- **Text-to-Speech**: Automatic audio generation for detected signs
- **Bidirectional Communication**: Doctors can send text responses to patients
- **Consultation Notes**: Doctors can save session notes
- **WebSocket Communication**: Real-time data streaming

## 📋 Prerequisites

- Python 3.8+
- Trained ISL model (sign_model.keras)
- Webcam
- Modern web browser (Chrome/Firefox recommended)

## 🚀 Installation

### 1. Install Dependencies

```bash
cd webapp
pip install -r requirements_webapp.txt
```

### 2. Configure Doctor Emails

Edit `config.py` and add doctor email addresses:

```python
DOCTOR_EMAILS = [
    'doctor@example.com',
    'dr.smith@hospital.com',
    # Add more doctor emails
]
```

### 3. Verify Model Files

Ensure these files exist in the `models/` directory:
- `sign_model.keras`
- `label_encoder.pkl`
- `max_length.txt`

## 🏃 Running the Application

### Start the Server

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

### Access the Application

1. **Register**: Create an account at `/register`
   - Use a doctor email for doctor access
   - Use any other email for patient access

2. **Login**: Login at `/login`
   - Doctors are redirected to `/doctor/dashboard`
   - Patients are redirected to `/patient/dashboard`

## 👤 Patient Interface

### Features:
- **Video Feed**: Live webcam capture
- **ISL Detection**: Real-time sign language recognition
- **Detected Signs**: Display of recognized signs with confidence
- **Conversation History**: Log of all communications
- **Audio Feedback**: Text-to-speech for doctor responses

### Usage:
1. Click "Start Detection" to begin
2. Perform sign language gestures
3. Detected signs appear in real-time
4. Receive doctor responses as text and audio

## 👨‍⚕️ Doctor Interface

### Features:
- **Translation Feed**: Real-time patient sign translations
- **Audio Playback**: Hear patient's messages
- **Text Response**: Send messages to patient
- **Session Info**: Track consultation duration and message count
- **Consultation Notes**: Save session notes

### Usage:
1. View incoming patient translations
2. Listen to audio playback
3. Type responses in the input field
4. Save consultation notes for records

## 🔧 Configuration

### config.py Settings

```python
# Database
DATABASE_PATH = 'database/telehealth.db'

# Model paths
MODEL_PATH = '../models/sign_model.keras'
ENCODER_PATH = '../models/label_encoder.pkl'
MAX_LENGTH_PATH = '../models/max_length.txt'

# Doctor emails
DOCTOR_EMAILS = ['doctor@example.com']
```

## 🏗️ Architecture

```
Patient Browser                    Server                    Doctor Browser
     │                               │                             │
     ├─── Video Frames ──────────────▶ ISL Detector               │
     │                               │      │                      │
     │                               │      ▼                      │
     │                               │  Translation                │
     │                               │      │                      │
     │                               │      ▼                      │
     │                               ├─── Text + Audio ───────────▶│
     │                               │                             │
     │◀──── Doctor Response ─────────┤◀──── Text Message ─────────┤
```

## 📁 Project Structure

```
webapp/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── requirements_webapp.txt     # Dependencies
│
├── models/
│   └── user.py                 # User model and database
│
├── services/
│   ├── isl_detector.py         # ISL detection service
│   └── tts_service.py          # Text-to-speech service
│
├── templates/
│   ├── base.html               # Base template
│   ├── login.html              # Login page
│   ├── register.html           # Registration page
│   ├── patient_dashboard.html  # Patient interface
│   └── doctor_dashboard.html   # Doctor interface
│
├── static/
│   ├── css/
│   │   └── style.css           # Styles
│   └── js/
│       ├── patient.js          # Patient logic
│       └── doctor.js           # Doctor logic
│
└── database/
    └── telehealth.db           # SQLite database
```

## 🔐 Security

- Passwords are hashed using bcrypt
- Session management with Flask-Login
- HTTPS recommended for production
- Input validation on all forms

## 🐛 Troubleshooting

### Camera Not Working
- Check browser permissions
- Ensure HTTPS (required for camera access in production)
- Try different browsers

### Model Not Loading
- Verify model files exist in `../models/`
- Check file paths in `config.py`
- Ensure TensorFlow is installed

### WebSocket Connection Failed
- Check firewall settings
- Verify port 5000 is available
- Try disabling browser extensions

### Low Detection Accuracy
- Ensure good lighting
- Position camera to show full upper body
- Hold signs for 1-2 seconds
- Check model training quality

## 🚀 Deployment

### Production Checklist

1. **Set SECRET_KEY**:
   ```python
   SECRET_KEY = 'your-secure-random-key'
   ```

2. **Enable HTTPS**:
   ```python
   SESSION_COOKIE_SECURE = True
   ```

3. **Use Production Server**:
   ```bash
   gunicorn -k eventlet -w 1 app:app
   ```

4. **Database**: Migrate to PostgreSQL for production

5. **Environment Variables**: Use `.env` file for sensitive data

### Deployment Options

- **Heroku**: Easy deployment with Procfile
- **AWS EC2**: Full control, scalable
- **DigitalOean**: Simple VPS hosting
- **Google Cloud Run**: Serverless option

## 📊 Performance

- **Latency**: ~300-500ms from sign to translation
- **FPS**: 10 frames/second processing
- **Concurrent Users**: Supports multiple simultaneous consultations
- **Model Inference**: ~30ms per prediction

## 🔄 Future Enhancements

- [ ] Video call integration (WebRTC)
- [ ] Appointment scheduling
- [ ] Medical records integration
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Prescription system
- [ ] Payment integration
- [ ] Analytics dashboard

## 📝 API Endpoints

### HTTP Routes
- `GET /` - Home (redirects based on role)
- `GET/POST /login` - Login page
- `GET/POST /register` - Registration page
- `GET /logout` - Logout
- `GET /patient/dashboard` - Patient interface
- `GET /doctor/dashboard` - Doctor interface

### WebSocket Events

**Client → Server:**
- `join_session` - Join a room
- `video_frame` - Send video frame for processing
- `doctor_message` - Send doctor's message to patient

**Server → Client:**
- `translation` - Send ISL translation to doctor
- `detection_result` - Send detection result to patient
- `doctor_response` - Send doctor's message to patient

## 🤝 Contributing

Contributions welcome! Please:
1. Test thoroughly
2. Follow existing code style
3. Update documentation
4. Submit pull request

## 📄 License

MIT License - See main project LICENSE file

## 📧 Support

For issues or questions:
- GitHub Issues
- Email: support@example.com

---

**Built with ❤️ for accessible healthcare**
