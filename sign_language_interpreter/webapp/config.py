import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DATABASE_PATH = BASE_DIR / 'database' / 'telehealth.db'
    
    # Model paths
    MODEL_DIR = BASE_DIR.parent / 'models'
    MODEL_PATH = MODEL_DIR / 'sign_model.keras'
    ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
    MAX_LENGTH_PATH = MODEL_DIR / 'max_length.txt'
    
    # Session configuration
    SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # SocketIO configuration
    SOCKETIO_ASYNC_MODE = 'eventlet'
    
    # Doctor emails (for role identification)
    DOCTOR_EMAILS = [
        'doctor@example.com',
        'dr.smith@hospital.com',
        # Add more doctor emails here
    ]
