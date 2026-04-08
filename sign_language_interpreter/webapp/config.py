"""
Environment-aware configuration hierarchy.

Usage:
    from config import config_map
    app.config.from_object(config_map[os.environ.get('FLASK_ENV', 'development')])
"""

import os
from pathlib import Path


class _BaseConfig:
    """Shared settings across all environments."""

    BASE_DIR = Path(__file__).parent
    DATABASE_PATH = BASE_DIR / "database" / "telehealth.db"

    # Model paths
    MODEL_DIR = BASE_DIR.parent / "models"
    MODEL_PATH = MODEL_DIR / "sign_model.keras"
    ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
    MAX_LENGTH_PATH = MODEL_DIR / "max_length.txt"

    # Session security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # SocketIO
    SOCKETIO_ASYNC_MODE = "eventlet"


class DevelopmentConfig(_BaseConfig):
    DEBUG = True
    SECRET_KEY = "dev-only-not-for-production"
    SESSION_COOKIE_SECURE = False
    CORS_ORIGINS = ["*"]

    DOCTOR_EMAILS = [
        "doctor@example.com",
        "dr.smith@hospital.com",
    ]


class ProductionConfig(_BaseConfig):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
    SESSION_COOKIE_SECURE = True
    CORS_ORIGINS = os.environ.get(
        "ALLOWED_ORIGINS", "https://signhealth.example.com"
    ).split(",")

    DOCTOR_EMAILS = os.environ.get("DOCTOR_EMAILS", "").split(",")


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
