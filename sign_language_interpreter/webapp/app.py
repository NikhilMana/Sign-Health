"""
SignHealth — Telehealth platform with real-time ISL interpretation.

Application entry point using the factory pattern.
"""

# eventlet monkey-patch MUST happen before any other imports
import eventlet
eventlet.monkey_patch()

import os
import sys
import logging
from pathlib import Path

from flask import Flask
from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_cors import CORS

# Ensure project root is on the path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config_map
from models.user import Database, User
from services.tts_service import TTSService
from routes.auth import auth_bp, init_auth
from routes.dashboard import dashboard_bp
from events import register_events

# ── Globals ──────────────────────────────────────

socketio = SocketIO()
login_manager = LoginManager()
db = None
isl_detector = None

# ── Logging ──────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Factory ──────────────────────────────────────


def get_isl_detector():
    """Lazy-load the ISL detector (heavy model stays in memory once loaded)."""
    global isl_detector
    if isl_detector is None:
        from flask import current_app
        from services.isl_detector import ISLDetector  # lazy import — heavy ML deps
        isl_detector = ISLDetector(
            current_app.config["MODEL_PATH"],
            current_app.config["ENCODER_PATH"],
            current_app.config["MAX_LENGTH_PATH"],
        )
    return isl_detector


def create_app(env=None):
    """Application factory — returns a configured Flask + SocketIO app."""
    global db

    if env is None:
        env = os.environ.get("FLASK_ENV", "development")

    config_cls = config_map[env]
    app = Flask(__name__)
    app.config.from_object(config_cls)

    # Extensions
    cors_origins = getattr(config_cls, "CORS_ORIGINS", ["*"])
    CORS(app, origins=cors_origins)
    socketio.init_app(app, cors_allowed_origins=cors_origins, async_mode="eventlet")

    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Database
    db = Database(app.config["DATABASE_PATH"])

    # Inject dependencies into blueprints
    init_auth(db, app.config.get("DOCTOR_EMAILS", []))

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)

    # Register SocketIO events
    register_events(socketio, get_isl_detector, TTSService)

    @login_manager.user_loader
    def load_user(user_id):
        return User.get_by_id(db, int(user_id))

    logger.info("SignHealth app created [env=%s]", env)
    return app


# ── Direct execution ─────────────────────────────

if __name__ == "__main__":
    app = create_app()
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
