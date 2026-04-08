# Services package
# ISLDetector is imported lazily via app.get_isl_detector() to avoid
# loading heavy ML libraries (tensorflow, mediapipe) at startup.
from services.tts_service import TTSService

__all__ = ["TTSService"]
