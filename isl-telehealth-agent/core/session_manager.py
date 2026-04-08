import uuid
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("session_manager")

class SessionManager:
    """
    Manages active user sessions and their data persistence during telehealth consultations.
    """
    
    # Class-level in-memory storage
    _sessions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def create_session(cls, patient_id: str) -> str:
        """
        Creates a new telehealth session and returns the session_id.
        """
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "patient_id": patient_id,
            "start_time": time.time(),
            "status": "active",
            "turn_count": 0,
            "signs_detected": [],
            "intents_classified": [],
            "last_updated": time.time()
        }
        cls._sessions[session_id] = session_data
        logger.info(f"Created session {session_id} for patient {patient_id}")
        return session_id

    @classmethod
    def get_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an existing session by ID."""
        return cls._sessions.get(session_id)

    @classmethod
    def update_session(cls, session_id: str, update_dict: dict):
        """Updates session data with provided dictionary values."""
        if session_id in cls._sessions:
            cls._sessions[session_id].update(update_dict)
            cls._sessions[session_id]["last_updated"] = time.time()
            logger.debug(f"Updated session {session_id}")

    @classmethod
    def end_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Marks a session as ended, calculates duration, and returns the summary.
        """
        if session_id in cls._sessions:
            session = cls._sessions[session_id]
            session["status"] = "ended"
            session["end_time"] = time.time()
            session["duration_seconds"] = session["end_time"] - session["start_time"]
            logger.info(f"Ended session {session_id}. Duration: {session['duration_seconds']:.2f}s")
            return session
        return None

    @classmethod
    def get_active_sessions(cls) -> List[Dict[str, Any]]:
        """Returns a list of all sessions currently in 'active' status."""
        return [s for s in cls._sessions.values() if s["status"] == "active"]
