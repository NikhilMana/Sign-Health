"""
Session recording for telehealth consultations.

Records timestamped sign detections and doctor messages,
then saves them as a structured transcript for medical records.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records a telehealth session transcript for medical compliance."""

    def __init__(self, patient_id: int, doctor_id: int):
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.start_time = datetime.now()
        self.entries: list[dict] = []

    def add_sign(self, text: str, confidence: float):
        """Log a sign detected from the patient."""
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "type": "sign",
            "speaker": "patient",
            "text": text,
            "confidence": confidence,
        })

    def add_doctor_message(self, text: str):
        """Log a message sent by the doctor."""
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "type": "text",
            "speaker": "doctor",
            "text": text,
        })

    def save(self, db):
        """Persist the transcript to the ``consultations`` table."""
        transcript = json.dumps(self.entries, indent=2)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO consultations (patient_id, doctor_id, transcript, created_at)
                   VALUES (?, ?, ?, ?)""",
                (self.patient_id, self.doctor_id, transcript, self.start_time.isoformat()),
            )
            conn.commit()
            consultation_id = cursor.lastrowid
        logger.info(
            "Saved consultation %d (%d entries)", consultation_id, len(self.entries)
        )
        return consultation_id

    def export_text(self) -> str:
        """Return a human-readable plaintext transcript."""
        lines = [f"Session: {self.start_time.strftime('%Y-%m-%d %H:%M')}"]
        lines.append("=" * 50)
        for entry in self.entries:
            ts = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
            speaker = "🤟 Patient" if entry["speaker"] == "patient" else "👨‍⚕️ Doctor"
            lines.append(f"[{ts}] {speaker}: {entry['text']}")
        return "\n".join(lines)
