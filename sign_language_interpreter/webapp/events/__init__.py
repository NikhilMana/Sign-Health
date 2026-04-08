"""
SocketIO event handlers for video processing and doctor-patient messaging.
"""

import logging
import base64
import cv2
import numpy as np
from flask import request
from flask_socketio import emit, join_room

logger = logging.getLogger(__name__)


def register_events(socketio, get_isl_detector, tts_service_cls):
    """Bind all SocketIO events.  Called once by the application factory."""

    @socketio.on("connect")
    def handle_connect():
        logger.info("Client connected: %s", request.sid)

    @socketio.on("disconnect")
    def handle_disconnect():
        logger.info("Client disconnected: %s", request.sid)

    @socketio.on("join_session")
    def handle_join_session(data):
        room = data.get("room")
        join_room(room)
        emit("joined", {"room": room}, room=request.sid)

    @socketio.on("video_frame")
    def handle_video_frame(data):
        try:
            img_data = base64.b64decode(data["frame"].split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            detector = get_isl_detector()
            result = detector.process_frame(frame)

            if result["text"] and result["confidence"] > 0.7:
                audio_base64 = tts_service_cls.text_to_speech(result["text"])

                emit(
                    "translation",
                    {
                        "text": result["text"],
                        "confidence": result["confidence"],
                        "audio": audio_base64,
                        "patient_id": data.get("patient_id"),
                    },
                    room=data.get("doctor_room"),
                    broadcast=True,
                )
                emit("detection_result", result, room=request.sid)
        except Exception:
            logger.error("Frame processing failed", exc_info=True)
            emit("error", {"message": "Processing error"}, room=request.sid)

    @socketio.on("doctor_message")
    def handle_doctor_message(data):
        """Relay doctor's text message to the patient room."""
        emit(
            "doctor_response",
            {
                "text": data["text"],
                "doctor_name": data.get("doctor_name", "Doctor"),
            },
            room=data.get("patient_room"),
        )

    # ── WebRTC signaling ──────────────────────────────────

    @socketio.on("call_request")
    def handle_call_request(data):
        """Doctor requests to start a video call with the patient."""
        logger.info("Call request from doctor → patient room %s", data.get("patient_room"))
        emit("incoming_call", {"doctor_name": data.get("doctor_name", "Doctor")},
             room=data.get("patient_room"))

    @socketio.on("call_accepted")
    def handle_call_accepted(data):
        """Patient accepts the call → notify doctor to start WebRTC offer."""
        logger.info("Call accepted by patient → doctor room %s", data.get("doctor_room"))
        emit("call_accepted", {}, room=data.get("doctor_room"))

    @socketio.on("webrtc_offer")
    def handle_webrtc_offer(data):
        """Relay SDP offer from doctor to patient."""
        emit("webrtc_offer", {"sdp": data["sdp"]}, room=data.get("patient_room"))

    @socketio.on("webrtc_answer")
    def handle_webrtc_answer(data):
        """Relay SDP answer from patient to doctor."""
        emit("webrtc_answer", {"sdp": data["sdp"]}, room=data.get("doctor_room"))

    @socketio.on("ice_candidate")
    def handle_ice_candidate(data):
        """Relay ICE candidate to the other peer."""
        target_room = data.get("target_room")
        emit("ice_candidate", {"candidate": data["candidate"]}, room=target_room)

    @socketio.on("end_call")
    def handle_end_call(data):
        """Notify both sides that the call has ended."""
        emit("call_ended", {}, room=data.get("target_room"))
