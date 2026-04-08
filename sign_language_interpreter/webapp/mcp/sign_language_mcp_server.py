"""
MCP Server Wrapper for ISL (Indian Sign Language) Gesture Recognition.

Exposes the existing ISLDetector pipeline as MCP-compatible Flask endpoints.
Three tools:
  - start_session()        → creates detector, returns session_id
  - process_frame()        → runs inference on a single base64 frame
  - stop_session()         → tears down detector, frees memory
"""

import uuid
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

# ---------------------------------------------------------------------------
# Path setup – allow imports from the parent webapp package
# ---------------------------------------------------------------------------
WEBAPP_DIR = Path(__file__).resolve().parent.parent        # webapp/
SLI_DIR = WEBAPP_DIR.parent                                 # sign_language_interpreter/
sys.path.insert(0, str(WEBAPP_DIR))

from services.isl_detector import ISLDetector

# ---------------------------------------------------------------------------
# Model artefact paths (mirrors webapp/config.py)
# ---------------------------------------------------------------------------
MODEL_DIR = SLI_DIR / "models"
MODEL_PATH = MODEL_DIR / "sign_model.keras"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
MAX_LENGTH_PATH = MODEL_DIR / "max_length.txt"

# ---------------------------------------------------------------------------
# Blueprint (registered in the main Flask app)
# ---------------------------------------------------------------------------
mcp_blueprint = Blueprint("mcp", __name__, url_prefix="/mcp")

# In-memory session store:  { session_id: ISLDetector }
sessions: dict[str, ISLDetector] = {}


# ===========================================================================
# 1. start_session
# ===========================================================================
@mcp_blueprint.route("/start_session", methods=["POST"])
def start_session():
    """Create a new detection session and return its ID."""
    session_id = str(uuid.uuid4())

    try:
        detector = ISLDetector(
            model_path=str(MODEL_PATH),
            encoder_path=str(ENCODER_PATH),
            max_length_path=str(MAX_LENGTH_PATH),
        )
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    sessions[session_id] = detector

    return jsonify({
        "session_id": session_id,
        "status": "created",
    }), 201


# ===========================================================================
# 2. process_frame
# ===========================================================================
@mcp_blueprint.route("/process_frame", methods=["POST"])
def process_frame():
    """Accept a base64-encoded video frame and return the prediction."""
    data = request.get_json(silent=True) or {}

    session_id = data.get("session_id")
    base64_frame = data.get("frame")

    # --- Validate inputs ---------------------------------------------------
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    if not base64_frame:
        return jsonify({"error": "Missing 'frame' field"}), 400

    # --- Decode base64 to OpenCV frame -------------------------------------
    try:
        # Strip optional data-URI prefix (e.g. "data:image/jpeg;base64,")
        if "," in base64_frame:
            base64_frame = base64_frame.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_frame)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        return jsonify({"error": f"Frame decode failed: {e}"}), 400

    # --- Run prediction (reuses existing ML logic) -------------------------
    detector = sessions[session_id]
    result = detector.process_frame(frame)

    return jsonify({
        "text": result["text"],
        "confidence": result["confidence"],
        "detected": result["detected"],
    })


# ===========================================================================
# 3. stop_session
# ===========================================================================
@mcp_blueprint.route("/stop_session", methods=["POST"])
def stop_session():
    """Destroy a detection session and free resources."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    detector = sessions.pop(session_id)
    detector.close()

    return jsonify({
        "session_id": session_id,
        "status": "stopped",
    })


# ===========================================================================
# Health-check
# ===========================================================================
@mcp_blueprint.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "active_sessions": len(sessions),
    })
