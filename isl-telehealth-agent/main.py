import sys
import time
import logging
import asyncio
import threading
import numpy as np

from core.agent_registry import AgentRegistry
from core.session_manager import SessionManager
from core.message_bus import MessageBus
from core.websocket_bridge import WebSocketBridge

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("orchestrator")

class TelehealthOrchestrator:
    def __init__(self):
        self.registry = AgentRegistry()
        self.session_manager = SessionManager()
        self.message_bus = MessageBus()
        self.ws_bridge = WebSocketBridge()
        self.active_session_id = None

        print("=" * 48)
        print("  ISL Telehealth Multi-Agent System")
        print("  Powered by MCP + Multi-Agent Architecture")
        print("  Model: sign_language_interpreter/models/")
        print("=" * 48)

        # Wire up the message bus
        self.ws_bridge.subscribe_to_bus(self.message_bus)

        # Give the bridge a reference back to us so it can call process_frame
        self.ws_bridge.set_orchestrator(self)

    def start_session(self, patient_id: str) -> str:
        session_id = SessionManager.create_session(patient_id)
        self.active_session_id = session_id
        logger.info(f"Session started. ID: {session_id} | Patient: {patient_id}")
        return session_id

    async def process_webcam_frame(self, frame: np.ndarray, session_id: str) -> dict:
        result = await self.registry.run_pipeline(frame, session_id, message_bus=self.message_bus)
        if result:
            status = result.get("pipeline_status")
            if status == "complete":
                self.message_bus.publish("response_ready", result)
        return result or {}

    def end_session(self, session_id: str) -> dict:
        summary = SessionManager.end_session(session_id)
        if summary:
            total_signs = len(summary.get("signs_detected", []))
            total_intents = len(summary.get("intents_classified", []))
            duration = summary.get("duration_seconds", 0.0)

            print("\n" + "=" * 50)
            print("  SESSION SUMMARY")
            print("=" * 50)
            print(f"  Session ID    : {summary.get('session_id')}")
            print(f"  Duration      : {duration:.1f} seconds")
            print(f"  Total Signs   : {total_signs}")
            print(f"  Total Intents : {total_intents}")
            print("=" * 50 + "\n")
        return summary or {}

    async def run_demo(self):
        logger.info("Starting demo mode (simulating 5 signs)...")
        session_id = self.start_session("demo_patient_001")

        demo_sequence = [
            (["chest", "pain"], "SYMPTOM_REPORT"),
            (["three_days"], "context update"),
            (["help", "breathe"], "EMERGENCY_ALERT"),
            (["doctor"], "intent update"),
            (["appointment"], "APPOINTMENT_REQUEST")
        ]

        self.registry.context.clear_window()

        for step_idx, (signs, expected) in enumerate(demo_sequence, 1):
            print(f"\n--- STEP {step_idx}: {signs} ---")
            
            for sign in signs:
                self.registry.context.process_sign(sign, 0.95)
            
            complaint = self.registry.context.extract_complaint()
            intent_result = self.registry.intent.classify_intent(complaint)
            action = self.registry.dialogue.manage_turn(intent_result, session_id)

            if action["action"] == "execute_tool":
                tool_result = await self.registry.tool.execute(action)
                final_response = self.registry.response.format_response(tool_result, intent_result["intent"])
            elif action["action"] == "ask_clarification":
                final_response = {
                    "text_response": action["question"],
                    "patient_card": {
                        "icon": "❓",
                        "headline": "Clarification Needed",
                        "detail": action["question"][:50],
                        "color": "yellow"
                    },
                    "voice_sentence": action["question"],
                    "urgency_color": "yellow"
                }
            elif action["action"] == "emergency_escalation":
                final_response = self.registry.response.format_response(
                    {"urgency_level": "emergency", "immediate_action": action["message"]},
                    "EMERGENCY_ALERT"
                )

            intent = intent_result.get("intent", "UNKNOWN")
            confidence = intent_result.get("confidence", 0.0)

            print(f"Intent: {intent} ({confidence:.2f})")
            print(f"Action: {action['action']}")
            print(f"Response: {final_response.get('text_response')}")

            card = final_response.get("patient_card", {})
            icon = card.get("icon", "?")
            headline = card.get("headline", "")
            detail = card.get("detail", "")

            print(f"Patient Card: {icon} {headline} | {detail}")
            print(f"Voice: {final_response.get('voice_sentence')}")

            await asyncio.sleep(1)

        self.end_session(session_id)

    async def run_live(self):
        """
        Live mode: starts a session and keeps the server alive.
        Frames arrive from the browser via WebSocket and are processed
        by the WebSocketBridge handler. No cv2 windows here.
        """
        patient_id = input("Enter patient ID (or press Enter for default): ").strip()
        if not patient_id:
            patient_id = "live_patient_001"

        self.start_session(patient_id)
        logger.info("Live mode active. Open http://localhost:8000 in your browser.")
        logger.info("The browser captures your webcam and streams frames to this backend.")
        logger.info("Press Ctrl+C to end the session.")

        try:
            # Keep the main thread alive while WS bridge processes frames
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.active_session_id:
                self.end_session(self.active_session_id)


async def main():
    orchestrator = TelehealthOrchestrator()

    # Start WebSocket server in a background thread
    ws_thread = threading.Thread(target=orchestrator.ws_bridge.run_in_thread, daemon=True)
    ws_thread.start()

    # Small delay to let the WS server bind
    await asyncio.sleep(1)

    # Automatically start Live Dashboard Web Server Mode
    await orchestrator.run_live()


if __name__ == "__main__":
    asyncio.run(main())
