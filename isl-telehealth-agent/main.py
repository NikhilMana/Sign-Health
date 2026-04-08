import sys
import time
import logging
import asyncio
import threading
import numpy as np

try:
    import cv2
except ImportError:
    from unittest.mock import MagicMock
    cv2 = MagicMock()

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

        # Subscribe WebSocketBridge to message_bus topics
        self.ws_bridge.subscribe_to_bus(self.message_bus)

        print("╔══════════════════════════════════════════════╗")
        print("║  ISL Telehealth Multi-Agent System           ║")
        print("║  Powered by MCP + Multi-Agent Architecture   ║")
        print("║  Model: sign_language_interpreter/models/    ║")
        print("╚══════════════════════════════════════════════╝")

    def start_session(self, patient_id: str) -> str:
        session_id = SessionManager.create_session(patient_id)
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
            # Print formatted summary with total signs, intents, duration.
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
            
            # 1. Process each sign through context agent individually
            for sign in signs:
                self.registry.context.process_sign(sign, 0.95)
            
            # 2. Extract complaint and run remainder of pipeline
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

            # 3. Print formatted output
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


async def main():
    orchestrator = TelehealthOrchestrator()
    
    use_demo = input("Run demo mode? (y/n): ").strip().lower()

    if use_demo == 'y':
        # Start WebSocket bridge in background thread for demo too
        ws_thread = threading.Thread(target=lambda: asyncio.run(orchestrator.ws_bridge.run()), daemon=True)
        ws_thread.start()
        await orchestrator.run_demo()
    else:
        # Start WebSocket bridge in background thread
        ws_thread = threading.Thread(target=lambda: asyncio.run(orchestrator.ws_bridge.run()), daemon=True)
        ws_thread.start()

        patient_id = input("Enter patient ID: ").strip()
        if not patient_id:
            patient_id = "live_patient_001"
            
        session_id = orchestrator.start_session(patient_id)

        cap = cv2.VideoCapture(0)
        logger.info("Starting webcam loop. Press 'q' to end session and quit.")

        # Used to store the last mediapipe results for drawing since run_pipeline suppresses it
        last_results = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame from webcam. Exiting loop.")
                break

            # If perception agent processed holistic but didn't return it upstream, we extract
            # results safely by monkey-patching or fetching the latest property if available. 
            # Process via pipeline
            result = await orchestrator.process_webcam_frame(frame, session_id)
            
            # Simple retrieval of last results if perception agent caches them. We can also
            # extract them directly. Wait, perception agent's sequence holds keypoints.
            # I will patch agent_registry to return perception_results.
            results_obj = result.get("perception_results", None) if result else None
            if results_obj:
                last_results = results_obj

            annotated_frame = orchestrator.registry.perception.draw_landmarks(frame, last_results)

            cv2.imshow("ISL Telehealth Agent", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        orchestrator.end_session(session_id)


if __name__ == "__main__":
    asyncio.run(main())
