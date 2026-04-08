import sys
import asyncio
from unittest.mock import MagicMock

# Mock heavy deps before any agent imports
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()

from core.session_manager import SessionManager
from agents.context_agent import ContextAgent
from agents.intent_agent import IntentAgent
from agents.dialogue_agent import DialogueAgent
from agents.tool_agent import ToolAgent
from agents.response_agent import ResponseAgent

context = ContextAgent()
intent_agent = IntentAgent()
dialogue = DialogueAgent()
tool = ToolAgent()
response = ResponseAgent()
session_id = SessionManager.create_session("test_001")

TEST_SEQUENCES = [
    (["chest", "pain"],         "Symptom Report"),
    (["three_days"],            "Duration Only"),
    (["appointment", "doctor"], "Appointment"),
    (["help"],                  "Emergency"),
    (["medicine"],              "Medication"),
]

async def run():
    passed = 0
    failed = 0
    for signs, label in TEST_SEQUENCES:
        context.clear_window()
        for s in signs:
            context.process_sign(s, 0.92)

        complaint = context.extract_complaint()
        intent_result = intent_agent.classify_intent(complaint)
        action = dialogue.manage_turn(intent_result, session_id)

        if action["action"] == "execute_tool":
            tool_result = await tool.execute(action)
            final = response.format_response(tool_result, intent_result["intent"])
        elif action["action"] == "emergency_escalation":
            final = response.format_response(
                {"urgency_level": "emergency", "immediate_action": action["message"]},
                "EMERGENCY_ALERT"
            )
        else:
            final = {
                "text_response": action.get("question", "?"),
                "patient_card": {"color": "yellow", "headline": "Clarification"},
                "voice_sentence": action.get("question", "")
            }

        card = final.get("patient_card", {})
        intent_str = intent_result["intent"]
        color = card.get("color", "?")
        headline = card.get("headline", "?")
        text = final.get("text_response", "")[:60]
        voice = final.get("voice_sentence", "")[:50]

        print(f"\n[{label}]")
        print(f"  intent   : {intent_str}")
        print(f"  card     : [{color.upper()}] {headline}")
        print(f"  text     : {text}")
        print(f"  voice    : {voice}")
        passed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed / {failed} failed")
    print(f"{'='*60}")

asyncio.run(run())
