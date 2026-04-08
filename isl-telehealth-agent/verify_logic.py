"""
End-to-end logic verification for the ISL Telehealth Agent pipeline.
Mocks heavy dependencies (cv2, mediapipe, tensorflow) and injects demo signs directly
into the context and intent agents, bypassing perception entirely.
"""

import sys
import asyncio
import logging
import numpy as np
from unittest.mock import MagicMock

# --- Mock heavy dependencies BEFORE any agent imports ---
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()

from core.session_manager import SessionManager
from agents.context_agent import ContextAgent
from agents.intent_agent import IntentAgent
from agents.dialogue_agent import DialogueAgent
from agents.tool_agent import ToolAgent
from agents.response_agent import ResponseAgent

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_logic")

async def run_sign_sequence(signs: list, session_id: str,
                            context: ContextAgent, intent: IntentAgent,
                            dialogue: DialogueAgent, tool: ToolAgent,
                            response: ResponseAgent) -> dict | None:
    """Injects a sequence of signs directly into the pipeline, bypassing perception."""
    context.clear_window()
    for sign in signs:
        context.process_sign(sign, 0.92)

    complaint = context.extract_complaint()
    intent_result = intent.classify_intent(complaint)
    action = dialogue.manage_turn(intent_result, session_id)

    if action["action"] == "execute_tool":
        tool_result = await tool.execute(action)
        final_response = response.format_response(tool_result, intent_result["intent"])
        dialogue.add_doctor_response(final_response["text_response"])
        return final_response
    elif action["action"] == "ask_clarification":
        return {
            "text_response": action["question"],
            "patient_card": {"color": "yellow", "headline": "Clarification", "detail": action["question"], "icon": "❓"},
            "voice_sentence": action["question"],
            "urgency_color": "yellow"
        }
    elif action["action"] == "emergency_escalation":
        final_response = response.format_response({"urgency_level": "emergency", "immediate_action": action["message"]}, "EMERGENCY_ALERT")
        return final_response

    return None


async def main():
    logger.info("=" * 60)
    logger.info(" ISL TELEHEALTH PIPELINE — LOGIC VERIFICATION")
    logger.info("=" * 60)

    session_id = SessionManager.create_session("test_patient_001")

    context = ContextAgent()
    intent = IntentAgent()
    dialogue = DialogueAgent()
    tool = ToolAgent()
    response = ResponseAgent()

    TEST_SEQUENCES = [
        (["chest", "pain"],        "Symptom Report"),
        (["three_days"],           "Incomplete (Duration Only)"),
        (["appointment", "doctor"],"Appointment Request"),
        (["help"],                 "Emergency"),
        (["medicine"],             "Medication Question"),
    ]

    all_passed = True
    for signs, label in TEST_SEQUENCES:
        logger.info(f"\n--- TEST: {label} | Signs: {signs} ---")
        result = await run_sign_sequence(
            signs, session_id, context, intent, dialogue, tool, response
        )
        if result:
            card = result.get("patient_card", {})
            logger.info(f"  ✅ Text    : {result.get('text_response', '')[:80]}")
            logger.info(f"  ✅ Card    : [{card.get('color','?').upper()}] {card.get('headline')} — {card.get('detail','')[:50]}")
            logger.info(f"  ✅ Voice   : {result.get('voice_sentence')}")
        else:
            logger.error(f"  ❌ No response returned for: {signs}")
            all_passed = False

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info(" ✅ ALL PIPELINE TESTS PASSED")
    else:
        logger.error(" ❌ SOME TESTS FAILED — see above")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
