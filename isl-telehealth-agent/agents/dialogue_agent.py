import time
import logging
from typing import Dict, Any, Optional

from mcp_servers.medical_notes_server import create_session_note_direct

logger = logging.getLogger("dialogue_agent")


class DialogueAgent:
    """
    Manages the conversational flow between patient sign inputs and the medical system.
    Maintains full turn history and orchestrates routing decisions based on intent.
    """

    def __init__(self):
        """
        Initializes dialogue history and links session state.
        """
        self.history: list = []
        self.session_id: Optional[str] = None
        self.turn_count = 0

    def manage_turn(self, intent_result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Evaluates the classified intent and decides the system's next action.

        Args:
            intent_result (Dict[str, Any]): The intent classification packet from IntentAgent.
            session_id (str): The current active session identifier.

        Returns:
            Dict[str, Any]: Action routing dict consumed by the ToolAgent or Dialogue pipeline.

        Example returns:
            {"action": "ask_clarification", "question": "Can you show where the pain is?"}
            {"action": "emergency_escalation", "message": "EMERGENCY DETECTED..."}
            {"action": "execute_tool", "intent_result": {...}}
        """
        self.session_id = session_id
        self.turn_count += 1

        # Record patient side of this turn
        self.history.append({
            "turn": self.turn_count,
            "speaker": "patient",
            "content": str(intent_result.get("extracted_entities", {})),
            "timestamp": time.time(),
            "intent": intent_result.get("intent", "UNKNOWN")
        })

        intent = intent_result.get("intent", "")

        # EMERGENCY OVERRIDE — Highest priority, bypasses clarification logic
        if intent == "EMERGENCY_ALERT":
            logger.warning(f"[Session {session_id}] EMERGENCY_ALERT triggered!")
            return {
                "action": "emergency_escalation",
                "message": "EMERGENCY DETECTED — alerting medical staff immediately"
            }

        # CLARIFICATION ROUTING — Insufficient context to proceed
        if intent_result.get("requires_clarification"):
            question = intent_result.get("clarification_question", "Can you please repeat that?")
            logger.info(f"[Session {session_id}] Clarification needed: {question}")
            return {
                "action": "ask_clarification",
                "question": question
            }

        # NORMAL EXECUTION ROUTING
        logger.info(f"[Session {session_id}] Routing to tool execution for intent: {intent}")
        return {
            "action": "execute_tool",
            "intent_result": intent_result
        }

    def add_doctor_response(self, response_text: str) -> None:
        """
        Appends the system-generated doctor response to the conversation history
        and stubs a medical notes server persistence call.

        Args:
            response_text (str): The formatted text output from ResponseAgent.
        """
        self.turn_count += 1
        self.history.append({
            "turn": self.turn_count,
            "speaker": "doctor",
            "content": response_text,
            "timestamp": time.time(),
            "intent": "SYSTEM_RESPONSE"
        })

        # Stub: direct Python import call in dev/hackathon mode
        try:
            create_session_note_direct(
                session_id=self.session_id or "unknown",
                doctor_response=response_text
            )
        except Exception as e:
            logger.debug(f"Medical notes stub (non-critical): {e}")

    def get_conversation_context(self) -> str:
        """
        Returns the last 5 turns of conversation as a formatted, readable string.

        Returns:
            str: Formatted conversation context for feeding upstream agents.

        Example:
            "[Turn 1] patient: chest pain | intent: SYMPTOM_REPORT
             [Turn 2] doctor: Symptom logged. Checking..."
        """
        last_five = self.history[-5:]
        lines = []
        for turn in last_five:
            speaker = turn["speaker"].upper()
            lines.append(f"[Turn {turn['turn']}] {speaker}: {turn['content']} | intent: {turn['intent']}")
        return "\n".join(lines) if lines else "No conversation history yet."
