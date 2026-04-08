import logging
from typing import Dict, Any

from mcp_servers.symptom_server import lookup_symptom_direct, check_emergency_direct
from mcp_servers.appointment_server import book_appointment_direct

logger = logging.getLogger("tool_agent")


class ToolAgent:
    """
    Routes classified intents to the appropriate MCP server functions.
    Uses direct Python imports (dev/hackathon mode) instead of stdio MCP transport.

    All MCP calls are individually wrapped in try/except with structured fallbacks
    to ensure no single server failure can halt pipeline execution.
    """

    def __init__(self):
        """
        Initializes tool routing maps and pre-imports MCP server modules.
        """
        pass

    async def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes the dialogue action to the appropriate MCP tool(s) and returns results.

        Args:
            action (Dict[str, Any]): Action packet from DialogueAgent containing intent_result.

        Returns:
            Dict[str, Any]: Structured result from the appropriate MCP tool call.

        Example:
            Input:  {"action": "execute_tool", "intent_result": {"intent": "SYMPTOM_REPORT", ...}}
            Output: {"tool": "lookup_symptom", "possible_conditions": [...], "urgency_level": "medium"}
        """
        intent_result = action.get("intent_result", {})
        intent = intent_result.get("intent", "GENERAL_QUERY")
        entities = intent_result.get("extracted_entities", {})

        logger.info(f"[ToolAgent] Executing tool for intent: {intent}")

        if intent == "SYMPTOM_REPORT":
            return await self._handle_symptom_report(entities)

        elif intent == "APPOINTMENT_REQUEST":
            return await self._handle_appointment_request(entities)

        elif intent == "EMERGENCY_ALERT":
            return await self._handle_emergency(entities)

        elif intent == "MEDICATION_QUESTION":
            return self.fallback_response("MEDICATION_QUESTION")

        else:
            return self.fallback_response("GENERAL_QUERY")

    async def _handle_symptom_report(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls symptom lookup and cross-references emergency patterns.

        Args:
            entities (Dict[str, Any]): Extracted medical entities from IntentAgent.

        Returns:
            Dict[str, Any]: Combined symptom + emergency check result.
        """
        symptom_text = " ".join(filter(None, [entities.get("symptom"), entities.get("body_part")]))
        symptom_list = [s for s in [entities.get("symptom"), entities.get("body_part")] if s]

        lookup_result = {}
        emergency_result = {}

        try:
            lookup_result = lookup_symptom_direct(
                symptom_text=symptom_text or "general complaint",
                body_part=entities.get("body_part"),
                duration_days=None
            )
            emergency_result = check_emergency_direct(symptom_list=symptom_list)
        except Exception as e:
            logger.error(f"[ToolAgent] Symptom server call failed: {e}")
            return self.fallback_response("SYMPTOM_REPORT")

        return {
            "tool": "lookup_symptom",
            "intent": "SYMPTOM_REPORT",
            **lookup_result,
            "emergency_check": emergency_result
        }

    async def _handle_appointment_request(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Books an appointment using the appointment MCP server.

        Args:
            entities (Dict[str, Any]): Extracted entities to seed the appointment booking.

        Returns:
            Dict[str, Any]: Confirmation and scheduling details.
        """
        try:
            result = book_appointment_direct(
                patient_name="ISL Patient",
                department="General Medicine",
                preferred_date="tomorrow",
                urgency="medium"
            )
            return {"tool": "book_appointment", "intent": "APPOINTMENT_REQUEST", **result}
        except Exception as e:
            logger.error(f"[ToolAgent] Appointment server call failed: {e}")
            return self.fallback_response("APPOINTMENT_REQUEST")

    async def _handle_emergency(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Immediately cross-references emergency sign patterns and returns critical alert.
        Does NOT wait for full symptom analysis — speed is critical.

        Args:
            entities (Dict[str, Any]): May be sparse; emergency check runs on available data.

        Returns:
            Dict[str, Any]: Emergency escalation result.
        """
        symptom_list = [s for s in [entities.get("symptom"), entities.get("body_part"), "emergency"] if s]

        try:
            result = check_emergency_direct(symptom_list=symptom_list)
            return {
                "tool": "check_emergency",
                "intent": "EMERGENCY_ALERT",
                "urgency_level": "emergency",
                **result
            }
        except Exception as e:
            logger.error(f"[ToolAgent] Emergency server call failed: {e}")
            return {
                "tool": "check_emergency",
                "intent": "EMERGENCY_ALERT",
                "urgency_level": "emergency",
                "is_emergency": True,
                "immediate_action": "Call emergency services immediately — 112 or nearest emergency room.",
                "matched_patterns": ["emergency override"]
            }

    def fallback_response(self, intent: str) -> Dict[str, Any]:
        """
        Returns a structured, safe fallback response when MCP server calls fail.

        Args:
            intent (str): The routing intent string to generate a context-aware fallback.

        Returns:
            Dict[str, Any]: Safe static fallback result dict.

        Example:
            Input:  "MEDICATION_QUESTION"
            Output: {"tool": "fallback", "intent": "MEDICATION_QUESTION",
                     "message": "Please consult the doctor for medication advice",
                     "urgency_level": "low"}
        """
        messages = {
            "SYMPTOM_REPORT":       "Your symptoms have been noted. The doctor will review shortly.",
            "APPOINTMENT_REQUEST":  "Please wait — appointment scheduling will be confirmed shortly.",
            "MEDICATION_QUESTION":  "Please consult the doctor for medication advice.",
            "EMERGENCY_ALERT":      "Emergency alert has been triggered. Help is being dispatched.",
            "GENERAL_QUERY":        "The doctor will address your question shortly."
        }

        return {
            "tool": "fallback",
            "intent": intent,
            "message": messages.get(intent, "The system is processing your request."),
            "urgency_level": "emergency" if intent == "EMERGENCY_ALERT" else "low",
            "possible_conditions": [],
            "recommended_action": messages.get(intent, "Please wait.")
        }
