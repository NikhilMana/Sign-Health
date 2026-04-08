import logging
from typing import Dict, Any

logger = logging.getLogger("response_agent")

# Urgency → color mapping table
URGENCY_COLOR_MAP = {
    "low":       "green",
    "medium":    "yellow",
    "high":      "orange",
    "emergency": "red"
}

# Intent → emoji icon mapping
INTENT_ICON_MAP = {
    "SYMPTOM_REPORT":      "🏥",
    "APPOINTMENT_REQUEST": "📅",
    "MEDICATION_QUESTION": "💊",
    "EMERGENCY_ALERT":     "🚨",
    "GENERAL_QUERY":       "❓"
}


class ResponseAgent:
    """
    Converts raw MCP tool results into three simultaneous output formats:
    text (for doctor screen), patient_card (for UI tile), and voice_sentence (for TTS).

    EMERGENCY RULE: When intent is EMERGENCY_ALERT, patient_card color is always
    "red" and headline is always "GET HELP NOW" — no exceptions.
    """

    def format_response(self, tool_result: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        Generates all three output formats simultaneously from the tool execution result.

        Args:
            tool_result (Dict[str, Any]): Structured output from ToolAgent.execute().
            intent (str): The classified intent string from IntentAgent.

        Returns:
            Dict[str, Any]: Three output formats plus urgency metadata.

        Example output:
            {
                "text_response": "Patient reports chest pain for 3 days. Possible: Angina (high urgency).",
                "patient_card": {"icon": "🏥", "headline": "Chest Pain Reported",
                                 "detail": "Possible angina — seek urgent care", "color": "orange"},
                "voice_sentence": "Chest pain detected. Doctor is being notified now.",
                "urgency_color": "orange"
            }
        """
        # EMERGENCY OVERRIDE — Unconditional. No tool_result can alter this.
        if intent == "EMERGENCY_ALERT":
            return self._emergency_response(tool_result)

        urgency_level = tool_result.get("urgency_level", "low")
        urgency_color = self.get_urgency_color(urgency_level)
        icon = INTENT_ICON_MAP.get(intent, "❓")

        if intent == "SYMPTOM_REPORT":
            return self._format_symptom_response(tool_result, urgency_color, icon)
        elif intent == "APPOINTMENT_REQUEST":
            return self._format_appointment_response(tool_result, urgency_color, icon)
        elif intent == "MEDICATION_QUESTION":
            return self._format_medication_response(tool_result, urgency_color, icon)
        else:
            return self._format_general_response(tool_result, urgency_color, icon)

    def _emergency_response(self, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a hardcoded maximum-urgency emergency output.
        patient_card.color is always "red", headline always "GET HELP NOW".
        """
        immediate_action = tool_result.get(
            "immediate_action",
            "Call emergency services immediately — 112 or nearest emergency room."
        )
        return {
            "text_response": f"⚠️ EMERGENCY ALERT: {immediate_action} Medical staff have been notified.",
            "patient_card": {
                "icon": "🚨",
                "headline": "GET HELP NOW",
                "detail": immediate_action[:80],
                "color": "red"   # HARDCODED — never override
            },
            "voice_sentence": "Emergency detected. Help is on the way. Stay calm.",
            "urgency_color": "red"
        }

    def _format_symptom_response(self, tool_result: Dict[str, Any], urgency_color: str, icon: str) -> Dict[str, Any]:
        """
        Formats a SYMPTOM_REPORT response from lookup_symptom tool output.

        Example output text: "Patient reports chest pain. Possible conditions: Angina, GERD.
                              Recommended: Seek urgent medical evaluation."
        """
        conditions = tool_result.get("possible_conditions", [])
        conditions_str = ", ".join(conditions) if conditions else "under evaluation"
        recommended = tool_result.get("recommended_action", "Please consult the doctor.")
        urgency_level = tool_result.get("urgency_level", "low")

        text = f"Patient reports symptom. Possible conditions: {conditions_str}. Recommended: {recommended}"

        headline_map = {
            "green":  "Symptom Noted",
            "yellow": "Review Required",
            "orange": "Urgent Review",
            "red":    "Critical Symptom"
        }

        detail = f"{conditions_str[:40]} — {recommended[:30]}" if conditions else recommended[:60]

        voice_map = {
            "low":    f"Symptom recorded. The doctor will review your case.",
            "medium": f"Your symptom has been noted. Please wait for the doctor.",
            "high":   f"Urgent symptom detected. Doctor is being notified now.",
            "emergency": "Critical symptom. Seeking emergency help immediately."
        }

        return {
            "text_response": text,
            "patient_card": {
                "icon": icon,
                "headline": headline_map.get(urgency_color, "Symptom Noted"),
                "detail": detail,
                "color": urgency_color
            },
            "voice_sentence": voice_map.get(urgency_level, "Your concern has been noted."),
            "urgency_color": urgency_color
        }

    def _format_appointment_response(self, tool_result: Dict[str, Any], urgency_color: str, icon: str) -> Dict[str, Any]:
        """
        Formats an APPOINTMENT_REQUEST response from book_appointment tool output.

        Example voice output: "Your appointment has been booked for tomorrow morning."
        """
        doctor = tool_result.get("doctor_name", "the next available doctor")
        scheduled = tool_result.get("scheduled_datetime", "soon")
        confirmation = tool_result.get("confirmation_id", "N/A")
        instructions = tool_result.get("instructions", "Arrive 15 minutes early.")

        text = f"Appointment booked with {doctor} on {scheduled}. Confirmation ID: {confirmation}. {instructions}"

        return {
            "text_response": text,
            "patient_card": {
                "icon": icon,
                "headline": "Appointment Booked",
                "detail": f"With {doctor} · {str(scheduled)[:20]}",
                "color": urgency_color or "green"
            },
            "voice_sentence": f"Your appointment is confirmed with {doctor}.",
            "urgency_color": urgency_color or "green"
        }

    def _format_medication_response(self, tool_result: Dict[str, Any], urgency_color: str, icon: str) -> Dict[str, Any]:
        """
        Formats a MEDICATION_QUESTION response with a standardized safety message.

        Example: "Please consult the doctor before taking any medication."
        """
        message = tool_result.get("message", "Please consult the doctor for medication advice.")

        return {
            "text_response": f"Medication query received. {message}",
            "patient_card": {
                "icon": icon,
                "headline": "Medication Query",
                "detail": "Doctor will advise on safe dosage.",
                "color": "yellow"
            },
            "voice_sentence": "Please wait. The doctor will advise on your medication.",
            "urgency_color": "yellow"
        }

    def _format_general_response(self, tool_result: Dict[str, Any], urgency_color: str, icon: str) -> Dict[str, Any]:
        """
        Formats a GENERAL_QUERY fallback response.

        Example: "Your question has been received. The doctor will address it shortly."
        """
        message = tool_result.get("message", "The doctor will address your question shortly.")

        return {
            "text_response": f"Patient query received. {message}",
            "patient_card": {
                "icon": icon,
                "headline": "Query Received",
                "detail": "Doctor will respond shortly.",
                "color": "green"
            },
            "voice_sentence": "Your question has been received. Please wait.",
            "urgency_color": "green"
        }

    def get_urgency_color(self, urgency_level: str) -> str:
        """
        Maps a medical urgency level string to its UI color code.

        Args:
            urgency_level (str): One of "low", "medium", "high", "emergency".

        Returns:
            str: CSS-compatible color string.

        Examples:
            "low"       → "green"
            "medium"    → "yellow"
            "high"      → "orange"
            "emergency" → "red"
        """
        return URGENCY_COLOR_MAP.get(urgency_level.lower() if urgency_level else "low", "green")
