import logging
from typing import Dict, Any

logger = logging.getLogger("intent_agent")

class IntentAgent:
    """
    Classifies the user's ultimate telehealth intent based on the structured medical 
    complaint mapping. Calculates rule-based logic to evaluate ambiguity and 
    prompt follow-up clarification questions directly.
    """
    def __init__(self):
        """
        Initializes explicit ontological groupings for intent cross-referencing.
        """
        self.EMERGENCY_WORDS = ["emergency", "help", "breathe", "fall", "unconscious"]
        self.APPOINTMENT_WORDS = ["appointment", "doctor", "schedule", "when", "book"]
        self.MEDICATION_WORDS = ["medicine", "tablet", "dose", "drug", "prescription"]

    def classify_intent(self, complaint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps the clinical complaint extracted by ContextAgent into a predefined system intent.
        Evaluates requirements for doctor-side clarification logic.
        
        Args:
            complaint (Dict[str, Any]): Structured diagnostic semantic grouping.
            
        Returns:
            Dict[str, Any]: The classification packet routing the dialogue manager.
        """
        raw_signs = complaint.get("raw_signs", [])
        
        # 1. EMERGENCY ROUTING (Absolute Priority Override)
        if any(word in raw_signs for word in self.EMERGENCY_WORDS):
            return self._build_intent("EMERGENCY_ALERT", 0.95, complaint, requires_clarification=False)
            
        # 2. SCHEDULING ROUTING
        if any(word in raw_signs for word in self.APPOINTMENT_WORDS):
            return self._build_intent("APPOINTMENT_REQUEST", 0.90, complaint, requires_clarification=False)
            
        # 3. PHARMACY ROUTING
        if any(word in raw_signs for word in self.MEDICATION_WORDS):
            return self._build_intent("MEDICATION_QUESTION", 0.90, complaint, requires_clarification=False)
            
        # 4. SYMPTOM ROUTING
        if complaint.get("body_part") and complaint.get("symptom"):
            # A full definitive reporting
            return self._build_intent("SYMPTOM_REPORT", 0.90, complaint, requires_clarification=False)
        elif complaint.get("body_part") or complaint.get("symptom"):
            # Partial reporting. Falls back conceptually but inherently lacks completeness
            return self._build_intent("SYMPTOM_REPORT", 0.60, complaint, requires_clarification=True)
            
        # 5. UNKNOWN/GENERAL ROUTING
        return self._build_intent("GENERAL_QUERY", 0.50, complaint, requires_clarification=True)
        
    def _build_intent(self, intent: str, confidence: float, complaint: dict, requires_clarification: bool = False) -> Dict[str, Any]:
        """
        Utility template generating standard agent output representations.
        Dynamically applies clarification hooks against lower bounding confidences.
        """
        
        # Ensure programmatic triggers are evaluated cohesively 
        if confidence < 0.75:
            requires_clarification = True
            
        clarification_question = None
        if requires_clarification:
            if intent == "SYMPTOM_REPORT":
                if not complaint.get("body_part"):
                    clarification_question = "Can you show me where the pain or issue is located?"
                elif not complaint.get("symptom"):
                    clarification_question = "Can you describe what you are feeling? (e.g., pain, ache)"
                elif not complaint.get("duration"):
                    clarification_question = "How long have you had this symptom?"
                else:
                    clarification_question = "Can you provide more details about how you are feeling?"
            else:
                clarification_question = "I did not fully understand. Can you repeat or clarify your request?"
                
        return {
            "intent": intent,
            "confidence": confidence,
            "extracted_entities": {
                "body_part": complaint.get("body_part"),
                "symptom": complaint.get("symptom"),
                "duration": complaint.get("duration"),
                "severity": complaint.get("severity")
            },
            "requires_clarification": requires_clarification,
            "clarification_question": clarification_question
        }
