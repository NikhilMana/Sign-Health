import os
import sys
import pytest

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.perception_agent import PerceptionAgent
from agents.context_agent import ContextAgent
from agents.intent_agent import IntentAgent
from agents.response_agent import ResponseAgent

def test_perception_demo_mode():
    agent = PerceptionAgent()
    agent.demo_mode = True
    result = agent.get_demo_prediction()
    
    assert "top_predictions" in result, "Result must contain top_predictions"
    assert isinstance(result["top_predictions"], list), "top_predictions must be a list"
    assert len(result["top_predictions"]) == 3, "top_predictions must contain exactly 3 items"

def test_perception_model_paths():
    # The models are in ../sign_language_interpreter/models/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # We check the actual path that perception agent uses or the expected files
    model_path = os.path.join(base_dir, "sign_language_interpreter", "models", "sign_model.keras")
    encoder_path = os.path.join(base_dir, "sign_language_interpreter", "models", "label_encoder.pkl")
    
    # We only test if these paths map to existing files (as required)
    assert os.path.exists(model_path) or os.path.exists(os.path.join(base_dir, "sign_language_interpreter", "models", "best_model.keras")), f"Model path must exist"
    assert os.path.exists(encoder_path), f"Encoder path must exist at {encoder_path}"

def test_context_complaint_extraction():
    agent = ContextAgent()
    signs = ["chest", "pain", "three_days"]
    
    for sign in signs:
        agent.process_sign(sign, 0.95)
        
    complaint = agent.extract_complaint()
    
    assert complaint.get("body_part") == "chest"
    assert complaint.get("symptom") == "pain"
    assert complaint.get("completeness_score", 0) >= 0.8

def test_intent_emergency():
    agent = IntentAgent()
    complaint = {
        "complaint_type": "symptom_report",
        "body_part": None,
        "symptom": None,
        "duration": None,
        "severity": None,
        "raw_signs": ["emergency", "help"],
        "completeness_score": 0.5,
        "ambiguous": True
    }
    
    result = agent.classify_intent(complaint)
    
    assert result.get("intent") == "EMERGENCY_ALERT"
    assert result.get("confidence", 0) >= 0.90

def test_response_all_formats():
    agent = ResponseAgent()
    tool_result = {
        "urgency_level": "medium",
        "possible_conditions": ["Headache", "Dehydration"],
        "recommended_action": "Rest and hydrate."
    }
    intent = "SYMPTOM_REPORT"
    
    result = agent.format_response(tool_result, intent)
    
    assert "text_response" in result
    assert "patient_card" in result
    assert "voice_sentence" in result
    
    # Ensure patient_card is a dict
    assert isinstance(result["patient_card"], dict)
    
    # Specifics for symptom report
    assert "Headache" in result["text_response"]
