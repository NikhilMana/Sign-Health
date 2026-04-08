import os
import json
import logging
from collections import deque
from typing import Dict, Any, List, Optional

logger = logging.getLogger("context_agent")

class ContextAgent:
    """
    Analyzes sequences of ISL signs sequentially to construct a structured clinical context
    and extract definitive medical complaints based on a target medical vocabulary.
    """
    def __init__(self, config_path: str = "config/medical_vocabulary.json"):
        """
        Initializes the agent, the sliding window of 10 maximum signs, and bounds the vocabulary.
        """
        self.window = deque(maxlen=10)
        self.medical_vocabulary = self._load_vocabulary(config_path)
        
    def _load_vocabulary(self, config_path: str) -> dict:
        """
        Loads the JSON medical schema. Falls back to a specific inline dict if unavailable
        or if the file does not contain the expected 'symptoms' and 'body_parts' keys.
        """
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                # Validate: file must have meaningful clinical content
                if data.get("symptoms") and data.get("body_parts"):
                    logger.info(f"Loaded medical vocabulary from {config_path}")
                    return data
                else:
                    logger.warning(f"Vocabulary file exists at {config_path} but is missing 'symptoms'/'body_parts'. Using inline fallback.")
            except Exception as e:
                logger.warning(f"Failed to load vocabulary from {config_path}: {e}")

                
        # Inline fallback payload mapping exactly to model expectations
        logger.info("Using inline fallback medical vocabulary.")
        return {
            "symptoms": [
                "pain", "ache", "burn", "itch", "swell", "bleed", "dizzy", 
                "nausea", "vomit", "cough", "fever", "headache", "fatigue", 
                "numb", "rash"
            ],
            "body_parts": [
                "head", "neck", "chest", "back", "stomach", "arm", "leg", 
                "hand", "foot", "eye", "ear", "nose", "throat", "heart", "lung"
            ],
            "time_indicators": {
                "today": 0, "yesterday": 1, "two_days": 2, 
                "three_days": 3, "one_week": 7, "one_month": 30
            },
            "severity_words": {
                "mild": 2, "moderate": 5, "severe": 8, 
                "unbearable": 10, "slight": 1
            }
        }
        
    def process_sign(self, sign: str, confidence: float) -> Dict[str, Any]:
        """
        Receives an individually recognized sign, classifies its medical ontological category,
        and adds it to the sliding window history.
        
        Args:
            sign (str): The string name of the recognized sign.
            confidence (float): Model confidence between 0.0 and 1.0.
            
        Returns:
            Dict[str, Any]: A dictionary capturing the current window scope.
        """
        sign_lower = sign.lower()
        category = "unknown"
        
        if sign_lower in self.medical_vocabulary.get("symptoms", []):
            category = "symptom"
        elif sign_lower in self.medical_vocabulary.get("body_parts", []):
            category = "body_part"
        elif sign_lower in self.medical_vocabulary.get("time_indicators", {}):
            category = "duration"
        elif sign_lower in self.medical_vocabulary.get("severity_words", {}):
            category = "severity"
            
        entry = {
            "sign": sign_lower,
            "category": category,
            "confidence": confidence
        }
        
        self.window.append(entry)
        
        return {
            "current_window": list(self.window),
            "latest_sign": entry
        }
        
    def extract_complaint(self) -> Dict[str, Any]:
        """
        Evolves the current sliding window into a structured diagnostic grouping.
        Calculates extraction completeness scalars determining ambiguity.
        
        Returns:
            Dict[str, Any]: A structured semantic complaint object.
        """
        complaint = {
            "complaint_type": "unknown",
            "body_part": None,
            "symptom": None,
            "duration": None,
            "severity": None,
            "raw_signs": [],
            "completeness_score": 0.2, # Baseline assumption
            "ambiguous": True
        }
        
        # Sequentially map components recognizing earliest instances first
        for item in self.window:
            complaint["raw_signs"].append(item["sign"])
            
            if item["category"] == "body_part" and not complaint["body_part"]:
                complaint["body_part"] = item["sign"]
            elif item["category"] == "symptom" and not complaint["symptom"]:
                complaint["symptom"] = item["sign"]
            elif item["category"] == "duration" and not complaint["duration"]:
                complaint["duration"] = item["sign"]
            elif item["category"] == "severity" and not complaint["severity"]:
                complaint["severity"] = item["sign"]
                
        has_part = complaint["body_part"] is not None
        has_symptom = complaint["symptom"] is not None
        
        if has_part and has_symptom:
            complaint["completeness_score"] = 1.0
            complaint["ambiguous"] = False
            complaint["complaint_type"] = "symptom_report"
        elif has_part or has_symptom:
            complaint["completeness_score"] = 0.5
            complaint["ambiguous"] = True
            complaint["complaint_type"] = "partial_symptom_report"
            
        return complaint
        
    def clear_window(self) -> None:
        """
        Erases the sliding window. Must be called downstream once a complaint 
        has successfully reached tool resolution or intent completion.
        """
        self.window.clear()
        
