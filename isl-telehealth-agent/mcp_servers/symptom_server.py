"""
Symptom Server: MCP server for checking and recording medical symptoms.
"""
import sys
import json
import asyncio
import logging
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, ValidationError

# MCP SDK Standard Imports
from mcp.server import Server, NotificationOptions
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

# =====================================================================
# Server & Logging Setup
# Ensure logging goes to stderr so it doesn't corrupt MCP's stdout JSON
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("symptom_server")

app = Server("symptom_server")

# =====================================================================
# Hardcoded Knowledge Databases
# =====================================================================
SYMPTOMS_DB = {
    "chest pain": {
        "conditions": ["Myocardial Infarction", "Angina", "GERD"],
        "urgency": "emergency",
        "action": "Immediate emergency room visit.",
        "emergency_trigger": "If radiating to jaw/arm or accompanied by sweating."
    },
    "headache": {
        "conditions": ["Migraine", "Tension Headache", "Hypertension"],
        "urgency": "medium",
        "action": "Rest, hydrate, and take OTC pain relievers.",
        "emergency_trigger": "If sudden and severe ('worst headache of life') or with vision loss."
    },
    "fever": {
        "conditions": ["Viral Infection", "Bacterial Infection", "Influenza"],
        "urgency": "medium",
        "action": "Rest, take antipyretics, monitor temperature.",
        "emergency_trigger": "If above 103 F (39.4 C) or accompanied by stiff neck / confusion."
    },
    "breathlessness": {
        "conditions": ["Asthma", "COPD", "Pulmonary Embolism"],
        "urgency": "high",
        "action": "Use inhaler if prescribed; seek eval if persistent.",
        "emergency_trigger": "If sudden onset, resting breathlessness, or blue lips."
    },
    "nausea": {
        "conditions": ["Gastroenteritis", "Food Poisoning", "Pregnancy"],
        "urgency": "low",
        "action": "Hydrate with clear fluids, eat bland foods.",
        "emergency_trigger": "If unable to keep fluids down for 24+ hours or blood in vomit."
    },
    "dizziness": {
        "conditions": ["Vertigo", "Dehydration", "Hypotension"],
        "urgency": "medium",
        "action": "Sit or lie down, hydrate slowly.",
        "emergency_trigger": "If accompanied by fainting, chest pain, or slurred speech."
    },
    "back pain": {
        "conditions": ["Muscle Strain", "Herniated Disc", "Sciatica"],
        "urgency": "low",
        "action": "Rest, apply heat/ice, take OTC analgesics.",
        "emergency_trigger": "If accompanied by loss of bowel/bladder control or severe numbness."
    },
    "joint pain": {
        "conditions": ["Osteoarthritis", "Rheumatoid Arthritis", "Gout"],
        "urgency": "low",
        "action": "Rest affected joint, apply ice, elevate.",
        "emergency_trigger": "If joint is hot, exceptionally swollen, or unable to bear any weight."
    },
    "cough": {
        "conditions": ["Common Cold", "Bronchitis", "Pneumonia"],
        "urgency": "low",
        "action": "Stay hydrated, use cough suppressants if needed.",
        "emergency_trigger": "If coughing up blood or experiencing severe shortness of breath."
    },
    "fatigue": {
        "conditions": ["Anemia", "Sleep Apnea", "Depression"],
        "urgency": "low",
        "action": "Ensure adequate rest, hydration, and nutrition.",
        "emergency_trigger": "If sudden, profound fatigue combined with chest pain or confusion."
    },
    "abdominal pain": {
        "conditions": ["Appendicitis", "Gallstones", "Gastritis"],
        "urgency": "high",
        "action": "Monitor pain progression; avoid eating solid foods temporarily.",
        "emergency_trigger": "If pain is severe, localized in lower right quadrant, or abdomen is rigid."
    },
    "vision blur": {
        "conditions": ["Refractive Error", "Cataracts", "Diabetic Retinopathy"],
        "urgency": "medium",
        "action": "Schedule an appointment with an optometrist or ophthalmologist.",
        "emergency_trigger": "If sudden vision loss or accompanied by severe eye pain or headache."
    },
    "hearing loss": {
        "conditions": ["Earwax Blockage", "Ear Infection", "Sensorineural Hearing Loss"],
        "urgency": "medium",
        "action": "Schedule evaluation with a doctor or audiologist.",
        "emergency_trigger": "If sudden onset in one or both ears, or accompanied by vertigo."
    },
    "skin rash": {
        "conditions": ["Contact Dermatitis", "Eczema", "Allergic Reaction"],
        "urgency": "low",
        "action": "Apply hydrocortisone, avoid suspected triggers.",
        "emergency_trigger": "If rapidly spreading, painful, blistering, or accompanied by difficulty breathing."
    },
    "numbness": {
        "conditions": ["Neuropathy", "Pinched Nerve", "Transient Ischemic Attack (TIA)"],
        "urgency": "high",
        "action": "Determine if persistent or transient; seek medical advice.",
        "emergency_trigger": "If sudden onset on one side of face/body or accompanied by weakness/slurred speech."
    }
}

EMERGENCY_PATTERNS = [
    {
        "symptoms": ["chest pain", "breathlessness"],
        "type": "Cardiac Emergency",
        "action": "Call emergency services immediately. Do not drive yourself."
    },
    {
        "symptoms": ["severe headache", "vision blur"],
        "type": "Neurological Emergency",
        "action": "Seek immediate emergency medical care; suspicious for stroke or hemorrhage."
    },
    {
        "symptoms": ["numbness", "vision blur"],
        "type": "Stroke or TIA",
        "action": "Call emergency services. Time is critical."
    },
    {
        "symptoms": ["fever", "skin rash"],
        "type": "Severe Infection / Meningitis or Anaphylaxis",
        "action": "Seek urgent medical evaluation."
    },
    {
        "symptoms": ["abdominal pain", "fever"],
        "type": "Appendicitis or intra-abdominal infection",
        "action": "Go to the emergency department."
    }
]

VOCABULARY = {
    "symptoms": list(SYMPTOMS_DB.keys()),
    "body_parts": ["head", "chest", "arm", "leg", "stomach", "back", "eye", "ear", "neck", "hand", "foot", "abdomen", "joint", "skin"],
    "medications": ["paracetamol", "ibuprofen", "antibiotics", "aspirin", "antihistamine", "antacids", "cough syrup"],
    "procedures": ["x-ray", "blood test", "mri", "surgery", "ultrasound", "ecg", "ct scan"]
}

# =====================================================================
# Pydantic Schemas for Tools
# =====================================================================

class LookupSymptomInput(BaseModel):
    symptom_text: str = Field(..., description="The symptom to look up")
    body_part: Optional[str] = Field(None, description="Optional affected body part")
    duration_days: Optional[int] = Field(None, description="Duration in days")

class CheckEmergencyInput(BaseModel):
    symptom_list: List[str] = Field(..., description="List of current symptoms experienced by the patient")

class GetMedicalVocabularyInput(BaseModel):
    category: Literal["symptoms", "body_parts", "medications", "procedures"] = Field(
        ..., description="Category of medical vocabulary to retrieve"
    )

# =====================================================================
# Tool Routing Setup
# =====================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Client requested list of tools.")
    return [
        Tool(
            name="lookup_symptom",
            description="Look up a symptom in the medical knowledge dictionary to get possible conditions, urgency level, and recommended actions.",
            inputSchema=LookupSymptomInput.model_json_schema(),
        ),
        Tool(
            name="check_emergency",
            description="Cross-reference a list of symptoms against emergency patterns to identify if immediate action is needed.",
            inputSchema=CheckEmergencyInput.model_json_schema(),
        ),
        Tool(
            name="get_medical_vocabulary",
            description="Get a list of ISL-recognized medical terms for a given category (symptoms, body_parts, medications, procedures).",
            inputSchema=GetMedicalVocabularyInput.model_json_schema(),
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Client called tool '{name}' with arguments: {arguments}")
    
    if name == "lookup_symptom":
        try:
            args = LookupSymptomInput(**arguments)
        except ValidationError as e:
            logger.error(f"Validation error for lookup_symptom: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        
        symp = args.symptom_text.lower()
        if symp in SYMPTOMS_DB:
            data = SYMPTOMS_DB[symp]
            result = {
                "possible_conditions": data["conditions"][:3],
                "urgency_level": data["urgency"],
                "recommended_action": data["action"],
                "when_to_seek_emergency": data["emergency_trigger"]
            }
        else:
            result = {
                "possible_conditions": ["Unknown Condition"],
                "urgency_level": "medium",
                "recommended_action": "Consult a healthcare professional for an accurate assessment.",
                "when_to_seek_emergency": "If symptoms worsen rapidly, become intolerable, or new severe symptoms appear."
            }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "check_emergency":
        try:
            args = CheckEmergencyInput(**arguments)
        except ValidationError as e:
            logger.error(f"Validation error for check_emergency: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        sl = [s.lower() for s in args.symptom_list]
        is_emergency = False
        emergency_type = "None"
        immediate_action = "Standard care appropriate."

        for pattern in EMERGENCY_PATTERNS:
            if all(ps in sl for ps in pattern["symptoms"]):
                is_emergency = True
                emergency_type = pattern["type"]
                immediate_action = pattern["action"]
                break

        result = {
            "is_emergency": is_emergency,
            "emergency_type": emergency_type,
            "immediate_action": immediate_action
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_medical_vocabulary":
        try:
            args = GetMedicalVocabularyInput(**arguments)
        except ValidationError as e:
            logger.error(f"Validation error for get_medical_vocabulary: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        resultList = VOCABULARY.get(args.category, [])
        return [TextContent(type="text", text=json.dumps({args.category: resultList}, indent=2))]

    else:
        logger.warning(f"Client attempted to call unsupported tool: {name}")
        raise ValueError(f"Tool '{name}' is not supported.")

# =====================================================================
# Direct Python-Callable Functions (Dev / Hackathon Mode)
# ToolAgent imports these directly without stdio MCP transport.
# =====================================================================

def lookup_symptom_direct(
    symptom_text: str,
    body_part: Optional[str] = None,
    duration_days: Optional[int] = None
) -> dict:
    """
    Directly callable version of the lookup_symptom MCP tool.
    Searches the SYMPTOMS_DB for matching conditions and urgency.

    Args:
        symptom_text (str): The symptom to look up.
        body_part (Optional[str]): Affected body part for context.
        duration_days (Optional[int]): Duration in days.

    Returns:
        dict: possible_conditions, urgency_level, recommended_action, when_to_seek_emergency.
    """
    # Try composite key first (e.g. "chest pain"), then individual symptom
    query = symptom_text.lower().strip()
    
    # Attempt direct match
    if query in SYMPTOMS_DB:
        data = SYMPTOMS_DB[query]
    else:
        # Fuzzy partial match — find any key that appears in the query string
        matched_key = next((k for k in SYMPTOMS_DB if k in query or query in k), None)
        data = SYMPTOMS_DB.get(matched_key) if matched_key else None
    
    if data:
        return {
            "possible_conditions": data["conditions"][:3],
            "urgency_level": data["urgency"],
            "recommended_action": data["action"],
            "when_to_seek_emergency": data["emergency_trigger"]
        }
    
    return {
        "possible_conditions": ["Unspecified Condition"],
        "urgency_level": "medium",
        "recommended_action": "Please consult a healthcare professional for an accurate assessment.",
        "when_to_seek_emergency": "If symptoms worsen rapidly or new severe symptoms appear."
    }


def check_emergency_direct(symptom_list: List[str]) -> dict:
    """
    Directly callable version of the check_emergency MCP tool.
    Cross-references given symptoms against known emergency patterns.

    Args:
        symptom_list (List[str]): Signs/symptoms reported by the patient.

    Returns:
        dict: is_emergency, emergency_type, immediate_action, matched_patterns.
    """
    sl = [s.lower().strip() for s in symptom_list]
    matched_patterns = []
    is_emergency = False
    emergency_type = "None"
    immediate_action = "Standard care appropriate. Monitor for changes."

    # Broad emergency keywords override pattern matching
    hard_triggers = ["emergency", "unconscious", "fall", "help", "breathe"]
    if any(t in sl for t in hard_triggers):
        return {
            "is_emergency": True,
            "emergency_type": "General Emergency",
            "immediate_action": "Call emergency services immediately — 112 or nearest emergency room.",
            "matched_patterns": ["emergency keyword detected"]
        }

    for pattern in EMERGENCY_PATTERNS:
        if all(ps in sl for ps in pattern["symptoms"]):
            is_emergency = True
            emergency_type = pattern["type"]
            immediate_action = pattern["action"]
            matched_patterns.append(pattern["symptoms"])

    return {
        "is_emergency": is_emergency,
        "emergency_type": emergency_type,
        "immediate_action": immediate_action,
        "matched_patterns": matched_patterns
    }


# =====================================================================
# Main Server Entrypoint
# =====================================================================
async def main():
    logger.info("Starting symptom_server MCP Server...")
    # Initialize basic options
    init_options = InitializationOptions(
        server_name="symptom_server",
        server_version="1.0.0",
        capabilities=app.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        )
    )
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Streams connected. Running server...")
        await app.run(
            read_stream,
            write_stream,
            init_options
        )

if __name__ == "__main__":
    # Handle graceful exit on Windows/Linux environments
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down by user interruption.")
    except Exception as e:
        logger.error(f"Server encountered a critical error: {e}")
        sys.exit(1)
