"""
ISL Telehealth Master Server: MCP server that wraps the entire sign language telehealth agent pipeline.
This allows other external AI Assistants (like Claude, Cursor, etc.) to use the Telehealth agent as a tool.
"""
import sys
import os
import asyncio
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

# Add parent directory to sys.path so we can import the telehealth core
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
# Force CWD so relative config files (like agents_config.yaml) resolve correctly
os.chdir(PROJECT_DIR)

# Initialize orchestrator globally but lazily
_orchestrator = None

# MCP SDK Standard Imports
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# =====================================================================
# Server & Logging Setup
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("isl_telehealth_mcp")

app = Server("isl_master_server")

# =====================================================================
# Pydantic Schema for the Tool
# =====================================================================
class SimulateConsultationArgs(BaseModel):
    patient_id: str = Field(
        default="synth_patient_001",
        description="ID of the patient."
    )
    sign_language_tokens: List[str] = Field(
        ...,
        description="A list of consecutive sign language words recognized from the camera. E.g., ['chest', 'pain', 'three_days']"
    )

# =====================================================================
# Tool Definitions
# =====================================================================
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for this server."""
    return [
        Tool(
            name="simulate_patient_consultation",
            description="Runs a patient's recognized sign language tokens through the ISL Telehealth multi-agent pipeline. It performs medical context extraction, intent classification, symptom verification via MCP databases, and generates a clinical response.",
            inputSchema=SimulateConsultationArgs.model_json_schema(),
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute the requested tool."""
    if name != "simulate_patient_consultation":
        raise ValueError(f"Unknown tool: {name}")

    patient_id = arguments.get("patient_id", "synth_patient_001")
    tokens = arguments.get("sign_language_tokens", [])

    if not tokens:
        return [TextContent(type="text", text="Error: No sign language tokens provided.")]

    logger.info(f"Running consultation for {patient_id} with signs: {tokens}")

    import io
    from contextlib import redirect_stdout
    
    global _orchestrator
    
    if _orchestrator is None:
        logger.info("First tool invocation: Loading TensorFlow and Telehealth pipeline dynamically...")
        from main import TelehealthOrchestrator
        _orchestrator = TelehealthOrchestrator()
            
    final_response = {}
    intent_result = {}
    action = {"action": "none"}
    complaint = {}
    session_id = "unknown"
    
    # Hide all rogue print() statements to protect the MCP JSON-RPC protocol
    # and to prevent Windows cp1252 terminal Unicode crashes!
    with redirect_stdout(io.StringIO()):        
        session_id = _orchestrator.start_session(patient_id)
        
        # 1. Process all signs through the Medical Context Agent
        for sign in tokens:
            # Simulate high confidence prediction from the camera Perception tracking
            _orchestrator.registry.context.process_sign(sign, confidence=0.95)
    
        # 2. Extract structured medical complaint
        complaint = _orchestrator.registry.context.extract_complaint()
        
        # 3. Classify the Intent (e.g., SYMPTOM_REPORT vs EMERGENCY_ALERT)
        intent_result = _orchestrator.registry.intent.classify_intent(complaint)
        
        # 4. Route intent to Dialogue Agent for action logic
        action = _orchestrator.registry.dialogue.manage_turn(intent_result, session_id)
    
        # 5. Execute required Tools (via ToolAgent querying other internal MCPs) and format the final response
        if action["action"] == "execute_tool":
            tool_result = await _orchestrator.registry.tool.execute(action)
            final_response = _orchestrator.registry.response.format_response(tool_result, intent_result["intent"])
        elif action["action"] == "ask_clarification":
            final_response = {
                "text_response": action["question"],
                "urgency_level": "none"
            }
        elif action["action"] == "emergency_escalation":
            final_response = _orchestrator.registry.response.format_response(
                {"urgency_level": "emergency", "immediate_action": action["message"]},
                "EMERGENCY_ALERT"
            )
        else:
            final_response["text_response"] = "I am unable to assist with this right now."
    
        # End the session cleanly
        _orchestrator.end_session(session_id)

    # Compile the final MCP Tool Response formatted neatly for the external AI
    response_lines = [
        f"### ISL Telehealth Consultation Result (Session: {session_id})",
        f"**Input Signs:** `{', '.join(tokens)}`",
        f"**Extracted Context:** {complaint}",
        f"**Classified Intent:** {intent_result.get('intent')} (Confidence: {intent_result.get('confidence', 0):.2f})",
        f"**Pipeline Action Taken:** {action['action']}",
        "---",
        f"### Final Doctor Response",
        f"> {final_response.get('text_response')}",
        "",
        f"**Urgency / Status:** {final_response.get('urgency_color', 'blue').upper()}"
    ]

    return [TextContent(type="text", text="\n".join(response_lines))]

# =====================================================================
# Main Entry Point
# =====================================================================
async def main():
    logger.info("Starting ISL Telehealth Master MCP Server on stdio...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Local bypass mode for Cursor AI testing & Terminal Testing
        patient = "synth_patient_001"
        test_tokens = sys.argv[1:]
        print(f"--- LOCAL CLI MODE BYPASS ---")
        print(f"Running Telehealth Pipeline for patient {patient} with signs: {test_tokens}")
        print("Loading AI Models... Please wait...\n")
        
        # Instantiate and run just like the tool would
        from main import TelehealthOrchestrator
        orchestrator = TelehealthOrchestrator()
        session_id = orchestrator.start_session(patient)
        
        for sign in test_tokens:
            orchestrator.registry.context.process_sign(sign, confidence=0.95)
            
        complaint = orchestrator.registry.context.extract_complaint()
        intent_result = orchestrator.registry.intent.classify_intent(complaint)
        action = orchestrator.registry.dialogue.manage_turn(intent_result, session_id)
        
        if action["action"] == "execute_tool":
            tool_res = asyncio.run(orchestrator.registry.tool.execute(action))
            final_res = orchestrator.registry.response.format_response(tool_res, intent_result["intent"])
        elif action["action"] == "emergency_escalation":
            final_res = orchestrator.registry.response.format_response(
                {"urgency_level": "emergency", "immediate_action": action["message"]}, "EMERGENCY_ALERT"
            )
        else:
            final_res = {"text_response": "Uncertain logic.", "urgency_color": "yellow"}
            
        orchestrator.end_session(session_id)
        
        print("\n" + "="*50)
        print("FINAL CLINICAL OUTCOME:")
        print("="*50)
        print(f"Extracted Context: {complaint}")
        print(f"Action Taken:      {action['action']}")
        print(f"Urgency Color:     {final_res.get('urgency_color', 'UNKNOWN').upper()}")
        print(f"Doctor Response:   {final_res.get('text_response')}")
        print("="*50)
    else:
        # Standard MCP Server Mode
        asyncio.run(main())
