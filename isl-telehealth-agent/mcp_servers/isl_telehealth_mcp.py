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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TelehealthOrchestrator

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

    # Instantiate the mastermind orchestrator
    orchestrator = TelehealthOrchestrator()
    session_id = orchestrator.start_session(patient_id)
    
    # 1. Process all signs through the Medical Context Agent
    for sign in tokens:
        # Simulate high confidence prediction from the camera Perception tracking
        orchestrator.registry.context.process_sign(sign, confidence=0.95)

    # 2. Extract structured medical complaint
    complaint = orchestrator.registry.context.extract_complaint()
    
    # 3. Classify the Intent (e.g., SYMPTOM_REPORT vs EMERGENCY_ALERT)
    intent_result = orchestrator.registry.intent.classify_intent(complaint)
    
    # 4. Route intent to Dialogue Agent for action logic
    action = orchestrator.registry.dialogue.manage_turn(intent_result, session_id)

    final_response = {}

    # 5. Execute required Tools (via ToolAgent querying other internal MCPs) and format the final response
    if action["action"] == "execute_tool":
        tool_result = await orchestrator.registry.tool.execute(action)
        final_response = orchestrator.registry.response.format_response(tool_result, intent_result["intent"])
    elif action["action"] == "ask_clarification":
        final_response = {
            "text_response": action["question"],
            "urgency_level": "none"
        }
    elif action["action"] == "emergency_escalation":
        final_response = orchestrator.registry.response.format_response(
            {"urgency_level": "emergency", "immediate_action": action["message"]},
            "EMERGENCY_ALERT"
        )
    else:
        final_response["text_response"] = "I am unable to assist with this right now."

    # End the session cleanly
    orchestrator.end_session(session_id)

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
    asyncio.run(main())
