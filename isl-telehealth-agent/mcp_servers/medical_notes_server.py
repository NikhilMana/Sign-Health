"""
Medical Notes Server: MCP server for retrieving and updating patient medical notes.
"""
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

# MCP SDK Standard Imports
from mcp.server import Server, NotificationOptions
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

# =====================================================================
# Server & Logging Setup
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("medical_notes_server")

app = Server("medical_notes_server")

# =====================================================================
# In-memory Store
# =====================================================================
session_notes = {}

# =====================================================================
# Pydantic Schemas for Tools
# =====================================================================

class CreateSessionNoteInput(BaseModel):
    session_id: str = Field(..., description="Unique ID for the current telehealth session")
    patient_signs: List[str] = Field(..., description="List of sign language tokens detected during the session")
    interpreted_intent: str = Field(..., description="The clinical intent interpreted from the signs")
    doctor_response: str = Field(..., description="The response or advice given by the doctor")

class GetSessionSummaryInput(BaseModel):
    session_id: str = Field(..., description="Unique ID for the telehealth session to summarize")

# =====================================================================
# Tool Routing Setup
# =====================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="create_session_note",
            description="Create and store a note for the current telehealth session.",
            inputSchema=CreateSessionNoteInput.model_json_schema(),
        ),
        Tool(
            name="get_session_summary",
            description="Retrieve the summary and history of a specific telehealth session.",
            inputSchema=GetSessionSummaryInput.model_json_schema(),
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Tool call: {name} with args: {arguments}")
    
    if name == "create_session_note":
        try:
            args = CreateSessionNoteInput(**arguments)
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        
        timestamp = datetime.now().isoformat()
        note_id = f"note_{int(datetime.now().timestamp())}"
        
        # Simple summary logic
        summary = f"Patient presented with signs related to '{args.interpreted_intent}'. Doctor advised: {args.doctor_response[:50]}..."
        
        entry = {
            "note_id": note_id,
            "timestamp": timestamp,
            "patient_signs": args.patient_signs,
            "interpreted_intent": args.interpreted_intent,
            "doctor_response": args.doctor_response,
            "summary": summary
        }
        
        if args.session_id not in session_notes:
            session_notes[args.session_id] = []
        
        session_notes[args.session_id].append(entry)
        
        return [TextContent(type="text", text=json.dumps({"note_id": note_id, "timestamp": timestamp, "summary": summary}, indent=2))]

    elif name == "get_session_summary":
        try:
            args = GetSessionSummaryInput(**arguments)
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        
        history = session_notes.get(args.session_id, [])
        total_signs = sum(len(h["patient_signs"]) for h in history)
        
        # Simulate duration
        duration = float(len(history) * 5.5) # Fake multiplier
        
        result = {
            "session_id": args.session_id,
            "history": history,
            "total_signs_detected": total_signs,
            "session_duration_minutes": round(duration, 2)
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        raise ValueError(f"Tool '{name}' not found.")

# =====================================================================
# Direct Python-Callable Functions (Dev / Hackathon Mode)
# =====================================================================

def create_session_note_direct(
    session_id: str,
    doctor_response: str,
    patient_signs: Optional[List[str]] = None,
    interpreted_intent: str = "UNKNOWN"
) -> dict:
    """
    Directly callable version of the create_session_note MCP tool.
    Persists a turn note to the in-memory session_notes store.

    Args:
        session_id (str): Active session identifier.
        doctor_response (str): Latest doctor system response text.
        patient_signs (Optional[List[str]]): Signs detected this turn.
        interpreted_intent (str): The classified intent for this turn.

    Returns:
        dict: note_id, timestamp, summary.
    """
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    note_id = f"note_{int(datetime.now().timestamp())}"
    signs = patient_signs or []

    summary = (
        f"Patient signed: {', '.join(signs) if signs else 'N/A'}. "
        f"Intent: {interpreted_intent}. "
        f"Response: {doctor_response[:60]}..."
    )

    entry = {
        "note_id": note_id,
        "timestamp": timestamp,
        "patient_signs": signs,
        "interpreted_intent": interpreted_intent,
        "doctor_response": doctor_response,
        "summary": summary
    }

    if session_id not in session_notes:
        session_notes[session_id] = []
    session_notes[session_id].append(entry)

    logger.info(f"[Direct] Session note saved: {note_id} for session {session_id}")
    return {"note_id": note_id, "timestamp": timestamp, "summary": summary}


# =====================================================================
# Main Server Entrypoint
# =====================================================================
async def main():
    init_options = InitializationOptions(
        server_name="medical_notes_server",
        server_version="1.0.0",
        capabilities=app.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        )
    )
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, init_options)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
