"""
Appointment Server: MCP server for managing medical appointments and schedules.
"""
import sys
import json
import asyncio
import logging
import uuid
import random
from datetime import datetime, timedelta
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
logger = logging.getLogger("appointment_server")

app = Server("appointment_server")

# =====================================================================
# Hardcoded Data
# =====================================================================
DOCTORS = [
    "Dr. Aditi Sharma",
    "Dr. Rajesh Khanna",
    "Dr. Meera Patel",
    "Dr. Vikram Singh",
    "Dr. Sunita Reddy"
]

# In-memory store for appointments
appointments_db = {}

# =====================================================================
# Pydantic Schemas for Tools
# =====================================================================

class BookAppointmentInput(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient")
    department: str = Field(..., description="Medical department (e.g., Cardiology, Dermatology, General)")
    preferred_date: str = Field(..., description="Preferred date for the appointment (YYYY-MM-DD)")
    urgency: str = Field(..., description="Urgency level (Low, Medium, High)")

class GetAvailableSlotsInput(BaseModel):
    department: str = Field(..., description="Medical department")
    date: str = Field(..., description="Date to check availability (YYYY-MM-DD)")

# =====================================================================
# Tool Routing Setup
# =====================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="book_appointment",
            description="Book a new medical appointment and return confirmation details.",
            inputSchema=BookAppointmentInput.model_json_schema(),
        ),
        Tool(
            name="get_available_slots",
            description="Get available time slots for a specific department and date.",
            inputSchema=GetAvailableSlotsInput.model_json_schema(),
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Tool call: {name} with args: {arguments}")
    
    if name == "book_appointment":
        try:
            args = BookAppointmentInput(**arguments)
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        
        confirmation_id = str(uuid.uuid4())
        doctor_name = random.choice(DOCTORS)
        
        # Fake a realistic time
        fake_time = ["09:30 AM", "11:00 AM", "02:15 PM", "04:45 PM"]
        scheduled_time = random.choice(fake_time)
        scheduled_datetime = f"{args.preferred_date} {scheduled_time}"
        
        appointment_data = {
            "confirmation_id": confirmation_id,
            "patient_name": args.patient_name,
            "department": args.department,
            "doctor_name": doctor_name,
            "scheduled_datetime": scheduled_datetime,
            "instructions": "Arrive 15 minutes early, bring a valid government ID and any previous medical records."
        }
        
        appointments_db[confirmation_id] = appointment_data
        
        return [TextContent(type="text", text=json.dumps(appointment_data, indent=2))]

    elif name == "get_available_slots":
        try:
            args = GetAvailableSlotsInput(**arguments)
        except ValidationError as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
        
        # Static realistic slots
        slots = [
            "09:00 AM - 09:30 AM",
            "10:30 AM - 11:00 AM",
            "02:00 PM - 02:30 PM",
            "04:00 PM - 04:30 PM"
        ]
        
        result = {
            "department": args.department,
            "date": args.date,
            "available_slots": slots
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        raise ValueError(f"Tool '{name}' not found.")

# =====================================================================
# Direct Python-Callable Functions (Dev / Hackathon Mode)
# =====================================================================

def book_appointment_direct(
    patient_name: str,
    department: str,
    preferred_date: str,
    urgency: str
) -> dict:
    """
    Directly callable version of the book_appointment MCP tool.

    Args:
        patient_name (str): Name of the patient.
        department (str): Medical department.
        preferred_date (str): Preferred date string (e.g. 'tomorrow', '2025-04-10').
        urgency (str): Urgency level string.

    Returns:
        dict: confirmation_id, doctor_name, scheduled_datetime, instructions.
    """
    import uuid
    from datetime import datetime, timedelta

    confirmation_id = str(uuid.uuid4())[:8].upper()
    doctor_name = random.choice(DOCTORS)

    # Build a realistic datetime
    fake_times = ["09:30 AM", "11:00 AM", "02:15 PM", "04:45 PM"]
    scheduled_time = random.choice(fake_times)

    try:
        base_date = datetime.strptime(preferred_date, "%Y-%m-%d")
    except ValueError:
        base_date = datetime.now() + timedelta(days=1)

    scheduled_datetime = f"{base_date.strftime('%d %b %Y')} at {scheduled_time}"

    data = {
        "confirmation_id": confirmation_id,
        "patient_name": patient_name,
        "department": department,
        "doctor_name": doctor_name,
        "scheduled_datetime": scheduled_datetime,
        "instructions": "Arrive 15 minutes early. Bring a valid government ID and any previous medical records."
    }

    appointments_db[confirmation_id] = data
    logger.info(f"[Direct] Appointment booked: {confirmation_id} with {doctor_name}")
    return data


# =====================================================================
# Main Server Entrypoint
# =====================================================================
async def main():
    init_options = InitializationOptions(
        server_name="appointment_server",
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
