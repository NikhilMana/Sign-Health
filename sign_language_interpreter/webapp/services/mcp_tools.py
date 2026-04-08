"""
MCP-style tools for post-intent action execution.

Each function simulates an external service that would be triggered
after a gesture is classified and mapped to an intent.  All tools
run locally with no external APIs or database dependencies.

Usage:
    from services.mcp_tools import navigation_tool, medical_alert_tool

    print(navigation_tool("washroom"))
    # → "Washroom is on 2nd floor near lift"
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 1. Navigation Tool
# ══════════════════════════════════════════════════════════

_LOCATION_DIRECTORY: dict[str, str] = {
    "washroom":  "Washroom is on 2nd floor near lift",
    "canteen":   "Canteen is on the ground floor, east wing",
    "reception": "Reception is at the main entrance on the ground floor",
    "pharmacy":  "Pharmacy is on the 1st floor next to the billing counter",
    "exit":      "Main exit is straight ahead past reception",
    "lab":       "Pathology lab is on the 3rd floor, room 302",
    "icu":       "ICU is on the 4th floor, restricted access",
}


def navigation_tool(location: str) -> str:
    """
    Return directions for a known hospital location.

    Args:
        location: name of the destination (case-insensitive).

    Returns:
        Directions string, or a helpful fallback if the location
        is not in the directory.
    """
    key = location.strip().lower()
    directions = _LOCATION_DIRECTORY.get(key)

    if directions:
        logger.info("Navigation request: %s", key)
        return directions

    logger.warning("Unknown location requested: %s", key)
    return f"Location '{location}' not found. Please ask staff for directions."


# ══════════════════════════════════════════════════════════
# 2. Medical Alert Tool
# ══════════════════════════════════════════════════════════

def medical_alert_tool() -> str:
    """
    Simulate notifying the on-duty medical staff.

    Returns:
        Confirmation message.
    """
    logger.info("MEDICAL ALERT triggered at %s", datetime.now().isoformat())
    return "Medical staff notified"


# ══════════════════════════════════════════════════════════
# 3. Assistance Alert Tool
# ══════════════════════════════════════════════════════════

def assistance_alert_tool() -> str:
    """
    Simulate notifying the support / nursing staff.

    Returns:
        Confirmation message.
    """
    logger.info("ASSISTANCE ALERT triggered at %s", datetime.now().isoformat())
    return "Support staff notified"


# ══════════════════════════════════════════════════════════
# 4. Information Tool
# ══════════════════════════════════════════════════════════

_INFO_RESPONSES: dict[str, str] = {
    "water":    "Water will be provided shortly",
    "food":     "Food service available during lunch hours",
    "bed":      "Bed assistance request recorded",
    "medicine": "Medicine schedule will be shared by the nurse",
    "visiting": "Visiting hours are 10 AM to 1 PM and 4 PM to 7 PM",
}


def information_tool(topic: str) -> str:
    """
    Return informational text about a given topic.

    Args:
        topic: subject the patient is asking about (case-insensitive).

    Returns:
        Informational response string.
    """
    key = topic.strip().lower()
    response = _INFO_RESPONSES.get(key)

    if response:
        logger.info("Info request: %s", key)
        return response

    logger.warning("Unknown info topic: %s", key)
    return f"No information available for '{topic}'. A staff member will assist you."


# ══════════════════════════════════════════════════════════
# 5. Interaction Logger
# ══════════════════════════════════════════════════════════

_interaction_log: list[dict] = []


def log_interaction(data: dict) -> str:
    """
    Store an interaction record in an in-memory list.

    Args:
        data: arbitrary dict describing the interaction
              (e.g. sign, intent, timestamp, patient_id).

    Returns:
        Confirmation string.
    """
    entry = {
        **data,
        "logged_at": datetime.now().isoformat(),
    }
    _interaction_log.append(entry)
    logger.debug("Interaction logged: %s", entry)
    return "log stored"


def get_interaction_log() -> list[dict]:
    """Return a copy of all logged interactions."""
    return list(_interaction_log)


def clear_interaction_log() -> None:
    """Clear the in-memory interaction log."""
    _interaction_log.clear()
