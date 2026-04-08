"""
Agent Controller — orchestrates post-processing of gesture predictions.

Flow:
    1. Receive prediction result from the ISL model  (``{"text": ..., "confidence": ...}``)
    2. Map the raw label to a structured intent via ``IntentMapper``
    3. Select and invoke the appropriate MCP tool based on the intent
    4. Log the full interaction via ``log_interaction()``
    5. Return a unified result dict to the caller

Usage:
    from services.agent_controller import process_prediction

    result = process_prediction({"text": "help", "confidence": 0.87})
"""

from __future__ import annotations

import logging

from services.intent_mapper import IntentMapper
from services.mcp_tools import (
    navigation_tool,
    medical_alert_tool,
    assistance_alert_tool,
    information_tool,
    log_interaction,
)

logger = logging.getLogger(__name__)

# ── Shared mapper instance ───────────────────────────────

_mapper = IntentMapper()


# ── Tool dispatch ────────────────────────────────────────

_GREETING_RESPONSE = "Hello! How can I help you today?"


def _dispatch_tool(intent: str, sign: str) -> tuple[str, str]:
    """
    Select and invoke the right MCP tool for a given intent.

    Args:
        intent:  structured intent string (e.g. ``"navigation"``).
        sign:    original sign label (passed as argument to some tools).

    Returns:
        Tuple of (tool_name, action_result).
    """
    if intent == "navigation":
        return "navigation_tool", navigation_tool(sign)

    if intent == "medical_help":
        return "medical_alert_tool", medical_alert_tool()

    if intent == "assistance_request":
        return "assistance_alert_tool", assistance_alert_tool()

    if intent == "basic_need":
        return "information_tool", information_tool(sign)

    if intent in ("greeting", "gratitude"):
        return "none", _GREETING_RESPONSE

    # general / unmapped → try information lookup
    return "information_tool", information_tool(sign)


# ── Public API ───────────────────────────────────────────

def process_prediction(prediction: dict) -> dict:
    """
    End-to-end post-processing of a single gesture prediction.

    Args:
        prediction: dict with at least ``"text"`` (label) and
                    ``"confidence"`` (float) keys.

    Returns:
        Structured result dict::

            {
                "sign":          str,
                "intent":        str,
                "category":      str,
                "priority":      str,
                "tool_used":     str,
                "action_result": str,
                "confidence":    float,
            }
    """
    sign = prediction.get("text", "").strip().lower()
    confidence = prediction.get("confidence", 0.0)

    # Step 1 — map label → intent
    intent_info = _mapper.map(sign)

    # Step 2 — dispatch to the appropriate tool
    tool_used, action_result = _dispatch_tool(intent_info["intent"], sign)

    # Step 3 — build the unified result
    result = {
        "sign":          sign,
        "intent":        intent_info["intent"],
        "category":      intent_info["category"],
        "priority":      intent_info["priority"],
        "tool_used":     tool_used,
        "action_result": action_result,
        "confidence":    confidence,
    }

    # Step 4 — log the interaction
    log_interaction(result)

    logger.info(
        "Processed '%s' → intent=%s  tool=%s  priority=%s",
        sign, intent_info["intent"], tool_used, intent_info["priority"],
    )

    return result
