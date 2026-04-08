"""
Post-processing module for gesture predictions.

Maps raw ISL model labels to structured intent objects for downstream
consumption (e.g. nurse dashboard, routing, priority queuing).

Usage:
    from services.intent_mapper import IntentMapper

    mapper = IntentMapper()
    result = mapper.map("help")
    # → {"original_sign": "help", "intent": "assistance_request",
    #    "category": "support", "priority": "high"}
"""

from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


# ── Type hint for the structured intent ──────────────────

class Intent(TypedDict):
    original_sign: str
    intent: str
    category: str
    priority: str


# ── Default mapping table ────────────────────────────────
# Add new entries here to extend the mapper — no code changes needed.

_DEFAULT_MAPPINGS: dict[str, dict] = {
    # ─── High-priority: health & safety ───
    "help":      {"intent": "assistance_request", "category": "support",  "priority": "high"},
    "doctor":    {"intent": "medical_help",       "category": "health",   "priority": "high"},
    "pain":      {"intent": "medical_help",       "category": "health",   "priority": "high"},
    "emergency": {"intent": "emergency_alert",    "category": "health",   "priority": "high"},
    "medicine":  {"intent": "medical_help",       "category": "health",   "priority": "high"},

    # ─── Medium-priority: basic needs & navigation ───
    "washroom":  {"intent": "navigation",         "category": "facility", "priority": "medium"},
    "water":     {"intent": "basic_need",         "category": "facility", "priority": "medium"},
    "food":      {"intent": "basic_need",         "category": "facility", "priority": "medium"},
    "bed":       {"intent": "basic_need",         "category": "facility", "priority": "medium"},

    # ─── Low-priority: social & informational ───
    "hello":     {"intent": "greeting",           "category": "social",   "priority": "low"},
    "thank you": {"intent": "gratitude",          "category": "social",   "priority": "low"},
    "yes":       {"intent": "affirmation",        "category": "response", "priority": "low"},
    "no":        {"intent": "negation",           "category": "response", "priority": "low"},
}

_FALLBACK: dict = {
    "intent":   "general",
    "category": "general",
    "priority": "low",
}


class IntentMapper:
    """
    Stateless mapper: predicted sign label → structured intent JSON.

    The mapping dictionary can be extended at construction time or at
    runtime via :meth:`add_mapping` without touching ML code.
    """

    def __init__(self, extra_mappings: dict[str, dict] | None = None):
        self._mappings: dict[str, dict] = {**_DEFAULT_MAPPINGS}
        if extra_mappings:
            self._mappings.update(extra_mappings)

    # ── Public API ───────────────────────────────────────

    def map(self, label: str) -> Intent:
        """
        Convert a predicted gesture label into a structured intent.

        Args:
            label: raw lowercase label string from the ISL model.

        Returns:
            Intent dict with keys ``original_sign``, ``intent``,
            ``category``, and ``priority``.
        """
        normalised = label.strip().lower()
        entry = self._mappings.get(normalised, _FALLBACK)

        result: Intent = {
            "original_sign": normalised,
            "intent":        entry["intent"],
            "category":      entry["category"],
            "priority":      entry["priority"],
        }

        logger.debug("Mapped '%s' → %s", normalised, result)
        return result

    def add_mapping(
        self,
        sign: str,
        intent: str,
        category: str,
        priority: str = "low",
    ) -> None:
        """Register an additional sign → intent mapping at runtime."""
        self._mappings[sign.strip().lower()] = {
            "intent":   intent,
            "category": category,
            "priority": priority,
        }

    def has_mapping(self, label: str) -> bool:
        """Check whether a specific label has an explicit mapping."""
        return label.strip().lower() in self._mappings

    @property
    def known_signs(self) -> list[str]:
        """Return all explicitly mapped sign labels."""
        return sorted(self._mappings.keys())
