"""
Phrase builder — accumulates individual sign detections into sentences.

Supports special gestures for punctuation, clear, and send actions,
plus auto-send after a configurable silence timeout.
"""

import time


class PhraseBuilder:
    """Stateful accumulator that turns individual signs into coherent phrases."""

    SPECIAL_GESTURES = {
        "PERIOD": ".",
        "QUESTION": "?",
        "CLEAR": "__CLEAR__",
        "SEND": "__SEND__",
    }

    def __init__(self, word_timeout: float = 3.0):
        self.words: list[str] = []
        self.last_word_time: float = 0
        self.word_timeout = word_timeout

    # ── public API ───────────────────────────────

    def add_sign(self, sign_text: str) -> dict:
        """
        Process a newly detected sign.

        Returns:
            dict with ``action`` (``'accumulate'``, ``'send'``, or ``'clear'``)
            and optionally ``phrase``.
        """
        now = time.time()
        upper = sign_text.upper()

        # Handle special gestures
        if upper in self.SPECIAL_GESTURES:
            return self._handle_special(self.SPECIAL_GESTURES[upper])

        # Auto-send on silence timeout
        if self.words and (now - self.last_word_time) > self.word_timeout:
            old_phrase = " ".join(self.words)
            self.words = [sign_text]
            self.last_word_time = now
            return {"action": "send", "phrase": old_phrase}

        # Normal accumulation
        self.words.append(sign_text)
        self.last_word_time = now
        return {"action": "accumulate", "phrase": " ".join(self.words)}

    def reset(self):
        """Clear accumulated words."""
        self.words.clear()
        self.last_word_time = 0

    # ── internals ────────────────────────────────

    def _handle_special(self, action: str) -> dict:
        if action == "__CLEAR__":
            self.words.clear()
            return {"action": "clear"}
        if action == "__SEND__":
            phrase = " ".join(self.words)
            self.words.clear()
            return {"action": "send", "phrase": phrase}
        # Punctuation
        if self.words:
            self.words[-1] += action
        return {"action": "accumulate", "phrase": " ".join(self.words)}
