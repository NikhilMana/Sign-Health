"""Tests for the PhraseBuilder service."""

import time
import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

from services.phrase_builder import PhraseBuilder


class TestPhraseBuilder:
    """Test phrase accumulation, special gestures, and auto-send."""

    def test_accumulate_single_word(self):
        pb = PhraseBuilder()
        result = pb.add_sign("hello")
        assert result["action"] == "accumulate"
        assert result["phrase"] == "hello"

    def test_accumulate_multiple_words(self):
        pb = PhraseBuilder()
        pb.add_sign("hello")
        result = pb.add_sign("world")
        assert result["action"] == "accumulate"
        assert result["phrase"] == "hello world"

    def test_send_gesture(self):
        pb = PhraseBuilder()
        pb.add_sign("hello")
        pb.add_sign("doctor")
        result = pb.add_sign("SEND")
        assert result["action"] == "send"
        assert result["phrase"] == "hello doctor"

    def test_clear_gesture(self):
        pb = PhraseBuilder()
        pb.add_sign("hello")
        result = pb.add_sign("CLEAR")
        assert result["action"] == "clear"

    def test_clear_resets_phrase(self):
        pb = PhraseBuilder()
        pb.add_sign("hello")
        pb.add_sign("CLEAR")
        result = pb.add_sign("goodbye")
        assert result["phrase"] == "goodbye"

    def test_period_punctuation(self):
        pb = PhraseBuilder()
        pb.add_sign("hello")
        result = pb.add_sign("PERIOD")
        assert result["action"] == "accumulate"
        assert result["phrase"] == "hello."

    def test_question_punctuation(self):
        pb = PhraseBuilder()
        pb.add_sign("how")
        pb.add_sign("are")
        pb.add_sign("you")
        result = pb.add_sign("QUESTION")
        assert result["phrase"] == "how are you?"

    def test_auto_send_on_timeout(self):
        pb = PhraseBuilder(word_timeout=0.1)
        pb.add_sign("hello")
        time.sleep(0.15)
        result = pb.add_sign("new")
        assert result["action"] == "send"
        assert result["phrase"] == "hello"

    def test_reset_clears_state(self):
        pb = PhraseBuilder()
        pb.add_sign("hello")
        pb.reset()
        assert pb.words == []
