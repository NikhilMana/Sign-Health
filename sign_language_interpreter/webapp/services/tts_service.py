"""
Text-to-Speech service with in-memory caching.

Uses Google TTS (gTTS) but caches results so repeated signs
(common in sign language) don't incur network round-trips.
"""

import io
import base64
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)


class TTSService:
    """Stateless TTS helper with a class-level audio cache."""

    _cache: dict[str, str] = {}

    @staticmethod
    def text_to_speech(text, lang="en"):
        """
        Convert *text* to speech and return base64-encoded MP3 audio.

        Results are cached — subsequent calls with the same text/lang
        return instantly without hitting the network.
        """
        cache_key = f"{text}:{lang}"

        if cache_key in TTSService._cache:
            return TTSService._cache[cache_key]

        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_base64 = base64.b64encode(audio_fp.read()).decode("utf-8")

            TTSService._cache[cache_key] = audio_base64
            return audio_base64

        except Exception as e:
            logger.error("TTS failed for '%s': %s", text, e)
            return None

    @staticmethod
    def warm_cache(sign_labels, lang="en"):
        """Pre-generate audio for all known sign labels at startup."""
        for label in sign_labels:
            TTSService.text_to_speech(label, lang)
        logger.info("TTS cache warmed with %d labels", len(sign_labels))
