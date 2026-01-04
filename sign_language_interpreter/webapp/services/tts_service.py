from gtts import gTTS
import io
import base64

class TTSService:
    @staticmethod
    def text_to_speech(text, lang='en'):
        """Convert text to speech and return base64 encoded audio"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_base64 = base64.b64encode(audio_fp.read()).decode('utf-8')
            return audio_base64
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
