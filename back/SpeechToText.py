import whisper
import tempfile
import os

class SpeechToText:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, file_path):
        result = self.model.transcribe(file_path)
        return result["text"]
    
    def transcribe_bytes(self, audio_bytes, suffix=".webm"):  # ← ADD THIS
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self.transcribe(tmp_path)
        finally:
            os.remove(tmp_path)