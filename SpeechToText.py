import whisper

class SpeechToText:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, file_path):
        result = self.model.transcribe(file_path)
        return result["text"]



if __name__ == "__main__":
    s = SpeechToText()
    res = s.transcribe(r".\assets\my_sample_en_de.m4a")
    print(res)