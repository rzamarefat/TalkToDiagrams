from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from LLM import LLM
from SpeechToText import SpeechToText
from TextToSpeech import TextToSpeech
import os

app = Flask(__name__)
CORS(app)

agent = LLM(model="gpt-4o-mini")
speech2txt = SpeechToText()

def generate_response(question, images):
    """Generator function for streaming response."""
    for chunk in agent.chat_stream(question, images):
        yield chunk


def convert_speech2text():
    res = speech2txt.transcribe(r".\assets\my_sample_en_de.m4a")
    print(res)



@app.route("/", methods=["GET"])
def home():
    return "Flask LLM App is running!"


@app.route("/analyze", methods=["POST"])
def analyze():
    # Handle both FormData (with audio) and plain JSON
    if request.content_type and "multipart/form-data" in request.content_type:
        question = request.form.get("question", "You can see a plot in the provided image. Please explain the curves.")
        images = request.form.getlist("images")
        audio_file = request.files.get("audio")
        print("here")
        if audio_file:
            suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
            transcription = speech2txt.transcribe_bytes(audio_file.read(), suffix=suffix)
            print(transcription)
            question = f"{transcription}\n{question}".strip() if question else transcription
    else:
        print("hereeeeee")
        data = request.json
        question = data.get("question", "You can see a plot in the provided image. Please explain the curves.")
        images = data.get("images", ["./assets/normal.png"])

    return Response(generate_response(question, images), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)