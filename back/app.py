from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from LLM import LLM
from SpeechToText import SpeechToText
from TextToSpeech import TextToSpeech
import os
import time
import json

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024
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


import uuid

UPLOAD_FOLDER = "./assets/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask import send_from_directory

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(UPLOAD_FOLDER), filename)

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    ext = os.path.splitext(file.filename)[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    return jsonify({
        "path": save_path,                                    # sent to LLM
        "url": f"http://localhost:5000/uploads/{filename}"   # shown in carousel
    })



@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio provided"}), 400
    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    transcription = speech2txt.transcribe_bytes(audio_file.read(), suffix=suffix)
    return jsonify({"text": transcription})


@app.route("/analyze", methods=["POST"])
def analyze():
    # Handle both FormData (with audio) and plain JSON
    if request.content_type and "multipart/form-data" in request.content_type:
        question = request.form.get("question", "You can see a plot in the provided image. Please explain the curves.")

        images = json.loads(request.form.get("images", "[]"))

        audio_file = request.files.get("audio")
        if audio_file:
            suffix = os.path.splitext(audio_file.filename)[1] or ".webm"

            # audio_bytes = audio_file.read()
            # os.makedirs("recordings", exist_ok=True)
            # save_path = os.path.join("recordings", f"recording_{int(time.time())}{suffix}")
            # with open(save_path, "wb") as f:
            #     f.write(audio_bytes)

            transcription = speech2txt.transcribe_bytes(audio_file.read(), suffix=suffix)
            print(transcription)
            question = f"{transcription}\n{question}".strip() if question else transcription
            # question = "hi"
    else:
        data = request.json
        question = data.get("question", "You can see a plot in the provided image. Please explain the curves.")
        images = data.get("images", ["./assets/normal.png"])

    return Response(generate_response(question, images), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True)