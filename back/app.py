from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from LLM import LLM
from SpeechToText import SpeechToText
from TextToSpeech import TextToSpeech
from database import init_db, ensure_conversation, save_message, get_messages, get_conversations, delete_conversation
import os
import time
import json
import uuid

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1000 * 1024 * 1024
CORS(app)

init_db()

agent = LLM(model="gpt-4o-mini")
speech2txt = SpeechToText()


def generate_and_save(question, images, conversation_id):
    """Stream LLM chunks and persist the full exchange to the DB."""
    save_message(conversation_id, "user", question)
    accumulated = []
    for chunk in agent.chat_stream(question, images):
        accumulated.append(chunk)
        yield chunk
    save_message(conversation_id, "assistant", "".join(accumulated))


@app.route("/", methods=["GET"])
def home():
    return "Flask LLM App is running!"


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
        "path": save_path,
        "url": f"http://localhost:5000/uploads/{filename}"
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
    if request.content_type and "multipart/form-data" in request.content_type:
        question = request.form.get("question", "You can see a plot in the provided image. Please explain the curves.")
        images = json.loads(request.form.get("images", "[]"))
        conversation_id = request.form.get("conversation_id", uuid.uuid4().hex)
        label = request.form.get("label", "")

        audio_file = request.files.get("audio")
        if audio_file:
            suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
            transcription = speech2txt.transcribe_bytes(audio_file.read(), suffix=suffix)
            question = f"{transcription}\n{question}".strip() if question else transcription
    else:
        data = request.json
        question = data.get("question", "You can see a plot in the provided image. Please explain the curves.")
        images = data.get("images", ["./assets/normal.png"])
        conversation_id = data.get("conversation_id", uuid.uuid4().hex)
        label = data.get("label", "")

    ensure_conversation(conversation_id, label)
    return Response(generate_and_save(question, images, conversation_id), mimetype="text/plain")


@app.route("/conversations", methods=["GET"])
def list_conversations():
    return jsonify(get_conversations())


@app.route("/conversations/<conversation_id>/messages", methods=["GET"])
def list_messages(conversation_id):
    return jsonify(get_messages(conversation_id))


@app.route("/conversations/<conversation_id>", methods=["DELETE"])
def delete_conv(conversation_id):
    delete_conversation(conversation_id)
    return "", 204


if __name__ == "__main__":
    app.run(debug=True)
