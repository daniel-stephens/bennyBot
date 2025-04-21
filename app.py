from flask import Flask, request, send_file, render_template, jsonify
from flask_cors import CORS
import openai
import io
from .util import *
from openai import OpenAI
from pydub import AudioSegment
import base64
import time


client = OpenAI()

app = Flask(__name__)
load_dotenv()
openai.api_key  = os.getenv("OPENAI_API_KEY")
CORS(app)

memory = ConversationBufferMemory(return_messages=True)
last_interaction_time = time.time()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    
    global last_interaction_time, memory
    now = time.time()

    # Reset memory if silent for more than 60 seconds
    if now - last_interaction_time > 60:
        memory.clear()
    last_interaction_time = now

    # Read audio and convert
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    webm_io = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(webm_io, format="webm")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav", codec="pcm_s16le", parameters=["-ar", "16000", "-ac", "1"])
    wav_io.seek(0)
    wav_io.name = "audio.wav"

    # Transcribe
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_io,
        response_format="text"
    )
    user_text = transcription.strip()

    # Generate assistant response (uses memory)
    response_text = generate_response(user_text, memory)

    # Generate TTS
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice="ash",
        input=response_text
    )

    audio_base64 = base64.b64encode(tts_response.content).decode("utf-8")

    return jsonify({
        "transcription": user_text,
        "response_text": response_text,
        "audio_base64": audio_base64
    })



@app.route("/")
def index():
    return render_template("voices.html")

if __name__ == "__main__":
    app.run(debug=True)
