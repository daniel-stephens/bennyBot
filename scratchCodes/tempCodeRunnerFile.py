from kokoro import KPipeline
from playsound import playsound
import soundfile as sf
import tempfile
import sounddevice as sd
import numpy as np
import torch
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import whisper

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("Please provide an OpenAI API key in the .env file")

# Initialize models
chat_model = ChatOpenAI(api_key=openai_key)
whisper_model = whisper.load_model("small").to("cuda" if torch.cuda.is_available() else "cpu")
pipeline = KPipeline(lang_code='a')

# Audio settings
samplerate = 16000
channels = 1
silence_duration = 2  # seconds
threshold = 0.01

# Silence detection function
def is_silent(data, threshold):
    return max(abs(data)) < threshold

# Main conversational loop
while True:
    print("Listening for your question...")

    audio_chunks = []

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype="float32") as stream:
        silent_chunks = 0
        while True:
            chunk, _ = stream.read(int(samplerate * 0.5))
            audio_chunk = chunk.flatten()

            if is_silent(audio_chunk, threshold):
                silence_duration -= 0.5
                if silence_duration <= 0:
                    break
            else:
                silence_duration = 2

            full_audio = audio_chunk if 'full_audio' not in locals() else np.concatenate([full_audio, audio_chunk])

        audio_data = audio_chunk.astype('float32')

        # Whisper transcription
        segments, _ = whisper_model.transcribe(audio_data)
        question = " ".join(segment.text for segment in segments)

        print(f"Question: {question}")

        # Query ChatGPT
        llm = ChatOpenAI(api_key=openai_key)
        response = llm.invoke(question)
        answer_text = response.content

        print(f"Answer: {answer_text}")

        # Kokoro TTS
        pipeline = KPipeline(lang_code='a')
        audio_gen = pipeline(answer_text, voice='af_heart', speed=1)
        audio_data = []
        for _, _, audio in audio_gen:
            full_audio.extend(audio)

        # Playback audio immediately
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, full_audio, 24000)
            playsound(tmp.name)

        # Reset silence duration for next round
        silence_duration = 2
