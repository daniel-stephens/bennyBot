import os
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import whisper
from playsound import playsound
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from kokoro import KPipeline
from langchain.memory import ConversationBufferMemory
from flask import Flask, jsonify, Response, stream_with_context
from flask import Flask, jsonify, render_template
import time
import queue
import noisereduce as nr
import numpy as np

app = Flask(__name__)


def reduce_noise(audio_data, samplerate):
    return nr.reduce_noise(y=audio_data, sr=samplerate)

# Load environment variables
def load_api_keys():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Please provide an OpenAI API key.")
    return openai_key

# Initialize models
def initialize_models():
    whisper_model = whisper.load_model("large").to("cuda" if torch.cuda.is_available() else "cpu")
    kokoro_pipeline = KPipeline(lang_code='a')  # American English
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history")
    return whisper_model, kokoro_pipeline, memory

# Check if audio is silent
def is_silent(data, threshold):
    return np.abs(data).mean() < threshold



def record_and_transcribe(whisper_model, samplerate=16000, threshold=0.01, silence_duration=2):
    audio_chunks = []
    current_silence = 0 # Dynamically adjust silence threshold

    print(f"Using silence threshold: {threshold:.5f}")
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
        recording_time = 0

        while True:
            audio_chunk, _ = stream.read(int(samplerate * 1))  # ✅ Increased chunk size to 1 second
            audio_chunks.append(audio_chunk)
            recording_time += 1  # ✅ Track total recorded time more accurately

            if is_silent(audio_chunk, threshold):
                current_silence += 1  # ✅ Increase silence check to 1s intervals
            else:
                current_silence = 0  # Reset silence counter on voice detection

            # ✅ Ensure at least 2 seconds of recording to avoid early cut-off
            if current_silence >= silence_duration:
                print("Silence detected, stopping recording.")
                break

    # Process the recorded audio
    audio_data = np.concatenate(audio_chunks).flatten().astype(np.float32)
    audio_data = reduce_noise(audio_data, samplerate)
    print("Transcribing...")
    result = whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available(), language="en")
    return result["text"]



# # Check if audio is silent
# def is_silent(data, threshold):
#     return np.abs(data).mean() < threshold

# # Record and transcribe audio
# def record_and_transcribe(whisper_model, samplerate=16000, threshold=0.3, silence_duration=2):
#     audio_chunks = []
#     current_silence = 0

#     with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
#         while True:
#             audio_chunk, _ = stream.read(int(samplerate * 0.5))
#             audio_chunks.append(audio_chunk)

#             if is_silent(audio_chunk, threshold):
#                 current_silence += 0.5
#             else:
#                 current_silence = 0

#             if current_silence >= silence_duration:
#                 break

#     audio_data = np.concatenate(audio_chunks).flatten()
#     # Ensure audio is in correct format (float32 numpy array)
#     audio_data = audio_data.astype(np.float32)
#     result = whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available(), language="en")
#     return result["text"]

# Generate speech audio from text
def text_to_speech(pipeline, text, voice='am_onyx', speed=1):
    generator = pipeline(text, voice=voice, speed=speed)
    audio_data = []
    for _, _, audio in generator:
        audio_data.extend(audio)
    return np.array(audio_data, dtype=np.float32)

# Play audio from numpy array
def play_audio(audio_data, samplerate=24000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
        sf.write(tmp_file.name, audio_data, samplerate)
        playsound(tmp_file.name)

# # Generate response using GPT
# def generate_response(question):
#     openai_key = load_api_keys()
#     gpt_model = ChatOpenAI(api_key=openai_key, model="gpt-4o")
#     prompt = f"Give a concise answer to the question asked: {question}"
#     response = gpt_model.invoke(prompt)
#     return response.content

def generate_response(question, memory):
    openai_key = load_api_keys()
    gpt_model = ChatOpenAI(api_key=openai_key, model="gpt-4o")

    # Retrieve past conversation history
    chat_history = memory.load_memory_variables({}).get("chat_history", "")
    
    prompt = f"You are Benny! The mascot of Morgan State University and you provide answers to those who need help on campus:\n\n"
    prompt += f"Answer the following question to the best of your ability \n\n"
    prompt += f"Uses the conversation history if needed"
    prompt += f"Conversation History:\n{chat_history}\n\n"
    prompt += f"Question: {question}"
    prompt += f"Do not use a preamble everytime you answer"

    response = gpt_model.invoke(prompt)
    
    # Save the new interaction
    memory.save_context({"input": question}, {"output": response.content})

    return response.content

# Continuous transcription and interaction loop
# def continuous_interaction():
#     whisper_model, kokoro_pipeline, memory = initialize_models()

#     print("Starting continuous transcription...")

#     while True:
#         print("Listening...")
#         transcription = record_and_transcribe(whisper_model)
#         print("Transcription:", transcription)

#         response = generate_response(transcription, memory)
#         audio_response = text_to_speech(kokoro_pipeline, response)
#         play_audio(audio_data=audio_response)



# Function to simulate continuous interaction
def continuous_interaction():
    whisper_model, kokoro_pipeline, memory = initialize_models()
    print("Starting continuous transcription...")

    def event_stream():
        silence_timer = 0  
        last_response_time = time.time()

        while True:
            if silence_timer >= 20:  # Stop after 20 seconds of silence
                yield f"data: Session Ended\n\n"
                break

            yield f"data: Listening...\n\n"
            time.sleep(1)

            transcription = record_and_transcribe(whisper_model)

            if transcription:  # If user speaks
                silence_timer = 0  
                last_response_time = time.time()

                yield f"data: Question: {transcription}\n\n"  # Send the transcribed question
                time.sleep(1)

                response = generate_response(transcription, memory)
                audio_response = text_to_speech(kokoro_pipeline, response)

                yield f"data: Speaking...\n\n"

                play_audio(audio_data=audio_response)  # ✅ This now properly waits

                time.sleep(1)  # Short delay before going back to listening
            else:
                silence_timer = time.time() - last_response_time  

    return event_stream()

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/stream/")
def stream():
    return Response(stream_with_context(continuous_interaction()), content_type="text/event-stream")

@app.route("/question/")
def chat():
    return jsonify({"message": "Started listening..."})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)