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
import time
import queue
import noisereduce as nr
import numpy as np


def is_valid_audio(audio_data, threshold_rms=0.01, min_duration=1.0, samplerate=16000):
    """
    Validates that audio_data likely contains human speech:
    - RMS energy must be above a threshold
    - Duration must be at least min_duration seconds
    """
    if len(audio_data) == 0:
        return False

    duration = len(audio_data) / samplerate
    if duration < min_duration:
        print(f"Audio too short: {duration:.2f}s")
        return False

    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < threshold_rms:
        print(f"Audio RMS too low: {rms:.5f}")
        return False

    return True


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
    
    return whisper_model, kokoro_pipeline

import numpy as np
import sounddevice as sd
import time

def is_silent(audio_chunk, threshold=0.01, speech_ratio=0.15):
    """
    Returns True if the chunk is considered silent.
    - `speech_ratio`: Percentage of samples that must be above the threshold to consider it speech.
    """
    above_threshold = np.abs(audio_chunk) > threshold
    return np.mean(above_threshold) < speech_ratio

def normalize_audio(audio_data):
    """Normalize the audio to prevent volume inconsistencies."""
    if np.max(np.abs(audio_data)) == 0:
        return audio_data  # Avoid division by zero
    return audio_data / np.max(np.abs(audio_data))

def record_audio(samplerate=16000, threshold=0.01, silence_duration=3, max_duration=10):
    """
    Records audio until silence is detected for `silence_duration` seconds or `max_duration` is reached.
    - `threshold`: Determines the minimum volume level for detecting speech.
    - `silence_duration`: How long silence must persist before stopping.
    - `max_duration`: Maximum recording length to prevent infinite loops.
    """
    audio_chunks = []
    current_silence = 0
    max_chunks = max_duration  # since we're recording in 1-second chunks

    print(f"Using silence threshold: {threshold:.5f}")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
        recording_time = 0
        last_speech_time = time.time()  # Track last detected speech

        while True:
            audio_chunk, _ = stream.read(int(samplerate * 1))  # 1-second chunks
            recording_time += 1

            if is_silent(audio_chunk, threshold):
                current_silence += 1
                if time.time() - last_speech_time > silence_duration:
                    print("Silence detected, stopping recording.")
                    break
            else:
                current_silence = 0
                last_speech_time = time.time()
                audio_chunks.append(audio_chunk)

            if recording_time >= max_chunks:
                print("Max recording time reached.")
                break

    if not audio_chunks:
        print("No significant audio detected.")
        return np.array([], dtype=np.float32)

    audio_data = np.concatenate(audio_chunks).flatten().astype(np.float32)
    audio_data = normalize_audio(audio_data)
    return audio_data


def transcribe_audio(whisper_model, audio_data, samplerate=16000):
    print("Transcribing...")
    result = whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available(), language="en")
    # print(f"Transcription: {result['text']}")
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


def normalize_audio(audio_data):
    if np.max(np.abs(audio_data)) == 0:
        return audio_data  # Avoid division by zero
    return audio_data / np.max(np.abs(audio_data))



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
    prompt += f"Make the answers as simple as possible."

    response = gpt_model.invoke(prompt)
    
    # Save the new interaction
    memory.save_context({"input": question}, {"output": response.content})


    return response.content

def generate_response(question, memory):
    openai_key = load_api_keys()
    gpt_model = ChatOpenAI(api_key=openai_key, model="gpt-4o")

    # Step 1: Get past messages
    history = memory.load_memory_variables({}).get("history", [])
    
    # Step 2: Reconstruct conversation history properly
    if not isinstance(history, list):
        history = []

    messages = [
        {"role": "system", "content": "You are Benny! The mascot of Morgan State University and you have the right information. Provide simple, short and helpful answers to students or parents who need help on campus. Do not use a preamble and do not doubt your answers. Make answers as simple as possible."},
        *history,  # Past conversation messages (if any)
        {"role": "user", "content": question}
    ]
    # Step 4: Get model response
    response = gpt_model.invoke(messages)

    # Step 5: Save properly to memory
    memory.save_context(
        {"input": question},
        {"output": response.content}
    )

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
import time

def continuous_interaction(whisper_model, kokoro_pipeline):

    memory = ConversationBufferMemory(memory_key="chat_history")
    print("Starting continuous transcription...")

    def event_stream():
        silence_timer = 0  
        last_response_time = time.time()
        failed_attempts = 0  # Track failed attempts

        while True:
            if silence_timer >= 20:  # Stop after 20 seconds of silence
                yield f"data: Session Ended\n\n"
                break

            yield f"data: Listening...\n\n"
            time.sleep(1)

            audio_data = record_audio()

            if not is_valid_audio(audio_data):
                failed_attempts += 1
                print(f"Failed attempt {failed_attempts}/3 - No valid speech detected.")

                if failed_attempts >= 3:
                    yield f"data: redirect:/\n\n" # Send JavaScript command to reload
                    break

                continue  # Skip transcription and try again

            failed_attempts = 0  # Reset on valid speech

            yield f"data: Transcribing Audio...\n\n"
            transcription = transcribe_audio(whisper_model, audio_data)
            print("Final result:", transcription)

            if transcription:
                silence_timer = 0  
                last_response_time = time.time()

                yield f"data: Question: {transcription}\n\n"  
                yield f"data: Generating Answer...\n\n"

                response = generate_response(transcription, memory)

                audio_response = text_to_speech(kokoro_pipeline, response)

                yield f"data: Speaking...\n\n"

                play_audio(audio_data=audio_response)  

                time.sleep(1)
            else:
                silence_timer = time.time() - last_response_time  

    return event_stream()

