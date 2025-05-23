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

# Record and transcribe audio
def record_and_transcribe(whisper_model, samplerate=16000, threshold=0.01, silence_duration=2):
    audio_chunks = []
    current_silence = 0

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
        while True:
            audio_chunk, _ = stream.read(int(samplerate * 0.5))
            audio_chunks.append(audio_chunk)

            if is_silent(audio_chunk, threshold):
                current_silence += 0.5
            else:
                current_silence = 0

            if current_silence >= silence_duration:
                break

    audio_data = np.concatenate(audio_chunks).flatten()
    # Ensure audio is in correct format (float32 numpy array)
    audio_data = audio_data.astype(np.float32)
    result = whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available(), language="en")
    return result["text"]

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

    prompt = f"Using the conversation history below, answer the following question concisely:\n\n"
    prompt += f"Conversation History:\n{chat_history}\n\n"
    prompt += f"Question: {question}"

    response = gpt_model.invoke(prompt)
    
    # Save the new interaction
    memory.save_context({"input": question}, {"output": response.content})

    return response.content

# Continuous transcription and interaction loop
def continuous_interaction():
    whisper_model, kokoro_pipeline, memory = initialize_models()

    print("Starting continuous transcription...")

    while True:
        print("Listening...")
        transcription = record_and_transcribe(whisper_model)
        print("Transcription:", transcription)

        response = generate_response(transcription, memory)
        audio_response = text_to_speech(kokoro_pipeline, response)
        play_audio(audio_data=audio_response)



if __name__ == '__main__':

    continuous_interaction()
