import os
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.memory import ConversationBufferMemory
import time
import queue
import numpy as np



# Load environment variables
def load_api_keys():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Please provide an OpenAI API key.")
    return openai_key





def transcribe_audio(whisper_model, audio_data, samplerate=16000):
    print("Transcribing...")
    result = whisper_model.transcribe(audio_data, fp16=torch.cuda.is_available(), language="en")
    # print(f"Transcription: {result['text']}")
    return result["text"]




# Generate speech audio from text
def text_to_speech(pipeline, text, voice='am_onyx', speed=1):
    generator = pipeline(text, voice=voice, speed=speed)
    audio_data = []
    for _, _, audio in generator:
        audio_data.extend(audio)
    return np.array(audio_data, dtype=np.float32)


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
