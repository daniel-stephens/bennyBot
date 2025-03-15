from kokoro import KPipeline
from playsound import playsound
import soundfile as sf
import tempfile
import whisper
import sounddevice as sd
import numpy as np
import torch
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv() # Load environment variables from .env 
openai_keys = os.getenv("OPENAI_API_KEY")

if not openai_keys: 
   raise ValueError("Please provide an OpenAI API key.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
gptmodel = ChatOpenAI(model="gpt-4o")


# Load Whisper model with GPU support
model = whisper.load_model("small").to("cuda" if torch.cuda.is_available() else "cpu")

# Audio parameters
samplerate = 16000
channels = 1
duration_of_silence = 2  # seconds
threshold = 0.01  # adjust sensitivity as needed
# Initialize the pipeline



# Check for silence in audio
def is_silent(data, threshold):
    return np.abs(data).mean() < threshold

# Continuous audio recording and transcription
def continuous_transcription():
    pipeline = KPipeline(lang_code='a')  # American English
    while True:
        print("\nListening... (start speaking)")
        audio_chunks = []
        silence_duration = 0

        # Start audio stream
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32') as stream:
            while True:
                audio_chunk, _ = stream.read(int(samplerate * 0.5))  # read 0.5 second chunks
                audio_chunks.append(audio_chunk)

                if is_silent(audio_chunk, threshold):
                    silence_duration += 0.5
                else:
                    silence_duration = 0

                if silence_duration >= duration_of_silence:
                    print("Detected silence, processing audio...")
                    break

        # Concatenate audio data
        audio_data = np.concatenate(audio_chunks).flatten()

        # Ensure audio is in correct format (float32 numpy array)
        audio_data = audio_data.astype(np.float32)

        # Transcribe audio using Whisper directly from numpy array
        result = model.transcribe(audio_data, fp16=torch.cuda.is_available(), language="en")

        # Output transcription
        print("Transcription:", result['text'])

        question = result["text"]

        PROMPT = """ Give a concise answer for the question asked. {question} """
        ConversationPrompt = PROMPT.format(question=question)

        output = gptmodel.invoke(ConversationPrompt)
        text = output.content

        # Generate audio
        generator = pipeline(
            text, voice='af_heart',  # choose voice
            speed=1
        )

        full_audio = []
        for _, _, audio in generator:
            full_audio.extend(audio)

        # Temporarily save audio and play directly
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
            sf.write(tmp_file.name, full_audio, 24000)
            playsound(tmp_file.name)
        

# Run the transcription
continuous_transcription()




