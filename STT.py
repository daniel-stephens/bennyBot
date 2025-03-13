import whisper
import sounddevice as sd
import numpy as np

# Load Whisper model
model = whisper.load_model("small")

samplerate = 16000
duration_of_silence = 2  # seconds
threshold = 0.01  # adjust if too sensitive or not sensitive enough

# Function to detect silence
def is_silent(data, threshold):
    return np.abs(data).mean() < threshold

# Continuous audio recording and transcription
def continuous_transcription():
    while True:
        print("Listening... (start speaking)")
        audio_chunks = []
        silence_duration = 0

        with sd.InputStream(samplerate=samplerate, channels=1) as stream:
            while True:
                audio_chunk, _ = stream.read(samplerate // 2)  # read 0.5 seconds
                audio_chunks.append(audio_chunk)

                if is_silent(audio_chunk, threshold):
                    silence_duration += 0.5
                else:
                    silence_duration = 0

                if silence_duration >= duration_of_silence:
                    break

        audio_data = np.concatenate(audio_chunks, axis=0)
        audio_data = audio_data.flatten()

        # Transcribe audio
        result = model.transcribe(audio_data, language="en")

        # Output transcription
        print("Transcription:", result['text'])

# Start continuous transcription
continuous_transcription()
