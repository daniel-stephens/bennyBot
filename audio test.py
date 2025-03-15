import sounddevice as sd
from scipy.io.wavfile import write

# Sampling frequency
fs = 44100  

# Record duration in seconds
seconds = 5  

print("Recording started...")
# Start recording
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)

# Wait until recording finishes
sd.wait()
print("Recording finished.")

# Save the recording as a WAV file
write("my_voice.wav", fs, recording)
print("Audio saved as my_voice.wav")
