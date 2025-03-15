from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf

# Initialize the pipeline
pipeline = KPipeline(lang_code='a')  # American English

# Input text
text = '''
Hey I am Benny!! Ask me anything
'''

# Generate audio for the whole text
generator = pipeline(
    text, voice='af_heart',  # choose voice
    speed=1
)

# Concatenate audio segments
full_audio = []

for _, _, audio in generator:
    full_audio.extend(audio)

# Save and display the entire audio
sf.write('full_text.wav', full_audio, 24000)
display(Audio(data=full_audio, rate=24000, autoplay=True))
