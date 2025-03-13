import os
import soundfile as sf
# from IPython.display import Audio, display
from kokoro import KPipeline

output_folder = 'output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

pipeline = KPipeline(lang_code='p')

text = '''
Olá e bem-vindo! Neste tutorial, exploraremos como usar o Kokoro-82M!
'''

generator = pipeline(
    text, 
    voice='pf_dora', 
    speed=1, split_pattern=r'\n+'
)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f'{output_folder}/pt_{i}.wav', audio, 24000) 

