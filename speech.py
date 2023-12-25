# from config import *
# from openai import OpenAI
# import os

# def openai_generate_speech(audiofile, voice, text):
# 	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# 	response = client.audio.speech.create(
# 		model="tts-1",
# 		voice=voice,
# 		input=text
# 	)
# 	response.stream_to_file(audiofile)

import os

import torch
import torchaudio
import time
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices
import humanize
import datetime as dt

def generate_speech(path_id, outfile, voice, text, speed="standard"):
    tts = TextToSpeech(kv_cache=True, half=True)
    selected_voices = voice.split(',')
    for k, selected_voice in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel)

        gen, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples, 
                                             conditioning_latents=conditioning_latents,
                                            return_deterministic_state=True,
                                            preset=speed)
        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join("temp", path_id, outfile), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join("temp", path_id, outfile), gen.squeeze(0).cpu(), 24000)
 


if __name__ == '__main__':
    path_id = os.path.join("temp", "audio", str(int(time.time())))
    os.makedirs(path_id, exist_ok=True)
    tstart = time.time()
    message = """Apple today confirmed that it will be permanently closing its Infinite Loop retail store in 
Cupertino, California on January 20. Infinite Loop served as Apple's headquarters between the mid-1990s and 
2017, when its current Apple Park headquarters opened a few miles away."""
    generate_speech(os.path.join("audio", str(int(time.time()))), "christmas.wav", "train_grace", 
                    message, "ultra_fast")
        
    # openai_generate_speech("speech.mp3", "onyx", 
    #                 "Merry Christmas! May the holiday bring you endless joy, laughter, \
    #                 and quality time with friends and family!")    
    print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - tstart))))