from config import *
from openai import OpenAI

def generate_speech(voice, text):
	client = OpenAI(api_key="")
	response = client.audio.speech.create(
		model="tts-1",
		voice=voice,
		input=text
	)
	response.stream_to_file(audiofile)