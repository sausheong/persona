from config import *
from openai import OpenAI
import os

def generate_speech(voice, text):
	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
	response = client.audio.speech.create(
		model="tts-1",
		voice=voice,
		input=text
	)
	response.stream_to_file(audiofile)