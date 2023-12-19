from config import *
from openai import OpenAI

def generate_speech(voice, text):
	client = OpenAI(api_key="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")
	response = client.audio.speech.create(
		model="tts-1",
		voice=voice,
		input=text
	)
	response.stream_to_file(audiofile)