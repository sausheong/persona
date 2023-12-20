from config import improve_input, improve_output
from speech import generate_speech
from image import generate_image
from lips import modify_lips

import os
import glob
from improve import improve_faces, vid2frames, restore_frames
from animate_face import animate_face

def main():
	print("-----------------------------------------")
	print("generating speech")
	generate_speech("alloy", "Three people were injured by an armed lone attacker in a slashing incident at a mall in Pasir Ris Street 72, with shoppers and business owners running for cover amid the chaos.")
	print("-----------------------------------------")
	print("generating avatar image")
	avatar_description = "young man with short brunette hair, slightly smiling"
	generate_image(f"hyperrealistic digital avatar, {avatar_description}")
	print("-----------------------------------------")
	print("animating face with driver")
	animate_face()
	print("-----------------------------------------")
	print("modifying lips")
	modify_lips()
	print("-----------------------------------------")
	print("converting video to frames")
	for filename in glob.glob(improve_input + "/frames/*.png"):
		os.remove(filename)		
	vid2frames("temp/lips.mp4", improve_input + "/frames")
	print("-----------------------------------------")
	print("improving face")
	improve_faces(improve_input, improve_output)
	print("-----------------------------------------")
	print("restoring frames")
	restore_frames("final.mp4")
	print("done")
if __name__ == '__main__':
	main()