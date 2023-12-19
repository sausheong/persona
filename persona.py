from config import improve_input, improve_output
from speech import generate_speech
from image import generate_image
from lips import modify_lips, vid2frames, restore_frames

import os
import glob
from improve import improve_faces


def main():
	# generate_speech("alloy", "A diner was happily tucking into slices of pizza bought from Little Caesars \
	# 				eatery in Funan Mall on Dec 11 when a cockroach suddenly crawled out from \
	# 				under a pizza.")

	# avatar_description = "young man with short brunette hair, slightly smiling"
	# generate_image(f"hyperrealistic digital avatar, {avatar_description}")
	modify_lips()
	# for filename in glob.glob(improve_input + "/frames/*.png"):
	# 	os.remove(filename)		
	# vid2frames("results.mp4", improve_input + "/frames")
	# improve_faces(improve_input, improve_output)
	# restore_frames("final.mp4")
	
if __name__ == '__main__':
	main()