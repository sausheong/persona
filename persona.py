from config import *
from speech import generate_speech
from image import generate_image
from lips import modify_lips
import humanize
import datetime as dt
from argparse import ArgumentParser
import shutil

import os
import glob
from improve import improve, vid2frames, restore_frames
from animate_face import animate_face

def main():
	parser = ArgumentParser()
	parser.add_argument("--improve", action="store_true", help="use Real ESRGAN to improve the video")
	parser.add_argument("--speech", default=audiofile, help="path to WAV speech file")
	parser.add_argument("--image", default=imgfile, help="path to avatar file")

	# path_id = "temp/1703250663"

	## SET PATH
	path_id, path = init_path_id()
	print("path_id:", path_id, "path:", path)

	args = parser.parse_args()
	## GENERATE SPEECH	
	if args.speech == audiofile:
		print("-----------------------------------------")
		print("generating speech")
		t0 = time.time()
		generate_speech(path_id, audiofile, "train_lescault&tom", "Merry Christmas! May the holiday bring \
					you endless joy, laughter, and quality time with friends and family!")
		print("\ngenerating speech:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t0))))
	else:
		print("using:", args.speech)
		shutil.copyfile(args.speech, os.path.join("temp", path_id, audiofile))

	## GENERATE AVATAR IMAGE
	if args.image == imgfile:
		print("-----------------------------------------")
		print("generating avatar image")
		t1 = time.time()
		avatar_description = "Santa Claus in a white beard and wearing a red hat, slightly smiling"
		generate_image(path_id, imgfile, f"hyperrealistic digital avatar, centered, {avatar_description}, \
					rim lighting, studio lighting, looking at the camera")
		print("\ngenerating avatar:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t1))))
	else:
		shutil.copyfile(args.image, os.path.join("temp", path_id, imgfile))

	## ANIMATE AVATAR IMAGE

	print("-----------------------------------------")
	print("animating face with driver")
	t2 = time.time()	
	# audiofile determines the length of the driver movie to trim
	# driver movie is imposed on the image file to produce the animated file
	animate_face(path_id, audiofile, driverfile, imgfile, animatedfile)
	print("\nanimating face:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t2))))

	## MODIFY LIPS TO FIT THE SPEECH

	print("-----------------------------------------")
	print("modifying lips")
	t3 = time.time()
	modify_lips(path_id, audiofile, animatedfile, outfile)
	print("\nmodifying lips:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t3))))

	## IMPROVE THE OUTPUT VIDEO
	if args.improve:
		t4 = time.time()
		print("-----------------------------------------")
		print("converting video to frames")
		shutil.rmtree(os.path.join(path, "improve"), ignore_errors=True)
		os.makedirs(os.path.join(path, "improve", "disassembled"), exist_ok=True)
		os.makedirs(os.path.join(path, "improve", "improved"), exist_ok=True)	

		vid2frames(os.path.join(path, outfile), os.path.join(path, "improve", "disassembled"))
		print("-----------------------------------------")
		print("improving face")
		improve(os.path.join(path, "improve", "disassembled"), os.path.join(path, "improve", "improved"))
		print("-----------------------------------------")
		print("restoring frames")
		restore_frames(os.path.join(path, audiofile), os.path.join(path, "final.mp4"), os.path.join(path, "improve", "improved"))
		
		print("\nimproving video:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t4))))
	
	print("done")
	print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t0))))

if __name__ == '__main__':
	main()