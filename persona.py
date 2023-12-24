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

message = """24 hours a day, 365 days a year, NORAD tracks everything that flies in and around North America in defense of our homelands. On December 24, we have the very special mission of also tracking Santa."""

def main():
	parser = ArgumentParser()
	parser.add_argument("--improve", action="store_true", help="use Real ESRGAN to improve the video")
	parser.add_argument("--path_id", default=str(int(time.time())), help="set the path id to use")
	parser.add_argument("--speech", default=audiofile, help="path to WAV speech file")
	parser.add_argument("--image", default=imgfile, help="path to avatar file")
	args = parser.parse_args()
	tstart = time.time()

	## SET PATH
	# path_id, path = init_path_id()
	path_id = args.path_id
	path = os.path.join("temp", path_id)
	print("path_id:", path_id, "path:", path)
	os.makedirs(path, exist_ok=True)
	
	## GENERATE SPEECH	
	tspeech = "None"
	if args.speech == audiofile:
		print("-----------------------------------------")
		print("generating speech")
		t0 = time.time()
		generate_speech(path_id, audiofile, "train_grace", message, "ultra_fast")
		tspeech = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t0)))
		print("\ngenerating speech:", tspeech)
	else:
		print("using:", args.speech)
		shutil.copyfile(args.speech, os.path.join("temp", path_id, audiofile))

	## GENERATE AVATAR IMAGE
	timage = "None"
	if args.image == imgfile:
		print("-----------------------------------------")
		print("generating avatar image")
		t1 = time.time()
		avatar_description = "Young woman with long, blonde hair, smiling slightly"
		generate_image(path_id, imgfile, f"hyperrealistic digital avatar, centered, {avatar_description}, \
					rim lighting, studio lighting, looking at the camera")
		timage = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t1)))
		print("\ngenerating avatar:", timage)
	else:
		shutil.copyfile(args.image, os.path.join("temp", path_id, imgfile))

	## ANIMATE AVATAR IMAGE

	print("-----------------------------------------")
	print("animating face with driver")
	t2 = time.time()	
	# audiofile determines the length of the driver movie to trim
	# driver movie is imposed on the image file to produce the animated file
	animate_face(path_id, audiofile, driverfile, imgfile, animatedfile)
	tanimate = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t2)))
	print("\nanimating face:", tanimate)

	## MODIFY LIPS TO FIT THE SPEECH

	print("-----------------------------------------")
	print("modifying lips")
	t3 = time.time()
	os.makedirs("results", exist_ok=True)
	outfile = os.path.join("results", path_id + "_small.mp4")
	modify_lips(path_id, audiofile, animatedfile, outfile)
	tlips = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t3)))
	print("\nmodifying lips:", tlips)

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
		finalfile = outfile = os.path.join("results", path_id + "_large.mp4")
		restore_frames(os.path.join(path, audiofile), os.path.join(path, finalfile), os.path.join(path, "improve", "improved"))		
		timprove = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t4)))
		print("\nimproving video:", timprove)
	
	print("done")
	print("Overall timing")
	print("--------------")
	print("generating speech:", tspeech)
	print("generating avatar image:", timage)
	print("animating face:", tanimate)
	print("modifying lips:", tlips)
	if args.improve:
		print("improving finished video:", timprove)
	print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - tstart))))

if __name__ == '__main__':
	main()