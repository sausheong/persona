from config import *
from image import generate_image
import humanize
import datetime as dt
from argparse import ArgumentParser
import shutil

import os
from animate_face import animate_face
import subprocess, platform

avatar_description = "Young asian man, with short brunette hair, slightly smiling"

def main():
	parser = ArgumentParser()
	parser.add_argument("--image", default=imgfile, help="path to avatar file")
	parser.add_argument("--path_id", default=str(int(time.time())), help="set the path id to use")
	parser.add_argument("--pitch", default=1.0, help="change pitch of voice, 1.0 is original, higher number is higher pitch")
	args = parser.parse_args()
	tstart = time.time()

	## SET PATH
	path_id = args.path_id
	path = os.path.join("temp", path_id)
	os.makedirs(path, exist_ok=True)

	## GENERATE AVATAR IMAGE
	timage = "None"
	if args.image == imgfile:
		print("-----------------------------------------")
		print("generating avatar image")
		t1 = time.time()	
		generate_image(path_id, imgfile, f"hyperrealistic digital avatar, centered, \
			{avatar_description}, rim lighting, studio lighting, looking at the camera")
		timage = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t1)))
		print("\ngenerating avatar:", timage)
	else:
		shutil.copyfile(args.image, os.path.join("temp", path_id, imgfile))

	## EXTRACT SPEECH FROM MP4
	print("-----------------------------------------")
	print("extracting speech from mp4")		
	t2 = time.time()
	wavoutfile = os.path.join(path, audiofile)
	command = 'ffmpeg -i {} -acodec pcm_s16le -ar 44100 -ac 1 {}'.format(driverfile, wavoutfile)
	subprocess.call(command, shell=platform.system() != 'Windows')		
	tspeech = humanize.naturaldelta(dt.timedelta(microseconds=int(time.time() - t2)))
	print("\nextracting speech:", tspeech)

	## ANIMATE AVATAR IMAGE
	print("-----------------------------------------")
	print("animating face with driver")
	t3 = time.time()	
	# audiofile determines the length of the driver movie to trim
	# driver movie is imposed on the image file to produce the animated file
	animate_face(path_id, audiofile, driverfile, imgfile, animatedfile)
	tanimate = humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - t3)))
	print("\nanimating face:", tanimate)

	## CHANGING THE PITCH OF THE VOICE
	print("-----------------------------------------")
	print("changing pitch of voice")		
	t4 = time.time()
	wavpitchedfile = os.path.join(path, "pitched.wav")
	# command = 'ffmpeg -i {} -af "rubberband=pitch={}" {}'.format(wavoutfile, args.pitch, wavpitchedfile)
	command = 'ffmpeg -i {} -af "asetrate=44100*{},aresample=44100,atempo=1/{}" {}'.format(wavoutfile, args.pitch, args.pitch, wavpitchedfile)
	
	subprocess.call(command, shell=platform.system() != 'Windows')		
	tpitch = humanize.naturaldelta(dt.timedelta(microseconds=int(time.time() - t4)))
	print("\changing pitch:", tpitch)

	## COMBINING ANIMATION WITH SPPECH
	print("-----------------------------------------")
	print("combining animation with speech")	
	t5 = time.time()
	animatedoutfile = os.path.join(path, animatedfile)
	finaloutfile = os.path.join("results", path_id + "_animated.mp4")
	command = 'ffmpeg -i {} -i {} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {}'.format(animatedoutfile, wavpitchedfile, finaloutfile)
	subprocess.call(command, shell=platform.system() != 'Windows')	
	tcombi = humanize.naturaldelta(dt.timedelta(microseconds=int(time.time() - t5)))
	print("\combining animation with speech:", tcombi)


	print("done")
	print("Overall timing")
	print("--------------")
	print("generating avatar image:", timage)
	print("extracting speech from mp4:", tspeech)
	print("animating face:", tanimate)
	print("changing pitch of voice:", tpitch)
	print("combining animation with speech:", tcombi)
	print("total time:", humanize.naturaldelta(minimum_unit="microseconds", value=dt.timedelta(seconds=int(time.time() - tstart))))

if __name__ == '__main__':
	main()