from config import *
from diffusers import AutoPipelineForText2Image
from argparse import ArgumentParser
import humanize
import datetime as dt

def generate_image(path_id, imgfile, prompt):
	pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
	image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
	image.save(os.path.join("temp", path_id, imgfile))

def generate_images(path_id, imgfile, prompt, times=1):
	pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
	for i in range(times):
		image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
		image.save(os.path.join("temp", path_id, str(i) + "_" + imgfile))

if __name__ == '__main__':
	path_id = str(int(time.time()))
	path = os.path.join("temp", "image", path_id)
	os.makedirs(path, exist_ok=True)

	parser = ArgumentParser()
	parser.add_argument("--prompt", default="Young woman with long, blonde hair, smiling slightly", 
					 help="avatar prompt")
	parser.add_argument("--times", type=int, default=1, help="number of avatars to generate")	
	args = parser.parse_args()

	tstart = time.time()	

	generate_images(os.path.join("image", path_id), "avatar.png", 
				f"hyper-realistic digital avatar, centered, {args.prompt}, \
				rim lighting, studio lighting, looking at the camera", args.times)

	print("total time:", humanize.naturaldelta(dt.timedelta(seconds=int(time.time() - tstart))))	