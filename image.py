from config import *
from diffusers import AutoPipelineForText2Image

def generate_image(path_id, imgfile, prompt):
	pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
	image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
	image.save(os.path.join("temp", path_id, imgfile))