import torch

checkpoint_path="checkpoints/wav2lip_gan.pth"
outfile="temp/lips.mp4"
audiofile="temp/tmp.mp3"
imgfile="temp/tmp.png"
sourcefile="temp/source.mp4"
static=False
fps=25
pads=[0, 10, 0, 0]
face_det_batch_size=16
wav2lip_batch_size=128
resize_factor=1
crop=[0, -1, 0, -1]
box=[-1, -1, -1, -1]
img_size = 96
rotate=False
nosmooth=False
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

improve_input="temp/improve/framesout"
improve_output="temp/improve/gfpgan_results"