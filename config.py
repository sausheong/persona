import torch
import time
import os

path_id = ""
checkpoint_path="checkpoints/wav2lip_gan.pth"
outfile="out.mp4"
audiofile="tmp.wav"
imgfile="avatar.png"
driverfile="assets/driver06.mp4"
animatedfile="animated.mp4"
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

import warnings
warnings.filterwarnings('ignore')

def init_path_id():    
    path_id = str(int(time.time()))
    path = os.path.join("temp", path_id)
    os.makedirs(path, exist_ok=True)
    return path_id, path


