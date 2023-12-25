import time
from config import *
import cv2
import glob
import numpy as np
import os
from basicsr.utils import imwrite
from pathos.pools import ParallelPool
import subprocess
import platform
from mutagen.wave import WAVE
import tqdm
from p_tqdm import *
import torch
from PIL import Image
from RealESRGAN import RealESRGAN

def vid2frames(vidPath, framesOutPath):
    print(vidPath)
    print(framesOutPath)
    vidcap = cv2.VideoCapture(vidPath)
    success,image = vidcap.read()
    frame = 1
    while success:
      cv2.imwrite(os.path.join(framesOutPath, str(frame).zfill(5) + '.png'), image)
      success,image = vidcap.read()
      frame += 1

def restore_frames(audiofilePath, videoOutPath, improveOutputPath):
    no_of_frames = count_files(improveOutputPath)
    audio_duration = get_audio_duration(audiofilePath)
    framesPath = improveOutputPath + "/%5d.png"
    fps = no_of_frames/audio_duration
    command = f"ffmpeg -y -r {fps} -f image2 -i {framesPath} -i {audiofilePath} -vcodec mpeg4 -b:v 20000k {videoOutPath}"
    print(command)
    subprocess.call(command, shell=platform.system() != 'Windows')

def get_audio_duration(audioPath):
    audio = WAVE(audioPath)
    duration = audio.info.length
    return duration    

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def improve(disassembledPath, improvedPath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    files = glob.glob(os.path.join(disassembledPath,"*.png"))
    
    # pool = ParallelPool(nodes=20)    
    # results = pool.amap(real_esrgan, files, [model]*len(files), [improvedPath] * len(files))
    results = t_map(real_esrgan, files, [model]*len(files), [improvedPath] * len(files))

def real_esrgan(img_path, model, improvedPath):
    image = Image.open(img_path).convert('RGB')
    sr_image = model.predict(image)
    img_name = os.path.basename(img_path)
    sr_image.save(os.path.join(improvedPath, img_name))	


# def process(img_path, improveOutputPath):
#     only_center_face=True
#     aligned=True
#     ext='auto'
#     weight=0.5
#     upscale=1
#     arch = 'clean'
#     channel_multiplier = 2
#     model_name = 'GFPGANv1.3'
#     url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

#     # determine model paths
#     model_path = os.path.join('gfpgan_models', model_name + '.pth')
#     if not os.path.isfile(model_path):
#         model_path = os.path.join('gfpgan/weights', model_name + '.pth')
#     if not os.path.isfile(model_path):
#         # download pre-trained models from url
#         model_path = url

#     restorer = GFPGANer(
#         model_path=model_path,
#         upscale=upscale,
#         arch=arch,
#         channel_multiplier=channel_multiplier,
#         bg_upsampler=None)

#     # read image
#     img_name = os.path.basename(img_path)   
#     basename, ext = os.path.splitext(img_name)
#     input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#     # restore faces and background if necessary
#     cropped_faces, restored_faces, restored_img = restorer.enhance(
#         input_img,
#         has_aligned=aligned,
#         only_center_face=only_center_face,
#         paste_back=True,
#         weight=weight)

#     # save faces
#     for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
#         # save cropped face
#         save_crop_path = os.path.join(improveOutputPath, 'cropped_faces', f'{basename}.png')
#         imwrite(cropped_face, save_crop_path)
#         # save restored face
#         save_face_name = f'{basename}.png'
#         save_restore_path = os.path.join(improveOutputPath, 'restored_faces', save_face_name)
#         imwrite(restored_face, save_restore_path)
#         # save comparison image
#         cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
#         imwrite(cmp_img, os.path.join(improveOutputPath, 'cmp', f'{basename}.png'))

#     # save restored img
#     if restored_img is not None:
#         if ext == 'auto':
#             extension = ext[1:]
#         else:
#             extension = ext
        
#         save_restore_path = os.path.join(improveOutputPath, 'restored_imgs', f'{basename}.{extension}')
#         imwrite(restored_img, save_restore_path)    
#     print(f'Processed {img_name} ...')

# def improve_faces(improveInputPath, improveOutputPath):
#     if improveInputPath.endswith('/'):
#         improveInputPath = improveInputPath[:-1]
#     if os.path.isfile(improveInputPath):
#         img_list = [improveInputPath]
#     else:
#         img_list = sorted(glob.glob(os.path.join(improveInputPath, '*')))

#     os.makedirs(improveInputPath, exist_ok=True)
#     os.makedirs(improveOutputPath, exist_ok=True)
    
#     pool = ParallelPool(nodes=10)    
#     results = pool.amap(process, img_list, [improveOutputPath] * len(img_list))
#     while not results.ready():
#         time.sleep(5); print(".", end=' ')
