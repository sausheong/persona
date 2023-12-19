import time
from config import *
import cv2
import glob
import numpy as np
import os
from basicsr.utils import imwrite
from pathos.pools import ParallelPool

from gfpgan import GFPGANer

def process(img_path):
    only_center_face=True
    aligned=True
    ext='auto'
    weight=0.5
    upscale=1
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

    # determine model paths
    model_path = os.path.join('gfpgan_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None)

    # read image
    img_name = os.path.basename(img_path)   
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=aligned,
        only_center_face=only_center_face,
        paste_back=True,
        weight=weight)

    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # save cropped face
        save_crop_path = os.path.join(improve_output, 'cropped_faces', f'{basename}.png')
        imwrite(cropped_face, save_crop_path)
        # save restored face
        save_face_name = f'{basename}.png'
        save_restore_path = os.path.join(improve_output, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)
        # save comparison image
        cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
        imwrite(cmp_img, os.path.join(improve_output, 'cmp', f'{basename}.png'))

    # save restored img
    if restored_img is not None:
        if ext == 'auto':
            extension = ext[1:]
        else:
            extension = ext
        
        save_restore_path = os.path.join(improve_output, 'restored_imgs', f'{basename}.{extension}')
        imwrite(restored_img, save_restore_path)    
    print(f'Processed {img_name} ...')

def improve_faces(improve_input, improve_output):
    if improve_input.endswith('/'):
        improve_input = improve_input[:-1]
    if os.path.isfile(improve_input):
        img_list = [improve_input]
    else:
        img_list = sorted(glob.glob(os.path.join(improve_input, '*')))

    os.makedirs(improve_input, exist_ok=True)
    os.makedirs(improve_output, exist_ok=True)
    
    pool = ParallelPool(nodes=10)
    results = pool.amap(process, img_list)
    while not results.ready():
        time.sleep(5); print(".", end=' ')

    # for img_path in img_list:
    #     process(img_path)

# if __name__ == '__main__':
#     main()
