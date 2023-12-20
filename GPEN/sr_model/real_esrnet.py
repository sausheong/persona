import os
import torch
import numpy as np
from rrdbnet_arch import RRDBNet
from torch.nn import functional as F
import torch


class RealESRNet(object):
    def __init__(self, base_dir=os.path.dirname(__file__), model=None, scale=2):
        self.base_dir = base_dir
        self.scale = scale
        self.load_srmodel(base_dir, model)
        self.srmodel_trt = None

    def load_srmodel(self, base_dir, model):
        self.scale = 2 if "x2" in model else 4 if "x4" in model else -1
        if self.scale == -1:
            raise Exception("Scale not supported")
        self.srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=23, num_grow_ch=32, scale=self.scale)
        if model is None:
            loadnet = torch.load(os.path.join(self.base_dir, 'weights', 'realesrnet_x2.pth'))
        else:
            loadnet = torch.load(os.path.join(self.base_dir, 'weights', model+'.pth'))
        self.srmodel.load_state_dict(loadnet['params_ema'], strict=True)
        self.srmodel.eval()
        self.srmodel = self.srmodel.cuda()

    def build_trt(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).cuda()
        print('building trt model srmodel')
        from torch2trt import torch2trt
        self.srmodel_trt = torch2trt(self.srmodel, [img], fp16_mode=True)
        print('sucessfully built')
        del self.srmodel

    def process_trt(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).cuda()

        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')
        try:
            with torch.no_grad():
                output = self.srmodel_trt(img)
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            return output
        except:
            return None

    def process(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).cuda()
        # print(img.shape)

        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')
        try:
            with torch.no_grad():
                output = self.srmodel(img)
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            return output
        except:
            return None

