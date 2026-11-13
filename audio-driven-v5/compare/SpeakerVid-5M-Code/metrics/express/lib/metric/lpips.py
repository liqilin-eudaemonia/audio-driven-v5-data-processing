import os
# import cv2
import torch
import lpips
import numpy as np


class Lpips(object):
    def __init__(self, net="alex"):
        self.loss_fn = lpips.LPIPS(net=net).to('cuda')
        
    def inference(self, img_gt, img_pred, gt_seg=None, pred_seg=None, is_seg = False):
        ### input img: RGB, [0, 255]
        ### lpips input: img should be RGB and normalized to [-1, 1]

        img_gt = img_gt / 255. * 2. - 1.
        img_pred = img_pred / 255. * 2. - 1.

        if is_seg:
            img_gt = (0.5*(img_gt+1)*gt_seg)*2 - 1
            img_pred = (0.5*(img_pred+1)*pred_seg)*2 - 1

        with torch.no_grad():
            lpips_s = self.loss_fn(img_gt,img_pred)
        
        lpips_s = lpips_s.detach().cpu().numpy()

        return lpips_s
