#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import math
import numpy as np
# from skimage.measure import compare_ssim as ssim_f
from skimage.metrics import structural_similarity as ssim_f

class SSIMCompute(object):
    def __init__(self, s_offset):
        self.s_offset = s_offset
        if self.s_offset:
            self.gt_image_dir = "images_crop16"
        else:
            self.gt_image_dir = "images"
    
    def psnr_f(self, img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    def evaluate_single_pair(self, gt_dir, gen_dir):
        ssims = []
        psnrs =[]
        
        if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
            return None, None
        gen_image_item_list = list(os.listdir(os.path.join(gen_dir, "images")))
        gt_image_item_list = list(os.listdir(os.path.join(gt_dir, self.gt_image_dir)))
        if len(gen_image_item_list) < 5 or len(gt_image_item_list) < 5:
            return None, None
        gen_image_item_list.sort()
        gt_image_item_list.sort()
        leng = min(len(gen_image_item_list), len(gt_image_item_list))
        gen_image_item_list = gen_image_item_list[:leng]
        gen_image_path_list = [os.path.join(gen_dir, "images", img_item) for img_item in gen_image_item_list]

        img_format = gt_image_item_list[0].split('.')[-1]
        gt_image_path_list = [os.path.join(gt_dir, self.gt_image_dir, img_item.split('.')[0] + ".{}".format(img_format)) for img_item in gen_image_item_list]
        
        for real_path, fake_path in zip(gt_image_path_list, gen_image_path_list):
            
            f_i = cv2.imread(fake_path)
            r_i = cv2.imread(real_path)
            
            r_i = cv2.resize(r_i, (256,256), interpolation = cv2.INTER_AREA)
            f_i = cv2.resize(f_i, (256,256), interpolation = cv2.INTER_AREA)
            
            ssim = ssim_f(f_i,r_i,multichannel=True)
            
            f_i = cv2.cvtColor(f_i, cv2.COLOR_BGR2GRAY)
            r_i = cv2.cvtColor(r_i, cv2.COLOR_BGR2GRAY)
            psnr = self.psnr_f(f_i,r_i)

            psnrs.append(psnr)
            ssims.append(ssim)

        # return each image's result
        return  ssims, psnrs
    
        