#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../../")

from lib.rt_gene.gaze_blink_detect import GazeBlink

class GazeMetric(object):
    def __init__(self, s_offset=False):
        self.gaze_model = GazeBlink()
        
        # self.gaze_model.eval()
        # self.gaze_model.cuda()
        
        self.batch_size = 8
        
        self.s_offset = s_offset
        if self.s_offset:
            self.gt_image_dir = "images_crop16"
            self.gt_gaze_dir = "gaze_blink_crop16"
        else:
            self.gt_image_dir = "images"
            self.gt_gaze_dir = "gaze_blink"
        
    def inference(self, images_list):
        gaze0_list = []
        gaze1_list = []
        blink_list = []
        gaze_feat_list = []
        b = len(images_list)
        
        for i in range(round(b / self.batch_size + 0.49999)):
            image_clip = images_list[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            
            for img in image_clip:
                gaze0, gaze1, blink, gaze_feat = self.gaze_model.get_gaze_blink(img)
                if gaze0 is None or gaze1 is None or blink is None:
                    continue
                gaze0_list.append(gaze0)
                gaze1_list.append(gaze1)
                blink_list.append(blink)
                gaze_feat_list.append(gaze_feat)
        
        gaze = np.stack([gaze0_list, gaze1_list], 1)
        blink = np.array(blink_list)
        gaze_feat = np.array(gaze_feat_list)
        # print(gaze.shape, blink.shape)
        
        return gaze, blink[:, None], gaze_feat
    
    def inference_single(self, image_path_list):
        
        images_list = []
        
        for path in image_path_list:
            images_list.append(cv2.imread(path))
        
        gaze, blink, gaze_feat = self.inference(images_list)
        
        return gaze, blink, gaze_feat
    
    
    def detect_single(self, gen_dir):
        
        if not os.path.exists(gen_dir):
            return 
        gen_img_items = list(os.listdir(os.path.join(gen_dir, "images")))
        if len(gen_img_items) < 5:
            return None, None
        gen_img_items.sort()
        gen_image_path_list = [os.path.join(gen_dir, "images", img_item) for img_item in gen_img_items]
        # gen_gaze_dir = os.path.join(gen_dir, "gaze_blink_crop16")
        # os.makedirs(gen_gaze_dir, exist_ok=True)
            
        gaze, blink, gaze_feat = self.inference_single(gen_image_path_list)
        gen_params = np.concatenate([gaze, blink], 1)
        gen_gaze_path = os.path.join(gen_dir, "gaze_blink_crop16.npy")
        gen_gaze_feat_path = os.path.join(gen_dir, "gaze_feat_crop16.npy")
        np.save(gen_gaze_path, gen_params)
        np.save(gen_gaze_feat_path, gaze_feat)
        
        
        # return each images' result
        return gen_params
            

    