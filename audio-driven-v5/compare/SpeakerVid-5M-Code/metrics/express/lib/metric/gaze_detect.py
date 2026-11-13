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

from lib.l2cs.l2cs import L2CS_Res50

class GazeMetric(object):
    def __init__(self, model_path="assets/L2CSNet_gaze360.pkl", s_offset=False):
        self.gaze_model = L2CS_Res50(model_path)
        
        self.gaze_model.eval()
        self.gaze_model.cuda()
        
        self.batch_size = 8
        
        self.s_offset = s_offset
        if self.s_offset:
            self.gt_image_dir = "images_crop16"
            self.gt_gaze_dir = "gaze_crop16"
        else:
            self.gt_image_dir = "images"
            self.gt_gaze_dir = "gaze"
        
    def inference(self, images_tensor):
        pitch_list = []
        yaw_list = []
        
        b, c, h, w = images_tensor.shape
        
        images_tensor = images_tensor.cuda()
        for i in range(round(b / self.batch_size + 0.49999)):
            image_clip = images_tensor[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            _, yaw, pitch = self.gaze_model(image_clip)
            
            pitch_list.append(pitch)
            yaw_list.append(yaw)
            
        pitch = torch.cat(pitch_list, 0).detach().cpu().numpy()
        yaw = torch.cat(yaw_list, 0).detach().cpu().numpy()
        
        return pitch, yaw
    
    def inference_single(self, image_path_list):
        images_tensor = self.transform(image_path_list)
        
        with torch.no_grad():
        
            pitch, yaw = self.inference(images_tensor)
        
        return pitch, yaw
    
    def evaluate_single_pair(self, gt_dir, gen_dir):
        
        if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
            return None, None
        gen_img_items = list(os.listdir(os.path.join(gen_dir, "images")))
        gt_img_items = list(os.listdir(os.path.join(gt_dir, self.gt_image_dir)))
        if len(gt_img_items) < 5 or len(gen_img_items) < 5:
            return None, None
        gen_img_items.sort()
        gt_img_items.sort()
        leng = min(len(gen_img_items), len(gt_img_items))
        gen_img_items = gen_img_items[:leng]
        gen_image_path_list = [os.path.join(gen_dir, "images", img_item) for img_item in gen_img_items]
        img_format = gen_img_items[0].split('.')[-1]

        gt_img_format = gt_img_items[0].split('.')[-1]
        
        ### new version: each results save and read individual
        gt_params_dict = {}
        gt_image_path_list = []
        gt_gaze_path_list = []
        for img_item in gt_img_items:
            gt_gaze_dir = os.path.join(gt_dir, self.gt_gaze_dir)
            os.makedirs(gt_gaze_dir, exist_ok=True)
            if os.path.exists(os.path.join(gt_dir, self.gt_gaze_dir, img_item.split(".")[0] + ".npy")):
                gt_param_temp = np.load(os.path.join(gt_dir, self.gt_gaze_dir, img_item.split(".")[0] + ".npy"), allow_pickle=True)
                if img_item.split(".")[0].replace(self.gt_image_dir, "images")  + ".{}".format(img_format) in gen_img_items:
                    gt_params_dict[os.path.join(gen_dir, self.gt_image_dir, img_item)] = gt_param_temp
            else:
                img_path = os.path.join(gt_dir, self.gt_image_dir, img_item)
                gt_image_path_list.append(img_path)
                gt_gaze_path = os.path.join(gt_dir, self.gt_gaze_dir, img_item.split(".")[0] + ".npy")
                gt_gaze_path_list.append(gt_gaze_path)
                
        if len(gt_image_path_list) > 0:
            gt_pitch, gt_yaw = self.inference_single(gt_image_path_list)
            gt_params = np.concatenate([gt_pitch, gt_yaw], 1)
            for i in range(len(gt_image_path_list)):
                np.save(gt_gaze_path_list[i], gt_params[i])
                gt_params_dict[gt_image_path_list[i].replace(gt_dir, gen_dir)] = gt_params[i]
                
        gt_params_list = []
        
        for key in gen_image_path_list:
            gt_params_list.append(gt_params_dict[key.replace(img_format, gt_img_format).replace("images", self.gt_image_dir)])
        
        gt_params = np.array(gt_params_list)
            
        gen_pitch, gen_yaw = self.inference_single(gen_image_path_list)
        gen_params = np.concatenate([gen_pitch, gen_yaw], 1)
        
        gaze_diff = ((gen_params - gt_params) ** 2).mean(-1) # n, 2
        
        return gaze_diff
    
    
    def detect_single(self, gen_dir):
        
        if not os.path.exists(gen_dir):
            return 
        gen_img_items = list(os.listdir(os.path.join(gen_dir, "images")))
        if len(gen_img_items) < 5:
            return None, None
        gen_img_items.sort()
        gen_image_path_list = [os.path.join(gen_dir, "images", img_item) for img_item in gen_img_items]
        gen_gaze_dir = os.path.join(gen_dir, "gaze_crop16")
        os.makedirs(gen_gaze_dir, exist_ok=True)
            
        gen_pitch, gen_yaw = self.inference_single(gen_image_path_list)
        gen_params = np.concatenate([gen_pitch, gen_yaw], 1)
        for i, img_item in enumerate(gen_img_items):
            gen_gaze_path = os.path.join(gen_dir, "gaze_crop16", img_item.split(".")[0] + ".npy")
            np.save(gen_gaze_path, gen_params[i])
        
        
        # return each images' result
        return gen_params
            

    def transform(self, image_path_list):
        images_np = []
        for img_path in image_path_list:
            img_np = cv2.imread(img_path)
            img_np = img_np[20:158, 60:164]
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = img_np / 255.
            
            images_np.append(img_np)
            
        images_np = np.stack(images_np, 0).transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(images_np).float()
        images_tensor = F.interpolate(images_tensor, [448, 448], mode='bilinear')
        images_tensor = (images_tensor - torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        
        return images_tensor
    