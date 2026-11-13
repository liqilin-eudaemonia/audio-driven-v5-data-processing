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


class ExpENet(object):
    def __init__(self, model_path="assets/enet_b0_8_best_afew.pt", s_offset=False):
        self.exp_model = torch.load(model_path)
        
        self.exp_model.eval()
        self.exp_model.cuda()
        
        self.batch_size = 8
        
        self.s_offset = s_offset
        if self.s_offset:
            self.gt_image_dir = "images_crop16"
            self.gt_exp_dir = "exps_crop16"
        else:
            self.gt_image_dir = "images"
            self.gt_exp_dir = "exps"
            
    def model_inference(self, images_tensor):
        x = self.exp_model.conv_stem(images_tensor)
        x = self.exp_model.bn1(x)
        x = self.exp_model.act1(x)
        x = self.exp_model.blocks(x)
        x = self.exp_model.conv_head(x)
        x = self.exp_model.bn2(x)
        x = self.exp_model.act2(x)
        x = self.exp_model.global_pool(x)
        
        return x
    
    # def model_inference(self, images_tensor):
    #     x = self.exp_model(images_tensor)
        
    #     score = torch.argmax(x, -1)
        
    #     cls_label = F.one_hot(score, num_classes=x.shape[-1])
        
    #     return cls_label
        
    def inference(self, images_tensor):
        results = []
        
        b, c, h, w = images_tensor.shape
        
        images_tensor = images_tensor.cuda()
        for i in range(round(b / self.batch_size + 0.49999)):
            image_clip = images_tensor[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            res_temp = self.model_inference(image_clip)
            
            results.append(res_temp)
            
        exp_features = torch.cat(results, 0).detach().cpu().numpy()
        
        
        return exp_features
    
    def inference_single(self, image_path_list):
        images_tensor = self.transform(image_path_list)
        
        with torch.no_grad():
        
            exp_features = self.inference(images_tensor)
        
        return exp_features
    
    def detect_single(self, gen_dir):
        
        if not os.path.exists(gen_dir):
            return 
        gen_img_items = list(os.listdir(os.path.join(gen_dir, "images")))
        if len(gen_img_items) < 5:
            return None, None
        gen_img_items.sort()
        gen_image_path_list = [os.path.join(gen_dir, "images", img_item) for img_item in gen_img_items]
        gen_exp_dir = os.path.join(gen_dir, "exps_crop16")
        os.makedirs(gen_exp_dir, exist_ok=True)
            
        exp_features = self.inference_single(gen_image_path_list)
        # gen_exp_path = os.path.join(gen_dir, "exps_crop16.npy")
        gen_exp_path = os.path.join(gen_dir, "exps_feat_crop16.npy")
        np.save(gen_exp_path, exp_features)
        
        return 
            

    def transform(self, image_path_list):
        images_np = []
        for img_path in image_path_list:
            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = img_np / 255.
            
            img_np = img_np[32:196, 38:186]
            images_np.append(img_np)
            
        images_np = np.stack(images_np, 0).transpose(0, 3, 1, 2)
        images_tensor = torch.from_numpy(images_np).float()
        images_tensor = F.interpolate(images_tensor, [224, 224], mode='bilinear')
        images_tensor = (images_tensor - torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        
        
        return images_tensor
    