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
import skimage.transform as transform
import torch.nn.functional as F
import sys
sys.path.append("../../")

from lib.metric.webface import ResnetModel

class IdentityMetric(object):
    def __init__(self, model_path="assets/webface_r50.pth", s_offset=False, h3r=False):
        self.model = ResnetModel()
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.cuda()
        self.model.eval()
        
        self.SRC = np.array(
            [
                [38.2946, 51.6963],
                [63.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
        
        self.st = transform.SimilarityTransform()
        
        self.batch_size = 8
        
        self.s_offset = s_offset
        self.h3r = h3r
        if self.s_offset:
            if self.h3r:
                self.gt_image_dir = "images_crop16"
                self.gt_ldmk_dir = "ldmks_98pt_crop16_2"
                self.stand_index = np.array([60, 64, 68, 72, 54, 76, 82])
            else:
                self.gt_image_dir = "images_crop16"
                self.gt_ldmk_dir = "ldmks_crop16"
                self.stand_index = np.array([36, 39, 42, 45, 30, 48, 54])
        else:
            if self.h3r:
                self.gt_image_dir = "images"
                self.gt_ldmk_dir = "ldmks_98pt_2"
                self.stand_index = np.array([60, 64, 68, 72, 54, 76, 82])
            else:
                self.gt_image_dir = "images"
                self.gt_ldmk_dir = "ldmks"
                self.stand_index = np.array([36, 39, 42, 45, 30, 48, 54])
        
    def inference(self, images_tensor):
        results = []
        
        b, c, h, w = images_tensor.shape
        
        images_tensor = images_tensor.cuda()
        for i in range(round(b / self.batch_size + 0.49999)):
            image_clip = images_tensor[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            res_temp = self.model(image_clip)
            
            results.append(res_temp)
            
        id_embeddings = torch.cat(results, 0).detach().cpu().numpy()
        
        return id_embeddings
    
    def inference_single(self, images_dir, ldmks_dir, gt = False):
        if not os.path.exists(images_dir) or not os.path.exists(ldmks_dir):
            return None
        image_items = list(os.listdir(images_dir))
        ldmk_items = list(os.listdir(ldmks_dir))
        if len(ldmk_items) < 5 or len(image_items) < 5:
            return None
        image_items.sort()
        img_format = image_items[0].split('.')[-1]
        # image_path_list = [os.path.join(images_dir, img_item) for img_item in image_items]
        
        ldmk_items.sort()
        ldmk_path_list = [os.path.join(ldmks_dir, ldmk_item) for ldmk_item in ldmk_items]
        
        image_path_list = [os.path.join(images_dir, ldmk_item.split('.')[0] + ".{}".format(img_format)) for ldmk_item in ldmk_items]
        
        if gt:
            for i in range(len(image_path_list)):
                images_tensor = self.transform(image_path_list[i:i+1], ldmk_path_list[i:i+1])
                if images_tensor is None:
                    continue
                else:
                    break
        else:
            images_tensor = self.transform(image_path_list, ldmk_path_list)
        
        if images_tensor is None:
            return None
        
        with torch.no_grad():
            id_embeddings = self.inference(images_tensor)
        
        return id_embeddings
    
    def evaluate_single_pair(self, gt_dir, gen_dir):
        if os.path.exists(os.path.join(gt_dir, "identity.npy")):
            gt_id_embeddings = np.load(os.path.join(gt_dir, "identity.npy"))
        else:
            gt_id_embeddings = self.inference_single(os.path.join(gt_dir, self.gt_image_dir), os.path.join(gt_dir, self.gt_ldmk_dir), gt = True)
            if gt_id_embeddings is None:
                return np.array([])
            np.save(os.path.join(gt_dir, "identity.npy"), gt_id_embeddings)
        
        if self.h3r:   
            gen_id_embeddings = self.inference_single(os.path.join(gen_dir, "images"), os.path.join(gen_dir, "ldmks_98pt"))
        else:
            gen_id_embeddings = self.inference_single(os.path.join(gen_dir, "images"), os.path.join(gen_dir, "ldmks"))
            
        if gen_id_embeddings is None:
            return np.array([])
        
        id_sim = np.diag(np.dot(gen_id_embeddings, gt_id_embeddings.T) / (np.linalg.norm(gen_id_embeddings, axis=1)[None, :] * np.linalg.norm(gt_id_embeddings, axis=1)[:, None]))
        
        # return each image's result
        return id_sim
            

    def transform(self, image_path_list, ldmk_path_list):
        images_np = []
        landmarks = []
        for idx, ldmk_path in enumerate(ldmk_path_list):
            ldmk = np.load(ldmk_path_list[idx], allow_pickle=True)
            landmarks.append(ldmk)
            
        if landmarks[-1][0] is None:
            return None

        for idx in range(len(landmarks)-1, 0, -1):
            if landmarks[idx-1][0] is None:
                landmarks[idx-1] = landmarks[idx]
            
        for idx, img_path in enumerate(image_path_list):
            img_np = cv2.imread(img_path)
            ldmk = landmarks[idx]
            
            img_np = self.recog_crop(img_np, ldmk)
            
            images_np.append(img_np)
            
        images_np = np.concatenate(images_np, 0)
        images_tensor = torch.from_numpy(images_np).float()
        
        return images_tensor
    
    
    def recog_crop(self, img, ldmk):
        
        ldmk_7p = ldmk[self.stand_index]
        ldmk_5p = np.concatenate([ldmk_7p[:2].mean(0)[None], ldmk_7p[2:4].mean(0)[None], ldmk_7p[4:]], 0)
        self.st.estimate(ldmk_5p, self.SRC)
        img = cv2.warpAffine(img, self.st.params[0:2, :], (112, 112), borderValue=0.0)
        img = np.expand_dims(img, 0)
        img = (np.transpose(img[:, :, :, ::-1], (0, 3, 1, 2)) - 127.5) / 127.5
            
        return img