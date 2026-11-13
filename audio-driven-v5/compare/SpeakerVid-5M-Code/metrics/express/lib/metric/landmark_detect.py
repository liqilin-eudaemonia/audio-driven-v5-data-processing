#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import dlib
import numpy as np 
from imutils import face_utils
import sys
sys.path.append("../../")

class LandmarkDetector(object):
    def __init__(self, model_path="assets/shape_predictor_68_face_landmarks.dat", s_offset=False):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.s_offset = s_offset
        if self.s_offset:
            self.gt_image_dir = "images_crop16"
            self.gt_ldmk_dir = "ldmks_crop16"
        else:
            self.gt_image_dir = "images"
            self.gt_ldmk_dir = "ldmks"
        
    
    def generate_landmark_single(self, img_path):
        image = cv2.imread(img_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(image_gray, 1)
        if len(rects) == 0:
            return [None]
        # print(len(rects))
        
        shape = self.predictor(image_gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        
        return shape
    
    def get_lips_landmark(self, ldmk):
        for name, (i, j) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # print name
            if name != 'mouth':
                continue

            mouth_land = ldmk[i:j].copy()
            original = np.sum(mouth_land, axis=0) / 20.0
            
            mouth_land = mouth_land - original
            
        return mouth_land
    
    def get_landmark_nme(self, gt_ldmk, gen_ldmk):
        xmin = np.min(gt_ldmk[:, 0])
        xmax = np.max(gt_ldmk[:, 0])
        ymin = np.min(gt_ldmk[:, 1])
        ymax = np.max(gt_ldmk[:, 1])
        
        normalizer = np.sqrt((xmax - xmin) * (ymax - ymin))
        
        gt_ldmk = gt_ldmk - np.mean(gt_ldmk, axis=0)
        gen_ldmk = gen_ldmk - np.mean(gen_ldmk, axis=0)
        
        nme = np.mean(np.sqrt(np.sum((gt_ldmk-gen_ldmk)**2, axis=-1)), axis=-1) / normalizer
        return nme
    
    def get_landmark_nme_area(self, gt_ldmk, gen_ldmk):
        xmin = np.min(gt_ldmk[:, 0])
        xmax = np.max(gt_ldmk[:, 0])
        ymin = np.min(gt_ldmk[:, 1])
        ymax = np.max(gt_ldmk[:, 1])
        
        normalizer = np.sqrt((xmax - xmin) * (ymax - ymin))
        
        gt_ldmk = gt_ldmk - np.mean(gt_ldmk, axis=0)
        gen_ldmk = gen_ldmk - np.mean(gen_ldmk, axis=0)
        
        area_index = [
            [0, 17], # cheek
            [17, 27], # brown
            [27, 36], # nose
            [36, 48], # eye
            [48, 68], # mouth
            [0, 68], # face
        ]
        area_name = ['cheek', 'brown', 'nose', 'eye', 'mouth', 'face']
        
        nme_result = {'cheek':0, 'brown':0, 'nose':0, 'eye':0, 'mouth':0, 'face':0}
        
        for idx, index in enumerate(area_index):
            nme_result[area_name[idx]] = np.mean(np.sqrt(np.sum((gt_ldmk[index[0]:index[1]]-gen_ldmk[index[0]:index[1]])**2, axis=-1)), axis=-1) / normalizer
        
        # nme = np.mean(np.sqrt(np.sum((gt_ldmk-gen_ldmk)**2, axis=-1)), axis=-1) / normalizer
        return nme_result
    
    def get_ldmk_area(self, ldmk_list, img_size):
        area_list = []
        
        for ldmk in ldmk_list:
            mouth_ldmk = ldmk[48:68]
            mouth_area = np.zeros(img_size)
            mouth_ldmk = mouth_ldmk.astype(np.int32)
            cv2.fillConvexPoly(mouth_area, mouth_ldmk, color=(1))
            
            lip_area = np.sum(mouth_area[:, :, 0])
            lip_area_ratio = lip_area / (img_size[0] * img_size[1])
            
            area_list.append(lip_area_ratio)
        return area_list
    
    def get_d1(self, area_list):
        res = []
        
        for i in range(len(area_list) - 1):
            res.append(abs(area_list[i+1] - area_list[i]))
            
        return res
    
    def get_d2(self, area_list):
        res = []
        
        for i in range(len(area_list) - 2):
            res.append(abs(abs(area_list[i+2] - area_list[i+1]) - abs(area_list[i+1] - area_list[i])))
            
        return res
    
    def get_lips_jitter(self, ldmk_list, img_size):
        
        d1_res_list = []
        d2_res_list = []
        
        temp_ldmk_list = []
        for ldmk in ldmk_list:
            if ldmk[0] is None:
                if len(temp_ldmk_list) < 20:
                    temp_ldmk_list.clear()
                else:
                    area_list = self.get_ldmk_area(temp_ldmk_list, img_size)
                    temp_ldmk_list.clear()
                    d1_res = self.get_d1(area_list)
                    d2_res = self.get_d2(area_list)
                    d1_res_list.extend(d1_res)
                    d2_res_list.extend(d2_res)
            else:
                temp_ldmk_list.append(ldmk)
                continue
            
        if len(temp_ldmk_list) >= 20:
            area_list = self.get_ldmk_area(temp_ldmk_list, img_size)
            temp_ldmk_list.clear()
            d1_res = self.get_d1(area_list)
            d2_res = self.get_d2(area_list)
            d1_res_list.extend(d1_res)
            d2_res_list.extend(d2_res)
            
        if len(d1_res_list) == 0:
            return None, None, None, None
        return [np.mean(d1_res_list), np.std(d1_res_list), np.mean(d2_res_list), np.std(d2_res_list)]
    
    def get_lips_lmd_norm(self, gt_ldmk, gen_ldmk):
        xmin = np.min(gt_ldmk[:, 0])
        xmax = np.max(gt_ldmk[:, 0])
        ymin = np.min(gt_ldmk[:, 1])
        ymax = np.max(gt_ldmk[:, 1])
        
        normalizer = np.sqrt((xmax - xmin) * (ymax - ymin))
        
        gt_lip_ldmk = self.get_lips_landmark(gt_ldmk)
        gen_lip_ldmk = self.get_lips_landmark(gen_ldmk)
        
        mouth_error = np.mean(np.sqrt(np.sum((gt_lip_ldmk-gen_lip_ldmk)**2, axis=-1)), axis=-1) / normalizer
        
        return mouth_error
    
    def evaluate_single_pair_lip(self, gt_dir, gen_dir):
        
        gt_ldmk_list = []
        gen_ldmk_list = []
        
        if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
            return None, None
        gt_img_items = list(os.listdir(os.path.join(gt_dir, self.gt_image_dir)))
        gen_img_items = list(os.listdir(os.path.join(gen_dir, "images")))
        
        if len(gt_img_items) < 5 or len(gen_img_items) < 5:
            return None, None
        
        img_format = gen_img_items[0].split('.')[-1]
        
        gt_img_items.sort()
        gen_img_items.sort()
        leng = min(len(gen_img_items), len(gt_img_items))
        gen_img_items = gen_img_items[:leng]
        
        for img_item in gt_img_items:
            if os.path.exists(os.path.join(gt_dir, self.gt_ldmk_dir, img_item.split(".")[0] + ".npy")):
                ldmk_temp = np.load(os.path.join(gt_dir, self.gt_ldmk_dir, img_item.split(".")[0] + ".npy"), allow_pickle=True)
            else:
                img_path = os.path.join(gt_dir, self.gt_image_dir, img_item)
                gt_ldmk_dir = os.path.join(gt_dir, self.gt_ldmk_dir)
                os.makedirs(gt_ldmk_dir, exist_ok=True)
                gt_ldmk_path = os.path.join(gt_dir, self.gt_ldmk_dir, img_item.split(".")[0] + ".npy")
                ldmk_temp = self.generate_landmark_single(img_path)
                np.save(gt_ldmk_path, ldmk_temp)
            
            if img_item.split(".")[0].replace(self.gt_image_dir, "images") + ".{}".format(img_format) in gen_img_items:
                gt_ldmk_list.append(ldmk_temp)
            
        for img_item in gen_img_items:
            img_path = os.path.join(gen_dir, "images", img_item)
            gen_ldmk_dir = os.path.join(gen_dir, "ldmks")
            os.makedirs(gen_ldmk_dir, exist_ok=True)
            gen_ldmk_path = os.path.join(gen_dir, "ldmks", img_item.split(".")[0] + ".npy")
            ldmk_temp = self.generate_landmark_single(img_path)
            gen_ldmk_list.append(ldmk_temp)
            np.save(gen_ldmk_path, ldmk_temp)
            
        ldmk_lip_dist_list = []
        ldmk_nme_list = []
        
        print("gen ldmk len", len(gen_ldmk_list))
        print("gt ldmk len", len(gt_ldmk_list))
        for idx in range(len(gen_ldmk_list)):
            gt_ldmk = gt_ldmk_list[idx]
            gen_ldmk = gen_ldmk_list[idx]
            
            if gt_ldmk[0] is None or gen_ldmk[0] is None:
                continue
            
            nme_temp = self.get_landmark_nme(gt_ldmk, gen_ldmk)
            ldmk_nme_list.append(nme_temp)
            
            # gt_lip_ldmk = self.get_lips_landmark(gt_ldmk)
            # gen_lip_ldmk = self.get_lips_landmark(gen_ldmk)
            
            # mouth_error = np.mean(np.sqrt(np.sum((gt_lip_ldmk-gen_lip_ldmk)**2, axis=-1)), axis=-1)
            # ldmk_lip_dist_list.append(mouth_error)
            mouth_error = self.get_lips_lmd_norm(gt_ldmk, gen_ldmk)
            ldmk_lip_dist_list.append(mouth_error)
            
        area_results = self.get_lips_jitter(gen_ldmk_list, img_shape)
        # return each image's result
        return ldmk_lip_dist_list, ldmk_nme_list, area_results
            
            
def evaluate_jitter_lip_from_file(self, gen_dir):
            
        gen_ldmk_list = []
        
        if not os.path.exists(gen_dir):
            return None
        
        ldmk_dir = os.path.join(gen_dir, "ldmks_98pt")
        if not os.path.exists(ldmk_dir):
            return None
        gen_ldmk_path_list = [os.path.join(ldmk_dir, item) for item in os.listdir(ldmk_dir)]
        gen_ldmk_path_list.sort()
        
        try:
            img_shape = cv2.imread(gen_ldmk_path_list[0].replace("ldmks_98pt", "images").replace(".npy", ".jpg")).shape
        except Exception as e:
            try:
                img_shape = cv2.imread(gen_ldmk_path_list[0].replace("ldmks_98pt", "images").replace(".npy", ".png")).shape
            except Exception as e1:
                return None
            
        for gen_ldmk_path in gen_ldmk_path_list:
            ldmk = np.load(gen_ldmk_path, allow_pickle=True)
            gen_ldmk_list.append(ldmk)
            
        area_results = self.get_lips_jitter(gen_ldmk_list, img_shape)
        
        # return each image's result
        return area_results