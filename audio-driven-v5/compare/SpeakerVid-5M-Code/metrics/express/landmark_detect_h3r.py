#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
# import dlib
import numpy as np 
# from imutils import face_utils
from lib.models.fd.fd import faceDetector
from lib.models.ldmk.ldmk import ldmkDetector
from lib.metric.face_detect_ldmk_pipeline import FaceLdmkDetector
import sys
sys.path.append("../../")

class LandmarkDetector(object):
    def __init__(self, fd_model_path="/data/pipline/human_centric_vw_model/benchmark/express/assets/Resnet50_Final.pth", fl_model_path="/data/pipline/human_centric_vw_model/benchmark/express/assets/h3r_model.pth"):
        self.net_fd = faceDetector(fd_model_path)
        self.net_ldmk = ldmkDetector(fl_model_path)
        
        self.fd_ldmk_detector = FaceLdmkDetector(self.net_fd, self.net_ldmk)
    
    def generate_landmark_single(self, img_path):
        image = cv2.imread(img_path)
        shape = self.fd_ldmk_detector.inference(image)
        self.fd_ldmk_detector.reset()
        
        if shape is None:
            return [None]
        
        return shape[0]
    
    def detect_single(self, image_dir, meta_dir):
        save_dir = os.path.join(meta_dir, "ldmks_98pt")
        os.makedirs(save_dir, exist_ok=True)
        
        if not os.path.exists(image_dir):
            return
        gen_img_items = [item for item in os.listdir(image_dir) if (item.endswith(".jpg") or item.endswith(".png"))]

        # if len(gen_img_items) == 0:
        #     print("ldmk h3r: num of items is 0!")
        #     return
        
        img_format = gen_img_items[0].split('.')[-1]
        
        gen_img_items.sort()
            
        for img_item in gen_img_items:
            img_path = os.path.join(image_dir, img_item)
            gen_ldmk_path = os.path.join(save_dir, img_item.replace(img_format, "npy"))
            if os.path.exists(gen_ldmk_path):
                continue
            ldmk_temp = self.generate_landmark_single(img_path)
            np.save(gen_ldmk_path, ldmk_temp)

    def get_lips_landmark(self, ldmk):

        mouth_land = ldmk[76:96].copy()
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
    
    
    def evaluate_single_pair_lip(self, gt_dir, gen_dir, gt_meta_dir, gen_meta_dir, start_num, max_compute_num, item_list=None):
        
        gt_ldmk_list = []
        gen_ldmk_list = []
        
        if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
            print("ldmk h3r: data dir not exists!")
            return None, None

        # gt_img_items = [item for item in os.listdir(gt_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        if item_list is None:
            gen_img_items = [item for item in os.listdir(gen_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
            gt_img_items_ori = [item for item in os.listdir(gt_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        else:
            gen_img_items = [item+".png" for item in item_list]
            gt_img_items_ori = [item+".png" for item in item_list]


        if len(gt_img_items_ori) == 0 or len(gen_img_items) == 0:
            print("ldmk h3r: num of items is 0!")
            return None, None

        if len(gt_img_items_ori[0].split('.')[0]) == 8:
            gt_img_prefix = "{:08d}" 
        elif len(gt_img_items_ori[0].split('.')[0]) == 7:
            gt_img_prefix = "{:07d}" 
        elif len(gt_img_items_ori[0].split('.')[0]) == 6:
            gt_img_prefix = "{:06d}" 
        elif len(gt_img_items_ori[0].split('.')[0]) == 5:
            gt_img_prefix = "{:05d}" 
        elif len(gt_img_items_ori[0].split('.')[0]) == 4:
            gt_img_prefix = "{:04d}" 
        
        gt_img_format = gt_img_items_ori[0].split('.')[-1]
        gen_img_format = gen_img_items[0].split('.')[-1]

        gt_img_shape = cv2.imread(os.path.join(gt_dir, gt_img_items_ori[0])).shape
        gen_img_shape = cv2.imread(os.path.join(gen_dir, gen_img_items[0])).shape

        gt_img_items = [gt_img_prefix.format(int(item.split(".")[0])) + '.{}'.format(gt_img_format) for item in gen_img_items]
        # gt_img_items = ["{:05d}" .format(int(item.split(".")[0])) + '.{}'.format(gt_img_format) for item in gen_img_items]


        gt_ldmk_dir = os.path.join(gt_meta_dir, "ldmks_98pt")
        os.makedirs(gt_ldmk_dir, exist_ok=True)

        gen_ldmk_dir = os.path.join(gen_meta_dir, "ldmks_98pt")
        os.makedirs(gen_ldmk_dir, exist_ok=True)
        
        gt_img_items.sort()
        gen_img_items.sort()
        
        for img_item in gt_img_items:
            gt_ldmk_path = os.path.join(gt_ldmk_dir, img_item.replace(gt_img_format, "npy"))

            if os.path.exists(gt_ldmk_path):
                try:
                    ldmk_temp = np.load(gt_ldmk_path, allow_pickle=True)
                except Exception as e:
                    print(e)
                    img_path = os.path.join(gt_dir, img_item)
                    ldmk_temp = self.generate_landmark_single(img_path)
                    np.save(gt_ldmk_path, ldmk_temp)
            else:
                img_path = os.path.join(gt_dir, img_item)
                ldmk_temp = self.generate_landmark_single(img_path)
                np.save(gt_ldmk_path, ldmk_temp)
            
            # if img_item.split(".")[0] + ".{}".format(img_format) in gen_img_items:
            gt_ldmk_list.append(ldmk_temp)
            
        for img_item in gen_img_items:
            gen_ldmk_path = os.path.join(gen_ldmk_dir, img_item.split(".")[0] + ".npy")
            
            if os.path.exists(gen_ldmk_path):
                try:
                    ldmk_temp = np.load(gen_ldmk_path, allow_pickle=True)
                except Exception as e:
                    img_path = os.path.join(gen_dir, img_item)
                    ldmk_temp = self.generate_landmark_single(img_path)
                    np.save(gen_ldmk_path, ldmk_temp)
            else:
                img_path = os.path.join(gen_dir, img_item)
                ldmk_temp = self.generate_landmark_single(img_path)
                np.save(gen_ldmk_path, ldmk_temp)

            gen_ldmk_list.append(ldmk_temp)
            
        ldmk_lip_dist_list = []
        ldmk_nme_list = []
        
        print("gen ldmk len", len(gen_ldmk_list))
        print("gt ldmk len", len(gt_ldmk_list))
        assert len(gen_ldmk_list) == len(gt_ldmk_list)

        for jdx in range(len(gen_ldmk_list[start_num:max_compute_num])):
            idx = jdx+start_num
            if gen_ldmk_list[idx][0] is None or gt_ldmk_list[idx][0] is None:
                continue
            gt_ldmk = gt_ldmk_list[idx] / np.array(gt_img_shape[:2]).reshape(1, 2) * 256
            gen_ldmk = gen_ldmk_list[idx] / np.array(gen_img_shape[:2]).reshape(1, 2) * 256
            
            if gt_ldmk[0] is None or gen_ldmk[0] is None:
                continue
            
            nme_temp = self.get_landmark_nme(gt_ldmk, gen_ldmk)
            ldmk_nme_list.append(nme_temp)
            
            mouth_error = self.get_lips_lmd_norm(gt_ldmk, gen_ldmk)
            ldmk_lip_dist_list.append(mouth_error)
        
        # return each image's result
        return ldmk_lip_dist_list, ldmk_nme_list
    

# model = LandmarkDetector()
# model.detect_single(image_dir="/data/pipline/human_centric_vw_model/benchmark/express/meta/face",meta_dir="/data/pipline/human_centric_vw_model/benchmark/express/meta")

            