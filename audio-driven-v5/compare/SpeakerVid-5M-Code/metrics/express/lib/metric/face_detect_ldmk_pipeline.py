#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

##### python internal and external package
import os
import cv2
import time
import math
import torch
import torch.nn as nn
import numpy as np


class FaceLdmkDetector(object):
    def __init__(self, facedetector, ldmkdetector):
        self.facedetector = facedetector
        self.ldmkdetector = ldmkdetector
        
        self.last_frame = None
        self.frame_count = 0
        self.frame_margin = 15
        
    def reset(self):
        self.last_frame = None
        self.frame_count = 0
    
    def get_box_from_ldmk(self, ldmks):
        boxes_return = []
        for ldmk in ldmks:
            xmin = np.min(ldmk[:, 0])
            xmax = np.max(ldmk[:, 0])
            ymin = np.min(ldmk[:, 1])
            ymax = np.max(ldmk[:, 1])
            
            boxes_return.append([xmin, ymin, xmax, ymax])
            
        return np.array(boxes_return)
    
    def extend_box(self, boxes, ratio=1.5):
        boxes_return = []
        for box in boxes:
            xmin = box[0]; ymin = box[1]
            xmax = box[2]; ymax = box[3]
            
            center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            h = (ymax - ymin + 1)
            w = (xmax - xmin + 1)
            
            size = (h + w) / 2
            
            extend_size = size * ratio
            
            xmine = center[0] - extend_size / 2
            xmaxe = center[0] + extend_size / 2
            ymine = center[1] - extend_size / 2
            ymaxe = center[1] + extend_size / 2
            
            boxes_return.append([xmine, ymine, xmaxe, ymaxe])
            
        return np.array(boxes_return)
    
    def ldmk_detect(self, img, boxes):
        ldmks = []
        h, w, c = img.shape
        for box in boxes:
            xmin = int(box[0]); ymin = int(box[1])
            xmax = int(box[2]); ymax = int(box[3])
            
            img_padding = cv2.copyMakeBorder(img, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=(0,0,0))
            
            # img_crop = img[
            #     max(ymin, 0):min(ymax, h), 
            #     max(xmin, 0):min(xmax, w)
            # ]
            
            xmin_ = xmin + 150
            ymin_ = ymin + 150
            xmax_ = xmax + 150
            ymax_ = ymax + 150
            img_crop = img_padding[
                ymin_:ymax_, 
                xmin_:xmax_
            ]
            
            ldmk = self.ldmkdetector(img_crop)
            ldmk = ldmk[0] + np.array([xmin, ymin]).reshape(1, 2)
            ldmks.append(ldmk)
            
        return np.array(ldmks)
            
    
    def inference(self, img):
        
        if self.last_frame is None and self.frame_count % self.frame_margin == 0:
            boxes_scores = self.facedetector(img)
            boxes = boxes_scores[:, :4]
            boxes_extend = self.extend_box(boxes, ratio=1.5)
            
            self.frame_count += 1
            
            if len(boxes_scores) == 0:
                return None
            
        else:
            boxes = self.get_box_from_ldmk(self.last_frame)
            boxes_extend = self.extend_box(boxes, ratio=1.8)
            
            self.frame_count += 1
        
        try:
            ldmks = self.ldmk_detect(img, boxes_extend)
        except:
            return None
        
        self.last_frame = ldmks
        
        return ldmks
        
        
    
    