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
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

##### self defined package
from lib.models.ldmk.hrnet import LandmarkDetector

class ldmkDetector(nn.Module):
    def __init__(self, model_path):
        super(ldmkDetector, self).__init__()
        self.model = LandmarkDetector(model_path)

        self.model.eval()
        self.model.cuda()
        
        self.size = 256 # 256
        
        self.landmark_transform = transforms.Compose([
            transforms.Resize(size=(self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _transform(self, img):
        
        h, w, c = img.shape
        img = img[:, :, ::-1]
        # print(img.shape)
        img = Image.fromarray(img.astype(np.uint8))
        img = self.landmark_transform(img)
        img = img.type(torch.FloatTensor).unsqueeze(0)
        
        
        img = img.cuda()
        
        return img, h, w
        
    def forward(self, img):
        img, h, w  = self._transform(img)
        
        _, landmarks = self.model(img)
        
        landmarks = landmarks / torch.Tensor([self.size / w, self.size / h]).reshape(1, 1, 2).cuda()
        
        landmarks = landmarks.detach().cpu().numpy()
        
        return landmarks