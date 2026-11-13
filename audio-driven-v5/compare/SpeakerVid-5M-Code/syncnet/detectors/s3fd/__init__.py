import time
import numpy as np
import cv2
import torch
import sys
import torch.nn.functional as F

from torchvision import transforms
from syncnet.detectors.s3fd.nets import S3FDNet
from syncnet.detectors.s3fd.box_utils import nms_

img_mean = np.array([104., 117., 123.])[np.newaxis, :, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, s3fd_model_file, device):

        tstamp = time.time()
        self.device = device

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(s3fd_model_file, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))

        self.img_size = (384, 384)
        self.batch_size = 32
    
    def detect_faces(self, images_all, conf_th=0.8):

        w, h = images_all.shape[2], images_all.shape[1]

        boxes_list = []
        with torch.no_grad():

            # for i in range(int(round(images_all.shape[0] / self.batch_size + 0.5))):
            for i in range(0, len(images_all), self.batch_size):
                images = images_all[i:i+self.batch_size].copy()
                images = images.transpose(0, 3, 1, 2)
                images = images[:, [2, 1, 0], :, :]
                images = images.astype('float32')
                images -= img_mean
                images = images[:, [2, 1, 0], :, :]
                x = torch.from_numpy(images).to(self.device)
                x = F.interpolate(x, self.img_size)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for b in range(images.shape[0]):
                    bboxes = np.empty(shape=(0, 5))
                    for i in range(detections.size(1)):
                        j = 0
                        while detections[b, i, j, 0] > conf_th:
                            score = detections[b, i, j, 0]
                            pt = (detections[b, i, j, 1:] * scale).cpu().numpy()
                            bbox = (pt[0], pt[1], pt[2], pt[3], score)
                            bboxes = np.vstack((bboxes, bbox))
                            j += 1

                    keep = nms_(bboxes, 0.1)
                    bboxes = bboxes[keep]
                    boxes_list.append(bboxes)
                
                del images

        return boxes_list