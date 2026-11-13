
import os
import cv2
import numpy as np
from scipy import signal
from tqdm import tqdm

from syncnet.detectors import S3FD

class PreProcessor(object):
    def __init__(self, s3fd_model_file, device):

        self.crop_scale = 0.4
        
        self.det = S3FD(s3fd_model_file, device)

    def align(self, video_ndarry):
        bboxes_list = self.det.detect_faces(video_ndarry, conf_th=0.7)
        dets_res = {'frame': [], 'bbox': [], 'conf': []}
        for fidx, bboxes in tqdm(enumerate(bboxes_list)):
            if len(bboxes) == 0:
                if fidx == 0:
                    dets_res['frame'].append(fidx)
                    dets_res['bbox'].append(None)
                    dets_res['conf'].append(None)
                else:
                    dets_res['frame'].append(fidx)
                    dets_res['bbox'].append(dets_res['bbox'][-1])
                    dets_res['conf'].append(dets_res['conf'][-1])
            else:
                dets_res['frame'].append(fidx)
                dets_res['bbox'].append((bboxes[0, :-1]).tolist())
                dets_res['conf'].append(bboxes[0, -1])
        
        if dets_res['bbox'][-1] is None:
            return None

        for idx in range(len(dets_res['bbox'])-1, 0, -1):
            if dets_res['bbox'][idx-1] is None:
                dets_res['bbox'][idx-1] = dets_res['bbox'][idx]
                dets_res['conf'][idx-1] = dets_res['conf'][idx]
        
        dets = {'x': [], 'y': [], 's': []}

        for det in dets_res['bbox']:
            dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y

        # Smooth detections
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

        res_img_list = []
        for i, img in enumerate(video_ndarry):
            cs = self.crop_scale
            bs = dets['s'][i]
            bsi = int(bs * (1 + 2 * cs))

            frame = np.pad(img,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
            my  = dets['y'][i]+bsi  # BBox center Y
            mx  = dets['x'][i]+bsi  # BBox center X

            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            face = cv2.resize(face, (224, 224))
            res_img_list.append(face)

        return res_img_list
