#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import numpy as np
import face_alignment
import skimage.transform as trans
import argparse
import torch
from scipy import signal
from tqdm import tqdm

from lib.detectors import S3FD

class PreProcessor(object):
    def __init__(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.crop_scale = 0.4
        
        self.det = S3FD(device)
        print(device)

    def align_folder(self, folder_path, folder_save_path):
        
        print("begin detect face")
        dets_res = {'frame': [], 'bbox': [], 'conf': [], 'img_path': []}
        images_list = list(os.listdir(folder_path))
        images_list.sort()
        for fidx, img_item in tqdm(enumerate(images_list)):
            img_path = os.path.join(folder_path, img_item)
            image = cv2.imread(img_path)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.det.detect_faces(image_np, conf_th=0.9, scales=[0.25])
            if len(bboxes) == 0:
                if fidx == 0:
                    dets_res['frame'].append(fidx)
                    dets_res['bbox'].append(None)
                    dets_res['conf'].append(None)
                    dets_res['img_path'].append(img_path)
                else:
                    dets_res['frame'].append(fidx)
                    dets_res['bbox'].append(dets_res['bbox'][-1])
                    dets_res['conf'].append(dets_res['conf'][-1])
                    dets_res['img_path'].append(img_path)
            else:
                dets_res['frame'].append(fidx)
                dets_res['bbox'].append((bboxes[0, :-1]).tolist())
                dets_res['conf'].append(bboxes[0, -1])
                dets_res['img_path'].append(img_path)
            # dets_res.append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1], 'img_path': img_path})
        
        if dets_res['bbox'][-1] is None:
            return

        for idx in range(len(dets_res['bbox'])-1, 0, -1):
            if dets_res['bbox'][idx-1] is None:
                dets_res['bbox'][idx-1] = dets_res['bbox'][idx]
                dets_res['conf'][idx-1] = dets_res['conf'][idx]
        
        print("all face detected!")

        dets = {'x': [], 'y': [], 's': []}

        for det in dets_res['bbox']:
            dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y

        # Smooth detections
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

        for i, img_pth in enumerate(dets_res['img_path']):
            cs = self.crop_scale
            bs = dets['s'][i]
            bsi = int(bs * (1 + 2 * cs))

            img = cv2.imread(img_pth)
            frame = np.pad(img,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
            my  = dets['y'][i]+bsi  # BBox center Y
            mx  = dets['x'][i]+bsi  # BBox center X

            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            face = cv2.resize(face, (224, 224))
            img_save_path = os.path.join(folder_save_path, img_pth.split('/')[-1])
            cv2.imwrite(img_save_path, face)
        print('cropped files saved at {}'.format(folder_save_path))

    def split(self, video_path, data_path, data_dir):
        wav_path = os.path.join(data_dir, "audio.wav")
        
        if not os.path.exists(wav_path):
            cmd_wav = "ffmpeg -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}".format(video_path, wav_path)
            os.system(cmd_wav)
        
        if not len(os.listdir(data_path)) > 0:
            cmd_img = "ffmpeg -y -i {} -r 25 -f image2 {}/%4d.jpg".format(video_path, data_path)
            os.system(cmd_img)



    ### pid-mp4id-images version
    def process_video(self, video_path, data_dir):
        
        data_path = os.path.join(data_dir, "images")
        save_path = os.path.join(data_dir, "aligned_syncnet_crop_images_official")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        
        self.split(video_path, data_path, data_dir)

        if len(list(os.listdir(data_path))) == len(list(os.listdir(save_path))):
            return

        self.align_folder(data_path, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='the ori video path')
    parser.add_argument('--folder_path', help='the folder which needs processing')
    args = parser.parse_args()

    video_path= args.video_path
    data_dir= args.folder_path
    
    preprocessor = PreProcessor()
    preprocessor.process_video(video_path, data_dir)