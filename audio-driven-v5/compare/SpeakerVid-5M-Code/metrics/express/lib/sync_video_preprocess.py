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

class PreProcessor(object):
    def __init__(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.crop_scale = 0.4
        
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        print(device)

    def align_folder(self, folder_path, folder_save_path):
        
        print("begin detect ldmk")
        preds = self.fa.get_landmarks_from_directory(folder_path, return_bboxes=True)
        print("all ldmks detected!")

        dets = {'x': [], 'y': [], 's': []}

        for idx, img in enumerate(preds.keys()):
            pred_points, _, pred_bboxes_socres = preds[img]
            pred_points = np.array(pred_points)
            if pred_points is None or len(pred_points.shape) != 3:
                print('preprocessing failed 1')
                if idx == 0:
                    dets['s'].append(None)
                    dets['y'].append(None)
                    dets['x'].append(None)
                else:
                    dets['s'].append(dets['s'][-1])
                    dets['y'].append(dets['y'][-1])
                    dets['x'].append(dets['x'][-1])
                
            else:
                pred_bboxes = np.array(pred_bboxes_socres)[:, :4]
                dets['s'].append(max((pred_bboxes[0][3] - pred_bboxes[0][1]), (pred_bboxes[0][2] - pred_bboxes[0][0])) / 2)
                dets['y'].append((pred_bboxes[0][1] + pred_bboxes[0][3]) / 2)
                dets['x'].append((pred_bboxes[0][0] + pred_bboxes[0][2]) / 2)

        if dets['s'][-1] is None:
            return

        for idx in range(len(dets['s'])-1, 0, -1):
            if dets['s'][idx-1] is None:
                dets['s'][idx-1] = dets['s'][idx]
                dets['x'][idx-1] = dets['x'][idx]
                dets['ymkd'][idx-1] = dets['y'][idx]

        # Smooth detections
        dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

        for i, img_pth in enumerate(preds.keys()):
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
            cmd_img = "ffmpeg -y -i {} -r 25 -f image2 {}/%6d.png".format(video_path, data_path)
            os.system(cmd_img)



    ### pid-mp4id-images version
    def process_video(self, video_path, data_dir):
        
        data_path = os.path.join(data_dir, "images")
        save_path = os.path.join(data_dir, "aligned_syncnet_crop_images")
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