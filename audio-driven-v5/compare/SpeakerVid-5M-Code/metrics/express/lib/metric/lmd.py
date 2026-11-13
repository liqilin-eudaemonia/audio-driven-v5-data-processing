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

def compare_landmarks(data_dir, save_dir):
    fake_path = os.path.join(data_dir + 'gen')
    real_path = os.path.join(data_dir + 'real')

    rps = []
    fps = []

    for video in os.listdir(fake_path):
        if video.endswith(".mp4") or video.endswith("_temp"):
            continue
        for frame in os.listdir( os.path.join(fake_path, video) ):
            fake_ldmk_path = os.path.join(fake_path, video, frame, "ldmk.npy")
            real_ldmk_path = os.path.join(real_path, video, frame, "ldmk.npy")
            if os.path.exists(fake_ldmk_path) and os.path.exists(real_ldmk_path):                
                rps.append(real_ldmk_path)
                fps.append(fake_ldmk_path)

    print (len(rps))
    print (len(fps))

    dis_txt = open(save_dir + 'landmark_distance.txt','w')
    distances = []
    
    for inx in range(len(rps)):
        
            rp = np.load(rps[inx])
            fp = np.load(fps[inx])
            
            dis = (rp-fp)**2
            dis = np.sum(dis,axis=1)
            dis = np.sqrt(dis)

            dis = np.mean(dis,axis=0)
            distances.append(dis)
            dis_txt.write(rps[inx] + ':' + str(dis) + '\n')
    dis_txt.close()

    average_distance = sum(distances) / len(rps)
    
    return average_distance