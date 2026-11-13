#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# Revised by wangduomin@xiaobing.ai             #
#################################################

#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import os
import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from lib.metric.SyncNetModel import *
from shutil import rmtree
import numpy as np

global vshift, batch_size
vshift = 15
batch_size = 20

# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift = 10):
    
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i: i + win_size, :]))

    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, model_path="/mnt/wangduomin/workspace/talking_head_metric/assets/syncnet_v2.model", num_layers_in_fc_layers = 1024, av_offset = 2):
        super(SyncNetInstance, self).__init__()

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda()
        model_dict = torch.load(model_path)
        self.__S__.load_state_dict(model_dict)
        self.__S__.eval()
        self.audio_visual_offset_in_data = av_offset

    def evaluate_single(self, video_dir, audio_path):

        # ========== ==========
        # Load video 
        # ========== ==========

        images = []

        for frame in os.listdir(video_dir):
            img_path = os.path.join(video_dir, frame, "img.jpg")
            # img_path = img_path.replace("gen", "real")
            img_input = cv2.imread(img_path)
            images.append(img_input)

        im = numpy.stack(images, axis = 3)
        im = numpy.expand_dims(im, axis = 0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        wav_path = audio_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            command = "ffmpeg -y -i {} -ac 1 -vn -acodec pcm_s16le -ar 16000 {}".format(audio_path, wav_path)
            os.system(command)
        sample_rate, audio = wavfile.read(wav_path)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis = 0), axis = 0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        cct = cct[:, :, :, self.audio_visual_offset_in_data*4:-self.audio_visual_offset_in_data*4] # because generated images is 4 frames less than audio, 2 in head and 2 in tail


        min_length = min(len(images), math.floor(cct.shape[3] / 4))
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, batch_size):
            
            im_batch = [ imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size)) ]
            im_in = torch.cat(im_batch, 0)
            im_out  = self.__S__.forward_lip(im_in.cuda())
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:, :, :, vframe*4:vframe*4 + 20] for vframe in range(i, min(lastframe, i + batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        dists = calc_pdist(im_feat, cc_feat, vshift = vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)

        offset = vshift - minidx
        conf   = torch.median(mdist) - minval

        return offset.numpy(), conf.numpy(), minval.numpy()

    def evalulate(self, data_dir):

        self.__S__.eval()
        fake_path = os.path.join(data_dir, 'gen')
        real_path = os.path.join(data_dir, 'real')

        avoff_list = 0
        lsed_list = 0
        lsec_list = 0

        count = 0
        for video in os.listdir(fake_path):
            if video.endswith(".mp4") or video.endswith("_temp"):
                continue
            audio_path = os.path.join(real_path, video, "audio.mp3")
            video_dir = os.path.join(fake_path, video)
            offset, conf, minval = self.evaluate_single(video_dir, audio_path)
            avoff_list += offset
            lsec_list += conf
            lsed_list += minval
            count += 1

        avoff = avoff_list / count
        lsed = lsed_list / count
        lsec = lsec_list / count

        return avoff, lsed, lsec
