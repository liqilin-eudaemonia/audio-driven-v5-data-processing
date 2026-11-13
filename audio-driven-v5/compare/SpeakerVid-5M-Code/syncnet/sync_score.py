
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
from syncnet.SyncNetModel import *
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

    def __init__(self, model_path, device, num_layers_in_fc_layers = 1024, av_offset = 2):
        super(SyncNetInstance, self).__init__()

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).to(device)
        model_dict = torch.load(model_path)
        self.__S__.load_state_dict(model_dict)
        self.__S__.eval()
        self.audio_visual_offset_in_data = av_offset
        self.device = device

    def evaluate_single(self, aligned_img_list, audio_data, sample_rate=16000):

        # ========== ==========
        # Load video 
        # ========== ==========

        im = numpy.stack(aligned_img_list, axis = 3)
        im = numpy.expand_dims(im, axis = 0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        mfcc = zip(*python_speech_features.mfcc(audio_data, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis = 0), axis = 0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        # cct = cct[:, :, :, self.audio_visual_offset_in_data*4:-self.audio_visual_offset_in_data*4] # because generated images is 4 frames less than audio, 2 in head and 2 in tail

        # ========== ==========
        if cct.shape[-1] / im.shape[2] >= 12:
            cct = cct[...,::3]
        min_length = min(len(aligned_img_list), math.floor(cct.shape[3] / 4))
        # print(min_length, len(aligned_img_list), math.floor(cct.shape[3] / 4), mfcc.shape, cc.shape, cct.shape)
        
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
            im_out  = self.__S__.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:, :, :, vframe*4:vframe*4 + 20] for vframe in range(i, min(lastframe, i + batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.to(self.device))
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

        return float(offset.numpy()), float(conf.numpy()), float(minval.numpy())

