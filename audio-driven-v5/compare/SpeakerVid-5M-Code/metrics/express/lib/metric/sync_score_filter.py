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
import json

from scipy import signal
from scipy.io import wavfile
from lib.metric.SyncNetModel import *
from shutil import rmtree
import numpy as np
from tqdm import tqdm
import sys

from lib.sync_video_preprocess_official import PreProcessor

sys.path.append("../../")
global vshift, batch_size, bin_num#, sample_rate
vshift = 15
batch_size = 20
bin_num = 4
# sample_rate = 16000

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

    def __init__(self, model_path="assets/syncnet_v2.model", num_layers_in_fc_layers = 1024, av_offset = 2):
        super(SyncNetInstance, self).__init__()

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda()
        model_dict = torch.load(model_path)
        self.__S__.load_state_dict(model_dict)
        self.__S__.eval()
        self.audio_visual_offset_in_data = av_offset

        self.preprocessor = PreProcessor()

    def parsing_name(self, name):
        
        sp_item = name.split('.')
        img_type = sp_item[1]
        num_count = 0
        for i in sp_item[0]:
            num_count += 1

        # return_str = "{:0" + "{}d".format(num_count) + "}.{}".format(img_type)
        return_str = "{:0{}d}.{}".format(int(sp_item[0]), num_count, img_type)

        return return_str, int(sp_item[0])

    def evaluate_single(self, video_dir, audio_path):
        """ compute sync metric for a single video clip

        Args:
            video_dir (str): a single video clip's images dir
            audio_path (str): the corresponding audio in wav format follow the offifial extract parameters

        Returns:
            offset: the offset 
            conf: the confidence
            minval: minimal distance between audio and video frames in window
        """

        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        items_list = os.listdir(video_dir)
        # print(video_dir, items_list)
        items_list.sort()
        name, fn = self.parsing_name(items_list[0])
        last_frame = fn - 1
        for idx, frame in enumerate(items_list):
            # frame = format_str.format(idx)
            _, fn = self.parsing_name(frame)
            # print(last_frame, fn)
            if last_frame + 1 != fn:
                break
            last_frame = fn
            img_path = os.path.join(video_dir, frame)
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
        
        # if os.path.exists(wav_path):
        #     os.remove(wav_path)

        if not os.path.exists(wav_path):
            # if audio_path.endswith(".wav"):
            #     audio_path = audio_path.replace(".wav", ".mp3")
            # video_path = 
            # command = "ffmpeg -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}".format(video_path, wav_path)
            # os.system(command)
            return 0, 0, 0
            
        sample_rate, audio = wavfile.read(wav_path)
        mfcc = python_speech_features.mfcc(audio, sample_rate) #, winlen=0.05, winstep=0.02)
        mfcc = zip(*mfcc)
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        # print(mfcc.shape)

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis = 0), axis = 0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        # print(imtv.shape, cct.shape, len(audio), sample_rate)
        # print(cct.shape)
        if self.audio_visual_offset_in_data > 0:
            cct = cct[:, :, :, self.audio_visual_offset_in_data*bin_num:-self.audio_visual_offset_in_data*bin_num] # because generated images is 4 frames less than audio, 2 in head and 2 in tail
        # print(cct.shape)
        # print(len(images), len(audio), len(audio) // 640)
        min_length = min(len(images), math.floor(len(audio) / 640))
        # print(len(images), math.floor(cct.shape[3] / bin_num))
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

            cc_batch = [ cct[:, :, :, vframe*bin_num:vframe*bin_num + 5 * bin_num] for vframe in range(i, min(lastframe, i + batch_size)) ]
            # print(i, lastframe, batch_size)
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
    
    def evaluate_gen_single(self, gt_dir, gen_dir):
        images_dir = os.path.join(gen_dir, "aligned_syncnet_crop_images_official")
        ori_images_dir = os.path.join(gen_dir, "images")
        
        if len(list(os.listdir(ori_images_dir))) < 5:
            return None, None, None
        if not os.path.exists(images_dir) or not len(os.listdir(ori_images_dir)) == len(os.listdir(images_dir)):
            
            os.makedirs(images_dir, exist_ok=True)
            self.preprocessor.align_folder(ori_images_dir, images_dir)
        
        if len(list(os.listdir(images_dir))) < 5:
            return None, None, None
        
        audio_path = os.path.join(gt_dir, "audio.wav")
        offset, conf, minval = self.evaluate_single(images_dir, audio_path)
        
        return offset, minval, conf
    
    def evaluate_gt_single(self, gt_dir):
        images_dir = os.path.join(gt_dir, "aligned_syncnet_crop_images_official")
        audio_path = os.path.join(gt_dir, "audio.wav")
        offset, conf, minval = self.evaluate_single(images_dir, audio_path)
        
        res_dict = {
            'avoff': offset,
            'lsed': minval,
            'lsec': conf
        }

        with open(os.path.join(gt_dir, "sync_res.json"), 'w') as f:
            json.dump(res_dict, f, indent=4)
        
        return offset, minval, conf

    def evalulate(self, gt_data_dir, gen_data_dir, gt):
        avoff_list = 0
        lsed_list = 0
        lsec_list = 0

        count = 0
        if gt_data_dir == gen_data_dir:
            for video in os.listdir(gen_data_dir):
                # if not "2022_09_07" in video:
                #     continue
                # if not ("IMG_9083" in video or "IMG_9081" in video):
                #     continue
                audio_path = os.path.join(gt_data_dir, video, "audio.wav")
                sync_path = os.path.join(gen_data_dir, video, "sync_dict_res.json")
                if gt:
                    video_dir = os.path.join(gt_data_dir, video, "aligned_syncnet_crop_images_official")
                    # video_dir = os.path.join(gt_data_dir, video, "images")
                else:
                    video_dir = os.path.join(gen_data_dir, video, "aligned_syncnet_crop_images_official")
                if len(list(os.listdir(video_dir))) < 5 or (not os.path.exists(audio_path) and not os.path.exists(audio_path.replace(".mp3", ".wav"))):
                    continue
                offset, conf, minval = self.evaluate_single(video_dir, audio_path)
                print(offset, conf, minval)
                res_dict = {
                    'avoff': float(offset),
                    'lsed': float(minval),
                    'lsec': float(conf)
                }

                with open(sync_path, 'w') as f:
                    json.dump(res_dict, f, indent=4)
                avoff_list += offset
                lsec_list += conf
                lsed_list += minval
                count += 1
        else:
            for pid in tqdm(os.listdir(gen_data_dir)):
                # if pid.endswith(".json") or not "id00061" in pid:
                #     continue
                
                pid_dir = os.path.join(gen_data_dir, pid)
                for video in os.listdir(pid_dir):
                    audio_path = os.path.join(gt_data_dir, pid, video, "audio.wav")
                    sync_path = os.path.join(gen_data_dir, pid, video, "sync_dict_res.json")
                    if gt:
                        video_dir = os.path.join(gt_data_dir, pid, video, "aligned_syncnet_crop_images_official")
                    else:
                        video_dir = os.path.join(gen_data_dir, pid, video, "aligned_syncnet_crop_images_official")
                    if len(list(os.listdir(video_dir))) < 5 or (not os.path.exists(audio_path) and not os.path.exists(audio_path.replace(".mp3", ".wav"))):
                        continue
                    offset, conf, minval = self.evaluate_single(video_dir, audio_path)
                    print(offset, conf, minval)
                    res_dict = {
                        'avoff': float(offset),
                        'lsed': float(minval),
                        'lsec': float(conf)
                    }

                    # with open(sync_path, 'w') as f:
                    #     json.dump(res_dict, f, indent=4)
                    avoff_list += offset
                    lsec_list += conf
                    lsed_list += minval
                    count += 1

        avoff = avoff_list / count
        lsed = lsed_list / count
        lsec = lsec_list / count

        return avoff, lsed, lsec
    
    
    def inference_feature(self, video_dir, audio_path):
        """ compute sync metric for a single video clip

        Args:
            video_dir (str): a single video clip's images dir
            audio_path (str): the corresponding audio in wav format follow the offifial extract parameters

        Returns:
            offset: the offset 
            conf: the confidence
            minval: minimal distance between audio and video frames in window
        """

        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        items_list = os.listdir(video_dir)
        # print(items_list)
        items_list.sort()
        name, fn = self.parsing_name(items_list[0])
        last_frame = fn - 1
        for idx, frame in enumerate(items_list):
            # frame = format_str.format(idx)
            _, fn = self.parsing_name(frame)
            # print(last_frame, fn)
            if last_frame + 1 != fn:
                break
            last_frame = fn
            img_path = os.path.join(video_dir, frame)
            # img_path = img_path.replace("gen", "real")
            img_input = cv2.imread(img_path)
            images.append(img_input)

        im = numpy.stack(images, axis = 3)
        im = numpy.expand_dims(im, axis = 0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
 
        min_length = len(images)
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []

        tS = time.time()
        for i in range(0, lastframe, batch_size):
            
            im_batch = [ imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size)) ]
            im_in = torch.cat(im_batch, 0)
            im_out  = self.__S__.forward_lip(im_in.cuda())
            im_feat.append(im_out.data.cpu())

        # ========== ==========
        # Load audio
        # ========== ==========

        wav_path = audio_path.replace(".mp3", ".wav")
        
        # if os.path.exists(wav_path):
        #     os.remove(wav_path)

        if not os.path.exists(wav_path):
            # if audio_path.endswith(".wav"):
            #     audio_path = audio_path.replace(".wav", ".mp3")
            # video_path = 
            # command = "ffmpeg -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}".format(video_path, wav_path)
            # os.system(command)
            return 0, 0
            
        sample_rate, audio = wavfile.read(wav_path)
        mfcc = python_speech_features.mfcc(audio, sample_rate) #, winlen=0.05, winstep=0.02)
        mfcc = zip(*mfcc)
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        # print(mfcc.shape)

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis = 0), axis = 0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        # print(imtv.shape, cct.shape, len(audio), sample_rate)
        # print(cct.shape)
        if self.audio_visual_offset_in_data > 0:
            cct = cct[:, :, :, self.audio_visual_offset_in_data*bin_num:-self.audio_visual_offset_in_data*bin_num] # because generated images is 4 frames less than audio, 2 in head and 2 in tail
        # print(cct.shape)
        # print(len(images), len(audio), len(audio) // 640)
        min_length = min(len(images), math.floor(len(audio) / 640))
        # print(len(images), math.floor(cct.shape[3] / bin_num))
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

            cc_batch = [ cct[:, :, :, vframe*bin_num:vframe*bin_num + 5 * bin_num] for vframe in range(i, min(lastframe, i + batch_size)) ]
            # print(i, lastframe, batch_size)
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)
        
        im_feat = im_feat.detach().cpu().numpy()
        cc_feat = cc_feat.detach().cpu().numpy()
        
        return im_feat, cc_feat
    
    def detect_single(self, gen_data_dir, gt_data_dir):
        
        
        video_dir = os.path.join(gen_data_dir, "aligned_syncnet_crop_images_official")
        ori_images_dir = os.path.join(gen_data_dir, "images")
        
        if not os.path.exists(video_dir) or not len(os.listdir(ori_images_dir)) == len(os.listdir(video_dir)):
        
            os.makedirs(video_dir, exist_ok=True)
            self.preprocessor.align_folder(ori_images_dir, video_dir)
        
        audio_path = os.path.join(gt_data_dir, "audio.wav")
        save_img_path = os.path.join(gen_data_dir, "sync_img_feat.npy")
        save_audio_path = os.path.join(gen_data_dir, "sync_audio_feat.npy")
        if len(list(os.listdir(video_dir))) < 5:
            return
        im_feat, au_feat = self.inference_feature(video_dir, audio_path)
        if isinstance(im_feat, int):
            return
        np.save(save_img_path, im_feat)
        np.save(save_audio_path, au_feat)

        return
