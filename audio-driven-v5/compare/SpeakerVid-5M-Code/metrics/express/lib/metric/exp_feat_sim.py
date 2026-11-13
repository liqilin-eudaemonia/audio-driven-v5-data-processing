#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import cv2
import PIL.Image as Image
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import sys
from torchvision import transforms 
sys.path.append("../../")

from lib.metric.utils.preprocess import align_img
# from lib.deep3dface.single_recon_model import FaceReconModel
from lib.emonet.emonet import EmoNet
from lib.emoca_emo.FRNet import resnet50
from lib.metric.utils.data_augmentation import DataAugmentor

def load_lm3d(bfm_folder):

    Lm3D = loadmat(os.path.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

class EmoresEncoder(nn.Module):
    def __init__(self):
        super(EmoresEncoder, self).__init__()
        self.backbone = resnet50()

    def forward(self, x):
        feature = self.backbone(x)
        return feature

class EmoMetric(object):
    def __init__(self, model_path1="assets/resnet50_emo.pth", model_path2="assets/H3R.pth", bfm_folder="assets"):
        self.emoca_emo_model = EmoresEncoder()
        model_dict = {}
        state_dict = torch.load(model_path1)
        for key in state_dict:
            if key == "linear.weight" or key == "linear.bias":
                continue
            model_dict[key] = state_dict[key]
        self.emoca_emo_model.load_state_dict(model_dict)

        self.emonet_emo_model = EmoNet()
        self.emonet_emo_model.load_state_dict(torch.load(model_path2))
        self.lm3d_std = load_lm3d(bfm_folder) 
        
        self.emoca_emo_model.eval()
        self.emoca_emo_model.cuda()
        self.emonet_emo_model.eval()
        self.emonet_emo_model.cuda()
        
        self.batch_size = 8

        self.transform_image = transforms.Compose([transforms.ToTensor()])
        self.transform_image_shape = DataAugmentor(256, 256)
        
    def inference(self, images_tensor_emoca, images_tensor_emonet):
        results1 = []
        results2 = []
        
        b, c, h, w = images_tensor_emoca.shape
        
        images_tensor_emoca = images_tensor_emoca.cuda()
        images_tensor_emonet = images_tensor_emonet.cuda()
        for i in range(round(b / self.batch_size + 0.49999)):
            image_clip1 = images_tensor_emoca[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            image_clip2 = images_tensor_emonet[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            res_temp1 = self.emoca_emo_model(image_clip1)
            res_temp2 = self.emonet_emo_model(image_clip2)['feature']
            
            results1.append(res_temp1)
            results2.append(res_temp2)
            
        emo_feat_emoca = torch.cat(results1, 0).detach().cpu().numpy()
        emo_feat_emonet = torch.cat(results2, 0).detach().cpu().numpy()
        
        return emo_feat_emoca, emo_feat_emonet
    
    def inference_single(self, image_path_list, ldmk_path_list):
        ori_image_list = []
        ldmk_list = []

        for img_path in image_path_list:
            ori_image_list.append(Image.open(img_path))
        for ldmk_path in ldmk_path_list:
            ldmk_list.append(np.load(ldmk_path))

        images_tensor_emoca, iamges_tensor_emonet, valid_list = self.image_transform(ori_image_list, ldmk_list) # n, c, h, w
        
        with torch.no_grad():
        
            emo_feat_emoca, emo_feat_emonet = self.inference(images_tensor_emoca, iamges_tensor_emonet)
        
        return emo_feat_emoca, emo_feat_emonet, valid_list

    def detect_single(self, image_dir, meta_dir):
        
        if not os.path.exists(image_dir):
            return 

        # gen_img_items_ = list(os.listdir(image_dir))
        gen_img_items_ = [item for item in os.listdir(image_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        gen_img_items_.sort()
        img_format = gen_img_items_[0].split('.')[-1]

        gen_img_items = [img_item for img_item in gen_img_items_ if not os.path.exists(os.path.join(meta_dir, "emonet_emofeat", img_item.replace(img_format, "npy")) )]
        if len(gen_img_items) == 0:
            return

        gen_image_path_list = [os.path.join(image_dir, img_item) for img_item in gen_img_items]
        gen_ldmks_path_list = [os.path.join(meta_dir, "ldmks_98pt", img_item.replace(img_format, "npy")) for img_item in gen_img_items]

        gen_emonet_emofeat_dir = os.path.join(meta_dir, "emonet_emofeat")
        gen_emoca_emofeat_dir = os.path.join(meta_dir, "emoca_emofeat")
        os.makedirs(gen_emonet_emofeat_dir, exist_ok=True)
        os.makedirs(gen_emoca_emofeat_dir, exist_ok=True)
            
        gen_emo_feat_emoca, gen_emo_feat_emonet, valid_list = self.inference_single(gen_image_path_list, gen_ldmks_path_list)
        for i in valid_list:
            img_item = gen_img_items[i]
            gen_emonet_emofeat_path = os.path.join(meta_dir, "emonet_emofeat", img_item.replace(img_format, "npy"))
            gen_emoca_emofeat_path = os.path.join(meta_dir, "emoca_emofeat", img_item.replace(img_format, "npy"))
            np.save(gen_emoca_emofeat_path, gen_emo_feat_emoca[i])
            np.save(gen_emonet_emofeat_path, gen_emo_feat_emonet[i])
        
        return 
    
    def evaluate_single_pair(self, gt_dir, gen_dir, gt_meta_dir, gen_meta_dir, start_num, max_compute_num, item_list=None):
        
        if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
            return None, None

        # gen_img_items = [item for item in os.listdir(gen_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        # gt_img_items_ori = [item for item in os.listdir(gt_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        if item_list is None:
            gen_img_items = [item for item in os.listdir(gen_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
            gt_img_items_ori = [item for item in os.listdir(gt_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        else:
            gen_img_items = [item+".png" for item in item_list]
            gt_img_items_ori = [item+".png" for item in item_list]

        gen_img_items.sort()
        gt_img_items_ori.sort()

        if len(gt_img_items_ori) == 0 or len(gen_img_items) == 0:
            return None, None

        # if len(gt_img_items_ori[0].split('.')[0]) == 8:
        #     gt_img_prefix = "{:08d}" 
        # elif len(gt_img_items_ori[0].split('.')[0]) == 6:
        #     gt_img_prefix = "{:06d}" 
        
        gt_img_format = gt_img_items_ori[0].split('.')[-1]
        gen_img_format = gen_img_items[0].split('.')[-1]

        # gt_img_items = [gt_img_prefix.format(int(item.split(".")[0])) + '.{}'.format(gt_img_format) for item in gen_img_items]
        gt_img_items = ["{:08d}".format(int(item.split(".")[0])) + '.{}'.format(gt_img_format) for item in gen_img_items]

        
        gt_emoca_emofeat_dir = os.path.join(gt_meta_dir, "emoca_emofeat")
        gt_emonet_emofeat_dir = os.path.join(gt_meta_dir, "emonet_emofeat")
        os.makedirs(gt_emoca_emofeat_dir, exist_ok=True)
        os.makedirs(gt_emonet_emofeat_dir, exist_ok=True)

        gen_emoca_emofeat_dir = os.path.join(gen_meta_dir, "emoca_emofeat")
        gen_emonet_emofeat_dir = os.path.join(gen_meta_dir, "emonet_emofeat")
        os.makedirs(gen_emoca_emofeat_dir, exist_ok=True)
        os.makedirs(gen_emonet_emofeat_dir, exist_ok=True)
        
        ### new version: each results save and read individual
        gt_emoca_emofeat_dict = {}
        gt_emoca_emofeat_path_list = []
        gt_emonet_emofeat_dict = {}
        gt_emonet_emofeat_path_list = []
        gt_image_path_list = []
        gt_ldmk_path_list = []

        # gt_deep3d_list = []
        # gen_deep3d_list = []

        for img_item in gt_img_items:
            gt_emoca_emofeat_path = os.path.join(gt_emoca_emofeat_dir, img_item.split(".")[0] + ".npy")
            gt_emonet_emofeat_path = os.path.join(gt_emonet_emofeat_dir, img_item.split(".")[0] + ".npy")
            if os.path.exists(gt_emoca_emofeat_path) and os.path.exists(gt_emonet_emofeat_path):
                gt_emoca_emofeat_temp = np.load(gt_emoca_emofeat_path, allow_pickle=True)
                gt_emoca_emofeat_dict[int(img_item.split(".")[0])] = gt_emoca_emofeat_temp
                gt_emonet_emofeat_temp = np.load(gt_emonet_emofeat_path, allow_pickle=True)
                gt_emonet_emofeat_dict[int(img_item.split(".")[0])] = gt_emonet_emofeat_temp
            else:
                img_path = os.path.join(gt_dir, img_item)
                ldmk_path = os.path.join(gt_meta_dir, "ldmks_98pt", img_item.replace(gt_img_format, "npy"))
                gt_image_path_list.append(img_path)
                gt_ldmk_path_list.append(ldmk_path)
                gt_emoca_emofeat_path_list.append(gt_emoca_emofeat_path)
                gt_emonet_emofeat_path_list.append(gt_emonet_emofeat_path)
                
        if len(gt_image_path_list) > 0:
            gt_emoca_emofeat, gt_emonet_emofeat, gt_valid_list = self.inference_single(gt_image_path_list, gt_ldmk_path_list)
            for idx, i in enumerate(gt_valid_list):
                np.save(gt_emoca_emofeat_path_list[i], gt_emoca_emofeat[idx])
                np.save(gt_emonet_emofeat_path_list[i], gt_emonet_emofeat[idx])
                gt_emoca_emofeat_dict[int(gt_image_path_list[i].split('/')[-1].split('.')[0])] = gt_emoca_emofeat[idx]
                gt_emonet_emofeat_dict[int(gt_image_path_list[i].split('/')[-1].split('.')[0])] = gt_emonet_emofeat[idx]
                
        # gt_params_list = []
        
        gt_keys = list(gt_emoca_emofeat_dict.keys())
        
        gen_emoca_emofeat_dict = {}
        gen_emoca_emofeat_path_list = []
        gen_emonet_emofeat_dict = {}
        gen_emonet_emofeat_path_list = []
        gen_image_path_list = []
        gen_ldmks_path_list = []

        for img_item in gen_img_items:
            gen_emoca_emofeat_path = os.path.join(gen_emoca_emofeat_dir, img_item.split(".")[0] + ".npy")
            gen_emonet_emofeat_path = os.path.join(gen_emonet_emofeat_dir, img_item.split(".")[0] + ".npy")
            if os.path.exists(gen_emoca_emofeat_path) and os.path.exists(gen_emonet_emofeat_path):
                gen_emoca_emofeat_temp = np.load(gen_emoca_emofeat_path, allow_pickle=True)
                gen_emoca_emofeat_dict[int(img_item.split(".")[0])] = gen_emoca_emofeat_temp
                gen_emonet_emofeat_temp = np.load(gen_emonet_emofeat_path, allow_pickle=True)
                gen_emonet_emofeat_dict[int(img_item.split(".")[0])] = gen_emonet_emofeat_temp
            else:
                img_path = os.path.join(gen_dir, img_item)
                ldmk_path = os.path.join(gen_meta_dir, "ldmks_98pt", img_item.replace(gen_img_format, "npy"))
                gen_image_path_list.append(img_path)
                gen_ldmks_path_list.append(ldmk_path)
                gen_emoca_emofeat_path_list.append(gen_emoca_emofeat_path)
                gen_emonet_emofeat_path_list.append(gen_emonet_emofeat_path)

        if len(gen_image_path_list) > 0:
            gen_emoca_emofeat, gen_emonet_emofeat, gen_valid_list = self.inference_single(gen_image_path_list, gen_ldmks_path_list)
            for idx, i in enumerate(gen_valid_list):
                np.save(gen_emoca_emofeat_path_list[i], gen_emoca_emofeat[idx])
                np.save(gen_emonet_emofeat_path_list[i], gen_emonet_emofeat[idx])
                gen_emoca_emofeat_dict[int(gen_image_path_list[i].split('/')[-1].split('.')[0])] = gen_emoca_emofeat[idx]
                gen_emonet_emofeat_dict[int(gen_image_path_list[i].split('/')[-1].split('.')[0])] = gen_emonet_emofeat[idx]
        
        gen_keys = list(gen_emoca_emofeat_dict.keys())

        gen_keys_ = gen_keys[start_num:max_compute_num]
        gt_keys_ = gt_keys[start_num:max_compute_num]

        keys = list(set(gen_keys_).intersection(set(gt_keys)))

        _gen_emoca_emofeat = []
        _gt_emoca_emofeat = []
        _gen_emonet_emofeat = []
        _gt_emonet_emofeat = []

        for k in keys:
            _gen_emoca_emofeat.append(gen_emoca_emofeat_dict[k])
            _gt_emoca_emofeat.append(gt_emoca_emofeat_dict[k])
            _gen_emonet_emofeat.append(gen_emonet_emofeat_dict[k])
            _gt_emonet_emofeat.append(gt_emonet_emofeat_dict[k])

        _gen_emoca_emofeat = torch.from_numpy(np.stack(_gen_emoca_emofeat, 0))
        _gt_emoca_emofeat = torch.from_numpy(np.stack(_gt_emoca_emofeat, 0))
        _gen_emonet_emofeat = torch.from_numpy(np.stack(_gen_emonet_emofeat, 0))
        _gt_emonet_emofeat = torch.from_numpy(np.stack(_gt_emonet_emofeat, 0))

        # print(_gen_emoca_emofeat.shape)
        # print(_gen_emonet_emofeat.shape)

        emoca_emofeat_diff = (((_gen_emoca_emofeat - _gt_emoca_emofeat) / torch.linalg.norm(_gt_emoca_emofeat, 2, 1)[:, None]) ** 2).sum(-1) # n, 64
        emonet_emofeat_diff = (((_gen_emonet_emofeat - _gt_emonet_emofeat) / torch.linalg.norm(_gt_emonet_emofeat, 2, 1)[:, None]) ** 2).sum(-1) # n, 3

        # emoca_emo_sim = F.cosine_similarity(_gen_emoca_emofeat, _gt_emoca_emofeat, dim=-1).squeeze().mean(-1)
        # emonet_emo_sim = F.cosine_similarity(_gen_emonet_emofeat, _gt_emonet_emofeat, dim=-1).squeeze().mean(-1)
        # print(emoca_emofeat_diff, emonet_emofeat_diff)
        # print(emoca_emo_sim, emonet_emo_sim)
        
        # return each images' result
        # return emoca_emo_sim, emonet_emo_sim
        return emoca_emofeat_diff, emonet_emofeat_diff
    
    def image_transform(self, images_list, lm_list):
        emoca_warped_img_list = []
        emonet_warped_img_list = []
        valid_list = []
        
        count = -1
        for image, lm in zip(images_list, lm_list):
            count += 1
            
            if lm[0] is None:
                continue

            valid_list.append(count)

            bounding_box = [lm.min(axis=0)[0], lm.min(axis=0)[1],
                            lm.max(axis=0)[0], lm.max(axis=0)[1]]
            # cv2.imwrite("test_ori.png", np.array(image))     
            img0, _ = self.transform_image_shape(np.array(image), bb=bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            img0 = np.ascontiguousarray(img0)
            # img0 = np.array(img0).astype(np.uint8)
            # cv2.imwrite("test_emonet.png", img0)     
            img0 = self.transform_image(img0)
            emonet_warped_img_list.append(img0)

            W,H = image.size
            if np.mean(lm) == -1:
                lm = (self.lm3d_std[:, :2]+1)/2.
                lm = np.concatenate(
                    [lm[:, :1]*W, lm[:, 1:2]*H], 1
                )
            else:
                lm[:, -1] = H - 1 - lm[:, -1]

            trans_params, img, lm, _ = align_img(image, lm, self.lm3d_std)
            # img = np.array(img).astype(np.uint8)
            # cv2.imwrite("test_emoca.png", img)     
            img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)

            emoca_warped_img_list.append(img)

        emoca_warped_img_list = torch.stack(emoca_warped_img_list, 0)
        emonet_warped_img_list = torch.stack(emonet_warped_img_list, 0)
        # valid_list = np.array(valid_list)
        assert emoca_warped_img_list.shape[0] == len(valid_list)

        return emoca_warped_img_list, emonet_warped_img_list, valid_list