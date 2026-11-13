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
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import sys
sys.path.append("../../")

from lib.metric.utils.preprocess import align_img
from lib.deep3dface.single_recon_model import FaceReconModel

def load_lm3d(bfm_folder):

    Lm3D = loadmat(os.path.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

class Deep3dMetric(object):
    def __init__(self, model_path="/data/pipline/human_centric_vw_model/benchmark/express/assets/epoch_20.pth", bfm_folder="/data/pipline/human_centric_vw_model/benchmark/express/assets"):
        self.deep3d_model = FaceReconModel()
        self.deep3d_model.load_state_dict(torch.load(model_path)['net_recon'])
        self.lm3d_std = load_lm3d(bfm_folder) 
        
        self.deep3d_model.eval()
        self.deep3d_model.cuda()
        
        self.batch_size = 8
        
    def inference(self, images_tensor):
        results = []
        
        b, c, h, w = images_tensor.shape
        
        images_tensor = images_tensor.cuda()
        for i in range(round(b / self.batch_size + 0.49999)):
            image_clip = images_tensor[i * self.batch_size:min((i + 1) * self.batch_size, b)]
            res_temp = self.deep3d_model(image_clip)
            
            results.append(res_temp)
            
        results = torch.cat(results, 0).detach().cpu().numpy()
        
        exp_params = results[:, 80: 144]
        pose_params = results[:, 224: 227]
        
        return exp_params, pose_params
    
    def inference_single(self, image_path_list, ldmk_path_list):
        ori_image_list = []
        ldmk_list = []

        for img_path in image_path_list:
            ori_image_list.append(Image.open(img_path))
        for ldmk_path in ldmk_path_list:
            # ldmk_list.append(np.load(ldmk_path))
            ldmk_list.append(np.load(ldmk_path, allow_pickle=True))



        images_tensor, valid_list = self.image_transform(ori_image_list, ldmk_list) # n, c, h, w
        
        with torch.no_grad():
        
            exp_params, pose_params = self.inference(images_tensor)
        
        return exp_params, pose_params, valid_list

    def detect_single(self, image_dir, meta_dir):
        
        if not os.path.exists(image_dir):
            return 

        # gen_img_items_ = list(os.listdir(image_dir))
        gen_img_items_ = [item for item in os.listdir(image_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        gen_img_items_.sort()
        img_format = gen_img_items_[0].split('.')[-1]

        gen_img_items = [img_item for img_item in gen_img_items_ if not os.path.exists(os.path.join(meta_dir, "deep3ds", img_item.replace(img_format, "npy")) )]
        if len(gen_img_items) == 0:
            return

        gen_image_path_list = [os.path.join(image_dir, img_item) for img_item in gen_img_items]
        gen_ldmks_path_list = [os.path.join(meta_dir, "ldmks_98pt", img_item.replace(img_format, "npy")) for img_item in gen_img_items]

        gen_deep3d_dir = os.path.join(meta_dir, "deep3ds")
        os.makedirs(gen_deep3d_dir, exist_ok=True)
            
        gen_exp_params, gen_pose_params, valid_list = self.inference_single(gen_image_path_list, gen_ldmks_path_list)
        gen_params = np.concatenate([gen_exp_params, gen_pose_params], 1)
        for i in range(len(valid_list)):
            img_item = gen_img_items[i]
            gen_deep3d_path = os.path.join(meta_dir, "deep3ds", img_item.replace(img_format, "npy"))
            np.save(gen_deep3d_path, gen_params[i])
        
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

        
        gt_deep3ds_dir = os.path.join(gt_meta_dir, "deep3ds")
        os.makedirs(gt_deep3ds_dir, exist_ok=True)

        gen_deep3ds_dir = os.path.join(gen_meta_dir, "deep3ds")
        os.makedirs(gen_deep3ds_dir, exist_ok=True)
        
        # gt_ldmks_path_list = [os.path.join(gt_meta_dir, "ldmks_98pt", img_item.replace(gt_img_format, "npy")) for img_item in gt_img_items]
        # gen_ldmks_path_list = [os.path.join(gen_meta_dir, "ldmks_98pt", img_item.replace(gen_img_format, "npy")) for img_item in gen_img_items]
        
        # gen_image_path_list = [os.path.join(gen_dir, img_item) for img_item in gen_img_items]
        # gt_image_path_list = [os.path.join(gt_dir, img_item) for img_item in gt_img_items]

        # gt_image_path_list = [os.path.join(gt_dir, "images", img_item.split('.')[0] + ".{}".format(img_format)) for img_item in gen_img_items]
        
        ### new version: each results save and read individual
        gt_params_dict = {}
        gt_deep3d_path_list = []
        gt_image_path_list = []
        gt_ldmk_path_list = []

        # gt_deep3d_list = []
        # gen_deep3d_list = []

        for img_item in gt_img_items:
            gt_deep3ds_path = os.path.join(gt_deep3ds_dir, img_item.split(".")[0] + ".npy")
            if os.path.exists(gt_deep3ds_path):
                gt_param_temp = np.load(gt_deep3ds_path, allow_pickle=True)
                gt_params_dict[int(img_item.split(".")[0])] = gt_param_temp
            else:
                img_path = os.path.join(gt_dir, img_item)
                ldmk_path = os.path.join(gt_meta_dir, "ldmks_98pt", img_item.replace(gt_img_format, "npy"))
                gt_image_path_list.append(img_path)
                gt_ldmk_path_list.append(ldmk_path)
                gt_deep3d_path_list.append(gt_deep3ds_path)
                
        if len(gt_image_path_list) > 0:
            gt_exp_params, gt_pose_params, gt_valid_list = self.inference_single(gt_image_path_list, gt_ldmk_path_list)
            gt_params = np.concatenate([gt_exp_params, gt_pose_params], 1)
            for idx, i in enumerate(gt_valid_list):
                np.save(gt_deep3d_path_list[i], gt_params[idx])
                gt_params_dict[int(gt_image_path_list[i].split('/')[-1].split('.')[0])] = gt_params[idx]
                
        # gt_params_list = []
        
        gt_keys = list(gt_params_dict.keys())
        
        gen_params_dict = {}
        gen_deep3d_path_list = []
        gen_image_path_list = []
        gen_ldmks_path_list = []

        for img_item in gen_img_items:
            gen_deep3ds_path = os.path.join(gen_deep3ds_dir, img_item.split(".")[0] + ".npy")
            if os.path.exists(gen_deep3ds_path):
                gen_param_temp = np.load(gen_deep3ds_path, allow_pickle=True)
                gen_params_dict[int(img_item.split(".")[0])] = gen_param_temp
            else:
                img_path = os.path.join(gen_dir, img_item)
                ldmk_path = os.path.join(gen_meta_dir, "ldmks_98pt", img_item.replace(gen_img_format, "npy"))
                gen_image_path_list.append(img_path)
                gen_ldmks_path_list.append(ldmk_path)
                gen_deep3d_path_list.append(gen_deep3ds_path)

        if len(gen_image_path_list) > 0:
            gen_exp_params, gen_pose_params, gen_valid_list = self.inference_single(gen_image_path_list, gen_ldmks_path_list)
            gen_params = np.concatenate([gen_exp_params, gen_pose_params], 1)
            for idx, i in enumerate(gen_valid_list):
                np.save(gen_deep3d_path_list[i], gen_params[idx])
                gen_params_dict[int(gen_image_path_list[i].split('/')[-1].split('.')[0])] = gen_params[idx]
        
        gen_keys = list(gen_params_dict.keys())

        gen_keys_ = gen_keys[start_num:max_compute_num]
        gt_keys_ = gt_keys[start_num:max_compute_num]

        keys = list(set(gen_keys_).intersection(set(gt_keys)))

        _gen_exp_params = []
        _gt_exp_params = []
        _gen_pose_params = []
        _gt_pose_params = []

        for k in keys:
            _gen_exp_params.append(gen_params_dict[k][:-3])
            _gt_exp_params.append(gt_params_dict[k][:-3])
            _gen_pose_params.append(gen_params_dict[k][-3:])
            _gt_pose_params.append(gt_params_dict[k][-3:])

        _gen_exp_params = np.stack(_gen_exp_params, 0)
        _gt_exp_params = np.stack(_gt_exp_params, 0)
        _gen_pose_params = np.stack(_gen_pose_params, 0)
        _gt_pose_params = np.stack(_gt_pose_params, 0)


        exp_diff = ((_gen_exp_params - _gt_exp_params) ** 2).mean(-1) # n, 64
        pose_diff = ((_gen_pose_params - _gt_pose_params) ** 2).mean(-1) # n, 3
        
        
        # return each images' result
        return exp_diff, pose_diff
    
    def image_transform(self, images_list, lm_list):
        warped_img_list = []
        valid_list = []
        
        count = -1
        for image, lm in zip(images_list, lm_list):
            count += 1
            
            if lm[0] is None:
                continue

            valid_list.append(count)

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
            # cv2.imwrite("test.png", img)     
            img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)

            warped_img_list.append(img)

        warped_img_list = torch.stack(warped_img_list, 0)
        # valid_list = np.array(valid_list)
        assert warped_img_list.shape[0] == len(valid_list)

        return warped_img_list, valid_list
    

# model = Deep3dMetric()
# model.detect_single(image_dir="/data/pipline/human_centric_vw_model/benchmark/express/meta/face",meta_dir="/data/pipline/human_centric_vw_model/benchmark/express/meta")
