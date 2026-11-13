import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

from lib.arcface.iresnet import iresnet18
from kornia.geometry import warp_affine
from lib.metric.utils.preprocess import estimate_norm_torch

sys.path.append("../")

class IDSim(object):
    def __init__(self, model_path='lib/arcface/ms1mv3_arcface_r18_fp16/backbone.pth'):
        self.facenet = iresnet18()
        self.facenet.load_state_dict(torch.load(model_path))
        self.facenet = self.facenet.eval().to('cuda')
        
    def inference(self, img_gt, img_pred, gt_seg, pred_seg, ldmk_gt, ldmk_pred, return_img=False):
        ### input img: RGB, [0, 255]
        ### facenet input: img should be RGB and normalized to [-1, 1]

        img_gt = img_gt / 255. * 2. - 1.
        img_pred = img_pred / 255. * 2. - 1.

        img_gt = (0.5*(img_gt+1)*gt_seg)*2 - 1
        img_pred = (0.5*(img_pred+1)*pred_seg)*2 - 1
        
        # print(ldmk_gt)
        # print(ldmk_pred)
        trans_m_gt = estimate_norm_torch(ldmk_gt, img_gt.shape[-1]).to(img_gt.device)
        trans_m_pred = estimate_norm_torch(ldmk_pred, img_pred.shape[-1]).to(img_gt.device)

        img_gt_warp = torch.clamp(warp_affine(img_gt, trans_m_gt, dsize=(112, 112)),-1,1)
        img_pred_warp = torch.clamp(warp_affine(img_pred, trans_m_pred, dsize=(112, 112)),-1,1)

        with torch.no_grad():
            gt_id = self.facenet(img_gt_warp)
            pred_id = self.facenet(img_pred_warp)

        id_sim = F.cosine_similarity(gt_id, pred_id, dim=-1).squeeze()

        if return_img:
            return id_sim.detach().cpu().numpy(), img_gt_warp.permute(0, 2, 3, 1).detach().cpu().numpy(), img_pred_warp.permute(0, 2, 3, 1).detach().cpu().numpy()
            # for l in ldmk_gt[0]:
            #     x = int(l[0])
            #     y = int(l[1])
            #     for m in range(-2, 2):
            #         for n in range(-2, 2):
            #             img_gt[:, 1, y+n, x+m] = 1
            # for l in ldmk_pred[0]:
            #     x = int(l[0])
            #     y = int(l[1])
            #     for m in range(-2, 2):
            #         for n in range(-2, 2):
            #             img_pred[:, 1, y+n, x+m] = 1
            # return id_sim.detach().cpu().numpy(), img_gt.permute(0, 2, 3, 1).detach().cpu().numpy(), img_pred.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            return id_sim.detach().cpu().numpy()

    def inference2(self, img_gt, img_pred, ldmk_gt, ldmk_pred, return_img=False):
        ### input img: RGB, [0, 255]
        ### facenet input: img should be RGB and normalized to [-1, 1]

        img_gt = img_gt / 255. * 2. - 1.
        img_pred = img_pred / 255. * 2. - 1.

        img_gt = (0.5*(img_gt+1))*2 - 1
        img_pred = (0.5*(img_pred+1))*2 - 1
        
        # print(ldmk_gt)
        # print(ldmk_pred)
        trans_m_gt = estimate_norm_torch(ldmk_gt, img_gt.shape[-1]).to(img_gt.device)
        trans_m_pred = estimate_norm_torch(ldmk_pred, img_pred.shape[-1]).to(img_gt.device)

        img_gt_warp = torch.clamp(warp_affine(img_gt, trans_m_gt, dsize=(112, 112)),-1,1)
        img_pred_warp = torch.clamp(warp_affine(img_pred, trans_m_pred, dsize=(112, 112)),-1,1)

        with torch.no_grad():
            gt_id = self.facenet(img_gt_warp)
            pred_id = self.facenet(img_pred_warp)

        id_sim = F.cosine_similarity(gt_id, pred_id, dim=-1).squeeze()

        if return_img:
            return id_sim.detach().cpu().numpy(), img_gt_warp.permute(0, 2, 3, 1).detach().cpu().numpy(), img_pred_warp.permute(0, 2, 3, 1).detach().cpu().numpy()
            # for l in ldmk_gt[0]:
            #     x = int(l[0])
            #     y = int(l[1])
            #     for m in range(-2, 2):
            #         for n in range(-2, 2):
            #             img_gt[:, 1, y+n, x+m] = 1
            # for l in ldmk_pred[0]:
            #     x = int(l[0])
            #     y = int(l[1])
            #     for m in range(-2, 2):
            #         for n in range(-2, 2):
            #             img_pred[:, 1, y+n, x+m] = 1
            # return id_sim.detach().cpu().numpy(), img_gt.permute(0, 2, 3, 1).detach().cpu().numpy(), img_pred.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            return id_sim.detach().cpu().numpy()