#################################################
# Copyright (c) 2021-present, xiaobing.ai, Inc. #
# All rights reserved.                          #
#################################################
# CV Research, DEV(USA) xiaobing.               #
# written by wangduomin@xiaobing.ai             #
#################################################

import os
import json
import numpy as np
import pickle

##### import metric for motion and id
from lib.metric.landmark_detect import LandmarkDetector
from lib.metric.pose_expression import Deep3dMetric
from lib.metric.identity import IdentityMetric
from lib.metric.sync_score_filter import SyncNetInstance
##### import metric for image quality
from lib.metric.fid import FidCompute
from lib.metric.ssim import SSIMCompute
from lib.metric.cpbd import CpbdCompute


########### evalutaion pipeline class
class EvalPipeline(object):
    def __init__(self, offset, s_offset):
        self.s_offset = s_offset
        
        ### inference must follow the construction order of cls-object
        # 1
        self.landmark_metric = LandmarkDetector(s_offset=self.s_offset)
        # 2
        self.deep3d_metric = Deep3dMetric(offset = offset, s_offset=self.s_offset)
        # 3 
        self.identity_metric = IdentityMetric(s_offset=self.s_offset)
        # 4
        self.syncnet_metric = SyncNetInstance(av_offset = offset)
        
        # 5
        self.fid_metric = FidCompute(s_offset = self.s_offset)
        # 6
        self.ssim_metric = SSIMCompute(s_offset = self.s_offset)
        # 7
        self.cpbd_metric = CpbdCompute()
        
        ### list objects for temp result memory
        # step1
        self.ldmk_lip_dist_list = []
        self.ldmk_nme_list = []
        # step2
        self.exp_diff_list = []
        self.pose_diff_list = []
        # step3
        self.id_similarity_list = []
        # step4
        self.av_offset_list = []
        self.lsed_list = []
        self.lsec_list = []
        # step5
        self.gt_fid_activate_feature_list = []
        self.gen_fid_activate_feature_list = []
        
        # step6
        self.ssim_list = []
        self.psnr_list = []
        # step7
        self.cpbd_list = []
        
        
    def get_eval_result(self, meta_save_path):
        """ eval function only compute single video's result, 
        we use self.xx_list to save the history results and this
        function gives the final avg result
        """
        result_dict = {}
        meta_data_res_dict = {}
        
        # step1: landmark related
        print("get landmark related metric...")
        if len(self.ldmk_lip_dist_list) > 0:
            lip_dist_mean = np.mean(self.ldmk_lip_dist_list)
            lip_dist_std = np.std(self.ldmk_lip_dist_list)
            result_dict['lip_lmd'] = {
                'lip_mean': float(lip_dist_mean),
                'lip_std': float(lip_dist_std)
            }
            meta_data_res_dict['lip_lmd'] = self.ldmk_lip_dist_list
                    
        if len(self.ldmk_nme_list) > 0:
            nme_mean = np.mean(self.ldmk_nme_list)
            result_dict['face_nme'] = {
                'nme_mean': float(nme_mean)
            }
            meta_data_res_dict['face_nme'] = self.ldmk_nme_list
        
        # step2: pose and exp
        print("get pose and expression related metric...")
        if len(self.exp_diff_list) > 0:
            exp_diffs = np.concatenate(self.exp_diff_list)
            exp_mean = np.mean(exp_diffs)
            exp_std = np.std(exp_diffs)
            result_dict['exp'] = {
                'exp_mean': float(exp_mean),
                'exp_std': float(exp_std)
            }
            meta_data_res_dict['exp'] = self.exp_diff_list
        
        if len(self.pose_diff_list) > 0:
            pose_diffs = np.concatenate(self.pose_diff_list)
            pose_mean = np.mean(pose_diffs)
            pose_std = np.std(pose_diffs)
            result_dict['pose'] = {
                'pose_mean': float(pose_mean),
                'pose_std': float(pose_std)
            }
            meta_data_res_dict['pose'] = self.pose_diff_list
        
        # step3
        print("get id similarity metric...")
        if len(self.id_similarity_list) > 0:
            id_sim = np.concatenate(self.id_similarity_list)
            id_sim_mean = np.mean(id_sim)
            id_sim_std = np.std(id_sim)
            result_dict['id'] = {
                'id_sim_mean': float(id_sim_mean),
                'id_sim_std': float(id_sim_std)
            }
            meta_data_res_dict['id_sim'] = self.id_similarity_list
        
        # step4
        print("get lip sync metric...")
        if len(self.av_offset_list) > 0:
            av_offset = np.mean(self.av_offset_list)
            result_dict['lip_sync'] = {
                'av_offset': float(av_offset)
            }
            meta_data_res_dict['av_offset'] = self.av_offset_list
        
        if len(self.lsed_list) > 0:
            lsed = np.mean(self.lsed_list)
            result_dict['lip_sync']['lsed'] = float(lsed)
            meta_data_res_dict['lsed'] = self.lsed_list
        
        if len(self.lsec_list) > 0:
            lsec = np.mean(self.lsec_list)
            result_dict['lip_sync']['lsec'] = float(lsec)
            meta_data_res_dict['lsec'] = self.lsec_list
        
        # step5
        print("get fid metric...")
        if len(self.gt_fid_activate_feature_list) > 0:
            gt_act_feature = np.concatenate(self.gt_fid_activate_feature_list, 0)
            gen_act_feature = np.concatenate(self.gen_fid_activate_feature_list, 0)
            m1, s1 = self.fid_metric.calculate_statistics(gt_act_feature)
            m2, s2 = self.fid_metric.calculate_statistics(gen_act_feature)
            # fid_value = self.fid_metric.calculate_frechet_distance(m1, s1, m2, s2)
            fid_value = self.fid_metric.compute_fid(m1, s1, m2, s2)
            
            result_dict['img_quality'] = {
                'fid': float(fid_value)
            }
            meta_data_res_dict['gt_statics'] = [m1, s1]
            meta_data_res_dict['gen_statics'] = [m2, s2]
        
        # stepimage.png6
        print("get ssim and psnr metric...")
        if len(self.ssim_list) > 0:
            ssim = np.mean(self.ssim_list)
            if "img_quality" in result_dict:
                result_dict['img_quality']['ssim'] = float(ssim)
            else:
                result_dict['img_quality'] = {'ssim': float(ssim)}
            meta_data_res_dict['ssim'] = self.ssim_list
        
        if len(self.psnr_list) > 0:
            psnr = np.mean(self.psnr_list)
            if "img_quality" in result_dict:
                result_dict['img_quality']['psnr'] = float(psnr)
            else:
                result_dict['img_quality'] = {'psnr': float(psnr)}
            meta_data_res_dict['psnr'] = self.psnr_list
        
        # step7
        print("get cpbd metric...")
        if len(self.cpbd_list) > 0:
            cpbd = np.mean(self.cpbd_list)
            if "img_quality" in result_dict:
                result_dict['img_quality']['cpbd'] = float(cpbd)
            else:
                result_dict['img_quality'] = {'cpbd': float(cpbd)}
            meta_data_res_dict['cpbd'] = self.cpbd_list
        
        with open(meta_save_path, 'wb') as f:
            pickle.dump(meta_data_res_dict, f)
        
        # remove the list's memory
        self.clear_memory()
        
        return result_dict
        
    def clear_memory(self):
        """ clear the temp result memory, usually call after get_eval_result
        """
        
        all_attr = dir(self)
        for item in all_attr:
            if "list" in item:
                getattr(self, item).clear()
        
    def eval_audio_drive_inference(self, gt_dir, gen_dir, id_path=None):
        """ for audio drive metric compute

        Args:
            gt_dir (str): ground truth single video's dir
            gen_dir (str): generated single video's dir
        """
        
        # step1: compute landmarks and lip accuracy metric lmd
        ldmk_lip_dist, ldmk_nme = self.landmark_metric.evaluate_single_pair_lip(gt_dir, gen_dir)
        self.ldmk_lip_dist_list.extend(ldmk_lip_dist)
        self.ldmk_nme_list.extend(ldmk_nme)
        
        # step2: compute pose and exp metric
        exp_diff, pose_diff = self.deep3d_metric.evaluate_single_pair(gt_dir, gen_dir)
        self.exp_diff_list.append(exp_diff)
        self.pose_diff_list.append(pose_diff)
        
        # step3: compute identity metric
        if id_path is None:
            id_sim = self.identity_metric.evaluate_single_pair(gt_dir, gen_dir)
        else:
            id_sim = self.identity_metric.evaluate_single_pair(id_path, gen_dir)
        self.id_similarity_list.append(id_sim)
        
        # step4: compute syncnet metric
        av_off, lsed, lsec = self.syncnet_metric.evaluate_gen_single(gt_dir, gen_dir)
        self.av_offset_list.append(av_off)
        self.lsed_list.append(lsed)
        self.lsec_list.append(lsec)
        
        # step5: compute fid metric
        if id_path is None:
            feat_act_gt, feat_act_gen = self.fid_metric.evaluate_act_single_pair(gt_dir, gen_dir)
        else:
            feat_act_gt, feat_act_gen = self.fid_metric.evaluate_act_single_pair(id_path, gen_dir)
        self.gt_fid_activate_feature_list.append(feat_act_gt)
        self.gen_fid_activate_feature_list.append(feat_act_gen)
        # fid_score = self.fid_metric.evaluate_single_pair_fid(gt_dir, gen_dir)
        # self.fid_score_list.append(fid_score)
            
        if id_path is None:
            # step6: compute ssim and psnr metric
            ssim, psnr = self.ssim_metric.evaluate_single_pair(gt_dir, gen_dir)
            self.ssim_list.extend(ssim)
            self.psnr_list.extend(psnr)
            
        # step7: compute cpbd metric
        cpbd = self.cpbd_metric.evaluate_single_cpbd(gen_dir)
        self.cpbd_list.extend(cpbd)
        
        
    def eval_reenactment_inference(self, gt_dir, gen_dir):
        """ for reenactment metric compute

        Args:
            gt_dir (str): ground truth single video's dir
            gen_dir (str): generated single video's dir
        """
        
        # step1: compute landmarks and lip accuracy metric lmd
        ldmk_lip_dist, ldmk_nme = self.landmark_metric.evaluate_single_pair_lip(gt_dir, gen_dir)
        self.ldmk_lip_dist_list.extend(ldmk_lip_dist)
        self.ldmk_nme_list.extend(ldmk_nme)
        
        # step2: compute pose and exp metric
        exp_diff, pose_diff = self.deep3d_metric.evaluate_single_pair(gt_dir, gen_dir)
        self.exp_diff_list.append(exp_diff)
        self.pose_diff_list.append(pose_diff)
        
        # step3: compute identity metric
        id_sim = self.identity_metric.evaluate_single_pair(gt_dir, gen_dir)
        self.id_similarity_list.append(id_sim)
        
        # step4: compute fid metric
        feat_act_gt, feat_act_gen = self.fid_metric.evaluate_act_single_pair(gt_dir, gen_dir)
        self.gt_fid_activate_feature_list.append(feat_act_gt)
        self.gen_fid_activate_feature_list.append(feat_act_gen)
        # fid_score = self.fid_metric.evaluate_single_pair_fid(gt_dir, gen_dir)
        # self.fid_score_list.append(fid_score)
        
        # step5: compute ssim and psnr metric
        ssim, psnr = self.ssim_metric.evaluate_single_pair(gt_dir, gen_dir)
        self.ssim_list.extend(ssim)
        self.psnr_list.extend(psnr)
        
        # step6: compute cpbd metric
        cpbd = self.cpbd_metric.evaluate_single_cpbd(gen_dir)
        self.cpbd_list.extend(cpbd)