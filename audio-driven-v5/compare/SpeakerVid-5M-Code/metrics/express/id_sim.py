import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

from lib.arcface.iresnet import iresnet18
from kornia.geometry import warp_affine
from lib.metric.utils.preprocess import estimate_norm_torch
import PIL.Image as Image
import torchvision.transforms as transforms
sys.path.append("../")
os.environ["KORNIA_USE_RS"] = "0"
def to_tensor_img(img_np):
    """将 numpy 格式的 RGB 图像 (H, W, 3) 转为 torch tensor 格式 (1, 3, H, W)，值范围 [0,255]"""
    if isinstance(img_np, Image.Image):
        img_np = np.array(img_np)

    img_tensor = torch.from_numpy(img_np.astype(np.float32))  # [H, W, 3]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)     # [1, 3, H, W]
    return img_tensor

def to_tensor_ldmk(ldmk_np):
    """将 numpy 格式的 landmark (N, 2/3) 转为 torch tensor 格式 (1, N, 2/3)"""
    return torch.from_numpy(ldmk_np.astype(np.float32)).unsqueeze(0)  # [1, N, 2/3]
class IDSim(object):
    def __init__(self, model_path='lib/arcface/ms1mv3_arcface_r18_fp16/backbone.pth'):
        self.facenet = iresnet18()
        self.facenet.load_state_dict(torch.load(model_path))
        self.facenet = self.facenet.eval().to('cuda')
        self.transform = transforms.ToTensor() 
        
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
        
        trans_m_gt = estimate_norm_torch(ldmk_gt, img_gt.shape[-1]).to(img_gt.device)
        trans_m_pred = estimate_norm_torch(ldmk_pred, img_pred.shape[-1]).to(img_gt.device)

        img_gt_warp = torch.clamp(warp_affine(img_gt, trans_m_gt, dsize=(112, 112)),-1,1)
        img_pred_warp = torch.clamp(warp_affine(img_pred, trans_m_pred, dsize=(112, 112)),-1,1)

        with torch.no_grad():
            gt_id = self.facenet(img_gt_warp.to('cuda'))
            pred_id = self.facenet(img_pred_warp.to('cuda'))

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
        
    
    def inference_single(self, image_path_list, ldmk_path_list,ref_img_path, ref_ldmk_path):
        ori_image_list = []
        ldmk_list = []

        for img_path in image_path_list:
            ori_image_list.append(Image.open(img_path).convert('RGB'))
        for ldmk_path in ldmk_path_list:
            ldmk_list.append(np.load(ldmk_path, allow_pickle=True))
        ref_img = Image.open(ref_img_path).convert('RGB')
        ref_img = to_tensor_img(ref_img)
        ref_ldmk = np.load(ref_ldmk_path, allow_pickle=True)
        ref_ldmk = to_tensor_ldmk(ref_ldmk)

        id_sim_list = []
        for i in range(len(ori_image_list) - 1):
            pred_img = to_tensor_img(ori_image_list[i + 1])
            pred_ldmk = to_tensor_ldmk(ldmk_list[i + 1])

            id_sim = self.inference2(ref_img, pred_img, ref_ldmk, pred_ldmk)
            id_sim_list.append(id_sim)

        
        return id_sim_list 
    def detect_single(self, image_dir, meta_dir,ref_img_path,ref_ldmk_path):
        
        if not os.path.exists(image_dir):
            return 
        gen_img_items_ = [item for item in os.listdir(image_dir) if (item.endswith(".jpg") or item.endswith(".png"))]
        gen_img_items_.sort()
        img_format = gen_img_items_[0].split('.')[-1]

        gen_img_items = [img_item for img_item in gen_img_items_]
        if len(gen_img_items) == 0:
            return

        gen_image_path_list = [os.path.join(image_dir, img_item) for img_item in gen_img_items]
        gen_ldmks_path_list = [os.path.join(meta_dir, "ldmks_98pt", img_item.replace(img_format, "npy")) for img_item in gen_img_items]
            
        id_sim_list = self.inference_single(gen_image_path_list, gen_ldmks_path_list,ref_img_path, ref_ldmk_path)
        
        return sum(id_sim_list) / len(id_sim_list)