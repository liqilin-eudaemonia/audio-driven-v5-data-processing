from ultralytics import YOLO
import os
import cv2
from express.pose_expression import Deep3dMetric
from express.landmark_detect_h3r import LandmarkDetector

def adjust_bbox_scaled(box, image_shape, scale=1.5):
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = map(int, box)

    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * scale
    new_h = h * scale

    x1_new = int(cx - new_w / 2)
    x2_new = int(cx + new_w / 2)
    y1_new = int(cy - new_h / 2)
    y2_new = int(cy + new_h / 2)

    # 边界裁剪
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(img_w, x2_new)
    y2_new = min(img_h, y2_new)

    return [x1_new, y1_new, x2_new, y2_new]


def crop_and_save(image, box, save_img_path, save_txt_path=None):
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(save_img_path, crop)

    # 保存 bbox 到 txt 文件
    if save_txt_path:
        with open(save_txt_path, 'w') as f:
            f.write(f"{x1} {y1} {x2} {y2}\n")

def process_images(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read {img_path}")
            continue

        results = model.predict(source=image, conf=0.5)
        result = results[0]

        if len(result.boxes) == 0:
            print(f"No person detected in {img_file}")
            continue

        box = result.boxes.xyxy[0].cpu().numpy()
        adjusted_box = adjust_bbox_scaled(box, image.shape)

        name, _ = os.path.splitext(img_file)
        save_img_path = os.path.join(output_folder, f"{name}.jpg")
        crop_and_save(image, adjusted_box, save_img_path, save_txt_path=None)
        print(f"Saved: {save_img_path}")
def process_videos_org(video_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        video_name, _ = os.path.splitext(video_file)
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(source=frame, conf=0.5)
            result = results[0]

            if len(result.boxes) == 0:
                print(f"No person detected in {video_file} frame {frame_idx}")
                frame_idx += 1
                continue

            box = result.boxes.xyxy[0].cpu().numpy()
            adjusted_box = adjust_bbox_scaled(box, frame.shape)

            save_img_path = os.path.join(video_output_dir, f"{frame_idx:05d}.jpg")
            crop_and_save(frame, adjusted_box, save_img_path)

            print(f"Saved: {save_img_path}")
            frame_idx += 1

        cap.release()

import os
import cv2

def process_videos(video_folder, output_folder, model, N=1):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        video_name, _ = os.path.splitext(video_file)
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frame_idx = 0  

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % N == 0:
                results = model.predict(source=frame, conf=0.5)
                result = results[0]

                if len(result.boxes) == 0:
                    print(f"No person detected in {video_file} frame {frame_idx}")
                else:
                    box = result.boxes.xyxy[0].cpu().numpy()
                    adjusted_box = adjust_bbox_scaled(box, frame.shape)

                    save_img_path = os.path.join(video_output_dir, f"{frame_idx:05d}.jpg")
                    crop_and_save(frame, adjusted_box, save_img_path)

                    print(f"Saved: {save_img_path}")

            frame_idx += 1

        cap.release()



        
import numpy as np
from scipy import linalg
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

def compute_intra_diversity(features_list):
    """
    features_list: list of shape (T_i, 64), len=N
    Return: average intra-video diversity across all videos
    """
    diversities = []
    for feat in features_list:
        if len(feat) < 2:
            continue
        dist_mat = cosine_distances(feat)  # shape (T, T)
        triu_vals = dist_mat[np.triu_indices(len(feat), k=1)]
        diversities.append(np.mean(triu_vals))
    return np.mean(diversities)
def compute_inter_diversity(features_list):
    """
    features_list: list of shape (T_i, 64), len=N
    Return: average pairwise cosine distance between videos
    """
    avg_feats = [np.mean(feat, axis=0) for feat in features_list]
    avg_feats = np.stack(avg_feats, axis=0)  # shape (N, 64)
    dist_mat = cosine_distances(avg_feats)
    triu_vals = dist_mat[np.triu_indices(len(avg_feats), k=1)]
    return np.mean(triu_vals)

def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    real_features: numpy array of shape (N1, D)
    fake_features: numpy array of shape (N2, D)
    """
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)

    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)
import shutil
if __name__ == "__main__":
    weight_path = "SpeakerVid-5M/weights/best_face.pt"  
    need_compute = True
    if need_compute:
        yolo_model = YOLO(weight_path)
        deep3d_model = Deep3dMetric()
        ldmk_model = LandmarkDetector(fd_model_path="SpeakerVid-5M/weights/Resnet50_Final.pth",fl_model_path="SpeakerVid-5M/weights/h3r_model.pth")

    pred_input_folder = "/data/pipline/dialog/out_8_0_32_VA_eval_5s_dialog_pretrain_step_51000"  
    gt_input_folder = "/data/pipline/human_centric_vw_model/benchmark/testset_gt"  
    mid_save_dir = "/data/pipline/human_centric_vw_model/benchmark/mid_exp"
    pred_name = pred_input_folder.split("/")[-1]
    gt_name = gt_input_folder.split("/")[-1]
    pred_save_dir = os.path.join(mid_save_dir, pred_name)
    gt_save_dir = os.path.join(mid_save_dir, gt_name)
    os.makedirs(pred_save_dir, exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)


    output_image_folder = os.path.join(pred_save_dir,'face')  
    meta_dir = os.path.join(pred_save_dir,'meta')  
    if os.path.exists(meta_dir):
        shutil.rmtree(meta_dir)
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    if need_compute:
        process_videos(pred_input_folder, output_image_folder, yolo_model)
    pred_deep3d_list = []
    for subdir in tqdm(os.listdir(output_image_folder)):
        subdir_path = os.path.join(output_image_folder, subdir)
        if os.path.isdir(subdir_path):
            sub_meta_dir = os.path.join(meta_dir, subdir)
            os.makedirs(sub_meta_dir, exist_ok=True)
            try:
                if need_compute:
                    ldmk_model.detect_single(image_dir=subdir_path, meta_dir=sub_meta_dir)
                    deep3d_model.detect_single(image_dir=subdir_path, meta_dir=sub_meta_dir)
                deep3ds_dir = os.path.join(sub_meta_dir, "deep3ds")
                deep3d_data = []
                for deep3d_path in os.listdir(deep3ds_dir):
                    if not deep3d_path.endswith('.npy'):
                        continue
                    deep3d_data.append(np.load(os.path.join(deep3ds_dir, deep3d_path)))
                deep3d_data = np.array(deep3d_data)[:, :-3]
                pred_deep3d_list.append(deep3d_data)
            except:
                print(f"Error processing {subdir_path}, skipping...")
                continue



    output_image_folder = os.path.join(gt_save_dir,'face')  
    meta_dir = os.path.join(gt_save_dir,'meta') 
    if os.path.exists(meta_dir):
        shutil.rmtree(meta_dir)
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    if need_compute:
        process_videos(gt_input_folder, output_image_folder, yolo_model, N=3)
    gt_deep3d_list = []
    for subdir in tqdm(os.listdir(output_image_folder)):
        subdir_path = os.path.join(output_image_folder, subdir)
        if os.path.isdir(subdir_path):
            sub_meta_dir = os.path.join(meta_dir, subdir)
            os.makedirs(sub_meta_dir, exist_ok=True)
            if need_compute:
                ldmk_model.detect_single(image_dir=subdir_path, meta_dir=sub_meta_dir)
                deep3d_model.detect_single(image_dir=subdir_path, meta_dir=sub_meta_dir)
            deep3ds_dir = os.path.join(sub_meta_dir, "deep3ds")
            deep3d_data = []
            for deep3d_path in os.listdir(deep3ds_dir):
                if not deep3d_path.endswith('.npy'):
                    continue
                deep3d_data.append(np.load(os.path.join(deep3ds_dir, deep3d_path)))
            deep3d_data = np.array(deep3d_data)[:, :-3]
            gt_deep3d_list.append(deep3d_data)

    
    intra = compute_intra_diversity(pred_deep3d_list)
    inter = compute_inter_diversity(pred_deep3d_list)
    print(f"Pred Intra-Diversity: {intra:.4f}")
    print(f"Pred Inter-Diversity: {inter:.4f}")

    intra = compute_intra_diversity(gt_deep3d_list)
    inter = compute_inter_diversity(gt_deep3d_list)
    print(f"GT Intra-Diversity: {intra:.4f}")
    print(f"GT Inter-Diversity: {inter:.4f}")

    pred_deep3d_list = np.concatenate(pred_deep3d_list, axis=0)
    gt_deep3d_list = np.concatenate(gt_deep3d_list, axis=0)
    fid_score = compute_fid(gt_deep3d_list, pred_deep3d_list)
    print(f"FID: {fid_score:.4f}")

    

