import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

cv2.setNumThreads(0)

def extract_hand_regions_from_keypoints(frame, hand_kps, image_size=128, padding=0):
    h, w, _ = frame.shape
    hands_imgs = []

    for i in range(2):
        kps = hand_kps[i]
        valid_kps = kps[kps[:, 0] >= 0]
        if valid_kps.shape[0] == 0:
            hands_imgs.append(None)
            continue
        pixel_kps = (valid_kps * np.array([w, h])).astype(int)
        xmin = np.clip(np.min(pixel_kps[:, 0]) - padding, 0, w)
        xmax = np.clip(np.max(pixel_kps[:, 0]) + padding, 0, w)
        ymin = np.clip(np.min(pixel_kps[:, 1]) - padding, 0, h)
        ymax = np.clip(np.max(pixel_kps[:, 1]) + padding, 0, h)
        crop = frame[ymin:ymax, xmin:xmax]
        resized = cv2.resize(crop, (image_size, image_size)) if crop.size != 0 else None
        hands_imgs.append(resized)

    return hands_imgs[0], hands_imgs[1]

def extract_face_region_from_keypoints(frame, face_kps, image_size=128, padding=0):
    h, w, _ = frame.shape
    kps = face_kps[0]
    valid_kps = kps[kps[:, 0] >= 0]
    if valid_kps.shape[0] == 0:
        return None
    pixel_kps = (valid_kps * np.array([w, h])).astype(int)
    xmin = np.clip(np.min(pixel_kps[:, 0]) - padding, 0, w)
    xmax = np.clip(np.max(pixel_kps[:, 0]) + padding, 0, w)
    ymin = np.clip(np.min(pixel_kps[:, 1]) - padding, 0, h)
    ymax = np.clip(np.max(pixel_kps[:, 1]) + padding, 0, h)
    crop = frame[ymin:ymax, xmin:xmax]
    resized = cv2.resize(crop, (image_size, image_size)) if crop.size != 0 else None
    return resized

def laplacian_variance(img, ksize=3):
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return lap.std()

def read_video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    cap.release()
    return frame_list

def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def process_single_video(args):
    video_path, anno_root, json_root, l_score_root = args
    video_name = os.path.basename(video_path)
    anno_path = os.path.join(anno_root, video_name.replace('.mp4', '.npy'))
    json_path = os.path.join(json_root, video_name.replace(".mp4", ".json"))
    l_score_path = os.path.join(l_score_root, video_name.replace(".mp4", ".json"))
    if os.path.exists(l_score_path) and os.path.exists(json_path):
        return f"[SKIP] {video_name}: l_score already exists"

    try:
        frame_list = read_video_to_frames(video_path)
        anno_list = np.load(anno_path, allow_pickle=True)
    except Exception as e:
        return f"[ERROR] {video_name}: {e}"

    try:
        with open(json_path, "w") as f:
            json.dump(anno_list[None][0], f, default=convert_np, indent=2)

        left_laps, right_laps, face_laps = [], [], []
        for frame, anno in zip(frame_list, anno_list):
            hand_kps = anno['hands']
            face_kps = anno['faces']
            left, right = extract_hand_regions_from_keypoints(frame, hand_kps)
            face = extract_face_region_from_keypoints(frame, face_kps)
            left_laps.append(laplacian_variance(left))
            right_laps.append(laplacian_variance(right))
            face_laps.append(laplacian_variance(face))

        cur_data = {}
        L_left, L_right, L_face = np.array(left_laps), np.array(right_laps), np.array(face_laps)
        L_left_pivot = np.percentile(L_left[L_left > 0], 90) if np.any(L_left > 0) else 1e-8
        L_right_pivot = np.percentile(L_right[L_right > 0], 90) if np.any(L_right > 0) else 1e-8
        L_face_pivot = np.percentile(L_face[L_face > 0], 90) if np.any(L_face > 0) else 1e-8

        cur_data["L_left"] = L_left
        cur_data["L_right"] = L_right
        cur_data["L_face"] = L_face
        cur_data["new_L_hat_left"] = L_left / (L_left_pivot + 1e-8)
        cur_data["new_L_hat_right"] = L_right / (L_right_pivot + 1e-8)
        cur_data["new_L_hat_face"] = L_face / (L_face_pivot + 1e-8)
        cur_data["mean_clarity_left"] = np.mean(cur_data["new_L_hat_left"][L_left > 0]) if np.any(L_left > 0) else -1
        cur_data["mean_clarity_right"] = np.mean(cur_data["new_L_hat_right"][L_right > 0]) if np.any(L_right > 0) else -1
        cur_data["mean_clarity_face"] = np.mean(cur_data["new_L_hat_face"][L_face > 0]) if np.any(L_face > 0) else -1

        with open(l_score_path, "w") as f:
            json.dump(cur_data, f, default=convert_np, indent=2)

        return f"[OK] {video_name}"
    except Exception as e:
        return f"[ERROR] {video_name}: {e}"
import argparse
def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--num_cur", type=int, default=24)
    parser.add_argument("--num_Total", type=int, default=96)
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/bianli/mp4_0707.txt")
    parser.add_argument("--dwpose_dir", type=str, default="./dwpose_result")
    parser.add_argument("--dwjson_dir", type=str, default="./dwpose_json_result")
    parser.add_argument("--l_score_dir", type=str, default="./l_score_result")

    return parser.parse_args()

def main():
    opt = get_parse()
    video_list_path = opt.video_list_path
    anno_root = opt.dwpose_dir
    json_root = opt.dwjson_dir
    l_score_root =  opt.l_score_dir
    os.makedirs(json_root, exist_ok=True)
    os.makedirs(l_score_root, exist_ok=True)

    with open(video_list_path, "r") as f:
        video_list = f.read().splitlines()
    args_list = [(v, anno_root, json_root, l_score_root) for v in video_list]
    results = []
    max_workers = 100 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_video, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            results.append(result)
    with open("process_log.txt", "w") as log_f:
        log_f.write("\n".join(results))

if __name__ == "__main__":
    main()