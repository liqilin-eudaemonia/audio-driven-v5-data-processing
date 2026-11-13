import os
import glob
import json
from tqdm import tqdm
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

lock = threading.Lock()

consider_dover = False
consider_l_score = False
consider_asr = True
consider_anno = True
ASR_strict = False
face_and_side_strict = False
motion_strict = False

def load_wav(wav, target_sr):
    """Load and resample an audio file (thread-safe)"""
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f'wav sample rate {sample_rate} must be greater than {target_sr}'
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def process_clip(clip_path, ANNO_root, dover_root, l_score_root, asr_root, thres, time_thres, motion_thres, dover_thres, l_score_thres):
    """Process the multithreaded task for a single clip directory"""
    clip_path = os.path.dirname(clip_path)
    parts = clip_path.split('/')
    batch_num = parts[-3]
    video_num = parts[-2]
    
    try:
        # Get files under the current clip
        video_list = glob.glob(f'{clip_path}/*.mp4', recursive=True)
        cur_video_file = video_list[0]
        audio_list = glob.glob(f'{clip_path}/*.wav', recursive=True)
        cur_audio_file = audio_list[0]
        json_list = glob.glob(f'{clip_path}/*.json', recursive=True)
        cur_json_file = json_list[0]
        
        # Read the JSON configuration
        with open(cur_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filtering conditions
        if data['conf'] < thres or data['video_total_duration'] > time_thres or data['clear'] < 10:
            return None
        
        # Check annotation file
        anno_file = os.path.join(ANNO_root, batch_num, video_num, os.path.basename(cur_video_file).replace('.mp4', '.json'))
        # dover_file = os.path.join(dover_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
        dover_file = os.path.join(dover_root, batch_num, video_num, str(os.path.basename(cur_video_file)).replace('.mp4', '') + '.json')
        
        l_score_file = os.path.join(l_score_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
        video_name = str(os.path.basename(cur_video_file)).replace('.mp4', '').rsplit('_', 1)[0]
        asr_file = os.path.join(asr_root, video_name + '.json')
        agent_asr_file = asr_file
        agent_anno_file = anno_file
        if not os.path.exists(anno_file) and consider_anno:
            return None
        if not os.path.exists(dover_file) and consider_dover:
            return None
        if not os.path.exists(l_score_file) and consider_l_score:
            return None
        if not os.path.exists(asr_file) and consider_asr:
            return None
        
        if consider_asr:
            with open(asr_file, 'r', encoding='utf-8') as f:
                asr_dict = json.load(f)
            text = asr_dict.get("text", "").strip()
            segments = asr_dict.get("segments", [])
            language = asr_dict.get("language", "unknown")
            has_low_conf = False
            has_bad_content = False
            has_silent_seg = False
            
            for seg in segments:
                if ASR_strict:
                    if seg.get("avg_logprob", -10) <= -0.5:  
                        has_low_conf = True
                    if seg.get("compression_ratio", 10) >= 2.0:  
                        has_bad_content = True
                    if seg.get("no_speech_prob", 1.0) >= 0.5:  
                        has_silent_seg = True
                else:
                    if seg.get("avg_logprob", -10) <= -1.0:
                        has_low_conf = True
                    if seg.get("compression_ratio", 10) >= 2.4:
                        has_bad_content = True
                    if seg.get("no_speech_prob", 1.0) >= 0.8:
                        has_silent_seg = True
            if has_low_conf or has_bad_content or has_silent_seg or text == "":
                return None
            if language != "en":
                return None
        
        if consider_anno:
            with open(anno_file, 'r', encoding='utf-8') as f:
                anno_dict = json.load(f)
            json_str = anno_dict['anno'].strip('```json\n').strip('```').strip()
            data_dict = json.loads(json_str)

            if face_and_side_strict:
                if data_dict['是否半身'] == "否" or data_dict['面朝方向'] != "正面":
                    return None
            if motion_strict:
                if int(data_dict['视频运动等级']) < motion_thres:
                    return None
            if data_dict['是否运动'] == "否":
                return None
            if data_dict['是否说话'] == "否":
                return None
            if data_dict['面朝方向'] == "背面":
                return None
            if int(data_dict['视频中出现的人数']) > 1:
                return None
        
        if consider_dover:
            with open(dover_file, 'r', encoding='utf-8') as f:
                dover_dict = json.load(f)
            if dover_dict['dover'] < dover_thres:
                return None
        if consider_l_score:
            with open(l_score_file, 'r', encoding='utf-8') as f:
                l_score_dict = json.load(f)
            if l_score_dict['mean_clarity_left'] < l_score_thres or l_score_dict['mean_clarity_right'] < l_score_thres or l_score_dict['mean_clarity_face'] < l_score_thres:
                return None
        
        # Load audio (thread-safe)
        try:
            agent_audio_16k = load_wav(cur_audio_file, 16000)
        except:
            return None
        
        return {
            'video': cur_video_file,
            'audio': cur_audio_file,
            'anno': anno_file
        }
    except Exception as e:
        return None

import argparse
def get_parse():
    parser = argparse.ArgumentParser(description="Video Luminance Checker")
    parser.add_argument("--anno_root", type=str)
    parser.add_argument("--dover_root", type=str)
    parser.add_argument("--l_score_root", type=str)
    parser.add_argument("--asr_root", type=str)
    parser.add_argument("--cur_num", type=int, default=4)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--dataset_root", type=str, default='/cpfs/dialog/captioned_video/anno')
    parser.add_argument("--video_list_path", type=str, default="/data/pipline/human_centric_vw_model/data_process/statistic/asr/mp4_list0626.txt")
    parser.add_argument("--save_dir", type=str, default="./single_dataset")
    return parser.parse_args()

def main():
    opt = get_parse()
    read_from_txt = True
    video_list_path = opt.video_list_path
    dataset_root = opt.dataset_root
    json_save_dir = opt.save_dir
    os.makedirs(json_save_dir, exist_ok=True)
    
    ANNO_root = opt.anno_root
    dover_root = opt.dover_root
    l_score_root = opt.l_score_root
    asr_root = opt.asr_root
    
    thres = 3
    time_thres = 15
    motion_thres = 1
    dover_thres = 0.3
    l_score_thres = 0.3

    max_workers = 100
    if read_from_txt:
        with open(video_list_path, 'r') as f:
            data_flo_list = f.readlines()
    else:
        data_flo_list = []
        for batch in tqdm(os.listdir(dataset_root), desc="Scanning directories"):
            batch_path = os.path.join(dataset_root, batch)
            for video in os.listdir(batch_path):
                vides_path = os.path.join(batch_path, video)
                for clip in os.listdir(vides_path):
                    clip_path = os.path.join(vides_path, clip)
                    data_flo_list.append(clip_path)

    dialog_dataset_single = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust thread number based on I/O performance [5,9](@ref)
        futures = []
        for clip_path in data_flo_list:
            futures.append(
                executor.submit(
                    process_clip, 
                    clip_path, ANNO_root, dover_root, l_score_root, asr_root , thres, time_thres, motion_thres, dover_thres, l_score_thres
                )
            )
        
        # Progress bar monitoring
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing clips"):
            result = future.result()
            if result:
                with lock: 
                    dialog_dataset_single.append(result)
    
    # Save results
    data_dict = {'dialog_dataset_single': dialog_dataset_single}
    json_file = os.path.join(json_save_dir, 'single.json')
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    print(f"Processing complete, valid data count: {len(dialog_dataset_single)}")

if __name__ == "__main__":
    main()
