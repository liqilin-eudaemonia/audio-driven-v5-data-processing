import glob
import json
import os
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech
from tqdm import tqdm
import tarfile
from collections import defaultdict
def extract_tar(tar_path, extract_path='.'):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)
def check_one_video(ANNO_root,dover_root,l_score_root,asr_root,file_flo):
    file_flo = os.path.dirname(file_flo)
    parts = file_flo.split('/')
    batch_num = parts[-3]
    video_num = parts[-2]
    video_name = parts[-1]
    anno_file = ''

    try:
        video_list = glob.glob(f'{file_flo}/*.mp4', recursive=True)
        cur_video_file = video_list[0]
        audio_list = glob.glob(f'{file_flo}/*.wav', recursive=True)
        cur_audio_file = audio_list[0]
        json_list = glob.glob(f'{file_flo}/*.json', recursive=True)
        cur_json_file = json_list[0]
        with open(cur_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return None, 0
    start = data['start']
    # if consider_vision:
    #     if data['conf'] < thres or data['video_total_duration'] > time_thres or data['clear'] < 10:
    #         return None, start
        
    video_name = str(os.path.basename(cur_video_file)).replace('.mp4', '').rsplit('_', 1)[0]
    asr_file = os.path.join(asr_root, video_name + '.json')
    if not os.path.exists(asr_file) and consider_asr:
        return None,start
    
    try:
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
                if seg.get("avg_logprob", -10) <= -1.0:
                    has_low_conf = True
                if seg.get("compression_ratio", 10) >= 2.4:
                    has_bad_content = True
                if seg.get("no_speech_prob", 1.0) >= 0.7:
                    has_silent_seg = True
            if has_low_conf or has_bad_content or has_silent_seg or text == "":
                return None,start
            if language != "en":
                return None,start
    except:
        return None,start
    
    return True, start

def process_clip_dir(clip_dir, file_type='mp4'):
    try:
        mp4_files = glob.glob(f'{clip_dir}/*.{file_type}', recursive=True)
        return mp4_files[0] if mp4_files else None
    except Exception as e:
        print(f"Error processing {clip_dir}: {e}")
        return None
def construct_video_dict(dataset_dir, file_type='mp4',save_dir = None):
    max_workers = 256
    # video_id -> clip_num -> list of file paths
    video_dict = defaultdict(lambda: defaultdict(list))
    for batch_path in os.listdir(dataset_dir):
        batch_path_full = os.path.join(dataset_dir, batch_path)
        for video_path in os.listdir(batch_path_full):
            video_path_full = os.path.join(batch_path_full, video_path)
            clip_paths = os.listdir(video_path_full)

            clip_dirs = [os.path.join(video_path_full, clip_path) for clip_path in clip_paths]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_clip_dir, clip_dir): clip_dir for clip_dir in clip_dirs}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        filename = os.path.basename(result)
                        if "full_video" not in filename:
                            print(f"[WARN] Skipping malformed filename: {filename}")
                            continue
                        try:
                            # Split by 'full_video'
                            video_id, rest = filename.split("_full_video_")
                            parts = rest.split("_")
                            if len(parts) >= 2:
                                clip_num = parts[0]
                                video_dict[video_id][clip_num].append(result)
                            else:
                                print(f"[WARN] Skipping unexpected format: {filename}")
                        except Exception as e:
                            print(f"[ERROR] Failed to parse filename {filename}: {e}")
    if save_dir:
        # output_path = f"{save_dir}/{file_type}_video_dict.json"
        with open(save_dir, "w") as f:
            json.dump(video_dict, f, indent=2)
    return video_dict



# input_path should be a valid dialog dataset json obtained from build_dialog_dataset.py
input_path = '/data/pipline/human_centric_vw_model/data_process/statistic/dialog_anno_dialog_7s_asr_clear_womotion0626.json'
video_dict_path = f"/data/pipline/human_centric_vw_model/data_process/statistic/multi_turn/mp4_video_dict.json"
output_path = f"/data/pipline/human_centric_vw_model/data_process/statistic/multi_turn/multi_turn.json"
dataset_dir = "/cpfs/dialog/captioned_video/anno"
ANNO_root = '/cpfs/dialog/anno'
dover_root = '/cpfs-model/dialog/dover'
dwpose_root = '/cpfs-model/dialog/dwpose'
l_score_root = '/cpfs-model/dialog/l_score'
asr_root = '/cpfs/dialog/new_asr'
# === 加载 JSON ===
with open(input_path, 'r') as f:
    data_items = json.load(f)['dialog_dataset_asyn']

video_dict = construct_video_dict(dataset_dir,save_dir=video_dict_path)
# with open(video_dict_path, "r") as f:
#     video_dict = json.load(f)

use_duration = True
if use_duration:
    duration_len = 5
else:
    duration_len = 1

time_thres = 15
thres = 2
consider_dover = False
consider_l_score = False
consider_asr = True
consider_anno = True
consider_vision = False

motion_thres = 1
need_half = False
need_face = False
l_score_thres = 0.5


max_search_num = 20
max_search_time = 300
multi_turn_list = []
# === 统计 ===
stats = {}
video_id_items = {}

multi_turn_list = []
multi_turn_lock = threading.Lock()

def process_item(item):
    try:
        file_flo = item['human_video']
        file_flo = os.path.dirname(file_flo)
        parts = file_flo.split('/')
        video_name = parts[-1]
        video_id, rest = video_name.split("_full_video_")
        clip_num = rest.split("_")[0]

        cur_clip_num = int(clip_num)
        valid_pre_list = []
        json_list = glob.glob(f'{file_flo}/*.json', recursive=True)
        cur_json_file = json_list[0]
        with open(cur_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cur_time = float(data['start'])
        start_time = cur_time
        
        for _ in range(max_search_num):
            if cur_time + max_search_time < start_time :
                break
            if cur_clip_num - 1 < 0:
                break
            pre_clip_num = str(cur_clip_num - 1).zfill(3)
            pre_file_list = video_dict.get(video_id, {}).get(pre_clip_num, None)
            if pre_file_list:
                min_start = 0
                for pre_file in pre_file_list:
                    res, start = check_one_video(
                        ANNO_root=ANNO_root,
                        dover_root=dover_root,
                        l_score_root=l_score_root,
                        asr_root=asr_root,
                        file_flo=pre_file
                    )
                    if res and start + max_search_time > start_time:
                        valid_pre_list.append(pre_file)
                        min_start = min(min_start, start)
                cur_time = min_start
            cur_clip_num = int(pre_clip_num)

        if len(valid_pre_list) > 0:
            item['history_list'] = valid_pre_list
            return item
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Failed processing item: {e}")
        return None

# 多线程执行
with ThreadPoolExecutor(max_workers=128) as executor:
    futures = [executor.submit(process_item, item) for item in data_items]
    pbar = tqdm(as_completed(futures), total=len(futures), desc="Valid items: 0")

    for future in pbar:
        result = future.result()
        if result:
            with multi_turn_lock:
                multi_turn_list.append(result)
                pbar.set_description(f"Valid items: {len(multi_turn_list)}")

with open(output_path, "w") as f:
    json.dump(multi_turn_list, f, indent=2)
    
