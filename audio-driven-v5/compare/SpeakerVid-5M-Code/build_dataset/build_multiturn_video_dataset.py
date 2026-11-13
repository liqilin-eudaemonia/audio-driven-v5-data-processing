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
        return None, 15
    video_total_duration = data['video_total_duration']
    if consider_vision:
        if data['conf'] < thres or data['video_total_duration'] > time_thres or data['clear'] < 10:
            return None, video_total_duration
    
    anno_file = os.path.join(ANNO_root, batch_num, video_num, os.path.basename(cur_video_file).replace('.mp4', '.json'))
    # dover_file = os.path.join(dover_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
    dover_file = os.path.join(dover_root, batch_num, video_num, str(os.path.basename(cur_video_file)).replace('.mp4', '') + '.json')

    l_score_file = os.path.join(l_score_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
    video_name = str(os.path.basename(cur_video_file)).replace('.mp4', '').rsplit('_', 1)[0]
    asr_file = os.path.join(asr_root, video_name + '.json')
    agent_asr_file = asr_file
    agent_anno_file = anno_file
    if not os.path.exists(anno_file) and consider_anno:
        # print(f"anno_file not exists: {anno_file}")
        return None,video_total_duration
    if not os.path.exists(dover_file) and consider_dover:
        # print(f"dover_file not exists: {dover_file}")
        return None,video_total_duration
    if not os.path.exists(l_score_file) and consider_l_score:
        # print(f"l_score_file not exists: {l_score_file}")
        return None,video_total_duration
    if not os.path.exists(asr_file) and consider_asr:
        # print(f"asr_file not exists: {asr_file}")
        return None,video_total_duration
    
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
                return None,video_total_duration
            if language != "en":
                return None,video_total_duration

        if consider_anno:
            with open(anno_file, 'r', encoding='utf-8') as f:
                anno_dict = json.load(f)
            json_str = anno_dict['anno'].strip('```json\n').strip('```').strip()
            data_dict = json.loads(json_str)
            
            if int(data_dict['视频运动等级']) < motion_thres or \
            (need_half and data_dict['是否半身'] == "否") or \
            (need_face and data_dict['面朝方向'] != "正面"):
                return None,video_total_duration
            if data_dict['是否运动'] == "否":
                return None,video_total_duration
            if data_dict['是否说话'] == "否":
                return None,video_total_duration
            if data_dict['面朝方向'] == "背面":
                return None,video_total_duration
            if int(data_dict['视频中出现的人数']) > 1:
                return None,video_total_duration
        
        if consider_dover:
            with open(dover_file, 'r', encoding='utf-8') as f:
                dover_dict = json.load(f)
            if dover_dict['dover'] < dover_thres:
                # print(f"dover_file dover_score < 0.3: {dover_file}")
                return None,video_total_duration
        if consider_l_score:
            with open(l_score_file, 'r', encoding='utf-8') as f:
                l_score_dict = json.load(f)
            if l_score_dict['mean_clarity_left'] < l_score_thres or l_score_dict['mean_clarity_right'] < l_score_thres or l_score_dict['mean_clarity_face'] < l_score_thres:
                # print(f"l_score_file mean_clarity < 0.6: {l_score_file}")
                return None,video_total_duration
            
    except:
        return None,video_total_duration

    try:
        agent_audio_16k = load_wav(cur_audio_file, 16000)
    except:
        print('error audio')
        return None,video_total_duration
    
    return True, video_total_duration

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
dover_thres = 0.4


max_search_num = 20
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

        need_next = True
        cur_clip_num = int(clip_num)
        valid_pre_list = []
        gap_num = 0

        for _ in range(max_search_num):
            if not need_next:
                break
            if cur_clip_num - 1 < 0:
                break
            pre_clip_num = str(cur_clip_num - 1).zfill(3)
            pre_file_list = video_dict.get(video_id, {}).get(pre_clip_num, None)
            has_valid = False
            if pre_file_list is None:
                gap_num += duration_len
            else:
                max_video_total_duration = 0
                for pre_file in pre_file_list:
                    res, video_total_duration = check_one_video(
                        ANNO_root=ANNO_root,
                        dover_root=dover_root,
                        l_score_root=l_score_root,
                        asr_root=asr_root,
                        file_flo=pre_file
                    )
                    max_video_total_duration = max(max_video_total_duration, video_total_duration)
                    if res:
                        valid_pre_list.append(pre_file)
                        has_valid = True
                if not has_valid:
                    max_video_total_duration = 1 if not use_duration else max_video_total_duration
                    gap_num += max_video_total_duration
            if gap_num > duration_len:
                need_next = False
            if has_valid:
                gap_num = 0
            cur_clip_num = int(pre_clip_num)

        if len(valid_pre_list) > 0:
            item['history_list'] = valid_pre_list
            return item
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Failed processing item: {e}")
        return None

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
    
