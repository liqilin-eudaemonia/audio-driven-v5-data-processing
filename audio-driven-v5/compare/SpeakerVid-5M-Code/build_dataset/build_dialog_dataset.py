import os
import glob
import json
from tqdm import tqdm
import torchaudio
def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

import tarfile
def extract_tar(tar_path, extract_path='.'):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)


consider_dover = False
consider_l_score = False
consider_asr = True
consider_anno = True
read_from_txt = True
face_and_side_strict = False
motion_strict = False


video_list_txt = "/data/pipline/human_centric_vw_model/data_process/statistic/mp4_list0626.txt"
dataset_root = '/cpfs/dialog/captioned_video/anno'
json_save_dir = '/cpfs/exp/anno_dialog_7s_asr_visionhuman_womotion0626'
os.makedirs(json_save_dir, exist_ok=True)
ANNO_root = '/cpfs/dialog/anno'
dover_root = '/cpfs-model/dialog/dover'
dwpose_root = '/cpfs-model/dialog/dwpose'
l_score_root = '/cpfs-model/dialog/l_score'
asr_root = '/cpfs/dialog/new_asr'


human_save_root = '/cpfs/dialog/human_video'
os.makedirs(human_save_root,exist_ok=True)
build_temp_dir = '/cpfs/dialog/human_video_temp'
os.makedirs(build_temp_dir,exist_ok=True)

thres = 3
time_thres = 15
motion_thres = 1
l_score_thres = 0.3
dover_thres = 0.3
single_language = 'en'

max_workers = 100
if read_from_txt:
    with open(video_list_txt, 'r') as f:
        data_flo_list = f.readlines()
else:
    data_flo_list = []
    for org_video in tqdm(os.listdir(dataset_root), desc="Scanning directories"):
        org_video_path = os.path.join(dataset_root, org_video)
        for clip in os.listdir(org_video_path):
            if clip.endswith('.mp4'):
                clip_path = os.path.join(org_video_path, clip)
                data_flo_list.append(clip_path)


from concurrent.futures import ThreadPoolExecutor
import threading

dialog_dataset_sync = []
dialog_dataset_asyc = []
data_dict = {}
data_lock = threading.Lock()

def check_one_file(human_video_file, human_audio_file,human_json_file, batch_num, video_num):
    try:
        with open(human_json_file, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
    except:
        return None
    if human_data['conf'] < thres or human_data['video_total_duration'] > time_thres:
        return
    anno_file = os.path.join(ANNO_root, batch_num, video_num, os.path.basename(human_video_file).replace('.mp4', '.json'))
    # dover_file = os.path.join(dover_root, batch_num, video_num, os.path.basename(human_video_file).replace('.mp4', '.json'))
    dover_file = os.path.join(dover_root, batch_num, video_num, str(os.path.basename(human_video_file)).replace('.mp4', '') + '.json')
    l_score_file = os.path.join(l_score_root, os.path.basename(human_video_file).replace('.mp4', '.json'))
    video_name = str(os.path.basename(human_video_file)).replace('.mp4', '').rsplit('_', 1)[0]
    asr_file = os.path.join(asr_root, video_name + '.json')
    if not os.path.exists(anno_file) and consider_anno:
        return None
    if not os.path.exists(dover_file) and consider_dover:
        return None
    if not os.path.exists(l_score_file) and consider_l_score:
        return None
    if not os.path.exists(asr_file) and consider_asr:
        return None

    try:
        if consider_asr:
            with open(asr_file, 'r', encoding='utf-8') as f:
                asr_dict = json.load(f)
            text = asr_dict.get("text", "").strip()
            segments = asr_dict.get("segments", [])
            language = asr_dict.get("language", "unknown")
            if text == "":
                return None
            if single_language and language != single_language:
                return None
        if consider_anno:
            with open(anno_file, 'r', encoding='utf-8') as f:
                anno_dict = json.load(f)
            json_str = anno_dict['anno'].strip('```json\n').strip('```').strip()
            data_dict = json.loads(json_str)
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
            
    except:
        return None
    try:
        human_audio_16k = load_wav(human_audio_file, 16000)
    except:
        print('error audio')
        return
    return True

def process_file(file_flo):
    global dialog_dataset_sync, dialog_dataset_asyc
    
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
        return
    
    if data['conf'] < thres or data['video_total_duration'] > time_thres or data['clear'] < 10:
        return
    
    anno_file = os.path.join(ANNO_root, batch_num, video_num, os.path.basename(cur_video_file).replace('.mp4', '.json'))
    dover_file = os.path.join(dover_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
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
                return None
            if single_language and language != single_language:
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
            if dover_dict['dover_score'] < dover_thres:
                return None
        if consider_l_score:
            with open(l_score_file, 'r', encoding='utf-8') as f:
                l_score_dict = json.load(f)
            if l_score_dict['mean_clarity_left'] < l_score_thres or l_score_dict['mean_clarity_right'] < l_score_thres or l_score_dict['mean_clarity_face'] < l_score_thres:
                return None
            
    except:
        return

    try:
        agent_audio_16k = load_wav(cur_audio_file, 16000)
    except:
        print('error audio')
        return
    

    A_B = video_name.split('_')[-1]
    if A_B == 'None':
        return
    if A_B != 'A' and A_B != 'B':
        return
    oAB = 'A' if A_B == 'B' else 'B'
    cur_clip_num = video_name.split('full_video')[1].split('_')[1]
    if cur_clip_num == '000':
        return
    human_save_dir = os.path.join(human_save_root,batch_num,video_num)
    os.makedirs(human_save_dir,exist_ok=True)
    
    pre_clip_num = str(int(cur_clip_num)-1).zfill(3)
    cur_other_video_name = (video_name + '.tar').replace('_'+A_B+'.tar','_'+oAB+'.tar')
    pre_other_video_name = (video_name + '.tar').replace('_'+A_B+'.tar','_'+oAB+'.tar').replace('_'+cur_clip_num+'_','_'+pre_clip_num+'_')
    duet_exist = False
    cur_other_video_path = os.path.join(human_save_dir,cur_other_video_name.replace('.tar',''))
    pre_other_video_path = os.path.join(human_save_dir,pre_other_video_name.replace('.tar',''))
    if os.path.exists(cur_other_video_path):
        duet_exist = True
        is_sync = 1
        extract_path = cur_other_video_path
        try:
            video_list = glob.glob(f'{extract_path}/*.mp4', recursive=True)
            human_video_file = video_list[0]
            audio_list = glob.glob(f'{extract_path}/*.wav', recursive=True)
            human_audio_file = audio_list[0]
            json_list = glob.glob(f'{extract_path}/*.json', recursive=True)
            human_json_file = json_list[0]
        except:
            return
        if check_one_file(human_video_file,human_audio_file,human_json_file,batch_num,video_num):
            item = {
                'human_video': human_video_file,
                'human_audio': human_audio_file,
                'agent_video': cur_video_file,
                'agent_audio': cur_audio_file,
                'agent_ASR': agent_asr_file,
                'caption': agent_anno_file,
                'is_sync': is_sync
            }
            with data_lock:
                if is_sync == 1:
                    dialog_dataset_sync.append(item)
                else:
                    dialog_dataset_asyc.append(item)


    if os.path.exists(pre_other_video_path):
        duet_exist = True
        is_sync = 0
        extract_path = pre_other_video_path
        try:
            video_list = glob.glob(f'{extract_path}/*.mp4', recursive=True)
            human_video_file = video_list[0]
            audio_list = glob.glob(f'{extract_path}/*.wav', recursive=True)
            human_audio_file = audio_list[0]
            json_list = glob.glob(f'{extract_path}/*.json', recursive=True)
            human_json_file = json_list[0]
        except:
            return
        if check_one_file(human_video_file,human_audio_file,human_json_file,batch_num,video_num):
            item = {
                'human_video': human_video_file,
                'human_audio': human_audio_file,
                'agent_video': cur_video_file,
                'agent_audio': cur_audio_file,
                'agent_ASR': agent_asr_file,
                'caption': agent_anno_file,
                'is_sync': is_sync
            }
            with data_lock:
                if is_sync == 1:
                    dialog_dataset_sync.append(item)
                else:
                    dialog_dataset_asyc.append(item)

    if duet_exist == False:
        return


with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_file, data_flo_list), total=len(data_flo_list)))

data_dict['dialog_dataset_asyn'] = dialog_dataset_asyc
data_dict['dialog_dataset_sync'] = dialog_dataset_sync

json_file = os.path.join(json_save_dir,'dialog.json')
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=False)




