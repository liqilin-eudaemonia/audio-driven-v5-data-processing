import os
import glob
import json
from tqdm import tqdm
import torchaudio
def find_overlapping_intervals(A,speaker, start, end, consider_len=False, min_len=0.3):
    result = []
    result_speaker = []
    for i in range(len(A)):
        a, b = A[i]
        s = speaker[i]
        if a <= end and b >= start:
            overlap_start = max(a, start)
            overlap_end = min(b, end)
            if (overlap_end - overlap_start >= min_len) and consider_len:
                result.append([overlap_start, overlap_end])
                result_speaker.append(s)
            elif not consider_len:
                result.append([overlap_start, overlap_end])
                result_speaker.append(s)
    return result,result_speaker

def match_video_audio(video_data, audio_ids, threshold=0.6):
    result = {}
    result_score = {}
    matched_audios = set()
    
    candidates = []
    for vid, scores in video_data.items():
        for i, score in enumerate(scores):
            # if score >= threshold:
            candidates.append((score, vid, audio_ids[i]))
    
    candidates.sort(reverse=True, key=lambda x: x[0])
    for score, vid, aid in candidates:
        if vid not in result and aid not in matched_audios:
            result[vid] = aid
            result_score[vid] = score
            matched_audios.add(aid)
    
    
    for vid in video_data:
        if vid not in result:
            result[vid] = None
            result_score[vid] = 0
    
    cur_auid = audio_ids[0]
    if cur_auid == 'A':
        next_auid = 'B'
    elif cur_auid == 'B':
        next_auid = 'A'
    else:
        next_auid = None
    if len(video_data.keys()) == 2 and len(audio_ids) == 1 and next_auid is not None:
        for vid in video_data:
            if result[vid] == None:
                result[vid] = next_auid + '_None'
            
    return result, result_score
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



video_list_txt = "/data/pipline/human_centric_vw_model/data_process/statistic/asr/mp4_list0626.txt"
dataset_root = '/cpfs/dialog/captioned_video/anno'
json_save_dir = '/cpfs/exp/anno_listen_15s_asr_visionhuman_womotion0626'
os.makedirs(json_save_dir, exist_ok=True)
ANNO_root = '/cpfs/dialog/anno'
dover_root = '/cpfs-model/dialog/dover'
dwpose_root = '/cpfs-model/dialog/dwpose'
l_score_root = '/cpfs-model/dialog/l_score'
asr_root = '/cpfs/dialog/new_asr'
scene_root = '/cpfs/new_dialog/scene_json'
speaker_root = '/cpfs/new_dialog/speaker_json'
human_save_root = '/cpfs/dialog/captioned_video/anno'
os.makedirs(human_save_root,exist_ok=True)

thres = 3
time_thres = 20
motion_thres = 1
l_score_thres = 0.3
dover_thres = 0.3
listen_thres = 3
consider_dover = False
consider_l_score = False
consider_asr = True
consider_anno = True
read_from_txt = True
face_and_side_strict = False
motion_strict = False

max_workers = 100
print('reading from txt')
if read_from_txt:
    with open(video_list_txt, 'r') as f:
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


from concurrent.futures import ThreadPoolExecutor
import threading

listen_dataset_sync = []
listen_dataset_asyc = []
data_dict = {}
data_lock = threading.Lock()

def process_file(file_flo):
    global listen_dataset_sync, listen_dataset_asyc
    
    file_flo = os.path.dirname(file_flo)
    parts = file_flo.split('/')
    batch_num = parts[-3]
    video_num = parts[-2]
    video_name = parts[-1]
    anno_file = ''

    A_B = video_name.split('_')[-1]
    if A_B == 'None':
        return
    if A_B != 'A' and A_B != 'B':
        return
    oAB = 'A' if A_B == 'B' else 'B'
    human_save_dir = os.path.join(human_save_root,batch_num,video_num)
    cur_other_video_name = (video_name + '.tar').replace('_'+A_B+'.tar','_'+oAB+'.tar')
    duet_exist = False
    cur_other_video_path = os.path.join(human_save_dir,cur_other_video_name.replace('.tar',''))
    if os.path.exists(cur_other_video_path):
        duet_exist = True
        is_sync = 1
        extract_path = cur_other_video_path
    if duet_exist == False:
        return

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
    # dover_file = os.path.join(dover_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
    dover_file = os.path.join(dover_root, batch_num, video_num, str(os.path.basename(cur_video_file)).replace('.mp4', '') + '.json')
    l_score_file = os.path.join(l_score_root, os.path.basename(cur_video_file).replace('.mp4', '.json'))
    video_name = str(os.path.basename(cur_video_file)).replace('.mp4', '').rsplit('_', 1)[0]
    video_id = video_name.split('_full_video_')[0] + "_full_video"
    clip_id = video_name.split('_full_video_')[1].split('_')[0]
    asr_file = os.path.join(asr_root, video_name + '.json')
    scene_file = os.path.join(scene_root,video_id + '.json')
    speaker_file = os.path.join(speaker_root,video_id + '.json')
    if not os.path.exists(scene_file) or not os.path.exists(speaker_file): 
        return
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
            if dover_dict['dover_score'] < 0.3:
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
    # pre_other_video_path = os.path.join(human_save_dir,pre_other_video_name.replace('.tar',''))
    if os.path.exists(cur_other_video_path):
        duet_exist = True
        is_sync = 1
        extract_path = cur_other_video_path
    
    if duet_exist == False:
        return
    
    try:
        video_list = glob.glob(f'{extract_path}/*.mp4', recursive=True)
        human_video_file = video_list[0]
        audio_list = glob.glob(f'{extract_path}/*.wav', recursive=True)
        human_audio_file = audio_list[0]
        json_list = glob.glob(f'{extract_path}/*.json', recursive=True)
        human_json_file = json_list[0]
        with open(human_json_file, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
    except:
        return
    
    anno_file = os.path.join(ANNO_root, batch_num, video_num, os.path.basename(human_video_file).replace('.mp4', '.json'))
    # dover_file = os.path.join(dover_root, os.path.basename(human_video_file).replace('.mp4', '.json'))s
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
        sync_score_map = human_data['sync']
        speaker_data = human_data['speaker']
        with open(speaker_file, 'r', encoding='utf-8') as f:
            audio_ver_data = json.load(f)
        with open(scene_file, 'r', encoding='utf-8') as f:
            video_scene_data = json.load(f)
        scene_clip_id = "scene_" + clip_id
        cur_video_start = video_scene_data[scene_clip_id]['start_seconds']
        cur_video_end = video_scene_data[scene_clip_id]['end_seconds']
        A = audio_ver_data['A']
        speaker = audio_ver_data['spaeker']
        if sync_score_map.get('0',None) == None or sync_score_map.get('1',None) == None:
            return
        person0_sync_score = sync_score_map['0']
        person1_sync_score = sync_score_map['1']

        overlapping_intervals, result_speaker = find_overlapping_intervals(A, speaker, cur_video_start, cur_video_end)
        if len(result_speaker) == 0:
            return
        if len(overlapping_intervals) != len(person0_sync_score) or len(overlapping_intervals) != len(speaker_data):
            overlapping_intervals, result_speaker = find_overlapping_intervals(A, speaker, cur_video_start, cur_video_end, consider_len=True)
        if len(overlapping_intervals) != len(person0_sync_score) or len(overlapping_intervals) != len(speaker_data):
            print('ver not same!!!!!!!!!')
            return
        
        if len(person0_sync_score) != len(person1_sync_score):
            print('sync not same!!!!!!!!!')
            return
        talking_video_span = []
        listening_video_span = []
        human_video_name = os.path.basename(human_video_file).replace('.mp4', '')
        agent_video_name = os.path.basename(cur_video_file).replace('.mp4', '')
        if '_00' in human_video_name and '_01' in agent_video_name:
            person0_video_name = human_video_name
            person1_video_name = agent_video_name
            person0_video_path = os.path.join(extract_path, person0_video_name + '.mp4')
            person1_video_path = os.path.join(file_flo, person1_video_name + '.mp4')
        elif '_01' in human_video_name and '_00' in agent_video_name:
            person0_video_name = agent_video_name
            person1_video_name = human_video_name
            person0_video_path = os.path.join(file_flo, person0_video_name + '.mp4')
            person1_video_path = os.path.join(extract_path, person1_video_name + '.mp4')
        else:
            return
        for i in range(len(overlapping_intervals)):
            cur_audio_start = overlapping_intervals[i][0]
            cur_audio_end = overlapping_intervals[i][1]
            if (person0_sync_score[i][0] - person1_sync_score[i][0]) > listen_thres:
                talking_video_span.append(
                    (person0_video_path, cur_audio_start - cur_video_start, cur_audio_end - cur_video_start)
                )
                listening_video_span.append(
                    (person1_video_path, cur_audio_start - cur_video_start, cur_audio_end - cur_video_start)
                )
            if (person1_sync_score[i][0] - person0_sync_score[i][0]) > listen_thres:
                listening_video_span.append(
                    (person0_video_path, cur_audio_start - cur_video_start, cur_audio_end - cur_video_start)
                )
                talking_video_span.append(
                    (person1_video_path, cur_audio_start - cur_video_start, cur_audio_end - cur_video_start)
                )
    except:
        return
    def merge_consecutive_spans(spans):
        merged_spans = []
        for i in range(len(spans)):
            if i > 0 and spans[i][0] == spans[i-1][0]:
                merged_spans[-1] = (spans[i][0], merged_spans[-1][1], spans[i][2])
            else:
                merged_spans.append(spans[i])
        return merged_spans
    listening_video_span = merge_consecutive_spans(listening_video_span)
    talking_video_span = merge_consecutive_spans(talking_video_span)
    
    item = {
        'listener': listening_video_span,
        'speaker': talking_video_span,
        'is_sync': is_sync
    }
    
    with data_lock:
        if is_sync == 1:
            listen_dataset_sync.append(item)
        else:
            listen_dataset_asyc.append(item)

print('flitering')

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_file, data_flo_list), total=len(data_flo_list)))
data_dict['listen_dataset_asyn'] = listen_dataset_asyc
data_dict['listen_dataset_sync'] = listen_dataset_sync

json_file = os.path.join(json_save_dir,'listen.json')
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=False)




