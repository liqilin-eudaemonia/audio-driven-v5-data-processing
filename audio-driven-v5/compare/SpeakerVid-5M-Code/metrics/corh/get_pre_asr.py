import os
import re
import json
from tqdm import tqdm
json_dir = '/cpfs/dialog/new_asr'  
wav_dir = '/data/pipline/human_centric_vw_model/benchmark/testset'  
matched_json_path = 'matched_clips.json'
output_json_path = 'matched_clips_with_content.json'  


pattern = re.compile(r'^(.*_full_video)_(\d{3})_([AB])$')
target_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
target_files = [os.path.splitext(f)[0] for f in target_files]

result = {}

for target in tqdm(target_files):
    match = pattern.match(target)
    if not match:
        continue
    video_id, clip_str, _ = match.groups()
    clip_num = int(clip_str)

    matched_paths = []
    for i in range(clip_num):  
        for pid in ['A', 'B']:
            candidate_name = f"{video_id}_{i:03d}_{pid}.json"
            candidate_path = os.path.join(json_dir, candidate_name)
            if os.path.exists(candidate_path):
                matched_paths.append(candidate_path)

    self_json_file = os.path.join(json_dir, f"{target}.json")
    if os.path.exists(self_json_file):
        matched_paths.append(self_json_file)
    result[target] = matched_paths

with open(matched_json_path, 'w') as f:
    json.dump(result, f, indent=2)

with open(matched_json_path, 'r') as f:
    matched_data = json.load(f)

result = {}
for key, file_paths in tqdm(matched_data.items(), desc="Loading file contents"):
    content_list = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                person_id = file_path.split('/')[-1].split('.')[0][-1] 
                with open(file_path, 'r') as f:
                    content = json.load(f)['text']
                    content_list.append(person_id + ': ' + content)
            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")
        else:
            print(f"[Warning] File not found: {file_path}")
    result[key] = content_list

with open(output_json_path, 'w') as f:
    json.dump(result, f, indent=2)