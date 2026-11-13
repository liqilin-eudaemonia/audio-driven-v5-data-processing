

import json
import os
import re
from tqdm import tqdm
import clip
from PIL import Image
import torch
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
def cal_similarity(texts):
    if len(texts) != 2:
        raise ValueError("texts should contain exactly two elements.")
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features[0] @ text_features[1]).item()
    return similarity


ans_json_path = "/data/pipline/human_centric_vw_model/benchmark/corh/MLLM_results.json"
with open(ans_json_path, 'r', encoding='utf-8') as f:
    ans_json = json.load(f)
res_json_path = "/data/pipline/human_centric_vw_model/benchmark/corh/generated_response.json"
with open(res_json_path, 'r', encoding='utf-8') as f:
    res_json = json.load(f)
asr_root = "/cpfs/dialog/new_asr"
sim_score_map = [0.2, 0.4, 0.6, 0.8, 1.0]  
sim_score_list = []
for key, text in tqdm(ans_json.items()):
    ans = json.loads(text)
    try:
        res = json.loads(res_json[key])
        cur_ans_list = [
            ans['very bad'],
            ans['bad'],
            ans['normal'],
            ans['good'],
        ]
        gt_ans_path = os.path.join(asr_root, key + ".json")
        if not os.path.exists(gt_ans_path):
            print(f"GT answer file {gt_ans_path} does not exist.")
            continue
        with open(gt_ans_path, 'r', encoding='utf-8') as f:
            gt_ans = json.load(f)['text']
        cur_ans_list.append(gt_ans)
        sim_list = []
        for i in range(5):
            cur_sim = cal_similarity([cur_ans_list[i], res['response']])
            sim_list.append(cur_sim)
        max_sim_idx = sim_list.index(max(sim_list))    
        cur_sim_score = sim_score_map[max_sim_idx]
        sim_score_list.append(cur_sim_score)
        print(sim_list, cur_sim_score)
    except:
        print('worong',key)
        continue
print(sim_score_list)
print(len(sim_score_list))
print(sum(sim_score_list) / len(sim_score_list))
