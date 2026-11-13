import json
import os


# "/cpfs/wangduomin2/workspace/data/portrait/hcvm/process/labels/cn_label.json",
# "/cpfs/wangduomin2/workspace/data/portrait/hcvm/process/labels/hdtf_label.json",
# "/cpfs/wangduomin2/workspace/data/portrait/hcvm/process/labels/vox2_high_label.json",
# "/cpfs/wangduomin2/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250407_v0.json",
# "/cpfs/wangduomin2/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250407_v1.json",

# source_path = "/data/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250417_v1_bk.json"
# target_path = "/data/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250417_v1.json"

# with open(source_path, 'r') as f:
#     data_dict = json.load(f)

# for key in data_dict:
#     data_dict[key]['data_url'] = data_dict[key]['data_url'].replace("tar_data_20250414", "tar_data_20250417").replace("batch1", "batch2")

# with open(target_path, 'w') as f:
#     json.dump(data_dict, f, indent=4, ensure_ascii=False)



# source_path = "/cpfs/_wangduomin2/workspace/data/portrait/hcvm/process/labels/cn_label.json"
# target_path = "/data/workspace/data/portrait/hcvm/process/labels/cn_label.json"
# source_path = "/cpfs/_wangduomin2/workspace/data/portrait/hcvm/process/labels/hdtf_label.json"
# target_path = "/data/workspace/data/portrait/hcvm/process/labels/hdtf_label.json"
# source_path = "/cpfs/_wangduomin2/workspace/data/portrait/hcvm/process/labels/vox2_high_label.json"
# target_path = "/data/workspace/data/portrait/hcvm/process/labels/vox2_high_label.json"
# source_path = "/cpfs/_wangduomin2/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250407_v0.json"
# target_path = "/data/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250407_v0.json"
source_path = "/cpfs/wangduomin2/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250407_v1.json"
target_path = "/cpfs/wangduomin/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250407_v1.json"

with open(source_path, 'r') as f:
    data_dict = json.load(f)

for key in data_dict:
    data_dict[key]['data_url'] = data_dict[key]['data_url'].replace("qy-hcvm-data", "hcvm-data")

with open(target_path, 'w') as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)


# source_path = "/cpfs/_wangduomin/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250414_v1.json"
# target_path = "/cpfs/_wangduomin2/workspace/data/portrait/hcvm/process/labels/hcvm_youtube_20250414_v1.json"

# with open(source_path, 'r') as f:
#     data_dict = json.load(f)

# for key in data_dict:
#     data_dict[key]['data_url'] = data_dict[key]['data_url'].replace("hcvm-data", "qy-hcvm-data")

# with open(target_path, 'w') as f:
#     json.dump(data_dict, f, indent=4, ensure_ascii=False)