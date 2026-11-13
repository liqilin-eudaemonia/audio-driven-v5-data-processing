from huggingface_hub import snapshot_download,HfApi
from pathlib import Path
import json
repo='Qwen/Qwen2-Audio-7B-Instruct'
revision=HfApi().model_info(repo).sha
path=snapshot_download(repo_id=repo,revision=revision,local_dir='/private/ckpts/Qwen2-Audio-7B-Instruct',max_workers=4)
Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_labeling_full_20261023/qwen_audio_download.json').write_text(json.dumps(dict(repo=repo,revision=revision,path=path),indent=2)+'\n')
print(path,flush=True)
